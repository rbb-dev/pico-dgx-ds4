// The runtime abliteration (ablit.h).
#include "model/ablit.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/check.h"
#include "kernels/common.cuh"

namespace model {
namespace ablit {

namespace {

constexpr int kDimA = 5120, kInterA = 2304, kInterR = 576, kThreads = 256;
constexpr char kMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'A', 'B'};

[[noreturn]] void bad_file(const char* what, const std::string& path) {
  std::fprintf(stderr, "ablit: %s: %s\n", path.c_str(), what);
  std::exit(1);
}

__device__ __forceinline__ float row_lam(const ops::StepIn* in, int r) {
  const int k = min(max(in->sess[r], 0), ops::kMaxSess - 1);
  return in->samp[k].ablit;
}

__device__ __forceinline__ float e4m3(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

// The block's sums of a and b, in every thread.
__device__ __forceinline__ void block_sum2(float& a, float& b) {
  __shared__ float sa[kThreads / 32], sb[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, o);
    b += __shfl_xor_sync(0xffffffffu, b, o);
  }
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  if (l == 0) sa[w] = a, sb[w] = b;
  __syncthreads();
  a = 0.f, b = 0.f;
  for (int i = 0; i < kThreads / 32; ++i) a += sa[i], b += sb[i];
  __syncthreads();
}

// y[r] -= lam Q (Q^T y[r]) over the first 5120 elements of row r (a block a row). in: the step input whose rows carry
// their strength (lam = row's * s); nullptr: lam = lam_all for every row.
template <class T>
__global__ void k_rows(T* y, int ld, const float* __restrict__ Q, float s, const ops::StepIn* in, float lam_all) {
  const int r = blockIdx.x;
  const float lam = (in ? row_lam(in, r) : lam_all) * s;
  if (lam == 0.f) return;
  T* row = y + (size_t)r * ld;
  float d0 = 0.f, d1 = 0.f;
  for (int i = threadIdx.x; i < kDimA; i += kThreads) {
    const float v = (float)row[i];
    d0 = fmaf(Q[2 * i], v, d0);
    d1 = fmaf(Q[2 * i + 1], v, d1);
  }
  block_sum2(d0, d1);
  for (int i = threadIdx.x; i < kDimA; i += kThreads) {
    const float v = (float)row[i];
    row[i] = (T)(v - lam * fmaf(Q[2 * i], d0, Q[2 * i + 1] * d1));
  }
}

// G^T h of one row: h the shared expert's e4m3 activations with their ue8m0 scale per 32, G [576][2]
__device__ __forceinline__ void dots_g(const uint8_t* hq, const uint8_t* hs, const float* __restrict__ G, float& c0,
                                       float& c1) {
  c0 = 0.f, c1 = 0.f;
  for (int j = threadIdx.x; j < kInterR; j += kThreads) {
    const float h = e4m3(hq[j]) * dsk::ue8m0_to_f32(hs[j / 32]);
    c0 = fmaf(G[2 * j], h, c0);
    c1 = fmaf(G[2 * j + 1], h, c1);
  }
  block_sum2(c0, c1);
}

__global__ void k_shexp(float* y, int ld, const uint8_t* hq, size_t hq_row, const uint8_t* hs, size_t hs_row,
                        const float* __restrict__ G, const float* __restrict__ Q, float s, const ops::StepIn* in) {
  const int r = blockIdx.x;
  const float lam = row_lam(in, r) * s;
  if (lam == 0.f) return;
  float c0, c1;
  dots_g(hq + (size_t)r * hq_row, hs + (size_t)r * hs_row, G, c0, c1);
  float* row = y + (size_t)r * ld;
  for (int i = threadIdx.x; i < kDimA; i += kThreads) row[i] -= lam * fmaf(Q[2 * i], c0, Q[2 * i + 1] * c1);
}

__global__ void k_shexp_dots(float* c, const uint8_t* hq, size_t hq_row, const uint8_t* hs, size_t hs_row,
                             const float* __restrict__ G) {
  const int r = blockIdx.x;
  float c0, c1;
  dots_g(hq + (size_t)r * hq_row, hs + (size_t)r * hs_row, G, c0, c1);
  if (threadIdx.x == 0) c[2 * r] = c0, c[2 * r + 1] = c1;
}

// rows [r0, r0 + n) of the chunk: c = the four ranks' dot products added in rank order (every rank adds the same
// numbers in the same order, so a row's edit is the same whichever rank holds the row)
__global__ void k_shexp_apply(__nv_bfloat16* y, int ld, const float* c_all, int Mp, int r0, const float* __restrict__ Q,
                              float lam) {
  const int i = blockIdx.x, r = r0 + i;
  float c0 = 0.f, c1 = 0.f;
  for (int k = 0; k < 4; ++k) c0 += c_all[((size_t)k * Mp + r) * 2], c1 += c_all[((size_t)k * Mp + r) * 2 + 1];
  __nv_bfloat16* row = y + (size_t)i * ld;
  for (int j = threadIdx.x; j < kDimA; j += kThreads) {
    const float v = __bfloat162float(row[j]);
    row[j] = __float2bfloat16_rn(v - lam * fmaf(Q[2 * j], c0, Q[2 * j + 1] * c1));
  }
}

}  // namespace

void load(Pack& p, const std::string& path, int rank) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) bad_file("cannot open", path);
  uint8_t head[64];
  if (std::fread(head, 1, 64, f) != 64 || std::memcmp(head, kMagic, 8)) bad_file("not an ablit file (magic)", path);
  uint32_t h[6];
  std::memcpy(h, head + 8, sizeof h);
  const uint32_t version = h[0], layers = h[1], dim = h[2], inter = h[3], k = h[4];
  if (version != 1 || layers != (uint32_t)kLayers || dim != (uint32_t)kDimA || inter != (uint32_t)kInterA || k < 1 ||
      k > (uint32_t)kK)
    bad_file("unsupported version or shape", path);
  std::vector<float> q((size_t)dim * k), sc((size_t)layers * 2), g((size_t)layers * inter * k);
  if (std::fread(q.data(), 4, q.size(), f) != q.size() || std::fread(sc.data(), 4, sc.size(), f) != sc.size() ||
      std::fread(g.data(), 4, g.size(), f) != g.size())
    bad_file("short file", path);
  if (std::fgetc(f) != EOF) bad_file("trailing bytes", path);
  std::fclose(f);
  // Q and G as kK columns (a k = 1 file's second column is zero)
  std::vector<float> q2((size_t)kDimA * kK, 0.f);
  for (int i = 0; i < kDimA; ++i)
    for (uint32_t c = 0; c < k; ++c) q2[(size_t)i * kK + c] = q[(size_t)i * k + c];
  CUDA_CHECK(cudaMalloc(&p.Q, q2.size() * 4));
  CUDA_CHECK(cudaMemcpy(p.Q, q2.data(), q2.size() * 4, cudaMemcpyHostToDevice));
  std::vector<float> gr((size_t)kInterR * kK);
  for (int L = 0; L < kLayers; ++L) {
    p.s[L][0] = sc[(size_t)L * 2], p.s[L][1] = sc[(size_t)L * 2 + 1];
    if (p.s[L][1] == 0.f) continue;
    std::fill(gr.begin(), gr.end(), 0.f);
    for (int j = 0; j < kInterR; ++j)  // rank r holds intermediate units [576 r, 576 r + 576) (weights/plan.cpp)
      for (uint32_t c = 0; c < k; ++c) gr[(size_t)j * kK + c] = g[((size_t)L * inter + (size_t)rank * kInterR + j) * k + c];
    CUDA_CHECK(cudaMalloc(&p.G[L], gr.size() * 4));
    CUDA_CHECK(cudaMemcpy(p.G[L], gr.data(), gr.size() * 4, cudaMemcpyHostToDevice));
  }
  p.on = true;
}

void reserve(Pack& p, int M) {
  const int Mp = (M + 1) & ~1;
  if (Mp <= p.c_rows) return;
  if (p.c_part) CUDA_CHECK(cudaFree(p.c_part));
  if (p.c_all) CUDA_CHECK(cudaFree(p.c_all));
  CUDA_CHECK(cudaMalloc(&p.c_part, (size_t)Mp * kK * 4));
  CUDA_CHECK(cudaMalloc(&p.c_all, (size_t)4 * Mp * kK * 4));
  CUDA_CHECK(cudaMemset(p.c_part, 0, (size_t)Mp * kK * 4));
  p.c_rows = Mp;
}

void rows_bf16(uint16_t* y, int R, int ld, const Pack& p, int L, const ops::StepIn* in, cudaStream_t st) {
  if (!p.on || p.s[L][0] == 0.f || R <= 0) return;
  k_rows<__nv_bfloat16><<<R, kThreads, 0, st>>>(reinterpret_cast<__nv_bfloat16*>(y), ld, p.Q, p.s[L][0], in, 0.f);
  CUDA_CHECK(cudaGetLastError());
}

void shexp(float* y, int R, int ld, const uint8_t* hq, size_t hq_row, const uint8_t* hs, size_t hs_row, const Pack& p,
           int L, const ops::StepIn* in, cudaStream_t st) {
  if (!p.on || p.s[L][1] == 0.f || R <= 0) return;
  k_shexp<<<R, kThreads, 0, st>>>(y, ld, hq, hq_row, hs, hs_row, p.G[L], p.Q, p.s[L][1], in);
  CUDA_CHECK(cudaGetLastError());
}

void rows_bf16_lam(uint16_t* y, int n, int ld, const Pack& p, int L, float lam, cudaStream_t st) {
  if (!p.on || p.s[L][0] == 0.f || lam == 0.f || n <= 0) return;
  k_rows<__nv_bfloat16><<<n, kThreads, 0, st>>>(reinterpret_cast<__nv_bfloat16*>(y), ld, p.Q, p.s[L][0], nullptr, lam);
  CUDA_CHECK(cudaGetLastError());
}

void shexp_dots(const Pack& p, int L, int M, const uint8_t* hq, size_t hq_row, const uint8_t* hs, size_t hs_row,
                cudaStream_t st) {
  if (M <= 0) return;
  k_shexp_dots<<<M, kThreads, 0, st>>>(p.c_part, hq, hq_row, hs, hs_row, p.G[L]);
  CUDA_CHECK(cudaGetLastError());
}

void shexp_apply_bf16(uint16_t* y, int n, int ld, int r0, int M, const Pack& p, int L, float lam, cudaStream_t st) {
  if (n <= 0) return;
  const int Mp = (M + 1) & ~1;
  k_shexp_apply<<<n, kThreads, 0, st>>>(reinterpret_cast<__nv_bfloat16*>(y), ld, p.c_all, Mp, r0, p.Q, lam * p.s[L][1]);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ablit
}  // namespace model
