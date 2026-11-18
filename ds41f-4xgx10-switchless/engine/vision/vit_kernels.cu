// The image encoder's kernels. Rules for every one of them: no atomics, and no reduction whose order depends on
// the launch (grid, tile, number of heads or output columns). An output element is computed the same way
// whatever slice of the matrix it sits in, which is what lets four ranks give an unsplit run's bytes.
#include <cmath>
#include <cstdio>

#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/pa_common.cuh"
#include "vision/vit.h"

namespace vit {
namespace {

using dsk::pa::cp16;
using dsk::pa::cp_commit;
using dsk::pa::cp_wait;

__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2b(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ uint32_t pack2(float lo, float hi) { return dsk::pack_bf16x2(lo, hi); }
__device__ __forceinline__ void mma16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                      uint32_t b0, uint32_t b1) {
  dsk::mma_bf16(d, a0, a1, a2, a3, b0, b1);
}

template <class P>
__device__ __forceinline__ P at(const ViewT<P>& v, int r, int c) {
  return v.p + (size_t)(c / v.cb) * v.blk + (size_t)r * v.ld + (c % v.cb);
}

// ------------------------------------------------------------------------------------------------ gemm
constexpr int kBM = 128, kBN = 64, kBK = 64, kSK = kBK + 8, kStages = 3, kGemmThreads = 256;
constexpr int kGemmSmem = kStages * (kBM + kBN) * kSK * 2;  // 82,944 bytes

struct GemmArgs {
  View a;
  const bf16* w;
  const bf16* bias;
  OutView out;
  int M, N, K;
};

// 128 x 64 tile, 8 warps of 32 x 32, stages of 64 k, three stages in flight. Each output element is one chain
// of m16n8k16 products over k in order; nothing about the tile changes that chain.
__global__ void __launch_bounds__(kGemmThreads) k_gemm(GemmArgs a) {
  extern __shared__ __align__(128) uint8_t smem[];
  bf16* const sm = reinterpret_cast<bf16*>(smem);
  constexpr int kStageElems = (kBM + kBN) * kSK;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp & 3, wn = warp >> 2;
  const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
  const int KT = a.K / kBK;

  auto load = [&](int kt, int buf) {
    bf16* const sa = sm + buf * kStageElems;
    bf16* const sb = sa + kBM * kSK;
    const int k0 = kt * kBK;
#pragma unroll
    for (int j = 0; j < 4; j++) {  // 128 rows x 8 chunks of 8 k
      const int i = tid + j * kGemmThreads, r = i >> 3, c = i & 7, row = m0 + r;
      const bool ok = row < a.M;
      cp16(sa + r * kSK + c * 8, at(a.a, ok ? row : 0, k0 + c * 8), ok);
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {  // 64 rows x 8 chunks
      const int i = tid + j * kGemmThreads, r = i >> 3, c = i & 7;
      cp16(sb + r * kSK + c * 8, a.w + (size_t)(n0 + r) * a.K + k0 + c * 8, true);
    }
  };

  float acc[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; mi++)
#pragma unroll
    for (int ni = 0; ni < 4; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.f;

#pragma unroll
  for (int s = 0; s < kStages - 1; s++) {
    if (s < KT) load(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < KT; kt++) {
    cp_wait<kStages - 2>();
    __syncthreads();
    if (kt + kStages - 1 < KT) load(kt + kStages - 1, (kt + kStages - 1) % kStages);
    cp_commit();
    const bf16* const sa = sm + (kt % kStages) * kStageElems;
    const bf16* const sb = sa + kBM * kSK;
#pragma unroll
    for (int ks = 0; ks < kBK / 16; ks++) {
      uint32_t af[2][4], bfr[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; mi++) {
        const bf16* p = sa + (wm * 32 + mi * 16 + g) * kSK + ks * 16 + 2 * t;
        af[mi][0] = *reinterpret_cast<const uint32_t*>(p);
        af[mi][1] = *reinterpret_cast<const uint32_t*>(p + 8 * kSK);
        af[mi][2] = *reinterpret_cast<const uint32_t*>(p + 8);
        af[mi][3] = *reinterpret_cast<const uint32_t*>(p + 8 * kSK + 8);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ni++) {
        const bf16* p = sb + (wn * 32 + ni * 8 + g) * kSK + ks * 16 + 2 * t;
        bfr[ni][0] = *reinterpret_cast<const uint32_t*>(p);
        bfr[ni][1] = *reinterpret_cast<const uint32_t*>(p + 8);
      }
#pragma unroll
      for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
          mma16(acc[mi][ni], af[mi][0], af[mi][1], af[mi][2], af[mi][3], bfr[ni][0], bfr[ni][1]);
    }
  }
  cp_wait<0>();

#pragma unroll
  for (int mi = 0; mi < 2; mi++)
#pragma unroll
    for (int ni = 0; ni < 4; ni++) {
      const int col = n0 + wn * 32 + ni * 8 + 2 * t;
      const float b0 = a.bias ? b2f(a.bias[col]) : 0.f, b1 = a.bias ? b2f(a.bias[col + 1]) : 0.f;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int row = m0 + wm * 32 + mi * 16 + g + h * 8;
        if (row >= a.M) continue;
        const uint32_t v = pack2(acc[mi][ni][2 * h] + b0, acc[mi][ni][2 * h + 1] + b1);
        *reinterpret_cast<uint32_t*>(at(a.out, row, col)) = v;
      }
    }
}

// ------------------------------------------------------------------------------------------- attention
constexpr int kAQ = 64, kAK = 64, kAS = 72, kAThreads = 128;
constexpr float kScaleLog2 = 0.125f * 1.44269504088896340736f;

// One CTA: 64 queries of one head against all keys, 64 at a time. Warp w owns query rows 16w..16w+15.
// Scores are fp32; p = 2^(s - m) with a running maximum m; the row sum l and the output are rescaled by
// 2^(m_old - m_new) per tile; p is rounded to bf16 for the value product and the output is o / l.
__global__ void __launch_bounds__(kAThreads) k_attn(const bf16* q, int ldq, const bf16* k, int ldk, const bf16* v,
                                                    int ldv, bf16* o, int ldo, int n) {
  __shared__ __align__(16) bf16 ks[kAK * kAS];  // [key][d]
  __shared__ __align__(16) bf16 vt[64 * kAS];    // [d][key]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int head = blockIdx.y, q0 = blockIdx.x * kAQ + warp * 16;
  const int r0 = q0 + g, r1 = q0 + g + 8;

  uint32_t qa[4][4];
#pragma unroll
  for (int ks_ = 0; ks_ < 4; ks_++) {
    const int c = head * 64 + ks_ * 16 + 2 * t;
    qa[ks_][0] = r0 < n ? *reinterpret_cast<const uint32_t*>(q + (size_t)r0 * ldq + c) : 0u;
    qa[ks_][1] = r1 < n ? *reinterpret_cast<const uint32_t*>(q + (size_t)r1 * ldq + c) : 0u;
    qa[ks_][2] = r0 < n ? *reinterpret_cast<const uint32_t*>(q + (size_t)r0 * ldq + c + 8) : 0u;
    qa[ks_][3] = r1 < n ? *reinterpret_cast<const uint32_t*>(q + (size_t)r1 * ldq + c + 8) : 0u;
  }
  float oacc[8][4];
#pragma unroll
  for (int d = 0; d < 8; d++)
#pragma unroll
    for (int e = 0; e < 4; e++) oacc[d][e] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  for (int kb = 0; kb < n; kb += kAK) {
    __syncthreads();  // the previous tile is no longer read
#pragma unroll
    for (int j = 0; j < 4; j++) {  // 64 keys x 8 chunks of 8 d
      const int i = tid + j * kAThreads, r = i >> 3, c = i & 7, key = kb + r;
      const bool ok = key < n;
      cp16(ks + r * kAS + c * 8, k + (size_t)(ok ? key : 0) * ldk + head * 64 + c * 8, ok);
    }
    cp_commit();
    uint4 vr[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int i = tid + j * kAThreads, r = i >> 3, c = i & 7, key = kb + r;
      vr[j] = key < n ? *reinterpret_cast<const uint4*>(v + (size_t)key * ldv + head * 64 + c * 8)
                      : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int i = tid + j * kAThreads, r = i >> 3, c = i & 7;
      const uint32_t w[4] = {vr[j].x, vr[j].y, vr[j].z, vr[j].w};
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t bits = (w[e >> 1] >> (16 * (e & 1))) & 0xFFFFu;
        reinterpret_cast<uint16_t*>(vt)[(c * 8 + e) * kAS + r] = (uint16_t)bits;
      }
    }
    cp_wait<0>();
    __syncthreads();

    float s[8][4];
#pragma unroll
    for (int nj = 0; nj < 8; nj++) {
#pragma unroll
      for (int e = 0; e < 4; e++) s[nj][e] = 0.f;
#pragma unroll
      for (int ks_ = 0; ks_ < 4; ks_++) {
        const bf16* p = ks + (nj * 8 + g) * kAS + ks_ * 16 + 2 * t;
        mma16(s[nj], qa[ks_][0], qa[ks_][1], qa[ks_][2], qa[ks_][3], *reinterpret_cast<const uint32_t*>(p),
              *reinterpret_cast<const uint32_t*>(p + 8));
      }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nj = 0; nj < 8; nj++) {
      const int key = kb + nj * 8 + 2 * t;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const bool ok = key + (e & 1) < n;
        s[nj][e] = ok ? s[nj][e] * kScaleLog2 : -INFINITY;
      }
      mx0 = fmaxf(mx0, fmaxf(s[nj][0], s[nj][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nj][2], s[nj][3]));
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
    m0 = n0, m1 = n1;
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int nj = 0; nj < 8; nj++) {
      s[nj][0] = exp2f(s[nj][0] - n0), s[nj][1] = exp2f(s[nj][1] - n0);
      s[nj][2] = exp2f(s[nj][2] - n1), s[nj][3] = exp2f(s[nj][3] - n1);
      ps0 += s[nj][0], ps0 += s[nj][1];
      ps1 += s[nj][2], ps1 += s[nj][3];
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      ps0 += __shfl_xor_sync(0xffffffffu, ps0, off);
      ps1 += __shfl_xor_sync(0xffffffffu, ps1, off);
    }
    l0 = l0 * c0 + ps0, l1 = l1 * c1 + ps1;
#pragma unroll
    for (int d = 0; d < 8; d++) {
      oacc[d][0] *= c0, oacc[d][1] *= c0, oacc[d][2] *= c1, oacc[d][3] *= c1;
    }
#pragma unroll
    for (int kc = 0; kc < 4; kc++) {  // 16 keys at a time
      const uint32_t a0 = pack2(s[2 * kc][0], s[2 * kc][1]), a1 = pack2(s[2 * kc][2], s[2 * kc][3]);
      const uint32_t a2 = pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]), a3 = pack2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
      for (int d = 0; d < 8; d++) {
        const bf16* p = vt + (d * 8 + g) * kAS + kc * 16 + 2 * t;
        mma16(oacc[d], a0, a1, a2, a3, *reinterpret_cast<const uint32_t*>(p),
              *reinterpret_cast<const uint32_t*>(p + 8));
      }
    }
  }
#pragma unroll
  for (int d = 0; d < 8; d++) {
    const int c = head * 64 + d * 8 + 2 * t;
    if (r0 < n) *reinterpret_cast<uint32_t*>(o + (size_t)r0 * ldo + c) = pack2(oacc[d][0] / l0, oacc[d][1] / l0);
    if (r1 < n) *reinterpret_cast<uint32_t*>(o + (size_t)r1 * ldo + c) = pack2(oacc[d][2] / l1, oacc[d][3] / l1);
  }
}

// ------------------------------------------------------------------------------------------------ rope
struct InvFreq {
  float f[16];
};

__global__ void k_rope_table(int nh, int nw, InvFreq inv, float* cs, float* sn, bool swap_axes) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t total = (size_t)nh * nw * 32;
  if (i >= total) return;
  const int p = (int)(i / 32), j = (int)(i % 32), ph = p / nw, pw = p % nw;
  int pos = j < 16 ? ph : pw;
  if (swap_axes) pos = j < 16 ? pw : ph;
  const float f = (float)pos * inv.f[j & 15];
  cs[i] = cosf(f), sn[i] = sinf(f);
}

// Separate multiplies and one add or subtract each: no fused multiply-add, as the reference's elementwise ops.
__global__ void k_rope_apply(bf16* q, bf16* k, int ld, int n, int heads, const float* cs, const float* sn) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)n * heads * 32) return;
  const int j = (int)(i % 32), h = (int)((i / 32) % heads), r = (int)(i / (32 * heads));
  const float c = cs[(size_t)r * 32 + j], s = sn[(size_t)r * 32 + j];
  bf16* const bufs[2] = {q, k};
#pragma unroll
  for (int b = 0; b < 2; b++) {
    bf16* p = bufs[b] + (size_t)r * ld + h * 64 + j;
    const float x1 = b2f(p[0]), x2 = b2f(p[32]);
    p[0] = f2b(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
    p[32] = f2b(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
  }
}

// ---------------------------------------------------------------------------------------------- rmsnorm
// A warp a row. Lane l sums columns 128 j + 4 l .. +3 for j = 0..7 in order, then the lanes add by a fixed
// butterfly: the order is the same for contiguous rows and for column blocks.
__global__ void k_rmsnorm(View x, const bf16* w, bf16* out, int n) {
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= n) return;
  float v[8][4];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint2 u = *reinterpret_cast<const uint2*>(at(x, row, j * 128 + lane * 4));
    v[j][0] = dsk::bf16_lo(u.x), v[j][1] = dsk::bf16_hi(u.x), v[j][2] = dsk::bf16_lo(u.y), v[j][3] = dsk::bf16_hi(u.y);
#pragma unroll
    for (int e = 0; e < 4; e++) ss += v[j][e] * v[j][e];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float r = rsqrtf(ss / 1024.0f + 1e-6f);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint2 wu = *reinterpret_cast<const uint2*>(w + j * 128 + lane * 4);
    const float wv[4] = {dsk::bf16_lo(wu.x), dsk::bf16_hi(wu.x), dsk::bf16_lo(wu.y), dsk::bf16_hi(wu.y)};
    uint2 o;
    o.x = pack2(wv[0] * (v[j][0] * r), wv[1] * (v[j][1] * r));
    o.y = pack2(wv[2] * (v[j][2] * r), wv[3] * (v[j][3] * r));
    *reinterpret_cast<uint2*>(out + (size_t)row * 1024 + j * 128 + lane * 4) = o;
  }
}

// ------------------------------------------------------------------------------------------- pointwise
__global__ void k_add(const bf16* a, int lda, const bf16* b, int ldb, bf16* out, int ldo, int rows, int cols) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const int half = cols / 2;
  if (i >= (size_t)rows * half) return;
  const int r = (int)(i / half), c = (int)(i % half) * 2;
  const uint32_t x = *reinterpret_cast<const uint32_t*>(a + (size_t)r * lda + c);
  const uint32_t y = *reinterpret_cast<const uint32_t*>(b + (size_t)r * ldb + c);
  *reinterpret_cast<uint32_t*>(out + (size_t)r * ldo + c) =
      pack2(dsk::bf16_lo(x) + dsk::bf16_lo(y), dsk::bf16_hi(x) + dsk::bf16_hi(y));
}

__device__ __forceinline__ float rn_bf16(float v) { return b2f(f2b(v)); }

__global__ void k_swiglu(const bf16* gu, int ldg, int width, bf16* out, int ldo, int rows) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const int half = width / 2;
  if (i >= (size_t)rows * half) return;
  const int r = (int)(i / half), c = (int)(i % half) * 2;
  const uint32_t gt = *reinterpret_cast<const uint32_t*>(gu + (size_t)r * ldg + c);
  const uint32_t up = *reinterpret_cast<const uint32_t*>(gu + (size_t)r * ldg + width + c);
  const float g0 = dsk::bf16_lo(gt), g1 = dsk::bf16_hi(gt);
  const float s0 = rn_bf16(g0 / (1.0f + expf(-g0))), s1 = rn_bf16(g1 / (1.0f + expf(-g1)));
  *reinterpret_cast<uint32_t*>(out + (size_t)r * ldo + c) = pack2(s0 * dsk::bf16_lo(up), s1 * dsk::bf16_hi(up));
}

__global__ void k_gelu(const bf16* x, int ldx, bf16* out, int ldo, int rows, int cols) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const int half = cols / 2;
  if (i >= (size_t)rows * half) return;
  const int r = (int)(i / half), c = (int)(i % half) * 2;
  const uint32_t u = *reinterpret_cast<const uint32_t*>(x + (size_t)r * ldx + c);
  const float a = dsk::bf16_lo(u), b = dsk::bf16_hi(u);
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  *reinterpret_cast<uint32_t*>(out + (size_t)r * ldo + c) =
      pack2(a * 0.5f * (1.0f + erff(a * kInvSqrt2)), b * 0.5f * (1.0f + erff(b * kInvSqrt2)));
}

// ----------------------------------------------------------------------------------------- copy kernels
__global__ void k_patchify(const uint8_t* px, int nh, int nw, const bf16* table, bf16* out) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)nh * nw * kPatchKPad) return;
  const int p = (int)(i / kPatchKPad), k = (int)(i % kPatchKPad);
  bf16 v = f2b(0.f);
  if (k < kPatchK) {
    const int c = k / 196, rem = k % 196, py = rem / 14, pxx = rem % 14;
    const int y = (p / nw) * 14 + py, x = (p % nw) * 14 + pxx;
    v = table[px[((size_t)y * nw * 14 + x) * 3 + c]];
  }
  out[i] = v;
}

__global__ void k_unfold3(const bf16* x, int nh, int nw, bf16* u, int variant) {
  const int lw = (nw + 2) / 3, lh = (nh + 2) / 3;
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)lh * lw * kMergeIn) return;
  const int l = (int)(i / kMergeIn), col = (int)(i % kMergeIn);
  int c, w9;
  if (variant == 1) c = col % kDim, w9 = col / kDim;
  else c = col / 9, w9 = col % 9;
  const int bh = variant == 2 ? l % lh : l / lw, bw = variant == 2 ? l / lh : l % lw;
  const int y = 3 * bh + w9 / 3, xx = 3 * bw + w9 % 3;
  u[i] = (y < nh && xx < nw) ? x[((size_t)y * nw + xx) * kDim + c] : (variant == 3 ? f2b(1.0f) : f2b(0.f));
}

__global__ void k_span(View g, const bf16* start, const bf16* nl, const bf16* end, int lh, int lw, int first,
                       int count, bf16* dst) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  constexpr int kChunks = kLm / 8;
  if (i >= (size_t)count * kChunks) return;
  const int r = first + (int)(i / kChunks), c = (int)(i % kChunks) * 8;
  const int total = lh * (lw + 1) + 2;
  const bf16* src;
  if (r == 0) src = start + c;
  else if (r == total - 1) src = end + c;
  else {
    const int q = r - 1, ri = q / (lw + 1), ci = q % (lw + 1);
    src = ci == lw ? nl + c : at(g, ri * lw + ci, c);
  }
  *reinterpret_cast<uint4*>(dst + (size_t)(i / kChunks) * kLm + c) = *reinterpret_cast<const uint4*>(src);
}

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x ^= x >> 33, x *= 0xff51afd7ed558ccdull, x ^= x >> 33, x *= 0xc4ceb9fe1a85ec53ull, x ^= x >> 33;
  return x;
}

// A block covers 16 rows; the partial is a sum of per-element mixes, so no order matters.
__global__ void k_digest(View x, int rows, int cols, uint64_t* part) {
  uint64_t h = 0;
  for (int e = threadIdx.x; e < 16 * cols; e += blockDim.x) {
    const int r = blockIdx.x * 16 + e / cols, c = e % cols;
    if (r >= rows) break;
    const uint16_t v = *reinterpret_cast<const uint16_t*>(at(x, r, c));
    h += mix64(((uint64_t)r * cols + c) * 0x9E3779B97F4A7C15ull ^ v);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) h += __shfl_xor_sync(0xffffffffu, h, off);
  __shared__ uint64_t w[32];
  if ((threadIdx.x & 31) == 0) w[threadIdx.x / 32] = h;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t s = 0;
    for (int i = 0; i < (int)blockDim.x / 32; i++) s += w[i];
    part[blockIdx.x] = s;
  }
}

constexpr int kT = 256;
inline unsigned blocks_for(size_t n) { return (unsigned)((n + kT - 1) / kT); }

}  // namespace

void gemm(const View& a, const bf16* w, const bf16* bias, const OutView& out, int M, int N, int K,
          cudaStream_t st) {
  if (N % kBN || K % kBK || a.cb % kBK) die("vit::gemm: N %d, K %d, block %d are not multiples of 64", N, K, a.cb);
  static const bool once = [] {
    CUDA_CHECK(cudaFuncSetAttribute(k_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, kGemmSmem));
    return true;
  }();
  (void)once;
  GemmArgs args{a, w, bias, out, M, N, K};
  k_gemm<<<dim3(N / kBN, (M + kBM - 1) / kBM), kGemmThreads, kGemmSmem, st>>>(args);
  CUDA_CHECK(cudaGetLastError());
}

void attention(const bf16* q, int ldq, const bf16* k, int ldk, const bf16* v, int ldv, bf16* o, int ldo, int n,
               int heads, cudaStream_t st) {
  k_attn<<<dim3((n + kAQ - 1) / kAQ, heads), kAThreads, 0, st>>>(q, ldq, k, ldk, v, ldv, o, ldo, n);
  CUDA_CHECK(cudaGetLastError());
}

void rope_table(int nh, int nw, const float* inv_freq16, float* cos, float* sin, bool swap_axes, cudaStream_t st) {
  InvFreq f;
  for (int i = 0; i < 16; i++) f.f[i] = inv_freq16[i];
  k_rope_table<<<blocks_for((size_t)nh * nw * 32), kT, 0, st>>>(nh, nw, f, cos, sin, swap_axes);
  CUDA_CHECK(cudaGetLastError());
}

void rope_apply(bf16* q, bf16* k, int ld, int n, int heads, const float* cos, const float* sin, cudaStream_t st) {
  k_rope_apply<<<blocks_for((size_t)n * heads * 32), kT, 0, st>>>(q, k, ld, n, heads, cos, sin);
  CUDA_CHECK(cudaGetLastError());
}

void rmsnorm(const View& x, const bf16* w, bf16* out, int n, cudaStream_t st) {
  if (x.cb % 128) die("vit::rmsnorm: block width %d is not a multiple of 128", x.cb);
  k_rmsnorm<<<(n + 7) / 8, 256, 0, st>>>(x, w, out, n);
  CUDA_CHECK(cudaGetLastError());
}

void add(const bf16* a, int lda, const bf16* b, int ldb, bf16* out, int ldo, int rows, int cols, cudaStream_t st) {
  k_add<<<blocks_for((size_t)rows * cols / 2), kT, 0, st>>>(a, lda, b, ldb, out, ldo, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void swiglu(const bf16* gu, int ldg, int width, bf16* out, int ldo, int rows, cudaStream_t st) {
  k_swiglu<<<blocks_for((size_t)rows * width / 2), kT, 0, st>>>(gu, ldg, width, out, ldo, rows);
  CUDA_CHECK(cudaGetLastError());
}

void gelu(const bf16* x, int ldx, bf16* out, int ldo, int rows, int cols, cudaStream_t st) {
  k_gelu<<<blocks_for((size_t)rows * cols / 2), kT, 0, st>>>(x, ldx, out, ldo, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

void patchify(const uint8_t* pixels, int nh, int nw, const bf16* table256, bf16* out, cudaStream_t st) {
  k_patchify<<<blocks_for((size_t)nh * nw * kPatchKPad), kT, 0, st>>>(pixels, nh, nw, table256, out);
  CUDA_CHECK(cudaGetLastError());
}

void unfold3(const bf16* x, int nh, int nw, bf16* u, int variant, cudaStream_t st) {
  const Grid g = make_grid(nh, nw);
  k_unfold3<<<blocks_for((size_t)g.merged() * kMergeIn), kT, 0, st>>>(x, nh, nw, u, variant);
  CUDA_CHECK(cudaGetLastError());
}

void span_rows(const View& g, const bf16* start, const bf16* newline, const bf16* end, int lh, int lw, int first,
               int count, bf16* dst, cudaStream_t st) {
  if (g.cb % 8) die("vit::span_rows: block width %d", g.cb);
  k_span<<<blocks_for((size_t)count * (kLm / 8)), kT, 0, st>>>(g, start, newline, end, lh, lw, first, count, dst);
  CUDA_CHECK(cudaGetLastError());
}

size_t digest_scratch_bytes(int rows) { return (size_t)((rows + 15) / 16) * 8; }

uint64_t digest(const View& x, int rows, int cols, uint64_t* scratch_dev, cudaStream_t st) {
  const int nb = (rows + 15) / 16;
  k_digest<<<nb, 256, 0, st>>>(x, rows, cols, scratch_dev);
  CUDA_CHECK(cudaGetLastError());
  std::vector<uint64_t> h(nb);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), scratch_dev, (size_t)nb * 8, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  uint64_t s = 0;
  for (uint64_t v : h) s += v;
  return s;
}

}  // namespace vit
