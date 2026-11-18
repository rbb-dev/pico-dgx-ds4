// RoPE tables per YaRN factor: (a) factor 16 equals the reference builder below bit for bit over 0..1,048,576 positions;
// (b) factors 32 and 64 match a double-precision evaluation within kFreqRel / kTrig and keep factor 16's bits outside
// the ramp; (c) the computed plain rotation equals the plain table over 0..4,194,304 positions. rope_cs equals the
// table at sampled positions for each factor. Exits non-zero on a failure.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "attn/attn.h"
#include "attn/rope_calc.cuh"
#include "common/check.h"

namespace {

// Reference builder for (a): the single-factor table kernel (factor 16), kept verbatim.
__global__ void ref_kernel(float2* table, int max_pos, float base, int yarn, float low, float inv_span) {
  int i = threadIdx.x;
  int t = blockIdx.x * blockDim.y + threadIdx.y;
  if (t >= max_pos) return;
  float e = __fmul_rn((float)(2 * i), 1.0f / 64.0f);
  float f = __fdiv_rn(1.0f, powf(base, e));
  if (yarn) {
    float ramp = fminf(fmaxf(__fmul_rn(__fsub_rn((float)i, low), inv_span), 0.0f), 1.0f);
    float smooth = __fsub_rn(1.0f, ramp);
    float a = __fmul_rn(__fmul_rn(f, 1.0f / 16.0f), __fsub_rn(1.0f, smooth));
    float b = __fmul_rn(f, smooth);
    f = __fadd_rn(a, b);
  }
  float ang = __fmul_rn((float)t, f);
  table[(size_t)t * 32 + i] = make_float2(cosf(ang), sinf(ang));
}

void ref_build(float2* table, int max_pos, bool compressed) {
  float base = compressed ? 160000.0f : 10000.0f;
  float low = 0.f, inv_span = 1.f;
  if (compressed) {
    const double dim = 64, orig = 65536, b = 160000.0;
    auto corrected = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(b)); };
    double lo = std::max(std::floor(corrected(32)), 0.0);
    double hi = std::min(std::ceil(corrected(1)), dim - 1);
    low = (float)lo;
    inv_span = 1.0f / (float)std::max(hi - lo, 1e-3);
  }
  ref_kernel<<<(max_pos + 7) / 8, dim3(32, 8)>>>(table, max_pos, base, compressed, low, inv_span);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void freq_kernel(float* out, attn::RopeParams p) { out[threadIdx.x] = attn::rope_freq(p, threadIdx.x); }

__global__ void calc_kernel(float2* out, const int* pos, int n, attn::RopeParams p) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n * 32) out[k] = attn::rope_cs(p, pos[k / 32], k % 32);
}

int g_fail = 0;
void check(bool ok, const char* what, long long a = 0, long long b = 0) {
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s (%lld, %lld)\n", what, a, b);
  }
}

constexpr double kFreqRel = 4e-7, kTrig = 4e-7;
constexpr int kChunk = 1 << 18;  // positions per host copy

// The YaRN frequency of pair i at `factor`, double precision, model.py's formula.
double freq_double(int i, int factor) {
  const double dim = 64, orig = 65536, base = 160000.0;
  auto corrected = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(base)); };
  double lo = std::max(std::floor(corrected(32)), 0.0), hi = std::min(std::ceil(corrected(1)), dim - 1);
  double f = 1.0 / std::pow(base, 2.0 * i / dim);
  double ramp = std::min(std::max((i - lo) / std::max(hi - lo, 1e-3), 0.0), 1.0), smooth = 1.0 - ramp;
  return f / factor * (1 - smooth) + f * smooth;
}

// Visits the table [0, n) of `d` in host chunks: fn(first position, entries of the chunk).
template <class Fn>
void each_chunk(const float2* d, int n, Fn fn) {
  std::vector<float2> h((size_t)kChunk * 32);
  for (int t0 = 0; t0 < n; t0 += kChunk) {
    int m = std::min(kChunk, n - t0);
    CUDA_CHECK(cudaMemcpy(h.data(), d + (size_t)t0 * 32, (size_t)m * 32 * sizeof(float2), cudaMemcpyDeviceToHost));
    fn(t0, m, h.data());
  }
}

void part_a(bool compressed) {
  const int n = 1048576 + 1;  // positions 0..1,048,576
  float2 *ref, *a, *b;
  size_t bytes = (size_t)n * 32 * sizeof(float2);
  CUDA_CHECK(cudaMalloc(&ref, bytes));
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  ref_build(ref, n, compressed);
  attn::rope_table_build(a, n, compressed, 16, 0);
  attn::rope_table_build(b, n, compressed, 0);  // the factor-less call
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float2> hr((size_t)kChunk * 32), ha(hr.size());
  long long bad = 0;
  for (int pass = 0; pass < 2; ++pass)
    for (int t0 = 0; t0 < n; t0 += kChunk) {
      int m = std::min(kChunk, n - t0);
      CUDA_CHECK(cudaMemcpy(hr.data(), ref + (size_t)t0 * 32, (size_t)m * 32 * sizeof(float2), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ha.data(), (pass ? b : a) + (size_t)t0 * 32, (size_t)m * 32 * sizeof(float2),
                            cudaMemcpyDeviceToHost));
      bad += std::memcmp(hr.data(), ha.data(), (size_t)m * 32 * sizeof(float2)) != 0;
    }
  std::printf("(a) %s table, factor 16 and factor-less call vs the previous builder, %d positions x 32 pairs: %s\n",
              compressed ? "YaRN " : "plain", n, bad ? "DIFFERENT" : "bit-identical");
  check(bad == 0, "factor 16 differs from the previous builder", compressed, bad);
  CUDA_CHECK(cudaFree(ref));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
}

void part_b(int factor, int n, const float2* t16) {
  attn::RopeParams p = attn::rope_params(true, factor);
  float* df;
  CUDA_CHECK(cudaMalloc(&df, 32 * sizeof(float)));
  freq_kernel<<<1, 32>>>(df, p);
  float f[32];
  CUDA_CHECK(cudaMemcpy(f, df, sizeof f, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(df));
  double worst_rel = 0;
  for (int i = 0; i < 32; ++i) worst_rel = std::max(worst_rel, std::fabs(f[i] / freq_double(i, factor) - 1.0));
  check(worst_rel <= kFreqRel, "frequency beyond bound", factor, (long long)(worst_rel * 1e9));

  float2* tab;
  CUDA_CHECK(cudaMalloc(&tab, (size_t)n * 32 * sizeof(float2)));
  attn::rope_table_build(tab, n, true, factor, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  double worst = 0;
  long long over = 0, same_low = 0, diff_high = 0;
  each_chunk(tab, n, [&](int t0, int m, const float2* h) {
    for (int k = 0; k < m; ++k)
      for (int i = 0; i < 32; ++i) {
        float ang = (float)(t0 + k) * f[i];  // IEEE fp32 product, as the device's __fmul_rn
        double c = std::cos((double)ang), s = std::sin((double)ang);
        double e = std::max(std::fabs(h[k * 32 + i].x - c), std::fabs(h[k * 32 + i].y - s));
        worst = std::max(worst, e);
        over += e > kTrig;
      }
  });
  // pairs outside the ramp keep factor 16's bits; the ramp's pairs differ (positions up to 1M exist in t16)
  std::vector<float2> h(32);
  for (int t : {1, 1000, 65536, 1000000}) {
    CUDA_CHECK(cudaMemcpy(h.data(), tab + (size_t)t * 32, 32 * sizeof(float2), cudaMemcpyDeviceToHost));
    std::vector<float2> g(32);
    CUDA_CHECK(cudaMemcpy(g.data(), t16 + (size_t)t * 32, 32 * sizeof(float2), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 32; ++i) {
      bool eq = !std::memcmp(&h[i], &g[i], sizeof(float2));
      if (i <= 14) same_low += eq ? 0 : 1;
      if (i >= 15 && !eq) ++diff_high;
    }
  }
  check(same_low == 0, "pairs 0..14 differ from factor 16", factor, same_low);
  check(diff_high > 0, "no ramp pair differs from factor 16", factor);

  // rope_cs (no table) at sampled positions equals the table entry bit for bit
  std::vector<int> pos;
  for (int t = 0; t < n; t += 4099) pos.push_back(t);
  pos.push_back(n - 1);
  int* dpos;
  float2* dout;
  CUDA_CHECK(cudaMalloc(&dpos, pos.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dout, pos.size() * 32 * sizeof(float2)));
  CUDA_CHECK(cudaMemcpy(dpos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice));
  calc_kernel<<<(int)((pos.size() * 32 + 255) / 256), 256>>>(dout, dpos, (int)pos.size(), p);
  CUDA_CHECK(cudaGetLastError());
  std::vector<float2> hc(pos.size() * 32), ht(32);
  CUDA_CHECK(cudaMemcpy(hc.data(), dout, hc.size() * sizeof(float2), cudaMemcpyDeviceToHost));
  long long cs_bad = 0;
  for (size_t k = 0; k < pos.size(); ++k) {
    CUDA_CHECK(cudaMemcpy(ht.data(), tab + (size_t)pos[k] * 32, 32 * sizeof(float2), cudaMemcpyDeviceToHost));
    cs_bad += std::memcmp(ht.data(), &hc[k * 32], 32 * sizeof(float2)) != 0;
  }
  check(cs_bad == 0, "rope_cs differs from the table", factor, cs_bad);
  CUDA_CHECK(cudaFree(dpos));
  CUDA_CHECK(cudaFree(dout));
  CUDA_CHECK(cudaFree(tab));
  std::printf("(b) factor %2d, positions 0..%d: frequency worst rel %.2e (bound %.0e); table vs double cos/sin of "
              "fp32 angle worst abs %.2e (bound %.0e, %lld over); rope_cs = table at %zu positions; %s\n",
              factor, n - 1, worst_rel, kFreqRel, worst, kTrig, over, pos.size(),
              (worst_rel <= kFreqRel && over == 0 && !cs_bad && !same_low && diff_high) ? "ok" : "FAILED");
  check(over == 0, "entries beyond the trig bound", factor, over);
}

}  // namespace

__global__ void plain_cmp_kernel(const float2* table, int n, attn::RopeParams p, unsigned long long* bad) {
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= (size_t)n * 32) return;
  float2 c = attn::rope_cs(p, (int)(k >> 5), (int)(k & 31));
  if (__float_as_uint(c.x) != __float_as_uint(table[k].x) || __float_as_uint(c.y) != __float_as_uint(table[k].y))
    atomicAdd(bad, 1ull);
}

void part_c() {
  const int n = 4194304 + 1;  // positions 0..4,194,304
  float2* tab;
  CUDA_CHECK(cudaMalloc(&tab, (size_t)n * 32 * sizeof(float2)));
  attn::rope_table_build(tab, n, false, 0);
  unsigned long long* bad;
  CUDA_CHECK(cudaMalloc(&bad, sizeof *bad));
  CUDA_CHECK(cudaMemset(bad, 0, sizeof *bad));
  const attn::Rope r = attn::rope_plain_calc();
  check(r.table == nullptr && r.p.yarn == 0, "rope_plain_calc is not a computed plain rotation");
  plain_cmp_kernel<<<(unsigned)(((size_t)n * 32 + 255) / 256), 256>>>(tab, n, r.p, bad);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long hb = 0;
  CUDA_CHECK(cudaMemcpy(&hb, bad, sizeof hb, cudaMemcpyDeviceToHost));
  std::printf("(c) plain rotation computed vs plain table, %d positions x 32 pairs: %llu differ\n", n, hb);
  check(hb == 0, "computed plain rotation differs from the plain table", 0, (long long)hb);
  CUDA_CHECK(cudaFree(tab));
  CUDA_CHECK(cudaFree(bad));
  // the tables' rows do not follow the window
  const int w1 = attn::rope_table_rows(1 << 20), w2 = attn::rope_table_rows(1 << 21), w4 = attn::rope_table_rows(1 << 22);
  const int d1 = attn::rope_table_rows(1 << 20, 8), d2 = attn::rope_table_rows(1 << 21, 8), d4 = attn::rope_table_rows(1 << 22, 8);
  std::printf("(c) table rows at max_ctx 1M / 2M / 4M: %d / %d / %d; drafter %d / %d; max_ctx 65536: %d\n", w1, w2, w4, d1, d4,
              attn::rope_table_rows(65536));
  check(w2 == w4 && d2 == d4 && w1 <= w4 && d1 <= d4, "table rows follow the window", w2, w4);
  check(w4 == (1 << 20) + 4096 && w4 < (1 << 22), "table rows beyond the factor-16 window and margin", w4);
  check(attn::rope_table_rows(65536) == 65536 && attn::rope_table_rows(65536, 8) == 65544, "small windows lose rows", 0);
}

int main() {
  part_a(false);
  part_a(true);
  float2* t16;
  CUDA_CHECK(cudaMalloc(&t16, (size_t)1048577 * 32 * sizeof(float2)));
  attn::rope_table_build(t16, 1048577, true, 16, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  part_b(32, 2097152, t16);
  part_b(64, 4194304, t16);
  CUDA_CHECK(cudaFree(t16));
  part_c();
  part_d();
  std::printf(g_fail ? "FAILED (%d)\n" : "all ok\n", g_fail);
  return g_fail != 0;
}
