// The runtime abliteration's kernels (model/ablit.h) against a double-precision host reference: a file loads on every
// rank with that rank's G rows; rows_bf16 and shexp project exactly the rows whose session asks for it (others keep their
// bits); rows_bf16_lam and shexp_dots / shexp_apply_bf16 on the prefill path. One GB10, no ring, no weights.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/check.h"
#include "model/ablit.h"

using namespace model;

namespace {

constexpr int kD = 5120, kI = 2304, kIR = 576, kL = kLayers;
int fails = 0;
void expect(bool ok, const char* what, double got, double want) {
  if (!ok && fails++ < 10) std::printf("FAIL %s: got %.6g want %.6g\n", what, got, want);
}

float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
uint16_t bits(float x) {
  __nv_bfloat16 b = __float2bfloat16_rn(x);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
float from_bits(uint16_t u) {
  __nv_bfloat16 b;
  std::memcpy(&b, &u, 2);
  return __bfloat162float(b);
}
float e4m3(uint8_t v) {
  __nv_fp8_e4m3 x;
  x.__x = v;
  return float(x);
}

// a random orthonormal [kD][2] by Gram-Schmidt
std::vector<double> basis(std::mt19937& g) {
  std::normal_distribution<double> n;
  std::vector<double> q(2 * kD);
  for (auto& v : q) v = n(g);
  double a = 0;
  for (int i = 0; i < kD; ++i) a += q[2 * i] * q[2 * i];
  a = std::sqrt(a);
  for (int i = 0; i < kD; ++i) q[2 * i] /= a;
  double d = 0;
  for (int i = 0; i < kD; ++i) d += q[2 * i] * q[2 * i + 1];
  for (int i = 0; i < kD; ++i) q[2 * i + 1] -= d * q[2 * i];
  double b = 0;
  for (int i = 0; i < kD; ++i) b += q[2 * i + 1] * q[2 * i + 1];
  b = std::sqrt(b);
  for (int i = 0; i < kD; ++i) q[2 * i + 1] /= b;
  return q;
}

}  // namespace

int main() {
  std::mt19937 g(1234);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  const std::vector<double> qd = basis(g);
  std::vector<float> q(qd.begin(), qd.end()), sc(2 * kL), G((size_t)kL * kI * 2);
  for (int L = 0; L < kL; ++L) sc[2 * L] = L < 32 ? 0.5f + 0.04f * L : 0.f, sc[2 * L + 1] = L >= 10 ? 0.1f + 0.05f * (L - 10) : 0.f;
  for (auto& v : G) v = u(g);
  const std::string path = "/tmp/ablit_test.bin";
  {
    FILE* f = std::fopen(path.c_str(), "wb");
    const uint32_t h[6] = {1, (uint32_t)kL, (uint32_t)kD, (uint32_t)kI, 2, 0};
    std::fwrite("DSGB10AB", 1, 8, f);
    std::fwrite(h, 4, 6, f);
    const uint8_t sha[32] = {};
    std::fwrite(sha, 1, 32, f);
    std::fwrite(q.data(), 4, q.size(), f);
    std::fwrite(sc.data(), 4, sc.size(), f);
    std::fwrite(G.data(), 4, G.size(), f);
    std::fclose(f);
  }
  // every rank's slice
  ablit::Pack P[4];
  for (int r = 0; r < 4; ++r) {
    ablit::load(P[r], path, r);
    std::vector<float> got((size_t)kIR * 2);
    CUDA_CHECK(cudaMemcpy(got.data(), P[r].G[20], got.size() * 4, cudaMemcpyDeviceToHost));
    for (int j = 0; j < kIR; ++j)
      expect(got[2 * j] == G[((size_t)20 * kI + r * kIR + j) * 2] && got[2 * j + 1] == G[((size_t)20 * kI + r * kIR + j) * 2 + 1],
             "G slice", got[2 * j], G[((size_t)20 * kI + r * kIR + j) * 2]);
    expect(P[r].G[5] == nullptr, "no G where the shared expert is not edited", 0, 0);
  }
  ablit::Pack& p = P[1];

  // a step of R rows in three sessions: sessions 0 and 2 ask for the edit, session 1 does not
  const int R = 7, L = 20;
  ops::StepIn in{};
  const int sess[R] = {0, 0, 1, 1, 1, 2, 2};
  for (int r = 0; r < R; ++r) in.sess[r] = sess[r];
  in.samp[0].ablit = 1.f, in.samp[1].ablit = 0.f, in.samp[2].ablit = 1.f;
  ops::StepIn* in_d;
  CUDA_CHECK(cudaMalloc(&in_d, sizeof in));
  CUDA_CHECK(cudaMemcpy(in_d, &in, sizeof in, cudaMemcpyHostToDevice));

  // rows_bf16
  std::vector<uint16_t> y((size_t)R * kD);
  for (auto& v : y) v = bits(u(g) * 4.f);
  uint16_t* y_d;
  CUDA_CHECK(cudaMalloc(&y_d, y.size() * 2));
  CUDA_CHECK(cudaMemcpy(y_d, y.data(), y.size() * 2, cudaMemcpyHostToDevice));
  ablit::rows_bf16(y_d, R, kD, p, L, in_d, 0);
  std::vector<uint16_t> y2(y.size());
  CUDA_CHECK(cudaMemcpy(y2.data(), y_d, y.size() * 2, cudaMemcpyDeviceToHost));
  for (int r = 0; r < R; ++r) {
    const bool on = in.samp[sess[r]].ablit != 0.f;
    double d0 = 0, d1 = 0;
    for (int i = 0; i < kD; ++i) d0 += qd[2 * i] * from_bits(y[r * kD + i]), d1 += qd[2 * i + 1] * from_bits(y[r * kD + i]);
    double resid0 = 0, resid1 = 0;
    for (int i = 0; i < kD; ++i) {
      const double x = from_bits(y[r * kD + i]);
      const double want = on ? x - sc[2 * L] * (qd[2 * i] * d0 + qd[2 * i + 1] * d1) : x;
      const double got = from_bits(y2[r * kD + i]);
      if (!on) expect(y2[r * kD + i] == y[r * kD + i], "rows_bf16: a row without the edit keeps its bits", got, want);
      else expect(std::fabs(got - want) <= 0.01 * std::fabs(want) + 1e-2, "rows_bf16: projected row", got, want);
      resid0 += qd[2 * i] * got, resid1 += qd[2 * i + 1] * got;
    }
    if (on) {  // the component along Q is (1 - s) of what it was
      expect(std::fabs(resid0 - (1 - sc[2 * L]) * d0) <= 0.05 * std::fabs(d0) + 0.05, "rows_bf16: residual along q0", resid0, (1 - sc[2 * L]) * d0);
      expect(std::fabs(resid1 - (1 - sc[2 * L]) * d1) <= 0.05 * std::fabs(d1) + 0.05, "rows_bf16: residual along q1", resid1, (1 - sc[2 * L]) * d1);
    }
  }

  // shexp: activations e4m3 [R][576] with a ue8m0 scale per 32, a fp32 partial
  std::vector<uint8_t> hq((size_t)R * kIR), hs((size_t)R * (kIR / 32));
  std::uniform_int_distribution<int> byte(0, 255), ex(120, 130);
  for (auto& v : hq) {
    do v = (uint8_t)byte(g); while ((v & 0x7f) == 0x7f);  // no NaN codes
  }
  for (auto& v : hs) v = (uint8_t)ex(g);
  std::vector<float> ym((size_t)R * kD);
  for (auto& v : ym) v = u(g) * 2.f;
  uint8_t *hq_d, *hs_d;
  float* ym_d;
  CUDA_CHECK(cudaMalloc(&hq_d, hq.size()));
  CUDA_CHECK(cudaMalloc(&hs_d, hs.size()));
  CUDA_CHECK(cudaMalloc(&ym_d, ym.size() * 4));
  CUDA_CHECK(cudaMemcpy(hq_d, hq.data(), hq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hs_d, hs.data(), hs.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ym_d, ym.data(), ym.size() * 4, cudaMemcpyHostToDevice));
  ablit::shexp(ym_d, R, kD, hq_d, kIR, hs_d, kIR / 32, p, L, in_d, 0);
  std::vector<float> ym2(ym.size());
  CUDA_CHECK(cudaMemcpy(ym2.data(), ym_d, ym.size() * 4, cudaMemcpyDeviceToHost));
  auto cdots = [&](int r, double& c0, double& c1) {
    c0 = c1 = 0;
    for (int j = 0; j < kIR; ++j) {
      const double h = e4m3(hq[r * kIR + j]) * std::ldexp(1.0, hs[r * (kIR / 32) + j / 32] - 127);
      c0 += G[((size_t)L * kI + 1 * kIR + j) * 2] * h, c1 += G[((size_t)L * kI + 1 * kIR + j) * 2 + 1] * h;
    }
  };
  for (int r = 0; r < R; ++r) {
    const bool on = in.samp[sess[r]].ablit != 0.f;
    double c0, c1;
    cdots(r, c0, c1);
    for (int i = 0; i < kD; ++i) {
      const double want = on ? ym[r * kD + i] - sc[2 * L + 1] * (qd[2 * i] * c0 + qd[2 * i + 1] * c1) : ym[r * kD + i];
      if (!on) expect(std::memcmp(&ym2[r * kD + i], &ym[r * kD + i], 4) == 0, "shexp: a row without the edit keeps its bits", ym2[r * kD + i], want);
      else expect(std::fabs(ym2[r * kD + i] - want) <= 1e-4 * std::fabs(want) + 1e-4, "shexp: projected row", ym2[r * kD + i], want);
    }
  }

  // prefill: one strength for the chunk's rows
  CUDA_CHECK(cudaMemcpy(y_d, y.data(), y.size() * 2, cudaMemcpyHostToDevice));
  ablit::rows_bf16_lam(y_d, R, kD, p, L, 0.f, 0);
  CUDA_CHECK(cudaMemcpy(y2.data(), y_d, y.size() * 2, cudaMemcpyDeviceToHost));
  expect(std::memcmp(y2.data(), y.data(), y.size() * 2) == 0, "rows_bf16_lam at 0 keeps every bit", 0, 0);
  // shexp_dots on the four ranks' slices of the same rows (each its own G), the gather by copies, the apply on rank 1
  ablit::reserve(p, R);
  const int Mp = (R + 1) & ~1;
  std::vector<float> call((size_t)4 * Mp * 2, 0.f);
  for (int r = 0; r < 4; ++r) {
    ablit::reserve(P[r], R);
    ablit::shexp_dots(P[r], L, R, hq_d, kIR, hs_d, kIR / 32, 0);
    CUDA_CHECK(cudaMemcpy(call.data() + (size_t)r * Mp * 2, P[r].c_part, (size_t)R * 2 * 4, cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaMemcpy(p.c_all, call.data(), call.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(y_d, y.data(), y.size() * 2, cudaMemcpyHostToDevice));
  const int r0 = 2, n = 3;
  ablit::shexp_apply_bf16(y_d + (size_t)0, n, kD, r0, R, p, L, 1.f, 0);  // y's first n rows stand for rows r0..
  CUDA_CHECK(cudaMemcpy(y2.data(), y_d, y.size() * 2, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    double c0 = 0, c1 = 0;
    for (int rk = 0; rk < 4; ++rk)
      for (int j = 0; j < kIR; ++j) {
        const double h = e4m3(hq[(r0 + i) * kIR + j]) * std::ldexp(1.0, hs[(r0 + i) * (kIR / 32) + j / 32] - 127);
        c0 += G[((size_t)L * kI + rk * kIR + j) * 2] * h, c1 += G[((size_t)L * kI + rk * kIR + j) * 2 + 1] * h;
      }
    for (int j = 0; j < kD; ++j) {
      const double want = bf((float)(from_bits(y[i * kD + j]) - sc[2 * L + 1] * (qd[2 * j] * c0 + qd[2 * j + 1] * c1)));
      expect(std::fabs(from_bits(y2[i * kD + j]) - want) <= 0.01 * std::fabs(want) + 1e-2, "shexp_apply_bf16", from_bits(y2[i * kD + j]), want);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
  return fails ? 1 : 0;
}
