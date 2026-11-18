// Host references of the encoder's operations, written from the model's definition (vision.py), used by the
// host and GPU tests. Slow and plain on purpose.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "vision/vit_weights.h"

namespace vit {
namespace ref {

// sum_k a[k] * w[k] in double
inline double dot(const u16* a, const u16* w, int k) {
  double s = 0;
  for (int i = 0; i < k; i++) s += (double)from_bf16(a[i]) * (double)from_bf16(w[i]);
  return s;
}

// x [nh*nw][1024] -> [lh*lw][9216], channel-major 3 x 3 windows of the zero-padded grid, windows in reading order
inline std::vector<u16> unfold3(const std::vector<u16>& x, int nh, int nw) {
  const Grid g = make_grid(nh, nw);
  std::vector<u16> u((size_t)g.merged() * kMergeIn, 0);
  for (int bh = 0; bh < g.lh; bh++)
    for (int bw = 0; bw < g.lw; bw++)
      for (int c = 0; c < kDim; c++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            const int y = 3 * bh + i, xx = 3 * bw + j;
            if (y < nh && xx < nw) u[((size_t)bh * g.lw + bw) * kMergeIn + c * 9 + i * 3 + j] = x[((size_t)y * nw + xx) * kDim + c];
          }
  return u;
}

// the span's rows from the aligner's output [lh*lw][5120]
inline std::vector<u16> span(const std::vector<u16>& g5120, const Weights& d, int lh, int lw) {
  std::vector<u16> s;
  auto put = [&](const u16* p) { s.insert(s.end(), p, p + kLm); };
  put(d.start.data());
  for (int r = 0; r < lh; r++) {
    for (int c = 0; c < lw; c++) put(g5120.data() + ((size_t)r * lw + c) * kLm);
    put(d.newline.data());
  }
  put(d.end.data());
  return s;
}

// u8 pixels [nh*14][nw*14][3] -> [n][640] through the table; columns are (channel, row, column), the tail zero
inline std::vector<u16> patchify(const std::vector<uint8_t>& px, int nh, int nw, const std::vector<u16>& table) {
  std::vector<u16> o((size_t)nh * nw * kPatchKPad, 0);
  for (int p = 0; p < nh * nw; p++)
    for (int c = 0; c < 3; c++)
      for (int py = 0; py < kPatch; py++)
        for (int pxx = 0; pxx < kPatch; pxx++) {
          const int y = (p / nw) * kPatch + py, x = (p % nw) * kPatch + pxx;
          o[(size_t)p * kPatchKPad + c * 196 + py * 14 + pxx] = table[px[((size_t)y * nw * kPatch + x) * 3 + c]];
        }
  return o;
}

// RMS norm of one row in double: w * (x * rsqrt(mean(x^2) + 1e-6))
inline std::vector<double> rmsnorm_row(const u16* x, const u16* w) {
  double ss = 0;
  for (int i = 0; i < kDim; i++) ss += (double)from_bf16(x[i]) * from_bf16(x[i]);
  const double r = 1.0 / std::sqrt(ss / kDim + 1e-6);
  std::vector<double> o(kDim);
  for (int i = 0; i < kDim; i++) o[i] = (double)from_bf16(w[i]) * ((double)from_bf16(x[i]) * r);
  return o;
}

// One query row of one head against n keys (q, k, v [n][ld] with the head at column h * 64), in double:
// softmax(q . k / 8) . v
inline std::vector<double> attention_row(const u16* q, const u16* k, const u16* v, int ld, int n, int h, int row) {
  std::vector<double> s(n);
  double mx = -1e300;
  for (int j = 0; j < n; j++) {
    double d = 0;
    for (int i = 0; i < 64; i++) d += (double)from_bf16(q[(size_t)row * ld + h * 64 + i]) * from_bf16(k[(size_t)j * ld + h * 64 + i]);
    s[j] = d / 8.0, mx = std::fmax(mx, s[j]);
  }
  double l = 0;
  std::vector<double> o(64, 0.0);
  for (int j = 0; j < n; j++) {
    const double p = std::exp(s[j] - mx);
    l += p;
    for (int i = 0; i < 64; i++) o[i] += p * from_bf16(v[(size_t)j * ld + h * 64 + i]);
  }
  for (double& x : o) x /= l;
  return o;
}

}  // namespace ref
}  // namespace vit
