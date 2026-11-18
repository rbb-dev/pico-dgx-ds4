// RoPE tables and the standalone RoPE kernel.
#include <cmath>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/rope_calc.cuh"
#include "common/check.h"

namespace attn {
namespace {

// model.py precompute_freqs_cis, op by op in fp32 on the GPU (torch computes it on the GPU too):
//   freqs = 1 / base**(arange(0,64,2)/64); YaRN: freqs/factor*(1-smooth) + freqs*smooth;
//   angle = fp32(t) * freqs; (cos, sin) = polar(1, angle).
// torch divides a tensor by a Python scalar as a multiply by the scalar's fp32 reciprocal, which
// matters for the ramp: (i - low) * fp32(1/span), not (i - low) / span.
__global__ void rope_table_kernel(float2* table, int max_pos, RopeParams p) {
  int i = threadIdx.x;  // 0..31
  int t = blockIdx.x * blockDim.y + threadIdx.y;
  if (t >= max_pos) return;
  table[(size_t)t * 32 + i] = rope_cs(p, t, i);
}

__global__ void rope_rows_kernel(uint16_t* x, int rows, int heads, int dim, const int* pos,
                                 Rope table, int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;  // over rows * heads * 32
  if (pair >= rows * heads * 32) return;
  int i = pair & 31, rh = pair >> 5, r = rh / heads;
  uint16_t* p = x + (size_t)rh * dim + (dim - kRope) + 2 * i;
  float2 cs = rope_at(table, pos[r], i);
  float x0 = bf2f(p[0]), x1 = bf2f(p[1]);
  rope_pair(x0, x1, cs.x, cs.y, inverse);
  p[0] = f2bf(x0);
  p[1] = f2bf(x1);
}

}  // namespace

RopeParams rope_params(bool compressed, int factor) {
  RopeParams p{compressed ? 160000.0f : 10000.0f, compressed ? 1 : 0, 0.f, 1.f, 1.0f / (float)factor};
  if (compressed) {  // corrected_dim() in Python doubles, exactly as model.py
    const double dim = 64, orig = 65536, b = 160000.0;
    auto corrected = [&](double rot) { return dim * std::log(orig / (rot * 2 * M_PI)) / (2 * std::log(b)); };
    double lo = std::max(std::floor(corrected(32)), 0.0);
    double hi = std::min(std::ceil(corrected(1)), dim - 1);
    p.low = (float)lo;
    p.inv_span = 1.0f / (float)std::max(hi - lo, 1e-3);
  }
  return p;
}

Rope rope_yarn_calc(int factor) {
  if (factor != 16 && factor != 32 && factor != 64) die("rope_yarn_calc: YaRN factor %d (16, 32 or 64)", factor);
  return Rope(rope_params(true, factor));
}

Rope rope_plain_calc() { return Rope(rope_params(false, 16)); }

void rope_table_build(float2* table, int max_pos, bool compressed, int factor, cudaStream_t s) {
  if (factor != 16 && factor != 32 && factor != 64) die("rope_table_build: YaRN factor %d (16, 32 or 64)", factor);
  dim3 block(32, 8);
  rope_table_kernel<<<(max_pos + 7) / 8, block, 0, s>>>(table, max_pos, rope_params(compressed, factor));
  CUDA_CHECK(cudaGetLastError());
}

void rope_table_build(float2* table, int max_pos, bool compressed, cudaStream_t s) {
  rope_table_build(table, max_pos, compressed, 16, s);
}

void rope_rows(uint16_t* x, int rows, int heads, int dim, const int* pos, Rope table, bool inverse,
               cudaStream_t s) {
  int n = rows * heads * 32;
  rope_rows_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, rows, heads, dim, pos, table, inverse);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace attn
