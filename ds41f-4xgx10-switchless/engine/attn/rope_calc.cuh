// The RoPE recipe as device functions, shared by the table builder (rope.cu) and any kernel that computes
// (cos, sin) from (position, pair) instead of reading a table. Contract: rope_cs(p, t, i) is bit-identical to
// the table entry [t][i] that rope_table_build writes for the same params, because that builder calls it.
// The operation order is model.py precompute_freqs_cis in fp32 (see rope.cu).
#pragma once
#include "attn/attn.h"

namespace attn {

// Frequency of pair i (0..31) after YaRN, fp32.
__device__ __forceinline__ float rope_freq(const RopeParams& p, int i) {
  float e = __fmul_rn((float)(2 * i), 1.0f / 64.0f);
  float f = __fdiv_rn(1.0f, powf(p.base, e));
  if (p.yarn) {
    float ramp = fminf(fmaxf(__fmul_rn(__fsub_rn((float)i, p.low), p.inv_span), 0.0f), 1.0f);
    float smooth = __fsub_rn(1.0f, ramp);
    float a = __fmul_rn(__fmul_rn(f, p.inv_factor), __fsub_rn(1.0f, smooth));
    float b = __fmul_rn(f, smooth);
    f = __fadd_rn(a, b);
  }
  return f;
}

// (cos, sin) of position t at pair i, from that pair's frequency f = rope_freq(p, i).
__device__ __forceinline__ float2 rope_cs_f(float f, int t) {
  float ang = __fmul_rn((float)t, f);
  return make_float2(cosf(ang), sinf(ang));
}

__device__ __forceinline__ float2 rope_cs(const RopeParams& p, int t, int i) { return rope_cs_f(rope_freq(p, i), t); }

// Entry [t][i] of a Rope: the table's, or the computed one (the same bits).
__device__ __forceinline__ float2 rope_at(const Rope& r, int t, int i) {
  return r.table ? r.table[(size_t)t * 32 + i] : rope_cs(r.p, t, i);
}

// One row's entries by pair index: row[i] = rope_at(r, t, i).
struct RopeRow {
  Rope r;
  int t;
  __device__ __forceinline__ float2 operator[](int i) const { return rope_at(r, t, i); }
};

}  // namespace attn
