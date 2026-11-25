// The draft module's copy of the sampler's noise and of the Gumbel-path test (model/samp_rule.h: mix64, unit, noise_base,
// gumbel_noise, samp_gumbel). The module includes none of the model's headers, so the code is here; the constants and the
// order of operations must stay the same there and here, and tests/samp_rule_test.cpp compares the two bit for bit.
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define NZ_HD __host__ __device__ __forceinline__
#else
#define NZ_HD inline
#endif

namespace dspark {

inline constexpr int32_t kNoiseGumbel = 1 << 30;  // StepSamp::top_k's bit (ops::kSampGumbel)

NZ_HD uint64_t nz_mix64(uint64_t x) {
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 32;
  return x;
}
NZ_HD float nz_gumbel(uint64_t seed, int pos, int id) {
  const uint64_t base = nz_mix64(seed) ^ ((uint64_t)(uint32_t)pos * 0x9E3779B97F4A7C15ull);
  const uint64_t h = nz_mix64(base + (uint64_t)id * 0xD1B54A32D192ED03ull);
  const float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
  return -logf(-logf(u));
}
// a session is coupled: it samples and is on the Gumbel path (ops::samp_gumbel)
NZ_HD bool nz_on(float temperature, int32_t top_k) { return temperature > 0.f && top_k > 0 && (top_k & kNoiseGumbel) != 0; }

}  // namespace dspark
