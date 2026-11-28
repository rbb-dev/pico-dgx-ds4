// The sampler's rules that more than one file applies, as one code for the device (sample.cu's k_sample, spec.cu's k_spec_in)
// and the host (tests/samp_rule_test.cpp checks them against their own oracles, no GPU needed): which requests ride the Gumbel
// path, how the path's bit shares StepSamp::top_k with a real top-k, the certified head's row test, the sampler's noise and the
// Gumbel-max pick among kept tokens, and the settings' parsers.
//
// The Gumbel path (DSGB10_GSAMPLE, model.h): a sampled request on it carries kSampGumbel in StepSamp::top_k, its real top-k in the
// bits below. A truncated (top-p / top-k) row of such a request takes the Gumbel-max winner among the SAME kept tokens the
// walk keeps, with the noise g(seed, position, id) the untruncated sampler uses, so that a draft row and the row that verifies it
// see one noise (DSGB10_GCOUPLE, dspark.h).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#define SAMP_HD __host__ __device__
#else
#define SAMP_HD
#endif

namespace model::ops {

constexpr int32_t kSampGumbel = 1 << 30;  // StepSamp::top_k's bit: the session is on the Gumbel path
constexpr int kGDepths = 5;                // the drafter's depths (dspark::kDraft; spec.cu asserts it)

// the real top-k of a StepSamp::top_k (0: none); the bit is not a top-k
SAMP_HD inline int32_t samp_top_k(int32_t top_k) { return top_k > 0 ? (top_k & (kSampGumbel - 1)) : 0; }
// the session samples (T > 0) and is on the Gumbel path
SAMP_HD inline bool samp_gumbel(int32_t top_k, float temperature) {
  return temperature > 0.f && top_k > 0 && (top_k & kSampGumbel) != 0;
}
// a client's top-k as a request carries it: never a value that reads as the bit (2^30 - 1 and above all keep every token)
SAMP_HD inline int32_t client_top_k(uint32_t v) { return (int32_t)(v < (uint32_t)kSampGumbel ? v : (uint32_t)kSampGumbel - 1u); }

// StepSamp::top_k of a request: its top-k, with the bit when the request rides the Gumbel path.
//   mode 0: never. 1: a sampled request that is unseeded or not truncated (a seeded top-p / top-k request keeps its sampler
//   and its text). 2: every sampled request. Greedy requests never ride it.
SAMP_HD inline int32_t gsample_top_k_mode(int mode, int32_t top_k, float temperature, float top_p, bool seeded) {
  const int32_t k = top_k > 0 ? (top_k < kSampGumbel ? top_k : kSampGumbel - 1) : 0;
  if (mode <= 0) return top_k > 0 ? k : top_k;
  const bool trunc = k > 0 || top_p < 1.f;
  return temperature > 0.f && (mode >= 2 || !seeded || !trunc) ? (k | kSampGumbel) : k;
}

// The certified head's row test (spec.cu k_spec_in): a row takes the certified bodies when it asks no top logprobs and is
// greedy, or (cert_samp) samples with the temperature alone, up to 1: no real top-k, no top-p.
SAMP_HD inline bool cert_row_ok(int32_t n_top, float temperature, float top_p, int32_t top_k, bool cert_samp) {
  return n_top <= 0 && (!(temperature > 0.f) ||
                        (cert_samp && temperature <= 1.f && samp_top_k(top_k) == 0 && !(top_p < 1.f)));
}

// ---- the noise --------------------------------------------------------------------------------------------------------
SAMP_HD inline uint64_t mix64(uint64_t x) {
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 32;
  return x;
}
// The uniform of the noise, strictly inside (0, 1): the top bin's (2^24 - 1 + 0.5) / 2^24 rounds to 1.0 in float32, where
// -log(-log(u)) is +inf and that id would win whatever its logit; it takes the float just under 1, every other bin its own value.
SAMP_HD inline float unit(uint64_t h) {
  const float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
  return u < 1.0f ? u : 0x1.fffffep-1f;
}
// a row's noise key: its seed and position
SAMP_HD inline uint64_t noise_base(uint64_t seed, int pos) { return mix64(seed) ^ ((uint64_t)(uint32_t)pos * 0x9E3779B97F4A7C15ull); }
// the noise of token id at the row keyed by base (standard Gumbel)
SAMP_HD inline float gumbel_noise_at(uint64_t base, int id) {
  return -logf(-logf(unit(mix64(base + (uint64_t)id * 0xD1B54A32D192ED03ull))));
}
SAMP_HD inline float gumbel_noise(uint64_t seed, int pos, int id) { return gumbel_noise_at(noise_base(seed, pos), id); }

// The Gumbel-max winner among the kept tokens uv[0 .. keep) (logits, ids ui): the index of the largest uv / T + noise, ties to
// the lower id. Token i wins with probability exp(uv[i] / T) / sum over the kept: what the walk over one uniform draws.
SAMP_HD inline int gumbel_pick(const float* uv, const int* ui, int keep, float invT, uint64_t base) {
  int pick = keep - 1;
  float bg = -INFINITY;
  int bid = 0x7fffffff;
  for (int i = 0; i < keep; ++i) {
    const float g = uv[i] * invT - logf(-logf(unit(mix64(base + (uint64_t)ui[i] * 0xD1B54A32D192ED03ull))));
    if (g > bg || (g == bg && ui[i] < bid)) bg = g, bid = ui[i], pick = i;
  }
  return pick;
}

// ---- the settings -----------------------------------------------------------------------------------------------------
// What the four ranks must hold alike (main.cpp compares them at the start): a rank with another A, B computes another survival
// and chooses other rows. No padding: compared as bytes.
struct GSettings {
  int32_t mode = 0;     // DSGB10_GSAMPLE: 0, 1 or 2
  int32_t coupled = 0;  // DSGB10_GCOUPLE set: 1
  float td[kGDepths] = {};            // the drafter temperature factor per depth (0 when not coupled)
  float cf_a = 1.f, cf_b = 0.f;       // DSGB10_GCOUPLE_CF: a coupled entry's gate logit A lg + B (1, 0: off)
};
static_assert(sizeof(GSettings) == 4 * (2 + kGDepths + 2), "GSettings has no padding");

// DSGB10_GSAMPLE: unset or empty 0; "0", "1", "2"; anything else -1
inline int parse_gsample(const char* e) {
  if (!e || !*e) return 0;
  if (e[1] == 0 && e[0] >= '0' && e[0] <= '2') return e[0] - '0';
  return -1;
}
// DSGB10_GCOUPLE=Td[,Td2,..]: each in (0, 4], the last repeats to every depth. 0: off (unset, empty or a first value <= 0);
// 1: td[] filled; -1: malformed
inline int parse_gcouple(const char* e, float* td) {
  if (!e || !*e || !(atof(e) > 0.)) return 0;
  int n = 0;
  for (const char* p = e; n < kGDepths && *p;) {
    char* end = nullptr;
    const float v = std::strtof(p, &end);
    if (end == p || !(v > 0.f) || v > 4.f) return -1;
    td[n++] = v;
    p = *end == ',' ? end + 1 : end;
    if (*end && *end != ',') return -1;
  }
  for (int j = n; j < kGDepths; ++j) td[j] = td[n - 1];
  return 1;
}
// DSGB10_GCOUPLE_CF=A,B: 0 unset or empty, 1 filled, -1 malformed
inline int parse_ab(const char* e, float* v) {
  if (!e || !*e) return 0;
  return std::sscanf(e, "%f,%f", &v[0], &v[1]) == 2 ? 1 : -1;
}

}  // namespace model::ops
