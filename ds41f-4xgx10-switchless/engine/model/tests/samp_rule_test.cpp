// The sampler's shared rules (model/samp_rule.h) and the draft module's copy of the noise (dspark/noise.h), on the host, no GPU.
// Holds:
//  H1  the certified head's row test: a row of T <= 1, top-p 1, no real top-k, no logprobs takes the certified bodies whether
//      or not it carries the Gumbel-path bit; a real top-k, a top-p or logprobs still keep a row off them
//  H2  which requests ride the Gumbel path, for modes 0, 1 and 2: mode 0 never; greedy never; mode 1 leaves a seeded truncated
//      request alone; a client value of 2^30 or more never reads as the bit; the real top-k survives the bit
//  H3  the Gumbel-max pick over 1 .. 64 kept tokens draws token i with probability exp(l_i / T) / sum (chi-square, p > 1e-4,
//      2,000,000 positions a case)
//  H4  the draft module's noise equals the sampler's bit for bit, and its path test equals the rule's
//  H5  the settings' parsers
//  H6  the noise's uniform stays inside the open interval (0, 1): the top bin, (2^24 - 1 + 0.5) / 2^24, rounds to 1.0 in float32,
//      where -log(-log(u)) is +inf and that id would win whatever its logit; it takes the float just under 1, and every other bin
//      keeps its bits; the draft module's copy the same
// Every row has its control: the wrong rule compiled into this test must FAIL the same check (reported as "control rejected").
// Command: samp_rule_test (exit 0 = all pass)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "dspark/noise.h"
#include "model/samp_rule.h"

using namespace model::ops;

namespace {

int fails = 0, checks = 0;
void check(bool ok, const char* what, long a = 0, long b = 0) {
  ++checks;
  if (ok) return;
  if (++fails <= 40) std::printf("FAIL: %s (%ld, %ld)\n", what, a, b);
}

uint64_t g_rng = 0x2545F4914F6CDD1Dull;
uint64_t rnd64() {
  g_rng ^= g_rng << 13, g_rng ^= g_rng >> 7, g_rng ^= g_rng << 17;
  return g_rng;
}
double rnd01() { return (double)(rnd64() >> 11) * (1.0 / 9007199254740992.0); }

// ---- H1 ---------------------------------------------------------------------------------------------------------------
// the predicate before the bit existed: `!(top_k > 0)`
bool old_cert(int32_t n_top, float T, float top_p, int32_t top_k, bool cert_samp) {
  return n_top <= 0 && (!(T > 0.f) || (cert_samp && T <= 1.f && !(top_k > 0) && !(top_p < 1.f)));
}
template <class F>
int h1(F rule) {
  int bad = 0;
  const float Ts[] = {0.f, 0.3f, 1.f, 1.2f};
  const float Ps[] = {1.f, 0.95f};
  const int32_t Ks[] = {-1, 0, 5, 1 << 20, (1 << 30) - 1};
  for (float T : Ts)
    for (float P : Ps)
      for (int32_t k : Ks)
        for (int32_t flag : {0, (int32_t)kSampGumbel})
          for (int nt : {0, 1})
            for (int cs : {0, 1}) {
              const int32_t top_k = k > 0 ? (k | flag) : flag;  // a request's StepSamp::top_k
              const bool real_k = k > 0;
              // the expectation, from the contract: logprobs keep a row off; greedy is certified; a sampled row needs
              // cert_samp, T <= 1, no real top-k and no top-p (the bit alone changes nothing)
              const bool want = nt == 0 && (!(T > 0.f) || (cs && T <= 1.f && !real_k && !(P < 1.f)));
              if (rule(nt, T, P, top_k, cs != 0) != want) ++bad;
            }
  return bad;
}

// ---- H2 ---------------------------------------------------------------------------------------------------------------
// a client's top-k as the request path clamped it before the bit existed
int32_t old_clamp(uint32_t v) { return (int32_t)std::min<uint32_t>(v, 1u << 30); }
// the rule as the old clamp would feed it: no clamp inside, the bit set from the (already clamped) value
int32_t ctl_gsample(int mode, int32_t top_k, float T, float top_p, bool seeded) {
  if (mode <= 0) return top_k;
  const bool trunc = top_k > 0 || top_p < 1.f;
  return T > 0.f && (mode >= 2 || !seeded || !trunc) ? (top_k | kSampGumbel) : top_k;
}
template <class F, class C>
int h2(F rule, C clamp) {
  int bad = 0;
  const uint32_t clients[] = {0u, 5u, (1u << 30) - 1u, 1u << 30, 0x7FFFFFFFu, 0xFFFFFFFFu};
  for (int mode = 0; mode <= 2; ++mode)
    for (float T : {0.f, 0.7f, 1.f})
      for (int seeded = 0; seeded < 2; ++seeded)
        for (float P : {1.f, 0.95f})
          for (uint32_t v : clients) {
            const int32_t k = clamp(v);
            const int32_t r = rule(mode, k, T, P, seeded != 0);
            const int32_t k_exp = (int32_t)std::min<uint32_t>(v, (1u << 30) - 1u);  // 0 stays 0
            const bool trunc = k_exp > 0 || P < 1.f;
            const bool flag_exp = mode > 0 && T > 0.f && (mode == 2 || !seeded || !trunc);
            if (samp_top_k(r) != k_exp) ++bad;
            if (samp_gumbel(r, T) != flag_exp) ++bad;
            if (r < 0) ++bad;  // the field stays a non-negative int
          }
  return bad;
}

// ---- H3 ---------------------------------------------------------------------------------------------------------------
// the regularised upper incomplete gamma Q(a, x) (series below a + 1, continued fraction above)
double gamma_q(double a, double x) {
  if (x <= 0) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; ++n) {
      ap += 1, del *= x / ap, sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-14) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  double b = x + 1 - a, c = 1e300, d = 1.0 / b, h = d;
  for (int i = 1; i < 1000; ++i) {
    const double an = -i * (i - a);
    b += 2;
    d = an * d + b, d = std::fabs(d) < 1e-300 ? 1e-300 : d;
    c = b + an / c, c = std::fabs(c) < 1e-300 ? 1e-300 : c;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < 1e-14) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

struct Pick {  // the rule under test: the index it picks among n tokens
  int (*fn)(const float*, const int*, int, float, uint64_t, int keep);
  const char* name;
};
int pick_real(const float* uv, const int* ui, int keep, float invT, uint64_t base, int) { return gumbel_pick(uv, ui, keep, invT, base); }
int pick_no_invT(const float* uv, const int* ui, int keep, float, uint64_t base, int) { return gumbel_pick(uv, ui, keep, 1.f, base); }
// the pick over one token more than are kept: the scored arrays hold keep + 1 tokens
int pick_keep_plus_1(const float* uv, const int* ui, int keep, float invT, uint64_t base, int) { return gumbel_pick(uv, ui, keep + 1, invT, base); }

// returns the smallest p over the cases (1.0 when every case drew exactly the kept tokens); fills *extra with picks past keep
double h3_case(int keep, float T, Pick rule, long N, long* extra, bool plus1) {
  std::vector<float> uv(keep + 1);
  std::vector<int> ui(keep + 1);
  // distinct ids, logits spread over [-3, 3] (every kept token's probability at T = 0.7 stays above 1e-5)
  for (int i = 0; i <= keep; ++i) {
    ui[i] = (int)(rnd64() % 129280);
    for (int j = 0; j < i; ++j)
      if (ui[j] == ui[i]) ui[i] = (ui[i] + 1) % 129280, j = -1;
    uv[i] = (float)(rnd01() * 6.0 - 3.0);
  }
  std::vector<double> p(keep);
  double z = 0;
  for (int i = 0; i < keep; ++i) p[i] = std::exp((double)uv[i] / T), z += p[i];
  for (int i = 0; i < keep; ++i) p[i] /= z;
  std::vector<long> cnt(keep + 1, 0);
  const uint64_t seed = rnd64();
  const float invT = 1.f / T;
  for (long n = 0; n < N; ++n) {
    const uint64_t base = noise_base(seed, (int)n);
    const int k = rule.fn(uv.data(), ui.data(), keep, invT, base, keep);
    ++cnt[k < keep + 1 ? k : keep];
  }
  *extra = plus1 ? cnt[keep] : 0;
  // bins: tokens in order, pooling any whose expected count is under 5 into one
  double chi = 0;
  int bins = 0;
  double pool_e = 0;
  long pool_o = 0;
  for (int i = 0; i < keep; ++i) {
    const double e = p[i] * N;
    if (e < 5) pool_e += e, pool_o += cnt[i];
    else chi += (cnt[i] - e) * (cnt[i] - e) / e, ++bins;
  }
  if (pool_e >= 5) chi += (pool_o - pool_e) * (pool_o - pool_e) / pool_e, ++bins;
  if (bins < 2) return keep == 1 ? (cnt[0] == N ? 1.0 : 0.0) : 1.0;
  return gamma_q((bins - 1) / 2.0, chi / 2.0);
}

// ---- H4 / H5 ----------------------------------------------------------------------------------------------------------
uint32_t bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

}  // namespace

int main() {
  // H1
  {
    const int real = h1([](int nt, float T, float P, int32_t k, bool cs) { return cert_row_ok(nt, T, P, k, cs); });
    const int ctl = h1(old_cert);
    check(real == 0, "H1 the certified head's row test", real);
    check(ctl > 0, "H1 control: the old predicate (top_k > 0) must fail the test", ctl);
    std::printf("H1: rule %d wrong rows, control (the old predicate) %d wrong rows\n", real, ctl);
  }
  // H2
  {
    const int real = h2([](int m, int32_t k, float T, float P, bool s) { return gsample_top_k_mode(m, k, T, P, s); },
                        [](uint32_t v) { return (int32_t)std::min<uint32_t>(v, (1u << 30) - 1u); });
    const int via = h2([](int m, int32_t k, float T, float P, bool s) { return gsample_top_k_mode(m, k, T, P, s); },
                       [](uint32_t v) { return client_top_k(v); });
    const int ctl = h2(ctl_gsample, old_clamp);
    check(real == 0, "H2 the flag rule", real);
    check(via == 0, "H2 the flag rule through client_top_k", via);
    check(ctl > 0, "H2 control: the old clamp (1u << 30) must fail the test", ctl);
    std::printf("H2: rule %d wrong rows, client_top_k path %d, control (the old clamp) %d wrong rows\n", real, via, ctl);
    // the raw values the design lists, straight into the rule, in every mode (a raw -1 is no top-k)
    for (int m = 0; m <= 2; ++m)
      for (int32_t k : {-1, 0, 5, (1 << 30) - 1, 1 << 30, 0x7FFFFFFF}) {
        const int32_t r = gsample_top_k_mode(m, k, 1.f, 0.95f, false);
        check(samp_top_k(r) == (k > 0 ? std::min<int32_t>(k, (1 << 30) - 1) : 0), "H2 raw top-k survives", m, k);
        check(samp_gumbel(r, 1.f) == (m > 0), "H2 raw top-k flag", m, k);
        check(!samp_gumbel(r, 0.f), "H2 greedy never flagged", m, k);
      }
    check(gsample_top_k_mode(0, -1, 1.f, 1.f, false) == -1, "H2 mode 0 returns the value it was given");
  }
  // H3
  {
    const long N = 2000000;
    double worst = 1.0;
    int ctl_a_rejected = 0, ctl_a_cases = 0, ctl_b_rejected = 0, ctl_b_cases = 0;
    for (int keep : {1, 3, 8, 24, 64})
      for (float T : {0.7f, 1.0f, 1.3f}) {
        long ex = 0;
        const double p = h3_case(keep, T, {pick_real, "real"}, N, &ex, false);
        worst = std::min(worst, p);
        check(p > 1e-4, "H3 the pick's distribution", keep, (long)(T * 10));
        if (keep >= 8) {  // controls: a pick that scores without 1 / T (T != 1), a pick over keep + 1 tokens
          if (T != 1.0f) {
            ++ctl_a_cases;
            const double pc = h3_case(keep, T, {pick_no_invT, "no 1/T"}, N, &ex, false);
            if (pc < 1e-4) ++ctl_a_rejected;
          }
          if (T == 1.0f) {
            ++ctl_b_cases;
            const double pc = h3_case(keep, T, {pick_keep_plus_1, "keep + 1"}, N, &ex, true);
            if (pc < 1e-4 || ex > 0) ++ctl_b_rejected;
          }
        }
      }
    check(ctl_a_rejected == ctl_a_cases && ctl_a_cases > 0, "H3 control: scores without 1 / T must be rejected", ctl_a_rejected, ctl_a_cases);
    check(ctl_b_rejected == ctl_b_cases && ctl_b_cases > 0, "H3 control: a pick over keep + 1 tokens must be rejected", ctl_b_rejected, ctl_b_cases);
    std::printf("H3: 15 cases of %ld positions, smallest p %.3g; control without 1/T rejected %d of %d, control over keep+1 rejected %d of %d\n", N,
                worst, ctl_a_rejected, ctl_a_cases, ctl_b_rejected, ctl_b_cases);
  }
  // H4
  {
    long diff = 0;
    for (int i = 0; i < 1000000; ++i) {
      const uint64_t seed = rnd64();
      const int pos = (int)(rnd64() % 3000000), id = (int)(rnd64() % 129280);
      if (bits(dspark::nz_gumbel(seed, pos, id)) != bits(gumbel_noise(seed, pos, id))) ++diff;
      if (dspark::nz_mix64(seed) != mix64(seed)) ++diff;
    }
    check(diff == 0, "H4 the draft module's noise equals the sampler's", diff);
    // a changed constant in either copy must be seen by the same comparison
    long ctl = 0;
    for (int i = 0; i < 1000; ++i) {
      const uint64_t seed = rnd64();
      const int pos = (int)(rnd64() % 3000000), id = (int)(rnd64() % 129280);
      const uint64_t base = mix64(seed) ^ ((uint64_t)(uint32_t)pos * 0x9E3779B97F4A7C16ull);  // one bit off
      const float other = -logf(-logf(unit(mix64(base + (uint64_t)id * 0xD1B54A32D192ED03ull))));
      if (bits(other) != bits(gumbel_noise(seed, pos, id))) ++ctl;
    }
    check(ctl > 900, "H4 control: a changed constant must be seen", ctl);
    long pd = 0;
    for (int32_t k : {-1, 0, 5, (1 << 30) - 1, 1 << 30, (1 << 30) | 7, 0x7FFFFFFF})
      for (float T : {0.f, 0.5f, 1.f})
        if (dspark::nz_on(T, k) != samp_gumbel(k, T)) ++pd;
    check(pd == 0, "H4 the draft module's path test equals the rule's", pd);
    check(dspark::kNoiseGumbel == kSampGumbel, "H4 the bit is the same constant");
    std::printf("H4: 1,000,000 draws equal bit for bit; control (one constant changed) seen in %ld of 1000\n", ctl);
  }
  // H6
  {
    const auto old_unit = [](uint64_t h) { return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f); };
    long moved = 0, out = 0, draft = 0;
    for (uint64_t b = 0; b < (1ull << 24); ++b) {
      const uint64_t h = (b << 40) | (b * 0x9E3779B9ull & 0xFFFFFFFFFFull);  // the low 40 bits are not the bin's
      const float u = unit(h);
      if (!(u > 0.f && u < 1.f)) ++out;
      if (b + 1 < (1ull << 24) && bits(u) != bits(old_unit(h))) ++moved;
      if (bits(dspark::nz_unit(h)) != bits(u)) ++draft;
    }
    const uint64_t top = 0xFFFFFFull << 40;
    const float g = -logf(-logf(unit(top)));
    check(out == 0, "H6 every bin's uniform inside (0, 1)", out);
    check(moved == 0, "H6 every bin but the top keeps its bits", moved);
    check(draft == 0, "H6 the draft module's uniform equals the sampler's", draft);
    check(std::isfinite(g), "H6 the top bin's noise is finite");
    check(unit(top) == 0x1.fffffep-1f, "H6 the top bin takes the float just under 1");
    check(old_unit(top) == 1.0f && std::isinf(-logf(-logf(old_unit(top)))), "H6 control: the old formula reaches 1.0 and +inf");
    std::printf("H6: 16,777,216 bins: %ld outside (0, 1), %ld moved below the top, top bin %a, noise %.3f; control: old top %a\n", out,
                moved, unit(top), g, old_unit(top));
  }
  // H5
  {
    float td[kGDepths];
    check(parse_gsample(nullptr) == 0 && parse_gsample("") == 0 && parse_gsample("0") == 0 && parse_gsample("1") == 1 &&
              parse_gsample("2") == 2 && parse_gsample("3") == -1 && parse_gsample("11") == -1 && parse_gsample("x") == -1,
          "H5 DSGB10_GSAMPLE");
    check(parse_gcouple(nullptr, td) == 0 && parse_gcouple("", td) == 0 && parse_gcouple("0", td) == 0 && parse_gcouple("-1", td) == 0,
          "H5 DSGB10_GCOUPLE off");
    check(parse_gcouple("0.8", td) == 1 && td[0] == 0.8f && td[4] == 0.8f, "H5 one Td repeats");
    check(parse_gcouple("0.8,0.9,1", td) == 1 && td[1] == 0.9f && td[2] == 1.f && td[3] == 1.f && td[4] == 1.f, "H5 the last Td repeats");
    check(parse_gcouple("0.8,0.9,1,1,1,1,1", td) == 1, "H5 more values than depths are ignored");
    check(parse_gcouple("0.8x", td) == -1 && parse_gcouple("0.8,0", td) == -1 && parse_gcouple("5", td) == -1 && parse_gcouple("0.8,,1", td) == -1,
          "H5 DSGB10_GCOUPLE malformed");
    float ab[2] = {1.f, 0.f};
    check(parse_ab(nullptr, ab) == 0 && parse_ab("", ab) == 0 && parse_ab("1.5,-0.25", ab) == 1 && ab[0] == 1.5f && ab[1] == -0.25f &&
              parse_ab("1.5", ab) == -1 && parse_ab("a,b", ab) == -1,
          "H5 DSGB10_GCOUPLE_CF");
  }
  std::printf("samp_rule_test: %d checks, %d failed\n", checks, fails);
  return fails ? 1 : 0;
}
