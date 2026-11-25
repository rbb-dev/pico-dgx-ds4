// The exact shared steps' settings (docs/batching.md, "Exact shared steps"): their ranges and parsers, on the host, no GPU
// (tests/exact_rule_test.cpp). model.cu reads the values through these; server/main.cpp refuses a start whose value is outside
// its range (exact_env_error), so a mistyped value never runs as another one.
//
//   DSGB10_NORM_EXACT       0 | 1       the separate RMSNorm launches of a 9..64-row step add a row's squares in the fused kernels' order
//   DSGB10_WIDE_EXACT       0..95       bit set of the families of a 17..64-row step that run through the decode kernels' sums:
//                                       1 wq_a + wkv and wq_b, 2 wo_a + wo_b, 4 the gate, 8 the MoE, 16 the Engram's GEMV;
//                                       64 runs the selected families through the row-group kernels (one read of the weights);
//                                       95 = all of them. Bit 32 is not a family.
//   DSGB10_WIDE_EXACT_MT    1 | 2       wo_b's row-group kernel's tiles of 8 weight rows per warp (the same bits)
//   DSGB10_WIDE_EXACT_GQ    0 | 2 | 4 (+16)   moe_rows' groups of 16 row slots a pass; + 16: the raw scales are read
//   DSGB10_R16_EXACT        0..63       test only: the 9..16-row families run as two groups of 8 rows (bits as WIDE_EXACT, 32 the head)
#pragma once
#include <cstdio>
#include <cstdlib>

namespace model::exact {

// A whole decimal number, no sign, no trailing text. false: not one.
inline bool parse_uint(const char* e, int* v) {
  if (!e || !*e) return false;
  long x = 0;
  for (const char* p = e; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
    x = x * 10 + (*p - '0');
    if (x > 1000000) return false;
  }
  *v = (int)x;
  return true;
}

constexpr int kWideFamilies = 1 | 2 | 4 | 8 | 16;  // wq_a/wq_b, wo, gate, MoE, Engram
constexpr int kWideRowGroup = 64;                  // the row-group kernels
constexpr int kWideMax = kWideFamilies | kWideRowGroup;

// Each returns the setting's value (unset or empty: the default, which is the present behaviour) or -1 when the text is not
// a value of its range.
inline int norm_exact(const char* e) {
  if (!e || !*e) return 0;
  int v;
  return parse_uint(e, &v) && v <= 1 ? v : -1;
}
inline int wide_exact(const char* e) {
  if (!e || !*e) return 0;
  int v;
  return parse_uint(e, &v) && v <= kWideMax && !(v & 32) ? v : -1;
}
inline int wide_mt(const char* e) {
  if (!e || !*e) return 1;
  int v;
  return parse_uint(e, &v) && (v == 1 || v == 2) ? v : -1;
}
inline int wide_gq(const char* e) {
  if (!e || !*e) return 0;
  int v;
  return parse_uint(e, &v) && (v & ~16) <= 4 && ((v & 15) == 0 || (v & 15) == 2 || (v & 15) == 4) ? v : -1;
}
inline int r16_exact(const char* e) {
  if (!e || !*e) return 0;
  int v;
  return parse_uint(e, &v) && v <= 63 ? v : -1;
}

// The first setting of the five whose text is outside its range, as "NAME=text: what it takes" in msg; 0 when all are fine.
inline int env_error(const char* norm, const char* wide, const char* mt, const char* gq, const char* r16, char* msg,
                     size_t n) {
  if (norm_exact(norm) < 0) return std::snprintf(msg, n, "DSGB10_NORM_EXACT=%s: 0 or 1", norm), 1;
  if (wide_exact(wide) < 0)
    return std::snprintf(msg, n, "DSGB10_WIDE_EXACT=%s: 0..95 without bit 32 (1 2 4 8 16 families, 64 row groups, 95 all)", wide), 1;
  if (wide_mt(mt) < 0) return std::snprintf(msg, n, "DSGB10_WIDE_EXACT_MT=%s: 1 or 2", mt), 1;
  if (wide_gq(gq) < 0) return std::snprintf(msg, n, "DSGB10_WIDE_EXACT_GQ=%s: 0, 2 or 4, plus 16 for the raw scales", gq), 1;
  if (r16_exact(r16) < 0) return std::snprintf(msg, n, "DSGB10_R16_EXACT=%s: 0..63", r16), 1;
  return 0;
}
inline int env_error(char* msg, size_t n) {
  return env_error(std::getenv("DSGB10_NORM_EXACT"), std::getenv("DSGB10_WIDE_EXACT"), std::getenv("DSGB10_WIDE_EXACT_MT"),
                   std::getenv("DSGB10_WIDE_EXACT_GQ"), std::getenv("DSGB10_R16_EXACT"), msg, n);
}

}  // namespace model::exact
