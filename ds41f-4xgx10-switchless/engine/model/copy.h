// Copy drafting (docs/batching.md, "Copy drafting"): a second source of drafts. The tokens that followed the latest
// earlier occurrence of the conversation's last kOrder tokens are appended to DSpark's drafts as extra rows of the same
// step; the target verifies them as it verifies the chain's, so only which drafts are verified changes, never a token.
// Host and device share these functions (CPY_HD); a hit is compared with the tail, so a collision is a miss, never a wrong copy.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#define CPY_HD __host__ __device__
#else
#define CPY_HD
#endif

namespace model::copy {

constexpr int kOrder = 8;         // tokens that key the table, and the shortest match
constexpr int kMaxDrafts = 7;     // drafts of one session in a step, at most: its rows (kMaxSessRows = 8) less the bonus
constexpr int kBack = 64;         // tokens a match extends backward past kOrder, at most (its length, for the gate)
constexpr int kBuckets = 4;       // match length buckets: 8-15, 16-31, 32-63, 64+
constexpr int kVocabIds = 129280;  // token ids are below this (config.h kVocab); a position of an image holds -1
constexpr int kMaxUnitsCopy = 128;  // config.h kMaxUnits

// The mode (the same on every rank: environment variables of the start). Every switch is off unless set, but the learner.
//  DSGB10_COPY=1:       copy rows on (verified with the chain's drafts, by the gate)
//  DSGB10_COPY_MIN=N:   the least match length (kOrder..kOrder + kBack) a proposal needs; default kOrder
//  DSGB10_COPY_ADAPT:   the gate reads chances learned from the outcomes of its own proposals (the learner below): on whenever
//                       DSGB10_COPY is; 0 turns it off; a relaxed accept (DSGB10_RELAX, DSGB10_RELAX_P) keeps it off
//                       unless it is 1 (then the start refuses: the relaxed accept would make its outcomes read low)
enum : int32_t { kRows = 1, kAdapt = 8 };
// The mode from the switches' values (null: unset); mode() reads them from the environment once.
inline int32_t mode_from(const char* c, const char* a, const char* relax, const char* relax_p) {
  int32_t v = 0;
  if (c && *c && atoi(c) > 0) v |= kRows;
  const bool relaxed = (relax && *relax && atof(relax) > 0.) || (relax_p && *relax_p && atof(relax_p) > 0.);
  const bool asked = a && *a && atoi(a) > 0, off = a && *a && atoi(a) <= 0;
  if (v && !off && (asked || !relaxed)) v |= kAdapt;
  return v;
}
inline int32_t mode() {
  static const int32_t m = mode_from(getenv("DSGB10_COPY"), getenv("DSGB10_COPY_ADAPT"), getenv("DSGB10_RELAX"), getenv("DSGB10_RELAX_P"));
  return m;
}
inline int min_match() {
  static const int n = [] {
    const char* e = getenv("DSGB10_COPY_MIN");
    const int v = e && *e ? atoi(e) : kOrder;
    return v < kOrder ? kOrder : v > kOrder + kBack ? kOrder + kBack : v;
  }();
  return n;
}
// Long copy steps (copy_run.h): DSGB10_COPY_ROWS = 8 (unset: the steps of before) | 32 | 64, DSGB10_COPY_ENTER = the match length
// (kOrder..kOrder + kBack, default 32) a cursor needs before its first long step. A text outside the range is -1.
constexpr int kRowsDefault = 8, kEnterDefault = 32, kEnterMax = kOrder + kBack;
inline int rows_from(const char* e) {
  if (!e || !*e) return kRowsDefault;
  if (e[0] == '8' && !e[1]) return 8;
  if (e[0] == '3' && e[1] == '2' && !e[2]) return 32;
  if (e[0] == '6' && e[1] == '4' && !e[2]) return 64;
  return -1;
}
inline int enter_from(const char* e) {
  if (!e || !*e) return kEnterDefault;
  long x = 0;
  for (const char* p = e; *p; ++p) {
    if (*p < '0' || *p > '9') return -1;
    x = x * 10 + (*p - '0');
    if (x > 100000) return -1;
  }
  return x >= kOrder && x <= kEnterMax ? (int)x : -1;
}
// The first of the two settings outside its range, as "NAME=text: what it takes" in msg; 0 when both are fine.
inline int run_env_error(const char* rows, const char* enter, char* msg, size_t n) {
  if (rows_from(rows) < 0) return std::snprintf(msg, n, "DSGB10_COPY_ROWS=%s: 8, 32 or 64", rows), 1;
  if (enter_from(enter) < 0) return std::snprintf(msg, n, "DSGB10_COPY_ENTER=%s: a match length %d..%d", enter, kOrder, kEnterMax), 1;
  return 0;
}
inline int run_env_error(char* msg, size_t n) { return run_env_error(getenv("DSGB10_COPY_ROWS"), getenv("DSGB10_COPY_ENTER"), msg, n); }
// The values the engine runs with (main.cpp has refused a bad text by the time these are read).
inline int copy_rows() {
  static const int v = rows_from(getenv("DSGB10_COPY_ROWS")) > 0 ? rows_from(getenv("DSGB10_COPY_ROWS")) : kRowsDefault;
  return v;
}
inline int copy_enter() {
  static const int v = enter_from(getenv("DSGB10_COPY_ENTER")) > 0 ? enter_from(getenv("DSGB10_COPY_ENTER")) : kEnterDefault;
  return v;
}
// Long steps' bits in StepIn::copy, beside the mode bits: the ladder's top (0 / 1 / 2) in bits 8-9, the entry length in bits 16-23.
// 0 (DSGB10_COPY_ROWS unset or 8): no long step, the step as without them.
inline int32_t long_bits() {
  const int top = copy_rows() >= 64 ? 2 : copy_rows() >= 32 ? 1 : 0;
  return top ? (int32_t)(top << 8 | copy_enter() << 16) : 0;
}
CPY_HD inline int long_top(int32_t copy_word) { return (copy_word >> 8) & 3; }
CPY_HD inline int long_enter(int32_t copy_word) { return (copy_word >> 16) & 255; }

// The drafts a step's rows can reach past the previous step's last row, for the host's mapped-memory bounds (cmd.cpp:
// verify(pos + 1 + reach)): the chain's kDrafts, or kMaxDrafts when copy rows are on.
// With long copy steps (DSGB10_COPY_ROWS above 8, copy_run.h) a step of one session reaches 63 rows past the previous step's last.
CPY_HD inline int verify_reach(int chain_drafts, bool rows_on, int long_rows = 8) {
  return rows_on ? (long_rows > 8 ? 63 : kMaxDrafts) : chain_drafts;
}

CPY_HD inline bool valid_token(int32_t t) { return t >= 0 && t < kVocabIds; }

CPY_HD inline uint32_t hash8(const int32_t* t) {
  uint32_t h = 0x9E3779B1u;
  for (int i = 0; i < kOrder; ++i) {
    h ^= (uint32_t)t[i] + 0x7F4A7C15u + (h << 6) + (h >> 2);
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
  }
  h ^= h >> 16;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 15);
}

// One session unit's view. tok: the history, valid below len (and below cap); tab: 2^k entries, mask = 2^k - 1.
struct Unit {
  int32_t* tok;
  uint32_t* tab;
  uint32_t mask;
  int32_t cap;  // positions tok holds
  int32_t len;  // positions known: [0, len)
};

// The slot takes the larger end: the latest occurrence wins, and parallel writers agree on the winner.
CPY_HD inline void slot_put(uint32_t* p, uint32_t e) {
#ifdef __CUDA_ARCH__
  atomicMax(p, e);
#else
  if (*p < e) *p = e;
#endif
}

// Index the grams that end at positions [from, to] (clipped to what the history holds). Every one of the kOrder
// tokens must be a real token id, or the gram is not indexed.
CPY_HD inline void index_one(const Unit& u, int e) {
  if (e < kOrder - 1 || e >= u.len || e >= u.cap) return;
  const int32_t* g = u.tok + e - (kOrder - 1);
  for (int i = 0; i < kOrder; ++i)
    if (!valid_token(g[i])) return;
  slot_put(u.tab + (hash8(g) & u.mask), (uint32_t)e);
}
CPY_HD inline void index_range(const Unit& u, int from, int to) {
  for (int e = from < kOrder - 1 ? kOrder - 1 : from; e <= to; ++e) index_one(u, e);
}

struct Match {
  int e = -1;      // the end of the source occurrence (< 0: none)
  int len = 0;     // match length: kOrder plus the tokens that also agree before it
  int bucket = 0;  // len's bucket
};
CPY_HD inline int bucket_of(int len) { return len < 16 ? 0 : len < 32 ? 1 : len < 64 ? 2 : 3; }

// The history up to P (the position of the bonus token) followed by the chain's k kept drafts: the string whose last
// kOrder tokens the lookup keys on. Position q <= P is the unit's, past it the draft q - P.
CPY_HD inline int32_t tail_at(const Unit& u, int P, const int32_t* d, int q) { return q <= P ? u.tok[q] : d[q - P - 1]; }

// Latest earlier occurrence of the last kOrder tokens of (history[0..P] ++ d[0..k)): the table's entry, compared.
// P < len is required (the bonus is in the history). A source must end before P, so that at least one token follows it.
CPY_HD inline Match find(const Unit& u, int P, const int32_t* d, int k) {
  Match m;
  const int Q = P + k;
  if (P < 0 || P >= u.len || P >= u.cap || Q < kOrder - 1) return m;
  int32_t t[kOrder];
  for (int i = 0; i < kOrder; ++i) {
    t[i] = tail_at(u, P, d, Q - (kOrder - 1) + i);
    if (!valid_token(t[i])) return m;
  }
  const int e = (int)u.tab[hash8(t) & u.mask];
  if (e < kOrder - 1 || e >= P) return m;
  for (int i = 0; i < kOrder; ++i)
    if (u.tok[e - (kOrder - 1) + i] != t[i]) return m;
  int len = kOrder;
  for (int j = 0; j < kBack; ++j) {
    const int a = e - kOrder - j, b = Q - kOrder - j;
    if (a < 0 || b < 0) break;
    const int32_t x = tail_at(u, P, d, b);
    if (!valid_token(x) || u.tok[a] != x) break;
    ++len;
  }
  m.e = e, m.len = len, m.bucket = bucket_of(len);
  return m;
}

// The proposal: the tokens after the source, at most maxm, never past P (the known history) and ending at the first
// token that is not a real id. Returns how many (0: none, or the match is shorter than minlen).
CPY_HD inline int propose(const Unit& u, int P, const int32_t* d, int k, int maxm, int minlen, int32_t* out,
                          Match* mt) {
  const Match m = find(u, P, d, k);
  *mt = m;
  if (m.e < 0 || m.len < minlen) return 0;
  int n = P - m.e;
  if (n > maxm) n = maxm;
  if (n > kMaxDrafts) n = kMaxDrafts;
  int c = 0;
  for (; c < n; ++c) {
    const int32_t t = u.tok[m.e + 1 + c];
    if (!valid_token(t)) break;
    out[c] = t;
  }
  return c;
}

// ---- the gate ----------------------------------------------------------------------------------------------
// T[b][j]: the chance that copy tokens 1..j+1 are all accepted, given match bucket b and that the chain's kept drafts all
// were; row j + 1 has survival chain_prev * T[b][j] (chain_prev: the chain's last kept survival, 1 for none), non-increasing.
// PRIOR: a starting table; a fitted one comes through DSGB10_COPY_TABLE.
struct Table {
  float t[kBuckets][kMaxDrafts];
};
inline Table prior_table() {
  Table T{};
  const float q[kBuckets] = {0.90f, 0.93f, 0.96f, 0.93f}, c[kBuckets] = {0.91f, 0.93f, 0.95f, 0.94f};
  for (int b = 0; b < kBuckets; ++b) {
    float p = q[b];
    for (int j = 0; j < kMaxDrafts; ++j) T.t[b][j] = p, p *= c[b];
  }
  return T;
}

// The gate's table from a file (DSGB10_COPY_TABLE): kBuckets lines of kMaxDrafts numbers in (0, 1], T[bucket][row]
// False when the file is missing, short or out of range.
inline bool load_table(const char* path, Table* T) {
  FILE* f = std::fopen(path, "r");
  if (!f) return false;
  Table t{};
  bool ok = true;
  for (int b = 0; b < kBuckets && ok; ++b)
    for (int j = 0; j < kMaxDrafts && ok; ++j) {
      float v;
      ok = std::fscanf(f, "%f", &v) == 1 && v > 0.f && v <= 1.f;
      t.t[b][j] = v;
    }
  std::fclose(f);
  if (ok) *T = t;
  return ok;
}

// How many of the m proposed copy rows are verified after the chain's k kept drafts. surv_out[j]: row k + j + 1's
// survival. e_chain: 1 + the chain's kept survivals (the expected tokens of a step with k drafts). step_ms: the modelled
// ms of a step verifying n drafts, n = 0..kMaxDrafts (ops.h kGate*: thresh, ratio, marginal).
CPY_HD inline int gate_rows(int gate, float thresh, const float* step_ms, float chain_prev, float e_chain, int k, int m,
                            int bucket, const float (*T)[kMaxDrafts], float* surv_out) {
  float prev = chain_prev;
  for (int j = 0; j < m; ++j) {
    float s = chain_prev * T[bucket][j];
    if (s > prev) s = prev;
    surv_out[j] = s, prev = s;
  }
  int n = 0;
  if (gate == 1) {  // kGateRatio
    float e = e_chain, best = e / step_ms[k];
    for (int j = 1; j <= m; ++j) {
      e += surv_out[j - 1];
      const float r = e / step_ms[k + j];
      if (r > best) best = r, n = j;
    }
  } else if (gate == 2) {  // kGateMarginal
    while (n < m && surv_out[n] >= thresh * (step_ms[k + n + 1] - step_ms[k + n])) ++n;
  } else {
    while (n < m && surv_out[n] >= thresh) ++n;
  }
  return n;
}

// ---- the learner: the table that follows the traffic ----------------------------------------------------------------------
// One step after the gate proposes copy rows, the device holds what came of them (the verdict in SpecState), so the table can
// be counts that follow the traffic, the same on every rank with nothing sent. A proposal is scored by outcome(): the copy
// tokens that landed, and the first one the gate did not keep, which is right exactly when it equals the sampler's token after
// the accepted rows (the bonus). That free outcome keeps a cell that has fallen low receiving data, so it can climb back.
// Two levels: shared cells by match bucket and depth class (slow), and a conversation's ratio of its misses to what the shared
// cells expected of the same outcomes (quick). All state is integers; the floats are derived from them in a fixed order, each
// operation rounded alone (the device's _rn intrinsics, the host's separate statements), so host and device agree to the bit.
constexpr int kDepthClasses = 3;    // the first copy token, the second, the third and later
constexpr int kPriorWeight = 64;    // the starting value counts as this many outcomes
constexpr int kWindow = 1024;       // a cell halves here: it holds its last 512 to 1,024 outcomes
constexpr int kSessPrior = 4;       // a conversation's ratio starts as 4 misses expected and 4 seen
constexpr int kSessWindow = 256;    // a conversation's counts halve here
constexpr float kRatioLo = 0.25f, kRatioHi = 8.f, kHeldMax = 0.98f;  // the ratio's bounds; the most a conversation's token is lowered to miss
constexpr float kPriorLo = 0.02f, kPriorHi = 0.999f;                  // a starting value stays inside these, whatever the file holds

#ifdef COPY_ADAPT_TEST_RULES  // tests/copy_adapt_test.cpp: the wrong rules its controls expect to fail; no engine build defines it
struct TestRules {
  bool verified_only = false;  // score only rows that were verified
  bool no_n0 = false;          // the conversation's formula even with no outcome
  bool prior0 = false;         // no weight on the starting value
  bool no_ratio = false;       // the shared cells alone
  bool no_clamp = false;       // the starting values as the table gives them
  bool no_pristine = false;    // the learned formula even before any outcome
};
inline TestRules& rules() {
  static TestRules r;
  return r;
}
#define CPY_RULE(x) (::model::copy::rules().x)
#else
#define CPY_RULE(x) false
#endif

// One rounded operation each. On the device the intrinsics (no contraction into an fma); on the host a call the compiler keeps apart.
#ifdef __CUDA_ARCH__
CPY_HD inline float fmul_rn(float a, float b) { return __fmul_rn(a, b); }
CPY_HD inline float fadd_rn(float a, float b) { return __fadd_rn(a, b); }
CPY_HD inline float fsub_rn(float a, float b) { return __fsub_rn(a, b); }
CPY_HD inline float fdiv_rn(float a, float b) { return __fdiv_rn(a, b); }
CPY_HD inline uint32_t round_u32(float x) { return __float2uint_rn(x); }
#else
#define CPY_ALONE __attribute__((noinline))
CPY_ALONE CPY_HD inline float fmul_rn(float a, float b) { return a * b; }
CPY_ALONE CPY_HD inline float fadd_rn(float a, float b) { return a + b; }
CPY_ALONE CPY_HD inline float fsub_rn(float a, float b) { return a - b; }
CPY_ALONE CPY_HD inline float fdiv_rn(float a, float b) { return a / b; }
// round to nearest, ties to even, for 0 <= x < 2^23 (the device's __float2uint_rn on that range)
CPY_HD inline uint32_t round_u32(float x) {
  volatile float t = x + 8388608.f;
  t = t - 8388608.f;
  return (uint32_t)t;
}
#endif

struct Cell {
  uint32_t n, r;  // outcomes, and of them right
};
struct Sess {
  uint32_t n, miss, exp16;  // a conversation's outcomes, its misses, and the misses the shared cells expected of the same ones
                            // (the sum of (1 - chance) x 65536)
};
// A unit's proposal, kept until the next verify step's verdict. Every gate run for the unit writes it whole, so it always names the
// latest run's: pos the bonus's position; k the chain's kept drafts before the copy rows; n the copy rows kept; m the tokens
// proposed (0: none, and the record is void); whole: the trims left all k.
struct Pend {
  int32_t pos;
  int8_t k, n, m, bucket, whole, pad[3];
  int32_t tok[kMaxDrafts];
};
struct Learn {
  float prior[kBuckets][kDepthClasses];   // from the table, by division only
  float chance[kBuckets][kDepthClasses];  // what the gate reads
  Cell cell[kBuckets][kDepthClasses];
};
struct UnitLearn {
  Sess sess;
  Pend pend;
};

CPY_HD inline int depth_class(int j) { return j < kDepthClasses - 1 ? j : kDepthClasses - 1; }

// The starting values: the table's first token, then each token's chance given the ones before it, all kept inside
// [kPriorLo, kPriorHi] (a table file may hold any values in (0, 1], not only falling ones).
CPY_HD inline float clamp_prior(float v) { return CPY_RULE(no_clamp) ? v : v < kPriorLo ? kPriorLo : v > kPriorHi ? kPriorHi : v; }
CPY_HD inline void learn_init(Learn* L, const Table& T) {
  for (int b = 0; b < kBuckets; ++b) {
    L->prior[b][0] = clamp_prior(T.t[b][0]);
    for (int c = 1; c < kDepthClasses; ++c) L->prior[b][c] = clamp_prior(fdiv_rn(T.t[b][c], T.t[b][c - 1]));
    for (int c = 0; c < kDepthClasses; ++c) L->chance[b][c] = L->prior[b][c], L->cell[b][c] = Cell{0, 0};
  }
}
// No outcome has been taken yet, by the shared cells or by the conversation: the gate reads the table itself.
CPY_HD inline bool pristine(const Learn& L, const Sess& s) {
  if (CPY_RULE(no_pristine)) return false;
  if (s.n) return false;
  for (int b = 0; b < kBuckets; ++b)
    for (int c = 0; c < kDepthClasses; ++c)
      if (L.cell[b][c].n) return false;
  return true;
}

// One scored proposal: the bucket, the copy tokens that landed, the free outcome (-1 none, 0 miss, 1 hit) after them, and the
// rows that were verified (for the counters).
struct Taken {
  int8_t bucket = -1, landed = 0, fr = -1, n = 0;
};

// What the step's verdict says of the proposal in p, or no outcome (bucket < 0): in doubt it is dropped, never guessed. The
// proposal is the latest gate run's (every run rewrites it), so a verify step whose first row is at its position is its verdict.
// verify: the step verified rows (not the host's given rows); count and bonus are the verdict (ss->count, ss->bonus), pos0 the
// position of its first row, force >= 0 when the bonus was forced.
CPY_HD inline Taken outcome(const Pend& p, bool verify, int32_t force, int pos0, int count, int32_t bonus) {
  Taken t;
  if (p.m <= 0 || !p.whole || !verify || pos0 != p.pos || force >= 0 || count < 1) return t;
  const int accepted = count - 1;
  if (accepted < p.k) return t;  // the chain's drafts did not all land: the proposal was built on them
  const int a = accepted - p.k;
  if (a > p.n) return t;
  t.bucket = p.bucket, t.landed = (int8_t)a, t.n = p.n;
  if (a < p.m && !(CPY_RULE(verified_only) && a >= p.n)) t.fr = bonus == p.tok[a] ? 1 : 0;
  return t;
}

// ---- the shared cells ----
CPY_HD inline void cell_take(Learn& L, int b, int j, bool hit) {
  Cell& c = L.cell[b][depth_class(j)];
  c.n += 1, c.r += hit ? 1u : 0u;
  if (c.n >= (uint32_t)kWindow) c.n >>= 1, c.r >>= 1;
}
CPY_HD inline void rebuild(Learn& L, int b) {
  const float w = CPY_RULE(prior0) ? 0.f : (float)kPriorWeight;
  for (int c = 0; c < kDepthClasses; ++c) {
    const float num = fadd_rn((float)L.cell[b][c].r, fmul_rn(w, L.prior[b][c]));
    L.chance[b][c] = fdiv_rn(num, fadd_rn((float)L.cell[b][c].n, w));
  }
}
// Takes one outcome into the cells (no rebuild); the bucket's bit, 0 when there is none.
CPY_HD inline uint32_t fold_one(Learn& L, const Taken& t) {
  if (t.bucket < 0) return 0;
  for (int j = 0; j < t.landed; ++j) cell_take(L, t.bucket, j, true);
  if (t.fr >= 0) cell_take(L, t.bucket, t.landed, t.fr == 1);
  return 1u << t.bucket;
}
// The outcomes of a step's entries, in entry order, then the chances of the buckets that changed.
CPY_HD inline void fold_all(Learn& L, const Taken* t, int n) {
  uint32_t mask = 0;
  for (int q = 0; q < n; ++q) mask |= fold_one(L, t[q]);
  for (int b = 0; b < kBuckets; ++b)
    if (mask >> b & 1u) rebuild(L, b);
}

// ---- a conversation's ratio ----
CPY_HD inline void sess_take(Sess& s, const Learn& L, int b, int j, bool hit) {
  s.n += 1, s.miss += hit ? 0u : 1u;
  s.exp16 += round_u32(fmul_rn(fsub_rn(1.f, L.chance[b][depth_class(j)]), 65536.f));
  if (s.n >= (uint32_t)kSessWindow) s.n >>= 1, s.miss >>= 1, s.exp16 >>= 1;
}
CPY_HD inline void sess_apply(Sess& s, const Learn& L, const Taken& t) {
  if (t.bucket < 0) return;
  for (int j = 0; j < t.landed; ++j) sess_take(s, L, t.bucket, j, true);
  if (t.fr >= 0) sess_take(s, L, t.bucket, t.landed, t.fr == 1);
}
// How many times as often as the average proposal of the same kind this conversation's copies miss.
CPY_HD inline float ratio(const Sess& s) {
  const float r = fdiv_rn(fmul_rn((float)(s.miss + kSessPrior), 65536.f), fadd_rn((float)s.exp16, (float)(kSessPrior * 65536)));
  return r < kRatioLo ? kRatioLo : r > kRatioHi ? kRatioHi : r;
}

// The chance that copy token class c is right, for a conversation (s may be null): the shared cell when it has no outcome yet.
CPY_HD inline float held(const Learn& L, const Sess* s, int b, int c) {
  const float ch = L.chance[b][c];
  if (!s || CPY_RULE(no_ratio) || (s->n == 0 && !CPY_RULE(no_n0))) return ch;
  const float x = fmul_rn(ratio(*s), fsub_rn(1.f, ch));
  return fsub_rn(1.f, x < kHeldMax ? x : kHeldMax);
}
// The survival row of a proposal of bucket b: t[j], the chance that copy tokens 1..j+1 are all accepted, for j < m. base is the
// table's row for the bucket: each entry is scaled by how far the learned chances stand from the starting ones, so a table whose
// later depths are not the first ones' ratios keeps them.
CPY_HD inline void learned_row(const Learn& L, const Sess* s, int b, const float* base, int m, float* t) {
  float R = 1.f;
  for (int j = 0; j < m; ++j) {
    const int c = depth_class(j);
    const float r = fdiv_rn(held(L, s, b, c), L.prior[b][c]);
    R = j == 0 ? r : fmul_rn(R, r);
    const float v = fmul_rn(base[j], R);
    t[j] = v < 1.f ? v : 1.f;
  }
}
// The row the gate reads: the table's own before any outcome (so the rows chosen are today's), the learned one after.
CPY_HD inline void gate_row(const Learn& L, const Sess& s, int b, const float* base, int m, float* t) {
  if (pristine(L, s)) {
    for (int j = 0; j < m; ++j) t[j] = base[j];
    return;
  }
  learned_row(L, &s, b, base, m, t);
}

// What a gate kernel records for the host, one word an entry: bit 0 an outcome was taken; 1-2 its bucket; 3-5 the copy rows that
// landed; 6-7 the free outcome (0 none, 1 miss, 2 hit); 8-10 the copy rows that were verified; 11-13 the copy rows kept for the
// next step; 14 the gate made a proposal; 16-31 the conversation's ratio in 1/256. A word of 0 is a step that had nothing to say.
CPY_HD inline uint32_t pack_obs(const Taken& t, int kept, bool proposed, float rho) {
  uint32_t w = (uint32_t)round_u32(fmul_rn(rho, 256.f)) << 16;
  w |= (uint32_t)(kept & 7) << 11 | (proposed ? 1u << 14 : 0u);
  if (t.bucket >= 0)
    w |= 1u | (uint32_t)(t.bucket & 3) << 1 | (uint32_t)(t.landed & 7) << 3 | (uint32_t)(t.fr + 1) << 6 | (uint32_t)(t.n & 7) << 8;
  return w;
}
struct Obs {
  bool taken, proposed;
  int bucket, landed, fr, verified, kept;
  float rho;
};
CPY_HD inline Obs unpack_obs(uint32_t w) {
  Obs o;
  o.taken = (w & 1u) != 0, o.proposed = (w >> 14 & 1u) != 0;
  o.bucket = (int)(w >> 1 & 3u), o.landed = (int)(w >> 3 & 7u), o.fr = (int)(w >> 6 & 3u) - 1;
  o.verified = (int)(w >> 8 & 7u), o.kept = (int)(w >> 11 & 7u), o.rho = (float)(w >> 16) / 256.f;
  return o;
}

// The settings every rank must hold alike (compared at start-up on the host all-gather): the mode bits, the shortest match, the long steps' rows and entry length and the
// table file's bytes (0 without one). A rank that differs would verify other rows than the others and stop at the next collective.
struct Settings {
  uint32_t mode, min_match, rows, enter;
  uint64_t table_hash;
};
inline Settings settings() {
  Settings s{(uint32_t)mode(), (uint32_t)min_match(), (uint32_t)copy_rows(), (uint32_t)copy_enter(), 0};
  const char* f = getenv("DSGB10_COPY_TABLE");
  if (f && *f) {
    uint64_t h = 1469598103934665603ull;
    if (FILE* fp = std::fopen(f, "rb")) {
      for (int c; (c = std::fgetc(fp)) != EOF;) h = (h ^ (uint64_t)(uint8_t)c) * 1099511628211ull;
      std::fclose(fp);
    } else {
      h = 1;  // unreadable: the start stops on this rank anyway
    }
    s.table_hash = h;
  }
  return s;
}

// ---- the unit's history ------------------------------------------------------------------------------------
// What a spec step appends to the unit's history: the rows' tokens at pos0 .. pos0 + count - 1 and the bonus at pos0 +
// count; the grams ending in [pos0, pos0 + count - 1] are indexed (the bonus's gram waits for the next step, so a lookup
// of the bare tail finds the occurrence before it). Nothing is appended when the history has a hole before pos0.
CPY_HD inline void append_step(Unit& u, int pos0, int count, const int32_t* rows, int32_t bonus) {
  if (pos0 < 0 || count < 1 || pos0 > u.len || pos0 + count >= u.cap) return;
  for (int r = 0; r < count; ++r) u.tok[pos0 + r] = rows[r];
  u.tok[pos0 + count] = bonus;
  u.len = pos0 + count + 1;
  index_range(u, pos0, pos0 + count - 1);
}

// ---- memory ------------------------------------------------------------------------------------------------
// The arena (Options::copy_mib a rank) is shared by the units: a unit of ctx positions takes a mirror of cap tokens and
// a table of 2^tab_log2 entries (4 bytes each), from 4 x ctx entries down to ctx / 4 while the arena is short (older
// entries are displaced: a missed copy, never a wrong one), then the mirrors shrink; a unit that cannot fit gets none.
struct Plan {
  int cap;       // positions the mirror holds (0: the unit has no copy drafting)
  int tab_log2;  // table entries: 1 << tab_log2
};
constexpr size_t kBlockAlign = 256;
inline size_t unit_bytes(const Plan& p) {
  auto up = [](size_t b) { return (b + kBlockAlign - 1) / kBlockAlign * kBlockAlign; };
  return p.cap > 0 ? up((size_t)p.cap * 4) + up(((size_t)1 << p.tab_log2) * 4) : 0;
}
inline int log2_ceil(long long v) {
  int l = 4;
  while ((1ll << l) < v) ++l;
  return l;
}
inline size_t plan_units(const int* ctx, int n, size_t arena_bytes, Plan* out) {
  auto total = [&] {
    size_t t = 0;
    for (int i = 0; i < n; ++i) t += unit_bytes(out[i]);
    return t;
  };
  for (int ml = 2; ml >= -2; --ml) {  // table entries = ctx * 2^ml, rounded up to a power of two
    for (int i = 0; i < n; ++i)
      out[i] = ctx[i] > 0 ? Plan{ctx[i], log2_ceil(ml >= 0 ? (long long)ctx[i] << ml : (long long)ctx[i] >> -ml)} : Plan{0, 0};
    if (total() <= arena_bytes) return total();
  }
  for (int s = 1; s < 24; ++s) {  // mirrors shrink too
    for (int i = 0; i < n; ++i) {
      const int c = ctx[i] >> s;
      out[i] = c >= 4 * kOrder ? Plan{c, log2_ceil(c >> 1)} : Plan{0, 0};
    }
    if (total() <= arena_bytes) return total();
  }
  for (int i = 0; i < n; ++i) out[i] = Plan{0, 0};
  return 0;
}

}  // namespace model::copy
