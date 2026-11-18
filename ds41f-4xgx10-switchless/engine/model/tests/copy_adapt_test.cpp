// Copy drafting's learner (model/copy.h: outcome, the shared cells, a conversation's ratio, the gate's row), on the host.
// Holds: a proposal is scored from the verdict alone (the copy tokens that landed, and the first one the gate did not keep);
// before any outcome the gate reads the table itself, bit for bit, whatever the table holds; a cell follows its outcomes with
// the starting value worth 64 of them; a conversation whose copies fail does not take the others' rows away, and one that stopped
// failing gets its rows back; the ranks' state is the same bytes whatever order a batch step's threads ran in; the starting
// values stay inside their bounds for any table file; the learner's counts equal the ground truth of the simulated stream.
// Each property also runs under the wrong rule it exists to catch (COPY_ADAPT_TEST_RULES): those lines print "expected red".
#define COPY_ADAPT_TEST_RULES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model/copy.h"

using namespace model::copy;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33);
}
double urand() { return (rnd() & 0xFFFFFF) / 16777216.0; }

int fails = 0, checks = 0;
bool say(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++fails;
    std::printf("FAIL: %s\n", what);
  }
  return ok;
}

// A property under its right rules must hold; under the named wrong rule it must not.
struct Wrong {
  explicit Wrong(bool TestRules::*f) { rules().*f = true; }
  ~Wrong() { rules() = TestRules{}; }
};
void green(const char* name, bool (*prop)()) { say(prop(), name); }
// A property of a wrong way of doing it (no rule switch): it must not hold.
void red_plain(const char* name, bool (*wrong)()) {
  ++checks;
  if (wrong()) {
    ++fails;
    std::printf("FAIL: %s: the wrong way passed\n", name);
  } else {
    std::printf("expected red: %s\n", name);
  }
}
void red(const char* name, bool TestRules::*rule, bool (*prop)()) {
  Wrong w(rule);
  const bool holds = prop();
  ++checks;
  if (holds) {
    ++fails;
    std::printf("FAIL: %s: the wrong rule passed the property\n", name);
  } else {
    std::printf("expected red: %s\n", name);
  }
}

Pend mk(int pos, int k, int n, int m, int bucket, bool whole, std::initializer_list<int32_t> toks) {
  Pend p{};
  p.pos = pos, p.k = (int8_t)k, p.n = (int8_t)n, p.m = (int8_t)m, p.bucket = (int8_t)bucket, p.whole = whole ? 1 : 0;
  int j = 0;
  for (int32_t t : toks) p.tok[j++] = t;
  return p;
}

// ---- 1. outcome() by cases ----
bool outcomes_hold() {
  bool ok = true;
  auto is = [&](const Taken& t, int bucket, int landed, int fr, const char* what) {
    const bool good = t.bucket == bucket && (bucket < 0 || (t.landed == landed && t.fr == fr));
    if (!good) std::printf("  case %s: bucket %d landed %d fr %d (want %d %d %d)\n", what, t.bucket, t.landed, t.fr, bucket, landed, fr);
    ok = ok && good;
  };
  const Pend p = mk(100, 2, 1, 3, 1, true, {11, 12, 13});  // 2 chain drafts kept, 1 copy row verified, 3 proposed
  is(outcome(p, true, -1, 100, 2, 99), -1, 0, 0, "the chain's drafts did not all land");
  is(outcome(p, true, -1, 100, 3, 99), 1, 0, 0, "the verified row was refused: a miss at depth 0");
  is(outcome(p, true, -1, 100, 4, 12), 1, 1, 1, "the verified row landed; the first one not kept is the bonus");
  is(outcome(p, true, -1, 100, 4, 5), 1, 1, 0, "the verified row landed; the bonus differs from the next proposed token");
  const Pend q = mk(100, 0, 2, 3, 2, true, {11, 12, 13});
  is(outcome(q, true, -1, 100, 3, 13), 2, 2, 1, "two rows landed and the bonus is the third proposed token: hits plus a free hit");
  const Pend r = mk(100, 0, 3, 3, 0, true, {11, 12, 13});
  is(outcome(r, true, -1, 100, 4, 5), 0, 3, -1, "every proposed row verified and landed: no free outcome");
  is(outcome(r, true, -1, 100, 5, 5), -1, 0, 0, "more landed than were verified: dropped");
  // the first row the gate kept none of: the chain landed, and the bonus says whether the first copy token would have
  const Pend s = mk(100, 1, 0, 2, 3, true, {21, 22});
  is(outcome(s, true, -1, 100, 2, 21), 3, 0, 1, "no copy row kept, chain landed: the free outcome is a hit");
  is(outcome(s, true, -1, 100, 2, 20), 3, 0, 0, "no copy row kept, chain landed: the free outcome is a miss");
  // stale or doubtful: dropped, never guessed
  is(outcome(p, true, -1, 101, 4, 12), -1, 0, 0, "another position");
  is(outcome(p, false, -1, 100, 4, 12), -1, 0, 0, "a given step, not a verify step");
  is(outcome(p, true, 7, 100, 4, 12), -1, 0, 0, "a forced token");
  is(outcome(mk(100, 2, 1, 3, 1, false, {11, 12, 13}), true, -1, 100, 4, 12), -1, 0, 0, "a trim cut into the chain");
  is(outcome(mk(100, 0, 0, 0, 0, true, {}), true, -1, 100, 1, 12), -1, 0, 0, "no proposal");
  is(outcome(Pend{}, true, -1, 0, 1, 0), -1, 0, 0, "a cleared record");
  is(outcome(p, true, -1, 100, 0, 12), -1, 0, 0, "no rows emitted");
  return ok;
}

// ---- 2. no outcome, no change ----
Table random_table() {  // not geometric, not monotone: anything load_table accepts
  Table T{};
  for (auto& row : T.t)
    for (float& v : row) v = 0.05f + 0.95f * (float)urand();
  return T;
}
bool no_change_holds() {
  bool ok = true;
  std::vector<Table> tables{prior_table()};
  for (int i = 0; i < 200; ++i) tables.push_back(random_table());
  float step_ms[kMaxDrafts + 1];
  for (int i = 0; i <= kMaxDrafts; ++i) step_ms[i] = 20.f + 2.5f * (float)i;
  for (const Table& T : tables) {
    Learn L;
    learn_init(&L, T);
    const Sess s{};
    for (int b = 0; b < kBuckets; ++b)
      for (int m = 1; m <= kMaxDrafts; ++m) {
        float row[kMaxDrafts] = {};
        gate_row(L, s, b, T.t[b], m, row);
        if (std::memcmp(row, T.t[b], sizeof(float) * (size_t)m) != 0) ok = false;
        // the rows the gate keeps are the ones it keeps with the table's own values
        float lt[1][kMaxDrafts];
        std::memcpy(lt[0], row, sizeof row);
        for (int gate = 0; gate < 2; ++gate)
          for (int trial = 0; trial < 3; ++trial) {
            const float chain_prev = 0.3f + 0.7f * (float)urand(), thresh = 0.2f + 0.3f * (float)urand();
            const int k = (int)(rnd() % (unsigned)std::min(3, kMaxDrafts - m + 1));  // k + m rows stay within step_ms
            float a[kMaxDrafts] = {}, c[kMaxDrafts] = {};
            const int na = gate_rows(gate, thresh, step_ms, chain_prev, 1.f + chain_prev, k, m, b, T.t, false, a);
            const int nc = gate_rows(gate, thresh, step_ms, chain_prev, 1.f + chain_prev, k, m, 0, lt, false, c);
            if (na != nc || std::memcmp(a, c, sizeof(float) * (size_t)m) != 0) ok = false;
          }
      }
  }
  return ok;
}
// the formula itself (what the gate would read were the table not read directly), with no outcome, is the table's row too
bool no_change_formula_holds() {
  bool ok = true;
  for (int i = 0; i < 200; ++i) {
    const Table T = random_table();
    Learn L;
    learn_init(&L, T);
    const Sess s{};
    for (int b = 0; b < kBuckets; ++b) {
      float row[kMaxDrafts];
      learned_row(L, &s, b, T.t[b], kMaxDrafts, row);
      for (int j = 0; j < kMaxDrafts; ++j) {
        const float want = T.t[b][j] < 1.f ? T.t[b][j] : 1.f;
        if (std::memcmp(&row[j], &want, 4) != 0) ok = false;
      }
    }
  }
  return ok;
}

// ---- 3. the estimate ----
bool estimate_holds() {
  bool ok = true;
  for (double p : {0.5, 0.9, 0.97}) {
    Learn L;
    learn_init(&L, prior_table());
    const int N = 600;
    for (int i = 0; i < N; ++i) {
      Taken t;
      t.bucket = 2, t.landed = 0, t.fr = urand() < p ? 1 : 0;
      fold_all(L, &t, 1);
    }
    const double se = std::sqrt(p * (1 - p) / N) * 3 + 64.0 / (N + 64) * std::fabs(L.prior[2][0] - p);
    if (std::fabs(L.chance[2][0] - p) > se) {
      std::printf("  p %.2f: chance %.4f, bound %.4f\n", p, L.chance[2][0], se);
      ok = false;
    }
  }
  {  // halving keeps r <= n and the window
    Learn L;
    learn_init(&L, prior_table());
    for (int i = 0; i < 10000; ++i) {
      Taken t;
      t.bucket = 0, t.landed = 0, t.fr = urand() < 0.8 ? 1 : 0;
      fold_all(L, &t, 1);
      if (L.cell[0][0].r > L.cell[0][0].n || L.cell[0][0].n >= (uint32_t)kWindow) ok = false;
    }
  }
  {  // eight misses in a row move a first-token cell by 0.10 at most
    Learn L;
    learn_init(&L, prior_table());
    const float before = L.chance[0][0];
    for (int i = 0; i < 8; ++i) {
      Taken t;
      t.bucket = 0, t.landed = 0, t.fr = 0;
      fold_all(L, &t, 1);
    }
    if (before - L.chance[0][0] > 0.10f + 1e-6f) ok = false;
  }
  return ok;
}

// ---- 4 and 5. simulated conversations through the gate ----
struct Conv {
  Sess sess{};
  Pend pend{};
  int32_t vcount = 0, vbonus = 0;
  bool have = false;
  double p = 0.9, rate = 1.0;
  int bucket = 1;
  int last_n = 0;
};
struct Tally {
  long proposals = 0, verified = 0, right_kept = 0, right_potential = 0, wasted = 0;
};
// One gate run of a conversation: the last proposal's verdict first, then the next proposal, then the shared cells take the outcome.
// learn false: the fixed table (no outcome is taken, the table's rows are read).
void conv_step(Learn& L, Conv& c, const Table& T, float thresh, bool learn, Tally& ty) {
  Taken tk;
  if (learn && c.have) {
    tk = outcome(c.pend, true, -1, c.pend.pos, c.vcount, c.vbonus);
    sess_apply(c.sess, L, tk);
  }
  Pend pd{};
  c.have = false, c.last_n = 0;
  if (urand() < c.rate) {
    const int m = 3, b = c.bucket;
    int32_t tok[3] = {1000, 1001, 1002};
    float lt[1][kMaxDrafts], surv[kMaxDrafts], step_ms[kMaxDrafts + 1] = {};
    if (learn) gate_row(L, c.sess, b, T.t[b], m, lt[0]);
    else std::memcpy(lt[0], T.t[b], sizeof(float) * (size_t)m);
    const int n = gate_rows(0, thresh, step_ms, 1.f, 1.f, 0, m, 0, lt, false, surv);
    bool right[3];
    for (int j = 0; j < m; ++j) right[j] = urand() < c.p;
    int a = 0;
    while (a < n && right[a]) ++a;
    int pot = 0;
    while (pot < m && right[pot]) ++pot;
    ty.proposals++, ty.verified += n, ty.right_kept += a, ty.right_potential += pot, ty.wasted += n - a;
    pd = mk(4000, 0, n, m, b, true, {tok[0], tok[1], tok[2]});
    c.vcount = a + 1;
    c.vbonus = a < m ? (right[a] ? tok[a] : tok[a] + 1) : 7777;
    c.have = true, c.last_n = n;
  }
  c.pend = pd;
  if (learn) fold_all(L, &tk, 1);
}

bool no_dead_end_holds() {
  Learn L;
  const Table T = prior_table();
  learn_init(&L, T);
  Conv c;
  c.p = 0.15;
  Tally ty;
  for (int i = 0; i < 1500; ++i) conv_step(L, c, T, 0.45f, true, ty);
  if (c.last_n != 0) std::printf("  the failing stream still verifies %d rows\n", c.last_n);
  c.p = 0.95;
  int back = -1;
  for (int i = 0; i < 2000 && back < 0; ++i) {
    conv_step(L, c, T, 0.45f, true, ty);
    if (c.have && c.last_n == 3) back = i;
  }
  std::printf("  the rows came back after %d proposals\n", back);
  return back >= 0 && back <= 500;
}

bool no_starving_holds() {
  Learn L;
  const Table T = prior_table();
  learn_init(&L, T);
  Conv good, bad;
  good.p = 0.92, good.rate = 0.25;
  bad.p = 0.15, bad.rate = 0.75;
  Tally tg, tb, fg, fb;  // the learner's and the fixed table's
  for (int i = 0; i < 6000; ++i) {
    conv_step(L, good, T, 0.45f, true, tg);
    conv_step(L, bad, T, 0.45f, true, tb);
  }
  Learn F;
  learn_init(&F, T);
  Conv g2, b2;
  g2.p = good.p, g2.rate = good.rate, b2.p = bad.p, b2.rate = bad.rate;
  for (int i = 0; i < 6000; ++i) {
    conv_step(F, g2, T, 0.45f, false, fg);
    conv_step(F, b2, T, 0.45f, false, fb);
  }
  const double kept = (double)tg.right_kept / (double)std::max(1L, tg.right_potential);
  std::printf("  the good conversation keeps %.1f %% of its right copy tokens (fixed table %.1f %%); the failing one wastes %ld rows (fixed table %ld)\n",
              100 * kept, 100.0 * (double)fg.right_kept / (double)std::max(1L, fg.right_potential), tb.wasted, fb.wasted);
  return kept >= 0.98 && tb.wasted * 5 < fb.wasted;
}

// ---- 6. the ranks' state ----
bool order_holds_with(bool arrival_fold) {
  bool ok = true;
  const int kInst = 4;
  Learn L[kInst];
  for (auto& l : L) learn_init(&l, prior_table());
  Learn M;  // folded in another order than the entries'
  learn_init(&M, prior_table());
  for (int step = 0; step < 2000; ++step) {
    const int n = 1 + (int)(rnd() % 8);
    Taken t[8];
    for (int q = 0; q < n; ++q) {
      if (rnd() % 4 == 0) continue;  // an entry without an outcome
      t[q].bucket = (int8_t)(rnd() % kBuckets), t[q].landed = (int8_t)(rnd() % 4), t[q].fr = (int8_t)((int)(rnd() % 3) - 1);
    }
    for (auto& l : L) fold_all(l, t, n);
    if (arrival_fold) {  // the entries' outcomes taken in the order their threads finished
      int order[8];
      for (int q = 0; q < n; ++q) order[q] = q;
      for (int q = n - 1; q > 0; --q) std::swap(order[q], order[rnd() % (q + 1)]);
      uint32_t mask = 0;
      for (int q = 0; q < n; ++q) mask |= fold_one(M, t[order[q]]);
      for (int b = 0; b < kBuckets; ++b)
        if (mask >> b & 1u) rebuild(M, b);
    } else {
      fold_all(M, t, n);
    }
  }
  for (int i = 1; i < kInst; ++i)
    if (std::memcmp(&L[0], &L[i], sizeof(Learn)) != 0 || L[0].sum != L[i].sum) ok = false;
  if (std::memcmp(&L[0], &M, sizeof(Learn)) != 0 || L[0].sum != M.sum) ok = false;
  // a conversation's own counts do not depend on the shared fold's order: two copies given the same outcomes end alike
  Sess a{}, b{};
  Learn Z;
  learn_init(&Z, prior_table());
  for (int i = 0; i < 3000; ++i) {
    Taken t;
    t.bucket = (int8_t)(rnd() % kBuckets), t.landed = (int8_t)(rnd() % 3), t.fr = (int8_t)((int)(rnd() % 3) - 1);
    sess_apply(a, Z, t);
    sess_apply(b, Z, t);
  }
  if (std::memcmp(&a, &b, sizeof a) != 0 || a.n >= (uint32_t)kSessWindow) ok = false;
  return ok;
}
bool order_holds() { return order_holds_with(false); }
bool arrival_order_holds() { return order_holds_with(true); }

// ---- 7. any table file ----
bool bounds_hold() {
  bool ok = true;
  std::vector<Table> tables;
  Table up{};  // a table that rises (load_table accepts it) and one at 1.0
  for (int b = 0; b < kBuckets; ++b)
    for (int j = 0; j < kMaxDrafts; ++j) up.t[b][j] = std::min(1.f, 0.4f + 0.1f * (float)j), up.t[0][j] = 1.f;
  tables.push_back(up);
  for (int i = 0; i < 100; ++i) tables.push_back(random_table());
  for (const Table& T : tables) {
    Learn L;
    learn_init(&L, T);
    for (int b = 0; b < kBuckets; ++b)
      for (int c = 0; c < kDepthClasses; ++c)
        if (!(L.prior[b][c] >= kPriorLo && L.prior[b][c] <= kPriorHi)) ok = false;
    Sess s{};
    for (int i = 0; i < 400; ++i) {
      Taken t;
      t.bucket = (int8_t)(rnd() % kBuckets), t.landed = (int8_t)(rnd() % 3), t.fr = (int8_t)((int)(rnd() % 3) - 1);
      sess_apply(s, L, t);
      fold_all(L, &t, 1);
      float row[kMaxDrafts];
      gate_row(L, s, t.bucket < 0 ? 0 : t.bucket, T.t[0], kMaxDrafts, row);
      for (int j = 0; j < kMaxDrafts; ++j)
        if (!(row[j] > 0.f && row[j] <= 1.f)) ok = false;
      for (int b = 0; b < kBuckets; ++b)
        for (int c = 0; c < kDepthClasses; ++c)
          if (!(L.chance[b][c] > 0.f && L.chance[b][c] < 1.f)) ok = false;
    }
  }
  return ok;
}

// ---- 8. against the stream's own truth ----
// A proposal's chain and copy tokens each land or not by the stream; the verdict is what the engine's sampler would say. The learner's
// reading of it (outcome) must give, for every proposal whose chain landed, the first copy token right exactly when the stream says so,
// and the rows accepted as the verdict's count.
bool oracle_holds() {
  Learn L;
  learn_init(&L, prior_table());
  bool ok = true;
  long scored = 0, truth_scored = 0;
  for (int i = 0; i < 20000; ++i) {
    const int k = (int)(rnd() % 3), m = 1 + (int)(rnd() % 5), nk = (int)(rnd() % (m + 1));  // chain drafts, tokens proposed, copy rows kept
    const double p = i % 3 == 0 ? 0.95 : i % 3 == 1 ? 0.6 : 0.1;  // landing; a stream whose sample differs from the proposal is the last
    const bool chain_ok = urand() < 0.8;
    bool right[8];
    int32_t tok[8], samp[8];
    for (int j = 0; j < m; ++j) {
      tok[j] = 500 + j, right[j] = urand() < p, samp[j] = right[j] ? tok[j] : tok[j] + 1;
    }
    const Pend pd = mk(300, k, nk, m, (int)(rnd() % kBuckets), true, {tok[0], tok[1 % m], tok[2 % m]});
    Pend pp = pd;
    for (int j = 0; j < m; ++j) pp.tok[j] = tok[j];
    // the verdict: the chain lands or is refused at its last draft; then the verified copy rows land in order
    int accepted, a = 0;
    int32_t bonus;
    if (!chain_ok && k > 0) {
      accepted = (int)(rnd() % k), bonus = 4242;
    } else {
      while (a < nk && right[a]) ++a;
      accepted = k + a;
      bonus = a < m ? samp[a] : 4243;
    }
    const Taken t = outcome(pp, true, -1, 300, accepted + 1, bonus);
    if (!chain_ok && k > 0) {
      ok = ok && t.bucket < 0;  // built on drafts that did not land: no outcome
      continue;
    }
    ++truth_scored;
    if (t.bucket < 0) {
      ok = false;
      continue;
    }
    ++scored;
    const bool first_right = right[0];
    const bool read_right = t.landed >= 1 || t.fr == 1;
    if (first_right != read_right || t.landed != a) ok = false;
    // the second token is known only when the first was right and the gate kept it or the bonus tells it
    if (right[0] && m > 1 && nk >= 1 && a >= 1 && a < m && t.fr != (right[a] ? 1 : 0)) ok = false;
    fold_all(L, &t, 1);
  }
  std::printf("  %ld of %ld proposals whose chain landed were scored\n", scored, truth_scored);
  return ok && scored == truth_scored;
}

// ---- 9. the word the gate leaves for the host ----
bool obs_hold() {
  bool ok = true;
  for (int bk = -1; bk < kBuckets; ++bk)
    for (int landed = 0; landed < 8; ++landed)
      for (int fr = -1; fr <= 1; ++fr)
        for (int n = 0; n < 8; ++n)
          for (int prop = 0; prop < 2; ++prop) {
            Taken t;
            if (bk >= 0) t.bucket = (int8_t)bk, t.landed = (int8_t)landed, t.fr = (int8_t)fr, t.n = (int8_t)n;
            const uint32_t w = pack_obs(t, (n + 1) & 7, prop != 0, 0.75f);
            const Obs o = unpack_obs(w);
            if (o.taken != (bk >= 0) || o.proposed != (prop != 0) || o.kept != ((n + 1) & 7) || std::fabs(o.rho - 0.75f) > 1e-6f) ok = false;
            if (bk >= 0 && (o.bucket != bk || o.landed != landed || o.fr != fr || o.verified != n)) ok = false;
          }
  ok = ok && pack_obs(Taken{}, 0, false, 0.f) == 0;  // nothing to say is the word 0
  return ok;
}

// ---- the same results with the clock running between the gate runs ----
// Every gate run rewrites the record, so a step that is not a gate run (a prompt chunk, another request's rows) leaves it as it was and
// the next verify step still scores it.
bool gaps_do_not_lose_outcomes() {
  Learn L;
  learn_init(&L, prior_table());
  const Pend pd = mk(77, 0, 1, 2, 1, true, {5, 6});
  return outcome(pd, true, -1, 77, 2, 6).bucket == 1;  // the verdict arrives however many other steps ran between
}

// 11. The switches: the learner is on whenever copy drafting is (rows or the dump), DSGB10_COPY_ADAPT=0 turns it off, a relaxed
// accept (DSGB10_RELAX, DSGB10_RELAX_P) keeps it off unless it is asked for by name (then the start refuses: spec.cu).
bool switches_hold() {
  struct Case {
    const char *c, *d, *a, *r, *rp;
    int32_t want;
  } cases[] = {
      {"1", nullptr, nullptr, nullptr, nullptr, kRows | kAdapt},       // copy drafting on: the learner with it
      {"1", nullptr, "0", nullptr, nullptr, kRows},                     // turned off by name
      {"1", nullptr, "1", nullptr, nullptr, kRows | kAdapt},           // asked for by name
      {nullptr, nullptr, nullptr, nullptr, nullptr, 0},                 // nothing set: nothing on
      {nullptr, nullptr, "1", nullptr, nullptr, 0},                     // the learner alone is nothing
      {"0", nullptr, nullptr, nullptr, nullptr, 0},                     // copy drafting off
      {nullptr, "1", nullptr, nullptr, nullptr, kDump | kAdapt},       // the dump: the learner with it
      {nullptr, "2", nullptr, nullptr, nullptr, kDump | kRows | kAll | kAdapt},
      {"1", nullptr, nullptr, "0.5", nullptr, kRows},                   // a relaxed accept: off unless asked for
      {"1", nullptr, nullptr, nullptr, "0.1", kRows},
      {"1", nullptr, nullptr, "0", "0", kRows | kAdapt},                // relax set to 0 is no relax
      {"1", nullptr, "1", "0.5", nullptr, kRows | kAdapt},             // asked for by name beside relax: kept, the start refuses
      {"1", nullptr, "", nullptr, nullptr, kRows | kAdapt},            // set empty is unset
  };
  bool ok = true;
  for (const Case& k : cases) {
    const int32_t got = mode_from(k.c, k.d, k.a, k.r, k.rp);
    if (got != k.want) {
      std::printf("  mode_from(%s, %s, %s, %s, %s) = %d, want %d\n", k.c ? k.c : "-", k.d ? k.d : "-", k.a ? k.a : "-",
                  k.r ? k.r : "-", k.rp ? k.rp : "-", got, k.want);
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  green("1 outcome() by cases", outcomes_hold);
  red("1 outcome() by cases: only verified rows scored", &TestRules::verified_only, outcomes_hold);
  green("2 no outcome, no change: the gate's row is the table's, bit for bit, for any table", no_change_holds);
  red("2 no outcome, no change: the learned formula read before any outcome", &TestRules::no_pristine, [] {
    Wrong w2(&TestRules::no_n0);
    return no_change_holds();
  });
  green("2b the learned formula with no outcome is the table's row", no_change_formula_holds);
  red("2b the learned formula with no outcome: a conversation's formula without the empty case", &TestRules::no_n0, no_change_formula_holds);
  green("3 the estimate", estimate_holds);
  red("3 the estimate: no weight on the starting value", &TestRules::prior0, estimate_holds);
  green("4 no dead end", no_dead_end_holds);
  red("4 no dead end: only verified rows scored", &TestRules::verified_only, no_dead_end_holds);
  green("5 one conversation cannot starve another", no_starving_holds);
  red("5 one conversation cannot starve another: the shared cells alone", &TestRules::no_ratio, no_starving_holds);
  green("6 four instances and the entries' order give one state", order_holds);
  red_plain("6 four instances and the entries' order: folded in arrival order", arrival_order_holds);
  green("7 starting values stay inside their bounds for any table", bounds_hold);
  red("7 starting values: no clamp", &TestRules::no_clamp, bounds_hold);
  green("8 the learner's counts equal the stream's truth", oracle_holds);
  red("8 the learner's counts equal the stream's truth: only verified rows scored", &TestRules::verified_only, oracle_holds);
  green("9 the observation word", obs_hold);
  green("10 gaps between gate runs lose no outcome", gaps_do_not_lose_outcomes);
  green("11 the learner is on whenever copy drafting is", switches_hold);
  std::printf("copy_adapt_test: %d checks, %d failures\n", checks, fails);
  return fails ? 1 : 0;
}
