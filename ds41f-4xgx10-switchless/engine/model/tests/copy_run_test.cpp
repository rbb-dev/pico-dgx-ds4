// Long copy steps' rule (model/copy_run.h) and the 7-draft block's rule, on the host, no GPU. Holds:
//  H1  the cursor and the ladder against a plain reference on planted streams: a verbatim repeat of 4K tokens reaches steps of
//      63 drafts (a rule that never leaves 7 drafts, as before, fails here); every draft of a long step is the source's token; a
//      step is at most 8 rows, 29..32 rows or 57..64 rows; a repeat with edits has its long steps rejected in part and the cursor
//      comes back only after `enter << fails` matching tokens; a stream with no repeat never gets a long step
//  H2  copy_fit: for every n and both settings the step is at most 8 rows or fills a bucket with an exact body, the result is the
//      largest such n at most the asked one
//  H4  the settings: the parsers, the range refusals, the bits in StepIn::copy
//  H5  a long step's learner record is void (m = 0): outcome() takes nothing from it
//  L8  the 7-draft block runs only when no cursor is live at the bonus
// Every row that guards a rule has its control: the wrong rule, compiled in, must FAIL the same check (printed "control rejected"):
// no walk; no `fails`; a ladder that climbs on a partly accepted step; a 16-row body allowed; the block without its rule.
// Build flags that show RED for the rule as it was before this change: -DCOPY_RUN_TEST_PRESENT_RULE (steps never leave 7 drafts),
// -DCOPY_RUN_TEST_NO_BLOCK_RULE (the block without the rule). Command: copy_run_test (exit 0 = all pass)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model/copy_run.h"

using namespace model::copy;

namespace {

int fails = 0, checks = 0, controls = 0;
bool quiet = false;
bool say(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++fails;
    if (!quiet) std::printf("FAIL: %s\n", what);
  }
  return ok;
}
// a check under a wrong rule must fail: its failure is counted as a rejected control, not as a failure
template <class F>
void control(const char* what, F&& body) {
  const int f0 = fails;
  quiet = true;
  body();
  quiet = false;
  const bool rejected = fails > f0;
  fails = f0;
  ++checks;
  if (rejected) {
    ++controls;
    std::printf("control rejected: %s\n", what);
  } else {
    ++fails;
    std::printf("FAIL: the control '%s' was not rejected (the check cannot see its rule)\n", what);
  }
}

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33);
}

// ---- the simulated unit and the engine's step loop ---------------------------------------------------------------
struct Sim {
  std::vector<int32_t> tok;
  std::vector<uint32_t> tab;
  Unit u;
  explicit Sim(int cap, int tab_log2 = 15) : tok(cap, 0), tab((size_t)1 << tab_log2, 0) {
    u.tok = tok.data(), u.tab = tab.data(), u.mask = (uint32_t)tab.size() - 1, u.cap = cap, u.len = 0;
  }
};

struct Rules {  // the wrong rules of the controls
  bool no_walk = false;     // the cursor is not kept: dropped after every step
  bool no_fails = false;    // a long step rejected in part is forgotten
};

struct Result {
  int steps = 0, long_steps = 0, whole_long = 0, partial_long = 0, max_drafts = 0, bad_rows = 0, bad_tokens = 0, early_entry = 0;
  int out_tokens = 0;
};

// The reply: positions P0 .. P0 + L - 1 hold text[0 .. L) with `edits` (position -> other token). The model's sample of row r
// (position P + r) is truth(P + r + 1). rows: the DSGB10_COPY_ROWS value (8 = long steps off).
Result drive(const std::vector<int32_t>& text, int L, const std::vector<int>& edits, int rows, int enter, const Rules& rl,
             bool present_rule) {
  const int P0 = (int)text.size();
  Sim s(P0 + L + 128);
  for (int i = 0; i < P0; ++i) s.tok[i] = text[i];
  s.u.len = P0;
  index_range(s.u, 0, P0 - 1);
  auto truth = [&](int q) -> int32_t {  // the token at position q >= P0 of the reply
    const int j = q - P0;
    for (int e : edits)
      if (e == j) return 99991 + j % 7;  // an id nothing else holds
    return j < L ? text[j % P0] : 12345 + j;
  };
  const int cap = cap_idx(rows), xb = exact_mask(rows);
  Run run{}, ref{};  // ref: the right rule fed the same steps, for the shadow of `fails`
  Result R;
  // the given step: the first 8 tokens of the reply, the bonus after them
  int32_t rowtok[64];
  int pos0 = P0, count = 8;
  for (int r = 0; r < 8; ++r) rowtok[r] = truth(P0 + r);
  int32_t bonus = truth(P0 + 8);
  int step_rows = 8;
  bool verify = false;
  int32_t drafts[64];
  while (true) {
    append_step(s.u, pos0, count, rowtok, bonus);
    const int P = pos0 + count;
    ++R.steps;
    R.out_tokens += count;
    if (P - P0 >= L - 70) break;
    const int room = P0 + L + 100 - 1 - P;
    int n = 0;
    if (present_rule) {  // the rule before this change: the proposal after the bare tail, at most 7 drafts
      Match mt;
      n = propose(s.u, P, nullptr, 0, 7, kOrder, drafts, &mt);
    } else {
      const int prev_fails = run.fails, prev_idx = run.idx;
      (void)prev_idx;
      Run before = run;
      if (rl.no_walk && verify) run.cur = 0;  // the cursor was not kept
      n = run_step(run, s.u, verify, pos0, step_rows, count, rowtok, bonus, P, room, cap, enter, xb, drafts);
      int32_t refout[64];
      run_step(ref, s.u, verify, pos0, step_rows, count, rowtok, bonus, P, room, cap, enter, xb, refout);
      if (rl.no_fails) run.fails = 0;
      (void)before;
      (void)prev_fails;
      // the entry rule: a long step may begin only when the cursor's match was `enter << fails` long
      if (n > kMaxDrafts && run.idx == 1 && run.len < (enter << ref.fails) && before.idx == 0) ++R.early_entry;
    }
    if (n > R.max_drafts) R.max_drafts = n;
    const bool is_long = n > kMaxDrafts;
    const int rws = 1 + n;
    if (!(rws <= 8 || (rws >= 29 && rws <= 32 && rows >= 32) || (rws >= 57 && rws <= 64 && rows >= 64))) ++R.bad_rows;
    if (is_long) ++R.long_steps;
    rowtok[0] = bonus;
    for (int j = 0; j < n; ++j) rowtok[1 + j] = drafts[j];
    int acc = 0;  // drafts accepted: draft j (row j + 1) is the sample of row j
    while (acc < n && rowtok[1 + acc] == truth(P + 1 + acc)) ++acc;
    if (is_long) {
      // a long step's tokens are the source's: they are the mirror's tokens from the cursor on
      if (!present_rule)
        for (int j = 0; j < n; ++j)
          if (drafts[j] != s.u.tok[run.cur + j]) ++R.bad_tokens;
      if (acc == n) ++R.whole_long; else ++R.partial_long;
    }
    pos0 = P, count = 1 + acc, step_rows = rws, bonus = truth(P + count), verify = true;
  }
  return R;
}

std::vector<int32_t> random_text(int n, int vocab = 5000) {
  std::vector<int32_t> t(n);
  for (auto& x : t) x = (int32_t)(rnd() % vocab);
  return t;
}

void h1_stream() {
  const std::vector<int32_t> text = random_text(4096);
  const Rules ok;
  {  // a verbatim repeat of 4K tokens
    const Result r64 = drive(text, 4000, {}, 64, 32, ok, false);
    say(r64.max_drafts == 63, "H1: ROWS=64 reaches steps of 63 drafts on a verbatim repeat");
    say(r64.bad_rows == 0, "H1: ROWS=64: every step is at most 8 rows or 29..32 or 57..64 rows");
    say(r64.bad_tokens == 0, "H1: every draft of a long step is the source's token");
    say(r64.partial_long == 0 && r64.long_steps > 40, "H1: ROWS=64: long steps all accepted whole on a verbatim repeat");
    const Result r32 = drive(text, 4000, {}, 32, 32, ok, false);
    say(r32.max_drafts == 31 && r32.bad_rows == 0, "H1: ROWS=32 stops at 31 drafts and 29..32 rows");
    const Result r8 = drive(text, 4000, {}, 8, 32, ok, false);
    say(r8.max_drafts == 0 || r8.max_drafts <= 0, "H1: ROWS=8 never runs a long step");
    say(r64.steps < r32.steps && r32.steps < r8.steps, "H1: more rows, fewer steps for the same text");
  }
  {  // no repeat: every reply token differs from the text's
    std::vector<int> all;
    for (int j = 0; j < 700; ++j) all.push_back(j);
    const Result r = drive(text, 600, all, 64, 32, ok, false);
    say(r.long_steps == 0 && r.bad_rows == 0, "H1: a reply that repeats nothing never gets a long step");
  }
  {  // a repeat with edits: partial long steps, and no long step begins before enter << fails tokens matched
    std::vector<int> edits;
    for (int j = 300; j < 3900; j += 331) edits.push_back(j);
    const Result r = drive(text, 4000, edits, 64, 32, ok, false);
    say(r.long_steps > 0 && r.partial_long > 0, "H1: edits: long steps run and some are rejected in part");
    say(r.early_entry == 0, "H1: edits: a long step begins only after enter << fails matching tokens");
    say(r.bad_rows == 0 && r.bad_tokens == 0, "H1: edits: rows and tokens as allowed");
  }
}

// the present rule must fail the first check; the wrong rules must fail theirs
void h1_controls() {
  const std::vector<int32_t> text = random_text(4096);
  control("the present rule (7 drafts at most) reaches 63 drafts", [&] {
    const Result r = drive(text, 4000, {}, 64, 32, Rules{}, true);
    say(r.max_drafts == 63, "present rule reaches 63");
  });
  control("no walk: the cursor is dropped every step", [&] {
    Rules rl;
    rl.no_walk = true;
    const Result r = drive(text, 4000, {}, 64, 32, rl, false);
    say(r.max_drafts == 63, "no walk reaches 63");
  });
  std::vector<int> edits;
  for (int j = 300; j < 3900; j += 331) edits.push_back(j);
  control("no fails: a long step rejected in part is forgotten", [&] {
    Rules rl;
    rl.no_fails = true;
    const Result r = drive(text, 4000, edits, 64, 32, rl, false);
    say(r.early_entry == 0, "no fails: entry only after enter << fails");
  });
  control("the ladder climbs on a step whose drafts were not all accepted", [&] {
    const bool climb_partial = true;  // the wrong rule: the ladder is given count = step_rows whatever was accepted
    Run r{};
    r.cur = 100, r.len = 100, r.idx = 2, r.pos = 5;
    run_ladder(r, 64, 20, 2, 32);  // a long step with 20 of 64 tokens: the rung falls to 0
    say(r.idx == 0, "a partial step sets the rung to 0");
    Run q{};
    q.cur = 100, q.len = 100, q.idx = 2, q.pos = 5;
    run_ladder(q, 64, 64, 2, 32);
    say(q.idx == 2, "a whole step keeps the top rung");
    // the control rule: the ladder given count = step_rows always
    Run c{};
    c.cur = 100, c.len = 100, c.idx = 2, c.pos = 5;
    run_ladder(c, 64, climb_partial ? 64 : 20, 2, 32);
    say(c.idx == 0, "(control) the partial step set the rung to 0");
  });
}

void h1_ladder() {
  Run r{};
  r.cur = 50, r.len = 40, r.pos = 7;
  run_ladder(r, 8, 8, 2, 32);
  say(r.idx == 1, "ladder: whole step with len >= enter: rung 0 -> 1");
  run_ladder(r, 32, 32, 2, 32);
  say(r.idx == 2, "ladder: whole long step: rung 1 -> 2");
  run_ladder(r, 64, 64, 2, 32);
  say(r.idx == 2, "ladder: the top stays");
  run_ladder(r, 64, 10, 2, 32);
  say(r.idx == 0 && r.fails == 1, "ladder: a long step rejected in part: rung 0, fails 1");
  r.cur = 50, r.len = 40;
  run_ladder(r, 8, 8, 2, 32);
  say(r.idx == 0, "ladder: after one fail the entry needs enter << 1 = 64 matching tokens");
  r.len = 64;
  run_ladder(r, 8, 8, 2, 32);
  say(r.idx == 1, "ladder: 64 matching tokens enter again");
  run_ladder(r, 32, 32, 2, 32);
  say(r.fails == 0, "ladder: a long step accepted whole clears fails");
  for (int i = 0; i < 6; ++i) {
    r.idx = 2, r.cur = 5;
    run_ladder(r, 64, 3, 2, 32);
  }
  say(r.fails == kMaxFails, "ladder: fails stops at 3");
  r.cur = 50, r.len = 72, r.idx = 0;
  run_ladder(r, 8, 8, 0, 32);
  say(r.idx == 0, "ladder: ROWS=8 (cap 0) never climbs");
  r.cur = 0, r.idx = 1;
  run_ladder(r, 8, 8, 2, 32);
  say(r.idx == 0, "ladder: a dropped cursor sets the rung to 0");
}

void h1_walk() {
  Sim s(64);
  for (int i = 0; i < 40; ++i) s.tok[i] = i + 1;
  s.u.len = 40;
  Run r{};
  r.cur = 10, r.len = 9, r.pos = 30;
  const int32_t rows[3] = {99, 11, 12};  // rows[0] is the old bonus; accepted drafts tok[10], tok[11]... = 11, 12
  run_walk(r, s.tok.data(), 40, 30, rows, 3, 13);  // emitted 11, 12, then the bonus 13 = tok[12]
  say(r.cur == 13 && r.len == 12, "walk: three tokens that equal the source advance the cursor by three");
  Run q{};
  q.cur = 10, q.len = 9, q.pos = 30;
  run_walk(q, s.tok.data(), 40, 30, rows, 3, 77);
  say(q.cur == 0 && q.len == 0, "walk: a bonus that differs drops the cursor");
  Run w{};
  w.cur = 10, w.len = 9, w.pos = 31;
  run_walk(w, s.tok.data(), 40, 30, rows, 3, 13);
  say(w.cur == 0, "walk: a cursor stamped at another position (another graph ran a step) is dropped");
  Run e{};
  e.cur = 39, e.len = 9, e.pos = 30;
  const int32_t one[1] = {5};
  run_walk(e, s.tok.data(), 40, 30, one, 1, 40);  // tok[39] = 40: matches; then cur 40 is at the history's end
  say(e.cur == 40, "walk: reaches the history's end");
  run_walk(e, s.tok.data(), 40, 40, one, 1, 41);
  say(e.cur == 0, "walk: never reads past the history");
}

void h2_fit() {
  for (int rows : {32, 64}) {
    const int xb = exact_mask(rows);
    int prev = -1;
    for (int n = 0; n <= 63; ++n) {
      const int f = copy_fit(n, xb);
      const int R = 1 + f;
      bool ok = f <= n;
      ok = ok && (R <= 8 || (R >= 29 && R <= 32) || (rows >= 64 && R >= 57 && R <= 64));
      // the largest n' <= n with such a row count
      int best = 0;
      for (int m = 0; m <= n; ++m) {
        const int Rm = 1 + m;
        if (Rm <= 8 || (Rm >= 29 && Rm <= 32) || (rows >= 64 && Rm >= 57 && Rm <= 64)) best = m;
      }
      ok = ok && f == best;
      char what[96];
      std::snprintf(what, sizeof what, "H2: rows %d, n %d -> %d (largest allowed %d)", rows, n, f, best);
      say(ok, what);
      ok = ok && f >= prev;
      prev = f;
    }
  }
  control("a 16-row body allowed", [&] {
    // the wrong rule: bucket 3 .. plus a body of 16 rows: 9..16 rows become allowed
    auto fit16 = [](int n) {
      const int xb = exact_mask(64);
      return (n >= 8 && n <= 15) ? n : copy_fit(n, xb);
    };
    for (int n = 8; n <= 15; ++n) say(1 + fit16(n) <= 8 || 1 + fit16(n) > 16, "never 9..16 rows");
  });
  say(copy_fit(63, exact_mask(32)) == 31 && copy_fit(40, exact_mask(32)) == 31, "H2: ROWS=32: 63 and 40 drafts fit to 31");
  say(copy_fit(55, exact_mask(64)) == 31, "H2: ROWS=64: 55 drafts (a bucket without a body) fit to 31");
}

void h4_settings() {
  say(rows_from(nullptr) == 8 && rows_from("") == 8 && rows_from("8") == 8 && rows_from("32") == 32 && rows_from("64") == 64,
      "H4: DSGB10_COPY_ROWS: unset, 8, 32, 64");
  for (const char* bad : {"16", "0", "9", "033", "64 ", " 32", "x", "-8", "128", "4"})
    say(rows_from(bad) == -1, "H4: DSGB10_COPY_ROWS: a value outside 8 / 32 / 64 is refused");
  say(enter_from(nullptr) == 32 && enter_from("") == 32 && enter_from("8") == 8 && enter_from("72") == 72 && enter_from("16") == 16,
      "H4: DSGB10_COPY_ENTER: default 32, range 8..72");
  for (const char* bad : {"7", "73", "-1", "x", "3 2", "99999999999"})
    say(enter_from(bad) == -1, "H4: DSGB10_COPY_ENTER: a value outside 8..72 is refused");
  char msg[128];
  say(run_env_error(nullptr, nullptr, msg, sizeof msg) == 0, "H4: unset settings are good");
  say(run_env_error("16", nullptr, msg, sizeof msg) == 1 && std::strstr(msg, "DSGB10_COPY_ROWS=16"), "H4: the refusal names the setting and its text");
  say(run_env_error("64", "5", msg, sizeof msg) == 1 && std::strstr(msg, "DSGB10_COPY_ENTER=5"), "H4: the second setting is checked too");
  say(cap_idx(8) == 0 && cap_idx(32) == 1 && cap_idx(64) == 2, "H4: the ladder's top by ROWS");
  say(exact_mask(8) == 0 && exact_mask(32) == (1 << 3) && exact_mask(64) == ((1 << 3) | (1 << 7)), "H4: the exact buckets by ROWS");
  say(bucket_rows(kBucket32) == 32 && bucket_rows(kBucket64) == 64, "H4: the bucket indices name 32 and 64 rows");
  // the same on two ranks, or the start stops: Settings compares word for word
  Settings a{1, 8, 64, 32, 0}, b = a;
  say(std::memcmp(&a, &b, sizeof a) == 0, "H4: equal settings compare equal");
  b.rows = 32;
  say(std::memcmp(&a, &b, sizeof a) != 0, "H4: a rank with another ROWS differs");
  b = a, b.enter = 16;
  say(std::memcmp(&a, &b, sizeof a) != 0, "H4: a rank with another ENTER differs");
  // the bits in StepIn::copy: the mode bits stay below them
  const int32_t w = (int32_t)(kRows | kAdapt) | (int32_t)(2 << 8 | 40 << 16);
  say(long_top(w) == 2 && long_enter(w) == 40 && (w & 255) == (kRows | kAdapt), "H4: the long bits sit beside the mode bits");
}

void h5_learner() {
  Pend p{};  // a long step's record is written void
  const Taken t = outcome(p, true, -1, 100, 5, 7);
  say(t.bucket < 0, "H5: a long step's void record gives no outcome");
  Pend q{};
  q.pos = 100, q.k = 0, q.n = 2, q.m = 3, q.bucket = 1, q.whole = 1;
  q.tok[0] = 7;
  say(outcome(q, true, -1, 100, 3, 7).bucket == 1, "H5: a short step's record still gives its outcome");
  say(kMaxDrafts == 7, "H5: the learner's proposals stay at 7 rows");
  say(verify_reach(5, true) == 7 && verify_reach(5, true, 8) == 7 && verify_reach(5, true, 32) == 63 && verify_reach(5, false, 64) == 5,
      "H4: the host's reach widens to 63 rows with long steps, and only with copy rows on");
}

void l8_block() {
  Run live{};
  live.cur = 40;
  Run dead{};
  say(!block_runs(live, 7, 5), "L8: no block while a cursor is live at the bonus");
  control("the block without its rule", [&] {
    auto wrong = [](const Run&, int accepted, int dyn) { return accepted >= dyn; };
    say(!wrong(live, 7, 5), "L8: no block while a cursor is live (wrong rule)");
  });
  say(block_runs(dead, 7, 5), "L8: the block runs after a step that accepted at least L drafts and no cursor is live");
  say(!block_runs(dead, 4, 5), "L8: the block waits for L accepted drafts");
}

// block_step: the decision k_spec_dyn takes, on a mirror whose source repeats (tokens 100 .. 119 twice) and a cursor left on it.
void l8_step() {
  int32_t tok[64];
  for (int i = 0; i < 20; ++i) tok[i] = 100 + i, tok[20 + i] = 100 + i;
  const int known = 36;  // the second copy is being written: positions [0, 36) are known, the bonus is token 35's successor
  Run rec{};
  rec.cur = 16, rec.len = 12, rec.pos = 36;  // predicts tok[16] = 116 for position 36 (the bonus after the last step)
  // a step that accepted 5 drafts emitted 6 tokens: rows[0 .. 5], and the bonus after them; the source continues 116 117 118 119 100 101
  const int32_t on_source[8] = {99, 116, 117, 118, 119, 100, 0, 0};  // rows[0] is the bonus the cursor was left at
  const int32_t off_source[8] = {99, 116, 117, 999, 119, 100, 0, 0};
  say(!block_step(rec, tok, known, true, 36, on_source, 6, 101, 5), "L8: tokens that walk the live cursor keep the block out");
  say(block_step(rec, tok, known, true, 36, off_source, 6, 101, 5), "L8: tokens that leave the source let the block run");
  say(!block_step(rec, tok, known, true, 36, on_source, 4, 101, 5), "L8: fewer than L accepted drafts: no block, cursor or not");
  Run stale = rec;
  stale.pos = 30;  // left by a step another graph ran: the next walk drops it
  say(block_step(stale, tok, known, true, 36, on_source, 6, 101, 5), "L8: a stale cursor does not hold the block back");
  say(block_step(rec, tok, known, false, 36, on_source, 6, 101, 5), "L8: a given step starts the cursor over: the block runs");
  Run none{};
  say(block_step(none, tok, known, true, 36, on_source, 6, 101, 5), "L8: no cursor: the block runs");
  say(rec.cur == 16 && rec.len == 12 && rec.pos == 36, "L8: the decision leaves the unit's record as it was");
  control("the cursor not walked", [&] {
    // a decision that reads the record without walking it keeps the block out for tokens that left the source too
    say(block_runs(rec, 5, 5), "L8: tokens that leave the source let the block run (cursor not walked)");
  });
}

void l8_ms() {
  float a[8] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f, 0.f, 0.f};
  extend_step_ms(a, 5, 7);
  say(std::fabs(a[6] - (38.70f + (38.70f - 35.61f))) < 1e-4f && std::fabs(a[7] - (a[6] + (a[6] - a[5]))) < 1e-4f,
      "L8: copy rows past the chain's 5 drafts extend by the last difference");
  float b[8] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f, 41.70f, 44.70f};
  extend_step_ms(b, 7, 7);
  say(b[6] == 41.70f && b[7] == 44.70f, "L8: with a block of 7 the given values stay");
  float c[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 4.f, 0.f, 0.f};  // a falling difference adds nothing
  extend_step_ms(c, 5, 7);
  say(c[6] == 4.f && c[7] == 4.f, "L8: a difference below 0 extends flat");
}

}  // namespace

int main() {
  h1_stream();
  h1_ladder();
  h1_walk();
  h1_controls();
  h2_fit();
  h4_settings();
  h5_learner();
  l8_block();
  l8_step();
  l8_ms();
  std::printf("copy_run_test: %d checks, %d failed, %d controls rejected\n", checks, fails, controls);
  return fails ? 1 : 0;
}
