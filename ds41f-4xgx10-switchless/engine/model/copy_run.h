// Long copy steps (docs/batching.md, "Long copy steps"): a request that decodes alone and repeats text it already holds may verify
// 31 or 63 copy drafts in one step, run as an exact body of the one-session step graph (the bits of a step of at most 8 rows).
// This header is the rule, host and device alike (CPY_HD, no GPU types): the cursor that walks a source, the ladder of draft counts,
// the row counts an exact body can hold, and the rule that keeps the 7-draft block out of a cursor's way. The settings
// (DSGB10_COPY_ROWS, DSGB10_COPY_ENTER) are parsed in copy.h. tests/copy_run_test.cpp checks every function against a plain reference.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "model/copy.h"

namespace model::copy {

constexpr int kMaxFails = 3;
constexpr int kStepRows = 8;                            // rows of one session's step before long steps (config.h kMaxSessRows)
constexpr int kMaxLong = 63;                            // the most drafts of a long step: 64 rows less the bonus
// The ladder: drafts a step may hold at rung 0 / 1 / 2. (Functions, not arrays: the device reads them too.)
CPY_HD constexpr int long_drafts(int idx) { return idx >= 2 ? 63 : idx == 1 ? 31 : kMaxDrafts; }
// The rows of wide bucket i: ops::wide_bucket_rows (spec.cu checks the two agree at the start).
CPY_HD constexpr int bucket_rows(int i) { return i == 0 ? 20 : i == 1 ? 24 : i == 2 ? 28 : i == 3 ? 32 : i == 4 ? 40 : i == 5 ? 48 : i == 6 ? 56 : 64; }
constexpr int kBucket32 = 3, kBucket64 = 7;             // the buckets that have an exact body

// The ladder's top for a DSGB10_COPY_ROWS value: 0 (8: no long steps), 1 (32), 2 (64).
CPY_HD inline int cap_idx(int rows) { return rows >= 64 ? 2 : rows >= 32 ? 1 : 0; }
// The buckets with an exact body, as a mask over kBucketRows: 32 -> {32 rows}, 64 -> {32 rows, 64 rows}.
CPY_HD inline int exact_mask(int rows) { return rows >= 64 ? (1 << kBucket32) | (1 << kBucket64) : rows >= 32 ? 1 << kBucket32 : 0; }

// The smallest bucket that holds R rows (R in 17..64).
CPY_HD inline int bucket_of_rows(int R) {
  int i = 0;
  while (i < 7 && bucket_rows(i) < R) ++i;
  return i;
}
// A one-session step holds 1 + n rows when that is at most 8 rows (the bodies of before) or its bucket has an exact body (xb: bit
// i = kBucketRows[i]); n drafts become the most drafts below them that do. Never 9..16 rows, never a bucket without an exact body.
CPY_HD inline int copy_fit(int n, int xb) {
  while (n > kMaxDrafts) {
    const int R = 1 + n;
    if (R <= 16) return kMaxDrafts;
    const int b = bucket_of_rows(R);
    if ((xb >> b) & 1) return n;
    n = (b > 0 ? bucket_rows(b - 1) : 16) - 1;
  }
  return n;
}

// The cursor of one unit (device memory, beside the learner's record). cur: the mirror index of the token predicted for the
// position after the bonus (0: none; a source token is never at index 0, a match ends at kOrder - 1 or later); len: the tokens in a
// row that matched behind it; pos: the bonus' position when the cursor was last brought up to date (a cursor whose step was run by
// another graph does not match the next step's first row and is dropped); idx: the ladder's rung (0 / 1 / 2: at most 7 / 31 / 63
// drafts); fails: long steps not accepted whole since the last that was (at most 3).
struct Run {
  int32_t cur, len, pos;
  int8_t idx, fails;
  int16_t pad;
};
static_assert(sizeof(Run) == 16, "a unit's cursor is four words");

CPY_HD inline void run_drop(Run& r) { r.cur = 0, r.len = 0; }
// A request's first spec step (a given step) starts the unit over.
CPY_HD inline void run_reset(Run& r) { r = Run{}; }

// The tokens the step emitted past its first row, then its bonus, walk the cursor while each equals the source's token. tok /
// known: the unit's mirror and the positions it holds; pos0: the step's first row's position (= the bonus the cursor was left at);
// rows: the step's tokens (rows[0] is that bonus, rows[1 .. count) the accepted drafts); count: the tokens it emitted.
CPY_HD inline void run_walk(Run& r, const int32_t* tok, int known, int pos0, const int32_t* rows, int count, int32_t bonus) {
  if (r.cur <= 0) return run_drop(r);
  if (r.pos != pos0) return run_drop(r);
  for (int j = 1; j <= count; ++j) {
    const int32_t t = j < count ? rows[j] : bonus;
    if (r.cur >= known || tok[r.cur] != t) return run_drop(r);
    ++r.cur, ++r.len;
  }
}

// The ladder, from the device's own verdict on the step just run (step_rows rows, count tokens emitted) and the walk: a step whose
// drafts were all accepted and whose tokens all walked the cursor raises the rung by one up to the top (cap), but from 0 only when the
// match is at least `enter << fails` long; any other step sets it to 0. A long step (more than 8 rows) not accepted whole adds one
// to fails (at most 3); one accepted whole clears it.
CPY_HD inline void run_ladder(Run& r, int step_rows, int count, int cap, int enter) {
  const bool longs = step_rows > kStepRows;
  const bool whole = count == step_rows && r.cur > 0;
  if (longs) r.fails = whole ? 0 : (int8_t)(r.fails + 1 < kMaxFails ? r.fails + 1 : kMaxFails);
  if (!whole) {
    r.idx = 0;
  } else if (r.idx == 0) {
    if (cap > 0 && r.len >= (enter << r.fails)) r.idx = 1;
  } else {
    r.idx = (int8_t)(r.idx + 1 < cap ? r.idx + 1 : cap);
  }
  if (r.idx > cap) r.idx = (int8_t)cap;
}

// No cursor: the latest earlier occurrence of the conversation's bare tail (the bonus at P and the 7 tokens before it).
CPY_HD inline void run_find(Run& r, const Unit& u, int P) {
  if (r.cur > 0) return;
  const Match m = find(u, P, nullptr, 0);
  if (m.e >= 0) r.cur = m.e + 1, r.len = m.len;
}

// The drafts of a long step: from the cursor, at most the rung's width, the tokens known (the source ends at P, the bonus), the
// room below the step's end (room: pos_end - 1 - P), stopping at the first token that is not a real id, then copy_fit to a body
// that exists. Returns the drafts (more than 7), out[0 ..) their tokens; 0: no long step (the present rows apply).
CPY_HD inline int run_plan(const Run& r, const Unit& u, int P, int room, int xb, int32_t* out) {
  if (r.idx <= 0 || r.cur <= 0 || xb == 0) return 0;
  int n = long_drafts(r.idx);
  if (n > P - r.cur + 1) n = P - r.cur + 1;
  if (n > room) n = room;
  if (n <= kMaxDrafts) return 0;
  int c = 0;
  while (c < n && valid_token(u.tok[r.cur + c])) ++c;
  n = copy_fit(c, xb);
  if (n <= kMaxDrafts) return 0;
  for (int j = 0; j < n; ++j) out[j] = u.tok[r.cur + j];
  return n;
}

// The whole of k_spec_next's cursor work for one step, in order: a given step starts over, a verify step walks and climbs; then
// the cursor is found if there is none, stamped, and the plan made. pos0 / count / rows / bonus: the step just verified; step_rows:
// its rows; P: the bonus' position; cap / enter / xb: the settings. Returns run_plan's drafts.
CPY_HD inline int run_step(Run& r, const Unit& u, bool verify, int pos0, int step_rows, int count, const int32_t* rows, int32_t bonus,
                           int P, int room, int cap, int enter, int xb, int32_t* out) {
  if (!verify) {
    run_reset(r);
  } else {
    run_walk(r, u.tok, u.len, pos0, rows, count, bonus);
    run_ladder(r, step_rows, count, cap, enter);
  }
  run_find(r, u, P);
  r.pos = P;
  return run_plan(r, u, P, room, xb, out);
}

// The 7-draft block (DSGB10_DRAFT_MAX=7, DSGB10_DRAFT_DYN=L) runs after a step that accepted at least L drafts, but never
// while a cursor is live at the bonus: a copy row costs a lookup and its chance is learned from our traffic; rows 6 and 7 of the block
// cost two more rows of the draft pass and lie past the depths the confidence network covers.
CPY_HD inline bool block_runs(const Run& r, int accepted, int dyn) {
#ifdef COPY_RUN_TEST_NO_BLOCK_RULE  // tests/copy_run_test.cpp: the block without the rule; no engine build defines it
  return accepted >= dyn;
#else
  return accepted >= dyn && r.cur <= 0;
#endif
}

}  // namespace model::copy
