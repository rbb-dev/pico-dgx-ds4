// When the server demotes a conversation from L1 (whole) to L2 (quarter slots), and when it gives L2 back (host only:
// tests/split_pace_test.cpp). Two factors decide a demotion: the conversation's idle age and the pool's pressure, read from the
// spare numbers (Model::slot_spare) against two marks.
//   --kv-split off     nothing is split.
//   --kv-split tiered  level 0, spare at or above `high`: nothing, whatever the age.
//                      level 1, between the marks: pressure p = (high - spare) / (high - low); the idle age a conversation must
//                      have is after_s x (1 - p), and a pass may run pass_ms x (0.2 + 0.8 p) with requests in flight.
//                      level 2, under `low`: L2 is given back first (its least recently used conversation goes to the SSD, no
//                      write: its slots have their images), and with no L2 left the least recently used L1 conversation is
//                      demoted at any age.
//   --kv-split max     level 1 at the full age and the full pass whatever the spare; under `low` as level 2.
// A pass is a run of commands at the top of the server's loop: between decoding runs, never inside one, in no mode. A list held
// back by a deferral (a piece found no quarter number) is due again when a number can be opened (`can_open`), else at its retry
// time; a drop is never held. A command splits at most `budget` pieces (--kv-split-budget, 16); a conversation whose pieces found no
// quarter number is retried after `retry_s` (5). PICO_OLD_RULES builds the engine that never asks.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace splitpace {

enum Mode { kOff = 0, kTiered = 1, kMax = 2 };

struct Marks {
  int64_t high = 0, low = 0;  // spare numbers
  double after_s = 300;       // the base idle age
  double pass_ms = 50;        // the pass's time with requests in flight at full pressure
};

struct Level {
  int level = 0;          // 0 none, 1 aging, 2 reclaim
  double pressure = 0;    // 0..1
  double age_s = 0;       // the idle age a conversation must have for a demotion now
  double pass_ms = 0;     // how long a pass may run
};

inline Level level(Mode mode, int64_t spare, const Marks& m, bool flights) {
  Level l;
#ifdef PICO_OLD_RULES
  (void)mode, (void)spare, (void)m, (void)flights;
  return l;
#else
  if (mode == kOff) return l;
  const double idle_pass = 250;  // no request in flight: nothing is held up by a longer pass
  if (spare < m.low) {
    l.level = 2, l.pressure = 1, l.age_s = 0, l.pass_ms = flights ? m.pass_ms : idle_pass;
    return l;
  }
  if (spare >= m.high) {
    if (mode == kTiered) return l;
    l.level = 1, l.pressure = 0, l.age_s = m.after_s, l.pass_ms = flights ? m.pass_ms : idle_pass;  // max: the full age, the full pass
    return l;
  }
  const double span = (double)(m.high - m.low);
  const double p = span > 0 ? (double)(m.high - spare) / span : 1.0;
  l.level = 1, l.pressure = p, l.age_s = m.after_s * (1 - p), l.pass_ms = flights ? m.pass_ms * (0.2 + 0.8 * p) : idle_pass;
  return l;
#endif
}

enum class Act { kNothing, kDemote, kDropL2 };

// A list held back by a deferral: not before a number can be opened (a deferral that left pieces whole for want of one), else not
// before the retry time.
inline bool held(bool waits, bool can_open, double now, double retry_at) {
  return waits ? !(can_open || now >= retry_at) : now < retry_at;
}

// What the pass does next. `l2`: a conversation holds quarter slots. `cand_idle_s` < 0: no candidate to demote (the least recently
// used listed conversation no request is in flight on); else its idle age in seconds. `held`: a deferral holds the list.
inline Act next(const Level& l, bool l2, double cand_idle_s, bool held) {
#ifdef PICO_OLD_RULES
  (void)l, (void)l2, (void)cand_idle_s, (void)held;
  return Act::kNothing;
#else
  if (l.level == 0) return Act::kNothing;
  if (l.level == 2) {
    if (l2) return Act::kDropL2;  // L2 first: its memory comes back with no write, and L1 keeps its recent conversations
    return cand_idle_s >= 0 && !held ? Act::kDemote : Act::kNothing;
  }
  return cand_idle_s >= 0 && !held && cand_idle_s >= l.age_s ? Act::kDemote : Act::kNothing;
#endif
}

// The least recently used conversation no request is in flight on: `used` is the server's LRU clock.
struct Cand {
  int h = -1;
  uint64_t used = 0;
  bool in_flight = false;
};
inline int pick(const std::vector<Cand>& cands) {
  int best = -1;
  uint64_t used = 0;
  for (const Cand& c : cands) {
    if (c.in_flight) continue;
    if (best < 0 || c.used < used) best = c.h, used = c.used;
  }
  return best;
}

// A pass goes on while its time is not used.
inline bool go_on(double elapsed_ms, double pass_ms) {
#ifdef PICO_OLD_RULES
  (void)elapsed_ms, (void)pass_ms;
  return false;
#else
  return elapsed_ms < pass_ms;
#endif
}

// The command's size: at most `budget` pieces; the conversation leaves the list when a command split fewer than it was given (all it
// could) unless some piece was deferred (no number to open), when it is retried later.
struct Outcome {
  bool done = false;        // leave the list
  double retry_at = 0;      // else: not before this
};
inline Outcome after(int split_now, int budget, int deferred_now, double now, double retry_s = 5.0) {
  Outcome o;
#ifdef PICO_OLD_RULES
  o.done = true;
  (void)split_now, (void)budget, (void)deferred_now, (void)now, (void)retry_s;
#else
  if (deferred_now > 0) o.retry_at = now + retry_s;
  else o.done = split_now < budget;
#endif
  return o;
}

// ---- the imager's pace ---------------------------------------------------------------------------------------
// Nothing between a prompt's chunks, at an admission, in finish() or in persist_slot: the imager works when nothing is in flight, queued, waiting or
// arriving (an idle command, back to back), and inside a decoding run by rides (a toggle in the step's control word, one host all-gather and the
// store's bookkeeping between two steps). A split or an image write never shares the model's stream with a decode step, a prefill chunk or an admission.

// Whether the loop may send an idle imager command now: every one of the five is empty. (flying: requests decoding; rot: the rotation's;
// pfq: prompts queued or being prefilled; ready: admitted, not begun; wait: waiting for a slot or a session; submit: a request arriving.)
// `waiter_images_only`: a request waits and only images can free what it needs, with nothing flying, rotating, queued, admitted or
// arriving: that waiter is the imager's reason to run, so it does not make the engine busy.
// `stalled`: the rotation cannot move (rot_stalled): every request in flight is paused and waits for the pool, so no step, chunk or admission is launched or queued
// and the model's stream is empty; the paused flights then do not make the engine busy (a prompt being prefilled, an admitted request and a submit arriving still do).
inline bool idle_ok(bool flying, bool rot, bool pfq, bool ready, bool wait, bool submit_waiting, bool waiter_images_only = false, bool stalled = false) {
#ifdef PICO_OLD_RULES
  (void)flying, (void)rot, (void)pfq, (void)ready, (void)wait, (void)submit_waiting, (void)waiter_images_only, (void)stalled;
  return true;  // without the rule: a pass runs whatever is in flight (it is level()'s pass_ms that bounds it)
#else
  return (!flying || stalled) && (!rot || stalled) && !pfq && !ready && (!wait || waiter_images_only) && !submit_waiting;
#endif
}

// The rule ("nothing between a prompt's chunks"): a request that waits for quarters not yet written while a prompt is prefilled alone (no decoder
// runs, a prompt is being prefilled, the request waits for a slot or a session, images are still to be made) waits for the reader's end; the writes do not
// ride on the chunks' commands. True exactly then: the server counts it once per reader and says so.
inline bool reader_wait(bool flying, bool pfq, bool wait, bool work_pending) { return !flying && pfq && wait && work_pending; }

// A pass of the server's run() that took no flight and put back at least one for the pool (a window it must wait for, no slot the pool can fill, an extension it
// cannot hold, a start that waits), with no prompt being prefilled: the rotation cannot move until something is freed, and the imager is what can free it.
inline bool rot_stalled(int taken, int pool_waits, bool reading) { return taken == 0 && pool_waits > 0 && !reading; }  // `reading`: a prompt is being prefilled (begun, or its return's units in flight)
// a long prompt queued in pfq_ that has not begun and has no unit of its return left (`front_queued`), whose begin answered "waits" or for which no slot can be
// picked, waits for the pool exactly as a request in wait_ does.
inline bool pf_start_waits(bool front_queued, bool picked, bool begin_waited) { return front_queued && (begin_waited || !picked); }
// The queue keeps the idle imager out while a prompt is being prefilled, or while its front has yet to be asked or can begin now; not while its front only waits for the pool.
inline bool pfq_busy(bool queued, bool reading, bool start_waits) { return queued && (reading || !start_waits); }
// No request in flight can take a step, read a chunk or begin until memory is freed.
inline bool all_wait(bool rot, bool rot_stalled_, bool pfq, bool reading, bool pf_waits) {
  return !reading && (!rot || rot_stalled_) && (!pfq || pf_waits) && (rot_stalled_ || pf_waits);
}
// Whoever waits for the pool: a request in wait_, or a flight put back for the pool in the last pass.
inline bool pool_waiter(bool wait, int pool_waits, bool pf_waits = false) { return wait || pool_waits > 0 || pf_waits; }

constexpr double kRideMs = 250;   // --kv-split-ride-ms: the most often a ride is made
constexpr int kRidePieces = 4;    // --kv-split-ride-pieces: pieces a ride queues
constexpr int kRideRaised = 64;   // pieces a ride queues under the low mark

// Whether rank 0 flips the split bit now. `ready`: the blocks that can be given back without a write (Model::slot_spare plus the clean pieces
// of conversations on no line), read against the marks; `pending`: a conversation has pieces to image, or a command's images are not settled.
// tiered (and off, which images by the same gate: it has the same disk tier and only marks nothing) ride only while `ready` is under the high mark,
// so on a pool that is not short a run's steps are as they were; max rides whenever work is pending. A pause being asked or a flip no step has read
// yet: not due. PICO_OLD_RULES: never.
inline bool ride_due(Mode mode, int64_t ready, const Marks& m, bool pending, double now, double last, bool pausing, bool flip_waiting,
                     double ride_ms = kRideMs, bool waiter = false) {
#ifdef PICO_OLD_RULES
  (void)mode, (void)ready, (void)m, (void)pending, (void)now, (void)last, (void)pausing, (void)flip_waiting, (void)ride_ms, (void)waiter;
  return false;
#else
  if (!pending || pausing || flip_waiting || now - last < ride_ms / 1e3) return false;
  return waiter || mode == kMax || ready < m.high;  // a request waiting for clean blocks is the reason to ride, in every mode
#endif
}

// The pieces a ride queues: `pieces` normally, kRideRaised under the low mark (the pool is short: the imager must outrun the takers)
inline int ride_pieces(int64_t ready, const Marks& m, int pieces = kRidePieces, bool waiter = false) {
  return ready < m.low || waiter ? std::max(pieces, kRideRaised) : pieces;
}
// The units a ride or an idle command may queue: a cache entry's piece and a window bookmark's slice are one unit (786,432 B), the end bookmark's
// slice two. The budget is agreed by the four ranks in this unit and every rank then queues exactly the same jobs: no rank reads its own staging again.
constexpr int kEndUnits = 2;

// What the four ranks settle up to: the smallest of their readings (a rank whose image stream is slow holds the settle back for all; nothing is
// marked early anywhere)
inline uint64_t ride_settled(const uint64_t* done, int n) {
  uint64_t m = n > 0 ? done[0] : 0;
  for (int i = 1; i < n; ++i) m = done[i] < m ? done[i] : m;
  return m;
}
// The pieces a ride may queue: what rank 0 asked for, at most what the smallest staging room takes
inline int ride_budget(int asked, const int* room, int n) {
  int b = asked;
  for (int i = 0; i < n; ++i) b = room[i] < b ? room[i] : b;
  return b < 0 ? 0 : b;
}

// whether a decoding run is ended for another request. Every term is a fact of the moment: `pf`/`pf_busy` a long prompt is queued and is being read or has yet to
// be asked (the interleaver's window decides); `submit` an arrival nobody has looked at; `ends` a request of the run itself ended; `wait` a request waits in wait_; `can_begin` the one
// that waits can begin now with what is free, what idle lines reserve and what is clean (no request in flight loses its line). PICO_OLD_RULES: the older expressions.
struct PauseIn {
  bool batch = false, pf = false, pf_busy = false, burst_over = false, submit = false, admissible = false, ends = false, wait = false, can_begin = false;
};
inline bool pause_for_others(const PauseIn& o) {
#ifdef PICO_OLD_RULES
  if (o.pf) return o.burst_over;
  return o.batch ? (o.submit || o.wait) && (o.ends || o.admissible) : o.submit && o.admissible;
#else
  if (o.pf && o.pf_busy) return o.burst_over;
  if (o.submit && (o.ends || o.admissible)) return true;
  if (o.ends && (o.wait || o.pf)) return true;
  return o.can_begin;
#endif
}
// the decoders' owed time after a decode run of `ran_ms`. A run the interleaver gave a window pays the debt (as always); a run the loop started with no long
// request queued pays it too (the decoders were served in kind), never below 0. PICO_OLD_RULES: only a run with a window.
inline double owed_left(double owed_ms, double ran_ms, bool window) {
#ifdef PICO_OLD_RULES
  if (!window) return owed_ms;
#else
  (void)window;
#endif
  return owed_ms - ran_ms > 0 ? owed_ms - ran_ms : 0;
}
// A slot is free for an arrival's question "can it begin if the run ends" when no request is in flight on it, or the one that was ended in this run (its DONE sent): its slot is
// freed at the run's end whatever the arrival does. PICO_OLD_RULES: only a slot with no request in flight.
inline bool slot_free_for_arrival(bool in_flight, bool ended_in_run) {
#ifdef PICO_OLD_RULES
  (void)ended_in_run;
  return !in_flight;
#else
  return !in_flight || ended_in_run;
#endif
}
// The window a decode run gets between two reads of a long prompt: none while the front only waits for the pool (nobody is being prefilled, nobody owes the decoders), else the owed time.
inline double run_window(bool pf_waits, bool reading, double owed_ms, double burst_max_ms) {
#ifdef PICO_OLD_RULES
  (void)pf_waits, (void)reading;
  return std::min(owed_ms > 0 ? owed_ms : burst_max_ms, burst_max_ms);
#else
  if (pf_waits && !reading) return 0;
  return std::min(owed_ms > 0 ? owed_ms : burst_max_ms, burst_max_ms);
#endif
}
// The ask's pace: not within 50 ms of the last ask, not within 1 s of a miss.
inline bool ask_due(double now, double last, double miss_until) { return now - last >= 0.05 && now >= miss_until; }

}  // namespace splitpace
