// Which memory of the pool gives way: the decisions of the server's reclaim, window plan, window close and live set, with
// no engine state (tested by tests/reclaim_test.cpp and tests/resv_test.cpp). The rule (docs/engine.md, "Memory given back
// on demand"): everything in the pool that no running request uses is cache and gives way, to the disk tier or dropped; a
// request never waits on, fails on or is degraded by cache; memory is given back only when a consumer asks.
//
// A request is running when it is in flight and not paused between turns. A paused request (time-slicing) holds nothing a
// step is using: its line can be detached, and it resumes through go() when its line did not survive.
//
// PICO_OLD_RULES (never defined by the engine) builds each decision without the rule, so the tests can show they fail on
// it.
#pragma once

#include <cstdint>
#include <vector>

namespace reclaim {

inline bool running(bool in_flight, bool paused) { return in_flight && !paused; }

// Whether a slot's session goes into Cmd::live: the store never evicts the sessions of the slots a command names live
// (state_pin), and drops the images of none of them to make room in the disk tier's file.
inline bool counts_as_live(bool in_flight, bool paused) {
#ifdef PICO_OLD_RULES
  (void)in_flight, (void)paused;
  return true;  // without the rule: every slot's session, whether or not a request uses it
#else
  return running(in_flight, paused);
#endif
}

// Whether finish() zeroes a kept session's reservation on its line. The reservation is a reservation for a running request;
// a finished one has none, and a leftover one (up to --grow-ahead positions) is counted as reserved by every sum that
// refuses a consumer (Model::slot_uncommitted, take_extra, the window's unheld sum, the dashboard's reserved).
inline bool unreserve_at_finish() {
#ifdef PICO_OLD_RULES
  return false;  // without the rule: finish() issues kStSave only, and the reservation stays until the line is released
#else
  return true;
#endif
}

// The numbers the lines are reserved but do not hold, summed over the slots. Zero whenever no request is in flight (the
// server's tripwire reads it).
struct SlotNums {
  int64_t committed = 0, mapped = 0;
};
inline int64_t idle_reserves(const std::vector<SlotNums>& slots) {
  int64_t n = 0;
  for (const SlotNums& s : slots) n += s.committed > s.mapped ? s.committed - s.mapped : 0;
  return n;
}

// A line whose conversation a consumer may take (idle: no running request uses it, and it is not the consumer's own
// conversation or its base, nor in the run being built).
struct Line {
  int slot = -1;
  uint64_t id = 0;
  uint64_t used = 0;      // Session::used: the least recently used goes first
  int64_t committed = 0;  // Model::slot_committed: a line holding and reserving nothing frees nothing
  bool idle = false;
};

// The least recently used idle line that has something to give, ties to the lowest slot; -1: none. `skip_slot` is the
// consumer's own line.
inline int pick_victim(const std::vector<Line>& lines, int skip_slot) {
  int best = -1;
  uint64_t used = 0;
  for (const Line& l : lines) {
    if (l.slot == skip_slot || !l.idle || l.committed <= 0) continue;
    if (best < 0 || l.used < used || (l.used == used && l.slot < best)) best = l.slot, used = l.used;
  }
  return best;
}

// Server::reclaim's loop: while `short_by()` holds, the least recently used idle line is detached (`detach(slot)`: the
// conversation goes to the store, the line's memory to the pool). Each pass takes one line, so at most lines.size() passes.
// Returns the lines detached; the caller asks `short_by()` again for the verdict.
template <class Short, class Detach>
int detach_until(std::vector<Line> lines, int skip_slot, Short short_by, Detach detach) {
  int n = 0;
  while (short_by()) {
    const int v = pick_victim(lines, skip_slot);
    if (v < 0) break;
    detach(v);
    for (Line& l : lines)
      if (l.slot == v) l.idle = false, l.committed = 0;
    ++n;
  }
  return n;
}

// A session that holds a number at or above a window's top (Model::window_plan); slot < 0: it is on no line, so the model
// evicts it.
struct Holder {
  uint64_t id = 0;
  int slot = -1;
};
struct WindowActions {
  std::vector<int> detach;        // slots of idle holders: each detached before the window command
  std::vector<uint64_t> running;  // holders a running request uses: the window waits for them, with no command
};
// What a window plan asks of the server: the holders on lines that no running request uses are detached, the others are
// named. `idle(id)` is the server's test, with no own or base exemption: the request's own conversation and its base give
// way to the window like any other (the model evicts them last).
template <class Idle>
WindowActions window_actions(const std::vector<Holder>& holders, Idle idle) {
  WindowActions a;
  for (const Holder& h : holders) {
    if (h.slot < 0) continue;
#ifdef PICO_OLD_RULES
    (void)idle;
    a.running.push_back(h.id);  // without the rule: a holder on a line is busy; the request's own and its base are pinned
#else
    if (idle(h.id)) a.detach.push_back(h.slot);
    else a.running.push_back(h.id);
#endif
  }
  return a;
}

// The windows (factors 32 and 64) as the close sees them.
struct Win {
  int factor = 0;
  bool open = false;
  bool in_use = false;  // a request of this factor exists anywhere: queued, waiting, held, in a run, in turns, or being read
  double used_at = 0;
};
// The window to close when a consumer is short: the least recently used open one no request needs; 0: none.
inline int window_to_close(const std::vector<Win>& w) {
  int best = 0;
  double at = 0;
  for (const Win& x : w) {
    if (!x.open || x.in_use) continue;
    if (!best || x.used_at < at) best = x.factor, at = x.used_at;
  }
  return best;
}

// --window-idle-s: a window unused this long is freed by time when an operator turned it on. Never by default.
#ifdef PICO_OLD_RULES
constexpr int kWindowIdleS = 300;  // without the rule: freed after 300 s whatever a request wants next
#else
constexpr int kWindowIdleS = 0;
#endif
inline bool timer_may_close(int idle_s, double now, double used_at, bool in_use) {
#ifdef PICO_OLD_RULES
  (void)in_use;
  return idle_s > 0 && now - used_at >= idle_s;
#else
  return idle_s > 0 && !in_use && now - used_at >= idle_s;
#endif
}

// Window bookmarks a decode run takes: one at each multiple of `every` in (pos0, pos0 + max_new + slack], an upper bound
// (the run may stop early, and keep_snap thins some). slack covers the positions a queued or drafted step writes past
// max_new.
inline int run_snaps(int pos0, int max_new, int every, int slack) {
  if (every <= 0 || max_new <= 0) return 0;
  const int64_t end = (int64_t)pos0 + max_new + slack;
  const int64_t n = end / every - (int64_t)pos0 / every;
  return n > 0 ? (int)n : 0;
}

}  // namespace reclaim
