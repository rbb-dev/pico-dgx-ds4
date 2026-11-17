// Which memory of the pool gives way: the decisions of the server's reclaim and its live set, with no engine state (tested
// by tests/reclaim_test.cpp). The rule: everything in the pool that no running request uses is cache and gives way, to the
// disk tier or dropped; a request never waits on, fails on or is degraded by cache; memory is given back only when a
// consumer asks.
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
// (state_pin).
inline bool counts_as_live(bool in_flight, bool paused) {
#ifdef PICO_OLD_RULES
  (void)in_flight, (void)paused;
  return true;  // without the rule: every slot's session, whether or not a request uses it
#else
  return running(in_flight, paused);
#endif
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

}  // namespace reclaim
