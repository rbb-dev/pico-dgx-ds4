// The imager's ride on the wire (host only: no engine state, tested by tests/ride_wire_test.cpp). Each rank sends one
// Wire in the ride's host all-gather: its own reading of its image stream (`done`), its staging room (`room`), its number and the replicated readings the ride acts
// on (the image command count, the queued jobs, the allocator's ghost count and free numbers), and, from rank 0, the request (h, budget, live). After the
// gather every rank runs the same function over the same four words and gets one decision; the four readings that must be equal are checked first, and a ride
// whose ranks disagree stops (the caller dies with the text) before it settles, queues or marks anything.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ridewire {

constexpr int kLive = 64;  // model::kMaxSlots (cmd.cpp static_asserts it)
constexpr int kRanks = 4;

struct Wire {
  uint64_t done = 0;                                                    // this rank's reading of its image stream: the highest command whose work ended
  uint64_t img_seq = 0, jobs = 0, ghosts = 0, free_numbers = 0;         // replicated: every rank holds the same
  int32_t rank = 0, room = 0, h = 0, budget = 0;                        // the sender; the pieces its staging takes; rank 0's request
  int32_t live[kLive] = {};                                             // rank 0's: the sessions a request is in flight on
};

struct Decision {
  uint64_t settled = 0;
  int room = 0, h = 0, budget = 0;
  int32_t live[kLive] = {};
};

// "" and the decision, or what differs between the ranks (the four values of each field)
inline std::string decide(const Wire* all, int n, Decision* d) {
  char b[400];
  for (int r = 0; r < n; ++r)
    if (all[r].rank != r) {
      std::snprintf(b, sizeof b, "the word at place %d says it is rank %d: the ranks are not in the same ride", r, all[r].rank);
      return b;
    }
  const auto same = [&](const char* name, auto get) -> std::string {
    for (int r = 1; r < n; ++r)
      if (get(all[r]) != get(all[0])) {
        std::snprintf(b, sizeof b, "ranks differ in %s: %llu %llu %llu %llu", name, (unsigned long long)get(all[0]), (unsigned long long)(n > 1 ? get(all[1]) : 0),
                      (unsigned long long)(n > 2 ? get(all[2]) : 0), (unsigned long long)(n > 3 ? get(all[3]) : 0));
        return b;
      }
    return "";
  };
  std::string e;
  if (!(e = same("image commands", [](const Wire& w) { return w.img_seq; })).empty()) return e;
  if (!(e = same("queued jobs", [](const Wire& w) { return w.jobs; })).empty()) return e;
  if (!(e = same("ghosts", [](const Wire& w) { return w.ghosts; })).empty()) return e;
  if (!(e = same("free numbers", [](const Wire& w) { return w.free_numbers; })).empty()) return e;
  d->settled = all[0].done, d->room = all[0].room;
  for (int r = 1; r < n; ++r) d->settled = std::min(d->settled, all[r].done), d->room = std::min(d->room, all[r].room);
  d->h = all[0].h;
  d->budget = std::min(all[0].budget, d->room);
  std::copy(all[0].live, all[0].live + kLive, d->live);
  return "";
}

// the branch with no ring, kept only as the named control of ride_wire_test: no ring, so every rank decides from its own word alone (ranks 1 to 3 hold no request)
inline Decision decide_alone(const Wire& mine) {
  Decision d;
  d.settled = mine.done, d.room = mine.room, d.h = mine.h, d.budget = std::min(mine.budget, mine.room);
  std::copy(mine.live, mine.live + kLive, d.live);
  return d;
}

}  // namespace ridewire
