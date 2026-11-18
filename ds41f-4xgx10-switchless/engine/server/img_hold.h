// What the server does with a flight's image rows, as decisions with no engine state (tested by tests/img_hold_test.cpp):
// the rows are taken from the store at the request's first go, read by every later go without a byte sent, and freed when
// the request ends however it ends; a reservation for the numbers counts in admission until they are taken, and an idle
// server that still holds rows says so and lets them go. Numbers: model/img_rows.h.
//
// PICO_OLD_RULES (never defined by the engine) builds each decision as it was before the rows lived in the store: the rows
// sent with every go, nothing freed at the end, no tripwire, so the tests can show they fail on it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>

#include "model/img_rows.h"

namespace imghold {

// A flight's rows as rank 0 records them: the key the ranks hold them under (0: none held yet), and the rows of the
// request's spans.
struct Hold {
  int key = 0;
  size_t rows = 0;
};

// What a go() with spans does about the rows. `take`: the numbers are taken (kStImgTake); `fill`: the rows follow, from
// rank 0's copy, which is freed when the last piece is out. Without either, the ranks read the rows they hold.
struct Go {
  bool take = false, fill = false;
};
inline Go on_go(const Hold& h) {
#ifdef PICO_OLD_RULES
  (void)h;
  return {true, true};  // without the rule: every go, resumes too, sends the rows again
#else
  const bool first = h.key == 0;
  return {first, first};
#endif
}

// The numbers admission adds to a request's want: those it will take and has not yet.
inline int64_t admission_numbers(const Hold& h) {
#ifdef PICO_OLD_RULES
  (void)h;
  return 0;
#else
  return h.key == 0 ? (int64_t)model::imgrows::numbers_for(h.rows) : 0;
#endif
}

// Whether the end of a flight (finish: a stop, a length, a cancel, an error, a client that left) gives its rows back.
inline bool free_at_end(const Hold& h) {
#ifdef PICO_OLD_RULES
  (void)h;
  return false;  // without the rule: nothing is freed at the end
#else
  return h.key != 0;
#endif
}

// With no request in flight the store holds no image rows: a number left means a path ended a flight without freeing
// them. True: say so once and free them all.
inline bool tripwire(size_t flights_in_flight, size_t held_numbers) {
#ifdef PICO_OLD_RULES
  (void)flights_in_flight, (void)held_numbers;
  return false;
#else
  return flights_in_flight == 0 && held_numbers > 0;
#endif
}

// The numbers each admitted request's rows will take, by session (admission and room_for ask by session), until its first
// go takes them or the request ends. An entry belongs to the request that made it: a request of the same session that ends
// while it is held (cancelled before it ran) leaves the admitted request's entry alone, or admission would count too few
// numbers and the take would find no room.
class Asks {
 public:
  void note(uint64_t sid, uint64_t req, int64_t numbers) { m_[sid] = {req, numbers}; }
  void taken(uint64_t sid) { m_.erase(sid); }
  void ended(uint64_t sid, uint64_t req) {
    const auto it = m_.find(sid);
#ifdef PICO_OLD_RULES
    (void)req;
    if (it != m_.end()) m_.erase(it);  // without the rule: any request of the session ends the entry
#else
    if (it != m_.end() && it->second.req == req) m_.erase(it);
#endif
  }
  int64_t extra(uint64_t sid) const {
    const auto it = m_.find(sid);
    return it == m_.end() ? 0 : it->second.numbers;
  }
  void clear() { m_.clear(); }

 private:
  struct Ask {
    uint64_t req = 0;
    int64_t numbers = 0;
  };
  std::map<uint64_t, Ask> m_;
};

}  // namespace imghold
