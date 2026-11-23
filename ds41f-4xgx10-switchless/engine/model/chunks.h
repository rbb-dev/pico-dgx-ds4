// The chunk pool's numbers (attn.h Compressed, docs/engine.md "The chunk pool"). A number names one slice of each of the
// four pool arrays (8,192 entries each), or, for a bookmark, 2 MiB of its KV slice. Host only. Every rank takes and gives
// the same numbers in the same order, lowest free first; digest() lets rank 0 check that they agree. Number 0 is the
// guard: never taken, every unlisted list entry names it.
//
// A number can be a ghost: given back by a split, free, and still holding the bytes it held (a returning conversation takes
// it back and nothing crosses the ring). A ghost carries a stamp, a count the caller advances, never a time. The lend zone
// is the numbers at or above a cut: a window or the Engram lender takes the top of the pool, so a number is handed out from
// the zone last. Both the stamps and the zone are replicated values, the same on every rank.
#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace model::chunks {

class Alloc {
 public:
  explicit Alloc(uint32_t n);  // numbers [1, n)
  // The next number; dies when none is free (callers count first: free()). In this order:
  //   1. the lowest free number that is not a ghost, below the zone;  2. the oldest ghost below the zone (lowest number on a
  //   tie);  3. the lowest free number that is not a ghost, in the zone;  4. the oldest ghost in the zone.
  // With no ghost and no zone that is the lowest free number.
  uint32_t take();
  // n numbers, lowest first, appended to *out: all of them, or none and false.
  bool take(uint32_t n, std::vector<uint32_t>* out);
  // c back to the free set; dies on a number that is not taken (a double give, or the guard).
  void give(uint32_t c);
  // c back to the free set as a ghost with this stamp; dies on a number that is not taken
  void give_ghost(uint32_t c, uint64_t stamp);
  bool is_ghost(uint32_t c) const;
  // the ghost c becomes a plain free number (its link was cut: nothing can revive it, so it is no ghost to be kept for)
  void unghost(uint32_t c);
  uint64_t stamp(uint32_t c) const;  // 0 for a number that is not a ghost
  // the ghost c taken back: it becomes a taken number again; dies on a number that is not a ghost
  void revive(uint32_t c);
  // step 1 or 2 of take() only: false, nothing taken, when no free number is below the zone
  bool take_below_zone(uint32_t* c);
  // the numbers at or above `from` are the zone (kNoZone: none)
  static constexpr uint32_t kNoZone = 0xFFFFFFFFu;
  void set_zone(uint32_t from) { zone_ = from; }
  uint32_t zone() const { return zone_; }
  uint32_t ghosts() const { return ghosts_; }
  uint32_t free_plain() const { return free_ - ghosts_; }  // free numbers that are not ghosts
  uint32_t free_below_zone() const;                         // free numbers (ghosts too) below the zone
  std::vector<uint32_t> ghosts_from(uint32_t c) const;      // the ghosts at or above c, lowest first
  // the ghosts and the zone, mixed: 0 when there is no ghost and no zone, so a store that never splits mixes nothing
  uint64_t split_digest() const;
  // numbers [numbers(), n) join the free set (the store's share of the pool, or a closed window's numbers coming back)
  void grow(uint32_t n);
  // numbers [n, numbers()) leave the pool (a window's memory taken out of it, Model::window_open): false, nothing
  // changed, when one of them is taken or n would leave no number to take; the taken set and its digest stay as they were.
  // A ghost there is cut (it leaves with the number)
  bool shrink(uint32_t n);
  // the first taken number at or above c (the numbers a shrink to c would have to wait for), 0 when none
  uint32_t first_taken_from(uint32_t c) const;
  uint32_t numbers() const { return n_; }         // the arrays' slices, the guard's included
  uint32_t total() const { return n_ - 1; }       // takeable numbers
  uint32_t free() const { return free_; }
  uint32_t held() const { return total() - free_; }
  bool taken(uint32_t c) const;
  uint64_t digest() const { return digest_; }     // of the taken set, whatever the order it was taken in

 private:
  uint32_t pick(bool below_zone_only, bool* found) const;
  void claim(uint32_t c, bool lowest);  // c leaves the free set (a ghost leaves the ghost set with it); lowest: c was the lowest free
  uint32_t n_, free_, lo_;      // lo_: no free number below it
  std::vector<uint64_t> bits_;  // bit c set: c is free (a ghost is free)
  uint64_t digest_ = 0;
  uint32_t zone_ = kNoZone, ghosts_ = 0;
  std::vector<uint64_t> ghost_bits_;           // bit c set: c is a ghost
  std::vector<uint64_t> stamp_;                // a ghost's stamp, by number
  std::set<std::pair<uint64_t, uint32_t>> by_stamp_;  // the ghosts, oldest first
  uint64_t ghost_digest_ = 0;
};

// The quarter slots (the split's unit): a number taken for quarters holds four slots, s = (c << 2) | q, each a quarter of the
// number in every array. A slot is named by as many holders as hold it; the number goes back when its four slots are free.
// The numbers themselves come from the allocator through the store's take, so the slot map never takes one.
class SlotMap {
 public:
  static uint32_t slot(uint32_t c, int q) { return c << 2 | (uint32_t)q; }
  static uint32_t number(uint32_t s) { return s >> 2; }
  static int quarter(uint32_t s) { return (int)(s & 3); }
  void open(uint32_t c);          // c is a quarter number with four free slots; dies when c is open already
  bool has_room() const { return !roomy_.empty(); }
  uint32_t take_slot();           // the lowest free slot of the lowest open number with room, one reference; dies when none
  void ref(uint32_t s);           // one more holder; dies on a slot that is free
  // one holder less: the references left. When that frees the number's last used slot the number is closed and *freed is it
  // (the caller gives it back to the allocator); else *freed is 0.
  uint32_t unref(uint32_t s, uint32_t* freed);
  uint32_t refs(uint32_t s) const;  // 0: free (or its number is not open)
  bool is_open(uint32_t c) const { return nums_.count(c) != 0; }
  uint32_t open_numbers() const { return (uint32_t)nums_.size(); }
  uint32_t used() const { return used_; }  // slots with a holder
  std::vector<uint32_t> open_from(uint32_t c) const;  // open numbers at or above c, lowest first
  uint64_t digest() const;                            // 0 when nothing is open

 private:
  struct Num {
    uint32_t refs[4] = {};
  };
  static bool room(const Num& n) { return !(n.refs[0] && n.refs[1] && n.refs[2] && n.refs[3]); }
  std::map<uint32_t, Num> nums_;
  std::set<uint32_t> roomy_;  // open numbers with a free slot
  uint32_t used_ = 0;
};

uint64_t mix(uint64_t x);  // the digest's per-number hash (splitmix64's finaliser)

}  // namespace model::chunks
