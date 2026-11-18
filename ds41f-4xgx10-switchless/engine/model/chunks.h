// The chunk pool's numbers (attn.h Compressed, docs/engine.md "The chunk pool"). A number names one slice of each of the
// four pool arrays (8,192 entries each), or, for a bookmark, 2 MiB of its KV slice. Host only. Every rank takes and gives
// the same numbers in the same order, lowest free first; digest() lets rank 0 check that they agree. Number 0 is the
// guard: never taken, every unlisted list entry names it.
#pragma once
#include <cstdint>
#include <vector>

namespace model::chunks {

class Alloc {
 public:
  explicit Alloc(uint32_t n);  // numbers [1, n)
  // The lowest free number; dies when none is free (callers count first: free()).
  uint32_t take();
  // n numbers, lowest first, appended to *out: all of them, or none and false.
  bool take(uint32_t n, std::vector<uint32_t>* out);
  // c back to the free set; dies on a number that is not taken (a double give, or the guard).
  void give(uint32_t c);
  // numbers [numbers(), n) join the free set (the store's share of the pool, or a closed window's numbers coming back)
  void grow(uint32_t n);
  // numbers [n, numbers()) leave the pool (a window's memory taken out of it, Model::window_open): false, nothing
  // changed, when one of them is taken or n would leave no number to take; the taken set and its digest stay as they were
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
  uint32_t n_, free_, lo_;      // lo_: no free number below it
  std::vector<uint64_t> bits_;  // bit c set: c is free
  uint64_t digest_ = 0;
};

uint64_t mix(uint64_t x);  // the digest's per-number hash (splitmix64's finaliser)

}  // namespace model::chunks
