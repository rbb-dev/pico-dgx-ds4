// A unit's token history for the Engram hashing (n-grams read 3 positions back): the ids of positions [base, end()),
// kept by absolute position. A prompt in flight is held whole; once its prefill has ended the history is trimmed to
// its last kHistKeep ids, which is more than a speculative rewind reaches, so every position the hashing or a rewind
// asks for is still held and memory follows the live window and not the longest prompt ever seen. A position below
// `base` reads as id 0, as the zero-pad of a history that was never trimmed does. Host only; no engine state.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace model {

constexpr int kHistKeep = 4096;         // ids a trimmed history keeps (at least a speculative rewind's reach)
constexpr size_t kHistCapKeep = 1 << 20;  // a history whose capacity is above this many ids gives it back when it is cleared

struct TokWin {
  int base = 0;           // the position of v[0]
  std::vector<int32_t> v;  // the ids of positions [base, base + v.size())
  int pend = 0;           // a prompt's end whose prefill has not reached it: the history is not trimmed

  int end() const { return base + (int)v.size(); }
  const int32_t* data() const { return v.data(); }  // the id at position p is data()[p - base]
  int32_t at(int p) const { return p < base ? 0 : v[p - base]; }

  // The history ends at position n (positions past the old end read 0). False, and nothing done, when n is below `base`:
  // the ids that a rewind to there would keep are not held.
  bool resize(int n) {
    if (n < base) return false;
    v.resize((size_t)(n - base));
    return true;
  }
  // Positions [pos0, pos0 + n) are `tokens`, the history ending after them. A pos0 below `base` starts the window again
  // at pos0 (the caller sends the tokens from there); one above keeps the ids before it that fit kHistKeep.
  void assign(int pos0, int n, const int32_t* tokens) {
    if (pos0 < base) {
      base = pos0;
      v.assign(tokens, tokens + n);
    } else {
      v.resize((size_t)(pos0 + n - base));
      std::copy(tokens, tokens + n, v.begin() + (pos0 - base));
      drop_below(pos0 - kHistKeep);
    }
    pend = pos0 + n;
  }
  // The rows' ids at [pos0, pos0 + R) after the history was cut to pos0 (a decode or verify step's tokens). False when
  // pos0 is below `base`.
  bool put(int pos0, int R, const int32_t* tokens) {
    if (!resize(pos0)) return false;
    v.insert(v.end(), tokens, tokens + R);
    if (pend && end() >= pend) pend = 0;
    return true;
  }
  // The prefill has read the prompt up to position e.
  void prefilled(int e) {
    if (pend && e >= pend) pend = 0;
  }
  // Cut the history to its last kHistKeep ids, unless a prompt's prefill is still to read it. A window already within
  // twice that is left as it is (the cut is a copy of the kept ids).
  void trim() {
#ifndef PICO_OLD_RULES
    if (pend || v.size() <= 2 * (size_t)kHistKeep) return;
    drop_below(end() - kHistKeep);
    if (v.capacity() > kHistCapKeep) v.shrink_to_fit();
#endif
  }
  // Nothing held; a capacity above kHistCapKeep is given back.
  void clear() {
    base = pend = 0;
    v.clear();
#ifndef PICO_OLD_RULES
    if (v.capacity() > kHistCapKeep) std::vector<int32_t>().swap(v);
#endif
  }
  void swap(TokWin& o) {
    std::swap(base, o.base), std::swap(pend, o.pend), v.swap(o.v);
  }

 private:
  void drop_below(int p) {
#ifndef PICO_OLD_RULES
    if (p <= base) return;
    v.erase(v.begin(), v.begin() + std::min<size_t>((size_t)(p - base), v.size()));
    base = p;
#else
    (void)p;
#endif
  }
};

}  // namespace model
