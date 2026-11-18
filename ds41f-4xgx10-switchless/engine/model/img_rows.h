// A request's image rows as store numbers: where each row lives and which flight holds which numbers (host only, no GPU;
// tested by tests/img_rows_test.cpp). A row is 5,120 bf16 values (10,240 B); a chunk number holds 2 MiB of its first
// array's slice (8,192 entries of 256 B), so 204 whole rows fit in one number and the last 8,192 B of it stay unused.
// The rows of a flight are taken once, read by every prefill chunk of every run of the request (a resume sends nothing),
// and given back when the request ends, whichever way it ends. A flight is named by a key the server hands out.
//
// PICO_OLD_RULES (never defined by the engine) builds the rows in a pinned host buffer, no store numbers,
// one run for any span, so the tests can show they fail on it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace model::imgrows {

constexpr size_t kRowBytes = 5120 * 2;
constexpr size_t kNumberBytes = 8192 * 256;  // static_assert'ed against the pool's geometry in state.cu
constexpr size_t kPerNumber = kNumberBytes / kRowBytes;
static_assert(kPerNumber == 204, "204 whole rows in a chunk number");

// Numbers the rows of one request take.
inline size_t numbers_for(size_t rows) {
#ifdef PICO_OLD_RULES
  (void)rows;
  return 0;  // the rows were in a pinned host buffer
#else
  return (rows + kPerNumber - 1) / kPerNumber;
#endif
}

// Where row r sits: the index into the flight's numbers and the byte offset inside that number.
struct Where {
  size_t number = 0, offset = 0;
};
inline Where where(size_t r) { return {r / kPerNumber, r % kPerNumber * kRowBytes}; }

// Rows [r0, r1) cut at every number's edge: f(first_row, rows) for each piece inside one number.
template <class F>
void runs(size_t r0, size_t r1, F&& f) {
#ifdef PICO_OLD_RULES
  if (r0 < r1) f(r0, r1 - r0);
#else
  while (r0 < r1) {
    const size_t end = std::min(r1, (r0 / kPerNumber + 1) * kPerNumber);
    f(r0, end - r0);
    r0 = end;
  }
#endif
}

using Nums = std::vector<uint32_t>;

enum class Take { kTaken, kHeld, kNoRoom, kBad };

// The flights' rows: key -> its numbers. The numbers come from and go back to the pool through the functors, so that the
// pool's own accounting (refs, the allocator) stays where it is.
class Ledger {
 public:
  struct Rows {
    Nums nums;
    size_t rows = 0;
  };
  // Rows for `key`: numbers_for(rows) from take_fn(n, &out) (false: none taken). kHeld: the key has its rows already, and
  // nothing is taken (a second go of the same request).
  template <class TakeFn>
  Take take(int key, size_t rows, TakeFn&& take_fn) {
    if (key <= 0 || rows == 0) return Take::kBad;
    if (m_.count(key)) return Take::kHeld;
    Rows r;
    r.rows = rows;
    if (!take_fn(numbers_for(rows), &r.nums)) return Take::kNoRoom;
    m_.emplace(key, std::move(r));
    return Take::kTaken;
  }
  const Rows* find(int key) const {
    const auto it = m_.find(key);
    return it == m_.end() ? nullptr : &it->second;
  }
  // key's numbers back through give_fn(nums); false when the key holds none.
  template <class GiveFn>
  bool release(int key, GiveFn&& give_fn) {
    const auto it = m_.find(key);
    if (it == m_.end()) return false;
    give_fn(it->second.nums);
    m_.erase(it);
    return true;
  }
  // every key's, whatever the server did; the numbers given back.
  template <class GiveFn>
  size_t release_all(GiveFn&& give_fn) {
    size_t n = 0;
    for (auto& kv : m_) n += kv.second.nums.size(), give_fn(kv.second.nums);
    m_.clear();
    return n;
  }
  size_t numbers() const {
    size_t n = 0;
    for (const auto& kv : m_) n += kv.second.nums.size();
    return n;
  }
  size_t flights() const { return m_.size(); }
  template <class F>
  void each(F&& f) const {
    for (const auto& kv : m_) f(kv.first, kv.second);
  }

 private:
  std::map<int, Rows> m_;
};

}  // namespace model::imgrows
