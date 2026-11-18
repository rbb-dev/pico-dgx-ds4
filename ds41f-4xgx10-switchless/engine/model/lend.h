// Memory lent by the chunk pool to a borrower that is not a request: a YaRN window's buffers (window.cu) or the Engram
// row cache's block (lend.cu). The pool gives whole groups of 64 numbers (the smallest set whose slices of the four arrays
// are whole 2 MiB chunks), the borrower holds them as device or host memory, and gives them back. This header is the
// accounting and the decisions, with no GPU and no engine state: the ledger of groups out, the take and give-back
// sequences over a backend, the proof that a hand-over really moved memory, the cache's sizes, the resize plan and the
// retry back-off. Built with -DPICO_OLD_RULES each rule is built without its change (a failed give-back kills the process,
// the proof's tolerance is the null control's spread alone, an oversized setting is refused, a failed size is retried at
// once): the tests that mark "red" fail there. Host only.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace model::lend {

constexpr int kGroupNums = 64;
constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kTolFloor = 256 * kMiB;  // the proof's least tolerance: a release and re-create cycle lands this far off
constexpr uint64_t kPagesSlack = 2 * kMiB;  // the block's own resident pages are exact to a page; one huge page of slack
constexpr double kBackoffS = 300;           // the same size is not tried again before this, unless the setting changes

enum Status { kOk = 0, kNoRoom, kBusy, kFailed };

// ---- the groups out of the pool ----

struct Ledger {
  size_t n0 = 0;   // the pool's numbers before the first group left (0: none out)
  int groups = 0;  // groups out of the pool, every borrower's
  int owed = 0;    // of them, groups a borrower has finished with whose numbers could not be made again yet
  // the pool's numbers with `g` groups out: its size before the first, less whole groups below the last group boundary
  size_t numbers_for(int g) const { return g == 0 ? n0 : cut_top(n0, (size_t)g); }
};

// The most groups the pool can lend in all (every borrower's) and still hold one conversation of `whole` numbers and the
// guard: `numbers` is the pool's size before the first group left.
inline int max_total_groups(size_t numbers, size_t whole) {
  const size_t base = numbers / kGroupNums * kGroupNums;
  if (base < whole + 2 + kGroupNums) return 0;
  return (int)((base - whole - 2) / kGroupNums);
}

// The pool's side of a take and a give-back. Every call is a collective on the real engine (every rank, the same answer),
// the test's fake is one rank.
//   numbers()            the pool's numbers now
//   vacate(top)          makes the numbers at or above `top` free (evicting what holds them): false, or true when a running
//                        request holds one (busy)
//   make_room(removed)   the lines' reservations stay covered with `removed` numbers gone: false when they cannot
//   release(top, g)      the numbers at or above `top` leave the allocator and their chunks the driver; false when the
//                        driver's handles that went are not the ones the arithmetic says (the numbers are back then)
//   grow_all(n)          n more numbers on every rank, or none of them anywhere: false when any rank could not
template <class B>
class Lender {
 public:
  Lender(B& b, Ledger& l) : b_(b), l_(l) {}

  // `g` more groups out of the pool. kNoRoom: it could not hold one conversation of `whole` numbers; kBusy: running requests
  // hold what the cut takes, nothing left the pool (the conversations evicted on the way stay evicted); kFailed: the
  // driver's accounting did not match, the groups are back (or owed).
  int take(int g, size_t whole) {
    const size_t now = b_.numbers();
    const size_t n0 = l_.groups ? l_.n0 : now;
    const size_t base = n0 / kGroupNums * kGroupNums, cut = (size_t)(l_.groups + g) * kGroupNums;
    if (cut + 2 > base || base - cut < whole + 2) return kNoRoom;
    const size_t top = base - cut;
    if (top < now) {
      if (b_.vacate(top)) return kBusy;
      if (!b_.make_room(now - top)) return kBusy;
      if (!b_.release(top, g)) {
        if (!l_.groups) l_.n0 = n0;
        l_.groups += g;
        if (!give(g)) l_.owed += g;
        return kFailed;
      }
    }
    if (!l_.groups) l_.n0 = n0;
    l_.groups += g;
    return kOk;
  }

  // `g` groups back into the pool. false, with nothing changed, when the pool could not grow on some rank.
  bool give(int g) {
    if (g < 1) return true;
    const int left = l_.groups - g;
    if (left < 0) {
      std::fprintf(stderr, "model: lend: %d groups back of %d\n", g, l_.groups);
      std::abort();
    }
    const size_t to = l_.numbers_for(left), now = b_.numbers();
    if (to > now && !b_.grow_all(to - now)) {
#ifdef PICO_OLD_RULES
      std::fprintf(stderr, "model: the chunk pool's %zu more numbers: no memory\n", to - now);
      std::exit(3);  // without the rule: a window's undo and a failed grow kill the engine
#else
      return false;
#endif
    }
    l_.groups = left;
    if (!left) l_.n0 = 0;
    return true;
  }

  void owe(int g) { l_.owed += g; }

  // What is owed, asked for again: `memory_ok` is whether every rank has the memory for it. kOk: nothing owed any more;
  // kBusy: some rank's memory is still taken; kFailed: the pool could not grow, the groups stay owed.
  int settle(bool memory_ok) {
    if (l_.owed < 1) return kOk;
    if (!memory_ok) return kBusy;
    if (!give(l_.owed)) return kFailed;
    l_.owed = 0;
    return kOk;
  }

 private:
  B& b_;
  Ledger& l_;
};

// ---- the proof that a hand-over moved memory ----

// One rank's three independent readings at a point: the pool's handle count, MemAvailable, and the block's resident bytes.
struct Reading {
  uint64_t handles = 0, avail = 0, rss = 0;
};
enum Proof { kProofOk = 0, kProofHandles, kProofAvail, kProofPages };

// The tolerance of the MemAvailable readings: the null control's spread, never under the floor.
inline uint64_t tol_of(uint64_t null_spread) {
#ifdef PICO_OLD_RULES
  return null_spread;
#else
  return std::max(null_spread, kTolFloor);
#endif
}

// A (before the groups left) against B (after): the pool's handle count fell by exactly `expect_handles`; MemAvailable rose
// by at least `released_bytes - tol` (one sided: a rise above is normal). With tol at the floor this cannot discriminate
// at one or two groups: the handle count and the block's pages are the witnesses there.
inline int proof_ab(const Reading& a, const Reading& b, uint64_t expect_handles, uint64_t released_bytes, uint64_t tol) {
  if (a.handles < b.handles || a.handles - b.handles != expect_handles) return kProofHandles;
  const uint64_t rise = b.avail > a.avail ? b.avail - a.avail : 0;
  if (rise + tol < released_bytes) return kProofAvail;
  return kProofOk;
}

// C (after the block was committed and populated): MemAvailable is at least A less tol (the memory the groups returned is
// in the block now), and the block's resident bytes rose between B and C by the committed bytes.
inline int proof_c(const Reading& a, const Reading& b, const Reading& c, uint64_t committed_bytes, uint64_t tol) {
  if (c.avail + tol < a.avail) return kProofAvail;
  const uint64_t rise = c.rss > b.rss ? c.rss - b.rss : 0;
  if (rise + kPagesSlack < committed_bytes) return kProofPages;
  return kProofOk;
}

// A shrink's proof: the block's resident bytes fell by the decommitted bytes; the pool's handle count rose by exactly
// `expect_handles` once the groups came back.
inline int proof_shrink(uint64_t rss_before, uint64_t rss_after, uint64_t decommitted_bytes, uint64_t handles_before,
                        uint64_t handles_after, uint64_t expect_handles) {
  const uint64_t fall = rss_before > rss_after ? rss_before - rss_after : 0;
  if (fall + kPagesSlack < decommitted_bytes) return kProofPages;
  if (handles_after < handles_before || handles_after - handles_before != expect_handles) return kProofHandles;
  return kProofOk;
}

// ---- the Engram cache's sizes ----

// The index's table: entries of 4 bytes, a power of two at least twice the slots (engram/cache.h SlotTable).
inline uint64_t table_entries(uint64_t slots) {
  uint64_t n = 1;
  while (n < slots * 2) n <<= 1;
  return n;
}
// The index's bytes for `slots` slots: 8-byte keys, 1-byte reference bits, the table.
inline uint64_t index_bytes(uint64_t slots) { return slots ? slots * 9 + table_entries(slots) * 4 : 0; }

struct Sizes {
  uint64_t slots = 0;           // the cache's slots, the display reservation's included
  uint64_t ordinary_bytes = 0;  // the slots outside the display reservation
  uint64_t index_bytes = 0;
  uint64_t block_bytes = 0;     // ordinary slots + index: what the block commits
  int groups = 0;               // groups of the pool the block's bytes take
};
// `cache_mib` is the slots' bytes (the setting); the first `display_bytes / row_bytes` slots sit in the display reservation.
inline Sizes sizes(uint64_t cache_mib, uint64_t display_bytes, uint64_t row_bytes, uint64_t group_bytes) {
  Sizes s;
  s.slots = cache_mib * kMiB / row_bytes;
  const uint64_t split = display_bytes / row_bytes;
  s.ordinary_bytes = (s.slots > split ? s.slots - split : 0) * row_bytes;
  s.index_bytes = lend::index_bytes(s.slots);
  s.block_bytes = s.ordinary_bytes + s.index_bytes;
  s.groups = (int)((s.block_bytes + group_bytes - 1) / group_bytes);
  return s;
}

// ---- a resize request ----

enum PlanStatus { kPlanSame = 0, kPlanGrow, kPlanShrink, kPlanRefused };
struct Plan {
  int status = kPlanSame;
  uint64_t mib = 0;  // the size to go to (the cut one when `cut`)
  int groups = 0;    // its groups
  bool cut = false;  // smaller than asked for: the store cannot lend more, or the block's address space ends
};
// From `current_mib` to `want_mib`. `max_groups` is what the store can lend this cache (the pool's lendable groups less
// every other borrower's), `max_mib` the block's address space. A setting past either is cut to the largest size that
// fits (a start is never refused by the cache), except in the old rules, which refuse it.
inline Plan plan_resize(uint64_t want_mib, uint64_t current_mib, uint64_t display_bytes, uint64_t row_bytes, uint64_t group_bytes,
                        int max_groups, uint64_t max_mib) {
  Plan p;
  uint64_t mib = std::min(want_mib, max_mib);
  const auto fits = [&](uint64_t m) { return sizes(m, display_bytes, row_bytes, group_bytes).groups <= max_groups; };
  bool cut = mib != want_mib;
  if (!fits(mib)) {
#ifdef PICO_OLD_RULES
    p.status = kPlanRefused;
    return p;
#else
    uint64_t lo = 0, hi = mib;  // the largest size that fits; sizes grow with the setting
    while (lo < hi) {
      const uint64_t mid = lo + (hi - lo + 1) / 2;
      if (fits(mid)) lo = mid;
      else hi = mid - 1;
    }
    mib = lo;
    cut = true;
#endif
  }
  p.mib = mib, p.cut = cut, p.groups = sizes(mib, display_bytes, row_bytes, group_bytes).groups;
  p.status = mib == current_mib ? kPlanSame : mib > current_mib ? kPlanGrow : kPlanShrink;
  return p;
}

// ---- the retry back-off ----

// A failed attempt records its size, its time and the setting's generation: the same size is not tried again for kBackoffS
// seconds, or until the setting changes.
struct Backoff {
  bool has = false;
  uint64_t mib = 0, gen = 0;
  double at = 0;
  void fail(uint64_t m, double now, uint64_t setting_gen) { has = true, mib = m, at = now, gen = setting_gen; }
  bool blocked(uint64_t m, double now, uint64_t setting_gen) const {
#ifdef PICO_OLD_RULES
    (void)m, (void)now, (void)setting_gen;
    return false;
#else
    return has && m == mib && gen == setting_gen && now - at < kBackoffS;
#endif
  }
};

}  // namespace model::lend
