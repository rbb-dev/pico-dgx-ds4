// The chunk pool's rules as pure functions (no engine state; tested by server/tests/pool_rules_test.cpp): which holder
// of a number blocks a window, in what order the store gives pieces up, what a reservation is worth once its request
// has finished, and what a restore or a bookmark may take beside the reservations of running requests. Everything in the
// pool that no running request uses is cache and gives way; these functions are where the model decides that. The pool's
// size (pool_offer) and the per-rank memory ledger with the start check are here too.
// A session is "running" when a command lists it in `live` (the server passes the running requests' sessions only).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace model::rules {

inline bool listed(const std::vector<int>& v, int k) { return std::find(v.begin(), v.end(), k) != v.end(); }

// A session that holds a chunk number: its handle (<= 0: a unit's own session of a tool or test), the unit holding it
// (-1: none), the clock of its last use.
struct Holder {
  int handle = 0, line = -1;
  uint64_t used = 0;
};

// Whether a holder blocks vacating numbers: a unit's own session, a session on a line, a pinned one or a running one.
// `last` is the evict-last list (the command's own conversation and its fork base): those give way too, after the rest.
inline bool holder_busy(const Holder& x, const std::vector<int>& pinned, const std::vector<int>& live, const std::vector<int>& last) {
#ifdef PICO_OLD_RULES
  // without the rule: a unit's own session, a session on a line, a pinned or a live one blocks, and the window command pins
  // its own conversation and its base (kStPin), so those in `last` count as pinned.
  return x.handle <= 0 || x.line >= 0 || listed(pinned, x.handle) || listed(live, x.handle) || listed(last, x.handle);
#else
  return x.handle <= 0 || x.line >= 0 || listed(pinned, x.handle) || listed(live, x.handle);
#endif
}

// The session that gives up a piece first. Offer the eligible sessions in ascending handle order (the order of
// Store::sess); `victim` is then the evict-last sessions after all others, the least recently used among them, the
// lowest handle on a tie (the same on every rank). -1: none offered.
struct VictimPick {
  int victim = -1;
  bool victim_last = false;
  uint64_t oldest = UINT64_MAX;
  void offer(int handle, uint64_t used, bool is_last) {
#ifdef PICO_OLD_RULES
    // without the rule: the smallest `used`, the lowest handle on a tie; no evict-last
    (void)is_last;
    if (used < oldest) victim = handle, oldest = used;
#else
    if (victim < 0 || (is_last != victim_last ? !is_last : used < oldest)) victim = handle, victim_last = is_last, oldest = used;
#endif
  }
};

// A slot's reservation once its request has finished (kStUnreserve): a finished request reserves nothing for anyone.
inline int resv_after_finish(int resv) {
#ifdef PICO_OLD_RULES
  return resv;  // without the rule: a kept line stays attached with its whole reservation until the slot is released
#else
  (void)resv;
  return 0;
#endif
}

// Numbers the slots' reservations still reserve (a reservation past what the slot's session holds), over n slots.
// state.cu take_extra's `reserved` and window.cu window_make_room's `unheld`: one sum.
template <class Resv, class Held>
int64_t unheld_sum(int n, Resv resv, Held held) {
  int64_t sum = 0;
  for (int t = 0; t < n; ++t) sum += std::max(0, (int)resv(t) - (int)held(t));
  return sum;
}

// Whether the pool can spare `n` numbers beside the reservations: free ones, or free plus what evicting stored sessions gives
// (`giveable()`, asked only when free ones do not cover it). state.cu take_extra.
template <class Giveable>
bool covers(int64_t free, int64_t reserved, int64_t n, Giveable giveable) {
  if (free >= reserved + n) return true;
  return free + (int64_t)giveable() >= reserved + n;
}

// Whether a restore from the disk tier may take `n` numbers for a piece.
template <class Giveable>
bool restore_may_take(int64_t free, int64_t reserved, int64_t n, Giveable giveable) {
#ifdef PICO_OLD_RULES
  // without the rule: a restore evicts while fewer than n are free; no reservation is counted
  (void)reserved;
  return free >= n || free + (int64_t)giveable() >= n;
#else
  return covers(free, reserved, n, giveable);
#endif
}

// Whether the window can be made once the holders are vacated: the numbers cut off the pool (`removed`) are free and
// the reservations still to be taken (`unheld`) stay covered. window.cu window_make_room.
inline bool window_room(int64_t free_after, int64_t removed, int64_t unheld) { return free_after >= removed + unheld; }

// The disk tier's file is full: whether session k may give up its images. A running or pinned session never does; the
// session making room (h) neither.
inline bool disk_victim_ok(const Holder& x, int h, const std::vector<int>& pinned, const std::vector<int>& live) {
#ifdef PICO_OLD_RULES
  // without the rule: live names every slot's session, so every conversation on a line is skipped
  return !(x.handle == h || x.handle <= 0 || listed(pinned, x.handle) || listed(live, x.handle) || x.line >= 0);
#else
  return !(x.handle == h || x.handle <= 0 || listed(pinned, x.handle) || listed(live, x.handle));
#endif
}

// Whether a victim of disk_take is freed whole (its pieces are only on the disk) or only loses its images. A
// conversation on a line is never freed: its pieces in the pool are the truth.
inline bool disk_victim_frees(const Holder& x, bool out) {
#ifdef PICO_OLD_RULES
  (void)x;
  return out;  // without the rule: a victim with a piece only on the disk is freed whole, on a line or not
#else
  return out && x.line < 0;
#endif
}


// ---- the pool's size and the memory ledger ----

// One rank's count of chunk numbers when the pool leaves `keep` bytes free of device memory and of available host memory
// (`cache`: the Engram cache still to grow in ordinary memory, counted as used). `have` numbers exist already and
// `number_bytes` is one number's image.
inline uint64_t pool_offer(uint64_t have, uint64_t dfree, uint64_t avail, uint64_t keep, uint64_t cache, uint64_t number_bytes) {
  const uint64_t by_device = dfree > keep ? dfree - keep : 0;
  const uint64_t by_host = avail > keep + cache ? avail - keep - cache : 0;
  return have + std::min(by_device, by_host) / number_bytes;
}

// The ledger: what a rank holds or may grow to outside the pool, one line each, in bytes, from the rank's own counters.
// A Counted line is inside the free-memory reads the pool is sized from (allocated before, or subtracted in the offer):
// it is shown, not summed. A Fixed line is memory that is taken or grows after sizing and is bounded by its own cap: the
// start check sums those and keeps them inside what the pool leaves free. A line that only rank 0 holds is named so
// (rank0_only); any other line must agree between ranks. To add a line: one name in Line, one entry in kLines.
enum Line { kHeap, kEngramCache, kFront, kEncoder, kOutQueue, kLineCount };
enum class Kind : uint8_t { Counted, Fixed };
struct LineInfo {
  const char* name;
  Kind kind;
  bool rank0_only;
};
inline constexpr LineInfo kLines[] = {
    {"glibc heap bound", Kind::Fixed, false},
    {"Engram cache still to grow", Kind::Counted, false},
    {"front-end table and request rows", Kind::Fixed, true},
    {"image encoder (device and host)", Kind::Fixed, true},
    {"front-end output queue", Kind::Fixed, true},
};
static_assert(sizeof kLines / sizeof kLines[0] == kLineCount, "one entry in kLines for every Line");
constexpr int kLedgerSlots = 16;  // the exchanged width of a ledger: room for lines to come
static_assert(kLineCount <= kLedgerSlots, "kLedgerSlots is too small for the lines");
constexpr int kRanks = 4;
constexpr uint64_t kLineTol = 64ull << 20;  // two ranks' values of one line may differ by this much

struct Ledger {
  uint64_t bytes[kLedgerSlots] = {};
};

inline uint64_t fixed_sum(const Ledger& l) {
  uint64_t s = 0;
  for (int i = 0; i < kLineCount; ++i)
    if (kLines[i].kind == Kind::Fixed) s += l.bytes[i];
  return s;
}

// The lines that differ by more than kLineTol between ranks and are not rank 0's own.
inline std::vector<int> unequal_lines(const Ledger* all, int n = kRanks) {
  std::vector<int> v;
#ifndef PICO_OLD_RULES
  for (int i = 0; i < kLineCount; ++i) {
    if (kLines[i].rank0_only) continue;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int r = 0; r < n; ++r) lo = std::min(lo, all[r].cap[i]), hi = std::max(hi, all[r].cap[i]);
    if (hi - lo > kLineTol) v.push_back(i);
  }
#else
  (void)all, (void)n;  // without the rule: no line is compared
#endif
  return v;
}

// The start check: the largest sum of fixed lines over the ranks plus `slack` (page cache the kernel gives back, slab)
// must fit in `keep`, the free memory the pool was sized to leave. `worst` is the rank with the largest sum.
struct StartCheck {
  uint64_t need = 0;
  int worst = 0;
  bool refuse = false;
};
inline StartCheck start_check(const Ledger* all, uint64_t slack, uint64_t keep, int n = kRanks) {
  StartCheck c;
  uint64_t top = 0;
  for (int r = 0; r < n; ++r) {
    const uint64_t s = fixed_sum(all[r]);
    if (s > top || r == 0) top = s, c.worst = r;
  }
  c.need = top + slack;
#ifndef PICO_OLD_RULES
  c.refuse = c.need > keep;
#endif  // without the rule: every combination starts
  return c;
}

// What a rank has left once the pool has grown to its size, the Engram cache has filled and the fixed lines are taken.
inline uint64_t projected_available(uint64_t avail, uint64_t pool_growth_bytes, const Ledger& l) {
  const uint64_t used = pool_growth_bytes + l.bytes[kEngramCache] + fixed_sum(l);
  return avail > used ? avail - used : 0;
}
constexpr uint64_t kProjectedFloor = 2ull << 30;  // a rank projected below this much available memory is warned about

// The table printed once after sizing (GiB), the flagged lines marked, then the check's line.
inline std::string ledger_table(const Ledger* all, const uint64_t* projected, uint64_t slack, uint64_t keep, const StartCheck& c,
                                int n = kRanks) {
  constexpr double gib = 1073741824.0;
  const std::vector<int> bad = unequal_lines(all, n);
  std::string o = "memory ledger, GiB per rank (F: fixed, taken after sizing and kept inside the free memory; C: already counted in the pool's offer):\n";
  char b[256];
  for (int i = 0; i < kLineCount; ++i) {
    std::snprintf(b, sizeof b, "  %-34s %c ", kLines[i].name, kLines[i].kind == Kind::Fixed ? 'F' : 'C');
    o += b;
    for (int r = 0; r < n; ++r) {
      std::snprintf(b, sizeof b, "%6.2f", all[r].bytes[i] / gib);
      o += b;
    }
    if (kLines[i].rank0_only) o += "  (rank 0 only)";
    for (int x : bad)
      if (x == i) o += "  <- DIFFERS between ranks by more than 64 MiB";
    o += "\n";
  }
  o += "  fixed lines summed                   ";
  for (int r = 0; r < n; ++r) {
    std::snprintf(b, sizeof b, "%6.2f", fixed_sum(all[r]) / gib);
    o += b;
  }
  o += "\n  projected available memory           ";
  for (int r = 0; r < n; ++r) {
    std::snprintf(b, sizeof b, "%6.2f", projected[r] / gib);
    o += b;
    if (projected[r] < kProjectedFloor) o += "(!)";
  }
  std::snprintf(b, sizeof b, "\n  need %.2f GiB on rank %d (fixed lines %.2f + slack %.2f) against %.2f GiB kept free\n", c.need / gib, c.worst,
                fixed_sum(all[c.worst]) / gib, slack / gib, keep / gib);
  o += b;
  return o;
}

}  // namespace model::rules
