// The chunk pool's rules as pure functions (no engine state; tested by server/tests/pool_rules_test.cpp): which holder
// of a number blocks a window, in what order the store gives pieces up, what a reservation is worth once its request
// has finished, and what a restore or a bookmark may take beside the reservations of running requests. Everything in the
// pool that no running request uses is cache and gives way; these functions are where the model decides that. The store's
// size (store_offer) and the per-rank memory ledger with the start check are here too.
// A session is "running" when a command lists it in `live` (the server passes the running requests' sessions only).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
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

// The same with the reservations for image rows not yet taken (`img(t)`: numbers slot t's request will take for its rows):
// they are numbers no other consumer may use either.
template <class Resv, class Held, class Img>
int64_t unheld_sum(int n, Resv resv, Held held, Img img) {
  int64_t sum = unheld_sum(n, resv, held);
#ifdef PICO_OLD_RULES
  (void)img;  // without the rule: the rows' numbers are not reserved, so a bookmark or a restore may take them first
#else
  for (int t = 0; t < n; ++t) sum += std::max(0, (int)img(t));
#endif
  return sum;
}

// The free numbers beside the reservations, signed: where the running requests are owed more than is free (stored conversations back
// the difference, which the pool can give back), the count is negative, so that what eviction gives is not counted for the reservation
// a second time. `covers` is the same rule: `spare(free, reserved) + giveable >= n` exactly when `free + giveable >= reserved + n`.
// Old rules: clamped at 0, which counted the backing numbers again.
inline int64_t spare(int64_t free, int64_t reserved) {
#ifdef PICO_OLD_RULES
  return std::max<int64_t>(0, free - reserved);
#else
  return free - reserved;
#endif
}

// Whether the pool can spare `n` numbers beside the reservations: free ones, or free plus what evicting stored sessions gives
// (`giveable()`, asked only when free ones do not cover it). state.cu take_extra.
template <class Giveable>
bool covers(int64_t free, int64_t reserved, int64_t n, Giveable giveable) {
  if (free >= reserved + n) return true;
  return free + (int64_t)giveable() >= reserved + n;
}

// ---- a request on a conversation partly on the disk tier (serve.cpp mem_ok, room_for, unspill) ---------------------------------
// What a request needs from the pool, in numbers nobody holds yet. The conversation it continues (its own, or a fork's base below
// the fork point) holds lists in the pool already: they are credited, because the attach that follows the reservation makes them
// the slot's. The lists that are only on the disk must come back (the turn reads them). So
//   want = max(chunks_for - credit + cow, back) + image numbers          (cow: the copies a fork makes at its restart point)
// Window bookmarks on the disk are not in it: they come back only into what is spare. Old rules: the whole list count plus every
// piece of the record on the disk, lists and bookmarks (the lists counted twice), no credit, no cow.
// ONE function gives this to mem_ok, room_for / make_room and the give-backs, each reading the credit at its own moment.
inline int64_t list_credit(int64_t lists_in_pool) {
#ifdef PICO_OLD_RULES
  (void)lists_in_pool;
  return 0;
#else
  return lists_in_pool;
#endif
}

inline int64_t request_want(int64_t chunks_for, int64_t credit, int64_t back, int64_t bookmarks_on_disk, int64_t cow, int64_t img) {
#ifdef PICO_OLD_RULES
  (void)credit, (void)cow;
  return chunks_for + back + bookmarks_on_disk + img;
#else
  (void)bookmarks_on_disk;
  return std::max<int64_t>(chunks_for - credit + cow, back) + img;
#endif
}

// What the reservation still needs from the pool once the read-back is done (room_for, before kStReserve): the positions' lists
// less those the conversation holds (they are the slot's at the attach), and the image rows. The caller takes off the slot's own
// commitment.
inline int64_t room_want(int64_t chunks_for, int64_t credit, int64_t img) { return chunks_for - credit + img; }

// The pool's count of what it can give a request (serve.cpp mem_ok, mem_sums, own, give_back's short_by): the free numbers beside the
// reservations, the reservations of the lines the request may take (their reservations end with the detach: max(0, reservation - mapped) each),
// and what eviction can give back for the conversations of those lines counted as if no line held them (the core's giveable_joint:
// the numbers only those conversations name, once). One sum, so that a number two lines name is counted once. Old rules: the caller
// adds each idle line's commitment and then the giveable of the conversations no line holds (a shared number is counted by each).
inline int64_t pool_have(int64_t free_unreserved, int64_t reserves_of_lines, int64_t giveable_joint) {
  return free_unreserved + reserves_of_lines + giveable_joint;
}

// What a line holds or has reserved, whichever is more; a conversation split on the line counts the whole numbers its quarter slots
// stand for (4 slots = 1, rounded up), so that an idle line whose conversation is fully split still has something to give
// (reclaim::pick_victim reads `committed`). Old rules: the slots are not counted (a fully split line holds and reserves nothing).
inline int line_committed(int resv, int held, int slots) {
#ifdef PICO_OLD_RULES
  (void)slots;
  return std::max(resv, held);
#else
  return std::max(resv, held + (slots + 3) / 4);
#endif
}

// The split's buffers (state.cu EngineDev), made at the start with --kv-split cache BEFORE the free-memory read the store is sized from
// (Model::split_prepare; class Fixed), so the read is already short by them: a Counted line of the start table, kSplit, shown not
// summed. The buffers: one quarter's scratch, the exchange buffer of 8 pieces' four quarters twice (the all-gather's output and the
// piece-by-piece layout), 8 pieces' quarters to send, the sums of 8 pieces, and the five recorded sums of every quarter slot at the
// pool's largest count of numbers.
constexpr uint64_t kQuarterBytes = 729088, kJoinBatch = 8, kSplitMaxNumbers = 41500;
inline uint64_t split_buffer_bytes(bool on) {
  return on ? (1 + kJoinBatch + 2 * 4 * kJoinBatch) * kQuarterBytes + kJoinBatch * 5 * 8 + kSplitMaxNumbers * 4 * 5 * 8 : 0;
}
// The imager's scratch (state.cu EngineDev), made with the buffers above: one quarter's image block (a quarter and its
// trailer: 12 grains), one bookmark's bytes with its image block after them (`gather_bytes` is the rounded size of the larger bookmark, the end
// bookmark's image block 1,572,864 bytes more), the five sum words. Part of the same start-table line (kSplit); in every mode that has a disk tier.
constexpr uint64_t kImageBlockBytes = 786432, kEndImageBlockBytes = 1572864;
inline uint64_t imager_buffer_bytes(bool on, uint64_t gather_bytes) { return on ? kImageBlockBytes + gather_bytes + kEndImageBlockBytes + 5 * 8 : 0; }

// The count a join command (kStJoin) carries, from what is left of the join's budget (restore_budget, read once in the pass that makes the
// reservation; each command takes some of it): never 0, which on the wire means no limit. -1: nothing is left, no command is sent. The old
// rules sent 0 every time (the bookmarks' numbers then came out of what admission counted for the reservation).
inline int64_t join_command_count(int64_t left) {
#ifdef PICO_OLD_RULES
  (void)left;
  return 0;
#else
  return left > 0 ? left : -1;
#endif
}

// The numbers a read-back may take (kStUnspill's count): the lists that must come back, then what the admission found spare
// (have - want: mem_ok's own sums), at least 1. Old rules: 0, no limit.
inline int64_t restore_budget(int64_t back, int64_t have, int64_t want) {
#ifdef PICO_OLD_RULES
  (void)back, (void)have, (void)want;
  return 0;
#else
  return std::max<int64_t>(1, back + std::max<int64_t>(0, have - want));  // never 0: that would mean no limit
#endif
}

// Whether a read-back was refused a take the count paid for: it had a budget, fewer lists came back than were asked, and the
// shortfall is not a cut for sums (pieces whose sums differ are cut by the core, and say so in its own lines). Old rules: the
// last term is missing, so a cut for sums reads as a refused take.
inline bool readback_refused(int64_t cap, int64_t asked, int64_t back, bool sums_cut) {
#ifdef PICO_OLD_RULES
  (void)sums_cut;
  return cap > 0 && back < asked;
#else
  return cap > 0 && back < asked && !sums_cut;
#endif
}

// The numbers a join before an operation may take (kStJoin's budget, read once): the lists that are split and the numbers the
// conversation's bookmarks stand on (the operation reads the restart point's bookmark whole, and a join that stops short of it
// leaves the core's own join to land inside the operation), then what the admission found spare.
inline int64_t join_budget(int64_t split, int64_t bookmarks, int64_t have, int64_t want) {
  return restore_budget(split + std::max<int64_t>(0, bookmarks), have, want);
}

// The idle engine's rescue, step 1 (serve.cpp rescue): which of the request's own conversation's window bookmarks to drop to
// make it fit. `numbers` = the numbers each holds in the pool, oldest first (0: only on the disk, so dropping it frees nothing).
// Only when everything those bookmarks hold would make it fit (have + their sum >= want) is anything dropped; then the oldest
// that hold numbers, until it fits. Otherwise nothing: a request that ends in an error costs its conversation no restart
// point. Returns indices into `numbers`. Old rules: oldest first whatever they hold, until it fits or none is left.
inline std::vector<size_t> rescue_drops(int64_t have, int64_t want, const std::vector<int>& numbers) {
  std::vector<size_t> drop;
  if (have >= want) return drop;
#ifndef PICO_OLD_RULES
  int64_t all = 0;
  for (const int n : numbers) all += n;
  if (have + all < want) return drop;
#endif
  for (size_t i = 0; i < numbers.size() && have < want; ++i) {
#ifndef PICO_OLD_RULES
    if (numbers[i] <= 0) continue;
#endif
    drop.push_back(i);
    have += numbers[i];
  }
  return drop;
}

// Whether the pool can hold the request now: what is free and what eviction gives back, against what it needs.
inline bool admissible(int64_t have, int64_t want) { return have >= want; }

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


// ---- the store's size and the memory ledger ----

constexpr uint64_t kGroupNums = 64;  // numbers in a group (model/window.cu): the store is a whole number of them

// One rank's count of numbers when the store leaves `keep` bytes free of device memory and of available host memory, and the
// lines outside the store (`outside`: for each, what it may still take beyond what it holds at the read, outside_sum) are
// at their caps. `counted`: memory the store's own offer is already short of (a cache still to grow in ordinary memory,
// counted as used on the host side only). `have` numbers exist already and `number_bytes` is one number's image.
// offer_numbers is the count before rounding; store_offer rounds it down to whole groups, so that the first group a
// borrower takes out leaves the rest of the store whole.
inline uint64_t offer_numbers(uint64_t have, uint64_t dfree, uint64_t avail, uint64_t keep, uint64_t outside, uint64_t counted,
                              uint64_t number_bytes) {
  const uint64_t by_device = dfree > keep ? dfree - keep : 0;
  const uint64_t by_host = avail > keep + counted ? avail - keep - counted : 0;
  const uint64_t room = std::min(by_device, by_host);
#ifdef PICO_OLD_RULES
  (void)outside;  // without the rule: the lines outside the store sit inside `keep`, the store is not shortened for them
  return have + room / number_bytes;
#else
  return have + (room > outside ? room - outside : 0) / number_bytes;
#endif
}
inline uint64_t store_offer(uint64_t have, uint64_t dfree, uint64_t avail, uint64_t keep, uint64_t outside, uint64_t counted,
                            uint64_t number_bytes) {
  const uint64_t n = offer_numbers(have, dfree, avail, keep, outside, counted, number_bytes);
#ifdef PICO_OLD_RULES
  return n;  // without the rule: any count
#else
  return n / kGroupNums * kGroupNums;
#endif
}

// The ledger: what a rank holds or may grow to outside the store, one line each, in bytes, from the rank's own counters.
// A Counted line is inside the free-memory reads the store is sized from: it is shown, not summed. An Outside line is memory that
// is taken or grows after sizing under its own cap: the store is shortened by what it may still take (cap less what it holds at the
// read) on that line's rank, so that the memory kept free stays free with every line at its cap. A line is `own` when it is the
// engine's own memory (over its cap at the read is a fault); another process's line over its cap is a warning and takes nothing
// more off the store. `enforced`: something holds the line to its cap; otherwise it is shown as observed plus margin.
// A line that only rank 0 holds is named so (rank0_only); any other line must agree between ranks. To add a line: one
// name in Line, one entry in kLines.
enum Line { kHeap, kEngramCache, kFront, kEncoder, kOutQueue, kFrames, kIds, kGather, kDrift, kSplit, kLineCount };
enum class Kind : uint8_t { Counted, Outside };
struct LineInfo {
  const char* name;
  Kind kind;
  bool rank0_only;
  bool own;
};
inline constexpr LineInfo kLines[] = {
    {"glibc heap", Kind::Outside, false, true},
    {"Engram cache still to grow", Kind::Counted, false, true},
    {"front-end process", Kind::Outside, true, false},
    {"image process outside the engine", Kind::Outside, true, false},
    {"front-end output queue", Kind::Outside, true, true},
    {"request frames (frame + rows)", Kind::Outside, true, true},
    {"server token ids", Kind::Outside, true, true},
    {"state gather buffer", Kind::Outside, false, true},
    {"kernel drift", Kind::Outside, false, false},
    {"the split's buffers (--kv-split cache)", Kind::Counted, false, true},
};
static_assert(sizeof kLines / sizeof kLines[0] == kLineCount, "one entry in kLines for every Line");
constexpr int kLedgerSlots = 16;  // the exchanged width of a ledger: room for lines to come
static_assert(kLineCount <= kLedgerSlots, "kLedgerSlots is too small for the lines");
constexpr int kRanks = 4;
constexpr uint64_t kLineTol = 64ull << 20;  // two ranks' caps of one line may differ by this much

struct Ledger {
  uint64_t cap[kLedgerSlots] = {};  // what the line may hold
  uint64_t now[kLedgerSlots] = {};  // what it holds at the read of free memory
  uint8_t enforced[kLedgerSlots] = {};
  uint8_t image_what = 0;  // what the image-process line counts, named in the table (the ledger is exchanged by memory): 0 none, 1 media worker, 2 encoder container
  void set(int i, uint64_t cap_bytes, uint64_t now_bytes, bool enforce) { cap[i] = cap_bytes, now[i] = now_bytes, enforced[i] = enforce; }
};

// The cap of the image process outside the engine (line kEncoder): with the encoder in the engine (--vision-weights) the media worker
// (tools/vision/media_worker.py), whose container's memory cap is `media_mib`; otherwise the encoder container (`encoder_mib`).
inline uint64_t image_process_cap(bool engine_encodes, uint64_t encoder_mib, uint64_t media_mib) {
#ifdef PICO_OLD_RULES
  return engine_encodes ? 0 : encoder_mib << 20;  // without the rule: the media worker is no line
#else
  return (engine_encodes ? media_mib : encoder_mib) << 20;
#endif
}

// What the line may still take beyond what it holds at the read (0 for a line over its cap).
inline uint64_t line_room(const Ledger& l, int i) { return l.cap[i] > l.now[i] ? l.cap[i] - l.now[i] : 0; }

// The bytes the store is shortened by on a rank: every Outside line's room.
inline uint64_t outside_sum(const Ledger& l) {
  uint64_t s = 0;
  for (int i = 0; i < kLineCount; ++i)
    if (kLines[i].kind == Kind::Outside) s += line_room(l, i);
  return s;
}
// The same with every line at its cap, whatever it holds (a worst case for the projection).
inline uint64_t outside_cap_sum(const Ledger& l) {
  uint64_t s = 0;
  for (int i = 0; i < kLineCount; ++i)
    if (kLines[i].kind == Kind::Outside) s += l.cap[i];
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

// The lines of other processes that hold more than their cap at the read: (rank, line). They take nothing more off the
// store; the table prints them and the watch repeats them.
inline std::vector<std::pair<int, int>> over_cap_lines(const Ledger* all, int n = kRanks) {
  std::vector<std::pair<int, int>> v;
  for (int r = 0; r < n; ++r)
    for (int i = 0; i < kLineCount; ++i)
      if (kLines[i].kind == Kind::Outside && !kLines[i].own && all[r].now[i] > all[r].cap[i]) v.push_back({r, i});
  return v;
}

// The start check. Refused: (1) the store holds less than `need_min` numbers (one conversation of --max-ctx, the Engram
// cache's groups, one largest image request, the largest window a request may open); (2) one of the engine's own lines
// holds more than its cap at the read. Another process's line over its cap is not a refusal (over_cap_lines).
enum class Refuse : uint8_t { None, TooSmall, OwnLineOver };
struct StoreCheck {
  bool refuse = false;
  Refuse why = Refuse::None;
  int rank = 0, line = -1;
  uint64_t store = 0, need = 0;
};
inline StoreCheck store_check(const Ledger* all, uint64_t store, uint64_t need_min, int n = kRanks) {
  StoreCheck c;
  c.store = store, c.need = need_min;
#ifndef PICO_OLD_RULES
  for (int r = 0; r < n && !c.refuse; ++r)
    for (int i = 0; i < kLineCount; ++i)
      if (kLines[i].kind == Kind::Outside && kLines[i].own && all[r].now[i] > all[r].cap[i]) {
        c.refuse = true, c.why = Refuse::OwnLineOver, c.rank = r, c.line = i;
        return c;
      }
  if (store < need_min) c.refuse = true, c.why = Refuse::TooSmall;
#endif  // without the rule: every combination starts
  return c;
}

// Serve mode needs a cap on the front-end process: without one nothing holds that line and the store cannot be sized
// beside it. The message that refuses the start, or null.
inline const char* front_cap_refusal(int front_mib) {
#ifdef PICO_OLD_RULES
  (void)front_mib;
  return nullptr;  // without the rule: an unset cap starts
#else
  return front_mib > 0 ? nullptr : "the front-end has no memory cap: set PICO_FRONT_MIB (the engine's --front-mib, dsfront's --front-mib)";
#endif
}

// What one rank has left once the store has grown to its size, the Engram cache has filled and the lines are at their caps.
inline uint64_t projected_available(uint64_t avail, uint64_t pool_growth_bytes, const Ledger& l) {
  const uint64_t used = pool_growth_bytes + l.cap[kEngramCache] + outside_cap_sum(l);
  return avail > used ? avail - used : 0;
}
constexpr uint64_t kProjectedFloor = 2ull << 30;  // a rank projected below this much available memory is warned about

// What a rank keeps free with the store full and every line at its cap: the smaller of the two reads, less the store's
// growth and the lines' room.
inline uint64_t left_free(uint64_t dfree, uint64_t avail, uint64_t counted, uint64_t growth_bytes, const Ledger& l) {
  const uint64_t room = std::min(dfree, avail > counted ? avail - counted : 0);
  const uint64_t used = growth_bytes + outside_sum(l);
  return room > used ? room - used : 0;
}

// The numbers the store must hold to start: one conversation, the Engram cache's groups, one largest image request's rows
// and the largest window a request may open (each in numbers; the groups of 64).
inline uint64_t need_min_numbers(uint64_t conversation, uint64_t engram_groups, uint64_t image_numbers, uint64_t window_groups) {
  return conversation + (engram_groups + window_groups) * kGroupNums + image_numbers;
}

// The table printed once after sizing (GiB by rank 0-3), the lines marked, the citizens, then any refusal's reason.
struct StoreRead {
  uint64_t dfree[kRanks] = {}, avail[kRanks] = {};
};
inline std::string store_table(const Ledger* all, const StoreRead& rd, uint64_t have, uint64_t counted, uint64_t store, uint64_t number_bytes,
                               uint64_t keep, uint64_t citizens_engram_groups, uint64_t citizens_window_groups, const StoreCheck& c,
                               int n = kRanks) {
  constexpr double gib = 1073741824.0, mib = 1048576.0;
  char b[512];
  std::string o;
  std::snprintf(b, sizeof b, "store: %llu numbers of %.2f MiB (%.2f GiB) on every rank, whole groups of %llu (the smallest rank's count), GiB by rank 0-%d:\n",
                (unsigned long long)store, number_bytes / mib, store * (double)number_bytes / gib, (unsigned long long)kGroupNums, n - 1);
  o += b;
  auto row = [&](const char* label, auto f) {
    std::snprintf(b, sizeof b, "  %-46s", label);
    o += b;
    for (int r = 0; r < n; ++r) {
      std::snprintf(b, sizeof b, " %7.2f", f(r) / gib);
      o += b;
    }
    o += "\n";
  };
  o += "  free device / available host at the read       ";
  for (int r = 0; r < n; ++r) {
    std::snprintf(b, sizeof b, " %.2f / %.2f", rd.dfree[r] / gib, rd.avail[r] / gib);
    o += b;
  }
  o += "\n";
  row("kept free (--pool-free-gib)", [&](int) { return (double)keep; });
  row("outside the store (cap - held at the read)", [&](int r) { return (double)outside_sum(all[r]); });
  const std::vector<int> bad = unequal_lines(all, n);
  for (int i = 0; i < kLineCount; ++i) {
    if (i == kEncoder && all[0].cap[i] == 0 && all[0].now[i] == 0) continue;  // no image process beside the engine: the line does not exist
    std::string nm = kLines[i].name;
    if (i == kEncoder && all[0].image_what) nm += all[0].image_what == 1 ? " (media worker)" : " (encoder container)";
    std::snprintf(b, sizeof b, "    %-30s %s", nm.c_str(), kLines[i].kind == Kind::Counted ? "counted in the store's offer" : "outside the store");
    o += b;
    // cap - held at the read, as rank 0 reads it (the ranks agree on the caps; a line only rank 0 holds is rank 0's)
    std::snprintf(b, sizeof b, " [%.2f - %.2f%s]", all[0].cap[i] / gib, all[0].now[i] / gib, all[0].enforced[i] ? ", enforced" : ", not enforced");
    o += b;
    if (kLines[i].rank0_only) o += " (rank 0 only)";
    for (int x : bad)
      if (x == i) o += "  <- DIFFERS between ranks by more than 64 MiB";
    o += "\n";
  }
  for (const auto& rl : over_cap_lines(all, n)) {
    std::snprintf(b, sizeof b, "  over its cap at the read: %s holds %.2f GiB, cap %.2f GiB (rank %d; it takes nothing more off the store)\n",
                  kLines[rl.second].name, all[rl.first].now[rl.second] / gib, all[rl.first].cap[rl.second] / gib, rl.first);
    o += b;
  }
  row("left free with the store full, every line at its cap", [&](int r) {
    return (double)left_free(rd.dfree[r], rd.avail[r], counted, (store > have ? store - have : 0) * number_bytes, all[r]);
  });
  std::snprintf(b, sizeof b, "store citizens: Engram cache %llu groups (%llu numbers); windows %llu groups at most; KV and image rows the rest\n",
                (unsigned long long)citizens_engram_groups, (unsigned long long)(citizens_engram_groups * kGroupNums),
                (unsigned long long)citizens_window_groups);
  o += b;
  if (c.refuse) {
    if (c.why == Refuse::OwnLineOver)
      std::snprintf(b, sizeof b, "refused: rank %d's own line %s holds %.2f GiB at the read, more than its cap %.2f GiB\n", c.rank, kLines[c.line].name,
                    all[c.rank].now[c.line] / gib, all[c.rank].cap[c.line] / gib);
    else
      std::snprintf(b, sizeof b, "refused: the store is %llu numbers, under the %llu one conversation of --max-ctx, the Engram cache, one largest image request and the largest window need\n",
                    (unsigned long long)c.store, (unsigned long long)c.need);
    o += b;
  }
  return o;
}

// The restart a conversation can still reuse after a join that stopped for want of a number: the largest q <= `planned` with no
// list entry below it still a quarter slot (`split_below(u)` counts them, never falls as u grows). The store's guard would cut the
// conversation at its first slot inside the operation, so the reuse never goes past it.
template <class F>
int whole_below(int planned, F split_below) {
  int lo = 0, hi = planned;
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (split_below(mid) == 0) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

}  // namespace model::rules
