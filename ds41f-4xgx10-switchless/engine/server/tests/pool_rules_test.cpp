// model/pool_rules.h against a pool with seven finished lines holding its top numbers and their leftover reservations (157
// numbers reserved to nobody), the request's own conversation holding a tail number, a 543-number cut for a window and a
// running request with a reservation, and, for the store (--pool-free-gib as the sizing rule), against the serving start's reads: the
// count rounded to whole groups, the lines outside the store shortening it by cap less what they hold at the read, and the start
// check's refusals. Built twice: with -DPICO_OLD_RULES (each rule built without its change) the cases marked
// "red" fail; without it every case passes. Host only: no engine, no GPU.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../../model/pool_rules.h"

namespace {

using model::rules::Holder;
namespace rules = model::rules;

int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

struct Slot {
  int resv, held;
};

// the slots: seven finished lines (reservation past what they hold: 22 each = 154, and 3 more on the last = 157)
std::vector<Slot> finished_lines() {
  std::vector<Slot> v;
  for (int i = 0; i < 7; ++i) v.push_back({40, 18});
  v[6].resv += 3;
  return v;
}

int64_t reserved_of(const std::vector<Slot>& s) {
  return rules::unheld_sum((int)s.size(), [&](int t) { return s[t].resv; }, [&](int t) { return s[t].held; });
}

}  // namespace

int main() {
  {
    const int slot_at = 5000;  // entries from position 5,000 on are quarter slots
    const auto below = [&](int u) { return u > slot_at ? (u - slot_at + 999) / 1000 : 0; };
    check(rules::whole_below(60000, below) == slot_at, "the reuse after a stopped join stops at the first slot");
    check(rules::whole_below(3000, below) == 3000, "a reuse below every slot is kept");
    check(rules::whole_below(60000, [](int) { return 0; }) == 60000, "nothing split: the reuse is kept");
    check(rules::whole_below(0, below) == 0, "no reuse stays none");
  }
  const std::vector<int> none;

  // the finished lines: the server's finish() unreserves each (kStUnreserve), the model applies resv_after_finish
  {
    auto s = finished_lines();
    check(reserved_of(s) == 157, "fixture: the seven finished lines reserve 157 numbers before they are unreserved");
    for (Slot& x : s) x.resv = rules::resv_after_finish(x.resv);
    check(reserved_of(s) == 0, "red: a finished request's reservation reserves nothing for anyone");
  }

  // the window: 543 numbers cut, 650 free once the holders are vacated; the lines' reservations decide
  {
    auto s = finished_lines();
    for (Slot& x : s) x.resv = rules::resv_after_finish(x.resv);
    check(rules::window_room(650, 543, reserved_of(s)), "red: the window opens when the finished lines reserve nothing");
    check(!rules::window_room(600, 543, 157), "a window short of numbers by the reservations of running requests stays busy");
    check(rules::window_room(700, 543, 157), "free covering the cut and a running request's reservation opens it");
  }

  // the holders of the cut: the request's own conversation (a tail number), a stored one, a finished line's one (detached)
  {
    const std::vector<int> pinned = none, live = none, last = {100};
    const Holder own{100, -1, 1}, stored{5, -1, 9}, detached{6, -1, 3};
    check(!rules::holder_busy(own, pinned, live, last), "red: the command's own conversation (evict-last) does not block the window");
    check(!rules::holder_busy(stored, pinned, live, last), "a stored conversation never blocks");
    check(!rules::holder_busy(detached, pinned, live, last), "a detached line's conversation never blocks");
    const Holder running{7, 2, 4}, unit_own{-1, 0, 2}, idle_line{8, 3, 5};
    check(rules::holder_busy(running, pinned, {7}, last), "a running request's conversation blocks");
    check(rules::holder_busy(unit_own, pinned, live, last), "a unit's own session (a tool or test) blocks");
    check(rules::holder_busy(idle_line, pinned, live, last), "a session still on a line blocks the model (the server detaches idle lines first)");
    check(rules::holder_busy(stored, {5}, live, last), "a pinned session blocks");
  }

  // the order pieces go in: evict-last after every other, least recently used first, the lowest handle on a tie
  {
    const std::vector<int> last = {100};
    rules::VictimPick p;
    p.offer(5, 9, false), p.offer(6, 3, false), p.offer(7, 3, false), p.offer(100, 1, true);
    check(p.victim == 6, "red: the own conversation (used longest ago) goes after the others; a tie goes to the lowest handle");
    rules::VictimPick q;
    q.offer(100, 1, true);
    check(q.victim == 100, "the evict-last session goes when nothing else is left");
    rules::VictimPick e;
    check(e.victim == -1, "no session offered: none");
    (void)last;
  }

  // a restore beside a running request's reservation: 157 reserved, 100 free, nothing stored to give
  {
    const auto g0 = [] { return 0; };
    check(!rules::restore_may_take(100, 157, 1, g0), "red: a restore does not eat a running request's reservation");
    check(rules::restore_may_take(300, 157, 1, g0), "a restore takes numbers beyond the reservations");
    check(rules::restore_may_take(100, 157, 100, [] { return 157; }), "stored sessions that eviction gives back count toward a restore");
    check(!rules::covers(100, 157, 1, g0) && rules::covers(158, 157, 1, g0), "a bookmark's take keeps the reservations covered");
  }

  // the disk tier's file is full: an idle line's conversation gives up its images, a running one never does
  {
    const std::vector<int> pinned = none, live = {7};
    const Holder idle_line{8, 3, 5}, running{7, 2, 4}, self{9, -1, 1}, stored{5, -1, 2};
    check(rules::disk_victim_ok(idle_line, 9, pinned, live), "red: an idle line's conversation may give up its images");
    check(!rules::disk_victim_ok(running, 9, pinned, live), "a running request's conversation never does");
    check(!rules::disk_victim_ok(self, 9, pinned, live), "the session making room never does");
    check(rules::disk_victim_ok(stored, 9, pinned, live), "a stored conversation may");
    check(!rules::disk_victim_frees(idle_line, true), "red: a conversation on a line is never freed whole, only its images go");
    check(rules::disk_victim_frees(stored, true), "a stored conversation with a piece only on the disk is freed whole");
    check(!rules::disk_victim_frees(stored, false), "a stored conversation with every piece in the pool loses its images only");
  }

  // the pool's size from the engine's real start lines (a rank 0 Spark's journal, --pool-free-gib 10): the pool each chose, then the free
  // device / available host memory of ranks 0-3 in GiB rounded to 0.01. The engine's own inputs, from its constants:
  // a number's image is attn::kChunkEntries (8,192) x the four kv arrays' entry bytes (model/state.cu chunk_image, impl.h
  // kKvEntryBytes); the numbers made before sizing are one slot's lists at --max-ctx 4194304 (model/model.cu numbers_for:
  // per kv source ceil((upto / ratio + 1 + 64) / 8,192), ratio 2 for the first three and 1 for the fourth) plus the guard;
  // the Engram cache still to grow is --engram-cache-mib 4096 less the display reservation's 2,032 MiB (main.cpp).
  constexpr double G = 1073741824.0;
  const uint64_t img = 8192ull * (256 + 32 + 64 + 4);
  uint64_t have = 1;
  {
    const int max_ctx = 4194304, list_len = ((max_ctx + 64) / 8192 + 2 + 7) / 8 * 8;
    for (int k = 0; k < 4; ++k) {
      const int ratio = k < 3 ? 2 : 1, entries = (max_ctx - 1) / ratio + 1 + 64;
      have += std::min(list_len, (entries + 8191) / 8192);
    }
  }
  const uint64_t keep10 = 10ull << 30, cache2 = (4096ull - 2032) << 20;
  check(img == 2916352 && have == 1285, "fixture: a number's image is 2,916,352 bytes and 1,285 numbers exist before sizing");
  {
    struct Line {
      uint64_t pool;
      double r[4][2];  // free device / available host, GiB, as the journal prints them
    };
    const Line lines[] = {
        {5957, {{23.93, 24.71}, {31.16, 30.56}, {31.22, 30.62}, {31.26, 30.66}}},
        {7276, {{28.62, 28.29}, {31.25, 30.63}, {31.19, 30.58}, {31.21, 30.60}}},
        {7404, {{29.00, 28.64}, {31.19, 30.58}, {31.11, 30.50}, {31.24, 30.63}}},
        {7407, {{29.01, 28.65}, {31.22, 30.62}, {31.21, 30.60}, {31.22, 30.61}}},
        {7419, {{29.04, 28.68}, {31.25, 30.65}, {31.15, 30.54}, {31.21, 30.60}}},
        {7434, {{29.09, 28.72}, {31.26, 30.64}, {31.29, 30.67}, {31.24, 30.62}}},
        {7439, {{29.06, 28.73}, {31.24, 30.62}, {31.27, 30.64}, {31.28, 30.65}}},
        {7446, {{29.09, 28.75}, {31.27, 30.64}, {31.17, 30.54}, {31.29, 30.66}}},
        {7453, {{29.06, 28.77}, {31.23, 30.63}, {31.25, 30.64}, {31.25, 30.64}}},
        {7458, {{29.14, 28.79}, {31.20, 30.60}, {31.09, 30.49}, {31.17, 30.57}}},
        {7606, {{29.81, 29.19}, {31.24, 30.58}, {31.12, 30.49}, {31.26, 30.63}}},
        {7608, {{29.81, 29.19}, {31.06, 30.44}, {31.26, 30.63}, {31.28, 30.65}}},
        {7617, {{29.85, 29.22}, {31.32, 30.66}, {31.27, 30.64}, {31.28, 30.65}}},
        {7630, {{29.88, 29.25}, {31.31, 30.65}, {31.27, 30.64}, {31.28, 30.65}}},
    };
    // each figure is rounded to 0.01 GiB, so it is within 0.005 GiB (5.4 MB, 1.84 numbers) of the engine's reading; the offer
    // of the rank that sets the pool moves by that and by the floor to a whole number: the pool is within 2 numbers
    int rank0_sets = 0, within = 0, n = 0;
    uint64_t worst = 0;
    for (const Line& l : lines) {
      uint64_t pool = UINT64_MAX;
      int who = 0;
      for (int r = 0; r < 4; ++r) {
        const uint64_t o = rules::offer_numbers(have, (uint64_t)std::llround(l.r[r][0] * G), (uint64_t)std::llround(l.r[r][1] * G), keep10, 0, cache2, img);
        if (o < pool) pool = o, who = r;
      }
      rank0_sets += who == 0;
      const uint64_t diff = pool > l.pool ? pool - l.pool : l.pool - pool;
      within += diff <= 2, worst = std::max(worst, diff);
      ++n;
    }
    std::printf("start lines: %d, rank 0 sets the pool in %d, within 2 numbers in %d, largest difference %llu\n", n, rank0_sets, within,
                (unsigned long long)worst);
    check(rank0_sets == n, "rank 0's offer sets the pool in every start line");
    check(within == n, "the pool of every start line is reproduced within 2 numbers (the journal's 0.01 GiB rounding)");
    check(rules::offer_numbers(have, (uint64_t)(5 * G), (uint64_t)(28.75 * G), keep10, 0, cache2, img) == have,
          "less free than keep: no numbers beyond those made");
    check(rules::offer_numbers(have, (uint64_t)(29 * G), (uint64_t)(11 * G), keep10, 0, cache2, img) == have,
          "the Engram cache still to grow counts as used");
  }

  // ---- the store: rounding to whole groups, the lines outside it, the start check ----
  using rules::kHeap, rules::kEngramCache, rules::kFront, rules::kEncoder, rules::kOutQueue, rules::kFrames, rules::kIds, rules::kGather,
      rules::kDrift;
  const uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
  // the lines at their caps on rank 0, as the serving start sets them (heap 1 GiB, dsfront 840 MiB, the encoder 1,260 MiB, the output
  // queue 64 MiB, frames 400 MiB, ids 1,024 MiB, the gather buffer 6 MiB, drift 0.75 GiB); the workers hold the heap, the gather
  // buffer and drift
  auto caps = [&](rules::Ledger* l, uint64_t front_now, uint64_t heap_now) {
    for (int r = 0; r < 4; ++r) {
      l[r] = rules::Ledger{};
      l[r].set(kHeap, GiB, heap_now, true);
      l[r].set(kEngramCache, 0, 0, true);
      l[r].set(kGather, 6 * MiB, 0, true);
      l[r].set(kDrift, 768 * MiB, 0, false);
    }
    l[0].set(kFront, 840 * MiB, front_now, true);
    l[0].set(kEncoder, 1260 * MiB, 0, false);
    l[0].set(kOutQueue, 64 * MiB, 0, true);
    l[0].set(kFrames, 400 * MiB, 0, true);
    l[0].set(kIds, 1024 * MiB, 0, true);
  };
  const double read7617[4][2] = {{29.85, 29.22}, {31.32, 30.66}, {31.27, 30.64}, {31.28, 30.65}};
  rules::Ledger L[4];
  caps(L, 0, 0);
  uint64_t out0 = rules::outside_sum(L[0]);
  check(out0 == GiB + 840 * MiB + 1260 * MiB + 64 * MiB + 400 * MiB + 1024 * MiB + 6 * MiB + 768 * MiB, "fixture: rank 0's lines at their caps sum to the lines' caps");
  {
    // rounding to whole groups: every rank's count is a multiple of 64
    int round_ok = 0;
    uint64_t store = UINT64_MAX;
    for (int r = 0; r < 4; ++r) {
      const uint64_t o = rules::store_offer(have, (uint64_t)std::llround(read7617[r][0] * G), (uint64_t)std::llround(read7617[r][1] * G), keep10,
                                            rules::outside_sum(L[r]), 0, img);
      round_ok += o % 64 == 0;
      store = std::min(store, o);
    }
    check(round_ok == 4, "red: the store's count is a whole number of groups of 64 on every rank");
    // "left free": the smallest rank keeps `keep`, within the rounding, with every line at its cap
    uint64_t lf_min = UINT64_MAX;
    for (int r = 0; r < 4; ++r)
      lf_min = std::min(lf_min, rules::left_free((uint64_t)std::llround(read7617[r][0] * G), (uint64_t)std::llround(read7617[r][1] * G), 0,
                                                 (store - have) * img, L[r]));
    std::printf("left free with every line at its cap, smallest rank: %.3f GiB of 10 kept\n", lf_min / G);
    check(lf_min >= keep10, "red: with the store full and every line at its cap no rank keeps less than --pool-free-gib");
    check(lf_min - std::min(lf_min, keep10) < 64 * img + img, "the smallest rank keeps the 10 GiB within one group of rounding");
  }
  {
    // what a line holds at the read raises the store by exactly that: dsfront holding 1.78 GiB against the same line holding nothing
    const uint64_t d0 = (uint64_t)std::llround(29.85 * G), a0 = (uint64_t)std::llround(29.22 * G);
    rules::Ledger A[4], B[4];
    caps(A, 0, 0);
    caps(B, 1824 * MiB, 0);  // 1.78 GiB
    const uint64_t na = rules::offer_numbers(have, d0, a0, keep10, rules::outside_sum(A[0]), 0, img);
    const uint64_t nb = rules::offer_numbers(have, d0, a0, keep10, rules::outside_sum(B[0]), 0, img);
    const uint64_t held = 840 * MiB;  // a line holds at most its cap in the sum: only the held part of the cap comes back
    check(nb - na == held / img || nb - na == held / img + 1, "red: a line holding part of its cap at the read raises the store by exactly what is held");
    rules::Ledger C[4];
    caps(C, 400 * MiB, 0);
    const uint64_t nc = rules::offer_numbers(have, d0, a0, keep10, rules::outside_sum(C[0]), 0, img);
    check(nc - na == (400 * MiB) / img || nc - na == (400 * MiB) / img + 1, "red: dsfront holding 400 MiB of its 840 raises the store by those 400 MiB");
    check(rules::store_offer(have, d0, a0, keep10, rules::outside_sum(A[0]), 0, img) <
              rules::store_offer(have, d0, a0, keep10, 0, 0, img),
          "red: the lines outside the store shorten it");
  }
  // the start check
  {
    const uint64_t need = rules::need_min_numbers(1280, 14, 38, 15);
    check(need == 1280 + 29 * 64 + 38, "one conversation of 4M positions, 14 Engram groups, one image request and a 15-group window");
    rules::Ledger A[4];
    caps(A, 0, 0);
    const rules::StoreCheck ok = rules::store_check(A, need + 1, need);
    check(!ok.refuse, "a store that holds the minimum starts");
    const rules::StoreCheck small = rules::store_check(A, need - 1, need);
    check(small.refuse && small.why == rules::Refuse::TooSmall, "red: a store under conversation + Engram cache + image request + window is refused with the reason");
    rules::Ledger H[4];
    caps(H, 0, 0);
    H[2].now[kHeap] = 2 * GiB;  // a worker's heap over its cap at the read: the engine's own line
    const rules::StoreCheck own = rules::store_check(H, need + 100, need);
    check(own.refuse && own.why == rules::Refuse::OwnLineOver && own.rank == 2 && own.line == kHeap,
          "red: one of the engine's own lines over its cap at the read refuses the start and names the rank and the line");
    rules::Ledger F[4];
    caps(F, 3200 * MiB, 0);  // dsfront holding 3.1 GiB against a cap of 840 MiB: a neighbour, not the engine
    const rules::StoreCheck other = rules::store_check(F, need + 100, need);
    check(!other.refuse, "control: another process over its cap at the read does not refuse the start (passes on both rule sets)");
    check(rules::line_room(F[0], kFront) == 0 && rules::over_cap_lines(F).size() == 1 && rules::over_cap_lines(F)[0].second == kFront,
          "its outside term is 0 and the line is listed as over its cap");
    rules::StoreRead rd;
    for (int r = 0; r < 4; ++r) rd.dfree[r] = (uint64_t)std::llround(read7617[r][0] * G), rd.avail[r] = (uint64_t)std::llround(read7617[r][1] * G);
    const std::string t = rules::store_table(F, rd, have, 0, 6272, img, keep10, 14, 15, other);
    check(t.find("over its cap at the read: front-end process holds 3.12 GiB, cap 0.82 GiB") != std::string::npos &&
              t.find("left free with the store full, every line at its cap") != std::string::npos && t.find("(rank 0 only)") != std::string::npos,
          "the table prints the neighbour's overrun as a warning, the left-free row and the rank 0 lines");
    const std::string t2 = rules::store_table(H, rd, have, 0, 6272, img, keep10, 14, 15, own);
    check(t2.find("refused: rank 2's own line glibc heap holds 2.00 GiB at the read, more than its cap 1.00 GiB") != std::string::npos,
          "the table gives the refusal's reason");
    const std::string t3 = rules::store_table(A, rd, have, 0, need - 1, img, keep10, 14, 15, small);
    check(t3.find("refused: the store is") != std::string::npos, "the table gives the too-small reason");
    check(rules::front_cap_refusal(0) != nullptr && std::string(rules::front_cap_refusal(0)).find("set PICO_FRONT_MIB") != std::string::npos,
          "red: serve mode with no cap on the front-end is refused with the instruction to set PICO_FRONT_MIB");
    check(rules::front_cap_refusal(840) == nullptr, "a cap of 840 MiB starts");
    uint64_t proj[4];
    for (int r = 0; r < 4; ++r) proj[r] = rules::projected_available((uint64_t)(28.75 * G), (7446 - have) * img, A[r]);
    check(proj[0] < proj[1] && proj[1] == proj[3], "projected available memory: rank 0 below the workers by its own lines");

    rules::Ledger skew[4];
    caps(skew, 0, 0);
    skew[2].cap[kHeap] = 2 * GiB;                          // a worker's heap cap 1 GiB above the others
    skew[3].cap[kDrift] = 768 * MiB + 1024 * MiB;          // another line, a worker's drift above the rest
    skew[1].cap[kHeap] = GiB + 60 * MiB;                   // within 64 MiB: not flagged
    const std::vector<int> bad_lines = rules::unequal_lines(skew);
    check(bad_lines.size() == 2 && bad_lines[0] == kHeap && bad_lines[1] == kDrift,
          "red: a line whose cap differs by more than 64 MiB between ranks and is not rank 0's own is flagged");
    const std::string t4 = rules::store_table(skew, rd, have, 0, 6272, img, keep10, 14, 15, rules::store_check(skew, 6272, need));
    check(t4.find("DIFFERS") != std::string::npos, "red: the table marks the flagged line");
    check(t4.find("image process outside the engine") != std::string::npos, "an encoder process beside the engine has its line in the table");
    rules::Ledger noenc[4];
    for (int r = 0; r < 4; ++r) noenc[r] = skew[r];
    noenc[0].set(kEncoder, 0, 0, false);
    const std::string t5 = rules::store_table(noenc, rd, have, 0, 6272, img, keep10, 14, 15, rules::store_check(noenc, 6272, need));
    // with the engine encoding (--vision-weights) the media worker is the line: its cap comes off the store
    {
      check(rules::image_process_cap(true, 1260, 512) == 512 * MiB, "red: with vision weights the image-process line is the media worker's cap (--media-mib)");
      check(rules::image_process_cap(false, 1260, 512) == 1260 * MiB, "without vision weights the line is the encoder container's cap");
      rules::Ledger zero[4], media[4];
      for (int r = 0; r < 4; ++r) zero[r] = media[r] = skew[r];
      zero[0].set(kEncoder, rules::image_process_cap(true, 1260, 0), 0, false);
      media[0].set(kEncoder, rules::image_process_cap(true, 1260, 512), 0, false);
      media[0].image_what = 1;
      const uint64_t oz = rules::store_offer(have, (uint64_t)std::llround(read7617[0][0] * G), (uint64_t)std::llround(read7617[0][1] * G), keep10,
                                             rules::outside_sum(zero[0]), 0, img);
      const uint64_t om = rules::store_offer(have, (uint64_t)std::llround(read7617[0][0] * G), (uint64_t)std::llround(read7617[0][1] * G), keep10,
                                             rules::outside_sum(media[0]), 0, img);
      check(rules::outside_sum(media[0]) - rules::outside_sum(zero[0]) == 512 * MiB && om < oz,
            "red: with vision weights and --media-mib 512 the lines outside the store are 512 MiB higher and the offer lower than with 0");
      const std::string t6 = rules::store_table(media, rd, have, 0, 6272, img, keep10, 14, 15, rules::store_check(media, 6272, need));
      check(t6.find("(media worker)") != std::string::npos, "the table names the media worker");
    }
    check(t5.find("image process outside the engine") == std::string::npos, "no image process beside the engine: the table has no such line");
  }

  // ---- the split (--kv-split cache): the start table's line, the pool's one count, a quarter slot's return ----
  {
    namespace R = model::rules;
    check(R::split_buffer_bytes(false) == 0, "the split's line is 0 without --kv-split cache");
    const uint64_t kq = 729088, want = (1 + 8 + 2 * 4 * 8) * kq + 8 * 5 * 8 + 41500ull * 4 * 5 * 8;
    check(R::split_buffer_bytes(true) == want && want > 50000000ull && want < 70000000ull,
          "the split's buffers: a quarter, the batch's quarters to send, the exchange twice, the sums of a batch and of every slot at the pool's largest count (about 57 MiB)");
    check(std::string(R::kLines[R::kSplit].name).find("split") != std::string::npos && R::kLines[R::kSplit].kind == R::Kind::Counted &&
              R::kLines[R::kSplit].own && !R::kLines[R::kSplit].rank0_only,
          "red: the split's buffers are a line of the start table, a Counted line of the engine's own, on every rank (made before the read)");
    R::Ledger l;
    l.set(R::kSplit, R::split_buffer_bytes(true), 0, true);
    check(R::outside_sum(l) == 0, "the store is not shortened again by the split's buffers: the free-memory read already lacks them (Counted)");
    check(R::pool_have(10, 4, 7) == 21, "the pool's one count: free beside the reservations, the lines' reservations, and what eviction gives once");
    // a conversation whose pieces are quarter slots needs a whole number for each to come back (back = on the disk + split)
    check(R::request_want(10, 0, 0 + 6, 0, 0, 0) == 10 && R::request_want(10, 4, 0 + 6, 0, 0, 0) == 6,
          "red: a request on a conversation with 6 split lists: the want is the larger of the lists it needs and what must come back");
#ifndef PICO_OLD_RULES
    check(R::request_want(10, 0, 12, 0, 0, 0) == 12, "red: more split and disk lists than the request's reservation: the return is the want");
#endif
  }

  // ---- the pool's spare count is signed: the server's sum and the core's `covers` are one rule ----
  {
    // the moment of the stop: free 0, the running requests owed 8 numbers, stored conversations can give 7, a request wants 7.
    // The core refuses the read-back's first number (0 + 7 < 8 + 1); the server's count admitted the request (clamped: 0 + 7 >= 7).
    const int64_t free_n = 0, reserved = 8, giveable = 7, want = 7;
    check(!rules::covers(free_n, reserved, 1, [&] { return giveable; }), "the core: free 0, reserved 8, giveable 7: not even one number is covered");
    check(!rules::admissible(rules::spare(free_n, reserved) + giveable, want),
          "red: the server's count for the same pool: spare + giveable (-1) does not cover a want of 7 (the clamped count said 7)");
    check(rules::spare(3, 8) == -5 && rules::spare(8, 3) == 5 && rules::spare(4, 4) == 0, "the spare is free less reserved, signed");
    uint64_t seed = 99;
    const auto rnd = [&](int64_t n) {
      seed = seed * 6364136223846793005ull + 1442695040888963407ull;
      return (int64_t)((seed >> 33) % (uint64_t)n);
    };
    long pools = 0, over = 0, bad = 0;
    for (int i = 0; i < 4000; ++i) {
      const int64_t f = rnd(60), p = rnd(70), g = rnd(70), n = 1 + rnd(70);
      ++pools;
      over += p > f;
      if ((rules::spare(f, p) + g >= n) != rules::covers(f, p, n, [&] { return g; })) ++bad;
    }
    char what[160];
    std::snprintf(what, sizeof what, "red: over %ld random pools (%ld with reserved > free) spare + giveable >= n exactly when covers (%ld differ)", pools, over, bad);
    check(bad == 0 && over > 500, what);
  }

  {  // the read-back tripwire: a refused take, not a cut for sums
    check(!rules::readback_refused(428, 309, 0, true), "red: a read-back whose pieces were cut for their sums (budget 428, 309 asked, 0 back) is not a refused take");
    check(rules::readback_refused(7, 7, 0, false), "a refused take (budget 7, 7 asked, 0 back) is reported");
    check(!rules::readback_refused(428, 309, 309, false), "every list back: nothing to report");
    check(!rules::readback_refused(0, 309, 0, false), "no budget: nothing to report");
  }

  std::printf("%s\n", fails ? "pool_rules_test FAILED" : "pool_rules_test passed");
  return fails ? 1 : 0;
}
