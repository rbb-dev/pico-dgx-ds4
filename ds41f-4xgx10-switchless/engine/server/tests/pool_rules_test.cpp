// model/pool_rules.h against a pool with seven finished lines holding its top numbers and their leftover reservations (157
// numbers reserved for nobody), the request's own conversation holding a tail number, a 543-number cut for a window and a
// running request with a reservation. Built twice: with -DPICO_OLD_RULES (each rule built without its change) the cases marked
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

  // the ledger of that start: production's lines against keep 10 GiB, slack 2 GiB
  {
    using rules::kHeap, rules::kEngramCache, rules::kFront, rules::kEncoder, rules::kOutQueue;
    const uint64_t slack = 2ull << 30;
    rules::Ledger all[4];
    for (int r = 0; r < 4; ++r) {
      all[r].bytes[kHeap] = 1ull << 30;
      all[r].bytes[kEngramCache] = cache2;
    }
    all[0].bytes[kFront] = 3200ull << 20, all[0].bytes[kEncoder] = 1260ull << 20;
    all[0].bytes[kOutQueue] = 64ull << 20;
    const rules::StartCheck ok = rules::start_check(all, slack, keep10);
    check(!ok.refuse && ok.worst == 0 && ok.need == (1ull << 30) + (3200ull << 20) + (1260ull << 20) + (64ull << 20) + slack,
          "the shipped lines need 7.4 GiB of the 10 kept: the start goes on");
    check(rules::unequal_lines(all).empty(), "lines only rank 0 holds are named, nothing is flagged");
    uint64_t proj[4];
    for (int r = 0; r < 4; ++r) proj[r] = rules::projected_available((uint64_t)(28.75 * G), (7446 - have) * img, all[r]);
    check(proj[0] < proj[1] && proj[1] == proj[3], "projected available memory: rank 0 below the workers by its own lines");

    rules::Ledger big[4];
    std::copy(all, all + 4, big);
    big[0].bytes[kFront] = 9ull << 30;  // a front-end table that may take 9 GiB
    const rules::StartCheck bad = rules::start_check(big, slack, keep10);
    check(bad.refuse && bad.worst == 0, "red: lines that do not fit in keep refuse the start and name rank 0");
    const std::string t = rules::ledger_table(big, proj, slack, keep10, bad);
    check(t.find("need 13.") != std::string::npos && t.find("on rank 0") != std::string::npos && t.find("front-end table") != std::string::npos,
          "the table names the largest rank, its need and the line");

    rules::Ledger skew[4];
    std::copy(all, all + 4, skew);
    skew[2].bytes[kHeap] = 2ull << 30;                  // a worker's heap 1 GiB above the others
    skew[3].bytes[kEngramCache] = 0;                    // a rank with its Engram cache grown already
    skew[1].bytes[kHeap] = (1ull << 30) + (60ull << 20);  // within 64 MiB: not flagged
    const std::vector<int> bad_lines = rules::unequal_lines(skew);
    check(bad_lines.size() == 2 && bad_lines[0] == kHeap && bad_lines[1] == kEngramCache,
          "red: a line that differs by more than 64 MiB between ranks and is not rank 0's own is flagged");
    const std::string t2 = rules::ledger_table(skew, proj, slack, keep10, rules::start_check(skew, slack, keep10));
    check(t2.find("DIFFERS") != std::string::npos, "red: the table marks the flagged line");
  }

  std::printf("%s\n", fails ? "pool_rules_test FAILED" : "pool_rules_test passed");
  return fails ? 1 : 0;
}
