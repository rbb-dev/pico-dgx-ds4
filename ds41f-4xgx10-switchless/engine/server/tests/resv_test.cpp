// A finished request's reservation and the admission arithmetic (host only, no GPU). The sums mirror model.cu
// (slot_committed = max(resv, held), slot_uncommitted = max(0, free - the sum of max(0, resv - held))) and serve.cpp's
// mem_ok (uncommitted, plus the committed numbers of the idle lines it may take). Over random sets of lines:
//   - once finish() zeroes the reservation of each finished line (reclaim::unreserve_at_finish), nothing is reserved while
//     nothing runs (the dashboard's reserved, the tripwire's sum), and
//   - mem_ok's count of what an admission can use is the same with the reservations zeroed as without wherever
//     the spare is signed: wherever the reservations exceed the free numbers too.
// The last block is the instrument's control: an admission that dropped the re-add of the idle lines' committed numbers
// is told apart from the real one by these same sums. Built with -DPICO_OLD_RULES (a finished line keeps its reservation)
// the "nothing reserved" cases fail (exit 1). Exit 0 when every case passes.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../model/pool_rules.h"
#include "../reclaim.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}

struct Line {
  int64_t resv = 0, held = 0;
  bool idle = false;  // finished (or paused): its conversation may be taken
};

static int64_t committed(const Line& l) { return std::max(l.resv, l.held); }
static int64_t reserved(const std::vector<Line>& v) {
  int64_t p = 0;
  for (const Line& l : v) p += std::max<int64_t>(0, l.resv - l.held);
  return p;
}
static int64_t uncommitted(int64_t free_nums, const std::vector<Line>& v) { return model::rules::spare(free_nums, reserved(v)); }
// mem_ok's `have` for an admission that takes an empty line and counts every idle line (the want is out of reach)
static int64_t have(int64_t free_nums, const std::vector<Line>& v) {
  int64_t h = uncommitted(free_nums, v);
  for (const Line& l : v)
    if (l.idle) h += committed(l);
  return h;
}
// the control: the same without the idle lines' re-add
static int64_t have_no_readd(int64_t free_nums, const std::vector<Line>& v) { return uncommitted(free_nums, v); }

static uint64_t g_seed = 12345;
static int64_t rnd(int64_t n) {
  g_seed = g_seed * 6364136223846793005ull + 1442695040888963407ull;
  return (int64_t)((g_seed >> 33) % (uint64_t)n);
}

int main() {
  int outside = 0, inside = 0, differs = 0;
  bool idle_zero_all = true, reserved_zero_all = true;
  for (int round = 0; round < 4000; ++round) {
    const int n = 1 + (int)rnd(32);
    std::vector<Line> before(n);
    for (Line& l : before) {
      l.held = rnd(300);
      l.idle = rnd(3) != 0;
      // a running line reserves up to its horizon; a finished one still carries the horizon it had (up to 131,072 positions)
      l.resv = l.held + rnd(120);
    }
    const int64_t free_nums = rnd(2000);
    std::vector<Line> after = before;
    for (Line& l : after)
      if (l.idle && reclaim::unreserve_at_finish()) l.resv = 0;  // finish() zeroes a finished line's reservation
    const bool clamped = free_nums < reserved(before);
    const int64_t hb = have(free_nums, before), ha = have(free_nums, after);
    if (!clamped) {
      ++outside;
      if (reclaim::unreserve_at_finish()) check(hb == ha, "outside the clamp: the admission's count is unchanged by the zeroing");
    } else {
      ++inside;
      if (reclaim::unreserve_at_finish()) check(ha == hb, "where the reservations exceed the free numbers: the count is unchanged by the zeroing (signed)");
      check(uncommitted(free_nums, before) == free_nums - reserved(before), "red: the spare is not clamped at 0 where the reservations exceed the free numbers");
    }
    if (have_no_readd(free_nums, before) != hb) ++differs;
    // nothing runs: every line's request has finished
    std::vector<Line> quiet = before;
    for (Line& l : quiet) {
      l.idle = true;
      if (reclaim::unreserve_at_finish()) l.resv = 0;
    }
    std::vector<reclaim::SlotNums> s;
    for (const Line& l : quiet) s.push_back({committed(l), l.held});
    if (reclaim::idle_reserves(s) != 0) idle_zero_all = false;
    if (reserved(quiet) != 0) reserved_zero_all = false;
  }
  check(outside > 500 && inside > 500, "the random sets cover both sides of the clamp");
  check(idle_zero_all, "nothing runs: no line is reserved numbers it does not hold (the tripwire's sum)");
  check(reserved_zero_all, "nothing runs: the dashboard's reserved is 0");
  check(differs > 500, "control: an admission that dropped the idle lines' re-add is told apart by these sums");

  std::printf("%d cases, %d failed\n", g_n, g_bad);
  return g_bad ? 1 : 0;
}
