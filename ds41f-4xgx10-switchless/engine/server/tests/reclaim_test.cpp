// reclaim.h's decisions against tables of cases (host only, no GPU): which line gives way first, the loop that detaches lines
// while a consumer is short, the live set, the reservation at a request's end. Built with -DPICO_OLD_RULES the cases that
// describe the rule fail (exit 1): the live set is every slot's session, a finished request keeps its reservation.
// Exit 0 when every case passes.
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "../reclaim.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}
static void eq(long long got, long long want, const char* what) {
  ++g_n;
  if (got != want) ++g_bad, std::printf("FAIL: %s: %lld, want %lld\n", what, got, want);
}

// The server's session sets as the test sees them: ids in flight, ids in the turns (paused)
struct World {
  std::set<uint64_t> flying, paused;
  bool run(uint64_t id) const { return reclaim::running(flying.count(id) > 0, paused.count(id) > 0); }
};

int main() {
  // ---- who is running
  {
    World w;
    w.flying = {1, 2, 3};
    w.paused = {2};
    check(w.run(1), "in flight and not paused: running");
    check(!w.run(2), "in flight and paused: not running (its line can be detached)");
    check(!w.run(4), "not in flight: not running");
    // a queued Long is in flying_ from its admission and in no turns: running, never detached, named when it blocks
    check(w.run(3), "a queued long prompt (in flight, in no turns) is running");
  }

  // ---- the victim: least recently used idle line with something to give, ties to the lowest slot
  {
    std::vector<reclaim::Line> l = {{0, 10, 50, 5, true}, {1, 11, 20, 5, true}, {2, 12, 20, 5, true}, {3, 13, 5, 5, false},
                                    {4, 14, 1, 0, true}};
    eq(reclaim::pick_victim(l, -1), 1, "least recently used idle line; tie 20/20 to the lower slot; a running line (used 5) and an empty one (used 1) skipped");
    eq(reclaim::pick_victim(l, 1), 2, "the consumer's own line is skipped");
    for (auto& x : l) x.idle = false;
    eq(reclaim::pick_victim(l, -1), -1, "no idle line: none");
  }

  // ---- the loop: lines detach until the consumer is no longer short, one per pass, then it stops
  {
    std::vector<reclaim::Line> l;
    for (int i = 0; i < 32; ++i) l.push_back({i, (uint64_t)(100 + i), (uint64_t)(1000 - i), 100, true});  // slot 31 is the oldest
    std::vector<int> order;
    long long freed = 0;
    int n = reclaim::detach_until(
        l, -1, [&] { return freed < 250; }, [&](int s) { order.push_back(s), freed += 100; });
    eq(n, 3, "short by 250 with 100 a line: three lines detach");
    check(order.size() == 3 && order[0] == 31 && order[1] == 30 && order[2] == 29, "oldest first");
    freed = 0, order.clear();
    n = reclaim::detach_until(
        l, -1, [&] { return freed < 100000; }, [&](int s) { order.push_back(s), freed += 100; });
    eq(n, 32, "32 idle lines, still short: 32 detaches, then it stops (no line left)");
    std::set<int> seen(order.begin(), order.end());
    eq((long long)seen.size(), 32, "no line detached twice");
    freed = 0;
    n = reclaim::detach_until(
        l, -1, [&] { return false; }, [&](int) { ++freed; });
    eq(n, 0, "not short: nothing detaches");
  }

  // ---- the live set: running sessions only, so an idle line's conversation is cache (disk file space, eviction)
  {
    check(reclaim::counts_as_live(true, false), "a running request's session is live");
    check(!reclaim::counts_as_live(false, false), "a finished request's session on its line is not live");
    check(!reclaim::counts_as_live(true, true), "a paused request's session is not live");
  }

  // ---- a finished request's reservation
  {
    check(reclaim::unreserve_at_finish(), "finish() zeroes the reservation of the line it keeps");
    // two lines finished with their horizon left over; a third is held by a running request
    std::vector<reclaim::SlotNums> s = {{300, 120}, {200, 200}, {50, 10}};
    eq(reclaim::idle_reserves(s), 180 + 0 + 40, "the reservations over the slots: committed beyond mapped");
    std::vector<reclaim::SlotNums> z = {{120, 120}, {200, 200}, {10, 10}};
    eq(reclaim::idle_reserves(z), 0, "nothing reserved once the reservations are zeroed");
  }

  std::printf("%d cases, %d failed\n", g_n, g_bad);
  return g_bad ? 1 : 0;
}
