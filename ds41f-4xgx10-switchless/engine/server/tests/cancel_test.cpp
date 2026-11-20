// cancel.h against cases (host only, no GPU): the queues a client that left is swept from (waiting for a slot, held for
// its session, let go to run), and the connection's id set (a CANCEL counts only for an id in flight, so a long queue of
// pending ids never un-cancels an old one). Built with -DPICO_OLD_RULES the cases fail (exit 1): the sweep covers the
// waiting queue only, and the cancel set clears itself past 4096 marks. Exit 0 when every case passes.
#include <cstdint>
#include <cstdio>
#include <deque>
#include <set>
#include <vector>

#include "../cancel.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}
static void eq(long long got, long long want, const char* what) {
  ++g_n;
  if (got != want) ++g_bad, std::printf("FAIL: %s: %lld, want %lld\n", what, got, want);
}

struct Req {
  uint64_t id;
};

int main() {
  // ---- the sweep over the three queues
  {
    cancel::Set cs;
    for (uint64_t id = 1; id <= 9; ++id) cs.submit(id);
    std::deque<Req> wait = {{1}, {2}, {3}}, held = {{4}, {5}, {6}}, ready = {{7}, {8}, {9}};
    cs.cancel(2), cs.cancel(5), cs.cancel(6), cs.cancel(7), cs.cancel(9);
    std::vector<uint64_t> dropped;
    const int n = cancel::sweep_queues(
        wait, held, ready, [&](const Req& r) { return cs.cancelled(r.id); }, [&](const Req& r) { dropped.push_back(r.id), cs.forget(r.id); });
    eq(n, 5, "a cancelled request is dropped from each of wait_, held_ and ready_");
    check(wait.size() == 2 && wait[0].id == 1 && wait[1].id == 3, "wait_: the others stay, in order");
    check(held.size() == 1 && held[0].id == 4, "held_: the cancelled requests of a session in flight are dropped");
    check(ready.size() == 1 && ready[0].id == 8, "ready_: the cancelled requests let go to run are dropped");
    eq((long long)cs.marks(), 0, "dropped requests leave no cancel mark");
    eq((long long)cs.live(), 4, "and leave the live set");
  }

  // ---- an id that is not in flight is not marked
  {
    cancel::Set cs;
    cs.submit(1);
    cs.cancel(99);  // never sent
    check(!cs.cancelled(99), "a CANCEL for an id that was never in flight marks nothing");
    cs.forget(1);
    cs.cancel(1);  // ended already
    check(!cs.cancelled(1), "a CANCEL that arrives after the request ended marks nothing");
    eq((long long)cs.marks(), 0, "no marks left behind");
  }

  // ---- 4,098 requests pending, each cancelled: the oldest stays cancelled (a cap of 4,096 marks would clear it)
  {
    cancel::Set cs;
    const uint64_t n = 4098;
    for (uint64_t id = 1; id <= n; ++id) cs.submit(id);
    cs.cancel(1);
    for (uint64_t id = 2; id <= n; ++id) cs.cancel(id);
    check(cs.cancelled(1), "the oldest of 4,098 pending cancelled requests is still cancelled");
    check(cs.cancelled(n), "and the newest");
    eq((long long)cs.marks(), (long long)n, "every mark kept: no cap, no clear");
    for (uint64_t id = 1; id <= n; ++id) cs.forget(id);
    eq((long long)cs.marks() + (long long)cs.live(), 0, "all forgotten at their ends");
  }

  std::printf("%d cases, %d failed\n", g_n, g_bad);
  return g_bad ? 1 : 0;
}
