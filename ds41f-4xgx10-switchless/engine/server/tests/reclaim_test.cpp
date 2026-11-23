// reclaim.h's decisions against tables of cases (host only, no GPU): which line gives way first, the loop that detaches lines
// while a consumer is short, what a window plan asks of the server, the window close, the live set, the reservation at a
// request's end, the bookmarks a run takes. Built with -DPICO_OLD_RULES the cases that describe the rule fail (exit 1): the
// window plan names every holder on a line as busy (the own conversation and its base pinned), the live set is every slot's
// session, a finished request keeps its reservation, the window timer ignores the requests that need the window.
// Exit 0 when every case passes.
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "../../model/pool_rules.h"
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

  // ---- the split: a request runs from its admission to its finish
  {
    check(reclaim::split_names_live(true, false), "split: a decoding request's session is named live");
    check(reclaim::split_names_live(true, true), "split: a paused request's session is named live (a split command runs where every decoder is paused)");
    check(!reclaim::split_names_live(false, false) && !reclaim::split_names_live(false, true), "split: a conversation no request is in flight on is split");
  }

  // ---- a unit of a return detaches only lines no request is in flight on
  {
    check(!reclaim::unit_may_detach(true, true), "red: a paused request's line is not detached by a read-back unit");
    check(!reclaim::unit_may_detach(true, false), "a running request's line is not detached");
    check(reclaim::unit_may_detach(false, false) && reclaim::unit_may_detach(false, true), "a line no request is in flight on may be");
    // detach_until over lines marked by it leaves the paused request's line: slot 1 is paused, slots 0 and 2 idle
    std::vector<reclaim::Line> l = {{0, 10, 30, 5, reclaim::unit_may_detach(false, false)}, {1, 11, 10, 5, reclaim::unit_may_detach(true, true)},
                                    {2, 12, 20, 5, reclaim::unit_may_detach(false, false)}};
    std::vector<int> gone;
    reclaim::detach_until(l, -1, [] { return true; }, [&](int t) { gone.push_back(t); });
    check(gone == std::vector<int>({2, 0}), "red: with the shortage never ending, the idle lines go (least recently used first) and the paused request's line stays");
  }

  // ---- the victim: least recently used idle line with something to give, ties to the lowest slot
  {
    std::vector<reclaim::Line> l = {{0, 10, 50, 5, true}, {1, 11, 20, 5, true}, {2, 12, 20, 5, true}, {3, 13, 5, 5, false},
                                    {4, 14, 1, 0, true}};
    eq(reclaim::pick_victim(l, -1), 1, "least recently used idle line; tie 20/20 to the lower slot; a running line (used 5) and an empty one (used 1) skipped");
    eq(reclaim::pick_victim(l, 1), 2, "the consumer's own line is skipped");
    for (auto& x : l) x.idle = false;
    eq(reclaim::pick_victim(l, -1), -1, "no idle line: none");
    // a line whose conversation is fully split holds no whole number and reserves none, yet its slots stand for numbers: still a victim
    namespace rules = model::rules;
    std::vector<reclaim::Line> sp = {{0, 20, 30, rules::line_committed(0, 0, 40), true}, {1, 21, 10, rules::line_committed(0, 0, 0), true}};
    eq(reclaim::pick_victim(sp, -1), 0, "red: an idle line whose conversation is fully split (40 slots = 10 numbers) is a victim; an empty one (older) is not");
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

  // ---- the window plan (lines of finished requests hold the numbers above the window's top)
  {
    World w;
    w.flying = {20, 21, 22};  // 20 runs; 21 is paused; 22 is a queued long prompt
    w.paused = {21};
    auto idle = [&](uint64_t id) { return !w.run(id); };
    // holders: 241's finished line (slot 3), the request's own conversation on a line (slot 5), a paused flight (slot 2),
    // a stored conversation on no line, a running request (20, slot 1), a queued long prompt (22, slot 6)
    const std::vector<reclaim::Holder> holders = {{241, 3}, {248, 5}, {21, 2}, {300, -1}, {20, 1}, {22, 6}};
    const reclaim::WindowActions a = reclaim::window_actions(holders, idle);
    check(a.detach == std::vector<int>({3, 5, 2}), "idle holders on lines, the own conversation and a paused flight among them, are detached");
    check(a.running == std::vector<uint64_t>({20, 22}), "the running request and the queued long prompt are named, never detached");
    // only idle holders: the window command may go
    const std::vector<reclaim::Holder> idle_only = {{241, 3}, {248, 5}};
    const reclaim::WindowActions b = reclaim::window_actions(idle_only, idle);
    check(b.running.empty() && b.detach.size() == 2, "idle blockers only: nothing to wait for");
    // the long prompt alone, whose own conversation holds the tail: the window waits on it
    const reclaim::WindowActions c = reclaim::window_actions({{22, 6}}, idle);
    check(c.detach.empty() && c.running == std::vector<uint64_t>({22}), "a queued long prompt whose conversation holds the tail: the window waits, no command");
  }

  // ---- the window close: the least recently used open window no request needs
  {
    std::vector<reclaim::Win> w = {{32, true, false, 100}, {64, true, false, 50}};
    eq(reclaim::window_to_close(w), 64, "both open and unused: the less recently used");
    w[1].in_use = true;
    eq(reclaim::window_to_close(w), 32, "the window a request needs is never closed");
    w[0].in_use = true;
    eq(reclaim::window_to_close(w), 0, "every open window needed: none");
    w[0].in_use = false, w[0].open = false;
    eq(reclaim::window_to_close(w), 0, "a window that is closed already is not a candidate");
  }

  // ---- the window timer: off by default, and never under a request of its factor
  {
    eq(reclaim::kWindowIdleS, 0, "--window-idle-s defaults to 0: no window is freed by time");
    check(!reclaim::timer_may_close(0, 1000, 0, false), "timer 0: never");
    check(reclaim::timer_may_close(300, 1000, 0, false), "timer on, idle long enough, no request of the factor: freed");
    check(!reclaim::timer_may_close(300, 1000, 0, true), "timer on, but a request of that factor exists: kept");
    check(!reclaim::timer_may_close(300, 100, 0, false), "timer on, not idle long enough: kept");
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
    // an idle line holding nothing and reserving nothing, whose 348 quarter slots stand for 87 numbers (committed): no reservation
    std::vector<reclaim::SlotNums> sp = {{87, 0, 0}, {16, 0, 0}};
    eq(reclaim::idle_reserves(sp), 0, "red: an idle split line (committed 87, held 0, reserved 0) is no reservation (the tripwire read 87)");
    std::vector<reclaim::SlotNums> rs = {{87, 0, 40}, {300, 120, 0}};
    eq(reclaim::idle_reserves(rs), 40, "a real reservation is one (40), whatever the slots stand for");
  }

  // ---- the bookmarks a decode run takes: one at each multiple of snap_every it crosses
  {
    eq(reclaim::run_snaps(1000, 64, 1024, 7), 1, "1000 + 64 crosses 1024");
    eq(reclaim::run_snaps(100, 64, 1024, 7), 0, "100 + 64 crosses none");
    eq(reclaim::run_snaps(1000, 2100, 1024, 7), 3, "1000 + 2100 crosses 1024, 2048, 3072");
    eq(reclaim::run_snaps(1024, 64, 1024, 0), 0, "starting on a multiple: that one is not taken again");
    eq(reclaim::run_snaps(1000, 64, 0, 7), 0, "no bookmarks when snap_every is 0");
    eq(reclaim::run_snaps(1020, 0, 1024, 7), 0, "no tokens, no bookmark");
  }

  // ---- the lines of 32 idle sessions, a window wanted: every holder on a line detaches, in at most 33 passes
  {
    World w;  // nothing in flight
    std::vector<reclaim::Holder> holders;
    for (int i = 0; i < 32; ++i) holders.push_back({(uint64_t)(500 + i), i});
    const reclaim::WindowActions a = reclaim::window_actions(holders, [&](uint64_t id) { return !w.run(id); });
    eq((long long)a.detach.size(), 32, "32 idle lines holding the tail: 32 detaches planned");
    check(a.running.empty(), "and nobody to wait for");
  }

  // a request that is not running takes nothing from one that is. The only room is a paused request's line: give_back gives 0 of it, the newcomer waits.
  {
    check(!reclaim::idle_for_taker(true, true), "red: a paused request's line is not idle for any taker (the old rule took it: idle for a reservation)");
    check(!reclaim::idle_for_taker(true, false) && reclaim::idle_for_taker(false, false) && reclaim::idle_for_taker(false, true), "a running request's line is not, a line no request is in flight on is");
    // scenario: a pool whose only room is slot 0's line, a paused decoder's (the oldest); slot 1 is a running one's. A taker asks for the room.
    std::vector<reclaim::Line> lines = {{0, 11, 5, 40, reclaim::idle_for_taker(true, true)}, {1, 12, 9, 40, reclaim::idle_for_taker(true, false)}};
    int gave = 0;
    int short_ = 40;
    reclaim::detach_until(lines, -1, [&] { return short_ > 0; }, [&](int t) { ++gave, short_ -= 40; (void)t; });
    eq(gave, 0, "red: give_back gives 0 of a running or paused request's line: the newcomer waits (the old rule took the paused decoder's)");
    lines[0].idle = true;  // an idle conversation's line beside them: it, and only it, goes
    gave = 0, short_ = 40;
    reclaim::detach_until(lines, -1, [&] { return short_ > 0; }, [&](int t) { ++gave, short_ -= 40; eq(t, 0, "the idle line is the one detached"); });
    eq(gave, 1, "an idle line completes the room: one detach");
  }

  // the standstill
  {
    check(!reclaim::ext_standstill(0, 1, 1, true, false, false), "red (control: today's rule prints): one waiter alone with image work pending: no standstill");
    check(reclaim::ext_standstill(0, 5, 5, false, false, false), "red (control: today's rule is silent when a line-less flight waits too): five holders all waiting for an extension, nothing to image: standstill");
    check(!reclaim::ext_standstill(1, 5, 4, false, false, false), "a holder that was taken: no standstill");
    check(!reclaim::ext_standstill(0, 5, 5, false, true, false) && !reclaim::ext_standstill(0, 5, 5, false, false, true), "a join running, or a prompt being read: no standstill");
    check(!reclaim::ext_standstill(0, 0, 0, false, false, false), "no holder at all: no standstill");
    check(reclaim::ext_blocks(0) && reclaim::ext_blocks(-3) && !reclaim::ext_blocks(1) && !reclaim::ext_blocks(500), "red (control: old rules always true): a refused extension blocks only when the reservation is used up");
    check(reclaim::ext_pick({10, 40, 25, 7}) == 1, "the pick: the waiter with the longest reply");
    check(reclaim::ext_pick({10, 40, 40, 7}) == 2, "ties go to the last in the rotation");
    check(reclaim::ext_pick({}) == -1, "none: -1");
    check(!reclaim::ext_end_due(1.9, 10) && reclaim::ext_end_due(2.0, 10), "red (control: never under old rules): not before 2 s, then due");
    check(!reclaim::ext_end_due(5, 1.9) && reclaim::ext_end_due(5, 2.0), "at most one every 2 s");
  }

  std::printf("%d cases, %d failed\n", g_n, g_bad);
  return g_bad ? 1 : 0;
}
