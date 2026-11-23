// The admission count for a request whose own conversation is partly in the disk tier, and the read-back's budget (host
// only, no GPU). The sums are model/pool_rules.h's, which serve.cpp mem_ok and unspill use: want = what the request needs from
// the pool beside what its own conversation holds there (restore_want / fork_want, plus the image rows); have = what is free
// and unreserved and what eviction gives back, the request's own conversation and its base excepted (they are pinned); the
// read-back's budget is the lists that must come back and what is spare (restore_budget).
// Built with -DPICO_OLD_RULES the sums are the old ones (the whole list count plus every piece of the record on the disk,
// bookmarks included, the lists counted twice, and no limit on the read-back): the same cases are then the stall.
// Exit 0 when every case passes.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../model/pool_rules.h"

namespace rules = model::rules;

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}

// What the request continues from: its lists below its need (a fork: below its fork point p) in the pool and on the disk, the
// bookmarks' numbers on the disk, and a fork's copies at p; and what the pool can give.
struct Case {
  int64_t cf;           // Model::slot_chunks_for(s, need)
  int64_t in_pool;      // lists in the pool (pinned: not offered by state_giveable)
  int64_t on_disk;      // lists only on the disk
  int64_t bookmarks;    // bookmark numbers on the disk
  int64_t cow;          // a fork's copies at its restart point
  int64_t img;          // image rows' numbers
  int64_t have;         // Model::slot_uncommitted() + what eviction gives back (the request's own conversation excepted)
};

static int64_t want_of(const Case& c) {
  return rules::request_want(c.cf, rules::list_credit(c.in_pool), c.on_disk, c.bookmarks, c.cow, c.img);
}
static bool admitted(const Case& c) { return rules::admissible(c.have, want_of(c)); }
// room_for after a read-back that took `taken` numbers from what was free: every list of the conversation is in the pool then
static bool room_holds(const Case& c, int64_t taken) {
  const int64_t all_lists = c.in_pool + c.on_disk;
  return rules::admissible(c.have - taken, rules::room_want(c.cf, rules::list_credit(all_lists), c.img));
}
// the most numbers a read-back takes: the budget (the lists, then bookmarks up to it); with no limit (old rules) every piece
static int64_t most_taken(const Case& c) {
  const int64_t want = want_of(c);
  const int64_t budget = rules::restore_budget(c.on_disk, c.have, want);
  const int64_t all = c.on_disk + c.bookmarks;
  // a budget of 1 with nothing to read back and nothing spare is the "never 0" floor: a bookmark is 2 numbers or more, so none fits
  if (budget > 0 && c.on_disk + std::max<int64_t>(0, c.have - want) <= 0) return c.on_disk;
  return budget > 0 ? std::min(budget, all) : all;
}

int main() {
  // a recorded disk_test: pool 35, 9 needed, 5 lists in the pool, 4 on the disk, 22 bookmark numbers in the pool and 4 on the disk, 8 free
  const Case wr1{9, 5, 4, 4, 0, 0, 8};
  check(admitted(wr1), "disk_test: 9 needed, 5 lists held, 4 back, 8 free: admitted");
#ifndef PICO_OLD_RULES
  check(rules::restore_budget(wr1.on_disk, wr1.have, want_of(wr1)) == 8, "its read-back budget: the 4 lists and the 4 spare");
  check(want_of(wr1) == 4, "its want is the 4 lists that must come back");
#endif
  // the sequence: admitted, read back within the budget (every amount from the lists alone to the most), then room_for holds
  bool seq = admitted(wr1);
  for (int64_t t = wr1.on_disk; t <= most_taken(wr1); ++t) seq = seq && room_holds(wr1, t);
  check(seq, "(a) the sequence on a recorded disk_test's numbers: admitted, read back within the budget, room_for's sum holds");
  // image_link_test's wait: 19 needed, nothing on the disk, the conversation's 12 lists in the pool, 17 free
  check(admitted({19, 12, 0, 0, 0, 0, 17}), "image_link_test's own-conversation wait: 19 needed, 12 lists held: admitted");
  // image_link_test's fork: 19 needed, the base whole in the pool (12 of its lists below the fork point), 4 copies at the fork
  // point, 17 free: admitted with its base, not read cold
  const Case fork{19, 12, 0, 0, 4, 0, 17};
  check(admitted(fork), "(c) the fork of image_link's numbers is admitted with its base (not cold)");
  // a fork whose base lists are on the disk: they must come back
  check(!admitted({19, 2, 15, 0, 3, 0, 12}), "a fork whose base needs 15 lists back does not fit beside 12 free");
  // bookmarks on the disk are not in the want; a record that is only its lists resident needs nothing from the pool
  check(admitted({9, 9, 0, 6, 0, 0, 1}), "a record that is only its 9 lists, resident: admitted with 1 free");
  // a request larger than the pool is never admissible
  check(!admitted({40, 0, 0, 0, 0, 0, 35}), "40 numbers in a pool of 35: never admissible");
  // (b) a sweep: whenever the admission holds, room_for holds after ANY read-back within the budget
  {
    long cases = 0, bad = 0;
    for (int64_t cf = 3; cf <= 30; cf += 3)
      for (int64_t in = 0; in <= cf; in += 2)
        for (int64_t disk = 0; in + disk <= cf; disk += 2)
          for (int64_t bm = 0; bm <= 12; bm += 4)
            for (int64_t cow = 0; cow <= 3; ++cow)
              for (int64_t img = 0; img <= 2; ++img)
                for (int64_t have = 0; have <= 60; ++have) {
                  const Case c{cf, in, disk, bm, cow, img, have};
                  if (!admitted(c)) continue;
                  ++cases;
                  const int64_t lo = c.on_disk, hi = most_taken(c);
                  for (int64_t t = lo; t <= hi; ++t)
                    if (t <= have && !room_holds(c, t)) {
                      ++bad;
                      break;
                    }
                }
    char what[160];
    std::snprintf(what, sizeof what, "(b) the sweep: %ld admitted cases, room_for holds after any read-back within the budget (%ld do not)", cases, bad);
    check(bad == 0, what);
  }
  // the idle engine's rescue, step 1: which of its own bookmarks it drops (pool_rules.h rescue_drops)
  {
    // (a) it cannot fit even with every bookmark that holds numbers dropped: nothing is dropped, the request is answered
    const std::vector<size_t> a = rules::rescue_drops(5, 20, {2, 2, 3});
    check(a.empty(), "(a) a request that cannot fit with every bookmark dropped keeps all its bookmarks");
    // (b) bookmarks only on the disk free nothing: never dropped
    const std::vector<size_t> b = rules::rescue_drops(3, 5, {0, 0, 2, 0});
    bool only_held = !b.empty();
    for (const size_t i : b) only_held = only_held && i == 2;
    check(only_held, "(b) bookmarks that are only on the disk are never dropped");
    // (c) it stops at the first drop that makes it fit: the oldest that holds numbers
    const std::vector<size_t> c = rules::rescue_drops(5, 6, {0, 2, 2});
    check(c.size() == 1 && c[0] == 1, "(c) it stops at the first drop that makes it fit");
    // and several are dropped when one is not enough, oldest first
    const std::vector<size_t> d = rules::rescue_drops(4, 7, {2, 2, 2, 2});
    check(d.size() == 2 && d[0] == 0 && d[1] == 1, "two drops, oldest first, when one is not enough");
    check(rules::rescue_drops(9, 7, {2, 2}).empty(), "it fits already: nothing is dropped");
  }
  // ---- a conversation whose lists are quarter slots (--kv-split cache): each needs a whole number to come back into
  {
    // cf = 4 numbers for the request's positions, none of its lists in the pool or on the disk: 6 are slots; the pool can give 5
    Case c{4, 0, 0, 0, 0, 0, 5};
    const int64_t split = 6;
    const int64_t want_split = rules::request_want(c.cf, rules::list_credit(c.in_pool), c.on_disk + split, c.bookmarks, c.cow, c.img);
    const int64_t want_control = rules::request_want(c.cf, rules::list_credit(c.in_pool), c.on_disk, c.bookmarks, c.cow, c.img);  // the count that forgets the slots
    check(!rules::admissible(c.have, want_split), "red: 6 slots need 6 whole numbers; the pool gives 5: not admitted (the join would find no number)");
    check(rules::admissible(c.have, want_control), "control: counting only the lists on the disk admits it, and its join then stops for want of a number");
  }
  // ---- the join's count is the budget's, never 0 (0 on the wire is no limit)
  {
    const int64_t budget = rules::restore_budget(6, 9, 7);  // 6 lists must come back, 2 numbers spare for bookmarks
    check(budget == 8, "red: the budget is the lists then the spare (6 + 2)");
    check(rules::join_command_count(budget) == 8, "red: a command carries what is left, never 0");
    check(rules::join_command_count(0) == -1 && rules::join_command_count(-3) == -1, "red: nothing left: no command is sent");
    check(rules::join_command_count(budget - 8) == -1, "the budget spent: the next command is not sent");
  }
  // ---- the signed spare (WCM2's stop): the request the running requests' reservations leave no room for is not admitted ----
  {
    // 10 numbers for its positions, 3 of its lists in the pool, 7 only on the disk: it wants max(10 - 3, 7) = 7 from the pool
    const int64_t free_n = 0, reserved = 8, giveable = 7;
    const auto have_of = [&](int64_t f) { return rules::spare(f, reserved) + giveable; };  // the server's sum: spare + what eviction gives
    Case c{10, 3, 7, 0, 0, 0, have_of(free_n)};
#ifndef PICO_OLD_RULES
    check(want_of(c) == 7, "the request: 10 needed, 3 lists held, 7 back: it wants 7");
#endif
    check(!rules::admissible(have_of(free_n), 7), "red: free 0, 8 reserved, 7 giveable: not admitted (the clamped count admitted it)");
    // the sequence that ended in the die, with the old (clamped) count spelled out: admitted, the read-back is refused at its first
    // piece and cuts the conversation (0 held, 0 back, the 3 held lists freed), and the request then wants all 10 with free 3
    const auto clamped = [&](int64_t f) { return std::max<int64_t>(0, f - reserved) + giveable; };
    const bool admitted_clamped = rules::admissible(clamped(free_n), 7);
    const bool after_cut = rules::admissible(clamped(3), 10);
    check(admitted_clamped && !after_cut, "the old count: admitted (7 >= 7), then after the cut 0 + 0 < 10: no slot, the die");
    check(!rules::admissible(have_of(free_n), 7) && !rules::admissible(have_of(3), 10),
          "red: the signed count stops at the first step: nothing is read back, nothing is cut");
    // enough spare, and the request is admitted as before
    check(rules::admissible(rules::spare(20, reserved) + 0, 7), "free 20, 8 reserved: 12 spare, admitted");
  }
  std::printf("%d of %d cases passed\n", g_n - g_bad, g_n);
  return g_bad ? 1 : 0;
}
