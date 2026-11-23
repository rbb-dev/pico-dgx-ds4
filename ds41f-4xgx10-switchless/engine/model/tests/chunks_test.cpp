// model/chunks.h: lowest free number first, all-or-none multi-takes, counts, and a digest that is the same for the
// same taken set whatever the order (rank coherence), against a reference model over a million random operations;
// four simulated ranks replaying one stream agree; one that skips an operation does not (the negative control).
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "model/chunks.h"

using model::chunks::Alloc;
using model::chunks::SlotMap;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s\n", what);
  }
}
uint64_t digest_of(const std::set<uint32_t>& held) {
  uint64_t d = 0;
  for (const uint32_t c : held) d += model::chunks::mix(c);
  return d;
}
}  // namespace

int main() {
  {  // lowest first, the guard never taken, give makes a number the next again
    Alloc a(10);
    check(a.total() == 9 && a.free() == 9, "counts at the start");
    check(a.take() == 1 && a.take() == 2 && a.take() == 3, "lowest first from 1 (0 is the guard)");
    a.give(2);
    check(a.take() == 2, "a given number is the lowest again");
    std::vector<uint32_t> v;
    check(!a.take(7, &v) && v.empty(), "more than free: none taken");
    check(a.take(6, &v) && v.size() == 6 && v.front() == 4 && v.back() == 9, "n at once, lowest first");
    check(a.free() == 0 && a.held() == 9, "all held");
    check(!a.taken(0) && a.taken(5), "taken()");
    a.grow(70);  // 60 more, across a word
    check(a.free() == 60 && a.total() == 69 && a.take() == 10, "grow: new numbers free, lowest first from the old end");
    a.give(3);
    check(a.take() == 3 && a.take() == 11, "after grow, a given number and then the next new one");
  }
  {  // shrink: a window's numbers leave the pool when they are all free, and come back as free numbers
    Alloc a(200);
    std::vector<uint32_t> v;
    check(a.take(120, &v), "120 taken");  // 1..120
    const uint64_t d0 = a.digest();
    check(a.first_taken_from(100) == 100 && a.first_taken_from(121) == 0, "first_taken_from");
    check(!a.shrink(110) && a.numbers() == 200 && a.free() == 79, "RED: a shrink below a taken number changes nothing");
    check(a.shrink(121) && a.numbers() == 121 && a.total() == 120 && a.free() == 0 && a.digest() == d0,
          "shrink to the end of the taken: the free numbers leave, the digest stays");
    check(!a.shrink(1) && !a.shrink(300), "no shrink to the guard alone or to more");
    a.give(7);
    check(a.take() == 7 && a.free() == 0, "the numbers left are the old ones");
    for (uint32_t c = 100; c <= 120; ++c) a.give(c);
    check(a.shrink(100) && a.numbers() == 100 && a.free() == 0 && a.total() == 99, "shrink again, across a word edge");
    a.grow(260);
    check(a.numbers() == 260 && a.free() == 160, "grow after a shrink: every new number free");
    check(a.take() == 100 && a.take() == 101, "lowest first from the old end");
    check(!a.taken(259) && a.first_taken_from(102) == 0, "nothing taken past it");
  }
  {  // model-based: random operations against a reference
    const uint32_t n = 5000;
    Alloc a(n);
    std::set<uint32_t> free, held;
    for (uint32_t c = 1; c < n; ++c) free.insert(c);
    std::mt19937_64 rng(7);
    int bad = 0;
    for (int i = 0; i < 1000000 && !bad; ++i) {
      const int op = (int)(rng() % 10);
      if (op < 4 && !free.empty()) {
        const uint32_t want = *free.begin(), got = a.take();
        if (got != want) ++bad;
        free.erase(got), held.insert(got);
      } else if (op < 6) {
        const uint32_t k = (uint32_t)(rng() % 40);
        std::vector<uint32_t> v;
        const bool ok = a.take(k, &v);
        if (ok != (k <= free.size())) ++bad;
        for (const uint32_t c : v) {
          if (c != *free.begin()) ++bad;
          free.erase(c), held.insert(c);
        }
      } else if (!held.empty()) {
        auto it = held.begin();
        std::advance(it, rng() % held.size());
        const uint32_t c = *it;
        a.give(c);
        held.erase(c), free.insert(c);
      }
      if (a.free() != free.size() || a.held() != held.size() || a.free() + a.held() != a.total()) ++bad;
      if (i % 1000 == 0 && a.digest() != digest_of(held)) ++bad;
    }
    check(!bad, "a million operations equal the reference (lowest first, counts, all-or-none, digest)");
    check(a.digest() == digest_of(held), "the digest is the taken set's");
  }
  {  // four ranks replay one stream and agree; a rank that skips one operation differs (negative control)
    std::mt19937_64 rng(11);
    std::vector<std::pair<int, uint32_t>> stream;  // (0: take, 1: give the i-th held), ...
    for (int i = 0; i < 20000; ++i) stream.push_back({(int)(rng() % 3 == 0), (uint32_t)rng()});
    auto replay = [&](int skip) {  // a pool no take can fill, so every operation, the skipped one too, does something
      Alloc a(30000);
      std::vector<uint32_t> held;
      for (int i = 0; i < (int)stream.size(); ++i) {
        if (i == skip) continue;
        if (stream[i].first == 0 || held.empty()) {
          if (!a.free()) std::printf("FAIL the replay's pool filled: a skipped take could be a no-op\n"), ++g_fail;
          held.push_back(a.take());
        } else {
          const size_t j = stream[i].second % held.size();
          a.give(held[j]);
          held[j] = held.back(), held.pop_back();
        }
      }
      return a.digest();
    };
    const uint64_t d0 = replay(-1);
    check(replay(-1) == d0 && replay(-1) == d0 && replay(-1) == d0, "four ranks replaying one stream agree");
    check(replay(12345) != d0, "a rank that skips one operation differs (negative control)");
  }
  {  // ghosts and the take order with a lend zone: the zone is the numbers from 10 up in a pool of 14
    Alloc a(14);
    std::vector<uint32_t> v;
    check(a.take(13, &v), "all 13 taken");
    a.give_ghost(3, 5), a.give_ghost(12, 2), a.give_ghost(4, 9);  // oldest first: 12 (stamp 2), 3 (5), 4 (9)
    a.give(11);
    check(a.free() == 4 && a.ghosts() == 3 && a.free_plain() == 1, "free() counts ghosts; free_plain() does not");
    check(a.is_ghost(3) && !a.taken(3) && a.stamp(12) == 2 && a.stamp(11) == 0 && !a.is_ghost(11), "ghost, stamp");
    a.set_zone(10);
    check(a.free_below_zone() == 2, "free_below_zone counts ghosts below the zone (3 and 4)");
    check(a.take() == 3, "step 2: the oldest ghost below the zone (3, stamp 5) before a plain free number in the zone");
    check(a.take() == 4, "then the next ghost below the zone (4, stamp 9)");
    uint32_t c = 0;
    check(!a.take_below_zone(&c), "take_below_zone: none below the zone");
    check(a.take() == 11, "step 3: the lowest plain free number in the zone before a ghost in the zone");
    check(a.take() == 12 && a.free() == 0, "step 4: the ghost in the zone");
    // no zone: step 1 then 2
    Alloc b(10);
    check(b.take(9, &v), "b: 9 taken");
    v.clear();
    b.give_ghost(2, 1), b.give(5);
    check(b.take() == 5 && b.take() == 2, "without a zone: a plain free number first, then the oldest ghost");
  }
  {  // revive: a ghost taken back by name; digest() is the taken set's, a ghost is not in it
    Alloc a(20);
    std::vector<uint32_t> v;
    a.take(10, &v);
    const uint64_t d_all = a.digest();
    a.give_ghost(5, 7);
    check(a.digest() == d_all - model::chunks::mix(5), "a ghost is free for the digest()");
    check(a.split_digest() != 0, "split_digest sees the ghost");
    a.revive(5);
    check(a.taken(5) && a.digest() == d_all && a.free() == 9 && a.ghosts() == 0 && a.split_digest() == 0, "revive: taken again, no ghost, digest as before");
    check(a.take() == 11, "revive leaves the lowest-free bound right: the next take is the lowest free number");
    a.give_ghost(6, 8), a.give(7);
    check(a.take() == 7 && a.take() == 12 && a.take() == 13, "a ghost waits behind every plain free number");
    check(a.is_ghost(6), "and is still a ghost");
    a.give_ghost(8, 9);
    a.revive(8);
    a.give(9);
    check(a.take() == 9, "a revived number above a lower free one does not hide it");
  }
  {  // shrink cuts the ghosts it covers, and what is left is as if it had never been there
    Alloc a(100);
    std::vector<uint32_t> v;
    a.take(60, &v);
    a.give_ghost(40, 1), a.give_ghost(50, 2), a.give_ghost(55, 3);
    check(a.ghosts() == 3 && a.ghosts_from(50) == std::vector<uint32_t>({50, 55}), "three ghosts; ghosts_from lists those at or above");
    check(!a.shrink(52) && a.numbers() == 100 && a.ghosts() == 3, "RED: a shrink over taken numbers changes nothing, ghosts included");
    for (uint32_t x = 52; x <= 60; ++x)
      if (x != 55) a.give(x);
    check(a.shrink(52) && a.numbers() == 52 && a.ghosts() == 2 && !a.is_ghost(55) && a.is_ghost(50), "the ghost over the cut leaves with its number");
    Alloc r(52);
    std::vector<uint32_t> w;
    r.take(51, &w);
    r.give_ghost(40, 1), r.give_ghost(50, 2);
    check(a.free() == 2 && a.split_digest() == r.split_digest() && a.digest() == r.digest(),
          "the ghost digest and the taken digest are as if the cut ghost had never been");
    check(a.take() == 40 && a.take() == 50 && a.free() == 0, "what is left of the free set is the two ghosts, oldest first");
  }
  {  // the model-based check with ghosts and a zone: a plain-set / ghost-map reference with the four steps spelled out
    const uint32_t n = 400, zone = 300;
    Alloc a(n);
    a.set_zone(zone);
    std::set<uint32_t> plain, held;
    std::map<uint32_t, uint64_t> ghost;  // number -> stamp
    for (uint32_t c = 1; c < n; ++c) plain.insert(c);
    std::mt19937_64 rng(23);
    uint64_t seq = 0;
    int bad = 0;
    auto want = [&]() -> uint32_t {
      auto low = plain.begin();
      if (low != plain.end() && *low < zone) return *low;
      uint32_t best = 0;
      for (const auto& g : ghost)
        if (g.first < zone && (!best || g.second < ghost[best] || (g.second == ghost[best] && g.first < best))) best = g.first;
      if (best) return best;
      if (low != plain.end()) return *low;
      for (const auto& g : ghost)
        if (!best || g.second < ghost[best] || (g.second == ghost[best] && g.first < best)) best = g.first;
      return best;
    };
    for (int i = 0; i < 200000 && !bad; ++i) {
      const int op = (int)(rng() % 10);
      if (op < 4 && (!plain.empty() || !ghost.empty())) {
        const uint32_t w = want(), got = a.take();
        if (got != w) ++bad;
        plain.erase(got), ghost.erase(got), held.insert(got);
      } else if (op < 6 && !held.empty()) {
        auto it = held.begin();
        std::advance(it, rng() % held.size());
        const uint32_t c = *it;
        a.give_ghost(c, ++seq);
        held.erase(c), ghost[c] = seq;
      } else if (op < 8 && !held.empty()) {
        auto it = held.begin();
        std::advance(it, rng() % held.size());
        const uint32_t c = *it;
        a.give(c);
        held.erase(c), plain.insert(c);
      } else if (op < 9 && !ghost.empty()) {
        auto it = ghost.begin();
        std::advance(it, rng() % ghost.size());
        const uint32_t c = it->first;
        a.revive(c);
        ghost.erase(c), held.insert(c);
      }
      if (a.free() != plain.size() + ghost.size() || a.ghosts() != ghost.size() || a.free_plain() != plain.size()) ++bad;
      if (i % 500 == 0) {
        if (a.digest() != digest_of(held)) ++bad;
        uint32_t below = 0;
        for (const uint32_t c : plain) below += c < zone;
        for (const auto& g : ghost) below += g.first < zone;
        if (a.free_below_zone() != below) ++bad;
      }
    }
    check(!bad, "200,000 operations with ghosts and a zone equal the reference (four steps, counts, digest)");
  }
  {  // the stamp is a count: two cores given the same commands hold the same stamps and hand out the same numbers
    auto run = [](bool clock_stamps, uint64_t clock_skew) {
      Alloc a(41);
      std::vector<uint32_t> v, got;
      a.take(40, &v);  // every number is taken: only the ghosts are free
      uint64_t seq = 0, clock = clock_skew;
      const uint32_t order[] = {9, 31, 4, 17, 22};  // a split gives these back in this order
      for (const uint32_t c : order) {
        clock = clock * 7 % 11 + 1;  // a clock that moves with unrelated work, and differs between the cores
        a.give_ghost(c, clock_stamps ? clock : ++seq);
      }
      for (int i = 0; i < 5; ++i) got.push_back(a.take());
      return std::make_pair(got, a.split_digest());
    };
    const auto x = run(false, 0), y = run(false, 5);
    check(x.first == y.first && x.second == y.second, "two cores given the same commands hold the same stamps and hand out the same numbers");
    check(x.first == std::vector<uint32_t>({9, 31, 4, 17, 22}), "the stamps order the ghosts as the split gave them back");
    const auto cx = run(true, 0), cy = run(true, 5);
    check(cx.first != cy.first, "control: a stamp taken from a clock that differs between the cores makes the two cores differ");
  }
  {  // unghost: a ghost whose link is cut is a plain free number, handed out before a ghost that can still be revived
    Alloc a(20);
    std::vector<uint32_t> v;
    a.take(10, &v);
    a.give_ghost(4, 1), a.give_ghost(5, 2);
    a.unghost(5);
    check(a.ghosts() == 1 && !a.is_ghost(5) && a.is_ghost(4) && a.free() == 11, "unghost: one ghost left, the number still free");
    check(a.take() == 5 && a.take() == 11, "the plain numbers first, lowest first: 5, then 11 (not the ghost 4)");
    check(a.take() == 12, "a take of 12 follows");
    a.unghost(4);
    check(a.ghosts() == 0 && a.split_digest() == 0, "no ghost, the ghost digest is 0 again");
  }
  {  // the quarter slots
    SlotMap m;
    uint32_t freed = 9;
    check(!m.has_room() && m.digest() == 0, "an empty map");
    m.open(7), m.open(3);
    check(m.has_room() && m.open_numbers() == 2 && m.used() == 0, "two numbers open, no slot used");
    check(m.take_slot() == (3u << 2 | 0) && m.take_slot() == (3u << 2 | 1), "the lowest open number's lowest free quarter first");
    check(m.take_slot() == (3u << 2 | 2) && m.take_slot() == (3u << 2 | 3) && m.take_slot() == (7u << 2 | 0), "a full number is left for the next");
    check(m.used() == 5 && m.refs(3u << 2 | 1) == 1, "five slots used, one holder each");
    m.ref(3u << 2 | 1);
    check(m.refs(3u << 2 | 1) == 2 && m.unref(3u << 2 | 1, &freed) == 1 && freed == 0, "a second holder: one reference left, the slot stays");
    check(m.unref(3u << 2 | 1, &freed) == 0 && freed == 0 && m.used() == 4, "the last holder: the slot is free, its number stays (three used)");
    check(m.take_slot() == (3u << 2 | 1), "a freed slot is taken again before a higher number's");
    const uint64_t d = m.digest();
    for (int q : {0, 1, 2}) m.unref(3u << 2 | q, &freed), check(freed == 0, "a number with a used slot is not freed");
    check(m.unref(3u << 2 | 3, &freed) == 0 && freed == 3 && !m.is_open(3), "the last used slot of a number frees the number");
    check(m.digest() != d && m.open_numbers() == 1 && m.open_from(0) == std::vector<uint32_t>({7}) && m.open_from(8).empty(), "open_from, digest moves");
    m.unref(7u << 2 | 0, &freed);
    check(freed == 7 && m.digest() == 0 && m.used() == 0, "all gone: the digest is 0");
    // two maps given the same commands agree, one that skips a command does not
    auto run = [](int skip) {
      SlotMap x;
      uint32_t f;
      x.open(5), x.open(2);
      for (int i = 0; i < 6; ++i) {
        if (i == skip) continue;
        x.take_slot();
      }
      x.ref(2u << 2 | 1);
      x.unref(2u << 2 | 0, &f);
      return x.digest();
    };
    check(run(-1) == run(-1) && run(3) != run(-1), "slot digests: equal on equal commands, different when a command is skipped");
  }
  std::printf("chunks_test: %s\n", g_fail ? "FAILED" : "PASSED");
  return g_fail ? 1 : 0;
}
