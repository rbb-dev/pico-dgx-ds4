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
  std::printf("chunks_test: %s\n", g_fail ? "FAILED" : "PASSED");
  return g_fail ? 1 : 0;
}
