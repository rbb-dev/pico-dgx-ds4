// turns.h against a model of serve.cpp's run(): the line of waiting requests, each run taking the first request's
// factor (the others set aside to the front, in their order), then turn-ended requests to the back and paused ones to
// the front. The ends follow what two factor-64 requests with drafts do: one ends its turn, the last one left pauses.
// Host only: no engine, no GPU.
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>

#include "../turns.h"

namespace {

struct R {
  int id, fac;
};

int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

// runs the line `runs` times; returns how many runs each request took part in
std::vector<int> simulate(std::deque<R> line, int runs, bool lead) {
  std::vector<int> ran(16, 0);
  int last_fac = 0;
  for (int r = 0; r < runs; ++r) {
    if (lead) turns::lead_other_factor(line, last_fac, [](const R& x) { return x.fac; });
    std::vector<R> fl, later;
    while (!line.empty()) {  // serve.cpp run(): a run takes the first one's factor; the others wait in line
      R f = line.front();
      line.pop_front();
      if (!fl.empty() && f.fac != fl[0].fac) {
        later.push_back(f);
        continue;
      }
      fl.push_back(f);
    }
    for (auto it = later.rbegin(); it != later.rend(); ++it) line.push_front(*it);
    last_fac = fl[0].fac;
    for (const R& f : fl) ++ran[f.id];
    // the ends: alone, a request ends its turn; two together, the first ends its turn and the last one left pauses
    std::vector<R> paused;
    for (size_t k = 0; k < fl.size(); ++k) {
      const bool pauses = fl.size() > 1 && k == 1;
      if (pauses) paused.push_back(fl[k]);
      else line.push_back(fl[k]);
    }
    for (auto it = paused.rbegin(); it != paused.rend(); ++it) line.push_front(*it);
  }
  return ran;
}

}  // namespace

int main() {
  // two 4M requests (factor 64) and one at factor 16; a paused factor-64 request goes back to the front of the line
  const std::deque<R> mixed = {{1, 64}, {2, 16}, {0, 64}};
  {
    const auto ran = simulate(mixed, 100, false);
    std::printf("     without the lead: factor 64 ran %d and %d runs, factor 16 ran %d of 100\n", ran[0], ran[1], ran[2]);
    check(ran[2] == 0, "without the lead the factor-16 request never runs (it starves)");
  }
  {
    const auto ran = simulate(mixed, 100, true);
    std::printf("     with the lead: factor 64 ran %d and %d runs, factor 16 ran %d of 100\n", ran[0], ran[1], ran[2]);
    check(ran[2] == 50, "with the lead the factor-16 request runs every second run");
    check(ran[0] == 50 && ran[1] == 50, "the two factor-64 requests still decode together in the other runs");
  }
  {
    // one factor only: nothing moves
    std::deque<R> line = {{0, 16}, {1, 16}, {2, 16}};
    turns::lead_other_factor(line, 16, [](const R& x) { return x.fac; });
    check(line[0].id == 0 && line[1].id == 1 && line[2].id == 2, "one factor waiting: the order is kept");
  }
  {
    // the first other-factor request leads; the rest keep their order
    std::deque<R> line = {{0, 64}, {1, 64}, {2, 16}, {3, 32}, {4, 16}};
    turns::lead_other_factor(line, 64, [](const R& x) { return x.fac; });
    check(line[0].id == 2 && line[1].id == 0 && line[2].id == 1 && line[3].id == 3 && line[4].id == 4,
          "the first request of another factor leads, the others keep their order");
  }
  {
    // the first run has no last factor; a line already led by another factor stays as it is
    std::deque<R> line = {{0, 64}, {1, 16}};
    turns::lead_other_factor(line, 0, [](const R& x) { return x.fac; });
    check(line[0].id == 0, "no run before: nothing moves");
    turns::lead_other_factor(line, 16, [](const R& x) { return x.fac; });
    check(line[0].id == 0, "already led by another factor: nothing moves");
  }
  {
    // three factors: each gets runs
    const auto ran = simulate({{0, 64}, {1, 64}, {2, 32}, {3, 16}}, 99, true);
    check(ran[2] > 0 && ran[3] > 0, "three factors: every factor gets runs");
  }
  std::printf("%s\n", fails ? "turns_test FAILED" : "turns_test passed");
  return fails ? 1 : 0;
}
