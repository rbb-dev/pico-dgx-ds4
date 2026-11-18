// A request's image rows as store numbers (model/img_rows.h, pool_rules.h; host only, no GPU): 204 whole rows to a number
// and no row across a number's edge; the runs a span is cut into; and the flights' rows against the real allocator:
// taken once per flight (a second take of the key takes nothing), held while other flights come and go, every number back
// when the flight is released or all are, and no room means nothing taken. The reservation for rows not yet taken counts with
// the reservations, so a bookmark or a restore cannot take the numbers first.
// Built with -DPICO_OLD_RULES (the rows in a pinned host buffer: no numbers, one run for any span, no reservation) the cases
// that need store numbers fail (exit 1). Exit 0 when every case passes.
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "model/chunks.h"
#include "model/img_rows.h"
#include "model/pool_rules.h"

using model::chunks::Alloc;
namespace ir = model::imgrows;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s\n", what);
  }
}
}  // namespace

int main() {
  {  // the geometry: a number holds 204 whole rows
    check(ir::kRowBytes == 10240 && ir::kPerNumber == 204 && 204 * ir::kRowBytes <= ir::kNumberBytes && 205 * ir::kRowBytes > ir::kNumberBytes,
          "204 rows fit a number, 205 do not");
    check(ir::numbers_for(1) == 1 && ir::numbers_for(204) == 1 && ir::numbers_for(205) == 2, "numbers_for at the edges");
    check(ir::numbers_for(40000) == 197 && ir::numbers_for(110000) == 540, "numbers for the 40,000-row cap and the protocol's 110,000 rows");
    check(ir::numbers_for(0) == 0, "no rows, no numbers");
    bool inside = true;
    for (size_t r = 0; r < 110000; ++r) {
      const ir::Where w = ir::where(r);
      inside = inside && w.number == r / 204 && w.offset + ir::kRowBytes <= ir::kNumberBytes && w.offset % 16 == 0;
    }
    check(inside, "every row of 110,000 lies inside one number, 16-byte aligned");
    check(ir::where(203).number == 0 && ir::where(204).number == 1 && ir::where(204).offset == 0, "row 204 starts the second number");
  }
  {  // runs: pieces inside one number, in order, covering the span exactly
    std::mt19937_64 g(7);
    bool ok = true;
    for (int i = 0; i < 20000; ++i) {
      const size_t r0 = g() % 5000, len = g() % 1500;
      size_t next = r0, pieces = 0;
      ir::runs(r0, r0 + len, [&](size_t first, size_t n) {
        ok = ok && first == next && n > 0 && first / 204 == (first + n - 1) / 204;
        next = first + n, ++pieces;
      });
      ok = ok && next == r0 + len;
    }
    check(ok, "runs cover [r0, r1) in order, each inside one number");
    size_t pieces = 0;
    ir::runs(100, 500, [&](size_t, size_t) { ++pieces; });
    check(pieces == 3, "rows 100..500 cut at 204 and 408: three pieces");
    pieces = 0;
    ir::runs(5, 5, [&](size_t, size_t) { ++pieces; });
    check(pieces == 0, "an empty span has no run");
  }
  {  // the flights' rows against the allocator
    Alloc a(1000);  // numbers 1..999
    ir::Ledger L;
    const auto take = [&](size_t n, ir::Nums* out) { return a.take((uint32_t)n, out); };
    const auto give = [&](ir::Nums& v) {
      for (const uint32_t c : v) a.give(c);
      v.clear();
    };
    check(L.take(0, 100, take) == ir::Take::kBad && L.take(3, 0, take) == ir::Take::kBad, "key 0 and no rows are refused");
    check(L.take(1, 1000, take) == ir::Take::kTaken, "flight 1 takes its rows");
    const size_t n1 = ir::numbers_for(1000);
    check(L.numbers() == n1 && a.held() == n1 && L.flights() == 1, "its numbers are held in the allocator and the ledger");
    const ir::Nums first = L.find(1) ? L.find(1)->nums : ir::Nums{};
    check(L.take(1, 1000, take) == ir::Take::kHeld && a.held() == n1, "a second take of the key (a resume) takes nothing");
    check(L.take(2, 408, take) == ir::Take::kTaken && L.flights() == 2, "flight 2 beside it");
    const ir::Ledger::Rows* r2 = L.find(2);
    bool apart = r2 != nullptr;
    for (const uint32_t c : first)
      for (const uint32_t d : r2 ? r2->nums : ir::Nums{}) apart = apart && c != d;
    check(apart, "two flights' numbers are different");
    check(L.release(2, give) && a.held() == n1 && L.find(1) && L.find(1)->nums == first, "flight 2 ends: its numbers back, flight 1's unchanged");
    check(!L.release(2, give), "a flight that holds nothing releases nothing");
    check(L.take(3, 204 * 2000, take) == ir::Take::kNoRoom && L.flights() == 1 && a.held() == n1, "no room: nothing taken, nothing recorded");
    check(L.take(4, 204, take) == ir::Take::kTaken, "flight 4");
    check(L.release_all(give) == n1 + 1 && L.numbers() == 0 && L.flights() == 0 && a.held() == 0 && a.free() == 999,
          "release_all gives every flight's numbers back");
  }
  {  // a random run of flights: numbers held = the live flights' sum, all back at the end
    std::mt19937_64 g(11);
    Alloc a(5000);
    ir::Ledger L;
    const auto take = [&](size_t n, ir::Nums* out) { return a.take((uint32_t)n, out); };
    const auto give = [&](ir::Nums& v) {
      for (const uint32_t c : v) a.give(c);
      v.clear();
    };
    std::set<int> live;
    bool ok = true;
    for (int i = 0; i < 5000; ++i) {
      const int key = 1 + (int)(g() % 40);
      if (live.count(key)) {
        ok = ok && L.release(key, give);
        live.erase(key);
      } else if (L.take(key, 1 + g() % 3000, take) == ir::Take::kTaken) {
        live.insert(key);
      }
      ok = ok && L.flights() == live.size() && a.held() == L.numbers();
    }
    for (const int key : live) ok = ok && L.release(key, give);
    check(ok && a.held() == 0, "random takes and releases: held = the live flights' numbers, nothing left");
  }
  {  // the reservation for rows not yet taken counts with the reservations
    const auto resv = [](int t) { return t == 0 ? 50 : 0; };
    const auto held = [](int t) { return t == 0 ? 20 : 0; };
    const auto img = [](int t) { return t == 1 ? 60 : 0; };
    const int64_t reserved = model::rules::unheld_sum(2, resv, held, img);
    check(reserved == 30 + 60, "the sum is the reservations' unheld part plus the rows' reservation");
    const auto none = []() { return 0; };
    check(!model::rules::covers(100, reserved, 20, none), "a bookmark may not take the numbers a request's rows are reserved");
    check(model::rules::covers(100, reserved, 10, none), "one that leaves them covered may");
  }
  std::printf(g_fail ? "img_rows_test: %d FAILED\n" : "img_rows_test: all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
