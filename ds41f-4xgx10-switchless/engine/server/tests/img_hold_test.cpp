// What the server does with a flight's image rows (server/img_hold.h), run over the real ledger and allocator (host only, no
// GPU). A small stand-in for the ranks applies what serve.cpp's images(), finish() and idle_check() send them, with the
// same decision calls: a first go takes the numbers and sends the rows once; a resume sends no row and reads the rows
// the ranks hold; every way a request ends gives its numbers back; two image requests hold apart; a path that forgot to
// end a flight is found when nothing is in flight; admission counts the rows until they are taken.
// Built with -DPICO_OLD_RULES (the rows sent with every go, nothing freed at the end, no tripwire, no count in admission)
// the cases fail (exit 1). Exit 0 when every case passes.
#include <cstdio>
#include <vector>

#include "model/chunks.h"
#include "model/img_rows.h"
#include "server/img_hold.h"

namespace ir = model::imgrows;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s\n", what);
  }
}

// every rank's side: the allocator and the ledger; counts what crossed the ring
struct Ranks {
  model::chunks::Alloc a{4000};
  ir::Ledger L;
  size_t rows_sent = 0, bugs = 0, reads_missing = 0;
  bool take(int key, size_t rows) {
    return L.take(key, rows, [&](size_t n, ir::Nums* out) { return a.take((uint32_t)n, out); }) == ir::Take::kTaken;
  }
  void free_key(int key) {
    L.release(key, [&](ir::Nums& v) {
      for (const uint32_t c : v) a.give(c);
      v.clear();
    });
  }
  void free_all() {
    L.release_all([&](ir::Nums& v) {
      for (const uint32_t c : v) a.give(c);
      v.clear();
    });
  }
};

// the server's calls
struct Server {
  Ranks r;
  int next_key = 1;
  void go(imghold::Hold& h) {  // images(): kStImgTake and the rows on a first go, the spans only after
    const imghold::Go g = imghold::on_go(h);
    if (g.take) {
      h.key = next_key++;
      r.take(h.key, h.rows);
    }
    if (g.fill) r.rows_sent += h.rows;
    if (!r.L.find(h.key)) ++r.reads_missing;  // set_images would die: the ranks hold no rows for the key
  }
  void end(imghold::Hold& h) {  // finish()
    if (imghold::free_at_end(h)) r.free_key(h.key), h.key = 0;
  }
  void idle(size_t in_flight) {  // idle_check()
    if (imghold::tripwire(in_flight, r.L.numbers())) ++r.bugs, r.free_all();
  }
};
}  // namespace

int main() {
  const size_t kRows = 1000, kNums = ir::numbers_for(kRows);
  {  // taken at the first go, the rows sent once
    Server s;
    imghold::Hold h;
    h.rows = kRows;
    s.go(h);
    check(h.key != 0 && s.r.L.numbers() == kNums, "the first go takes the numbers its rows need");
    check(s.r.rows_sent == kRows, "and sends its rows once");
  }
  {  // a resume: no row sent, the same numbers, the rows still there
    Server s;
    imghold::Hold h;
    h.rows = kRows;
    s.go(h);
    const int key = h.key;
    const ir::Nums nums = s.r.L.find(key) ? s.r.L.find(key)->nums : ir::Nums{};
    s.go(h);  // the line was detached; the request resumes
    s.go(h);  // and again
    check(s.r.rows_sent == kRows, "a resume sends no row");
    check(h.key == key && s.r.L.numbers() == kNums && s.r.L.find(key) && s.r.L.find(key)->nums == nums, "the same numbers are held across resumes");
    check(s.r.reads_missing == 0, "every go finds its rows on the ranks");
    check(!imghold::on_go(h).take && !imghold::on_go(h).fill, "the decision for a resume is: neither take nor send");
  }
  {  // every way a request ends gives its numbers back
    const char* names[] = {"a request that stops", "a request that reaches its length", "a cancel mid-prefill (after the first go)",
                           "a client that left (after a resume)", "an error after the first go"};
    const int goes[] = {1, 3, 1, 2, 1};
    for (int i = 0; i < 5; ++i) {
      Server s;
      imghold::Hold h;
      h.rows = kRows;
      for (int g = 0; g < goes[i]; ++g) s.go(h);
      s.end(h);
      char what[160];
      std::snprintf(what, sizeof what, "%s: its numbers are back and the allocator is whole", names[i]);
      check(s.r.L.numbers() == 0 && s.r.a.held() == 0 && h.key == 0, what);
    }
    Server s;  // ended before any go: nothing to free
    imghold::Hold h;
    h.rows = kRows;
    s.end(h);
    check(s.r.a.held() == 0, "a request that ends before its first go frees nothing and holds nothing");
  }
  {  // two image requests at once
    Server s;
    imghold::Hold a, b;
    a.rows = 500, b.rows = 1500;
    s.go(a);
    s.go(b);
    check(a.key != b.key && s.r.L.numbers() == ir::numbers_for(500) + ir::numbers_for(1500), "two flights hold their own numbers");
    const ir::Nums nb = s.r.L.find(b.key) ? s.r.L.find(b.key)->nums : ir::Nums{};
    s.end(a);
    check(s.r.L.find(b.key) && s.r.L.find(b.key)->nums == nb && s.r.L.numbers() == ir::numbers_for(1500), "one ends: the other's rows are untouched");
    s.go(b);
    check(s.r.reads_missing == 0 && s.r.rows_sent == 500 + 1500, "and it resumes without a row sent");
    s.end(b);
    check(s.r.a.held() == 0, "both ended: nothing held");
  }
  {  // the tripwire
    Server s;
    imghold::Hold h;
    h.rows = kRows;
    s.go(h);  // a path that ends the flight without end(): the bug the tripwire is for
    s.idle(1);
    check(s.r.bugs == 0 && s.r.L.numbers() == kNums, "a flight in flight: the tripwire is quiet");
    s.idle(0);
    check(s.r.bugs == 1 && s.r.L.numbers() == 0 && s.r.a.held() == 0, "nothing in flight and numbers held: one report, the numbers freed");
    s.idle(0);
    check(s.r.bugs == 1, "and no second report");
  }
  {  // admission counts the rows until they are taken
    imghold::Hold h;
    h.rows = 40000;
    check(imghold::admission_numbers(h) == 197, "a request of 40,000 rows adds 197 numbers to its want");
    h.key = 3;
    check(imghold::admission_numbers(h) == 0, "a flight that holds its rows adds none");
    imghold::Hold t;
    check(imghold::admission_numbers(t) == 0, "a text request adds none");
    Server s;  // the reservation is spent by the take: nothing reserved is left held back
    imghold::Hold x;
    x.rows = 2000;
    const int64_t reserved = imghold::admission_numbers(x);
    s.go(x);
    check(reserved == (int64_t)ir::numbers_for(2000) && imghold::admission_numbers(x) == 0, "the reservation equals what the take takes, then it is gone");
  }
  {  // a held request of the same session that ends leaves the admitted request's ask alone
    imghold::Asks a;
    a.note(7, 1, 197);  // request 1 admitted, its rows not yet taken
    a.ended(7, 2);      // request 2 of the same session, held behind it, cancelled
    check(a.extra(7) == 197, "a cancelled held request of the same session leaves the admitted request's 197 numbers counted");
    a.ended(7, 1);
    check(a.extra(7) == 0, "the admitted request's own end clears its ask");
    a.note(8, 3, 5);
    a.taken(8);
    check(a.extra(8) == 0, "the take clears the ask");
  }
  std::printf(g_fail ? "img_hold_test: %d FAILED\n" : "img_hold_test: all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
