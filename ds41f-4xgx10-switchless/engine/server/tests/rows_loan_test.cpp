// server/rows_loan.h: when the lent prefill row set is asked for, given back, or left alone. A table of cases; the policy under test
// is checked against the table, and `--naive` runs the same table against a policy that asks whenever numbers are free (the RED run:
// it must fail the rows where the pool's top groups are in use and where the free numbers are under the high mark). Host only.
#include <cstdio>
#include <cstring>

#include "../rows_loan.h"

using rowsloan::Act;
using rowsloan::State;

namespace {
int fails = 0;
bool naive = false, noreason = false, old = false;

// What a careless policy would do: ask for the set whenever the pool has the groups free at all, give back only when it is empty.
Act decide_naive(const State& s) {
  if (!s.enabled || s.change_active) return Act::Keep;
  if (s.held > 0) return s.free_groups < 1 && !s.in_use ? Act::GiveBack : Act::Keep;
  return s.free_groups >= s.loan ? Act::Ask : Act::Keep;
}

// The policy before the going-back state was sticky: held and short and nothing reading: go back; otherwise keep.
Act decide_old(const State& s) {
  if (!s.enabled || s.change_active) return Act::Keep;
  if (s.held > 0) return !s.in_use && s.free_groups < s.marks.low_free ? Act::GiveBack : Act::Keep;
  if (s.backoff || !s.top_free) return Act::Keep;
  return s.free_groups >= s.loan + s.marks.high_extra ? Act::Ask : Act::Keep;
}

const char* name(Act a) { return a == Act::Ask ? "ask" : a == Act::GiveBack ? "give back" : a == Act::Withdraw ? "withdraw" : "keep"; }

void row(const char* what, const State& s, Act want) {
  const Act got = naive ? decide_naive(s) : old ? decide_old(s) : rowsloan::decide(s);
  const bool ok = got == want;
  std::printf("%s %s: wanted %s, got %s\n", ok ? "ok  " : "FAIL", what, name(want), name(got));
  if (!ok) ++fails;
}

State base() {
  State s;
  s.enabled = true, s.loan = 24, s.free_groups = 200, s.top_free = true;
  return s;
}
}  // namespace

int main(int argc, char** argv) {
  naive = argc > 1 && !std::strcmp(argv[1], "--naive");
  old = argc > 1 && !std::strcmp(argv[1], "--old");
  noreason = argc > 1 && !std::strcmp(argv[1], "--noreason");
  State s;
  // not held
  s = base();
  row("room, top groups free: ask", s, Act::Ask);
  s = base(), s.enabled = false;
  row("no second set configured: keep", s, Act::Keep);
  s = base(), s.top_free = false;
  row("top groups hold a number in use: do not ask (the take would evict)", s, Act::Keep);
  s = base(), s.free_groups = 24 + 7;
  row("under the high mark (the set plus 8 free): do not ask", s, Act::Keep);
  s = base(), s.free_groups = 24 + 8;
  row("exactly the high mark: ask", s, Act::Ask);
  s = base(), s.free_groups = 24;
  row("the groups free but no reserve after the take: do not ask", s, Act::Keep);
  s = base(), s.backoff = true;
  row("the last ask failed, not repeated yet: keep", s, Act::Keep);
  s = base(), s.change_active = true;
  row("a change of a tenant in flight: keep", s, Act::Keep);
  // held
  s = base(), s.held = 24, s.free_groups = 40;
  row("held, plenty free: keep", s, Act::Keep);
  s = base(), s.held = 24, s.free_groups = 1;
  row("held, free under the low mark, nothing reads on it: give back", s, Act::GiveBack);
  s = base(), s.held = 24, s.free_groups = 2;
  row("held, free exactly at the low mark: keep", s, Act::Keep);
  s = base(), s.held = 24, s.free_groups = 0, s.in_use = true;
  row("held, free under the low mark, a prompt reads on it: it starts going back (the command waits for the prompt)", s, Act::GiveBack);
  s = base(), s.held = 24, s.free_groups = 0, s.change_active = true;
  row("held, short, a change in flight: keep", s, Act::Keep);
  s = base(), s.held = 24, s.free_groups = 1, s.top_free = false, s.backoff = true;
  row("held, short: the ask's conditions do not matter, it goes back", s, Act::GiveBack);
  // why(): the reason that is logged, one row per reason; --noreason runs the table against a stub that always says the same (RED)
  auto w = [&](const char* what, const State& st, rowsloan::Why want) {
    const rowsloan::Why got = noreason ? rowsloan::Why::Off : rowsloan::why(st);
    const bool ok = got == want;
    std::printf("%s why: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
  };
  using rowsloan::Why;
  s = base(), s.free_groups = 29;
  w("29 free of 24 + 8 needed: waits for free groups (the count a log line states)", s, Why::WaitFewFree);
  s = base(), s.top_free = false;
  w("top groups busy", s, Why::WaitTopBusy);
  s = base(), s.backoff = true;
  w("backoff", s, Why::WaitBackoff);
  s = base();
  w("all conditions hold", s, Why::CanAsk);
  s = base(), s.change_active = true;
  w("a change in flight", s, Why::Changing);
  s = base(), s.enabled = false;
  w("off", s, Why::Off);
  s = base(), s.held = 24;
  w("held, plenty", s, Why::HeldPlenty);
  s = base(), s.held = 24, s.free_groups = 0, s.in_use = true;
  w("held, short, a prompt reads on it: waiting for the reader", s, Why::HeldReading);
  s = base(), s.held = 24, s.free_groups = 0;
  w("held, short, nothing reads on it", s, Why::HeldShort);
  {
    const bool got = old ? false : rowsloan::start_backoff(true, 0);  // --old: the policy that left a failed start's ask free to repeat at once
    const bool ok = got && !rowsloan::start_backoff(true, 23) && !rowsloan::start_backoff(false, 0);
    std::printf("%s a start that made no set sets the backoff (a held or a disabled loan does not)\n", ok ? "ok  " : "FAIL");
    if (!ok) ++fails;
  }
  // the sticky going-back state
  s = base(), s.held = 23, s.loan = 23, s.going_back = true, s.free_groups = 2, s.in_use = false;
  row("going back, free back at exactly the low mark, readers just ended: it goes back", s, Act::GiveBack);
  s = base(), s.held = 23, s.loan = 23, s.going_back = true, s.free_groups = 2, s.in_use = true;
  row("going back, free at the low mark, a prompt still reads: stays going back", s, Act::GiveBack);
  s = base(), s.held = 23, s.loan = 23, s.going_back = true, s.free_groups = 23 + 8 - 1;
  row("going back, free one under the ask condition: still going back", s, Act::GiveBack);
  s = base(), s.held = 23, s.loan = 23, s.going_back = true, s.free_groups = 23 + 8;
  row("going back, free at the ask condition (set + high mark): withdrawn", s, Act::Withdraw);
  s = base(), s.held = 23, s.loan = 23, s.going_back = true, s.free_groups = 5, s.in_use = true;
  w("going back, readers: waits for them", s, Why::HeldReading);
  s = base(), s.held = 23, s.loan = 23, s.going_back = true, s.free_groups = 5;
  w("going back, no readers", s, Why::HeldShort);
  std::printf("%s rows_loan_test (%s): %d failed\n", fails ? "FAIL" : "PASS", naive ? "naive policy" : "the policy", fails);
  return fails ? 1 : 0;
}
