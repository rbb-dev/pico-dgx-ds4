// server/lend_pace.h: when a decoding run asks to pause. A waiting CONFIG, a change to begin or a poll of one in flight, and a due settle
// pause a run; the memory watch does not (it rides the run: mem_watch.h). Built twice: with -DPICO_OLD_RULES (the run pauses for
// the watch too) the cases marked "red" fail. Host only.
#include <cstdio>
#include <string>

#include "../lend_pace.h"

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}
}  // namespace

int main() {
  {
    lendpace::State s;
    s.watch_due = true, s.now = 11.0;
    check(!lendpace::pause_due(s), "red: the watch due alone pauses nothing");
  }
  {
    lendpace::State s;
    check(!lendpace::pause_due(s), "nothing wanted, nothing waiting: the run goes on");
  }
  {
    lendpace::State s;
    s.config_waiting = true, s.watch_due = true;
    check(lendpace::pause_due(s), "a CONFIG waiting pauses, the watch due or not");
  }
  {
    lendpace::State s;
    s.begin_wanted = true;
    check(lendpace::pause_due(s), "a change the server would begin now pauses");
  }
  {
    lendpace::State s;
    s.change_active = true, s.begin_wanted = true, s.now = 1.0, s.poll_at = 1.1;
    check(!lendpace::pause_due(s), "a change in flight whose poll is not due does not pause");
    s.now = 1.1;
    check(lendpace::pause_due(s), "a change in flight whose poll is due pauses");
  }
  {
    lendpace::State s;
    s.settle_due = true;
    check(lendpace::pause_due(s), "a due settle pauses");
  }
  {
    // a run of 30 s at 10 ms a step with the watch due every 10 s and nothing else: it never pauses
    int pauses = 0;
    double now = 0, watch_at = 0;
    for (int step = 1; step <= 3000; ++step) {
      now += 0.01;
      lendpace::State s;
      s.watch_due = now - watch_at >= 10, s.now = now;
      if (lendpace::pause_due(s)) watch_at = now, ++pauses;
    }
    check(pauses == 0, "red: a run of 30 s with the watch due three times is never ended for it");
  }
  {  // which term paused a run: the first true one in the order setting, change, begin, settle
    using lendpace::Why;
    lendpace::State s;
    check(lendpace::pause_why(s) == Why::None && !lendpace::pause_due(s), "nothing true: no reason and no pause");
    s = {};
    s.config_waiting = true, s.settle_due = true, s.begin_wanted = true;
    check(lendpace::pause_why(s) == Why::Setting, "a setting waiting is named before the others");
    s = {};
    s.change_active = true, s.now = 2.0, s.poll_at = 1.0, s.settle_due = true;
    check(lendpace::pause_why(s) == Why::Change, "a change whose poll is due is named before a settle");
    s = {};
    s.begin_wanted = true, s.settle_due = true;
    check(lendpace::pause_why(s) == Why::Begin, "a change the server would begin is named before a settle");
    s = {};
    s.change_active = true, s.now = 1.0, s.poll_at = 1.1, s.begin_wanted = true, s.settle_due = true;
    check(lendpace::pause_why(s) == Why::Settle, "a change in flight whose poll is not due: its begin is not a reason; the settle is");
    s = {};
    s.change_active = true, s.now = 1.0, s.poll_at = 1.1, s.begin_wanted = true;
    check(lendpace::pause_why(s) == Why::None && !lendpace::pause_due(s), "a change in flight not yet due and a begin wanted: no reason");
    s = {};
    s.watch_due = true, s.now = 11.0;
    check(lendpace::pause_why(s) == Why::None, "the watch due alone has no reason to name (under the old rules it pauses with none)");
    // pause_due is exactly "some reason" without the old rules' watch term
    int disagree = 0;
    for (int bits = 0; bits < 32; ++bits) {
      lendpace::State x;
      x.config_waiting = bits & 1, x.change_active = bits & 2, x.begin_wanted = bits & 4, x.settle_due = bits & 8;
      x.now = (bits & 16) ? 2.0 : 1.0, x.poll_at = 1.5;
      disagree += lendpace::pause_due(x) != (lendpace::pause_why(x) != Why::None);
    }
    check(disagree == 0, "pause_due is true exactly when a reason is named (32 combinations)");
    check(std::string(lendpace::why_text(Why::Setting)) == "a setting" && std::string(lendpace::why_text(Why::Change)) == "a tenant's change" &&
              std::string(lendpace::why_text(Why::Begin)) == "a tenant's change" && std::string(lendpace::why_text(Why::Settle)) == "groups owed",
          "the words of the log line");
  }
  std::printf("%s\n", fails ? "lend_pace_test FAILED" : "lend_pace_test passed");
  return fails ? 1 : 0;
}
