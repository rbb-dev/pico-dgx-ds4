// When a decoding run must end so that a tenant's change can move (host only: tests/lend_pace_test.cpp and model/tests/tenant_test.cpp).
// A tenant's change (Model::tenant_lend) is a command: begin and each poll go to every rank between runs, and the front-end's CONFIG is
// read between runs too. A batch of long responses is one run for minutes, so the run itself asks to pause, after the step it is on, when
//   - a CONFIG frame is waiting (the setting would otherwise wait for the run's end),
//   - a change is wanted and the server would begin it now (the same test lend_tick uses: not backed off, not waiting out a busy answer),
//   - a change is in flight and its next poll is due,
//   - a settle of owed groups is due (otherwise owed groups wait for the run's end).
// The memory watch is not a reason: it rides the run (mem_watch.h), read by every rank between two steps.
// A pause is what a new request already causes: the flights go on at once, no decode step computes anything differently.
// STATS is published from the run loops at its 0.5 s limit, and at once when a change lands or fails: so a long response does not freeze it.
// A pause costs every decoder a whole step, so a run that ends on one of these names it in the log (pause_why). The server's step hook asks once a step,
// after the next step is queued (cmd.h OnStep), and publishes STATS there at its limit.
// PICO_OLD_RULES builds the run that pauses for the watch as well (and publishes only at the loop's top).
#pragma once

namespace lendpace {

struct State {
  bool config_waiting = false;
  bool change_active = false;   // a tenant's change is in flight
  bool begin_wanted = false;    // the server would begin a change now
  bool settle_due = false;      // groups are owed, the settle's interval has passed and a window command may run
  bool watch_due = false;       // the memory watch's period has passed (a reason only under PICO_OLD_RULES)
  double now = 0, poll_at = 0;  // the time, and when the next poll of a change in flight is due
};

// Which of the terms is true, the first in this order: a setting (the CONFIG waiting), a tenant's change in flight whose poll is due, a change the server
// would begin, a settle of owed groups. None when no term is: under PICO_OLD_RULES a pause for the watch alone has no reason of ours to name.
enum class Why { None, Setting, Change, Begin, Settle };

inline Why pause_why(const State& s) {
  if (s.config_waiting) return Why::Setting;
  if (s.change_active && s.now >= s.poll_at) return Why::Change;
  if (!s.change_active && s.begin_wanted) return Why::Begin;
  if (s.settle_due) return Why::Settle;
  return Why::None;
}

// The words of the line a run that ended on it prints (serve: the run was paused for ...): a begin and a poll are both a tenant's change.
inline const char* why_text(Why w) {
  switch (w) {
    case Why::Setting: return "a setting";
    case Why::Change:
    case Why::Begin: return "a tenant's change";
    case Why::Settle: return "groups owed";
    case Why::None: break;
  }
  return "";
}

inline bool pause_due(const State& s) {
#ifdef PICO_OLD_RULES
  if (s.watch_due) return true;
#endif
  return pause_why(s) != Why::None;
}

constexpr double kPublishEveryS = 0.5;

// Whether STATS is published now: from the top of the loop, from a decode run's steps (`in_run`), and at once (`forced`) when a tenant's
// change landed or failed. The old rules publish only at the loop's top, rate-limited.
inline bool publish_due(double now, double last, bool forced, bool in_run) {
#ifdef PICO_OLD_RULES
  (void)forced;
  return !in_run && now - last >= kPublishEveryS;
#else
  (void)in_run;
  return forced || now - last >= kPublishEveryS;
#endif
}

}  // namespace lendpace
