// split_pace.h against a table of cases (host only, no GPU). Built with -DPICO_OLD_RULES the engine that never asks: every row that
// needs a demotion, a drop or a level fails there.
#include <cmath>
#include <cstdio>

#include "../interleave.h"
#include "../split_pace.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}
static bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

int main() {
  using namespace splitpace;
  Marks m;
  m.high = 500, m.low = 250, m.after_s = 300, m.pass_ms = 50;
  const double day = 86400;

  // (a) tiered, spare at the high mark: level 0, nothing for a conversation idle a day
  {
    const Level l = level(kTiered, 500, m, true);
    check(l.level == 0, "tiered, spare at the high mark: level 0");
    check(next(l, false, day, false) == Act::kNothing && next(l, true, day, false) == Act::kNothing, "level 0: nothing, whatever the age and whatever L2 holds");
    check(level(kTiered, 8000, m, true).level == 0, "tiered, spare far above: level 0");
  }
  // (b) between the marks: p 0.5, age 150 s, pass 30 ms with requests in flight, 250 ms without
  {
    const Level l = level(kTiered, 375, m, true);
    check(l.level == 1 && near(l.pressure, 0.5) && near(l.age_s, 150) && near(l.pass_ms, 30), "high 500, low 250, spare 375: level 1, p 0.5, age 150 s, pass 30 ms");
    check(next(l, false, 149, false) == Act::kNothing, "a conversation idle 149 s is not demoted at age 150");
    check(next(l, false, 150, false) == Act::kDemote, "idle 150 s: demoted");
    check(near(level(kTiered, 375, m, false).pass_ms, 250), "no request in flight: the pass may run 250 ms");
  }
  // (c) the ends of the line
  {
    const Level a = level(kTiered, 499, m, true);
    check(a.level == 1 && near(a.age_s, 298.8), "spare 499: age asked 298.8 s");
    const Level b = level(kTiered, 250, m, true);
    check(b.level == 1 && near(b.age_s, 0), "spare 250: level 1, age 0");
    check(level(kTiered, 249, m, true).level == 2, "spare 249: level 2");
  }
  // (d) level 2: L2 first, then L1 at any age, then nothing
  {
    const Level l = level(kTiered, 100, m, true);
    check(l.level == 2 && near(l.pressure, 1) && near(l.pass_ms, 50), "under the low mark: level 2, full pressure, full pass");
    check(next(l, true, 0, false) == Act::kDropL2 && next(l, true, day, false) == Act::kDropL2 && next(l, true, -1, false) == Act::kDropL2,
          "level 2 with L2: drop, whatever the candidate");
    check(next(l, false, 0, false) == Act::kDemote, "level 2 without L2: a candidate idle 0 s is demoted");
    check(next(l, false, -1, false) == Act::kNothing, "level 2, no L2, no candidate: nothing");
  }
  // (e) max: level 1 at the full age and the full pass whatever the spare
  {
    const Level l = level(kMax, 8000, m, true);
    check(l.level == 1 && near(l.age_s, 300) && near(l.pass_ms, 50), "max, spare far above high: level 1, age 300 s, the full pass");
    check(next(l, false, 299, false) == Act::kNothing && next(l, false, 300, false) == Act::kDemote, "idle 299 s nothing, 300 s demote");
    const Level u = level(kMax, 100, m, true);
    check(u.level == 2 && next(u, true, 0, false) == Act::kDropL2 && next(u, false, 0, false) == Act::kDemote, "max under the low mark: as tiered");
  }
  // (f) from high down to low the age asked never rises and the pass never falls
  {
    bool mono = true;
    double age = 1e18, pass = -1;
    for (int k = 0; k <= 25; ++k) {
      const int64_t spare = 500 - (int64_t)k * 10;  // 500 down to 250
      const Level l = level(kTiered, spare, m, true);
      if (spare < 500 && (l.age_s > age + 1e-9 || l.pass_ms < pass - 1e-9)) mono = false;
      if (spare < 500) age = l.age_s, pass = l.pass_ms;
    }
    check(mono, "from high down to low in 25 steps the age asked never rises and the pass never falls");
  }
  // (g) a deferral holds a demotion, not a drop
  {
    check(held(true, false, 100, 104) && !held(true, true, 100, 104) && !held(true, false, 104, 104), "a deferral waits for a number to open, else its retry time");
    check(held(false, true, 100, 104) && !held(false, true, 104, 104), "no deferral: the retry time alone");
    const Level l = level(kTiered, 100, m, true);
    check(next(l, false, 10, true) == Act::kNothing, "held: no demotion");
    check(next(l, true, 10, true) == Act::kDropL2, "held: a drop is not held");
    check(next(level(kTiered, 375, m, true), false, 200, true) == Act::kNothing, "held between the marks: no demotion");
  }
  // (h) pick: the lowest `used` not in flight
  {
    std::vector<Cand> c = {{1, 30, false}, {2, 10, true}, {3, 20, false}};
    check(pick(c) == 3, "three candidates, the lowest used (2) in flight: the next lowest");
    c[0].in_flight = c[2].in_flight = true;
    check(pick(c) == -1, "all in flight: -1");
    check(pick({}) == -1, "none: -1");
    c[2].in_flight = false;
    check(pick(c) == 3, "one free: it");
  }
  // (i) off and the old rules ask for nothing at any input
  {
    bool none = true;
    for (int64_t spare : {-5, 0, 100, 249, 250, 375, 500, 8000}) {
      const Level l = level(kOff, spare, m, true);
      if (l.level != 0 || next(l, true, day, false) != Act::kNothing || next(l, false, day, false) != Act::kNothing) none = false;
    }
    check(none, "off: level 0 and nothing at any spare, any age, with or without L2");
  }
  // the pass's time
  check(go_on(49, 50) && !go_on(50, 50), "a pass goes on while its time is not used (49 ms of 50, not 50)");
  check(go_on(249, 250) && !go_on(250, 250), "no request in flight: 249 of 250 goes on, 250 does not");

  // what a command leaves on the list
  check(after(64, 64, 0, 100).done == false, "a command that used its whole budget: more to split, stays");
  check(after(10, 64, 0, 100).done == true, "a command that split fewer than its budget: all it could, leaves");
  check(after(0, 64, 0, 100).done == true, "nothing left to split: leaves");
  const Outcome d = after(5, 64, 3, 100);
  check(!d.done && d.retry_at == 105, "a piece found no quarter number: stays, retried in 5 s");
  // ---- the imager's pace
  // idle_ok: false for each of flights, the rotation's, a queued prompt, an admitted request, a waiting request, a submit waiting; true with none
  check(idle_ok(false, false, false, false, false, false), "red: nothing in flight, queued, waiting or arriving: an idle command may go (fin6 passes 50 ms with requests in flight)");
  check(!idle_ok(true, false, false, false, false, false) && !idle_ok(false, true, false, false, false, false) && !idle_ok(false, false, true, false, false, false),
        "red: requests decoding, the rotation or a prompt queued or being read: no command between a prompt's chunks");
  check(!idle_ok(false, false, false, true, false, false) && !idle_ok(false, false, false, false, true, false) && !idle_ok(false, false, false, false, false, true),
        "red: an admitted request, a waiting one or a submit arriving: nothing at an admission");
  // ride_due: tiered at or above the high mark never; under it every 250 ms; max whenever work is pending; a pause or a flip waiting: not due
  check(!ride_due(kTiered, 500, m, true, 10, 0, false, false), "red: tiered with the ready blocks at the high mark: no ride");
  check(!ride_due(kTiered, 800, m, true, 10, 0, false, false), "tiered above the high mark: no ride");
  check(ride_due(kTiered, 499, m, true, 10, 0, false, false), "red: tiered under the high mark, 10 s since the last: a ride");
  check(!ride_due(kTiered, 499, m, true, 0.2, 0, false, false) && ride_due(kTiered, 499, m, true, 0.25, 0, false, false), "tiered under the high mark: every 250 ms, not before");
  check(ride_due(kMax, 8000, m, true, 10, 0, false, false), "red: max rides whenever work is pending, whatever the ready blocks");
  check(!ride_due(kMax, 8000, m, false, 10, 0, false, false), "max with no work pending: no ride");
  check(!ride_due(kMax, 8000, m, true, 10, 0, true, false) && !ride_due(kMax, 8000, m, true, 10, 0, false, true), "a pause being asked, or a flip no step read yet: not due");
  check(ride_due(kOff, 100, m, true, 10, 0, false, false) && !ride_due(kOff, 800, m, true, 10, 0, false, false),
        "off images by tiered's gate (the same disk tier): under the high mark a ride, at it none");
  check(ride_pieces(300, m) == kRidePieces && ride_pieces(249, m) == kRideRaised, "four pieces a ride, 64 under the low mark");
  // the four ranks settle up to the smallest reading, and the budget is the smallest room
  {
    const uint64_t done[4] = {5, 3, 7, 4};
    check(ride_settled(done, 4) == 3, "red: ride_settled({5, 3, 7, 4}) = 3: a slow stream holds the settle back for all");
    const int room[4] = {8, 2, 8, 8};
    check(ride_budget(4, room, 4) == 2 && ride_budget(1, room, 4) == 1 && ride_budget(4, room, 0) == 4, "the budget is the smallest room, at most what rank 0 asked");
  }
  // The rule: a request waiting for unwritten quarters while a prompt is read alone waits for the reader's end; counted
  check(reader_wait(false, true, true, true), "red: no decoder, a prompt being read, a request waiting, images pending: the wait is counted, nothing rides on the chunks");
  check(!reader_wait(true, true, true, true) && !reader_wait(false, false, true, true) && !reader_wait(false, true, false, true) && !reader_wait(false, true, true, false),
        "not counted with a decoder running, no reader, nobody waiting or nothing to image");
  check(!idle_ok(false, false, true, false, true, false), "and while the reader reads no image command is sent (the wait is the reader's end)");
  // a request waiting for clean blocks is the imager's reason to run
  check(idle_ok(false, false, false, false, true, false, true), "red: a waiter alone (only images can free what it needs) is an idle engine for the imager");
  check(!idle_ok(false, false, false, false, true, false, false), "a waiter is busy when the caller does not say images alone can free it (as before)");
  check(!idle_ok(true, false, false, false, true, false, true) && !idle_ok(false, false, true, false, true, false, true) && !idle_ok(false, false, false, true, true, false, true) &&
            !idle_ok(false, false, false, false, true, true, true),
        "a waiter beside decoders, a reader, an admitted request or an arriving one is not idle");
  check(ride_due(kTiered, 800, m, true, 10, 0, false, false, kRideMs, true), "red: tiered with the ready blocks above the high mark and a request waiting: a ride");
  check(ride_due(kOff, 800, m, true, 10, 0, false, false, kRideMs, true) && ride_due(kMax, 800, m, true, 10, 0, false, false, kRideMs, true), "off and max likewise");
  check(!ride_due(kTiered, 800, m, false, 10, 0, false, false, kRideMs, true) && !ride_due(kTiered, 800, m, true, 0.1, 0, false, false, kRideMs, true) &&
            !ride_due(kTiered, 800, m, true, 10, 0, true, false, kRideMs, true),
        "a waiter with no work pending, inside the pace, or with a pause being asked: not due");
  check(ride_pieces(900, m, kRidePieces, true) == kRideRaised && ride_pieces(900, m, kRidePieces, false) == kRidePieces, "red: a waiter raises a ride to 64 pieces");
  // a rotation that cannot move is the imager's reason, as a waiting request is
  check(rot_stalled(0, 4, false), "red: no flight taken, four put back for the pool, no prompt being read: the rotation is stalled");
  check(!rot_stalled(1, 3, false) && !rot_stalled(0, 0, false) && !rot_stalled(0, 4, true), "a flight taken, none put back for the pool, or a prompt being read: not stalled");
  check(!idle_ok(true, true, false, false, true, false, true), "named control (row q's hang): paused flights and a rotation, a waiter, no `stalled`: not idle");
  check(idle_ok(true, true, false, false, true, false, true, true), "red: with `stalled` the paused flights and the rotation do not make the engine busy");
  check(!idle_ok(true, true, true, false, true, false, true, true) && !idle_ok(true, true, false, true, true, false, true, true) && !idle_ok(true, true, false, false, true, true, true, true),
        "with `stalled` a prompt being read, an admitted request and a submit arriving are each still busy");
  check(pool_waiter(false, 1) && !pool_waiter(false, 0) && pool_waiter(true, 0), "red: pool_waiter: a flight put back for the pool is a waiter, as a request in wait_ is");
  // a long prompt that waits at its start is the imager's reason, as a request in wait_ is
  check(pf_start_waits(true, true, true), "red: the front's begin answered `waits`: it waits for the pool (qi)");
  check(pf_start_waits(true, false, false), "red: no slot can be picked for the front (a resume in pieces): it waits for the pool");
  check(!pf_start_waits(true, true, false) && !pf_start_waits(false, false, true), "the front not asked yet (it begins this pass), or already begun: not waiting at its start");
  check(!pfq_busy(true, false, true), "red: the queue whose front only waits for the pool does not keep the idle imager out");
  check(pfq_busy(true, true, true) && pfq_busy(true, false, false) && !pfq_busy(false, false, false), "a prompt being read, or a front yet to be asked, keeps it out; an empty queue does not");
  check(all_wait(false, false, true, false, true) && idle_ok(true, false, false, false, false, false, true, true), "red: qi: no decoder, a prompt waiting at its start: all wait, and the engine is idle for the imager");
  check(!idle_ok(true, false, true, false, false, false, true, false), "named control (qi's hang as built): the waiting prompt in pfq_ and flying_, no `stalled`: not idle");
  check(all_wait(true, true, true, false, true), "a stalled rotation and a waiting prompt: all wait");
  check(!all_wait(true, false, true, false, true), "decoders step (the rotation is not stalled): not all wait: the ride's");
  check(!all_wait(false, false, true, true, true), "a prompt being read: not all wait");
  check(!all_wait(false, false, false, false, false), "nothing waits: not all wait");
  check(all_wait(true, true, false, false, false), "the stall (a stalled rotation, no prompt queued): kept");
  check(pool_waiter(false, 0, true) && !pool_waiter(false, 0, false), "red: pool_waiter: a prompt that waits at its start is a waiter");
  // the pause rule, one pure function
  {
    PauseIn o;
    o.batch = true, o.wait = true, o.admissible = true;
    check(!pause_for_others(o), "red (named control: exch as built, true under OLD=1): a request in wait_ that cannot begin does not end the run");
    o.can_begin = true;
    check(pause_for_others(o), "+ can_begin: the run is ended for it");
    PauseIn e;
    e.batch = true, e.wait = true, e.ends = true;
    check(pause_for_others(e), "a request of the run ended while something waits: ended, as today");
    PauseIn a;
    a.submit = true, a.admissible = true;
    check(pause_for_others(a), "an arrival nobody has looked at is looked at (admissible)");
    PauseIn b;
    b.submit = true;
    check(!pause_for_others(b), "an arrival with no slot to be had ends nothing");
    PauseIn p;
    p.pf = p.pf_busy = p.burst_over = true;
    check(pause_for_others(p), "a prompt being read: its window is over: ended");
    p.burst_over = false, p.wait = true, p.can_begin = true;
    check(!pause_for_others(p), "a prompt being read keeps its window whatever waits");
    PauseIn q;
    q.pf = true, q.burst_over = true;
    check(!pause_for_others(q), "red (control: qs as built, true): the front only waits for the pool: no window to be over");
    q.can_begin = true;
    check(pause_for_others(q), "the front waits and can begin: ended");
    PauseIn one;
    one.wait = true, one.can_begin = true;
    check(pause_for_others(one), "red (control: false as built): one decoder, a request that can begin: ended");
    check(run_window(true, false, 335, 1000) == 0 && run_window(true, true, 335, 1000) == 335 && run_window(false, false, 0, 1000) == 1000, "red: run_window: 0 for a front that only waits, else today's");
    check(ask_due(10.0, 9.9, 0) && !ask_due(10.0, 9.97, 0) && !ask_due(10.0, 9.0, 10.5) && ask_due(10.6, 9.0, 10.5), "the ask's pace: 50 ms apart, not within 1 s of a miss");
  }
  check(slot_free_for_arrival(false, false) && !slot_free_for_arrival(true, false), "a slot with no request in flight is free; one with a request in flight is not");
  check(slot_free_for_arrival(true, true), "red (named control: false under OLD=1): the slot of a request that ended in the run is free for an arrival's question");
  // the decoders' owed time is settled by every decode run
  {
    check(near(owed_left(152, 289, false), 0), "red (named control, exch as built: 152): a run with no window of 289 ms pays a debt of 152");
    check(near(owed_left(152, 100, false), 52), "red: a run with no window of 100 ms leaves 52");
    check(near(owed_left(300, 100, true), 200), "a run with a window pays as before (both rule sets)");
    check(near(owed_left(10, 500, true), 0) && near(owed_left(0, 50, false), 0), "never under 0");
    il::State s;
    s.decoders = true, s.reader = true, s.owed_ms = owed_left(152, 2833, false);
    check(il::next(s) == il::Next::Piece, "red (control: Decode): the debt settled by a long run, the reader reads the next piece");
  }
  std::printf("%d cases, %d failed\n", g_n, g_bad);
  return g_bad ? 1 : 0;
}
