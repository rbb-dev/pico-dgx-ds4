// interleave.h's decisions against a table of cases (host only, no GPU): piece sizes from the measured cost, decode
// windows from the share, whole or long prompts by rows to compute, the second reader, the CED tail and alignment rule,
// the scheduler's next action, and a timeline of decoders beside an image request (the longest silence). Exit 0 when
// every case passes.
#include <cmath>
#include <cstdio>
#include <vector>

#include "../interleave.h"
#include "../vit_sched.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}
static void eq(int got, int want, const char* what) {
  ++g_n;
  if (got != want) ++g_bad, std::printf("FAIL: %s: %d, want %d\n", what, got, want);
}
static void near(double got, double want, const char* what) {
  ++g_n;
  if (std::fabs(got - want) > 1e-6 * std::max(1.0, std::fabs(want))) ++g_bad, std::printf("FAIL: %s: %g, want %g\n", what, got, want);
}

// The timeline of one request read beside decoders stepping every step_ms: the scheduler's choice (whole or pieces, the piece
// sizes, a decode window after each piece) with every unit costing what the fit says. Returns the longest time between two
// decode steps from the request's arrival (the decoders stepped just before it) to the end of its read plus one step.
struct Timeline {
  double silence_ms = 0;
  int units = 0;  // runs of the request: 1 = read whole
  bool whole = false;
};
static Timeline timeline(const il::Settings& st, const il::Cost& c, double step_ms, int rows, bool image, bool decoders = true) {
  Timeline t;
  double now = 0, last_step = 0;
  auto steps_until = [&](double until) {  // decode steps that start before until: the first ends one step after the window opens
    for (double at = now + step_ms; at - step_ms < until - 1e-9; at += step_ms)
      t.silence_ms = std::max(t.silence_ms, at - last_step), last_step = at, now = at;
  };
  t.whole = il::read_whole(st, rows, image && decoders ? il::img_read_ms(c, rows) : 0.0, decoders ? step_ms : 0.0);
  if (t.whole) {
    now = c.piece_ms(rows), t.units = 1;
  } else {
    for (int pos = 0; pos < rows;) {
      const int R = il::adjust_piece(il::piece_rows(st, c, step_ms, decoders), pos, rows, false, st.prefill_rows);
      now += c.piece_ms(R), pos += R, ++t.units;
      if (decoders && pos < rows) {
        const double w = std::max(step_ms, il::window_ms(st, c.piece_ms(R)));
        steps_until(now + w);  // the window ends with the step that was running when it closed
      }
    }
  }
  if (decoders) steps_until(now + step_ms);  // the first step after the read
  return t;
}

// The same for a pixel request of n images (each `patches` patches, `rows` rows of span): the encode is units of its own (vit_sched.h)
// between the decoders' windows, then the rows in pieces. Whole: read_whole counted the encode in the read's time, so the request is one
// run, and the encode is not cut. silence_ms: the longest time between two decode steps; encode_ms: the longest single encode unit.
struct TimelineE {
  double silence_ms = 0, encode_ms = 0, encode_silence_ms = 0;  // encode_silence_ms: the decoders' silence across the longest encode unit
  int units = 0;
  bool whole = false;
};
// `edge_extra_ms`: the head's run edge beyond one step: a paused run's last step emits no token and the next run's first token comes one step
// after its start, so the decoders are silent for the unit's time, this extra and one step. The server's budget for a unit is the gap less
// il::Edge (measured from those silences); the old rules count one step.
static TimelineE timeline_enc(const il::Settings& st, const il::Cost& c, const vitsched::EncCost& ec, double step_ms, int n, int patches, int rows_each,
                              bool decoders = true, double edge_extra_ms = 0) {
  TimelineE t;
  std::vector<vitsched::Img> im;
  for (int i = 0; i < n; ++i) im.push_back({i * (rows_each + 2), rows_each, patches, false});
  const int rows = n * (rows_each + 2);
  double now = 0, last_step = 0;
  auto steps_until = [&](double until) {
    for (double at = now + step_ms; at - step_ms < until - 1e-9; at += step_ms)
      t.silence_ms = std::max(t.silence_ms, at - last_step), last_step = at, now = at;
  };
  const double enc_all = vitsched::encode_ms(im, 0, ec);
  il::Edge edge;
  t.whole = il::read_whole(st, rows, decoders ? il::img_read_ms(c, rows, enc_all) : 0.0, decoders ? edge.value(step_ms) : 0.0);
  if (t.whole) {
    now = c.piece_ms(rows) + enc_all, t.units = 1, t.encode_ms = enc_all;
  } else {
    vitsched::Mark mark;
    for (int pos = 0; pos < rows && t.units < 10000;) {
      const double budget = decoders ? st.gap_ms - edge.value(step_ms) : 1e18;
      const int R = il::adjust_piece(il::piece_rows(st, c, step_ms, decoders), pos, rows, false, st.prefill_rows);
      const vitsched::Unit u = vitsched::next_unit(im, pos, R, budget, ec, mark, 1);
      double ms = 0;
      if (u.kind == vitsched::Unit::Rows) {
        ms = c.piece_ms(R), pos += R;
      } else {
        for (int k = 0; k < u.count; ++k) {
          const int i = u.first + k, b0 = u.count == 1 ? u.b0 : 0, b1 = u.count == 1 ? u.b1 : vitsched::kBlocks;
          ms += ec.ms(patches, b1 - b0);
          mark.ran(1, i, b1);
          if (b1 == vitsched::kBlocks) im[(size_t)i].written = true;
        }
        t.encode_ms = std::max(t.encode_ms, ms);
      }
      now += ms, ++t.units;
      if (decoders && pos < rows) {
        steps_until(now + edge_extra_ms + std::max(step_ms, il::window_ms(st, ms)));
        if (u.kind != vitsched::Unit::Rows) {
          edge.observe(ms + edge_extra_ms + step_ms, ms);  // the silence the server reads across this unit
          t.encode_silence_ms = std::max(t.encode_silence_ms, ms + edge_extra_ms + step_ms);
        }
      }
    }
  }
  if (decoders) steps_until(now + step_ms);
  return t;
}

// Two pixel requests at a gap of `gap_ms` beside one decoder: A is the largest image (9,189 patches, 1,024 rows), cut by blocks when the room is
// smaller than its time; B, a single 768 x 768 image, arrives in A's first window. B may start only when no other request's image is between its
// blocks (vitsched::may_start); a B that starts anyway meets Hold, which stops the engine. b_start is when B's first unit runs, a_encoded when A's
// last block ended; silence_ms the longest time between two decode steps.
struct TwoE {
  bool stopped = false;
  double b_start = -1, a_encoded = -1, silence_ms = 0;
};
static TwoE timeline_two(const il::Settings& st, const il::Cost& c, const vitsched::EncCost& ec, double step_ms) {
  TwoE t;
  std::vector<vitsched::Img> A = {{0, 1024, 9189, false}}, B = {{0, 382, 3025, false}};
  vitsched::Mark mark;
  const double budget = st.gap_ms - step_ms;
  double now = 0, last_step = 0;
  auto steps_until = [&](double until) {
    for (double at = now + step_ms; at - step_ms < until - 1e-9; at += step_ms)
      t.silence_ms = std::max(t.silence_ms, at - last_step), last_step = at, now = at;
  };
  int pa = 0, pb = 0;
  bool arrived = false;
  auto unit = [&](std::vector<vitsched::Img>& im, int& pos, int rows, uint64_t req) {  // one unit of a request; false: Hold
    const int R = il::adjust_piece(il::piece_rows(st, c, step_ms, true), pos, rows, false, st.prefill_rows);
    const vitsched::Unit u = vitsched::next_unit(im, pos, R, budget, ec, mark, req);
    if (u.kind == vitsched::Unit::Hold) return false;
    double ms = 0;
    if (u.kind == vitsched::Unit::Rows) {
      ms = c.piece_ms(R), pos += R;
    } else {
      for (int k = 0; k < u.count; ++k) {
        const int i = u.first + k, b0 = u.count == 1 ? u.b0 : 0, b1 = u.count == 1 ? u.b1 : vitsched::kBlocks;
        ms += ec.ms(im[(size_t)i].patches, b1 - b0);
        mark.ran(req, i, b1);
        if (b1 == vitsched::kBlocks) im[(size_t)i].written = true;
      }
    }
    now += ms;
    steps_until(now + std::max(step_ms, il::window_ms(st, ms)));
    if (req == 1 && t.a_encoded < 0 && A[0].written) t.a_encoded = now;
    return true;
  };
  while ((pa < 1024 || pb < 382) && !t.stopped) {
    if (pa < 1024) unit(A, pa, 1024, 1), arrived = true;
    if (arrived && pb < 382) {
      if (!vitsched::may_start(mark, true, 2)) continue;  // waits
      if (t.b_start < 0) t.b_start = now;
      if (!unit(B, pb, 382, 2)) t.stopped = true;
    }
  }
  steps_until(now + step_ms);
  return t;
}

int main() {
  il::Settings st;
  il::Cost c;
  // piece sizes (the default cost fit, and the decoders' step for one and for three sessions)
  eq(il::piece_rows(st, c, 25.77, false), 8192, "nobody decodes: prefill_rows");
  eq(il::piece_rows(st, c, 25.77, true), 5120, "one decoder, gap 600: (600-25.77-76)/0.092 = 5415 -> 5120");
  eq(il::piece_rows(st, c, 57.6, true), 4096, "three decoders, gap 600: 5069 -> 4096");
  st.gap_ms = 300;
  eq(il::piece_rows(st, c, 25.77, true), 2048, "gap 300: 2154 -> 2048");
  st.gap_ms = 150;
  eq(il::piece_rows(st, c, 25.77, true), 1024, "gap 150: below the floor -> 1024");
  st.gap_ms = 5000;
  eq(il::piece_rows(st, c, 25.77, true), 8192, "a long gap: capped by prefill_rows");
  st.gap_ms = 600;
  c.k = 1.25;
  eq(il::piece_rows(st, c, 25.77, true), 4096, "pieces 25 % slower than the fit: (600-25.77-95)/0.115 = 4167 -> 4096");
  c.k = 1.0;
  st.rows = 3000;
  eq(il::piece_rows(st, c, 25.77, true), 2048, "a fixed size: to a multiple of 1024");
  st.rows = 100000;
  eq(il::piece_rows(st, c, 25.77, true), 8192, "a fixed size: at most prefill_rows");
  st.rows = 0;
  st.prefill_rows = 4096;
  eq(il::piece_rows(st, c, 25.77, true), 4096, "exact mode's 4096 buffers cap it");
  eq(il::piece_rows(st, c, 25.77, false), 4096, "and nobody decoding takes them whole");
  st.prefill_rows = 8192;
  // decode windows: share of the time (default 40: unit x 40/60), capped
  near(il::window_ms(st, 547.0), 547.0 * 40 / 60, "share 40, a 5120 piece (547 ms)");
  st.share = 10;
  near(il::window_ms(st, 547.0), 547.0 * 10 / 90, "share 10");
  st.share = 90;
  near(il::window_ms(st, 547.0), 1000.0, "share 90: 4923 ms capped at 1000");
  st.share = 50;
  near(il::window_ms(st, 453.0), 453.0, "share 50: as long as the piece");
  st.share = 0;
  near(il::window_ms(st, 100.0), 100.0 / 99, "share 0 clamps to 1 %");
  st.share = 40;
  // whole or long, by rows to compute (what is left after the restart point), not by the prompt's length
  check(il::read_whole(st, 10000), "650K prompt with 640K cached: 10K rows -> whole");
  check(il::read_whole(st, 32768), "32,768 rows -> whole");
  check(!il::read_whole(st, 32769), "32,769 rows -> long");
  st.on = false;
  check(il::read_whole(st, 1000000), "interleave off: everything whole");
  check(!il::second_reader(st, 50000), "interleave off: no second reader");
  st.on = true;
  check(il::second_reader(st, 50000), "50K rows: may read beside a long prompt");
  check(il::second_reader(st, 131072), "131,072 rows: at the push size");
  check(!il::second_reader(st, 131073), "131,073 rows: waits");
  check(!il::second_reader(st, 20000), "20K rows: read whole, not a reader");
  // pieces: alignment of a first piece from an unaligned restart, and the CED tail
  eq(il::adjust_piece(5120, 5000, 200000, true, 8192), 4216, "restart at 5000: the first piece ends at 9216");
  eq(il::adjust_piece(5120, 9216, 200000, true, 8192), 5120, "aligned: unchanged");
  eq(il::adjust_piece(4096, 0, 4150, true, 8192), 4150, "CED: 54 rows would be left: they join this piece");
  eq(il::adjust_piece(8192, 0, 8250, true, 8192), 8122, "CED: 58 left and the buffers full: this piece leaves 128");
  eq(il::adjust_piece(4096, 0, 4150, false, 8192), 4096, "not CED: no tail rule");
  eq(il::adjust_piece(8192, 196608, 200000, true, 8192), 3392, "the last piece: what is left");
  // the scheduler's next action
  il::State s;
  check(il::next(s) == il::Next::Idle, "nothing: idle");
  s.arrivals = true;
  check(il::next(s) == il::Next::Admit, "a request, nobody decoding: admit");
  s.decoders = true, s.owed_ms = 200;
  check(il::next(s) == il::Next::Decode, "a request while decoders are owed time: decode first");
  s.owed_ms = 0;
  check(il::next(s) == il::Next::Admit, "owed nothing: admit");
  s = {};
  s.queued = true, s.reader_fits = true, s.decoders = true, s.owed_ms = 300;
  check(il::next(s) == il::Next::BeginReader, "a queued long prompt starts even while others decode (setup only)");
  s.reader_fits = false;
  check(il::next(s) == il::Next::Decode, "it cannot fit yet: decode");
  s = {};
  s.reader = true;
  check(il::next(s) == il::Next::Piece, "a reader, nobody decoding: pieces back to back");
  s.decoders = true, s.owed_ms = 50;
  check(il::next(s) == il::Next::Decode, "a reader and decoders owed time: decode");
  s.owed_ms = 0;
  check(il::next(s) == il::Next::Piece, "owed nothing: the next piece");
  s = {};
  s.reader = true, s.arrivals = true;
  check(il::next(s) == il::Next::Admit, "a reader and a short request: admit it");
  s.admitted_last = true;
  check(il::next(s) == il::Next::Piece, "right after an admission: the reader's piece (no starving behind a stream)");
  s.reader = false, s.queued = true, s.reader_fits = false;
  check(il::next(s) == il::Next::Admit, "no reader yet: admissions do not alternate");
  s = {};
  s.reader = true, s.push_fits = true, s.decoders = true, s.owed_ms = 100;
  check(il::next(s) == il::Next::BeginReader, "a reader, a push-sized prompt queued: it begins beside (setup only)");
  s.push_fits = false;
  check(il::next(s) == il::Next::Decode, "no push-sized prompt: decode as owed");
  // an image request beside decoders: the longest silence (default settings, one decoder at 25.77 ms a step)
  {
    il::Settings k;
    il::Cost cost;
    const double step = 25.77;
    const int clip_rows = 32 * 382;  // 32 images of 768 x 768: 382 rows each
    const Timeline clip = timeline(k, cost, step, clip_rows, true);
    check(clip.silence_ms <= k.gap_ms, "a 32-image clip beside one decoder: no silence above the gap");
    check(!clip.whole && clip.units == 3, "the clip is read in pieces (5120 + 5120 + 1984 rows)");
    const Timeline one = timeline(k, cost, step, 382, true);
    check(one.whole && one.units == 1, "a single 768 x 768 image stays one run");
    check(one.silence_ms <= k.gap_ms, "and its silence is within the gap");
    const Timeline big = timeline(k, cost, step, 8000, true);
    check(!big.whole && big.units > 1 && big.silence_ms <= k.gap_ms, "an image larger than the room is cut, within the gap");
    const Timeline alone = timeline(k, cost, step, clip_rows, true, false);
    check(alone.whole, "nobody decoding: the clip is one run (nothing to keep waiting)");
    const Timeline text = timeline(k, cost, step, 12288, false);
    check(text.whole && text.silence_ms > k.gap_ms, "a text prompt of the same rows is still read whole (unchanged)");
    k.gap_ms = 300;
    const Timeline tight = timeline(k, cost, step, clip_rows, true);
    check(tight.silence_ms <= 300 && tight.units > 3, "a gap of 300: smaller pieces, still within it");
    k.gap_ms = 600;
    eq(il::read_whole(k, 382, il::img_read_ms(cost, 382), step) ? 1 : 0, 1, "image: 111 ms + a step fits the gap");
    check(!il::read_whole(k, 12224, il::img_read_ms(cost, 12224), step), "image: 1,201 ms does not");
    check(!il::read_whole(k, 12224, il::img_read_ms(cost, 12224, 0.0), 0.0), "decoders absent: the step is 0, the read alone still above the gap");
    check(!il::read_whole(k, 382, il::img_read_ms(cost, 382, 600.0), step), "the encode time joins the read's: 111 + 600 ms does not fit");
    check(il::read_whole(k, 10000), "no image time given: the rows rule as before");
    check(!il::read_whole(k, 40000, 1.0, 0.0), "an image request above whole_rows is a long prompt as before");
    k.on = false;
    check(il::read_whole(k, 12224, il::img_read_ms(cost, 12224), step), "interleave off: whole");
    check(il::may_begin_beside(false, true) && il::may_begin_beside(true, false) && il::may_begin_beside(false, false),
          "a prompt may begin beside a reader unless both are images");
    check(!il::may_begin_beside(true, true), "an image never begins while another image is between its pieces");
    check(!il::may_begin_beside(false, false, true) && il::may_begin_beside(false, false, false), "a prompt whose return is not whole is not begun beside a reader");
    // a queued prompt whose start waited is not begun again before its retry, and the decoders decode meanwhile
    il::JoinWait w;
    w.stuck(10.0);
    check(!il::begin_fits(true, w, 10.01), "red: a prompt whose start waited is not begun before its retry time");
    check(il::begin_fits(true, w, 10.06) && !il::begin_fits(false, w, 99.0), "its retry time reached it is begun when a slot is picked; never without one");
    il::State ws;
    ws.decoders = true, ws.queued = true;
    ws.reader_fits = il::begin_fits(true, w, 10.01);
    check(il::next(ws) == il::Next::Decode, "red: the start waited: the decoders decode meanwhile, the prompt is not begun again");
    ws.reader_fits = il::begin_fits(true, w, 10.06);
    check(il::next(ws) == il::Next::BeginReader, "after the retry time the prompt is begun");
    il::JoinWait fresh;
    check(il::begin_fits(true, fresh, 0.0), "a prompt that never waited is begun at once");
  }
  // a pixel request: the encode inside the scheduler's units (default settings, one decoder at 25.77 ms a step)
  {
    il::Settings k;
    il::Cost cost;
    const vitsched::EncCost ec;
    const double step = 25.77;
    const TimelineE clip = timeline_enc(k, cost, ec, step, 32, 3025, 382);
    check(!clip.whole && clip.silence_ms <= k.gap_ms, "encode: a 32-image clip beside one decoder: no silence above the gap (red on the old rules: one run of 2.5 s)");
    check(clip.encode_ms + step <= k.gap_ms, "encode: no encode piece above the gap less a step (red on the old rules)");
    const TimelineE one = timeline_enc(k, cost, ec, step, 1, 3025, 382);
    check(one.whole && one.units == 1 && one.silence_ms <= k.gap_ms, "encode: a single 768 x 768 image is one run, within the gap");
    const TimelineE alone = timeline_enc(k, cost, ec, step, 32, 3025, 382, false);
    check(alone.whole && alone.units == 1, "encode: nobody decoding: the clip is one run, the encode not cut");
    k.gap_ms = 200;
    const TimelineE big = timeline_enc(k, cost, ec, step, 1, 9189, 1024);
    check(!big.whole && big.units >= 3 && big.silence_ms <= k.gap_ms, "encode: an image larger than the room is cut by blocks, within the gap (red on the old rules: one 274 ms run)");
    k.gap_ms = 300;
    const TimelineE tight = timeline_enc(k, cost, ec, step, 32, 3025, 382);
    check(tight.silence_ms <= 300, "encode: a gap of 300: smaller encode pieces, still within it (red on the old rules)");
    // the head's run edge: 1.4 steps beyond the one the budget counts (WF3B's clip arm: units of 389 ms in a 462.9 ms budget, silences of 0.64 to 0.70 s)
    k.gap_ms = 600;
    {
      const double s130 = 130.0;
      const TimelineE edged = timeline_enc(k, cost, ec, s130, 32, 3025, 382, true, 1.4 * s130);
      check(edged.encode_silence_ms <= k.gap_ms,
            "encode: beside 8 decoders at the head's run edge (1.4 steps beyond one) the silence across an encode unit is at most the gap (red on the old rules: gap and a half step over)");
    }
    k.gap_ms = 200;
    const TwoE two = timeline_two(k, cost, ec, step);
    check(!two.stopped && two.b_start >= two.a_encoded && two.a_encoded > 0 && two.silence_ms <= k.gap_ms,
          "encode: a single 768 x 768 image arriving in the first window of a largest image cut by blocks starts after that image's last block, no silence above the gap (red on the old rules: Hold)");
    k.gap_ms = 600;
    check(!il::read_whole(k, 12224, il::img_read_ms(cost, 12224, vitsched::encode_ms([&] {
                                                                   std::vector<vitsched::Img> v;
                                                                   for (int i = 0; i < 32; ++i) v.push_back({i * 384, 382, 3025, false});
                                                                   return v;
                                                                 }(), 0, ec)), step),
          "encode: the clip's encode time keeps it from being one run beside decoders (red on the old rules)");
  }
  // ---- a split conversation's return: the estimate is the time ahead of the rows
  {
    il::Settings k;
    il::JoinCost jc;
    near(jc.est_ms(10, 0, 0), 0.6, "10 pieces taken back: 0.06 ms each");
    near(jc.est_ms(0, 10, 0), 2.0, "10 exchanged: 0.2 ms each");
    near(jc.est_ms(0, 0, 4), 3.6, "red: 4 from the SSD: 0.9 ms each (measured 298.1 ms for 346 pieces; the start value was 0.5)");
    near(jc.est_ms(100, 3000, 0), 606.0, "a 1M conversation (3,000 pieces exchanged) is about 0.6 s");
    il::JoinCost c2 = jc;
    c2.learn_join(40.0, 0, 100);  // 100 exchanged pieces took 40 ms: 0.4 ms each
    check(c2.exch_ms > jc.exch_ms && c2.exch_ms < 0.4 && c2.back_ms == jc.back_ms, "an exchange command moves the exchanged average, a step toward 0.4");
    c2.learn_join(3.0, 100, 0);
    check(c2.back_ms < 0.06 && c2.back_ms > 0.03, "a command of pieces taken back moves the taken-back average toward 0.03");
    c2.learn_join(10.0, 20, 10);  // the taken-back part (20 x 0.06) comes off the time before the exchanged ones are averaged
    check(c2.exch_ms > 0, "a mixed command: the exchanged average from what is left of its time");
    // the request is a long prompt when the join does not fit the gap, whatever its rows
    const double step = 100.0;
    check(il::read_whole(k, 5000, jc.est_ms(10, 0, 0), step), "a join of 0.6 ms beside a 100 ms step fits the 600 ms gap: read whole");
    check(!il::read_whole(k, 5000, jc.est_ms(0, 3000, 0), step), "red: a join of 600 ms and a step does not fit the gap: a long prompt, though its rows are few");
    check(il::read_whole(k, 5000, 0.0, step), "nobody's join: the rows rule as before");
  }
  // ---- units by the per-command bound, not by the gap; a unit that found no number waits
  {
    // 8 decoders: a step of 137 ms, the gap 600; a join estimated at 463 ms went out whole by the old rule, where the bound allows 250 ms over
    check(il::join_in_units(300.0, 250, 137.0, 600, true), "red: a join of 300 ms beside decoders is units with --kv-join-whole-ms 250 (the gap rule let it go whole)");
    check(!il::join_in_units(250.0, 250, 137.0, 600, true), "a join of exactly 250 ms goes back to back");
    check(!il::join_in_units(900.0, 250, 137.0, 600, false), "nobody decoding: back to back whatever it costs");
    check(il::join_in_units(0.5, 0, 137.0, 600, true), "red: --kv-join-whole-ms 0 makes every join beside decoders units (the row that proves the units path)");
    il::JoinWait jw;
    check(jw.may_try(10.0) && !jw.live, "a join not begun: a unit may go, nothing in flight");
    jw.moved();
    check(jw.live, "a unit that made progress: a join is in flight (the store change in the tenants' one queue)");
    jw.stuck(10.0);
    check(!jw.live && !jw.may_try(10.01) && jw.may_try(10.06), "red: a unit that found no number holds nothing up and tries again 50 ms later");
  }
  // ---- a return's pieces on the SSD are units of at most J like the rest
  {
    il::JoinCost jc;
    check(il::join_in_units(jc.est_ms(0, 0, 346), 250, 137.0, 600, true),
          "red: a 1M conversation (346 pieces on the SSD, none split) is estimated over 250 ms: units beside decoders (it was 172 ms: one command of 298)");
    check(il::return_asks(0, 0, 346), "red: a return of pieces only on the SSD has work for units (it had none: one read-back outside the plan)");
    check(!il::return_asks(0, 0, 0), "a conversation whole in the pool has no return");
    // the plan of a return: 346 lists on the SSD then 40 slots, J = 64: every command at most 64, every piece in exactly one
    int disk = 346, split = 40, cmds = 0, biggest = 0, total = 0, order_ok = 1, last_read = -1;
    while (disk + split > 0 && cmds < 100) {
      const il::ReturnUnit u = il::return_unit(disk, split, 64);
      const int n = u.read + u.join;
      if (n == 0) break;
      biggest = std::max(biggest, n), total += n, ++cmds;
      if (u.read > 0 && u.join > 0) order_ok = 0;  // a command is one kind
      if (u.read > 0) last_read = cmds;
      if (u.join > 0 && disk > 0) order_ok = 0;      // the SSD's pieces come before the slots
      disk -= u.read, split -= u.join;
    }
    check(biggest <= 64, "red: no command of a return handles more than J = 64 pieces whatever their source (the old plan: one command of 346)");
    eq(total, 386, "every piece of the return is in exactly one command");
    eq(cmds, 6 + 1, "346 on the SSD are 6 commands (5 of 64, one of 26) and the 40 slots are one more");
    check(order_ok == 1 && last_read == 6, "the SSD's pieces first, then the slots");
    const il::ReturnUnit one = il::return_unit(3, 0, 64);
    check(one.read == 3 && one.join == 0, "a short return is one command of its own size");
    const il::ReturnUnit j1 = il::return_unit(0, 100, 64);
    check(j1.read == 0 && j1.join == 64, "slots only: a join of J");
    // a return's SSD pieces are units only while the pool gives them without a running request's line
    check(!il::return_fits(310, 64), "red: 310 lists on the SSD and room for 64: the units would take decoders' lines, they are not units");
    check(il::return_fits(310, 310), "room for exactly the lists: units");
    check(il::return_fits(0, 0), "nothing on the SSD: fits");
    check(!il::return_fits(1, 0) && il::return_fits(5, 428), "no room for one list: not units; room for five among 428: units");
  }
  std::printf("interleave_test: %d of %d cases pass\n", g_n - g_bad, g_n);
  return g_bad ? 1 : 0;
}
