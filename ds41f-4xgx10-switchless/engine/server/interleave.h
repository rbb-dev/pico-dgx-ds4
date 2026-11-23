// Interleaved prefill: the scheduler's decisions as pure functions (no engine state; tested by
// tests/interleave_test.cpp). While other sessions decode, a long prompt is prefilled in pieces, each followed by a decode
// window; nobody waits longer than the gap setting between steps, and the decoders get the share setting of the time.
// Solo paths (one decoder alone, one prompt alone) never reach these functions' interleaving branches.
#pragma once

#include <algorithm>
#include <cstdint>

namespace il {

struct Settings {
  bool on = true;            // --interleave / --no-interleave
  int whole_rows = 32768;    // --interleave-whole-rows: a prompt with at most this many rows to compute is read whole
  int push_rows = 131072;    // --interleave-push-rows: a second reader may start when its rows to compute are at most this
  int gap_ms = 600;          // --interleave-gap-ms: the longest a decoding session waits between steps
  int share = 40;            // --interleave-share: % of the time decoders get while a prompt is prefilled (10-90)
  int burst_max_ms = 1000;   // --interleave-burst-max-ms: one decode window at most
  int rows = 0;              // --interleave-rows: a fixed piece size while others decode (0: from the gap)
  int prefill_rows = 8192;   // --prefill-rows: the buffers, and the piece size when nobody decodes
};

// A piece's cost on the device: k x (fixed + per_row x rows); k scales the fit to what the pieces take (EWMA)
struct Cost {
  double fixed_ms = 76.0;
  double per_row_ms = 0.092;
  double k = 1.0;
  double piece_ms(int rows) const { return k * (fixed_ms + per_row_ms * rows); }
};

// The edge of a decode run, measured: a run that is paused ends with a step whose tokens are not emitted, and the next run's first
// tokens come after a one-row step, so the decoders are silent for a unit's own time plus the edge (about 1.4 steps more than the one
// step "the unit plus one step fits the gap" counts). `ms` is the moving average of (the silence across a unit - the unit's time), like
// Cost::k; unset it is 2.5 steps, and it is never under one step. An encode unit's budget is the gap less value(). PICO_OLD_RULES
// counts one step.
struct Edge {
  double ms = 0;  // 0: no unit measured yet
  double value(double step_ms) const {
#ifdef PICO_OLD_RULES
    return step_ms;
#else
    return std::max(step_ms, ms > 0 ? ms : 2.5 * step_ms);
#endif
  }
  // the decoders' silence across a unit of unit_ms (their last token before it to their first after it)
  void observe(double silence_ms, double unit_ms) {
#ifndef PICO_OLD_RULES
    const double s = std::max(0.0, silence_ms - unit_ms);
    ms = ms > 0 ? 0.8 * ms + 0.2 * s : s;
#else
    (void)silence_ms, (void)unit_ms;
#endif
  }
};

constexpr int kAlign = 1024;   // pieces in multiples of this (window snapshots sit on its multiples)
constexpr int kReplay = 128;   // a CED prompt's last piece holds at least this many rows

// The piece size while sessions decode (step_ms: their step): the largest multiple of kAlign, at most prefill_rows,
// whose cost plus one step fits the gap; never below kAlign. With nobody decoding: prefill_rows.
inline int piece_rows(const Settings& st, const Cost& c, double step_ms, bool decoders) {
  if (!decoders) return st.prefill_rows;
  if (st.rows > 0) return std::max(kAlign, std::min(st.rows / kAlign * kAlign, st.prefill_rows));
  const double room = st.gap_ms - step_ms - c.k * c.fixed_ms;
  const int fit = room > 0 ? (int)(room / (c.k * c.per_row_ms)) : 0;
  return std::max(kAlign, std::min(fit / kAlign * kAlign, st.prefill_rows / kAlign * kAlign));
}

// The decoders' window after unit_ms of prompt work, so they get `share` % of the time; at most burst_max_ms
inline double window_ms(const Settings& st, double unit_ms) {
  const int sh = std::clamp(st.share, 1, 99);
  return std::min<double>(st.burst_max_ms, unit_ms * sh / (100.0 - sh));
}

// The device time of an image request's read while sessions decode: its rows as pieces cost (the fit's fixed part once)
// plus encode_ms, the time of the image work ahead of the rows (0 while the rows arrive encoded)
inline double img_read_ms(const Cost& c, int rows, double encode_ms = 0.0) { return c.piece_ms(rows) + encode_ms; }

// A prompt is prefilled whole when interleave is off or its rows to compute (n - q, after the restart point)
// are at most whole_rows; else it is a long prompt, read in pieces. An image request (img_ms > 0: the device time of its
// read, img_read_ms; 0 for any other request, and for an image request while nobody decodes) is read whole only when
// that time plus one decode step fits the gap, so the decoders beside it are never silent longer than the gap.
#ifdef PICO_OLD_RULES
inline bool read_whole(const Settings& st, int rows_to_compute, double = 0.0, double = 0.0) {
  return !st.on || rows_to_compute <= st.whole_rows;
}
#else
inline bool read_whole(const Settings& st, int rows_to_compute, double img_ms = 0.0, double step_ms = 0.0) {
  if (!st.on) return true;
  if (rows_to_compute > st.whole_rows) return false;
  return img_ms <= 0.0 || img_ms + step_ms <= st.gap_ms;
}
#endif

// What a conversation's return costs before its first row (a split conversation's join): per-piece times kept as moving averages,
// one for a piece taken back from its ghost (or shared), one for an exchanged piece, one for a piece read from the SSD; start values
// 0.06 / 0.2 / 0.9 ms (0.5 under PICO_OLD_RULES). The estimate is the time ahead of the rows, the argument read_whole takes for an
// image's encode: when it plus a decode step fits the gap the commands go out back to back, else the request is read as a long prompt
// whose first units are its join commands, each followed by the decoders' window.
struct JoinCost {
#ifdef PICO_OLD_RULES
  double back_ms = 0.06, exch_ms = 0.2, disk_ms = 0.5;
#else
  // the SSD's start value: ms a piece of a 1M-token conversation's quarter images takes to read back beside decoders
  double back_ms = 0.06, exch_ms = 0.2, disk_ms = 0.9;
#endif
  double est_ms(int back, int exch, int disk) const { return back * back_ms + exch * exch_ms + disk * disk_ms; }
  // after a command or a read-back: its time and what it did; the part of the time the taken-back pieces account for comes off first
  void learn_join(double ms, int back, int exch) {
    if (exch > 0) exch_ms = 0.7 * exch_ms + 0.3 * std::max(0.0, ms - back * back_ms) / exch;
    else if (back > 0) back_ms = 0.7 * back_ms + 0.3 * ms / back;
  }
  void learn_disk(double ms, int pieces) {
    if (pieces > 0) disk_ms = 0.7 * disk_ms + 0.3 * ms / pieces;
  }
};

// Whether a split conversation's return goes out as units of the request's read (join commands, each followed by the decoders' window)
// or back to back : by the per-command bound, not by the gap. `whole_ms` (--kv-join-whole-ms, 250): a join
// estimated at most that, or with nobody decoding, goes back to back; above it its lists are units. 0 makes every join beside
// decoders units. PICO_OLD_RULES goes back to back unless est + one step passes the gap.
inline bool join_in_units(double est_ms, int whole_ms, double step_ms, int gap_ms, bool decoders) {
#ifdef PICO_OLD_RULES
  (void)whole_ms;
  return decoders && est_ms + step_ms > gap_ms;
#else
  (void)step_ms, (void)gap_ms;
  return decoders && est_ms > whole_ms;
#endif
}

// What a return still has to do when pieces of the conversation are on the SSD as well as in slots: the next command and its size. No
// command handles more than `J` pieces whatever their source: the pieces on the SSD first (a read-back
// of lists, `read`), then the slots (a join, `join`). 0/0: the return is whole. PICO_OLD_RULES sends every piece of the SSD in one command.
struct ReturnUnit {
  int read = 0, join = 0;
};
inline ReturnUnit return_unit(int disk_left, int split_left, int J) {
  ReturnUnit u;
#ifdef PICO_OLD_RULES
  (void)J;
  if (disk_left > 0) u.read = disk_left;
  else u.join = split_left;
#else
  if (disk_left > 0) u.read = std::min(disk_left, std::max(1, J));
  else u.join = std::min(split_left, std::max(1, J));
#endif
  return u;
}
// Whether the units of a return's SSD pieces may run: the numbers the pool gives without a running request's line (free and unreserved
// numbers, the idle store, the lines no request is in flight on) cover every list still on the SSD below `upto`. What a unit reads back
// is idle store until the request begins; where the room is short each unit would take it from a decoder's line, and the next would take
// it back. PICO_OLD_RULES: always.
inline bool return_fits(int left, int64_t room) {
#ifdef PICO_OLD_RULES
  (void)left, (void)room;
  return true;
#else
  return room >= left;
#endif
}
// Whether a request's return has anything to do before its rows: the pieces to take back, to exchange, and (as units now) the ones on the SSD.
inline bool return_asks(int back, int exch, int disk) {
#ifdef PICO_OLD_RULES
  (void)disk;
  return back + exch > 0;
#else
  return back + exch + disk > 0;
#endif
}

// A join unit that found no number (or whose pieces `give_back` could not cover) waits: the next unit is tried after `retry_s`, and
// nothing else is held up meanwhile (arrivals are admitted, tenant changes and split commands run).
struct JoinWait {
  double retry_at = 0;
  bool live = false;  // the last unit made progress: a join is in flight (the store change in the tenants' one queue)
  void moved() { live = true, retry_at = 0; }
  void stuck(double now, double retry_s = 0.05) { live = false, retry_at = now + retry_s; }
  bool may_try(double now) const { return now >= retry_at; }
};

// Whether a queued prompt may be begun now: a slot and memory for it (`picked`), and its retry time reached when its start waited
// (`jw`, the same wait a join unit has). PICO_OLD_RULES: `picked` (a start that found no slot ended the engine).
inline bool begin_fits(bool picked, const JoinWait& jw, double now) {
#ifdef PICO_OLD_RULES
  (void)jw, (void)now;
  return picked;
#else
  return picked && jw.may_try(now);
#endif
}

// An image request is never started while another is between its pieces (two clips would take the decoders' windows
// twice over): may a queued prompt begin beside the one being read?
// A prompt whose return (the join of its split conversation) is not whole is never begun beside a reader: its own units come first.
inline bool may_begin_beside(bool reader_is_image, bool candidate_is_image, bool candidate_joining = false) {
#ifdef PICO_OLD_RULES
  candidate_joining = false;
#endif
  return !(reader_is_image && candidate_is_image) && !candidate_joining;
}

// A second long prompt may read beside the active one when its rows to compute are at most push_rows
inline bool second_reader(const Settings& st, int rows_to_compute) {
  return st.on && rows_to_compute > st.whole_rows && rows_to_compute <= st.push_rows;
}

// A CED prompt's piece of R rows at pos of n: shortened (or lengthened up to the buffers) so that the rows left after it
// are 0 or at least kReplay; the first piece ends on a kAlign multiple when it can
inline int adjust_piece(int R, int pos, int n, bool ced, int prefill_rows) {
  R = std::min(R, n - pos);
  if (pos % kAlign && R > kAlign - pos % kAlign) R = R - (pos + R) % kAlign;  // end on a multiple
  const int left = n - pos - R;
  if (ced && left > 0 && left < kReplay) {
    if (R + left <= prefill_rows) R += left;  // the tail joins this piece
    else R -= kReplay - left;                 // or this piece leaves kReplay rows
  }
  return R;
}

// What the scheduler does next (one decision a pass of its loop). With two readers, the second (at most push_rows rows to
// compute) reads first: it is the short one, so it does not wait behind the long one
enum class Next { Admit, BeginReader, Piece, Decode, Idle };
struct State {
  bool decoders = false;      // sessions are decoding (rot_ non-empty)
  double owed_ms = 0;         // decode time owed to them
  bool reader = false;        // a long prompt is being prefilled
  bool queued = false;        // long prompts wait (pfq_)
  bool reader_fits = false;   // the front of the queue can take a slot and its memory now
  bool arrivals = false;      // requests wait to be admitted (ready_, wait_ or a submit)
  bool admitted_last = false; // the last unit was an admission: while a prompt reads, admissions and pieces alternate
  bool push_fits = false;     // a queued prompt of at most push_rows rows, not yet begun, can take a slot now while
                              // one reader reads (the second reader)
};
inline Next next(const State& s) {
  const bool owed = s.decoders && s.owed_ms > 0;
  if (s.arrivals && !owed && !(s.reader && s.admitted_last)) return Next::Admit;  // one a pass, nothing owed
  if (!s.reader && s.queued && s.reader_fits) return Next::BeginReader;
  if (s.reader && s.push_fits) return Next::BeginReader;  // a short long prompt starts beside it (setup only)
  if (s.reader && !owed) return Next::Piece;
  if (s.decoders) return Next::Decode;
  return Next::Idle;
}

}  // namespace il
