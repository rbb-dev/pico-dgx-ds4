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

// A prompt is prefilled whole when interleave is off or its rows to compute (n - q, after the restart point)
// are at most whole_rows; else it is a long prompt, read in pieces
inline bool read_whole(const Settings& st, int rows_to_compute) { return !st.on || rows_to_compute <= st.whole_rows; }

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
