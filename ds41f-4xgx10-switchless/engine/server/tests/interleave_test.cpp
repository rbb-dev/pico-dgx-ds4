// interleave.h's decisions against a table of cases (host only, no GPU): piece sizes from the measured cost, decode
// windows from the share, whole or long prompts by rows to compute, the second reader, the CED tail and alignment rule,
// and the scheduler's next action. Exit 0 when every case passes.
#include <cmath>
#include <cstdio>

#include "../interleave.h"

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
  std::printf("interleave_test: %d of %d cases pass\n", g_n - g_bad, g_n);
  return g_bad ? 1 : 0;
}
