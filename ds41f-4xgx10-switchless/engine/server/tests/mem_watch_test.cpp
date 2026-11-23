// server/mem_watch.h: when the watch's readings are worth a line. A reading under keep - 64 MiB on any rank prints once; the same
// again does not; a lower one does; back at keep, the next dip prints again; MemAvailable is read from /proc/meminfo's text.
// The watch rides a decoding run as a toggle bit of rank 0's control word: the step that shows the other value than the one a loop saw last
// runs the reading, once; the bit ends no single session's reply; a flip no step read leaves the watch due.
// Built twice: with -DPICO_OLD_RULES (every reading under the threshold prints; the watch never rides a run) the cases marked "red" fail.
// Host only.
#include <cstdint>
#include <cstdio>
#include <string>

#include "../mem_watch.h"

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}
}  // namespace

int main() {
  const uint64_t G = 1ull << 30, M = 1ull << 20;
  memwatch::Watch w;
  w.keep = 10 * G;
  memwatch::Sample ok[4] = {{10 * G + 5 * M, 0}, {16 * G, 0}, {16 * G, 0}, {16 * G, 0}};
  check(!w.observe(ok) && w.min_bytes == 10 * G + 5 * M && w.min_rank == 0, "every rank above keep: no line; the smallest reading is kept for STATS");
  memwatch::Sample near[4] = {{10 * G - 50 * M, 0}, {16 * G, 0}, {16 * G, 0}, {16 * G, 0}};
  check(!w.observe(near) && w.low_count == 0, "a reading under keep by less than 64 MiB (the noise floor of a read) is not reported");
  memwatch::Sample low[4] = {{16 * G, 0}, {16 * G, 0}, {9 * G, 0}, {16 * G, 0}};
  check(w.observe(low) && w.min_rank == 2 && w.low_count == 1, "a reading under keep - 64 MiB on a worker is a new low: reported once, naming the rank");
  check(!w.observe(low) && w.low_count == 1, "red: the same reading again is not reported again");
  memwatch::Sample lower[4] = {{16 * G, 0}, {16 * G, 0}, {8 * G, 0}, {16 * G, 0}};
  check(w.observe(lower) && w.low_count == 2, "a lower reading is a new low");
  check(!w.observe(low) && w.low_count == 2, "red: a reading above the last low, still under keep, is not a new low");
  check(!w.observe(ok) && w.last_low == UINT64_MAX, "back above keep ends the episode");
  check(w.observe(low) && w.low_count == 3, "red: the next dip is reported again");
  const std::string l = memwatch::low_line(2, 9 * G, 10 * G, "rank 0 server token ids 1.20 GiB > 1.00 GiB");
  check(l == "memory: rank 2 has 9.00 GiB available, under the 10.00 GiB left to the host; lines over their cap: rank 0 server token ids 1.20 GiB > 1.00 GiB",
        "the line's text");
  check(memwatch::low_line(1, G, 10 * G, "").find("none known") != std::string::npos, "no line known over its cap is said so");
  check(memwatch::parse_mem_available("MemTotal: 1 kB\nMemFree: 2 kB\nMemAvailable:   30664032 kB\nBuffers: 3 kB\n") == 30664032ull * 1024,
        "MemAvailable is read from /proc/meminfo's text");
  check(memwatch::parse_mem_available("MemTotal: 1 kB\n") == 0, "a text without the line reads 0");
  // the watch stays out of the store's command path
  check(!memwatch::due(false, 1000.0, 0.0), "red: with the watch off (no reservation to hold: a --pool-tokens start) no watch command is due, however long it waits");
  check(memwatch::due(true, 1000.0, 0.0) && !memwatch::due(true, 5.0, 0.0) && memwatch::due(true, 10.0, 0.0), "with the watch on one is due every 10 s");
  const int kWatch = 99, kFree = 6;
  const int only[] = {kWatch, kWatch}, mixed[] = {kWatch, kFree}, free_only[] = {kFree};
  check(memwatch::watch_only(2, [&](int i) { return only[i]; }, kWatch), "red: a command whose operations are all the watch runs the watch only: no trim, no pin, no digest line");
  check(!memwatch::watch_only(2, [&](int i) { return mixed[i]; }, kWatch), "a command with another operation beside the watch takes the ordinary path");
  check(!memwatch::watch_only(1, [&](int i) { return free_only[i]; }, kWatch), "a command of another operation takes the ordinary path");
  check(!memwatch::watch_only(0, [&](int) { return kWatch; }, kWatch), "a command with no operation is not the watch");
  {
    // the toggle as a pure function of what each step's output shows
    const uint32_t K = memwatch::kCtlBit, P = 1u << 31;
    check(K == (1u << 30), "the watch's bit is bit 30: the entries' bits (0-15) and the pause (31) are free of it");
    uint32_t seen = 0;
    const uint32_t steps[] = {0, 0, K, K, K, 0, 0, K, K | P, K};
    std::string at;
    for (uint32_t ctl : steps) at += memwatch::flipped(seen, ctl) ? '1' : '0';
    check(at == "0010010100", "red: one watch a flip: steps that show the same value again do not run it (the value last seen)");
    seen = 0;
    check(!memwatch::flipped(seen, 0) && !memwatch::flipped(seen, 0), "a run's start sees 0: nothing runs until rank 0 flips");
    seen = 0;
    check(!memwatch::flipped(seen, P) && !memwatch::flipped(seen, 1u << 3), "the pause and an entry's bit are not the watch");
    seen = 0;
    check(memwatch::flipped(seen, K | (1u << 3)) && !memwatch::flipped(seen, K | P), "the bit beside others is still one flip");
    // rank 0's side: when the bit is flipped
    check(memwatch::ride_due(true, 11.0, 0.0, false, false), "red: the watch is due, nothing pauses, no flip waits: rank 0 flips");
    check(!memwatch::ride_due(true, 5.0, 0.0, false, false), "not due inside the period");
    check(!memwatch::ride_due(false, 100.0, 0.0, false, false), "red: with the watch off the bit is never set");
    check(!memwatch::ride_due(true, 11.0, 0.0, true, false), "a pause being asked: the run ends and the watch is a command between runs");
    check(!memwatch::ride_due(true, 11.0, 0.0, false, true), "a flip no step has read yet is not flipped back");
    // a flip nobody read: the run ended first; the next run starts with the bit clear and the watch is still due
    check(memwatch::ride_due(true, 11.0, 0.0, false, false), "red: a flip no step read leaves the watch due (its time is set only when a reading was made)");
    // a single session's reply: the bit ends nothing, an entry's bit still does
    check(memwatch::ends_reply(K) == 0, "red: the watch's bit alone ends no single session's reply");
    check(memwatch::ends_reply(1) != 0, "bit 0 still ends a single session's reply as a cancel");
    check(memwatch::ends_reply(K | 1) != 0, "the watch's bit beside bit 0 still ends it");
    check(memwatch::ends_reply(0) == 0, "no bit: the reply goes on");
  }
  {  // the split's ride: bit 29 flips once; ends_reply masks 29 and 30
    const uint32_t K = memwatch::kCtlBit, S = memwatch::kCtlSplitBit, P = 1u << 31;
    check(S == (1u << 29) && S != K && S != P, "red: the split's bit is bit 29, apart from the watch's (30) and the pause (31)");
    uint32_t seen = 0;
    std::string at;
    for (uint32_t ctl : {0u, 0u, S, S, S, 0u, 0u, S, 0u}) at += memwatch::split_flipped(seen, ctl) ? '1' : '0';
    check(at == "001001011", "red: one ride a flip: steps that show the same value again do not run it");
    seen = 0;
    check(!memwatch::split_flipped(seen, K) && !memwatch::split_flipped(seen, P) && !memwatch::split_flipped(seen, 1u << 3), "the watch's bit, the pause and an entry's bit are not the ride");
    seen = 0;
    check(memwatch::split_flipped(seen, S | K | (1u << 3)) && !memwatch::split_flipped(seen, S | K | P), "the bit beside others is still one flip");
    check(memwatch::ends_reply(S) == 0 && memwatch::ends_reply(S | K) == 0, "red: the ride's bit, alone or beside the watch's, ends no single session's reply");
    check(memwatch::ends_reply(S | 1) != 0 && memwatch::ends_reply(1) != 0, "bit 0 still ends it beside the ride's bit");
    // the two toggles are independent: a step that shows both flips both
    uint32_t ws = 0, ss = 0;
    check(memwatch::flipped(ws, K | S) && memwatch::split_flipped(ss, K | S) && !memwatch::flipped(ws, K | S) && !memwatch::split_flipped(ss, K | S),
          "the watch and the ride are each one flip on the same word");
  }
  {  // the watch rides a prompt's chunk: kPfWatch on the chunk's command, no command of its own between two chunks
    check(memwatch::chunk_ride_due(true, 10.0, 0.0), "red: a chunk carries the watch when the period has passed");
    check(!memwatch::chunk_ride_due(true, 9.9, 0.0), "not before the period");
    check(!memwatch::chunk_ride_due(false, 100.0, 0.0), "red: with the watch off no chunk carries it");
    // a 49 s prompt of 57 chunks, 0.86 s each, the watch last done at 0: the chunks carry it at 10, 20, 30 and 40 s and the loop issues none
    int rode = 0, issued = 0;
    double last = 0;
    for (int chunk = 1; chunk <= 57; ++chunk) {
      const double now = 0.86 * chunk;
      if (memwatch::chunk_ride_due(true, now, last)) last = now, ++rode;  // pf_piece's chunk
      else if (memwatch::due(true, now, last) && !memwatch::defer_to_chunk(true, now, last)) last = now, ++issued;  // watch_tick between two chunks
    }
    check(rode == 4 && issued == 0, "red: a 49 s prompt: the watch rides four chunks (every 10 s) and the loop issues no command between chunks");
    check(memwatch::defer_to_chunk(true, 15.0, 0.0) && !memwatch::defer_to_chunk(true, 20.0, 0.0), "red: the loop leaves a due watch to the chunks until it is two periods late");
    check(!memwatch::defer_to_chunk(false, 15.0, 0.0), "no prompt being read: nothing to defer to, the loop issues the watch");
  }
  std::printf("%s\n", fails ? "mem_watch_test FAILED" : "mem_watch_test passed");
  return fails ? 1 : 0;
}
