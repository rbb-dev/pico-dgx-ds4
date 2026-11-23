// The memory watch (host only: no engine state; tested by tests/mem_watch_test.cpp). Every 10 seconds between commands each
// rank reads MemAvailable and its heap, one host all-gather gives rank 0 the four, and rank 0 keeps them for STATS and prints a
// line when the smallest reading is a new low under `keep - 64 MiB`: once per new low, not once per reading; a reading back at or
// above `keep` ends the episode, so the next dip prints again.
//
// The watch rides a decoding run: rank 0 toggles kCtlBit of its control word, every rank's loop sees the other value in the same step's
// output and runs the reading there, between two steps, without ending the run (flipped, ride_due, ends_reply).
//
// PICO_OLD_RULES (never defined by the engine) builds the report without the rule: every reading under the threshold prints, and the
// watch never rides a run (a watch that is due ends the run, as a pause; the bit is never set).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace memwatch {

constexpr double kEveryS = 10;               // the watch's period
constexpr uint64_t kMargin = 64ull << 20;    // a reading is reported when it is under keep less this

// What one rank sends: MemAvailable and the glibc heap it holds, bytes.
struct Sample {
  uint64_t avail = 0, heap = 0;
};

// Whether a watch is due: only when the watch is on (a keep to hold: a --pool-free-gib start) and a period has passed. A start with the watch
// off issues nothing, so its command stream is the same as one without the watch.
inline bool due(bool on, double now, double last) {
#ifdef PICO_OLD_RULES
  (void)on;
  return now - last >= kEveryS;  // without the rule: a watch command every period, whatever the settings
#else
  return on && now - last >= kEveryS;
#endif
}

// Whether a command's state operations are the watch and nothing else: such a command runs the watch on every rank and does nothing the
// store would see (no history trim, no pin, no digest line), so the store's traces and digests do not depend on the clock.
template <class OpAt>
inline bool watch_only(int n_ops, OpAt op_at, int watch_op) {
#ifdef PICO_OLD_RULES
  (void)n_ops, (void)op_at, (void)watch_op;
  return false;  // without the rule: the watch goes through the store's ordinary command path
#else
  if (n_ops <= 0) return false;
  for (int i = 0; i < n_ops; ++i)
    if (op_at(i) != watch_op) return false;
  return true;
#endif
}

// The watch's bit in rank 0's control word (bits 0-15 are the entries', bit 31 is the pause).
constexpr uint32_t kCtlBit = 1u << 30;
// The split's ride bit: a toggle like the watch's, read in the same step's output on every rank; the ride's own work is one
// host all-gather and the store's bookkeeping, between two steps, the run going on.
constexpr uint32_t kCtlSplitBit = 1u << 29;

// Whether rank 0 flips the bit now: the watch is on and due, no flip is waiting for a step to read it, and no pause is being asked
// (a pause ends the run, and the watch is then a command between runs).
inline bool ride_due(bool on, double now, double last, bool pausing, bool flip_waiting) {
#ifdef PICO_OLD_RULES
  (void)on, (void)now, (void)last, (void)pausing, (void)flip_waiting;
  return false;
#else
  return on && !pausing && !flip_waiting && now - last >= kEveryS;
#endif
}

// Whether a prompt's chunk carries the watch: the watch is on and a period has passed. The chunk's command says so (kPfWatch) and every
// rank reads its memory and joins the host all-gather after queuing the chunk, while the device computes it: no command of its own between
// two chunks. With PICO_OLD_RULES a chunk never carries it.
inline bool chunk_ride_due(bool on, double now, double last) {
#ifdef PICO_OLD_RULES
  (void)on, (void)now, (void)last;
  return false;
#else
  return on && now - last >= kEveryS;
#endif
}

// Whether the server's loop leaves a due watch to the chunks of the prompt being prefilled: while one is, unless the watch is two periods late (a
// reader that waits for a line reads no chunk, so the watch is then a command between turns as before).
inline bool defer_to_chunk(bool reading, double now, double last) {
#ifdef PICO_OLD_RULES
  (void)reading, (void)now, (void)last;
  return false;
#else
  return reading && now - last < 2 * kEveryS;
#endif
}

// Whether a step whose output shows control word `ctl` runs the watch on this rank. `seen` is the bit's value this loop saw last (0 when
// a run starts: rank 0 starts every run with the bit clear); a step that shows the other value is the one, once.
inline bool flipped(uint32_t& seen, uint32_t ctl) {
#ifdef PICO_OLD_RULES
  (void)seen;
  return (ctl & kCtlBit) != 0;  // without the value last seen: every step after a flip runs the watch
#else
  const uint32_t now = ctl & kCtlBit;
  if (now == seen) return false;
  seen = now;
  return true;
#endif
}
// The same for the split's bit: the step that shows the other value than the one this loop saw last is the ride, once. With
// PICO_OLD_RULES the bit is never set and never read (the engine that has no ride).
inline bool split_flipped(uint32_t& seen, uint32_t ctl) {
#ifdef PICO_OLD_RULES
  (void)seen, (void)ctl;
  return false;
#else
  const uint32_t now = ctl & kCtlSplitBit;
  if (now == seen) return false;
  seen = now;
  return true;
#endif
}

// The part of a control word that ends a single session's response as a cancel: any bit but the watch's. (The batch loops test their own
// bits: an entry's and the pause.)
inline uint32_t ends_reply(uint32_t ctl) {
#ifdef PICO_OLD_RULES
  return ctl;
#else
  return ctl & ~(kCtlBit | kCtlSplitBit);
#endif
}

// MemAvailable from /proc/meminfo's text (a missing line: 0)
inline uint64_t parse_mem_available(const char* meminfo) {
  const char* p = std::strstr(meminfo, "MemAvailable:");
  unsigned long long kb = 0;
  if (!p || std::sscanf(p, "MemAvailable: %llu kB", &kb) != 1) return 0;
  return (uint64_t)kb << 10;
}

// MemAvailable now, bytes (0 when /proc/meminfo cannot be read)
inline uint64_t read_mem_available() {
  std::FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char text[8192];
  const size_t n = std::fread(text, 1, sizeof text - 1, f);
  std::fclose(f);
  text[n] = 0;
  return parse_mem_available(text);
}

struct Watch {
  uint64_t keep = 0;
  uint64_t last_low = UINT64_MAX;  // the lowest reading reported in the episode now running
  uint64_t low_count = 0;          // lows reported since the start
  uint64_t min_bytes = 0;          // the smallest of the last readings of the four ranks
  uint64_t last[4] = {};           // the four ranks' last readings
  int min_rank = 0;

  // The four readings. true: the smallest is a new low under keep - 64 MiB, to be printed.
  bool observe(const Sample* s, int n = 4) {
    uint64_t lo = UINT64_MAX;
    for (int r = 0; r < n; ++r)
      if (s[r].avail < lo) lo = s[r].avail, min_rank = r;
    min_bytes = lo;
    for (int r = 0; r < n && r < 4; ++r) last[r] = s[r].avail;
    if (lo >= keep) {
      last_low = UINT64_MAX;
      return false;
    }
    const bool under = keep > kMargin && lo < keep - kMargin;
    if (!under) return false;
#ifdef PICO_OLD_RULES
    ++low_count;
    return true;  // without the rule: every reading under the threshold prints
#else
    if (lo >= last_low) return false;
    last_low = lo;
    ++low_count;
    return true;
#endif
  }
};

// The line printed: the rank, its reading, the memory left to the host, and the lines known to be over their caps (text from the caller).
inline std::string low_line(int rank, uint64_t avail, uint64_t keep, const std::string& over) {
  char b[256];
  std::snprintf(b, sizeof b, "memory: rank %d has %.2f GiB available, under the %.2f GiB left to the host; lines over their cap: ", rank,
                avail / 1073741824.0, keep / 1073741824.0);
  return std::string(b) + (over.empty() ? "none known" : over);
}

}  // namespace memwatch
