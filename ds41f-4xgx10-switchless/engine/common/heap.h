// The process's glibc heap: how many arenas it may spread over, giving free pages back, and what it holds now. Threads
// that allocate and free in turn leave free pages in each arena; with at most two arenas and a trim after the work that
// freed them, the heap stays near what is live instead of the most it ever held.
#pragma once

#include <malloc.h>

#include <cstddef>

namespace heap {

constexpr int kArenas = 2;  // arenas the process may use (M_ARENA_MAX)
constexpr size_t kCap = 1ull << 30;  // what the ledger allows the heap

// Once, before the process starts threads.
inline void limit_arenas() {
#ifndef PICO_OLD_RULES
  mallopt(M_ARENA_MAX, kArenas);
#endif  // without the rule: one arena per busy thread, as glibc chooses
}

// Returns the free pages of every arena to the system (after a checkpoint, when the buffers it used are freed).
inline void release() {
#ifndef PICO_OLD_RULES
  malloc_trim(0);
#endif  // without the rule: free pages stay in the arenas
}

// Bytes the heap holds from the system: arenas and mapped blocks, less what is free inside them.
inline size_t in_use() {
  const struct mallinfo2 mi = mallinfo2();
  return mi.uordblks + mi.hblkhd;
}

}  // namespace heap
