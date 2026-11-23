// The allocation seal: after the pool's sizing read, every allocation call the engine's own code makes is inside a scope that
// names its class (one `alloc_seal::Scoped` line in the function that makes the call, on the calling thread), or it is counted
// as outside the store. Before the read nothing is counted. common/alloc_wrap.cpp puts a counting function in front of each call
// of the audit's list. An outside call never stops the engine: it is counted and printed once a site, and the ready line
// carries the totals. Without the rule (-DPICO_OLD_RULES) a call is counted nowhere.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace alloc_seal {

enum Scope : int { kNone = 0, kStore = 1, kRegion = 2, kSide = 3 };

struct Counts {
  uint64_t side, store, region, outside;
};

constexpr int kSites = 64;  // outside sites whose line is printed; more sites are counted and not printed

struct State {
  std::atomic<bool> open{false};
  std::atomic<uint64_t> n[4]{};  // by Scope
  std::atomic<uintptr_t> seen[kSites]{};
  void (*sink)(const char*) = nullptr;  // where a line goes; null = stdout
};

inline State g_state;

inline Scope& current() {
  thread_local Scope s = kNone;
  return s;
}

// A scope over the calls of the enclosing block, on this thread.
class Scoped {
 public:
  explicit Scoped(Scope s) : prev_(current()) { current() = s; }
  ~Scoped() { current() = prev_; }
  Scoped(const Scoped&) = delete;
  Scoped& operator=(const Scoped&) = delete;

 private:
  Scope prev_;
};

// The sizing read has happened: from here calls are counted.
inline void open() { g_state.open.store(true, std::memory_order_release); }

// Tests: back to the state before the read.
inline void reset() {
  g_state.open.store(false);
  for (auto& c : g_state.n) c.store(0);
  for (auto& s : g_state.seen) s.store(0);
  g_state.sink = nullptr;
}

inline Counts counts() {
  return {g_state.n[kSide].load(), g_state.n[kStore].load(), g_state.n[kRegion].load(), g_state.n[kNone].load()};
}

// One counted call: its name, its size, the caller's return address. A cheap compare when the seal is not open.
inline void note(const char* call, uint64_t bytes, const void* ret) {
#ifndef PICO_OLD_RULES
  if (!g_state.open.load(std::memory_order_acquire)) return;
  const Scope s = current();
  g_state.n[s].fetch_add(1, std::memory_order_relaxed);
  if (s != kNone) return;
  const uintptr_t a = reinterpret_cast<uintptr_t>(ret);
  for (auto& slot : g_state.seen) {
    uintptr_t cur = slot.load(std::memory_order_relaxed);
    if (cur == a) return;  // printed already
    if (cur == 0 && slot.compare_exchange_strong(cur, a)) {
      char line[192];
      std::snprintf(line, sizeof line, "alloc: outside the store: %s of %llu bytes after the sizing read (return address 0x%llx)\n", call,
                    (unsigned long long)bytes, (unsigned long long)a);
      if (g_state.sink) g_state.sink(line);
      else std::fputs(line, stdout);
      return;
    }
    if (cur == a) return;
  }
#else
  (void)call, (void)bytes, (void)ret;
#endif  // without the rule: a call after the read is counted nowhere
}

// The ready line, every rank.
inline int ready_line(char* buf, size_t n) {
  const Counts c = counts();
  return std::snprintf(buf, n, "alloc: since the sizing read: %llu side, %llu store, %llu region calls, %llu outside the store",
                       (unsigned long long)c.side, (unsigned long long)c.store, (unsigned long long)c.region, (unsigned long long)c.outside);
}

}  // namespace alloc_seal
