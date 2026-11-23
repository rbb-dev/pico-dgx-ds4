// The byte account of a Spark at ready: what the system says is in use, what the engine and the kernel and the other processes
// account for, and the unexplained remainder as its own line. Measured, never summed from the engine's own lines alone: the
// remainder is what a soak must keep flat. Host only (tests/byte_account_test.cpp).
//
//   used        = MemTotal - MemAvailable
//   explained   = device memory in use (CUDA's total less free: on a GB10 the same counter as the system's, every process's)
//                 + this process's resident host memory + Slab + KernelStack + PageTables + Shmem
//                 + the other processes' resident host memory
//   unexplained = used - explained   (signed: shared pages are in more than one of them)
#pragma once
#include <dirent.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace byte_account {

// One rank's reading, bytes. Plain numbers: it goes through one host all-gather.
struct Sample {
  uint64_t total = 0, avail = 0;
  uint64_t slab = 0, kernel_stack = 0, page_tables = 0, shmem = 0;
  uint64_t device_used = 0;  // cudaMemGetInfo: total less free
  uint64_t own_rss = 0;      // this process's resident host memory
  uint64_t others_rss = 0;   // every other process's
};

inline uint64_t meminfo_kb(const char* text, const char* key) {
  const size_t n = std::strlen(key);
  for (const char* p = text; (p = std::strstr(p, key)); p += n) {
    if ((p == text || p[-1] == '\n') && p[n] == ':') {
      unsigned long long kb = 0;
      if (std::sscanf(p + n + 1, " %llu kB", &kb) == 1) return (uint64_t)kb << 10;
    }
  }
  return 0;
}

// The system's part from /proc/meminfo's text (a missing line is 0).
inline void parse_meminfo(const char* text, Sample* s) {
  s->total = meminfo_kb(text, "MemTotal"), s->avail = meminfo_kb(text, "MemAvailable");
  s->slab = meminfo_kb(text, "Slab"), s->kernel_stack = meminfo_kb(text, "KernelStack");
  s->page_tables = meminfo_kb(text, "PageTables"), s->shmem = meminfo_kb(text, "Shmem");
}

// VmRSS of a /proc/PID/status text, bytes (0: none)
inline uint64_t parse_vmrss(const char* status) { return meminfo_kb(status, "VmRSS"); }

inline std::string slurp(const char* path) {
  std::string out;
  if (FILE* f = std::fopen(path, "r")) {
    char b[8192];
    size_t n;
    while ((n = std::fread(b, 1, sizeof b - 1, f)) > 0) out.append(b, n);
    std::fclose(f);
  }
  return out;
}

// Reads the system's part and every process's resident memory from /proc: this process's into own_rss, the rest into others_rss.
inline Sample read_system(uint64_t device_used) {
  Sample s;
  parse_meminfo(slurp("/proc/meminfo").c_str(), &s);
  s.device_used = device_used;
  const long self = (long)getpid();
  if (DIR* d = opendir("/proc")) {
    while (const dirent* e = readdir(d)) {
      char* end = nullptr;
      const long pid = std::strtol(e->d_name, &end, 10);
      if (!end || *end || pid <= 0) continue;
      char path[64];
      std::snprintf(path, sizeof path, "/proc/%ld/status", pid);
      const uint64_t rss = parse_vmrss(slurp(path).c_str());
      (pid == self ? s.own_rss : s.others_rss) += rss;
    }
    closedir(d);
  }
  return s;
}

inline int64_t used_of(const Sample& s) { return (int64_t)s.total - (int64_t)s.avail; }
inline int64_t explained_of(const Sample& s) {
  return (int64_t)(s.device_used + s.own_rss + s.slab + s.kernel_stack + s.page_tables + s.shmem + s.others_rss);
}
inline int64_t unexplained_of(const Sample& s) { return used_of(s) - explained_of(s); }

// Whether the remainder is inside `bound` bytes either way (the bound is set from the first soak: 0 = not set, nothing is bounded).
inline bool within(const Sample& s, uint64_t bound) {
  if (bound == 0) return true;
  const int64_t u = unexplained_of(s);
  return (u < 0 ? -u : u) <= (int64_t)bound;
}

// The start table's block: one row a kind of memory, a column a rank, GiB.
inline std::string table(const Sample* all, int n) {
  constexpr double gib = 1073741824.0;
  std::string o = "byte account, GiB by rank 0-" + std::to_string(n - 1) + " (measured at ready; MemTotal, MemAvailable, /proc):\n";
  char b[160];
  const auto row = [&](const char* label, auto f) {
    std::snprintf(b, sizeof b, "  %-46s", label);
    o += b;
    for (int r = 0; r < n; ++r) {
      std::snprintf(b, sizeof b, " %7.2f", (double)f(all[r]) / gib);
      o += b;
    }
    o += "\n";
  };
  row("MemTotal", [](const Sample& s) { return (int64_t)s.total; });
  row("MemAvailable", [](const Sample& s) { return (int64_t)s.avail; });
  row("in use (MemTotal - MemAvailable)", [](const Sample& s) { return used_of(s); });
  row("  device memory in use (CUDA, every process)", [](const Sample& s) { return (int64_t)s.device_used; });
  row("  this process, resident", [](const Sample& s) { return (int64_t)s.own_rss; });
  row("  Slab + KernelStack + PageTables + Shmem", [](const Sample& s) { return (int64_t)(s.slab + s.kernel_stack + s.page_tables + s.shmem); });
  row("  the other processes, resident", [](const Sample& s) { return (int64_t)s.others_rss; });
  row("unexplained remainder (signed)", [](const Sample& s) { return unexplained_of(s); });
  return o;
}

}  // namespace byte_account
