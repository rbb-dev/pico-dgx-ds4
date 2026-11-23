// The engine's fatal error: the message to stderr, then abort. A host test installs a handler (die_handler) that throws, so
// "the process ended here" is something a test can observe; the engine installs none.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

using DieHandler = void (*)(const std::string& message);
inline DieHandler g_die_handler = nullptr;

// The rank, set once at the start: the fatal line then begins with the rank and the wall time, which a worker's log otherwise
// lacks. -1 (a tool, a test): no prefix. A handler (tests) is called with the message alone.
inline int g_die_rank = -1;
inline void die_set_rank(int rank) { g_die_rank = rank; }
inline void die_prefix(char* out, size_t n) {
  if (g_die_rank < 0) {
    if (n) out[0] = 0;
    return;
  }
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  tm t;
  time_t sec = ts.tv_sec;
  localtime_r(&sec, &t);
  std::snprintf(out, n, "rank %d %02d:%02d:%02d.%03d: ", g_die_rank, t.tm_hour, t.tm_min, t.tm_sec, (int)(ts.tv_nsec / 1000000));
}

[[noreturn]] inline void die(const char* fmt, ...) {
  if (g_die_handler) {
    va_list ap;
    va_start(ap, fmt);
    va_list cp;
    va_copy(cp, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, cp);
    va_end(cp);
    std::string msg((size_t)(n > 0 ? n : 0), '\0');
    std::vsnprintf(msg.data(), msg.size() + 1, fmt, ap);
    va_end(ap);
    g_die_handler(msg);
  }
  char pre[48];
  die_prefix(pre, sizeof pre);
  std::fputs(pre, stderr);
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
  std::abort();
}
