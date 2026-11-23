// The engine's last words: one handler for SIGABRT, SIGSEGV, SIGBUS, SIGFPE and SIGILL that writes the signal, the rank, the wall time and the call stack to
// stderr and then lets the default action happen (the handler resets itself, so returning from it re-raises). A worker's log has no time on its lines: these
// two lines, this and die()'s prefix, are what a crash is read from. The handler calls only async-signal-safe functions (clock_gettime, write,
// backtrace_symbols_fd): the rank's text and the local time's offset are made at install, and backtrace is called once there so that its first call in the
// handler does not load a library and allocate.
#pragma once
#include <execinfo.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace lastwords {
inline int g_rank = -1;
inline char g_rank_text[16];  // the rank in decimal, made at install
inline int g_rank_len = 0;
inline long g_gmtoff = 0;  // seconds east of UTC at install
inline void put(char*& p, const char* s) {
  while (*s) *p++ = *s++;
}
inline void put_num(char*& p, long v, int width) {  // v in decimal, at least `width` digits (zero padded), no sign
  char d[24];
  int k = 0;
  do d[k++] = (char)('0' + v % 10), v /= 10;
  while (v > 0 || k < width);
  while (k) *p++ = d[--k];
}
inline const char* name_of(int sig) {  // strsignal's words, as literals
  switch (sig) {
    case SIGABRT: return "Aborted";
    case SIGSEGV: return "Segmentation fault";
    case SIGBUS: return "Bus error";
    case SIGFPE: return "Floating point exception";
    case SIGILL: return "Illegal instruction";
    default: return "Unknown signal";
  }
}
inline void handler(int sig) {
  char b[160];
  char* p = b;
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  const long day = (long)(ts.tv_sec + g_gmtoff) % 86400, sod = day < 0 ? day + 86400 : day;
  put(p, "rank ");
  put(p, g_rank_text);
  *p++ = ' ';
  put_num(p, sod / 3600, 2), *p++ = ':';
  put_num(p, sod / 60 % 60, 2), *p++ = ':';
  put_num(p, sod % 60, 2), *p++ = '.';
  put_num(p, ts.tv_nsec / 1000000, 3);
  put(p, ": signal ");
  put_num(p, sig, 1);
  put(p, " (");
  put(p, name_of(sig));
  put(p, "): call stack follows\n");
  (void)!::write(2, b, (size_t)(p - b));
  void* frames[64];
  const int k = backtrace(frames, 64);
  backtrace_symbols_fd(frames, k, 2);
}
inline void install(int rank) {
  g_rank = rank;
  char* p = g_rank_text;
  if (rank < 0) *p++ = '-';
  put_num(p, rank < 0 ? -(long)rank : rank, 1);
  *p = 0;
  g_rank_len = (int)(p - g_rank_text);
  time_t now = time(nullptr);
  struct tm t;
  if (localtime_r(&now, &t)) g_gmtoff = t.tm_gmtoff;
  void* prime[2];
  backtrace(prime, 2);  // the first call loads the unwinder and allocates: here, not in a handler
  struct sigaction sa;
  std::memset(&sa, 0, sizeof sa);
  sa.sa_handler = handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND | SA_NODEFER;
  for (const int sig : {SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL}) sigaction(sig, &sa, nullptr);
}
}  // namespace lastwords
