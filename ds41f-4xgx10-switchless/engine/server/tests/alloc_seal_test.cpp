// common/alloc_seal.h: nothing is counted before the sizing read; after it a scoped call counts in its class, an unscoped one is
// outside and printed once a site; scopes nest and are per thread; the ready line carries the totals. Built with -DPICO_OLD_RULES
// (the rule off: a call after the read is counted nowhere) the cases marked "red" fail. Host only.
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "common/alloc_seal.h"

namespace {
int fails = 0;
std::string lines;
void sink(const char* l) { lines += l; }
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}
int count_lines() {
  int n = 0;
  for (char c : lines) n += c == '\n';
  return n;
}
}  // namespace

int main() {
  using namespace alloc_seal;
  reset();
  g_state.sink = sink;
  int a = 0, b = 0;
  note("cudaMalloc", 100, &a);
  Counts c = counts();
  check(c.side + c.store + c.region + c.outside == 0 && lines.empty(), "before the read nothing is counted or printed");

  open();
  { Scoped s(kStore); note("cuMemCreate", 2 << 20, &a); }
  { Scoped s(kRegion); note("cudaMalloc", 1 << 20, &a); note("cudaMalloc", 1 << 20, &a); }
  { Scoped s(kSide); note("mmap", 4096, &a); }
  c = counts();
  check(c.store == 1 && c.region == 2 && c.side == 1 && c.outside == 0, "red: after the read a scoped call counts in its class");
  check(lines.empty(), "a scoped call prints nothing");

  note("cudaMalloc", 1048576, &a);
  note("cudaMalloc", 1048576, &a);
  note("cudaMalloc", 1048576, &b);
  c = counts();
  check(c.outside == 3, "red: an unscoped call after the read is outside, every time");
  check(count_lines() == 2, "red: its line is printed once a site (two sites, three calls)");
  check(lines.find("outside the store: cudaMalloc of 1048576 bytes after the sizing read (return address 0x") != std::string::npos,
        "red: the line names the call, its bytes and the return address");

  {
    Scoped outer(kRegion);
    { Scoped inner(kSide); note("mmap", 8, &a); }
    note("mmap", 8, &a);
  }
  c = counts();
  check(c.side == 2 && c.region == 3, "red: scopes nest and restore the outer one");
  note("mmap", 8, &a);
  check(counts().outside == 4, "red: after the scope ends the call is outside again");

  Counts before = counts();
  std::thread t([&] { Scoped s(kStore); note("cuMemCreate", 1, &a); });
  note("cuMemCreate", 1, &a);  // this thread has no scope while the other's is open
  t.join();
  c = counts();
  check(c.store == before.store + 1 && c.outside == before.outside + 1, "red: a scope is the thread's own");

  char buf[256];
  ready_line(buf, sizeof buf);
  char want[256];
  std::snprintf(want, sizeof want, "alloc: since the sizing read: %llu side, %llu store, %llu region calls, %llu outside the store",
                (unsigned long long)c.side, (unsigned long long)c.store, (unsigned long long)c.region, (unsigned long long)c.outside);
  check(std::strcmp(buf, want) == 0 && c.outside > 0, "red: the ready line carries the four counts");

  if (fails) return 1;
  std::printf("all cases ok\n");
  return 0;
}
