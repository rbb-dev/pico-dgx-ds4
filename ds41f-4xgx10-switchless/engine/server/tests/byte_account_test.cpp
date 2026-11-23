// common/byte_account.h on the host: the parse of /proc/meminfo and /proc/PID/status texts, the account's arithmetic, the bound on
// the unexplained remainder (a memory that no listed line accounts for raises it and breaks the bound; one that a line accounts for
// does not), and the table's shape. Reads this machine's /proc once, for a sanity check of the reader only.
#include <cstdio>
#include <string>

#include "common/byte_account.h"

namespace ba = byte_account;

static int fails = 0;
static void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

int main() {
  const char* mi =
      "MemTotal:       124974080 kB\nMemFree:         2000000 kB\nMemAvailable:   30000000 kB\nShmem:             100000 kB\n"
      "Slab:              900000 kB\nKernelStack:        30000 kB\nPageTables:         50000 kB\nSwapCached: 0 kB\n";
  ba::Sample s;
  ba::parse_meminfo(mi, &s);
  check(s.total == 124974080ull << 10 && s.avail == 30000000ull << 10 && s.slab == 900000ull << 10 && s.kernel_stack == 30000ull << 10 &&
            s.page_tables == 50000ull << 10 && s.shmem == 100000ull << 10,
        "the meminfo text is read key by key (Shmem is not SwapCached, Slab is not SReclaimable)");
  check(ba::parse_vmrss("Name:\tx\nVmPeak:\t 99 kB\nVmRSS:\t  1234 kB\nVmSwap:\t0 kB\n") == 1234ull << 10, "VmRSS is read");
  check(ba::parse_vmrss("Name:\tx\n") == 0, "a status with no VmRSS reads 0");

  s.device_used = 60ull << 30, s.own_rss = 3ull << 30, s.others_rss = 4ull << 30;
  const int64_t used = ba::used_of(s), expl = ba::explained_of(s);
  check(used == (int64_t)(s.total - s.avail) && expl == (int64_t)(s.device_used + s.own_rss + s.others_rss + s.slab + s.kernel_stack + s.page_tables + s.shmem),
        "used = MemTotal - MemAvailable; explained = the listed parts");
  check(ba::unexplained_of(s) == used - expl, "the remainder is used less explained");
  // a memory nothing accounts for: MemAvailable falls by 2 GiB, no listed line moves
  ba::Sample t = s;
  t.avail -= 2ull << 30;
  check(ba::unexplained_of(t) == ba::unexplained_of(s) + (2ll << 30), "memory that no line accounts for raises the remainder by exactly its size");
  // memory a line accounts for: the other processes grow by 2 GiB and MemAvailable falls by the same
  ba::Sample u = s;
  u.avail -= 2ull << 30, u.others_rss += 2ull << 30;
  check(ba::unexplained_of(u) == ba::unexplained_of(s), "memory a line accounts for leaves the remainder where it was");
  const uint64_t bound = (uint64_t)(ba::unexplained_of(s) < 0 ? -ba::unexplained_of(s) : ba::unexplained_of(s)) + (1ull << 30);
  check(ba::within(s, bound) && !ba::within(t, bound), "the bound holds for the account and breaks for a remainder that grew past it");
  check(ba::within(t, 0), "an unset bound (0) bounds nothing");
  ba::Sample v = s;
  v.avail += 8ull << 30;
  check(ba::unexplained_of(v) < ba::unexplained_of(s) && !ba::within(v, 1ull << 20), "the bound is two-sided: an account over-explained fails it too");

  ba::Sample all[4] = {s, t, u, v};
  const std::string tab = ba::table(all, 4);
  check(tab.find("unexplained remainder") != std::string::npos && tab.find("MemAvailable") != std::string::npos &&
            tab.find("byte account") != std::string::npos,
        "the table has the remainder as its own row");

  const ba::Sample live = ba::read_system(0);
  check(live.total > 0 && live.avail > 0 && live.avail <= live.total && live.own_rss > 0, "this machine's /proc is readable by the reader (sanity of the reader only)");

  std::printf("%s byte_account_test: %d failed\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
