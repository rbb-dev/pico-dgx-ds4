// comm/gid_pick.h on the host, no device: which GID table entry a link uses for its fleet address.
// Holds:
//  G1  the address at GID_INDEX as a RoCE v2 entry: that index
//  G2  GID_INDEX holds something else and the address is a RoCE v2 entry at another index: that index (the kernel
//      can number the table differently, e.g. 4 instead of 3)
//  G3  the address only as a RoCE v1 entry, at GID_INDEX or elsewhere: none (-1)
//  G4  no entry with the address, or an empty table: none; GID_INDEX past the table: the scan finds it
//  G5  two RoCE v2 entries with the address: GID_INDEX's when it is one, else the lower index
// The control: the rule before (GID_INDEX's entry when its bytes are the address, whatever its type, else none) must
// fail: it gets 7 of the 11 rows wrong, among them all of G2 and G3 (reported as "control rejected").
// Command: gid_pick_test (exit 0 = all pass)
#include <cstdio>
#include <vector>

#include "comm/gid_pick.h"

using comm::GidEntry;

namespace {

int fails = 0, checks = 0;
void check(bool ok, const char* what, int a = 0, int b = 0) {
  ++checks;
  if (!ok) std::printf("FAIL: %s (%d, %d)\n", what, a, b);
  fails += !ok;
}

GidEntry v4(const uint8_t ip[4], bool v2) {
  GidEntry e;
  e.present = true, e.roce_v2 = v2;
  const uint8_t head[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
  std::memcpy(e.raw, head, 12), std::memcpy(e.raw + 12, ip, 4);
  return e;
}
GidEntry link_local() {  // fe80::..., the kind an interface also carries
  GidEntry e;
  e.present = true, e.roce_v2 = true;
  e.raw[0] = 0xfe, e.raw[1] = 0x80, e.raw[15] = 1;
  return e;
}

using Rule = int (*)(int, const uint8_t*, const std::vector<GidEntry>&);
int rule_now(int pref, const uint8_t* want, const std::vector<GidEntry>& t) {
  return comm::pick_gid(pref, want, (int)t.size(), [&](int i) { return t[i]; });
}
int rule_before(int pref, const uint8_t* want, const std::vector<GidEntry>& t) {  // ring.cpp before: GID_INDEX or nothing
  return pref >= 0 && pref < (int)t.size() && t[pref].present && std::memcmp(t[pref].raw, want, 16) == 0 ? pref : -1;
}

// the rows; returns how many give the wrong index
int rows(Rule rule) {
  const uint8_t a[4] = {10, 0, 0, 1}, b[4] = {10, 0, 1, 1};
  GidEntry want = v4(a, true);
  int wrong = 0;
  auto expect = [&](const std::vector<GidEntry>& t, int pref, int idx) { wrong += rule(pref, want.raw, t) != idx; };
  // G1
  expect({link_local(), link_local(), v4(a, false), v4(a, true)}, 3, 3);
  // G2: GID_INDEX=3 holds the other rail's address, the link's own is at 4; or 3 is empty
  expect({link_local(), link_local(), v4(a, false), v4(b, true), v4(a, true)}, 3, 4);
  expect({link_local(), link_local(), v4(a, false), GidEntry{}, v4(a, true)}, 3, 4);
  expect({link_local(), v4(a, true), v4(a, false), link_local()}, 3, 1);
  // G3: only as RoCE v1
  expect({link_local(), link_local(), v4(a, false), v4(a, false)}, 3, -1);
  expect({link_local(), link_local(), v4(a, false)}, 2, -1);
  // G4
  expect({link_local(), link_local(), v4(b, true), v4(b, true)}, 3, -1);
  expect({}, 3, -1);
  expect({link_local(), v4(a, true)}, 3, 1);  // GID_INDEX past the table: found by the scan
  // G5
  expect({v4(a, true), link_local(), link_local(), v4(a, true)}, 3, 3);
  expect({link_local(), v4(a, true), link_local(), v4(b, true), v4(a, true)}, 3, 1);
  return wrong;
}

}  // namespace

int main() {
  const int now = rows(rule_now), before = rows(rule_before);
  check(now == 0, "G1-G5 the rule", now);
  check(before > 0, "control: the rule before (GID_INDEX or nothing, any type) must fail", before);
  std::printf("gid_pick: the rule %d wrong rows of 11; control (the rule before) %d wrong rows%s\n", now, before,
              before > 0 ? " (control rejected)" : "");
  std::printf("gid_pick_test: %d checks, %d failed\n", checks, fails);
  return fails ? 1 : 0;
}
