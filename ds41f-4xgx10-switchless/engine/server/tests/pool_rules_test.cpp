// model/pool_rules.h against a pool held by reservations: seven lines whose reservations go past what they hold (157
// numbers), and a bookmark's take beside them. Host only: no engine, no GPU.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../model/pool_rules.h"

namespace {

namespace rules = model::rules;

int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

struct Slot {
  int resv, held;
};

// seven lines whose reservations go past what they hold (22 each = 154, and 3 more on the last = 157)
std::vector<Slot> finished_lines() {
  std::vector<Slot> v;
  for (int i = 0; i < 7; ++i) v.push_back({40, 18});
  v[6].resv += 3;
  return v;
}

int64_t reserved_of(const std::vector<Slot>& s) {
  return rules::unheld_sum((int)s.size(), [&](int t) { return s[t].resv; }, [&](int t) { return s[t].held; });
}

}  // namespace

int main() {
  // a bookmark beside a running request's reservation: 157 reserved, 100 free, nothing stored to give
  {
    const auto g0 = [] { return 0; };
    check(reserved_of(finished_lines()) == 157, "fixture: the seven lines reserve 157 numbers");
    check(!rules::covers(100, 157, 1, g0) && rules::covers(158, 157, 1, g0), "a bookmark's take keeps the reservations covered");
  }

  std::printf("%s\n", fails ? "pool_rules_test FAILED" : "pool_rules_test passed");
  return fails ? 1 : 0;
}
