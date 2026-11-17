// The chunk pool's rules as pure functions (no engine state; tested by server/tests/pool_rules_test.cpp): what a
// bookmark may take beside the reservations of running requests. Everything in the pool that no running request uses is
// cache and gives way; these functions are where the model decides that.
#pragma once

#include <algorithm>
#include <cstdint>

namespace model::rules {

// Numbers the slots' reservations still reserve (a reservation past what the slot's session holds), over n slots.
// state.cu take_extra's `reserved`.
template <class Resv, class Held>
int64_t unheld_sum(int n, Resv resv, Held held) {
  int64_t sum = 0;
  for (int t = 0; t < n; ++t) sum += std::max(0, (int)resv(t) - (int)held(t));
  return sum;
}

// Whether the pool can spare `n` numbers beside the reservations: free ones, or free plus what evicting stored sessions gives
// (`giveable()`, asked only when free ones do not cover it). state.cu take_extra.
template <class Giveable>
bool covers(int64_t free, int64_t reserved, int64_t n, Giveable giveable) {
  if (free >= reserved + n) return true;
  return free + (int64_t)giveable() >= reserved + n;
}

}  // namespace model::rules
