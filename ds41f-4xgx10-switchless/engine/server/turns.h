// The turns' line between runs: which waiting request leads the next run (no engine state; tested by
// tests/turns_test.cpp). A run's rows share one YaRN factor (the kernels rotate a step's rows by one table), so
// requests of different factors cannot decode together and take turns instead.
#pragma once

#include <algorithm>
#include <utility>

namespace turns {

// Before a run: when a request of another factor than the last run's waits, the first such request leads the run, so
// the factors alternate. Without it a request paused part-way through its turn (with drafts, the last row of a batch
// whose others ended their turn pauses to go on alone), which goes back to the front of the line, leads every run at
// its factor, and a request of another factor runs only when every request of that factor ends its turn in the same
// run. Requests of the same factor keep their order; with one factor waiting, nothing moves.
template <class Line, class FacOf>
void lead_other_factor(Line& line, int last_fac, FacOf fac_of) {
  if (last_fac == 0) return;
  auto it = std::find_if(line.begin(), line.end(), [&](const auto& f) { return fac_of(f) != last_fac; });
  if (it == line.end() || it == line.begin()) return;
  auto f = std::move(*it);
  line.erase(it);
  line.push_front(std::move(f));
}

}  // namespace turns
