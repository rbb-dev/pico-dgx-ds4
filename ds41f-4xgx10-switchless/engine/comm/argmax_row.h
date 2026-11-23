// argmax_rows_row's contract (collectives.h): each rank's send block is its `rows` candidates (value, id) padded to 16
// bytes, then one row of row_bytes per candidate: the table row of the candidate's id when this rank's table slice holds
// it (a vocabulary-split table: the rank's own candidate always lies in its own slice), else zeros. The winner of a row is
// the largest value, ties to the lowest id (argmax_rows' rule); its row is the winning rank's.
#pragma once
#include <cstddef>

#include "common/embed_shard.h"

namespace comm {

DSH_FN size_t row_block_header(int rows, size_t cand_bytes) { return (static_cast<size_t>(rows) * cand_bytes + 15) / 16 * 16; }
DSH_FN size_t row_block_bytes(int rows, size_t cand_bytes, size_t row_bytes) {
  return row_block_header(rows, cand_bytes) + static_cast<size_t>(rows) * row_bytes;
}
// True when `a` beats `b`.
template <class A>
DSH_FN bool candidate_wins(const A& a, const A& b) {
#ifdef PICO_OLD_RULES
  return a.v > b.v || (a.v == b.v && a.i > b.i);
#else
  return a.v > b.v || (a.v == b.v && a.i < b.i);
#endif
}
// The winning rank of one row's four candidates (rank order; an equal value and id keeps the earlier rank).
template <class A>
DSH_FN int winning_rank(const A (&c)[4]) {
  int w = 0;
  for (int k = 1; k < 4; k++)
    if (candidate_wins(c[k], c[w])) w = k;
  return w;
}
// Whether a rank whose slice starts at `base` and holds `rows` table rows holds row `id`.
DSH_FN bool slice_holds(int id, int base, int rows) { return static_cast<unsigned>(id - base) < static_cast<unsigned>(rows); }

}  // namespace comm
