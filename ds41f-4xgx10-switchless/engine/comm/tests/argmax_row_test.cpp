// argmax_row_test: the packing and the winner rule of argmax_rows_row (comm/argmax_row.h) against the replicated Markov
// table. Four ranks each hold their vocabulary slice's rows and post a candidate in their own slice with that row; the
// winner's carried row must be the row of the winning id in the whole table, byte for byte, over many random rows with ties
// in value, equal values at different ids, and ids at the slices' edges. The send block's layout is checked too. Host only.
// Built with -DPICO_OLD_RULES ties go to the highest id and the tie rows fail.
//   argmax_row_test           exit 0 when every check passes
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "comm/argmax_row.h"

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++fails;
}

struct Cand {
  float v;
  int i;
};

constexpr int kPart = 32320, kRowWords = 16;  // 64-byte rows stand in for the 512-byte Markov rows
uint64_t g = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
  g = g * 6364136223846793005ull + 1442695040888963407ull;
  return uint32_t(g >> 33);
}
uint32_t table_word(int id, int w) { return uint32_t(id) * 2654435761u + uint32_t(w) * 40503u; }

}  // namespace

int main() {
  using namespace comm;
  expect(row_block_header(1, 8) == 16 && row_block_header(2, 8) == 16 && row_block_header(3, 8) == 32,
         "the candidates are padded to 16 bytes");
  expect(row_block_bytes(5, 8, 512) == 48 + 5 * 512, "a block is the padded candidates, then a row each");
  expect(slice_holds(0, 0, kPart) && slice_holds(kPart - 1, 0, kPart) && !slice_holds(kPart, 0, kPart) &&
             slice_holds(2 * kPart, 2 * kPart, kPart) && !slice_holds(2 * kPart - 1, 2 * kPart, kPart),
         "a slice holds exactly its own ids");
  long bad_id = 0, bad_row = 0, ties = 0;
  for (int it = 0; it < 200000; it++) {
    Cand c[4];
    const int tie_kind = it % 4;  // 0: distinct values, 1: all equal, 2: two equal on top, 3: the top shared by ranks 1 and 3
    for (int k = 0; k < 4; k++) {
      const int edge = rnd() % 8;
      const int off = edge == 0 ? 0 : edge == 1 ? kPart - 1 : int(rnd() % kPart);
      c[k].i = k * kPart + off;
      c[k].v = tie_kind == 1 ? 1.5f : float(int(rnd() % 1000)) / 7.f;
    }
    if (tie_kind == 2) c[2].v = c[0].v = 99.f;
    if (tie_kind == 3) c[1].v = c[3].v = 99.f;
    const int w = winning_rank(c);
    // the reference: the largest value, ties to the lowest id
    int want = 0;
    for (int k = 1; k < 4; k++)
      if (c[k].v > c[want].v || (c[k].v == c[want].v && c[k].i < c[want].i)) want = k;
    ties += tie_kind != 0;
    bad_id += c[w].i != c[want].i;
    // the carried row: the winning rank's post for its own candidate, equal to the whole table's row of the winning id
    for (int x = 0; x < kRowWords; x++) {
      const uint32_t posted = slice_holds(c[w].i, w * kPart, kPart) ? table_word(c[w].i, x) : 0u;
      bad_row += posted != table_word(c[want].i, x);
    }
  }
  expect(ties > 100000, "the draws hold " + std::to_string(ties) + " rows with equal values");
  expect(bad_id == 0, "winners == the largest value, ties to the lowest id: " + std::to_string(bad_id) + " differ");
  expect(bad_row == 0, "the carried row == the replicated table's row of the winning id: " + std::to_string(bad_row) + " words differ");
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
