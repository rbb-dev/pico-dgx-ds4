// embed_shard_test: the embedding's vocabulary cut and its exchange (common/embed_shard.h) against the replicated row. Every
// token id 0 .. 129,279: rank r's slot (the owner's row, bf16 -0.0 on the other three) summed over the four ranks the way the
// ring's all-reduce sums (fp32, ((c0 + c1) + c2) + c3, then bf16) is the replicated row, byte for byte, on a table that holds
// +0.0 and -0.0 elements; prefill's pick (the owner's slot of each row) is it too; ids outside the table are found. Host
// only. Built with -DPICO_OLD_RULES the other ranks write +0.0 and the rows with a -0.0 element fail.
//   embed_shard_test            exit 0 when every check passes
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/embed_shard.h"

using namespace embed_shard;

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++fails;
}

constexpr int kWords = 8;  // a row of 16 bf16: the cut does not look at the width

uint32_t mix(uint32_t x) {
  x ^= x >> 16, x *= 0x7feb352dU, x ^= x >> 15, x *= 0x846ca68bU, x ^= x >> 16;
  return x;
}
// An element: a finite bf16 of any sign, with +0.0 and -0.0 in one of eight (the cases the sum must keep apart).
uint16_t element(uint32_t id, uint32_t c) {
  const uint32_t h = mix(id * 31 + c);
  if ((h & 7) == 0) return (h >> 8) & 1 ? 0x8000 : 0x0000;
  uint16_t v = uint16_t(h >> 16);
  if ((v & 0x7f80) == 0x7f80) v &= 0xbf7f;  // no infinity and no NaN
  return v;
}

float bf2f(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f2bf(float f) {  // round to nearest even (no NaN here)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}

}  // namespace

int main() {
  // the shards: rank r's rows [r * kPart, +kPart) of the replicated table (the table is element(id, c))
  std::vector<std::vector<uint32_t>> shard(kRanks, std::vector<uint32_t>((size_t)kPart * kWords));
  for (int r = 0; r < kRanks; r++)
    for (int row = 0; row < kPart; row++)
      for (int w = 0; w < kWords; w++) {
        const uint32_t id = uint32_t(r * kPart + row);
        shard[r][(size_t)row * kWords + w] = uint32_t(element(id, 2 * w)) | uint32_t(element(id, 2 * w + 1)) << 16;
      }
  expect(kPart == 32320 && kVocab == 129280, "four ranks of 32,320 rows");

  long bad_sum = 0, bad_pick = 0, minus_zero_rows = 0;
  for (int id = 0; id < kVocab; id++) {
    uint32_t slot[kRanks][kWords];
    for (int r = 0; r < kRanks; r++)
      for (int w = 0; w < kWords; w++) slot[r][w] = slot_word(shard[r].data(), r * kPart, id, kWords, w);
    bool has_neg_zero = false;
    for (int w = 0; w < kWords; w++)
      for (int half = 0; half < 2; half++) {
        const int c = 2 * w + half;
        float acc = bf2f(uint16_t(slot[0][w] >> (16 * half)));
        for (int r = 1; r < kRanks; r++) acc += bf2f(uint16_t(slot[r][w] >> (16 * half)));  // rank order
        const uint16_t want = element(uint32_t(id), uint32_t(c));
        has_neg_zero |= want == 0x8000;
        if (f2bf(acc) != want) ++bad_sum;
      }
    minus_zero_rows += has_neg_zero;
    // prefill: the owner's slot is the row
    const int o = owner(id);
    for (int w = 0; w < kWords; w++)
      if (slot[o][w] != (uint32_t(element(uint32_t(id), 2 * w)) | uint32_t(element(uint32_t(id), 2 * w + 1)) << 16))
        ++bad_pick;
  }
  expect(minus_zero_rows > 10000, "the table has -0.0 elements in " + std::to_string(minus_zero_rows) + " of its 129,280 rows");
  expect(bad_sum == 0, "all 129,280 ids: sharded rows summed over four slots == the replicated row, " +
                           std::to_string(bad_sum) + " elements differ");
  expect(bad_pick == 0, "prefill's pick from the owner's slot == the replicated row, " + std::to_string(bad_pick) + " words differ");

  expect(owner(0) == 0 && owner(kPart - 1) == 0 && owner(kPart) == 1 && owner(kVocab - 1) == 3, "owners at the cut's edges");
  expect(id_ok(0) && id_ok(kVocab - 1) && !id_ok(-1) && !id_ok(kVocab) && !id_ok(1 << 30),
         "an id below 0 or from 129,280 up has no row");
  // an id with no owner: owner() would say rank 4 and no rank writes it; the checks find it before it becomes a row of -0.0
  const int ids[5] = {5, 129279, 70000, 129280, 3};
  expect(first_bad_id(ids, 3) == -1 && first_bad_id(ids, 5) == 3, "a chunk's ids: the first without a row is found");
  const int neg[2] = {7, -1};
  expect(first_bad_id(neg, 2) == 1, "and a negative one");
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
