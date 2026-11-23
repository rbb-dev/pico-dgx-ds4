// The embedding and the Markov embedding cut by vocabulary rows: rank r holds token ids [r * kPart, (r + 1) * kPart),
// the rows its head slice scores. A lookup is an exchange: the rank that owns a token's id writes the table's row into
// its slot, every other rank writes bf16 -0.0 (0x8000) there, and the all-reduce (fp32 sum in rank order, then bf16)
// returns the row bit for bit. x + (-0.0) == x for every x, +0.0 and -0.0 included, so the sum of one row and three
// -0.0 rows is that row in any order; +0.0 in the other slots would turn a -0.0 element into +0.0.
// Prefill moves whole rows to the rank that needs them (an all-to-all) and picks each from its owner's slot.
#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define DSH_FN __host__ __device__ __forceinline__
#else
#define DSH_FN inline
#endif

namespace embed_shard {

constexpr int kRanks = 4, kVocab = 129280, kPart = kVocab / kRanks;  // 32,320 rows a rank
constexpr uint32_t kNegZeroPair = 0x80008000u;                       // two bf16 -0.0 in one word

// A token id the tables hold. An id outside it has no owner: the exchange would hand back a row of -0.0.
DSH_FN bool id_ok(int id) { return id >= 0 && id < kVocab; }
// The rank whose slice holds the id's row (id_ok(id) first).
DSH_FN int owner(int id) { return id / kPart; }

// Word w of the slot a rank with first row `base` (rank * kPart) writes for token `id`; the table's row has
// `row_words` 32-bit words. Under PICO_OLD_RULES the other ranks write +0.0, which the exactness test catches.
DSH_FN uint32_t slot_word(const uint32_t* shard, int base, int id, int row_words, int w) {
  if (static_cast<unsigned>(id - base) < static_cast<unsigned>(kPart))
    return shard[static_cast<size_t>(id - base) * row_words + w];
#ifdef PICO_OLD_RULES
  return 0u;
#else
  return kNegZeroPair;
#endif
}

// The first index whose id has no row, or -1: the host's check of a chunk's ids before they are submitted.
inline int first_bad_id(const int* ids, int n) {
  for (int i = 0; i < n; ++i)
    if (!id_ok(ids[i])) return i;
  return -1;
}

}  // namespace embed_shard
