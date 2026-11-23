// The oracle of a piece's quarters (model/quarter_map.h), written from the entry's arithmetic with division and remainder and
// one entry at a time, so it shares no line of the map's bit arithmetic or the kernels' unit walk. Host only; both quarter tests
// include it.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

#include "model/quarter_map.h"

namespace quarter_oracle {

using model::quarter::kEntries;
using model::quarter::kImageBytes;
using model::quarter::kQuarterBytes;
using model::quarter::kQuarterEntries;

inline const uint32_t kSize[4] = {256, 32, 64, 4};

// who owns entry e of array a, and where it sits in the owner's slice of that array
inline int owner(int a, uint32_t e) { return a < 2 ? (int)(e % 4) : (int)((e / 512) % 4); }
inline uint32_t local(int a, uint32_t e) { return a < 2 ? e / 4 : (e / 2048) * 512 + e % 512; }

inline size_t slice(int a, bool whole) {  // an array's byte offset in a whole piece's image or in a quarter
  size_t o = 0;
  for (int j = 0; j < a; ++j) o += (size_t)(whole ? kEntries : kQuarterEntries) * kSize[j];
  return o;
}

// rank q's quarter of `image` (kImageBytes), entries from `written` on zero
inline std::vector<uint8_t> quarter_of(const std::vector<uint8_t>& image, int q, uint32_t written) {
  std::vector<uint8_t> out(kQuarterBytes, 0);
  for (int a = 0; a < 4; ++a)
    for (uint32_t e = 0; e < kEntries; ++e) {
      if (owner(a, e) != q || e >= written) continue;
      std::memcpy(&out[slice(a, false) + (size_t)local(a, e) * kSize[a]], &image[slice(a, true) + (size_t)e * kSize[a]], kSize[a]);
    }
  return out;
}

// the image `image` with the entries from `written` on zeroed
inline std::vector<uint8_t> written_of(const std::vector<uint8_t>& image, uint32_t written) {
  std::vector<uint8_t> out = image;
  for (int a = 0; a < 4; ++a)
    for (uint32_t e = written; e < kEntries; ++e) std::memset(&out[slice(a, true) + (size_t)e * kSize[a]], 0, kSize[a]);
  return out;
}

// four quarters in a row, to an image
inline std::vector<uint8_t> image_of(const std::vector<uint8_t>& quarters /* 4 x kQuarterBytes */) {
  std::vector<uint8_t> out(kImageBytes, 0);
  for (int a = 0; a < 4; ++a)
    for (uint32_t e = 0; e < kEntries; ++e)
      std::memcpy(&out[slice(a, true) + (size_t)e * kSize[a]],
                  &quarters[(size_t)owner(a, e) * kQuarterBytes + slice(a, false) + (size_t)local(a, e) * kSize[a]], kSize[a]);
  return out;
}

// block_sum's term, as state.cu writes it, plus the seed's term
inline uint64_t sum_of(const uint8_t* p, size_t bytes, uint64_t seed) {
  uint64_t s = 0;
  for (size_t i = 0; i < bytes / 8; ++i) {
    uint64_t w;
    std::memcpy(&w, p + i * 8, 8);
    uint64_t x = w + i * 0x9E3779B97F4A7C15ull + seed * 0xD6E8FEB86659FD93ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    s += x ^ (x >> 27);
  }
  return s;
}

inline std::vector<uint8_t> random_image(uint64_t seed) {
  std::vector<uint8_t> v(kImageBytes);
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (auto& b : v) {
    x ^= x << 13, x ^= x >> 7, x ^= x << 17;
    b = (uint8_t)(x >> 24);
  }
  return v;
}

}  // namespace quarter_oracle
