// The tower's work-area layout, shared by the unsplit run, the four-rank run (vit_tower.cu) and a rank's own run
// (vit_rank.cu). Internal to engine/vision: nothing outside includes it.
#pragma once
#include <cstddef>
#include <vector>

#include "vision/vit.h"

namespace vit {

// Offsets into one work area, 256-byte aligned.
struct Arena {
  size_t off = 0;
  size_t take(size_t elems, size_t elem_bytes = 2) {
    const size_t o = off;
    off += (elems * elem_bytes + 255) & ~size_t(255);
    return o;
  }
};

// One rank's buffers of the split run for n patches and m aligner tokens.
struct Split {
  size_t x, xn, qkv, big, y, cos, sin, inv, patch, unf, g1, g2, dig, total;
  Split(int n, int m) {
    Arena a;
    x = a.take((size_t)n * 1024), xn = a.take((size_t)n * 1024), qkv = a.take((size_t)n * 768);
    big = a.take((size_t)n * 2816), y = a.take((size_t)n * 1408);
    cos = a.take((size_t)n * 32, 4), sin = a.take((size_t)n * 32, 4), inv = a.take(16, 4);
    patch = a.take((size_t)n * kPatchKPad), unf = a.take((size_t)m * kMergeIn);
    g1 = a.take((size_t)m * kLm), g2 = a.take((size_t)m * kLm), dig = a.take((size_t)(n + 15) / 16, 8);
    total = a.off;
  }
};

template <class T>
T* ptr(void* base, size_t off) {
  return reinterpret_cast<T*>(static_cast<char*>(base) + off);
}

// the 16 inverse frequencies in host memory: rope_table copies them by value into its kernel's arguments
inline const float* inv_freq_host() {
  static const std::vector<float> f = rope_inv_freq();
  return f.data();
}

}  // namespace vit
