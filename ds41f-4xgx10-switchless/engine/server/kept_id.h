// The last part of a start's kept-session id (main.cpp run_id), after the binary's bytes and the weight, Engram and abliteration
// files: the options that change what a request computes, then the image encoder's set digest (vit::VisionWeights::digest, equal on
// the four ranks; 0 without an encoder). A kept session's image rows were made by that encoder, so a start with another set (or with
// none, or one where there was none) starts clean. Pure, host only: tests/kept_id_test.cpp. Built with -DPICO_OLD_RULES the encoder
// is left out, as it was, and the test fails.
#pragma once
#include <cstddef>
#include <cstdint>

#include "common/keepfile.h"

namespace keptid {

inline uint64_t fold(uint64_t h, const int64_t* opts, size_t n, uint64_t vision_set) {
  h = keepfile::fnv(opts, n * sizeof(int64_t), h);
#ifndef PICO_OLD_RULES
  h = keepfile::fnv(&vision_set, sizeof vision_set, h);
#else
  (void)vision_set;
#endif
  return h;
}

}  // namespace keptid
