// A SUBMIT that carries its images as pixels (fe::kFlagPixels) instead of rows (host only; tested by tests/pixels_test.cpp).
//
// The image part, after the prompt ids (`base` bytes into the payload):
//   u32 n                                   1 .. fe::kMaxImages
//   ImageSpan spans[n]                      as for rows: start, rows, hash
//   {u16 n_h, u16 n_w} grids[n]             in patches; the array padded with zeros to a multiple of 8 bytes
//   image i: 588 * n_h * n_w bytes of u8    the media worker's padded image, each padded to a multiple of 8 bytes
// A request carries rows or pixels, never both.
//
// parse() tells apart the two kinds of wrong frame the reader has: kMalformed (the frame's bytes cannot be accounted for: the
// connection closes) and kRefused (a well-formed frame the engine will not run: DONE with `why`, the connection lives).
//
// PICO_OLD_RULES (never defined by the engine) builds the reader that does not know the flag: every pixel frame is malformed.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../common/frontend_protocol.h"
#include "../vision/vit_weights.h"

namespace pixels {

constexpr size_t kPatchBytes = vit::kPatchK;  // u8 values of one patch: 3 x 14 x 14
constexpr uint32_t kMaxRows = 1024;           // the encoder's bound on one image's span (IMAGE_START .. IMAGE_END)
constexpr const char* kNoEncoder = "this engine has no image encoder";

struct Grid {
  uint16_t n_h, n_w;  // patches
};

enum class Verdict { kOk, kMalformed, kRefused };

struct Parsed {
  std::vector<fe::ImageSpan> spans;
  std::vector<Grid> grids;
  std::vector<size_t> off;  // each image's first byte in the payload; its bytes are 588 * n_h * n_w
};

struct Result {
  Verdict v = Verdict::kMalformed;
  std::string why;  // kRefused
  Parsed px;        // kOk (and kRefused: what was read)
};

inline size_t pad8(size_t n) { return (n + 7) & ~size_t(7); }

// The span's rows for a grid: the aligner merges 3 x 3 patches; IMAGE_START and IMAGE_END and one IMAGE_NEW_LINE per grid row.
inline uint64_t rows_of(const Grid& g) {
  const uint64_t lh = (g.n_h + 2) / 3, lw = (g.n_w + 2) / 3;
  return lh * (lw + 1) + 2;
}

// The payload `p` (n bytes) of a SUBMIT with `flags`, whose image part starts at `base`. kMalformed: both flags, n_images 0 or over
// fe::kMaxImages, or a byte count that is not exactly the layout's. kRefused: a grid of 0, a grid whose rows are not its span's,
// rows over kMaxRows, patches over vit::kMaxPatches.
inline Result parse(const char* p, size_t n, size_t base, uint32_t flags) {
  Result r;
#ifdef PICO_OLD_RULES
  (void)p, (void)n, (void)base, (void)flags;
  return r;
#else
  if ((flags & fe::kFlagImages) || !(flags & fe::kFlagPixels)) return r;
  if (n < base + 4) return r;
  uint32_t ni = 0;
  std::memcpy(&ni, p + base, 4);
  if (ni < 1 || ni > fe::kMaxImages) return r;
  const size_t spans_at = base + 4, grids_at = spans_at + size_t(ni) * sizeof(fe::ImageSpan), data_at = grids_at + pad8(size_t(ni) * 4);
  if (n < data_at) return r;
  r.px.spans.resize(ni);
  std::memcpy(r.px.spans.data(), p + spans_at, size_t(ni) * sizeof(fe::ImageSpan));
  r.px.grids.resize(ni);
  std::memcpy(r.px.grids.data(), p + grids_at, size_t(ni) * 4);
  size_t at = data_at;  // u16 x u16 x 588 stays far below 2^63: no overflow
  for (const Grid& g : r.px.grids) {
    r.px.off.push_back(at);
    at += pad8(kPatchBytes * g.n_h * g.n_w);
  }
  if (at != n) return r;
  r.v = Verdict::kRefused;
  for (uint32_t i = 0; i < ni; ++i) {
    const Grid& g = r.px.grids[i];
    const std::string at_i = "image " + std::to_string(i) + ": ";
    if (g.n_h < 1 || g.n_w < 1) {
      r.why = at_i + "a grid of " + std::to_string(g.n_h) + " x " + std::to_string(g.n_w) + " patches";
      return r;
    }
    if (uint64_t(g.n_h) * g.n_w > uint64_t(vit::kMaxPatches)) {
      r.why = at_i + std::to_string(uint64_t(g.n_h) * g.n_w) + " patches, over the " + std::to_string(vit::kMaxPatches) + " the encoder takes";
      return r;
    }
    const uint64_t rows = rows_of(g);
    if (rows != r.px.spans[i].rows) {
      r.why = at_i + "a grid of " + std::to_string(g.n_h) + " x " + std::to_string(g.n_w) + " patches makes " + std::to_string(rows) +
              " rows, its span has " + std::to_string(r.px.spans[i].rows);
      return r;
    }
    if (rows > kMaxRows) {
      r.why = at_i + std::to_string(rows) + " rows, over the " + std::to_string(kMaxRows) + " an image takes";
      return r;
    }
  }
  r.v = Verdict::kOk;
  return r;
#endif
}

}  // namespace pixels
