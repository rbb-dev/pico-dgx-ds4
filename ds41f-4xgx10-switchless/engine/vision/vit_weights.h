// The image encoder's weights on the host (bf16 as raw 16-bit words): the whole tensors as the checkpoint
// holds them, a synthetic set for tests, and the quarter each rank of four holds (cut by output rows,
// the slices of docs/vision.md). Host only: no CUDA here.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vit {

constexpr int kDim = 1024;        // width of the tower
constexpr int kHeads = 16;        // attention heads
constexpr int kHeadDim = 64;
constexpr int kInter = 2816;      // MLP width (gate and up are each this wide)
constexpr int kLayers = 32;
constexpr int kPatchK = 588;      // 3 x 14 x 14 values of one patch
constexpr int kPatchKPad = 640;   // the patch embedding's K, padded with zeros to whole stages of 64
constexpr int kPatch = 14;
constexpr int kMerge = 3;         // the aligner takes 3 x 3 neighbouring tokens
constexpr int kMergeIn = kDim * kMerge * kMerge;  // 9216
constexpr int kLm = 5120;         // the language model's width
constexpr int kRopeHalf = 32;     // rotated pairs of one head
constexpr int kMaxPatches = 9216; // bounds every buffer (an image of at most 1,024 rows)
constexpr int kRanks = 4;

using u16 = uint16_t;  // one bf16 value

struct BlockW {
  std::vector<u16> wqkv, bqkv, wo, bo, w1, w2, norm1, norm2;
};

// Whole tensors, or one rank's slices (the same struct: the shapes differ).
struct Weights {
  std::vector<BlockW> blocks;
  std::vector<u16> patch_w, patch_b;  // [1024][588] (whole), [1024]
  std::vector<u16> norm;              // [1024]
  std::vector<u16> a1w, a1b, a2w, a2b;
  std::vector<u16> start, newline, end;  // image_start, image_newline, image_end
};

// float <-> bf16 (round to nearest even; the inputs are finite)
u16 to_bf16(float f);
float from_bf16(u16 v);

// The checkpoint's tensors (vision.*, aligner.*, image_*) read from `dir`'s shards; dies when one is missing or
// not BF16 or has another shape. Reads only the first `blocks` blocks of the tower when blocks < 32.
Weights load_checkpoint(const std::string& dir, int blocks = kLayers);

// Reproducible random weights with the checkpoint's shapes: uniform in +-sqrt(3 / fan_in) so the
// activations keep a useful scale, norm weights near 1, small biases.
Weights make_synthetic(uint64_t seed, int blocks = kLayers);

// Rank r's quarter: wqkv rows of heads 4r..4r+3 of each of q, k, v ([768][1024]); wo rows 256r.. ; w1 rows 704r..
// of the gate and of the up half ([1408][1024]); w2 rows 256r.. ; the aligner's rows 1280r.. ; the rest whole.
Weights slice_rank(const Weights& whole, int rank);

// Rows [r0, r0 + n) of a row-major matrix with `cols` columns.
std::vector<u16> rows_of(const std::vector<u16>& m, int cols, int r0, int n);

// Zero-pad the patch embedding's rows from 588 to 640 columns.
std::vector<u16> pad_patch_weight(const std::vector<u16>& w588);

// The 256 bf16 values of a pixel byte: ((v / 255) - 0.5) / 0.5 in fp32, rounded once.
std::vector<u16> pixel_table();

// inv_freq of the 2-D rotary table: 1 / theta^(2i / 32), i < 16, in fp32.
std::vector<float> rope_inv_freq(float theta = 10000.0f);

// The grid of the aligner for an image of n_h x n_w patches.
struct Grid {
  int nh, nw;       // patches
  int lh, lw;       // aligner tokens (ceil(n / 3))
  int patches() const { return nh * nw; }
  int merged() const { return lh * lw; }
  int rows() const { return lh * (lw + 1) + 2; }  // the span: start, rows with a newline each, end
};
Grid make_grid(int nh, int nw);

}  // namespace vit
