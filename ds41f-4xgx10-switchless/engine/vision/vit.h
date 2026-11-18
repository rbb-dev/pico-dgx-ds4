// The image encoder's kernels (bf16, fp32 accumulation) and the two ways to run the tower: unsplit on one GPU
// (whole matrices, all heads, contiguous rows) and split in four (each rank a quarter of every large matrix,
// cut by output rows). No sum crosses a rank and no kernel's arithmetic depends on its launch shape, so the two
// runs give the same bytes; vit_test checks that.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <vector>

#include "vision/vit_weights.h"

namespace vit {

using bf16 = __nv_bfloat16;

// A [rows][cols] matrix held as column blocks: element (r, c) is at p[(c / cb) * blk + r * ld + c % cb].
// Contiguous rows are cb = cols, blk = 0, ld = cols. A gather writes the blocks one after another,
// each [rows][cb], so a gathered tensor is blk = rows * cb, ld = cb.
template <class P>
struct ViewT {
  P p;
  size_t blk;
  int cb, ld;
};
using View = ViewT<const bf16*>;
using OutView = ViewT<bf16*>;

inline View rows_view(const bf16* p, int cols) { return {p, 0, cols, cols}; }
inline OutView rows_out(bf16* p, int cols) { return {p, 0, cols, cols}; }
inline View blocks_view(const bf16* p, int rows, int cb) { return {p, (size_t)rows * cb, cb, cb}; }
inline OutView blocks_out(bf16* p, int rows, int cb) { return {p, (size_t)rows * cb, cb, cb}; }

// out = bf16(a . w^T + bias): a [M][K], w [N][K] row-major, bias [N] bf16 or null; fp32 accumulation in k
// order, the bias added in fp32, one rounding. N and K are multiples of 64, a's blocks (cb) multiples of 64,
// every pointer 16-byte aligned. An output element never depends on M, N or its tile.
void gemm(const View& a, const bf16* w, const bf16* bias, const OutView& out, int M, int N, int K,
          cudaStream_t st);

// Dense bidirectional attention over n tokens, `heads` heads of 64: q, k, v are [n][heads * 64] with their own
// row strides, scale 1/8, fp32 online softmax over all n keys in order, probabilities rounded to bf16 before
// the value product, one rounding of the output [n][ldo]. A head's result does not depend on the other heads.
void attention(const bf16* q, int ldq, const bf16* k, int ldk, const bf16* v, int ldv, bf16* o, int ldo, int n,
               int heads, cudaStream_t st);

// cos / sin [n][32] fp32 of the 2-D rotary table of an nh x nw grid (inv_freq16: rope_inv_freq()'s 16 values in HOST memory).
// swap_axes is for tests (the row and column positions exchanged).
void rope_table(int nh, int nw, const float* inv_freq16, float* cos, float* sin, bool swap_axes, cudaStream_t st);

// Rotates `heads` heads of q and of k in place (each [n][ld], head h at columns 64 h..) with the table.
void rope_apply(bf16* q, bf16* k, int ld, int n, int heads, const float* cos, const float* sin, cudaStream_t st);

// out [n][1024] contiguous = bf16(w * (x * rsqrt(mean(x^2) + 1e-6))), fp32 inside; w is bf16 [1024].
void rmsnorm(const View& x, const bf16* w, bf16* out, int n, cudaStream_t st);

// out = bf16(a + b) over [rows][cols] with row strides.
void add(const bf16* a, int lda, const bf16* b, int ldb, bf16* out, int ldo, int rows, int cols, cudaStream_t st);

// out[r][c] = bf16(bf16(silu(gu[r][c])) * gu[r][width + c]); gu rows hold gate then up, ldg apart.
void swiglu(const bf16* gu, int ldg, int width, bf16* out, int ldo, int rows, cudaStream_t st);

// out = bf16(gelu(x)), exact (erf) form.
void gelu(const bf16* x, int ldx, bf16* out, int ldo, int rows, int cols, cudaStream_t st);

// u8 pixels [nh*14][nw*14][3] through the 256-entry table to patches [n][640]: (channel, row, column) order,
// columns 588..639 zero.
void patchify(const uint8_t* pixels, int nh, int nw, const bf16* table256, bf16* out, cudaStream_t st);

// x [nh*nw][1024] contiguous -> u [lh*lw][9216]: 3 x 3 windows of the grid padded with zeros, channel-major
// (column c * 9 + i * 3 + j), windows in reading order. variant is for tests: 0 correct; 1 window-major columns
// (i * 3 + j) * 1024 + c; 2 windows in column order; 3 pad with one instead of zero.
void unfold3(const bf16* x, int nh, int nw, bf16* u, int variant, cudaStream_t st);

// Rows [first, first + count) of the image's span, written to dst [count][5120]: start, then per aligner row
// its lw tokens and a newline, then end. g is the aligner's output [lh*lw][5120] as a view.
void span_rows(const View& g, const bf16* start, const bf16* newline, const bf16* end, int lh, int lw, int first,
               int count, bf16* dst, cudaStream_t st);

// 64-bit digest of a [rows][cols] view: the sum over elements of a mix of (row, column, value), so equal
// contents give equal digests whatever the layout. Blocks of 16 rows each write a partial; sum them on the host.
uint64_t digest(const View& x, int rows, int cols, uint64_t* scratch_dev, cudaStream_t st);
size_t digest_scratch_bytes(int rows);

// ---------------------------------------------------------------------------------------------------------
// Weights on the device and the two tower runs.

struct DevBlock {
  bf16 *wqkv, *bqkv, *wo, *bo, *w1, *w2, *norm1, *norm2;
};
struct DevWeights {
  std::vector<DevBlock> blocks;
  bf16 *patch_w, *patch_b, *norm, *a1w, *a1b, *a2w, *a2b, *start, *newline, *end;
  bf16* table;  // the 256-entry pixel table
  std::vector<void*> owned;
};
// Uploads whole tensors (patch embedding padded to 640 columns) or a rank's slices (from slice_rank).
DevWeights upload(const Weights& w);
void release(DevWeights& d);

// Deliberate errors for the test builds that must fail (see vit_test --wrong).
enum class Wrong {
  None,
  SwapShard,      // ranks 1 and 2 hold each other's slices in block 17
  SwapHeads,      // two heads' rows exchanged in wqkv (not in wo) on rank 1, block 5
  RopeAxes,       // the table's row and column positions exchanged
  PadGrey,        // the unfold pads with one instead of zero
  NormSwap,       // block 17's norm1 and norm2 weights exchanged
  UnfoldWindow,   // unfold with window-major columns
  UnfoldOrder,    // unfold with windows in column order
  QkvOrder,       // q, k, v segments in the order k, q, v on every rank, block 3
  GateUp,         // gate and up halves exchanged in block 9
  Delimiters,     // newline and end delimiters exchanged
};
const char* name_of(Wrong w);
const std::vector<Wrong>& all_wrong();
// The weights a wrong build uses for rank r (whole tensors are not touched; rank is 0..3 for the split).
Weights wrong_slice(const Weights& whole, int rank, Wrong w);

struct Image {
  Grid grid;
  const uint8_t* pixels;  // device, [nh*14][nw*14][3]
};

// What a run reports: the span rows [rows][5120] on the device (owned by the caller) and, when `trace` is set,
// a digest of x after every block (layout independent).
struct RunOut {
  bf16* span;                          // [grid.rows()][5120], filled by the run (the split run: rank 0's copy)
  bf16* span_others[kRanks - 1] = {};  // the split run: ranks 1..3's copies, when set
  std::vector<uint64_t>* trace = nullptr;
  // The unsplit run only: called with each point of the tower as the reference's dump names it ("patch_embed",
  // "blockNN.attn_add", "blockNN.mlp_add", "norm", "unfold", "g", "aligner", "span") and its [rows][cols] contiguous
  // device rows, after the stream has been synchronized: the callee may read them at once. Null: no cost.
  std::function<void(const char* point, const bf16* rows, int nrows, int ncols)> tap;
};

// Peak buffer sizes of a run, in bytes, for the grid.
size_t unsplit_work_bytes(int patches, int merged);
size_t split_work_bytes(int patches, int merged);  // one rank's

// One GPU, whole matrices. `work` holds at least unsplit_work_bytes() of device memory.
// wrong: the builds' errors that live in the run (RopeAxes, PadGrey, UnfoldWindow, UnfoldOrder); the others are made in the weights.
void run_unsplit(const DevWeights& w, const Image& img, void* work, RunOut out, cudaStream_t st, Wrong wrong = Wrong::None);

// Four ranks simulated one after another on one GPU, each with its own buffers and its quarter of the
// weights; `gather` is called where the ranks exchange the block r of a tensor (default: copies on the device).
// works[r] holds split_work_bytes() each. wrong: the test builds' errors that live in the run itself.
struct SplitHooks {
  std::function<void(bf16* const bufs[kRanks], size_t block_elems, cudaStream_t)> gather;  // null: device copies
};
void run_split(const DevWeights* w /*[4]*/, const Image& img, void* const works[kRanks], RunOut out, Wrong wrong,
               const SplitHooks& hooks, cudaStream_t st);

}  // namespace vit
