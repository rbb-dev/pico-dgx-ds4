// One rank's own run of the split tower, as a phase machine, and the one-GPU driver that steps four of them.
//
// run_split (vit.h) is a sequence of phases, each "all four ranks compute, then one gather": it cannot be cut into
// four calls a rank after another. A RankRun is the same sequence seen from one rank: step() runs this rank's compute
// up to the next gather and says what to gather; whoever drives it (the ring, or run_lockstep's copies) gathers and
// calls step() again. The arithmetic is the split run's, kernel for kernel, so the bytes are the same.
//
// The gathers of one image, in the order every rank must issue them:
//   pixels            (only when the range starts at block 0) four blocks of pixel_block_bytes(n)
//   per block         o, x after wo, h, x after w2: four blocks of n * 256, n * 256, n * 704, n * 256 elements
//   aligner           (only when the range ends at the last block) g and the aligner's output: four blocks of m * 1280
// 4 per block x 32 + 2 + the pixel gather = 131 for a whole image.
//
// A gather is in place: `buf` holds four equal blocks, this rank's own at buf + rank * block_bytes already written;
// after the gather every block is there. Block sizes are multiples of 16 bytes.
#pragma once
#include <cstdint>
#include <vector>

#include "vision/vit.h"

namespace vit {

// What to gather next.
struct GatherPoint {
  void* buf;           // four blocks of block_bytes each, device memory inside the rank's work area
  size_t block_bytes;  // one rank's block
  const char* what;    // "pixels", "o", "x1", "h", "x2", "g", "aligner": equal on the four ranks at each point
};

struct RankRunConfig {
  const DevWeights* w = nullptr;  // this rank's quarter of the weights (slice_rank / its weight file)
  int rank = 0;
  Grid grid{};
  void* work = nullptr;  // rank_work_bytes(grid.patches(), grid.merged()) of device memory
  // The image's padded pixels, u8 [nh*14][nw*14][3] on the device: rank 0 only (the other ranks pass null; their
  // copy of the image arrives with the pixel gather). Needed when the range starts at block 0. When it is
  // pixel_slot(work, ...) itself, the caller has already written block 0 of the pixel slot (the image, then zeros up to
  // pixel_block_bytes) and the run neither clears nor copies it.
  const uint8_t* pixels = nullptr;
  bf16* span = nullptr;  // [grid.rows()][5120] on the device: written when the range ends at the last block; may be null
  Wrong wrong = Wrong::None;  // the test errors that live in the run (as run_split's)
  cudaStream_t st = nullptr;
  // Blocks [b0, b1) of the tower; b1 < 0: through the last block. A range that starts at 0 begins with the pixel
  // gather and the patch embedding; one that ends at the last block ends with the aligner and the span.
  int b0 = 0, b1 = -1;
  // When set (rank 0 of a test): a digest of x after every block of the range, layout independent (blocks the stream:
  // test use only).
  std::vector<uint64_t>* trace = nullptr;
};

class RankRun {
 public:
  explicit RankRun(const RankRunConfig& c);

  // Runs this rank's compute up to the next gather and returns true with it in *g; the caller gathers, in place and
  // on the same stream, and calls step() again. Returns false (nothing to gather) once the range is complete: the
  // work is queued on the stream, the caller synchronizes. Every call only queues work.
  bool step(GatherPoint* g);
  bool done() const { return stage_ == kEnd; }
  int gathers_issued() const { return issued_; }

  // The gathers a range makes (131 for [0, 32)).
  static int gather_count(int b0, int b1, int nblocks);

  // What persists between two calls of one image (a RankRun for [0, k) and another for [k, nblocks)): the work area's
  // x (all four blocks, gathered) and the rotary table. Nothing else is read before it is written; every other buffer
  // is rewritten by the range that uses it, and the RankRun itself keeps nothing the next one needs.
  // Test use: sets every other buffer of the work area to 0xFF (bf16 NaN) so a later range that read one would show it.
  static void poison_all_but_state(void* work, const Grid& g, cudaStream_t st);

 private:
  enum Stage { kPix, kO, kX1, kH, kX2, kG1, kG2, kTail, kEnd };
  void flush_trace();

  RankRunConfig c_;
  int n_, m_, nblocks_, bi_, b1_, issued_ = 0;
  Stage stage_;
  bool embed_pending_, trace_pending_ = false;
};

// Bytes of one block of the pixel gather for n patches: 588 n rounded up to 16 (the padding is zero).
size_t pixel_block_bytes(int patches);
// One rank's work area for a RankRun: split_work_bytes plus the pixel slot (four pixel blocks).
size_t rank_work_bytes(int patches, int merged);
// The pixel slot inside a work area made for (patches, merged): four blocks of pixel_block_bytes(patches), block 0 first.
uint8_t* pixel_slot(void* work, int patches, int merged);
// After a range that ended at the last block: rows [first, first + count) of the image's span (the aligner's output in the
// work area, with the start, newline and end rows) written to dst [count][5120] on the device. The caller cuts the span into
// the runs it needs (rows of one store number), so the rows go to their place with no copy.
void write_span_rows(const DevWeights& w, void* work, const Grid& g, int first, int count, bf16* dst, cudaStream_t st);

// What the one-GPU driver does beyond a plain run.
struct LockstepOpts {
  int b0 = 0, b1 = -1;                       // the blocks to run (as RankRunConfig)
  int skip_gather = -1;                      // test only (a negative control): do not perform the gather of this index
  std::vector<uint64_t>* trace = nullptr;    // rank 0's digests of x after each block
  int* gathers = nullptr;                    // out: the gathers rank 0 issued
};

// Steps four RankRuns on one GPU, in lockstep: every rank computes to its next gather point, then the four blocks
// are copied into every rank's buffer (what run_split does), until all four are done. Dies when the four ranks do
// not stop at the same gather. works[r] holds rank_work_bytes(); spans[r] (may be null) gets rank r's span rows.
// Rank 0 takes the pixels from `img`, the others get them from the pixel gather.
void run_lockstep(const DevWeights* w /*[4]*/, const Image& img, void* const works[kRanks], bf16* const spans[kRanks],
                  Wrong wrong, const LockstepOpts& opts, cudaStream_t st);

}  // namespace vit
