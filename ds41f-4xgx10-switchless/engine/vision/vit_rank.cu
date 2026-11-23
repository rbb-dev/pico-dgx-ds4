// A rank's own run of the split tower (vit_rank.h): the phases of vit_tower.cu's run_split for one rank, and the one-GPU
// driver that steps four of them with the copy gather.
#include "vision/vit_rank.h"

#include <string>

#include "common/check.h"
#include "vision/vit_layout.h"

namespace vit {
namespace {

constexpr int Q = kDim / kRanks, MQ = kInter / kRanks, AQ = kLm / kRanks;  // 256, 704, 1280

size_t align256(size_t b) { return (b + 255) & ~size_t(255); }

}  // namespace

size_t pixel_block_bytes(int patches) { return ((size_t)patches * kPatchK + 15) & ~size_t(15); }

size_t rank_work_bytes(int patches, int merged) {
  return Split(patches, merged).total + align256(kRanks * pixel_block_bytes(patches));
}

uint8_t* pixel_slot(void* work, int patches, int merged) { return ptr<uint8_t>(work, Split(patches, merged).total); }

void write_span_rows(const DevWeights& w, void* work, const Grid& g, int first, int count, bf16* dst, cudaStream_t st) {
  const Split L(g.patches(), g.merged());
  span_rows(blocks_view(ptr<bf16>(work, L.g2), g.merged(), AQ), w.start, w.newline, w.end, g.lh, g.lw, first, count, dst, st);
}

int RankRun::gather_count(int b0, int b1, int nblocks) {
  if (b1 < 0) b1 = nblocks;
  return (b0 == 0 ? 1 : 0) + 4 * (b1 - b0) + (b1 == nblocks ? 2 : 0);
}

RankRun::RankRun(const RankRunConfig& c) : c_(c) {
  if (!c.w || !c.work) die("RankRun: no weights or no work area");
  if (c.rank < 0 || c.rank >= kRanks) die("RankRun: rank %d", c.rank);
  n_ = c.grid.patches(), m_ = c.grid.merged();
  if (n_ < 1 || n_ > kMaxPatches) die("RankRun: %d patches (1 to %d)", n_, kMaxPatches);
  nblocks_ = (int)c.w->blocks.size();
  b1_ = c_.b1 < 0 ? nblocks_ : c_.b1;
  if (c_.b0 < 0 || c_.b0 >= b1_ || b1_ > nblocks_) die("RankRun: blocks [%d, %d) of %d", c_.b0, b1_, nblocks_);
  if (c_.b0 == 0 && c_.rank == 0 && !c_.pixels) die("RankRun: rank 0 needs the image's pixels to begin at block 0");
  bi_ = c_.b0;
  stage_ = c_.b0 == 0 ? kPix : kO;
  embed_pending_ = c_.b0 == 0;
}

void RankRun::flush_trace() {
  if (!trace_pending_) return;
  trace_pending_ = false;
  const Split L(n_, m_);
  CUDA_CHECK(cudaStreamSynchronize(c_.st));  // x is complete after the gather the digest follows
  c_.trace->push_back(digest(blocks_view(ptr<bf16>(c_.work, L.x), n_, Q), n_, kDim, ptr<uint64_t>(c_.work, L.dig), c_.st));
}

bool RankRun::step(GatherPoint* g) {
  flush_trace();
  const Split L(n_, m_);
  void* const work = c_.work;
  const int i = c_.rank, n = n_, m = m_;
  const DevWeights& w = *c_.w;
  const Grid& gr = c_.grid;
  cudaStream_t st = c_.st;
  bf16 *X = ptr<bf16>(work, L.x), *XN = ptr<bf16>(work, L.xn), *QKV = ptr<bf16>(work, L.qkv),
       *BIG = ptr<bf16>(work, L.big), *Y = ptr<bf16>(work, L.y), *PATCH = ptr<bf16>(work, L.patch),
       *UNF = ptr<bf16>(work, L.unf), *G1 = ptr<bf16>(work, L.g1), *G2 = ptr<bf16>(work, L.g2);
  float *COS = ptr<float>(work, L.cos), *SIN = ptr<float>(work, L.sin);
  uint8_t* const PIX = ptr<uint8_t>(work, L.total);
  const size_t xb = (size_t)n * Q;  // elements of one x block
  const int variant =
      c_.wrong == Wrong::UnfoldWindow ? 1 : c_.wrong == Wrong::UnfoldOrder ? 2 : c_.wrong == Wrong::PadGrey ? 3 : 0;
  auto issue = [&](void* buf, size_t elems, const char* what) {
    *g = {buf, elems * 2, what};
    issued_++;
    return true;
  };

  switch (stage_) {
    case kPix: {
      // every rank zeroes its own block; rank 0 writes the padded image into block 0; the gather brings block 0 to all
      const size_t pb = pixel_block_bytes(n);
      if (!(i == 0 && c_.pixels == PIX)) {  // else the caller wrote block 0 (RankRunConfig::pixels)
        CUDA_CHECK(cudaMemsetAsync(PIX + (size_t)i * pb, 0, pb, st));
        if (i == 0) CUDA_CHECK(cudaMemcpyAsync(PIX, c_.pixels, (size_t)n * kPatchK, cudaMemcpyDeviceToDevice, st));
      }
      stage_ = kO;
      *g = {PIX, pb, "pixels"};
      issued_++;
      return true;
    }
    case kO: {
      if (embed_pending_) {
        patchify(PIX, gr.nh, gr.nw, w.table, PATCH, st);
        gemm(rows_view(PATCH, kPatchKPad), w.patch_w, w.patch_b, blocks_out(X, n, Q), n, kDim, kPatchKPad, st);
        rope_table(gr.nh, gr.nw, inv_freq_host(), COS, SIN, c_.wrong == Wrong::RopeAxes, st);
        embed_pending_ = false;
      }
      const DevBlock& b = w.blocks[bi_];
      rmsnorm(blocks_view(X, n, Q), b.norm1, XN, n, st);
      gemm(rows_view(XN, kDim), b.wqkv, b.bqkv, rows_out(QKV, 3 * Q), n, 3 * Q, kDim, st);
      rope_apply(QKV, QKV + Q, 3 * Q, n, kHeads / kRanks, COS, SIN, st);
      attention(QKV, 3 * Q, QKV + Q, 3 * Q, QKV + 2 * Q, 3 * Q, BIG + i * xb, Q, n, kHeads / kRanks, st);
      stage_ = kX1;
      return issue(BIG, xb, "o");
    }
    case kX1: {
      const DevBlock& b = w.blocks[bi_];
      gemm(blocks_view(BIG, n, Q), b.wo, b.bo, rows_out(Y, Q), n, Q, kDim, st);
      add(X + i * xb, Q, Y, Q, X + i * xb, Q, n, Q, st);
      stage_ = kH;
      return issue(X, xb, "x1");
    }
    case kH: {
      const DevBlock& b = w.blocks[bi_];
      rmsnorm(blocks_view(X, n, Q), b.norm2, XN, n, st);
      gemm(rows_view(XN, kDim), b.w1, nullptr, rows_out(Y, 2 * MQ), n, 2 * MQ, kDim, st);
      swiglu(Y, 2 * MQ, MQ, BIG + i * (size_t)n * MQ, MQ, n, st);
      stage_ = kX2;
      return issue(BIG, (size_t)n * MQ, "h");
    }
    case kX2: {
      const DevBlock& b = w.blocks[bi_];
      gemm(blocks_view(BIG, n, MQ), b.w2, nullptr, rows_out(Y, Q), n, Q, kInter, st);
      add(X + i * xb, Q, Y, Q, X + i * xb, Q, n, Q, st);
      trace_pending_ = c_.trace != nullptr;
      bi_++;
      stage_ = bi_ < b1_ ? kO : b1_ == nblocks_ ? kG1 : kEnd;
      return issue(X, xb, "x2");
    }
    case kG1: {
      rmsnorm(blocks_view(X, n, Q), w.norm, XN, n, st);
      unfold3(XN, gr.nh, gr.nw, UNF, variant, st);
      bf16* own = G1 + i * (size_t)m * AQ;
      gemm(rows_view(UNF, kMergeIn), w.a1w, w.a1b, rows_out(own, AQ), m, AQ, kMergeIn, st);
      gelu(own, AQ, own, AQ, m, AQ, st);
      stage_ = kG2;
      return issue(G1, (size_t)m * AQ, "g");
    }
    case kG2: {
      gemm(blocks_view(G1, m, AQ), w.a2w, w.a2b, rows_out(G2 + i * (size_t)m * AQ, AQ), m, AQ, kLm, st);
      stage_ = kTail;
      return issue(G2, (size_t)m * AQ, "aligner");
    }
    case kTail: {
      if (c_.span)
        span_rows(blocks_view(G2, m, AQ), w.start, w.newline, w.end, gr.lh, gr.lw, 0, gr.rows(), c_.span, st);
      stage_ = kEnd;
      return false;
    }
    case kEnd: return false;
  }
  return false;
}

void RankRun::poison_all_but_state(void* work, const Grid& g, cudaStream_t st) {
  const Split L(g.patches(), g.merged());
  const size_t total = rank_work_bytes(g.patches(), g.merged());
  CUDA_CHECK(cudaMemsetAsync(ptr<char>(work, L.xn), 0xFF, L.cos - L.xn, st));  // xn, qkv, big, y
  CUDA_CHECK(cudaMemsetAsync(ptr<char>(work, L.inv), 0xFF, total - L.inv, st));  // inv .. pixel slot
}

void run_lockstep(const DevWeights* w, const Image& img, void* const works[kRanks], bf16* const spans[kRanks],
                  Wrong wrong, const LockstepOpts& opts, cudaStream_t st) {
  std::vector<RankRun> runs;
  runs.reserve(kRanks);
  for (int r = 0; r < kRanks; r++) {
    RankRunConfig c;
    c.w = &w[r], c.rank = r, c.grid = img.grid, c.work = works[r];
    c.pixels = r == 0 ? img.pixels : nullptr;
    c.span = spans ? spans[r] : nullptr;
    c.wrong = wrong, c.st = st, c.b0 = opts.b0, c.b1 = opts.b1;
    c.trace = r == 0 ? opts.trace : nullptr;
    runs.emplace_back(c);
  }
  for (int index = 0;; index++) {
    GatherPoint gp[kRanks];
    bool more[kRanks];
    for (int r = 0; r < kRanks; r++) more[r] = runs[r].step(&gp[r]);
    for (int r = 1; r < kRanks; r++) {
      if (more[r] != more[0]) die("lockstep: rank %d is %s at gather %d, rank 0 is not", r, more[r] ? "gathering" : "done", index);
      if (more[0] && (gp[r].block_bytes != gp[0].block_bytes || std::string(gp[r].what) != gp[0].what))
        die("lockstep: rank %d stops at %s (%zu B), rank 0 at %s (%zu B), gather %d", r, gp[r].what, gp[r].block_bytes,
            gp[0].what, gp[0].block_bytes, index);
    }
    if (!more[0]) break;
    if (index == opts.skip_gather) continue;
    const size_t bb = gp[0].block_bytes;
    for (int src = 0; src < kRanks; src++)
      for (int dst = 0; dst < kRanks; dst++)
        if (dst != src)
          CUDA_CHECK(cudaMemcpyAsync((char*)gp[dst].buf + src * bb, (char*)gp[src].buf + src * bb, bb,
                                     cudaMemcpyDeviceToDevice, st));
  }
  if (opts.gathers) *opts.gathers = runs[0].gathers_issued();
  CUDA_CHECK(cudaStreamSynchronize(st));
}

}  // namespace vit
