// The tower: weights on the device, the unsplit run and the four-rank run (one rank after another on one GPU,
// each with its own buffers; a gather is the copy of each rank's block into the others' buffers).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#include "common/check.h"
#include "vision/vit.h"
#include "vision/vit_layout.h"

namespace vit {
namespace {

bf16* put(DevWeights& d, const std::vector<u16>& h) {
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, h.size() * 2));
  CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  d.owned.push_back(p);
  return static_cast<bf16*>(p);
}

struct Unsplit {
  size_t x, xn, qkv, o, gu, h, y, cos, sin, inv, patch, unf, g1, g2, dig, total;
  Unsplit(int n, int m) {
    Arena a;
    x = a.take((size_t)n * 1024), xn = a.take((size_t)n * 1024), qkv = a.take((size_t)n * 3072);
    o = a.take((size_t)n * 1024), gu = a.take((size_t)n * 5632), h = a.take((size_t)n * 2816);
    y = a.take((size_t)n * 1024), cos = a.take((size_t)n * 32, 4), sin = a.take((size_t)n * 32, 4);
    inv = a.take(16, 4), patch = a.take((size_t)n * kPatchKPad), unf = a.take((size_t)m * kMergeIn);
    g1 = a.take((size_t)m * kLm), g2 = a.take((size_t)m * kLm), dig = a.take((size_t)(n + 15) / 16, 8);
    total = a.off;
  }
};

}  // namespace

DevWeights upload(const Weights& w) {
  DevWeights d;
  for (const BlockW& b : w.blocks)
    d.blocks.push_back({put(d, b.wqkv), put(d, b.bqkv), put(d, b.wo), put(d, b.bo), put(d, b.w1), put(d, b.w2),
                        put(d, b.norm1), put(d, b.norm2)});
  d.patch_w = put(d, pad_patch_weight(w.patch_w)), d.patch_b = put(d, w.patch_b), d.norm = put(d, w.norm);
  d.a1w = put(d, w.a1w), d.a1b = put(d, w.a1b), d.a2w = put(d, w.a2w), d.a2b = put(d, w.a2b);
  d.start = put(d, w.start), d.newline = put(d, w.newline), d.end = put(d, w.end);
  d.table = put(d, pixel_table());
  return d;
}

void release(DevWeights& d) {
  for (void* p : d.owned) CUDA_CHECK(cudaFree(p));
  d = DevWeights{};
}

const char* name_of(Wrong w) {
  switch (w) {
    case Wrong::None: return "none";
    case Wrong::SwapShard: return "swap-shard";
    case Wrong::SwapHeads: return "swap-heads";
    case Wrong::RopeAxes: return "rope-axes";
    case Wrong::PadGrey: return "pad-grey";
    case Wrong::NormSwap: return "norm-swap";
    case Wrong::UnfoldWindow: return "unfold-window";
    case Wrong::UnfoldOrder: return "unfold-order";
    case Wrong::QkvOrder: return "qkv-order";
    case Wrong::GateUp: return "gate-up";
    case Wrong::Delimiters: return "delimiters";
  }
  return "?";
}

const std::vector<Wrong>& all_wrong() {
  static const std::vector<Wrong> v = {Wrong::SwapShard, Wrong::SwapHeads, Wrong::RopeAxes,    Wrong::PadGrey,
                                       Wrong::NormSwap,  Wrong::UnfoldWindow, Wrong::UnfoldOrder, Wrong::QkvOrder,
                                       Wrong::GateUp,    Wrong::Delimiters};
  return v;
}

Weights wrong_slice(const Weights& whole, int rank, Wrong w) {
  Weights s = slice_rank(whole, rank);
  const size_t nb = s.blocks.size();
  auto swap_rows = [](std::vector<u16>& m, int cols, int a, int b, int n) {
    for (int r = 0; r < n; r++)
      for (int c = 0; c < cols; c++) std::swap(m[(size_t)(a + r) * cols + c], m[(size_t)(b + r) * cols + c]);
  };
  switch (w) {
    case Wrong::SwapShard:
      if (rank == 1 || rank == 2) s.blocks[17 % nb] = slice_rank(whole, 3 - rank).blocks[17 % nb];
      break;
    case Wrong::SwapHeads:
      if (rank == 1) {
        BlockW& b = s.blocks[5 % nb];
        for (int seg = 0; seg < 3; seg++) {
          swap_rows(b.wqkv, kDim, seg * 256, seg * 256 + 64, 64);
          swap_rows(b.bqkv, 1, seg * 256, seg * 256 + 64, 64);
        }
      }
      break;
    case Wrong::NormSwap: std::swap(s.blocks[17 % nb].norm1, s.blocks[17 % nb].norm2); break;
    case Wrong::QkvOrder: {
      BlockW& b = s.blocks[3 % nb];
      swap_rows(b.wqkv, kDim, 0, 256, 256);
      swap_rows(b.bqkv, 1, 0, 256, 256);
      break;
    }
    case Wrong::GateUp: swap_rows(s.blocks[9 % nb].w1, kDim, 0, 704, 704); break;
    case Wrong::Delimiters: std::swap(s.newline, s.end); break;
    default: break;
  }
  return s;
}

size_t unsplit_work_bytes(int patches, int merged) { return Unsplit(patches, merged).total; }
size_t split_work_bytes(int patches, int merged) { return Split(patches, merged).total; }

void run_unsplit(const DevWeights& w, const Image& img, void* work, RunOut out, cudaStream_t st, Wrong wrong) {
  const Grid g = img.grid;
  const int n = g.patches(), m = g.merged();
  const Unsplit L(n, m);
  bf16 *X = ptr<bf16>(work, L.x), *XN = ptr<bf16>(work, L.xn), *QKV = ptr<bf16>(work, L.qkv);
  bf16 *O = ptr<bf16>(work, L.o), *GU = ptr<bf16>(work, L.gu), *H = ptr<bf16>(work, L.h), *Y = ptr<bf16>(work, L.y);
  bf16 *PATCH = ptr<bf16>(work, L.patch), *UNF = ptr<bf16>(work, L.unf);
  bf16 *G1 = ptr<bf16>(work, L.g1), *G2 = ptr<bf16>(work, L.g2);
  float *COS = ptr<float>(work, L.cos), *SIN = ptr<float>(work, L.sin);
  uint64_t* DIG = ptr<uint64_t>(work, L.dig);
  const int variant = wrong == Wrong::UnfoldWindow ? 1 : wrong == Wrong::UnfoldOrder ? 2 : wrong == Wrong::PadGrey ? 3 : 0;
  auto tap = [&](const char* name, const bf16* p, int rows, int cols) {
    if (!out.tap) return;
    CUDA_CHECK(cudaStreamSynchronize(st));
    out.tap(name, p, rows, cols);
  };

  patchify(img.pixels, g.nh, g.nw, w.table, PATCH, st);
  gemm(rows_view(PATCH, kPatchKPad), w.patch_w, w.patch_b, rows_out(X, kDim), n, kDim, kPatchKPad, st);
  tap("patch_embed", X, n, kDim);
  rope_table(g.nh, g.nw, inv_freq_host(), COS, SIN, wrong == Wrong::RopeAxes, st);
  int bi = 0;
  for (const DevBlock& b : w.blocks) {
    char nm[32];
    rmsnorm(rows_view(X, kDim), b.norm1, XN, n, st);
    gemm(rows_view(XN, kDim), b.wqkv, b.bqkv, rows_out(QKV, 3 * kDim), n, 3 * kDim, kDim, st);
    rope_apply(QKV, QKV + kDim, 3 * kDim, n, kHeads, COS, SIN, st);
    attention(QKV, 3 * kDim, QKV + kDim, 3 * kDim, QKV + 2 * kDim, 3 * kDim, O, kDim, n, kHeads, st);
    gemm(rows_view(O, kDim), b.wo, b.bo, rows_out(Y, kDim), n, kDim, kDim, st);
    add(X, kDim, Y, kDim, X, kDim, n, kDim, st);
    if (out.tap) std::snprintf(nm, sizeof nm, "block%02d.attn_add", bi), tap(nm, X, n, kDim);
    rmsnorm(rows_view(X, kDim), b.norm2, XN, n, st);
    gemm(rows_view(XN, kDim), b.w1, nullptr, rows_out(GU, 2 * kInter), n, 2 * kInter, kDim, st);
    swiglu(GU, 2 * kInter, kInter, H, kInter, n, st);
    gemm(rows_view(H, kInter), b.w2, nullptr, rows_out(Y, kDim), n, kDim, kInter, st);
    add(X, kDim, Y, kDim, X, kDim, n, kDim, st);
    if (out.tap) std::snprintf(nm, sizeof nm, "block%02d.mlp_add", bi), tap(nm, X, n, kDim);
    if (out.trace) out.trace->push_back(digest(rows_view(X, kDim), n, kDim, DIG, st));
    bi++;
  }
  rmsnorm(rows_view(X, kDim), w.norm, XN, n, st);
  tap("norm", XN, n, kDim);
  unfold3(XN, g.nh, g.nw, UNF, variant, st);
  tap("unfold", UNF, m, kMergeIn);
  gemm(rows_view(UNF, kMergeIn), w.a1w, w.a1b, rows_out(G1, kLm), m, kLm, kMergeIn, st);
  gelu(G1, kLm, G1, kLm, m, kLm, st);
  tap("g", G1, m, kLm);
  gemm(rows_view(G1, kLm), w.a2w, w.a2b, rows_out(G2, kLm), m, kLm, kLm, st);
  tap("aligner", G2, m, kLm);
  span_rows(rows_view(G2, kLm), w.start, w.newline, w.end, g.lh, g.lw, 0, g.rows(), out.span, st);
  tap("span", out.span, g.rows(), kLm);
  CUDA_CHECK(cudaStreamSynchronize(st));
}

void run_split(const DevWeights* w, const Image& img, void* const works[kRanks], RunOut out, Wrong wrong,
               const SplitHooks& hooks, cudaStream_t st) {
  const Grid g = img.grid;
  const int n = g.patches(), m = g.merged();
  const Split L(n, m);
  constexpr int Q = kDim / kRanks, MQ = kInter / kRanks, AQ = kLm / kRanks;  // 256, 704, 1280
  struct R {
    bf16 *X, *XN, *QKV, *BIG, *Y, *PATCH, *UNF, *G1, *G2;
    float *COS, *SIN, *INV;
    uint64_t* DIG;
  } r[kRanks];
  for (int i = 0; i < kRanks; i++) {
    void* b = works[i];
    r[i] = {ptr<bf16>(b, L.x),   ptr<bf16>(b, L.xn),  ptr<bf16>(b, L.qkv), ptr<bf16>(b, L.big), ptr<bf16>(b, L.y),
            ptr<bf16>(b, L.patch), ptr<bf16>(b, L.unf), ptr<bf16>(b, L.g1),  ptr<bf16>(b, L.g2),
            ptr<float>(b, L.cos), ptr<float>(b, L.sin), ptr<float>(b, L.inv), ptr<uint64_t>(b, L.dig)};
  }
  auto gather = [&](bf16* R::*field, size_t block_elems) {
    bf16* bufs[kRanks];
    for (int i = 0; i < kRanks; i++) bufs[i] = r[i].*field;
    if (hooks.gather) {
      hooks.gather(bufs, block_elems, st);
      return;
    }
    for (int src = 0; src < kRanks; src++)
      for (int dst = 0; dst < kRanks; dst++)
        if (dst != src)
          CUDA_CHECK(cudaMemcpyAsync(bufs[dst] + src * block_elems, bufs[src] + src * block_elems, block_elems * 2,
                                     cudaMemcpyDeviceToDevice, st));
  };
  const int variant = wrong == Wrong::UnfoldWindow ? 1 : wrong == Wrong::UnfoldOrder ? 2 : wrong == Wrong::PadGrey ? 3 : 0;

  for (int i = 0; i < kRanks; i++) {
    patchify(img.pixels, g.nh, g.nw, w[i].table, r[i].PATCH, st);
    gemm(rows_view(r[i].PATCH, kPatchKPad), w[i].patch_w, w[i].patch_b, blocks_out(r[i].X, n, Q), n, kDim, kPatchKPad,
         st);
    rope_table(g.nh, g.nw, inv_freq_host(), r[i].COS, r[i].SIN, wrong == Wrong::RopeAxes, st);
  }
  const size_t xb = (size_t)n * Q;  // elements of one x block
  const int nblocks = (int)w[0].blocks.size();
  for (int bi = 0; bi < nblocks; bi++) {
    for (int i = 0; i < kRanks; i++) {
      const DevBlock& b = w[i].blocks[bi];
      rmsnorm(blocks_view(r[i].X, n, Q), b.norm1, r[i].XN, n, st);
      gemm(rows_view(r[i].XN, kDim), b.wqkv, b.bqkv, rows_out(r[i].QKV, 3 * Q), n, 3 * Q, kDim, st);
      rope_apply(r[i].QKV, r[i].QKV + Q, 3 * Q, n, kHeads / kRanks, r[i].COS, r[i].SIN, st);
      attention(r[i].QKV, 3 * Q, r[i].QKV + Q, 3 * Q, r[i].QKV + 2 * Q, 3 * Q, r[i].BIG + i * xb, Q, n,
                kHeads / kRanks, st);
    }
    gather(&R::BIG, xb);  // o
    for (int i = 0; i < kRanks; i++) {
      const DevBlock& b = w[i].blocks[bi];
      gemm(blocks_view(r[i].BIG, n, Q), b.wo, b.bo, rows_out(r[i].Y, Q), n, Q, kDim, st);
      add(r[i].X + i * xb, Q, r[i].Y, Q, r[i].X + i * xb, Q, n, Q, st);
    }
    gather(&R::X, xb);
    for (int i = 0; i < kRanks; i++) {
      const DevBlock& b = w[i].blocks[bi];
      rmsnorm(blocks_view(r[i].X, n, Q), b.norm2, r[i].XN, n, st);
      gemm(rows_view(r[i].XN, kDim), b.w1, nullptr, rows_out(r[i].Y, 2 * MQ), n, 2 * MQ, kDim, st);
      swiglu(r[i].Y, 2 * MQ, MQ, r[i].BIG + i * (size_t)n * MQ, MQ, n, st);
    }
    gather(&R::BIG, (size_t)n * MQ);  // h
    for (int i = 0; i < kRanks; i++) {
      const DevBlock& b = w[i].blocks[bi];
      gemm(blocks_view(r[i].BIG, n, MQ), b.w2, nullptr, rows_out(r[i].Y, Q), n, Q, kInter, st);
      add(r[i].X + i * xb, Q, r[i].Y, Q, r[i].X + i * xb, Q, n, Q, st);
    }
    gather(&R::X, xb);
    if (out.trace) out.trace->push_back(digest(blocks_view(r[0].X, n, Q), n, kDim, r[0].DIG, st));
  }
  for (int i = 0; i < kRanks; i++) {
    rmsnorm(blocks_view(r[i].X, n, Q), w[i].norm, r[i].XN, n, st);
    unfold3(r[i].XN, g.nh, g.nw, r[i].UNF, variant, st);
    bf16* own = r[i].G1 + i * (size_t)m * AQ;
    gemm(rows_view(r[i].UNF, kMergeIn), w[i].a1w, w[i].a1b, rows_out(own, AQ), m, AQ, kMergeIn, st);
    gelu(own, AQ, own, AQ, m, AQ, st);
  }
  gather(&R::G1, (size_t)m * AQ);
  for (int i = 0; i < kRanks; i++)
    gemm(blocks_view(r[i].G1, m, AQ), w[i].a2w, w[i].a2b, rows_out(r[i].G2 + i * (size_t)m * AQ, AQ), m, AQ, kLm, st);
  gather(&R::G2, (size_t)m * AQ);
  for (int i = 0; i < kRanks; i++) {
    bf16* dst = i == 0 ? out.span : out.span_others[i - 1];
    if (!dst) continue;
    span_rows(blocks_view(r[i].G2, m, AQ), w[i].start, w[i].newline, w[i].end, g.lh, g.lw, 0, g.rows(), dst, st);
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
}

}  // namespace vit
