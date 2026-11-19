// Prefill in chunks of many rows (model.h prefill): every layer, every row, with the prefill GEMMs
// (engine/kernels prefill.cu) and the attention kernels' multi-row forms. The arithmetic follows the
// decode step (model.cu): the same weights, sums in rank order, bf16 roundings at the same points, so
// a position's state does not depend on which path computed it beyond summation order.
//
// Sequence parallel around the residual stream (prefill.h): rank r owns rows [r q, r q + n) of the
// chunk. Its stream, the mixes and each sublayer boundary (dsk::hc_boundary_p: hc_post, the next
// sublayer's mixes, hc_pre, its norm and act_quant) cover those rows only, and a sublayer's partial
// sums arrive by a reduce-scatter by rows instead of an all-reduce. Per layer:
// - attention: wq_a, wkv (and the compressor's wkv / wgate) for this rank's rows with every rank's
//   weights (PLayer), one all-gather of the results; then on every row, as the decode step splits it:
//   the window through a linear buffer (the 127 positions before the chunk, from the ring, then the
//   chunk; the last 256 back into the ring), q_norm, wq_b (this rank's heads), the index layers' top-k
//   (the indexer for this rank's rows, all-gathered), sparse attention, wo_a; an all-to-all of wo_a's
//   outputs, so that wo_b runs on this rank's rows with every rank's slice of it (PLayer), the four
//   partial sums added in rank order as a reduce-scatter adds them; the boundary into the FFN;
// - FFN: the router for this rank's rows; one all-gather of the FFN input's act_quant and the routes;
//   the MoE over every row (this rank's 576 columns of every expert); a reduce-scatter in fp32
//   (model.py's), with the experts' combine inside it unless the MoE fuses it into its down kernel;
//   the boundary into the next layer;
// - Engram (layers 1 and 14): this rank's rows with every rank's wkv (PLayer).
//
// CED (docs/ced.md "What the engine has to do"): an encoder chunk runs layers 0-19, then layer 20's
// compressor for its rows (the decoder's global KV and index keys, the exact path's), and keeps its
// last 128 positions' layer-20 input (ced_keep). The chunk that samples then replays layers 20-39 over
// [P - 128, P) as a chunk of 128 rows whose window starts at P - 128 (ced_replay), and the head reads it.
#include "model/prefill.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/embed_shard.h"
#include "kernels/prefill_attn.h"
#include "dspark/dspark.h"
#include "model/nvtx.h"
#include "model/spec.h"
#include "model/state.h"

namespace model {

using namespace detail;

// The standing set's allocator: a failed allocation stops the engine, the buffers are zeroed (dalloc, halloc).
struct StandingAlloc {
  bool ok = true;
  template <class T>
  T* d(size_t n) { return dalloc<T>(n); }
  template <class T>
  T* h(size_t n) { return halloc<T>(n); }
};

void Model::Impl::prefill_alloc(comm::Ring* ring) {
  if (pf) return;
  dsk::prefill_tune_env(rank);  // DSGB10_PF_TUNE
  size_t free0 = 0, free1 = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  pf = std::make_unique<Prefill>();
  Prefill& p = *pf;
  const int M = std::max(opt.prefill_rows, kReplay);  // CED replays 128 rows
  StandingAlloc sa;
  rows_alloc(p.rows, M, sa);
  const int Q = p.rows.Q;
  if (opt.prefill_rows_max > p.rows.cap && !tenants.find(lend::kTenantPrefill))
    tenants.add(lend::kTenantPrefill, lend::kKindDevice, lend::kOnDemand);  // the lent set (rows_lend.cu): never evicts, the first to give way
  int mf = M;  // the largest chunk whose MoE fuses its combine: only those use ymoe and ysum
  while (mf > 1 && !dsk::moe_prefill_fuses(mf)) --mf;
  p.ysum = dalloc<float>((size_t)q_rows(mf) * kDim);
  p.ymoe = dalloc<float>((size_t)4 * q_rows(mf) * kDim);
  p.gath = dalloc<uint8_t>((size_t)kRanks * kDim * 2);
  p.key_q = q_rows(std::max(M, opt.prefill_rows_max));  // the lent set's chunks use these keys too
  p.keys = dalloc<uint16_t>(std::max((size_t)kIdxBatch * scan_[0].stride, (size_t)p.key_q * kPoolBlocks * 8));
  p.bkeys = dalloc<uint16_t>((size_t)kIdxBatch * scan_[0].stride / 8);
  scan_[0].pkeys = p.keys, scan_[0].pbkeys = p.bkeys;
  {
    int lo = 0, hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&p.cs, cudaStreamNonBlocking, hi));
    for (auto& e : p.ev_grp) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&p.ev_cs, cudaEventDisableTiming));
    const char* ps = std::getenv("DSGB10_PF_STALL");
    p.stall = ps && std::atoi(ps) != 0;
    if (p.stall) {
      for (auto& pr : p.ev_jn)
        for (auto& e : pr) CUDA_CHECK(cudaEventCreate(&e));
      for (auto& e : p.ev_fl) CUDA_CHECK(cudaEventCreate(&e));
    }
  }
  p.ced_rows = dalloc<uint8_t>((size_t)kReplay * kCedRow);
  p.ced_gath = dalloc<uint8_t>((size_t)kRanks * kReplay * kCedRow);
  std::fill(p.ced_pos, p.ced_pos + kReplay, -1);
  // bf16 weights the prefill GEMMs read in the GEMV's tiled layout (tile_fp8 with 2 bytes per element),
  // and every rank's slices of the weights this rank's rows need whole (PLayer)
  uint8_t* tmp = dalloc<uint8_t>((size_t)kHead * kDim * 2);  // a gathered compressor matrix before tiling
  const size_t ewb = (size_t)kEngramN * kEngramK, esb = (size_t)kEngramN / 32 * (kEngramK / 32);
  for (int L = 0; L < kLayers; ++L) {
    LayerW& w = lw[L];
    PLayer& g = p.pl[L];
    if (w.ix_wproj) {
      w.p_wproj = dalloc<uint8_t>((size_t)kIdxHeads * kDim * 2);
      dsk::tile_fp8(reinterpret_cast<const uint8_t*>(w.ix_wproj), w.p_wproj, kIdxHeads, kDim * 2, st);
    }
    for (int h = 0; h < 2; ++h)
      w.p_hcfn[h] = reinterpret_cast<uint32_t*>(dalloc<uint8_t>(dsk::hc_fn_parts_bytes()));
    // made before the weights load (reserve_gathered): this rank's slot already holds its slice, which decode reads
    const GatheredW pre = pgw.empty() ? GatheredW{} : pgw[L];
    auto take = [](uint8_t* have, size_t n) { return have ? have : dalloc<uint8_t>(n); };
    g.wqa = take(pre.wqa, (size_t)kQLora * kDim);
    g.sqa = take(pre.sqa, (size_t)kQLora / 32 * (kDim / 32));
    g.wkv = take(pre.wkv, (size_t)kHead * kDim);
    g.skv = take(pre.skv, (size_t)kHead / 32 * (kDim / 32));
    g.wob = take(pre.wob, (size_t)kRanks * kDim * kOPart);
    g.sob = take(pre.sob, (size_t)kRanks * (kDim / 32) * (kOPart / 32));
    if (kv_src(L)) {
      g.ckv = dalloc<uint8_t>((size_t)kHead * kDim * 2);
      if (w.c_wgate) g.cgate = dalloc<uint8_t>((size_t)kHead * kDim * 2);
    }
    if (engram_layer(L) && store) g.ewkv = take(pre.ewkv, kRanks * ewb), g.eskv = take(pre.eskv, kRanks * esb);
  }
  if (ring) ring->barrier();
  float *hc_whole = nullptr, *hc_gath = nullptr;
  for (int L = 0; L < kLayers; ++L) {
    const LayerW& w = lw[L];
    PLayer& g = p.pl[L];
    // L1 tiles of 8 rows and scale rows of 32 concatenate, so the gathered slices are the whole matrices
    // the parts come from the whole projection: a quarter file's whole is gathered into one transient buffer, reused
    // by every sublayer and freed below; a file cut by rows holds its whole already
    for (int h = 0; h < 2; ++h) {
      const float* fn = w.hc_fn[h];
      if (w.hc_ld) {
        if (!hc_whole) hc_whole = dalloc<float>((size_t)kMix * kHcDim), hc_gath = dalloc<float>((size_t)kMix * kHcDim);
        comm::allgather(ch, w.hc_fn[h], hc_gath, (size_t)kMix * kDim * 4, st);
        for (int r = 0; r < kRanks; ++r)
          CUDA_CHECK(cudaMemcpy2DAsync(hc_whole + (size_t)kDim * r, (size_t)kHcDim * 4, hc_gath + (size_t)kMix * kDim * r,
                                       (size_t)kDim * 4, (size_t)kDim * 4, kMix, cudaMemcpyDeviceToDevice, st));
        fn = hc_whole;
      }
      dsk::hc_fn_parts(fn, w.p_hcfn[h], st);
    }
    // (this rank's own slot is already in place, where decode reads it: the gather writes the same bytes back)
    const comm::AgSeg ga[6] = {{w.wqa, g.wqa, (size_t)kQaPart * kDim},
                               {w.sqa, g.sqa, (size_t)kQaPart / 32 * (kDim / 32)},
                               {w.wkv, g.wkv, (size_t)kKvPart * kDim},
                               {w.skv, g.skv, (size_t)kKvPart / 32 * (kDim / 32)},
                               {w.wob, g.wob, (size_t)kDim * kOPart},
                               {w.sob, g.sob, (size_t)(kDim / 32) * (kOPart / 32)}};
    comm::allgather_multi(ch, ga, 6, st);
    if (kv_src(L)) {  // this rank's 128 rows of each compressor matrix, gathered, then tiled
      comm::allgather(ch, w.c_wkv, tmp, (size_t)kCompPart * kDim * 2, st);
      dsk::tile_fp8(tmp, g.ckv, kHead, kDim * 2, st);
      if (w.c_wgate) {
        comm::allgather(ch, w.c_wgate, tmp, (size_t)kCompPart * kDim * 2, st);
        dsk::tile_fp8(tmp, g.cgate, kHead, kDim * 2, st);
      }
    }
    if (engram_layer(L) && store) {
      const comm::AgSeg ge[2] = {{w.e_wkv, g.ewkv, ewb}, {w.e_skv, g.eskv, esb}};
      comm::allgather_multi(ch, ge, 2, st);
    }
  }
  sync();
  CUDA_CHECK(cudaFree(tmp));
  if (hc_whole) {
    CUDA_CHECK(cudaFree(hc_whole));
    CUDA_CHECK(cudaFree(hc_gath));
  }
  CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  std::printf("rank %d: prefill buffers and gathered weights for %d-row chunks: %.2f GiB (%.2f GiB free)\n", rank,
              M, (double)(free0 - free1 + pgw_bytes) / 1073741824.0, free1 / 1073741824.0);
  pgw.clear(), pgw_bytes = 0;
}

#ifdef DSGB10_NVTX
// Profiling builds: DSGB10_ROUTE_DUMP=<dir> has rank 0 write each pass's routes (every row, after the
// FFN's all-gather) to <dir>/a<pos>.m<M>.L<LL>.i32, int32 [M][6] (bench_kernels moereal's layout). It
// synchronizes the stream once a layer: for counting the experts a turn touches, never for timing.
static const char* route_dump_dir() {
  static const char* d = std::getenv("DSGB10_ROUTE_DUMP");
  return d && *d ? d : nullptr;
}
static int g_dump_a = -1;  // the pass's first position (the replay's: P - 128)
static void dump_routes(const Prefill& p, int L, int M, cudaStream_t st) {
  std::vector<int> h((size_t)kRanks * p.rq * kTopK), rows;
  CUDA_CHECK(cudaMemcpyAsync(h.data(), p.rs->ridx, h.size() * 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  for (int r = 0; r < kRanks; ++r) {
    const int n = std::max(0, std::min(p.rq, M - r * p.rq));
    rows.insert(rows.end(), h.begin() + (size_t)r * p.rq * kTopK, h.begin() + ((size_t)r * p.rq + n) * kTopK);
  }
  char path[512];
  std::snprintf(path, sizeof path, "%s/a%d.m%d.L%02d.i32", route_dump_dir(), g_dump_a, M, L);
  FILE* f = std::fopen(path, "wb");
  if (!f) die("route dump: cannot write %s", path);
  std::fwrite(rows.data(), 4, rows.size(), f);
  std::fclose(f);
}
#endif


// ---- drafter2: feature dump for drafter self-distillation (DSGB10_FEAT_DUMP=<dir>; data collection only) ----
static const std::string& feat_dir() {
  static const std::string d = [] {
    const char* e = std::getenv("DSGB10_FEAT_DUMP");
    std::string v = e ? e : "";
    if (v.rfind("~/", 0) == 0 && std::getenv("HOME")) v = std::string(std::getenv("HOME")) + v.substr(1);
    return v;
  }();
  return d;
}
static int feat_rank() {
  static const int r = std::getenv("DSGB10_FEAT_RANK") ? std::atoi(std::getenv("DSGB10_FEAT_RANK")) : 3;
  return r;
}
constexpr int kFeatRow = dspark::kMainK + kDim;  // bf16 values per row: main_hidden, then the final collapse
static uint16_t *g_feat = nullptr, *g_feat_all = nullptr;
static int g_feat_rows = 0;
static long g_feat_ctr = 0;
static void feat_hidden(const Prefill& p, int L, cudaStream_t st) {
  if (g_feat_rows < p.rs->Q) {
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (g_feat) CUDA_CHECK(cudaFree(g_feat));
    if (g_feat_all) CUDA_CHECK(cudaFree(g_feat_all));
    g_feat = dalloc<uint16_t>((size_t)p.rs->Q * kFeatRow);
    g_feat_all = dalloc<uint16_t>((size_t)kRanks * p.rs->Q * kFeatRow);
    g_feat_rows = p.rs->Q;
  }
  for (int r = 0; r < p.n; r += dspark::kMaxHidden) {
    const int m = std::min(dspark::kMaxHidden, p.n - r);
    // main_hidden writes columns [5120 * slot, +5120) of rows of stride kMainK; here rows of stride kFeatRow
    dspark::main_hidden(U(p.rs->x) + (size_t)r * kHcDim, m, L - 37, g_feat_all + (size_t)r * dspark::kMainK, false, st);
    CUDA_CHECK(cudaMemcpy2DAsync(g_feat + (size_t)r * kFeatRow + (L - 37) * kDim, (size_t)kFeatRow * 2,
                                 g_feat_all + (size_t)r * dspark::kMainK + (L - 37) * kDim, (size_t)dspark::kMainK * 2,
                                 (size_t)kDim * 2, m, cudaMemcpyDeviceToDevice, st));
  }
}
static void feat_write(const Prefill& p, const comm::DevChannel& ch, int a, int M, int rank, const int32_t* toks,
                       cudaStream_t st) {
  if (p.n > 0)
    CUDA_CHECK(cudaMemcpy2DAsync(g_feat + dspark::kMainK, (size_t)kFeatRow * 2, p.rs->yraw, (size_t)kDim * 2,
                                 (size_t)kDim * 2, p.n, cudaMemcpyDeviceToDevice, st));
  comm::allgather(ch, g_feat, g_feat_all, (size_t)p.rs->Q * kFeatRow * 2, st);  // every rank, the same order
  const long c = g_feat_ctr++;
  if (rank != feat_rank()) return;
  CUDA_CHECK(cudaStreamSynchronize(st));
  char path[1024];
  std::snprintf(path, sizeof path, "%s/c%07ld.a%d.bin", feat_dir().c_str(), c, a);
  FILE* f = std::fopen(path, "wb");
  if (!f) die("feat dump: cannot write %s", path);
  const int32_t hdr[8] = {0x44465432, a, M, p.rq, 0, 0, 0, 0};
  std::fwrite(hdr, 4, 8, f);
  std::fwrite(toks, 4, M, f);
  std::vector<uint16_t> h((size_t)p.rq * kFeatRow);
  for (int r = 0; r < kRanks; ++r) {
    const int n = std::max(0, std::min(p.rq, M - r * p.rq));
    if (n == 0) continue;
    CUDA_CHECK(cudaMemcpy(h.data(), g_feat_all + (size_t)r * p.rs->Q * kFeatRow, (size_t)n * kFeatRow * 2,
                          cudaMemcpyDeviceToHost));
    std::fwrite(h.data(), 2, (size_t)n * kFeatRow, f);
  }
  std::fclose(f);
}

// ---- Engram ---------------------------------------------------------------------------------------

void Model::Impl::prefill_engram(int a, int M, int set) {
  Prefill& p = *pf;
  p.rs->e_pos0[set] = a, p.rs->e_n[set] = M;
  p.rs->e_unit[set] = cur_unit, p.rs->e_ep[set] = hist_ep[cur_unit];
  if (!store) return;
  // a lookup no chunk used (a cancelled request's) may still be writing the set's buffers
  for (int i = 2 * set; i < 2 * set + 2; ++i)
    while (__atomic_load_n(&p.rs->e_done[i], __ATOMIC_ACQUIRE) < p.rs->e_wait[i]) {
    }
  if (hist.end() < a + M) die("prefill: Engram lookups for [%d, %d) with %d tokens", a, a + M, hist.end());
  if (hist.base > 0 && a < hist.base + 3) die("prefill: Engram lookups from %d, history held from %d", a, hist.base);
  p.rs->ids[set].resize((size_t)M * 2 * kEngramCols);
  engram::hash_positions(*emap, hist.data(), nullptr, a, M, p.rs->ids[set].data(), hist.base);
  for (int e = 0; e < 2; ++e) {
    engram::CellMap cm;
    cm.build(store->split(), e, p.rs->ids[set].data(), M);
    cm.offsets(p.rs->e_off[set][e]);
    p.rs->e_seg[set][e] = cm.seg_bytes();
    engram::Request rq;
    rq.layer = e;
    rq.ids = p.rs->ids[set].data();
    rq.n_pos = M;
    rq.out = p.rs->e_send[set][e];
    rq.done = reinterpret_cast<std::atomic<uint64_t>*>(&p.rs->e_done[2 * set + e]);
    rq.seq = p.rs->e_wait[2 * set + e] = ++e_seq;
    store->submit(rq);
  }
}

// This rank's rows: the lookups (every rank's cells, all-gathered), then every rank's quarter of the
// keys and the value with every rank's weights (one GEMM of four segments), the partial dots in the
// decode step's per-rank pieces, and the combine summing them in rank order.
void Model::Impl::engram_apply_p(int L, int M, int set) {
  NVTX_RANGE("engram L%02d", L);
  Prefill& p = *pf;
  const int e = engram_index(L);
  const LayerW& w = lw[L];
  const PLayer& g = p.pl[L];
  ops::wait_seq(&p.rs->e_done[2 * set + e], &p.rs->e_wait[2 * set + e], st);
  comm::allgather(ch, p.rs->e_send[set][e], p.rs->e_gath, p.rs->e_seg[set][e], st);
  const int n = p.n;
  if (n == 0) return;
  ops::engram_pack(p.rs->e_gath, p.rs->e_off[set][e] + (size_t)p.r0 * kEngramCols, n, p.rs->e_xq, p.rs->e_xs, st);
  const size_t wb = (size_t)kEngramN * kEngramK, sb = (size_t)kEngramN / 32 * (kEngramK / 32);
  dsk::Fp8Seg seg[kRanks];
  for (int r = 0; r < kRanks; ++r) seg[r] = {g.ewkv + r * wb, g.eskv + r * sb, p.rs->e_kv + (size_t)r * p.rs->Q * kEngramN, kEngramN};
  dsk::gemm_fp8_p(p.rs->e_xq, p.rs->e_xs, n, seg, kRanks, kEngramK, st);
  for (int r = 0; r < kRanks; ++r)
    ops::engram_partial(p.rs->x, p.rs->e_kv + (size_t)r * p.rs->Q * kEngramN, w.e_w, r, n, p.rs->e_part + (size_t)r * n * kEngramPartBytes,
                        st);
  if (!img_span.empty())  // vision: image rows pass through (gate * 0), as model.py's token_mask
    img_runs(p.a_cur + p.r0, n, [&](size_t, int i0, int i1) {
      for (int r = 0; r < kRanks; ++r)
        CUDA_CHECK(cudaMemset2DAsync(p.rs->e_part + ((size_t)r * n + i0) * kEngramPartBytes + 32, kEngramPartBytes, 0,
                                     (size_t)kEngramSeg * 2, i1 - i0, st));
    });
  ops::engram_combine(p.rs->x, p.rs->e_part, n, p.rs->x, st);
  (void)M;
}

// ---- mHC -------------------------------------------------------------------------------------------

// A sublayer boundary on this rank's M rows (model.py's single-pass mHC; every pointer at the rank's
// first row): the stream out = hc_post(sub, res, post, comb) (no sub: res itself, not copied), then for
// the next sublayer (nL, nh) its input y = norm(hc_pre(out, pre)) with yq / ys (at the rank's rows of the
// all-row arrays), and its mixes from out (pre_out: the pre of the sublayer after it). nL < 0: nothing to
// prepare (the stream changes before the next sublayer: an Engram layer; or the head's collapse, which
// takes yraw = hc_pre(out, pre) without a norm).
void Model::Impl::boundary_p(int M, const bf16* sub, const bf16* res, const float* post, const float* comb,
                             const float* pre, bf16* out, int nL, int nh, float* pre_out, float* post_out,
                             float* comb_out) {
  Prefill& p = *pf;
  dsk::HcBoundary b;
  b.sub = U(sub), b.res = U(res), b.post = post, b.comb = comb, b.pre = pre, b.y = U(out);
  b.x_out = U(p.rs->yraw);
  if (nL >= 0) {
    const LayerW& w = lw[nL];
    b.fn_parts = w.p_hcfn[nh], b.scale = w.hc_scale[nh], b.base = w.hc_base[nh];
    b.norm_w = U(nh ? w.ffn_norm : w.attn_norm);
    b.x_out = U(p.rs->y), b.xq = p.rs->yq + (size_t)p.r0 * kDim, b.xs = p.rs->ys + (size_t)p.r0 * (kDim / 32);
    b.pre_out = pre_out, b.post_out = post_out, b.comb_out = comb_out;
  }
  dsk::hc_boundary_p(b, M, st);
}

// ---- row-split pieces ------------------------------------------------------------------------------

// An index layer's top-k (layer 20: also the candidate pool) for this rank's rows, all-gathered into
// the layer's slot. On entry qq/qs hold every row's queries' act_quant and y this rank's normed input.
void Model::Impl::index_p(int L, int a, int M) {
  NVTX_RANGE("index L%02d", L);
  Prefill& p = *pf;
  const LayerW& w = lw[L];
  const int Q = p.rq, r0 = p.r0, n = p.n;
  const int ratio = ratio_of(L), slot = idx_slot(L);
  if (n > 0) {
    const dsk::Fp8Seg qs{w.ix_wqb, w.ix_sqb, p.rs->iq, kIdxQ};
    dsk::gemm_fp8_p(p.rs->qq + (size_t)r0 * kQLora, p.rs->qs + (size_t)r0 * kQLora / 32, n, &qs, 1, kQLora, st);
    dsk::gemm_bf16_p(p.rs->y, n, w.p_wproj, kIdxHeads, kDim, p.rs->wpf, st);
    ops::round_bf16(p.rs->wpf, (size_t)n * kIdxHeads, st);  // a bf16 F.linear, as the decode step's
    const attn::IndexKeys ik = kvs[kv_slot(kv_source(L))].keys();
    if (reindex(L)) {  // layer 20's pool: a CTA per (2048 positions, row)
      attn::IndexScan sc;
      sc.blocks = p.rs->pool;
      sc.nblocks = kPoolBlocks;
      const attn::IndexSel top{kIdxTopk, p.rs->topk_mine, p.rs->topkey, kIdxTopk};
      attn::index_fused(U(p.rs->iq), p.rs->wpf, p.rs->pos + r0, yarn(), n, ratio, ik, sc, kPoolBlocks * 8, scan().pkeys,
                        kPoolBlocks * 8, nullptr, 0, top, attn::IndexSel{}, p.rs->counters, false, st);
    } else {  // every position so far: the prefill scan, a CTA per 256 positions and rows, in batches of rows
      const int max_scan =
          std::min(scan().stride, ((a + M) / ratio + attn::kScoreAlign) / attn::kScoreAlign * attn::kScoreAlign);
      const attn::IndexQ q{p.rs->iqf, p.rs->iqsc, p.rs->iqw};
      attn::index_q_prep(U(p.rs->iq), p.rs->wpf, p.rs->pos + r0, yarn(), q, nullptr, n, st);
      dsk::index_topk_p(q, p.rs->pos + r0, n, ratio, ik, max_scan, kIdxBatch, scan().pkeys, scan().stride, kIdxTopk, p.rs->topk_mine,
                        p.rs->topkey, L == 20 ? scan().pbkeys : nullptr, kPoolBlocks, p.rs->pool, p.rs->poolkey, st);
    }
  }
  comm::allgather(ch, p.rs->topk_mine, p.rs->topk[slot], (size_t)Q * kIdxTopk * 4, st);
}

// ---- the window ------------------------------------------------------------------------------------

void Model::Impl::ring_copy(int L, int p0, int p1, int base, bool to_lin) {
  Prefill& p = *pf;
  constexpr size_t kv = kHead, sc = kHead / 32;
  for (int q = p0; q < p1;) {
    const int slot = q & (attn::kRing - 1), n = std::min(p1 - q, attn::kRing - slot);
    uint8_t* rk = ring_kv[L] + slot * kv;
    uint8_t* rs = ring_sc[L] + slot * sc;
    uint8_t* lk = p.rs->lin_kv + (size_t)(q - base) * kv;
    uint8_t* ls = p.rs->lin_sc + (size_t)(q - base) * sc;
    CUDA_CHECK(cudaMemcpyAsync(to_lin ? lk : rk, to_lin ? rk : lk, n * kv, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(to_lin ? ls : rs, to_lin ? rs : ls, n * sc, cudaMemcpyDeviceToDevice, st));
    q += n;
  }
}

// ---- column groups -----------------------------------------------------------------------------------

// The comm stream continues after the main stream's work so far (column group c is complete): the
// collective issued next on it overlaps what the main stream does next. The comm stream runs the
// collectives in order and the main stream issues none until rs_join, so every rank still sees one
// sequence of collectives.
static void cs_after(Prefill& p, cudaStream_t st, int c) {
  CUDA_CHECK(cudaEventRecord(p.ev_grp[c], st));
  CUDA_CHECK(cudaStreamWaitEvent(p.cs, p.ev_grp[c], 0));
}

// The main stream waits for the column groups' reduce-scatters. `kind` is which join (0 the attention's, 1 the MoE's);
// with the join clock on, two events on the main stream bracket the wait: their distance is the time the stream stood
// at the join (0 when the comm stream had finished).
static void rs_join(Prefill& p, cudaStream_t st, int kind) {
  CUDA_CHECK(cudaEventRecord(p.ev_cs, p.cs));
  const bool clk = p.jn_on && p.jn_n < Prefill::kJoinMax;
  if (clk) CUDA_CHECK(cudaEventRecord(p.ev_jn[p.jn_n][0], st));
  CUDA_CHECK(cudaStreamWaitEvent(st, p.ev_cs, 0));
  if (clk) {
    CUDA_CHECK(cudaEventRecord(p.ev_jn[p.jn_n][1], st));
    p.jn_kind[p.jn_n++] = (uint8_t)kind;
  }
}

// ---- attention ---------------------------------------------------------------------------------------

// On entry y / yq / ys hold this rank's rows' normed input and a_pre / a_post / a_comb their mixes.
void Model::Impl::attention_p(int L, int a, int M, const std::vector<std::pair<int, const std::vector<uint32_t>*>>& snaps, int win_lo) {
  NVTX_RANGE("attn L%02d", L);
  Prefill& p = *pf;
  const LayerW& w = lw[L];
  const PLayer& g = p.pl[L];
  const int Q = p.rq, r0 = p.r0, n = p.n;
  // this rank's rows of wq_a, wkv and the compressor (all of each), then every row by one all-gather
  const int ncomp = kv_src(L) && win_lo < 0 ? (ratio_of(L) > 1 ? 2 : 1) : 0;
  if (n > 0) {
    const dsk::Fp8Seg seg[2] = {{g.wqa, g.sqa, p.rs->qa + (size_t)r0 * kQLora, kQLora},
                                {g.wkv, g.skv, p.rs->kvx + (size_t)r0 * kHead, kHead}};
    dsk::gemm_fp8_p(p.rs->yq + (size_t)r0 * kDim, p.rs->ys + (size_t)r0 * (kDim / 32), n, seg, 2, kDim, st);
    if (ncomp) {
      dsk::gemm_bf16_p(p.rs->y, n, g.ckv, kHead, kDim, p.rs->ckvf + (size_t)r0 * kHead, st);
      if (ncomp > 1) dsk::gemm_bf16_p(p.rs->y, n, g.cgate, kHead, kDim, p.rs->cscf + (size_t)r0 * kHead, st);
      else ops::round_bf16(p.rs->ckvf + (size_t)r0 * kHead, (size_t)n * kHead, st);  // ratio 1: model.py projects in bf16
    }
  }
  const comm::AgSeg ag[4] = {{p.rs->qa + (size_t)r0 * kQLora, p.rs->qa, (size_t)Q * kQLora * 2},
                             {p.rs->kvx + (size_t)r0 * kHead, p.rs->kvx, (size_t)Q * kHead * 2},
                             {p.rs->ckvf + (size_t)r0 * kHead, p.rs->ckvf, (size_t)Q * kHead * 4},
                             {p.rs->cscf + (size_t)r0 * kHead, p.rs->cscf, (size_t)Q * kHead * 4}};
  comm::allgather_multi(ch, ag, 2 + ncomp, st);
  // the window: the 127 positions before the chunk (none in the replay), then the chunk's own keys
  const int base = win_lo >= 0 ? win_lo : std::max(0, a - 127);
  const attn::Rope tab = L < 2 ? plain() : yarn();
  const attn::Window win{p.rs->lin_kv, p.rs->lin_sc, -1, base};
  ring_copy(L, base, a, base, true);
  attn::kv_write(U(p.rs->kvx), U(w.kv_norm), p.rs->pos, tab, win, nullptr, M, st);
  if (ncomp) {
    const KvSrc& k = kvs[kv_slot(L)];
    const attn::CompressorWeights cw{ratio_of(L), nullptr, nullptr, U(w.c_norm), U(w.c_wk), U(w.c_knorm)};
    attn::compressor_emit(p.rs->ckvf, p.rs->cscf, p.rs->pos, M, cw, attn::CompressorCarry{k.carry_kv, k.carry_sc}, yarn(),
                          k.compressed(), k.keys(), nullptr, 0, st);
  }
  // queries: q_norm, act_quant, wq_b; the index layers' top-k
  attn::q_norm(U(p.rs->qa), U(w.q_norm), U(p.rs->qr), M, st);
  dsk::act_quant(p.rs->qr, M, kQLora, p.rs->qq, p.rs->qs, st);
  const dsk::Fp8Seg qseg{w.wqb, w.sqb, p.rs->q, kQ};
  dsk::gemm_fp8_p(p.rs->qq, p.rs->qs, M, &qseg, 1, kQLora, st);
  if (idx_src(L)) index_p(L, a, M);
  // attention over the window and the selected compressed entries
  const int kmax = L < 2 ? 0 : kIdxTopk;
  const int* tk = L < 2 ? nullptr : p.rs->topk[idx_slot(index_source(L))];
  const attn::Compressed ckv = L < 2 ? attn::Compressed{nullptr, nullptr, 0} : kvs[kv_slot(kv_source(L))].compressed();
  dsk::sparse_attn_p(U(p.rs->q), p.rs->pos, M, tab, win, ckv, tk, kmax, w.sink, U(p.rs->o), st, std::max(win_lo, 0));
  // the window's last 256 positions back into the ring; snapshots of it inside the chunk
  ring_copy(L, std::max(base, a + M - attn::kRing), a + M, base, false);
  for (const auto& sn : snaps) {
    const size_t off = (size_t)(sn.first - kSnapRows - base);
    bk_copy(*sn.second, snap_kv_off(L), p.rs->lin_kv + off * kHead, kSnapKv, true);
    bk_copy(*sn.second, snap_sc_off(L), p.rs->lin_sc + off * (kHead / 32), kSnapSc, true);
  }
  // output projections: wo_a on every row (this rank's heads); then wo_b on this rank's rows with every
  // rank's slice of it, whose inputs (every rank's wo_a outputs for these rows) an all-to-all brings on
  // the comm stream while this rank's own slice computes. The bf16 partials, added in rank order, are
  // the sums a reduce-scatter of every row's partials gives, for a fifth of its traffic. Then the
  // boundary on this rank's rows: mid and the FFN's input
  dsk::gemm_woa_p(p.rs->o, M, w.woa, w.soa, kOPart, kWoaK, kOGroupRows, nullptr, p.rs->oq, p.rs->os, st);
  cs_after(p, st, 0);
  const comm::A2aSeg oa[2] = {{p.rs->oq, kOPart, p.rs->oqa, kOPart, kOPart},
                              {p.rs->os, kOPart / 32, p.rs->osa, kOPart / 32, kOPart / 32}};
  comm::alltoall_rows(ch, oa, 2, Q, p.cs, kRsBlocks);
  auto wob = [&](int k, const uint8_t* xq, const uint8_t* xs) {
    const dsk::Fp8Seg seg{g.wob + (size_t)k * kDim * kOPart, g.sob + (size_t)k * (kDim / 32) * (kOPart / 32),
                          p.rs->part + (size_t)k * Q * kDim, kDim};
    dsk::gemm_fp8_p(xq, xs, n, &seg, 1, kOPart, st);
  };
  if (n > 0) wob(rank, p.rs->oq + (size_t)r0 * kOPart, p.rs->os + (size_t)r0 * (kOPart / 32));
  rs_join(p, st, 0);
  if (n > 0) {
    for (int k = 0; k < kRanks; ++k)
      if (k != rank) wob(k, p.rs->oqa + (size_t)k * Q * kOPart, p.rs->osa + (size_t)k * Q * (kOPart / 32));
    dsk::sum4_bf16_p(p.rs->part, (size_t)Q * kDim, (size_t)n * kDim, p.rs->sub, st);
    ablit::rows_bf16_lam(U(p.rs->sub), n, kDim, ab, L, pf_ablit, st);  // these rows' whole sums
    boundary_p(n, p.rs->sub, p.rs->x, p.rs->a_post, p.rs->a_comb, p.rs->a_pre, p.rs->mid, L, 1, p.rs->pm, p.rs->f_post, p.rs->f_comb);
  }
}

// ---- FFN ---------------------------------------------------------------------------------------------

// On entry y / yq / ys hold this rank's rows' normed input and f_post / f_comb their mixes.
void Model::Impl::ffn_p(int L, int M) {
  NVTX_RANGE("ffn L%02d", L);
  Prefill& p = *pf;
  const LayerW& w = lw[L];
  const int Q = p.rq, r0 = p.r0, n = p.n;
  if (n > 0) {
    dsk::gemm_bf16_p(p.rs->y, n, w.gate, kExperts, kDim, p.rs->gl, st);
    dsk::router_topk(p.rs->gl, w.gate_bias, n, kExperts, p.rs->ridx + (size_t)r0 * kTopK, p.rs->rwts + (size_t)r0 * kTopK,
                     kRouteScale, st);
    if (!img_span.empty() && w.gate_bias_vl)  // vision: image-span rows route with the VL bias (model.py Gate)
      img_runs(p.a_cur + r0, n, [&](size_t, int i0, int i1) {
        dsk::router_topk(p.rs->gl + (size_t)i0 * kExperts, w.gate_bias_vl, i1 - i0, kExperts,
                         p.rs->ridx + (size_t)(r0 + i0) * kTopK, p.rs->rwts + (size_t)(r0 + i0) * kTopK, kRouteScale, st);
      });
  }
  // every row's act_quant and routes for the MoE
  const comm::AgSeg ag[4] = {{p.rs->yq + (size_t)r0 * kDim, p.rs->yq, (size_t)Q * kDim},
                             {p.rs->ys + (size_t)r0 * (kDim / 32), p.rs->ys, (size_t)Q * (kDim / 32)},
                             {p.rs->ridx + (size_t)r0 * kTopK, p.rs->ridx, (size_t)Q * kTopK * 4},
                             {p.rs->rwts + (size_t)r0 * kTopK, p.rs->rwts, (size_t)Q * kTopK * 4}};
  comm::allgather_multi(ch, ag, 4, st);
#ifdef DSGB10_NVTX
  if (route_dump_dir() && rank == 0) dump_routes(p, L, M, st);
#endif
  // the MoE (bucketing, shared expert, up), then down in column groups. Each group's combine and its
  // reduce-scatter by rows (fp32 sums in rank order, rounded to bf16) run as one kernel on the comm
  // stream under the next group's down; for chunks whose combine runs inside the down kernel
  // (moe_prefill_fuses), each group's fp32 outputs are reduce-scattered instead
  dsk::MoePCfg mc;
  mc.phases = 1 | 2 | 4;
  dsk::moe_prefill(p.rs->yq, p.rs->ys, M, p.rs->ridx, p.rs->rwts, kExperts, w.moe, p.rs->moe_ws, p.ymoe, st, mc);
  const bool fuses = dsk::moe_prefill_fuses(M);
  for (int c = 0; c < kColGroups; ++c) {
    mc.phases = fuses ? 8 | 16 : 8, mc.col0 = c * kGroupCols, mc.cols = kGroupCols;
    dsk::moe_prefill(p.rs->yq, p.rs->ys, M, p.rs->ridx, p.rs->rwts, kExperts, w.moe, p.rs->moe_ws, p.ymoe, st, mc);
    cs_after(p, st, c);
    if (fuses)
      comm::reduce_scatter_rows_f32(ch, p.ymoe + mc.col0, kDim, p.ysum + mc.col0, kDim, Q, kGroupCols, p.cs,
                                    kRsBlocks);
    else
      dsk::moe_combine_rs(ch, p.rs->moe_ws, M, kExperts, Q, mc.col0, kGroupCols, p.rs->sub, kDim, p.cs, kCombRsBlocks);
  }
  rs_join(p, st, 1);
  // the abliteration's shared-expert share: every rank's dot products of every row (each rank holds its 576 of the
  // intermediate units), gathered on every rank (a collective: a rank without rows of its own takes part)
  const bool ab_sh = pf_ablit != 0.f && ab.s[L][1] != 0.f;
  if (ab_sh) {
    const uint8_t *hq, *hs;
    dsk::moe_prefill_shared(p.rs->moe_ws, M, kExperts, &hq, &hs);
    ablit::shexp_dots(ab, L, M, hq, 576, hs, 18, st);
    comm::allgather(ch, ab.c_part, ab.c_all, (size_t)((M + 1) & ~1) * ablit::kK * 4, st);
  }
  if (n == 0) return;
  if (fuses) ops::f32_to_bf16(p.ysum, p.rs->sub, n * kDim, st);
  if (ab_sh) ablit::shexp_apply_bf16(U(p.rs->sub), n, kDim, r0, M, ab, L, pf_ablit, st);
  // the boundary: x and the next layer's attention input, unless Engram changes x first; after the last
  // layer only the head's collapse
  const int nL = L + 1 < kLayers && !(engram_layer(L + 1) && store) ? L + 1 : -1;
  boundary_p(n, p.rs->sub, p.rs->mid, p.rs->f_post, p.rs->f_comb, p.rs->pm, L + 1 < kLayers ? p.rs->x : nullptr, nL, 0, p.rs->a_pre, p.rs->a_post,
             p.rs->a_comb);
}

// ---- CED -------------------------------------------------------------------------------------------

// Rows [i, i + cnt) of this rank's rows to (or from) consecutive records starting at rec.
static void ced_copy(Prefill& p, uint8_t* rec, int i, int cnt, bool to_rec, cudaStream_t st) {
  const struct {
    uint8_t* buf;
    size_t off, row;
  } f[7] = {{reinterpret_cast<uint8_t*>(p.rs->x + (size_t)i * kHcDim), kCedX, (size_t)kHcDim * 2},
            {reinterpret_cast<uint8_t*>(p.rs->y + (size_t)i * kDim), kCedY, (size_t)kDim * 2},
            {p.rs->yq + (size_t)(p.r0 + i) * kDim, kCedYq, (size_t)kDim},
            {p.rs->ys + (size_t)(p.r0 + i) * (kDim / 32), kCedYs, (size_t)kDim / 32},
            {reinterpret_cast<uint8_t*>(p.rs->a_pre + (size_t)i * kHc), kCedPre, (size_t)kHc * 4},
            {reinterpret_cast<uint8_t*>(p.rs->a_post + (size_t)i * kHc), kCedPost, (size_t)kHc * 4},
            {reinterpret_cast<uint8_t*>(p.rs->a_comb + (size_t)i * kHc * kHc), kCedComb, (size_t)kHc * kHc * 4}};
  for (const auto& e : f) {
    if (to_rec)
      CUDA_CHECK(cudaMemcpy2DAsync(rec + e.off, kCedRow, e.buf, e.row, e.row, cnt, cudaMemcpyDeviceToDevice, st));
    else
      CUDA_CHECK(cudaMemcpy2DAsync(e.buf, e.row, rec + e.off, kCedRow, e.row, cnt, cudaMemcpyDeviceToDevice, st));
  }
}

// Layer 20's compressor for every row of an encoder chunk: on entry y holds this rank's rows' layer-20
// input (the boundary after layer 19), as the exact path's layer 20 has it.
void Model::Impl::ced_project(int M) {
  NVTX_RANGE("ced_project");
  Prefill& p = *pf;
  constexpr int L = 20;
  const LayerW& w = lw[L];
  if (p.n > 0) {
    dsk::gemm_bf16_p(p.rs->y, p.n, p.pl[L].ckv, kHead, kDim, p.rs->ckvf + (size_t)p.r0 * kHead, st);
    ops::round_bf16(p.rs->ckvf + (size_t)p.r0 * kHead, (size_t)p.n * kHead, st);  // ratio 1: model.py projects in bf16
  }
  const comm::AgSeg ag{p.rs->ckvf + (size_t)p.r0 * kHead, p.rs->ckvf, (size_t)p.rq * kHead * 4};
  comm::allgather_multi(ch, &ag, 1, st);
  const KvSrc& k = kvs[kv_slot(L)];
  const attn::CompressorWeights cw{ratio_of(L), nullptr, nullptr, U(w.c_norm), U(w.c_wk), U(w.c_knorm)};
  attn::compressor_emit(p.rs->ckvf, p.rs->cscf, p.rs->pos, M, cw, attn::CompressorCarry{k.carry_kv, k.carry_sc}, yarn(),
                        k.compressed(), k.keys(), nullptr, 0, st);
}

// After an encoder chunk [a, a + M): the layer-20 input of its last kReplay positions (this rank's rows
// of them) into their slots.
void Model::Impl::ced_keep(int a, int M) {
  NVTX_RANGE("ced_keep");
  Prefill& p = *pf;
  const int lo = std::max(a, a + M - kReplay);
  for (int q = lo; q < a + M; ++q) p.ced_pos[q % kReplay] = q, p.ced_src[q % kReplay] = (q - a) / p.rq;
  for (int i = std::max(0, lo - a - p.r0); i < p.n;) {
    const int slot = (a + p.r0 + i) % kReplay, cnt = std::min(p.n - i, kReplay - slot);
    ced_copy(p, p.ced_rows + (size_t)slot * kCedRow, i, cnt, true, st);
    i += cnt;
  }
}

// The decoder replay: layers 20-39 over [P - 128, P) as a chunk of 128 rows (this rank's 32 from the
// records of the ranks that ran them), the window truncated at P - 128, the decoder rings rewritten.
void Model::Impl::ced_replay(int P, const std::vector<std::pair<int, const std::vector<uint32_t>*>>& snaps) {
  NVTX_RANGE("replay P=%d", P);
  Prefill& p = *pf;
  const int s = P - kReplay, M = kReplay;
#ifdef DSGB10_NVTX
  g_dump_a = s;
#endif
  for (int q = s; q < P; ++q)
    if (p.ced_pos[q % kReplay] != q) die("model: CED replay of [%d, %d) without the encoder's position %d", s, P, q);
  p.rq = q_rows(M), p.r0 = rank * p.rq, p.n = std::max(0, std::min(p.rq, M - p.r0));
  p.a_cur = s;
  comm::allgather(ch, p.ced_rows, p.ced_gath, (size_t)kReplay * kCedRow, st);
  for (int i = 0; i < p.n;) {  // runs of consecutive slots from one rank
    const int slot = (s + p.r0 + i) % kReplay, src = p.ced_src[slot];
    int cnt = 1;
    while (i + cnt < p.n && slot + cnt < kReplay && p.ced_src[slot + cnt] == src) ++cnt;
    ced_copy(p, p.ced_gath + ((size_t)src * kReplay + slot) * kCedRow, i, cnt, false, st);
    i += cnt;
  }
  ops::iota(p.rs->pos, s, M, st);
  for (int L = 20; L < kLayers; ++L) {
    NVTX_RANGE("rep L%02d", L);
    // here x holds this rank's replay rows' layer-L input (DSpark's hidden-state hook, a = s, M = 128)
    if (spec) spec_prefill_hidden(L, s, M, {});  // DSpark: main_hidden of the replay's rows
    attention_p(L, s, M, snaps, s);
    ffn_p(L, M);
  }
}

// ---- a chunk ---------------------------------------------------------------------------------------

void Model::Impl::prefill_use(int rows_set) {
  Prefill& p = *pf;
  if (rows_set == 1 && pf_big_ready()) {
    p.rs = &p.big;
  } else if (rows_set == 0) {
    p.rs = &p.rows;
  } else {
    die("model: prefill on row set %d, which this rank does not hold (%s)", rows_set, rows_set == 1 ? "the lent set is not whole" : "no such set");
  }
}

uint64_t Model::Impl::prefill(int a, int M, const Sampling* sp, int handle, const std::vector<int>& snap_pos,
                              bool ced, float ablit, int rows_set) {
  NVTX_RANGE("chunk a=%d M=%d%s%s", a, M, ced ? " ced" : "", sp ? " sample" : "");
  if (!bound) die("model: prefill before bind");
  sf = slot_fac[cur_slot];  // a batch step before it may have left another factor
  pf_ablit = ab.on ? ablit : 0.f;  // every rank gets the same command, so every rank takes the same branches
  if (pf_ablit != 0.f) ablit::reserve(ab, M);
  prefill_alloc();
  prefill_use(rows_set);
  Prefill& p = *pf;
  if (M < 1 || M > p.rs->cap || a < 0 || a + M > slot_ctx_[cur_slot]) die("model: bad prefill [%d, +%d) in slot %d", a, M, cur_slot);
  ensure_mapped(cur_slot, a + M - 1);  // the chunk's entries and the Reindex layers' over-read, before any work is queued
  if (hist.end() < a + M) die("model: prefill [%d, %d) with %d tokens", a, a + M, hist.end());
  if (a < hist.base) die("model: prefill [%d, %d), history held from %d", a, a + M, hist.base);
  if (opt.first_layer != 0 || opt.n_layers != kLayers) die("model: prefill runs the whole model");
  if (ced && sp && a + M < kReplay) die("model: CED prefill ending at %d (the replay needs %d positions)", a + M, kReplay);
  drain();  // the decode buffers and the Engram sets are free; earlier steps' state is in place
  dsk::prefill_tune_poll();  // DSGB10_PF_TUNE=@FILE
  // this chunk's Engram lookups, unless prefill_lookup() started them
  int set = -1;
  for (int k = 0; k < 2; ++k)
    if (p.rs->e_pos0[k] == a && p.rs->e_n[k] == M && p.rs->e_unit[k] == cur_unit && p.rs->e_ep[k] == hist_ep[cur_unit]) set = k;
  if (set < 0) {  // after the drain either set is free on the device
    set = 1 - p.rs->e_last;
    prefill_engram(a, M, set);
  }
  p.rs->e_last = set;
  std::vector<std::pair<int, const std::vector<uint32_t>*>> snaps;
  for (int s : snap_pos) {
    if (s <= a || s > a + M || s < kSnapRows) die("model: snapshot at %d outside (%d, %d]", s, a, a + M);
    if (const std::vector<uint32_t>* b = state_snap_buf(handle, s)) {  // none: no room
      // a pass of the encoder writes layers 0-19 of it: the rest reads as zero on every rank, whatever its numbers held
      if (ced && !(sp && s == a + M)) bk_zero(*b, core->snap_bytes());
      snaps.emplace_back(s, b);
    }
  }
  p.rq = q_rows(M), p.r0 = rank * p.rq, p.n = std::max(0, std::min(p.rq, M - p.r0));
  const int slot = (int)(n_steps % ops::kOutSlots);
  CUDA_CHECK(cudaEventRecord(ev_beg[slot], st));
  CUDA_CHECK(cudaEventRecord(ev_pf[0], st));
  p.jn_n = 0, p.jn_on = p.stall;
  if (p.stall) {  // the pair's own cost: two records with nothing between them
    CUDA_CHECK(cudaEventRecord(p.ev_fl[0], st));
    CUDA_CHECK(cudaEventRecord(p.ev_fl[1], st));
  }
  std::memcpy(p.rs->tok_h, hist.data() + (a - hist.base), (size_t)M * 4);
  p.a_cur = a;
  if (!img_span.empty())  // vision: the spans' ids are -1 in the history (Engram); embed a real id, then the rows
    for (int i = 0; i < M; ++i)
      if (p.rs->tok_h[i] < 0) p.rs->tok_h[i] = 129264;
  CUDA_CHECK(cudaMemcpyAsync(p.rs->tok, p.rs->tok_h, (size_t)M * 4, cudaMemcpyHostToDevice, st));
  ops::iota(p.rs->pos, a, M, st);
  {  // the embedding: every rank sends the rows its shard holds to the rank whose rows they are, and each rank takes its own
    const int bad = embed_shard::first_bad_id(p.rs->tok_h, M);
    if (bad >= 0) die("model: prefill [%d, +%d): token id %d at row %d is outside the vocabulary", a, M, p.rs->tok_h[bad], bad);
    ops::embed_send(embed, rank, p.rs->tok, M, p.rs->part, st);
    const comm::A2aSeg es{p.rs->part, (size_t)kDim * 2, p.rs->mid, (size_t)kDim * 2, (size_t)kDim * 2};
    comm::alltoall_rows(ch, &es, 1, p.rq, st);
    ops::embed_pick(p.rs->mid, p.rq, p.rs->tok + p.r0, p.n, p.rs->x, p.rs->pm, st);
  }
  if (!img_span.empty() && p.n > 0)
    img_runs(a + p.r0, p.n, [&](size_t k, int i0, int i1) {
      // an image encoded in the engine is encoded before the first chunk that reaches its span (serve.cpp): a span reached with its rows
      // unwritten is a bug in that rule, and the engine stops
      if (!vit_span_written(img_key, k)) die("model: the prefill at position %d reached image %zu of flight %d before its rows were written", a, k, img_key);
      const size_t row = img_row0[k] + (size_t)(a + p.r0 + i0 - img_span[k].first);
      imgrows::runs(row, row + (size_t)(i1 - i0), [&](size_t r, size_t cnt) {  // the rows sit 204 to a number
        ops::inject_rows(reinterpret_cast<const bf16*>(img_row(img_key, r)), (int)cnt, p.rs->x + (size_t)(i0 + (r - row)) * kHcDim, st);
      });
    });
  if (ced && a != p.ced_end) std::fill(p.ced_pos, p.ced_pos + kReplay, -1);  // a new run of positions
  p.ced_end = ced ? a + M : -1;
#ifdef DSGB10_NVTX
  g_dump_a = a;
#endif
  for (int L = 0; L < (ced ? 20 : kLayers); ++L) {
    NVTX_RANGE("%s L%02d", ced ? "enc" : "ex", L);
    bool fresh = L == 0;
    if (engram_layer(L) && store) {
      engram_apply_p(L, M, set);
      fresh = true;
    }
    if (fresh && p.n > 0)
      boundary_p(p.n, nullptr, p.rs->x, nullptr, nullptr, p.rs->pm, nullptr, L, 0, p.rs->a_pre, p.rs->a_post, p.rs->a_comb);
    if (spec) spec_prefill_hidden(L, a, M, snap_pos);  // DSpark: main_hidden of the rows its windows take
    if (!ced && L >= 37 && !feat_dir().empty()) feat_hidden(p, L, st);  // drafter2: feature dump
    attention_p(L, a, M, snaps);
    if ((opt.wave_probe >> L & 1) && a == opt.wave_probe_at) wave_probe(L, a, M);
    ffn_p(L, M);
  }
  p.rs->e_pos0[set] = -1;  // used
  p.jn_on = false;     // the replay's layers are not the join clock's
  CUDA_CHECK(cudaEventRecord(ev_pf[1], st));
  if (!ced && !feat_dir().empty()) feat_write(p, ch, a, M, rank, hist.data() + (a - hist.base), st);  // drafter2: feature dump
  if (ced) {
    ced_project(M);
    ced_keep(a, M);
    if (sp) {  // the replay; of the snapshots only one at its end covers the decoder layers
      std::vector<std::pair<int, const std::vector<uint32_t>*>> at_end;
      for (const auto& sn : snaps)
        if (sn.first == a + M) at_end.push_back(sn);
      ced_replay(a + M, at_end);
    }
  }
  CUDA_CHECK(cudaEventRecord(ev_pf[2], st));
  const int Mh = ced ? kReplay : M;  // the rows of the pass the head reads
  uint64_t n = ~0ull;
  if (sp) {  // the last row through the head and the sampling, as a step of one row
    NVTX_RANGE("head");
    // the last row's collapsed stream, from the rank that owns it
    comm::allgather(ch, p.rs->yraw + (size_t)std::max(p.n - 1, 0) * kDim, p.gath, (size_t)kDim * 2, st);
    const bf16* last = reinterpret_cast<const bf16*>(p.gath) + (size_t)((Mh - 1) / p.rq) * kDim;
    n = n_steps;
    if (n >= (uint64_t)ops::kInSlots) wait_done(n - ops::kInSlots);
    const int32_t t = hist.at(a + M - 1) < 0 ? 129264 : hist.at(a + M - 1);  // vision: -1 marks an image position
    next_in(n, 1, a + M - 1, &t, *sp);
    ops::step_in(in_h, in_ctr, fb, in_d, st);
    if (head_coded) dsk::gemv_bf16x_norm(last, norm_w, kNormEps, 1, head, kVocabPart, kDim, logits, nullptr, chead, st);
    else dsk::gemv_bf16_norm(last, norm_w, kNormEps, 1, head, kVocabPart, kDim, logits, nullptr, chead, st);
    ops::sample(ch, logits, kVocabPart * rank, in_d, ctl_h, out_h, fb, samp_ctr, st);
    n_steps = n + 1;
  }
  CUDA_CHECK(cudaEventRecord(ev_pf[3], st));
  if (spec) spec_prefill_accept(a, M, handle, snap_pos, ced, sp);  // into the draft module's windows
  CUDA_CHECK(cudaEventRecord(ev_end[slot], st));
  CUDA_CHECK(cudaEventRecord(ev_pf[4], st));
  hist.prefilled(a + M);  // the prompt's last chunk: the history may be trimmed (Model::trim_history)
  return n;
}

// ---- Model -----------------------------------------------------------------------------------------

void Model::set_tokens(int pos0, int n, const int32_t* tokens) {
  auto& h = d_->hist;
  if (pos0 < 0 || n < 0) die("model: set_tokens(%d, %d)", pos0, n);
  d_->hist_ep[d_->cur_unit] = ++d_->tok_epoch;  // lookups hashed from the unit's earlier tokens are stale
  h.assign(pos0, n, tokens);
  if (d_->spec && d_->spec->copy) d_->spec->copy->set_tokens(d_->cur_unit, pos0, n, tokens, d_->st);  // copy drafting's mirror
}

size_t Model::prefill_init(comm::Ring& ring) {
  d_->prefill_alloc(&ring);
  size_t free = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}

int Model::prefill_rows_cap(int rows_set) const {
  if (!d_->pf) return rows_set == 0 ? std::max(d_->opt.prefill_rows, detail::kReplay) : 0;
  if (rows_set == 1) return d_->pf_big_ready() ? d_->pf->big.cap : 0;
  return rows_set == 0 ? d_->pf->rows.cap : 0;
}

void Model::prefill_lookup(int a, int M, int rows_set) {
  d_->prefill_alloc();
  d_->prefill_use(rows_set);
  Prefill& p = *d_->pf;
  // not the set of the chunk in flight; the other one's chunk has finished (prefill() drains first)
  d_->prefill_engram(a, M, 1 - p.rs->e_last);
}

uint64_t Model::prefill(int a, int M, const Sampling* sp, int handle, const std::vector<int>& snaps, bool ced,
                        float ablit, int rows_set) {
  return d_->prefill(a, M, sp, handle, snaps, ced, ablit, rows_set);
}

}  // namespace model
