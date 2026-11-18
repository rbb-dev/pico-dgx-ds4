// Model::Impl, shared by the model's translation units: model.cu (weights, the decode/verify step,
// launch and wait), prefill.cu (chunks of many rows), state.cu (sessions' saved state) and spec.cu
// (DSpark: the draft module and the speculative steps).
#pragma once
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attn/attn.h"
#include "comm/collectives.h"
#include "common/check.h"
#include "kernels/dsk.h"
#include "model/ablit.h"
#include "model/chunks.h"
#include "model/tokwin.h"
#include "model/model.h"
#include "model/ops.h"
#include "model/pool_rules.h"
#include "model/vmm.h"

namespace model {
namespace detail {

using bf16 = __nv_bfloat16;
inline uint16_t* U(bf16* p) { return reinterpret_cast<uint16_t*>(p); }
inline const uint16_t* U(const bf16* p) { return reinterpret_cast<const uint16_t*>(p); }

// Zeroed before it returns: cudaMemset runs on the legacy stream, which the model's non-blocking
// streams do not wait for, so a kernel writing the buffer right after could be overtaken by the zeroing.
template <class T>
T* dalloc(size_t n) {
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
  CUDA_CHECK(cudaStreamSynchronize(0));
  return static_cast<T*>(p);
}
template <class T>
T* halloc(size_t n) {  // pinned, mapped host memory (the GPU reads it in place)
  void* p = nullptr;
  CUDA_CHECK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped));
  std::memset(p, 0, n * sizeof(T));
  return static_cast<T*>(p);
}

constexpr int kMaxSplits = 2 + kIdxTopk / 64;  // sparse_attn split-K: 2 window blocks + 8 selected blocks
constexpr size_t kEngramSegMax = (size_t)kMaxRows * kEngramCols * kEngramRow;  // one rank owning every cell

struct LayerW {
  const uint8_t *wqa, *sqa, *wkv, *skv, *wqb, *sqb, *woa, *soa, *wob, *sob;
  const bf16 *q_norm, *kv_norm, *attn_norm, *ffn_norm;
  const float* sink;
  // compressor (kv sources): this rank's 128 rows of wkv / wgate, the norm, the index key projection
  const bf16 *c_wkv = nullptr, *c_wgate = nullptr, *c_norm = nullptr, *c_wk = nullptr, *c_knorm = nullptr;
  // indexer (index sources; replicated)
  const uint8_t *ix_wqb = nullptr, *ix_sqb = nullptr;
  const bf16* ix_wproj = nullptr;
  // FFN
  uint8_t* gate = nullptr;  // bf16 [384][5120], tile_fp8 layout: every rank's 96 rows, all-gathered at bind
  const float* gate_bias = nullptr;
  const float* gate_bias_vl = nullptr;  // vision: image-span rows' selection bias (nullptr: the file lacks it)
  dsk::MoeWeights moe{};
  // mHC: the full projections [24][20480] (all-gathered at bind), base [24], scale [3]
  float* hc_fn[2] = {};
  const float *hc_base[2] = {}, *hc_scale[2] = {};
  // Engram
  const uint8_t *e_wkv = nullptr, *e_skv = nullptr;
  float* e_w = nullptr;  // q_weight * k_weight [4][5120]
  // prefill only (prefill.cu): bf16 weights in the GEMMs' tiled layout, made on the first prefill
  uint8_t *p_ckv = nullptr, *p_cgate = nullptr, *p_wproj = nullptr;
  uint32_t* p_hcfn[2] = {};  // hc_fn as dsk::hc_fn_parts
};

// Bytes per entry of a kv source's four arrays: compressed KV, its scales, index keys, their scales.
constexpr size_t kKvEntryBytes[4] = {256, 32, 64, 4};
constexpr int kMapLookahead = 4096;  // positions a line's lists grow beyond what a step needs: a growth in the host gap, not every step

struct KvSrc {  // a kv-source layer's compressed KV, index keys and compressor carry (replicated)
  uint8_t *ckv, *csc, *ik, *iksc;
  int cap;
  // entries a slot's region spans when the arrays are laid out with padding (the slot memory pool,
  // vmm.h): a multiple of kStrideEntries, >= cap; 0: cap (the arrays are plain allocations). No kernel
  // reads cap, so the padding changes the strides only.
  int stride_cap = 0;
  // ik's stride is a multiple of kChunk * 16 so that iksc's (ik's / 16, attn_common.cuh check_slots) is one too
  static constexpr int kStrideEntries = (int)(vmm::kChunk * 16 / kKvEntryBytes[2]);
  float *carry_kv, *carry_sc;
  // the kernels' view (attn.h Compressed): the arrays' bases and this source's chunk lists, [units][list_len]
  // (at(u): unit u's row); ckv..iksc above stay the unit's own region, for the store's copies (state.cu)
  uint8_t *pkv = nullptr, *psc = nullptr, *pik = nullptr, *pisc = nullptr;
  const uint32_t* list = nullptr;  // device
  int list_len = 0;
  attn::Compressed compressed() const { return attn::Compressed{pkv, psc, cap, nullptr, list, list_len}; }
  attn::IndexKeys keys() const { return attn::IndexKeys{pik, pisc, cap, nullptr, list, list_len}; }
  // session slots: slot s's buffers are slot 0's + s strides (whole caches of cap entries, or of stride_cap
  // when padded; the carries' 16 rows of kHead floats)
  size_t stride_entries() const { return (size_t)(stride_cap ? stride_cap : cap); }
  size_t ckv_stride() const { return stride_entries() * kKvEntryBytes[0]; }
  size_t csc_stride() const { return stride_entries() * kKvEntryBytes[1]; }
  size_t ik_stride() const { return stride_entries() * kKvEntryBytes[2]; }
  size_t iksc_stride() const { return stride_entries() * kKvEntryBytes[3]; }
  static constexpr size_t kCarryStride = 16 * kHead;  // floats
  KvSrc at(int s) const {
    KvSrc k = *this;
    if (ckv)  // per-unit arrays (kernel tests and benches); the chunk pool's are reached through the lists
      k.ckv += s * ckv_stride(), k.csc += s * csc_stride(), k.ik += s * ik_stride(), k.iksc += s * iksc_stride();
    k.carry_kv += s * kCarryStride, k.carry_sc += s * kCarryStride;
    k.list += (size_t)s * list_len;
    return k;
  }
};
constexpr size_t kRingKvBytes = (size_t)attn::kRing * kHead, kRingScBytes = (size_t)attn::kRing * kHead / 32;

// DIAGNOSTICS (model.cu k_route_copy): a step's routed experts
constexpr int kRouteRing = 8;
struct RouteRec {
  uint32_t step;
  int32_t R;
  int32_t slot[kMaxRows], pos[kMaxRows];
  int16_t e[kLayers][kMaxRows][kTopK];
};

struct Prefill;  // prefill.cu's buffers
struct Store;    // state.cu's sessions
struct Spec;     // spec.cu's draft module and spec steps

}  // namespace detail

struct Model::Impl {
  using bf16 = detail::bf16;
  int rank;
  Options opt;
  cudaStream_t st = nullptr;
  comm::DevChannel ch{};
  comm::DevChannel ch2{};  // --wave-probe
  comm::Ring* host_ring = nullptr;  // bind: host-side agreement across ranks (a read-back's cut, state.cu)
  bool has_ch2 = false;
  bool bound = false;

  // weights
  detail::LayerW lw[kLayers];
  const bf16 *embed = nullptr, *norm_w = nullptr;
  const uint8_t* head = nullptr;
  bool head_coded = false;  // head holds the coded head (code_head): the head GEMVs read it with gemv_bf16x*

  // state: Options::slots session slots side by side, slot s at slot 0's pointers + s strides. ring_kv,
  // ring_sc and kvs are the current slot's (use_slot), which the eager paths act on; a step reaches each
  // row's slot through slot 0's (ring_kv0, ring_sc0, kvs0) and the step's slot table (step_win and the others)
  uint8_t *ring_kv[kLayers], *ring_sc[kLayers];
  detail::KvSrc kvs[kKvSlots];
  uint8_t *ring_kv0[kLayers], *ring_sc0[kLayers];
  detail::KvSrc kvs0[kKvSlots];
  int nslots = 1, cur_slot = 0;  // logical slots (the server's), the current one
  int nunits = 1, cur_unit = 0;  // physical units (config.h kUnitMul): what slot tables and strides index
  int unit_of[kMaxSlots] = {}, slot_ctx_[kMaxSlots] = {};
  // the slot memory pool (vmm.h; Options::vmm): the kv sources' sixteen arrays are ranges of it, mapped per
  // slot as its session grows (ensure_mapped): array a of kv source k is the pool's array 4 * k + a
  // (kKvEntryBytes order); mapped_upto[s]: the position up to which every array of slot s is mapped (-1: none)
  std::unique_ptr<vmm::Pool> slot_pool;
  int mapped_upto[kMaxSlots] = {};
  // The chunk pool (attn.h Compressed): slot_pool holds four pool-wide arrays (pool_arr: KV, KV scales, index keys, key
  // scales) cut into chunk numbers of attn::kChunkEntries entries, taken from the allocator `calloc` (chunks.h), lowest free first. line_h[u]: the
  // session unit u holds (kNoLine: none; negative: the unit's own); its rows of lists_d [kKvSlots][nunits][list_len] are
  // that session's lists, lists_h the pinned mirror. resv[s]: numbers reserved for slot s's request; store_numbers: the store's share.
  std::unique_ptr<chunks::Alloc> calloc;
  std::vector<uint32_t> refs;  // refs[c]: the lists and bookmarks naming number c (forks share numbers by count)
  uint8_t* pool_arr[4] = {};
  uint32_t *lists_d = nullptr, *lists_h = nullptr;
  int list_len = 0;
  static constexpr int kNoLine = -2147483647 - 1;  // no session (a unit's own are -1 - u)
  int line_h[kMaxUnits];
  size_t store_numbers = 0, pool_max = 0;  // the store's share; the numbers the arrays' address space holds
  int resv[kMaxSlots] = {};
  size_t held_peak = 0;
  bool pool_poison = false;
  int held(int s) const;                 // chunk numbers slot s's session's lists hold
  int numbers_for(int s, int upto) const;  // chunk numbers slot s needs to hold positions [0, upto]
  uint8_t* ent_addr(int k, int a, int j) const;  // the current unit's entry j of kv source k's array a
  // The rotation tables cover what a factor-16 session can reach (attn::rope_table_rows: its 1M window and a margin) at any
  // max_ctx, read by the kernels; factors 32 and 64 compute theirs (attn.h Rope: the same bits either way).
  // rope_plain: layers 0-1 (theta 1e4); rope_y16: the YaRN layers.
  float2* rope_plain = nullptr;
  int rope_plain_rows = 0;
  float2* rope_y16 = nullptr;
  int rope_y16_rows = 0;
  // The YaRN factor (16, 32 or 64) of every slot (Model::set_slot_factor) and the one the step being built or
  // captured rotates with (sf: set from the rows' slots before any work is queued; a step's rows share it).
  // Captured graphs are keyed by it (graphs, Spec::graphs, Spec::bgraphs): their kernel parameters bake it.
  std::array<int, kMaxSlots> slot_fac = [] {
    std::array<int, kMaxSlots> a;
    a.fill(16);
    return a;
  }();
  int sf = 16;
  attn::Rope plain() const { return attn::rope_plain_calc(); }
  attn::Rope yarn() const { return sf == 16 ? attn::Rope(rope_y16) : attn::rope_yarn_calc(sf); }
  void use_factor(int f) { sf = f; }
  int factor_of_slots(const int32_t* slots, int n) const;  // the rows' slots' factor; dies if they differ
  // The dense index scans' score buffers by window level (factor 16, 32, 64): decode's rows (keys, block maxima) and the
  // prefill's batch (pkeys, pbkeys). Factor 16's cover the 1,048,576 positions of its window; with lazy_win, 32 and 64
  // are made when their window opens and freed when it closes. Without lazy_win every factor reads level 0, which covers max_ctx.
  struct Scan {
    int stride = 0;  // keys per row, a multiple of attn::kScoreAlign: the scan's longest context
    uint16_t *keys = nullptr, *bkeys = nullptr, *pkeys = nullptr, *pbkeys = nullptr;
    bool open = false;
    int groups = 0;  // of the pool's memory this level's graphs and buffers took (window_open)
    size_t bytes = 0;  // what its buffers took
  };
  bool lazy_win = false;
  Scan scan_[3];
  static int level_of(int f) { return f == 16 ? 0 : f == 32 ? 1 : 2; }
  const Scan& scan() const { return lazy_win ? scan_[level_of(sf)] : scan_[0]; }
  // window.cu: the pool's memory a window takes while it is open (docs/engine.md, "Context past 1M")
  size_t win_n0 = 0;                // the pool's numbers before the first window took any (0: none taken)
  int win_groups = 0;               // groups of window.cu's kGroupNums (64) numbers out of the pool
  int win_owed = 0;                 // of them, owed back by a window that closed (its memory not yet checked free)
  size_t win_gcost = 0;             // device bytes factor 16's step and batch step graphs took at the start
  size_t win_free_end = 0;          // free device memory when the last window opened
  size_t scan_bytes(int f, bool prefill) const;
  int scan_stride_of(int f) const;
  bool win_alloc(int f);
  void win_free(int f);
  void win_graphs_drop(int f);
  size_t win_numbers_for(int g) const;
  int window_make_room(int f, int groups);
  // The top number of the pool with `g` more groups of factor f's window out of it, and false when the pool could not
  // hold one conversation of the window even whole (kWindowNoRoom). Reads the allocator only.
  bool window_top(int f, int g, uint32_t* top) const;
  size_t win_need(int f) const;  // the bytes factor f's window takes by this rank's own estimate
  int win_busy_g[3] = {0, 0, 0};  // by window level: the exact groups of the last open that found it busy (0: none)
  void window_regrow(int groups);
  int window_open(int f);
  int window_close(int f);

  // step buffers
  bf16 *x, *mid, *y, *yraw, *sub, *part, *qa, *kvx, *q, *iqpre, *o, *hfin;
  float *pm[2], *a_pre, *a_post, *a_comb, *f_post, *f_comb, *hc_chunks, *amix;
  float *ckvf, *cscf, *wproj, *gl, *rwts, *ymoe, *logits, *lgath, *po, *pml;
  uint8_t *yq, *ys, *oq, *os, *pack;
  uint32_t* dense_scratch;  // index_dense's counters and histograms (zero between launches)
  uint16_t *topkey[kIdxSlots], *poolkey;
  int *topk[kIdxSlots], *pool, *ridx;
  void* moe_ws;
  void* moe_pws = nullptr;  // wide64: moe_prefill's workspace (kMaxRows rows)

  // step inputs and results
  ops::StepIn *in_h, *in_d;  // pinned ring [kInSlots] (the host writes), device copy (kernels read)
  const int *tok_d, *pos_d;  // in_d->tokens, in_d->pos
  uint32_t* in_ctr;          // device: steps started (step_in's slot)
  int32_t* fb;               // device [kMaxRows]: the last step's sampled tokens
  ops::SampleOut* out_h;     // pinned ring [kOutSlots]
  uint32_t* ctl_h;           // pinned: rank 0's control word
  uint32_t* samp_ctr;        // device: the sampler's completion count
  uint64_t n_steps = 0;      // launched
  int pos_stride = 1;        // Model::set_pos_stride (bench)
  int pos_group = 1;         // its runs of consecutive rows
  int route_shift = 0;       // Model::set_route_shift (bench)
  int route_group = 1;       // its rows per rotation
  uint64_t n_done = 0;       // known finished
  uint64_t last_n = 0;       // the last step waited for
  cudaEvent_t ev_beg[ops::kOutSlots], ev_end[ops::kOutSlots];
  cudaEvent_t ev_pf[5];    // the last prefill chunk: begin, layers 0-19 done, replay done, head done, end
  cudaEvent_t ev_mark[4];  // Model::mark

  // captured steps: one graph per (rows, context bucket), captured on first use (up to kMaxRows rows: a row
  // count never used costs nothing); a row count runs eagerly once first
  std::map<std::tuple<int, int, int>, cudaGraphExec_t> graphs;  // (rows, context bucket, YaRN factor)
  bool warm[kMaxRows + 1] = {};
  // side streams: the mHC mixes (sa); the window/compressed KV writes and the indexer's head weights (sb)
  cudaStream_t sa = nullptr, sb = nullptr;
  cudaEvent_t ev_fork, ev_join[2];
  // Options::l2pf (kernels/l2pf.cu): the L2 prefetch's side stream and its progress
  cudaStream_t sc = nullptr;
  dsk::L2pfState* l2s = nullptr;

  // Engram
  const engram::HashMap* emap = nullptr;
  engram::Store* store = nullptr;
  TokWin hist;  // token ids of the sequence so far, by position from hist.base (hashing reads 3 back): the current slot's
  TokWin hists[kMaxUnits];  // the other slots' by unit (use_slot swaps them in and out)
  TokWin& hist_of(int u) { return u == cur_unit ? hist : hists[u]; }
  // The rows' ids put at [pos0, pos0 + R), the history cut to pos0 first (a rewind never reaches below the window, which
  // keeps more than a draft's reach).
  void hist_put(TokWin& h, int pos0, int R, const int32_t* tokens) {
    if (!h.put(pos0, R, tokens)) die("model: a history cut to position %d, below its first held position %d", pos0, h.base);
  }
  // each unit's token epoch: set_tokens (a prompt's tokens) gives the unit a new one, so Engram lookups hashed before it
  // (a cancelled prompt's look-ahead) are never taken by the unit's next prompt or by another unit
  uint64_t tok_epoch = 0, hist_ep[kMaxUnits] = {};
  std::vector<uint32_t> ids;  // [R][2][24]
  // vision (docs/vision.md, set_images): the spans of the prefill in progress and their rows (pinned,
  // mapped: the inject kernel reads them in place); empty for every text request
  std::vector<std::pair<int, int>> img_span;
  std::vector<size_t> img_row0;  // span i's first row in img_rows
  uint16_t* img_rows = nullptr;
  size_t img_cap = 0;  // rows img_rows holds
  // the other units' image sets: a prompt's images stay with its unit (use_slot swaps them in and out, as hists), so that
  // another prompt's chunks read between its pieces see their own (interleaved prefill)
  struct ImgSet {
    std::vector<std::pair<int, int>> span;
    std::vector<size_t> row0;
    uint16_t* rows = nullptr;
    size_t cap = 0;
  };
  ImgSet imgs[kMaxUnits];
  // this rank's rows [i0, i1) of the current chunk (absolute first position a) inside span k, clipped to [0, n)
  template <class F>
  void img_runs(int a, int n, F&& f) const {
    for (size_t k = 0; k < img_span.size(); ++k) {
      const int i0 = std::max(0, img_span[k].first - a), i1 = std::min(n, img_span[k].second - a);
      if (i0 < i1) f(k, i0, i1);
    }
  }
  uint8_t* e_send[2];         // pinned: this rank's packed records per Engram layer
  uint32_t* e_off[2];         // pinned: [R][24] offsets into the gathered segments
  uint64_t* e_done;           // pinned: the store's completion flags [2]
  uint64_t e_seq = 0;
  uint8_t *e_gath, *e_xq, *e_xs, *e_part, *e_pgath;
  bf16* e_kv;
  struct Late {  // a launched step's Engram lookups, waiting for its tokens (engram_late)
    bool on = false;
    bool any = false;    // a verify step: its positions and rows are the device's choice
    bool batch = false;  // a batch step (engram_late_batch): its rows' positions and slots
    int pos0 = -1, R = 0;
    uint64_t seq[2] = {};
    int32_t pos[kMaxRows] = {}, slot[kMaxRows] = {};
  } late;

  detail::RouteRec* route_ring = nullptr;  // DSGB10_ROUTE_DUMP
  FILE* route_f = nullptr;

  // re-laying weights the file holds in their natural layout (freed after bind)
  uint8_t* scratch = nullptr;
  size_t scratch_bytes = 0;

  // golden injection / dumps
  uint8_t *inj_h, *inj_g;  // staging [R][20480] bf16 + [R][4] f32 (pinned), gathered (device)
  std::vector<uint8_t> dump_buf;

  dsk::GemvCfg c8k{1, 1, 8}, c8{1, 1, 16}, cwoa{1, 4, 4}, cgate{1, 1, 8}, chead{2, 1, 16};

  ablit::Pack ab;        // the runtime abliteration (Options::ablit)
  float pf_ablit = 0.f;  // the prefill chunk in progress: its session's strength (0: none)

  std::unique_ptr<detail::Prefill> pf;
  std::unique_ptr<detail::Store> store_s;
  std::unique_ptr<detail::Spec> spec;
  bool mh_on = false;  // capturing a spec step: main_hidden at the inputs of layers 37-39
  // The certified head (DSGB10_CERT=1; one-session spec steps whose rows are all greedy without top logprobs,
  // or sampled at a temperature up to 1 without top-k, top-p or top logprobs, bounding the Gumbel score):
  // a coarse read of every vocabulary row (the drafter's FP8 copy of the head) with a rigorous bound per row, one
  // cross-rank threshold per step row, and the exact head kernel on the CTAs that the bound cannot exclude. The
  // sampler then finds the token the full head gives (cert_head, model.cu).
  struct Cert {
    bool on = false, check = false;
    bool samp = false;    // sampled rows too (DSGB10_CERT_SAMPLED=0: off): the bound on l / T - log(-log(u))
    int force = 0;        // DSGB10_CERT_FORCE=N: every Nth step runs every CTA (the fallback, exercised)
    float bscale = 1.f;   // DSGB10_CERT_BSCALE: the bound scaled up (more survivors; measurement)
    const uint8_t *h8 = nullptr, *h8s = nullptr;  // the FP8 copy and its scales (dspark's)
    float *B = nullptr, *clog = nullptr, *xn = nullptr, *logits2 = nullptr;
    comm::ArgMax *cand = nullptr, *gcand = nullptr;
    uint8_t* mask = nullptr;
    uint32_t* stats = nullptr;  // steps, rows checked, mismatches, surviving CTAs (sum), max ratio bits, max CTAs, forced
  } cert;
  bool cert_cap = false;  // capturing a certified body
  void cert_setup(const uint8_t* head_raw, const uint8_t* q, const uint8_t* s);
  void cert_head(int R);

  Impl(int r, const Options& op);
  ~Impl();

  // ---- weights ------------------------------------------------------------------------------
  void bind(dsw::WeightFile& wf, comm::Ring& ring);
  size_t code_head(int* raw_tiles);

  // ---- session slots ------------------------------------------------------------------------
  void use_slot(int s);
  // The slot memory pool. ensure_mapped: slot s's arrays hold every entry a step or copy reaching position `upto`
  // (inclusive) touches, called before the work is queued; dies when the pool has no chunk left, which a slot_reserve()d
  // request cannot reach. slot_release: unmap all of slot s (its session's state is saved and no work for it is queued).
  void ensure_mapped(int s, int upto);
  void slot_release(int s);
  void slot_reserve(int s, int need);
  void slot_unreserve(int s);
  int slot_chunks_for(int s, int need) const;
  // A step's view of the state: with one slot, its pointers (the kernels' one-session code, no table); with
  // more, slot 0's and the step's slot table (in_d->slot), so that one captured step serves rows in any slots.
  attn::Window step_win(int L) const;
  attn::Compressed step_ckv(int k) const;
  attn::IndexKeys step_keys(int k) const;
  attn::CompressorCarry step_carry(int k) const;

  // ---- one step -----------------------------------------------------------------------------
  void sync() { CUDA_CHECK(cudaStreamSynchronize(st)); }
  void dump(Sink* sink, int L, const char* name, int pos0, int R, const void* dev, size_t row_bytes);
  void inject(Sink* sink, int L, const char* name, int pos0, int R, void* dev, size_t row_bytes);
  // Engram's all-gather moves a fixed segment per row count (one rank may own every cell), so a
  // captured step does not depend on where the step's rows live.
  static size_t engram_seg(int R) { return (size_t)R * kEngramCols * kEngramRow; }
  // The positions the dense index scans are checked against: the whole cache. index_dense takes the scan's
  // length from the step's positions, so its launch, and the graph of a row count, is the same for every
  // context (the graph maps keep this key).
  int scan_ctx(int) const { return scan().stride; }
  // side-stream work: fork(s) makes s wait for the main stream so far; join(s) the main stream for s
  void fork(cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(ev_fork, st));
    CUDA_CHECK(cudaStreamWaitEvent(s, ev_fork, 0));
  }
  void join(cudaStream_t s) {
    cudaEvent_t e = ev_join[s == sb];
    CUDA_CHECK(cudaEventRecord(e, s));
    CUDA_CHECK(cudaStreamWaitEvent(st, e, 0));
  }
  void engram_prepare(int pos0, int R, const uint64_t seq[2]);
  void engram_submit(int R, const uint64_t seq[2]);  // the store's requests for the rows' ids
  // a batch step's rows: each one's token into its slot's history, then its session's hashes
  void engram_rows(int R, const int32_t* tok, const int32_t* pos, const int32_t* slot, const uint64_t seq[2],
                   int pad_to = 0);
  void engram_apply(int L, int R);
  void project(const bf16* stream, int L, int h, int R);
  void attention(int L, int R, int ctx, bool dumps, bool sub_in, Sink* sink, int pos0);
  void indexer(int L, int R, int ctx);
  void ffn(int L, int R, bool dumps, Sink* sink, int pos0);
  void l2pf(int w, int L, int R);  // Options::l2pf: window w's prefetch of layer L's weights, on sc
  // The step's device work, from the StepIn ring onwards: eager (flags: dumps, injection, logits) or captured.
  void forward(int R, int ctx, uint32_t flags, Sink* sink, int pos0);
  // its rows through the model: the embedding to this rank's logits (and kFlagLogits' gather)
  void forward_rows(int R, int ctx, uint32_t flags, Sink* sink, int pos0);
  cudaGraphExec_t graph(int R, int ctx);  // of the current factor sf; captured on first use
  uint64_t launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, uint32_t flags, Sink* sink);
  uint64_t launch_batch(const Batch& b);
  void engram_late_batch(const int32_t* tokens);
  // a launch's device work, once its StepIn slot n is filled: the graph of R rows, or eagerly
  uint64_t run_step(uint64_t n, int R, int ctx, uint32_t flags, Sink* sink, int pos0);
  // fill the next StepIn slot (the step's number n) and record its Engram seqs
  ops::StepIn& next_in(uint64_t n, int R, int pos0, const int32_t* tokens, const Sampling& sp);
  void engram_late(int pos0, int R, const int32_t* tokens);
  const ops::SampleOut& wait(uint64_t n);
  void wait_done(uint64_t n) {  // step n has finished (its StepIn slot and Engram buffers are free)
    if (n < n_done) return;
    wait(n);
  }
  void drain() {  // every launched step has finished
    if (n_steps) wait_done(n_steps - 1);
    sync();
  }

  // ---- chunks of many rows (prefill.cu) -----------------------------------------------------
  void prefill_alloc(comm::Ring* ring = nullptr);
  void prefill_engram(int a, int M, int set);  // the Engram lookups of positions [a, a + M) (hist's tokens)
  void engram_apply_p(int L, int M, int set);
  void boundary_p(int M, const bf16* sub, const bf16* res, const float* post, const float* comb, const float* pre,
                  bf16* out, int nL, int nh, float* pre_out, float* post_out, float* comb_out);
  void index_p(int L, int a, int M);
  // win_lo >= 0: CED's decoder replay (the window starts at win_lo with nothing before it; layer 20's
  // compressor ran in the encoder chunks)
  void attention_p(int L, int a, int M, const std::vector<std::pair<int, const std::vector<uint32_t>*>>& snaps, int win_lo = -1);
  void ffn_p(int L, int M);
  void wave_probe(int L, int a, int M);
  // positions [p0, p1) between layer L's ring and the linear window buffer that starts at position base
  void ring_copy(int L, int p0, int p1, int base, bool to_lin);
  uint64_t prefill(int a, int M, const Sampling* sp, int handle, const std::vector<int>& snaps, bool ced, float ablit,
                   int rows_set);
  // CED (docs/ced.md): layer 20's compressor for an encoder chunk's rows; keeping the chunk's last
  // positions' layer-20 input; the decoder replay of [P - 128, P)
  void ced_project(int M);
  void ced_keep(int a, int M);
  void ced_replay(int P, const std::vector<std::pair<int, const std::vector<uint32_t>*>>& snaps);

  // ---- sessions' saved state (state.cu) -----------------------------------------------------
  detail::Store& store_of();
  void state_reserve(size_t bytes);
  // The store in the chunk pool. take_nums: n numbers for session h, lowest free first, evicting the least recently used
  // idle session (store_evict_step) while too few are free; false, nothing taken, when none is left. give_nums: back to
  // the pool. bk_at / bk_copy: a bookmark's byte `off` (its numbers' KV slices, 2 MiB each), copies split at their edges.
  size_t chunk_image() const;  // bytes a chunk number holds across the four arrays
  bool take_nums(int h, size_t n, std::vector<uint32_t>* out);
  // take_extra: the same for a bookmark of session h, only from room no line's reservation needs (false: none taken)
  bool take_extra(int h, size_t n, std::vector<uint32_t>* out);
  // numbers for a piece read back from the disk tier: only what no running request has reserved (take_extra's rule)
  bool take_restore(int h, size_t n, std::vector<uint32_t>* out);
  void give_nums(std::vector<uint32_t>& v);
  // share_nums: the first n numbers of `from` appended to *out, each counted once more (nothing copied). own_tail: session
  // h's numbers from the one holding position pos on become its own (refs 1) before anything writes them: a shared one
  // holding pos is copied (its entries below pos), shared ones above it are replaced; false when the pool has no room.
  void share_nums(const std::vector<uint32_t>& from, size_t n, std::vector<uint32_t>* out);
  bool own_tail(int h, int pos);
  // The disk tier per piece: img_write / bk_write put an image of a number / a bookmark on disk (kNone without room);
  // img_drop drops a reference to an image (its file space is freed at none); drop_images / free_lists: all of a session's
  // images / its entries out of the pool and the disk; write_sealed: a turn's full entries and bookmarks, write-behind.
  size_t img_write(int h, uint32_t c);
  size_t bk_write(int h, const std::vector<uint32_t>& b, size_t bytes);
  void img_drop(size_t off, size_t bytes);
  void drop_images(int h);
  void free_lists(int h);
  void write_sealed(int h);
  bool evict_piece(int v, int h);  // one piece of session v out of the pool (the last first)
  void cut_at(int h, int pos);     // session h cut to [0, pos)
  bool own_bk(int h, std::vector<uint32_t>& b);
  bool store_evict_step(int h);
  uint8_t* bk_at(const std::vector<uint32_t>& n, size_t off) const;
  void bk_copy(const std::vector<uint32_t>& n, size_t off, uint8_t* dev, size_t bytes, bool to_bk);
  void slot_attach(int s, int h);  // slot s's unit holds session h (its rows are h's lists)
  void pool_grow(size_t more);     // model.cu: more numbers in the pool, made and mapped
  size_t state_giveable(const std::vector<int>& keep) const;
  uint64_t state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes, std::string* why);
  void state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id);
  void state_keep(uint64_t gen);
  // state_checkpoint: a planned stop's index while serving (every rank alike; the sessions it names in ckpt_sess).
  // kept_index: an index body of gen: the sessions as a planned stop keeps them (all false) or every one as far as its
  // pieces have images (true), the images in *names. space_free: an image's file space back (inside the tier's size).
  void state_checkpoint(uint64_t gen);
  std::vector<uint8_t> kept_index(uint64_t gen, bool all, std::vector<int>* hs, std::set<size_t>* names);
  void space_free(size_t off, size_t bytes);
  size_t disk_take(int h, size_t bytes);  // room in the disk tier's file for session h (dropping), or kNone
  void state_spill(int h);
  void state_unspill(int h);
  void state_save(int h, int from, int to);
  void state_load(int h, int to);
  // a new bookmark's numbers (the caller fills them: bk_copy), nullptr: no room
  const std::vector<uint32_t>* state_snap_buf(int h, int pos);
  void state_snap(int h, int pos);
  void state_snap_end(int h, int pos);
  void state_restore(int h, int pos);
  void state_restore_end(int h);
  void state_snap_spec(int h, int pos);  // the draft module's windows at positions [pos - 128, pos)

  // ---- DSpark (spec.cu) -----------------------------------------------------------------------
  void bind_spec(dsw::WeightFile& wf, const dsw::WeightFile* draft = nullptr);
  void spec_head_coded();  // code_head: the draft module's bf16 head GEMV on the coded head
  void spec_use_slot(int s);  // the draft module's eager paths act on slot s's windows
  void spec_hidden(int L, int R);  // main_hidden of the step's rows at the input of layer L
  // prefill: main_hidden of the rows that enter the draft's window (the chunk's last 256, the 128 before
  // each snapshot), then their accept()s and the snapshots' copies of the windows. Under CED only the
  // replay (of the chunk that samples) runs layers 37-39: its 128 rows, and a snapshot at its end.
  void spec_prefill_hidden(int L, int a, int M, const std::vector<int>& snaps);
  void spec_prefill_accept(int a, int M, int handle, const std::vector<int>& snaps, bool ced, bool sampled);
  cudaGraphExec_t spec_graph(int ctx);
  void spec_graphs();
  void spec_fill(ops::StepIn& in, int mode, const SpecCfg& c);
  uint64_t launch_spec(int R, int pos0, const int32_t* tokens, const Sampling& sp, const SpecCfg& c);
  uint64_t launch_verify(const Sampling& sp, const SpecCfg& c, int pos_bound);
  // B3 (docs/batching.md): the batch spec step, drafts across sessions
  cudaGraphExec_t spec_batch_graph(int ctx);
  void spec_fill_batch(ops::StepIn& in, int mode, const SpecBatch& b);
  int spec_batch_check(const SpecBatch& b, bool given);
  uint64_t launch_spec_batch(const SpecBatch& b);
  uint64_t launch_verify_batch(const SpecBatch& b);
  void engram_late_rows(int R, const int32_t* tok, const int32_t* pos, const int32_t* slot);
};

}  // namespace model
