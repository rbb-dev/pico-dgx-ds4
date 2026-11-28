// The TP=4 forward (model.h). Layer structure follows ref/inference/model.py (Block.forward with the
// single-pass mHC, Attention, Indexer, Compressor, MoE, Engram, ParallelHead); docs/engine.md has the
// per-step sequence and where each collective sits.
#include "model/model.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "attn/launch.h"
#include "common/alloc_seal.h"
#include "model/exact_rule.h"
#include "model/impl.h"
#include "model/store_trace.h"
#include "model/prefill.h"
#include "model/spec.h"
#include "model/state.h"
#include "model/spec_batch.h"

namespace model {

using namespace detail;

// Long copy steps (copy_run.h): true while an exact body of the one-session step graph is captured. Inside it the families of a wide
// step run through the decode kernels' sums whatever DSGB10_WIDE_EXACT and DSGB10_NORM_EXACT say, so no variable can make a long
// copy step inexact.
namespace detail {
bool g_xbody = false;
}

// DSGB10_GSAMPLE, DSGB10_GCOUPLE (Td per depth) and DSGB10_GCOUPLE_CF (A, B): read once, every rank (main.cpp compares them)
const ops::GSettings& gsample_settings() {
  static const ops::GSettings g = [] {
    ops::GSettings v;
    const char* e = getenv("DSGB10_GSAMPLE");
    v.mode = ops::parse_gsample(e);
    if (v.mode < 0) die("DSGB10_GSAMPLE=%s: 0, 1 or 2", e);
    const char* c = getenv("DSGB10_GCOUPLE");
    const int cp = ops::parse_gcouple(c, v.td);
    if (cp < 0) die("DSGB10_GCOUPLE=%s: Td[,Td2,..], each in (0, 4]", c);
    v.coupled = cp;
    if (cp) {
      const char* f = getenv("DSGB10_GCOUPLE_CF");
      float ab[2] = {1.f, 0.f};
      if (ops::parse_ab(f, ab) < 0) die("DSGB10_GCOUPLE_CF=%s: A,B", f);
      v.cf_a = ab[0], v.cf_b = ab[1];
    }
    if (v.mode)
      std::fprintf(stderr, "sample: DSGB10_GSAMPLE=%d: %s on the Gumbel path (top-p / top-k rows: Gumbel-max among the kept tokens)\n", v.mode,
                   v.mode == 1 ? "sampled requests without a client seed" : "every sampled request");
    return v;
  }();
  return g;
}

namespace {

// Bench only (Model::set_route_shift): row blockIdx.x's router logits [kExperts] rotated by
// shift * (row / group).
// DIAGNOSTICS (DSGB10_ROUTE_DUMP=FILE, rank 0): layer L's routed experts of every row into a ring by step,
// with the step's rows, slots and positions; the host appends each finished step to FILE (wait()).
__global__ void k_route_copy(const int* ridx, const ops::StepIn* in, int L, RouteRec* ring) {
  RouteRec& rr = ring[in->step % kRouteRing];
  const int R = in->rows < kMaxRows ? in->rows : kMaxRows;
  for (int i = threadIdx.x; i < R * kTopK; i += blockDim.x) rr.e[L][i / kTopK][i % kTopK] = (int16_t)ridx[i];
  if (L == 0 && threadIdx.x < kMaxRows) {
    rr.slot[threadIdx.x] = in->slot[threadIdx.x], rr.pos[threadIdx.x] = in->pos[threadIdx.x];
    if (threadIdx.x == 0) rr.R = R, rr.step = in->step;
  }
}

// DSGB10_GATE_SEP=1: a kernel that does nothing, launched without PDL between the router gate's GEMV and the MoE (ffn). At 2-4
// rows the gate's CTAs and the MoE's first grid otherwise share the SMs while the gate streams its rows; with the gate ended
// first the step is shorter. No arithmetic changes.
__global__ void k_gate_sep(const float* gl, int* sink) {
  if (sink && gl[blockIdx.x * kExperts + threadIdx.x] > 3.0e38f) *sink = 1;
}

__global__ void route_rotate(float* gl, int shift, int group) {
  __shared__ float s[kExperts];
  float* g = gl + (size_t)blockIdx.x * kExperts;
  for (int i = threadIdx.x; i < kExperts; i += blockDim.x) s[i] = g[i];
  __syncthreads();
  for (int i = threadIdx.x; i < kExperts; i += blockDim.x) g[(i + shift * (blockIdx.x / group)) % kExperts] = s[i];
}

// DSGB10_WIDE_SH=1: a wide step's shared expert (moe_prefill phase 2: its w1 + w3 SwiGLU GEMM and its w2 GEMM) runs on the
// side stream sb from ffn_norm on, beside the gate GEMM, router_topk and the bucketing + routed up (phases 1 | 4), which
// otherwise run one after another with the shared expert between the router and the routed experts. It needs only the
// normed, quantised input, not the routes; the routed down (phases 8 | 16) starts after its join. The same kernels on
// the same inputs: every output has the same bits.
bool wide_sh() {
  static const bool u = [] {
    const bool on = getenv("DSGB10_WIDE_SH") && atoi(getenv("DSGB10_WIDE_SH"));
    if (on) std::fprintf(stderr, "wide steps: the shared expert beside the gate and the router (DSGB10_WIDE_SH)\n");
    return on;
  }();
  return u;
}

// DSGB10_WIDE_AR=1|2: a wide step's all-reduces (the attention's bf16 partials, the FFN's fp32 ones) run as
// comm::allreduce_bulk_* (each quarter reduced on one rank, then all-gathered) followed by the separate hc_post and
// hc_pre kernels, instead of ops::allreduce_hc, in which every rank receives the other three partials whole. The same
// sums in the same order and the same hc_post / hc_pre arithmetic (ops.h: allreduce_hc is bit-identical to this chain).
// 2: the FFN's bulk all-reduce rounds each reduced quarter to bf16 on the rank that sums it, before the all-gather
// (comm::allreduce_bulk_f32_bf16): the all-gather carries 2 bytes an element instead of 4 and the separate
// f32_to_bf16 kernel goes; the fp32 sums and the one rounding are the same.
int wide_ar_mode() {
  static const int u = [] {
    const int m = getenv("DSGB10_WIDE_AR") ? atoi(getenv("DSGB10_WIDE_AR")) : 0;
    if (m >= 1)
      std::fprintf(stderr, "wide steps: the all-reduce as reduce-scatter + all-gather, %s (DSGB10_WIDE_AR=%d)\n",
                   m >= 2 ? "the FFN's gather in bf16" : "the FFN's gather in fp32", m);
    return m;
  }();
  return u;
}
bool wide_ar() { return wide_ar_mode() >= 1; }

// night (technique 5): at 9..16 rows the normed GEMVs' prologue (every CTA normalising all rows) made wq_b,
// wq_a+wkv and the gate run at 50-150 GB/s; one rmsnorm launch, then the plain r16 GEMVs, gives the same bits
// (test_kernels rows16 checks fused == unfused). DSGB10_FUSED16=1 restores the fused prologues.
bool unfused16() {
  static const bool u = !(getenv("DSGB10_FUSED16") && atoi(getenv("DSGB10_FUSED16")));
  return u;
}

// Exact shared steps (docs/batching.md): a row of a 9..64-row step computes with the arithmetic of the 1..8-row kernels, so a
// greedy or seeded request gets beside others the bits it gets alone. Every setting defaults to the kernels and bodies of
// before; server/main.cpp has refused a value outside its range (model/exact_rule.h) by the time these read it.
// DSGB10_WIDE_EXACT=bits runs parts of a wide step's body (17..64 rows) through the decode kernels in groups of kDecodeRows
// rows, each row computing what it computes in a 16-row step (the weights are read once per group): 1 wq_a + wkv and wq_b,
// 2 wo_a + wo_b, 4 the router gate, 8 the MoE, 16 the Engram's GEMV. Bit 64 runs the families selected by bits 1..16 through
// the row-group kernels instead (dsk::gemv_fp8_rows, gemv_woa_rows, gemv_bf16_rows, moe_rows): the same bits, the weights
// read once for all the rows. 95 = the whole exact wide body. 0 (default): the wide body.
int wide_exact() {
  if (g_xbody) return exact::kWideMax;
  static const int v = [] {
    const int x = exact::wide_exact(getenv("DSGB10_WIDE_EXACT"));
    if (x > 0) std::fprintf(stderr, "exact shared steps: 17..64-row steps through the decode kernels' sums, families %d (DSGB10_WIDE_EXACT)\n", x);
    return x;
  }();
  return v;
}
// DSGB10_WIDE_EXACT_GQ (0, 2, 4, + 16): moe_rows' groups of 16 row slots a pass; 0 picks by the step's rows.
int wide_gq() {
  static const int v = std::max(0, exact::wide_gq(getenv("DSGB10_WIDE_EXACT_GQ")));
  return v;
}
// DSGB10_WIDE_EXACT_MT (1, 2): wo_b's row-group kernel's tiles of 8 weight rows per warp (the same bits; 2 halves the
// activation reads per output: 64 rows of wo_b's input no longer fit L1).
int wide_mt() {
  static const int v = std::max(1, exact::wide_mt(getenv("DSGB10_WIDE_EXACT_MT")));
  return v;
}
// DSGB10_NORM_EXACT=1: the separate RMSNorm launches of a 9..64-row step (attn_norm, q_norm, ffn_norm) add each row's squares in
// the order of the fused kernels a step of at most 8 rows runs (their thread count), so a row's normed input has the same bits
// in a step of any size. 0 / unset: D / 8 threads, as before (the 1/rms may differ in its last bit, which flips a bf16 rounding
// now and then). Steps of 9..16 rows, and wide steps when DSGB10_WIDE_EXACT is set: a plain wide step is not bit-equal whatever
// its norm, and the fused order costs it about 2 % (64 CTAs of 256 threads instead of 640; pico-dgx-ds4's measure).
int norm_nt(const dsk::GemvCfg& c, int R) {
  static const bool on = [] {
    const bool x = exact::norm_exact(getenv("DSGB10_NORM_EXACT")) > 0;
    if (x) std::fprintf(stderr, "exact shared steps: the RMSNorm launches add in the fused kernels' order (DSGB10_NORM_EXACT)\n");
    return x;
  }();
  return (on || g_xbody) && (R <= kDecodeRows || wide_exact()) ? 32 * c.wn * c.wk : 0;
}
// DSGB10_R16_EXACT=bits (test only) runs parts of a 9..16-row step through the kernels of a step of at most 8 rows, in two
// groups (the weights are read once per group): 1 wq_a + wkv and wq_b, 2 wo_a + wo_b, 4 the router gate, 8 the MoE, 16 the
// Engram's GEMV, 32 the head. 0 (default): the 16-row kernels.
int r16_exact() {
  static const int v = [] {
    const int x = std::max(0, exact::r16_exact(getenv("DSGB10_R16_EXACT")));
    if (x > 0) std::fprintf(stderr, "exact shared steps: 9..16-row steps in two groups of 8 rows, families %d (DSGB10_R16_EXACT)\n", x);
    return x;
  }();
  return v;
}
// test only (DSGB10_TF_SOLO=1, the tool's teacher-forced runs of one session with DSGB10_TF_SESSROWS): a given step of more than
// kDecodeRows rows runs as the exact bodies of the one-session step graph do. Never set in serve.
bool tf_solo() {
  static const bool on = [] {
    const char* e = getenv("DSGB10_TF_SOLO");
    const bool v = e && atoi(e) > 0;
    if (v) std::fprintf(stderr, "batch steps: DSGB10_TF_SOLO: steps of more than %d rows run as an exact body (test only)\n", kDecodeRows);
    return v;
  }();
  return on;
}
// Any exact setting in force: a step of one session's rows is then never re-routed by DSGB10_OEA (its text must not depend on its
// step's width). Unset: the re-routing applies to every call of its minimum rows, as before.
bool exact_any() {
  if (g_xbody) return true;
  static const bool v = exact::norm_exact(getenv("DSGB10_NORM_EXACT")) > 0 || exact::wide_exact(getenv("DSGB10_WIDE_EXACT")) > 0 ||
                        exact::r16_exact(getenv("DSGB10_R16_EXACT")) > 0;
  return v;
}

}  // namespace

Model::Impl::Impl(int r, const Options& op) : rank(r), opt(op) {
  tenants.add(lend::kTenantWindows, lend::kKindDevice, lend::kRequired);  // a request's window must not be given away under it
#ifdef DSGB10_TEST_FAULTS
  tenants.add(15, lend::kKindDevice, lend::kOnDemand);  // the dummy block of the hand-over's GPU test
#endif
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  // the side streams' few small kernels go ahead of the main stream's queued GEMV CTAs, or they wait
  // for whole GEMVs to drain (and their results, e.g. the mixes, end up on the critical path)
  int pri_lo = 0, pri_hi = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&pri_lo, &pri_hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&sa, cudaStreamNonBlocking, pri_hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&sb, cudaStreamNonBlocking, pri_hi));
  if (opt.l2pf) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&sc, cudaStreamNonBlocking, pri_hi));
    l2s = dalloc<dsk::L2pfState>(1);
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
  for (auto& e : ev_join) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  c8k.tiled = c8.tiled = true;
  const int R = kMaxRows;
  if (opt.slots < 1 || opt.slots > kMaxSlots) die("model: %d session slots (1..%d)", opt.slots, kMaxSlots);
  nslots = opt.slots;
  // big slots (Options::big_slots): kUnitMul units each, the small ones one unit of max_ctx / kUnitMul
  const int nbig = opt.big_slots > 0 && opt.big_slots < nslots ? opt.big_slots : nslots;
  const int mul = nbig < nslots ? kUnitMul : 1;
  if (mul > 1 && opt.max_ctx % (2 * kUnitMul)) die("model: --max-ctx %d with big slots: a multiple of %d", opt.max_ctx, 2 * kUnitMul);
  const int unit_ctx = opt.max_ctx / mul;
  nunits = 0;
  for (int s = 0; s < nslots; ++s) {
    unit_of[s] = nunits;
    slot_ctx_[s] = s < nbig ? opt.max_ctx : unit_ctx;
    nunits += s < nbig ? mul : 1;
  }
  if (nunits > kMaxUnits) die("model: %d session units (at most %d)", nunits, kMaxUnits);
  // per-layer window rings, and the kv sources' caches, keys and carries: one of each per unit
  for (int L = 0; L < kLayers; ++L) {
    ring_kv0[L] = dalloc<uint8_t>(nunits * kRingKvBytes);
    ring_sc0[L] = dalloc<uint8_t>(nunits * kRingScBytes);
  }
  {
    // The chunk pool (attn.h Compressed): the kv sources' caches are four pool-wide arrays cut into chunk numbers of
    // attn::kChunkEntries entries (KV 2 MiB, KV scales 256 KiB, index keys 512 KiB, key scales 32 KiB a number), made
    // and mapped whole at the start. A slot takes numbers as its session grows, lowest free first, the same on every
    // rank. Number 0 is the guard.
    if (nunits != nslots) die("model: --big-slots with the chunk pool: every slot holds the whole context");
    for (int L : {2, 8, 14, 20}) kvs0[kv_slot(L)].cap = opt.max_ctx / ratio_of(L) + 64;
    list_len = ((opt.max_ctx + 64) / attn::kChunkEntries + 2 + 7) / 8 * 8;  // layer 20's entries, a guard, rounded
    const int per = numbers_for(0, opt.max_ctx - 1);
    const size_t image = (size_t)attn::kChunkEntries * (kKvEntryBytes[0] + kKvEntryBytes[1] + kKvEntryBytes[2] + kKvEntryBytes[3]);
    size_t n = (size_t)nslots * per;
    const size_t full = n;
    if (opt.slot_gib > 0) n = std::min(full, (size_t)(opt.slot_gib * 1073741824.0) / image);
    if (opt.pool_one_line) n = per;  // serve: the pool grows to its chosen size before serving (main.cpp, state_reserve)
    if (n < (size_t)per)
      die("model: a chunk pool of %zu numbers (%.2f GiB) cannot hold one slot of %d positions (%d numbers)", n,
          n * (double)image / 1073741824.0, opt.max_ctx, per);
    const size_t nums = n + 1;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    pool_max = total_b / image + 1;  // the arrays' address space: the whole device (the store's numbers join later)
    size_t bytes[4], vch = 0;
    for (int a = 0; a < 4; ++a) {
      bytes[a] = (nums * attn::kChunkEntries * kKvEntryBytes[a] + vmm::kChunk - 1) / vmm::kChunk * vmm::kChunk;
      vch += bytes[a] / vmm::kChunk;
    }
    std::string err;
    std::unique_ptr<vmm::Backend> be = vmm::make_cuda_backend(&err);
    if (!be) die("model: chunk pool: %s", err.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    slot_pool = std::make_unique<vmm::Pool>(std::move(be), vch);
    if (!slot_pool->ok())
      die("model: chunk pool: %s (%zu of %zu chunks made): a smaller pool (--pool-free-gib, --pool-tokens) or fewer --slots", slot_pool->error().c_str(),
          slot_pool->created(), vch);
    const char* pz = getenv("DSGB10_VMM_POISON");
    pool_poison = pz && std::atoi(pz) > 0;  // debug: the pool starts 0xFF, a chunk number is filled again when taken
    slot_pool->set_poison(pool_poison);
    for (int a = 0; a < 4; ++a) {
      const size_t va = (pool_max * attn::kChunkEntries * kKvEntryBytes[a] + vmm::kChunk - 1) / vmm::kChunk * vmm::kChunk;
      const int id = slot_pool->add_array(va, 1);
      if (id != a) die("model: chunk pool: array %d: %s", a, slot_pool->error().c_str());
      pool_arr[a] = reinterpret_cast<uint8_t*>(slot_pool->base(id));
    }
    if (slot_pool->add_slot(0, 1) != 0 || !slot_pool->ensure(0, bytes)) die("model: chunk pool: %s", slot_pool->error().c_str());
    calloc = std::make_unique<chunks::Alloc>((uint32_t)nums);
    make_store();
    if (!pool_poison)  // the guard (number 0) reads as zero; with poison 0xFF, so an over-read that is not masked shows
      for (int a = 0; a < 4; ++a) CUDA_CHECK(cudaMemset(pool_arr[a], 0, (size_t)attn::kChunkEntries * kKvEntryBytes[a]));
    for (int L : {2, 8, 14, 20}) {
      KvSrc& k = kvs0[kv_slot(L)];
      k.ckv = k.csc = k.ik = k.iksc = nullptr;  // no per-unit regions: the store's copies go through the lists
      k.carry_kv = dalloc<float>(nunits * KvSrc::kCarryStride);
      k.carry_sc = dalloc<float>(nunits * KvSrc::kCarryStride);
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rank == 0)
      std::printf("model: chunk pool: %zu numbers of %.2f MiB (%.2f GiB; the %d slots whole would take %zu, %d each) made and "
                  "mapped once in %.2f s; a slot takes numbers as its session grows, lowest free first\n",
                  n, image / 1048576.0, n * (double)image / 1073741824.0, nslots, full, per, dt);
  }
  if (rank == 0 && nunits != nslots)
    std::printf("model: %d session slots, %d of %d positions and %d of %d (%d units)\n", nslots, nbig, opt.max_ctx,
                nslots - nbig, unit_ctx, nunits);
  {  // the chunk lists (attn.h Compressed): every entry names the guard until a slot grows
    const size_t cnt = (size_t)kKvSlots * nunits * list_len;
    lists_d = dalloc<uint32_t>(cnt);
    CUDA_CHECK(cudaMemset(lists_d, 0, cnt * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&lists_h, cnt * sizeof(uint32_t)));
    std::memset(lists_h, 0, cnt * sizeof(uint32_t));
    for (int k = 0; k < kKvSlots; ++k) {
      KvSrc& s = kvs0[k];
      s.pkv = pool_arr[0], s.psc = pool_arr[1], s.pik = pool_arr[2], s.pisc = pool_arr[3];
      s.list = lists_d + (size_t)k * nunits * list_len, s.list_len = list_len;
    }
  }
  use_slot(0);
  if (const char* e = getenv("DSGB10_GATE_SEP")) {
    if (std::sscanf(e, "%d,%d,%d", &gate_sep, &gate_sep_lo, &gate_sep_hi) < 1 || gate_sep < 0 || gate_sep > 1)
      die("model: DSGB10_GATE_SEP=%s (MODE[,LO,HI], MODE 0 or 1)", e);
    if (rank == 0 && gate_sep) std::printf("model: DSGB10_GATE_SEP=%d in steps of %d..%d rows\n", gate_sep, gate_sep_lo, gate_sep_hi);
  }
  if (const char* rd = getenv("DSGB10_ROUTE_DUMP"); rd && rank == 0) {
    route_ring = dalloc<RouteRec>(kRouteRing);
    route_f = fopen(rd, "wb");
    if (!route_f) die("model: DSGB10_ROUTE_DUMP %s", rd);
  }
  // The YaRN layers' rotation at factor 16 is a table; at 32 and 64 it is computed by the kernels from (position, factor).
  // The plain rotation (layers 0-1 and the drafter) is always computed (attn.h Rope: the same bits as a table), so no
  // table is taken for it.
  if (rank == 0)
    std::printf("model: plain RoPE computed, no table: %.2f GiB a rank not taken\n",
                (double)attn::rope_table_rows(opt.max_ctx, 8) * 32 * sizeof(float2) / (1024.0 * 1024.0 * 1024.0));
  rope_y16_rows = attn::rope_table_rows(opt.max_ctx);
  rope_y16 = dalloc<float2>((size_t)rope_y16_rows * 32);
  attn::rope_table_build(rope_y16, rope_y16_rows, true, 16, st);
  // Serving past 1M opens the windows of factors 32 and 64 when a request first needs them (window.cu): what is made
  // here is what factor 16's window of 1,048,576 positions needs, whatever --max-ctx is.
  lazy_win = opt.pool_one_line && opt.max_ctx > 65536 * 16;
  scan_[0].stride = scan_stride_of(16);
  scan_[0].open = true;

  x = dalloc<bf16>((size_t)R * kHcDim);
  mid = dalloc<bf16>((size_t)R * kHcDim);
  y = dalloc<bf16>((size_t)R * kDim);
  yraw = dalloc<bf16>((size_t)R * kDim);
  sub = dalloc<bf16>((size_t)R * kDim);
  part = dalloc<bf16>((size_t)R * kDim);
  qa = dalloc<bf16>((size_t)R * kQLora);
  kvx = dalloc<bf16>((size_t)R * kHead);
  q = dalloc<bf16>((size_t)R * kQ);
  iqpre = dalloc<bf16>((size_t)R * kIdxQ);
  o = dalloc<bf16>((size_t)R * kQ);
  hfin = dalloc<bf16>((size_t)R * kDim);
  for (auto& p : pm) p = dalloc<float>((size_t)R * kHc);
  a_pre = dalloc<float>((size_t)R * kHc);
  a_post = dalloc<float>((size_t)R * kHc);
  a_comb = dalloc<float>((size_t)R * kHc * kHc);
  f_post = dalloc<float>((size_t)R * kHc);
  f_comb = dalloc<float>((size_t)R * kHc * kHc);
  hc_chunks = dalloc<float>((size_t)attn::hc_chunks(0, kDim) * R * 25);
  amix = dalloc<float>((size_t)kRanks * R * ops::kMixStride);
  ckvf = dalloc<float>((size_t)R * kHead);
  cscf = dalloc<float>((size_t)R * kHead);
  wproj = dalloc<float>((size_t)R * kIdxHeads);
  dense_scratch = dalloc<uint32_t>(attn::index_dense_scratch() / sizeof(uint32_t));
  gl = dalloc<float>((size_t)R * kExperts);
  ridx = dalloc<int>((size_t)R * kTopK);
  rwts = dalloc<float>((size_t)R * kTopK);
  ymoe = dalloc<float>((size_t)R * kDim);
  logits = dalloc<float>((size_t)R * kVocabPart);
  lgath = dalloc<float>((size_t)kRanks * R * kVocabPart);
  po = dalloc<float>((size_t)R * kMaxSplits * kHeads * kHead);
  pml = dalloc<float>((size_t)R * kMaxSplits * kHeads * 2);
  yq = dalloc<uint8_t>((size_t)R * kDim);
  ys = dalloc<uint8_t>((size_t)R * kDim / 32);
  oq = dalloc<uint8_t>((size_t)R * kOPart);
  os = dalloc<uint8_t>((size_t)R * kOPart / 32);
  pack = dalloc<uint8_t>(ops::attn_pack_bytes(R, 2));
  // the index scans' key buffers: one group of kDecodeRows rows (wide64 steps run the indexer in groups)
  scan_[0].keys = dalloc<uint16_t>((size_t)kDecodeRows * scan_[0].stride);
  scan_[0].bkeys = dalloc<uint16_t>((size_t)kDecodeRows * scan_[0].stride / 8);
  for (int i = 0; i < kIdxSlots; ++i) {
    topk[i] = dalloc<int>((size_t)R * kIdxTopk);
    topkey[i] = dalloc<uint16_t>((size_t)R * kIdxTopk);
  }
  pool = dalloc<int>((size_t)R * kPoolBlocks);
  poolkey = dalloc<uint16_t>((size_t)R * kPoolBlocks);
  if (opt.pscan != PScan::Off) {  // the candidates of the four ranks' parts (one set serves the four layers: they run in turn)
    cand_pos = dalloc<int>((size_t)kRanks * kDecodeRows * kIdxTopk);
    cand_key = dalloc<uint16_t>((size_t)kRanks * kDecodeRows * kIdxTopk);
    pcand_pos = dalloc<int>((size_t)kRanks * kDecodeRows * kPoolBlocks);
    pcand_key = dalloc<uint16_t>((size_t)kRanks * kDecodeRows * kPoolBlocks);
  }
  moe_ws = dalloc<uint8_t>(dsk::moe_workspace_bytes());
  moe_pws = dalloc<uint8_t>(std::max(dsk::moe_prefill_workspace_bytes(kMaxRows, kExperts), dsk::moe_rows_workspace_bytes()));  // wide64: moe_prefill's (exact wide: moe_rows')
  in_h = halloc<ops::StepIn>(ops::kInSlots);
  in_d = dalloc<ops::StepIn>(1);
  tok_d = in_d->tokens;
  pos_d = in_d->pos;
  in_ctr = dalloc<uint32_t>(1);
  fb = dalloc<int32_t>(kMaxRows);
  emb_slot = dalloc<bf16>((size_t)kMaxRows * kDim);
  out_h = halloc<ops::SampleOut>(ops::kOutSlots);
  ctl_h = halloc<uint32_t>(1);
  samp_ctr = dalloc<uint32_t>(1);
  for (auto& e : ev_beg) CUDA_CHECK(cudaEventCreate(&e));
  for (auto& e : ev_end) CUDA_CHECK(cudaEventCreate(&e));
  for (auto& e : ev_pf) CUDA_CHECK(cudaEventCreate(&e));
  for (auto& e : ev_mark) CUDA_CHECK(cudaEventCreate(&e));

  for (int e = 0; e < 2; ++e) {
    e_send[e] = halloc<uint8_t>(kEngramSegMax);
    e_off[e] = halloc<uint32_t>((size_t)R * kEngramCols);
  }
  e_done = halloc<uint64_t>(2);
  e_gath = dalloc<uint8_t>(kRanks * kEngramSegMax);
  e_xq = dalloc<uint8_t>((size_t)R * kEngramK);
  e_xs = dalloc<uint8_t>((size_t)R * kEngramK / 32);
  e_kv = dalloc<bf16>((size_t)R * kEngramN);
  e_part = dalloc<uint8_t>((size_t)R * kEngramPartBytes);
  e_pgath = dalloc<uint8_t>((size_t)kRanks * R * kEngramPartBytes);
  ids.resize((size_t)R * 2 * kEngramCols);

  inj_h = halloc<uint8_t>((size_t)R * (kHcDim * 2 + kHc * 4));
  inj_g = dalloc<uint8_t>((size_t)kRanks * R * (kHcDim * 2 + kHc * 4));
  CUDA_CHECK(cudaStreamSynchronize(st));
}

Model::Impl::~Impl() {
  for (auto& g : graphs) cudaGraphExecDestroy(g.second);
  cudaStreamDestroy(sa);
  cudaStreamDestroy(sb);
}

void Model::Impl::dump(Sink* sink, int L, const char* name, int pos0, int R, const void* dev, size_t row_bytes) {
  sync();
  dump_buf.resize(R * row_bytes);
  CUDA_CHECK(cudaMemcpy(dump_buf.data(), dev, R * row_bytes, cudaMemcpyDeviceToHost));
  sink->tensor(L, name, pos0, R, dump_buf.data(), row_bytes);
}

// golden injection: rank 0's Sink::fill, all-gathered so every rank takes the same rows
void Model::Impl::inject(Sink* sink, int L, const char* name, int pos0, int R, void* dev, size_t row_bytes) {
  sync();
  if (rank == 0) sink->fill(L, name, pos0, R, inj_h, row_bytes);
  comm::allgather(ch, inj_h, inj_g, (size_t)R * row_bytes, st);
  CUDA_CHECK(cudaMemcpyAsync(dev, inj_g, (size_t)R * row_bytes, cudaMemcpyDeviceToDevice, st));
}

// ---- weights ----------------------------------------------------------------------------------

// The slices of wq_a, wkv, wo_b (and their scales) and Engram's wkv concatenate rank by rank (the L1 tiles of 8 rows
// and the scale rows of 32 are whole per rank), so slot r of a gathered buffer holds rank r's tensor byte for byte.
// Slot offsets are multiples of 64 bytes (the scales' 1,600, 640, 10,240 and 38,400 B slots are the smallest steps);
// the GEMVs need 4 (scales) and 16 (weights) of them (gemv_fp8.cu: seg.S, seg.W).
namespace {
constexpr size_t kSlotQa = (size_t)kQaPart * kDim, kSlotQaS = (size_t)kQaPart / 32 * (kDim / 32);
constexpr size_t kSlotKv = (size_t)kKvPart * kDim, kSlotKvS = (size_t)kKvPart / 32 * (kDim / 32);
constexpr size_t kSlotOb = (size_t)kDim * kOPart, kSlotObS = (size_t)(kDim / 32) * (kOPart / 32);
constexpr size_t kSlotGate = (size_t)kGatePart * kDim * 2;
constexpr size_t kSlotEw = (size_t)kEngramN * kEngramK, kSlotEwS = (size_t)kEngramN / 32 * (kEngramK / 32);
static_assert(kSlotQa % 64 == 0 && kSlotQaS % 64 == 0 && kSlotKv % 64 == 0 && kSlotKvS % 64 == 0 &&
                  kSlotOb % 64 == 0 && kSlotObS % 64 == 0 && kSlotGate % 64 == 0 && kSlotEw % 64 == 0 && kSlotEwS % 64 == 0,
              "a slot of a gathered weight must keep every rank's slice 64-byte aligned");
}  // namespace

void Model::Impl::reserve_gathered(bool engram) {
  if (!pgw.empty()) die("reserve_gathered: twice");
  pgw.assign(kLayers, GatheredW{});
  for (int L = opt.first_layer; L < opt.n_layers; ++L) {
    GatheredW& g = pgw[L];
    g.wqa = dalloc<uint8_t>(kRanks * kSlotQa), g.sqa = dalloc<uint8_t>(kRanks * kSlotQaS);
    g.wkv = dalloc<uint8_t>(kRanks * kSlotKv), g.skv = dalloc<uint8_t>(kRanks * kSlotKvS);
    g.wob = dalloc<uint8_t>(kRanks * kSlotOb), g.sob = dalloc<uint8_t>(kRanks * kSlotObS);
    pgw_bytes += kRanks * (kSlotQa + kSlotQaS + kSlotKv + kSlotKvS + kSlotOb + kSlotObS);
    g.gate = dalloc<uint8_t>(kRanks * kSlotGate);  // not in pgw_bytes: bind's buffer of decode, whatever prefill does
    if (engram && engram_layer(L)) {
      g.ewkv = dalloc<uint8_t>(kRanks * kSlotEw), g.eskv = dalloc<uint8_t>(kRanks * kSlotEwS);
      pgw_bytes += kRanks * (kSlotEw + kSlotEwS);
    }
  }
}

void* Model::Impl::gathered_dest(const std::string& name, uint64_t bytes) {
  if (pgw.empty() || name.rfind("layers.", 0) != 0) return nullptr;
  char* end = nullptr;
  const long L = std::strtol(name.c_str() + 7, &end, 10);
  if (end == name.c_str() + 7 || *end != '.' || L < opt.first_layer || L >= opt.n_layers) return nullptr;
  const std::string what = end + 1;
  const GatheredW& g = pgw[L];
  struct Row {
    const char* name;
    uint8_t* GatheredW::*buf;
    size_t slot;
  };
  static const Row rows[] = {
      {"attn.wq_a.weight", &GatheredW::wqa, kSlotQa},   {"attn.wq_a.scale", &GatheredW::sqa, kSlotQaS},
      {"attn.wkv.weight", &GatheredW::wkv, kSlotKv},    {"attn.wkv.scale", &GatheredW::skv, kSlotKvS},
      {"attn.wo_b.weight", &GatheredW::wob, kSlotOb},   {"attn.wo_b.scale", &GatheredW::sob, kSlotObS},
      {"ffn.gate.weight", &GatheredW::gate, kSlotGate}, {"engram.wkv.weight", &GatheredW::ewkv, kSlotEw}, {"engram.wkv.scale", &GatheredW::eskv, kSlotEwS}};
  for (const Row& r : rows) {
    if (what != r.name || !(g.*r.buf) || bytes != r.slot) continue;
    uint8_t* at = g.*r.buf + (size_t)rank * r.slot;
    if (reinterpret_cast<uintptr_t>(at) % 64) die("gathered weights: %s would sit at %p, not 64-byte aligned", name.c_str(), (void*)at);
    ++alias_n, alias_bytes += r.slot;
    return at;
  }
  return nullptr;
}

void Model::Impl::bind(dsw::WeightFile& wf, comm::Ring& ring) {
  ch = ring.dev();
  host_ring = &ring;
  store_notes();
  auto need = [&](const std::string& n, dsw::DType t, std::initializer_list<int64_t> shape) -> uint8_t* {
    const dsw::DeviceTensor& d = wf[n];
    if (d.dtype != t || d.ndim != (int)shape.size())
      die("weights: %s has dtype %s ndim %d", n.c_str(), dsw::dtype_name(d.dtype), d.ndim);
    int i = 0;
    for (int64_t s : shape) {
      if (s >= 0 && d.shape[i] != s)
        die("weights: %s dim %d is %lld, expected %lld", n.c_str(), i, (long long)d.shape[i], (long long)s);
      ++i;
    }
    return static_cast<uint8_t*>(d.data);
  };
  using dsw::DType;
  using dsw::Layout;
  // The kernels read tiled layouts (engine/kernels/README.md); a file from before version 2 (or a tensor
  // the converter leaves natural, e.g. Engram's wkv) is re-laid here.
  auto grow = [&](size_t bytes) {
    if (bytes <= scratch_bytes) return;
    sync();
    if (scratch) CUDA_CHECK(cudaFree(scratch));
    scratch = dalloc<uint8_t>(bytes);
    scratch_bytes = bytes;
  };
  auto layout = [&](const std::string& n) { return wf[n].layout; };
  auto tile8 = [&](const std::string& n, uint8_t* p, int N, int Kb, Layout want) {
    const Layout l = layout(n);
    if (l == want) return;
    if (l != Layout::Natural) die("weights: %s is %s, expected %s", n.c_str(), dsw::layout_name(l), dsw::layout_name(want));
    grow((size_t)N * Kb);
    dsk::tile_fp8(p, scratch, N, Kb, st);
    CUDA_CHECK(cudaMemcpyAsync(p, scratch, (size_t)N * Kb, cudaMemcpyDeviceToDevice, st));
  };
  auto tile4 = [&](const std::string& n, uint8_t* w, uint8_t* s, int N, int K) {
    const Layout l = layout(n + ".weight");
    if (l != layout(n + ".scale")) die("weights: %s values and scales differ in layout", n.c_str());
    if (l == Layout::L3) return;
    if (l != Layout::Natural) die("weights: %s is %s, expected L3", n.c_str(), dsw::layout_name(l));
    const size_t wb = (size_t)N * K / 2, sb = (size_t)N * K / 32;
    grow(wb + sb);
    dsk::tile_fp4(w, s, scratch, scratch + wb, N, K, st);
    CUDA_CHECK(cudaMemcpyAsync(w, scratch, wb, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(s, scratch + wb, sb, cudaMemcpyDeviceToDevice, st));
  };

  embed = reinterpret_cast<const bf16*>(need("embed.weight", DType::BF16, {kVocabPart, kDim}));
  std::printf("model: embedding by vocabulary rows: rank %d holds token ids [%d, %d), %.3f GiB of the table's %.3f GiB\n", rank,
              rank * kVocabPart, (rank + 1) * kVocabPart, (double)kVocabPart * kDim * 2 / 1073741824.0,
              (double)kVocab * kDim * 2 / 1073741824.0);
  norm_w = reinterpret_cast<const bf16*>(need("norm.weight", DType::BF16, {kDim}));
  {
    uint8_t* h = need("head.weight", DType::BF16, {kVocabPart, kDim});
    tile8("head.weight", h, kVocabPart, kDim * 2, Layout::L2);
    head = h;
  }
  double code_share = 0;  // the experts' scale codes: the share of 32-scale groups coded, summed over layers
  size_t code_bytes = 0;
  for (int L = opt.first_layer; L < opt.n_layers; ++L) {
    LayerW& w = lw[L];
    const std::string p = "layers." + std::to_string(L) + ".";
    auto f8 = [&](const std::string& n, int N, int K, const uint8_t** W, const uint8_t** S) {
      uint8_t* wp = need(p + n + ".weight", DType::F8_E4M3, {N, K});
      *S = need(p + n + ".scale", DType::F8_E8M0, {(N + 31) / 32, K / 32});
      tile8(p + n + ".weight", wp, N, K, Layout::L1);
      *W = wp;
    };
    f8("attn.wq_a", kQaPart, kDim, &w.wqa, &w.sqa);
    f8("attn.wkv", kKvPart, kDim, &w.wkv, &w.skv);
    f8("attn.wq_b", kQ, kQLora, &w.wqb, &w.sqb);
    f8("attn.wo_a", kOPart, kWoaK, &w.woa, &w.soa);
    f8("attn.wo_b", kDim, kOPart, &w.wob, &w.sob);
    w.q_norm = reinterpret_cast<const bf16*>(need(p + "attn.q_norm.weight", DType::BF16, {kQLora}));
    w.kv_norm = reinterpret_cast<const bf16*>(need(p + "attn.kv_norm.weight", DType::BF16, {kHead}));
    w.attn_norm = reinterpret_cast<const bf16*>(need(p + "attn_norm.weight", DType::BF16, {kDim}));
    w.ffn_norm = reinterpret_cast<const bf16*>(need(p + "ffn_norm.weight", DType::BF16, {kDim}));
    w.sink = reinterpret_cast<const float*>(need(p + "attn.attn_sink", DType::F32, {kHeads}));
    if (kv_src(L)) {
      // version 1 holds the whole compressor on every rank; this rank projects rows [128r, +128)
      const std::string c = p + "attn.compressor.";
      const dsw::DeviceTensor& t = wf[c + "wkv.weight"];
      const int64_t rows = t.shape[0];
      if (rows != kHead && rows != kCompPart) die("weights: %swkv.weight has %lld rows", c.c_str(), (long long)rows);
      const size_t off = rows == kHead ? (size_t)kCompPart * rank * kDim : 0;
      w.c_wkv = reinterpret_cast<const bf16*>(need(c + "wkv.weight", DType::BF16, {rows, kDim})) + off;
      if (ratio_of(L) > 1)
        w.c_wgate = reinterpret_cast<const bf16*>(need(c + "wgate.weight", DType::BF16, {rows, kDim})) + off;
      w.c_norm = reinterpret_cast<const bf16*>(need(c + "norm.weight", DType::BF16, {kHead}));
      w.c_wk = reinterpret_cast<const bf16*>(need(p + "attn.indexer.wk.weight", DType::BF16, {kIdxDim, kHead}));
      w.c_knorm = reinterpret_cast<const bf16*>(need(p + "attn.indexer.k_norm.weight", DType::BF16, {kIdxDim}));
    }
    if (idx_src(L)) {
      f8("attn.indexer.wq_b", kIdxQ, kQLora, &w.ix_wqb, &w.ix_sqb);
      w.ix_wproj = reinterpret_cast<const bf16*>(need(p + "attn.indexer.weights_proj.weight", DType::BF16, {kIdxHeads, kDim}));
    }
    {
      // replicated: all 384 router logits on every rank cost 2.9 MB more reads per layer than this
      // rank's 96 rows, less than their all-gather on the critical path (the tiles of 8 rows concatenate)
      uint8_t* g = need(p + "ffn.gate.weight", DType::BF16, {kGatePart, kDim});
      tile8(p + "ffn.gate.weight", g, kGatePart, kDim * 2, Layout::L2);
      w.gate = !pgw.empty() && pgw[L].gate ? pgw[L].gate : dalloc<uint8_t>((size_t)kExperts * kDim * 2);
      comm::allgather(ch, g, w.gate, (size_t)kGatePart * kDim * 2, st);
      w.gate_bias = reinterpret_cast<const float*>(need(p + "ffn.gate.bias", DType::F32, {kExperts}));
      if (const dsw::DeviceTensor* vl = wf.find(p + "ffn.gate.bias_vl"))  // vision (prefill of image spans only)
        if (vl->dtype == DType::F32 && vl->ndim == 1 && vl->shape[0] == kExperts)
          w.gate_bias_vl = reinterpret_cast<const float*>(vl->data);
    }
    {
      dsk::MoeWeights& m = w.moe;
      const std::string e = p + "ffn.experts.";
      uint8_t* w1 = need(e + "w1.weight", DType::E2M1X2, {kExperts, kInterPart, kDim / 2});
      uint8_t* s1 = need(e + "w1.scale", DType::F8_E8M0, {kExperts, kInterPart, kDim / 32});
      uint8_t* w3 = need(e + "w3.weight", DType::E2M1X2, {kExperts, kInterPart, kDim / 2});
      uint8_t* s3 = need(e + "w3.scale", DType::F8_E8M0, {kExperts, kInterPart, kDim / 32});
      uint8_t* w2 = need(e + "w2.weight", DType::E2M1X2, {kExperts, kDim, kInterPart / 2});
      uint8_t* s2 = need(e + "w2.scale", DType::F8_E8M0, {kExperts, kDim, kInterPart / 32});
      tile4(e + "w1", w1, s1, kExperts * kInterPart, kDim);
      tile4(e + "w3", w3, s3, kExperts * kInterPart, kDim);
      tile4(e + "w2", w2, s2, kExperts * kDim, kInterPart);
      m.w1 = w1, m.s1 = s1, m.w3 = w3, m.s3 = s3, m.w2 = w2, m.s2 = s2;
      // the scales' 1-bit codes (dsk::scale_code), which the decode and prefill MoE kernels read; the codes'
      // raw groups read the scales themselves, which stay
      const size_t n13 = dsk::scale_code_words(kExperts, kInterPart, kDim / 32);
      const size_t n2 = dsk::scale_code_words(kExperts, kDim, kInterPart / 32);
      uint32_t* code = dalloc<uint32_t>(2 * n13 + n2);
      int b1, b3, b2;
      code_share += dsk::scale_code(s1, kExperts, kInterPart, kDim / 32, code, &b1, st) / 3;  // as many groups each
      code_share += dsk::scale_code(s3, kExperts, kInterPart, kDim / 32, code + n13, &b3, st) / 3;
      code_share += dsk::scale_code(s2, kExperts, kDim, kInterPart / 32, code + 2 * n13, &b2, st) / 3;
      m.c1 = {code, (uint32_t)b1 * 0x01010101u};
      m.c3 = {code + n13, (uint32_t)b3 * 0x01010101u};
      m.c2 = {code + 2 * n13, (uint32_t)b2 * 0x01010101u};
      code_bytes += (2 * n13 + n2) * 4;
      f8("ffn.shared_experts.w1", kInterPart, kDim, &m.sh_w1, &m.sh_s1);
      f8("ffn.shared_experts.w3", kInterPart, kDim, &m.sh_w3, &m.sh_s3);
      f8("ffn.shared_experts.w2", kDim, kInterPart, &m.sh_w2, &m.sh_s2);
    }
    // mHC: version 1 splits hc_fn by rows (6 of 24), version 2 by columns (rank r: [5120r, +5120)). Decode reads only
    // this rank's columns, so a version 2 file keeps its quarter as the operand (the whole is made by prefill_alloc, for
    // the prefill's parts only); a version 1 file's rows are all-gathered into the whole, which decode reads.
    for (int h = 0; h < 2; ++h) {
      const std::string n = p + (h ? "hc_ffn_" : "hc_attn_");
      const dsw::DeviceTensor& t = wf[n + "fn"];
      const bool by_rows = t.shape[0] == kMix / kRanks;
      const float* local = reinterpret_cast<const float*>(
          need(n + "fn", DType::F32, {by_rows ? kMix / kRanks : kMix, by_rows ? kHcDim : kDim}));
      if (by_rows) {
        float* whole = dalloc<float>((size_t)kMix * kHcDim);
        comm::allgather(ch, local, whole, (size_t)kMix * kHcDim * 4 / kRanks, st);
        w.hc_fn[h] = whole;
        w.hc_ld = 0;
      } else {
        w.hc_fn[h] = local;
        w.hc_ld = kDim;
      }
      w.hc_base[h] = reinterpret_cast<const float*>(need(n + "base", DType::F32, {kMix}));
      w.hc_scale[h] = reinterpret_cast<const float*>(need(n + "scale", DType::F32, {3}));
    }
    if (engram_layer(L)) {
      uint8_t* ew = need(p + "engram.wkv.weight", DType::F8_E4M3, {kEngramN, kEngramK});
      w.e_skv = need(p + "engram.wkv.scale", DType::F8_E8M0, {kEngramN / 32, kEngramK / 32});
      tile8(p + "engram.wkv.weight", ew, kEngramN, kEngramK, Layout::L1);
      w.e_wkv = ew;
      const bf16* qw = reinterpret_cast<const bf16*>(need(p + "engram.q_weight", DType::BF16, {kHc, kDim}));
      const bf16* kw = reinterpret_cast<const bf16*>(need(p + "engram.k_weight", DType::BF16, {kHc, kDim}));
      w.e_w = dalloc<float>((size_t)kHc * kDim);
      ops::mul_bf16(qw, kw, w.e_w, kHc * kDim, st);
    }
  }
  sync();
  if (scratch) CUDA_CHECK(cudaFree(scratch));
  scratch = nullptr;
  scratch_bytes = 0;
  if (opt.n_layers > opt.first_layer)
    std::printf("moe: the experts' scale codes, %.1f MiB: %.2f %% of their 32-scale groups coded\n", code_bytes / 1048576.0,
                100 * code_share / (opt.n_layers - opt.first_layer));
  if (!opt.ablit.empty()) {
    ablit::load(ab, opt.ablit, rank);
    if (rank == 0) std::printf("model: runtime abliteration from %s\n", opt.ablit.c_str());
  }
  bound = true;
}

// The head's coding: made beside the head, decoded and compared with it in full, then copied over it (the
// weight file's arena cannot give back the rest of the slot). Every reader moves at once: the head GEMVs
// here and in prefill.cu by head_coded, DSpark's bf16 head by spec_head_coded (its FP8 copy, made in bind_spec, is
// not touched). No step graph may hold the raw head's GEMV yet.
size_t Model::Impl::code_head(int* raw_tiles) {
  if (!bound || head_coded) die("model: code_head before bind, or twice");
  if (!graphs.empty()) die("model: code_head after a step was captured");
  sync();
  size_t bytes = 0;
  int raw = 0;
  uint8_t* x = dsk::bf16x_encode(head, kVocabPart, kDim, &bytes, &raw, st);
  if (const uint64_t bad = dsk::bf16x_check(x, head, kVocabPart, kDim, st))
    die("model: the coded head decodes wrong in %llu places", (unsigned long long)bad);
  if (raw_tiles) *raw_tiles = raw;
  if (bytes < (size_t)kVocabPart * kDim * 2) {
    CUDA_CHECK(cudaMemcpyAsync(const_cast<uint8_t*>(head), x, bytes, cudaMemcpyDeviceToDevice, st));
    head_coded = true;
  }
  sync();
  CUDA_CHECK(cudaFree(x));
  if (head_coded && spec) spec_head_coded();
  return head_coded ? bytes : 0;
}

// ---- session slots ----------------------------------------------------------------------------

void Model::Impl::use_slot(int s) {
  if (s < 0 || s >= nslots) die("model: slot %d of %d", s, nslots);
  const int u = unit_of[s];
  if (u != cur_unit) hist.swap(hists[cur_unit]), hist.swap(hists[u]);  // Engram's history: slot s's
  if (u != cur_unit) {  // and its prompt's images
    const auto park = [&](ImgSet& d) {
      std::swap(img_span, d.span), std::swap(img_row0, d.row0), std::swap(img_key, d.key);
    };
    park(imgs[cur_unit]), park(imgs[u]);
  }
  cur_slot = s, cur_unit = u;
  sf = slot_fac[s];  // the eager paths rotate with the slot's factor
  for (int L = 0; L < kLayers; ++L) ring_kv[L] = ring_kv0[L] + u * kRingKvBytes, ring_sc[L] = ring_sc0[L] + u * kRingScBytes;
  for (int k = 0; k < kKvSlots; ++k) kvs[k] = kvs0[k].at(u);
  spec_use_slot(u);  // the draft module's windows, once bound
}

// ---- session slot memory ------------------------------------------------------------------------------
//
// A step or copy reaching position P touches entries [0, P / ratio + 1 + kMapSlackEntries) of every kv source; layer
// 20's block-list scan loads whole 8-entry blocks up to entry P + 7 (indexer.cu load_tile). A chunk is not cleared when
// mapped: whatever a kernel reads was written, and over-read entries only feed rows it masks. DSGB10_VMM_POISON=1
// fills every chunk with 0xFF, which turns any other read into different tokens.

namespace {
constexpr int kMapSlackEntries = 64;  // entries past the position's own (layer 20's block over-read needs 7)
}  // namespace

// ---- the chunk pool -------------------------------------------------------------------------------------
// A slot's lists hold its session's entries [0, P / ratio + 1 + kMapSlackEntries) of every kv source when the step
// that reaches position P runs (the reach above), plus the lookahead, so a step takes numbers in the host gap, not every
// step. Numbers are taken lowest free first, in the same order on every rank. A list entry past the list's end names the
// guard (number 0).

int Model::Impl::numbers_for(int s, int upto) const {
  upto = std::max(0, std::min(upto, slot_ctx_[s] - 1));
  int n = 0;
  for (int k = 0; k < kKvSlots; ++k) {
    const int ratio = k < 3 ? 2 : 1, entries = upto / ratio + 1 + kMapSlackEntries;
    n += std::min(list_len, (entries + attn::kChunkEntries - 1) / attn::kChunkEntries);
  }
  return n;
}

int Model::Impl::slot_chunks_for(int s, int need) const {
  // the request's last position, the lookahead past it and the drafts' rows (a verify step's bound)
  return numbers_for(s, need - 1 + kMapLookahead + kMapSlackEntries);
}

// More numbers in the pool (the store's share, or a window's memory coming back): made and mapped, into the arrays'
// address space (reserved for the whole device at the start).
void Model::Impl::pool_grow(size_t more) {
  const size_t nums = calloc->numbers() + more;
  if (nums > pool_max) die("model: a chunk pool of %zu numbers is past the device's %zu", nums, pool_max);
  size_t bytes[4], want = 0;
  for (int a = 0; a < 4; ++a) {
    bytes[a] = (nums * attn::kChunkEntries * kKvEntryBytes[a] + vmm::kChunk - 1) / vmm::kChunk * vmm::kChunk;
    want += bytes[a] / vmm::kChunk;
  }
  const size_t have = slot_pool->total();
  if (want > have && slot_pool->add_chunks(want - have) != want - have)
    die("model: the chunk pool's %zu more numbers: %s: a smaller pool (--pool-free-gib, --pool-tokens)", more, slot_pool->error().c_str());
  if (!slot_pool->ensure(0, bytes)) die("model: the chunk pool's %zu more numbers: %s", more, slot_pool->error().c_str());
  calloc->grow((uint32_t)nums);
}

attn::Window Model::Impl::step_win(int L) const {
  attn::Window w{ring_kv0[L], ring_sc0[L]};
  if (nunits > 1) w.slot = in_d->slot, w.kv_stride = kRingKvBytes, w.sc_stride = kRingScBytes;
  return w;
}

attn::Compressed Model::Impl::step_ckv(int k) const {
  attn::Compressed c = kvs0[k].compressed();  // unit 0's row: with a slot table, the rows of every unit
  if (nunits > 1) c.slot = in_d->slot;
  return c;
}

attn::IndexKeys Model::Impl::step_keys(int k) const {
  attn::IndexKeys ik = kvs0[k].keys();
  if (nunits > 1) ik.slot = in_d->slot;
  return ik;
}

attn::CompressorCarry Model::Impl::step_carry(int k) const {
  attn::CompressorCarry c{kvs0[k].carry_kv, kvs0[k].carry_sc};
  if (nunits > 1) c.slot = in_d->slot, c.stride = KvSrc::kCarryStride;
  return c;
}

// ---- Engram -----------------------------------------------------------------------------------

// Host side, once a step's tokens are known: hash the rows' n-grams and ask the store for this rank's
// records of both Engram layers (completion seqs seq[0], seq[1]); the GPU waits for them only when it
// reaches layer 1 / 14.
// pad_to > R (a long copy step in an exact body of pad_to rows): the pad rows look up row R - 1's n-grams again, as a wide bucket's do in
// engram_rows, so that the gathered segments and offsets are the body's and the pad rows compute row R - 1's values
void Model::Impl::engram_prepare(int pos0, int R, const uint64_t seq[2], int pad_to) {
  engram::hash_positions(*emap, hist.data(), nullptr, pos0, R, ids.data(), hist.base);
  for (int r = R; r < pad_to; ++r)
    std::copy(ids.begin() + (size_t)(R - 1) * 2 * kEngramCols, ids.begin() + (size_t)R * 2 * kEngramCols,
              ids.begin() + (size_t)r * 2 * kEngramCols);
  engram_submit(std::max(R, pad_to), seq);
}

void Model::Impl::engram_rows(int R, const int32_t* tok, const int32_t* pos, const int32_t* slot, const uint64_t seq[2],
                              int pad_to) {
  for (int r = 0; r < R; ++r) {
    TokWin& h = hist_of(slot[r]);
    if (h.end() < pos[r]) die("model: slot %d's history ends at %d, a row at %d", slot[r], h.end(), pos[r]);
    hist_put(h, pos[r], 1, tok + r);
  }
  for (int r0 = 0, r1; r0 < R; r0 = r1) {  // a session's rows: consecutive positions of one slot
    for (r1 = r0 + 1; r1 < R && slot[r1] == slot[r0] && pos[r1] == pos[r1 - 1] + 1;) ++r1;
    const TokWin& h = hist_of(slot[r0]);
    engram::hash_positions(*emap, h.data(), nullptr, pos[r0], r1 - r0, ids.data() + (size_t)r0 * 2 * kEngramCols, h.base);
  }
  // wide64: a wide bucket's pad rows look up row R - 1's n-grams again, so that the gathered segments and offsets
  // are the bucket's (the captured body's engram_seg) and the pad rows compute row R - 1's values
  for (int r = R; r < pad_to; ++r)
    std::copy(ids.begin() + (size_t)(R - 1) * 2 * kEngramCols, ids.begin() + (size_t)R * 2 * kEngramCols,
              ids.begin() + (size_t)r * 2 * kEngramCols);
  engram_submit(std::max(R, pad_to), seq);
}

void Model::Impl::engram_submit(int R, const uint64_t seq[2]) {
  for (int e = 0; e < 2; ++e) {
    engram::CellMap cm;
    cm.build(store->split(), e, ids.data(), R);
    for (int i = 0; i < R * kEngramCols; ++i)
      e_off[e][i] = (uint32_t)(cm.owner[i] * engram_seg(R) + (size_t)cm.index[i] * kEngramRow);
    engram::Request rq;
    rq.layer = e;
    rq.ids = ids.data();
    rq.n_pos = R;
    rq.out = e_send[e];
    rq.done = reinterpret_cast<std::atomic<uint64_t>*>(&e_done[e]);
    rq.seq = seq[e];
    store->submit(rq);
  }
}

void Model::Impl::engram_apply(int L, int R) {
  const int e = engram_index(L);
  const LayerW& w = lw[L];
  ops::wait_seq(&e_done[e], &in_d->e_wait[e], st);
  comm::allgather(ch, e_send[e], e_gath, engram_seg(R), st);
  ops::engram_pack(e_gath, e_off[e], R, e_xq, e_xs, st);
  const int eg = R > kDecodeRows ? ((wide_exact() & 16) ? kDecodeRows : 0)
                                 : R > dsk::kLowRows && (r16_exact() & 16) ? dsk::kLowRows : 0;
  if (eg && R > kDecodeRows && (wide_exact() & 64)) {  // exact wide: the decode GEMV's sums, one read of the weights
    const dsk::Fp8Seg es{w.e_wkv, w.e_skv, e_kv, kEngramN};
    dsk::gemv_fp8_rows(e_xq, e_xs, R, &es, 1, kEngramK, c8k, st);
  } else if (eg) {  // exact: the decode GEMV in groups
    for (int r0 = 0; r0 < R; r0 += eg)
      dsk::gemv_fp8(e_xq + (size_t)r0 * kEngramK, e_xs + (size_t)r0 * (kEngramK / 32), std::min(eg, R - r0), w.e_wkv, w.e_skv,
                    kEngramN, kEngramK, e_kv + (size_t)r0 * kEngramN, c8k, st);
  } else if (R > kDecodeRows) {  // wide64
    const dsk::Fp8Seg es{w.e_wkv, w.e_skv, e_kv, kEngramN};
    dsk::gemm_fp8_p(e_xq, e_xs, R, &es, 1, kEngramK, st);
  } else {
    dsk::gemv_fp8(e_xq, e_xs, R, w.e_wkv, w.e_skv, kEngramN, kEngramK, e_kv, c8k, st);
  }
  ops::engram_partial(x, e_kv, w.e_w, rank, R, e_part, st);
  comm::allgather(ch, e_part, e_pgath, (size_t)R * kEngramPartBytes, st);
  ops::engram_combine(x, e_pgath, R, x, st);
}

// ---- attention --------------------------------------------------------------------------------

// The index scores and top-k (layer 20: also the candidate pool) in one launch, from the indexer's
// wq_b output iqpre and head weights wproj (both made by attention()). Every rank scans everything: the
// dense scans with index_dense, the Reindex layers' (layer 20's pool) with index_dense_blocks. With pscan_split each rank
// scans its quarter of every session's keys (index_dense's part), one all-gather brings the four quarters' candidates, and
// index_merge2 merges them in place into the picks the whole scan would write (the same bits: attn/indexer.cu).
// wide64: a step of more than kDecodeRows rows scans in groups of kDecodeRows rows, in row order (each row's
// selection is what it is in any call: attn.h).
void Model::Impl::indexer(int L, int R, int ctx) {
  const int ratio = ratio_of(L), slot = idx_slot(L);
  const attn::IndexKeys ik0 = step_keys(kv_slot(kv_source(L)));
  for (int r0 = 0; r0 < R; r0 += kDecodeRows) {
    const int m = std::min(kDecodeRows, R - r0);
    attn::IndexKeys ik = ik0;
    if (ik.slot) ik.slot += r0;
    const attn::IndexSel top{kIdxTopk, topk[slot] + (size_t)r0 * kIdxTopk, topkey[slot] + (size_t)r0 * kIdxTopk, kIdxTopk};
    const uint16_t* iq = U(iqpre) + (size_t)r0 * kIdxQ;
    const float* wp = wproj + (size_t)r0 * kIdxHeads;
    if (reindex(L)) {
      attn::index_dense_blocks(iq, wp, pos_d + r0, yarn(), m, ratio, ik, pool + (size_t)r0 * kPoolBlocks, kPoolBlocks,
                               scan().keys, scan().stride, top, dense_scratch, true, st);
      continue;
    }
    const attn::IndexSel ps = L == 20 ? attn::IndexSel{kPoolBlocks, pool + (size_t)r0 * kPoolBlocks,
                                                       poolkey + (size_t)r0 * kPoolBlocks, kPoolBlocks}
                                      : attn::IndexSel{};
    if (pscan_split(R)) {  // one group: R <= kDecodeRows
      const size_t t = (size_t)R * kIdxTopk, p = (size_t)R * kPoolBlocks;
      const attn::IndexSel ctop{kIdxTopk, cand_pos + rank * t, cand_key + rank * t, kIdxTopk};
      const attn::IndexSel cpool = L == 20 ? attn::IndexSel{kPoolBlocks, pcand_pos + rank * p, pcand_key + rank * p, kPoolBlocks}
                                           : attn::IndexSel{};
      attn::index_dense(iq, wp, pos_d, yarn(), R, ratio, ik, ctx / ratio, scan().keys, scan().stride, scan().bkeys,
                        scan().stride / 8, ctop, cpool, dense_scratch, true, st, -1, rank, kRanks);
      // each rank's own quarter is in place: segment bytes a rank are R rows of 512 positions (4 B) and keys (2 B) and, on layer 20,
      // of 2,048 pool blocks (4 B) and keys (2 B): all multiples of 16
      comm::AgSeg seg[4] = {{cand_pos + rank * t, cand_pos, t * sizeof(int)},
                            {cand_key + rank * t, cand_key, t * sizeof(uint16_t)},
                            {pcand_pos + rank * p, pcand_pos, p * sizeof(int)},
                            {pcand_key + rank * p, pcand_key, p * sizeof(uint16_t)}};
      comm::allgather_multi(ch, seg, L == 20 ? 4 : 2, st);
      attn::index_merge2(R, cand_pos, cand_key, kRanks * kIdxTopk, top, L == 20 ? pcand_pos : nullptr,
                         L == 20 ? pcand_key : nullptr, kRanks * kPoolBlocks, ps, true, st, kRanks);
      continue;
    }
    attn::index_dense(iq, wp, pos_d + r0, yarn(), m, ratio, ik, ctx / ratio, scan().keys, scan().stride, scan().bkeys,
                      scan().stride / 8, top, ps, dense_scratch, true, st);
  }
}

// A sublayer's mixes, sinkhorn(hc_fn . stream), are needed only by its own hc_post (post, comb) and
// the next sublayer's collapse (pre), both in the all-reduce ending it: each rank projects its copy of
// the stream (columns [5120 rank, +5120)) on the side stream, and the partial sums are summed across
// the ranks there too, off the critical path: riding the attention's all-gather (ops::mix_ranks then
// makes the mixes), or by a small collective of their own during the FFN (ops::mix_allreduce: the FFN
// has no collective before its all-reduce).
void Model::Impl::project(const bf16* stream, int L, int h, int R) {
  fork(sa);
  attn::hc_project(U(stream), R, lw[L].hc_fn[h], kDim * rank, kDim * (rank + 1), hc_chunks, false, sa,
                  lw[L].hc_ld);
}

// On entry yraw holds hc_pre(x, pre_mix) before attn_norm (the previous FFN's allreduce_hc made it).
// The attention's mix partials ride its all-gather instead (allgather_attn_mix), which comes early enough
// for the mixes to be worked out on the side stream (ops::mix_ranks) before its all-reduce needs them.
void Model::Impl::attention(int L, int R, int ctx, bool dumps, bool sub_in, Sink* sink, int pos0) {
  const LayerW& w = lw[L];
  project(x, L, 0, R);
  // attn_norm + wq_a + wkv of this rank in one launch (CTA 0 also writes the normed input y);
  // the compressor's slice, then all into the full rows by one fused all-gather
  const int ncomp = kv_src(L) ? (ratio_of(L) > 1 ? 2 : 1) : 0;
  bf16* pqa = reinterpret_cast<bf16*>(pack);
  bf16* pkv = pqa + (size_t)R * kQaPart;
  float* pck = reinterpret_cast<float*>(pkv + (size_t)R * kKvPart);
  float* pcs = pck + (size_t)R * kCompPart;
  const dsk::Fp8Seg seg[2] = {{w.wqa, w.sqa, pqa, kQaPart}, {w.wkv, w.skv, pkv, kKvPart}};
  const bool wide = R > kDecodeRows;  // wide64 (docs/wide64.md): the prefill GEMMs
  const int wx = wide ? wide_exact() : 0;                       // exact: the families of a wide step on the decode kernels' sums
  const int lx = R > dsk::kLowRows && !wide ? r16_exact() : 0;  // exact (test): the families of a 9..16-row step in groups of 8
  if (lx & 1) {  // exact: the fused kernels of a step of at most 8 rows, in two groups
    for (int r0 = 0; r0 < R; r0 += dsk::kLowRows) {
      const dsk::Fp8Seg g[2] = {{w.wqa, w.sqa, pqa + (size_t)r0 * kQaPart, kQaPart}, {w.wkv, w.skv, pkv + (size_t)r0 * kKvPart, kKvPart}};
      dsk::gemv_fp8_norm(yraw + (size_t)r0 * kDim, w.attn_norm, kNormEps, std::min<int>(dsk::kLowRows, R - r0), g, 2, kDim, c8k, st,
                         y + (size_t)r0 * kDim);
    }
  } else if ((wx & 65) == 65) {  // exact wide: the 16-row GEMV's sums, one read of the weights
    dsk::rmsnorm(yraw, w.attn_norm, R, kDim, kNormEps, y, yq, ys, st, norm_nt(c8k, R));
    dsk::gemv_fp8_rows(yq, ys, R, seg, 2, kDim, c8k, st);
  } else if (wx & 1) {  // exact: the 16-row GEMV in groups
    dsk::rmsnorm(yraw, w.attn_norm, R, kDim, kNormEps, y, yq, ys, st, norm_nt(c8k, R));
    for (int r0 = 0; r0 < R; r0 += kDecodeRows) {
      const dsk::Fp8Seg g[2] = {{w.wqa, w.sqa, pqa + (size_t)r0 * kQaPart, kQaPart}, {w.wkv, w.skv, pkv + (size_t)r0 * kKvPart, kKvPart}};
      dsk::gemv_fp8_multi(yq + (size_t)r0 * kDim, ys + (size_t)r0 * (kDim / 32), std::min(kDecodeRows, R - r0), g, 2, kDim, c8k, st);
    }
  } else if (wide) {
    dsk::rmsnorm(yraw, w.attn_norm, R, kDim, kNormEps, y, yq, ys, st, norm_nt(c8k, R));
    dsk::gemm_fp8_p(yq, ys, R, seg, 2, kDim, st);
  } else if (R > dsk::kLowRows && unfused16()) {  // night: 9..16 rows: norm once, then the plain r16 GEMV (same bits)
    dsk::rmsnorm(yraw, w.attn_norm, R, kDim, kNormEps, y, yq, ys, st, norm_nt(c8k, R));
    dsk::gemv_fp8_multi(yq, ys, R, seg, 2, kDim, c8k, st);  // c8k: the fused call's split-K, same bits
  } else {
    dsk::gemv_fp8_norm(yraw, w.attn_norm, kNormEps, R, seg, 2, kDim, c8k, st, y);
  }
  if (dumps) dump(sink, L, "attn_in", pos0, R, y, kDim * 2);
  if (ncomp) {  // this rank's 128 rows of the compressor's wkv (and wgate)
    const attn::CompressorWeights cw{ratio_of(L), U(w.c_wkv), U(w.c_wgate), nullptr, nullptr, nullptr};
    for (int r0 = 0; r0 < R; r0 += kDecodeRows)  // attn's rows per call (wide64: groups)
      attn::compressor_project_rows(U(y) + (size_t)r0 * kDim, cw, 0, kCompPart, pck + (size_t)r0 * kCompPart,
                                    pcs + (size_t)r0 * kCompPart, std::min(kDecodeRows, R - r0), true, st);
  }
  if (idx_src(L)) {  // the indexer's head weights need only y
    fork(sb);
    attn::gemv_bf16(U(y), R, kDim, U(w.ix_wproj), nullptr, kIdxHeads, kIdxHeads, wproj, nullptr, true, sb);
  }
  l2pf(2, L, R);
  join(sa);  // the mix partials
  ops::allgather_attn_mix(ch, pack, R, ncomp, hc_chunks, attn::hc_chunks(0, kDim), qa, kvx, ckvf, cscf, amix, st);
  fork(sa);
  ops::mix_ranks(amix, R, w.hc_scale[0], w.hc_base[0], a_pre, a_post, a_comb, sa);
  // window KV and compressed KV on the side; q_norm + wq_b (+ the indexer's wq_b) in one launch
  const attn::Rope tab = L < 2 ? plain() : yarn();
  const attn::Window win = step_win(L);
  fork(sb);
  attn::kv_write(U(kvx), U(w.kv_norm), pos_d, tab, win, nullptr, R, sb, R > kDecodeRows ? &in_d->rows : nullptr);
  if (ncomp) {
    const int k = kv_slot(L);
    const attn::CompressorWeights cw{ratio_of(L), nullptr, nullptr, U(w.c_norm), U(w.c_wk), U(w.c_knorm)};
    // the carry keeps a call's last 16 rows: wide64 steps emit in groups of kDecodeRows rows, in row order (a
    // session split between two groups is two calls, as two steps)
    for (int r0 = 0; r0 < R; r0 += kDecodeRows) {
      attn::CompressorCarry ca = step_carry(k);
      attn::Compressed cc = step_ckv(k);
      attn::IndexKeys ck = step_keys(k);
      if (ca.slot) ca.slot += r0;
      if (cc.slot) cc.slot += r0;
      if (ck.slot) ck.slot += r0;
      attn::compressor_emit(ckvf + (size_t)r0 * kHead, cscf + (size_t)r0 * kHead, pos_d + r0, std::min(kDecodeRows, R - r0),
                            cw, ca, yarn(), cc, ck, nullptr, 0, sb, R > kDecodeRows ? &in_d->rows : nullptr, r0);
    }
    if (g_xbody && mh_on && ncomp == 2 && R > kDecodeRows) spec_copy_save(k, R, sb);  // long copy steps (spec.cu): the projections the carries are repaired from
  }
  const dsk::Fp8Seg qseg[2] = {{w.wqb, w.sqb, q, kQ}, {w.ix_wqb, w.ix_sqb, iqpre, kIdxQ}};
  if (lx & 1) {  // exact: the fused kernels of a step of at most 8 rows, in two groups
    for (int r0 = 0; r0 < R; r0 += dsk::kLowRows) {
      const dsk::Fp8Seg g[2] = {{w.wqb, w.sqb, q + (size_t)r0 * kQ, kQ}, {w.ix_wqb, w.ix_sqb, iqpre + (size_t)r0 * kIdxQ, kIdxQ}};
      dsk::gemv_fp8_norm(qa + (size_t)r0 * kQLora, w.q_norm, kNormEps, std::min<int>(dsk::kLowRows, R - r0), g, idx_src(L) ? 2 : 1,
                         kQLora, c8k, st);
    }
  } else if ((wx & 65) == 65) {  // exact wide
    dsk::rmsnorm(qa, w.q_norm, R, kQLora, kNormEps, o, oq, os, st, norm_nt(c8k, R));
    dsk::gemv_fp8_rows(oq, os, R, qseg, idx_src(L) ? 2 : 1, kQLora, c8k, st);
  } else if (wx & 1) {  // exact: the 16-row GEMV in groups
    dsk::rmsnorm(qa, w.q_norm, R, kQLora, kNormEps, o, oq, os, st, norm_nt(c8k, R));
    for (int r0 = 0; r0 < R; r0 += kDecodeRows) {
      const dsk::Fp8Seg g[2] = {{w.wqb, w.sqb, q + (size_t)r0 * kQ, kQ}, {w.ix_wqb, w.ix_sqb, iqpre + (size_t)r0 * kIdxQ, kIdxQ}};
      dsk::gemv_fp8_multi(oq + (size_t)r0 * kQLora, os + (size_t)r0 * (kQLora / 32), std::min(kDecodeRows, R - r0), g,
                          idx_src(L) ? 2 : 1, kQLora, c8k, st);
    }
  } else if (wide) {
    dsk::rmsnorm(qa, w.q_norm, R, kQLora, kNormEps, o, oq, os, st, norm_nt(c8k, R));
    dsk::gemm_fp8_p(oq, os, R, qseg, idx_src(L) ? 2 : 1, kQLora, st);
  } else if (R > dsk::kLowRows && unfused16()) {  // night: wq_b's input normed once (oq/os are free until wo_a)
    dsk::rmsnorm(qa, w.q_norm, R, kQLora, kNormEps, o, oq, os, st, norm_nt(c8k, R));  // o: scratch for the bf16 copy
    dsk::gemv_fp8_multi(oq, os, R, qseg, idx_src(L) ? 2 : 1, kQLora, c8k, st);
  } else {
    dsk::gemv_fp8_norm(qa, w.q_norm, kNormEps, R, qseg, idx_src(L) ? 2 : 1, kQLora, c8k, st);
  }
  l2pf(3, L, R);
  join(sb);
  if (idx_src(L)) {
    indexer(L, R, ctx);
    if (dumps) dump(sink, L, "index_topk", pos0, R, topk[idx_slot(L)], kIdxTopk * 4);
    if (dumps && L == 20) dump(sink, L, "pool", pos0, R, pool, kPoolBlocks * 4);
  }
  // sparse attention over the window and the selected compressed entries
  const int kmax = L < 2 ? 0 : kIdxTopk;
  const int* tk = L < 2 ? nullptr : topk[idx_slot(index_source(L))];
  const attn::Compressed ckv = L < 2 ? attn::Compressed{nullptr, nullptr, 0} : step_ckv(kv_slot(kv_source(L)));
  attn::sparse_attn(U(q), pos_d, attn::WinBounds{}, R, tab, win, ckv, tk, kmax, w.sink, U(o), 2 + kmax / 64,
                    attn::AttnScratch{po, pml, kMaxSplits}, st, true);
  // output projections; wo_b is row-parallel: partial sums, all-reduced in fp32 in rank order, then
  // hc_post into mid and the FFN's collapse (by attn_pre) in the same kernel
  if ((wx & 66) == 66) {  // exact wide: wo_a and wo_b with the 16-row kernels' sums, one read of the weights
    dsk::gemv_woa_rows(o, R, w.woa, w.soa, kOPart, kWoaK, kOGroupRows, nullptr, cwoa, st, oq, os);
    const dsk::Fp8Seg bseg{w.wob, w.sob, part, kDim};
    dsk::GemvCfg cb = c8;
    cb.mt = wide_mt();
    dsk::gemv_fp8_rows(oq, os, R, &bseg, 1, kOPart, cb, st);
  } else if ((wx | lx) & 2) {  // exact: wo_a and wo_b as the 16-row (wide) or 8-row (9..16 rows) step runs them, in groups
    const size_t ostride = (size_t)(kOPart / kOGroupRows) * kWoaK;
    const int kDecodeRows = wide ? model::kDecodeRows : dsk::kLowRows;  // the group
    for (int r0 = 0; r0 < R; r0 += kDecodeRows) {
      const int m = std::min(kDecodeRows, R - r0);
      dsk::gemv_woa(o + r0 * ostride, m, w.woa, w.soa, kOPart, kWoaK, kOGroupRows, nullptr, cwoa, false, st,
                    oq + (size_t)r0 * kOPart, os + (size_t)r0 * (kOPart / 32));
    }
    for (int r0 = 0; r0 < R; r0 += kDecodeRows)
      dsk::gemv_fp8(oq + (size_t)r0 * kOPart, os + (size_t)r0 * (kOPart / 32), std::min(kDecodeRows, R - r0), w.wob, w.sob, kDim,
                    kOPart, part + (size_t)r0 * kDim, c8, st);
  } else if (wide) {
    dsk::gemm_woa_p(o, R, w.woa, w.soa, kOPart, kWoaK, kOGroupRows, nullptr, oq, os, st);
    const dsk::Fp8Seg bseg{w.wob, w.sob, part, kDim};
    dsk::gemm_fp8_p(oq, os, R, &bseg, 1, kOPart, st);
  } else {
    dsk::gemv_woa(o, R, w.woa, w.soa, kOPart, kWoaK, kOGroupRows, nullptr, cwoa, false, st, oq, os);
    dsk::gemv_fp8(oq, os, R, w.wob, w.sob, kDim, kOPart, part, c8, st);
  }
  ablit::rows_bf16(U(part), R, kDim, ab, L, in_d, st);  // this rank's partial: the all-reduce sums projected parts
  l2pf(4, L, R);
  join(sa);
  if (R > kDecodeRows && wide_ar()) {  // bulk all-reduce + the separate kernels (same bits)
    ops::allreduce_hc_bulk(ch, part, false, wide_ar_mode(), R, x, a_post, a_comb, a_pre, mid, yraw, sub, nullptr, st);
  } else {
    ops::allreduce_hc(ch, part, false, R, x, a_post, a_comb, a_pre, mid, yraw, dumps ? sub : nullptr, st);
  }
  if (dumps) {
    dump(sink, L, "attn_out", pos0, R, sub, kDim * 2);
    dump(sink, L, "mid", pos0, R, mid, kHcDim * 2);
    dump(sink, L, "hc_attn_pre", pos0, R, a_pre, kHc * 4);
    dump(sink, L, "hc_attn_post", pos0, R, a_post, kHc * 4);
    dump(sink, L, "hc_attn_comb", pos0, R, a_comb, kHc * kHc * 4);
  }
  if (sub_in) {  // the golden FFN input stream, collapsed again
    inject(sink, L, "mid", pos0, R, mid, kHcDim * 2);
    attn::hc_pre(U(mid), a_pre, R, nullptr, U(yraw), nullptr, st);
  }
}

// Options::l2pf (kernels/l2pf.cu): while the window's collective (1, 2, 4) or the attention (3) leaves DRAM idle, a
// paced prefetch into L2 of the weights layer L's step reads next, on sc. It writes nothing the model reads.
void Model::Impl::l2pf(int win, int L, int R) {
  if (!sc || L >= opt.n_layers) return;
  const LayerW& w = lw[L];
  auto s8 = [](int N, int K) { return (size_t)((N + 31) / 32) * (K / 32); };  // an f8 tensor's scales
  const size_t ix = idx_src(L) ? (size_t)kIdxQ * kQLora : 0;
  const dsk::L2pfTarget t[dsk::kL2pfTargets] = {
      {{w.wqa, w.sqa, w.wkv, w.skv},
       {(size_t)kQaPart * kDim, s8(kQaPart, kDim), (size_t)kKvPart * kDim, s8(kKvPart, kDim)}},
      {{w.wqb, w.sqb, w.ix_wqb, w.ix_sqb}, {(size_t)kQ * kQLora, s8(kQ, kQLora), ix, ix ? s8(kIdxQ, kQLora) : 0}},
      {{w.woa, w.soa}, {(size_t)kOPart * kWoaK, s8(kOPart, kWoaK)}},
      {{w.wob, w.sob}, {(size_t)kDim * kOPart, s8(kDim, kOPart)}},
      {{w.gate}, {(size_t)kExperts * kDim * 2}}};
  fork(sc);
  const bool coll = win != 3;  // the attention's window has no collective to stop it
  dsk::l2pf_window(t, L, win, R, coll ? ch.ready : nullptr, coll ? ch.seq : nullptr, l2s, sc);
}

// ---- FFN ----------------------------------------------------------------------------------------

// On entry yraw holds hc_pre(mid, attn_pre) before ffn_norm.
void Model::Impl::ffn(int L, int R, bool dumps, Sink* sink, int pos0) {
  const LayerW& w = lw[L];
  float* next_pm = pm[(L + 1) & 1];
  project(mid, L, 1, R);
  ops::mix_allreduce(ch, hc_chunks, attn::hc_chunks(0, kDim), R, w.hc_scale[1], w.hc_base[1], next_pm, f_post, f_comb,
                     sa);
  if (dumps) {
    dsk::rmsnorm(yraw, w.ffn_norm, R, kDim, kNormEps, y, nullptr, nullptr, st);
    dump(sink, L, "ffn_in", pos0, R, y, kDim * 2);
  }
  // ffn_norm + the router's 384 logits in one launch (CTA 0 also quantises the normed input for the
  // experts); the MoE runs the top-6 selection itself
  const bool wide = R > kDecodeRows;
  const int wx = wide ? wide_exact() : 0;                       // exact: the families of a wide step on the decode kernels' sums
  const int lx = R > dsk::kLowRows && !wide ? r16_exact() : 0;  // exact (test): the families of a 9..16-row step in groups of 8
  const int* osess = exact_any() ? in_d->sess : nullptr;        // exact: no re-routing of one session's rows
  // the wide step's shared expert beside the gate and the router (side stream sb), whenever the wide MoE (moe_prefill) runs; not
  // with DSGB10_WIDE_EXACT's decode MoE (bit 8), which computes the shared expert itself
  const bool wsh = wide && !(wx & 8) && wide_sh();
  auto shared_side = [&] {
    fork(sb);
    dsk::MoePCfg sh;
    sh.phases = 2;
    dsk::moe_prefill(yq, ys, R, ridx, rwts, kExperts, w.moe, moe_pws, ymoe, sb, sh);
  };
  if (lx & 4) {  // exact: the fused gate of a step of at most 8 rows, in two groups
    for (int r0 = 0; r0 < R; r0 += dsk::kLowRows)
      dsk::gemv_bf16_norm(yraw + (size_t)r0 * kDim, w.ffn_norm, kNormEps, std::min<int>(dsk::kLowRows, R - r0), w.gate, kExperts,
                          kDim, gl + (size_t)r0 * kExperts, nullptr, cgate, st, yq + (size_t)r0 * kDim,
                          ys + (size_t)r0 * (kDim / 32));
  } else if (wx & 4) {  // exact: the gate's 16-row GEMV
    dsk::rmsnorm(yraw, w.ffn_norm, R, kDim, kNormEps, y, yq, ys, st, norm_nt(cgate, R));
    if (wsh) shared_side();
    if (wx & 64) {  // exact wide: one read of the gate
      dsk::gemv_bf16_rows(y, R, w.gate, kExperts, kDim, gl, cgate, st);
    } else {
      for (int r0 = 0; r0 < R; r0 += kDecodeRows)
        dsk::gemv_bf16(y + (size_t)r0 * kDim, std::min(kDecodeRows, R - r0), w.gate, kExperts, kDim, gl + (size_t)r0 * kExperts,
                       nullptr, cgate, false, st);
    }
  } else if (wide) {  // wide64: the gate as a GEMM, then router_topk and the grouped prefill MoE (one read of the union)
    dsk::rmsnorm(yraw, w.ffn_norm, R, kDim, kNormEps, y, yq, ys, st, norm_nt(cgate, R));
    if (wsh) shared_side();  // the shared expert beside the gate and the router
    dsk::gemm_bf16_p(y, R, w.gate, kExperts, kDim, gl, st);
  } else if (R > dsk::kLowRows && unfused16()) {  // night: the gate on the normed rows (y is free after the attention)
    dsk::rmsnorm(yraw, w.ffn_norm, R, kDim, kNormEps, y, yq, ys, st, norm_nt(cgate, R));
    dsk::gemv_bf16(y, R, w.gate, kExperts, kDim, gl, nullptr, cgate, false, st);
  } else {
    dsk::gemv_bf16_norm(yraw, w.ffn_norm, kNormEps, R, w.gate, kExperts, kDim, gl, nullptr, cgate, st, yq, ys);
  }
  if (!wide && gate_sep == 1 && R >= gate_sep_lo && R <= gate_sep_hi) k_gate_sep<<<R, 128, 0, st>>>(gl, nullptr);  // no PDL
  if (route_shift) route_rotate<<<R, 128, 0, st>>>(gl, route_shift, route_group);  // bench only
  if ((wx & 72) == 72) {  // exact wide: the decode MoE's sums over the whole step's union, every expert read once
    static const std::pair<int, int> oea = [] {  // the wide step's re-routing (lossy), as the plain wide body below
      int k0 = 0, kp = 0, mr = 0;
      if (const char* e = getenv("DSGB10_WIDE_OEA")) std::sscanf(e, "%d,%d", &k0, &kp);
      else if (const char* e2 = getenv("DSGB10_OEA")) std::sscanf(e2, "%d,%d,%d", &k0, &kp, &mr);
      return std::make_pair(k0, kp);
    }();
    if (oea.first > 0)
      dsk::router_topk_oea(gl, w.gate_bias, R, kExperts, ridx, rwts, kRouteScale, oea.first, oea.second, st, osess, &in_d->rows);
    else
      dsk::router_topk(gl, w.gate_bias, R, kExperts, ridx, rwts, kRouteScale, st);
    dsk::moe_rows(yq, ys, R, ridx, rwts, w.moe, moe_pws, ymoe, st, wide_gq());
  } else if ((wx | lx) & 8) {  // exact: the decode MoE (its own top-6) in groups of 16 (wide) or 8 rows: each group reads its experts
    if (ab.on && ab.s[L][1] != 0.f)
      die("DSGB10_WIDE_EXACT / DSGB10_R16_EXACT bit 8 without the row-group MoE (bit 64 with bit 8) keeps only the last group's "
          "shared-expert activations: the abliteration needs 95");
    const int kDecodeRows = wide ? model::kDecodeRows : dsk::kLowRows;  // the group
    for (int r0 = 0; r0 < R; r0 += kDecodeRows) {
      const int m = std::min(kDecodeRows, R - r0);
      dsk::MoeCfg mc{};
      mc.down = m == 1 ? 2 : 4;
      dsk::moe_routed(yq + (size_t)r0 * kDim, ys + (size_t)r0 * (kDim / 32), m, gl + (size_t)r0 * kExperts, w.gate_bias, kExperts,
                      kRouteScale, w.moe, moe_ws, ymoe + (size_t)r0 * kDim, mc, st, ridx + (size_t)r0 * kTopK,
                      rwts + (size_t)r0 * kTopK, osess, &in_d->rows, r0);
    }
  } else if (wide) {
    // wide64: DSGB10_WIDE_OEA=K0,KP (default: DSGB10_OEA's K0,KP when that is set) re-routes the wide step's rows
    // toward the batch's core experts (lossy, as DSGB10_OEA in moe_routed)
    static const std::pair<int, int> oea = [] {
      int k0 = 0, kp = 0, mr = 0;
      if (const char* e = getenv("DSGB10_WIDE_OEA")) std::sscanf(e, "%d,%d", &k0, &kp);
      else if (const char* e2 = getenv("DSGB10_OEA")) std::sscanf(e2, "%d,%d,%d", &k0, &kp, &mr);
      return std::make_pair(k0, kp);
    }();
    if (oea.first > 0)
      dsk::router_topk_oea(gl, w.gate_bias, R, kExperts, ridx, rwts, kRouteScale, oea.first, oea.second, st, osess, &in_d->rows);
    else
      dsk::router_topk(gl, w.gate_bias, R, kExperts, ridx, rwts, kRouteScale, st);
    if (wide_sh()) {  // bucketing + routed up, the shared expert's join, then down (+ combine)
      dsk::MoePCfg mc;
      mc.phases = 1 | 4;
      dsk::moe_prefill(yq, ys, R, ridx, rwts, kExperts, w.moe, moe_pws, ymoe, st, mc);
      join(sb);
      mc.phases = 8 | 16;
      dsk::moe_prefill(yq, ys, R, ridx, rwts, kExperts, w.moe, moe_pws, ymoe, st, mc);
    } else {
      dsk::moe_prefill(yq, ys, R, ridx, rwts, kExperts, w.moe, moe_pws, ymoe, st);
    }
  } else {
    dsk::MoeCfg mc{};
    mc.down = R == 1 ? 2 : 4;
    if (R == 1 && dsk::amoe_on()) mc.carveout = 100;  // DSK_AMOE: the all-reduce below is the other half
    dsk::moe_routed(yq, ys, R, gl, w.gate_bias, kExperts, kRouteScale, w.moe, moe_ws, ymoe, mc, st, ridx, rwts, osess, &in_d->rows);
  }
  if (ab.on && ab.s[L][1] != 0.f) {  // the shared expert's share of this rank's partial, through its activations
    const uint8_t *hq, *hs;
    if (wide && (wx & 72) == 72) {  // exact wide: the shared expert's activations moe_rows left
      dsk::moe_rows_shared(moe_pws, &hq, &hs);
    } else if (wide) {
      dsk::moe_prefill_shared(moe_pws, R, kExperts, &hq, &hs);
    } else {  // moe_routed's workspace: the shared expert is the slot after the union's (dsk.h moe_hs_offset)
      const int RB = dsk::row_budget(R);
      hq = static_cast<const uint8_t*>(moe_ws) + (size_t)RB * kTopK * RB * 576;
      hs = static_cast<const uint8_t*>(moe_ws) + dsk::moe_hs_offset(R) + (size_t)RB * kTopK * RB * 18;
    }
    ablit::shexp(ymoe, R, kDim, hq, 576, hs, 18, ab, L, in_d, st);
  }
  if (route_ring) k_route_copy<<<1, 128, 0, st>>>(ridx, in_d, L, route_ring);
  l2pf(1, L + 1, R);
  join(sa);  // the mixes
  // the experts' partial sums all-reduced, hc_post into the next block's input, and its collapse
  if (R > kDecodeRows && wide_ar()) {  // bulk all-reduce + the separate kernels (same bits)
    ops::allreduce_hc_bulk(ch, ymoe, true, wide_ar_mode(), R, mid, f_post, f_comb, next_pm, x, yraw, sub, ymoe, st);
  } else {
    ops::allreduce_hc(ch, ymoe, true, R, mid, f_post, f_comb, next_pm, x, yraw, dumps ? sub : nullptr, st,
                      R == 1 && dsk::amoe_on());
  }
  if (dumps) {
    dump(sink, L, "ffn_out", pos0, R, sub, kDim * 2);
    dump(sink, L, "route_idx", pos0, R, ridx, kTopK * 4);
    dump(sink, L, "route_w", pos0, R, rwts, kTopK * 4);
    dump(sink, L, "hc_ffn_pre", pos0, R, next_pm, kHc * 4);
    dump(sink, L, "hc_ffn_post", pos0, R, f_post, kHc * 4);
    dump(sink, L, "hc_ffn_comb", pos0, R, f_comb, kHc * kHc * 4);
    dump(sink, L, "out", pos0, R, x, kHcDim * 2);
    dump(sink, L, "pre_mix", pos0, R, next_pm, kHc * 4);
  }
}

// ---- the step -----------------------------------------------------------------------------------

ops::StepIn& Model::Impl::next_in(uint64_t n, int R, int pos0, const int32_t* tokens, const Sampling& sp) {
  if (R > kMaxSessRows)
    die("model: a step of %d rows of one session (at most %d: config.h kMaxSessRows; more rows need a batch of "
        "several sessions)", R, kMaxSessRows);
  ops::StepIn& in = in_h[n % ops::kInSlots];
  for (int r = 0; r < kMaxRows; ++r) {
    in.tokens[r] = r < R ? tokens[r] : 0;
    const int d = R - 1 - std::min(r, R - 1);  // rows after this one (bench: set_pos_stride)
    in.pos[r] = pos0 + R - 1 - d / pos_group * pos_stride - d % pos_group;
    in.slot[r] = cur_unit;
    in.sess[r] = 0;
  }
  for (ops::StepSamp& s : in.samp) s = samp_of(sp);
  in.rows = R;
  in.nsess = 1;
  in.step = (uint32_t)n;
  in.sbatch = 0;
  return in;
}

uint64_t Model::Impl::launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, uint32_t flags, Sink* sink) {
  if (!bound) die("model: step before bind");
  const int pos_last = pos0 + R - 1;
  if (R < 1 || R > kMaxRows || pos0 < 0 || pos_last >= slot_ctx_[cur_slot] ||
      pos_last - (R - 1) / pos_group * pos_stride - (R - 1) % pos_group < 0)
    die("model: bad step R=%d pos0=%d", R, pos0);
  ensure_mapped(cur_slot, pos_last);
  sf = slot_fac[cur_slot];
  if (opt.first_layer > 0 && !(flags & kFlagInject) && (flags & (kFlagDump | kFlagLogits)))
    die("model: starting at layer %d needs golden injection", opt.first_layer);
  bool device_tokens = false;
  for (int r = 0; r < R; ++r) device_tokens |= tokens[r] < 0;
  if (device_tokens && (flags & ~kFlagProfile)) die("model: golden checks need the step's tokens");
  const uint64_t n = n_steps;
  if (n >= (uint64_t)ops::kInSlots) wait_done(n - ops::kInSlots);  // its StepIn slot has been read
  ops::StepIn& in = next_in(n, R, pos0, tokens, sp);
  if (store) {
    const uint64_t seq[2] = {e_seq + 1, e_seq + 2};
    e_seq += 2;
    in.e_wait[0] = seq[0], in.e_wait[1] = seq[1];
    if (device_tokens) {
      if (late.on) die("model: two steps wait for their tokens");
      late.on = true, late.any = false, late.batch = false;
      late.pos0 = pos0, late.R = R, late.seq[0] = seq[0], late.seq[1] = seq[1];
    } else {
      if (late.on) die("model: a step waits for its tokens");
      if (n) wait_done(n - 1);  // the previous step has read the Engram buffers
      hist_put(hist, pos0, R, tokens);
      engram_prepare(pos0, R, seq);
    }
  }
  return run_step(n, R, scan_ctx(pos_last), flags, sink, pos0);
}

uint64_t Model::Impl::launch_batch(const Batch& b) {
  if (!bound) die("model: step before bind");
  const int R = b.R;
  if (R < 1 || R > kMaxRows || b.nsess < 1 || b.nsess > ops::kMaxSess || b.sess[0] != 0 || b.sess[R - 1] != b.nsess - 1)
    die("model: a batch step of %d rows, %d sessions", R, b.nsess);
  if (opt.first_layer > 0 && !(b.flags & kFlagInject) && (b.flags & (kFlagDump | kFlagLogits)))
    die("model: starting at layer %d needs golden injection", opt.first_layer);
  bool device_tokens = false;
  int pos_last = 0;
  for (int r = 0; r < R; ++r) {
    const bool same = r > 0 && b.sess[r] == b.sess[r - 1];  // the session of the row before
    bool ok = b.slot[r] >= 0 && b.slot[r] < nslots && b.pos[r] >= 0 && b.pos[r] < slot_ctx_[b.slot[r]];
    if (r > 0 && !same) ok &= b.sess[r] == b.sess[r - 1] + 1;
    if (same) ok &= b.slot[r] == b.slot[r - 1] && b.pos[r] == b.pos[r - 1] + 1;
    int run = 1;  // the session's rows so far: at most kMaxSessRows (config.h: attn's race past it)
    while (run <= r && b.sess[r - run] == b.sess[r]) ++run;
    // test only (the tool's teacher-forced batch mode with DSGB10_TF_SESSROWS set): a session may hold more rows in a step
    static const bool tf_sessrows = [] {
      const bool on = getenv("DSGB10_TF_SESSROWS") != nullptr;
      if (on) std::fprintf(stderr, "batch steps: a session may hold up to %d rows of a step (DSGB10_TF_SESSROWS, test only)\n", kMaxRows);
      return on;
    }();
    if (run > kMaxSessRows && !tf_sessrows)
      die("model: a batch step with %d rows of one session (at most %d: config.h kMaxSessRows)", run, kMaxSessRows);
    for (int q = 0; q < r; ++q) ok &= b.sess[q] == b.sess[r] || b.slot[q] != b.slot[r];  // a slot each
    if (!ok) die("model: batch row %d: position %d, slot %d, session %d", r, b.pos[r], b.slot[r], b.sess[r]);
    device_tokens |= b.tokens[r] < 0;
    pos_last = std::max(pos_last, b.pos[r]);
  }
  for (int r = 0; r < R; ++r) ensure_mapped(b.slot[r], b.pos[r]);  // every row's slot, before any work is queued
  sf = factor_of_slots(b.slot, R);  // for this step's launch or capture; use_slot / launch put the current slot's back
  if (device_tokens && (b.flags & ~kFlagProfile)) die("model: golden checks need the batch step's tokens");
  const uint64_t n = n_steps;
  if (n >= (uint64_t)ops::kInSlots) wait_done(n - ops::kInSlots);  // its StepIn slot has been read
  ops::StepIn& in = in_h[n % ops::kInSlots];
  for (int r = 0; r < kMaxRows; ++r) {  // the entries past R repeat the last row's
    const int q = std::min(r, R - 1);
    in.tokens[r] = r < R ? b.tokens[r] : 0;
    in.pos[r] = b.pos[q], in.slot[r] = unit_of[b.slot[q]], in.sess[r] = b.sess[q];
  }
  for (int k = 0; k < ops::kMaxSess; ++k) in.samp[k] = samp_of(b.sp[std::min(k, b.nsess - 1)]);
  in.rows = R;
  in.nsess = b.nsess;
  in.step = (uint32_t)n;
  in.sbatch = 0;
  if (store) {
    const uint64_t seq[2] = {e_seq + 1, e_seq + 2};
    e_seq += 2;
    in.e_wait[0] = seq[0], in.e_wait[1] = seq[1];
    if (late.on) die("model: a step waits for its tokens");
    if (device_tokens) {
      late.on = true, late.any = false, late.batch = true;
      late.pos0 = b.pos[0], late.R = R, late.seq[0] = seq[0], late.seq[1] = seq[1];
      std::copy(b.pos, b.pos + kMaxRows, late.pos);
      std::copy(in.slot, in.slot + kMaxRows, late.slot);  // units
    } else {
      if (n) wait_done(n - 1);  // the previous step has read the Engram buffers
      engram_rows(R, b.tokens, b.pos, in.slot, seq);
    }
  }
  return run_step(n, R, scan_ctx(pos_last), b.flags, b.sink, b.pos[0]);
}

int Model::Impl::factor_of_slots(const int32_t* slots, int n) const {
  const int f = slot_fac[slots[0]];
  for (int r = 1; r < n; ++r)
    if (slot_fac[slots[r]] != f)
      die("model: a step mixes YaRN factors (slot %d: %d, slot %d: %d); the server never batches them", slots[0], f,
          slots[r], slot_fac[slots[r]]);
  return f;
}

uint64_t Model::Impl::run_step(uint64_t n, int R, int ctx, uint32_t flags, Sink* sink, int pos0) {
  // Golden checks run eagerly. Rank 0's dumps add no collectives, so the other ranks may still replay.
  uint32_t eager = flags & (kFlagInject | kFlagInjectSub | kFlagLogits);
  if (rank == 0 && sink) eager |= flags & kFlagDump;
  const int slot = (int)(n % ops::kOutSlots);
  CUDA_CHECK(cudaEventRecord(ev_beg[slot], st));
  if (opt.graphs && !eager && warm[R]) {
    CUDA_CHECK(cudaGraphLaunch(graph(R, ctx), st));
  } else {
    forward(R, ctx, flags, sink, pos0);
    warm[R] = true;  // the kernels' one-time setup (smem attributes) has run outside a capture
  }
  CUDA_CHECK(cudaEventRecord(ev_end[slot], st));
  n_steps = n + 1;
  return n;
}

void Model::Impl::engram_late(int pos0, int R, const int32_t* tokens) {
  if (!store) return;
  if (!late.on || late.batch || (!late.any && (late.pos0 != pos0 || late.R != R)))
    die("model: engram_late(%d, %d) for a step at %d, %d rows", pos0, R, late.on ? late.pos0 : -1, late.R);
  if (R < 1 || R > kMaxRows) die("model: engram_late(%d, %d)", pos0, R);
  if (n_steps >= 2) wait_done(n_steps - 2);  // the step before it has read the Engram buffers
  hist_put(hist, pos0, R, tokens);
  engram_prepare(pos0, R, late.seq, spec && spec->xlong && R > kDecodeRows ? ops::spec_body_rows(R) : 0);  // a long copy step: its body's rows
  late.on = false;
}

void Model::Impl::engram_late_batch(const int32_t* tokens) {
  if (!store) return;
  if (!late.on || !late.batch || late.any) die("model: engram_late_batch with no batch step waiting for its tokens");
  if (n_steps >= 2) wait_done(n_steps - 2);  // the step before it has read the Engram buffers
  engram_rows(late.R, tokens, late.pos, late.slot, late.seq);
  late.on = false;
}

// B3: a batch verify step's rows, as the device lays them out (spec.cu k_spec_in_batch), from the host's
// copy of the step before's choices
void Model::Impl::engram_late_rows(int R, const int32_t* tok, const int32_t* pos, const int32_t* slot) {
  if (!store) return;
  if (!late.on || !late.batch || !late.any) die("model: engram_late_rows with no batch verify step waiting for its tokens");
  if (R < 1 || R > kMaxRows) die("model: engram_late_rows(%d)", R);
  if (n_steps >= 2) wait_done(n_steps - 2);  // the step before it has read the Engram buffers
  int32_t u[kMaxRows];  // the server's slots -> units
  for (int r = 0; r < R; ++r) u[r] = unit_of[slot[r]];
  engram_rows(R, tok, pos, u, late.seq, ops::spec_body_rows(R));  // wide64: the bucket's rows
  late.on = false;
}

const ops::SampleOut& Model::Impl::wait(uint64_t n) {
  if (n >= n_steps) die("model: wait for step %llu, %llu launched", (unsigned long long)n, (unsigned long long)n_steps);
  const ops::SampleOut& o = out_h[n % ops::kOutSlots];
  const uint32_t want = (uint32_t)n + 1;
  for (;;) {
    const uint32_t s = __atomic_load_n(&o.seq, __ATOMIC_ACQUIRE);
    if (s == want) break;
    if ((int32_t)(s - want) > 0) die("model: the sample of step %llu was overwritten", (unsigned long long)n);
  }
  n_done = std::max(n_done, n + 1);
  last_n = n;
  if (route_f) {
    RouteRec r;
    CUDA_CHECK(cudaMemcpy(&r, route_ring + n % kRouteRing, sizeof r, cudaMemcpyDeviceToHost));
    if (r.step == (uint32_t)n) fwrite(&r, sizeof r, 1, route_f), fflush(route_f);
  }
  return o;
}

cudaGraphExec_t Model::Impl::graph(int R, int ctx) {
  const auto key = std::make_tuple(R, ctx, sf);  // the captured kernels' parameters bake the factor
  const auto it = graphs.find(key);
  if (it != graphs.end()) return it->second;
  // a plain step graph of a factor above 16 is its window's (win_graphs_drop frees it at the close): the window's class; a graph
  // of factor 16 keeps the scope the caller is in
  std::optional<alloc_seal::Scoped> seal;
  if (sf > 16) seal.emplace(alloc_seal::kRegion);
  cudaGraph_t g;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  forward(R, ctx, 0, nullptr, 0);
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  cudaGraphExec_t e;
  CUDA_CHECK(cudaGraphInstantiate(&e, g, cudaGraphInstantiateFlagUseNodePriority));
  CUDA_CHECK(cudaGraphDestroy(g));
  graphs.emplace(key, e);
  return e;
}

void Model::Impl::forward(int R, int ctx, uint32_t flags, Sink* sink, int pos0) {
  ops::step_in(in_h, in_ctr, fb, in_d, st);
  // test only (DSGB10_TF_SOLO): a given step of more than kDecodeRows rows runs as the exact bodies of the one-session step graph
  // do, eager or captured alike (a row count's first step runs eagerly, outside graph())
  const bool solo = tf_solo() && R > kDecodeRows;
  if (solo) g_xbody = true;
  forward_rows(R, ctx, flags, sink, pos0);
  if (solo) g_xbody = false;
  // the sampler's grid by the row budget: a step of at most 8 rows launches what it did before B4
  ops::sample(ch, logits, kVocabPart * rank, in_d, ctl_h, out_h, fb, samp_ctr, st, nullptr, ops::row_budget(R));
}

// ---- stack4: the certified head ------------------------------------------------------------------------------------
namespace {
constexpr int kCertRowsCta = 16;                       // the head GEMV's output rows per CTA (chead: mt 2, wn 1)
constexpr int kCertCtas = kVocabPart / kCertRowsCta;
constexpr int kCertRows = 8;                           // step rows of a certified body, at most (argmax_rows: an even count)
static_assert(kVocabPart % kCertRowsCta == 0 && kMaxSessRows <= kCertRows, "cert");

// A sampled row (Cert::samp: a temperature up to 1, no top-k or top-p) takes the Gumbel-max winner of l / T - log(-log(u)), u
// hashed from the session's seed, the row's position and the id (ops::sample, sample.cu). That score is monotone in l, so
// the logit bounds give bounds of it, each side with another 1e-3 of slack for the roundings. A greedy row keeps the
// logit bounds.
struct CertRow {
  bool samp;
  float invT;
  uint64_t base;
};
__device__ __forceinline__ uint64_t cert_mix64(uint64_t x) {
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 32;
  return x;
}
__device__ __forceinline__ CertRow cert_row(const ops::StepIn* in, int b) {
  const ops::StepSamp sp = in->samp[min(max(in->sess[b], 0), ops::kMaxSess - 1)];
  CertRow r;
  r.samp = sp.temperature > 0.f;
  r.invT = r.samp ? 1.f / sp.temperature : 1.f;
  r.base = cert_mix64(sp.seed) ^ ((uint64_t)(uint32_t)in->pos[b] * 0x9E3779B97F4A7C15ull);
  return r;
}
__device__ __forceinline__ float cert_noise(uint64_t base, int id) {  // log(-log(u)), as the sampler takes it
  const uint64_t h = cert_mix64(base + (uint64_t)id * 0xD1B54A32D192ED03ull);
  return logf(-logf(ops::unit(h)));  // the sampler's uniform (samp_rule.h), strictly inside (0, 1)
}

// Row b = blockIdx.x: xn[b] = an upper bound of ||x_b||_2 (x the normed final hidden); cand[b] = this rank's largest
// lower bound of a logit, coarse - xn B (B[v]: the row's residual norm plus the rounding slack of both kernels), or of
// a sampled row's score. Block 0 clears the CTA mask and counts the step.
__global__ void __launch_bounds__(1024) k_cert_lo(const float* C, const bf16* x, const float* B, int R, int vocab0,
                                                  float* xn, comm::ArgMax* cand, uint8_t* mask, uint32_t* stats,
                                                  float bscale, const ops::StepIn* in) {
  __shared__ float sv[1024];
  __shared__ int si[1024];
  const int b = blockIdx.x, t = threadIdx.x;
  if (b == 0) {
    for (int i = t; i < kCertCtas; i += 1024) mask[i] = 0;
#ifdef DSGB10_DEV  // the step count DSGB10_CERT_FORCE reads (development builds only)
    if (t == 0) stats[0] += 1;
#else
    (void)stats;
#endif
  }
  if (b >= R) {
    if (t == 0) cand[b] = comm::ArgMax{-INFINITY, INT_MAX};
    return;
  }
  float s = 0.f;
  for (int i = t; i < kDim; i += 1024) {
    const float v = __bfloat162float(x[(size_t)b * kDim + i]);
    s = fmaf(v, v, s);
  }
  sv[t] = s;
  __syncthreads();
  if (t == 0) {
    float a = 0.f;
    for (int i = 0; i < 1024; ++i) a += sv[i];
    si[0] = __float_as_int(sqrtf(a) * 1.001f * bscale);
    xn[b] = __int_as_float(si[0]);
  }
  __syncthreads();
  const float n = __int_as_float(si[0]);
  __syncthreads();
  const float* c = C + (size_t)b * kVocabPart;
  const CertRow cr = cert_row(in, b);
  float m = -INFINITY;
  int mi = INT_MAX;
  for (int i = t; i < kVocabPart; i += 1024) {
    float lo = c[i] - n * B[i];
    if (cr.samp) lo = (lo - 1e-3f) * cr.invT - cert_noise(cr.base, vocab0 + i);
    if (lo > m) m = lo, mi = vocab0 + i;
  }
  sv[t] = m, si[t] = mi;
  __syncthreads();
  if (t == 0) {
    for (int i = 1; i < 1024; ++i)
      if (sv[i] > m || (sv[i] == m && si[i] < mi)) m = sv[i], mi = si[i];
    cand[b] = comm::ArgMax{m - 1e-3f, mi};
  }
}
// Row b: with g[b].v the largest lower bound over the ranks, a vocabulary row whose upper bound coarse + xn B (a
// sampled row: its score's) reaches it may hold the winner: its CTA's mask byte is set; every logit of the row becomes
// -inf and the masked exact GEMV writes the surviving CTAs'. A forced step, or a non-finite threshold or norm, sets every CTA.
__global__ void __launch_bounds__(1024) k_cert_mask(const float* C, const float* B, const float* xn,
                                                    const comm::ArgMax* g, int R, float* logits, uint8_t* mask,
                                                    uint32_t* stats, int force, const ops::StepIn* in, int vocab0) {
  const int b = blockIdx.x, t = threadIdx.x;
  if (b >= R) return;
  const float T = g[b].v, n = xn[b];
#ifdef DSGB10_DEV  // DSGB10_CERT_FORCE: every Nth step runs every CTA (development builds only)
  const bool forced = force > 0 && stats[0] % (uint32_t)force == 0;
  if (b == 0 && t == 0 && forced) stats[6] += 1;
#else
  const bool forced = false;
  (void)stats, (void)force;
#endif
  const bool all = forced || !(fabsf(T) < INFINITY) || !(n < INFINITY);
  const float* c = C + (size_t)b * kVocabPart;
  float* l = logits + (size_t)b * kVocabPart;
  const CertRow cr = cert_row(in, b);
  for (int i = t; i < kVocabPart; i += 1024) {
    l[i] = -INFINITY;
    float hi = c[i] + n * B[i] + 1e-3f;
    if (cr.samp) hi = hi * cr.invT - cert_noise(cr.base, vocab0 + i) + 1e-3f;
    if (all || !(hi < T)) mask[i / kCertRowsCta] = 1;
  }
}
#ifdef DSGB10_DEV  // the certified head's check against the full head (development builds only)
// DSGB10_CERT_CHECK: row b's maximum and its lowest id in the certified logits against the full head's (l2). The token
// is the best (max, lowest id) over the ranks, so the certified step gives the full head's token when on every rank
// the two agree, or the full head's maximum of this rank is below the threshold g[b].v (a lower bound of some logit, so
// this rank does not hold the winner under either head). Anything else counts in stats[2]; stats[7] counts the rows
// where this rank's two maxima agree. Also the largest |coarse - exact| / (xn B) (stats[4], float bits), the CTAs set.
__global__ void __launch_bounds__(1024) k_cert_check(const float* logits, const float* l2, const float* C, const float* B,
                                                     const float* xn, int R, const uint8_t* mask, uint32_t* stats,
                                                     const comm::ArgMax* g, const ops::StepIn* in, int vocab0) {
  __shared__ float sv[1024], sw[1024], sr[1024];
  __shared__ int si[1024], sj[1024];
  const int b = blockIdx.x, t = threadIdx.x;
  if (b == 0) {
    int cnt = 0;
    for (int i = t; i < kCertCtas; i += 1024) cnt += mask[i] != 0;
    si[t] = cnt;
    __syncthreads();
    if (t == 0) {
      int tot = 0;
      for (int i = 0; i < 1024; ++i) tot += si[i];
      stats[3] += (uint32_t)tot;
      if ((uint32_t)tot > stats[5]) stats[5] = (uint32_t)tot;
    }
    __syncthreads();
  }
  if (b >= R) return;
  const float* a = logits + (size_t)b * kVocabPart;
  const float* e = l2 + (size_t)b * kVocabPart;
  const float* c = C + (size_t)b * kVocabPart;
  const float n = xn[b];
  const CertRow cr = cert_row(in, b);  // a sampled row: the scores' winners compared
  float m1 = -INFINITY, m2 = -INFINITY, rt = 0.f;
  int i1 = INT_MAX, i2 = INT_MAX;
  for (int i = t; i < kVocabPart; i += 1024) {
    float av = a[i], ev = e[i];
    if (cr.samp) {
      const float nz = cert_noise(cr.base, vocab0 + i);
      av = av * cr.invT - nz, ev = ev * cr.invT - nz;
    }
    if (av > m1) m1 = av, i1 = i;
    if (ev > m2) m2 = ev, i2 = i;
    rt = fmaxf(rt, fabsf(c[i] - e[i]) / (n * B[i]));
  }
  sv[t] = m1, si[t] = i1, sw[t] = m2, sj[t] = i2, sr[t] = rt;
  __syncthreads();
  if (t == 0) {
    for (int i = 1; i < 1024; ++i) {
      if (sv[i] > m1 || (sv[i] == m1 && si[i] < i1)) m1 = sv[i], i1 = si[i];
      if (sw[i] > m2 || (sw[i] == m2 && sj[i] < i2)) m2 = sw[i], i2 = sj[i];
      rt = fmaxf(rt, sr[i]);
    }
    atomicAdd(stats + 1, 1u);
    const bool same = __float_as_uint(m1) == __float_as_uint(m2) && i1 == i2;
    if (same) atomicAdd(stats + 7, 1u);
    else if (!(m2 < g[b].v)) atomicAdd(stats + 2, 1u);
    atomicMax(stats + 4, __float_as_uint(rt));
  }
}
#endif
}  // namespace

void Model::Impl::cert_head(int R) {
  if (R > kCertRows) die("model: a certified body of %d rows", R);
  dsk::rmsnorm(yraw, norm_w, R, kDim, kNormEps, hfin, nullptr, nullptr, st);
  dsk::gemv_fp8w_f32(hfin, R, cert.h8, cert.h8s, kVocabPart, kDim, cert.clog, dsk::GemvCfg{2, 1, 8}, st);
  attn::launch(k_cert_lo, dim3(kCertRows), dim3(1024), 0, st, false, (const float*)cert.clog, (const bf16*)hfin,
               (const float*)cert.B, R, kVocabPart * rank, cert.xn, cert.cand, cert.mask, cert.stats, cert.bscale,
               (const ops::StepIn*)in_d);
  comm::argmax_rows(ch, cert.cand, cert.gcand, kCertRows, st);  // COLLECTIVE: each row's threshold over the ranks
  attn::launch(k_cert_mask, dim3(kCertRows), dim3(1024), 0, st, false, (const float*)cert.clog, (const float*)cert.B,
               (const float*)cert.xn, (const comm::ArgMax*)cert.gcand, R, logits, cert.mask, cert.stats, cert.force,
               (const ops::StepIn*)in_d, kVocabPart * rank);
  dsk::gemv_bf16x_norm_masked(yraw, norm_w, kNormEps, R, head, kVocabPart, kDim, logits, chead, cert.mask, st);
#ifdef DSGB10_DEV
  if (cert.check) {
    dsk::gemv_bf16x_norm(yraw, norm_w, kNormEps, R, head, kVocabPart, kDim, cert.logits2, nullptr, chead, st);
    attn::launch(k_cert_check, dim3(kCertRows), dim3(1024), 0, st, false, (const float*)logits, (const float*)cert.logits2,
                 (const float*)cert.clog, (const float*)cert.B, (const float*)cert.xn, R, (const uint8_t*)cert.mask,
                 cert.stats, (const comm::ArgMax*)cert.gcand, (const ops::StepIn*)in_d, kVocabPart * rank);
  }
#endif
}

// The bound of every vocabulary row of this rank's slice, from the raw bf16 head and its FP8 copy (bind_spec, before the
// head is coded): B[v] = 1.001 (||w_v - w8_v|| + K 2^-21 (||w_v|| + ||w8_v||)). Then |exact kernel's logit - FP8 kernel's|
// <= ||x|| B[v]: Cauchy-Schwarz on the residual, and each kernel's fp32 sums within K 2^-21 sum |w x| of the true dot.
void Model::Impl::cert_setup(const uint8_t* head_raw, const uint8_t* q, const uint8_t* s) {
  const char* e = getenv("DSGB10_CERT");
  if (!e || atoi(e) == 0) return;
  if (getenv("DSGB10_RELAX") || getenv("DSGB10_RELAX_P")) die("model: DSGB10_CERT with a relaxed accept");
  double* acc = dalloc<double>((size_t)3 * kVocabPart);
  CUDA_CHECK(cudaMemsetAsync(acc, 0, (size_t)3 * kVocabPart * 8, st));
  dspark::head_fp8_bound(head_raw, q, s, acc, st);
  sync();
  std::vector<double> h((size_t)3 * kVocabPart);
  CUDA_CHECK(cudaMemcpy(h.data(), acc, h.size() * 8, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(acc));
  std::vector<float> B(kVocabPart);
  double sum = 0, mx = 0;
  for (int v = 0; v < kVocabPart; ++v) {
    const double b = 1.001 * (std::sqrt(h[3 * v]) + kDim * 0x1p-21 * (std::sqrt(h[3 * v + 1]) + std::sqrt(h[3 * v + 2])));
    B[v] = std::nextafter((float)b, INFINITY);
    sum += b, mx = std::max(mx, b);
  }
  cert.B = dalloc<float>(kVocabPart);
  CUDA_CHECK(cudaMemcpy(cert.B, B.data(), (size_t)kVocabPart * 4, cudaMemcpyHostToDevice));
  cert.h8 = q, cert.h8s = s;
  cert.clog = dalloc<float>((size_t)kCertRows * kVocabPart);
  cert.xn = dalloc<float>(kCertRows);
  cert.cand = dalloc<comm::ArgMax>(kCertRows);
  cert.gcand = dalloc<comm::ArgMax>(kCertRows);
  cert.mask = dalloc<uint8_t>(kCertCtas);
  cert.stats = dalloc<uint32_t>(8);
  CUDA_CHECK(cudaMemset(cert.stats, 0, 32));
  CUDA_CHECK(cudaMemset(cert.mask, 0, kCertCtas));
#ifdef DSGB10_DEV  // the check and the forced steps: development builds only
  cert.check = getenv("DSGB10_CERT_CHECK") && atoi(getenv("DSGB10_CERT_CHECK")) != 0;
  if (cert.check) cert.logits2 = dalloc<float>((size_t)kCertRows * kVocabPart);
  cert.force = getenv("DSGB10_CERT_FORCE") ? atoi(getenv("DSGB10_CERT_FORCE")) : 0;
#endif
  cert.bscale = getenv("DSGB10_CERT_BSCALE") ? (float)atof(getenv("DSGB10_CERT_BSCALE")) : 1.f;
  cert.samp = !getenv("DSGB10_CERT_SAMPLED") || atoi(getenv("DSGB10_CERT_SAMPLED")) != 0;
  if (!(cert.bscale >= 1.f)) die("model: DSGB10_CERT_BSCALE below 1");
  cert.on = true;
  std::fprintf(stderr, "cert: rank %d: certified head on (FP8 coarse read; bound mean %.4f max %.4f; sampled rows %d; check %d "
                       "force %d bscale %g)\n",
               rank, sum / kVocabPart, mx, (int)cert.samp, (int)cert.check, cert.force, cert.bscale);
}

void Model::Impl::forward_rows(int R, int ctx, uint32_t flags, Sink* sink, int pos0) {
  const bool lead = rank == 0 && sink;
  const bool inj = flags & kFlagInject, sub_in = flags & kFlagInjectSub, dumps = (flags & kFlagDump) && lead;
  // the embedding: each rank's rows of the table, summed over the ranks (bf16 -0.0 where a rank holds no row)
  ops::embed_part(embed, rank, tok_d, R, emb_slot, st);
  if ((size_t)R * kDim * 2 <= ch.cap) comm::allreduce_bf16(ch, emb_slot, emb_slot, (size_t)R * kDim, st);
  else comm::allreduce_bulk_bf16(ch, emb_slot, emb_slot, (size_t)R * kDim, st);
  ops::embed_fan(emb_slot, R, x, pm[0], st);
  // Each sublayer starts from yraw, its input collapsed by the pre mix but not yet normed: the
  // all-reduce kernel ending the previous sublayer makes it; here only where the stream changed since.
  for (int L = opt.first_layer; L < opt.n_layers; ++L) {
    bool fresh = L == opt.first_layer;
    if (inj && L > 0) {  // golden block inputs
      inject(sink, L, "x", pos0, R, x, kHcDim * 2);
      inject(sink, L, "pre_mix", pos0, R, pm[L & 1], kHc * 4);
      fresh = true;
    }
    if (engram_layer(L) && store) {
      engram_apply(L, R);
      if (dumps) dump(sink, L, "engram_out", pos0, R, x, kHcDim * 2);
      fresh = true;
    }
    if (mh_on && L >= kSpecLayer0) spec_hidden(L, R);  // DSpark's main_hidden: the layer's input stream
    if (fresh) attn::hc_pre(U(x), pm[L & 1], R, nullptr, U(yraw), nullptr, st);
    attention(L, R, ctx, dumps, sub_in, sink, pos0);
    ffn(L, R, dumps, sink, pos0);
  }
  if (sc) join(sc);
  // final norm (fused into the vocab-split head; yraw is the final collapse) and the sampling
  if (dumps) {
    dsk::rmsnorm(yraw, norm_w, R, kDim, kNormEps, hfin, nullptr, nullptr, st);
    dump(sink, -1, "final_pre_norm", pos0, R, yraw, kDim * 2);
    dump(sink, -1, "final_hidden", pos0, R, hfin, kDim * 2);
  }
  // wide64: DSGB10_HEADG (default on; 0: off) runs a wide step's head in one read of the weights (the coded-head
  // GEMM, dsk::gemv_bf16_norm_rows: the same bits as the 16-row groups below)
  static const bool headg = !getenv("DSGB10_HEADG") || atoi(getenv("DSGB10_HEADG")) != 0;
  if (cert_cap) cert_head(R);  // stack4: a certified body of the one-session spec graph
  else if (R > kDecodeRows && headg) {
    // DSGB10_HEADG=2: the rows are normalised once into hfin (the dumps' final_hidden otherwise) and the head
    // kernel loads them
    static const bool headpre = [] {
      const bool on = getenv("DSGB10_HEADG") && atoi(getenv("DSGB10_HEADG")) >= 2;
      if (on) std::fprintf(stderr, "wide steps: the head on rows normalised once (DSGB10_HEADG=2)\n");
      return on;
    }();
    dsk::gemv_bf16_norm_rows(yraw, norm_w, kNormEps, R, head, head_coded, kVocabPart, kDim, logits, chead, st,
                             headpre && !dumps ? hfin : nullptr);
  } else
  // wide64: the head GEMVs in groups of kDecodeRows rows (exact, DSGB10_R16_EXACT bit 32: a 9..16-row step in groups of 8)
  for (int r0 = 0, hg = R > dsk::kLowRows && R <= kDecodeRows && (r16_exact() & 32) ? dsk::kLowRows : kDecodeRows; r0 < R; r0 += hg) {
    const int m = std::min(hg, R - r0);
    const bf16* hin = yraw + (size_t)r0 * kDim;
    float* lo = logits + (size_t)r0 * kVocabPart;
    if (head_coded) dsk::gemv_bf16x_norm(hin, norm_w, kNormEps, m, head, kVocabPart, kDim, lo, nullptr, chead, st);
    else dsk::gemv_bf16_norm(hin, norm_w, kNormEps, m, head, kVocabPart, kDim, lo, nullptr, chead, st);
  }
  if (flags & kFlagLogits) {
    comm::allgather(ch, logits, lgath, (size_t)R * kVocabPart * 4, st);
    if (lead) {
      sync();
      std::vector<float> g((size_t)kRanks * R * kVocabPart), full((size_t)R * kVocab);
      CUDA_CHECK(cudaMemcpy(g.data(), lgath, g.size() * 4, cudaMemcpyDeviceToHost));
      for (int r = 0; r < kRanks; ++r)
        for (int p = 0; p < R; ++p)
          std::memcpy(&full[(size_t)p * kVocab + (size_t)r * kVocabPart], &g[((size_t)r * R + p) * kVocabPart],
                      kVocabPart * 4);
      sink->logits(pos0, R, full.data());
    }
  }
}

// ---- Model ----------------------------------------------------------------------------------------

Model::Model(int rank, const Options& opt) : d_(std::make_unique<Impl>(rank, opt)) { store_trace::init(); }
Model::~Model() = default;
void Model::bind(dsw::WeightFile& wf, comm::Ring& ring) { d_->bind(wf, ring); }
void Model::reserve_gathered(bool engram) { d_->reserve_gathered(engram); }
void* Model::gathered_dest(const std::string& name, uint64_t bytes) { return d_->gathered_dest(name, bytes); }
void Model::aliased(size_t* tensors, size_t* bytes) const { *tensors = d_->alias_n, *bytes = d_->alias_bytes; }
void Model::bind_ring2(comm::Ring& ring) { d_->ch2 = ring.dev(), d_->has_ch2 = true; }
size_t Model::code_head(int* raw_tiles) { return d_->code_head(raw_tiles); }
void Model::use_slot(int s) { d_->use_slot(s); }
void Model::set_slot_factor(int s, int factor) {
  if (s < 0 || s >= d_->nslots) die("model: set_slot_factor: slot %d of %d", s, d_->nslots);
  if (factor != 16 && factor != 32 && factor != 64) die("model: set_slot_factor: YaRN factor %d (16, 32 or 64)", factor);
  if (d_->lazy_win && !d_->scan_[Impl::level_of(factor)].open)
    die("model: set_slot_factor: factor %d's window is not open (window_open comes first)", factor);
  d_->slot_fac[s] = factor;
  if (s == d_->cur_slot) d_->sf = factor;
}
int Model::slot_factor(int s) const {
  if (s < 0 || s >= d_->nslots) die("model: slot_factor: slot %d of %d", s, d_->nslots);
  return d_->slot_fac[s];
}
int Model::slot() const { return d_->cur_slot; }
int Model::slots() const { return d_->nslots; }
int Model::slot_ctx(int s) const { return d_->slot_ctx_[s]; }
int Model::slot_chunks_for(int s, int need) const { return d_->slot_chunks_for(s, need); }
// what the line's request holds or has reserved, whichever is more; a conversation split on the line counts the whole numbers its
// slots stand for (4 slots = 1), so that an idle line whose conversation is fully split still has something to give (reclaim.h)
int Model::slot_committed(int s) const {
  return d_->calloc ? rules::line_committed(d_->core->resv[s], d_->held(s), d_->core->held_slots(s)) : 0;
}
// the numbers the line is reserved beyond what it holds (they end with its detach)
int Model::slot_unheld_reservation(int s) const { return d_->calloc ? std::max(0, d_->core->resv[s] - d_->held(s)) : 0; }
int Model::slot_mapped(int s) const { return d_->calloc ? d_->held(s) : 0; }
// free numbers no running request has been reserved (its reservation past what its session holds), signed (rules::spare): below 0
// where the reservations are backed by stored conversations; what evicting the sessions no line holds would give back is the server's
// to add (state_giveable)
int64_t Model::slot_spare() const {
  if (!d_->calloc) return 0;
  int64_t reserved = 0;
  for (int s = 0; s < d_->nslots; ++s) reserved += std::max(0, d_->core->resv[s] - d_->held(s)) + d_->core->resv_img[s];
  return rules::spare((int64_t)d_->calloc->free(), reserved);
}
// the same held at 0 (the log lines')
int Model::slot_uncommitted() const { return (int)std::max<int64_t>(0, slot_spare()); }
void Model::slot_attach(int s, int h, int upto) { d_->slot_attach(s, h, upto); }
int Model::slot_chunks_total() const { return d_->calloc ? (int)d_->calloc->total() : 0; }
int Model::slot_chunks_peak() const { return (int)d_->core->held_peak; }
void Model::slot_reserve(int s, int need, int img) { d_->slot_reserve(s, need, img); }
void Model::slot_release(int s) { d_->slot_release(s); }
void Model::set_engram(const engram::HashMap* map, engram::Store* store) {
  d_->emap = map;
  d_->store = store;
  d_->el.enabled = store && store->cache_max_bytes() > 0;
  if (d_->el.enabled) d_->tenants.add(lend::kTenantEngram, lend::kKindHost, lend::kCache);
}
uint64_t Model::launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, uint32_t flags, Sink* sink) {
  return d_->launch(R, pos0, tokens, sp, flags, sink);
}
void Model::engram_late(int pos0, int R, const int32_t* tokens) { d_->engram_late(pos0, R, tokens); }
uint64_t Model::launch_batch(const Batch& b) { return d_->launch_batch(b); }
void Model::engram_late_batch(const int32_t* tokens) { d_->engram_late_batch(tokens); }
void Model::engram_late_rows(int R, const int32_t* tokens, const int32_t* pos, const int32_t* slot) {
  d_->engram_late_rows(R, tokens, pos, slot);
}
const SampleOut& Model::wait(uint64_t n) { return d_->wait(n); }
void Model::set_ctl(uint32_t v) { __atomic_store_n(d_->ctl_h, v, __ATOMIC_RELEASE); }
void Model::step(const Ctl& c, int32_t* sampled, Sink* sink) {
  Sampling sp;
  sp.temperature = c.temperature;
  sp.seed = c.seed;
  const SampleOut& o = d_->wait(d_->launch(c.rows, c.pos0, c.tokens, sp, c.flags, sink));
  for (int r = 0; r < c.rows; ++r) sampled[r] = o.row[r].token;
}
void Model::reset() { d_->hist.clear(); }
void Model::trim_history() {
  for (TokWin& h : d_->hists) h.trim();
  d_->hist.trim();
}
void Model::set_images(int n, const int32_t* lo, const int32_t* hi, int key) {
  Impl& I = *d_;
  I.img_span.clear(), I.img_row0.clear(), I.img_key = 0;
  size_t total = 0;
  for (int i = 0; i < n; ++i) {
    if (hi[i] <= lo[i] || lo[i] < 0 || (i && lo[i] < hi[i - 1])) die("model: image span %d [%d, %d)", i, lo[i], hi[i]);
    I.img_span.emplace_back(lo[i], hi[i]), I.img_row0.push_back(total), total += hi[i] - lo[i];
  }
  if (!n) return;
  const imgrows::Ledger::Rows* r = I.core->img_find(key);
  if (!r) die("model: the images of flight %d are not in the store", key);
  if (r->rows != total) die("model: flight %d holds %zu image rows, its spans name %zu", key, r->rows, total);
  I.img_key = key;
}
bool Model::image_take(int key, size_t rows, int slot) { return d_->img_take(key, rows, slot); }
void Model::image_fill(int key, size_t row0, size_t n, const uint16_t* src) { d_->img_fill(key, row0, n, src); }
void Model::image_free(int key) { d_->img_free(key); }
size_t Model::image_numbers() const { return d_->img_numbers(); }
void Model::set_pos_stride(int stride, int group) {
  d_->pos_stride = std::max(1, stride);
  d_->pos_group = std::max(1, group);
}

void Model::set_route_shift(int shift, int group) {
  group = std::max(1, group);
  if (shift == d_->route_shift && group == d_->route_group) return;
  CUDA_CHECK(cudaDeviceSynchronize());
  for (auto& g : d_->graphs) CUDA_CHECK(cudaGraphExecDestroy(g.second));
  d_->graphs.clear();
  d_->route_shift = shift;
  d_->route_group = group;
}
float Model::last_gpu_ms() const {
  const int slot = (int)(d_->last_n % ops::kOutSlots);
  CUDA_CHECK(cudaEventSynchronize(d_->ev_end[slot]));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, d_->ev_beg[slot], d_->ev_end[slot]));
  return ms;
}
void Model::prefill_split(float ms[4]) const {
  CUDA_CHECK(cudaEventSynchronize(d_->ev_pf[4]));
  for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaEventElapsedTime(&ms[i], d_->ev_pf[i], d_->ev_pf[i + 1]));
}
bool Model::prefill_joins(float ms[3], int* pairs) const {
  const detail::Prefill* p = d_->pf.get();
  if (!p || !p->stall) return false;
  CUDA_CHECK(cudaEventSynchronize(d_->ev_pf[4]));
  ms[0] = ms[1] = 0.f;
  for (int i = 0; i < p->jn_n; ++i) {
    float t = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t, p->ev_jn[i][0], p->ev_jn[i][1]));
    ms[p->jn_kind[i]] += t;
  }
  CUDA_CHECK(cudaEventElapsedTime(&ms[2], p->ev_fl[0], p->ev_fl[1]));
  *pairs = p->jn_n;
  return true;
}
void Model::mark(int i) { CUDA_CHECK(cudaEventRecord(d_->ev_mark[i], d_->st)); }
float Model::mark_ms(int i, int j) const {
  CUDA_CHECK(cudaEventSynchronize(d_->ev_mark[j]));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, d_->ev_mark[i], d_->ev_mark[j]));
  return ms;
}
float Model::last_logit(int r) const { return d_->out_h[d_->last_n % ops::kOutSlots].row[r].value; }
int Model::graph_count() const { return (int)d_->graphs.size(); }
#ifdef DSGB10_DEV
bool Model::cert_stats(uint32_t out[8]) const {
  if (!d_->cert.on) return false;
  CUDA_CHECK(cudaMemcpy(out, d_->cert.stats, 32, cudaMemcpyDeviceToHost));
  return true;
}
#endif
int Model::rank() const { return d_->rank; }
cudaStream_t Model::stream() const { return d_->st; }

}  // namespace model
