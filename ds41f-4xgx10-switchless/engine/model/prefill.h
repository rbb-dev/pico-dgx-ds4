// The many-row path (prefill.cu): buffers for chunks of up to `cap` rows, made on the first prefill.
#pragma once
#include <atomic>
#include <thread>
#include <vector>

#include "model/impl.h"

namespace model::detail {

// A window snapshot: the ring slots of positions [pos - 128, pos) of every layer (state.cu keeps them).
constexpr int kSnapRows = 128;
constexpr size_t kSnapKv = (size_t)kSnapRows * kHead;       // bytes per layer
constexpr size_t kSnapSc = (size_t)kSnapRows * kHead / 32;
constexpr size_t kSnapBytes = kLayers * (kSnapKv + kSnapSc);
// a layer's rows within a bookmark (its numbers' bytes: Model::Impl::bk_copy)
inline size_t snap_kv_off(int L) { return (size_t)L * kSnapKv; }
inline size_t snap_sc_off(int L) { return (size_t)kLayers * kSnapKv + (size_t)L * kSnapSc; }

// Sequence parallel: of a chunk of M rows, rank r owns rows [r q, r q + n) with q = q_rows(M) (n = q
// but for the last ranks): their residual stream, mixes, sublayer boundaries and the row-split work.
// Arrays of every row have 4q rows, rank r's at row r q, so that all-gathers fill them in place; the
// rows past M are padding.
inline int q_rows(int M) { return ((M + 3) / 4 + 7) & ~7; }

// CED (docs/ced.md): the decoder replays the last kReplay positions. Its input is layer 20's input as
// the encoder left it, kept per position (slot = position mod kReplay) in records of kCedRow bytes: the
// stream x, the normed input y with its act_quant (yq, ys) and layer 20's attention mixes.
constexpr int kReplay = 128;
// The dense index scans' rows per batch (dsk::index_topk_p: a batch's keys are selected while in L2).
constexpr int kIdxBatch = 128;
constexpr size_t kCedX = 0, kCedY = kCedX + (size_t)kHcDim * 2, kCedYq = kCedY + (size_t)kDim * 2,
                 kCedYs = kCedYq + kDim, kCedPre = kCedYs + kDim / 32, kCedPost = kCedPre + kHc * 4,
                 kCedComb = kCedPost + kHc * 4, kCedRow = kCedComb + kHc * kHc * 4;

// The MoE's down and combine run in column groups, and each group's reduce-scatter goes out on the
// comm stream while the next group computes.
constexpr int kColGroups = 4, kGroupCols = kDim / kColGroups;
constexpr int kRsBlocks = 8;  // blocks of a column group's reduce-scatter (the rest of the GPU keeps computing)
constexpr int kCombRsBlocks = 32;  // blocks of 256 threads of the MoE's combine fused into its reduce-scatter

// Weights the sequence-parallel parts need whole: every rank's slices, gathered on the first prefill.
struct PLayer {
  uint8_t *wqa = nullptr, *sqa = nullptr, *wkv = nullptr, *skv = nullptr;  // wq_a [1280], wkv [512] (L1), scales
  uint8_t *ckv = nullptr, *cgate = nullptr;  // compressor wkv / wgate [512] (kv sources), bf16 tiled (L2)
  uint8_t *ewkv = nullptr, *eskv = nullptr;  // Engram wkv: [rank][6400] (L1), scales [rank][200][192]
  uint8_t *wob = nullptr, *sob = nullptr;    // wo_b: [rank][5120][2048] (L1, each rank's K slice), scales
};

struct Prefill {
  // The loan's body thread (rows_lend.cu) is joined before anything it writes goes: an engine that stops while a grow is under way waits for it.
  ~Prefill() {
    if (loan.th.joinable()) loan.th.join();
  }
  // The buffers whose size follows the chunk's rows, one complete set: a chunk of up to `cap` rows runs on one set and
  // on no other. The standing set (`rows`) is made at start and never freed; a second set may be made later in
  // memory lent by the pool (`big`). A set is never grown or re-made in place: the Engram lookups of the next chunk
  // write into its pinned buffers between commands.
  struct Rows {
    int cap = 0, Q = 0;  // Q = q_rows(cap)
    // this rank's rows (row 0 = the chunk's row r0)
    bf16 *x = nullptr, *mid = nullptr, *y = nullptr, *yraw = nullptr, *sub = nullptr;
    // the mixes: the attention's own (post, comb) and the FFN's pre (from x), the FFN's own and the next
    // attention's pre (from mid)
    float *a_pre = nullptr, *a_post = nullptr, *a_comb = nullptr, *pm = nullptr, *f_post = nullptr, *f_comb = nullptr;
    float *gl = nullptr, *wpf = nullptr;
    bf16 *iq = nullptr, *e_kv = nullptr;
    // every row
    bf16 *part = nullptr, *qa = nullptr, *kvx = nullptr, *qr = nullptr, *q = nullptr, *o = nullptr;
    float *ckvf = nullptr, *cscf = nullptr, *rwts = nullptr;
    uint8_t *yq = nullptr, *ys = nullptr, *qq = nullptr, *qs = nullptr, *oq = nullptr, *os = nullptr;
    uint8_t *oqa = nullptr, *osa = nullptr;  // every rank's wo_a output (act_quant) for this rank's rows: [rank][Q][2048], scales
    uint8_t *lin_kv = nullptr, *lin_sc = nullptr;  // one layer's window: up to 127 positions before the chunk, then the chunk
    int *tok = nullptr, *pos = nullptr, *ridx = nullptr, *counters = nullptr, *topk_mine = nullptr, *topk[kIdxSlots] = {},
        *pool = nullptr;
    // the Reindex scans' rows (Q x the pool), the index top-k's keys
    uint16_t *topkey = nullptr, *poolkey = nullptr;
    uint8_t* iqf = nullptr;  // the dense scans' queries (attn::IndexQ): codes, scales, head weights of this rank's rows
    uint32_t* iqsc = nullptr;
    float* iqw = nullptr;
    void* moe_ws = nullptr;
    int32_t* tok_h = nullptr;  // pinned staging of the chunk's tokens
    // Engram: two sets of lookups, so the next chunk's run while this one computes
    uint8_t *e_send[2][2] = {}, *e_gath = nullptr, *e_xq = nullptr, *e_xs = nullptr, *e_part = nullptr;
    uint32_t* e_off[2][2] = {};
    uint64_t *e_done = nullptr, *e_wait = nullptr;  // pinned [set][layer]: the store's completion flags, the seqs to wait for
    size_t e_seg[2][2] = {};
    int e_pos0[2] = {-1, -1}, e_n[2] = {0, 0};  // the positions of each set's pending lookups (-1: none)
    int e_unit[2] = {-1, -1};                    // and the unit and token epoch they were hashed from: a chunk takes a
    uint64_t e_ep[2] = {0, 0};                   // set only for its own unit's current tokens
    int e_last = 1;  // the set of the chunk enqueued last: the next chunk's lookups go to the other
    std::vector<uint32_t> ids[2];
  };
  Rows rows;           // the standing set
  Rows* rs = &rows;    // the set the command in progress runs on (Model::Impl::prefill_use)
  int rq = 0, r0 = 0, n = 0;  // the chunk in progress: rows per rank, this rank's first row and count
  int a_cur = 0;              // vision: the chunk's (or the replay's) first position
  float *ysum = nullptr, *ymoe = nullptr;
  uint8_t* gath = nullptr;
  // keys: a batch of the dense scans' rows (kIdxBatch x kstride), or the Reindex scans' rows (Q x the pool)
  uint16_t *keys = nullptr, *bkeys = nullptr;
  int key_q = 0;  // the Q of the largest set that can run (the lent one's when there is one): `keys` and the windows' scan buffers are sized by it
  cudaStream_t cs = nullptr;  // the reduce-scatters of the column groups (high priority)
  cudaEvent_t ev_grp[kColGroups], ev_cs;
  // The join clock (DSGB10_PF_STALL, off: nothing below is created or recorded): a timing-event pair around the main
  // stream's wait at each rs_join of the encoder layers, the join's kind (0 the attention's, 1 the MoE's) and the
  // joins of the chunk in progress; one pair recorded back to back at the chunk's start reads what a pair costs by itself.
  static constexpr int kJoinMax = 2 * kLayers;
  bool stall = false;  // the switch, read once at allocation
  bool jn_on = false;  // the chunk's encoder layers are running: rs_join records its pair
  int jn_n = 0;        // pairs recorded in the chunk in progress (reset at its start)
  uint8_t jn_kind[kJoinMax] = {};
  cudaEvent_t ev_jn[kJoinMax][2], ev_fl[2];
  // CED: this rank's records of the positions kept (its own rows of them), every rank's (the replay's
  // all-gather); per slot the position (-1: none) and the rank whose row it was; the end of the last
  // encoder chunk (a chunk that does not continue it starts a new run of positions)
  uint8_t *ced_rows, *ced_gath;
  int ced_pos[kReplay], ced_src[kReplay];
  int ced_end = -1;
  PLayer pl[kLayers];
  // The second set of row buffers, lent by the pool (tenant lend::kTenantPrefill, model/rows_lend.cu): `big` is whole and usable only
  // while that tenant holds its groups and no change of it is in flight (Model::Impl::pf_big_ready); the loan's body makes and drops
  // it on a thread of its own.
  Rows big;
  struct Loan {
    std::thread th;
    std::atomic<int> st{0};  // 0 idle, 1 running, 2 done, 3 failed
    int device = 0;
    // the bytes of the set: `cur` as the thread leaves it, `shown` as the engine thread has seen it (what the driver reads), `before` the
    // shown size when the change began
    uint64_t cur = 0, shown = 0, before = 0;
    Rows made;                         // the thread's result, taken into `big` by the engine thread when it is done
    std::vector<std::pair<void*, bool>> owned;  // every allocation of `made` and then of `big`: pointer, pinned host
  } loan;
};

// One set of the row-sized buffers for chunks of up to M rows (at least the CED replay's kReplay), every buffer made by the
// allocator `a` (a.d<T>(n) device, a.h<T>(n) pinned host; a.ok false: a call failed, the rest are skipped and the set is not whole).
// The standing set, the lent set and the count of its bytes (rows_bytes) are this one function, so they cannot differ.
template <class A>
void rows_alloc(Prefill::Rows& r, int M, A& a) {
  M = std::max(M, kReplay);
  const int Q = q_rows(M), M4 = 4 * Q;
  r.cap = M, r.Q = Q;
  r.x = a.template d<bf16>((size_t)Q * kHcDim);
  r.mid = a.template d<bf16>((size_t)Q * kHcDim);
  r.y = a.template d<bf16>((size_t)Q * kDim);
  r.yraw = a.template d<bf16>((size_t)Q * kDim);
  r.sub = a.template d<bf16>((size_t)Q * kDim);
  r.a_pre = a.template d<float>((size_t)Q * kHc);
  r.a_post = a.template d<float>((size_t)Q * kHc);
  r.a_comb = a.template d<float>((size_t)Q * kHc * kHc);
  r.pm = a.template d<float>((size_t)Q * kHc);
  r.f_post = a.template d<float>((size_t)Q * kHc);
  r.f_comb = a.template d<float>((size_t)Q * kHc * kHc);
  r.gl = a.template d<float>((size_t)Q * kExperts);
  r.wpf = a.template d<float>((size_t)Q * kIdxHeads);
  r.iq = a.template d<bf16>((size_t)Q * kIdxQ);
  r.part = a.template d<bf16>((size_t)M4 * kDim);
  r.qa = a.template d<bf16>((size_t)M4 * kQLora);
  r.kvx = a.template d<bf16>((size_t)M4 * kHead);
  r.qr = a.template d<bf16>((size_t)M * kQLora);
  r.q = a.template d<bf16>((size_t)M * kQ);
  r.o = a.template d<bf16>((size_t)M * kQ);
  r.ckvf = a.template d<float>((size_t)M4 * kHead);
  r.cscf = a.template d<float>((size_t)M4 * kHead);
  r.rwts = a.template d<float>((size_t)M4 * kTopK);
  r.yq = a.template d<uint8_t>((size_t)M4 * kDim);
  r.ys = a.template d<uint8_t>((size_t)M4 * kDim / 32);
  r.qq = a.template d<uint8_t>((size_t)M * kQLora);
  r.qs = a.template d<uint8_t>((size_t)M * kQLora / 32);
  r.oq = a.template d<uint8_t>((size_t)M * kOPart);
  r.os = a.template d<uint8_t>((size_t)M * kOPart / 32);
  r.oqa = a.template d<uint8_t>((size_t)M4 * kOPart);
  r.osa = a.template d<uint8_t>((size_t)M4 * kOPart / 32);
  r.lin_kv = a.template d<uint8_t>((size_t)(M + 127) * kHead);
  r.lin_sc = a.template d<uint8_t>((size_t)(M + 127) * kHead / 32);
  r.tok = a.template d<int>(M);
  r.pos = a.template d<int>(M);
  r.ridx = a.template d<int>((size_t)M4 * kTopK);
  r.counters = a.template d<int>(Q);
  r.topk_mine = a.template d<int>((size_t)Q * kIdxTopk);
  for (auto& t : r.topk) t = a.template d<int>((size_t)M4 * kIdxTopk);
  r.pool = a.template d<int>((size_t)Q * kPoolBlocks);
  r.iqf = a.template d<uint8_t>((size_t)Q * kIdxHeads * 64);
  r.iqsc = a.template d<uint32_t>((size_t)Q * kIdxHeads);
  r.iqw = a.template d<float>((size_t)Q * kIdxHeads);
  r.topkey = a.template d<uint16_t>((size_t)Q * kIdxTopk);
  r.poolkey = a.template d<uint16_t>((size_t)Q * kPoolBlocks);
  r.moe_ws = a.template d<uint8_t>(dsk::moe_prefill_workspace_bytes(M, kExperts));
  r.tok_h = a.template h<int32_t>(M);
  const size_t seg = (size_t)M * kEngramCols * kEngramRow;  // one rank may own every cell
  for (int set = 0; set < 2; ++set)
    for (int e = 0; e < 2; ++e) {
      r.e_send[set][e] = a.template h<uint8_t>(seg);
      r.e_off[set][e] = a.template h<uint32_t>((size_t)M * kEngramCols);
    }
  r.e_done = a.template h<uint64_t>(4);
  r.e_wait = a.template h<uint64_t>(4);
  r.e_gath = a.template d<uint8_t>(kRanks * seg);
  r.e_xq = a.template d<uint8_t>((size_t)Q * kEngramK);
  r.e_xs = a.template d<uint8_t>((size_t)Q * kEngramK / 32);
  r.e_kv = a.template d<bf16>((size_t)kRanks * Q * kEngramN);
  r.e_part = a.template d<uint8_t>((size_t)kRanks * Q * kEngramPartBytes);
}

// Counts what rows_alloc asks for: device bytes and pinned host bytes of a set of M rows (nothing is allocated).
struct CountAlloc {
  bool ok = true;
  uint64_t dev = 0, host = 0;
  template <class T>
  T* d(size_t n) { dev += n * sizeof(T); return reinterpret_cast<T*>(16); }
  template <class T>
  T* h(size_t n) { host += n * sizeof(T); return reinterpret_cast<T*>(16); }
};
inline uint64_t rows_bytes(int M) {
  Prefill::Rows r;
  CountAlloc c;
  rows_alloc(r, M, c);
  return c.dev + c.host;
}

}  // namespace model::detail
