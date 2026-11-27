// The TP=4 forward of DeepSeek-V4.1-Flash on one rank (docs/engine.md). Every rank holds a Model and
// runs the same steps in the same order; the residual stream, the routing and the index choices are
// bit-identical on all four (collectives sum in rank order). A step takes R <= kMaxRows rows: one session's
// R <= kMaxSessRows consecutive positions (a decode step, R = 1; a verify step, R <= 6; a prefill chunk by
// steps, exact mode: every layer, every row), or several sessions' runs of consecutive positions (Batch).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comm/ring.h"
#include "engram/engram.h"
#include "model/config.h"
#include "model/ops.h"
#include "weights/loader.h"

namespace vit {
struct DevWeights;
}

namespace model {

// How a step's rows are sampled (ops::StepIn); every rank must pass the same.
struct Sampling {
  float temperature = 0.f;  // 0: greedy
  float top_p = 1.f;        // 1: off
  int32_t top_k = 0;        // 0: off
  int32_t n_top = 0;        // top logprobs per row, 0..ops::kTopLogprobs
  uint64_t seed = 0;
  float ablit = 0.f;  // 1: the runtime abliteration (Options::ablit) on this request's rows; 0: the checkpoint's model
};
// A session's sampling as a step's input carries it, every field: every step kind (one session, batch, batch spec)
// builds its input with this one function, so a field such as the abliteration's strength (read per row by
// ablit.cu) reaches every kind.
inline ops::StepSamp samp_of(const Sampling& s) {
  return ops::StepSamp{s.seed, s.temperature, s.top_p, s.top_k, s.n_top, s.ablit};
}
using ops::SampleOut;

// Which sampled requests ride the Gumbel path (ops::kSampGumbel in Sampling::top_k; samp_rule.h has the rule, model.cu reads the
// variables once).
//   DSGB10_GSAMPLE unset / 0: none: every request is sampled as before.
//   1: a sampled request whose client sent NO seed, and a sampled request without top-p / top-k (its sampler is the Gumbel-max
//      one already; only its drafts change). A seeded top-p / top-k request keeps its sampler and its text.
//   2: every sampled request (a seeded top-p / top-k request then gives another text for its seed, from the same distribution).
// On the path a truncated row takes the Gumbel-max winner among the kept tokens, and with DSGB10_GCOUPLE (dspark.h) the drafts
// are picked with the request's own noise. Greedy requests are never on it.
const ops::GSettings& gsample_settings();  // model.cu: DSGB10_GSAMPLE, DSGB10_GCOUPLE, DSGB10_GCOUPLE_CF, read once; a bad value ends the start
inline int gsample_mode() { return gsample_settings().mode; }
inline int32_t gsample_top_k(int32_t top_k, float temperature, float top_p, bool seeded) {
  return ops::gsample_top_k_mode(gsample_mode(), top_k, temperature, top_p, seeded);
}

struct Sink;

// A step of rows from several sessions (docs/batching.md, B2; Model::launch_batch). Row r is position
// pos[r] of the session in slot slot[r], fed tokens[r] (>= 0, or -1 - j: row j of the previous step's
// sample, as launch()'s). A session's rows are consecutive and ascend one position at a time, at most
// kMaxSessRows of them (B4: up to kMaxRows rows in all); sess[r] numbers the step's sessions in row order
// from 0, each in a slot of its own, and session k is sampled by sp[k]. flags and sink are launch()'s
// (kFlagDump with a sink: an eager step handing rank 0's sink each layer's results, e.g. the bench's
// routed experts; the tokens must then be given).
struct Batch {
  int R = 0, nsess = 0;
  int32_t tokens[kMaxRows] = {}, pos[kMaxRows] = {}, slot[kMaxRows] = {}, sess[kMaxRows] = {};
  Sampling sp[ops::kMaxSess];
  uint32_t flags = 0;
  Sink* sink = nullptr;
};

// What a spec step (spec.cu) lets its draft pass choose for the step after it: by default draft j is
// verified while the confidence head's survival[j - 1] (P(d_1 .. d_j all accepted)) >= thresh, at most
// max_drafts of them, every row below pos_end; the other gates (ops.h kGate*) use step_ms. Every rank
// must pass the same.
struct SpecCfg {
  float thresh = 0.21f;
  int max_drafts = 5;  // 0..5 (0..dspark::draft_rows() with DSGB10_DRAFT_MAX)
  int pos_end = 0;
  int force = -1;  // >= 0: draft from this token instead of the step's last sample (teacher-forced checks)
  int probe[ops::kDrafts] = {-1, -1, -1, -1, -1, -1, -1};  // EXPERIMENT (Options::tree_probe): the probe's path
  int gate = ops::kGateThresh;
  float step_ms[ops::kDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f, 41.70f, 44.70f};  // 0..7 drafts (6 and 7: DSGB10_DRAFT_MAX)
};

// A batch spec step (docs/batching.md, B3: drafts across sessions; Model::launch_spec_batch and
// launch_verify_batch): n entries, each one session in a slot of its own, taking part when on[k]. In a
// given step entry k is one row, token[k] at pos0[k]; in a verify step its rows are what the step before
// chose for it (its bonus token, then at most cap[k] drafts, every row below pos_end[k], thresh[k] its
// kGateThresh threshold; cfg's gate, step_ms, force and probe serve every entry, its thresh, max_drafts
// and pos_end are unused). The entries on, in entry order, make the step's rows; the caller keeps their
// next rows (1 + cap[k] each) within kMaxRows. Every rank must pass the same.
struct SpecBatch {
  int n = 0;
  int32_t on[ops::kMaxSess] = {}, slot[ops::kMaxSess] = {}, token[ops::kMaxSess] = {}, pos0[ops::kMaxSess] = {};
  int32_t cap[ops::kMaxSess] = {}, pos_end[ops::kMaxSess] = {};
  float thresh[ops::kMaxSess] = {};
  Sampling sp[ops::kMaxSess];
  SpecCfg cfg;
  int pos_bound = 0;  // launch_verify_batch: the rows' last position at most (the context bucket)
  // launch_verify_batch: with per_entry_bound, entry k's rows' last position at most is bound[k] (the slot
  // memory it needs mapped, Options::vmm); else every entry's slot is mapped up to pos_bound
  bool per_entry_bound = false;
  int32_t bound[ops::kMaxSess] = {};
  int budget = 0;     // night T2/T4: > 0: the next step's rows across the entries, at most (StepIn::rbudget)
};

// One step, as rank 0 decided it (the golden and bench modes: every rank receives the same).
struct Ctl {
  int32_t op = 0;     // kOpStep / kOpStop
  int32_t rows = 0;   // R: positions pos0 .. pos0 + R - 1
  int32_t pos0 = 0;
  uint32_t flags = 0;
  int32_t tokens[kMaxRows] = {};
  float temperature = 0.f;  // 0: greedy; else Gumbel-max with `seed`
  uint32_t seed = 0;
  int32_t pad[2] = {};
};
enum : int32_t { kOpStop = 0, kOpStep = 1 };
enum : uint32_t {
  kFlagInject = 1,     // before layers >= 1: residual and pre_mix := rank 0's Sink::fill (golden inputs)
  kFlagDump = 2,       // rank 0 hands each layer's results to Sink::tensor
  kFlagLogits = 4,     // all-gather the full logits; rank 0 hands them to Sink::logits
  kFlagInjectSub = 8,  // also the FFN's input stream (mid), after it was dumped
  kFlagProfile = 16,   // every rank starts the CUDA profiler's capture range (nsys) before this step
};

// Rank 0's view into a step (eager steps only), for the golden comparison.
struct Sink {
  virtual ~Sink() = default;
  // R rows of row_bytes: the golden `name` of layer L at positions pos0 .. : "x" (the block input, i.e.
  // layer L-1's out), "pre_mix" (L-1's), "attn_in", "mid", "ffn_in" (bf16 bits, pre_mix fp32).
  virtual void fill(int L, const char* name, int pos0, int R, void* host, size_t row_bytes) = 0;
  // A step tensor on the host, R rows of row_bytes. L = -1 for the final ones.
  virtual void tensor(int L, const char* name, int pos0, int R, const void* data, size_t row_bytes) = 0;
  virtual void logits(int pos0, int R, const float* full) = 0;  // [R][kVocab]
};

// Contexts in steps of kCtxBucket positions (the graph sweep, spec_graphs). The step graphs do not depend on
// the context: index_dense reads the scan's length from the positions (impl.h, scan_ctx).
inline constexpr int kCtxBucket = 4096;

// P-scan (attn/indexer.cu index_dense's part, nparts; docs/engine.md "P-scan"): the decode index scan split by positions over the
// four ranks, each rank's quarter of every session's keys by index_dense, the picks all-gathered and merged by index_merge2
// (exactly the whole scan's picks). off: every rank scans every key (no extra buffer, the graphs and launches unchanged);
// ext: the step graphs of factors 32 and 64 (windows past 1,048,576 positions) split; all: factor 16's too.
enum class PScan { Off, Ext, All };

struct Options {
  int copy_mib = 512;       // copy drafting (copy.h, DSGB10_COPY): MiB a rank for the units' token
                            // histories and tables; a unit that does not fit gets a smaller table, then none
  int max_ctx = 65536;      // positions the RoPE tables and compressed caches cover
  int n_layers = kLayers;   // run only blocks [first_layer, n_layers) (tests with golden injection; the
  int first_layer = 0;      // logits are then meaningless)
  bool graphs = true;       // replay steps without golden checks as CUDA graphs (one per row count and
                            // context bucket, captured on first use)
  int prefill_rows = 4096;  // rows per prefill chunk, at most
  // > prefill_rows: a second set of row buffers for chunks of this many rows is borrowed from the pool (tenant lend::kTenantPrefill)
  int prefill_rows_max = 0;
  int wave_probe = 0;       // probe: bit L - layer L's FFN beside a second attention side, in the chunk
  int wave_probe_at = -1;  // at this position (wave_probe())
  bool draft_fp8 = false;   // DSpark's head GEMV on an FP8 copy of the vocab head (bind_spec; +165 MB): the
                            // drafts' arithmetic changes, the tokens cannot
  // The draft pass's cheaper formats (docs/engine.md, "The draft pass's approximations"; bundle B), drafts only:
  bool draft_head4 = false;    // the head as MXFP4, made from the FP8 copy (dspark::head_fp4; +88 MB, the FP8
                               // copy freed): the study's simulated FP4 head bit for bit
  bool draft_markov8 = false;  // the Markov head as FP8 (dspark::markov_fp8; +8.3 MB)
  int slots = 1;            // session slots (docs/batching.md): live sessions' states side by side (each one's
                            // window rings, compressed caches, index keys, carries and draft windows), up to
                            // kMaxSlots; ~240 MB a slot at 262,144 positions
  int big_slots = 0;        // with 0 < big_slots < slots: slots [0, big_slots) cover max_ctx, the others
                            // max_ctx / kUnitMul (slot_ctx); 0: every slot covers max_ctx
  // The kv sources' arrays (compressed KV, index keys and their scales) as one chunk pool made at the start
  // (docs/engine.md, "The chunk pool"): what the slots would take whole, or slot_gib GiB of it (> 0), or with
  // pool_one_line one line's worth (serve: the pool grows to its chosen size before serving, state_reserve).
  double slot_gib = 0;
  bool pool_one_line = false;
  bool kv_split = false;    // --kv-split tiered | max: a stored conversation is marked (clean pieces become ghosts) once its images are on the SSD
  bool imager = false;      // the disk tier is written in the background as each Spark's own quarter, on a second stream; every mode
  int kv_split_mode = 0;    // splitpace::Mode (0 off, 1 tiered, 2 max): the riding of the image commands
  int kv_join_pieces = 64;  // --kv-join-pieces: the most pieces one join command takes
  int bgraph_bodies = 0;    // DIAGNOSTICS (spec.cu, 2026-10-13): the batch spec graph's switch bodies (0: kMaxRows)
  bool bgraph_noif = false; // DIAGNOSTICS: the entries' accepts at the top level, without if nodes
  bool bgraph_check = false;  // DIAGNOSTICS (B3 phase 2a): after the batch step's draft pass over the entries, every
                              // entry's one-session draft too, and a count of the words of its drafts and
                              // confidence that differ (spec_batch_mismatches(); the pass's bit check)
  bool l2pf = true;         // the decode step's paced L2 prefetch in its windows (kernels/l2pf.cu, docs/engine.md
                            // "A paced L2 prefetch"): a side stream, the same bits
  // EXPERIMENT, bench only (docs/engine.md, "Draft trees"): every spec step's draft pass also runs the
  // drafter's probe (dspark::Draft::probe) along SpecCfg::probe, into SampleOut::probe
  bool tree_probe = false;
  // the runtime abliteration (model/ablit.h, tools/ablit/make_ablit.py): its file; empty: none (Sampling::ablit is ignored)
  std::string ablit;
  // the hand-over proof's MemAvailable tolerance is at least 256 MiB; this is the null control's spread (MiB) when that is larger
  int lend_null_mib = 0;
  PScan pscan = PScan::Off;  // --pscan off|ext|all: which step graphs split the decode index scan over the ranks
};

class Model {
 public:
  // CUDA must be initialised. Allocates the state (do this before reading weights).
  Model(int rank, const Options& opt);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Binds the loaded weights: re-lays a version-1 file into the kernels' layouts (tile_fp8 / tile_fp4)
  // and all-gathers the mHC projections. Collective: every rank calls it.
  void bind(dsw::WeightFile& wf, comm::Ring& ring);
  void bind_ring2(comm::Ring& ring);  // --wave-probe
  // The vocab head coded in place (dsk::bf16x_encode: each tile of 8 rows x 32 values with its exponents as
  // 4-bit offsets, 3/4 of the bytes; checked against the head before it replaces it): every reader of the
  // head moves to the coded GEMVs, which give the same bits. After bind and bind_spec (DSpark's FP8 copy is
  // made from the raw head), before the first step. Returns the coded bytes (in the head's place; the rest
  // of its slot stays allocated), or 0 if coding would not make it smaller (the head stays raw);
  // *raw_tiles: the tiles kept raw.
  size_t code_head(int* raw_tiles = nullptr);
  // The Engram row store of this rank (owned by the caller).
  void set_engram(const engram::HashMap* map, engram::Store* store);
  // Session slots (Options::slots). From now on the eager paths (prefill chunks, the state operations,
  // snapshots) act on slot s, and the steps launched have their rows in slot s. Every rank must make the
  // same calls. Slot 0 at the start.
  void use_slot(int s);
  int slot() const;
  // The YaRN factor (16, 32 or 64: windows of 1M, 2M, 4M over the original 65536) slot s rotates with, from its next
  // step or chunk on; 16 until set. A decode step's rows must share one factor (the model dies otherwise); graphs are
  // kept per factor. Every rank makes the same calls. A slot's cache is not rewritten: move a session to another
  // factor only with an empty or re-prefilled slot.
  void set_slot_factor(int s, int factor);
  int slot_factor(int s) const;
  // Serving past 1M (Options::pool_one_line, max_ctx over 1,048,576; window_lazy): window_open(factor) takes a factor 32
  // or 64 window's memory out of the chunk pool (whole groups of numbers, evicting the conversations holding them first),
  // makes its buffers and graphs and gives back the groups the build did not use; window_close(factor) frees them and the
  // pool makes the numbers again (factor 0: only numbers an earlier close could not take back). set_slot_factor refuses a
  // factor whose window is not open. Every rank calls alike, one command at a time (srv::run_window).
  // kWindowBusy: sessions on lines, pinned ones or a unit's own hold too much of the pool, try later (the server detaches idle and paused lines first, so these are running requests); kWindowNoRoom: the pool cannot hold one
  // conversation of that window; kWindowFailed: a rank had too little device memory (nothing changed).
  enum WindowStatus { kWindowOk = 0, kWindowBusy, kWindowNoRoom, kWindowFailed };
  bool window_lazy() const;
  bool window_is_open(int factor) const;  // always true for 16, and for every factor when not lazy
  int window_open(int factor);
  int window_close(int factor);
  int window_owed() const;     // groups of the pool's numbers a closed window has not yet given back
  int window_groups() const;   // groups out of the pool
  // What opening factor f's window with `g` groups would need out of the pool, without evicting anything (rank 0's server
  // plans with it; it reads only the store's records, so it changes nothing on any rank): `top` is the first number that
  // leaves the pool, `removed` the numbers cut off, `holders` every session holding a number >= top (handle, the unit that
  // holds it or -1, and whether it is a unit's own session of a tool or test, which can never be evicted). A pool that
  // could not hold one conversation of the window gives removed 0 and no holders (the open answers kWindowNoRoom).
  struct WindowHolder {
    int handle;
    int line;
    bool unit_own;
  };
  struct WindowPlan {
    uint32_t top = 0;
    int64_t removed = 0;
    std::vector<WindowHolder> holders;
    int64_t image_numbers = 0;  // numbers >= top that image flights hold (they cannot move: the window waits for those flights)
    int image_flights = 0;      // how many flights hold them
  };
  WindowPlan window_plan(int f, int g) const;
  // The groups factor f's window needs by this rank's own estimate (window_open takes the largest of every rank's, so the
  // exact value is only known to the command), and the exact groups of the last open of factor f that found the pool busy
  // (0: none), the same on every rank: a plan made with the first is made again with the second.
  int window_groups_estimate(int f) const;
  int window_busy_groups(int f) const;
  // ---- the Engram row cache as a borrower of the pool's memory (tenant.cu; docs/engine.md, "The Engram cache's memory") ----
  // The cache's block (slots, keys, reference bits, tables) is host memory taken from the pool by whole groups: the store
  // gives the groups' chunks back to the driver, the cache's worker commits and populates the same bytes as one block and
  // gives them back the other way. engram_lend(kLendBegin, mib) starts a change to `mib` MiB of slots (the display
  // reservation's included) and engram_lend(kLendPoll, 0) finishes it once every rank's worker has: both are commands
  // (every rank, the same answer, one at a time; the server calls them at run boundaries). A setting past what the pool can
  // lend, or the block's address space, is cut to what fits (kLendDone, engram_lend_info().mib says what it is). A failed
  // hand-over leaves the cache at its old size and the groups back in the pool, or owed (window_owed) when a rank's driver
  // cannot make the chunks yet; nothing dies. Test builds (DSGB10_TEST_FAULTS): DSGB10_LEND_FAIL=rank:R,step:B|C|back[,skip:N].
  enum LendStatus { kLendIdle = 0, kLendDone, kLendPending, kLendBusy, kLendNoRoom, kLendFailed, kLendSame };
  enum { kLendBegin = 1, kLendPoll = 2 };
  bool engram_lend_enabled() const;  // the store's cache can change size
  int engram_lend(int phase, int64_t mib);
  struct EngramLendInfo {
    uint64_t mib = 0;            // the cache's slots now, MiB (the display reservation's included)
    uint64_t slots = 0;
    uint64_t block_bytes = 0;    // committed behind the slots and the index, this rank
    uint64_t groups = 0;         // groups of the pool the cache holds
    uint64_t pending_mib = 0;    // the size a running change is going to (0: none)
    uint64_t ok_total = 0, failed_total = 0;
    uint64_t huge_pct = 0;       // of the block's resident bytes in huge pages, at the last change's end
    uint64_t lookups = 0, hits = 0, evictions = 0;  // the cache's, since the start
    int running = 0;             // 1 growing, 2 shrinking, 3 going back
    int last_status = 0;
    const char* reason = "";     // why the last change failed
  };
  EngramLendInfo engram_lend_info() const;
  // The largest size (MiB) whose block takes at most `groups` groups, within what the cache can grow to.
  uint64_t engram_mib_for_groups(int groups) const;
  // ---- the tenants of the pool (tenant.cu; lend.h's table) ----
  // Every borrower of the pool's groups is a tenant with an id: 0 the windows, 1 the Engram cache, 2 not assigned, 3 a work area,
  // 8-15 models loaded while the engine runs. tenant_lend(id, kLendBegin, mib) starts a change of tenant `id` to `mib` MiB and
  // tenant_lend(id, kLendPoll, 0) finishes it (commands, as engram_lend, which is tenant 1). Only one change is in flight at a
  // time, and while it is no window command and no settle of owed groups may run (tenant_active: the server waits). Test
  // builds (DSGB10_TEST_FAULTS) have tenant 15, a dummy device block filled with a pattern; DSGB10_LEND_FAIL's `tenant:N` aims a fault.
  int tenant_lend(int tenant, int phase, int64_t mib);
  bool tenant_active() const;     // a change is in flight
  bool window_may_run() const;    // no window command and no settle of owed groups while a tenant's change is in flight (lend::StoreQueue)
  int tenant_group_count(int id) const;  // the groups tenant `id` holds (settled)
  int tenant_change_id() const;   // the tenant it is for (-1: none)
  int tenant_victim() const;      // the tenant to ask for one group when the reclaim's first tiers are spent (-1: none)
  bool tenant_table_ok() const;   // sum(groups) + sum(pending) + owed = the groups out of the pool
  bool tenant_other_enabled() const;  // a tenant besides the windows and the Engram cache is registered
  uint64_t group_bytes() const;   // one group's bytes on this rank (64 numbers)
  struct TenantRow {
    int id = 0, kind = 0, standing = 0;  // lend.h's TenantKind and Standing
    int groups = 0, pending = 0;
    uint64_t size_bytes = 0;  // the tenant's own size
    bool idle = true;
  };
  std::vector<TenantRow> tenant_rows() const;
  // ---- the image encoder in the engine (vit_area.cu, vit_cmd.cu; docs/vision.md) ----
  // set_vision (every rank, once the vision weights are on the device and before the store is sized): this rank's quarter of the
  // encoder, the pinned staging buffer for pixels (kVitStageBytes: it is made now, so the sizing read already counts it) and
  // tenant 3, the work area (device memory, Required). The work area itself is taken by the start's hand-over, tenant_lend(
  // lend::kTenantWork, kLendBegin, vision_area_mib()), and kept: it is never given back while the engine runs.
  static constexpr size_t kVitStageBytes = 6u << 20;  // the largest image's pixels (588 B x 9,216 patches) in one piece
  void set_vision(const vit::DevWeights* w);
  static uint64_t vision_area_bytes();   // what the work area must hold: the largest image (vit::kMaxPatches, at most 1,024 rows)
  static uint64_t vision_area_mib() { return (vision_area_bytes() + (1u << 20) - 1) >> 20; }
  // 0: no encoder (no vision weights); 1: ready (the work area is held on this rank); 2: weights loaded, no work area (the start's take failed)
  int vision_state() const;
  uint64_t vision_area_held() const;     // the work area's bytes on this rank
  // A pixel flight's spans (kCmdImages with the flag): flight `key` has `n` spans whose rows are not written yet. Rows written by
  // image_fill need no record (they are written when the flight is made).
  void vision_register(int key, int n);
  // One range of blocks [b0, b1) of image `span` of flight `key` (nh x nw patches; its span's first row among the flight's rows
  // is row0), the same call on every rank. Rank 0 passes the image's pixels (588 B a patch, in the request's frame; read when
  // b0 = 0), the others null. At b1 = the last block the span's rows are written straight into the flight's numbers and the four
  // ranks vote on them. Returns the same value on every rank. The engine stops (die) when a range is not the next of the one image
  // that is between its blocks, or an image starts while one is.
  enum VitStatus { kVitOk = 0, kVitVote, kVitFail };
  static constexpr int kVitBlocks = 32;
  int vision_run(int key, int span, int row0, int nh, int nw, int b0, int b1, const uint8_t* pixels);
  int slots() const;
  int slot_ctx(int s) const;  // positions slot s holds (Options::big_slots)
  // Slot memory (Options::vmm; every call is a no-op or zero without it). slot_reserve: commit the chunks a request of
  // `need` positions can use in slot s, replacing its earlier reservation; dies when other commitments leave too little
  // (check slot_chunks_for first). slot_release: slot s's session has left (state saved, nothing queued): its chunks go back.
  int slot_chunks_for(int s, int need) const;  // chunks slot s needs from empty to hold `need` positions
  int slot_committed(int s) const;             // chunks slot s holds or has reserved, whichever is more
  int slot_mapped(int s) const;                // chunks mapped in slot s
  int slot_unheld_reservation(int s) const;               // numbers slot s is reserved beyond what it holds (they end with its detach)
  int slot_uncommitted() const;                // chunks no slot holds or has reserved
  int64_t slot_spare() const;                  // the same, signed: below 0 where the reservations exceed the free numbers (rules::spare)
  int slot_chunks_total() const;               // chunks the pool made
  int slot_chunks_peak() const;                // most chunks mapped at once
  // `img`: numbers also reserved to the request's image rows, which image_take takes next (counted like the reservation)
  void slot_reserve(int s, int need, int img = 0);
  void slot_unreserve(int s);  // slot s's reservation to 0: its request has finished (the line keeps what it holds)
  void slot_release(int s);
  void slot_attach(int s, int h, int upto = INT32_MAX);  // slot s's unit holds stored session h from now on (no copy: its rows are h's); what is split of it below `upto` comes back, the tail stays split

  // Queue one step of R rows at positions pos0 .. pos0 + R - 1 on every rank, without waiting for it;
  // returns its number. tokens[r] >= 0 is the token; -1 - j stands for row j of the previous step's
  // sample, which the device fills in (a decode step queued before the one before it has ended): call
  // engram_late() with the real tokens once they are known. Steps with golden checks (flags) run
  // eagerly and finish before this returns.
  uint64_t launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, uint32_t flags = 0, Sink* sink = nullptr);
  // The history and Engram lookups of a launched step whose tokens were left to the device (a verify
  // step: its positions and rows too, SampleOut::next_* of the step before it).
  void engram_late(int pos0, int R, const int32_t* tokens);
  // A step of rows from several sessions (Batch; Options::slots > 1 for more than one): as launch(), with
  // each row in its own session's slot and Engram history, and each session's own sampling. Every row
  // computes what it would in a step of its session alone: the same tokens and logits, bit for bit.
  uint64_t launch_batch(const Batch& b);
  // engram_late() of a batch step: its rows' tokens, in row order.
  void engram_late_batch(const int32_t* tokens);
  // DSpark, the model's draft module (spec.cu; wf must hold mtp.*, embed and head). A spec step runs its
  // rows with main_hidden taken at layers 37-39, samples them, accept()s its final rows into the draft
  // module's windows, drafts from the token after them, and chooses the next step's rows: that token,
  // then the drafts worth verifying (SpecCfg). launch_spec: R rows from the host, all final;
  // launch_verify: the rows the step before chose (queued before it ends; pos_bound: their last
  // position at most, for the context bucket). A verify step's SampleOut holds the tokens it emits:
  // rows (1 + the drafts accepted) of row[], then next_* for the step after it.
  void bind_spec(dsw::WeightFile& wf, const dsw::WeightFile* draft = nullptr);  // draft: mtp.* from it
  // the spec step's graph of every context bucket, captured now rather than at its first step (serve's
  // startup: ~0.2 GiB of device memory each)
  void spec_graphs();
  bool spec() const;
  uint64_t launch_spec(int R, int pos0, const int32_t* tokens, const Sampling& sp, const SpecCfg& c);
  uint64_t launch_verify(const Sampling& sp, const SpecCfg& c, int pos_bound);
  // B3 (docs/batching.md): drafts across sessions, a spec step of several sessions' rows (SpecBatch;
  // Options::slots > 1). launch_spec_batch: a given step, one row per entry on; launch_verify_batch: the
  // rows the step before chose for each entry on (queued before it ends). The step samples its rows with
  // the verdicts per entry, runs each entry's accept in its own slot and one draft pass over the entries
  // (phase 2a); every row computes what it would in a step of its session alone, bit for bit. SampleOut:
  // sess_* per entry (its rows, the tokens it emits), next_s* for the step after (0 rows: the entry was
  // off). spec_batch_entries(): entries a batch step takes at most (0 without DSpark or slots).
  // spec_batch_mismatches() (Options::bgraph_check): the words of the passes' drafts and confidence that
  // differed from the entries' one-session drafts so far (synchronises).
  int spec_batch_entries() const;
  // wide64: the rows a batch spec step may reach (kDecodeRows, or DSGB10_WIDE's)
  int spec_batch_rows() const;
  uint64_t launch_spec_batch(const SpecBatch& b);
  uint64_t launch_verify_batch(const SpecBatch& b);
  unsigned spec_batch_mismatches() const;
  // engram_late() of a batch verify step: its rows as the device laid them out (the entries on in entry
  // order, each its next_s* of the step before): their tokens, positions and slots.
  void engram_late_rows(int R, const int32_t* tokens, const int32_t* pos, const int32_t* slot);
  // Spin until step n's sample is in. The result stays valid until step n + ops::kOutSlots is launched.
  const SampleOut& wait(uint64_t n);
  // Rank 0's control word: every step's sampling reads it when it runs and hands it to every rank
  // (SampleOut::ctl), e.g. to stop a decode loop that runs ahead of the host on all ranks at once.
  void set_ctl(uint32_t v);
  // Prefill: positions [a, a + M) (M <= Options::prefill_rows) through every layer, with the tokens of
  // set_tokens(). Window snapshots (state_snap) of session `handle` at the positions `snaps`, all in
  // (a, a + M], are taken on the way. With sp the last row is sampled: returns its step number (wait()
  // for it), else ~0. prefill_lookup() starts a chunk's Engram lookups early (the next chunk's, while
  // this one runs); prefill() starts them itself when they were not.
  // With ced (docs/ced.md) the chunk runs the encoder (layers 0-19) and layer 20's KV projection only;
  // the chunk that samples then replays layers 20-39 over its last 128 positions [P - 128, P) from
  // fresh decoder rings, and samples that replay's last row. Those 128 positions must have run through
  // CED chunks of one run (each chunk starting where the one before ended), P >= 128. Of the snapshots
  // only one at P covers the decoder layers; the others hold the encoder layers' rings only.
  void set_tokens(int pos0, int n, const int32_t* tokens);
  // Vision (docs/vision.md): image spans [lo[i], hi[i]) of the current slot's sequence, whose rows are flight `key`'s in the
  // store (image_take, image_fill): used by its prefill chunks until set_images(0, ...) (which leaves the rows where they
  // are): the rows replace the embeddings there, the span routes with ffn.gate.bias_vl and Engram leaves it alone. The
  // history (set_tokens) must hold -1 at those positions. A request's every go sets the spans again and sends no rows.
  // Text requests never call it.
  void set_images(int n, const int32_t* lo, const int32_t* hi, int key);
  // A flight's rows in the store (model/img_rows.h; the same on every rank): image_take takes the numbers for `rows` rows
  // (false: a key that holds rows already, or no room), clearing slot `slot`'s reservation (slot_reserve's `img`); image_fill
  // writes rows [row0, row0 + n) from host memory, n at most kImgStageRows; image_free gives a flight's numbers back (key 0:
  // every flight's); image_numbers counts the numbers the flights hold. state_free does not touch them: a flight's rows
  // outlive the session handle its conversation has at any one time.
  static constexpr size_t kImgStageRows = 408;  // two numbers' rows, 4,177,920 B: the staging buffer's piece
  bool image_take(int key, size_t rows, int slot);
  void image_fill(int key, size_t row0, size_t n, const uint16_t* src);
  void image_free(int key);
  size_t image_numbers() const;
  // rows_set: the set of row-sized buffers the chunk (and its look-ahead) runs on, 0 the standing set (M <= Options::prefill_rows),
  // 1 the set lent by the pool (M <= prefill_rows_cap(1)); a set this rank does not hold stops the engine with a message.
  void prefill_lookup(int a, int M, int rows_set = 0);
  // The rows a set takes at most (0: this rank holds no such set).
  int prefill_rows_cap(int rows_set) const;
  // The lent set (Options::prefill_rows_max): its size in MiB (0: no such set), the groups it takes, and the pool's read-only answers the
  // policy needs: whether the top `groups` groups of the pool hold no number in use, and the free numbers.
  int64_t prefill_loan_mib() const;
  int prefill_loan_groups() const;
  bool pool_top_free(int groups) const;
  int64_t pool_free_numbers() const;
  // The server's mark on an OnDemand tenant (rank 0's table: next_victim reads it): nothing is using it now.
  void tenant_mark_idle(int id, bool idle);
  // The prefill path's buffers and the weights it gathers, made now instead of by the first prefill:
  // every rank at the same point, after set_engram(). This rank's allocations come before ring's host
  // barrier and the weights' all-gathers after it, so a rank that cannot allocate stops every rank at
  // the barrier. Returns this rank's free device memory afterwards.
  size_t prefill_init(comm::Ring& ring);
  // Before the weights load, when prefill_init will follow: allocates the prefill's gathered wq_a, wkv, wo_b (and
  // their scales), with `engram` the Engram wkv, whole, and the router gate (decode's own whole buffer). gathered_dest(name, bytes) then gives the address in
  // slot `rank` of the tensor `name` of this rank's weight file (nullptr for any other tensor, or when nothing was
  // reserved): the loader reads the tensor there (dsw::LoadOptions::dest), decode uses it from there, and
  // prefill_init gathers only the other ranks' slices into the rest. Without reserve_gathered the weights stay in
  // the arena. aliased(): the tensors and bytes aimed so far.
  void reserve_gathered(bool engram);
  void* gathered_dest(const std::string& name, uint64_t bytes);
  void aliased(size_t* tensors, size_t* bytes) const;
  // ablit: the chunk's session's Sampling::ablit (every chunk carries it; sp only the one that samples)
  uint64_t prefill(int a, int M, const Sampling* sp = nullptr, int handle = -1, const std::vector<int>& snaps = {},
                   bool ced = false, float ablit = 0.f, int rows_set = 0);
  // The last prefill chunk's GPU time in ms (waits for the chunk): [0] the encoder (layers 0-19; every
  // layer without ced), [1] CED's projection, keep and replay, [2] the head and the sampling, [3] the
  // draft module's windows.
  void prefill_split(float ms[4]) const;
  // The last prefill chunk's join clock (DSGB10_PF_STALL; waits for the chunk): [0] the ms the main stream stood at the
  // attention's joins of layers 0-19 (every layer without ced), [1] at the MoE's, [2] one pair's own cost; *pairs the
  // joins timed. False, and nothing set, when the switch is off.
  bool prefill_joins(float ms[3], int* pairs) const;
  // Marks on the model's stream (i < 4), and the GPU time from mark i to mark j (waits for j).
  void mark(int i);
  float mark_ms(int i, int j) const;

  // Sessions' saved state: every rank keeps the same store (rank 0 decides; commands make every rank do
  // the same). Session h holds its compressed caches up to some position, window snapshots (ring
  // positions [pos - 128, pos) of every layer) and an end snapshot (the whole ring and the compressor
  // carries). Copies are queued on the model's stream, in order with the steps.
  // The store's memory is a pool of `bytes` (state_reserve, once, the same on every rank; none: nothing
  // is kept). With the slot memory pool (Options::vmm) the store shares it: state_reserve makes `bytes` more
  // chunks (the pool then holds the slots' and the store's), the store's blocks are mapped chunk by chunk
  // where they lie and the chunks given back to the slots when a reservation or a map finds the pool short
  // (state.cu "the store on the slot pool"; model/vmm.h regions): an idle machine's memory is the store's,
  // a busy one's the slots'. The sessions the store gives up for the slots go to the disk tier, as when it
  // is full. state_capacity() is then the store's size at the start, state_ceiling() all of the pool, and
  // tier_stats().store_cap what the slots have not committed. An operation that finds it full evicts the least recently used session that the current
  // command does not use (state_pin: its operations' sessions and its handle), the same on every rank,
  // and skips what still does not fit: a snapshot not taken, pages saved only as far as they go.
  // state_evicted() returns the sessions evicted since it was last called.
  void state_reserve(size_t bytes);
  size_t state_capacity() const;
  size_t state_ceiling() const;  // the most the store could hold: its pool (shared: every chunk of the slot pool)
  // Chunks the store could give back to the slots without touching the sessions in `keep` (handles): the chunks
  // it has mapped less those under their blocks in RAM. 0 when it does not share the pool.
  size_t state_giveable(const std::vector<int>& keep) const;
  // the same with the conversations of `lines` (handles) counted as if no line held them: the numbers only those conversations
  // name, each once, and the quarter numbers all of whose used slots only they name (one count for every line the request may take)
  size_t state_giveable_joint(const std::vector<int>& keep, const std::vector<int>& lines) const;
  // An upper bound of the chunks bringing session h back from the disk tier can map (0: it is not there, or the store
  // has a pool of its own): each block of its record maps at most ceil(bytes / chunk) + 1 chunks.
  size_t state_restore_chunks(int h) const;
  // the read-backs cut so far at a piece whose sums differed from the ones written (a cut for sums leaves lists not back without a refused take)
  uint64_t state_sum_cuts() const;
  // the numbers h's lists hold in the pool and the lists only on the disk tier (the other numbers state_restore_chunks counts are bookmarks)
  void state_list_numbers(int h, int upto, int* in_pool, int* on_disk) const;
  // and, in `split`, the lists below `upto` that are quarter slots (counted in neither of the others)
  void state_list_numbers(int h, int upto, int* in_pool, int* on_disk, int* split) const;
  // h's window bookmarks by position with the numbers each holds in the pool (0: only on the disk tier)
  std::vector<std::pair<int, int>> state_snap_numbers(int h) const;
  struct StoreChunks {  // the shared budget (zero without it)
    uint64_t total = 0, committed = 0, mapped = 0, idle = 0, live_bytes = 0, trimmed = 0, give_backs = 0;
  };
  StoreChunks state_chunks() const;  // the pool's chunks; the slots' commitments; the store's mapped and idle; the
                                     // bytes of its blocks; the chunks it handed back since the start; how many times
  // h: the sessions the command uses (never evicted); live: the sessions running in the slots (never evicted; a session on
  // an idle or paused line is not named, so it is cache); last: the command's own conversation and its fork base, evicted
  // after every other session but not protected.
  void state_pin(const std::vector<int>& h, const std::vector<int>& live = {}, const std::vector<int>& last = {});
  // Numbers `snaps` window bookmarks would take; numbers a copy on write at pos would take (see state.cu): the server makes
  // room for them before the command.
  int state_bookmark_numbers(int snaps) const;
  int state_cow_numbers(int h_src, int h_own, int pos) const;
  // A hash of the replicated store state, printed by the server under DSGB10_STORE_CHECK=1 and compared across the ranks.
  uint64_t store_digest() const;
  std::vector<int> state_evicted();
  void state_save(int h, int from, int to);    // compressed caches of positions [from, to) -> session h
  void state_load(int h, int to);              // session h's compressed caches of positions [0, to) -> working
  void state_snap(int h, int pos);             // ring -> a snapshot at pos (pos - 128 .. pos - 1 still in the ring)
  void state_snap_end(int h, int pos);         // ring and carries -> session h's end snapshot (at pos)
  void state_restore(int h, int pos);          // snapshot at pos -> ring
  void state_restore_end(int h);               // the end snapshot -> ring and carries
  void state_drop(int h, int from);            // forget what session h holds from position `from` on
  void state_free(int h);
  void state_copy_snaps(int src, int dst, int upto);  // session src's window snapshots at <= upto -> dst
  void state_drop_snap(int h, int pos);
  // Session h's pieces in the pool that have no image on the disk tier get one (an idle conversation: its tail chunk and
  // bookmarks; the full chunks were written at their save). Nothing without a disk tier.
  void state_flush(int h);
  // The disk tier's size while serving, at most the file made at the start (every rank alike; state.cu): false when
  // nothing changed.
  bool state_disk_resize(size_t bytes);
  // The store's disk tier: base.spill of `bytes` (the same on every rank; an earlier file is removed
  // first, and this one at exit), with `staging` bytes of pinned memory for writes in flight. A session
  // the store evicts is then spilled to it: its blocks are copied out after the work queued on the
  // model's stream and freed at once, and a writer thread writes them. state_unspill, or an operation
  // that needs a spilled session's blocks, reads it back, as far as the store has room. When the file is
  // full, its least recently used sessions are dropped (state_evicted).
  // Kept sessions (docs/engine.md): state_keep(gen), on every rank at a planned stop, syncs the file and
  // writes base.index (the sessions in the file: their blocks and checksums, the run id, gen), and the
  // file stays at exit. The next start's state_kept_check reads this rank's index: gen when it is whole,
  // of this run_id (main.cpp: the binary, the weights, the options), at most max_age_s old and beside
  // its file (whose size goes to *bytes), else 0 (*why: the reason, "" when there is none). With reuse
  // (every rank's check returned the same gen) state_disk opens that file as it is, and its sessions
  // are spilled ones, and the index stays: it is the latest checkpoint until the next (its images keep
  // their space); without reuse it is removed. A piece that comes back different from what went out
  // cuts its session at the same position on every rank (state_unspill).
  // state_checkpoint(gen): the same index while serving (rank 0's server, when nothing runs; every rank
  // alike), naming every session as far as its pieces have images, so that a crash comes back to it;
  // state_checkpointed: the sessions the last one named on this rank (the server writes theirs);
  // state_version: images written or dropped since the start (a checkpoint is due when it moved).
  uint64_t state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes, std::string* why);
  void state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse = false, uint64_t run_id = 0);
  void state_keep(uint64_t gen);
  void state_checkpoint(uint64_t gen);
  std::vector<int> state_checkpointed() const;
  uint64_t state_version() const;
  uint64_t state_kept() const;              // the gen whose sessions this run's store holds (0: none)
  std::vector<int> state_sessions() const;  // the sessions in the store
  size_t state_disk_capacity() const;
  size_t state_disk_bytes() const;  // the file's bytes in spilled sessions
  std::string state_disk_path() const;
  void state_spill(int h);    // session h's state to the disk tier (freed without one: state_evicted)
  // and back; `max_numbers` > 0: at most that many numbers are taken (the lists first, then bookmarks): store_core.h unspill
  // `upto` > 0 makes it one unit of a return: the pieces below `upto` only, at most Options::kv_join_pieces of them, and `max_numbers` < 0
  // says the budget is spent (it cuts where the count ended); `upto` 0: the whole conversation in one command.
  void state_unspill(int h, int max_numbers = 0, int upto = 0);
  int state_unspill_left(int h, int upto) const;  // what a read-back to `upto` still brings (lists on the disk below it, bookmarks at or below it)
  // The split (--kv-split cache). state_split: at most `budget` pieces of h's whole pieces become quarter slots, never evicting.
  // state_join: the pieces below position `upto` that are slots become whole numbers again, at most `max_numbers` taken (0: no
  // limit) and at most Options::kv_join_pieces pieces in this command; every rank cuts at the smallest position whose piece arrived
  // different. Both are commands (cmd.h kStSplit, kStJoin); the counters below are read by rank 0's STATS and its pacing.
  void state_split(int h, int budget);
  void state_join(int h, int upto, int max_numbers);
  void state_join_plan(int h, int upto, int* back, int* exch) const;  // what a join to `upto` would do: pieces taken back, pieces exchanged
  struct SplitStats {
    uint64_t split_pieces = 0, split_deferred = 0, revives = 0, joins = 0, join_bytes = 0, join_ns = 0, join_max_ns = 0;
    uint64_t join_short = 0, join_bad = 0, exchanged = 0, join_taken = 0, unspill_taken = 0;
    uint64_t conv_split = 0, quarter_slots = 0, quarter_numbers = 0, ghost_numbers = 0;
    // the imager
    uint64_t imaged = 0, image_voids = 0, image_waits = 0, returns_read = 0, rides = 0, clean_numbers = 0, image_pending = 0;
  };
  SplitStats split_stats() const;
  // image: at most `budget` of h's plain pieces get a quarter image queued on the image stream (live: the conversations a request is in flight
  // on get nothing). settle: the commands at or below `upto` are made true and the idle conversations marked; returns the pieces marked.
  void ride_readings(uint64_t out[4]) const;  // the replicated readings a ride checks across the ranks: image commands, queued jobs, ghosts, free numbers
  bool has_host_ring() const;                 // the engine has its ring: a ride without one in the server would be one rank deciding alone
  void state_ride(uint64_t settled, int h, int budget, const std::vector<int>& live);  // settle up to `settled`, then image (h > 0)
  int image_room_pieces(int maxn) const;  // the most pieces (up to maxn) staging takes now
  uint64_t image_pending() const;  // image jobs queued and not settled
  int image_todo(int h) const;     // pieces of h with no image and none queued
  uint64_t image_done() const;  // this rank's reading: the highest command whose work on the image stream has ended (no wait)
  bool image_room(size_t bytes) const;
  uint64_t image_seq() const;   // image commands queued so far
  bool imager() const;
  bool split_can_open() const;  // a split command could open a quarter number now (a replicated count: no command, no collective)
  // The lend zone (chunks.h): the numbers a window or the Engram lender will take from the top. A collective: every rank calls alike,
  // at the start and inside a window's open and close; a no-op without Options::kv_split.
  void split_zone_update();
  void split_prepare();  // the split's buffers, before the free-memory read the store is sized from (main.cpp)

  struct StateInfo {
    int saved = 0;           // compressed caches of positions [0, saved) are kept
    int end_pos = -1;        // the end snapshot's position (-1: none)
    std::vector<int> snaps;  // window snapshot positions, ascending
    size_t bytes = 0;        // in the store's device pool
    bool on_disk = false;    // spilled: its blocks are in the disk tier (disk_bytes), none in the pool
    size_t disk_bytes = 0;
    int restart = -1;        // the latest position it can restart from with what it holds (state.cu restart_of); -1: none
    bool split = false;      // a piece of it is a quarter slot (kStJoin brings it back)
    int slots = 0;           // the quarter slots it names, lists and bookmarks
  };
  StateInfo state_info(int h) const;
  // The YaRN factor session h's stored state is rotated with (16, 32, 64; 16 when unknown). Set once, when a slot takes the
  // session (server kStFactor); a session that holds state under another factor is a fatal mismatch.
  void state_set_factor(int h, int factor);
  int state_factor(int h) const;
  size_t state_bytes() const;                  // device memory the store holds
  // The store and its disk tier for rank 0's dashboard (serve.cpp STATS), with their counters since the start: sessions
  // written to the disk tier and read back, the positions restored, the time requests waited for them.
  struct TierStats {
    uint64_t store_cap = 0, store_used = 0, resident = 0, on_disk = 0, disk_cap = 0, disk_used = 0;
    uint64_t spills = 0, spill_bytes = 0, restores = 0, restore_bytes = 0, restored_pos = 0, wait_ns = 0, wait_max_ns = 0;
  };
  TierStats tier_stats() const;
  // The chunk pool for rank 0's dashboard (the StatsFrame, common/frontend_protocol.h): numbers in all, free, held by conversations,
  // shared by more than one, reserved for the running requests' horizons and not yet held; one number's bytes; the
  // conversations on lines, idle in memory, in the disk tier, and those naming a shared number.
  struct PoolStats {
    uint64_t numbers = 0, free = 0, held = 0, shared = 0, reserved = 0, number_bytes = 0, image_numbers = 0;
    uint64_t on_lines = 0, in_memory = 0, on_disk = 0, sharing = 0;
  };
  PoolStats pool_stats() const;
  // Wait for everything queued.
  void drain();

  // launch + wait: sampled [R] (host) gets the chosen token of each row, the same everywhere.
  void step(const Ctl& c, int32_t* sampled, Sink* sink = nullptr);
  // Forget the sequence (positions restart at 0); a history capacity above 1 Mi ids is given back.
  void reset();
  // Cut every unit's token history to its last 4,096 ids, except a unit whose prompt has not been prefilled up to its
  // end. Between commands only (no step in flight reads a history below that); hashing and a rewind never go below it.
  void trim_history();
  // Bench only: the rows of a step of R rows ending at P = pos0 + R - 1 sit at P - (R - 1 - r) * stride
  // instead of P - (R - 1 - r) (1, the default), so stride >= 128 puts each row's window apart. With
  // group G > 1 they come in runs of G consecutive positions (a session's verify rows), the runs stride
  // apart: row r at P - (d / G) * stride - d % G, d = R - 1 - r. The state the steps leave is not a
  // sequence's; use it for timing only.
  void set_pos_stride(int stride, int group = 1);
  // Bench only: row r's router logits rotated by shift * (r / group) experts before the top-6 (0, the
  // default: none), so the rows (or runs of group rows) of a step route about as apart as rows of
  // unrelated sequences could. Drops the captured graphs (they are captured again, with or without it).
  void set_route_shift(int shift, int group = 1);
  // The GPU time of the last step waited for, from its first kernel to its last.
  float last_gpu_ms() const;
  // The same of step n, one of the last ops::kOutSlots launched (waits for it to finish).
  float gpu_ms(uint64_t n) const;
  // The value that won row r's sampling in the last step waited for (comparing runs bit for bit).
  float last_logit(int r) const;
  // Step graphs instantiated so far.
  int graph_count() const;

#ifdef DSGB10_DEV  // development builds only (engine/model/CMakeLists.txt DSGB10_DEV)
  // stack4 (the certified head's counters; false: off): steps, rows checked against the full head, mismatches,
  // surviving CTAs (sum), the largest |coarse - exact| / bound seen (float bits), the most CTAs of a step, forced steps
  bool cert_stats(uint32_t out[8]) const;
  // stack4 (DSGB10_MH_DUMP): step n's rows of main_hidden [kMaxSessRows][15360] bf16 in pinned memory (valid once
  // wait(n) returned, until step n + 4 runs); null: off or not this rank
  const uint16_t* mh_dump(uint64_t n) const;
#endif

  int rank() const;
  cudaStream_t stream() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

}  // namespace model
