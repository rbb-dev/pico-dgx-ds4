// Rank 0's commands and what every rank runs alike (cmd.h).
#include <cstdlib>
#include "server/cmd.h"
#include "server/ride_wire.h"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "common/check.h"
#include "common/heap.h"
#include "model/copy.h"
#include "model/nvtx.h"
#include "server/mem_watch.h"

namespace srv {

std::vector<double> g_t_wait, g_t_late, g_t_launch;

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void pin_self(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

Watchdog::Watchdog(int timeout_s, comm::Ring& ring) {
  th = std::thread([this, timeout_s, &ring] {
    while (!stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      const double t = started;
      if (t > 0 && now_s() - t > timeout_s) {
        std::fprintf(stderr, "watchdog: step running for %d s, aborting\n%s\n", timeout_s, ring.describe().c_str());
        std::fflush(stderr);
        _exit(3);
      }
    }
  });
}

Watchdog::~Watchdog() {
  stop = true;
  th.join();
}

namespace {
// The watch's state: every rank has the ring; only rank 0 keeps readings.
struct MemWatchState {
  comm::Ring* ring = nullptr;
  memwatch::Watch w;
  std::string over;
  uint64_t seq = 0;  // watches read by rank 0
};
MemWatchState g_watch;

// One watch, on every rank at the same point of the command stream: this rank's reading, the host all-gather, rank 0's report.
void mem_watch_run(int rank) {
  if (!g_watch.ring || g_watch.w.keep == 0) return;
  memwatch::Sample mine{memwatch::read_mem_available(), heap::in_use()}, all[comm::kRanks];
  g_watch.ring->allgather_host(&mine, all, sizeof mine, comm::kCallWatch);
  if (rank != 0) return;
  const bool report = g_watch.w.observe(all, comm::kRanks);
  g_watch.seq++;
  if (!report) return;
  std::string over = g_watch.over;
  for (int r = 0; r < comm::kRanks; ++r)
    if (all[r].heap > heap::kCap) {
      char b[128];
      std::snprintf(b, sizeof b, "%srank %d glibc heap %.2f GiB > %.2f GiB", over.empty() ? "" : "; ", r, all[r].heap / 1073741824.0,
                    heap::kCap / 1073741824.0);
      over += b;
    }
  std::printf("%s\n", memwatch::low_line(g_watch.w.min_rank, g_watch.w.min_bytes, g_watch.w.keep, over).c_str());
  std::fflush(stdout);
}
}  // namespace

namespace {
// The imager's ride (cmd.h): rank 0's request and the ride count; the ring is every rank's.
struct SplitRideState {
  comm::Ring* ring = nullptr;
  RideReq req;
  uint64_t rides = 0, longest_ns = 0;
};
SplitRideState g_ride;
static_assert(ridewire::kLive == model::kMaxSlots && ridewire::kRanks == comm::kRanks, "the ride's wire is as wide as the engine's slot list and ring");
using RideWire = ridewire::Wire;
constexpr int kRideRoomProbe = 64;
}  // namespace

void split_ride_init(comm::Ring* ring) { g_ride.ring = ring; }
void split_ride_set(const RideReq& r) { g_ride.req = r; }
uint64_t split_ride_count() { return g_ride.rides; }
void split_ride_take_longest(uint64_t* ns) {
  *ns = g_ride.longest_ns;
  g_ride.longest_ns = 0;
}

void split_ride_run(model::Model& m, const RideReq& mine_req) {
  const double t_ride = now_s();
  RideWire mine{}, all[comm::kRanks];
  mine.rank = m.rank();
  mine.done = m.image_done();
  mine.room = m.image_room_pieces(kRideRoomProbe);
  uint64_t rd[4];
  m.ride_readings(rd);  // the replicated readings the ride acts on: every rank must hold the same
  mine.img_seq = rd[0], mine.jobs = rd[1], mine.ghosts = rd[2], mine.free_numbers = rd[3];
  if (m.rank() == 0) {
    mine.h = mine_req.h, mine.budget = mine_req.budget;
    std::copy(mine_req.live, mine_req.live + model::kMaxSlots, mine.live);
  }
  int n = comm::kRanks;
  if (g_ride.ring) {
    g_ride.ring->allgather_host(&mine, all, sizeof mine, comm::kCallRide);
  } else {
    // no ring is a model that has no host ring (one rank, a tool); on an engine with a ring a ride without it would be one rank deciding alone
    if (m.has_host_ring()) die("ride: rank %d has no ring for the ride's all-gather", m.rank());
    all[0] = mine, n = 1;
  }
  ridewire::Decision dc;
  const std::string bad = ridewire::decide(all, n, &dc);
  if (!bad.empty()) die("ride: rank %d: %s", m.rank(), bad.c_str());
  std::vector<int> live;
  for (const int32_t h : dc.live)
    if (h > 0) live.push_back(h);
  m.state_ride(dc.settled, dc.h, dc.budget, live);
  if (const char* e = std::getenv("DSGB10_STORE_CHECK"); e && std::atoi(e) == 1)
    std::printf("store-check: rank %d ride digest %016llx\n", m.rank(), (unsigned long long)m.store_digest());
  if (m.rank() == 0) {
    ++g_ride.rides;
    g_ride.longest_ns = std::max<uint64_t>(g_ride.longest_ns, (uint64_t)((now_s() - t_ride) * 1e9));
  }
}

void mem_watch_init(comm::Ring* ring, uint64_t keep_bytes) {
  g_watch.ring = ring;
  g_watch.w.keep = keep_bytes;
}
void mem_watch_note(const std::string& over) { g_watch.over = over; }
bool mem_watch_on() { return g_watch.ring && g_watch.w.keep > 0; }
MemWatchView mem_watch_view() {
  MemWatchView v;
  v.min_bytes = g_watch.w.min_bytes, v.low_count = g_watch.w.low_count;
  for (int r = 0; r < 4; ++r) v.avail[r] = g_watch.w.last[r];
  v.seq = g_watch.seq;
  return v;
}

void apply_state(model::Model& m, const StateOp& o) {
  switch (o.op) {
    case kStSave: m.state_save(o.h, o.a, o.b); break;
    case kStSnapEnd: m.state_snap_end(o.h, o.a); break;
    case kStLoad: m.state_load(o.h, o.a); break;
    case kStRestore: m.state_restore(o.h, o.a); break;
    case kStRestoreEnd: m.state_restore_end(o.h); break;
    case kStDrop: m.state_drop(o.h, o.a); break;
    case kStFree: m.state_free(o.h); break;
    case kStCopySnaps: m.state_copy_snaps(o.h, o.a, o.b); break;
    case kStDropSnap: m.state_drop_snap(o.h, o.a); break;
    case kStSnap: m.state_snap(o.h, o.a); break;
    case kStSpill: m.state_spill(o.h); break;
    case kStUnspill: m.state_unspill(o.h, o.a, o.b); break;
    case kStKeep: m.state_keep((uint64_t)(uint32_t)o.a | (uint64_t)(uint32_t)o.b << 32); break;
    case kStReserve: m.slot_reserve(o.a, o.b, o.h); break;
    case kStRelease: m.slot_release(o.a); break;
    case kStPin: break;  // apply_ops pinned its session
    case kStAttach: m.slot_attach(o.a, o.h, o.b > 0 ? o.b : INT32_MAX); break;
    case kStFlush: m.state_flush(o.h); break;
    case kStDiskSize: m.state_disk_resize((size_t)(uint32_t)o.a << 16); break;
    case kStFactor:
      m.set_slot_factor(o.a, o.b);
      if (o.h > 0) m.state_set_factor(o.h, o.b);
      break;
    case kStCheckpoint:
      m.state_checkpoint((uint64_t)(uint32_t)o.a | (uint64_t)(uint32_t)o.b << 32);
      heap::release();  // the index's buffers are freed: their pages go back
      break;
    case kStUnreserve: m.slot_unreserve(o.a); break;
    case kStImgTake:
      if (!m.image_take(o.h, (size_t)(uint32_t)o.a, o.b)) die("state: no room for the image rows of flight %d (%d rows), or the flight holds them already", o.h, o.a);
      break;
    case kStImgFree: m.image_free(o.h); break;
    case kStSplit: m.state_split(o.h, o.a); break;
    case kStJoin: m.state_join(o.h, o.a, o.b); break;
    case kStLast: break;  // apply_ops passed its session as the evict-last list
    case kStMemWatch: mem_watch_run(m.rank()); break;
    case kStRide: die("state: a ride is a command of its own");
    default: die("state: unknown operation %d", o.op);
  }
}

namespace {
// DSGB10_STORE_CHECK=1: every rank prints a digest of the replicated store state after a window or state command; the
// four ranks' logs must show the same value on the same line (a rank-local input to a replicated decision shows as a split).
bool store_check() {
  static const bool on = [] {
    const char* e = std::getenv("DSGB10_STORE_CHECK");
    return e && std::atoi(e) != 0;
  }();
  return on;
}
}  // namespace

void apply_ops(model::Model& m, const Cmd& c) {
  // a command of the watch alone runs the watch and touches nothing of the store: no history trim, no pin, no digest line
  if (c.op == kCmdState && memwatch::watch_only(c.n_ops, [&](int i) { return c.ops[i].op; }, kStMemWatch)) {
    for (int i = 0; i < c.n_ops; ++i) apply_state(m, c.ops[i]);
    return;
  }
  // the imager's ride with nothing in flight: no trim, no pin of its own (the sessions a request is in flight on are `live`), no digest line
  if (c.op == kCmdState && c.n_ops == 1 && c.ops[0].op == kStRide) {
    std::vector<int> live;
    for (const int32_t h : c.live)
      if (h > 0) live.push_back(h);
    m.state_pin({}, live, {});
    RideReq r;
    r.h = c.ops[0].h, r.budget = c.ops[0].a;
    std::copy(c.live, c.live + model::kMaxSlots, r.live);
    split_ride_run(m, r);
    return;
  }
  if (c.slot >= 0) m.use_slot(c.slot);
  m.trim_history();
  std::vector<int> pin, last;
  for (int i = 0; i < c.n_ops; ++i) {
    if (c.ops[i].op == kStLast) {  // evicted after every other session, not protected
      last.push_back(c.ops[i].h);
      continue;
    }
    if (c.ops[i].op == kStUnreserve || c.ops[i].op == kStReserve || c.ops[i].op == kStImgTake || c.ops[i].op == kStImgFree ||
        c.ops[i].op == kStMemWatch || c.ops[i].op == kStSplit)
      continue;  // names a slot or a flight, or nothing: no session
    pin.push_back(c.ops[i].h);
    if (c.ops[i].op == kStCopySnaps) pin.push_back(c.ops[i].a);
  }
  if (c.handle >= 0) pin.push_back(c.handle);
  std::vector<int> live;
  for (const int32_t h : c.live)
    if (h > 0) live.push_back(h);
  m.state_pin(pin, live, last);
  for (int i = 0; i < c.n_ops; ++i) apply_state(m, c.ops[i]);
  if (c.n_ops > 0 && store_check())
    std::printf("store-check: rank %d state command (%d ops, first %d) digest %016llx\n", m.rank(), c.n_ops, c.ops[0].op,
                (unsigned long long)m.store_digest());
}

Cmd bcast(comm::Ring& ring, const Cmd& mine) {
  Cmd all[4];
  ring.allgather_host(&mine, all, sizeof(Cmd), comm::kCallFrame);
  return all[0];
}

void bcast_payload(comm::Ring& ring, void* data, size_t nbytes, int rank) {
  constexpr size_t kPiece = 65536;  // allgather_host's limit
  std::vector<uint8_t> all(4 * kPiece), mine(kPiece);
  for (size_t off = 0; off < nbytes; off += kPiece) {
    const size_t n = std::min(kPiece, nbytes - off);
    if (rank == 0) std::memcpy(mine.data(), static_cast<uint8_t*>(data) + off, n);
    ring.allgather_host(mine.data(), all.data(), n, comm::kCallPayload);
    if (rank != 0) std::memcpy(static_cast<uint8_t*>(data) + off, all.data(), n);
  }
}

model::Ctl to_ctl(const Cmd& c) {
  model::Ctl t{};
  t.op = model::kOpStep;
  t.rows = c.rows;
  t.pos0 = c.pos0;
  t.flags = c.flags;
  std::memcpy(t.tokens, c.tokens, sizeof t.tokens);
  t.temperature = c.sp.temperature;
  t.seed = (uint32_t)c.sp.seed;
  return t;
}

int run_window(model::Model& m, const Cmd& c) {
  const int st = (c.pos0 == 1 || c.rows == 0) ? m.window_close(c.rows) : m.window_open(c.rows);
  if (store_check())
    std::printf("store-check: rank %d window command (factor %d, %s) status %d digest %016llx\n", m.rank(), c.rows,
                c.pos0 == 1 ? "close" : "open", st, (unsigned long long)m.store_digest());
  if (store_check()) std::printf("store-check: rank %d tenant table %s\n", m.rank(), m.tenant_table_ok() ? "adds up" : "BROKEN");
  return st;
}

int run_vit(model::Model& m, const Cmd& c, const uint8_t* pixels) {
  return m.vision_run(c.vit_key, c.vit_span, c.vit_row0, c.vit_nh, c.vit_nw, c.vit_b0, c.vit_b1, pixels);
}

int run_lend(model::Model& m, const Cmd& c) {
  const int st = m.tenant_lend(c.tenant, c.rows, c.pos0);
  if (store_check()) {
    if (c.tenant == 1)
      std::printf("store-check: rank %d lend command (phase %d, %d MiB) status %d digest %016llx\n", m.rank(), c.rows, c.pos0, st,
                  (unsigned long long)m.store_digest());
    else
      std::printf("store-check: rank %d lend command of tenant %d (phase %d, %d MiB) status %d digest %016llx\n", m.rank(), c.tenant, c.rows, c.pos0, st,
                  (unsigned long long)m.store_digest());
    std::printf("store-check: rank %d tenant table %s\n", m.rank(), m.tenant_table_ok() ? "adds up" : "BROKEN");
  }
  return st;
}

uint64_t run_prefill(model::Model& m, const Cmd& c) {
  const std::vector<int> snaps(c.snaps, c.snaps + c.n_snaps);
  const uint64_t n =
      m.prefill(c.pos0, c.rows, (c.flags & kPfSample) ? &c.sp : nullptr, c.handle, snaps, (c.flags & kPfCed) != 0,
                c.sp.ablit, c.rows_set);
  if (c.next > 0) m.prefill_lookup(c.pos0 + c.rows, c.next, c.rows_set);
  if (c.flags & kPfWatch) mem_watch_run(m.rank());  // the chunk is queued: the reading and the all-gather go beside the device's work
  return n;
}

uint64_t run_rows(model::Model& m, const Cmd& c) {
  uint64_t n;
  if (m.spec()) {  // the rows go into the draft module's windows too
    model::SpecCfg sc;
    sc.max_drafts = 0;
    sc.pos_end = c.pos0 + c.rows + 1;
    sc.force = c.spec_force;
    std::copy(c.spec_probe, c.spec_probe + model::ops::kDrafts, sc.probe);
    n = m.launch_spec(c.rows, c.pos0, c.tokens, c.sp, sc);
  } else {
    n = m.launch(c.rows, c.pos0, c.tokens, c.sp);
  }
  for (int i = 0; i < c.n_snaps; ++i) m.state_snap(c.handle, c.snaps[i]);
  return n;
}

void SpecRate::add(const Cmd& c, int emitted, int rows) {
  constexpr double a = 1.0 / 16;
  if (c.spec_gate != model::ops::kGateThresh) {
    if (ms == 0) ms = c.spec_row_ms[0];
    tok += a * (emitted - tok);
    ms += a * (c.spec_row_ms[std::clamp(rows - 1, 0, model::ops::kDrafts)] - ms);
    return;
  }
  if (ms == 0) ms = c.spec_cost[0] + c.spec_cost[2];
  tok += a * (emitted - tok);
  ms += a * (c.spec_cost[0] + (rows - 1) * c.spec_cost[1] + c.spec_cost[2] - ms);
}

float SpecRate::thresh(const Cmd& c) const {
  if (c.spec_gate != model::ops::kGateThresh) return (float)(tok / (ms > 0 ? ms : c.spec_row_ms[0]));
  if (c.spec_thresh >= 0) return c.spec_thresh;
  const double m = ms > 0 ? ms : c.spec_cost[0] + c.spec_cost[2];
  return (float)(tok / m * c.spec_cost[1]);
}

namespace {

// DSGB10_COPY_DUMP (copy.h; DSGB10_COPY_DUMP_FILE=FILE, on rank 0's server): a line per step and entry with a proposal
// record: what the step emitted and the proposal made after its kept drafts (source end, match length, drafts kept,
// proposed tokens); tools/copy/fit.py scores them. Tokens are ';'-separated in a field.
FILE* copy_dump_file() {
  static FILE* const f = [] {
    const char* v = getenv("DSGB10_COPY_DUMP_FILE");
    if (!(model::copy::mode() & model::copy::kDump) || !v || !*v) return (FILE*)nullptr;
    FILE* g = std::fopen(v, "a");
    if (g && std::ftell(g) == 0)
      std::fprintf(g, "tag,step,entry,pos0,emitted_n,emitted,next_pos0,chain_k,chain,src,match_len,m,copy\n");
    return g;
  }();
  return f;
}
void copy_dump_line(FILE* f, int step, int entry, int pos0, int n_emit, const int32_t* emitted, int next_pos0, const int32_t* next_tok,
                    const int32_t* info, const int32_t* tok) {
  static const char* const tag = getenv("DSGB10_COPY_TAG") ? getenv("DSGB10_COPY_TAG") : "";
  std::fprintf(f, "%s,%d,%d,%d,%d,", tag, step, entry, pos0, n_emit);
  for (int j = 0; j < n_emit; ++j) std::fprintf(f, "%s%d", j ? ";" : "", emitted[j]);
  std::fprintf(f, ",%d,%d,", next_pos0, info[2]);
  for (int j = 1; j <= info[2]; ++j) std::fprintf(f, "%s%d", j > 1 ? ";" : "", next_tok[j]);
  std::fprintf(f, ",%d,%d,%d,", info[0], info[1], info[3]);
  for (int j = 0; j < info[3]; ++j) std::fprintf(f, "%s%d", j ? ";" : "", tok[j]);
  std::fprintf(f, "\n");
}

// decode_loop with DSpark (cmd.h)
DecodeEnd spec_loop(model::Model& m, const Cmd& c, Watchdog& wd, const OnSample& on, const OnStep& step) {
  constexpr int kD = model::ops::kDrafts;
  // rows a step can reach past the previous step's last: the chain's drafts, or copy drafting's (copy.h) when copy rows are on
  static const int kR = model::copy::verify_reach(kD, (model::copy::mode() & model::copy::kRows) != 0);
  DecodeEnd e;
  e.written = c.pos0;
  // verify rows stay below pos_end: the tokens they emit stay within max_new and the context. As in
  // plain decode, the step after the last token still runs (up to `feed`), so the KV of every emitted
  // token is there for the session's next request.
  const int pos_end = (int)std::min<int64_t>(c.limit, (int64_t)c.pos0 + c.max_new);
  const int feed = std::min(pos_end, c.limit - 1);  // the last position a step may start at
  e.spec_tok = c.spec_tok, e.spec_ms = c.spec_ms;
  if (c.max_new < 1 || c.pos0 >= c.limit) {
    e.why = c.max_new < 1 ? kEndLength : kEndContext;
    return e;
  }
  SpecRate rate{c.spec_tok, c.spec_ms};
  model::SpecCfg cfg;
  cfg.max_drafts = c.spec_drafts;
  cfg.pos_end = pos_end;
  cfg.gate = c.spec_gate;
  std::copy(c.spec_row_ms, c.spec_row_ms + model::ops::kDrafts + 1, cfg.step_ms);
  // DSGB10_RELAX_THRESH=T: a greedy one-session decode verifies drafts while survival >= T (fixed), the gate
  // that pairs with DSGB10_RELAX's relaxed accept (sample.cu); sampled requests and batch steps keep the default gate
  static const float relax_thresh = [] {
    const char* v = getenv("DSGB10_RELAX_THRESH");
    return v ? (float)atof(v) : -1.f;
  }();
  const bool rgate = relax_thresh >= 0.f && !(c.sp.temperature > 0.f);
  // DSGB10_CF_ALPHA=A scales the one-session gate's threshold (the confidence MLP's calibration differs from the
  // shipped head's; a threshold only decides which drafts are verified, never a token)
  static const float cf_alpha = getenv("DSGB10_CF_ALPHA") ? (float)atof(getenv("DSGB10_CF_ALPHA")) : 1.f;
  auto gate_of = [&] { return rgate ? relax_thresh : rate.thresh(c) * cf_alpha; };
  auto verify = [&](int bound) {
    cfg.thresh = gate_of();
    e.gate(cfg.thresh);
    return m.launch_verify(c.sp, cfg, std::min(bound, c.limit - 1));
  };
  std::vector<int32_t> toks{c.tokens[0]};  // tok_0, tok_1, ...: positions pos0, pos0 + 1, ...
  int next_snap = c.snap_every > 0 ? (c.pos0 / c.snap_every + 1) * c.snap_every : INT32_MAX;
  cfg.thresh = gate_of();
  e.gate(cfg.thresh);
  static FILE* const slog = [] {
    const char* v = getenv("DSGB10_STEP_LOG");
    return v && *v ? std::fopen(v, "a") : (FILE*)nullptr;
  }();
  static FILE* const mhf = [&] {  // (DSGB10_MH_DUMP=DIR on the rank that holds the pinned ring): DIR/mh.bin
    const char* v = getenv("DSGB10_MH_DUMP");
    if (!v || !*v || !m.mh_dump(0)) return (FILE*)nullptr;
    std::string p = v;
    if (p.rfind("~/", 0) == 0 && getenv("HOME")) p = std::string(getenv("HOME")) + p.substr(1);
    return std::fopen((p + "/mh.bin").c_str(), "ab");
  }();
  if (mhf) {
    const int32_t hd[4] = {-1, c.pos0, c.max_new, 0};
    std::fwrite(hd, 4, 4, mhf);
  }
  std::vector<int32_t> slog_in{c.tokens[0]};
  if (slog) std::fprintf(slog, "{\"req\":1,\"pos0\":%d,\"max_new\":%d,\"temp\":%.3f,\"n_top\":%d}\n", c.pos0, c.max_new, c.sp.temperature, c.sp.n_top);
  const uint64_t first = m.launch_spec(1, c.pos0, c.tokens, c.sp, cfg);
  int launched = 1, prev_rows = 1;
  if (c.ahead && c.pos0 + 1 <= feed) verify(c.pos0 + 1 + kR), ++launched;
  // The watch riding the run: the step whose output shows the bit's other value runs the reading on every rank, between two steps.
  uint32_t watch_seen = 0, split_seen = 0;
  auto ride = [&](uint32_t ctl) {
    // a ride is watched as a step is (only when the bit flipped: no cost on a step without one): a rank that stops inside it ends with the watchdog's line
    if (memwatch::flipped(watch_seen, ctl)) wd.started = now_s(), mem_watch_run(m.rank()), wd.started = 0;
    if (memwatch::split_flipped(split_seen, ctl)) wd.started = now_s(), split_ride_run(m, g_ride.req), wd.started = 0;
  };
  for (int i = 0;; ++i) {
    wd.started = now_s();
    const model::SampleOut& o = m.wait(first + i);
    const uint32_t wctl = o.ctl;
    wd.started = 0;
    g_t_wait.push_back(now_s());
    if (i + 1 < launched) m.engram_late(o.next_pos0, o.next_rows, o.next_tok);  // step i + 1's rows
    if (mhf) {  // the step's rows of main_hidden
      const int32_t hd[4] = {i, o.pos0, o.step_rows, o.rows};
      std::fwrite(hd, 4, 4, mhf);
      std::fwrite(m.mh_dump(first + i), 2 * 15360, (size_t)std::max(0, std::min(o.step_rows, 8)), mhf);
    }
    if (slog) {  // DSGB10_STEP_LOG, measurement only:: every row's sample, the rejected rows' too
      std::fprintf(slog, "{\"i\":%d,\"p\":%d,\"R\":%d,\"n\":%d,\"in\":[", i, o.pos0, o.step_rows, o.rows);
      for (size_t j = 0; j < slog_in.size(); ++j) std::fprintf(slog, "%s%d", j ? "," : "", slog_in[j]);
      std::fprintf(slog, "],\"out\":[");
      for (int j = 0; j < o.step_rows; ++j) std::fprintf(slog, "%s%d", j ? "," : "", o.row[j].token);
      std::fprintf(slog, "],\"lp\":[");
      for (int j = 0; j < o.step_rows; ++j) std::fprintf(slog, "%s%.4f", j ? "," : "", o.row[j].logprob);
      std::fprintf(slog, "],\"np\":%d,\"nt\":[", o.next_pos0);
      for (int j = 0; j < o.next_rows; ++j) std::fprintf(slog, "%s%d", j ? "," : "", o.next_tok[j]);
      std::fprintf(slog, "],\"d\":[");
      for (int j = 0; j < kD; ++j) std::fprintf(slog, "%s%d", j ? "," : "", o.drafts[j]);
      std::fprintf(slog, "],\"s\":[");
      for (int j = 0; j < kD; ++j) std::fprintf(slog, "%s%.5f", j ? "," : "", o.survival[j]);
      std::fprintf(slog, "]");
      static const bool slog_cf = getenv("DSGB10_CF_MLP") != nullptr;
      if (slog_cf) {
        std::fprintf(slog, ",\"cf\":[");
        for (int j = 0; j < 15; ++j) std::fprintf(slog, "%s%.4f", j ? "," : "", o.cf[j]);
        std::fprintf(slog, "]");
      }
      if (c.sp.n_top > 0) {
        std::fprintf(slog, ",\"ti\":[");
        for (int j = 0; j < o.step_rows; ++j) {
          std::fprintf(slog, "%s[", j ? "," : "");
          for (int k = 0; k < c.sp.n_top; ++k) std::fprintf(slog, "%s%d", k ? "," : "", o.top_id[j][k]);
          std::fprintf(slog, "]");
        }
        std::fprintf(slog, "],\"tl\":[");
        for (int j = 0; j < o.step_rows; ++j) {
          std::fprintf(slog, "%s[", j ? "," : "");
          for (int k = 0; k < c.sp.n_top; ++k) std::fprintf(slog, "%s%.3f", k ? "," : "", o.top_lp[j][k] > -1e30f ? o.top_lp[j][k] : -99.f);
          std::fprintf(slog, "]");
        }
        std::fprintf(slog, "]");
      }
      std::fprintf(slog, "}\n");
      slog_in.assign(o.next_tok, o.next_tok + std::max(0, std::min<int>(o.next_rows, model::kMaxRows)));
    }
    g_t_late.push_back(now_s());
    if (on && copy_dump_file()) {
      int32_t em[model::kMaxRows];
      const int ne = std::max(0, std::min<int>(o.rows, model::kMaxRows));
      for (int j = 0; j < ne; ++j) em[j] = o.row[j].token;
      copy_dump_line(copy_dump_file(), i, 0, o.pos0, ne, em, o.next_pos0, o.next_tok, o.copy_info[0], o.copy_tok[0]);
      std::fflush(copy_dump_file());
    }
    if (model::copy::mode() & model::copy::kAdapt) e.copy_obs(o.copy_obs[0], o.copy_chance);
    e.written = std::max(e.written, o.pos0 + std::max(o.step_rows, o.rows + kR));
    if (i > 0) {
      ++e.vsteps, e.verified += o.step_rows - 1, e.accepted += o.rows - 1;
      ++e.rows_hist[std::min(o.step_rows - 1, kD)];
    }
    if (i > 4)  // the step before (done; past the graphs' captures): its GPU time by its rows
      e.gpu_ms[prev_rows] += m.gpu_ms(first + i - 1), ++e.gpu_n[prev_rows];
    prev_rows = o.step_rows;
    rate.add(c, o.rows, o.step_rows);
    e.spec_tok = rate.tok, e.spec_ms = rate.ms;
    // the tokens it emits, until a stop
    bool stop = false;
    int u = 0;
    for (int j = 0; j < o.rows && !stop; ++j) {
      const int32_t t = o.row[j].token;
      if (on) on(e.sampled, o, j);
      ++e.sampled, ++u;
      toks.push_back(t);
      if (memwatch::ends_reply(o.ctl)) stop = true, e.why = kEndCancel;
      for (int k = 0; !stop && k < c.n_stop; ++k)
        if (t == c.stop[k]) stop = true, e.why = kEndStop;
      if (!stop && e.sampled >= c.max_new) stop = true, e.why = kEndLength;
    }
    if (!stop && o.next_pos0 >= pos_end) stop = true, e.why = kEndContext;
    // positions o.pos0 .. F hold their final KV; the history now ends at H (F + 1: the token after them,
    // whose KV the next step's first row makes)
    const int F = o.pos0 + o.rows - 1, H = o.pos0 + u;
    bool next_ran = false;
    if (stop && i + 1 == launched && H > F && H <= feed) {  // the step that feeds the last token
      verify(H), ++launched;
      m.engram_late(o.next_pos0, o.next_rows, o.next_tok);
    }
    if (stop && i + 1 < launched) {  // the queued step still runs
      wd.started = now_s();
      const model::SampleOut& q = m.wait(first + i + 1);
      wd.started = 0;
      e.written = std::max(e.written, q.pos0 + std::max(q.step_rows, q.rows + kR));
      next_ran = true;
    }
    const int kv_end = H <= F || next_ran ? H + 1 : F + 1;  // positions [pos0, kv_end) hold the history's KV
    for (; next_snap <= kv_end; next_snap += c.snap_every)  // window snapshots (the ring holds s - 128 .. s - 1)
      if (c.handle >= 0 && next_snap > c.pos0 && next_snap >= 128) m.state_snap(c.handle, next_snap);
    if (stop) {
      if (model::copy::mode())  // the rank-agreement digest of the rows every step chose (copy.h): equal on all four ranks
        std::fprintf(stderr, "copy: digest %08x after %d steps (request at %d)\n", o.copy_digest, i + 1, c.pos0);
      if (model::copy::mode() & model::copy::kAdapt)  // and the fold of every outcome the learner took: equal on all four ranks too
        std::fprintf(stderr, "copy: learn %08x after %d steps (request at %d)\n", o.copy_learn, i + 1, c.pos0);
      if (slog) std::fflush(slog);
      if (mhf) std::fflush(mhf);
      static const bool cert_log = getenv("DSGB10_CERT_CHECK") || getenv("DSGB10_CERT_LOG");
      if (uint32_t cs[8]; cert_log && m.cert_stats(cs)) {
        float ratio;
        std::memcpy(&ratio, &cs[4], 4);
        std::fprintf(stderr, "cert: steps %u, rows checked %u, mismatches %u, rows whose winner is on this rank or equal %u, "
                             "surviving CTAs %u (most in a step %u), largest |coarse - exact| / bound %.4f, forced steps %u\n",
                     cs[0], cs[1], cs[2], cs[7], cs[3], cs[5], ratio, cs[6]);
      }
      e.fed = kv_end - c.pos0;
      e.last = toks[e.fed - 1];
      ride(wctl);
      return e;
    }
    if (i + 1 == launched) {  // not queued ahead: the rows are known now
      verify(o.next_pos0 + o.next_rows - 1), ++launched;
      m.engram_late(o.next_pos0, o.next_rows, o.next_tok);
    }
    // the step after it starts at most at next_pos0 + next_rows (every draft accepted)
    if (c.ahead && launched == i + 2 && o.next_pos0 + o.next_rows <= feed)
      verify(o.next_pos0 + o.next_rows + kR), ++launched;
    g_t_launch.push_back(now_s());
    ride(wctl);
    if (step) step();
  }
}

}  // namespace

DecodeEnd decode_loop(model::Model& m, const Cmd& c, Watchdog& wd, const OnSample& on, const OnStep& step) {
  g_t_wait.clear(), g_t_late.clear(), g_t_launch.clear();
  if (c.spec) return spec_loop(m, c, wd, on, step);
  auto may_run = [&](int i) { return i <= c.max_new && c.pos0 + i < c.limit; };
  // a step at position pos; the window snapshot at pos first (the ring then holds pos - 128 .. pos - 1)
  auto launch = [&](int pos, const int32_t* tok) {
    if (c.snap_every > 0 && c.handle >= 0 && pos > c.pos0 && pos >= 128 && pos % c.snap_every == 0)
      m.state_snap(c.handle, pos);
    return m.launch(1, pos, tok, c.sp);
  };
  DecodeEnd e;
  if (c.max_new < 1 || !may_run(0)) {
    e.why = c.max_new < 1 ? kEndLength : kEndContext;
    e.written = c.pos0;
    return e;
  }
  const int32_t from_device = -1;
  const uint64_t first = launch(c.pos0, c.tokens);
  int launched = 1;
  if (c.ahead && may_run(1)) launch(c.pos0 + 1, &from_device), ++launched;
  int32_t fed = c.tokens[0];
  // The watch riding the run: the step whose output shows the bit's other value runs the reading on every rank, between two steps.
  uint32_t watch_seen = 0, split_seen = 0;
  auto ride = [&](uint32_t ctl) {
    // a ride is watched as a step is (only when the bit flipped: no cost on a step without one): a rank that stops inside it ends with the watchdog's line
    if (memwatch::flipped(watch_seen, ctl)) wd.started = now_s(), mem_watch_run(m.rank()), wd.started = 0;
    if (memwatch::split_flipped(split_seen, ctl)) wd.started = now_s(), split_ride_run(m, g_ride.req), wd.started = 0;
  };
  for (int i = 0;; ++i) {
    wd.started = now_s();
    const model::SampleOut& o = m.wait(first + i);
    const uint32_t wctl = o.ctl;
    wd.started = 0;
    g_t_wait.push_back(now_s());
    const int32_t t = o.row[0].token;
    if (on) on(i, o, 0);
    e.sampled = i + 1;
    if (!c.ahead && i + 1 <= c.max_new && may_run(i + 1)) {  // diagnostics: launch only now
      bool stop = memwatch::ends_reply(o.ctl) != 0 || i + 1 >= c.max_new;
      for (int k = 0; k < c.n_stop; ++k) stop |= t == c.stop[k];
      if (!stop) launch(c.pos0 + i + 1, &t), ++launched;
    }
    if (c.ahead && i + 1 < launched) m.engram_late(c.pos0 + i + 1, 1, &t);  // step i + 1 feeds t
    g_t_late.push_back(now_s());
    bool stop = false;
    if (memwatch::ends_reply(o.ctl)) stop = true, e.why = kEndCancel;
    for (int k = 0; !stop && k < c.n_stop; ++k)
      if (t == c.stop[k]) stop = true, e.why = kEndStop;
    if (!stop && i + 1 >= c.max_new) stop = true, e.why = kEndLength;
    if (!stop && i + 1 >= launched) stop = true, e.why = kEndContext;
    if (stop) {
      if (i + 1 < launched) {  // the queued step: its KV counts, its sample does not
        wd.started = now_s();
        m.wait(first + i + 1);
        wd.started = 0;
        e.fed = i + 2, e.last = t;
      } else {
        e.fed = i + 1, e.last = fed;
      }
      e.written = c.pos0 + e.fed;
      ride(wctl);
      return e;
    }
    fed = t;
    if (c.ahead && may_run(i + 2)) launch(c.pos0 + i + 2, &from_device), ++launched;
    g_t_launch.push_back(now_s());
    ride(wctl);
    if (step) step();
  }
}

std::vector<BatchEnd> batch_loop(model::Model& m, const Cmd& c, const std::vector<BatchEntry>& e, Watchdog& wd,
                                 const OnBatch& on, const OnStep& step) {
  g_t_wait.clear(), g_t_late.clear(), g_t_launch.clear();
  const int n = (int)e.size(), slot0 = m.slot();
  static_assert(model::ops::kMaxSess < 29, "the cancel bits are the entries' (bit 29 is kCtlSplit, bit 30 is kCtlWatch, bit 31 is kCtlPause)");
  if (n < 1 || n > model::ops::kMaxSess) die("batch: %d entries (1..%d)", n, model::ops::kMaxSess);
  std::vector<BatchEnd> end(n);
  // per entry: its steps launched, its row in the last step launched, sampling (live) or with its last
  // row still queued (ending)
  std::vector<int> launched(n, 0), last_row(n, -1);
  std::vector<char> live(n, 1), ending(n, 0);
  auto may_run = [&](int k, int j) { return j <= e[k].max_new && e[k].pos0 + j < e[k].limit; };
  std::vector<int> pin, live_h;
  for (const int32_t h : c.live)
    if (h > 0) live_h.push_back(h);
  for (int k = 0; k < n; ++k) {
    if (e[k].handle >= 0) pin.push_back(e[k].handle);
    if (!may_run(k, 0)) live[k] = 0, end[k].why = e[k].max_new < 1 ? kEndLength : kEndContext;
  }
  m.state_pin(pin, live_h);  // the entries' sessions; the other slots' keep their snapshots, and give up their pages first
  struct Step {
    uint64_t id = 0;
    int R = 0;
    int ent[model::kMaxRows] = {};  // row r's entry
  };
  std::deque<Step> q;  // launched, not yet waited for
  // the next step of every session still decoding, a row each; its window snapshot first (the ring then
  // holds pos - 128 .. pos - 1)
  auto launch = [&] {
    model::Batch b;
    Step s;
    for (int k = 0; k < n; ++k) {
      const int j = launched[k];
      if (!live[k] || ending[k] || !may_run(k, j)) {
        last_row[k] = -1;
        continue;
      }
      const int r = b.R++, pos = e[k].pos0 + j;
      if (e[k].snap_every > 0 && e[k].handle >= 0 && j > 0 && pos >= 128 && pos % e[k].snap_every == 0) {
        m.use_slot(e[k].slot);
        m.state_snap(e[k].handle, pos);
      }
      b.tokens[r] = j == 0 ? e[k].token : -1 - last_row[k];  // the step before holds its row
      b.pos[r] = pos, b.slot[r] = e[k].slot, b.sess[r] = r, b.sp[r] = e[k].sp;
      s.ent[r] = k;
      last_row[k] = r;
      ++launched[k];
    }
    if (!b.R) return;
    b.nsess = s.R = b.R;
    s.id = m.launch_batch(b);
    q.push_back(s);
  };
  // entry k ends with why after its sample tok_{sampled}: its row in the step queued, if any, still runs
  auto stop = [&](int k, uint32_t why) {
    end[k].why = why;
    if (launched[k] > end[k].sampled) ending[k] = 1;
    else live[k] = 0, end[k].fed = launched[k];
  };
  launch();
  launch();  // queued behind the first, its tokens from the device
  // The watch riding the run: the step whose output shows the bit's other value runs the reading on every rank, between two steps.
  uint32_t watch_seen = 0, split_seen = 0;
  auto ride = [&](uint32_t ctl) {
    // a ride is watched as a step is (only when the bit flipped: no cost on a step without one): a rank that stops inside it ends with the watchdog's line
    if (memwatch::flipped(watch_seen, ctl)) wd.started = now_s(), mem_watch_run(m.rank()), wd.started = 0;
    if (memwatch::split_flipped(split_seen, ctl)) wd.started = now_s(), split_ride_run(m, g_ride.req), wd.started = 0;
  };
  while (!q.empty()) {
    const Step s = q.front();
    q.pop_front();
    wd.started = now_s();
    const model::SampleOut& o = m.wait(s.id);
    const uint32_t wctl = o.ctl;
    wd.started = 0;
    g_t_wait.push_back(now_s());
    std::vector<int> row_of(n, -1);  // entry -> its row in this step
    for (int r = 0; r < s.R; ++r) row_of[s.ent[r]] = r;
    if (!q.empty()) {  // the step queued behind this one feeds its samples: their Engram lookups
      const Step& nx = q.front();
      int32_t t[model::kMaxRows];
      for (int r = 0; r < nx.R; ++r) t[r] = o.row[row_of[nx.ent[r]]].token;
      m.engram_late_batch(t);
    }
    g_t_late.push_back(now_s());
    for (int r = 0; r < s.R; ++r) {
      const int k = s.ent[r];
      BatchEnd& x = end[k];
      if (ending[k]) {  // its row behind its end: the KV counts; the sample too on a pause, within max_new
        if (x.why == kEndPause && x.sampled < e[k].max_new) {
          if (on) on(k, o, r);
          ++x.sampled;
        }
        live[k] = 0, ending[k] = 0, x.fed = launched[k];
        continue;
      }
      const int32_t t = o.row[r].token;
      if (on) on(k, o, r);
      ++x.sampled;
      if (o.ctl & (1u << k)) stop(k, kEndCancel);
      else if (std::find(e[k].stop, e[k].stop + e[k].n_stop, t) != e[k].stop + e[k].n_stop) stop(k, kEndStop);
      else if (x.sampled >= e[k].max_new) stop(k, kEndLength);
      else if (o.ctl & kCtlPause) stop(k, kEndPause);           // one that ends here ends, not pauses
      else if (launched[k] <= x.sampled) stop(k, kEndContext);  // no room for its next step
    }
    if ((c.flags & kBatchSolo) && n > 1) {  // one session left decoding: it pauses, for the drafts
      int left = 0, k1 = -1;
      for (int k = 0; k < n; ++k)
        if (live[k] && !ending[k]) ++left, k1 = k;
      if (left == 1) stop(k1, kEndPause);
    }
    launch();
    g_t_launch.push_back(now_s());
    ride(wctl);
    if (step) step();
  }
  m.use_slot(slot0);
  return end;
}

void bcast_tf(comm::Ring& ring, const Cmd& c, std::vector<BatchEntry>& e, std::vector<std::vector<int32_t>>& tokens,
              int rank) {
  if (rank != 0) e.assign(c.n, BatchEntry{}), tokens.assign(c.n, {});
  if (c.n < 1 || (int)e.size() != c.n || (int)tokens.size() != c.n) die("batch_tf: %d entries", c.n);
  bcast_payload(ring, e.data(), e.size() * sizeof(BatchEntry), rank);
  size_t total = 0;
  for (int k = 0; k < c.n; ++k) total += (size_t)std::max(0, e[k].max_new);
  std::vector<int32_t> all;
  if (rank == 0)
    for (const std::vector<int32_t>& t : tokens) all.insert(all.end(), t.begin(), t.end());
  else
    all.resize(total);
  if (all.size() != total) die("batch_tf: %zu tokens for %zu", all.size(), total);
  if (total) bcast_payload(ring, all.data(), total * 4, rank);
  if (rank != 0)
    for (int k = 0, off = 0; k < c.n; off += e[k].max_new, ++k)
      tokens[k].assign(all.begin() + off, all.begin() + off + std::max(0, e[k].max_new));
}

std::vector<float> batch_tf_loop(model::Model& m, const Cmd& c, const std::vector<BatchEntry>& e,
                                 const std::vector<std::vector<int32_t>>& tokens, Watchdog& wd, const OnBatch& on,
                                 model::Sink* sink) {
  const int n = (int)e.size(), G = c.rows;
  if (n < 1 || n > model::ops::kMaxSess || G < 1 || G > model::kMaxSessRows || (int)tokens.size() != n)
    die("batch_tf: %d entries, %d rows of each per step (1..%d)", n, G, model::kMaxSessRows);
  for (int k = 0; k < n; ++k)
    if (e[k].max_new < 0 || (int)tokens[k].size() != e[k].max_new || e[k].pos0 < 0 ||
        e[k].pos0 + e[k].max_new > e[k].limit)
      die("batch_tf: entry %d: %d tokens at %d, limit %d", k, e[k].max_new, e[k].pos0, e[k].limit);
  std::vector<int> fed(n, 0);
  std::vector<float> ms;
  for (;;) {
    model::Batch b;
    int ent[model::kMaxRows] = {};  // row r's entry
    for (int k = 0; k < n; ++k) {
      const int rows = std::min(G, e[k].max_new - fed[k]);
      if (rows <= 0) continue;
      if (b.R + rows > model::kMaxRows) die("batch_tf: a step of more than %d rows", model::kMaxRows);
      const int s = b.nsess++;
      b.sp[s] = e[k].sp;
      for (int i = 0; i < rows; ++i) {
        const int r = b.R++;
        b.tokens[r] = tokens[k][fed[k] + i];
        b.pos[r] = e[k].pos0 + fed[k] + i;
        b.slot[r] = e[k].slot;
        b.sess[r] = s;
        ent[r] = k;
      }
      fed[k] += rows;
    }
    if (!b.R) break;
    b.flags = c.flags;
    b.sink = sink;
    wd.started = now_s();
    const model::SampleOut& o = m.wait(m.launch_batch(b));
    ms.push_back(m.last_gpu_ms());
    wd.started = 0;
    if (on)
      for (int r = 0; r < b.R; ++r) on(ent[r], o, r);
  }
  return ms;
}

// kCmdSpecBatch (cmd.h): spec_loop's accounting per entry, over batch spec steps
std::vector<DecodeEnd> spec_batch_loop(model::Model& m, const Cmd& c, const std::vector<BatchEntry>& e, Watchdog& wd,
                                       const OnBatch& on, SpecBatchStats* st, const OnStep& step) {
  constexpr int kD = model::ops::kDrafts;
  static const int kR = model::copy::verify_reach(kD, (model::copy::mode() & model::copy::kRows) != 0);
  g_t_wait.clear(), g_t_late.clear(), g_t_launch.clear();
  const int n = (int)e.size(), slot0 = m.slot();
  if (n < 1 || n > model::ops::kMaxSess || n > m.spec_batch_entries())
    die("spec batch: %d entries (the model takes %d)", n, m.spec_batch_entries());
  std::vector<DecodeEnd> end(n);
  SpecBatchStats stats;
  // per entry: the rows' bounds (spec_loop's pos_end and feed), live (not finished), ending (stopped, its
  // rows of one more step still to run: queued already (pend), or in the next step launched), the rows
  // of its next step as the last result chose them (nx), the drafts' rate, its history and snapshots
  std::vector<int> pos_end(n), feed(n), next_snap(n), F(n, 0), H(n, 0);
  std::vector<char> live(n, 1), ending(n, 0), pend(n, 0);
  std::vector<SpecRate> rate(n);
  std::vector<std::vector<int32_t>> toks(n);  // tok_0, tok_1, ..: positions pos0, pos0 + 1, ..
  struct Next {
    int rows = 1, pos0 = 0;
    int32_t tok[model::ops::kSessTok] = {};
  };
  std::vector<Next> nx(n);
  // DSGB10_GATE_DUMP (with DSGB10_GATE_DUMP_FILE=FILE, rank 0's server only: the callback is its mark): a line a verified
  // entry and batch step: the step's entries and rows, the drafts verified, the drafts accepted, and what the gate saw
  // for those drafts (the shipped survival, the MLP's, the one it used, the MLP's inputs), taken from the step that chose them
  static FILE* const gdump = [&] {
    const char* v = getenv("DSGB10_GATE_DUMP_FILE");
    if (!on || !v || !*v || !getenv("DSGB10_GATE_DUMP") || !atoi(getenv("DSGB10_GATE_DUMP"))) return (FILE*)nullptr;
    FILE* f = std::fopen(v, "a");
    if (f && std::ftell(f) == 0) {
      std::fprintf(f, "tag,step,entry,sessions,step_rows,verified,accepted");
      for (const char* nm : {"ship", "mlp", "used"})
        for (int j = 1; j <= kD; ++j) std::fprintf(f, ",%s%d", nm, j);
      for (int j = 0; j < 15; ++j) std::fprintf(f, ",f%d", j);
      std::fprintf(f, "\n");
    }
    return f;
  }();
  struct GateRec {
    bool valid = false;
    int keep = 0;
    float v[3 * model::ops::kDrafts + 16] = {};
  };
  std::vector<GateRec> grec(n);
  std::vector<int> pin, live_h;
  for (const int32_t h : c.live)
    if (h > 0) live_h.push_back(h);
  for (int k = 0; k < n; ++k) {
    const BatchEntry& x = e[k];
    DecodeEnd& d = end[k];
    d.written = x.pos0;
    d.spec_tok = x.spec_tok, d.spec_ms = x.spec_ms;
    rate[k] = SpecRate{x.spec_tok, x.spec_ms};
    pos_end[k] = (int)std::min<int64_t>(x.limit, (int64_t)x.pos0 + x.max_new);
    feed[k] = std::min(pos_end[k], x.limit - 1);
    toks[k].push_back(x.token);
    nx[k].pos0 = x.pos0, nx[k].tok[0] = x.token;
    next_snap[k] = x.snap_every > 0 ? (x.pos0 / x.snap_every + 1) * x.snap_every : INT32_MAX;
    if (x.handle >= 0) pin.push_back(x.handle);
    if (x.max_new < 1 || x.pos0 >= x.limit) live[k] = 0, d.why = x.max_new < 1 ? kEndLength : kEndContext;
  }
  m.state_pin(pin, live_h);  // the entries' sessions; the other slots' keep their snapshots, and give up their pages first
  // entry k's window snapshots up to kv_end (positions below it hold its history's KV; the ring holds
  // s - 128 .. s - 1), in its slot
  auto snaps = [&](int k, int kv_end) {
    for (; next_snap[k] <= kv_end; next_snap[k] += e[k].snap_every)
      if (e[k].handle >= 0 && next_snap[k] > e[k].pos0 && next_snap[k] >= 128) {
        m.use_slot(e[k].slot);
        m.state_snap(e[k].handle, next_snap[k]);
      }
  };
  // entry k is done: positions [pos0, kv_end) hold its history's KV
  auto finish = [&](int k, int kv_end) {
    live[k] = 0, ending[k] = 0, pend[k] = 0;
    end[k].fed = kv_end - e[k].pos0;
    end[k].last = toks[k][end[k].fed - 1];
  };
  struct Step {
    uint64_t id = 0;
    bool late = false;  // a verify step whose Engram lookups wait for its rows (engram_late_rows)
    char on[model::ops::kMaxSess] = {};
  };
  std::deque<Step> q;  // launched, not yet waited for: the one running, then the one queued behind it
  auto on_now = [&](int k) { return live[k] && !(ending[k] && pend[k]); };
  // a verify step's rows, as the device lays them out (the entries on in order, each its nx), for its
  // Engram lookups: once the step before it has reported (nx), and before it is waited for
  auto late_rows = [&](Step& s) {
    int32_t t[model::kMaxRows], p[model::kMaxRows], sl[model::kMaxRows];
    int R = 0;
    for (int k = 0; k < n; ++k) {
      if (!s.on[k]) continue;
      if (nx[k].rows < 1 || nx[k].rows > model::ops::kSessTok || R + nx[k].rows > model::kMaxRows)
        die("spec batch: entry %d brings %d rows to a step of %d", k, nx[k].rows, R);
      for (int j = 0; j < nx[k].rows; ++j, ++R) t[R] = nx[k].tok[j], p[R] = nx[k].pos0 + j, sl[R] = e[k].slot;
    }
    m.engram_late_rows(R, t, p, sl);
    s.late = false;
  };
  // A launch: the entries on (live, or ending with their rows still to run), each with the drafts' cap
  // by their count; the first step given (one row per entry), the others verify steps (the rows the step
  // before chose), bound by their rows' last position at most: queued ahead, every draft of the step
  // before accepted and kD more rows after them.
  auto launch = [&](bool given, bool ahead) {
    model::SpecBatch b;
    Step s;
    int S = 0, bound = 0, lim = INT32_MAX;
    for (int k = 0; k < n; ++k)
      if (on_now(k)) ++S;
    // DSGB10_BUDGET=1: every entry may draft up to the rows left after one per entry, and the
    // device's gate keeps the step within kMaxRows by survival across the entries (lossless)
    static const bool budget = getenv("DSGB10_BUDGET") && atoi(getenv("DSGB10_BUDGET")) > 0;
    // With DSGB10_WIDE (the model's spec_batch_rows() > 16), steps of at least DSGB10_WIDE_S entries (6) may
    // reach those rows; fewer entries keep the 16-row steps
    static const int wide_s = getenv("DSGB10_WIDE_S") ? atoi(getenv("DSGB10_WIDE_S")) : 6;
    const int rows_cap = S >= wide_s ? m.spec_batch_rows() : model::kDecodeRows;
    const int cap = !S ? 0 : budget ? std::min({kD, c.spec_drafts, rows_cap - S})
                                    : std::min({kD, c.spec_drafts, (rows_cap - S) / S});
    b.budget = budget ? rows_cap : 0;
    b.n = n;
    b.cfg.max_drafts = std::max(0, std::min(kD, c.spec_drafts));
    b.cfg.gate = c.spec_gate;
    std::copy(c.spec_row_ms, c.spec_row_ms + kD + 1, b.cfg.step_ms);
    for (int k = 0; k < n; ++k) {
      b.on[k] = s.on[k] = on_now(k) ? 1 : 0;
      b.slot[k] = e[k].slot, b.sp[k] = e[k].sp, b.cap[k] = cap, b.pos_end[k] = pos_end[k];
      b.token[k] = nx[k].tok[0], b.pos0[k] = nx[k].pos0;
      b.cfg.pos_end = std::max(b.cfg.pos_end, pos_end[k]);
      if (!b.on[k]) continue;
      b.thresh[k] = rate[k].thresh(c);
      end[k].gate(b.thresh[k]);
      const int hi = ahead ? nx[k].pos0 + nx[k].rows + kR : nx[k].pos0 + nx[k].rows - 1;  // entry k's rows' last position at most
      b.bound[k] = std::min(hi, e[k].limit - 1);  // the memory its slot needs mapped (Model::launch_verify_batch)
      bound = std::max(bound, hi);
      lim = std::min(lim, e[k].limit);
      if (ending[k]) pend[k] = 1;
    }
    b.pos_bound = std::min(bound, lim - 1);
    b.per_entry_bound = true;
    s.id = given ? m.launch_spec_batch(b) : m.launch_verify_batch(b);
    s.late = !given;  // a given step's rows are the host's: launch_spec_batch looked them up
    q.push_back(s);
  };
  // the next step may be queued before the one running ends when every entry decoding fits it: its
  // rows start at most where the running step's end (every draft accepted), at or below feed
  auto fits = [&] {
    int any = 0;
    for (int k = 0; k < n; ++k) {
      if (!live[k] || ending[k]) continue;
      if (nx[k].pos0 + nx[k].rows > feed[k]) return false;
      ++any;
    }
    return any > 0;
  };
  bool any_live = false;
  for (int k = 0; k < n; ++k) any_live |= live[k] != 0;
  if (any_live) {
    launch(true, false);
    if (c.ahead && fits()) launch(false, true);
  }
  uint64_t prev_id = 0;
  int prev_R = 0;
  uint32_t copy_digest = 0, copy_learn = 0;
  // The watch riding the run: the step whose output shows the bit's other value runs the reading on every rank, between two steps.
  uint32_t watch_seen = 0, split_seen = 0;
  auto ride = [&](uint32_t ctl) {
    // a ride is watched as a step is (only when the bit flipped: no cost on a step without one): a rank that stops inside it ends with the watchdog's line
    if (memwatch::flipped(watch_seen, ctl)) wd.started = now_s(), mem_watch_run(m.rank()), wd.started = 0;
    if (memwatch::split_flipped(split_seen, ctl)) wd.started = now_s(), split_ride_run(m, g_ride.req), wd.started = 0;
  };
  for (int i = 0; !q.empty(); ++i) {
    const Step s = q.front();
    q.pop_front();
    wd.started = now_s();
    const model::SampleOut& o = m.wait(s.id);
    const uint32_t wctl = o.ctl;
    wd.started = 0;
    g_t_wait.push_back(now_s());
    for (int k = 0; k < n; ++k) {  // each entry's next rows, the device's choice
      if (!s.on[k]) continue;
      nx[k].rows = o.next_srows[k], nx[k].pos0 = o.next_spos0[k];
      std::copy(o.next_stok[k], o.next_stok[k] + model::ops::kSessTok, nx[k].tok);
    }
    if (!q.empty() && q.front().late) late_rows(q.front());  // the step queued ahead: its rows are known now
    g_t_late.push_back(now_s());
    if (i > 4)  // the step before (done; past the graphs' captures): its GPU time by its rows
      stats.gpu_ms[std::min(prev_R, model::kMaxRows)] += m.gpu_ms(prev_id), ++stats.gpu_n[std::min(prev_R, model::kMaxRows)];
    prev_id = s.id, prev_R = o.step_rows;
    ++stats.steps, stats.rows += o.step_rows;
    if (model::copy::mode()) copy_digest = o.copy_digest, copy_learn = o.copy_learn;
    if (on && copy_dump_file()) {
      for (int k = 0; k < n; ++k) {
        if (!s.on[k]) continue;
        int32_t em[model::kMaxRows];
        const int ne = std::max(0, std::min<int>(o.sess_rows[k], model::kMaxRows - o.sess_row0[k]));
        for (int j = 0; j < ne; ++j) em[j] = o.row[o.sess_row0[k] + j].token;
        copy_dump_line(copy_dump_file(), i, k, o.sess_pos0[k], ne, em, o.next_spos0[k], o.next_stok[k], o.copy_info[k], o.copy_tok[k]);
      }
      std::fflush(copy_dump_file());
    }
    if (gdump) {
      static const char* const tag = getenv("DSGB10_GATE_TAG") ? getenv("DSGB10_GATE_TAG") : "";
      int son = 0;
      for (int k = 0; k < n; ++k) son += s.on[k] ? 1 : 0;
      for (int k = 0; k < n; ++k) {
        if (!s.on[k]) continue;
        GateRec& g = grec[k];
        const int sr = o.sess_step_rows[k], er = o.sess_rows[k];
        if (g.valid && sr - 1 == g.keep && sr >= 1 && er >= 1 && er <= sr) {
          std::fprintf(gdump, "%s,%d,%d,%d,%d,%d,%d", tag, i, k, son, o.step_rows, sr - 1, er - 1);
          for (int j = 0; j < 3 * kD; ++j) std::fprintf(gdump, ",%.7g", g.v[j]);
          for (int j = 0; j < 15; ++j) std::fprintf(gdump, ",%.7g", g.v[3 * kD + j]);
          std::fprintf(gdump, "\n");
        }
        g.valid = true, g.keep = o.gate_keep[k];
        std::copy(o.gate_ship[k], o.gate_ship[k] + kD, g.v);
        std::copy(o.gate_mlp[k], o.gate_mlp[k] + kD, g.v + kD);
        std::copy(o.gate_used[k], o.gate_used[k] + kD, g.v + 2 * kD);
        std::copy(o.gate_feat[k], o.gate_feat[k] + 15, g.v + 3 * kD);
      }
      std::fflush(gdump);
    }
    for (int k = 0; k < n; ++k) {
      if (!s.on[k]) continue;
      DecodeEnd& d = end[k];
      const int p0 = o.sess_pos0[k], sr = o.sess_step_rows[k], er = o.sess_rows[k], r0 = o.sess_row0[k];
      if (sr < 1 || er < 1 || er > sr || r0 < 0 || r0 + sr > o.step_rows)
        die("spec batch: entry %d: rows %d (%d emitted) from row %d of %d", k, sr, er, r0, o.step_rows);
      d.written = std::max(d.written, p0 + std::max(sr, er + kR));
      if (ending[k]) {  // its rows behind its end: their KV counts, their tokens do not
        snaps(k, H[k] + 1);
        finish(k, H[k] + 1);
        continue;
      }
      if (model::copy::mode() & model::copy::kAdapt) d.copy_obs(o.copy_obs[k], o.copy_chance);
      if (i > 0) {
        ++d.vsteps, d.verified += sr - 1, d.accepted += er - 1;
        ++d.rows_hist[std::min(sr - 1, kD)];
      }
      rate[k].add(c, er, sr);
      d.spec_tok = rate[k].tok, d.spec_ms = rate[k].ms;
      // the tokens it emits, until a stop
      bool stop = false;
      int u = 0;
      for (int j = 0; j < er && !stop; ++j) {
        const int r = r0 + j;
        const int32_t t = o.row[r].token;
        if (on) on(k, o, r);
        ++d.sampled, ++u, ++stats.emitted;
        toks[k].push_back(t);
        if (o.ctl & (1u << k)) stop = true, d.why = kEndCancel;
        for (int q2 = 0; !stop && q2 < e[k].n_stop; ++q2)
          if (t == e[k].stop[q2]) stop = true, d.why = kEndStop;
        if (!stop && d.sampled >= e[k].max_new) stop = true, d.why = kEndLength;
      }
      if (!stop && (o.ctl & kCtlPause)) stop = true, d.why = kEndPause;
      if (!stop && nx[k].pos0 >= pos_end[k]) stop = true, d.why = kEndContext;
      // positions p0 .. F hold their final KV; the history now ends at H
      F[k] = p0 + er - 1, H[k] = p0 + u;
      if (!stop) {
        snaps(k, H[k]);
        continue;
      }
      ending[k] = 1;
      if (!q.empty() && q.front().on[k]) {  // its rows in the step queued still run: the KV of its last token too
        pend[k] = 1;
      } else if (H[k] > F[k] && H[k] <= feed[k] && d.why != kEndPause) {  // the next step launched feeds its last token
        pend[k] = 0;
      } else {
        const int kv_end = H[k] <= F[k] ? H[k] + 1 : F[k] + 1;
        snaps(k, kv_end);
        finish(k, kv_end);
      }
    }
    if ((c.flags & kBatchSolo) && n > 1) {  // one entry left decoding: it pauses, to go on alone
      int left = 0, k1 = -1;
      for (int k = 0; k < n; ++k)
        if (live[k] && !ending[k]) ++left, k1 = k;
      if (left == 1) {
        end[k1].why = kEndPause;
        ending[k1] = 1;
        if (!q.empty() && q.front().on[k1]) {
          pend[k1] = 1;
        } else {
          const int kv_end = H[k1] <= F[k1] ? H[k1] + 1 : F[k1] + 1;
          snaps(k1, kv_end);
          finish(k1, kv_end);
        }
      }
    }
    bool any = false;
    for (int k = 0; k < n; ++k) any |= on_now(k);
    if (q.empty() && any) {  // not queued ahead: the rows are known now, its lookups go at once
      launch(false, false);
      late_rows(q.back());
    }
    if (c.ahead && q.size() == 1 && fits()) launch(false, true);
    g_t_launch.push_back(now_s());
    ride(wctl);
    if (step) step();
  }
  m.use_slot(slot0);
  if (model::copy::mode())  // the rank-agreement digest of the rows every step chose (copy.h): equal on all four ranks
    std::fprintf(stderr, "copy: digest %08x after %d batch steps\n", copy_digest, stats.steps);
  if (model::copy::mode() & model::copy::kAdapt)
    std::fprintf(stderr, "copy: learn %08x after %d batch steps\n", copy_learn, stats.steps);
  if (st) *st = stats;
  return end;
}

void follow(model::Model& m, comm::Ring& ring, Watchdog& wd, int rank) {
  bool prof = false;
  int32_t sampled[model::kMaxRows];
  for (;;) {
    const Cmd c = bcast(ring, Cmd{});
    if (c.op == kCmdStop) break;
    NVTX_RANGE("cmd %d pos0 %d rows %d n %d", c.op, c.pos0, c.rows, c.n);
    if ((c.flags & model::kFlagProfile) && !prof) {
      CUDA_CHECK(cudaProfilerStart());
      prof = true;
    }
    wd.started = now_s();     // a state operation is watched as a step is: a rank that waits in one for a partner ends with the watchdog's line
    apply_ops(m, c);         // first, whatever the command
    wd.started = 0;
    (void)m.state_evicted();  // rank 0's server keeps track of the sessions
    if (c.op == kCmdStep) {
      wd.started = now_s();
      m.step(to_ctl(c), sampled, nullptr);
      wd.started = 0;
    } else if (c.op == kCmdDecode) {
      decode_loop(m, c, wd, nullptr);
    } else if (c.op == kCmdBatch || c.op == kCmdSpecBatch) {
      std::vector<BatchEntry> e(c.n);
      bcast_payload(ring, e.data(), e.size() * sizeof(BatchEntry), rank);
      if (c.op == kCmdSpecBatch) spec_batch_loop(m, c, e, wd, nullptr);
      else batch_loop(m, c, e, wd, nullptr);
    } else if (c.op == kCmdBatchTf) {
      std::vector<BatchEntry> e;
      std::vector<std::vector<int32_t>> t;
      bcast_tf(ring, c, e, t, rank);
      batch_tf_loop(m, c, e, t, wd, nullptr);
    } else if (c.op == kCmdTokens) {
      std::vector<int32_t> t(c.n);
      bcast_payload(ring, t.data(), t.size() * 4, rank);
      m.set_tokens(c.pos0, c.n, t.data());
    } else if (c.op == kCmdImages) {
      if (c.n <= 0) {
        m.set_images(0, nullptr, nullptr, 0);
      } else {
        std::vector<int32_t> lh(2 * c.n);
        bcast_payload(ring, lh.data(), lh.size() * 4, rank);
        size_t rows = 0;
        for (int i = 0; i < c.n; ++i) rows += lh[c.n + i] - lh[i];
        if (c.rows) {  // the first go: the rows follow, a piece at a time into the numbers kStImgTake took
          std::vector<uint16_t> piece(model::Model::kImgStageRows * 5120);
          for (size_t r0 = 0; r0 < rows; r0 += model::Model::kImgStageRows) {
            const size_t n = std::min(model::Model::kImgStageRows, rows - r0);
            bcast_payload(ring, piece.data(), n * 5120 * 2, rank);
            m.image_fill(c.pos0, r0, n, piece.data());
          }
        }
        if (c.vit_reg) m.vision_register(c.pos0, c.n);
        m.set_images(c.n, lh.data(), lh.data() + c.n, c.pos0);
      }
    } else if (c.op == kCmdVit) {
      wd.started = now_s();
      (void)run_vit(m, c, nullptr);
      wd.started = 0;
    } else if (c.op == kCmdWindow) {
      wd.started = now_s();  // capturing the graphs takes seconds of CPU
      run_window(m, c);
      wd.started = 0;
      (void)m.state_evicted();
    } else if (c.op == kCmdLend) {
      wd.started = now_s();
      run_lend(m, c);
      wd.started = 0;
      (void)m.state_evicted();
    } else if (c.op == kCmdPrefill) {
      wd.started = now_s();
      const uint64_t n = run_prefill(m, c);
      if (n != ~0ull) m.wait(n);
      m.drain();
      wd.started = 0;
    } else if (c.op == kCmdRows) {
      wd.started = now_s();
      m.wait(run_rows(m, c));
      wd.started = 0;
    } else if (c.op != kCmdState) {
      die("rank %d: unknown command %d", rank, c.op);
    }
  }
  if (prof) CUDA_CHECK(cudaProfilerStop());
}

}  // namespace srv
