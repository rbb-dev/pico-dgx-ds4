// A YaRN factor's window memory, taken from the chunk pool while needed (docs/engine.md, "Context past 1M: a YaRN factor
// per request"). window_open takes whole groups of numbers (evicting idle conversations holding them), makes the
// factor's buffers and graphs and returns the groups the build did not need; window_close frees them and the pool makes
// the numbers again. Every rank calls alike (srv::run_window); decisions use the store, the allocator or host_ring's minimum.
#include <algorithm>
#include <chrono>
#include <cstdio>

#include "common/alloc_seal.h"
#include "model/impl.h"
#include "model/lend_ops.h"
#include "model/prefill.h"
#include "model/spec.h"
#include "model/state.h"

namespace model {

using namespace detail;

namespace {

constexpr int kGroupNums = 64;  // numbers in a group: the smallest set whose slices of all four arrays are whole 2 MiB chunks
constexpr size_t kMargin = 32ull << 20;

double wall() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t min_of(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine, comm::kCallWindowMin);
  return *std::min_element(all, all + 4);
}
uint64_t max_of(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine, comm::kCallWindowMax);
  return *std::max_element(all, all + 4);
}
size_t free_device() {
  size_t f = 0, t = 0;
  CUDA_CHECK(cudaMemGetInfo(&f, &t));
  return f;
}

}  // namespace

int Model::Impl::scan_stride_of(int f) const {
  const int ctx = lazy_win ? std::min(opt.max_ctx, 65536 * f) : opt.max_ctx;
  const int k = std::max(ctx, kPoolBlocks * 8);
  return (k + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
}

// The bytes factor f's score buffers take: decode's rows, and with `prefill` the prefill's batch.
size_t Model::Impl::scan_bytes(int f, bool prefill) const {
  const size_t stride = (size_t)scan_stride_of(f);
  if (!prefill) return ((size_t)kDecodeRows * stride + (size_t)kDecodeRows * stride / 8) * sizeof(uint16_t);
  const size_t pk = std::max((size_t)kIdxBatch * stride, pf ? (size_t)pf->key_q * kPoolBlocks * 8 : 0);
  return (pk + (size_t)kIdxBatch * stride / 8) * sizeof(uint16_t);
}

// The pool's top number with `g` more groups of factor f's window out of it; false when the pool could not hold one
// conversation of the window even whole.
bool Model::Impl::window_top(int f, int g, uint32_t* top) const {
  const size_t n0 = lent.groups ? lent.n0 : calloc->numbers();
  const size_t base = n0 / kGroupNums * kGroupNums;
  const size_t cut = (size_t)(lent.groups + g) * kGroupNums;
  const int whole = numbers_for(0, std::min(65536 * f, opt.max_ctx) - 1);
  if (cut + 2 > base || base - cut < (size_t)whole + 2) return false;
  *top = (uint32_t)lend::cut_top(n0, (size_t)(lent.groups + g));
  return true;
}

// The chunk handles the pool holds for `nums` numbers, and the bytes of each array: the arrays' mapped ends rounded up to chunks.
size_t Model::Impl::pool_chunks_for(size_t nums, size_t* bytes) const {
  size_t chunks = 0;
  for (int a = 0; a < 4; ++a) {
    const size_t b = (nums * attn::kChunkEntries * kKvEntryBytes[a] + vmm::kChunk - 1) / vmm::kChunk * vmm::kChunk;
    if (bytes) bytes[a] = b;
    chunks += b / vmm::kChunk;
  }
  return chunks;
}

// The numbers at or above `top` leave the pool: the allocator, the arrays' mapped ends, their chunks (the physical memory goes
// back to the system). The numbers must all be free (pool_vacate).
void Model::Impl::pool_cut(size_t top) {
  CUDA_CHECK(cudaDeviceSynchronize());
  const size_t was = calloc->numbers();
  core->alloc_shrink(top);
  size_t bytes[4];
  pool_chunks_for(top, bytes);
  slot_pool->shrink(0, bytes);
  // exactly the chunks the numbers' ends held: the handle count falls by what the arithmetic says (the hand-over's proof)
  slot_pool->trim(std::min(slot_pool->free_chunks(), pool_chunks_for(was) - pool_chunks_for(top)));
}

// `more` numbers in the pool on this rank, or none: the chunks are made and mapped, and when the driver cannot, what was made
// is given back at once (the pool is as it was).
bool Model::Impl::pool_grow_try(size_t more) {
  const size_t nums = calloc->numbers() + more;
  if (nums > pool_max) return false;
  size_t bytes[4];
  const size_t want = pool_chunks_for(nums, bytes), have = slot_pool->total();
  size_t added = 0;
  if (want > have) {
    added = slot_pool->add_chunks(want - have);
    if (added != want - have) {
      slot_pool->trim(added);
      return false;
    }
  }
  if (!slot_pool->ensure(0, bytes)) {
    size_t was_bytes[4];
    pool_chunks_for(calloc->numbers(), was_bytes);
    slot_pool->shrink(0, was_bytes);  // ensure keeps what it mapped before the driver failed: unmapped, so trim can take the chunks
    slot_pool->trim(added);
    return false;
  }
  core->alloc_grow(nums);
  return true;
}

// `more` numbers on every rank, or none of them anywhere: a rank that could grow goes back when another could not.
bool Model::Impl::pool_grow_all(size_t more) {
  const size_t was = calloc->numbers();
#ifdef DSGB10_TEST_FAULTS
  const bool fault = lend_fault("back");
#else
  const bool fault = false;
#endif
  const bool local = !fault && pool_grow_try(more);
  const bool all = min_of(*host_ring, local) != 0;
  if (local && !all) pool_cut(was);
  return all;
}

// numbers one conversation of the longest window open needs (the base window's at least)
size_t Model::Impl::lend_whole() const {
  size_t whole = (size_t)numbers_for(0, std::min(65536 * 16, opt.max_ctx) - 1);
  for (const int f : {32, 64})
    if (lazy_win && scan_[level_of(f)].open) whole = std::max(whole, (size_t)numbers_for(0, std::min(65536 * f, opt.max_ctx) - 1));
  return whole;
}

// Numbers one conversation of the longest window the options allow needs, open or not: the pool keeps that room whatever its
// tenants take, so a request past 1M never finds its window refused (kWindowNoRoom) because a tenant took the room first. The
// groups of the windows still to open (`later`) are the driver's, agreed across the ranks (TenantEnv::later_groups).
size_t Model::Impl::lend_room_whole() const {
  size_t whole = (size_t)numbers_for(0, std::min(65536 * 16, opt.max_ctx) - 1);
  for (const int f : {32, 64}) {
    if (opt.max_ctx <= 65536 * (f / 2)) continue;  // no request can need factor f
    whole = std::max(whole, (size_t)numbers_for(0, std::min(65536 * f, opt.max_ctx) - 1));
  }
  return whole;
}

// `g` more groups out of the pool for factor f's window: the numbers above the new end are made free (pool_vacate), the
// lines' reservations still covered by what is left, then the numbers leave the allocator and their chunks the driver.
// kWindowNoRoom: the pool could not hold one conversation of the window even whole; kWindowBusy: running requests hold too
// much of it now (win_busy_g records g then). The result is the same on every rank. On any other than kWindowOk the
// conversations evicted on the way stay evicted; no number left the pool.
int Model::Impl::window_make_room(int f, int g) {
  uint32_t top = 0;
  if (!window_top(f, g, &top)) return kWindowNoRoom;
  LendOps ops{*this};
  lend::Lender<LendOps> L(ops, lent);
  const int st = L.take(g, (size_t)numbers_for(0, std::min(65536 * f, opt.max_ctx) - 1));
  if (st == lend::kNoRoom) return kWindowNoRoom;
  if (st == lend::kBusy) return win_busy_g[level_of(f)] = g, kWindowBusy;
  if (st != lend::kOk) return kWindowFailed;
  win_busy_g[level_of(f)] = 0;
  return kWindowOk;
}

// `groups` groups back into the pool: the numbers are made and mapped again (the driver must have the memory on every rank).
// false: some rank could not; nothing changed on any, the groups stay out and are the caller's to owe.
bool Model::Impl::window_regrow(int groups) {
  if (groups < 1) return true;
  LendOps ops{*this};
  lend::Lender<LendOps> L(ops, lent);
  return L.give(groups);
}

// What is owed, asked for again: every rank has the memory for it (device memory free), then the numbers are made.
int Model::Impl::lend_settle() {
  if (lent.owed < 1) return lend::kOk;
  const size_t back = (size_t)lent.owed * lend::kGroupNums * chunk_image();
  const bool ok = free_device() >= back + 4 * kMargin;
  LendOps ops{*this};
  lend::Lender<LendOps> L(ops, lent);
  return L.settle(min_of(*host_ring, ok) != 0);
}

bool Model::Impl::win_alloc(int f) {
  const alloc_seal::Scoped seal(alloc_seal::kRegion);
  Scan& sc = scan_[level_of(f)];
  sc.stride = scan_stride_of(f);
  const size_t stride = (size_t)sc.stride;
  const auto make = [&](uint16_t** p, size_t n) {
    void* q = nullptr;
    if (cudaMalloc(&q, n * sizeof(uint16_t)) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    *p = static_cast<uint16_t*>(q);
    if (cudaMemset(q, 0, n * sizeof(uint16_t)) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  };
  bool ok = make(&sc.keys, (size_t)kDecodeRows * stride) && make(&sc.bkeys, (size_t)kDecodeRows * stride / 8);
  if (ok && pf)
    ok = make(&sc.pkeys, std::max((size_t)kIdxBatch * stride, (size_t)pf->key_q * kPoolBlocks * 8)) &&
         make(&sc.pbkeys, (size_t)kIdxBatch * stride / 8);
  cudaStreamSynchronize(0);  // the zeroing runs on the legacy stream: finished before a kernel can read the buffers
  if (!ok) win_free(f);
  return ok;
}

void Model::Impl::win_free(int f) {
  Scan& sc = scan_[level_of(f)];
  for (uint16_t** p : {&sc.keys, &sc.bkeys, &sc.pkeys, &sc.pbkeys}) {
    if (*p) cudaFree(*p);
    *p = nullptr;
  }
}

// Factor f's graphs: the step and batch step of the spec module, and the plain steps captured on first use.
void Model::Impl::win_graphs_drop(int f) {
  if (spec) {
    for (auto it = spec->graphs.begin(); it != spec->graphs.end();)
      if (it->first.second == f) cudaGraphExecDestroy(it->second), it = spec->graphs.erase(it);
      else ++it;
    for (auto it = spec->bgraphs.begin(); it != spec->bgraphs.end();)
      if (it->first.second == f) cudaGraphExecDestroy(it->second), it = spec->bgraphs.erase(it);
      else ++it;
  }
  for (auto it = graphs.begin(); it != graphs.end();)
    if (std::get<2>(it->first) == f) cudaGraphExecDestroy(it->second), it = graphs.erase(it);
    else ++it;
}

// The bytes factor f's window takes by this rank's own estimate: its buffers, its graphs and a margin.
size_t Model::Impl::win_need(int f) const {
  const size_t bufs = scan_bytes(f, false) + (pf ? scan_bytes(f, true) : 0), gr = win_gcost + win_gcost / 4;
  return bufs + gr + kMargin;
}

// Opens factor f's window (a no-op for 16, an open window, and an engine whose windows are all made at the start).
int Model::Impl::window_open(int f) {
  if (!lazy_win || f == 16) return kWindowOk;
  struct Running {  // a window command runs: no tenant change begins meanwhile (lend::StoreQueue)
    lend::StoreQueue& q;
    explicit Running(lend::StoreQueue& x) : q(x) { q.window = true; }
    ~Running() { q.window = false; }
  } running(drv.queue);
  Scan& sc = scan_[level_of(f)];
  if (sc.open) return kWindowOk;
  if (!host_ring || !calloc) die("model: window_open(%d) before the chunk pool and the ring exist", f);
  const double t0 = wall();
  const size_t gb = (size_t)kGroupNums * chunk_image();
  const size_t bufs = scan_bytes(f, false) + (pf ? scan_bytes(f, true) : 0), gr = win_gcost + win_gcost / 4;
  const size_t need = max_of(*host_ring, win_need(f));  // the largest estimate of any rank: one count for all
  const int g = (int)((need + gb - 1) / gb);
  const size_t n0 = calloc->numbers();  // before the groups (and the top numbers short of a group) leave
  const int room = window_make_room(f, g);
  if (room != kWindowOk) return room;
  const size_t n_before = calloc->numbers();
  // buffers (a failed allocation or too little memory left on any rank undoes it on every rank)
  CUDA_CHECK(cudaDeviceSynchronize());
  const size_t free_a = free_device();
  const bool made = free_a >= bufs + gr + kMargin && win_alloc(f);  // win_alloc frees its own buffers when it fails
#ifdef DSGB10_TEST_FAULTS
  const bool fault = made && lend_fault("win");
#else
  const bool fault = false;
#endif
  const bool ok = made && !fault && free_device() >= gr + kMargin;
  if (!min_of(*host_ring, ok)) {
    if (made) win_free(f);  // the buffers are this rank's to free whichever check failed, or they stay held
    const bool back_in = window_regrow(g);
    if (!back_in) lent.owed += g;  // a rank could not make the numbers again: owed, asked for again by the next settle
    if (rank == 0)
      std::printf("window: factor %d: too little memory on a rank for its %.2f GiB; the %d groups are %s\n", f, (bufs + gr) / 1073741824.0, g,
                  back_in ? "back in the pool" : "owed to the pool");
    return kWindowFailed;
  }
  // the graphs: the window's own (win_graphs_drop frees them at its close), so the window tenant's body like its buffers
  if (spec) {
    const alloc_seal::Scoped seal(alloc_seal::kRegion);
    const int sf0 = sf;
    sf = f;
    spec->quiet = true;
    spec_graph(sc.stride);
    if (spec->bents >= 2) spec_batch_graph(sc.stride);
    spec->quiet = false;
    sf = sf0;
  }
  sc.open = true;
  const size_t free_c = free_device();
  const size_t used = free_a > free_c ? free_a - free_c : 0;
  // the groups the build did not need go back (the smallest number any rank can spare; the next 32 MiB is the margin)
  const size_t have = (size_t)g * gb;
  const int spare = used + kMargin < have ? (int)std::min<size_t>((have - used - kMargin) / gb, (size_t)g - 1) : 0;
  const int back = (int)min_of(*host_ring, (uint64_t)spare);
  if (!window_regrow(back)) lent.owed += back;  // owed: the window no longer holds them
  sc.groups = g - back;
  sc.bytes = bufs;
  book_windows();
  win_free_end = free_device();
  if (rank == 0)
    std::printf("window: factor %d open in %.2f s: %zu numbers (%.2f GiB) out of the pool: %d groups of %d and %zu above the last whole "
                "group; the pool has %zu numbers (%.2f GiB) now, was %zu; buffers %.2f GiB, graphs %.2f GiB (this rank's measure)\n",
                f, wall() - t0, n0 - (size_t)calloc->numbers(), (n0 - (size_t)calloc->numbers()) * (double)chunk_image() / 1073741824.0,
                sc.groups, kGroupNums, n0 - (size_t)calloc->numbers() - (size_t)sc.groups * kGroupNums, (size_t)calloc->numbers(),
                calloc->total() * (double)chunk_image() / 1073741824.0, n0, bufs / 1073741824.0,
                (used > bufs ? used - bufs : 0) / 1073741824.0);
  split_zone_update();
  return kWindowOk;
}

// Closes factor f's window (f 0: only settles what a closed window still owes the pool): its graphs and buffers are
// freed and, when every rank has the memory, its groups are numbers of the pool again.
int Model::Impl::window_close(int f) {
  if (!host_ring || (!lazy_win && f != 0)) return kWindowOk;  // f 0 settles what any borrower owes, windows or not
  struct Running {
    lend::StoreQueue& q;
    explicit Running(lend::StoreQueue& x) : q(x) { q.window = true; }
    ~Running() { q.window = false; }
  } running(drv.queue);
  const double t0 = wall();
  if (f > 16 && scan_[level_of(f)].open) {
    Scan& sc = scan_[level_of(f)];
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int s = 0; s < nslots; ++s)
      if (slot_fac[s] == f) slot_fac[s] = 16;  // an idle line's factor is set again when a request takes it
    sf = slot_fac[cur_slot];
    win_graphs_drop(f);
    win_free(f);
    sc.open = false;
    lent.owed += sc.groups;
    sc.groups = 0;
    book_windows();
  }
  split_zone_update();
  if (lent.owed < 1) return kWindowOk;
  const int owed = lent.owed;
  const int st = lend_settle();
  if (st == lend::kBusy) return kWindowBusy;  // some rank's memory is still taken: owed until the next settle
  if (st != lend::kOk) return kWindowFailed;
  if (rank == 0)
    std::printf("window: factor %d closed in %.2f s: %d groups back, the pool has %u numbers (%.2f GiB)\n", f, wall() - t0, owed,
                calloc->numbers(), calloc->total() * (double)chunk_image() / 1073741824.0);
  return kWindowOk;
}

bool Model::window_lazy() const { return d_->lazy_win; }
bool Model::window_is_open(int factor) const {
  if (factor != 16 && factor != 32 && factor != 64) return false;
  return !d_->lazy_win || d_->scan_[Impl::level_of(factor)].open;
}
int Model::window_open(int factor) { return d_->window_open(factor); }
int Model::window_close(int factor) { return d_->window_close(factor); }
int Model::window_owed() const { return d_->lent.owed; }
int Model::window_groups() const { return d_->lent.groups; }

int Model::window_groups_estimate(int f) const {
  if (f != 32 && f != 64) return 0;
  const size_t gb = (size_t)kGroupNums * d_->chunk_image();
  return (int)((d_->win_need(f) + gb - 1) / gb);
}
int Model::window_busy_groups(int f) const {
  if (f != 32 && f != 64) return 0;
  return d_->win_busy_g[Impl::level_of(f)];
}

// Evicts nothing: the numbers a window of `g` groups for factor f cuts off the pool and who holds them (window_make_room's
// arithmetic and the core's holder scan, read only).
Model::WindowPlan Model::window_plan(int f, int g) const {
  WindowPlan p;
  const Impl& m = *d_;
  uint32_t top = 0;
  if (!m.calloc || !m.core->has_store() || !m.window_top(f, g, &top)) return p;
  p.top = top;
  if (top >= m.calloc->numbers()) return p;
  p.removed = (int64_t)m.calloc->numbers() - (int64_t)top;
  const detail::StoreCore::Holders hs = m.core->holders_from(top);
  for (const auto& h : hs.holders) p.holders.push_back({h.handle, h.line, h.unit_own});
  p.image_numbers = hs.image_numbers, p.image_flights = hs.image_flights;
  return p;
}

}  // namespace model
