// A YaRN factor's window memory, taken from the chunk pool while needed (docs/engine.md, "Context past 1M: a YaRN factor
// per request"). window_open takes whole groups of numbers (evicting idle conversations holding them), makes the
// factor's buffers and graphs and returns the groups the build did not need; window_close frees them and the pool makes
// the numbers again. Every rank calls alike (srv::run_window); decisions use the store, the allocator or host_ring's minimum.
#include <algorithm>
#include <chrono>
#include <cstdio>

#include "model/impl.h"
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
bool has(const std::vector<int>& v, int k) { return std::find(v.begin(), v.end(), k) != v.end(); }
uint64_t min_of(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine);
  return *std::min_element(all, all + 4);
}
uint64_t max_of(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine);
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

// Numbers the pool holds with `g` groups out of it: its size before the first window, less whole groups below the
// last group boundary of that size, so the arrays' ends are whole chunks.
size_t Model::Impl::win_numbers_for(int g) const { return g == 0 ? win_n0 : win_n0 / kGroupNums * kGroupNums - (size_t)g * kGroupNums; }

namespace {
// whether session x holds a chunk number at or above `from` (in a list, a window bookmark or the end bookmark)
bool holds_from(const Store::Sess& x, uint32_t from) {
  for (const Nums& l : x.lists)
    for (const uint32_t c : l)
      if (c >= from) return true;
  for (const auto& sn : x.snaps)
    for (const uint32_t c : sn.second)
      if (c >= from) return true;
  for (const uint32_t c : x.end)
    if (c >= from) return true;
  return false;
}
}  // namespace

// Makes the numbers at or above `from` free: the conversations that hold them go, a piece at a time (to the disk tier, or
// whole without one; the same on every rank), the command's evict-last sessions after the others, then the least recently
// used. 0: none is taken; 1: a line, a pinned or a running conversation holds one (the pieces of the holders evicted before
// it was found stay evicted: they are cache), or no piece is left to give.
int Model::Impl::pool_vacate(uint32_t from) {
  Store& s = store_of();
  for (;;) {
    if (!calloc->first_taken_from(from)) return 0;
    rules::VictimPick pick;
    bool busy = false;
    for (const auto& [k, x] : s.sess) {
      if (!holds_from(x, from)) continue;
      if (rules::holder_busy({k, x.line, x.used}, s.pinned, s.live, s.last)) busy = true;
      else pick.offer(k, x.used, has(s.last, k));
    }
    const int victim = pick.victim;
    if (busy || victim < 0) return 1;
    if (!s.disk) {
      state_free(victim);
      s.evicted.push_back(victim);
    } else if (!evict_piece(victim, 0)) {
      return 1;
    }
  }
}

// The pool's top number with `g` more groups of factor f's window out of it; false when the pool could not hold one
// conversation of the window even whole.
bool Model::Impl::window_top(int f, int g, uint32_t* top) const {
  const size_t n0 = win_groups ? win_n0 : calloc->numbers();
  const size_t base = n0 / kGroupNums * kGroupNums;
  const size_t cut = (size_t)(win_groups + g) * kGroupNums;
  const int whole = numbers_for(0, std::min(65536 * f, opt.max_ctx) - 1);
  if (cut + 2 > base || base - cut < (size_t)whole + 2) return false;
  *top = (uint32_t)(base - cut);
  return true;
}

// `g` more groups out of the pool for factor f's window: the numbers above the new end are made free (pool_vacate), the
// lines' reservations still covered by what is left, then the numbers leave the allocator and their chunks the driver.
// kWindowNoRoom: the pool could not hold one conversation of the window even whole; kWindowBusy: running requests hold too
// much of it now (win_busy_g records g then). The result is the same on every rank. On any other than kWindowOk the
// conversations evicted on the way stay evicted; no number left the pool.
int Model::Impl::window_make_room(int f, int g) {
  const size_t n0 = win_groups ? win_n0 : calloc->numbers();
  uint32_t top = 0;
  if (!window_top(f, g, &top)) return kWindowNoRoom;
  if (top < calloc->numbers()) {
    if (pool_vacate(top)) return win_busy_g[level_of(f)] = g, kWindowBusy;
    const size_t removed = calloc->numbers() - top;
    const int64_t unheld = rules::unheld_sum(nslots, [&](int s) { return resv[s]; }, [&](int s) { return held(s); });
    while (!rules::window_room((int64_t)calloc->free(), (int64_t)removed, unheld))
      if (!store_evict_step(0)) return win_busy_g[level_of(f)] = g, kWindowBusy;
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!calloc->shrink(top)) die("model: window: the pool's numbers %u.. are not all free after vacating them", top);
    size_t bytes[4];
    for (int a = 0; a < 4; ++a)
      bytes[a] = ((size_t)top * attn::kChunkEntries * kKvEntryBytes[a] + vmm::kChunk - 1) / vmm::kChunk * vmm::kChunk;
    slot_pool->shrink(0, bytes);
    slot_pool->trim(slot_pool->free_chunks());  // the physical memory goes back to the system
  }
  if (!win_groups) win_n0 = n0;
  win_groups += g;
  win_busy_g[level_of(f)] = 0;
  return kWindowOk;
}

// `groups` groups back into the pool: the numbers are made and mapped again (the driver must have the memory).
void Model::Impl::window_regrow(int groups) {
  if (groups < 1) return;
  const int left = win_groups - groups;
  if (left < 0) die("model: window: %d groups back of %d", groups, win_groups);
  const size_t to = win_numbers_for(left);
  if (to > calloc->numbers()) pool_grow(to - calloc->numbers());
  win_groups = left;
  if (!left) win_n0 = 0;
}

bool Model::Impl::win_alloc(int f) {
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
    window_regrow(g);
    if (rank == 0) std::printf("window: factor %d: too little memory on a rank for its %.2f GiB; the %d groups are back in the pool\n", f,
                               (bufs + gr) / 1073741824.0, g);
    return kWindowFailed;
  }
  // the graphs
  if (spec) {
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
  window_regrow(back);
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
  if (!lazy_win || !host_ring) return kWindowOk;
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
    win_owed += sc.groups;
    sc.groups = 0;
  }
  if (win_owed < 1) return kWindowOk;
  const size_t back = (size_t)win_owed * kGroupNums * chunk_image();
  const bool ok = free_device() >= back + 4 * kMargin;
  if (!min_of(*host_ring, ok)) return kWindowBusy;  // some rank's memory is still taken: owed until the next settle
  window_regrow(win_owed);
  if (rank == 0)
    std::printf("window: factor %d closed in %.2f s: %d groups back, the pool has %u numbers (%.2f GiB)\n", f, wall() - t0, win_owed,
                calloc->numbers(), calloc->total() * (double)chunk_image() / 1073741824.0);
  win_owed = 0;
  return kWindowOk;
}

bool Model::window_lazy() const { return d_->lazy_win; }
bool Model::window_is_open(int factor) const {
  if (factor != 16 && factor != 32 && factor != 64) return false;
  return !d_->lazy_win || d_->scan_[Impl::level_of(factor)].open;
}
int Model::window_open(int factor) { return d_->window_open(factor); }
int Model::window_close(int factor) { return d_->window_close(factor); }
int Model::window_owed() const { return d_->win_owed; }
int Model::window_groups() const { return d_->win_groups; }

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
// arithmetic and pool_vacate's holder scan, read only).
Model::WindowPlan Model::window_plan(int f, int g) const {
  WindowPlan p;
  const Impl& m = *d_;
  uint32_t top = 0;
  if (!m.calloc || !m.store_s || !m.window_top(f, g, &top)) return p;
  p.top = top;
  if (top >= m.calloc->numbers()) return p;
  p.removed = (int64_t)m.calloc->numbers() - (int64_t)top;
  for (const auto& [k, x] : m.store_s->sess)
    if (holds_from(x, top)) p.holders.push_back({k, x.line, k <= 0});
  return p;
}

}  // namespace model
