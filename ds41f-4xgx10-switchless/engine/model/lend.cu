// The Engram row cache as a borrower of the chunk pool's memory. The cache's block is host memory (engram/rowcache.h): to
// grow it, whole groups of the pool's numbers leave the pool (their chunks go back to the driver, so the system has the
// bytes), the cache's worker commits and populates the same bytes behind one address range, and three independent readings
// per rank (the pool's handle count, MemAvailable, the block's own resident pages) say the memory really moved. A shrink
// runs the other way. The work that takes time (populating pages, rebuilding the index) is the worker's, a slice at a time,
// beside its lookups; the engine thread runs only two bounded commands per change (begin and poll). Every rank takes the same
// decisions from replicated values and one agreement per step, so the four ranks' pools stay equal; a failure on any rank is a
// failed hand-over on all (the cache at its old size, the groups back in the pool or owed), never a dead engine. lend.h has the
// rules and their tests.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "engram/engram.h"
#include "model/impl.h"
#include "model/lend_ops.h"

namespace model {

using namespace detail;

namespace {

uint64_t min_u(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine);
  return *std::min_element(all, all + 4);
}
uint64_t max_u(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine);
  return *std::max_element(all, all + 4);
}

// /proc/meminfo's MemAvailable, bytes
uint64_t mem_avail() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t kb = 0;
  while (std::fgets(line, sizeof line, f))
    if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
  std::fclose(f);
  return (uint64_t)kb << 10;
}

}  // namespace

#ifdef DSGB10_TEST_FAULTS
// DSGB10_LEND_FAIL=rank:R,step:B|C|back[,skip:N]: this rank, once, at that step's N+1-th occurrence (N = 0 when no skip is
// given; B: the check after the groups left; C: the check after the block was populated; back: the pool's grow when groups
// come back). `skip:1` leaves the start's hand-over alone and fails the first change after it.
bool Model::Impl::lend_fault(const char* step) const {
  static int seen[3] = {0, 0, 0};
  static bool fired[3] = {false, false, false};
  const char* env = std::getenv("DSGB10_LEND_FAIL");
  if (!env) return false;
  int r = -1, skip = 0;
  char st[16] = {};
  if (std::sscanf(env, "rank:%d,step:%15[^,]", &r, st) != 2 || r != rank) return false;
  if (const char* k = std::strstr(env, "skip:")) skip = std::max(0, std::atoi(k + 5));
  const int i = !std::strcmp(step, "B") ? 0 : !std::strcmp(step, "C") ? 1 : 2;
  if (std::strcmp(st, step) != 0 || fired[i]) return false;
  if (seen[i]++ < skip) return false;
  fired[i] = true;
  std::printf("rank %d: lend fault injected at step %s\n", rank, step);
  return true;
}
#endif

void Model::Impl::lend_fail(const char* why) {
  el.failed_total++;
  el.reason = why;
  el.phase = 0;
  el.last_status = kLendFailed;
  if (rank == 0) std::printf("store: Engram cache %llu -> %llu MiB failed: %s; the cache stays at %llu MiB\n", (unsigned long long)el.from_mib,
                             (unsigned long long)el.target_mib, why, (unsigned long long)el.cur_mib);
}

int Model::Impl::engram_lend(int phase, int64_t mib) {
  if (!el.enabled || !store || !calloc || !host_ring) return kLendIdle;
  return phase == kLendBegin ? lend_begin((uint64_t)std::max<int64_t>(mib, 0)) : lend_poll();
}

// Phase 1: the plan, then (a grow) the groups out of the pool with readings A and B, then the worker started.
int Model::Impl::lend_begin(uint64_t want_mib) {
  if (el.phase != 0) return kLendPending;
  const uint64_t gb = (uint64_t)lend::kGroupNums * chunk_image(), row = engram::kRowBytes;
  const uint64_t disp = store->cache_display_bytes(), max_mib = store->cache_max_bytes() >> 20;
  const int others = lent.groups - el.cur_groups;  // windows' groups, and any owed
  const int room = lend_cache_room(others);
  lend::Plan plan = lend::plan_resize(want_mib, el.cur_mib, disp, row, gb, room, max_mib);
  const uint64_t least = min_u(*host_ring, plan.mib);  // every rank takes the smallest any can have
  if (least != plan.mib) plan = lend::plan_resize(least, el.cur_mib, disp, row, gb, room, max_mib);
  if (plan.status == lend::kPlanRefused) return el.last_status = kLendNoRoom;
  if (plan.status == lend::kPlanSame) return kLendSame;
  el.from_mib = el.cur_mib, el.from_groups = el.cur_groups;
  el.target_mib = plan.mib, el.target_groups = plan.groups;
  el.local_fail = false;
  el.tol = lend::tol_of((uint64_t)opt.lend_null_mib << 20);
  const uint64_t bytes = plan.mib << 20;
  if (plan.status == lend::kPlanShrink) {  // the worker drops the slots and the pages first; the groups come back at the poll
    el.local_fail = !store->resize_begin(bytes);
    el.block_before = store->cache_info().block_bytes;
    el.phase = 2;
    return kLendPending;
  }
  // a grow
  const int g = plan.groups - el.cur_groups;
  lend::Reading a;
  a.handles = slot_pool->total(), a.avail = mem_avail();
  const size_t before = calloc->numbers();
  if (g > 0) {
    LendOps ops{*this};
    lend::Lender<LendOps> L(ops, lent);
    const int st = L.take(g, lend_whole());
    if (st != lend::kOk) {  // replicated: nothing left the pool on any rank
      el.last_status = st == lend::kBusy ? kLendBusy : st == lend::kNoRoom ? kLendNoRoom : kLendFailed;
      return el.last_status;
    }
  }
  lend::Reading b;
  b.handles = slot_pool->total(), b.avail = mem_avail();
  const uint64_t expect = pool_chunks_for(before) - pool_chunks_for(calloc->numbers());
  int pr = lend::proof_ab(a, b, expect, expect * vmm::kChunk, el.tol);
#ifdef DSGB10_TEST_FAULTS
  if (lend_fault("B")) pr = lend::kProofAvail;
#endif
  if (!min_u(*host_ring, pr == lend::kProofOk)) {  // some rank's readings disagree: the groups go back on every rank
    if (g > 0 && !window_regrow(g)) lent.owed += g;
    lend_fail(pr == lend::kProofHandles ? "the pool's handle count did not fall by the groups' chunks"
                                        : pr == lend::kProofAvail ? "the memory did not reach the system" : "a rank's check failed");
    return kLendFailed;
  }
  el.a = a, el.b = b;
  el.block_before = store->cache_info().block_bytes;
  el.local_fail = !store->resize_begin(bytes);
  el.phase = 1;
  return kLendPending;
}

// Phase 3: once every rank's worker is done, the block's pages are checked and the change is final, or it goes back.
int Model::Impl::lend_poll() {
  if (el.phase == 0) return kLendIdle;
  const engram::Store::Resize rs = store->resize_state();
  if (max_u(*host_ring, rs == engram::Store::Resize::Running)) return kLendPending;  // some worker is still at it
  const engram::Store::CacheInfo ci = store->cache_info();
  if (el.phase == 1) {
    bool bad = el.local_fail || rs != engram::Store::Resize::Done;
#ifdef DSGB10_TEST_FAULTS
    bad = bad || lend_fault("C");
#endif
    bad = max_u(*host_ring, bad) != 0;
    store->resize_ack();
    const char* why = "a rank's worker could not make the block";
    if (!bad) {
      lend::Reading b = el.b, c;
      b.rss = ci.rss_before;
      c.handles = slot_pool->total(), c.avail = mem_avail(), c.rss = ci.rss_after;
      const uint64_t committed = ci.block_bytes > el.block_before ? ci.block_bytes - el.block_before : 0;
      const int pc = lend::proof_c(el.a, b, c, committed, el.tol);
      if (min_u(*host_ring, pc == lend::kProofOk)) {
        el.cur_mib = el.target_mib, el.cur_groups = el.target_groups;
        el.ok_total++;
        el.phase = 0;
        el.last_status = kLendDone;
        if (rank == 0)
          std::printf("store: Engram cache %llu -> %llu MiB: %d groups (%.0f MiB) left the pool; this rank's block %.0f MiB resident, "
                      "%llu %% in huge pages, made in %llu ms\n", (unsigned long long)el.from_mib, (unsigned long long)el.target_mib,
                      el.target_groups - el.from_groups, (el.target_groups - el.from_groups) * (double)lend::kGroupNums * chunk_image() / 1048576.0, committed / 1048576.0,
                      (unsigned long long)(ci.rss_after ? ci.huge_after * 100 / ci.rss_after : 0), (unsigned long long)ci.resize_ms);
        return kLendDone;
      }
      why = "the block's pages or the memory did not add up";
    }
    // back: the worker drops what it made (a worker that failed is at the old size already), then the groups come back
    el.reason = why;
    el.local_fail = !store->resize_begin(el.from_mib << 20);
    el.phase = 3;
    return kLendPending;
  }
  store->resize_ack();
  const uint64_t row = engram::kRowBytes;
  if (el.phase == 3) {
    // a rank whose block is not at the old size could not drop it: the groups are owed, not made again beside its pages
    const bool stuck = max_u(*host_ring, ci.slots != (el.from_mib << 20) / row) != 0;
    const int g = el.target_groups - el.from_groups;
    if (stuck) {
      el.proof_failed_total++;
      if (rank == 0) std::printf("store: Engram cache change failed and a rank's block did not go back to %llu MiB: its groups are owed\n", (unsigned long long)el.from_mib);
    }
    if (g > 0 && (stuck || !window_regrow(g))) lent.owed += g;
    lend_fail(el.reason);
    return kLendFailed;
  }
  // a shrink: a rank that could not drop its block fails the change on every rank (the groups never left the cache: they stay
  // out), the ranks that dropped theirs make them again
  if (max_u(*host_ring, el.local_fail || ci.slots != (el.target_mib << 20) / row) != 0) {
    el.reason = "a rank could not drop the block's pages";
    el.local_fail = !store->resize_begin(el.from_mib << 20);
    el.phase = 3;
    return kLendPending;
  }
  // a shrink: the pages are gone, the groups come back
  const int g = el.cur_groups - el.target_groups;
  const uint64_t h0 = slot_pool->total();
  const size_t n_before = calloc->numbers();
  const bool back = g <= 0 || window_regrow(g);
  if (!back) lent.owed += g;
  const uint64_t expect = back ? pool_chunks_for(calloc->numbers()) - pool_chunks_for(n_before) : 0;
  const uint64_t decommitted = el.block_before > ci.block_bytes ? el.block_before - ci.block_bytes : 0;
  if (back && lend::proof_shrink(ci.rss_before, ci.rss_after, decommitted, h0, slot_pool->total(), expect) != lend::kProofOk) {
    el.proof_failed_total++;
    if (rank == 0) std::printf("store: Engram cache shrink: the block's pages or the pool's handles did not add up (counted)\n");
  }
  el.cur_mib = el.target_mib, el.cur_groups = el.target_groups;
  el.ok_total++;
  el.phase = 0;
  el.last_status = kLendDone;
  if (rank == 0)
    std::printf("store: Engram cache %llu -> %llu MiB: %d groups %s the pool\n", (unsigned long long)el.from_mib, (unsigned long long)el.target_mib,
                g, back ? "back in" : "owed to");
  return kLendDone;
}

bool Model::engram_lend_enabled() const { return d_->el.enabled; }
int Model::engram_lend(int phase, int64_t mib) { return d_->engram_lend(phase, mib); }

Model::EngramLendInfo Model::engram_lend_info() const {
  EngramLendInfo i;
  const Impl::EngramLend& e = d_->el;
  i.mib = e.cur_mib, i.groups = (uint64_t)e.cur_groups, i.ok_total = e.ok_total, i.failed_total = e.failed_total;
  i.pending_mib = e.phase ? e.target_mib : 0, i.running = e.phase, i.last_status = e.last_status, i.reason = e.reason;
  if (d_->store && e.enabled) {
    const engram::Store::CacheInfo ci = d_->store->cache_info();
    i.slots = ci.slots, i.block_bytes = ci.block_bytes;
    i.huge_pct = ci.rss_after ? ci.huge_after * 100 / ci.rss_after : 0;
    const engram::StoreStats ss = d_->store->stats();
    i.lookups = ss.lookups, i.hits = ss.hits, i.evictions = ss.evictions;
  }
  return i;
}

uint64_t Model::engram_mib_for_groups(int groups) const {
  const Impl& m = *d_;
  if (!m.el.enabled || !m.store) return 0;
  const uint64_t gb = (uint64_t)lend::kGroupNums * m.chunk_image(), max_mib = m.store->cache_max_bytes() >> 20;
  return lend::plan_resize(max_mib, 0, m.store->cache_display_bytes(), engram::kRowBytes, gb, groups, max_mib).mib;
}

}  // namespace model
