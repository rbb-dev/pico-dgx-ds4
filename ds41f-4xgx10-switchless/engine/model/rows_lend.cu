// The prefill's second set of row-sized buffers: tenant 4 of the pool (lend.h's table, tenant.h's driver). The set is the one
// rows_alloc makes for Options::prefill_rows_max rows, device memory and pinned host memory, lent by the pool as whole groups: OnDemand,
// floor 0, so the pool takes it back before any cached conversation is evicted for room (next_victim names an idle one first).
// Its body is a thread of its own: a grow makes every array of the set (zeroed, the way the standing set's are) inside
// alloc_seal::kRegion and reports Done only when every array exists; a drop waits until the Engram lookups of the set are done (a
// cancelled request's lookup may still be writing its pinned buffers), synchronises the device and frees every array, on the engine thread. A set that a
// rank cannot make fails the change on every rank (the driver's agreement): the groups are back in the pool and the standing set is
// all there is. The engine thread takes the made set into `Prefill::big` when the body is done; chunks use it only through
// Model::Impl::prefill_use after the driver has settled the tenant's groups (pf_big_ready).
//
// DSGB10_PFLOAN_FAIL=rank:R (fault builds): rank R's twentieth allocation of the next grow is refused, once, so that a half-made set
// is rolled back.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/alloc_seal.h"
#include "model/prefill.h"

namespace model {

using namespace detail;

namespace {

#ifdef DSGB10_TEST_FAULTS
bool loan_fault(int rank) {
  static bool fired = false;
  const char* env = std::getenv("DSGB10_PFLOAN_FAIL");
  int r = -1;
  if (fired || !env || std::sscanf(env, "rank:%d", &r) != 1 || r != rank) return false;
  fired = true;
  std::printf("rank %d: prefill loan fault injected\n", rank);
  return true;
}
#endif

// rows_alloc's allocator for the lent set: a refused call is a failed set, not a dead engine; every array is recorded so that a failed
// or finished set is freed whole.
struct LentAlloc {
  bool ok = true;
  uint64_t bytes = 0;
  int calls = 0, fail_at = -1;
  std::vector<std::pair<void*, bool>> owned;  // pointer, pinned host
  template <class T>
  T* d(size_t n) {
    if (!ok || ++calls == fail_at) return ok = false, nullptr;
    void* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) return cudaGetLastError(), ok = false, nullptr;
    if (cudaMemset(p, 0, n * sizeof(T)) != cudaSuccess || cudaStreamSynchronize(0) != cudaSuccess) {
      cudaGetLastError();
      cudaFree(p);
      return ok = false, nullptr;
    }
    owned.emplace_back(p, false);
    bytes += n * sizeof(T);
    return static_cast<T*>(p);
  }
  template <class T>
  T* h(size_t n) {
    if (!ok || ++calls == fail_at) return ok = false, nullptr;
    void* p = nullptr;
    if (cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped) != cudaSuccess) return cudaGetLastError(), ok = false, nullptr;
    std::memset(p, 0, n * sizeof(T));
    owned.emplace_back(p, true);
    bytes += n * sizeof(T);
    return static_cast<T*>(p);
  }
};

void free_owned(std::vector<std::pair<void*, bool>>& owned) {
  for (auto& o : owned) {
    if (o.second) cudaFreeHost(o.first);
    else cudaFree(o.first);
  }
  cudaGetLastError();
  owned.clear();
}

struct RowsBody {
  Model::Impl& m;
  uint64_t gb() const { return (uint64_t)lend::kGroupNums * m.chunk_image(); }
  int groups_needed() const { return (int)((rows_bytes(m.opt.prefill_rows_max) + gb() - 1) / gb()); }
  // the set is taken whole or not at all: a part of it holds no chunk, and groups held for nothing are memory lost
  tenant::BodyPlan plan(uint64_t want, int room) {
    tenant::BodyPlan p;
    const int g = want ? groups_needed() : 0;
    if (g > room) {
      p.status = lend::kPlanRefused;
      return p;
    }
    p.groups = g, p.bytes = (uint64_t)g * gb();
    const int have = m.tenants.groups_of(lend::kTenantPrefill);
    p.status = g == have ? lend::kPlanSame : g > have ? lend::kPlanGrow : lend::kPlanShrink;
    return p;
  }
  uint64_t size_for_groups(int groups) { return groups >= groups_needed() ? (uint64_t)groups_needed() * gb() : 0; }
  bool begin(uint64_t bytes) {
    Prefill& P = *m.pf;
    Prefill::Loan& L = P.loan;
    if (L.th.joinable()) L.th.join();
    CUDA_CHECK(cudaGetDevice(&L.device));
    L.before = L.shown;
    L.st = 1;
    const int rank = m.rank, rows = m.opt.prefill_rows_max;
    bool refuse = false;
#ifdef DSGB10_TEST_FAULTS
    refuse = bytes > 0 && loan_fault(rank);
#endif
    Prefill* p = &P;  // the body object is a temporary of the command; the set lives in the Prefill
    if (!bytes) {
      // The drop runs HERE, on the engine thread, at the command's place in the stream: the same point on every rank. A cudaFree (which
      // synchronises the whole device and holds back every launch) on a thread of its own would block this rank's next launches at a time
      // no other rank shares, while the steps the other ranks have launched wait for it in their collectives: the ranks then disagree about
      // how many exchanges ran, and the watchdog ends the engine. Nothing of the set is in use (the server asks only when no request reads
      // on it); the waits are for the lookups a cancelled request left in flight.
      const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      Prefill::Rows& r = P.big;
      if (r.cap > 0 && r.e_done)
        for (int i = 0; i < 4; ++i)
          while (__atomic_load_n(&r.e_done[i], __ATOMIC_ACQUIRE) < r.e_wait[i]) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      cudaDeviceSynchronize();
      free_owned(L.owned);
      r = Prefill::Rows{};
      L.cur = 0;
      L.st = 2;
      if (m.rank == 0)
        std::printf("store: prefill rows: the second set freed on the engine thread in %.0f ms\n",
                    1e3 * (std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0));
      return true;
    }
    L.th = std::thread([p, bytes, rows, refuse, rank] {
      (void)rank;
      Prefill::Loan& L = p->loan;
      cudaSetDevice(L.device);
      LentAlloc a;
      a.fail_at = refuse ? 20 : -1;
      Prefill::Rows r;
      {
        const alloc_seal::Scoped seal(alloc_seal::kRegion);  // this thread, for the lent set's allocations
        rows_alloc(r, rows, a);
      }
      if (!a.ok || cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        free_owned(a.owned);
        L.st = 3;
        return;
      }
      L.made = r, L.owned = std::move(a.owned), L.cur = a.bytes;
      L.st = 2;
    });
    return true;
  }
  tenant::BodyState state() {
    Prefill::Loan& L = m.pf->loan;
    const int s = L.st.load();
    if (s == 1) return tenant::kBodyRunning;
    if (L.th.joinable()) L.th.join();
    if (s == 2 && L.made.cap > 0) {  // a made set: the engine thread takes it
      m.pf->big = std::move(L.made);
      L.made = Prefill::Rows{};
    }
    if (s == 2) L.shown = L.cur;
    return s == 2 ? tenant::kBodyDone : tenant::kBodyFailed;
  }
  tenant::BodyInfo info() { return tenant::block_info(m.pf->loan.before, m.pf->loan.shown); }
  bool at_size(uint64_t bytes) { return (bytes > 0) == (m.pf->loan.shown > 0); }
  void ack() {}
  void say(tenant::Note n, const tenant::Change& c, const tenant::BodyInfo& ci, int groups, bool back) {
    if (m.rank != 0) return;
    switch (n) {
      case tenant::kNoteFailed:
        std::printf("store: prefill rows %d -> %d groups failed: %s; chunks stay at %d rows\n", c.from_groups, c.target_groups, c.reason,
                    m.opt.prefill_rows);
        break;
      case tenant::kNoteGrown:
        std::printf("store: prefill rows %d -> %d groups (%.0f MiB): %d groups left the pool; chunks of up to %d rows are possible; this rank's "
                    "set %.0f MiB\n", c.from_groups, c.target_groups, c.target_size / 1048576.0, groups, m.opt.prefill_rows_max,
                    ci.block_bytes / 1048576.0);
        break;
      case tenant::kNoteShrunk:
        std::printf("store: prefill rows %d -> %d groups: %d groups %s the pool; chunks of up to %d rows\n", c.from_groups, c.target_groups,
                    groups, back ? "back in" : "owed to", m.opt.prefill_rows);
        break;
      case tenant::kNoteStuck:
        std::printf("store: prefill rows change failed and a rank's set did not go back: its groups are owed\n");
        break;
      case tenant::kNoteShrinkProof:
        std::printf("store: prefill rows shrink: the pool's handles did not add up (counted)\n");
        break;
    }
  }
};

}  // namespace

// The lent set's change: whole or not at all, so one command whatever its groups (the take of 24 groups is ~140 ms of the engine thread,
// measured per group at 5.7 ms; stepping it would make the proof's net-zero reading wrong between the steps).
int Model::Impl::rows_tenant_lend(int phase, int64_t mib) {
  RowsBody b{*this};
  return phase == kLendBegin ? drv.begin(lend::kTenantPrefill, b, (uint64_t)std::max<int64_t>(mib, 0) << 20, false) : drv.poll(b);
}

bool Model::Impl::pf_big_ready() const {
  return pf && pf->loan.st.load() != 1 && pf->big.cap > 0 && !drv.active() && tenants.groups_of(lend::kTenantPrefill) > 0 &&
         tenants.pending_of(lend::kTenantPrefill) == 0;
}

int64_t Model::prefill_loan_mib() const {
  const Impl& I = *d_;
  if (I.opt.prefill_rows_max <= std::max(I.opt.prefill_rows, kReplay)) return 0;
  return (int64_t)((rows_bytes(I.opt.prefill_rows_max) + (1 << 20) - 1) >> 20);
}

int Model::prefill_loan_groups() const {
  const Impl& I = *d_;
  if (prefill_loan_mib() == 0) return 0;
  const uint64_t gb = group_bytes();
  return (int)((rows_bytes(I.opt.prefill_rows_max) + gb - 1) / gb);
}

}  // namespace model
