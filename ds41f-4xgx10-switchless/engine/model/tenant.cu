// The tenants of the chunk pool (lend.h's table, tenant.h's driver) on the engine: the pool's side and the ranks' agreement
// (TenantEnv), the Engram row cache's body, a test build's dummy device block, and the one command every tenant's change goes
// through. The Engram cache's block is host memory (engram/rowcache.h): to grow it, whole groups of the pool's numbers leave the
// pool (their chunks go back to the driver, so the system has the bytes), the cache's worker commits and populates the same bytes
// behind one address range, and three independent readings per rank (the pool's handle count, MemAvailable, the block's own
// resident pages) say the memory really moved. A shrink runs the other way. The work that takes time (populating pages,
// rebuilding the index) is the body's, beside the engine thread, which runs only two bounded commands per change (begin and
// poll). Every rank takes the same decisions from replicated values and one agreement per step, so the four ranks' pools stay
// equal; a failure on any rank is a failed hand-over on all (the tenant at its old size, the groups back in the pool or owed),
// never a dead engine. lend.h has the rules and tenant.h the steps; their host tests are model/tests/lend_test.cpp and
// tenant_test.cpp.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "common/alloc_seal.h"
#include "engram/engram.h"
#include "model/impl.h"
#include "model/lend_ops.h"

namespace model {

using namespace detail;

namespace {

uint64_t min_u(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine, comm::kCallTenant0);
  return *std::min_element(all, all + 4);
}
uint64_t max_u(comm::Ring& r, uint64_t mine) {
  uint64_t all[4];
  r.allgather_host(&mine, all, sizeof mine, comm::kCallTenant0 + 1);
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
// DSGB10_LEND_FAIL=rank:R,step:B|C|back|win[,skip:N][,tenant:T]: this rank, once, at that step's N+1-th occurrence (N = 0 when no skip
// is given; B: the check after the groups left; C: the check after the body made its part; back: the pool's grow when groups come
// back; win: the memory check after a window's buffers were made) of tenant T's changes (1, the Engram cache, when none is named). `skip:1` leaves the start's hand-over alone and fails
// the first change after it.
bool Model::Impl::lend_fault(const char* step) const {
  static int seen[4] = {0, 0, 0, 0};
  static bool fired[4] = {false, false, false, false};
  const char* env = std::getenv("DSGB10_LEND_FAIL");
  if (!env) return false;
  int r = -1, skip = 0, tn = 1;
  char st[16] = {};
  if (std::sscanf(env, "rank:%d,step:%15[^,]", &r, st) != 2 || r != rank) return false;
  if (const char* k = std::strstr(env, "skip:")) skip = std::max(0, std::atoi(k + 5));
  if (const char* k = std::strstr(env, "tenant:")) tn = std::atoi(k + 7);
  if (tn != fault_tenant) return false;
  const int i = !std::strcmp(step, "B") ? 0 : !std::strcmp(step, "C") ? 1 : !std::strcmp(step, "win") ? 3 : 2;
  if (std::strcmp(st, step) != 0 || fired[i]) return false;
  if (seen[i]++ < skip) return false;
  fired[i] = true;
  std::printf("rank %d: lend fault injected at step %s\n", rank, step);
  return true;
}
#endif

// ---- the pool's side and the agreement ----

size_t Model::Impl::TenantEnv::numbers() const { return m.calloc->numbers(); }
bool Model::Impl::TenantEnv::vacate(size_t top) { return LendOps{m}.vacate(top); }
bool Model::Impl::TenantEnv::make_room(size_t removed) { return LendOps{m}.make_room(removed); }
bool Model::Impl::TenantEnv::release(size_t top, int g) { return LendOps{m}.release(top, g); }
bool Model::Impl::TenantEnv::grow_all(size_t more) { return LendOps{m}.grow_all(more); }
uint64_t Model::Impl::TenantEnv::handles() const { return m.slot_pool->total(); }
uint64_t Model::Impl::TenantEnv::avail() const { return mem_avail(); }
uint64_t Model::Impl::TenantEnv::chunks_for(size_t n) const { return m.pool_chunks_for(n); }
uint64_t Model::Impl::TenantEnv::chunk_bytes() const { return vmm::kChunk; }
size_t Model::Impl::TenantEnv::whole() const { return m.lend_whole(); }
size_t Model::Impl::TenantEnv::room_whole() const { return m.lend_room_whole(); }
uint64_t Model::Impl::TenantEnv::null_bytes() const { return (uint64_t)m.opt.lend_null_mib << 20; }

// The groups of the windows the options allow that are not open: each is the largest estimate of any rank, one all-gather for
// both levels. Every tenant's room keeps them, so a request past 1M finds its window's room still there.
int Model::Impl::TenantEnv::later_groups() {
  const uint64_t gb = (uint64_t)lend::kGroupNums * m.chunk_image();
  uint64_t mine[2] = {0, 0}, all[4][2];
  int i = 0;
  for (const int f : {32, 64}) {
    if (m.opt.max_ctx > 65536 * (f / 2) && m.lazy_win && !m.scan_[Impl::level_of(f)].open) mine[i] = m.win_need(f);
    ++i;
  }
  m.host_ring->allgather_host(mine, all, sizeof mine, comm::kCallTenant0 + 2);
  int later = 0;
  for (int k = 0; k < 2; ++k) {
    uint64_t need = 0;
    for (int r = 0; r < 4; ++r) need = std::max(need, all[r][k]);
    later += (int)((need + gb - 1) / gb);
  }
  return later;
}

bool Model::Impl::TenantEnv::fault(tenant::Step s) {
#ifdef DSGB10_TEST_FAULTS
  if (s == tenant::kStepB) return m.lend_fault("B");
  if (s == tenant::kStepC) return m.lend_fault("C");
#else
  (void)s;
#endif
  return false;
}
uint64_t Model::Impl::TenantEnv::agree_min(uint64_t v, tenant::Step) { return min_u(*m.host_ring, v); }
bool Model::Impl::TenantEnv::agree_all(bool ok, tenant::Step) { return min_u(*m.host_ring, ok) != 0; }
bool Model::Impl::TenantEnv::agree_any(bool bad, tenant::Step) { return max_u(*m.host_ring, bad) != 0; }

// ---- the Engram cache's body ----

namespace {

struct EngramBody {
  Model::Impl& m;
  uint64_t cur_mib() const {
    const lend::Tenant* t = m.tenants.find(lend::kTenantEngram);
    return t ? t->size >> 20 : 0;
  }
  tenant::BodyPlan plan(uint64_t want, int room) {
    const uint64_t gb = (uint64_t)lend::kGroupNums * m.chunk_image(), row = engram::kRowBytes;
    const uint64_t disp = m.store->cache_display_bytes(), max_mib = m.store->cache_max_bytes() >> 20;
    const lend::Plan p = lend::plan_resize(want >> 20, cur_mib(), disp, row, gb, room, max_mib);
    tenant::BodyPlan b;
    b.status = p.status, b.bytes = p.mib << 20, b.groups = p.groups, b.cut = p.cut;
    return b;
  }
  uint64_t size_for_groups(int groups) {
    const uint64_t gb = (uint64_t)lend::kGroupNums * m.chunk_image(), max_mib = m.store->cache_max_bytes() >> 20;
    return lend::plan_resize(max_mib, 0, m.store->cache_display_bytes(), engram::kRowBytes, gb, groups, max_mib).mib << 20;
  }
  bool begin(uint64_t bytes) { return m.store->resize_begin(bytes); }
  tenant::BodyState state() {
    const engram::Store::Resize rs = m.store->resize_state();
    return rs == engram::Store::Resize::Running ? tenant::kBodyRunning : rs == engram::Store::Resize::Done ? tenant::kBodyDone : tenant::kBodyFailed;
  }
  tenant::BodyInfo info() {
    const engram::Store::CacheInfo ci = m.store->cache_info();
    tenant::BodyInfo i;
    i.block_bytes = ci.block_bytes, i.rss_before = ci.rss_before, i.rss_after = ci.rss_after, i.huge_after = ci.huge_after, i.ms = ci.resize_ms;
    return i;
  }
  bool at_size(uint64_t bytes) { return m.store->cache_info().slots == bytes / engram::kRowBytes; }
  void ack() { m.store->resize_ack(); }
  void say(tenant::Note n, const tenant::Change& c, const tenant::BodyInfo& ci, int groups, bool back) {
    const unsigned long long from = c.from_size >> 20, to = c.target_size >> 20;
    switch (n) {
      case tenant::kNoteFailed:
        m.el.reason = c.reason;
        if (m.rank == 0) std::printf("store: Engram cache %llu -> %llu MiB failed: %s; the cache stays at %llu MiB\n", from, to, c.reason, from);
        break;
      case tenant::kNoteGrown: {
        const uint64_t committed = ci.block_bytes > c.block_before ? ci.block_bytes - c.block_before : 0;
        if (m.rank == 0)
          std::printf("store: Engram cache %llu -> %llu MiB: %d groups (%.0f MiB) left the pool; this rank's block %.0f MiB resident, "
                      "%llu %% in huge pages, made in %llu ms\n", from, to, groups, groups * (double)lend::kGroupNums * m.chunk_image() / 1048576.0,
                      committed / 1048576.0, (unsigned long long)(ci.rss_after ? ci.huge_after * 100 / ci.rss_after : 0), (unsigned long long)ci.ms);
        break;
      }
      case tenant::kNoteShrunk:
        if (m.rank == 0) std::printf("store: Engram cache %llu -> %llu MiB: %d groups %s the pool\n", from, to, groups, back ? "back in" : "owed to");
        break;
      case tenant::kNoteStuck:
        if (m.rank == 0) std::printf("store: Engram cache change failed and a rank's block did not go back to %llu MiB: its groups are owed\n", from);
        break;
      case tenant::kNoteShrinkProof:
        if (m.rank == 0) std::printf("store: Engram cache shrink: the block's pages or the pool's handles did not add up (counted)\n");
        break;
    }
  }
};

}  // namespace

#ifdef DSGB10_TEST_FAULTS
// ---- a test build's dummy device block (tenant 15) ----
// The body's own thread allocates a block of the size asked for, fills it with a pattern (which touches every page, so reading C
// is taken after the memory is really used) and, before freeing it, reads the pattern back. A change is Done only when its part
// is made. The tests drive it with CONFIG's test_tenant_groups.
struct Model::Impl::DummyBlock {
  static constexpr uint8_t kPat = 0xA5;
  static constexpr size_t kPiece = 16u << 20;
  Model::Impl& m;
  void* dev = nullptr;
  std::atomic<uint64_t> cur{0}, before{0};  // the bytes held now, and when the last change began
  std::atomic<int> st{0};  // 0 idle, 1 running, 2 done, 3 failed
  std::atomic<bool> intact{true};
  std::thread th;
  int device = 0;
  explicit DummyBlock(Model::Impl& x) : m(x) {}
  ~DummyBlock() {
    if (th.joinable()) th.join();
    if (dev) cudaFree(dev);
  }
  // reads the block back against the pattern; the block is then freed
  bool verify_and_free() {
    if (!dev) return true;
    std::vector<uint8_t> host(kPiece);
    const uint64_t n = cur.load();
    bool ok = true;
    for (uint64_t off = 0; off < n && ok; off += kPiece) {
      const size_t len = (size_t)std::min<uint64_t>(kPiece, n - off);
      if (cudaMemcpy(host.data(), static_cast<uint8_t*>(dev) + off, len, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        ok = false;
        break;
      }
      for (size_t i = 0; i < len; ++i)
        if (host[i] != kPat) {
          ok = false;
          break;
        }
    }
    std::printf("rank %d: tenant 15: the block's pattern %s (%.0f MiB read back)\n", m.rank, ok ? "intact" : "BROKEN", n / 1048576.0);
    cudaFree(dev);
    dev = nullptr;
    cur = 0;
    return ok;
  }
  void run(uint64_t bytes) {
    const alloc_seal::Scoped seal(alloc_seal::kRegion);
    cudaSetDevice(device);
    bool ok = verify_and_free();
    if (!ok) intact = false;
    if (ok && bytes) {
      if (cudaMalloc(&dev, bytes) != cudaSuccess) {
        cudaGetLastError();
        dev = nullptr;
        ok = false;
      } else if (cudaMemset(dev, kPat, bytes) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        cudaFree(dev);
        dev = nullptr;
        ok = false;
      } else {
        cur = bytes;
      }
    }
    st = ok ? 2 : 3;
  }
};

namespace {

struct DummyBody {
  Model::Impl& m;
  Model::Impl::DummyBlock& b;
  uint64_t gb() const { return (uint64_t)lend::kGroupNums * m.chunk_image(); }
  tenant::BodyPlan plan(uint64_t want, int room) {
    tenant::BodyPlan p;
    int g = (int)((want + gb() - 1) / gb());
    if (g > room) g = room, p.cut = true;
    p.groups = g, p.bytes = (uint64_t)g * gb();
    const int have = m.tenants.groups_of(15);
    p.status = g == have ? lend::kPlanSame : g > have ? lend::kPlanGrow : lend::kPlanShrink;
    return p;
  }
  uint64_t size_for_groups(int groups) { return (uint64_t)groups * gb(); }
  bool begin(uint64_t bytes) {
    if (b.th.joinable()) b.th.join();
    cudaGetDevice(&b.device);
    b.before = b.cur.load();
    b.st = 1;
    Model::Impl::DummyBlock* blk = &b;  // the body object is a temporary of the command; the block lives in the Impl
    b.th = std::thread([blk, bytes] { blk->run(bytes); });
    return true;
  }
  tenant::BodyState state() {
    const int s = b.st.load();
    if (s == 1) return tenant::kBodyRunning;
    if (b.th.joinable()) b.th.join();
    return s == 2 ? tenant::kBodyDone : tenant::kBodyFailed;
  }
  tenant::BodyInfo info() {
    return tenant::block_info(b.before.load(), b.cur.load());
  }
  bool at_size(uint64_t bytes) { return b.cur.load() == bytes; }
  void ack() {}
  void say(tenant::Note n, const tenant::Change& c, const tenant::BodyInfo&, int groups, bool back) {
    if (m.rank != 0) return;
    switch (n) {
      case tenant::kNoteFailed: std::printf("store: tenant 15 %d -> %d groups failed: %s; it stays at %d groups\n", c.from_groups, c.target_groups, c.reason, c.from_groups); break;
      case tenant::kNoteGrown: std::printf("store: tenant 15 %d -> %d groups: %d groups left the pool\n", c.from_groups, c.target_groups, groups); break;
      case tenant::kNoteShrunk: std::printf("store: tenant 15 %d -> %d groups: %d groups %s the pool\n", c.from_groups, c.target_groups, groups, back ? "back in" : "owed to"); break;
      case tenant::kNoteStuck: std::printf("store: tenant 15 change failed and a rank's block did not go back: its groups are owed\n"); break;
      case tenant::kNoteShrinkProof: std::printf("store: tenant 15 shrink: the pool's handles did not add up (counted)\n"); break;
    }
  }
};

}  // namespace
#endif

// ---- the command ----

// Phase kLendBegin starts tenant `id`'s change to `mib` MiB, kLendPoll finishes it once every rank's body has. The same status on
// every rank (Model::LendStatus; tenant.h's Status has the same values).
int Model::Impl::tenant_lend(int id, int phase, int64_t mib, bool capped) {
  if (!host_ring || !calloc || !tenants.find(id)) return kLendIdle;
  struct Ctx {  // DSGB10_LEND_FAIL's `tenant:N` is matched against the tenant this command runs for
    int& t;
    int was;
    Ctx(int& x, int id) : t(x), was(x) { t = id; }
    ~Ctx() { t = was; }
  } ctx(fault_tenant, id);
  if (id == lend::kTenantEngram) {
    if (!el.enabled || !store) return kLendIdle;
    EngramBody b{*this};
    return phase == kLendBegin ? drv.begin(id, b, (uint64_t)std::max<int64_t>(mib, 0) << 20, capped) : drv.poll(b);
  }
  if (id == lend::kTenantPrefill) return rows_tenant_lend(phase, mib);
  if (id == lend::kTenantWork) return vit_tenant_lend(phase, mib, capped);  // the image encoder's work area (vit_area.cu)
#ifdef DSGB10_TEST_FAULTS
  if (id == 15) {
    if (!dummy) dummy = std::make_shared<DummyBlock>(*this);
    DummyBody b{*this, *dummy};
    return phase == kLendBegin ? drv.begin(id, b, (uint64_t)std::max<int64_t>(mib, 0) << 20, capped) : drv.poll(b);
  }
#endif
  return kLendIdle;
}

// The start's hand-over (main.cpp, before any request) is not capped: nothing decodes to wait for it.
int Model::Impl::engram_lend(int phase, int64_t mib) { return tenant_lend(lend::kTenantEngram, phase, mib, false); }

bool Model::engram_lend_enabled() const { return d_->el.enabled; }
int Model::engram_lend(int phase, int64_t mib) { return d_->engram_lend(phase, mib); }
int Model::tenant_lend(int tenant, int phase, int64_t mib) { return d_->tenant_lend(tenant, phase, mib); }
bool Model::tenant_active() const { return d_->drv.active(); }
bool Model::window_may_run() const { return d_->drv.queue.window_may_run(); }
int Model::tenant_change_id() const { return d_->drv.active() ? d_->drv.change().id : -1; }
int Model::tenant_victim() const { return d_->tenants.next_victim(); }
bool Model::tenant_table_ok() const { return d_->tenants.consistent(d_->lent); }
bool Model::tenant_other_enabled() const {
  for (int i = 0; i < d_->tenants.size(); ++i)
    if (d_->tenants.at(i).id > lend::kTenantEngram) return true;
  return false;
}
bool Model::pool_top_free(int groups) const {
  const Impl& I = *d_;
  if (!I.calloc || groups <= 0 || (size_t)groups * lend::kGroupNums >= I.calloc->numbers()) return false;
  const uint32_t top = (uint32_t)(I.calloc->numbers() - (size_t)groups * lend::kGroupNums);
  return !I.calloc->first_taken_from(top) && I.core->holders_from(top).image_numbers == 0;
}
int64_t Model::pool_free_numbers() const { return d_->calloc ? (int64_t)d_->calloc->free() : 0; }
int Model::tenant_group_count(int id) const { return d_->tenants.groups_of(id); }
void Model::tenant_mark_idle(int id, bool idle) {
  if (lend::Tenant* t = d_->tenants.find(id)) t->idle = idle;
}
uint64_t Model::group_bytes() const { return (uint64_t)lend::kGroupNums * d_->chunk_image(); }
std::vector<Model::TenantRow> Model::tenant_rows() const {
  std::vector<TenantRow> out;
  for (int i = 0; i < d_->tenants.size(); ++i) {
    const lend::Tenant& t = d_->tenants.at(i);
    TenantRow r;
    r.id = t.id, r.kind = t.kind, r.standing = t.standing, r.groups = t.groups, r.pending = t.pending, r.size_bytes = t.size, r.idle = t.idle;
    out.push_back(r);
  }
  return out;
}

Model::EngramLendInfo Model::engram_lend_info() const {
  EngramLendInfo i;
  const Impl& m = *d_;
  const lend::Tenant* t = m.tenants.find(lend::kTenantEngram);
  const tenant::Counts& c = m.drv.counts(lend::kTenantEngram);
  const bool mine = m.drv.active() && m.drv.change().id == lend::kTenantEngram;
  i.mib = t ? t->size >> 20 : 0, i.groups = t ? (uint64_t)t->groups : 0, i.ok_total = c.ok, i.failed_total = c.failed;
  i.pending_mib = mine ? m.drv.change().target_size >> 20 : 0, i.running = mine ? m.drv.change().phase : 0, i.last_status = c.last_status;
  i.reason = m.el.reason;
  if (m.store && m.el.enabled) {
    const engram::Store::CacheInfo ci = m.store->cache_info();
    i.slots = ci.slots, i.block_bytes = ci.block_bytes;
    i.huge_pct = ci.rss_after ? ci.huge_after * 100 / ci.rss_after : 0;
    const engram::StoreStats ss = m.store->stats();
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
