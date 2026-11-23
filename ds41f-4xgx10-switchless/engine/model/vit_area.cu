// The image encoder's work area: tenant 3 of the pool (lend.h's table, tenant.h's driver). The area is device memory, Required, two
// groups, taken once by the start's hand-over (main.cpp), before any request and before kept conversations are read back, and held
// while the engine runs: no idle timer, no give-back, never named by next_victim. Its body is a thread of its own that allocates the
// block, zeroes every byte and synchronises before it reports Done, so the hand-over's last reading (MemAvailable on every rank) is
// taken after the memory is really used. A rank that cannot make it fails the change on every rank (the driver's agreement): the
// groups are back in the pool and the engine serves text, answering a pixel request with an error (vision_state 2). A pool whose room
// is under the groups the area needs refuses the take (0 groups, no block): never a part of the area.
//
// DSGB10_VIT_FAIL=rank:R,at:alloc (fault builds): rank R's allocation is refused.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/alloc_seal.h"
#include "model/impl.h"
#include "vision/vit_rank.h"

namespace model {

using namespace detail;

namespace {

constexpr int kMaxMerged = 1021;  // a span of at most 1,024 rows = lh (lw + 1) + 2 holds at most this many aligner tokens
constexpr uint64_t kAlign = 256;

uint64_t scratch_off() { return (vit::rank_work_bytes(vit::kMaxPatches, kMaxMerged) + kAlign - 1) / kAlign * kAlign; }

#ifdef DSGB10_TEST_FAULTS
// DSGB10_VIT_FAIL=rank:R,at:alloc|vote: this rank, once, at that point (the allocation of the work area; the vote of an image's rows).
bool vit_fault(int rank, const char* at) {
  static bool fired[2] = {false, false};
  const char* env = std::getenv("DSGB10_VIT_FAIL");
  int r = -1;
  char w[16] = {};
  if (!env || std::sscanf(env, "rank:%d,at:%15s", &r, w) != 2 || r != rank || std::strcmp(w, at) != 0) return false;
  bool& f = fired[std::strcmp(at, "alloc") == 0 ? 0 : 1];
  if (f) return false;
  f = true;
  std::printf("rank %d: vision fault injected at %s\n", rank, at);
  return true;
}
#endif

struct VitBody {
  Model::Impl& m;
  uint64_t gb() const { return (uint64_t)lend::kGroupNums * m.chunk_image(); }
  tenant::BodyPlan plan(uint64_t want, int room) {
    tenant::BodyPlan p;
    const int g = (int)((want + gb() - 1) / gb());
    if (g > room) {  // the area is taken whole or not at all: a part of it holds no image, and a block held for nothing is memory lost
      p.status = lend::kPlanRefused;
      return p;
    }
    p.groups = g, p.bytes = (uint64_t)g * gb();
    const int have = m.tenants.groups_of(lend::kTenantWork);
    p.status = g == have ? lend::kPlanSame : g > have ? lend::kPlanGrow : lend::kPlanShrink;
    return p;
  }
  uint64_t size_for_groups(int groups) { return (uint64_t)groups * gb(); }
  bool begin(uint64_t bytes) {
    Model::Impl::Vit& V = m.vit;
    if (V.th.joinable()) V.th.join();
    CUDA_CHECK(cudaGetDevice(&V.device));
    V.before = V.cur.load();
    V.st = 1;
    bool refuse = false;
#ifdef DSGB10_TEST_FAULTS
    refuse = bytes > 0 && vit_fault(m.rank, "alloc");
#endif
    Model::Impl::Vit* v = &V;  // the body object is a temporary of the command; the block lives in the Impl
    v->th = std::thread([v, bytes, refuse] {
      cudaSetDevice(v->device);
      bool ok = true;
      if (v->area) {
        cudaFree(v->area);
        v->area = nullptr;
        v->cur = 0, v->area_bytes = 0;
      }
      if (bytes && refuse) {
        ok = false;
      } else if (bytes) {
        void* p = nullptr;
        const alloc_seal::Scoped seal(alloc_seal::kRegion);  // this thread, for the work area's allocation
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
          cudaGetLastError();
          ok = false;
        } else if (cudaMemset(p, 0, bytes) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
          cudaGetLastError();
          cudaFree(p);
          ok = false;
        } else {
          v->area = p, v->cur = bytes, v->area_bytes = bytes;
        }
      }
      v->st = ok ? 2 : 3;
    });
    return true;
  }
  tenant::BodyState state() {
    Model::Impl::Vit& V = m.vit;
    const int s = V.st.load();
    if (s == 1) return tenant::kBodyRunning;
    if (V.th.joinable()) V.th.join();
    return s == 2 ? tenant::kBodyDone : tenant::kBodyFailed;
  }
  tenant::BodyInfo info() { return tenant::block_info(m.vit.before.load(), m.vit.cur.load()); }
  bool at_size(uint64_t bytes) { return m.vit.cur.load() == bytes; }
  void ack() {}
  void say(tenant::Note n, const tenant::Change& c, const tenant::BodyInfo& ci, int groups, bool back) {
    if (m.rank != 0) return;
    switch (n) {
      case tenant::kNoteFailed:
        std::printf("store: work area %d -> %d groups failed: %s; the engine has no image encoder until the next start\n", c.from_groups,
                    c.target_groups, c.reason);
        break;
      case tenant::kNoteGrown:
        std::printf("store: work area %d -> %d groups (%.0f MiB): %d groups left the pool; this rank's block %.0f MiB, zeroed before the "
                    "change was Done\n", c.from_groups, c.target_groups, c.target_size / 1048576.0, groups, ci.block_bytes / 1048576.0);
        break;
      case tenant::kNoteShrunk:
        std::printf("store: work area %d -> %d groups: %d groups %s the pool\n", c.from_groups, c.target_groups, groups, back ? "back in" : "owed to");
        break;
      case tenant::kNoteStuck:
        std::printf("store: work area change failed and a rank's block did not go back: its groups are owed\n");
        break;
      case tenant::kNoteShrinkProof:
        std::printf("store: work area shrink: the pool's handles did not add up (counted)\n");
        break;
    }
  }
};

}  // namespace

int Model::Impl::vit_tenant_lend(int phase, int64_t mib, bool capped) {
  VitBody b{*this};
  return phase == kLendBegin ? drv.begin(lend::kTenantWork, b, (uint64_t)std::max<int64_t>(mib, 0) << 20, capped) : drv.poll(b);
}

uint64_t Model::vision_area_bytes() { return scratch_off() + vit::digest_scratch_bytes(1024); }

void Model::set_vision(const vit::DevWeights* w) {
  Impl& I = *d_;
  if (!w || (int)w->blocks.size() != kVitBlocks) die("model: the image encoder holds %zu blocks, the command runs %d", w ? w->blocks.size() : 0, kVitBlocks);
  I.vit.w = w;
  if (!I.vit.stage) {
    I.vit.stage = halloc<uint8_t>(kVitStageBytes);
    CUDA_CHECK(cudaEventCreateWithFlags(&I.vit.stage_ev, cudaEventDisableTiming));
  }
  if (!I.tenants.find(lend::kTenantWork)) I.tenants.add(lend::kTenantWork, lend::kKindDevice, lend::kRequired);
}

int Model::vision_state() const {
  const Impl& I = *d_;
  if (!I.vit.w) return 0;
  const lend::Tenant* t = I.tenants.find(lend::kTenantWork);
  return t && t->groups > 0 && I.vit.area && I.vit.area_bytes >= vision_area_bytes() ? 1 : 2;
}

uint64_t Model::vision_area_held() const { return d_->vit.area ? d_->vit.area_bytes : 0; }

}  // namespace model
