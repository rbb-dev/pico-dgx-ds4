// Copy drafting's memory and the kernels that fill it (copy.h is the matcher, copy_dev.h the declarations; spec.cu's gate
// kernels append each step's tokens and look up the proposals with the same functions).
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "common/check.h"
#include "model/copy_dev.h"

namespace model::copy {

namespace {

constexpr int kThreads = 256;

// set_tokens, in three kernels ordered on the stream: the length (and whether this is a rewrite), the table cleared on a
// rewrite, the grams of the new range indexed (a rewrite indexes the whole kept history again: an older gram that a stale
// entry's slot had displaced comes back).
__global__ void k_set_prep(Dev* dev, int unit, int pos0, int n, int32_t* flag) {
  Unit& u = dev->unit[unit];
  *flag = pos0 < u.len ? 1 : 0;
  u.len = pos0 + n;
}
__global__ void k_set_clear(Dev* dev, int unit, const int32_t* flag) {
  if (!*flag) return;
  Unit& u = dev->unit[unit];
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i <= u.mask; i += (size_t)gridDim.x * blockDim.x) u.tab[i] = 0;
}
__global__ void k_set_index(Dev* dev, int unit, int pos0, int n, const int32_t* flag) {
  const Unit& u = dev->unit[unit];
  const int from = *flag ? 0 : pos0;
  for (int e = from + (int)(blockIdx.x * blockDim.x + threadIdx.x); e < pos0 + n; e += gridDim.x * blockDim.x) index_one(u, e);
}

__global__ void k_propose_q(const Dev* dev, const Index::Query* q, int nq, int32_t* out, int32_t* counts, int32_t* matches) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nq) return;
  Match m;
  counts[i] = propose(dev->unit[q[i].unit], q[i].P, q[i].d, q[i].k, q[i].maxm, q[i].minlen, out + i * kMaxDrafts, &m);
  matches[3 * i] = m.e, matches[3 * i + 1] = m.len, matches[3 * i + 2] = m.bucket;
}
__global__ void k_append_one(Dev* dev, int unit, int pos0, int count, const int32_t* rows, int32_t bonus) {
  if (threadIdx.x == 0) append_step(dev->unit[unit], pos0, count, rows, bonus);
}

}  // namespace

Index::Index(const int* ctx, int nunits, size_t arena_bytes, cudaStream_t s) {
  if (nunits < 1 || nunits > kMaxUnitsCopy) die("copy: %d units", nunits);
  plan_.assign(nunits, Plan{0, 0});
  used_ = plan_units(ctx, nunits, arena_bytes, plan_.data());
  if (used_) {
    CUDA_CHECK(cudaMalloc(&arena_, used_));
    CUDA_CHECK(cudaMemsetAsync(arena_, 0, used_, s));
  }
  host_.assign(kMaxUnitsCopy, Unit{});
  size_t off = 0;
  auto up = [](size_t b) { return (b + kBlockAlign - 1) / kBlockAlign * kBlockAlign; };
  for (int u = 0; u < nunits; ++u) {
    const Plan& p = plan_[u];
    if (p.cap <= 0) continue;
    Unit& h = host_[u];
    h.tok = reinterpret_cast<int32_t*>(arena_ + off);
    off += up((size_t)p.cap * 4);
    h.tab = reinterpret_cast<uint32_t*>(arena_ + off);
    off += up(((size_t)1 << p.tab_log2) * 4);
    h.mask = (uint32_t)(((size_t)1 << p.tab_log2) - 1);
    h.cap = p.cap, h.len = 0;
  }
  Dev d{};
  std::copy(host_.begin(), host_.end(), d.unit);
  d.table = prior_table();
  adapt_ = (mode() & kAdapt) != 0;
  attached_.assign(kMaxUnitsCopy, -1);
  if (const char* f = getenv("DSGB10_COPY_TABLE"); f && *f && !load_table(f, &d.table)) {
    // every rank reads the table: a file present on one Spark only stops the others here
    if (access(f, R_OK) != 0) die("copy: DSGB10_COPY_TABLE=%s: cannot read it on this rank (%s)", f, std::strerror(errno));
    die("copy: DSGB10_COPY_TABLE=%s: not %d lines of %d numbers in (0, 1]", f, kBuckets, kMaxDrafts);
  }
  learn_init(&d.learn, d.table);  // the table is the learner's starting values (the file's, with DSGB10_COPY_TABLE)
  CUDA_CHECK(cudaMalloc(&dev_, sizeof(Dev)));
  CUDA_CHECK(cudaMemcpyAsync(dev_, &d, sizeof d, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMalloc(&flag_, 4));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

Index::~Index() {
  cudaFree(dev_);
  cudaFree(arena_);
  cudaFree(flag_);
  cudaFree(stage_);
}

void Index::set_tokens(int unit, int pos0, int n, const int32_t* tokens, cudaStream_t s) {
  if (unit < 0 || unit >= (int)plan_.size() || plan_[unit].cap <= 0 || pos0 < 0 || n <= 0) return;
  n = std::min(n, plan_[unit].cap - pos0);
  if (n <= 0) return;
  CUDA_CHECK(cudaMemcpyAsync(host_[unit].tok + pos0, tokens, (size_t)n * 4, cudaMemcpyHostToDevice, s));
  k_set_prep<<<1, 1, 0, s>>>(dev_, unit, pos0, n, flag_);
  const int blocks = std::min(1024, (int)((((size_t)host_[unit].mask + 1) + kThreads - 1) / kThreads));
  k_set_clear<<<blocks, kThreads, 0, s>>>(dev_, unit, flag_);
  k_set_index<<<std::min(1024, (n + kThreads - 1) / kThreads + 1), kThreads, 0, s>>>(dev_, unit, pos0, n, flag_);
  CUDA_CHECK(cudaGetLastError());
}

Unit Index::view(int unit) const {
  Dev d;
  CUDA_CHECK(cudaMemcpy(&d, dev_, sizeof d, cudaMemcpyDeviceToHost));
  return d.unit[unit];
}
std::vector<uint32_t> Index::table_copy(int unit) const {
  const Unit u = view(unit);
  std::vector<uint32_t> t((size_t)u.mask + 1);
  CUDA_CHECK(cudaMemcpy(t.data(), u.tab, t.size() * 4, cudaMemcpyDeviceToHost));
  return t;
}
std::vector<int32_t> Index::tokens_copy(int unit, int n) const {
  const Unit u = view(unit);
  std::vector<int32_t> t(std::min(n, u.cap));
  CUDA_CHECK(cudaMemcpy(t.data(), u.tok, t.size() * 4, cudaMemcpyDeviceToHost));
  return t;
}

void Index::propose_on_device(const std::vector<Query>& q, std::vector<int32_t>& out, std::vector<int32_t>& counts,
                              std::vector<int32_t>& matches, cudaStream_t s) {
  const int nq = (int)q.size();
  Query* dq;
  int32_t *dout, *dc, *dm;
  CUDA_CHECK(cudaMalloc(&dq, nq * sizeof(Query)));
  CUDA_CHECK(cudaMalloc(&dout, (size_t)nq * kMaxDrafts * 4));
  CUDA_CHECK(cudaMalloc(&dc, (size_t)nq * 4));
  CUDA_CHECK(cudaMalloc(&dm, (size_t)nq * 12));
  CUDA_CHECK(cudaMemcpyAsync(dq, q.data(), nq * sizeof(Query), cudaMemcpyHostToDevice, s));
  k_propose_q<<<(nq + 63) / 64, 64, 0, s>>>(dev_, dq, nq, dout, dc, dm);
  CUDA_CHECK(cudaGetLastError());
  out.assign((size_t)nq * kMaxDrafts, 0), counts.assign(nq, 0), matches.assign((size_t)nq * 3, 0);
  CUDA_CHECK(cudaMemcpyAsync(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(counts.data(), dc, counts.size() * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(matches.data(), dm, matches.size() * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaFree(dq), cudaFree(dout), cudaFree(dc), cudaFree(dm);
}

void Index::append_on_device(int unit, int pos0, int count, const int32_t* rows, int32_t bonus, cudaStream_t s) {
  int32_t* d;
  CUDA_CHECK(cudaMalloc(&d, (size_t)count * 4));
  CUDA_CHECK(cudaMemcpyAsync(d, rows, (size_t)count * 4, cudaMemcpyHostToDevice, s));
  k_append_one<<<1, 32, 0, s>>>(dev_, unit, pos0, count, d, bonus);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaFree(d);
}

void Index::attach(int unit, int handle, cudaStream_t s) {
  if (!adapt_ || unit < 0 || unit >= (int)plan_.size() || handle == attached_[unit]) return;
  attached_[unit] = handle;
  uint8_t* p = reinterpret_cast<uint8_t*>(dev_) + offsetof(Dev, ul) + (size_t)unit * sizeof(UnitLearn);
  CUDA_CHECK(cudaMemsetAsync(p, 0, sizeof(UnitLearn), s));
}

void Index::learn_copy(Learn* L, std::vector<UnitLearn>* units) const {
  Dev d;
  CUDA_CHECK(cudaMemcpy(&d, dev_, sizeof d, cudaMemcpyDeviceToHost));
  *L = d.learn;
  units->assign(d.ul, d.ul + kMaxUnitsCopy);
}

}  // namespace model::copy
