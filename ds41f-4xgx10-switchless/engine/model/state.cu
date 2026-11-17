// Sessions' saved state (model.h state_*, state.h). Every copy is queued on the model's stream, so it
// lands in order with the steps around it, and a block freed on the host can be handed out again at
// once: whatever reads it later is queued after whatever wrote it before. Blocks come from the pool
// reserved at startup (state.h), so taking a snapshot inside the decode loop costs no allocation.
#include "model/state.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <unordered_map>
#include <iterator>
#include <set>

#include "common/keepfile.h"
#include "model/spec.h"
#include "model/spill.h"

namespace model {

using namespace detail;

namespace {

constexpr size_t kArr[4] = {256, 32, 64, 4};  // bytes per entry: KV, KV scales, index key, key scales

double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Page offset of kv slot k's array a (entries of kPagePos / ratio positions).
size_t page_off(int k, int a) {
  size_t off = 0;
  for (int j = 0; j < k; ++j) off += (size_t)(kPagePos / (j < 3 ? 2 : 1)) * kEntryBytes;
  const int E = kPagePos / (k < 3 ? 2 : 1);
  for (int j = 0; j < a; ++j) off += (size_t)E * kArr[j];
  return off;
}

// A window snapshot's block and an end snapshot's: the draft module's windows follow the layers' rows
size_t snap_bytes(bool spec) { return kSnapBytes + (spec ? kSpecSnapBytes : 0); }
size_t end_bytes(bool spec) { return kEndBytes + (spec ? kSpecEndBytes : 0); }

// What a restart needs (serve.cpp choose): the end snapshot at e when the pages hold [0, e) (saved >= e),
// a window snapshot at s when they hold its restart position q (s, or s - 1 for an odd s; saved >= q).
// The pages a restart at q reads are the first pages_for(q).
int pages_for(int q) { return (q + kPagePos - 1) / kPagePos; }
int snap_q(int s) { return s & 1 ? s - 1 : s; }
bool end_usable(int end_pos, int saved) { return end_pos >= 0 && end_pos <= saved; }
bool snap_usable(int s, int saved) { return snap_q(s) <= saved; }
// The latest position a session holding `saved` positions of pages, the end snapshot at end_pos (-1: none)
// and window snapshots at the keys of `snaps` can restart from: the end snapshot's, else the latest
// usable snapshot's q; -1 when there is none.
template <class M>
int restart_of(int saved, int end_pos, const M& snaps) {
  if (end_usable(end_pos, saved)) return end_pos;
  for (auto i = snaps.rbegin(); i != snaps.rend(); ++i)
    if (snap_usable(i->first, saved)) return snap_q(i->first);
  return -1;
}

// A block's checksum for the disk tier's round trip: every 8-byte word mixed with its index, summed.
// kSumCtas CTAs each sum a stride of it; the last one to finish adds their partial sums in order and
// stores the result in *out (pinned memory) as one store, so nothing is zeroed first and the sum does
// not depend on which CTA finished when. Launches are serialized on the model's stream, so they share
// the partials and the counter.
constexpr int kSumCtas = 16;
__device__ uint64_t g_sum_part[kSumCtas];
__device__ unsigned g_sum_done;

__global__ void block_sum(const uint64_t* p, size_t n, uint64_t* out) {
  __shared__ uint64_t part[32];
  uint64_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t x = p[i] + i * 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    acc += x ^ (x >> 27);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x >= 32) return;
  acc = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0;
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (threadIdx.x) return;
  g_sum_part[blockIdx.x] = acc;
  __threadfence();
  if (atomicAdd(&g_sum_done, 1u) != gridDim.x - 1) return;
  __threadfence();  // the last CTA: every partial sum is in
  uint64_t s = 0;
  for (unsigned j = 0; j < gridDim.x; ++j) s += ((volatile uint64_t*)g_sum_part)[j];
  *out = s;
  g_sum_done = 0;
}

// over the bytes the disk tier copies: the block rounded to the pool's grain
void sum_block(const void* p, size_t bytes, uint64_t* out, cudaStream_t st) {
  block_sum<<<kSumCtas, 1024, 0, st>>>((const uint64_t*)p, Pool::round(bytes) / 8, out);
}

}  // namespace

size_t Pool::take_off(size_t bytes) {
  const size_t n = round(bytes);
  auto best = free.end();
  for (auto it = free.begin(); it != free.end(); ++it)
    if (it->second >= n && (best == free.end() || it->second < best->second)) best = it;
  if (best == free.end()) return kNone;
  const size_t off = best->first, len = best->second;
  free.erase(best);
  if (len > n) free.emplace(off + n, len - n);
  return off;
}

size_t Pool::take_fit(size_t bytes, const std::function<bool(size_t, size_t)>& ok) {
  const size_t n = round(bytes);
  for (auto it = free.begin(); it != free.end(); ++it) {
    if (it->second < n || !ok(it->first, n)) continue;
    const size_t off = it->first, len = it->second;
    free.erase(it);
    if (len > n) free.emplace(off + n, len - n);
    return off;
  }
  return kNone;
}

uint8_t* Pool::take(size_t bytes) {
  const size_t off = take_off(bytes);
  return off == kNone ? nullptr : base + off;
}

void Pool::give(uint8_t* p, size_t bytes) { give_off((size_t)(p - base), bytes); }

bool Pool::take_at(size_t off, size_t bytes) {
  const size_t n = round(bytes);
  auto it = free.upper_bound(off);
  if (it == free.begin()) return false;
  --it;
  const size_t fo = it->first, fl = it->second;
  if (off + n > fo + fl) return false;
  free.erase(it);
  if (off > fo) free.emplace(fo, off - fo);
  if (off + n < fo + fl) free.emplace(off + n, fo + fl - off - n);
  return true;
}

size_t Pool::largest() const {
  size_t n = 0;
  for (const auto& f : free) n = std::max(n, f.second);
  return n;
}

void Pool::give_off(size_t off, size_t bytes) {
  size_t len = round(bytes);
  auto next = free.lower_bound(off);
  if (next != free.end() && next->first == off + len) {
    len += next->second;
    next = free.erase(next);
  }
  if (next != free.begin()) {
    const auto prev = std::prev(next);
    if (prev->first + prev->second == off) {
      off = prev->first;
      len += prev->second;
      free.erase(prev);
    }
  }
  free.emplace(off, len);
}

Store::~Store() = default;

Store& Model::Impl::store_of() {
  if (!store_s) store_s = std::make_unique<Store>();
  return *store_s;
}

// ---- kept sessions ---------------------------------------------------------------------------------
//
// A planned stop's index of the disk tier's file (base.index beside base.spill; keepfile.h: mode 0600
// in a 0700 directory, whole or not at all): a header, then every session in the file with its blocks'
// offsets and their checksums as they went out. Every rank writes its own with the same gen; the next
// start reuses the files only when every rank's index is whole, of this run's id, young enough, and of
// the same gen (main.cpp).

namespace {

constexpr char kKeptMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'K', '1'};

struct KeptHeader {
  char magic[8];
  uint64_t gen, run_id, file_bytes;
  int64_t written;  // unix time, seconds
  int32_t rank, n_sess;
  uint64_t clock;                              // the store's
  uint64_t page_bytes, snap_bytes, end_bytes;  // its layout
  uint64_t body_bytes, body_hash;
};

}  // namespace

uint64_t Model::Impl::state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                       std::string* why) {
  Store& s = store_of();
  s.kept.clear();
  s.kept_gen = s.kept_clock = 0;
  *bytes = 0;
  std::vector<uint8_t> f;
  if (!keepfile::read_private(base + ".index", 256u << 20, &f, why)) return 0;
  auto bad = [&](const std::string& w) {
    *why = base + ".index: " + w;
    s.kept.clear();
    return (uint64_t)0;
  };
  KeptHeader hd;
  if (f.size() < sizeof hd) return bad("too short");
  std::memcpy(&hd, f.data(), sizeof hd);
  const bool sp = spec != nullptr;
  const double age = (double)std::time(nullptr) - (double)hd.written;
  if (std::memcmp(hd.magic, kKeptMagic, sizeof kKeptMagic) != 0) return bad("not a kept index");
  if (hd.run_id != run_id) return bad("of another engine binary, weights or options");
  if (hd.rank != rank) return bad("of another rank");
  if (age > max_age_s || age < -600) return bad("written " + std::to_string((long long)age) + " s ago");
  if (hd.page_bytes != kPageBytes || hd.snap_bytes != snap_bytes(sp) || hd.end_bytes != end_bytes(sp))
    return bad("of another layout");
  if (!hd.gen || hd.body_bytes != f.size() - sizeof hd || keepfile::fnv(f.data() + sizeof hd, hd.body_bytes) != hd.body_hash)
    return bad("incomplete or damaged");
  if (!hd.file_bytes || hd.file_bytes % kPoolGrain || hd.n_sess < 0) return bad("a bad header");
  // every block on the grain and inside the file, none overlapping another
  std::vector<std::pair<size_t, size_t>> blocks;
  auto block = [&](size_t off, size_t n) {
    blocks.emplace_back(off, Pool::round(n));
    return off % kPoolGrain == 0 && off < hd.file_bytes && Pool::round(n) <= hd.file_bytes - off;
  };
  keepfile::Reader r{f.data() + sizeof hd, (size_t)hd.body_bytes};
  std::set<int> hs;
  const int max_pages = opt.max_ctx / kPagePos + 1;
  for (int i = 0; i < hd.n_sess && r.ok; ++i) {
    Store::Kept k;
    k.h = r.get<int32_t>(), k.saved = r.get<int32_t>(), k.end_pos = r.get<int32_t>();
    const int np = r.get<int32_t>(), ns = r.get<int32_t>(), he = r.get<int32_t>();
    k.used = r.get<uint64_t>();
    if (!r.ok || k.h <= 0 || !hs.insert(k.h).second || np < 0 || np > max_pages || ns < 0 || ns > 4096 || (he != 0 && he != 1) ||
        k.saved < 0 || k.saved > np * kPagePos || k.end_pos < -1 || k.end_pos > opt.max_ctx)
      return bad("a bad session record");
    for (int j = 0; j < np; ++j) {
      const size_t off = r.get<uint64_t>();
      const uint64_t sum = r.get<uint64_t>();
      if (!block(off, kPageBytes)) return bad("a block outside the file");
      k.pages.emplace_back(off, sum);
    }
    for (int j = 0; j < ns; ++j) {
      const int pos = r.get<int32_t>();
      (void)r.get<int32_t>();
      const size_t off = r.get<uint64_t>();
      const uint64_t sum = r.get<uint64_t>();
      if (pos < kSnapRows || pos > opt.max_ctx || !k.snaps.emplace(pos, std::make_pair(off, sum)).second ||
          !block(off, snap_bytes(sp)))
        return bad("a bad snapshot record");
    }
    if (he) {
      k.dend = r.get<uint64_t>(), k.dend_sum = r.get<uint64_t>();
      if (!block(k.dend, end_bytes(sp))) return bad("a block outside the file");
    }
    s.kept.push_back(std::move(k));
  }
  if (!r.ok || r.at != r.n) return bad("a bad body");
  std::sort(blocks.begin(), blocks.end());
  for (size_t i = 1; i < blocks.size(); ++i)
    if (blocks[i - 1].first + blocks[i - 1].second > blocks[i].first) return bad("overlapping blocks");
  struct stat st;  // the file beside it: ours, 0600, of the size the index names
  if (lstat((base + ".spill").c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != getuid() ||
      (st.st_mode & 0777) != 0600 || (uint64_t)st.st_size != hd.file_bytes)
    return bad("its file " + base + ".spill is missing or changed");
  s.kept_gen = hd.gen, s.kept_clock = hd.clock;
  *bytes = hd.file_bytes;
  why->clear();
  return hd.gen;
}

void Model::Impl::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  Store& s = store_of();
  if (s.disk) die("state: the disk tier is opened once");
  if (reuse && !s.kept_gen) die("state: reusing a kept file that was not checked");
  bytes = bytes / kPoolGrain * kPoolGrain;
  s.base = base, s.run_id = run_id;
  if (!bytes) return;
  const double t0 = wall_s();
  s.disk = std::make_unique<SpillFile>(base, bytes, staging, reuse);
  s.dpool.size = s.disk->size();
  s.dpool.free.emplace(0, s.dpool.size);
  unlink((base + ".index").c_str());  // whatever happens from now on, a crash starts clean
  unlink((base + ".index.new").c_str());
  if (reuse) {  // the kept sessions, spilled ones: back on demand, each block checked (state_unspill)
    const bool sp = spec != nullptr;
    auto take = [&](size_t off, size_t n, uint64_t sum) {
      if (!s.dpool.take_at(off, n)) die("state: a kept block at %zu is not free", off);
      s.dbytes += Pool::round(n);
      *s.disk->sum(off, false) = sum;
    };
    s.clock = s.kept_clock;
    for (const Store::Kept& k : s.kept) {
      Store::Sess& ss = s.sess[k.h];
      ss.disk = ss.kept = true;
      ss.saved = k.saved, ss.end_pos = k.end_pos, ss.used = k.used;
      for (const auto& [off, sum] : k.pages) take(off, kPageBytes, sum), ss.dpages.push_back(off);
      for (const auto& [pos, os] : k.snaps) take(os.first, snap_bytes(sp), os.second), ss.dsnaps.emplace(pos, os.first);
      if (k.dend != Pool::kNone) take(k.dend, end_bytes(sp), k.dend_sum), ss.dend = k.dend;
      s.clock = std::max(s.clock, k.used);
    }
    s.loaded_gen = s.kept_gen;
    if (rank == 0)
      std::printf("state: %zu kept sessions in %s (%.2f of %.2f GiB), in the store in %.1f ms\n", s.kept.size(),
                  s.disk->path().c_str(), s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0, 1e3 * (wall_s() - t0));
  }
  s.kept.clear();
  s.kept.shrink_to_fit();
}

// A planned stop (every rank alike, after the server spilled what it keeps): the file synced, then its
// index, and the file stays at exit. A rank whose index cannot be written keeps nothing (the next
// start's gens differ: every rank starts clean).
void Model::Impl::state_keep(uint64_t gen) {
  Store& s = store_of();
  if (!s.disk || !gen) return;
  const double t0 = wall_s();
  CUDA_CHECK(cudaStreamSynchronize(st));  // the checksums of what went out are in
  s.disk->sync();
  const double t1 = wall_s();
  const bool sp = spec != nullptr;
  std::vector<uint8_t> f(sizeof(KeptHeader));
  using keepfile::put;
  int n = 0;
  for (const auto& [h, ss] : s.sess) {
    if (!ss.disk) continue;
    ++n;
    put<int32_t>(f, h), put<int32_t>(f, ss.saved), put<int32_t>(f, ss.end_pos);
    put<int32_t>(f, (int32_t)ss.dpages.size()), put<int32_t>(f, (int32_t)ss.dsnaps.size());
    put<int32_t>(f, ss.dend != Pool::kNone), put<uint64_t>(f, ss.used);
    for (const size_t o : ss.dpages) put<uint64_t>(f, o), put<uint64_t>(f, *s.disk->sum(o, false));
    for (const auto& [pos, o] : ss.dsnaps)
      put<int32_t>(f, pos), put<int32_t>(f, 0), put<uint64_t>(f, o), put<uint64_t>(f, *s.disk->sum(o, false));
    if (ss.dend != Pool::kNone) put<uint64_t>(f, ss.dend), put<uint64_t>(f, *s.disk->sum(ss.dend, false));
  }
  KeptHeader hd{};
  std::memcpy(hd.magic, kKeptMagic, sizeof kKeptMagic);
  hd.gen = gen, hd.run_id = s.run_id, hd.file_bytes = s.disk->size(), hd.written = (int64_t)std::time(nullptr);
  hd.rank = rank, hd.n_sess = n, hd.clock = s.clock;
  hd.page_bytes = kPageBytes, hd.snap_bytes = snap_bytes(sp), hd.end_bytes = end_bytes(sp);
  hd.body_bytes = f.size() - sizeof hd;
  hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
  std::memcpy(f.data(), &hd, sizeof hd);
  std::string why;
  if (!keepfile::write_atomic(s.base + ".index", f.data(), f.size(), &why)) {
    std::printf("state: rank %d keeps nothing: %s\n", rank, why.c_str());
    return;
  }
  s.disk->keep();
  if (rank == 0)
    std::printf("state: %d sessions kept for the next start (%.2f GiB of the file): synced in %.1f ms, index of %.1f KiB "
                "in %.1f ms\n", n, s.dbytes / 1073741824.0, 1e3 * (t1 - t0), f.size() / 1024.0, 1e3 * (wall_s() - t1));
}

void Model::Impl::state_reserve(size_t bytes) {
  Store& s = store_of();
  if (s.gather) die("state: the store is reserved once");
  if (!calloc) die("state: the store needs the chunk pool");
  // a page or a bookmark at a time between the chunks and the disk tier (the draft module's windows counted in)
  s.gather = dalloc<uint8_t>(Pool::round(std::max({kPageBytes, end_bytes(true), snap_bytes(true)})));
  const size_t more = bytes / chunk_image();
  if (!more) return;
  const double t0 = wall_s();
  pool_grow(more);
  store_numbers = more;
  if (rank == 0)
    std::printf("state: the store's %zu chunk numbers (%.2f GiB) join the pool, made and mapped in %.2f s: %u numbers (%.2f GiB) "
                "for the lines and the sessions kept between requests together\n",
                more, more * (double)chunk_image() / 1073741824.0, wall_s() - t0, calloc->total(),
                calloc->total() * (double)chunk_image() / 1073741824.0);
}

size_t Model::Impl::chunk_image() const {
  return (size_t)attn::kChunkEntries * (kKvEntryBytes[0] + kKvEntryBytes[1] + kKvEntryBytes[2] + kKvEntryBytes[3]);
}

namespace {
constexpr size_t kBk = (size_t)attn::kChunkEntries * 256;  // a bookmark number's bytes: its KV slice, 2 MiB
size_t bk_nums(size_t bytes) { return (bytes + kBk - 1) / kBk; }
bool in(const std::vector<int>& v, int k) { return std::find(v.begin(), v.end(), k) != v.end(); }
// session h (made if new) for an operation that needs its blocks: back from the disk tier first
Store::Sess& use_blocks(Model::Impl& m, int h) {
  Store& s = m.store_of();
  const auto it = s.sess.find(h);
  if (it != s.sess.end() && it->second.disk) m.state_unspill(h);
  return s.use(h);
}
// numbers kv source k's list needs for positions [0, pos): its entries and the step's reach past them (model.cu)
int nums_for_pos(int k, int pos) {
  if (pos <= 0) return 0;
  const int ratio = k < 3 ? 2 : 1, entries = (pos - 1) / ratio + 1 + 64;
  return (entries + attn::kChunkEntries - 1) / attn::kChunkEntries;
}
}  // namespace

uint8_t* Model::Impl::bk_at(const Nums& n, size_t off) const {
  return pool_arr[0] + ((size_t)n[off / kBk] << attn::kChunkShift) * kKvEntryBytes[0] + off % kBk;
}

void Model::Impl::bk_copy(const Nums& n, size_t off, uint8_t* dev, size_t bytes, bool to_bk) {
  while (bytes) {
    const size_t m = std::min(bytes, kBk - off % kBk);
    uint8_t* b = bk_at(n, off);
    CUDA_CHECK(cudaMemcpyAsync(to_bk ? b : dev, to_bk ? dev : b, m, cudaMemcpyDeviceToDevice, st));
    off += m, dev += m, bytes -= m;
  }
}

bool Model::Impl::take_nums(int h, size_t n, Nums* out) {
  while (calloc->free() < n)
    if (!store_evict_step(h)) return false;
  const size_t i0 = out->size();
  calloc->take((uint32_t)n, out);
  if (refs.size() < calloc->numbers()) refs.resize(calloc->numbers(), 0);
  for (size_t i = i0; i < out->size(); ++i) refs[(*out)[i]] = 1;
  if (pool_poison)  // a taken number holds 0xFF until written: a read of what was not written shows
    for (size_t i = i0; i < out->size(); ++i)
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemsetAsync(pool_arr[a] + ((size_t)(*out)[i] << attn::kChunkShift) * kKvEntryBytes[a], 0xFF,
                                   (size_t)attn::kChunkEntries * kKvEntryBytes[a], st));
  held_peak = std::max(held_peak, (size_t)calloc->held());
  return true;
}

void Model::Impl::give_nums(Nums& v) {
  for (const uint32_t c : v) {
    if (c >= refs.size() || !refs[c]) die("state: number %u given back with no list naming it", c);
    if (--refs[c]) continue;  // another list or bookmark still names it
    if (pool_poison)  // a number given back holds 0xFD until taken again: a read through a stale list shows
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemsetAsync(pool_arr[a] + ((size_t)c << attn::kChunkShift) * kKvEntryBytes[a], 0xFD,
                                   (size_t)attn::kChunkEntries * kKvEntryBytes[a], st));
    calloc->give(c);
  }
  v.clear();
}

void Model::Impl::share_nums(const Nums& from, size_t n, Nums* out) {
  if (n > from.size()) die("state: %zu numbers shared from a list of %zu", n, from.size());
  for (size_t i = 0; i < n; ++i) {
    const uint32_t c = from[i];
    if (c >= refs.size() || !refs[c]) die("state: number %u shared with no list naming it", c);
    ++refs[c];
    out->push_back(c);
  }
}

// One step of making room for session h: the least recently used session that the current command does not use and
// no line holds is evicted (every rank does the same): spilled to the disk tier, or freed without one. False when
// none is left.
bool Model::Impl::store_evict_step(int h) {
  Store& s = store_of();
  int victim = -1;
  uint64_t oldest = UINT64_MAX;
  for (const auto& [k, ss] : s.sess)
    if (k != h && k > 0 && !ss.disk && ss.line < 0 && ss.numbers() && ss.used < oldest && !in(s.pinned, k) && !in(s.live, k))
      victim = k, oldest = ss.used;
  if (victim < 0) return false;
  state_spill(victim);
  return true;
}

// The chunk numbers of sessions no line holds and `keep` does not name: what eviction can give back. The same on
// every rank (it reads only the store's records).
size_t Model::Impl::state_giveable(const std::vector<int>& keep) const {
  if (!store_s) return 0;
  std::unordered_map<uint32_t, uint32_t> seen;  // a shared number goes back only when every list naming it does
  const auto add = [&](const Nums& v) {
    for (const uint32_t c : v) ++seen[c];
  };
  for (const auto& [k, ss] : store_s->sess)
    if (k > 0 && !ss.disk && ss.line < 0 && !in(keep, k)) {
      for (const Nums& l : ss.lists) add(l);
      for (const auto& sn : ss.snaps) add(sn.second);
      add(ss.end);
    }
  size_t n = 0;
  for (const auto& [c, m] : seen) n += m == refs[c];
  return n;
}

size_t Model::state_restore_chunks(int h) const {
  if (!d_->store_s) return 0;
  const auto it = d_->store_s->sess.find(h);
  if (it == d_->store_s->sess.end() || !it->second.disk) return 0;
  const Store::Sess& ss = it->second;
  const bool sp = d_->spec != nullptr;
  size_t n = 0;
  for (int k = 0; k < kKvSlots; ++k) n += (size_t)nums_for_pos(k, std::min(ss.saved, (int)ss.dpages.size() * kPagePos));
  return n + ss.dsnaps.size() * bk_nums(snap_bytes(sp)) + (ss.dend != Pool::kNone ? bk_nums(end_bytes(sp)) : 0);
}

// ---- lines: a slot's unit holds a session (its rows are the session's lists) ------------------------------------

namespace {

// slot s's session: steps with no session named (tools, tests) get one of the unit's own, freed at release
Store::Sess& line_sess(Model::Impl& m, int s) {
  Store& st_ = m.store_of();
  const int u = m.unit_of[s];
  if (m.line_h[u] == Model::Impl::kNoLine) {
    m.line_h[u] = -1 - u;
    st_.sess[m.line_h[u]].line = u;
  }
  return st_.sess.at(m.line_h[u]);
}

// unit u's rows from session x's lists (the guard past their ends); the stream waits first, so no queued copy of the
// mirror reads an entry being rewritten
void write_rows(Model::Impl& m, int u, const Store::Sess* x) {
  CUDA_CHECK(cudaStreamSynchronize(m.st));
  for (int k = 0; k < kKvSlots; ++k) {
    uint32_t* row = m.lists_h + ((size_t)k * m.nunits + u) * m.list_len;
    std::fill(row, row + m.list_len, 0u);
    if (x && x->lists[k].size() > (size_t)m.list_len)
      die("model: kv source %d's list of %zu numbers is longer than a row (%d)", k, x->lists[k].size(), m.list_len);
    if (x) std::copy(x->lists[k].begin(), x->lists[k].end(), row);
    CUDA_CHECK(cudaMemcpyAsync(m.lists_d + ((size_t)k * m.nunits + u) * m.list_len, row, m.list_len * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, m.st));
  }
}

}  // namespace

int Model::Impl::held(int s) const {
  const int h = line_h[unit_of[s]];
  if (h == kNoLine || !store_s) return 0;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return 0;
  int n = 0;
  for (const Nums& l : it->second.lists) n += (int)l.size();
  return n;
}

bool Model::Impl::grow(int s, int upto) {
  const int u = unit_of[s];
  upto = std::max(0, std::min(upto, slot_ctx_[s] - 1));
  Store::Sess& x = line_sess(*this, s);
  const int h = line_h[u];
  int want[kKvSlots];
  size_t total = 0;
  for (int k = 0; k < kKvSlots; ++k) {
    want[k] = std::max(0, std::min(list_len, nums_for_pos(k, upto + 1)) - (int)x.lists[k].size());
    total += want[k];
  }
  if (!total) return true;
  Nums got;
  store_of().pinned.push_back(h);
  const bool ok = take_nums(h, total, &got);
  store_of().pinned.pop_back();
  if (!ok) return false;
  size_t g = 0;
  for (int k = 0; k < kKvSlots; ++k) {
    if (!want[k]) continue;
    const size_t i0 = x.lists[k].size(), row = ((size_t)k * nunits + u) * list_len;
    for (int i = 0; i < want[k]; ++i, ++g) {
      x.lists[k].push_back(got[g]);
      lists_h[row + i0 + i] = got[g];  // an entry no queued copy reads yet (rewrites wait: write_rows)
    }
    CUDA_CHECK(cudaMemcpyAsync(lists_d + row + i0, lists_h + row + i0, want[k] * sizeof(uint32_t), cudaMemcpyHostToDevice, st));
  }
  return true;
}

uint8_t* Model::Impl::ent_addr(int k, int a, int j) const {
  const int h = line_h[cur_unit];
  if (h == kNoLine) die("model: entry %d of kv source %d in slot %d, which holds no session", j, k, cur_slot);
  const Nums& l = store_s->sess.at(h).lists[k];
  if ((size_t)(j >> attn::kChunkShift) >= l.size()) die("model: entry %d of kv source %d past slot %d's list", j, k, cur_slot);
  return pool_arr[a] + (((size_t)l[j >> attn::kChunkShift] << attn::kChunkShift) | (j & (attn::kChunkEntries - 1))) * kKvEntryBytes[a];
}

void Model::Impl::ensure_mapped(int s, int upto) {
  if (s < 0 || s >= nslots) die("model: ensure_mapped in slot %d of %d", s, nslots);
  upto = std::max(0, std::min(upto, slot_ctx_[s] - 1));
  if (upto <= mapped_upto[s]) return;  // the whole check of a step that needs nothing new
  int got = std::min(upto + kMapLookahead, slot_ctx_[s] - 1);
  if (!grow(s, got)) {  // no room for the lookahead: what the step needs (inside its reservation, there with eviction)
    got = upto;
    if (!grow(s, got))
      die("model: slot %d cannot hold positions up to %d: %u of %u chunk numbers free, %d held by its session", s, upto,
          calloc->free(), calloc->total(), held(s));
  }
  mapped_upto[s] = got;
}

void Model::Impl::slot_reserve(int s, int need) {
  if (s < 0 || s >= nslots) die("model: slot_reserve in slot %d of %d", s, nslots);
  resv[s] = slot_chunks_for(s, need);  // the server counted the pool first (serve.cpp mem_ok)
}

// The slot's unit holds session h from now on (its rows are h's lists): the session it held leaves it (keeping its
// numbers; a unit's own session is freed). h comes back from the disk tier first.
void Model::Impl::slot_attach(int s, int h) {
  if (s < 0 || s >= nslots || h <= 0) die("model: session %d attaches to slot %d", h, s);
  const int u = unit_of[s];
  if (line_h[u] == h) return;
  const int r = resv[s];  // the request's reservation (kStReserve comes first) stays with the slot
  slot_release(s);
  resv[s] = r;
  Store::Sess& x = use_blocks(*this, h);
  if (x.line >= 0) die("model: session %d attaches to slot %d while unit %d holds it", h, s, x.line);
  x.line = u;
  line_h[u] = h;
  write_rows(*this, u, &x);
  mapped_upto[s] = -1;  // the next step grows from the lists as they are
}

// Slot s's unit leaves its session (which keeps its numbers: no copy, no drain); a unit's own session is freed.
void Model::Impl::slot_release(int s) {
  if (s < 0 || s >= nslots) die("model: slot_release of slot %d of %d", s, nslots);
  const int u = unit_of[s];
  const int h = line_h[u];
  if (h != kNoLine) {
    Store& st_ = store_of();
    write_rows(*this, u, nullptr);
    if (h < 0) {
      Store::Sess& x = st_.sess.at(h);
      for (Nums& l : x.lists) give_nums(l);
      st_.sess.erase(h);
    } else if (auto it = st_.sess.find(h); it != st_.sess.end()) {
      it->second.line = -1;
    }
    line_h[u] = kNoLine;
  }
  mapped_upto[s] = -1;
  resv[s] = 0;
}

namespace {
// entries [0, n) of number `from` into number `to`, all four arrays (a fork's partial number; copy on write)
void copy_entries(Model::Impl& m, uint32_t to, uint32_t from, int n) {
  for (int a = 0; a < 4; ++a)
    CUDA_CHECK(cudaMemcpyAsync(m.pool_arr[a] + ((size_t)to << attn::kChunkShift) * kKvEntryBytes[a],
                               m.pool_arr[a] + ((size_t)from << attn::kChunkShift) * kKvEntryBytes[a], (size_t)n * kKvEntryBytes[a],
                               cudaMemcpyDeviceToDevice, m.st));
}
}  // namespace

bool Model::Impl::own_tail(int h, int pos) {
#ifdef DSGB10_POOL_NOCOW
  (void)h, (void)pos;
  return true;  // the negative control: shared numbers are written in place (fork_test must fail)
#else
  Store::Sess& x = store_of().sess.at(h);
  bool rows = false;
  for (int k = 0; k < kKvSlots; ++k) {
    const int e = std::max(0, pos) / (k < 3 ? 2 : 1), i0 = e >> attn::kChunkShift;
    Nums& l = x.lists[k];
    for (size_t i = (size_t)i0; i < l.size(); ++i) {
      if (refs[l[i]] < 2) continue;
      Nums fresh;
      store_of().pinned.push_back(h);
      const bool ok = take_nums(h, 1, &fresh);
      store_of().pinned.pop_back();
      if (!ok) return false;
      const int below = (int)i == i0 ? (e & (attn::kChunkEntries - 1)) : 0;  // the entries under pos stay valid
      if (below) copy_entries(*this, fresh[0], l[i], below);
      Nums old{l[i]};
      give_nums(old);
      l[i] = fresh[0];
      rows = true;
    }
  }
  if (rows && x.line >= 0) write_rows(*this, x.line, &x);
  return true;
#endif
}

// A bookmark is a restart point, not something a step needs: it takes only numbers that leave every line's reservation
// covered (a reservation past what its lists hold may count on what evicting idle sessions gives back, serve.cpp mem_ok).
// With none to spare the bookmark is not taken: a later restart starts at an earlier one.
bool Model::Impl::take_extra(int h, size_t n, Nums* out) {
  Store& s = store_of();
  const int64_t reserved = rules::unheld_sum(nslots, [&](int t) { return resv[t]; }, [&](int t) { return held(t); });
  const auto giveable = [&]() {  // what store_evict_step can still give back counts too
    std::vector<int> keep = s.pinned;
    keep.insert(keep.end(), s.live.begin(), s.live.end());
    keep.push_back(h);
    return state_giveable(keep);
  };
  if (!rules::covers((int64_t)calloc->free(), reserved, (int64_t)n, giveable)) return false;
  s.pinned.push_back(h);
  const bool ok = take_nums(h, n, out);
  s.pinned.pop_back();
  return ok;
}

bool Model::Impl::own_bk(int h, Nums& b) {
  bool shared = false;
  for (const uint32_t c : b) shared |= refs[c] > 1;
  if (!shared) return true;
  Nums fresh;  // the whole bookmark is written again: nothing to copy
  if (!take_extra(h, b.size(), &fresh)) return false;
  give_nums(b);
  b = std::move(fresh);
  return true;
}

// ---- the disk tier --------------------------------------------------------------------------------

namespace {

// a spilled session's blocks, released from the file
void disk_release(Store& s, Store::Sess& ss, bool spec) {
  for (const size_t o : ss.dpages) s.dpool.give_off(o, kPageBytes), s.dbytes -= Pool::round(kPageBytes);
  for (const auto& sn : ss.dsnaps) s.dpool.give_off(sn.second, snap_bytes(spec)), s.dbytes -= Pool::round(snap_bytes(spec));
  if (ss.dend != Pool::kNone) s.dpool.give_off(ss.dend, end_bytes(spec)), s.dbytes -= Pool::round(end_bytes(spec));
  ss.dpages.clear();
  ss.dsnaps.clear();
  ss.dend = Pool::kNone;
}

}  // namespace

// Room in the file for a block of session h: while there is none, the least recently used spilled
// session that the current command does not use is dropped (gone: state_evicted). kNone when none is left.
size_t Model::Impl::disk_take(int h, size_t bytes) {
  Store& s = store_of();
  for (;;) {
    const size_t off = s.dpool.take_off(bytes);
    if (off != Pool::kNone) {
      s.dbytes += Pool::round(bytes);
      return off;
    }
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (const auto& [k, ss] : s.sess)
      if (k != h && ss.disk && ss.used < oldest && !in(s.pinned, k)) victim = k, oldest = ss.used;
    if (victim < 0) return Pool::kNone;
    if (rank == 0) std::printf("state: session %d dropped from the disk tier to make room\n", victim);
    state_free(victim);
    s.evicted.push_back(victim);
  }
}

namespace {

// page i of session x (positions [i, i + 1) * kPagePos of every kv source) between its chunks and the gather buffer:
// a page's run of a source's entries lies in one chunk number (kPagePos / ratio divides kChunkEntries)
void page_copy(Model::Impl& m, const Store::Sess& x, int i, bool to_gather) {
  uint8_t* g = m.store_of().gather;
  for (int k = 0; k < kKvSlots; ++k) {
    const int ratio = k < 3 ? 2 : 1, E = kPagePos / ratio, j = i * E;
    if ((size_t)(j >> attn::kChunkShift) >= x.lists[k].size()) die("state: page %d of a session past its kv source %d's list", i, k);
    const size_t c = x.lists[k][j >> attn::kChunkShift], e = ((size_t)c << attn::kChunkShift) | (j & (attn::kChunkEntries - 1));
    for (int a = 0; a < 4; ++a) {
      uint8_t* p = m.pool_arr[a] + e * kArr[a];
      uint8_t* q = g + page_off(k, a);
      CUDA_CHECK(cudaMemcpyAsync(to_gather ? q : p, to_gather ? p : q, (size_t)E * kArr[a], cudaMemcpyDeviceToDevice, m.st));
    }
  }
}

}  // namespace

// Session h's state into the disk tier: its pages and bookmarks, each gathered, its checksum taken, then copied to
// staging after the work queued so far; its numbers are given back at once, and the writer thread writes the file.
// Without a disk tier, or without room in it, h is freed (gone: state_evicted).
void Model::Impl::state_spill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || it->second.disk) return;
  Store::Sess& ss = it->second;
  if (ss.line >= 0) die("state: session %d spilled while unit %d holds it", h, ss.line);
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  const size_t n0 = ss.numbers();
  if (s.disk) {
    // what goes out is what the session can restart from: bookmarks no restart can use, and pages past its latest
    // restart point, are not written
    if (!ss.end.empty() && !end_usable(ss.end_pos, ss.saved)) give_nums(ss.end), ss.end_pos = -1;
    for (auto i = ss.snaps.begin(); i != ss.snaps.end();)
      if (snap_usable(i->first, ss.saved)) ++i;
      else give_nums(i->second), i = ss.snaps.erase(i);
    const int keep = std::max(0, restart_of(ss.saved, ss.end.empty() ? -1 : ss.end_pos, ss.snaps));
    ss.saved = std::min(ss.saved, pages_for(keep) * kPagePos);
  }
  const int np = pages_for(ss.saved);
  std::vector<size_t> dp;
  std::map<int, size_t> ds;
  size_t de = Pool::kNone;
  bool room = s.disk != nullptr;
  for (int i = 0; room && i < np; ++i) room = (dp.emplace_back(disk_take(h, kPageBytes)), dp.back() != Pool::kNone);
  for (auto i = ss.snaps.begin(); room && i != ss.snaps.end(); ++i)
    room = (ds[i->first] = disk_take(h, snap_bytes(sp))) != Pool::kNone;
  if (room && !ss.end.empty()) room = (de = disk_take(h, end_bytes(sp))) != Pool::kNone;
  if (!room) {
    if (s.disk) {
      if (!dp.empty() && dp.back() == Pool::kNone) dp.pop_back();
      for (auto i = ds.begin(); i != ds.end();) i = i->second == Pool::kNone ? ds.erase(i) : std::next(i);
      Store::Sess tmp;
      tmp.dpages = dp, tmp.dsnaps = ds, tmp.dend = de;
      disk_release(s, tmp, sp);
    }
    state_free(h);
    s.evicted.push_back(h);
    return;
  }
  s.disk->begin(st);  // the checksums of what goes out, then the copies (each timed: spill.cpp)
  for (int i = 0; i < np; ++i) {
    page_copy(*this, ss, i, true);
    sum_block(s.gather, kPageBytes, s.disk->sum(dp[i], false), st);
    ss.dseq = s.disk->write(s.gather, kPageBytes, dp[i], st);
  }
  for (const auto& sn : ss.snaps) {
    bk_copy(sn.second, 0, s.gather, snap_bytes(sp), false);
    sum_block(s.gather, snap_bytes(sp), s.disk->sum(ds.at(sn.first), false), st);
    ss.dseq = s.disk->write(s.gather, snap_bytes(sp), ds.at(sn.first), st);
  }
  if (!ss.end.empty()) {
    bk_copy(ss.end, 0, s.gather, end_bytes(sp), false);
    sum_block(s.gather, end_bytes(sp), s.disk->sum(de, false), st);
    ss.dseq = s.disk->write(s.gather, end_bytes(sp), de, st);
  }
  s.disk->flush(st);
  for (Nums& l : ss.lists) give_nums(l);
  for (auto& sn : ss.snaps) give_nums(sn.second);
  give_nums(ss.end);
  ss.snaps.clear();
  ss.disk = true;
  ss.dpages = std::move(dp);
  ss.dsnaps = std::move(ds);
  ss.dend = de;
  ++tier.spills, tier.spill_bytes += n0 * chunk_image();
  if (rank == 0)
    std::printf("state: session %d spilled to disk (%zu chunk numbers given back, restart point %d; %.2f of %.2f GiB of the file in "
                "use; host %.2f ms)\n", h, n0, restart_of(ss.saved, ss.dend != Pool::kNone ? ss.end_pos : -1, ss.dsnaps),
                s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0, 1e3 * (wall_s() - t0));
}

// Session h back from the disk tier (after its writes), as far as the pool has room, taking what gives the latest
// restart point first. The candidates are the end bookmark and the window bookmarks, latest first, each usable when
// the pages up to its restart position are in the record (end_usable, snap_usable); the first whose numbers all find
// room is restored with exactly those pages (saved shrinks to them), then every earlier window bookmark those pages
// serve for as long as room lasts. When no candidate fits, nothing comes back: pages alone restart nothing. Every
// block that comes back is checked against its checksum from the way out; every rank logs a mismatch, and rank 0 the
// time it took and the restart point that came back against the one that was spilled.
void Model::Impl::state_unspill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || !it->second.disk) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  s.disk->wait(ss.dseq);
  const double t1 = wall_s();
  size_t got = 0;
  std::vector<size_t> back;  // the file offsets of the blocks that came back
  const auto read_block = [&](size_t bytes, size_t off) {  // into the gather buffer, checked
    s.disk->read(s.gather, bytes, off, st);
    sum_block(s.gather, bytes, s.disk->sum(off, true), st);
    back.push_back(off);
    got += Pool::round(bytes);
  };
  const int spilled_restart = restart_of(ss.saved, ss.dend != Pool::kNone ? ss.end_pos : -1, ss.dsnaps);
  struct Cand {
    int q;     // the restart position
    int snap;  // the window bookmark's position (-1: the end bookmark)
  };
  std::vector<Cand> cand;
  if (ss.dend != Pool::kNone && end_usable(ss.end_pos, ss.saved)) cand.push_back({ss.end_pos, -1});
  for (auto i = ss.dsnaps.rbegin(); i != ss.dsnaps.rend(); ++i)
    if (snap_usable(i->first, ss.saved)) cand.push_back({snap_q(i->first), i->first});
  std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.q > b.q; });
  s.pinned.push_back(h);  // its own numbers' allocations never evict it
  int pick = -1, np = 0;
  Nums lists[kKvSlots], blk;
  size_t failed = SIZE_MAX;  // the fewest pages a candidate needed that found no room: it and any needing more fail
  for (size_t c = 0; c < cand.size() && pick < 0; ++c) {
    np = std::min((int)ss.dpages.size(), pages_for(cand[c].q));
    if ((size_t)np >= failed) continue;
    size_t want = bk_nums(cand[c].snap < 0 ? end_bytes(sp) : snap_bytes(sp));
    for (int k = 0; k < kKvSlots; ++k) want += (size_t)nums_for_pos(k, np * kPagePos);
    Nums all;
    if (take_nums(h, want, &all)) {
      pick = (int)c;
      size_t g = 0;
      for (int k = 0; k < kKvSlots; ++k)
        for (int i = 0, n = nums_for_pos(k, np * kPagePos); i < n; ++i) lists[k].push_back(all[g++]);
      blk.assign(all.begin() + g, all.end());
    } else {
      failed = (size_t)np;
    }
  }
  if (pick >= 0) {
    for (int k = 0; k < kKvSlots; ++k) ss.lists[k] = std::move(lists[k]);
    for (int i = 0; i < np; ++i) read_block(kPageBytes, ss.dpages[i]), page_copy(*this, ss, i, false);
    ss.saved = std::min(ss.saved, np * kPagePos);
    if (cand[pick].snap < 0) {
      read_block(end_bytes(sp), ss.dend);
      bk_copy(blk, 0, s.gather, end_bytes(sp), true);
      ss.end = std::move(blk);
    } else {
      read_block(snap_bytes(sp), ss.dsnaps.at(cand[pick].snap));
      bk_copy(blk, 0, s.gather, snap_bytes(sp), true);
      ss.snaps.emplace(cand[pick].snap, std::move(blk));
    }
    for (auto i = ss.dsnaps.rbegin(); i != ss.dsnaps.rend(); ++i) {  // the earlier window bookmarks these pages serve
      if (i->first == cand[pick].snap || !snap_usable(i->first, ss.saved)) continue;
      Nums b;
      if (!take_nums(h, bk_nums(snap_bytes(sp)), &b)) break;
      read_block(snap_bytes(sp), i->second);
      bk_copy(b, 0, s.gather, snap_bytes(sp), true);
      ss.snaps.emplace(i->first, std::move(b));
    }
  } else {
    ss.saved = 0;
  }
  if (ss.end.empty()) ss.end_pos = -1;
  s.pinned.pop_back();
  CUDA_CHECK(cudaStreamSynchronize(st));
  const double t2 = wall_s();
  {
    const uint64_t w = (uint64_t)((t2 - t0) * 1e9);
    ++tier.restores, tier.restore_bytes += got, tier.restored_pos += (uint64_t)std::max(ss.saved, 0);
    tier.wait_ns += w, tier.wait_max_ns = std::max(tier.wait_max_ns, w);
  }
  size_t bad = 0;
  uint64_t fp = 0;  // the checksums' xor, logged: a check that compared nothing would show as 0
  for (const size_t o : back) bad += *s.disk->sum(o, false) != *s.disk->sum(o, true), fp ^= *s.disk->sum(o, true);
  if (bad)
    std::printf("state: session %d: ERROR: %zu of %zu blocks read back from the disk differ from what was spilled "
                "(rank %d)\n", h, bad, back.size(), rank);
  // a kept session is not continued from a state that is not the one it left: the next start is clean
  if (bad && ss.kept) die("state: session %d, kept by an earlier run, came back different from the disk (rank %d)", h, rank);
  ss.kept = false;
  disk_release(s, ss, sp);
  ss.disk = false;
  if (rank == 0)
    std::printf("state: session %d back from disk: %.1f MiB in %.1f ms (%.2f GB/s; %.1f ms waiting for its writes; "
                "%zu blocks checked, %016llx); restart point %d of the %d spilled\n", h, got / 1048576.0, 1e3 * (t2 - t0),
                got / std::max(1e-9, t2 - t1) / 1e9, 1e3 * (t1 - t0), back.size(), (unsigned long long)fp,
                restart_of(ss.saved, ss.end.empty() ? -1 : ss.end_pos, ss.snaps), spilled_restart);
}

// Compressed caches of positions [from, to) into session h: the line holding h writes into h's own numbers, so what it
// holds up to `to` is h's already.
void Model::Impl::state_save(int h, int from, int to) {
  Store::Sess& ss = use_blocks(*this, h);
  if (from > ss.saved) die("state: session %d saves [%d, %d) with %d saved", h, from, to, ss.saved);
  if (to > from && ss.line != cur_unit) die("state: session %d saves [%d, %d) from unit %d, which does not hold it", h, from, to, cur_unit);
  for (int k = 0; k < kKvSlots; ++k)  // the chunk of the last entry saved is the session's
    if (to > from && (int)ss.lists[k].size() <= ((to - 1) / (k < 3 ? 2 : 1) >> attn::kChunkShift))
      die("state: session %d saves up to %d past its kv source %d's numbers", h, to, k);
  ss.saved = std::max(ss.saved, to);
}

// Session h's compressed caches of positions [0, to) for the current slot's session. h itself is cut to `to` (the numbers
// from the one holding `to` on made its own: own_tail). A fork's base: the slot's session gives its lists back and takes
// the base's numbers wholly below `to`, shared by count, and a copy of the entries below `to` of the one holding it.
void Model::Impl::state_load(int h, int to) {
  if (!store_of().sess.count(h)) die("state: session %d loads with nothing stored", h);
  Store::Sess& src = use_blocks(*this, h);
  if (to > src.saved) die("state: session %d loads %d positions of %d", h, to, src.saved);
  const int hd = line_h[cur_unit];
  if (hd == h) {
    src.saved = std::min(src.saved, to);
    if (!own_tail(h, to)) die("state: no room to copy session %d's shared numbers from %d on", h, to);
    return;
  }
  if (hd == kNoLine || hd < 0) die("state: a fork of %d into slot %d, which holds no session", h, cur_slot);
  Store::Sess& dst = store_of().sess.at(hd);
  for (Nums& l : dst.lists) give_nums(l);
  dst.saved = 0;
  store_of().pinned.push_back(h);  // the partial copies' numbers never evict the base
  size_t shared = 0, copied = 0;
  for (int k = 0; k < kKvSlots; ++k) {
    const int e = to / (k < 3 ? 2 : 1), full = e >> attn::kChunkShift, part = e & (attn::kChunkEntries - 1);
    share_nums(src.lists[k], (size_t)full, &dst.lists[k]);
    shared += full;
    if (part) {
      Nums one;
      if (!take_nums(hd, 1, &one)) die("state: no room for a fork of %d at %d (one number of kv source %d)", h, to, k);
      copy_entries(*this, one[0], src.lists[k][full], part);
      dst.lists[k].push_back(one[0]);
      copied += part;
    }
  }
  store_of().pinned.pop_back();
  write_rows(*this, cur_unit, &dst);
  mapped_upto[cur_slot] = -1;  // the next step grows the lists past `to`
  if (rank == 0)
    std::printf("state: session %d forks %d at %d: %zu numbers shared, %zu entries copied\n", hd, h, to, shared, copied);
}

namespace {

// positions [p0, p1) between a ring and bookmark bytes from `off` on (row bytes per position), at most 256 of them
void ring_rows(Model::Impl& m, uint8_t* ring, const Nums& b, size_t off, size_t row, int p0, int p1, bool to_bk) {
  for (int q = p0; q < p1;) {
    const int slot = q & (attn::kRing - 1), n = std::min(p1 - q, attn::kRing - slot);
    m.bk_copy(b, off + (size_t)(q - p0) * row, ring + (size_t)slot * row, (size_t)n * row, to_bk);
    q += n;
  }
}

// the draft module's windows, positions [p0, p1) or whole (p0 < 0), to or from bookmark bytes from `off` on
void spec_windows(Model::Impl& m, const Spec& sp, const Nums& b, size_t off, int p0, int p1, bool to_bk) {
  const size_t n = p0 < 0 ? attn::kRing : (size_t)(p1 - p0);
  for (int k = 0; k < dspark::kBlocks; ++k) {
    const attn::Window w = sp.draft->window(k);
    const size_t kv = off + (size_t)k * n * kSpecSlot, sc = kv + n * kHead;
    if (p0 < 0) {
      m.bk_copy(b, kv, w.kv, n * kHead, to_bk);
      m.bk_copy(b, sc, w.sc, n * kHead / 32, to_bk);
    } else {
      ring_rows(m, w.kv, b, kv, kHead, p0, p1, to_bk);
      ring_rows(m, w.sc, b, sc, kHead / 32, p0, p1, to_bk);
    }
  }
}

}  // namespace

// A window bookmark's numbers (the caller fills them, bk_copy), or nullptr when the pool has no room for it.
const Nums* Model::Impl::state_snap_buf(int h, int pos) {
  Store::Sess& ss = use_blocks(*this, h);
  const auto it = ss.snaps.find(pos);
  if (it != ss.snaps.end()) return own_bk(h, it->second) ? &it->second : nullptr;  // shared: fresh numbers first
  Nums b;
  if (!take_extra(h, bk_nums(snap_bytes(spec != nullptr)), &b)) {
    if (rank == 0) std::printf("state: no room for a bookmark of session %d at %d\n", h, pos);
    return nullptr;
  }
  return &ss.snaps.emplace(pos, std::move(b)).first->second;
}

void Model::Impl::state_snap(int h, int pos) {
  if (pos < kSnapRows) die("state: snapshot at %d", pos);
  const Nums* b = state_snap_buf(h, pos);
  if (!b) return;
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(*this, ring_kv[L], *b, snap_kv_off(L), kHead, pos - kSnapRows, pos, true);
    ring_rows(*this, ring_sc[L], *b, snap_sc_off(L), kHead / 32, pos - kSnapRows, pos, true);
  }
  if (spec) spec_windows(*this, *spec, *b, kSnapBytes, pos - kSnapRows, pos, true);
}

// The draft module's windows into the bookmark at pos, which the prefill took (none: not taken).
void Model::Impl::state_snap_spec(int h, int pos) {
  const Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || it->second.disk) return;
  const auto b = it->second.snaps.find(pos);
  if (b != it->second.snaps.end()) spec_windows(*this, *spec, b->second, kSnapBytes, pos - kSnapRows, pos, true);
}

void Model::Impl::state_restore(int h, int pos) {
  if (!store_of().sess.count(h)) die("state: session %d restores with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (!ss.snaps.count(pos)) die("state: session %d has no snapshot at %d", h, pos);
  const Nums& b = ss.snaps.at(pos);
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(*this, ring_kv[L], b, snap_kv_off(L), kHead, pos - kSnapRows, pos, false);
    ring_rows(*this, ring_sc[L], b, snap_sc_off(L), kHead / 32, pos - kSnapRows, pos, false);
  }
  if (spec) spec_windows(*this, *spec, b, kSnapBytes, pos - kSnapRows, pos, false);
}

void Model::Impl::state_snap_end(int h, int pos) {
  Store::Sess& ss = use_blocks(*this, h);
  if (ss.end.empty()) {
    if (!take_extra(h, bk_nums(end_bytes(spec != nullptr)), &ss.end)) {
      ss.end_pos = -1;
      return;
    }
  } else if (!own_bk(h, ss.end)) {  // shared and no room for its own: no end bookmark
    give_nums(ss.end);
    ss.end_pos = -1;
    return;
  }
  ss.end_pos = pos;
  size_t off = 0;
  for (int L = 0; L < kLayers; ++L, off += kRingBytes) {
    bk_copy(ss.end, off, ring_kv[L], (size_t)attn::kRing * kHead, true);
    bk_copy(ss.end, off + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32, true);
  }
  for (int k = 0; k < kKvSlots; ++k, off += kCarryBytes) {
    bk_copy(ss.end, off, reinterpret_cast<uint8_t*>(kvs[k].carry_kv), kCarryBytes / 2, true);
    bk_copy(ss.end, off + kCarryBytes / 2, reinterpret_cast<uint8_t*>(kvs[k].carry_sc), kCarryBytes / 2, true);
  }
  if (spec) spec_windows(*this, *spec, ss.end, kEndBytes, -1, 0, true);
}

void Model::Impl::state_restore_end(int h) {
  if (!store_of().sess.count(h)) die("state: session %d restores its end with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (ss.end.empty()) die("state: session %d has no end snapshot", h);
  size_t off = 0;
  for (int L = 0; L < kLayers; ++L, off += kRingBytes) {
    bk_copy(ss.end, off, ring_kv[L], (size_t)attn::kRing * kHead, false);
    bk_copy(ss.end, off + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32, false);
  }
  for (int k = 0; k < kKvSlots; ++k, off += kCarryBytes) {
    bk_copy(ss.end, off, reinterpret_cast<uint8_t*>(kvs[k].carry_kv), kCarryBytes / 2, false);
    bk_copy(ss.end, off + kCarryBytes / 2, reinterpret_cast<uint8_t*>(kvs[k].carry_sc), kCarryBytes / 2, false);
  }
  if (spec) spec_windows(*this, *spec, ss.end, kEndBytes, -1, 0, false);
}

// What session h holds from position `from` on no longer belongs to its history: bookmarks above it, the end bookmark
// if it is above, and saved positions from it on (the numbers past them go back unless a line holds the session: the
// line writes over them as it goes on).
void Model::Impl::state_drop(int h, int from) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  const bool sp = spec != nullptr;
  if (ss.disk) {  // the file's blocks
    for (auto i = ss.dsnaps.upper_bound(from); i != ss.dsnaps.end();) {
      s.dpool.give_off(i->second, snap_bytes(sp)), s.dbytes -= Pool::round(snap_bytes(sp));
      i = ss.dsnaps.erase(i);
    }
    if (ss.end_pos > from) {
      if (ss.dend != Pool::kNone) s.dpool.give_off(ss.dend, end_bytes(sp)), s.dbytes -= Pool::round(end_bytes(sp));
      ss.dend = Pool::kNone;
      ss.end_pos = -1;
    }
    ss.saved = std::min(ss.saved, from);
    while ((int)ss.dpages.size() > (ss.saved + kPagePos - 1) / kPagePos) {
      s.dpool.give_off(ss.dpages.back(), kPageBytes), s.dbytes -= Pool::round(kPageBytes);
      ss.dpages.pop_back();
    }
    return;
  }
  for (auto i = ss.snaps.upper_bound(from); i != ss.snaps.end();) {
    give_nums(i->second);
    i = ss.snaps.erase(i);
  }
  if (ss.end_pos > from) {
    give_nums(ss.end);
    ss.end_pos = -1;
  }
  ss.saved = std::min(ss.saved, from);
  if (ss.line >= 0 && !own_tail(h, from)) die("state: no room to copy session %d's shared numbers from %d on", h, from);
  if (ss.line < 0)
    for (int k = 0; k < kKvSlots; ++k)
      while ((int)ss.lists[k].size() > nums_for_pos(k, ss.saved)) {
        Nums last{ss.lists[k].back()};
        ss.lists[k].pop_back();
        give_nums(last);
      }
}

void Model::Impl::state_free(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  if (ss.line >= 0) {  // a line still holds it (a release frees it first): the line leaves it
    const int u = ss.line;
    write_rows(*this, u, nullptr);
    line_h[u] = kNoLine;
    for (int t = 0; t < nslots; ++t)
      if (unit_of[t] == u) mapped_upto[t] = -1, resv[t] = 0;
  }
  for (Nums& l : ss.lists) give_nums(l);
  for (auto& sn : ss.snaps) give_nums(sn.second);
  give_nums(ss.end);
  disk_release(s, ss, spec != nullptr);  // a spilled session's writes may still be queued: harmless
  s.sess.erase(it);
}

// src's bookmarks up to upto are dst's too: the same numbers, shared by count (a fork's; immutable until one is written again).
void Model::Impl::state_copy_snaps(int src, int dst, int upto) {
  Store& s = store_of();
  const auto it = s.sess.find(src);
  if (it == s.sess.end() || src == dst) return;
  if (it->second.disk) {
    s.pinned.push_back(dst);
    state_unspill(src);
    s.pinned.pop_back();
  }
  it->second.used = ++s.clock;
  s.pinned.push_back(src);
  Store::Sess& d = use_blocks(*this, dst);
  s.pinned.pop_back();
  for (auto i = it->second.snaps.begin(); i != it->second.snaps.upper_bound(upto); ++i) {  // shared by count
    const auto j = d.snaps.find(i->first);
    if (j != d.snaps.end()) give_nums(j->second), d.snaps.erase(j);
    Nums b;
    share_nums(i->second, i->second.size(), &b);
    d.snaps.emplace(i->first, std::move(b));
  }
}

void Model::Impl::state_drop_snap(int h, int pos) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  it->second.used = ++s.clock;
  if (it->second.disk) {
    const auto i = it->second.dsnaps.find(pos);
    if (i == it->second.dsnaps.end()) return;
    s.dpool.give_off(i->second, snap_bytes(spec != nullptr)), s.dbytes -= Pool::round(snap_bytes(spec != nullptr));
    it->second.dsnaps.erase(i);
    return;
  }
  const auto i = it->second.snaps.find(pos);
  if (i == it->second.snaps.end()) return;
  give_nums(i->second);
  it->second.snaps.erase(i);
}

void Model::state_reserve(size_t bytes) { d_->state_reserve(bytes); }
size_t Model::state_capacity() const { return d_->store_numbers * d_->chunk_image(); }  // the store's share of the pool
size_t Model::state_ceiling() const { return d_->calloc ? d_->calloc->total() * d_->chunk_image() : 0; }  // all of it
size_t Model::state_giveable(const std::vector<int>& keep) const { return d_->state_giveable(keep); }
Model::StoreChunks Model::state_chunks() const {  // the chunk pool: numbers in all, held by the sessions no line holds
  StoreChunks c;
  if (!d_->calloc || !d_->store_s) return c;
  c.total = d_->calloc->total();
  for (const auto& kv : d_->store_s->sess)
    if (kv.first > 0 && !kv.second.disk && kv.second.line < 0) c.mapped += kv.second.numbers();
  c.live_bytes = c.mapped * d_->chunk_image();
  return c;
}
void Model::state_pin(const std::vector<int>& h, const std::vector<int>& live) {
  auto& s = d_->store_of();
  s.pinned = h;
  s.live = live;
}

// Numbers a copy on write at position `pos` takes: with h_own == h_src, own_tail's count (every list's numbers from the
// cut's chunk on that another list or bookmark also names, one fresh number each); with h_own another session (a fork of
// h_src at pos), one number per list for the partial chunk at pos. Reads the store's records only (the same on every rank).
int Model::state_cow_numbers(int h_src, int h_own, int pos) const {
  const Impl& m = *d_;
  if (!m.store_s) return 0;
  const auto it = m.store_s->sess.find(h_src);
  if (it == m.store_s->sess.end()) return 0;
  const Store::Sess& x = it->second;
  int n = 0;
  for (int k = 0; k < kKvSlots; ++k) {
    const int e = std::max(0, pos) / (k < 3 ? 2 : 1);
    if (h_own != h_src) {
      n += (e & (attn::kChunkEntries - 1)) != 0;
      continue;
    }
    for (size_t i = (size_t)(e >> attn::kChunkShift); i < x.lists[k].size(); ++i) {
      const uint32_t c = x.lists[k][i];
      n += c && c < m.refs.size() && m.refs[c] >= 2;
    }
  }
  return n;
}

// A hash of the state every rank must hold alike (sessions with their lists, bookmarks and images, the numbers' counts and
// images, the reservations, the evict-last list): compared across the ranks' logs under DSGB10_STORE_CHECK=1.
uint64_t Model::store_digest() const {
  const Impl& m = *d_;
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  if (m.store_s) {
    const Store& s = *m.store_s;
    for (const auto& [k, x] : s.sess) {
      mix((uint64_t)(uint32_t)k), mix((uint64_t)(uint32_t)x.line), mix((uint64_t)(uint32_t)x.saved), mix((uint64_t)(uint32_t)x.end_pos);
      for (int a = 0; a < kKvSlots; ++a) {
        mix(x.lists[a].size());
        for (const uint32_t c : x.lists[a]) mix(c);
      }
      for (const auto& [pos, b] : x.snaps) {
        mix((uint64_t)(uint32_t)pos), mix(b.size());
        for (const uint32_t c : b) mix(c);
      }
      for (const auto& [pos, o] : x.dsnaps) mix((uint64_t)(uint32_t)pos), mix(o);
      for (const uint32_t c : x.end) mix(c);
      mix(x.dend);
    }
  }
  for (const uint32_t r : m.refs) mix(r);
  for (int t = 0; t < m.nslots; ++t) mix((uint64_t)(uint32_t)m.resv[t]);
  return h;
}
std::vector<int> Model::state_evicted() {
  std::vector<int> v;
  if (d_->store_s) v.swap(d_->store_s->evicted);
  return v;
}
void Model::state_save(int h, int from, int to) { d_->state_save(h, from, to); }
void Model::state_load(int h, int to) { d_->state_load(h, to); }
void Model::state_snap(int h, int pos) { d_->state_snap(h, pos); }
void Model::state_snap_end(int h, int pos) { d_->state_snap_end(h, pos); }
void Model::state_restore(int h, int pos) { d_->state_restore(h, pos); }
void Model::state_restore_end(int h) { d_->state_restore_end(h); }
void Model::state_drop(int h, int from) { d_->state_drop(h, from); }
void Model::state_free(int h) { d_->state_free(h); }
void Model::state_copy_snaps(int src, int dst, int upto) { d_->state_copy_snaps(src, dst, upto); }
void Model::state_drop_snap(int h, int pos) { d_->state_drop_snap(h, pos); }
uint64_t Model::state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                 std::string* why) {
  return d_->state_kept_check(base, run_id, max_age_s, bytes, why);
}
void Model::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  d_->state_disk(base, bytes, staging, reuse, run_id);
}
void Model::state_keep(uint64_t gen) { d_->state_keep(gen); }
uint64_t Model::state_kept() const { return d_->store_s ? d_->store_s->loaded_gen : 0; }
std::vector<int> Model::state_sessions() const {
  std::vector<int> v;
  if (d_->store_s)
    for (const auto& kv : d_->store_s->sess)
      if (kv.first > 0) v.push_back(kv.first);  // not a unit's own (tools' steps)
  return v;
}
size_t Model::state_disk_capacity() const { return d_->store_s && d_->store_s->disk ? d_->store_s->dpool.size : 0; }
size_t Model::state_disk_bytes() const { return d_->store_s ? d_->store_s->dbytes : 0; }
std::string Model::state_disk_path() const { return d_->store_s && d_->store_s->disk ? d_->store_s->disk->path() : ""; }
void Model::state_spill(int h) { d_->state_spill(h); }
void Model::state_unspill(int h) { d_->state_unspill(h); }
Model::StateInfo Model::state_info(int h) const {
  StateInfo r;
  if (!d_->store_s) return r;
  const auto it = d_->store_s->sess.find(h);
  if (it == d_->store_s->sess.end()) return r;
  const Store::Sess& ss = it->second;
  const bool spec = d_->spec != nullptr;
  r.saved = ss.saved;
  r.on_disk = ss.disk;
  r.restart = ss.disk ? restart_of(ss.saved, ss.dend != Pool::kNone ? ss.end_pos : -1, ss.dsnaps)
                      : restart_of(ss.saved, ss.end.empty() ? -1 : ss.end_pos, ss.snaps);
  if (ss.disk) {
    r.end_pos = ss.dend != Pool::kNone ? ss.end_pos : -1;
    for (const auto& sn : ss.dsnaps) r.snaps.push_back(sn.first);
    r.disk_bytes = ss.dpages.size() * Pool::round(kPageBytes) + ss.dsnaps.size() * Pool::round(snap_bytes(spec)) +
                   (ss.dend != Pool::kNone ? Pool::round(end_bytes(spec)) : 0);
    return r;
  }
  r.end_pos = ss.end.empty() ? -1 : ss.end_pos;
  for (const auto& sn : ss.snaps) r.snaps.push_back(sn.first);
  r.bytes = ss.numbers() * d_->chunk_image();
  return r;
}
size_t Model::state_bytes() const {  // device memory the store's sessions hold (no line holds them)
  return state_chunks().live_bytes;
}
Model::TierStats Model::tier_stats() const {
  TierStats t = d_->tier;
  if (const detail::Store* s = d_->store_s.get()) {
    t.store_cap = state_ceiling(), t.store_used = state_bytes();  // one pool: the lines and the sessions kept
    for (const auto& kv : s->sess)
      if (kv.first > 0) ++(kv.second.disk ? t.on_disk : t.resident);
    if (s->disk) t.disk_cap = s->dpool.size, t.disk_used = s->dbytes;
  }
  return t;
}
Model::PoolStats Model::pool_stats() const {
  PoolStats p;
  const Impl& m = *d_;
  if (!m.calloc) return p;
  p.numbers = m.calloc->total(), p.free = m.calloc->free(), p.held = m.calloc->held(), p.number_bytes = m.chunk_image();
  for (const uint32_t r : m.refs) p.shared += r > 1;
  for (int s = 0; s < m.nslots; ++s) p.reserved += (uint64_t)std::max(0, m.resv[s] - m.held(s));
  if (const detail::Store* st = m.store_s.get())
    for (const auto& [k, ss] : st->sess) {
      if (k <= 0) continue;  // a line's own session (tools, tests)
      ++(ss.line >= 0 ? p.on_lines : ss.disk ? p.on_disk : p.in_memory);
      bool sh = false;
      for (const detail::Nums& l : ss.lists)
        for (const uint32_t c : l) sh = sh || m.refs[c] > 1;
      p.sharing += sh;
    }
  return p;
}
void Model::drain() { d_->drain(); }

}  // namespace model
