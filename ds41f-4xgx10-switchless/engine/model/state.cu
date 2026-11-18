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

// A window snapshot's block and an end snapshot's: the draft module's windows follow the layers' rows
size_t snap_bytes(bool spec) { return kSnapBytes + (spec ? kSpecSnapBytes : 0); }
size_t end_bytes(bool spec) { return kEndBytes + (spec ? kSpecEndBytes : 0); }

// What a restart needs (serve.cpp choose): the end snapshot at e when the caches hold [0, e) (saved >= e),
// a window snapshot at s when they hold its restart position q (s, or s - 1 for an odd s; saved >= q).
int snap_q(int s) { return s & 1 ? s - 1 : s; }
// the first position kv source k's entry i holds
int ent_pos(int k, int i) { return i * attn::kChunkEntries * (k < 3 ? 2 : 1); }
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

constexpr char kKeptMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'K', '3'};  // the last byte is the index layout's version; another version is "not a kept index"

struct KeptHeader {
  char magic[8];
  uint64_t gen, run_id, file_bytes;
  int64_t written;  // unix time, seconds
  int32_t rank, n_sess;
  uint64_t clock;                              // the store's
  uint64_t image_bytes, snap_bytes, end_bytes;  // its layout
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
  if (hd.image_bytes != chunk_image() || hd.snap_bytes != snap_bytes(sp) || hd.end_bytes != end_bytes(sp))
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
  const int max_ents = (opt.max_ctx + 64) / attn::kChunkEntries + 2;
  for (int i = 0; i < hd.n_sess && r.ok; ++i) {
    Store::Kept k;
    k.h = r.get<int32_t>(), k.saved = r.get<int32_t>(), k.end_pos = r.get<int32_t>();
    const int ns = r.get<int32_t>(), he = r.get<int32_t>();
    k.used = r.get<uint64_t>();
    k.factor = r.get<int32_t>();
    if (!r.ok || (k.factor != 16 && k.factor != 32 && k.factor != 64) || k.h <= 0 || !hs.insert(k.h).second || ns < 0 || ns > 4096 || (he != 0 && he != 1) || k.saved < 0 ||
        k.saved > opt.max_ctx || k.end_pos < -1 || k.end_pos > opt.max_ctx)
      return bad("a bad session record");
    for (int kk = 0; kk < kKvSlots; ++kk) {
      const int ne = r.get<int32_t>();
      if (!r.ok || ne < 0 || ne > max_ents) return bad("a bad session record");
      for (int j = 0; j < ne; ++j) {
        const size_t off = r.get<uint64_t>();
        const uint64_t sum = r.get<uint64_t>();
        if (!block(off, chunk_image())) return bad("an image outside the file");
        k.ents[kk].emplace_back(off, sum);
      }
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
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());  // a shared image is named by every holder
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
  if (!reuse) unlink((base + ".index").c_str());  // a reused index stays the latest checkpoint (its images held)
  unlink((base + ".index.new").c_str());
  if (reuse) {  // the kept sessions, every piece only on the disk: back on demand, each checked (state_unspill)
    s.disk->keep();
    const bool sp = spec != nullptr;
    auto take = [&](size_t off, size_t n, uint64_t sum) {  // an image named again (a shared one) is taken once
      if (s.drefs[off]++) return;
      s.named.insert(off);
      if (!s.dpool.take_at(off, n)) die("state: a kept image at %zu is not free", off);
      s.dbytes += Pool::round(n);
      *s.disk->sum(off, false) = sum;
    };
    s.clock = s.kept_clock;
    for (const Store::Kept& k : s.kept) {
      Store::Sess& ss = s.sess[k.h];
      ss.kept = true;
      ss.saved = k.saved, ss.end_pos = k.end_pos, ss.used = k.used, ss.factor = k.factor;
      for (int kk = 0; kk < kKvSlots; ++kk)
        for (const auto& [off, sum] : k.ents[kk]) take(off, chunk_image(), sum), ss.lists[kk].push_back(0), ss.dl[kk].push_back(off);
      for (const auto& [pos, os] : k.snaps) take(os.first, snap_bytes(sp), os.second), ss.dsnaps.emplace(pos, os.first), ss.snaps[pos];
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

// An index body of gen (its KeptHeader first). A planned stop (all false): every session no line holds whose pieces all
// have an image and are only on the disk. A checkpoint (all true): every session as far as its pieces from position 0 on
// have images (its length cut to its first entry without one, the bookmarks inside that). *hs: the sessions named; *names: the images.
std::vector<uint8_t> Model::Impl::kept_index(uint64_t gen, bool all, std::vector<int>* hs, std::set<size_t>* names) {
  Store& s = store_of();
  const bool sp = spec != nullptr;
  std::vector<uint8_t> f(sizeof(KeptHeader));
  using keepfile::put;
  int n = 0;
  for (const auto& [h, ss] : s.sess) {
    if (h <= 0) continue;
    int saved = ss.saved;
    for (int k = 0; k < kKvSlots; ++k)
      for (size_t i = 0; i < ss.lists[k].size(); ++i)
        if (i >= ss.dl[k].size() || ss.dl[k][i] == Pool::kNone) {
          saved = std::min(saved, ent_pos(k, (int)i));
          break;
        }
    std::vector<std::pair<int, size_t>> bks;  // the window bookmarks it keeps
    for (const auto& sn : ss.snaps) {
      const auto d = ss.dsnaps.find(sn.first);
      if (d != ss.dsnaps.end() && snap_q(sn.first) <= saved) bks.emplace_back(sn.first, d->second);
    }
    const bool end = ss.end_pos >= 0 && ss.end_pos <= saved && ss.dend != Pool::kNone;
    if (all) {
      if (saved <= 0 || (bks.empty() && !end)) continue;
    } else {
      bool whole = ss.line < 0 && saved == ss.saved && bks.size() == ss.snaps.size() && (ss.end_pos < 0 || end) && ss.out();
      for (int k = 0; k < kKvSlots && whole; ++k)
        for (size_t i = 0; i < ss.lists[k].size() && whole; ++i) whole = i < ss.dl[k].size() && ss.dl[k][i] != Pool::kNone;
      if (!whole) continue;
    }
    ++n;
    hs->push_back(h);
    put<int32_t>(f, h), put<int32_t>(f, saved), put<int32_t>(f, end ? ss.end_pos : -1);
    put<int32_t>(f, (int32_t)bks.size()), put<int32_t>(f, end), put<uint64_t>(f, ss.used), put<int32_t>(f, ss.factor);
    for (int k = 0; k < kKvSlots; ++k) {
      int ne = 0;  // a checkpoint lists the entries below its saved length; a planned stop every one
      while (ne < (int)ss.lists[k].size() && (!all || ent_pos(k, ne) < saved)) ++ne;
      put<int32_t>(f, ne);
      for (int i = 0; i < ne; ++i) {
        put<uint64_t>(f, ss.dl[k][i]), put<uint64_t>(f, *s.disk->sum(ss.dl[k][i], false));
        names->insert(ss.dl[k][i]);
      }
    }
    for (const auto& [pos, o] : bks) {
      put<int32_t>(f, pos), put<int32_t>(f, 0), put<uint64_t>(f, o), put<uint64_t>(f, *s.disk->sum(o, false));
      names->insert(o);
    }
    if (end) {
      put<uint64_t>(f, ss.dend), put<uint64_t>(f, *s.disk->sum(ss.dend, false));
      names->insert(ss.dend);
    }
  }
  KeptHeader hd{};
  std::memcpy(hd.magic, kKeptMagic, sizeof kKeptMagic);
  hd.gen = gen, hd.run_id = s.run_id, hd.file_bytes = s.disk->size(), hd.written = (int64_t)std::time(nullptr);
  hd.rank = rank, hd.n_sess = n, hd.clock = s.clock;
  hd.image_bytes = chunk_image(), hd.snap_bytes = snap_bytes(sp), hd.end_bytes = end_bytes(sp);
  hd.body_bytes = f.size() - sizeof hd;
  hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
  std::memcpy(f.data(), &hd, sizeof hd);
  return f;
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
  std::vector<int> hs;
  std::set<size_t> names;
  const std::vector<uint8_t> f = kept_index(gen, false, &hs, &names);
  std::string why;
  if (!keepfile::write_atomic(s.base + ".index", f.data(), f.size(), &why)) {
    std::printf("state: rank %d keeps nothing: %s\n", rank, why.c_str());
    return;
  }
  s.disk->keep();
  if (rank == 0)
    std::printf("state: %zu sessions kept for the next start (%.2f GiB of the file): synced in %.1f ms, index of %.1f KiB "
                "in %.1f ms\n", hs.size(), s.dbytes / 1073741824.0, 1e3 * (t1 - t0), f.size() / 1024.0, 1e3 * (wall_s() - t1));
}

// A checkpoint while serving (rank 0's server, when nothing runs; every rank alike): the file synced, then the index of
// every session as far as its pieces have images, written whole or not at all; images the previous index named that were
// dropped since are freed once this one is durable. A rank whose index cannot be written keeps the previous one.
void Model::Impl::state_checkpoint(uint64_t gen) {
  Store& s = store_of();
  s.ckpt_sess.clear();
  if (!s.disk || !gen) return;
  const double t0 = wall_s();
  CUDA_CHECK(cudaStreamSynchronize(st));
  s.disk->sync();
  const double t1 = wall_s();
  std::vector<int> hs;
  std::set<size_t> names;
  const std::vector<uint8_t> f = kept_index(gen, true, &hs, &names);
  std::string why;
  if (!keepfile::write_atomic(s.base + ".index", f.data(), f.size(), &why)) {
    std::printf("state: rank %d: no checkpoint: %s\n", rank, why.c_str());
    return;
  }
  s.disk->keep();
  s.named = std::move(names);
  size_t freed = 0;
  for (const auto& [off, bytes] : s.retired) space_free(off, bytes), freed += Pool::round(bytes);
  s.retired.clear();
  s.ckpt_sess = std::move(hs);
  if (rank == 0)
    std::printf("state: checkpoint: %zu sessions (%zu images), the file synced in %.1f ms, index of %.1f KiB in %.1f ms; "
                "%.1f MiB of dropped images freed\n", s.ckpt_sess.size(), s.named.size(), 1e3 * (t1 - t0), f.size() / 1024.0,
                1e3 * (wall_s() - t1), freed / 1048576.0);
}

void Model::Impl::state_reserve(size_t bytes) {
  Store& s = store_of();
  if (s.gather) die("state: the store is reserved once");
  if (!calloc) die("state: the store needs the chunk pool");
  // a page or a bookmark at a time between the chunks and the disk tier (the draft module's windows counted in)
  s.gather = dalloc<uint8_t>(Pool::round(std::max({chunk_image(), end_bytes(true), snap_bytes(true)})));
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
  if (it != s.sess.end() && it->second.out()) m.state_unspill(h);
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
  if (store_of().ndisk.size() < calloc->numbers()) store_of().ndisk.resize(calloc->numbers(), Pool::kNone);
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
    if (store_s && c < store_s->ndisk.size() && store_s->ndisk[c] != Pool::kNone)  // its image outlives it, held by entries
      store_s->dnum.erase(store_s->ndisk[c]), store_s->ndisk[c] = Pool::kNone;
    if (pool_poison)  // a number given back holds 0xFD until taken again: a read through a stale list shows
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemsetAsync(pool_arr[a] + ((size_t)c << attn::kChunkShift) * kKvEntryBytes[a], 0xFD,
                                   (size_t)attn::kChunkEntries * kKvEntryBytes[a], st));
    calloc->give(c);
  }
  v.clear();
}

// ---- image rows (img_rows.h): a running request's rows are numbers no session names ---------------------------------------------

static_assert(imgrows::kNumberBytes == (size_t)attn::kChunkEntries * 256, "an image row's number is a bookmark's: the first array's slice");

size_t Model::Impl::img_numbers() const { return store_s ? store_s->img.numbers() : 0; }

const uint8_t* Model::Impl::img_row(int key, size_t row) const {
  const imgrows::Ledger::Rows* r = store_s ? store_s->img.find(key) : nullptr;
  if (!r || row >= r->rows) die("model: image row %zu of flight %d is not in the store", row, key);
  const imgrows::Where w = imgrows::where(row);
  return pool_arr[0] + ((size_t)r->nums[w.number] << attn::kChunkShift) * kKvEntryBytes[0] + w.offset;
}

// The numbers for flight `key`'s rows: lowest free first, evicting idle conversations while too few are free (the
// command's pinned sessions stay). The slot's reservation for them is spent either way.
bool Model::Impl::img_take(int key, size_t rows, int slot) {
  Store& s = store_of();
  const imgrows::Take t = s.img.take(key, rows, [&](size_t n, Nums* out) { return take_nums(0, n, out); });
  if (slot >= 0 && slot < nslots) resv_img[slot] = 0;
  return t == imgrows::Take::kTaken;
}

// Rows [row0, row0 + n) from host memory into the flight's numbers, one staging piece at a time: the piece waits for the
// copy that last read the buffer, and its own copy is queued on the model's stream behind everything that read the numbers.
void Model::Impl::img_fill(int key, size_t row0, size_t n, const uint16_t* src) {
  const imgrows::Ledger::Rows* r = store_s ? store_s->img.find(key) : nullptr;
  if (!r || row0 + n > r->rows || n > Model::kImgStageRows) die("model: image rows [%zu, %zu) of flight %d do not fit its %zu", row0, row0 + n, key, r ? r->rows : 0);
  if (!img_stage) {
    img_stage = halloc<uint8_t>(Model::kImgStageRows * imgrows::kRowBytes);
    CUDA_CHECK(cudaEventCreateWithFlags(&img_stage_ev, cudaEventDisableTiming));
  }
  CUDA_CHECK(cudaEventSynchronize(img_stage_ev));
  std::memcpy(img_stage, src, n * imgrows::kRowBytes);
  size_t at = 0;
  imgrows::runs(row0, row0 + n, [&](size_t r0, size_t cnt) {
    CUDA_CHECK(cudaMemcpyAsync(const_cast<uint8_t*>(img_row(key, r0)), img_stage + at, cnt * imgrows::kRowBytes, cudaMemcpyHostToDevice, st));
    at += cnt * imgrows::kRowBytes;
  });
  CUDA_CHECK(cudaEventRecord(img_stage_ev, st));
}

// key's numbers back to the pool (key 0: every flight's). Reads of them are queued on the model's stream ahead of any
// later write, so the numbers can be taken again at once.
void Model::Impl::img_free(int key) {
  if (!store_s) return;
  Store& s = *store_s;
  const auto give = [&](Nums& v) { give_nums(v); };
  if (key == 0) s.img.release_all(give);
  else s.img.release(key, give);
  for (ImgSet& d : imgs)  // a view parked on a unit that read the rows
    if (d.key == key || key == 0) d.span.clear(), d.row0.clear(), d.key = 0;
  if (img_key == key || key == 0) img_span.clear(), img_row0.clear(), img_key = 0;
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

// One step of making room for session h: the session that the current command does not use and no line holds gives up
// its last piece in the pool (evict_piece; every rank does the same): to the disk tier, or dropped without one. The
// command's evict-last sessions (Store::last) go after every other, then the least recently used (rules::VictimPick).
// False when none is left.
bool Model::Impl::store_evict_step(int h) {
  Store& s = store_of();
  rules::VictimPick pick;
  for (const auto& [k, ss] : s.sess)
    if (k != h && k > 0 && ss.line < 0 && ss.numbers() && !in(s.pinned, k) && !in(s.live, k)) pick.offer(k, ss.used, in(s.last, k));
  const int victim = pick.victim;
  if (victim < 0) return false;
  if (!s.disk) {  // nowhere to put a piece: the session goes whole
    state_free(victim);
    s.evicted.push_back(victim);
    return true;
  }
  return evict_piece(victim, h);
}

// The chunk numbers of sessions no line holds and `keep` does not name: what eviction can give back. The same on
// every rank (it reads only the store's records).
size_t Model::Impl::state_giveable(const std::vector<int>& keep) const {
  if (!store_s) return 0;
  std::unordered_map<uint32_t, uint32_t> seen;  // a shared number goes back only when every list naming it does
  const auto add = [&](const Nums& v) {
    for (const uint32_t c : v)
      if (c) ++seen[c];
  };
  for (const auto& [k, ss] : store_s->sess)
    if (k > 0 && ss.line < 0 && !in(keep, k)) {
      for (const Nums& l : ss.lists) add(l);
      for (const auto& sn : ss.snaps) add(sn.second);
      add(ss.end);
    }
  size_t n = 0;
  for (const auto& [c, m] : seen) n += m == refs[c];
  return n;
}

size_t Model::state_restore_chunks(int h) const {  // the numbers reading h's pieces on the disk tier back takes
  if (!d_->store_s) return 0;
  const auto it = d_->store_s->sess.find(h);
  if (it == d_->store_s->sess.end()) return 0;
  const Store::Sess& ss = it->second;
  const bool sp = d_->spec != nullptr;
  size_t n = 0;
  for (const Nums& l : ss.lists)
    for (const uint32_t c : l) n += !c;
  for (const auto& sn : ss.snaps) n += sn.second.empty() ? bk_nums(snap_bytes(sp)) : 0;
  return n + (ss.end.empty() && ss.dend != Pool::kNone ? bk_nums(end_bytes(sp)) : 0);
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

void Model::Impl::slot_reserve(int s, int need, int img) {
  if (s < 0 || s >= nslots) die("model: slot_reserve in slot %d of %d", s, nslots);
  resv[s] = slot_chunks_for(s, need);  // the server counted the pool first (serve.cpp mem_ok)
  resv_img[s] = std::max(0, img);      // and the rows' numbers the request takes next
}

// Slot s's reservation to nothing: its request has finished, and a finished request reserves no numbers for anyone (the
// slot's session keeps the numbers it holds; the next request on the line reserves again).
void Model::Impl::slot_unreserve(int s) {
  if (s < 0 || s >= nslots) die("model: slot_unreserve in slot %d of %d", s, nslots);
  resv[s] = rules::resv_after_finish(resv[s]);
  resv_img[s] = 0;
}

// The slot's unit holds session h from now on (its rows are h's lists): the session it held leaves it (keeping its
// numbers; a unit's own session is freed). h comes back from the disk tier first.
void Model::Impl::slot_attach(int s, int h) {
  if (s < 0 || s >= nslots || h <= 0) die("model: session %d attaches to slot %d", h, s);
  const int u = unit_of[s];
  if (line_h[u] == h) return;
  const int r = resv[s], ri = resv_img[s];  // the request's reservation (kStReserve comes first) stays with the slot
  slot_release(s);
  resv[s] = r, resv_img[s] = ri;
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
  resv[s] = 0, resv_img[s] = 0;
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
  Store& s = store_of();
  Store::Sess& x = s.sess.at(h);
  for (int k = 0; k < kKvSlots; ++k) {  // the copies from the cut on go stale: the line writes those entries again
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    const int i0 = (std::max(0, pos) / (k < 3 ? 2 : 1)) >> attn::kChunkShift;
    for (size_t i = (size_t)i0; i < x.lists[k].size(); ++i) {
      if (x.dl[k][i] == Pool::kNone) continue;
      const uint32_t c = x.lists[k][i];
      if (c && refs[c] == 1 && s.ndisk[c] != Pool::kNone)  // its content changes: its image is no longer its
        s.dnum.erase(s.ndisk[c]), s.ndisk[c] = Pool::kNone;
      img_drop(x.dl[k][i], chunk_image());
      x.dl[k][i] = Pool::kNone;
    }
  }
#ifdef DSGB10_POOL_NOCOW
  return true;  // the negative control: shared numbers are written in place (fork_test must fail)
#else
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
  const int64_t reserved = rules::unheld_sum(nslots, [&](int t) { return resv[t]; }, [&](int t) { return held(t); }, [&](int t) { return resv_img[t]; });
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

// Numbers for a piece session h reads back from the disk tier: taken only beyond what running requests have reserved, so
// a restore cuts at the first piece with no room instead of eating a reservation (the restore's cut_at).
bool Model::Impl::take_restore(int h, size_t n, Nums* out) {
  Store& s = store_of();
  const int64_t reserved = rules::unheld_sum(nslots, [&](int t) { return resv[t]; }, [&](int t) { return held(t); }, [&](int t) { return resv_img[t]; });
  const auto giveable = [&]() {
    std::vector<int> keep = s.pinned;
    keep.insert(keep.end(), s.live.begin(), s.live.end());
    keep.push_back(h);
    return state_giveable(keep);
  };
  if (!rules::restore_may_take((int64_t)calloc->free(), reserved, (int64_t)n, giveable)) return false;
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

// ---- the disk tier: one image per piece -------------------------------------------------------------------
//
// A conversation's list entry is in the pool (a number), on the disk tier (an image in the file), or both; so is each
// of its bookmarks. An image is written once while its content holds (a full entry at a turn's save: write_sealed; any
// piece when eviction takes its number: evict_piece) and stays after a read-back; its checksum is checked on read. The
// holders of a shared number share its image (ndisk); its file space goes back when the entries naming it (drefs) reach 0.
// A copy goes stale only where the conversation writes again: own_tail clears copies from a cut on, a bookmark taken again drops its copy.

namespace {

// the byte offset of array a's slice in a cache entry's image
size_t img_slice(int a) {
  size_t o = 0;
  for (int j = 0; j < a; ++j) o += (size_t)attn::kChunkEntries * kArr[j];
  return o;
}
// whether the session has a piece only on the disk tier (it is read back before use)
bool is_out(const Store::Sess& x) {
  for (const Nums& l : x.lists)
    for (const uint32_t c : l)
      if (!c) return true;
  for (const auto& sn : x.snaps)
    if (sn.second.empty()) return true;
  return x.end.empty() && x.dend != Pool::kNone;
}

}  // namespace

bool Store::Sess::out() const { return is_out(*this); }

// File space for an image of session h: while there is none, the least recently used session the current command does
// not use and no running request uses (a conversation on an idle or paused line is cache like a stored one), with an image
// the index on disk does not hold, gives up its images (dropped: state_evicted, if it needs them, a piece only on the
// disk; else it only loses its copies; a conversation on a line is never freed, its pieces in the pool are the truth). kNone when none is left (retired space returns at the next checkpoint).
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
    for (const auto& [k, ss] : s.sess) {
      if (ss.used >= oldest || !rules::disk_victim_ok({k, ss.line, ss.used}, h, s.pinned, s.live)) continue;
      bool gives = false;  // an image the index on disk does not hold: dropping the session's images gives space
      const auto g = [&](size_t o) { gives = gives || (o != Pool::kNone && !s.named.count(o)); };
      g(ss.dend);
      for (const auto& d : ss.dsnaps) g(d.second);
      for (const auto& d : ss.dl)
        for (const size_t o : d) g(o);
      if (gives) victim = k, oldest = ss.used;
    }
    if (victim < 0) return Pool::kNone;
    Store::Sess& v = s.sess.at(victim);
    if (rules::disk_victim_frees({victim, v.line, v.used}, v.out())) {  // a conversation on a line only loses its images
      if (rank == 0) std::printf("state: session %d dropped from the disk tier to make room\n", victim);
      state_free(victim);
      s.evicted.push_back(victim);
    } else {
      drop_images(victim);
    }
  }
}

// one reference less on image `off`; at none its file space goes back (and the number holding its content, if any,
// has no copy any more), unless the latest checkpoint's index names it: then at the next checkpoint (retired)
void Model::Impl::img_drop(size_t off, size_t bytes) {
  Store& s = store_of();
  const auto it = s.drefs.find(off);
  if (it == s.drefs.end() || !it->second) die("state: image %zu dropped with no reference", off);
  ++s.ver;
  if (--it->second) return;
  s.drefs.erase(it);
  if (const auto n = s.dnum.find(off); n != s.dnum.end()) s.ndisk[n->second] = Pool::kNone, s.dnum.erase(n);
  if (s.named.count(off)) s.retired.emplace_back(off, bytes);
  else space_free(off, bytes);
}

// file space back: the part inside the tier's present size (a smaller size leaves what lies past it unused)
void Model::Impl::space_free(size_t off, size_t bytes) {
  Store& s = store_of();
  s.dbytes -= Pool::round(bytes);
  if (off < s.dpool.size) s.dpool.give_off(off, std::min(Pool::round(bytes), s.dpool.size - off));
}

// every image of session x dropped (its pieces in the pool stay; one only on the disk is lost: callers free x then)
void Model::Impl::drop_images(int h) {
  Store::Sess& x = store_of().sess.at(h);
  const bool sp = spec != nullptr;
  for (auto& d : x.dl)
    for (size_t& o : d)
      if (o != Pool::kNone) img_drop(o, chunk_image()), o = Pool::kNone;
  for (const auto& sn : x.dsnaps) img_drop(sn.second, snap_bytes(sp));
  x.dsnaps.clear();
  if (x.dend != Pool::kNone) img_drop(x.dend, end_bytes(sp)), x.dend = Pool::kNone;
}

// session x's cache entries out of the pool and the disk tier (its lists left empty)
void Model::Impl::free_lists(int h) {
  Store::Sess& x = store_of().sess.at(h);
  for (int k = 0; k < kKvSlots; ++k) {
    Nums in_pool;
    for (const uint32_t c : x.lists[k])
      if (c) in_pool.push_back(c);
    give_nums(in_pool);
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (const size_t o : x.dl[k])
      if (o != Pool::kNone) img_drop(o, chunk_image());
    x.lists[k].clear(), x.dl[k].clear();
  }
}

// number c's image (written now unless its content has one already, a shared number's); kNone without room
size_t Model::Impl::img_write(int h, uint32_t c) {
  Store& s = store_of();
  if (!s.disk) return Pool::kNone;
  ++s.ver;
  if (s.ndisk[c] != Pool::kNone) {
    ++s.drefs[s.ndisk[c]];
    return s.ndisk[c];
  }
  const size_t bytes = chunk_image(), off = disk_take(h, bytes);
  if (off == Pool::kNone) return off;
  for (int a = 0; a < 4; ++a)
    CUDA_CHECK(cudaMemcpyAsync(s.gather + img_slice(a), pool_arr[a] + ((size_t)c << attn::kChunkShift) * kArr[a],
                               (size_t)attn::kChunkEntries * kArr[a], cudaMemcpyDeviceToDevice, st));
  sum_block(s.gather, bytes, s.disk->sum(off, false), st);
  s.wseq = s.disk->write(s.gather, bytes, off, st);
  s.ndisk[c] = off, s.dnum[off] = c, s.drefs[off] = 1;
  ++tier.spills, tier.spill_bytes += bytes;
  return off;
}

// a bookmark's image (its bytes through the gather buffer); kNone without room
size_t Model::Impl::bk_write(int h, const Nums& b, size_t bytes) {
  Store& s = store_of();
  if (!s.disk) return Pool::kNone;
  const size_t off = disk_take(h, bytes);
  if (off == Pool::kNone) return off;
  bk_copy(b, 0, s.gather, bytes, false);
  sum_block(s.gather, bytes, s.disk->sum(off, false), st);
  s.wseq = s.disk->write(s.gather, bytes, off, st);
  s.drefs[off] = 1;
  ++s.ver;
  ++tier.spills, tier.spill_bytes += bytes;
  return off;
}

// Session h's full entries (every entry below the saved length) and its window bookmarks below it that have no image
// get one now (a turn's save: write-behind, so that eviction later gives their numbers back without writing).
void Model::Impl::write_sealed(int h) {
  Store& s = store_of();
  if (!s.disk) return;
  Store::Sess& x = s.sess.at(h);
  const bool sp = spec != nullptr;
  bool any = false;
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    const int full = std::min((int)x.lists[k].size(), (x.saved / (k < 3 ? 2 : 1)) >> attn::kChunkShift);
    for (int i = 0; i < full; ++i)
      if (x.lists[k][i] && x.dl[k][i] == Pool::kNone) {
        x.dl[k][i] = img_write(h, x.lists[k][i]);
        any = any || x.dl[k][i] != Pool::kNone;
      }
  }
  for (const auto& [pos, b] : x.snaps)
    if (!b.empty() && snap_usable(pos, x.saved) && !x.dsnaps.count(pos)) {
      const size_t o = bk_write(h, b, snap_bytes(sp));
      if (o != Pool::kNone) x.dsnaps[pos] = o, any = true;
    }
  if (any) s.disk->flush(st);
}

// Session x is cut to positions [0, pos): every cache entry that starts at pos or later leaves the pool and the disk
// tier (so no entry a line's row reads is left out of the pool), and so do bookmarks no restart from it can use.
void Model::Impl::cut_at(int h, int pos) {
  Store::Sess& x = store_of().sess.at(h);
  const bool sp = spec != nullptr;
  x.saved = std::min(x.saved, pos);
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    while (!x.lists[k].empty() && ent_pos(k, (int)x.lists[k].size() - 1) >= pos) {
      if (x.lists[k].back()) {
        Nums last{x.lists[k].back()};
        give_nums(last);
      }
      if (x.dl[k].back() != Pool::kNone) img_drop(x.dl[k].back(), chunk_image());
      x.lists[k].pop_back(), x.dl[k].pop_back();
    }
  }
  for (auto i = x.snaps.begin(); i != x.snaps.end();) {
    if (snap_usable(i->first, x.saved)) {
      ++i;
      continue;
    }
    give_nums(i->second);
    if (const auto d = x.dsnaps.find(i->first); d != x.dsnaps.end()) img_drop(d->second, snap_bytes(sp)), x.dsnaps.erase(d);
    i = x.snaps.erase(i);
  }
  if (x.end_pos >= 0 && !end_usable(x.end_pos, x.saved)) {
    give_nums(x.end);
    if (x.dend != Pool::kNone) img_drop(x.dend, end_bytes(sp)), x.dend = Pool::kNone;
    x.end_pos = -1;
  }
}

// One piece of session v leaves the pool, its last first: the end bookmark, then window bookmarks from the latest, then
// cache entries from the latest position down. A piece with an image gives its number back at once; one without is
// written first; with no room on the disk the piece is dropped and v shrinks to below it. False when v has nothing in the pool.
bool Model::Impl::evict_piece(int v, int h) {
  Store& s = store_of();
  Store::Sess& x = s.sess.at(v);
  const bool sp = spec != nullptr;
  if (!x.end.empty()) {
    if (x.dend == Pool::kNone) x.dend = bk_write(h, x.end, end_bytes(sp));
    if (x.dend == Pool::kNone) x.end_pos = -1;
    give_nums(x.end);
    if (s.disk) s.disk->flush(st);
    return true;
  }
  for (auto i = x.snaps.rbegin(); i != x.snaps.rend(); ++i) {
    if (i->second.empty()) continue;
    auto d = x.dsnaps.find(i->first);
    if (d == x.dsnaps.end()) {
      const size_t o = bk_write(h, i->second, snap_bytes(sp));
      if (o != Pool::kNone) d = x.dsnaps.emplace(i->first, o).first;
    }
    if (d == x.dsnaps.end()) {  // no room: the bookmark is gone
      const int pos = i->first;
      give_nums(x.snaps.at(pos));
      x.snaps.erase(pos);
    } else {
      give_nums(i->second);
    }
    if (s.disk) s.disk->flush(st);
    return true;
  }
  int bk = -1, bi = -1, bpos = -1;  // the cache entry in the pool that starts latest
  for (int k = 0; k < kKvSlots; ++k)
    for (int i = (int)x.lists[k].size() - 1; i >= 0; --i)
      if (x.lists[k][i]) {
        if (ent_pos(k, i) > bpos) bk = k, bi = i, bpos = ent_pos(k, i);
        break;
      }
  if (bk < 0) return false;
  x.dl[bk].resize(x.lists[bk].size(), Pool::kNone);
  if (x.dl[bk][bi] == Pool::kNone) x.dl[bk][bi] = img_write(h, x.lists[bk][bi]);
  if (x.dl[bk][bi] == Pool::kNone) {
    cut_at(v, bpos);  // no room on the disk: v keeps what lies below this entry
  } else {
    Nums one{x.lists[bk][bi]};
    give_nums(one);
    x.lists[bk][bi] = 0;
  }
  if (s.disk) s.disk->flush(st);
  return true;
}

// Session h's pieces in the pool that have no image get one: its cache entries holding saved positions (the tail one too),
// its bookmarks (an idle conversation, rank 0's server after --ssd-flush-s: an eviction or a planned stop later writes
// nothing). A copy of the tail goes stale where the conversation goes on (own_tail clears it).
void Model::Impl::state_flush(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (!s.disk || it == s.sess.end()) return;
  Store::Sess& x = it->second;
  const bool sp = spec != nullptr;
  bool any = false;
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (int i = 0; i < (int)x.lists[k].size() && ent_pos(k, i) < x.saved; ++i)
      if (x.lists[k][i] && x.dl[k][i] == Pool::kNone) {
        x.dl[k][i] = img_write(h, x.lists[k][i]);
        any = any || x.dl[k][i] != Pool::kNone;
      }
  }
  for (const auto& [pos, b] : x.snaps)
    if (!b.empty() && !x.dsnaps.count(pos)) {
      const size_t o = bk_write(h, b, snap_bytes(sp));
      if (o != Pool::kNone) x.dsnaps[pos] = o, any = true;
    }
  if (!x.end.empty() && x.dend == Pool::kNone) {
    x.dend = bk_write(h, x.end, end_bytes(sp));
    any = any || x.dend != Pool::kNone;
  }
  if (any) s.disk->flush(st);
}

// The disk tier's size while serving (every rank alike): a limit inside the file made at the start, so it cannot fail on
// one rank and not another. Smaller: images at or past the new end are given up first (a session needing one, a piece
// only on the disk, is dropped: state_evicted; otherwise only the copy). Larger: up to the file's start size. False if unchanged.
bool Model::Impl::state_disk_resize(size_t bytes) {
  Store& s = store_of();
  if (!s.disk) return false;
  bytes = std::min(bytes / kPoolGrain * kPoolGrain, s.disk->size());
  const size_t cur = s.dpool.size;
  if (bytes == cur || !bytes) return false;
  const bool sp = spec != nullptr;
  if (bytes < cur) {
    const auto past = [&](size_t o, size_t n) { return o != Pool::kNone && o + Pool::round(n) > bytes; };
    std::vector<int> drop;
    for (auto& [h, x] : s.sess) {
      bool needs = false;
      for (int k = 0; k < kKvSlots; ++k) {
        x.dl[k].resize(x.lists[k].size(), Pool::kNone);
        for (size_t i = 0; i < x.dl[k].size(); ++i)
          if (past(x.dl[k][i], chunk_image())) {
            if (!x.lists[k][i]) needs = true;
            else img_drop(x.dl[k][i], chunk_image()), x.dl[k][i] = Pool::kNone;
          }
      }
      for (auto d = x.dsnaps.begin(); d != x.dsnaps.end();) {
        if (!past(d->second, snap_bytes(sp))) {
          ++d;
          continue;
        }
        if (x.snaps.count(d->first) && x.snaps.at(d->first).empty()) needs = true, ++d;
        else img_drop(d->second, snap_bytes(sp)), d = x.dsnaps.erase(d);
      }
      if (past(x.dend, end_bytes(sp))) {
        if (x.end.empty()) needs = true;
        else img_drop(x.dend, end_bytes(sp)), x.dend = Pool::kNone;
      }
      if (needs) drop.push_back(h);
    }
    for (const int h : drop) {
      if (rank == 0) std::printf("state: session %d dropped: the disk tier shrinks past its pieces\n", h);
      state_free(h);
      s.evicted.push_back(h);
    }
    // nothing is held at or past `bytes` now: its free ranges go
    for (auto it = s.dpool.free.begin(); it != s.dpool.free.end();) {
      if (it->first >= bytes) {
        it = s.dpool.free.erase(it);
      } else {
        if (it->first + it->second > bytes) it->second = bytes - it->first;
        ++it;
      }
    }
    s.dpool.size = bytes;
  } else {
    s.dpool.size = bytes;
    s.dpool.give_off(cur, bytes - cur);
    for (const auto& [off, n] : s.retired)  // a retired image past the old size still holds its space
      if (off >= cur && off < bytes && !s.dpool.take_at(off, std::min(Pool::round(n), bytes - off)))
        die("state: a retired image at %zu is not free", off);
  }
  if (rank == 0)
    std::printf("state: the disk tier is now %.1f GiB (was %.1f; %.2f GiB in images)\n", bytes / 1073741824.0, cur / 1073741824.0,
                s.dbytes / 1073741824.0);
  return true;
}

// Session h out of the pool: every piece to the disk tier (a planned stop, the server's spill), or freed without one.
void Model::Impl::state_spill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  if (it->second.line >= 0) die("state: session %d spilled while unit %d holds it", h, it->second.line);
  if (!s.disk) {
    state_free(h);
    s.evicted.push_back(h);
    return;
  }
  const size_t n0 = it->second.numbers();
  while (evict_piece(h, h)) {
  }
  if (rank == 0)
    std::printf("state: session %d spilled to disk (%zu chunk numbers given back, restart point %d; %.2f of %.2f GiB of the "
                "file in use)\n", h, n0, restart_of(it->second.saved, it->second.end_pos, it->second.snaps),
                s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0);
}

// Session h's pieces on the disk tier back into the pool (after the writes queued so far), each checked against its
// checksum: cache entries from the first position, then the end bookmark, then window bookmarks from the latest; the
// images stay. When the pool has no room for a piece, h shrinks to below it; a piece that comes back different shrinks
// h to below it too (every rank logs it; a session an earlier run kept makes the engine stop instead).
void Model::Impl::state_unspill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || !it->second.out()) return;
  Store::Sess& x = it->second;
  x.used = ++s.clock;
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  if (s.disk) {
    s.disk->flush(st);
    s.disk->wait(s.wseq);
  }
  const double t1 = wall_s();
  struct Back {
    int k, i, snap;  // a cache entry (k, i), or a bookmark (k -1: snap = its position, -1 the end)
    size_t off;
  };
  std::vector<Back> back;
  size_t got = 0;
  s.pinned.push_back(h);
  const int restart0 = restart_of(x.saved, x.end_pos, x.snaps);
  // cache entries in position order: the beginning is what every restart needs
  std::vector<std::pair<int, std::pair<int, int>>> ents;
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (int i = 0; i < (int)x.lists[k].size(); ++i)
      if (!x.lists[k][i]) ents.push_back({ent_pos(k, i), {k, i}});
  }
  std::sort(ents.begin(), ents.end());
  int cut = INT32_MAX;
  for (const auto& [pos, ki] : ents) {
    const auto [k, i] = ki;
    Nums one;
    if (x.dl[k][i] == Pool::kNone || !take_restore(h, 1, &one)) {
      cut = pos;
      break;
    }
    s.disk->read(s.gather, chunk_image(), x.dl[k][i], st);
    sum_block(s.gather, chunk_image(), s.disk->sum(x.dl[k][i], true), st);
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemcpyAsync(pool_arr[a] + ((size_t)one[0] << attn::kChunkShift) * kArr[a], s.gather + img_slice(a),
                                 (size_t)attn::kChunkEntries * kArr[a], cudaMemcpyDeviceToDevice, st));
    x.lists[k][i] = one[0];
    s.ndisk[one[0]] = x.dl[k][i], s.dnum[x.dl[k][i]] = one[0];
    back.push_back({k, i, 0, x.dl[k][i]});
    got += chunk_image();
  }
  if (cut != INT32_MAX) cut_at(h, cut);
  const auto read_bk = [&](Nums& b, size_t off, size_t bytes, int snap) {
    if (!take_restore(h, bk_nums(bytes), &b)) return false;
    s.disk->read(s.gather, bytes, off, st);
    sum_block(s.gather, bytes, s.disk->sum(off, true), st);
    bk_copy(b, 0, s.gather, bytes, true);
    back.push_back({-1, 0, snap, off});
    got += bytes;
    return true;
  };
  if (x.end.empty() && x.dend != Pool::kNone && x.end_pos >= 0 && !read_bk(x.end, x.dend, end_bytes(sp), -1)) {
    img_drop(x.dend, end_bytes(sp)), x.dend = Pool::kNone, x.end_pos = -1;
  }
  std::vector<int> outs;  // window bookmarks only on the disk, the latest first
  for (auto i = x.snaps.rbegin(); i != x.snaps.rend(); ++i)
    if (i->second.empty()) outs.push_back(i->first);
  for (const int pos : outs) {
    const auto d = x.dsnaps.find(pos);
    if (d != x.dsnaps.end() && read_bk(x.snaps.at(pos), d->second, snap_bytes(sp), pos)) continue;
    if (d != x.dsnaps.end()) img_drop(d->second, snap_bytes(sp)), x.dsnaps.erase(d);
    x.snaps.erase(pos);  // no room for it: the bookmark is gone
  }
  s.pinned.pop_back();
  CUDA_CHECK(cudaStreamSynchronize(st));
  const double t2 = wall_s();
  {
    const uint64_t w = (uint64_t)((t2 - t0) * 1e9);
    ++tier.restores, tier.restore_bytes += got, tier.restored_pos += (uint64_t)std::max(x.saved, 0);
    tier.wait_ns += w, tier.wait_max_ns = std::max(tier.wait_max_ns, w);
  }
  int bad_pos = INT32_MAX;
  size_t bad = 0;
  uint64_t fp = 0;  // the checksums' xor, logged: a check that compared nothing would show as 0
  for (const Back& b : back) {
    fp ^= *s.disk->sum(b.off, true);
    if (*s.disk->sum(b.off, false) == *s.disk->sum(b.off, true)) continue;
    ++bad;
    bad_pos = std::min(bad_pos, b.k >= 0 ? ent_pos(b.k, b.i) : b.snap >= 0 ? b.snap - kSnapRows : x.end_pos - 1);
  }
  if (bad)
    std::printf("state: session %d: ERROR: %zu of %zu pieces read back from the disk differ from what was written (rank %d)\n",
                h, bad, back.size(), rank);
  if (host_ring) {  // every rank cuts at the smallest first bad position (each reaches this in the same command): the
                    // session stays the same on every rank, and rank 0's server restarts below the cut
    int32_t all[4];
    const int32_t mine = bad_pos;
    host_ring->allgather_host(&mine, all, sizeof mine);
    for (const int32_t b : all) bad_pos = std::min<int>(bad_pos, b);
  }
  if (bad_pos != INT32_MAX) cut_at(h, std::max(0, bad_pos));
  x.kept = false;
  if (rank == 0)
    std::printf("state: session %d back from disk: %.1f MiB in %.1f ms (%.2f GB/s; %.1f ms waiting for writes; %zu pieces "
                "checked, %016llx); restart point %d of the %d it had\n", h, got / 1048576.0, 1e3 * (t2 - t0),
                got / std::max(1e-9, t2 - t1) / 1e9, 1e3 * (t1 - t0), back.size(), (unsigned long long)fp,
                restart_of(x.saved, x.end_pos, x.snaps), restart0);
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
  if (to > from)  // the entries written in [from, to) hold new content: a copy of one (an idle flush's of a partial tail,
                  // a turn on its line went on after) is stale
    for (int k = 0; k < kKvSlots; ++k) {
      const int r = k < 3 ? 2 : 1, i1 = std::min((int)ss.lists[k].size() - 1, ((to - 1) / r) >> attn::kChunkShift);
      ss.dl[k].resize(ss.lists[k].size(), Pool::kNone);
      for (int i = (from / r) >> attn::kChunkShift; i <= i1; ++i) {
        if (ss.dl[k][i] == Pool::kNone) continue;
        const uint32_t c = ss.lists[k][i];
        Store& s = store_of();
        if (c && refs[c] == 1 && s.ndisk[c] != Pool::kNone) s.dnum.erase(s.ndisk[c]), s.ndisk[c] = Pool::kNone;
        img_drop(ss.dl[k][i], chunk_image());
        ss.dl[k][i] = Pool::kNone;
      }
    }
  ss.saved = std::max(ss.saved, to);
  write_sealed(h);  // write-behind: the entries this save filled, and the bookmarks below it, get their images
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
  free_lists(hd);
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
  if (it != ss.snaps.end()) {  // written again: its image goes stale, and a shared one takes fresh numbers first
    if (const auto d = ss.dsnaps.find(pos); d != ss.dsnaps.end()) img_drop(d->second, snap_bytes(spec != nullptr)), ss.dsnaps.erase(d);
    return own_bk(h, it->second) ? &it->second : nullptr;
  }
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
  if (it == s.sess.end()) return;
  const auto b = it->second.snaps.find(pos);
  if (b != it->second.snaps.end() && !b->second.empty()) spec_windows(*this, *spec, b->second, kSnapBytes, pos - kSnapRows, pos, true);
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
  if (ss.dend != Pool::kNone) img_drop(ss.dend, end_bytes(spec != nullptr)), ss.dend = Pool::kNone;  // written again
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
  for (auto i = ss.snaps.upper_bound(from); i != ss.snaps.end();) {
    give_nums(i->second);
    if (const auto d = ss.dsnaps.find(i->first); d != ss.dsnaps.end()) img_drop(d->second, snap_bytes(sp)), ss.dsnaps.erase(d);
    i = ss.snaps.erase(i);
  }
  if (ss.end_pos > from) {
    give_nums(ss.end);
    if (ss.dend != Pool::kNone) img_drop(ss.dend, end_bytes(sp)), ss.dend = Pool::kNone;
    ss.end_pos = -1;
  }
  ss.saved = std::min(ss.saved, from);
  if (ss.line >= 0 && !own_tail(h, from)) die("state: no room to copy session %d's shared numbers from %d on", h, from);
  if (ss.line < 0)
    for (int k = 0; k < kKvSlots; ++k) {
      ss.dl[k].resize(ss.lists[k].size(), Pool::kNone);
      while ((int)ss.lists[k].size() > nums_for_pos(k, ss.saved)) {
        if (ss.lists[k].back()) {
          Nums last{ss.lists[k].back()};
          give_nums(last);
        }
        if (ss.dl[k].back() != Pool::kNone) img_drop(ss.dl[k].back(), chunk_image());
        ss.lists[k].pop_back(), ss.dl[k].pop_back();
      }
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
      if (unit_of[t] == u) mapped_upto[t] = -1, resv[t] = 0, resv_img[t] = 0;
  }
  free_lists(h);
  for (auto& sn : ss.snaps) give_nums(sn.second);
  give_nums(ss.end);
  drop_images(h);  // its writes may still be queued: harmless, the file space is taken again only after them
  s.sess.erase(it);
}

// src's bookmarks up to upto are dst's too: the same numbers, shared by count (a fork's; immutable until one is written again).
void Model::Impl::state_copy_snaps(int src, int dst, int upto) {
  Store& s = store_of();
  const auto it = s.sess.find(src);
  if (it == s.sess.end() || src == dst) return;
  if (it->second.out()) {
    s.pinned.push_back(dst);
    state_unspill(src);
    s.pinned.pop_back();
  }
  it->second.used = ++s.clock;
  s.pinned.push_back(src);
  Store::Sess& d = use_blocks(*this, dst);
  s.pinned.pop_back();
  const size_t sb = snap_bytes(spec != nullptr);
  for (auto i = it->second.snaps.begin(); i != it->second.snaps.upper_bound(upto); ++i) {  // shared by count
    const auto j = d.snaps.find(i->first);
    if (j != d.snaps.end()) give_nums(j->second), d.snaps.erase(j);
    if (const auto dd = d.dsnaps.find(i->first); dd != d.dsnaps.end()) img_drop(dd->second, sb), d.dsnaps.erase(dd);
    if (i->second.empty()) continue;  // not back from the disk (no room): not shared
    Nums b;
    share_nums(i->second, i->second.size(), &b);
    d.snaps.emplace(i->first, std::move(b));
    if (const auto ds = it->second.dsnaps.find(i->first); ds != it->second.dsnaps.end())  // and its image
      ++s.drefs[ds->second], d.dsnaps[i->first] = ds->second;
  }
}

void Model::Impl::state_drop_snap(int h, int pos) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  if (const auto d = ss.dsnaps.find(pos); d != ss.dsnaps.end()) img_drop(d->second, snap_bytes(spec != nullptr)), ss.dsnaps.erase(d);
  const auto i = ss.snaps.find(pos);
  if (i == ss.snaps.end()) return;
  give_nums(i->second);
  ss.snaps.erase(i);
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
    if (kv.first > 0 && kv.second.line < 0) c.mapped += kv.second.numbers();
  c.live_bytes = c.mapped * d_->chunk_image();
  return c;
}
void Model::state_pin(const std::vector<int>& h, const std::vector<int>& live, const std::vector<int>& last) {
  auto& s = d_->store_of();
  s.pinned = h;
  s.live = live;
  s.last = last;
}
void Model::slot_unreserve(int s) { d_->slot_unreserve(s); }

// Numbers `snaps` window bookmarks take (a bookmark is snap_bytes in 2 MiB numbers).
int Model::state_bookmark_numbers(int snaps) const {
  if (snaps <= 0) return 0;
  return (int)((size_t)snaps * bk_nums(snap_bytes(d_->spec != nullptr)));
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
// images, the reservations, the evict-last list, the pool's lending): compared across the ranks' logs under DSGB10_STORE_CHECK=1.
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
        for (const size_t o : x.dl[a]) mix(o);
      }
      for (const auto& [pos, b] : x.snaps) {
        mix((uint64_t)(uint32_t)pos), mix(b.size());
        for (const uint32_t c : b) mix(c);
      }
      for (const auto& [pos, o] : x.dsnaps) mix((uint64_t)(uint32_t)pos), mix(o);
      for (const uint32_t c : x.end) mix(c);
      mix(x.dend);
    }
    for (const size_t o : s.ndisk) mix(o);
    for (const int k : s.last) mix((uint64_t)(uint32_t)k);
  }
  for (const uint32_t r : m.refs) mix(r);
  // the pool's lending: the groups out of it (the windows' and the Engram cache's), those owed, the pool's size before any
  // left, the cache's own groups and the phase of its change
  mix((uint64_t)(uint32_t)m.lent.groups), mix((uint64_t)(uint32_t)m.lent.owed), mix((uint64_t)m.lent.n0);
  mix((uint64_t)(uint32_t)m.el.cur_groups), mix((uint64_t)(uint32_t)m.el.phase);
  if (m.calloc) mix((uint64_t)m.calloc->numbers());
  for (int t = 0; t < m.nslots; ++t) mix((uint64_t)(uint32_t)m.resv[t]), mix((uint64_t)(uint32_t)m.resv_img[t]);
  if (m.store_s)
    m.store_s->img.each([&](int key, const imgrows::Ledger::Rows& r) {
      mix((uint64_t)(uint32_t)key), mix(r.rows);
      for (const uint32_t c : r.nums) mix(c);
    });
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
void Model::state_flush(int h) { d_->state_flush(h); }
bool Model::state_disk_resize(size_t bytes) { return d_->state_disk_resize(bytes); }
uint64_t Model::state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                 std::string* why) {
  return d_->state_kept_check(base, run_id, max_age_s, bytes, why);
}
void Model::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  d_->state_disk(base, bytes, staging, reuse, run_id);
}
void Model::state_keep(uint64_t gen) { d_->state_keep(gen); }
void Model::state_checkpoint(uint64_t gen) { d_->state_checkpoint(gen); }
std::vector<int> Model::state_checkpointed() const { return d_->store_s ? d_->store_s->ckpt_sess : std::vector<int>{}; }
uint64_t Model::state_version() const { return d_->store_s ? d_->store_s->ver : 0; }
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
void Model::state_set_factor(int h, int factor) {
  if (factor != 16 && factor != 32 && factor != 64) die("state: session %d tagged with factor %d", h, factor);
  if (!d_->store_s) return;
  const auto it = d_->store_s->sess.find(h);
  if (it == d_->store_s->sess.end()) return;
  if (it->second.factor != factor && (it->second.saved > 0 || it->second.end_pos >= 0 || !it->second.snaps.empty()))
    die("state: session %d holds state of factor %d and is tagged %d (a session keeps its factor for life)", h,
        it->second.factor, factor);
  it->second.factor = factor;
}
int Model::state_factor(int h) const {
  if (!d_->store_s) return 16;
  const auto it = d_->store_s->sess.find(h);
  return it == d_->store_s->sess.end() ? 16 : it->second.factor;
}
Model::StateInfo Model::state_info(int h) const {
  StateInfo r;
  if (!d_->store_s) return r;
  const auto it = d_->store_s->sess.find(h);
  if (it == d_->store_s->sess.end()) return r;
  const Store::Sess& ss = it->second;
  const bool spec = d_->spec != nullptr;
  r.saved = ss.saved;
  r.on_disk = ss.out();
  r.restart = restart_of(ss.saved, ss.end_pos, ss.snaps);  // its pieces in the pool or on the disk
  r.end_pos = ss.end_pos;
  for (const auto& sn : ss.snaps) r.snaps.push_back(sn.first);
  r.bytes = ss.numbers() * d_->chunk_image();
  for (const auto& d : ss.dl)
    for (const size_t o : d) r.disk_bytes += o != Pool::kNone ? Pool::round(d_->chunk_image()) : 0;
  r.disk_bytes += ss.dsnaps.size() * Pool::round(snap_bytes(spec)) + (ss.dend != Pool::kNone ? Pool::round(end_bytes(spec)) : 0);
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
      if (kv.first > 0) ++(kv.second.out() ? t.on_disk : t.resident);
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
  for (int s = 0; s < m.nslots; ++s) p.reserved += (uint64_t)std::max(0, m.resv[s] - m.held(s)) + (uint64_t)m.resv_img[s];
  p.image_numbers = m.img_numbers();
  if (const detail::Store* st = m.store_s.get())
    for (const auto& [k, ss] : st->sess) {
      if (k <= 0) continue;  // a line's own session (tools, tests)
      ++(ss.line >= 0 ? p.on_lines : ss.out() ? p.on_disk : p.in_memory);
      bool sh = false;
      for (const detail::Nums& l : ss.lists)
        for (const uint32_t c : l) sh = sh || (c && m.refs[c] > 1);
      p.sharing += sh;
    }
  return p;
}
void Model::drain() { d_->drain(); }

}  // namespace model
