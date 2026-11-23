// Sessions' saved state (model.h state_*, state.h): the device's side. The bookkeeping (which chunk number holds what, the
// disk tier's images, the order eviction takes pieces in, the kept index) is detail::StoreCore (store_core.h, host only); this
// file is what the core asks of the device in the middle of an operation (EngineDev: StoreDev) and the device copies that
// follow one. Every copy is queued on the model's stream, so a block freed on the host can be handed out again at once. Blocks
// come from the pool reserved at startup (state.h): taking a snapshot inside the decode loop allocates nothing.
#include "model/state.h"

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

namespace qm = ::model::quarter;  // EngineDev has members named quarter and unquarter

namespace {

constexpr size_t kArr[4] = {256, 32, 64, 4};  // bytes per entry: KV, KV scales, index key, key scales

// The store's constants are the core's own copies (store_geom.h); each must equal the header that owns it.
static_assert(sgeom::kChunkShift == attn::kChunkShift && sgeom::kChunkEntries == attn::kChunkEntries, "the chunk number's entries");
static_assert(sgeom::kSnapRows == kSnapRows, "a window bookmark's positions");
static_assert(sgeom::kMapLookahead == kMapLookahead, "the lookahead");
static_assert(sgeom::kBk == (size_t)attn::kChunkEntries * 256, "a bookmark number's bytes");
static_assert(sgeom::kBk == imgrows::kNumberBytes, "an image row's number is a bookmark's: the first array's slice");
static_assert(quarter::arr_bytes(0) == kArr[0] && quarter::arr_bytes(1) == kArr[1] && quarter::arr_bytes(2) == kArr[2] &&
                  quarter::arr_bytes(3) == kArr[3], "the quarter map's arrays");
static_assert(kMaxSlots == 64 && kMaxUnits == 128, "the core's arrays");
static_assert(StoreCore::kNoLine == Model::Impl::kNoLine, "no session");

double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// A window snapshot's block and an end snapshot's: the draft module's windows follow the layers' rows
size_t snap_bytes(bool spec) { return kSnapBytes + (spec ? kSpecSnapBytes : 0); }
size_t end_bytes(bool spec) { return kEndBytes + (spec ? kSpecEndBytes : 0); }
// A bookmark's image is this rank's slice and the trailer's used words in one block: with the draft module's windows too, by the compiler's word
static_assert(qm::bk_trailer_off(kSnapBytes + kSpecSnapBytes) + qm::kTrailerUsed <= sgeom::kBookmarkImageBytes, "a window bookmark's slice and trailer fit its image block");
static_assert(qm::bk_trailer_off(kEndBytes + kSpecEndBytes) + qm::kTrailerUsed <= sgeom::kEndImageBytes, "the end bookmark's slice and trailer fit its image block");

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
  const alloc_seal::Scoped seal(alloc_seal::kStore);
  core->store_of();  // the store is made here
  EngineDev& d = static_cast<EngineDev&>(*store_dev);
  if (d.gather) die("state: the store is reserved once");
  if (!calloc) die("state: the store needs the chunk pool");
  // a page or a bookmark at a time between the chunks and the disk tier (the draft module's windows counted in)
  d.gather = dalloc<uint8_t>(Pool::round(std::max({chunk_image(), end_bytes(true), snap_bytes(true)})));
  if ((opt.kv_split || opt.imager) && !d.qbuf) die("state: the split's buffers were not made before the free-memory read (Model::split_prepare)");
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

uint8_t* Model::Impl::bk_at(const Nums& n, size_t off) const {
  return pool_arr[0] + ((size_t)n[off / sgeom::kBk] << attn::kChunkShift) * kKvEntryBytes[0] + off % sgeom::kBk;
}

void Model::Impl::bk_copy(const Nums& n, size_t off, uint8_t* dev, size_t bytes, bool to_bk, cudaStream_t on) {
  const cudaStream_t use = on ? on : st;  // the image stream's copy of a bookmark's bytes goes on that stream
  while (bytes) {
    const size_t m = std::min(bytes, sgeom::kBk - off % sgeom::kBk);
    uint8_t* b = bk_at(n, off);
    CUDA_CHECK(cudaMemcpyAsync(to_bk ? b : dev, to_bk ? dev : b, m, cudaMemcpyDeviceToDevice, use));
    off += m, dev += m, bytes -= m;
  }
}

// A bookmark's bytes all defined: zero, in the same pieces bk_copy moves them.
void Model::Impl::bk_zero(const Nums& n, size_t bytes) {
  size_t off = 0;
  while (bytes) {
    const size_t m = std::min(bytes, sgeom::kBk - off % sgeom::kBk);
    CUDA_CHECK(cudaMemsetAsync(bk_at(n, off), 0, m, st));
    off += m, bytes -= m;
  }
}

// ---- image rows (img_rows.h): a running request's rows are numbers no session names ---------------------------------------------

size_t Model::Impl::img_numbers() const { return core->img_numbers(); }

const uint8_t* Model::Impl::img_row(int key, size_t row) const {
  const imgrows::Ledger::Rows* r = core->img_find(key);
  if (!r || row >= r->rows) die("model: image row %zu of flight %d is not in the store", row, key);
  const imgrows::Where w = imgrows::where(row);
  return pool_arr[0] + ((size_t)r->nums[w.number] << attn::kChunkShift) * kKvEntryBytes[0] + w.offset;
}

// The numbers for flight `key`'s rows: lowest free first, evicting idle conversations while too few are free (the
// command's pinned sessions stay). The slot's reservation for them is spent either way.
bool Model::Impl::img_take(int key, size_t rows, int slot) { return core->img_take(key, rows, slot); }

// Rows [row0, row0 + n) from host memory into the flight's numbers, one staging piece at a time: the piece waits for the
// copy that last read the buffer, and its own copy is queued on the model's stream behind everything that read the numbers.
void Model::Impl::img_fill(int key, size_t row0, size_t n, const uint16_t* src) {
  const imgrows::Ledger::Rows* r = core->img_find(key);
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
  const bool had = core->has_store();
  core->img_free(key);
  if (!had) return;
  for (ImgSet& d : imgs)  // a view parked on a unit that read the rows
    if (d.key == key || key == 0) d.span.clear(), d.row0.clear(), d.key = 0;
  if (img_key == key || key == 0) img_span.clear(), img_row0.clear(), img_key = 0;
}

// ---- lines and slots: the core's bookkeeping, and the reads of the lists the steps make -----------------------------------------

int Model::Impl::held(int s) const { return core->held(s); }

uint8_t* Model::Impl::ent_addr(int k, int a, int j) const {
  const int h = core->line_h[cur_unit];
  if (h == kNoLine) die("model: entry %d of kv source %d in slot %d, which holds no session", j, k, cur_slot);
  const Refs& l = core->store_s->sess.at(h).lists[k];
  if ((size_t)(j >> attn::kChunkShift) >= l.size()) die("model: entry %d of kv source %d past slot %d's list", j, k, cur_slot);
  return pool_arr[a] + (((size_t)l[j >> attn::kChunkShift].num() << attn::kChunkShift) | (j & (attn::kChunkEntries - 1))) * kKvEntryBytes[a];
}

void Model::Impl::ensure_mapped(int s, int upto) { core->ensure(s, upto); }

void Model::Impl::slot_reserve(int s, int need, int img) {
  if (s < 0 || s >= nslots) die("model: slot_reserve in slot %d of %d", s, nslots);
  core->slot_reserve(s, slot_chunks_for(s, need), img);
}
void Model::Impl::slot_unreserve(int s) { core->slot_unreserve(s); }
void Model::Impl::slot_attach(int s, int h, int upto) {
  core->slot_attach(s, h, upto);
  if (spec && spec->copy && s >= 0 && s < nslots) spec->copy->attach(unit_of[s], h, st);  // copy drafting: a conversation's counts start over with a new one
}
void Model::Impl::slot_release(int s) { core->slot_release(s); }

// ---- bookmarks: the numbers from the core, the copies here -------------------------------------------------------------------

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
const Nums* Model::Impl::state_snap_buf(int h, int pos) { return core->snap_buf(h, pos); }

void Model::Impl::state_snap(int h, int pos) {
  if (pos < kSnapRows) die("state: snapshot at %d", pos);
  const Nums* b = core->snap_buf(h, pos);
  if (!b) return;
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(*this, ring_kv[L], *b, snap_kv_off(L), kHead, pos - kSnapRows, pos, true);
    ring_rows(*this, ring_sc[L], *b, snap_sc_off(L), kHead / 32, pos - kSnapRows, pos, true);
  }
  if (spec) spec_windows(*this, *spec, *b, kSnapBytes, pos - kSnapRows, pos, true);
}

// The draft module's windows into the bookmark at pos, which the prefill took (none: not taken).
void Model::Impl::state_snap_spec(int h, int pos) {
  const Nums* b = core->snap_peek(h, pos);
  if (b) spec_windows(*this, *spec, *b, kSnapBytes, pos - kSnapRows, pos, true);
}

void Model::Impl::state_restore(int h, int pos) {
  const Nums& b = core->snap_get(h, pos);
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(*this, ring_kv[L], b, snap_kv_off(L), kHead, pos - kSnapRows, pos, false);
    ring_rows(*this, ring_sc[L], b, snap_sc_off(L), kHead / 32, pos - kSnapRows, pos, false);
  }
  if (spec) spec_windows(*this, *spec, b, kSnapBytes, pos - kSnapRows, pos, false);
}

void Model::Impl::state_snap_end(int h, int pos) {
  const Nums* e = core->end_take(h, pos);
  if (!e) return;
  size_t off = 0;
  for (int L = 0; L < kLayers; ++L, off += kRingBytes) {
    bk_copy(*e, off, ring_kv[L], (size_t)attn::kRing * kHead, true);
    bk_copy(*e, off + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32, true);
  }
  for (int k = 0; k < kKvSlots; ++k, off += kCarryBytes) {
    bk_copy(*e, off, reinterpret_cast<uint8_t*>(kvs[k].carry_kv), kCarryBytes / 2, true);
    bk_copy(*e, off + kCarryBytes / 2, reinterpret_cast<uint8_t*>(kvs[k].carry_sc), kCarryBytes / 2, true);
  }
  if (spec) spec_windows(*this, *spec, *e, kEndBytes, -1, 0, true);
}

void Model::Impl::state_restore_end(int h) {
  const Nums& end = core->end_get(h);
  size_t off = 0;
  for (int L = 0; L < kLayers; ++L, off += kRingBytes) {
    bk_copy(end, off, ring_kv[L], (size_t)attn::kRing * kHead, false);
    bk_copy(end, off + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32, false);
  }
  for (int k = 0; k < kKvSlots; ++k, off += kCarryBytes) {
    bk_copy(end, off, reinterpret_cast<uint8_t*>(kvs[k].carry_kv), kCarryBytes / 2, false);
    bk_copy(end, off + kCarryBytes / 2, reinterpret_cast<uint8_t*>(kvs[k].carry_sc), kCarryBytes / 2, false);
  }
  if (spec) spec_windows(*this, *spec, end, kEndBytes, -1, 0, false);
}

// ---- Model's state_* : forwards to the core, and the copies of the bookmarks ---------------------------------------------------

void Model::state_reserve(size_t bytes) {
  d_->state_reserve(bytes);
  d_->core->trace_open();
}
size_t Model::state_capacity() const { return d_->store_numbers * d_->chunk_image(); }  // the store's share of the pool
size_t Model::state_ceiling() const { return d_->calloc ? d_->calloc->total() * d_->chunk_image() : 0; }  // all of it
size_t Model::state_giveable(const std::vector<int>& keep) const { return d_->core->giveable(keep); }
size_t Model::state_giveable_joint(const std::vector<int>& keep, const std::vector<int>& lines) const {
  return d_->core->giveable_joint(keep, lines);
}
std::vector<std::pair<int, int>> Model::state_snap_numbers(int h) const { return d_->core->snap_numbers(h); }
void Model::state_list_numbers(int h, int upto, int* in_pool, int* on_disk) const { d_->core->list_numbers(h, upto, in_pool, on_disk); }
uint64_t Model::state_sum_cuts() const { return d_->core->sum_cuts; }
size_t Model::state_restore_chunks(int h) const { return d_->core->restore_chunks(h); }  // the numbers reading h's pieces on the disk tier back takes
Model::StoreChunks Model::state_chunks() const {  // the chunk pool: numbers in all, held by the sessions no line holds
  StoreChunks c;
  const StoreCore::Chunks k = d_->core->chunks();
  c.total = k.total, c.mapped = k.mapped, c.live_bytes = k.live_bytes;
  return c;
}
void Model::state_pin(const std::vector<int>& h, const std::vector<int>& live, const std::vector<int>& last) {
  d_->core->pin(h, live, last);
}
void Model::slot_unreserve(int s) { d_->slot_unreserve(s); }

// Numbers `snaps` window bookmarks take (a bookmark is snap_bytes in 2 MiB numbers).
int Model::state_bookmark_numbers(int snaps) const { return d_->core->bookmark_numbers(snaps); }

// Numbers a copy on write at position `pos` takes: with h_own == h_src, own_tail's count (every list's numbers from the
// cut's chunk on that another list or bookmark also names, one fresh number each); with h_own another session (a fork of
// h_src at pos), one number per list for the partial chunk at pos. Reads the store's records only (the same on every rank).
int Model::state_cow_numbers(int h_src, int h_own, int pos) const { return d_->core->cow_numbers(h_src, h_own, pos); }

// A hash of the state every rank must hold alike (sessions with their lists, bookmarks and images, the numbers' counts and
// images, the reservations, the evict-last list, the pool's lending): compared across the ranks' logs under DSGB10_STORE_CHECK=1.
uint64_t Model::store_digest() const { return d_->core->digest(); }
std::vector<int> Model::state_evicted() { return d_->core->take_evicted(); }
void Model::state_save(int h, int from, int to) { d_->core->save(h, from, to, d_->cur_unit); }
void Model::state_load(int h, int to) { d_->core->load(h, to, d_->cur_unit, d_->cur_slot); }
void Model::state_snap(int h, int pos) { d_->state_snap(h, pos); }
void Model::state_snap_end(int h, int pos) { d_->state_snap_end(h, pos); }
void Model::state_restore(int h, int pos) { d_->state_restore(h, pos); }
void Model::state_restore_end(int h) { d_->state_restore_end(h); }
void Model::state_drop(int h, int from) { d_->core->drop(h, from); }
void Model::state_free(int h) { d_->core->free_session(h); }
void Model::state_copy_snaps(int src, int dst, int upto) { d_->core->copy_snaps(src, dst, upto); }
void Model::state_drop_snap(int h, int pos) { d_->core->drop_snap(h, pos); }
void Model::state_flush(int h) { d_->core->flush(h); }
bool Model::state_disk_resize(size_t bytes) { return d_->core->disk_resize(bytes); }
uint64_t Model::state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                 std::string* why) {
  return d_->core->kept_check(base, run_id, max_age_s, bytes, why);
}
void Model::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  d_->core->disk(base, bytes, staging, reuse, run_id);
}
void Model::state_keep(uint64_t gen) { d_->core->keep(gen); }
void Model::state_checkpoint(uint64_t gen) { d_->core->checkpoint(gen); }
std::vector<int> Model::state_checkpointed() const { return d_->core->checkpointed(); }
uint64_t Model::state_version() const { return d_->core->version(); }
uint64_t Model::state_kept() const { return d_->core->kept_generation(); }
std::vector<int> Model::state_sessions() const { return d_->core->sessions(); }
size_t Model::state_disk_capacity() const { return d_->core->disk_capacity(); }
size_t Model::state_disk_bytes() const { return d_->core->disk_bytes(); }
std::string Model::state_disk_path() const { return d_->core->disk_path(); }
void Model::state_spill(int h) { d_->core->spill(h); }
void Model::state_split(int h, int budget) { d_->core->split(h, budget); }
// The imager. image: queued on the image stream, never waited for; settle: the entries of the commands at or below `upto` are clean and
// the idle conversations are marked (host bookkeeping, no device call); image_done: this rank's reading of the highest command whose work ended.
// A ride (cmd.h split_ride_run): the commands every rank has finished are settled, then h's pieces are queued
void Model::state_ride(uint64_t settled, int h, int budget, const std::vector<int>& live) {
  ++d_->core->split_counts.rides;
  d_->core->settle(settled, live);
  if (h > 0 && budget > 0) d_->core->image(h, budget, live);
}
int Model::image_room_pieces(int maxn) const {
  int n = maxn;
  while (n > 0 && !d_->store_dev->image_room((size_t)n * sgeom::kQuarterDiskBytes)) --n;
  return n;
}
void Model::ride_readings(uint64_t out[4]) const {
  const StoreCore& c = *d_->core;
  out[0] = c.image_seq(), out[1] = c.image_pending(), out[2] = c.calloc ? c.calloc->ghosts() : 0, out[3] = c.calloc ? c.calloc->free() : 0;
}
bool Model::has_host_ring() const { return d_->host_ring != nullptr; }
uint64_t Model::image_pending() const { return d_->core->image_pending(); }
int Model::image_todo(int h) const { return d_->core->image_todo(h); }
uint64_t Model::image_done() const { return d_->store_dev->image_done(); }
bool Model::image_room(size_t bytes) const { return d_->store_dev->image_room(bytes); }
uint64_t Model::image_seq() const { return d_->core->image_seq(); }
bool Model::imager() const { return d_->core->imager(); }
void Model::state_join(int h, int upto, int max_numbers) {
  d_->core->join_pieces = std::max(1, d_->opt.kv_join_pieces);
  d_->core->join(h, upto, (size_t)std::max(0, max_numbers), d_->core->join_pieces);
}
// The split's buffers, made at the start before the free-memory read the store is sized from, so that the read already lacks them
// (pool_rules.h kSplit is a Counted line of the start table, class Fixed). A no-op without --kv-split cache.
void Model::split_prepare() {
  if (d_->opt.kv_split || d_->opt.imager) static_cast<EngineDev&>(*d_->store_dev).prepare();
}
void Model::state_join_plan(int h, int upto, int* back, int* exch) const { d_->core->join_plan(h, upto, back, exch); }
Model::SplitStats Model::split_stats() const {
  const StoreCore& c = *d_->core;
  SplitStats r;
  r.split_pieces = c.split_counts.split_pieces, r.split_deferred = c.split_counts.split_deferred, r.revives = c.split_counts.revives;
  r.joins = c.split_counts.joins, r.join_bytes = c.split_counts.join_bytes, r.join_ns = c.split_counts.join_ns;
  r.join_max_ns = c.split_counts.join_max_ns, r.join_short = c.split_counts.join_short, r.join_bad = c.split_counts.join_bad;
  r.exchanged = c.split_counts.exchanged, r.join_taken = c.split_counts.join_taken, r.unspill_taken = c.split_counts.unspill_taken;
  r.imaged = c.split_counts.imaged, r.image_voids = c.split_counts.image_voids, r.image_waits = c.split_counts.image_waits;
  r.returns_read = c.split_counts.returns_read, r.rides = c.split_counts.rides;
  const StoreCore::ImageView iv = c.image_view();
  r.clean_numbers = iv.clean_numbers, r.image_pending = iv.image_pending;
  const StoreCore::SplitView v = c.split_view();
  r.conv_split = v.conv_split, r.quarter_slots = v.quarter_slots, r.quarter_numbers = v.quarter_numbers, r.ghost_numbers = v.ghost_numbers;
  return r;
}
bool Model::split_can_open() const { return d_->core->split_can_open(); }
void Model::state_list_numbers(int h, int upto, int* in_pool, int* on_disk, int* split) const {
  d_->core->list_numbers(h, upto, in_pool, on_disk, split);
}
void Model::state_unspill(int h, int max_numbers, int upto) {
  const size_t cap = max_numbers < 0 ? StoreCore::kSpent : (size_t)max_numbers;
  if (upto > 0) d_->core->unspill(h, cap, upto, std::max(1, d_->opt.kv_join_pieces));
  else d_->core->unspill(h, cap);
}
int Model::state_unspill_left(int h, int upto) const { return d_->core->unspill_left(h, upto); }
void Model::state_set_factor(int h, int factor) { d_->core->set_factor(h, factor); }
int Model::state_factor(int h) const { return d_->core->factor(h); }
Model::StateInfo Model::state_info(int h) const {
  const StoreCore::Info i = d_->core->info(h);
  StateInfo r;
  r.saved = i.saved, r.end_pos = i.end_pos, r.snaps = i.snaps, r.bytes = i.bytes, r.on_disk = i.on_disk;
  r.disk_bytes = i.disk_bytes, r.restart = i.restart, r.split = i.split, r.slots = i.slots;
  return r;
}
size_t Model::state_bytes() const {  // device memory the store's sessions hold (no line holds them)
  return state_chunks().live_bytes;
}
Model::TierStats Model::tier_stats() const {
  const StoreCore::TierStats c = d_->core->tier_stats();
  TierStats t;
  t.store_cap = c.store_cap, t.store_used = c.store_used, t.resident = c.resident, t.on_disk = c.on_disk;
  t.disk_cap = c.disk_cap, t.disk_used = c.disk_used, t.spills = c.spills, t.spill_bytes = c.spill_bytes;
  t.restores = c.restores, t.restore_bytes = c.restore_bytes, t.restored_pos = c.restored_pos, t.wait_ns = c.wait_ns;
  t.wait_max_ns = c.wait_max_ns;
  return t;
}
Model::PoolStats Model::pool_stats() const {
  const StoreCore::PoolStats c = d_->core->pool_stats();
  PoolStats p;
  p.numbers = c.numbers, p.free = c.free, p.held = c.held, p.shared = c.shared, p.reserved = c.reserved;
  p.number_bytes = c.number_bytes, p.image_numbers = c.image_numbers, p.on_lines = c.on_lines, p.in_memory = c.in_memory;
  p.on_disk = c.on_disk, p.sharing = c.sharing;
  return p;
}
void Model::drain() { d_->drain(); }

}  // namespace model
