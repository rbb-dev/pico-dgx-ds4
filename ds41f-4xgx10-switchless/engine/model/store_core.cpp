// The session store's bookkeeping (store_core.h). Host only: nothing here includes a CUDA header.
#include "model/store_core.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iterator>
#include <unordered_map>

#include "common/keepfile.h"
#include "common/log_detail.h"
#include "model/pool_rules.h"

namespace attn = ::model::sgeom;  // the geometry's names, as the engine's sources spell them

namespace model::detail {

namespace {

double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool in(const std::vector<int>& v, int k) { return std::find(v.begin(), v.end(), k) != v.end(); }

// the entries the unit u's rows hold: what the trace records after an entry that puts a session in a line
int64_t trace_held(const StoreCore& m, int u) {
  const int h = m.line_h[u];
  if (h == StoreCore::kNoLine || !m.store_s) return 0;
  const auto it = m.store_s->sess.find(h);
  if (it == m.store_s->sess.end()) return 0;
  int64_t n = 0;
  for (const Refs& l : it->second.lists) n += (int64_t)l.size();
  return n;
}

// the entries of list k's piece i that hold saved positions: all of them for a full piece, the written count for the last
int written_entries(const Store::Sess& ss, int k, size_t i) {
  const int ratio = k < 3 ? 2 : 1, total = (ss.saved + ratio - 1) / ratio;
  return std::min(sgeom::kChunkEntries, total - (int)i * sgeom::kChunkEntries);
}

// whether the session has a piece only on the disk tier (it is read back before use)
bool is_out(const Store::Sess& x) {
  for (const Refs& l : x.lists)
    for (const Ref c : l)
      if (c.on_ssd()) return true;
  for (const auto& sn : x.snaps)
    if (sn.second.empty() && !x.qsnaps.count(sn.first)) return true;
  return x.end.empty() && x.qend.empty() && x.dend != Pool::kNone;
}

// whether session x holds a chunk number at or above `from` (in a list, a window bookmark or the end bookmark)
bool holds_from(const Store::Sess& x, uint32_t from) {
  for (const Refs& l : x.lists)
    for (const Ref c : l)
      if (c && c.holder() >= from) return true;
  for (const auto& sn : x.snaps)
    for (const uint32_t c : sn.second)
      if (c >= from) return true;
  for (const uint32_t c : x.end)
    if (c >= from) return true;
  return false;
}
// the numbers at or above `from` that image flights hold (img_rows.h), and how many flights hold them
size_t image_numbers_from(const imgrows::Ledger& img, uint32_t from, int* flights = nullptr) {
  size_t n = 0;
  int f = 0;
  img.each([&](int, const imgrows::Ledger::Rows& r) {
    size_t k = 0;
    for (const uint32_t c : r.nums) k += c >= from;
    n += k, f += k > 0;
  });
  if (flights) *flights = f;
  return n;
}

}  // namespace

// What a restart needs (serve.cpp choose): the end snapshot at e when the caches hold [0, e) (saved >= e),
// a window snapshot at s when they hold its restart position q (s, or s - 1 for an odd s; saved >= q).
int snap_q(int s) { return s & 1 ? s - 1 : s; }
// the first position kv source k's entry i holds
int ent_pos(int k, int i) { return i * attn::kChunkEntries * (k < 3 ? 2 : 1); }
bool end_usable(int end_pos, int saved) { return end_pos >= 0 && end_pos <= saved; }
bool snap_usable(int s, int saved) { return snap_q(s) <= saved; }
size_t bk_nums(size_t bytes) { return (bytes + attn::kBk - 1) / attn::kBk; }
// numbers kv source k's list needs for positions [0, pos): its entries and the step's reach past them (model.cu)
int nums_for_pos(int k, int pos) {
  if (pos <= 0) return 0;
  const int ratio = k < 3 ? 2 : 1, entries = (pos - 1) / ratio + 1 + 64;
  return (entries + attn::kChunkEntries - 1) / attn::kChunkEntries;
}

bool Store::Sess::out() const { return is_out(*this); }

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

// ---- kept sessions ---------------------------------------------------------------------------------
//
// The index of the disk tier's file (base.index beside base.spill; keepfile.h: private, whole or not at all): a header, then every
// session in the file with its blocks' offsets and their checksums as they went out. Every rank writes its own with the same gen;
// the next start reuses the files only when every rank's index is whole, of this run's id, young enough, and of the same gen.

namespace {

constexpr char kKeptMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'K', '3'};  // the last byte is the index layout's version; another version is "not a kept index"
// --kv-split cache: K4 = K3 with a flag byte after each cache entry's image (1: a quarter image, kDiskBytes, else a whole one). A K3 file
// under cache and a K4 file under off are another layout: a clean start, never read wrong.
constexpr char kKeptMagic4[8] = {'D', 'S', 'G', 'B', '1', '0', 'K', '4'};
// the imager: K5 = every image is a quarter image with a trailer (a cache entry's, or a bookmark's slice); no flag byte, no whole image.
// A K3 or K4 file under the imager is another layout: a clean start, never read wrong.
constexpr char kKeptMagic5[8] = {'D', 'S', 'G', 'B', '1', '0', 'K', '5'};

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

uint64_t StoreCore::kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                       std::string* why) {
  store_trace::Scope sc(store_trace::Kind::kept_check, this);
  sc.arg_s("base", base).arg_u("run_id", run_id).arg_d("max_age", max_age_s);
  Store& s = store_of();
  sc.answer_at_exit("kept_gen", &s.kept_gen);
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
  const std::time_t now = dev.now();
  store_trace::answer("now", (int64_t)now);
  const double age = (double)now - (double)hd.written;
  if (std::memcmp(hd.magic, g.imager ? kKeptMagic5 : g.split ? kKeptMagic4 : kKeptMagic, sizeof hd.magic) != 0) return bad("not a kept index");
  if (hd.run_id != run_id && !rule(TestRule::kKeptIgnoresRun)) return bad("of another engine binary, weights or options");
  if (hd.rank != rank) return bad("of another rank");
  if (age > max_age_s || age < -600) return bad("written " + std::to_string((long long)age) + " s ago");
  if (hd.image_bytes != chunk_image() || hd.snap_bytes != snap_bytes() || hd.end_bytes != end_bytes())
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
  const int max_ents = (max_ctx + 64) / attn::kChunkEntries + 2;
  for (int i = 0; i < hd.n_sess && r.ok; ++i) {
    Store::Kept k;
    k.h = r.get<int32_t>(), k.saved = r.get<int32_t>(), k.end_pos = r.get<int32_t>();
    const int ns = r.get<int32_t>(), he = r.get<int32_t>();
    k.used = r.get<uint64_t>();
    k.factor = r.get<int32_t>();
    if (!r.ok || (k.factor != 16 && k.factor != 32 && k.factor != 64) || k.h <= 0 || !hs.insert(k.h).second || ns < 0 || ns > 4096 || (he != 0 && he != 1) || k.saved < 0 ||
        k.saved > max_ctx || k.end_pos < -1 || k.end_pos > max_ctx)
      return bad("a bad session record");
    for (int kk = 0; kk < kKvSlots; ++kk) {
      const int ne = r.get<int32_t>();
      if (!r.ok || ne < 0 || ne > max_ents) return bad("a bad session record");
      for (int j = 0; j < ne; ++j) {
        const size_t off = r.get<uint64_t>();
        const uint64_t sum = r.get<uint64_t>();
        const uint8_t q = g.imager ? 1 : g.split ? r.get<uint8_t>() : 0;
        if (!r.ok || q > 1) return bad("a bad session record");
        if (!block(off, q ? sgeom::kQuarterDiskBytes : chunk_image())) return bad("an image outside the file");
        k.ents[kk].emplace_back(off, sum);
        k.qents[kk].push_back(q);
      }
    }
    for (int j = 0; j < ns; ++j) {
      const int pos = r.get<int32_t>();
      (void)r.get<int32_t>();
      const size_t off = r.get<uint64_t>();
      const uint64_t sum = r.get<uint64_t>();
      if (pos < attn::kSnapRows || pos > max_ctx || !k.snaps.emplace(pos, std::make_pair(off, sum)).second ||
          !block(off, snap_ibytes()))
        return bad("a bad snapshot record");
    }
    if (he) {
      k.dend = r.get<uint64_t>(), k.dend_sum = r.get<uint64_t>();
      if (!block(k.dend, end_ibytes())) return bad("a block outside the file");
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

void StoreCore::disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  store_trace::Scope sc(store_trace::Kind::disk, this);
  sc.arg_s("base", base).arg_u("bytes", bytes).arg_u("staging", staging).arg("reuse", reuse).arg_u("run_id", run_id);
  Store& s = store_of();
  if (s.has_disk) die("state: the disk tier is opened once");
  if (reuse && !s.kept_gen) die("state: reusing a kept file that was not checked");
  bytes = bytes / kPoolGrain * kPoolGrain;
  s.base = base, s.run_id = run_id;
  if (!bytes) return;
  const double t0 = wall_s();
  s.dpool.size = dev.disk_open(base, bytes, staging, reuse);
  s.has_disk = true;
  store_trace::answer("disk_open", (int64_t)s.dpool.size);
  s.dpool.free.emplace(0, s.dpool.size);
  if (!reuse) unlink((base + ".index").c_str());  // a reused index stays the latest checkpoint (its images held)
  unlink((base + ".index.new").c_str());
  if (reuse) {  // the kept sessions, every piece only on the disk: back on demand, each checked (state_unspill)
    dev.disk_keep();
    auto take = [&](size_t off, size_t n, uint64_t sum) {  // an image named again (a shared one) is taken once
      if (s.drefs[off]++) return;
      s.named.insert(off);
      if (!s.dpool.take_at(off, n)) die("state: a kept image at %zu is not free", off);
      s.dbytes += Pool::round(n);
      dev.sum_set(off, sum);
    };
    s.clock = s.kept_clock;
    for (const Store::Kept& k : s.kept) {
      Store::Sess& ss = s.sess[k.h];
      ss.kept = true;
      ss.saved = k.saved, ss.end_pos = k.end_pos, ss.used = k.used, ss.factor = k.factor;
      for (int kk = 0; kk < kKvSlots; ++kk)
        for (size_t e = 0; e < k.ents[kk].size(); ++e) {
          const auto& [off, sum] = k.ents[kk][e];
          const bool q = k.qents[kk][e] != 0;
          take(off, q ? sgeom::kQuarterDiskBytes : chunk_image(), sum), ss.lists[kk].push_back(kOnSsd), ss.dl[kk].push_back(off);
          if (q) s.qimg.insert(off);
        }
      for (const auto& [pos, os] : k.snaps) take(os.first, snap_ibytes(), os.second), ss.dsnaps.emplace(pos, os.first), ss.snaps[pos];
      if (k.dend != Pool::kNone) take(k.dend, end_ibytes(), k.dend_sum), ss.dend = k.dend;
      s.clock = std::max(s.clock, k.used);
    }
    s.loaded_gen = s.kept_gen;
    if (rank == 0)
      std::printf("state: %zu kept sessions in %s (%.2f of %.2f GiB), in the store in %.1f ms\n", s.kept.size(),
                  dev.disk_path().c_str(), s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0, 1e3 * (wall_s() - t0));
  }
  s.kept.clear();
  s.kept.shrink_to_fit();
}

// An index body of gen (its KeptHeader first). A planned stop (all false): every session no line holds whose pieces all
// have an image and are only on the disk. A checkpoint (all true): every session as far as its pieces from position 0 on
// have images (its length cut to its first entry without one, the bookmarks inside that). *hs: the sessions named; *names: the images.
std::vector<uint8_t> StoreCore::kept_index(uint64_t gen, bool all, std::vector<int>* hs, std::set<size_t>* names) {
  Store& s = store_of();
  std::vector<uint8_t> f(sizeof(KeptHeader));
  using keepfile::put;
  int n = 0;
  for (const auto& [h, ss] : s.sess) {
    if (h <= 0) continue;
    int saved = ss.saved;
    for (int k = 0; k < kKvSlots; ++k)
      for (size_t i = 0; i < ss.lists[k].size(); ++i)
        if (entry_image(ss, k, i) == Pool::kNone) {
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
        for (size_t i = 0; i < ss.lists[k].size() && whole; ++i) whole = entry_image(ss, k, i) != Pool::kNone;
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
        const size_t off = entry_image(ss, k, i);
        put<uint64_t>(f, off), put<uint64_t>(f, dev.sum(off, false));
        if (g.split && !g.imager) put<uint8_t>(f, s.qimg.count(off) ? 1 : 0);
        names->insert(off);
      }
    }
    for (const auto& [pos, o] : bks) {
      put<int32_t>(f, pos), put<int32_t>(f, 0), put<uint64_t>(f, o), put<uint64_t>(f, dev.sum(o, false));
      names->insert(o);
    }
    if (end) {
      put<uint64_t>(f, ss.dend), put<uint64_t>(f, dev.sum(ss.dend, false));
      names->insert(ss.dend);
    }
  }
  KeptHeader hd{};
  std::memcpy(hd.magic, g.imager ? kKeptMagic5 : g.split ? kKeptMagic4 : kKeptMagic, sizeof hd.magic);
  const std::time_t now = dev.now();
  store_trace::answer("now", (int64_t)now);
  hd.gen = gen, hd.run_id = s.run_id, hd.file_bytes = dev.disk_size(), hd.written = (int64_t)now;
  hd.rank = rank, hd.n_sess = n, hd.clock = s.clock;
  hd.image_bytes = chunk_image(), hd.snap_bytes = snap_bytes(), hd.end_bytes = end_bytes();
  hd.body_bytes = f.size() - sizeof hd;
  hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
  std::memcpy(f.data(), &hd, sizeof hd);
  return f;
}

// A planned stop (every rank alike, after the server spilled what it keeps): the file synced, then its
// index, and the file stays at exit. A rank whose index cannot be written keeps nothing (the next
// start's gens differ: every rank starts clean).
void StoreCore::keep(uint64_t gen) {
  store_trace::Scope sc(store_trace::Kind::keep, this);
  sc.arg_u("gen", gen);
  Store& s = store_of();
  if (!s.has_disk || !gen) return;
  const double t0 = wall_s();
  dev.sync();  // the checksums of what went out are in
  dev.disk_sync();
  const double t1 = wall_s();
  std::vector<int> hs;
  std::set<size_t> names;
  const std::vector<uint8_t> f = kept_index(gen, false, &hs, &names);
  std::string why;
  bool wrote = keepfile::write_atomic(s.base + ".index", f.data(), f.size(), &why);
  store_trace::answer("index_write", wrote);
  // with quarters a rank that drops a conversation the others keep breaks the next exchange: the write is agreed, all adopt or none
  if ((g.split || g.imager) && g.has_ring && !rule(TestRule::kIndexWriteUnagreed)) wrote = dev.agree_min(wrote ? 1 : 0, kSiteKeep) == 1;
  if (!wrote) {
    std::printf("state: rank %d keeps nothing: %s\n", rank, why.empty() ? "another rank could not write its index" : why.c_str());
    return;
  }
  dev.disk_keep();
  if (rank == 0)
    std::printf("state: %zu sessions kept for the next start (%.2f GiB of the file): synced in %.1f ms, index of %.1f KiB "
                "in %.1f ms\n", hs.size(), s.dbytes / 1073741824.0, 1e3 * (t1 - t0), f.size() / 1024.0, 1e3 * (wall_s() - t1));
}

// A checkpoint while serving (rank 0's server, when nothing runs; every rank alike): the file synced, then the index of
// every session as far as its pieces have images, written whole or not at all; images the previous index named that were
// dropped since are freed once this one is durable. A rank whose index cannot be written keeps the previous one.
void StoreCore::checkpoint(uint64_t gen) {
  store_trace::Scope sc(store_trace::Kind::checkpoint, this);
  sc.arg_u("gen", gen);
  Store& s = store_of();
  s.ckpt_sess.clear();
  if (!s.has_disk || !gen) return;
  const double t0 = wall_s();
  dev.sync();
  dev.disk_sync();
  const double t1 = wall_s();
  std::vector<int> hs;
  std::set<size_t> names;
  const std::vector<uint8_t> f = kept_index(gen, true, &hs, &names);
  std::string why;
  bool wrote = keepfile::write_atomic(s.base + ".index", f.data(), f.size(), &why);
  store_trace::answer("index_write", wrote);
  if ((g.split || g.imager) && g.has_ring && !rule(TestRule::kIndexWriteUnagreed)) wrote = dev.agree_min(wrote ? 1 : 0, kSiteCheckpoint) == 1;  // all ranks adopt the new set and free the retired images, or none
  if (!wrote) {
    std::printf("state: rank %d: no checkpoint: %s\n", rank, why.empty() ? "another rank could not write its index" : why.c_str());
    return;
  }
  dev.disk_keep();
  s.named = std::move(names);
  size_t freed = 0;
  for (const auto& [off, bytes] : s.retired) space_free(off, bytes), freed += Pool::round(bytes);
  s.retired.clear();
  s.ckpt_sess = std::move(hs);
  if (rank == 0 && g_log_detail)  // common/log_detail.h
    std::printf("state: checkpoint: %zu sessions (%zu images), the file synced in %.1f ms, index of %.1f KiB in %.1f ms; "
                "%.1f MiB of dropped images freed\n", s.ckpt_sess.size(), s.named.size(), 1e3 * (t1 - t0), f.size() / 1024.0,
                1e3 * (wall_s() - t1), freed / 1048576.0);
}

bool StoreCore::take_nums(int h, size_t n, Nums* out) {
  while (calloc->free() < n)
    if (!evict_step(h)) return false;
  const size_t i0 = out->size();
  calloc->take((uint32_t)n, out);
  if (refs.size() < calloc->numbers()) refs.resize(calloc->numbers(), 0);
  if (store_of().ndisk.size() < calloc->numbers()) store_of().ndisk.resize(calloc->numbers(), Pool::kNone);
  if (store_of().twin_s.size() < calloc->numbers()) store_of().twin_s.resize(calloc->numbers(), 0);
  for (size_t i = i0; i < out->size(); ++i) refs[(*out)[i]] = 1, rule(TestRule::kNoTwinCut) ? void() : cut_twin((*out)[i]);
  if (g.poison)  // a taken number holds 0xFF until written: a read of what was not written shows
    for (size_t i = i0; i < out->size(); ++i) dev.fill((*out)[i], 0xFF);
  held_peak = std::max(held_peak, (size_t)calloc->held());
  return true;
}

// Number c, named by a list entry that is about to change its content, no longer has its image as its own: a holder alone on
// the number cuts the link (a shared number is never written in place). True when a link was cut.
bool StoreCore::dirty(Ref c) {
  if (!c.is_number()) return false;  // only on the disk tier, or a quarter slot (never written in place)
  Store& s = store_of();
  const uint32_t n = c.num();
  if (refs[n] != 1 || s.ndisk[n] == Pool::kNone) return false;
  s.dnum.erase(s.ndisk[n]), s.ndisk[n] = Pool::kNone;
  return true;
}

// List entry (h, c, its image *dl) is about to hold new content: its number's link is cut and its own image dropped. The link is
// cut whether or not the entry still has an image of its own: an entry whose image the disk tier took away leaves the link on the
// number, and the next save would adopt an image that is not the number's content. With PICO_OLD_IMAGE_LINK (or the rule set
// PICO_OLD_RULES) the cut stays behind the entry's own image.
void StoreCore::entry_dirty(int h, Ref c, size_t* dl) {
  if (c.is_number() && refs[c.num()] == 1 && !rule(TestRule::kNoTwinCut)) cut_twin(c.num());  // written in place: no longer any slot's content, image or not
#if defined(PICO_OLD_RULES) || defined(PICO_OLD_IMAGE_LINK)
  if (*dl == Pool::kNone) return;
  dirty(c);
#else
  const bool cut = dirty(c);
  if (cut && *dl == Pool::kNone) {
    ++stale_cuts;
    if (rank == 0) std::printf("state: session %d: the stale image of number %u is cut\n", h, c.v);
  }
  if (*dl == Pool::kNone) return;
#endif
  img_drop(*dl, chunk_image());
  *dl = Pool::kNone;
}

// One reference to the entry's number goes (an entry only on the disk tier holds none).
void StoreCore::release(Ref c) {
  if (c == kOnSsd) return;
  if (c.is_slot()) {
    release_slot(c.slot_of());
    return;
  }
  Nums one{c.num()};
  give_nums(one);
}

// number c's content no longer equals the content of the slot it was linked to (it was handed out, written in place or cut)
void StoreCore::cut_twin(uint32_t c) {
  if (!store_s) return;
  Store& s = *store_s;
  if (g.imager) {  // a block handed out, written in place or given back: a queued image of it is void, its image link is cut (a ghost's
                   // bytes are about to change), and a bookmark's ghost group is dissolved (the other blocks of it are plain free)
    if (!s.job_of.empty() && !rule(TestRule::kSettleUnchecked)) s.job_of.erase(c);
    if (!rule(TestRule::kNoGhostCut)) {
      if (c < s.ndisk.size() && s.ndisk[c] != Pool::kNone) {
        const size_t off = s.ndisk[c];
        const auto n = s.dnum.find(off);
        if (n != s.dnum.end() && n->second == c) s.dnum.erase(n);
        s.ndisk[c] = Pool::kNone;
      }
      if (const auto b = s.ghost_bk.find(c); b != s.ghost_bk.end()) ghost_group_dissolve(b->second, c);
    }
  }
  if (c >= s.twin_s.size() || !s.twin_s[c]) {
    if (g.imager && calloc && !rule(TestRule::kNoUnghost) && !rule(TestRule::kNoGhostCut)) calloc->unghost(c);  // no link left: a plain free number
    return;
  }
  const auto it = s.twin_c.find(s.twin_s[c]);
  if (it != s.twin_c.end() && it->second == c) s.twin_c.erase(it);
  s.twin_s[c] = 0;
  if (calloc && !rule(TestRule::kNoUnghost)) calloc->unghost(c);  // nothing can revive it now: a plain free number
}

// A bookmark's ghost group (its numbers are ghosts that together hold its bytes) ends: every number of it but `except` becomes plain free.
void StoreCore::ghost_group_dissolve(size_t off, uint32_t except) {
  Store& s = store_of();
  const auto it = s.bk_ghost.find(off);
  if (it == s.bk_ghost.end()) return;
  const Nums group = std::move(it->second);
  s.bk_ghost.erase(it);
  for (const uint32_t b : group) {
    s.ghost_bk.erase(b);
    if (b != except && calloc->is_ghost(b)) calloc->unghost(b);
  }
}

// number c's content is image `off`: the number names it, the image names the number, and the image is still held by an entry
bool StoreCore::link_proven(uint32_t c, size_t off) const {
  if (!store_s || off == Pool::kNone) return false;
  const Store& s = *store_s;
  if (c >= s.ndisk.size() || s.ndisk[c] != off) return false;
  const auto n = s.dnum.find(off);
  return n != s.dnum.end() && n->second == c && s.drefs.count(off) != 0;
}

size_t StoreCore::bk_slots(size_t bytes) const { return (bytes / 4 + sgeom::kQuarterBytes - 1) / sgeom::kQuarterBytes; }

void StoreCore::qsnap_drop(Store::Sess& x, int pos) {
  const auto q = x.qsnaps.find(pos);
  if (q == x.qsnaps.end()) return;
  for (const Ref r : q->second) release_slot(r.slot_of());
  x.qsnaps.erase(q);
}

void StoreCore::qend_drop(Store::Sess& x) {
  for (const Ref r : x.qend) release_slot(r.slot_of());
  x.qend.clear();
}

// A number for quarter slots: the allocator's next below the lend zone, taken without evicting anyone (a split never makes room).
bool StoreCore::take_quarter_number(uint32_t* c) {
  Store& s = store_of();
  if (rule(TestRule::kSplitEvicts)) {  // the wrong rule: a quarter number is made room for like any other
    Nums one;
    if (!take_nums(0, 1, &one)) return false;
    *c = one[0];
    return true;
  }
  if (!calloc->take_below_zone(c)) return false;
  if (refs.size() < calloc->numbers()) refs.resize(calloc->numbers(), 0);
  if (s.ndisk.size() < calloc->numbers()) s.ndisk.resize(calloc->numbers(), Pool::kNone);
  if (s.twin_s.size() < calloc->numbers()) s.twin_s.resize(calloc->numbers(), 0);
  refs[*c] = 1;  // owned by the slot map while any slot of it is in use
  cut_twin(*c);
  if (g.poison) dev.fill(*c, 0xFF);
  held_peak = std::max(held_peak, (size_t)calloc->held());
  return true;
}

// One holder of the slot goes. The last one frees the slot (its links and its image link go) and, when it was the number's last
// used slot, the number goes back to the allocator.
void StoreCore::release_slot(uint32_t slot) {
  Store& s = store_of();
  uint32_t freed = 0;
  const uint32_t left = s.slots.unref(slot, &freed);
  if (!left) {
    if (const auto t = s.twin_c.find(slot); t != s.twin_c.end()) {
      const uint32_t n = t->second;
      s.twin_c.erase(t);
      if (n < s.twin_s.size() && s.twin_s[n] == slot) s.twin_s[n] = 0, calloc && !rule(TestRule::kNoUnghost) ? calloc->unghost(n) : void();
    }
    if (const auto d = s.slot_disk.find(slot); d != s.slot_disk.end()) {  // the slot's own reference to its image
      const size_t off = d->second;
      s.slot_disk.erase(d);
      img_drop(off, sgeom::kQuarterDiskBytes);
    }
    s.slot_written.erase(slot);
  }
  if (freed) {
    if (!refs[freed]) die("state: quarter number %u given back with no owner", freed);
    refs[freed] = 0;
    cut_twin(freed);
    if (g.poison) dev.fill(freed, 0xFD);
    calloc->give(freed);
  }
}

void StoreCore::give_nums(Nums& v) {
  for (const uint32_t c : v) {
    if (c >= refs.size() || !refs[c]) die("state: number %u given back with no list naming it", c);
    if (--refs[c] && !rule(TestRule::kGiveIgnoresRefs)) continue;  // another list or bookmark still names it
    cut_twin(c);  // a plain free number is not a ghost: nothing may revive it from a slot
    if (store_s && c < store_s->ndisk.size() && store_s->ndisk[c] != Pool::kNone)  // its image outlives it, held by entries
      store_s->dnum.erase(store_s->ndisk[c]), store_s->ndisk[c] = Pool::kNone;
    if (g.poison) dev.fill(c, 0xFD);  // a number given back holds 0xFD until taken again: a read through a stale list shows
    calloc->give(c);
  }
  v.clear();
}

// The numbers for flight `key`'s rows: lowest free first, evicting idle conversations while too few are free (the
// command's pinned sessions stay). The slot's reservation for them is spent either way.
bool StoreCore::img_take(int key, size_t rows, int slot) {
  store_trace::Scope sc(store_trace::Kind::img_take, this);
  sc.arg("key", key).arg_u("rows", rows).arg("slot", slot);
  Store& s = store_of();
  const imgrows::Take t = s.img.take(key, rows, [&](size_t n, Nums* out) { return take_nums(0, n, out); });
  if (slot >= 0 && slot < nslots) resv_img[slot] = 0;
  return sc.ret(t == imgrows::Take::kTaken);
}

void StoreCore::share_nums(const Nums& from, size_t n, Nums* out) {
  if (n > from.size()) die("state: %zu numbers shared from a list of %zu", n, from.size());
  for (size_t i = 0; i < n; ++i) {
    const uint32_t c = from[i];
    if (c >= refs.size() || !refs[c]) die("state: number %u shared with no list naming it", c);
    ++refs[c];
    out->push_back(c);
  }
}

void StoreCore::share_nums(const Refs& from, size_t n, Refs* out) {
  if (n > from.size()) die("state: %zu numbers shared from a list of %zu", n, from.size());
  for (size_t i = 0; i < n; ++i) {
    const Ref r = from[i];
    const uint32_t c = r.v;  // an entry that is not a number is not shared: num() stops there below
    if (c >= refs.size() || !refs[c]) die("state: number %u shared with no list naming it", c);
    ++refs[r.num()];
    out->push_back(r);
  }
}

// One step of making room for session h: the session that the current command does not use and no line holds gives up
// its last piece in the pool (evict_piece; every rank does the same): to the disk tier, or dropped without one. The
// command's evict-last sessions (Store::last) go after every other, then the least recently used (rules::VictimPick).
// False when none is left.
bool StoreCore::evict_step(int h) {
  store_trace::Scope sc(store_trace::Kind::evict_step, this);
  sc.arg("h", h);
  Store& s = store_of();
  STORE_CORE_COUNT(steps);
  rules::VictimPick pick;
  const bool im = g.imager && !rule(TestRule::kEvictPlain);
  for (const auto& [k, ss] : s.sess)
    if (k != h && k > 0 && ss.line < 0 && ss.in_pool() && !in(s.pinned, k) && !in(s.live, k) && (!im || has_clean(ss))) pick.offer(k, ss.used, in(s.last, k));
  bool fallback = false;  // only plain pieces are left: the count that reserved clean ones was broken; the piece is imaged and waited for
  if (im && pick.victim < 0) {
    for (const auto& [k, ss] : s.sess)
      if (k != h && k > 0 && ss.line < 0 && ss.in_pool() && !in(s.pinned, k) && !in(s.live, k)) pick.offer(k, ss.used, in(s.last, k));
    fallback = pick.victim >= 0;
  }
  int victim = pick.victim;
#ifdef STORE_CORE_TEST_RULES
  if (rule(TestRule::kVictimMru)) {  // the wrong rule: the most recently used goes first
    uint64_t best = 0;
    victim = -1;
    for (const auto& [k, ss] : s.sess)
      if (k != h && k > 0 && ss.line < 0 && ss.in_pool() && !in(s.pinned, k) && !in(s.live, k) && ss.used >= best) victim = k, best = ss.used;
  }
#endif
  if (victim < 0) return STORE_CORE_COUNT(step_none), sc.ret(false);
  if (!s.has_disk) {  // nowhere to put a piece: the session goes whole
    STORE_CORE_COUNT(step_whole);
    free_session(victim);
    s.evicted.push_back(victim);
    return sc.ret(true);
  }
  STORE_CORE_COUNT(step_piece);
  return sc.ret(evict_piece(victim, h, fallback));
}

// The chunk numbers of sessions no line holds and `keep` does not name: what eviction can give back; the conversations of `lines`
// (sessions) are counted as if no line held them. A whole number goes back when every list naming it is counted; a quarter number
// when every used slot of it is named only by counted lists. The same on every rank (it reads only the store's records).
size_t StoreCore::giveable_joint(const std::vector<int>& keep, const std::vector<int>& lines) const {
  if (!store_s) return 0;
  std::unordered_map<uint32_t, uint32_t> seen;  // a shared number goes back only when every list naming it does
  std::unordered_map<uint32_t, uint32_t> seen_slot;
  const auto add = [&](const Nums& v) {
    for (const uint32_t c : v)
      if (c) ++seen[c];
  };
  const auto add_refs = [&](const Refs& v) {
    for (const Ref c : v) {
      if (c.is_number()) ++seen[c.num()];
      else if (c.is_slot() && !rule(TestRule::kGiveableNoSlots)) ++seen_slot[c.slot_of()];
    }
  };
  for (const auto& [k, ss] : store_s->sess)
    if (k > 0 && (ss.line < 0 || in(lines, k)) && !in(keep, k)) {
      if (g.imager) {  // only a piece whose image went out can be given back without a write: a plain one is not counted, so a number
                       // two holders name is counted only when each of them is clean (the count falls short of its holders)
        for (int kk = 0; kk < kKvSlots; ++kk)
          for (size_t i = 0; i < ss.lists[kk].size(); ++i)
            if (ss.lists[kk][i].is_number() && i < ss.dl[kk].size() && ss.dl[kk][i] != Pool::kNone) ++seen[ss.lists[kk][i].num()];
        for (const auto& sn : ss.snaps)
          if (ss.dsnaps.count(sn.first)) add(sn.second);
        if (ss.dend != Pool::kNone) add(ss.end);
        continue;
      }
      for (const Refs& l : ss.lists) add_refs(l);
      for (const auto& sn : ss.snaps) add(sn.second);
      add(ss.end);
      if (!rule(TestRule::kGiveableNoSlots)) {
        for (const auto& q : ss.qsnaps) add_refs(q.second);
        add_refs(ss.qend);
      }
    }
  size_t n = 0;
  for (const auto& [c, m] : seen) n += m == refs[c];
  if (!rule(TestRule::kGiveableNoSlots))
  for (const uint32_t c : store_s->slots.open_from(0)) {  // a quarter number goes when none of its used slots is named elsewhere
    bool all = true;
    for (int q = 0; q < 4; ++q) {
      const uint32_t sl = chunks::SlotMap::slot(c, q), r = store_s->slots.refs(sl);
      if (!r) continue;
      const auto it = seen_slot.find(sl);
      all = all && it != seen_slot.end() && it->second == r;
    }
    n += all;
  }
  return n;
}

int StoreCore::held(int s) const {
  const int h = line_h[unit_of[s]];
  if (h == kNoLine || !store_s) return 0;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return 0;
  int n = 0;
  for (const Refs& l : it->second.lists)
    for (const Ref c : l) n += c.is_number();
  return n;
}

int StoreCore::held_slots(int s) const {
  const int h = line_h[unit_of[s]];
  if (h == kNoLine || !store_s) return 0;
  const auto it = store_s->sess.find(h);
  return it == store_s->sess.end() ? 0 : (int)(it->second.slot_entries() + it->second.bookmark_slots());
}

bool StoreCore::grow(int s, int upto) {
  const int u = unit_of[s];
  upto = std::max(0, std::min(upto, slot_ctx_[s] - 1));
  Store::Sess& x = line_sess(s);
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
  size_t gi = 0;
  for (int k = 0; k < kKvSlots; ++k) {
    if (!want[k]) continue;
    const size_t i0 = x.lists[k].size(), g0 = gi;
    for (int i = 0; i < want[k]; ++i, ++gi) x.lists[k].push_back(Ref(got[gi]));
    dev.rows_append(u, k, i0, got.data() + g0, (size_t)want[k]);  // entries no queued copy reads yet (rewrites wait: write_rows)
  }
  return true;
}

void StoreCore::ensure_more(int s, int upto) {
  store_trace::Scope sc(store_trace::Kind::ensure, this);
  sc.arg("slot", s).arg("upto", upto);
  int got = std::min(upto + attn::kMapLookahead, slot_ctx_[s] - 1);
  if (!grow(s, got)) {  // no room for the lookahead: what the step needs (inside its reservation, there with eviction)
    got = upto;
    if (!grow(s, got))
      die("model: slot %d cannot hold positions up to %d: %u of %u chunk numbers free, %d held by its session", s, upto,
          calloc->free(), calloc->total(), held(s));
  }
  mapped_upto[s] = got;
  sc.result(got);
}

void StoreCore::slot_reserve(int s, int numbers, int img) {
  store_trace::Scope sc(store_trace::Kind::slot_reserve, this);
  if (s < 0 || s >= nslots) die("model: slot_reserve in slot %d of %d", s, nslots);
  resv[s] = numbers;                   // the server counted the pool first (serve.cpp mem_ok)
  resv_img[s] = std::max(0, img);      // and the rows' numbers the request takes next
  sc.arg("slot", s).arg("numbers", resv[s]).arg("img", resv_img[s]);
}

// Slot s's reservation to nothing: its request has finished, and a finished request reserves no numbers (the
// slot's session keeps the numbers it holds; the next request on the line reserves again).
void StoreCore::slot_unreserve(int s) {
  store_trace::Scope sc(store_trace::Kind::slot_unreserve, this);
  sc.arg("slot", s);
  if (s < 0 || s >= nslots) die("model: slot_unreserve in slot %d of %d", s, nslots);
  resv[s] = rules::resv_after_finish(resv[s]);
  resv_img[s] = 0;
}

// The slot's unit holds session h from now on (its rows are h's lists): the session it held leaves it (keeping its
// numbers; a unit's own session is freed). h comes back from the disk tier first, and what is split of it below `upto` (what the
// line will hold) comes back whole: the tail above stays split until the restart's drop.
void StoreCore::slot_attach(int s, int h, int upto) {
  store_trace::Scope sc(store_trace::Kind::slot_attach, this);
  sc.arg("slot", s).arg("h", h);
  sc.result_fn([this, s] { return trace_held(*this, unit_of[s]); });
  if (s < 0 || s >= nslots || h <= 0) die("model: session %d attaches to slot %d", h, s);
  const int u = unit_of[s];
  if (line_h[u] == h) return;
  const int r = resv[s], ri = resv_img[s];  // the request's reservation (kStReserve comes first) stays with the slot
  slot_release(s);
  resv[s] = r, resv_img[s] = ri;
  Store::Sess& x = use_blocks(h, upto);
  if (x.line >= 0) die("model: session %d attaches to slot %d while unit %d holds it", h, s, x.line);
  x.line = u;
  line_h[u] = h;
  write_rows(u, &x);
  mapped_upto[s] = -1;  // the next step grows from the lists as they are
}

// Slot s's unit leaves its session (which keeps its numbers: no copy, no drain); a unit's own session is freed.
void StoreCore::slot_release(int s) {
  store_trace::Scope sc(store_trace::Kind::slot_release, this);
  sc.arg("slot", s);
  if (s < 0 || s >= nslots) die("model: slot_release of slot %d of %d", s, nslots);
  const int u = unit_of[s];
  const int h = line_h[u];
  if (h != kNoLine) {
    Store& st_ = store_of();
    write_rows(u, nullptr);
    if (h < 0) {
      Store::Sess& x = st_.sess.at(h);
      for (Refs& l : x.lists) {
        for (const Ref c : l) release(c);
        l.clear();
      }
      st_.sess.erase(h);
    } else if (auto it = st_.sess.find(h); it != st_.sess.end()) {
      it->second.line = -1;
      if (g.imager) st_.to_mark.insert(h);  // its clean pieces become ghosts at the next settle
    }
    line_h[u] = kNoLine;
  }
  mapped_upto[s] = -1;
  resv[s] = 0, resv_img[s] = 0;
}

bool StoreCore::own_tail(int h, int pos) {
  Store& s = store_of();
  Store::Sess& x = s.sess.at(h);
  for (int k = 0; k < kKvSlots; ++k) {  // the copies from the cut on go stale: the line writes those entries again
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    const int i0 = (std::max(0, pos) / (k < 3 ? 2 : 1)) >> attn::kChunkShift;
    for (size_t i = (size_t)i0; i < x.lists[k].size(); ++i) {
      entry_dirty(h, x.lists[k][i], &x.dl[k][i]);
    }
  }
#ifdef DSGB10_POOL_NOCOW
  return true;  // the negative control: shared numbers are written in place (fork_test must fail)
#else
  bool rows = false;
  for (int k = 0; k < kKvSlots; ++k) {
    const int e = std::max(0, pos) / (k < 3 ? 2 : 1), i0 = e >> attn::kChunkShift;
    Refs& l = x.lists[k];
    for (size_t i = (size_t)i0; i < l.size(); ++i) {
      if (!l[i].is_number() || refs[l[i].num()] < 2 || rule(TestRule::kNoCow)) continue;
      Nums fresh;
      store_of().pinned.push_back(h);
      const bool ok = take_nums(h, 1, &fresh);
      store_of().pinned.pop_back();
      if (!ok) return false;
      const int below = (int)i == i0 ? (e & (attn::kChunkEntries - 1)) : 0;  // the entries under pos stay valid
      if (below) dev.copy_entries(fresh[0], l[i].num(), below);
      release(l[i]);
      l[i] = Ref(fresh[0]);
      rows = true;
    }
  }
  if (rows && x.line >= 0) write_rows(x.line, &x);
  return true;
#endif
}

// A bookmark is a restart point, not something a step needs: it takes only numbers that leave every line's reservation
// covered (a reservation past what its lists hold may count on what evicting idle sessions gives back, serve.cpp mem_ok).
// With none to spare the bookmark is not taken: a later restart starts at an earlier one.
bool StoreCore::take_extra(int h, size_t n, Nums* out) {
  Store& s = store_of();
  const int64_t reserved = rules::unheld_sum(nslots, [&](int t) { return resv[t]; }, [&](int t) { return held(t); }, [&](int t) { return resv_img[t]; });
  const auto giveable = [&]() {  // what store_evict_step can still give back counts too
    std::vector<int> keep = s.pinned;
    keep.insert(keep.end(), s.live.begin(), s.live.end());
    keep.push_back(h);
    return this->giveable(keep);
  };
  if (!rule(TestRule::kTakeIgnoresResv) && !rules::covers((int64_t)calloc->free(), reserved, (int64_t)n, giveable)) return false;
  s.pinned.push_back(h);
  const bool ok = take_nums(h, n, out);
  s.pinned.pop_back();
  return ok;
}

// Numbers for a piece session h reads back from the disk tier: taken only beyond what running requests are reserved, so
// a restore cuts at the first piece with no room instead of eating a reservation (the restore's cut_at).
bool StoreCore::join_take_ok(int h, size_t n) {
  Store& s = store_of();
  if (fault_join_take > 0 && ++join_take_n == fault_join_take) return false;  // refused once (a fault build)
  int64_t reserved = rules::unheld_sum(nslots, [&](int t) { return resv[t]; }, [&](int t) { return held(t); }, [&](int t) { return resv_img[t]; });
  if (!rule(TestRule::kJoinTakeUnpaid)) {
    const auto it = s.sess.find(h);
    if (it != s.sess.end() && it->second.line >= 0)
      for (int t = 0; t < nslots; ++t)
        if (unit_of[t] == it->second.line) {  // the line the conversation is on pays first: its reservation falls by what it takes
          reserved -= std::min<int64_t>((int64_t)n, std::max<int64_t>(0, (int64_t)resv[t] - held(t)));
          break;
        }
  }
  const auto giveable = [&]() {
    std::vector<int> keep = s.pinned;
    keep.insert(keep.end(), s.live.begin(), s.live.end());
    keep.push_back(h);
    return this->giveable(keep);
  };
  return rules::restore_may_take((int64_t)calloc->free(), reserved, (int64_t)n, giveable);
}

bool StoreCore::take_join(int h, size_t n, Nums* out) {
  if (!join_take_ok(h, n)) return false;
  Store& s = store_of();
  s.pinned.push_back(h);
  const bool ok = take_nums(h, n, out);
  s.pinned.pop_back();
  return ok;
}

bool StoreCore::take_restore(int h, size_t n, Nums* out) {
  Store& s = store_of();
  const int64_t reserved = rules::unheld_sum(nslots, [&](int t) { return resv[t]; }, [&](int t) { return held(t); }, [&](int t) { return resv_img[t]; });
  const auto giveable = [&]() {
    std::vector<int> keep = s.pinned;
    keep.insert(keep.end(), s.live.begin(), s.live.end());
    keep.push_back(h);
    return this->giveable(keep);
  };
  if (!rules::restore_may_take((int64_t)calloc->free(), reserved, (int64_t)n, giveable)) return false;
  s.pinned.push_back(h);
  const bool ok = take_nums(h, n, out);
  s.pinned.pop_back();
  return ok;
}

bool StoreCore::own_bk(int h, Nums& b) {
  bool shared = false;
  for (const uint32_t c : b) shared |= refs[c] > 1;
  if (!shared) return true;
  Nums fresh;  // the whole bookmark is written again: nothing to copy
  if (!take_extra(h, b.size(), &fresh)) return false;
  give_nums(b);
  b = std::move(fresh);
  return true;
}

// File space for an image of session h: while there is none, the least recently used session the current command does
// not use and no running request uses (a conversation on an idle or paused line is cache like a stored one), with an image
// the index on disk does not hold, gives up its images (dropped: state_evicted, if it needs them, a piece only on the
// disk; else it only loses its copies; a conversation on a line is never freed, its pieces in the pool are the truth). kNone when none is left (retired space returns at the next checkpoint).
size_t StoreCore::disk_take(int h, size_t bytes, bool victims) {
  Store& s = store_of();
  for (;;) {
    const size_t off = s.dpool.take_off(bytes);
    if (off != Pool::kNone) {
      s.dbytes += Pool::round(bytes);
      return off;
    }
    if (!victims) return Pool::kNone;  // inside a decoding run no other conversation's images are dropped for space
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (const auto& [k, ss] : s.sess) {
      if (ss.used >= oldest || (!rule(TestRule::kDiskVictimAny) && !rules::disk_victim_ok({k, ss.line, ss.used}, h, s.pinned, s.live))) continue;
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
      free_session(victim);
      s.evicted.push_back(victim);
    } else {
      drop_images(victim);
    }
  }
}

// one reference less on image `off`; at none its file space goes back (and the number holding its content, if any,
// has no copy any more), unless the latest checkpoint's index names it: then at the next checkpoint (retired)
void StoreCore::img_drop(size_t off, size_t bytes) {
  Store& s = store_of();
  const auto it = s.drefs.find(off);
  if (it == s.drefs.end() || !it->second) die("state: image %zu dropped with no reference", off);
  ++s.ver;
  if (s.qimg.count(off)) bytes = sgeom::kQuarterDiskBytes;  // a quarter's image is its own size whatever the caller names
  if (--it->second) return;
  s.drefs.erase(it);
  s.qimg.erase(off);
  if (const auto n = s.dnum.find(off); n != s.dnum.end()) {
    const uint32_t c = n->second;
    s.ndisk[c] = Pool::kNone, s.dnum.erase(n);
    if (g.imager && calloc && calloc->is_ghost(c)) calloc->unghost(c);  // its image is gone: nothing can revive it
  }
  if (g.imager) ghost_group_dissolve(off, UINT32_MAX);
  if (s.named.count(off)) s.retired.emplace_back(off, bytes);
  else space_free(off, bytes);
}

// file space back: the part inside the tier's present size (a smaller size leaves what lies past it unused)
void StoreCore::space_free(size_t off, size_t bytes) {
  Store& s = store_of();
  s.dbytes -= Pool::round(bytes);
  if (off < s.dpool.size) s.dpool.give_off(off, std::min(Pool::round(bytes), s.dpool.size - off));
}

// every image of session x dropped (its pieces in the pool stay; one only on the disk is lost: callers free x then)
void StoreCore::drop_images(int h) {
  Store::Sess& x = store_of().sess.at(h);
  for (auto& d : x.dl)
    for (size_t& o : d)
      if (o != Pool::kNone) img_drop(o, chunk_image()), o = Pool::kNone;
  for (const auto& sn : x.dsnaps) img_drop(sn.second, snap_ibytes());
  x.dsnaps.clear();
  if (x.dend != Pool::kNone) img_drop(x.dend, end_ibytes()), x.dend = Pool::kNone;
}

// session x's cache entries out of the pool and the disk tier (its lists left empty)
void StoreCore::free_lists(int h) {
  Store::Sess& x = store_of().sess.at(h);
  for (int k = 0; k < kKvSlots; ++k) {
    for (const Ref c : x.lists[k]) release(c);
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (const size_t o : x.dl[k])
      if (o != Pool::kNone) img_drop(o, chunk_image());
    x.lists[k].clear(), x.dl[k].clear();
  }
}

size_t StoreCore::entry_image(const Store::Sess& x, int k, size_t i) const {
  if (i < x.dl[k].size() && x.dl[k][i] != Pool::kNone) return x.dl[k][i];
  if (i < x.lists[k].size() && x.lists[k][i].is_slot() && store_s) {
    const auto d = store_s->slot_disk.find(x.lists[k][i].slot_of());
    if (d != store_s->slot_disk.end()) return d->second;
  }
  return Pool::kNone;
}

// A slot's quarter image: the quarter and its trailer in 12 grains, written now when the slot has none and the file has room (the
// split's write-behind; an eviction or a flush writes it when it is missing). A slot is immutable, so its image is its content for as
// long as the slot lives: the slot holds one reference to it, and the image goes with the slot's last holder (release_slot).
void StoreCore::slot_image(int h, int k, size_t i, uint32_t slot) {
  Store& s = store_of();
  if (!g.split || !s.has_disk || s.slot_disk.count(slot)) return;
  const size_t off = disk_take(h, sgeom::kQuarterDiskBytes);
  if (off == Pool::kNone) return;
  s.wseq = dev.slot_out(slot, off, h, k, (int)i, s.slot_written.count(slot) ? s.slot_written.at(slot) : (int)sgeom::kChunkEntries);
  s.slot_disk[slot] = off, s.drefs[off] = 1, s.qimg.insert(off);
  ++s.ver, ++tier.spills, tier.spill_bytes += sgeom::kQuarterDiskBytes;
}

// number c's image (written now unless its content has one already, a shared number's); kNone without room
size_t StoreCore::img_write(int h, uint32_t c) {
  Store& s = store_of();
  if (!s.has_disk) return Pool::kNone;
  ++s.ver;
  if (s.ndisk[c] != Pool::kNone) {
#if !defined(PICO_OLD_RULES) && !defined(PICO_OLD_IMAGE_LINK)
    // A number adopts the image it links to only while the image is still held and names this number: the link of a number whose
    // image was dropped (or went to another holder's number) is cut, and the number gets an image of its own.
    const size_t off = s.ndisk[c];
    const auto n = s.dnum.find(off);
    if (!s.drefs.count(off) || n == s.dnum.end() || n->second != c) {
      if (n != s.dnum.end() && n->second == c) s.dnum.erase(n);
      s.ndisk[c] = Pool::kNone;
      ++stale_cuts;
      if (rank == 0) std::printf("state: session %d: the image link of number %u is not proven: cut\n", h, c);
    } else
#endif
    {
      ++s.drefs[s.ndisk[c]];
      return s.ndisk[c];
    }
  }
  const size_t bytes = chunk_image(), off = disk_take(h, bytes);
  if (off == Pool::kNone) return off;
  s.wseq = dev.image_out(c, off);
  s.ndisk[c] = off, s.dnum[off] = c, s.drefs[off] = 1;
  ++tier.spills, tier.spill_bytes += bytes;
  return off;
}

// a bookmark's image (its bytes through the gather buffer); kNone without room
size_t StoreCore::bk_write(int h, const Nums& b, size_t bytes) {
  Store& s = store_of();
  if (!s.has_disk) return Pool::kNone;
  const size_t off = disk_take(h, bytes);
  if (off == Pool::kNone) return off;
  s.wseq = dev.bk_out(b, bytes, off);
  s.drefs[off] = 1;
  ++s.ver;
  ++tier.spills, tier.spill_bytes += bytes;
  return off;
}

// Session h's full entries (every entry below the saved length) and its window bookmarks below it that have no image
// get one now (a turn's save: write-behind, so that eviction later gives their numbers back without writing).
void StoreCore::write_sealed(int h) {
  Store& s = store_of();
  if (!s.has_disk || g.split || g.imager) return;  // with --kv-split cache a save writes no image: the split's command is the write-behind; the imager's is the writer for every mode
  Store::Sess& x = s.sess.at(h);
  bool any = false;
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    const int full = std::min((int)x.lists[k].size(), (x.saved / (k < 3 ? 2 : 1)) >> attn::kChunkShift);
    for (int i = 0; i < full; ++i)
      if (x.lists[k][i].is_number() && x.dl[k][i] == Pool::kNone) {
        x.dl[k][i] = img_write(h, x.lists[k][i].num());
        any = any || x.dl[k][i] != Pool::kNone;
      }
  }
  for (const auto& [pos, b] : x.snaps)
    if (!b.empty() && snap_usable(pos, x.saved) && !x.dsnaps.count(pos)) {
      const size_t o = bk_write(h, b, snap_bytes());
      if (o != Pool::kNone) x.dsnaps[pos] = o, any = true;
    }
  if (any) dev.disk_flush();
}

// Session x is cut to positions [0, pos): every cache entry that starts at pos or later leaves the pool and the disk
// tier (so no entry a line's row reads is left out of the pool), and so do bookmarks no restart from it can use.
void StoreCore::cut_at(int h, int pos) {
  Store::Sess& x = store_of().sess.at(h);
  x.saved = std::min(x.saved, pos);
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    while (!x.lists[k].empty() && ent_pos(k, (int)x.lists[k].size() - 1) >= pos) {
      release(x.lists[k].back());
      if (x.dl[k].back() != Pool::kNone) img_drop(x.dl[k].back(), chunk_image());
      x.lists[k].pop_back(), x.dl[k].pop_back();
    }
  }
  for (auto i = x.snaps.begin(); i != x.snaps.end();) {
    if (snap_usable(i->first, x.saved) || rule(TestRule::kCutKeepsBookmarks)) {
      ++i;
      continue;
    }
    give_nums(i->second);
    qsnap_drop(x, i->first);
    if (const auto d = x.dsnaps.find(i->first); d != x.dsnaps.end()) img_drop(d->second, snap_ibytes()), x.dsnaps.erase(d);
    i = x.snaps.erase(i);
  }
  if (x.end_pos >= 0 && !end_usable(x.end_pos, x.saved)) {
    give_nums(x.end);
    qend_drop(x);
    if (x.dend != Pool::kNone) img_drop(x.dend, end_ibytes()), x.dend = Pool::kNone;
    x.end_pos = -1;
  }
}

// One piece of session v leaves the pool, its last first: the end bookmark, then window bookmarks from the latest, then cache
// entries from the latest position down. A piece with an image gives its number back at once; one without is written first; with
// no room on the disk the piece is dropped and v shrinks to below it. False when v has nothing in the pool. Under the imager
// eviction takes clean pieces only; `allow_plain` also images a plain piece on the image stream and waits for it.
bool StoreCore::evict_piece(int v, int h, bool allow_plain) {
  store_trace::Scope sc(store_trace::Kind::evict_piece, this);
  sc.arg("victim", v).arg("h", h);
  Store& s = store_of();
  Store::Sess& x = s.sess.at(v);
  const bool im = g.imager && !rule(TestRule::kEvictPlain);
  const auto plain_ok = [&](bool clean) { return !im || clean || allow_plain; };
  if (!x.end.empty() && plain_ok(x.dend != Pool::kNone)) {
    STORE_CORE_COUNT(piece_end);
    if (x.dend == Pool::kNone) {
      if (im) {
        if (!image_now(v, Store::kBkEnd, x.end_pos, &x.end)) x.end_pos = -1;
      } else {
        x.dend = bk_write(h, x.end, end_bytes());
      }
    }
    if (x.dend == Pool::kNone) x.end_pos = -1;
    give_nums(x.end);
    if (s.has_disk && !im) dev.disk_flush();
    return sc.ret(true);
  }
  if (!x.qend.empty()) {  // its slots go back; the whole image the split wrote stays (none: no room on the disk then, and it is dropped)
    qend_drop(x);
    if (x.dend == Pool::kNone) x.end_pos = -1;
    return sc.ret(true);
  }
  for (auto i = x.qsnaps.rbegin(); i != x.qsnaps.rend(); ++i) {
    const int pos = i->first;
    qsnap_drop(x, pos);
    if (!x.dsnaps.count(pos)) x.snaps.erase(pos);
    return sc.ret(true);
  }
  for (auto i = x.snaps.rbegin(); i != x.snaps.rend(); ++i) {
    if (i->second.empty()) continue;
    auto d = x.dsnaps.find(i->first);
    if (!plain_ok(d != x.dsnaps.end())) continue;
    if (d == x.dsnaps.end()) {
      if (im) {
        if (image_now(v, Store::kBkWindow, i->first, &i->second)) d = x.dsnaps.find(i->first);
      } else {
        const size_t o = bk_write(h, i->second, snap_bytes());
        if (o != Pool::kNone) d = x.dsnaps.emplace(i->first, o).first;
      }
    }
    if (d == x.dsnaps.end()) {  // no room: the bookmark is gone
      STORE_CORE_COUNT(piece_snap_dropped);
      const int pos = i->first;
      give_nums(x.snaps.at(pos));
      x.snaps.erase(pos);
    } else {
      STORE_CORE_COUNT(piece_snap_written);
      give_nums(i->second);
    }
    if (s.has_disk && !im) dev.disk_flush();
    return sc.ret(true);
  }
  int bk = -1, bi = -1, bpos = -1;  // the cache entry in the pool that starts latest (under the imager: the latest clean one)
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (int i = (int)x.lists[k].size() - 1; i >= 0; --i)
      if (x.lists[k][i] && plain_ok(x.lists[k][i].is_slot() || x.dl[k][i] != Pool::kNone)) {
        if (ent_pos(k, i) > bpos) bk = k, bi = i, bpos = ent_pos(k, i);
        break;
      }
  }
  if (bk < 0) return STORE_CORE_COUNT(piece_none), sc.ret(false);
  x.dl[bk].resize(x.lists[bk].size(), Pool::kNone);
  if (x.lists[bk][bi].is_slot()) {  // a quarter goes to the disk tier as its own image (written when it has none); the entry then names it
    const uint32_t slot = x.lists[bk][bi].slot_of();
    slot_image(v, bk, (size_t)bi, slot);
    const auto d = s.slot_disk.find(slot);
    if (d == s.slot_disk.end()) {  // no room on the disk (or no tier): the piece is dropped and v shrinks to below it
      STORE_CORE_COUNT(piece_cut);
      cut_at(v, bpos);
    } else {
      STORE_CORE_COUNT(piece_entry_written);
      x.dl[bk][bi] = d->second, ++s.drefs[d->second];  // the entry holds the image too (the slot's own reference goes with its last holder)
      release(x.lists[bk][bi]);
      x.lists[bk][bi] = kOnSsd;
    }
    if (s.has_disk) dev.disk_flush();
    return sc.ret(true);
  }
  if (x.dl[bk][bi] == Pool::kNone) {
    if (im) {
      const int w = written_entries(x, bk, (size_t)bi);
      if (w > 0 && image_now(v, bk, bi, nullptr)) {}  // the image is the entry's at the settle inside
    } else {
      x.dl[bk][bi] = img_write(h, x.lists[bk][bi].num());
    }
  }
  if (x.dl[bk][bi] == Pool::kNone) {
    STORE_CORE_COUNT(piece_cut);
    cut_at(v, bpos);  // no room on the disk: v keeps what lies below this entry
  } else {
    STORE_CORE_COUNT(piece_entry_written);
    release(x.lists[bk][bi]);
    x.lists[bk][bi] = kOnSsd;
  }
  if (s.has_disk && !im) dev.disk_flush();
  return sc.ret(true);
}

// Session h's pieces in the pool that have no image get one: its cache entries holding saved positions (the tail one too),
// its bookmarks (an idle conversation, rank 0's server after --ssd-flush-s: an eviction or a planned stop later writes
// nothing). A copy of the tail goes stale where the conversation goes on (own_tail clears it).
void StoreCore::flush(int h) {
  store_trace::Scope sc(store_trace::Kind::flush, this);
  sc.arg("h", h);
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (!s.has_disk || it == s.sess.end() || g.imager) return;  // the imager is the write-behind: an idle flush writes nothing
  Store::Sess& x = it->second;
  bool any = false;
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (int i = 0; i < (int)x.lists[k].size() && ent_pos(k, i) < x.saved; ++i) {
      if (x.lists[k][i].is_slot() && !s.slot_disk.count(x.lists[k][i].slot_of())) {
        slot_image(h, k, (size_t)i, x.lists[k][i].slot_of());
        any = any || s.slot_disk.count(x.lists[k][i].slot_of());
      }
      if (x.lists[k][i].is_number() && x.dl[k][i] == Pool::kNone) {
        x.dl[k][i] = img_write(h, x.lists[k][i].num());
        any = any || x.dl[k][i] != Pool::kNone;
      }
    }
  }
  for (const auto& [pos, b] : x.snaps)
    if (!b.empty() && !x.dsnaps.count(pos)) {
      const size_t o = bk_write(h, b, snap_bytes());
      if (o != Pool::kNone) x.dsnaps[pos] = o, any = true;
    }
  if (!x.end.empty() && x.dend == Pool::kNone) {
    x.dend = bk_write(h, x.end, end_bytes());
    any = any || x.dend != Pool::kNone;
  }
  if (any) dev.disk_flush();
}

// The disk tier's size while serving (every rank alike). Smaller: images at or past the new end are given up first (a session that
// needs one, a piece only on the disk, is dropped; otherwise only the copy), then the file is cut. Larger: the file is made that
// long first, as far as every rank's free space (less the start's margin) allows, the smallest offer taken. False if unchanged.
bool StoreCore::disk_resize(size_t bytes) {
  store_trace::Scope sc(store_trace::Kind::disk_resize, this);
  sc.arg_u("bytes", bytes);
  Store& s = store_of();
  if (!s.has_disk) return false;
  bytes = bytes / kPoolGrain * kPoolGrain;
  if (const size_t have = dev.disk_size(); bytes > have) {
    size_t room = dev.disk_room() / kPoolGrain;
    if (g.has_ring) {  // every rank reaches this in the same command: all grow by the same amount
      room = (size_t)dev.agree_min((int)std::min<size_t>(room, 0x7fffffff), kSiteDiskRoom);
      store_trace::answer("agree_min", (int64_t)room);
    }
    const size_t target = have + std::min(bytes - have, room * kPoolGrain);
    if (target > have && dev.disk_grow(target) && rank == 0)
      std::printf("serve: the disk tier grew from %.1f to %.1f GiB\n", have / 1073741824.0, dev.disk_size() / 1073741824.0);
    if (target < bytes && rank == 0)
      std::printf("serve: the disk tier could grow only to %.1f GiB: free space less 32 GiB\n", dev.disk_size() / 1073741824.0);
  }
  bytes = std::min(bytes, dev.disk_size());
  const size_t cur = s.dpool.size;
  if (bytes == cur || !bytes) return false;
  if (bytes < cur) {
    const auto past = [&](size_t o, size_t n) {
      if (o != Pool::kNone && s.qimg.count(o)) n = sgeom::kQuarterDiskBytes;
      return o != Pool::kNone && o + Pool::round(n) > bytes;
    };
    for (auto d = s.slot_disk.begin(); d != s.slot_disk.end();) {  // a slot's image the smaller file does not hold: the slot has none (it is written again)
      if (!past(d->second, sgeom::kQuarterDiskBytes)) {
        ++d;
        continue;
      }
      const size_t off = d->second;
      d = s.slot_disk.erase(d);
      img_drop(off, sgeom::kQuarterDiskBytes);
    }
    for (Store::ImgJob& j : s.jobs)  // an image queued for a place the smaller file does not hold: void (its space goes with the file)
      if (!j.dead && j.off != Pool::kNone && j.off + Pool::round(j.bytes) > bytes) {
        j.dead = true;
        for (const uint32_t b : j.blocks)
          if (const auto f = s.job_of.find(b); f != s.job_of.end() && f->second == j.id) s.job_of.erase(f);
      }
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
        if (!past(d->second, snap_ibytes())) {
          ++d;
          continue;
        }
        if (x.snaps.count(d->first) && x.snaps.at(d->first).empty() && !x.qsnaps.count(d->first)) needs = true, ++d;
        else img_drop(d->second, snap_ibytes()), d = x.dsnaps.erase(d);
      }
      if (past(x.dend, end_ibytes())) {
        if (x.end.empty() && x.qend.empty()) needs = true;
        else img_drop(x.dend, end_ibytes()), x.dend = Pool::kNone;
      }
      if (needs) drop.push_back(h);
    }
    for (const int h : drop) {
      if (rank == 0) std::printf("state: session %d dropped: the disk tier shrinks past its pieces\n", h);
      free_session(h);
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
    for (const auto& [off, n] : s.retired) {  // a retired image over the old end still holds its space there, a straddler's tail too
      const size_t lo = std::max(off, cur), hi = std::min(off + Pool::round(n), bytes);
      if (lo < hi && !s.dpool.take_at(lo, hi - lo)) die("state: a retired image at %zu is not free", off);
    }
  }
  if (rank == 0)
    std::printf("state: the disk tier is now %.1f GiB (was %.1f; %.2f GiB in images)\n", bytes / 1073741824.0, cur / 1073741824.0,
                s.dbytes / 1073741824.0);
  if (bytes < cur) {  // the smaller tier's file is cut too: the disk's space is given back, not held until the next start
    const size_t was = dev.disk_size();
    if (was > bytes && dev.disk_shrink(bytes) && rank == 0)
      std::printf("state: the disk tier's file is now %.1f GiB (was %.1f)\n", dev.disk_size() / 1073741824.0, was / 1073741824.0);
  }
  return true;
}

// Session h out of the pool: every piece to the disk tier (a planned stop, the server's spill), or freed without one.
void StoreCore::spill(int h) {
  store_trace::Scope sc(store_trace::Kind::spill, this);
  sc.arg("h", h);
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  if (it->second.line >= 0) die("state: session %d spilled while unit %d holds it", h, it->second.line);
  // a split conversation's slots go to the disk tier as their quarter images (evict_piece); nothing is joined for it
  if (!s.has_disk) {
    free_session(h);
    s.evicted.push_back(h);
    return;
  }
  const size_t n0 = it->second.in_pool();
  if (g.imager) image_all(h);  // a planned stop's or the server's spill: what has no image goes out first, and the wait is this command's own
  while (evict_piece(h, h, g.imager)) {
  }
  if (rank == 0)
    std::printf("state: session %d spilled to disk (%zu chunk numbers given back, restart point %d; %.2f of %.2f GiB of the "
                "file in use)\n", h, n0, restart_of(it->second.saved, it->second.end_pos, it->second.snaps),
                s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0);
}

// Session h's pieces on the disk tier back into the pool (after the writes queued so far), each checked against its
// checksum: cache entries from the first position, then the end bookmark, then window bookmarks from the latest; the
// images stay. When the pool has no room for a piece, h shrinks to below it; so it does for a piece that comes back different (a
// session an earlier run kept makes the engine stop instead). A command with `upto` or `J` is one unit of a return: it brings the
// pieces below `upto`, at most J and at most its count, and leaves the rest on the disk (a used-up count stops it like J: nothing
// is cut; only a spent count, kSpent, cuts).
void StoreCore::unspill(int h, size_t cap, int upto, int J) {
  store_trace::Scope sc(store_trace::Kind::unspill, this);
  sc.arg("h", h).arg_u("cap", cap);
  if (upto != INT32_MAX) sc.arg("upto", upto);  // the record names upto only when it is given
  if (J > 0) sc.arg("J", J);
  if (rule(TestRule::kUnspillUnbounded)) upto = INT32_MAX, J = 0;
  sc.result_fn([this, h]() -> int64_t {  // the restart point the session has when it ends (-1: no such session)
    if (!store_s) return -1;
    const auto i = store_s->sess.find(h);
    return i == store_s->sess.end() ? -1 : restart_of(i->second.saved, i->second.end_pos, i->second.snaps);
  });
  Store& s = store_of();
  // a request's read-back never joins: the quarters on the disk tier come back through the exchange, the slots in the pool are the
  // server's kStJoin units (the guard in use_blocks is for an operation that reaches them anyway)
  const auto it = s.sess.find(h);
  // under a host ring no rank leaves a read-back before the collective its partners meet. A rank whose session is absent or not out sends kNotOut in the
  // same all-gather the others send their first bad position in; all four kNotOut: return early; mixed: every rank stops naming who is out.
  const bool ring_agree = g.has_ring && g.imager && !rule(TestRule::kUnspillReturnsAlone);
  constexpr int kNotOut = -2;
  const auto mixed = [&](const int all[4]) {
    std::string out, not_out;
    for (int r = 0; r < 4; ++r) {
      std::string& to = all[r] == kNotOut ? not_out : out;
      to += (to.empty() ? "" : ",") + std::to_string(r);
    }
    die("state: session %d: out on ranks %s, not on ranks %s: the ranks are not in the same state", h, out.c_str(), not_out.c_str());
  };
  if (it == s.sess.end() || !it->second.out()) {
    if (ring_agree) {
      int all[4];
      dev.agree_all(kNotOut, kSiteUnspill, all);
      for (int r = 0; r < 4; ++r)
        if (all[r] != kNotOut) mixed(all);
    }
    return;
  }
  // the inputs the loop decides from, read before anything is taken: the lists below `upto` only on the SSD, how many of them are live ghosts, the ghost count, the free numbers
  uint64_t inputs[4] = {};
  if (ring_agree) {
    for (int k = 0; k < kKvSlots; ++k)
      for (size_t i = 0; i < it->second.lists[k].size(); ++i) {
        if (it->second.lists[k][i] || ent_pos(k, (int)i) >= upto) continue;
        ++inputs[0];
        const size_t off = i < it->second.dl[k].size() ? it->second.dl[k][i] : Pool::kNone;
        const auto d = off == Pool::kNone ? s.dnum.end() : s.dnum.find(off);
        if (d != s.dnum.end() && link_proven(d->second, off) && calloc->is_ghost(d->second)) ++inputs[1];
      }
    inputs[2] = calloc->ghosts(), inputs[3] = calloc->free();
  }
  bool inputs_agreed = false;
  const auto agree_inputs = [&]() {  // before the first piece is read: no rank waits in a device exchange for partners that are not coming
    if (!ring_agree || inputs_agreed) return;
    inputs_agreed = true;
    uint64_t all[16];
    dev.agree_words(inputs, kSiteUnspill, all);
    for (int r = 1; r < 4; ++r)
      if (std::memcmp(all + 4 * r, all, 4 * sizeof(uint64_t)) != 0) {
        std::string v;
        for (int q = 0; q < 4; ++q)
          v += " rank " + std::to_string(q) + " [only on the SSD " + std::to_string(all[4 * q]) + ", live ghosts " + std::to_string(all[4 * q + 1]) + ", ghosts " +
               std::to_string(all[4 * q + 2]) + ", free " + std::to_string(all[4 * q + 3]) + "]";
        die("state: session %d: the ranks' read-back inputs differ:%s", h, v.c_str());
      }
  };
  if (rule(TestRule::kUnspillJoins) && it->second.split()) join_whole(h, INT32_MAX);  // the wrong rule: the guard joins first
  Store::Sess& x = it->second;
  x.used = ++s.clock;
  const double t0 = wall_s();
  ++split_counts.unspill_cmds;  // a read-back command that got past the early returns (the four-rank rows count these)
  bool writes_in = false;  // the imager: the writes are waited for before the first piece that is really read, not before a command that only revives
  const auto wait_writes = [&]() {
    agree_inputs();
    if (!writes_in && s.has_disk) dev.disk_settle(s.wseq);
    writes_in = true;
  };
  if (!g.imager) wait_writes();
  const double t1 = wall_s();
  struct Back {
    int k, i, snap;  // a cache entry (k, i), or a bookmark (k -1: snap = its position, -1 the end)
    size_t off;
  };
  std::vector<Back> back;
  size_t got = 0, taken = 0;
  const auto over_cap = [&](size_t n) { return cap && !rule(TestRule::kUnspillIgnoresCap) && (cap == kSpent || taken + n > cap); };
  size_t pieces = 0;      // the pieces this command handled (lists, then bookmarks): at most J
  bool j_full = false;    // it stopped at J or at its count: nothing is cut or dropped, the next command goes on
  // a command of a return (J > 0) ends at J pieces or when its count is used up (a spent count, kSpent, is the cut's own: it goes on to cut)
  const auto at_j = [&]() {
    return J > 0 && (pieces >= (size_t)J || (!rule(TestRule::kUnspillCountDrops) && cap && cap != kSpent && taken >= cap));
  };
  s.pinned.push_back(h);
  const int restart0 = restart_of(x.saved, x.end_pos, x.snaps);
  // cache entries in position order: the beginning is what every restart needs
  std::vector<std::pair<int, std::pair<int, int>>> ents;
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (int i = 0; i < (int)x.lists[k].size(); ++i)
      if (!x.lists[k][i] && ent_pos(k, i) < upto) ents.push_back({ent_pos(k, i), {k, i}});  // a fork's tail above its point stays where it is
  }
  std::sort(ents.begin(), ents.end());
  int cut = INT32_MAX;
  // A quarter image is read into the batch to exchange, eight pieces at a time: the other three ranks' quarters come over the
  // ring, the piece is put together in its number and checked against the sums its trailer holds. A whole image is read into its number directly.
  struct QPiece {
    int k, i, pos;
    uint32_t num;
    quarter::Trailer t;
    bool ok;  // the trailer is one and is this rank's, this list's and this piece's
  };
  std::vector<QPiece> qb;
  int qbad_pos = INT32_MAX;
  size_t qbad = 0, qread = 0, qbad_trailer = 0, qbad_sums = 0;  // the quarter images read, bad by their trailer, bad by their sums after the exchange
  const auto flush_q = [&]() {
    if (qb.empty()) return;
    std::vector<uint32_t> nums;
    std::vector<quarter::Trailer> ts;
    for (const QPiece& p : qb) nums.push_back(p.num), ts.push_back(p.t);
    dev.exchange_staged((int)qb.size());
    dev.unquarter(nums);
    std::vector<uint8_t> flags;
    dev.compare_sums(nums, ts, &flags);
    qread += qb.size();
    for (size_t j = 0; j < qb.size(); ++j) {
      qbad_trailer += !qb[j].ok, qbad_sums += flags[j] != 0;
      if (flags[j] || !qb[j].ok) ++qbad, qbad_pos = std::min(qbad_pos, qb[j].pos);
    }
    split_counts.exchanged += qb.size();
    split_counts.join_bytes += qb.size() * 3 * sgeom::kQuarterBytes;  // the other three ranks' quarters
    qb.clear();
  };
  for (const auto& [pos, ki] : ents) {
    const auto [k, i] = ki;
    if (at_j()) {
      j_full = true;
      break;
    }
    if (g.imager && x.dl[k][i] != Pool::kNone && !over_cap(1) && revive_piece(h, k, i, x.dl[k][i])) {
      taken += 1;  // a ghost taken back: no read, no exchange, no device call, and it costs no J (it costs nothing)
      continue;
    }
    Nums one;
    if (x.dl[k][i] == Pool::kNone || over_cap(1) || !take_restore(h, 1, &one)) {
      cut = pos;
      break;
    }
    taken += 1, ++pieces;
    const size_t off = x.dl[k][i];
    if (s.qimg.count(off)) {
      if (g.imager) wait_writes();
      ++split_counts.returns_read;
      quarter::Trailer t;
      const bool ok = dev.quarter_read(off, (int)qb.size(), &t) &&
                      ((t.rank == (uint32_t)rank && t.list == (uint32_t)k && t.piece == (uint32_t)i) || rule(TestRule::kQuarterTrailerUnchecked));
      x.lists[k][i] = Ref(one[0]);
#ifdef PICO_OLD_IMAGE_LINK
      s.ndisk[one[0]] = off, s.dnum[off] = one[0];  // without the proven-link check: any image a number was read from is that number's whole image
#else
      if (g.imager && !s.dnum.count(off) && one[0] < s.ndisk.size() && s.ndisk[one[0]] == Pool::kNone)
        s.ndisk[one[0]] = off, s.dnum[off] = one[0];  // the block holds exactly the image's bytes (the sums below check it): marking it again is bookkeeping
#endif
      qb.push_back({k, i, pos, one[0], t, ok});
      back.push_back({k, i, 0, off});
      got += sgeom::kQuarterDiskBytes;
      if (qb.size() == (size_t)model::rules::kJoinBatch) flush_q();
      continue;
    }
    flush_q();
    dev.image_in(off, one[0]);
    x.lists[k][i] = Ref(one[0]);
    s.ndisk[one[0]] = off, s.dnum[off] = one[0];
    back.push_back({k, i, 0, off});
    got += chunk_image();
  }
  flush_q();
  if (cut != INT32_MAX && !rule(TestRule::kUnspillNoCut)) cut_at(h, cut);
  const auto read_bk = [&](Nums& b, size_t off, size_t bytes, int snap) {
    if (g.imager) {  // a bookmark's image is this rank's slice: its ghost group taken back whole, else the four slices read and exchanged
      const size_t n = bk_nums(bytes);
      if (const auto gi = s.bk_ghost.find(off); gi != s.bk_ghost.end() && !over_cap(n)) {
        bool all = true;
        for (const uint32_t c : gi->second) all = all && calloc->is_ghost(c);
        if (all && (rule(TestRule::kJoinTakeUnpaid) || join_take_ok(h, n))) {
          const Nums group = gi->second;
          for (const uint32_t c : group) calloc->revive(c), refs[c] = 1, s.ghost_bk.erase(c);
          s.bk_ghost.erase(off);
          b = group;
          taken += n, ++split_counts.revives;
          held_peak = std::max(held_peak, (size_t)calloc->held());
          return true;
        }
      }
      if (over_cap(n) || !take_restore(h, n, &b)) return false;
      taken += n, ++pieces;
      wait_writes();
      ++split_counts.returns_read;
      const int kind = snap < 0 ? Store::kBkEnd : Store::kBkWindow, units = snap < 0 ? 3 : 1, at = snap < 0 ? x.end_pos : snap;
      quarter::Trailer t;
      bool ok = dev.bookmark_read(off, bytes, 0, &t) && t.rank == (uint32_t)rank && t.list == (uint32_t)kind && t.piece == (uint32_t)at;
      dev.exchange_staged(units);
      const bool put = dev.bookmark_put(b, bytes, 0, t);
      split_counts.exchanged += 1, split_counts.join_bytes += 3 * ((bytes / 4 + 15) / 16 * 16);
      ok = ok && put;
      if (!ok) {
        ++qbad, qbad_trailer += 0, qbad_sums += 1;
        qbad_pos = std::min(qbad_pos, snap >= 0 ? snap - sgeom::kSnapRows : x.end_pos - 1);
      }
      ++qread;
      back.push_back({-1, 0, snap, off});
      got += bytes;
      return true;
    }
    if (over_cap(bk_nums(bytes)) || !take_restore(h, bk_nums(bytes), &b)) return false;
    taken += bk_nums(bytes), ++pieces;
    dev.bk_in(off, b, bytes);
    back.push_back({-1, 0, snap, off});
    got += bytes;
    return true;
  };
  // the bookmarks come after the lists: a command that stopped at J has nothing to say about them yet (what a cut leaves stays as it is)
  const bool bk_turn = !j_full;
  // a bookmark that is slots is in the pool, not on the disk, whatever images it has: the join brings it
  const bool rb_split = rule(TestRule::kUnspillReadsSplitBookmarks);
  if (bk_turn && x.end.empty() && (x.qend.empty() || rb_split) && x.dend != Pool::kNone && x.end_pos >= 0 && x.end_pos <= upto) {
    if (at_j()) j_full = true;
    else if (!read_bk(x.end, x.dend, end_bytes(), -1)) img_drop(x.dend, end_ibytes()), x.dend = Pool::kNone, x.end_pos = -1;
  }
  std::vector<int> outs;  // window bookmarks only on the disk, the latest first
  for (auto i = x.snaps.rbegin(); i != x.snaps.rend(); ++i)
    if (i->second.empty() && (!x.qsnaps.count(i->first) || rb_split) && i->first <= upto && bk_turn) outs.push_back(i->first);
  for (const int pos : outs) {
    if (at_j()) {
      j_full = true;
      break;
    }
    const auto d = x.dsnaps.find(pos);
    if (d != x.dsnaps.end() && read_bk(x.snaps.at(pos), d->second, snap_bytes(), pos)) continue;
    if (d != x.dsnaps.end()) img_drop(d->second, snap_ibytes()), x.dsnaps.erase(d);
    x.snaps.erase(pos);  // no room for it: the bookmark is gone
  }
  s.pinned.pop_back();
  if (!g.imager || !back.empty()) dev.sync();  // a return that only revived ghosts made no device call: nothing to wait for
  const double t2 = wall_s();
  {
    const uint64_t w = (uint64_t)((t2 - t0) * 1e9);
    // the counters count what was read. Tokens = the positions the session holds x (list pieces this command read) / (the session's list pieces below
    // its saved length): over a return's commands they add to its positions when every piece was read and to 0 when none was (a revived ghost reads nothing). A return counts once,
    // at its first command that read a piece, not once a command.
    size_t lists_read = 0, lists_total = 0;
    for (const Back& b : back) lists_read += b.k >= 0;
    for (int k = 0; k < kKvSlots; ++k)
      for (int i = 0; i < (int)x.lists[k].size(); ++i) lists_total += ent_pos(k, i) < x.saved;
    if (lists_read && !s.restoring.count(h)) ++tier.restores;
    auto& rs = s.restoring[h];  // {pieces read so far in this return, tokens counted so far}: the sum over the commands is exact, not a sum of roundings
    rs.first += lists_read;
    if (lists_total) {
      const uint64_t cum = (uint64_t)std::max(x.saved, 0) * std::min(rs.first, lists_total) / lists_total;
      tier.restored_pos += cum - rs.second, rs.second = cum;
    }
    if (rs.first == 0) s.restoring.erase(h);
    tier.restore_bytes += got;
    if (unspill_left(h, upto) == 0) s.restoring.erase(h);
    tier.wait_ns += w, tier.wait_max_ns = std::max(tier.wait_max_ns, w);
  }
  int bad_pos = INT32_MAX;
  size_t bad = 0;
  uint64_t fp = 0;  // the checksums' xor, logged: a check that compared nothing would show as 0
  for (const Back& b : back) {
    fp ^= dev.sum(b.off, true);
    if (dev.sum(b.off, false) == dev.sum(b.off, true) || rule(TestRule::kUnspillIgnoresSum)) continue;
    ++bad;
    bad_pos = std::min(bad_pos, b.k >= 0 ? ent_pos(b.k, b.i) : b.snap >= 0 ? b.snap - attn::kSnapRows : x.end_pos - 1);
  }
  bad += qbad, bad_pos = std::min(bad_pos, qbad_pos);
  if (bad)
    std::printf("state: session %d: ERROR: %zu of %zu pieces read back from the disk differ from what was written (rank %d)\n",
                h, bad, back.size(), rank);
  if (bad && qread)
    std::printf("state: session %d: quarter images: %zu read, %zu bad by their trailer, %zu bad by their sums after the exchange (rank %d)\n", h,
                qread, qbad_trailer, qbad_sums, rank);
  if (g.has_ring) {  // every rank cuts at the smallest first bad position (each reaches this in the same command): the
                    // session stays the same on every rank, and rank 0's server restarts below the cut
    if (ring_agree) {
      int all[4];
      dev.agree_all(bad_pos, kSiteUnspill, all);
      bad_pos = INT32_MAX;
      for (int r = 0; r < 4; ++r) {
        if (all[r] == kNotOut) mixed(all);
        bad_pos = std::min(bad_pos, all[r]);
      }
    } else {
      bad_pos = dev.agree_min(bad_pos, kSiteUnspill);
    }
    store_trace::answer("agree_min", bad_pos);
  }
#ifndef PICO_OLD_RULES
  if (bad_pos != INT32_MAX) ++sum_cuts;
#endif
  if (bad_pos != INT32_MAX) cut_at(h, std::max(0, bad_pos));
  split_counts.unspill_taken += taken;
  if (g.imager && has_clean(x)) s.to_mark.insert(h);  // pieces read back or revived keep their images: marked again when the conversation is idle off its line
  if (unspill_left(h, INT32_MAX) == 0) x.kept = false;  // a piece of a kept session still unread is checked as an earlier run's: fatal if it differs
  if (rank == 0)
    std::printf("state: session %d back from disk: %.1f MiB in %.1f ms (%.2f GB/s; %.1f ms waiting for writes; %zu pieces "
                "checked, %016llx); restart point %d of the %d it had\n", h, got / 1048576.0, 1e3 * (t2 - t0),
                got / std::max(1e-9, t2 - t1) / 1e9, 1e3 * (t1 - t0), back.size(), (unsigned long long)fp,
                restart_of(x.saved, x.end_pos, x.snaps), restart0);
}

int StoreCore::unspill_left(int h, int upto) const {
  if (!store_s) return 0;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return 0;
  const Store::Sess& x = it->second;
  int n = 0;
  for (int k = 0; k < kKvSlots; ++k)
    for (int i = 0; i < (int)x.lists[k].size(); ++i) n += !x.lists[k][i] && ent_pos(k, i) < upto;
  const bool rb_split = rule(TestRule::kUnspillReadsSplitBookmarks);
  n += x.end.empty() && (x.qend.empty() || rb_split) && x.dend != Pool::kNone && x.end_pos >= 0 && x.end_pos <= upto;
  for (const auto& sn : x.snaps) n += sn.second.empty() && (!x.qsnaps.count(sn.first) || rb_split) && sn.first <= upto;
  return n;
}

// ---- the imager -------------------------------------------------------------------------------------------
//
// A piece is plain (no image), clean (its entry names an image that went out and its block still holds those bytes) or a ghost (the entry
// says "on the SSD", its block is free in the allocator and linked to the image). image() only queues quarter images on the image
// stream; settle() makes the entries clean once the command's work has ended on every rank, then marks the conversations that are on no
// line and run no request (bookkeeping, no device call). A cut of a block voids its queued job and cuts a ghost's link.

bool StoreCore::has_clean(const Store::Sess& x) const {
  for (int k = 0; k < kKvSlots; ++k)
    for (size_t i = 0; i < x.lists[k].size(); ++i)
      if (x.lists[k][i].is_number() && i < x.dl[k].size() && x.dl[k][i] != Pool::kNone) return true;
  for (const auto& sn : x.snaps)
    if (!sn.second.empty() && x.dsnaps.count(sn.first)) return true;
  return !x.end.empty() && x.dend != Pool::kNone;
}

bool StoreCore::queue_piece(int h, int k, int i, int written, int mode) {
  Store& s = store_of();
  Store::Sess& x = s.sess.at(h);
  const uint32_t c = x.lists[k][i].num();
  const size_t bytes = sgeom::kQuarterDiskBytes;
  // The ride or idle command agreed its budget (in units) with the smallest staging room of the four ranks: no rank asks its own staging again, which would
  // make one rank refuse what three queue. A rank that finds none says so and lets the write wait: the same decision everywhere.
  if (mode != kQNow && !dev.image_room(bytes)) std::printf("state: BUG: staging had no room for an agreed image (rank %d)\n", rank);
  const size_t off = disk_take(h, bytes, mode != kQRide);
  if (off == Pool::kNone) {
    ++split_counts.split_deferred;
    return false;
  }
  s.wseq = dev.image_quarter(c, off, h, k, i, written);
  Store::ImgJob j;
  j.id = ++s.job_id, j.seq = s.img_seq + 1, j.h = h, j.k = k, j.piece = i, j.blocks = {c}, j.off = off, j.bytes = bytes, j.written = written;
  j.holders.push_back({h, k, i});
  s.job_of[c] = j.id;
  s.jobs.push_back(std::move(j));
  return true;
}

bool StoreCore::queue_bookmark(int h, int k, int pos, const Nums& b, int mode) {
  Store& s = store_of();
  const bool end = k == Store::kBkEnd;
  const size_t bytes = end ? end_ibytes() : snap_ibytes(), whole = end ? end_bytes() : snap_bytes();
  if (quarter::bk_trailer_off(whole) + quarter::kTrailerUsed > bytes) {  // the slice and the trailer do not fit the block: no image (a bookmark of a layout this build does not size for)
    ++split_counts.split_deferred;
    return false;
  }
  const size_t off = disk_take(h, bytes, mode != kQRide);
  if (off == Pool::kNone) {
    ++split_counts.split_deferred;
    return false;
  }
  s.wseq = dev.image_bookmark(b, whole, off, h, k, pos);
  Store::ImgJob j;
  j.id = ++s.job_id, j.seq = s.img_seq + 1, j.h = h, j.k = k, j.piece = pos, j.blocks = b, j.off = off, j.bytes = bytes, j.written = (int)whole;
  for (const uint32_t c : b) s.job_of[c] = j.id;
  s.jobs.push_back(std::move(j));
  return true;
}

int StoreCore::image_impl(int h, int budget, const std::vector<int>& live_, int mode) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (!g.imager || it == s.sess.end() || h <= 0 || budget <= 0 || !s.has_disk || in(live_, h)) return 0;
  Store::Sess& x = it->second;
  const uint64_t seq = s.img_seq + 1;
  int queued = 0, used = 0;  // jobs queued, and units spent: a piece and a window bookmark's slice are one unit, the end bookmark's slice two (agreed by the four ranks)
  bool stop = false;
  for (int k = 0; k < kKvSlots && !stop; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (size_t i = 0; i < x.lists[k].size() && used < budget; ++i) {
      const Ref r = x.lists[k][i];
      if (!r.is_number() || x.dl[k][i] != Pool::kNone) continue;
      const int w = written_entries(x, k, i);
      if (w <= 0) continue;  // a number past the saved positions holds nothing
      const uint32_t c = r.num();
      if (c < s.ndisk.size() && link_proven(c, s.ndisk[c])) {  // another holder's image of the same bytes: this entry names it too
        const size_t off = s.ndisk[c];
        ++s.drefs[off];
        x.dl[k][i] = off;
        s.to_mark.insert(h);
        continue;
      }
      if (s.job_of.count(c)) continue;  // an image of it is queued already
      if (!queue_piece(h, k, (int)i, w, mode)) {
        stop = true;
        break;
      }
      ++queued, ++used;
    }
  }
  for (auto& [pos, b] : x.snaps) {
    if (stop || used >= budget) break;
    if (b.empty() || x.dsnaps.count(pos) || !snap_usable(pos, x.saved) || s.job_of.count(b[0])) continue;
    if (!queue_bookmark(h, Store::kBkWindow, pos, b, mode)) {
      stop = true;
      break;
    }
    ++queued, ++used;
  }
  if (!stop && used + 2 <= budget && !x.end.empty() && x.dend == Pool::kNone && x.end_pos >= 0 && end_usable(x.end_pos, x.saved) &&
      !s.job_of.count(x.end[0])) {
    if (queue_bookmark(h, Store::kBkEnd, x.end_pos, x.end, mode)) ++queued, used += 2;
  }
  if (queued) {
    s.img_seq = seq;
    dev.image_end(seq);
    ++split_counts.image_cmds;
  }
  return queued;
}

int StoreCore::image(int h, int budget, const std::vector<int>& live_) {
  store_trace::Scope sc(store_trace::Kind::image, this);
  sc.arg("h", h).arg("budget", budget).arg_v("live", live_);
  return sc.ret(image_impl(h, budget, live_, live_.empty() ? kQIdle : kQRide));
}

// every job of a command at or below `upto` is made true or void
void StoreCore::settle_jobs(uint64_t upto, const std::vector<int>& live_) {
  Store& s = store_of();
  while (!s.jobs.empty() && s.jobs.front().seq <= upto) {
    Store::ImgJob j = std::move(s.jobs.front());
    s.jobs.pop_front();
    bool valid = !j.dead;
    for (const uint32_t b : j.blocks) {
      const auto f = s.job_of.find(b);
      if (f == s.job_of.end() || f->second != j.id) {
        if (!rule(TestRule::kSettleUnchecked)) valid = false;
      } else {
        s.job_of.erase(f);
      }
    }
    // a partial tail piece of a conversation a request is now running on may have been appended to while the image was made: void
    if (valid && j.k >= 0 && j.written < sgeom::kChunkEntries && in(live_, j.h)) valid = false;
    const auto void_job = [&]() {
      ++split_counts.image_voids, ++split_counts.split_deferred;
      if (!j.dead) space_free(j.off, j.bytes);  // the write may still go out: the writer writes in flush order, a later write of the space ends with the later bytes
    };
    if (!valid) {
      void_job();
      continue;
    }
    if (j.k >= 0) {  // a cache entry's piece: every holder that still names the block takes the image
      const uint32_t c = j.blocks[0];
      bool any = false;
      for (const auto& hd : j.holders) {
        const auto it = s.sess.find(hd[0]);
        if (it == s.sess.end()) continue;
        Store::Sess& x = it->second;
        x.dl[hd[1]].resize(x.lists[hd[1]].size(), Pool::kNone);
        if ((size_t)hd[2] >= x.lists[hd[1]].size() || x.lists[hd[1]][hd[2]] != Ref(c) || x.dl[hd[1]][hd[2]] != Pool::kNone) continue;
        x.dl[hd[1]][hd[2]] = j.off;
        ++s.drefs[j.off];
        s.to_mark.insert(hd[0]);
        any = true;
      }
      if (!any) {
        void_job();
        continue;
      }
      s.qimg.insert(j.off);
      if (c >= s.ndisk.size()) s.ndisk.resize(calloc->numbers(), Pool::kNone);
      s.ndisk[c] = j.off, s.dnum[j.off] = c;
    } else {  // a bookmark: its entry names the image when the bookmark is still the same numbers
      const auto it = s.sess.find(j.h);
      bool ok = it != s.sess.end();
      if (ok && j.k == Store::kBkWindow) {
        Store::Sess& x = it->second;
        const auto b = x.snaps.find(j.piece);
        ok = b != x.snaps.end() && b->second == j.blocks && !x.dsnaps.count(j.piece);
        if (ok) x.dsnaps[j.piece] = j.off;
      } else if (ok) {
        Store::Sess& x = it->second;
        ok = x.end == j.blocks && x.end_pos == j.piece && x.dend == Pool::kNone;
        if (ok) x.dend = j.off;
      }
      if (!ok) {
        void_job();
        continue;
      }
      s.drefs[j.off] = 1;
      s.to_mark.insert(j.h);
    }
    ++s.ver;
    ++tier.spills, tier.spill_bytes += j.bytes;
    ++split_counts.imaged;
  }
}

// conversation h's clean pieces become ghosts: the entry says "on the SSD", the block is free and stamped and stays linked to its image.
// A conversation on a line, or one a request runs on, is left as it is (it is marked when its line is let go).
int StoreCore::mark_conv(int h, const std::vector<int>& live_) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || h <= 0) {
    s.to_mark.erase(h);
    return 0;
  }
  Store::Sess& x = it->second;
  if ((x.line >= 0 && !rule(TestRule::kMarkIgnoresLine)) || in(live_, h) || in(s.pinned, h)) return 0;
  const auto idle = [&](int hh) {
    const auto f = s.sess.find(hh);
    return f != s.sess.end() && f->second.line < 0 && !in(live_, hh) && !in(s.pinned, hh);
  };
  int n = 0;
  const auto give = [&](uint32_t c) {  // the block back to the allocator with its bytes: a ghost (poison fills what is given back: a plain number)
    refs[c] = 0;
    if (g.poison) {
      dev.fill(c, 0xFD);
      if (c < s.ndisk.size() && s.ndisk[c] != Pool::kNone) {
        const auto d = s.dnum.find(s.ndisk[c]);
        if (d != s.dnum.end() && d->second == c) s.dnum.erase(d);
        s.ndisk[c] = Pool::kNone;
      }
      calloc->give(c);
    } else {
      calloc->give_ghost(c, ++s.ghost_seq);
    }
  };
  for (int k = 0; k < kKvSlots; ++k) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    for (size_t i = 0; i < x.lists[k].size(); ++i) {
      const Ref r = x.lists[k][i];
      const size_t off = x.dl[k][i];
      if (!r.is_number() || off == Pool::kNone || !s.drefs.count(off)) continue;
      const uint32_t c = r.num();
      if (!link_proven(c, off)) {
        if (s.dnum.count(off) || c >= s.ndisk.size() || s.ndisk[c] != Pool::kNone) continue;  // the image is another number's: not this block's content
        s.ndisk[c] = off, s.dnum[off] = c;  // a piece read back from its image: the block holds exactly those bytes
      }
      if (refs[c] > 1 && !live_.empty()) continue;  // a ride inside a run marks no piece several entries name: it stays clean, as ready as a ghost, and an idle
                                                     // command marks it (the walk over every stored entry would cost a ride milliseconds)
      bool all = true;
      if (refs[c] > 1) {  // every entry naming the block is marked together, or none
        for (auto& [hh, ss] : s.sess)
          for (int kk = 0; kk < kKvSlots; ++kk)
            for (size_t ii = 0; ii < ss.lists[kk].size(); ++ii)
              if (ss.lists[kk][ii] == r) {
                ss.dl[kk].resize(ss.lists[kk].size(), Pool::kNone);
                if (ss.dl[kk][ii] == Pool::kNone) ++s.drefs[off], ss.dl[kk][ii] = off;  // the same bytes: it names the image too
                all = all && idle(hh) && ss.dl[kk][ii] == off;
              }
      }
      if (!all) continue;
      if (refs[c] > 1) {
        for (auto& [hh, ss] : s.sess)
          for (int kk = 0; kk < kKvSlots; ++kk)
            for (size_t ii = 0; ii < ss.lists[kk].size(); ++ii)
              if (ss.lists[kk][ii] == r) ss.lists[kk][ii] = kOnSsd;
      } else {
        x.lists[k][i] = kOnSsd;
      }
      give(c);
      ++n, ++split_counts.split_pieces;
    }
  }
  const auto group = [&](Nums& b, size_t off) {
    if (b.empty() || off == Pool::kNone || !s.drefs.count(off)) return;
    for (const uint32_t c : b)
      if (refs[c] != 1) return;  // a shared bookmark stays whole
    const Nums blocks = b;
    if (!g.poison) {
      s.bk_ghost[off] = blocks;
      for (const uint32_t c : blocks) s.ghost_bk[c] = off;
    }
    for (const uint32_t c : blocks) give(c);
    b.clear();
    ++n, ++split_counts.split_pieces, ++split_counts.marked_bk;
  };
  for (auto& [pos, b] : x.snaps)
    if (const auto d = x.dsnaps.find(pos); d != x.dsnaps.end()) group(b, d->second);
  group(x.end, x.dend);
  if (!has_clean(x)) s.to_mark.erase(h);
  return n;
}

int StoreCore::settle(uint64_t upto, const std::vector<int>& live_) {
  store_trace::Scope sc(store_trace::Kind::settle, this);
  sc.arg_u("upto", upto).arg_v("live", live_);
  if (!g.imager) return sc.ret(0);
  Store& s = store_of();
  settle_jobs(upto, live_);
  int marked = 0;
  const std::vector<int> hs(s.to_mark.begin(), s.to_mark.end());
  if (g.mark)
    for (const int h : hs) marked += mark_conv(h, live_);
  else s.to_mark.clear();  // off: nothing is marked, a conversation holds its blocks until a take evicts them clean
  return sc.ret(marked);
}

// a piece or a bookmark of h images now and the image stream is waited for: the count that reserved clean pieces was broken (a take found only
// plain ones), or a spill or a planned stop wants every piece out. Every rank reaches this at the same point in the same command, so each
// waits for its own stream and the settle after it needs no agreement. False: no room for it (the caller's no-room path follows).
bool StoreCore::image_now(int h, int k, int piece, const Nums* b) {
  Store& s = store_of();
  if (!g.imager || !s.has_disk) return false;
  const uint64_t seq = s.img_seq + 1;
  Store::Sess& x = s.sess.at(h);
  bool ok;
  if (k >= 0) {
    x.dl[k].resize(x.lists[k].size(), Pool::kNone);
    const int w = written_entries(x, k, (size_t)piece);
    ok = w > 0 && queue_piece(h, k, piece, w, kQNow);
  } else {
    ok = b && !b->empty() && queue_bookmark(h, k, piece, *b, kQNow);
  }
  if (!ok) return false;
  s.img_seq = seq;
  dev.image_end(seq);
  ++split_counts.image_cmds;
  const double t0 = wall_s();
  dev.image_wait();
  const double ms = 1e3 * (wall_s() - t0);
  ++split_counts.image_waits;
  split_counts.image_wait_ns += (uint64_t)(ms * 1e6);
  if (rank == 0) std::printf("state: a take waited for an image (1 pieces, %.2f ms)\n", ms);
  settle_jobs(seq, {});
  const auto d = k >= 0 ? x.dl[k][(size_t)piece] != Pool::kNone : k == Store::kBkEnd ? x.dend != Pool::kNone : x.dsnaps.count(piece) != 0;
  return d;
}

// every piece of h imaged and settled, waiting for the image stream once at the end
void StoreCore::image_all(int h) {
  Store& s = store_of();
  if (!g.imager || !s.has_disk) return;
  const uint64_t before = s.img_seq;
  image_impl(h, INT32_MAX, {}, kQNow);  // one command: a piece whose job is queued is not queued again
  if (s.img_seq == before) return;
  dev.image_wait();
  settle_jobs(s.img_seq, {});
}

// the ghost c (or another holder's block already revived) is entry (h, k, i)'s block again: no read, no exchange, no device call
bool StoreCore::revive_piece(int h, int k, int i, size_t off) {
  Store& s = store_of();
  if (!g.imager) return false;
  const auto d = s.dnum.find(off);
  if (d == s.dnum.end() || !link_proven(d->second, off)) return false;
  const uint32_t c = d->second;
  Store::Sess& x = s.sess.at(h);
  if (calloc->is_ghost(c)) {
    if (!rule(TestRule::kJoinTakeUnpaid) && !join_take_ok(h, 1)) return false;  // taking it back leaves a reservation uncovered: the read-back's own take decides
    calloc->revive(c);
    refs[c] = 1;
    ++split_counts.revives;
    x.lists[k][i] = Ref(c);
    held_peak = std::max(held_peak, (size_t)calloc->held());
    return true;
  }
  if (calloc->taken(c) && refs[c] >= 1) {  // another holder (a fork's) revived it: shared
    ++refs[c];
    ++split_counts.shares;
    x.lists[k][i] = Ref(c);
    return true;
  }
  return false;
}

int StoreCore::image_todo(int h) const {
  if (!g.imager || !store_s || !store_s->has_disk) return 0;
  const Store& s = *store_s;
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return 0;
  const Store::Sess& x = it->second;
  int n = 0;
  for (int k = 0; k < kKvSlots; ++k)
    for (size_t i = 0; i < x.lists[k].size(); ++i) {
      const Ref r = x.lists[k][i];
      if (!r.is_number() || (i < x.dl[k].size() && x.dl[k][i] != Pool::kNone) || written_entries(x, k, i) <= 0) continue;
      const uint32_t c = r.num();
      if (s.job_of.count(c) || (c < s.ndisk.size() && s.ndisk[c] != Pool::kNone)) continue;
      ++n;
    }
  for (const auto& [pos, b] : x.snaps)
    n += !b.empty() && !x.dsnaps.count(pos) && snap_usable(pos, x.saved) && !s.job_of.count(b[0]);
  n += !x.end.empty() && x.dend == Pool::kNone && x.end_pos >= 0 && end_usable(x.end_pos, x.saved) && !s.job_of.count(x.end[0]);
  return n;
}

StoreCore::ImageView StoreCore::image_view() const {
  ImageView v;
  if (!store_s) return v;
  const Store& s = *store_s;
  for (const auto& [k, x] : s.sess) {
    if (k <= 0) continue;
    for (int kk = 0; kk < kKvSlots; ++kk)
      for (size_t i = 0; i < x.lists[kk].size(); ++i) v.clean_numbers += x.lists[kk][i].is_number() && i < x.dl[kk].size() && x.dl[kk][i] != Pool::kNone;
    for (const auto& sn : x.snaps)
      if (!sn.second.empty() && x.dsnaps.count(sn.first)) v.clean_numbers += sn.second.size();
    if (!x.end.empty() && x.dend != Pool::kNone) v.clean_numbers += x.end.size();
  }
  v.image_pending = s.jobs.size();
  v.ghost_numbers = calloc ? calloc->ghosts() : 0;
  return v;
}

// ---- the split ---------------------------------------------------------------------------------------------------------------

// Session h's whole pieces become quarter slots: a piece whose every holder is cache and whole (no holder pinned or running; the
// piece a full one in each), at most `budget` of them. The piece's number goes back as a ghost with the same bytes; every holder
// names the slot. A piece whose number is still some slot's content (it came back whole and was not written) reuses that slot:
// one more reference, no copy. Never evicts: with no quarter number to be had the piece stays whole.
int StoreCore::split(int h, int budget) {
  store_trace::Scope sc(store_trace::Kind::split, this);
  sc.arg("h", h).arg("budget", budget);
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || h <= 0 || budget <= 0) return sc.ret(0);
  if (g.imager) return sc.ret(0);  // no slot exists; a conversation is imaged and marked (image, settle)
  const auto busy = [&](int hh) { return (in(s.pinned, hh) || in(s.live, hh)) && !rule(TestRule::kSplitIgnoresPin); };
  if (busy(h)) return sc.ret(0);
  struct At {
    int h, k;
    size_t i;
  };
  std::unordered_map<uint32_t, std::vector<At>> holders;  // every list entry that is a number, by number
  for (const auto& [hh, ss] : s.sess)
    for (int k = 0; k < kKvSlots; ++k)
      for (size_t i = 0; i < ss.lists[k].size(); ++i)
        if (ss.lists[k][i].is_number()) holders[ss.lists[k][i].num()].push_back({hh, k, i});
  std::set<int> lines;
  int done = 0;
  for (int k = 0; k < kKvSlots && done < budget; ++k)
    for (size_t i = 0; i < it->second.lists[k].size() && done < budget; ++i) {
      const Ref r = it->second.lists[k][i];
      if (!r.is_number()) continue;
      // the entries of the piece that hold saved positions: all of them for a full piece, the written count for the last
      const auto written_of = [&](const Store::Sess& ss, int kk, size_t ii) {
        const int ratio = kk < 3 ? 2 : 1, total = (ss.saved + ratio - 1) / ratio;
        return std::min(sgeom::kChunkEntries, total - (int)ii * sgeom::kChunkEntries);
      };
      const int written = written_of(it->second, k, i);
      if (written <= 0) continue;  // a number past the saved positions holds nothing
      if (written < sgeom::kChunkEntries && rule(TestRule::kSplitFullOnly)) continue;
      const uint32_t c = r.num();
      const std::vector<At>& hv = holders[c];
      if (hv.size() != refs[c]) continue;  // named by something that is not a list entry
      bool ok = true;
      for (const At& a : hv) {
        const Store::Sess& ss = s.sess.at(a.h);
        ok = ok && a.h > 0 && !busy(a.h) && written_of(ss, a.k, a.i) == written;
      }
      if (written < sgeom::kChunkEntries && hv.size() != 1) ok = false;  // a partial piece is never shared: it stays whole
      if (!ok) continue;
      uint32_t slot = c < s.twin_s.size() ? s.twin_s[c] : 0;
      const bool reuse = written == sgeom::kChunkEntries && slot && s.twin_c.count(slot) && s.twin_c.at(slot) == c &&
                         s.slots.refs(slot) > 0 && s.slot_written.count(slot) && s.slot_written.at(slot) == written;
      if (reuse) {
        for (size_t j = 0; j < hv.size(); ++j) s.slots.ref(slot);
      } else {
        if (!s.slots.has_room()) {
          uint32_t q = 0;
          if (!take_quarter_number(&q)) {
            ++split_counts.split_deferred;
            continue;
          }
          s.slots.open(q);
        }
        cut_twin(c);  // a link to another slot (a partial piece's, or one written since) is not this slot's
        slot = s.slots.take_slot();
        for (size_t j = 1; j < hv.size(); ++j) s.slots.ref(slot);
        s.slot_written[slot] = written;
        dev.quarter(c, slot, written);
        slot_image(hv[0].h, hv[0].k, hv[0].i, slot);
      }
      for (const At& a : hv) {
        Store::Sess& ss = s.sess.at(a.h);
        ss.dl[a.k].resize(ss.lists[a.k].size(), Pool::kNone);
        if (ss.dl[a.k][a.i] != Pool::kNone) img_drop(ss.dl[a.k][a.i], chunk_image()), ss.dl[a.k][a.i] = Pool::kNone;  // a whole image of a piece that is a quarter now
        ss.lists[a.k][a.i] = Ref::slot(slot);
        if (ss.line >= 0) lines.insert(a.h);
      }
      // the number back to the allocator: a ghost with its bytes, unless poison fills what is given back
      refs[c] = 0;
      if (s.ndisk[c] != Pool::kNone) s.dnum.erase(s.ndisk[c]), s.ndisk[c] = Pool::kNone;
      if (g.poison) {
        dev.fill(c, 0xFD);
        cut_twin(c);
        calloc->give(c);
      } else {
        calloc->give_ghost(c, ++s.ghost_seq);
        s.twin_s[c] = slot, s.twin_c[slot] = c;
      }
      ++done, ++split_counts.split_pieces;
    }
  // the window bookmarks by position, then the end bookmark: a quarter of each, in slots (a window bookmark's one slot, an end
  // bookmark's three); one that is shared (a fork's) stays whole
  Store::Sess& x = it->second;
  const auto take_slots = [&](size_t n, Refs* out) {
    for (size_t j = 0; j < n; ++j) {
      if (!s.slots.has_room()) {
        uint32_t q = 0;
        if (!take_quarter_number(&q)) {
          for (const Ref r : *out) release_slot(r.slot_of());
          out->clear();
          return false;
        }
        s.slots.open(q);
      }
      const uint32_t sl = s.slots.take_slot();
      s.slot_written[sl] = sgeom::kChunkEntries;
      out->push_back(Ref::slot(sl));
    }
    return true;
  };
  const auto sole = [&](const Nums& b) {
    for (const uint32_t c : b)
      if (refs[c] != 1) return false;
    return !b.empty();
  };
  if (!rule(TestRule::kBookmarksWhole)) {
    std::vector<int> at;
    for (const auto& [pos, b] : x.snaps)
      if (sole(b)) at.push_back(pos);
    for (const int pos : at) {
      if (done >= budget) break;
      Refs q;
      if (!take_slots(bk_slots(snap_bytes()), &q)) {
        ++split_counts.split_deferred;
        break;
      }
      Nums& b = x.snaps.at(pos);
      // the split is the bookmark's write-behind: its whole image first (a save writes none in cache mode), so that eviction gives the
      // slots back and the image stays; with no room on the disk it is quartered without one and eviction drops it, as with a piece
      if (s.has_disk && !x.dsnaps.count(pos) && !rule(TestRule::kSplitBookmarkNoImage)) {
        const size_t o = bk_write(h, b, snap_bytes());
        if (o != Pool::kNone) x.dsnaps[pos] = o;
      }
      dev.quarter_bk(b, [&] {
        std::vector<uint32_t> v;
        for (const Ref r : q) v.push_back(r.slot_of());
        return v;
      }());
      give_nums(b);
      x.qsnaps[pos] = std::move(q);
      ++done, ++split_counts.split_pieces;
    }
    if (done < budget && sole(x.end)) {
      Refs q;
      if (!take_slots(bk_slots(end_bytes()), &q)) {
        ++split_counts.split_deferred;
      } else {
        std::vector<uint32_t> v;
        for (const Ref r : q) v.push_back(r.slot_of());
        if (s.has_disk && x.dend == Pool::kNone && !rule(TestRule::kSplitBookmarkNoImage)) x.dend = bk_write(h, x.end, end_bytes());
        dev.quarter_bk(x.end, v);
        give_nums(x.end);
        x.qend = std::move(q);
        ++done, ++split_counts.split_pieces;
      }
    }
  }
  if (g.split && s.has_disk && done) dev.disk_flush();
  for (const int hh : lines) write_rows(s.sess.at(hh).line, &s.sess.at(hh));
  return sc.ret(done);
}

// The return (see store_core.h). Every rank decides alike: the slot map, the twins and the ghosts are replicated, the sums are
// compared on the device and the first bad position agreed.
int StoreCore::join(int h, int upto, size_t cap, int J) {
  store_trace::Scope sc(store_trace::Kind::join, this);
  sc.arg("h", h).arg("upto", upto).arg_u("cap", cap).arg("J", J);
  Store& s = store_of();
  if (g.imager) die("state: join(session %d) reached under the imager: no slot exists there, a return is unspill", h);  // a join that returns early on one rank leaves its partners in the collective
  const auto it = s.sess.find(h);
  const double t_join = wall_s();
  const auto left = [&]() {
    int n = 0;
    for (const Refs& l : it->second.lists)
      for (size_t i = 0; i < l.size(); ++i) n += l[i].is_slot() && ent_pos((int)(&l - it->second.lists), (int)i) < upto;
    for (const auto& q : it->second.qsnaps) n += q.first <= upto;
    n += !it->second.qend.empty() && it->second.end_pos <= upto;
    return n;
  };
  if (it == s.sess.end()) return sc.ret(0);
  Store::Sess& x = it->second;
  std::vector<std::pair<int, std::pair<int, int>>> ents;  // the slot pieces below `upto`, in position order
  for (int k = 0; k < kKvSlots; ++k)
    for (int i = 0; i < (int)x.lists[k].size(); ++i)
      if (x.lists[k][i].is_slot() && ent_pos(k, i) < upto) ents.push_back({ent_pos(k, i), {k, i}});
  if (ents.empty() && x.qsnaps.empty() && x.qend.empty()) return sc.ret(0);
  if (!rule(TestRule::kJoinNoStamp)) x.used = ++s.clock;  // the pieces just joined are not the next victim (as a read-back stamps its conversation)
  std::sort(ents.begin(), ents.end());
  ++split_counts.joins;
  struct Piece {
    int k, i;
    uint32_t slot, number;
    const char* how;  // how it came back, for the line that names a piece that differs
  };
  const auto name_piece = [&](const Piece& p) {
    if (rank != 0) return;
    const int w = s.slot_written.count(p.slot) ? s.slot_written.at(p.slot) : (int)sgeom::kChunkEntries;
    std::printf("state: session %d: list %d piece %d at %d (%s, %d entries) differs from its sums\n", h, p.k, p.i, ent_pos(p.k, p.i), p.how, w);
  };
  std::vector<Piece> sums, xs;  // revived and shared (compared one by one), exchanged (compared as a batch)
  std::vector<uint32_t> release_after;
  int bad_pos = INT32_MAX, handled = 0, stopped = -1;
  size_t taken = 0, bad = 0, total = ents.size();
  const auto flush = [&]() {
    if (xs.empty()) return;
    std::vector<uint32_t> slots, nums;
    for (const Piece& p : xs) slots.push_back(p.slot), nums.push_back(p.number);
    dev.exchange(slots);
    dev.unquarter(nums);
    std::vector<uint8_t> flags;
    dev.compare(slots, nums, &flags);
    for (size_t j = 0; j < xs.size(); ++j)
      if (flags[j]) ++bad, bad_pos = std::min(bad_pos, ent_pos(xs[j].k, xs[j].i)), name_piece(xs[j]);
    split_counts.exchanged += xs.size();
    split_counts.join_bytes += xs.size() * 3 * sgeom::kQuarterBytes;  // the three other ranks' quarters of each piece
    xs.clear();
  };
  s.pinned.push_back(h);
  for (const auto& [pos, ki] : ents) {
    if (handled >= J) break;
    const auto [k, i] = ki;
    const uint32_t sl = x.lists[k][i].slot_of();
    const auto tw = s.twin_c.find(sl);
    const uint32_t c = tw == s.twin_c.end() ? 0 : tw->second;
    const bool linked = c && c < s.twin_s.size() && s.twin_s[c] == sl;
    if (linked && calloc->is_ghost(c)) {  // a ghost is a free number whose bytes are the slot's
      if (cap && taken + 1 > cap) break;
      if (!rule(TestRule::kJoinTakeUnpaid) && !join_take_ok(h, 1)) {  // taking it back leaves a reservation uncovered: the join stops here
        stopped = pos;
        break;
      }
      calloc->revive(c);
      refs[c] = 1;
      ++taken, ++split_counts.revives;
      x.lists[k][i] = Ref(c);
      sums.push_back({k, i, sl, c, "taken back"});
    } else if (linked && calloc->taken(c) && refs[c] >= 1) {  // another holder made it whole again and has not written it
      ++refs[c];
      ++split_counts.shares;
      x.lists[k][i] = Ref(c);
      sums.push_back({k, i, sl, c, "shared"});
    } else {
      if (cap && taken + 1 > cap) break;
      Nums one;
      if (!(rule(TestRule::kJoinTakeUnpaid) ? take_restore(h, 1, &one) : take_join(h, 1, &one))) {
        stopped = pos;
        break;
      }
      ++taken;
      const uint32_t n = one[0];
      s.twin_s[n] = sl, s.twin_c[sl] = n;  // the number will hold exactly the slot's piece: other holders may share it
      x.lists[k][i] = Ref(n);
      xs.push_back({k, i, sl, n, "exchanged"});
    }
    release_after.push_back(sl);
    ++handled;
    if (xs.size() == 8) flush();
  }
  flush();
  // the bookmarks at or below `upto`, the end bookmark first, then the window bookmarks from the latest, while the count lasts
  // (one that does not fit stays split); each is checked on its own and a different one cuts the conversation below it
  if (stopped < 0 && handled < J) {
    const auto bring = [&](Refs& q, size_t bytes, int cut_below, Nums* whole, const char* kind, int at) {
      const size_t n = bk_nums(bytes);
      if (cap && taken + n > cap) return false;
      Nums b;
      if (!(rule(TestRule::kJoinTakeUnpaid) ? take_restore(h, n, &b) : take_join(h, n, &b))) return false;
      std::vector<uint32_t> sl;
      for (const Ref r : q) sl.push_back(r.slot_of());
      taken += n;
      if (!dev.join_bk(b, sl)) {
        ++bad, bad_pos = std::min(bad_pos, cut_below);
        if (rank == 0)
          std::printf("state: session %d: the %s bookmark at %d (exchanged, %zu slots) differs from its sum\n", h, kind, at, sl.size());
        give_nums(b);
        for (const uint32_t t : sl) release_slot(t);
        q.clear();
        return true;
      }
      for (const uint32_t t : sl) release_slot(t);
      q.clear();
      *whole = std::move(b);
      ++handled, ++split_counts.exchanged;
      return true;
    };
    if (!x.qend.empty() && x.end_pos <= upto && handled < J) bring(x.qend, end_bytes(), std::max(0, x.end_pos - 1), &x.end, "end", x.end_pos);
    std::vector<int> ps;
    for (auto i = x.qsnaps.rbegin(); i != x.qsnaps.rend(); ++i)
      if (i->first <= upto) ps.push_back(i->first);
    for (const int pos : ps) {
      if (handled >= J) break;
      bring(x.qsnaps.at(pos), snap_bytes(), std::max(0, pos - sgeom::kSnapRows), &x.snaps[pos], "window", pos);
    }
    for (auto i = x.qsnaps.begin(); i != x.qsnaps.end();)
      i = i->second.empty() ? x.qsnaps.erase(i) : std::next(i);
  }
  s.pinned.pop_back();
  for (const Piece& p : sums)
    if (!dev.sum_exact(p.number, p.slot, s.slot_written.count(p.slot) ? s.slot_written.at(p.slot) : (int)sgeom::kChunkEntries))
      ++bad, bad_pos = std::min(bad_pos, ent_pos(p.k, p.i)), name_piece(p);
  for (const uint32_t sl : release_after) release_slot(sl);
  if (g.has_ring) {  // every rank cuts at the smallest first bad position (each reaches this in the same command)
    bad_pos = dev.agree_min(bad_pos, kSiteJoin);
    store_trace::answer("agree_min", bad_pos);
  }
  if (bad)
    std::printf("state: session %d: ERROR: %zu of %d pieces joined from quarters differ from their sums (rank %d)\n", h, bad, handled, rank);
  if (bad_pos != INT32_MAX) {
    ++split_counts.join_bad;
    cut_at(h, std::max(0, bad_pos));
  }
  if (stopped >= 0) {
    ++split_counts.join_short;
    if (rank == 0) std::printf("state: session %d: join stopped at %d: no number (%d of %zu pieces)\n", h, stopped, handled, total);
  }
  if (x.line >= 0 && handled) write_rows(x.line, &x);
  const uint64_t ns = (uint64_t)((wall_s() - t_join) * 1e9);
  split_counts.join_taken += taken;
  split_counts.join_ns += ns, split_counts.join_max_ns = std::max(split_counts.join_max_ns, ns);
  return sc.ret(left());
}

// Every split piece of session h back to a whole number (no limit); a piece no number was found for, and everything after it, is cut.
void StoreCore::join_whole(int h, int upto) {
  Store::Sess& x = store_of().sess.at(h);
  int before = 0, in = 0, disk = 0;
  list_numbers(h, upto, &in, &disk, &before);
  const size_t bk_before = x.bookmark_slots();
  int bookmarks = !x.qend.empty() && x.end_pos <= upto;  // those join brings back below `upto`, as its own left() counts
  for (const auto& q : x.qsnaps) bookmarks += q.first <= upto;
  if (before == 0 && bookmarks == 0) return;
  join(h, upto, 0, INT32_MAX);
  int after = 0;
  list_numbers(h, upto, &in, &disk, &after);
  // the guard says so whenever it joins anything (the server joins by kStJoin before an operation reaches it), bookmarks included,
  // then where it cut
  const int joined = (before - after) + (int)(bk_before - std::min(bk_before, x.bookmark_slots()));
  if (rank == 0 && joined > 0) std::printf("state: session %d: joined inside an operation: %d pieces\n", h, joined);
  int pos = INT32_MAX;
  for (int k = 0; k < kKvSlots; ++k)
    for (int i = 0; i < (int)x.lists[k].size(); ++i)
      if (x.lists[k][i].is_slot() && ent_pos(k, i) < upto) pos = std::min(pos, ent_pos(k, i));
  if (pos != INT32_MAX) {
    if (rank == 0) std::printf("state: session %d: joined inside an operation: cut at %d\n", h, pos);
    cut_at(h, pos);
  }
  while (!x.qsnaps.empty() && x.qsnaps.begin()->first <= upto) {  // a bookmark no number was found for is gone, as a read-back's is
    const int bp = x.qsnaps.begin()->first;
    qsnap_drop(x, bp);
    if (!x.dsnaps.count(bp)) x.snaps.erase(bp);
  }
  if (!x.qend.empty() && x.end_pos <= upto) {  // only what is below `upto` is an operation's: the rest of the conversation stays as it is
    qend_drop(x);
    if (x.dend == Pool::kNone) x.end_pos = -1;
  }
}

// Compressed caches of positions [from, to) into session h: the line holding h writes into h's own numbers, so what it
// holds up to `to` is h's already.
void StoreCore::save(int h, int from, int to, int unit) {
  store_trace::Scope sc(store_trace::Kind::save, this);
  sc.arg("h", h).arg("from", from).arg("to", to).arg("unit", unit);
  Store::Sess& ss = use_blocks(h);
  if (from > ss.saved) die("state: session %d saves [%d, %d) with %d saved", h, from, to, ss.saved);
  if (to > from && ss.line != unit) die("state: session %d saves [%d, %d) from unit %d, which does not hold it", h, from, to, unit);
  for (int k = 0; k < kKvSlots; ++k)  // the chunk of the last entry saved is the session's
    if (to > from && (int)ss.lists[k].size() <= ((to - 1) / (k < 3 ? 2 : 1) >> attn::kChunkShift))
      die("state: session %d saves up to %d past its kv source %d's numbers", h, to, k);
  if (to > from)  // the entries written in [from, to) hold new content: a copy of one (an idle flush's of a partial tail,
                  // a turn on its line went on after) is stale
    for (int k = 0; k < kKvSlots; ++k) {
      const int r = k < 3 ? 2 : 1, i1 = std::min((int)ss.lists[k].size() - 1, ((to - 1) / r) >> attn::kChunkShift);
      ss.dl[k].resize(ss.lists[k].size(), Pool::kNone);
      for (int i = (from / r) >> attn::kChunkShift; i <= i1; ++i) {
        entry_dirty(h, ss.lists[k][i], &ss.dl[k][i]);
      }
    }
  ss.saved = std::max(ss.saved, to);
  write_sealed(h);  // write-behind: the entries this save filled, and the bookmarks below it, get their images
}

// Session h's compressed caches of positions [0, to) for the current slot's session. h itself is cut to `to` (the numbers
// from the one holding `to` on made its own: own_tail). A fork's base: the slot's session gives its lists back and takes
// the base's numbers wholly below `to`, shared by count, and a copy of the entries below `to` of the one holding it.
void StoreCore::load(int h, int to, int unit, int slot) {
  store_trace::Scope sc(store_trace::Kind::load, this);
  sc.arg("h", h).arg("to", to).arg("unit", unit).arg("slot", slot);
  sc.result_fn([this, unit] { return trace_held(*this, unit); });
  if (!store_of().sess.count(h)) die("state: session %d loads with nothing stored", h);
  Store::Sess& src = use_blocks(h, to, true);  // a fork leaves its base's tail alone: only the pieces below its fork point are joined and read back
  if (to > src.saved) die("state: session %d loads %d positions of %d", h, to, src.saved);
  const int hd = line_h[unit];
  if (hd == h) {
    src.saved = std::min(src.saved, to);
    if (!own_tail(h, to)) die("state: no room to copy session %d's shared numbers from %d on", h, to);
    return;
  }
  if (hd == kNoLine || hd < 0) die("state: a fork of %d into slot %d, which holds no session", h, slot);
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
      dev.copy_entries(one[0], src.lists[k][full].num(), part);
      dst.lists[k].push_back(Ref(one[0]));
      copied += part;
    }
  }
  store_of().pinned.pop_back();
  write_rows(unit, &dst);
  mapped_upto[slot] = -1;  // the next step grows the lists past `to`
  if (rank == 0)
    std::printf("state: session %d forks %d at %d: %zu numbers shared, %zu entries copied\n", hd, h, to, shared, copied);
}

// A window bookmark's numbers (the caller fills them, bk_copy), or nullptr when the pool has no room for it.
const Nums* StoreCore::snap_buf(int h, int pos) {
  store_trace::Scope sc(store_trace::Kind::snap_buf, this);
  sc.arg("h", h).arg("pos", pos);
  Store& s = store_of();
  Store::Sess& ss = use_blocks(h);
  const auto it = ss.snaps.find(pos);
  if (it != ss.snaps.end()) {  // written again: its image goes stale, and a shared one takes fresh numbers first
    for (const uint32_t c : it->second) s.job_of.erase(c);  // a queued image of the old bytes is void
    if (const auto d = ss.dsnaps.find(pos); d != ss.dsnaps.end()) img_drop(d->second, snap_ibytes()), ss.dsnaps.erase(d);
    return sc.ret_list(own_bk(h, it->second) ? &it->second : nullptr);
  }
  Nums b;
  if (!take_extra(h, bk_nums(snap_bytes()), &b)) {
    if (rank == 0) std::printf("state: no room for a bookmark of session %d at %d\n", h, pos);
    return sc.ret_list<Nums>(nullptr);
  }
  return sc.ret_list(&ss.snaps.emplace(pos, std::move(b)).first->second);
}

// The bookmark at pos, for the draft module's windows, which the prefill took (null: not taken, or only on the disk tier).
const Nums* StoreCore::snap_peek(int h, int pos) {
  store_trace::Scope sc(store_trace::Kind::snap_peek, this);
  sc.arg("h", h).arg("pos", pos);
  const Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return nullptr;
  const auto b = it->second.snaps.find(pos);
  return b != it->second.snaps.end() && !b->second.empty() ? &b->second : nullptr;
}

const Nums& StoreCore::snap_get(int h, int pos) {
  store_trace::Scope sc(store_trace::Kind::snap_get, this);
  sc.arg("h", h).arg("pos", pos);
  if (!store_of().sess.count(h)) die("state: session %d restores with nothing stored", h);
  Store::Sess& ss = use_blocks(h, pos);  // the bookmark it reads and what is below it, nothing above
  if (!ss.snaps.count(pos)) die("state: session %d has no snapshot at %d", h, pos);
  return ss.snaps.at(pos);
}

// The end bookmark's numbers, for the caller to fill; null when none was taken (no room, or shared and none of its own).
const Nums* StoreCore::end_take(int h, int pos) {
  store_trace::Scope sc(store_trace::Kind::end_take, this);
  sc.arg("h", h).arg("pos", pos);
  Store::Sess& ss = use_blocks(h, 0);  // it reads nothing of the conversation: nothing is joined for it
  for (const uint32_t c : ss.end) store_of().job_of.erase(c);  // written again: a queued image of the old bytes is void
  if (!ss.qend.empty()) qend_drop(ss);  // the split end bookmark it replaces
  if (ss.dend != Pool::kNone) img_drop(ss.dend, end_ibytes()), ss.dend = Pool::kNone;  // written again
  if (ss.end.empty()) {
    if (!take_extra(h, bk_nums(end_bytes()), &ss.end)) {
      ss.end_pos = -1;
      return nullptr;
    }
  } else if (!own_bk(h, ss.end)) {  // shared and no room for its own: no end bookmark
    give_nums(ss.end);
    ss.end_pos = -1;
    return nullptr;
  }
  ss.end_pos = pos;
  return &ss.end;
}

const Nums& StoreCore::end_get(int h) {
  store_trace::Scope sc(store_trace::Kind::end_get, this);
  sc.arg("h", h);
  if (!store_of().sess.count(h)) die("state: session %d restores its end with nothing stored", h);
  Store::Sess& ss = use_blocks(h);
  if (ss.end.empty()) die("state: session %d has no end snapshot", h);
  return ss.end;
}

// What session h holds from position `from` on no longer belongs to its history: bookmarks above it, the end bookmark
// if it is above, and saved positions from it on (the numbers past them go back unless a line holds the session: the
// line writes over them as it goes on).
void StoreCore::drop(int h, int from) {
  store_trace::Scope sc(store_trace::Kind::drop, this);
  sc.arg("h", h).arg("from", from);
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  for (auto i = ss.snaps.upper_bound(from); i != ss.snaps.end();) {
    give_nums(i->second);
    qsnap_drop(ss, i->first);
    if (const auto d = ss.dsnaps.find(i->first); d != ss.dsnaps.end()) img_drop(d->second, snap_ibytes()), ss.dsnaps.erase(d);
    i = ss.snaps.erase(i);
  }
  if (ss.end_pos > from) {
    give_nums(ss.end);
    qend_drop(ss);
    if (ss.dend != Pool::kNone) img_drop(ss.dend, end_ibytes()), ss.dend = Pool::kNone;
    ss.end_pos = -1;
  }
  ss.saved = std::min(ss.saved, from);
  if (ss.line >= 0 && !own_tail(h, from)) die("state: no room to copy session %d's shared numbers from %d on", h, from);
  if (ss.line < 0)
    for (int k = 0; k < kKvSlots; ++k) {
      ss.dl[k].resize(ss.lists[k].size(), Pool::kNone);
      while ((int)ss.lists[k].size() > nums_for_pos(k, ss.saved)) {
        release(ss.lists[k].back());
        if (ss.dl[k].back() != Pool::kNone) img_drop(ss.dl[k].back(), chunk_image());
        ss.lists[k].pop_back(), ss.dl[k].pop_back();
      }
    }
}

void StoreCore::free_session(int h) {
  store_trace::Scope sc(store_trace::Kind::free, this);
  sc.arg("h", h);
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  if (ss.line >= 0) {  // a line still holds it (a release frees it first): the line leaves it
    const int u = ss.line;
    write_rows(u, nullptr);
    line_h[u] = kNoLine;
    for (int t = 0; t < nslots; ++t)
      if (unit_of[t] == u) mapped_upto[t] = -1, resv[t] = 0, resv_img[t] = 0;
  }
  free_lists(h);
  for (auto& sn : ss.snaps) give_nums(sn.second);
  give_nums(ss.end);
  while (!ss.qsnaps.empty()) qsnap_drop(ss, ss.qsnaps.begin()->first);
  qend_drop(ss);
  drop_images(h);  // its writes may still be queued: harmless, the file space is taken again only after them
  s.sess.erase(it);
}

// src's bookmarks up to upto are dst's too: the same numbers, shared by count (a fork's; immutable until one is written again).
void StoreCore::copy_snaps(int src, int dst, int upto) {
  store_trace::Scope sc(store_trace::Kind::copy_snaps, this);
  sc.arg("src", src).arg("dst", dst).arg("upto", upto);
  Store& s = store_of();
  const auto it = s.sess.find(src);
  if (it == s.sess.end() || src == dst) return;
  // out() is "anything of the session is out" (its tail too); under the imager a read-back that has nothing at or below `upto` to take is not made: no
  // collective, no line, no re-queue. Every rank holds the same state, so every rank skips together.
  if (it->second.out() && (!g.imager || rule(TestRule::kUnspillNoOpReturn) || unspill_left(src, upto) > 0)) {  // the bookmarks it shares, at or below `upto`: nothing of the base's tail comes back for it
    s.pinned.push_back(dst);
    unspill(src, 0, upto);
    s.pinned.pop_back();
  }
  // a fork shares the bookmarks that are whole and joins nothing: a split bookmark stays the base's own
  it->second.used = ++s.clock;
  s.pinned.push_back(src);
  Store::Sess& d = use_blocks(dst);
  s.pinned.pop_back();
  const size_t sb = snap_ibytes();
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

void StoreCore::drop_snap(int h, int pos) {
  store_trace::Scope sc(store_trace::Kind::drop_snap, this);
  sc.arg("h", h).arg("pos", pos);
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  if (const auto d = ss.dsnaps.find(pos); d != ss.dsnaps.end()) img_drop(d->second, snap_ibytes()), ss.dsnaps.erase(d);
  const auto i = ss.snaps.find(pos);
  if (i == ss.snaps.end()) return;
  give_nums(i->second);
  qsnap_drop(ss, pos);
  ss.snaps.erase(i);
}
// ---- the store -------------------------------------------------------------------------------------------------------

StoreCore::StoreCore(const sgeom::StoreGeom& geom, StoreDev& device, chunks::Alloc* alloc)
    : g(geom), dev(device), calloc(alloc), rank(geom.rank), nslots(geom.nslots), nunits(geom.nunits), list_len(geom.list_len),
      max_ctx(geom.max_ctx) {
  std::copy(geom.unit_of, geom.unit_of + kMaxSlots, unit_of);
  std::copy(geom.slot_ctx, geom.slot_ctx + kMaxSlots, slot_ctx_);
  for (int s = 0; s < nslots; ++s) mapped_upto[s] = -1;
  for (int u = 0; u < kMaxUnits; ++u) line_h[u] = kNoLine;
}

Store& StoreCore::store_of() {
  if (!store_s) store_s = std::make_unique<Store>();
  return *store_s;
}

// session h (made if new) for an operation that needs its blocks: back from the disk tier first
Store::Sess& StoreCore::use_blocks(int h, int upto, bool bound_readback) {
  Store& s = store_of();
  if (const auto it = s.sess.find(h); it != s.sess.end()) {
    if (it->second.split()) join_whole(h, upto);
    const int bound = bound_readback && !rule(TestRule::kForkReadsTail) ? upto : INT32_MAX;
    if (it->second.out() && (!g.imager || rule(TestRule::kUnspillNoOpReturn) || unspill_left(h, bound) > 0)) unspill(h, 0, bound);  // nothing to take at or below `bound`: no return
  }
  return s.use(h);
}

// ---- lines: a slot's unit holds a session (its rows are the session's lists) ------------------------------------

// slot s's session: steps with no session named (tools, tests) get one of the unit's own, freed at release
Store::Sess& StoreCore::line_sess(int s) {
  Store& st_ = store_of();
  const int u = unit_of[s];
  if (line_h[u] == kNoLine) {
    line_h[u] = -1 - u;
    st_.sess[line_h[u]].line = u;
  }
  return st_.sess.at(line_h[u]);
}

// unit u's rows from session x's lists (the guard past their ends); the device waits first, so no queued copy of the
// mirror reads an entry being rewritten
void StoreCore::write_rows(int u, const Store::Sess* x) {
  if (x)
    for (int k = 0; k < kKvSlots; ++k)
      if (x->lists[k].size() > (size_t)list_len)
        die("model: kv source %d's list of %zu numbers is longer than a row (%d)", k, x->lists[k].size(), list_len);
  Nums rows[kKvSlots];  // the device's rows hold numbers; an entry only on the disk tier is the guard
  if (x)
    for (int k = 0; k < kKvSlots; ++k)
      for (const Ref c : x->lists[k]) rows[k].push_back(c.is_number() ? c.num() : 0);  // a slot is the guard too
  dev.rows_set(u, x ? rows : nullptr);
}

void StoreCore::pin(const std::vector<int>& h, const std::vector<int>& live_, const std::vector<int>& last_) {
  store_trace::Scope sc(store_trace::Kind::pin, this);
  sc.arg_v("pin", h).arg_v("live", live_).arg_v("last", last_);
  Store& s = store_of();
  s.pinned = h;
  s.live = live_;
  s.last = last_;
}

void StoreCore::set_factor(int h, int factor_) {
  store_trace::Scope sc(store_trace::Kind::set_factor, this);
  sc.arg("h", h).arg("factor", factor_);
  if (factor_ != 16 && factor_ != 32 && factor_ != 64) die("state: session %d tagged with factor %d", h, factor_);
  if (!store_s) return;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return;
  if (it->second.factor != factor_ && (it->second.saved > 0 || it->second.end_pos >= 0 || !it->second.snaps.empty()))
    die("state: session %d holds state of factor %d and is tagged %d (a session keeps its factor for life)", h,
        it->second.factor, factor_);
  it->second.factor = factor_;
}

std::vector<int> StoreCore::take_evicted() {
  std::vector<int> v;
  if (store_s) v.swap(store_s->evicted);
  store_trace::evicted(v);
  return v;
}

// key's numbers back to the pool (key 0: every flight's). Reads of them are queued on the model's stream ahead of any
// later write, so the numbers can be taken again at once.
void StoreCore::img_free(int key) {
  store_trace::Scope sc(store_trace::Kind::img_free, this);
  sc.arg("key", key);
  if (!store_s || rule(TestRule::kImgFreeLeaks)) return;
  Store& s = *store_s;
  const auto give = [&](Nums& v) { give_nums(v); };
  if (key == 0) s.img.release_all(give);
  else s.img.release(key, give);
}

const imgrows::Ledger::Rows* StoreCore::img_find(int key) const { return store_s ? store_s->img.find(key) : nullptr; }
size_t StoreCore::img_numbers() const { return store_s ? store_s->img.numbers() : 0; }

// ---- the pool's cut and growth, and the numbers a window needs vacated ---------------------------------------------

// Makes the numbers at or above `from` free: the conversations that hold them go, a piece at a time (to the disk tier, or whole
// without one), the command's evict-last sessions after the others, then the least recently used. 0: done; 1: a line, a pinned or
// a running conversation holds one, or an image flight's rows do (they are read by address and never move: the caller waits for
// the flight), or no piece is left to give. Pieces already evicted stay evicted.
int StoreCore::vacate(uint32_t from) {
  store_trace::Scope sc(store_trace::Kind::vacate, this);
  sc.arg_u("from", from);
  Store& s = store_of();
  if (image_numbers_from(s.img, from)) return sc.ret(1);
  for (;;) {
    if (!calloc->first_taken_from(from)) return sc.ret(0);
    rules::VictimPick pick;
    bool busy = false;
    for (const auto& [k, x] : s.sess) {
      if (!holds_from(x, from)) continue;
      if (rules::holder_busy({k, x.line, x.used}, s.pinned, s.live, s.last)) busy = true;
      else pick.offer(k, x.used, in(s.last, k));
    }
    const int victim = pick.victim;
    if (busy || victim < 0) return sc.ret(1);
    if (!s.has_disk) {
      free_session(victim);
      s.evicted.push_back(victim);
    } else if (!evict_piece(victim, 0)) {
      return sc.ret(1);
    }
  }
}

bool StoreCore::make_room(size_t removed) {
  store_trace::Scope sc(store_trace::Kind::make_room, this);
  sc.arg_u("removed", removed);
  const int64_t unheld = rules::unheld_sum(nslots, [&](int s) { return resv[s]; }, [&](int s) { return held(s); }, [&](int s) { return resv_img[s]; });
  while (!rules::window_room((int64_t)calloc->free(), (int64_t)removed, unheld))
    if (!evict_step(0)) return sc.ret(false);
  return sc.ret(true);
}

void StoreCore::alloc_shrink(size_t top) {
  store_trace::Scope sc(store_trace::Kind::alloc_shrink, this);
  sc.arg_u("top", top);
  if (!calloc->first_taken_from((uint32_t)top))
    for (const uint32_t gh : calloc->ghosts_from((uint32_t)top)) cut_twin(gh);  // a ghost above the cut leaves with its number
  if (!calloc->shrink((uint32_t)top)) die("model: the pool's numbers %zu.. are not all free after vacating them", top);
}

void StoreCore::alloc_grow(size_t nums) {
  store_trace::Scope sc(store_trace::Kind::alloc_grow, this);
  sc.arg_u("numbers", nums);
  calloc->grow((uint32_t)nums);
}

StoreCore::Holders StoreCore::holders_from(uint32_t from) const {
  Holders h;
  if (!store_s) return h;
  for (const auto& [k, x] : store_s->sess)
    if (holds_from(x, from)) h.holders.push_back({k, x.line, k <= 0});
  h.image_numbers = (int64_t)image_numbers_from(store_s->img, from, &h.image_flights);
  return h;
}

// ---- reads -----------------------------------------------------------------------------------------------------------

std::vector<std::pair<int, int>> StoreCore::snap_numbers(int h) const {
  std::vector<std::pair<int, int>> v;
  if (!store_s) return v;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return v;
  for (const auto& sn : it->second.snaps) v.emplace_back(sn.first, (int)sn.second.size());
  return v;
}

void StoreCore::list_numbers(int h, int upto, int* in_pool, int* on_disk) const {
  int split = 0;
  list_numbers(h, upto, in_pool, on_disk, &split);
}

void StoreCore::list_numbers(int h, int upto, int* in_pool, int* on_disk, int* split) const {
  *in_pool = *on_disk = *split = 0;
  if (!store_s) return;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return;
  for (int k = 0; k < kKvSlots; ++k) {
    const Refs& l = it->second.lists[k];
    for (size_t i = 0; i < l.size() && ent_pos(k, (int)i) < upto; ++i) ++(l[i].is_slot() ? *split : l[i] ? *in_pool : *on_disk);
  }
}

size_t StoreCore::restore_chunks(int h) const {  // the numbers reading h's pieces on the disk tier back takes
  if (!store_s) return 0;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return 0;
  const Store::Sess& ss = it->second;
  size_t n = 0;
  for (const Refs& l : ss.lists)
    for (const Ref c : l) n += c.on_ssd();
  for (const auto& sn : ss.snaps) n += sn.second.empty() ? bk_nums(snap_bytes()) : 0;
  return n + ((ss.end.empty() && (ss.dend != Pool::kNone || !ss.qend.empty())) ? bk_nums(end_bytes()) : 0);
}

StoreCore::Chunks StoreCore::chunks() const {  // the chunk pool: numbers in all, held by the sessions no line holds
  Chunks c;
  if (!calloc || !store_s) return c;
  c.total = calloc->total();
  for (const auto& kv : store_s->sess)
    if (kv.first > 0 && kv.second.line < 0) c.mapped += kv.second.numbers();
  c.live_bytes = c.mapped * chunk_image();
  return c;
}

int StoreCore::bookmark_numbers(int snaps) const {
  if (snaps <= 0) return 0;
  return (int)((size_t)snaps * bk_nums(snap_bytes()));
}

// Numbers a copy on write at position `pos` takes: with h_own == h_src, own_tail's count (every list's numbers from the
// cut's chunk on that another list or bookmark also names, one fresh number each); with h_own another session (a fork of
// h_src at pos), one number per list for the partial chunk at pos. Reads the store's records only (the same on every rank).
int StoreCore::cow_numbers(int h_src, int h_own, int pos) const {
  if (!store_s) return 0;
  const auto it = store_s->sess.find(h_src);
  if (it == store_s->sess.end()) return 0;
  const Store::Sess& x = it->second;
  int n = 0;
  for (int k = 0; k < kKvSlots; ++k) {
    const int e = std::max(0, pos) / (k < 3 ? 2 : 1);
    if (h_own != h_src) {
      n += (e & (attn::kChunkEntries - 1)) != 0;
      continue;
    }
    for (size_t i = (size_t)(e >> attn::kChunkShift); i < x.lists[k].size(); ++i) {
      const Ref c = x.lists[k][i];
      n += c && c.v < refs.size() && refs[c.num()] >= 2;
    }
  }
  return n;
}

std::vector<int> StoreCore::sessions() const {
  std::vector<int> v;
  if (store_s)
    for (const auto& kv : store_s->sess)
      if (kv.first > 0) v.push_back(kv.first);  // not a unit's own (tools' steps)
  return v;
}

int StoreCore::factor(int h) const {
  if (!store_s) return 16;
  const auto it = store_s->sess.find(h);
  return it == store_s->sess.end() ? 16 : it->second.factor;
}

StoreCore::Info StoreCore::info(int h) const {
  Info r;
  if (!store_s) return r;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return r;
  const Store::Sess& ss = it->second;
  r.saved = ss.saved;
  r.split = ss.split();
  r.slots = (int)(ss.slot_entries() + ss.bookmark_slots());
  r.on_disk = ss.out();
  r.restart = restart_of(ss.saved, ss.end_pos, ss.snaps);  // its pieces in the pool or on the disk
  r.end_pos = ss.end_pos;
  for (const auto& sn : ss.snaps) r.snaps.push_back(sn.first);
  r.bytes = ss.numbers() * chunk_image() + ss.slot_entries() * sgeom::kQuarterBytes;
  for (const auto& d : ss.dl)
    for (const size_t o : d) r.disk_bytes += o != Pool::kNone ? Pool::round(store_s->qimg.count(o) ? sgeom::kQuarterDiskBytes : chunk_image()) : 0;
  r.disk_bytes += ss.dsnaps.size() * Pool::round(snap_ibytes()) + (ss.dend != Pool::kNone ? Pool::round(end_ibytes()) : 0);
  return r;
}

size_t StoreCore::disk_capacity() const { return store_s && store_s->has_disk ? store_s->dpool.size : 0; }
size_t StoreCore::disk_bytes() const { return store_s ? store_s->dbytes : 0; }
std::string StoreCore::disk_path() { return store_s && store_s->has_disk ? dev.disk_path() : ""; }
std::vector<int> StoreCore::checkpointed() const { return store_s ? store_s->ckpt_sess : std::vector<int>{}; }
void StoreCore::join_plan(int h, int upto, int* back, int* exch) const {
  *back = *exch = 0;
  if (!store_s) return;
  const auto it = store_s->sess.find(h);
  if (it == store_s->sess.end()) return;
  const Store& s = *store_s;
  for (int k = 0; k < kKvSlots; ++k)
    for (size_t i = 0; i < it->second.lists[k].size() && ent_pos(k, (int)i) < upto; ++i) {
      const Ref r = it->second.lists[k][i];
      if (!r.is_slot()) continue;
      const auto tw = s.twin_c.find(r.slot_of());
      const bool linked = tw != s.twin_c.end() && tw->second < s.twin_s.size() && s.twin_s[tw->second] == r.slot_of();
      ++(linked && (calloc->is_ghost(tw->second) || (calloc->taken(tw->second) && refs[tw->second] >= 1)) ? *back : *exch);
    }
  for (const auto& q : it->second.qsnaps) *exch += q.first <= upto;
  *exch += !it->second.qend.empty() && it->second.end_pos <= upto;
  if (g.imager)  // an entry only on the SSD whose block is still a ghost of its image comes back at the taken-back rate (no read, no exchange)
    for (int k = 0; k < kKvSlots; ++k)
      for (size_t i = 0; i < it->second.lists[k].size() && ent_pos(k, (int)i) < upto; ++i) {
        if (it->second.lists[k][i] || i >= it->second.dl[k].size() || it->second.dl[k][i] == Pool::kNone) continue;
        const auto d = s.dnum.find(it->second.dl[k][i]);
        if (d != s.dnum.end() && link_proven(d->second, it->second.dl[k][i]) && calloc->is_ghost(d->second)) ++*back;
      }
}

StoreCore::SplitView StoreCore::split_view() const {
  SplitView v;
  if (!store_s) return v;
  for (const auto& [k, ss] : store_s->sess)
    if (k > 0 && ss.split()) ++v.conv_split;
  v.quarter_slots = store_s->slots.used(), v.quarter_numbers = store_s->slots.open_numbers();
  v.ghost_numbers = calloc ? calloc->ghosts() : 0;
  return v;
}
bool StoreCore::split_can_open() const {
  if (!store_s || !calloc) return false;
  return store_s->slots.has_room() || calloc->free_below_zone() > 0;
}
uint64_t StoreCore::version() const { return store_s ? store_s->ver : 0; }
uint64_t StoreCore::kept_generation() const { return store_s ? store_s->loaded_gen : 0; }

StoreCore::TierStats StoreCore::tier_stats() const {
  TierStats t;
  t.spills = tier.spills, t.spill_bytes = tier.spill_bytes, t.restores = tier.restores, t.restore_bytes = tier.restore_bytes;
  t.restored_pos = tier.restored_pos, t.wait_ns = tier.wait_ns, t.wait_max_ns = tier.wait_max_ns;
  if (const Store* s = store_s.get()) {
    t.store_cap = calloc ? calloc->total() * chunk_image() : 0, t.store_used = chunks().live_bytes;  // one pool: the lines and the sessions kept
    for (const auto& kv : s->sess)
      if (kv.first > 0) ++(kv.second.out() ? t.on_disk : t.resident);
    if (s->has_disk) t.disk_cap = s->dpool.size, t.disk_used = s->dbytes;
  }
  return t;
}

StoreCore::PoolStats StoreCore::pool_stats() const {
  PoolStats p;
  if (!calloc) return p;
  p.numbers = calloc->total(), p.free = calloc->free(), p.held = calloc->held(), p.number_bytes = chunk_image();
  for (const uint32_t r : refs) p.shared += r > 1;
  for (int s = 0; s < nslots; ++s) p.reserved += (uint64_t)std::max(0, resv[s] - held(s)) + (uint64_t)resv_img[s];
  p.image_numbers = img_numbers();
  if (const Store* st = store_s.get())
    for (const auto& [k, ss] : st->sess) {
      if (k <= 0) continue;  // a line's own session (tools, tests)
      ++(ss.line >= 0 ? p.on_lines : ss.out() ? p.on_disk : p.in_memory);
      bool sh = false;
      for (const Refs& l : ss.lists)
        for (const Ref c : l) sh = sh || (c.is_number() && refs[c.num()] > 1);
      p.sharing += sh;
    }
  return p;
}

// A hash of the state every rank must hold alike (sessions with their lists, bookmarks and images, the numbers' counts and
// images, the reservations, the evict-last list, the pool's lending): compared across the ranks' logs under DSGB10_STORE_CHECK=1.
uint64_t StoreCore::digest(const Outside& o) const {
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  if (store_s) {
    const Store& s = *store_s;
    for (const auto& [k, x] : s.sess) {
      mix((uint64_t)(uint32_t)k), mix((uint64_t)(uint32_t)x.line), mix((uint64_t)(uint32_t)x.saved), mix((uint64_t)(uint32_t)x.end_pos);
      for (int a = 0; a < kKvSlots; ++a) {
        mix(x.lists[a].size());
        for (const Ref c : x.lists[a]) mix(c.v);
        for (const size_t o2 : x.dl[a]) mix(o2);
      }
      for (const auto& [pos, b] : x.snaps) {
        mix((uint64_t)(uint32_t)pos), mix(b.size());
        for (const uint32_t c : b) mix(c);
      }
      for (const auto& [pos, o2] : x.dsnaps) mix((uint64_t)(uint32_t)pos), mix(o2);
      for (const uint32_t c : x.end) mix(c);
      for (const auto& q : x.qsnaps) {
        mix((uint64_t)(uint32_t)q.first | 1ull << 40);
        for (const Ref r : q.second) mix(r.v);
      }
      for (const Ref r : x.qend) mix(r.v | 1ull << 41);
      mix(x.dend);
    }
    for (const size_t o2 : s.ndisk) mix(o2);
    if (!s.jobs.empty() || !s.bk_ghost.empty()) {  // the imager's queue and the bookmarks' ghost groups: nothing is mixed while none exists
      mix(s.img_seq);
      for (const Store::ImgJob& j : s.jobs) mix(j.id), mix(j.seq), mix(j.off), mix((uint64_t)(uint32_t)j.h);
      for (const auto& [off, grp] : s.bk_ghost) {
        mix(off);
        for (const uint32_t c : grp) mix(c);
      }
    }
    for (const int k : s.last) mix((uint64_t)(uint32_t)k);
  }
  for (const uint32_t r : refs) mix(r);
  // the pool's lending: the groups out of it (the windows' and the Engram cache's), those owed, the pool's size before any
  // left, the cache's own groups and the phase of its change
  mix(o.groups), mix(o.owed), mix(o.n0);
  mix(o.cur_groups), mix(o.phase);
  for (int i = 2; i < 16; ++i)
    if (o.tenant_groups[i] || o.tenant_pending[i]) mix((uint64_t)i), mix(o.tenant_groups[i]), mix(o.tenant_pending[i]);
  if (calloc) mix((uint64_t)calloc->numbers());
  if (calloc && calloc->split_digest()) mix(calloc->split_digest());  // ghosts and the lend zone: nothing is mixed while none exists
  if (store_s && store_s->slots.digest()) mix(store_s->slots.digest());
  for (int t = 0; t < nslots; ++t) mix((uint64_t)(uint32_t)resv[t]), mix((uint64_t)(uint32_t)resv_img[t]);
  if (store_s)
    store_s->img.each([&](int key, const imgrows::Ledger::Rows& r) {
      mix((uint64_t)(uint32_t)key), mix(r.rows);
      for (const uint32_t c : r.nums) mix(c);
    });
  return h;
}

// The wide digest: the digest, then what it leaves out and a replay must reproduce: the allocator's state, the clock and version,
// the disk tier's bytes, free ranges, image references and holders, the named and retired images, the kept and loaded generations,
// every session's use, factor and kept flag, the pinned and live lists, and every unit's session and slot's mapped position. Not the
// evicted list (only rank 0's server empties it), nor a write sequence, a time or a counter.
uint64_t StoreCore::wide_digest(uint64_t today) const {
  store_trace::Mixer x;
  x.mix(today);
  if (calloc) x.mix(calloc->digest()), x.mix(calloc->numbers()), x.mix(calloc->free());
  else x.mix(0), x.mix(0), x.mix(0);
  for (int t = 0; t < nslots; ++t) x.mix_i(mapped_upto[t]);
  for (int u = 0; u < nunits; ++u) x.mix_i(line_h[u]);
  if (!store_s) return x.h;
  const Store& s = *store_s;
  x.mix(s.clock), x.mix(s.ver), x.mix(s.dbytes), x.mix(s.kept_gen), x.mix(s.loaded_gen);
  for (const auto& [k, e] : s.sess) x.mix_i(k), x.mix(e.used), x.mix_i(e.factor), x.mix(e.kept);
  x.mix(s.drefs.size());
  for (const auto& [off, n] : s.drefs) x.mix(off), x.mix(n);
  x.mix(s.dnum.size());
  for (const auto& [off, c] : s.dnum) x.mix(off), x.mix(c);
  x.mix(s.dpool.size), x.mix(s.dpool.free.size());
  for (const auto& [off, n] : s.dpool.free) x.mix(off), x.mix(n);
  x.mix(s.named.size());
  for (const size_t off : s.named) x.mix(off);
  x.mix(s.retired.size());
  for (const auto& [off, n] : s.retired) x.mix(off), x.mix(n);
  x.mix(s.ckpt_sess.size());
  for (const int k : s.ckpt_sess) x.mix_i(k);
  x.mix(s.pinned.size());
  for (const int k : s.pinned) x.mix_i(k);
  x.mix(s.live.size());
  for (const int k : s.live) x.mix_i(k);
  if (s.slots.digest() || !s.twin_c.empty() || !s.slot_disk.empty() || s.ghost_seq) {  // the split's links, only once there is one
    x.mix(s.ghost_seq), x.mix(s.twin_c.size());
    for (const auto& [sl, c] : s.twin_c) x.mix(sl), x.mix(c);
    x.mix(s.slot_disk.size());
    for (const auto& [sl, off] : s.slot_disk) x.mix(sl), x.mix(off);
    x.mix(s.slot_written.size());
    for (const auto& [sl, w] : s.slot_written) x.mix(sl), x.mix_i(w);
  }
  return x.h;
}

namespace {

void trace_snapshot(const void* ctx, store_trace::Snapshot* o) {
  const StoreCore& c = *static_cast<const StoreCore*>(ctx);
  const StoreCore::Outside out = c.outside();
  o->digest = c.digest(out);
  o->wide = c.wide_digest(o->digest);
  o->out[0] = out.groups, o->out[1] = out.owed, o->out[2] = out.n0, o->out[3] = out.cur_groups, o->out[4] = out.phase;
  o->out[5] = c.calloc ? (uint64_t)c.calloc->numbers() : 0;
  o->free_numbers = c.calloc ? c.calloc->free() : 0;
  o->numbers = o->out[5];
}

[[maybe_unused]] const bool g_trace_registered = (store_trace::set_snapshot(&trace_snapshot), true);

std::string ints(const int* v, int n) {
  std::string o;
  for (int i = 0; i < n; ++i) o += (i ? "," : "") + std::to_string(v[i]);
  return o;
}

}  // namespace

void StoreCore::trace_open() {
  store_trace::Writer* w = store_trace::g_w;
  if (!w || w->is_open()) return;
  char g1[400];
  std::snprintf(g1, sizeof g1,
                "rank=%d nslots=%d nunits=%d list_len=%d max_ctx=%lld image=%zu snap=%zu end=%zu spec=%d poison=%d host_ring=%d", rank,
                nslots, nunits, list_len, (long long)max_ctx, chunk_image(), snap_bytes(), end_bytes(), g.spec ? 1 : 0, g.poison ? 1 : 0,
                g.has_ring ? 1 : 0);
  const std::string g2 = "unit_of=" + ints(unit_of, nslots) + " slot_ctx=" + ints(slot_ctx_, nslots);
  w->open(rank, g1, g2, calloc ? calloc->numbers() : 0, store_s && store_s->has_disk ? dev.disk_size() : 0);
}

}  // namespace model::detail
