// Sessions' saved state (state.cu): what a session needs to continue from a position without recomputing. Its
// compressed caches are its own chunk lists in the chunk pool; a line holding it points its rows at them, so no cache is
// copied. Window rings and compressor carries are snapshotted into bookmarks (window bookmark: the 128 positions before a
// restart point, 2 numbers; end bookmark: whole rings and carries, 3). Details: docs/engine.md "The chunk pool", "Kept sessions".
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model/impl.h"
#include "model/prefill.h"

namespace model::detail {

constexpr size_t kRingBytes = (size_t)attn::kRing * (kHead + kHead / 32);
constexpr size_t kCarryBytes = (size_t)16 * kHead * 4 * 2;
constexpr size_t kEndBytes = kLayers * kRingBytes + kKvSlots * kCarryBytes;
constexpr size_t kPoolGrain = 64 << 10;  // the pool's blocks: multiples of 64 KiB, at 64 KiB offsets

// Best fit over [base, base + size); free ranges are coalesced, ties go to the lowest offset. The
// offsets alone (take_off, give_off) serve the disk tier's file and its staging buffer.
struct Pool {
  static constexpr size_t kNone = SIZE_MAX;
  uint8_t* base = nullptr;
  size_t size = 0;
  std::map<size_t, size_t> free;  // offset -> bytes
  static size_t round(size_t bytes) { return (bytes + kPoolGrain - 1) / kPoolGrain * kPoolGrain; }
  uint8_t* take(size_t bytes);  // nullptr when no free range holds round(bytes)
  void give(uint8_t* p, size_t bytes);
  size_t take_off(size_t bytes);  // kNone when no free range holds round(bytes)
  bool take_at(size_t off, size_t bytes);  // [off, off + round(bytes)) when it is free (kept sessions)
  void give_off(size_t off, size_t bytes);
  size_t largest() const;
  // The first free range, lowest offset first, whose start `ok(offset, rounded bytes)` accepts: the block goes at
  // its start (first fit, so that the blocks stay at the bottom of the range). kNone when there is none.
  size_t take_fit(size_t bytes, const std::function<bool(size_t, size_t)>& ok);
};

class SpillFile;

using Nums = std::vector<uint32_t>;  // chunk numbers (model/chunks.h)

struct Store {
  struct Sess {
    Nums lists[kKvSlots];   // compressed caches: kv source k's chunk numbers (attn.h Compressed); 0: that entry is only on
                            // the disk tier (dl), read back before use
    std::vector<size_t> dl[kKvSlots];  // each entry's image in the disk tier's file (Pool::kNone: none), beside lists
    int saved = 0;          // positions [0, saved) are in them
    std::map<int, Nums> snaps;  // window bookmarks by position: their chunk numbers (empty: only on the disk tier)
    Nums end;               // every layer's ring and the carries (and the draft module's windows); empty: none in the pool
    int end_pos = -1;       // its position (-1: no end bookmark, in the pool or on the disk tier)
    int line = -1;          // the unit whose rows are its lists (-1: in no line)
    uint64_t used = 0;      // the clock when an operation last used the session
    std::map<int, size_t> dsnaps;  // window bookmarks' images in the file (a bookmark with a copy)
    size_t dend = Pool::kNone;     // the end bookmark's image
    bool kept = false;  // as an earlier run kept it, not read back since: a piece that differs is fatal
    bool out() const;   // a piece is only on the disk tier: read back before the session is used (state.cu)
    size_t numbers() const {  // chunk numbers it names in the pool (a shared one counts in every session naming it)
      size_t n = end.size();
      for (const Nums& l : lists)
        for (const uint32_t c : l) n += c != 0;
      for (const auto& sn : snaps) n += sn.second.size();
      return n;
    }
  };
  // A session in a kept file's index: its pieces' images and their checksums as they went out.
  struct Kept {
    int h = 0, saved = 0, end_pos = -1;
    uint64_t used = 0;
    std::vector<std::pair<size_t, uint64_t>> ents[kKvSlots];
    std::map<int, std::pair<size_t, uint64_t>> snaps;
    size_t dend = Pool::kNone;
    uint64_t dend_sum = 0;
  };
  std::map<int, Sess> sess;
  uint8_t* gather = nullptr;        // device: one image at a time, between the pool and the disk tier
  std::unique_ptr<SpillFile> disk;  // the disk tier (none: evicted sessions are freed)
  Pool dpool;                       // its file's offsets
  size_t dbytes = 0;                // the file's bytes in images
  uint64_t wseq = 0;                // the last write queued (a read-back waits for it)
  std::vector<size_t> ndisk;        // [number]: the image of its present content (Pool::kNone: none); shared holders share it
  std::map<size_t, uint32_t> dnum;  // image -> the number holding its content, while one does
  std::map<size_t, uint32_t> drefs; // image -> the entries and bookmarks naming it
  std::string base;                 // the file is base.spill, its index base.index
  uint64_t run_id = 0;              // what the kept sessions' state depends on (main.cpp run_id)
  std::vector<Kept> kept;           // a kept index as state_kept_check read it, for state_disk
  uint64_t kept_gen = 0, kept_clock = 0;  // its stop's generation and the store's clock then
  uint64_t loaded_gen = 0;                // the generation this run's store came from (0: none)
  uint64_t clock = 0;
  std::vector<int> pinned;   // the sessions the current command uses: never evicted during it
  std::vector<int> live;     // the sessions running in the slots (a running request's): never evicted either; a session on an
                             // idle or paused line is not named here, so its conversation is cache like a stored one
  std::vector<int> evicted;  // sessions evicted to make room, since rank 0's server last asked
  Sess& use(int h) {         // session h (made if new), used now
    Sess& s = sess[h];
    s.used = ++clock;
    return s;
  }
  ~Store();  // state.cu: waits for the disk tier's writes and removes its file
};

}  // namespace model::detail
