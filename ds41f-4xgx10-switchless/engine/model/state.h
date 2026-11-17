// Sessions' saved state (state.cu): what a session needs to continue from a position without recomputing what it
// already holds. Its compressed caches are its own chunk lists in the chunk pool (attn.h Compressed, model.cu "the
// chunk pool"): a line (a slot's unit) that holds the session points its rows at them, so taking a session into a
// line or out of one copies no cache. The window rings and the compressor carries are per position mod 256 / 16, so
// they are snapshotted into bookmarks: a window bookmark (the 128 positions before a restart point) and an end
// bookmark (whole rings and carries), each in whole chunk numbers of the pool, used as raw bytes in their 2 MiB KV
// slices (2 and 3 numbers). The draft module's windows (DSpark, spec.cu) go with them.
//
// Every rank runs the same operations in the same order and takes the same chunk numbers (lowest free first), so
// every rank holds the same numbers for the same sessions. When the pool has too few free numbers, the least
// recently used session that no line holds and the current command does not use is evicted (the same one on every
// rank): spilled to the disk tier, or freed without one. A bookmark that still finds no room is not taken. Nothing
// in the store fails a request: admission (serve.cpp) counts the numbers a request needs first.
//
// With a disk tier (Model::state_disk, spill.h) an evicted session is spilled: its compressed caches go out as pages
// of kPagePos positions (gathered from its chunks), its bookmarks as blocks, into a file of the same size on every
// rank at the same offsets. When the file is full, its least recently used sessions are dropped. A spilled session
// comes back (state_unspill, or any operation that needs it) as far as the pool has room, latest restart point first:
// the end bookmark with the pages it needs, else the latest window bookmark whose pages fit, then the earlier ones
// those pages serve; never a block no restart can use. A session is spilled as what it can restart from
// (StateInfo::restart).
//
// Kept sessions (state.cu): a planned stop keeps the file and writes an index of the sessions in it beside it; the
// next start that finds both as this run would have left them (the same run id, not too old, on every rank alike)
// opens the file as it is, and its sessions sit in the disk tier as spilled ones until a request reads them back.
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

constexpr int kPagePos = 4096;  // positions per page of compressed caches
constexpr size_t kEntryBytes = 256 + 32 + 64 + 4;  // KV entry + scales, index key + scales
constexpr size_t kPageBytes = (size_t)(kPagePos / 2 * 3 + kPagePos) * kEntryBytes;  // sources 2, 8, 14 (ratio 2), 20
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
    Nums lists[kKvSlots];   // compressed caches: kv source k's chunk numbers (attn.h Compressed), its own
    int saved = 0;          // positions [0, saved) are in them
    std::map<int, Nums> snaps;  // window bookmarks by position: their chunk numbers (the draft module's windows after)
    Nums end;               // every layer's ring and the carries (and the draft module's windows); empty: none
    int end_pos = -1;       // at this position
    int line = -1;          // the unit whose rows are its lists (-1: in no line)
    uint64_t used = 0;      // the clock when an operation last used the session
    // spilled: its caches as pages and its bookmarks as blocks at offsets in the disk tier's file, nothing in the pool
    bool disk = false;
    std::vector<size_t> dpages;
    std::map<int, size_t> dsnaps;
    size_t dend = Pool::kNone;
    uint64_t dseq = 0;  // its last write (reading it back waits for that)
    bool kept = false;  // as an earlier run kept it, not read back since: a block that differs is fatal
    size_t numbers() const {  // chunk numbers it names (a shared one counts in every session naming it)
      size_t n = end.size();
      for (const Nums& l : lists) n += l.size();
      for (const auto& sn : snaps) n += sn.second.size();
      return n;
    }
  };
  // A session in a kept file's index: its blocks' offsets and their checksums as they went out.
  struct Kept {
    int h = 0, saved = 0, end_pos = -1;
    uint64_t used = 0;
    std::vector<std::pair<size_t, uint64_t>> pages;
    std::map<int, std::pair<size_t, uint64_t>> snaps;
    size_t dend = Pool::kNone;
    uint64_t dend_sum = 0;
  };
  std::map<int, Sess> sess;
  uint8_t* gather = nullptr;        // device: one page or bookmark at a time, between the chunks and the disk tier
  std::unique_ptr<SpillFile> disk;  // the disk tier (none: evicted sessions are freed)
  Pool dpool;                       // its file's offsets
  size_t dbytes = 0;                // the file's bytes in spilled sessions' blocks
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
