// The store's geometry: the constants the store's bookkeeping needs, and the plain values the engine hands it at start. Host
// only (no CUDA header), so the bookkeeping builds and runs on the host. Each constant is checked against the header that owns
// it in state.cu (static_assert), so the two cannot drift.
#pragma once
#include <cstddef>
#include <cstdint>

#include "model/config.h"
#include "model/quarter_map.h"

namespace model::sgeom {

constexpr int kChunkShift = 13, kChunkEntries = 1 << kChunkShift;  // entries in a chunk number (attn.h)
constexpr int kSnapRows = 128;                                      // positions a window bookmark holds (prefill.h)
constexpr int kMapLookahead = 4096;                                 // positions a line's lists grow beyond what a step needs (impl.h)
constexpr size_t kPoolGrain = 64 << 10;                             // the disk tier's blocks: multiples of 64 KiB, at 64 KiB offsets
constexpr size_t kBk = (size_t)kChunkEntries * 256;                 // a bookmark number's bytes: its KV slice, 2 MiB
constexpr int kSlotShift = 2;                                       // a quarter slot is (number << 2) | quarter
constexpr size_t kQuarterBytes = quarter::kQuarterBytes;            // a quarter slot's bytes: a quarter of a number in every array
constexpr size_t kQuarterDiskBytes = quarter::kDiskBytes;           // a quarter's image on the disk tier: 12 grains, its trailer after it
// An image of a window bookmark and of the end bookmark under the imager: this rank's slice of its bytes and the trailer's used words,
// in one block of whole quarter images (a quarter's disk bytes, twice for the end)
constexpr size_t kBookmarkImageBytes = kQuarterDiskBytes, kEndImageBytes = 2 * kQuarterDiskBytes;
static_assert(quarter::kEntries == (1u << kChunkShift) && kQuarterDiskBytes % kPoolGrain == 0, "a quarter's block is whole grains");

// What the engine tells the store about itself once, when it makes it.
struct StoreGeom {
  int rank = 0;
  int nslots = 1, nunits = 1;
  int list_len = 0;                       // numbers a line's list can hold
  int max_ctx = 0;
  int unit_of[kMaxSlots] = {};            // slot -> unit
  int slot_ctx[kMaxSlots] = {};           // positions slot s can hold
  size_t image_bytes = 0;                 // a chunk number's bytes across the four arrays
  size_t snap_bytes = 0, end_bytes = 0;   // a window bookmark and an end bookmark (the draft module's part counted when loaded)
  bool spec = false;                      // the draft module is loaded
  bool poison = false;                    // taken numbers hold 0xFF until written, given-back ones 0xFD
  bool has_ring = false;                  // a host ring exists: a read-back's cut is agreed across the ranks
  bool split = false;                     // --kv-split cache: quarter images on the disk tier, the kept index of layout K4, its write agreed
  // Every write to the disk tier is each rank's own quarter, made in the background on the image stream (image,
  // settle); a stored conversation is marked (clean pieces become ghosts) and never copied; eviction takes clean pieces only; the
  // kept index is K5. The engine sets it (state.cu make_store); a host test that tests the old rules leaves it false.
  bool imager = false;
  bool mark = false;                      // with the imager: a settle marks the idle conversations (--kv-split tiered | max); off images and marks nothing
};

}  // namespace model::sgeom
