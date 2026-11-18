#pragma once
#include "engram/block.h"
#include "engram/cache.h"
#include "engram/engram.h"

namespace engram {

// The hot-row cache: a CLOCK index over slots of kRowBytes. One slot numbering over two regions: the first
// `region_bytes / kRowBytes` slots live in `region` (the display reservation's mapping, display.h), the rest in
// ordinary memory. The index (keys, reference bits, table) is always ordinary memory.
//
// Two ways to make one. init(): a fixed size, its memory from mmap at once. init_growable(): an empty cache over a Block of
// address space whose size changes while rows are served: resize_begin() and resize_step() (a slice of the work each, from
// the thread that owns the cache) commit and populate or release the block's pages, never moving what is there; the rows
// cached stay cached across a grow and the slots above a shrink's end leave.
struct RowCache {
  ClockIndex idx;
  uint8_t* data = nullptr;        // the ordinary slots, from slot `split`
  uint8_t* region = nullptr;      // not owned
  size_t split = 0;               // slots in the region

  // `region` may be null (all ordinary). The region holds at most the cache's slots.
  void init(size_t bytes, uint8_t* reg = nullptr, size_t region_bytes = 0) {
    const size_t slots = bytes / kRowBytes;
    if (!slots) return;
    region = reg;
    split = reg ? std::min(slots, region_bytes / kRowBytes) : 0;
    if (slots > split) data = static_cast<uint8_t*>(big_alloc((slots - split) * kRowBytes));
    idx.init(slots);
  }
  ~RowCache() {
    if (data && !growable_) munmap(data, (idx.slots() - split) * kRowBytes);
  }

  // An empty cache that can grow to `max_bytes` of slots. The first `region_bytes / kRowBytes` slots, however many the
  // cache has, are the region's. False when the address space cannot be reserved.
  bool init_growable(size_t max_bytes, uint8_t* reg = nullptr, size_t region_bytes = 0) {
    max_slots_ = max_bytes / kRowBytes;
    region = reg;
    split = reg ? region_bytes / kRowBytes : 0;
    const size_t ordinary = max_slots_ > split ? (max_slots_ - split) * kRowBytes : 0;
    const size_t table = index_table_entries(max_slots_) * 4;
    if (!blk_.reserve({ordinary, max_slots_ * 8, max_slots_, table, table})) return false;
    idx.attach(blk_, 1, 2, 3, 4);
    data = blk_.at(0);
    growable_ = true;
    return true;
  }
  bool growable() const { return growable_; }
  size_t max_slots() const { return max_slots_; }
  size_t slots() const { return idx.slots(); }
  // The ordinary bytes the block commits for `slots` slots of the cache (what its slot range holds)
  size_t ordinary_bytes(size_t slots) const { return slots > split ? (slots - split) * kRowBytes : 0; }

  // ---- resizing (growable only) ----
  // Starts a change to `slots` slots; false when one is running, the size is the same, or it is past the maximum.
  bool resize_begin(size_t slots) {
    if (!growable_ || resizing() || slots == idx.slots() || slots > max_slots_) return false;
    target_ = slots;
    if (slots > idx.slots()) {
      mode_ = Mode::Grow, stage_ = 0;
      return true;
    }
    if (!idx.shrink_begin(slots)) return false;
    mode_ = Mode::Shrink;
    return true;
  }
  bool resizing() const { return mode_ != Mode::Idle; }
  bool failed() const { return failed_; }
  // One slice of the change. Done: the cache is at its new size (or, Failed, as it was).
  Step resize_step(size_t slice = size_t(32) << 20, size_t build = 32768) {
    if (mode_ == Mode::Idle) return Step::Done;
    if (mode_ == Mode::Grow) {
      if (stage_ == 0) {  // the ordinary slots' pages first: a slice of them per step
        const size_t need = ordinary_bytes(target_);
        if (!blk_.commit(0, std::min(blk_.committed(0) + slice, need))) return undo();
        if (blk_.committed(0) >= need) {
          stage_ = 1;
          if (!idx.grow_begin(target_)) return undo();
        }
        return Step::More;
      }
      const Step st = idx.grow_step(slice, build);
      if (st == Step::Failed) return undo();
      if (st == Step::Done) return stop(Step::Done);
      return Step::More;
    }
    const Step st = idx.shrink_step(build, slice);
    if (st == Step::Done) {
      blk_.decommit(0, ordinary_bytes(target_));
      return stop(Step::Done);
    }
    return Step::More;
  }
  Block& block() { return blk_; }
  const Block& block() const { return blk_; }

  void clear() { idx.clear(); }
  uint8_t* slot(uint32_t s) { return s < split ? region + size_t(s) * kRowBytes : data + (size_t(s) - split) * kRowBytes; }
  const uint8_t* find(uint64_t key) {
    const uint32_t s = idx.find(key);
    return s == kNoSlot ? nullptr : slot(s);
  }
  // Returns true if a row was evicted.
  bool insert(uint64_t key, const uint8_t* rec) {
    bool evicted;
    const uint32_t s = idx.insert(key, &evicted);
    if (s != kNoSlot) std::memcpy(slot(s), rec, kRowBytes);
    return evicted;
  }

 private:
  enum class Mode { Idle, Grow, Shrink };
  Step stop(Step st) {
    mode_ = Mode::Idle;
    failed_ = st == Step::Failed;
    return st;
  }
  Step undo() {  // a grow that failed: the slots it committed go back
    blk_.decommit(0, ordinary_bytes(idx.slots()));
    return stop(Step::Failed);
  }
  Block blk_;  // ranges: 0 ordinary slots, 1 keys, 2 reference bits, 3 and 4 the tables
  bool growable_ = false, failed_ = false;
  size_t max_slots_ = 0, target_ = 0;
  Mode mode_ = Mode::Idle;
  int stage_ = 0;
};

}  // namespace engram
