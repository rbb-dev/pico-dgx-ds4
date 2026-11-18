// The hot-row cache's index: CLOCK replacement over a fixed number of slots, found through an
// open-addressing table (linear probing, backward-shift deletion). New rows enter unreferenced, so
// rows seen once leave before rows seen again. Shared by the store and the hit-rate simulator.
#pragma once
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/check.h"
#include "engram/block.h"

namespace engram {

constexpr uint32_t kNoSlot = 0xffffffffu;
constexpr uint64_t kNoKey = ~uint64_t(0);

inline uint64_t mix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return k;
}

inline void* big_alloc(size_t bytes) {
  if (!bytes) return nullptr;
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) die("engram: mmap %zu bytes: %s", bytes, strerror(errno));
  madvise(p, bytes, MADV_HUGEPAGE);
  return p;
}

// Slot indices in an open-addressing table; the keys live with the caller (key_of(slot)). The entries are the table's own
// vector, or memory the caller attached (a block's range: the growable index).
struct SlotTable {
  uint32_t* t = nullptr;
  size_t mask = 0;
  std::vector<uint32_t> own;
  void init(size_t entries) {
    size_t n = 1;
    while (n < entries) n <<= 1;
    own.assign(n, kNoSlot);
    t = own.data();
    mask = n - 1;
  }
  // `entries` (a power of two) entries at p, all kNoSlot after clear()
  void attach(uint32_t* p, size_t entries) {
    own.clear();
    own.shrink_to_fit();
    t = p;
    mask = entries - 1;
  }
  void detach() { t = nullptr, mask = 0; }
  size_t entries() const { return t ? mask + 1 : 0; }
  void clear() {
    if (t) std::fill(t, t + mask + 1, kNoSlot);
  }
  template <class KeyOf>
  uint32_t find(uint64_t key, KeyOf key_of) const {
    for (size_t h = mix64(key) & mask;; h = (h + 1) & mask) {
      const uint32_t s = t[h];
      if (s == kNoSlot || key_of(s) == key) return s;
    }
  }
  void insert(uint64_t key, uint32_t s) {
    size_t h = mix64(key) & mask;
    while (t[h] != kNoSlot) h = (h + 1) & mask;
    t[h] = s;
  }
  template <class KeyOf>
  void erase(uint64_t key, KeyOf key_of) {
    size_t i = mix64(key) & mask;
    while (key_of(t[i]) != key) i = (i + 1) & mask;
    for (size_t j = i;;) {
      j = (j + 1) & mask;
      if (t[j] == kNoSlot) break;
      const size_t k = mix64(key_of(t[j])) & mask;  // home of the entry at j
      const bool stays = i <= j ? (i < k && k <= j) : (i < k || k <= j);
      if (!stays) {
        t[i] = t[j];
        i = j;
      }
    }
    t[i] = kNoSlot;
  }
};

// The table's entries for `slots` slots: a power of two at least twice the slots (4 bytes each).
inline size_t index_table_entries(size_t slots) {
  size_t n = 1;
  while (n < slots * 2) n <<= 1;
  return n;
}
// The index's bytes for `slots` slots: 8-byte keys, 1-byte reference bits and the table.
inline size_t index_bytes_for(size_t slots) { return slots ? slots * 9 + index_table_entries(slots) * 4 : 0; }

// Where the work of a resize stands: one slice of it is done by each step.
enum class Step { More, Done, Failed };

class ClockIndex {
 public:
  explicit ClockIndex(size_t slots = 0) { init(slots); }
  ~ClockIndex() {
    if (cap_ && heap_) {
      munmap(keys_, cap_ * 8);
      munmap(ref_, cap_);
    }
  }
  ClockIndex(const ClockIndex&) = delete;
  ClockIndex& operator=(const ClockIndex&) = delete;

  void init(size_t slots) {
    cap_ = slots;
    heap_ = true;
    if (!cap_) return;
    keys_ = static_cast<uint64_t*>(big_alloc(cap_ * 8));
    ref_ = static_cast<uint8_t*>(big_alloc(cap_));
    table_.init(cap_ * 2);
    clear();
  }
  void clear() {
    for (size_t i = 0; i < cap_; i++) keys_[i] = kNoKey;
    if (cap_) std::memset(ref_, 0, cap_);
    table_.clear();
    hand_ = 0;
  }
  size_t slots() const { return cap_; }

  // The slot holding `key`, marked referenced, or kNoSlot.
  uint32_t find(uint64_t key) {
    if (!cap_) return kNoSlot;
    const uint32_t s = table_.find(key, [this](uint32_t i) { return keys_[i]; });
    if (s != kNoSlot) ref_[s] = 1;
    return s;
  }
  // A slot for a new key (evicting by CLOCK if full); `evicted` says whether a row was dropped. While the table is being
  // rebuilt (growing) no row is taken in: kNoSlot, as for an index with no slots.
  uint32_t insert(uint64_t key, bool* evicted) {
    *evicted = false;
    if (!cap_ || building_) return kNoSlot;
    uint32_t s;
    for (;;) {
      s = uint32_t(hand_);
      hand_ = hand_ + 1 == cap_ ? 0 : hand_ + 1;
      if (keys_[s] == kNoKey) break;
      if (ref_[s]) {
        ref_[s] = 0;
        continue;
      }
      table_.erase(keys_[s], [this](uint32_t i) { return keys_[i]; });
      *evicted = true;
      break;
    }
    keys_[s] = key;
    ref_[s] = 0;
    table_.insert(key, s);
    return s;
  }

  // ---- a growable index: its arrays are ranges of a Block (keys, reference bits, two tables), none of it committed at
  // first and no slots. Growing and shrinking are done a slice at a time by the thread that owns the index; lookups go on
  // against the old size until a grow's last step. ----

  // The ranges of `b` for the keys, the reference bits and the two tables (which take turns when the table grows).
  void attach(Block& b, int r_keys, int r_ref, int r_tab0, int r_tab1) {
    blk_ = &b, rk_ = r_keys, rr_ = r_ref, rt_[0] = r_tab0, rt_[1] = r_tab1;
    heap_ = false, cap_ = 0, tab_cur_ = 0;
    keys_ = reinterpret_cast<uint64_t*>(b.at(r_keys));
    ref_ = b.at(r_ref);
    table_.detach();
  }
  bool growable() const { return blk_ != nullptr; }
  bool busy() const { return op_ != Op::Idle; }
  bool rebuilding() const { return building_; }
  size_t table_entries() const { return table_.entries(); }

  // Starts a grow to `slots` slots; false when it is not larger, past the block's ranges, or another resize is running.
  bool grow_begin(size_t slots) {
    if (!blk_ || op_ != Op::Idle || slots <= cap_ || slots * 8 > blk_->max_of(rk_) || slots > blk_->max_of(rr_) ||
        index_table_entries(slots) * 4 > blk_->max_of(rt_[0]))
      return false;
    op_ = Op::Grow, target_ = slots, stage_ = kKeys, pos_ = 0, filled_ = cap_;  // the keys below cap_ are live
    return true;
  }
  // One slice (about `slice` bytes of commit, or slice / 4 entries of table, or `build` slots of rows) of the grow.
  Step grow_step(size_t slice, size_t build) {
    if (op_ != Op::Grow) return Step::Done;
    const size_t T = target_;
    if (stage_ == kKeys) {
      const size_t to = std::min(blk_->committed(rk_) + slice, T * 8);
      if (!blk_->commit(rk_, to)) return fail();
      const size_t upto = std::min(blk_->committed(rk_) / 8, T);
      for (size_t i = filled_; i < upto; ++i) keys_[i] = kNoKey;
      filled_ = upto;
      if (blk_->committed(rk_) < T * 8 || filled_ < T) return Step::More;
      stage_ = kRef;
      return Step::More;
    }
    if (stage_ == kRef) {
      if (!blk_->commit(rr_, std::min(blk_->committed(rr_) + slice, T))) return fail();
      if (blk_->committed(rr_) < T) return Step::More;
      const size_t ne = index_table_entries(T);
      if (ne > table_.entries()) {
        stage_ = kTabCommit;
        spare_ = table_.entries() ? 1 - tab_cur_ : tab_cur_;
      } else {
        stage_ = kFinish;
      }
      return Step::More;
    }
    if (stage_ == kTabCommit) {
      const size_t ne = index_table_entries(T);
      if (!blk_->commit(rt_[spare_], std::min(blk_->committed(rt_[spare_]) + slice, ne * 4))) return fail();
      if (blk_->committed(rt_[spare_]) < ne * 4) return Step::More;
      spare_table_.attach(reinterpret_cast<uint32_t*>(blk_->at(rt_[spare_])), ne);
      stage_ = kTabFill, pos_ = 0;
      return Step::More;
    }
    if (stage_ == kTabFill) {
      const size_t ne = spare_table_.entries(), n = std::min(slice / 4, ne - pos_);
      std::fill(spare_table_.t + pos_, spare_table_.t + pos_ + n, kNoSlot);
      pos_ += n;
      if (pos_ < ne) return Step::More;
      stage_ = kTabBuild, pos_ = 0;
      building_ = true;  // from here the rows are copied: none is taken in until the swap
      return Step::More;
    }
    if (stage_ == kTabBuild) {
      const size_t n = std::min(build, cap_ - std::min(pos_, cap_));
      for (size_t i = 0; i < n; ++i) {
        const size_t s = pos_ + i;
        if (keys_[s] != kNoKey) spare_table_.insert(keys_[s], uint32_t(s));
      }
      pos_ += n;
      if (pos_ < cap_) return Step::More;
      stage_ = kFinish;
      return Step::More;
    }
    // kFinish: the new size takes over in one step (no row was taken in since the copy began)
    if (spare_table_.t) {
      const bool had = table_.entries() != 0;
      table_ = std::move(spare_table_);
      if (had) blk_->decommit(rt_[tab_cur_], 0);
      tab_cur_ = spare_;
      spare_table_ = SlotTable();
    }
    hand_ = cap_;  // the new slots are empty: the hand starts on them
    cap_ = T;
    building_ = false;
    op_ = Op::Idle;
    return Step::Done;
  }
  // Starts a shrink to `slots` slots (fewer): the slots above leave at once (no row goes into them), their rows are
  // dropped from the table a slice at a time and their pages decommitted. The table then follows the size down: a table
  // the smaller size needs is built beside the old one (no row is taken in while it is copied, lookups go on against the
  // old one) and the old one's pages go back; a cache shrunk to nothing keeps no table. A table that cannot be committed
  // stays as it is: the shrink is done and the larger table is valid, only its pages stay.
  bool shrink_begin(size_t slots) {
    if (!blk_ || op_ != Op::Idle || slots >= cap_) return false;
    op_ = Op::Shrink, target_ = slots, old_cap_ = cap_, pos_ = slots, stage_ = kShErase;
    cap_ = slots;
    if (hand_ >= cap_) hand_ = 0;
    return true;
  }
  Step shrink_step(size_t build, size_t slice = size_t(32) << 20) {
    if (op_ != Op::Shrink) return Step::Done;
    if (stage_ == kShErase) {
      const size_t n = std::min(build, old_cap_ - pos_);
      for (size_t i = 0; i < n; ++i) {
        const size_t s = pos_ + i;
        if (keys_[s] != kNoKey) {
          table_.erase(keys_[s], [this](uint32_t j) { return keys_[j]; });
          keys_[s] = kNoKey;
        }
        ref_[s] = 0;
      }
      pos_ += n;
      if (pos_ < old_cap_) return Step::More;
      blk_->decommit(rk_, target_ * 8);
      blk_->decommit(rr_, target_);
      const size_t ne = target_ ? index_table_entries(target_) : 0;
      if (ne >= table_.entries()) return finish_shrink();
      if (!ne) {  // nothing is cached: no table
        table_.detach();
        blk_->decommit(rt_[tab_cur_], 0);
        return finish_shrink();
      }
      spare_ = 1 - tab_cur_;
      stage_ = kShCommit;
      return Step::More;
    }
    const size_t ne = index_table_entries(target_);
    if (stage_ == kShCommit) {
      if (!blk_->commit(rt_[spare_], std::min(blk_->committed(rt_[spare_]) + slice, ne * 4))) {
        blk_->decommit(rt_[spare_], 0);
        return finish_shrink();
      }
      if (blk_->committed(rt_[spare_]) < ne * 4) return Step::More;
      spare_table_.attach(reinterpret_cast<uint32_t*>(blk_->at(rt_[spare_])), ne);
      stage_ = kShFill, pos_ = 0;
      return Step::More;
    }
    if (stage_ == kShFill) {
      const size_t n = std::min(slice / 4, ne - pos_);
      std::fill(spare_table_.t + pos_, spare_table_.t + pos_ + n, kNoSlot);
      pos_ += n;
      if (pos_ < ne) return Step::More;
      stage_ = kShBuild, pos_ = 0;
      building_ = true;  // from here the rows are copied: none is taken in until the swap
      return Step::More;
    }
    if (stage_ == kShBuild) {
      const size_t n = std::min(build, cap_ - pos_);
      for (size_t i = 0; i < n; ++i)
        if (keys_[pos_ + i] != kNoKey) spare_table_.insert(keys_[pos_ + i], uint32_t(pos_ + i));
      pos_ += n;
      if (pos_ < cap_) return Step::More;
      table_ = std::move(spare_table_);
      spare_table_ = SlotTable();
      blk_->decommit(rt_[tab_cur_], 0);
      tab_cur_ = spare_;
      building_ = false;
    }
    return finish_shrink();
  }
  // Undoes a grow that failed or was given up: what it committed goes back, the index is as it was.
  void abort_grow() {
    if (op_ != Op::Grow) return;
    blk_->decommit(rk_, cap_ * 8);
    blk_->decommit(rr_, cap_);
    blk_->decommit(rt_[table_.entries() ? 1 - tab_cur_ : tab_cur_], 0);  // the spare table, whatever of it was made
    spare_table_ = SlotTable();
    building_ = false;
    op_ = Op::Idle;
  }

 private:
  enum class Op { Idle, Grow, Shrink };
  enum { kKeys, kRef, kTabCommit, kTabFill, kTabBuild, kFinish, kShErase, kShCommit, kShFill, kShBuild };
  Step finish_shrink() {
    op_ = Op::Idle;
    return Step::Done;
  }
  Step fail() {
    abort_grow();
    return Step::Failed;
  }

  size_t cap_ = 0, hand_ = 0;
  uint64_t* keys_ = nullptr;
  uint8_t* ref_ = nullptr;
  SlotTable table_;
  bool heap_ = true;
  // growable
  Block* blk_ = nullptr;
  int rk_ = 0, rr_ = 0, rt_[2] = {0, 0}, tab_cur_ = 0, spare_ = 0;
  Op op_ = Op::Idle;
  int stage_ = 0;
  size_t target_ = 0, pos_ = 0, filled_ = 0, old_cap_ = 0;
  bool building_ = false;
  SlotTable spare_table_;
};

}  // namespace engram
