// Session-slot memory made once and mapped on demand (docs/engine.md, "Session slots mapped on demand").
//
// A Pool owns B physical chunks of kChunk bytes, all created when it is built and never released while it
// lives, and any number of virtual address ranges ("arrays"), one per per-session array, each reserved
// whole (virtual addresses cost nothing). A slot's part of an array grows from its front, one chunk at a
// time: ensure() takes a chunk from the free list and maps it, release() unmaps every chunk of a slot and
// returns it to the free list. Mapping and unmapping never create or free physical memory, so a request
// never waits for an allocation and the addresses a captured graph holds never change.
//
// Contracts:
//  - Pool code is host only (no CUDA headers): the driver calls are behind Backend, which the engine
//    implements with the CUDA driver's virtual memory calls (vmm.cu, make_cuda_backend) and the unit test
//    with FakeBackend.
//  - A slot's mapped chunks are a prefix of its region of each array: bytes [0, n * kChunk).
//  - ensure() is all or nothing across the arrays: it maps nothing when the free list cannot cover every
//    array's growth, so a failure leaves the slot as it was.
//  - A chunk holds whatever its last user left (a new chunk's content is not defined either): the readers
//    must read only what was written (model.cu ensure_mapped lists why none reads more). poison() fills
//    every chunk it maps with 0xFF, to prove that.
//  - Commitments (reserve) are a host-side account: the sum of every slot's max(reserved, mapped) chunks
//    never exceeds the pool, so a slot that stays within its reservation can always be mapped.
//  - Regions (add_region) are ranges of the same pool that map chunk by chunk, in any order, under a reference count
//    per chunk: the session store's block space (state.cu). A chunk is mapped while any live block overlaps it,
//    stays mapped (idle) when the last block leaves, and goes back to the free list only at region_trim(), which
//    the caller calls once nothing queued can still read it. The store's mapped chunks count against the same
//    budget as the slots' commitments: committed_total() + regions_mapped() never exceeds total(), so a slot that
//    stays within its reservation can always be mapped, and the store maps only chunks no slot has committed.
//  - Chunks are made at the start only: the constructor, and add_chunks() before serving. Nothing is created or
//    destroyed after that.
//  - Not thread safe: one caller (the model's host thread).
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace model::vmm {

constexpr size_t kChunk = 2u << 20;  // the mapping granularity of every array (the driver's minimum on GB10)

// The driver calls a Pool makes. Handles and addresses are opaque 64-bit values.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual bool reserve(size_t bytes, uint64_t* va) = 0;  // a virtual range, aligned to kChunk, nothing behind it
  virtual void release_va(uint64_t va, size_t bytes) = 0;
  virtual bool create(uint64_t* handle) = 0;             // one chunk of physical memory
  virtual void destroy(uint64_t handle) = 0;
  virtual bool map(uint64_t va, uint64_t handle) = 0;    // the chunk at va (one chunk, not yet accessible)
  virtual bool allow(uint64_t va, size_t bytes) = 0;     // read/write access for the device, over mapped chunks
  virtual void unmap(uint64_t va) = 0;                   // one chunk
  virtual void poison(uint64_t va) { (void)va; }         // fill the chunk at va with 0xFF (debug)
  virtual std::string error() const { return {}; }       // the last failure's text
};

// A backend with no GPU: records what is mapped and accessible, so a test can touch an address.
class FakeBackend : public Backend {
 public:
  // max_chunks: create() fails after that many chunks (an allocation failure); fail_map_at: the map() call
  // (counted from 1) that fails, 0 for none
  explicit FakeBackend(size_t max_chunks = SIZE_MAX, size_t fail_map_at = 0) : max_chunks_(max_chunks), fail_map_at_(fail_map_at) {}
  bool reserve(size_t bytes, uint64_t* va) override;
  void release_va(uint64_t, size_t) override {}
  bool create(uint64_t* handle) override;
  void destroy(uint64_t handle) override;
  bool map(uint64_t va, uint64_t handle) override;
  bool allow(uint64_t va, size_t bytes) override;
  void unmap(uint64_t va) override;
  void poison(uint64_t va) override { poisoned_.insert(va); }
  std::string error() const override { return err_; }
  // whether every byte of [addr, addr + len) is in a chunk that is mapped and accessible (what a kernel
  // may touch without a fault)
  bool touch(uint64_t addr, size_t len) const;
  size_t mapped_chunks() const { return mapped_.size(); }
  size_t live_handles() const { return handles_.size(); }
  size_t created_total() const { return created_; }
  size_t poisoned_total() const { return poisoned_.size(); }  // distinct chunk addresses poisoned so far

 private:
  size_t max_chunks_, fail_map_at_, created_ = 0, map_calls_ = 0;
  uint64_t next_va_ = 0x100000000ull, next_handle_ = 1;
  std::set<uint64_t> handles_;
  std::map<uint64_t, uint64_t> mapped_;             // chunk address -> the handle mapped there
  std::set<uint64_t> allowed_, poisoned_;           // chunk addresses
  std::set<uint64_t> in_use_;                       // handles that are mapped somewhere
  std::string err_;
};

class Pool {
 public:
  // Creates `chunks` chunks now (all of them or fewer: created() says how many, ok() whether every one was).
  Pool(std::unique_ptr<Backend> be, size_t chunks);
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  bool ok() const { return created() == want_; }
  size_t created() const { return handles_.size(); }
  size_t total() const { return handles_.size(); }
  size_t free_chunks() const { return free_.size(); }
  size_t used_chunks() const { return handles_.size() - free_.size(); }
  size_t peak_used() const { return peak_; }
  const std::string& error() const { return err_; }
  void set_poison(bool on) { poison_ = on; }

  // One array: nunits regions of unit_stride bytes (a multiple of kChunk) one after the other, reserved
  // whole. Returns its id, or -1 (error()). Arrays come before slots.
  int add_array(size_t unit_stride, int nunits);
  uint64_t base(int array) const { return arrays_[array].base; }
  size_t array_bytes(int array) const { return arrays_[array].stride * (size_t)arrays_[array].nunits; }
  int arrays() const { return (int)arrays_.size(); }

  // A slot: units regions, from region first_unit of every array on. Returns its id.
  int add_slot(int first_unit, int units);
  int slots() const { return (int)slots_.size(); }
  size_t slot_capacity(int slot, int array) const;  // bytes of the slot's region of the array

  // Map the slot's part of array a to hold at least bytes[a] bytes from its start, for every array (bytes
  // has arrays() entries). False, with nothing mapped, when a request is past the region or the free list
  // is short; false with the slot grown as far as it got when the driver fails.
  bool ensure(int slot, const size_t* bytes);
  // Chunks ensure(slot, bytes) would take from the free list.
  size_t growth(int slot, const size_t* bytes) const;
  // Unmap everything of the slot; its chunks go back to the free list; its reservation is dropped.
  void release(int slot);

  size_t mapped(int slot) const { return slots_[slot].mapped; }
  size_t mapped(int slot, int array) const { return slots_[slot].held.size() > (size_t)array ? slots_[slot].held[array].size() : 0; }
  // Commitments: reserve(slot, n) sets the slot's reservation to n chunks when the sum of every slot's
  // max(reserved, mapped), and the chunks the regions have mapped, stays within the pool (else false, nothing
  // changed).
  bool reserve(int slot, size_t chunks);
  size_t reserved(int slot) const { return slots_[slot].reserved; }
  size_t committed(int slot) const { return std::max(slots_[slot].reserved, slots_[slot].mapped); }
  size_t committed_total() const;
  // chunks no slot has committed and no region has mapped
  size_t uncommitted() const {
    const size_t c = committed_total() + regions_mapped_;
    return total() > c ? total() - c : 0;
  }

  // ---- regions (the session store's block space)
  // Makes `n` more chunks now, for the start-up (the pool's ok() then covers them); returns how many were made.
  size_t add_chunks(size_t n);
  // A range of `chunks` chunks of address space, nothing mapped (the address space costs nothing). Returns its id,
  // or -1 (error()).
  int add_region(size_t chunks);
  uint64_t region_base(int r) const { return regions_[r].base; }
  size_t region_chunks(int r) const { return regions_[r].chunks; }
  size_t region_mapped(int r) const { return regions_[r].mapped; }  // chunks mapped (blocks over them, or idle)
  size_t region_idle(int r) const { return regions_[r].idle; }      // mapped with no block over them
  size_t regions_mapped() const { return regions_mapped_; }
  bool region_is_mapped(int r, size_t i) const { return regions_[r].held[i] != kNone; }
  size_t region_refs(int r, size_t i) const { return regions_[r].refs[i]; }
  // Chunks a block of `bytes` at byte offset `off` of the region would map (those of its span not mapped yet).
  size_t region_need(int r, size_t off, size_t bytes) const;
  // A block at [off, off + bytes): maps the chunks of its span that are not mapped (an idle one is taken again
  // as it is) and counts the block on each. False, with nothing changed, when the span is past the region, fewer
  // chunks are uncommitted() than it needs, or the driver fails.
  bool region_acquire(int r, size_t off, size_t bytes);
  // The block at [off, off + bytes) is gone: a chunk no block is left on stays mapped, idle.
  void region_drop(int r, size_t off, size_t bytes);
  // Unmaps every idle chunk of the region (back to the free list); returns how many. The caller guarantees that
  // nothing queued on the device reads them.
  size_t region_trim(int r);

 private:
  struct Array {
    uint64_t base;
    size_t stride;
    int nunits;
  };
  static constexpr uint32_t kNone = UINT32_MAX;
  struct Region {
    uint64_t base = 0;
    size_t chunks = 0, mapped = 0, idle = 0;
    std::vector<uint32_t> held;  // [chunk]: the chunk id mapped there, kNone when none
    std::vector<uint32_t> refs;  // [chunk]: blocks over it
  };
  bool map_region_chunk(Region& g, size_t i);
  void unmap_region_chunk(Region& g, size_t i);
  struct Slot {
    int first_unit, units;
    size_t mapped = 0, reserved = 0;
    std::vector<std::vector<uint32_t>> held;  // [array]: the chunk ids mapped, in order
  };
  uint64_t va_of(const Slot& s, int a, size_t i) const {
    return arrays_[a].base + (size_t)s.first_unit * arrays_[a].stride + i * kChunk;
  }
  std::unique_ptr<Backend> be_;
  size_t want_;
  std::vector<uint64_t> handles_;  // chunk id -> handle
  std::vector<uint32_t> free_;     // chunk ids not mapped, the next to map last
  std::vector<Array> arrays_;
  std::vector<Slot> slots_;
  std::vector<Region> regions_;
  size_t regions_mapped_ = 0;
  size_t peak_ = 0;
  bool poison_ = false;
  std::string err_;
};

// The CUDA driver's virtual memory calls on the current device (vmm.cu): chunks of kChunk from device
// memory; poison() is a cudaMemset to 0xFF that has finished when it returns. Null with error text in *err
// when the device's allocation granularity does not divide kChunk.
std::unique_ptr<Backend> make_cuda_backend(std::string* err);

}  // namespace model::vmm
