// The slot memory pool (vmm.h): chunks made once, mapped into slots' regions on demand. Host only.
#include "model/vmm.h"

#include <cstdio>
#include <utility>

namespace model::vmm {

// ---- FakeBackend ------------------------------------------------------------------------------------

bool FakeBackend::reserve(size_t bytes, uint64_t* va) {
  *va = next_va_;
  next_va_ += (bytes + kChunk - 1) / kChunk * kChunk + kChunk;  // a guard chunk between ranges: never mapped
  return true;
}

bool FakeBackend::create(uint64_t* handle) {
  if (created_ >= max_chunks_) {
    err_ = "fake: out of memory";
    return false;
  }
  ++created_;
  *handle = next_handle_++;
  handles_.insert(*handle);
  return true;
}

void FakeBackend::destroy(uint64_t handle) {
  handles_.erase(handle);
  in_use_.erase(handle);
}

bool FakeBackend::map(uint64_t va, uint64_t handle) {
  if (fail_map_at_ && ++map_calls_ == fail_map_at_) {
    err_ = "fake: map failed";
    return false;
  }
  if (!handles_.count(handle) || in_use_.count(handle) || mapped_.count(va) || va % kChunk) {
    err_ = "fake: bad map";
    return false;
  }
  in_use_.insert(handle);
  mapped_.emplace(va, handle);
  return true;
}

bool FakeBackend::allow(uint64_t va, size_t bytes) {
  for (uint64_t a = va; a < va + bytes; a += kChunk)
    if (!mapped_.count(a)) {
      err_ = "fake: access over an unmapped chunk";
      return false;
    }
  for (uint64_t a = va; a < va + bytes; a += kChunk) allowed_.insert(a);
  return true;
}

void FakeBackend::unmap(uint64_t va) {
  if (const auto it = mapped_.find(va); it != mapped_.end()) in_use_.erase(it->second);  // the handle maps again
  mapped_.erase(va);
  allowed_.erase(va);
  poisoned_.erase(va);
}

bool FakeBackend::touch(uint64_t addr, size_t len) const {
  if (!len) return true;
  for (uint64_t a = addr / kChunk * kChunk; a < addr + len; a += kChunk)
    if (!mapped_.count(a) || !allowed_.count(a)) return false;
  return true;
}

// ---- Pool -------------------------------------------------------------------------------------------

Pool::Pool(std::unique_ptr<Backend> be, size_t chunks) : be_(std::move(be)), want_(chunks) {
  handles_.reserve(chunks);
  free_.reserve(chunks);
  for (size_t i = 0; i < chunks; ++i) {
    uint64_t h = 0;
    if (!be_->create(&h)) {
      err_ = "chunk " + std::to_string(i) + " of " + std::to_string(chunks) + ": " + be_->error();
      break;
    }
    handles_.push_back(h);
  }
  for (size_t i = handles_.size(); i-- > 0;) free_.push_back((uint32_t)i);  // chunk 0 is the first to map
}

Pool::~Pool() {
  for (int s = 0; s < (int)slots_.size(); ++s) release(s);
  for (Region& g : regions_) {
    for (size_t i = 0; i < g.chunks; ++i)
      if (g.held[i] != kNone) unmap_region_chunk(g, i);
    be_->release_va(g.base, g.chunks * kChunk);
  }
  for (const uint64_t h : handles_) be_->destroy(h);
  for (const Array& a : arrays_) be_->release_va(a.base, a.stride * (size_t)a.nunits);
}

int Pool::add_array(size_t unit_stride, int nunits) {
  if (!slots_.empty() || unit_stride == 0 || unit_stride % kChunk || nunits < 1) {
    err_ = "add_array: stride " + std::to_string(unit_stride) + ", " + std::to_string(nunits) + " units, after " +
           std::to_string(slots_.size()) + " slots";
    return -1;
  }
  uint64_t va = 0;
  if (!be_->reserve(unit_stride * (size_t)nunits, &va)) {
    err_ = "reserve of " + std::to_string(unit_stride * (size_t)nunits) + " bytes: " + be_->error();
    return -1;
  }
  arrays_.push_back({va, unit_stride, nunits});
  return (int)arrays_.size() - 1;
}

int Pool::add_slot(int first_unit, int units) {
  for (const Array& a : arrays_)
    if (first_unit < 0 || units < 1 || first_unit + units > a.nunits) return -1;
  Slot s;
  s.first_unit = first_unit, s.units = units;
  s.held.resize(arrays_.size());
  slots_.push_back(std::move(s));
  return (int)slots_.size() - 1;
}

size_t Pool::slot_capacity(int slot, int array) const { return arrays_[array].stride * (size_t)slots_[slot].units; }

size_t Pool::growth(int slot, const size_t* bytes) const {
  const Slot& s = slots_[slot];
  size_t n = 0;
  for (int a = 0; a < (int)arrays_.size(); ++a) {
    const size_t want = (bytes[a] + kChunk - 1) / kChunk;
    if (want > s.held[a].size()) n += want - s.held[a].size();
  }
  return n;
}

bool Pool::ensure(int slot, const size_t* bytes) {
  Slot& s = slots_[slot];
  for (int a = 0; a < (int)arrays_.size(); ++a)
    if (bytes[a] > slot_capacity(slot, a)) {
      err_ = "slot " + std::to_string(slot) + " array " + std::to_string(a) + ": " + std::to_string(bytes[a]) +
             " bytes past its region of " + std::to_string(slot_capacity(slot, a));
      return false;
    }
  const size_t need = growth(slot, bytes);
  if (need > free_.size()) {
    err_ = "slot " + std::to_string(slot) + " needs " + std::to_string(need) + " chunks, " + std::to_string(free_.size()) +
           " free";
    return false;
  }
  for (int a = 0; a < (int)arrays_.size(); ++a) {
    const size_t want = (bytes[a] + kChunk - 1) / kChunk;
    std::vector<uint32_t>& held = s.held[a];
    const size_t first = held.size();
    if (want <= first) continue;
    bool ok = true;
    for (size_t i = first; i < want && ok; ++i) {
      const uint32_t id = free_.back();
      if (!be_->map(va_of(s, a, i), handles_[id])) {
        err_ = "map of slot " + std::to_string(slot) + " array " + std::to_string(a) + " chunk " + std::to_string(i) + ": " +
               be_->error();
        ok = false;
        break;
      }
      free_.pop_back();
      held.push_back(id);
      ++s.mapped;
    }
    const size_t got = held.size();
    if (got > first && !be_->allow(va_of(s, a, first), (got - first) * kChunk)) {
      err_ = "access of slot " + std::to_string(slot) + " array " + std::to_string(a) + ": " + be_->error();
      while (held.size() > first) {  // not usable: back to the free list
        be_->unmap(va_of(s, a, held.size() - 1));
        free_.push_back(held.back());
        held.pop_back();
        --s.mapped;
      }
      return false;
    }
    if (poison_)
      for (size_t i = first; i < got; ++i) be_->poison(va_of(s, a, i));
    if (!ok) {
      peak_ = std::max(peak_, used_chunks());
      return false;
    }
  }
  peak_ = std::max(peak_, used_chunks());
  return true;
}

void Pool::release(int slot) {
  Slot& s = slots_[slot];
  for (int a = 0; a < (int)arrays_.size(); ++a) {
    std::vector<uint32_t>& held = s.held[a];
    for (size_t i = held.size(); i-- > 0;) {
      be_->unmap(va_of(s, a, i));
      free_.push_back(held[i]);
    }
    held.clear();
  }
  s.mapped = 0, s.reserved = 0;
}

size_t Pool::committed_total() const {
  size_t n = 0;
  for (int s = 0; s < (int)slots_.size(); ++s) n += committed(s);
  return n;
}

bool Pool::reserve(int slot, size_t chunks) {
  Slot& s = slots_[slot];
  const size_t others = committed_total() - committed(slot);
  if (others + std::max(chunks, s.mapped) + regions_mapped_ > total()) return false;
  s.reserved = chunks;
  return true;
}

// ---- regions ----------------------------------------------------------------------------------------

size_t Pool::add_chunks(size_t n) {
  want_ += n;
  std::vector<uint32_t> made;
  made.reserve(n);
  handles_.reserve(handles_.size() + n);
  for (size_t i = 0; i < n; ++i) {
    uint64_t h = 0;
    if (!be_->create(&h)) {
      err_ = "chunk " + std::to_string(handles_.size()) + " of " + std::to_string(want_) + ": " + be_->error();
      break;
    }
    handles_.push_back(h);
    made.push_back((uint32_t)handles_.size() - 1);
  }
  // the new chunks map after the ones there are: the lowest of them first
  free_.insert(free_.begin(), made.rbegin(), made.rend());
  return made.size();
}

int Pool::add_region(size_t chunks) {
  if (chunks == 0) {
    err_ = "add_region: no chunks";
    return -1;
  }
  uint64_t va = 0;
  if (!be_->reserve(chunks * kChunk, &va)) {
    err_ = "reserve of " + std::to_string(chunks * kChunk) + " bytes: " + be_->error();
    return -1;
  }
  Region g;
  g.base = va, g.chunks = chunks;
  g.held.assign(chunks, kNone);
  g.refs.assign(chunks, 0);
  regions_.push_back(std::move(g));
  return (int)regions_.size() - 1;
}

bool Pool::map_region_chunk(Region& g, size_t i) {
  if (free_.empty()) {
    err_ = "region chunk " + std::to_string(i) + ": no chunk free";
    return false;
  }
  const uint64_t va = g.base + i * kChunk;
  const uint32_t id = free_.back();
  if (!be_->map(va, handles_[id])) {
    err_ = "map of region chunk " + std::to_string(i) + ": " + be_->error();
    return false;
  }
  if (!be_->allow(va, kChunk)) {
    err_ = "access of region chunk " + std::to_string(i) + ": " + be_->error();
    be_->unmap(va);
    return false;
  }
  free_.pop_back();
  g.held[i] = id;
  ++g.mapped;
  ++regions_mapped_;
  peak_ = std::max(peak_, used_chunks());
  return true;
}

void Pool::unmap_region_chunk(Region& g, size_t i) {
  be_->unmap(g.base + i * kChunk);
  free_.push_back(g.held[i]);
  g.held[i] = kNone;
  --g.mapped;
  --regions_mapped_;
}

size_t Pool::region_need(int r, size_t off, size_t bytes) const {
  const Region& g = regions_[r];
  if (!bytes) return 0;
  size_t n = 0;
  const size_t last = std::min((off + bytes - 1) / kChunk, g.chunks - 1);
  for (size_t i = off / kChunk; i <= last; ++i) n += g.held[i] == kNone;
  return n;
}

bool Pool::region_acquire(int r, size_t off, size_t bytes) {
  Region& g = regions_[r];
  if (!bytes) return true;
  const size_t first = off / kChunk, last = (off + bytes - 1) / kChunk;
  if (last >= g.chunks) {
    err_ = "region " + std::to_string(r) + ": a block at " + std::to_string(off) + " of " + std::to_string(bytes) +
           " bytes is past its " + std::to_string(g.chunks) + " chunks";
    return false;
  }
  const size_t need = region_need(r, off, bytes);
  if (need > uncommitted()) {
    err_ = "region " + std::to_string(r) + ": a block needs " + std::to_string(need) + " chunks, " +
           std::to_string(uncommitted()) + " uncommitted";
    return false;
  }
  size_t revived = 0;
  for (size_t i = first; i <= last; ++i) revived += g.held[i] != kNone && g.refs[i] == 0;
  std::vector<size_t> fresh;
  for (size_t i = first; i <= last; ++i)
    if (g.held[i] == kNone) {
      if (!map_region_chunk(g, i)) {
        for (const size_t j : fresh) unmap_region_chunk(g, j);  // all or nothing
        return false;
      }
      fresh.push_back(i);
    }
  for (size_t i = first; i <= last; ++i) ++g.refs[i];
  g.idle -= revived;
  return true;
}

void Pool::region_drop(int r, size_t off, size_t bytes) {
  Region& g = regions_[r];
  if (!bytes) return;
  const size_t last = std::min((off + bytes - 1) / kChunk, g.chunks - 1);
  for (size_t i = off / kChunk; i <= last; ++i) {
    if (g.refs[i] == 0) continue;  // a block dropped twice: nothing to take off
    if (--g.refs[i] == 0 && g.held[i] != kNone) ++g.idle;
  }
}

size_t Pool::region_trim(int r) {
  Region& g = regions_[r];
  size_t n = 0;
  if (!g.idle) return 0;
  for (size_t i = 0; i < g.chunks; ++i)
    if (g.held[i] != kNone && g.refs[i] == 0) unmap_region_chunk(g, i), ++n;
  g.idle = 0;
  return n;
}

}  // namespace model::vmm
