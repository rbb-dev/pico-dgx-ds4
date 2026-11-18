// One block of address space with sub-ranges at fixed offsets, each a committed prefix that grows and shrinks while the
// addresses stay: the Engram row cache's slots, keys, reference bits and tables (rowcache.h). The whole block is one
// PROT_NONE, MAP_NORESERVE mapping made once; committing a range makes its next pages readable and writable, asks for huge
// pages and populates them (so the memory is taken now, not when a row first lands), decommitting gives the pages back to
// the system. Host only; one thread owns a block.
#pragma once
#include <sys/mman.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

namespace engram {

constexpr size_t kBlockAlign = size_t(2) << 20;  // sub-ranges start and grow in whole 2 MiB pages

class Block {
 public:
  Block() = default;
  ~Block() {
    if (raw_) munmap(raw_, raw_bytes_);
  }
  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;

  // Address space for ranges of at most `maxima[i]` bytes (each rounded up to 2 MiB). False when the mapping fails.
  bool reserve(const std::vector<size_t>& maxima) {
    size_t total = 0;
    for (const size_t m : maxima) {
      r_.push_back({total, round(m), 0});
      total += round(m);
    }
    raw_bytes_ = total + kBlockAlign;
    raw_ = mmap(nullptr, raw_bytes_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw_ == MAP_FAILED) {
      raw_ = nullptr;
      r_.clear();
      return false;
    }
    const uintptr_t a = (reinterpret_cast<uintptr_t>(raw_) + kBlockAlign - 1) & ~(uintptr_t)(kBlockAlign - 1);
    base_ = reinterpret_cast<uint8_t*>(a);
    total_ = total;
    return true;
  }
  int ranges() const { return (int)r_.size(); }
  uint8_t* at(int r) const { return base_ + r_[r].off; }
  size_t max_of(int r) const { return r_[r].max; }
  size_t committed(int r) const { return r_[r].committed; }
  size_t committed_total() const {
    size_t s = 0;
    for (const R& x : r_) s += x.committed;
    return s;
  }
  const uint8_t* base() const { return base_; }
  size_t span() const { return total_; }

  // The range's prefix committed up to `bytes` (rounded up to 2 MiB, at most the range's maximum): zero pages, readable and
  // writable, huge where the kernel gives them, populated. False, with the prefix as it was, when the kernel refuses.
  bool commit(int r, size_t bytes) {
    R& x = r_[r];
    bytes = round(bytes);
    if (bytes > x.max || bytes <= x.committed) return bytes <= x.committed;
    uint8_t* lo = base_ + x.off + x.committed;
    const size_t len = bytes - x.committed;
    if (mprotect(lo, len, PROT_READ | PROT_WRITE) != 0) return false;
    madvise(lo, len, MADV_HUGEPAGE);
    if (madvise(lo, len, MADV_POPULATE_WRITE) != 0) {
      if (errno != EINVAL) {  // the kernel could not give the pages (no memory, a signal): the prefix as it was, the caller fails
        madvise(lo, len, MADV_DONTNEED);
        mprotect(lo, len, PROT_NONE);
        return false;
      }
      for (size_t i = 0; i < len; i += 4096) lo[i] = 0;  // a kernel before MADV_POPULATE_WRITE (5.14)
    }
    x.committed = bytes;
    return true;
  }
  // The range's pages above `bytes` (rounded up to 2 MiB) go back to the system; the addresses stay reserved.
  void decommit(int r, size_t bytes) {
    R& x = r_[r];
    bytes = round(bytes);
    if (bytes >= x.committed) return;
    uint8_t* lo = base_ + x.off + bytes;
    const size_t len = x.committed - bytes;
    madvise(lo, len, MADV_DONTNEED);
    mprotect(lo, len, PROT_NONE);
    x.committed = bytes;
  }

  // The block's resident bytes (Rss) and those in huge pages (AnonHugePages), from /proc/self/smaps: the kernel's own count
  // of the pages behind the block, independent of what this class thinks it committed. False when the file cannot be read.
  bool resident(uint64_t* rss, uint64_t* huge) const { return resident_of(base_, total_, rss, huge); }
  static bool resident_of(const uint8_t* lo, size_t len, uint64_t* rss, uint64_t* huge) {
    *rss = *huge = 0;
    FILE* f = std::fopen("/proc/self/smaps", "r");
    if (!f) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(lo), b = a + len;
    bool in = false;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
      unsigned long long s, e;
      if (std::sscanf(line, "%llx-%llx ", &s, &e) == 2 && std::strchr(line, '-') && line[0] != 'R' && line[0] != 'A') {
        in = s >= a && s < b;
        continue;
      }
      unsigned long long kb;
      if (in && std::sscanf(line, "Rss: %llu kB", &kb) == 1) *rss += kb << 10;
      else if (in && std::sscanf(line, "AnonHugePages: %llu kB", &kb) == 1) *huge += kb << 10;
    }
    std::fclose(f);
    return true;
  }

 private:
  struct R {
    size_t off, max, committed;
  };
  static size_t round(size_t n) { return (n + kBlockAlign - 1) / kBlockAlign * kBlockAlign; }
  void* raw_ = nullptr;
  size_t raw_bytes_ = 0, total_ = 0;
  uint8_t* base_ = nullptr;
  std::vector<R> r_;
};

}  // namespace engram
