#include "model/chunks.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace model::chunks {

namespace {
// host only (no CUDA headers: the allocator's test builds without them), as common/check.h's die
[[noreturn]] void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
  std::abort();
}
}  // namespace

uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

Alloc::Alloc(uint32_t n)
    : n_(n), free_(n > 0 ? n - 1 : 0), lo_(1), bits_((n + 63) / 64, ~0ull), ghost_bits_((n + 63) / 64, 0ull), stamp_(n, 0ull) {
  if (n < 2) die("chunks: a pool of %u numbers (the guard and at least one more)", n);
  bits_[0] &= ~1ull;  // the guard
  if (n % 64) bits_.back() &= (1ull << (n % 64)) - 1;  // no numbers past n
}

bool Alloc::taken(uint32_t c) const { return c > 0 && c < n_ && !(bits_[c >> 6] >> (c & 63) & 1); }

uint32_t Alloc::take() {
  for (size_t w = lo_ >> 6; w < bits_.size(); ++w)
    if (bits_[w]) {
      const uint32_t c = (uint32_t)(w * 64 + __builtin_ctzll(bits_[w]));
      bits_[w] &= ~(1ull << (c & 63));
      --free_;
      lo_ = c + 1;
      digest_ += mix(c);
      return c;
    }
  die("chunks: none of %u free", total());
}

bool Alloc::take(uint32_t n, std::vector<uint32_t>* out) {
  if (n > free_) return false;
  for (uint32_t i = 0; i < n; ++i) out->push_back(take());
  return true;
}

void Alloc::grow(uint32_t n) {
  if (n < n_) die("chunks: a pool of %u numbers cannot shrink to %u", n_, n);
  bits_.resize((n + 63) / 64, 0ull);
  for (uint32_t c = n_; c < n; ++c) bits_[c >> 6] |= 1ull << (c & 63);
  free_ += n - n_;
  if (lo_ > n_) lo_ = n_;
  n_ = n;
}

uint32_t Alloc::first_taken_from(uint32_t c) const {
  for (uint32_t x = std::max<uint32_t>(c, 1); x < n_; ++x)
    if (taken(x)) return x;
  return 0;
}

bool Alloc::shrink(uint32_t n) {
  if (n < 2 || n > n_ || first_taken_from(n)) return false;
  free_ -= n_ - n;
  n_ = n;
  bits_.resize((n + 63) / 64);
  if (n % 64) bits_.back() &= (1ull << (n % 64)) - 1;  // no numbers past n
  if (lo_ > n) lo_ = n;
  return true;
}

void Alloc::give(uint32_t c) {
  if (!taken(c)) die("chunks: number %u given back without being taken", c);
  bits_[c >> 6] |= 1ull << (c & 63);
  ++free_;
  if (c < lo_) lo_ = c;
  digest_ -= mix(c);
}

}  // namespace model::chunks
