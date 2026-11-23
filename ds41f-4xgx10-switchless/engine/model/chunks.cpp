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
bool Alloc::is_ghost(uint32_t c) const { return c < n_ && (ghost_bits_[c >> 6] >> (c & 63) & 1); }
uint64_t Alloc::stamp(uint32_t c) const { return is_ghost(c) ? stamp_[c] : 0; }

namespace {
uint64_t ghost_term(uint32_t c, uint64_t stamp) { return mix(c) + 3 * mix(stamp ^ mix(c)); }
}  // namespace

// The number take() hands out. The lowest free number that is not a ghost is the first set bit of free-and-not-ghost; the
// oldest ghost below the zone is the first of by_stamp_ under it.
uint32_t Alloc::pick(bool below_zone_only, bool* found) const {
  *found = true;
  uint32_t plain = 0;
  for (size_t w = lo_ >> 6; w < bits_.size(); ++w)
    if (const uint64_t m = bits_[w] & ~ghost_bits_[w]) {
      plain = (uint32_t)(w * 64 + __builtin_ctzll(m));
      break;
    }
  if (plain && plain < zone_) return plain;                       // 1
  for (const auto& g : by_stamp_)
    if (g.second < zone_) return g.second;                        // 2
  if (below_zone_only) {
    *found = false;
    return 0;
  }
  if (plain) return plain;                                        // 3
  if (!by_stamp_.empty()) return by_stamp_.begin()->second;       // 4
  *found = false;
  return 0;
}

void Alloc::claim(uint32_t c, bool lowest) {
  if (is_ghost(c)) {
    ghost_bits_[c >> 6] &= ~(1ull << (c & 63));
    by_stamp_.erase({stamp_[c], c});
    ghost_digest_ -= ghost_term(c, stamp_[c]);
    stamp_[c] = 0;
    --ghosts_;
  }
  bits_[c >> 6] &= ~(1ull << (c & 63));
  --free_;
  if (lowest || c == lo_) lo_ = c + 1;
  digest_ += mix(c);
}

uint32_t Alloc::take() {
  bool found;
  const uint32_t c = pick(false, &found);
  if (!found) die("chunks: none of %u free", total());
  claim(c, ghosts_ == 0);  // with no ghost the number is the lowest free one
  return c;
}

bool Alloc::take_below_zone(uint32_t* c) {
  bool found;
  const uint32_t x = pick(true, &found);
  if (!found) return false;
  claim(x, ghosts_ == 0);
  *c = x;
  return true;
}

uint32_t Alloc::free_below_zone() const {
  const uint32_t top = std::min(zone_, n_);
  uint32_t k = 0;
  for (uint32_t w = 0; (w << 6) < top; ++w) {
    uint64_t m = bits_[w];
    if (top - (w << 6) < 64) m &= (1ull << (top - (w << 6))) - 1;
    k += (uint32_t)__builtin_popcountll(m);
  }
  return k;
}

std::vector<uint32_t> Alloc::ghosts_from(uint32_t c) const {
  std::vector<uint32_t> v;
  for (uint32_t x = c; x < n_; ++x)
    if (is_ghost(x)) v.push_back(x);
  return v;
}

uint64_t Alloc::split_digest() const {
  if (!ghosts_ && zone_ == kNoZone) return 0;
  return ghost_digest_ + mix(0x5a0e000000000000ull + zone_) + mix(ghosts_);
}

bool Alloc::take(uint32_t n, std::vector<uint32_t>* out) {
  if (n > free_) return false;
  for (uint32_t i = 0; i < n; ++i) out->push_back(take());
  return true;
}

void Alloc::grow(uint32_t n) {
  if (n < n_) die("chunks: a pool of %u numbers cannot shrink to %u", n_, n);
  bits_.resize((n + 63) / 64, 0ull);
  ghost_bits_.resize((n + 63) / 64, 0ull);
  stamp_.resize(n, 0ull);
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
  for (const uint32_t g : ghosts_from(n)) {  // a ghost leaves with its number
    ghost_digest_ -= ghost_term(g, stamp_[g]);
    by_stamp_.erase({stamp_[g], g});
    stamp_[g] = 0;
    --ghosts_;
  }
  free_ -= n_ - n;
  n_ = n;
  bits_.resize((n + 63) / 64);
  ghost_bits_.resize((n + 63) / 64);
  stamp_.resize(n);
  if (n % 64) bits_.back() &= (1ull << (n % 64)) - 1;  // no numbers past n
  if (n % 64) ghost_bits_.back() &= (1ull << (n % 64)) - 1;
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

void Alloc::give_ghost(uint32_t c, uint64_t stamp) {
  give(c);
  ghost_bits_[c >> 6] |= 1ull << (c & 63);
  stamp_[c] = stamp;
  by_stamp_.insert({stamp, c});
  ghost_digest_ += ghost_term(c, stamp);
  ++ghosts_;
}

void Alloc::unghost(uint32_t c) {
  if (!is_ghost(c)) return;
  ghost_bits_[c >> 6] &= ~(1ull << (c & 63));
  by_stamp_.erase({stamp_[c], c});
  ghost_digest_ -= ghost_term(c, stamp_[c]);
  stamp_[c] = 0;
  --ghosts_;
}

void Alloc::revive(uint32_t c) {
  if (!is_ghost(c)) die("chunks: number %u revived without being a ghost", c);
  claim(c, false);
}

// ---- SlotMap ---------------------------------------------------------------------------------------------------------------

void SlotMap::open(uint32_t c) {
  if (!nums_.emplace(c, Num{}).second) die("chunks: number %u opened for quarters twice", c);
  roomy_.insert(c);
}

uint32_t SlotMap::take_slot() {
  if (roomy_.empty()) die("chunks: no quarter slot free");
  const uint32_t c = *roomy_.begin();
  Num& n = nums_.at(c);
  int q = 0;
  while (n.refs[q]) ++q;
  n.refs[q] = 1;
  ++used_;
  if (!room(n)) roomy_.erase(c);
  return slot(c, q);
}

void SlotMap::ref(uint32_t s) {
  const auto it = nums_.find(number(s));
  if (it == nums_.end() || !it->second.refs[quarter(s)]) die("chunks: slot %u referenced without a holder", s);
  ++it->second.refs[quarter(s)];
}

uint32_t SlotMap::unref(uint32_t s, uint32_t* freed) {
  *freed = 0;
  const auto it = nums_.find(number(s));
  if (it == nums_.end() || !it->second.refs[quarter(s)]) die("chunks: slot %u released without a holder", s);
  Num& n = it->second;
  const uint32_t left = --n.refs[quarter(s)];
  if (left) return left;
  --used_;
  if (!n.refs[0] && !n.refs[1] && !n.refs[2] && !n.refs[3]) {
    *freed = number(s);
    roomy_.erase(number(s));
    nums_.erase(it);
  } else {
    roomy_.insert(number(s));
  }
  return 0;
}

uint32_t SlotMap::refs(uint32_t s) const {
  const auto it = nums_.find(number(s));
  return it == nums_.end() ? 0 : it->second.refs[quarter(s)];
}

std::vector<uint32_t> SlotMap::open_from(uint32_t c) const {
  std::vector<uint32_t> v;
  for (auto it = nums_.lower_bound(c); it != nums_.end(); ++it) v.push_back(it->first);
  return v;
}

uint64_t SlotMap::digest() const {
  uint64_t d = 0;
  for (const auto& [c, n] : nums_) {
    d += mix(c | 1ull << 40);
    for (int q = 0; q < 4; ++q)
      if (n.refs[q]) d += mix((uint64_t)slot(c, q) << 20 ^ n.refs[q]);
  }
  return d;
}

}  // namespace model::chunks
