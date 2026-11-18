// The Engram cache's growable block (block.h, cache.h, rowcache.h): commit and decommit change the block's resident pages
// by the bytes asked, a cache grows and shrinks while rows are served with every surviving row found in its slot with its
// bytes, rows are not taken in while the table is rebuilt, the slots a shrink drops are gone, and the sizes lend.h plans
// with are the sizes the index takes. Correctness only: no timing. Host only.
//   engram_block_test
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <cstring>
#include <vector>

#include "engram/rowcache.h"
#include "model/lend.h"

using namespace engram;

static int fails = 0;
#define CHECK(c, ...) do { const bool ok_ = (c); std::printf("%s ", ok_ ? "ok  " : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); fails += !ok_; } while (0)

static void row_of(uint64_t key, uint8_t* r) {
  uint64_t x = key * 0x9E3779B97F4A7C15ull + 1;
  for (int i = 0; i < kRowBytes; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; r[i] = (uint8_t)x; }
}

// runs the resize to its end; returns the steps taken
static int finish(RowCache& c, Step* last = nullptr) {
  int n = 0;
  Step st;
  do { st = c.resize_step(size_t(4) << 20, 4096); ++n; } while (st == Step::More && n < 1000000);
  if (last) *last = st;
  return n;
}

// every key in `keys` is found with its bytes, or (when `absent`) none is found
static size_t found(RowCache& c, const std::vector<uint64_t>& keys) {
  size_t n = 0;
  uint8_t want[kRowBytes];
  for (const uint64_t k : keys) {
    const uint8_t* p = c.find(k);
    if (!p) continue;
    row_of(k, want);
    if (memcmp(p, want, kRowBytes) == 0) ++n;
  }
  return n;
}

int main() {
  // the block: pages follow the commit
  {
    Block b;
    CHECK(b.reserve({size_t(64) << 20, size_t(32) << 20}), "a block of two ranges reserved");
    uint64_t rss0, huge0, rss1, huge1, rss2, huge2;
    b.resident(&rss0, &huge0);
    CHECK(b.commit(0, size_t(40) << 20), "commit 40 MiB of range 0");
    b.resident(&rss1, &huge1);
    CHECK(rss1 >= rss0 + (40u << 20) - (2u << 20) && rss1 <= rss0 + (40u << 20) + (2u << 20), "the block's resident pages rose by 40 MiB (%llu MiB)",
          (unsigned long long)((rss1 - rss0) >> 20));
    std::printf("      %llu MiB of it in huge pages\n", (unsigned long long)(huge1 >> 20));
    b.at(0)[(size_t(40) << 20) - 1] = 7;
    b.decommit(0, size_t(10) << 20);
    b.resident(&rss2, &huge2);
    CHECK(rss1 - rss2 >= (30u << 20) - (2u << 20) && b.committed(0) == (10u << 20), "decommit to 10 MiB: pages fell by 30 MiB (%llu MiB)", (unsigned long long)((rss1 - rss2) >> 20));
    CHECK(!b.commit(0, size_t(65) << 20), "a commit past the range's maximum is refused");
    CHECK(b.commit(1, 1), "a one byte commit rounds up to a page");
    CHECK(b.committed(1) == (2u << 20), "and the range holds 2 MiB");
  }

  // the sizes the plan uses are the sizes the index takes
  for (const size_t slots : {size_t(1), size_t(1000), size_t(8070000), size_t(16268815), size_t(27000000)})
    CHECK(model::lend::index_bytes(slots) == index_bytes_for(slots) && model::lend::table_entries(slots) == index_table_entries(slots),
          "lend.h's index bytes for %zu slots equal the cache's", slots);

  // a cache that grows, shrinks and grows with rows in it
  {
    RowCache c;
    CHECK(c.init_growable(size_t(8) << 20), "an empty growable cache over 8 MiB of slots (%zu slots)", c.max_slots());
    CHECK(c.find(1) == nullptr && !c.insert(1, (const uint8_t*)memset(alloca(kRowBytes), 0, kRowBytes)), "an empty cache finds and keeps nothing");
    const size_t s1 = 1000, s2 = 5000, s3 = 300;
    CHECK(c.resize_begin(s1), "grow from nothing to %zu slots", s1);
    Step st;
    finish(c, &st);
    CHECK(st == Step::Done && c.slots() == s1 && !c.resizing(), "done: %zu slots", c.slots());
    std::vector<uint64_t> k1;
    uint8_t rec[kRowBytes];
    for (uint64_t i = 0; i < s1; ++i) {
      row_of(1000 + i, rec);
      c.insert(1000 + i, rec);
      k1.push_back(1000 + i);
    }
    CHECK(found(c, k1) == s1, "all %zu rows cached and found", s1);
    // grow: the table changes size (2,048 -> 16,384 entries) and is rebuilt beside the old one
    const size_t te0 = c.idx.table_entries();
    CHECK(c.resize_begin(s2), "grow to %zu slots", s2);
    int steps = 0;
    bool refused_while_building = false, found_while_building = true;
    do {
      st = c.resize_step(size_t(1) << 20, 256);
      ++steps;
      if (c.idx.rebuilding()) {  // the rows are being copied: none is taken in, the old index still answers
        bool evicted;
        row_of(777777, rec);
        const uint32_t s = c.idx.insert(777777, &evicted);
        refused_while_building |= s == kNoSlot && !evicted;
        found_while_building &= found(c, k1) == s1;
      }
    } while (st == Step::More && steps < 100000);
    CHECK(st == Step::Done && c.slots() == s2, "grown to %zu slots in %d steps", c.slots(), steps);
    CHECK(c.idx.table_entries() == index_table_entries(s2) && c.idx.table_entries() > te0, "the table went from %zu to %zu entries", te0, c.idx.table_entries());
    CHECK(refused_while_building, "no row was taken in while the table was rebuilt");
    CHECK(found_while_building, "every cached row was found throughout the rebuild");
    CHECK(found(c, k1) == s1, "every row cached before the grow is found with its bytes after it");
    std::vector<uint64_t> k2;
    bool no_evict = true;
    for (uint64_t i = 0; i < s2 - s1; ++i) {
      row_of(900000 + i, rec);
      no_evict &= !c.insert(900000 + i, rec);
      k2.push_back(900000 + i);
    }
    CHECK(no_evict && found(c, k2) == s2 - s1 && found(c, k1) == s1, "the new slots fill before anything is evicted: %zu rows, all found", s1 + k2.size());
    // shrink: the slots above leave
    CHECK(c.resize_begin(s3), "shrink to %zu slots", s3);
    finish(c, &st);
    CHECK(st == Step::Done && c.slots() == s3, "shrunk to %zu slots", c.slots());
    std::vector<uint64_t> all = k1;
    all.insert(all.end(), k2.begin(), k2.end());
    const size_t kept = found(c, all);
    CHECK(kept <= s3 && kept > 0, "at most the %zu slots' rows are found after the shrink (%zu), none with wrong bytes", s3, kept);
    size_t wrong = 0;
    for (const uint64_t k : all) {
      const uint8_t* p = c.find(k);
      uint8_t want[kRowBytes];
      row_of(k, want);
      if (p && memcmp(p, want, kRowBytes)) ++wrong;
    }
    CHECK(wrong == 0, "no row found has other bytes than it was given");
    // the cache still works: fill and evict
    size_t evictions = 0;
    for (uint64_t i = 0; i < 3 * s3; ++i) {
      row_of(5000000 + i, rec);
      evictions += c.insert(5000000 + i, rec);
    }
    CHECK(evictions > 0, "after the shrink the cache holds %zu rows and evicts (%zu evictions)", s3, evictions);
    // and grows again from where it is
    CHECK(c.resize_begin(s2), "grow again to %zu slots", s2);
    finish(c, &st);
    CHECK(st == Step::Done && c.slots() == s2, "grown again");
    CHECK(!c.resize_begin(s2) && !c.resize_begin(c.max_slots() + 1), "the same size and a size past the maximum are refused");
  }

  // the index's table follows the cache's size down as well as up: after a shrink it is the table the smaller size needs, the
  // block holds what the smaller size holds, and a cache shrunk to nothing holds no page of it
  {
    const size_t big = 4000000, small = 1000000;
    const size_t region_bytes = (big + 100000) * kRowBytes;
    void* region = mmap(nullptr, region_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    RowCache c;
    CHECK(region != MAP_FAILED && c.init_growable(region_bytes, static_cast<uint8_t*>(region), region_bytes), "a cache over a region of %zu slots", region_bytes / kRowBytes);
    Step st;
    uint8_t rec[kRowBytes];
    c.resize_begin(small);
    finish(c, &st);
    for (uint64_t i = 0; i < 2000; ++i) { row_of(i, rec); c.insert(i, rec); }
    const size_t held_small = c.block().committed_total(), te_small = c.idx.table_entries();
    CHECK(st == Step::Done && te_small == index_table_entries(small), "grown to %zu slots: table %zu entries, block %zu MiB", small, te_small, held_small >> 20);
    c.resize_begin(big);
    finish(c, &st);
    CHECK(st == Step::Done && c.idx.table_entries() == index_table_entries(big) && c.block().committed_total() > held_small + (30u << 20),
          "grown to %zu slots: table %zu entries, block %zu MiB", big, c.idx.table_entries(), c.block().committed_total() >> 20);
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 2000; ++i) keys.push_back(i);
    CHECK(c.resize_begin(small), "shrink back to %zu slots", small);
    int steps = 0;
    bool refused = false, served = true;
    do {
      st = c.resize_step(size_t(1) << 20, 65536);
      ++steps;
      if (c.idx.rebuilding()) {
        bool evicted;
        refused |= c.idx.insert(777777, &evicted) == kNoSlot && !evicted;
        served &= found(c, keys) == keys.size();
      }
    } while (st == Step::More && steps < 100000);
    CHECK(st == Step::Done && c.slots() == small, "shrunk to %zu slots in %d steps", c.slots(), steps);
    CHECK(c.idx.table_entries() == te_small, "the table is the %zu entries the smaller size needs (%zu)", te_small, c.idx.table_entries());
    CHECK(c.block().committed_total() == held_small, "the block holds what it held at the smaller size: %zu MiB (was %zu MiB)", c.block().committed_total() >> 20, held_small >> 20);
    CHECK(found(c, keys) == keys.size() && served, "every row below the cut is found after the shrink and throughout the table's rebuild");
    CHECK(refused, "no row was taken in while the smaller table was built");
    // to nothing: no page of the block stays, and the cache grows again from nothing
    CHECK(c.resize_begin(0), "shrink to nothing");
    finish(c, &st);
    CHECK(st == Step::Done && c.slots() == 0 && c.block().committed_total() == 0 && c.idx.table_entries() == 0,
          "a cache of 0 slots: %zu bytes of the block committed, table %zu entries", c.block().committed_total(), c.idx.table_entries());
    uint64_t rss, huge;
    c.block().resident(&rss, &huge);
    CHECK(rss == 0, "no resident page behind the block (%llu bytes)", (unsigned long long)rss);
    CHECK(c.find(5) == nullptr, "a cache of 0 slots finds nothing");
    c.resize_begin(small);
    finish(c, &st);
    for (uint64_t i = 0; i < 2000; ++i) { row_of(i, rec); c.insert(i, rec); }
    CHECK(st == Step::Done && c.block().committed_total() == held_small && found(c, keys) == keys.size(), "grown again from nothing: the same %zu MiB, rows cached and found", held_small >> 20);
    munmap(region, region_bytes);
  }

  // the pages follow a resize: a cache of 200,000 slots holds about 55 MiB, one of 1,000 slots about 8 MiB less than that
  {
    RowCache c;
    c.init_growable(size_t(64) << 20);
    Step st;
    uint64_t r0, h0, r1, h1, r2, h2;
    c.block().resident(&r0, &h0);
    c.resize_begin(200000);
    finish(c, &st);
    c.block().resident(&r1, &h1);
    const uint64_t want = c.ordinary_bytes(200000) + 200000 * 9;
    CHECK(st == Step::Done && r1 - r0 + (8u << 20) >= want && r1 - r0 <= want + (16u << 20), "grown to 200,000 slots: %llu MiB resident for %llu MiB of slots and index (%llu MiB in huge pages)",
          (unsigned long long)((r1 - r0) >> 20), (unsigned long long)(want >> 20), (unsigned long long)(h1 >> 20));
    c.resize_begin(1000);
    finish(c, &st);
    c.block().resident(&r2, &h2);
    CHECK(st == Step::Done && r1 - r2 + (8u << 20) >= c.ordinary_bytes(199000), "shrunk to 1,000 slots: %llu MiB given back", (unsigned long long)((r1 - r2) >> 20));
  }

  // a cache whose first slots are in a region keeps the numbering across a resize
  {
    RowCache c;
    std::vector<uint8_t> region(300 * kRowBytes);
    c.init_growable(size_t(2) << 20, region.data(), region.size());
    CHECK(c.split == 300, "the region's %zu slots", c.split);
    c.resize_begin(200);
    Step st;
    finish(c, &st);
    uint8_t rec[kRowBytes];
    row_of(42, rec);
    c.insert(42, rec);
    CHECK(c.find(42) == region.data(), "a cache smaller than its region is all in it");
    c.resize_begin(2000);
    finish(c, &st);
    CHECK(st == Step::Done && c.find(42) == region.data() && c.slot(300) == c.block().at(0), "grown past the region: slot 0 stays, slot 300 starts the ordinary slots");
  }

  // the control: a table copied without rehashing does not find its keys
  {
    std::vector<uint64_t> keys(1000);
    for (size_t i = 0; i < keys.size(); ++i) keys[i] = 31 * i + 7;
    SlotTable small;
    small.init(2048);
    for (size_t i = 0; i < keys.size(); ++i) small.insert(keys[i], uint32_t(i));
    SlotTable big;
    big.init(16384);
    std::copy(small.t, small.t + 2048, big.t);  // the entries as they were, no rehash
    size_t lost = 0;
    for (size_t i = 0; i < keys.size(); ++i) lost += big.find(keys[i], [&](uint32_t s) { return keys[s]; }) != uint32_t(i);
    CHECK(lost > 500, "control: a table copied without a rehash loses most of its keys (%zu of %zu)", lost, keys.size());
  }

  std::printf("%s engram_block_test: %d failed\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
