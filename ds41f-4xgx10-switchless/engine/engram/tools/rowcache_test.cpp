// The hot-row cache over two regions (rowcache.h, display.h): a cache whose first slots sit in the display reservation's
// mapping and the rest in ordinary memory returns the same rows as an all-ordinary one, under inserts, hits, evictions
// and clear(), and the reservation's fallbacks refuse cleanly. Correctness only: no timing. Host only.
//   engram_rowcache_test [--card /dev/dri/card0]     (the real-buffer stage is skipped when the card does not open)
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "engram/display.h"
#include "engram/rowcache.h"

using namespace engram;

static int fails = 0;
#define CHECK(c, ...) do { const bool ok_ = (c); std::printf("  %s: ", ok_ ? "ok" : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); fails += !ok_; } while (0)

static void row_of(uint64_t key, uint8_t* r) {
  uint64_t x = key * 0x9E3779B97F4A7C15ull + 1;
  for (int i = 0; i < kRowBytes; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; r[i] = (uint8_t)x; }
}

// the same operation stream on a reference (all ordinary) and a split cache; every find and eviction must agree
static bool compare(size_t cache_bytes, uint8_t* region, size_t region_bytes, size_t ops, uint64_t universe, const char* what) {
  RowCache ref, sp;
  ref.init(cache_bytes);
  sp.init(cache_bytes, region, region_bytes);
  const size_t slots = cache_bytes / kRowBytes;
  const size_t want_split = region ? std::min(slots, region_bytes / kRowBytes) : 0;
  bool ok = sp.split == want_split && ref.split == 0;
  uint64_t rng = 12345, hits = 0, evictions = 0;
  uint8_t rec[kRowBytes], exp[kRowBytes];
  for (size_t i = 0; i < ops && ok; ++i) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    // a hot third of the keys, so that hits happen at every size
    const uint64_t key = (rng % 3 ? (rng >> 8) % (universe / 3 + 1) : (rng >> 8) % universe) | (uint64_t(i % 2) << 32);
    const uint8_t* a = ref.find(key);
    const uint8_t* b = sp.find(key);
    if ((a == nullptr) != (b == nullptr)) { ok = false; break; }
    row_of(key, exp);
    if (a) {
      ++hits;
      if (memcmp(a, exp, kRowBytes) || memcmp(b, exp, kRowBytes)) { ok = false; break; }
    } else {
      row_of(key, rec);
      const bool e1 = ref.insert(key, rec), e2 = sp.insert(key, rec);
      if (e1 != e2) { ok = false; break; }
      evictions += e1;
    }
    if (i == ops / 2) { ref.clear(); sp.clear(); }  // clear() drops every row in both
  }
  CHECK(ok, "%s: %zu slots (%zu in the region), %zu ops, %llu hits, %llu evictions, every find, row and eviction equal",
        what, slots, sp.split, ops, (unsigned long long)hits, (unsigned long long)evictions);
  CHECK(hits > 0 && evictions > 0, "%s: the stream hit and evicted", what);
  return ok;
}

int main(int argc, char** argv) {
  std::string card = "/dev/dri/card0";
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], "--card")) card = argv[i + 1];

  std::printf("the split slot numbering over a stand-in region (anonymous memory)\n");
  const size_t cache = 24u << 20;
  uint8_t* fake = (uint8_t*)mmap(nullptr, cache, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  compare(cache, nullptr, 0, 600000, 300000, "all ordinary");
  compare(cache, fake, 100, 600000, 300000, "a region smaller than one slot (none placed)");
  compare(cache, fake, 8u << 20, 600000, 300000, "a third in the region");
  compare(cache, fake, 8u << 20 | 100, 600000, 300000, "a region ending inside a slot");
  compare(cache, fake, cache, 600000, 300000, "all in the region");
  compare(cache, fake, cache * 2, 600000, 300000, "a region larger than the cache (all of it, no more)");
  {
    RowCache c;
    c.init(cache, fake, 4u << 20);
    CHECK(c.split * kRowBytes <= (4u << 20) && (c.split + 1) * kRowBytes > (4u << 20), "whole slots only: %zu slots, %zu bytes", c.split, c.split * kRowBytes);
    CHECK(c.slot(0) == fake && c.slot((uint32_t)c.split) == c.data && c.slot((uint32_t)c.split + 1) == c.data + kRowBytes,
          "slot s < split is in the region, the next one starts the ordinary array");
  }
  munmap(fake, cache);

  std::printf("the reservation's fallbacks\n");
  {
    char tmpl[] = "/tmp/engram_display_XXXXXX";
    const std::string root = mkdtemp(tmpl);
    const std::string d = root + "/class/drm";
    CHECK(system(("mkdir -p " + d + "/card0-HDMI-A-1 " + d + "/card0-DP-1 " + d + "/card1-DP-1; echo disconnected > " + d + "/card0-HDMI-A-1/status; echo disconnected > " + d + "/card0-DP-1/status; echo connected > " + d + "/card1-DP-1/status").c_str()) == 0, "the fake tree's step %d", 1);
    CHECK(Display::connected_output(root, "card0").empty(), "no connected output on card0 (card1's does not count)");
    CHECK(system(("echo connected > " + d + "/card0-DP-1/status").c_str()) == 0, "the fake tree's step %d", 2);
    CHECK(Display::connected_output(root, "card0") == "card0-DP-1", "a connected output of card0 is named");
    CHECK(Display::connected_output(root + "/none", "card0").empty(), "no sysfs tree: no output");
    Display c;
    CHECK(!c.create("/dev/dri/card99", "/sys", 64u << 20) && !c.active() && !c.why().empty(), "a card that does not exist: %s", c.why().c_str());
    CHECK(!c.create("/dev/null", "/sys", 64u << 20) && !c.active() && !c.why().empty(), "a file that is not a card: %s", c.why().c_str());
    CHECK(!c.create("/dev/null", "/sys", 1u << 20) && !c.active() && !c.why().empty(), "a size under one step: %s", c.why().c_str());
    CHECK(system(("mkdir -p " + root + "/dev; ln -s /dev/null " + root + "/dev/card0").c_str()) == 0, "the fake tree's step %d", 3);
    CHECK(!c.create(root + "/dev/card0", root, 64u << 20) && c.why().find("connected") != std::string::npos, "a connected output refuses: %s", c.why().c_str());
    CHECK(system(("rm -rf " + root).c_str()) == 0, "the fake tree's step %d", 4);
  }

  std::printf("the real reservation (%s)\n", card.c_str());
  {
    Display dsp;
    if (!dsp.create(card, "/sys", 32u << 20)) {
      std::printf("  no reservation here (%s): stage skipped\n", dsp.why().c_str());
    } else {
      CHECK(dsp.size() >= (16u << 20) && dsp.data() != nullptr, "a %zu MiB buffer, mapped", dsp.size() >> 20);
      // the whole buffer holds what is written: a pattern per page
      bool ok = true;
      for (size_t off = 0; off < dsp.size(); off += 4096) dsp.data()[off] = (uint8_t)(off >> 12);
      for (size_t off = 0; off < dsp.size() && ok; off += 4096) ok = dsp.data()[off] == (uint8_t)(off >> 12);
      CHECK(ok, "every page of the buffer reads back what was written");
      const size_t region = dsp.capacity();
      compare(region + (region / 2 / kRowBytes) * kRowBytes, dsp.data(), region, 900000, 400000, "the reservation then ordinary memory");
      dsp.limit(region / 2);
      CHECK(dsp.capacity() == region / 2, "limit() lowers the usable size");
      dsp.limit(region);
      CHECK(dsp.capacity() == region / 2, "limit() never raises it");
      dsp.destroy();
      CHECK(!dsp.active() && dsp.size() == 0, "destroy() releases the mapping and the buffer");
      // the next create works at once: the dumb buffer is free again
      CHECK(dsp.create(card, "/sys", 32u << 20), "created again after destroy: %s", dsp.why().c_str());
    }
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
