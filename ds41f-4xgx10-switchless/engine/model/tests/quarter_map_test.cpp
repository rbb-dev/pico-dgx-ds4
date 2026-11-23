// model/quarter_map.h on the host: the ownership maps against the oracle's arithmetic (every entry of every array has one owner
// and one place, and the maps are each other's inverse), the quarter layout's unit walk against the oracle's quarter and its
// inverse, the exact sum, the trailer's round trip and its refusals (another rank's, another piece's, a spoiled byte).
// A wrong-map control: owners taken from the wrong bits must fail the same checks.
#include <cstdio>
#include <vector>

#include "model/quarter_map.h"
#include "model/tests/quarter_oracle.h"

namespace q = model::quarter;
namespace o = quarter_oracle;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s\n", what);
  }
}

// a quarter as the map's unit walk builds it from an image (the host twin of k_quarter), entries from `written` on zero
std::vector<uint8_t> walk_quarter(const std::vector<uint8_t>& image, int qi, uint32_t written, bool wrong_bits) {
  std::vector<uint8_t> out(q::kQuarterBytes, 0);
  for (uint32_t b = 0; b < q::kQuarterBytes; b += 4) {
    q::Src s = q::quarter_src(qi, b);
    if (wrong_bits) s.off ^= 4;  // control: the walk reads the neighbouring unit
    if (s.entry < written) std::memcpy(&out[b], &image[q::image_off(s.a) + s.off], 4);
  }
  return out;
}
}  // namespace

int main() {
  // the maps against the oracle, and one place for every entry
  {
    bool same = true, bijection = true, inverse = true;
    for (int a = 0; a < 4; ++a) {
      std::vector<int> hit(4 * q::kQuarterEntries, 0);
      for (uint32_t e = 0; e < q::kEntries; ++e) {
        const int ow = q::owner_of(a, e);
        const uint32_t lo = q::local_of(a, e);
        same &= ow == o::owner(a, e) && lo == o::local(a, e);
        bijection &= lo < q::kQuarterEntries && ++hit[ow * q::kQuarterEntries + lo] == 1;
        inverse &= q::entry_of(a, ow, lo) == e;
      }
      for (const int h : hit) bijection &= h == 1;
    }
    check(same, "the maps equal the oracle's arithmetic");
    check(bijection, "every (owner, local index) is an entry, once");
    check(inverse, "entry_of is the maps' inverse");
    // the layout's sizes
    check(q::kQuarterBytes == 729088 && q::kImageBytes == 2916352 && q::kDiskBytes == 786432 && q::kTrailerBytes == 57344, "the sizes");
  }
  // the unit walk against the oracle's quarter, and the inverse walk
  for (const uint32_t written : {8192u, 8191u, 5000u, 2049u, 1u, 0u}) {
    const std::vector<uint8_t> image = o::random_image(written + 7);
    std::vector<uint8_t> all;
    bool eq = true;
    for (int qi = 0; qi < 4; ++qi) {
      const std::vector<uint8_t> w = walk_quarter(image, qi, written, false), r = o::quarter_of(image, qi, written);
      eq &= w == r;
      all.insert(all.end(), r.begin(), r.end());
    }
    char what[96];
    std::snprintf(what, sizeof what, "the quarters' unit walk equals the oracle's quarters, written %u", written);
    check(eq, what);
    std::snprintf(what, sizeof what, "four quarters make the written image again, written %u", written);
    check(o::image_of(all) == o::written_of(image, written), what);
  }
  {  // control: a walk that reads the neighbouring unit differs
    const std::vector<uint8_t> image = o::random_image(3);
    check(walk_quarter(image, 1, q::kEntries, true) != o::quarter_of(image, 1, q::kEntries), "control: a wrong walk is not the oracle's quarter");
  }
  // the exact sums
  {
    const std::vector<uint8_t> image = o::random_image(11);
    bool eq = true;
    for (const size_t bytes : {(size_t)8, (size_t)64, (size_t)65536, (size_t)q::kQuarterBytes, (size_t)q::kImageBytes})
      for (const uint64_t seed : {0ull, 1ull, 3ull, 4ull}) eq &= q::sum_exact_host(image.data(), bytes, seed) == o::sum_of(image.data(), bytes, seed);
    check(eq, "sum_exact_host equals the oracle's block_sum terms");
    check(q::sum_exact_host(image.data(), 4096, 0) != q::sum_exact_host(image.data(), 4096, 1), "the seed changes the sum");
    std::vector<uint8_t> flipped = image;
    flipped[1000] ^= 1;
    check(q::sum_exact_host(image.data(), 4096, 0) != q::sum_exact_host(flipped.data(), 4096, 0), "a flipped bit changes the sum");
  }
  // the trailer
  {
    q::Trailer t;
    for (int i = 0; i < 4; ++i) t.qsum[i] = 0x1111111111111111ull * (i + 1);
    t.wsum = 0xABCDEF0123456789ull, t.rank = 2, t.handle = 17, t.list = 3, t.piece = 91, t.written = 8192;
    std::vector<uint8_t> buf(q::kTrailerBytes, 0xEE);
    q::trailer_pack(t, buf.data());
    bool tail_zero = true;
    for (size_t i = 128; i < buf.size(); ++i) tail_zero &= buf[i] == 0;
    check(tail_zero, "the trailer fills its block, the rest zero");
    q::Trailer r;
    check(q::trailer_unpack(buf.data(), &r), "a trailer round trip");
    bool eq = true;
    for (int i = 0; i < 4; ++i) eq &= r.qsum[i] == t.qsum[i];
    check(eq && r.wsum == t.wsum && r.rank == 2 && r.handle == 17 && r.list == 3 && r.piece == 91 && r.written == 8192, "every field comes back");
    check(q::trailer_is(r, 2, 17, 3, 91), "the trailer is this rank's and this piece's");
    check(!q::trailer_is(r, 1, 17, 3, 91), "refused: another rank's");
    check(!q::trailer_is(r, 2, 18, 3, 91) && !q::trailer_is(r, 2, 17, 2, 91) && !q::trailer_is(r, 2, 17, 3, 90),
          "refused: another session's, list's or piece's");
    int bad = 0;
    for (size_t i = 0; i < 80; ++i) {  // every byte of the packed fields is covered by the check (68-71 pad the check's alignment)
      if (i >= 68 && i < 72) continue;
      std::vector<uint8_t> s = buf;
      s[i] ^= 0x10;
      q::Trailer x;
      bad += q::trailer_unpack(s.data(), &x);
    }
    check(bad == 0, "refused: a spoiled byte in the packed fields");
    std::vector<uint8_t> zero(q::kTrailerBytes, 0);
    check(!q::trailer_unpack(zero.data(), &r), "refused: a block of zeros (a quarter written without a trailer)");
  }
  std::printf(g_fail ? "FAILED %d\n" : "quarter_map_test: all passed\n", g_fail);
  return g_fail != 0;
}
