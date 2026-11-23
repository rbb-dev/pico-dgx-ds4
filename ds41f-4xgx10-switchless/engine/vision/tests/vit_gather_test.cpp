// Host only: the chunked host all-gather of vit_ring_test (host_gather.h) against a fake of comm::Ring::allgather_host
// that refuses more than 65,536 bytes a call, as the ring's does. The control's 6,424,576-byte exchange of block 17's slices.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "host_gather.h"

static int fails = 0;
static void expect(bool ok, const char* what) {
  if (!ok) fails++;
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

int main() {
  constexpr int R = 4;
  const size_t n = 6424576;
  std::vector<std::vector<unsigned char>> mine(R, std::vector<unsigned char>(n));
  for (int r = 0; r < R; r++)
    for (size_t i = 0; i < n; i++) mine[r][i] = (unsigned char)(i * 31 + r * 7 + (i >> 9));
  size_t calls = 0, biggest = 0;
  // the fake: every rank's contribution of this piece, in rank order, refused above the ring's limit
  auto fake = [&](size_t off, bool* refused) {
    return [&, off, refused](const unsigned char*, unsigned char* all, size_t c) {
      calls++, biggest = std::max(biggest, c);
      if (c > 65536) {
        *refused = true;
        return;
      }
      for (int r = 0; r < R; r++) std::memcpy(all + (size_t)r * c, mine[r].data() + off, c);
    };
  };
  // RED: the call as vit_ring_test made it (one call of the whole slice set) is refused
  bool refused = false;
  std::vector<unsigned char> sink((size_t)R * n);
  fake(0, &refused)(mine[0].data(), sink.data(), n);
  expect(refused, "one call of 6,424,576 bytes is refused by the 65,536-byte limit (the failure seen on the Sparks)");
  // GREEN: the chunked helper, for each rank's view
  calls = 0, biggest = 0;
  bool refused2 = false;
  std::vector<unsigned char> all((size_t)R * n);
  size_t off = 0;
  chunked_host_allgather(
      [&](const unsigned char* m, unsigned char* a, size_t c) {
        (void)m;
        fake(off, &refused2)(m, a, c);
        off += c;
      },
      mine[1].data(), all.data(), n);
  expect(!refused2 && biggest <= 65536, "chunked: no call above 65,536 bytes");
  expect(calls == (n + 65535) / 65536, "chunked: 98 calls");
  bool same = true;
  for (int r = 0; r < R; r++) same &= std::memcmp(all.data() + (size_t)r * n, mine[r].data(), n) == 0;
  expect(same, "chunked: all four ranks' contributions whole and in rank order");
  // a size under the limit and an odd tail
  for (size_t m : {(size_t)16, (size_t)65536, (size_t)65537, (size_t)200000}) {
    std::vector<unsigned char> a((size_t)R * m);
    size_t o = 0;
    chunked_host_allgather(
        [&](const unsigned char*, unsigned char* p, size_t c) {
          for (int r = 0; r < R; r++) std::memcpy(p + (size_t)r * c, mine[r].data() + o, c);
          o += c;
        },
        mine[0].data(), a.data(), m);
    bool ok = true;
    for (int r = 0; r < R; r++) ok &= std::memcmp(a.data() + (size_t)r * m, mine[r].data(), m) == 0;
    expect(ok, "chunked: other sizes");
  }
  std::printf("%s\n", fails ? "vit_gather_test FAILED" : "vit_gather_test passed");
  return fails ? 1 : 0;
}
