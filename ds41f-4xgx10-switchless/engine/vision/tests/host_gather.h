// An all-gather of any size over a host all-gather that takes at most kHostGatherMax bytes a call (comm::Ring::allgather_host
// dies above it: ring.cpp's bootstrap sockets). `gather(mine, all, n)` is that call: all = 4 x n bytes, rank order.
// The bytes go in pieces of at most kHostGatherMax; `all` ends up as four ranks' whole contributions, rank order.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

constexpr size_t kHostGatherMax = 65536;

template <class Gather>
void chunked_host_allgather(Gather&& gather, const unsigned char* mine, unsigned char* all, size_t n, int ranks = 4) {
  std::vector<unsigned char> piece(ranks * std::min(n, kHostGatherMax));
  for (size_t off = 0; off < n; off += kHostGatherMax) {
    const size_t c = std::min(kHostGatherMax, n - off);
    gather(mine + off, piece.data(), c);
    for (int r = 0; r < ranks; r++) std::memcpy(all + r * n + off, piece.data() + (size_t)r * c, c);
  }
}
