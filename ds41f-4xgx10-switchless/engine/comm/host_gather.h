// The host all-gather over the bootstrap sockets (comm/ring.cpp Ring::allgather_host), as a header the host tests can build: two file descriptors,
// no GPU. Every hop sends an 8-byte head (the call's id and its size) and the payload in one writev; the receiver compares the head with its own
// call before it uses a byte, and a difference stops it naming both calls. Without the head (`framed` false: the framing with no head, kept as the
// named control of host_gather_test) a rank that is in another call than its partners takes their bytes for its own answer, and nobody stops.
#pragma once
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "common/die.h"

namespace comm {

// Every host all-gather site has an id of its own, so that two ranks in different sites die naming both sites and never read each other's bytes.
// 0 is a call in no site of this list (its size is still compared).
enum HostCall : uint32_t {
  kCallOther = 0, kCallFrame = 1, kCallPayload = 2, kCallAgree = 3, kCallRide = 4, kCallWatch = 5,
  kCallMain0 = 10,  // main.cpp's start-up calls: 10 + the order they are made in
  kCallKeepIndex = 20, kCallCheckpointIndex = 21, kCallDiskRoom = 22, kCallUnspill = 23, kCallJoin = 24,  // the store's agreements (store_core.cpp)
  kCallTenant0 = 30,  // tenant.cu: 30 and 31 the two tenant minimums, 32 the lend's vote
  kCallVit = 40, kCallWindowMin = 50, kCallWindowMax = 51, kCallCopyAgree = 60,
};

inline const char* host_call_name(uint32_t id) {
  switch (id) {
    case kCallFrame: return "command frame";
    case kCallPayload: return "command payload";
    case kCallAgree: return "agree";
    case kCallRide: return "ride";
    case kCallWatch: return "watch";
    case kCallCopyAgree: return "copy drafting settings";
    case 10: case 11: case 12: case 13: case 14: case 15: return "start-up (main.cpp)";
    case kCallKeepIndex: return "kept index write (keep)";
    case kCallCheckpointIndex: return "kept index write (checkpoint)";
    case kCallDiskRoom: return "disk tier room";
    case kCallUnspill: return "read-back (unspill)";
    case kCallJoin: return "join";
    case 30: case 31: case 32: return "tenant (tenant.cu)";
    case kCallVit: return "image encoder vote";
    case kCallWindowMin: return "window minimum";
    case kCallWindowMax: return "window maximum";
    default: return "other";
  }
}

struct HostHead {
  uint32_t call, size;
};

namespace hg {
inline void send_iov(int fd, iovec* iov, int n) {
  while (n > 0) {
    const ssize_t w = ::writev(fd, iov, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      die("host all-gather: send: %s", std::strerror(errno));
    }
    size_t left = (size_t)w;
    while (n > 0 && left >= iov->iov_len) left -= iov->iov_len, ++iov, --n;
    if (n > 0 && left) iov->iov_base = (char*)iov->iov_base + left, iov->iov_len -= left;
  }
}
inline void recv_n(int fd, void* p, size_t n) {
  char* c = (char*)p;
  while (n > 0) {
    const ssize_t r = ::read(fd, c, n);
    if (r == 0) die("host all-gather: the peer closed the socket");
    if (r < 0) {
      if (errno == EINTR) continue;
      die("host all-gather: receive: %s", std::strerror(errno));
    }
    c += r, n -= (size_t)r;
  }
}
}  // namespace hg

// all = nranks x n bytes in rank order; `mine` is this rank's n bytes. next/prev: the ring's sockets.
inline void host_allgather(int rank, int nranks, int fd_next, int fd_prev, const void* mine, void* all, size_t n, uint32_t call, bool framed = true) {
  char* a = (char*)all;
  std::memcpy(a + (size_t)rank * n, mine, n);
  const int prev_rank = (rank - 1 + nranks) % nranks;
  for (int step = 0; step < nranks - 1; ++step) {
    const int out = (rank - step + nranks) % nranks, in = (rank - step - 1 + nranks) % nranks;
    if (framed) {
      HostHead h{call, (uint32_t)n};
      iovec iov[2] = {{&h, sizeof h}, {a + (size_t)out * n, n}};
      hg::send_iov(fd_next, iov, 2);
      HostHead g{};
      hg::recv_n(fd_prev, &g, sizeof g);
      if (g.call != call || g.size != (uint32_t)n)
        die("host all-gather: rank %d is in call %s of %zu B, rank %d sent call %s of %u B: the ranks are out of step", rank, host_call_name(call), n,
            prev_rank, host_call_name(g.call), g.size);
      hg::recv_n(fd_prev, a + (size_t)in * n, n);
    } else {
      iovec iov[1] = {{a + (size_t)out * n, n}};
      hg::send_iov(fd_next, iov, 1);
      hg::recv_n(fd_prev, a + (size_t)in * n, n);
    }
  }
}

}  // namespace comm
