// comm/host_gather.h on four threads over socket pairs (host only, no GPU). the framing of the host all-gather.
//   row 1  four ranks in one call: every rank holds the same four words.
//   row 2  the crash's own arithmetic: ranks 1 to 3 in the command frame (1,000 B), rank 0 in a 4-byte agree. With the head every rank that sees the other
//          call dies naming both. The named control, `framed` false = the framing with no head: rank 0 returns from the others' frame bytes and nothing stops it.
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "comm/host_gather.h"

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
  if (!ok) ++g_fail, std::printf("FAIL %s\n", what.c_str());
  else std::printf("ok   %s\n", what.c_str());
}
static void die_throws(const std::string& m) { throw std::runtime_error(m); }

struct Ring4 {
  int next[4], prev[4];  // rank r sends on next[r] and receives on prev[r]
  Ring4() {
    for (int r = 0; r < 4; ++r) {
      int sv[2];
      socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
      next[r] = sv[0], prev[(r + 1) % 4] = sv[1];
    }
    timeval tv{1, 0};  // a rank waiting for a partner that is not coming gives up after a second (the test's own watchdog)
    for (int r = 0; r < 4; ++r) setsockopt(prev[r], SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  }
  void close_rank(int r) { ::shutdown(next[r], SHUT_RDWR), ::shutdown(prev[r], SHUT_RDWR); }
  ~Ring4() {
    for (int r = 0; r < 4; ++r) ::close(next[r]), ::close(prev[r]);
  }
};

struct Out {
  bool died = false;
  std::string why;
  std::vector<uint8_t> all;
};

// rank r runs call[r] of size n[r]
static std::vector<Out> run(const uint32_t call[4], const size_t n[4], bool framed) {
  Ring4 ring;
  std::vector<Out> out(4);
  std::vector<std::thread> th;
  for (int r = 0; r < 4; ++r)
    th.emplace_back([&, r] {
      std::vector<uint8_t> mine(n[r], (uint8_t)(0x10 + r)), all(4 * n[r]);
      try {
        comm::host_allgather(r, 4, ring.next[r], ring.prev[r], mine.data(), all.data(), n[r], call[r], framed);
        out[r].all = all;
      } catch (const std::exception& e) {
        out[r].died = true, out[r].why = e.what();
        ring.close_rank(r);  // a rank that stops closes its sockets, as a process does
      }
    });
  for (auto& t : th) t.join();
  return out;
}

int main() {
  signal(SIGPIPE, SIG_IGN);  // a rank that stopped closes its sockets: a write to it is an error to die on, not a signal
  g_die_handler = die_throws;
  {
    const uint32_t call[4] = {comm::kCallRide, comm::kCallRide, comm::kCallRide, comm::kCallRide};
    const size_t n[4] = {16, 16, 16, 16};
    const std::vector<Out> o = run(call, n, true);
    bool same = true;
    for (int r = 0; r < 4; ++r) same = same && !o[r].died && o[r].all == o[0].all && o[r].all.size() == 64;
    for (int k = 0; k < 4 && same; ++k) same = o[0].all[k * 16] == 0x10 + k;
    check(same, "row 1: four ranks in one call hold the same four words, in rank order");
  }
  {
    const uint32_t call[4] = {comm::kCallAgree, comm::kCallFrame, comm::kCallFrame, comm::kCallFrame};
    const size_t n[4] = {4, 1000, 1000, 1000};
    const std::vector<Out> c = run(call, n, false);
    check(!c[0].died && c[0].all.size() == 16, "control (no head): rank 0 returns its 4-byte answer from the others' frame bytes and nothing stops it");
    int stuck = 0;
    for (int r = 1; r < 4; ++r) stuck += c[r].died;
    check(stuck == 3, "control: ranks 1 to 3 are left waiting (their own timeout here, the engine's next command frame in production)");
    const std::vector<Out> g = run(call, n, true);
    check(g[0].died && g[0].why.find("agree") != std::string::npos && g[0].why.find("command frame") != std::string::npos && g[0].why.find("out of step") != std::string::npos,
          "row 2: rank 0 dies naming both calls: " + g[0].why);
    check(g[1].died && g[1].why.find("agree") != std::string::npos && g[1].why.find("command frame") != std::string::npos,
          "row 2: rank 1 dies naming both calls: " + g[1].why);
    check(g[2].died && g[3].died, "row 2: ranks 2 and 3 stop too (their partners are gone)");
  }
  {  // every site has an id of its own; two ranks in different sites (the same size, the worst case) die naming both sites
    bool distinct = true, named = true;
    const uint32_t ids[] = {comm::kCallFrame, comm::kCallPayload, comm::kCallAgree, comm::kCallRide, comm::kCallWatch, comm::kCallMain0, comm::kCallMain0 + 5,
                            comm::kCallKeepIndex, comm::kCallCheckpointIndex, comm::kCallDiskRoom, comm::kCallUnspill, comm::kCallJoin, comm::kCallTenant0,
                            comm::kCallTenant0 + 2, comm::kCallVit, comm::kCallWindowMin, comm::kCallWindowMax};
    for (size_t i = 0; i < sizeof ids / sizeof *ids; ++i) {
      named = named && std::string(comm::host_call_name(ids[i])) != "other";
      for (size_t j = i + 1; j < sizeof ids / sizeof *ids; ++j) distinct = distinct && ids[i] != ids[j];
    }
    check(distinct && named, "the 17 sites' ids are distinct and each has a name");
    const uint32_t call[4] = {comm::kCallUnspill, comm::kCallJoin, comm::kCallJoin, comm::kCallJoin};
    const size_t n[4] = {16, 16, 16, 16};
    const std::vector<Out> o = run(call, n, true);
    check(o[0].died && o[0].why.find("read-back (unspill)") != std::string::npos && o[0].why.find("join") != std::string::npos, "unspill's and join's agreements of the same size meet: rank 0 dies naming both: " + o[0].why);
    check(o[1].died && o[1].why.find("read-back (unspill)") != std::string::npos, "and rank 1 dies naming both: " + o[1].why);
    const std::vector<Out> c = run(call, n, false);
    bool silent = true;
    for (int r = 0; r < 4; ++r) silent = silent && !c[r].died;
    check(silent, "control (no head): two sites of the same size read each other's bytes and nobody stops");
  }
  std::printf(g_fail ? "host_gather_test FAILED %d\n" : "host_gather_test passed\n", g_fail);
  return g_fail != 0;
}
