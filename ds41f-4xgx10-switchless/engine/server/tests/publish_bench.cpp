// What STATS publishing from the decode loops costs on the host (no model, no GPU): the call every decode step makes (a clock read and
// lendpace::publish_due, which says no for 0.5 s) and, for the once-per-0.5 s case, the part of publish() that has no model behind it:
// the 63-field StatsFrame filled, copied into the frame's byte vector and swapped under a mutex. The model's own reads (tier_stats,
// pool_stats, tenant_rows: a few loads and a vector of at most 16 rows) are not in it: they are measured in the speed session's arm.
// Prints ns per call and the share of a decode step they take at 0.5 s and at a 20 ms step.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/frontend_protocol.h"
#include "server/lend_pace.h"

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
  const int N = 20000000;
  double last = now_s();
  long yes = 0;
  const double t0 = now_s();
  for (int i = 0; i < N; ++i)
    if (lendpace::publish_due(now_s(), last, false, true)) ++yes, last = now_s();
  const double check_ns = (now_s() - t0) / N * 1e9;

  std::mutex mu;
  std::vector<char> stats;
  const int M = 2000000;
  volatile uint64_t sink = 0;
  const double t1 = now_s();
  for (int i = 0; i < M; ++i) {
    fe::StatsFrame sf{};
    sf.version = fe::kStatsVersion, sf.n_fields = fe::kStatsFields;
    uint64_t* f = &sf.pool_numbers;
    for (uint32_t k = 0; k < fe::kStatsFields; ++k) f[k] = (uint64_t)i + k;
    std::vector<char> frame(sizeof sf + 8);
    std::memcpy(frame.data() + 8, &sf, sizeof sf);
    std::lock_guard<std::mutex> l(mu);
    stats.swap(frame);
    sink += stats[16];
  }
  const double build_ns = (now_s() - t1) / M * 1e9;
  std::printf("publish_due path (a decode step, no publish): %.1f ns a call (%ld publishes in %d calls)\n", check_ns, yes, N);
  std::printf("publish's host part (frame fill, copy, swap): %.0f ns a publish\n", build_ns);
  std::printf("at a 20 ms step: the check is %.5f %% of a step; a publish every 0.5 s adds %.7f %% of the time\n", check_ns / 20e6 * 100,
              build_ns / 0.5e9 * 100);
  std::printf("PASS publish_bench\n");
  return 0;
}
