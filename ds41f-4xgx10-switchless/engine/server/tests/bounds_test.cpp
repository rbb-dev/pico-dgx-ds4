// server/bounds.h against cases (host only, no GPU): a connection whose reader stopped is dropped at the output cap and the
// bytes come back as the writer drains. Built with -DPICO_OLD_RULES the cases fail (exit 1): the queue is unbounded.
// Exit 0 when every case passes.
#include <cstdio>

#include "../bounds.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}

int main() {
  using bounds::OutQueue;

  // ---- the output queue: a front-end that stops reading
  {
    OutQueue q;
    const size_t frame = 1 << 20;
    int queued = 0;
    while (queued < 1000 && q.add(frame)) ++queued;
    check(queued == 64, "a stalled connection is dropped at 64 MiB of queued frames (not 1000 frames)");
    check(q.queued() == 64 * frame, "the frame that did not fit was not counted");
    for (int i = 0; i < 10; ++i) q.done(frame);
    check(q.add(frame), "a connection whose writer drains takes frames again");
    OutQueue small;
    check(small.add(100) && small.add(100) && small.queued() == 200, "small frames queue and count");
  }

  if (g_bad) {
    std::printf("%d of %d cases failed\n", g_bad, g_n);
    return 1;
  }
  std::printf("%d cases ok\n", g_n);
  return 0;
}
