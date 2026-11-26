// server/kept_id.h against cases (host only, no GPU): the kept-session id changes with the image encoder's set, so a start whose
// vision weights differ from the stop's never reuses a session whose image rows the other encoder made. Built with
// -DPICO_OLD_RULES the cases fail (exit 1): the set was not part of the id. Exit 0 when every case passes.
#include <cstdio>

#include "../kept_id.h"

static int g_bad = 0, g_n = 0;
static void check(bool ok, const char* what) {
  ++g_n;
  if (!ok) ++g_bad, std::printf("FAIL: %s\n", what);
}

int main() {
  const uint64_t h = keepfile::fnv("dsgb10 kept sessions", 20);  // as run_id starts; the files' part is the same in every case
  const int64_t o[14] = {4194304, 0, 1, 0, 8192, 0, 0, 0, 43, 0, 4 << 20, 1, 1, 1};
  const uint64_t none = keptid::fold(h, o, 14, 0), a = keptid::fold(h, o, 14, 0x3d77c1be2d85a01cull),
                 b = keptid::fold(h, o, 14, 0xf4e512ecdf3e95d6ull);
  check(a == keptid::fold(h, o, 14, 0x3d77c1be2d85a01cull), "the same set: the same id (a restart reuses its sessions)");
  check(a != b, "another vision set: another id (the kept sessions are not reused)");
  check(a != none && b != none, "an encoder where there was none, or none where there was one: another id");
  int64_t o2[14];
  for (int i = 0; i < 14; ++i) o2[i] = o[i];
  o2[0] = 1048576;
  check(keptid::fold(h, o2, 14, 0x3d77c1be2d85a01cull) != a, "control: another option still changes the id");
  std::printf("kept_id_test: %d checks, %d failed\n", g_n, g_bad);
  return g_bad ? 1 : 0;
}
