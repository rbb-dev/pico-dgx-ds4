// server/ride_wire.h: the ride's wire as functions (host only). four ranks' words through a fake gather give one decision on
// all four (rank 0's request, the smallest `done` and `room`); the control "no ring" (the branch with no ring) gives ranks 1 to 3 `h 0, budget 0`; unequal readings are named.
#include <cstdio>
#include <string>

#include "../ride_wire.h"

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
  if (!ok) ++g_fail, std::printf("FAIL %s\n", what.c_str());
  else std::printf("ok   %s\n", what.c_str());
}

int main() {
  using namespace ridewire;
  Wire all[4];
  const uint64_t done[4] = {7, 5, 9, 6};
  const int room[4] = {8, 8, 3, 8};
  for (int r = 0; r < 4; ++r) {
    all[r].rank = r, all[r].done = done[r], all[r].room = room[r];
    all[r].img_seq = 12, all[r].jobs = 4, all[r].ghosts = 300, all[r].free_numbers = 5000;
  }
  all[0].h = 41, all[0].budget = 16, all[0].live[0] = 9, all[0].live[1] = 11;
  Decision d[4];
  bool ok = true;
  for (int r = 0; r < 4; ++r) ok = ok && decide(all, 4, &d[r]).empty();
  check(ok, "four ranks' words decide with no complaint");
  check(d[0].settled == 5 && d[0].room == 3 && d[0].h == 41 && d[0].budget == 3 && d[0].live[1] == 11, "rank 0's request, the smallest done (5) and the smallest room (3): h 41, budget min(16, 3) = 3");
  bool same = true;
  for (int r = 1; r < 4; ++r) same = same && d[r].settled == d[0].settled && d[r].h == d[0].h && d[r].budget == d[0].budget && d[r].live[0] == d[0].live[0];
  check(same, "one decision on all four ranks");
  // the control: the branch with no ring, every rank from its own word (only rank 0 holds the request)
  Wire mine[4];
  for (int r = 0; r < 4; ++r) {
    mine[r] = all[r];
    if (r) mine[r].h = 0, mine[r].budget = 0;  // only rank 0 fills the request
  }
  bool alone_differs = false;
  for (int r = 1; r < 4; ++r) {
    const Decision a = decide_alone(mine[r]);
    alone_differs = alone_differs || (a.h == 0 && a.budget == 0 && a.settled != d[0].settled);
  }
  check(alone_differs, "control (no ring): ranks 1 to 3 decide h 0, budget 0 and settle by their own reading: the four ranks do not agree");
  // unequal readings stop the ride and name the values
  Wire bad[4];
  for (int r = 0; r < 4; ++r) bad[r] = all[r];
  bad[2].ghosts = 0;
  Decision x;
  const std::string e = decide(bad, 4, &x);
  check(e.find("ghosts") != std::string::npos && e.find("300 300 0 300") != std::string::npos, "a rank with another ghost count: the ride stops naming the four values: " + e);
  Wire bad2[4];
  for (int r = 0; r < 4; ++r) bad2[r] = all[r];
  bad2[3].img_seq = 11;
  check(decide(bad2, 4, &x).find("image commands") != std::string::npos, "another command count is named");
  Wire bad3[4];
  for (int r = 0; r < 4; ++r) bad3[r] = all[r];
  bad3[1].rank = 2;
  check(!decide(bad3, 4, &x).empty(), "a word that is not from the rank whose place it is stops the ride");
  std::printf(g_fail ? "ride_wire_test FAILED %d\n" : "ride_wire_test passed\n", g_fail);
  return g_fail != 0;
}
