// Four StoreCores in step (host only): four cores over four fake devices whose collectives meet at a barrier (a rank that does not
// arrive within 2 s fails the row naming the ranks that did), each device with its own stream reading and staging room. The operations:
// save, release the line, ride until settled and marked, unspill.
//   A  all four ride: every rank arrived at every collective, the four digests are equal after each phase, revives = the pieces on every rank, nothing read.
//   B  the ride applied on rank 0 only: the digests differ at the first compare, and the return stops EVERY rank naming who is out
//      (ranks 1 to 3 must not return before the collective while rank 0 waits in it).
//   C  the named control `kUnspillReturnsAlone` (the return before the collective): B ends with rank 0 alone at the barrier.
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "model/chunks.h"
#include "model/store_core.h"
#include "model/tests/store_fake.h"

using namespace model;
using namespace model::detail;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
  if (!ok) ++g_fail, std::printf("FAIL %s\n", what.c_str());
  else std::printf("ok   %s\n", what.c_str());
}
static void die_throws(const std::string& m) { throw std::runtime_error(m); }

struct Timeout : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// the four ranks' collective: each rank brings four words and gets the sixteen back; a rank that does not come in 2 s ends the others naming who was there
struct Meet {
  std::mutex m;
  std::condition_variable cv;
  uint64_t data[16] = {}, result[16] = {};
  bool present[4] = {};
  int arrived = 0;
  long gen = 0;
  uint32_t sites[4] = {}, result_sites[4] = {};
  void meet(int rank, uint32_t site, const uint64_t mine[4], uint64_t out[16]) {
    std::unique_lock<std::mutex> lk(m);
    sites[rank] = site;
    for (int k = 0; k < 4; ++k) data[rank * 4 + k] = mine[k];
    present[rank] = true;
    ++arrived;
    const long my = gen;
    if (arrived == 4) {
      std::copy(data, data + 16, result);
      std::copy(sites, sites + 4, result_sites);
      arrived = 0;
      for (bool& p : present) p = false;
      ++gen;
      cv.notify_all();
    } else if (!cv.wait_for(lk, std::chrono::seconds(2), [&] { return gen != my; })) {
      std::string who;
      for (int r = 0; r < 4; ++r)
        if (present[r]) who += (who.empty() ? "" : ",") + std::to_string(r);
      throw Timeout("rank " + std::to_string(rank) + " waits at a collective: only ranks " + who + " arrived");
    }
    // the host all-gather's framing (comm/host_gather.h): ranks in different sites stop naming both
    for (int r = 1; r < 4; ++r)
      if (result_sites[r] != result_sites[0])
        throw std::runtime_error("host all-gather: rank " + std::to_string(rank) + " is in site " + std::to_string(site) + ", rank " + std::to_string(r) + " in site " +
                                 std::to_string(result_sites[r]) + ": the ranks are out of step");
    std::copy(result, result + 16, out);
  }
};

struct Rank {
  FakeDev dev;
  chunks::Alloc alloc{400};
  sgeom::StoreGeom g;
  std::unique_ptr<StoreCore> core;
  int rank;
  Rank(int r, Meet& meet, const std::string& dir, StoreCore::TestRule rule) : rank(r) {
    g.rank = r, g.nslots = g.nunits = 4, g.list_len = 48, g.max_ctx = 262144;
    for (int s = 0; s < 4; ++s) g.unit_of[s] = s, g.slot_ctx[s] = g.max_ctx;
    g.image_bytes = 2916352, g.snap_bytes = 3u << 20, g.end_bytes = 5u << 20;
    g.has_ring = true, g.imager = true, g.mark = true;
    core = std::make_unique<StoreCore>(g, dev, &alloc);
    core->test_rule = rule;
    dev.rank = r;
    dev.gather = [&meet, r](int v, uint32_t site, int* all) {
      uint64_t mine[4] = {(uint64_t)(int64_t)v, 0, 0, 0}, out[16];
      meet.meet(r, site, mine, out);
      for (int k = 0; k < 4; ++k) all[k] = (int)(int64_t)out[k * 4];
    };
    dev.gather_words = [&meet, r](const uint64_t* mine, uint32_t site, uint64_t* out) { meet.meet(r, site, mine, out); };
    core->disk(dir + "/rank" + std::to_string(r), 400 * Pool::round(g.image_bytes), 512 << 20, false, 7);
  }
  void conversation() {  // 100,000 positions on slot 0, then the line is let go
    core->slot_reserve(0, 40, 0), core->slot_attach(0, 1);
    core->ensure(0, 100000);
    for (const Refs& l : core->store_of().sess.at(1).lists)
      for (const Ref c : l) dev.wrote(c.num());
    core->save(1, 0, 100000, 0);
    core->slot_release(0);
  }
  void ride() {
    core->image(1, 1000, {});
    core->settle(core->image_seq(), {});
  }
};

struct Result {
  std::vector<std::string> error;      // "" or what stopped the rank
  std::vector<uint64_t> digest_after_ride, digest_after_return;
  std::vector<uint64_t> revives, read;
};

static Result scenario(bool ride_all, StoreCore::TestRule rule, const std::string& dir) {
  Meet meet;
  std::vector<std::unique_ptr<Rank>> r;
  for (int k = 0; k < 4; ++k) r.push_back(std::make_unique<Rank>(k, meet, dir, rule));
  Result res;
  res.error.assign(4, "");
  res.digest_after_ride.assign(4, 0), res.digest_after_return.assign(4, 0), res.revives.assign(4, 0), res.read.assign(4, 0);
  const auto phase = [&](const std::function<void(int)>& f) {
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
      th.emplace_back([&, k] {
        try {
          f(k);
        } catch (const std::exception& e) {
          if (res.error[k].empty()) res.error[k] = e.what();
        }
      });
    for (auto& t : th) t.join();
  };
  phase([&](int k) {
    r[k]->conversation();
    if (ride_all || k == 0) r[k]->ride();  // the as-built wire: rank 0 alone imaged, settled and marked
    res.digest_after_ride[k] = r[k]->core->digest();
  });
  phase([&](int k) {
    r[k]->core->unspill(1);
    res.digest_after_return[k] = r[k]->core->digest();
    res.revives[k] = r[k]->core->split_counts.revives, res.read[k] = r[k]->core->split_counts.returns_read;
  });
  return res;
}

// a fork from an idle conversation, as the server makes it: its own return to the fork point (explicit unspill), then the bookmarks (copy_snaps) and the load.
// Collectives and read-back lines a rank made; `marked` false: the conversation was never imaged or marked (whole in the pool).
struct ForkResult {
  std::vector<std::string> error;
  std::vector<int> collectives, lines;
  std::vector<uint64_t> digest;
};
static ForkResult fork_scenario(bool marked, StoreCore::TestRule rule, const std::string& dir) {
  Meet meet;
  std::vector<std::unique_ptr<Rank>> r;
  for (int k = 0; k < 4; ++k) r.push_back(std::make_unique<Rank>(k, meet, dir, rule));
  ForkResult res;
  res.error.assign(4, ""), res.collectives.assign(4, 0), res.lines.assign(4, 0), res.digest.assign(4, 0);
  std::vector<std::thread> th;
  for (int k = 0; k < 4; ++k)
    th.emplace_back([&, k] {
      try {
        Rank& x = *r[k];
        x.conversation();
        if (marked) x.ride();
        const int a0 = x.dev.agree_alls;
        const uint64_t l0 = x.core->split_counts.unspill_cmds;
        if (marked) x.core->unspill(1, 0, 40000, 64);  // the server's own return to the fork point
        x.core->copy_snaps(1, 2, 40000);
        x.core->slot_reserve(1, 40, 0), x.core->slot_attach(1, 2);
        x.core->load(1, 40000, 1, 1);                  // the fork
        res.collectives[k] = x.dev.agree_alls - a0, res.lines[k] = (int)(x.core->split_counts.unspill_cmds - l0);
        res.digest[k] = x.core->digest();
      } catch (const std::exception& e) {
        res.error[k] = e.what();
      }
    });
  for (auto& t : th) t.join();
  return res;
}

static bool all_equal(const std::vector<uint64_t>& v) {
  for (const uint64_t x : v)
    if (x != v[0]) return false;
  return true;
}

int main() {
  g_die_handler = die_throws;
  char t[] = "/tmp/four_rank_test.XXXXXX";
  const std::string dir = mkdtemp(t);
  {
    const Result a = scenario(true, StoreCore::TestRule::kNone, dir + "/");
    bool clean = true;
    std::string why;
    for (int k = 0; k < 4; ++k) clean = clean && a.error[k].empty(), why += a.error[k];
    check(clean, "A: every rank arrived at every collective, nothing stopped: " + why);
    check(all_equal(a.digest_after_ride) && all_equal(a.digest_after_return), "A: the four digests are equal after the ride and after the return");
    bool counts = true;
    for (int k = 0; k < 4; ++k) counts = counts && a.revives[k] == 34 && a.read[k] == 0;
    check(counts, "A: revives = 34 pieces on every rank, nothing read");
  }
  {
    const Result b = scenario(false, StoreCore::TestRule::kNone, dir + "/");
    check(!all_equal(b.digest_after_ride), "B: the ride on rank 0 alone makes the digests differ at the first compare");
    bool named = true;
    std::string seen;
    for (int k = 0; k < 4; ++k) {
      named = named && b.error[k].find("out on ranks 0, not on ranks 1,2,3") != std::string::npos;
      seen += " [" + b.error[k] + "]";
    }
    check(named, "B: the return stops every rank naming who is out:" + seen);
  }
  {
    const ForkResult f = fork_scenario(true, StoreCore::TestRule::kNone, dir + "/");
    std::string why;
    bool ok = true, once = true, same = all_equal(f.digest);
    for (int k = 0; k < 4; ++k) ok = ok && f.error[k].empty(), why += f.error[k], once = once && f.collectives[k] == 1 && f.lines[k] == 1;
    check(ok, "fork (marked base): nothing stopped: " + why);
    std::string got;
    for (int k = 0; k < 4; ++k) got += " [" + std::to_string(f.collectives[k]) + " collectives, " + std::to_string(f.lines[k]) + " lines]";
    check(once, "fork (marked base): exactly ONE return a rank, one collective and one line:" + got);
    check(same, "fork (marked base): the four digests are equal");
  }
  {
    const ForkResult f = fork_scenario(false, StoreCore::TestRule::kNone, dir + "/");
    bool ok = true, zero = true;
    for (int k = 0; k < 4; ++k) ok = ok && f.error[k].empty(), zero = zero && f.collectives[k] == 0 && f.lines[k] == 0;
    check(ok && zero && all_equal(f.digest), "fork (whole base): zero returns, zero collectives, zero lines, digests equal");
  }
  {  // a rank that skips (or goes to another site) while the others call still dies naming both sites
    Meet meet;
    std::vector<std::string> err(4);
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
      th.emplace_back([&, k] {
        try {
          uint64_t mine[4] = {0, 0, 0, 0}, out[16];
          meet.meet(k, k == 0 ? kSiteDiskRoom : kSiteUnspill, mine, out);
        } catch (const std::exception& e) {
          err[k] = e.what();
        }
      });
    for (auto& t : th) t.join();
    bool named = true;
    for (int k = 0; k < 4; ++k) named = named && err[k].find("is in site") != std::string::npos && err[k].find("out of step") != std::string::npos;
    check(named, "a rank in another site than its partners: every rank stops naming both sites: " + err[1]);
  }
#ifndef H21_OLD_TREE
  {
    const ForkResult f = fork_scenario(true, StoreCore::TestRule::kUnspillNoOpReturn, dir + "/");
    check(f.collectives[0] == 3 && f.lines[0] == 3, "control kUnspillNoOpReturn (the earlier behaviour): the fork makes three returns: " + std::to_string(f.collectives[0]) + " collectives, " + std::to_string(f.lines[0]) + " lines");
  }
#endif
#ifndef H21_OLD_TREE
  {
    const Result c = scenario(false, StoreCore::TestRule::kUnspillReturnsAlone, dir + "/");
    check(c.error[0].find("waits at a collective: only ranks 0 arrived") != std::string::npos, "C (control kUnspillReturnsAlone): rank 0 waits at the collective alone: " + c.error[0]);
    check(c.error[1].empty() && c.error[2].empty() && c.error[3].empty(), "C: ranks 1 to 3 returned before the collective, without a word");
  }
#endif
  std::printf(g_fail ? "four_rank_test FAILED %d\n" : "four_rank_test passed\n", g_fail);
  return g_fail != 0;
}
