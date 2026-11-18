// model/lend.h against a pool made of the real allocator and the real VMM pool over a fake driver: the groups a borrower
// takes leave exactly 64 numbers and 89 chunk handles each, come back to the same numbers and handles, a window and the
// Engram block share one ledger, groups that cannot come back are owed and settled later without killing anything (one
// rank failing leaves four ranks equal), the proof's tolerance has a floor, a setting past what the store can lend is cut
// and a failed size is not retried at once. Built with -DPICO_OLD_RULES the cases marked "red" fail (the last of them
// kills the process, as a failed grow does without the rule). Host only: no engine, no GPU.
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "model/chunks.h"
#include "model/lend.h"
#include "model/vmm.h"

namespace lend = model::lend;
namespace vmm = model::vmm;

namespace {

int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  std::fflush(stdout);
  if (!ok) ++fails;
}

// a fake driver whose chunk creation can be made to fail
class Flaky : public vmm::FakeBackend {
 public:
  bool fail = false;
  bool create(uint64_t* h) override { return !fail && vmm::FakeBackend::create(h); }
};

// bytes of one number in each of four arrays: 64 numbers are 32 + 32 + 16 + 9 = 89 chunks of 2 MiB
constexpr uint64_t kPerNumber[4] = {1048576, 1048576, 524288, 294912};
constexpr uint64_t kGroupBytes = 64 * (kPerNumber[0] + kPerNumber[1] + kPerNumber[2] + kPerNumber[3]);  // 186,646,528

// One rank: the number allocator and the memory pool behind it, and the Lender's backend over both
struct Rank {
  Flaky* fb = nullptr;
  std::unique_ptr<vmm::Pool> pool;
  std::unique_ptr<model::chunks::Alloc> calloc;
  std::function<bool(bool)> agree = [](bool local) { return local; };  // the ranks' minimum of "this one could"
  std::vector<uint32_t> held;

  static size_t chunks_for(size_t nums) {
    size_t c = 0;
    for (const uint64_t b : kPerNumber) c += (nums * b + vmm::kChunk - 1) / vmm::kChunk;
    return c;
  }
  static void bytes_for(size_t nums, size_t* out) {
    for (int a = 0; a < 4; ++a) out[a] = (nums * kPerNumber[a] + vmm::kChunk - 1) / vmm::kChunk * vmm::kChunk;
  }
  explicit Rank(size_t numbers) {
    auto b = std::make_unique<Flaky>();
    fb = b.get();
    pool = std::make_unique<vmm::Pool>(std::move(b), chunks_for(numbers));
    for (int a = 0; a < 4; ++a) pool->add_array(chunks_for(20000) * vmm::kChunk, 1);
    pool->add_slot(0, 1);
    size_t bytes[4];
    bytes_for(numbers, bytes);
    pool->ensure(0, bytes);
    calloc = std::make_unique<model::chunks::Alloc>((uint32_t)numbers);
  }
  // the lender's backend
  size_t numbers() const { return calloc->numbers(); }
  bool vacate(size_t top) { return calloc->first_taken_from((uint32_t)top) != 0; }
  bool make_room(size_t) { return true; }
  bool shrink_to(size_t top) {
    const size_t before = pool->total();
    const size_t was = calloc->numbers();
    if (!calloc->shrink((uint32_t)top)) return false;
    size_t bytes[4];
    bytes_for(top, bytes);
    pool->shrink(0, bytes);
    pool->trim(pool->free_chunks());
    return before - pool->total() == chunks_for(was) - chunks_for(top);
  }
  bool release(size_t top, int) { return shrink_to(top); }
  bool try_grow(size_t more) {
    const size_t nums = calloc->numbers() + more;
    size_t bytes[4], want = 0;
    bytes_for(nums, bytes);
    for (int a = 0; a < 4; ++a) want += bytes[a] / vmm::kChunk;
    const size_t have = pool->total();
    size_t added = 0;
    if (want > have) {
      added = pool->add_chunks(want - have);
      if (added != want - have) {
        pool->trim(added);
        return false;
      }
    }
    if (!pool->ensure(0, bytes)) {
      pool->trim(added);
      return false;
    }
    calloc->grow((uint32_t)nums);
    return true;
  }
  bool grow_all(size_t more) {
    const size_t was = calloc->numbers();
    const bool local = try_grow(more);
    const bool all = agree(local);
    if (local && !all) shrink_to(was);  // another rank could not: this one goes back to where it was
    return all;
  }
  size_t handles() const { return pool->total(); }
};

}  // namespace

int main() {
  const size_t kN = 7616;  // numbers at the start: a multiple of 64
  const size_t kWhole = 40;

  check(kGroupBytes == 89 * vmm::kChunk, "fixture: 64 numbers are 89 chunks of 2 MiB (186,646,528 bytes)");

  // the sizes of the Engram block (the design's table)
  {
    const uint64_t row = 264, disk = 2032 * lend::kMiB;
    const auto g = [&](uint64_t mib, uint64_t d) { return lend::sizes(mib, d, row, kGroupBytes).groups; };
    check(g(2032, disk) == 1, "2,032 MiB all in the display reservation: the index alone, 1 group");
    check(g(4096, disk) == 14, "4,096 MiB with the display reservation: 14 groups");
    check(g(2048, 0) == 13, "2,048 MiB without it: 13 groups");
    check(g(4096, 0) == 25, "4,096 MiB without it: 25 groups");
    check(g(6912, 0) == 42, "6,912 MiB (6.75 GiB) without it: 42 groups");
    const lend::Sizes s = lend::sizes(4096, disk, row, kGroupBytes);
    check(s.ordinary_bytes / lend::kMiB == 2063 || s.ordinary_bytes / lend::kMiB == 2064, "4,096 MiB: about 2,064 MiB of ordinary slots");
    check(s.index_bytes / lend::kMiB >= 267 && s.index_bytes / lend::kMiB <= 269, "4,096 MiB: an index of about 268 MiB");
    check(lend::sizes(0, disk, row, kGroupBytes).groups == 0, "no cache: no groups");
  }

  // the proof's tolerance: the null control's spread, with a floor of 256 MiB
  {
    const uint64_t released = 14 * 178 * lend::kMiB;  // 14 groups
    lend::Reading a{10000, 12000 * lend::kMiB, 0}, b = a;
    b.handles = a.handles - 14 * 89;
    b.avail = a.avail + released - 150 * lend::kMiB;  // a good cycle that lands 150 MiB short of the expectation
    check(lend::proof_ab(a, b, 14 * 89, released, lend::tol_of(12 * lend::kMiB)) == lend::kProofOk,
          "red: a good hand-over that lands 150 MiB short passes with the null spread of 12 MiB (the tolerance has a floor)");
    check(lend::tol_of(300 * lend::kMiB) == 300 * lend::kMiB, "a null spread above the floor is the tolerance");
    check(lend::tol_of(0) == 256 * lend::kMiB, "the floor is 256 MiB");
    lend::Reading leak = b;
    leak.handles = a.handles - 14 * 89 + 1;
    check(lend::proof_ab(a, leak, 14 * 89, released, 256 * lend::kMiB) == lend::kProofHandles, "a leaked handle fails the handle count");
    lend::Reading stuck = a;
    stuck.handles = b.handles;
    stuck.avail = a.avail + 1000 * lend::kMiB;  // 4 groups' worth of 14 reached the system
    check(lend::proof_ab(a, stuck, 14 * 89, released, 256 * lend::kMiB) == lend::kProofAvail, "memory that did not reach the system fails MemAvailable");
    lend::Reading one_a{500, 8000 * lend::kMiB, 0}, one_b{411, 8000 * lend::kMiB, 0};  // one group, no rise at all
    check(lend::proof_ab(one_a, one_b, 89, 178 * lend::kMiB, 256 * lend::kMiB) == lend::kProofOk,
          "at one group MemAvailable cannot discriminate (the floor is above 178 MiB); the handle count decides");
    check(lend::proof_ab(one_a, one_a, 89, 178 * lend::kMiB, 256 * lend::kMiB) == lend::kProofHandles, "one group: no handle gone fails");
    lend::Reading c = b, bb = b;
    bb.rss = 100 * lend::kMiB;
    c.rss = bb.rss + 2332 * lend::kMiB;
    c.avail = a.avail - 40 * lend::kMiB;
    check(lend::proof_c(a, bb, c, 2332 * lend::kMiB, 256 * lend::kMiB) == lend::kProofOk, "C: the block populated and the memory back to A within tol");
    c.rss = bb.rss + 1000 * lend::kMiB;
    check(lend::proof_c(a, bb, c, 2332 * lend::kMiB, 256 * lend::kMiB) == lend::kProofPages, "C: a block that was not populated fails the pages reading");
    c.rss = bb.rss + 2332 * lend::kMiB;
    c.avail = a.avail - 900 * lend::kMiB;
    check(lend::proof_c(a, bb, c, 2332 * lend::kMiB, 256 * lend::kMiB) == lend::kProofAvail, "C: memory missing against A fails MemAvailable");
    check(lend::proof_shrink(3000 * lend::kMiB, 700 * lend::kMiB, 2300 * lend::kMiB, 100, 100 + 14 * 89, 14 * 89) == lend::kProofOk, "a shrink's proof");
    check(lend::proof_shrink(3000 * lend::kMiB, 2900 * lend::kMiB, 2300 * lend::kMiB, 100, 100 + 14 * 89, 14 * 89) == lend::kProofPages,
          "a shrink that decommitted nothing fails");
  }

  // the resize plan: a setting past what the store can lend is cut, never refused
  {
    const uint64_t row = 264;
    const lend::Plan p = lend::plan_resize(12000, 0, 0, row, kGroupBytes, 20, 16384);
    check(p.status != lend::kPlanRefused && p.cut && p.groups <= 20 && p.mib > 0 && p.mib < 12000,
          "red: 12,000 MiB with 20 groups to lend is cut to the largest size that fits, not refused");
    check(p.status == lend::kPlanGrow && lend::sizes(p.mib + 1, 0, row, kGroupBytes).groups > 20, "the cut size is the largest: one MiB more does not fit");
    const lend::Plan q = lend::plan_resize(6144, 4096, 2032 * lend::kMiB, row, kGroupBytes, 60, 16384);
    check(q.status == lend::kPlanGrow && !q.cut && q.mib == 6144, "a setting that fits is taken whole");
    const lend::Plan r = lend::plan_resize(2048, 6144, 0, row, kGroupBytes, 60, 16384);
    check(r.status == lend::kPlanShrink && r.mib == 2048, "a smaller setting is a shrink");
    check(lend::plan_resize(4096, 4096, 0, row, kGroupBytes, 60, 16384).status == lend::kPlanSame, "the same size is no change");
    const lend::Plan m = lend::plan_resize(30000, 0, 0, row, kGroupBytes, 200, 16384);
    check(m.cut && m.mib == 16384, "past the block's address space it is cut to the maximum");
    check(lend::max_total_groups(7616, 40) == (7616 - 40 - 2) / 64, "the most groups the pool can lend keeps one conversation");
  }

  // the back-off
  {
    lend::Backoff b;
    check(!b.blocked(4096, 0, 1), "nothing failed yet: nothing blocked");
    b.fail(4096, 100, 1);
    check(b.blocked(4096, 130, 1), "red: the same size is not tried again 30 s after it failed");
    check(!b.blocked(6144, 130, 1), "another size is tried");
    check(!b.blocked(4096, 130, 2), "the same size is tried at once when the setting changed");
    check(!b.blocked(4096, 100 + lend::kBackoffS + 1, 1), "after five minutes the same size is tried again");
  }

  // takes and give-backs over the real allocator and pool
  {
    Rank r(kN);
    lend::Ledger led;
    lend::Lender<Rank> L(r, led);
    const size_t h0 = r.handles();
    check(L.take(14, kWhole) == lend::kOk, "an Engram take of 14 groups");
    check(r.numbers() == kN - 14 * 64 && led.groups == 14 && led.n0 == kN, "14 groups are 896 numbers out of the pool");
    check(h0 - r.handles() == 14 * 89, "and exactly 14 x 89 chunk handles went to the driver");
    check(L.take(2, kWhole) == lend::kOk, "a window takes 2 more groups from the same ledger");
    check(r.numbers() == kN - 16 * 64 && h0 - r.handles() == 16 * 89, "16 groups out: 1,024 numbers and 1,424 handles");
    check(L.give(14) && led.groups == 2 && r.numbers() == kN - 2 * 64, "the Engram's 14 groups come back while the window stays");
    check(L.give(2) && led.groups == 0 && led.n0 == 0 && r.numbers() == kN && r.handles() == h0, "the window's groups come back: the start's numbers and handles exactly");
  }
  {  // a running request holds a number the cut takes: busy, nothing left the pool
    Rank r2(kN);
    lend::Ledger led2;
    lend::Lender<Rank> L2(r2, led2);
    std::vector<uint32_t> all;
    check(r2.calloc->take((uint32_t)r2.calloc->free(), &all), "fixture: every number taken");
    const size_t h0 = r2.handles();
    check(L2.take(1, kWhole) == lend::kBusy && r2.numbers() == kN && r2.handles() == h0 && led2.groups == 0,
          "a holder above the cut answers busy and nothing leaves the pool");
    check(L2.take(200, kWhole) == lend::kNoRoom, "a cut that would leave no room for one conversation is refused");
  }
  {  // a mismatch of the driver's handles: the groups are back
    Rank r(kN);
    lend::Ledger led;
    lend::Lender<Rank> L(r, led);
    struct Bad {
      Rank& r;
      size_t numbers() const { return r.numbers(); }
      bool vacate(size_t t) { return r.vacate(t); }
      bool make_room(size_t n) { return r.make_room(n); }
      bool release(size_t top, int g) { return r.release(top, g) && false; }  // the arithmetic disagrees
      bool grow_all(size_t n) { return r.grow_all(n); }
    } bad{r};
    lend::Lender<Bad> LB(bad, led);
    const size_t h0 = r.handles();
    check(LB.take(3, kWhole) == lend::kFailed && led.groups == 0 && r.numbers() == kN && r.handles() == h0 && led.owed == 0,
          "a hand-over whose handle count is off puts its groups back");
  }

  // four ranks: one cannot make the chunks again
  {
    std::vector<std::unique_ptr<Rank>> rk;
    std::vector<lend::Ledger> led(4);
    for (int i = 0; i < 4; ++i) rk.push_back(std::make_unique<Rank>(kN));
    bool any_fail = false;
    for (auto& x : rk) x->agree = [&any_fail](bool) { return !any_fail; };
    const auto take_all = [&](int g) {
      int st = 0;
      for (int i = 0; i < 4; ++i) {
        lend::Lender<Rank> L(*rk[i], led[i]);
        st = L.take(g, kWhole);
      }
      return st;
    };
    check(take_all(14) == lend::kOk, "four ranks take 14 groups alike");
    rk[2]->fb->fail = true;  // rank 2 cannot create chunks now
    any_fail = true;
    bool all_false = true;
    int owed_after = 0;
    for (int i = 0; i < 4; ++i) {
      lend::Lender<Rank> L(*rk[i], led[i]);
      all_false &= !L.give(14);
      L.owe(14);
      owed_after += led[i].owed;
    }
    bool equal = true;
    for (int i = 1; i < 4; ++i)
      equal &= rk[i]->numbers() == rk[0]->numbers() && rk[i]->handles() == rk[0]->handles() && led[i].groups == led[0].groups;
    check(all_false && equal && rk[0]->numbers() == kN - 14 * 64 && owed_after == 4 * 14,
          "one rank failing leaves every rank on the smaller pool, the same on all four, the groups owed everywhere");
    rk[2]->fb->fail = false;
    any_fail = false;
    int st = 0;
    for (int i = 0; i < 4; ++i) {
      lend::Lender<Rank> L(*rk[i], led[i]);
      st = L.settle(true);
    }
    equal = true;
    for (int i = 0; i < 4; ++i) equal &= rk[i]->numbers() == kN && led[i].owed == 0 && led[i].groups == 0 && rk[i]->handles() == rk[0]->handles();
    check(st == lend::kOk && equal, "the retry gives the owed groups back on all four ranks: numbers and handles equal");
    {
      lend::Lender<Rank> L(*rk[0], led[0]);
      check(L.settle(false) == lend::kOk, "nothing owed: settle is a no-op");
    }
  }

  // red, the last: the give-back of groups the pool cannot make again must owe them. Without the rule it kills the process.
  {
    Rank r(kN);
    lend::Ledger led;
    lend::Lender<Rank> L(r, led);
    check(L.take(14, kWhole) == lend::kOk, "fixture: 14 groups out");
    r.fb->fail = true;  // the driver cannot create chunks
    const bool back = L.give(14);
    check(!back && led.groups == 14 && r.numbers() == kN - 14 * 64, "red: a give-back the driver cannot make fails cleanly: the ledger and the pool stay as they were");
    L.owe(14);
    check(L.settle(true) == lend::kFailed && led.owed == 14, "red: the retry fails and the groups stay owed, nothing dies");
    r.fb->fail = false;
    check(L.settle(true) == lend::kOk && led.owed == 0 && led.groups == 0 && r.numbers() == kN, "red: once the driver can, the owed groups come back");
  }

  std::printf("%s lend_test: %d failed\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
