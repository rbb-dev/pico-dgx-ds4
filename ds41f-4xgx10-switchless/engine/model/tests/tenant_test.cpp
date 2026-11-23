// model/lend.h's tenants' table and model/tenant.h's driver on the host, over the real number allocator and the real VMM pool
// on a fake driver: three tenants and the windows in mixed order, a grow in flight and groups owed (the invariant after every
// call, every tenant's room exact), the three steps with fake bodies (a body failing on one rank at its begin, at reading C,
// going back; a rank's readings disagreeing at B), the one queue of store changes (a window queued during a change waits; with
// the rule off its allocation falls between readings B and C and a good change is rolled back), the order tenants give way in,
// and the digest's tenant terms. Built with -DPICO_OLD_RULES the cases marked "red" fail. Host only: no engine, no GPU.
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "model/chunks.h"
#include "model/store_core.h"
#include "model/tenant.h"
#include "model/tests/store_fake.h"
#include "model/vmm.h"
#include "server/lend_pace.h"

namespace lend = model::lend;
namespace tenant = model::tenant;
namespace vmm = model::vmm;

namespace {

int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  std::fflush(stdout);
  if (!ok) ++fails;
}

class Flaky : public vmm::FakeBackend {
 public:
  bool fail = false;
  bool create(uint64_t* h) override { return !fail && vmm::FakeBackend::create(h); }
};

constexpr uint64_t kPerNumber[4] = {1048576, 1048576, 524288, 294912};
constexpr uint64_t kGroupBytes = 64 * (kPerNumber[0] + kPerNumber[1] + kPerNumber[2] + kPerNumber[3]);  // 89 chunks
constexpr uint64_t kRam = 96ull << 30;  // the fake system's memory: MemAvailable = this less the pool's chunks less what bodies use

// One rank: the number allocator and the memory pool behind it, the system's memory, and what tenants and windows use of it.
struct Rank {
  Flaky* fb = nullptr;
  std::unique_ptr<vmm::Pool> pool;
  std::unique_ptr<model::chunks::Alloc> calloc;
  uint64_t used = 0;  // bytes in use outside the pool: the tenants' blocks, a window's buffers

  static size_t chunks_for_n(size_t nums) {
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
    pool = std::make_unique<vmm::Pool>(std::move(b), chunks_for_n(numbers));
    for (int a = 0; a < 4; ++a) pool->add_array(chunks_for_n(20000) * vmm::kChunk, 1);
    pool->add_slot(0, 1);
    size_t bytes[4];
    bytes_for(numbers, bytes);
    pool->ensure(0, bytes);
    calloc = std::make_unique<model::chunks::Alloc>((uint32_t)numbers);
  }
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
    return before - pool->total() == chunks_for_n(was) - chunks_for_n(top);
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
  bool grow_all(size_t more) { return try_grow(more); }
  size_t handles() const { return pool->total(); }
  uint64_t avail() const { return kRam - handles() * vmm::kChunk - used; }
};

// The pool's side and the ranks' agreement for the driver: this rank is real, the others are a script: `peer_bad[step]` makes
// the other ranks say no at that step once; `fault_at[step]` is this rank's own injected failure.
struct Env {
  Rank& r;
  int later = 0;
  size_t whole_n = 40;
  bool peer_bad[8] = {}, fault_at[8] = {};
  int votes[8] = {};
  size_t numbers() const { return r.numbers(); }
  bool vacate(size_t t) { return r.vacate(t); }
  bool make_room(size_t n) { return r.make_room(n); }
  bool release(size_t t, int g) { return r.release(t, g); }
  bool grow_all(size_t m) { return r.grow_all(m); }
  uint64_t handles() const { return r.handles(); }
  uint64_t avail() const { return r.avail(); }
  uint64_t chunks_for(size_t n) const { return Rank::chunks_for_n(n); }
  uint64_t chunk_bytes() const { return vmm::kChunk; }
  size_t whole() const { return whole_n; }
  size_t room_whole() const { return whole_n; }
  int later_groups() { return later; }
  uint64_t null_bytes() const { return 12 * lend::kMiB; }
  bool fault(tenant::Step s) {
    const bool f = fault_at[s];
    fault_at[s] = false;
    return f;
  }
  bool peer(tenant::Step s) {
    ++votes[s];
    const bool p = peer_bad[s];
    peer_bad[s] = false;
    return p;
  }
  uint64_t agree_min(uint64_t v, tenant::Step) { return v; }
  bool agree_all(bool ok, tenant::Step s) { return !peer(s) && ok; }
  bool agree_any(bool bad, tenant::Step s) { return peer(s) || bad; }
};

// A tenant's body: its size is its groups' bytes, committed from the system's memory (the fake `used`) on begin; Done after
// `delay` polls.
struct Body {
  Rank& r;
  uint64_t cur = 0, rss_before = 0, rss_after = 0;
  bool fail_begin = false, fail_state = false, stuck = false;  // this rank's failures
  int delay = 1, polls = 0;
  int last_room = -1, planned = 0;
  int notes[8] = {};
  bool running_ = false;
  uint64_t target = 0;
  tenant::BodyPlan plan(uint64_t want, int room) {
    last_room = room;
    tenant::BodyPlan p;
    int g = (int)((want + kGroupBytes - 1) / kGroupBytes);
    if (g > room) g = room, p.cut = true;
    p.groups = g, p.bytes = (uint64_t)g * kGroupBytes;
    const uint64_t cg = cur / kGroupBytes;
    p.status = (uint64_t)g == cg ? lend::kPlanSame : (uint64_t)g > cg ? lend::kPlanGrow : lend::kPlanShrink;
    return p;
  }
  uint64_t size_for_groups(int g) const { return (uint64_t)g * kGroupBytes; }
  bool begin(uint64_t bytes) {
    if (fail_begin) {
      fail_begin = false;
      running_ = false;
      return false;
    }
    if (stuck && bytes < cur) {  // could not drop what it holds
      running_ = true, polls = 0, target = cur;
      return true;
    }
    rss_before = cur;
    target = bytes;
    running_ = true, polls = 0;
    return true;
  }
  tenant::BodyState state() {
    if (!running_) return tenant::kBodyFailed;
    if (++polls <= delay) return tenant::kBodyRunning;
    if (fail_state) {
      fail_state = false;
      running_ = false;
      return tenant::kBodyFailed;
    }
    if (target != cur) {
      if (target > cur) r.used += target - cur;
      else r.used -= cur - target;
      cur = target;
    }
    rss_after = cur;
    running_ = false;
    return tenant::kBodyDone;
  }
  bool meter = false;  // report like a device block (tenant::block_info) instead of the fixture's own bookkeeping
  tenant::BodyInfo info() const {
    if (meter) return tenant::block_info(rss_before, cur);
    tenant::BodyInfo i;
    i.block_bytes = cur, i.rss_before = rss_before, i.rss_after = rss_after;
    return i;
  }
  bool at_size(uint64_t bytes) const { return cur == bytes; }
  void ack() {}
  void say(tenant::Note n, const tenant::Change&, const tenant::BodyInfo&, int, bool) { ++notes[n]; }
};

constexpr size_t kN = 7616;  // numbers at the start: a multiple of 64
constexpr int kMaxTotal = (int)((kN - 40 - 2) / 64);  // groups the pool can lend while it holds one conversation of 40 numbers

struct Fixture {
  Rank r;
  Env e{r};
  lend::Ledger led;
  lend::Tenants tn;
  tenant::Driver<Env> d{e, led, tn};
  Body eng{r}, dummy{r};
  explicit Fixture(size_t n = kN) : r(n) {
    tn.add(lend::kTenantWindows, lend::kKindDevice, lend::kRequired);
    tn.add(lend::kTenantEngram, lend::kKindHost, lend::kCache, 1);
    tn.add(15, lend::kKindDevice, lend::kOnDemand);
  }
  int finish(int id, Body& b, int begin_status, int limit = 8) {
    int st = begin_status;
    for (int i = 0; i < limit && st == tenant::kPending; ++i) st = d.poll(b);
    (void)id;
    return st;
  }
};

}  // namespace

int main() {
  check(kGroupBytes == 89 * vmm::kChunk, "fixture: 64 numbers are 89 chunks of 2 MiB");
  check(kMaxTotal == 118, "fixture: the pool lends 118 groups at most with one 40-number conversation held");

  // ---- the table: three tenants and the windows, mixed order, a grow in flight, one group owed ----
  {
    Fixture f;
    lend::Lender<Env> L(f.e, f.led);
    bool inv = true;
    L.take(3, 40);  // a window's three groups
    f.tn.find(lend::kTenantWindows)->groups = 3;
    inv &= f.tn.consistent(f.led);
    L.take(5, 40);  // the Engram cache's five
    f.tn.find(lend::kTenantEngram)->groups = 5;
    inv &= f.tn.consistent(f.led);
    L.take(2, 40);  // tenant 15's grow in flight: left the pool, not yet its own
    f.tn.find(15)->pending = 2;
    inv &= f.tn.consistent(f.led);
    L.take(1, 40);  // a group a borrower finished with and the pool could not make again
    L.owe(1);
    inv &= f.tn.consistent(f.led);
    check(inv && f.led.groups == 11, "the invariant sum(groups) + sum(pending) + owed = the ledger's groups holds after every call");
    const int later = 4;
    const int exact = kMaxTotal - (3 + 5) - 1 - later;  // tenant 15's room: nothing of the others' besides, its own pending is not counted
    check(f.tn.room(15, f.led, kN, 40, later) == exact, "red: a third tenant's room: the others' groups and pending, the owed and the windows to come, not its own pending in flight");
    const int engram_exact = kMaxTotal - 3 - 2 - 1 - later;  // the Engram cache's room counts tenant 15's grow in flight
    check(f.tn.room(lend::kTenantEngram, f.led, kN, 40, later) == engram_exact, "the Engram cache's room counts a grow in flight of another tenant");
    check(f.tn.room(lend::kTenantWindows, f.led, kN, 40, later) == kMaxTotal - 5 - 2 - 1 - later, "the windows' room has the same arithmetic");
    // the control: today's rule, "the others = lent - mine" with `mine` the settled groups, copied here
    const int today = std::max(0, kMaxTotal - (f.led.groups - f.tn.groups_of(15)) - later);
    check(today != exact, "control: today's rule gives a wrong room for a third tenant with a grow in flight");
    check(f.tn.room(15, f.led, kN, 40, 200) == 0, "a room is never negative");
    // the grow lands: pending becomes groups
    f.tn.find(15)->groups = 2, f.tn.find(15)->pending = 0;
    check(f.tn.consistent(f.led), "after the grow lands the invariant holds");
    // a shrink gives groups back
    check(L.give(2), "tenant 15's two groups go back to the pool");
    f.tn.find(15)->groups = 0;
    check(f.tn.consistent(f.led) && f.led.groups == 9, "after the give-back the invariant holds");
  }

  // ---- the driver: a grow and a shrink of the Engram cache's tenant ----
  {
    Fixture f;
    const uint64_t h0 = f.r.handles();
    int st = f.d.begin(lend::kTenantEngram, f.eng, 5 * kGroupBytes);
    check(st == tenant::kPending && f.d.active() && f.tn.pending_of(1) == 5 && f.tn.groups_of(1) == 0 && f.tn.consistent(f.led),
          "a grow begins: its groups left the pool and are pending, not yet the tenant's");
    check(f.r.handles() == h0 - 5 * 89 && f.r.numbers() == kN - 5 * 64, "the pool gave up exactly 64 numbers and 89 chunks a group");
    check(f.eng.last_room == kMaxTotal, "the body was planned with the table's room");
    st = f.finish(1, f.eng, st);
    check(st == tenant::kDone && !f.d.active() && f.tn.groups_of(1) == 5 && f.tn.pending_of(1) == 0 && f.tn.consistent(f.led) &&
              f.tn.find(1)->size == 5 * kGroupBytes && f.d.counts(1).ok == 1,
          "the grow lands after the poll: groups booked, pending cleared, size kept");
    check(f.eng.notes[tenant::kNoteGrown] == 1, "the body said its line once");
    // the same size again: nothing to do
    check(f.d.begin(lend::kTenantEngram, f.eng, 5 * kGroupBytes) == tenant::kSame, "the same size is no change");
    // a shrink to two groups
    st = f.d.begin(lend::kTenantEngram, f.eng, 2 * kGroupBytes);
    check(st == tenant::kPending && f.tn.groups_of(1) == 5 && f.tn.pending_of(1) == 0 && f.tn.consistent(f.led),
          "a shrink in flight has no pending: its groups leave the tenant when they are back");
    st = f.finish(1, f.eng, st);
    check(st == tenant::kDone && f.tn.groups_of(1) == 2 && f.led.groups == 2 && f.r.numbers() == kN - 2 * 64 && f.tn.consistent(f.led),
          "the shrink lands: three groups are back in the pool");
    check(f.eng.notes[tenant::kNoteShrunk] == 1, "the shrink said its line");
  }

  // ---- fake bodies failing ----
  {
    Fixture f;  // a body failing on this rank at its begin
    f.eng.fail_begin = true;
    int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    st = f.finish(1, f.eng, st);
    check(st == tenant::kFailed && f.tn.groups_of(1) == 0 && f.tn.pending_of(1) == 0 && f.led.groups == 0 && f.led.owed == 0 &&
              f.r.numbers() == kN && f.d.counts(1).failed == 1 && f.tn.consistent(f.led) && f.eng.cur == 0,
          "a body that cannot begin on one rank: the change fails, the groups are back, nothing is owed");
  }
  {
    Fixture f;  // the other ranks fail at reading C
    f.e.peer_bad[tenant::kStepBad] = true;
    int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    st = f.finish(1, f.eng, st);
    check(st == tenant::kFailed && f.led.groups == 0 && f.led.owed == 0 && f.eng.cur == 0 && f.r.numbers() == kN && f.tn.consistent(f.led) &&
              f.r.used == 0,
          "a body failing on another rank: this rank's block goes back too, the groups are back");
  }
  {
    Fixture f;  // this rank's reading C fails
    f.e.fault_at[tenant::kStepC] = true;
    int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    st = f.finish(1, f.eng, st);
    check(st == tenant::kFailed && f.led.groups == 0 && f.eng.cur == 0 && f.tn.consistent(f.led) && f.d.counts(1).failed == 1,
          "reading C failing on this rank: the change goes back on all");
  }
  {
    Fixture f;  // a rank's readings disagree at B: rolled back at the begin
    f.e.peer_bad[tenant::kStepB] = true;
    const int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    check(st == tenant::kFailed && f.led.groups == 0 && f.r.numbers() == kN && f.tn.groups_of(1) == 0 && f.tn.pending_of(1) == 0 &&
              !f.d.active() && f.tn.consistent(f.led) && f.eng.cur == 0,
          "a rank's readings disagree at B: the groups go back at the begin, the body never started");
  }
  {
    Fixture f;  // a body that fails and cannot go back: its groups are owed
    f.e.fault_at[tenant::kStepC] = true;
    f.eng.delay = 0;
    int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    // the block was made (cur = 4 groups); going back it cannot drop it
    f.eng.stuck = true;
    st = f.finish(1, f.eng, st);
    check(st == tenant::kFailed && f.led.owed == 4 && f.tn.groups_of(1) == 0 && f.tn.pending_of(1) == 0 && f.tn.consistent(f.led) &&
              f.d.counts(1).proof_failed == 1 && f.eng.notes[tenant::kNoteStuck] == 1,
          "a block that did not go back: its groups are owed, not made again beside its pages");
    f.eng.stuck = false;
    lend::Lender<Env> L(f.e, f.led);
    check(L.settle(true) == lend::kOk && f.led.owed == 0 && f.led.groups == 0 && f.r.numbers() == kN, "the owed groups come back when the memory is free");
  }
  {
    Fixture f;  // a failed shrink: the body could not drop its pages; the groups never left the tenant
    int st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 5 * kGroupBytes));
    check(st == tenant::kDone, "fixture: five groups held");
    f.eng.stuck = true;
    st = f.d.begin(lend::kTenantEngram, f.eng, 2 * kGroupBytes);
    f.eng.stuck = false;
    st = f.finish(1, f.eng, st);
    check(st == tenant::kFailed && f.tn.groups_of(1) == 5 && f.led.groups == 5 && f.led.owed == 0 && f.tn.consistent(f.led),
          "a shrink whose body could not drop its part fails with the groups still the tenant's");
  }

  // ---- the room: later is kept and the table's room reaches the body ----
  {
    Fixture f;
    f.e.later = 10;
    f.tn.find(lend::kTenantWindows)->groups = 0;
    f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 200 * kGroupBytes, false));
    check(f.eng.last_room == kMaxTotal - 10, "the groups of the windows still to open are kept out of every tenant's room");
    check(f.tn.groups_of(1) == kMaxTotal - 10 && f.eng.cur == (uint64_t)(kMaxTotal - 10) * kGroupBytes, "a size past the room is cut to it");
  }
  {
    Fixture f;  // a third tenant's grow takes room another tenant is then refused
    f.finish(15, f.dummy, f.d.begin(15, f.dummy, 100 * kGroupBytes, false));
    check(f.tn.groups_of(15) == 100, "fixture: tenant 15 holds 100 groups");
    const int st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 40 * kGroupBytes, false));
    check(st == tenant::kDone && f.tn.groups_of(1) == kMaxTotal - 100, "the Engram cache's room is what tenant 15 left");
    check(f.tn.consistent(f.led), "the table adds up with two tenants grown");
  }

  // ---- the one queue ----
  {
    Fixture f;
    // a window queued during a change waits; the change ends; the window runs
    int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    check(st == tenant::kPending && f.d.queue.change, "fixture: a change is in flight");
    bool window_ran = false;
    const int64_t window_bytes = 600ll << 20;  // a window's buffers: more than the proof's tolerance
    int polls = 0;
    while (st == tenant::kPending && polls++ < 8) {
      if (f.d.queue.window_may_run() && !window_ran) f.r.used += window_bytes, window_ran = true;  // the server runs the window now
      st = f.d.poll(f.eng);
    }
    check(st == tenant::kDone && !window_ran, "red: while a change is in flight the window waits and the good change lands");
    if (!window_ran) {
      check(f.d.queue.window_may_run(), "the change is over: the window may run");
      f.r.used += window_bytes;
    }
  }
  {
    Fixture f;  // the control: the window's allocation falls between readings B and C and a good change is rolled back
    int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    f.r.used += 600ll << 20;  // the rule off: the window ran
    st = f.finish(1, f.eng, st);
    check(st == tenant::kFailed && f.led.groups == 0, "control: with the queue's rule off a window's allocation between B and C rolls a good change back");
  }
  {
    Fixture f;  // a window command running: no change begins
    f.d.queue.window = true;
    const int st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
    check(st == tenant::kBusy && !f.d.active() && f.led.groups == 0, "red: a tenant change does not begin while a window command runs");
    f.d.queue.window = false;
    check(f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes) == tenant::kPending, "it begins when the window command is over");
  }

  // ---- a run that never ends ----
  // A batch of long replies is one decoding run for minutes. The fake run loop below has no end of its own: it asks lendpace
  // after every step whether to pause, and the fake server's tick between runs begins and polls the change as lend_tick does.
  {
    Fixture f;
    double now = 0, poll_at = 0;
    bool config_waiting = true;  // the setting (CONFIG test_tenant_groups) arrived during the run
    bool wanted = false;
    int steps = 0, runs = 0, landed_at = -1, cfg_seen_at = -1, st = tenant::kIdle;
    const int kTotal = 5000, kWithin = 300;  // 10 ms a step: 50 s of run, a change must land within 3 s of steps
    while (steps < kTotal && landed_at < 0) {
      ++runs;
      while (steps < kTotal) {  // one run: a decode step each 10 ms, never ended by the flights
        ++steps, now += 0.01;
        lendpace::State ps;
        ps.config_waiting = config_waiting, ps.change_active = f.d.active(), ps.begin_wanted = wanted && !f.d.active();
        ps.now = now, ps.poll_at = poll_at;
        if (lendpace::pause_due(ps)) break;
      }
      // between runs: the server reads the CONFIG, then lend_tick
      if (config_waiting) config_waiting = false, wanted = true, cfg_seen_at = steps;
      if (f.d.active() && now >= poll_at) {
        poll_at = now + 0.1;
        st = f.d.poll(f.eng);
      } else if (!f.d.active() && wanted) {
        st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
        poll_at = now + 0.05;
        if (st != tenant::kPending) wanted = false;
      }
      if (st == tenant::kDone) landed_at = steps, wanted = false;
    }
    check(landed_at > 0 && landed_at <= kWithin && runs > 2, "red: a change lands within 300 steps of a run that never ends on its own (the run pauses for the CONFIG, the begin and each poll)");
    check(cfg_seen_at == 1, "the waiting CONFIG is read at the run's first step");
    check(f.tn.groups_of(1) == 4 && f.tn.consistent(f.led), "the change that landed under the long run adds up");
  }
  {
    lendpace::State ps;
    ps.change_active = true, ps.now = 1.0, ps.poll_at = 1.1;
    check(!lendpace::pause_due(ps), "a change in flight whose poll is not due does not pause the run");
    ps.begin_wanted = false, ps.change_active = false;
    check(!lendpace::pause_due(ps), "nothing wanted, nothing waiting: the run goes on");
  }

  // ---- a run that never ends: a due settle, a due watch, STATS ----
  {
    // groups owed and a run that never ends: the settle must run within its interval (5 s), not at the run's end
    Fixture f;
    lend::Lender<Env> L(f.e, f.led);
    L.take(3, 40);
    L.owe(3);  // owed
    double now = 0, settled = 0;
    int settle_at = -1;
    for (int step = 1; step <= 3000 && settle_at < 0; ++step) {  // 10 ms a step
      now += 0.01;
      lendpace::State ps;
      ps.settle_due = f.led.owed > 0 && now - settled >= 5;
      ps.now = now;
      if (!lendpace::pause_due(ps)) continue;
      settled = now;  // between runs: lend_tick's settle
      if (L.settle(true) == lend::kOk) settle_at = step;
    }
    check(settle_at > 0 && settle_at <= 520 && f.led.owed == 0, "red: owed groups are settled within the interval (5 s) of a run that never ends");
  }
  {
    // the memory watch due during a run that never ends: it does not end the run (it rides it: server/tests/mem_watch_test.cpp)
    double now = 0, watch_at = 0;
    int paused = 0;
    for (int step = 1; step <= 3000; ++step) {
      now += 0.01;
      lendpace::State ps;
      ps.watch_due = now - watch_at >= 10;
      ps.now = now;
      if (lendpace::pause_due(ps)) watch_at = now, ++paused;
    }
    check(paused == 0, "red: the memory watch due every 10 s of a run that never ends pauses it never (it rides the run)");
  }
  {
    // STATS after a change, in a run that never ends: fresh within 0.5 s
    Fixture f;
    double now = 0, poll_at = 0, published = 0;
    bool forced = false, wanted = true;
    uint64_t stats_groups = 0;  // what the last STATS showed
    int landed_at = -1, fresh_at = -1, st = tenant::kIdle;
    for (int step = 1; step <= 5000 && fresh_at < 0; ++step) {
      now += 0.01;
      if (lendpace::publish_due(now, published, forced, true)) stats_groups = (uint64_t)f.tn.groups_of(1), published = now, forced = false;
      if (landed_at >= 0 && stats_groups == 4) fresh_at = step;
      lendpace::State ps;
      ps.change_active = f.d.active(), ps.begin_wanted = wanted && !f.d.active(), ps.now = now, ps.poll_at = poll_at;
      if (!lendpace::pause_due(ps)) continue;
      if (f.d.active() && now >= poll_at) {
        poll_at = now + 0.1;
        st = f.d.poll(f.eng);
      } else if (!f.d.active() && wanted) {
        st = f.d.begin(lend::kTenantEngram, f.eng, 4 * kGroupBytes);
        poll_at = now + 0.05;
        wanted = false;
      }
      if (st == tenant::kDone && landed_at < 0) landed_at = step, forced = true;
    }
    check(landed_at > 0 && fresh_at > 0 && fresh_at - landed_at <= 50, "red: STATS shows a landed change within 0.5 s in a run that never ends");
  }

  // ---- at most 16 groups a command ----
  {
    Fixture f;
    int st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 30 * kGroupBytes));
    check(st == tenant::kDone && f.tn.groups_of(1) == 16 && f.tn.consistent(f.led), "red: a 30-group grow moves 16 groups in its first command");
    st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 30 * kGroupBytes));
    check(st == tenant::kDone && f.tn.groups_of(1) == 30 && f.tn.consistent(f.led) && f.r.numbers() == kN - 30 * 64, "the second command moves the other 14");
    st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 0));
    check(st == tenant::kDone && f.tn.groups_of(1) == 14, "red: a 30-group shrink to nothing gives back 16 in its first command");
    st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 0));
    check(st == tenant::kDone && f.tn.groups_of(1) == 0 && f.led.groups == 0 && f.r.numbers() == kN, "and the other 14 in the second");
    Fixture g;  // the start's hand-over is not capped
    st = g.finish(1, g.eng, g.d.begin(lend::kTenantEngram, g.eng, 30 * kGroupBytes, false));
    check(st == tenant::kDone && g.tn.groups_of(1) == 30, "an uncapped begin (the start) moves 30 at once");
  }

  // ---- the pool's top against a group boundary ----
  // Index 0 is the reserved null number, so a pool of n numbers has n - 1 usable ones. A lend of g groups removes exactly 64 g usable
  // numbers from any top: on a group boundary (n = 64 k), one number past it (n = 64 k + 1, what the sizing rule makes of a whole
  // number of groups of usable numbers) and one short of the last whole group. The old rules cut at the boundary below n and so
  // took the numbers above it with the first group. The shrink's proof must add up in every case.
  for (const int extra : {0, 1, -1}) {
    const size_t n = kN + extra;  // on a boundary, one number past it, one short of the last
    const size_t above = n - n / 64 * 64;
    char what[160];
    Fixture f(n);
    f.eng.meter = true;
    const uint64_t h0 = f.r.handles();
    int st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 2 * kGroupBytes));
    std::snprintf(what, sizeof what, "red: n = %zu: the first 2 groups take exactly 128 numbers (%zu above the last whole group stay), the pool is %zu", n, above, f.r.numbers());
    check(st == tenant::kDone && f.r.numbers() == n - 128 && f.tn.consistent(f.led), what);
    st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 12 * kGroupBytes));
    std::snprintf(what, sizeof what, "n = %zu: 10 groups more take exactly 640 numbers", n);
    check(st == tenant::kDone && f.r.numbers() == n - 128 - 640, what);
    st = f.finish(1, f.eng, f.d.begin(lend::kTenantEngram, f.eng, 0));
    std::snprintf(what, sizeof what, "red: n = %zu: the shrink back lands, its proof adds up (%d noted), the pool is whole (%zu numbers, handles %llu = %llu)", n,
                  f.eng.notes[tenant::kNoteShrinkProof], f.r.numbers(), (unsigned long long)f.r.handles(), (unsigned long long)h0);
    check(st == tenant::kDone && f.tn.groups_of(1) == 0 && f.r.numbers() == n && f.r.handles() == h0 && f.d.counts(1).proof_failed == 0 &&
              f.eng.notes[tenant::kNoteShrinkProof] == 0,
          what);
  }

  // ---- giving way ----
  {
    lend::Tenants t;
    t.add(lend::kTenantWindows, lend::kKindDevice, lend::kRequired);
    t.add(lend::kTenantEngram, lend::kKindHost, lend::kCache, 1);
    t.add(lend::kTenantWork, lend::kKindDevice, lend::kOnDemand);
    t.add(15, lend::kKindDevice, lend::kOnDemand);
    t.find(lend::kTenantWindows)->groups = 2;
    t.find(lend::kTenantEngram)->groups = 6;
    t.find(lend::kTenantWork)->groups = 1;
    t.find(15)->groups = 2;
    check(t.next_victim() == lend::kTenantWork, "red: idle first: an idle work area gives before the Engram cache");
    t.find(lend::kTenantWork)->idle = false;
    check(t.next_victim() == 15, "red: the next idle OnDemand tenant after a busy one");
    t.find(15)->idle = false;
    check(t.next_victim() == lend::kTenantEngram, "red: then the Engram cache, above its floor");
    t.find(lend::kTenantEngram)->groups = 1;
    check(t.next_victim() == -1, "red: a Cache tenant at its floor and the windows are never named");
    t.find(lend::kTenantEngram)->groups = 6;
    t.find(lend::kTenantEngram)->pending = 1;
    check(t.next_victim() == -1, "red: a tenant with a grow in flight is not asked");
    t.find(lend::kTenantEngram)->pending = 0;
  }

  // ---- the prefill's lent row set gives way before a Cache tenant, and only when nothing reads on it ----
  {
    lend::Tenants t;
    t.add(lend::kTenantWindows, lend::kKindDevice, lend::kRequired);
    t.add(lend::kTenantEngram, lend::kKindHost, lend::kCache, 1);
    t.add(lend::kTenantPrefill, lend::kKindDevice, lend::kOnDemand);
    t.find(lend::kTenantWindows)->groups = 2;
    t.find(lend::kTenantEngram)->groups = 6;
    t.find(lend::kTenantPrefill)->groups = 24;
    check(t.next_victim() == lend::kTenantPrefill, "red: the lent prefill rows, idle, give way before the Engram cache");
    t.find(lend::kTenantPrefill)->idle = false;
    check(t.next_victim() == lend::kTenantEngram, "red: the lent prefill rows with a prompt reading on them are not the victim");
    t.find(lend::kTenantPrefill)->idle = true;
    t.find(lend::kTenantPrefill)->pending = 24;
    check(t.next_victim() == lend::kTenantEngram, "red: the lent prefill rows with a grow in flight are not asked");
  }

  // ---- the digest's tenant terms ----
  {
    model::detail::FakeDev dev;
    model::chunks::Alloc alloc(1000);
    model::sgeom::StoreGeom g;
    g.nslots = g.nunits = 4;
    g.list_len = 48;
    g.max_ctx = 262144;
    for (int s = 0; s < 4; ++s) g.unit_of[s] = s, g.slot_ctx[s] = 262144;
    g.image_bytes = 2916352, g.snap_bytes = 3u << 20, g.end_bytes = 5u << 20;
    model::detail::StoreCore core(g, dev, &alloc);
    model::detail::StoreCore::Outside o;
    o.groups = 8, o.cur_groups = 5, o.n0 = 7616;
    const uint64_t base = core.digest(o);
    check(core.digest(model::detail::StoreCore::Outside(o)) == base, "the digest of the same terms is the same");
    // the windows and the Engram cache are in the terms before: a store with only them hashes alike
    model::detail::StoreCore::Outside z = o;
    z.tenant_groups[0] = 3, z.tenant_groups[1] = 5;
    check(core.digest(z) == base, "tenants 0 and 1 add nothing: the digest equals the one without the table's terms");
    model::detail::StoreCore::Outside t = o;
    t.tenant_groups[15] = 2;
    check(core.digest(t) != base, "red: a tenant of id 2 or more with groups changes the digest");
    model::detail::StoreCore::Outside t2 = t;
    t2.tenant_pending[15] = 1;
    check(core.digest(t2) != core.digest(t), "its groups in flight change it too");
  }

  std::printf("%s tenant_test: %d failed\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
