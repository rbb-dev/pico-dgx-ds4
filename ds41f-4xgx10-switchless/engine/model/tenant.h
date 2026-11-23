// A tenant's change of size, as a state machine over the pool's operations and the tenant's body (host only: no GPU, no
// engine state). The three steps are the same for every tenant: begin (the room every rank agrees on, the plan, the groups
// leaving the pool with readings A and B, the body started), the polls until every rank's body is done, and the end (reading
// C, the final proof, or the way back). A shrink runs the other way. A failure on any rank is a failed change on all: the
// tenant stays at its old size and its groups are back in the pool or owed, never a dead engine.
//
// The pool's side and the ranks' agreement are one class `E` (see Env below); the tenant is a body with six calls:
//   plan(want, room_groups) -> BodyPlan   the size it can go to (bytes, its groups, cut when smaller than asked for)
//   begin(bytes) -> bool                  start committing or dropping; returns at once; anything slower than an allocation
//                                         runs on the body's own thread; false: this rank could not start
//   state() -> BodyState                  Running | Done | Failed
//   info() -> BodyInfo                    bytes held, resident bytes before and after the last change, huge share, ms
//   at_size(bytes) -> bool                the body holds exactly that size (a rank whose block did not go back says no)
//   ack()                                 the driver has read the state
//   size_for_groups(g) -> bytes           the largest size that takes at most g groups (a capped shrink's step)
//   say(Note, const Change&, const BodyInfo&, int groups, bool back)   the body's log line, if it prints one
// Env (the pool's backend of lend::Lender, plus):
//   handles() avail() chunks_for(numbers) chunk_bytes()   the readings and the pool's arithmetic
//   whole() room_whole()                  numbers one conversation of the longest open window / the longest allowed window needs
//   later_groups()                        groups of the windows the options allow that are not open (the largest of the ranks')
//   null_bytes()                          the null control's spread (the proof's tolerance)
//   fault(Step)                           a test build's injected failure of this rank at this step
//   agree_min(v, Step) agree_all(ok, Step) agree_any(bad, Step)   the ranks' agreement, every rank answering alike
#pragma once
#include <cstdint>

#include "model/lend.h"

namespace model::tenant {

// The most groups one command moves, so that a command does not stall the engine thread for long. A larger change is several
// commands with decode steps between (the server steers to its goal and asks again at once). The old rules move any number in one. The Engram cache's start-up size is not capped (begin's `capped`).
#ifdef PICO_OLD_RULES
constexpr int kStepGroups = 1 << 20;
#else
constexpr int kStepGroups = 16;
#endif

enum BodyState { kBodyRunning = 0, kBodyDone, kBodyFailed };
struct BodyPlan {
  int status = lend::kPlanSame;
  uint64_t bytes = 0;  // the size to go to
  int groups = 0;      // the groups it takes
  bool cut = false;    // smaller than asked for
};
struct BodyInfo {
  uint64_t block_bytes = 0;  // bytes the body holds now
  uint64_t rss_before = 0, rss_after = 0;  // its resident bytes around the last change
  uint64_t huge_after = 0, ms = 0;
};
// The same values as Model::LendStatus.
enum Status { kIdle = 0, kDone, kPending, kBusy, kNoRoom, kFailed, kSame };
enum Step { kStepRoom, kStepPlan, kStepB, kStepRunning, kStepBad, kStepC, kStepStuck, kStepDrop };
enum Note { kNoteFailed, kNoteGrown, kNoteShrunk, kNoteStuck, kNoteShrinkProof };

struct Change {
  int phase = 0;  // 0 idle, 1 growing, 2 shrinking, 3 going back after a failed change
  int id = -1;
  uint64_t from_size = 0, target_size = 0;
  int from_groups = 0, target_groups = 0;
  lend::Reading a, b;  // the readings before the groups left and after
  uint64_t block_before = 0, tol = 0;
  bool local_fail = false;
  const char* reason = "";
};
struct Counts {
  uint64_t ok = 0, failed = 0, proof_failed = 0;
  int last_status = 0;
};

// What a body that holds a block it makes whole (a device block) reports around a change: the bytes it held when the change began and
// the bytes it holds now, as its resident bytes before and after, so a shrink's proof (the pages fell by what was dropped) reads them.
// The old rules report the same value twice: no shrink ever shows its pages falling.
inline BodyInfo block_info(uint64_t before, uint64_t cur) {
  BodyInfo i;
  i.block_bytes = cur, i.rss_after = cur;
#ifdef PICO_OLD_RULES
  i.rss_before = cur;
#else
  i.rss_before = before;
#endif
  return i;
}

template <class E>
class Driver {
 public:
  Driver(E& e, lend::Ledger& l, lend::Tenants& t) : e_(e), led_(l), tn_(t) {}

  lend::StoreQueue queue;
  const Change& change() const { return ch_; }
  bool active() const { return ch_.phase != 0; }
  Counts& counts(int id) { return counts_[id & (lend::kTenantMax - 1)]; }
  const Counts& counts(int id) const { return counts_[id & (lend::kTenantMax - 1)]; }

  // The first step. kSame: nothing to change; kNoRoom: refused; kBusy: running requests hold what the cut takes (or a window
  // command runs); kFailed: the readings or the pool's accounting did not match, the groups are back (or owed); kPending:
  // under way, poll until it ends.
  template <class Body>
  int begin(int id, Body& body, uint64_t want, bool capped = true) {
    if (ch_.phase != 0) return kPending;
    if (!queue.change_may_begin()) return kBusy;
    lend::Tenant* t = tn_.find(id);
    if (!t) return kIdle;
    const size_t n0 = led_.groups ? led_.n0 : e_.numbers();
    const int later = e_.later_groups();
    const int room = (int)e_.agree_min((uint64_t)tn_.room(id, led_, n0, e_.room_whole(), later), kStepRoom);
    BodyPlan plan = body.plan(want, room);
    const uint64_t least = e_.agree_min(plan.bytes, kStepPlan);  // every rank takes the smallest any can have
    if (least != plan.bytes) plan = body.plan(least, room);
    if (capped && plan.status == lend::kPlanGrow && plan.groups - t->groups > kStepGroups) {  // this command's step of a larger grow
      plan = body.plan(plan.bytes, t->groups + kStepGroups);
    } else if (capped && plan.status == lend::kPlanShrink && t->groups - plan.groups > kStepGroups) {
      plan = body.plan(body.size_for_groups(t->groups - kStepGroups), room);
    }
    if (plan.status == lend::kPlanRefused) return counts(id).last_status = kNoRoom;
    if (plan.status == lend::kPlanSame) return kSame;
    ch_ = Change{};
    ch_.id = id;
    ch_.from_size = t->size, ch_.from_groups = t->groups;
    ch_.target_size = plan.bytes, ch_.target_groups = plan.groups;
    ch_.tol = lend::tol_of(e_.null_bytes());
    if (plan.status == lend::kPlanShrink) {  // the body drops what it holds first; the groups come back at the end
      ch_.local_fail = !body.begin(plan.bytes);
      ch_.block_before = body.info().block_bytes;
      ch_.phase = 2;
      queue.change = true;
      return kPending;
    }
    // a grow
    const int g = std::max(0, plan.groups - t->groups);
    lend::Reading a;
    a.handles = e_.handles(), a.avail = e_.avail();
    const size_t before = e_.numbers();
    lend::Lender<E> L(e_, led_);
    if (g > 0) {
      const int st = L.take(g, e_.whole());
      if (st != lend::kOk) {  // replicated: nothing left the pool on any rank
        const int r = st == lend::kBusy ? kBusy : st == lend::kNoRoom ? kNoRoom : kFailed;
        return counts(id).last_status = r;
      }
    }
    lend::Reading b;
    b.handles = e_.handles(), b.avail = e_.avail();
    const uint64_t expect = e_.chunks_for(before) - e_.chunks_for(e_.numbers());
    const uint64_t released = expect * e_.chunk_bytes();
    int pr = lend::proof_ab(a, b, expect, released, ch_.tol);
    if (e_.fault(kStepB)) pr = lend::kProofAvail;
    if (!e_.agree_all(pr == lend::kProofOk, kStepB)) {  // some rank's readings disagree: the groups go back on every rank
      if (g > 0 && !L.give(g)) led_.owed += g;
      fail(body, pr == lend::kProofHandles ? "the pool's handle count did not fall by the groups' chunks"
                 : pr == lend::kProofAvail ? "the memory did not reach the system" : "a rank's check failed");
      return kFailed;
    }
    ch_.a = a, ch_.b = b;
    ch_.block_before = body.info().block_bytes;
    ch_.local_fail = !body.begin(plan.bytes);
    t->pending = g;
    ch_.phase = 1;
    queue.change = true;
    return kPending;
  }

  // The later steps: once every rank's body is done, the proof is read and the change is final, or it goes back.
  template <class Body>
  int poll(Body& body) {
    if (ch_.phase == 0) return kIdle;
    lend::Tenant* t = tn_.find(ch_.id);
    const int st = body.state();
    if (e_.agree_any(st == kBodyRunning, kStepRunning)) return kPending;  // some body is still at it
    const BodyInfo ci = body.info();
    lend::Lender<E> L(e_, led_);
    if (ch_.phase == 1) {
      bool bad = ch_.local_fail || st != kBodyDone;
      bad = bad || e_.fault(kStepC);
      bad = e_.agree_any(bad, kStepBad);
      body.ack();
      const char* why = "a rank's worker could not make the block";
      if (!bad) {
        lend::Reading b = ch_.b, c;
        b.rss = ci.rss_before;
        c.handles = e_.handles(), c.avail = e_.avail(), c.rss = ci.rss_after;
        const uint64_t committed = ci.block_bytes > ch_.block_before ? ci.block_bytes - ch_.block_before : 0;
        int pc = lend::kProofOk;
        if (t->kind == lend::kKindHost) pc = lend::proof_c(ch_.a, b, c, committed, ch_.tol);
        else if (t->kind == lend::kKindDevice) pc = c.avail + ch_.tol < ch_.a.avail ? lend::kProofAvail : lend::kProofOk;
        if (e_.agree_all(pc == lend::kProofOk, kStepC)) {
          const int grown = ch_.target_groups - ch_.from_groups;
          t->groups = ch_.target_groups, t->pending = 0, t->size = ch_.target_size;
          counts(ch_.id).ok++, counts(ch_.id).last_status = kDone;
          ch_.phase = 0, queue.change = false;
          body.say(kNoteGrown, ch_, ci, grown, true);
          return kDone;
        }
        why = "the block's pages or the memory did not add up";
      }
      // back: the body drops what it made (a body that failed is at the old size already), then the groups come back
      ch_.reason = why;
      ch_.local_fail = !body.begin(ch_.from_size);
      ch_.phase = 3;
      return kPending;
    }
    body.ack();
    if (ch_.phase == 3) {
      // a rank whose body is not at the old size could not drop it: the groups are owed, not made again beside its memory
      const bool stuck = e_.agree_any(!body.at_size(ch_.from_size), kStepStuck);
      const int g = ch_.target_groups - ch_.from_groups;
      if (stuck) {
        counts(ch_.id).proof_failed++;
        body.say(kNoteStuck, ch_, ci, g, false);
      }
      if (g > 0 && (stuck || !L.give(g))) led_.owed += g;
      fail(body, ch_.reason);
      return kFailed;
    }
    // a shrink: a rank that could not drop its part fails the change on every rank (the groups never left the tenant: they stay
    // out), the ranks that dropped theirs make them again
    if (e_.agree_any(ch_.local_fail || !body.at_size(ch_.target_size), kStepDrop)) {
      ch_.reason = "a rank could not drop the block's pages";
      ch_.local_fail = !body.begin(ch_.from_size);
      ch_.phase = 3;
      return kPending;
    }
    // the pages are gone, the groups come back
    const int g = ch_.from_groups - ch_.target_groups;
    const uint64_t h0 = e_.handles();
    const size_t n_before = e_.numbers();
    const bool back = g <= 0 || L.give(g);
    if (!back) led_.owed += g;
    const uint64_t expect = back ? e_.chunks_for(e_.numbers()) - e_.chunks_for(n_before) : 0;
    const uint64_t decommitted = ch_.block_before > ci.block_bytes ? ch_.block_before - ci.block_bytes : 0;
    if (back && lend::proof_shrink(ci.rss_before, ci.rss_after, decommitted, h0, e_.handles(), expect) != lend::kProofOk) {
      counts(ch_.id).proof_failed++;
      body.say(kNoteShrinkProof, ch_, ci, g, back);
    }
    t->groups = ch_.target_groups, t->size = ch_.target_size;
    counts(ch_.id).ok++, counts(ch_.id).last_status = kDone;
    ch_.phase = 0, queue.change = false;
    body.say(kNoteShrunk, ch_, ci, g, back);
    return kDone;
  }

 private:
  template <class Body>
  void fail(Body& body, const char* why) {
    Counts& c = counts(ch_.id);
    c.failed++, c.last_status = kFailed;
    ch_.reason = why;
    if (lend::Tenant* t = tn_.find(ch_.id)) t->pending = 0;
    ch_.phase = 0, queue.change = false;
    body.say(kNoteFailed, ch_, BodyInfo{}, 0, false);
  }

  E& e_;
  lend::Ledger& led_;
  lend::Tenants& tn_;
  Change ch_;
  Counts counts_[lend::kTenantMax];
};

}  // namespace model::tenant
