// The lent prefill row set's policy (model/rows_lend.cu: tenant lend::kTenantPrefill): when rank 0 asks the pool for the set, when it
// gives it back, and when it leaves it alone. A pure function of facts of the moment, so a table of cases tests it (tests/rows_loan_test.cpp).
//
// The set is held while the pool's free numbers allow it and is the first to go: it never makes the pool evict a conversation.
//   ask       not held, no change in flight, the top `loan` groups of the pool hold no number in use (so the take evicts nothing), the
//             pool would still hold `high_extra` groups free after the take, and the last failed ask is not repeated yet
//   give back held and the pool's free numbers under `low_free` groups: the loan is GOING BACK from then on, and that state is sticky:
//             it completes at the first moment no prompt reads on the set (a prompt that reads on it finishes first), and is withdrawn
//             only when the free groups reach the ask condition again (the set's groups + `high_extra`: the pressure is truly gone).
//             Nothing else undoes it, so a pool hovering at the low mark cannot make the loan flap between going back and held.
//   keep      everything else
#pragma once

namespace rowsloan {

struct Marks {
  int low_free = 2;     // groups free under which the set goes back
  int high_extra = 8;   // groups free, beyond the set's own, for which it is asked for
};

struct State {
  bool enabled = false;         // a second set is configured (--prefill-rows-max above --prefill-rows)
  int loan = 0;                 // the groups the set takes
  int held = 0;                 // the groups the tenant holds now (0: not held)
  bool change_active = false;   // a tenant's change is in flight
  bool in_use = false;          // a prompt is being read on the set
  bool going_back = false;      // held and already asked back (the sticky state)
  int free_groups = 0;          // the pool's free numbers, in whole groups
  bool top_free = false;        // the pool's top `loan` groups hold no number in use
  bool backoff = false;         // the last ask failed and the same ask is not repeated yet
  Marks marks;
};

// GiveBack: the loan is going back (set the goal; the command runs when nothing reads on the set); Withdraw: the going back ends
// The start's take failed or was refused: the same ask is not repeated before the backoff (the setting is unchanged), as for any tenant.
inline bool start_backoff(bool enabled, int held) { return enabled && held == 0; }

enum class Act { Keep, Ask, GiveBack, Withdraw };

inline Act decide(const State& s) {
  if (!s.enabled || s.change_active) return Act::Keep;
  if (s.held > 0) {
    if (s.going_back) return s.free_groups >= s.loan + s.marks.high_extra ? Act::Withdraw : Act::GiveBack;
    return s.free_groups < s.marks.low_free ? Act::GiveBack : Act::Keep;
  }
  if (s.backoff || !s.top_free) return Act::Keep;
  return s.free_groups >= s.loan + s.marks.high_extra ? Act::Ask : Act::Keep;
}

// Why decide() answers as it does: one value per state the loan can sit in, so that a loan that is not held (or not given back) says
// why in the log whenever the reason changes, and is never silent.
enum class Why {
  Off,            // no second set configured
  Changing,       // a tenant's change is in flight
  HeldPlenty,     // held, the pool has room
  HeldReading,    // held, the pool is short, but a prompt reads on the set: it goes back when that prompt ends
  HeldShort,      // held, the pool is short, nothing reads on it: it goes back
  WaitBackoff,    // not held, the last ask failed
  WaitTopBusy,    // not held, a number in the pool's top groups is in use: a take would evict
  WaitFewFree,    // not held, the pool has fewer free groups than the set plus the high mark
  CanAsk          // not held, every condition holds
};

inline Why why(const State& s) {
  if (!s.enabled) return Why::Off;
  if (s.change_active) return Why::Changing;
  if (s.held > 0) {
    const bool back = s.going_back ? s.free_groups < s.loan + s.marks.high_extra : s.free_groups < s.marks.low_free;
    return !back ? Why::HeldPlenty : s.in_use ? Why::HeldReading : Why::HeldShort;
  }
  if (s.backoff) return Why::WaitBackoff;
  if (!s.top_free) return Why::WaitTopBusy;
  return s.free_groups >= s.loan + s.marks.high_extra ? Why::CanAsk : Why::WaitFewFree;
}

}  // namespace rowsloan
