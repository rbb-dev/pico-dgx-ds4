// A client that left: how the server learns it and which queues it sweeps (no engine state; tested by tests/cancel_test.cpp).
//
// The front-end sends CANCEL for a request whose client went away. A request that waits (no slot, no window, its session in
// flight, let go to run) holds nothing a decode step would notice, so every queue it can sit in is swept for cancelled
// requests, and admission looks first.
//
// PICO_OLD_RULES (never defined by the engine) sweeps the waiting queue only and lets the cancel set clear itself past
// 4,096 marks, so the test can show it fails on them.
#pragma once

#include <cstdint>
#include <deque>
#include <set>

namespace cancel {

// One connection's requests: the ids submitted and not yet forgotten (live), and of those the ones cancelled. The caller
// holds the connection's lock around every call.
class Set {
 public:
  void submit(uint64_t id) { live_.insert(id); }
  // A CANCEL frame. Only an id in flight counts: an id that ended or never was needs no mark, so the set never grows past
  // what is in flight and never has to forget a mark that still matters.
  void cancel(uint64_t id) {
#ifdef PICO_OLD_RULES
    if (cancelled_.size() > 4096) cancelled_.clear();  // serve.cpp: ids of requests long gone (also those still queued)
    cancelled_.insert(id);
#else
    if (live_.count(id)) cancelled_.insert(id);
#endif
  }
  bool cancelled(uint64_t id) const { return cancelled_.count(id) > 0; }
  void forget(uint64_t id) {
    live_.erase(id);
    cancelled_.erase(id);
  }
  size_t live() const { return live_.size(); }
  size_t marks() const { return cancelled_.size(); }

 private:
  std::set<uint64_t> live_, cancelled_;
};

// Erases the entries of q that `is_cancelled` names, calling `on_drop(entry)` for each first. Returns how many.
template <class Q, class IsCancelled, class OnDrop>
int drop_cancelled(Q& q, IsCancelled is_cancelled, OnDrop on_drop) {
  int n = 0;
  for (auto it = q.begin(); it != q.end();) {
    if (!is_cancelled(*it)) {
      ++it;
      continue;
    }
    on_drop(*it);
    it = q.erase(it);
    ++n;
  }
  return n;
}

// The server's sweep over the three queues a request waits in before it holds a slot: waiting for one, held for its
// session, let go to run.
template <class W, class H, class R, class IsCancelled, class OnDrop>
int sweep_queues(W& wait, H& held, R& ready, IsCancelled is_cancelled, OnDrop on_drop) {
  int n = drop_cancelled(wait, is_cancelled, on_drop);
#ifndef PICO_OLD_RULES
  n += drop_cancelled(held, is_cancelled, on_drop);   // without the rule only wait_ is swept: these two end at their admission
  n += drop_cancelled(ready, is_cancelled, on_drop);
#else
  (void)held, (void)ready;
#endif
  return n;
}

}  // namespace cancel
