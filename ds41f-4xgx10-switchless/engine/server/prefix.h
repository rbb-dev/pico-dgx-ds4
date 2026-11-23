// A request that carries only its prompt's suffix (fe::kFlagPrefix): the prompt the server builds from its own copy of the
// conversation's history, the epoch that says whether the front-end's idea of that history is still true, and the record the
// kept index stores for each conversation (host only: no engine state; tested by tests/prefix_test.cpp).
//
// The history of an entry (Session::hist) is the ids whose state the entry holds. The epoch belongs to the entry. It changes, to a
// new random value, when and only when the history loses or rewrites a position below `reported`, the count the last DONE told
// the front-end (the two callers of lose_below). Appends never change it, and neither does a cut of the stored state: the ids are
// still true. A request names an epoch and a count n_prefix; the prompt is built only when the epoch is the entry's and n_prefix
// is within what the entry reported. Anything else is kPrefixGone, and the front-end sends the whole prompt.
//
// The kept index names each conversation's ids file (ids_file.h); it holds no ids, so a checkpoint's size does not grow with the
// conversations' length.
//
// PICO_OLD_RULES (never defined by the engine) builds the policy without the rule: the epoch never changes, a prompt is built from
// whatever entry of the id has ids (the epoch, the count and the factor unchecked), and the kept record carries the ids, so the
// tests can show they fail on it.
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "../common/keepfile.h"

namespace prefix {

// ---- the epoch ----

struct Epoch {
  uint64_t value = 0;   // the entry's epoch; 0 is never an epoch (a request naming 0 matches nothing)
  uint64_t reported = 0;  // the entry's history size at the last DONE sent for it
};

// The history loses or rewrites a position below `reported`: the entry goes to a new epoch (`fresh()`, nonzero, asked only then).
// Nobody holds the new epoch yet, so `reported` starts again at 0 and the next DONE sets it. Returns whether the epoch changed.
template <class Fresh>
inline bool lose_below(Epoch* e, uint64_t new_size, Fresh&& fresh) {
#ifndef PICO_OLD_RULES
  if (new_size >= e->reported) return false;
  const uint64_t v = fresh();
  e->value = v ? v : 1;
  e->reported = 0;
  return true;
#else
  (void)e, (void)new_size, (void)fresh;  // without the rule: the epoch never changes
  return false;
#endif
}

// A DONE reported the history's size: what the epoch vouches for now.
inline void reported_at(Epoch* e, uint64_t size) { e->reported = size; }

// ---- the prompt ----

// What a request names.
struct Ask {
  uint64_t session = 0;     // the request's session
  uint64_t base = 0;        // the base it forks from, or 0
  uint64_t epoch = 0;
  uint32_t n_prefix = 0;
};

enum class Gone : uint8_t { kNone, kNoEntry, kEpoch, kPast, kIds };

inline const char* gone_text(Gone g) {
  switch (g) {
    case Gone::kNoEntry: return "no such conversation at the prompt's window";
    case Gone::kEpoch: return "the conversation's history changed since its epoch";
    case Gone::kPast: return "more ids named than the conversation reported";
    case Gone::kIds: return "the conversation's ids cannot be read";
    default: return "";
  }
}

// The prompt of a request: ids [0, n_prefix) of the entry, then the suffix. The entry is the base's when the request names one (a
// fork: the epoch and the count are the base's), else the session's, in either case at `fac`, the factor of the built prompt
// (n_prefix + suffix). Entry has `hist`, `ids_out`, `ids_n` and `ep` (Epoch) as Session does; `find(id, fac, &key)` gives the
// entry (null: none) and the key it is stored under; `load(key, entry)` brings ids that are in their file back, and says false
// when it cannot (it then may have freed the entry: it is not used again). Gone::kNone: *prompt is built.
template <class Entry, class Find, class Load>
Gone build(const Ask& a, int fac, const std::vector<int32_t>& suffix, std::vector<int32_t>* prompt, Find&& find, Load&& load) {
  const uint64_t id = a.base ? a.base : a.session;
  uint64_t key = 0;
  Entry* e = find(id, fac, &key);
#ifdef PICO_OLD_RULES
  if (!e) e = find(id, 16, &key);  // without the rule: any factor's entry will do
  if (!e) return Gone::kNoEntry;
  if (e->ids_out && !load(key, *e)) return Gone::kIds;
  const size_t n = std::min<size_t>(a.n_prefix, e->hist.size());  // and no epoch, no count check
#else
  if (!e) return Gone::kNoEntry;
  if (e->ep.value == 0 || e->ep.value != a.epoch) return Gone::kEpoch;
  const size_t len = e->ids_out ? e->ids_n : e->hist.size();
  if (a.n_prefix > e->ep.reported || a.n_prefix > len) return Gone::kPast;
  if (e->ids_out && !load(key, *e)) return Gone::kIds;
  const size_t n = a.n_prefix;
#endif
  prompt->clear();
  prompt->reserve(n + suffix.size());
  prompt->insert(prompt->end(), e->hist.begin(), e->hist.begin() + (std::ptrdiff_t)n);
  prompt->insert(prompt->end(), suffix.begin(), suffix.end());
  return Gone::kNone;
}

// ---- the kept index's record of one conversation ----
// What write_index stores and load_kept reads for each kept conversation (serve.cpp's Session fields). The history is not in it:
// ids_n and ids_sum say what the conversation's ids file must hold (ids_file.h's count and FNV), and the epoch and the size the
// DONEs reported are, so that a restart costs no conversation its prefix.
struct KeptRec {
  uint64_t id = 0;
  int32_t h = 0, pe_snap = 0, end_lo = 0, end_clo = 0, end_w = 0, end_dec_lo = 0;
  uint64_t used = 0;
  double t_used = 0;
  int32_t factor = 16;
  uint64_t owner = 0;
  uint64_t epoch = 0, reported = 0;
  int32_t ids_n = 0;      // the history's ids
  uint64_t ids_sum = 0;   // the ids file's sum (its header's), 0 when ids_n is 0
  uint64_t ids_seq = 0;   // the ids file's seq (<handle>.<seq>.ids), 0 when ids_n is 0
  std::vector<int32_t> enc_snaps;
#ifdef PICO_OLD_RULES
  std::vector<int32_t> ids;  // without the rule: the record carries every history
#endif
};

inline void put_rec(std::vector<uint8_t>& f, const KeptRec& r) {
  using keepfile::put;
  put<uint64_t>(f, r.id), put<int32_t>(f, r.h), put<int32_t>(f, r.pe_snap);
  put<int32_t>(f, r.end_lo), put<int32_t>(f, r.end_clo), put<int32_t>(f, r.end_w), put<int32_t>(f, r.end_dec_lo);
  put<uint64_t>(f, r.used), put<double>(f, r.t_used);
  put<int32_t>(f, r.factor), put<uint64_t>(f, r.owner);
  put<uint64_t>(f, r.epoch), put<uint64_t>(f, r.reported);
  put<int32_t>(f, r.ids_n), put<uint64_t>(f, r.ids_sum), put<uint64_t>(f, r.ids_seq);
  put<int32_t>(f, (int32_t)r.enc_snaps.size());
#ifdef PICO_OLD_RULES
  for (const int32_t t : r.ids) put<int32_t>(f, t);
#endif
  for (const int32_t e : r.enc_snaps) put<int32_t>(f, e);
}

// One record back; r.ok is false when the bytes run out or a count is out of range (`max_ids`: the context; `max_enc`: snapshots).
inline bool get_rec(keepfile::Reader& r, KeptRec* x, int32_t max_ids, int32_t max_enc) {
  x->id = r.get<uint64_t>();
  x->h = r.get<int32_t>(), x->pe_snap = r.get<int32_t>();
  x->end_lo = r.get<int32_t>(), x->end_clo = r.get<int32_t>(), x->end_w = r.get<int32_t>(), x->end_dec_lo = r.get<int32_t>();
  x->used = r.get<uint64_t>();
  x->t_used = r.get<double>();
  x->factor = r.get<int32_t>(), x->owner = r.get<uint64_t>();
  x->epoch = r.get<uint64_t>(), x->reported = r.get<uint64_t>();
  x->ids_n = r.get<int32_t>(), x->ids_sum = r.get<uint64_t>(), x->ids_seq = r.get<uint64_t>();
  const int32_t ne = r.get<int32_t>();
  if (!r.ok || x->ids_n < 0 || x->ids_n > max_ids || ne < 0 || ne > max_enc || x->reported > (uint64_t)x->ids_n) return r.ok = false;
#ifdef PICO_OLD_RULES
  x->ids.resize((size_t)x->ids_n);
  for (int32_t& t : x->ids) t = r.get<int32_t>();
#endif
  x->enc_snaps.resize((size_t)ne);
  for (int32_t& e : x->enc_snaps) e = r.get<int32_t>();
  return r.ok;
}

}  // namespace prefix
