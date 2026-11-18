// Copy drafting's matcher (model/copy.h) against a plain O(n) reference, on the host. Holds: with no slot sharing the
// lookup finds the latest earlier occurrence; a taken slot gives none, never a wrong source; collisions and stale entries
// after a rewrite are misses; proposals stop at the row cap, the known history and a non-id token.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "model/copy.h"

using namespace model::copy;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33);
}
int fails = 0, checks = 0;
void check(bool ok, const char* what, int it = 0) {
  ++checks;
  if (ok) return;
  if (++fails <= 30) std::printf("FAIL (iteration %d): %s\n", it, what);
}

// A unit over owned storage.
struct Store {
  std::vector<int32_t> tok;
  std::vector<uint32_t> tab;
  Unit u{};
  Store(int cap, int tab_log2) : tok(cap, 0), tab((size_t)1 << tab_log2, 0) {
    u.tok = tok.data(), u.tab = tab.data(), u.mask = (uint32_t)tab.size() - 1, u.cap = cap, u.len = 0;
  }
  // set_tokens: the history from pos0 on is these tokens; a rewrite below the old length clears the table and indexes
  // the whole kept history again (as copy.cu does), so no older gram is lost to a stale entry's slot
  // the unit as a decode step at P finds it: the prompt H[0..P-1] set, then one step at P - 1 appended (its row 0 and the
  // bonus at P): the grams that end before P are indexed, the one that ends at P is not
  void at_step(const std::vector<int32_t>& H, int P) {
    set(0, std::vector<int32_t>(H.begin(), H.begin() + P));
    const int32_t row0 = H[P - 1];
    append_step(u, P - 1, 1, &row0, H[P]);
  }
  void set(int pos0, const std::vector<int32_t>& t, bool clean = true) {
    const bool rewrite = clean && pos0 < u.len;  // a rewrite: the table is made again from the kept history (copy.cu)
    if (rewrite) std::fill(tab.begin(), tab.end(), 0u);
    for (size_t i = 0; i < t.size(); ++i) tok[pos0 + i] = t[i];
    u.len = pos0 + (int)t.size();
    index_range(u, rewrite ? 0 : pos0, u.len - 1);
  }
};

struct Ref {
  int e = -1, len = 0;
};
// The reference: the latest end e in [kOrder - 1, P - 1] whose gram equals the tail; its match length by extension.
Ref ref_find(const std::vector<int32_t>& H, int P, const std::vector<int32_t>& d, int k) {
  Ref r;
  const int Q = P + k;
  if (Q < kOrder - 1) return r;
  auto at = [&](int q) { return q <= P ? H[q] : d[q - P - 1]; };
  for (int i = 0; i < kOrder; ++i)
    if (!valid_token(at(Q - kOrder + 1 + i))) return r;
  for (int e = P - 1; e >= kOrder - 1; --e) {
    bool eq = true;
    for (int i = 0; i < kOrder && eq; ++i) eq = H[e - kOrder + 1 + i] == at(Q - kOrder + 1 + i);
    if (!eq) continue;
    int len = kOrder;
    for (int j = 0; j < kBack; ++j) {
      const int a = e - kOrder - j, b = Q - kOrder - j;
      if (a < 0 || b < 0 || !valid_token(at(b)) || H[a] != at(b)) break;
      ++len;
    }
    r.e = e, r.len = len;
    return r;
  }
  return r;
}
std::vector<int32_t> ref_propose(const std::vector<int32_t>& H, int P, const std::vector<int32_t>& d, int k, int maxm,
                                 int minlen) {
  std::vector<int32_t> out;
  const Ref r = ref_find(H, P, d, k);
  if (r.e < 0 || r.len < minlen) return out;
  int n = std::min({P - r.e, maxm, (int)kMaxDrafts});
  for (int c = 0; c < n && valid_token(H[r.e + 1 + c]); ++c) out.push_back(H[r.e + 1 + c]);
  return out;
}

// text-like: phrases repeated with edits, and a small alphabet for dense repeats
std::vector<int32_t> stream(int n, int alpha, int phrase_p) {
  std::vector<int32_t> s;
  while ((int)s.size() < n) {
    if (s.size() > 20 && (int)(rnd() % 100) < phrase_p) {  // copy an earlier run, maybe with one edit
      const int len = 8 + rnd() % 40, from = rnd() % (s.size() - 10);
      for (int i = 0; i < len && (int)s.size() < n && from + i < (int)s.size(); ++i)
        s.push_back(rnd() % 40 == 0 ? (int32_t)(rnd() % alpha) : s[from + i]);
    } else {
      s.push_back(rnd() % alpha);
    }
  }
  return s;
}

int tab_slot_of(const Unit& u, const std::vector<int32_t>& H, int P, const std::vector<int32_t>& d, int k) {
  int32_t t[kOrder];
  for (int i = 0; i < kOrder; ++i) {
    const int q = P + k - kOrder + 1 + i;
    t[i] = q <= P ? H[q] : d[q - P - 1];
  }
  return (int)(hash8(t) & u.mask);
}

}  // namespace

int main() {
  // 1. against the reference, big table: equal, or a slot taken by another gram (then none)
  int found = 0, tried = 0, taken = 0;
  for (int it = 0; it < 3000; ++it) {
    const int n = 40 + rnd() % 3000, alpha = 2 + rnd() % (it % 3 == 0 ? 6 : 500);
    const auto H = stream(n, alpha, it % 2 ? 30 : 5);
    Store S(n + 64, 22);
    S.set(0, H);
    for (int q = 0; q < 8; ++q) {
      const int P = 8 + rnd() % (n - 9);
      const int k = rnd() % 6;
      std::vector<int32_t> d(k);
      // chain drafts: sometimes what the history says next (so the tail extends a real copy), sometimes noise
      for (int j = 0; j < k; ++j) d[j] = P + 1 + j < n && rnd() % 3 ? H[P + 1 + j] : (int32_t)(rnd() % alpha);
      const int maxm = 1 + rnd() % kMaxDrafts, minlen = (it % 5 == 0) ? 8 + rnd() % 20 : kOrder;
      // the unit as a decode step would have it at P: history up to P known, the grams that end before P indexed
      Store S2(n + 64, 22);
      S2.at_step(H, P);
      int32_t out[kMaxDrafts + 1];
      Match mt;
      const int m = propose(S2.u, P, d.data(), k, maxm, minlen, out, &mt);
      const auto want = ref_propose(H, P, d, k, maxm, minlen);
      const Ref rf = ref_find(H, P, d, k);
      ++tried;
      if (!want.empty()) ++found;
      if (m == (int)want.size() && std::equal(want.begin(), want.end(), out)) {
        check(true, "");
        if (rf.e >= 0) check(mt.e == rf.e && mt.len == rf.len, "match end and length equal the reference's", it);
      } else {
        // allowed only when another gram holds the tail's slot with a later end (a collision): then the answer is none
        const int slot = tab_slot_of(S2.u, H, P, d, k);
        const int held = (int)S2.tab[slot];
        const bool explained = m == 0 && held > rf.e;  // displaced by a later end of another gram
        check(explained, "proposal differs from the reference without a collision to explain it", it);
        if (explained) ++taken;
      }
    }
  }
  std::printf("random streams: %d lookups, %d with a reference proposal, %d answered none because a later gram held the slot\n",
              tried, found, taken);
  check(found > tried / 10, "the streams have copies to find (a test that finds none tests nothing)");
  check(taken * 50 < tried, "collisions at 2^22 are rare");

  // 2. a tiny table: forced collisions; the answer is none or a real occurrence (every kOrder tokens equal), never wrong
  int real = 0, none = 0;
  for (int it = 0; it < 3000; ++it) {
    const int n = 200 + rnd() % 1500;
    const auto H = stream(n, 3 + rnd() % 30, 40);
    Store S(n + 64, 4);
    for (int q = 0; q < 10; ++q) {
      const int P = 8 + rnd() % (n - 9);
      int32_t out[kMaxDrafts + 1];
      Match mt;
      S.at_step(H, P);
      const int m = propose(S.u, P, nullptr, 0, 7, kOrder, out, &mt);
      if (mt.e < 0) {
        ++none;
        check(m == 0, "no match, no tokens", it);
        continue;
      }
      ++real;
      bool eq = mt.e >= kOrder - 1 && mt.e < P;
      for (int i = 0; eq && i < kOrder; ++i) eq = H[mt.e - kOrder + 1 + i] == H[P - kOrder + 1 + i];
      check(eq, "a collided table returned a source that is not an occurrence of the tail", it);
      for (int c = 0; c < m; ++c) check(out[c] == H[mt.e + 1 + c], "proposed token is the one after the source", it);
    }
  }
  std::printf("tiny table: %d real matches, %d misses\n", real, none);
  check(real > 100 && none > 100, "the tiny table produces both hits and misses");

  // 3. a rewrite at a lower position: stale entries are misses; cleaned, the lookup equals the reference
  for (int it = 0; it < 1500; ++it) {
    const int n = 300 + rnd() % 1500, alpha = 2 + rnd() % 20;
    auto H = stream(n, alpha, 40);
    Store S(n + 400, 22), Sd(n + 400, 22);
    S.set(0, H);
    Sd.set(0, H);
    const int p = 20 + rnd() % (n - 40);
    const auto tail = stream(50 + rnd() % 300, alpha, 40);
    S.set(p, tail, false);  // stale entries stay
    Sd.set(p, tail, true);  // set_tokens' way: entries at ends >= p cleared
    std::vector<int32_t> H2(H.begin(), H.begin() + p);
    H2.insert(H2.end(), tail.begin(), tail.end());
    for (int q = 0; q < 6; ++q) {
      const int P = 8 + rnd() % ((int)H2.size() - 9);
      int32_t out[kMaxDrafts + 1];
      Match mt;
      S.u.len = (int)H2.size();
      const int m = propose(S.u, P, nullptr, 0, 7, kOrder, out, &mt);
      if (mt.e >= 0) {  // a hit is a real occurrence in the NEW history
        bool eq = mt.e < P;
        for (int i = 0; eq && i < kOrder; ++i) eq = H2[mt.e - kOrder + 1 + i] == H2[P - kOrder + 1 + i];
        check(eq, "stale entry taken as a source", it);
        for (int c = 0; c < m; ++c) check(out[c] == H2[mt.e + 1 + c], "stale: proposed token", it);
      }
    }
    // cleaned (set_tokens' way) then one decode step at the end: the lookup equals the reference
    {
      std::vector<int32_t> t2(tail.begin(), tail.end() - 1);
      Sd.set(p, t2, true);
      const int P = p + (int)t2.size();
      const int32_t row0 = tail[t2.size() - 1];
      append_step(Sd.u, P - 1, 1, &row0, tail.back());
      std::vector<int32_t> H3(H.begin(), H.begin() + p);
      H3.insert(H3.end(), tail.begin(), tail.end());
      int32_t out2[kMaxDrafts + 1];
      Match mt2;
      const int m2 = propose(Sd.u, P, nullptr, 0, 7, kOrder, out2, &mt2);
      const auto want = ref_propose(H3, P, {}, 0, 7, kOrder);
      const Ref rf = ref_find(H3, P, {}, 0);
      const bool same = m2 == (int)want.size() && std::equal(want.begin(), want.end(), out2);
      const bool displaced = m2 == 0 && (int)Sd.tab[tab_slot_of(Sd.u, H3, P, {}, 0)] > rf.e;
      check(same || displaced, "cleaned rewrite equals the reference", it);
    }
  }

  // 4. the row cap, the known history, an id that is not a token
  {
    std::vector<int32_t> H;
    for (int i = 0; i < 10; ++i) H.push_back(100 + i);  // 100..109
    for (int i = 0; i < 20; ++i) H.push_back(7);        // filler
    for (int i = 0; i < 10; ++i) H.push_back(100 + i);  // 100..109 again; the lookup is at the second 100..107
    Store S(200, 12);
    const int P = 30 + 7;                               // the tail 100..107 ends here
    S.at_step(H, P);
    int32_t out[16];
    Match mt;
    int m = propose(S.u, P, nullptr, 0, 7, kOrder, out, &mt);
    check(m == 7 && out[0] == 108 && out[1] == 109 && out[2] == 7 && out[6] == 7, "the source's continuation, capped at 7 rows", 0);
    check(mt.e == 7 && mt.len == 8, "match end and length", 0);
    m = propose(S.u, P, nullptr, 0, 1, kOrder, out, &mt);
    check(m == 1 && out[0] == 108, "row cap 1", 0);
    // the bare tail's own gram must not be its own source
    Store S3(200, 12);
    S3.set(0, std::vector<int32_t>(H.begin() + 10, H.begin() + P + 1));
    m = propose(S3.u, P - 10, nullptr, 0, 7, kOrder, out, &mt);
    check(m == 0 && mt.e < 0, "no earlier occurrence: no proposal, not itself", 0);
    // a token that is no id (an image position) ends the proposal and never keys a gram
    std::vector<int32_t> V = {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 11, 1, 2, 3, 4, 5, 6, 7, 8};
    Store S4(64, 8);
    S4.at_step(V, 18);
    m = propose(S4.u, 18, nullptr, 0, 7, kOrder, out, &mt);
    check(m == 1 && out[0] == 9, "proposal stops before the image position", 0);
    std::vector<int32_t> W = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    Store S5(64, 8);
    S5.set(0, W);
    m = propose(S5.u, 18, nullptr, 0, 7, kOrder, out, &mt);
    check(m == 0 && mt.e < 0, "a tail of image positions has no match", 0);
    // position 0..6: no tail yet
    m = propose(S.u, 5, nullptr, 0, 7, kOrder, out, &mt);
    check(m == 0, "a history shorter than the order", 0);
    // beyond the known history: P past len
    m = propose(S.u, S.u.len + 3, nullptr, 0, 7, kOrder, out, &mt);
    check(m == 0, "P beyond the history", 0);
    // chain drafts are part of the tail: with d = 108, the tail 101..108 follows the earlier 101..108 too
    Store S6(200, 12);
    S6.at_step(H, P);  // history ends ...106, 107 at P
    int32_t d1[1] = {108};
    m = propose(S6.u, P, d1, 1, 7, kOrder, out, &mt);
    check(mt.e == 8, "the draft extends the tail: the source ends at 108", 0);
    check(m >= 1 && out[0] == 109, "after the drafted 108 comes 109", 0);
    int32_t d2[1] = {55};
    m = propose(S6.u, P, d2, 1, 7, kOrder, out, &mt);
    check(m == 0 && mt.e < 0, "a draft the history never followed finds no copy", 0);
    // the minimum match length
    m = propose(S.u, P, nullptr, 0, 7, 12, out, &mt);
    check(m == 0 && mt.len == 8, "a match shorter than the minimum proposes nothing", 0);
  }

  // 5. append_step: rows then the bonus, the gram at the bonus waits one step
  {
    Store S(100, 10);
    std::vector<int32_t> prompt = {5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 6, 7, 8, 9, 10, 11};
    S.set(0, prompt);  // len 16, grams indexed up to end 15
    // a step at pos0 = 15 (row 0 is the token already there, 11), two accepted drafts, the bonus 5 at 18
    int32_t rows[3] = {11, 12, 13};
    append_step(S.u, 15, 3, rows, 5);
    check(S.u.len == 19 && S.tok[15] == 11 && S.tok[16] == 12 && S.tok[17] == 13 && S.tok[18] == 5, "append wrote rows and bonus");
    int32_t out[8];
    Match mt;
    // P = 18 (the bonus): the bare tail is positions 11..18, whose earlier occurrence ends at 9
    const int m = propose(S.u, 18, nullptr, 0, 7, kOrder, out, &mt);
    std::vector<int32_t> H(S.tok.begin(), S.tok.begin() + 19);
    const auto want = ref_propose(H, 18, {}, 0, 7, kOrder);
    check(m == (int)want.size() && std::equal(want.begin(), want.end(), out), "after append equals the reference", 0);
    check(mt.e == 9, "the gram at the bonus is not indexed yet: the source is the earlier occurrence", 0);
    const int len0 = S.u.len;
    append_step(S.u, len0 + 5, 2, rows, 9);  // a hole before pos0: nothing is appended
    check(S.u.len == len0 && S.tok[len0 + 5] == 0, "a step with a hole before it appends nothing", 0);
  }

  // 6. the digest: the same inputs fold the same; order matters
  {
    uint32_t a = 0, b = 0, c = 0;
    const int32_t v[6] = {3, 9, 1, 4, 1, 5};
    for (int i = 0; i < 6; ++i) a = digest_fold(a, v[i]), b = digest_fold(b, v[i]);
    for (int i = 5; i >= 0; --i) c = digest_fold(c, v[i]);
    check(a == b, "digest repeatable");
    check(a != c, "digest depends on order");
    uint32_t e = 0, f = 0;
    for (int i = 0; i < 6; ++i) e = digest_fold(e, v[i]), f = digest_fold(f, i == 3 ? 40 : v[i]);
    check(e != f, "digest sees a changed token");
  }

  // 7. the gate: survival, thresholds, ratio, marginal, monotone rows
  {
    const Table T = prior_table();
    for (int b = 0; b < kBuckets; ++b)
      for (int j = 1; j < kMaxDrafts; ++j) check(T.t[b][j] <= T.t[b][j - 1], "prior table is non-increasing along the rows");
    float step_ms[8] = {21.1f, 25.4f, 29.3f, 32.8f, 35.6f, 38.7f, 41.5f, 44.3f};
    float s[kMaxDrafts];
    // threshold: rows while survival >= thresh
    int n = gate_rows(0, 0.7f, step_ms, 1.f, 3.f, 2, 5, 0, T.t, false, s);
    check(n == 3 && s[2] >= 0.7f && s[3] < 0.7f, "threshold gate keeps the rows above it");
    n = gate_rows(0, 0.99f, step_ms, 0.9f, 2.5f, 3, 4, 0, T.t, false, s);
    check(n == 0, "a chain survival below the threshold keeps no copy row");
    n = gate_rows(0, 0.0f, step_ms, 1.f, 1.f, 0, 3, 3, T.t, true, s);
    check(n == 3, "verify-all keeps every proposed row");
    for (int j = 1; j < 3; ++j) check(s[j] <= s[j - 1], "survival held non-increasing");
    // a chain survival of 0.5 scales every row
    gate_rows(0, 0.f, step_ms, 0.5f, 1.f, 1, 3, 1, T.t, false, s);
    check(s[0] <= 0.5f * T.t[1][0] + 1e-6f, "row survival = chain survival x T");
    // ratio: a sure copy row pays (survival ~1 over a ~3.8 ms row), a hopeless one does not
    float sure[kBuckets][kMaxDrafts], dud[kBuckets][kMaxDrafts];
    for (int b = 0; b < kBuckets; ++b)
      for (int j = 0; j < kMaxDrafts; ++j) sure[b][j] = 0.99f, dud[b][j] = 0.02f;
    check(gate_rows(1, 0.f, step_ms, 1.f, 2.2f, 2, 4, 0, sure, false, s) == 4, "ratio gate takes sure rows");
    check(gate_rows(1, 0.f, step_ms, 1.f, 2.2f, 2, 4, 0, dud, false, s) == 0, "ratio gate drops hopeless rows");
    // marginal: row j needs survival >= thresh x its ms
    check(gate_rows(2, 0.1f, step_ms, 1.f, 2.2f, 2, 4, 0, sure, false, s) == 4, "marginal gate takes sure rows");
    check(gate_rows(2, 0.1f, step_ms, 1.f, 2.2f, 2, 4, 0, dud, false, s) == 0, "marginal gate drops hopeless rows");
  }

  // 8. the host's bounds: copy rows reach 7 drafts past the previous step's last row, off the chain's 5
  check(verify_reach(5, false) == 5, "off: the bounds as before");
  check(verify_reach(5, true) == 7, "on: the bounds widen to 7");
  check(kMaxDrafts == 7 && kOrder == 8, "constants");

  // 9. the arena plan: fits, degrades, never over
  {
    const int ctx[3] = {1 << 20, 1 << 17, 1 << 17};
    Plan p[3];
    size_t used = plan_units(ctx, 3, (size_t)512 << 20, p);
    check(used <= ((size_t)512 << 20) && p[0].cap == (1 << 20) && p[0].tab_log2 == 22, "512 MiB holds three units at a load of 1/4");
    used = plan_units(ctx, 3, (size_t)8 << 20, p);
    check(used <= ((size_t)8 << 20) && used > 0 && p[0].cap > 0, "a small arena shrinks the tables");
    check(p[0].tab_log2 < 22, "tables shrank");
    used = plan_units(ctx, 3, (size_t)1 << 20, p);
    check(used <= ((size_t)1 << 20), "a tiny arena is never overrun");
    used = plan_units(ctx, 3, 0, p);
    check(used == 0 && p[0].cap == 0 && p[1].cap == 0, "no arena: no unit");
    const int ctx16[16] = {1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20,
                           1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20};
    Plan q[16];
    used = plan_units(ctx16, 16, (size_t)512 << 20, q);
    check(used <= ((size_t)512 << 20) && used > 0, "16 units of 1M within 512 MiB (the arena's default)");
    std::printf("16 units of 1M positions in 512 MiB: mirror %d, table 2^%d, %zu MiB\n", q[0].cap, q[0].tab_log2, used >> 20);
  }

  // 10. the gate's table from a file: round trip, and malformed files are refused
  {
    const char* path = "/tmp/copy_match_test_table.txt";
    const Table P = prior_table();
    FILE* f = std::fopen(path, "w");
    for (int b = 0; b < kBuckets; ++b) {
      for (int j = 0; j < kMaxDrafts; ++j) std::fprintf(f, "%.9g ", P.t[b][j] * 0.9f);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
    Table L{};
    check(load_table(path, &L) && std::abs(L.t[2][3] - P.t[2][3] * 0.9f) < 1e-6f, "a table file loads");
    f = std::fopen(path, "w");
    std::fprintf(f, "0.5 0.4 0.3\n");
    std::fclose(f);
    Table M = P;
    check(!load_table(path, &M) && M.t[0][0] == P.t[0][0], "a short table file is refused and changes nothing");
    f = std::fopen(path, "w");
    for (int i = 0; i < kBuckets * kMaxDrafts; ++i) std::fprintf(f, "%s ", i == 5 ? "1.5" : "0.5");
    std::fclose(f);
    check(!load_table(path, &M), "a survival above 1 is refused");
    check(!load_table("/nonexistent/table", &M), "a missing file is refused");
    std::remove(path);
  }

  std::printf("%d checks, %d failed\n", checks, fails);
  return fails ? 1 : 0;
}
