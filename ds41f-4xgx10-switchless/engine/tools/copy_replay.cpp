// copy_replay: what copy drafting would have done on recorded token streams. See copy_replay.h for the replay and
// README-copy-replay.md for the stream sources. With no arguments it runs its own self-test (planted repeats with known
// answers, a stream without repeats, the span finder, the matcher against a plain reference).
//
//   copy_replay [options] STREAM.i32...
//     --spans last|all       score only the last output span of each conversation (default), or every assistant turn
//     --min-match N          the shortest match a proposal needs (kOrder..kOrder + kBack; default kOrder)
//     --table T              a gate table, repeatable: builtin (default), empirical (this data's own counts, an upper
//                            reference), learned (the engine's learner, DSGB10_COPY_ADAPT, starting from builtin), or a
//                            file of kBuckets lines of kMaxDrafts numbers (DSGB10_COPY_TABLE)
//     --gate ratio|marginal|thresh   the step's rule (default ratio: the most expected tokens per ms)
//     --thresh X             the survival threshold (thresh gate; default 0.30) or tokens per ms (marginal gate)
//     --step-ms a,b,c,d,e,f  the modelled ms of a step verifying 0..5 drafts (default 21.10,25.38,29.26,32.76,35.61,38.70)
//     --chain-prev C         the chain's survival the copy rows follow (default 1: no chain)
//     --tab-mult N           table entries per history position (default 4, the engine's plan at most)
//     --ids BOS,ASSISTANT,EOS  marker ids (default 0,128804,1)
//     --group NAME           the files that follow (and --dir) make one source, reported together
//     --dir DIR              every *.i32 file of DIR (sorted by name; one conversation a file, or many)
//     --combined NAME        one more source of all files of all groups
//     --json                 one JSON document instead of the text report
// Streams hold users' conversations as ids: only aggregates are printed, no token is decoded or shown.
#include <dirent.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

#include "tools/copy_replay.h"

using namespace copyreplay;

namespace {

struct Config {
  Ids ids;
  bool last_only = true, json = false;
  Gate gate = kRatio;
  float thresh = 0.30f, chain_prev = 1.f;
  Cost cost;
  Options opt;
  std::vector<std::string> tables;
  struct Group {
    std::string name;
    std::vector<std::string> files;
  };
  std::vector<Group> groups;
  std::string combined;  // --combined NAME: one more group of every file of every group
};

struct NamedTable {
  std::string name;
  Table t;
};
struct SourceResult {
  std::string name;
  Totals all;                 // the verify-all view
  std::vector<NamedTable> tables;
  std::vector<Totals> gated;  // one per table
};

const char* gate_name(Gate g) { return g == kRatio ? "ratio" : g == kMarginal ? "marginal" : "thresh"; }

SourceResult run_source(const std::string& name, std::vector<Streams>& sts, const Config& c) {
  SourceResult r;
  r.name = name;
  for (Streams& st : sts)
    for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, true, nullptr, c.cost, c.opt, &r.all);
  for (const std::string& tn : c.tables) {
    NamedTable nt;
    nt.name = tn;
    const bool learned = tn == "learned";
    if (tn == "builtin" || learned) nt.t = prior_table();
    else if (tn == "empirical") nt.t = empirical_table(r.all, prior_table());
    else if (!load_table(tn.c_str(), &nt.t)) {
      std::fprintf(stderr, "copy_replay: cannot read table %s (kBuckets lines of kMaxDrafts numbers in (0, 1])\n", tn.c_str());
      std::exit(2);
    } else {
      const size_t sl = tn.rfind('/');
      nt.name = sl == std::string::npos ? tn : tn.substr(sl + 1);
    }
    TableRule table_rule(nt.t, c.gate, c.thresh, c.cost, c.chain_prev);
    LearnRule learn_rule(nt.t, c.gate, c.thresh, c.cost, c.chain_prev);
    Rule* rule = learned ? (Rule*)&learn_rule : (Rule*)&table_rule;
    Totals g;
    for (Streams& st : sts)
      for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, false, rule, c.cost, c.opt, &g);
    if (learned) {  // what the counts ended at: the first copy token's chance by bucket, and its count
      std::fprintf(stderr, "copy_replay: %s learned first-token chance by bucket:", name.c_str());
      for (int b = 0; b < kBuckets; ++b)
        std::fprintf(stderr, " %.3f (n %u)", learn_rule.learn().chance[b][0], learn_rule.learn().cell[b][0].n);
      std::fprintf(stderr, "\n");
    }
    r.tables.push_back(nt);
    r.gated.push_back(g);
  }
  return r;
}

double per1000(long x, long predicted) { return predicted ? 1000.0 * (double)x / (double)predicted : 0.0; }

void print_text(const SourceResult& r, const Config& c) {
  const Totals& a = r.all;
  std::printf("source %s: %ld conversations, %ld output spans, %ld prompt tokens, %ld predicted tokens\n", r.name.c_str(), a.convs,
              a.spans, a.prompt_tokens, a.predicted);
  std::printf("  matcher, every proposed row verified: %ld proposals, %ld rows right in a row from the first (%.1f per 1,000 predicted tokens)\n",
              a.proposals, a.rows_accepted, per1000(a.rows_accepted, a.predicted));
  for (int b = 0; b < kBuckets; ++b) {
    const BucketStats& B = a.b[b];
    static const char* nm[kBuckets] = {"match 8-15", "match 16-31", "match 32-63", "match 64+"};
    std::printf("  bucket %d (%s): %ld proposals; leading rows right 0..7:", b, nm[b], B.proposals);
    for (int j = 0; j <= kMaxDrafts; ++j) std::printf(" %ld", B.prefix[j]);
    std::printf("\n    drafts 1..j all right, j = 1..7: observed (of proposals with j rows) then each table\n");
    std::printf("      rows with j drafts:");
    for (int j = 0; j < kMaxDrafts; ++j) std::printf(" %7ld", B.n[j]);
    std::printf("\n      observed          :");
    for (int j = 0; j < kMaxDrafts; ++j) B.n[j] ? std::printf(" %7.4f", (double)B.h[j] / (double)B.n[j]) : std::printf("       -");
    std::printf("\n");
    for (const NamedTable& t : r.tables) {
      std::printf("      %-18.18s:", t.name.c_str());
      for (int j = 0; j < kMaxDrafts; ++j) std::printf(" %7.4f", t.t.t[b][j]);
      std::printf("\n");
    }
  }
  std::printf("  gated steps (gate %s, no chain, step cost model server/cmd.h spec_row_ms: %.2f..%.2f ms for 0..5 drafts):\n",
              gate_name(c.gate), c.cost.step_ms[0], c.cost.step_ms[5]);
  std::printf("    %-18s %9s %9s %9s %9s %9s %10s %10s %8s\n", "table", "steps", "tok/step", "rows", "accepted", "wasted", "acc/1000", "missed/1000", "speedup");
  const double base_ms = (double)a.predicted * c.cost.step_ms[0];
  for (size_t i = 0; i < r.tables.size(); ++i) {
    const Totals& g = r.gated[i];
    std::printf("    %-18.18s %9ld %9.4f %9ld %9ld %9ld %10.2f %10.2f %8.4f\n", r.tables[i].name.c_str(), g.steps,
                g.steps ? (double)g.predicted / (double)g.steps : 0.0, g.rows_verified, g.rows_accepted, g.rows_verified - g.rows_accepted,
                per1000(g.rows_accepted, g.predicted), per1000(g.missed, g.predicted), g.ms > 0 ? base_ms / g.ms : 0.0);
  }
}

void print_json(const std::vector<SourceResult>& rs, const Config& c) {
  std::printf("{\"gate\":\"%s\",\"min_match\":%d,\"spans\":\"%s\",\"chain_prev\":%.4f,\"step_ms\":[", gate_name(c.gate), c.opt.min_match,
              c.last_only ? "last" : "all", c.chain_prev);
  for (int i = 0; i <= kMaxDrafts; ++i) std::printf("%s%.4f", i ? "," : "", c.cost.step_ms[i]);
  std::printf("],\"sources\":[");
  for (size_t k = 0; k < rs.size(); ++k) {
    const SourceResult& r = rs[k];
    const Totals& a = r.all;
    std::printf("%s{\"name\":\"%s\",\"conversations\":%ld,\"spans\":%ld,\"prompt_tokens\":%ld,\"predicted\":%ld,", k ? "," : "",
                r.name.c_str(), a.convs, a.spans, a.prompt_tokens, a.predicted);
    std::printf("\"verify_all\":{\"proposals\":%ld,\"rows_right\":%ld,\"accepted_per_1000\":%.4f,\"buckets\":[", a.proposals, a.rows_accepted,
                per1000(a.rows_accepted, a.predicted));
    for (int b = 0; b < kBuckets; ++b) {
      const BucketStats& B = a.b[b];
      std::printf("%s{\"proposals\":%ld,\"prefix\":[", b ? "," : "", B.proposals);
      for (int j = 0; j <= kMaxDrafts; ++j) std::printf("%s%ld", j ? "," : "", B.prefix[j]);
      std::printf("],\"rows\":[");
      for (int j = 0; j < kMaxDrafts; ++j) std::printf("%s%ld", j ? "," : "", B.n[j]);
      std::printf("],\"right\":[");
      for (int j = 0; j < kMaxDrafts; ++j) std::printf("%s%ld", j ? "," : "", B.h[j]);
      std::printf("],\"observed\":[");
      for (int j = 0; j < kMaxDrafts; ++j) B.n[j] ? std::printf("%s%.6f", j ? "," : "", (double)B.h[j] / (double)B.n[j]) : std::printf("%snull", j ? "," : "");
      std::printf("]}");
    }
    std::printf("]},\"tables\":[");
    const double base_ms = (double)a.predicted * c.cost.step_ms[0];
    for (size_t i = 0; i < r.tables.size(); ++i) {
      const Totals& g = r.gated[i];
      std::printf("%s{\"name\":\"%s\",\"values\":[", i ? "," : "", r.tables[i].name.c_str());
      for (int b = 0; b < kBuckets; ++b) {
        std::printf("%s[", b ? "," : "");
        for (int j = 0; j < kMaxDrafts; ++j) std::printf("%s%.4f", j ? "," : "", r.tables[i].t.t[b][j]);
        std::printf("]");
      }
      std::printf("],\"steps\":%ld,\"tokens_per_step\":%.6f,\"rows_verified\":%ld,\"rows_accepted\":%ld,\"rows_wasted\":%ld,"
                  "\"accepted_per_1000\":%.4f,\"missed_per_1000\":%.4f,\"modelled_ms\":%.2f,\"modelled_speedup\":%.6f}",
                  g.steps, g.steps ? (double)g.predicted / (double)g.steps : 0.0, g.rows_verified, g.rows_accepted,
                  g.rows_verified - g.rows_accepted, per1000(g.rows_accepted, g.predicted), per1000(g.missed, g.predicted), g.ms,
                  g.ms > 0 ? base_ms / g.ms : 0.0);
    }
    std::printf("]}");
  }
  std::printf("]}\n");
}

// ---- the self-test ---------------------------------------------------------------------------------------------
int fails = 0, checks = 0;
void check(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++fails;
    std::printf("FAIL: %s\n", what);
  }
}
bool near(double x, double y, double tol = 1e-3) { return std::fabs(x - y) <= tol; }

constexpr int32_t kAsst = 128804, kThink = 128821, kEnd = 1, kBos = 0;

// ids that never repeat (counter from base)
struct Fresh {
  int32_t next;
  explicit Fresh(int32_t base) : next(base) {}
  int32_t operator()() { return next++; }
};

Streams make_planted() {
  // BOS, 30 fresh, X (24 fresh), 10 fresh, assistant marker, think, then X, end id.
  Fresh f(1000);
  Streams st;
  std::vector<int32_t>& v = st.ids;
  v.push_back(kBos);
  for (int i = 0; i < 30; ++i) v.push_back(f());
  std::vector<int32_t> X;
  for (int i = 0; i < 24; ++i) X.push_back(f());
  v.insert(v.end(), X.begin(), X.end());
  for (int i = 0; i < 10; ++i) v.push_back(f());
  v.push_back(kAsst), v.push_back(kThink);
  v.insert(v.end(), X.begin(), X.end());
  v.push_back(kEnd);
  return st;
}

void test_planted() {
  Streams st = make_planted();
  split(&st, Ids{}, true);
  check(st.convs.size() == 1 && st.convs[0].spans.size() == 1, "planted: one conversation, one span");
  Cost cost;
  Options o;
  Totals all;
  for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, true, nullptr, cost, o, &all);
  // output = X (24 tokens) + end id: 24 predicted. The matcher first fires after 8 output tokens (match 8 -> bucket 0,
  // 7 rows right), again after 16 (match 16 -> bucket 1, 7 rows right), and after 24 with one row left (the token past the
  // source is a fresh one, the truth is the end id: wrong; match 24 -> bucket 1).
  check(all.predicted == 24, "planted: 24 predicted tokens");
  check(all.proposals == 3, "planted: 3 proposals");
  check(all.b[0].proposals == 1 && all.b[1].proposals == 2 && all.b[2].proposals == 0, "planted: buckets 1 / 2 / 0");
  check(all.b[0].prefix[7] == 1, "planted: bucket 0's proposal has 7 leading rows right");
  check(all.b[1].prefix[7] == 1 && all.b[1].prefix[0] == 1, "planted: bucket 1: one with 7 right, one with 0 right");
  bool ok = true;
  for (int j = 0; j < kMaxDrafts; ++j) ok = ok && all.b[0].n[j] == 1 && all.b[0].h[j] == 1;
  check(ok, "planted: bucket 0's table: 1 of 1 at every depth");
  ok = all.b[1].n[0] == 2 && all.b[1].h[0] == 1;
  for (int j = 1; j < kMaxDrafts; ++j) ok = ok && all.b[1].n[j] == 1 && all.b[1].h[j] == 1;
  check(ok, "planted: bucket 1's table: 1 of 2 at depth 1, 1 of 1 deeper");
  check(all.rows_accepted == 14, "planted: 14 copy tokens accepted (583.3 per 1,000)");
  check(all.steps == 7 + 1 + 1 + 1, "planted: 10 verify-all steps (7 plain, 3 with a proposal)");
  // the gate with the built-in table (ratio rule) verifies every row of both full proposals and the single row
  TableRule rule(prior_table(), kRatio, 0.3f, cost, 1.f);
  Totals g;
  for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, false, &rule, cost, o, &g);
  check(g.rows_verified == 15 && g.rows_accepted == 14, "planted gate: 15 rows verified, 14 accepted");
  check(g.rows_verified - g.rows_accepted == 1 && g.missed == 0, "planted gate: 1 wasted row, none missed");
  const double ms = 7 * 21.10 + 2 * cost.step_ms[7] + cost.step_ms[1];
  check(near(g.ms, ms, 0.01), "planted gate: modelled ms");
  // a table that says 0.01 keeps no row: nothing verified, 14 right rows missed
  Table low;
  for (int b = 0; b < kBuckets; ++b)
    for (int j = 0; j < kMaxDrafts; ++j) low.t[b][j] = 0.01f;
  TableRule none(low, kRatio, 0.3f, cost, 1.f);
  Totals z;
  for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, false, &none, cost, o, &z);
  check(z.rows_verified == 0 && z.rows_accepted == 0 && z.missed > 14, "planted low table: no rows verified, right rows missed at every proposal");
  check(z.steps == 24 && near(z.ms, 24 * 21.10, 0.01), "planted low table: one token a step");
  // a shortest match of 17: the replay steps one token at a time until the match has grown to 17 (after 17 output tokens),
  // then one proposal of 7 right rows (bucket 1) takes it to the end
  Options o17;
  o17.min_match = 17;
  Totals m17;
  for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, true, nullptr, cost, o17, &m17);
  check(m17.proposals == 1 && m17.b[1].proposals == 1 && m17.b[1].prefix[7] == 1, "planted min-match 17: one proposal, bucket 1, 7 rows right");
  check(m17.steps == 16 + 1, "planted min-match 17: 16 plain steps, then the proposal's step");
  // empirical table from this data: depth 1 of bucket 1 is 1/2
  Table e = empirical_table(all, prior_table());
  check(near(e.t[1][0], 0.5) && near(e.t[0][0], 1.0) && e.t[1][1] <= e.t[1][0], "planted: empirical table");
}

void test_no_repeats() {
  Fresh f(5000);
  Streams st;
  st.ids.push_back(kBos);
  for (int i = 0; i < 200; ++i) st.ids.push_back(f());
  st.ids.push_back(kAsst), st.ids.push_back(kThink);
  for (int i = 0; i < 100; ++i) st.ids.push_back(f());
  st.ids.push_back(kEnd);
  split(&st, Ids{}, true);
  Cost cost;
  Options o;
  Totals all;
  for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, true, nullptr, cost, o, &all);
  check(all.proposals == 0 && all.rows_accepted == 0, "no repeats: zero proposals");
  check(all.predicted == 100 && all.steps == 100, "no repeats: 100 predicted tokens, one a step");
  TableRule rule(prior_table(), kRatio, 0.3f, cost, 1.f);
  Totals g;
  for (const Conv& cv : st.convs) replay_conv(st.ids.data() + cv.off, cv.len, cv.spans, false, &rule, cost, o, &g);
  check(g.rows_verified == 0 && near(g.ms, 100 * 21.10, 0.01), "no repeats gated: nothing verified, base cost");
}

void test_spans() {
  // conversation 1: two assistant turns; conversation 2: no assistant marker; conversation 3: output cut (no end id)
  std::vector<int32_t> v = {kBos, 7, 8, kAsst, kThink, 9, 10, 11, kEnd, 12, kAsst, 128822, 13, 14, kEnd,
                            kBos, 5, 6, 7,
                            kBos, 3, kAsst, kThink, 4, 5, 6};
  Streams a;
  a.ids = v;
  split(&a, Ids{}, false);
  check(a.convs.size() == 3, "spans: three conversations");
  check(a.convs[0].spans.size() == 2 && a.convs[0].spans[0].s == 5 && a.convs[0].spans[0].e == 9, "spans: first turn [5, 9) with its end id");
  check(a.convs[0].spans[1].s == 12 && a.convs[0].spans[1].e == 15, "spans: second turn [12, 15)");
  check(a.convs[1].spans.empty(), "spans: no marker, no span");
  check(a.convs[2].spans.size() == 1 && a.convs[2].spans[0].e == a.convs[2].len, "spans: a cut output runs to the end");
  Streams b;
  b.ids = v;
  split(&b, Ids{}, true);
  check(b.convs[0].spans.size() == 1 && b.convs[0].spans[0].s == 12, "spans: last only keeps the last turn");
  Streams c;  // ids before the first begin id are no conversation
  c.ids = {9, 9, kBos, 1, kAsst, kThink, 2, 3, kEnd};
  split(&c, Ids{}, true);
  check(c.convs.size() == 1 && c.convs[0].off == 2, "spans: leading ids dropped");
}

// the matcher as the replay drives it, against a plain reference: at every position of the span the proposal's source end,
// match length and tokens equal the latest earlier occurrence (a table large enough that no slot is shared)
// LearnRule's outcome is the engine's: for proposals of every length m, rows verified n (the rule's own gate: thresholds that
// keep all rows, two, or none) and leading rows right a, the outcome the rule builds from replay_conv's calls equals
// copy::outcome() on the verdict a decode step gives (count = accepted rows + 1, bonus = the sampler's token after them).
void test_learn_outcome() {
  Cost cost;
  int cases = 0, bad = 0, nseen[kMaxDrafts + 1] = {};
  for (float th : {0.f, 0.8f, 2.f})
    for (int m = 1; m <= kMaxDrafts; ++m)
      for (int a = 0; a <= m; ++a) {
        int32_t prop[kMaxDrafts], truth[kMaxDrafts + 1];
        for (int j = 0; j < m; ++j) prop[j] = 100 + j, truth[j] = j < a ? 100 + j : 900 + j;
        truth[m] = 999;
        LearnRule r(prior_table(), kThresh, th, cost, 1.f);
        float surv[kMaxDrafts];
        const int n = r.rows({0, m, 8, prop}, surv);
        ++nseen[n];
        const int acc = a < n ? a : n;  // replay_conv: the leading right rows the step verified
        for (int j = 0; j < acc; ++j) r.outcome(0, j, true);
        if (acc < m) r.outcome(0, acc, prop[acc] == truth[acc]);
        Pend pd{};
        pd.pos = 50, pd.k = 0, pd.n = (int8_t)n, pd.m = (int8_t)m, pd.bucket = 0, pd.whole = 1;
        for (int j = 0; j < m; ++j) pd.tok[j] = prop[j];
        const Taken e = outcome(pd, true, -1, 50, acc + 1, truth[acc]);
        const Taken& t = r.pending();
        bad += !(e.bucket == t.bucket && e.landed == t.landed && e.fr == t.fr);
        ++cases;
      }
  check(bad == 0, "learner: the replay's outcome equals copy::outcome() for every m, rows verified and rows right");
  check(cases == 105 && nseen[0] > 0 && nseen[2] > 0 && nseen[kMaxDrafts] > 0, "learner: 105 cases, with none, two and all rows verified");
}

void test_reference() {
  uint64_t rng = 0x9E3779B97F4A7C15ull;
  auto rnd = [&] {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(rng >> 33);
  };
  int compared = 0, bad = 0;
  for (int trial = 0; trial < 20; ++trial) {
    const int alpha = 3 + trial % 5, n = 300;
    std::vector<int32_t> H;
    H.push_back(kBos);
    for (int i = 0; i < n; ++i) H.push_back(10 + (int32_t)(rnd() % alpha));
    H.push_back(kAsst), H.push_back(kThink);
    const int s = (int)H.size();
    for (int i = 0; i < 200; ++i) H.push_back(10 + (int32_t)(rnd() % alpha));
    H.push_back(kEnd);
    // every position as a step start (a replay's steps skip positions), so the matcher is driven directly the way a step finds it
    std::vector<int32_t> hist = H;
    size_t tab = 16;
    while (tab < H.size() * 4096) tab <<= 1;
    std::vector<uint32_t> table(tab, 0);
    Unit u{};
    u.tok = hist.data(), u.tab = table.data(), u.mask = (uint32_t)(tab - 1), u.cap = (int)H.size(), u.len = 0;
    int indexed = 0;
    const int end = (int)H.size();
    for (int P = s; P < end - 1; ++P) {
      u.len = P + 1;
      index_range(u, indexed, P - 1);
      indexed = P;
      int32_t out[kMaxDrafts];
      Match mt;
      const int m = propose(u, P, nullptr, 0, std::min(kMaxDrafts, end - 1 - P), kOrder, out, &mt);
      // reference
      int re = -1, rl = 0;
      for (int e = P - 1; e >= kOrder - 1 && re < 0; --e) {
        bool eq = true;
        for (int i = 0; i < kOrder && eq; ++i) eq = H[e - kOrder + 1 + i] == H[P - kOrder + 1 + i];
        if (!eq) continue;
        re = e, rl = kOrder;
        for (int j = 0; j < kBack; ++j) {
          const int a = e - kOrder - j, b = P - kOrder - j;
          if (a < 0 || b < 0 || H[a] != H[b]) break;
          ++rl;
        }
      }
      ++compared;
      bool ok = re == mt.e && (re < 0 || rl == mt.len);
      if (ok && re >= 0) {
        const int want = std::min({P - re, kMaxDrafts, end - 1 - P});
        ok = m == want;
        for (int c = 0; c < m && ok; ++c) ok = out[c] == H[re + 1 + c];
      }
      if (!ok) ++bad;
    }
  }
  check(compared > 3000 && bad == 0, "matcher: every position equals the plain reference");
}

int selftest() {
  test_spans();
  test_planted();
  test_no_repeats();
  test_reference();
  test_learn_outcome();
  std::printf("copy_replay self-test: %d checks, %d failed\n", checks, fails);
  if (!fails) std::printf("PASS copy_replay\n");
  return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1 || (argc == 2 && !std::strcmp(argv[1], "--selftest"))) return selftest();
  Config c;
  auto need = [&](int& i) -> const char* {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "copy_replay: %s needs a value\n", argv[i]);
      std::exit(2);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--spans") c.last_only = std::string(need(i)) != "all";
    else if (a == "--min-match") c.opt.min_match = std::min(std::max(std::atoi(need(i)), (int)kOrder), kOrder + kBack);
    else if (a == "--table") c.tables.push_back(need(i));
    else if (a == "--gate") {
      const std::string g = need(i);
      c.gate = g == "marginal" ? kMarginal : g == "thresh" ? kThresh : kRatio;
    } else if (a == "--thresh") c.thresh = (float)std::atof(need(i));
    else if (a == "--chain-prev") c.chain_prev = (float)std::atof(need(i));
    else if (a == "--tab-mult") c.opt.tab_mult = std::max(1, std::atoi(need(i)));
    else if (a == "--step-ms") {
      float v[6];
      if (std::sscanf(need(i), "%f,%f,%f,%f,%f,%f", v, v + 1, v + 2, v + 3, v + 4, v + 5) != 6) {
        std::fprintf(stderr, "copy_replay: --step-ms wants six numbers\n");
        return 2;
      }
      for (int j = 0; j < 6; ++j) c.cost.step_ms[j] = v[j];
      for (int j = 6; j <= kMaxDrafts; ++j) c.cost.step_ms[j] = c.cost.step_ms[j - 1] + std::max(0.f, c.cost.step_ms[j - 1] - c.cost.step_ms[j - 2]);
    } else if (a == "--ids") {
      if (std::sscanf(need(i), "%d,%d,%d", &c.ids.bos, &c.ids.assistant, &c.ids.eos) != 3) {
        std::fprintf(stderr, "copy_replay: --ids wants BOS,ASSISTANT,EOS\n");
        return 2;
      }
    } else if (a == "--json") c.json = true;
    else if (a == "--group") c.groups.push_back({need(i), {}});
    else if (a == "--combined") c.combined = need(i);
    else if (a == "--dir") {
      const std::string d = need(i);
      DIR* dp = opendir(d.c_str());
      if (!dp) {
        std::fprintf(stderr, "copy_replay: cannot open directory %s\n", d.c_str());
        return 2;
      }
      std::vector<std::string> fs;
      while (dirent* e = readdir(dp)) {
        const std::string n = e->d_name;
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".i32") == 0) fs.push_back(d + "/" + n);
      }
      closedir(dp);
      std::sort(fs.begin(), fs.end());
      if (c.groups.empty()) c.groups.push_back({"streams", {}});
      c.groups.back().files.insert(c.groups.back().files.end(), fs.begin(), fs.end());
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "copy_replay: unknown option %s\n", a.c_str());
      return 2;
    } else {
      if (c.groups.empty()) c.groups.push_back({"streams", {}});
      c.groups.back().files.push_back(a);
    }
  }
  if (c.tables.empty()) c.tables.push_back("builtin");
  if (c.groups.empty()) {
    std::fprintf(stderr, "copy_replay: no stream files\n");
    return 2;
  }
  if (!c.combined.empty()) {
    Config::Group all{c.combined, {}};
    for (const Config::Group& g : c.groups) all.files.insert(all.files.end(), g.files.begin(), g.files.end());
    c.groups.push_back(all);
  }
  std::vector<SourceResult> rs;
  for (const Config::Group& g : c.groups) {
    std::vector<Streams> sts(g.files.size());
    for (size_t k = 0; k < g.files.size(); ++k) {
      if (!read_i32(g.files[k], &sts[k].ids)) {
        std::fprintf(stderr, "copy_replay: cannot read %s as int32 ids\n", g.files[k].c_str());
        return 2;
      }
      split(&sts[k], c.ids, c.last_only);
    }
    rs.push_back(run_source(g.name, sts, c));
  }
  if (c.json) print_json(rs, c);
  else
    for (const SourceResult& r : rs) print_text(r, c);
  return 0;
}
