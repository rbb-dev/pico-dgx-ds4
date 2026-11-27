// Offline replay of copy drafting (model/copy.h) on recorded token streams: no GPU, no engine, no timing.
//
// A stream file is little-endian int32 ids, conversations one after another. A conversation starts at the begin-of-
// sentence id; the model's own output is what follows an assistant marker (plus the one think / end-think id that
// the generation marker carries) up to and including the next end-of-sentence id, or up to the conversation's end.
// At every step of an output span the replay runs the engine's own matcher (copy::index_range, copy::propose) on the
// history so far, takes the proposed copy tokens and scores them against the tokens the model produced next. Two views:
//  - verify-all: every proposed row is verified, so the data is the matcher's own;
//  - gated: a rule (a table through copy::gate_rows, or any class with the Rule interface) picks how many rows a step
//    verifies, the step emits the accepted rows plus the sampler's token, and the next proposal starts from there.
// The model has no chain drafts here (k = 0): the replay measures copy rows alone.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "model/copy.h"

namespace copyreplay {

using namespace model::copy;

struct Ids {
  int32_t bos = 0, assistant = 128804, eos = 1;  // the 1017 encoding (frontend/src/encoding.rs markers)
};

struct Span {
  int s, e;  // output tokens [s, e) of the conversation (s: the first token the model sampled)
};
struct Conv {
  size_t off = 0;  // into the stream's ids
  int len = 0;
  std::vector<Span> spans;
};
struct Streams {
  std::vector<int32_t> ids;
  std::vector<Conv> convs;
};

inline bool read_i32(const std::string& path, std::vector<int32_t>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out->assign((size_t)n / 4, 0);
  const bool ok = n % 4 == 0 && std::fread(out->data(), 4, out->size(), f) == out->size();
  std::fclose(f);
  return ok;
}

// Split at the begin id; find the output spans (all of them, or only the last one of each conversation).
inline void split(Streams* st, const Ids& id, bool last_only) {
  st->convs.clear();
  const std::vector<int32_t>& v = st->ids;
  size_t i = 0;
  while (i < v.size() && v[i] != id.bos) ++i;  // ids before the first begin id belong to no conversation
  while (i < v.size()) {
    size_t j = i + 1;
    while (j < v.size() && v[j] != id.bos) ++j;
    Conv c;
    c.off = i, c.len = (int)(j - i);
    for (int p = 0; p + 2 < c.len; ++p) {
      if (v[i + p] != id.assistant) continue;
      const int s = p + 2;  // past the marker and its think / end-think id
      int e = s;
      while (e < c.len && v[i + e] != id.eos) ++e;
      if (e < c.len) ++e;  // the end id is the span's last token
      if (e - s >= 2) c.spans.push_back({s, e});
      p = e - 1;
    }
    if (last_only && c.spans.size() > 1) c.spans.erase(c.spans.begin(), c.spans.end() - 1);
    st->convs.push_back(std::move(c));
    i = j;
  }
}

// ---- the rule: how many of the proposed rows a step verifies ---------------------------------------------------
struct Proposal {
  int bucket, m, len;
  const int32_t* tok;
};
class Rule {
 public:
  virtual ~Rule() {}
  virtual void begin_conversation() {}
  // rows to verify of the m proposed; surv receives each row's modelled survival
  virtual int rows(const Proposal& p, float* surv) = 0;
  // what the step's verdict taught: copy token `depth` (0-based) of a proposal of `bucket` was right or not (the
  // outcomes a step's verdict gives: the accepted rows, and the one token after them)
  virtual void outcome(int bucket, int depth, bool hit) { (void)bucket; (void)depth; (void)hit; }
};

enum Gate { kThresh = 0, kRatio = 1, kMarginal = 2 };  // model::ops::kGate*

struct Cost {
  // the modelled ms of a step verifying n drafts, n = 0..7: server/cmd.h spec_row_ms (measured at 1k context) for 0..5,
  // then model/spec.cu spec_fill's rule (the last difference again) for 6 and 7
  float step_ms[kMaxDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f};
  Cost() {
    for (int i = 6; i <= kMaxDrafts; ++i) step_ms[i] = step_ms[i - 1] + std::max(0.f, step_ms[i - 1] - step_ms[i - 2]);
  }
};

class TableRule : public Rule {
 public:
  TableRule(const Table& t, Gate g, float thresh, const Cost& c, float chain_prev)
      : t_(t), gate_(g), thresh_(thresh), cost_(c), chain_prev_(chain_prev) {}
  int rows(const Proposal& p, float* surv) override {
    return gate_rows((int)gate_, thresh_, cost_.step_ms, chain_prev_, 1.f, 0, p.m, p.bucket, t_.t, surv);
  }

 private:
  Table t_;
  Gate gate_;
  float thresh_;
  Cost cost_;
  float chain_prev_;
};

// The engine's learner (copy.h, DSGB10_COPY_ADAPT): shared counts by bucket and depth class starting from a table, and a
// conversation's ratio. In the engine's order: a step first takes the outcome of the conversation's last proposal into its
// ratio, the gate reads the chances, then the shared counts take the outcome. A new conversation starts its ratio over.
class LearnRule : public Rule {
 public:
  LearnRule(const Table& t, Gate g, float thresh, const Cost& c, float chain_prev)
      : t_(t), gate_(g), thresh_(thresh), cost_(c), chain_prev_(chain_prev) {
    learn_init(&L_, t_);
  }
  void begin_conversation() override {
    flush();
    s_ = Sess{};
  }
  int rows(const Proposal& p, float* surv) override {
    Taken last = tk_;
    const bool had = have_;
    if (had) sess_apply(s_, L_, last);
    float lt[1][kMaxDrafts];
    gate_row(L_, s_, p.bucket, t_.t[p.bucket], p.m, lt[0]);
    const int n = gate_rows((int)gate_, thresh_, cost_.step_ms, chain_prev_, 1.f, 0, p.m, 0, lt, surv);
    if (had) fold_all(L_, &last, 1);
    tk_ = Taken{};
    tk_.bucket = (int8_t)p.bucket, tk_.n = (int8_t)n;
    have_ = true;
    return n;
  }
  void outcome(int bucket, int depth, bool hit) override {
    (void)bucket;
    if (depth != tk_.landed) return;
    if (hit && depth < tk_.n) {  // a verified copy row that was accepted
      ++tk_.landed;
      return;
    }
    tk_.fr = hit ? 1 : 0;  // the first row after them: a verified row rejected, or the free outcome of one not verified
  }
  const Learn& learn() const { return L_; }
  const Taken& pending() const { return tk_; }  // the outcome being built for the last proposal (tests)

 private:
  void flush() {
    if (have_) fold_all(L_, &tk_, 1);
    have_ = false, tk_ = Taken{};
  }
  Table t_;
  Gate gate_;
  float thresh_;
  Cost cost_;
  float chain_prev_;
  Learn L_;
  Sess s_{};
  Taken tk_;
  bool have_ = false;
};

// ---- the statistics --------------------------------------------------------------------------------------------
struct BucketStats {
  long proposals = 0;
  long n[kMaxDrafts] = {}, h[kMaxDrafts] = {};  // proposals with at least j+1 rows, of them with the first j+1 all right
  long prefix[kMaxDrafts + 1] = {};              // verify-all: proposals by the number of leading rows that were right
};
struct Totals {
  long convs = 0, spans = 0, prompt_tokens = 0, predicted = 0;
  long steps = 0, proposals = 0, rows_verified = 0, rows_accepted = 0, missed = 0;
  double ms = 0;
  BucketStats b[kBuckets];
};

struct Options {
  int min_match = kOrder;
  int tab_mult = 4;  // table entries per history position (the engine's plan: 4 at most)
};

// One conversation. all_rows: the verify-all view (rule unused). Else rule decides. Totals accumulate.
inline void replay_conv(const int32_t* tok, int len, const std::vector<Span>& spans, bool all_rows, Rule* rule, const Cost& cost,
                        const Options& o, Totals* T) {
  std::vector<int32_t> hist(tok, tok + len);
  size_t tab = 16;
  while (tab < (size_t)len * (size_t)o.tab_mult) tab <<= 1;
  std::vector<uint32_t> table(tab, 0);
  Unit u{};
  u.tok = hist.data(), u.tab = table.data(), u.mask = (uint32_t)(tab - 1), u.cap = len, u.len = 0;
  int indexed = 0;  // grams ending below this position are in the table
  if (rule) rule->begin_conversation();
  ++T->convs;
  T->prompt_tokens += spans.empty() ? len : spans.front().s;
  for (const Span& sp : spans) {
    ++T->spans;
    T->predicted += sp.e - 1 - sp.s;
    int P = sp.s;  // the last known position: the first sampled token is the step's bonus
    while (P < sp.e - 1) {
      // as a decode step at P finds the unit: the grams that end before P are indexed, the bonus's waits
      u.len = P + 1;
      index_range(u, indexed, P - 1);
      if (P > indexed) indexed = P;
      int32_t prop[kMaxDrafts];
      Match mt;
      const int m = propose(u, P, nullptr, 0, std::min(kMaxDrafts, sp.e - 1 - P), o.min_match, prop, &mt);
      int n = 0, a = 0;
      if (m > 0) {
        BucketStats& B = T->b[mt.bucket];
        float surv[kMaxDrafts];
        n = all_rows ? m : rule->rows({mt.bucket, m, mt.len, prop}, surv);
        while (a < m && prop[a] == tok[P + 1 + a]) ++a;  // leading rows that are right (the truth lies in the span)
        if (all_rows) {
          ++T->proposals, ++B.proposals, ++B.prefix[a];
          for (int j = 0; j < m; ++j) ++B.n[j], B.h[j] += j < a;
        } else {
          ++T->proposals;
          T->missed += std::max(0, a - n);
        }
        const int acc = std::min(a, n);
        T->rows_verified += n, T->rows_accepted += acc;
        if (!all_rows && rule) {
          for (int j = 0; j < acc; ++j) rule->outcome(mt.bucket, j, true);
          if (acc < m) rule->outcome(mt.bucket, acc, prop[acc] == tok[P + 1 + acc]);
        }
        a = acc;
      }
      const int emitted = std::min(a + 1, sp.e - 1 - P);
      ++T->steps;
      T->ms += cost.step_ms[n];
      P += emitted;
    }
  }
}

inline Table empirical_table(const Totals& v, const Table& fallback) {
  Table t = fallback;
  for (int b = 0; b < kBuckets; ++b) {
    float prev = 1.f;
    for (int j = 0; j < kMaxDrafts; ++j) {
      float x = v.b[b].n[j] ? (float)v.b[b].h[j] / (float)v.b[b].n[j] : fallback.t[b][j];
      x = std::min(std::max(x, 0.001f), prev);
      t.t[b][j] = prev = x;
    }
  }
  return t;
}

}  // namespace copyreplay
