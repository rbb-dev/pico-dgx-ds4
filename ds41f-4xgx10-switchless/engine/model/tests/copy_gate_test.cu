// Copy drafting's learner in the two gate kernels of spec.cu, on a GB10 (the CMake target copy_gate_test; one rank, no ring, no
// weights, allocations of a few MiB, so it can run beside an engine). Not run by the build: it needs a GPU.
//   copy_gate_test
// What it proves, each against something other than the kernel's own code:
//  1. The one-session kernel (k_spec_next) on a stream with a known continuation, a step at a time with the verdict the sampler would
//     give: every step's cells, conversation counts, pending record, observation word and chosen rows equal the host
//     reference of model/copy.h bit for bit, on a stream with phrases and one whose copies fail.
//  2. The batch kernel (k_spec_gate_batch_all) with entries that join and leave the step (sess_on changing), each over its own stream:
//     the same bit-for-bit equality, and a word of 0 for every entry that did not run, every step.
//  3. The attach reset: a conversation's counts are cleared by a new handle and kept by the same one.
//  4. Cost: both kernels with and without the learner, 2,000 launches each in a graph, per-launch microseconds (a measurement, no pass bar).
// Settings it needs: none (the kernel's mode bits are passed in the step's input, as the engine passes copy::mode()); a GPU not in use by
// a benchmark when the timing is read.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "model/copy_dev.h"

using namespace model;
using namespace model::copy;

namespace {

uint64_t g_rng = 0xA24BAED4963EE407ull;
uint32_t rnd() {
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33);
}
int fails = 0, checks = 0;
void check(bool ok, const char* what, int it = 0) {
  ++checks;
  if (ok) return;
  if (++fails <= 40) std::printf("FAIL (step %d): %s\n", it, what);
}

// A token stream with repeated phrases; noise changes a token now and then (a copy that fails); `noise` per thousand.
std::vector<int32_t> stream(int n, int alpha, int phrase_p, int noise) {
  std::vector<int32_t> s;
  while ((int)s.size() < n) {
    if (s.size() > 20 && (int)(rnd() % 100) < phrase_p) {
      const int len = 8 + rnd() % 40, from = rnd() % (s.size() - 10);
      for (int i = 0; i < len && (int)s.size() < n && from + i < (int)s.size(); ++i)
        s.push_back((int)(rnd() % 1000) < noise ? (int32_t)(rnd() % alpha) : s[from + i]);
    } else {
      s.push_back(rnd() % alpha);
    }
  }
  return s;
}

struct Host {  // the host's unit with the same shape as the device's
  std::vector<int32_t> tok;
  std::vector<uint32_t> tab;
  Unit u{};
  explicit Host(const Unit& d) : tok(d.cap, 0), tab((size_t)d.mask + 1, 0) {
    u.tok = tok.data(), u.tab = tab.data(), u.mask = d.mask, u.cap = d.cap, u.len = 0;
  }
  void set(const std::vector<int32_t>& t) {
    for (size_t i = 0; i < t.size(); ++i) tok[i] = t[i];
    u.len = (int)t.size();
    index_range(u, 0, u.len - 1);
  }
};

// What a gate run does for one unit, from the host's side (the same functions the kernels call, in the kernels' order).
struct Ref {
  Host h;
  UnitLearn ul{};
  Taken tk;
  int n_copy = 0, m = 0;
  int32_t ctok[kMaxDrafts] = {};
  Match mt;
  explicit Ref(const Unit& d) : h(d) {}
  // phase 1: the outcome and the conversation's counts, then the proposal and the rows the gate keeps
  void run(const Learn& L, int pos0, int count, const int32_t* rows, int32_t bonus, float thresh, int pos_end, int min_match) {
    append_step(h.u, pos0, count, rows, bonus);
    tk = outcome(ul.pend, true, -1, pos0, count, bonus);
    sess_apply(ul.sess, L, tk);
    const int P = pos0 + count;
    n_copy = 0, m = 0, mt = Match{};
    std::memset(ctok, 0, sizeof ctok);
    const int maxm = std::min(kMaxDrafts, pos_end - 1 - P);
    if (maxm > 0) {
      m = propose(h.u, P, nullptr, 0, maxm, min_match, ctok, &mt);
      if (m > 0) {
        float lt[1][kMaxDrafts], cs[kMaxDrafts], step_ms[kMaxDrafts + 1] = {};
        gate_row(L, ul.sess, mt.bucket, prior_table().t[mt.bucket], m, lt[0]);
        n_copy = gate_rows(0, thresh, step_ms, 1.f, 1.f, 0, m, 0, lt, cs);
      }
    }
    Pend pd{};
    if (m > 0) {
      pd.pos = P, pd.k = 0, pd.n = (int8_t)n_copy, pd.m = (int8_t)m, pd.bucket = (int8_t)mt.bucket, pd.whole = 1;
      for (int j = 0; j < m; ++j) pd.tok[j] = ctok[j];
    }
    ul.pend = pd;
  }
  uint32_t obs() const { return pack_obs(tk, n_copy, m > 0, ratio(ul.sess)); }
};

bool same_learn(const Learn& a, const Learn& b) { return std::memcmp(&a, &b, sizeof(Learn)) == 0; }
bool same_unit(const UnitLearn& a, const UnitLearn& b) { return std::memcmp(&a, &b, sizeof(UnitLearn)) == 0; }

ops::StepIn* new_in() {
  ops::StepIn* p;
  CUDA_CHECK(cudaMalloc(&p, sizeof(ops::StepIn)));
  return p;
}
void put_in(ops::StepIn* d, const ops::StepIn& h) { CUDA_CHECK(cudaMemcpy(d, &h, sizeof h, cudaMemcpyHostToDevice)); }

// ---- 1. one session, a stream of known continuation ----
struct Counts {
  long scored[kBuckets] = {}, first_right[kBuckets] = {};
};

Counts run_single(const char* name, const std::vector<int32_t>& S, int prompt, int steps, cudaStream_t st) {
  const int ctx[1] = {(int)S.size() + 64};
  Index ix(ctx, 1, (size_t)8 << 20, st);
  const std::vector<int32_t> pr(S.begin(), S.begin() + prompt);
  ix.set_tokens(0, 0, prompt, pr.data(), st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  Learn L;  // the host reference's shared cells
  learn_init(&L, prior_table());
  Ref ref(ix.view(0));
  ref.h.set(pr);

  ops::StepIn* din = new_in();
  ops::SpecState* dss;
  int *ddr;
  float* dsv;
  ops::SampleOut* outs;
  CUDA_CHECK(cudaMalloc(&dss, sizeof(ops::SpecState)));
  CUDA_CHECK(cudaMalloc(&ddr, sizeof(int) * ops::kDrafts));
  CUDA_CHECK(cudaMalloc(&dsv, sizeof(float) * ops::kDrafts));
  CUDA_CHECK(cudaMemset(ddr, 0, sizeof(int) * ops::kDrafts));
  CUDA_CHECK(cudaMemset(dsv, 0, sizeof(float) * ops::kDrafts));  // no chain drafts: survival 0 is under every threshold
  CUDA_CHECK(cudaMallocHost(&outs, sizeof(ops::SampleOut) * ops::kOutSlots));
  std::memset(outs, 0, sizeof(ops::SampleOut) * ops::kOutSlots);
  Args cp{ix.dev(), nullptr};

  Counts C;
  std::vector<int32_t> rows{S[prompt]};
  int pos0 = prompt;
  const float thresh = 0.30f;
  const int pos_end = (int)S.size() - 2;
  for (int i = 0; i < steps; ++i) {
    // the sampler's verdict: row j's token is the stream's at pos0 + j + 1; a draft is accepted when it is that token
    const int R = (int)rows.size();
    int acc = 0;
    while (acc < R - 1 && rows[acc + 1] == S[pos0 + acc + 1]) ++acc;
    const int count = acc + 1;
    const int32_t bonus = S[pos0 + count];
    if (pos0 + count + 9 >= (int)S.size()) break;
    ops::StepIn in{};
    for (int j = 0; j < R; ++j) in.tokens[j] = rows[j], in.pos[j] = pos0 + j;
    in.rows = R, in.nsess = 1, in.step = (uint32_t)(100 + i), in.spec = ops::kSpecVerify, in.thresh = thresh, in.max_drafts = ops::kDrafts;
    in.pos_end = pos_end, in.force = -1, in.gate = ops::kGateThresh, in.copy = kRows | kAdapt, in.copy_min = kOrder;
    put_in(din, in);
    ops::SpecState ss{};
    ss.count = count, ss.pos0 = pos0, ss.bonus = bonus;
    CUDA_CHECK(cudaMemcpy(dss, &ss, sizeof ss, cudaMemcpyHostToDevice));
    test::gate_one(din, dss, ddr, dsv, outs, cp, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const ops::SampleOut& o = outs[in.step % ops::kOutSlots];
    ref.run(L, pos0, count, rows.data(), bonus, thresh, pos_end, kOrder);
    Taken one[1] = {ref.tk};
    fold_all(L, one, 1);
    Learn dl;
    std::vector<UnitLearn> du;
    ix.learn_copy(&dl, &du);
    check(same_learn(dl, L), "the shared cells equal the host reference", i);
    check(same_unit(du[0], ref.ul), "the conversation's counts and the pending record equal the host reference", i);
    check(o.copy_obs[0] == ref.obs(), "the observation word equals the host reference", i);
    check(o.next_rows == 1 + ref.n_copy && o.next_tok[0] == bonus, "the rows chosen equal the host reference", i);
    for (int j = 0; j < ref.n_copy; ++j) check(o.next_tok[1 + j] == ref.ctok[j], "a copy row's token equals the host reference", i);
    for (int b = 0; b < kBuckets; ++b)
      check(o.copy_chance[b] == round_u32(fmul_rn(L.chance[b][0], 65535.f)), "the published chance equals the host reference", i);
    const Obs ob = unpack_obs(o.copy_obs[0]);
    if (ob.taken) C.scored[ob.bucket]++, C.first_right[ob.bucket] += (ob.landed >= 1 || ob.fr == 1) ? 1 : 0;
    rows.assign(o.next_tok, o.next_tok + o.next_rows);
    pos0 = o.next_pos0;
  }
  std::printf("%s: %ld proposals scored by the kernel\n", name, C.scored[0] + C.scored[1] + C.scored[2] + C.scored[3]);
  cudaFree(din), cudaFree(dss), cudaFree(ddr), cudaFree(dsv), cudaFreeHost(outs);
  return C;
}

// ---- 2. the batch kernel with entries that join and leave ----
void run_batch(cudaStream_t st) {
  const int kE = 2;
  std::vector<int32_t> S[kE] = {stream(30000, 40, 80, 20), stream(30000, 40, 60, 150)};
  const int ctx[kE] = {30064, 30064};
  Index ix(ctx, kE, (size_t)16 << 20, st);
  Learn L;
  learn_init(&L, prior_table());
  std::vector<Ref> ref;
  ref.reserve(kE);
  std::vector<std::vector<int32_t>> rows(kE);
  int pos0[kE];
  const int prompt = 200;
  for (int k = 0; k < kE; ++k) {
    const std::vector<int32_t> pr(S[k].begin(), S[k].begin() + prompt);
    ix.set_tokens(k, 0, prompt, pr.data(), st);
    ref.emplace_back(ix.view(k));
    ref[k].h.set(pr);
    rows[k] = {S[k][prompt]};
    pos0[k] = prompt;
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
  ops::StepIn* din = new_in();
  ops::SpecState* dss;
  int *dtok, *drow0;
  float* dsurv;
  ops::SampleOut* outs;
  CUDA_CHECK(cudaMalloc(&dss, sizeof(ops::SpecState) * ops::kMaxSess));
  CUDA_CHECK(cudaMalloc(&dtok, sizeof(int) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMalloc(&dsurv, sizeof(float) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMalloc(&drow0, sizeof(int) * ops::kMaxSess));
  CUDA_CHECK(cudaMemset(dss, 0, sizeof(ops::SpecState) * ops::kMaxSess));
  CUDA_CHECK(cudaMemset(dtok, 0, sizeof(int) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMemset(dsurv, 0, sizeof(float) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMallocHost(&outs, sizeof(ops::SampleOut) * ops::kOutSlots));
  std::memset(outs, 0xAB, sizeof(ops::SampleOut) * ops::kOutSlots);  // stale words from other steps: the kernel must overwrite every one it owns
  const float thresh = 0.30f;
  const int pos_end = 29900;
  long idle_words_nonzero = 0, steps_run = 0;
  for (int i = 0; i < 1500; ++i) {
    bool on[kE];
    on[0] = rnd() % 4 != 0, on[1] = rnd() % 3 != 0;  // entries join and leave the step
    if (!on[0] && !on[1]) on[rnd() % 2] = true;
    ops::StepIn in{};
    ops::SpecState ss[kE] = {};
    int row0[kE] = {}, r = 0;
    bool room = true;
    for (int k = 0; k < kE; ++k) {
      if (!on[k]) continue;
      const int R = (int)rows[k].size();
      int acc = 0;
      while (acc < R - 1 && rows[k][acc + 1] == S[k][pos0[k] + acc + 1]) ++acc;
      ss[k].count = acc + 1, ss[k].pos0 = pos0[k], ss[k].bonus = S[k][pos0[k] + acc + 1];
      if (pos0[k] + acc + 12 >= (int)S[k].size() || pos0[k] + acc + 12 >= pos_end) room = false;
      row0[k] = r;
      for (int j = 0; j < R; ++j) in.tokens[r + j] = rows[k][j], in.pos[r + j] = pos0[k] + j, in.sess[r + j] = k, in.slot[r + j] = k;
      r += R;
    }
    if (!room) break;
    in.rows = r, in.nsess = kE, in.step = (uint32_t)(500 + i), in.spec = ops::kSpecVerify, in.sbatch = 1, in.force = -1;
    in.gate = ops::kGateThresh, in.copy = kRows | kAdapt, in.copy_min = kOrder, in.rbudget = kDecodeRows;
    for (int k = 0; k < kE; ++k)
      in.sess_on[k] = on[k] ? 1 : 0, in.sess_slot[k] = k, in.scap[k] = ops::kDrafts, in.spos_end[k] = pos_end, in.sthresh[k] = thresh;
    put_in(din, in);
    // ss of an entry that did not run is what its last run left it: the device keeps it, the test only writes the entries that run
    for (int k = 0; k < kE; ++k)
      if (on[k]) CUDA_CHECK(cudaMemcpy(dss + k, &ss[k], sizeof(ops::SpecState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(drow0, row0, sizeof row0, cudaMemcpyHostToDevice));
    test::gate_batch(din, dss, dtok, 8, dsurv, 8, outs, Args{ix.dev(), drow0}, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const ops::SampleOut& o = outs[in.step % ops::kOutSlots];
    Taken tk[kE];
    for (int k = 0; k < kE; ++k) {
      if (!on[k]) continue;
      ref[k].run(L, ss[k].pos0, ss[k].count, in.tokens + row0[k], ss[k].bonus, thresh, pos_end, kOrder);
      tk[k] = ref[k].tk;
    }
    fold_all(L, tk, kE);
    Learn dl;
    std::vector<UnitLearn> du;
    ix.learn_copy(&dl, &du);
    check(same_learn(dl, L), "batch: the shared cells equal the host reference", i);
    for (int k = 0; k < ops::kMaxSess; ++k) {
      if (k < kE && on[k]) {
        check(o.copy_obs[k] == ref[k].obs(), "batch: an entry's observation word equals the host reference", i);
        check(same_unit(du[k], ref[k].ul), "batch: an entry's counts and pending record equal the host reference", i);
        check(o.next_srows[k] == 1 + ref[k].n_copy, "batch: an entry's rows equal the host reference", i);
        rows[k].assign(o.next_stok[k], o.next_stok[k] + o.next_srows[k]);
        pos0[k] = o.next_spos0[k];
      } else {
        idle_words_nonzero += o.copy_obs[k] != 0;  // an entry that did not run, or does not exist, says nothing
      }
    }
    ++steps_run;
  }
  check(idle_words_nonzero == 0, "batch: every word of an entry that did not run is 0 on every step");
  std::printf("batch: %ld steps, entries joining and leaving; %ld words of idle entries not 0\n", steps_run, idle_words_nonzero);
  cudaFree(din), cudaFree(dss), cudaFree(dtok), cudaFree(dsurv), cudaFree(drow0), cudaFreeHost(outs);
}

// ---- 3. the attach reset ----
void run_attach(cudaStream_t st) {
  const int ctx[2] = {4096, 4096};
  Index ix(ctx, 2, (size_t)4 << 20, st);
  // the mode bit is read at construction: without DSGB10_COPY_ADAPT in the environment attach does nothing, which this checks too
  const bool adapt = (mode() & kAdapt) != 0;
  UnitLearn mark{};
  mark.sess.n = 5, mark.sess.miss = 2, mark.sess.exp16 = 777;
  UnitLearn* dev_ul = reinterpret_cast<UnitLearn*>(reinterpret_cast<uint8_t*>(ix.dev()) + offsetof(Dev, ul));
  CUDA_CHECK(cudaMemcpy(dev_ul, &mark, sizeof mark, cudaMemcpyHostToDevice));
  ix.attach(0, 41, st);  // a new handle
  Learn L;
  std::vector<UnitLearn> u;
  CUDA_CHECK(cudaStreamSynchronize(st));
  ix.learn_copy(&L, &u);
  check(adapt ? u[0].sess.n == 0 : u[0].sess.n == 5, "attach with a new handle clears the counts (with DSGB10_COPY_ADAPT=1 in the environment; else it leaves them)");
  CUDA_CHECK(cudaMemcpy(dev_ul, &mark, sizeof mark, cudaMemcpyHostToDevice));
  ix.attach(0, 41, st);  // the same handle again
  CUDA_CHECK(cudaStreamSynchronize(st));
  ix.learn_copy(&L, &u);
  check(u[0].sess.n == 5 && u[0].sess.exp16 == 777, "attach with the same handle keeps the counts");
  std::printf("attach: tested with %s\n", adapt ? "DSGB10_COPY_ADAPT=1 (the reset)" : "the switch off (nothing to reset): run again with DSGB10_COPY_ADAPT=1 DSGB10_COPY=1");
}

// ---- 4. the cost ----
void time_kernels(cudaStream_t st) {
  const std::vector<int32_t> S = stream(30000, 40, 80, 20);
  const int ctx[2] = {30064, 30064};
  Index ix(ctx, 2, (size_t)16 << 20, st);
  const std::vector<int32_t> pr(S.begin(), S.begin() + 2000);
  for (int k = 0; k < 2; ++k) ix.set_tokens(k, 0, 2000, pr.data(), st);
  ops::StepIn* din = new_in();
  ops::SpecState* dss;
  int *ddr, *drow0;
  float* dsv;
  ops::SampleOut* outs;
  CUDA_CHECK(cudaMalloc(&dss, sizeof(ops::SpecState) * ops::kMaxSess));
  CUDA_CHECK(cudaMalloc(&ddr, sizeof(int) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMalloc(&dsv, sizeof(float) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMalloc(&drow0, sizeof(int) * ops::kMaxSess));
  CUDA_CHECK(cudaMemset(ddr, 0, sizeof(int) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMemset(dsv, 0, sizeof(float) * ops::kMaxSess * 8));
  CUDA_CHECK(cudaMemset(drow0, 0, sizeof(int) * ops::kMaxSess));
  CUDA_CHECK(cudaMallocHost(&outs, sizeof(ops::SampleOut) * ops::kOutSlots));
  std::memset(outs, 0, sizeof(ops::SampleOut) * ops::kOutSlots);
  ops::SpecState ss[2] = {};
  for (int k = 0; k < 2; ++k) ss[k].count = 3, ss[k].pos0 = 1990, ss[k].bonus = S[1993];
  CUDA_CHECK(cudaMemcpy(dss, ss, sizeof ss, cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  const int kLaunches = 2000;
  double us[2][2];  // [batch][adapt]
  for (int batch = 0; batch < 2; ++batch)
    for (int adapt = 0; adapt < 2; ++adapt) {
      ops::StepIn in{};
      for (int j = 0; j < 8; ++j) in.tokens[j] = S[1990 + j], in.pos[j] = 1990 + j;
      in.rows = batch ? 8 : 4, in.nsess = batch ? 2 : 1, in.step = 900, in.spec = ops::kSpecVerify, in.thresh = 0.30f, in.max_drafts = ops::kDrafts;
      in.pos_end = 29900, in.force = -1, in.gate = ops::kGateThresh, in.copy = kRows | (adapt ? kAdapt : 0), in.copy_min = kOrder;
      in.sbatch = batch;
      for (int k = 0; k < 2; ++k) in.sess_on[k] = 1, in.sess_slot[k] = k, in.scap[k] = ops::kDrafts, in.spos_end[k] = 29900, in.sthresh[k] = 0.30f;
      in.rbudget = batch ? kDecodeRows : 0;
      put_in(din, in);
      const int row0[2] = {0, 4};
      CUDA_CHECK(cudaMemcpy(drow0, row0, sizeof row0, cudaMemcpyHostToDevice));
      cudaGraph_t g;
      cudaGraphExec_t ge;
      CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < kLaunches; ++i) {
        if (batch) test::gate_batch(din, dss, ddr, 8, dsv, 8, outs, Args{ix.dev(), drow0}, st);
        else test::gate_one(din, dss, ddr, dsv, outs, Args{ix.dev(), nullptr}, st);
      }
      CUDA_CHECK(cudaStreamEndCapture(st, &g));
      CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
      CUDA_CHECK(cudaGraphLaunch(ge, st));  // warm
      CUDA_CHECK(cudaStreamSynchronize(st));
      double best = 1e30;
      for (int rep = 0; rep < 5; ++rep) {
        CUDA_CHECK(cudaEventRecord(e0, st));
        CUDA_CHECK(cudaGraphLaunch(ge, st));
        CUDA_CHECK(cudaEventRecord(e1, st));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, 1000.0 * ms / kLaunches);
      }
      us[batch][adapt] = best;
      cudaGraphExecDestroy(ge), cudaGraphDestroy(g);
    }
  std::printf("cost, microseconds a launch (best of 5 graphs of %d launches, a stream of phrases, state evolving): one-session off %.2f on %.2f (+%.2f); "
              "batch off %.2f on %.2f (+%.2f)\n", kLaunches, us[0][0], us[0][1], us[0][1] - us[0][0], us[1][0], us[1][1], us[1][1] - us[1][0]);
  cudaFree(din), cudaFree(dss), cudaFree(ddr), cudaFree(dsv), cudaFree(drow0), cudaFreeHost(outs);
}

}  // namespace

int main() {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  // 1. a stream with phrases and little noise, and one whose copies mostly fail
  for (int variant = 0; variant < 2; ++variant) {
    const std::vector<int32_t> S = variant == 0 ? stream(40000, 40, 80, 20) : stream(40000, 40, 70, 400);
    const char* name = variant ? "one session, copies that fail" : "one session, phrases";
    run_single(name, S, 300, 6000, st);
  }
  run_batch(st);
  run_attach(st);
  time_kernels(st);
  std::printf("copy_gate_test: %d checks, %d failures\n", checks, fails);
  return fails ? 1 : 0;
}
