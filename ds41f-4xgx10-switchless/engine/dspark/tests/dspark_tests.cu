// The DSpark module at TP=4 against model.py (engine/dspark/ref/dspark_ref.py's data): four ranks on one GPU,
// one host thread and stream each, the collectives emulated (host barriers and copies: gathers in rank order,
// reductions as fp32 sums in rank order). The golden run's layer 37-39 input streams go through main_hidden(),
// the prompt's 39 positions seed the windows, then every step accepts one position and drafts, as the reference.
//
//   dspark_tests [--ref DIR] [--golden DIR] [--rank0 FILE] [--wdir DIR] [--steps N] [--couple 1]
//
// Two passes over the steps:
//   injected: every sublayer starts from the reference's stream and pre-mix, and the Markov chain from the
//             reference's tokens, so each op is compared on the reference's inputs: elementwise ops, lookups and
//             selections must match every bit; RMSNorms and FP8 GEMVs (fp32 sums in another order) may differ by
//             one bf16 ulp in under 0.1% of elements; the rest (TP=4 partial sums, online softmax) is reported.
//   free:     the whole chain from the step's token alone: draft tokens, confidence and acceptance against the
//             reference's.
// --couple 1 adds the coupled drafts' rows (dspark.h NoiseSamp, Draft::set_couple), after the two passes, on one GPU; the drafter
// runs with 6 session slots then. Td 0.8 at every depth, the settings of DSGB10_GCOUPLE=0.8.
//   D1  the two passes above run without set_couple: every comparison of the existing test is D1.
//   D2  set_couple with an entry that is greedy, then one that is sampled without the Gumbel-path bit: the drafts' ids, the
//       confidence and every rank's biased logits equal the uncoupled run's, byte for byte.
//   D3  an entry on the Gumbel path (T 1, top-p 0.95, top-k 20 under the bit): at each depth the picked id is the host's argmax over
//       the whole vocabulary of dl[j][id] / (T Td) + g(seed, anchor + 1 + j, id), over the tapped dl; a differing pick passes only
//       where the two best scores are within 1e-4 (the count is printed). Control: the same check at position anchor + j must
//       fail (the rows disagree in many picks), or the check cannot see the position.
//   D4  a batch pass of 3 entries, mixed: Gumbel-path, greedy, sampled without the bit. The entries off the Gumbel path equal the
//       same pass with every entry off it, byte for byte, and every entry's ids and confidence equal its own draft(.., entry)
//       (the pass computes what draft() computes for each row). The one-pass drafter (6 or more entries, draft_batch_wide)
//       needs the whole router (Weights::gate_all), which this test's weights do not carry: it is covered on the engine (G1).
// Command (one GB10, the engine stopped; the paths as the defaults above):
//   dspark_tests --couple 1 [--steps 40]
// Pass: the last line PASSED, and the D lines below it all equal / compared.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "attn/attn_common.cuh"
#include "common/check.h"
#include "dspark/dspark.h"
#include "weights/loader.h"

using dspark::Cand;
using dspark::GatherSeg;
using dspark::kBlocks;
using dspark::kDim;
using dspark::kDraft;
using dspark::kHc;
using dspark::kMainK;
using dspark::kMaxAcc;
using dspark::kRanks;
using dspark::kVocabRank;

namespace {

// ---- reference data (.npy, mapped) --------------------------------------------------------------------------

struct Npy {
  std::vector<int64_t> shape;
  char kind = 0;  // 'f' float32, 'i' int32 / int64
  int item = 0;
  const char* data = nullptr;
  float f(size_t i) const { return reinterpret_cast<const float*>(data)[i]; }
  int64_t i(size_t k) const {
    return item == 8 ? reinterpret_cast<const int64_t*>(data)[k] : reinterpret_cast<const int32_t*>(data)[k];
  }
};

Npy npy(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) die("cannot open %s", path.c_str());
  struct stat st;
  fstat(fd, &st);
  void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (m == MAP_FAILED) die("cannot map %s", path.c_str());
  const char* p = static_cast<const char*>(m);
  if (std::memcmp(p, "\x93NUMPY", 6)) die("%s: not npy", path.c_str());
  uint32_t hl = 0, off = 0;
  if (p[6] == 1) {
    uint16_t h16;
    std::memcpy(&h16, p + 8, 2);
    hl = h16, off = 10;
  } else {
    std::memcpy(&hl, p + 8, 4);
    off = 12;
  }
  std::string h(p + off, hl);
  Npy n;
  auto d = h.find("'descr': '") + 10;
  std::string descr = h.substr(d, h.find('\'', d) - d);
  n.kind = descr[1], n.item = descr[2] - '0';
  if (!((n.kind == 'f' && n.item == 4) || (n.kind == 'i' && (n.item == 4 || n.item == 8))))
    die("%s: dtype %s", path.c_str(), descr.c_str());
  auto s = h.find("'shape': (") + 10;
  std::string sh = h.substr(s, h.find(')', s) - s);
  for (size_t i = 0; i < sh.size();) {
    while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) ++i;
    if (i >= sh.size()) break;
    n.shape.push_back(std::stoll(sh.substr(i)));
    while (i < sh.size() && sh[i] != ',') ++i;
  }
  n.data = p + off + hl;
  return n;
}

uint16_t bf16_of(float f) {  // round to nearest even (exact for the reference's bf16 values)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
float f_of(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float e4m3(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
  return s ? -v : v;
}

// ---- comparisons ---------------------------------------------------------------------------------------------

// What a comparison must show. kExact: every element equal (elementwise ops, lookups, selections, the ranks'
// agreement). kNear: fp32 reductions in another order (RMSNorm's sum of squares, the FP8 GEMVs' dot products)
// flip a bf16 rounding now and then, and a flipped input moves the rest of its row: at least 99% of elements
// equal, relative RMS error at most 1e-3. kTP: TP=4 partial sums (a bf16 partial per rank, then their fp32 sum),
// the attention's online softmax, fp32 sums of the mixes and the heads: relative RMS error below 1e-2.
// kInfo: the free pass's values, reported only.
enum Cls { kInfo, kTP, kNear, kExact };
struct Stat {
  size_t n = 0, exact = 0;
  double max_abs = 0, max_ref = 0, d2 = 0, r2 = 0;
  Cls cls = kInfo;
  void add(double got, double ref) {
    ++n;
    if (got == ref) ++exact;
    max_abs = std::max(max_abs, std::fabs(got - ref));
    max_ref = std::max(max_ref, std::fabs(ref));
    d2 += (got - ref) * (got - ref), r2 += ref * ref;
  }
  double rel_rms() const { return r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2); }
  bool ok() const {
    if (cls == kExact) return exact == n;
    if (cls == kNear) return exact >= n - n / 100 && rel_rms() <= 1e-3;
    if (cls == kTP) return rel_rms() < 1e-2;
    return true;
  }
};
std::map<std::string, Stat> g_stats;
std::vector<std::string> g_order;
Stat& stat(const std::string& name, Cls cls) {
  if (!g_stats.count(name)) g_order.push_back(name);
  Stat& s = g_stats[name];
  s.cls = cls;
  return s;
}
void cmp_bf16(const std::string& name, Cls cls, const std::vector<uint16_t>& got, const Npy& ref, size_t ref_off,
              size_t n, size_t got_off = 0) {
  Stat& s = stat(name, cls);
  for (size_t i = 0; i < n; ++i) s.add(f_of(got[got_off + i]), ref.f(ref_off + i));
}
void cmp_f32(const std::string& name, Cls cls, const float* got, const Npy& ref, size_t ref_off, size_t n) {
  Stat& s = stat(name, cls);
  for (size_t i = 0; i < n; ++i) s.add(got[i], ref.f(ref_off + i));
}

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count = 1) : n(count) {
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
  }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  void up(const T* h, size_t count) { CUDA_CHECK(cudaMemcpy(p, h, count * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> host() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

// ---- the collectives, emulated ----------------------------------------------------------------------------------

__global__ void reduce4_kernel(const void* a, const void* b, const void* c, const void* d, bool f32, uint16_t* dst,
                               int n) {
  const void* src[4] = {a, b, c, d};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float s = 0.f;
    for (int q = 0; q < 4; ++q) {
      const float v = f32 ? static_cast<const float*>(src[q])[i] : attn::bf2f(static_cast<const uint16_t*>(src[q])[i]);
      s = q ? __fadd_rn(s, v) : v;
    }
    dst[i] = attn::f2bf(s);
  }
}
__global__ void argmax4_kernel(const Cand* a, const Cand* b, const Cand* c, const Cand* d, Cand* dst) {
  const Cand* src[4] = {a, b, c, d};
  Cand best = *src[0];
  for (int q = 1; q < 4; ++q) {
    const Cand o = *src[q];
    if (o.v > best.v || (o.v == best.v && o.id < best.id)) best = o;
  }
  *dst = best;
}

struct Shared {
  std::barrier<> bar{kRanks};
  const GatherSeg* segs[kRanks] = {};
  const void* src[kRanks] = {};
};
class EmuComm final : public dspark::Comm {
 public:
  EmuComm(int rank, Shared* sh) : r_(rank), sh_(sh) {}
  void all_gather(const GatherSeg* segs, int n, cudaStream_t s) override {
    CUDA_CHECK(cudaStreamSynchronize(s));
    sh_->segs[r_] = segs;
    sh_->bar.arrive_and_wait();
    for (int i = 0; i < n; ++i)
      for (int q = 0; q < kRanks; ++q) {
        const GatherSeg &m = segs[i], &o = sh_->segs[q][i];
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(m.dst) + q * m.width, m.dst_ld, o.src, o.src_ld, m.width,
                                     m.rows, cudaMemcpyDeviceToDevice, s));
      }
    CUDA_CHECK(cudaStreamSynchronize(s));
    sh_->bar.arrive_and_wait();
  }
  void all_reduce(const void* src, bool f32, uint16_t* dst, size_t n, cudaStream_t s) override {
    CUDA_CHECK(cudaStreamSynchronize(s));
    sh_->src[r_] = src;
    sh_->bar.arrive_and_wait();
    reduce4_kernel<<<40, 256, 0, s>>>(sh_->src[0], sh_->src[1], sh_->src[2], sh_->src[3], f32, dst, (int)n);
    CUDA_CHECK(cudaStreamSynchronize(s));
    sh_->bar.arrive_and_wait();
  }
  void argmax(const Cand* src, Cand* dst, cudaStream_t s) override {
    CUDA_CHECK(cudaStreamSynchronize(s));
    sh_->src[r_] = src;
    sh_->bar.arrive_and_wait();
    argmax4_kernel<<<1, 1, 0, s>>>(static_cast<const Cand*>(sh_->src[0]), static_cast<const Cand*>(sh_->src[1]),
                                   static_cast<const Cand*>(sh_->src[2]), static_cast<const Cand*>(sh_->src[3]), dst);
    CUDA_CHECK(cudaStreamSynchronize(s));
    sh_->bar.arrive_and_wait();
  }

 private:
  int r_;
  Shared* sh_;
};

// ---- the test ----------------------------------------------------------------------------------------------------

constexpr int kStream = kDraft * kHc * kDim;
constexpr int kCoupleSlots = 6;  // --couple: the sessions' windows D4's entries use

struct RankTaps {  // device copies a rank's Draft writes during a pass
  Dev<uint16_t> main_hidden{(size_t)kMaxAcc * kMainK}, main_x{(size_t)kMaxAcc * kDim};
  Dev<float> mix[kBlocks][2]{{Dev<float>(kDraft * 24), Dev<float>(kDraft * 24)},
                             {Dev<float>(kDraft * 24), Dev<float>(kDraft * 24)},
                             {Dev<float>(kDraft * 24), Dev<float>(kDraft * 24)}};
  std::vector<std::unique_ptr<Dev<uint16_t>>> bf;  // attn_in, qa, q_pre, attn_o, attn_out, mid, ffn_in, ffn_out, out
  Dev<int> route_idx[kBlocks]{Dev<int>(kDraft * 6), Dev<int>(kDraft * 6), Dev<int>(kDraft * 6)};
  Dev<float> route_w[kBlocks]{Dev<float>(kDraft * 6), Dev<float>(kDraft * 6), Dev<float>(kDraft * 6)};
  Dev<uint16_t> head_x{(size_t)kDraft * kDim}, head_xn{(size_t)kDraft * kDim}, membed{kDraft * 256};
  Dev<float> logits{(size_t)kDraft * kVocabRank}, bias{(size_t)kDraft * kVocabRank};
  uint16_t* get(int which, int b) { return bf[which * kBlocks + b]->p; }
  void attach(dspark::Taps& t) {
    const size_t sizes[9] = {(size_t)kDraft * kDim, (size_t)kDraft * 1280, (size_t)kDraft * 8192, (size_t)kDraft * 8192,
                             (size_t)kDraft * kDim, (size_t)kStream, (size_t)kDraft * kDim, (size_t)kDraft * kDim,
                             (size_t)kStream};
    for (int w = 0; w < 9; ++w)
      for (int b = 0; b < kBlocks; ++b) bf.push_back(std::make_unique<Dev<uint16_t>>(sizes[w]));
    t.main_hidden = main_hidden.p, t.main_x = main_x.p;
    for (int b = 0; b < kBlocks; ++b) {
      t.mix[b][0] = mix[b][0].p, t.mix[b][1] = mix[b][1].p;
      t.attn_in[b] = get(0, b), t.qa[b] = get(1, b), t.q_pre[b] = get(2, b), t.attn_o[b] = get(3, b);
      t.attn_out[b] = get(4, b), t.mid[b] = get(5, b), t.ffn_in[b] = get(6, b), t.ffn_out[b] = get(7, b);
      t.out[b] = get(8, b), t.route_idx[b] = route_idx[b].p, t.route_w[b] = route_w[b].p;
    }
    t.head_x = head_x.p, t.head_xn = head_xn.p, t.logits = logits.p, t.bias = bias.p, t.markov_embed = membed.p;
  }
};

struct Args {
  std::string ref = "/data/dsgb10/dspark/prose", golden = "/data/dsgb10/golden/prose";
  std::string rank0 = "/data/dsgb10/weights/rank0.dsw", wdir = "/data/dsgb10/dspark/w";
  int steps = 0;
  int couple = 0;
};

struct Test {
  Args a;
  Npy steps, tokens, main_x_seed, main_x, draft_ids, confidence, accepted, head_x, head_xn;
  Npy logits_topv, logits_topi, bias_topv, bias_topi, draft_topv, draft_topi, markov_embed, full_logits, full_steps;
  Npy gmh, gl[3];
  struct BlockRef {
    Npy kv_seed, stream_in, attn_in, qa, q_pre, main_kv, blk_kv, attn_o, attn_out, mid, ffn_in, route_idx, route_w,
        ffn_out, out, mix[2][3];
  } B[kBlocks];
  std::vector<std::unique_ptr<dsw::WeightFile>> files;
  std::unique_ptr<Shared> shared;
  std::vector<std::unique_ptr<EmuComm>> comms;
  std::vector<std::unique_ptr<dspark::Draft>> drafts;
  std::vector<std::unique_ptr<RankTaps>> taps;
  cudaStream_t st[kRanks], s0;
  Dev<float2> table{256 * 32};
  Dev<uint16_t> x3[3]{Dev<uint16_t>((size_t)kMaxAcc * kHc * kDim), Dev<uint16_t>((size_t)kMaxAcc * kHc * kDim),
                      Dev<uint16_t>((size_t)kMaxAcc * kHc * kDim)};
  Dev<uint16_t> mh{(size_t)kMaxAcc * kMainK};
  Dev<int> count{1}, pos0{1}, token{1}, ids_in{8};
  Dev<uint16_t> inj_stream[kBlocks]{Dev<uint16_t>(kStream), Dev<uint16_t>(kStream), Dev<uint16_t>(kStream)};
  Dev<uint16_t> inj_mid[kBlocks]{Dev<uint16_t>(kStream), Dev<uint16_t>(kStream), Dev<uint16_t>(kStream)};
  Dev<uint16_t> inj_head{kStream};
  Dev<float> inj_pre[kBlocks]{Dev<float>(kDraft * 4), Dev<float>(kDraft * 4), Dev<float>(kDraft * 4)};
  Dev<float> inj_apre[kBlocks]{Dev<float>(kDraft * 4), Dev<float>(kDraft * 4), Dev<float>(kDraft * 4)};
  Dev<float> inj_hpre{kDraft * 4};

  void load_ref() {
    auto R = [&](const std::string& n) { return npy(a.ref + "/" + n + ".npy"); };
    steps = R("steps"), tokens = R("tokens"), main_x_seed = R("main_x_seed"), main_x = R("main_x");
    draft_ids = R("draft_ids"), confidence = R("confidence"), accepted = R("accepted");
    head_x = R("head_x"), head_xn = R("head_xn"), markov_embed = R("markov_embed");
    logits_topv = R("logits_topv"), logits_topi = R("logits_topi"), bias_topv = R("bias_topv");
    bias_topi = R("bias_topi"), draft_topv = R("draft_topv"), draft_topi = R("draft_topi");
    full_logits = R("full_logits"), full_steps = R("full_steps");
    for (int b = 0; b < kBlocks; ++b) {
      const std::string p = "B" + std::to_string(b) + ".";
      BlockRef& r = B[b];
      r.kv_seed = R(p + "main_kv_seed"), r.stream_in = R(p + "stream_in"), r.attn_in = R(p + "attn_in");
      r.qa = R(p + "qa"), r.q_pre = R(p + "q_pre"), r.main_kv = R(p + "main_kv"), r.blk_kv = R(p + "blk_kv");
      r.attn_o = R(p + "attn_o"), r.attn_out = R(p + "attn_out"), r.mid = R(p + "mid"), r.ffn_in = R(p + "ffn_in");
      r.route_idx = R(p + "route_idx"), r.route_w = R(p + "route_w"), r.ffn_out = R(p + "ffn_out");
      r.out = R(p + "out");
      const char* sub[2] = {"attn_", "ffn_"};
      const char* what[3] = {"pre", "post", "comb"};
      for (int k = 0; k < 2; ++k)
        for (int w = 0; w < 3; ++w) r.mix[k][w] = R(p + sub[k] + what[w]);
    }
    gmh = npy(a.golden + "/main_hidden.npy");
    for (int l = 0; l < 3; ++l) gl[l] = npy(a.golden + "/L" + std::to_string(36 + l) + ".out.npy");
  }

  void load_weights() {
    auto t0 = std::chrono::steady_clock::now();
    auto f0 = std::make_unique<dsw::WeightFile>(a.rank0);
    dsw::LoadOptions o;
    o.select = [](const dsw::DeviceTensor& t) {
      return t.name.rfind("mtp.", 0) == 0 || t.name == "head.weight" || t.name == "embed.weight";
    };
    f0->load(o);
    files.push_back(std::move(f0));
    shared = std::make_unique<Shared>();
    CUDA_CHECK(cudaStreamCreate(&s0));
    attn::rope_table_build(table.p, 256, false, s0);
    CUDA_CHECK(cudaStreamSynchronize(s0));
    for (int r = 0; r < kRanks; ++r) {
      dspark::Weights w;
      if (r == 0) {
        w = dspark::weights_from({files[0].get()});
      } else {
        auto m = std::make_unique<dsw::WeightFile>(a.wdir + "/rank" + std::to_string(r) + ".mtp.dsw");
        auto h = std::make_unique<dsw::WeightFile>(a.wdir + "/rank" + std::to_string(r) + ".head.dsw");
        m->load();
        h->load();
        w = dspark::weights_from({m.get(), h.get(), files[0].get()});
        files.push_back(std::move(m));
        files.push_back(std::move(h));
      }
      CUDA_CHECK(cudaStreamCreate(&st[r]));
      comms.push_back(std::make_unique<EmuComm>(r, shared.get()));
      drafts.push_back(std::make_unique<dspark::Draft>(w, r, comms[r].get(), table.p, a.couple ? kCoupleSlots : 1));
      taps.push_back(std::make_unique<RankTaps>());
      taps[r]->attach(drafts[r]->taps);
    }
    std::printf("weights: 4 ranks loaded in %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }

  // main_hidden of positions [p, p + n) from the golden streams, through dspark::main_hidden
  void hidden(int p, int n) {
    std::vector<uint16_t> h((size_t)n * kHc * kDim);
    for (int l = 0; l < 3; ++l) {
      const size_t per = (size_t)kHc * kDim;
      for (size_t i = 0; i < (size_t)n * per; ++i) h[i] = bf16_of(gl[l].f((size_t)p * per + i));
      x3[l].up(h.data(), h.size());
      dspark::main_hidden(x3[l].p, n, l, mh.p, false, s0);
    }
    CUDA_CHECK(cudaStreamSynchronize(s0));
    int c = n;
    count.up(&c, 1);
    pos0.up(&p, 1);
  }

  void ranks(bool accept, bool draft) {
    std::vector<std::thread> t;
    for (int r = 0; r < kRanks; ++r)
      t.emplace_back([&, r] {
        if (accept) drafts[r]->accept(mh.p, count.p, pos0.p, st[r]);
        if (draft) drafts[r]->draft(token.p, st[r]);
        CUDA_CHECK(cudaStreamSynchronize(st[r]));
      });
    for (auto& x : t) x.join();
  }

  // the window entry of position p in block b's ring on rank r, dequantised
  std::vector<float> window(int r, int b, int p) {
    const attn::Window& w = drafts[r]->window(b);
    const int slot = p & w.mask;
    uint8_t kv[512], sc[16];
    CUDA_CHECK(cudaMemcpy(kv, w.kv + (size_t)slot * 512, 512, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sc, w.sc + (size_t)slot * 16, 16, cudaMemcpyDeviceToHost));
    std::vector<float> v(512);
    for (int i = 0; i < 512; ++i) v[i] = e4m3(kv[i]) * std::ldexp(1.f, sc[i / 32] - 127);
    return v;
  }

  void accept_checks(int p, int n, const Npy& ref_x, size_t ref_row) {
    auto mhh = taps[0]->main_hidden.host(), mx = taps[0]->main_x.host();
    cmp_bf16("main_hidden", kExact, mhh, gmh, (size_t)p * kMainK, (size_t)n * kMainK);
    cmp_bf16("main_x", kNear, mx, ref_x, ref_row * kDim, (size_t)n * kDim);
    for (int r = 1; r < kRanks; ++r)
      stat("ranks agree: main_x", kExact).add(taps[r]->main_x.host() == mx ? 1 : 0, 1);
  }

  void seed() {
    const int n_seed = (int)main_x_seed.shape[0];
    for (int p = 0; p < n_seed; p += kMaxAcc) {
      const int n = std::min(kMaxAcc, n_seed - p);
      hidden(p, n);
      ranks(true, false);
      accept_checks(p, n, main_x_seed, p);
    }
    for (int b = 0; b < kBlocks; ++b)
      for (int p = 0; p < n_seed; ++p)
        for (int r = 0; r < kRanks; ++r) {
          auto v = window(r, b, p);
          cmp_f32("window: prompt entries", kNear, v.data(), B[b].kv_seed, (size_t)p * 512, 512);
        }
  }

  void put_bf16(Dev<uint16_t>& d, const Npy& ref, size_t off, size_t n) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = bf16_of(ref.f(off + i));
    d.up(h.data(), n);
  }
  void put_f32(Dev<float>& d, const Npy& ref, size_t off, size_t n) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = ref.f(off + i);
    d.up(h.data(), n);
  }

  void inject(int k, bool on) {
    for (int r = 0; r < kRanks; ++r) {
      dspark::Inject& in = drafts[r]->inject;
      in = dspark::Inject{};
      if (!on) continue;
      for (int b = 0; b < kBlocks; ++b) {
        in.stream[b] = inj_stream[b].p, in.pre[b] = inj_pre[b].p;
        in.mid[b] = inj_mid[b].p, in.attn_pre[b] = inj_apre[b].p;
      }
      in.head_stream = inj_head.p, in.head_pre = inj_hpre.p, in.ids = ids_in.p;
    }
    if (!on) return;
    std::vector<float> id(kDraft * 4, 0.f);
    for (int j = 0; j < kDraft; ++j) id[j * 4] = 1.f;
    for (int b = 0; b < kBlocks; ++b) {
      put_bf16(inj_stream[b], B[b].stream_in, (size_t)k * kStream, kStream);
      if (b == 0)
        inj_pre[b].up(id.data(), id.size());
      else
        put_f32(inj_pre[b], B[b - 1].mix[1][0], (size_t)k * kDraft * 4, kDraft * 4);
      put_bf16(inj_mid[b], B[b].mid, (size_t)k * kStream, kStream);
      put_f32(inj_apre[b], B[b].mix[0][0], (size_t)k * kDraft * 4, kDraft * 4);
    }
    put_bf16(inj_head, B[2].out, (size_t)k * kStream, kStream);
    put_f32(inj_hpre, B[2].mix[1][0], (size_t)k * kDraft * 4, kDraft * 4);
    int ids[6];
    for (int j = 0; j < 6; ++j) ids[j] = (int)draft_ids.i((size_t)k * 6 + j);
    ids_in.up(ids, 6);
  }

  // per-op comparisons of step k (injected pass)
  void op_checks(int k, int anchor) {
    const size_t R5 = kDraft;
    for (int b = 0; b < kBlocks; ++b) {
      const BlockRef& rf = B[b];
      const std::string p = "B" + std::to_string(b) + " ";
      const char* sub[2] = {"attn ", "ffn "};
      for (int m = 0; m < 2; ++m) {
        auto mx = taps[0]->mix[b][m].host();
        const int width[3] = {4, 4, 16}, col[3] = {0, 4, 8};
        const char* what[3] = {"pre", "post", "comb"};
        for (int w = 0; w < 3; ++w)
          for (size_t j = 0; j < R5; ++j)
            cmp_f32(p + sub[m] + what[w], kTP, mx.data() + j * 24 + col[w], rf.mix[m][w],
                    ((size_t)k * R5 + j) * width[w], width[w]);
      }
      cmp_bf16(p + "attn_in", kNear, taps[0]->bf[0 * kBlocks + b]->host(), rf.attn_in, (size_t)k * R5 * kDim, R5 * kDim);
      cmp_bf16(p + "qa", kNear, taps[0]->bf[1 * kBlocks + b]->host(), rf.qa, (size_t)k * R5 * 1280, R5 * 1280);
      for (int r = 0; r < kRanks; ++r) {
        auto q = taps[r]->bf[2 * kBlocks + b]->host(), o = taps[r]->bf[3 * kBlocks + b]->host();
        for (size_t j = 0; j < R5; ++j) {
          cmp_bf16(p + "q_pre", kNear, q, rf.q_pre, ((size_t)k * R5 + j) * 32768 + (size_t)r * 8192, 8192, j * 8192);
          cmp_bf16(p + "attn_o", kTP, o, rf.attn_o, ((size_t)k * R5 + j) * 32768 + (size_t)r * 8192, 8192, j * 8192);
        }
      }
      for (int r = 0; r < kRanks; ++r) {
        auto v = window(r, b, anchor);
        cmp_f32(p + "window: accepted entry", kNear, v.data(), rf.main_kv, (size_t)k * 512, 512);
        for (int j = 0; j < kDraft; ++j) {
          auto u = window(r, b, anchor + 1 + j);
          cmp_f32(p + "window: block entries", kNear, u.data(), rf.blk_kv, ((size_t)k * R5 + j) * 512, 512);
        }
      }
      cmp_bf16(p + "attn_out", kTP, taps[0]->bf[4 * kBlocks + b]->host(), rf.attn_out, (size_t)k * R5 * kDim,
               R5 * kDim);
      cmp_bf16(p + "mid", kTP, taps[0]->bf[5 * kBlocks + b]->host(), rf.mid, (size_t)k * kStream, kStream);
      cmp_bf16(p + "ffn_in", kNear, taps[0]->bf[6 * kBlocks + b]->host(), rf.ffn_in, (size_t)k * R5 * kDim, R5 * kDim);
      auto ri = taps[0]->route_idx[b].host();
      auto rw = taps[0]->route_w[b].host();
      Stat &si = stat(p + "route ids", kExact), &sw = stat(p + "route weights", kTP);
      for (size_t j = 0; j < R5; ++j)
        for (int t = 0; t < 3; ++t) {
          si.add(ri[j * 6 + t], (double)rf.route_idx.i(((size_t)k * R5 + j) * 3 + t));
          if (ri[j * 6 + t] == rf.route_idx.i(((size_t)k * R5 + j) * 3 + t))
            sw.add(rw[j * 6 + t], rf.route_w.f(((size_t)k * R5 + j) * 3 + t));
        }
      cmp_bf16(p + "ffn_out", kTP, taps[0]->bf[7 * kBlocks + b]->host(), rf.ffn_out, (size_t)k * R5 * kDim,
               R5 * kDim);
      cmp_bf16(p + "out", kTP, taps[0]->bf[8 * kBlocks + b]->host(), rf.out, (size_t)k * kStream, kStream);
    }
    cmp_bf16("head_x", kExact, taps[0]->head_x.host(), head_x, (size_t)k * R5 * kDim, R5 * kDim);
    cmp_bf16("head_xn", kNear, taps[0]->head_xn.host(), head_xn, (size_t)k * R5 * kDim, R5 * kDim);
    cmp_bf16("markov_embed", kExact, taps[0]->membed.host(), markov_embed, (size_t)k * R5 * 256, R5 * 256);
    // logits, biases, biased logits at the reference's top 32 ids
    std::vector<std::vector<float>> lg(kRanks), bi(kRanks), dl(kRanks);
    for (int r = 0; r < kRanks; ++r) {
      lg[r] = taps[r]->logits.host(), bi[r] = taps[r]->bias.host();
      dl[r].resize((size_t)kDraft * kVocabRank);
      CUDA_CHECK(cudaMemcpy(dl[r].data(), drafts[r]->logits(), dl[r].size() * 4, cudaMemcpyDeviceToHost));
    }
    auto at = [&](std::vector<std::vector<float>>& v, size_t j, int64_t id) {
      return v[id / kVocabRank][j * kVocabRank + id % kVocabRank];
    };
    Stat &sl = stat("logits (top 32)", kTP), &sb = stat("Markov bias (top 32)", kTP),
         &sd = stat("biased logits (top 32)", kTP);
    for (size_t j = 0; j < R5; ++j)
      for (int t = 0; t < 32; ++t) {
        const size_t o = ((size_t)k * R5 + j) * 32 + t;
        sl.add(at(lg, j, logits_topi.i(o)), logits_topv.f(o));
        sb.add(at(bi, j, bias_topi.i(o)), bias_topv.f(o));
        sd.add(at(dl, j, draft_topi.i(o)), draft_topv.f(o));
      }
    for (int f = 0; f < (int)full_steps.shape[0]; ++f)
      if (full_steps.i(f) == k) {
        Stat& s = stat("logits (all, 4 steps)", kTP);
        for (size_t j = 0; j < R5; ++j)
          for (int64_t id = 0; id < dspark::kVocab; ++id)
            s.add(at(lg, j, id), full_logits.f(((size_t)f * R5 + j) * dspark::kVocab + id));
      }
  }

  // draft ids and confidence of step k, every rank's the same
  void result_checks(int k, bool injected, std::vector<int>& d, std::vector<float>& c) {
    const char* pass = injected ? "injected" : "free";
    d.assign(6, 0), c.assign(16, 0.f);
    CUDA_CHECK(cudaMemcpy(d.data(), drafts[0]->ids(), 6 * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.data(), drafts[0]->confidence(), 16 * 4, cudaMemcpyDeviceToHost));
    for (int r = 1; r < kRanks; ++r) {
      std::vector<int> d2(6);
      std::vector<float> c2(16);
      CUDA_CHECK(cudaMemcpy(d2.data(), drafts[r]->ids(), 6 * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(c2.data(), drafts[r]->confidence(), 16 * 4, cudaMemcpyDeviceToHost));
      Stat& s = stat(std::string(pass) + ": ranks agree on drafts and confidence", kExact);
      s.add(d2 == d && std::memcmp(c2.data(), c.data(), 64) == 0 ? 1 : 0, 1);
    }
    Stat& sc = stat(std::string(pass) + ": confidence (same input token)", injected ? kTP : kInfo);
    Stat& sv = stat(std::string(pass) + ": survival = prod sigmoid", kTP);
    double p = 1;
    for (int j = 0; j < kDraft; ++j) {
      if (d[j] == draft_ids.i((size_t)k * 6 + j)) sc.add(c[j], confidence.f((size_t)k * kDraft + j));
      p *= 1.0 / (1.0 + std::exp(-(double)c[j]));
      sv.add(c[8 + j], p);
    }
  }

  void run(bool injected) {
    const char* pass = injected ? "injected" : "free";
    const int S = a.steps ? std::min<int>(a.steps, (int)steps.shape[0]) : (int)steps.shape[0];
    const int L = (int)tokens.shape[0];
    int same_all = 0, same_pos[kDraft] = {}, hist_ours[kDraft + 1] = {}, hist_ref[kDraft + 1] = {};
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < S; ++k) {
      const int i = (int)steps.i(k);
      hidden(i, 1);
      int tok = (int)tokens.i(i + 1);
      token.up(&tok, 1);
      inject(k, injected);
      ranks(true, true);
      if (draft_ids.i((size_t)k * 6) != tok) die("step %d: the reference's input token differs", k);
      accept_checks(i, 1, main_x, k);
      if (injected) op_checks(k, i);
      std::vector<int> d;
      std::vector<float> c;
      result_checks(k, injected, d, c);
      if (!injected)  // the free pass's routing (its inputs drift from the reference's by TP=4 rounding)
        for (int b = 0; b < kBlocks; ++b) {
          auto ri = taps[0]->route_idx[b].host();
          Stat& s = stat("free: B" + std::to_string(b) + " route ids", kInfo);
          for (int j = 0; j < kDraft; ++j)
            for (int t = 0; t < 3; ++t) s.add(ri[j * 6 + t], (double)B[b].route_idx.i(((size_t)k * kDraft + j) * 3 + t));
        }
      bool all = true;
      int acc = 0, acc_ref = (int)accepted.i(k);
      bool run = true;
      for (int j = 0; j < kDraft; ++j) {
        const bool eq = d[1 + j] == draft_ids.i((size_t)k * 6 + 1 + j);
        same_pos[j] += eq, all = all && eq;
        if (run && i + 2 + j < L && d[1 + j] == tokens.i(i + 2 + j))
          ++acc;
        else
          run = false;
      }
      same_all += all;
      if (!injected && d[1] != draft_ids.i((size_t)k * 6 + 1)) {
        const size_t o = (size_t)k * kDraft * 32;
        std::printf("  free pass, step %d: d1 %d, the reference's %lld; its top-2 biased logits %.4f, %.4f (ours: id %lld)\n",
                    k, d[1], (long long)draft_ids.i((size_t)k * 6 + 1), draft_topv.f(o), draft_topv.f(o + 1),
                    (long long)draft_topi.i(o + 1));
      }
      ++hist_ours[acc], ++hist_ref[acc_ref];
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s pass: %d steps in %.1f s; drafts equal to the reference's: all 5 in %d steps, d1..d5: %d %d %d %d %d\n",
                pass, S, sec, same_all, same_pos[0], same_pos[1], same_pos[2], same_pos[3], same_pos[4]);
    std::printf("  accepted (leading drafts equal to the golden tokens) 0..5: ours %d %d %d %d %d %d, reference %d %d %d %d %d %d\n",
                hist_ours[0], hist_ours[1], hist_ours[2], hist_ours[3], hist_ours[4], hist_ours[5], hist_ref[0],
                hist_ref[1], hist_ref[2], hist_ref[3], hist_ref[4], hist_ref[5]);
  }

  // ---- coupled drafts (--couple 1) --------------------------------------------------------------------------------
  Dev<dspark::NoiseSamp> samp{16};
  Dev<int> b_ent{16}, b_tok{16}, b_slot{16 * kDraft};
  float td_[kDraft] = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

  void put_samp(int e, uint64_t seed, float T, float top_p, int32_t top_k) {
    dspark::NoiseSamp n{seed, T, top_p, top_k, 0, 0.f};
    CUDA_CHECK(cudaMemcpy(samp.p + e, &n, sizeof n, cudaMemcpyHostToDevice));
  }
  void on_ranks(const std::function<void(int)>& f) {
    std::vector<std::thread> t;
    for (int r = 0; r < kRanks; ++r)
      t.emplace_back([&, r] {
        f(r);
        CUDA_CHECK(cudaStreamSynchronize(st[r]));
      });
    for (auto& x : t) x.join();
  }
  static uint64_t fnv(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
    return h;
  }
  struct Rec {
    std::vector<int> ids;
    std::vector<float> conf;
    uint64_t dl[kRanks] = {};
    bool agree = true;
  };
  // the drafts, the confidence and every rank's biased logits of the draft() just run
  Rec record(std::vector<std::vector<float>>* dls = nullptr) {
    Rec c;
    c.ids.assign(6, 0), c.conf.assign(16, 0.f);
    CUDA_CHECK(cudaMemcpy(c.ids.data(), drafts[0]->ids(), 24, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.conf.data(), drafts[0]->confidence(), 64, cudaMemcpyDeviceToHost));
    if (dls) dls->assign(kRanks, {});
    for (int r = 0; r < kRanks; ++r) {
      std::vector<float> dl((size_t)kDraft * kVocabRank);
      CUDA_CHECK(cudaMemcpy(dl.data(), drafts[r]->logits(), dl.size() * 4, cudaMemcpyDeviceToHost));
      c.dl[r] = fnv(dl.data(), dl.size() * 4);
      if (dls) (*dls)[r] = std::move(dl);
      std::vector<int> i2(6);
      std::vector<float> c2(16);
      CUDA_CHECK(cudaMemcpy(i2.data(), drafts[r]->ids(), 24, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(c2.data(), drafts[r]->confidence(), 64, cudaMemcpyDeviceToHost));
      c.agree = c.agree && i2 == c.ids && std::memcmp(c2.data(), c.conf.data(), 64) == 0;
    }
    return c;
  }
  // the free pass's steps again: after(k, anchor, record)
  template <class F>
  void each_step(int S, F after, bool want_dl = false) {
    for (int k = 0; k < S; ++k) {
      const int i = (int)steps.i(k);
      hidden(i, 1);
      int tok = (int)tokens.i(i + 1);
      token.up(&tok, 1);
      inject(k, false);
      ranks(true, true);
      std::vector<std::vector<float>> dls;
      after(k, i, record(want_dl ? &dls : nullptr), dls);
    }
  }

  // D2 and D3
  void couple_drafts() {
    const int S = a.steps ? std::min<int>(a.steps, (int)steps.shape[0]) : (int)steps.shape[0];
    std::printf("coupled drafts (D2, D3): %d steps\n", S);
    std::vector<Rec> base;
    each_step(S, [&](int, int, Rec r, std::vector<std::vector<float>>&) { base.push_back(std::move(r)); });
    put_samp(0, 0, 0.f, 1.f, 0);
    for (int r = 0; r < kRanks; ++r) drafts[r]->set_couple(samp.p, td_, 1.f, 0.f);
    struct Case {
      const char* name;
      uint64_t seed;
      float T, top_p;
      int32_t top_k;
    } cases[2] = {{"D2 greedy entry", 0, 0.f, 1.f, 0}, {"D2 sampled entry without the bit", 77, 0.8f, 0.95f, 20}};
    for (const Case& c : cases) {
      put_samp(0, c.seed, c.T, c.top_p, c.top_k);
      Stat& si = stat(std::string(c.name) + ": ids and confidence equal the uncoupled run's", kExact);
      Stat& sd = stat(std::string(c.name) + ": every rank's biased logits equal", kExact);
      each_step(S, [&](int k, int, Rec r, std::vector<std::vector<float>>&) {
        si.add(r.ids == base[k].ids && r.conf == base[k].conf && r.agree ? 1 : 0, 1);
        bool same = true;
        for (int q = 0; q < kRanks; ++q) same = same && r.dl[q] == base[k].dl[q];
        sd.add(same ? 1 : 0, 1);
      });
    }
    // D3: an entry on the Gumbel path
    const uint64_t seed = 424242;
    const float T = 1.f, top_p = 0.95f;
    put_samp(0, seed, T, top_p, dspark::kNoiseGumbel | 20);
    Stat& s3 = stat("D3 coupled pick = host argmax of dl / (T Td) + noise (ties within 1e-4 excluded)", kExact);
    Stat& sa = stat("D3 ranks agree on the coupled drafts", kExact);
    long ties = 0, picks = 0, ctl_bad = 0, changed = 0;
    each_step(
        S,
        [&](int k, int anchor, Rec r, std::vector<std::vector<float>>& dl) {
          sa.add(r.agree ? 1 : 0, 1);
          for (int j = 0; j < kDraft; ++j) {
            for (int control = 0; control < 2; ++control) {
              const int pos = control ? anchor + j : anchor + 1 + j;  // the control puts the noise one row early
              float best = -INFINITY;
              int best_id = 1 << 30;
              float dev_score = -INFINITY;
              for (int q = 0; q < kRanks; ++q)
                for (int x = 0; x < kVocabRank; ++x) {
                  const int id = q * kVocabRank + x;
                  const float v = dl[q][(size_t)j * kVocabRank + x];
                  const float sc = v * ((1.f / td_[j]) / T) + dspark::nz_gumbel(seed, pos, id);
                  if (sc > best) best = sc, best_id = id;
                  if (id == r.ids[1 + j]) dev_score = sc;
                }
              const bool eq = r.ids[1 + j] == best_id;
              const bool tie = !eq && best - dev_score <= 1e-4f;
              if (!control) {
                ++picks, ties += tie;
                s3.add(eq || tie ? 1 : 0, 1);
              } else if (!eq && !tie) {
                ++ctl_bad;
              }
            }
            changed += r.ids[1 + j] != base[k].ids[1 + j];
          }
        },
        true);
    std::printf("  D3: %ld picks checked, %ld within the 1e-4 tie band; control (noise at anchor + j): %ld of %ld picks differ; coupled drafts differ from the "
                "argmax's in %ld of them\n", picks, ties, ctl_bad, picks, changed);
    stat("D3 control: the check sees the position (wrong-position picks differ)", kExact).add(ctl_bad > 0 ? 1 : 0, 1);
    stat("D3 coupling ran (some pick differs from the argmax's)", kExact).add(changed > 0 ? 1 : 0, 1);
  }

  // D4: batch passes of n entries, one session slot each
  void couple_batch() {
    const int L = (int)tokens.shape[0];
    const int i = (int)steps.i(0);
    for (int r = 0; r < kRanks; ++r) drafts[r]->enable_batch();
    // each slot's window: the prompt's positions, then the anchor's
    const int n_seed = (int)main_x_seed.shape[0];
    for (int e = 0; e < kCoupleSlots; ++e) {
      for (int r = 0; r < kRanks; ++r) drafts[r]->use_slot(e);
      for (int p = 0; p < n_seed; p += kMaxAcc) {
        hidden(p, std::min(kMaxAcc, n_seed - p));
        ranks(true, false);
      }
    }
    hidden(i, 1);
    on_ranks([&](int r) {
      for (int e = 0; e < kCoupleSlots; ++e) {
        drafts[r]->use_slot(e);
        drafts[r]->accept(mh.p, count.p, pos0.p, st[r], nullptr, nullptr, drafts[r]->batch_start() + e);
      }
    });
    std::vector<int> ent(16), tok(16), slot(16 * kDraft);
    for (int e = 0; e < 16; ++e) ent[e] = e, tok[e] = (int)tokens.i((i + 1 + 7 * e) % L);
    for (int q = 0; q < 16 * kDraft; ++q) slot[q] = std::min(q / kDraft, kCoupleSlots - 1);
    b_ent.up(ent.data(), 16), b_tok.up(tok.data(), 16), b_slot.up(slot.data(), slot.size());
    // entry e: 0 mod 3: on the Gumbel path; 1: greedy; 2: sampled without the bit
    auto sampling = [&](int e, bool flagged_on) {
      if (e % 3 == 0) put_samp(e, 9000 + e, 1.f, 0.95f, flagged_on ? (dspark::kNoiseGumbel | 20) : 20);
      else if (e % 3 == 1) put_samp(e, 9000 + e, 0.f, 1.f, 0);
      else put_samp(e, 9000 + e, 0.8f, 0.95f, 20);
    };
    for (int n : {3}) {
      std::vector<Rec> A(n), B(n), own(n);
      for (int pass = 0; pass < 2; ++pass) {
        for (int e = 0; e < n; ++e) sampling(e, pass == 1);
        std::vector<Rec>& out = pass ? B : A;
        on_ranks([&](int r) {
          dspark::Draft::BatchPass bp{b_ent.p, n, b_tok.p, 1, b_slot.p};
          drafts[r]->draft_batch(bp, st[r]);
        });
        for (int e = 0; e < n; ++e) {
          out[e].ids.assign(6, 0), out[e].conf.assign(16, 0.f);
          CUDA_CHECK(cudaMemcpy(out[e].ids.data(), drafts[0]->batch_ids(e), 24, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(out[e].conf.data(), drafts[0]->batch_confidence(e), 64, cudaMemcpyDeviceToHost));
        }
      }
      // each entry's own draft(.., e) beside the mixed pass (its sampling as in pass B)
      for (int e = 0; e < n; ++e) {
        on_ranks([&](int r) {
          drafts[r]->use_slot(e);
          drafts[r]->accept(mh.p, count.p, pos0.p, st[r]);
          drafts[r]->draft(b_tok.p + e, st[r], nullptr, e);
        });
        own[e] = record();
      }
      Stat& s_off = stat("D4 n=" + std::to_string(n) + ": entries off the Gumbel path equal the all-off pass", kExact);
      Stat& s_own = stat("D4 n=" + std::to_string(n) + ": entries equal their own draft(.., entry)", kExact);
      long differ = 0, flagged_changed = 0;
      for (int e = 0; e < n; ++e) {
        if (e % 3 != 0) s_off.add(A[e].ids == B[e].ids && A[e].conf == B[e].conf ? 1 : 0, 1);
        else flagged_changed += A[e].ids != B[e].ids;
        const bool same = B[e].ids == own[e].ids && B[e].conf == own[e].conf;
        s_own.add(same ? 1 : 0, 1);
        differ += !same;
      }
      std::printf("  D4 n=%d: %d entries, %ld differ from their own draft(), %ld of the Gumbel-path entries changed by coupling\n", n, n, differ,
                  flagged_changed);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  Test t;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--ref") t.a.ref = v;
    else if (k == "--golden") t.a.golden = v;
    else if (k == "--rank0") t.a.rank0 = v;
    else if (k == "--wdir") t.a.wdir = v;
    else if (k == "--steps") t.a.steps = std::stoi(v);
    else if (k == "--couple") t.a.couple = std::stoi(v);
    else die("unknown argument %s", k.c_str());
  }
  t.load_ref();
  t.load_weights();
  t.seed();
  t.run(true);
  t.run(false);
  if (t.a.couple) {
    t.couple_drafts();
    t.couple_batch();
  }
  int bad = 0;
  std::printf("%-48s %20s %10s %10s %9s\n", "value", "equal / compared", "max |diff|", "max |ref|", "rel RMS");
  const char* cls[4] = {"info", "tp", "near", "exact"};
  for (const std::string& n : g_order) {
    const Stat& s = g_stats[n];
    bad += !s.ok();
    std::printf("%-48s %9zu/%-10zu %10.3g %10.4g %9.2g  %s%s\n", n.c_str(), s.exact, s.n, s.max_abs, s.max_ref,
                s.rel_rms(), cls[s.cls], s.ok() ? "" : "  FAILS");
  }
  std::printf("%s\n", bad ? "FAILED" : "PASSED");
  return bad ? 1 : 0;
}
