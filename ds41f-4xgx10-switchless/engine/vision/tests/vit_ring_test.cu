// A rank's own run of the split image encoder over the real fabric: vit::RankRun on every rank, each gather the in-place
// comm::allgather_multi of four equal blocks. Every rank runs the same command (engine/comm/run4-vit.sh does that);
// rank 0's output is the report.
//
// What it proves, by image, for every image of the set:
//   correct   each rank's span rows (every rank writes the whole span) equal, in bytes, the unsplit run's on rank 0
//             (vit::run_unsplit on the whole weights: the checkpoint under --model, or the synthetic set under
//             --synthetic): the phase machine, the pixel gather (rank 0 alone holds the image), the in-place
//             all-gather at the encoder's block sizes (n * 512, n * 1408 and m * 2560 bytes, chunked above the ring's
//             cap) and the weights files together. With --blocks k (default 13; 0: off) every image is also run as
//             blocks [0, k) then [k, 32) by two RankRuns, every buffer but x and the rotary table poisoned between.
//   control   --wrong swap-shard: ranks 1 and 2 exchange block 17's slices (over the ring's host sockets) and every
//             rank's span must then differ from the unsplit run's; swapped back, the bytes are equal again.
//   time      an image's time on rank 0's stream, with the part of it inside the gathers, at 1,521 / 3,025 / 5,625 /
//             8,649 / 9,189 patches: one uncounted run, then --repeats (5) runs, ranks started together by a host
//             barrier; median, min and max, and each rank's median. "in gathers" is the GPU time from before each
//             gather's launch to after it, summed: it includes the wait for the slowest rank's compute before it.
//
// usage: vit_ring_test [--fleet FILE] [--rank r] [--weights FMT] [--model DIR | --synthetic SEED [--layers N]]
//        [--blocks k] [--wrong swap-shard] [--raw FILE:HxW ...] [--repeats N] [--cap bytes] [--direct d] [tests...]
//        [--only control,time]   tests: correct control time (default: all three; --only names a comma list)
//   --weights FMT  this rank's vision weight file, "{rank}" replaced by the rank and a leading "~/" by $HOME
//                  (default ~/dsgb10/weights/vision.rank{rank}.dsw). The four files' set digest is compared.
//   --model DIR    the checkpoint's directory (rank 0 reads it for the unsplit reference; the others ignore it)
//   --synthetic S  no files: every rank cuts its quarter from make_synthetic(S); --layers N shortens the tower
//   --raw F:HxW    one more image: F holds the padded u8 pixels [H*14][W*14][3] (588 * H * W bytes), rank 0 reads it
//                  (the reference run's images are .png; convert them with the reference's image processor first)
//   --cap / --direct  the ring's per-rank collective size (default 4 MiB, the engine's) and its direct mode (default 2,
//                  as the engine runs the fleet)
// Run it with the engine stopped on all four Sparks (it takes the ring's port and the whole GPU; about 0.5 GiB of device
// memory a rank, 2 GiB more on rank 0 for the reference). Time (estimated, not measured): the load of the checkpoint's tower and
// the ring's start dominate; about 3 to 5 minutes. Exit status 0 only when every check passed on every rank.
#include <cuda_bf16.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "comm/collectives.h"
#include "comm/ring.h"
#include "common/check.h"
#include "vision/vit.h"
#include "vision/vit_device.h"
#include "vision/vit_rank.h"
#include "host_gather.h"

using namespace vit;

namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t fnv(const void* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
  return h;
}

uint64_t splitmix(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::string expand(std::string s, int rank) {
  const std::string tag = "{rank}";
  for (size_t p; (p = s.find(tag)) != std::string::npos;) s.replace(p, tag.size(), std::to_string(rank));
  if (s.rfind("~/", 0) == 0) {
    const char* h = std::getenv("HOME");
    if (!h) die("--weights starts with ~/ and HOME is not set");
    s = std::string(h) + s.substr(1);
  }
  return s;
}

int g_fails = 0;
comm::Ring* g_ring = nullptr;
// A check every rank reaches the same verdict on (it is made from gathered data); rank 0 prints it.
void expect(bool ok, const std::string& what) {
  if (!ok) g_fails++;
  if (g_ring && g_ring->rank() != 0) return;
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  std::fflush(stdout);
}

struct Case {
  std::string name;
  int nh, nw;
  std::vector<uint8_t> px;  // rank 0 only: u8 [nh*14][nw*14][3]
  int patches() const { return nh * nw; }
};

// A deterministic image: noise whose amplitude varies along the rows of 29 pixels, so no two patches alike.
std::vector<uint8_t> gen_image(int nh, int nw, uint64_t seed) {
  std::vector<uint8_t> px((size_t)nh * 14 * nw * 14 * 3);
  uint64_t s = seed * 0x2545F4914F6CDD1Dull + 1;
  for (size_t i = 0; i < px.size(); i++) {
    const double u = (double)(int64_t)(splitmix(s) >> 11) / (double)(1ull << 52) - 1.0;  // [-1, 1)
    px[i] = (uint8_t)(128 + 127 * u * (0.4 + 0.6 * ((i / 3) % 29) / 29.0));
  }
  return px;
}

std::vector<uint8_t> read_raw(const std::string& path, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) die("--raw: cannot open %s", path.c_str());
  std::vector<uint8_t> v(bytes);
  const size_t got = std::fread(v.data(), 1, bytes, f);
  std::fseek(f, 0, SEEK_END);
  const long total = std::ftell(f);
  std::fclose(f);
  if (got != bytes || (size_t)total != bytes) die("--raw: %s holds %ld bytes, %zu expected (588 * H * W)", path.c_str(), total, bytes);
  return v;
}

template <class T>
T* to_dev(const std::vector<T>& h) {
  void* p;
  CUDA_CHECK(cudaMalloc(&p, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return (T*)p;
}

void sync_or_die(comm::Ring& ring, cudaStream_t st, double timeout_s, const char* what) {
  const double t0 = now_s();
  for (;;) {
    const cudaError_t e = cudaStreamQuery(st);
    if (e == cudaSuccess) return;
    if (e != cudaErrorNotReady) die("%s: %s", what, cudaGetErrorString(e));
    if (now_s() - t0 > timeout_s) {
      std::fprintf(stderr, "rank %d: %s timed out\n%s", ring.rank(), what, ring.describe().c_str());
      std::fflush(stderr);
      _exit(2);
    }
    usleep(50);
  }
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Run {
  comm::Ring& ring;
  const DevWeights& w;  // this rank's quarter
  cudaStream_t st;
  void* work;           // rank_work_bytes of the largest image
  bf16* span;           // rows of the largest image
  std::vector<cudaEvent_t> ev;  // two per gather and one at each end
};

struct Timing {
  double wall_ms = 0;   // host: begin to the stream's end
  double gpu_ms = 0;    // the stream: first to last event
  double gather_ms = 0; // the stream: inside the gathers, summed
  int gathers = 0;
};

// One range of blocks of one image: the RankRun, each gather the in-place all-gather. Returns when the stream is done.
void run_range(Run& R, const Case& c, const uint8_t* dpx, int b0, int b1, bool timed, Timing* t) {
  const int rank = R.ring.rank();
  RankRunConfig cfg;
  cfg.w = &R.w, cfg.rank = rank, cfg.grid = make_grid(c.nh, c.nw), cfg.work = R.work;
  cfg.pixels = rank == 0 ? dpx : nullptr;
  cfg.span = R.span, cfg.st = R.st, cfg.b0 = b0, cfg.b1 = b1;
  RankRun rr(cfg);
  GatherPoint g;
  size_t k = 0;
  const double t0 = now_s();
  if (timed) CUDA_CHECK(cudaEventRecord(R.ev[k++], R.st));
  int n = 0;
  while (rr.step(&g)) {
    const comm::AgSeg seg{(const char*)g.buf + (size_t)rank * g.block_bytes, g.buf, g.block_bytes};
    if (timed) {
      if (k + 2 >= R.ev.size()) die("vit_ring_test: out of timing events");
      CUDA_CHECK(cudaEventRecord(R.ev[k++], R.st));
    }
    comm::allgather_multi(R.ring.dev(), &seg, 1, R.st);
    if (timed) CUDA_CHECK(cudaEventRecord(R.ev[k++], R.st));
    n++;
  }
  if (timed) CUDA_CHECK(cudaEventRecord(R.ev[k++], R.st));
  sync_or_die(R.ring, R.st, 120, "an image's run");
  if (!timed) return;
  t->wall_ms = (now_s() - t0) * 1e3;
  t->gathers = n;
  t->gather_ms = 0;
  for (size_t j = 1; j + 1 < k; j += 2) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, R.ev[j], R.ev[j + 1]));
    t->gather_ms += ms;
  }
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, R.ev[0], R.ev[k - 1]));
  t->gpu_ms = ms;
}

std::vector<uint8_t> span_to_host(Run& R, const Case& c) {
  const Grid gr = make_grid(c.nh, c.nw);
  std::vector<uint8_t> h((size_t)gr.rows() * kLm * 2);
  CUDA_CHECK(cudaMemcpy(h.data(), R.span, h.size(), cudaMemcpyDeviceToHost));
  return h;
}

struct Verdict {
  uint64_t mine, ref;  // this rank's span digest; rank 0's reference digest (the others' is not used)
};

// Every rank's span digest against rank 0's reference digest. All four ranks reach the same verdict.
// Returns {all equal, none equal}.
std::pair<bool, bool> compare(comm::Ring& ring, const std::vector<uint8_t>& span, const std::vector<uint8_t>* ref,
                              std::string* detail) {
  Verdict v{fnv(span.data(), span.size()), ref ? fnv(ref->data(), ref->size()) : 0}, all[4];
  ring.allgather_host(&v, all, sizeof v);
  int equal = 0;
  for (int r = 0; r < 4; r++) equal += all[r].mine == all[0].ref;
  if (detail) {
    *detail = "ranks equal to the reference:";
    for (int r = 0; r < 4; r++) *detail += all[r].mine == all[0].ref ? " " + std::to_string(r) : "";
    if (ref && span.size() == ref->size()) {  // rank 0 names the first differing byte
      size_t d = 0;
      while (d < span.size() && span[d] == (*ref)[d]) d++;
      if (d < span.size() && ring.rank() == 0) *detail += "; rank 0's first differing byte " + std::to_string(d) + " (row " + std::to_string(d / (kLm * 2)) + ")";
    }
  }
  return {equal == 4, equal == 0};
}

// Block 17's slices of ranks 1 and 2 exchanged over the ring's host sockets, in pieces under their size limit: the swap-shard control (vit_test's
// Wrong::SwapShard), made with the rank files alone. Done twice, it puts them back. Norms are whole on every rank
// and not exchanged.
void swap_shard(comm::Ring& ring, const DevWeights& w) {
  const DevBlock& b = w.blocks[17 % w.blocks.size()];
  struct T {
    bf16* p;
    size_t elems;
  } ts[] = {{b.wqkv, 768ull * kDim}, {b.bqkv, 768}, {b.wo, 256ull * kDim}, {b.bo, 256},
            {b.w1, 1408ull * kDim},  {b.w2, 256ull * kInter}};
  size_t total = 0;
  for (const T& t : ts) total += t.elems * 2;
  std::vector<uint8_t> mine(total), all(4 * total);
  size_t o = 0;
  for (const T& t : ts) {
    CUDA_CHECK(cudaMemcpy(mine.data() + o, t.p, t.elems * 2, cudaMemcpyDeviceToHost));
    o += t.elems * 2;
  }
  // the ring's host all-gather takes at most kHostGatherMax bytes a call: the slices go in pieces of that size
  chunked_host_allgather([&](const void* m, void* a, size_t c) { ring.allgather_host(m, a, c); }, mine.data(), all.data(), total);
  const int r = ring.rank();
  if (r != 1 && r != 2) return;
  o = 0;
  for (const T& t : ts) {
    CUDA_CHECK(cudaMemcpy(t.p, all.data() + (size_t)(3 - r) * total + o, t.elems * 2, cudaMemcpyHostToDevice));
    o += t.elems * 2;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char** argv) {
  comm::Options o;
  o.bootstrap_timeout_s = 120;  // run4-vit.sh starts all ranks within seconds
  o.cap = 4u << 20;             // the engine's ring cap (--ring-cap's default)
  o.direct = 2;                 // as the engine runs this fleet (DSGB10_RING_DIRECT=2)
  std::string weights = "~/dsgb10/weights/vision.rank{rank}.dsw", model, wrong = "swap-shard", dump_spans;
  long synthetic = -1;
  int layers = kLayers, blocks_split = 13, repeats = 5;
  bool blocks_given = false;
  std::vector<std::string> raws, tests;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--fleet") o.fleet = next();
    else if (a == "--rank") o.rank = std::atoi(next().c_str());
    else if (a == "--weights") weights = next();
    else if (a == "--model") model = next();
    else if (a == "--synthetic") synthetic = std::atol(next().c_str());
    else if (a == "--layers") layers = std::atoi(next().c_str());
    else if (a == "--blocks") blocks_split = std::atoi(next().c_str()), blocks_given = true;
    else if (a == "--only") {
      const std::string v = next();
      for (size_t p = 0; p <= v.size();) {
        size_t q = v.find(',', p);
        if (q == std::string::npos) q = v.size();
        if (q > p) tests.push_back(v.substr(p, q - p));
        p = q + 1;
      }
    }
    else if (a == "--wrong") wrong = next();
    else if (a == "--raw") raws.push_back(next());
    else if (a == "--dump-spans") dump_spans = next();
    else if (a == "--repeats") repeats = std::atoi(next().c_str());
    else if (a == "--cap") o.cap = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--direct") o.direct = std::atoi(next().c_str());
    else if (a == "--bootstrap-timeout") o.bootstrap_timeout_s = std::atoi(next().c_str());
    else if (a.rfind("--", 0) == 0) die("unknown option %s", a.c_str());
    else tests.push_back(a);
  }
  if (tests.empty()) tests = {"correct", "control", "time"};
  for (const std::string& t : tests)
    if (t != "correct" && t != "control" && t != "time") die("unknown test %s (correct, control, time)", t.c_str());
  if (wrong != "swap-shard") die("--wrong %s: only swap-shard is made with the rank files alone", wrong.c_str());
  if (repeats < 1) die("--repeats at least 1");
  if (layers != kLayers && synthetic < 0) die("--layers needs --synthetic: the weight files hold all %d blocks", kLayers);
  if (!blocks_given && blocks_split >= layers) blocks_split = layers / 2;  // the default 13 does not fit a short synthetic tower
  if (blocks_split < 0 || blocks_split >= layers) die("--blocks k with 0 <= k < %d", layers);

  CUDA_CHECK(cudaSetDevice(0));
  comm::Ring ring(o);
  g_ring = &ring;
  const int rank = ring.rank();
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

  // ---- weights: this rank's quarter; rank 0 also the whole set for the unsplit reference
  std::unique_ptr<VisionWeights> vw;
  DevWeights own, whole_dev;
  const DevWeights* w;
  Weights whole;
  const bool ref_here = rank == 0;
  if (synthetic >= 0) {
    whole = make_synthetic((uint64_t)synthetic, layers);
    own = upload(slice_rank(whole, rank));
    w = &own;
  } else {
    const std::string path = expand(weights, rank);
    vw = std::make_unique<VisionWeights>(path, rank);
    uint64_t d = vw->digest(), all[4];
    ring.allgather_host(&d, all, sizeof d);
    for (int r = 1; r < 4; r++)
      if (all[r] != all[0]) die("the four vision weight files are not one set (digests differ at rank %d)", r);
    const dsw::LoadStats ls = vw->load();
    (void)ls;
    w = &vw->weights();
    if (rank == 0 && model.empty()) die("rank 0 needs --model DIR (or --synthetic) for the unsplit reference");
    if (rank == 0) whole = load_checkpoint(model);
  }
  if (ref_here) whole_dev = upload(whole);
  if (rank == 0) std::printf("vit_ring_test: %zu blocks a rank, weights %s, ring cap %zu, direct %d\n", w->blocks.size(),
                             synthetic >= 0 ? "synthetic" : "from the files", (size_t)o.cap, o.direct);

  // ---- images
  std::vector<Case> cases;
  auto add = [&](const std::string& name, int nh, int nw, uint64_t seed) {
    Case c{name, nh, nw, {}};
    if (rank == 0) c.px = gen_image(nh, nw, seed);
    cases.push_back(std::move(c));
  };
  add("39x39", 39, 39, 1), add("55x55", 55, 55, 2), add("73x55", 73, 55, 3), add("75x75", 75, 75, 4),
      add("93x93", 93, 93, 5), add("3x3063", 3, 3063, 6), add("1533x3", 1533, 3, 7);
  for (const std::string& r : raws) {
    const size_t colon = r.rfind(':'), x = r.rfind('x');
    int h, wd;
    if (colon == std::string::npos || x == std::string::npos || x < colon ||
        std::sscanf(r.c_str() + colon + 1, "%dx%d", &h, &wd) != 2 || h * wd > kMaxPatches || h < 1 || wd < 1)
      die("--raw FILE:HxW with H * W <= %d", kMaxPatches);
    Case c{r.substr(0, colon), h, wd, {}};
    if (rank == 0) c.px = read_raw(r.substr(0, colon), (size_t)h * wd * kPatchK);
    cases.push_back(std::move(c));
  }
  int max_patches = 0, max_merged = 0;
  size_t max_rows = 0;
  for (const Case& c : cases) {
    const Grid g = make_grid(c.nh, c.nw);
    max_patches = std::max(max_patches, g.patches()), max_merged = std::max(max_merged, g.merged());
    max_rows = std::max(max_rows, (size_t)g.rows());
  }
  Run R{ring, *w, st, nullptr, nullptr, std::vector<cudaEvent_t>(2 * 140 + 4)};
  // the work area is sized for the largest image: a smaller image's layout is a prefix-compatible one of its own sizes
  CUDA_CHECK(cudaMalloc(&R.work, rank_work_bytes(max_patches, max_merged)));
  CUDA_CHECK(cudaMalloc((void**)&R.span, max_rows * kLm * 2));
  for (auto& e : R.ev) CUDA_CHECK(cudaEventCreate(&e));

  std::vector<std::vector<uint8_t>> ref(cases.size());
  std::vector<uint8_t*> dpx(cases.size(), nullptr);
  auto reference = [&](size_t i) {  // rank 0: the unsplit run on the whole weights
    const Case& c = cases[i];
    const Grid g = make_grid(c.nh, c.nw);
    void* work;
    CUDA_CHECK(cudaMalloc(&work, unsplit_work_bytes(g.patches(), g.merged())));
    bf16* span;
    CUDA_CHECK(cudaMalloc((void**)&span, (size_t)g.rows() * kLm * 2));
    run_unsplit(whole_dev, Image{g, dpx[i]}, work, RunOut{span}, st);
    ref[i].resize((size_t)g.rows() * kLm * 2);
    CUDA_CHECK(cudaMemcpy(ref[i].data(), span, ref[i].size(), cudaMemcpyDeviceToHost));
    cudaFree(span);
    cudaFree(work);
  };
  auto is_on = [&](const char* t) { return std::find(tests.begin(), tests.end(), t) != tests.end(); };
  if (rank == 0)
    for (size_t i = 0; i < cases.size(); i++) {
      dpx[i] = to_dev(cases[i].px);
      if (is_on("correct") || (is_on("control") && i < 2)) reference(i);  // only what the selected tests compare against
      if (!dump_spans.empty() && !ref[i].empty()) {  // the unsplit run's span rows of each image: the engine's tests compare what it writes to them
        const std::string base = cases[i].name.substr(cases[i].name.find_last_of('/') + 1);
        if (FILE* f = std::fopen((dump_spans + "/" + base + ".span.bin").c_str(), "wb")) {
          std::fwrite(ref[i].data(), 1, ref[i].size(), f);
          std::fclose(f);
        } else {
          die("--dump-spans: cannot write %s/%s.span.bin", dump_spans.c_str(), base.c_str());
        }
      }
    }

  // ---- correct
  if (is_on("correct")) {
    if (rank == 0) std::printf("== correct\n");
    for (size_t i = 0; i < cases.size(); i++) {
      const Case& c = cases[i];
      const std::string tag = c.name + " (" + std::to_string(c.patches()) + " patches)";
      ring.barrier();
      Timing t;
      run_range(R, c, dpx[i], 0, -1, false, &t);
      std::string detail;
      auto v = compare(ring, span_to_host(R, c), rank == 0 ? &ref[i] : nullptr, &detail);
      expect(v.first, tag + ": every rank's span rows equal the unsplit run's, bytes (" + detail + ")");
      if (blocks_split > 0) {
        ring.barrier();
        run_range(R, c, dpx[i], 0, blocks_split, false, &t);
        RankRun::poison_all_but_state(R.work, make_grid(c.nh, c.nw), st);
        run_range(R, c, dpx[i], blocks_split, -1, false, &t);
        auto v2 = compare(ring, span_to_host(R, c), rank == 0 ? &ref[i] : nullptr, &detail);
        expect(v2.first, tag + ": blocks [0, " + std::to_string(blocks_split) + ") then [" + std::to_string(blocks_split) +
                             ", " + std::to_string(layers) + ") equal too (" + detail + ")");
      }
    }
  }

  // ---- control
  if (is_on("control")) {
    if (rank == 0) std::printf("== control: %s\n", wrong.c_str());
    if (rank == 0 && !is_on("correct")) std::printf("(the reference of each image is computed here: correct is not run)\n");
    swap_shard(ring, *w);
    for (size_t i = 0; i < 2; i++) {
      const Case& c = cases[i];
      ring.barrier();
      Timing t;
      run_range(R, c, dpx[i], 0, -1, false, &t);
      std::string detail;
      auto v = compare(ring, span_to_host(R, c), rank == 0 ? &ref[i] : nullptr, &detail);
      expect(v.second, c.name + ": with block 17's slices of ranks 1 and 2 exchanged, no rank's span equals the unsplit run's (control detected; " + detail + ")");
    }
    swap_shard(ring, *w);  // back
    {
      const Case& c = cases[0];
      ring.barrier();
      Timing t;
      run_range(R, c, dpx[0], 0, -1, false, &t);
      std::string detail;
      auto v = compare(ring, span_to_host(R, c), rank == 0 ? &ref[0] : nullptr, &detail);
      expect(v.first, c.name + ": swapped back, every rank's span equals the unsplit run's again (" + detail + ")");
    }
  }

  // ---- time
  if (is_on("time")) {
    if (rank == 0) {
      std::printf("== time: an image through RankRun and the ring, rank 0's stream, ms; one uncounted run then %d\n", repeats);
      std::printf("%-8s %8s | %8s %8s %8s %6s | %8s %6s | %7s | each rank's median wall\n", "image", "patches", "median", "min",
                  "max", "spr %", "gathers", "of it", "gathers");
    }
    for (int target : {1521, 3025, 5625, 8649, 9189}) {
      size_t i = 0;
      while (i < cases.size() && cases[i].patches() != target) i++;
      if (i == cases.size()) {
        if (rank == 0) std::printf("(no image of %d patches in the set)\n", target);
        continue;
      }
      const Case& c = cases[i];
      Timing t;
      ring.barrier();
      run_range(R, c, dpx[i], 0, -1, false, &t);  // uncounted
      std::vector<double> wall, gath, ngat;
      for (int rep = 0; rep < repeats; rep++) {
        ring.barrier();
        run_range(R, c, dpx[i], 0, -1, true, &t);
        wall.push_back(t.wall_ms), gath.push_back(t.gather_ms), ngat.push_back(t.gathers);
      }
      const double mine = median(wall);
      double each[4];
      ring.allgather_host(&mine, each, sizeof mine);
      if (rank == 0) {
        const double med = median(wall), lo = *std::min_element(wall.begin(), wall.end()), hi = *std::max_element(wall.begin(), wall.end());
        const double gm = median(gath);
        std::printf("%-8s %8d | %8.1f %8.1f %8.1f %6.1f | %8.1f %5.0f%% | %7d | %.1f %.1f %.1f %.1f\n", c.name.c_str(),
                    c.patches(), med, lo, hi, 100.0 * (hi - lo) / med, gm, 100.0 * gm / med, (int)ngat[0], each[0], each[1],
                    each[2], each[3]);
        std::fflush(stdout);
      }
    }
  }

  std::printf("rank %d: %s\n", rank, g_fails ? "FAILED" : "passed");
  return g_fails ? 1 : 0;
}
