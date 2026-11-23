// The image encoder on the GPU: every kernel and the tower, four ranks against the unsplit run, in bytes.
//   vit_test [--kernels] [--tower] [--repeat] [--lockstep] [--wrong NAME|all] [--model DIR] [--blocks N] [--grid HxW ...]
// With no selection it runs the five parts kernels, tower, repeat, lockstep (a rank's phase machine, vit::RankRun,
// stepped four at a time on this one GPU) and wrong. The weights are the checkpoint's under --model DIR, else a
// reproducible random set. --blocks N runs the first N blocks only (default 4; 32 is the whole tower).
// --wrong runs the split tower with one deliberate error: it must give other bytes than the unsplit run.
// --lockstep: four RankRuns with the copy gather give run_split's bytes (spans, x after every block, the number of
// gathers), also run as two calls with every buffer but x and the rotary table poisoned between them; a driver that skips
// one gather and the swap-shard weights must give other bytes.
// Run it with the engine stopped (it takes about 2 GiB of device memory and the whole GPU).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "vision/vit.h"
#include "vision/vit_rank.h"
#include "vision/vit_ref.h"

using namespace vit;

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  if (!ok) fails++;
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  std::fflush(stdout);
}

struct Rng {
  uint64_t s = 88172645463325252ull;
  explicit Rng(uint64_t seed) { s ^= seed * 0x9E3779B97F4A7C15ull; }
  float uniform() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return (float)(int32_t)(s >> 32) * (1.0f / 2147483648.0f);
  }
};

std::vector<u16> rand_bf16(Rng& g, size_t n, float scale, float offset = 0.f) {
  std::vector<u16> v(n);
  for (auto& x : v) x = to_bf16(offset + scale * g.uniform());
  return v;
}

template <class T>
T* dev(const std::vector<T>& h) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
bf16* devb(const std::vector<u16>& h) { return reinterpret_cast<bf16*>(dev(h)); }

std::vector<u16> down(const bf16* p, size_t n) {
  std::vector<u16> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
  return h;
}

// rows x cols row-major -> four column blocks, each [rows][cols/4], one after another
std::vector<u16> to_blocks(const std::vector<u16>& m, int rows, int cols) {
  const int cb = cols / 4;
  std::vector<u16> o(m.size());
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) o[(size_t)(c / cb) * rows * cb + (size_t)r * cb + c % cb] = m[(size_t)r * cols + c];
  return o;
}

// within a relative part of the wanted value plus a part of the typical size of the values
bool close_to(double got, double want, double rms, double rel = 0.01, double absf = 3e-3) {
  return std::fabs(got - want) <= rel * std::fabs(want) + absf * rms;
}

cudaStream_t st = 0;

// ------------------------------------------------------------------------------------------------ kernels

void test_gemm() {
  Rng g(1);
  const int M = 300;  // not a multiple of the tile
  // shapes: whole N, K, slice N
  const struct { const char* name; int N, K, ns, nseg; } shapes[] = {
      {"wqkv", 3072, 1024, 768, 3}, {"wo", 1024, 1024, 256, 1}, {"w1", 5632, 1024, 1408, 2},
      {"w2", 1024, 2816, 256, 1},   {"aligner1", 5120, 9216, 1280, 1}, {"aligner2", 5120, 5120, 1280, 1}};
  for (auto& s : shapes) {
    const float sc = std::sqrt(3.0f / s.K);
    const std::vector<u16> a = rand_bf16(g, (size_t)M * s.K, 1.0f), w = rand_bf16(g, (size_t)s.N * s.K, sc),
                           b = rand_bf16(g, s.N, 0.05f);
    bf16 *da = devb(a), *dw = devb(w), *db = devb(b), *dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)M * s.N * 2));
    gemm(rows_view(da, s.K), dw, db, rows_out(dout, s.N), M, s.N, s.K, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<u16> out = down(dout, (size_t)M * s.N);
    // against the definition on a sample
    int bad = 0;
    double rms = 0;
    int count = 0;
    for (int r = 0; r < M; r += 7)
      for (int c = 0; c < s.N; c += 61) {
        const double want = ref::dot(&a[(size_t)r * s.K], &w[(size_t)c * s.K], s.K) + from_bf16(b[c]);
        rms += want * want, count++;
      }
    rms = std::sqrt(rms / count);
    for (int r = 0; r < M; r += 7)
      for (int c = 0; c < s.N; c += 61) {
        const double want = ref::dot(&a[(size_t)r * s.K], &w[(size_t)c * s.K], s.K) + from_bf16(b[c]);
        bad += !close_to(from_bf16(out[(size_t)r * s.N + c]), want, rms);
      }
    expect(bad == 0, std::string("gemm ") + s.name + ": sample against the definition (bad " + std::to_string(bad) + ")");
    // each rank's slice against the whole matrix, in bytes; input as contiguous rows and as four column blocks
    bool same = true, same_blocks = true;
    const int seg = s.N / s.nseg, sn = s.ns / s.nseg;
    bf16* dblk = devb(to_blocks(a, M, s.K));
    for (int r = 0; r < 4; r++) {
      std::vector<u16> ws, bs;
      for (int k = 0; k < s.nseg; k++)
        for (int j = 0; j < sn; j++) {
          const int row = k * seg + sn * r + j;
          ws.insert(ws.end(), w.begin() + (size_t)row * s.K, w.begin() + (size_t)(row + 1) * s.K);
          bs.push_back(b[row]);
        }
      bf16 *dws = devb(ws), *dbs = devb(bs), *douts;
      CUDA_CHECK(cudaMalloc(&douts, (size_t)M * s.ns * 2));
      gemm(rows_view(da, s.K), dws, dbs, rows_out(douts, s.ns), M, s.ns, s.K, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      std::vector<u16> o1 = down(douts, (size_t)M * s.ns);
      gemm(blocks_view(dblk, M, s.K / 4), dws, dbs, rows_out(douts, s.ns), M, s.ns, s.K, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      std::vector<u16> o2 = down(douts, (size_t)M * s.ns);
      for (int row = 0; row < M; row++)
        for (int k = 0; k < s.nseg; k++)
          for (int j = 0; j < sn; j++) {
            const u16 want = out[(size_t)row * s.N + k * seg + sn * r + j];
            same &= o1[(size_t)row * s.ns + k * sn + j] == want;
            same_blocks &= o2[(size_t)row * s.ns + k * sn + j] == want;
          }
      cudaFree(dws), cudaFree(dbs), cudaFree(douts);
    }
    expect(same, std::string("gemm ") + s.name + ": four slices equal the whole matrix, bytes");
    expect(same_blocks, std::string("gemm ") + s.name + ": input in four column blocks equals contiguous rows, bytes");
    // an output written as column blocks holds the same elements
    cudaFree(da), cudaFree(dw), cudaFree(db), cudaFree(dout), cudaFree(dblk);
  }
  // the patch embedding: K = 640 with a zero tail
  {
    const int K = 640, N = 1024;
    std::vector<u16> a = rand_bf16(g, (size_t)M * K, 1.0f), w = rand_bf16(g, (size_t)N * K, 0.1f);
    for (int r = 0; r < M; r++) std::fill(a.begin() + (size_t)r * K + 588, a.begin() + (size_t)(r + 1) * K, 0);
    bf16 *da = devb(a), *dw = devb(w), *dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)M * N * 2));
    gemm(rows_view(da, K), dw, nullptr, rows_out(dout, N), M, N, K, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<u16> out = down(dout, (size_t)M * N);
    int bad = 0;
    for (int r = 0; r < M; r += 5)
      for (int c = 0; c < N; c += 37) bad += !close_to(from_bf16(out[(size_t)r * N + c]), ref::dot(&a[(size_t)r * K], &w[(size_t)c * K], K), 0.5);
    expect(bad == 0, "gemm patch embedding shape (K = 640, no bias)");
    cudaFree(da), cudaFree(dw), cudaFree(dout);
  }
}

void test_attention() {
  Rng g(2);
  for (int n : {200, 1521, 3025, 9189}) {
    const int ld = 3 * kDim;
    std::vector<u16> qkv = rand_bf16(g, (size_t)n * ld, 1.5f);
    bf16* dq = devb(qkv);
    bf16 *dout16, *dout4;
    CUDA_CHECK(cudaMalloc(&dout16, (size_t)n * kDim * 2));
    CUDA_CHECK(cudaMalloc(&dout4, (size_t)n * 256 * 2));
    attention(dq, ld, dq + kDim, ld, dq + 2 * kDim, ld, dout16, kDim, n, kHeads, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<u16> o16 = down(dout16, (size_t)n * kDim);
    bool same = true;
    for (int r = 0; r < 4; r++) {  // rank r: heads 4r..4r+3 of q, k, v as a [n][768] buffer
      std::vector<u16> part((size_t)n * 768);
      for (int row = 0; row < n; row++)
        for (int seg = 0; seg < 3; seg++)
          std::memcpy(&part[(size_t)row * 768 + seg * 256], &qkv[(size_t)row * ld + seg * kDim + 256 * r], 256 * 2);
      bf16* dp = devb(part);
      attention(dp, 768, dp + 256, 768, dp + 512, 768, dout4, 256, n, 4, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      const std::vector<u16> o4 = down(dout4, (size_t)n * 256);
      for (int row = 0; row < n; row++)
        same &= std::memcmp(&o4[(size_t)row * 256], &o16[(size_t)row * kDim + 256 * r], 256 * 2) == 0;
      cudaFree(dp);
    }
    expect(same, "attention n = " + std::to_string(n) + ": 4 heads a rank equal 16 heads in one launch, bytes");
    if (n <= 1521) {  // against softmax in double on sampled rows and heads
      int bad = 0;
      double rms = 0;
      std::vector<std::vector<double>> want;
      std::vector<std::pair<int, int>> at;
      for (int row = 0; row < n; row += n / 9 + 1)
        for (int h = 0; h < kHeads; h += 5) {
          want.push_back(ref::attention_row(&qkv[0], &qkv[kDim], &qkv[2 * kDim], ld, n, h, row));
          at.push_back({row, h});
          for (double x : want.back()) rms += x * x;
        }
      rms = std::sqrt(rms / (want.size() * 64));
      for (size_t i = 0; i < want.size(); i++)
        for (int d = 0; d < 64; d++)
          bad += !close_to(from_bf16(o16[(size_t)at[i].first * kDim + at[i].second * 64 + d]), want[i][d], rms, 0.03, 0.03);
      expect(bad == 0, "attention n = " + std::to_string(n) + ": sample against softmax in double (bad " + std::to_string(bad) + ")");
    }
    cudaFree(dq), cudaFree(dout16), cudaFree(dout4);
  }
}

void test_norm_pointwise() {
  Rng g(3);
  const int n = 333;
  const std::vector<u16> x = rand_bf16(g, (size_t)n * kDim, 2.0f, 0.3f), w = rand_bf16(g, kDim, 0.3f, 1.0f);
  bf16 *dx = devb(x), *dxb = devb(to_blocks(x, n, kDim)), *dw = devb(w), *d1, *d2;
  CUDA_CHECK(cudaMalloc(&d1, (size_t)n * kDim * 2));
  CUDA_CHECK(cudaMalloc(&d2, (size_t)n * kDim * 2));
  rmsnorm(rows_view(dx, kDim), dw, d1, n, st);
  rmsnorm(blocks_view(dxb, n, 256), dw, d2, n, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  const std::vector<u16> o1 = down(d1, (size_t)n * kDim), o2 = down(d2, (size_t)n * kDim);
  expect(o1 == o2, "rmsnorm: four column blocks equal contiguous rows, bytes");
  int bad = 0;
  for (int r = 0; r < n; r += 11) {
    const std::vector<double> want = ref::rmsnorm_row(&x[(size_t)r * kDim], w.data());
    for (int c = 0; c < kDim; c++) bad += !close_to(from_bf16(o1[(size_t)r * kDim + c]), want[c], 1.0, 0.01);
  }
  expect(bad == 0, "rmsnorm: sample against the definition in double");
  // add: exact
  const std::vector<u16> y = rand_bf16(g, (size_t)n * kDim, 1.0f);
  bf16* dy = devb(y);
  add(dx, kDim, dy, kDim, d1, kDim, n, kDim, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  const std::vector<u16> s = down(d1, (size_t)n * kDim);
  bool ok = true;
  for (size_t i = 0; i < s.size(); i++) ok &= s[i] == to_bf16(from_bf16(x[i]) + from_bf16(y[i]));
  expect(ok, "add: bf16(a + b), bytes");
  // swiglu and gelu against the formulas (fp32 libm, within a bf16 step or two)
  const int W = 704;
  std::vector<u16> gu = rand_bf16(g, (size_t)n * 2 * W, 3.0f);
  bf16 *dgu = devb(gu), *dh;
  CUDA_CHECK(cudaMalloc(&dh, (size_t)n * W * 2));
  swiglu(dgu, 2 * W, W, dh, W, n, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  const std::vector<u16> h = down(dh, (size_t)n * W);
  bad = 0;
  for (int r = 0; r < n; r++)
    for (int c = 0; c < W; c++) {
      const double gt = from_bf16(gu[(size_t)r * 2 * W + c]), up = from_bf16(gu[(size_t)r * 2 * W + W + c]);
      const double want = from_bf16(to_bf16((float)(gt / (1.0 + std::exp(-gt))))) * up;
      bad += !close_to(from_bf16(h[(size_t)r * W + c]), want, 1.0, 0.02);
    }
  expect(bad == 0, "swiglu against the formula");
  gelu(dgu, 2 * W, dh, W, n, W, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  const std::vector<u16> ge = down(dh, (size_t)n * W);
  bad = 0;
  for (int r = 0; r < n; r++)
    for (int c = 0; c < W; c++) {
      const double v = from_bf16(gu[(size_t)r * 2 * W + c]);
      bad += !close_to(from_bf16(ge[(size_t)r * W + c]), 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))), 1.0, 0.02);
    }
  expect(bad == 0, "gelu against the exact formula");
  cudaFree(dx), cudaFree(dxb), cudaFree(dw), cudaFree(d1), cudaFree(d2), cudaFree(dy), cudaFree(dgu), cudaFree(dh);
}

void test_rope() {
  Rng g(4);
  const int nh = 7, nw = 11, n = nh * nw;
  const std::vector<float> inv = rope_inv_freq();
  float *dc, *ds;
  CUDA_CHECK(cudaMalloc(&dc, (size_t)n * 128));
  CUDA_CHECK(cudaMalloc(&ds, (size_t)n * 128));
  rope_table(nh, nw, inv.data(), dc, ds, false, st);  // host memory: the kernel takes the 16 values by value
  CUDA_CHECK(cudaStreamSynchronize(st));
  std::vector<float> c(n * 32), s(n * 32);
  CUDA_CHECK(cudaMemcpy(c.data(), dc, n * 128, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(s.data(), ds, n * 128, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int p = 0; p < n; p++)
    for (int j = 0; j < 32; j++) {
      const double pos = j < 16 ? p / nw : p % nw, f = (double)(float)(pos * inv[j & 15]);
      bad += std::fabs(c[p * 32 + j] - std::cos(f)) > 2e-6 || std::fabs(s[p * 32 + j] - std::sin(f)) > 2e-6;
    }
  expect(bad == 0, "rope table: row positions first, then column positions, against the definition");
  // rotation: the same fp32 operations on the host, from the device's table, bytes
  const int heads = 3, ld = 3 * 64 + 5 * 8;  // a row stride larger than the data
  std::vector<u16> q = rand_bf16(g, (size_t)n * ld, 2.0f), k = rand_bf16(g, (size_t)n * ld, 2.0f);
  bf16 *dq = devb(q), *dk = devb(k);
  rope_apply(dq, dk, ld, n, heads, dc, ds, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  const std::vector<u16> q2 = down(dq, q.size()), k2 = down(dk, k.size());
  bool ok = true;
  for (int pass = 0; pass < 2; pass++) {
    const std::vector<u16>& in = pass ? k : q;
    const std::vector<u16>& got = pass ? k2 : q2;
    for (int r = 0; r < n; r++)
      for (int h = 0; h < heads; h++)
        for (int j = 0; j < 32; j++) {
          const float x1 = from_bf16(in[(size_t)r * ld + h * 64 + j]), x2 = from_bf16(in[(size_t)r * ld + h * 64 + 32 + j]);
          const float cc = c[r * 32 + j], ss = s[r * 32 + j];
          const volatile float a = x1 * cc, b = x2 * ss, c2 = x2 * cc, d = x1 * ss;  // no fused multiply-add
          ok &= got[(size_t)r * ld + h * 64 + j] == to_bf16(a - b) && got[(size_t)r * ld + h * 64 + 32 + j] == to_bf16(c2 + d);
        }
    // columns beyond the heads are untouched
    for (int r = 0; r < n; r++)
      for (int cidx = heads * 64; cidx < ld; cidx++) ok &= got[(size_t)r * ld + cidx] == in[(size_t)r * ld + cidx];
  }
  expect(ok, "rope rotation: bytes equal the same fp32 operations; other columns untouched");
  rope_table(nh, nw, inv.data(), dc, ds, true, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  std::vector<float> c2(n * 32);
  CUDA_CHECK(cudaMemcpy(c2.data(), dc, n * 128, cudaMemcpyDeviceToHost));
  expect(c2 != c, "rope table with swapped axes differs (for the wrong-build test)");
  cudaFree(dc), cudaFree(ds), cudaFree(dq), cudaFree(dk);
}

void test_copies() {
  Rng g(5);
  // patchify
  const std::vector<std::pair<int, int>> pgrids = {{3, 5}, {39, 39}};
  for (auto hw : pgrids) {
    const int nh = hw.first, nw = hw.second;
    std::vector<uint8_t> px((size_t)nh * nw * 14 * 14 * 3);
    for (auto& v : px) v = (uint8_t)(g.uniform() * 127 + 128);
    const std::vector<u16> tab = pixel_table();
    uint8_t* dpx = dev(px);
    bf16 *dt = devb(tab), *dout;
    CUDA_CHECK(cudaMalloc(&dout, (size_t)nh * nw * 640 * 2));
    patchify(dpx, nh, nw, dt, dout, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    expect(down(dout, (size_t)nh * nw * 640) == ref::patchify(px, nh, nw, tab),
           "patchify " + std::to_string(nh) + "x" + std::to_string(nw) + ": bytes equal the reference");
    cudaFree(dpx), cudaFree(dt), cudaFree(dout);
  }
  // unfold and span on grids with one or both axes not a multiple of 3 and on the strip
  const std::vector<std::pair<int, int>> ugrids = {{4, 5}, {6, 6}, {49, 65}, {55, 73}, {3, 3063}};
  for (auto hw : ugrids) {
    const int nh = hw.first, nw = hw.second;
    const Grid gr = make_grid(nh, nw);
    const std::vector<u16> x = rand_bf16(g, (size_t)nh * nw * kDim, 1.0f);
    bf16 *dx = devb(x), *du;
    CUDA_CHECK(cudaMalloc(&du, (size_t)gr.merged() * kMergeIn * 2));
    unfold3(dx, nh, nw, du, 0, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    expect(down(du, (size_t)gr.merged() * kMergeIn) == ref::unfold3(x, nh, nw),
           "unfold " + std::to_string(nh) + "x" + std::to_string(nw) + ": bytes equal the reference");
    cudaFree(dx), cudaFree(du);
    Weights d;
    d.start = rand_bf16(g, kLm, 1), d.newline = rand_bf16(g, kLm, 1), d.end = rand_bf16(g, kLm, 1);
    const std::vector<u16> g5 = rand_bf16(g, (size_t)gr.merged() * kLm, 1.0f);
    bf16 *dg = devb(g5), *ds = devb(d.start), *dn = devb(d.newline), *de = devb(d.end), *dst_;
    CUDA_CHECK(cudaMalloc(&dst_, (size_t)gr.rows() * kLm * 2));
    span_rows(rows_view(dg, kLm), ds, dn, de, gr.lh, gr.lw, 0, gr.rows(), dst_, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<u16> want = ref::span(g5, d, gr.lh, gr.lw);
    expect(down(dst_, want.size()) == want, "span " + std::to_string(gr.lh) + "x" + std::to_string(gr.lw) + ": bytes equal the reference");
    // a run of rows from the middle, and the aligner output as four column blocks
    const int first = gr.rows() / 3, count = gr.rows() - first - 1;
    span_rows(blocks_view(devb(to_blocks(g5, gr.merged(), kLm)), gr.merged(), kLm / 4), ds, dn, de, gr.lh, gr.lw,
              first, count, dst_, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<u16> part(want.begin() + (size_t)first * kLm, want.begin() + (size_t)(first + count) * kLm);
    expect(down(dst_, part.size()) == part, "span run of rows from four column blocks equals the same rows");
    cudaFree(dg), cudaFree(ds), cudaFree(dn), cudaFree(de), cudaFree(dst_);
  }
  // the wrong unfold variants really change the bytes
  {
    const int nh = 49, nw = 65;
    const Grid gr = make_grid(nh, nw);
    const std::vector<u16> x = rand_bf16(g, (size_t)nh * nw * kDim, 1.0f);
    bf16 *dx = devb(x), *du;
    CUDA_CHECK(cudaMalloc(&du, (size_t)gr.merged() * kMergeIn * 2));
    unfold3(dx, nh, nw, du, 0, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<u16> good = down(du, (size_t)gr.merged() * kMergeIn);
    for (int v = 1; v <= 3; v++) {
      unfold3(dx, nh, nw, du, v, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      expect(down(du, good.size()) != good, "unfold variant " + std::to_string(v) + " differs from the correct one");
    }
    cudaFree(dx), cudaFree(du);
  }
  // the digest: layout independent, and sensitive to one element
  {
    const int n = 100;
    const std::vector<u16> x = rand_bf16(g, (size_t)n * kDim, 1.0f);
    bf16 *d1 = devb(x), *d2 = devb(to_blocks(x, n, kDim));
    uint64_t* scratch;
    CUDA_CHECK(cudaMalloc(&scratch, digest_scratch_bytes(n)));
    const uint64_t a = digest(rows_view(d1, kDim), n, kDim, scratch, st), b = digest(blocks_view(d2, n, 256), n, kDim, scratch, st);
    std::vector<u16> y = x;
    y[(size_t)57 * kDim + 900] ^= 1;
    bf16* d3 = devb(y);
    expect(a == b, "digest: column blocks and contiguous rows give the same value");
    expect(digest(rows_view(d3, kDim), n, kDim, scratch, st) != a, "digest changes with one flipped bit");
    cudaFree(d1), cudaFree(d2), cudaFree(d3), cudaFree(scratch);
  }
}

// ------------------------------------------------------------------------------------------------- tower

struct Setup {
  Weights whole;
  DevWeights dw, dr[4];
  Setup(const std::string& model, int blocks) {
    whole = model.empty() ? make_synthetic(42, blocks) : load_checkpoint(model, blocks);
    dw = upload(whole);
    for (int r = 0; r < 4; r++) dr[r] = upload(slice_rank(whole, r));
  }
};

struct Result {
  std::vector<u16> span[4];  // split: ranks 0..3; unsplit in span[0]
  std::vector<uint64_t> trace;
};

std::vector<uint8_t> noise_image(int nh, int nw, uint64_t seed) {
  Rng g(seed);
  std::vector<uint8_t> px((size_t)nh * 14 * nw * 14 * 3);
  for (size_t i = 0; i < px.size(); i++) px[i] = (uint8_t)(128 + 127 * g.uniform() * (0.4 + 0.6 * ((i / 3) % 29) / 29.0));
  return px;
}

Result run_one(Setup& S, int nh, int nw, bool split, Wrong wrong, uint64_t seed) {
  const Grid gr = make_grid(nh, nw);
  const std::vector<uint8_t> px = noise_image(nh, nw, seed);
  uint8_t* dpx = dev(px);
  Image img{gr, dpx};
  const size_t rows = (size_t)gr.rows() * kLm;
  bf16* span[4];
  for (auto& p : span) CUDA_CHECK(cudaMalloc(&p, rows * 2));
  Result res;
  RunOut out{span[0], {span[1], span[2], span[3]}, &res.trace};
  if (!split) {
    void* work;
    CUDA_CHECK(cudaMalloc(&work, unsplit_work_bytes(gr.patches(), gr.merged())));
    run_unsplit(S.dw, img, work, out, st);
    cudaFree(work);
    res.span[0] = down(span[0], rows);
  } else {
    void* works[4];
    for (auto& w : works) CUDA_CHECK(cudaMalloc(&w, split_work_bytes(gr.patches(), gr.merged())));
    DevWeights wr[4];
    const DevWeights* use = S.dr;
    if (wrong != Wrong::None) {
      for (int r = 0; r < 4; r++) wr[r] = upload(wrong_slice(S.whole, r, wrong));
      use = wr;
    }
    run_split(use, img, works, out, wrong, SplitHooks{}, st);
    if (wrong != Wrong::None)
      for (auto& w : wr) release(w);
    for (auto& w : works) cudaFree(w);
    for (int r = 0; r < 4; r++) res.span[r] = down(span[r], rows);
  }
  for (auto& p : span) cudaFree(p);
  cudaFree(dpx);
  return res;
}

int first_diff(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
  for (size_t i = 0; i < a.size() && i < b.size(); i++)
    if (a[i] != b[i]) return (int)i;
  return -1;
}

void test_tower(Setup& S, const std::vector<std::pair<int, int>>& grids) {
  for (auto g : grids) {
    const Result u = run_one(S, g.first, g.second, false, Wrong::None, 100 + g.first);
    const Result s = run_one(S, g.first, g.second, true, Wrong::None, 100 + g.first);
    const std::string tag = "tower " + std::to_string(g.first) + "x" + std::to_string(g.second) + " (" + std::to_string(S.whole.blocks.size()) + " blocks)";
    const int d = first_diff(u.trace, s.trace);
    expect(d < 0 && u.trace.size() == s.trace.size(), tag + ": x after every block, split equals unsplit" + (d < 0 ? "" : " (first differs after block " + std::to_string(d) + ")"));
    for (int r = 0; r < 4; r++)
      expect(s.span[r] == u.span[0], tag + ": rank " + std::to_string(r) + "'s span rows equal the unsplit run's, bytes");
    bool finite = true;
    for (u16 v : u.span[0]) finite &= (v & 0x7F80) != 0x7F80;
    expect(finite, tag + ": the rows are finite");
  }
}

// The same image through four RankRuns stepped in lockstep. ranges: each [b0, b1) in turn (b1 < 0: through the last
// block), a new set of RankRuns for each, with every buffer but x and the rotary table poisoned between two calls.
Result run_lock(Setup& S, int nh, int nw, Wrong wrong, uint64_t seed, const std::vector<std::pair<int, int>>& ranges,
                int skip_gather = -1, int* gathers = nullptr) {
  const Grid gr = make_grid(nh, nw);
  const std::vector<uint8_t> px = noise_image(nh, nw, seed);
  uint8_t* dpx = dev(px);
  Image img{gr, dpx};
  const size_t rows = (size_t)gr.rows() * kLm;
  bf16* span[4];
  for (auto& p : span) CUDA_CHECK(cudaMalloc(&p, rows * 2));
  void* works[4];
  for (auto& w : works) CUDA_CHECK(cudaMalloc(&w, rank_work_bytes(gr.patches(), gr.merged())));
  DevWeights wr[4];
  const DevWeights* use = S.dr;
  if (wrong != Wrong::None) {
    for (int r = 0; r < 4; r++) wr[r] = upload(wrong_slice(S.whole, r, wrong));
    use = wr;
  }
  Result res;
  for (size_t k = 0; k < ranges.size(); k++) {
    if (k)
      for (auto& w : works) RankRun::poison_all_but_state(w, gr, st);
    LockstepOpts o;
    o.b0 = ranges[k].first, o.b1 = ranges[k].second, o.skip_gather = skip_gather, o.trace = &res.trace, o.gathers = gathers;
    run_lockstep(use, img, works, span, wrong, o, st);
  }
  if (wrong != Wrong::None)
    for (auto& w : wr) release(w);
  for (auto& w : works) cudaFree(w);
  for (int r = 0; r < 4; r++) res.span[r] = down(span[r], rows);
  for (auto& p : span) cudaFree(p);
  cudaFree(dpx);
  return res;
}

bool spans_equal(const Result& a, const Result& b) {
  for (int r = 0; r < 4; r++)
    if (a.span[r] != b.span[r]) return false;
  return true;
}

void test_lockstep(Setup& S, const std::vector<std::pair<int, int>>& grids) {
  const int nb = (int)S.whole.blocks.size();
  for (auto g : grids) {
    const uint64_t seed = 100 + g.first;
    const std::string tag = "lockstep " + std::to_string(g.first) + "x" + std::to_string(g.second) + " (" + std::to_string(nb) + " blocks)";
    const Result ref = run_one(S, g.first, g.second, true, Wrong::None, seed);
    int gathers = 0;
    const Result a = run_lock(S, g.first, g.second, Wrong::None, seed, {{0, -1}}, -1, &gathers);
    expect(spans_equal(a, ref), tag + ": the four RankRuns' span rows equal run_split's, bytes, every rank");
    expect(a.trace == ref.trace && a.trace.size() == (size_t)nb, tag + ": x after every block equals run_split's");
    expect(gathers == RankRun::gather_count(0, -1, nb) && gathers == 1 + 4 * nb + 2,
           tag + ": " + std::to_string(gathers) + " gathers (1 pixel + 4 a block + 2 aligner)");
    if (nb >= 2) {
      const int k = nb / 2;
      const Result b = run_lock(S, g.first, g.second, Wrong::None, seed, {{0, k}, {k, -1}});
      expect(spans_equal(b, ref) && b.trace == ref.trace,
             tag + ": blocks [0, " + std::to_string(k) + ") then [" + std::to_string(k) + ", " + std::to_string(nb) +
                 ") with all but x and the rotary table poisoned between: run_split's bytes");
      const Result c = run_lock(S, g.first, g.second, Wrong::None, seed, {{0, 1}, {1, nb}});
      expect(spans_equal(c, ref), tag + ": blocks [0, 1) then [1, " + std::to_string(nb) + "): run_split's bytes");
    }
    // negative controls: each must differ from run_split's bytes
    for (int skip : {0, 3, 1 + 4 * nb}) {  // the pixel gather, a block's h gather, the first aligner gather
      const Result d = run_lock(S, g.first, g.second, Wrong::None, seed, {{0, -1}}, skip);
      expect(!spans_equal(d, ref), tag + ": a driver that skips gather " + std::to_string(skip) + " gives other bytes (control)");
    }
  }
  const Result u = run_one(S, 49, 65, false, Wrong::None, 9);
  const Result rs = run_one(S, 49, 65, true, Wrong::SwapShard, 9);
  const Result ls = run_lock(S, 49, 65, Wrong::SwapShard, 9, {{0, -1}});
  expect(spans_equal(ls, rs), "lockstep 49x65 swap-shard: the same bytes as run_split's swap-shard build");
  expect(ls.span[0] != u.span[0], "lockstep 49x65 swap-shard: differs from the unsplit run (control)");
}

void test_repeat(Setup& S, int nh, int nw) {
  const Result a = run_one(S, nh, nw, false, Wrong::None, 5), b = run_one(S, nh, nw, false, Wrong::None, 5);
  expect(a.span[0] == b.span[0] && a.trace == b.trace, "unsplit run twice: the same bytes");
  const Result c = run_one(S, nh, nw, true, Wrong::None, 5), d = run_one(S, nh, nw, true, Wrong::None, 5);
  expect(c.span[0] == d.span[0] && c.trace == d.trace, "split run twice: the same bytes");
  const Result e = run_one(S, nh, nw, false, Wrong::None, 6);
  expect(e.span[0] != a.span[0], "another image gives other bytes (the images are not ignored)");
}

void test_wrong(Setup& S, const std::vector<Wrong>& which, int nh, int nw) {
  const Result u = run_one(S, nh, nw, false, Wrong::None, 9);
  for (Wrong w : which) {
    const Result s = run_one(S, nh, nw, true, w, 9);
    const int d = first_diff(u.trace, s.trace);
    expect(s.span[0] != u.span[0], std::string("wrong build ") + name_of(w) + ": span rows differ from the unsplit run" +
                                       (d >= 0 ? " (x differs from block " + std::to_string(d) + ")" : " (x equal in every block: the error is in the aligner or the span)"));
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool kernels = false, tower = false, repeat = false, lockstep = false;
  std::vector<Wrong> wrong;
  std::string model;
  int blocks = 4;
  std::vector<std::pair<int, int>> grids;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--kernels") kernels = true;
    else if (a == "--tower") tower = true;
    else if (a == "--repeat") repeat = true;
    else if (a == "--lockstep") lockstep = true;
    else if (a == "--model") model = next();
    else if (a == "--blocks") blocks = std::atoi(next().c_str());
    else if (a == "--grid") {
      int h, w;
      if (std::sscanf(next().c_str(), "%dx%d", &h, &w) != 2 || h * w > kMaxPatches) die("--grid HxW with H * W <= %d", kMaxPatches);
      grids.push_back({h, w});
    } else if (a == "--wrong") {
      const std::string v = next();
      if (v == "all") wrong = all_wrong();
      else {
        bool found = false;
        for (Wrong w : all_wrong())
          if (v == name_of(w)) wrong.push_back(w), found = true;
        if (!found) die("unknown wrong build %s", v.c_str());
      }
    } else die("usage: vit_test [--kernels] [--tower] [--repeat] [--lockstep] [--wrong NAME|all] [--model DIR] [--blocks N] [--grid HxW]");
  }
  if (!kernels && !tower && !repeat && !lockstep && wrong.empty()) kernels = tower = repeat = lockstep = true, wrong = all_wrong();
  if (grids.empty()) grids = {{39, 39}, {49, 65}, {55, 55}, {3, 3063}};
  CUDA_CHECK(cudaSetDevice(0));
  if (kernels) test_gemm(), test_attention(), test_norm_pointwise(), test_rope(), test_copies();
  if (tower || repeat || lockstep || !wrong.empty()) {
    Setup S(model, blocks);
    if (tower) test_tower(S, grids);
    if (repeat) test_repeat(S, 39, 39);
    if (lockstep) test_lockstep(S, grids);
    if (!wrong.empty()) test_wrong(S, wrong, 49, 65);
  }
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
