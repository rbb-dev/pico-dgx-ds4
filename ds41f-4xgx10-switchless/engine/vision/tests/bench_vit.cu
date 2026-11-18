// Time of the encoder's kernels at its shapes, on random data: each matrix product, the attention, a rank's
// share of one block (the four-rank run's per-rank work, without the exchanges) and the unsplit block.
// Median of N runs, each a launch timed with events after a warm-up.
//   bench_vit [--iters N] [n ...]     n = patches (default 1521 3025 9189)
// Run with the engine stopped. A rank's share + the exchange time of the design (about 1.44 ms a block at
// n = 3,025) times 32 is the split encode's time.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/check.h"
#include "vision/vit.h"

using namespace vit;

namespace {

cudaStream_t st = 0;
cudaEvent_t e0, e1;

template <class F>
double time_ms(int iters, F f) {
  for (int i = 0; i < 3; i++) f();
  CUDA_CHECK(cudaStreamSynchronize(st));
  std::vector<float> t;
  for (int i = 0; i < iters; i++) {
    CUDA_CHECK(cudaEventRecord(e0, st));
    f();
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

bf16* rnd(size_t n, float scale) {
  std::vector<u16> h(n);
  uint64_t s = 0x2545F4914F6CDD1Dull + n;
  for (auto& x : h) s ^= s << 13, s ^= s >> 7, s ^= s << 17, x = to_bf16(scale * (float)(int32_t)(s >> 32) / 2147483648.0f);
  bf16* d;
  CUDA_CHECK(cudaMalloc(&d, n * 2));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
  return d;
}

void row(const char* name, double ms, double flop) {
  if (flop > 0) std::printf("  %-34s %8.3f ms  %6.1f TFLOPS\n", name, ms, flop / (ms * 1e-3) / 1e12);
  else std::printf("  %-34s %8.3f ms\n", name, ms);
}

}  // namespace

int main(int argc, char** argv) {
  int iters = 20;
  std::vector<int> ns;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else ns.push_back(std::atoi(a.c_str()));
  }
  if (ns.empty()) ns = {1521, 3025, 9189};
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  for (int n : ns) {
    std::printf("n = %d patches\n", n);
    const struct { const char* name; int N, K; } mm[] = {
        {"gemm wqkv, rank (768 x 1024)", 768, 1024},  {"gemm wo, rank (256 x 1024)", 256, 1024},
        {"gemm w1, rank (1408 x 1024)", 1408, 1024},  {"gemm w2, rank (256 x 2816)", 256, 2816},
        {"gemm wqkv, whole (3072 x 1024)", 3072, 1024}, {"gemm wo, whole (1024 x 1024)", 1024, 1024},
        {"gemm w1, whole (5632 x 1024)", 5632, 1024}, {"gemm w2, whole (1024 x 2816)", 1024, 2816}};
    bf16* x = rnd((size_t)n * 3072, 1.0f);
    bf16* big = rnd((size_t)n * 5632, 1.0f);
    bf16* out;
    CUDA_CHECK(cudaMalloc(&out, (size_t)n * 5632 * 2));
    double rank_block = 0, whole_block = 0;
    double t_mm[8];
    for (int i = 0; i < 8; i++) {
      const int N = mm[i].N, K = mm[i].K;
      bf16* w = rnd((size_t)N * K, 0.05f);
      bf16* b = rnd(N, 0.05f);
      // inputs in column blocks where the split run has them (wo: o, w2: h) cost the same as contiguous rows
      t_mm[i] = time_ms(iters, [&] { gemm(rows_view(big, K), w, b, rows_out(out, N), n, N, K, st); });
      row(mm[i].name, t_mm[i], 2.0 * n * N * K);
      cudaFree(w), cudaFree(b);
    }
    bf16* q = x;
    const double t_a4 = time_ms(iters, [&] { attention(q, 768, q + 256, 768, q + 512, 768, out, 256, n, 4, st); });
    const double t_a16 = time_ms(iters, [&] { attention(q, 3072, q + 1024, 3072, q + 2048, 3072, out, 1024, n, 16, st); });
    row("attention, 4 heads (a rank)", t_a4, 4.0 * 4 * 64 * (double)n * n);
    row("attention, 16 heads", t_a16, 16.0 * 4 * 64 * (double)n * n);
    bf16* w = rnd(1024, 0.3f);
    const double t_norm = time_ms(iters, [&] { rmsnorm(rows_view(big, 1024), w, out, n, st); });
    const double t_add = time_ms(iters, [&] { add(big, 1024, big, 1024, out, 1024, n, 1024, st); });
    const double t_swi4 = time_ms(iters, [&] { swiglu(big, 1408, 704, out, 704, n, st); });
    const double t_swi = time_ms(iters, [&] { swiglu(big, 5632, 2816, out, 2816, n, st); });
    float* cs;
    CUDA_CHECK(cudaMalloc(&cs, (size_t)n * 128 * 2));
    const double t_rope4 = time_ms(iters, [&] { rope_apply(q, q + 256, 768, n, 4, cs, cs + (size_t)n * 32, st); });
    const double t_rope16 = time_ms(iters, [&] { rope_apply(q, q + 1024, 3072, n, 16, cs, cs + (size_t)n * 32, st); });
    row("rmsnorm", t_norm, 0), row("add (1024 wide)", t_add, 0), row("swiglu, rank (704)", t_swi4, 0);
    row("swiglu, whole (2816)", t_swi, 0);
    row("rope on q and k, rank (4 heads)", t_rope4, 0), row("rope on q and k, whole (16 heads)", t_rope16, 0);
    // a rank's block: 2 norms, wqkv, rope, attention (4 heads), wo, add, w1, swiglu, w2, add
    rank_block = 2 * t_norm + t_mm[0] + t_rope4 + t_a4 + t_mm[1] + t_add + t_mm[2] + t_swi4 + t_mm[3] + t_add;
    whole_block = 2 * t_norm + t_mm[4] + t_rope16 + t_a16 + t_mm[5] + t_add + t_mm[6] + t_swi + t_mm[7] + t_add;
    std::printf("  a rank's share of one block        %8.3f ms   x32 = %7.1f ms\n", rank_block, 32 * rank_block);
    std::printf("  one block on one GPU, unsplit      %8.3f ms   x32 = %7.1f ms\n", whole_block, 32 * whole_block);
    cudaFree(x), cudaFree(big), cudaFree(out), cudaFree(w), cudaFree(cs);
  }
  return 0;
}
