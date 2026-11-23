// The embedding lookup's kernels on a GB10 (the CMake target embed_shard_dev_test; one rank, no ring; four shards of
// 331 MB, so stop the engine first): ops::embed_part for each of four ranks' shards into four slots, summed on the host the way
// the ring sums (fp32 in rank order, then bf16), then ops::embed_fan, against the replicated row, for all 129,280 ids in steps of
// kMaxRows rows. The table is element(id, c) (+0.0 and -0.0 in one element of eight), so no whole table is held. Then
// ops::embed_send into four ranks' parts and ops::embed_pick from the layout the all-to-all makes (rank k's rows for rank r at
// recv[k][rq]), against the same rows. Exit 0 when every check passes. The out-of-range stop (__trap) is not run here: run
// `embed_shard_dev_test --trap` to see the kernel stop on id 129,280 (a nonzero exit with a CUDA error).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "common/embed_shard.h"
#include "model/ops.h"

using namespace model;

namespace {

__host__ __device__ uint32_t mix(uint32_t x) {
  x ^= x >> 16, x *= 0x7feb352dU, x ^= x >> 15, x *= 0x846ca68bU, x ^= x >> 16;
  return x;
}
__host__ __device__ uint16_t element(uint32_t id, uint32_t c) {
  const uint32_t h = mix(id * 31 + c);
  if ((h & 7) == 0) return (h >> 8) & 1 ? 0x8000 : 0x0000;
  uint16_t v = uint16_t(h >> 16);
  if ((v & 0x7f80) == 0x7f80) v &= 0xbf7f;
  return v;
}
__global__ void fill(uint16_t* shard, int base) {
  const size_t row = blockIdx.x;
  for (int c = threadIdx.x; c < kDim; c += blockDim.x) shard[row * kDim + c] = element(uint32_t(base + (int)row), uint32_t(c));
}

float bf2f(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}

int fails = 0;
void expect(bool ok, const char* what, long n = 0) {
  std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, n ? ": " : "", n ? std::to_string(n).c_str() : "");
  if (!ok) ++fails;
}

}  // namespace

int main(int argc, char** argv) {
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  using bf16 = ops::bf16;
  bf16* shard[kRanks];
  for (int r = 0; r < kRanks; r++) {
    CUDA_CHECK(cudaMalloc(&shard[r], (size_t)kVocabPart * kDim * 2));
    fill<<<kVocabPart, 256, 0, st>>>(reinterpret_cast<uint16_t*>(shard[r]), r * kVocabPart);
  }
  int* tok;
  bf16 *slot[kRanks], *x;
  float* pm;
  CUDA_CHECK(cudaMalloc(&tok, kMaxRows * 4));
  CUDA_CHECK(cudaMalloc(&x, (size_t)kMaxRows * kHcDim * 2));
  CUDA_CHECK(cudaMalloc(&pm, (size_t)kMaxRows * kHc * 4));
  for (auto& s : slot) CUDA_CHECK(cudaMalloc(&s, (size_t)kMaxRows * kDim * 2));
  bf16 *sum, *part[kRanks], *recv;
  CUDA_CHECK(cudaMalloc(&sum, (size_t)kMaxRows * kDim * 2));
  for (auto& s : part) CUDA_CHECK(cudaMalloc(&s, (size_t)kMaxRows * kDim * 2 * 4));
  CUDA_CHECK(cudaMalloc(&recv, (size_t)kMaxRows * kDim * 2 * 4));
  CUDA_CHECK(cudaStreamSynchronize(st));
  if (argc > 1 && !std::strcmp(argv[1], "--trap")) {
    const int bad = kVocab;
    CUDA_CHECK(cudaMemcpy(tok, &bad, 4, cudaMemcpyHostToDevice));
    ops::embed_part(shard[0], 0, tok, 1, slot[0], st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::printf("the kernel did not stop on id %d\n", bad);
    return 1;
  }

  long bad_sum = 0, bad_x = 0, bad_pm = 0, bad_pick = 0;
  std::vector<int> ids(kMaxRows);
  std::vector<uint16_t> h_slot[kRanks], h_x((size_t)kMaxRows * kHcDim), h_recv;
  std::vector<float> h_pm((size_t)kMaxRows * kHc);
  for (auto& v : h_slot) v.resize((size_t)kMaxRows * kDim);
  for (int id0 = 0; id0 < kVocab; id0 += kMaxRows) {
    const int R = std::min(kMaxRows, kVocab - id0);
    for (int i = 0; i < R; i++) ids[i] = id0 + i;
    CUDA_CHECK(cudaMemcpy(tok, ids.data(), R * 4, cudaMemcpyHostToDevice));
    for (int r = 0; r < kRanks; r++) {
      ops::embed_part(shard[r], r, tok, R, slot[r], st);
      CUDA_CHECK(cudaMemcpyAsync(h_slot[r].data(), slot[r], (size_t)R * kDim * 2, cudaMemcpyDeviceToHost, st));
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<uint16_t> h_sum((size_t)R * kDim);
    for (size_t k = 0; k < h_sum.size(); k++) {
      float acc = bf2f(h_slot[0][k]);
      for (int r = 1; r < kRanks; r++) acc += bf2f(h_slot[r][k]);
      h_sum[k] = f2bf(acc);
      bad_sum += h_sum[k] != element(uint32_t(id0 + k / kDim), uint32_t(k % kDim));
    }
    CUDA_CHECK(cudaMemcpy(sum, h_sum.data(), h_sum.size() * 2, cudaMemcpyHostToDevice));
    ops::embed_fan(sum, R, x, pm, st);
    CUDA_CHECK(cudaMemcpy(h_x.data(), x, (size_t)R * kHcDim * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pm.data(), pm, (size_t)R * kHc * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < R; i++) {
      for (int c = 0; c < kHc; c++)
        for (int k = 0; k < kDim; k++) bad_x += h_x[(size_t)i * kHcDim + c * kDim + k] != h_sum[(size_t)i * kDim + k];
      for (int c = 0; c < kHc; c++) bad_pm += h_pm[i * kHc + c] != (c == 0 ? 1.f : 0.f);
    }
    // prefill's form for chunks of kMaxRows rows over four ranks: rq rows each, rank r's rows [r rq, +rq)
    const int rq = (R + 3) / 4;
    for (int r = 0; r < kRanks; r++) ops::embed_send(shard[r], r, tok, R, part[r], st);
    for (int r = 0; r < kRanks; r++) {  // the all-to-all's layout at rank r: recv[k][t] = rank k's part row r * rq + t
      for (int k = 0; k < kRanks; k++)
        CUDA_CHECK(cudaMemcpyAsync(recv + (size_t)(k * rq) * kDim, part[k] + (size_t)(r * rq) * kDim,
                                   (size_t)rq * kDim * 2, cudaMemcpyDeviceToDevice, st));
      const int n = std::max(0, std::min(rq, R - r * rq));
      ops::embed_pick(recv, rq, tok + r * rq, n, x, pm, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      CUDA_CHECK(cudaMemcpy(h_x.data(), x, (size_t)std::max(n, 1) * kHcDim * 2, cudaMemcpyDeviceToHost));
      for (int t = 0; t < n; t++)
        for (int c = 0; c < kHc; c++)
          for (int k = 0; k < kDim; k++)
            bad_pick += h_x[(size_t)t * kHcDim + c * kDim + k] != element(uint32_t(id0 + r * rq + t), uint32_t(k));
    }
  }
  expect(bad_sum == 0, "all 129,280 ids: four slots summed == the replicated row, elements differing", bad_sum);
  expect(bad_x == 0, "embed_fan puts the row in all four copies, elements differing", bad_x);
  expect(bad_pm == 0, "the pre mix is (1, 0, 0, 0), differing", bad_pm);
  expect(bad_pick == 0, "prefill's send and pick == the replicated row, elements differing", bad_pick);
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
