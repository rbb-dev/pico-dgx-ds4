// A window's memory leaving the chunk pool and coming back (model/vmm.h shrink / trim / add_chunks), on one GB10 with
// the driver's virtual memory calls: the unmapped tail's chunks go to the driver and the free device memory rises by
// what they were, a plain cudaMalloc of that size then succeeds, what stayed mapped keeps its bytes and its addresses,
// a graph captured over the arrays replays after the memory comes back, and the chunks made again map and hold writes.
// One PASS or FAIL line per check; the exit status is 0 only when every check passed. The free-memory checks are loose
// (half the released bytes) if another process moves device memory.
//   vmm_giveback [--chunks N]   (chunks per array, default 64: two arrays of 128 MiB)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "model/vmm.h"

using namespace model::vmm;

namespace {

int g_fail = 0;

void report(bool ok, const char* name, const char* detail = "") {
  std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, *detail ? ": " : "", detail);
  if (!ok) ++g_fail;
}

#define RT(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t e_ = (x);                                                                                  \
    if (e_ != cudaSuccess) {                                                                               \
      std::printf("FAIL runtime call %s (%s:%d): %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                                        \
    }                                                                                                      \
  } while (0)

double free_mib() {
  size_t f = 0, t = 0;
  RT(cudaMemGetInfo(&f, &t));
  return f / 1048576.0;
}

__global__ void fill(uint32_t* p, size_t words, uint32_t v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) p[i] = v + (uint32_t)(i & 1023);
}

// words [0, n) of p against fill's pattern for v
bool holds(const uint32_t* p, size_t n, uint32_t v) {
  std::vector<uint32_t> h(n);
  RT(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i)
    if (h[i] != v + (uint32_t)(i & 1023)) return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  size_t per = 64;  // chunks an array holds
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--chunks")) per = (size_t)std::atoi(argv[i + 1]);
  RT(cudaSetDevice(0));
  RT(cudaFree(nullptr));
  std::string err;
  std::unique_ptr<Backend> be = make_cuda_backend(&err);
  if (!be) {
    std::printf("FAIL no backend: %s\n", err.c_str());
    return 2;
  }
  const double f0 = free_mib();
  Pool pool(std::move(be), 2 * per);
  report(pool.ok(), "the pool is made", pool.error().c_str());
  if (pool.add_array(per * kChunk, 1) != 0 || pool.add_array(per * kChunk, 1) != 1 || pool.add_slot(0, 1) != 0) {
    std::printf("FAIL arrays\n");
    return 2;
  }
  const size_t all[2] = {per * kChunk, per * kChunk};
  report(pool.ensure(0, all), "every chunk mapped", pool.error().c_str());
  uint32_t* a0 = reinterpret_cast<uint32_t*>(pool.base(0));
  uint32_t* a1 = reinterpret_cast<uint32_t*>(pool.base(1));
  const double f1 = free_mib();
  // a graph over the arrays: fills the first chunk of each, captured now, replayed after the memory comes back
  cudaStream_t st;
  RT(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  RT(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  fill<<<64, 256, 0, st>>>(a0, kChunk / 4, 100);
  fill<<<64, 256, 0, st>>>(a1, kChunk / 4, 200);
  RT(cudaStreamEndCapture(st, &g));
  RT(cudaGraphInstantiate(&ge, g, 0));
  RT(cudaGraphDestroy(g));
  RT(cudaGraphLaunch(ge, st));
  RT(cudaStreamSynchronize(st));
  report(holds(a0, kChunk / 4, 100) && holds(a1, kChunk / 4, 200), "the graph filled both arrays' first chunks");
  // the tail of the last kept chunk, written eagerly
  const size_t keep = per / 4;
  fill<<<64, 256, 0, st>>>(a0 + (keep - 1) * (kChunk / 4), kChunk / 4, 300);
  RT(cudaStreamSynchronize(st));

  // give the tail back: unmap down to `keep` chunks, then the driver gets every free chunk
  const size_t part[2] = {keep * kChunk, keep * kChunk};
  pool.shrink(0, part);
  const size_t released = pool.trim(pool.free_chunks());
  const double f2 = free_mib();
  const size_t want = 2 * (per - keep);
  char d[256];
  std::snprintf(d, sizeof d, "%zu chunks released (%zu expected); free memory %.0f -> %.0f MiB", released, want, f1, f2);
  report(released == want && pool.total() == 2 * keep && pool.ok(), "shrink and trim gave back the tail", d);
  std::snprintf(d, sizeof d, "free memory rose %.0f MiB; %.0f MiB released", f2 - f1, (double)released * 2);
  report(f2 - f1 >= 0.5 * released * 2, "the free device memory rose by about what was released", d);
  void* extra = nullptr;
  const size_t take = released * kChunk * 3 / 4;
  report(cudaMalloc(&extra, take) == cudaSuccess, "a cudaMalloc of three quarters of it succeeds");
  cudaGetLastError();
  if (extra) cudaFree(extra);
  report(holds(a0 + (keep - 1) * (kChunk / 4), kChunk / 4, 300) && holds(a0, kChunk / 4, 100) && holds(a1, kChunk / 4, 200),
         "what stayed mapped holds its bytes at the same addresses");

  // the memory comes back: chunks made again, mapped over the same addresses
  report(pool.add_chunks(released) == released && pool.ok(), "chunks made again", pool.error().c_str());
  report(pool.ensure(0, all), "the tail mapped again", pool.error().c_str());
  const double f3 = free_mib();
  std::snprintf(d, sizeof d, "free memory %.0f MiB (%.0f before the give-back, %.0f at the start)", f3, f1, f0);
  report(f3 <= f2 - 0.5 * released * 2, "the free device memory fell again", d);
  fill<<<64, 256, 0, st>>>(a0 + (per - 1) * (kChunk / 4), kChunk / 4, 400);
  fill<<<64, 256, 0, st>>>(a1 + (per - 1) * (kChunk / 4), kChunk / 4, 500);
  RT(cudaStreamSynchronize(st));
  report(holds(a0 + (per - 1) * (kChunk / 4), kChunk / 4, 400) && holds(a1 + (per - 1) * (kChunk / 4), kChunk / 4, 500),
         "the last chunks of both arrays take writes");
  RT(cudaMemsetAsync(a0, 0, kChunk, st));
  RT(cudaGraphLaunch(ge, st));
  RT(cudaStreamSynchronize(st));
  report(holds(a0, kChunk / 4, 100) && holds(a1, kChunk / 4, 200), "the graph captured before replays unchanged");
  RT(cudaGraphExecDestroy(ge));
  RT(cudaStreamDestroy(st));
  std::printf("%s\n", g_fail ? "FAIL" : "PASS: vmm_giveback");
  return g_fail ? 1 : 0;
}
