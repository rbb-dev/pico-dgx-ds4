// The slot memory pool's gate (model/vmm.h): the driver's virtual memory calls under the engine's pattern, on one GB10,
// no weights, no ring. A graph captured over a reserved range replays after more is mapped; chunks map and unmap beside a
// replaying graph; a remapped chunk holds what is written; cuMemCreate's time per GiB. One PASS or FAIL line per check,
// exit status 0 only when all pass. Stop the engine first (the probe takes its memory).
//   vmm_probe [--chunks N] [--rounds R] [--create-gib G] [--negative]
// --negative (off by default): a kernel touches a chunk that is not mapped, which the device faults on
// (Xid 31 in the kernel log): the control that shows the probe can detect an unmapped access at all.
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kChunk = 2u << 20;
constexpr size_t kWords = kChunk / 4;

int g_fail = 0;

#define CU(x)                                                                                      \
  do {                                                                                             \
    CUresult r_ = (x);                                                                             \
    if (r_ != CUDA_SUCCESS) {                                                                      \
      const char* s_ = nullptr;                                                                    \
      cuGetErrorString(r_, &s_);                                                                   \
      std::printf("FAIL driver call %s (%s:%d): %s\n", #x, __FILE__, __LINE__, s_ ? s_ : "?");     \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)
#define RT(x)                                                                                      \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      std::printf("FAIL runtime call %s (%s:%d): %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

void report(bool ok, const char* name, const char* detail) {
  std::printf("%s %s: %s\n", ok ? "PASS" : "FAIL", name, detail);
  if (!ok) ++g_fail;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// every word of [p, p + words) += 1, `iters` times: reads and writes through memory, so a replay is long enough
// for the host's maps to land during it
__global__ void k_inc(volatile uint32_t* p, size_t words, uint32_t iters) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x)
    for (uint32_t k = 0; k < iters; ++k) p[i] = p[i] + 1;
}

// word i = i * 2654435761 + salt
__global__ void k_pattern(uint32_t* p, size_t words, uint32_t salt) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (uint32_t)i * 2654435761u + salt;
}

struct Pool {
  CUdeviceptr base = 0;
  size_t nch = 0;
  std::vector<CUmemGenericAllocationHandle> h;
  std::vector<int> mapped;  // chunk index (of the physical ones) -> slot of the range it is at, or -1
  CUmemAllocationProp prop{};
  CUmemAccessDesc acc{};
};

CUdeviceptr va_of(const Pool& P, size_t i) { return P.base + i * kChunk; }

void map_chunk(Pool& P, size_t slot, size_t phys) {
  CU(cuMemMap(va_of(P, slot), kChunk, 0, P.h[phys], 0));
  CU(cuMemSetAccess(va_of(P, slot), kChunk, &P.acc, 1));
  P.mapped[phys] = (int)slot;
}
void unmap_chunk(Pool& P, size_t slot, size_t phys) {
  CU(cuMemUnmap(va_of(P, slot), kChunk));
  P.mapped[phys] = -1;
}

// the first `words` words of the range hold start + launches * iters (what k_inc leaves); returns the mismatches
size_t check_counts(const Pool& P, size_t first_chunk, size_t nch, uint32_t want) {
  std::vector<uint32_t> host(nch * kWords);
  RT(cudaMemcpy(host.data(), (const void*)va_of(P, first_chunk), nch * kChunk, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (const uint32_t v : host) bad += v != want;
  return bad;
}

void m6(double gib) {
  const size_t n = (size_t)(gib * 1073741824.0 / kChunk);
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  RT(cudaGetDevice(&prop.location.id));
  std::vector<CUmemGenericAllocationHandle> h;
  size_t failed = 0;
  const double t0 = now_s();
  for (size_t i = 0; i < n; ++i) {
    CUmemGenericAllocationHandle x;
    if (cuMemCreate(&x, kChunk, &prop, 0) != CUDA_SUCCESS) {
      ++failed;
      break;  // a failure ends the run: every later one would be too
    }
    h.push_back(x);
  }
  const double dt = now_s() - t0;
  const double made = (double)h.size() * kChunk / 1073741824.0;
  char d[256];
  std::snprintf(d, sizeof d, "%zu chunks of 2 MiB (%.2f GiB) in %.3f s = %.1f ms a GiB, %zu failures (asked %.2f GiB)",
                h.size(), made, dt, made > 0 ? 1e3 * dt / made : 0.0, failed, gib);
  report(failed == 0, "M6 cuMemCreate", d);
  const double t1 = now_s();
  for (auto x : h) cuMemRelease(x);
  std::snprintf(d, sizeof d, "released %zu chunks in %.3f s", h.size(), now_s() - t1);
  std::printf("INFO M6 %s\n", d);
}

}  // namespace

int main(int argc, char** argv) {
  size_t nch = 64;
  int rounds = 400;
  double create_gib = 2.0;
  bool negative = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--chunks") && i + 1 < argc) nch = (size_t)std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--create-gib") && i + 1 < argc) create_gib = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--negative")) negative = true;
    else {
      std::printf("usage: vmm_probe [--chunks N] [--rounds R] [--create-gib G] [--negative]\n");
      return 2;
    }
  }
  if (nch < 8) nch = 8;
  RT(cudaFree(nullptr));
  int dev = 0;
  RT(cudaGetDevice(&dev));

  Pool P;
  P.nch = nch;
  P.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  P.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  P.prop.location.id = dev;
  P.acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  P.acc.location.id = dev;
  P.acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  size_t gran = 0;
  CU(cuMemGetAllocationGranularity(&gran, &P.prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  {
    char d[96];
    std::snprintf(d, sizeof d, "minimum granularity %zu bytes, chunk %zu", gran, kChunk);
    report(gran != 0 && kChunk % gran == 0, "granularity", d);
    if (kChunk % gran) return 1;
  }

  // the pool's start: one range reserved, every chunk made once
  CU(cuMemAddressReserve(&P.base, nch * kChunk, kChunk, 0, 0));
  const double tc0 = now_s();
  for (size_t i = 0; i < nch; ++i) {
    CUmemGenericAllocationHandle x;
    CU(cuMemCreate(&x, kChunk, &P.prop, 0));
    P.h.push_back(x);
  }
  P.mapped.assign(nch, -1);
  {
    char d[160];
    std::snprintf(d, sizeof d, "%zu chunks (%.2f GiB) created once in %.1f ms, range of %zu MiB reserved", nch,
                  (double)nch * kChunk / 1073741824.0, 1e3 * (now_s() - tc0), nch * kChunk >> 20);
    report(true, "start", d);
  }

  cudaStream_t sa, sb;
  RT(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
  RT(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;

  if (negative) {  // chunk 0 is not mapped: the kernel faults
    std::printf("INFO negative control: touching an unmapped chunk (expect an illegal address, Xid 31 in dmesg)\n");
    k_inc<<<64, 256, 0, sa>>>((volatile uint32_t*)va_of(P, 0), kWords, 1);
    const cudaError_t e = cudaStreamSynchronize(sa);
    report(e != cudaSuccess, "negative control", e == cudaSuccess ? "NO error: the probe cannot see an unmapped access" : cudaGetErrorString(e));
    return e != cudaSuccess ? 0 : 1;
  }

  // ---- capture over the reserved range, with only chunk 0 mapped; map chunks 1..3; replay; no recapture
  constexpr size_t kSpan = 4;
  constexpr uint32_t kIters = 4;
  map_chunk(P, 0, 0);
  RT(cudaMemsetAsync((void*)va_of(P, 0), 0, kChunk, sb));
  RT(cudaStreamSynchronize(sb));
  RT(cudaStreamBeginCapture(sa, cudaStreamCaptureModeThreadLocal));
  k_inc<<<128, 256, 0, sa>>>((volatile uint32_t*)va_of(P, 0), kSpan * kWords, kIters);  // chunks 0..3, as captured
  RT(cudaStreamEndCapture(sa, &graph));
  RT(cudaGraphInstantiate(&exec, graph, 0));
  for (size_t i = 1; i < kSpan; ++i) {
    map_chunk(P, i, i);
    RT(cudaMemsetAsync((void*)va_of(P, i), 0, kChunk, sb));
  }
  RT(cudaStreamSynchronize(sb));
  const int launches_a = 50;
  for (int i = 0; i < launches_a; ++i) RT(cudaGraphLaunch(exec, sa));
  RT(cudaStreamSynchronize(sa));
  {
    const size_t bad = check_counts(P, 0, kSpan, launches_a * kIters);
    char d[160];
    std::snprintf(d, sizeof d, "%zu words of %zu chunks mapped after the capture: %zu wrong", kSpan * kWords, kSpan, bad);
    report(bad == 0, "M5a graph replays after maps without recapture", d);
  }

  // ---- the graph on stream A over chunks 0..3 keeps replaying while the host maps and unmaps chunks
  // 4.. into the same range (SetAccess each time) and checks what a kernel on stream B writes into them
  size_t maps = 0, bad_b = 0;
  double t_map = 0;
  uint32_t launches = launches_a;
  const size_t dyn = nch - kSpan;  // chunks 4 .. nch - 1 take turns
  for (int r = 0; r < rounds; ++r) {
    for (int i = 0; i < 8; ++i) {
      RT(cudaGraphLaunch(exec, sa));
      ++launches;
    }
    const size_t slot = kSpan + (size_t)r % dyn;
    // a physical chunk that is free: the one that sits at no slot (the first such)
    size_t pc = nch;
    for (size_t i = kSpan; i < nch; ++i)
      if (P.mapped[i] < 0) {
        pc = i;
        break;
      }
    if (pc == nch) {
      std::printf("FAIL probe bookkeeping: no free chunk\n");
      return 2;
    }
    const double t0 = now_s();
    map_chunk(P, slot, pc);
    t_map += now_s() - t0;
    ++maps;
    const uint32_t salt = (uint32_t)r * 977u + 1u;
    k_pattern<<<32, 256, 0, sb>>>((uint32_t*)va_of(P, slot), kWords, salt);
    std::vector<uint32_t> host(kWords);
    RT(cudaMemcpyAsync(host.data(), (const void*)va_of(P, slot), kChunk, cudaMemcpyDeviceToHost, sb));
    RT(cudaStreamSynchronize(sb));
    for (size_t i = 0; i < kWords; ++i) bad_b += host[i] != (uint32_t)i * 2654435761u + salt;
    unmap_chunk(P, slot, pc);
    if (r % 50 == 49) {  // the graph's own chunks are not disturbed: counts so far
      RT(cudaStreamSynchronize(sa));
      bad_b += check_counts(P, 0, kSpan, launches * kIters);
    }
  }
  RT(cudaStreamSynchronize(sa));
  bad_b += check_counts(P, 0, kSpan, launches * kIters);
  RT(cudaDeviceSynchronize());
  {
    char d[256];
    std::snprintf(d, sizeof d, "%d rounds, %u graph replays over chunks 0..3, %zu maps (%.1f us each with SetAccess) and unmaps beside them: %zu wrong words, last error: %s",
                  rounds, launches, maps, maps ? 1e6 * t_map / maps : 0.0, bad_b, cudaGetErrorString(cudaGetLastError()));
    report(bad_b == 0, "M5b map and unmap while a graph replays", d);
  }

  // ---- unmap chunks 1..3, map other physical chunks there: they hold what is written, and the same graph
  // (no recapture) works on them
  for (size_t i = 1; i < kSpan; ++i) unmap_chunk(P, i, i);
  for (size_t i = 1; i < kSpan; ++i) {
    map_chunk(P, i, nch - i);  // other physical chunks than the ones that were there
    RT(cudaMemsetAsync((void*)va_of(P, i), 0, kChunk, sb));
  }
  RT(cudaMemsetAsync((void*)va_of(P, 0), 0, kChunk, sb));
  RT(cudaStreamSynchronize(sb));
  for (int i = 0; i < 20; ++i) RT(cudaGraphLaunch(exec, sa));
  RT(cudaStreamSynchronize(sa));
  {
    const size_t bad = check_counts(P, 0, kSpan, 20 * kIters);
    char d[128];
    std::snprintf(d, sizeof d, "chunks 1..3 backed by other physical chunks, written then replayed: %zu wrong words", bad);
    report(bad == 0, "M5c remapped chunks", d);
  }
  RT(cudaGraphExecDestroy(exec));
  RT(cudaGraphDestroy(graph));
  for (size_t i = 0; i < nch; ++i)
    if (P.mapped[i] >= 0) unmap_chunk(P, (size_t)P.mapped[i], i);
  for (auto x : P.h) cuMemRelease(x);
  CU(cuMemAddressFree(P.base, nch * kChunk));

  m6(create_gib);

  std::printf("%s: %d failed check%s\n", g_fail ? "FAIL" : "PASS", g_fail, g_fail == 1 ? "" : "s");
  return g_fail ? 1 : 0;
}
