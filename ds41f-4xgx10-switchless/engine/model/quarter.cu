// The device's side of a cache piece's quarters (quarter_map.h): the quartering and its inverse, and the exact sums. Each kernel
// moves or adds 4-byte units, one a thread, through the map in quarter_map.h; no kernel that computes a token is touched.
#include <cuda_runtime.h>

#include <algorithm>

#include "common/check.h"
#include "model/quarter_map.h"

namespace model::quarter {

namespace {

constexpr int kThreads = 256;
constexpr unsigned kMaxBlocks = 128;

// unit u (4 bytes) of rank q's quarter layout, with entries from `written` on read as zero
__device__ __forceinline__ uint32_t quarter_unit(const Piece& p, int q, uint32_t u, uint32_t written) {
  const Src s = quarter_src(q, u * 4);
  return s.entry < written ? *(const uint32_t*)(p.a[s.a] + s.off) : 0u;
}
// unit u of the whole piece's image layout, likewise
__device__ __forceinline__ uint32_t image_unit(const Piece& p, uint32_t u, uint32_t written) {
  const uint32_t ib = u * 4;
  int a = 0;
  while (a < 3 && ib >= image_off(a + 1)) ++a;
  const uint32_t lb = ib - (uint32_t)image_off(a);
  return lb / arr_bytes(a) < written ? *(const uint32_t*)(p.a[a] + lb) : 0u;
}

__global__ void k_quarter(Piece src, uint8_t* dst, int q, uint32_t written) {
  for (uint32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < kQuarterBytes / 4; u += gridDim.x * blockDim.x)
    ((uint32_t*)dst)[u] = quarter_unit(src, q, u, written);
}

// per unit of the piece: the quarter that owns it and its byte there
__global__ void k_unquarter(const uint8_t* quarters, PieceOut dst) {
  for (uint32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < kImageBytes / 4; u += gridDim.x * blockDim.x) {
    const uint32_t ib = u * 4;
    int a = 0;
    while (a < 3 && ib >= image_off(a + 1)) ++a;
    const uint32_t lb = ib - (uint32_t)image_off(a), sz = arr_bytes(a), e = lb / sz;
    const size_t qb = (size_t)owner_of(a, e) * kQuarterBytes + quarter_off(a) + (size_t)local_of(a, e) * sz + lb % sz;
    *(uint32_t*)(dst.a[a] + lb) = *(const uint32_t*)(quarters + qb);
  }
}

// the block's partial sum added into *out (an addition modulo 2^64: no order of the blocks changes it)
__device__ __forceinline__ void add_block(uint64_t acc, unsigned long long* out) {
  __shared__ uint64_t part[kThreads / 32];
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x) return;
  uint64_t s = 0;
  for (int w = 0; w < kThreads / 32; ++w) s += part[w];
  atomicAdd(out, (unsigned long long)s);
}

__global__ void k_sum_exact(const uint64_t* p, size_t n, uint64_t seed, unsigned long long* out) {
  uint64_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    acc += sum_term(i, p[i], seed);
  add_block(acc, out);
}

// grid.y = the view: 0-3 a quarter's layout (seed: its rank), 4 the whole image
__global__ void k_piece_sums(Piece src, uint32_t written, unsigned long long* out) {
  const int v = blockIdx.y;
  const uint32_t words = (v < 4 ? kQuarterBytes : kImageBytes) / 8;
  uint64_t acc = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) {
    const uint32_t lo = v < 4 ? quarter_unit(src, v, 2 * i, written) : image_unit(src, 2 * i, written);
    const uint32_t hi = v < 4 ? quarter_unit(src, v, 2 * i + 1, written) : image_unit(src, 2 * i + 1, written);
    acc += sum_term(i, (uint64_t)hi << 32 | lo, v < 4 ? (uint64_t)v : kWholeSeed);
  }
  add_block(acc, out + v);
}

__global__ void k_trailer(const uint64_t* five, uint32_t rank, uint32_t handle, uint32_t list, uint32_t piece, uint32_t written, uint64_t* dst) {
  if (threadIdx.x || blockIdx.x) return;
  uint64_t w[10], q[4] = {five[0], five[1], five[2], five[3]};
  trailer_words(q, five[4], rank, handle, list, piece, written, w);
  for (int i = 0; i < 10; ++i) dst[i] = w[i];
}

unsigned blocks_for(size_t units) { return (unsigned)std::min<size_t>(kMaxBlocks, (units + kThreads - 1) / kThreads); }

}  // namespace

void quarter(const Piece& src, uint8_t* dst, int q, int written, CUstream_st* st) {
  k_quarter<<<blocks_for(kQuarterBytes / 4), kThreads, 0, st>>>(src, dst, q, (uint32_t)written);
  CUDA_CHECK(cudaGetLastError());
}

void unquarter(const uint8_t* quarters, const PieceOut& dst, CUstream_st* st) {
  k_unquarter<<<blocks_for(kImageBytes / 4), kThreads, 0, st>>>(quarters, dst);
  CUDA_CHECK(cudaGetLastError());
}

void sum_exact(const void* p, size_t bytes, uint64_t seed, uint64_t* out, CUstream_st* st) {
  CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof *out, st));
  k_sum_exact<<<blocks_for(bytes / 8), kThreads, 0, st>>>((const uint64_t*)p, bytes / 8, seed, (unsigned long long*)out);
  CUDA_CHECK(cudaGetLastError());
}

void piece_sums(const Piece& src, int written, uint64_t* out, CUstream_st* st) {
  CUDA_CHECK(cudaMemsetAsync(out, 0, 5 * sizeof *out, st));
  k_piece_sums<<<dim3(blocks_for(kImageBytes / 8), 5), kThreads, 0, st>>>(src, (uint32_t)written, (unsigned long long*)out);
  CUDA_CHECK(cudaGetLastError());
}

void trailer_device(const uint64_t* five, uint32_t rank, uint32_t handle, uint32_t list, uint32_t piece, uint32_t written, uint8_t* dst,
                    size_t zero, CUstream_st* st) {
  if (zero) CUDA_CHECK(cudaMemsetAsync(dst, 0, zero, st));
  k_trailer<<<1, 32, 0, st>>>(five, rank, handle, list, piece, written, (uint64_t*)dst);
  CUDA_CHECK(cudaGetLastError());
}

// the table: every kernel this file launches, by name, with the call that launches it once (the selftest names each `<<<` of this file in this body)
int warm(const Piece& piece, const PieceOut& piece_out, uint8_t* quarter_dst, const uint8_t* quarters, uint64_t* five, uint64_t* five_st, CUstream_st* sg, CUstream_st* st) {
  int n = 0;
  const auto one = [&n](const char* name, auto&& launch) {
    (void)name;
    launch();
    ++n;
  };
  one("k_quarter", [&] { quarter(piece, quarter_dst, 0, (int)kEntries, sg); });
  one("k_piece_sums", [&] { piece_sums(piece, (int)kEntries, five, sg); });
  one("k_trailer", [&] { trailer_device(five, 0, 0, 0, 0, (uint32_t)kEntries, quarter_dst + kQuarterBytes, kTrailerBytes, sg); });
  one("k_sum_exact", [&] { sum_exact(quarter_dst, 4096, 0, five, sg); });
  one("k_unquarter", [&] { unquarter(quarters, piece_out, st); });
  piece_sums(piece, (int)kEntries, five_st, st);  // the read-back's sums on the model's stream (the same kernel, its own stream)
  return n;
}

void sums_read(const uint64_t* rec, uint64_t* out, CUstream_st* st) {
#ifdef PICO_OLD_RULES
  (void)st;
  CUDA_CHECK(cudaMemcpy(out, rec, 5 * sizeof *out, cudaMemcpyDeviceToHost));
#else
  CUDA_CHECK(cudaMemcpyAsync(out, rec, 5 * sizeof *out, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
#endif
}

}  // namespace model::quarter
