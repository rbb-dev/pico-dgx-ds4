// Counting functions for the calls of tools/alloc_audit.py's list, put in front of every call in the engine's objects by the
// linker (--wrap=SYMBOL, server/CMakeLists.txt, dsgb10_rank only). Each counts the call in common/alloc_seal.h and makes it.
// Library-internal calls that do not go through an undefined symbol of an engine object (glibc's own mmap, the CUDA context) are
// not seen. dalloc, halloc and big_alloc are templates over the calls below.
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

#include <cstdlib>

#include "common/alloc_seal.h"

struct ibv_pd;
struct ibv_mr;

extern "C" {
cudaError_t __real_cudaMalloc(void** p, size_t bytes);
cudaError_t __real_cudaMallocHost(void** p, size_t bytes);
cudaError_t __real_cudaHostAlloc(void** p, size_t bytes, unsigned int flags);
cudaError_t __real_cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long flags);
CUresult __real_cuMemCreate(CUmemGenericAllocationHandle* h, size_t bytes, const CUmemAllocationProp* prop, unsigned long long flags);
CUresult __real_cuMemAddressReserve(CUdeviceptr* p, size_t bytes, size_t align, CUdeviceptr addr, unsigned long long flags);
void* __real_mmap(void* addr, size_t len, int prot, int flags, int fd, off_t off);
int __real_posix_memalign(void** p, size_t align, size_t bytes);
ibv_mr* __real_ibv_reg_mr(ibv_pd* pd, void* addr, size_t len, unsigned int access);

cudaError_t __wrap_cudaMalloc(void** p, size_t bytes) {
  alloc_seal::note("cudaMalloc", bytes, __builtin_return_address(0));
  return __real_cudaMalloc(p, bytes);
}
cudaError_t __wrap_cudaMallocHost(void** p, size_t bytes) {
  alloc_seal::note("cudaMallocHost", bytes, __builtin_return_address(0));
  return __real_cudaMallocHost(p, bytes);
}
cudaError_t __wrap_cudaHostAlloc(void** p, size_t bytes, unsigned int flags) {
  alloc_seal::note("cudaHostAlloc", bytes, __builtin_return_address(0));
  return __real_cudaHostAlloc(p, bytes, flags);
}
cudaError_t __wrap_cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long flags) {
  alloc_seal::note("cudaGraphInstantiate", 0, __builtin_return_address(0));  // the graph's size is not known to the caller
  return __real_cudaGraphInstantiate(e, g, flags);
}
CUresult __wrap_cuMemCreate(CUmemGenericAllocationHandle* h, size_t bytes, const CUmemAllocationProp* prop, unsigned long long flags) {
  alloc_seal::note("cuMemCreate", bytes, __builtin_return_address(0));
  return __real_cuMemCreate(h, bytes, prop, flags);
}
CUresult __wrap_cuMemAddressReserve(CUdeviceptr* p, size_t bytes, size_t align, CUdeviceptr addr, unsigned long long flags) {
  alloc_seal::note("cuMemAddressReserve", bytes, __builtin_return_address(0));
  return __real_cuMemAddressReserve(p, bytes, align, addr, flags);
}
void* __wrap_mmap(void* addr, size_t len, int prot, int flags, int fd, off_t off) {
  alloc_seal::note("mmap", len, __builtin_return_address(0));
  return __real_mmap(addr, len, prot, flags, fd, off);
}
int __wrap_posix_memalign(void** p, size_t align, size_t bytes) {
  alloc_seal::note("posix_memalign", bytes, __builtin_return_address(0));
  return __real_posix_memalign(p, align, bytes);
}
ibv_mr* __wrap_ibv_reg_mr(ibv_pd* pd, void* addr, size_t len, unsigned int access) {
  alloc_seal::note("ibv_reg_mr", len, __builtin_return_address(0));
  return __real_ibv_reg_mr(pd, addr, len, access);
}
}
