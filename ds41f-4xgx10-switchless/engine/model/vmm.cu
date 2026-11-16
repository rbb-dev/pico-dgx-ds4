// The slot memory pool's CUDA backend (vmm.h): the driver's virtual memory calls on the current device.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#include "model/vmm.h"

namespace model::vmm {
namespace {

class CudaBackend : public Backend {
 public:
  explicit CudaBackend(int dev) : dev_(dev) {
    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = dev;
    acc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc_.location.id = dev;
    acc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  bool reserve(size_t bytes, uint64_t* va) override {
    CUdeviceptr p = 0;
    if (!ok(cuMemAddressReserve(&p, bytes, kChunk, 0, 0), "cuMemAddressReserve")) return false;
    *va = (uint64_t)p;
    return true;
  }
  void release_va(uint64_t va, size_t bytes) override { cuMemAddressFree((CUdeviceptr)va, bytes); }
  bool create(uint64_t* handle) override {
    CUmemGenericAllocationHandle h = 0;
    if (!ok(cuMemCreate(&h, kChunk, &prop_, 0), "cuMemCreate")) return false;
    *handle = (uint64_t)h;
    return true;
  }
  void destroy(uint64_t handle) override { cuMemRelease((CUmemGenericAllocationHandle)handle); }
  bool map(uint64_t va, uint64_t handle) override {
    return ok(cuMemMap((CUdeviceptr)va, kChunk, 0, (CUmemGenericAllocationHandle)handle, 0), "cuMemMap");
  }
  bool allow(uint64_t va, size_t bytes) override { return ok(cuMemSetAccess((CUdeviceptr)va, bytes, &acc_, 1), "cuMemSetAccess"); }
  void unmap(uint64_t va) override { cuMemUnmap((CUdeviceptr)va, kChunk); }
  void poison(uint64_t va) override {
    // the legacy stream's memset, finished before this returns: the model's non-blocking streams do not wait for it
    cudaMemset((void*)va, 0xFF, kChunk);
    cudaStreamSynchronize(0);
  }
  std::string error() const override { return err_; }

 private:
  bool ok(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    err_ = std::string(what) + ": " + (s ? s : "unknown error");
    return false;
  }
  int dev_;
  CUmemAllocationProp prop_{};
  CUmemAccessDesc acc_{};
  std::string err_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(std::string* err) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    if (err) *err = "no current CUDA device";
    return nullptr;
  }
  cudaFree(nullptr);  // the runtime's context, which the driver calls below run in
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  size_t gran = 0;
  const CUresult r = cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (r != CUDA_SUCCESS || gran == 0 || kChunk % gran) {
    if (err) *err = "allocation granularity " + std::to_string(gran) + " does not divide " + std::to_string(kChunk);
    return nullptr;
  }
  return std::make_unique<CudaBackend>(dev);
}

}  // namespace model::vmm
