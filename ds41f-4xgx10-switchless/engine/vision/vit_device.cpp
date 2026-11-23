#include "vision/vit_device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

#include "common/check.h"
#include "common/heap.h"

namespace vit {

namespace {
// /proc/meminfo's MemAvailable in bytes (0 when unreadable)
uint64_t mem_available() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  unsigned long long kb = 0;
  while (std::fgets(line, sizeof line, f))
    if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
  std::fclose(f);
  return kb << 10;
}
uint64_t device_free() {
  size_t f = 0, t = 0;
  CUDA_CHECK(cudaMemGetInfo(&f, &t));
  return f;
}
const char* kPatchW = "vision.patch_embed.proj.weight";
}  // namespace

VisionWeights::VisionWeights(const std::string& path, int rank) : path_(path) {
  file_ = std::make_unique<dsw::WeightFile>(path, rank, kRanks);
  const std::string why = check_vision_toc(file_->tensors());
  if (!why.empty()) die("%s: not rank %d's vision weight file: %s", path.c_str(), rank, why.c_str());
  digest_ = file_set_digest(file_->header(), file_->tensors());
}

VisionWeights::~VisionWeights() {
  for (void* p : dev_.owned) cudaFree(p);  // the arena itself belongs to the WeightFile
}

dsw::LoadStats VisionWeights::load(int threads) {
  if (arena_bytes_) die("%s: vision weights already loaded", path_.c_str());
  dsw::LoadOptions lo;
  lo.threads = threads;
  const dsw::LoadStats st = file_->load(lo);
  arena_bytes_ = file_->arena_bytes();
  auto at = [&](const std::string& n) { return static_cast<bf16*>(file_->operator[](n).data); };
  for (int i = 0; i < kLayers; i++) {
    const std::string p = "vision.blocks." + std::to_string(i) + ".";
    dev_.blocks.push_back({at(p + "attn.wqkv.weight"), at(p + "attn.wqkv.bias"), at(p + "attn.wo.weight"),
                           at(p + "attn.wo.bias"), at(p + "mlp.w1.weight"), at(p + "mlp.w2.weight"),
                           at(p + "norm1.weight"), at(p + "norm2.weight")});
  }
  dev_.patch_b = at("vision.patch_embed.proj.bias"), dev_.norm = at("vision.norm.weight");
  dev_.a1w = at("aligner.w1.weight"), dev_.a1b = at("aligner.w1.bias");
  dev_.a2w = at("aligner.w2.weight"), dev_.a2b = at("aligner.w2.bias");
  dev_.start = at("image_start"), dev_.newline = at("image_newline"), dev_.end = at("image_end");
  // the patch embedding: K = 588 is not a whole number of 64-wide stages, so one copy with zero columns to 640
  void* pad = nullptr;
  const size_t prow = (size_t)kPatchKPad * 2;
  CUDA_CHECK(cudaMalloc(&pad, prow * kDim));
  dev_.owned.push_back(pad);
  CUDA_CHECK(cudaMemset(pad, 0, prow * kDim));
  CUDA_CHECK(cudaMemcpy2D(pad, prow, scratch, (size_t)kPatchK * 2, (size_t)kPatchK * 2, kDim, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(scratch));
  dev_.patch_w = static_cast<bf16*>(pad);
  const std::vector<u16> tab = pixel_table();
  void* t = nullptr;
  CUDA_CHECK(cudaMalloc(&t, tab.size() * 2));
  dev_.owned.push_back(t);
  CUDA_CHECK(cudaMemcpy(t, tab.data(), tab.size() * 2, cudaMemcpyHostToDevice));
  dev_.table = static_cast<bf16*>(t);
  CUDA_CHECK(cudaDeviceSynchronize());
  extra_bytes_ = prow * kDim + tab.size() * 2;
  heap::release();  // the threads' free pages go back to the system with the staging
  held_.staging_peak = st.staging_peak, held_.staging_held = st.staging_held;
  held_.device_fell = dev0 - (int64_t)device_free(), held_.host_fell = host0 - (int64_t)mem_available();
  return st;
}

}  // namespace vit
