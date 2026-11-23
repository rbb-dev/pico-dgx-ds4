// The vision weights of this rank on the device: the rank's vision weight file loaded with O_DIRECT reads into one
// device arena (dsw::WeightFile), checked against what the rank must hold, plus the two small buffers the kernels
// need outside the file (the patch embedding zero-padded to 640 columns, the 256-entry pixel table).
#pragma once
#include <memory>
#include <string>

#include "vision/vit.h"
#include "vision/vit_file.h"

namespace vit {

class VisionWeights {
 public:
  // Opens `path` as rank `rank` of four (dies with the path and both ranks when the file was cut for another) and
  // checks its table of contents against expected_file(): dies in plain words on a file that is not this rank's
  // vision weights. Reads no tensor data.
  VisionWeights(const std::string& path, int rank);
  ~VisionWeights();
  static constexpr uint64_t kStageChunk = 16ull << 20;  // one read; two threads x two buffers = 64 MiB of staging at most
  static constexpr int kStageThreads = 2;
  VisionWeights(const VisionWeights&) = delete;
  VisionWeights& operator=(const VisionWeights&) = delete;

  // What the four ranks compare (file_set_digest): equal on the four files of one conversion.
  uint64_t digest() const { return digest_; }

  // Streams every tensor into the device arena and builds the kernels' view. Call once. The reads are O_DIRECT through
  // small pinned staging buffers (kStageChunk, at most two threads), all freed before it returns; the patch
  // embedding is read into a scratch buffer and padded once, so no tensor is held twice; the file's pages are dropped.
  // Afterwards the vision path holds device_bytes() on the device and nothing on the host (held()).
  dsw::LoadStats load(int threads = 4);

  // The kernels' view of the weights (after load()): pointers into the arena, plus the two small buffers.
  const DevWeights& weights() const { return dev_; }

  // Device bytes held once loaded: the arena and the two small buffers.
  uint64_t device_bytes() const { return arena_bytes_ + extra_bytes_; }
  // What the vision path holds beyond the weights' own bytes, as measured around load(): device memory the driver
  // reports free before and after (cudaMemGetInfo), the host's MemAvailable before and after, and the pinned staging
  // still held at return. Readings, in bytes; a reading of another process's change is inside them.
  struct Held {
    uint64_t staging_peak = 0, staging_held = 0;
    int64_t device_fell = 0, host_fell = 0;  // free before - free after (positive: it fell)
  };
  const Held& held() const { return held_; }
  int tensors() const { return int(file_->tensors().size()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  std::unique_ptr<dsw::WeightFile> file_;
  uint64_t digest_ = 0, arena_bytes_ = 0, extra_bytes_ = 0;
  DevWeights dev_;
  Held held_;
};

}  // namespace vit
