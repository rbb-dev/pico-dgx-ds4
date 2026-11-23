// Loading a rank's weight file into one device arena: O_DIRECT reads by several threads into pinned
// staging buffers (two per thread, or one when there are no more reads than threads), each copied to the device while the
// next is read. Reads format versions 1 to 3; DeviceTensor::layout says how each tensor's bytes are arranged (dsw.h).
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "weights/dsw.h"

namespace dsw {

struct DeviceTensor {
  std::string name;
  DType dtype = DType::F32;
  Layout layout = Layout::Natural;
  int ndim = 0;
  int64_t shape[4] = {0, 0, 0, 0};  // natural (logical) shape, whatever the layout
  uint64_t bytes = 0;
  uint64_t file_offset = 0;  // absolute, in the file
  uint64_t offset = 0;       // in the arena, once loaded
  uint32_t crc32c = 0;
  void* data = nullptr;  // device pointer, nullptr until loaded (or not selected)
};

struct LoadOptions {
  int threads = 4;               // concurrent O_DIRECT reads
  uint64_t chunk = 64ull << 20;  // bytes per read
  uint64_t limit = 0;            // stop once the arena would exceed this many bytes (tests); 0 = no limit
  bool verify_crc = false;       // afterwards copy every loaded tensor back and check its CRC
  // load only the tensors this accepts (in file order, packed into a smaller arena); empty = all
  std::function<bool(const DeviceTensor&)> select;
  // where a selected tensor goes: a caller-supplied device address with room for its `bytes` (the tensor is then
  // read straight there, takes no space in the arena, and its `data` is that address), or nullptr for the arena
  std::function<void*(const DeviceTensor&)> dest;
  bool verify_dest = false;      // afterwards check the CRC of the tensors that went to a `dest` (not the arena's)
  // afterwards tell the kernel the file's pages are not wanted (O_DIRECT reads leave none; this drops any that another
  // reader or writer of the file left in the page cache)
  bool drop_page_cache = false;
};

struct LoadStats {
  uint64_t bytes = 0;
  double seconds = 0, gbps = 0;      // streaming file -> device
  double verify_seconds = 0;         // when verify_crc
  // pinned host staging: the most held at once during the load (only as many workers as there are reads, each with a
  // second buffer only when it has a second read to overlap, each buffer as long as the longest read), and what is
  // still held when load() returns (0: every buffer is freed)
  uint64_t staging_peak = 0, staging_held = 0;
};

class WeightFile {
 public:
  // Reads and validates the header and TOC; with rank >= 0, also that the file is rank `rank` of `world`'s
  // (dsw::rank_mismatch), dying with the path and both ranks if not.
  explicit WeightFile(const std::string& path, int rank = -1, int world = 0);
  ~WeightFile();
  WeightFile(const WeightFile&) = delete;
  WeightFile& operator=(const WeightFile&) = delete;

  LoadStats load(const LoadOptions& opt = {});  // allocates the arena and streams the data in

  const DeviceTensor& operator[](const std::string& name) const;  // dies if missing or not loaded
  const DeviceTensor* find(const std::string& name) const;        // nullptr if missing
  const std::vector<DeviceTensor>& tensors() const { return tensors_; }
  const Header& header() const { return header_; }
  void* arena() const { return arena_; }
  uint64_t arena_bytes() const { return arena_bytes_; }

 private:
  std::string path_;
  int fd_ = -1;
  Header header_{};
  std::vector<DeviceTensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
  void* arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
};

}  // namespace dsw
