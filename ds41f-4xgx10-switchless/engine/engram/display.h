// The GB10's display reservation as CPU memory for the hot-row cache: a DRM dumb buffer made on the card, mapped
// write-combined through MAP_DUMB + mmap (writes are fast, reads are uncached and slow, so only a row store written
// often and read rarely belongs in it: RowCache's slot array). The buffer lives while the process holds the card's
// fd; destroy() releases munmap, GEM_CLOSE, close.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace engram {

class Display {
 public:
  Display() = default;
  Display(const Display&) = delete;
  Display& operator=(const Display&) = delete;
  ~Display() { destroy(); }

  // The first connected output of `card_name` ("card0") under sysfs_root/class/drm, "" when none: a screen uses the
  // reservation.
  static std::string connected_output(const std::string& sysfs_root, const std::string& card_name);

  // Opens `card`, refuses when an output is connected, makes the largest dumb buffer of at most max_bytes (max_bytes,
  // then lower in 16 MiB steps; the sweep is retried up to 3 times a second apart) and maps it. False with why() set
  // when any step fails; nothing is held then.
  bool create(const std::string& card, const std::string& sysfs_root, size_t max_bytes);
  // Keeps only the first `bytes` for use (every rank takes the smallest rank's size); never raises.
  void limit(size_t bytes) { if (bytes < cap_) cap_ = bytes; }

  uint8_t* data() const { return base_; }
  size_t capacity() const { return cap_; }  // bytes for use
  size_t size() const { return size_; }     // the whole buffer
  bool active() const { return base_ != nullptr; }
  const std::string& why() const { return why_; }
  void destroy();

 private:
  bool fail(const char* step, int err);
  int card_ = -1;
  uint32_t handle_ = 0;
  uint8_t* base_ = nullptr;
  size_t size_ = 0, cap_ = 0;
  std::string why_;
};

}  // namespace engram
