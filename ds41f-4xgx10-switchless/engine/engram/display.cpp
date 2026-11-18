#include "engram/display.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace engram {

// uapi drm.h / drm_mode.h, spelled out so that no libdrm is needed
struct DrmCreateDumb {
  uint32_t height, width, bpp, flags, handle, pitch;
  uint64_t size;
};
struct DrmMapDumb {
  uint32_t handle, pad;
  uint64_t offset;
};
struct DrmGemClose {
  uint32_t handle, pad;
};
static_assert(sizeof(DrmCreateDumb) == 32 && sizeof(DrmMapDumb) == 16 && sizeof(DrmGemClose) == 8, "DRM ioctl layouts");
constexpr unsigned long kIoctlCreateDumb = 0xC02064B2ul, kIoctlMapDumb = 0xC01064B3ul, kIoctlGemClose = 0x40086409ul;
constexpr uint32_t kPitch = 16384;  // width 4096 at 32 bits a pixel: a buffer is height * kPitch bytes
constexpr size_t kStep = 16u << 20;

std::string Display::connected_output(const std::string& sysfs_root, const std::string& card_name) {
  const std::string dir = sysfs_root + "/class/drm", pre = card_name + "-";
  DIR* d = opendir(dir.c_str());
  if (!d) return "";
  std::string found;
  while (const dirent* e = readdir(d)) {
    if (strncmp(e->d_name, pre.c_str(), pre.size())) continue;
    FILE* f = fopen((dir + "/" + e->d_name + "/status").c_str(), "r");
    if (!f) continue;
    char b[32] = {0};
    if (fgets(b, sizeof b, f) && !strncmp(b, "connected", 9)) found = e->d_name;
    fclose(f);
    if (!found.empty()) break;
  }
  closedir(d);
  return found;
}

bool Display::fail(const char* step, int err) {
  why_ = std::string(step) + ": " + strerror(err);
  destroy();
  return false;
}

bool Display::create(const std::string& card, const std::string& sysfs_root, size_t max_bytes) {
  destroy();
  why_.clear();
  card_ = open(card.c_str(), O_RDWR | O_CLOEXEC);
  if (card_ < 0) return fail(("open " + card).c_str(), errno);
  const size_t slash = card.rfind('/');
  const std::string name = slash == std::string::npos ? card : card.substr(slash + 1);
  const std::string out = connected_output(sysfs_root, name);
  if (!out.empty()) {
    why_ = "output " + out + " is connected";
    destroy();
    return false;
  }
  if (max_bytes < kStep) {
    why_ = "the cache is smaller than the reservation's step (16 MiB)";
    destroy();
    return false;
  }
  int err = 0;
  for (int attempt = 0; attempt < 3 && !handle_; ++attempt) {
    if (attempt) sleep(1);
    for (size_t b = max_bytes & ~(kStep - 1); b >= kStep; b -= kStep) {
      DrmCreateDumb c{};
      c.width = 4096, c.bpp = 32, c.height = (uint32_t)((b + kPitch - 1) / kPitch);
      if (ioctl(card_, kIoctlCreateDumb, &c) == 0) {
        handle_ = c.handle, size_ = c.size;
        break;
      }
      err = errno;
      if (err != ENOMEM && err != EINVAL) break;  // not a size problem: lower sizes fail alike
    }
    if (err != ENOMEM && err != EINVAL) break;
  }
  if (!handle_) return fail("CREATE_DUMB", err ? err : ENOMEM);
  DrmMapDumb m{};
  m.handle = handle_;
  if (ioctl(card_, kIoctlMapDumb, &m)) return fail("MAP_DUMB", errno);
  void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, card_, (off_t)m.offset);
  if (p == MAP_FAILED) return fail("mmap", errno);
  base_ = static_cast<uint8_t*>(p);
  cap_ = size_;
  return true;
}

void Display::destroy() {
  if (base_) munmap(base_, size_);
  if (card_ >= 0) {
    if (handle_) {
      DrmGemClose gc{handle_, 0};
      ioctl(card_, kIoctlGemClose, &gc);
    }
    close(card_);
  }
  base_ = nullptr, card_ = -1, handle_ = 0, size_ = cap_ = 0;
}

}  // namespace engram
