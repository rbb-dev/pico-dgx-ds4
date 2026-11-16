// This Spark's health for the dashboard (node_stats.h).
#include "server/node_stats.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace node_stats {
namespace {

bool slurp(const std::string& path, std::string& out) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return false;
  char buf[4096];
  out.clear();
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
  std::fclose(f);
  return true;
}

bool read_u64(const std::string& path, uint64_t& v) {
  std::string t;
  if (!slurp(path, t)) return false;
  char* end = nullptr;
  v = std::strtoull(t.c_str(), &end, 10);
  return end != t.c_str();
}

// NVML, opened on the first request; a missing library or a refused call leaves its fields null
struct Nvml {
  bool tried = false;
  void* dev = nullptr;
  int (*temp)(void*, int, unsigned*) = nullptr;
  int (*util)(void*, unsigned*) = nullptr;  // nvmlUtilization_t {gpu, memory}
  int (*clock)(void*, int, unsigned*) = nullptr;
  int (*clock_max)(void*, int, unsigned*) = nullptr;
  int (*power)(void*, unsigned*) = nullptr;
  int (*reasons)(void*, unsigned long long*) = nullptr;
  void open() {
    if (tried) return;
    tried = true;
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return;
    auto init = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
    auto by_index = reinterpret_cast<int (*)(unsigned, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
    if (!init || !by_index || init() != 0 || by_index(0, &dev) != 0) {
      dev = nullptr;
      return;
    }
    temp = reinterpret_cast<decltype(temp)>(dlsym(lib, "nvmlDeviceGetTemperature"));
    util = reinterpret_cast<decltype(util)>(dlsym(lib, "nvmlDeviceGetUtilizationRates"));
    clock = reinterpret_cast<decltype(clock)>(dlsym(lib, "nvmlDeviceGetClockInfo"));
    clock_max = reinterpret_cast<decltype(clock_max)>(dlsym(lib, "nvmlDeviceGetMaxClockInfo"));
    power = reinterpret_cast<decltype(power)>(dlsym(lib, "nvmlDeviceGetPowerUsage"));
    reasons = reinterpret_cast<decltype(reasons)>(dlsym(lib, "nvmlDeviceGetCurrentClocksEventReasons"));
    if (!reasons) reasons = reinterpret_cast<decltype(reasons)>(dlsym(lib, "nvmlDeviceGetCurrentClocksThrottleReasons"));
  }
};

// The NVMe's SMART log page (Get Log Page 02h) by one admin ioctl; refused without the capability (then the hwmon
// temperature alone).
struct NvmeAdminCmd {
  uint8_t opcode, flags;
  uint16_t rsvd1;
  uint32_t nsid, cdw2, cdw3;
  uint64_t metadata, addr;
  uint32_t metadata_len, data_len;
  uint32_t cdw10, cdw11, cdw12, cdw13, cdw14, cdw15;
  uint32_t timeout_ms, result;
};
static_assert(sizeof(NvmeAdminCmd) == 72, "struct nvme_admin_cmd");
constexpr unsigned long kNvmeAdmin = 0xC0484E41;  // _IOWR('N', 0x41, struct nvme_admin_cmd)

bool smart_page(uint8_t page[512]) {
  const int fd = ::open("/dev/nvme0", O_RDONLY);
  if (fd < 0) return false;
  NvmeAdminCmd c{};
  c.opcode = 0x02, c.nsid = 0xffffffffu, c.addr = reinterpret_cast<uint64_t>(page), c.data_len = 512;
  c.cdw10 = ((512 / 4 - 1) << 16) | 2;
  const bool ok = ioctl(fd, kNvmeAdmin, &c) == 0;
  ::close(fd);
  return ok;
}

uint64_t le64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

struct Sampler {
  int rank;
  Memory mem;
  std::string rails[2];
  Nvml nvml;
  uint64_t cpu_busy0 = 0, cpu_total0 = 0;

  std::string sample() {
    std::string j = "{\"rank\": " + std::to_string(rank);
    auto num = [&](const char* k, double v, bool ok) {
      char b[96];
      if (ok) std::snprintf(b, sizeof b, ", \"%s\": %.17g", k, v);
      else std::snprintf(b, sizeof b, ", \"%s\": null", k);
      j += b;
    };
    // memory
    std::string t;
    uint64_t avail = 0, total = 0;
    if (slurp("/proc/meminfo", t)) {
      if (const char* p = std::strstr(t.c_str(), "MemAvailable:")) avail = std::strtoull(p + 13, nullptr, 10) * 1024;
      if (const char* p = std::strstr(t.c_str(), "MemTotal:")) total = std::strtoull(p + 9, nullptr, 10) * 1024;
    }
    num("mem_available_bytes", (double)avail, avail > 0);
    num("mem_total_bytes", (double)total, total > 0);
    // the ACPI zones and their names
    std::string zc = "[", zn = "[";
    for (int i = 0; i < 64; ++i) {
      uint64_t mc;
      if (!read_u64("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp", mc)) break;
      std::string name;
      slurp("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/type", name);
      while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
      for (char& ch : name)
        if (ch == '"' || ch == '\\' || (unsigned char)ch < 32) ch = '_';
      char b[32];
      std::snprintf(b, sizeof b, "%s%.1f", i ? ", " : "", mc / 1000.0);
      zc += b;
      zn += std::string(i ? ", \"" : "\"") + name + "\"";
    }
    j += ", \"zones_c\": " + zc + "], \"zone_names\": " + zn + "]";
    // the GPU
    nvml.open();
    unsigned v = 0, u[2] = {0, 0};
    unsigned long long r = 0;
    j += ", \"gpu\": {\"temp_c\": ";
    j += nvml.dev && nvml.temp && nvml.temp(nvml.dev, 0, &v) == 0 ? std::to_string(v) : "null";
    j += ", \"util_pct\": ";
    j += nvml.dev && nvml.util && nvml.util(nvml.dev, u) == 0 ? std::to_string(u[0]) : "null";
    j += ", \"clock_mhz\": ";
    j += nvml.dev && nvml.clock && nvml.clock(nvml.dev, 0, &v) == 0 ? std::to_string(v) : "null";
    j += ", \"clock_max_mhz\": ";
    j += nvml.dev && nvml.clock_max && nvml.clock_max(nvml.dev, 0, &v) == 0 ? std::to_string(v) : "null";
    j += ", \"power_w\": ";
    if (nvml.dev && nvml.power && nvml.power(nvml.dev, &v) == 0) {
      char b[32];
      std::snprintf(b, sizeof b, "%.1f", v / 1000.0);
      j += b;
    } else {
      j += "null";
    }
    j += ", \"reasons\": ";
    if (nvml.dev && nvml.reasons && nvml.reasons(nvml.dev, &r) == 0) {
      // NVML's names; the idle bit (0x1) is the GPU resting, not a limit
      static const std::pair<unsigned long long, const char*> names[] = {
          {0x2, "applications_clocks_setting"}, {0x4, "sw_power_cap"}, {0x8, "hw_slowdown"}, {0x10, "sync_boost"},
          {0x20, "sw_thermal_slowdown"}, {0x40, "hw_thermal_slowdown"}, {0x80, "hw_power_brake_slowdown"},
          {0x100, "display_clock_setting"}};
      std::string a = "[";
      for (const auto& n : names)
        if (r & n.first) a += std::string(a.size() > 1 ? ", \"" : "\"") + n.second + "\"";
      j += a + "]";
    } else {
      j += "null";
    }
    j += "}";
    // the CPU: the load average, the busy share since the last request
    double load1 = -1;
    if (slurp("/proc/loadavg", t)) load1 = std::strtod(t.c_str(), nullptr);
    num("load1", load1, load1 >= 0);
    int cpus = 0;
    double busy_pct = -1;
    if (slurp("/proc/stat", t)) {
      uint64_t f[8] = {};
      if (std::sscanf(t.c_str(), "cpu %lu %lu %lu %lu %lu %lu %lu %lu", &f[0], &f[1], &f[2], &f[3], &f[4], &f[5], &f[6],
                      &f[7]) == 8) {
        uint64_t tot = 0;
        for (uint64_t x : f) tot += x;
        const uint64_t busy = tot - f[3] - f[4];
        if (cpu_total0 && tot > cpu_total0) busy_pct = 100.0 * (busy - cpu_busy0) / (tot - cpu_total0);
        cpu_busy0 = busy, cpu_total0 = tot;
      }
      for (const char* p = t.c_str(); (p = std::strstr(p, "\ncpu")) != nullptr; ++p)
        if (p[4] >= '0' && p[4] <= '9') ++cpus;
    }
    num("cpus", cpus, cpus > 0);
    num("cpu_busy_pct", busy_pct, busy_pct >= 0);
    // the NVMe: the SMART page when the ioctl is allowed, else the hwmon temperature; the root file system's space
    j += ", \"nvme\": {";
    alignas(8) uint8_t page[512];
    struct statvfs sv {};
    const bool root = statvfs("/", &sv) == 0;
    char b[512];
    if (smart_page(page)) {
      uint16_t k;
      std::memcpy(&k, page + 1, 2);
      std::snprintf(b, sizeof b,
                    "\"temp_c\": %d, \"wear_pct\": %u, \"available_spare_pct\": %u, \"spare_threshold_pct\": %u, "
                    "\"critical_warning\": %u, \"data_units_written\": %llu, \"data_unit_bytes\": 512000, "
                    "\"power_on_hours\": %llu, \"unsafe_shutdowns\": %llu, \"media_errors\": %llu",
                    (int)k - 273, page[5], page[3], page[4], page[0], (unsigned long long)le64(page + 48),
                    (unsigned long long)le64(page + 128), (unsigned long long)le64(page + 144),
                    (unsigned long long)le64(page + 160));
      j += b;
    } else {
      long tc = -1000000;
      for (int i = 0; i < 32 && tc == -1000000; ++i) {
        uint64_t mc;
        if (read_u64("/sys/class/nvme/nvme0/hwmon" + std::to_string(i) + "/temp1_input", mc)) tc = (long)mc;
      }
      if (tc != -1000000) std::snprintf(b, sizeof b, "\"temp_c\": %.1f, \"wear_pct\": null", tc / 1000.0);
      else std::snprintf(b, sizeof b, "\"temp_c\": null, \"wear_pct\": null");
      j += b;
    }
    if (root)
      std::snprintf(b, sizeof b, ", \"root_free_bytes\": %llu, \"root_total_bytes\": %llu",
                    (unsigned long long)sv.f_bavail * sv.f_frsize, (unsigned long long)sv.f_blocks * sv.f_frsize);
    else
      std::snprintf(b, sizeof b, ", \"root_free_bytes\": null, \"root_total_bytes\": null");
    j += b;
    j += "}";
    // the rails: InfiniBand data counters count 4-byte words
    j += ", \"rails\": [";
    for (int i = 0; i < 2; ++i) {
      const std::string c = "/sys/class/infiniband/" + rails[i] + "/ports/1/counters/";
      uint64_t rx, tx, er = 0, di = 0;
      const bool ok = read_u64(c + "port_rcv_data", rx) && read_u64(c + "port_xmit_data", tx);
      read_u64(c + "port_rcv_errors", er);
      read_u64(c + "port_xmit_discards", di);
      if (ok)
        std::snprintf(b, sizeof b, "%s{\"rx_bytes\": %llu, \"tx_bytes\": %llu, \"errors\": %llu, \"discards\": %llu}",
                      i ? ", " : "", (unsigned long long)rx * 4, (unsigned long long)tx * 4, (unsigned long long)er,
                      (unsigned long long)di);
      else
        std::snprintf(b, sizeof b, "%s{\"rx_bytes\": null, \"tx_bytes\": null, \"errors\": null, \"discards\": null}",
                      i ? ", " : "");
      j += b;
    }
    j += "]";
    std::snprintf(b, sizeof b,
                  ", \"memory\": {\"device_bytes\": %llu, \"weights_bytes\": %llu, \"work_bytes\": %llu, \"kv_bytes\": %llu}}",
                  (unsigned long long)mem.device_bytes, (unsigned long long)mem.weights_bytes,
                  (unsigned long long)mem.work_bytes, (unsigned long long)mem.kv_bytes);
    j += b;
    return j + "\n";
  }
};

void serve(int fd, Sampler* s) {
  // the small cores (Cortex-A725; cores 1-4 and 11-14 of a GB10), away from the cores the engine pins (5-8 comm
  // proxies, 9 the Engram worker, 15 the main thread)
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : {1, 2, 3, 4, 11, 12, 13, 14}) CPU_SET(c, &set);
  pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  for (;;) {
    const int c = accept(fd, nullptr, nullptr);
    if (c < 0) continue;
    timeval tv{1, 0};
    setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    char req[64];
    size_t n = 0;
    while (n < sizeof req) {
      const ssize_t k = recv(c, req + n, sizeof req - n, 0);
      if (k <= 0) break;
      n += (size_t)k;
      if (std::memchr(req, '\n', n)) break;
    }
    if (n >= 4 && std::memcmp(req, "GET\n", 4) == 0) {
      const std::string out = s->sample();
      size_t off = 0;
      while (off < out.size()) {
        const ssize_t k = send(c, out.data() + off, out.size() - off, MSG_NOSIGNAL);
        if (k <= 0) break;
        off += (size_t)k;
      }
    }
    ::close(c);
  }
}

}  // namespace

void start(int rank, const std::string& addr, int port, const Memory& mem, const std::string rails[2]) {
  if (port <= 0) return;
  const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port = htons((uint16_t)port);
  if (fd < 0 || inet_pton(AF_INET, addr.c_str(), &sa.sin_addr) != 1 ||
      bind(fd, reinterpret_cast<sockaddr*>(&sa), sizeof sa) != 0 || listen(fd, 4) != 0) {
    std::printf("rank %d: no node sampler: cannot listen on %s:%d (%s)\n", rank, addr.c_str(), port, std::strerror(errno));
    if (fd >= 0) ::close(fd);
    return;
  }
  auto* s = new Sampler{rank, mem, {rails[0], rails[1]}, {}, 0, 0};
  std::thread(serve, fd, s).detach();
  std::printf("rank %d: node sampler on %s:%d (rails %s, %s)\n", rank, addr.c_str(), port, rails[0].c_str(),
              rails[1].c_str());
}

}  // namespace node_stats
