// dsw_dest_test: LoadOptions::dest, a tensor aimed at a caller's device buffer. Needs one GPU (cudaMalloc).
//   dsw_dest_test [DIR]    a small synthetic weight file in DIR (default .; read with O_DIRECT, so not on tmpfs) with
//                          tensors of 5,000, 9,000 and 4,096 bytes; the middle one is aimed into a caller buffer
//                          between guard bytes, in 4 KiB chunks and in one read. Checks: the bytes land at the address
//                          and not one byte beyond the tensor's size (the read is padded to 4 KiB), `data` is that
//                          address, the CRC check (verify_crc and verify_dest) passes there, the arena holds only the
//                          other tensors, and a tensor whose bytes are corrupted at the destination fails the CRC.
// Exit 0 when every check passes.
#include <cuda_runtime.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "weights/io.h"
#include "weights/loader.h"

using namespace dsw;

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++fails;
}

constexpr uint64_t kBytes[3] = {5000, 9000, 4096};

std::vector<uint8_t> payload(int i) {
  std::vector<uint8_t> v(kBytes[i]);
  for (size_t k = 0; k < v.size(); k++) v[k] = uint8_t(k * 31 + i * 17 + (k >> 8));
  return v;
}

void make_file(const std::string& path) {
  const uint64_t data_off = 2 * kAlign;
  uint64_t off[3], at = data_off;
  for (int i = 0; i < 3; i++) off[i] = at, at += align_up(kBytes[i]);
  std::vector<uint8_t> f(at, 0);
  Header h{};
  std::memcpy(h.magic, kMagic, 8);
  h.version = kVersion;
  h.header_bytes = sizeof(Header);
  h.rank = 0, h.world = 4;
  h.n_tensors = 3;
  h.toc_entry_bytes = sizeof(TocEntry);
  h.toc_offset = kAlign;
  h.data_offset = data_off;
  h.data_bytes = at - data_off;
  h.file_bytes = at;
  h.alignment = kAlign;
  std::snprintf(h.model, sizeof h.model, "dsw_dest_test");
  TocEntry e[3]{};
  for (int i = 0; i < 3; i++) {
    std::snprintf(e[i].name, sizeof e[i].name, "t%d.weight", i);
    e[i].dtype = uint32_t(DType::F8_E4M3);
    e[i].ndim = 1;
    e[i].shape[0] = kBytes[i];
    e[i].offset = off[i];
    e[i].bytes = kBytes[i];
    e[i].kind = uint32_t(SliceKind::Replicated);
    const std::vector<uint8_t> p = payload(i);
    std::memcpy(&f[off[i]], p.data(), p.size());
    e[i].crc32c = crc32c(p.data(), p.size());
  }
  std::memcpy(&f[kAlign], e, sizeof e);
  h.toc_crc32c = crc32c(e, sizeof e);
  std::memcpy(&f[0], &h, sizeof h);
  FILE* out = std::fopen(path.c_str(), "wb");
  if (!out || std::fwrite(f.data(), 1, f.size(), out) != f.size() || std::fclose(out) != 0)
    die("%s: cannot write", path.c_str());
}

std::vector<uint8_t> dev_read(const void* p, size_t n) {
  std::vector<uint8_t> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost));
  return h;
}

void run(const std::string& path, uint64_t chunk, bool verify_all) {
  const std::string tag = "chunk " + std::to_string(chunk) + (verify_all ? ", verify_crc" : ", verify_dest");
  constexpr size_t kGuard = 8192, kRoom = 12288;  // the tensor's padded read length
  uint8_t* buf = nullptr;
  CUDA_CHECK(cudaMalloc(&buf, kGuard + kRoom + kGuard));
  CUDA_CHECK(cudaMemset(buf, 0xAB, kGuard + kRoom + kGuard));
  // the guard is written on the default stream and the loader copies on streams that do not wait for it: the guard must be
  // in place before the load, or a busy GPU lands it over the tensor's bytes
  CUDA_CHECK(cudaDeviceSynchronize());
  uint8_t* dst = buf + kGuard;
  WeightFile wf(path);
  LoadOptions lo;
  lo.chunk = chunk;
  lo.threads = 2;
  lo.verify_crc = verify_all, lo.verify_dest = !verify_all;
  lo.dest = [&](const DeviceTensor& t) -> void* { return t.name == "t1.weight" ? dst : nullptr; };
  wf.load(lo);  // dies on a CRC mismatch
  const DeviceTensor& t1 = wf["t1.weight"];
  expect(t1.data == dst, tag + ": t1's data is the caller's address");
  const std::vector<uint8_t> want = payload(1), got = dev_read(dst, kBytes[1]);
  expect(got == want, tag + ": t1's bytes are at the address, byte for byte");
  const std::vector<uint8_t> before = dev_read(buf, kGuard), after = dev_read(dst + kBytes[1], kRoom - kBytes[1] + kGuard);
  bool intact = true;
  for (uint8_t b : before) intact = intact && b == 0xAB;
  for (uint8_t b : after) intact = intact && b == 0xAB;
  expect(intact, tag + ": not a byte written outside t1's " + std::to_string(kBytes[1]) + " bytes");
  expect(wf.arena_bytes() == align_up(kBytes[0]) + align_up(kBytes[2]),
         tag + ": the arena holds t0 and t2 only (" + std::to_string(wf.arena_bytes()) + " bytes, not " +
             std::to_string(align_up(kBytes[0]) + align_up(kBytes[1]) + align_up(kBytes[2])) + ")");
  expect(dev_read(wf["t0.weight"].data, kBytes[0]) == payload(0) && dev_read(wf["t2.weight"].data, kBytes[2]) == payload(2),
         tag + ": t0 and t2 are in the arena, byte for byte");
  CUDA_CHECK(cudaFree(buf));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  const std::string path = dir + "/dsw_dest_test.dsw";
  make_file(path);
  run(path, 4096, true);
  run(path, 4096, false);
  run(path, 64ull << 20, true);
  run(path, 64ull << 20, false);
  unlink(path.c_str());
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
