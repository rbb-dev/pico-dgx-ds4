// dsw_stage_test: what the loader holds on the host while it loads and after it returns (needs one GPU).
//   dsw_stage_test [DIR]   a small synthetic weight file in DIR (default .; read with O_DIRECT, so not on tmpfs) of three
//                          tensors that are one 24,576-byte extent. Checks, from LoadStats and from the bytes: staging is
//                          as long as the longest read (not the 64 MiB chunk) and there are no more threads than reads
//                          (4 threads asked, one read: one buffer of 24,576 B, not 4 x 2 x 64 MiB), a second buffer only
//                          where a thread has a second read, the CRC check's host buffer is not longer than the
//                          largest tensor, nothing is held when load() returns, and the bytes are right with
//                          drop_page_cache.
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

constexpr uint64_t kRead = 24576;  // the three tensors' padded sizes in one extent

void one(const std::string& path, int threads, uint64_t chunk, bool verify, uint64_t want_peak, const std::string& tag) {
  WeightFile wf(path);
  LoadOptions lo;
  lo.threads = threads, lo.chunk = chunk, lo.verify_crc = verify, lo.drop_page_cache = true;
  const LoadStats st = wf.load(lo);
  // threads that finish early free before the others allocate, so the peak is at most the sum, and one buffer at least
  expect(st.staging_peak <= want_peak && st.staging_peak >= std::min<uint64_t>(want_peak, 4096),
         tag + ": staging at most " + std::to_string(want_peak) + " B (got " + std::to_string(st.staging_peak) + ")");
  expect(st.staging_held == 0, tag + ": nothing held on return (" + std::to_string(st.staging_held) + " B)");
  expect(dev_read(wf["t0.weight"].data, kBytes[0]) == payload(0) && dev_read(wf["t2.weight"].data, kBytes[2]) == payload(2),
         tag + ": bytes right");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  const std::string path = dir + "/dsw_stage_test.dsw";
  make_file(path);
  // one read: one thread, one buffer of that read, whatever the threads asked and the chunk allowed
  one(path, 4, 64ull << 20, false, kRead, "4 threads, one read of 24,576 B in a 64 MiB chunk");
  // 6 reads of 4,096 B, 2 threads: each thread has more reads to overlap: two buffers of 4,096 B each
  one(path, 2, 4096, false, 2 * 2 * 4096, "2 threads, six reads of 4,096 B");
  // 6 reads, 8 threads asked: six threads, one read each, one buffer each
  one(path, 8, 4096, false, 6 * 4096, "8 threads asked, six reads");
  // the CRC check's host buffer is as long as the largest tensor's piece (9,000 B), not the chunk, and is freed
  one(path, 4, 64ull << 20, true, kRead, "verify_crc");
  unlink(path.c_str());
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
