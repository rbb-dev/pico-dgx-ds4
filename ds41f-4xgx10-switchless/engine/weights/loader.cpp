#include "weights/loader.h"

#include <cuda_runtime.h>

#include <atomic>
#include <algorithm>
#include <thread>

#include "weights/io.h"

namespace dsw {

namespace {
// pinned host bytes held now and the most held at once, by this load's staging and verify buffers
struct HostHold {
  std::atomic<uint64_t> now{0}, peak{0};
  void* alloc(uint64_t n) {
    void* p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, n, cudaHostAllocDefault));
    const uint64_t v = now.fetch_add(n) + n;
    uint64_t m = peak.load();
    while (v > m && !peak.compare_exchange_weak(m, v)) {}
    return p;
  }
  void free(void* p, uint64_t n) {
    CUDA_CHECK(cudaFreeHost(p));
    now.fetch_sub(n);
  }
};
}  // namespace

WeightFile::WeightFile(const std::string& path, int rank, int world) : path_(path) {
  fd_ = open_or_die(path, O_RDONLY | O_DIRECT);
  uint64_t fsize = file_size_of(fd_);
  AlignedBuf hb(sizeof(Header));
  pread_all(fd_, hb.p, sizeof(Header), 0);
  std::memcpy(&header_, hb.p, sizeof(Header));
  const Header& h = header_;
  if (std::memcmp(h.magic, kMagic, 8) != 0) die("%s: not a dsw file", path.c_str());
  if (h.version < kMinVersion || h.version > kVersion || h.header_bytes != sizeof(Header) ||
      h.toc_entry_bytes != sizeof(TocEntry) || h.alignment != kAlign || h.file_bytes != fsize ||
      h.data_offset + h.data_bytes != fsize || h.data_offset % kAlign ||
      h.n_tensors * sizeof(TocEntry) > h.data_offset - h.toc_offset)
    die("%s: bad header (version %u, %llu bytes)", path.c_str(), h.version, (unsigned long long)fsize);
  if (rank >= 0) {
    const std::string why = rank_mismatch(h, uint32_t(rank), uint32_t(world));
    if (!why.empty()) die("%s: %s (a weight file serves only the rank it was cut for)", path.c_str(), why.c_str());
  }
  uint64_t toc_padded = h.data_offset - h.toc_offset;
  AlignedBuf tb(toc_padded);
  pread_all(fd_, tb.p, toc_padded, h.toc_offset);
  if (crc32c(tb.p, h.n_tensors * sizeof(TocEntry)) != h.toc_crc32c) die("%s: TOC CRC mismatch", path.c_str());
  auto* toc = reinterpret_cast<const TocEntry*>(tb.p);
  tensors_.resize(h.n_tensors);
  for (uint32_t i = 0; i < h.n_tensors; i++) {
    const TocEntry& e = toc[i];
    DeviceTensor& t = tensors_[i];
    t.name.assign(e.name, strnlen(e.name, sizeof e.name));
    t.dtype = DType(e.dtype);
    if (e.layout > uint32_t(Layout::L3) || (h.version < 2 && e.layout != 0)) die("%s: bad layout %u", e.name, e.layout);
    t.layout = Layout(e.layout);
    t.ndim = int(e.ndim);
    for (uint32_t d = 0; d < e.ndim && d < 4; d++) t.shape[d] = int64_t(e.shape[d]);
    t.bytes = e.bytes;
    t.file_offset = e.offset;
    t.offset = e.offset - h.data_offset;
    t.crc32c = e.crc32c;
    if (e.offset < h.data_offset || e.offset % kAlign || e.offset + e.bytes > fsize) die("%s: bad extent", e.name);
    if (!index_.emplace(t.name, i).second) die("%s: tensor %s twice", path.c_str(), e.name);
    if (old_embed_cut(t.name, h.version))
      die("%s: version %u holds %s whole, and this engine reads it cut by vocabulary rows (version %u): write the rank files "
          "again with dsw_convert",
          path.c_str(), h.version, t.name.c_str(), kEmbedCutVersion);
  }
}

WeightFile::~WeightFile() {
  if (arena_) cudaFree(arena_);
  if (fd_ >= 0) ::close(fd_);
}

const DeviceTensor* WeightFile::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

const DeviceTensor& WeightFile::operator[](const std::string& name) const {
  const DeviceTensor* t = find(name);
  if (!t) die("%s: no tensor %s", path_.c_str(), name.c_str());
  if (!t->data) die("%s: tensor %s is not loaded", path_.c_str(), name.c_str());
  return *t;
}

LoadStats WeightFile::load(const LoadOptions& opt) {
  if (arena_) die("%s: already loaded", path_.c_str());
  const uint64_t chunk = align_up(opt.chunk);

  // The chosen tensors, packed in file order (with no selection the arena is the file's data region
  // byte for byte); runs of tensors adjacent in the file are read as one extent.
  // dst: the caller's address (reads of a `dest` tensor: only `copy` of the `len` read bytes belong to it)
  struct Job {
    uint64_t file, arena, len;
    uint8_t* dst;
    uint64_t copy;
  };
  std::vector<Job> ext, jobs;
  std::vector<size_t> chosen, dest_idx;
  std::vector<uint8_t*> dest_ptr;
  uint64_t total = 0;
  for (size_t i = 0; i < tensors_.size(); i++) {
    DeviceTensor& t = tensors_[i];
    if (opt.select && !opt.select(t)) continue;
    const uint64_t padded = align_up(t.bytes);
    if (void* d = opt.dest ? opt.dest(t) : nullptr) {
      dest_idx.push_back(i), dest_ptr.push_back(static_cast<uint8_t*>(d));
      for (uint64_t o = 0; o < t.bytes; o += chunk) {
        const uint64_t len = std::min(chunk, padded - o);
        jobs.push_back({t.file_offset + o, 0, len, static_cast<uint8_t*>(d) + o, std::min(len, t.bytes - o)});
      }
      continue;
    }
    if (opt.limit && total + padded > opt.limit) break;
    t.offset = total;
    chosen.push_back(i);
    if (!ext.empty() && ext.back().file + ext.back().len == t.file_offset) ext.back().len += padded;
    else ext.push_back({t.file_offset, total, padded, nullptr, padded});
    total += padded;
  }
  if (!total && dest_idx.empty()) die("%s: no tensor selected", path_.c_str());
  for (auto& x : ext)
    for (uint64_t o = 0; o < x.len; o += chunk) {
      const uint64_t len = std::min(chunk, x.len - o);
      jobs.push_back({x.file + o, x.arena + o, len, nullptr, len});
    }
  if (total) CUDA_CHECK(cudaMalloc(&arena_, total));
  arena_bytes_ = total;
  uint64_t stage = 0;  // the longest read: no buffer is longer
  for (const Job& j : jobs) stage = std::max(stage, j.len);
  const size_t workers = std::min<size_t>(std::max(1, opt.threads), jobs.size());
  const int bufs = jobs.size() > workers ? 2 : 1;  // a second buffer only where a thread has a second read to overlap
  HostHold hold;

  double t0 = now_s();
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    void* buf[2] = {nullptr, nullptr};
    cudaEvent_t ev[2];
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (int b = 0; b < bufs; b++) {
      buf[b] = hold.alloc(stage);
      CUDA_CHECK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming));
    }
    bool pending[2] = {false, false};
    int cur = 0;
    for (size_t k; (k = next.fetch_add(1)) < jobs.size();) {
      const Job& j = jobs[k];
      if (pending[cur]) CUDA_CHECK(cudaEventSynchronize(ev[cur]));
      pread_all(fd_, buf[cur], j.len, j.file);
      CUDA_CHECK(cudaMemcpyAsync(j.dst ? j.dst : static_cast<uint8_t*>(arena_) + j.arena, buf[cur], j.copy,
                                 cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaEventRecord(ev[cur], st));
      pending[cur] = true;
      cur = (cur + 1) % bufs;
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    for (int b = 0; b < bufs; b++) {
      hold.free(buf[b], stage);
      CUDA_CHECK(cudaEventDestroy(ev[b]));
    }
    CUDA_CHECK(cudaStreamDestroy(st));
  };
  std::vector<std::thread> pool;
  for (size_t t = 0; t < workers; t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();

  LoadStats s;
  s.bytes = total;  // the arena's; a `dest` tensor's bytes are not counted
  s.seconds = now_s() - t0;
  s.gbps = total / s.seconds / 1e9;
  for (size_t i : chosen) tensors_[i].data = static_cast<uint8_t*>(arena_) + tensors_[i].offset;
  std::vector<bool> to_dest(tensors_.size(), false);
  for (size_t k = 0; k < dest_idx.size(); k++) tensors_[dest_idx[k]].data = dest_ptr[k], to_dest[dest_idx[k]] = true;

  if (opt.verify_crc || opt.verify_dest) {
    double t1 = now_s();
    uint64_t vbuf = 0;  // the longest piece a CRC check copies back
    for (const DeviceTensor& t : tensors_) vbuf = std::max(vbuf, std::min(chunk, t.bytes));
    void* host = hold.alloc(vbuf);
    for (size_t i = 0; i < tensors_.size(); i++) {
      DeviceTensor& t = tensors_[i];
      if (!t.data || !(opt.verify_crc || to_dest[i])) continue;
      uint32_t crc = 0;
      for (uint64_t o = 0; o < t.bytes; o += chunk) {
        uint64_t n = std::min(chunk, t.bytes - o);
        CUDA_CHECK(cudaMemcpy(host, static_cast<uint8_t*>(t.data) + o, n, cudaMemcpyDeviceToHost));
        crc = crc32c(host, n, crc);
      }
      if (crc != t.crc32c) die("%s: %s CRC mismatch after load", path_.c_str(), t.name.c_str());
    }
    hold.free(host, vbuf);
    s.verify_seconds = now_s() - t1;
  }
  if (opt.drop_page_cache) posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
  s.staging_peak = hold.peak.load(), s.staging_held = hold.now.load();
  return s;
}

}  // namespace dsw
