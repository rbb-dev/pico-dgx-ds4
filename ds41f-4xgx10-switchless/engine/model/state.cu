// Sessions' saved state (model.h state_*, state.h): the device's side. The bookkeeping (which chunk number holds what, the
// disk tier's images, the order eviction takes pieces in, the kept index) is detail::StoreCore (store_core.h, host only); this
// file is what the core asks of the device in the middle of an operation (EngineDev: StoreDev) and the device copies that
// follow one. Every copy is queued on the model's stream, so a block freed on the host can be handed out again at once. Blocks
// come from the pool reserved at startup (state.h): taking a snapshot inside the decode loop allocates nothing.
#include "model/state.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <sys/statvfs.h>

#include "comm/host_gather.h"
#include "common/alloc_seal.h"
#include "model/pool_rules.h"
#include "model/quarter_map.h"
#include "model/spec.h"
#include "model/spill.h"

namespace model {

using namespace detail;

namespace qm = ::model::quarter;  // EngineDev has members named quarter and unquarter

namespace {

constexpr size_t kArr[4] = {256, 32, 64, 4};  // bytes per entry: KV, KV scales, index key, key scales

// The store's constants are the core's own copies (store_geom.h); each must equal the header that owns it.
static_assert(sgeom::kChunkShift == attn::kChunkShift && sgeom::kChunkEntries == attn::kChunkEntries, "the chunk number's entries");
static_assert(sgeom::kSnapRows == kSnapRows, "a window bookmark's positions");
static_assert(sgeom::kMapLookahead == kMapLookahead, "the lookahead");
static_assert(sgeom::kBk == (size_t)attn::kChunkEntries * 256, "a bookmark number's bytes");
static_assert(sgeom::kBk == imgrows::kNumberBytes, "an image row's number is a bookmark's: the first array's slice");
static_assert(quarter::arr_bytes(0) == kArr[0] && quarter::arr_bytes(1) == kArr[1] && quarter::arr_bytes(2) == kArr[2] &&
                  quarter::arr_bytes(3) == kArr[3], "the quarter map's arrays");
static_assert(kMaxSlots == 64 && kMaxUnits == 128, "the core's arrays");
static_assert(StoreCore::kNoLine == Model::Impl::kNoLine, "no session");

double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// A window snapshot's block and an end snapshot's: the draft module's windows follow the layers' rows
size_t snap_bytes(bool spec) { return kSnapBytes + (spec ? kSpecSnapBytes : 0); }
size_t end_bytes(bool spec) { return kEndBytes + (spec ? kSpecEndBytes : 0); }
// A bookmark's image is this rank's slice and the trailer's used words in one block: with the draft module's windows too, by the compiler's word
static_assert(qm::bk_trailer_off(kSnapBytes + kSpecSnapBytes) + qm::kTrailerUsed <= sgeom::kBookmarkImageBytes, "a window bookmark's slice and trailer fit its image block");
static_assert(qm::bk_trailer_off(kEndBytes + kSpecEndBytes) + qm::kTrailerUsed <= sgeom::kEndImageBytes, "the end bookmark's slice and trailer fit its image block");

// A block's checksum for the disk tier's round trip: every 8-byte word mixed with its index, summed.
// kSumCtas CTAs each sum a stride of it; the last one to finish adds their partial sums in order and
// stores the result in *out (pinned memory) as one store, so nothing is zeroed first and the sum does
// not depend on which CTA finished when. Launches are serialized on the model's stream, so they share
// the partials and the counter.
constexpr int kSumCtas = 16;
__device__ uint64_t g_sum_part[kSumCtas];
__device__ unsigned g_sum_done;

__global__ void block_sum(const uint64_t* p, size_t n, uint64_t* out) {
  __shared__ uint64_t part[32];
  uint64_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t x = p[i] + i * 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    acc += x ^ (x >> 27);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x >= 32) return;
  acc = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0;
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (threadIdx.x) return;
  g_sum_part[blockIdx.x] = acc;
  __threadfence();
  if (atomicAdd(&g_sum_done, 1u) != gridDim.x - 1) return;
  __threadfence();  // the last CTA: every partial sum is in
  uint64_t s = 0;
  for (unsigned j = 0; j < gridDim.x; ++j) s += ((volatile uint64_t*)g_sum_part)[j];
  *out = s;
  g_sum_done = 0;
}

// over the bytes the disk tier copies: the block rounded to the pool's grain
void sum_block(const void* p, size_t bytes, uint64_t* out, cudaStream_t st) {
  block_sum<<<kSumCtas, 1024, 0, st>>>((const uint64_t*)p, Pool::round(bytes) / 8, out);
}

// the byte offset of array a's slice in a cache entry's image
size_t img_slice(int a) {
  size_t o = 0;
  for (int j = 0; j < a; ++j) o += (size_t)attn::kChunkEntries * kArr[j];
  return o;
}

}  // namespace

// ---- the engine's StoreDev -------------------------------------------------------------------------------------------
//
// The device work the store does in the middle of an operation. The disk tier's file and its staging are made at the first disk_open; `gather` is the one image
// at a time between the pool and the file (state_reserve).

static_assert(kSiteKeep == comm::kCallKeepIndex && kSiteCheckpoint == comm::kCallCheckpointIndex && kSiteDiskRoom == comm::kCallDiskRoom && kSiteUnspill == comm::kCallUnspill &&
                  kSiteJoin == comm::kCallJoin, "the store's sites are the host all-gather's ids");

class EngineDev : public StoreDev {
 public:
  explicit EngineDev(Model::Impl& impl) : m(impl) {}
  Model::Impl& m;
  uint8_t* gather = nullptr;
  std::unique_ptr<SpillFile> file;

  void fill(uint32_t c, int byte) override {
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemsetAsync(m.pool_arr[a] + ((size_t)c << attn::kChunkShift) * kKvEntryBytes[a], byte,
                                 (size_t)attn::kChunkEntries * kKvEntryBytes[a], m.st));
  }
  // entries [0, n) of number `from` into number `to`, all four arrays (a fork's partial number; copy on write)
  void copy_entries(uint32_t to, uint32_t from, int n) override {
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemcpyAsync(m.pool_arr[a] + ((size_t)to << attn::kChunkShift) * kKvEntryBytes[a],
                                 m.pool_arr[a] + ((size_t)from << attn::kChunkShift) * kKvEntryBytes[a], (size_t)n * kKvEntryBytes[a],
                                 cudaMemcpyDeviceToDevice, m.st));
  }
  // unit u's rows from the lists (the guard past their ends); the stream waits first, so no queued copy of the mirror reads an
  // entry being rewritten
  void rows_set(int u, const Nums* lists) override {
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    for (int k = 0; k < kKvSlots; ++k) {
      uint32_t* row = m.lists_h + ((size_t)k * m.nunits + u) * m.list_len;
      std::fill(row, row + m.list_len, 0u);
      if (lists) std::copy(lists[k].begin(), lists[k].end(), row);
      CUDA_CHECK(cudaMemcpyAsync(m.lists_d + ((size_t)k * m.nunits + u) * m.list_len, row, m.list_len * sizeof(uint32_t),
                                 cudaMemcpyHostToDevice, m.st));
    }
  }
  void rows_append(int u, int k, size_t i0, const uint32_t* numbers, size_t n) override {
    const size_t row = ((size_t)k * m.nunits + u) * m.list_len;
    for (size_t i = 0; i < n; ++i) m.lists_h[row + i0 + i] = numbers[i];  // an entry no queued copy reads yet (rewrites wait: rows_set)
    CUDA_CHECK(cudaMemcpyAsync(m.lists_d + row + i0, m.lists_h + row + i0, n * sizeof(uint32_t), cudaMemcpyHostToDevice, m.st));
  }
  uint64_t image_out(uint32_t c, size_t off) override {
    const size_t bytes = m.chunk_image();
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemcpyAsync(gather + img_slice(a), m.pool_arr[a] + ((size_t)c << attn::kChunkShift) * kArr[a],
                                 (size_t)attn::kChunkEntries * kArr[a], cudaMemcpyDeviceToDevice, m.st));
    sum_block(gather, bytes, file->sum(off, false), m.st);
    return file->write(gather, bytes, off, m.st);
  }
  uint64_t bk_out(const Nums& numbers, size_t bytes, size_t off) override {
    m.bk_copy(numbers, 0, gather, bytes, false);
    sum_block(gather, bytes, file->sum(off, false), m.st);
    return file->write(gather, bytes, off, m.st);
  }
  void image_in(size_t off, uint32_t c) override {
    file->read(gather, m.chunk_image(), off, m.st);
    sum_block(gather, m.chunk_image(), file->sum(off, true), m.st);
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemcpyAsync(m.pool_arr[a] + ((size_t)c << attn::kChunkShift) * kArr[a], gather + img_slice(a),
                                 (size_t)attn::kChunkEntries * kArr[a], cudaMemcpyDeviceToDevice, m.st));
  }
  void bk_in(size_t off, const Nums& numbers, size_t bytes) override {
    file->read(gather, bytes, off, m.st);
    sum_block(gather, bytes, file->sum(off, true), m.st);
    m.bk_copy(numbers, 0, gather, bytes, true);
  }
  // the writes not yet flushed leave on the stream they were queued on (the image stream's, or the model's under PICO_OLD_RULES)
  void flush_pending() { file->flush(file->pending_stream() ? file->pending_stream() : m.st); }
  void disk_flush() override { flush_pending(); }
  void disk_settle(uint64_t seq) override {
    flush_pending();
    file->wait(seq);
  }
  void sync() override { CUDA_CHECK(cudaStreamSynchronize(m.st)); }
  void disk_sync() override { file->sync(); }
  void disk_keep() override { file->keep(); }
  size_t disk_open(const std::string& base, size_t bytes, size_t staging, bool reuse) override {
    file = std::make_unique<SpillFile>(base, bytes, staging, reuse);
    return file->size();
  }
  size_t disk_size() override { return file->size(); }
  // free space of the file's directory less the 32 GiB margin the start keeps (server/main.cpp)
  size_t disk_room() override {
    const std::string p = file->path();
    const size_t cut = p.rfind('/');
    struct statvfs v;
    if (statvfs(cut == std::string::npos ? "." : p.substr(0, cut).c_str(), &v) != 0) return 0;
    const uint64_t fr = (uint64_t)v.f_bavail * v.f_frsize, margin = 32ull << 30;
    return fr > margin ? (size_t)(fr - margin) : 0;
  }
  // the staged writes are in the file and the stream holds no checksum or read of the slots the grow moves
  bool disk_grow(size_t bytes) override {
    flush_pending();
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    if (sg) CUDA_CHECK(cudaStreamSynchronize(sg));
    return file->grow(bytes);
  }
  // the staged writes are in the file and the stream holds no checksum or read of the slots the shrink moves
  bool disk_shrink(size_t bytes) override {
    flush_pending();
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    if (sg) CUDA_CHECK(cudaStreamSynchronize(sg));
    return file->shrink(bytes);
  }
  std::string disk_path() override { return file->path(); }
  uint64_t sum(size_t off, bool back) override { return *file->sum(off, back); }
  void sum_set(size_t off, uint64_t value) override { *file->sum(off, false) = value; }
  // every rank cuts at the smallest first bad position (each reaches this in the same command)
  int agree_min(int pos, uint32_t site) override {
    int32_t all[4];
    const int32_t mine = pos;
    m.host_ring->allgather_host(&mine, all, sizeof mine, site);
    for (const int32_t b : all) pos = std::min<int>(pos, b);
    return pos;
  }
  void agree_all(int v, uint32_t site, int all[4]) override {
    int32_t a[4];
    const int32_t mine = v;
    m.host_ring->allgather_host(&mine, a, sizeof mine, site);
    for (int r = 0; r < 4; ++r) all[r] = a[r];
  }
  void agree_words(const uint64_t mine[4], uint32_t site, uint64_t all[16]) override { m.host_ring->allgather_host(mine, all, 4 * sizeof(uint64_t), site); }
  std::time_t now() override { return std::time(nullptr); }

  // ---- the split: a piece's quarters. Slot s = (number << 2) | quarter lives in the quarter number's four slices, at
  // quarter * 2,048 entries of each array; the sums recorded at a split (five words a slot) are in a device array sized for the pool's
  // largest count of numbers. The buffers are made at the start (state_reserve, with --kv-split cache), itemised on the start table
  // as pool_rules.h's kSplit, never at first use.
  uint64_t* d_sums = nullptr;
  size_t sums_slots = 0;
  uint8_t* qbuf = nullptr;    // one quarter, as quarter() makes it
  uint8_t* qpack = nullptr;   // this rank's quarters of up to kBatch pieces, to send
  uint8_t* exbuf = nullptr;   // the four ranks' quarters of up to kBatch pieces, piece by piece (what unquarter reads)
  uint8_t* gat = nullptr;     // the all-gather's output: rank by rank
  uint64_t* d_tmp = nullptr;

  static constexpr int kBatch = (int)model::rules::kJoinBatch;  // pieces a join exchanges at once

  // ---- the imager: a second stream that is not the model's, its own scratch, and one event per image command. Nothing on it waits
  // for the model's stream except the first read of a command (one way: it waits for the work queued before the command, so the bytes it
  // reads are the ones the steps wrote), and nothing on the model's stream waits for it.
  cudaStream_t sg = nullptr;
  cudaEvent_t ev_m = nullptr;        // recorded on the model's stream at the first call of a command; sg waits for it
  uint8_t* gq = nullptr;             // a quarter and its trailer: kDiskBytes
  uint8_t* gather_g = nullptr;       // a bookmark's bytes, then its image block
  size_t gather_g_bytes = 0;
  uint64_t* d_five = nullptr;        // the five sum words of the image being made
  bool cmd_open = false;             // the model's stream has been waited for in this command
  struct End {
    uint64_t seq;
    cudaEvent_t ev;
    std::chrono::steady_clock::time_point t;
  };
  std::deque<End> ends;  // image_end's events, oldest first
  uint64_t done_seq = 0;

  void begin_cmd() {
    if (cmd_open) return;
    CUDA_CHECK(cudaEventRecord(ev_m, m.st));
    CUDA_CHECK(cudaStreamWaitEvent(sg, ev_m, 0));
    cmd_open = true;
  }

  void prepare() {
    if (qbuf) return;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sg, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_m, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&gq, qm::kDiskBytes));
    CUDA_CHECK(cudaMalloc(&d_five, 5 * sizeof(uint64_t)));
    gather_g_bytes = Pool::round(std::max(end_bytes(true), snap_bytes(true))) + sgeom::kEndImageBytes;
    CUDA_CHECK(cudaMalloc(&gather_g, gather_g_bytes));
    CUDA_CHECK(cudaMalloc(&qbuf, qm::kQuarterBytes));
    CUDA_CHECK(cudaMalloc(&qpack, (size_t)kBatch * qm::kQuarterBytes));
    CUDA_CHECK(cudaMalloc(&exbuf, (size_t)kBatch * 4 * qm::kQuarterBytes));
    CUDA_CHECK(cudaMalloc(&gat, (size_t)kBatch * 4 * qm::kQuarterBytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)kBatch * 5 * sizeof(uint64_t)));
    sums_slots = (size_t)model::rules::kSplitMaxNumbers * 4;
    CUDA_CHECK(cudaMalloc(&d_sums, sums_slots * 5 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_sums, 0, sums_slots * 5 * sizeof(uint64_t)));
    warm_kernels();
  }
  // every kernel the imager and the read-back use is launched once here, at the start, with the model's stream empty, on scratch held already (a piece laid over
  // gather_g, gq, exbuf, d_five, d_tmp: no pool block, nothing written to the file). An image kernel's first launch inside a decoding run does not return on any rank,
  // so no store kernel has its first launch there. The names of the kernels are the tables' (quarter.cu warm(), and "block_sum" below).
  void warm_kernels() {
    const double t0 = wall_s();
    qm::Piece p;
    qm::PieceOut po;
    for (int a = 0; a < 4; ++a) p.a[a] = gather_g + qm::image_off(a), po.a[a] = gather_g + qm::image_off(a);
    int n = qm::warm(p, po, gq, exbuf, d_five, d_tmp, sg, m.st);
    sum_block(gq, qm::kDiskBytes, d_tmp + 5, sg), ++n;  // "block_sum"
    uint8_t* host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, 4096, cudaHostAllocDefault));
    CUDA_CHECK(cudaMemcpyAsync(host, gq, 4096, cudaMemcpyDeviceToHost, sg));
    CUDA_CHECK(cudaStreamSynchronize(sg));
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    cudaFreeHost(host);
    std::printf("rank %d: the store's kernels ran once (%d kernels, %.1f ms)\n", m.rank, n, 1e3 * (wall_s() - t0));
  }
  void need_scratch() {
    if (!qbuf) prepare();  // a test that never reserved the store
    if ((size_t)m.core->calloc->numbers() * 4 > sums_slots)
      die("state: the pool has %u numbers, more than the split's sums cover (%zu)", m.core->calloc->numbers(), sums_slots / 4);
  }
  qm::Piece piece_of(uint32_t c) const {
    qm::Piece p;
    for (int a = 0; a < 4; ++a) p.a[a] = m.pool_arr[a] + ((size_t)c << attn::kChunkShift) * kArr[a];
    return p;
  }
  qm::PieceOut piece_out(uint32_t c) const {
    qm::PieceOut p;
    for (int a = 0; a < 4; ++a) p.a[a] = m.pool_arr[a] + ((size_t)c << attn::kChunkShift) * kArr[a];
    return p;
  }
  uint8_t* slot_slice(uint32_t slot, int a) const {
    return m.pool_arr[a] + ((size_t)(slot >> 2) << attn::kChunkShift) * kArr[a] + (size_t)(slot & 3) * qm::kQuarterEntries * kArr[a];
  }
  void quarter(uint32_t c, uint32_t slot, int written) override {
    need_scratch();
    qm::quarter(piece_of(c), qbuf, m.rank, written, m.st);
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemcpyAsync(slot_slice(slot, a), qbuf + qm::quarter_off(a), (size_t)qm::kQuarterEntries * kArr[a],
                                 cudaMemcpyDeviceToDevice, m.st));
    qm::piece_sums(piece_of(c), written, d_sums + (size_t)slot * 5, m.st);
  }
  // The quarters of `slots` (this rank's, in its slots) to every rank and the other ranks' to this one: one all-gather of the batch,
  // then each piece's four quarters side by side for unquarter. Every rank calls with the same slots in the same command.
  void exchange(const std::vector<uint32_t>& slots) override {
    need_scratch();
    const size_t B = slots.size();
    if (!B || B > (size_t)kBatch) die("state: an exchange of %zu pieces (at most %d)", B, kBatch);
    for (size_t i = 0; i < B; ++i)
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemcpyAsync(qpack + i * qm::kQuarterBytes + qm::quarter_off(a), slot_slice(slots[i], a),
                                   (size_t)qm::kQuarterEntries * kArr[a], cudaMemcpyDeviceToDevice, m.st));
    gather_batch(B);
  }
  // the batch in qpack (this rank's quarters) to every rank, and every rank's quarters piece by piece into exbuf
  void gather_batch(size_t B) {
    comm::allgather(m.ch, qpack, gat, B * qm::kQuarterBytes, m.st);
    for (int r = 0; r < 4; ++r)
      for (size_t i = 0; i < B; ++i)
        CUDA_CHECK(cudaMemcpyAsync(exbuf + (i * 4 + r) * qm::kQuarterBytes, gat + ((size_t)r * B + i) * qm::kQuarterBytes, qm::kQuarterBytes,
                                   cudaMemcpyDeviceToDevice, m.st));
  }
  // ---- a quarter's image on the disk tier: the quarter (laid out as a slot holds it) and the trailer, kDiskBytes in all
  uint64_t slot_out(uint32_t slot, size_t off, int handle, int list, int piece, int written) override {
    need_scratch();
    slot_get(slot, gather);
    uint64_t rec[5];
    qm::sums_read(d_sums + (size_t)slot * 5, rec, m.st);  // on the stream: the sums kernel was queued there, and the stream is non-blocking
    qm::Trailer t;
    for (int q = 0; q < 4; ++q) t.qsum[q] = rec[q];
    t.wsum = rec[4], t.rank = (uint32_t)m.rank, t.handle = (uint32_t)handle, t.list = (uint32_t)list, t.piece = (uint32_t)piece, t.written = (uint32_t)written;
    uint8_t tb[qm::kTrailerBytes];
    qm::trailer_pack(t, tb);
    CUDA_CHECK(cudaMemcpyAsync(gather + qm::kQuarterBytes, tb, qm::kTrailerBytes, cudaMemcpyHostToDevice, m.st));
    sum_block(gather, qm::kDiskBytes, file->sum(off, false), m.st);
    return file->write(gather, qm::kDiskBytes, off, m.st);
  }
  bool quarter_read(size_t off, int idx, qm::Trailer* t) override {
    need_scratch();
    file->read(gather, qm::kDiskBytes, off, m.st);
    sum_block(gather, qm::kDiskBytes, file->sum(off, true), m.st);
    CUDA_CHECK(cudaMemcpyAsync(qpack + (size_t)idx * qm::kQuarterBytes, gather, qm::kQuarterBytes, cudaMemcpyDeviceToDevice, m.st));
    uint8_t tb[qm::kTrailerBytes];
    CUDA_CHECK(cudaMemcpyAsync(tb, gather + qm::kQuarterBytes, qm::kTrailerBytes, cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    return qm::trailer_unpack(tb, t);
  }
  void exchange_staged(int n) override {
    need_scratch();
    gather_batch((size_t)n);
  }
  void compare_sums(const std::vector<uint32_t>& numbers, const std::vector<qm::Trailer>& trailers, std::vector<uint8_t>* bad) override {
    need_scratch();
    bad->assign(numbers.size(), 0);
    for (size_t i = 0; i < numbers.size(); ++i) qm::piece_sums(piece_of(numbers[i]), (int)attn::kChunkEntries, d_tmp + i * 5, m.st);
    std::vector<uint64_t> now(numbers.size() * 5);
    CUDA_CHECK(cudaMemcpyAsync(now.data(), d_tmp, now.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    for (size_t i = 0; i < numbers.size(); ++i) {
      bool eq = now[i * 5 + 4] == trailers[i].wsum;
      for (int q = 0; q < 4; ++q) eq = eq && now[i * 5 + q] == trailers[i].qsum[q];
      (*bad)[i] = !eq;
    }
  }
  void unquarter(const std::vector<uint32_t>& numbers) override {
    need_scratch();
    for (size_t i = 0; i < numbers.size(); ++i)
      qm::unquarter(exbuf + i * 4 * qm::kQuarterBytes, piece_out(numbers[i]), m.st);
  }
  // the sums of piece `c` against the five words recorded for `slot`
  bool sums_match(const uint64_t* now, uint32_t slot, bool whole_only) {
    uint64_t rec[5];
    qm::sums_read(d_sums + (size_t)slot * 5, rec, m.st);
    for (int i = whole_only ? 4 : 0; i < 5; ++i)
      if (now[i] != rec[i]) return false;
    return true;
  }
  void compare(const std::vector<uint32_t>& slots, const std::vector<uint32_t>& numbers, std::vector<uint8_t>* bad) override {
    need_scratch();
    bad->assign(slots.size(), 0);
    for (size_t i = 0; i < numbers.size(); ++i) qm::piece_sums(piece_of(numbers[i]), (int)attn::kChunkEntries, d_tmp + i * 5, m.st);
    std::vector<uint64_t> now(numbers.size() * 5);
    CUDA_CHECK(cudaMemcpyAsync(now.data(), d_tmp, now.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaStreamSynchronize(m.st));  // one sync for the batch
    for (size_t i = 0; i < slots.size(); ++i) (*bad)[i] = !sums_match(&now[i * 5], slots[i], false);
  }
  // A bookmark's quarters. A bookmark is replicated (every rank holds the same bytes), so rank r keeps the
  // bytes [r * Q, (r + 1) * Q) of it, Q = a quarter of its bytes rounded up to 16, laid in its slots one slot's capacity at a time
  // (a window bookmark one slot, the end bookmark three). The bookmark's own bytes are read through the gather buffer (bk_copy), the
  // sum of the whole bookmark is recorded in the first slot's word 4. A slot holds its bytes in the four arrays' slices at the
  // positions a piece's quarter does (quarter_off), so the slot map and the exchange treat both alike.
  size_t bk_len(size_t nslots) const {
    const StoreCore& c = *m.core;
    return nslots == c.bk_slots(c.snap_bytes()) ? c.snap_bytes() : c.end_bytes();
  }
  static size_t bk_q(size_t bytes) { return (bytes / 4 + 15) / 16 * 16; }
  void slot_put(uint32_t slot, const uint8_t* src) {  // one slot's bytes (kQuarterBytes) from a contiguous buffer
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemcpyAsync(slot_slice(slot, a), src + qm::quarter_off(a), (size_t)qm::kQuarterEntries * kArr[a], cudaMemcpyDeviceToDevice, m.st));
  }
  void slot_get(uint32_t slot, uint8_t* dst) {
    for (int a = 0; a < 4; ++a)
      CUDA_CHECK(cudaMemcpyAsync(dst + qm::quarter_off(a), slot_slice(slot, a), (size_t)qm::kQuarterEntries * kArr[a], cudaMemcpyDeviceToDevice, m.st));
  }
  void quarter_bk(const Nums& numbers, const std::vector<uint32_t>& slots) override {
    need_scratch();
    const size_t bytes = bk_len(slots.size()), Q = bk_q(bytes);
    if (Q > slots.size() * qm::kQuarterBytes) die("state: a bookmark's quarter of %zu bytes does not fit %zu slots", Q, slots.size());
    m.bk_copy(numbers, 0, gather, bytes, false);
    const size_t mine = (size_t)m.rank * Q;
    for (size_t j = 0; j < slots.size(); ++j) {
      const size_t from = mine + j * qm::kQuarterBytes, rem = Q > j * qm::kQuarterBytes ? Q - j * qm::kQuarterBytes : 0;
      CUDA_CHECK(cudaMemsetAsync(qbuf, 0, qm::kQuarterBytes, m.st));
      if (from < bytes && rem) CUDA_CHECK(cudaMemcpyAsync(qbuf, gather + from, std::min({qm::kQuarterBytes, rem, bytes - from}), cudaMemcpyDeviceToDevice, m.st));
      slot_put(slots[j], qbuf);
    }
    qm::sum_exact(gather, bytes & ~(size_t)7, qm::kWholeSeed, d_sums + (size_t)slots[0] * 5 + 4, m.st);
  }
  bool join_bk(const Nums& numbers, const std::vector<uint32_t>& slots) override {
    need_scratch();
    const size_t n = slots.size(), bytes = bk_len(n), Q = bk_q(bytes);
    if (n > (size_t)kBatch) die("state: a bookmark of %zu slots (at most %d)", n, kBatch);
    for (size_t j = 0; j < n; ++j) slot_get(slots[j], qpack + j * qm::kQuarterBytes);
    comm::allgather(m.ch, qpack, gat, n * qm::kQuarterBytes, m.st);  // every rank's quarter, rank by rank, slot after slot
    for (int r = 0; r < 4; ++r) {
      const size_t from = (size_t)r * Q;
      if (from < bytes)
        CUDA_CHECK(cudaMemcpyAsync(gather + from, gat + (size_t)r * n * qm::kQuarterBytes, std::min(Q, bytes - from), cudaMemcpyDeviceToDevice, m.st));
    }
    m.bk_copy(numbers, 0, gather, bytes, true);
    qm::sum_exact(gather, bytes & ~(size_t)7, qm::kWholeSeed, d_tmp, m.st);
    uint64_t now = 0, rec = 0;
    CUDA_CHECK(cudaMemcpyAsync(&now, d_tmp, sizeof now, cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaMemcpyAsync(&rec, d_sums + (size_t)slots[0] * 5 + 4, sizeof rec, cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaStreamSynchronize(m.st));  // one sync a bookmark
    return now == rec;
  }
  bool sum_exact(uint32_t c, uint32_t slot, int written) override {
    need_scratch();
    qm::piece_sums(piece_of(c), written, d_tmp, m.st);
    uint64_t now[5];
    CUDA_CHECK(cudaMemcpyAsync(now, d_tmp, sizeof now, cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    return sums_match(now, slot, true);
  }

  // ---- the imager's calls (store_core.h): every one is queued on sg
  bool image_room(size_t bytes) override { return file->room(bytes); }
  // piece c's quarter (the layout a slot holds), its five sums, and the trailer built on the device from them: no read of the sums by the host
  uint64_t image_quarter(uint32_t c, size_t off, int handle, int list, int piece, int written) override {
    need_scratch();
    begin_cmd();
    qm::quarter(piece_of(c), gq, m.rank, written, sg);
    qm::piece_sums(piece_of(c), written, d_five, sg);
    qm::trailer_device(d_five, (uint32_t)m.rank, (uint32_t)handle, (uint32_t)list, (uint32_t)piece, (uint32_t)written, gq + qm::kQuarterBytes,
                       qm::kTrailerBytes, sg);
    sum_block(gq, qm::kDiskBytes, file->sum(off, false), sg);
    return file->write(gq, qm::kDiskBytes, off, sg);
  }
  // a bookmark's bytes through gather_g; this rank's slice [r * Q, (r + 1) * Q) and the trailer's words after it, in one image block
  uint64_t image_bookmark(const Nums& numbers, size_t bytes, size_t off, int handle, int list, int piece) override {
    need_scratch();
    begin_cmd();
    const size_t Q = qm::bk_slice(bytes), imgb = list == Store::kBkEnd ? sgeom::kEndImageBytes : sgeom::kBookmarkImageBytes;
    uint8_t* img = gather_g + Pool::round(bytes);
    if (qm::bk_trailer_off(bytes) + qm::kTrailerUsed > imgb || Pool::round(bytes) + imgb > gather_g_bytes)
      die("state: a bookmark of %zu bytes does not fit its image block", bytes);
    m.bk_copy(numbers, 0, gather_g, bytes, false, sg);
    for (int r = 0; r < 4; ++r) {  // the four slices' sums (the slice's owner is the seed), then the whole's
      const size_t from = (size_t)r * Q, len = from < bytes ? std::min(Q, bytes - from) & ~(size_t)7 : 0;
      if (len) qm::sum_exact(gather_g + from, len, (uint64_t)r, d_five + r, sg);
      else CUDA_CHECK(cudaMemsetAsync(d_five + r, 0, sizeof(uint64_t), sg));
    }
    qm::sum_exact(gather_g, bytes & ~(size_t)7, qm::kWholeSeed, d_five + 4, sg);
    CUDA_CHECK(cudaMemsetAsync(img, 0, imgb, sg));
    const size_t mine = (size_t)m.rank * Q, len = mine < bytes ? std::min(Q, bytes - mine) : 0;
    if (len) CUDA_CHECK(cudaMemcpyAsync(img, gather_g + mine, len, cudaMemcpyDeviceToDevice, sg));
    qm::trailer_device(d_five, (uint32_t)m.rank, (uint32_t)handle, (uint32_t)list, (uint32_t)piece, (uint32_t)bytes, img + qm::bk_trailer_off(bytes), 0, sg);
    sum_block(img, imgb, file->sum(off, false), sg);
    return file->write(img, imgb, off, sg);
  }
  void image_end(uint64_t seq) override {
    cmd_open = false;
    flush_pending();
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(e, sg));
    ends.push_back({seq, e, std::chrono::steady_clock::now()});
  }
  uint64_t image_done() override {
#ifdef DSGB10_TEST_FAULTS
    // Fault seam (DSGB10_IMAGE_SLOW=rank:R,ms:N, a fault build): rank R reads a command as done only N ms after its image_end, as if its stream
    // were slow. The ranks' settle is the smallest reading: nothing is marked anywhere before rank R's stream ends.
    if (const char* e = std::getenv("DSGB10_IMAGE_SLOW")) {
      int rk = -1, ms = 0;
      std::sscanf(e, "rank:%d,ms:%d", &rk, &ms);
      if (rk == m.rank && !ends.empty() && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ends.front().t).count() < ms)
        return done_seq;
    }
#endif
    while (!ends.empty()) {
      const cudaError_t r = cudaEventQuery(ends.front().ev);
      if (r == cudaErrorNotReady) break;
      CUDA_CHECK(r);
      done_seq = ends.front().seq;
      CUDA_CHECK(cudaEventDestroy(ends.front().ev));
      ends.pop_front();
    }
    return done_seq;
  }
  void image_wait() override {
    if (sg) CUDA_CHECK(cudaStreamSynchronize(sg));
  }
  // a bookmark's slice image read into the batch at unit idx (the units are consecutive quarters of the batch buffer)
  bool bookmark_read(size_t off, size_t bytes, int idx, qm::Trailer* t) override {
    need_scratch();
    const size_t Q = qm::bk_slice(bytes), imgb = bytes == m.core->end_bytes() ? sgeom::kEndImageBytes : sgeom::kBookmarkImageBytes;
    const int units = (int)((Q + qm::kQuarterBytes - 1) / qm::kQuarterBytes);
    if (idx + units > kBatch) die("state: a bookmark of %d units at %d of a batch of %d", units, idx, kBatch);
    file->read(gather, imgb, off, m.st);
    sum_block(gather, imgb, file->sum(off, true), m.st);
    CUDA_CHECK(cudaMemsetAsync(qpack + (size_t)idx * qm::kQuarterBytes, 0, (size_t)units * qm::kQuarterBytes, m.st));
    CUDA_CHECK(cudaMemcpyAsync(qpack + (size_t)idx * qm::kQuarterBytes, gather, Q, cudaMemcpyDeviceToDevice, m.st));
    uint8_t tb[qm::kTrailerUsed];
    CUDA_CHECK(cudaMemcpyAsync(tb, gather + qm::bk_trailer_off(bytes), sizeof tb, cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    uint8_t full[qm::kTrailerBytes] = {};
    std::memcpy(full, tb, sizeof tb);
    return qm::trailer_unpack(full, t);
  }
  // the four ranks' slices (units idx.. of the exchanged batch, rank by rank) laid end to end, put into `numbers`, and checked against the
  // trailer's four slice sums and the whole's
  bool bookmark_put(const Nums& numbers, size_t bytes, int idx, const qm::Trailer& t) override {
    need_scratch();
    const size_t Q = qm::bk_slice(bytes);
    const int units = (int)((Q + qm::kQuarterBytes - 1) / qm::kQuarterBytes);
    for (int r = 0; r < 4; ++r) {
      const size_t from = (size_t)r * Q;
      if (from >= bytes) continue;
      const size_t len = std::min(Q, bytes - from);
      for (int j = 0; j < units; ++j) {
        const size_t o = (size_t)j * qm::kQuarterBytes;
        if (o >= len) break;
        CUDA_CHECK(cudaMemcpyAsync(gather + from + o, exbuf + ((size_t)(idx + j) * 4 + r) * qm::kQuarterBytes, std::min(qm::kQuarterBytes, len - o),
                                   cudaMemcpyDeviceToDevice, m.st));
      }
    }
    m.bk_copy(numbers, 0, gather, bytes, true);
    uint64_t* five = d_tmp;
    for (int r = 0; r < 4; ++r) {
      const size_t from = (size_t)r * Q, len = from < bytes ? std::min(Q, bytes - from) & ~(size_t)7 : 0;
      if (len) qm::sum_exact(gather + from, len, (uint64_t)r, five + r, m.st);
      else CUDA_CHECK(cudaMemsetAsync(five + r, 0, sizeof(uint64_t), m.st));
    }
    qm::sum_exact(gather, bytes & ~(size_t)7, qm::kWholeSeed, five + 4, m.st);
    uint64_t now[5];
    CUDA_CHECK(cudaMemcpyAsync(now, five, sizeof now, cudaMemcpyDeviceToHost, m.st));
    CUDA_CHECK(cudaStreamSynchronize(m.st));
    bool ok = now[4] == t.wsum;
    for (int r = 0; r < 4; ++r) ok = ok && now[r] == t.qsum[r];
    return ok;
  }
};

// The core over the device's side, once the allocator exists (model.cu).
void Model::Impl::make_store() {
  sgeom::StoreGeom g;
  g.rank = rank, g.nslots = nslots, g.nunits = nunits, g.list_len = list_len, g.max_ctx = opt.max_ctx;
  std::copy(unit_of, unit_of + kMaxSlots, g.unit_of);
  std::copy(slot_ctx_, slot_ctx_ + kMaxSlots, g.slot_ctx);
  g.image_bytes = chunk_image(), g.snap_bytes = snap_bytes(false), g.end_bytes = end_bytes(false);
  g.poison = pool_poison;
  g.split = opt.kv_split;
#ifndef PICO_OLD_RULES
  g.imager = opt.imager, g.mark = opt.imager && opt.kv_split;  // every write is a quarter in the background; a split mode also marks
#endif
  store_dev = std::make_unique<EngineDev>(*this);
  core = std::make_unique<StoreCore>(g, *store_dev, calloc.get());
#ifdef DSGB10_TEST_FAULTS
  if (const char* e = std::getenv("DSGB10_JOIN_SHORT")) core->fault_join_take = std::atoi(e);  // the N-th take of a join is refused once
#endif
  core->outside = [this] {  // the pool's lending, for the digest
    StoreCore::Outside o;
    o.groups = (uint64_t)(uint32_t)lent.groups, o.owed = (uint64_t)(uint32_t)lent.owed, o.n0 = (uint64_t)lent.n0;
    o.cur_groups = (uint64_t)(uint32_t)tenants.groups_of(lend::kTenantEngram);
    o.phase = (uint64_t)(uint32_t)(drv.active() && drv.change().id == lend::kTenantEngram ? drv.change().phase : 0);
    for (int i = 0; i < tenants.size(); ++i) {  // the other tenants' groups and groups in flight (ids 0 and 1 are in the terms above)
      const lend::Tenant& t = tenants.at(i);
      if (t.id >= 2) o.tenant_groups[t.id] = (uint32_t)t.groups, o.tenant_pending[t.id] = (uint32_t)t.pending;
    }
    return o;
  };
}

// What is bound after the core exists: the host ring, and the draft module with its windows in every bookmark.
void Model::Impl::store_notes() {
  core->g.has_ring = host_ring != nullptr;
  core->g.spec = spec != nullptr;
  core->g.snap_bytes = snap_bytes(spec != nullptr), core->g.end_bytes = end_bytes(spec != nullptr);
}

void Model::Impl::state_reserve(size_t bytes) {
  const alloc_seal::Scoped seal(alloc_seal::kStore);
  core->store_of();  // the store is made here
  EngineDev& d = static_cast<EngineDev&>(*store_dev);
  if (d.gather) die("state: the store is reserved once");
  if (!calloc) die("state: the store needs the chunk pool");
  // a page or a bookmark at a time between the chunks and the disk tier (the draft module's windows counted in)
  d.gather = dalloc<uint8_t>(Pool::round(std::max({chunk_image(), end_bytes(true), snap_bytes(true)})));
  if ((opt.kv_split || opt.imager) && !d.qbuf) die("state: the split's buffers were not made before the free-memory read (Model::split_prepare)");
  const size_t more = bytes / chunk_image();
  if (!more) return;
  const double t0 = wall_s();
  pool_grow(more);
  store_numbers = more;
  if (rank == 0)
    std::printf("state: the store's %zu chunk numbers (%.2f GiB) join the pool, made and mapped in %.2f s: %u numbers (%.2f GiB) "
                "for the lines and the sessions kept between requests together\n",
                more, more * (double)chunk_image() / 1073741824.0, wall_s() - t0, calloc->total(),
                calloc->total() * (double)chunk_image() / 1073741824.0);
}

size_t Model::Impl::chunk_image() const {
  return (size_t)attn::kChunkEntries * (kKvEntryBytes[0] + kKvEntryBytes[1] + kKvEntryBytes[2] + kKvEntryBytes[3]);
}

uint8_t* Model::Impl::bk_at(const Nums& n, size_t off) const {
  return pool_arr[0] + ((size_t)n[off / sgeom::kBk] << attn::kChunkShift) * kKvEntryBytes[0] + off % sgeom::kBk;
}

void Model::Impl::bk_copy(const Nums& n, size_t off, uint8_t* dev, size_t bytes, bool to_bk, cudaStream_t on) {
  const cudaStream_t use = on ? on : st;  // the image stream's copy of a bookmark's bytes goes on that stream
  while (bytes) {
    const size_t m = std::min(bytes, sgeom::kBk - off % sgeom::kBk);
    uint8_t* b = bk_at(n, off);
    CUDA_CHECK(cudaMemcpyAsync(to_bk ? b : dev, to_bk ? dev : b, m, cudaMemcpyDeviceToDevice, use));
    off += m, dev += m, bytes -= m;
  }
}

// A bookmark's bytes all defined: zero, in the same pieces bk_copy moves them.
void Model::Impl::bk_zero(const Nums& n, size_t bytes) {
  size_t off = 0;
  while (bytes) {
    const size_t m = std::min(bytes, sgeom::kBk - off % sgeom::kBk);
    CUDA_CHECK(cudaMemsetAsync(bk_at(n, off), 0, m, st));
    off += m, bytes -= m;
  }
}

// ---- image rows (img_rows.h): a running request's rows are numbers no session names ---------------------------------------------

size_t Model::Impl::img_numbers() const { return core->img_numbers(); }

const uint8_t* Model::Impl::img_row(int key, size_t row) const {
  const imgrows::Ledger::Rows* r = core->img_find(key);
  if (!r || row >= r->rows) die("model: image row %zu of flight %d is not in the store", row, key);
  const imgrows::Where w = imgrows::where(row);
  return pool_arr[0] + ((size_t)r->nums[w.number] << attn::kChunkShift) * kKvEntryBytes[0] + w.offset;
}

// The numbers for flight `key`'s rows: lowest free first, evicting idle conversations while too few are free (the
// command's pinned sessions stay). The slot's reservation for them is spent either way.
bool Model::Impl::img_take(int key, size_t rows, int slot) { return core->img_take(key, rows, slot); }

// Rows [row0, row0 + n) from host memory into the flight's numbers, one staging piece at a time: the piece waits for the
// copy that last read the buffer, and its own copy is queued on the model's stream behind everything that read the numbers.
void Model::Impl::img_fill(int key, size_t row0, size_t n, const uint16_t* src) {
  const imgrows::Ledger::Rows* r = core->img_find(key);
  if (!r || row0 + n > r->rows || n > Model::kImgStageRows) die("model: image rows [%zu, %zu) of flight %d do not fit its %zu", row0, row0 + n, key, r ? r->rows : 0);
  if (!img_stage) {
    img_stage = halloc<uint8_t>(Model::kImgStageRows * imgrows::kRowBytes);
    CUDA_CHECK(cudaEventCreateWithFlags(&img_stage_ev, cudaEventDisableTiming));
  }
  CUDA_CHECK(cudaEventSynchronize(img_stage_ev));
  std::memcpy(img_stage, src, n * imgrows::kRowBytes);
  size_t at = 0;
  imgrows::runs(row0, row0 + n, [&](size_t r0, size_t cnt) {
    CUDA_CHECK(cudaMemcpyAsync(const_cast<uint8_t*>(img_row(key, r0)), img_stage + at, cnt * imgrows::kRowBytes, cudaMemcpyHostToDevice, st));
    at += cnt * imgrows::kRowBytes;
  });
  CUDA_CHECK(cudaEventRecord(img_stage_ev, st));
}

// key's numbers back to the pool (key 0: every flight's). Reads of them are queued on the model's stream ahead of any
// later write, so the numbers can be taken again at once.
void Model::Impl::img_free(int key) {
  const bool had = core->has_store();
  core->img_free(key);
  if (!had) return;
  for (ImgSet& d : imgs)  // a view parked on a unit that read the rows
    if (d.key == key || key == 0) d.span.clear(), d.row0.clear(), d.key = 0;
  if (img_key == key || key == 0) img_span.clear(), img_row0.clear(), img_key = 0;
}

// ---- lines and slots: the core's bookkeeping, and the reads of the lists the steps make -----------------------------------------

int Model::Impl::held(int s) const { return core->held(s); }

uint8_t* Model::Impl::ent_addr(int k, int a, int j) const {
  const int h = core->line_h[cur_unit];
  if (h == kNoLine) die("model: entry %d of kv source %d in slot %d, which holds no session", j, k, cur_slot);
  const Refs& l = core->store_s->sess.at(h).lists[k];
  if ((size_t)(j >> attn::kChunkShift) >= l.size()) die("model: entry %d of kv source %d past slot %d's list", j, k, cur_slot);
  return pool_arr[a] + (((size_t)l[j >> attn::kChunkShift].num() << attn::kChunkShift) | (j & (attn::kChunkEntries - 1))) * kKvEntryBytes[a];
}

void Model::Impl::ensure_mapped(int s, int upto) { core->ensure(s, upto); }

void Model::Impl::slot_reserve(int s, int need, int img) {
  if (s < 0 || s >= nslots) die("model: slot_reserve in slot %d of %d", s, nslots);
  core->slot_reserve(s, slot_chunks_for(s, need), img);
}
void Model::Impl::slot_unreserve(int s) { core->slot_unreserve(s); }
void Model::Impl::slot_attach(int s, int h, int upto) {
  core->slot_attach(s, h, upto);
  if (spec && spec->copy && s >= 0 && s < nslots) spec->copy->attach(unit_of[s], h, st);  // copy drafting: a conversation's counts start over with a new one
}
void Model::Impl::slot_release(int s) { core->slot_release(s); }

// ---- bookmarks: the numbers from the core, the copies here -------------------------------------------------------------------

namespace {

// positions [p0, p1) between a ring and bookmark bytes from `off` on (row bytes per position), at most 256 of them
void ring_rows(Model::Impl& m, uint8_t* ring, const Nums& b, size_t off, size_t row, int p0, int p1, bool to_bk) {
  for (int q = p0; q < p1;) {
    const int slot = q & (attn::kRing - 1), n = std::min(p1 - q, attn::kRing - slot);
    m.bk_copy(b, off + (size_t)(q - p0) * row, ring + (size_t)slot * row, (size_t)n * row, to_bk);
    q += n;
  }
}

// the draft module's windows, positions [p0, p1) or whole (p0 < 0), to or from bookmark bytes from `off` on
void spec_windows(Model::Impl& m, const Spec& sp, const Nums& b, size_t off, int p0, int p1, bool to_bk) {
  const size_t n = p0 < 0 ? attn::kRing : (size_t)(p1 - p0);
  for (int k = 0; k < dspark::kBlocks; ++k) {
    const attn::Window w = sp.draft->window(k);
    const size_t kv = off + (size_t)k * n * kSpecSlot, sc = kv + n * kHead;
    if (p0 < 0) {
      m.bk_copy(b, kv, w.kv, n * kHead, to_bk);
      m.bk_copy(b, sc, w.sc, n * kHead / 32, to_bk);
    } else {
      ring_rows(m, w.kv, b, kv, kHead, p0, p1, to_bk);
      ring_rows(m, w.sc, b, sc, kHead / 32, p0, p1, to_bk);
    }
  }
}

}  // namespace

// A window bookmark's numbers (the caller fills them, bk_copy), or nullptr when the pool has no room for it.
const Nums* Model::Impl::state_snap_buf(int h, int pos) { return core->snap_buf(h, pos); }

void Model::Impl::state_snap(int h, int pos) {
  if (pos < kSnapRows) die("state: snapshot at %d", pos);
  const Nums* b = core->snap_buf(h, pos);
  if (!b) return;
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(*this, ring_kv[L], *b, snap_kv_off(L), kHead, pos - kSnapRows, pos, true);
    ring_rows(*this, ring_sc[L], *b, snap_sc_off(L), kHead / 32, pos - kSnapRows, pos, true);
  }
  if (spec) spec_windows(*this, *spec, *b, kSnapBytes, pos - kSnapRows, pos, true);
}

// The draft module's windows into the bookmark at pos, which the prefill took (none: not taken).
void Model::Impl::state_snap_spec(int h, int pos) {
  const Nums* b = core->snap_peek(h, pos);
  if (b) spec_windows(*this, *spec, *b, kSnapBytes, pos - kSnapRows, pos, true);
}

void Model::Impl::state_restore(int h, int pos) {
  const Nums& b = core->snap_get(h, pos);
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(*this, ring_kv[L], b, snap_kv_off(L), kHead, pos - kSnapRows, pos, false);
    ring_rows(*this, ring_sc[L], b, snap_sc_off(L), kHead / 32, pos - kSnapRows, pos, false);
  }
  if (spec) spec_windows(*this, *spec, b, kSnapBytes, pos - kSnapRows, pos, false);
}

void Model::Impl::state_snap_end(int h, int pos) {
  const Nums* e = core->end_take(h, pos);
  if (!e) return;
  size_t off = 0;
  for (int L = 0; L < kLayers; ++L, off += kRingBytes) {
    bk_copy(*e, off, ring_kv[L], (size_t)attn::kRing * kHead, true);
    bk_copy(*e, off + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32, true);
  }
  for (int k = 0; k < kKvSlots; ++k, off += kCarryBytes) {
    bk_copy(*e, off, reinterpret_cast<uint8_t*>(kvs[k].carry_kv), kCarryBytes / 2, true);
    bk_copy(*e, off + kCarryBytes / 2, reinterpret_cast<uint8_t*>(kvs[k].carry_sc), kCarryBytes / 2, true);
  }
  if (spec) spec_windows(*this, *spec, *e, kEndBytes, -1, 0, true);
}

void Model::Impl::state_restore_end(int h) {
  const Nums& end = core->end_get(h);
  size_t off = 0;
  for (int L = 0; L < kLayers; ++L, off += kRingBytes) {
    bk_copy(end, off, ring_kv[L], (size_t)attn::kRing * kHead, false);
    bk_copy(end, off + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32, false);
  }
  for (int k = 0; k < kKvSlots; ++k, off += kCarryBytes) {
    bk_copy(end, off, reinterpret_cast<uint8_t*>(kvs[k].carry_kv), kCarryBytes / 2, false);
    bk_copy(end, off + kCarryBytes / 2, reinterpret_cast<uint8_t*>(kvs[k].carry_sc), kCarryBytes / 2, false);
  }
  if (spec) spec_windows(*this, *spec, end, kEndBytes, -1, 0, false);
}

// ---- Model's state_* : forwards to the core, and the copies of the bookmarks ---------------------------------------------------

void Model::state_reserve(size_t bytes) {
  d_->state_reserve(bytes);
  d_->core->trace_open();
}
size_t Model::state_capacity() const { return d_->store_numbers * d_->chunk_image(); }  // the store's share of the pool
size_t Model::state_ceiling() const { return d_->calloc ? d_->calloc->total() * d_->chunk_image() : 0; }  // all of it
size_t Model::state_giveable(const std::vector<int>& keep) const { return d_->core->giveable(keep); }
size_t Model::state_giveable_joint(const std::vector<int>& keep, const std::vector<int>& lines) const {
  return d_->core->giveable_joint(keep, lines);
}
std::vector<std::pair<int, int>> Model::state_snap_numbers(int h) const { return d_->core->snap_numbers(h); }
void Model::state_list_numbers(int h, int upto, int* in_pool, int* on_disk) const { d_->core->list_numbers(h, upto, in_pool, on_disk); }
uint64_t Model::state_sum_cuts() const { return d_->core->sum_cuts; }
size_t Model::state_restore_chunks(int h) const { return d_->core->restore_chunks(h); }  // the numbers reading h's pieces on the disk tier back takes
Model::StoreChunks Model::state_chunks() const {  // the chunk pool: numbers in all, held by the sessions no line holds
  StoreChunks c;
  const StoreCore::Chunks k = d_->core->chunks();
  c.total = k.total, c.mapped = k.mapped, c.live_bytes = k.live_bytes;
  return c;
}
void Model::state_pin(const std::vector<int>& h, const std::vector<int>& live, const std::vector<int>& last) {
  d_->core->pin(h, live, last);
}
void Model::slot_unreserve(int s) { d_->slot_unreserve(s); }

// Numbers `snaps` window bookmarks take (a bookmark is snap_bytes in 2 MiB numbers).
int Model::state_bookmark_numbers(int snaps) const { return d_->core->bookmark_numbers(snaps); }

// Numbers a copy on write at position `pos` takes: with h_own == h_src, own_tail's count (every list's numbers from the
// cut's chunk on that another list or bookmark also names, one fresh number each); with h_own another session (a fork of
// h_src at pos), one number per list for the partial chunk at pos. Reads the store's records only (the same on every rank).
int Model::state_cow_numbers(int h_src, int h_own, int pos) const { return d_->core->cow_numbers(h_src, h_own, pos); }

// A hash of the state every rank must hold alike (sessions with their lists, bookmarks and images, the numbers' counts and
// images, the reservations, the evict-last list, the pool's lending): compared across the ranks' logs under DSGB10_STORE_CHECK=1.
uint64_t Model::store_digest() const { return d_->core->digest(); }
std::vector<int> Model::state_evicted() { return d_->core->take_evicted(); }
void Model::state_save(int h, int from, int to) { d_->core->save(h, from, to, d_->cur_unit); }
void Model::state_load(int h, int to) { d_->core->load(h, to, d_->cur_unit, d_->cur_slot); }
void Model::state_snap(int h, int pos) { d_->state_snap(h, pos); }
void Model::state_snap_end(int h, int pos) { d_->state_snap_end(h, pos); }
void Model::state_restore(int h, int pos) { d_->state_restore(h, pos); }
void Model::state_restore_end(int h) { d_->state_restore_end(h); }
void Model::state_drop(int h, int from) { d_->core->drop(h, from); }
void Model::state_free(int h) { d_->core->free_session(h); }
void Model::state_copy_snaps(int src, int dst, int upto) { d_->core->copy_snaps(src, dst, upto); }
void Model::state_drop_snap(int h, int pos) { d_->core->drop_snap(h, pos); }
void Model::state_flush(int h) { d_->core->flush(h); }
bool Model::state_disk_resize(size_t bytes) { return d_->core->disk_resize(bytes); }
uint64_t Model::state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                 std::string* why) {
  return d_->core->kept_check(base, run_id, max_age_s, bytes, why);
}
void Model::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  d_->core->disk(base, bytes, staging, reuse, run_id);
}
void Model::state_keep(uint64_t gen) { d_->core->keep(gen); }
void Model::state_checkpoint(uint64_t gen) { d_->core->checkpoint(gen); }
std::vector<int> Model::state_checkpointed() const { return d_->core->checkpointed(); }
uint64_t Model::state_version() const { return d_->core->version(); }
uint64_t Model::state_kept() const { return d_->core->kept_generation(); }
std::vector<int> Model::state_sessions() const { return d_->core->sessions(); }
size_t Model::state_disk_capacity() const { return d_->core->disk_capacity(); }
size_t Model::state_disk_bytes() const { return d_->core->disk_bytes(); }
std::string Model::state_disk_path() const { return d_->core->disk_path(); }
void Model::state_spill(int h) { d_->core->spill(h); }
void Model::state_split(int h, int budget) { d_->core->split(h, budget); }
// The imager. image: queued on the image stream, never waited for; settle: the entries of the commands at or below `upto` are clean and
// the idle conversations are marked (host bookkeeping, no device call); image_done: this rank's reading of the highest command whose work ended.
// A ride (cmd.h split_ride_run): the commands every rank has finished are settled, then h's pieces are queued
void Model::state_ride(uint64_t settled, int h, int budget, const std::vector<int>& live) {
  ++d_->core->split_counts.rides;
  d_->core->settle(settled, live);
  if (h > 0 && budget > 0) d_->core->image(h, budget, live);
}
int Model::image_room_pieces(int maxn) const {
  int n = maxn;
  while (n > 0 && !d_->store_dev->image_room((size_t)n * sgeom::kQuarterDiskBytes)) --n;
  return n;
}
void Model::ride_readings(uint64_t out[4]) const {
  const StoreCore& c = *d_->core;
  out[0] = c.image_seq(), out[1] = c.image_pending(), out[2] = c.calloc ? c.calloc->ghosts() : 0, out[3] = c.calloc ? c.calloc->free() : 0;
}
bool Model::has_host_ring() const { return d_->host_ring != nullptr; }
uint64_t Model::image_pending() const { return d_->core->image_pending(); }
int Model::image_todo(int h) const { return d_->core->image_todo(h); }
uint64_t Model::image_done() const { return d_->store_dev->image_done(); }
bool Model::image_room(size_t bytes) const { return d_->store_dev->image_room(bytes); }
uint64_t Model::image_seq() const { return d_->core->image_seq(); }
bool Model::imager() const { return d_->core->imager(); }
void Model::state_join(int h, int upto, int max_numbers) {
  d_->core->join_pieces = std::max(1, d_->opt.kv_join_pieces);
  d_->core->join(h, upto, (size_t)std::max(0, max_numbers), d_->core->join_pieces);
}
// The split's buffers, made at the start before the free-memory read the store is sized from, so that the read already lacks them
// (pool_rules.h kSplit is a Counted line of the start table, class Fixed). A no-op without --kv-split cache.
void Model::split_prepare() {
  if (d_->opt.kv_split || d_->opt.imager) static_cast<EngineDev&>(*d_->store_dev).prepare();
}
void Model::state_join_plan(int h, int upto, int* back, int* exch) const { d_->core->join_plan(h, upto, back, exch); }
Model::SplitStats Model::split_stats() const {
  const StoreCore& c = *d_->core;
  SplitStats r;
  r.split_pieces = c.split_counts.split_pieces, r.split_deferred = c.split_counts.split_deferred, r.revives = c.split_counts.revives;
  r.joins = c.split_counts.joins, r.join_bytes = c.split_counts.join_bytes, r.join_ns = c.split_counts.join_ns;
  r.join_max_ns = c.split_counts.join_max_ns, r.join_short = c.split_counts.join_short, r.join_bad = c.split_counts.join_bad;
  r.exchanged = c.split_counts.exchanged, r.join_taken = c.split_counts.join_taken, r.unspill_taken = c.split_counts.unspill_taken;
  r.imaged = c.split_counts.imaged, r.image_voids = c.split_counts.image_voids, r.image_waits = c.split_counts.image_waits;
  r.returns_read = c.split_counts.returns_read, r.rides = c.split_counts.rides;
  const StoreCore::ImageView iv = c.image_view();
  r.clean_numbers = iv.clean_numbers, r.image_pending = iv.image_pending;
  const StoreCore::SplitView v = c.split_view();
  r.conv_split = v.conv_split, r.quarter_slots = v.quarter_slots, r.quarter_numbers = v.quarter_numbers, r.ghost_numbers = v.ghost_numbers;
  return r;
}
bool Model::split_can_open() const { return d_->core->split_can_open(); }
void Model::state_list_numbers(int h, int upto, int* in_pool, int* on_disk, int* split) const {
  d_->core->list_numbers(h, upto, in_pool, on_disk, split);
}
void Model::state_unspill(int h, int max_numbers, int upto) {
  const size_t cap = max_numbers < 0 ? StoreCore::kSpent : (size_t)max_numbers;
  if (upto > 0) d_->core->unspill(h, cap, upto, std::max(1, d_->opt.kv_join_pieces));
  else d_->core->unspill(h, cap);
}
int Model::state_unspill_left(int h, int upto) const { return d_->core->unspill_left(h, upto); }
void Model::state_set_factor(int h, int factor) { d_->core->set_factor(h, factor); }
int Model::state_factor(int h) const { return d_->core->factor(h); }
Model::StateInfo Model::state_info(int h) const {
  const StoreCore::Info i = d_->core->info(h);
  StateInfo r;
  r.saved = i.saved, r.end_pos = i.end_pos, r.snaps = i.snaps, r.bytes = i.bytes, r.on_disk = i.on_disk;
  r.disk_bytes = i.disk_bytes, r.restart = i.restart, r.split = i.split, r.slots = i.slots;
  return r;
}
size_t Model::state_bytes() const {  // device memory the store's sessions hold (no line holds them)
  return state_chunks().live_bytes;
}
Model::TierStats Model::tier_stats() const {
  const StoreCore::TierStats c = d_->core->tier_stats();
  TierStats t;
  t.store_cap = c.store_cap, t.store_used = c.store_used, t.resident = c.resident, t.on_disk = c.on_disk;
  t.disk_cap = c.disk_cap, t.disk_used = c.disk_used, t.spills = c.spills, t.spill_bytes = c.spill_bytes;
  t.restores = c.restores, t.restore_bytes = c.restore_bytes, t.restored_pos = c.restored_pos, t.wait_ns = c.wait_ns;
  t.wait_max_ns = c.wait_max_ns;
  return t;
}
Model::PoolStats Model::pool_stats() const {
  const StoreCore::PoolStats c = d_->core->pool_stats();
  PoolStats p;
  p.numbers = c.numbers, p.free = c.free, p.held = c.held, p.shared = c.shared, p.reserved = c.reserved;
  p.number_bytes = c.number_bytes, p.image_numbers = c.image_numbers, p.on_lines = c.on_lines, p.in_memory = c.in_memory;
  p.on_disk = c.on_disk, p.sharing = c.sharing;
  return p;
}
void Model::drain() { d_->drain(); }

}  // namespace model
