// model/quarter.cu on a GB10 (about 20 MB of device memory, so it can run beside an engine): k_quarter against the oracle's
// bytes for every rank and several written counts (stale bytes past the count in the pool), k_unquarter back to the written image,
// sum_exact and piece_sums against the oracle's sums, and a flip in a quarter's last word, which changes that quarter's sum and
// the unquartered piece's, and nothing else.
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "common/check.h"
#include "model/quarter_map.h"
#include "model/tests/quarter_oracle.h"

namespace q = model::quarter;
namespace o = quarter_oracle;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s\n", what);
  }
}

// a piece on the device: the four arrays, each its own allocation, filled from an image
struct DevPiece {
  uint8_t* a[4];
  explicit DevPiece(const std::vector<uint8_t>& image) {
    for (int i = 0; i < 4; ++i) {
      const size_t n = (size_t)q::kEntries * o::kSize[i];
      CUDA_CHECK(cudaMalloc(&a[i], n));
      CUDA_CHECK(cudaMemcpy(a[i], image.data() + o::slice(i, true), n, cudaMemcpyHostToDevice));
    }
  }
  ~DevPiece() {
    for (auto p : a) cudaFree(p);
  }
  q::Piece in() const { return q::Piece{{a[0], a[1], a[2], a[3]}}; }
  q::PieceOut out() const { return q::PieceOut{{a[0], a[1], a[2], a[3]}}; }
  std::vector<uint8_t> image() const {
    std::vector<uint8_t> v(q::kImageBytes);
    for (int i = 0; i < 4; ++i)
      CUDA_CHECK(cudaMemcpy(v.data() + o::slice(i, true), a[i], (size_t)q::kEntries * o::kSize[i], cudaMemcpyDeviceToHost));
    return v;
  }
};

std::vector<uint8_t> fetch(const void* d, size_t n) {
  std::vector<uint8_t> v(n);
  CUDA_CHECK(cudaMemcpy(v.data(), d, n, cudaMemcpyDeviceToHost));
  return v;
}
}  // namespace

int main() {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));  // as the model's stream is made
  uint8_t* dq;  // four quarters in a row
  uint64_t* dsum;
  CUDA_CHECK(cudaMalloc(&dq, 4 * q::kQuarterBytes));
  CUDA_CHECK(cudaMalloc(&dsum, 8 * sizeof(uint64_t)));
  char what[128];

  for (const uint32_t written : {8192u, 8191u, 5000u, 2049u, 1u, 0u}) {
    const std::vector<uint8_t> image = o::random_image(100 + written);  // bytes past `written` are junk the kernels must not use
    DevPiece p(image);
    std::vector<uint8_t> all;
    bool eq = true;
    for (int r = 0; r < 4; ++r) {
      q::quarter(p.in(), dq + (size_t)r * q::kQuarterBytes, r, (int)written, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      const std::vector<uint8_t> got = fetch(dq + (size_t)r * q::kQuarterBytes, q::kQuarterBytes), want = o::quarter_of(image, r, written);
      eq &= got == want;
      all.insert(all.end(), want.begin(), want.end());
    }
    std::snprintf(what, sizeof what, "k_quarter equals the oracle's quarter for ranks 0-3, written %u", written);
    check(eq, what);

    // the inverse: into a piece that held other bytes
    DevPiece junk(o::random_image(7));
    q::unquarter(dq, junk.out(), st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::snprintf(what, sizeof what, "k_unquarter gives the written image, written %u", written);
    check(junk.image() == o::written_of(image, written), what);

    // the sums: the four quarters' over their layout and the whole piece's, from the stale piece
    q::piece_sums(p.in(), (int)written, dsum, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<uint8_t> sums = fetch(dsum, 5 * sizeof(uint64_t));
    const uint64_t* s = (const uint64_t*)sums.data();
    bool seq = true;
    for (int r = 0; r < 4; ++r) seq &= s[r] == o::sum_of(&all[(size_t)r * q::kQuarterBytes], q::kQuarterBytes, (uint64_t)r);
    const std::vector<uint8_t> wimg = o::written_of(image, written);
    seq &= s[4] == o::sum_of(wimg.data(), q::kImageBytes, q::kWholeSeed);
    std::snprintf(what, sizeof what, "piece_sums equals the oracle's four quarter sums and whole sum, written %u", written);
    check(seq, what);

    // sum_exact over a quarter in the slot equals the recorded quarter sum (what a join compares)
    bool join_eq = true;
    for (int r = 0; r < 4; ++r) {
      q::sum_exact(dq + (size_t)r * q::kQuarterBytes, q::kQuarterBytes, (uint64_t)r, dsum + 5, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      join_eq &= *(const uint64_t*)fetch(dsum + 5, 8).data() == s[r];
    }
    std::snprintf(what, sizeof what, "sum_exact over a stored quarter equals the sum recorded at the split, written %u", written);
    check(join_eq, what);
  }

  {  // sum_exact against the oracle: a quarter, a piece, 64 KiB multiples and plain multiples of 8, seeds
    const std::vector<uint8_t> image = o::random_image(5);
    DevPiece p(image);
    uint8_t* flat;
    CUDA_CHECK(cudaMalloc(&flat, q::kImageBytes));
    CUDA_CHECK(cudaMemcpy(flat, image.data(), q::kImageBytes, cudaMemcpyHostToDevice));
    bool eq = true;
    for (const size_t bytes : {(size_t)8, (size_t)4096, (size_t)65536, (size_t)q::kQuarterBytes, (size_t)q::kImageBytes})
      for (const uint64_t seed : {0ull, 2ull, 4ull}) {
        q::sum_exact(flat, bytes, seed, dsum, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        eq &= *(const uint64_t*)fetch(dsum, 8).data() == o::sum_of(image.data(), bytes, seed);
      }
    check(eq, "sum_exact equals the oracle's sum over 8 B to a whole piece, three seeds");
    cudaFree(flat);
  }

  {  // a flip in a quarter's last word
    const std::vector<uint8_t> image = o::random_image(9);
    DevPiece p(image);
    for (int r = 0; r < 4; ++r) {
      q::quarter(p.in(), dq + (size_t)r * q::kQuarterBytes, r, q::kEntries, st);
    }
    q::piece_sums(p.in(), q::kEntries, dsum, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    const std::vector<uint8_t> rec = fetch(dsum, 5 * sizeof(uint64_t));
    const uint64_t* s = (const uint64_t*)rec.data();
    for (int r = 0; r < 4; ++r) {
      uint8_t* last = dq + (size_t)r * q::kQuarterBytes + q::kQuarterBytes - 8;
      uint8_t b;
      CUDA_CHECK(cudaMemcpy(&b, last + 3, 1, cudaMemcpyDeviceToHost));
      const uint8_t flipped = b ^ 0x04;
      CUDA_CHECK(cudaMemcpy(last + 3, &flipped, 1, cudaMemcpyHostToDevice));
      int diff = 0;
      for (int k = 0; k < 4; ++k) {
        q::sum_exact(dq + (size_t)k * q::kQuarterBytes, q::kQuarterBytes, (uint64_t)k, dsum + 5, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        diff += *(const uint64_t*)fetch(dsum + 5, 8).data() != s[k];
      }
      std::snprintf(what, sizeof what, "a flip in rank %d's quarter's last word is found in that quarter's sum only", r);
      check(diff == 1, what);
      DevPiece back(o::random_image(1));
      q::unquarter(dq, back.out(), st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      const std::vector<uint8_t> got = back.image();
      size_t bad = 0;
      for (size_t i = 0; i < got.size(); ++i) bad += got[i] != image[i];
      std::snprintf(what, sizeof what, "the flip in rank %d's quarter reaches exactly one byte of the piece", r);
      check(bad == 1, what);
      CUDA_CHECK(cudaMemcpy(last + 3, &b, 1, cudaMemcpyHostToDevice));
    }
  }

  {  // the sums as a split command reads them: queued on the stream behind the quartering, read at once, as slot_out does
    const std::vector<uint8_t> image = o::random_image(77);
    DevPiece p(image);
    q::piece_sums(p.in(), q::kEntries, dsum, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    uint64_t want[5];
    CUDA_CHECK(cudaMemcpy(want, dsum, sizeof want, cudaMemcpyDeviceToHost));
    uint64_t* d64;
    CUDA_CHECK(cudaMalloc(&d64, 64 * 5 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d64, 0, 64 * 5 * sizeof(uint64_t)));
    int early = 0;
    for (int i = 0; i < 64; ++i) {
      for (int r = 0; r < 4; ++r) q::quarter(p.in(), dq + (size_t)r * q::kQuarterBytes, r, q::kEntries, st);
      q::piece_sums(p.in(), q::kEntries, d64 + (size_t)i * 5, st);
      uint64_t got[5];
      q::sums_read(d64 + (size_t)i * 5, got, st);
      for (int k = 0; k < 5; ++k)
        if (got[k] != want[k]) {
          ++early;
          break;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::printf("%d of 64 trailers' sums read early\n", early);
    check(early == 0, "sums_read waits for the sums kernel queued on a non-blocking stream: 0 of 64 read early");
    cudaFree(d64);
  }

  {  // the imager's row (written, not run on the workstation: it needs a GPU).
     // What it proves: a quarter image made on a SECOND stream, beside a first stream that is kept busy, is byte-equal to the image the
     // as-built slot_out makes (the quarter, then the trailer: sums read by the host, trailer_pack), with the trailer BUILT ON THE DEVICE
     // (trailer_device: no read of the sums, no wait); the host's exact sum of the image block equals the device's.
     // Its settings: a 20 MB piece pair, two non-blocking streams, 4000 memsets of 64 MB on the first.
     // Its premise: the first stream is still busy when the second one's work is queued (checked by an event query: when it is not, the row
     // prints "premise not reached" and exits 2, never 0).
    const std::vector<uint8_t> image = o::random_image(5140);
    DevPiece p(image);
    cudaStream_t sg;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sg, cudaStreamNonBlocking));
    uint8_t* gq;
    uint64_t* five;
    CUDA_CHECK(cudaMalloc(&gq, q::kDiskBytes));
    CUDA_CHECK(cudaMalloc(&five, 5 * sizeof(uint64_t)));
    float* junk;
    CUDA_CHECK(cudaMalloc(&junk, 64 << 20));
    for (int i = 0; i < 4000; ++i) CUDA_CHECK(cudaMemsetAsync(junk, i & 0xFF, 64 << 20, st));  // the first stream is busy
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreate(&ev));
    CUDA_CHECK(cudaEventRecord(ev, st));
    const bool busy = cudaEventQuery(ev) == cudaErrorNotReady;
    for (const int written : std::initializer_list<int>{(int)q::kEntries, 5000}) {
      const int rank = 2;
      q::quarter(p.in(), gq, rank, written, sg);
      q::piece_sums(p.in(), written, five, sg);
      q::trailer_device(five, (uint32_t)rank, 7u, 3u, 11u, (uint32_t)written, gq + q::kQuarterBytes, q::kTrailerBytes, sg);
      CUDA_CHECK(cudaStreamSynchronize(sg));
      const std::vector<uint8_t> img = fetch(gq, q::kDiskBytes);
      // the as-built image: the quarter from the same call, the trailer packed by the host from the sums the host read
      const std::vector<uint8_t> rec = fetch(five, 5 * sizeof(uint64_t));
      const uint64_t* w5 = (const uint64_t*)rec.data();
      q::Trailer t;
      for (int k = 0; k < 4; ++k) t.qsum[k] = w5[k];
      t.wsum = w5[4], t.rank = (uint32_t)rank, t.handle = 7, t.list = 3, t.piece = 11, t.written = (uint32_t)written;
      std::vector<uint8_t> want(q::kDiskBytes);
      std::memcpy(want.data(), img.data(), q::kQuarterBytes);
      q::trailer_pack(t, want.data() + q::kQuarterBytes);
      std::snprintf(what, sizeof what, "the image made on the second stream (written %d) is byte-equal to the as-built slot image, trailer included", written);
      check(img == want, what);
      const std::vector<uint8_t> oq = o::quarter_of(image, rank, (uint32_t)written);
      check(std::memcmp(img.data(), oq.data(), q::kQuarterBytes) == 0, "and its quarter is the oracle's");
      check(q::sum_exact_host(img.data(), q::kDiskBytes, 0) == q::sum_exact_host(want.data(), q::kDiskBytes, 0), "the host's sum of the block equals the sum of the as-built block");
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaFree(junk), cudaFree(gq), cudaFree(five);
    cudaStreamDestroy(sg);
    if (!busy) {
      std::printf("premise not reached: the first stream was idle when the second one's work was queued\n");
      return 2;
    }
  }

  cudaFree(dq);
  cudaFree(dsum);
  cudaStreamDestroy(st);
  std::printf(g_fail ? "FAILED %d\n" : "quarter_test: all passed\n", g_fail);
  return g_fail != 0;
}
