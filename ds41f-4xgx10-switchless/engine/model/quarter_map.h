// A cache piece's quarters: who owns which entry, where a quarter's bytes sit, the checksum of a view, the trailer of a
// quarter's disk image. Included by the kernels (quarter.cu) and by the host code and its oracle, so both read one map.
//
// A chunk number holds 8,192 entries in four arrays (bytes an entry: 256, 32, 64, 4). The first two (codes and their
// scales) are cut by the entry's low two bits: rank q owns entries with e & 3 == q, local index (e & 8191) >> 2. The last two
// (index keys and their scales) are cut by bits 9 and 10: rank q owns (e >> 9) & 3 == q, local index
// (((e & 8191) >> 11) << 9) | (e & 511). A quarter is 2,048 entries of each array, laid out array after array. A quarter slot
// is the number c and the quarter q: s = (c << 2) | q.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define QM_HD __host__ __device__
#else
#define QM_HD
#endif

struct CUstream_st;  // cudaStream_t is a pointer to it

namespace model::quarter {

constexpr uint32_t kEntries = 8192, kQuarterEntries = 2048;
QM_HD constexpr uint32_t arr_bytes(int a) { return a == 0 ? 256 : a == 1 ? 32 : a == 2 ? 64 : 4; }
// the byte offset of array a's slice in a whole piece's image, and in a quarter
QM_HD constexpr size_t image_off(int a) {
  size_t o = 0;
  for (int j = 0; j < a; ++j) o += (size_t)kEntries * arr_bytes(j);
  return o;
}
QM_HD constexpr size_t quarter_off(int a) {
  size_t o = 0;
  for (int j = 0; j < a; ++j) o += (size_t)kQuarterEntries * arr_bytes(j);
  return o;
}
constexpr size_t kImageBytes = image_off(4), kQuarterBytes = quarter_off(4);
// the disk tier's block for a quarter: 12 grains of 64 KiB; what the quarter leaves unused holds its trailer
constexpr size_t kDiskBytes = 12 * (size_t)(64 << 10), kTrailerBytes = kDiskBytes - kQuarterBytes;
static_assert(kImageBytes == 2916352 && kQuarterBytes == 729088 && kImageBytes == 4 * kQuarterBytes, "a piece and its quarter");
static_assert(kDiskBytes == 786432 && kTrailerBytes == 57344, "a quarter's image on the disk");

QM_HD inline int code_owner(uint32_t e) { return (int)(e & 3); }
QM_HD inline uint32_t code_local(uint32_t e) { return (e & 8191) >> 2; }
QM_HD inline int key_owner(uint32_t e) { return (int)((e >> 9) & 3); }
QM_HD inline uint32_t key_local(uint32_t e) { return (((e & 8191) >> 11) << 9) | (e & 511); }
// the entry (in [0, 8192)) that is local index i of rank q's quarter, in array a
QM_HD inline uint32_t code_entry(int q, uint32_t i) { return (i << 2) | (uint32_t)q; }
QM_HD inline uint32_t key_entry(int q, uint32_t i) { return ((i >> 9) << 11) | ((uint32_t)q << 9) | (i & 511); }
QM_HD inline uint32_t entry_of(int a, int q, uint32_t i) { return a < 2 ? code_entry(q, i) : key_entry(q, i); }
QM_HD inline int owner_of(int a, uint32_t e) { return a < 2 ? code_owner(e) : key_owner(e); }
QM_HD inline uint32_t local_of(int a, uint32_t e) { return a < 2 ? code_local(e) : key_local(e); }

// Rank q's quarter, a 4-byte unit at byte qb (a multiple of 4) of its layout: the array and the byte of that array's slice in
// the whole piece that it comes from, and the entry that byte belongs to.
struct Src {
  int a;
  uint32_t off;    // bytes into array a's slice (8,192 entries)
  uint32_t entry;  // the entry in [0, 8192)
};
QM_HD inline Src quarter_src(int q, uint32_t qb) {
  int a = 0;
  while (a < 3 && qb >= quarter_off(a + 1)) ++a;
  const uint32_t lb = qb - (uint32_t)quarter_off(a), sz = arr_bytes(a), i = lb / sz;
  const uint32_t e = entry_of(a, q, i);
  return Src{a, e * sz + lb % sz, e};
}

// ---- the checksum of a view -------------------------------------------------------------------------------------------
// block_sum's mix (model/state.cu) over exactly the words given (no rounding to the pool's grain); the seed is added before the
// mix, and seed 0 is block_sum's own term. The terms add, so the order a sum is taken in does not matter.
constexpr uint64_t kSeedStep = 0xD6E8FEB86659FD93ull;
QM_HD inline uint64_t sum_term(uint64_t index, uint64_t word, uint64_t seed) {
  uint64_t x = word + index * 0x9E3779B97F4A7C15ull + seed * kSeedStep;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  return x ^ (x >> 27);
}
// a view's seeds: a quarter's is its owner rank, the whole piece's is kWholeSeed
constexpr uint64_t kWholeSeed = 4;
// host: the sum of `bytes` (a multiple of 8) at p
inline uint64_t sum_exact_host(const void* p, size_t bytes, uint64_t seed) {
  uint64_t s = 0;
  for (size_t i = 0; i < bytes / 8; ++i) {
    uint64_t w;
    std::memcpy(&w, (const uint8_t*)p + i * 8, 8);
    s += sum_term(i, w, seed);
  }
  return s;
}

// ---- the device's side (quarter.cu): every call is queued on the stream it is given -------------------------------------------
// The four arrays' slices of one chunk number (pool array a + (c << 13) entries), read or written as a piece.
struct Piece {
  const uint8_t* a[4];
};
struct PieceOut {
  uint8_t* a[4];
};
// dst (kQuarterBytes): rank q's quarter of the piece. Entries from `written` on are written as zero, whatever the piece
// holds there, so a quarter's bytes and sums depend on the written entries only.
void quarter(const Piece& src, uint8_t* dst, int q, int written, CUstream_st* st);
// the piece's arrays from four quarters, q * kQuarterBytes apart at `quarters`: every entry written (a quarter holds zero where
// its piece was not written)
void unquarter(const uint8_t* quarters, const PieceOut& dst, CUstream_st* st);
// *out (device or pinned memory) = the sum of `bytes` (a multiple of 8) at p with the seed; *out is zeroed first
void sum_exact(const void* p, size_t bytes, uint64_t seed, uint64_t* out, CUstream_st* st);
// out[0..3] = the four quarters' sums, each over its quarter layout with its owner's seed; out[4] = the whole piece's over its
// image layout with kWholeSeed; entries from `written` on read as zero in every view. out has 5 words; zeroed first
void piece_sums(const Piece& src, int written, uint64_t* out, CUstream_st* st);
// out[0..4] (host) = the five words piece_sums wrote at `rec`, once the work queued on `st` before the call has run: the copy is
// made on `st` and waited for (the model's stream does not synchronize with the default stream, so a plain blocking copy would
// not wait for the kernel)
void sums_read(const uint64_t* rec, uint64_t* out, CUstream_st* st);
// The trailer's used words (kTrailerUsed bytes at dst) built on the stream from the five sum words at `five` (device memory) and the
// fields, the `zero` bytes from dst cleared first: no read of the sums by the host, so nothing waits (the image stream's own work).
// Every kernel of this file launched once, with the model's stream empty (the first launch of a kernel inside a decoding run, behind a step that waits for the host,
// does not return). `piece` / `piece_out` are one scratch piece, `quarter_dst` kQuarterBytes then the trailer's kTrailerBytes (kDiskBytes in all), `quarters` four
// quarters, `five` and `five_st` five device words each. `sg` is the image stream, `st` the model's. Returns the kernels launched; the table of their names is in warm()'s body.
int warm(const Piece& piece, const PieceOut& piece_out, uint8_t* quarter_dst, const uint8_t* quarters, uint64_t* five, uint64_t* five_st, CUstream_st* sg, CUstream_st* st);
void trailer_device(const uint64_t* five, uint32_t rank, uint32_t handle, uint32_t list, uint32_t piece, uint32_t written, uint8_t* dst,
                    size_t zero, CUstream_st* st);

// ---- a quarter's image on the disk: the trailer -----------------------------------------------------------------------------
// After the quarter's 729,088 bytes: the four quarters' sums and the whole piece's, and whose piece this is. A rank reads its
// quarter back, restores the sums from the trailer and refuses a file that is another rank's or another piece's.
struct Trailer {
  uint64_t qsum[4] = {};
  uint64_t wsum = 0;
  uint32_t rank = 0, handle = 0, list = 0, piece = 0, written = 0;
};
constexpr uint64_t kTrailerMagic = 0x3130525451474344ull;  // "DCGQTR01", little endian
namespace detail {
// the check word over the trailer's fields: host and device (k_trailer builds the trailer on the image stream)
QM_HD inline uint64_t trailer_check_f(const uint64_t* qsum, uint64_t wsum, uint32_t rank, uint32_t handle, uint32_t list, uint32_t piece,
                                      uint32_t written) {
  uint64_t h = kTrailerMagic;
  auto mix = [&h](uint64_t v) {
    h = (h ^ v) * 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
  };
  for (int i = 0; i < 4; ++i) mix(qsum[i]);
  mix(wsum);
  mix(rank), mix(handle), mix(list), mix(piece), mix(written);
  return h;
}
inline uint64_t trailer_check(const Trailer& t) { return trailer_check_f(t.qsum, t.wsum, t.rank, t.handle, t.list, t.piece, t.written); }
}  // namespace detail
// The used part of a trailer: ten 64-bit words, the layout trailer_pack writes (bytes 0 to 79; the rest of a list piece's trailer is
// zero). A bookmark's image, whose slice leaves less room than kTrailerBytes, holds just these words after its slice.
constexpr size_t kTrailerUsed = 80;
QM_HD inline void trailer_words(const uint64_t* qsum, uint64_t wsum, uint32_t rank, uint32_t handle, uint32_t list, uint32_t piece,
                                uint32_t written, uint64_t* w) {
  w[0] = kTrailerMagic;
  for (int i = 0; i < 4; ++i) w[1 + i] = qsum[i];
  w[5] = wsum;
  w[6] = (uint64_t)rank | (uint64_t)handle << 32;
  w[7] = (uint64_t)list | (uint64_t)piece << 32;
  w[8] = (uint64_t)written;
  w[9] = detail::trailer_check_f(qsum, wsum, rank, handle, list, piece, written);
}
// a bookmark's image: this rank's slice of its bytes (`slice` bytes, rounded up to 16) and then the trailer's used words
QM_HD constexpr size_t bk_slice(size_t bytes) { return (bytes / 4 + 15) / 16 * 16; }
QM_HD constexpr size_t bk_trailer_off(size_t bytes) { return (bk_slice(bytes) + 15) / 16 * 16; }
// out: kTrailerBytes, written whole (the rest zero)
inline void trailer_pack(const Trailer& t, uint8_t* out) {
  std::memset(out, 0, kTrailerBytes);
  size_t o = 0;
  auto put = [&](const void* v, size_t n) { std::memcpy(out + o, v, n), o += n; };
  put(&kTrailerMagic, 8);
  put(t.qsum, 32);
  put(&t.wsum, 8);
  put(&t.rank, 4), put(&t.handle, 4), put(&t.list, 4), put(&t.piece, 4), put(&t.written, 4);
  const uint64_t c = detail::trailer_check(t);
  o = (o + 7) & ~(size_t)7;
  put(&c, 8);
}
// false: not a trailer (the magic or the check is wrong)
inline bool trailer_unpack(const uint8_t* in, Trailer* t) {
  size_t o = 0;
  auto get = [&](void* v, size_t n) { std::memcpy(v, in + o, n), o += n; };
  uint64_t magic = 0, c = 0;
  Trailer r;
  get(&magic, 8);
  get(r.qsum, 32);
  get(&r.wsum, 8);
  get(&r.rank, 4), get(&r.handle, 4), get(&r.list, 4), get(&r.piece, 4), get(&r.written, 4);
  o = (o + 7) & ~(size_t)7;
  get(&c, 8);
  if (magic != kTrailerMagic || c != detail::trailer_check(r)) return false;
  *t = r;
  return true;
}
// this rank's and this piece's
inline bool trailer_is(const Trailer& t, uint32_t rank, uint32_t handle, uint32_t list, uint32_t piece) {
  return t.rank == rank && t.handle == handle && t.list == list && t.piece == piece;
}

}  // namespace model::quarter
