// The session store's bookkeeping, as one host class (docs/engine.md "The chunk pool", "Kept sessions"): which chunk number
// each conversation's cache entries and bookmarks hold, who shares which, the reservations, the line each conversation is on,
// the disk tier's images and their file space, the order eviction takes pieces in, and the kept index. No CUDA header is
// included below this one: everything the device does in the middle of an operation is a call on StoreDev, which the engine
// implements (state.cu) and the host tests fake (tests/store_fake.h).
//
// Every rank runs the same operations in the same order and so holds the same state; anything that depends on one rank's own
// memory is agreed through StoreDev::agree_min before it changes anything.
//
// The entry points are the public functions of the "entries" block; each is one record of the trace (store_trace.h), and
// the ones the store calls on itself write nothing.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/die.h"
#include "model/chunks.h"
#include "model/config.h"
#include "model/img_rows.h"
#include "model/store_geom.h"
#include "model/store_trace.h"

namespace model::detail {

// The ids of the store's host all-gathers (comm/host_gather.h HostCall: state.cu static_asserts they are the same numbers)
constexpr uint32_t kSiteKeep = 20, kSiteCheckpoint = 21, kSiteDiskRoom = 22, kSiteUnspill = 23, kSiteJoin = 24;

constexpr size_t kPoolGrain = sgeom::kPoolGrain;  // the disk tier's blocks: multiples of 64 KiB, at 64 KiB offsets

// Best fit over [base, base + size); free ranges are coalesced, ties go to the lowest offset. The offsets alone (take_off,
// give_off) serve the disk tier's file.
struct Pool {
  static constexpr size_t kNone = SIZE_MAX;
  uint8_t* base = nullptr;
  size_t size = 0;
  std::map<size_t, size_t> free;  // offset -> bytes
  static size_t round(size_t bytes) { return (bytes + kPoolGrain - 1) / kPoolGrain * kPoolGrain; }
  uint8_t* take(size_t bytes);  // nullptr when no free range holds round(bytes)
  void give(uint8_t* p, size_t bytes);
  size_t take_off(size_t bytes);  // kNone when no free range holds round(bytes)
  bool take_at(size_t off, size_t bytes);  // [off, off + round(bytes)) when it is free (kept sessions)
  void give_off(size_t off, size_t bytes);
  size_t largest() const;
  // The first free range, lowest offset first, whose start `ok(offset, rounded bytes)` accepts: the block goes at
  // its start (first fit, so that the blocks stay at the bottom of the range). kNone when there is none.
  size_t take_fit(size_t bytes, const std::function<bool(size_t, size_t)>& ok);
};

using Nums = std::vector<uint32_t>;  // chunk numbers (model/chunks.h)

// A cache list's entry: where one piece of a conversation's cache is. 0: only on the disk tier; bit 31 clear: a chunk number;
// bit 31 set: a quarter slot (model/chunks.h SlotMap, number << 2 | quarter). A Ref does not convert to a number: the places that
// need one say num(), which stops on an entry that is not one.
struct Ref {
  static constexpr uint32_t kSlotBit = 1u << 31;
  uint32_t v = 0;
  constexpr Ref() = default;
  explicit constexpr Ref(uint32_t number) : v(number) {}
  static constexpr Ref slot(uint32_t s) { return Ref(s | kSlotBit); }
  constexpr bool on_ssd() const { return v == 0; }
  constexpr bool is_slot() const { return (v & kSlotBit) != 0; }
  constexpr bool is_number() const { return v != 0 && !is_slot(); }
  explicit constexpr operator bool() const { return v != 0; }  // in the pool, as a number or as a slot
  uint32_t num() const {
    if (!is_number()) die("state: list entry %08x is not a chunk number", v);
    return v;
  }
  uint32_t slot_of() const {
    if (!is_slot()) die("state: list entry %08x is not a quarter slot", v);
    return v & ~kSlotBit;
  }
  // the chunk number the entry stands on: the number itself, or the quarter number of its slot (0: only on the disk tier)
  constexpr uint32_t holder() const { return is_slot() ? (v & ~kSlotBit) >> 2 : v; }
  constexpr bool operator==(Ref o) const { return v == o.v; }
  constexpr bool operator!=(Ref o) const { return v != o.v; }
};
using Refs = std::vector<Ref>;
constexpr Ref kOnSsd{};

struct Store {
  struct Sess {
    Refs lists[kKvSlots];   // compressed caches: kv source k's chunk numbers (attn.h Compressed); 0: that entry is only on
                            // the disk tier (dl), read back before use
    std::vector<size_t> dl[kKvSlots];  // each entry's image in the disk tier's file (Pool::kNone: none), beside lists
    int saved = 0;          // positions [0, saved) are in them
    int factor = 16;        // the YaRN factor every cache entry and bookmark was rotated with (16, 32, 64): for life, set when
                            // a slot takes the session (set_factor); kept sessions record it
    std::map<int, Nums> snaps;  // window bookmarks by position: their chunk numbers (empty: only on the disk tier)
    // A bookmark that is split: its key stays in `snaps` (empty, as when it is only on the disk tier) and its quarter's slots are
    // here; the end bookmark's are qend (`end` empty). A split bookmark is not "only on the disk tier": out() does not count it.
    std::map<int, Refs> qsnaps;
    Refs qend;
    Nums end;               // every layer's ring and the carries (and the draft module's windows); empty: none in the pool
    int end_pos = -1;       // its position (-1: no end bookmark, in the pool or on the disk tier)
    int line = -1;          // the unit whose rows are its lists (-1: in no line)
    uint64_t used = 0;      // the clock when an operation last used the session
    std::map<int, size_t> dsnaps;  // window bookmarks' images in the file (a bookmark with a copy)
    size_t dend = Pool::kNone;     // the end bookmark's image
    bool kept = false;  // as an earlier run kept it, not read back since: a piece that differs is fatal
    bool out() const;   // a piece is only on the disk tier: read back before the session is used
    size_t in_pool() const {  // chunk numbers it names in the pool (a shared one counts in every session naming it)
      size_t n = end.size() + bookmark_slots();
      for (const Refs& l : lists)
        for (const Ref c : l) n += c ? 1 : 0;
      for (const auto& sn : snaps) n += sn.second.size();
      return n;
    }
    size_t numbers() const {  // chunk numbers it names whole: list entries that are numbers and bookmarks' numbers
      size_t n = end.size();
      for (const Refs& l : lists)
        for (const Ref c : l) n += c.is_number();
      for (const auto& sn : snaps) n += sn.second.size();
      return n;
    }
    size_t slot_entries() const {  // list entries that are quarter slots
      size_t n = 0;
      for (const Refs& l : lists)
        for (const Ref c : l) n += c.is_slot();
      return n;
    }
    size_t bookmark_slots() const {  // quarter slots the bookmarks stand on
      size_t n = qend.size();
      for (const auto& q : qsnaps) n += q.second.size();
      return n;
    }
    bool split() const { return slot_entries() != 0 || bookmark_slots() != 0; }  // a piece of it is a quarter slot: joined before the session is used
  };
  // A session in a kept file's index: its pieces' images and their checksums as they went out.
  struct Kept {
    int h = 0, saved = 0, end_pos = -1, factor = 16;
    uint64_t used = 0;
    std::vector<std::pair<size_t, uint64_t>> ents[kKvSlots];
    std::vector<uint8_t> qents[kKvSlots];  // 1 where the entry's image is a quarter image (kDiskBytes), else a whole one
    std::map<int, std::pair<size_t, uint64_t>> snaps;
    size_t dend = Pool::kNone;
    uint64_t dend_sum = 0;
  };
  std::map<int, Sess> sess;
  imgrows::Ledger img;              // the running requests' image rows, by flight key (img_rows.h): numbers no session names
  bool has_disk = false;            // the disk tier is open (none: evicted sessions are freed)
  Pool dpool;                       // its file's offsets
  size_t dbytes = 0;                // the file's bytes in images
  uint64_t wseq = 0;                // the last write queued (a read-back waits for it)
  std::vector<size_t> ndisk;        // [number]: the image of its present content (Pool::kNone: none); shared holders share it
  // The split. A quarter slot is s = (number << 2) | quarter: this rank's quarter of a piece, kept in a quarter number
  // that the slot map took from the allocator. A piece given back by a split is a ghost whose bytes are intact: twin_s[c] is the
  // slot holding number c's content as it is now, twin_c[s] the number whose content slot s holds (a link is cut the moment either
  // side changes). slot_disk[s] is the slot's image on the disk tier while it is proven to be the slot's content.
  chunks::SlotMap slots;
  std::vector<uint32_t> twin_s;
  std::map<uint32_t, uint32_t> twin_c;
  std::map<uint32_t, size_t> slot_disk;  // slot -> its quarter image in the file while the slot exists (immutable: never stale; one reference in drefs)
  std::set<size_t> qimg;                 // the images that are quarter images (kDiskBytes with a trailer), not a number's whole image
  std::map<uint32_t, int> slot_written;  // slot -> the entries written when it was made (a partial piece's count; 8,192 for a full one)
  uint64_t ghost_seq = 0;           // the stamp of the next ghost: a count, advanced by each number a split gives back
  // The imager (geom.imager). An image command queues a job; a later settle makes it true. A piece is plain (no
  // image), clean (its entry names a quarter image that went out and the block still holds the same bytes: ndisk / dnum say so), or a
  // ghost (the entry says "on the SSD", the block is free in the allocator, stamped, and linked to that image: its bytes are intact).
  struct ImgJob {
    uint64_t id = 0, seq = 0;  // the job, and the image command that queued it
    int h = 0, k = 0;          // the conversation, and the list (kBkWindow / kBkEnd: a bookmark)
    int piece = 0;             // the list entry, or the bookmark's position
    Nums blocks;               // the block of a list piece (one), the numbers of a bookmark
    size_t off = 0, bytes = 0; // the image's file offset and block
    bool dead = false;         // void: the file shrank past it (its space went with the file)
    int written = 0;           // entries the image covers (a bookmark: its bytes)
    std::vector<std::array<int, 3>> holders;  // (h, k, i) of every list entry that names the block and takes the image at the settle
  };
  static constexpr int kBkWindow = -1, kBkEnd = -2;
  std::deque<ImgJob> jobs;                        // queued, not yet settled, in command order
  std::unordered_map<uint32_t, uint64_t> job_of;  // block -> the job whose image is still its content (a cut erases it: the job is void)
  uint64_t img_seq = 0, job_id = 0;               // commands queued; jobs made
  std::map<size_t, Nums> bk_ghost;                // bookmark image -> its ghost group's numbers
  std::unordered_map<uint32_t, size_t> ghost_bk;  // number -> the bookmark image whose group it is in
  std::map<int, std::pair<size_t, uint64_t>> restoring;  // sessions whose return has read a piece and is not whole yet: (pieces read, tokens counted); a return counts once
  std::set<int> to_mark;                          // conversations with a clean piece that is not a ghost yet and may become one
  std::map<size_t, uint32_t> dnum;  // image -> the number holding its content, while one does
  std::map<size_t, uint32_t> drefs; // image -> the entries and bookmarks naming it
  std::string base;                 // the file is base.spill, its index base.index
  uint64_t run_id = 0;              // what the kept sessions' state depends on (main.cpp run_id)
  std::vector<Kept> kept;           // a kept index as kept_check read it, for disk
  uint64_t kept_gen = 0, kept_clock = 0;  // its stop's generation and the store's clock then
  uint64_t loaded_gen = 0;                // the generation this run's store came from (0: none)
  // Checkpoints: the images the index on disk names keep their file space until the next index is durable (an image dropped
  // meanwhile is retired, not freed), so no write lands where that index points.
  std::set<size_t> named;                          // images the latest index names
  std::vector<std::pair<size_t, size_t>> retired;  // (image, bytes) dropped while named: freed at the next checkpoint
  uint64_t ver = 0;                                // images written or dropped since the start (a checkpoint is due when it moved)
  std::vector<int> ckpt_sess;                      // the sessions the last checkpoint named (rank 0's server writes theirs)
  uint64_t clock = 0;
  std::vector<int> pinned;   // the sessions the current command uses: never evicted during it
  std::vector<int> live;     // the sessions running in the slots (a running request's): never evicted either; a session on an
                             // idle or paused line is not named here, so its conversation is cache like a stored one
  std::vector<int> last;     // the sessions the current command evicts after every other (its own conversation and its fork
                             // base): not pinned, so a window or a restore can still take their pieces
  std::vector<int> evicted;  // sessions evicted to make room, since rank 0's server last asked
  Sess& use(int h) {         // session h (made if new), used now
    Sess& s = sess[h];
    s.used = ++clock;
    return s;
  }
};

// The device's side of a store operation: everything that touches device memory, the disk tier's file, the ring or the clock
// in the middle of one. None of these is reached by a step that needs no new number.
class StoreDev {
 public:
  virtual ~StoreDev() = default;
  // the four arrays' slices of number c set to `byte` (only with poison: 0xFF when taken, 0xFD when given back)
  virtual void fill(uint32_t c, int byte) = 0;
  // entries [0, n) of the four arrays from number `from` into number `to`, queued on the stream (a fork's partial number)
  virtual void copy_entries(uint32_t to, uint32_t from, int n) = 0;
  // unit u's rows are the lists (kKvSlots of them; null: empty rows), after the stream has gone idle
  virtual void rows_set(int u, const Nums* lists) = 0;
  // n entries at the end of unit u's row of kv source k, from index i0 on, no wait
  virtual void rows_append(int u, int k, size_t i0, const uint32_t* numbers, size_t n) = 0;
  // number c's four slices to the gather buffer, their sum to the "out" sum of `off`, the write queued; the write's sequence
  virtual uint64_t image_out(uint32_t c, size_t off) = 0;
  // a bookmark's bytes likewise
  virtual uint64_t bk_out(const Nums& numbers, size_t bytes, size_t off) = 0;
  // the image at `off` read, its sum to the "back" sum, scattered into number c
  virtual void image_in(size_t off, uint32_t c) = 0;
  // a bookmark's image likewise
  virtual void bk_in(size_t off, const Nums& numbers, size_t bytes) = 0;
  // the staged writes go out
  virtual void disk_flush() = 0;
  // flush, then wait for write `seq` (before a read-back)
  virtual void disk_settle(uint64_t seq) = 0;
  // the model's stream is idle
  virtual void sync() = 0;
  // the file synced; the file stays at exit
  virtual void disk_sync() = 0;
  virtual void disk_keep() = 0;
  // the tier's file of `bytes` (reused as it is, or made); its size
  virtual size_t disk_open(const std::string& base, size_t bytes, size_t staging, bool reuse) = 0;
  virtual size_t disk_size() = 0;
  // the file made `bytes` long (disk_size() then says so); false, nothing changed, when the disk has no room
  virtual bool disk_grow(size_t bytes) = 0;
  // the file cut to `bytes` (disk_size() then says so): the space of a smaller tier given back; the checksums of the slots
  // below the new end stay. Nothing may lie at or past `bytes`.
  virtual bool disk_shrink(size_t bytes) = 0;
  // the bytes the file could still grow by: its directory's free space less the margin the start keeps
  virtual size_t disk_room() = 0;
  virtual std::string disk_path() = 0;
  // an image's checksum as it went out (back false) or as it came back (back true); a test may set the first
  virtual uint64_t sum(size_t off, bool back) = 0;
  virtual void sum_set(size_t off, uint64_t value) = 0;
  // the smallest `pos` over the ranks (one host all-gather); `pos` itself with no ring
  // `site`: the call's id (comm/host_gather.h HostCall): the ranks in different sites stop naming both
  virtual int agree_min(int pos, uint32_t site = 0) = 0;
  // the four ranks' values themselves, rank order (one host all-gather): the minimum alone cannot tell a rank that left early from one that has an answer
  virtual void agree_all(int v, uint32_t site, int all[4]) = 0;
  // four words of the inputs a decision is made from, compared by the caller across the ranks (one host all-gather): all = 4 x 4 words, rank order
  virtual void agree_words(const uint64_t mine[4], uint32_t site, uint64_t all[16]) = 0;
  // unix time, seconds
  virtual std::time_t now() = 0;
  // ---- the split: a piece's quarters (model/quarter_map.h)
  // this rank's quarter of number c's first `written` entries into slot `slot`, and the sums of the four quarters and of the whole
  // piece recorded for the slot; the number's bytes stay
  virtual void quarter(uint32_t c, uint32_t slot, int written) = 0;
  // the quarters of `slots` from the four ranks into the exchange buffer: this rank's slot quarters sent, the other ranks' received
  virtual void exchange(const std::vector<uint32_t>& slots) = 0;
  // the exchanged quarters into numbers[i], all four quarters of piece i (this rank's from its slot)
  virtual void unquarter(const std::vector<uint32_t>& numbers) = 0;
  // bad[i] = piece i in numbers[i] does not match the sums recorded for slots[i]; one sync for all
  virtual void compare(const std::vector<uint32_t>& slots, const std::vector<uint32_t>& numbers, std::vector<uint8_t>* bad) = 0;
  // whether number c's whole piece, re-read, matches the whole-piece sum recorded for slot `slot` (a ghost revived, a number shared)
  virtual bool sum_exact(uint32_t c, uint32_t slot, int written) = 0;
  // ---- a quarter's image on the disk tier: kDiskBytes = the quarter and a trailer (quarter_map.h), the file's own sum over both
  // slot's quarter and its recorded sums, with whose piece it is (handle, list, piece, written count), to `off`; the write's sequence
  virtual uint64_t slot_out(uint32_t slot, size_t off, int handle, int list, int piece, int written) = 0;
  // the image at `off` read, its sum to the "back" sum, its quarter into position idx (< 8) of the batch to exchange; false: the trailer
  // is not one (the piece is bad); *t is what the trailer says
  virtual bool quarter_read(size_t off, int idx, ::model::quarter::Trailer* t) = 0;
  // the quarters read so far (positions 0 .. n - 1) exchanged with the other ranks, as exchange() does for slots
  virtual void exchange_staged(int n) = 0;
  // bad[i] = piece numbers[i] does not match the sums its image's trailer recorded (the four quarters' and the whole piece's)
  virtual void compare_sums(const std::vector<uint32_t>& numbers, const std::vector<::model::quarter::Trailer>& trailers, std::vector<uint8_t>* bad) = 0;
  // a bookmark's quarter into `slots` (its numbers' bytes stay), and its return: the quarters exchanged, put into `numbers` and
  // checked against the sums recorded at the quarter; false: it arrived different
  virtual void quarter_bk(const Nums& numbers, const std::vector<uint32_t>& slots) = 0;
  virtual bool join_bk(const Nums& numbers, const std::vector<uint32_t>& slots) = 0;
  // ---- the imager: the image stream, never the model's. Nothing below waits for the model's stream or is waited for by it,
  // but image_wait (the broken-count fallback and a planned stop).
  // staging takes `bytes` more now (a write of them would not wait)
  virtual bool image_room(size_t bytes) = 0;
  // piece c's first `written` entries: this rank's quarter, the five sums and the trailer built on the image stream and queued for the
  // write at `off`; the write's sequence
  virtual uint64_t image_quarter(uint32_t c, size_t off, int handle, int list, int piece, int written) = 0;
  // a bookmark's bytes (the numbers' content): this rank's slice and the trailer's words, likewise; list is Store::kBkWindow or kBkEnd,
  // piece its position
  virtual uint64_t image_bookmark(const Nums& numbers, size_t bytes, size_t off, int handle, int list, int piece) = 0;
  // ends an image command: its writes go to the writer, and `seq` (the command's count) is recorded on the image stream
  virtual void image_end(uint64_t seq) = 0;
  // the highest command count whose work on the image stream has ended; no wait
  virtual uint64_t image_done() = 0;
  // waits for the image stream
  virtual void image_wait() = 0;
  // a bookmark's slice image at `off` read into the batch to exchange at unit idx (units: 1 for a window bookmark, 3 for the end), its sum
  // to the "back" sum; false: the trailer is not one
  virtual bool bookmark_read(size_t off, size_t bytes, int idx, ::model::quarter::Trailer* t) = 0;
  // the units read by bookmark_read (already exchanged, exchange_staged) laid end to end in the order of the ranks, put into `numbers`;
  // false: the four slices' sums or the whole's differ from the trailer's
  virtual bool bookmark_put(const Nums& numbers, size_t bytes, int idx, const ::model::quarter::Trailer& t) = 0;
};

class StoreCore {
 public:
  static constexpr int kNoLine = -2147483647 - 1;  // no session (a unit's own are -1 - u)

  // A host test builds the core with STORE_CORE_TEST_RULES and can switch one wrong rule on, to show that a test of that rule
  // fails when the rule's line is not there. The engine's build has no such member and every test of it is false.
  enum class TestRule {
    kNone,
    kVictimMru,           // eviction takes the most recently used session first
    kTakeIgnoresResv,     // a bookmark takes numbers the reservations need
    kNoCow,               // a shared number is written in place
    kCutKeepsBookmarks,   // a cut leaves the bookmarks above it
    kUnspillNoCut,        // a read-back with no number to take leaves the session whole
    kUnspillIgnoresSum,   // a piece that comes back different is kept
    kDiskVictimAny,       // the disk tier's victims include the sessions the command pins
    kGiveIgnoresRefs,     // a number given back goes to the pool though another list names it
    kImgFreeLeaks,        // a flight's numbers are not given back
    kKeptIgnoresRun,      // a kept index of another run is accepted
    kUnspillIgnoresCap,   // a read-back under a cap takes what fits, the cap aside
    kSplitIgnoresPin,     // a split takes a conversation a command pins or a request runs
    kNoTwinCut,           // a number handed out or written in place keeps its link to the slot it once equalled
    kSplitEvicts,         // a split that finds no quarter number makes room by evicting
    kGiveableNoSlots,     // the giveable count leaves out quarter numbers
    kSplitFullOnly,       // a split leaves every piece that is not full whole
    kBookmarksWhole,      // a split leaves the bookmarks whole
    kNoUnghost,           // a ghost whose link is cut stays a ghost
    kQuarterTrailerUnchecked,  // a quarter image whose trailer is another rank's or another piece's is read as if it were this piece's
    kIndexWriteUnagreed,  // a rank adopts its own index write's result, whatever the other ranks' were
    kJoinTakeUnpaid,      // a join takes a ghost back with no check, and an exchanged piece's take counts its own line's reservation as another's
    kJoinNoStamp,         // a join does not stamp its conversation used
    kUnspillUnbounded,    // a read-back ignores its `upto` and its piece limit J: one command for everything it can take
    kUnspillJoins,        // a read-back begins with the guard's join of the conversation's slots (a request's read-back never joins)
    kForkReadsTail,       // a fork's load reads its base's tail above the fork point back from the disk as well
    kUnspillCountDrops,   // a unit of a return whose lists used up its count goes on to the bookmarks and drops them (a count stops a command as J does)
    kSplitBookmarkNoImage,  // the split quarters a bookmark with no whole image written first: eviction drops it
    kUnspillReadsSplitBookmarks,  // a read-back reads the whole image of a bookmark that is still slots, beside its slots
    kNoGhostCut,          // a ghost taken keeps its image link: the return revives another's bytes
    kMarkIgnoresLine,     // a conversation on a line is marked
    kEvictPlain,          // eviction ignores the imager: a piece is written whole, as with no imager
    kSettleUnchecked,     // a job whose block was cut is settled as if valid
    kUnspillNoOpReturn,   // a fork site calls unspill though nothing at or below its point is out (three returns for a fork)
    kUnspillReturnsAlone, // a rank whose session is not out returns before the read-back's collective
  };
#ifdef STORE_CORE_TEST_RULES
  TestRule test_rule = TestRule::kNone;
  bool rule(TestRule r) const { return test_rule == r; }
  // How often each eviction step and each outcome of an eviction ran. evict_step and evict_piece are called only inside the core
  // (a nested entry writes no record), so no trace holds them; a replay counts them instead.
  struct EvictCounts {
    uint64_t steps = 0;           // evict_step ran
    uint64_t step_none = 0;       //   no session was left to give a piece
    uint64_t step_whole = 0;      //   no disk tier: the session went whole
    uint64_t step_piece = 0;      //   one piece of a session left (evict_piece)
    uint64_t piece_end = 0;       // evict_piece: an end bookmark left the pool
    uint64_t piece_snap_written = 0;  //   a window bookmark left, its image written or already there
    uint64_t piece_snap_dropped = 0;  //   a window bookmark dropped: no room on the disk
    uint64_t piece_entry_written = 0; //   a cache entry left, its image written or already there
    uint64_t piece_cut = 0;           //   a cache entry cut: no room on the disk
    uint64_t piece_none = 0;          //   the session had nothing left in the pool
  } evict_counts;
#define STORE_CORE_COUNT(f) (++evict_counts.f)
#else
  static constexpr bool rule(TestRule) { return false; }
#define STORE_CORE_COUNT(f) ((void)0)
#endif

  // What the digest takes from outside the store: the pool's lending (the windows' groups and the Engram cache's), and every
  // other tenant's groups and groups in flight (the lend table's ids 2 to 15; ids 0 and 1 are in the terms before them).
  // A tenant with neither adds nothing.
  struct Outside {
    uint64_t groups = 0, owed = 0, n0 = 0, cur_groups = 0, phase = 0;
    uint32_t tenant_groups[16] = {}, tenant_pending[16] = {};
  };
  // The disk tier's and the pool's counters since the start.
  struct TierCounters {
    uint64_t spills = 0, spill_bytes = 0, restores = 0, restore_bytes = 0, restored_pos = 0, wait_ns = 0, wait_max_ns = 0;
  };
  struct TierStats {
    uint64_t store_cap = 0, store_used = 0, resident = 0, on_disk = 0, disk_cap = 0, disk_used = 0;
    uint64_t spills = 0, spill_bytes = 0, restores = 0, restore_bytes = 0, restored_pos = 0, wait_ns = 0, wait_max_ns = 0;
  };
  struct PoolStats {
    uint64_t numbers = 0, free = 0, held = 0, shared = 0, reserved = 0, number_bytes = 0, image_numbers = 0;
    uint64_t on_lines = 0, in_memory = 0, on_disk = 0, sharing = 0;
  };
  struct Info {
    int saved = 0, end_pos = -1, restart = -1;
    bool split = false;  // a piece of it is a quarter slot
    int slots = 0;       // quarter slots it names (lists and bookmarks)
    std::vector<int> snaps;
    size_t bytes = 0, disk_bytes = 0;
    bool on_disk = false;
  };
  struct Chunks {
    uint64_t total = 0, mapped = 0, live_bytes = 0;
  };
  struct Holder {
    int handle = 0, line = -1;
    bool unit_own = false;
  };
  struct Holders {
    std::vector<Holder> holders;
    int64_t image_numbers = 0;  // numbers >= top that image flights hold
    int image_flights = 0;
  };

  StoreCore(const sgeom::StoreGeom& geom, StoreDev& device, chunks::Alloc* alloc);
  StoreCore(const StoreCore&) = delete;
  StoreCore& operator=(const StoreCore&) = delete;

  // ---- state: read by the engine's forwards and by the tests' invariants ---------------------------------------------
  sgeom::StoreGeom g;  // what is bound after the core exists (the host ring, the draft module) is set by the engine
  StoreDev& dev;
  chunks::Alloc* calloc;  // the pool's allocator: the windows and the lender use it too
  const int rank, nslots, nunits, list_len, max_ctx;
  int unit_of[kMaxSlots] = {}, slot_ctx_[kMaxSlots] = {};
  std::vector<uint32_t> refs;  // refs[c]: the lists and bookmarks naming number c (forks share numbers by count)
  int mapped_upto[kMaxSlots] = {};  // the position up to which every array of slot s is mapped (-1: none)
  int line_h[kMaxUnits];            // the session unit u holds (kNoLine: none; negative: the unit's own)
  int resv[kMaxSlots] = {};         // numbers reserved for slot s's request
  int resv_img[kMaxSlots] = {};     // numbers reserved for slot s's request's image rows, until it takes them
  size_t held_peak = 0;
  uint64_t sum_cuts = 0;    // the read-backs cut at a piece whose sums differed from the ones written (the agreed cut, equal on every rank)
  uint64_t stale_cuts = 0;  // the times the link of a number was cut where an entry had no image of its own to drop
  struct SplitCounts {
    uint64_t split_pieces = 0, split_deferred = 0, revives = 0, shares = 0, exchanged = 0, joins = 0, join_short = 0, join_bad = 0;
    uint64_t join_bytes = 0, join_ns = 0, join_max_ns = 0;  // join_ns, join_max_ns: a join command's own time on the engine thread, summed and its longest
    uint64_t unspill_cmds = 0;  // read-back commands that went past the early returns
    uint64_t unspill_taken = 0;  // numbers the read-backs took (the count a read-back command carries is counted against it)
    // the imager
    uint64_t imaged = 0, image_cmds = 0, image_voids = 0, image_waits = 0, image_wait_ns = 0, rides = 0, returns_read = 0, marked_bk = 0;
    uint64_t join_taken = 0;  // whole numbers the joins took (the count a join command carries is counted against it)
  } split_counts;
  int join_pieces = 64;  // J: the most pieces one join command takes (--kv-join-pieces)
  // Fault seam (DSGB10_JOIN_SHORT=N, a fault build): the N-th take of a join is refused once, as if no number were to be had. 0: none.
  int fault_join_take = 0, join_take_n = 0;
  // What STATS shows of the split: conversations with a slot, the slots in use, the quarter numbers open, the ghosts
  struct SplitView {
    uint64_t conv_split = 0, quarter_slots = 0, quarter_numbers = 0, ghost_numbers = 0;
  };
  SplitView split_view() const;
  // Whether a split command could open a quarter number now: a quarter slot with room, or a free number below the lend zone. A count
  // every rank holds alike; reading it sends nothing.
  bool split_can_open() const;
  TierCounters tier;
  std::unique_ptr<Store> store_s;  // made where the store is first needed
  // the pool's lending, for the digest (the engine reads its window and lender state)
  std::function<Outside()> outside = [] { return Outside(); };

  Store& store_of();
  bool has_store() const { return store_s != nullptr; }

  // ---- entries (each is one record of the trace) ---------------------------------------------------------------------
  // The command's sessions: never evicted; the running ones; the evict-last ones.
  void pin(const std::vector<int>& h, const std::vector<int>& live, const std::vector<int>& last);
  // Slot s's lists hold position `upto`: the first three lines of a step's check are inline, so a step that needs nothing new
  // makes no call and no device call.
  void ensure(int s, int upto) {
    if (s < 0 || s >= nslots) die("model: ensure_mapped in slot %d of %d", s, nslots);
    upto = std::max(0, std::min(upto, slot_ctx_[s] - 1));
    if (upto <= mapped_upto[s]) return;  // the whole check of a step that needs nothing new
    ensure_more(s, upto);
  }
  void slot_reserve(int s, int numbers, int img);  // `numbers`: what the server counted for the request (Model::slot_chunks_for)
  void slot_unreserve(int s);
  void slot_release(int s);
  void slot_attach(int s, int h, int upto = INT32_MAX);
  void save(int h, int from, int to, int unit);
  void load(int h, int to, int unit, int slot);
  const Nums* snap_buf(int h, int pos);          // a new bookmark's numbers (the caller fills them); null: no room
  const Nums& snap_get(int h, int pos);
  const Nums* snap_peek(int h, int pos);         // the bookmark's numbers, null when none or only on the disk tier
  const Nums* end_take(int h, int pos);          // the end bookmark's numbers to fill; null: none taken
  const Nums& end_get(int h);
  void drop(int h, int from);
  void drop_snap(int h, int pos);
  void free_session(int h);
  void copy_snaps(int src, int dst, int upto);
  void spill(int h);
  // `cap` > 0: the numbers this read-back may take (the lists first, in position order, then bookmarks while the count lasts;
  // at the cap it stops as it does when the pool has no room: the newest restart point that is whole). 0: no limit. kSpent: nothing
  // is left to take (the server's last command of a budget that ran out while pieces were still on the disk): it cuts as the cap did.
  // `upto` < INT32_MAX: only the pieces holding positions below it (the bookmarks at or below it): a fork's tail above its point stays
  // on the disk. `J` > 0: at most J pieces in this command, lists first and then bookmarks; stopping at J cuts nothing and drops
  // nothing, the next command goes on (a return is units of J). It never joins slots.
  static constexpr size_t kSpent = SIZE_MAX;
  void unspill(int h, size_t cap = 0, int upto = INT32_MAX, int J = 0);
  // What a read-back to `upto` still has to bring: lists below `upto` that are only on the disk and the bookmarks at or below it
  int unspill_left(int h, int upto) const;
  // The split. Each of the conversation's whole pieces that every holder keeps as cache (none pinned or running) and that has a
  // slot to be had becomes a quarter slot and its number a ghost: at most `budget` pieces. Never evicts: a piece with no
  // number to open a quarter number from stays whole (counted as deferred). Returns the pieces split.
  int split(int h, int budget);
  // The return: the pieces below position `upto` that are slots, in position order, become whole numbers again: a ghost taken
  // back, a number another holder made whole shared, else the quarters exchanged. At most `J` pieces and `cap` numbers (0: no
  // limit); stops at the first piece with no number. A piece that does not match its sums cuts the conversation at the smallest
  // such position on every rank. Returns the pieces below `upto` that are still slots.
  int join(int h, int upto, size_t cap, int J);
  void flush(int h);
  // The imager (geom.imager). image: at most `budget` of h's plain pieces get a quarter image queued on the image stream
  // (file space taken, a job appended; the entry is not touched); a conversation a request is in flight on (`live`) gets nothing. Returns
  // the pieces queued. settle: every job of a command at or below `upto` that is still valid makes its entry clean; then every conversation that
  // is on no line and has no request in flight (`live`) has its clean pieces marked (ghosts: host bookkeeping, no device call). Returns the
  // pieces marked. A job whose block was cut since is void and its file space goes back.
  int image(int h, int budget, const std::vector<int>& live);
  int settle(uint64_t upto, const std::vector<int>& live);
  // image everything of h and settle at once (a planned stop, a spill): the wait is the caller's own
  void image_all(int h);
  bool imager() const { return g.imager; }
  uint64_t image_seq() const { return store_s ? store_s->img_seq : 0; }
  struct ImageView {
    uint64_t clean_numbers = 0, image_pending = 0, ghost_numbers = 0;
  };
  ImageView image_view() const;
  uint64_t image_pending() const { return store_s ? store_s->jobs.size() : 0; }  // jobs queued, not settled (a replicated count: no collective)
  int image_todo(int h) const;  // pieces of h that have no image and none queued (what the imager still has to queue for it)
  bool disk_resize(size_t bytes);
  uint64_t kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes, std::string* why);
  void disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id);
  void keep(uint64_t gen);
  void checkpoint(uint64_t gen);
  bool img_take(int key, size_t rows, int slot);
  void img_free(int key);
  bool evict_step(int h);
  bool evict_piece(int v, int h, bool allow_plain = false);  // allow_plain: a plain piece is imaged and waited for first (a broken count's fallback, a planned stop)
  int vacate(uint32_t from);
  bool make_room(size_t removed);
  void alloc_shrink(size_t top);
  void alloc_grow(size_t nums);
  void set_factor(int h, int factor);
  std::vector<int> take_evicted();  // not an entry of the trace's numbering: a `V` line

  // The trace file opens here: after the store's memory is reserved and its disk tier opened, so its geometry and pool lines
  // describe the store as it stands. Without DSGB10_STORE_TRACE this is one test.
  void trace_open();

  // ---- reads (no record) ---------------------------------------------------------------------------------------------
  int held(int s) const;  // chunk numbers slot s's session's lists hold
  int held_slots(int s) const;  // quarter slots slot s's session names (lists and bookmarks)
  size_t giveable(const std::vector<int>& keep) const { return giveable_joint(keep, {}); }
  // giveable() with the conversations of `lines` counted as if no line held them: the whole numbers only the counted
  // conversations name, and the quarter numbers all of whose used slots only they name (one count, whoever asks)
  size_t giveable_joint(const std::vector<int>& keep, const std::vector<int>& lines) const;
  size_t restore_chunks(int h) const;
  // the numbers h's lists that hold positions below `upto` hold in the pool, and those lists that are only on the disk tier (0 for
  // both: no such session)
  void list_numbers(int h, int upto, int* in_pool, int* on_disk) const;
  // and, in `split`, those lists below `upto` that are quarter slots (counted in neither of the others)
  void list_numbers(int h, int upto, int* in_pool, int* on_disk, int* split) const;
  // what a join of h's pieces below `upto` would do: the pieces taken back from their ghosts (or shared) and the pieces exchanged
  // (a bookmark is always exchanged); reads the store's records only
  void join_plan(int h, int upto, int* back, int* exch) const;
  // the conversation's window bookmarks by position (ascending) with the numbers each holds in the pool (0: only on the disk tier)
  std::vector<std::pair<int, int>> snap_numbers(int h) const;
  Chunks chunks() const;
  Info info(int h) const;
  int factor(int h) const;
  int bookmark_numbers(int snaps) const;
  int cow_numbers(int h_src, int h_own, int pos) const;
  std::vector<int> sessions() const;
  const imgrows::Ledger::Rows* img_find(int key) const;
  size_t img_numbers() const;
  Holders holders_from(uint32_t from) const;
  size_t disk_capacity() const;
  size_t disk_bytes() const;
  std::string disk_path();
  std::vector<int> checkpointed() const;
  uint64_t version() const;
  uint64_t kept_generation() const;
  TierStats tier_stats() const;
  PoolStats pool_stats() const;
  // The store's digest: sessions with their lists, bookmarks and images, the numbers' counts and images, the reservations,
  // the evict-last list, the pool's lending (`o`); compared across the ranks' logs under DSGB10_STORE_CHECK=1.
  uint64_t digest(const Outside& o) const;
  uint64_t digest() const { return digest(outside()); }
  // The wide digest: `today` (the digest), then what it leaves out and a replay must reproduce (store_trace.h).
  uint64_t wide_digest(uint64_t today) const;

  // ---- inside the store ----------------------------------------------------------------------------------------------
  Store::Sess& use_blocks(int h, int upto = INT32_MAX, bool bound_readback = false);  // session h (made if new) for an operation that needs its blocks below `upto`: joined and read back first
  Store::Sess& line_sess(int s);
  void write_rows(int u, const Store::Sess* x);
  bool take_nums(int h, size_t n, Nums* out);
  void give_nums(Nums& v);
  void release(Ref c);  // one reference to the entry's number goes
  void join_whole(int h, int upto = INT32_MAX);  // join(h, all): what no number could be found for is cut, as a read-back cuts
  size_t entry_image(const Store::Sess& x, int k, size_t i) const;  // the image an entry's piece has on the disk tier (a slot's: its own); kNone: none
  void slot_image(int h, int k, size_t i, uint32_t slot);  // the slot's quarter image, when it has none and there is room (write-behind)
  void qsnap_drop(Store::Sess& x, int pos);  // the slots of a split window bookmark go (its key stays only if the caller keeps it)
  void qend_drop(Store::Sess& x);            // likewise the end bookmark's
  size_t bk_slots(size_t bytes) const;       // quarter slots a bookmark of `bytes` stands on: a quarter of it, in slots
  void cut_twin(uint32_t c);          // number c's content no longer equals a slot's
  bool take_quarter_number(uint32_t* c);  // a number for quarter slots, taken without evicting and outside the lend zone; false: none
  void release_slot(uint32_t slot);   // one holder of the slot goes; its number goes back when the last slot of it is free
  bool dirty(Ref c);   // the entry's number is about to change content: its disk image link is cut when it is the only holder
  void entry_dirty(int h, Ref c, size_t* dl);  // dirty(c), then the entry's own image dropped
  void share_nums(const Refs& from, size_t n, Refs* out);
  void share_nums(const Nums& from, size_t n, Nums* out);  // a bookmark's numbers
  bool take_extra(int h, size_t n, Nums* out);
  bool take_restore(int h, size_t n, Nums* out);
  // a take of a join (a ghost taken back, an exchanged piece's number, a bookmark's numbers): a take into a conversation on a line is paid
  // first by that line's own reservation (its held rises as the free numbers fall); any take must leave every other reservation covered
  bool join_take_ok(int h, size_t n);
  bool take_join(int h, size_t n, Nums* out);
  bool own_tail(int h, int pos);
  bool own_bk(int h, Nums& b);
  bool grow(int s, int upto);
  void ensure_more(int s, int upto);
  size_t disk_take(int h, size_t bytes, bool victims = true);  // victims false: no other conversation's images are dropped for the space
  void img_drop(size_t off, size_t bytes);
  void space_free(size_t off, size_t bytes);
  void drop_images(int h);
  void free_lists(int h);
  size_t img_write(int h, uint32_t c);
  size_t bk_write(int h, const Nums& b, size_t bytes);
  void write_sealed(int h);
  void cut_at(int h, int pos);
  std::vector<uint8_t> kept_index(uint64_t gen, bool all, std::vector<int>* hs, std::set<size_t>* names);
  // the imager's pieces
  size_t snap_ibytes() const { return g.imager ? sgeom::kBookmarkImageBytes : snap_bytes(); }  // a window bookmark's image on the disk tier
  size_t end_ibytes() const { return g.imager ? sgeom::kEndImageBytes : end_bytes(); }        // the end bookmark's
  bool link_proven(uint32_t c, size_t off) const;  // number c's content is image `off` (ndisk, dnum and a held image agree)
  bool has_clean(const Store::Sess& x) const;      // a piece with an image that is held in the pool (eviction can give it back, writing nothing)
  // how an image job gets its file space and staging: in a decoding run (no other conversation's images are dropped, no wait for staging), when
  // nothing runs (they may be), or as a fallback that may wait for staging
  static constexpr int kQRide = 0, kQIdle = 1, kQNow = 2;
  bool queue_piece(int h, int k, int i, int written, int mode);  // a list piece's job; false: no file space or no staging room
  bool queue_bookmark(int h, int k, int pos, const Nums& b, int mode);
  int image_impl(int h, int budget, const std::vector<int>& live, int mode);
  int mark_conv(int h, const std::vector<int>& live);  // conversation h's clean pieces become ghosts (a line or a request in flight: none)
  void ghost_group_dissolve(size_t off, uint32_t except);  // a bookmark's ghost group: every block of it becomes plain free
  void settle_jobs(uint64_t upto, const std::vector<int>& live);
  bool image_now(int h, int k, int piece, const Nums* b);  // queue one piece's image, wait for the image stream, settle it: the broken count's fallback and a stop's
  bool revive_piece(int h, int k, int i, size_t off);  // a ghost (or another holder's revived block) is the entry's block again; false: read it
  size_t chunk_image() const { return g.image_bytes; }
  size_t snap_bytes() const { return g.snap_bytes; }
  size_t end_bytes() const { return g.end_bytes; }
};

// the store's pieces of arithmetic on positions, shared with the engine's device copies
int snap_q(int s);
int ent_pos(int k, int i);
bool end_usable(int end_pos, int saved);
bool snap_usable(int s, int saved);
size_t bk_nums(size_t bytes);
int nums_for_pos(int k, int pos);

// The latest position a session holding `saved` positions of pages, the end snapshot at end_pos (-1: none)
// and window snapshots at the keys of `snaps` can restart from: the end snapshot's, else the latest
// usable snapshot's q; -1 when there is none.
template <class M>
int restart_of(int saved, int end_pos, const M& snaps) {
  if (end_usable(end_pos, saved)) return end_pos;
  for (auto i = snaps.rbegin(); i != snaps.rend(); ++i)
    if (snap_usable(i->first, saved)) return snap_q(i->first);
  return -1;
}

}  // namespace model::detail
