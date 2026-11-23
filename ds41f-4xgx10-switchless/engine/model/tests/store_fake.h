// A fake device for the store's bookkeeping (model/store_core.h; host only): every number carries a 64-bit tag, the content
// the line's kernels last wrote there; copies, images and read-backs move tags. An image's sum going out is a hash of the tags
// that went out, and coming back of the tags that came back, so a read-back checks equal unless a test spoils the image. That
// lets a test say "the bytes this conversation reads back are the bytes it wrote" with no GPU. The disk tier is a size;
// writes are immediate. Every call is recorded (name, arguments, in order). In a replay the answers the engine's device, the
// ring and the clock gave are the trace's.
#pragma once
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "model/store_core.h"

namespace model::detail {

class FakeDev : public StoreDev {
 public:
  // --- content
  std::map<uint32_t, uint64_t> tag;             // number -> content tag (absent: never written)
  std::map<size_t, std::vector<uint64_t>> image;  // file offset -> the tags that went out there (one for a cache entry)
  std::map<size_t, uint64_t> out_sum, back_sum;
  std::map<int, std::vector<Nums>> rows;         // unit -> the device's rows (kKvSlots lists)
  uint64_t next_tag = 1000;
  size_t file_size = 0;
  std::string file_path;
  std::time_t clock_now = 1700000000;
  bool opened = false;
  uint64_t seq = 0;
  // --- answers a replay feeds (name -> values in the order asked); a test leaves them empty and gets the defaults
  std::map<std::string, std::deque<int64_t>> answers;
  // --- the call log
  std::vector<std::string> calls;
  int flushes = 0, syncs = 0;

  uint64_t wrote(uint32_t c) { return tag[c] = ++next_tag; }  // what a line's kernels do to a number they write
  uint64_t tag_of(uint32_t c) const {
    const auto it = tag.find(c);
    return it == tag.end() ? 0 : it->second;
  }
  static uint64_t hash(const std::vector<uint64_t>& t) {
    uint64_t h = 1469598103934665603ull;
    for (const uint64_t x : t) h = (h ^ x) * 1099511628211ull;
    return h;
  }
  int64_t answer(const char* name, int64_t dflt) {
    auto it = answers.find(name);
    if (it == answers.end() || it->second.empty()) return dflt;
    const int64_t v = it->second.front();
    it->second.pop_front();
    return v;
  }
  void log(const std::string& s) { calls.push_back(s); }

  void fill(uint32_t c, int byte) override {
    log("fill " + std::to_string(c) + " " + std::to_string(byte));
    tag[c] = byte == 0xFF ? 0xFFFFull : 0xFDFDull;
  }
  void copy_entries(uint32_t to, uint32_t from, int n) override {
    log("copy_entries " + std::to_string(to) + " " + std::to_string(from) + " " + std::to_string(n));
    tag[to] = tag_of(from);
  }
  void rows_set(int u, const Nums* lists) override {
    log("rows_set " + std::to_string(u));
    std::vector<Nums>& r = rows[u];
    r.assign(kKvSlots, Nums());
    if (lists)
      for (int k = 0; k < kKvSlots; ++k) r[k] = lists[k];
  }
  void rows_append(int u, int k, size_t i0, const uint32_t* numbers, size_t n) override {
    log("rows_append " + std::to_string(u) + " " + std::to_string(k));
    std::vector<Nums>& r = rows[u];
    if (r.empty()) r.assign(kKvSlots, Nums());
    if (r[k].size() < i0 + n) r[k].resize(i0 + n, 0);
    for (size_t i = 0; i < n; ++i) r[k][i0 + i] = numbers[i];
  }
  uint64_t image_out(uint32_t c, size_t off) override {
    log("image_out " + std::to_string(c) + " " + std::to_string(off));
    image[off] = {tag_of(c)};
    out_sum[off] = hash(image[off]);
    return ++seq;
  }
  uint64_t bk_out(const Nums& numbers, size_t bytes, size_t off) override {
    log("bk_out " + std::to_string(numbers.size()) + " " + std::to_string(bytes) + " " + std::to_string(off));
    std::vector<uint64_t> t;
    for (const uint32_t c : numbers) t.push_back(tag_of(c));
    image[off] = t;
    out_sum[off] = hash(t);
    return ++seq;
  }
  void image_in(size_t off, uint32_t c) override {
    log("image_in " + std::to_string(off) + " " + std::to_string(c));
    const auto it = image.find(off);
    if (it == image.end()) {  // an image this run never wrote (a kept file's): its sum is the one given to sum_set
      tag[c] = 0x4B455054ull;
      back_sum[off] = out_sum[off];
      return;
    }
    tag[c] = it->second.empty() ? 0 : it->second[0];
    back_sum[off] = hash(it->second);
  }
  void bk_in(size_t off, const Nums& numbers, size_t bytes) override {
    log("bk_in " + std::to_string(off) + " " + std::to_string(numbers.size()) + " " + std::to_string(bytes));
    const auto it = image.find(off);
    if (it == image.end()) {
      for (const uint32_t c : numbers) tag[c] = 0x4B455054ull;
      back_sum[off] = out_sum[off];
      return;
    }
    for (size_t i = 0; i < numbers.size(); ++i) tag[numbers[i]] = i < it->second.size() ? it->second[i] : 0;
    back_sum[off] = hash(it->second);
  }
  void disk_flush() override { ++flushes; }
  void disk_settle(uint64_t) override { ++flushes; }
  void sync() override { ++syncs; }
  void disk_sync() override { log("disk_sync"); }
  void disk_keep() override { log("disk_keep"); }
  size_t disk_open(const std::string& base, size_t bytes, size_t, bool reuse) override {
    log("disk_open " + base + " " + std::to_string(bytes) + " " + std::to_string(reuse));
    opened = true;
    file_path = base + ".spill";
    file_size = (size_t)answer("disk_open", (int64_t)bytes);
    return file_size;
  }
  size_t disk_size() override { return file_size; }
  size_t room = (size_t)1 << 50;  // what disk_room() offers
  size_t disk_room() override { return room; }
  bool disk_grow(size_t bytes) override {
    log("disk_grow " + std::to_string(bytes));
    file_size = std::max(file_size, bytes);
    return true;
  }
  bool disk_shrink(size_t bytes) override {
    log("disk_shrink " + std::to_string(bytes));
    file_size = std::min(file_size, bytes);
    out_sum.erase(out_sum.lower_bound(file_size), out_sum.end());  // the slots past the new end go, those below stay
    back_sum.erase(back_sum.lower_bound(file_size), back_sum.end());
    return true;
  }
  std::string disk_path() override { return file_path; }
  uint64_t sum(size_t off, bool back) override { return back ? back_sum[off] : out_sum[off]; }
  void sum_set(size_t off, uint64_t value) override { out_sum[off] = value; }
  // the four ranks' values: a test that builds four cores gives each fake a `gather` that meets the others; one core is all four
  std::function<void(int, uint32_t, int*)> gather;
  int agree_min(int pos, uint32_t site) override {
    if (!gather) return (int)answer("agree_min", pos);
    int all[4];
    gather(pos, site, all);
    return std::min(std::min(all[0], all[1]), std::min(all[2], all[3]));
  }
  std::function<void(const uint64_t*, uint32_t, uint64_t*)> gather_words;
  int agree_alls = 0;
  void agree_all(int v, uint32_t site, int all[4]) override {
    ++agree_alls;
    if (gather) return gather(v, site, all);
    for (int r = 0; r < 4; ++r) all[r] = v;
  }
  void agree_words(const uint64_t mine[4], uint32_t site, uint64_t all[16]) override {
    if (gather_words) return gather_words(mine, site, all);
    for (int r = 0; r < 4; ++r)
      for (int k = 0; k < 4; ++k) all[r * 4 + k] = mine[k];
  }
  // --- the split: a quarter's content is the piece's tag (every rank holds the same bytes, so one tag stands for the four quarters)
  std::map<uint32_t, uint64_t> slot_tag;         // slot -> the tag of the piece it was made from
  std::vector<uint64_t> stage;                   // the exchange buffer: the tag of each piece exchanged
  std::set<size_t> spoil_pieces;                 // test seam: the exchanged piece at these indexes (in the call's order) arrives different
  size_t exchanged_pieces = 0;
  void quarter(uint32_t c, uint32_t slot, int written) override {
    log("quarter " + std::to_string(c) + " " + std::to_string(slot) + " " + std::to_string(written));
    slot_tag[slot] = tag_of(c);
    slot_entries[slot] = written;
  }
  std::map<uint32_t, int> slot_entries;  // slot -> the entries its recorded sum covers
  void exchange(const std::vector<uint32_t>& slots) override {
    log("exchange " + std::to_string(slots.size()));
    stage.clear();
    for (const uint32_t s : slots) {
      const auto it = slot_tag.find(s);
      uint64_t t = it == slot_tag.end() ? 0 : it->second;
      if (spoil_pieces.count(exchanged_pieces)) t ^= 0xBADull;
      ++exchanged_pieces;
      stage.push_back(t);
    }
  }
  void unquarter(const std::vector<uint32_t>& numbers) override {
    log("unquarter " + std::to_string(numbers.size()));
    for (size_t i = 0; i < numbers.size() && i < stage.size(); ++i) tag[numbers[i]] = stage[i];
  }
  void compare(const std::vector<uint32_t>& slots, const std::vector<uint32_t>& numbers, std::vector<uint8_t>* bad) override {
    log("compare " + std::to_string(slots.size()));
    bad->assign(slots.size(), 0);
    for (size_t i = 0; i < slots.size(); ++i) {
      const auto it = slot_tag.find(slots[i]);
      (*bad)[i] = it == slot_tag.end() || tag_of(numbers[i]) != it->second;
    }
  }
  std::map<uint32_t, std::vector<uint64_t>> bk_content;  // a bookmark's first slot -> the tags of its numbers
  void quarter_bk(const Nums& numbers, const std::vector<uint32_t>& slots) override {
    log("quarter_bk " + std::to_string(numbers.size()) + " " + std::to_string(slots.size()));
    std::vector<uint64_t> t;
    for (const uint32_t c : numbers) t.push_back(tag_of(c));
    bk_content[slots[0]] = t;
  }
  bool join_bk(const Nums& numbers, const std::vector<uint32_t>& slots) override {
    log("join_bk " + std::to_string(numbers.size()) + " " + std::to_string(slots.size()));
    const auto it = bk_content.find(slots[0]);
    if (it == bk_content.end()) return false;
    const bool spoiled = spoil_pieces.count(exchanged_pieces++) != 0;
    for (size_t i = 0; i < numbers.size(); ++i) tag[numbers[i]] = i < it->second.size() ? it->second[i] ^ (spoiled ? 0xBADull : 0) : 0;
    return !spoiled;
  }
  // --- a quarter's image: the piece's tag stands for the four quarters; the trailer is kept as it was written
  struct QImg {
    uint64_t tag = 0;
    model::quarter::Trailer t;
  };
  std::map<size_t, QImg> qimage;
  std::vector<uint64_t> stage_in = std::vector<uint64_t>(8, 0);
  std::set<size_t> spoil_qimg;      // test seam: the image at these offsets reads back with a different sum
  std::set<size_t> spoil_trailer;   // test seam: the trailer at these offsets is another rank's
  int rank = 0;
  uint64_t slot_out(uint32_t slot, size_t off, int handle, int list, int piece, int written) override {
    log("slot_out " + std::to_string(slot) + " " + std::to_string(off));
    QImg q;
    q.tag = slot_tag[slot];
    for (auto& s : q.t.qsum) s = q.tag;
    q.t.wsum = q.tag, q.t.rank = (uint32_t)rank, q.t.handle = (uint32_t)handle, q.t.list = (uint32_t)list, q.t.piece = (uint32_t)piece;
    q.t.written = (uint32_t)written;
    qimage[off] = q;
    image[off] = {q.tag};
    out_sum[off] = hash(image[off]);
    return ++seq;
  }
  bool quarter_read(size_t off, int idx, model::quarter::Trailer* t) override {
    log("quarter_read " + std::to_string(off) + " " + std::to_string(idx));
    const auto it = qimage.find(off);
    if (it == qimage.end()) return false;
    back_sum[off] = hash(image[off]) ^ (spoil_qimg.count(off) ? 0xBADull : 0);
    stage_in[(size_t)idx] = it->second.tag;
    *t = it->second.t;
    if (spoil_trailer.count(off)) t->rank ^= 1;
    return true;
  }
  void exchange_staged(int n) override {
    log("exchange_staged " + std::to_string(n));
    stage.assign(stage_in.begin(), stage_in.begin() + n);
  }
  void compare_sums(const std::vector<uint32_t>& numbers, const std::vector<model::quarter::Trailer>& trailers, std::vector<uint8_t>* bad) override {
    log("compare_sums " + std::to_string(numbers.size()));
    bad->assign(numbers.size(), 0);
    for (size_t i = 0; i < numbers.size(); ++i) (*bad)[i] = tag_of(numbers[i]) != trailers[i].wsum;
  }
  bool sum_exact(uint32_t c, uint32_t slot, int written) override {
    log("sum_exact " + std::to_string(c) + " " + std::to_string(slot));
    const auto it = slot_tag.find(slot);
    const auto w = slot_entries.find(slot);
    return it != slot_tag.end() && tag_of(c) == it->second && w != slot_entries.end() && w->second == written;  // the sum covers what was recorded
  }
  std::time_t now() override { return (std::time_t)answer("now", (int64_t)clock_now); }
  // --- the imager: every call below is on the image stream; none of them is a call on the model's. The names say so in the log.
  bool room_ok = true;                   // test seam: staging has no room for the next image
  uint64_t ended = 0, done_cap = UINT64_MAX;  // image_end's highest count; a test caps what image_done reads (one rank's slow stream)
  int waits = 0;                         // image_wait calls
  struct BkImg {
    std::vector<uint64_t> tags;
    model::quarter::Trailer t;
  };
  std::map<size_t, BkImg> bkimage;
  std::set<size_t> spoil_bk;             // test seam: the bookmark image at these offsets reads back different
  std::vector<uint64_t> bk_stage;
  bool bk_spoiled = false;
  model::quarter::Trailer bk_trailer;
  size_t room_bytes = SIZE_MAX;          // test seam: this rank's staging takes this many bytes more (each image call spends its block)
  bool image_room(size_t bytes) override { return room_ok && room_bytes >= bytes; }
  uint64_t image_quarter(uint32_t c, size_t off, int handle, int list, int piece, int written) override {
    log("image_quarter " + std::to_string(c) + " " + std::to_string(off));
    QImg q;
    q.tag = tag_of(c);
    for (auto& s : q.t.qsum) s = q.tag;
    q.t.wsum = q.tag, q.t.rank = (uint32_t)rank, q.t.handle = (uint32_t)handle, q.t.list = (uint32_t)list, q.t.piece = (uint32_t)piece;
    q.t.written = (uint32_t)written;
    qimage[off] = q;
    image[off] = {q.tag};
    out_sum[off] = hash(image[off]);
    quarter_bytes_out += model::sgeom::kQuarterDiskBytes;
    if (room_bytes != SIZE_MAX) room_bytes -= std::min(room_bytes, (size_t)model::sgeom::kQuarterDiskBytes);
    return ++seq;
  }
  uint64_t image_bookmark(const Nums& numbers, size_t bytes, size_t off, int handle, int list, int piece) override {
    log("image_bookmark " + std::to_string(numbers.size()) + " " + std::to_string(off));
    BkImg b;
    for (const uint32_t c : numbers) b.tags.push_back(tag_of(c));
    b.t.wsum = hash(b.tags), b.t.rank = (uint32_t)rank, b.t.handle = (uint32_t)handle, b.t.list = (uint32_t)list, b.t.piece = (uint32_t)piece;
    b.t.written = (uint32_t)bytes;
    bkimage[off] = b;
    image[off] = b.tags;
    out_sum[off] = hash(image[off]);
    const size_t blk = list == model::detail::Store::kBkEnd ? model::sgeom::kEndImageBytes : model::sgeom::kBookmarkImageBytes;
    quarter_bytes_out += blk;
    if (room_bytes != SIZE_MAX) room_bytes -= std::min(room_bytes, blk);
    return ++seq;
  }
  uint64_t quarter_bytes_out = 0;        // what the image calls wrote, in image bytes
  void image_end(uint64_t n) override {
    log("image_end " + std::to_string(n));
    ended = std::max(ended, n);
    ++flushes;
  }
  uint64_t image_done() override { return std::min(ended, done_cap); }
  void image_wait() override {
    log("image_wait");
    ++waits;
  }
  bool bookmark_read(size_t off, size_t, int, model::quarter::Trailer* t) override {
    log("bookmark_read " + std::to_string(off));
    const auto it = bkimage.find(off);
    if (it == bkimage.end()) return false;
    back_sum[off] = hash(image[off]) ^ (spoil_qimg.count(off) ? 0xBADull : 0);
    bk_stage = it->second.tags;
    bk_trailer = it->second.t;
    bk_spoiled = spoil_bk.count(off) != 0;
    *t = it->second.t;
    return true;
  }
  bool bookmark_put(const Nums& numbers, size_t, int, const model::quarter::Trailer& t) override {
    log("bookmark_put " + std::to_string(numbers.size()));
    const bool spoiled = bk_spoiled;
    for (size_t i = 0; i < numbers.size(); ++i) tag[numbers[i]] = i < bk_stage.size() ? bk_stage[i] ^ (spoiled ? 0xBADull : 0) : 0;
    return !spoiled && hash(bk_stage) == t.wsum;
  }
};

}  // namespace model::detail
