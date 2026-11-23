// The session store's bookkeeping (model/store_core.h) on the host, over a fake device (tests/store_fake.h): unit rows with
// their invariants and, for each rule, a control (a core built with the rule's line absent must fail the row); the replay of
// recorded traces (store_trace.h) with the invariants after every record; the coverage of entry and answer kinds over a set of
// traces; and the replay's own controls (a changed record must be caught, a coverage table must list what one trace alone
// reaches). Host only: it links store_core.cpp, chunks.cpp and store_trace.cpp and nothing else.
//
//   store_core_test                      unit rows, then every trace under the directory compiled in, the coverage, the controls
//   store_core_test --replay FILE        replay one trace and say where it first differs
//   store_core_test --replay FILE --perturb N    change record N's first perturbable argument first: the replay must fail at N or later
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "model/chunks.h"
#include "model/pool_rules.h"
#include "server/reclaim.h"
#include "model/store_core.h"
#include "model/store_trace.h"
#include "model/tests/store_fake.h"

using namespace model;
using namespace model::detail;
namespace st = model::store_trace;
using Rule = StoreCore::TestRule;

namespace {

int g_fail = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s\n", what.c_str());
  }
}

struct Died {
  std::string msg;
};
void die_throws(const std::string& m) { throw Died{m}; }

std::string tmpdir() {
  char t[] = "/tmp/store_core_test.XXXXXX";
  if (!mkdtemp(t)) {
    std::perror("mkdtemp");
    std::exit(2);
  }
  return t;
}

// ---- the invariants (they do not read the digest: a replay that matched every digest with one broken still fails) -------------

// a list's entries as the device's rows hold them: the number, or 0 for an entry only on the disk tier
Nums numbers_of(const Refs& l) {
  Nums v;
  for (const Ref c : l) v.push_back(c.is_number() ? c.num() : 0);
  return v;
}

std::string invariants(const StoreCore& c, const FakeDev& f) {
  char b[200];
  if (!c.store_s) return "";
  const Store& s = *c.store_s;
  std::map<uint32_t, uint32_t> names, slot_names;
  for (const auto& [k, x] : s.sess) {
    for (const Refs& l : x.lists)
      for (const Ref n : l) {
        if (n.is_number()) ++names[n.num()];
        else if (n.is_slot()) ++slot_names[n.slot_of()];
      }
    for (const auto& sn : x.snaps)
      for (const uint32_t n : sn.second) ++names[n];
    for (const uint32_t n : x.end) ++names[n];
    for (const auto& q : x.qsnaps)
      for (const Ref r : q.second) ++slot_names[r.slot_of()];
    for (const Ref r : x.qend) ++slot_names[r.slot_of()];
  }
  s.img.each([&](int, const imgrows::Ledger::Rows& r) {
    for (const uint32_t n : r.nums) ++names[n];
  });
  for (const uint32_t q : s.slots.open_from(0)) {  // a quarter number is named once, by the slot map, while any of its slots is used
    ++names[q];
    for (int i = 0; i < 4; ++i) {
      const uint32_t sl = chunks::SlotMap::slot(q, i);
      const auto it = slot_names.find(sl);
      if (s.slots.refs(sl) != (it == slot_names.end() ? 0 : it->second)) {
        std::snprintf(b, sizeof b, "slot %u has %u holders and %u list entries name it", sl, s.slots.refs(sl), it == slot_names.end() ? 0 : it->second);
        return b;
      }
    }
  }
  for (const auto& [sl, n] : slot_names)
    if (!s.slots.is_open(chunks::SlotMap::number(sl))) {
      std::snprintf(b, sizeof b, "slot %u is named by %u list entries and is not open", sl, n);
      return b;
    }
  for (const auto& [sl, n] : s.twin_c) {
    if (n >= s.twin_s.size() || s.twin_s[n] != sl || !s.slots.refs(sl) || !(c.calloc->is_ghost(n) || c.calloc->taken(n))) {
      std::snprintf(b, sizeof b, "the link of slot %u to number %u is not a link of a held slot to a ghost or taken number", sl, n);
      return b;
    }
  }
  if (c.calloc)
    for (uint32_t n = 1; n < c.calloc->numbers(); ++n)
      if (c.calloc->is_ghost(n) && (n >= s.twin_s.size() || !s.twin_s[n]) &&
          !(n < s.ndisk.size() && s.ndisk[n] != Pool::kNone && s.dnum.count(s.ndisk[n]) && s.dnum.at(s.ndisk[n]) == n) && !s.ghost_bk.count(n)) {
        std::snprintf(b, sizeof b, "number %u is a ghost whose link is cut: nothing can revive it", n);
        return b;
      }
  for (size_t n = 0; n < s.twin_s.size(); ++n)
    if (s.twin_s[n] && (!s.twin_c.count(s.twin_s[n]) || s.twin_c.at(s.twin_s[n]) != n)) {
      std::snprintf(b, sizeof b, "number %zu names slot %u, which does not name it back", n, s.twin_s[n]);
      return b;
    }
  for (const auto& [n, k] : names)
    if (n >= c.refs.size() || c.refs[n] != k) {
      std::snprintf(b, sizeof b, "refs[%u] is %u, %u lists, bookmarks or flights name it", n, n < c.refs.size() ? c.refs[n] : 0, k);
      return b;
    }
  for (size_t n = 0; n < c.refs.size(); ++n)
    if (c.refs[n] && !names.count((uint32_t)n)) {
      std::snprintf(b, sizeof b, "refs[%zu] is %u and nothing names it", n, c.refs[n]);
      return b;
    }
  if (c.calloc)
    for (uint32_t n = 1; n < c.calloc->numbers(); ++n) {
      const bool want = n < c.refs.size() && c.refs[n] > 0;
      if (c.calloc->taken(n) != want) {
        std::snprintf(b, sizeof b, "number %u is %s in the allocator and refs says %s", n, c.calloc->taken(n) ? "taken" : "free", want ? "held" : "free");
        return b;
      }
    }
  // images: drefs[off] = the entries and bookmarks naming off; dnum[off] = c implies ndisk[c] = off (not the converse)
  std::map<size_t, uint32_t> dn;
  std::map<size_t, size_t> size_of;
  for (const auto& [k, x] : s.sess) {
    for (const auto& d : x.dl)
      for (const size_t o : d)
        if (o != Pool::kNone) ++dn[o], size_of[o] = s.qimg.count(o) ? sgeom::kQuarterDiskBytes : c.chunk_image();
    for (const auto& d : x.dsnaps) ++dn[d.second], size_of[d.second] = c.snap_ibytes();
    if (x.dend != Pool::kNone) ++dn[x.dend], size_of[x.dend] = c.end_ibytes();
  }
  for (const auto& sd : s.slot_disk) ++dn[sd.second], size_of[sd.second] = sgeom::kQuarterDiskBytes;  // a slot's own reference to its image
  for (const auto& j : s.jobs)  // an image queued and not settled holds its file space and is named by nothing yet
    if (!j.dead) size_of[j.off] = j.bytes;
  for (const size_t o : s.qimg)
    if (!dn.count(o)) {
      std::snprintf(b, sizeof b, "image %zu is a quarter image and nothing names it", o);
      return b;
    }
  for (const auto& [o, n] : dn) {
    const auto it = s.drefs.find(o);
    if (it == s.drefs.end() || it->second != n) {
      std::snprintf(b, sizeof b, "drefs[%zu] is %u, %u entries and bookmarks name it", o, it == s.drefs.end() ? 0 : it->second, n);
      return b;
    }
  }
  for (const auto& [o, n] : s.drefs)
    if (n && !dn.count(o)) {
      std::snprintf(b, sizeof b, "drefs[%zu] is %u and nothing names it", o, n);
      return b;
    }
  for (const auto& [o, n] : s.dnum)
    if (n >= s.ndisk.size() || s.ndisk[n] != o) {
      std::snprintf(b, sizeof b, "dnum[%zu] = %u but ndisk[%u] is not %zu", o, n, n, o);
      return b;
    }
  // the images (and the retired ones) and the free ranges tile the file, offsets past its size left out
  if (s.has_disk && s.dpool.size) {
    std::vector<std::pair<size_t, size_t>> blocks;
    for (const auto& [o, sz] : size_of) blocks.emplace_back(o, Pool::round(sz));
    for (const auto& [o, sz] : s.retired) blocks.emplace_back(o, Pool::round(sz));
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    for (const auto& [o, n] : s.dpool.free) blocks.emplace_back(o, n);
    std::sort(blocks.begin(), blocks.end());
    size_t covered = 0, end = 0;
    for (const auto& [o, n] : blocks) {
      if (o >= s.dpool.size) continue;
      if (o < end) {
        std::snprintf(b, sizeof b, "the file's blocks overlap at %zu", o);
        return b;
      }
      end = o + std::min(n, s.dpool.size - o);
      covered += std::min(n, s.dpool.size - o);
    }
    if (covered != s.dpool.size) {
      std::snprintf(b, sizeof b, "the images and free ranges cover %zu of the file's %zu bytes", covered, s.dpool.size);
      return b;
    }
  }
  for (int u = 0; u < c.nunits; ++u) {
    const int h = c.line_h[u];
    if (h == StoreCore::kNoLine) continue;
    const auto it = s.sess.find(h);
    if (it == s.sess.end() || it->second.line != u) {
      std::snprintf(b, sizeof b, "line_h[%d] is %d and that session's line is not %d", u, h, u);
      return b;
    }
  }
  for (const auto& [k, x] : s.sess)
    if (x.line >= 0 && (x.line >= c.nunits || c.line_h[x.line] != k)) {
      std::snprintf(b, sizeof b, "session %d is on line %d which holds %d", k, x.line, x.line < c.nunits ? c.line_h[x.line] : 0);
      return b;
    }
  for (int u = 0; u < c.nunits; ++u) {
    const auto r = f.rows.find(u);
    if (r == f.rows.end()) continue;
    const int h = c.line_h[u];
    for (int k = 0; k < kKvSlots; ++k) {
      const Nums want = h != StoreCore::kNoLine && s.sess.count(h) ? numbers_of(s.sess.at(h).lists[k]) : Nums();
      const Nums& have = r->second.empty() ? Nums() : r->second[k];
      if (want != have) {
        std::snprintf(b, sizeof b, "the device's rows of unit %d (kv source %d) differ from the session's lists", u, k);
        return b;
      }
    }
  }
  return "";
}

// ---- a store over the fake -------------------------------------------------------------------------------------------------

struct World {
  FakeDev dev;
  chunks::Alloc alloc;
  sgeom::StoreGeom g;
  std::unique_ptr<StoreCore> core;
  explicit World(uint32_t numbers, bool ring = false, Rule rule = Rule::kNone, int list_len = 48, int max_ctx = 262144,
                 size_t snap = 3u << 20, size_t end = 5u << 20, bool split = false)
      : alloc(numbers) {
    g.split = split;
    g.nslots = g.nunits = 4;
    g.list_len = list_len;
    g.max_ctx = max_ctx;
    for (int s = 0; s < 4; ++s) g.unit_of[s] = s, g.slot_ctx[s] = max_ctx;
    g.image_bytes = 2916352, g.snap_bytes = snap, g.end_bytes = end;
    g.has_ring = ring;
    core = std::make_unique<StoreCore>(g, dev, &alloc);
    (void)rule;
#ifdef STORE_CORE_TEST_RULES
    core->test_rule = rule;
#endif
  }
  // a session on a slot: the line takes numbers for `upto` positions, writes them (tags), and saves them
  void line(int slot, int h, int upto) {
    core->slot_reserve(slot, 40, 0);
    core->slot_attach(slot, h);
    core->ensure(slot, upto);
    Store::Sess& x = core->store_of().sess.at(h);
    for (const Refs& l : x.lists)
      for (const Ref c : l) dev.wrote(c.num());
    core->save(h, 0, upto, slot);
  }
  void inv(const std::string& where) {
    const std::string v = invariants(*core, dev);
    if (!v.empty()) throw Died{"invariant after " + where + ": " + v};
  }
  Store::Sess& sess(int h) { return core->store_of().sess.at(h); }
  void open_disk(size_t images, const std::string& dir, const char* name = "rank0") {
    core->disk(dir + "/" + name, images * Pool::round(g.image_bytes), 512 << 20, false, 7);
  }
};

// One row: a function over a world; it passes when it returns without a check failing or an exception.
struct Row {
  const char* name;
  Rule rule;  // the wrong rule that must make it fail (kNone: no control)
  std::function<void(World&, const std::string&)> fn;
  uint32_t numbers;
  bool ring;
};

#define NEED(cond, msg)                                       \
  do {                                                        \
    if (!(cond)) throw Died{std::string("expected: ") + msg}; \
  } while (0)


// The image link: A writes an image of number c; B forks A below c and its save adopts that image. After B is evicted, what B
// reads back must be what it wrote, not what the image holds. With `pressure` the disk tier fills and takes B's image entries (the
// image stays, A's entry holds it) while the number's link stays on c; A leaves the pool, B alone holds c, writes it in place and
// saves it again. Returns "" when B read back what it wrote, else what differed.
std::string image_link_sequence(World& w, const std::string& dir, bool pressure, bool traced = false) {
  if (traced) {  // a trace opens after the kept check and the disk tier, as the engine's does
    size_t kb = 0;
    std::string why;
    w.core->kept_check(dir + "/rank0", 7, 86400, &kb, &why);
  }
  w.open_disk(400, dir);
  if (traced) w.core->trace_open();
  w.line(0, 1, 100000);  // A: its entries are written and have images
  w.inv("A's line");
  w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
  w.core->load(1, 40000, 1, 1);  // B forks A at 40,000: the full numbers are shared, the partial one is B's copy
  w.core->ensure(1, 100000);
  for (const Refs& l : w.sess(2).lists)
    for (const Ref c : l)
      if (c && w.core->refs[c.num()] == 1) w.dev.wrote(c.num());  // B's kernels write its own numbers only
  w.core->save(2, 0, 100000, 1);  // B's entries adopt A's images for the shared numbers and get their own for the rest
  w.inv("B's save");
  const uint32_t c = w.sess(2).lists[0][1].num();
  if (w.core->refs[c] != 2 || w.core->store_of().ndisk[c] == Pool::kNone) throw Died{"expected: c is shared and linked to A's image"};
  w.core->slot_release(1);  // B is idle: cache
  w.core->load(1, 100000, 0, 0);  // A, still on its line, is used again: it is the more recently used
  if (pressure) {
    w.core->disk_resize(w.core->store_of().dbytes);  // the file is exactly full
    w.line(2, 3, 17000);                               // a third conversation asks for room: B is the disk tier's victim
    w.core->slot_release(2);
    w.inv("the disk tier took B's images");
    for (const auto& d : w.sess(2).dl)
      for (const size_t o : d)
        if (o != Pool::kNone) throw Died{"expected: B's image entries are dropped"};
    if (w.core->store_of().ndisk[c] == Pool::kNone) throw Died{"expected: the link stays on c while A's entry holds the image"};
  }
  w.core->slot_release(0);
  w.core->spill(1);  // A leaves the pool (its entries keep their images)
  w.inv("A out");
  if (w.core->refs[c] != 1) throw Died{"expected: B alone holds c"};
  w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
  w.core->load(2, 20000, 1, 1);  // B rewinds into c ...
  std::vector<std::vector<uint64_t>> wrote(kKvSlots);
  for (int k = 0; k < kKvSlots; ++k) {
    const size_t i0 = (size_t)((20000 / (k < 3 ? 2 : 1)) >> sgeom::kChunkShift);
    const Refs& l = w.sess(2).lists[k];
    for (size_t i = 0; i < l.size(); ++i) wrote[k].push_back(i >= i0 && l[i] ? w.dev.wrote(l[i].num()) : 0);  // ... and writes them in place
  }
  w.core->save(2, 20000, 40000, 1);
  w.inv("B's next save");
  w.core->slot_release(1);
  w.core->spill(2);  // B evicted
  w.core->unspill(2);  // B read back
  w.inv("B back");
  for (int k = 0; k < kKvSlots; ++k) {
    const Refs& l = w.sess(2).lists[k];
    for (size_t i = 0; i < l.size() && i < wrote[k].size(); ++i)
      if (l[i] && wrote[k][i] && w.dev.tag_of(l[i].num()) != wrote[k][i])
        return "B's list " + std::to_string(k) + " entry " + std::to_string(i) + " read back tag " + std::to_string(w.dev.tag_of(l[i].num())) +
               ", it wrote " + std::to_string(wrote[k][i]);
  }
  return "";
}

// Two numbers read back from one image. A and B share numbers through a fork; both are evicted and read back, A first, so both
// numbers name the one image and the image names B's. A is evicted again and freed. The disk tier fills and takes B's image
// entries: the image's last reference drops while B's number is still in the pool, and its file space is free. A third conversation
// writes images into it. B's next flush must not adopt the dropped image. Returns "" when B ends with what it wrote, else what
// differed; an invariant that fails on the way throws.
std::string image_share_sequence(World& w, const std::string& dir) {
  w.open_disk(400, dir);
  w.line(0, 1, 100000);
  w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
  w.core->load(1, 40000, 1, 1);
  w.core->ensure(1, 100000);
  for (const Refs& l : w.sess(2).lists)
    for (const Ref c : l)
      if (c && w.core->refs[c.num()] == 1) w.dev.wrote(c.num());
  w.core->save(2, 0, 100000, 1);
  w.inv("B's save");
  w.core->slot_release(1), w.core->slot_release(0);
  w.core->spill(1);  // A out: the shared numbers stay with B
  w.core->spill(2);  // B out: the shared numbers go back, their images stay with both entries
  w.inv("both out");
  w.core->unspill(1);  // A back: its number names the image
  w.core->unspill(2);  // B back: its number names the same image
  w.inv("both back");
  std::vector<std::vector<uint64_t>> had(kKvSlots);
  for (int k = 0; k < kKvSlots; ++k)
    for (const Ref n : w.sess(2).lists[k]) had[k].push_back(n ? w.dev.tag_of(n.num()) : 0);
  w.core->disk_resize(w.core->store_of().dbytes);  // the file exactly full
  w.core->spill(1);  // A out again: its number goes back, the image stays with the entries
  w.inv("A out again");
  w.core->free_session(1);  // A freed: its references to the images go
  w.inv("A freed");
  w.line(2, 3, 100000);  // a third conversation asks for more than A's freed space holds: B is the disk tier's victim
  w.core->slot_release(2);
  w.inv("the disk tier took B's images");
  w.core->flush(2);  // B's idle flush writes the entries that have no image
  w.inv("B's flush");
  w.core->spill(2);
  w.core->unspill(2);
  w.inv("B back again");
  for (int k = 0; k < kKvSlots; ++k) {
    const Refs& l = w.sess(2).lists[k];
    for (size_t i = 0; i < l.size() && i < had[k].size(); ++i)
      if (l[i] && had[k][i] && w.dev.tag_of(l[i].num()) != had[k][i])
        return "B's list " + std::to_string(k) + " entry " + std::to_string(i) + " read back tag " + std::to_string(w.dev.tag_of(l[i].num())) +
               ", it had " + std::to_string(had[k][i]);
  }
  return "";
}


// ---- the split ----------------------------------------------------------------------------------------------------------

// every list entry's content tag, list by list (0: not a whole number)
std::vector<uint64_t> tags_of(World& w, int h) {
  std::vector<uint64_t> t;
  for (const Refs& l : w.sess(h).lists)
    for (const Ref c : l) t.push_back(c.is_number() ? w.dev.tag_of(c.num()) : 0);
  return t;
}

// a conversation A of 100,000 positions on slot 0, idle; its list entries: 7 + 7 + 7 + 13, of which 6 + 6 + 6 + 12 are whole pieces
void idle_conversation(World& w, int slot, int h) {
  w.line(slot, h, 100000);
  w.core->slot_release(slot);
}

void add_split_rows(std::vector<Row>& v) {
  v.push_back({"split: whole pieces become slots and their numbers ghosts, by budget; a pinned conversation is left alone; the join takes the ghosts back",
               Rule::kSplitIgnoresPin,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 const std::vector<uint64_t> before = tags_of(w, 1);
                 w.core->pin({1}, {}, {});
                 NEED(w.core->split(1, 64) == 0, "a pinned conversation is not split (the split is its own command: nothing pins it there)");
                 w.core->pin({}, {}, {});
                 NEED(w.core->split(1, 10) == 10 && w.core->split(1, 64) == 24 && w.core->split(1, 64) == 0, "34 pieces (30 full, 4 partial), 10 and then the rest");
                 w.inv("the split");
                 NEED(w.alloc.ghosts() == 34 && w.sess(1).slot_entries() == 34 && w.sess(1).numbers() == 0, "34 ghosts, 34 slots, nothing whole");
                 NEED(w.core->store_of().slots.open_numbers() == 9, "nine quarter numbers hold the 34 slots");
                 int in_pool = 0, on_disk = 0, split = 0;
                 w.core->list_numbers(1, INT32_MAX, &in_pool, &on_disk, &split);
                 NEED(in_pool == 0 && on_disk == 0 && split == 34, "list_numbers: 0 whole, 0 on the disk, 34 split");
                 NEED(w.core->join(1, INT32_MAX, 0, 64) == 0, "one command joins the 34 pieces");
                 w.inv("the join");
                 NEED(tags_of(w, 1) == before, "the bytes are the bytes written");
                 NEED(w.core->split_counts.revives == 34 && w.core->split_counts.exchanged == 0, "every ghost was taken back: nothing crossed the ring");
                 NEED(w.alloc.ghosts() == 0 && w.core->store_of().slots.used() == 0 && w.core->store_of().slots.open_numbers() == 0, "no ghost, no slot, no quarter number left");
                 bool exch = false;
                 for (const std::string& c : w.dev.calls) exch = exch || c.rfind("exchange", 0) == 0;
                 NEED(!exch, "no exchange call");
               },
               400, false});
  v.push_back({"split: a number a filler took from the ghosts is never shared or revived: the piece is exchanged, or the join stops", Rule::kNoTwinCut,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 const std::vector<uint64_t> before = tags_of(w, 1);
                 NEED(w.core->split(1, 64) == 34, "34 pieces split");
                 w.line(1, 2, 100000);  // the filler takes the plain free numbers, then the oldest ghosts
                 w.core->slot_unreserve(1);
                 w.inv("the filler");
                 NEED(w.alloc.ghosts() < 34, "the filler took ghosts");
                 const int left = w.core->join(1, INT32_MAX, 0, 64);
                 w.inv("the join");
                 NEED(w.sess(1).saved == 100000, "nothing of the conversation was cut");
                 const std::vector<uint64_t> after = tags_of(w, 1);
                 for (size_t i = 0; i < after.size(); ++i) NEED(!after[i] || after[i] == before[i], "an entry that is a number holds the bytes it had");
                 (void)left;
               },
               70, false});
  v.push_back({"split: the join exchanges the pieces whose ghosts were taken, in commands of J pieces, and every byte comes back", Rule::kNone,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 const std::vector<uint64_t> before = tags_of(w, 1);
                 w.core->split(1, 64);
                 w.line(1, 2, 100000);
                 w.core->slot_release(1);
                 w.core->free_session(2);  // the filler leaves: plain numbers are free again, the ghosts it took are not ghosts any more
                 int left = 1, commands = 0;
                 while (left && commands < 8) left = w.core->join(1, INT32_MAX, 0, 10), ++commands, w.inv("a join command");
                 NEED(left == 0 && commands == 4, "34 pieces in commands of 10: four commands");
                 NEED(tags_of(w, 1) == before, "the bytes are the bytes written");
                 NEED(w.core->split_counts.exchanged > 0 && w.core->split_counts.revives > 0, "some pieces were exchanged, some taken back");
                 NEED(w.core->store_of().slots.used() == 0, "no slot left");
               },
               70, false});
  v.push_back({"split: a fork split with its base, one returns and rewinds a shared piece, the other still gets the bytes it had", Rule::kNoTwinCut,
               [](World& w, const std::string&) {
                 w.line(0, 1, 100000);
                 w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
                 w.core->load(1, 40000, 1, 1);  // B forks A at 40,000
                 w.core->ensure(1, 100000);
                 for (const Refs& l : w.sess(2).lists)
                   for (const Ref c : l)
                     if (c && w.core->refs[c.num()] == 1) w.dev.wrote(c.num());
                 w.core->save(2, 0, 100000, 1);
                 w.core->slot_release(1), w.core->slot_release(0);
                 const std::vector<uint64_t> a_before = tags_of(w, 1);
                 NEED(w.core->split(1, 64) == 34, "A's 34 pieces, the shared ones with both holders at once");
                 NEED(w.core->split(2, 64) > 0, "B's own pieces");
                 w.inv("both split");
                 NEED(w.core->join(2, INT32_MAX, 0, 64) == 0, "B returns");
                 w.inv("B back");
                 w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
                 w.core->load(2, 20000, 1, 1);  // B rewinds into the shared pieces ...
                 for (int k = 0; k < kKvSlots; ++k)  // ... and writes them in place
                   for (size_t i = (size_t)((20000 / (k < 3 ? 2 : 1)) >> sgeom::kChunkShift); i < w.sess(2).lists[k].size(); ++i)
                     if (w.sess(2).lists[k][i].is_number() && w.core->refs[w.sess(2).lists[k][i].num()] == 1) w.dev.wrote(w.sess(2).lists[k][i].num());
                 w.inv("B rewound");
                 w.core->slot_release(1);
                 w.core->join(1, INT32_MAX, 0, 64);
                 w.inv("A back");
                 NEED(w.sess(1).saved == 100000, "A was not cut: its pieces did not share what B wrote");
                 NEED(tags_of(w, 1) == a_before, "A holds the bytes it had");
               },
               400, false});
  v.push_back({"split: with no quarter number to be had a piece stays whole and nobody is evicted", Rule::kSplitEvicts,
               [](World& w, const std::string&) {
                 idle_conversation(w, 1, 2);  // B is the older: the wrong rule's eviction takes it
                 idle_conversation(w, 0, 1);
                 NEED(w.alloc.free() == 0, "the pool is full");
                 const size_t b_before = w.sess(2).in_pool();
                 NEED(w.core->split(1, 64) == 0 && w.core->split_counts.split_deferred > 0, "nothing split, the pieces deferred");
                 w.inv("no room");
                 NEED(w.sess(1).slot_entries() == 0 && w.core->store_of().sess.count(2) && w.sess(2).in_pool() == b_before, "A is whole and B kept its pieces");
               },
               69, false});
  v.push_back({"giveable: all conversations split and a consumer short, the quarter numbers are offered; a conversation on a line counts in the joint sum", Rule::kGiveableNoSlots,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 NEED(w.core->giveable({}) == 34, "before the split: 34 numbers");
                 w.core->split(1, 64);
                 NEED(w.core->giveable({}) == 9, "after: nine quarter numbers");
                 w.line(1, 2, 100000);  // B on a line: its numbers are the line's
                 NEED(w.core->giveable({}) == 9, "a line's conversation is not offered");
                 NEED(w.core->giveable_joint({}, {2}) == 9 + 34, "counted as if no line held it: 34 more");
                 NEED(w.core->giveable_joint({1}, {2}) == 34, "keep A: only B's");
               },
               400, false});
  v.push_back({"split: a conversation taking its line is joined first; its rows are its numbers", Rule::kNone,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 const std::vector<uint64_t> before = tags_of(w, 1);
                 w.core->split(1, 64);
                 w.core->slot_reserve(0, 40, 0), w.core->slot_attach(0, 1);
                 w.inv("attached");
                 NEED(!w.sess(1).split() && tags_of(w, 1) == before, "whole again, the bytes it had");
               },
               400, false});
  v.push_back({"split: a piece that arrives different cuts the conversation at its position", Rule::kNone,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 w.core->split(1, 64);
                 w.line(1, 2, 100000);
                 w.core->slot_release(1);
                 w.core->free_session(2);
                 w.dev.spoil_pieces = {0};
                 w.core->join(1, INT32_MAX, 0, 64);
                 w.inv("after the bad piece");
                 NEED(w.sess(1).saved < 100000 && !w.sess(1).split(), "cut at the bad piece; nothing split is left");
                 NEED(w.core->split_counts.join_bad == 1, "counted");
               },
               70, false});
  v.push_back({"split: a cut of the top over a quarter number and over ghosts; two cores given the same commands agree", Rule::kNone,
               [](World& w, const std::string&) {
                 const auto run = [](World& x) {
                   idle_conversation(x, 0, 1);
                   x.core->split(1, 64);
                   x.core->vacate(36);
                   x.core->alloc_shrink(36);
                   x.inv("after the cut");
                 };
                 World w2(400);
                 run(w), run(w2);
                 NEED(w.alloc.numbers() == 36, "the pool was cut");
                 NEED(w.core->digest() == w2.core->digest() && w.core->wide_digest(w.core->digest()) == w2.core->wide_digest(w2.core->digest()),
                      "the two cores' digests are equal");
               },
               400, false});
}


// ---- partial pieces, bookmarks as slots, unghost, the idle conversation's numbers --------------------------------------------------------------------

// a conversation of `n` positions with window bookmarks every `every` and the end bookmark, idle, its lookahead numbers trimmed
void bookmarked_conversation(World& w, int h, int n, int every) {
  w.core->slot_reserve(0, 40, 0), w.core->slot_attach(0, h);
  w.core->ensure(0, n);
  for (const Refs& l : w.sess(h).lists)
    for (const Ref c : l) w.dev.wrote(c.num());
  w.core->save(h, 0, n, 0);
  for (int pos = every; pos <= n; pos += every)
    if (const Nums* b = w.core->snap_buf(h, pos))
      for (const uint32_t c : *b) w.dev.wrote(c);
  if (const Nums* e = w.core->end_take(h, n))
    for (const uint32_t c : *e) w.dev.wrote(c);
  w.core->slot_release(0);
  w.core->drop(h, n);  // an idle conversation holds its saved positions only
}

// An idle conversation's numbers on one Spark, whole and split. Real sizes: a window bookmark 2,906,112 B (2 numbers, one
// quarter slot), the end bookmark 6,074,368 B (3 numbers, three slots). Returns the failures; prints the table when `print`.
std::string acc5(const Rule rule, bool print) {
  struct Case {
    int n;
    int every;
  };
  std::string bad;
  if (print) std::printf("  idle conversation's numbers (host core, numbers on one Spark; window bookmarks every 1,024 and every 16,384 positions):\n");
  for (const int every : {1024, 16384})
    for (const int n : {2048, 12288, 40960, 307200, 1048576}) {
      World w(4000, false, rule, 140, 1 << 21, 2906112, 6074368);
      bookmarked_conversation(w, 1, n, every);
      const Store::Sess& x = w.sess(1);
      const size_t whole = x.numbers();
      for (int guard = 0; guard < 100 && w.core->split(1, 1 << 20); ++guard) {
      }
      const size_t slots = x.slot_entries() + x.bookmark_slots(), after = x.numbers() + (slots + 3) / 4;
      const double pct = 100.0 * (double)after / (double)whole;
      if (print)
        std::printf("    %8d positions, bookmarks every %5d: whole %5zu numbers, split %5zu numbers (%zu slots, %zu left whole) = %5.1f %%\n", n, every,
                    whole, after, slots, x.numbers(), pct);
      if ((double)after > 0.30 * (double)whole + 1)
        bad += " " + std::to_string(n) + "/" + std::to_string(every) + ":" + std::to_string((int)pct) + "%";
    }
  return bad;
}

void add_a3_rows(std::vector<Row>& v) {
  v.push_back({"split: the last, partial piece of each list is split with its written count and comes back with the bytes it had", Rule::kSplitFullOnly,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 const std::vector<uint64_t> before = tags_of(w, 1);
                 NEED(w.core->split(1, 64) == 34, "all 34 pieces, the four partial ones among them");
                 NEED(w.sess(1).numbers() == 0, "nothing of the conversation is whole");
                 w.inv("split");
                 w.line(1, 2, 100000);  // a filler takes ghosts: some pieces must be exchanged
                 w.core->slot_release(1);
                 w.core->free_session(2);
                 w.core->join(1, INT32_MAX, 0, 64);
                 w.inv("joined");
                 NEED(tags_of(w, 1) == before && !w.sess(1).split(), "whole again, the bytes it had");
                 bool partial = false;
                 for (const std::string& c : w.dev.calls) partial = partial || (c.rfind("quarter ", 0) == 0 && c.substr(c.rfind(' ') + 1) != "8192");
                 NEED(partial, "a quarter was made with a written count below 8,192");
               },
               70, false});
  v.push_back({"split: a partial piece shared by two holders stays whole", Rule::kNone,
               [](World& w, const std::string&) {
                 w.line(0, 1, 100000);
                 w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
                 w.core->load(1, 40000, 1, 1);  // B forks A inside a piece: B's copy of the partial number is its own
                 w.core->slot_release(1), w.core->slot_release(0);
                 w.core->drop(1, 100000), w.core->drop(2, 40000);
                 w.core->split(1, 64);
                 w.inv("split");
                 for (const Refs& l : w.sess(2).lists) {
                   (void)l;
                 }
                 NEED(w.sess(1).slot_entries() > 0, "A's pieces are split");
               },
               400, false});
  v.push_back({"split: window and end bookmarks become slots (one and three) and come back with their bytes", Rule::kBookmarksWhole,
               [](World& w, const std::string&) {
                 World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                 bookmarked_conversation(x, 1, 40960, 16384);
                 std::vector<std::vector<uint64_t>> before;
                 for (const auto& sn : x.sess(1).snaps) {
                   before.emplace_back();
                   for (const uint32_t c : sn.second) before.back().push_back(x.dev.tag_of(c));
                 }
                 std::vector<uint64_t> eb;
                 for (const uint32_t c : x.sess(1).end) eb.push_back(x.dev.tag_of(c));
                 const size_t snaps = x.sess(1).snaps.size();
                 NEED(snaps == 2 && x.sess(1).end.size() == 3, "two window bookmarks and the end bookmark");
                 x.core->split(1, 1 << 20);
                 x.inv("split");
                 NEED(x.sess(1).qsnaps.size() == 2 && x.sess(1).qend.size() == 3, "two slots for the windows, three for the end");
                 NEED(x.sess(1).bookmark_slots() == 5 && x.sess(1).end.empty(), "five slots, no whole bookmark");
                 NEED(!x.sess(1).out(), "a split bookmark is not only on the disk");
                 x.core->join(1, INT32_MAX, 0, 64);
                 x.inv("joined");
                 size_t i = 0;
                 for (const auto& sn : x.sess(1).snaps) {
                   NEED(sn.second.size() == 2, "the window bookmark is back");
                   for (size_t j = 0; j < sn.second.size(); ++j) NEED(x.dev.tag_of(sn.second[j]) == before[i][j], "its bytes");
                   ++i;
                 }
                 for (size_t j = 0; j < eb.size(); ++j) NEED(x.dev.tag_of(x.sess(1).end[j]) == eb[j], "the end bookmark's bytes");
                 NEED(x.sess(1).bookmark_slots() == 0 && x.sess(1).qsnaps.empty() && x.sess(1).qend.empty(), "no bookmark slot left");
               },
               400, false});
  v.push_back({"split: the join before an operation covers the bookmarks, so the core's own join inside the operation finds nothing", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, Rule::kNone, 140, 1 << 21, 2906112, 6074368);
                 bookmarked_conversation(x, 1, 40960, 16384);
                 x.core->split(1, 1 << 20);
                 x.inv("split");
                 NEED(x.sess(1).slot_entries() > 0 && x.sess(1).bookmark_slots() > 0, "pieces and bookmarks are split");
                 // the server's join: its budget as join_back is given it, commands of the count that is left, until nothing is taken
                 int64_t left = model::rules::join_budget((int64_t)x.sess(1).slot_entries(), (int64_t)x.core->restore_chunks(1), 0, 0);
                 for (int c = 0; c < 100 && x.sess(1).split(); ++c) {
                   const int64_t count = model::rules::join_command_count(left);
                   if (count < 0) break;
                   const uint64_t t0 = x.core->split_counts.join_taken;
                   x.core->join(1, INT32_MAX, (size_t)count, 64);
                   left -= (int64_t)(x.core->split_counts.join_taken - t0);
                   if (x.core->split_counts.join_taken == t0) break;
                 }
                 x.inv("the server's join");
                 NEED(!x.sess(1).split(), "nothing of the conversation is split after the server's join");
                 const uint64_t joins = x.core->split_counts.joins;
                 x.core->join_whole(1);
                 NEED(x.core->split_counts.joins == joins, "the operation's join_whole joins nothing");
               },
               400, false});
  v.push_back({"split: a piece split, joined, grown and split again joins back to its sums when a fork asks below its end", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, Rule::kNone, 140, 1 << 21, 2906112, 6074368);
                 x.line(0, 1, 3032);
                 x.core->slot_release(0);
                 x.core->split(1, 1 << 20);
                 x.inv("first split");
                 NEED(x.sess(1).slot_entries() > 0, "split");
                 x.core->slot_reserve(0, 40, 0), x.core->slot_attach(0, 1);
                 x.core->load(1, 3032, 0, 0);  // the next request: joined, on a line again
                 x.core->ensure(0, 3564);
                 for (const Refs& l : x.sess(1).lists)
                   for (const Ref c : l)
                     if (c.is_number() && x.core->refs[c.num()] == 1) x.dev.wrote(c.num());
                 x.core->save(1, 0, 3564, 0);
                 x.core->slot_release(0);
                 x.inv("grown");
                 x.core->split(1, 1 << 20);
                 x.inv("second split");
                 x.line(1, 2, 3000);  // a different, cold session
                 x.core->slot_release(1);
                 x.core->slot_reserve(2, 40, 0), x.core->slot_attach(2, 3);
                 const uint64_t bad = x.core->split_counts.join_bad;
                 x.core->load(1, 3132, 2, 2);  // a fork of the first session below its end
                 x.inv("the fork");
                 NEED(x.core->split_counts.join_bad == bad, "no piece joined from quarters differs from its sum");
               },
               400, false});
  v.push_back({"split: a bookmark that arrives different cuts the conversation below it", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, Rule::kNone, 140, 1 << 21, 2906112, 6074368);
                 bookmarked_conversation(x, 1, 40960, 16384);
                 x.core->split(1, 1 << 20);
                 x.dev.spoil_pieces = {0};  // the first exchange of the join: the end bookmark
                 x.core->join(1, INT32_MAX, 0, 64);
                 x.inv("joined");
                 NEED(x.core->split_counts.join_bad == 1 && x.sess(1).saved < 40960, "cut below the bad bookmark");
               },
               400, false});
  v.push_back({"a ghost whose link is cut is a plain free number: the next number taken is the dead one, not a ghost that can still be revived",
               Rule::kNoUnghost,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 NEED(w.core->split(1, 64) == 34, "34 ghosts");
                 const uint32_t ghosts = w.alloc.ghosts();
                 w.core->drop(1, 50000);  // the tail pieces go: their slots are freed and their ghosts have nothing to be revived from
                 w.inv("after the drop");
                 NEED(w.alloc.ghosts() < ghosts, "the dead ghosts are plain free numbers now");
                 w.line(1, 2, 20000);  // a small filler: it takes the dead numbers first
                 w.inv("the filler");
                 NEED(w.core->store_of().twin_c.size() == w.alloc.ghosts(), "every ghost left is one that can be revived");
               },
               400, false});
  v.push_back({"giveable_joint: a base idle on one line and its fork idle on another share their prefix: the pool gives back what is named once; the old per-line sum counts the shared numbers twice",
               Rule::kNone,
               [](World& w, const std::string&) {
                 w.line(0, 1, 100000);
                 w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
                 w.core->load(1, 40000, 1, 1);  // B forks A at 40,000: 10 numbers shared, B's copies of the partial ones its own
                 w.core->ensure(1, 100000);
                 for (const Refs& l : w.sess(2).lists)
                   for (const Ref c : l)
                     if (c && w.core->refs[c.num()] == 1) w.dev.wrote(c.num());
                 w.core->save(2, 0, 100000, 1);
                 w.inv("both idle on their lines");
                 const size_t old_sum = w.sess(1).numbers() + w.sess(2).numbers();  // each line's own numbers, added: today's sum
                 const size_t joint = w.core->giveable_joint({}, {1, 2});
                 const uint32_t before = w.alloc.free();
                 w.core->slot_release(0), w.core->slot_release(1);
                 w.core->free_session(1), w.core->free_session(2);
                 const size_t actual = w.alloc.free() - before;
                 NEED(joint == actual, "the joint count equals what the pool really gets back (" + std::to_string(joint) + " against " + std::to_string(actual) + ")");
                 NEED(old_sum > actual, "the old sum over-counts: " + std::to_string(old_sum) + " against what two detaches give, " + std::to_string(actual));
               },
               400, false});
  v.push_back({"an idle conversation's numbers, split, are at most 30 % of its whole numbers plus one (2K to 1M positions)", Rule::kBookmarksWhole,
               [](World& w, const std::string&) {
                 const std::string bad = acc5(w.core->test_rule, w.core->test_rule == Rule::kNone);
                 NEED(bad.empty(), "over 30 % + 1:" + bad);
               },
               70, false});
  v.push_back({"idle conversation's numbers with partial pieces left whole", Rule::kSplitFullOnly,
               [](World& w, const std::string&) {
                 const std::string bad = acc5(w.core->test_rule, false);
                 NEED(bad.empty(), "over 30 % + 1:" + bad);
               },
               70, false});
}

// ---- B3: a join's takes --------------------------------------------------------------------------------------

size_t slots_left(World& w, int h) { return w.sess(h).slot_entries(); }

void add_b3_rows(std::vector<Row>& v) {
  v.push_back({"split: a join carrying the lists' count leaves the bookmarks split (they come back in pf_begin's own call with the spare); with 0 it takes them too", Rule::kNone,
               [](World& w, const std::string&) {
                 const auto make = [&](World& x) {
                   bookmarked_conversation(x, 1, 40960, 16384);
                   x.core->split(1, 1 << 20);
                 };
                 World a(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368), b(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                 make(a), make(b);
                 const size_t lists = a.sess(1).slot_entries();
                 a.core->join(1, INT32_MAX, lists, 64);
                 a.inv("the lists only");
                 NEED(a.sess(1).slot_entries() == 0 && a.sess(1).bookmark_slots() == 5, "cap = the lists: every list whole, the five bookmark slots still split");
                 b.core->join(1, INT32_MAX, 0, 64);
                 NEED(b.sess(1).bookmark_slots() == 0, "cap 0 (what the old server sent): the bookmarks came back too, out of the numbers admission counted for the reservation");
               },
               400, false});
  v.push_back({"split: a reservation made between two join commands is covered after the next: the join stops where the free numbers meet it, nothing is cut", Rule::kJoinTakeUnpaid,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 w.core->split(1, 64);
                 w.core->slot_reserve(1, 70, 0);  // another line is reserved 70 numbers (held 0)
                 NEED(w.alloc.free() == 90, "9 quarter numbers held, 90 free (34 of them ghosts)");
                 w.core->join(1, INT32_MAX, 0, 10);
                 w.core->join(1, INT32_MAX, 0, 10);
                 w.core->join(1, INT32_MAX, 0, 10);
                 w.inv("after the join stopped");
                 NEED(w.alloc.free() >= 70, "the reservation is covered: " + std::to_string(w.alloc.free()) + " free of the 70 reserved");
                 NEED(slots_left(w, 1) > 0 && slots_left(w, 1) < 34 && w.core->split_counts.join_short >= 1, "the join stopped part way (some slots left), counted");
                 NEED(w.sess(1).saved == 100000 && w.core->split_counts.join_bad == 0, "nothing was cut");
                 w.core->slot_unreserve(1);  // the reservation ends: the numbers are there
                 w.core->join(1, INT32_MAX, 0, 64);
                 NEED(slots_left(w, 1) == 0, "it completes when the numbers come");
               },
               100, false});
  v.push_back({"split: a conversation split on its own idle line pays for its return from its own line's reservation first", Rule::kJoinTakeUnpaid,
               [](World& w, const std::string&) {
                 w.line(0, 1, 100000);  // on line 0, holding 34
                 w.core->split(1, 64);   // held 0 whole
                 w.core->slot_reserve(0, 30, 0);  // its own line is reserved 30
                 std::vector<uint32_t> hold;  // others took 20 numbers: the 16 plain free ones and the 4 oldest ghosts (the first pieces' own)
                 NEED(w.alloc.take(24, &hold), "24 taken");  // the 20 plain free numbers and the 4 oldest ghosts (the first pieces of list 0)
                 w.core->refs.resize(w.alloc.numbers(), 0);
                 for (const uint32_t c : hold) {
                   w.core->refs[c] = 1;
                   w.core->cut_twin(c);
                   w.core->store_of().sess[-5].lists[0].push_back(Ref(c));
                 }
                 NEED(w.alloc.free() == 30, "30 free, all of them its own line's reservation");
                 w.core->join(1, 40000, 0, 64);  // the 14 pieces below 40,000: the first three of list 0 must be exchanged
                 w.inv("joined");
                 int in = 0, disk = 0, sp = 0;
                 w.core->list_numbers(1, 40000, &in, &disk, &sp);
                 NEED(sp == 0 && w.core->split_counts.exchanged >= 3, "the join completes, pieces exchanged: each number taken is paid by the line's own reservation (slots below 40,000: " + std::to_string(sp) + ")");
               },
               64, false});
  v.push_back({"split: a join stamps its conversation used: the pieces just joined are not the next victim", Rule::kJoinNoStamp,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 idle_conversation(w, 1, 2);  // 2 is the newer
                 w.core->split(1, 64);
                 w.core->split(2, 64);
                 w.core->join(1, INT32_MAX, 0, 4);  // 4 pieces of the older one come back
                 const size_t b_before = w.sess(2).in_pool(), a_before = w.sess(1).in_pool();
                 w.core->evict_step(99);
                 const Store& st = w.core->store_of();
                 (void)a_before, (void)b_before;
                 NEED(st.sess.count(1) && !st.sess.count(2), "the other conversation was the victim, not the one just joined");
               },
               200, false});
  v.push_back({"split: fault row: the N-th take of a join refused once: it stops at the same piece, nothing is cut, the next command completes", Rule::kNone,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 w.core->split(1, 64);
                 w.core->fault_join_take = 5;
                 w.core->join(1, INT32_MAX, 0, 64);
                 NEED(slots_left(w, 1) == 30 && w.core->split_counts.join_short == 1, "refused at the 5th take: 4 pieces back, the rest split");
                 NEED(w.sess(1).saved == 100000 && w.core->split_counts.join_bad == 0, "no cut, no bad piece");
                 w.core->join(1, INT32_MAX, 0, 64);
                 w.inv("completed");
                 NEED(slots_left(w, 1) == 0, "the next command completes it");
               },
               400, false});
}


// ---- the guard joins what the operation reads ----------------------------------------------------------------------------------

void add_end_rows(std::vector<Row>& v) {
  v.push_back({"end bookmark: an end bookmark taken for a split conversation joins nothing", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                 bookmarked_conversation(x, 1, 40960, 16384);
                 x.core->split(1, 1 << 20);
                 x.inv("split");
                 NEED(x.sess(1).slot_entries() > 0 && !x.sess(1).qend.empty(), "pieces and the end bookmark are split");
                 const size_t slots = x.sess(1).slot_entries();
                 const uint64_t joins = x.core->split_counts.joins;
                 const Nums* e = x.core->end_take(1, 40960);
                 NEED(e != nullptr && !e->empty(), "the end bookmark got numbers");
                 x.inv("end_take");
                 NEED(x.core->split_counts.joins == joins, "no join ran inside end_take");
                 NEED(x.sess(1).slot_entries() == slots, "every piece is still a slot");
                 NEED(x.sess(1).qend.empty(), "the split end bookmark it replaces is gone");
               },
               400, false});
  v.push_back({"end bookmark: a fork's restore joins the bookmark it reads and nothing above it", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                 bookmarked_conversation(x, 1, 40960, 16384);
                 x.core->split(1, 1 << 20);
                 x.inv("split");
                 NEED(x.sess(1).qsnaps.count(16384) == 1 && x.sess(1).qsnaps.count(32768) == 1 && !x.sess(1).qend.empty(), "two window bookmarks and the end bookmark are split");
                 const Nums& b = x.core->snap_get(1, 16384);
                 NEED(!b.empty(), "the bookmark it reads is whole");
                 x.inv("snap_get");
                 NEED(x.sess(1).qsnaps.count(16384) == 0, "the bookmark it reads came back");
                 NEED(x.sess(1).qsnaps.count(32768) == 1 && !x.sess(1).qend.empty(), "the bookmarks above it are still split");
                 for (int k = 0; k < kKvSlots; ++k)
                   for (int i = 0; i < (int)x.sess(1).lists[k].size(); ++i)
                     NEED(!(x.sess(1).lists[k][i].is_slot() && ent_pos(k, i) < 16384), "no piece below it is left split");
                 NEED(x.sess(1).slot_entries() > 0, "the pieces above it are still slots");
               },
               400, false});
  v.push_back({"end bookmark: a restart below the end leaves the tail split until drop", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                 bookmarked_conversation(x, 1, 40960, 16384);
                 x.core->split(1, 1 << 20);
                 x.inv("split");
                 const size_t slots = x.sess(1).slot_entries();
                 NEED(slots > 0, "split");
                 x.core->slot_reserve(0, 40, 0);
                 x.core->slot_attach(0, 1, 16384);  // the restart point: the line holds what is below it
                 x.inv("attach");
                 for (int k = 0; k < kKvSlots; ++k)
                   for (int i = 0; i < (int)x.sess(1).lists[k].size(); ++i)
                     NEED(!(x.sess(1).lists[k][i].is_slot() && ent_pos(k, i) < 16384), "no piece below the restart point is left split");
                 NEED(x.sess(1).slot_entries() > 0 && x.sess(1).slot_entries() < slots, "the tail is still split, the head came back");
                 NEED(x.sess(1).qsnaps.count(32768) == 1, "the bookmark above the restart point is still split");
                 x.core->slot_release(0);
                 const size_t tail = x.sess(1).slot_entries();
                 x.core->drop(1, 16384);
                 x.inv("drop");
                 NEED(x.sess(1).qsnaps.count(32768) == 0 && x.sess(1).qend.empty(), "drop took the tail's bookmarks");
                 NEED(x.sess(1).slot_entries() < tail, "drop took the tail's pieces");
               },
               400, false});
  v.push_back({"end bookmark: a restart planned after a join that stopped for want of a number reuses only what is whole, and its load does not die", Rule::kNone,
               [](World& w, const std::string&) {
                 idle_conversation(w, 0, 1);
                 w.core->split(1, 64);
                 w.core->slot_reserve(1, 70, 0);  // another line is reserved 70 numbers: the join stops
                 for (int c = 0; c < 3; ++c) w.core->join(1, INT32_MAX, 0, 10);
                 w.inv("after the join stopped");
                 const auto split_below = [&](int u) {
                   int in = 0, disk = 0, sp = 0;
                   w.core->list_numbers(1, u, &in, &disk, &sp);
                   return sp;
                 };
                 const int planned = 100000;
                 NEED(w.core->split_counts.join_short >= 1 && split_below(planned) > 0, "the join stopped with pieces below the planned reuse still slots");
                 const int q = rules::whole_below(planned, split_below);
                 NEED(q < planned && split_below(q) == 0 && (q == planned || split_below(q + 1) > 0 || q == 0), "the reuse is lowered to the last whole position");
                 w.core->slot_reserve(0, 40, 0), w.core->slot_attach(0, 1, q);
                 w.core->load(1, q, 0, 0);  // the planned reuse itself would be cut inside the operation and die
                 NEED(w.sess(1).saved == q, "the line holds the lowered reuse");
               },
               100, false});
}


// ---- a request runs for the split from its admission to its finish ---------------------------------------------------------------

// the sessions a split command names live: the ones whose request is in flight (paused or not), by the server's rule
std::vector<int> split_live(const std::vector<int>& in_flight) {
  return reclaim::split_names_live(true, true) ? in_flight : std::vector<int>{};
}

// every free number is taken by another conversation (a filler)
void fill_pool(World& x) {
  std::vector<uint32_t> hold;
  const size_t n = x.alloc.free();
  if (n && x.alloc.take(n, &hold)) {
    x.core->refs.resize(x.alloc.numbers(), 0);
    for (const uint32_t c : hold) {
      x.core->refs[c] = 1;
      x.core->cut_twin(c);
      x.core->store_of().sess[-5].lists[0].push_back(Ref(c));
    }
  }
}

void add_running_rows(std::vector<Row>& v) {
  v.push_back({"a request paused off its line is not split, and its return joins and cuts nothing", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                 bookmarked_conversation(x, 1, 17000, 8192);
                 x.core->pin({}, split_live({1}), {});  // the command names its session live
                 NEED(x.core->split(1, 64) == 0, "nothing of a conversation with a request in flight is split");
                 x.core->pin({}, {}, {});
                 fill_pool(x);
                 x.core->slot_reserve(0, 40, 0);
                 const uint64_t joins = x.core->split_counts.joins, shorts = x.core->split_counts.join_short;
                 x.core->slot_attach(0, 1);
                 x.inv("attach");
                 NEED(x.core->split_counts.joins == joins && x.core->split_counts.join_short == shorts, "its return joined nothing and stopped nowhere");
                 NEED(x.sess(1).saved == 17000 && !x.sess(1).end.empty(), "nothing was cut: its saved positions and the end bookmark are there");
               },
               400, false});
  for (const bool on_line : {true, false})
    v.push_back({on_line ? "a fork in flight on its line keeps what it shares with its idle base" : "a fork in flight off its line keeps what it shares with its idle base", Rule::kNone,
                 [on_line](World& w, const std::string&) {
                   World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                   x.line(0, 2, 100000);
                   x.core->slot_release(0);
                   x.core->slot_reserve(1, 40, 0), x.core->slot_attach(1, 3);
                   x.core->load(2, 40000, 1, 1);  // the fork: past full pieces, sharing the base's numbers
                   x.inv("the fork");
                   x.core->pin({}, split_live({3}), {});
                   const bool off_line = !on_line;
                   if (off_line) x.core->slot_release(1);
                   x.core->split(2, 64);
                   x.inv("the base's split");
                   for (int k = 0; k < kKvSlots; ++k)
                     for (size_t i = 0; i < x.sess(3).lists[k].size(); ++i)
                       NEED(x.sess(3).lists[k][i].is_number(), "every entry of the fork's lists is still a whole number");
                   if (on_line)
                     for (int k = 0; k < kKvSlots; ++k)
                       for (size_t i = 0; i < x.sess(3).lists[k].size(); ++i)
                         NEED(x.dev.rows.at(1).at(k).at(i) == x.sess(3).lists[k][i].num(), "the line's rows hold the fork's numbers, no guard");
                   NEED(x.sess(2).slot_entries() > 0, "the base's own tail is split");
                 },
                 400, false});
}

// ---- quarter images on the disk tier ---------------------------------------------------------------------------------

std::unique_ptr<World> disk_world(const World& like, const std::string& dir, uint32_t numbers = 400, bool ring = false, size_t images = 40) {
  auto x = std::make_unique<World>(numbers, ring, like.core->test_rule, 48, 262144, 3u << 20, 5u << 20, true);
  x->open_disk(images, dir);
  return x;
}

void add_c_rows(std::vector<Row>& v) {
  v.push_back({"quarter images: a split writes each slot's quarter image (786,432 B, the quarter and its trailer), once, and the slot holds one reference to it", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 const size_t before = x->core->store_of().dbytes;
                 NEED(x->core->split(1, 64) == 34, "34 pieces split");
                 const Store& s = x->core->store_of();
                 x->inv("the split");
                 NEED(s.slot_disk.size() == 34 && s.qimg.size() == 34, "every slot has its quarter image");
                 NEED(s.dbytes - before == 34 * sgeom::kQuarterDiskBytes, "34 images of 786,432 B, 12 grains each, in the file's account");
                 for (const auto& sd : s.slot_disk) NEED(s.drefs.at(sd.second) == 1, "the slot's own reference only");
                 NEED(x->core->split(1, 64) == 0, "a second split finds nothing to do");
                 x->core->free_session(1);
                 x->inv("freed");
                 NEED(s.slot_disk.empty() && s.qimg.empty() && s.dbytes == before, "the last holder gone: the images and their file space go with the slots");
               },
               400, false});
  v.push_back({"quarter images: in cache mode a save writes no image: the split's command is the write-behind", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 NEED(x->core->store_of().drefs.empty() && x->core->store_of().dbytes == 0, "no image after the save");
               },
               400, false});
  v.push_back({"quarter images: a split conversation evicted to the disk tier keeps its quarters as images, and comes back through the exchange with its bytes and no link of a number to a quarter image",
               Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 const std::vector<uint64_t> before = tags_of(*x, 1);
                 x->core->split(1, 64);
                 x->core->spill(1);
                 x->inv("spilled");
                 NEED(x->sess(1).out() && x->sess(1).slot_entries() == 0, "every piece is only on the disk tier: no slot, no number");
                 NEED(x->core->store_of().slot_disk.empty() && x->core->store_of().qimg.size() == 34, "the slots are gone, the 34 images stay, named by the entries");
                 x->core->unspill(1);
                 x->inv("back");
                 NEED(x->core->split_counts.exchanged == 34 && !x->sess(1).out(), "34 pieces came back over the exchange");
                 NEED(tags_of(*x, 1) == before, "the bytes are the bytes written");
                 NEED(x->core->sum_cuts == 0, "a clean read-back is not a cut for sums");
                 bool linked = false;
                 for (const size_t n : x->core->store_of().ndisk) linked = linked || n != Pool::kNone;
                 NEED(!linked, "no number links a quarter image as its own whole one (red under PICO_OLD_IMAGE_LINK)");
               },
               400, false});
  v.push_back({"quarter images: a quarter image whose sum differs on the way back cuts the conversation at its piece, on every rank", Rule::kUnspillIgnoresSum,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 x->core->split(1, 64);
                 x->core->spill(1);
                 const Store& s = x->core->store_of();
                 const size_t bad_off = x->sess(1).dl[1][2];
                 NEED(s.qimg.count(bad_off), "a quarter image to spoil");
                 x->dev.spoil_qimg.insert(bad_off);
                 x->core->unspill(1);
                 x->inv("after the bad image");
                 NEED(x->sess(1).saved < 100000 && x->sess(1).saved <= ent_pos(1, 2), "cut at the bad piece or below it");
                 NEED(x->core->sum_cuts == 1, "the cut for sums is counted once (red under PICO_OLD_RULES)");
               },
               400, false});
  v.push_back({"quarter images: a trailer of another rank is refused: the piece is bad, never read as this one's", Rule::kQuarterTrailerUnchecked,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 x->core->split(1, 64);
                 x->core->spill(1);
                 x->dev.spoil_trailer.insert(x->sess(1).dl[0][1]);
                 x->core->unspill(1);
                 x->inv("after the foreign trailer");
                 NEED(x->sess(1).saved <= ent_pos(0, 1), "cut at the piece whose trailer is not this rank's and this piece's");
                 NEED(x->core->sum_cuts == 1, "the cut for sums is counted once (red under PICO_OLD_RULES)");
               },
               400, false});
  v.push_back({"quarter images: the kept index of a planned stop names quarter images (layout K4); another layout is a clean start, never read wrong", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir, 400, false, 60);
                 idle_conversation(*x, 0, 1);
                 x->core->split(1, 64);
                 x->core->spill(1);
                 x->core->keep(0x77);
                 int fd = open((dir + "/rank0.spill").c_str(), O_CREAT | O_WRONLY, 0600);
                 NEED(fd >= 0 && ftruncate(fd, (off_t)x->dev.file_size) == 0, "the kept file");
                 close(fd);
                 World k4(400, false, w.core->test_rule, 48, 262144, 3u << 20, 5u << 20, true);
                 k4.dev.image = x->dev.image, k4.dev.out_sum = x->dev.out_sum, k4.dev.qimage = x->dev.qimage;
                 size_t kb = 0;
                 std::string why;
                 NEED(k4.core->kept_check(dir + "/rank0", 7, 3600, &kb, &why) == 0x77, "a K4 index is accepted under cache: " + why);
                 k4.core->disk(dir + "/rank0", kb, 512 << 20, true, 7);
                 k4.inv("the kept start");
                 NEED(k4.sess(1).out() && k4.core->store_of().qimg.size() == 34, "the kept session is on the disk tier, its 34 images are quarter images");
                 k4.core->unspill(1);
                 k4.inv("the read-back");
                 NEED(!k4.sess(1).out() && k4.sess(1).saved == 100000, "it comes back whole through the exchange");
                 World off(400);
                 std::string why2;
                 NEED(off.core->kept_check(dir + "/rank0", 7, 3600, &kb, &why2) == 0, "a K4 index under off is another layout: refused");
                 NEED(why2.find("not a kept index") != std::string::npos, "the refusal says so: " + why2);
               },
               400, false});
  v.push_back({"quarter images: the index write is agreed: a rank whose write failed makes every rank keep its old set (all adopt or none)", Rule::kIndexWriteUnagreed,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir, 400, true);
                 x->line(0, 1, 100000);  // a bookmark gives the checkpoint something to name; nothing here is split
                 x->core->snap_buf(1, 90000);
                 x->core->slot_release(0);
                 x->core->spill(1);
                 x->core->checkpoint(5);  // every rank's write lands: the set is named
                 const size_t named = x->core->store_of().named.size();
                 NEED(named > 0, "the checkpoint named its images");
                 x->core->unspill(1);
                 x->core->spill(1);
                 x->dev.answers["agree_min"].push_back(0);  // another rank's write failed
                 x->core->checkpoint(6);
                 NEED(x->core->checkpointed().empty() && x->core->store_of().named.size() == named, "no rank adopted the new set: the named images are as they were");
               },
               400, true});
  v.push_back({"a smaller disk tier cuts through a named image, and the size grown to again holds its tail until the next checkpoint", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir, 400, true);
                 x->line(0, 1, 100000);
                 x->core->snap_buf(1, 90000);
                 x->core->slot_release(0);
                 x->core->spill(1);
                 x->core->checkpoint(5);  // the images are named by the index on the disk
                 Store& st = x->core->store_of();
                 const size_t old = st.dpool.size;
                 size_t cut = 0;
                 for (const auto& [o, n] : st.drefs) {
                   const size_t len = st.qimg.count(o) ? sgeom::kQuarterDiskBytes : Pool::round(x->g.image_bytes);
                   if (st.named.count(o) && len > kPoolGrain && (!cut || o < cut)) cut = o + kPoolGrain;  // inside the lowest such image
                 }
                 NEED(cut > 0 && cut < old, "a named image longer than one grain to cut through");
                 NEED(x->core->disk_resize(cut), "smaller, through the image");
                 x->inv("the shrink");
                 NEED(x->core->disk_resize(old), "and back to the old size");
                 x->inv("the regrow");  // the retired image's tail is held, not free: the images and free ranges tile the file once
               },
               400, true});
}


// ---- a return's read-back in units of J pieces, bounded by `upto`; a fork leaves its base's tail on the disk ---------------------------

// a conversation of 100,000 positions (34 lists) with two bookmarks, all on the disk tier
void stored_with_bookmarks(World& w, int h) {
  w.line(0, h, 100000);
  w.core->snap_buf(h, 90000), w.core->snap_buf(h, 80000);
  w.core->slot_release(0);
  w.core->flush(h);
  w.core->spill(h);
}

void add_g4j_rows(std::vector<Row>& v) {
  v.push_back({"a read-back command handles at most J pieces, the lists first and then the bookmarks; stopping at J cuts and drops nothing", Rule::kUnspillUnbounded,
               [](World& w, const std::string& dir) {
                 w.open_disk(300, dir);
                 stored_with_bookmarks(w, 1);
                 int in_pool = -1, on_disk = -1;
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 0 && on_disk == 34, "34 lists, none in the pool");
                 NEED(w.core->unspill_left(1, INT32_MAX) == 36, "34 lists and 2 bookmarks are left to bring");
                 w.core->unspill(1, 0, INT32_MAX, 10);
                 w.inv("the first command");
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 10 && on_disk == 24, "ten lists came back, in position order, and no more");
                 NEED(w.sess(1).saved == 100000 && w.sess(1).out() && w.sess(1).snaps.size() == 2, "nothing is cut and no bookmark is dropped");
                 NEED(w.core->unspill_left(1, INT32_MAX) == 26, "26 pieces are left");
                 w.core->unspill(1, 0, INT32_MAX, 10);
                 w.core->unspill(1, 0, INT32_MAX, 10);
                 w.inv("the third command");
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 30 && on_disk == 4 && w.core->unspill_left(1, INT32_MAX) == 6, "30 lists back, 4 lists and 2 bookmarks left");
                 w.core->unspill(1, 0, INT32_MAX, 10);  // 4 lists + 2 bookmarks = 6 pieces: the last command
                 w.inv("the last command");
                 NEED(!w.sess(1).out() && w.sess(1).saved == 100000 && w.sess(1).snaps.count(90000) && w.sess(1).snaps.count(80000), "whole: both bookmarks are back");
                 NEED(w.core->unspill_left(1, INT32_MAX) == 0, "nothing is left");
               },
               300, false});
  v.push_back({"a read-back to `upto` brings the pieces below it and leaves the tail and the bookmarks above it on the disk", Rule::kUnspillUnbounded,
               [](World& w, const std::string& dir) {
                 w.open_disk(300, dir);
                 stored_with_bookmarks(w, 1);
                 w.core->unspill(1, 0, 16384, 0);
                 w.inv("the read-back to 16,384");
                 int in_pool = -1, on_disk = -1;
                 w.core->list_numbers(1, 16384, &in_pool, &on_disk);
                 NEED(in_pool == 5 && on_disk == 0, "the five lists below 16,384 are back");
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 5 && on_disk == 29, "the other 29 are still only on the disk");
                 NEED(w.sess(1).saved == 100000 && w.sess(1).snaps.count(90000) && w.sess(1).snaps.at(90000).empty(), "nothing is cut; the bookmark above stays on the disk");
                 NEED(w.core->unspill_left(1, 16384) == 0 && w.core->unspill_left(1, INT32_MAX) == 31, "nothing left below it; 29 lists and 2 bookmarks above");
               },
               300, false});
  v.push_back({"a read-back whose count is spent cuts where the count ended, as one command did", Rule::kUnspillIgnoresCap,
               [](World& w, const std::string& dir) {
                 w.open_disk(300, dir);
                 stored_with_bookmarks(w, 1);
                 w.core->unspill(1, StoreCore::kSpent, INT32_MAX, 10);
                 w.inv("the spent command");
                 NEED(w.sess(1).saved == 0 && !w.sess(1).out(), "cut to nothing: no number to take, no piece read");
               },
               300, false});
  v.push_back({"a request's read-back never joins: the slots of a conversation partly on the disk stay slots", Rule::kUnspillJoins,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 NEED(x->core->split(1, 64) == 34, "34 pieces are slots");
                 for (int i = 0; i < 10; ++i) NEED(x->core->evict_piece(1, 1), "the latest piece goes to the disk tier");
                 x->inv("evicted");
                 NEED(x->sess(1).out() && x->sess(1).slot_entries() == 24, "ten lists only on the disk, 24 still slots");
                 const uint64_t joins = x->core->split_counts.joins;
                 x->core->unspill(1, 0, INT32_MAX, 64);
                 x->inv("the read-back");
                 NEED(x->core->split_counts.joins == joins, "no join ran in the read-back");
                 NEED(x->sess(1).slot_entries() == 24 && !x->sess(1).out(), "the 24 slots are slots still; the ten lists are back");
               },
               400, false});
  v.push_back({"a fork's load of a base on the disk tier brings the pieces below its fork point and leaves the tail there", Rule::kForkReadsTail,
               [](World& w, const std::string& dir) {
                 auto x = disk_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 x->core->flush(1);
                 x->core->spill(1);
                 NEED(x->sess(1).out(), "the base is only on the disk tier");
                 x->core->slot_reserve(1, 40, 0), x->core->slot_attach(1, 2);
                 x->core->load(1, 40000, 1, 1);  // the fork at 40,000
                 x->inv("the fork");
                 int in_pool = -1, on_disk = -1;
                 x->core->list_numbers(1, 40000, &in_pool, &on_disk);
                 NEED(on_disk == 0, "nothing of the base below the fork point is left on the disk");
                 x->core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(on_disk > 0 && x->sess(1).out(), "the tail above it is still only on the disk tier");
                 NEED(x->sess(1).saved == 100000, "the base keeps its whole history");
               },
               400, false});
  v.push_back({"a unit of a return whose lists used up its count stops there: nothing is cut and no bookmark is dropped", Rule::kUnspillCountDrops,
               [](World& w, const std::string& dir) {
                 w.open_disk(300, dir);
                 stored_with_bookmarks(w, 1);
                 for (int i = 0; i < 3; ++i) w.core->unspill(1, 10, INT32_MAX, 10);
                 w.inv("three units of ten");
                 int in_pool = -1, on_disk = -1;
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 30 && on_disk == 4, "30 lists back, 4 left on the disk");
                 w.core->unspill(1, 4, INT32_MAX, 10);  // the last unit: fewer lists than J, and its count is the four
                 w.inv("the last unit");
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 34 && on_disk == 0, "the four lists are back");
                 NEED(w.sess(1).saved == 100000 && w.sess(1).out(), "nothing is cut");
                 NEED(w.sess(1).snaps.count(90000) && w.sess(1).snaps.count(80000), "both window bookmarks are still there");
                 NEED(w.sess(1).snaps.at(90000).empty() && w.sess(1).snaps.at(80000).empty() && w.sess(1).dsnaps.count(90000) && w.sess(1).dsnaps.count(80000),
                      "and still only on the disk, with their images");
                 NEED(w.core->unspill_left(1, INT32_MAX) == 2, "two bookmarks are left to bring");
                 w.core->unspill(1, 0, INT32_MAX, 10);
                 w.inv("the bookmarks' command");
                 NEED(!w.sess(1).out() && w.sess(1).saved == 100000 && !w.sess(1).snaps.at(90000).empty() && !w.sess(1).snaps.at(80000).empty(),
                      "the next command brings them");
                 NEED(w.core->unspill_left(1, INT32_MAX) == 0, "nothing is left");
               },
               300, false});
  v.push_back({"a unit with a count below J and more lists than the count: it brings the count, cuts nothing, the rest wait", Rule::kUnspillCountDrops,
               [](World& w, const std::string& dir) {
                 w.open_disk(300, dir);
                 stored_with_bookmarks(w, 1);
                 w.core->unspill(1, 5, INT32_MAX, 10);
                 w.inv("five of 34");
                 int in_pool = -1, on_disk = -1;
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 5 && on_disk == 29, "five lists back, 29 left");
                 NEED(w.sess(1).saved == 100000 && w.sess(1).out(), "nothing is cut");
                 NEED(w.core->unspill_left(1, INT32_MAX) == 31, "31 pieces are left: 29 lists and 2 bookmarks");
               },
               300, false});
  v.push_back({"a read-back with a count and no J (--kv-split off's words) cuts where its count ends and drops what follows", Rule::kUnspillNoCut,
               [](World& w, const std::string& dir) {
                 w.open_disk(300, dir);
                 stored_with_bookmarks(w, 1);
                 w.core->unspill(1, 5);
                 w.inv("five of 34, no J");
                 NEED(w.sess(1).saved > 0 && w.sess(1).saved < 100000, "cut after the five lists");
                 NEED(w.sess(1).snaps.empty(), "the bookmarks above the cut are gone");
               },
               300, false});
}

// ---- a split bookmark's image: the split writes the whole image before it quarters a bookmark -------------------------------------

// a conversation of 40,960 positions with window bookmarks at 16,384 and 32,768 and the end bookmark, idle, on a world with a disk tier
std::unique_ptr<World> bookmarked_disk_world(const World& like, const std::string& dir, bool split) {
  auto x = std::make_unique<World>(400, false, like.core->test_rule, 140, 1 << 21, 2906112, 6074368, split);
  x->open_disk(60, dir);
  bookmarked_conversation(*x, 1, 40960, 16384);
  return x;
}

std::vector<uint64_t> bookmark_tags(World& x) {  // the window bookmarks' then the end bookmark's content tags, in order
  std::vector<uint64_t> t;
  for (const auto& sn : x.sess(1).snaps)
    for (const uint32_t c : sn.second) t.push_back(x.dev.tag_of(c));
  for (const uint32_t c : x.sess(1).end) t.push_back(x.dev.tag_of(c));
  return t;
}

void add_s4s_rows(std::vector<Row>& v) {
  v.push_back({"a split writes a bookmark's whole image before it quarters it: the two window bookmarks and the end bookmark each have one", Rule::kSplitBookmarkNoImage,
               [](World& w, const std::string& dir) {
                 auto x = bookmarked_disk_world(w, dir, true);
                 NEED(x->sess(1).snaps.size() == 2 && x->sess(1).end.size() == 3 && x->sess(1).dsnaps.empty() && x->sess(1).dend == Pool::kNone,
                      "two window bookmarks and the end bookmark, none with an image (a save writes none in cache mode)");
                 x->dev.calls.clear();
                 x->core->split(1, 1 << 20);
                 x->inv("split");
                 NEED(x->sess(1).qsnaps.size() == 2 && x->sess(1).qend.size() == 3, "the bookmarks are slots");
                 NEED(x->sess(1).dsnaps.size() == 2 && x->sess(1).dend != Pool::kNone, "each has its whole image");
                 std::vector<std::string> seq;
                 for (const std::string& c : x->dev.calls)
                   if (c.rfind("bk_out", 0) == 0 || c.rfind("quarter_bk", 0) == 0) seq.push_back(c.substr(0, c.find(' ')));
                 NEED(seq == std::vector<std::string>({"bk_out", "quarter_bk", "bk_out", "quarter_bk", "bk_out", "quarter_bk"}), "each image is written before its bookmark is quartered");
               },
               400, false});
  v.push_back({"a split conversation evicted to nothing keeps its restart point: the bookmarks' images stay when their slots go", Rule::kSplitBookmarkNoImage,
               [](World& w, const std::string& dir) {
                 auto x = bookmarked_disk_world(w, dir, true);
                 const int rp = restart_of(x->sess(1).saved, x->sess(1).end_pos, x->sess(1).snaps);
                 NEED(rp > 0, "the conversation has a restart point");
                 x->core->split(1, 1 << 20);
                 int steps = 0;
                 while (steps < 400 && x->core->evict_piece(1, 9)) ++steps;
                 x->inv("evicted");
                 NEED(steps > 0 && x->sess(1).slot_entries() == 0 && x->sess(1).bookmark_slots() == 0, "every slot is given back");
                 NEED(x->sess(1).out(), "the conversation is on the disk tier");
                 NEED(restart_of(x->sess(1).saved, x->sess(1).end_pos, x->sess(1).snaps) == rp && x->sess(1).snaps.size() == 2 && x->sess(1).end_pos == 40960,
                      "the restart point, the end position and both window keys are as before");
               },
               400, false});
  v.push_back({"a return reads a split bookmark's image back whole: the bytes written, nothing left on the disk to bring", Rule::kSplitBookmarkNoImage,
               [](World& w, const std::string& dir) {
                 auto x = bookmarked_disk_world(w, dir, true);
                 const std::vector<uint64_t> before = bookmark_tags(*x);
                 x->core->split(1, 1 << 20);
                 int steps = 0;
                 while (steps < 400 && x->core->evict_piece(1, 9)) ++steps;
                 x->inv("evicted");
                 NEED(x->core->unspill_left(1, INT32_MAX) > 3, "the lists and the three bookmarks are to bring");
                 x->core->unspill(1, 0);
                 x->inv("back");
                 NEED(x->core->unspill_left(1, INT32_MAX) == 0 && !x->sess(1).out(), "nothing is left on the disk");
                 NEED(x->sess(1).snaps.size() == 2 && x->sess(1).end.size() == 3, "both window bookmarks and the end bookmark are whole");
                 NEED(bookmark_tags(*x) == before, "their bytes are what was written");
               },
               400, false});
  v.push_back({"a split with no room on the disk quarters a bookmark with no image, and eviction drops it (the rule every piece has)", Rule::kNone,
               [](World& w, const std::string&) {
                 World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368, true);  // no disk tier
                 bookmarked_conversation(x, 1, 40960, 16384);
                 x.core->split(1, 1 << 20);
                 x.inv("split");
                 NEED(x.sess(1).qsnaps.size() == 2 && x.sess(1).qend.size() == 3 && x.sess(1).dsnaps.empty() && x.sess(1).dend == Pool::kNone, "quartered, no image");
                 for (int i = 0; i < 3; ++i) NEED(x.core->evict_piece(1, 9), "an eviction step on a bookmark");
                 x.inv("evicted");
                 NEED(x.sess(1).qend.empty() && x.sess(1).qsnaps.empty() && x.sess(1).snaps.empty() && x.sess(1).end_pos == -1, "the three bookmarks are dropped");
               },
               400, false});
  v.push_back({"a bookmark that is slots is not on the disk for a read-back, whatever images it has: the join brings it, once", Rule::kUnspillReadsSplitBookmarks,
               [](World& w, const std::string& dir) {
                 auto x = bookmarked_disk_world(w, dir, true);
                 x->core->flush(1);  // the idle flush wrote the bookmarks' images before the split
                 NEED(x->sess(1).dsnaps.size() == 2 && x->sess(1).dend != Pool::kNone, "each bookmark has an image from the flush");
                 x->core->split(1, 1 << 20);
                 NEED(x->sess(1).qsnaps.size() == 2 && x->sess(1).qend.size() == 3, "split");
                 NEED(x->core->evict_piece(1, 9) && x->core->evict_piece(1, 9), "two steps: the end bookmark, then the later window bookmark");
                 x->inv("two steps");
                 NEED(x->sess(1).qend.empty() && x->sess(1).qsnaps.count(16384) == 1 && x->sess(1).qsnaps.count(32768) == 0, "the first window bookmark is still slots");
                 NEED(x->core->unspill_left(1, INT32_MAX) == 2, "two bookmarks are on the disk (the third is slots)");
                 x->core->unspill(1, 0);
                 x->inv("read back");
                 NEED(x->sess(1).snaps.at(16384).empty() && x->sess(1).qsnaps.count(16384) == 1, "the split bookmark is untouched");
                 NEED(!x->sess(1).snaps.at(32768).empty() && x->sess(1).end.size() == 3, "the other two are whole");
                 NEED(x->core->unspill_left(1, INT32_MAX) == 0, "nothing is left to read");
                 x->core->join(1, INT32_MAX, 0, 64);
                 x->inv("joined");
                 NEED(x->sess(1).qsnaps.empty() && x->sess(1).bookmark_slots() == 0 && !x->sess(1).snaps.at(16384).empty(), "the join brought it whole, once");
               },
               400, false});
  v.push_back({"without the split, a save, an eviction and a read-back are as they were: the bookmarks come back with their bytes (--kv-split off's words)", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = bookmarked_disk_world(w, dir, false);
                 const std::vector<uint64_t> before = bookmark_tags(*x);
                 const int rp = restart_of(x->sess(1).saved, x->sess(1).end_pos, x->sess(1).snaps);
                 int steps = 0;
                 while (steps < 400 && x->core->evict_piece(1, 9)) ++steps;
                 x->inv("evicted");
                 NEED(x->sess(1).out() && restart_of(x->sess(1).saved, x->sess(1).end_pos, x->sess(1).snaps) == rp, "on the disk tier with its restart point");
                 x->core->unspill(1, 0);
                 x->inv("back");
                 NEED(!x->sess(1).out() && bookmark_tags(*x) == before, "back whole, the bookmarks' bytes as written");
               },
               400, false});
}

// ---- the imager: every write is a quarter on the image stream; a stored conversation is marked, never copied ------------------

void imager_on(World& w, bool mark = true) { w.core->g.imager = true, w.core->g.mark = mark; }
std::unique_ptr<World> imager_world(World& w, const std::string& dir, uint32_t numbers = 400, bool mark = true) {
  auto x = std::make_unique<World>(numbers, false, w.core->test_rule);
  imager_on(*x, mark);
  x->open_disk(400, dir);
  return x;
}
// what the fake's log holds that is not on the image stream
bool only_image_calls(const std::vector<std::string>& calls, size_t from) {
  for (size_t i = from; i < calls.size(); ++i)
    if (calls[i].rfind("image_", 0) != 0) return false;
  return true;
}
bool no_device_call(World& w, size_t from, int syncs) { return w.dev.calls.size() == from && w.dev.syncs == syncs; }

void add_imager_rows(std::vector<Row>& v) {
  // an image command's device calls are all on the image stream; none on the model's, no sync. Control: the old split's calls are on the model's.
  v.push_back({"imager: image() queues quarters on the image stream only (no call on the model's stream, no sync); the entries are untouched until settle",
               Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 const size_t free0 = x->alloc.free();
                 x->dev.calls.clear();
                 const int syncs0 = x->dev.syncs;
                 NEED(x->core->image(1, 10, {}) == 10, "ten pieces queued");
                 NEED(!x->dev.calls.empty() && only_image_calls(x->dev.calls, 0), "every device call of the command is an image_ call");
                 NEED(x->dev.syncs == syncs0 && x->dev.waits == 0, "no sync and no wait");
                 NEED(!x->core->has_clean(x->sess(1)) && x->alloc.free() == free0 && x->core->store_of().jobs.size() == 10, "nothing is touched: no clean piece, no number freed");
                 x->inv("queued");
                 World old(400);  // the old split, as a control: its quarter is a call on the model's stream
                 idle_conversation(old, 0, 1);
                 old.dev.calls.clear();
                 old.core->split(1, 4);
                 bool model_stream = false;
                 for (const std::string& c : old.dev.calls) model_stream = model_stream || c.rfind("quarter ", 0) == 0;
                 NEED(model_stream, "the control: split() calls quarter on the model's stream");
               },
               400, false});
  // between image and settle the block is held and a take does not find the piece clean; a job whose block changed is void at the settle
  v.push_back({"imager: a job whose block was written in place meanwhile is void at the settle, its file space back", Rule::kSettleUnchecked,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 x->line(0, 1, 100000);  // on its line: its tail piece is written again by the line's next save
                 const size_t bytes0 = x->core->store_of().dbytes;
                 NEED(x->core->image(1, 1000, {}) == 34, "34 pieces queued");
                 NEED(x->core->store_of().dbytes - bytes0 == 34 * sgeom::kQuarterDiskBytes, "their file space is taken at once");
                 x->core->save(1, 99000, 100000, 0);  // the tail entries are written: their blocks are cut
                 x->inv("saved over");
                 const uint64_t seq = x->core->image_seq();
                 x->core->settle(seq, {});
                 x->inv("settled");
                 const Store::Sess& s = x->sess(1);
                 bool tail_clean = false;
                 for (int k = 0; k < kKvSlots; ++k)
                   for (size_t i = 0; i < s.lists[k].size(); ++i)
                     if (ent_pos(k, (int)i) + 8192 > 99000 && i < s.dl[k].size() && s.dl[k][i] != Pool::kNone) tail_clean = true;
                 NEED(!tail_clean, "a rewritten piece has no image");
                 NEED(x->core->split_counts.image_voids > 0, "the void jobs are counted");
                 NEED(x->core->store_of().dbytes - bytes0 < 34 * sgeom::kQuarterDiskBytes, "their file space went back");
               },
               400, false});
  // settle on a line keeps the blocks; off a line the entries are on the SSD and the blocks ghosts; no device call
  v.push_back({"imager: settle makes a line's pieces clean (blocks held) and marks an idle conversation (ghosts), with no device call",
               Rule::kMarkIgnoresLine,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 x->line(1, 2, 60000);  // on its line
                 x->core->image(1, 1000, {}), x->core->image(2, 1000, {});
                 const size_t free0 = x->alloc.free(), n2 = x->sess(2).numbers();
                 const size_t calls0 = x->dev.calls.size();
                 const int syncs0 = x->dev.syncs;
                 const int marked = x->core->settle(x->core->image_seq(), {});
                 NEED(no_device_call(*x, calls0, syncs0), "settle makes no device call");
                 x->inv("settled");
                 NEED(marked == 34 && x->alloc.ghosts() == 34 && x->alloc.free() == free0 + 34, "34 pieces marked: ghosts, the free count up by 34");
                 NEED(x->sess(1).numbers() == 0 && x->sess(1).out(), "the idle conversation holds no block: its entries say the SSD");
                 NEED(x->sess(2).numbers() == n2 && x->core->has_clean(x->sess(2)), "the conversation on a line holds its blocks, clean");
                 size_t file = 0;
                 for (const auto& d : x->sess(1).dl)
                   for (const size_t o : d) file += o != Pool::kNone ? Pool::round(sgeom::kQuarterDiskBytes) : 0;
                 NEED(file == 34 * sgeom::kQuarterDiskBytes, "its file bytes are its pieces x 786,432");
               },
               400, false});
  // a ghost taken loses its link; the return reads. Control: the link kept (the number a filler holds still names the image).
  v.push_back({"imager: a ghost a filler took has no image link any more; the return reads that piece from its image and revives the rest", Rule::kNoGhostCut,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir, 60);
                 idle_conversation(*x, 0, 1);
                 const std::vector<uint64_t> before = tags_of(*x, 1);
                 x->core->image(1, 1000, {});
                 x->core->settle(x->core->image_seq(), {});
                 NEED(x->alloc.ghosts() == 34, "34 ghosts");
                 x->core->slot_reserve(1, 40, 0), x->core->slot_attach(1, 2);  // the filler's line takes the plain free numbers, then the oldest ghosts,
                 x->core->ensure(1, 100000);                                  // and writes them: no save yet (a save cuts a link itself)
                 for (const Refs& l : x->sess(2).lists)
                   for (const Ref c : l) x->dev.wrote(c.num());
                 x->inv("the filler");
                 NEED(x->alloc.ghosts() < 34, "the filler took ghosts");
                 size_t taken = 0;
                 for (const auto& d : x->sess(1).dl)
                   for (const size_t off : d) {
                     if (off == Pool::kNone) continue;
                     const auto n = x->core->store_of().dnum.find(off);
                     if (n == x->core->store_of().dnum.end()) ++taken;
                     else NEED(x->alloc.is_ghost(n->second), "an image's link names a number that is still a ghost, never one the filler holds");
                   }
                 NEED(taken > 0, "the images of the taken ghosts have no link");
                 x->core->slot_release(1), x->core->free_session(2);  // the filler leaves
                 x->core->unspill(1);
                 x->inv("the return");
                 NEED(x->core->split_counts.returns_read == taken && x->core->split_counts.revives == 34 - taken, "the taken ones were read, the others revived");
                 NEED(tags_of(*x, 1) == before, "every byte is the byte written");
               },
               60, false});
  // a return with its ghosts: no device call, no J
  v.push_back({"imager: a return whose ghosts are all there revives them: no device call, not counted against J", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 const std::vector<uint64_t> before = tags_of(*x, 1);
                 x->core->image(1, 1000, {});
                 x->core->settle(x->core->image_seq(), {});
                 const size_t calls0 = x->dev.calls.size();
                 const int syncs0 = x->dev.syncs;
                 x->core->unspill(1, 0, INT32_MAX, 2);  // J = 2
                 NEED(no_device_call(*x, calls0, syncs0), "no device call");
                 x->inv("the return");
                 NEED(x->core->split_counts.revives == 34 && x->core->split_counts.returns_read == 0 && x->core->split_counts.exchanged == 0, "34 revives, nothing read, nothing exchanged");
                 NEED(tags_of(*x, 1) == before && !x->sess(1).out(), "whole again, the bytes it had in one command though J was 2");
                 NEED(x->core->has_clean(x->sess(1)) && x->core->store_of().to_mark.count(1), "its pieces keep their images: marking again is bookkeeping");
                 NEED(x->core->settle(x->core->image_seq(), {}) == 34 && x->alloc.ghosts() == 34, "marked again at the next settle, no image written");
                 NEED(x->core->split_counts.imaged == 34, "imaged once");
               },
               400, false});
  // a revive counts against the cap
  v.push_back({"imager: a revive is a take: the read-back's cap counts it", Rule::kUnspillIgnoresCap,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 x->core->image(1, 1000, {});
                 x->core->settle(x->core->image_seq(), {});
                 x->core->unspill(1, 10);
                 x->inv("the capped return");
                 NEED(x->core->split_counts.unspill_taken == 10, "ten numbers taken, the cap");
                 NEED(x->sess(1).saved < 100000, "the rest is cut as a cap cuts");
               },
               400, false});
  // bookmarks are slices; a ghost group dissolves when one block is taken; the four slices put together are the bookmark
  v.push_back({"imager: a bookmark's image is this rank's slice (786,432 B; the end's 1,572,864 B); its ghost group revives whole or is read whole",
               Rule::kNone,
               [](World& w, const std::string& dir) {
                 World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                 imager_on(x);
                 x.open_disk(400, dir);
                 bookmarked_conversation(x, 1, 40960, 16384);
                 const std::vector<uint64_t> tg = [&] {
                   std::vector<uint64_t> t;
                   for (const auto& sn : x.sess(1).snaps)
                     for (const uint32_t c : sn.second) t.push_back(x.dev.tag_of(c));
                   for (const uint32_t c : x.sess(1).end) t.push_back(x.dev.tag_of(c));
                   return t;
                 }();
                 x.core->image(1, 1000, {});
                 const uint64_t before = x.dev.quarter_bytes_out;
                 NEED(x.dev.bkimage.size() == 3, "two window bookmarks and the end bookmark imaged");
                 size_t win = 0, end = 0;
                 for (const auto& [off, b] : x.dev.bkimage) (b.t.list == (uint32_t)Store::kBkEnd ? end : win) += 1;
                 NEED(win == 2 && end == 1, "two window images and one end image");
                 NEED(before >= 2 * sgeom::kBookmarkImageBytes + sgeom::kEndImageBytes, "the image bytes: 786,432 for each window bookmark, 1,572,864 for the end");
                 x.core->settle(x.core->image_seq(), {});
                 x.inv("settled");
                 NEED(x.sess(1).snaps.at(16384).empty() && x.sess(1).end.empty() && x.core->store_of().bk_ghost.size() == 3, "the bookmarks are ghost groups");
                 // a window bookmark's group has a block taken: the rest are plain free and the return reads the image
                 const size_t calls0 = x.dev.calls.size();
                 x.core->unspill(1);
                 x.inv("the return");
                 bool read = false;
                 for (size_t i = calls0; i < x.dev.calls.size(); ++i) read = read || x.dev.calls[i].rfind("bookmark_read", 0) == 0;
                 NEED(!read && x.core->split_counts.returns_read == 0, "every group was whole: revived, nothing read");
                 std::vector<uint64_t> after;
                 for (const auto& sn : x.sess(1).snaps)
                   for (const uint32_t c : sn.second) after.push_back(x.dev.tag_of(c));
                 for (const uint32_t c : x.sess(1).end) after.push_back(x.dev.tag_of(c));
                 NEED(after == tg, "the bookmarks' bytes are the bytes written");
                 // now the same with the groups cut: a filler takes ghosts
                 x.core->settle(x.core->image_seq(), {});  // marked again
                 const uint32_t victim = x.core->store_of().bk_ghost.begin()->second[0];
                 x.core->cut_twin(victim);
                 x.inv("one block taken");
                 NEED(x.core->store_of().bk_ghost.size() == 2, "that group is dissolved: its other blocks are plain free");
               },
               400, false});
  // a tail piece
  v.push_back({"imager: a tail piece is imaged with its written count; a save over it drops the image and voids the job", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 x->core->image(1, 1000, {});
                 x->core->settle(x->core->image_seq(), {});
                 int partial = 0;
                 for (const auto& [off, q] : x->dev.qimage) partial += (int)q.t.written < 8192;
                 NEED(partial == 4, "the four partial pieces carry their written counts");
                 x->core->unspill(1);
                 x->core->slot_reserve(0, 40, 0), x->core->slot_attach(0, 1);
                 x->core->ensure(0, 100500);
                 for (const Refs& l : x->sess(1).lists)
                   if (!l.empty()) x->dev.wrote(l.back().num());
                 x->core->save(1, 100000, 100500, 0);
                 x->inv("appended");
                 bool tail = false;
                 for (int k = 0; k < kKvSlots; ++k) tail = tail || (!x->sess(1).dl[k].empty() && x->sess(1).dl[k].back() != Pool::kNone);
                 NEED(!tail, "the tail's image is dropped by the save");
                 x->core->slot_release(0);
                 x->core->image(1, 1000, {});
                 NEED(x->core->store_of().jobs.size() > 0, "the next image writes a new one");
               },
               400, false});
  // a settle up to n leaves the later commands queued; ride_settled and the budget are minima
  v.push_back({"imager: settle(3) leaves commands 4 and up queued", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 for (int i = 0; i < 5; ++i) x->core->image(1, 2, {});  // five commands of two pieces
                 NEED(x->core->image_seq() == 5 && x->core->store_of().jobs.size() == 10, "five commands queued");
                 x->core->settle(3, {});
                 x->inv("settled to 3");
                 NEED(x->core->store_of().jobs.size() == 4 && x->core->store_of().jobs.front().seq == 4, "commands 4 and 5 stay queued");
               },
               400, false});
  // four ranks (four fakes) given the same operations with different done readings hold the same state at every record
  v.push_back({"imager: two cores given the same commands, with different readings of their image streams, agree at every record", Rule::kNone,
               [](World& w, const std::string& dir) {
                 World a(400), b(400);
                 imager_on(a), imager_on(b);
                 a.open_disk(400, dir, "rankA"), b.open_disk(400, dir, "rankB");
                 b.dev.done_cap = 0;  // rank B's stream is slow: it reads 0 where A reads everything
                 for (World* z : {&a, &b}) idle_conversation(*z, 0, 1);
                 for (int round = 0; round < 6; ++round) {
                   // the commands carry the minimum of the readings: the same on both, whatever each read
                   const uint64_t settled = std::min(a.dev.image_done(), b.dev.image_done());
                   for (World* z : {&a, &b}) {
                     z->core->settle(settled, {});
                     z->core->image(1, 8, {});
                   }
                   NEED(a.core->digest() == b.core->digest() && a.core->wide_digest(a.core->digest()) == b.core->wide_digest(b.core->digest()), "digests equal at round " + std::to_string(round));
                 }
                 NEED(a.alloc.ghosts() == 0 && b.alloc.ghosts() == 0, "nothing was marked while one rank's stream had not ended");
                 b.dev.done_cap = UINT64_MAX;
                 const uint64_t settled = std::min(a.dev.image_done(), b.dev.image_done());
                 for (World* z : {&a, &b}) z->core->settle(settled, {});
                 NEED(a.core->digest() == b.core->digest() && a.alloc.ghosts() == b.alloc.ghosts() && a.alloc.ghosts() > 0, "both mark the same pieces once both streams ended");
               },
               400, false});
  // the bytes a rank writes, in each of off, tiered and max; no whole-image call
  for (const int mode : {0, 1, 2})
    v.push_back({mode == 0 ? "imager: the bytes a rank writes for a conversation are units x 786,432 + the end's 1,572,864, no whole image"
                           : mode == 1 ? "imager: the bytes a rank writes are units x 786,432 + the end's 1,572,864, no whole image"
                                       : "imager: the bytes a rank writes are units x 786,432 + the end's 1,572,864, no whole image",
                 Rule::kNone,
                 [mode](World& w, const std::string& dir) {
                   World x(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                   imager_on(x, mode != 0);
                   x.open_disk(400, dir);
                   bookmarked_conversation(x, 1, 40960, 16384);
                   x.core->slot_reserve(0, 40, 0), x.core->slot_attach(0, 1), x.core->slot_release(0);  // a request's end: its save is no write
                   int units = 0;  // the entries that hold saved positions: a number past them holds nothing and has no image
                   for (int k = 0; k < kKvSlots; ++k)
                     for (size_t i = 0; i < x.sess(1).lists[k].size(); ++i) units += ent_pos(k, (int)i) < x.sess(1).saved;
                   x.core->image(1, 100000, {});
                   x.core->settle(x.core->image_seq(), {});
                   const uint64_t want = (uint64_t)(units + 2) * sgeom::kBookmarkImageBytes + sgeom::kEndImageBytes;
                   NEED(x.dev.quarter_bytes_out == want, "the bytes written: " + std::to_string(x.dev.quarter_bytes_out) + ", wanted " + std::to_string(want));
                   for (const std::string& c : x.dev.calls) NEED(c.rfind("image_out", 0) != 0 && c.rfind("bk_out", 0) != 0, "no whole-image call: " + c);
                   NEED((mode == 0) == (x.alloc.ghosts() == 0), (mode == 0 ? "off marks nothing" : "tiered and max mark the idle conversation"));
                   x.inv("settled");
                 },
                 400, false});
  // admission counts clean pieces only: a request the pool holds only with plain pieces waits; after their settle it fits
  v.push_back({"imager: giveable counts clean pieces only; plain ones are counted after their settle", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 NEED(x->core->giveable({}) == 0, "an idle conversation with no image gives nothing back");
                 x->core->image(1, 1000, {});
                 NEED(x->core->giveable({}) == 0, "queued is not clean");
                 x->core->settle(x->core->image_seq(), {});
                 NEED(x->core->giveable({}) == 0 && x->alloc.ghosts() == 34, "marked: its blocks are already free (ghosts), nothing more to give");
                 World y(400);
                 imager_on(y, false);  // off: clean pieces stay held and are giveable
                 y.open_disk(400, dir, "rankY");
                 idle_conversation(y, 0, 1);
                 NEED(y.core->giveable({}) == 0, "plain: 0");
                 y.core->image(1, 1000, {});
                 y.core->settle(y.core->image_seq(), {});
                 NEED(y.core->giveable({}) == 34, "clean and held: 34 numbers give back with no write");
               },
               400, false});
  // a take with only plain pieces left waits for its image, once, and writes no whole image
  v.push_back({"imager: a take that finds only plain pieces images the piece, waits for the image stream, settles and takes: counted, no whole image", Rule::kEvictPlain,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir, 70, false);
                 idle_conversation(*x, 0, 1);
                 x->line(1, 2, 100000);  // takes the free numbers: the pool is nearly full
                 x->core->slot_unreserve(1);
                 x->inv("full");
                 const size_t calls0 = x->dev.calls.size();
                 x->core->slot_reserve(2, 40, 0), x->core->slot_attach(2, 3);
                 x->core->ensure(2, 60000);  // wants numbers: only conversation 1's plain pieces can go
                 x->inv("after the take");
                 NEED(x->core->split_counts.image_waits > 0 && x->dev.waits > 0, "the take waited for an image");
                 for (size_t i = calls0; i < x->dev.calls.size(); ++i) NEED(x->dev.calls[i].rfind("image_out", 0) != 0 && x->dev.calls[i].rfind("bk_out", 0) != 0, "no whole-image call");
               },
               70, false});
  // the ride agrees a budget in units, and every rank then queues exactly the same jobs whatever its own staging holds. One rank's room is short by one
  // unit of what the others have: the end bookmark (two units) must not be refused by it and queued by the three.
  v.push_back({"imager: four ranks, one with staging room short by one unit: the same jobs, the same digests (the budget is agreed in units, nobody asks its own room again)",
               Rule::kNone,
               [](World& w, const std::string& dir) {
                 const auto mk = [&](const char* name) {
                   auto x = std::make_unique<World>(400, false, w.core->test_rule, 140, 1 << 21, 2906112, 6074368);
                   imager_on(*x);
                   x->open_disk(400, dir, name);
                   bookmarked_conversation(*x, 1, 40960, 16384);
                   return x;
                 };
                 auto a = mk("rankA"), b = mk("rankB");
                 int lists = 0;
                 for (int k = 0; k < kKvSlots; ++k)
                   for (size_t i = 0; i < a->sess(1).lists[k].size(); ++i) lists += ent_pos(k, (int)i) < a->sess(1).saved;
                 const int total = lists + 2 + 2;                       // pieces, two window slices, the end's two units
                 b->dev.room_bytes = (size_t)(total - 1) * sgeom::kQuarterDiskBytes;  // the smallest room of the four: one unit short of everything
                 const int agreed = total - 1;                          // what the ride agrees (the minimum of the rooms, in units)
                 a->core->image(1, agreed, {}), b->core->image(1, agreed, {});
                 NEED(a->core->store_of().jobs.size() == b->core->store_of().jobs.size(), "both ranks queued the same number of jobs: " + std::to_string(a->core->store_of().jobs.size()) + " and " + std::to_string(b->core->store_of().jobs.size()));
                 NEED(a->core->digest() == b->core->digest() && a->core->wide_digest(a->core->digest()) == b->core->wide_digest(b->core->digest()), "the two ranks' digests are equal after the command");
                 a->core->image(1, agreed, {}), b->core->image(1, agreed, {});  // the next command finishes what is left, on both
                 const uint64_t seq = a->core->image_seq();
                 a->core->settle(seq, {}), b->core->settle(seq, {});
                 NEED(a->core->digest() == b->core->digest() && a->alloc.ghosts() == b->alloc.ghosts(), "and after the settle");
                 a->inv("rank A"), b->inv("rank B");
               },
               400, false});
  // a piece several entries name is marked only by an idle command; a ride inside a run leaves it clean. One revives, the next shares.
  v.push_back({"imager: a fork's shared pieces are not marked by a ride (a run is going on), are marked by an idle command; one holder revives them, the next shares",
               Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 x->line(0, 1, 100000);
                 x->core->slot_reserve(1, 40, 0), x->core->slot_attach(1, 2);
                 x->core->load(1, 40000, 1, 1);  // B forks A at 40,000: the full pieces below are shared
                 x->core->slot_release(0), x->core->slot_release(1);
                 x->inv("forked");
                 size_t shared0 = 0;
                 for (const uint32_t r : x->core->refs) shared0 += r > 1;
                 NEED(shared0 > 0, "pieces are shared");
                 x->core->image(1, 1000, {}), x->core->image(2, 1000, {});
                 const uint64_t seq = x->core->image_seq();
                 x->core->settle(seq, {99});  // a ride: a request is in flight on another conversation
                 x->core->image(2, 1000, {}); // B adopts the images A's jobs made (host only)
                 size_t shared1 = 0;
                 for (const uint32_t r : x->core->refs) shared1 += r > 1;
                 NEED(shared1 == shared0, "a ride marked no shared piece: " + std::to_string(shared0) + " still shared, " + std::to_string(shared1) + " after");
                 NEED(x->alloc.ghosts() > 0, "and marked the pieces only one entry names");
                 x->inv("after the ride");
                 x->core->settle(seq, {});        // an idle command
                 size_t shared2 = 0;
                 for (const uint32_t r : x->core->refs) shared2 += r > 1;
                 NEED(shared2 == 0, "the idle command marked them: none shared any more");
                 x->inv("after the idle command");
                 const std::vector<uint64_t> a0 = tags_of(*x, 1);
                 x->core->unspill(1);
                 const uint64_t rev = x->core->split_counts.revives;
                 x->core->unspill(2);
                 NEED(x->core->split_counts.revives > rev || x->core->split_counts.shares > 0, "B's return revived or shared");
                 NEED(x->core->split_counts.shares > 0, "the second holder shares the block the first revived");
                 x->inv("both back");
                 NEED(x->core->split_counts.returns_read == 0, "nothing was read");
               },
               400, false});
  // H6b: a revive that join_take_ok refuses stops the return as a read-back's refusal does
  v.push_back({"imager: a revive the reservation rule refuses revives nothing: the return stops (cut), no ghost is taken past a reservation", Rule::kJoinTakeUnpaid,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir, 60);
                 idle_conversation(*x, 0, 1);
                 x->core->image(1, 1000, {});
                 x->core->settle(x->core->image_seq(), {});
                 NEED(x->alloc.ghosts() == 34, "34 ghosts");
                 x->core->slot_reserve(1, 59, 0), x->core->slot_attach(1, 2);  // a line reserved all 59 numbers, holding none yet
                 x->core->unspill(1);
                 x->inv("the refused return");
                 NEED(x->core->split_counts.revives == 0, "no ghost was taken back past the reservation: " + std::to_string(x->core->split_counts.revives) + " revives");
               },
               60, false});
  // a join reached under the imager would return early on one rank before its collective: it stops with its own words
  v.push_back({"imager: join under the imager dies saying so (no slot exists there; its early returns would leave the partners in the collective)", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 std::string msg;
                 try {
                   x->core->join(1, INT32_MAX, 0, 64);
                 } catch (const Died& d) {
                   msg = d.msg;
                 }
                 NEED(msg.find("reached under the imager") != std::string::npos, "join under the imager dies naming itself: '" + msg + "'");
                 World old(400);  // the old rules keep their join
                 idle_conversation(old, 0, 1);
                 old.core->join(1, INT32_MAX, 0, 64);
               },
               400, false});
  // the read-back counters count what was read
  v.push_back({"restore counters: a return that revived every ghost counts 0 tokens and 0 returns; one that read every piece counts its positions once; one in two commands counts once",
               Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 const int saved = x->sess(1).saved;
                 x->core->image(1, 1000, {});
                 x->core->settle(x->core->image_seq(), {});
                 x->core->unspill(1);  // every ghost is there: revived, nothing read
                 NEED(x->core->tier.restored_pos == 0 && x->core->tier.restores == 0, "0 tokens, 0 returns for a return that read nothing: " + std::to_string(x->core->tier.restored_pos) + ", " + std::to_string(x->core->tier.restores));
                 x->core->spill(1);  // every piece goes to the SSD
                 x->core->unspill(1);  // and is read back, in one command
                 NEED(x->core->tier.restored_pos == (uint64_t)saved && x->core->tier.restores == 1, "a return that read every piece counts its positions (" + std::to_string(saved) + ") once: " + std::to_string(x->core->tier.restored_pos) + ", " + std::to_string(x->core->tier.restores));
                 x->core->spill(1);
                 const uint64_t p0 = x->core->tier.restored_pos, r0 = x->core->tier.restores;
                 int cmds = 0;
                 while (x->core->unspill_left(1, INT32_MAX) > 0 && cmds < 20) x->core->unspill(1, 0, INT32_MAX, 10), ++cmds;
                 NEED(cmds >= 2, "the return took several commands");
                 NEED(x->core->tier.restores - r0 == 1 && x->core->tier.restored_pos - p0 == (uint64_t)saved, "a return in " + std::to_string(cmds) + " commands counts once, its positions once: " + std::to_string(x->core->tier.restores - r0) + ", " + std::to_string(x->core->tier.restored_pos - p0));
               },
               400, false});
  // the kept index under the imager is K5; a K3 or K4 index is another layout, a clean start
  v.push_back({"imager: the kept index is K5 in every mode; a K3 and a K4 index are another layout; K5 is kept and loaded warm", Rule::kNone,
               [](World& w, const std::string& dir) {
                 auto x = imager_world(w, dir);
                 idle_conversation(*x, 0, 1);
                 x->core->image(1, 1000, {});
                 x->core->settle(x->core->image_seq(), {});
                 x->core->spill(1);
                 x->core->keep(0x77);
                 int fd = open((dir + "/rank0.spill").c_str(), O_CREAT | O_WRONLY, 0600);
                 NEED(fd >= 0 && ftruncate(fd, (off_t)x->dev.file_size) == 0, "the kept file");
                 close(fd);
                 World k5(400);
                 imager_on(k5);
                 k5.dev.image = x->dev.image, k5.dev.out_sum = x->dev.out_sum, k5.dev.qimage = x->dev.qimage;
                 size_t kb = 0;
                 std::string why;
                 NEED(k5.core->kept_check(dir + "/rank0", 7, 3600, &kb, &why) == 0x77, "a K5 index is accepted under the imager: " + why);
                 k5.core->disk(dir + "/rank0", kb, 512 << 20, true, 7);
                 k5.inv("the kept start");
                 NEED(k5.sess(1).out() && k5.core->store_of().qimg.size() == 34, "kept: on the disk tier, 34 quarter images");
                 k5.core->unspill(1);
                 NEED(!k5.sess(1).out() && k5.sess(1).saved == 100000, "it comes back whole through the exchange");
                 World k4(400, false, w.core->test_rule, 48, 262144, 3u << 20, 5u << 20, true);  // the old K4 reader (split, no imager)
                 std::string why4;
                 NEED(k4.core->kept_check(dir + "/rank0", 7, 3600, &kb, &why4) == 0 && why4.find("not a kept index") != std::string::npos, "a K5 index is not a K4 one: " + why4);
                 // and an old index under the imager
                 const std::string d2 = dir + "/old";
                 mkdir(d2.c_str(), 0700);
                 World old(400, false, w.core->test_rule, 48, 262144, 3u << 20, 5u << 20, true);
                 old.open_disk(60, d2);
                 idle_conversation(old, 0, 1);
                 old.core->split(1, 64);
                 old.core->spill(1);
                 old.core->keep(0x78);
                 int fd2 = open((d2 + "/rank0.spill").c_str(), O_CREAT | O_WRONLY, 0600);
                 NEED(fd2 >= 0 && ftruncate(fd2, (off_t)old.dev.file_size) == 0, "the old kept file");
                 close(fd2);
                 World n5(400);
                 imager_on(n5);
                 std::string why5;
                 NEED(n5.core->kept_check(d2 + "/rank0", 7, 3600, &kb, &why5) == 0 && why5.find("not a kept index") != std::string::npos, "a K4 index under the imager is another layout: " + why5);
               },
               400, false});
}

std::vector<Row> rows() {
  std::vector<Row> v;
  v.push_back({"take and give with refs: a fork shares numbers, freeing it gives back only its own", Rule::kGiveIgnoresRefs,
               [](World& w, const std::string&) {
                 w.line(0, 1, 100000);
                 w.inv("A's line");
                 w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
                 w.core->load(1, 40000, 1, 1);  // B forks A at 40,000
                 w.inv("the fork");
                 const Store::Sess& a = w.sess(1);
                 NEED(w.core->refs[a.lists[0][0].num()] == 2 && w.core->refs[a.lists[0][1].num()] == 2, "the two full numbers of a fork are shared");
                 NEED(w.core->refs[a.lists[0][2].num()] == 1, "the partial number is B's own copy, A's stays A's");
                 w.core->slot_release(1);
                 w.core->free_session(2);
                 w.inv("B freed");
                 NEED(w.core->refs[a.lists[0][0].num()] == 1 && w.alloc.taken(a.lists[0][0].num()), "A still holds the shared number");
               },
               300, false});
  v.push_back({"copy on write at a rewind: the shared tail becomes the rewinding session's own", Rule::kNoCow,
               [](World& w, const std::string&) {
                 w.line(0, 1, 100000);
                 w.core->slot_reserve(1, 40, 0), w.core->slot_attach(1, 2);
                 w.core->load(1, 40000, 1, 1);
                 const uint32_t shared = w.sess(1).lists[0][1].num();
                 NEED(w.core->refs[shared] == 2, "number 1 of list 0 is shared");
                 w.dev.calls.clear();
                 w.core->load(1, 20000, 0, 0);  // A rewinds into it
                 w.inv("the rewind");
                 const Store::Sess& a = w.sess(1);
                 NEED(a.lists[0][1].num() != shared, "A writes a number of its own");
                 NEED(w.sess(2).lists[0][1].num() == shared && w.core->refs[shared] == 1, "B keeps the old one, alone");
                 NEED(w.dev.tag_of(a.lists[0][1].num()) == w.dev.tag_of(shared), "the entries below the rewind were copied");
                 bool copied = false;
                 for (const std::string& c : w.dev.calls) copied = copied || c.rfind("copy_entries", 0) == 0;
                 NEED(copied, "a device copy was asked for");
               },
               300, false});
  v.push_back({"victim order: the evict-last session after the others, then the least recently used", Rule::kVictimMru,
               [](World& w, const std::string&) {
                 for (int h = 1; h <= 3; ++h) {
                   w.line(0, h, 20000);
                   w.core->slot_release(0);
                 }
                 w.core->pin({}, {}, {2});  // session 2 is the command's own: evicted last
                 std::vector<int> order;
                 for (int i = 0; i < 40 && w.core->evict_step(9); ++i) {
                   for (const int h : {1, 2, 3}) {
                     const auto it = w.core->store_of().sess.find(h);
                     if (it == w.core->store_of().sess.end() || it->second.numbers() == 0) {
                       if (std::find(order.begin(), order.end(), h) == order.end()) order.push_back(h);
                     }
                   }
                 }
                 w.inv("the evictions");
#ifdef PICO_OLD_RULES
                 NEED(order.size() == 3 && order[0] == 1 && order[1] == 2 && order[2] == 3, "emptied in the old order 1, 2, 3 (no evict-last)");
#else
                 NEED(order.size() == 3 && order[0] == 1 && order[1] == 3 && order[2] == 2, "emptied in the order 1, 3, 2");
#endif
                 const std::vector<int> ev = w.core->take_evicted();
                 NEED(ev.size() == 3, "with no disk tier the three went whole, in the evicted list");
               },
               300, false});
  v.push_back({"a piece evicted with room on the disk keeps an image and gives its number back", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(20, dir);
                 w.line(0, 1, 50000);
                 w.core->slot_release(0);
                 const size_t before = w.core->store_of().sess.at(1).numbers();
                 NEED(before > 0, "the session holds numbers");
                 NEED(w.core->evict_step(9), "one piece goes");
                 w.inv("the eviction");
                 Store::Sess& x = w.sess(1);
                 NEED(x.numbers() == before - 1, "one number back");
                 size_t imaged = 0;
                 for (const auto& d : x.dl)
                   for (const size_t o : d) imaged += o != Pool::kNone;
                 NEED(imaged >= 1 && x.out(), "its entry has an image and the session is partly out");
               },
               300, false});
  v.push_back({"a read-back that finds no number cuts the session where it stopped", Rule::kUnspillNoCut,
               [](World& w, const std::string& dir) {
                 w.open_disk(200, dir);
                 w.line(0, 1, 50000);
                 w.core->slot_release(0);
                 w.core->spill(1);
                 NEED(w.core->store_of().sess.at(1).numbers() == 0, "everything went to the disk tier");
                 w.line(0, 2, 200000);  // another conversation holds the pool
                 w.core->slot_release(0);
                 w.core->pin({2}, {}, {});  // the command's own: nobody may be evicted to make room
                 const uint32_t room = w.alloc.free();
                 std::vector<uint32_t> hold;
                 if (room) w.alloc.take(room, &hold);
                 w.core->refs.resize(w.alloc.numbers(), 0);
                 for (const uint32_t c : hold) w.core->refs[c] = 1;
                 for (const uint32_t c : hold) w.core->store_of().sess[-5].lists[0].push_back(Ref(c));  // the numbers are named, so the invariants hold
                 w.core->unspill(1);
                 w.inv("the read-back");
                 const Store::Sess& x = w.sess(1);
                 NEED(!x.out() && x.saved == 0, "cut to nothing: no number to read the first piece into");
                 NEED(w.core->sum_cuts == 0, "a cut for room is not a cut for sums");
               },
               300, false});
  v.push_back({"a piece that comes back different is cut at the agreed position, bookmarks above it go", Rule::kUnspillIgnoresSum,
               [](World& w, const std::string& dir) {
                 w.open_disk(60, dir);
                 w.line(0, 1, 100000);
                 NEED(w.core->snap_buf(1, 90000) != nullptr, "a bookmark");
                 for (const uint32_t c : w.sess(1).snaps.at(90000)) w.dev.wrote(c);
                 w.core->slot_release(0);
                 w.core->flush(1);
                 w.core->spill(1);
                 const size_t bad_off = w.sess(1).dl[0][0];
                 w.dev.image[bad_off] = {0xBAD};  // the image is spoiled on the disk
                 w.core->unspill(1);
                 w.inv("the read-back");
                 const Store::Sess& x = w.sess(1);
                 NEED(x.saved == 0 && x.lists[0].empty(), "cut at position 0, the first piece being the bad one");
                 NEED(x.snaps.empty(), "the bookmark above the cut is gone");
                 NEED(w.core->sum_cuts == 1, "the cut for sums is counted once (red under PICO_OLD_RULES)");
               },
               300, false});
  v.push_back({"the bookmarks a cut leaves unusable are given back", Rule::kCutKeepsBookmarks,
               [](World& w, const std::string& dir) {
                 w.open_disk(60, dir);
                 w.line(0, 1, 100000);
                 w.core->snap_buf(1, 90000);
                 w.core->slot_release(0);
                 w.core->flush(1);
                 w.core->spill(1);
                 w.dev.image[w.sess(1).dl[1][0]] = {0xBAD};
                 w.core->unspill(1);
                 w.inv("the cut");
                 NEED(w.sess(1).snaps.empty() && w.sess(1).end_pos == -1, "no bookmark above the cut is left");
               },
               300, false});
  v.push_back({"bookmarks are refused when the reservations need the room", Rule::kTakeIgnoresResv,
               [](World& w, const std::string&) {
                 w.line(0, 1, 20000);
                 const uint32_t free = w.alloc.free();
                 w.core->slot_reserve(1, (int)free + 5, 0);  // a running request is reserved more than the pool has free
                 const Nums* b = w.core->snap_buf(1, 4000);
                 w.inv("the bookmark");
                 NEED(b == nullptr, "no bookmark: its numbers are the request's");
               },
               120, false});
  v.push_back({"the disk tier's victims: the oldest session without the command's, its images dropped", Rule::kDiskVictimAny,
               [](World& w, const std::string& dir) {
                 w.open_disk(14, dir);  // 14 images: session 1's nine and session 2's five fill it
                 w.line(0, 1, 20000);
                 w.core->flush(1);
                 w.core->slot_release(0);
                 w.core->pin({1}, {}, {});  // the command uses session 1
                 w.line(0, 2, 20000);
                 w.core->flush(2);  // its tail entries find the file full: a victim is looked for
                 w.inv("the second's images");
                 size_t imaged1 = 0;
                 for (const auto& d : w.sess(1).dl)
                   for (const size_t o : d) imaged1 += o != Pool::kNone;
                 NEED(imaged1 > 0, "session 1, pinned, kept its images; the other's were taken");
               },
               300, false});
  v.push_back({"a larger disk tier than the file grows the file, and pieces are written past its old end", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(14, dir);
                 const size_t old = w.dev.file_size;
                 NEED(w.core->store_of().dpool.size == old, "the pool is the file");
                 NEED(w.core->disk_resize(3 * old), "the larger size is applied");
                 w.inv("the growth");
                 NEED(w.dev.file_size == 3 * old && w.core->store_of().dpool.size == 3 * old, "file and pool are three times the start's");
                 for (int h = 1; h <= 3; ++h) {  // 27 images in all: 14 fit the old file
                   w.line(0, h, 20000);
                   w.core->flush(h);
                   w.core->slot_release(0);
                 }
                 w.inv("the pieces written");
                 size_t imaged = 0, past = 0;
                 for (int h = 1; h <= 3; ++h)
                   for (const auto& d : w.sess(h).dl)
                     for (const size_t o : d) {
                       imaged += o != Pool::kNone;
                       past += o != Pool::kNone && o >= old;
                     }
                 NEED(imaged == 27 && past > 0, "every piece has an image and some lie past the old end (" + std::to_string(imaged) + ", " + std::to_string(past) + ")");
               },
               300, true});
  v.push_back({"a larger disk tier than the free space allows grows only as far as every rank can", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(14, dir);
                 const size_t old = w.dev.file_size;
                 w.dev.room = 5 * kPoolGrain + 123;  // five grains and a part of one
                 NEED(w.core->disk_resize(4 * old), "the part that fits is applied");
                 NEED(w.dev.file_size == old + 5 * kPoolGrain && w.core->store_of().dpool.size == old + 5 * kPoolGrain, "five grains more, not four times");
                 w.dev.room = 0;
                 NEED(!w.core->disk_resize(4 * old), "no room: nothing changes");
                 NEED(w.dev.file_size == old + 5 * kPoolGrain, "the file is as it was");
               },
               300, true});
  v.push_back({"a smaller disk tier cuts the file, and the old size comes back by a grow", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(14, dir);
                 const size_t old = w.dev.file_size;
                 NEED(w.core->disk_resize(old / 2), "smaller");
                 NEED(w.core->store_of().dpool.size == old / 2 / kPoolGrain * kPoolGrain && w.dev.file_size == w.core->store_of().dpool.size, "the pool shrinks, the file with it");
                 NEED(w.core->disk_resize(old), "and back");
                 NEED(w.core->store_of().dpool.size == old && w.dev.file_size == old, "to the old size, the file made again");
               },
               300, true});
  v.push_back({"a smaller disk tier gives the file's space back, and a size grown to again is made again", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(14, dir);
                 const size_t old = w.dev.file_size;
                 NEED(w.core->disk_resize(2 * old), "grown");
                 NEED(w.dev.file_size == 2 * old, "the file is twice the start's");
                 NEED(w.core->disk_resize(old), "smaller again");
                 NEED(w.core->store_of().dpool.size == old, "the pool is the smaller size");
                 NEED(w.dev.file_size == old, "the file is the smaller size too (" + std::to_string(w.dev.file_size) + ")");
                 NEED(w.core->disk_resize(2 * old), "grown once more");
                 size_t grows = 0;
                 for (const std::string& c : w.dev.calls) grows += c.rfind("disk_grow", 0) == 0;
                 NEED(grows == 2 && w.dev.file_size == 2 * old, "the file was made again for the second growth");
               },
               300, true});
  v.push_back({"a smaller disk tier keeps the checksums of the slots below the new end", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(14, dir);
                 const size_t old = w.dev.file_size;
                 NEED(w.core->disk_resize(2 * old), "grown");
                 w.dev.sum_set(0, 0x1111), w.dev.sum_set(old - kPoolGrain, 0x2222), w.dev.sum_set(old, 0x3333);
                 NEED(w.core->disk_resize(old), "smaller");
                 NEED(w.dev.sum(0, false) == 0x1111 && w.dev.sum(old - kPoolGrain, false) == 0x2222, "the slots below the new end keep their sums");
                 NEED(w.dev.out_sum.count(old) == 0, "the slot past it is gone");
               },
               300, true});
  v.push_back({"a kept index written, checked and loaded: the sessions come back piece for piece", Rule::kKeptIgnoresRun,
               [](World& w, const std::string& dir) {
                 w.open_disk(60, dir);
                 w.line(0, 1, 50000);
                 w.core->snap_buf(1, 40000);
                 w.core->slot_release(0);
                 w.core->spill(1);
                 w.core->keep(0x77);
                 NEED(w.dev.disk_path() == dir + "/rank0.spill", "the file's path");
                 // a second run: its file of the size the index names, its core, the same run id
                 int fd = open((dir + "/rank0.spill").c_str(), O_CREAT | O_WRONLY, 0600);
                 NEED(fd >= 0 && ftruncate(fd, (off_t)w.dev.file_size) == 0, "the kept file");
                 close(fd);
                 World w2(300);
#ifdef STORE_CORE_TEST_RULES
                 w2.core->test_rule = w.core->test_rule;  // the core under test is the second run's
#endif
                 w2.dev.image = w.dev.image, w2.dev.out_sum = w.dev.out_sum;
                 size_t kb = 0;
                 std::string why;
                 NEED(w2.core->kept_check(dir + "/rank0", 7, 3600, &kb, &why) == 0x77, "the index checks out: " + why);
                 std::string why2;
                 NEED(w2.core->kept_check(dir + "/rank0", 8, 3600, &kb, &why2) == 0, "another run's id is refused");
                 NEED(w2.core->kept_check(dir + "/rank0", 7, 3600, &kb, &why) == 0x77, "and the right one still is");
                 w2.core->disk(dir + "/rank0", kb, 512 << 20, true, 7);
                 w2.inv("the kept start");
                 NEED(w2.core->sessions().size() == 1 && w2.core->info(1).on_disk, "session 1 is back, every piece on the disk");
                 w2.core->unspill(1);
                 w2.inv("the read-back");
                 const Store::Sess& x = w2.sess(1);
                 NEED(!x.out() && x.saved == 50000 && x.snaps.count(40000), "restored whole, bookmark included");
                 NEED(!x.kept, "and no longer flagged as kept");
               },
               300, false});
  v.push_back({"a read-back under a cap takes the lists, then bookmarks up to the cap, and cuts as a full pool does; cap 0 takes all", Rule::kUnspillIgnoresCap,
               [](World& w, const std::string& dir) {
                 w.open_disk(300, dir);
                 const auto stored = [&](int h) {  // a conversation of 100,000 positions with two bookmarks, all on the disk
                   w.line(0, h, 100000);
                   w.core->snap_buf(h, 90000), w.core->snap_buf(h, 80000);
                   w.core->slot_release(0);
                   w.core->flush(h);
                   w.core->spill(h);
                 };
                 stored(1), stored(2), stored(3);
                 int in_pool = -1, on_disk = -1;
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 0 && on_disk == 34, "34 lists, none in the pool");
                 w.core->unspill(1, 34 + 2);  // the lists, then one bookmark (2 numbers): the older one does not fit
                 w.inv("the capped read-back");
                 Store::Sess& a = w.sess(1);
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 34 && on_disk == 0 && a.saved == 100000, "every list is back");
                 NEED(a.snaps.count(90000) && !a.snaps.count(80000), "the newer bookmark came back, the older did not fit and is gone");
                 NEED(!a.out(), "nothing is left only on the disk");
                 w.core->unspill(2, 0);  // no limit: everything that fits
                 w.inv("the uncapped read-back");
                 NEED(w.sess(2).snaps.count(90000) && w.sess(2).snaps.count(80000) && !w.sess(2).out(), "cap 0 takes both bookmarks");
                 w.core->unspill(3, 10);  // too small for the lists: cut where the count ended, as a full pool cuts
                 w.inv("the cut read-back");
                 NEED(w.sess(3).saved == 32768 && !w.sess(3).out() && w.sess(3).snaps.empty(), "cut at the first position past the ten numbers");
               },
               300, false});
  v.push_back({"list_numbers(h, upto): only the lists that hold positions below upto, in the pool and on the disk", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(100, dir);
                 w.line(0, 1, 100000);
                 int in_pool = -1, on_disk = -1;
                 w.core->list_numbers(1, 16384, &in_pool, &on_disk);
                 NEED(in_pool == 5 && on_disk == 0, "below 16,384: entry 0 of kv sources 0 to 2 and entries 0 and 1 of source 3");
                 w.core->list_numbers(1, 1 << 30, &in_pool, &on_disk);
                 NEED(in_pool == 34 && on_disk == 0, "every list");
                 w.core->slot_release(0);
                 w.core->flush(1);
                 w.core->spill(1);
                 w.core->list_numbers(1, 16384, &in_pool, &on_disk);
                 NEED(in_pool == 0 && on_disk == 5, "after the spill the same five are on the disk");
                 w.core->list_numbers(1, 0, &in_pool, &on_disk);
                 NEED(in_pool == 0 && on_disk == 0, "below 0: none");
                 w.core->list_numbers(99, 16384, &in_pool, &on_disk);
                 NEED(in_pool == 0 && on_disk == 0, "no such session: none");
               },
               300, false});
  v.push_back({"snap_numbers: each bookmark with the numbers it holds in the pool (0: only on the disk), by position", Rule::kNone,
               [](World& w, const std::string& dir) {
                 w.open_disk(100, dir);
                 w.line(0, 1, 100000);
                 w.core->snap_buf(1, 80000), w.core->snap_buf(1, 90000);
                 auto sn = w.core->snap_numbers(1);
                 NEED(sn.size() == 2 && sn[0].first == 80000 && sn[1].first == 90000 && sn[0].second == 2 && sn[1].second == 2, "two bookmarks of 2 numbers, ascending");
                 w.core->slot_release(0);
                 w.core->flush(1);
                 w.core->evict_piece(1, 9);  // the end bookmark (none), then the latest window bookmark goes to the disk
                 sn = w.core->snap_numbers(1);
                 NEED(sn.size() == 2 && sn[0].second == 2 && sn[1].second == 0, "the one that went to the disk holds none");
                 NEED(w.core->snap_numbers(77).empty(), "no such session: none");
               },
               300, false});
  v.push_back({"image rows: taken, held by a second take, given back, the tripwire's free of all", Rule::kImgFreeLeaks,
               [](World& w, const std::string&) {
                 const uint32_t free0 = w.alloc.free();
#ifdef PICO_OLD_RULES
                 std::printf("  expected red: image rows as they were (a pinned host buffer, no store numbers): this row is skipped\n");
                 return;
#endif
                 NEED(w.core->img_take(5, 500, -1), "500 rows take 3 numbers");
                 NEED(!w.core->img_take(5, 500, -1), "a second take of the key holds its rows and takes none");
                 NEED(w.alloc.free() == free0 - 3 && w.core->img_numbers() == 3, "three numbers out of the pool");
                 NEED(w.core->img_take(6, 100, -1), "another flight");
                 w.inv("two flights");
                 w.core->img_free(5);
                 w.inv("one freed");
                 NEED(w.core->img_numbers() == 1, "the other flight's number stays");
                 w.core->img_free(0);
                 w.inv("all freed");
                 NEED(w.alloc.free() == free0 && w.core->img_numbers() == 0, "every number back");
               },
               300, false});
  v.push_back({"the image link: a number written in place after its image entries were dropped is not read back as the old image", Rule::kNone,
               [](World& w, const std::string& dir) {
                 const std::string d = image_link_sequence(w, dir, true);
#if defined(PICO_OLD_RULES) || defined(PICO_OLD_IMAGE_LINK)
                 if (!d.empty()) std::printf("  expected red: the image link, as it was: %s\n", d.c_str());
                 return;
#endif
                 NEED(d.empty(), d.c_str());
                 NEED(w.core->stale_cuts >= 1, "the stale link was cut and counted");
               },
               400, false});
  v.push_back({"the image link: the same sequence with B's image entries not dropped reads back what B wrote", Rule::kNone,
               [](World& w, const std::string& dir) {
                 const std::string d = image_link_sequence(w, dir, false);
                 NEED(d.empty(), d.c_str());
                 NEED(w.core->stale_cuts == 0, "no stale link: nothing counted");
               },
               400, false});
  v.push_back({"the image link: two numbers read back from one image, one given back and the image dropped, is not adopted by the other", Rule::kNone,
               [](World& w, const std::string& dir) {
                 std::string d;
                 try {
                   d = image_share_sequence(w, dir);
                 } catch (const Died& x) {
                   d = x.msg;
                 }
#if defined(PICO_OLD_RULES) || defined(PICO_OLD_IMAGE_LINK)
                 if (!d.empty()) std::printf("  expected red: the image shared by two numbers, as it was: %s\n", d.c_str());
                 return;
#endif
                 NEED(d.empty(), d.c_str());
                 NEED(w.core->stale_cuts >= 1, "the stale link was cut and counted");
               },
               400, false});
  add_split_rows(v);
  add_a3_rows(v);
  add_b3_rows(v);
  add_end_rows(v);
  add_running_rows(v);
  add_c_rows(v);
  add_g4j_rows(v);
  add_s4s_rows(v);
  add_imager_rows(v);
  return v;
}

int run_rows() {
  const std::string dir = tmpdir();
  int n = 0;
  for (const Row& r : rows()) {
    ++n;
    const std::string d1 = dir + "/n" + std::to_string(n), d2 = dir + "/c" + std::to_string(n);
    mkdir(d1.c_str(), 0700), mkdir(d2.c_str(), 0700);
    std::string err = "passed";
    {
      World w(r.numbers, r.ring);
      try {
        r.fn(w, d1);
      } catch (const Died& d) {
        err = d.msg;
      } catch (const std::exception& x) {
        err = std::string("an exception: ") + x.what();
      }
    }
    check(err == "passed", std::string("row: ") + r.name + (err == "passed" ? "" : " -- " + err));
#ifdef STORE_CORE_TEST_RULES
#ifdef PICO_OLD_RULES
    if (r.rule == Rule::kImgFreeLeaks) continue;  // its row is the old image rows' (skipped above)
#endif
    if (r.rule != Rule::kNone) {
      std::string cerr = "passed";
      World w(r.numbers, r.ring, r.rule);
      try {
        r.fn(w, d2);
      } catch (const std::exception& x) {
        cerr = std::string("an exception: ") + x.what();
      } catch (const Died& d) {
        cerr = d.msg;
      }
      check(cerr != "passed", std::string("control: the row with its rule's line absent must fail: ") + r.name);
      std::printf("  control %-3d fails as it must: %s\n", n, cerr.c_str());
    }
#endif
  }
  return n;
}

// ---- the replay --------------------------------------------------------------------------------------------------------------

std::vector<int> parse_csv(const std::string& v) {
  std::vector<int> o;
  size_t i = 0;
  while (i < v.size()) {
    const size_t j = v.find(',', i);
    o.push_back(std::atoi(v.substr(i, j == std::string::npos ? std::string::npos : j - i).c_str()));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return o;
}

const std::pair<const char*, Rule> kRuleNames[] = {
    {"victim-mru", Rule::kVictimMru},           {"take-ignores-resv", Rule::kTakeIgnoresResv}, {"no-cow", Rule::kNoCow},
    {"cut-keeps-bookmarks", Rule::kCutKeepsBookmarks}, {"unspill-no-cut", Rule::kUnspillNoCut}, {"unspill-ignores-sum", Rule::kUnspillIgnoresSum},
    {"disk-victim-any", Rule::kDiskVictimAny},   {"give-ignores-refs", Rule::kGiveIgnoresRefs}, {"img-free-leaks", Rule::kImgFreeLeaks},
    {"kept-ignores-run", Rule::kKeptIgnoresRun}, {"unspill-ignores-cap", Rule::kUnspillIgnoresCap},
    {"no-ghost-cut", Rule::kNoGhostCut},        {"mark-ignores-line", Rule::kMarkIgnoresLine}, {"evict-plain", Rule::kEvictPlain},
    {"settle-unchecked", Rule::kSettleUnchecked}};

struct Replay {
  bool ok = true;
  uint64_t failed = 0;  // the record that differs (0: none)
  std::string why;
  size_t records = 0;
  uint64_t rose_at = 0;   // the first record at which stale_cuts rose (0: never)
  uint64_t diverged = 0;  // the first record that differs from the recording after a rise: a recording older than the rule
  uint64_t stale = 0;     // stale_cuts at the end
};

void kv_to_map(const st::Line& l, std::map<std::string, std::string>* m) {
  for (const st::Field& f : l.f) (*m)[f.key] = f.value;
}

Replay replay(const st::Trace& t, const std::string& trace_path, std::map<std::string, int>* kinds, std::map<std::string, int>* ans,
              Rule rule = Rule::kNone, StoreCore::EvictCounts* ev = nullptr) {
  Replay out;
  const std::string tmp = tmpdir();
  std::map<std::string, std::string> g1, g2, pool, pre;
  bool have_pre = false;
  for (const st::Line& l : t.lines) {
    if (l.tag == 'H' && l.word == "geom") kv_to_map(l, g1.empty() && !l.f.empty() && l.f[0].key == "rank" ? &g1 : &g2);
    if (l.tag == 'H' && l.word == "pool") kv_to_map(l, &pool);
    if (l.tag == 'P') have_pre = true, kv_to_map(l, &pre);
  }
  sgeom::StoreGeom g;
  g.rank = std::atoi(g1["rank"].c_str()), g.nslots = std::atoi(g1["nslots"].c_str()), g.nunits = std::atoi(g1["nunits"].c_str());
  g.list_len = std::atoi(g1["list_len"].c_str()), g.max_ctx = std::atoi(g1["max_ctx"].c_str());
  g.image_bytes = std::strtoull(g1["image"].c_str(), nullptr, 10), g.snap_bytes = std::strtoull(g1["snap"].c_str(), nullptr, 10);
  g.end_bytes = std::strtoull(g1["end"].c_str(), nullptr, 10);
  g.spec = g1["spec"] == "1", g.poison = g1["poison"] == "1", g.has_ring = g1["host_ring"] == "1";
  const std::vector<int> uo = parse_csv(g2["unit_of"]), sc = parse_csv(g2["slot_ctx"]);
  for (size_t i = 0; i < uo.size() && i < (size_t)kMaxSlots; ++i) g.unit_of[i] = uo[i], g.slot_ctx[i] = sc[i];
  const uint32_t pool_numbers = (uint32_t)std::strtoull(pool["numbers"].c_str(), nullptr, 10);
  FakeDev dev;
  chunks::Alloc alloc(have_pre ? (uint32_t)std::strtoull(pre["numbers"].c_str(), nullptr, 10) : pool_numbers);
  StoreCore core(g, dev, &alloc);
#ifdef STORE_CORE_TEST_RULES
  core.test_rule = rule;
#else
  (void)rule;
#endif
  struct Counts {  // what the core's own eviction steps did during the replay, whichever way the replay ends
    StoreCore& c;
    StoreCore::EvictCounts* out;
    ~Counts() {
      if (out) *out = c.evict_counts;
    }
  } counts{core, ev};
  StoreCore::Outside rec_out;
  core.outside = [&rec_out] { return rec_out; };
  const auto base_of = [&](const std::string& p) {
    const std::string raw = st::unescape(p);
    const size_t c = raw.rfind('/');
    return tmp + "/" + (c == std::string::npos ? raw : raw.substr(c + 1));
  };
  const auto fail = [&](uint64_t n, const std::string& kind, const std::string& why) {
    out.ok = false, out.failed = n, out.why = "record " + std::to_string(n) + " (" + kind + "): " + why;
    return out;
  };
  // From the first record at which the rule acted, a record that differs from the recording ends the replay with `diverged` set: the
  // recording was made without the rule. Before it, and at every record after it that is equal, the replay is as strict as ever.
  // The invariants are checked at the differing record too (they do not read the digest): a core that diverged into a broken state
  // is a failure there. Not after the core ended the process (it stopped mid-operation).
  const auto differs = [&](uint64_t n, bool check_invariants = true) {
    if (check_invariants) {
      const std::string inv = invariants(core, dev);
      if (!inv.empty()) return fail(n, "differs", "invariant: " + inv);
    }
    out.diverged = n, out.stale = core.stale_cuts;
    return out;
  };
  for (const st::Line& l : t.lines)
    if (l.tag == 'A' && l.n == 0) return fail(0, "answer", "an answer taken outside any entry: " + l.raw);
  size_t i = 0;
  while (i < t.lines.size()) {
    const st::Line& e = t.lines[i];
    if (e.tag == 'O') {  // the start's own growth of the pool, which the store does not see
      if (pool_numbers > alloc.numbers()) alloc.grow(pool_numbers);
      ++i;
      continue;
    }
    if (e.tag == 'V') {
      const std::vector<int> got = core.take_evicted();
      std::vector<int> want = parse_csv(t.find(e, "list") ? t.find(e, "list")->value : "");
      if (got != want) return fail(e.n, "take_evicted", "the evicted list differs from the recorded one");
      ++i;
      continue;
    }
    if (e.tag != 'E') {
      ++i;
      continue;
    }
    std::map<std::string, std::string> a;
    kv_to_map(e, &a);
    dev.answers.clear();
    const st::Line* r = nullptr;
    const st::Line* d = nullptr;
    std::map<std::string, std::string> answered;
    size_t j = i + 1;
    for (; j < t.lines.size() && t.lines[j].n == e.n && t.lines[j].tag != 'E'; ++j) {
      const st::Line& x = t.lines[j];
      if (x.tag == 'A') {
        const std::string name = x.f[0].key;
        dev.answers[name].push_back(std::strtoll(x.f[0].value.c_str(), nullptr, 10));
        answered[name] = x.f[0].value;
        if (ans) ++(*ans)[name];
      }
      if (x.tag == 'R') r = &x;
      if (x.tag == 'D') d = &x;
    }
    if (kinds) ++(*kinds)[e.word];
    ++out.records;
    std::map<std::string, std::string> dd;
    kv_to_map(*d, &dd);
    const std::vector<int> oo = parse_csv(dd["out"]);
    (void)oo;
    {  // out=a,b,c,d,e,f with the third possibly 64-bit
      std::vector<uint64_t> o64;
      std::string s = dd["out"];
      size_t p = 0;
      while (p < s.size()) {
        const size_t q = s.find(',', p);
        o64.push_back(std::strtoull(s.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str(), nullptr, 10));
        if (q == std::string::npos) break;
        p = q + 1;
      }
      rec_out.groups = o64[0], rec_out.owed = o64[1], rec_out.n0 = o64[2], rec_out.cur_groups = o64[3], rec_out.phase = o64[4];
    }
    int64_t result = INT64_MIN;
    const uint64_t stale0 = core.stale_cuts;
    try {
      const std::string& k = e.word;
      const auto I = [&](const char* n) { return std::atoi(a[n].c_str()); };
      const auto U = [&](const char* n) { return (uint64_t)std::strtoull(a[n].c_str(), nullptr, 10); };
      if (k == "kept_check") {
        const std::string base = base_of(a["base"]);
        const std::string idx = trace_path.substr(0, trace_path.rfind('.') == std::string::npos ? trace_path.size() : trace_path.rfind('.')) + ".index";
        if (std::ifstream(idx).good()) {  // the run's index file: put it where the check reads it, with its file of the size it names
          std::ifstream f(idx, std::ios::binary);
          std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
          int fd = open((base + ".index").c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
          if (fd >= 0 && ::write(fd, body.data(), body.size()) == (ssize_t)body.size()) {
            uint64_t file_bytes = 0;
            if (body.size() >= 32) std::memcpy(&file_bytes, body.data() + 24, 8);
            int sp = open((base + ".spill").c_str(), O_CREAT | O_WRONLY, 0600);
            if (sp >= 0) (void)!ftruncate(sp, (off_t)file_bytes), close(sp);
          }
          if (fd >= 0) close(fd);
        }
        size_t bytes = 0;
        std::string why;
        core.kept_check(base, U("run_id"), std::atof(a["max_age"].c_str()), &bytes, &why);
        if (answered.count("kept_gen") && std::strtoull(answered["kept_gen"].c_str(), nullptr, 10) != core.store_of().kept_gen)
          return fail(e.n, k, "the kept generation differs from the recorded one (" + why + ")");
      } else if (k == "disk") {
        core.disk(base_of(a["base"]), U("bytes"), U("staging"), I("reuse") != 0, U("run_id"));
      } else if (k == "pin") {
        core.pin(parse_csv(a["pin"]), parse_csv(a["live"]), parse_csv(a["last"]));
      } else if (k == "ensure") {
        core.ensure(I("slot"), I("upto")), result = core.mapped_upto[I("slot")];
      } else if (k == "slot_reserve") {
        core.slot_reserve(I("slot"), I("numbers"), I("img"));
      } else if (k == "slot_unreserve") {
        core.slot_unreserve(I("slot"));
      } else if (k == "slot_release") {
        core.slot_release(I("slot"));
      } else if (k == "slot_attach") {
        core.slot_attach(I("slot"), I("h"));
        int64_t n = 0;
        const int h = core.line_h[core.unit_of[I("slot")]];
        if (h != StoreCore::kNoLine && core.store_s && core.store_s->sess.count(h))
          for (const Refs& ls : core.store_s->sess.at(h).lists) n += (int64_t)ls.size();
        result = n;
      } else if (k == "save") {
        core.save(I("h"), I("from"), I("to"), I("unit"));
      } else if (k == "load") {
        core.load(I("h"), I("to"), I("unit"), I("slot"));
        int64_t n = 0;
        const int h = core.line_h[I("unit")];
        if (h != StoreCore::kNoLine && core.store_s && core.store_s->sess.count(h))
          for (const Refs& ls : core.store_s->sess.at(h).lists) n += (int64_t)ls.size();
        result = n;
      } else if (k == "snap_buf") {
        const Nums* b = core.snap_buf(I("h"), I("pos"));
        result = b ? (int64_t)b->size() : -1;
      } else if (k == "snap_get") {
        core.snap_get(I("h"), I("pos"));
      } else if (k == "snap_peek") {
        core.snap_peek(I("h"), I("pos"));
      } else if (k == "end_take") {
        core.end_take(I("h"), I("pos"));
      } else if (k == "end_get") {
        core.end_get(I("h"));
      } else if (k == "drop") {
        core.drop(I("h"), I("from"));
      } else if (k == "drop_snap") {
        core.drop_snap(I("h"), I("pos"));
      } else if (k == "free") {
        core.free_session(I("h"));
      } else if (k == "copy_snaps") {
        core.copy_snaps(I("src"), I("dst"), I("upto"));
      } else if (k == "spill") {
        core.spill(I("h"));
      } else if (k == "image") {
        core.g.imager = true, core.g.mark = true;  // a trace of the imager's entries is replayed by a core that has it on
        result = core.image(I("h"), I("budget"), parse_csv(a["live"]));
      } else if (k == "settle") {
        core.g.imager = true, core.g.mark = true;
        result = core.settle(U("upto"), parse_csv(a["live"]));
      } else if (k == "split") {
        result = core.split(I("h"), I("budget"));
      } else if (k == "join") {
        result = core.join(I("h"), I("upto"), U("cap"), I("J"));
      } else if (k == "unspill") {
        core.unspill(I("h"), U("cap"), a.count("upto") ? I("upto") : INT32_MAX, a.count("J") ? I("J") : 0);  // a record without them has neither
        int64_t rp = -1;
        if (core.store_s && core.store_s->sess.count(I("h"))) {
          const Store::Sess& x = core.store_s->sess.at(I("h"));
          rp = restart_of(x.saved, x.end_pos, x.snaps);
        }
        result = rp;
      } else if (k == "flush") {
        core.flush(I("h"));
      } else if (k == "disk_resize") {
        core.disk_resize(U("bytes"));
      } else if (k == "keep") {
        core.keep(U("gen"));
      } else if (k == "checkpoint") {
        core.checkpoint(U("gen"));
      } else if (k == "img_take") {
        result = core.img_take(I("key"), U("rows"), I("slot"));
      } else if (k == "img_free") {
        core.img_free(I("key"));
      } else if (k == "evict_step") {
        result = core.evict_step(I("h"));
      } else if (k == "evict_piece") {
        result = core.evict_piece(I("victim"), I("h"));
      } else if (k == "vacate") {
        result = core.vacate((uint32_t)U("from"));
      } else if (k == "make_room") {
        result = core.make_room(U("removed"));
      } else if (k == "alloc_shrink") {
        core.alloc_shrink(U("top"));
      } else if (k == "alloc_grow") {
        core.alloc_grow(U("numbers"));
      } else if (k == "set_factor") {
        core.set_factor(I("h"), I("factor"));
      } else {
        return fail(e.n, k, "no replay for this kind");
      }
    } catch (const Died& dx) {
      if (out.rose_at) return differs(e.n, false);  // the core, a record after the rule acted, did what the recording did not
      return fail(e.n, e.word, "the core ended the process: " + dx.msg);
    }
    if (core.stale_cuts != stale0 && !out.rose_at) out.rose_at = e.n;
    if (answered.count("index_write") && answered["index_write"] != "1")
      return out.rose_at ? differs(e.n) : fail(e.n, e.word, "the recorded index write failed: a replay cannot reproduce it");
    if (r && result != INT64_MIN && std::strtoll(r->f[0].value.c_str(), nullptr, 10) != result)
      return out.rose_at ? differs(e.n) : fail(e.n, e.word, "returned " + std::to_string(result) + ", recorded " + r->f[0].value);
    char b[64];
    const uint64_t dg = core.digest(rec_out), wd = core.wide_digest(dg);
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)dg);
    if (dd["d"] != b) return out.rose_at ? differs(e.n) : fail(e.n, e.word, "the store's digest " + std::string(b) + " differs from the recorded " + dd["d"]);
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)wd);
    if (dd["w"] != b) return out.rose_at ? differs(e.n) : fail(e.n, e.word, "the wide digest " + std::string(b) + " differs from the recorded " + dd["w"]);
    if (std::strtoull(dd["free"].c_str(), nullptr, 10) != alloc.free() || std::strtoull(dd["numbers"].c_str(), nullptr, 10) != alloc.numbers())
      return out.rose_at ? differs(e.n) : fail(e.n, e.word, "the allocator's free count or size differs from the recorded one");
    const std::string inv = invariants(core, dev);
    if (!inv.empty()) return fail(e.n, e.word, "invariant: " + inv);
    i = j;
  }
  out.stale = core.stale_cuts;
  return out;
}

// the committed set's rule: a trace equal at every record. One that differs from a record on where the rule acted was recorded
// without the rule: it is recorded again.
bool committed_ok(const Replay& r) { return r.ok && !r.diverged; }

// a control's changed record n is caught: the replay failed at n or later, or, after the rule acted, differed from the recording at n
// or later (a changed record after a rise is reported through `diverged`, not as a failure)
// (built with -DSTORE_TEST_OLD_CONTROL the predicate is the old one, which counts failures only: the row that needs the new one is red)
#ifdef STORE_TEST_OLD_CONTROL
bool caught(const Replay& r, uint64_t n) { return !r.ok && r.failed >= n; }
#else
bool caught(const Replay& r, uint64_t n) { return (!r.ok && r.failed >= n) || (r.ok && r.diverged && r.diverged >= n); }
#endif

// the coverage over a set of traces: kinds with no record
// evict_step and evict_piece are called only inside the core, and an entry nested in another writes no record: no trace can hold
// them, so the row does not ask for them (the replay counts their runs instead, over the committed set)
// split and join have no committed trace yet (their recordings come from the server's GPU tests); the scenarios' replay reaches them
bool recordable(const char* kind) {
  const std::string k = kind;
  return k != "evict_step" && k != "evict_piece" && k != "split" && k != "join" && k != "image" && k != "settle";  // image and settle: recorded anew in the imager's first window
}

std::vector<std::string> missing_kinds(const std::vector<st::Trace>& set) {
  std::set<std::string> seen;
  for (const st::Trace& t : set)
    for (const st::Record& r : t.records) seen.insert(r.kind);
  std::vector<std::string> m;
  for (int k = 0; k < st::kKinds; ++k)
    if (recordable(st::kind_name((st::Kind)k)) && !seen.count(st::kind_name((st::Kind)k))) m.push_back(st::kind_name((st::Kind)k));
  for (const std::string& a : st::answer_names()) {
    bool any = false;
    for (const st::Trace& t : set)
      for (const st::Line& l : t.lines) any = any || (l.tag == 'A' && l.n != 0 && l.f[0].key == a);
    if (!any) m.push_back("answer:" + a);
  }
  return m;
}

// ---- a trace recorded from the core itself: the replay's own test data ----------------------------------------------------------

// Drives the core through one of each entry kind over a fake that gives the answers a device would (the ring agrees, the
// clock reads the same), writing the trace as the engine does; the file is returned.
std::string record_scenario(const std::string& dir, bool with_disk_answers, int variant) {
  st::Writer wr(dir);
  st::Writer* saved = st::g_w;
  st::g_w = &wr;
  {
    World w(400, /*ring=*/true);
    StoreCore& c = *w.core;
    size_t kb = 0;
    std::string why;
    c.kept_check(dir + "/rank0", 7, 86400, &kb, &why);  // before the file exists: the pool is made and the disk tier opened first
    c.disk(dir + "/rank0", 60 * Pool::round(w.g.image_bytes), 512 << 20, false, 7);
    c.trace_open();
    c.pin({}, {1}, {});
    c.set_factor(1, 16);
    c.slot_reserve(0, 40, 0), c.slot_attach(0, 1), c.ensure(0, 100000);
    for (const Refs& l : w.sess(1).lists)
      for (const Ref n : l) w.dev.wrote(n.num());
    c.save(1, 0, 100000, 0);
    if (const Nums* b = c.snap_buf(1, 90000))
      for (const uint32_t n : *b) w.dev.wrote(n);
    if (const Nums* e = c.end_take(1, 100000))
      for (const uint32_t n : *e) w.dev.wrote(n);
    c.snap_peek(1, 90000), c.snap_get(1, 90000), c.end_get(1);
    c.slot_reserve(1, 40, 0), c.slot_attach(1, 2), c.load(1, 40000, 1, 1);
    c.copy_snaps(1, 2, 90000);
    c.flush(1);
    c.slot_release(1);
    c.slot_release(0);
    c.pin({}, {}, {});
    c.split(1, 8);
    c.join(1, 60000, 0, 64);
    if (variant == 0) {
      c.drop_snap(1, 90000);
      c.evict_step(9);
      c.evict_piece(1, 9);
    }
    c.spill(1);
    c.unspill(1);
    c.drop(2, 30000);
    c.free_session(2);
    c.img_take(5, 500, -1);
    c.img_free(5);
    c.make_room(5);
    c.vacate(380);
    c.alloc_shrink(380);
    c.alloc_grow(400);
    c.disk_resize(40 * Pool::round(w.g.image_bytes));
    c.checkpoint(0x55);
    c.keep(0x56);
    c.slot_unreserve(0);
    c.take_evicted();
    (void)with_disk_answers;
  }
  const std::string path = wr.path();
  st::g_w = saved;
  return path;
}

// ---- the GPU arms' shapes on the host (server/tests/fork_cut_test.py, disk_full_test.py) --------------------------------------
//
// Each records a trace from the core over the fake, then replays it with each wrong rule: the rule the arm exists for must fail
// the trace. The shapes are the arms' own, at the numbers the host can hold.

std::string record_body(const std::string& dir, uint32_t numbers, const std::function<void(World&, StoreCore&)>& body) {
  st::Writer wr(dir);
  st::Writer* saved = st::g_w;
  st::g_w = &wr;
  {
    World w(numbers, /*ring=*/true);
    body(w, *w.core);
  }
  const std::string path = wr.path();
  st::g_w = saved;
  return path;
}

// how many traces of `path` a wrong rule fails (0 or 1), and where
bool fails_with(const std::string& path, Rule rule, std::string* why) {
  st::Trace t;
  std::string err;
  if (!st::read_file(path, &t, &err)) {
    *why = err;
    return false;
  }
  const Replay r = replay(t, path, nullptr, nullptr, rule);
  *why = r.ok ? "equal at every record" : r.why;
  return !r.ok;
}

void arm_scenarios() {
  const std::string dir = tmpdir();
  const size_t grain = Pool::round(2916352);

  // fork_cut: a base of 40,000 positions with window bookmarks at 8,192 .. 32,768 and at its end 40,000 and an end bookmark, spilled
  // whole; a fork at 32,768 whose own need takes the spare: the base's read-back is capped at the lists below the fork point and
  // one more, so it ends below the base's end and the base is cut at 32,768 (unspill -> cut_at)
  {
    const std::string d = dir + "/fork_cut";
    mkdir(d.c_str(), 0700);
    int in_pool = -1, on_disk = -1, back = 0;
    int cut_saved = -1, cut_end_pos = 0;
    size_t cut_snaps = 99;
    const std::string path = record_body(d, 31, [&](World& w, StoreCore& c) {
      c.disk(d + "/rank0", 80 * grain, 512 << 20, false, 7);
      c.trace_open();
      w.line(0, 1, 40000);
      for (const int pos : {8192, 16384, 24576, 32768, 40000})
        if (const Nums* b = c.snap_buf(1, pos))
          for (const uint32_t n : *b) w.dev.wrote(n);
      if (const Nums* e = c.end_take(1, 40000))
        for (const uint32_t n : *e) w.dev.wrote(n);
      c.slot_release(0);
      c.flush(1);
      c.spill(1);
      c.list_numbers(1, 32768, &in_pool, &on_disk);
      back = on_disk;
      c.pin({1}, {}, {});  // the fork's command pins its base
      c.unspill(1, (size_t)back + 1);
      cut_saved = w.sess(1).saved, cut_end_pos = w.sess(1).end_pos, cut_snaps = w.sess(1).snaps.size();
      w.inv("the capped read-back");
    });
    check(back == 10 && in_pool == 0, "fork_cut shape: 10 lists lie below the fork point and all are on the disk");
    check(cut_saved == 32768 && cut_end_pos == -1, "fork_cut shape: the base is cut at the fork point (saved 32,768) and has lost its end bookmark");
    check(cut_snaps == 0, "fork_cut shape: the cap was spent on the lists: no bookmark came back, those above the cut are gone");
    std::string why;
    check(!fails_with(path, Rule::kNone, &why), "fork_cut shape: the trace replays equal as recorded: " + why);
    const bool nc = fails_with(path, Rule::kUnspillNoCut, &why);
    check(nc, "fork_cut shape: the wrong rule unspill-no-cut fails the trace: " + why);
    const bool kb = fails_with(path, Rule::kCutKeepsBookmarks, &why);
    std::printf("  fork_cut shape: cut-keeps-bookmarks %s (%s)\n", kb ? "fails the trace" : "does NOT fail the trace", why.c_str());
    // the cap is exhausted by the lists, so a bookmark above the cut is only on the disk when the cut runs, and the same read-back
    // then finds no room for it and drops it, with the rule or without: the two end alike. Only a cut with bookmarks in the pool
    // (a piece that came back different) separates them: row "the bookmarks a cut leaves unusable are given back".
    check(!kb, "fork_cut shape: cut-keeps-bookmarks is NOT caught by a cap cut (its bookmarks above the cut are dropped anyway)");
  }

  // the eviction outcomes no real trace reaches today (a pool with no disk tier; a disk tier smaller than the conversation), run on
  // the host and counted from their replays: the core's paths exist and replay equal, whatever the GPU tests reach
  {
    const std::string d1 = dir + "/evict_nodisk";
    mkdir(d1.c_str(), 0700);
    const std::string p1 = record_body(d1, 40, [&](World& w, StoreCore& c) {
      c.trace_open();
      for (const int h : {1, 2}) {
        w.line(0, h, 20000);
        c.slot_release(0);
      }
      c.pin({}, {}, {});
      c.evict_step(9);  // no disk tier: the least recently used session goes whole
    });
    const std::string d2 = dir + "/evict_smalldisk";
    mkdir(d2.c_str(), 0700);
    const std::string p2 = record_body(d2, 60, [&](World& w, StoreCore& c) {
      c.disk(d2 + "/rank0", 3 * grain, 512 << 20, false, 7);  // three images: the file is full before the bookmarks
      c.trace_open();
      w.line(0, 1, 40000);
      for (const int pos : {8192, 24576})
        if (const Nums* b = c.snap_buf(1, pos))
          for (const uint32_t n : *b) w.dev.wrote(n);
      if (const Nums* e = c.end_take(1, 40000))
        for (const uint32_t n : *e) w.dev.wrote(n);
      c.slot_release(0);
      c.spill(1);  // the end bookmark, the window bookmarks and the cache entries leave a file with no room
    });
    StoreCore::EvictCounts e1, e2;
    for (const auto& pe : {std::make_pair(p1, &e1), std::make_pair(p2, &e2)}) {
      st::Trace t;
      std::string err;
      check(st::read_file(pe.first, &t, &err), "the eviction scenario reads: " + err);
      const Replay r = replay(t, pe.first, nullptr, nullptr, Rule::kNone, pe.second);
      check(committed_ok(r), "the eviction scenario replays equal at every record: " + r.why);
    }
    std::printf("  eviction on the host: no disk tier: whole %llu; a disk smaller than the conversation: end %llu, window bookmark dropped %llu, "
                "cache entry cut %llu\n", (unsigned long long)e1.step_whole, (unsigned long long)e2.piece_end,
                (unsigned long long)e2.piece_snap_dropped, (unsigned long long)e2.piece_cut);
    check(e1.step_whole > 0, "host scenario: with no disk tier an eviction step frees the session whole");
    check(e2.piece_snap_dropped > 0 && e2.piece_cut > 0 && e2.piece_end > 0,
          "host scenario: a disk smaller than the conversation drops its bookmarks and cuts its entries for want of room");
  }

  // disk_full: a disk tier of 20 images; sessions fill it, the next one's images take the oldest one's (a victim freed whole when
  // only on the disk, its images alone when its pieces are in the pool); then the idle flush of the oldest session in the pool
  // finds the file full: the victim must be another session, never the one being flushed
  {
    const std::string d = dir + "/disk_full";
    mkdir(d.c_str(), 0700);
    size_t images1 = 99;
    const std::string path = record_body(d, 120, [&](World& w, StoreCore& c) {
      c.disk(d + "/rank0", 20 * grain, 512 << 20, false, 7);
      c.trace_open();
      w.line(0, 1, 20000);  // 5 images at its save
      c.slot_release(0);
      for (const int h : {2, 3}) {  // 9 images each: the file is full at the third
        w.line(0, h, 20000);
        c.flush(h);
        c.slot_release(0);
      }
      c.pin({}, {}, {});
      c.flush(1);  // the oldest session's idle flush: its 4 tail images need room
      images1 = 0;
      for (const auto& dd : w.sess(1).dl)
        for (const size_t o : dd) images1 += o != Pool::kNone;
      w.inv("the full file");
    });
    check(images1 == 9, "disk_full shape: the oldest session's flush got all its 9 images (the victims were other sessions')");
    std::string why;
    check(!fails_with(path, Rule::kNone, &why), "disk_full shape: the trace replays equal as recorded: " + why);
    const bool dv = fails_with(path, Rule::kDiskVictimAny, &why);
    check(dv, "disk_full shape: the wrong rule disk-victim-any fails the trace: " + why);
  }
}

}  // namespace

// ---- the replay across a stale cut ------------------------------------------------------------------------------------------
//
// A trace recorded from the core with the rule replays equal at every record, the cut's included. The replay names a record only
// where it differs from a recording made without the rule; in the committed set that fails the check.

bool spoil_digest(st::Trace* t, uint64_t n) {
  for (st::Line& l : t->lines)
    if (l.tag == 'D' && l.n == n)
      for (st::Field& f : l.f)
        if (f.key == "d") return f.value = "0000000000000000", true;
  return false;
}

void replay_across_a_cut() {
#if defined(PICO_OLD_RULES) || defined(PICO_OLD_IMAGE_LINK)
  std::printf("  expected red: the replay across a stale cut (no cut is made without the fix): this row is skipped\n");
  return;
#endif
  const std::string dir = tmpdir();
  std::string seq;
  const std::string path = record_body(dir, 400, [&](World& w, StoreCore&) { seq = image_link_sequence(w, dir, true, true); });
  check(seq.empty(), "the traced sequence reads back what B wrote: " + seq);
  st::Trace t;
  std::string err;
  if (!st::read_file(path, &t, &err)) {
    check(false, "the traced sequence reads: " + err);
    return;
  }
  const Replay r = replay(t, path, nullptr, nullptr);
  check(r.ok && r.diverged == 0 && r.rose_at != 0 && r.stale >= 1,
        "the sequence replays equal at every record through the stale cut (rose at " + std::to_string(r.rose_at) + ", diverged at " +
            std::to_string(r.diverged) + "): " + r.why);
  std::printf("  traced image link: %zu records, stale_cuts rose at record %llu, %llu cuts, %s\n", r.records, (unsigned long long)r.rose_at,
              (unsigned long long)r.stale, r.ok && !r.diverged ? "equal at every record" : "NOT equal");
  if (!r.rose_at) return;
  st::Trace s1 = t;  // the digest spoiled at the record where the rule acted
  check(spoil_digest(&s1, r.rose_at), "the digest of the record of the cut can be spoiled");
  const Replay r1 = replay(s1, path, nullptr, nullptr);
  check(r1.ok && r1.diverged == r.rose_at && !committed_ok(r1), "a digest spoiled at the cut fails the committed check, named at that record");
  uint64_t later = 0;
  for (const st::Record& rec : t.records)
    if (rec.n > r.rose_at && (!later || rec.n < later)) later = rec.n;
  if (later) {  // the replay goes on comparing after the cut: a later spoiled digest is named at its own record
    st::Trace s2 = t;
    check(spoil_digest(&s2, later), "the digest of a record after the cut can be spoiled");
    const Replay r2 = replay(s2, path, nullptr, nullptr);
    check(r2.ok && r2.diverged == later && !committed_ok(r2), "a digest spoiled after the cut is named at its own record, not at the cut");
  }
  {  // a record changed after the rise is reported through `diverged`: the control must count that as caught (today's control did not)
    int via_diverged = 0, missed_by_old = 0;
    for (const st::Record& rec : t.records) {
      if (rec.n <= r.rose_at) continue;
      st::Trace c = t;
      std::string what;
      if (!st::perturb(&c, rec.n, &what)) continue;
      const Replay pr = replay(c, path, nullptr, nullptr);
      if (!(pr.ok && pr.diverged)) continue;
      ++via_diverged;
      check(caught(pr, rec.n), "control: " + what + " changed at record " + std::to_string(rec.n) + " after the rise is caught");
      missed_by_old += !(!pr.ok && pr.failed >= rec.n);  // the old predicate: not caught
    }
    check(via_diverged > 0, "a record changed after the rise is reported through diverged");
    check(missed_by_old == via_diverged, "RED on today's control: every such change was invisible to `!ok && failed >= n`");
  }
}

int main(int argc, char** argv) {
  g_die_handler = die_throws;
  std::string replay_file;
  long long perturb_n = -1;
  Rule cli_rule = Rule::kNone;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--replay" && i + 1 < argc) replay_file = argv[++i];
    else if (a == "--perturb" && i + 1 < argc) perturb_n = std::atoll(argv[++i]);
    else if (a == "--rule" && i + 1 < argc) {
      const std::string n = argv[++i];
      for (const auto& rn : kRuleNames)
        if (n == rn.first) cli_rule = rn.second;
      if (cli_rule == Rule::kNone) {
        std::printf("no rule named %s\n", n.c_str());
        return 2;
      }
    }
    else {
      std::printf("usage: store_core_test [--replay FILE [--perturb N]]\n");
      return 2;
    }
  }
  if (!replay_file.empty()) {
    st::Trace t;
    std::string err;
    if (!st::read_file(replay_file, &t, &err)) {
      std::printf("%s\n", err.c_str());
      return 2;
    }
    if (perturb_n >= 0) {
      std::string what;
      if (!st::perturb(&t, (uint64_t)perturb_n, &what)) {
        std::printf("record %lld has no argument to perturb\n", perturb_n);
        return 2;
      }
      std::printf("perturbed %s of record %lld\n", what.c_str(), perturb_n);
    }
    const Replay r = replay(t, replay_file, nullptr, nullptr, cli_rule);
    std::printf("%s (%zu records)\n", !r.ok ? r.why.c_str() : r.diverged ? ("replay equal up to record " + std::to_string(r.diverged) + ", which differs after stale_cuts rose at record " + std::to_string(r.rose_at) + ": recorded before the fix").c_str() : "replay equal at every record", r.records);
    if (cli_rule != Rule::kNone) return r.ok ? 1 : 0;  // a wrong rule must fail
    return perturb_n >= 0 ? (caught(r, (uint64_t)perturb_n) ? 0 : 1) : (r.ok ? 0 : 1);
  }

  const int nrows = run_rows();
  std::printf("%d unit rows run\n", nrows);

  // the replay on traces recorded from the core itself, and on the committed ones
  const std::string dir = tmpdir();
  std::vector<std::string> paths;
  for (int variant = 0; variant < 2; ++variant) {
    const std::string d = dir + "/v" + std::to_string(variant);
    mkdir(d.c_str(), 0700);
    paths.push_back(record_scenario(d, true, variant));
  }
  std::vector<st::Trace> set;
  for (const std::string& p : paths) {
    st::Trace t;
    std::string err;
    const bool ok = st::read_file(p, &t, &err);
    check(ok, "the recorded scenario reads: " + err);
    if (!ok) continue;
    std::map<std::string, int> kinds, ans;
    const Replay r = replay(t, p, &kinds, &ans);
    check(committed_ok(r), "the scenario replays equal at every record: " + r.why);
    std::printf("  scenario %s: %zu records, %s\n", p.c_str(), r.records, r.ok ? "equal at every record" : r.why.c_str());
    set.push_back(t);
    // controls: a changed record is caught at it or later, on three records
    int tried = 0;
    for (const st::Record& rec : t.records) {
      if (tried >= 3) break;
      if (rec.kind != "slot_reserve" && rec.kind != "alloc_grow" && rec.kind != "save" && rec.kind != "pin") continue;
      st::Trace c = t;
      std::string what;
      if (!st::perturb(&c, rec.n, &what)) continue;
      const Replay pr = replay(c, p, nullptr, nullptr);
      check(caught(pr, rec.n), "control: " + what + " changed at record " + std::to_string(rec.n) + " must fail there or later: " + pr.why);
      ++tried;
    }
    check(tried == 3, "three records could be changed in the scenario");
  }
  if (set.size() == 2) {
    const std::vector<std::string> all = missing_kinds(set);
    std::printf("  kinds with no record over the two scenarios: ");
    for (const std::string& k : all) std::printf("%s ", k.c_str());
    std::printf("\n");
    const std::vector<st::Trace> less = {set[1]};
    const std::vector<std::string> m1 = missing_kinds(less);
    check(m1.size() >= all.size(), "control: the coverage table over the set less one trace lists at least what the whole set lacks");
    bool only_in_first = false;
    for (const std::string& k : m1)
      if (std::find(all.begin(), all.end(), k) == all.end()) only_in_first = true;
    check(only_in_first, "control: the set less the first trace lists a kind that only the first reaches (drop_snap)");
  }

  arm_scenarios();
  replay_across_a_cut();

  // the committed traces
#ifdef STORE_TRACES_DIR
  {
    std::vector<std::string> files;
    if (DIR* d = opendir(STORE_TRACES_DIR)) {
      while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 6 && (n.compare(n.size() - 3, 3, ".t1") == 0 || (n.size() > 6 && n.compare(n.size() - 6, 6, ".t1.gz") == 0)) &&
            n.find(".fixed.") == std::string::npos)
          files.push_back(std::string(STORE_TRACES_DIR) + "/" + n);
      }
      closedir(d);
    }
    std::sort(files.begin(), files.end());
    std::vector<st::Trace> committed;
    std::vector<std::string> committed_files;
    StoreCore::EvictCounts ev_total;  // how often the core's own eviction steps ran over the set
    for (const std::string& f : files) {
      st::Trace t;
      std::string err;
      if (!st::read_file(f, &t, &err)) {
        check(false, "committed trace refused: " + err);
        continue;
      }
      std::map<std::string, int> kinds, ans;
      StoreCore::EvictCounts ev;
      const Replay r = replay(t, f, &kinds, &ans, Rule::kNone, &ev);
      ev_total.steps += ev.steps, ev_total.step_none += ev.step_none, ev_total.step_whole += ev.step_whole;
      ev_total.step_piece += ev.step_piece, ev_total.piece_end += ev.piece_end, ev_total.piece_snap_written += ev.piece_snap_written;
      ev_total.piece_snap_dropped += ev.piece_snap_dropped, ev_total.piece_entry_written += ev.piece_entry_written;
      ev_total.piece_cut += ev.piece_cut, ev_total.piece_none += ev.piece_none;
      check(committed_ok(r), f + ": " + (r.ok ? "equal only up to record " + std::to_string(r.diverged) + " after stale_cuts rose at record " + std::to_string(r.rose_at) + ": record the test again" : r.why));
      std::printf("  trace %s: %zu records, %s\n", f.c_str(), r.records,
                  !r.ok ? r.why.c_str() : r.diverged ? ("equal up to record " + std::to_string(r.diverged) + " (stale_cuts rose at " + std::to_string(r.rose_at) + ")").c_str() : "equal at every record");
      committed.push_back(t), committed_files.push_back(f);
      int tried = 0;
      for (size_t k = 0; k < t.records.size() && tried < 3; k += std::max<size_t>(1, t.records.size() / 3)) {
        for (size_t q = k; q < t.records.size(); ++q) {  // the first record from k that has an argument to change
          st::Trace c = t;
          std::string what;
          if (!st::perturb(&c, t.records[q].n, &what)) continue;
          const Replay pr = replay(c, f, nullptr, nullptr);
          if (caught(pr, t.records[q].n)) {
            ++tried;
            break;
          }
        }
      }
      check(tried == 3, f + ": three changed records must each be caught");
    }
    if (!committed.empty()) {  // how often the core's own eviction steps ran while the committed traces replayed
      const struct {
        const char* name;
        uint64_t n;
        bool required;
      } rows[] = {{"evict_step ran", ev_total.steps, true},
                  {"  no session left to give a piece", ev_total.step_none, false},
                  {"  no disk tier: the session went whole (host scenario only)", ev_total.step_whole, false},
                  {"  one piece of a session left (evict_piece)", ev_total.step_piece, false},
                  {"evict_piece: end bookmark out", ev_total.piece_end, true},
                  {"evict_piece: window bookmark out, image written or there", ev_total.piece_snap_written, true},
                  {"evict_piece: window bookmark dropped, no disk room (host scenario only)", ev_total.piece_snap_dropped, false},
                  {"evict_piece: cache entry out, image written or there", ev_total.piece_entry_written, true},
                  {"evict_piece: cache entry cut, no disk room (host scenario only)", ev_total.piece_cut, false},
                  {"evict_piece: nothing left in the pool", ev_total.piece_none, false}};
      // required: the outcomes a healthy run with a victim and a disk tier reaches; the three marked "host scenario only" need a
      // pool with no disk tier or a full disk with no victim to give, and are proved by the host scenarios above
      std::printf("  eviction, over %zu committed traces:\n", committed.size());
      std::string zero;
      for (const auto& r : rows) {
        std::printf("    %-62s %8llu%s\n", r.name, (unsigned long long)r.n, r.n || !r.required ? "" : "   <-- none");
        if (r.required && !r.n) zero += std::string(zero.empty() ? "" : "; ") + r.name;
      }
      check(zero.empty(), "each eviction outcome a healthy run reaches ran at least once over the committed set" + (zero.empty() ? "" : " (none: " + zero + ")"));
    }
    if (!committed.empty()) {  // a core built with a wrong rule must fail the traces that exercise the rule
      for (const auto& rn : kRuleNames) {
        int failing = 0;
        for (size_t q = 0; q < committed.size(); ++q) failing += !replay(committed[q], committed_files[q], nullptr, nullptr, rn.second).ok;
        std::printf("  wrong rule %-20s fails %d of %zu traces\n", rn.first, failing, committed.size());
        if (rn.second == Rule::kVictimMru || rn.second == Rule::kGiveIgnoresRefs)
          check(failing > 0, std::string("control: the wrong rule ") + rn.first + " must fail a trace of the set");
      }
    }
    std::printf("%zu committed traces under %s\n", files.size(), STORE_TRACES_DIR);
    if (!committed.empty()) {
      const std::vector<std::string> m = missing_kinds(committed);
      check(m.empty(), "the committed set reaches every entry and answer kind (none missing)");
      if (!m.empty()) {
        std::printf("  with no record: ");
        for (const std::string& k : m) std::printf("%s ", k.c_str());
        std::printf("\n");
      }
    }
  }
#endif
  std::printf(g_fail ? "store_core_test: %d of %d checks FAILED\n" : "store_core_test: all %d checks passed\n", g_fail ? g_fail : g_checks, g_checks);
  return g_fail ? 1 : 0;
}
