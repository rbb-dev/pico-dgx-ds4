// server/vit_sched.h: what an image costs on the four ranks (EncCost), the next unit of a pixel request (an encode piece, a range of one
// image's blocks, or the rows), the one-image-at-a-time mark and the rule that every way a request ends clears it.
// Built twice: with -DPICO_OLD_RULES (the encode of every image a piece reaches is one run however long the decoders wait; a request that
// ends leaves its mark) the cases marked "red" fail; without it every case passes. Host only: no engine, no GPU.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../interleave.h"
#include "../vit_sched.h"

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}
bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

using vitsched::EncCost;
using vitsched::Img;
using vitsched::Mark;
using vitsched::Unit;

constexpr int kPatches768 = 3025;  // 55 x 55 patches: a 768 x 768 image
constexpr int kRows768 = 382;      // its span: 19 x 20 + 2
constexpr int kPatchesBig = 9189;  // the largest image of the set

std::vector<Img> clip(int n, int patches = kPatches768, int rows = kRows768) {
  std::vector<Img> v;
  for (int i = 0; i < n; ++i) v.push_back({10 + i * (rows + 4), rows, patches, false});
  return v;
}

// Everything one request does to read `pos .. end`: the units next_unit gives with the mark and the written flags kept as the server
// keeps them. Returns the encode units' times (ms, by the cost) and whether every image's blocks were run once, in order.
struct Walk {
  std::vector<double> enc_ms;
  std::vector<int> enc_images;
  int rows_units = 0;
  bool blocks_ok = true;
  int units = 0;
};
Walk walk(std::vector<Img> im, int piece, double budget, const EncCost& c, Mark& mark, uint64_t req = 1, int start = 0) {
  Walk w;
  const int end = im.back().start + im.back().rows + 4;
  std::vector<int> next_block(im.size(), 0);
  for (int pos = start; pos < end && w.units < 10000;) {
    const Unit u = vitsched::next_unit(im, pos, piece, budget, c, mark, req);
    ++w.units;
    if (u.kind == Unit::Rows) {
      pos += piece, ++w.rows_units;
      continue;
    }
    if (u.kind == Unit::Hold) {
      w.blocks_ok = false;
      break;
    }
    double ms = 0;
    for (int k = 0; k < u.count; ++k) {
      const int i = u.first + k, b0 = u.count == 1 ? u.b0 : 0, b1 = u.count == 1 ? u.b1 : vitsched::kBlocks;
      if (!mark.may_run(req, i, b0) || next_block[(size_t)i] != b0) w.blocks_ok = false;
      ms += c.ms(im[(size_t)i].patches, b1 - b0);
      next_block[(size_t)i] = b1;
      mark.ran(req, i, b1);
      if (b1 == vitsched::kBlocks) im[(size_t)i].written = true;
      w.enc_images.push_back(i);
    }
    w.enc_ms.push_back(ms);
  }
  for (size_t i = 0; i < im.size(); ++i)
    if (im[i].start + im[i].rows > start && next_block[i] != vitsched::kBlocks) w.blocks_ok = false;
  return w;
}
double longest(const Walk& w) {
  double m = 0;
  for (double x : w.enc_ms) m = std::max(m, x);
  return m;
}
}  // namespace

int main() {
  const EncCost c;

  // ---- EncCost: the fit to F3a's five measured sizes ----
  check(near(c.image_ms(kPatches768), 77.75, 0.1), "EncCost: a 768 x 768 image (3,025 patches) takes 77.75 ms");
  check(near(c.image_ms(kPatchesBig), 273.9, 0.2), "EncCost: the largest image (9,189 patches) takes 273.9 ms");
  check(near(c.ms(kPatches768, 16), c.image_ms(kPatches768) / 2, 1e-9) && near(c.ms(kPatches768, 1) * 32, c.image_ms(kPatches768), 1e-9),
        "EncCost: a range of blocks costs its share of 32");
  {
    EncCost k;
    for (int i = 0; i < 40; ++i) k.observe(2.0 * c.ms(kPatches768, 8), kPatches768, 8);
    check(near(k.k, 2.0, 0.01), "EncCost: k follows pieces that take twice the fit");
    EncCost z;
    z.observe(0, kPatches768, 8);
    check(z.k == 1.0, "EncCost: a piece with no time leaves k alone");
  }
  {
    const std::vector<Img> im = clip(3);
    check(near(vitsched::encode_ms(im, 0, c), 3 * c.image_ms(kPatches768), 1e-9), "encode_ms: three unwritten images from position 0");
    check(near(vitsched::encode_ms(im, im[1].start + im[1].rows, c), c.image_ms(kPatches768), 1e-9),
          "encode_ms: images whose spans end by the restart position are not needed");
    check(near(vitsched::encode_ms(im, im[1].start + 5, c), 2 * c.image_ms(kPatches768), 1e-9), "encode_ms: a restart inside a span still needs that image");
    std::vector<Img> w = im;
    w[0].written = true;
    check(near(vitsched::encode_ms(w, 0, c), 2 * c.image_ms(kPatches768), 1e-9), "encode_ms: a written image costs nothing");
  }

  // ---- the unit choice ----
  Mark none;
  {
    const std::vector<Img> im = clip(2);
    Unit u = vitsched::next_unit({}, 0, 1000, 500, c, none, 1);
    check(u.kind == Unit::Rows, "unit: a text request has only rows");
    u = vitsched::next_unit(im, 0, 5, 500, c, none, 1);
    check(u.kind == Unit::Rows, "unit: a piece that ends before the first image's span is rows");
    u = vitsched::next_unit(im, 0, im[0].start + 1, 500, c, none, 1);
    check(u.kind == Unit::Encode && u.first == 0 && u.count == 1 && u.b0 == 0 && u.b1 == 32, "unit: a piece that reaches an image encodes it first, whole");
    u = vitsched::next_unit(im, im[0].start + im[0].rows, 2, 500, c, none, 1);
    check(u.kind == Unit::Rows, "unit: an image whose span ends at the restart position is skipped");
    u = vitsched::next_unit(im, im[0].start + 10, 100, 500, c, none, 1);
    check(u.kind == Unit::Encode && u.first == 0, "unit: a restart inside a span encodes that image");
    std::vector<Img> w = im;
    w[0].written = true;
    u = vitsched::next_unit(w, 0, im[0].start + im[0].rows + 1, 500, c, none, 1);
    check(u.kind == Unit::Rows, "unit: a written image is not encoded again, nor is a later one the piece does not reach");
    u = vitsched::next_unit(w, 0, 5000, 500, c, none, 1);
    check(u.kind == Unit::Encode && u.first == 1, "unit: a resume that moved back encodes the image it never encoded");
  }
  {  // a run of whole images fits the budget: six 768 x 768 images in 470 ms (gap 600 less a 130 ms step), the seventh does not
    const std::vector<Img> im = clip(32);
    const Unit u = vitsched::next_unit(im, 0, 12224, 470, c, none, 1);
    check(u.kind == Unit::Encode && u.first == 0 && u.count == 6, "budget: six 768 x 768 images fit 470 ms (red on the old rules: every image the piece reaches)");
    check(u.count <= 6, "budget: never more than fit (red on the old rules)");
  }
  {
    Mark m;
    const Walk w = walk(clip(32), 12224, 470, c, m);
    check(w.blocks_ok && w.enc_images.size() == 32, "a 32-image clip: every image encoded once, whole, in order");
    check(longest(w) <= 470 + 1e-9, "a 32-image clip: no encode piece above the budget (red on the old rules: one run of 2.5 s)");
    check(w.enc_ms.size() >= 6, "a 32-image clip: the encode is in at least six pieces (red on the old rules)");
    check(!m.active(), "a 32-image clip: no image is left between its blocks");
  }
  {  // an image larger than the room goes in ranges of blocks
    Mark m;
    const Walk w = walk(clip(1, kPatchesBig, 1024), 12224, 100, c, m);
    check(w.blocks_ok, "an image larger than the room: its ranges are contiguous from block 0 to 32, each run once");
    check(w.enc_ms.size() == 3, "an image larger than the room: 274 ms in 100 ms of room is three ranges (red on the old rules: one)");
    check(longest(w) <= 100 + 1e-9, "an image larger than the room: no range above the room (red on the old rules)");
    Mark m2;
    const Walk w2 = walk(clip(1, kPatchesBig, 1024), 12224, 1e-3, c, m2);
    check(w2.blocks_ok && w2.enc_ms.size() == 32, "a room under one block's time: one block at a time, never zero (red on the old rules)");
  }
  {  // the single 768 x 768 image is one run
    Mark m;
    const Walk w = walk(clip(1), 12224, 470, c, m);
    check(w.enc_ms.size() == 1 && w.blocks_ok, "a single 768 x 768 image is one encode run");
  }
  {  // the image in progress goes on, whatever the piece
    Mark m;
    m.ran(1, 0, 11);
    const std::vector<Img> im = clip(3);
    Unit u = vitsched::next_unit(im, 0, 10, 1000, c, m, 1);
    check(u.kind == Unit::Encode && u.first == 0 && u.b0 == 11 && u.b1 == 32, "in progress: the next blocks of the same image, though the piece reaches none");
    u = vitsched::next_unit(im, 0, 5000, 1000, c, m, 2);
    check(u.kind == Unit::Hold, "in progress: another request's image waits (Hold)");
  }

  // ---- one image at a time, and every way a request ends clears the mark ----
  {
    Mark m;
    check(m.may_run(1, 0, 0) && !m.may_run(1, 0, 5), "mark: with none in progress an image starts at block 0 only");
    m.ran(1, 0, 11);
    check(m.active() && m.req == 1 && m.image == 0 && m.next == 11, "mark: after a range that does not end at block 32 the image is in progress");
    check(m.may_run(1, 0, 11), "mark: the same image's next range may run");
    check(!m.may_run(1, 1, 0) && !m.may_run(2, 0, 0) && !m.may_run(1, 0, 0) && !m.may_run(1, 0, 12),
          "mark: no other image, no other request, no restart and no skipped block while one is in progress");
    m.ran(1, 0, 32);
    check(!m.active() && m.may_run(2, 0, 0), "mark: the last block ends it");
  }
  const char* ends[] = {"finish (stop or length)", "the cancel sweep", "a client that left", "an error", "a queued request that never ran"};
  for (const char* how : ends) {
    Mark m;
    m.ran(7, 3, 11);  // request 7 ends between two ranges of its image 3
    m.ended(7);
    char what[160];
    std::snprintf(what, sizeof what, "ended by %s: the mark is cleared and the next request's image starts at block 0 (red on the old rules)", how);
    check(!m.active() && m.may_run(8, 0, 0), what);
  }
  {
    Mark m;
    m.ran(7, 3, 11);
    m.ended(8);
    check(m.active() && m.req == 7, "another request's end leaves the mark alone");
    Mark idle;
    idle.ended(7);
    check(!idle.active(), "an end with nothing in progress changes nothing");
  }

  // ---- il::Edge: the run edge, measured ----
  {
    il::Edge e;
    check(near(e.value(100), 250, 1e-9), "Edge: unset it is 2.5 x the step (red on the old rules: one step)");
    e.observe(700, 400);  // a unit of 400 ms, 700 ms of silence across it: an edge of 300
    check(near(e.value(100), 300, 1e-9), "Edge: the first measurement is the edge itself (red on the old rules)");
    e.observe(700, 400);
    e.observe(1000, 400);
    check(near(e.value(100), 0.8 * 300 + 0.2 * 600, 1e-9), "Edge: a moving average, 0.2 of the newest (red on the old rules)");
    il::Edge s;
    s.observe(450, 400);
    check(near(s.value(100), 100, 1e-9), "Edge: never under one step");
    il::Edge z;
    z.observe(300, 400);  // a silence under the unit's time (clock noise): an edge of 0, not negative
    check(z.ms >= 0 && z.value(100) >= 100, "Edge: a silence under the unit's time does not go negative, and the value is a step or more");
    // the encode unit's budget is the gap less the edge: 600 - 300 = 300 ms: three 768 x 768 images (77.75 ms each), not five
    const std::vector<Img> im = clip(32);
    const Unit u = vitsched::next_unit(im, 0, 12224, 600 - e.value(100) > 0 ? 600 - e.value(100) : 1, c, none, 1);
    check(u.kind == Unit::Encode && u.count == 3, "Edge: with an edge of 360 ms a unit is three images (red on the old rules: four or more at a gap less one step)");
  }

  // ---- a pixel request is neither started nor resumed while another request's image is between its blocks ----
  {
    Mark m;
    check(vitsched::may_start(m, true, 2), "may_start: nothing in progress: a pixel request starts");
    m.ran(1, 0, 11);
    check(!vitsched::may_start(m, true, 2), "may_start: another request's image between its blocks: a pixel request waits (red on the old rules: it starts and meets Hold)");
    check(vitsched::may_start(m, true, 1), "may_start: the request whose image it is goes on");
    check(vitsched::may_start(m, false, 2), "may_start: a text request is never held by it");
    m.ran(1, 0, 32);
    check(vitsched::may_start(m, true, 2), "may_start: after the image's last block the waiting request starts");
    for (const char* how : ends) {
      Mark e;
      e.ran(7, 3, 11);
      e.ended(7);
      char what[160];
      std::snprintf(what, sizeof what, "may_start: the request ended by %s: the waiting pixel request starts (the mark is cleared)", how);
      check(vitsched::may_start(e, true, 8), what);
    }
    Mark h;
    h.ran(1, 0, 11);
    const std::vector<Img> im = clip(2);
    check(!vitsched::may_start(h, true, 2) && vitsched::next_unit(im, 0, 5000, 1000, c, h, 2).kind == Unit::Hold,
          "may_start: the request kept from starting is the one whose first unit would be Hold, the server's assert (red on the old rules)");
  }

  std::printf(fails ? "vit_sched_test FAILED (%d)\n" : "vit_sched_test passed\n", fails);
  return fails ? 1 : 0;
}
