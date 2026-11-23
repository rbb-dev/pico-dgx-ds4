// The scheduler's side of the image encoder as decisions with no engine state (host only; tested by tests/vit_sched_test.cpp and,
// beside the interleaver, tests/interleave_test.cpp): what an image costs, which unit a pixel request does next (an encode piece or a
// piece of its rows), and the one-image-at-a-time mark.
//
// An image is encoded before the first row piece that reaches its span, in position order, and only when its span ends after the
// position the read starts from. An encode piece is the longest run of whole images whose time fits the budget (the gap less one
// decode step while sessions decode; unbounded when nobody decodes or the request is read whole), at least one image; an image that
// alone is longer than the budget goes in ranges of blocks that fit it. Between its first block and its last an image owns the
// work area: no other image of any request starts (Mark), and every way a request ends clears its mark.
//
// PICO_OLD_RULES (never defined by the engine) builds the rules as the container had them: the budget is ignored (every image a
// piece reaches is one run, however long the others wait) and a request that ends leaves its mark.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace vitsched {

constexpr int kBlocks = 32;  // the tower's blocks; a range of blocks costs its share of an image

// The time of an image on the four ranks over the fabric: k x blocks / 32 x (a + b n + c n^2) ms for n patches. The constants are
// the fit to five measured image sizes (32 blocks, engine stopped); k starts at 1 and follows
// the measured pieces (observe), as the interleaver's Cost::k does.
struct EncCost {
  static constexpr double kA = 4.24, kB = 0.021823, kC = 8.192e-7;
  double k = 1.0;
  double image_ms(int patches) const { return k * (kA + kB * patches + kC * (double)patches * patches); }
  double ms(int patches, int blocks) const { return image_ms(patches) * blocks / kBlocks; }
  // a piece that took `measured_ms` for `blocks` blocks of an image of `patches` patches
  void observe(double measured_ms, int patches, int blocks) {
    const double fit = ms(patches, blocks) / k;
    if (fit > 0 && measured_ms > 0) k = 0.8 * k + 0.2 * measured_ms / fit;
  }
};

// One image of a request: its span (first position, rows), its patches, and whether its rows are written.
struct Img {
  int start = 0, rows = 0, patches = 0;
  bool written = false;
};

// The device time of the images a read from position `pos` needs to encode: those not written whose span ends after pos.
inline double encode_ms(const std::vector<Img>& im, int pos, const EncCost& c) {
  double t = 0;
  for (const Img& i : im)
    if (!i.written && i.start + i.rows > pos) t += c.image_ms(i.patches);
  return t;
}

// The image in progress: one at a time across every request. `req` is the request's id, `image` its span index, `next` the next block.
struct Mark {
  uint64_t req = 0;
  int image = -1, next = 0;
  bool active() const { return image >= 0; }
  // may image `i` of request `r` run its block range starting at b0 now?
  bool may_run(uint64_t r, int i, int b0) const {
    if (!active()) return b0 == 0;
    return req == r && image == i && next == b0;
  }
  // a range [b0, b1) of image i of request r ran
  void ran(uint64_t r, int i, int b1) {
    if (b1 >= kBlocks) clear();
    else req = r, image = i, next = b1;
  }
  void clear() { req = 0, image = -1, next = 0; }
  // the request ended, however it ended (its DONE, a cancel, a client that left, an error, a queued request that never ran)
  void ended(uint64_t r) {
#ifndef PICO_OLD_RULES
    if (active() && req == r) clear();
#else
    (void)r;  // without the rule: the mark stays, and no other image starts
#endif
  }
};

// May a request start (or resume) now? A pixel request may not while an image of another request is between its blocks: its first
// encode would find the work area taken. It waits for that image's last block (not for the other request's whole read); a text
// request is never held by it.
inline bool may_start(const Mark& mark, bool pixel, uint64_t req) {
#ifdef PICO_OLD_RULES
  (void)mark, (void)pixel, (void)req;  // without the rule: it starts, and its first unit meets Hold
  return true;
#else
  return !pixel || !mark.active() || mark.req == req;
#endif
}

struct Unit {
  enum Kind { Rows, Encode, Hold } kind = Rows;
  int first = 0, count = 0;  // Encode: images [first, first + count)
  int b0 = 0, b1 = kBlocks;  // Encode of one image: its blocks [b0, b1); a run of several is whole images
};

// The blocks of image `patches`, from b0, that fit budget_ms: at least one, at most through the last.
inline int fit_blocks(const EncCost& c, int patches, int b0, double budget_ms) {
  const double per = c.ms(patches, 1);
  const int n = per > 0 && budget_ms < std::numeric_limits<double>::max() / 2 ? (int)(budget_ms / per) : kBlocks;
  return std::min(kBlocks, b0 + std::max(1, n));
}

// The next unit of request `req` whose next row piece is [pos, pos + rows): Rows when no image that piece reaches is unwritten;
// Encode when one is (the images the piece reaches, in order, as many as fit budget_ms, or a range of blocks of one); Hold when the
// image in progress belongs to another request (the scheduler never asks: two image requests do not read together).
inline Unit next_unit(const std::vector<Img>& im, int pos, int rows, double budget_ms, const EncCost& c, const Mark& mark, uint64_t req) {
  Unit u;
#ifdef PICO_OLD_RULES
  budget_ms = std::numeric_limits<double>::max();  // without the rule: the images of a piece are one run, however long
#endif
  if (mark.active()) {  // the image in progress goes on, whatever the piece
    if (mark.req != req) return u.kind = Unit::Hold, u;
    const Img& i = im[(size_t)mark.image];
    u.kind = Unit::Encode, u.first = mark.image, u.count = 1, u.b0 = mark.next, u.b1 = fit_blocks(c, i.patches, mark.next, budget_ms);
    return u;
  }
  int f = -1;
  for (size_t i = 0; i < im.size(); ++i)
    if (!im[i].written && im[i].start + im[i].rows > pos) {
      f = (int)i;
      break;
    }
  if (f < 0 || im[(size_t)f].start >= pos + rows) return u;  // Rows
  u.kind = Unit::Encode, u.first = f, u.count = 1;
  if (c.image_ms(im[(size_t)f].patches) > budget_ms) {  // longer than the room: a range of its blocks
    u.b0 = 0, u.b1 = fit_blocks(c, im[(size_t)f].patches, 0, budget_ms);
    return u;
  }
  double t = c.image_ms(im[(size_t)f].patches);
  for (size_t i = (size_t)f + 1; i < im.size() && !im[i].written && im[i].start < pos + rows; ++i) {
    const double m = c.image_ms(im[i].patches);
    if (t + m > budget_ms) break;
    t += m, ++u.count;
  }
  return u;
}

}  // namespace vitsched
