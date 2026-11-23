// server/pixels.h: a SUBMIT that carries its images as pixels (fe::kFlagPixels). The layout (spans, the grids padded to 8 bytes, each
// image's 588 x n_h x n_w bytes padded to 8), both flags together, n_images 0 or over kMaxImages, a byte count that is not exactly the
// layout's (malformed: the connection closes), a grid that does not match its span's rows, rows over 1,024, patches over kMaxPatches
// (refused: DONE and a message, the connection lives), and the head's reader (frames.h) on a connection of either version with and
// without kFlagPrefix. The vectors are the frame's contract for the front-end that sends it.
// Built twice: with -DPICO_OLD_RULES (the reader that does not know the flag: a pixel frame is malformed) the cases marked "red"
// fail; without it every case passes. Host only: no engine, no GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../frames.h"
#include "../pixels.h"

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

struct Img {
  uint32_t start;
  int64_t rows;  // -1: the rows the grid makes
  uint16_t n_h, n_w;
};

template <class T>
void put(std::vector<char>& b, const T& v) {
  b.insert(b.end(), (const char*)&v, (const char*)&v + sizeof v);
}

size_t pad8(size_t n) { return (n + 7) & ~size_t(7); }

// A pixel SUBMIT's payload: header, [SubmitPrefix], stop ids, prompt ids, then the image part. `ni` is the count written (it need not
// be the number of images); `trim`: bytes cut from the end (positive) or zeros added (negative).
std::vector<char> frame(uint32_t flags, bool with_prefix, const std::vector<Img>& imgs, int64_t ni = -1, int trim = 0, uint32_t n_prompt = 40) {
  fe::SubmitHeader h{};
  h.req_id = 7, h.session_id = 3, h.flags = flags | (with_prefix ? fe::kFlagPrefix : 0), h.n_stop_ids = 1, h.n_prompt = n_prompt;
  std::vector<char> p;
  put(p, h);
  if (with_prefix) put(p, fe::SubmitPrefix{0xbeefull, 12, 0});
  p.resize(p.size() + 4 * (1 + n_prompt), 1);
  put(p, uint32_t(ni < 0 ? imgs.size() : ni));
  for (const Img& i : imgs) {
    const uint64_t rows = i.rows < 0 ? pixels::rows_of({i.n_h, i.n_w}) : uint64_t(i.rows);
    put(p, fe::ImageSpan{i.start, uint32_t(rows), 0x1111ull * (&i - imgs.data() + 1)});
  }
  std::vector<char> g;
  for (const Img& i : imgs) put(g, pixels::Grid{i.n_h, i.n_w});
  g.resize(pad8(g.size()), 0);
  p.insert(p.end(), g.begin(), g.end());
  for (const Img& i : imgs) {
    std::vector<char> px(pad8(size_t(588) * i.n_h * i.n_w), char(&i - imgs.data() + 1));
    p.insert(p.end(), px.begin(), px.end());
  }
  if (trim > 0) p.resize(p.size() - size_t(trim));
  if (trim < 0) p.resize(p.size() + size_t(-trim), 0);
  return p;
}

// The reader's two steps on a connection of `ver`: the head, then the image part.
pixels::Verdict read(const std::vector<char>& p, uint32_t ver, pixels::Parsed* out = nullptr, std::string* why = nullptr) {
  fe::SubmitHeader h;
  fe::SubmitPrefix px;
  size_t base = 0;
  if (!frames::parse_submit_head(p.data(), p.size(), ver, &h, &px, &base)) return pixels::Verdict::kMalformed;
  const pixels::Result r = pixels::parse(p.data(), p.size(), base, h.flags);
  if (out) *out = r.px;
  if (why) *why = r.why;
  return r.v;
}
}  // namespace

int main() {
  using V = pixels::Verdict;
  const std::vector<uint32_t> vers{fe::kVersion, fe::kVersionPrefix};

  // the layout: three images; 3 x 3 patches (rows 1 x 2 + 2 = 4), 39 x 39 (13 x 14 + 2 = 184), 5 x 7 (2 x 4 + 2 = 10; an odd patch
  // count: 588 x 35 bytes is not a multiple of 8, and three grids are 12 bytes, padded to 16)
  const std::vector<Img> three{{10, -1, 3, 3}, {20, -1, 39, 39}, {400, -1, 5, 7}};
  check(pixels::rows_of({3, 3}) == 4 && pixels::rows_of({39, 39}) == 184 && pixels::rows_of({5, 7}) == 10 && pixels::rows_of({93, 93}) == 31 * 32 + 2,
        "rows = ceil(n_h / 3) x (ceil(n_w / 3) + 1) + 2");
  for (uint32_t ver : vers)
    for (int pfx = 0; pfx < 2; ++pfx) {
      char what[160];
      std::snprintf(what, sizeof what, "version %u, %s kFlagPrefix", ver, pfx ? "with" : "without");
      if (pfx && ver == fe::kVersion) {
        check(read(frame(fe::kFlagPixels, true, three), ver) == V::kMalformed, "version 4 with kFlagPrefix: malformed, as for any frame (the prefix needs version 5)");
        continue;
      }
      const std::vector<char> f = frame(fe::kFlagPixels | fe::kFlagKeep, pfx, three);
      pixels::Parsed px;
      const V v = read(f, ver, &px);
      std::snprintf(what, sizeof what, "red: version %u, %s kFlagPrefix: the frame is read", ver, pfx ? "with" : "without");
      check(v == V::kOk, what);
      const size_t base = sizeof(fe::SubmitHeader) + (pfx ? sizeof(fe::SubmitPrefix) : 0) + 4 * (1 + 40);
      const size_t spans_at = base + 4, grids_at = spans_at + 3 * 16, data_at = grids_at + 16;
      check(px.spans.size() == 3 && px.spans[1].start == 20 && px.spans[1].rows == 184 && px.spans[2].hash == 0x3333ull && px.grids.size() == 3 &&
                px.grids[1].n_h == 39 && px.grids[2].n_w == 7,
            "red: the spans and the grids come back");
      check(px.off.size() == 3 && px.off[0] == data_at && px.off[1] == data_at + pad8(588 * 9) && px.off[2] == px.off[1] + pad8(588 * 39 * 39) &&
                px.off[2] + pad8(588 * 35) == f.size(),
            "red: the grids' array is padded to 8 bytes; each image's bytes start where the last one's padded end is, the last ends the frame");
      check(px.off.size() == 3 && f[px.off[0]] == 1 && f[px.off[1]] == 2 && f[px.off[2]] == 3, "red: an offset is its image's first byte");
    }

  // one image whose count of bytes makes a padded tail (588 x 35 = 20,580: 4 bytes of padding)
  check(pad8(588 * 35) - 588 * 35 == 4 && pad8(588 * 9) == 588 * 9 + 4 && pad8(588 * 2) == 588 * 2, "the padding the vectors above rely on (odd counts pad by 4)");

  // both flags together: malformed on every connection, in the head and in the image part
  for (uint32_t ver : vers) {
    check(read(frame(fe::kFlagPixels | fe::kFlagImages, false, three), ver) == V::kMalformed, "both flags together: malformed (the head's reader reads it as a rows frame; the pixel reader refuses it)");
    {
      fe::SubmitHeader hb;
      fe::SubmitPrefix pb;
      size_t bb = 0;
      const std::vector<char> bf = frame(fe::kFlagPixels | fe::kFlagImages, false, three);
      check(!frames::parse_submit_head(bf.data(), bf.size(), ver, &hb, &pb, &bb), "red: the head refuses both flags before the image part is looked at");
    }
    const std::vector<char> both = frame(fe::kFlagPixels | fe::kFlagImages, false, three);
    check(pixels::parse(both.data(), both.size(), 0, fe::kFlagPixels | fe::kFlagImages).v == V::kMalformed, "both flags: pixels::parse alone says malformed");
  }

  // n 0 and over kMaxImages (the count is read before anything else is believed)
  for (uint32_t ver : vers) {
    check(read(frame(fe::kFlagPixels, false, three, 0), ver) == V::kMalformed, "n_images 0: malformed");
    check(read(frame(fe::kFlagPixels, false, three, fe::kMaxImages + 1), ver) == V::kMalformed, "n_images over kMaxImages: malformed");
    check(read(frame(fe::kFlagPixels, false, {}, 0), ver) == V::kMalformed, "no image at all: malformed");
    std::vector<Img> many(fe::kMaxImages, Img{1, -1, 3, 3});
    check(read(frame(fe::kFlagPixels, false, many), ver) == V::kOk, "red: kMaxImages images (3 x 3 patches each) read");
    many.push_back(Img{1, -1, 3, 3});
    check(read(frame(fe::kFlagPixels, false, many), ver) == V::kMalformed, "kMaxImages + 1 images, the count written truthfully: malformed");
  }

  // a byte count that is not exactly the layout's: one byte short, one byte long, a whole image missing, the count of bytes
  // of a bigger grid than the data holds
  for (uint32_t ver : vers) {
    check(read(frame(fe::kFlagPixels, false, three, -1, 1), ver) == V::kMalformed, "one byte short: malformed");
    check(read(frame(fe::kFlagPixels, false, three, -1, -1), ver) == V::kMalformed, "one byte long: malformed");
    check(read(frame(fe::kFlagPixels, false, three, -1, 588 * 35), ver) == V::kMalformed, "an image's bytes missing: malformed");
    check(read(frame(fe::kFlagPixels, false, three, 2), ver) == V::kMalformed, "n says 2 of 3 images' bytes: malformed");
    std::vector<char> f = frame(fe::kFlagPixels, false, three);
    f.resize(sizeof(fe::SubmitHeader) + 4 * 41 + 3);
    check(read(f, ver) == V::kMalformed, "the image part cut inside n: malformed");
    // the grids say more than the bytes hold (2 x 3 patches' bytes, a 4 x 3 grid): the byte count is not the layout's
    std::vector<char> g = frame(fe::kFlagPixels, false, {Img{10, 4, 2, 3}});
    const size_t grid_at = sizeof(fe::SubmitHeader) + 4 * 41 + 4 + 16;
    const pixels::Grid big{4, 3};
    std::memcpy(g.data() + grid_at, &big, 4);
    check(read(g, ver) == V::kMalformed, "a grid whose bytes the frame does not hold: malformed (the bytes decide before the grid)");
  }

  // well-formed and refused: the message names the reason; the connection lives
  for (uint32_t ver : vers) {
    std::string why;
    check(read(frame(fe::kFlagPixels, false, {Img{10, 5, 3, 3}}), ver, nullptr, &why) == V::kRefused && why.find("makes 4 rows, its span has 5") != std::string::npos,
          "red: a grid that does not match its span's rows: refused, with both numbers");
    check(read(frame(fe::kFlagPixels, false, three, -1), ver) == V::kOk, "red: the same three with the right rows read");
    check(read(frame(fe::kFlagPixels, false, {Img{10, 3, 3, 3}, Img{50, 184, 39, 39}}), ver, nullptr, &why) == V::kRefused && why.find("image 0") == 0,
          "red: the first image that does not match is named");
    // 96 x 96 patches: 9,216 patches (the most), 32 x 33 + 2 = 1,058 rows: over 1,024
    check(read(frame(fe::kFlagPixels, false, {Img{10, -1, 96, 96}}), ver, nullptr, &why) == V::kRefused && why.find("1058 rows, over the 1024") != std::string::npos,
          "red: rows over 1,024 (matching its span): refused");
    // 31 x 32 patches' worth of rows is 1,024 exactly: lh 30 x (lw 33 + 1) + 2 = 1,022; 31 x 33 + 2 = 1,025; lh 32 x (lw 31 + 1) + 2 = 1,026
    check(pixels::rows_of({90, 93}) == 30 * 32 + 2 && read(frame(fe::kFlagPixels, false, {Img{10, -1, 90, 93}}), ver) == V::kOk,
          "red: an image of 962 rows (30 x 32 + 2) is read");
    check(pixels::rows_of({93, 96}) == 31 * 33 + 2 && pixels::rows_of({93, 96}) == 1025 &&
              read(frame(fe::kFlagPixels, false, {Img{10, -1, 93, 96}}), ver, nullptr, &why) == V::kRefused && why.find("1025 rows") != std::string::npos,
          "red: 1,025 rows refused");
    check(pixels::rows_of({96, 90}) == 32 * 31 + 2 && pixels::rows_of({96, 90}) == 994 && read(frame(fe::kFlagPixels, false, {Img{10, -1, 96, 90}}), ver) == V::kOk,
          "red: 994 rows read (8,640 patches)");
    // 97 x 96 = 9,312 patches: over kMaxPatches (checked before the rows)
    check(read(frame(fe::kFlagPixels, false, {Img{10, -1, 97, 96}}), ver, nullptr, &why) == V::kRefused && why.find("9312 patches, over the 9216") != std::string::npos,
          "red: patches over kMaxPatches: refused");
    check(read(frame(fe::kFlagPixels, false, {Img{10, -1, 3, 3063}}), ver) == V::kOk && read(frame(fe::kFlagPixels, false, {Img{10, -1, 1533, 3}}), ver) == V::kOk,
          "red: the reference run's strip 3 x 3,063 (9,189 patches) and the tall one 1,533 x 3 are read");
    check(read(frame(fe::kFlagPixels, false, {Img{10, 2, 0, 5}}), ver, nullptr, &why) == V::kRefused && why.find("grid of 0 x 5") != std::string::npos,
          "red: a grid of 0 patches: refused (it holds no bytes, so the byte count is exact)");
  }

  // the rows frames and the plain ones are as the head reads them
  for (uint32_t ver : vers) {
    fe::SubmitHeader h{};
    h.flags = fe::kFlagImages, h.n_stop_ids = 1, h.n_prompt = 2;
    std::vector<char> rows(sizeof h + 4 * 3 + 4 + 16 + 2 * 5120 * 2);
    std::memcpy(rows.data(), &h, sizeof h);
    fe::SubmitHeader hh;
    fe::SubmitPrefix px;
    size_t base = 0;
    check(frames::parse_submit_head(rows.data(), rows.size(), ver, &hh, &px, &base) && base == sizeof h + 12, "a rows frame is read by the head's rule, as before");
    check(!frames::parse_submit_head(rows.data(), sizeof h + 12, ver, &hh, &px, &base), "a rows frame with no image part is malformed, as before");
    h.flags = 0;
    std::memcpy(rows.data(), &h, sizeof h);
    rows.resize(sizeof h + 12);
    check(frames::parse_submit_head(rows.data(), rows.size(), ver, &hh, &px, &base), "a text frame is read, as before");
    rows.push_back(0);
    check(!frames::parse_submit_head(rows.data(), rows.size(), ver, &hh, &px, &base), "a text frame with a stray byte is malformed, as before");
    // the pixel flag without an image part
    h.flags = fe::kFlagPixels;
    std::memcpy(rows.data(), &h, sizeof h);
    rows.resize(sizeof h + 12);
    check(!frames::parse_submit_head(rows.data(), rows.size(), ver, &hh, &px, &base), "red: kFlagPixels with no image part: malformed");
  }

  std::printf(fails ? "pixels_test FAILED (%d)\n" : "pixels_test passed\n", fails);
  return fails != 0;
}
