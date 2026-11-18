// Host-only checks of the encoder's weights, slices, grids and host references (no GPU).
//   vit_host_test                run all
//   vit_host_test --roundtrip DIR  write a one-block checkpoint under DIR, read it back, compare (writes ~170 MB)
#include <sys/stat.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "vision/vit_ref.h"
#include "vision/vit_weights.h"

using namespace vit;

namespace {
int fails = 0;
void expect(bool ok, const std::string& what) {
  if (!ok) fails++;
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

size_t bytes_of(const Weights& w) {
  size_t n = 0;
  for (const BlockW& b : w.blocks)
    n += b.wqkv.size() + b.bqkv.size() + b.wo.size() + b.bo.size() + b.w1.size() + b.w2.size() + b.norm1.size() + b.norm2.size();
  n += w.patch_w.size() + w.patch_b.size() + w.norm.size() + w.a1w.size() + w.a1b.size() + w.a2w.size() + w.a2b.size();
  n += w.start.size() + w.newline.size() + w.end.size();
  return n * 2;
}

void test_bf16() {
  expect(to_bf16(1.0f) == 0x3F80 && to_bf16(-1.0f) == 0xBF80, "bf16 of 1 and -1");
  expect(to_bf16(from_bf16(0x1234)) == 0x1234, "bf16 round trip");
  // 1 + 2^-8 is exactly between 1.0 and 1 + 2^-7: ties go to even (1.0); 1 + 3 * 2^-8 goes up to 1 + 2^-6 ... even
  expect(to_bf16(1.0f + 1.0f / 256) == 0x3F80, "tie rounds to even (down)");
  expect(to_bf16(1.0f + 3.0f / 256) == 0x3F82, "tie rounds to even (up)");
}

void test_grid() {
  const struct { int nh, nw, lh, lw, rows; } c[] = {{39, 39, 13, 13, 184}, {55, 55, 19, 19, 382}, {93, 93, 31, 31, 994},
                                                    {3, 3063, 1, 1021, 1024}, {49, 65, 17, 22, 17 * 23 + 2}};
  for (auto& t : c) {
    const Grid g = make_grid(t.nh, t.nw);
    expect(g.lh == t.lh && g.lw == t.lw && g.rows() == t.rows,
           "grid " + std::to_string(t.nh) + "x" + std::to_string(t.nw) + " -> rows " + std::to_string(g.rows()));
  }
}

void test_tables() {
  const std::vector<u16> t = pixel_table();
  expect(t.size() == 256 && t[0] == to_bf16(-1.0f) && t[255] == to_bf16(1.0f), "pixel table ends are -1 and 1");
  bool mono = true;
  for (int i = 1; i < 256; i++) mono &= from_bf16(t[i]) >= from_bf16(t[i - 1]);
  expect(mono, "pixel table never decreases");
  const std::vector<float> f = rope_inv_freq();
  expect(f.size() == 16 && f[0] == 1.0f, "inv_freq[0] = 1");
  expect(std::fabs(f[15] - std::pow(10000.0, -30.0 / 32.0)) < 1e-8, "inv_freq[15] = theta^(-30/32)");
}

// The four ranks' slices put back give the whole tensor, byte for byte.
void test_slices() {
  const Weights w = make_synthetic(7, 2);
  Weights s[4];
  size_t total = 0;
  for (int r = 0; r < 4; r++) s[r] = slice_rank(w, r), total += bytes_of(s[r]);
  bool ok = true;
  for (size_t b = 0; b < w.blocks.size(); b++) {
    std::vector<u16> wqkv(3 * 1024 * 1024), w1(5632 * 1024), wo, w2;
    for (int r = 0; r < 4; r++) {
      const BlockW& o = s[r].blocks[b];
      for (int seg = 0; seg < 3; seg++)
        std::memcpy(&wqkv[((size_t)seg * 1024 + 256 * r) * 1024], &o.wqkv[(size_t)seg * 256 * 1024], 256 * 1024 * 2);
      for (int seg = 0; seg < 2; seg++)
        std::memcpy(&w1[((size_t)seg * 2816 + 704 * r) * 1024], &o.w1[(size_t)seg * 704 * 1024], 704 * 1024 * 2);
      wo.insert(wo.end(), o.wo.begin(), o.wo.end());
      w2.insert(w2.end(), o.w2.begin(), o.w2.end());
      ok &= o.wqkv.size() == 768 * 1024 && o.w1.size() == 1408 * 1024 && o.wo.size() == 256 * 1024 &&
            o.w2.size() == 256 * 2816 && o.bqkv.size() == 768 && o.bo.size() == 256;
    }
    ok &= wqkv == w.blocks[b].wqkv && w1 == w.blocks[b].w1 && wo == w.blocks[b].wo && w2 == w.blocks[b].w2;
  }
  std::vector<u16> a1, a2;
  for (int r = 0; r < 4; r++) a1.insert(a1.end(), s[r].a1w.begin(), s[r].a1w.end()), a2.insert(a2.end(), s[r].a2w.begin(), s[r].a2w.end());
  ok &= a1 == w.a1w && a2 == w.a2w;
  expect(ok, "four slices put together are the whole tensors");
  // a rank holds 6,428,672 bytes a block and 36,705,280 in the aligner, 1,239,040 replicated (32 blocks: 243,661,824)
  const size_t per_block = (768 * 1024 + 768 + 256 * 1024 + 256 + 1408 * 1024 + 256 * 2816 + 2 * 1024) * 2;
  const size_t rest = (size_t)(1280 * 9216 + 1280 + 1280 * 5120 + 1280) * 2 + 1239040;
  expect(per_block == 6428672, "one block of a rank is 6,428,672 bytes");
  expect(bytes_of(s[0]) == 2 * per_block + rest, "a rank of two blocks holds blocks + aligner + replicated");
  expect(32 * per_block + rest == 243661824, "32 blocks: 243,661,824 bytes a rank");
  expect(total == 4 * bytes_of(s[0]), "ranks hold equal bytes");
  expect(slice_rank(w, 1).blocks[0].wqkv != slice_rank(w, 2).blocks[0].wqkv, "ranks 1 and 2 differ");
}

void test_padding() {
  const Weights w = make_synthetic(3, 1);
  const std::vector<u16> p = pad_patch_weight(w.patch_w);
  bool ok = p.size() == 1024 * 640;
  for (int r = 0; r < 1024 && ok; r++) {
    for (int c = 0; c < 588; c++) ok &= p[(size_t)r * 640 + c] == w.patch_w[(size_t)r * 588 + c];
    for (int c = 588; c < 640; c++) ok &= p[(size_t)r * 640 + c] == 0;
  }
  expect(ok, "patch weight padded to 640 columns with zeros");
}

// Hand-checkable cases of the references
void test_refs() {
  // 4 x 5 grid, channel c of cell (y, x) holds 100 * c + y * 5 + x
  const int nh = 4, nw = 5;
  std::vector<u16> x((size_t)nh * nw * kDim);
  for (int y = 0; y < nh; y++)
    for (int xx = 0; xx < nw; xx++)
      for (int c = 0; c < kDim; c++) x[((size_t)y * nw + xx) * kDim + c] = to_bf16((float)(c % 4 * 100 + y * 5 + xx));
  const std::vector<u16> u = ref::unfold3(x, nh, nw);
  const Grid g = make_grid(nh, nw);  // 2 x 2 windows
  expect(g.lh == 2 && g.lw == 2 && u.size() == 4 * 9216, "unfold of 4 x 5: 2 x 2 windows");
  // window (1, 1): rows 3..5, columns 3..5; channel 2, offset (i, j) = (0, 1) is cell (3, 4) = 200 + 19
  expect(from_bf16(u[3 * 9216 + 2 * 9 + 0 * 3 + 1]) == 219.0f, "window (1,1), channel 2, offset (0,1) = cell (3,4)");
  // the grid has 4 rows: offset i = 2 of the last window falls on row 5: zero
  expect(from_bf16(u[3 * 9216 + 2 * 9 + 2 * 3 + 0]) == 0.0f, "padding below the grid is zero");
  expect(from_bf16(u[1 * 9216 + 2 * 9 + 0 * 3 + 2]) == 0.0f, "padding right of the grid is zero (window (0,1), column 5)");
  Weights d;
  d.start.assign(kLm, to_bf16(1)), d.newline.assign(kLm, to_bf16(2)), d.end.assign(kLm, to_bf16(3));
  std::vector<u16> g5(4 * (size_t)kLm);
  for (int r = 0; r < 4; r++) std::fill(g5.begin() + (size_t)r * kLm, g5.begin() + (size_t)(r + 1) * kLm, to_bf16(10.0f + r));
  const std::vector<u16> s = ref::span(g5, d, 2, 2);
  const float want[] = {1, 10, 11, 2, 12, 13, 2, 3};
  bool ok = s.size() == 8 * (size_t)kLm;
  for (int r = 0; r < 8 && ok; r++) ok &= from_bf16(s[(size_t)r * kLm]) == want[r];
  expect(ok, "span of 2 x 2: start, row, newline, row, newline, end");
  // pixels: a 1 x 1 patch image; byte value = c * 100 + py * 14 + px goes through the table
  std::vector<uint8_t> px(14 * 14 * 3);
  for (int py = 0; py < 14; py++)
    for (int pxx = 0; pxx < 14; pxx++)
      for (int c = 0; c < 3; c++) px[((size_t)py * 14 + pxx) * 3 + c] = (uint8_t)(c * 50 + py * 3 + pxx / 5);
  const std::vector<u16> tab = pixel_table();
  const std::vector<u16> pt = ref::patchify(px, 1, 1, tab);
  expect(pt[2 * 196 + 5 * 14 + 9] == tab[(uint8_t)(2 * 50 + 5 * 3 + 9 / 5)], "patch column = channel * 196 + row * 14 + column");
  bool tail = true;
  for (int c = 588; c < 640; c++) tail &= pt[c] == 0;
  expect(tail, "patch tail columns are zero");
}

void write_all(FILE* f, const void* p, size_t n) {
  if (std::fwrite(p, 1, n, f) != n) { std::perror("write"); std::exit(1); }
}

void test_roundtrip(const std::string& dir) {
  ::mkdir(dir.c_str(), 0755);
  const Weights w = make_synthetic(11, 1);
  struct T { std::string name; const std::vector<u16>* v; std::vector<uint64_t> shape; };
  const BlockW& b = w.blocks[0];
  const std::vector<T> ts = {
      {"vision.blocks.0.attn.wqkv.weight", &b.wqkv, {3072, 1024}}, {"vision.blocks.0.attn.wqkv.bias", &b.bqkv, {3072}},
      {"vision.blocks.0.attn.wo.weight", &b.wo, {1024, 1024}},     {"vision.blocks.0.attn.wo.bias", &b.bo, {1024}},
      {"vision.blocks.0.mlp.w1.weight", &b.w1, {5632, 1024}},      {"vision.blocks.0.mlp.w2.weight", &b.w2, {1024, 2816}},
      {"vision.blocks.0.norm1.weight", &b.norm1, {1024}},          {"vision.blocks.0.norm2.weight", &b.norm2, {1024}},
      {"vision.patch_embed.proj.weight", &w.patch_w, {1024, 588}}, {"vision.patch_embed.proj.bias", &w.patch_b, {1024}},
      {"vision.norm.weight", &w.norm, {1024}},                     {"aligner.w1.weight", &w.a1w, {5120, 9216}},
      {"aligner.w1.bias", &w.a1b, {5120}},                         {"aligner.w2.weight", &w.a2w, {5120, 5120}},
      {"aligner.w2.bias", &w.a2b, {5120}},                         {"image_start", &w.start, {5120}},
      {"image_newline", &w.newline, {5120}},                       {"image_end", &w.end, {5120}}};
  std::string hdr = "{";
  uint64_t off = 0;
  for (size_t i = 0; i < ts.size(); i++) {
    std::string shape = "[";
    for (size_t j = 0; j < ts[i].shape.size(); j++) shape += (j ? "," : "") + std::to_string(ts[i].shape[j]);
    shape += "]";
    const uint64_t nb = ts[i].v->size() * 2;
    hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"BF16\",\"shape\":" + shape +
           ",\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + nb) + "]}";
    off += nb;
  }
  hdr += "}";
  while (hdr.size() % 8) hdr += ' ';
  FILE* f = std::fopen((dir + "/model-00001-of-00001.safetensors").c_str(), "wb");
  if (!f) { std::perror("open"); std::exit(1); }
  const uint64_t hl = hdr.size();
  write_all(f, &hl, 8);
  write_all(f, hdr.data(), hdr.size());
  for (auto& t : ts) write_all(f, t.v->data(), t.v->size() * 2);
  std::fclose(f);
  const Weights r = load_checkpoint(dir, 1);
  const BlockW& rb = r.blocks[0];
  expect(rb.wqkv == b.wqkv && rb.bqkv == b.bqkv && rb.wo == b.wo && rb.bo == b.bo && rb.w1 == b.w1 && rb.w2 == b.w2 &&
             rb.norm1 == b.norm1 && rb.norm2 == b.norm2,
         "checkpoint read back: block 0 equal");
  expect(r.patch_w == w.patch_w && r.patch_b == w.patch_b && r.norm == w.norm && r.a1w == w.a1w && r.a1b == w.a1b &&
             r.a2w == w.a2w && r.a2b == w.a2b && r.start == w.start && r.newline == w.newline && r.end == w.end,
         "checkpoint read back: the rest equal");
  std::remove((dir + "/model-00001-of-00001.safetensors").c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && !std::strcmp(argv[1], "--roundtrip")) {
    test_roundtrip(argv[2]);
  } else if (argc == 1) {
    test_bf16(), test_grid(), test_tables(), test_slices(), test_padding(), test_refs();
  } else {
    std::fprintf(stderr, "usage: vit_host_test [--roundtrip DIR]\n");
    return 2;
  }
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
