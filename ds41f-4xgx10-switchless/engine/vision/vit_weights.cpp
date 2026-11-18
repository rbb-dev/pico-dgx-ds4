#include "vision/vit_weights.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include "weights/io.h"
#include "weights/safetensors.h"

namespace vit {

u16 to_bf16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  b += 0x7fffu + ((b >> 16) & 1u);
  return (u16)(b >> 16);
}

float from_bf16(u16 v) {
  uint32_t b = (uint32_t)v << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

namespace {

std::vector<u16> read_tensor(const dsw::Checkpoint& ck, const std::string& name, std::vector<uint64_t> shape) {
  const dsw::SrcTensor& t = ck.at(name);
  if (t.dtype != "BF16") die("%s: dtype %s, wanted BF16", name.c_str(), t.dtype.c_str());
  if (t.shape != shape) die("%s: unexpected shape", name.c_str());
  std::vector<u16> v(t.bytes / 2);
  int fd = ::open(ck.shards[t.shard].c_str(), O_RDONLY);
  if (fd < 0) die("open %s", ck.shards[t.shard].c_str());
  uint64_t done = 0;
  while (done < t.bytes) {
    ssize_t n = ::pread(fd, reinterpret_cast<char*>(v.data()) + done, t.bytes - done, t.offset + done);
    if (n <= 0) die("read %s", name.c_str());
    done += (uint64_t)n;
  }
  ::close(fd);
  return v;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  float uniform() {  // [-1, 1)
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return (float)(int32_t)(s >> 32) * (1.0f / 2147483648.0f);
  }
};

std::vector<u16> random_matrix(Rng& g, size_t n, float scale, float offset = 0.f) {
  std::vector<u16> v(n);
  for (auto& x : v) x = to_bf16(offset + scale * g.uniform());
  return v;
}

}  // namespace

Weights load_checkpoint(const std::string& dir, int blocks) {
  const dsw::Checkpoint ck = dsw::Checkpoint::open(dir);
  Weights w;
  w.blocks.resize(blocks);
  for (int i = 0; i < blocks; i++) {
    const std::string p = "vision.blocks." + std::to_string(i) + ".";
    BlockW& b = w.blocks[i];
    b.wqkv = read_tensor(ck, p + "attn.wqkv.weight", {3 * kDim, kDim});
    b.bqkv = read_tensor(ck, p + "attn.wqkv.bias", {3 * kDim});
    b.wo = read_tensor(ck, p + "attn.wo.weight", {kDim, kDim});
    b.bo = read_tensor(ck, p + "attn.wo.bias", {kDim});
    b.w1 = read_tensor(ck, p + "mlp.w1.weight", {2 * kInter, kDim});
    b.w2 = read_tensor(ck, p + "mlp.w2.weight", {kDim, kInter});
    b.norm1 = read_tensor(ck, p + "norm1.weight", {kDim});
    b.norm2 = read_tensor(ck, p + "norm2.weight", {kDim});
  }
  w.patch_w = read_tensor(ck, "vision.patch_embed.proj.weight", {kDim, kPatchK});
  w.patch_b = read_tensor(ck, "vision.patch_embed.proj.bias", {kDim});
  w.norm = read_tensor(ck, "vision.norm.weight", {kDim});
  w.a1w = read_tensor(ck, "aligner.w1.weight", {kLm, kMergeIn});
  w.a1b = read_tensor(ck, "aligner.w1.bias", {kLm});
  w.a2w = read_tensor(ck, "aligner.w2.weight", {kLm, kLm});
  w.a2b = read_tensor(ck, "aligner.w2.bias", {kLm});
  w.start = read_tensor(ck, "image_start", {kLm});
  w.newline = read_tensor(ck, "image_newline", {kLm});
  w.end = read_tensor(ck, "image_end", {kLm});
  return w;
}

Weights make_synthetic(uint64_t seed, int blocks) {
  Rng g(seed);
  auto lin = [&](int rows, int cols) { return random_matrix(g, (size_t)rows * cols, std::sqrt(3.0f / cols)); };
  auto bias = [&](int n) { return random_matrix(g, n, 0.05f); };
  auto norm = [&](int n) { return random_matrix(g, n, 0.25f, 1.0f); };
  Weights w;
  w.blocks.resize(blocks);
  for (BlockW& b : w.blocks) {
    b.wqkv = lin(3 * kDim, kDim), b.bqkv = bias(3 * kDim);
    b.wo = lin(kDim, kDim), b.bo = bias(kDim);
    b.w1 = lin(2 * kInter, kDim), b.w2 = lin(kDim, kInter);
    b.norm1 = norm(kDim), b.norm2 = norm(kDim);
  }
  w.patch_w = lin(kDim, kPatchK), w.patch_b = bias(kDim);
  w.norm = norm(kDim);
  w.a1w = lin(kLm, kMergeIn), w.a1b = bias(kLm);
  w.a2w = lin(kLm, kLm), w.a2b = bias(kLm);
  w.start = random_matrix(g, kLm, 1.0f), w.newline = random_matrix(g, kLm, 1.0f), w.end = random_matrix(g, kLm, 1.0f);
  return w;
}

std::vector<u16> rows_of(const std::vector<u16>& m, int cols, int r0, int n) {
  if ((size_t)(r0 + n) * cols > m.size()) die("rows_of: rows %d..%d past the matrix", r0, r0 + n);
  return std::vector<u16>(m.begin() + (size_t)r0 * cols, m.begin() + (size_t)(r0 + n) * cols);
}

namespace {
void append(std::vector<u16>& to, const std::vector<u16>& v) { to.insert(to.end(), v.begin(), v.end()); }
}  // namespace

Weights slice_rank(const Weights& whole, int r) {
  if (r < 0 || r >= kRanks) die("slice_rank: rank %d", r);
  Weights s;
  s.blocks.resize(whole.blocks.size());
  constexpr int hq = kDim / kRanks;      // 256: four heads of 64
  constexpr int mq = kInter / kRanks;    // 704
  for (size_t i = 0; i < whole.blocks.size(); i++) {
    const BlockW& w = whole.blocks[i];
    BlockW& o = s.blocks[i];
    for (int seg = 0; seg < 3; seg++) {  // q, k, v each give their heads 4r..4r+3
      append(o.wqkv, rows_of(w.wqkv, kDim, seg * kDim + hq * r, hq));
      append(o.bqkv, rows_of(w.bqkv, 1, seg * kDim + hq * r, hq));
    }
    o.wo = rows_of(w.wo, kDim, hq * r, hq);
    o.bo = rows_of(w.bo, 1, hq * r, hq);
    for (int seg = 0; seg < 2; seg++) append(o.w1, rows_of(w.w1, kDim, seg * kInter + mq * r, mq));  // gate, up
    o.w2 = rows_of(w.w2, kInter, hq * r, hq);
    o.norm1 = w.norm1, o.norm2 = w.norm2;
  }
  constexpr int aq = kLm / kRanks;  // 1280
  s.patch_w = whole.patch_w, s.patch_b = whole.patch_b, s.norm = whole.norm;
  s.a1w = rows_of(whole.a1w, kMergeIn, aq * r, aq), s.a1b = rows_of(whole.a1b, 1, aq * r, aq);
  s.a2w = rows_of(whole.a2w, kLm, aq * r, aq), s.a2b = rows_of(whole.a2b, 1, aq * r, aq);
  s.start = whole.start, s.newline = whole.newline, s.end = whole.end;
  return s;
}

std::vector<u16> pad_patch_weight(const std::vector<u16>& w) {
  std::vector<u16> p((size_t)kDim * kPatchKPad, 0);
  for (int r = 0; r < kDim; r++)
    for (int c = 0; c < kPatchK; c++) p[(size_t)r * kPatchKPad + c] = w[(size_t)r * kPatchK + c];
  return p;
}

std::vector<u16> pixel_table() {
  std::vector<u16> t(256);
  for (int v = 0; v < 256; v++) {
    const float x = (float)v / 255.0f;
    t[v] = to_bf16((x - 0.5f) / 0.5f);
  }
  return t;
}

std::vector<float> rope_inv_freq(float theta) {
  std::vector<float> f(16);
  for (int i = 0; i < 16; i++) f[i] = 1.0f / std::pow(theta, (float)(2 * i) / (float)kRopeHalf);
  return f;
}

Grid make_grid(int nh, int nw) {
  Grid g;
  g.nh = nh, g.nw = nw;
  g.lh = (nh + kMerge - 1) / kMerge, g.lw = (nw + kMerge - 1) / kMerge;
  return g;
}

}  // namespace vit
