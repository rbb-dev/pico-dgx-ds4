// The encoder's operations on the reference's own tensors, for tools/vision/vit_compare.py .
//
//   vit_teacher --model DIR --dumps DIR --out DIR --stems a,b [--mode all|teacher|free|weights|gemm]
//               [--op-blocks 0,1,2,7,16,31] [--full s1,s2] [--full-patches 4096] [--control NAME]
//
// --dumps is what dump_ref.py wrote in the container; --out gets what vit_compare.py reads:
//   teacher/<stem>/<point>.npy   each single vit:: operation fed the reference's input (read from dumps/ops/<stem>/),
//                                its output written under the point's name (patchify: patches, [n][640]);
//   free/<stem>/<point>.npy      the unsplit tower run free, every point of the reference's free-running dump (the rows
//                                of every 16th patch for an image not named in --full and over --full-patches patches);
//   weights/<name>.npy           the copy-type weights as the checkpoint reader holds them;
//   gemm_random/<shape>/out.npy  vit::gemm on dumps/gemm_random's random data (the floor against cuBLAS).
// --control NAME runs a deliberately wrong build (the positive controls of the exactness plan; the output goes to a
// folder of its own): swap-heads, rope-axes, norm-swap, qkv-order, gate-up (errors in blocks 0-2 and in the rotary table),
// unfold-window, unfold-order, pad-grey (the unfold), delimiters (newline and end exchanged). The split-only errors live
// in vit_test --wrong. Run it with the engine stopped (the checkpoint's tower is about 1.3 GiB of device memory plus the
// aligner's 0.2 GiB and the image's buffers: a few GiB for the largest image).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "vision/tests/npy.h"
#include "vision/vit.h"

using namespace vit;

namespace {

[[noreturn]] void fail(const std::string& m) {
  std::fprintf(stderr, "vit_teacher: %s\n", m.c_str());
  std::exit(2);
}

// device memory that frees itself
struct DBuf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit DBuf(size_t b) : bytes(b) { CUDA_CHECK(cudaMalloc(&p, b ? b : 1)); }
  ~DBuf() { cudaFree(p); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  DBuf(DBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; }
  bf16* b() const { return static_cast<bf16*>(p); }
  float* f() const { return static_cast<float*>(p); }
};

cudaStream_t st = 0;

// An array of the dumps, checked against the shape the operation wants.
std::vector<u16> load_u16(const std::string& path, size_t rows, size_t cols) {
  const npy::Array a = npy::read(path);
  if (a.shape.size() != 2 || a.shape[0] != rows || a.shape[1] != cols)
    fail(path + ": shape [" + std::to_string(a.shape.empty() ? 0 : a.shape[0]) + "," +
         std::to_string(a.shape.size() > 1 ? a.shape[1] : 0) + "], wanted [" + std::to_string(rows) + "," + std::to_string(cols) + "]");
  return npy::as_u16(a);
}

std::vector<u16> load_1d(const std::string& path, size_t n) {
  const npy::Array a = npy::read(path);
  if (a.shape.size() != 1 || a.shape[0] != n) fail(path + ": not a vector of " + std::to_string(n));
  return npy::as_u16(a);
}

std::vector<float> load_f32(const std::string& path, size_t rows, size_t cols) {
  const npy::Array a = npy::read(path);
  if (a.shape.size() != 2 || a.shape[0] != rows || a.shape[1] != cols) fail(path + ": unexpected shape");
  return npy::as_f32(a);
}

template <class T>
void upload_to(DBuf& d, const std::vector<T>& h) {
  CUDA_CHECK(cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

DBuf up(const std::vector<u16>& h) {
  DBuf d(h.size() * 2);
  upload_to(d, h);
  return d;
}

std::vector<u16> down(const bf16* p, size_t n) {
  std::vector<u16> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
  return h;
}

void put_u16(const std::string& path, const std::vector<u16>& h, std::vector<size_t> shape) {
  npy::write(path, "<u2", shape, h.data(), h.size() * 2);
}

// ----------------------------------------------------------------------------------------------- controls

const char* const kControls[] = {"swap-heads", "rope-axes", "norm-swap", "qkv-order", "gate-up", "unfold-window", "unfold-order",
                                 "pad-grey", "delimiters"};

void swap_rows(std::vector<u16>& m, int cols, int a, int b, int n) {
  for (int r = 0; r < n; r++)
    for (int c = 0; c < cols; c++) std::swap(m[(size_t)(a + r) * cols + c], m[(size_t)(b + r) * cols + c]);
}

// The errors of the weights, all in blocks 0-2 where the reference's own nulls leave room to see them.
void apply_control(Weights& w, const std::string& c) {
  if (c == "norm-swap") std::swap(w.blocks[1].norm1, w.blocks[1].norm2);
  if (c == "qkv-order") {  // q and k segments of block 0 exchanged
    swap_rows(w.blocks[0].wqkv, kDim, 0, kDim, kDim);
    swap_rows(w.blocks[0].bqkv, 1, 0, kDim, kDim);
  }
  if (c == "gate-up") swap_rows(w.blocks[2].w1, kDim, 0, kInter, kInter);
  if (c == "swap-heads") {  // heads 0 and 1 exchanged in q, k and v of block 1, wo untouched
    for (int seg = 0; seg < 3; seg++) {
      swap_rows(w.blocks[1].wqkv, kDim, seg * kDim, seg * kDim + kHeadDim, kHeadDim);
      swap_rows(w.blocks[1].bqkv, 1, seg * kDim, seg * kDim + kHeadDim, kHeadDim);
    }
  }
  if (c == "delimiters") std::swap(w.newline, w.end);
}

Wrong wrong_of(const std::string& c) {
  if (c == "rope-axes") return Wrong::RopeAxes;
  if (c == "unfold-window") return Wrong::UnfoldWindow;
  if (c == "unfold-order") return Wrong::UnfoldOrder;
  if (c == "pad-grey") return Wrong::PadGrey;
  return Wrong::None;
}

int unfold_variant(Wrong w) { return w == Wrong::UnfoldWindow ? 1 : w == Wrong::UnfoldOrder ? 2 : w == Wrong::PadGrey ? 3 : 0; }

// ----------------------------------------------------------------------------------------------- the context

struct Ctx {
  std::string dumps, out;
  Weights w;
  DevWeights dw;
  Wrong wrong = Wrong::None;
  std::vector<int> op_blocks;
  std::vector<std::string> full;
  int full_patches = 4096;
};

struct Img {
  Grid g;
  std::vector<uint8_t> pixels;
};

Img load_pixels(const Ctx& c, const std::string& stem) {
  const npy::Array a = npy::read(c.dumps + "/ops/" + stem + "/pixels.npy");
  if (a.descr != "|u1" || a.shape.size() != 3 || a.shape[2] != 3 || a.shape[0] % kPatch || a.shape[1] % kPatch)
    fail(stem + ": pixels.npy is not a u8 [H][W][3] image of whole patches");
  return {make_grid((int)a.shape[0] / kPatch, (int)a.shape[1] / kPatch), a.data};
}

// ----------------------------------------------------------------------------------------------- teacher-forced

void teacher_image(const Ctx& c, const std::string& stem) {
  const std::string in = c.dumps + "/ops/" + stem + "/", o = c.out + "/teacher/" + stem + "/";
  const Img img = load_pixels(c, stem);
  const Grid g = img.g;
  const int n = g.patches(), m = g.merged();
  auto ref = [&](const std::string& p, size_t rows, size_t cols) { return up(load_u16(in + p + ".npy", rows, cols)); };
  auto emit = [&](const std::string& name, const bf16* p, size_t rows, size_t cols) {
    CUDA_CHECK(cudaStreamSynchronize(st));
    put_u16(o + name + ".npy", down(p, rows * cols), {rows, cols});
  };

  // copy-type: patchify from the reference's own pixels; the rotary table from the grid
  {
    DBuf px(img.pixels.size());
    CUDA_CHECK(cudaMemcpy(px.p, img.pixels.data(), img.pixels.size(), cudaMemcpyHostToDevice));
    DBuf patches((size_t)n * kPatchKPad * 2);
    patchify(static_cast<const uint8_t*>(px.p), g.nh, g.nw, c.dw.table, patches.b(), st);
    emit("patches", patches.b(), n, kPatchKPad);
    DBuf cs((size_t)n * 32 * 4), sn((size_t)n * 32 * 4);
    static const std::vector<float> inv = rope_inv_freq();
    rope_table(g.nh, g.nw, inv.data(), cs.f(), sn.f(), c.wrong == Wrong::RopeAxes, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<float> hc((size_t)n * 32), hs((size_t)n * 32);
    CUDA_CHECK(cudaMemcpy(hc.data(), cs.p, hc.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hs.data(), sn.p, hs.size() * 4, cudaMemcpyDeviceToHost));
    npy::write(o + "rope_cos.npy", "<f4", {(size_t)n, 32}, hc.data(), hc.size() * 4);
    npy::write(o + "rope_sin.npy", "<f4", {(size_t)n, 32}, hs.data(), hs.size() * 4);
  }
  // patch_embed: the reference's patches (588 values), zero-padded to 640 as the engine holds them
  {
    const std::vector<u16> p588 = load_u16(in + "patches_in.npy", n, kPatch * kPatch * 3);
    std::vector<u16> p640((size_t)n * kPatchKPad, 0);
    for (int r = 0; r < n; r++) std::memcpy(&p640[(size_t)r * kPatchKPad], &p588[(size_t)r * kPatchK], kPatchK * 2);
    DBuf a = up(p640), out((size_t)n * kDim * 2);
    gemm(rows_view(a.b(), kPatchKPad), c.dw.patch_w, c.dw.patch_b, rows_out(out.b(), kDim), n, kDim, kPatchKPad, st);
    emit("patch_embed", out.b(), n, kDim);
  }
  // the blocks asked for
  const int H = kHeads;
  for (int bi : c.op_blocks) {
    if (bi < 0 || bi >= (int)c.dw.blocks.size()) fail("block out of range");
    const DevBlock& b = c.dw.blocks[bi];
    char pre[32];
    std::snprintf(pre, sizeof pre, "block%02d.", bi);
    const std::string p = pre;
    DBuf x_in = ref(p + "x_in", n, kDim), xn(n * kDim * 2ull);
    rmsnorm(rows_view(x_in.b(), kDim), b.norm1, xn.b(), n, st);
    emit(p + "norm1_out", xn.b(), n, kDim);
    {  // wqkv from the reference's norm output
      DBuf a = ref(p + "norm1_out", n, kDim), out((size_t)n * 3 * kDim * 2);
      gemm(rows_view(a.b(), kDim), b.wqkv, b.bqkv, rows_out(out.b(), 3 * kDim), n, 3 * kDim, kDim, st);
      emit(p + "qkv", out.b(), n, 3 * kDim);
    }
    {  // the rotation of the reference's q and k (its wqkv output, its table)
      DBuf qkv = ref(p + "qkv", n, 3 * kDim);
      DBuf cs((size_t)n * 32 * 4), sn((size_t)n * 32 * 4);
      upload_to(cs, load_f32(in + "rope_cos.npy", n, 32));
      upload_to(sn, load_f32(in + "rope_sin.npy", n, 32));
      rope_apply(qkv.b(), qkv.b() + kDim, 3 * kDim, n, H, cs.f(), sn.f(), st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      const std::vector<u16> h = down(qkv.b(), (size_t)n * 3 * kDim);
      std::vector<u16> q((size_t)n * kDim), k((size_t)n * kDim);
      for (int r = 0; r < n; r++) {
        std::memcpy(&q[(size_t)r * kDim], &h[(size_t)r * 3 * kDim], kDim * 2);
        std::memcpy(&k[(size_t)r * kDim], &h[(size_t)r * 3 * kDim + kDim], kDim * 2);
      }
      put_u16(o + p + "q_rope.npy", q, {(size_t)n, (size_t)kDim});
      put_u16(o + p + "k_rope.npy", k, {(size_t)n, (size_t)kDim});
    }
    {  // attention on the reference's rotated q, k and its v
      DBuf q = ref(p + "q_rope", n, kDim), k = ref(p + "k_rope", n, kDim), out((size_t)n * kDim * 2);
      const std::vector<u16> qkv = load_u16(in + p + "qkv.npy", n, 3 * kDim);
      std::vector<u16> vh((size_t)n * kDim);
      for (int r = 0; r < n; r++) std::memcpy(&vh[(size_t)r * kDim], &qkv[(size_t)r * 3 * kDim + 2 * kDim], kDim * 2);
      DBuf v = up(vh);
      attention(q.b(), kDim, k.b(), kDim, v.b(), kDim, out.b(), kDim, n, H, st);
      emit(p + "attn_o", out.b(), n, kDim);
    }
    {
      DBuf a = ref(p + "attn_o", n, kDim), out((size_t)n * kDim * 2);
      gemm(rows_view(a.b(), kDim), b.wo, b.bo, rows_out(out.b(), kDim), n, kDim, kDim, st);
      emit(p + "wo_out", out.b(), n, kDim);
    }
    {
      DBuf wo = ref(p + "wo_out", n, kDim), out((size_t)n * kDim * 2);
      add(x_in.b(), kDim, wo.b(), kDim, out.b(), kDim, n, kDim, st);
      emit(p + "attn_add", out.b(), n, kDim);
    }
    {
      DBuf a = ref(p + "attn_add", n, kDim), out((size_t)n * kDim * 2);
      rmsnorm(rows_view(a.b(), kDim), b.norm2, out.b(), n, st);
      emit(p + "norm2_out", out.b(), n, kDim);
    }
    {
      DBuf a = ref(p + "norm2_out", n, kDim), out((size_t)n * 2 * kInter * 2);
      gemm(rows_view(a.b(), kDim), b.w1, nullptr, rows_out(out.b(), 2 * kInter), n, 2 * kInter, kDim, st);
      emit(p + "w1_out", out.b(), n, 2 * kInter);
    }
    {
      DBuf a = ref(p + "w1_out", n, 2 * kInter), out((size_t)n * kInter * 2);
      swiglu(a.b(), 2 * kInter, kInter, out.b(), kInter, n, st);
      emit(p + "swiglu_out", out.b(), n, kInter);
    }
    {
      DBuf a = ref(p + "swiglu_out", n, kInter), out((size_t)n * kDim * 2);
      gemm(rows_view(a.b(), kInter), b.w2, nullptr, rows_out(out.b(), kDim), n, kDim, kInter, st);
      emit(p + "w2_out", out.b(), n, kDim);
    }
    {
      DBuf aa = ref(p + "attn_add", n, kDim), w2 = ref(p + "w2_out", n, kDim), out((size_t)n * kDim * 2);
      add(aa.b(), kDim, w2.b(), kDim, out.b(), kDim, n, kDim, st);
      emit(p + "mlp_add", out.b(), n, kDim);
    }
  }
  // the final norm and the aligner, each on the reference's input
  {
    DBuf a = ref("norm_in", n, kDim), out((size_t)n * kDim * 2);
    rmsnorm(rows_view(a.b(), kDim), c.dw.norm, out.b(), n, st);
    emit("norm", out.b(), n, kDim);
  }
  {
    DBuf a = ref("norm", n, kDim), out((size_t)m * kMergeIn * 2);
    unfold3(a.b(), g.nh, g.nw, out.b(), unfold_variant(c.wrong), st);
    emit("unfold", out.b(), m, kMergeIn);
  }
  {
    DBuf a = ref("unfold", m, kMergeIn), out((size_t)m * kLm * 2);
    gemm(rows_view(a.b(), kMergeIn), c.dw.a1w, c.dw.a1b, rows_out(out.b(), kLm), m, kLm, kMergeIn, st);
    emit("a1_out", out.b(), m, kLm);
  }
  {
    DBuf a = ref("a1_out", m, kLm), out((size_t)m * kLm * 2);
    gelu(a.b(), kLm, out.b(), kLm, m, kLm, st);
    emit("g", out.b(), m, kLm);
  }
  {
    DBuf a = ref("g", m, kLm), out((size_t)m * kLm * 2);
    gemm(rows_view(a.b(), kLm), c.dw.a2w, c.dw.a2b, rows_out(out.b(), kLm), m, kLm, kLm, st);
    emit("aligner", out.b(), m, kLm);
  }
  {
    DBuf a = ref("aligner", m, kLm), out((size_t)g.rows() * kLm * 2);
    span_rows(rows_view(a.b(), kLm), c.dw.start, c.dw.newline, c.dw.end, g.lh, g.lw, 0, g.rows(), out.b(), st);
    emit("span", out.b(), g.rows(), kLm);
  }
}

// ----------------------------------------------------------------------------------------------- free-running

bool is_full(const Ctx& c, const std::string& stem, int patches) {
  return patches <= c.full_patches || std::find(c.full.begin(), c.full.end(), stem) != c.full.end();
}

void free_image(const Ctx& c, const std::string& stem) {
  const Img img = load_pixels(c, stem);
  const Grid g = img.g;
  const int n = g.patches(), m = g.merged();
  const bool full = is_full(c, stem, n);
  DBuf px(img.pixels.size());
  CUDA_CHECK(cudaMemcpy(px.p, img.pixels.data(), img.pixels.size(), cudaMemcpyHostToDevice));
  DBuf work(unsplit_work_bytes(n, m)), span((size_t)g.rows() * kLm * 2);
  RunOut out{span.b()};
  const std::string o = c.out + "/free/" + stem + "/";
  int points = 0;
  out.tap = [&](const char* point, const bf16* rows, int nr, int nc) {
    const int stride = full ? 1 : 16;
    const int keep = (nr + stride - 1) / stride;
    std::vector<u16> sel((size_t)keep * nc);
    if (stride == 1) {
      CUDA_CHECK(cudaMemcpy(sel.data(), rows, sel.size() * 2, cudaMemcpyDeviceToHost));
    } else {
      for (int r = 0; r < keep; r++)
        CUDA_CHECK(cudaMemcpy(&sel[(size_t)r * nc], rows + (size_t)r * stride * nc, (size_t)nc * 2, cudaMemcpyDeviceToHost));
    }
    put_u16(o + point + ".npy", sel, {(size_t)keep, (size_t)nc});
    points++;
  };
  run_unsplit(c.dw, Image{g, static_cast<const uint8_t*>(px.p)}, work.p, out, st, c.wrong);
  std::printf("free %s: %d patches, %s, %d points\n", stem.c_str(), n, full ? "whole" : "every 16th row", points);
}

// ----------------------------------------------------------------------------------------------- weights, random gemm

void weights_out(const Ctx& c) {
  const std::string o = c.out + "/weights/";
  auto put = [&](const std::string& name, const std::vector<u16>& v) { put_u16(o + name + ".npy", v, {v.size()}); };
  put("pixel_table", pixel_table());
  put("start", c.w.start), put("newline", c.w.newline), put("end", c.w.end), put("norm_final", c.w.norm);
  for (size_t i = 0; i < c.w.blocks.size(); i++) {
    char nm[32];
    std::snprintf(nm, sizeof nm, "norm1_%02zu", i), put(nm, c.w.blocks[i].norm1);
    std::snprintf(nm, sizeof nm, "norm2_%02zu", i), put(nm, c.w.blocks[i].norm2);
  }
  std::printf("weights: %zu files\n", 5 + 2 * c.w.blocks.size());
}

void gemm_random_out(const Ctx& c) {
  const struct { const char* name; int N, K; bool bias; } shapes[] = {
      {"patch_embed", kDim, kPatchKPad, true}, {"wqkv", 3 * kDim, kDim, true}, {"wo", kDim, kDim, true},
      {"w1", 2 * kInter, kDim, false}, {"w2", kDim, kInter, false}, {"aligner1", kLm, kMergeIn, true},
      {"aligner2", kLm, kLm, true}};
  for (const auto& s : shapes) {
    const std::string d = c.dumps + "/gemm_random/" + s.name + "/";
    const npy::Array a0 = npy::read(d + "a.npy");
    if (a0.shape.size() != 2 || a0.shape[1] != (size_t)s.K) fail(d + "a.npy: wrong K");
    const int M = (int)a0.shape[0];
    DBuf a = up(npy::as_u16(a0)), w = up(load_u16(d + "w.npy", s.N, s.K)), out((size_t)M * s.N * 2);
    DBuf bias(s.bias ? (size_t)s.N * 2 : 2);
    if (s.bias) upload_to(bias, load_1d(d + "b.npy", s.N));
    gemm(rows_view(a.b(), s.K), w.b(), s.bias ? bias.b() : nullptr, rows_out(out.b(), s.N), M, s.N, s.K, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    put_u16(c.out + "/gemm_random/" + s.name + "/out.npy", down(out.b(), (size_t)M * s.N), {(size_t)M, (size_t)s.N});
    std::printf("gemm_random %s: [%d][%d] x [%d][%d]\n", s.name, M, s.K, s.N, s.K);
  }
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  size_t i = 0;
  while (i <= s.size()) {
    const size_t j = s.find(',', i);
    const std::string t = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!t.empty()) v.push_back(t);
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, mode = "all", stems, control, full, opb = "0,1,2,7,16,31";
  Ctx c;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) fail(a + " needs a value");
      return argv[++i];
    };
    if (a == "--model") model = val();
    else if (a == "--dumps") c.dumps = val();
    else if (a == "--out") c.out = val();
    else if (a == "--stems") stems = val();
    else if (a == "--mode") mode = val();
    else if (a == "--op-blocks") opb = val();
    else if (a == "--full") full = val();
    else if (a == "--full-patches") c.full_patches = std::atoi(val().c_str());
    else if (a == "--control") control = val();
    else fail("unknown argument " + a);
  }
  if (model.empty() || c.dumps.empty() || c.out.empty() || stems.empty()) fail("--model, --dumps, --out and --stems are required");
  if (mode != "all" && mode != "teacher" && mode != "free" && mode != "weights" && mode != "gemm") fail("--mode: all, teacher, free, weights or gemm");
  if (!control.empty()) {
    bool known = false;
    for (const char* k : kControls) known |= control == k;
    if (!known) fail("unknown control " + control);
  }
  for (const std::string& b : split(opb)) c.op_blocks.push_back(std::atoi(b.c_str()));
  c.full = split(full);

  CUDA_CHECK(cudaStreamCreate(&st));
  c.w = load_checkpoint(model);
  if (!control.empty()) apply_control(c.w, control);
  c.wrong = wrong_of(control);
  c.dw = upload(c.w);
  std::printf("checkpoint %s: %zu blocks uploaded%s\n", model.c_str(), c.w.blocks.size(),
              control.empty() ? "" : (", control " + control).c_str());
  const bool all = mode == "all";
  if (all || mode == "weights") weights_out(c);
  if (all || mode == "gemm") gemm_random_out(c);
  for (const std::string& stem : split(stems)) {
    if (all || mode == "teacher") {
      teacher_image(c, stem);
      std::printf("teacher %s: done\n", stem.c_str());
    }
    if (all || mode == "free") free_image(c, stem);
  }
  release(c.dw);
  std::printf("DONE\n");
  return 0;
}
