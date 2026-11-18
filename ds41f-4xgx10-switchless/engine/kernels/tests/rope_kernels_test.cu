// Every kernel that rotates with the YaRN table, run twice on the same inputs: reading the table (rope_table_build at the
// factor) and computing the rotation (rope_yarn_calc, rope_plain_calc for the plain one); the outputs must be
// bit-identical at factors 16, 32 and 64. For 32 and 64, rope_rows on (1, 0) pairs is also compared with the host's
// cos/sin of fp32(position) * the pair's fp32 frequency, bound 2.5e-3. Usage: rope_kernels_test [factor ...]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "attn/rope_calc.cuh"
#include "fill.cuh"
#include "kernels/prefill_attn.h"
#include "testutil.h"

using namespace attn_test;

namespace {

struct Named {
  std::string name;
  std::vector<uint8_t> bytes;
};
template <class T>
void grab(std::vector<Named>& o, const char* name, const Dev<T>& d) {
  CUDA_CHECK(cudaDeviceSynchronize());
  auto h = d.host();
  Named n{name, std::vector<uint8_t>(h.size() * sizeof(T))};
  std::memcpy(n.bytes.data(), h.data(), n.bytes.size());
  o.push_back(std::move(n));
}

constexpr int kTopk = 512;

// One run of every kernel with `rope`, deterministic: same data every call.
void suite(const attn::Rope& rope, int ctx, std::vector<Named>& out) {
  fill::g_seed = 1000;
  std::mt19937 rng(7);
  // ---- rope_rows
  {
    const int rows = 40, heads = 2, dim = 512;
    std::vector<int> ph(rows);
    for (int r = 0; r < rows; ++r) ph[r] = r < 8 ? r * 3 : (int)(rng() % (uint32_t)ctx);
    Dev<int> pos(ph.data(), rows);
    Dev<uint16_t> a((size_t)rows * heads * dim), b((size_t)rows * heads * dim);
    fill::bf16(a, 2.f);
    CUDA_CHECK(cudaMemcpy(b.p, a.p, a.n * 2, cudaMemcpyDeviceToDevice));
    attn::rope_rows(a.p, rows, heads, dim, pos.p, rope, false, 0);
    attn::rope_rows(b.p, rows, heads, dim, pos.p, rope, true, 0);
    grab(out, "rope_rows fwd", a);
    grab(out, "rope_rows inv", b);
  }
  // ---- kv_write: rows consecutive at the window's top, into a ring, with the dequantised output
  {
    const int R = 24, p0 = ctx - 1000;
    std::vector<int> ph(R);
    for (int r = 0; r < R; ++r) ph[r] = p0 + r;
    Dev<int> pos(ph.data(), R);
    Dev<uint16_t> kvp((size_t)R * 512), nw(512), kvo((size_t)R * 512);
    fill::bf16(kvp, 2.f);
    fill::bf16_range(nw, 0.5f, 1.5f);
    Dev<uint8_t> rk((size_t)attn::kRing * 512), rs((size_t)attn::kRing * 16);
    attn::kv_write(kvp.p, nw.p, pos.p, rope, attn::Window{rk.p, rs.p}, kvo.p, R, 0);
    grab(out, "kv_write kv_out", kvo);
    grab(out, "kv_write ring kv", rk);
    grab(out, "kv_write ring sc", rs);
  }
  // ---- kv_write_sets (the drafter's window writes): two sets, rows at the window's top, into rings
  {
    const int R = 6;
    Dev<int> cnt(std::vector<int>{R}.data(), 1), p0(std::vector<int>{ctx - 100}.data(), 1);
    Dev<uint16_t> kvp((size_t)2 * R * 512), nw(512);
    fill::bf16(kvp, 2.f);
    fill::bf16_range(nw, 0.5f, 1.5f);
    Dev<uint8_t> rk((size_t)2 * attn::kRing * 512), rs((size_t)2 * attn::kRing * 16);
    attn::KvSets sets;
    sets.n = 2;
    for (int k = 0; k < 2; ++k)
      sets.s[k] = {kvp.p + (size_t)k * R * 512, nw.p,
                   attn::Window{rk.p + (size_t)k * attn::kRing * 512, rs.p + (size_t)k * attn::kRing * 16}};
    attn::kv_write_sets(sets, R, cnt.p, p0.p, rope, false, 0);
    grab(out, "kv_write_sets ring kv", rk);
    grab(out, "kv_write_sets ring sc", rs);
  }
  // ---- decode_prep (the drafter's attention prep: q_norm and the window write): 6 rows at the window's top
  {
    const int R = 6;
    std::vector<int> ph(R);
    for (int r = 0; r < R; ++r) ph[r] = ctx - 50 + r;
    Dev<int> pos(ph.data(), R);
    Dev<uint16_t> qa((size_t)R * attn::kQLora), qn(attn::kQLora), kvp((size_t)R * 512), kn(512);
    fill::bf16(qa, 2.f);
    fill::bf16_range(qn, 0.5f, 1.5f);
    fill::bf16(kvp, 2.f);
    fill::bf16_range(kn, 0.5f, 1.5f);
    Dev<uint8_t> qq((size_t)R * attn::kQLora), qs((size_t)R * attn::kQLora / 32);
    Dev<uint8_t> rk((size_t)attn::kRing * 512), rs((size_t)attn::kRing * 16);
    attn::DecodePrep a;
    a.pos = pos.p, a.table = rope, a.qa = qa.p, a.q_norm = qn.p, a.qq = qq.p, a.qs = qs.p;
    a.kv_pre = kvp.p, a.kv_norm = kn.p, a.win = attn::Window{rk.p, rs.p};
    attn::decode_prep(a, R, false, 0);
    grab(out, "decode_prep ring kv", rk);
    grab(out, "decode_prep ring sc", rs);
  }
  // ---- index_q_prep: random positions
  {
    const int rows = 24;
    std::vector<int> ph(rows);
    for (int r = 0; r < rows; ++r) ph[r] = r < 4 ? r : (int)(rng() % (uint32_t)ctx);
    Dev<int> pos(ph.data(), rows);
    Dev<uint16_t> iq((size_t)rows * attn::kIdxHeads * attn::kIdxDim), dbg(iq.n);
    Dev<float> wp((size_t)rows * attn::kIdxHeads), qw(wp.n);
    Dev<uint8_t> q((size_t)rows * attn::kIdxHeads * 64);
    Dev<uint32_t> qsc(wp.n);
    fill::bf16(iq, 2.f);
    fill::f32(wp, 1.f);
    attn::index_q_prep(iq.p, wp.p, pos.p, rope, attn::IndexQ{q.p, qsc.p, qw.p}, dbg.p, rows, 0);
    grab(out, "index_q_prep q", q);
    grab(out, "index_q_prep sc", qsc);
    grab(out, "index_q_prep w", qw);
    grab(out, "index_q_prep dbg", dbg);
  }
  // ---- index_fused and index_dense: ratio 2, no pool, the last rows of the window
  {
    const int rows = 2, ratio = 2, bucket = (ctx + 4095) / 4096 * 4096, max_scan = bucket / ratio;
    const int kstride = (max_scan + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
    const int cap = whole_chunks(bucket / ratio + 64) * attn::kChunkEntries;
    ChunkLists lists(1, cap / attn::kChunkEntries);
    std::vector<int> ph(rows);
    for (int r = 0; r < rows; ++r) ph[r] = ctx - rows + r;
    Dev<int> pos(ph.data(), rows);
    Dev<uint8_t> ikk((size_t)cap * 64), iks((size_t)cap * 4);
    fill::u8(ikk, 0xFF, 0);
    fill::u8(iks, 7, 120);
    Dev<uint16_t> iq((size_t)rows * attn::kIdxHeads * attn::kIdxDim);
    Dev<float> wp((size_t)rows * attn::kIdxHeads);
    fill::bf16(iq, 2.f);
    fill::f32(wp, 1.f);
    attn::IndexKeys ik{ikk.p, iks.p, 0};
    ik.list = lists.d.p, ik.list_stride = lists.len;
    for (int kind = 0; kind < 2; ++kind) {
      Dev<uint16_t> keys((size_t)rows * kstride), bkeys((size_t)rows * kstride / 8), tkey((size_t)rows * kTopk);
      Dev<int> top((size_t)rows * kTopk), counters(std::max(rows, attn::kMaxDecodeRows));
      Dev<uint32_t> scratch(attn::index_dense_scratch() / 4);
      attn::IndexSel t{kTopk, top.p, tkey.p, kTopk}, ps{};
      if (kind == 0)
        attn::index_fused(iq.p, wp.p, pos.p, rope, rows, ratio, ik, attn::IndexScan{}, max_scan, keys.p, kstride,
                          nullptr, kstride / 8, t, ps, counters.p, false, 0);
      else
        attn::index_dense(iq.p, wp.p, pos.p, rope, rows, ratio, ik, max_scan, keys.p, kstride, bkeys.p, kstride / 8, t,
                          ps, scratch.p, false, 0);
      const char* nm = kind ? "index_dense" : "index_fused";
      grab(out, (std::string(nm) + " keys").c_str(), keys);
      grab(out, (std::string(nm) + " top").c_str(), top);
      grab(out, (std::string(nm) + " tkey").c_str(), tkey);
    }
  }
  // ---- compressor_emit (decode kernel) and compressor_emit_p (prefill kernel), ratio 2 and 1. The caches' chunk
  // lists all name chunk 0, so a 4M-position window needs one chunk of memory.
  for (int ratio : {2, 1}) {
    for (int pf = 0; pf < 2; ++pf) {
      const int rows = pf ? 64 : 16, p0 = (ctx - rows) / 16 * 16;  // even start, ratio 2 pairs
      const int cap = ctx / ratio + 64;
      const int nch = whole_chunks(cap) + 1;
      Dev<uint32_t> zl(nch);  // every chunk is chunk 0
      Dev<uint8_t> ckk((size_t)attn::kChunkEntries * 256), cks((size_t)attn::kChunkEntries * 32),
          ikk((size_t)attn::kChunkEntries * 64), iks((size_t)attn::kChunkEntries * 4);
      attn::Compressed ckv{ckk.p, cks.p, cap};
      ckv.list = zl.p, ckv.list_stride = nch;
      attn::IndexKeys ik{ikk.p, iks.p, cap};
      ik.list = zl.p, ik.list_stride = nch;
      Dev<uint16_t> norm(512), wk((size_t)128 * 512), knorm(128);
      fill::bf16_range(norm, 0.5f, 1.5f);
      fill::bf16(wk, 0.05f);
      fill::bf16_range(knorm, 0.5f, 1.5f);
      attn::CompressorWeights w{ratio, nullptr, nullptr, norm.p, wk.p, knorm.p};
      Dev<float> kv((size_t)rows * 512), score((size_t)rows * 512), ckv_c(16 * 512), csc_c(16 * 512);
      fill::f32(kv, 2.f);
      fill::f32(score, 2.f);
      fill::f32(ckv_c, 2.f);
      fill::f32(csc_c, 2.f);
      std::vector<int> ph(rows);
      for (int r = 0; r < rows; ++r) ph[r] = p0 + r;
      Dev<int> pos(ph.data(), rows);
      if (pf)
        dsk::compressor_emit_p(kv.p, score.p, pos.p, rows, w, {ckv_c.p, csc_c.p}, rope, ckv, ik, 0);
      else
        attn::compressor_emit(kv.p, score.p, pos.p, rows, w, {ckv_c.p, csc_c.p}, rope, ckv, ik, nullptr, 0, 0);
      std::string nm = std::string(pf ? "emit_p" : "emit") + " ratio " + std::to_string(ratio) + " ";
      grab(out, (nm + "ckv").c_str(), ckk);
      grab(out, (nm + "ckv sc").c_str(), cks);
      grab(out, (nm + "ik").c_str(), ikk);
      grab(out, (nm + "ik sc").c_str(), iks);
      grab(out, (nm + "carry kv").c_str(), ckv_c);
      grab(out, (nm + "carry sc").c_str(), csc_c);
    }
  }
  // ---- sparse_attn (decode) and sparse_attn_p (prefill): the window at the top, kmax 0 and 64
  for (int kmax : {0, 64}) {
    const int R = 4, p0 = ctx - R, ncap = 1024, slots = R + 127;
    std::vector<int> ph(R);
    for (int r = 0; r < R; ++r) ph[r] = p0 + r;
    Dev<int> pos(ph.data(), R);
    Dev<uint16_t> q((size_t)R * 16 * 512), o1(q.n), o4(q.n), op(q.n);
    fill::bf16(q, 3.f);
    Dev<uint8_t> wkv((size_t)slots * 512), wsc((size_t)slots * 16), ckk((size_t)ncap * 256), cks((size_t)ncap * 32);
    fill::u8(wkv, 0x77, 0);
    fill::u8(wsc, 7, 120);
    fill::u8(ckk, 0xFF, 0);
    fill::u8(cks, 0x1F, 0x10);
    Dev<float> sink(16);
    fill::f32(sink, 2.f);
    std::vector<int> th((size_t)R * std::max(kmax, 1), -1);
    for (int r = 0; r < R && kmax > 0; ++r) {
      std::vector<int> pick;
      std::vector<uint8_t> used(ncap, 0);
      while ((int)pick.size() < kmax) {
        int x = (int)(rng() % ncap);
        if (!used[x]) used[x] = 1, pick.push_back(x);
      }
      std::sort(pick.begin(), pick.end());
      for (int i = 0; i < kmax; ++i) th[(size_t)r * kmax + i] = pick[i];
    }
    Dev<int> topk(th.data(), th.size());
    attn::Window win{wkv.p, wsc.p, -1, p0 - 127};
    attn::Compressed ckv{nullptr, nullptr, 0};
    if (kmax) ckv = listed(attn::Compressed{ckk.p, cks.p, ncap}, ncap);
    const int max_splits = 16;
    Dev<float> po((size_t)R * max_splits * 16 * 512), pml((size_t)R * max_splits * 16 * 2);
    attn::AttnScratch sc{po.p, pml.p, max_splits};
    attn::sparse_attn(q.p, pos.p, {}, R, rope, win, ckv, kmax ? topk.p : nullptr, kmax, sink.p, o1.p, 1, sc, 0);
    const std::string nm = "sparse_attn kmax " + std::to_string(kmax) + " ";
    grab(out, (nm + "splits 1").c_str(), o1);
    if (kmax) {
      attn::sparse_attn(q.p, pos.p, {}, R, rope, win, ckv, topk.p, kmax, sink.p, o4.p, 4, sc, 0);
      grab(out, (nm + "splits 4").c_str(), o4);
    }
    dsk::sparse_attn_p(q.p, pos.p, R, rope, win, ckv, kmax ? topk.p : nullptr, kmax, sink.p, op.p, 0);
    grab(out, ("sparse_attn_p kmax " + std::to_string(kmax)).c_str(), op);
  }
}

__global__ void freq_kernel(float* f, attn::RopeParams p) { f[threadIdx.x] = attn::rope_freq(p, threadIdx.x); }

int run_factor(int factor) {
  const int ctx = factor * 65536;  // 1M, 2M, 4M
  std::printf("factor %d (window %d positions)\n", factor, ctx);
  Dev<float2> tab((size_t)(ctx + 64) * 32);
  attn::rope_table_build(tab.p, ctx + 64, true, factor, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<Named> a, b;
  suite(attn::Rope(tab.p), ctx, a);
  suite(attn::rope_yarn_calc(factor), ctx, b);
  int bad = 0;
  if (a.size() != b.size()) {
    std::printf("FAIL: %zu outputs from the table, %zu computed\n", a.size(), b.size());
    return 1;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    const bool same = a[i].bytes == b[i].bytes;
    bool nonzero = false;
    for (uint8_t c : a[i].bytes) nonzero |= c != 0;
    std::printf("  %-34s %9zu bytes  %s%s\n", a[i].name.c_str(), a[i].bytes.size(), same ? "bit-identical" : "DIFFERENT",
                nonzero ? "" : "  (all zero: the case wrote nothing)");
    bad += !same;
    // An output the kernels never wrote would make the comparison empty; the carry and entries must be written.
    if (!nonzero && a[i].name.find("ik sc") == std::string::npos) ++bad;
  }
  {  // the plain rotation (layers 0-1, the drafter): the table and the computed one, the same kernels
    Dev<float2> ptab((size_t)(ctx + 64) * 32);
    attn::rope_table_build(ptab.p, ctx + 64, false, factor, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Named> pa, pb;
    suite(attn::Rope(ptab.p), ctx, pa);
    suite(attn::rope_plain_calc(), ctx, pb);
    if (pa.size() != pb.size()) {
      std::printf("FAIL: plain: %zu outputs from the table, %zu computed\n", pa.size(), pb.size());
      return bad + 1;
    }
    for (size_t i = 0; i < pa.size(); ++i) {
      const bool same = pa[i].bytes == pb[i].bytes;
      bool nonzero = false;
      for (uint8_t c : pa[i].bytes) nonzero |= c != 0;
      std::printf("  plain %-34s %9zu bytes  %s%s\n", pa[i].name.c_str(), pa[i].bytes.size(),
                  same ? "bit-identical" : "DIFFERENT", nonzero ? "" : "  (all zero: the case wrote nothing)");
      bad += !same;
      if (!nonzero && pa[i].name.find("ik sc") == std::string::npos) ++bad;
    }
  }
  if (factor != 16) {  // against the host: (1, 0) pairs through rope_rows give (cos, sin) in bf16
    attn::RopeParams p = attn::rope_params(true, factor);
    Dev<float> df(32);
    freq_kernel<<<1, 32>>>(df.p, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto f = df.host();
    const int rows = 64;
    std::vector<int> ph(rows);
    for (int r = 0; r < rows; ++r) ph[r] = r < 16 ? r * 5 : ctx - 1 - (r - 16) * 20011;
    Dev<int> pos(ph.data(), rows);
    std::vector<uint16_t> hx((size_t)rows * 512, 0);
    for (int r = 0; r < rows; ++r)
      for (int i = 0; i < 32; ++i) hx[(size_t)r * 512 + 448 + 2 * i] = 0x3f80;  // 1.0 bf16
    Dev<uint16_t> x(hx.data(), hx.size());
    attn::rope_rows(x.p, rows, 1, 512, pos.p, attn::rope_yarn_calc(factor), false, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto y = x.host();
    double worst = 0;
    for (int r = 0; r < rows; ++r)
      for (int i = 0; i < 32; ++i) {
        const double ang = (double)((float)ph[r] * f[i]);  // the recipe's fp32 product
        const double c = std::cos(ang), s = std::sin(ang);
        worst = std::max(worst, std::fabs(bf2f(y[(size_t)r * 512 + 448 + 2 * i]) - c));
        worst = std::max(worst, std::fabs(bf2f(y[(size_t)r * 512 + 449 + 2 * i]) - s));
      }
    const bool ok = worst <= 2.5e-3;
    std::printf("  host cos/sin of fp32(position) * frequency, %d positions x 32 pairs: worst abs %.2e (bound 2.5e-3) %s\n",
                rows, worst, ok ? "ok" : "FAIL");
    bad += !ok;
  }
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> fs;
  for (int i = 1; i < argc; ++i) fs.push_back(std::atoi(argv[i]));
  if (fs.empty()) fs = {16, 32, 64};
  int bad = 0;
  for (int f : fs) {
    if (f != 16 && f != 32 && f != 64) die("rope_kernels_test: factor %d", f);
    bad += run_factor(f);
  }
  std::printf("%s\n", bad ? "FAILED" : "all kernels: table and computed rotation bit-identical");
  return bad ? 1 : 0;
}
