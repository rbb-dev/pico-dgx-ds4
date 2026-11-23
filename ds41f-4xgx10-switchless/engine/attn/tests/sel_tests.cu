// The decode selection (select_bins through select_dense) on synthetic keys at 256K, 1M, 2M and 4M keys a row: against a
// host top-k (ties to the lowest position) for six key distributions; up to 1,048,576 keys also against the 17-bit
// reference (sel_old.cuh); past 1M the reference must differ on the spread cases ("control"). Exit status 0: every
// check passed. Usage: sel_tests [--quick]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "indexer.cu"
#include "testutil.h"

namespace attn {
namespace {
#include "sel_old.cuh"

template <bool kNew, bool kWide>
__global__ void __launch_bounds__(kSelThreads)
    sel_kernel(SelectArgs a, uint32_t* gh, const uint16_t* bmax, size_t bstride, int n, int cap) {
  extern __shared__ __align__(16) uint8_t dsm[];
  uint16_t* ckey = reinterpret_cast<uint16_t*>(dsm);
  uint32_t* cidx = reinterpret_cast<uint32_t*>(dsm + kDenseCand * 2);
  uint16_t* blist = reinterpret_cast<uint16_t*>(dsm + kDenseCand * 6);
  const int r = blockIdx.x;
  const uint16_t* bm = bmax ? bmax + (size_t)r * bstride : nullptr;
  if (kNew)
    select_dense<kWide>(a, r, gh + (size_t)r * 2048, 0, n, cap, ckey, cidx, bm, blist);
  else
    select_dense_old(a, r, gh + (size_t)r * 2048, n, cap, ckey, cidx, bm, blist);
}

__global__ void hist_kernel(const uint16_t* keys, int n, uint32_t* gh) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    if (keys[i]) atomicAdd(&gh[keys[i] >> 5], 1u);
}
}  // namespace
}  // namespace attn

namespace {

using attn_test::Dev;

struct Sel {
  std::vector<int> pos;
  std::vector<uint16_t> key;
  bool operator==(const Sel& o) const { return pos == o.pos && key == o.key; }
};

// torch.topk's choice: the kmax highest nonzero keys, ties to the lower index, ascending by index
Sel host_select(const std::vector<uint16_t>& k, int n, int kmax) {
  std::vector<int> idx;
  for (int i = 0; i < n; ++i)
    if (k[i]) idx.push_back(i);
  const int m = std::min<int>(kmax, (int)idx.size());
  std::partial_sort(idx.begin(), idx.begin() + m, idx.end(),
                    [&](int a, int b) { return k[a] != k[b] ? k[a] > k[b] : a < b; });
  idx.resize(m);
  std::sort(idx.begin(), idx.end());
  Sel s{std::vector<int>(kmax, -1), std::vector<uint16_t>(kmax, 0)};
  for (int i = 0; i < m; ++i) s.pos[i] = idx[i], s.key[i] = k[idx[i]];
  return s;
}

const char* kCases[] = {"random", "ties", "spread", "ties in the gather", "one bin (fallback)", "fewer than kmax"};
constexpr int kNumCases = 6;

std::vector<uint16_t> make_keys(int c, int n, uint32_t seed) {
  std::mt19937 rng(seed);
  auto u = [&](uint32_t lo, uint32_t hi) { return lo + rng() % (hi - lo + 1); };
  std::vector<uint16_t> k(n, 0);
  auto spread = [&](int count, uint32_t lo, uint32_t hi) {  // count random positions, keys in [lo, hi]
    for (int i = 0; i < count; ++i) k[rng() % n] = (uint16_t)u(lo, hi);
  };
  switch (c) {
    case 0:
      for (auto& x : k) x = (uint16_t)u(1, 65535);
      break;
    case 1:
      for (auto& x : k) x = (uint16_t)u(40000, 40003);
      spread(600, 60000, 65000);
      break;
    case 2:
      for (auto& x : k) x = (uint16_t)u(1, 20000);
      spread(700, 50000, 65000);
      break;
    case 3:
      for (auto& x : k) x = (uint16_t)u(1, 30000);
      spread(3000, 50000, 50000);  // all in one bin: the first in scan order are taken
      spread(300, 60000, 65000);
      break;
    case 4:
      for (auto& x : k) x = (uint16_t)u(40000, 40031);
      break;
    case 5:
      spread(100, 1, 65535);
      break;
  }
  return k;
}

cudaStream_t g_s;
int g_bad = 0, g_checks = 0;

void check(bool ok, const char* what, int n, const char* c, int kmax) {
  ++g_checks;
  if (!ok) ++g_bad;
  if (!ok || std::getenv("SEL_VERBOSE")) std::printf("  %-4s %-34s n=%-8d kmax=%-5d %s\n", ok ? "ok" : "FAIL", what, n, kmax, c);
}

// one row through the old (kNew false) or the new selection; the kernel's candidate room is the dense kernel's
Sel run(const std::vector<uint16_t>& keys, int n, int kmax, bool kNew) {
  const size_t stride = (n + 255) / 256 * 256;
  std::vector<uint16_t> hk(stride, 0);
  std::copy(keys.begin(), keys.begin() + n, hk.begin());
  const int nb = (n + 7) / 8;
  const size_t bstride = (nb + 255) / 256 * 256;
  std::vector<uint16_t> hb(bstride, 0);
  for (int b = 0; b < nb; ++b) {
    uint16_t m = 0;
    for (int j = 0; j < 8 && b * 8 + j < n; ++j) m = std::max(m, hk[b * 8 + j]);
    hb[b] = m;
  }
  Dev<uint16_t> dk(hk.data(), hk.size()), db(hb.data(), hb.size());
  Dev<uint32_t> gh(2048);
  CUDA_CHECK(cudaMemset(gh.p, 0, 2048 * 4));
  attn::hist_kernel<<<256, 256, 0, g_s>>>(dk.p, n, gh.p);
  Dev<int> op(kmax);
  Dev<uint16_t> ok(kmax);
  attn::SelectArgs a{};
  a.keys = dk.p, a.keys_stride = (int)stride, a.n = n, a.rows = 1, a.ratio = 1, a.nparts = 1, a.unit = 1;
  a.kmax = kmax, a.out_pos = op.p, a.out_key = ok.p, a.out_stride = kmax;
  const size_t smem = attn::kDenseCand * 8;
  const bool bm = n > attn::kDenseBmax;  // as index_dense: block maxima for a long scan
  const bool wide = n > 1048576;  // as index_dense chooses it (bkeys_stride > 131072)
  auto kern = !kNew ? attn::sel_kernel<false, false> : wide ? attn::sel_kernel<true, true> : attn::sel_kernel<true, false>;
  CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  kern<<<1, attn::kSelThreads, smem, g_s>>>(a, gh.p, bm ? db.p : nullptr, bstride, n, attn::kDenseCand);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  return Sel{op.host(), ok.host()};
}

// index_merge2, row-major and rank-major, against a host merge. P parts of `width` candidates a row: part q's list of row r is
// ascending positions (q * 1,000,000 + a random walk), keys from a small range (ties) with a fifth of the slots empty (key 0).
// Candidates for the pool (width 2,048) are the same kind with blocks as positions.
void merge_layout(int rows, int P) {
  const int kT = 512, kP = 2048;  // top's and pool's candidate widths and picks
  std::mt19937 rng(7000 + rows * 31 + P);
  auto make = [&](int w, std::vector<int>& pos, std::vector<uint16_t>& key) {  // [rows][P][w]
    pos.assign((size_t)rows * P * w, -1);
    key.assign((size_t)rows * P * w, 0);
    for (int r = 0; r < rows; ++r)
      for (int q = 0; q < P; ++q) {
        int p = q * 1000000;
        for (int i = 0; i < w; ++i) {
          p += 1 + (int)(rng() % 5);
          const size_t at = ((size_t)r * P + q) * w + i;
          pos[at] = p;
          key[at] = rng() % 5 == 0 ? 0 : (uint16_t)(40000 + rng() % 40);  // ties; a fifth empty
        }
      }
  };
  std::vector<int> tp, pp;
  std::vector<uint16_t> tk, pk;
  make(kT, tp, tk);
  make(kP, pp, pk);
  // rank-major copies: [P][rows][w]
  auto rank_major = [&](const auto& v, int w) {
    std::decay_t<decltype(v)> o(v.size());
    for (int r = 0; r < rows; ++r)
      for (int q = 0; q < P; ++q)
        std::copy(v.begin() + ((size_t)r * P + q) * w, v.begin() + ((size_t)r * P + q + 1) * w,
                  o.begin() + ((size_t)q * rows + r) * w);
    return o;
  };
  const auto tp_r = rank_major(tp, kT), pp_r = rank_major(pp, kP);
  const auto tk_r = rank_major(tk, kT), pk_r = rank_major(pk, kP);
  auto launch_merge = [&](const std::vector<int>& cp, const std::vector<uint16_t>& ck, const std::vector<int>& pcp,
                          const std::vector<uint16_t>& pck, bool rm, std::vector<int>& otp, std::vector<uint16_t>& otk,
                          std::vector<int>& opp, std::vector<uint16_t>& opk) {
    Dev<int> dcp(cp.data(), cp.size()), dpcp(pcp.data(), pcp.size()), dot((size_t)rows * kT), dop((size_t)rows * kP);
    Dev<uint16_t> dck(ck.data(), ck.size()), dpck(pck.data(), pck.size()), dotk((size_t)rows * kT), dopk((size_t)rows * kP);
    attn::index_merge2(rows, dcp.p, dck.p, P * kT, attn::IndexSel{kT, dot.p, dotk.p, kT}, dpcp.p, dpck.p, P * kP,
                       attn::IndexSel{kP, dop.p, dopk.p, kP}, false, g_s, rm ? P : 0);
    CUDA_CHECK(cudaStreamSynchronize(g_s));
    otp = dot.host(), otk = dotk.host(), opp = dop.host(), opk = dopk.host();
  };
  std::vector<int> a_tp, a_pp, b_tp, b_pp;
  std::vector<uint16_t> a_tk, a_pk, b_tk, b_pk;
  launch_merge(tp, tk, pp, pk, false, a_tp, a_tk, a_pp, a_pk);
  launch_merge(tp_r, tk_r, pp_r, pk_r, true, b_tp, b_tk, b_pp, b_pk);
  const std::string name = "index_merge2 rank-major rows=" + std::to_string(rows) + " parts=" + std::to_string(P);
  check(a_tp == b_tp && a_tk == b_tk, "rank-major top == row-major top", rows, name.c_str(), kT);
  check(a_pp == b_pp && a_pk == b_pk, "rank-major pool == row-major pool", rows, name.c_str(), kP);
  bool host_ok = true;  // the row-major launch against a host merge (so that the pair is not vacuous)
  for (int r = 0; r < rows; ++r) {
    auto one = [&](const std::vector<int>& cp, const std::vector<uint16_t>& ck, int w, int kmax, const std::vector<int>& op,
                   const std::vector<uint16_t>& ok) {
      const std::vector<uint16_t> row(ck.begin() + (size_t)r * P * w, ck.begin() + (size_t)(r + 1) * P * w);
      const Sel want = host_select(row, P * w, kmax);
      Sel got{std::vector<int>(op.begin() + (size_t)r * kmax, op.begin() + (size_t)(r + 1) * kmax),
              std::vector<uint16_t>(ok.begin() + (size_t)r * kmax, ok.begin() + (size_t)(r + 1) * kmax)};
      Sel w2 = want;
      for (int& p : w2.pos)
        if (p >= 0) p = cp[(size_t)r * P * w + p];
      return got == w2;
    };
    host_ok = host_ok && one(tp, tk, kT, kT, a_tp, a_tk) && one(pp, pk, kP, kP, a_pp, a_pk);
  }
  check(host_ok, "row-major merge == host merge", rows, name.c_str(), kT);
  check(std::any_of(a_tk.begin(), a_tk.end(), [](uint16_t k) { return k != 0; }), "picks are not empty", rows, name.c_str(), kT);
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && !std::strcmp(argv[1], "--quick");
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
  int smem_max = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  std::printf("shared memory per block (opt-in) %d bytes; the kernel uses %zu (candidates) and the dense kernel %zu\n",
              smem_max, (size_t)attn::kDenseCand * 8, (size_t)attn::kDenseSmem);
  check((size_t)smem_max >= attn::kDenseSmem, "the device allows the dense kernel's shared memory", 0, "", 0);
  std::vector<int> ns{262144, 1048576, 2097152, 4194304};
  if (quick) ns = {1048576, 4194304};
  for (int n : ns)
    for (int kmax : {512, 2048})
      for (int c = 0; c < kNumCases; ++c) {
        const std::vector<uint16_t> keys = make_keys(c, n, 1000 + c * 7 + n / 65536);
        const Sel want = host_select(keys, n, kmax);
        const Sel got = run(keys, n, kmax, true);
        check(got == want, "new selection == host top-k", n, kCases[c], kmax);
        if (n <= 1048576) {
          const Sel old = run(keys, n, kmax, false);
          check(old == got, "new == old selection (<= 1M)", n, kCases[c], kmax);
        } else if (c == 2 || c == 3) {  // control: the old block list cannot place blocks past 131072
          const Sel old = run(keys, n, kmax, false);
          std::printf("  control: the old selection %s the host top-k at n=%d kmax=%d (%s); it must differ\n",
                      old == want ? "EQUALS" : "differs from", n, kmax, kCases[c]);
          check(!(old == want), "control: old selection is wrong past 1M", n, kCases[c], kmax);
        }
      }
  for (int rows : {1, 6, 16}) merge_layout(rows, 4);
  merge_layout(6, 2);  // another part count: the layout argument is the part count, not a constant
  std::printf("%d checks, %d failed\n", g_checks, g_bad);
  return g_bad ? 1 : 0;
}
