// Copy drafting on a GB10 (the CMake target copy_kernel_test; one rank, no ring, no weights; allocations of a few MiB):
// the device's mirror, table and lookups against the host reference of model/copy.h, bit for bit. The device builds the
// table with parallel writers (atomicMax) where the host indexes in order: the tables must come out equal. Random
// streams, repetitive streams, a rewrite at a lower position, decode appends, and lookups with chain drafts.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/check.h"
#include "model/copy_dev.h"

using namespace model::copy;

namespace {

uint64_t g_rng = 0xD1B54A32D192ED03ull;
uint32_t rnd() {
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33);
}
int fails = 0, checks = 0;
void check(bool ok, const char* what, int it = 0) {
  ++checks;
  if (ok) return;
  if (++fails <= 30) std::printf("FAIL (iteration %d): %s\n", it, what);
}

std::vector<int32_t> stream(int n, int alpha, int phrase_p) {
  std::vector<int32_t> s;
  while ((int)s.size() < n) {
    if (s.size() > 20 && (int)(rnd() % 100) < phrase_p) {
      const int len = 8 + rnd() % 40, from = rnd() % (s.size() - 10);
      for (int i = 0; i < len && (int)s.size() < n && from + i < (int)s.size(); ++i)
        s.push_back(rnd() % 40 == 0 ? (int32_t)(rnd() % alpha) : s[from + i]);
    } else {
      s.push_back(rnd() % alpha);
    }
  }
  return s;
}

// the host's unit with the same shape as the device's
struct Host {
  std::vector<int32_t> tok;
  std::vector<uint32_t> tab;
  Unit u{};
  Host(const Unit& d) : tok(d.cap, 0), tab((size_t)d.mask + 1, 0) {
    u.tok = tok.data(), u.tab = tab.data(), u.mask = d.mask, u.cap = d.cap, u.len = 0;
  }
  void set(int pos0, const std::vector<int32_t>& t) {
    const bool rewrite = pos0 < u.len;
    if (rewrite) std::fill(tab.begin(), tab.end(), 0u);
    const int n = std::min((int)t.size(), u.cap - pos0);
    for (int i = 0; i < n; ++i) tok[pos0 + i] = t[i];
    u.len = pos0 + n;
    index_range(u, rewrite ? 0 : pos0, u.len - 1);
  }
};

void same_state(const Index& ix, int unit, const Host& h, const char* what, int it) {
  const Unit d = ix.view(unit);
  check(d.len == h.u.len, what, it);
  const auto tab = ix.table_copy(unit);
  check(tab == h.tab, "the table equals the host's", it);
  const auto tok = ix.tokens_copy(unit, h.u.len);
  check(std::equal(tok.begin(), tok.end(), h.tok.begin()), "the mirror equals the host's", it);
}

}  // namespace

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {  // the plan degrades inside a small arena and the units get distinct blocks
    const int ctx[3] = {20000, 6000, 0};
    Index ix(ctx, 3, (size_t)4 << 20, s);
    const auto& p = ix.plan();
    check(p[0].cap == 20000 && p[1].cap == 6000 && p[2].cap == 0, "plan: caps");
    check(ix.bytes() <= ((size_t)4 << 20) && ix.bytes() > 0, "plan: within the arena");
    const Unit a = ix.view(0), b = ix.view(1);
    check(a.tok != b.tok && a.tab != b.tab && ix.view(2).cap == 0, "units have blocks of their own");
  }
  long total_q = 0, hits = 0;
  for (int it = 0; it < 60; ++it) {
    const int ctx[2] = {6000, 3000};
    Index ix(ctx, 2, (size_t)8 << 20, s);
    const int unit = it % 2;
    Host h(ix.view(unit));
    const int n = 200 + rnd() % 2500;
    const int alpha = it % 3 == 0 ? 3 + rnd() % 6 : 50 + rnd() % 1000;
    const auto H = stream(n, alpha, it % 2 ? 35 : 6);
    ix.set_tokens(unit, 0, n, H.data(), s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    h.set(0, H);
    same_state(ix, unit, h, "set: the length", it);

    // decode appends: a step at pos0 = len - 1 with rows and a bonus, as spec.cu's kernels do it
    for (int st = 0; st < 12; ++st) {
      const int pos0 = h.u.len - 1, count = 1 + rnd() % 8;
      if (pos0 + count + 2 >= h.u.cap) break;
      std::vector<int32_t> rows(count);
      for (int r = 0; r < count; ++r) rows[r] = r == 0 ? h.tok[pos0] : (rnd() % 3 ? h.tok[std::max(0, pos0 - 20 + r)] : (int32_t)(rnd() % alpha));
      const int32_t bonus = rnd() % alpha;
      append_step(h.u, pos0, count, rows.data(), bonus);
      ix.append_on_device(unit, pos0, count, rows.data(), bonus, s);
    }
    same_state(ix, unit, h, "append: the length", it);

    // lookups with chain drafts
    std::vector<Index::Query> q;
    for (int i = 0; i < 300; ++i) {
      Index::Query x{};
      x.unit = unit;
      x.P = h.u.len - 1 - rnd() % 60;  // near the end, as a decode step is: a later gram would take a slot's entry from it
      x.k = rnd() % 3 ? 0 : rnd() % 6;
      for (int j = 0; j < x.k; ++j) x.d[j] = rnd() % 3 ? h.tok[std::min(h.u.len - 1, x.P + 1 + j)] : (int32_t)(rnd() % alpha);
      x.maxm = 1 + rnd() % 7;
      x.minlen = rnd() % 4 ? kOrder : kOrder + rnd() % 16;
      q.push_back(x);
    }
    std::vector<int32_t> out, counts, matches;
    ix.propose_on_device(q, out, counts, matches, s);
    for (size_t i = 0; i < q.size(); ++i) {
      int32_t want[kMaxDrafts];
      Match mt;
      const int m = propose(h.u, q[i].P, q[i].d, q[i].k, q[i].maxm, q[i].minlen, want, &mt);
      ++total_q;
      hits += m > 0;
      check(counts[i] == m && matches[3 * i] == mt.e && matches[3 * i + 1] == mt.len && matches[3 * i + 2] == mt.bucket,
            "device proposal's count and match equal the host's", it);
      check(std::equal(want, want + m, out.begin() + i * kMaxDrafts), "device proposal's tokens equal the host's", it);
    }

    // a decode step as the engine has it at P: the prompt H[0..P-1] set, one step appended (row 0 and the bonus), the
    // lookup at P: the table holds no gram that ends at or after P, so a copy in the history is found
    for (int rep = 0; rep < 6; ++rep) {
      const int P = 20 + rnd() % (n - 21);
      ix.set_tokens(unit, 0, P, H.data(), s);
      h.set(0, std::vector<int32_t>(H.begin(), H.begin() + P));
      const int32_t row0 = H[P - 1];
      append_step(h.u, P - 1, 1, &row0, H[P]);
      ix.append_on_device(unit, P - 1, 1, &row0, H[P], s);
      same_state(ix, unit, h, "step: the length", it);
      std::vector<Index::Query> q2;
      for (int i = 0; i < 4; ++i) {
        Index::Query x{};
        x.unit = unit, x.P = P, x.k = i == 0 ? 0 : rnd() % 4, x.maxm = 1 + rnd() % 7, x.minlen = kOrder;
        for (int j = 0; j < x.k; ++j) x.d[j] = rnd() % 4 ? H[std::min(n - 1, P + 1 + j)] : (int32_t)(rnd() % alpha);
        q2.push_back(x);
      }
      ix.propose_on_device(q2, out, counts, matches, s);
      for (size_t i = 0; i < q2.size(); ++i) {
        int32_t want[kMaxDrafts];
        Match mt;
        const int m = propose(h.u, P, q2[i].d, q2[i].k, q2[i].maxm, kOrder, want, &mt);
        ++total_q, hits += m > 0;
        check(counts[i] == m && matches[3 * i] == mt.e && matches[3 * i + 1] == mt.len, "step lookup equals the host's", it);
        check(std::equal(want, want + m, out.begin() + i * kMaxDrafts), "step lookup's tokens equal the host's", it);
      }
    }
    // the unit as the first phase left it is gone: the rewrite below starts from the whole history again
    ix.set_tokens(unit, 0, n, H.data(), s);
    h.set(0, H);

    // a rewrite at a lower position (a new request on the unit): the table made again
    const int p = 10 + rnd() % (h.u.len / 2);
    const auto T = stream(100 + rnd() % 800, alpha, 30);
    ix.set_tokens(unit, p, (int)T.size(), T.data(), s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    h.set(p, T);
    same_state(ix, unit, h, "rewrite: the length", it);
  }
  std::printf("%ld lookups compared, %ld with a proposal\n", total_q, hits);
  check(hits > 300, "the streams have copies to find (a test that finds none tests nothing)");
  std::printf("%d checks, %d failed\n", checks, fails);
  return fails ? 1 : 0;
}
