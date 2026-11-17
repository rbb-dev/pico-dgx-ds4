// model/tokwin.h against the full history it replaces: every Engram hash a windowed history gives equals the one a
// never-trimmed vector gives, over a random run of the operations the engine does to a history (a prompt set at some
// position, prefill chunks, decode and verify steps with rewinds and draft rejections, trims between commands) and at
// the named positions; and 32 units cycled through 2M-token prompts hold memory to the live windows, giving capacity
// back on reset. Host only. Built with -DPICO_OLD_RULES (no trimming) the memory checks fail (exit 1).
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "engram/engram.h"
#include "model/tokwin.h"

using model::TokWin;

namespace {
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what, long long a = 0, long long b = 0) {
  ++g_checks;
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s (%lld, %lld)\n", what, a, b);
  }
}

constexpr int kVocab = 129280;
engram::HashMap make_map() {
  using namespace engram;
  std::mt19937_64 rng(11);
  HashMap m;
  m.n_tokens = kVocab, m.vocab = 9973, m.pad = 5;
  for (int l = 0; l < kLayers; ++l) {
    m.num_embeddings[l] = 1 << 20;
    for (int k = 0; k < kMaxNgram; ++k) m.mult[l][k] = int64_t(rng() % 100000) * 2 + 1;
    for (int k = 0; k < kOrders; ++k)
      for (int h = 0; h < kHeads; ++h) m.prime[l][k][h] = 100003 + 2 * int64_t(rng() % 5000) + 2 * (k * kHeads + h);
    for (int c = 0; c < kCols; ++c) m.offset[l][c] = int64_t(c) << 20;
  }
  m.token_map.resize(kVocab);
  for (int t = 0; t < kVocab; ++t) m.token_map[t] = int32_t(rng() % m.vocab);
  return m;
}

const engram::HashMap& map() {
  static const engram::HashMap m = make_map();
  return m;
}
constexpr size_t kRow = size_t(engram::kLayers) * engram::kCols;

// The ids the engine hashes for positions [first, first + n): from the window, and from the full vector.
bool same(const TokWin& w, const std::vector<int32_t>& full, int first, int n) {
  std::vector<uint32_t> a(size_t(n) * kRow), b(size_t(n) * kRow);
  engram::hash_positions(map(), w.data(), nullptr, first, n, a.data(), w.base);
  engram::hash_positions(map(), full.data(), nullptr, first, n, b.data());
  return a == b;
}
std::vector<int32_t> random_ids(std::mt19937& rng, size_t n) {
  std::vector<int32_t> v(n);
  for (int32_t& t : v) t = int32_t(rng() % kVocab);
  return v;
}
// The engine's two ways to put ids in a history, on both: a prompt from pos0 (hashed from pos0 + 3 on, as the server
// sends q - 3 and prefills from q) and a step's rows after a cut to pos0.
void set_tokens(TokWin& w, std::vector<int32_t>& full, int pos0, const std::vector<int32_t>& t) {
  w.assign(pos0, (int)t.size(), t.data());
  full.resize(pos0 + t.size());
  std::copy(t.begin(), t.end(), full.begin() + pos0);
}
void put(TokWin& w, std::vector<int32_t>& full, int pos0, const std::vector<int32_t>& t) {
  check(w.put(pos0, (int)t.size(), t.data()), "put inside the window", pos0, w.base);
  full.resize(pos0);
  full.insert(full.end(), t.begin(), t.end());
}
}  // namespace

int main() {
  std::mt19937 rng(5);
  {  // the named positions, history kept whole
    const std::vector<int32_t> s = random_ids(rng, 1000200);
    TokWin w;
    std::vector<int32_t> full;
    set_tokens(w, full, 0, s);
    for (const int p : {0, 1, 2, 3, 255, 256, 1000003}) check(same(w, full, p, 1), "hash at a named position, window whole", p);
    check(same(w, full, 0, 1000200), "hash of every position, window whole");
  }
  {  // a prompt of 1,000,010 ids, prefilled, trimmed: the positions the next steps ask for
    const int N = 1000010;
    const std::vector<int32_t> s = random_ids(rng, N + 300);
    TokWin w;
    std::vector<int32_t> full;
    set_tokens(w, full, 0, std::vector<int32_t>(s.begin(), s.begin() + N));
    w.prefilled(N - 100);
    w.trim();
    check(w.v.size() == (size_t)N, "no trim while the prompt's prefill is short of its end", (long long)w.v.size(), N);
    w.prefilled(N);
    w.trim();
    check(w.v.size() <= (size_t)model::kHistKeep, "trimmed to the last ids after the prefill", (long long)w.v.size());
    check(w.end() == N, "the end is where it was", w.end(), N);
    check(w.v.capacity() <= model::kHistCapKeep, "capacity given back", (long long)w.v.capacity());
    for (const int p : {1000003, N - 4093, N - 1}) check(same(w, full, p, 1), "hash at a position of the window", p);
    // decode steps with draft rejections: rows of 6 at pos0, then a rewind by up to 5 with new ids, across the boundary
    int pos = N;
    for (int i = 0; i < 40; ++i) {
      const int R = 1 + int(rng() % 6);
      put(w, full, pos, std::vector<int32_t>(s.begin() + N + i, s.begin() + N + i + R));
      check(same(w, full, pos, R), "hash of a verify step's rows", pos, R);
      const int acc = int(rng() % (R + 1));  // rows kept; the rest rejected
      pos += acc;
      put(w, full, pos, random_ids(rng, 1 + rng() % 6));
      check(same(w, full, pos, 1), "hash of the rows after a rejection", pos);
      pos = w.end();
    }
    w.trim();
    check(w.v.size() <= 2 * (size_t)model::kHistKeep, "a decoded window stays small", (long long)w.v.size());
    // a rewind across the boundary the trim made: end - 5
    const int e = w.end();
    put(w, full, e - 5, random_ids(rng, 4));
    check(same(w, full, e - 5, 4), "hash after a rewind below the trim boundary's reach", e - 5);
    // a zero-pad: rows put past the end
    put(w, full, w.end() + 40, random_ids(rng, 3));
    check(same(w, full, w.end() - 3, 3), "hash after a zero-pad", w.end());
    check(!w.resize(w.base - 1), "a cut below the window is refused");
  }
  {  // set_tokens below the window starts it again; a prompt from q - 3 hashes equal from q
    const int N = 200000;
    TokWin w;
    std::vector<int32_t> full;
    set_tokens(w, full, 0, random_ids(rng, N));
    w.prefilled(N), w.trim();
    for (const int q : {5, 4096, 100000, N - 4000, N - 100, N + 50}) {
      const int pos0 = std::max(0, q - 3);
      const int n = std::max(q + 400, N) - pos0;
      set_tokens(w, full, pos0, random_ids(rng, n));
      check(same(w, full, q, 400), "hash from q after a prompt set at q - 3", q, w.base);
      w.prefilled(pos0 + n), w.trim();
      check(same(w, full, pos0 + n - 5, 1), "hash at the prompt's tail after the trim", q);
    }
    set_tokens(w, full, 0, random_ids(rng, 70));
    check(same(w, full, 0, 70), "a short prompt after a long one", w.base);
  }
  {  // a random run of operations
    for (int run = 0; run < 20; ++run) {
      TokWin w;
      std::vector<int32_t> full;
      for (int op = 0; op < 300; ++op) {
        const int r = int(rng() % 10), e = (int)full.size();
        if (r < 2) {  // a prompt, from q - 3 where q is anywhere up to just past the end
          const int q = int(rng() % (e + 3)), pos0 = std::max(0, q - 3), n = (q ? q - pos0 : 0) + 1 + int(rng() % 9000);
          set_tokens(w, full, pos0, random_ids(rng, n));
          if (!same(w, full, q, std::min(n - (q - pos0), 64))) check(false, "random: hash from q", run, op);
          if (rng() % 4) w.prefilled(pos0 + n);
        } else if (r < 9 && e >= 6) {  // a step: a rewind of up to 5 and rows
          const int pos0 = e - int(rng() % 6), R = 1 + int(rng() % 6);
          if (w.base > 0 && pos0 - 3 < w.base) continue;  // steps never ask below a prompt's own start
          put(w, full, pos0, random_ids(rng, R));
          if (!same(w, full, pos0, R)) check(false, "random: hash of a step", run, op);
        } else {
          w.trim();
        }
        if (rng() % 7 == 0) w.trim();
      }
      check(true, "random run");
    }
  }
  {  // the memory: 32 units, each given a 2M-token prompt in turn, prefilled, decoded a little, trimmed
    const int N = 2000000;
    const std::vector<int32_t> prompt = random_ids(rng, N);
    TokWin units[32];
    auto rss = [] {
      long pages = 0, res = 0;
      if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0;
        std::fclose(f);
      }
      return res * 4096.0 / (1 << 20);
    };
    const double before = rss();
    double peak = before;
    for (int cycle = 0; cycle < 3; ++cycle)
      for (TokWin& w : units) {
        w.assign(0, N, prompt.data());
        w.prefilled(N);
        w.trim();
        for (int i = 0; i < 64; ++i) w.put(w.end(), 1, prompt.data() + i);
        peak = std::max(peak, rss());
      }
    const double after = rss();
    std::printf("host RSS: %.1f MiB before, %.1f MiB after 96 prompts of 2M ids over 32 units, peak %.1f MiB\n", before, after,
                peak);
    check(after - before <= 128.0, "host RSS growth over 32 units of 2M-token prompts at most 128 MiB", (long long)(after - before));
    size_t held = 0;
    for (const TokWin& w : units) held += w.v.capacity();
    check(held * 4 <= (size_t)128 << 20, "capacity the 32 units hold at most 128 MiB", (long long)(held * 4 >> 20));
    TokWin big;
    big.assign(0, N, prompt.data());
    check(big.v.capacity() > model::kHistCapKeep, "a 2M history's capacity is large");
    big.clear();
    check(big.v.capacity() == 0 && big.base == 0 && big.v.empty(), "capacity returned on reset", (long long)big.v.capacity());
  }
  if (g_fail) {
    std::printf("%d of %d checks failed\n", g_fail, g_checks);
    return 1;
  }
  std::printf("all %d checks passed\n", g_checks);
  return 0;
}
