// Proves the drafter's plain RoPE table can alias the model's: the model's table (attn::rope_table_rows(max_ctx, 8) rows of
// attn::rope_table_build(.., false)) and a table of the drafter's own size are equal byte for byte over their common rows at
// --max-ctx 4194304 (above the table cap), 1048576 (just under it), 1048572 (the cap less 4) and 65536, and the model's
// table covers every row the drafter's own would. Prints the first differing row if any and exits non-zero on a difference.
// Run (one GPU, about 0.6 GiB): ./rope_alias_test
#include <cstdio>
#include <cstring>
#include <vector>

#include "attn/attn.h"
#include "common/check.h"

namespace {

std::vector<float2> build(int rows) {
  float2* d;
  CUDA_CHECK(cudaMalloc(&d, (size_t)rows * 32 * sizeof(float2)));
  attn::rope_table_build(d, rows, false, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float2> h((size_t)rows * 32);
  CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(float2), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  return h;
}

int run(int max_ctx) {
  const int main_old = attn::rope_table_rows(max_ctx);     // the model's table without the drafter's extra rows
  const int main_new = attn::rope_table_rows(max_ctx, 8);  // the model's table with the drafter's extra rows
  const int drafter = attn::rope_table_rows(max_ctx, 8);   // the drafter's own table
  const std::vector<float2> m = build(main_new), d = build(drafter);
  const int common = main_old < drafter ? main_old : drafter;
  int fails = 0;
  long long first = -1;
  for (int r = 0; r < common && first < 0; ++r)
    if (std::memcmp(&m[(size_t)r * 32], &d[(size_t)r * 32], 32 * sizeof(float2)) != 0) first = r;
  const bool covers = main_new >= drafter;
  const bool same = std::memcmp(m.data(), d.data(), (size_t)common * 32 * sizeof(float2)) == 0;
  const bool same_all = main_new == drafter && std::memcmp(m.data(), d.data(), m.size() * sizeof(float2)) == 0;
  std::printf("max_ctx %d: model rows %d (was %d), drafter rows %d; compared %d common rows: %s; whole tables %s; model covers "
              "drafter %s\n", max_ctx, main_new, main_old, drafter, common, same ? "equal" : "DIFFER",
              same_all ? "equal" : "DIFFER", covers ? "yes" : "NO");
  if (first >= 0) {
    std::printf("  first differing row %lld\n", first);
    ++fails;
  }
  if (!covers || !same || !same_all) ++fails;
  return fails;
}

}  // namespace

int main() {
  int fails = 0;
  for (int c : {4194304, 1048576, 1048572, 65536}) fails += run(c);
  std::printf(fails ? "FAILED (%d)\n" : "all ok\n", fails);
  return fails != 0;
}
