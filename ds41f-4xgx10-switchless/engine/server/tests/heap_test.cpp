// common/heap.h against a heap that holds free pages: many small blocks allocated, nearly all freed, and the resident
// size read from /proc/self/statm before and after heap::release(). Built with -DPICO_OLD_RULES the case fails (exit 1):
// release() gives nothing back. Exit 0 when every case passes.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/heap.h"

static size_t rss() {
  size_t pages = 0, res = 0;
  FILE* f = std::fopen("/proc/self/statm", "r");
  if (f) {
    if (std::fscanf(f, "%zu %zu", &pages, &res) != 2) res = 0;
    std::fclose(f);
  }
  return res * 4096;
}

int main() {
  heap::limit_arenas();
  std::vector<char*> blocks;
  for (int i = 0; i < 100000; ++i) {  // 100,000 blocks of 4 KiB: about 400 MiB, below the mapping threshold each
    char* p = static_cast<char*>(std::malloc(4000));
    for (int k = 0; k < 4000; k += 4096 / 2) p[k] = 1;
    p[3999] = 1;
    blocks.push_back(p);
  }
  for (size_t i = 0; i < blocks.size(); ++i)
    if (i % 1000 != 0) std::free(blocks[i]);  // 99% freed, every thousandth block stays live and pins the heap's top
  const size_t before = rss();
  heap::release();
  const size_t after = rss();
  std::printf("resident before release %.0f MiB, after %.0f MiB; heap in use %.1f MiB\n", before / 1048576.0, after / 1048576.0,
              heap::in_use() / 1048576.0);
  int bad = 0;
  if (!(before > after + (200ull << 20))) std::printf("FAIL: release returns the free pages of a fragmented heap (at least 200 MiB)\n"), ++bad;
  if (!(heap::in_use() < (16ull << 20))) std::printf("FAIL: the heap in use is what is live (a few MiB)\n"), ++bad;
  if (bad) return 1;
  std::printf("2 cases ok\n");
  return 0;
}
