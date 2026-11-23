// server/ids_file.h: the ids of idle conversations past their cap go to files and come back equal; which conversations go
// (the least recently used, until the sum is within the cap); a freed conversation's file is gone; a damaged file is refused;
// the directory is emptied at the start. Built twice: with -DPICO_OLD_RULES (the policy without the rule: nothing goes out, a
// removed conversation's file stays) the cases marked "red" fail; without it every case passes. Host only: no engine, no GPU.
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../ids_file.h"

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}
bool exists(const std::string& p) {
  struct stat s;
  return stat(p.c_str(), &s) == 0;
}
}  // namespace

int main() {
  using idsfile::Cand;
  const uint64_t MiB = 1ull << 20;

  // which conversations go: ten idle ones of 50M positions (200 MB each, 2,000 MB in all) against the real cap of 1,024 MiB
  {
    std::vector<Cand> c;
    for (int i = 0; i < 10; ++i) c.push_back({100 + i, (uint64_t)(50 - i), 50000000});  // handle 109 is the least recently used
    uint64_t total = 0;
    for (const Cand& x : c) total += x.positions * 4;
    const std::vector<int> out = idsfile::pick_out(c, total, 1024 * MiB);
    check(!out.empty(), "red: a sum of 2,000 MB over the 1,024 MiB cap sends ids out");
    uint64_t left = total;
    for (int h : out) left -= 50000000ull * 4;
    check(!out.empty() && left <= 1024 * MiB && left + 200000000ull > 1024 * MiB, "red: enough go to bring the sum within the cap, and no more");
    check(out.size() == 5 && out[0] == 109 && out[4] == 105, "red: the least recently used go first (109 down to 105)");
    // the cap's own arithmetic: 1,024 MiB is 268,435,456 positions at 4 B
    std::vector<Cand> one{{7, 1, 268435457}};
    check(idsfile::pick_out(one, 268435457ull * 4, 1024 * MiB).size() == 1, "red: one position over 268,435,456 sends the conversation out");
    check(idsfile::pick_out(one, 268435456ull * 4, 1024 * MiB).empty(), "exactly the cap sends nothing");
    std::vector<Cand> tie{{9, 5, 10}, {4, 5, 10}, {6, 5, 10}};
    const std::vector<int> t = idsfile::pick_out(tie, 120, 79);
    check(t.size() == 2 && t[0] == 4 && t[1] == 6, "red: on a tie in use the lowest handle goes first");
    check(idsfile::pick_out({}, 10 * MiB, MiB).empty(), "no idle conversation: nothing can go");
  }

  // the files
  char tmpl[] = "/tmp/ids_file_test.XXXXXX";
  const std::string root = mkdtemp(tmpl) ? tmpl : "";
  check(!root.empty(), "fixture: a scratch directory");
  const std::string dir = root + "/ids";
  std::string why;
  {
    // a stale file of an earlier run goes when the directory is prepared
    mkdir(dir.c_str(), 0700);
    std::FILE* f = std::fopen(idsfile::path(dir, 5).c_str(), "wb");
    if (f) std::fputs("old", f), std::fclose(f);
    check(idsfile::prepare(dir, &why) && !exists(idsfile::path(dir, 5)), "the start empties the directory of an earlier run's files");
  }
  std::vector<int32_t> ids(1000003);
  uint32_t x = 12345;
  for (int32_t& v : ids) x = x * 1664525u + 1013904223u, v = (int32_t)(x % 129280);
  {
    check(idsfile::write(dir, 42, ids, &why), "the ids of a conversation are written");
    check(exists(idsfile::path(dir, 42)) && !exists(idsfile::path(dir, 42) + ".new"), "its file is there and no temporary is left");
    std::vector<int32_t> back;
    check(idsfile::read(dir, 42, &back, &why) && back == ids, "they read back equal");
    struct stat s;
    stat(idsfile::path(dir, 42).c_str(), &s);
    check((uint64_t)s.st_size == sizeof(idsfile::Head) + ids.size() * 4 && (s.st_mode & 0777) == 0600, "the file is a header and 4 bytes a position, mode 0600");
    std::vector<int32_t> none;
    check(!idsfile::read(dir, 43, &none, &why) && why.empty(), "a conversation with no file reads as missing, with no complaint");
    // a damaged byte, a short file
    const std::string p = idsfile::path(dir, 42);
    std::FILE* f = std::fopen(p.c_str(), "r+b");
    if (f) {
      std::fseek(f, (long)(sizeof(idsfile::Head) + 4000), SEEK_SET);
      const int was = std::fgetc(f);
      std::fseek(f, (long)(sizeof(idsfile::Head) + 4000), SEEK_SET);
      std::fputc(~was & 0xff, f);
      std::fclose(f);
    }
    std::vector<int32_t> bad;
    check(!idsfile::read(dir, 42, &bad, &why) && bad.empty() && !why.empty(), "a file with a damaged byte is refused");
    check(idsfile::write(dir, 44, ids, &why) && truncate(idsfile::path(dir, 44).c_str(), 4096) == 0 && !idsfile::read(dir, 44, &bad, &why),
          "a short file is refused");
    // freed conversations
    idsfile::write(dir, 42, ids, &why);
    idsfile::remove(dir, 42);
    check(!exists(idsfile::path(dir, 42)), "red: a freed conversation's file is gone");
  }
  // the policy and the files together: what pick_out chooses is written, freed, read back equal
  {
    std::vector<std::vector<int32_t>> mem(4, std::vector<int32_t>(250000, 7));
    std::vector<Cand> c;
    uint64_t total = 0;
    for (int i = 0; i < 4; ++i) mem[i][i] = i + 1, c.push_back({200 + i, (uint64_t)(10 - i), mem[i].size()}), total += mem[i].size() * 4;
    const std::vector<int> out = idsfile::pick_out(c, total, 2 * 1000000ull);  // 4 MB in memory against a 2 MB cap
    bool equal = !out.empty();
    for (int h : out) {
      auto& v = mem[h - 200];
      const std::vector<int32_t> copy = v;
      equal = equal && idsfile::write(dir, h, v, &why);
      v.clear(), v.shrink_to_fit();
      std::vector<int32_t> back;
      equal = equal && idsfile::read(dir, h, &back, &why) && back == copy;
      idsfile::remove(dir, h);
      equal = equal && out.size() == 2 && !exists(idsfile::path(dir, h));
    }
    check(equal, "red: the sum over the cap is written, freed, read back equal, and the files go with the conversations");
  }
  // a conversation whose ids cannot be had back is freed whole
  {
    std::vector<int32_t> v;
    std::string w;
    idsfile::write(dir, 55, ids, &w);
    const bool ok = idsfile::read(dir, 55, &v, &w);
    check(idsfile::back_verdict(ok, v.size(), ids.size()) == idsfile::Back::kUse, "a file that reads back whole with the recorded count is used");
    unlink(idsfile::path(dir, 55).c_str());  // the file is lost
    std::vector<int32_t> none;
    const bool lost = idsfile::read(dir, 55, &none, &w);
    check(idsfile::back_verdict(lost, none.size(), ids.size()) == idsfile::Back::kFree, "red: a lost file frees the conversation whole");
    idsfile::write(dir, 56, ids, &w);
    std::vector<int32_t> other;
    const bool ok2 = idsfile::read(dir, 56, &other, &w);
    check(ok2 && idsfile::back_verdict(ok2, other.size(), ids.size() + 1) == idsfile::Back::kFree, "red: a file holding another count than the recorded one frees it");
    std::FILE* f = std::fopen(idsfile::path(dir, 56).c_str(), "r+b");
    if (f) {
      std::fseek(f, (long)(sizeof(idsfile::Head) + 100), SEEK_SET);
      const int was = std::fgetc(f);
      std::fseek(f, (long)(sizeof(idsfile::Head) + 100), SEEK_SET);
      std::fputc(~was & 0xff, f);
      std::fclose(f);
    }
    std::vector<int32_t> bad;
    const bool ok3 = idsfile::read(dir, 56, &bad, &w);
    check(idsfile::back_verdict(ok3, bad.size(), ids.size()) == idsfile::Back::kFree, "red: a damaged file frees the conversation whole");
    unlink(idsfile::path(dir, 56).c_str());
  }
  for (int h : {42, 44, 200, 201, 202, 203}) unlink(idsfile::path(dir, h).c_str());
  rmdir(dir.c_str());
  rmdir(root.c_str());
  std::printf("%s\n", fails ? "ids_file_test FAILED" : "ids_file_test passed");
  return fails ? 1 : 0;
}
