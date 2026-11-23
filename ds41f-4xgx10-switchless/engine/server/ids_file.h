// The server's token-id histories beyond their cap (--ids-host-mib): which idle conversations' ids go to a file, the file's
// format, writing it, reading it back and removing it (host only: no engine state; tested by tests/ids_file_test.cpp).
// A conversation's ids are 4 bytes a position in rank 0's memory. When the sum passes the cap, the least recently used idle
// conversations' ids are written to <dir>/<handle>.ids and freed; they are read back before a request uses the conversation.
// A conversation freed for any reason removes its file; the directory is emptied at the start. Nothing is lost: a file that
// does not check out (length, sum) reads back as a failure and the caller treats the conversation as it treats a lost one.
// File: 8-byte magic, u64 count, u64 FNV-1a of the ids' bytes, then the ids (int32, little endian).
// The kept index (prefix.h) names a conversation's file by its count and sum: the file outlives the conversation's return to memory
// (it is removed when the conversation is freed, and replaced when the conversation's history has changed), and the files the kept
// index names survive a restart (sweep).
//
// PICO_OLD_RULES (never defined by the engine) builds the policy without the rule: nothing is moved out and a removed
// conversation's file stays, so the tests can show they fail on it.
#pragma once
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../common/keepfile.h"

namespace idsfile {

constexpr char kMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'I', 'D'};
struct Head {
  char magic[8];
  uint64_t count;
  uint64_t sum;
};
constexpr uint64_t kIdBytes = 4;

// <handle>.ids (seq 0: the name the first form of this file used) or <handle>.<seq>.ids. The server writes every version of a
// conversation's ids under a new seq (Keeper), so a file a kept index names is never changed.
inline std::string path(const std::string& dir, int handle, uint64_t seq = 0) {
  return dir + "/" + std::to_string(handle) + (seq ? "." + std::to_string(seq) : std::string()) + ".ids";
}
using Named = std::set<std::pair<int, uint64_t>>;  // (handle, seq) of files to keep

// The directory made (0700, ours).
inline bool ensure_dir(const std::string& dir, std::string* why) {
  if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
    *why = dir + ": " + std::strerror(errno);
    return false;
  }
  return keepfile::private_dir(dir, why);
}

// The files an earlier run left removed, except those named in `keep` (the files the kept index brought back).
inline bool sweep(const std::string& dir, const Named& keep, std::string* why) {
  DIR* d = opendir(dir.c_str());
  if (!d) {
    *why = dir + ": " + std::strerror(errno);
    return false;
  }
  while (const dirent* e = readdir(d)) {
    const std::string n = e->d_name;
    const auto ends = [&](const char* tail) { const size_t k = std::strlen(tail); return n.size() >= k && n.compare(n.size() - k, k, tail) == 0; };
    if (ends(".ids.new")) {
      unlink((dir + "/" + n).c_str());
    } else if (ends(".ids")) {
      char* end = nullptr;
      const long h = std::strtol(n.c_str(), &end, 10);
      unsigned long long seq = 0;
      bool ok = end != n.c_str();
      if (ok && *end == '.' && end[1] >= '0' && end[1] <= '9') seq = std::strtoull(end + 1, &end, 10);
      ok = ok && std::strcmp(end, ".ids") == 0 && keep.count({(int)h, (uint64_t)seq});
      if (!ok) unlink((dir + "/" + n).c_str());
    }
  }
  closedir(d);
  return true;
}

// The directory made and emptied of files an earlier run left.
inline bool prepare(const std::string& dir, std::string* why) { return ensure_dir(dir, why) && sweep(dir, {}, why); }

// ids written to the conversation's file (under a temporary name, then renamed: the file is whole or absent). Buffered: the
// page cache holds it and is reclaimable, so there is no sync.
inline bool write(const std::string& dir, int handle, const std::vector<int32_t>& ids, std::string* why, uint64_t* sum = nullptr, uint64_t seq = 0) {
  const std::string p = path(dir, handle, seq), tmp = p + ".new";
  unlink(tmp.c_str());
  const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) {
    *why = tmp + ": " + std::strerror(errno);
    return false;
  }
  Head h{};
  std::memcpy(h.magic, kMagic, sizeof h.magic);
  h.count = ids.size();
  h.sum = keepfile::fnv(ids.data(), ids.size() * kIdBytes);
  if (sum) *sum = h.sum;
  auto all = [&](const void* data, size_t n) {
    const auto* b = static_cast<const uint8_t*>(data);
    size_t done = 0;
    while (done < n) {
      const ssize_t r = ::write(fd, b + done, n - done);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) return false;
      done += (size_t)r;
    }
    return true;
  };
  const bool ok = all(&h, sizeof h) && all(ids.data(), ids.size() * kIdBytes);
  const int err = errno;
  close(fd);
  if (!ok || rename(tmp.c_str(), p.c_str()) != 0) {
    *why = tmp + ": " + std::strerror(ok ? errno : err);
    unlink(tmp.c_str());
    return false;
  }
  return true;
}

// The file's header: its count and sum, checked against the file's size (a restart's look at a file the kept index names, without
// reading the ids). false: why says what (a missing file says "").
inline bool peek(const std::string& dir, int handle, uint64_t* count, uint64_t* sum, std::string* why, uint64_t seq = 0) {
  const std::string p = path(dir, handle, seq);
  const int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    *why = errno == ENOENT ? "" : p + ": " + std::strerror(errno);
    return false;
  }
  struct stat s;
  Head h{};
  const bool ok = fstat(fd, &s) == 0 && S_ISREG(s.st_mode) && (uint64_t)s.st_size >= sizeof h && ::read(fd, &h, sizeof h) == (ssize_t)sizeof h &&
                  std::memcmp(h.magic, kMagic, sizeof h.magic) == 0 && (uint64_t)s.st_size == sizeof h + h.count * kIdBytes;
  close(fd);
  if (!ok) {
    *why = p + " is not an ids file of this server";
    return false;
  }
  *count = h.count, *sum = h.sum;
  return true;
}

// The ids back, checked against the file's own count, size and sum. false: why says what (a missing file says "").
inline bool read(const std::string& dir, int handle, std::vector<int32_t>* out, std::string* why, uint64_t seq = 0) {
  const std::string p = path(dir, handle, seq);
  const int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    *why = errno == ENOENT ? "" : p + ": " + std::strerror(errno);
    return false;
  }
  struct stat s;
  Head h{};
  bool ok = fstat(fd, &s) == 0 && S_ISREG(s.st_mode) && (uint64_t)s.st_size >= sizeof h;
  auto all = [&](void* data, size_t n) {
    auto* b = static_cast<uint8_t*>(data);
    size_t done = 0;
    while (done < n) {
      const ssize_t r = ::read(fd, b + done, n - done);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) break;
      done += (size_t)r;
    }
    return done == n;
  };
  if (ok) ok = all(&h, sizeof h) && std::memcmp(h.magic, kMagic, sizeof h.magic) == 0 && (uint64_t)s.st_size == sizeof h + h.count * kIdBytes;
  if (!ok) {
    *why = p + " is not an ids file of this server";
    close(fd);
    return false;
  }
  out->resize((size_t)h.count);
  ok = all(out->data(), (size_t)(h.count * kIdBytes)) && keepfile::fnv(out->data(), (size_t)(h.count * kIdBytes)) == h.sum;
  close(fd);
  if (!ok) {
    out->clear();
    *why = p + " does not check out (a short read or a wrong sum)";
  }
  return ok;
}

// The conversation's file removed (a freed conversation leaves none behind).
inline void remove(const std::string& dir, int handle, uint64_t seq = 0) {
#ifndef PICO_OLD_RULES
  unlink(path(dir, handle, seq).c_str());
#else
  (void)dir, (void)handle, (void)seq;  // without the rule: the file stays
#endif
}

// What a conversation becomes when its ids are asked for back: kUse (they read back whole, the count the server recorded), or kFree (the file is
// gone, does not check out, or holds another count: the history cannot be trusted, so the conversation is freed whole, its state in the store
// with it; a conversation with an empty history would never be matched again and would only hold memory).
enum class Back : uint8_t { kUse, kFree };
inline Back back_verdict(bool read_ok, uint64_t got, uint64_t expected) {
#ifdef PICO_OLD_RULES
  (void)read_ok, (void)got, (void)expected;
  return Back::kUse;  // without the rule: it stays, with an empty history
#else
  return read_ok && got == expected ? Back::kUse : Back::kFree;
#endif
}

// An idle conversation whose ids are in memory: its handle, the clock of its last use, its positions.
struct Cand {
  int handle = 0;
  uint64_t used = 0;
  uint64_t positions = 0;
};

// The handles whose ids go out when the ids in memory (`in_memory_bytes`, all conversations') pass `cap_bytes`: the least
// recently used first (the lowest handle on a tie), until what is left is within the cap. Only the candidates' own bytes can
// leave; a sum that stays over the cap with every candidate out is a running request's, which the cap does not touch.
inline std::vector<int> pick_out(std::vector<Cand> c, uint64_t in_memory_bytes, uint64_t cap_bytes) {
  std::vector<int> out;
#ifndef PICO_OLD_RULES
  std::sort(c.begin(), c.end(), [](const Cand& a, const Cand& b) { return a.used != b.used ? a.used < b.used : a.handle < b.handle; });
  for (const Cand& x : c) {
    if (in_memory_bytes <= cap_bytes) break;
    if (x.positions == 0) continue;
    out.push_back(x.handle);
    in_memory_bytes -= std::min(in_memory_bytes, x.positions * kIdBytes);
  }
#else
  (void)c, (void)in_memory_bytes, (void)cap_bytes;  // without the rule: the ids stay in memory whatever their sum
#endif
  return out;
}

// The files of the conversations' ids on one directory, and the rule that keeps a crash safe: a file a kept index names is never
// changed. Every put is a new file (<handle>.<seq>.ids, the whole history); a file leaves the disk only when no index names it:
// at once when there is no index to protect (index_on false), else when `committed` says an index naming the survivors is on the
// disk. After a crash the last index finds every file it names whole.
// PICO_OLD_RULES: a put replaces the conversation's one file (<handle>.ids) in place, so an index written before it names a
// file that no longer matches.
class Keeper {
 public:
  struct Ref {
    uint64_t seq = 0;  // the file's seq (0: none)
    uint64_t sum = 0;  // its header's sum
  };
  Keeper(std::string dir, bool index_on, uint64_t seq0) : dir_(std::move(dir)), index_on_(index_on), seq_(seq0) {}

  // The ids written to a new file; *out names it. The conversation's earlier files stay until an index that does not name them is on disk.
  bool put(int handle, const std::vector<int32_t>& ids, Ref* out, std::string* why) {
#ifndef PICO_OLD_RULES
    const uint64_t seq = ++seq_;
#else
    const uint64_t seq = 0;
#endif
    uint64_t sum = 0;
    if (!write(dir_, handle, ids, why, &sum, seq)) return false;
    auto& have = files_[handle];
    have.insert(seq);
    if (!index_on_)
      for (auto it = have.begin(); it != have.end();) {
        if (*it != seq) remove(dir_, handle, *it), it = have.erase(it);
        else ++it;
      }
    out->seq = seq, out->sum = sum;
    return true;
  }
  // A file the loaded index names (and its seq counts for the next one).
  void adopt(int handle, uint64_t seq) {
    files_[handle].insert(seq);
    seq_ = std::max(seq_, seq);
  }
  // The conversation is freed: its files go (now, or with the next index that does not name them).
  void drop(int handle) {
    if (index_on_) return;
    const auto it = files_.find(handle);
    if (it == files_.end()) return;
    for (const uint64_t q : it->second) remove(dir_, handle, q);
    files_.erase(it);
  }
  // An index that names `named` (handle -> seq) is on the disk: every other file is removed.
  void committed(const std::map<int, uint64_t>& named) {
    for (auto it = files_.begin(); it != files_.end();) {
      const auto n = named.find(it->first);
      for (auto q = it->second.begin(); q != it->second.end();) {
        if (n != named.end() && n->second == *q) ++q;
        else remove(dir_, it->first, *q), q = it->second.erase(q);
      }
      it = it->second.empty() ? files_.erase(it) : std::next(it);
    }
  }
  Named names() const {
    Named n;
    for (const auto& [h, v] : files_)
      for (const uint64_t q : v) n.insert({h, q});
    return n;
  }
  size_t count() const {
    size_t n = 0;
    for (const auto& [h, v] : files_) n += v.size();
    return n;
  }
  const std::string& dir() const { return dir_; }

 private:
  std::string dir_;
  bool index_on_;
  uint64_t seq_;
  std::map<int, std::set<uint64_t>> files_;
};

}  // namespace idsfile
