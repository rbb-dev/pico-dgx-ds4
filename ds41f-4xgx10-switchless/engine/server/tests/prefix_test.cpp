// server/prefix.h: a request that carries only its prompt's suffix. The prompt is built from the server's own history of the
// conversation, only when the request's epoch is the entry's and its count is within what the entry reported (a stale epoch, an
// unknown session, a count past the history, a fork whose base moved on, an entry of another factor, ids in their file, an empty
// suffix); the epoch changes exactly where the history loses or rewrites a position below what a DONE reported (a scripted
// history against a ground truth, and a client that remembers the ids it was told: whatever the engine builds begins with them);
// and the kept index's record holds no ids (its size and its round trip with epochs, the ids files it names).
// Built twice: with -DPICO_OLD_RULES (the policy without the rule: the epoch never changes, any entry of the id will do, the record
// carries the histories) the cases marked "red" fail; without it every case passes. Host only: no engine, no GPU.
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../frames.h"
#include "../ids_file.h"
#include "../prefix.h"

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

// What serve.cpp's Session has that the prompt's builder reads.
struct Entry {
  std::vector<int32_t> hist;
  bool ids_out = false;
  size_t ids_n = 0;
  prefix::Epoch ep;
};
using Store = std::map<std::pair<uint64_t, int>, Entry>;  // (conversation id, factor) -> entry; the key it is found under is the id

constexpr int kWindow16 = 65536 * 16;
int factor_for(int64_t n) { return n < kWindow16 ? 16 : n < 2 * kWindow16 ? 32 : 64; }  // serve.cpp's

std::vector<int32_t> ids(int n, int32_t from = 100) {
  std::vector<int32_t> v((size_t)n);
  for (int i = 0; i < n; ++i) v[(size_t)i] = from + i;
  return v;
}

struct Built {
  prefix::Gone gone;
  std::vector<int32_t> prompt;
  int loads = 0;
};
// One request against the store: the prompt, or why not. `fail_load`: the ids file cannot be read (the entry is then gone).
Built ask(Store& st, uint64_t session, uint64_t base, uint64_t epoch, uint32_t n_prefix, const std::vector<int32_t>& suffix,
          bool fail_load = false) {
  Built b;
  const prefix::Ask a{session, base, epoch, n_prefix};
  const int fac = factor_for((int64_t)n_prefix + (int64_t)suffix.size());
  b.gone = prefix::build<Entry>(
      a, fac, suffix, &b.prompt,
      [&](uint64_t id, int f, uint64_t* key) -> Entry* {
        const auto it = st.find({id, f});
        if (it == st.end()) return nullptr;
        *key = id;
        return &it->second;
      },
      [&](uint64_t key, Entry& e) {
        ++b.loads;
        if (fail_load) {
          st.erase({key, factor_for((int64_t)e.ids_n)});  // a lost file frees the conversation, as need_ids does
          return false;
        }
        e.hist = ids((int)e.ids_n, 100);  // the file's ids
        e.ids_out = false;
        return true;
      });
  return b;
}
Entry entry(int n, uint64_t epoch, uint64_t reported) {
  Entry e;
  e.hist = ids(n);
  e.ep.value = epoch, e.ep.reported = reported;
  return e;
}
bool equal_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b, size_t n) {
  if (a.size() < n || b.size() < n) return false;
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return false;
  return true;
}

uint64_t lcg_state = 88172645463325252ull;
uint64_t rnd() {
  lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
  return lcg_state >> 17;
}
}  // namespace

int main() {
  using prefix::Gone;
  const std::vector<int32_t> suffix = {7, 8, 9};

  // ---- the prompt's builder ----
  {
    Store st;
    st[{1, 16}] = entry(1000, 0xabc, 1000);
    Built b = ask(st, 1, 0, 0xabc, 600, suffix);
    check(b.gone == Gone::kNone && b.prompt.size() == 603 && equal_prefix(b.prompt, st[{1, 16}].hist, 600) && b.prompt[600] == 7 && b.prompt[602] == 9,
          "a request with the entry's epoch builds the prompt: its first 600 ids, then the suffix");
    b = ask(st, 1, 0, 0xabc, 1000, suffix);
    check(b.gone == Gone::kNone && b.prompt.size() == 1003, "the whole history as the prefix builds too");
    b = ask(st, 1, 0, 0xdef, 600, suffix);
    check(b.gone != Gone::kNone, "red: a stale epoch is gone (the history changed since the front-end's DONE)");
    b = ask(st, 1, 0, 0, 600, suffix);
    check(b.gone != Gone::kNone, "red: epoch 0 is no epoch: gone");
    b = ask(st, 2, 0, 0xabc, 600, suffix);
    check(b.gone == Gone::kNoEntry && b.prompt.empty(), "an unknown session is gone, nothing built");
    b = ask(st, 1, 0, 0xabc, 1001, suffix);
    check(b.gone != Gone::kNone, "red: n_prefix past the history (1001 of 1000) is gone");
  }
  {  // what the last DONE reported bounds the prefix, not what the entry holds now (a request in flight has more ids than its DONE told)
    Store st;
    st[{1, 16}] = entry(1000, 0xabc, 700);
    check(ask(st, 1, 0, 0xabc, 700, suffix).gone == Gone::kNone, "n_prefix at what the DONE reported builds");
    check(ask(st, 1, 0, 0xabc, 701, suffix).gone != Gone::kNone, "red: one id past what the DONE reported is gone, though the entry holds it");
  }
  {  // a fork: the epoch and the count are the base's, and the prompt is built from the base's entry
    Store st;
    st[{10, 16}] = entry(500, 0x111, 500);   // the base
    Entry own = entry(300, 0x222, 300);      // the fork's own session id, already holding other ids
    for (int32_t& t : own.hist) t += 5000;
    st[{20, 16}] = own;
    Built b = ask(st, 20, 10, 0x111, 400, suffix);
    check(b.gone == Gone::kNone && equal_prefix(b.prompt, st[{10, 16}].hist, 400) && b.prompt.size() == 403,
          "a fork builds from the base's entry, not from its own session's");
    b = ask(st, 20, 10, 0x222, 250, suffix);
    check(b.gone != Gone::kNone, "red: a fork naming its own session's epoch (not the base's) is gone");
    st[{10, 16}].ep.value = 0x333;  // the base moved on
    b = ask(st, 20, 10, 0x111, 400, suffix);
    check(b.gone != Gone::kNone, "red: a fork with a stale base epoch is gone");
    b = ask(st, 21, 99, 0x111, 400, suffix);
    check(b.gone == Gone::kNoEntry, "a fork of a base the engine does not hold is gone");
  }
  {  // an entry of another factor: a conversation has one entry per factor, each with its own history
    Store st;
    st[{1, 16}] = entry(1000, 0xabc, 1000);
    Built b = ask(st, 1, 0, 0xabc, 1000, ids(kWindow16 - 1000, 5));  // the built prompt is 1,048,576 ids: factor 32
    check(b.gone != Gone::kNone, "red: a prompt of factor 32 does not use the factor-16 entry");
    st[{1, 32}] = entry(1200, 0xfed, 1200);
    b = ask(st, 1, 0, 0xfed, 1100, ids(kWindow16 - 1100, 5));
    check(b.gone == Gone::kNone && b.prompt.size() == (size_t)kWindow16 && equal_prefix(b.prompt, st[{1, 32}].hist, 1100),
          "the factor-32 entry builds it, with its own epoch");
    b = ask(st, 1, 0, 0xabc, 1000, ids(1000, 5));
    check(b.gone == Gone::kNone && b.prompt.size() == 2000 && equal_prefix(b.prompt, st[{1, 16}].hist, 1000), "and the factor-16 entry builds a short prompt");
  }
  {  // ids in their file: read back before the prompt is built; unreadable: gone (the entry freed); a stale request reads nothing
    Store st;
    Entry e = entry(0, 0xabc, 800);
    e.hist.clear(), e.ids_out = true, e.ids_n = 800;
    st[{1, 16}] = e;
    Built b = ask(st, 1, 0, 0xabc, 500, suffix);
    check(b.gone == Gone::kNone && b.loads == 1 && b.prompt.size() == 503 && b.prompt[0] == 100 && b.prompt[499] == 599, "ids in their file are read back, then the prompt is built");
    st[{1, 16}] = e;
    b = ask(st, 1, 0, 0xdef, 500, suffix);
    check(b.gone != Gone::kNone && b.loads == 0, "red: a stale epoch reads no file");
    b = ask(st, 1, 0, 0xabc, 801, suffix);
    check(b.gone != Gone::kNone && b.loads == 0, "red: a count past the history reads no file");
    b = ask(st, 1, 0, 0xabc, 500, suffix, true);
    check(b.gone == Gone::kIds && b.prompt.empty() && st.empty(), "red: an unreadable file is gone, nothing built, the conversation freed");
  }
  {  // an empty suffix
    Store st;
    st[{1, 16}] = entry(1000, 0xabc, 1000);
    Built b = ask(st, 1, 0, 0xabc, 1000, {});
    check(b.gone == Gone::kNone && b.prompt.size() == 1000 && equal_prefix(b.prompt, st[{1, 16}].hist, 1000), "an empty suffix: the prompt is the prefix");
    b = ask(st, 1, 0, 0xdef, 1000, {});
    check(b.gone != Gone::kNone && b.prompt.empty(), "red: an empty suffix with a stale epoch is gone, not a prompt from the wrong ids");
    b = ask(st, 1, 0, 0xabc, 0, {});
    check(b.gone == Gone::kNone && b.prompt.empty(), "an empty suffix and no prefix builds an empty prompt (the server refuses it: 'empty prompt')");
  }

  // ---- the epoch ----
  {
    prefix::Epoch e;
    e.value = 5, e.reported = 100;
    uint64_t next = 1000;
    int asked = 0;
    auto fresh = [&] { ++asked; return ++next; };
    check(!prefix::lose_below(&e, 100, fresh) && !prefix::lose_below(&e, 250, fresh) && e.value == 5 && asked == 0, "a history at or past the reported size keeps its epoch");
    prefix::reported_at(&e, 300);
    check(e.value == 5 && e.reported == 300, "a DONE's report changes the size, not the epoch (appends never change it)");
    check(prefix::lose_below(&e, 299, fresh) && e.value == 1001 && e.reported == 0 && asked == 1, "red: a cut below the reported size is a new epoch, and nothing is reported of it yet");
    check(!prefix::lose_below(&e, 10, fresh) && e.value == 1001, "red: a second cut before the next DONE needs no third epoch");
    prefix::Epoch z;
    z.value = 7, z.reported = 50;
    check(prefix::lose_below(&z, 0, [] { return (uint64_t)0; }) && z.value != 0, "red: an epoch is never 0");
  }
  {  // a scripted history against the ground truth, and a client that remembers what its last DONE told it
    // truth, from the ids themselves: the client's epoch is still the entry's exactly while the history still begins with the ids its
    // last DONE told it (the script's ids are all new, so a position lost is never regrown equal)
    struct Model {
      std::vector<int32_t> hist;
      prefix::Epoch ep;
    } m;
    m.ep.value = 1;
    uint64_t counter = 1;
    auto fresh = [&] { return ++counter; };
    std::vector<int32_t> client_ids;  // what the front-end's last DONE told it: the history's first n_hist ids
    uint64_t client_epoch = 0;
    int32_t next_id = 1000;
    int cuts = 0, below = 0, changes = 0, wrong_epoch = 0, wrong_prompt = 0, builds = 0, gone = 0;
    for (int step = 0; step < 4000; ++step) {
      const uint64_t kind = rnd() % 6;
      const uint64_t before_epoch = m.ep.value;
      if (kind <= 1) {  // appends: a prefill piece or decoded tokens
        for (uint64_t k = rnd() % 50 + 1; k > 0; --k) m.hist.push_back(next_id++);
      } else if (kind == 2 || kind == 3) {  // the epoch's two sites: a request restarts the session at q (the history cut there); a run's
                                            // history resized to the position of its next token
        if (!m.hist.empty()) {
          const size_t q = rnd() % (m.hist.size() + 1);
          below += q < client_ids.size();
          ++cuts;
          prefix::lose_below(&m.ep, q, fresh);
          m.hist.resize(q);
        }
      } else if (kind == 4) {  // the stored state is cut (a snapshot, a read-back that cut): the history is the same, nothing is called
      } else {                 // a DONE: the front-end learns the epoch and the size
        prefix::reported_at(&m.ep, m.hist.size());
        client_epoch = m.ep.value;
        client_ids = m.hist;
      }
      changes += m.ep.value != before_epoch;
      if (client_epoch != 0) {
        const bool still_true = m.hist.size() >= client_ids.size() && equal_prefix(m.hist, client_ids, client_ids.size());
        wrong_epoch += still_true != (m.ep.value == client_epoch);
      }
      // a request from the client: its epoch, a prefix of what it knows, a suffix
      if (!client_ids.empty() && step % 3 == 0) {
        Store st;
        Entry e;
        e.hist = m.hist, e.ep = m.ep;
        st[{1, 16}] = e;
        const uint32_t n = (uint32_t)(rnd() % (client_ids.size() + 1));
        const Built b = ask(st, 1, 0, client_epoch, n, {1, 2});
        if (b.gone == Gone::kNone) {
          ++builds;
          if (!equal_prefix(b.prompt, client_ids, n) || b.prompt.size() != n + 2) ++wrong_prompt;
        } else {
          ++gone;
        }
      }
    }
    std::printf("     (a scripted history: %d cuts, %d below the last DONE's size, %d epoch changes, %d prompts built, %d gone)\n", cuts, below, changes, builds, gone);
    check(cuts > 500 && below > 100, "fixture: the script cuts the history often, below what a DONE reported often");
    check(wrong_epoch == 0, "red: the epoch changes exactly at the two sites, when a position below the reported size is lost, and never otherwise");
    check(builds > 100 && wrong_prompt == 0, "red: whatever the engine builds begins with the ids the front-end was told (never a prompt from the wrong ids)");
  }

  // ---- the kept index ----
  {
    const int n_conv = 97, n_ids = 200000;  // 97 conversations (the M row: 632 MB, 1.3 s), each 200,000 ids here
    std::vector<prefix::KeptRec> recs;
    for (int i = 0; i < n_conv; ++i) {
      prefix::KeptRec r;
      r.id = 0x5e550000ull + (uint64_t)i, r.h = i + 1, r.pe_snap = i * 3, r.end_lo = i, r.end_clo = i + 1, r.end_w = i + 2, r.end_dec_lo = i + 3;
      r.used = 1000 + (uint64_t)i, r.t_used = 1.7e9 + i, r.factor = i % 3 == 0 ? 16 : i % 3 == 1 ? 32 : 64, r.owner = i % 5 == 0 ? 0x77ull : 0;
      r.epoch = 0xe0000000ull + (uint64_t)i * 977, r.reported = (uint64_t)(n_ids - i), r.ids_n = n_ids, r.ids_sum = 0x1234567800000000ull + (uint64_t)i, r.ids_seq = 1700000000000ull + (uint64_t)i;
      for (int k = 0; k < i % 7; ++k) r.enc_snaps.push_back(2048 * (k + 1));
#ifdef PICO_OLD_RULES
      r.ids = ids(n_ids, i);  // without the rule: the record carries the history
#endif
      recs.push_back(std::move(r));
    }
    std::vector<uint8_t> f;
    const auto t0 = std::chrono::steady_clock::now();
    for (const prefix::KeptRec& r : recs) prefix::put_rec(f, r);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("     (index body of %d conversations of %d ids: %zu bytes, capacity %zu, %.3f ms)\n", n_conv, n_ids, f.size(), f.capacity(), ms);
    check(f.size() < 64 * 1024, "red: the index of 97 conversations of 200,000 ids is KiBs, not the histories (0.08 GB of ids here)");
    check(f.capacity() < 128 * 1024, "red: the checkpoint's own allocation is KiBs");
    check(ms < 50, "red: building it takes milliseconds (E: the histories' write took 1.3 s for 97 conversations)");
    keepfile::Reader rd{f.data(), f.size()};
    bool same = true;
    for (const prefix::KeptRec& r : recs) {
      prefix::KeptRec g;
      if (!prefix::get_rec(rd, &g, 1 << 22, 4096)) {
        same = false;
        break;
      }
      same = same && g.id == r.id && g.h == r.h && g.pe_snap == r.pe_snap && g.end_lo == r.end_lo && g.end_clo == r.end_clo && g.end_w == r.end_w &&
             g.end_dec_lo == r.end_dec_lo && g.used == r.used && g.t_used == r.t_used && g.factor == r.factor && g.owner == r.owner &&
             g.epoch == r.epoch && g.reported == r.reported && g.ids_n == r.ids_n && g.ids_sum == r.ids_sum && g.ids_seq == r.ids_seq && g.enc_snaps == r.enc_snaps;
    }
    check(same && rd.ok && rd.at == rd.n, "the round trip: every record back equal, with its epoch, its reported size and its ids file's count, sum and seq");
    // a record cut short, and counts out of range, are refused
    keepfile::Reader cut{f.data(), 40};
    prefix::KeptRec g;
    check(!prefix::get_rec(cut, &g, 1 << 22, 4096), "a record cut short is refused");
    std::vector<uint8_t> bad;
    prefix::KeptRec r = recs[0];
    r.reported = (uint64_t)r.ids_n + 1;
#ifdef PICO_OLD_RULES
    r.ids.clear();
    r.ids.resize((size_t)r.ids_n);
#endif
    prefix::put_rec(bad, r);
    keepfile::Reader br{bad.data(), bad.size()};
    check(!prefix::get_rec(br, &g, 1 << 22, 4096), "a record that reports more than it holds is refused");
  }
  {  // the ids files the index names: written with their sum, looked at without reading the ids, swept except the kept ones
    char tmpl[] = "/tmp/prefix_test.XXXXXX";
    const std::string root = mkdtemp(tmpl) ? tmpl : "";
    check(!root.empty(), "fixture: a scratch directory");
    const std::string dir = root + "/ids";
    std::string why;
    check(idsfile::ensure_dir(dir, &why), "the ids directory is made");
    uint64_t sum1 = 0, sum2 = 0, c = 0, s = 0;
    const std::vector<int32_t> a = ids(5000), b = ids(70, 9);
    check(idsfile::write(dir, 3, a, &why, &sum1) && idsfile::write(dir, 4, b, &why, &sum2) && sum1 != sum2, "two files written, each with its sum");
    check(idsfile::peek(dir, 3, &c, &s, &why) && c == 5000 && s == sum1, "peek: the count and the sum without reading the ids");
    std::vector<int32_t> back;
    check(idsfile::read(dir, 3, &back, &why) && back == a, "and the ids read back are the written ones");
    check(!idsfile::peek(dir, 99, &c, &s, &why) && why.empty(), "peek of a missing file: false, no message");
    std::vector<int32_t> a2 = a;
    a2.push_back(1);
    uint64_t sum3 = 0;
    check(idsfile::write(dir, 3, a2, &why, &sum3) && idsfile::peek(dir, 3, &c, &s, &why) && c == 5001 && s == sum3 && s != sum1,
          "a longer history replaces the file: the index's old count and sum no longer match it");
    {
      FILE* f = fopen(idsfile::path(dir, 8).c_str(), "w");
      if (f) fputs("not an ids file at all, of the wrong size", f), fclose(f);
    }
    check(!idsfile::peek(dir, 8, &c, &s, &why) && !why.empty(), "a file that is not an ids file is refused");
    check(idsfile::sweep(dir, {{4, 0}}, &why) && exists(idsfile::path(dir, 4)) && !exists(idsfile::path(dir, 3)) && !exists(idsfile::path(dir, 8)),
          "the sweep keeps the files of the kept conversations and removes the others");
    check(idsfile::write(dir, 6, a, &why, &sum1, 77) && exists(idsfile::path(dir, 6, 77)) && idsfile::peek(dir, 6, &c, &s, &why, 77) && c == 5000,
          "a file under a seq is <handle>.<seq>.ids and reads back");
    check(idsfile::sweep(dir, {{4, 0}}, &why) && !exists(idsfile::path(dir, 6, 77)) && exists(idsfile::path(dir, 4)),
          "the sweep removes a seq file the index does not name");
    check(idsfile::write(dir, 6, a, &why, &sum1, 77) && idsfile::write(dir, 6, a, &why, &sum1, 78) && idsfile::sweep(dir, {{6, 78}}, &why) &&
              !exists(idsfile::path(dir, 6, 77)) && exists(idsfile::path(dir, 6, 78)) && !exists(idsfile::path(dir, 4)),
          "and keeps exactly the seq it names");
    check(idsfile::prepare(dir, &why) && !exists(idsfile::path(dir, 6, 78)), "prepare still empties the directory");
    idsfile::prepare(dir, &why);
    rmdir(dir.c_str());
    rmdir(root.c_str());
  }

  // ---- the frames of the two protocol versions ----
  {
    auto submit = [&](uint32_t flags, size_t n_stop, size_t n_prompt, bool with_prefix_block, uint64_t epoch, uint32_t n_prefix) {
      fe::SubmitHeader h{};
      h.req_id = 9, h.session_id = 5, h.flags = flags, h.n_stop_ids = (uint32_t)n_stop, h.n_prompt = (uint32_t)n_prompt, h.max_new_tokens = 4;
      std::vector<char> p(sizeof h);
      std::memcpy(p.data(), &h, sizeof h);
      if (with_prefix_block) {
        const fe::SubmitPrefix px{epoch, n_prefix, 0};
        p.insert(p.end(), (const char*)&px, (const char*)&px + sizeof px);
      }
      p.resize(p.size() + 4 * (n_stop + n_prompt), 1);
      return p;
    };
    fe::SubmitHeader h;
    fe::SubmitPrefix px;
    size_t base = 0;
    const std::vector<char> plain = submit(fe::kFlagKeep, 1, 10, false, 0, 0);
    const std::vector<char> pre = submit(fe::kFlagKeep | fe::kFlagPrefix, 1, 3, true, 0xfeedull, 700);
    check(frames::version_ok(4) && frames::version_ok(5) && !frames::version_ok(3) && !frames::version_ok(6), "HELLO: today's version and the next are accepted, no other");
    check(frames::parse_submit_head(plain.data(), plain.size(), fe::kVersion, &h, &px, &base) && base == plain.size() && h.n_prompt == 10,
          "a connection of today's version: a plain SUBMIT parses");
    check(frames::parse_submit_head(plain.data(), plain.size(), fe::kVersionPrefix, &h, &px, &base) && base == plain.size(), "and on the next version too");
    check(!frames::parse_submit_head(pre.data(), pre.size(), fe::kVersion, &h, &px, &base), "red: a connection of today's version sending kFlagPrefix is malformed");
    check(frames::parse_submit_head(pre.data(), pre.size(), fe::kVersionPrefix, &h, &px, &base) && base == pre.size() && px.epoch == 0xfeedull && px.n_prefix == 700 &&
              h.n_prompt == 3 && (h.flags & fe::kFlagPrefix),
          "the next version: the prefix frame parses (72 + 16 + 4 x (stops + suffix) bytes), epoch and count back");
    std::vector<char> short_pre = pre;
    short_pre.pop_back();
    check(!frames::parse_submit_head(short_pre.data(), short_pre.size(), fe::kVersionPrefix, &h, &px, &base) &&
              !frames::parse_submit_head(plain.data(), plain.size() - 1, fe::kVersion, &h, &px, &base),
          "a frame of the wrong length is malformed on both versions");
    // a client's reading of a DONE: the tail is there when the payload is 56 + 16 + error_len
    auto read_done = [&](const std::vector<char>& f, bool* has_tail, fe::DoneTail* t, std::string* err, fe::DoneHeader* d) {
      fe::FrameHeader fh;
      std::memcpy(&fh, f.data(), sizeof fh);
      const char* p = f.data() + sizeof fh;
      std::memcpy(d, p, sizeof *d);
      *has_tail = fh.size == sizeof *d + sizeof *t + d->error_len;
      if (*has_tail) std::memcpy(t, p + sizeof *d, sizeof *t);
      *err = std::string(p + fh.size - d->error_len, d->error_len);
      return fh.type == fe::kDone && fh.size == f.size() - sizeof fh;
    };
    fe::DoneHeader d{};
    d.req_id = 9, d.reason = fe::kStop, d.prompt_tokens = 1000, d.cached_tokens = 900, d.completion_tokens = 4, d.error_len = 3;
    const fe::DoneTail tl{0xabcd, 1003, 0};
    for (const uint32_t ver : {fe::kVersion, fe::kVersionPrefix}) {
      const std::vector<char> f = frames::done_frame(ver, d, &tl, "bad");
      bool has = false;
      fe::DoneTail got{};
      fe::DoneHeader gd{};
      std::string err;
      const bool ok = read_done(f, &has, &got, &err, &gd);
      const bool next = ver == fe::kVersionPrefix;
      check(ok && has == next && f.size() == 8 + 56 + (next ? 16 : 0) + 3 && err == "bad" && gd.req_id == 9 && gd.cached_tokens == 900,
            next ? "the next version: DONE is 56 + 16 + the error text, the tail read back, the text intact"
                 : "red: today's version: DONE is 56 bytes and the error text, no tail");
      check(!next || (got.epoch == 0xabcd && got.n_hist == 1003), next ? "the tail's epoch and size round trip" : "(no tail to read on today's version)");
      const std::vector<char> g = frames::done_frame(ver, d, nullptr, "bad");
      check(g.size() == 8 + 56 + 3, "a DONE of a request that did not run has no tail on either version");
    }
  }

  // ---- the ids files an index names are never changed ----
  {
    char tmpl[] = "/tmp/prefix_keep.XXXXXX";
    const std::string root = mkdtemp(tmpl) ? tmpl : "";
    check(!root.empty(), "fixture: a scratch directory");
    const std::string dir = root + "/ids";
    std::string why;
    idsfile::ensure_dir(dir, &why);
    struct Row {
      int h;
      uint64_t seq, sum, n;
    };
    // what the last index on disk names, and what loading it finds
    std::vector<Row> index;
    auto load_last = [&](const char* when) {
      bool all = true;
      for (const Row& r : index) {
        uint64_t c = 0, sm = 0;
        std::string w;
        all = all && idsfile::peek(dir, r.h, &c, &sm, &w, r.seq) && c == r.n && sm == r.sum;
      }
      char b[160];
      std::snprintf(b, sizeof b, "red: a crash %s: the last index finds every conversation it names with its count and sum", when);
      check(all, b);
    };
    idsfile::Keeper k(dir, true, 1000);
    idsfile::Keeper::Ref r1, r2, r3;
    std::vector<int32_t> h1 = ids(3000), h2 = ids(500, 7), h3 = ids(80, 9);
    check(k.put(1, h1, &r1, &why) && k.put(2, h2, &r2, &why) && k.put(3, h3, &r3, &why), "three conversations' ids written");
    auto write_index = [&](const std::vector<Row>& rows) {
      index = rows;
      std::map<int, uint64_t> named;
      for (const Row& r : rows) named[r.h] = r.seq;
      k.committed(named);  // the index is on the disk
    };
    write_index({{1, r1.seq, r1.sum, 3000}, {2, r2.seq, r2.sum, 500}, {3, r3.seq, r3.sum, 80}});
    load_last("right after an index");
    h1.insert(h1.end(), 40, 31337);  // an append
    idsfile::Keeper::Ref n1;
    check(k.put(1, h1, &n1, &why), "conversation 1's history grew and its ids are written again");
    load_last("after an append, before the next index");
    std::vector<int32_t> rew = ids(500, 9000);  // a rewrite: other ids, another length
    rew.resize(430);
    idsfile::Keeper::Ref n2;
    check(k.put(2, rew, &n2, &why), "conversation 2's history was rewritten and its ids are written again");
    load_last("after a rewrite, before the next index");
    k.drop(3);
    load_last("after a conversation was freed, before the next index");
    check(exists(idsfile::path(dir, 1, r1.seq)) && exists(idsfile::path(dir, 3, r3.seq)), "the files the last index names are still on the disk until the next index");
    write_index({{1, n1.seq, n1.sum, 3040}, {2, n2.seq, n2.sum, 430}});
    load_last("after the next index");
    check(!exists(idsfile::path(dir, 1, r1.seq)) && !exists(idsfile::path(dir, 2, r2.seq)) && !exists(idsfile::path(dir, 3, r3.seq)) &&
              exists(idsfile::path(dir, 1, n1.seq)) && exists(idsfile::path(dir, 2, n2.seq)),
          "red: the files the new index does not name are removed after it is on the disk, the ones it names are there");
    check(idsfile::sweep(dir, k.names(), &why) && k.count() == 2, "a start's sweep keeps what the loaded index names");
    idsfile::Keeper free_run(dir, false, 5000);
    idsfile::Keeper::Ref a, b;
    check(free_run.put(9, h3, &a, &why) && free_run.put(9, h2, &b, &why) && !exists(idsfile::path(dir, 9, a.seq)) && exists(idsfile::path(dir, 9, b.seq)),
          "red: with no index to protect a replaced file goes at once");
    free_run.drop(9);
    check(!exists(idsfile::path(dir, 9, b.seq)), "red: and a freed conversation's file with it");
    idsfile::prepare(dir, &why);
    rmdir(dir.c_str());
    rmdir(root.c_str());
  }

  std::printf(fails ? "prefix_test FAILED (%d)\n" : "prefix_test passed\n", fails);
  return fails ? 1 : 0;
}
