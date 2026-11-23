// The store's trace format (model/store_trace.h; host only, no GPU): a scripted sequence of entries is written through the
// real writer and read back by the real reader; every field must round-trip, a nested entry must write nothing, the file
// is numbered by the first number not yet used, a record with one field changed must be reported at that record, and a
// truncated or malformed file must be refused with a message that says where.
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "model/store_trace.h"

namespace st = model::store_trace;

namespace {

int g_fail = 0, g_checks = 0;
void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s\n", what.c_str());
  }
}

// the store the scripted entries act on: one number every digest and the counts derive from
struct Fake {
  uint64_t v = 0;
};
void fake_snapshot(const void* ctx, st::Snapshot* s) {
  const uint64_t v = static_cast<const Fake*>(ctx)->v;
  s->digest = v * 0x9E3779B97F4A7C15ull, s->wide = v * 0xBF58476D1CE4E5B9ull + 1;
  for (int i = 0; i < 6; ++i) s->out[i] = v + (uint64_t)i;
  s->free_numbers = 1000 - v, s->numbers = 1024;
}

std::string slurp(const std::string& path) {
  std::string t;
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    char b[4096];
    for (size_t n; (n = std::fread(b, 1, sizeof b, f)) > 0;) t.append(b, n);
    std::fclose(f);
  }
  return t;
}

bool exists(const std::string& p) {
  struct stat sb;
  return stat(p.c_str(), &sb) == 0;
}

const char* kGeom1 = "rank=0 nslots=8 nunits=8 list_len=520 max_ctx=4194304 image=2916352 snap=100 end=200 spec=1 poison=0 host_ring=1";
const char* kGeom2 = "unit_of=0,1,2,3,4,5,6,7 slot_ctx=1,2,3,4,5,6,7,8";

// the scripted sequence: entries before the file opens, then the rest
void script(st::Writer& w, Fake& fk, const std::string& path0) {
  using st::Kind;
  uint64_t gen = 0x1234abcd;
  {
    st::Scope sc(Kind::kept_check, &fk);
    sc.arg_s("base", path0 + "/rank 0%x").arg_u("run_id", 18446744073709551615ull).arg_d("max_age", 86400.5);
    sc.answer_at_exit("kept_gen", &gen);
    st::answer("now", 1700000000);
    ++fk.v;
  }
  {
    st::Scope sc(Kind::disk, &fk);
    sc.arg_s("base", path0 + "/rank 0%x").arg_u("bytes", 68719476736ull).arg_u("staging", 536870912).arg("reuse", 1).arg_u("run_id", 7);
    st::answer("disk_open", 68719476736ll);
    ++fk.v;
  }
  w.open(0, kGeom1, kGeom2, 1024, 68719476736ull);
  {
    st::Scope sc(Kind::pin, &fk);
    sc.arg_v("pin", {3, 5}).arg_v("live", {}).arg_v("last", {9});
    ++fk.v;
  }
  {
    st::Scope sc(Kind::ensure, &fk);
    sc.arg("slot", 2).arg("upto", 40000);
    ++fk.v;
    sc.result(40064);
  }
  {  // an entry that calls two others: they write nothing, their answers belong to this one
    st::Scope sc(Kind::slot_attach, &fk);
    sc.arg("slot", 1).arg("h", 12);
    {
      st::Scope in(Kind::slot_release, &fk);
      in.arg("slot", 1);
      check(!in.top(), "a nested scope is not the outermost");
      ++fk.v;
    }
    {
      st::Scope in(Kind::unspill, &fk);
      in.arg("h", 12);
      st::answer("agree_min", 2147483647);
      ++fk.v;
      in.result(99);  // a nested result is dropped
    }
    sc.result_fn([&fk] { return (int64_t)(fk.v * 10); });
  }
  {
    st::Scope sc(Kind::evict_step, &fk);
    sc.arg("h", 4);
    ++fk.v;
    const bool r = sc.ret(true);
    check(r, "ret passes its value on");
  }
  {
    st::Scope sc(Kind::snap_buf, &fk);
    sc.arg("h", 4).arg("pos", 131072);
    std::vector<uint32_t> none;
    const std::vector<uint32_t>* p = nullptr;
    check(sc.ret_list(p) == nullptr, "ret_list passes a null on");
    (void)none;
  }
  st::evicted({4, 5});
  {
    st::Scope sc(Kind::keep, &fk);
    sc.arg_u("gen", 0xffffffff00000001ull);
    st::answer("index_write", 1);
    ++fk.v;
  }
  st::evicted({});
  st::answer("now", 5);  // outside any entry: an `A 0` line
  {
    st::Scope sc(Kind::alloc_grow, &fk);
    sc.arg("numbers", 2048);
    ++fk.v;
  }
}

std::string tmpdir() {
  char t[] = "/tmp/store_trace_test.XXXXXX";
  if (!mkdtemp(t)) {
    std::perror("mkdtemp");
    std::exit(2);
  }
  return t;
}

}  // namespace

int main() {
  st::Writer* saved = st::g_w;
  st::set_snapshot(&fake_snapshot);
  const std::string dir = tmpdir();

  // ---- off: nothing is built, nothing is written
  st::g_w = nullptr;
  {
    Fake fk;
    st::Scope sc(st::Kind::pin, &fk);
    sc.arg("x", 1).arg_v("l", {1, 2}).result(3);
    st::answer("now", 1);
    st::evicted({1});
    check(!sc.top() && !st::on(), "off: the scope is inert");
  }
  check(!exists(dir + "/store.rank0.1.t1"), "off: no file");

  // ---- the scripted sequence, written and read back
  Fake fk;
  st::Writer w(dir);
  st::g_w = &w;
  script(w, fk, dir);
  st::g_w = nullptr;
  check(w.path() == dir + "/store.rank0.1.t1", "the first file is store.rank0.1.t1");
  st::Trace t;
  std::string err;
  const bool ok = st::read_file(w.path(), &t, &err);
  check(ok, "the written trace reads back: " + err);
  if (!ok) return 1;
  const std::string good = slurp(w.path());  // before any later start takes the same name
  check(t.records.size() == 9, "nine outermost entries make nine records (nested ones none), got " + std::to_string(t.records.size()));
  const char* kinds[] = {"kept_check", "disk", "pin", "ensure", "slot_attach", "evict_step", "snap_buf", "keep", "alloc_grow"};
  for (size_t i = 0; i < t.records.size() && i < 9; ++i) {
    check(t.records[i].kind == kinds[i], std::string("record ") + std::to_string(i + 1) + " is " + kinds[i]);
    check(t.records[i].n == i + 1, "entries are numbered from 1 in order");
  }
  auto E = [&](size_t i) -> const st::Line& { return t.lines[t.records[i].first]; };
  auto val = [&](const st::Line& l, const char* k) -> std::string {
    const st::Field* f = t.find(l, k);
    return f ? f->value : "<missing>";
  };
  check(val(E(0), "base") == st::escape(dir + "/rank 0%x"), "a path with a space and a percent round-trips escaped");
  check(st::unescape(val(E(0), "base")) == dir + "/rank 0%x", "and unescapes to itself");
  check(val(E(0), "run_id") == "18446744073709551615" && val(E(0), "max_age") == "86400.5", "a 64-bit id and a double");
  check(val(E(2), "pin") == "3,5" && val(E(2), "live") == "" && val(E(2), "last") == "9", "lists, the empty one too");
  check(val(E(3), "slot") == "2" && val(E(3), "upto") == "40000", "ensure's arguments");
  check(val(E(7), "gen") == "18446744069414584321", "keep's 64-bit generation");
  // answers, results and digests, by line
  std::vector<std::string> a_lines, r_lines, d_lines, v_lines, tags;
  for (const st::Line& l : t.lines) {
    tags.push_back(std::string(1, l.tag));
    if (l.tag == 'A') a_lines.push_back(l.raw);
    if (l.tag == 'R') r_lines.push_back(l.raw);
    if (l.tag == 'D') d_lines.push_back(l.raw);
    if (l.tag == 'V') v_lines.push_back(l.raw);
  }
  check(a_lines.size() == 6, "six answer lines (now, kept_gen, disk_open, agree_min, index_write, the stray one), got " + std::to_string(a_lines.size()));
  if (a_lines.size() == 6) {
    check(a_lines[0] == "A 1 now=1700000000" && a_lines[1] == "A 1 kept_gen=" + std::to_string(0x1234abcdull), "kept_check's answers: the clock, then the generation read at its end");
    check(a_lines[2] == "A 2 disk_open=68719476736", "disk's returned size");
    check(a_lines[3] == "A 5 agree_min=2147483647", "an answer taken in a nested entry belongs to the outermost");
    check(a_lines[4] == "A 8 index_write=1", "the index write's result");
    check(a_lines[5] == "A 0 now=5", "an answer outside any entry is an A 0 line");
  }
  check(r_lines.size() == 4 && r_lines[0] == "R 4 40064" && r_lines[2] == "R 6 1" && r_lines[3] == "R 7 -1",
        "results: ensure's, slot_attach's, evict_step's, and snap_buf's -1 for none");
  // slot_attach's R is read when it ends: the increments by then (kept_check, disk, pin, ensure, release, unspill = 6), times 10
  check(r_lines.size() == 4 && r_lines[1] == "R 5 60", "a result function is read when the entry ends, after the nested entries");
  check(d_lines.size() == 9 && d_lines[0].rfind("D 1 d=", 0) == 0, "one D line per entry");
  check(d_lines.size() == 9 && d_lines[8] == [] {
    char b[300];
    const uint64_t v = 9;
    std::snprintf(b, sizeof b, "D 9 d=%016llx w=%016llx out=%llu,%llu,%llu,%llu,%llu,%llu free=%llu numbers=1024",
                  (unsigned long long)(v * 0x9E3779B97F4A7C15ull), (unsigned long long)(v * 0xBF58476D1CE4E5B9ull + 1),
                  (unsigned long long)v, (unsigned long long)v + 1, (unsigned long long)v + 2, (unsigned long long)v + 3,
                  (unsigned long long)v + 4, (unsigned long long)v + 5, (unsigned long long)(1000 - v));
    return std::string(b);
  }(), "the D line carries the digests, the six outside values, the free count and the allocator's size, as the snapshot gave them");
  check(v_lines.size() == 2 && v_lines[0] == "V 1 list=4,5" && v_lines[1] == "V 2 list=", "take_evicted lines, numbered apart from the entries");
  // the header and the held entries
  check(t.lines[0].raw == "pico-store-trace 1" && t.lines[1].raw == std::string("geom ") + kGeom1 &&
            t.lines[2].raw == std::string("geom ") + kGeom2 && t.lines[3].raw == "pool numbers=1024 disk=68719476736",
        "the header: version, two geom lines, pool");
  check(t.lines[4].raw == "pre numbers=1024" && t.lines[4].tag == 'P', "the pre line carries the allocator's size at the first held entry");
  size_t open_at = 0;
  for (size_t i = 0; i < t.lines.size(); ++i)
    if (t.lines[i].tag == 'O') open_at = i;
  check(open_at > 0 && t.lines[t.records[1].last + 1].tag == 'O' && t.records[2].first > open_at, "'open' stands between the held entries and the rest");
  check(st::to_text(t) == good, "to_text writes back exactly what was read");

  // ---- the file is the first number not yet used; another rank has its own
  {
    st::Writer w2(dir);
    st::g_w = &w2;
    Fake f2;
    w2.open(0, kGeom1, kGeom2, 1024, 0);
    st::g_w = nullptr;
    check(w2.path() == dir + "/store.rank0.2.t1", "the second start's file is .2");
    st::Writer w3(dir);
    w3.open(3, kGeom1, kGeom2, 1024, 0);
    check(w3.path() == dir + "/store.rank3.1.t1", "another rank starts at 1");
    unlink((dir + "/store.rank0.1.t1").c_str());
    st::Writer w4(dir);
    w4.open(0, kGeom1, kGeom2, 1024, 0);
    check(w4.path() == dir + "/store.rank0.1.t1", "the first number not used: a gap is filled");
  }

  // ---- one record of every kind, and the controls on the comparison
  const std::string d2 = tmpdir();
  std::string all_path;
  {
    st::Writer w2(d2);
    st::g_w = &w2;
    Fake f2;
    w2.open(0, kGeom1, kGeom2, 1024, 0);
    for (int k = 0; k < st::kKinds; ++k) {
      st::Scope sc((st::Kind)k, &f2);
      for (const std::string& a : st::kind_fields(st::kind_name((st::Kind)k))) sc.arg(a.c_str(), 5);
      ++f2.v;
      sc.result(k);
    }
    all_path = w2.path();
    st::g_w = nullptr;
  }
  st::Trace all;
  check(st::read_file(all_path, &all, &err), "a trace of every kind reads: " + err);
  check(all.records.size() == (size_t)st::kKinds, "a record of every kind");
  check(st::first_difference(all, all).empty(), "a trace equals itself");
  for (const st::Record& r : all.records) {
    check(!st::perturb_fields(r.kind).empty(), "kind " + r.kind + " has a field --perturb may change");
    for (const std::string& pf : st::perturb_fields(r.kind)) {
      bool in = false;
      for (const std::string& f : st::kind_fields(r.kind)) in = in || f == pf;
      check(in, "kind " + r.kind + ": perturb field " + pf + " is one of its arguments");
    }
    st::Trace c = all;
    std::string what;
    check(st::perturb(&c, r.n, &what), "perturb finds a field in record " + std::to_string(r.n) + " (" + r.kind + ")");
    const std::string d = st::first_difference(all, c);
    check(d.rfind("record " + std::to_string(r.n) + " (" + r.kind + ")", 0) == 0,
          "control: " + what + " changed, reported at its record: '" + d + "'");
  }
  {
    st::Trace c = all;  // a changed digest, a changed result, a swapped answer
    for (st::Line& l : c.lines)
      if (l.tag == 'D' && l.n == 3) l.f[1].value = "0000000000000000", l.raw = "D 3 d=" + l.f[0].value + " w=0000000000000000 out=" + l.f[2].value + " free=" + l.f[3].value + " numbers=" + l.f[4].value;
    const std::string d = st::first_difference(all, c);
    check(d.rfind("record 3", 0) == 0 && d.find("w") != std::string::npos, "control: a changed wide digest is reported at record 3: '" + d + "'");
    st::Trace e = all;
    for (st::Line& l : e.lines)
      if (l.tag == 'R' && l.n == 2) l.f[0].value = "-1", l.raw = "R 2 -1";
    const std::string d2s = st::first_difference(all, e);
    check(d2s.rfind("record 2", 0) == 0, "control: a changed result is reported at record 2: '" + d2s + "'");
    st::Trace sh = all;  // a shorter trace
    sh.lines.resize(all.records[all.records.size() - 2].last + 1);
    sh.records.pop_back();
    check(st::first_difference(all, sh).find("end at different places") != std::string::npos, "a shorter trace is reported");
  }
  {  // what the comparison leaves out, and that it can be told to look
    st::Trace c = all;
    c.lines[1].raw = "geom rank=2 " + c.lines[1].raw.substr(std::string("geom rank=0 ").size());
    c.lines[1].f[0].value = "2";
    check(st::first_difference(all, c).empty(), "the rank is not compared");
    st::Ignore strict;
    strict.header_rank = false;
    check(!st::first_difference(all, c, strict).empty(), "control: with the rank compared it differs");
  }
  {
    st::Trace a, b;
    check(st::parse(good, &a, &err), "reparse");
    b = a;
    for (st::Line& l : b.lines) {
      if (l.tag == 'V') l.f[0].value = "9,9";
      if (l.tag == 'A' && l.f[0].key == "now") l.f[0].value = "123";
      if (l.tag == 'E' && l.f.size() && l.f[0].key == "base") l.f[0].value = "/elsewhere/rank3";
    }
    check(st::first_difference(a, b).empty(), "the clock, the base path and the evicted lists are not compared");
    st::Ignore strict;
    strict.now = strict.base = strict.evicted = false;
    check(!st::first_difference(a, b, strict).empty(), "control: compared, they differ");
  }

  // ---- refused files
  auto refused = [&](const std::string& text, const std::string& want, const std::string& label) {
    st::Trace x;
    std::string e;
    const bool r = st::parse(text, &x, &e);
    check(!r && e.find(want) != std::string::npos, "refused (" + label + "): wanted '" + want + "', got '" + e + "'");
    std::printf("  %-34s -> %s\n", label.c_str(), e.c_str());
  };
  refused("", "empty file", "empty");
  refused(good.substr(0, good.size() - 1), "not complete", "last newline missing");
  refused(good.substr(0, good.size() / 2), "truncated", "cut in half");
  {
    const size_t d9 = good.rfind("D 9 ");
    refused(good.substr(0, d9), "no D line", "last entry without its D");
  }
  {
    const size_t e5 = good.find("E 5 ");
    const size_t d5 = good.find("D 5 ");
    (void)d5;
    refused(good.substr(0, e5 + 10) + "\n", "unknown", "cut inside an argument");
  }
  {
    const size_t p = good.find("\nopen\n");
    refused(good.substr(0, p + 1), "no 'open' line", "held entries, no open");
  }
  refused("pico-store-trace 2\n", "version 2", "another version");
  refused("hello\n", "not a store trace", "not a trace");
  {
    std::string g = good;
    const size_t e4 = g.find("E 4 ");
    g.replace(e4 + 2, 1, "7");
    refused(g, "where 4 is next", "numbers out of order");
  }
  {
    std::string g = good;
    const size_t a1 = g.find("A 1 now");
    g.replace(a1 + 2, 1, "3");
    refused(g, "outside its entry", "an answer under another entry");
  }
  {
    std::string g = good;
    const size_t d1 = g.find("D 1 ");
    g.insert(g.find('\n', d1), " extra=1");
    refused(g, "D line carries", "a D line with an extra field");
  }

  // ---- the engine thread only: another thread entering the store ends the process
  {
    const pid_t pid = fork();
    if (pid == 0) {
      const std::string d3 = tmpdir();
      st::Writer w5(d3);
      st::g_w = &w5;
      Fake f5;
      w5.open(0, kGeom1, kGeom2, 1, 0);
      { st::Scope sc(st::Kind::pin, &f5); }
      std::thread th([&f5] { st::Scope sc(st::Kind::free, &f5); });
      th.join();
      _exit(0);  // not reached
    }
    int status = 0;
    waitpid(pid, &status, 0);
    check(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "control: an entry from another thread aborts (the message is on stderr above)");
  }

  // ---- escapes
  for (const std::string& s : {std::string("a b"), std::string("50%"), std::string("%20"), std::string("x\ny\tz"), std::string(""), std::string("%")})
    check(st::unescape(st::escape(s)) == s && st::escape(s).find(' ') == std::string::npos, "escape round trip of '" + s + "'");

  st::g_w = saved;
  std::printf(g_fail ? "store_trace_test: %d of %d checks FAILED\n" : "store_trace_test: all %d checks passed\n", g_fail ? g_fail : g_checks, g_checks);
  return g_fail ? 1 : 0;
}
