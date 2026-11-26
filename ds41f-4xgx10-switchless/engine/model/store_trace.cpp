// The store's trace (store_trace.h): the writer, the reader and the comparison. Host only.
#include "model/store_trace.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace model::store_trace {

namespace {

SnapshotFn g_snap = nullptr;  // constant-initialised: set by state.cu's static initialiser in whatever order

struct KindRow {
  const char* name;
  const char* args;
  const char* perturb;
};
const KindRow kRows[] = {
#define X(name, args, perturb) {#name, args, perturb},
    STORE_TRACE_KINDS(X)
#undef X
};

std::vector<std::string> words(const char* s) {
  std::vector<std::string> v;
  std::istringstream is(s);
  for (std::string w; is >> w;) v.push_back(w);
  return v;
}

[[noreturn]] void fatal(const std::string& msg) {
  std::fprintf(stderr, "store trace: %s\n", msg.c_str());
  std::fflush(stderr);
  std::abort();
}

}  // namespace

Writer* g_w = nullptr;

void init() {
  if (g_w) return;
  const std::string d = option_dir();
  if (!d.empty()) g_w = new Writer(d);
}

std::string option_dir() {
  const char* e = std::getenv("DSGB10_STORE_TRACE");
  return e ? e : "";
}

void set_snapshot(SnapshotFn fn) { g_snap = fn; }

const char* kind_name(Kind k) { return kRows[(int)k].name; }
int kind_index(const std::string& name) {
  for (int i = 0; i < kKinds; ++i)
    if (name == kRows[i].name) return i;
  return -1;
}
std::vector<std::string> perturb_fields(const std::string& kind) {
  const int i = kind_index(kind);
  return i < 0 ? std::vector<std::string>() : words(kRows[i].perturb);
}
std::vector<std::string> kind_fields(const std::string& kind) {
  const int i = kind_index(kind);
  return i < 0 ? std::vector<std::string>() : words(kRows[i].args);
}
std::vector<std::string> answer_names() {
  std::vector<std::string> v;
#define X(name) v.push_back(#name);
  STORE_TRACE_ANSWERS(X)
#undef X
  return v;
}

std::string escape(const std::string& v) {
  std::string o;
  for (const char c : v) {
    if (c == '%') o += "%25";
    else if (c == ' ') o += "%20";
    else if (c == '\n') o += "%0A";
    else if (c == '\r') o += "%0D";
    else if (c == '\t') o += "%09";
    else o += c;
  }
  return o;
}

std::string unescape(const std::string& v) {
  std::string o;
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i] == '%' && i + 2 < v.size() + 0 && std::isxdigit((unsigned char)v[i + 1]) && std::isxdigit((unsigned char)v[i + 2])) {
      o += (char)std::strtol(v.substr(i + 1, 2).c_str(), nullptr, 16);
      i += 2;
    } else {
      o += v[i];
    }
  }
  return o;
}

std::string csv(const std::vector<int>& v) {
  std::string o;
  for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
  return o;
}

// ---- the writer ---------------------------------------------------------------------------------------------------

Writer::Writer(std::string dir) : dir_(std::move(dir)) {}

Writer::~Writer() {
  if (f_) std::fclose(f_);
}

void Writer::check_thread(const char* what) {
  const std::thread::id me = std::this_thread::get_id();
  if (!owner_set_) {
    owner_ = me, owner_set_ = true;
    return;
  }
  if (me != owner_) fatal(std::string(what) + " entered the store from a thread other than the engine thread");
}

void Writer::emit(const std::string& text) {
  if (f_) {
    if (std::fwrite(text.data(), 1, text.size(), f_) != text.size() || std::fflush(f_) != 0)
      fatal("cannot write " + path_ + ": " + std::strerror(errno));
  } else {
    pre_ += text;
  }
}

void Writer::open(int rank, const std::string& geom1, const std::string& geom2, uint64_t pool_numbers, uint64_t disk_bytes) {
  if (f_) fatal("the trace file is opened once");
  if (depth_) fatal("the trace file opened inside an entry");
  for (int n = 1;; ++n) {
    path_ = dir_ + "/store.rank" + std::to_string(rank) + "." + std::to_string(n) + ".t1";
    f_ = std::fopen(path_.c_str(), "wx");  // exclusive: the first number not yet used
    if (f_) break;
    if (errno != EEXIST) fatal("cannot create " + path_ + ": " + std::strerror(errno));
  }
  std::string h = "pico-store-trace 1\ngeom " + geom1 + "\ngeom " + geom2 + "\npool numbers=" + std::to_string(pool_numbers) +
                  " disk=" + std::to_string(disk_bytes) + "\n";
  if (!pre_.empty()) h += "pre numbers=" + std::to_string(pre_numbers_) + "\n" + pre_ + "open\n";
  pre_.clear();
  emit(h);
  owner_set_ = false;  // the entries before the file were the start's; the engine thread is whoever enters next
}

bool Writer::enter(Kind k, const void* ctx) {
  check_thread(kRows[(int)k].name);
  if (depth_++ > 0) return false;
  outer_ = k, ctx_ = ctx;
  args_.clear(), answers_.clear(), exit_answers_.clear();
  has_result_ = false, result_fn_ = nullptr;
  if (!f_ && pre_.empty() && g_snap) {  // the allocator's size when the first entry before the file began
    Snapshot s;
    g_snap(ctx, &s);
    pre_numbers_ = s.numbers;
  }
  return true;
}

void Writer::add_arg(const char* key, const std::string& value) { args_.emplace_back(key, value); }

void Writer::answer(const char* name, const std::string& value) {
  if (depth_ == 0) {
    emit("A 0 " + std::string(name) + "=" + value + "\n");
    return;
  }
  answers_.emplace_back(name, value);
}

void Writer::set_result(const std::string& value) {
  result_ = value;
  has_result_ = true;
}

void Writer::evicted(const std::vector<int>& list) {
  check_thread("take_evicted");
  emit("V " + std::to_string(++vseq_) + " list=" + csv(list) + "\n");
}

void Writer::leave(bool top) {
  --depth_;
  if (!top) return;
  const uint64_t n = ++seq_;
  const std::string ns = std::to_string(n);
  std::string t = "E " + ns + " " + kRows[(int)outer_].name;
  for (const auto& a : args_) t += " " + a.first + "=" + a.second;
  t += "\n";
  for (const auto& a : answers_) t += "A " + ns + " " + a.first + "=" + a.second + "\n";
  for (const auto& a : exit_answers_) t += "A " + ns + " " + a.first + "=" + std::to_string(*a.second) + "\n";
  if (result_fn_) set_result(std::to_string(result_fn_()));
  if (has_result_) t += "R " + ns + " " + result_ + "\n";
  Snapshot s;
  if (g_snap) g_snap(ctx_, &s);
  char b[320];
  std::snprintf(b, sizeof b, "D %s d=%016llx w=%016llx out=%llu,%llu,%llu,%llu,%llu,%llu free=%llu numbers=%llu\n", ns.c_str(),
                (unsigned long long)s.digest, (unsigned long long)s.wide, (unsigned long long)s.out[0], (unsigned long long)s.out[1],
                (unsigned long long)s.out[2], (unsigned long long)s.out[3], (unsigned long long)s.out[4],
                (unsigned long long)s.out[5], (unsigned long long)s.free_numbers, (unsigned long long)s.numbers);
  t += b;
  if (!f_) ++pre_n_;
  emit(t);
}

// ---- the reader ---------------------------------------------------------------------------------------------------

const Field* Trace::find(const Line& l, const char* key) const {
  for (const Field& f : l.f)
    if (f.key == key) return &f;
  return nullptr;
}

namespace {

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  size_t i = 0;
  while (i < s.size()) {
    const size_t j = s.find(' ', i);
    v.push_back(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return v;
}

bool is_number(const std::string& s, uint64_t* v) {
  if (s.empty() || s.size() > 19) return false;
  uint64_t x = 0;
  for (const char c : s) {
    if (c < '0' || c > '9') return false;
    x = x * 10 + (uint64_t)(c - '0');
  }
  *v = x;
  return true;
}

bool fields_of(const std::vector<std::string>& t, size_t from, std::vector<Field>* out) {
  for (size_t i = from; i < t.size(); ++i) {
    const size_t e = t[i].find('=');
    if (e == std::string::npos || e == 0) return false;
    out->push_back({t[i].substr(0, e), t[i].substr(e + 1)});
  }
  return true;
}

}  // namespace

bool parse(const std::string& text, Trace* out, std::string* err) {
  *out = Trace();
  const auto fail = [&](size_t line, const std::string& why) {
    *err = "line " + std::to_string(line) + ": " + why;
    return false;
  };
  if (text.empty()) {
    *err = "empty file: not a trace";
    return false;
  }
  if (text.back() != '\n') {
    size_t ln = 1;
    for (const char c : text) ln += c == '\n';
    *err = "truncated: line " + std::to_string(ln) + " is not complete (the file does not end in a newline)";
    return false;
  }
  size_t ln = 0, pos = 0;
  bool have_version = false, have_pool = false, pre_open = false, in_rec = false, in_body = false;
  int geoms = 0;
  uint64_t next = 1, open_n = 0;
  size_t open_first = 0;
  while (pos < text.size()) {
    const size_t e = text.find('\n', pos);
    const std::string raw = text.substr(pos, e - pos);
    pos = e + 1;
    ++ln;
    const std::vector<std::string> t = split(raw);
    if (t.empty()) return fail(ln, "an empty line");
    Line l;
    l.raw = raw;
    if (!have_version) {
      if (t.size() != 2 || t[0] != "pico-store-trace") return fail(ln, "not a store trace (no 'pico-store-trace 1' first line)");
      if (t[1] != "1") return fail(ln, "trace version " + t[1] + ": this reader reads version 1");
      have_version = true;
      l.tag = 'H', l.word = t[0], l.f.push_back({"version", t[1]});
      out->lines.push_back(l);
      continue;
    }
    const std::string& w = t[0];
    if (w == "geom" || w == "pool" || w == "pre") {
      if (in_rec || in_body) return fail(ln, "'" + w + "' after the first entry");
      l.tag = w == "pre" ? 'P' : 'H', l.word = w;
      if (!fields_of(t, 1, &l.f)) return fail(ln, "a field that is not key=value");
      if (w == "geom") ++geoms;
      if (w == "pool") have_pool = true;
      if (w == "pre") {
        if (!have_pool) return fail(ln, "'pre' before the pool line");
        pre_open = true;
      }
      out->lines.push_back(l);
      continue;
    }
    if (!have_pool || geoms < 2) return fail(ln, "an entry before the header's geom and pool lines");
    if (w == "open") {
      if (!pre_open || in_rec || t.size() != 1) return fail(ln, "'open' where no entries before the file are pending");
      pre_open = false, in_body = true;
      l.tag = 'O', l.word = w;
      out->lines.push_back(l);
      continue;
    }
    if (pre_open == false && !in_body) in_body = true;
    if (w == "V") {
      if (in_rec) return fail(ln, "a 'V' line inside entry " + std::to_string(open_n));
      l.tag = 'V';
      if (t.size() != 3 || !is_number(t[1], &l.n) || !fields_of(t, 2, &l.f)) return fail(ln, "a V line is 'V m list=..'");
      out->lines.push_back(l);
      continue;
    }
    if (w != "E" && w != "A" && w != "R" && w != "D") return fail(ln, "unknown line '" + w + "'");
    l.tag = w[0];
    if (t.size() < 3 || !is_number(t[1], &l.n)) return fail(ln, "a '" + w + "' line is '" + w + " n ..'");
    if (l.tag == 'E') {
      if (in_rec) return fail(ln, "entry " + std::to_string(l.n) + " begins before entry " + std::to_string(open_n) + " has its D line");
      if (l.n != next) return fail(ln, "entry " + std::to_string(l.n) + " where " + std::to_string(next) + " is next");
      l.word = t[2];
      if (kind_index(l.word) < 0) return fail(ln, "unknown entry kind '" + l.word + "'");
      if (!fields_of(t, 3, &l.f)) return fail(ln, "an argument that is not key=value");
      in_rec = true, open_n = l.n, open_first = out->lines.size();
      out->lines.push_back(l);
      continue;
    }
    if (l.tag == 'A' && l.n == 0) {  // an answer taken outside any entry: kept, for the tools to report
      if (t.size() != 3 || !fields_of(t, 2, &l.f)) return fail(ln, "an A line is 'A n name=value'");
      out->lines.push_back(l);
      continue;
    }
    if (!in_rec || l.n != open_n) return fail(ln, "'" + w + " " + std::to_string(l.n) + "' outside its entry");
    if (l.tag == 'A') {
      if (t.size() != 3 || !fields_of(t, 2, &l.f)) return fail(ln, "an A line is 'A n name=value'");
    } else if (l.tag == 'R') {
      if (t.size() != 3) return fail(ln, "an R line is 'R n value'");
      l.f.push_back({"value", t[2]});
    } else {
      if (!fields_of(t, 2, &l.f) || l.f.size() != 5) return fail(ln, "a D line carries d, w, out, free and numbers");
      Record r;
      r.n = open_n, r.kind = out->lines[open_first].word, r.first = open_first, r.last = out->lines.size();
      out->records.push_back(r);
      in_rec = false, ++next;
    }
    out->lines.push_back(l);
  }
  if (in_rec) {
    *err = "truncated: entry " + std::to_string(open_n) + " (" + out->lines[open_first].word + ") has no D line";
    return false;
  }
  if (pre_open) {
    *err = "truncated: the entries before the file opened have no 'open' line";
    return false;
  }
  if (!have_pool) {
    *err = "truncated: no pool line";
    return false;
  }
  return true;
}

bool read_file(const std::string& path, Trace* out, std::string* err) {
  std::string text;
  const bool gz = path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
  FILE* f = nullptr;
  if (gz) {
    if (path.find('\'') != std::string::npos) {
      *err = "a quote in the path " + path;
      return false;
    }
    f = popen(("gzip -dc '" + path + "'").c_str(), "r");
  } else {
    f = std::fopen(path.c_str(), "rb");
  }
  if (!f) {
    *err = "cannot open " + path + ": " + std::strerror(errno);
    return false;
  }
  char b[65536];
  for (size_t n; (n = std::fread(b, 1, sizeof b, f)) > 0;) text.append(b, n);
  if (gz) {
    if (pclose(f) != 0) {
      *err = path + ": gzip -dc failed";
      return false;
    }
  } else {
    std::fclose(f);
  }
  if (!parse(text, out, err)) {
    *err = path + ": " + *err;
    return false;
  }
  return true;
}

std::string to_text(const Trace& t) {
  std::string s;
  for (const Line& l : t.lines) s += l.raw + "\n";
  return s;
}

namespace {

// a line's text with the fields a comparison leaves out removed
std::string canon(const Line& l, const Ignore& ig) {
  std::string s;
  s += l.tag;
  s += ' ' + std::to_string(l.n) + ' ' + l.word;
  for (const Field& f : l.f) {
    if (ig.header_rank && l.tag == 'H' && l.word == "geom" && f.key == "rank") continue;
    if (ig.base && f.key == "base") continue;
    if (ig.now && l.tag == 'A' && f.key == "now") {
      s += " now=*";
      continue;
    }
    s += " " + f.key + "=" + f.value;
  }
  return s;
}

std::string where(const Trace& t, size_t i) {
  for (const Record& r : t.records)
    if (i >= r.first && i <= r.last) return "record " + std::to_string(r.n) + " (" + r.kind + ")";
  return "the header or before the first entry";
}

}  // namespace

std::string first_difference(const Trace& a, const Trace& b, const Ignore& ig) {
  std::vector<size_t> ia, ib;
  for (size_t i = 0; i < a.lines.size(); ++i)
    if (!(ig.evicted && a.lines[i].tag == 'V')) ia.push_back(i);
  for (size_t i = 0; i < b.lines.size(); ++i)
    if (!(ig.evicted && b.lines[i].tag == 'V')) ib.push_back(i);
  const size_t n = std::min(ia.size(), ib.size());
  for (size_t k = 0; k < n; ++k) {
    const Line &x = a.lines[ia[k]], &y = b.lines[ib[k]];
    if (canon(x, ig) == canon(y, ig)) continue;
    std::string why = std::string(1, x.tag) + " line";
    if (x.tag != y.tag || x.word != y.word || x.n != y.n) {
      why = "line '" + x.raw + "' against '" + y.raw + "'";
    } else {
      for (size_t j = 0; j < std::max(x.f.size(), y.f.size()); ++j) {
        const std::string xk = j < x.f.size() ? x.f[j].key : "(none)", yk = j < y.f.size() ? y.f[j].key : "(none)";
        const std::string xv = j < x.f.size() ? x.f[j].value : "", yv = j < y.f.size() ? y.f[j].value : "";
        if (xk == yk && xv == yv) continue;
        if (ig.header_rank && xk == "rank") continue;
        why = std::string(1, x.tag) + " field " + xk + ": " + xv + " != " + (xk == yk ? yv : yk + "=" + yv);
        break;
      }
    }
    return where(a, ia[k]) + ": " + why;
  }
  if (ia.size() != ib.size()) {
    const Trace& longer = ia.size() > ib.size() ? a : b;
    const std::vector<size_t>& li = ia.size() > ib.size() ? ia : ib;
    return std::string("the traces end at different places: ") + (ia.size() > ib.size() ? "the first" : "the second") +
           " goes on with " + where(longer, li[n]);
  }
  return "";
}

namespace {
std::string render(const Line& l) {
  std::string s;
  if (l.tag == 'E') s = "E " + std::to_string(l.n) + " " + l.word;
  else if (l.tag == 'R') return "R " + std::to_string(l.n) + " " + l.f[0].value;
  else if (l.tag == 'A' || l.tag == 'D' || l.tag == 'V') s = std::string(1, l.tag) + " " + std::to_string(l.n);
  else s = l.word;
  for (const Field& f : l.f) s += " " + f.key + "=" + f.value;
  return s;
}
}  // namespace

bool perturb(Trace* t, uint64_t n, std::string* what) {
  for (const Record& r : t->records) {
    if (r.n != n) continue;
    Line& e = t->lines[r.first];
    for (const std::string& name : perturb_fields(r.kind)) {
      for (Field& f : e.f) {
        if (f.key != name) continue;
        const size_t c = f.value.find(',');
        const std::string head = f.value.substr(0, c), rest = c == std::string::npos ? "" : f.value.substr(c);
        char* end = nullptr;
#ifndef PICO_OLD_RULES
        // a value that is not negative is read unsigned (ids and generations are 64-bit), and the top of the range steps down
        // instead of up: the changed value is always another number of the same kind, never an overflow
        std::string next;
        if (!head.empty() && head[0] != '-') {
          const unsigned long long u = std::strtoull(head.c_str(), &end, 10);
          if (*end == 0) next = std::to_string(u == ULLONG_MAX ? u - 1 : u + 1);
        } else if (!head.empty()) {
          const long long v = std::strtoll(head.c_str(), &end, 10);
          if (*end == 0) next = std::to_string(v + 1);  // v < 0: no overflow
        }
        if (head.empty()) {
          f.value = rest.empty() ? "1" : "1" + rest;
        } else if (!next.empty()) {
          f.value = next + rest;
        } else {
          continue;
        }
#else
        const long long v = std::strtoll(head.c_str(), &end, 10);
        if (head.empty()) {
          f.value = rest.empty() ? "1" : "1" + rest;
        } else if (*end == 0) {
          f.value = std::to_string(v + 1) + rest;
        } else {
          continue;
        }
#endif
        e.raw = render(e);
        if (what) *what = r.kind + "." + name;
        return true;
      }
    }
    return false;
  }
  return false;
}

}  // namespace model::store_trace
