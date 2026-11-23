// The store's trace: every entry into the store's bookkeeping, taken at its boundary (DSGB10_STORE_TRACE=<dir>).
// Host only: no CUDA header. Off (the variable unset), an entry point pays one load of `g_w` and tests of that value held
// in the scope; nothing is built, formatted or written, and no state is read.
//
// What a trace holds. One file per engine start, <dir>/store.rank<R>.<n>.t1, n the first number not yet used in the
// directory; text, one line each, `key=value` fields, lists comma-separated, a space or '%' inside a value escaped as %20 / %25:
//   pico-store-trace 1
//   geom rank=.. nslots=.. nunits=.. list_len=.. max_ctx=.. image=.. snap=.. end=.. spec=0|1 poison=0|1 host_ring=0|1
//   geom unit_of=a,b,.. slot_ctx=a,b,..
//   pool numbers=<the allocator's size when the file opens> disk=<the disk tier's file size, 0: none>
//   pre numbers=<the allocator's size at the first entry before the file opened>      (only when such entries exist)
//   E n kind key=value ..        the n-th entry, with every argument a replay needs to make the same call
//   A n name=value               an answer the device, the ring, the clock or the file system gave during it, in order asked
//   R n value                    what the entry returned, when the caller acts on it
//   D n d=<hex> w=<hex> out=a,b,c,d,e,f free=<n> numbers=<n>
//                                after it: the store's digest (Model::store_digest, unchanged), the wide digest, the six
//                                values of the digest that belong to the windows and the lender (lent.groups, lent.owed,
//                                lent.n0, el.cur_groups, el.phase, the allocator's size), the allocator's free numbers and size
//   open                         (only after `pre`) the allocator now has the size of the `pool` line
//   V m list                     a take of the evicted list: not an entry, not numbered with them (see take_evicted below)
// An entry nested in another (the store calls itself) writes nothing: its effects and answers belong to the outermost one.
// The entries the store gets before its file can open (the disk tier is checked and opened before the pool is made) are held
// and written between `pre` and `open`.
//
// The store is touched by the engine thread only: the scope checks it, and a call from another thread ends the process
// with a message naming both entry points' kinds.
#pragma once
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace model::store_trace {

// Entry kinds: name, every argument, and the arguments `--perturb` may change (each one is mixed into the wide digest by the
// entry's effect, so a replay that applies the changed record has to disagree with the recorded digests).
// KINDS-BEGIN (tools/store_trace_check.py reads these lines)
#define STORE_TRACE_KINDS(X)                                                                                             \
  X(pin, "pin live last", "pin live last")                                                                               \
  X(ensure, "slot upto", "upto")                                                                                         \
  X(slot_reserve, "slot numbers img", "numbers img")                                                                     \
  X(slot_unreserve, "slot", "slot")                                                                                      \
  X(slot_release, "slot", "slot")                                                                                        \
  X(slot_attach, "slot h", "h")                                                                                          \
  X(save, "h from to unit", "to")                                                                                        \
  X(load, "h to unit slot", "to")                                                                                        \
  X(snap_buf, "h pos", "pos")                                                                                            \
  X(snap_get, "h pos", "pos")                                                                                            \
  X(snap_peek, "h pos", "pos")                                                                                           \
  X(end_take, "h pos", "pos")                                                                                            \
  X(end_get, "h", "h")                                                                                                   \
  X(drop, "h from", "from")                                                                                              \
  X(drop_snap, "h pos", "pos")                                                                                           \
  X(free, "h", "h")                                                                                                      \
  X(copy_snaps, "src dst upto", "upto")                                                                                  \
  X(spill, "h", "h")                                                                                                     \
  X(unspill, "h cap upto J", "h")                                                                                            \
  X(flush, "h", "h")                                                                                                     \
  X(disk_resize, "bytes", "bytes")                                                                                       \
  X(kept_check, "base run_id max_age", "run_id")                                                                         \
  X(disk, "base bytes staging reuse run_id", "bytes")                                                                    \
  X(keep, "gen", "gen")                                                                                                  \
  X(checkpoint, "gen", "gen")                                                                                            \
  X(img_take, "key rows slot", "rows")                                                                                   \
  X(img_free, "key", "key")                                                                                              \
  X(evict_step, "h", "h")                                                                                                \
  X(evict_piece, "victim h", "victim")                                                                                   \
  X(vacate, "from", "from")                                                                                              \
  X(make_room, "removed", "removed")                                                                                     \
  X(alloc_shrink, "top", "top")                                                                                          \
  X(alloc_grow, "numbers", "numbers")                                                                                    \
  X(set_factor, "h factor", "factor")                                                                                   \
  X(split, "h budget", "budget")                                                                                        \
  X(join, "h upto cap J", "upto")                                                                                        \
  X(image, "h budget live", "budget")                                                                                    \
  X(settle, "upto live", "upto")
// The answers the store takes from outside itself, in the names the A lines carry.
#define STORE_TRACE_ANSWERS(X) X(now) X(disk_open) X(agree_min) X(index_write) X(kept_gen)
// KINDS-END

enum class Kind : int {
#define X(name, args, perturb) name,
  STORE_TRACE_KINDS(X)
#undef X
      kCount
};
constexpr int kKinds = (int)Kind::kCount;
const char* kind_name(Kind k);
// the kind named `name`, or -1
int kind_index(const std::string& name);
// the arguments `--perturb` may change for the kind (the first is the one the format's own tests use)
std::vector<std::string> perturb_fields(const std::string& kind);
// every argument of the kind, in the order its E line carries them
std::vector<std::string> kind_fields(const std::string& kind);
std::vector<std::string> answer_names();

// What the engine's side reads from the store at the end of each entry (state.cu).
struct Snapshot {
  uint64_t digest = 0;  // Model::store_digest()
  uint64_t wide = 0;    // the wide digest
  uint64_t out[6] = {}; // lent.groups, lent.owed, lent.n0, el.cur_groups, el.phase, the allocator's size
  uint64_t free_numbers = 0, numbers = 0;
};
using SnapshotFn = void (*)(const void* ctx, Snapshot* out);
// The engine registers its reader of the store once, before main (state.cu).
void set_snapshot(SnapshotFn fn);

// The wide digest's mixer: the same multiplicative mix as the store's digest, so one reads like the other.
struct Mixer {
  uint64_t h = 1469598103934665603ull;
  void mix(uint64_t v) { h = (h ^ v) * 1099511628211ull; }
  void mix_i(int64_t v) { mix((uint64_t)v); }
  void mix_u32(uint32_t v) { mix((uint64_t)v); }
  void mix_str(const std::string& s) {
    mix(s.size());
    for (const unsigned char c : s) mix(c);
  }
};

class Writer {
 public:
  // The trace in `dir` (which must exist). The file is made when `open` is called.
  explicit Writer(std::string dir);
  ~Writer();
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;

  // Make the file: the header (version line, the geom lines, the pool line), then what was held before it. n is the first
  // number not yet used in the directory. A directory that cannot be written ends the process.
  void open(int rank, const std::string& geom1, const std::string& geom2, uint64_t pool_numbers, uint64_t disk_bytes);
  bool is_open() const { return f_ != nullptr; }
  const std::string& path() const { return path_; }

  // an entry: true when it is the outermost (the one that writes)
  bool enter(Kind k, const void* ctx);
  void leave(bool top);
  void add_arg(const char* key, const std::string& value);
  void answer(const char* name, const std::string& value);
  void set_result(const std::string& value);
  void set_result_fn(std::function<int64_t()> fn) { result_fn_ = std::move(fn); }
  // an answer that is read when the entry ends (a value the body leaves in a member)
  void answer_at_exit(const char* name, const uint64_t* p) { exit_answers_.emplace_back(name, p); }
  // a take of the evicted list (not an entry)
  void evicted(const std::vector<int>& list);
  uint64_t entries() const { return seq_; }

 private:
  void emit(const std::string& text);
  void check_thread(const char* what);
  std::string dir_;
  FILE* f_ = nullptr;
  std::string path_;
  int depth_ = 0;       // a plain member: the store is touched by one thread (check_thread)
  std::thread::id owner_;
  bool owner_set_ = false;
  Kind outer_ = Kind::pin;
  const void* ctx_ = nullptr;
  uint64_t seq_ = 0, vseq_ = 0;
  std::string pre_;     // records made before the file opened
  uint64_t pre_numbers_ = 0, pre_n_ = 0;
  // the open entry
  std::vector<std::pair<std::string, std::string>> args_, answers_;
  std::vector<std::pair<const char*, const uint64_t*>> exit_answers_;
  std::string result_;
  bool has_result_ = false;
  std::function<int64_t()> result_fn_;
};

// null: the option is off. `init` makes the writer when DSGB10_STORE_TRACE is set; the model's constructor calls it, after the
// command line (--setenv) has been read. A test sets the pointer itself.
extern Writer* g_w;
void init();
inline bool on() { return g_w != nullptr; }
// the directory named by DSGB10_STORE_TRACE, or empty
std::string option_dir();

std::string escape(const std::string& v);
std::string unescape(const std::string& v);
std::string csv(const std::vector<int>& v);

// The guard at the top of an entry point. `ctx` is what the snapshot function needs (the engine's Impl).
class Scope {
 public:
  Scope(Kind k, const void* ctx) : w_(g_w) {
    if (__builtin_expect(w_ != nullptr, 0)) top_ = w_->enter(k, ctx);
  }
  ~Scope() {
    if (__builtin_expect(w_ != nullptr, 0)) w_->leave(top_);
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  bool top() const { return top_; }
  Scope& arg(const char* key, int64_t v) {
    if (top_) w_->add_arg(key, std::to_string(v));
    return *this;
  }
  Scope& arg_u(const char* key, uint64_t v) {
    if (top_) w_->add_arg(key, std::to_string(v));
    return *this;
  }
  Scope& arg_d(const char* key, double v) {
    if (top_) {
      char b[40];
      std::snprintf(b, sizeof b, "%.17g", v);
      w_->add_arg(key, b);
    }
    return *this;
  }
  Scope& arg_s(const char* key, const std::string& v) {
    if (top_) w_->add_arg(key, escape(v));
    return *this;
  }
  Scope& arg_v(const char* key, const std::vector<int>& v) {
    if (top_) w_->add_arg(key, csv(v));
    return *this;
  }
  void result(int64_t v) {
    if (top_) w_->set_result(std::to_string(v));
  }
  // evaluated when the entry ends, so it reads the state the entry left; capture the engine and the arguments, not locals
  template <class F>
  void result_fn(F&& f) {
    if (top_) w_->set_result_fn(std::function<int64_t()>(std::forward<F>(f)));
  }
  // the value an entry returns, recorded and passed on unchanged
  template <class T>
  T ret(T v) {
    if (top_) w_->set_result(std::to_string((int64_t)v));
    return v;
  }
  template <class V>
  const V* ret_list(const V* p) {  // a bookmark's numbers: their count, or -1 for none
    if (top_) w_->set_result(std::to_string(p ? (int64_t)p->size() : (int64_t)-1));
    return p;
  }
  void answer_at_exit(const char* name, const uint64_t* p) {
    if (top_) w_->answer_at_exit(name, p);
  }

 private:
  Writer* w_;
  bool top_ = false;
};

// An answer from outside the store, recorded in the open entry (outside any entry: as `A 0`, which a replay refuses).
inline void answer(const char* name, int64_t v) {
  if (__builtin_expect(g_w != nullptr, 0)) g_w->answer(name, std::to_string(v));
}
inline void evicted(const std::vector<int>& list) {
  if (__builtin_expect(g_w != nullptr, 0)) g_w->evicted(list);
}

// ---- reading ------------------------------------------------------------------------------------------------------

struct Field {
  std::string key, value;
};
struct Line {
  char tag = 0;        // 'E' 'A' 'R' 'D' 'V', or 'H' for a header line, 'P' for `pre`, 'O' for `open`
  uint64_t n = 0;      // the entry's number
  std::string word;    // E: the kind; H: geom / pool / pico-store-trace
  std::vector<Field> f;
  std::string raw;
};
struct Record {        // one entry: its E line, A lines, R line and D line
  uint64_t n = 0;
  std::string kind;
  size_t first = 0, last = 0;  // [first, last] in Trace::lines
};
struct Trace {
  std::vector<Line> lines;
  std::vector<Record> records;
  const Field* find(const Line& l, const char* key) const;
};
// Parse a trace file's text. false, with `err` naming the line and the reason, for anything that is not a whole trace: a
// missing or other version line, a line that is not one of the forms above, numbers out of order, an entry without its D line,
// a file that does not end in a newline.
bool parse(const std::string& text, Trace* out, std::string* err);
bool read_file(const std::string& path, Trace* out, std::string* err);  // popen("gzip -dc") for a name ending in .gz

// What a comparison leaves out: the rank, the files' own path, the clock values and the V lines.
struct Ignore {
  bool header_rank = true, base = true, now = true, evicted = true;
};
// "" when the traces are equal (after the ignored fields), else the first difference: the record, its kind and the field.
std::string first_difference(const Trace& a, const Trace& b, const Ignore& ig = Ignore());
// change one argument of record n (its first perturbable field that is there) by one; false when n has none. `what` names it.
bool perturb(Trace* t, uint64_t n, std::string* what);
// the text of a trace (write what read parsed: used by the tools and the tests)
std::string to_text(const Trace& t);

}  // namespace model::store_trace
