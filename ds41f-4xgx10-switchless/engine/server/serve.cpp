// Rank 0 as a server (serve.h, docs/protocol.md).
//
// Threads: the reader takes the front-end's frames (a new connection replaces the old one, whose
// requests then count as cancelled); the writer sends HELLO / TOKENS / DONE, so the engine never
// blocks on the socket; the engine (the caller's thread) runs the requests, every model call mirrored on
// the other ranks by a command (cmd.h).
//
// Scheduling (docs/batching.md, B1 and B2). A request's first run prefills its prompt into a session slot
// and sends its first token; then it decodes with the others, as many as there are slots, a row each
// per step (batch_loop), or alone through the speculative loop (with drafts). A request that comes while
// others decode pauses them (prefill first), if it can have a slot: a free one, one whose session no
// request uses, or, with time-slicing (--slice), the slot of the paused request whose turn comes last.
// While more requests decode than there are slots, they take turns of --slice tokens. One slot is B1:
// one request decodes at a time, and a new one pauses it.
//
// Sessions. Each slot's working state (window rings, compressor carries, compressed caches) belongs to
// one session, live there. Every other kept session lives in the store (state.cu): its compressed
// caches by pages, an end snapshot (the whole ring and the carries at its last position) and window
// snapshots (the ring's last 128 positions) at the end of its last prompt and every snap_every positions,
// thinned further back (keep_snap). A request continues from the longest common prefix p of the session's history and its prompt
// (at most n - 1: the last prompt position runs again for its logits), restarting at the largest
// position q <= p from which the state can be rebuilt exactly:
//   - live: the working ring still holds q - 127 .. q - 1 when q >= E - 128 (E: the history's end),
//     and a ratio-2 entry completing at an odd q finds q - 1's projection in the carry when q - 1 >= E - 16;
//   - the end snapshot: the same about its position;
//   - a window snapshot at s: q = s (s even) or s - 1 (no carry is kept with it, so q is even);
//   - otherwise q = 0.
// The compressed caches are per position and append-only: entries below q are kept as they are (in the
// working buffers, or loaded from the store's pages); entries from q on are rewritten as the prefill
// passes them, and no row reads an entry beyond its own position.
// The ring's entries are exact for positions [max(lo, w - 256), end) and the carries' for [max(clo,
// w - 16), end): lo is where the state's lineage begins (a restored snapshot's first position), w past
// the last position anything wrote (a queued step, rejected drafts, the draft module's block keys).
// The draft module's windows (DSpark) follow the ring: every snapshot and restart carries them.
//
// CED (ServeOptions::ced, docs/ced.md): a turn of at least 128 new positions (n - q >= 128) runs the
// encoder over them and replays the decoder over [n - 128, n) from fresh rings, so it needs only the
// encoder's state at q; a shorter one runs every layer (an exact continuation) and needs the decoder
// layers' rings too. Those are not valid everywhere: window snapshots taken inside a CED prefill hold
// the encoder's rings only (Session::enc_snaps), and the live rings are valid from dec_lo on (from the
// replay's first position after a CED turn; nowhere after a cancelled one). The draft module's windows
// are the decoder's: the replay seeds them, and the encoder's snapshots hold none.
//
// Kept sessions (docs/engine.md): a planned stop moves the sessions to the disk tier and every rank
// indexes its file (state.cu); this server writes their histories beside it (keep). The next start
// whose store reused those files reads them back in (load_kept), and the front-end's session ids find
// their sessions as before the stop.
#include "server/serve.h"
#include "server/bounds.h"
#include "server/cancel.h"
#include "server/reclaim.h"
#include "server/turns.h"

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include <cuda_profiler_api.h>

#include "common/check.h"
#include "common/heap.h"
#include "model/nvtx.h"
#include "common/frontend_protocol.h"
#include "common/keepfile.h"

namespace srv {
namespace {

// Image rows plus room for a 1M-token prompt and framing; matches frontend/src/protocol.rs.
constexpr uint32_t kMaxFrame = fe::kMaxImages * fe::kMaxImageRows * fe::kImageDim * 2u + (8u << 20);
constexpr int kIoCpu = 12;

std::atomic<bool> g_stop{false};
bool g_prof = false;  // rank 0 opened nsys's capture range (--profile-from-req)
void on_signal(int) { g_stop = true; }

bool read_all(int fd, void* p, size_t n) {
  auto* b = static_cast<char*>(p);
  while (n > 0) {
    const ssize_t r = read(fd, b, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    b += r;
    n -= size_t(r);
  }
  return true;
}

bool write_all(int fd, const void* p, size_t n) {
  auto* b = static_cast<const char*>(p);
  while (n > 0) {
    const ssize_t r = write(fd, b, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    b += r;
    n -= size_t(r);
  }
  return true;
}

uint64_t us(double s) { return s > 0 ? uint64_t(s * 1e6) : 0; }

double unix_now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

// One front-end connection. Once it is closed or replaced its requests count as cancelled and frames
// for it are dropped; the descriptor closes with the last reference.
struct Conn {
  const int fd;
  std::atomic<bool> dead{false};
  std::mutex mu;
  cancel::Set reqs;  // the requests in flight and the CANCELs received for them (ids are per connection; under mu)
  bounds::OutQueue outq;  // the bytes queued for the writer; past bounds::kOutCap the connection is dropped (Io::send)
  explicit Conn(int f) : fd(f) {}
  ~Conn() { close(fd); }
  void kill() {
    if (!dead.exchange(true)) shutdown(fd, SHUT_RDWR);
  }
};
using ConnP = std::shared_ptr<Conn>;

struct Request {
  ConnP conn;
  fe::SubmitHeader h{};
  std::vector<int32_t> stop, prompt;
  double t_recv = 0;
  // vision (fe::kFlagImages): the image spans and their rows, bf16 [sum rows][kImageDim]; empty for text
  std::vector<fe::ImageSpan> spans;
  std::vector<uint16_t> img;
};

struct Event {
  bool release = false;
  uint64_t session = 0;  // release
  Request rq;            // submit
};

// A frame: header, fixed struct, then arrays.
struct Frame {
  std::vector<char> buf;
  explicit Frame(uint16_t type) : buf(sizeof(fe::FrameHeader)) {
    const fe::FrameHeader h{0, type, 0};
    std::memcpy(buf.data(), &h, sizeof h);
  }
  template <class T>
  Frame& add(const T* p, size_t n = 1) {
    const char* c = reinterpret_cast<const char*>(p);
    buf.insert(buf.end(), c, c + n * sizeof(T));
    return *this;
  }
  std::vector<char> take() {
    const uint32_t size = uint32_t(buf.size() - sizeof(fe::FrameHeader));
    std::memcpy(buf.data(), &size, sizeof size);
    return std::move(buf);
  }
};

class Io {
 public:
  Io(const std::string& path, std::vector<char> hello) : path_(path), hello_(std::move(hello)) {
    lfd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (lfd_ < 0 || path.size() >= sizeof addr.sun_path) die("serve: cannot make the socket %s", path.c_str());
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    unlink(path.c_str());
    const mode_t old = umask(0177);  // the socket is created 0600
    const int rc = bind(lfd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
    umask(old);
    if (rc != 0 || listen(lfd_, 4) != 0) die("serve: cannot listen on %s: %s", path.c_str(), std::strerror(errno));
    chmod(path.c_str(), 0600);
    rd_ = std::thread([this] {
      pin_self(kIoCpu);
      reader();
    });
    wr_ = std::thread([this] {
      pin_self(kIoCpu);
      writer();
    });
  }
  ~Io() {
    stop_ = true;
    wcv_.notify_all();
    rd_.join();
    wr_.join();
    close(lfd_);
    unlink(path_.c_str());
  }
  bool pop(Event& e, int timeout_ms) {
    std::unique_lock<std::mutex> l(mu_);
    if (!cv_.wait_for(l, std::chrono::milliseconds(timeout_ms), [&] { return !ev_.empty(); })) return false;
    e = std::move(ev_.front());
    ev_.pop_front();
    return true;
  }
  // a SUBMIT waits to be taken (time-slicing pauses the decode in flight for it)
  bool submit_waiting() {
    std::lock_guard<std::mutex> l(mu_);
    for (const Event& e : ev_)
      if (!e.release && !e.rq.conn->dead) return true;
    return false;
  }
  // The dashboard's STATS (docs/dashboard.md): the main loop publishes a frame, the reader answers each StatsReq with the
  // latest one at once, so the reader never touches the model
  void publish_stats(std::vector<char> f) {
    std::lock_guard<std::mutex> l(smu_);
    stats_ = std::move(f);
  }
  // The latest CONFIG frame, once (the main loop applies it)
  bool take_config(fe::Config& c) {
    std::lock_guard<std::mutex> l(cmu_);
    if (!cfg_new_) return false;
    c = cfg_, cfg_new_ = false;
    return true;
  }
  // Queues a frame for c's writer. A connection whose queue passes bounds::kOutCap (its reader stopped) is dropped.
  void send(const ConnP& c, std::vector<char> f) {
    if (c->dead) return;
    if (!c->outq.add(f.size())) {
      std::printf("serve: front-end dropped: %zu MiB of frames waiting for it\n", c->outq.queued() >> 20);
      c->kill();
      return;
    }
    {
      std::lock_guard<std::mutex> l(wmu_);
      out_.emplace_back(c, std::move(f));
    }
    wcv_.notify_one();
  }
  static bool cancelled(const Request& r) {
    if (r.conn->dead || g_stop) return true;  // a stopping server ends the request in flight
    std::lock_guard<std::mutex> l(r.conn->mu);
    return r.conn->reqs.cancelled(r.h.req_id);
  }
  // the server's stop ends r, not its client or its front-end going away
  static bool stopping(const Request& r) {
    if (!g_stop || r.conn->dead) return false;
    std::lock_guard<std::mutex> l(r.conn->mu);
    return !r.conn->reqs.cancelled(r.h.req_id);
  }
  // r's request is over: its id leaves the connection's live set and its cancel mark with it
  static void forget(const Request& r) {
    std::lock_guard<std::mutex> l(r.conn->mu);
    r.conn->reqs.forget(r.h.req_id);
  }

 private:
  void push(Event&& e) {
    {
      std::lock_guard<std::mutex> l(mu_);
      ev_.push_back(std::move(e));
    }
    cv_.notify_one();
  }

  // One frame from c: false when the connection is to be closed.
  bool frame(const ConnP& c) {
    fe::FrameHeader h;
    if (!read_all(c->fd, &h, sizeof h) || h.size > kMaxFrame) return false;
    std::vector<char> p(h.size);
    if (!p.empty() && !read_all(c->fd, p.data(), p.size())) return false;
    if (h.type == fe::kHello) {
      fe::HelloFe hi;
      if (p.size() < sizeof hi) return false;
      std::memcpy(&hi, p.data(), sizeof hi);
      if (hi.version != fe::kVersion) {
        std::fprintf(stderr, "serve: front-end speaks version %u, not %u\n", hi.version, fe::kVersion);
        return false;
      }
    } else if (h.type == fe::kSubmit) {
      Event e;
      Request& r = e.rq;
      if (p.size() < sizeof r.h) return false;
      std::memcpy(&r.h, p.data(), sizeof r.h);
      const size_t base = sizeof r.h + 4 * (size_t(r.h.n_stop_ids) + r.h.n_prompt);
      if (!(r.h.flags & fe::kFlagImages) ? p.size() != base : p.size() < base + 4) return false;
      r.stop.resize(r.h.n_stop_ids);
      r.prompt.resize(r.h.n_prompt);
      const char* a = p.data() + sizeof r.h;
      if (!r.stop.empty()) std::memcpy(r.stop.data(), a, r.stop.size() * 4);
      if (!r.prompt.empty()) std::memcpy(r.prompt.data(), a + r.stop.size() * 4, r.prompt.size() * 4);
      if (r.h.flags & fe::kFlagImages) {  // vision: n, the spans, their rows
        uint32_t ni = 0;
        std::memcpy(&ni, p.data() + base, 4);
        if (ni < 1 || ni > fe::kMaxImages || p.size() < base + 4 + ni * sizeof(fe::ImageSpan)) return false;
        r.spans.resize(ni);
        std::memcpy(r.spans.data(), p.data() + base + 4, ni * sizeof(fe::ImageSpan));
        size_t rows = 0;
        for (const auto& s : r.spans) {
          if (s.rows < 3 || s.rows > fe::kMaxImageRows) return false;
          rows += s.rows;
        }
        const size_t off = base + 4 + ni * sizeof(fe::ImageSpan);
        if (p.size() != off + rows * fe::kImageDim * 2) return false;
        r.img.resize(rows * fe::kImageDim);
        std::memcpy(r.img.data(), p.data() + off, r.img.size() * 2);
      }
      r.conn = c;
      r.t_recv = now_s();
      {
        std::lock_guard<std::mutex> l(c->mu);
        c->reqs.submit(r.h.req_id);  // in flight from here: only a CANCEL for an id in this set counts
      }
      push(std::move(e));
    } else if (h.type == fe::kCancel) {
      fe::Cancel x;
      if (p.size() < sizeof x) return false;
      std::memcpy(&x, p.data(), sizeof x);
      std::lock_guard<std::mutex> l(c->mu);
      c->reqs.cancel(x.req_id);  // an id that is not in flight (ended, or never sent) is not marked: nothing to un-cancel later
    } else if (h.type == fe::kRelease) {
      fe::Release x;
      if (p.size() < sizeof x) return false;
      std::memcpy(&x, p.data(), sizeof x);
      Event e;
      e.release = true;
      e.session = x.session_id;
      push(std::move(e));
    } else if (h.type == fe::kConfig) {  // the main loop applies it (take_config)
      fe::Config x{};
      std::memcpy(&x, p.data(), std::min(p.size(), sizeof x));
      const size_t got = p.size() < 8 ? 0 : std::min<size_t>(x.n_fields, (p.size() - 8) / 8);
      int64_t* f = &x.on;
      for (size_t i = got; i < fe::kConfigFields; ++i) f[i] = -1;  // not sent: kept
      std::lock_guard<std::mutex> l(cmu_);
      cfg_ = x, cfg_new_ = true;
    } else if (h.type == fe::kStatsReq) {
      std::vector<char> f;
      {
        std::lock_guard<std::mutex> l(smu_);
        f = stats_;
      }
      if (!f.empty()) send(c, std::move(f));
    }  // other types are skipped
    return true;
  }

  void reader() {
    ConnP cur;
    while (!stop_) {
      pollfd fds[2] = {{lfd_, POLLIN, 0}, {cur ? cur->fd : -1, POLLIN, 0}};
      if (poll(fds, 2, 200) <= 0) continue;
      if (fds[0].revents & POLLIN) {
        const int fd = accept4(lfd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd >= 0) {
          if (cur) cur->kill();  // the new front-end replaces the old one
          cur = std::make_shared<Conn>(fd);
          send(cur, hello_);
          std::printf("serve: front-end connected\n");
        }
        continue;
      }
      if (cur && fds[1].revents && !frame(cur)) {
        cur->kill();
        cur.reset();
        std::printf("serve: front-end disconnected\n");
      }
    }
    if (cur) cur->kill();
  }

  void writer() {
    for (;;) {
      std::pair<ConnP, std::vector<char>> f;
      {
        std::unique_lock<std::mutex> l(wmu_);
        wcv_.wait(l, [&] { return stop_ || !out_.empty(); });
        if (stop_) return;
        f = std::move(out_.front());
        out_.pop_front();
      }
      const size_t n = f.second.size();
      if (!f.first->dead && !write_all(f.first->fd, f.second.data(), n)) f.first->kill();
      f.first->outq.done(n);
    }
  }

  std::string path_;
  std::vector<char> hello_;
  int lfd_ = -1;
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Event> ev_;
  std::mutex wmu_;
  std::mutex smu_;
  std::vector<char> stats_;  // the latest STATS frame (publish_stats)
  std::mutex cmu_;
  fe::Config cfg_{};  // the latest CONFIG frame (take_config)
  bool cfg_new_ = false;
  std::condition_variable wcv_;
  std::deque<std::pair<ConnP, std::vector<char>>> out_;
  std::thread rd_, wr_;
};

// The YaRN factor a session's state is rotated with: 16, 32 or 64 (windows of 1,048,576, 2,097,152 and 4,194,304
// positions). A session keeps its factor; a request runs at the smallest factor whose window holds its prompt and
// matches only state of that factor (Server::find_at); a conversation that outgrows its window goes on in a new session
// (Server::acquire, Server::cross).
constexpr int kBaseFactor = 16;
constexpr int window_of(int factor) { return 65536 * factor; }
constexpr int factor_for(int64_t n) { return n < window_of(16) ? 16 : n < window_of(32) ? 32 : 64; }

struct Session {
  int h = 0;                   // the store's handle, the same on every rank
  int factor = kBaseFactor;    // its rotation factor (the store's Sess::factor says the same: state_factor)
  uint64_t owner = 0;          // 0: the front-end's session id is this entry's key; else the id this one answers to at
                               // `factor` while another entry holds the id (Server::alt_)
  int slot = -1;               // the slot whose working state is this session's (-1: none, the store's only)
  std::vector<int32_t> hist;   // the tokens whose state the session holds: positions [0, size)
  uint64_t used = 0;           // LRU clock
  int pe_snap = -1;            // the snapshot at its last prompt's end (when not a multiple of snap_every)
  int lo = 0, clo = 0, w = 0;  // the live state's exact ranges (the comment at the top)
  int end_lo = 0, end_clo = 0, end_w = 0;  // the end snapshot's
  // CED: the decoder layers' rings hold valid positions from dec_lo on (kNever: none), the end
  // snapshot's from end_dec_lo on; window snapshots holding the encoder layers' rings only
  int dec_lo = 0, end_dec_lo = 0;
  std::vector<int> enc_snaps;
  double t_used = 0;  // unix time of its last request (kept sessions age out by it)
  uint64_t flushed = 0;  // the LRU clock at its last flush to the disk tier (flush_idle)
};

// The server's index of a planned stop's kept sessions (keep(); keepfile.h: 0600 in a 0700 directory,
// whole or not at all): a header, then each session: id, handle, restart ranges, LRU clock and time,
// history, encoder-only snapshots.
constexpr char kSrvMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'S', '2'};  // 2: each session's factor and owner
struct SrvHeader {
  char magic[8];
  uint64_t gen;  // the store's (state.cu): both indexes come from the same stop
  int64_t written;
  int32_t n_sess, next_h;
  uint64_t clock;
  uint64_t body_bytes, body_hash;
};

constexpr int kReplay = 128;          // CED's decoder replay
constexpr int kNever = 1 << 30;       // Session::dec_lo of rings that are valid nowhere

// The window snapshots a session keeps, by their distance d from its history's end E: every one (each
// `every` positions) while d < 8 * every; further back those at multiples of every * 2^k, k = 1, 2, ...
// as d passes 8, 16, 32, ... times every. That is four per doubling of d: a restart d positions before
// the end re-prefills at most d / 3 positions from the snapshot below it (at most a third of what the
// client dropped), and a session holds O(log E) snapshots (28 at 250k positions, not 245). As E grows
// the multiples coarsen, so a snapshot dropped once is never wanted again.
bool keep_snap(int s, int E, int every) {
  const int64_t d = (int64_t)E - s, dense = 8ll * every;
  if (d < dense) return true;
  int64_t g = 2ll * every;
  for (int64_t lim = 2 * dense; d >= lim; lim *= 2) g *= 2;
  return s % g == 0;
}

enum { kCold, kLive, kEnd, kSnap };
const char* const kHow[] = {"cold", "live", "end", "snapshot"};
struct Plan {
  int q = 0, how = kCold, snap = -1;
};

int lcp(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return (int)i;
}

// A run's time split for the log: host ms from its start to the prompt's tokens and state operations
// issued (and their GPU time), the chunks' GPU time (Model::prefill_split; rows: steps of run_rows), host
// ms from the first token to the decode command, and from it to the first sample.
struct Turn {
  double ops_ms = 0, ops_gpu_ms = 0, rows_ms = 0, tail_ms = 0, dec1_ms = -1;
  float pf[4] = {};
  int chunks = 0;
  bool pending = false;  // the last chunk's split, read once the first token is out
  void add_split(const model::Model& m) {
    float f[4];
    m.prefill_split(f);
    for (int i = 0; i < 4; ++i) pf[i] += f[i];
  }
};

// A request from its start to its DONE. Between its runs (decode with the others, or alone) it waits in
// the turns; its next token to feed is out.back(), at position prompt + out - 1 (fed again if its KV is
// there already: the same values). A request whose session lost its slot resumes from the store: the
// prompt and the tokens sent so far, one row.
struct Flight {
  Request rq;
  model::Sampling sp;        // the seed fixed at the start: each run samples what one run would
  std::vector<int32_t> out;  // the tokens sent
  int q = 0;                 // the first run's restart point (DONE's cached tokens), its plan and split
  Plan pl;
  Turn tu;
  double t_start = 0, t_first = 0, t_paused = 0;
  double paused_s = 0, resume_s = 0, dec_s = 0;  // waiting in turn, resuming (to each first token), decoding
  int slices = 0, decoded = 0, batched = 0;  // runs; tokens decoded, of them in batches
  int upto = 0;                              // positions its slot's reservation holds (growth: a horizon at a time)
  DecodeEnd de;                      // the slices' drafts, summed
  double spec_tok = 1, spec_ms = 0;  // the drafts' rate (SpecRate) from slice to slice
  bool done_sent = false;            // its DONE went out when it ended inside a batch (finish() sends no other)
  int fac = kBaseFactor;             // the factor its next run needs: its prompt's, then what its response has grown to
  bool cross = false;                // its decode reached the window of fac: cross() before its next run
};

// b's drafts and steps added to a's (a request's decode slices)
void add(DecodeEnd& a, const DecodeEnd& b) {
  a.vsteps += b.vsteps, a.verified += b.verified, a.accepted += b.accepted;
  for (int r = 0; r <= model::ops::kDrafts; ++r) a.rows_hist[r] += b.rows_hist[r];
  for (int r = 0; r <= model::kMaxRows; ++r) a.gpu_ms[r] += b.gpu_ms[r], a.gpu_n[r] += b.gpu_n[r];
  if (b.gate_n) {
    a.gate_min = a.gate_n ? std::min(a.gate_min, b.gate_min) : b.gate_min;
    a.gate_max = a.gate_n ? std::max(a.gate_max, b.gate_max) : b.gate_max;
    a.gate_sum += b.gate_sum, a.gate_n += b.gate_n;
  }
}

class Server {
 public:
  Server(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o, Io& io, size_t budget)
      : m_(m), ring_(ring), wd_(wd), o_(o), io_(io), budget_(budget), slot_(m.slots(), 0), il_(o.interleave), keep_age_s_(o.keep_age_s), extend_(o.extend) {}

  // Requests let go after their session's last one ended first, then those that waited for a slot (when
  // one is there), then the front-end's frames, then a run of the requests decoding (run).
  void loop() {
    while (!g_stop) {
      fe::Config cf;
      if (io_.take_config(cf)) apply_config(cf);
      publish();
      sweep();
      if (!pfq_.empty()) {  // interleaved prefill: long prompts and decoders take turns
        il_pass();
        continue;
      }
      Event e;
      if (!ready_.empty()) {
        Request rq = std::move(ready_.front());
        ready_.pop_front();
        admit(std::move(rq));
      } else if (!wait_.empty() && window_ready(wait_.front()) &&
                 pick_slot({}, need_clamped(wait_.front()), wait_.front().h.session_id, bkr(wait_.front())) >= 0) {
        Request rq = std::move(wait_.front());
        wait_.pop_front();
        admit(std::move(rq));
      } else if (io_.pop(e, rot_.empty() ? 250 : 0)) {
        if (e.release) {
          if (flying_.count(e.session)) later_.insert(e.session);
          else release(e.session);
        } else if (!e.rq.conn->dead) {  // a gone front-end's requests need no answer
          admit(std::move(e.rq));
        }
      } else if (!rot_.empty()) {
        run();
      } else {
        resize_idle();
        flush_idle();
        checkpoint();
        windows_idle();
        idle_check();
      }
    }
    // the stop ends what is paused or held with an error (Io::stopping): "the server is stopping"
    sweep();
    while (!pfq_.empty()) {  // a long prompt being prefilled stops after its last piece; the rest never started
      Long& L = *pfq_.front();
      if (L.begun) L.r.cancelled = true, pf_end(L.r);
      else end_queued(L, fe::kCancelled);
      pfq_.pop_front();
    }
    push_ = nullptr;
    for (std::deque<Request>* d : {&ready_, &held_, &wait_}) {
      for (const Request& rq : *d) done(rq, fe::kCancelled, 0, 0, now_s(), 0), Io::forget(rq);
      d->clear();
    }
    keep();
    bcast(ring_, Cmd{});  // stop the other ranks
  }

  // Nothing in flight: no line may be reserved numbers it does not hold (finish() zeroes each reservation). A reservation left
  // over means a path ended a flight without zeroing it; the line says so once per value.
  void idle_check() {
    if (!flying_.empty() || !rot_.empty() || !pfq_.empty()) return;
    std::vector<reclaim::SlotNums> v;
    for (int t = 0; t < (int)slot_.size(); ++t) v.push_back({m_.slot_committed(t), m_.slot_mapped(t)});
    const int64_t n = reclaim::idle_reserves(v);
    if (n != 0 && n != idle_reserved_) std::printf("serve: BUG: idle line holds a reservation (%lld numbers reserved, nothing in flight)\n", (long long)n);
    idle_reserved_ = n;
  }
  int64_t idle_reserved_ = 0;

  // The sessions kept at a planned stop or checkpoint, when the store reused those files (Model::state_kept):
  // histories and ranges from o_.keep_index, which stays as the latest checkpoint's and goes when it does not check out.
  // Sessions the store does not hold on disk, or unused for keep_age_s, are dropped, and so are the store's sessions the
  // index does not name.
  void load_kept() {
    if (o_.keep_index.empty()) return;
    const double t0 = now_s();
    std::vector<uint8_t> f;
    std::string why;
    const bool got = keepfile::read_private(o_.keep_index, 16ull << 30, &f, &why);
    unlink((o_.keep_index + ".new").c_str());
    const uint64_t gen = m_.state_kept();
    std::map<uint64_t, Session> kept;
    SrvHeader hd{};
    if (got && f.size() >= sizeof hd) std::memcpy(&hd, f.data(), sizeof hd);
    if (!got) {
      // why: "" when there is none
    } else if (!gen) {
      why = "the store did not reuse its files";
    } else if (f.size() < sizeof hd || std::memcmp(hd.magic, kSrvMagic, sizeof kSrvMagic) != 0 || hd.gen != gen ||
               hd.body_bytes != f.size() - sizeof hd ||
               keepfile::fnv(f.data() + sizeof hd, hd.body_bytes) != hd.body_hash || hd.n_sess < 0) {
      why = "the index is not of the stop whose files the store reused";
    } else {
      keepfile::Reader r{f.data() + sizeof hd, (size_t)hd.body_bytes};
      const double wall = unix_now();
      for (int i = 0; i < hd.n_sess && r.ok; ++i) {
        const uint64_t id = r.get<uint64_t>();
        Session s;
        s.h = r.get<int32_t>(), s.pe_snap = r.get<int32_t>();
        s.end_lo = r.get<int32_t>(), s.end_clo = r.get<int32_t>(), s.end_w = r.get<int32_t>(), s.end_dec_lo = r.get<int32_t>();
        s.used = r.get<uint64_t>();
        s.t_used = r.get<double>();
        s.factor = r.get<int32_t>(), s.owner = r.get<uint64_t>();
        const int nh = r.get<int32_t>(), ne = r.get<int32_t>();
        if (!r.ok || nh < 0 || nh > o_.max_ctx || ne < 0 || ne > 4096) {
          r.ok = false;
          break;
        }
        s.hist.resize(nh);
        for (int32_t& t : s.hist) t = r.get<int32_t>();
        s.enc_snaps.resize(ne);
        for (int& e : s.enc_snaps) e = r.get<int32_t>();
        bool ok = r.ok && id && !kept.count(id) && s.h > 0 && wall - s.t_used <= keep_age_s_;
        // the factor is a window the engine can hold, the same one the ranks' store recorded, and an entry that answers
        // to another id is not that id's own
        ok = ok && (s.factor == 16 || s.factor == 32 || s.factor == 64) && (s.factor == 16 || window_of(s.factor) / 2 < o_.max_ctx) && s.owner != id;
        for (const int32_t t : s.hist) ok = ok && t >= 0 && t < model::kVocab;
        const model::Model::StateInfo si = m_.state_info(s.h);
        ok = ok && si.on_disk && si.saved <= nh && si.end_pos <= nh && m_.state_factor(s.h) == s.factor;
        for (const auto& [k, x] : kept) ok = ok && x.h != s.h;
        s.lo = s.clo = s.w = s.dec_lo = 0;  // nothing is live
        if (ok) kept.emplace(id, std::move(s));
      }
      if (!r.ok || r.at != r.n) {
        kept.clear();
        why = "the index is damaged";
      }
    }
    if (!why.empty() || !gen) unlink(o_.keep_index.c_str());  // none to come back to
    // the store's sessions that no kept session names go
    std::vector<StateOp> v;
    std::set<int> named;
    for (const auto& [id, s] : kept) named.insert(s.h);
    int top = 0;
    for (const int h : m_.state_sessions()) {
      top = std::max(top, h);
      if (!named.count(h)) v.push_back({kStFree, h, 0, 0});
    }
    if (!v.empty()) state(v);
    next_h_ = std::max({next_h_, top + 1, kept.empty() ? 0 : hd.next_h});
    for (auto& [id, s] : kept) clock_ = std::max(clock_, s.used);
    size_t positions = 0;
    for (const auto& [id, s] : kept) positions += s.hist.size();
    if (!kept.empty()) clock_ = std::max<uint64_t>(clock_, hd.clock);
    sess_.insert(std::make_move_iterator(kept.begin()), std::make_move_iterator(kept.end()));
    for (const auto& [id, x] : sess_)
      if (x.owner) alt_[{x.owner, x.factor}] = id;
    if (!why.empty() || !sess_.empty() || !v.empty())
      std::printf("serve: kept sessions: %zu back (%zu positions, %.2f GiB on disk), %zu of the store's dropped, in %.1f ms%s%s\n",
                  sess_.size(), positions, m_.state_disk_bytes() / 1073741824.0, v.size(), 1e3 * (now_s() - t0),
                  why.empty() ? "" : "; ", why.c_str());
  }

 private:
  // Every rank: c's state operations, then c (rank 0's part of c.op is the caller's). c carries the
  // sessions of the slots that a running request uses, which the store keeps (pinned) and whose images it keeps in the
  // disk tier's file; a conversation on an idle or paused line is cache like any other.
  void issue(Cmd& c) {
    for (size_t s = 0; s < slot_.size(); ++s) {
      const auto it = slot_[s] ? sess_.find(slot_[s]) : sess_.end();
      c.live[s] = it == sess_.end() || !reclaim::counts_as_live(flying_.count(slot_[s]) > 0, paused(slot_[s])) ? 0 : it->second.h;
    }
    bcast(ring_, c);
    apply_ops(m_, c);
  }
  // State operations on every rank, in commands of kMaxOps.
  void state(const std::vector<StateOp>& v) {
    for (size_t i = 0; i < v.size(); i += kMaxOps) {
      Cmd c{};
      c.op = kCmdState;
      c.n_ops = (int)std::min<size_t>(kMaxOps, v.size() - i);
      std::copy(v.begin() + i, v.begin() + i + c.n_ops, c.ops);
      issue(c);
    }
    digest("state");
  }
  // DSGB10_STORE_CHECK=1: the store's digest after a state, window or reservation command, one line each. Every rank prints
  // its own (cmd.cpp), and the four logs must show the same value on the same line: a rank-local input to a replicated
  // decision shows up as the first line that differs.
  void digest(const char* what) {
    static const bool on = std::getenv("DSGB10_STORE_CHECK") && std::atoi(std::getenv("DSGB10_STORE_CHECK")) == 1;
    if (on) std::printf("serve: store digest %llu %s: %016llx\n", (unsigned long long)++digest_n_, what, (unsigned long long)m_.store_digest());
  }
  uint64_t digest_n_ = 0;
  // A planned stop (SIGTERM): the sessions kept for the next start (load_kept; docs/engine.md). The slots'
  // sessions' states go to the store; sessions unused for keep_age_s are freed; those in device memory
  // are spilled to the disk tier, the most recently used first, as far as keep_bytes goes (the rest
  // freed); then every rank syncs its file and writes its index (kStKeep), and this server the kept
  // sessions' histories with the same generation. Spilling may drop the file's least recently used
  // sessions to make room (forget_evicted).
  void keep() {
    if (!o_.keep_bytes || o_.keep_index.empty()) return;
    const double t0 = now_s(), wall = unix_now();
    for (int s = 0; s < (int)slot_.size(); ++s) persist_slot(s, true);
    std::vector<StateOp> v;
    std::vector<std::pair<uint64_t, uint64_t>> ram;  // (used, id): the sessions in device memory
    int old = 0, before = 0;
    for (auto it = sess_.begin(); it != sess_.end();) {
      if (wall - it->second.t_used > keep_age_s_) {
        v.push_back({kStFree, it->second.h, 0, 0});
        it = sess_.erase(it);
        ++old;
        continue;
      }
      if (m_.state_info(it->second.h).on_disk) ++before;
      else ram.emplace_back(it->second.used, it->first);
      ++it;
    }
    std::sort(ram.rbegin(), ram.rend());
    size_t bytes = 0;
    int spilled = 0, freed = 0;
    for (const auto& [used, id] : ram) {
      const Session& s = sess_.at(id);
      const size_t b = m_.state_info(s.h).bytes;
      if (bytes + b <= o_.keep_bytes) {
        v.push_back({kStSpill, s.h, 0, 0});
        bytes += b, ++spilled;
      } else {
        v.push_back({kStFree, s.h, 0, 0});
        sess_.erase(id);
        ++freed;
      }
    }
    state(v);
    forget_evicted();  // dropped from the file to make room
    uint64_t gen = rng_();
    gen += !gen;
    state({{kStKeep, -1, (int32_t)(uint32_t)gen, (int32_t)(uint32_t)(gen >> 32)}});
    const double t1 = now_s();
    std::set<int> hs;
    for (const auto& [id, s] : sess_)
      if (m_.state_info(s.h).on_disk) hs.insert(s.h);
    int n = 0;
    size_t fb = 0;
    std::string why;
    const bool ok = write_index(gen, hs, &n, &fb, &why);
    heap::release();
    std::printf("serve: kept sessions: %d for the next start (%d on disk before, %d spilled now: %.2f GiB; %d freed past "
                "%.1f GiB, %d unused for %.0f days), the ranks' files synced and indexed in %.2f s; histories %.1f KiB "
                "in %.1f ms%s%s\n",
                ok ? n : 0, before, spilled, bytes / 1073741824.0, freed, o_.keep_bytes / 1073741824.0, old,
                keep_age_s_ / 86400, t1 - t0, fb / 1024.0, 1e3 * (now_s() - t1), ok ? "" : ": NOT WRITTEN, ",
                why.c_str());
  }

  // The server's index of gen (keepfile.h: whole or not at all), beside the ranks' of the same gen: the sessions whose
  // handles hs names, each with its id, handle, restart ranges, LRU clock and time, history, encoder-only snapshots.
  bool write_index(uint64_t gen, const std::set<int>& hs, int* n, size_t* bytes, std::string* why) {
    std::vector<uint8_t> f(sizeof(SrvHeader));
    using keepfile::put;
    *n = 0;
    for (const auto& [id, s] : sess_) {
      if (!hs.count(s.h)) continue;
      ++*n;
      put<uint64_t>(f, id), put<int32_t>(f, s.h), put<int32_t>(f, s.pe_snap);
      put<int32_t>(f, s.end_lo), put<int32_t>(f, s.end_clo), put<int32_t>(f, s.end_w), put<int32_t>(f, s.end_dec_lo);
      put<uint64_t>(f, s.used), put<double>(f, s.t_used);
      put<int32_t>(f, s.factor), put<uint64_t>(f, s.owner);
      put<int32_t>(f, (int32_t)s.hist.size()), put<int32_t>(f, (int32_t)s.enc_snaps.size());
      for (const int32_t t : s.hist) put<int32_t>(f, t);
      for (const int e : s.enc_snaps) put<int32_t>(f, e);
    }
    SrvHeader hd{};
    std::memcpy(hd.magic, kSrvMagic, sizeof kSrvMagic);
    hd.gen = gen, hd.written = (int64_t)unix_now(), hd.n_sess = *n, hd.next_h = next_h_, hd.clock = clock_;
    hd.body_bytes = f.size() - sizeof hd;
    hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
    std::memcpy(f.data(), &hd, sizeof hd);
    *bytes = f.size();
    return keepfile::write_atomic(o_.keep_index, f.data(), f.size(), why);
  }

  // While nothing runs, a checkpoint a crash comes back to (docs/engine.md "Checkpoints while serving"): when the
  // store's images changed and --ssd-flush-s has passed since the last, the lines' conversations are saved as a planned
  // stop saves them, every rank writes its index (kStCheckpoint), then this server its sessions index of the same gen.
  // A crash between the two leaves gens that differ: the next start is clean or uses the previous pair.
  // DSGB10_CKPT_STOP=1 ends the process between them.
  void checkpoint() {
    if (!o_.keep_bytes || o_.keep_index.empty() || !m_.state_disk_capacity()) return;
    const double t0 = now_s();
    if (t0 - ckpt_at_ < std::max(1, o_.ssd_flush_s) || m_.state_version() == ckpt_ver_) return;
    // a line's conversation as a planned stop keeps it, the line kept: its caches to its history's end and its end
    // bookmark (persist_ops), then every piece of it written (kStFlush), so a crash restarts it where the line stood
    std::vector<StateOp> fl;
    for (int s = 0; s < (int)slot_.size(); ++s)
      if (slot_[s] && !flying_.count(slot_[s]))
        if (Session* x = find(slot_[s])) {
          persist_slot(s, false);
          fl.push_back({kStFlush, x->h, 0, 0});
          x->flushed = x->used;
        }
    if (!fl.empty()) state(fl);
    uint64_t gen = rng_();
    gen += !gen;
    state({{kStCheckpoint, -1, (int32_t)(uint32_t)gen, (int32_t)(uint32_t)(gen >> 32)}});
    ckpt_at_ = now_s(), ckpt_ver_ = m_.state_version();
    const std::vector<int> v = m_.state_checkpointed();
    if (const char* e = std::getenv("DSGB10_CKPT_STOP"); e && *e == '1' && !v.empty()) {
      std::printf("serve: DSGB10_CKPT_STOP: stopping between the ranks' indexes (%zu conversations) and the server's\n", v.size());
      std::fflush(stdout);
      _exit(9);
    }
    const std::set<int> hs(v.begin(), v.end());
    int n = 0;
    size_t fb = 0;
    std::string why;
    const bool ok = write_index(gen, hs, &n, &fb, &why);
    std::printf("serve: checkpoint: %d conversations a crash comes back to; the ranks' indexes in %.1f ms, histories %.1f KiB "
                "in %.1f ms%s%s\n", ok ? n : 0, 1e3 * (ckpt_at_ - t0), fb / 1024.0, 1e3 * (now_s() - ckpt_at_),
                ok ? "" : ": NOT WRITTEN, ", why.c_str());
  }

  // The disk tier's size the dashboard asked for (CONFIG ssd_gib), when no request is in flight or paused between turns:
  // a smaller size drops the conversations that need a piece past it (state_disk_resize; forget_evicted), and none of
  // them may be one a flight still uses.
  void resize_idle() {
    if (!ssd_pending_ || !flying_.empty() || !rot_.empty() || !pfq_.empty()) return;
    const int64_t g = ssd_pending_;
    ssd_pending_ = 0;
    if ((size_t)g << 30 == m_.state_disk_capacity()) return;
    state({{kStDiskSize, -1, (int32_t)(g << 14), 0}});  // in 64 KiB grains, every rank alike
    forget_evicted();
    published_ = 0;
  }

  // While nothing runs: conversations idle for --ssd-flush-s get their pieces without an image written (once per use),
  // so an eviction or a planned stop writes nothing for them; conversations unused for --keep-days are dropped.
  // One command for all of them.
  void flush_idle() {
    const double now = unix_now();
    std::vector<StateOp> v;
    int expired = 0;
    for (auto it = sess_.begin(); it != sess_.end() && v.size() < 64;) {
      Session& S = it->second;
      if (flying_.count(it->first)) {
        ++it;
        continue;
      }
      if (S.slot < 0 && keep_age_s_ > 0 && S.t_used > 0 && now - S.t_used > keep_age_s_) {  // a line's is kept
        v.push_back({kStFree, S.h, 0, 0});
        drop_ids(S);
        it = sess_.erase(it);
        ++expired;
        continue;
      }
      if (o_.ssd_flush_s > 0 && S.flushed != S.used && now - S.t_used >= o_.ssd_flush_s) {
        v.push_back({kStFlush, S.h, 0, 0});
        S.flushed = S.used;
      }
      ++it;
    }
    if (v.empty()) return;
    state(v);
    if (expired) std::printf("serve: %d sessions unused for %.0f days dropped\n", expired, keep_age_s_ / 86400);
  }
  // Operations that move session x's working state (its slot's) into the store: its compressed caches as
  // far as its history goes, and its end snapshot. They act on its slot (the command's).
  void persist_ops(Session& x, std::vector<StateOp>& v) {
    const int E = (int)x.hist.size();
    const model::Model::StateInfo info = m_.state_info(x.h);
    if (info.saved < E) v.push_back({kStSave, x.h, info.saved, E});
    if (info.end_pos != E) {
      v.push_back({kStSnapEnd, x.h, E, 0});
      x.end_lo = x.lo, x.end_clo = x.clo, x.end_w = x.w, x.end_dec_lo = x.dec_lo;
    }
  }
  // Slot s's session's state into the store, in a command of its own acting on s; `free`: the slot is let
  // go (its session then lives in the store only).
  void persist_slot(int s, bool free, const std::vector<int>& hold = {}) {
    Session* x = slot_[s] ? find(slot_[s]) : nullptr;
    if (free) {
      slot_[s] = 0;
      if (x) x->slot = -1;
    }
    std::vector<StateOp> v;
    if (x) persist_ops(*x, v);
    if (free) v.push_back({kStRelease, 0, s, 0});  // the slot's memory back to the pool, once its state is saved
    if (v.empty()) return;
    for (const int h : hold) v.push_back({kStPin, h, 0, 0});  // the store leaves these sessions alone while it makes room for this one's
    Cmd c{};
    c.op = kCmdState;
    c.slot = s;
    c.n_ops = (int)v.size();
    std::copy(v.begin(), v.end(), c.ops);
    issue(c);
  }
  // The conversation `id` let go: its session at every factor (the primary and the alternates)
  void release(uint64_t id) {
    std::vector<uint64_t> keys{id};
    for (const int fa : {16, 32, 64}) {
      const auto a = alt_.find({id, fa});
      if (a == alt_.end()) continue;
      keys.push_back(a->second);
      alt_.erase(a);
    }
    for (const uint64_t k : keys) {
      const auto it = sess_.find(k);
      if (it == sess_.end() || (k != id && it->second.owner != id)) continue;
      std::vector<StateOp> v{{kStFree, it->second.h, 0, 0}};
      if (it->second.slot >= 0) v.push_back({kStRelease, 0, it->second.slot, 0}), slot_[it->second.slot] = 0;
      state(v);
      sess_.erase(it);
    }
  }
  // Session s's window snapshots that keep_snap no longer keeps at its history's end, dropped (not the
  // one at its last prompt's end).
  void thin(Session& s) {
    if (!o_.snap_thin) return;
    const int E = (int)s.hist.size();
    auto gone = [&](int sn) { return sn != s.pe_snap && !keep_snap(sn, E, o_.snap_every); };
    std::vector<StateOp> v;
    for (const int sn : m_.state_info(s.h).snaps)
      if (gone(sn)) v.push_back({kStDropSnap, s.h, sn, 0});
    s.enc_snaps.erase(std::remove_if(s.enc_snaps.begin(), s.enc_snaps.end(), gone), s.enc_snaps.end());
    if (!v.empty()) state(v);
  }
  // Sessions the store evicted to make room (an operation found its pool full) are gone here too.
  void forget_evicted() {
    std::vector<StateOp> v;
    for (const int h : m_.state_evicted())
      for (auto it = sess_.begin(); it != sess_.end(); ++it)
        if (it->second.h == h) {
          std::printf("serve: session %016llx evicted to make room\n", (unsigned long long)it->first);
          if (it->second.slot >= 0) v.push_back({kStRelease, 0, it->second.slot, 0}), slot_[it->second.slot] = 0;
          sess_.erase(it);
          break;
        }
    if (!v.empty()) state(v);  // a live session's slot memory goes back with it
  }
  // Least recently used sessions out until the store fits the budget: spilled to the disk tier (one live
  // in a slot saved first, and the slot let go), or freed without one. Sessions in flight stay.
  void evict(uint64_t keep) {
    const bool disk = m_.state_disk_capacity() > 0;
    std::vector<StateOp> v;
    std::vector<uint64_t> out;
    size_t bytes = m_.state_bytes();
    while (bytes > budget_) {
      auto victim = sess_.end();
      for (auto it = sess_.begin(); it != sess_.end(); ++it)
        if (it->first != keep && !flying_.count(it->first) && std::find(out.begin(), out.end(), it->first) == out.end() &&
            (victim == sess_.end() || it->second.used < victim->second.used) &&
            !(disk && m_.state_info(it->second.h).on_disk))
          victim = it;
      if (victim == sess_.end()) break;
      const size_t b = m_.state_info(victim->second.h).bytes;
      std::printf("serve: %s session %016llx (%.1f MiB, %zu positions)\n", disk ? "spill" : "evict",
                  (unsigned long long)victim->first, b / 1048576.0, victim->second.hist.size());
      bytes -= std::min(bytes, b);
      if (disk) {
        if (victim->second.slot >= 0) persist_slot(victim->second.slot, true);  // what it holds live goes with it
        v.push_back({kStSpill, victim->second.h, 0, 0});
        out.push_back(victim->first);
        continue;
      }
      v.push_back({kStFree, victim->second.h, 0, 0});
      if (victim->second.slot >= 0) v.push_back({kStRelease, 0, victim->second.slot, 0}), slot_[victim->second.slot] = 0;
      sess_.erase(victim);
    }
    if (!v.empty()) state(v);
    forget_evicted();  // dropped from the disk tier to make room
  }
  // A request's session (or a fork's base) that was spilled comes back to the store before its plan
  // sees it. The slots' sessions stay where they are (every command pins them).
  void unspill(const fe::SubmitHeader& h, int fac) {
    uint64_t id = h.session_id;
    Session* x = find_at(id, fac, &id);
    if (!x && h.base_session_id) id = h.base_session_id, x = find_at(id, fac, &id);
    if (!x) return;
    const uint64_t base = bk(h, fac);
    const model::Model::StateInfo si = m_.state_info(x->h);
    if (!si.on_disk) return;
    const double t0 = now_s();
    // room for what comes back: idle lines let their conversations go first (they stay in the pool, now evictable), as
    // make_room does for a request's own need, so that the read-back does not find the pool held by idle lines. What
    // running requests hold is left: the read-back takes what fits (and never what a running request was reserved)
    give_back((int64_t)m_.state_restore_chunks(x->h), {}, h.session_id, base, -1, 0, "a conversation read back", false);
    state({{kStUnspill, x->h, 0, 0}});
    m_.drain();
    const model::Model::StateInfo back = m_.state_info(x->h);  // what came back: the room decides how much
    std::printf("serve: session %016llx back from disk: %.1f MiB of %.1f MiB, %zu positions, restart point %d of %d, in %.1f ms\n",
                (unsigned long long)id, back.bytes / 1048576.0, si.disk_bytes / 1048576.0, x->hist.size(), back.restart,
                si.restart, 1e3 * (now_s() - t0));
    forget_evicted();
  }

  // The earliest position a restart of s could start from (its snapshots, its end), for the log; -1:
  // none
  int first_restart(const Session& s) const {
    const model::Model::StateInfo info = m_.state_info(s.h);
    int r = info.end_pos;
    for (const int sn : info.snaps) {
      const int q = sn & 1 ? sn - 1 : sn;
      if (r < 0 || q < r) r = q;
    }
    return r;
  }

  // The restart point for continuing session s (its state live or stored) after a common prefix of p,
  // for a prompt of n positions.
  Plan choose(const Session& s, bool live, int p, int n) const {
    Plan best;
    const model::Model::StateInfo info = m_.state_info(s.h);
    p = live ? p : std::min(p, info.saved);  // a stored session's compressed caches go as far as its pages
    auto consider = [&](int q, int how, int snap) {
      if (q > best.q) best = Plan{q, how, snap};
    };
    // CED: a restart at q leaving fewer than 128 positions needs the decoder's rings valid at q
    auto usable = [&](int q, bool dec_ok) { return !o_.ced || n - q >= kReplay || dec_ok; };
    // with the ring and the carries as they were at position `end` (exact for [max(lo, w - 256), end)
    // and [max(clo, w - 16), end)), at most at `cap`
    auto from_ring = [](int end, int cap, int lo, int clo, int w) {
      int q = std::min(cap, end);
      if ((q & 1) && q - 1 < std::max(clo, w - 16)) --q;
      return q >= 0 && std::max(0, q - 128) >= std::max(lo, w - 256) ? q : -1;
    };
    // CED: else the latest restart that replays the decoder
    auto ring = [&](int end, int lo, int clo, int w, int dec_lo) {
      const int q = from_ring(end, p, lo, clo, w);
      return q < 0 || usable(q, std::max(0, q - 127) >= dec_lo) ? q
                                                                 : from_ring(end, std::min(p, n - kReplay), lo, clo, w);
    };
    if (live) consider(ring((int)s.hist.size(), s.lo, s.clo, s.w, s.dec_lo), kLive, -1);
    if (info.end_pos >= 0) consider(ring(info.end_pos, s.end_lo, s.end_clo, s.end_w, s.end_dec_lo), kEnd, -1);
    for (const int sn : info.snaps) {
      const int q = sn & 1 ? sn - 1 : sn;
      if (q <= p && usable(q, std::find(s.enc_snaps.begin(), s.enc_snaps.end(), sn) == s.enc_snaps.end()))
        consider(q, kSnap, sn);
    }
    return best;
  }

  void tokens(const Request& rq, const model::SampleOut& o, int row) {
    const bool lp = rq.h.flags & fe::kFlagLogprobs;
    const int nt = lp ? std::min<int>({(int)rq.h.n_top_logprobs, model::ops::kTopLogprobs, o.n_top}) : 0;
    const fe::TokensHeader t{rq.h.req_id, 1, (lp ? fe::kTokLogprobs : 0u) | (nt ? fe::kTokTop : 0u)};
    const uint32_t id = (uint32_t)o.row[row].token;
    Frame f(fe::kTokens);
    f.add(&t).add(&id);
    if (lp) f.add(&o.row[row].logprob);
    for (int k = 0; k < nt; ++k) {
      const fe::TopLogprob x{(uint32_t)o.top_id[row][k], o.top_lp[row][k]};
      f.add(&x);
    }
    io_.send(rq.conn, f.take());
  }

  void done(const Request& rq, uint32_t reason, uint32_t cached, uint32_t sent, double t_start, double t_first,
            std::string err = "") {
    // cut short by the server's stop, an answer is an error to its client, not a finished one
    if (reason == fe::kCancelled && Io::stopping(rq)) reason = fe::kError, err = "the server is stopping";
    const double t_end = now_s();
    if (t_first <= 0) t_first = t_end;
    fe::DoneHeader d{};
    d.req_id = rq.h.req_id;
    d.reason = reason;
    d.prompt_tokens = rq.h.n_prompt;
    d.cached_tokens = cached;
    d.completion_tokens = sent;
    d.queue_us = us(t_start - rq.t_recv);
    d.prefill_us = us(t_first - t_start);
    d.decode_us = us(t_end - t_first);
    d.error_len = (uint32_t)err.size();
    io_.send(rq.conn, Frame(fe::kDone).add(&d).add(err.data(), err.size()).take());
    if (!err.empty()) std::printf("serve: request %llu: %s\n", (unsigned long long)rq.h.req_id, err.c_str());
  }

  // A submit: held while its session is in flight (a session's requests run one at a time); waiting while
  // no slot can be had (every slot's session in flight, and no time-slicing); else started.
  void admit(Request&& rq) {
    if (Io::cancelled(rq)) {  // its client left (or the server stops) while it queued: nothing is made for it
      done(rq, fe::kCancelled, 0, 0, now_s(), 0);
      return Io::forget(rq);
    }
    if (flying_.count(rq.h.session_id)) return held_.push_back(std::move(rq));
    const Session* cs = find_at(rq.h.session_id, factor_for((int64_t)rq.prompt.size()));  // its session at its factor
    const int need = need_clamped(rq);
    const uint64_t sid = rq.h.session_id;
    uint64_t base = bkr(rq);
    if (const int fac = factor_for((int64_t)rq.prompt.size()); fac > kBaseFactor && (int)rq.prompt.size() < ctx_limit()) {
      std::vector<int> own;  // the request's own conversation and its base: the window takes them last, not never
      if (cs) own.push_back(cs->h);
      if (const Session* b = base ? find(base) : nullptr) own.push_back(b->h);
      const int w = ensure_window(fac, own, {}, rq.h.req_id);
      if (w == kWinWait) return wait_.push_back(std::move(rq));
      if (w == kWinRefuse) {
        done(rq, fe::kError, 0, 0, now_s(), 0,
             "the engine cannot make the memory a window of " + std::to_string(window_of(fac)) + " positions needs");
        return Io::forget(rq);
      }
      // making the window may have evicted conversations (no disk tier): cs points at one that is gone
      cs = find_at(rq.h.session_id, fac);
    }
    if ((!cs || cs->slot < 0 || !fits(cs->slot, need) || !mem_ok(cs->slot, need, {}, sid, base)) &&
        pick_slot({}, need, sid, base) < 0)
      return wait_.push_back(std::move(rq));
    start(std::move(rq));
  }

  // A request checked, then its flight's first run.
  void start(Request&& rq) {
    Flight f;
    f.rq = std::move(rq);
    const fe::SubmitHeader& h = f.rq.h;
    f.t_start = now_s();
    const int n = (int)f.rq.prompt.size();
    f.fac = factor_for(n);  // the smallest factor whose window holds the prompt
    FacBusy busy_fac(fac_busy_);  // its window stays open while the request is set up (it is in no queue yet)
    busy_fac.add(f.fac);
    std::string err;
    if (n < 1) err = "empty prompt";
    else if (n >= ctx_limit()) err = "a prompt of " + std::to_string(n) + " tokens does not fit the context of " + std::to_string(ctx_limit());
    else if (h.session_id == 0) err = "session id 0";
    else if (h.max_new_tokens < 1) err = "max_new_tokens 0";
    else if (f.rq.stop.size() > (size_t)kMaxStop) err = "more than " + std::to_string(kMaxStop) + " stop ids";
    else if (!(h.temperature >= 0.f) || !(h.top_p > 0.f && h.top_p <= 1.f)) err = "bad sampling parameters";
    for (const int32_t t : f.rq.prompt)
      if (err.empty() && (t < 0 || t >= model::kVocab)) err = "token id out of range";
    for (size_t i = 0, end = 0; err.empty() && i < f.rq.spans.size(); ++i) {  // vision: spans in order, all image ids
      const fe::ImageSpan& s = f.rq.spans[i];
      if (s.rows < 3 || s.start < end || (size_t)s.start + s.rows >= (size_t)n) err = "bad image span";
      for (uint32_t k = 0; err.empty() && k < s.rows; ++k)
        if ((uint32_t)f.rq.prompt[s.start + k] != fe::kImageTokenId) err = "image span over a text token";
      end = (size_t)s.start + s.rows;
    }
    if (!err.empty() || Io::cancelled(f.rq)) {
      done(f.rq, err.empty() ? fe::kCancelled : fe::kError, 0, 0, f.t_start, 0, err);
      return Io::forget(f.rq);
    }
    f.sp.temperature = h.temperature;
    f.sp.top_p = h.top_p;
    f.sp.top_k = (int32_t)std::min<uint32_t>(h.top_k, 1u << 30);
    f.sp.n_top = (h.flags & fe::kFlagLogprobs) ? (int32_t)std::min<uint32_t>(h.n_top_logprobs, model::ops::kTopLogprobs) : 0;
    // stack4: a request that wants logprobs keeps the full head (the certified head gives the argmax, not the partition sum)
    if ((h.flags & fe::kFlagLogprobs) && f.sp.n_top < 1) f.sp.n_top = 1;
    {  // stack4 (DSGB10_FORCE_NTOP, measurement only): every request's rows carry their top logprobs in SampleOut (the step log)
      static const int force_ntop = getenv("DSGB10_FORCE_NTOP") ? atoi(getenv("DSGB10_FORCE_NTOP")) : 0;
      if (force_ntop > 0) f.sp.n_top = std::min(force_ntop, (int)model::ops::kTopLogprobs);
    }
    f.sp.seed = (h.flags & fe::kFlagSeed) ? h.seed : rng_();
    f.sp.ablit = (h.flags & fe::kFlagAblit) ? 1.f : 0.f;  // the runtime abliteration on this request (--ablit)
    flying_.insert(h.session_id);
    const int rows = il_.on ? rows_to_compute(f) : 0;
    if (il_.on && !il::read_whole(il_, rows)) {
      auto L = std::make_unique<Long>();  // a long prompt: read in pieces, between the decoders' windows
      L->f = std::move(f);
      L->rows = rows;
      pfq_.push_back(std::move(L));
      return;
    }
    if (go(f, false)) {  // it decodes in the next run, first
      f.t_paused = now_s();
      rot_.push_front(std::move(f));
    }
  }

  // The rows a request's prompt would compute: after the restart point its session (or base) offers now. A resume's
  // prompt is the request's and the tokens it sent.
  int rows_to_compute(const Flight& f, bool resume = false) {
    const fe::SubmitHeader& h = f.rq.h;
    std::vector<int32_t> cont;
    if (resume) cont = f.rq.prompt, cont.insert(cont.end(), f.out.begin(), f.out.end());
    const std::vector<int32_t>& prompt = resume ? cont : f.rq.prompt;
    const int n = (int)prompt.size();
    forget_evicted();
    unspill(h, f.fac);
    const Session* src = find_at(h.session_id, f.fac);
    if (!src && h.base_session_id) src = find_at(h.base_session_id, f.fac);
    if (!src) return n;
    const int p = std::min(lcp(src->hist, prompt), n - 1);
    return n - choose(*src, src->slot >= 0, p, n).q;
  }

  // The decoders' step (ms) for the piece size: a line through two measured points, by the number of decoding sessions (at most 16)
  double step_ms() const { return rot_.empty() ? 0.0 : 25.77 + 15.9 * (double)(std::min<size_t>(rot_.size(), 16) - 1); }

  // One pass of the scheduler while long prompts exist (il::next): admit one request, begin the front long prompt,
  // read a piece of it, or give the decoders their window
  void il_pass() {
    Long& L = *pfq_.front();
    const il::Settings& st = il_;
    il::State s;
    s.decoders = !rot_.empty();
    s.owed_ms = owed_ms_;
    s.reader = L.begun;
    s.queued = !L.begun;
    s.reader_fits = !L.begun && pick_slot({}, need_long(L), L.f.rq.h.session_id, bkf(L.f)) >= 0;
    Long* cand = nullptr;  // the second reader's candidate: the first queued one of at most push_rows rows that fits
    if (L.begun && !push_ && st.push_rows > 0)
      for (size_t i = 1; i < pfq_.size() && !cand; ++i) {
        Long& q = *pfq_[i];
        if (!q.begun && q.rows <= st.push_rows &&
            pick_slot({L.f.rq.h.session_id}, need_long(q), q.f.rq.h.session_id, bkf(q.f)) >= 0)
          cand = &q;
      }
    s.push_fits = cand != nullptr;
    const bool waiting = !wait_.empty() && window_ready(wait_.front()) &&
                         pick_slot({}, need_clamped(wait_.front()), wait_.front().h.session_id, bkr(wait_.front())) >= 0;
    s.arrivals = !ready_.empty() || waiting || io_.submit_waiting();
    s.admitted_last = il_admitted_;
    const double t0 = now_s();
    const il::Next nx = il::next(s);
    il_admitted_ = nx == il::Next::Admit;
    switch (nx) {
      case il::Next::Admit: {
        Event e;
        if (!ready_.empty()) {
          Request rq = std::move(ready_.front());
          ready_.pop_front();
          admit(std::move(rq));
        } else if (waiting) {
          Request rq = std::move(wait_.front());
          wait_.pop_front();
          admit(std::move(rq));
        } else if (io_.pop(e, 0)) {
          if (e.release) {
            if (flying_.count(e.session)) later_.insert(e.session);
            else release(e.session);
          } else if (!e.rq.conn->dead) {
            admit(std::move(e.rq));
          }
        }
        if (s.decoders) owed_ms_ += il::window_ms(st, 1e3 * (now_s() - t0));  // a whole prompt read meanwhile
        break;
      }
      case il::Next::BeginReader: {
        Long& X = L.begun ? *cand : L;  // the second reader or the front one
        if (!X.resume) X.f.t_start = t0;  // a first run's prefill starts here: its wait in the queue is queue time (DONE queue_us)
        if (L.begun) {  // the second reader, beside the front one (whose slot it may not take)
          if (pf_begin(cand->f, cand->resume, {L.f.rq.h.session_id}, cand->r)) cand->begun = true, push_ = cand;
          else drop_long(cand);
        } else if (pf_begin(L.f, L.resume, {}, L.r)) {
          L.begun = true;
        } else {
          drop_long(&L);  // it ended at its start
        }
        // the decoders have waited since their last run (the arrival, its admission, this setup): owed before a piece
        if (s.decoders) owed_ms_ += il::window_ms(st, 1e3 * (now_s() - std::max(dec_end_, t0 - 1.0)));
        break;
      }
      case il::Next::Piece: {
        Long& P = push_ ? *push_ : L;  // the second reader reads first
        Reader& r = P.r;
        const bool dec = !rot_.empty();
        il::Settings sr = st;  // the pieces are bounded by this request's capacity, not the standing one
        sr.prefill_rows = r.cap;
        const int R = il::adjust_piece(il::piece_rows(sr, il_cost_, step_ms(), dec), r.pos, r.n, r.ced, r.cap);
        const int next = il::piece_rows(sr, il_cost_, step_ms(), dec);
        const bool ok = pf_piece(r, R, next);
        const double ms = 1e3 * (now_s() - t0);
        if (ok) il_cost_.k = 0.8 * il_cost_.k + 0.2 * ms / (il_cost_.fixed_ms + il_cost_.per_row_ms * R);
        if (dec) owed_ms_ += il::window_ms(st, ms);
        if (!ok || r.pos >= r.n) {
          if (pf_end(r)) {  // it decodes from here, first in the next run
            P.f.t_paused = now_s();
            rot_.push_front(std::move(P.f));
          }
          drop_long(&P);
        }
        break;
      }
      case il::Next::Decode:
        burst_ms_ = std::min<double>(owed_ms_ > 0 ? owed_ms_ : st.burst_max_ms, st.burst_max_ms);
        burst_end_ = 0;
        run();
        burst_ms_ = 0;
        owed_ms_ = std::max(0.0, owed_ms_ - 1e3 * (now_s() - t0));
        break;
      case il::Next::Idle: {
        Event e;  // the front long prompt waits for a slot: wait for a request or a release
        if (io_.pop(e, 50)) {
          if (e.release) {
            if (flying_.count(e.session)) later_.insert(e.session);
            else release(e.session);
          } else if (!e.rq.conn->dead) {
            admit(std::move(e.rq));
          }
        }
        break;
      }
    }
  }

  // A CONFIG frame's settings: each field in its flag's range taken (main.cpp's checks), the others kept
  void apply_config(const fe::Config& c) {
    il::Settings n = il_;
    if (c.on == 0 || c.on == 1) n.on = c.on == 1;
    if (c.whole_rows >= n.prefill_rows && c.whole_rows <= o_.max_ctx) n.whole_rows = (int)c.whole_rows;
    if (c.push_rows >= 0 && c.push_rows <= o_.max_ctx) n.push_rows = (int)c.push_rows;
    if (c.gap_ms >= 100 && c.gap_ms <= 60000) n.gap_ms = (int)c.gap_ms;
    if (c.share >= 10 && c.share <= 90) n.share = (int)c.share;
    if (c.burst_max_ms >= 50 && c.burst_max_ms <= 60000) n.burst_max_ms = (int)c.burst_max_ms;
    if (!o_.ced) n.on = false;  // --exact reads every prompt whole
    il_ = n;
    if (c.keep_days >= 1 && c.keep_days <= 3650 && c.keep_days * 86400.0 != keep_age_s_) {
      keep_age_s_ = c.keep_days * 86400.0;
      std::printf("serve: sessions unused for %lld days are dropped\n", (long long)c.keep_days);
    }
    if (c.ssd_gib >= 1 && c.ssd_gib <= 131071 && m_.state_disk_capacity() &&  // the disk tier's size: when nothing runs
        (size_t)c.ssd_gib << 30 != m_.state_disk_capacity()) {
      ssd_pending_ = c.ssd_gib;
      std::printf("serve: the disk tier goes to %lld GiB when no request is in flight\n", (long long)c.ssd_gib);
    }
    if ((c.extend == 0 || c.extend == 1) && (c.extend == 1) != extend_) {
      extend_ = c.extend == 1;
      std::printf("serve: extend context %s: %s\n", extend_ ? "on" : "off",
                  extend_ ? "the factor per request, up to the context" : "the 1,048,576-token window only");
    }
    published_ = 0;  // the next STATS shows it
    std::printf("serve: interleave settings: %s, whole %d rows, push %d rows, gap %d ms, share %d %%, window at most %d ms\n",
                il_.on ? "on" : "off", il_.whole_rows, il_.push_rows, il_.gap_ms, il_.share, il_.burst_max_ms);
  }

  // A decode run's window: set at its first step; over when a long prompt waits and the window has passed
  bool burst_over() {
    if (burst_ms_ <= 0 || pfq_.empty()) return false;
    const double t = now_s();
    if (burst_end_ == 0) burst_end_ = t + burst_ms_ / 1e3;
    return t >= burst_end_;
  }

  // A prompt being prefilled: what go()'s setup leaves for its pieces and its end (pf_begin, pf_piece, pf_end)
  struct Reader {
    Flight* f = nullptr;
    bool resume = false, ced = false, cancelled = false;
    std::vector<int32_t> cont;  // a resume's prompt: the request's and the tokens sent so far
    Session* S = nullptr;
    int s = -1, n = 0, q = 0, pos = 0, first_row = 0;
    model::SampleOut first{};
    Turn resumed;
    Turn* tu = nullptr;
    double t0 = 0;
    // The set of row-sized prefill buffers the prompt is read on (Cmd::rows_set) and the rows a piece may have at most: chosen at the
    // prompt's begin, one value for every piece (its look-ahead lookups, its CED state and its snapshots belong to one set).
    int rows_set = 0, cap = 0;
    const std::vector<int32_t>& prompt() const { return resume ? cont : f->rq.prompt; }
  };

  // Flight f's prompt run in a slot: the request's first (resume false), or the resume of a request whose
  // session lost its slot, which runs the prompt and the tokens sent so far (the session holds them: it
  // restarts at their last position, exact continuation, one row; `keep`: the sessions whose slots it may
  // not take). Either way it sends a token, the flight's next to feed, and leaves its session live in the
  // slot; false when that token (or a cancel) ended the request. The sampler's noise is a function of the
  // seed and the position, so a request run in pieces samples what a single run would. Here the pieces run
  // back to back: the whole-prompt path.
  bool go(Flight& f, bool resume, const std::set<uint64_t>& keep = {}) {
    Reader r;
    if (!pf_begin(f, resume, keep, r)) return false;
    while (r.pos < r.n)
      if (!pf_piece(r, std::min(r.cap, r.n - r.pos), r.cap)) break;
    return pf_end(r);
  }

  // The prompt's setup: the slot, where the state comes from, the restart point q, the state operations and the
  // prompt's tokens on every rank. False when the request ended here (a resume cancelled, or past the context).
  bool pf_begin(Flight& f, bool resume, const std::set<uint64_t>& keep, Reader& r) {
    r.f = &f, r.resume = resume;
    const fe::SubmitHeader& h = f.rq.h;
    const double t0 = r.t0 = now_s();
    std::vector<int32_t>& cont = r.cont;
    if (resume) {
      if (Io::cancelled(f.rq)) return ended_at_start(f, fe::kCancelled);
      cont.reserve(f.rq.prompt.size() + f.out.size());
      cont.assign(f.rq.prompt.begin(), f.rq.prompt.end());
      cont.insert(cont.end(), f.out.begin(), f.out.end());
      if ((int)cont.size() >= ctx_limit()) return ended_at_start(f, fe::kContext);
    }
    const std::vector<int32_t>& prompt = resume ? cont : f.rq.prompt;
    const int n = (int)prompt.size();
    r.tu = resume ? &r.resumed : &f.tu;  // the log splits a request's first run
    Turn& tu = *r.tu;
    forget_evicted();
    const int fac = f.fac = std::max(f.fac, factor_for(n));  // a resume's state is read again at the factor its length needs
    if (fac > kBaseFactor && !m_.window_is_open(fac)) {  // admit() and the turns' loop open it first: never reached
      std::printf("serve: req %llu: the window of factor %d is not open; the request ends\n", (unsigned long long)h.req_id, fac);
      return ended(f, fe::kError);
    }
    unspill(h, fac);

    // ---- the slot, where the state comes from, and from which position on it is computed
    // Only a session of this request's factor can continue it (find_at/acquire): its own, else its base's. A request
    // whose session is at another factor makes a new one (the old stays under an alternate key for its siblings).
    Session* S = acquire(h.session_id, fac);
    Session* src = S;
    uint64_t src_id = h.session_id;
    if (!S && h.base_session_id) src = find_at(h.base_session_id, fac, &src_id);
    if (src && src->factor != fac) die("serve: req %llu: a state of factor %d would continue a run of factor %d", (unsigned long long)h.req_id, src->factor, fac);
    // the session's own slot. A fork takes its base's when no request of the base is in flight: the base's
    // state goes to the store, and the fork goes on from it in place. A base in flight keeps its slot; its
    // state goes to the store, and the fork starts from there in a slot of its own.
    // With the pool, a slot is taken only when mem_ok() says the pool can hold the request there, with the same
    // `keep` admit() and run() checked: a slot that cannot is left (the session's state to the store), and
    // pick_slot() gives one that can, so make_room() below always finds the memory.
    const int need = need_now(f);
    const uint64_t sid = h.session_id, base = bk(h, fac);
    int s = S ? S->slot : -1;
    if (s >= 0 && (!fits(s, need) || !mem_ok(s, need, keep, sid, base)))
      persist_slot(s, true, holds(sid, base)), s = -1;  // a small slot (the session moves to a big one), or memory only another slot has
    if (!S && src && src->slot >= 0) {
      if (!flying_.count(src_id) && fits(src->slot, need) && mem_ok(src->slot, need, keep, sid, base)) s = src->slot;
      else persist_slot(src->slot, false, holds(sid, base));
    }
    if (s < 0) s = pick_slot(keep, need, sid, base);
    if (s < 0) die("serve: no slot for request %llu", (unsigned long long)h.req_id);  // admit() saw one
    const bool src_live = src && src->slot == s;
    // the session that holds slot s (unless it is the fork's own base) leaves the line first, in a command of its own:
    // its numbers then belong to a conversation no line holds, which make_room may evict for this request
    if (slot_[s] && slot_[s] != sid && !src_live) persist_slot(s, true, holds(sid, base));
    make_room(s, need, keep, sid, base);
    const int p = src ? std::min(lcp(src->hist, prompt), n - 1) : 0;
    const Plan pl = src ? choose(*src, src_live, p, n) : Plan{};
    const int q = pl.q;
    NVTX_RANGE("req %llu new %d ctx %d %s", (unsigned long long)h.req_id, n - q, n, kHow[pl.how]);
    if (!resume) {
      f.q = q, f.pl = pl;
      // where the state came from, for the log (counts only), and why a start is cold: no base named
      // (the front-end found no session to continue), a session or base this engine does not hold (lost
      // at a restart, or evicted), or a common prefix short of every restart point of the one it holds.
      // A request's first run only: a resume continues from the session's own end.
      char from[192];
      if (!src) {
        if (h.base_session_id)
          std::snprintf(from, sizeof from, "a fork of %016llx, which this engine does not hold",
                        (unsigned long long)h.base_session_id);
        else
          std::snprintf(from, sizeof from, "no base named, and this engine does not hold the session");
      } else {
        const int k = S ? std::snprintf(from, sizeof from, "its own history: common prefix %d ids", p)
                        : std::snprintf(from, sizeof from, "a fork of %016llx: common prefix %d ids",
                                        (unsigned long long)h.base_session_id, p);
        if (pl.how == kCold && k > 0 && k < (int)sizeof from) {
          const int r = first_restart(*src);
          if (r < 0) std::snprintf(from + k, sizeof from - k, ", and it has no restart point");
          else if (p < r) std::snprintf(from + k, sizeof from - k, ", under its first restart point %d", r);
          else std::snprintf(from + k, sizeof from - k, ", and no restart point up to it is usable");
        }
      }
      std::printf("serve: req %llu factor %d (window %d): restart: %s; reused %d (%s)\n", (unsigned long long)h.req_id, fac,
                  window_of(fac), from, q, kHow[pl.how]);
    }
    if (src && q > 0) {  // the cut at q copies the chunks the session shares with another (own_tail), a fork's partial piece
      // takes a number per list: those numbers must be there before the operations run (the model dies on a pool held by lines)
      const int64_t cow = (int64_t)m_.state_cow_numbers(src->h, S ? S->h : -1, q);
      // one count with the request's own reservation (make_room above): the copy's numbers must not come out of what the
      // reservation was admitted against
      if (cow > 0)
        give_back((int64_t)m_.slot_chunks_for(s, need) - own(s, keep, sid, base) + cow, keep, sid, base, s, 0,
                  "a request and its copy at the restart point");
    }
    std::vector<StateOp> v, pv;  // pv: a fork's base's turn on this line, saved after the reservation
    if (slot_[s] && slot_[s] != h.session_id) {  // the slot's session (a fork's base too) keeps its state in the store
      if (Session* y = find(slot_[s])) {
        // a fork's base on this line saves its turn after the reservation; the fork then takes its numbers by count
        persist_ops(*y, src_live ? pv : v);
        y->slot = -1;
      }
      // the line lets its conversation go (it keeps its numbers); a fork's base leaves at the attach below
      if (!src_live) v.push_back({kStRelease, 0, s, 0});
      slot_[s] = 0;
    }
    v.push_back({kStReserve, 0, s, need});  // the request's prompt and a horizon of response are committed (growth)
    f.upto = need;
    v.insert(v.end(), pv.begin(), pv.end());
    if (!S) {
      Session ns;
      ns.h = next_h_++;
      ns.factor = fac;
      S = &sess_.emplace(h.session_id, std::move(ns)).first->second;
      if (resume)
        std::printf("serve: req %llu factor %d (window %d): a new session for its %d positions, read again (%s)\n",
                    (unsigned long long)h.req_id, fac, window_of(fac), n - q, kHow[pl.how]);
    }
    v.push_back({kStAttach, S->h, s, 0});  // the slot holds S: its rows are S's chunk lists (the chunk pool, no copy)
    v.push_back({kStFactor, S->h, s, fac});  // the slot rotates with S's factor from its first step; the store tags S
    if (src) {
      // S's own state is cut to q; a fork's base's [0, q) is copied into S's numbers (from a base the slot held too)
      if (q > 0 && (!src_live || src != S)) v.push_back({kStLoad, src->h, q, 0});
      if (pl.how == kSnap) v.push_back({kStRestore, src->h, pl.snap, 0});
      if (pl.how == kEnd) v.push_back({kStRestoreEnd, src->h, 0, 0});
      if (src != S) v.push_back({kStCopySnaps, src->h, S->h, q});  // a fork: src's snapshots are S's too
      else v.push_back({kStDrop, S->h, q, 0});                     // S forgets what follows q
    }
    // the new live state's exact ranges
    if (pl.how == kCold) S->lo = S->clo = S->w = 0;
    else if (pl.how == kSnap) S->lo = pl.snap - 128, S->clo = q, S->w = q;
    else if (pl.how == kEnd) S->lo = src->end_lo, S->clo = src->end_clo, S->w = src->end_w;
    else S->lo = src->lo, S->clo = src->clo, S->w = src->w;
    slot_[s] = h.session_id;
    S->slot = s;
    S->used = ++clock_;
    S->t_used = unix_now();
    S->hist.assign(prompt.begin(), prompt.begin() + q);
    if (S->pe_snap > q) S->pe_snap = -1;  // dropped
    // CED: S's encoder-only snapshots up to q (a fork's are src's), where its decoder rings are valid
    if (src && src != S) S->enc_snaps = src->enc_snaps;
    S->enc_snaps.erase(std::remove_if(S->enc_snaps.begin(), S->enc_snaps.end(), [q](int x) { return x > q; }),
                       S->enc_snaps.end());
    S->dec_lo = !src || pl.how == kCold ? 0
                : pl.how == kSnap       ? std::max(0, q - 127)
                : pl.how == kEnd        ? src->end_dec_lo
                                        : src->dec_lo;
    const bool ced = o_.ced && n - q >= kReplay;  // else an exact continuation
    if (ced) S->dec_lo = kNever;                  // until the replay has run

    // ---- the prompt's tokens (Engram hashes 3 positions back), with the state operations, in the slot
    {
      Cmd c{};
      c.op = kCmdTokens;
      c.slot = s;
      c.pos0 = std::max(0, q - 3);
      c.n = n - c.pos0;
      if (v.size() > (size_t)kMaxOps) die("serve: %zu state operations for one command", v.size());
      c.n_ops = (int)v.size();
      std::copy(v.begin(), v.end(), c.ops);
      tu = Turn{};
      if (!resume && o_.profile_from_req && h.req_id == o_.profile_from_req && !g_prof) {
        c.flags |= model::kFlagProfile;  // the other ranks open it too (cmd.cpp follow)
        CUDA_CHECK(cudaProfilerStart());
        g_prof = true;
      }
      m_.mark(0);
      issue(c);
      m_.mark(1);
      std::vector<int32_t> t(prompt.begin() + c.pos0, prompt.end());
      for (const fe::ImageSpan& sp : f.rq.spans)  // vision: image positions are dead to Engram (-1)
        for (int k = std::max<int>(sp.start, c.pos0); k < (int)(sp.start + sp.rows); ++k) t[k - c.pos0] = -1;
      bcast_payload(ring_, t.data(), t.size() * 4, 0);
      m_.set_tokens(c.pos0, c.n, t.data());
      if (!f.rq.spans.empty()) images(f.rq, s);
      tu.ops_ms = 1e3 * (now_s() - (resume ? t0 : f.t_start));
    }
    r.S = S, r.s = s, r.n = n, r.q = q, r.ced = ced, r.pos = q;
    return true;
  }

  // One piece of the prompt: rows [pos, pos + R), the next piece's Engram look-ahead sized next_rows. False when
  // the request was cancelled before it (nothing run).
  bool pf_piece(Reader& r, int R, int next_rows) {
    Flight& f = *r.f;
    const fe::SubmitHeader& h = f.rq.h;
    const std::vector<int32_t>& prompt = r.prompt();
    Session* S = r.S;
    const int s = r.s, n = r.n, q = r.q;
    const bool ced = r.ced, resume = r.resume;
    int& pos = r.pos;
    model::SampleOut& first = r.first;
    int& first_row = r.first_row;
    Turn& tu = *r.tu;
    if (Io::cancelled(f.rq)) {
      r.cancelled = true;
      return false;
    }
    const bool last = pos + R == n;
    if (R > r.cap) die("serve: a piece of %d rows on a set of %d (req %llu)", R, r.cap, (unsigned long long)h.req_id);
    Cmd c{};
    c.slot = s;
    c.pos0 = pos;
    c.rows = R;
    c.rows_set = r.rows_set;
    c.sp = f.sp;
    c.handle = S->h;
    for (int sn = (pos / o_.snap_every + 1) * o_.snap_every; sn <= pos + R; sn += o_.snap_every)
      if (sn >= 128 && c.n_snaps < kMaxSnaps && (!o_.snap_thin || keep_snap(sn, n, o_.snap_every)))
        c.snaps[c.n_snaps++] = sn;  // none that the prompt's end would thin at once
    room_for_bookmarks(c.n_snaps, {}, h.session_id, bk(h, f.fac));
    wd_.started = now_s();
    bool img = false;  // vision: rows of an image span take the chunk path (the step path has no injection)
    for (const fe::ImageSpan& sp : f.rq.spans) img = img || (int)(sp.start + sp.rows) > pos;
    if (R <= o_.small_rows && !ced && !img) {
      c.op = kCmdRows;
      std::copy(prompt.begin() + pos, prompt.begin() + pos + R, c.tokens);
      issue(c);
      const uint64_t k = run_rows(m_, c);
      first = m_.wait(k);
      tu.rows_ms += m_.last_gpu_ms();
      first_row = R - 1;
      S->w = std::max(S->w, pos + R + (m_.spec() ? model::ops::kDrafts : 0));  // its draft's block keys
    } else {
      c.op = kCmdPrefill;
      c.flags = (last ? kPfSample : 0) | (ced ? kPfCed : 0);
      const int rest = n - pos - R;
      c.next = rest > o_.small_rows || (ced && rest > 0) ? std::min(next_rows, rest) : 0;
      for (int i = 0; i < c.n_snaps; ++i)
        if (ced && c.snaps[i] < n) S->enc_snaps.push_back(c.snaps[i]);  // the replay covers only one at n
      issue(c);
      const uint64_t k = run_prefill(m_, c);
      if (k != ~0ull) first = m_.wait(k), first_row = 0;
      if (!last) m_.drain();  // every chunk but the last drains here; the last chunk's tail (the draft windows' seeding) drains after tok0 is out
      if (!last) tu.add_split(m_);
      else tu.pending = true;
      ++tu.chunks;
      S->w = std::max(S->w, pos + R);
    }
    wd_.started = 0;
    S->hist.insert(S->hist.end(), prompt.begin() + pos, prompt.begin() + pos + R);
    pos += R;
    if (!resume) {  // the dashboard: this request's prompt tokens computed, of those it computes (after the reuse)
      const fe::Progress pr{h.req_id, (uint32_t)(pos - q), (uint32_t)(n - q)};
      io_.send(f.rq.conn, Frame(fe::kProgress).add(&pr).take());
    }
    return true;
  }

  // The prompt's end: images cleared, the first token sent, the prompt-end snapshot. True when the request goes on
  // decoding; false when the token (or a cancel) ended it.
  bool pf_end(Reader& r) {
    Flight& f = *r.f;
    const fe::SubmitHeader& h = f.rq.h;
    const bool resume = r.resume, ced = r.ced, cancelled = r.cancelled;
    const int s = r.s, n = r.n;
    Session* S = r.S;
    const double t0 = r.t0;
    model::SampleOut& first = r.first;
    const int first_row = r.first_row;
    Turn& tu = *r.tu;
    if (!f.rq.spans.empty()) {  // vision: the spans are for this prompt's chunks only
      Cmd c{};
      c.op = kCmdImages;
      c.slot = s;
      c.n = 0;
      issue(c);
      m_.set_images(0, nullptr, nullptr, nullptr);
    }
    const double t_first = now_s();
    if (resume) f.resume_s += t_first - t0;
    else f.t_first = t_first;
    if (ced && !cancelled) S->dec_lo = n - kReplay;
    auto drain = [&] {
      wd_.started = now_s();
      m_.drain();
      wd_.started = 0;
    };
    if (cancelled || Io::cancelled(f.rq)) {
      drain();
      return ended(f, fe::kCancelled);
    }

    // ---- the token, the next to feed; then the last chunk's tail drains
    const int32_t tok0 = first.row[first_row].token;
    tokens(f.rq, first, first_row);
    f.out.push_back(tok0);
    drain();
    tu.ops_gpu_ms = m_.mark_ms(0, 1);
    if (tu.pending) tu.add_split(m_);  // the last chunk's
    tu.tail_ms = 1e3 * (now_s() - t_first);
    // the window snapshot at the prompt's end (a first run's): a later request that keeps the prompt but
    // not what was generated after it (an agent dropping the reasoning, say) restarts there
    if (!resume && n >= 128 && n % o_.snap_every != 0 && n != S->pe_snap) {
      Cmd c{};
      c.op = kCmdState;
      c.slot = s;
      if (S->pe_snap >= 0) c.ops[c.n_ops++] = {kStDropSnap, S->h, S->pe_snap, 0};
      c.ops[c.n_ops++] = {kStSnap, S->h, n, 0};
      S->pe_snap = n;
      room_for_bookmarks(1, {}, h.session_id, bk(h, f.fac));
      issue(c);
    }
    if (stop_id(f, tok0)) return ended(f, fe::kStop);
    if ((int)f.out.size() >= (int)h.max_new_tokens) return ended(f, fe::kLength);
    return true;
  }

  // Vision: rq's spans and rows to every rank (kCmdImages), slot s's: for its prefill chunks, whatever runs between them.
  void images(const Request& rq, int s) {
    const double t0 = now_s();
    Cmd c{};
    c.op = kCmdImages;
    c.slot = s;
    c.n = (int)rq.spans.size();
    issue(c);
    std::vector<int32_t> lh(2 * rq.spans.size());
    for (size_t i = 0; i < rq.spans.size(); ++i)
      lh[i] = (int32_t)rq.spans[i].start, lh[rq.spans.size() + i] = (int32_t)(rq.spans[i].start + rq.spans[i].rows);
    bcast_payload(ring_, lh.data(), lh.size() * 4, 0);
    std::vector<uint16_t> rows(rq.img);
    bcast_payload(ring_, rows.data(), rows.size() * 2, 0);
    m_.set_images(c.n, lh.data(), lh.data() + c.n, rows.data());
    std::printf("serve: req %llu images: %d spans, %zu rows, sent to every rank in %.1f ms\n",
                (unsigned long long)rq.h.req_id, c.n, rows.size() / fe::kImageDim, 1e3 * (now_s() - t0));
  }

  bool ended(Flight& f, uint32_t why) {
    finish(f, why);
    return false;
  }
  static bool stop_id(const Flight& f, int32_t t) { return std::count(f.rq.stop.begin(), f.rq.stop.end(), t) > 0; }

  // Growth (the chunk pool): a request reserves its prompt and a horizon of response (--grow-ahead positions, at most its
  // max_new); a response that reaches its horizon ends its run (a turn) and extends the reservation at its next run when
  // the pool can hold it, else waits like a paused one. The running responses' reservations are the margin new requests
  // are admitted against.
  // Positions a new request may reach before its first extension: its prompt and the horizon (--grow-ahead).
  int need_clamped(const Request& rq) const {
    return (int)std::min<int64_t>((int64_t)rq.prompt.size() + std::min<int64_t>(rq.h.max_new_tokens, o_.grow_ahead),
                                  cap_of(factor_for((int64_t)rq.prompt.size())));
  }
  // A flight's: from where it is (its prompt and what it generated) a horizon on, at most its prompt + max_new and its
  // factor's window.
  int need_now(const Flight& f) const {
    const int64_t at = (int64_t)f.rq.prompt.size() + (int64_t)f.out.size();
    const int64_t end = (int64_t)f.rq.prompt.size() + f.rq.h.max_new_tokens;
    return (int)std::min<int64_t>({at + o_.grow_ahead, end, cap_of(f.fac)});
  }
  bool fits(int s, int need) const { return need <= m_.slot_ctx(s); }
  // The sessions the store leaves alone for a request of session `sid` (forking from `base`): theirs, and those of the
  // slots in `keep` or in flight (all_live: of every slot, which is what Model::slot_reserve leaves alone), as handles.
  std::vector<int> pins(uint64_t sid, uint64_t base, const std::set<uint64_t>* keep = nullptr, bool all_live = false) const {
    std::vector<int> v;
    for (const uint64_t id : {sid, base}) {
      const auto it = id ? sess_.find(id) : sess_.end();
      if (it != sess_.end()) v.push_back(it->second.h);
    }
    if (keep)
      for (const uint64_t id : slot_) {
        if (!id || (!all_live && idle_id(id, *keep, sid, base, paused_too))) continue;
        const auto it = sess_.find(id);
        if (it != sess_.end()) v.push_back(it->second.h);
      }
    return v;
  }
  // Numbers eviction can give back for this request (Model::state_giveable): those only conversations no line holds name,
  // the sessions in pins() excepted.
  int64_t giveable(const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, bool all_live) const {
    return (int64_t)m_.state_giveable(pins(sid, base, &keep, all_live));
  }
  // What persist_slot's commands name for the store to leave alone: the request's own session and its base
  std::vector<int> holds(uint64_t sid, uint64_t base) const { return pins(sid, base); }
  // A session paused between turns (time-slicing): in the turns' queue, not in the run being built. Its slot can be
  // taken: the session's state goes to the store, and it resumes in a slot of its own when its turn comes (go with
  // resume). Without turns (--slice 0) a paused request keeps its slot.
  bool paused(uint64_t id) const {
    if (o_.slice <= 0) return false;
    for (const Flight& f : rot_)
      if (f.rq.h.session_id == id) return true;
    return false;
  }
  // A session whose slot memory the request may take (make_room, mem_ok): not the request's own or its base, not in the
  // run (`keep`), and with no request in flight, or paused between turns. Counting paused ones keeps every admitted
  // request able to run: a flight that lost its slot finds one when its turn comes.
  bool idle_id(uint64_t id, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base) const {
    return id != sid && id != base && !keep.count(id) && (!flying_.count(id) || paused(id));
  }
  // A slot whose session no request uses and the request does not need: its memory can be given back (make_room)
  bool idle_slot(int t, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base) const {
    return idle_id(slot_[t], keep, sid, base);
  }
  // Chunks go() will map before it decides the slot: it brings the request's session (or its fork's base) back from the
  // disk tier first (unspill). An upper bound (Model::state_restore_chunks: every block of the record), counted in
  // mem_ok's want so that an admitted request finds its slot after the restore.
  int64_t restore_extra(uint64_t sid, uint64_t base) const {
    int64_t n = 0;
    for (const uint64_t id : {sid, base}) {
      const auto it = id ? sess_.find(id) : sess_.end();
      if (it != sess_.end()) n += (int64_t)m_.state_restore_chunks(it->second.h);
    }
    return n;
  }
  // What slot s's commitment gives a request that takes it: all of it, unless the slot's session stays (the request's base,
  // or one the run keeps). That session keeps its chunk numbers when the request's session takes the line (a line holds
  // no memory of its own; a fork's base is copied into new numbers), so only the reservation above them is the request's.
  int64_t own(int s, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base) const {
    const int64_t c = m_.slot_committed(s);
    return slot_[s] && slot_[s] != sid && !idle_id(slot_[s], keep, sid, base) ? c - m_.slot_mapped(s) : c;
  }
  // What one admission pass reads once (pick_slot asks mem_ok for many slots in turn; the state does not change inside it)
  struct MemCache {
    bool ex = false, gv = false;
    RestoreAsk ask;
    int64_t img = 0, giveable = 0;
  };
  // Whether the pool (Model::slot_*, Options::vmm) can hold `need` positions in slot s: s's own commitment is replaced
  // by the request's; what is uncommitted plus what can be given back (idle slots' memory and, with the store on the
  // pool, what the store holds under unused sessions: make_room, Model::slot_reserve) must cover it. The request's own
  // session and base on disk count as restored first (restore_extra). Without the pool: always.
  bool mem_ok(int s, int need, const std::set<uint64_t>& keep, uint64_t sid = 0, uint64_t base = 0,
              MemCache* mc = nullptr) const {
    MemCache tmp;
    MemCache& c = mc ? *mc : tmp;
    if (!c.ex) c.extra = restore_extra(sid, base), c.ex = true;
    const int64_t want = (int64_t)m_.slot_chunks_for(s, need) + c.extra;
    int64_t have = (int64_t)m_.slot_uncommitted() + own(s, keep, sid, base);
    for (int t = 0; t < (int)slot_.size() && have < want; ++t)
      if (t != s && idle_slot(t, keep, sid, base)) have += m_.slot_committed(t);
    if (have < want) {
      if (!c.gv) c.giveable = giveable(keep, sid, base, false), c.gv = true;
      have += c.giveable;
    }
    return have >= want;
  }
  // Numbers for `snaps` window bookmarks a command takes (a prefill piece's, a decode run's, the prompt's end): the pool must
  // give them, or the request loses a restart point it would have had. Idle and paused lines let go first (give_back).
  void room_for_bookmarks(int snaps, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base) {
    if (snaps <= 0) return;
    give_back((int64_t)m_.state_bookmark_numbers(snaps), keep, sid, base, -1, (int64_t)m_.state_bookmark_numbers(1), "window bookmarks",
              false);
  }
  // Slot memory for a request of `need` positions in slot s: the least recently used sessions with no request in
  // flight leave their slots (their state saved to the store, their chunks back to the pool) until the pool can
  // commit it, counting what the store will give back at the reservation (giveable, with every slot's session left
  // alone: what Model::slot_reserve does). go() took slot s only where mem_ok() said it can, with the same `keep`:
  // running out here is a broken count, which stops the engine rather than run a request on memory it does not hold.
  void make_room(int s, int need, const std::set<uint64_t>& keep, uint64_t sid = 0, uint64_t base = 0) {
    if (!room_for(s, need, keep, sid, base))
      die("serve: no slot memory for %d positions in slot %d (%d chunks wanted)", need, s, m_.slot_chunks_for(s, need));
  }
  // make_room without the die: false when only running requests hold the rest (the caller waits)
  bool room_for(int s, int need, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base) {
    char why[48];
    std::snprintf(why, sizeof why, "a request of %d positions", need);
    return give_back((int64_t)m_.slot_chunks_for(s, need) - own(s, keep, sid, base), keep, sid, base, s, 0, why);
  }
  // Whether every request in flight holds what it was reserved beyond what is free and what eviction gives: the numbers
  // a consumer asks for (`n`, and best effort `margin` more) must come from free and unreserved numbers and from the
  // conversations no running request uses. Idle and paused lines let their conversations go, least recently used first (the
  // request's own session `sid` and its base excepted, and the sessions in `keep`: they are the request's), then they are
  // evictable store like the rest (Model::state_giveable); when that is not enough, a window no request needs is closed.
  // false: only running requests hold the rest. `skip_slot` is the consumer's own line. `margin` is for consumers a refusal
  // would cost a restart point (a bookmark: each detach takes an end bookmark of its own); `n` alone decides the answer.
  // `may_close`: a window no request needs is closed only for a consumer that would otherwise wait or fail (a request, its
  // growth, a copy, a window); a bookmark or a read-back that finds no room only starts a later turn further back.
  bool give_back(int64_t n, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, int skip_slot, int64_t margin,
                 const char* why, bool may_close = true) {
    auto short_by = [&](int64_t w) {
      const int64_t u = (int64_t)m_.slot_uncommitted();
      return u < w && u + giveable(keep, sid, base, true) < w;
    };
    if (!short_by(n + margin)) return true;
    if (loan_give_way(now_s())) return false;  // tier 0: the second set of prefill row buffers goes back before any line or conversation does
    std::vector<reclaim::Line> lines;
    for (int t = 0; t < (int)slot_.size(); ++t) {
      if (!slot_[t]) continue;
      const auto it = sess_.find(slot_[t]);
      lines.push_back({t, slot_[t], it == sess_.end() ? 0 : it->second.used, (int64_t)m_.slot_committed(t),
                       idle_slot(t, keep, sid, base, paused_too)});
    }
    reclaim::detach_until(
        std::move(lines), skip_slot, [&] { return short_by(n + margin); },
        [&](int t) {
          std::printf("serve: slot %d (session %016llx) gives its %d chunks back to the pool for %s\n", t,
                      (unsigned long long)slot_[t], m_.slot_committed(t), why);
          persist_slot(t, true, holds(sid, base));
        });
    if (!short_by(n)) return true;
    return may_close && close_unused_window() && !short_by(n);
  }
  // A slot for a session that holds none, among those that hold `need` positions (with big slots, the small
  // ones first): a free one; else the least recently used one whose session no request uses (its state goes
  // to the store); else, with time-slicing, the slot of the paused request whose turn comes last. Never the
  // slot of a session in `keep`. -1: none. (sid, base: the request's session and its base, whose stored state the
  // store keeps.)
  int pick_slot(const std::set<uint64_t>& keep, int need = 0, uint64_t sid = 0, uint64_t base = 0) const {
    const int ns = (int)slot_.size();
    const int big = m_.slot_ctx(0);
    MemCache mc;
    for (int pass = 0; pass < 2; ++pass)  // small slots first (big_slots), so that the big ones stay for long requests
      for (int s = 0; s < ns; ++s)
        if (!slot_[s] && fits(s, need) && (pass == 1 || m_.slot_ctx(s) < big) && mem_ok(s, need, keep, sid, base, &mc, pr)) return s;
    int best = -1;
    uint64_t used = UINT64_MAX;
    for (int s = 0; s < ns; ++s) {
      if (!fits(s, need) || flying_.count(slot_[s]) || keep.count(slot_[s]) || !mem_ok(s, need, keep, sid, base, &mc)) continue;
      const auto it = sess_.find(slot_[s]);
      const uint64_t u = it == sess_.end() ? 0 : it->second.used;
      if (u < used) best = s, used = u;
    }
    if (best >= 0 || o_.slice <= 0) return best;
    for (auto it = rot_.rbegin(); it != rot_.rend(); ++it) {
      const auto x = sess_.find(it->rq.h.session_id);
      if (x != sess_.end() && x->second.slot >= 0 && !keep.count(x->first) && fits(x->second.slot, need) &&
          mem_ok(x->second.slot, need, keep, sid, base, &mc, pr))
        return x->second.slot;
    }
    return -1;
  }
  // whether a submit that waits may pause the requests decoding: it can have a slot
  bool admissible() const { return o_.slice > 0 || pick_slot({}) >= 0; }

  static constexpr uint32_t kPaused = ~0u, kTurn = ~0u - 1;  // a run's ends that are not the request's

  // A run: the requests at the front of the turns, as many as there are slots (at most a batch step's entries,
  // model::ops::kMaxSess; with more slots the rest take turns, --slice), each with its session live
  // in its slot (one that lost its slot resumes first, in a slot taken from a request further back), decode
  // together (batch_loop, a row of each per step) or one alone (the speculative loop, with drafts). Each runs
  // to its end, or until a new request that can have a slot pauses them all (prefill first), or, while
  // requests wait for a slot, for a turn of o_.slice tokens. Then they wait again: the paused ones first, in
  // their order, those whose turn is over last.
  void run() {
    std::vector<Flight> fl;
    std::set<uint64_t> keep;
    std::deque<Flight> later;  // big_slots: flights that need a big slot while every one is in this run
    // a run holds what one batch step takes (kMaxSess entries) and what the slots hold, whichever is less; the others
    // wait in rot_ and, with time-slicing, take their turn when this run's turns end (each turn goes to the back)
    const size_t cap = std::min<size_t>(slot_.size(), model::ops::kMaxSess);
    FacBusy busy_facs(fac_busy_);  // the factors of the flights taken out of the turns: their windows stay while this run builds
    int ext_waits = 0;             // flights of this pass that wait for the pool to extend their reservation
    turns::lead_other_factor(rot_, last_fac_, [](const Flight& f) { return f.fac; });  // the factors take turns
    while (fl.size() < cap && !rot_.empty()) {
      Flight f = std::move(rot_.front());
      rot_.pop_front();
      busy_facs.add(f.fac);
      // a step's rows all have one factor (the kernels rotate a step's rows by one table): the run takes the flights of
      // the first one's factor; the others wait in line for the next run (turns, --slice, make them alternate)
      if (!fl.empty() && f.fac != fl[0].fac) {
        later.push_back(std::move(f));
        continue;
      }
      if (f.fac > kBaseFactor) {  // a response that crossed: the window of its new factor first
        const Session* old = find(f.rq.h.session_id);
        const int w = ensure_window(f.fac, old ? std::vector<int>{old->h} : std::vector<int>{}, keep, f.rq.h.req_id);
        if (w == kWinWait) {
          later.push_back(std::move(f));
          std::this_thread::sleep_for(std::chrono::milliseconds(20));  // the others may be waiting for the same memory
          continue;
        }
        if (w == kWinRefuse) {
          ended(f, fe::kContext);  // the response ends at its window, as without Extend context
          continue;
        }
      }
      const Session* it = find_at(f.rq.h.session_id, f.fac);  // its session at its factor (none after a crossing)
      const int need = need_now(f);
      const bool live = it && it->slot >= 0 && fits(it->slot, need);
      if (!live && pick_slot(keep, need, f.rq.h.session_id, bkf(f)) < 0) {
        later.push_back(std::move(f));
        continue;
      }
      // a response within a quarter of --grow-ahead of the end of its reservation, in its own slot: the reservation goes a
      // horizon further when the pool can hold it; else it waits like a paused one (its slot's memory counts as idle
      // for the others: idle_id)
      if (live && f.upto < need && f.upto - (int)(f.rq.prompt.size() + f.out.size()) < o_.grow_ahead / 4) {
        const int s = it->slot;
        // the pool must hold it: first by the count (mem_ok), then in fact (the lines that count was based on let their
        // conversations go); a flight that waits is paused for the next pass, and an extender after it may detach its line
        if (!mem_ok(s, need, keep, f.rq.h.session_id, bkf(f)) || !room_for(s, need, keep, f.rq.h.session_id, bkf(f))) {
          ++ext_waits;
          later.push_back(std::move(f));
          continue;
        }
        state({{kStReserve, 0, s, need}});
        std::printf("serve: req %llu grows: its reservation reaches %d positions (was %d)\n", (unsigned long long)f.rq.h.req_id,
                    need, f.upto);
        f.upto = need;
      }
      f.paused_s += now_s() - f.t_paused;
      // a resume with more rows to read than a prompt read whole: read in pieces like a long prompt, between windows
      const int rows = !live && il_.on ? rows_to_compute(f, true) : 0;
      if (!live && il_.on && !il::read_whole(il_, rows)) {
        std::printf("serve: req %llu resumes in pieces: %d rows to read again\n", (unsigned long long)f.rq.h.req_id, rows);
        auto L = std::make_unique<Long>();
        L->f = std::move(f);
        L->resume = true;
        L->rows = rows;
        pfq_.push_back(std::move(L));
        continue;
      }
      if (!live && !go(f, true, keep)) continue;
      keep.insert(f.rq.h.session_id);
      fl.push_back(std::move(f));
    }
    for (auto it = later.rbegin(); it != later.rend(); ++it) rot_.push_front(std::move(*it));
    // every flight of two passes in a row waited for an extension: nothing can end, pause or free what they wait for
    // (--slice 0, whose flights are never paused, cannot break it; the production setting can)
    if (fl.empty() && ext_waits > 0 && (size_t)ext_waits == rot_.size() && ++ext_stall_ == 2)
      std::printf("serve: BUG: every flight waits for an extension (%d flights)\n", ext_waits);
    if (!fl.empty() || ext_waits == 0) ext_stall_ = 0;
    if (fl.empty()) return;
    last_fac_ = fl[0].fac;
    const bool turns = o_.slice > 0 && !rot_.empty();  // requests wait for a slot: a turn each
    std::vector<uint32_t> why(fl.size());
    if (fl.size() == 1) why[0] = decode_one(fl[0], turns);
    else decode_batch(fl, turns, why);
    std::vector<Flight> paused;
    for (size_t k = 0; k < fl.size(); ++k) {
      fl[k].t_paused = now_s();
      if (fl[k].cross) cross(fl[k]);
      if (why[k] == kPaused) paused.push_back(std::move(fl[k]));
      else if (why[k] == kTurn) rot_.push_back(std::move(fl[k]));
      else finish(fl[k], why[k]);
    }
    for (auto it = paused.rbegin(); it != paused.rend(); ++it) rot_.push_front(std::move(*it));
    dec_end_ = now_s();
  }

  // Session S's history for a run that feeds the flight's next token at pos: the positions before it (one
  // more, the token's own, when a run before fed it: it goes in again, with the same KV).
  void rejoin(Session& S, int pos, uint64_t req) {
    if ((int)S.hist.size() < pos)
      die("serve: req %llu: its session holds %zu positions, its next token goes at %d", (unsigned long long)req,
          S.hist.size(), pos);
    S.hist.resize(pos);
  }

  // Flight f's run alone, in its slot: the decode loop (speculative with drafts, unless --no-spec) from its
  // next token on. Its end (fe::k*), or kPaused / kTurn.
  uint32_t decode_one(Flight& f, bool turns) {
    const fe::SubmitHeader& h = f.rq.h;
    Session* S = find_at(h.session_id, f.fac);
    const int pos = (int)(f.rq.prompt.size() + f.out.size()) - 1, max_new = (int)h.max_new_tokens;
    const int left = max_new - (int)f.out.size();
    rejoin(*S, pos, h.req_id);
    Cmd c{};
    c.op = kCmdDecode;
    c.slot = S->slot;
    c.pos0 = pos;
    c.tokens[0] = f.out.back();
    c.sp = f.sp;
    c.max_new = std::min(turns ? std::min(left, o_.slice) : left, std::max(1, f.upto - pos));  // a turn; the horizon
    c.limit = std::min(m_.slot_ctx(S->slot), cap_of(f.fac));  // positions below the factor's window
    c.n_stop = (int)f.rq.stop.size();
    std::copy(f.rq.stop.begin(), f.rq.stop.end(), c.stop);
    c.ahead = 1;
    c.handle = S->h;
    c.snap_every = o_.snap_every;
    c.spec = o_.spec;
    c.spec_drafts = o_.spec_drafts;
    c.spec_thresh = o_.spec_thresh;
    std::copy(o_.spec_cost, o_.spec_cost + 3, c.spec_cost);
    c.spec_gate = o_.spec_gate;
    std::copy(o_.spec_row_ms, o_.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    c.spec_tok = f.spec_tok, c.spec_ms = f.spec_ms;  // the drafts' rate goes on from the last run
    room_for_bookmarks(reclaim::run_snaps(pos, c.max_new, o_.snap_every, model::ops::kDrafts + 2), {h.session_id}, h.session_id, bkf(f));
    const double t_dec = now_s();
    issue(c);
    std::vector<int32_t> samples;
    bool ctl = false, cancel = false, pause = false;
    const DecodeEnd e = decode_loop(m_, c, wd_, [&](int, const model::SampleOut& o, int row) {
      samples.push_back(o.row[row].token);
      if (f.tu.dec1_ms < 0) f.tu.dec1_ms = 1e3 * (now_s() - t_dec);
      if (cancel) return;
      if (Io::cancelled(f.rq)) {
        if (!ctl) m_.set_ctl(1);  // every rank stops after the step that reads it
        ctl = cancel = true;
        return;
      }
      tokens(f.rq, o, row);  // a paused request's last tokens too: the session holds them
      f.out.push_back(o.row[row].token);
      // a new request: this one pauses for its prefill; while long prompts are prefilled, at its window's end instead
      if (!ctl && (pfq_.empty() ? io_.submit_waiting() && admissible() : burst_over())) {
        m_.set_ctl(1);
        ctl = pause = true;
      }
    });
    if (ctl) m_.set_ctl(0);
    f.dec_s += now_s() - t_dec;
    f.decoded += e.sampled;
    f.spec_tok = e.spec_tok, f.spec_ms = e.spec_ms;
    add(f.de, e);
    ++f.slices;
    S->w = std::max(S->w, e.written);
    if (e.fed > 0) {  // the history: the token it started from, and the samples fed after it
      S->hist.push_back(c.tokens[0]);
      S->hist.insert(S->hist.end(), samples.begin(), samples.begin() + std::max(0, e.fed - 1));
    }
    // a pause's last token may be a stop id (the control word ends the loop before it checks them)
    return cancel || (e.why == kEndCancel && !pause) ? fe::kCancelled
           : e.why == kEndStop || stop_id(f, f.out.back()) ? fe::kStop
           : e.why == kEndContext                          ? window_end(f)
           : (int)f.out.size() >= max_new                  ? fe::kLength
           : pause                                         ? kPaused
                                                           : kTurn;
  }
  // A decode that stopped at its limit: at its factor's window the response goes on at the next factor (cross(), before the
  // flight's next run) when it may generate more and the engine's context holds that window; else the context reason
  uint32_t window_end(Flight& f) {
    if ((int64_t)f.rq.prompt.size() + (int64_t)f.out.size() < cap_of(f.fac) || !crossable(f)) return fe::kContext;
    f.cross = true;
    return kTurn;
  }

  // The flights' run together, each in its own slot: a row of each per step (batch_loop), or with drafts
  // (spec_batch_loop, B3) when the server has them and the batch is small enough for a draft each; with
  // drafts the last one left pauses, to go on alone. Each one's end (fe::k*), or kPaused / kTurn, in why.
  void decode_batch(std::vector<Flight>& fl, bool turns, std::vector<uint32_t>& why) {
    const int n = (int)fl.size();
    const bool drafts = o_.spec && m_.spec() && n <= m_.spec_batch_rows() / 2 && n <= m_.spec_batch_entries();
    std::vector<BatchEntry> e(n);
    std::vector<Session*> S(n);
    for (int k = 0; k < n; ++k) {
      const Flight& f = fl[k];
      S[k] = find_at(f.rq.h.session_id, f.fac);
      const int pos = (int)(f.rq.prompt.size() + f.out.size()) - 1;
      rejoin(*S[k], pos, f.rq.h.req_id);
      BatchEntry& x = e[k];
      x.slot = S[k]->slot, x.handle = S[k]->h, x.pos0 = pos, x.token = f.out.back();
      const int left = (int)f.rq.h.max_new_tokens - (int)f.out.size();
      x.max_new = std::min(turns ? std::min(left, o_.slice) : left, std::max(1, f.upto - pos));  // a turn; the horizon
      x.limit = std::min(m_.slot_ctx(S[k]->slot), cap_of(f.fac)), x.snap_every = o_.snap_every;
      x.n_stop = (int)f.rq.stop.size();
      std::copy(f.rq.stop.begin(), f.rq.stop.end(), x.stop);
      x.sp = f.sp;
      x.spec_tok = f.spec_tok, x.spec_ms = f.spec_ms;  // the drafts' rate goes on from the last run
    }
    Cmd c{};
    c.op = drafts ? kCmdSpecBatch : kCmdBatch;
    c.n = n;
    c.slot = e[0].slot;
    c.flags = o_.spec ? kBatchSolo : 0;
    if (drafts) {
      c.spec = 1;
      c.spec_drafts = o_.spec_drafts;
      c.spec_thresh = o_.spec_thresh;
      std::copy(o_.spec_cost, o_.spec_cost + 3, c.spec_cost);
      c.spec_gate = o_.spec_gate;
      std::copy(o_.spec_row_ms, o_.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    }
    {  // the run's bookmarks (one at each multiple of snap_every it crosses) need their numbers before the command
      int snaps = 0;
      std::set<uint64_t> run_ids;
      for (int k = 0; k < n; ++k) {
        snaps += reclaim::run_snaps(e[k].pos0, e[k].max_new, o_.snap_every, model::ops::kDrafts + 2);
        run_ids.insert(fl[k].rq.h.session_id);
      }
      room_for_bookmarks(snaps, run_ids, 0, 0);
    }
    const double t_dec = now_s();
    issue(c);
    bcast_payload(ring_, e.data(), e.size() * sizeof(BatchEntry), 0);
    uint32_t ctl = 0;
    std::vector<std::vector<int32_t>> samples(n);
    // ended: a request that ended in this run. Its DONE goes out at once, while the others go on: the front-end
    // completes a response only on DONE, and the run (whose end finishes the request's session) can last as long
    // as the longest of them. The session's next request waits for that end, as any request of a session in
    // flight does; its arrival pauses the run (below).
    std::vector<char> cancel(n, 0), ended(n, 0);
    const OnBatch cb = [&](int k, const model::SampleOut& o, int row) {
      Flight& f = fl[k];
      samples[k].push_back(o.row[row].token);
      if (f.tu.dec1_ms < 0) f.tu.dec1_ms = 1e3 * (now_s() - t_dec);
      if (cancel[k] || ended[k]) return;
      if (Io::cancelled(f.rq)) {  // its entry's bit: it ends after the step that reads it
        cancel[k] = 1;
        m_.set_ctl(ctl |= 1u << k);
        return;
      }
      const int32_t t = o.row[row].token;
      tokens(f.rq, o, row);  // a paused request's last tokens too: the session holds them
      f.out.push_back(t);
      // a new request pauses the batch when it can have a slot; one waiting for a slot does when this one
      // ends, which frees one (its session no longer in flight)
      const bool stop = stop_id(f, t), ends = stop || (int)f.out.size() >= (int)f.rq.h.max_new_tokens;
      if (ends) {
        ended[k] = 1;
        f.done_sent = true;
        done(f.rq, stop ? fe::kStop : fe::kLength, (uint32_t)f.q, (uint32_t)f.out.size(), f.t_start, f.t_first);
      }
      if (!(ctl & kCtlPause) && (pfq_.empty() ? (io_.submit_waiting() || !wait_.empty()) && (ends || admissible())
                                              : burst_over()))
        m_.set_ctl(ctl |= kCtlPause);
    };
    std::vector<BatchEnd> end(n);
    std::vector<DecodeEnd> dend;
    SpecBatchStats st;
    if (drafts) {
      dend = spec_batch_loop(m_, c, e, wd_, cb, &st);
      for (int k = 0; k < n; ++k) end[k].sampled = dend[k].sampled, end[k].fed = dend[k].fed, end[k].why = dend[k].why;
    } else {
      end = batch_loop(m_, c, e, wd_, cb);
    }
    if (ctl) m_.set_ctl(0);
    const double dt = now_s() - t_dec;
    std::string ends;
    for (int k = 0; k < n; ++k) {
      Flight& f = fl[k];
      const BatchEnd& x = end[k];
      f.dec_s += dt;
      f.decoded += x.sampled, f.batched += x.sampled;
      ++f.slices;
      if (drafts) {
        f.spec_tok = dend[k].spec_tok, f.spec_ms = dend[k].spec_ms;
        add(f.de, dend[k]);
        S[k]->w = std::max(S[k]->w, dend[k].written);
      } else {
        S[k]->w = std::max(S[k]->w, e[k].pos0 + x.fed);
      }
      if (x.fed > 0) {  // the history: the token it started from, and the samples fed after it
        S[k]->hist.push_back(e[k].token);
        S[k]->hist.insert(S[k]->hist.end(), samples[k].begin(), samples[k].begin() + std::max(0, x.fed - 1));
      }
      const int max_new = (int)f.rq.h.max_new_tokens;
      why[k] = cancel[k] || x.why == kEndCancel                ? fe::kCancelled
               : x.why == kEndStop || stop_id(f, f.out.back()) ? fe::kStop
               : x.why == kEndContext                          ? window_end(f)
               : (int)f.out.size() >= max_new                  ? fe::kLength
               : x.why == kEndPause                            ? kPaused
                                                               : kTurn;
      char b[64];
      std::snprintf(b, sizeof b, "%s%llu %d %s", k ? ", " : "", (unsigned long long)f.rq.h.req_id, x.sampled,
                    why[k] == kPaused ? "paused" : why[k] == kTurn ? "turn" : why[k] == fe::kStop ? "stop"
                    : why[k] == fe::kLength ? "length" : why[k] == fe::kCancelled ? "cancelled" : "context");
      ends += b;
    }
    // the run for the log: its steps' median (g_t_wait: when each step's samples came); with drafts, the
    // tokens and rows per step, the drafts accepted of those verified, and the steps' GPU ms by rows
    std::vector<double> d;
    for (size_t i = 1; i < g_t_wait.size(); ++i) d.push_back(1e3 * (g_t_wait[i] - g_t_wait[i - 1]));
    std::sort(d.begin(), d.end());
    std::string dr;
    if (drafts) {
      int ver = 0, acc = 0;
      for (int k = 0; k < n; ++k) ver += dend[k].verified, acc += dend[k].accepted;
      char b[160];
      std::snprintf(b, sizeof b, "; drafts: %.2f tokens and %.2f rows per step, %d of %d drafts accepted; GPU ms by rows:",
                    st.steps ? (double)st.emitted / st.steps : 0.0, st.steps ? (double)st.rows / st.steps : 0.0, acc, ver);
      dr = b;
      for (int r = 1; r <= model::kMaxRows; ++r)
        if (st.gpu_n[r]) {
          std::snprintf(b, sizeof b, " %d: %.2f (%d)", r, st.gpu_ms[r] / st.gpu_n[r], st.gpu_n[r]);
          dr += b;
        }
    }
    std::printf("serve: batch of %d: %zu steps in %.1f ms, %.2f ms a step (median); req, tokens, end: %s%s\n", n,
                g_t_wait.size(), 1e3 * dt, d.empty() ? 0.0 : d[d.size() / 2], ends.c_str(), dr.c_str());
  }

  // Paused requests that their client cancelled, or that the server's stop ends, end now, not at their
  // turn; so do requests waiting for a slot.
  // The dashboard's STATS frame (docs/dashboard.md), at most every half second, from this thread, the only one that
  // touches the model: the chunk pool and its conversations, the disk tier and its counters since the start, the
  // lines, interleaved prefill. The reader answers the front-end's StatsReq with the latest one.
  void publish() {
    const double t = now_s();
    if (t - published_ < 0.5) return;
    published_ = t;
    const model::Model::TierStats ts = m_.tier_stats();
    const model::Model::PoolStats ps = m_.pool_stats();
    fe::StatsFrame sf{};
    sf.version = fe::kStatsVersion, sf.n_fields = fe::kStatsFields;
    sf.pool_numbers = ps.numbers, sf.pool_free = ps.free, sf.pool_held = ps.held, sf.pool_shared = ps.shared;
    sf.pool_reserved = ps.reserved, sf.number_bytes = ps.number_bytes;
    sf.pool_positions = ps.numbers * 16384 / 5;  // a conversation takes 5 numbers every 16,384 positions (3 half-rate sources, layer 20)
    sf.conv_on_lines = ps.on_lines, sf.conv_in_memory = ps.in_memory, sf.conv_on_disk = ps.on_disk, sf.conv_sharing = ps.sharing;
    sf.lines = (uint64_t)m_.slots();
    sf.disk_cap_bytes = ts.disk_cap, sf.disk_used_bytes = ts.disk_used;
    sf.spills_total = ts.spills, sf.spill_bytes_total = ts.spill_bytes;
    sf.spills_pending = 0;  // a spill's writes are queued at once and waited for by the next reader; not counted
    sf.restores_total = ts.restores, sf.restore_bytes_total = ts.restore_bytes, sf.restored_tokens_total = ts.restored_pos;
    sf.restore_wait_ns_total = ts.wait_ns, sf.restore_wait_max_ns = ts.wait_max_ns;
    sf.il_on = il_.on, sf.il_whole_rows = (uint64_t)il_.whole_rows, sf.il_push_rows = (uint64_t)il_.push_rows;
    sf.il_gap_ms = (uint64_t)il_.gap_ms, sf.il_share = (uint64_t)il_.share, sf.il_burst_max_ms = (uint64_t)il_.burst_max_ms;
    uint64_t reading = 0;
    for (const auto& L : pfq_) reading += L->begun;
    sf.il_reading = reading, sf.il_queued = (uint64_t)pfq_.size() - reading;
    sf.keep_days = (uint64_t)(keep_age_s_ / 86400);
    sf.extend = extend_ ? 1 : 0;
    sf.ctx_limit = (uint64_t)ctx_limit(), sf.ranks = (uint64_t)model::kRanks;
    if (model::copy::mode() & model::copy::kAdapt) {
      sf.copy_adapt = 1, sf.copy_proposals_total = copy_proposals_total_, sf.copy_scored_total = copy_scored_total_;
      sf.copy_first_right_total = copy_first_right_total_, sf.copy_rows_verified_total = copy_rows_verified_total_;
      sf.copy_rows_accepted_total = copy_rows_accepted_total_, sf.copy_failing_requests_total = copy_failing_total_;
      for (int b = 0; b < 4; ++b) sf.copy_chance_pm[b] = ((uint64_t)copy_chance_[b] * 1000 + 32767) / 65535;
    }
    io_.publish_stats(Frame(fe::kStats).add(&sf).take());
  }
  double published_ = 0;

  void sweep() {
    // the requests that wait before they hold a slot (for one, for their session in flight, let go to run): a client that
    // left ends them now, whatever the others do
    cancel::sweep_queues(
        wait_, held_, ready_, [](const Request& r) { return Io::cancelled(r); },
        [&](const Request& r) {
          done(r, fe::kCancelled, 0, 0, now_s(), 0);
          Io::forget(r);
        });
    for (size_t i = 0; i < pfq_.size();) {  // a queued long prompt holds nothing: its cancel ends it now
      if (pfq_[i]->begun || !Io::cancelled(pfq_[i]->f.rq)) {
        ++i;
        continue;
      }
      end_queued(*pfq_[i], fe::kCancelled);
      pfq_.erase(pfq_.begin() + (std::ptrdiff_t)i);
    }
    for (size_t i = 0; i < rot_.size();) {
      if (!Io::cancelled(rot_[i].rq)) {
        ++i;
        continue;
      }
      Flight f = std::move(rot_[i]);
      rot_.erase(rot_.begin() + (std::ptrdiff_t)i);
      f.paused_s += now_s() - f.t_paused;
      finish(f, fe::kCancelled);
    }
  }

  // DONE, then the session kept (its snapshots thinned) or freed, and the store held to its budget; the
  // requests held for the session go on.
  void finish(Flight& f, uint32_t why) {
    const fe::SubmitHeader& h = f.rq.h;
    if (!f.done_sent) done(f.rq, why, (uint32_t)f.q, (uint32_t)f.out.size(), f.t_start, f.t_first);
    Io::forget(f.rq);
    const double t_end = now_s();
    flying_.erase(h.session_id);
    const bool released = later_.erase(h.session_id) > 0;  // the front-end let it go meanwhile
    forget_evicted();
    Session* S = (h.flags & fe::kFlagKeep) && !released ? find_at(h.session_id, f.fac) : nullptr;
    if (S) thin(*S);
    win_retry_at_ = 0;  // a window that waited for this request's memory asks again at once
    if (S && S->slot >= 0) {  // the turn saved where its line holds it: its full chunks get their images (write-behind)
      const int saved = m_.state_info(S->h).saved, E = (int)S->hist.size();
      Cmd c{};
      c.op = kCmdState;
      c.slot = S->slot;
      if (saved < E) c.ops[c.n_ops++] = {kStSave, S->h, saved, E};
      // the line stays attached, but the request is over: its reservation was a reservation for it, and nothing else
      if (reclaim::unreserve_at_finish()) c.ops[c.n_ops++] = {kStUnreserve, 0, S->slot, 0};
      if (c.n_ops > 0) {
        issue(c);
        if (reclaim::unreserve_at_finish()) digest("unreserve");
      }
    }
    const model::Model::StateInfo si = S ? m_.state_info(S->h) : model::Model::StateInfo{};
    const DecodeEnd* de = &f.de;
    if (de->vsteps) {
      std::printf("serve: req %llu drafts: %d verify steps, %.2f drafts verified and %.2f accepted per step, "
                  "steps verifying 0..5: %d %d %d %d %d %d; gate %.3f (%.3f-%.3f); GPU ms by rows:",
                  (unsigned long long)h.req_id, de->vsteps, (double)de->verified / de->vsteps,
                  (double)de->accepted / de->vsteps, de->rows_hist[0], de->rows_hist[1], de->rows_hist[2],
                  de->rows_hist[3], de->rows_hist[4], de->rows_hist[5], de->gate_sum / std::max(1, de->gate_n),
                  de->gate_min, de->gate_max);
      for (int r = 1; r <= model::ops::kDrafts + 1; ++r)
        if (de->gpu_n[r]) std::printf(" %d: %.2f (%d)", r, de->gpu_ms[r] / de->gpu_n[r], de->gpu_n[r]);
      std::printf("\n");
    }
    // the decode's time: from the first token to the end, less the time paused and resuming
    const double t_dec = f.t_first > 0 ? std::max(0.0, t_end - f.t_first - f.paused_s - f.resume_s) : 0;
    char sl[128] = "";  // (a request waits a moment in the turns between its first token and its decode)
    if (f.slices > 1 || f.paused_s >= 0.005 || f.batched)
      std::snprintf(sl, sizeof sl, "; %d slices, paused %.2f s, resumes %.1f ms, %d tokens in batches", f.slices,
                    f.paused_s, 1e3 * f.resume_s, f.batched);
    std::printf("serve: req %llu session %016llx: prompt %u, reused %d (%s), prefill %.1f ms (%.0f tok/s), %zu tokens "
                "in %.1f ms (%.1f tok/s), %s; store %.1f MiB; session %d positions, %.1f MiB, %zu window snapshots%s\n",
                (unsigned long long)h.req_id, (unsigned long long)h.session_id, h.n_prompt, f.q, kHow[f.pl.how],
                1e3 * (f.t_first > 0 ? f.t_first - f.t_start : 0.0),
                (h.n_prompt - f.q) / std::max(1e-9, f.t_first - f.t_start), f.out.size(), 1e3 * t_dec,
                f.decoded / std::max(1e-9, t_dec),
                why == fe::kStop ? "stop" : why == fe::kLength ? "length" : why == fe::kCancelled ? "cancelled" : "context",
                m_.state_bytes() / 1048576.0, S ? (int)S->hist.size() : 0, si.bytes / 1048576.0, si.snaps.size(), sl);
    const Turn& u = f.tu;
    std::printf("serve: req %llu turn: state ops %.1f ms (GPU %.1f), %d chunks: layers 0-19 %.1f ms, replay %.1f ms, "
                "head %.1f ms, draft windows %.1f ms, rows %.1f ms; first token to decode %.1f ms, first decode step "
                "%.1f ms\n",
                (unsigned long long)h.req_id, u.ops_ms, u.ops_gpu_ms, u.chunks, u.pf[0], u.pf[1], u.pf[2], u.pf[3],
                u.rows_ms, u.tail_ms, u.dec1_ms);
    if (u.jn_on)
      std::printf("serve: req %llu joins: %d chunks: attention %.1f ms, MoE %.1f ms (%d pairs), floor %.3f ms a pair\n",
                  (unsigned long long)h.req_id, u.jn_chunks, u.jn[0], u.jn[1], u.jn_pairs,
                  u.jn[2] / std::max(1, u.jn_chunks));
    {  // the chunk pool: the conversations no line holds, and what the lines hold and have reserved
      const model::Model::StoreChunks c = m_.state_chunks();
      int lines = 0;
      for (int t = 0; t < (int)slot_.size(); ++t) lines += m_.slot_committed(t);
      std::printf("serve: req %llu pool: %llu chunk numbers; stored sessions hold %llu (%.1f MiB), the lines %d with their "
                  "reservations, %d free and unreserved\n",
                  (unsigned long long)h.req_id, (unsigned long long)c.total, (unsigned long long)c.mapped,
                  c.live_bytes / 1048576.0, lines, m_.slot_uncommitted());
    }
    if (!S) release(h.session_id);
    for (auto it = held_.begin(); it != held_.end();)  // the session's held requests go on, in order
      if (it->h.session_id == h.session_id) ready_.push_back(std::move(*it)), it = held_.erase(it);
      else ++it;
    evict(h.session_id);
  }

  Session* find(uint64_t id) {
    const auto it = sess_.find(id);
    return it == sess_.end() ? nullptr : &it->second;
  }

  // ---- the factor in every match key. A conversation's state is held at one factor under the front-end's session id
  // (the primary) and at other factors under keys of its own (alt_: (id, factor) -> key, `owner` set). Requests and fork
  // bases are looked up at the request's factor only, so a state of another factor is never reused; a request at
  // another factor starts a new session (cold) and leaves the old one intact for its siblings.
  // The session of conversation `id` at `factor`, primary or alternate; *key: the entry's key in sess_ (set when found).
  Session* find_at(uint64_t id, int factor, uint64_t* key = nullptr) {
    if (!id) return nullptr;
    const auto it = sess_.find(id);
    if (it != sess_.end() && it->second.owner == 0 && it->second.factor == factor) {  // a primary, never an alternate's key
      if (key) *key = id;
      return &it->second;
    }
    const auto a = alt_.find({id, factor});
    if (a == alt_.end()) return nullptr;
    const auto j = sess_.find(a->second);
    if (j == sess_.end() || j->second.owner != id || j->second.factor != factor) {  // evicted or freed meanwhile
      alt_.erase(a);
      return nullptr;
    }
    if (key) *key = a->second;
    return &j->second;
  }
  // The key a request's base is known by for the pins and holds (the base at the request's factor; its id when none)
  uint64_t bk(const fe::SubmitHeader& h, int factor) {
    uint64_t k = h.base_session_id;
    if (k) find_at(k, factor, &k);
    return k;
  }
  uint64_t bkr(const Request& rq) { return bk(rq.h, factor_for((int64_t)rq.prompt.size())); }
  uint64_t bkf(const Flight& f) { return bk(f.rq.h, f.fac); }
  // Positions a request at `factor` can hold: its window, and never more than the engine's context
  int cap_of(int factor) const { return std::min(window_of(factor), ctx_limit()); }
  // The context this server serves now: --max-ctx with Extend context on, else at most the base factor's window
  int ctx_limit() const { return extend_ ? o_.max_ctx : std::min(o_.max_ctx, window_of(kBaseFactor)); }

  // The primary of conversation id leaves the id: it becomes an alternate under a key of its own (the node moves, so
  // pointers to it stay valid), its slot's owner key follows. The id is then free for a session of another factor.
  void demote(uint64_t id, int to = 0) {
    auto node = sess_.extract(id);
    if (node.empty()) return;
    uint64_t k;
    do k = (1ull << 63) | ++alt_next_;
    while (sess_.count(k));
    Session& x = node.mapped();
    x.owner = id;
    if (x.slot >= 0) slot_[x.slot] = k;
    alt_[{id, x.factor}] = k;
    node.key() = k;
    Session& y = sess_.insert(std::move(node)).position->second;
    // the conversation goes on at a larger factor (to): the copy it leaves is only for siblings under its window, so with
    // a disk tier it leaves the pool at once (every piece written, its numbers given back) and comes back if one asks
    if (y.factor < to && m_.state_disk_capacity() > 0 && y.len() > 0) {
      if (y.slot >= 0) persist_slot(y.slot, true);
      std::printf("serve: conversation %016llx goes on past its factor-%d window: that copy (%zu positions) to the disk tier\n",
                  (unsigned long long)id, y.factor, y.len());
      state({{kStSpill, y.h, 0, 0}});
      forget_evicted();
    }
  }
  // The session a request of conversation id runs in at `factor`, as its primary: the primary if it is at that factor,
  // else the alternate at it (the two trade places), else none (the primary of another factor becomes an alternate; the
  // caller makes the new one). Only for a request that is about to run: no other request of the id is in flight.
  Session* acquire(uint64_t id, int factor) {
    uint64_t k = 0;
    Session* x = find_at(id, factor, &k);
    if (x && k == id) return x;
    demote(id, factor);
    if (!x) return nullptr;
    auto node = sess_.extract(k);
    alt_.erase({id, factor});
    node.mapped().owner = 0;
    if (node.mapped().slot >= 0) slot_[node.mapped().slot] = id;
    node.key() = id;
    return &sess_.insert(std::move(node)).position->second;
  }
  // ---- the windows of factors 32 and 64 (Model::window_open): made out of the pool when a request first needs one,
  // freed when none has for --window-idle-s. A command like any other: every rank makes or frees the same.
  enum { kWinReady, kWinWait, kWinRefuse };
  static int win_level(int factor) { return factor == 32 ? 1 : 2; }
  // The factors of the requests the server has in its hands but in no queue (being started, or taken out of the turns for
  // the run being built): a bit per win_level. Their windows are not closed for want of memory.
  struct FacBusy {
    unsigned& m;
    const unsigned old;
    explicit FacBusy(unsigned& x) : m(x), old(x) {}
    void add(int factor) {
      if (factor > kBaseFactor) m |= 1u << win_level(factor);
    }
    ~FacBusy() { m = old; }
  };
  unsigned fac_busy_ = 0;
  int ext_stall_ = 0;  // passes in a row in which every flight waited for an extension (run)
  // Whether a request of `factor` exists anywhere: its window is needed
  bool factor_in_use(int factor) const {
    if (fac_busy_ & (1u << win_level(factor))) return true;
    for (const Flight& f : rot_)
      if (f.fac == factor) return true;
    for (const auto& L : pfq_)
      if (L->f.fac == factor) return true;
    for (const std::deque<Request>* d : {&wait_, &held_, &ready_})
      for (const Request& r : *d)
        if (factor_for((int64_t)r.prompt.size()) == factor) return true;
    return false;
  }
  // The window to give back when a consumer is short and no line has anything left: the least recently used open one that
  // no request needs (a paused flight is in rot_, so its window is never closed under it). Closes it, settles what the
  // close could not give back at once, and says whether a window was closed (the caller asks the pool again).
  bool close_unused_window() {
    if (!m_.window_lazy()) return false;
    std::vector<reclaim::Win> w;
    for (const int f : {32, 64}) w.push_back({f, m_.window_is_open(f), factor_in_use(f), win_used_[win_level(f)]});
    const int f = reclaim::window_to_close(w);
    if (!f) return false;
    Cmd c{};
    c.op = kCmdWindow, c.rows = f, c.pos0 = 1;
    wd_.started = now_s();
    issue(c);
    run_window(m_, c);
    wd_.started = 0;
    digest("window close");
    std::printf("serve: the window of factor %d freed for a request that needs the memory (%d groups of the pool's numbers still owed)\n", f,
                m_.window_owed());
    settle_windows();
    return true;
  }
  // What a closed window could not give back at once (a rank's device memory) is asked for again
  void settle_windows() {
    if (m_.window_owed() <= 0) return;
    win_settled_ = now_s();
    Cmd c{};
    c.op = kCmdWindow, c.rows = 0;
    wd_.started = now_s();
    issue(c);
    run_window(m_, c);
    wd_.started = 0;
    digest("window settle");
  }
  // Whether a request that waits is worth trying again now: a window it needs was refused for a moment ago
  bool window_ready(const Request& rq) const {
    const int fac = factor_for((int64_t)rq.prompt.size());
    return fac <= kBaseFactor || m_.window_is_open(fac) || now_s() >= win_retry_at_;
  }
  // Factor `fac`'s window open (made now when it is not). `own`: the handles of the request's own conversation and its base,
  // which the window takes last, like the rest of the cache; `keep`: the sessions of the run being built. The window is made
  // from a plan: Model::window_plan names the sessions that hold a number the window takes; those on idle or paused lines are
  // detached (they stay in the pool as stored conversations the model evicts piece by piece), then the pool gives back what is
  // still short (give_back), then one command makes the window. kWinWait: running requests hold what it needs (the log names
  // them, and no command goes out while they do) or a rank had too little device memory for a moment; finish() and cross()
  // end the wait. kWinRefuse: the pool cannot hold one conversation of that window, or the ranks failed three times in a row.
  int ensure_window(int fac, const std::vector<int>& own, const std::set<uint64_t>& keep, uint64_t req) {
    if (fac <= kBaseFactor) return kWinReady;
    if (m_.window_is_open(fac)) {
      win_used_[win_level(fac)] = now_s();
      return kWinReady;
    }
    if (now_s() < win_retry_at_) return kWinWait;
    FacBusy busy_fac(fac_busy_);  // the window being made is not one to close
    busy_fac.add(fac);
    int g = m_.window_groups_estimate(fac);
    for (int attempt = 0; attempt < 3; ++attempt) {
      const model::Model::WindowPlan plan = m_.window_plan(fac, g);
      std::vector<reclaim::Holder> holders;
      for (const model::Model::WindowHolder& hd : plan.holders) {
        uint64_t key = 0;
        const Session* x = by_handle(hd.handle, &key);
        if (x) holders.push_back({key, x->slot});  // a session on no line is the model's to evict
      }
      const reclaim::WindowActions act =
          reclaim::window_actions(holders, [&](uint64_t id) { return idle_id(id, keep, 0, 0); });
      if (!act.running.empty()) return window_wait(req, fac, act.running);  // no command while a running request holds it
      for (const int t : act.detach) {
        std::printf("serve: slot %d (session %016llx) gives its %d chunks back to the pool for the window of factor %d\n", t,
                    (unsigned long long)slot_[t], m_.slot_committed(t), fac);
        persist_slot(t, true);  // the own conversation and its base too: the window evicts them last (kStLast)
      }
      // what is still short after the holders left: free and unreserved numbers, stored conversations, idle lines' memory
      if (!give_back(plan.removed, keep, 0, 0, -1, (int64_t)m_.state_bookmark_numbers(1), "a window"))
        return window_wait(req, fac, running_holders());
      Cmd c{};
      c.op = kCmdWindow;
      c.rows = fac;
      for (const int h : own)
        if (c.n_ops < kMaxOps) c.ops[c.n_ops++] = StateOp{kStLast, h, 0, 0};
      wd_.started = now_s();
      issue(c);
      const int st = run_window(m_, c);
      wd_.started = 0;
      digest("window");
      forget_evicted();
      if (st == model::Model::kWindowOk) {
        win_fails_ = 0;
        win_used_[win_level(fac)] = now_s();
        return kWinReady;
      }
      if (st == model::Model::kWindowNoRoom) {
        std::printf("serve: req %llu needs the window of factor %d (%d positions), and the pool cannot hold one conversation of it\n",
                    (unsigned long long)req, fac, window_of(fac));
        return kWinRefuse;
      }
      if (st == model::Model::kWindowBusy) {
        ++lend_busy_total_;
        // the exact count of groups (the ranks' largest) may exceed this rank's estimate: plan again with it
        const int exact = m_.window_busy_groups(fac);
        if (exact > g) {
          g = exact;
          continue;
        }
        std::printf("serve: BUG: window waits on cache: req %llu, factor %d, %d groups, no running holder\n",
                    (unsigned long long)req, fac, g);
        return window_wait(req, fac, {});
      }
      if (st == model::Model::kWindowFailed && ++win_fails_ >= 3) {
        std::printf("serve: req %llu: the window of factor %d failed %d times in a row for want of device memory\n",
                    (unsigned long long)req, fac, win_fails_);
        win_fails_ = 0;
        return kWinRefuse;
      }
      win_retry_at_ = now_s() + 3.0;
      if (now_s() - win_note_ > 10) {
        win_note_ = now_s();
        std::printf("serve: req %llu waits for the window of factor %d (a rank had too little device memory)\n",
                    (unsigned long long)req, fac);
      }
      return kWinWait;
    }
    return window_wait(req, fac, {});
  }
  // A window that cannot be made now: no command goes out until the retry (a second on, or when a request ends or crosses:
  // finish(), cross()). `running`: the requests whose memory it waits for, in the log
  int window_wait(uint64_t req, int fac, const std::vector<uint64_t>& running) {
    win_retry_at_ = now_s() + 1.0;
    if (now_s() - win_note_ > 10) {
      win_note_ = now_s();
      std::string ids;
      for (const uint64_t id : running) {
        char b[24];
        std::snprintf(b, sizeof b, "%s%016llx", ids.empty() ? "" : ", ", (unsigned long long)id);
        ids += b;
      }
      std::printf("serve: req %llu waits for the window of factor %d (%s%s)\n", (unsigned long long)req, fac,
                  running.empty() ? "running requests hold the pool's memory it needs" : "running requests use the memory it needs: sessions ",
                  ids.c_str());
    }
    return kWinWait;
  }
  // The sessions of the running requests whose lines hold or are reserved numbers (what a consumer that is short waits for)
  std::vector<uint64_t> running_holders() const {
    std::vector<uint64_t> v;
    for (int t = 0; t < (int)slot_.size(); ++t)
      if (slot_[t] && m_.slot_committed(t) > 0 && reclaim::running(flying_.count(slot_[t]) > 0, paused(slot_[t]))) v.push_back(slot_[t]);
    return v;
  }
  // The session with store handle h and its key in sess_; null when the server holds none
  const Session* by_handle(int h, uint64_t* key = nullptr) const {
    for (const auto& [k, x] : sess_)
      if (x.h == h) {
        if (key) *key = k;
        return &x;
      }
    return nullptr;
  }
  // The windows of factors nobody asked to free are kept: nothing is freed by time unless --window-idle-s turned it on
  // (never by default); what a close could not yet give back is asked for again every few seconds. A window a request of
  // its factor exists for is never freed by time. Closing for memory is give_back's (close_unused_window).
  void windows_idle() {
    if (!m_.window_lazy() || !flying_.empty() || !rot_.empty() || !pfq_.empty()) return;
    const double t = now_s();
    for (const int f : {32, 64}) {
      if (!m_.window_is_open(f) || !reclaim::timer_may_close(o_.window_idle_s, t, win_used_[win_level(f)], factor_in_use(f))) continue;
      Cmd c{};
      c.op = kCmdWindow, c.rows = f, c.pos0 = 1;
      wd_.started = now_s();
      issue(c);
      run_window(m_, c);
      wd_.started = 0;
      digest("window close");
      std::printf("serve: the window of factor %d was idle %d s: freed (%d groups of the pool's numbers still owed)\n", f,
                  (int)(t - win_used_[win_level(f)]), m_.window_owed());
    }
    if (m_.window_owed() > 0 && t - win_settled_ >= 5) settle_windows();
  }

  // A flight whose response reached its window goes on at the next factor in a session of its own made cold by its resume
  // (the prompt and the response so far, read again at the new factor); the session it leaves holds the window's worth
  // of the conversation, unchanged, for its siblings, and gives up its slot at once.
  void cross(Flight& f) {
    const int to = factor_for((int64_t)f.rq.prompt.size() + (int64_t)f.out.size());
    std::printf("serve: req %llu crosses from factor %d to %d at %zu positions: its prompt and reply so far are read again at the new factor\n",
                (unsigned long long)f.rq.h.req_id, f.fac, to, f.rq.prompt.size() + f.out.size());
    const Session* old = find(f.rq.h.session_id);
    if (old && old->slot >= 0) persist_slot(old->slot, true);  // its state to the store, its slot's memory to the pool
    demote(f.rq.h.session_id, to);
    win_retry_at_ = 0;  // a window that waited for this request's memory asks again at once
    f.fac = to;
    f.cross = false;
    f.upto = 0;
  }
  // Whether a decode that hit its window ends here (the context reason) or goes on at the next factor
  bool crossable(const Flight& f) const {
    const int64_t at = (int64_t)f.rq.prompt.size() + (int64_t)f.out.size();
    return f.fac < 64 && at < ctx_limit() && window_of(f.fac) < ctx_limit() && (int)f.out.size() < (int)f.rq.h.max_new_tokens;
  }

  model::Model& m_;
  comm::Ring& ring_;
  Watchdog& wd_;
  const ServeOptions& o_;
  double keep_age_s_;  // --keep-days, changed live (CONFIG keep_days)
  bool extend_;  // --extend, changed live (CONFIG extend): a prompt is admitted, and a response crosses its window, under the setting of that moment
  int64_t ssd_pending_ = 0;  // a disk tier size the dashboard asked for (GiB), applied when nothing runs (resize_idle)
  double ckpt_at_ = 0;      // when the last checkpoint was written (checkpoint())
  uint64_t ckpt_ver_ = 0;   // the store's version it saw (Model::state_version)
  Io& io_;
  const size_t budget_;  // the store's bytes kept between requests (the rest: room for the next one)
  std::map<uint64_t, Session> sess_;
  std::map<std::pair<uint64_t, int>, uint64_t> alt_;  // (front-end session id, factor) -> the key of its alternate in sess_
  uint64_t alt_next_ = 0;
  std::vector<uint64_t> slot_;  // the session whose working state each slot holds (0: none)
  std::deque<Flight> rot_;      // requests between their first token and their end, in turn
  int last_fac_ = 0;            // the factor of the last run's rows (turns::lead_other_factor alternates them)
  // Interleaved prefill: long prompts, first in first out; the front is read once it holds a slot, a second one
  // (push_) may be read beside it, the rest hold nothing. A Long lives on the heap: its Reader points at its Flight.
  struct Long {
    Flight f;
    Reader r;
    bool begun = false;
    bool resume = false;  // a flight that lost its slot, re-reading its prompt and the tokens it sent
    int rows = 0;         // rows to compute when it was queued (the second reader: at most push_rows)
  };
  // The positions a queued long prompt needs a slot for: a first run's prompt and horizon, a resume's from where it is
  int need_long(const Long& L) const { return L.resume ? need_now(L.f) : need_clamped(L.f.rq); }
  Long* push_ = nullptr;  // the second reader, a short long prompt read beside the front one, and first

  // A long prompt out of the queue (ended, or read and decoding); the second reader that is now the front is the front
  void drop_long(Long* x) {
    for (auto it = pfq_.begin(); it != pfq_.end(); ++it)
      if (it->get() == x) {
        pfq_.erase(it);
        break;
      }
    if (push_ == x || (push_ && !pfq_.empty() && pfq_.front().get() == push_)) push_ = nullptr;
  }
  // A long prompt that ends in the queue, never read: a first run's time was all queue time (DONE queue_us)
  void end_queued(Long& L, uint32_t why) {
    if (!L.resume) L.f.t_start = now_s();
    finish(L.f, why);
  }
  std::deque<std::unique_ptr<Long>> pfq_;
  il::Settings il_;             // interleaved prefill's settings: the flags', then what CONFIG frames set (apply_config)
  il::Cost il_cost_;
  double owed_ms_ = 0;         // decode time owed to the decoders for the prompt work done while they waited
  double burst_ms_ = 0;        // the window of the decode run in progress (0: no window)
  double burst_end_ = 0;       // its end, set at the run's first step
  bool il_admitted_ = false;   // the scheduler's last unit was an admission (il::State::admitted_last)
  double dec_end_ = 0;         // when the last decode run ended: decoders owed for the time since (a long prompt's arrival)
  std::deque<Request> held_, ready_, wait_;  // submits for a session in flight; those let go when it ended;
                                             // those waiting for a slot
  std::set<uint64_t> flying_, later_;  // the sessions in flight; RELEASEs of them, applied at their end
  double win_used_[3] = {};   // when each factor's window was last used (index: win_level)
  double win_retry_at_ = 0;   // a window refused for lack of memory is not asked for again before this
  double win_note_ = 0, win_settled_ = 0;
  int win_fails_ = 0;
  int next_h_ = 1;
  uint64_t clock_ = 0;
  std::mt19937_64 rng_{std::random_device{}()};
};

}  // namespace

int serve(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o) {
  if (o.prefill_rows / o.snap_every > kMaxSnaps) die("serve: more than %d snapshots per prefill chunk", kMaxSnaps);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  fe::HelloEngine hi{};
  hi.version = fe::kVersion;
  hi.max_context = (uint32_t)o.max_ctx;
  hi.vocab_size = model::kVocab;
  hi.eos_id = 1;
  // between requests the conversations keep at most the pool less room for a long request (1/16 of it, at least 256
  // MiB); a request that needs more evicts the least recently used (state.cu store_evict_step)
  const size_t cap = m.state_ceiling(), room = std::max<size_t>(256ull << 20, cap / 16);
  const size_t budget = cap > room ? cap - room : 0;
  hi.max_sessions = (uint32_t)std::max<size_t>(1, budget / (64ull << 20));
  {  // the positions the engine can keep: the pool's and the disk tier's (a number's image there too), 5 numbers every 16,384
    const model::Model::PoolStats ps = m.pool_stats();
    const uint64_t numbers = ps.numbers + (ps.number_bytes ? m.tier_stats().disk_cap / ps.number_bytes : 0);
    hi.keep_positions = numbers * 16384 / 5;
  }
  hi.name_len = (uint32_t)o.name.size();
  Io io(o.socket, Frame(fe::kHelloEngine).add(&hi).add(o.name.data(), o.name.size()).take());
  std::printf("serve: listening on %s (context %d, session store %.2f GiB, kept between requests %.2f GiB)\n",
              o.socket.c_str(), o.max_ctx, m.state_capacity() / 1073741824.0, budget / 1073741824.0);
  Server s(m, ring, wd, o, io, budget);
  s.load_kept();
  s.loop();
  if (g_prof) CUDA_CHECK(cudaProfilerStop());
  std::printf("serve: stopped\n");
  return 0;
}

}  // namespace srv
