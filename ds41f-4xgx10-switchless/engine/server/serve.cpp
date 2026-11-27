// Rank 0 as a server (serve.h, docs/protocol.md).
//
// Threads: the reader takes the front-end's frames (a new connection replaces the old one, whose
// requests then count as cancelled); the writer sends HELLO / TOKENS / DONE, so the engine never
// blocks on the socket; the engine (the caller's thread) runs the requests, every model call mirrored on
// the other ranks by a command (cmd.h).
//
// Scheduling (docs/batching.md). A request's first run prefills its prompt into a session slot
// and sends its first token; then it decodes with the others, as many as there are slots, a row each
// per step (batch_loop), or alone through the speculative loop (with drafts). A request that comes while
// others decode pauses them (prefill first), if it can have a slot: a free one, one whose session no
// request uses, or, with time-slicing (--slice), the slot of the paused request whose turn comes last.
// While more requests decode than there are slots, they take turns of --slice tokens. With one slot,
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
#include "dspark/dspark.h"
#include "server/serve.h"
#include "server/bounds.h"
#include "server/cancel.h"
#include "server/img_hold.h"
#include "server/reclaim.h"
#include "server/rows_loan.h"
#include "server/split_pace.h"
#include "server/turns.h"
#include "server/vit_sched.h"

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
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
#include "model/copy.h"
#include "model/lend.h"
#include "server/lend_pace.h"
#include "model/nvtx.h"
#include "model/pool_rules.h"
#include "common/frontend_protocol.h"
#include "common/keepfile.h"
#include "server/ids_file.h"
#include "server/frames.h"
#include "server/mem_watch.h"
#include "server/pixels.h"
#include "vision/vit_rank.h"
#include "server/prefix.h"

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
  std::atomic<uint32_t> version{fe::kVersion};  // what its HELLO said (fe::kVersionPrefix: the prefix frames); fe::kVersion until it says
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
  // kFlagPrefix: `prompt` is the suffix until admit builds the whole prompt from the conversation's history (built); the epoch and
  // the count the frame named are used once
  uint64_t epoch = 0;
  uint32_t n_prefix = 0;
  bool built = true;
  // positions of the whole prompt, built or not
  size_t len() const { return prompt.size() + (built ? 0 : n_prefix); }
  double t_recv = 0;
  // vision (fe::kFlagImages): the image spans and their rows, bf16 [sum rows][kImageDim]; empty for text
  std::vector<fe::ImageSpan> spans;
  std::vector<uint16_t> img;
  // vision (fe::kFlagPixels): the frame's payload, held until the request ends (the images' bytes at px.off); `refuse`: why a
  // well-formed pixel frame is answered and not run (empty: nothing wrong with the frame)
  pixels::Parsed px;
  std::vector<char> pix;
  std::string refuse;
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
  Io(const std::string& path, std::vector<char> hello, uint64_t frame_cap) : path_(path), hello_(std::move(hello)), frame_cap_(frame_cap) {
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
  bool config_pending() {  // a CONFIG frame is waiting for the main loop
    std::lock_guard<std::mutex> l(cmu_);
    return cfg_new_;
  }
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

  // A frame larger than --frame-stage-mib is not read into memory: a SUBMIT's request id is read, the rest of the frame's bytes are
  // skipped through a small buffer, and the request ends with DONE kTooLarge naming the limit; any other frame is skipped. The
  // socket stays up: the next frame is read as usual. false only when the skip cannot be read (the connection went).
  bool too_large(const ConnP& c, const fe::FrameHeader& h) {
    uint64_t req = 0;
    uint64_t left = h.size;
    const bool submit = h.type == fe::kSubmit && left >= sizeof req;
    if (submit) {
      if (!read_all(c->fd, &req, sizeof req)) return false;
      left -= sizeof req;
    }
    std::vector<char> skip(64 << 10);
    while (left > 0) {
      const size_t n = (size_t)std::min<uint64_t>(left, skip.size());
      if (!read_all(c->fd, skip.data(), n)) return false;
      left -= n;
    }
    char why[160];
    std::snprintf(why, sizeof why, "the request is %llu MiB, over the %llu MiB the engine reads in one frame (--frame-stage-mib)",
                  (unsigned long long)(h.size >> 20), (unsigned long long)(frame_cap_ >> 20));
    std::printf("serve: %s frame of %llu bytes not read: %s\n", submit ? "request" : "a", (unsigned long long)h.size, why);
    if (submit) {
      fe::DoneHeader d{};
      d.req_id = req;
      d.reason = fe::kTooLarge;
      d.error_len = (uint32_t)std::strlen(why);
      send(c, Frame(fe::kDone).add(&d).add(why, d.error_len).take());
    }
    return true;
  }

  // One frame from c: false when the connection is to be closed.
  bool frame(const ConnP& c) {
    fe::FrameHeader h;
    if (!read_all(c->fd, &h, sizeof h) || h.size > kMaxFrame) return false;
    if (frame_cap_ && h.size > frame_cap_) return too_large(c, h);
    std::vector<char> p(h.size);
    if (!p.empty() && !read_all(c->fd, p.data(), p.size())) return false;
    if (h.type == fe::kHello) {
      fe::HelloFe hi;
      if (p.size() < sizeof hi) return false;
      std::memcpy(&hi, p.data(), sizeof hi);
      if (!frames::version_ok(hi.version)) {
        std::fprintf(stderr, "serve: front-end speaks version %u, not %u or %u\n", hi.version, fe::kVersion, fe::kVersionPrefix);
        return false;
      }
      c->version = hi.version;
      if (hi.version != fe::kVersion) {  // the engine's HELLO went out at the connect in fe::kVersion: this one answers the connection's
        std::vector<char> h2 = hello_;
        std::memcpy(h2.data() + sizeof(fe::FrameHeader), &hi.version, sizeof hi.version);
        send(c, std::move(h2));
      }
    } else if (h.type == fe::kSubmit) {
      Event e;
      Request& r = e.rq;
      fe::SubmitPrefix px;
      size_t base = 0;
      if (!frames::parse_submit_head(p.data(), p.size(), c->version, &r.h, &px, &base)) return false;  // malformed: the connection closes
      const bool pfx = r.h.flags & fe::kFlagPrefix;
      r.stop.resize(r.h.n_stop_ids);
      r.prompt.resize(r.h.n_prompt);
      const char* a = p.data() + sizeof r.h;
      if (pfx) {  // the suffix only: the epoch and the count of the history ids the prompt begins with
        a += sizeof px;
        r.epoch = px.epoch, r.n_prefix = px.n_prefix, r.built = false;
      }
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
      if (r.h.flags & fe::kFlagPixels) {  // vision: the images as pixels (pixels.h)
        pixels::Result pr = pixels::parse(p.data(), p.size(), base, r.h.flags);
        if (pr.v == pixels::Verdict::kMalformed) return false;
        r.spans = std::move(pr.px.spans);
        if (pr.v == pixels::Verdict::kRefused) r.refuse = std::move(pr.why);
        else r.px.grids = std::move(pr.px.grids), r.px.off = std::move(pr.px.off), r.pix = std::move(p);
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
  uint64_t frame_cap_ = 0;  // bytes; a frame over it is skipped (too_large); 0: none
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
  std::vector<int32_t> hist;   // the tokens whose state the session holds: positions [0, size); empty while ids_out
  bool ids_out = false;        // the ids are in the ids file (<ids_dir>/<h>.ids), not in `hist`: Server::need_ids reads them back
  size_t ids_n = 0;            // their count while ids_out
  size_t len() const { return ids_out ? ids_n : hist.size(); }  // positions the history holds, wherever the ids are
  prefix::Epoch ep;            // the epoch of the history and the size the last DONE reported (prefix.h)
  // The ids file (<ids_dir>/<h>.<seq>.ids, ids_file.h's Keeper): `file` names the newest (seq 0: none); `file_ok` it holds exactly the
  // history now (ids_out implies it). A run on the session makes the file stale (file_ok false); it stays, as the last kept index may
  // name it, and the conversation's next idle ids go to a new file.
  idsfile::Keeper::Ref file;
  bool file_ok = false;
  uint64_t used = 0;           // LRU clock
  int pe_snap = -1;            // the snapshot at its last prompt's end (when not a multiple of snap_every)
  int lo = 0, clo = 0, w = 0;  // the live state's exact ranges (the comment at the top)
  int end_lo = 0, end_clo = 0, end_w = 0;  // the end snapshot's
  // CED: the decoder layers' rings hold valid positions from dec_lo on (kNever: none), the end
  // snapshot's from end_dec_lo on; window snapshots holding the encoder layers' rings only
  int dec_lo = 0, end_dec_lo = 0;
  std::vector<int> enc_snaps;
  double t_used = 0;  // unix time of its last request (kept sessions age out by it)
  double t_fin = 0;   // now_s() when its last request finished: a split's idle age counts from it (0: from the server's start)
  uint64_t flushed = 0;  // the LRU clock at its last flush to the disk tier (flush_idle)
};

// The server's index of a planned stop's kept sessions (keep(); keepfile.h: 0600 in a 0700 directory,
// whole or not at all): a header, then each session (prefix.h's KeptRec): id, handle, restart ranges, LRU clock and time,
// factor and owner, epoch, the count and sum of its ids file (the ids are in that file, not here), encoder-only snapshots.
constexpr char kSrvMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'S', '3'};  // format 3: each session holds the epoch and the ids file's name (the ids are in that file)
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
  // the join clock (DSGB10_PF_STALL): the chunks' joins summed; jn_on says the switch is on
  bool jn_on = false;
  float jn[3] = {};  // attention joins, MoE joins, the pairs' own cost summed over the chunks
  int jn_pairs = 0, jn_chunks = 0;
  void add_split(const model::Model& m) {
    float f[4];
    m.prefill_split(f);
    for (int i = 0; i < 4; ++i) pf[i] += f[i];
    float j[3];
    int np = 0;
    if (m.prefill_joins(j, &np)) {
      jn_on = true;
      for (int i = 0; i < 3; ++i) jn[i] += j[i];
      jn_pairs += np, ++jn_chunks;
    }
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
  imghold::Hold img;                 // its image rows in the store: the key the ranks hold them under, and how many rows
  std::vector<char> enc;             // a pixel request: which spans' rows are written (the images encoded so far)
  std::string err;                   // the error its DONE carries when it ends in one
};

// b's drafts and steps added to a's (a request's decode slices)
void add(DecodeEnd& a, const DecodeEnd& b) {
  a.vsteps += b.vsteps, a.verified += b.verified, a.accepted += b.accepted;
  a.copy_proposals += b.copy_proposals, a.copy_scored += b.copy_scored, a.copy_first_right += b.copy_first_right;
  a.copy_rows_verified += b.copy_rows_verified, a.copy_rows_accepted += b.copy_rows_accepted;
  if (b.copy_rho256) a.copy_rho256 = b.copy_rho256;
  if (b.copy_chance[0] | b.copy_chance[1] | b.copy_chance[2] | b.copy_chance[3]) std::copy(b.copy_chance, b.copy_chance + 4, a.copy_chance);
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
  // What a request's return of its conversation (or its fork base) asks: which conversation, up to where, how many pieces of each
  // kind, and what it is estimated to cost (the time ahead of the rows)
  struct JoinAsk {
    bool found = false, deferred = false;  // deferred: the commands go out as the request's first units, between the decoders' windows
    int h = -1, upto = 0, back = 0, exch = 0, disk = 0;
    uint64_t key = 0;
    double est_ms = 0;
  };
  Server(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o, Io& io, size_t budget)
      : m_(m), ring_(ring), wd_(wd), o_(o), keep_age_s_(o.keep_age_s), extend_(o.extend), io_(io), budget_(budget), slot_(m.slots(), 0), il_(o.interleave) {
    if (o.kv_split || o.imager) {  // the marks, in numbers: the room off keeps between requests (max(256 MiB, 1/16 of the pool)) is the high mark
      const size_t cap = m.state_ceiling(), room = std::max<size_t>(256ull << 20, cap / 16);
      const model::Model::PoolStats ps = m.pool_stats();
      init_split_marks(ps.number_bytes ? (int64_t)(room / ps.number_bytes) : 0);
      if (o.imager) {
        std::printf("serve: kv split %s: a conversation's blocks are written to each Spark's SSD as quarters in the background (idle: %d pieces a command; "
                    "in a run: %d pieces every %d ms%s) and then %s; a return takes them back when nobody used them, else reads the quarters (--ssd-flush-s is the least time between two checkpoints while nothing runs; it starts no write of its own)\n",
                    o.kv_split_mode == 0 ? "off" : o.kv_split_mode == 1 ? "tiered" : "max", o.kv_split_budget, o.kv_split_ride_pieces, o.kv_split_ride_ms,
                    o.kv_split_mode == 2 ? "" : (", only under the high mark of " + std::to_string((long long)split_marks_.high)).c_str(),
                    o.kv_split_mode == 0 ? "evicted clean, writing nothing, when memory is wanted" : "marked free where they lie");
      } else
      std::printf("serve: kv split %s: demote at %.0f s idle with the spare numbers at the high mark of %lld, down to 0 s at the low mark of %lld "
                  "(under it the split conversations go to the SSD first), a pass at most %d ms with requests in flight, %d pieces a command\n",
                  o.kv_split_mode == 1 ? "tiered" : "max", o.kv_split_after_s, (long long)split_marks_.high, (long long)split_marks_.low,
                  o.kv_split_pass_ms, o.kv_split_budget);
    }
    engram_set_mib_ = engram_goal_mib_ = o.engram_mib;
    loan_mib_ = m.prefill_loan_mib(), loan_groups_ = m.prefill_loan_groups();
    loan_goal_ = loan_mib_ > 0 && m.tenant_group_count(model::lend::kTenantPrefill) > 0 ? 1 : -1;
    if (rowsloan::start_backoff(loan_mib_ > 0, m.tenant_group_count(model::lend::kTenantPrefill)))
      loan_backoff_.fail((uint64_t)loan_mib_, now_s(), loan_gen_);  // a start that made no set does not ask again at once
    // a start that made less than was asked for (cut, or the hand-over failed) is not tried again at once
    if (m.engram_lend_enabled() && (int64_t)m.engram_lend_info().mib != engram_goal_mib_)
      engram_backoff_.fail((uint64_t)engram_goal_mib_, now_s(), engram_gen_);
  }

  // Requests let go after their session's last one ended first, then those that waited for a slot (when
  // one is there), then the front-end's frames, then a run of the requests decoding (run).
  void loop() {
    while (!g_stop) {
      fe::Config cf;
      if (io_.take_config(cf)) apply_config(cf);
      lend_tick();
      split_tick();
      watch_tick();
      if (pfq_.empty()) publish();  // a prompt being prefilled publishes between its chunks' launch and wait (pf_piece), not between two chunks
      sweep();
      vit_rotate_wait();
      if (!pfq_.empty()) {  // interleaved prefill: long prompts and decoders take turns
        il_pass();
        continue;
      }
      Event e;
      if (!ready_.empty()) {
        Request rq = std::move(ready_.front());
        ready_.pop_front();
        admit(std::move(rq));
      } else if (!wait_.empty() && start_wait_.may_try(now_s()) && !vit_gated(wait_.front()) && window_ready(wait_.front()) &&
                 pick_slot({}, need_clamped(wait_.front()), wait_.front().h.session_id, bkr(wait_.front()), &wait_.front().prompt) >= 0) {
        Request rq = std::move(wait_.front());
        wait_.pop_front();
        admit(std::move(rq), true);
      } else if (!wait_.empty() && !vit_gated(wait_.front()) && flying_.empty() && pfq_.empty() && rot_.empty() && ready_.empty() && wait_.front().h.req_id != rescued_req_ &&
                 window_ready(wait_.front()) && !imager_work()) {
        rescued_req_ = wait_.front().h.req_id;  // once: a request that goes back to the queue (its window) is not tried again at once
        Request rq = std::move(wait_.front());  // nothing runs: what keeps it from a slot never goes by waiting (admit: rescue)
        wait_.pop_front();
        admit(std::move(rq), true);
      } else if (io_.pop(e, rot_.empty() ? 250 : 0)) {
        if (e.release) {
          if (flying_.count(e.session)) later_.insert(e.session);
          else release(e.session);
        } else if (!e.rq.conn->dead) {  // a gone front-end's requests need no answer
          admit(std::move(e.rq));
        }
      } else if (!rot_.empty()) {
        const double t_run = now_s();
        run();
        owed_ms_ = splitpace::owed_left(owed_ms_, 1e3 * (now_s() - t_run), false);  // a run with no long prompt queued settles the debt too
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
    for (int t = 0; t < (int)slot_.size(); ++t) v.push_back({m_.slot_committed(t), m_.slot_mapped(t), m_.slot_unheld_reservation(t)});  // the reservation: a split line's quarter slots are committed, not reserved
    const int64_t n = reclaim::idle_reserves(v);
    if (n != 0 && n != idle_reserved_) std::printf("serve: BUG: idle line holds a reservation (%lld numbers reserved, nothing in flight)\n", (long long)n);
    idle_reserved_ = n;
    // and no image rows: a number held with nothing in flight means a path ended a flight without freeing its rows
    if (imghold::tripwire(flying_.size() + rot_.size() + pfq_.size(), m_.image_numbers())) {
      std::printf("serve: BUG: %zu image numbers held with nothing in flight; freed\n", m_.image_numbers());
      state({{kStImgFree, 0, 0, 0}});
      enc_mark_.clear();
    }
    if (wait_.empty() && held_.empty() && ready_.empty()) img_ask_.clear();  // an ask whose request is gone
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
        prefix::KeptRec k;
        if (!prefix::get_rec(r, &k, o_.max_ctx, 4096)) break;
        const uint64_t id = k.id;
        Session s;
        s.h = k.h, s.pe_snap = k.pe_snap;
        s.end_lo = k.end_lo, s.end_clo = k.end_clo, s.end_w = k.end_w, s.end_dec_lo = k.end_dec_lo;
        s.used = k.used, s.t_used = k.t_used, s.factor = k.factor, s.owner = k.owner;
        s.ep.value = k.epoch ? k.epoch : new_epoch(), s.ep.reported = k.reported;
        s.enc_snaps.assign(k.enc_snaps.begin(), k.enc_snaps.end());
        const int nh = k.ids_n;
        bool ok = id && !kept.count(id) && s.h > 0 && wall - s.t_used <= keep_age_s_;
        // the factor is a window the engine can hold, the same one the ranks' store recorded, and an entry that answers
        // to another id is not that id's own
        ok = ok && (s.factor == 16 || s.factor == 32 || s.factor == 64) && (s.factor == 16 || window_of(s.factor) / 2 < o_.max_ctx) && s.owner != id;
        // the ids file the index names holds the count and the sum it recorded (the file's header; the ids are read back, and
        // checked whole, when a request uses the conversation)
        if (ok && nh > 0) {
          uint64_t c = 0, sm = 0;
          std::string w;
          ok = !o_.ids_dir.empty() && k.ids_seq && idsfile::peek(o_.ids_dir, s.h, &c, &sm, &w, k.ids_seq) && c == (uint64_t)nh && sm == k.ids_sum;
          if (ok) s.ids_out = true, s.ids_n = (size_t)nh, s.file_ok = true, s.file = {k.ids_seq, sm};
        }
        const model::Model::StateInfo si = m_.state_info(s.h);
        ok = ok && si.on_disk && si.saved <= nh && si.end_pos <= nh && m_.state_factor(s.h) == s.factor;
        for (const auto& [kk, x] : kept) ok = ok && x.h != s.h;
        s.lo = s.clo = s.w = s.dec_lo = 0;  // nothing is live
        if (ok) {
          ids_files_ += s.ids_out;
          if (s.file.seq) keeper_.adopt(s.h, s.file.seq);
          kept.emplace(id, std::move(s));
        }
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
    for (const auto& [id, s] : kept) positions += s.len();
    if (!kept.empty()) clock_ = std::max<uint64_t>(clock_, hd.clock);
    sess_.insert(std::make_move_iterator(kept.begin()), std::make_move_iterator(kept.end()));
    for (const auto& [id, x] : sess_)
      if (x.owner) alt_[{x.owner, x.factor}] = id;
    if (!why.empty() || !sess_.empty() || !v.empty())
      std::printf("serve: kept sessions: %zu back (%zu positions, %.2f GiB on disk), %zu of the store's dropped, in %.1f ms%s%s\n",
                  sess_.size(), positions, m_.state_disk_bytes() / 1073741824.0, v.size(), 1e3 * (now_s() - t0),
                  why.empty() ? "" : "; ", why.c_str());
  }

  // After load_kept: the ids files the kept index did not name go (the directory is ours alone).
  void sweep_ids() {
    if (o_.ids_dir.empty()) return;
    std::string why;
    if (!idsfile::sweep(o_.ids_dir, keeper_.names(), &why)) die("serve: the ids directory: %s", why.c_str());
  }

 private:
  // A new epoch for a history entry: random, never 0
  uint64_t new_epoch() {
    uint64_t e;
    do e = rng_();
    while (!e);
    return e;
  }
  // Every rank: c's state operations, then c (rank 0's part of c.op is the caller's). c carries the
  // sessions of the slots that a running request uses, which the store keeps (pinned) and whose images it keeps in the
  // disk tier's file; a conversation on an idle or paused line is cache like any other.
  // A split command (`split`) names live every session whose request is in flight, those on no line too: false, nothing sent, when
  // they do not all fit in the command's list.
  bool issue(Cmd& c, bool split = false) {
    for (size_t s = 0; s < slot_.size(); ++s) {
      const auto it = slot_[s] ? sess_.find(slot_[s]) : sess_.end();
      const bool live = split ? reclaim::split_names_live(flying_.count(slot_[s]) > 0, paused(slot_[s])) : reclaim::counts_as_live(flying_.count(slot_[s]) > 0, paused(slot_[s]));
      c.live[s] = it == sess_.end() || !live ? 0 : it->second.h;
    }
    if (split) {
      for (const auto& [k, x] : sess_) {
        if (!in_flight(k, x) || std::find(c.live, c.live + model::kMaxSlots, x.h) != c.live + model::kMaxSlots) continue;
        int* e = std::find(c.live, c.live + model::kMaxSlots, 0);
        if (e == c.live + model::kMaxSlots) return false;
        *e = x.h;
      }
    }
    bcast(ring_, c);
    wd_.started = now_s();  // a state command is watched as a step is (a rank that waits in a state operation for a partner ends with the watchdog's line)
    apply_ops(m_, c);
    wd_.started = 0;
    return true;
  }
  // A request is in flight on this conversation: its own id or the id it answers to, admitted and not yet finished.
  bool in_flight(uint64_t key, const Session& x) const { return flying_.count(key) > 0 || (x.owner && flying_.count(x.owner) > 0); }
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
        drop_ids(it->second);
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
        drop_ids(sess_.at(id));
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
  // handles hs names, each with its id, handle, restart ranges, LRU clock and time, epoch, the count and sum of its ids file,
  // encoder-only snapshots. The ids are not in it: a conversation whose file does not hold its history now has it written
  // first (the conversations that went idle since the last index; the rest have their file), so an index is KiBs.
  bool write_index(uint64_t gen, const std::set<int>& hs, int* n, size_t* bytes, std::string* why) {
    std::vector<uint8_t> f(sizeof(SrvHeader));
    *n = 0;
    std::vector<uint64_t> lost;  // conversations whose ids file is lost: not written, freed after
    std::map<int, uint64_t> named;  // handle -> the ids file this index names
    for (auto& [id, s] : sess_) {
      if (!hs.count(s.h)) continue;
      prefix::KeptRec k;
      if (s.ids_out) {
        uint64_t c = 0, sm = 0;
        std::string w;
        const bool ok = idsfile::peek(o_.ids_dir, s.h, &c, &sm, &w, s.file.seq) && c == s.ids_n && sm == s.file.sum;
        if (idsfile::back_verdict(ok, c, s.ids_n) == idsfile::Back::kFree) {
          std::printf("serve: conversation %016llx: its ids file %s; it is not written to the index and is freed\n", (unsigned long long)id,
                      w.empty() ? "is gone or holds another count" : w.c_str());
          lost.push_back(id);
          continue;
        }
      } else if (!s.hist.empty() && !s.file_ok) {
        std::string w;
        if (!ids_file_now(s, &w)) {
          std::printf("serve: conversation %016llx: its ids not written (%s); it is not in the index\n", (unsigned long long)id, w.c_str());
          continue;
        }
      }
      ++*n;
      k.id = id, k.h = s.h, k.pe_snap = s.pe_snap;
      k.end_lo = s.end_lo, k.end_clo = s.end_clo, k.end_w = s.end_w, k.end_dec_lo = s.end_dec_lo;
      k.used = s.used, k.t_used = s.t_used, k.factor = s.factor, k.owner = s.owner;
      k.ids_n = (int32_t)s.len(), k.ids_sum = k.ids_n ? s.file.sum : 0, k.ids_seq = k.ids_n ? s.file.seq : 0;
      if (k.ids_n) named[s.h] = s.file.seq;
      k.epoch = s.ep.value, k.reported = std::min<uint64_t>(s.ep.reported, (uint64_t)k.ids_n);
      k.enc_snaps.assign(s.enc_snaps.begin(), s.enc_snaps.end());
      prefix::put_rec(f, k);
    }
    SrvHeader hd{};
    std::memcpy(hd.magic, kSrvMagic, sizeof kSrvMagic);
    hd.gen = gen, hd.written = (int64_t)unix_now(), hd.n_sess = *n, hd.next_h = next_h_, hd.clock = clock_;
    hd.body_bytes = f.size() - sizeof hd;
    hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
    std::memcpy(f.data(), &hd, sizeof hd);
    *bytes = f.size();
    const bool wrote = keepfile::write_atomic(o_.keep_index, f.data(), f.size(), why);
    if (wrote) keeper_.committed(named);  // the files this index does not name are not needed by any index now
    for (const uint64_t id : lost) free_session(id);
    return wrote;
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
          if (!o_.imager) fl.push_back({kStFlush, x->h, 0, 0});
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
    if (g_log_detail || !ok)  // a checkpoint not written is always said
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
    ids_to_files();
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
      if (!o_.imager && o_.ssd_flush_s > 0 && S.flushed != S.used && now - S.t_used >= o_.ssd_flush_s) {  // the imager is the write-behind: nothing to flush
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
    if (free && x) split_note(x->h);  // it lost its line: its end bookmark was taken whole, its pieces wait for their split
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
      drop_ids(it->second);
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
          drop_ids(it->second);
          sess_.erase(it);
          break;
        }
    if (!v.empty()) state(v);  // a live session's slot memory goes back with it
  }
  // ---- the server's token ids past their cap (ids_file.h; --ids-host-mib, --disk-dir) ----
  // The ids of every conversation in memory, bytes (the rest are in the ids files).
  uint64_t ids_bytes() const {
    uint64_t b = 0;
    for (const auto& [id, s] : sess_) b += (uint64_t)s.hist.size() * idsfile::kIdBytes;
    return b;
  }
  // When the ids in memory pass the cap, the least recently used idle conversations' ids go to their files and are freed. Only a
  // conversation on no line and in no flight goes: nothing reads the ids of a conversation a request is not using, and a request's
  // use starts with unspill, which reads them back.
  void ids_to_files() {
    if (o_.ids_dir.empty()) return;
    std::vector<idsfile::Cand> c;
    std::map<int, Session*> by_h;
    for (auto& [id, s] : sess_) {
      if (s.ids_out || s.slot >= 0 || flying_.count(id) || s.hist.empty()) continue;
      c.push_back({s.h, s.used, s.hist.size()});
      by_h[s.h] = &s;
    }
    const uint64_t total = ids_bytes();
    for (const int h : idsfile::pick_out(std::move(c), total, o_.ids_host_bytes)) {
      Session& s = *by_h.at(h);
      std::string why;
      if (!ids_file_now(s, &why)) {
        std::printf("serve: the ids of session handle %d not written (%s); they stay in memory\n", s.h, why.c_str());
        return;
      }
      s.ids_n = s.hist.size(), s.ids_out = true;
      std::vector<int32_t>().swap(s.hist);
      ++ids_files_;
    }
  }
  // The file of `s` holding its history as it is now: written unless it is (a file of an older history is replaced).
  bool ids_file_now(Session& s, std::string* why) {
    if (s.file_ok) return true;
    if (o_.ids_dir.empty()) return *why = "no ids directory", false;
    idsfile::Keeper::Ref r;
    if (!keeper_.put(s.h, s.hist, &r, why)) return false;  // a new file: the one the last index names is left as it is
    s.file = r, s.file_ok = true;
    return true;
  }
  // The ids of `x` back in memory when they are out. A file that is gone or does not check out leaves the history empty: the
  // conversation is computed again from its prompt, and the line says so.
  bool need_ids(uint64_t key, Session& x) {
    if (!x.ids_out) return true;
    std::vector<int32_t> v;
    std::string why;
    const bool ok = idsfile::read(o_.ids_dir, x.h, &v, &why, x.file.seq);
    if (idsfile::back_verdict(ok, v.size(), x.ids_n) == idsfile::Back::kUse) {
      std::printf("serve: session %016llx: its ids read back from their file (%zu positions)\n", (unsigned long long)key, v.size());
      x.hist = std::move(v);
      x.ids_out = false, x.ids_n = 0;  // the file stays: it holds the history until a run changes it (the kept index may name it)
      --ids_files_;
      return true;
    }
    std::printf("serve: session %016llx: its ids file %s; the conversation is freed\n", (unsigned long long)key, why.empty() ? "is gone or holds another count" : why.c_str());
    free_session(key);
    return false;
  }
  // A conversation freed whole by the ordinary path: its state in the store, its line (if any), its ids file, its entry.
  void free_session(uint64_t key) {
    const auto it = sess_.find(key);
    if (it == sess_.end()) return;
    std::vector<StateOp> v{{kStFree, it->second.h, 0, 0}};
    if (it->second.slot >= 0) v.push_back({kStRelease, 0, it->second.slot, 0}), slot_[it->second.slot] = 0;
    state(v);
    drop_ids(it->second);
    sess_.erase(it);
  }
  // The conversation is freed: its ids files go (at once, or with the next index, which does not name them).
  void drop_ids(Session& x) {
    if (x.ids_out) --ids_files_;
    keeper_.drop(x.h);
    x.ids_out = false, x.ids_n = 0, x.file = {}, x.file_ok = false;
  }
  int64_t ids_files_ = 0;  // conversations whose ids are in a file now

  // Least recently used sessions out until the store fits the budget: spilled to the disk tier (one live
  // in a slot saved first, and the slot let go), or freed without one. Sessions in flight stay.
  void evict(uint64_t keep) {
    const bool disk = m_.state_disk_capacity() > 0;
    if ((o_.kv_split || o_.imager) && disk) return;  // a take evicts as the core does (clean pieces, writing nothing); no spill, which would wait for the image stream, on a request's path
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
                  (unsigned long long)victim->first, b / 1048576.0, victim->second.len());
      bytes -= std::min(bytes, b);
      if (disk) {
        if (victim->second.slot >= 0) persist_slot(victim->second.slot, true);  // what it holds live goes with it
        v.push_back({kStSpill, victim->second.h, 0, 0});
        out.push_back(victim->first);
        continue;
      }
      v.push_back({kStFree, victim->second.h, 0, 0});
      if (victim->second.slot >= 0) v.push_back({kStRelease, 0, victim->second.slot, 0}), slot_[victim->second.slot] = 0;
      drop_ids(victim->second);
      sess_.erase(victim);
    }
    if (!v.empty()) state(v);
    forget_evicted();  // dropped from the disk tier to make room
  }
  // A request's session (or a fork's base) that was spilled comes back to the store before its plan
  // sees it. The slots' sessions stay where they are (every command pins them).
  // The read-back has a budget (kStUnspill's count): the lists that must come back, then what the admission found spare for
  // bookmarks (pool_rules.h restore_budget), so that it leaves the numbers the request's reservation takes next (it is made
  // after this): a request admitted on its need finds it. `need` and `keep` are the request's, as mem_ok is asked.
  void unspill(const fe::SubmitHeader& h, int fac, int need, const std::set<uint64_t>& keep, const std::vector<int32_t>& prompt, bool join_now = true) {
    uint64_t id = h.session_id;
    Session* x = find_at(id, fac, &id);
    if (!x && h.base_session_id) id = h.base_session_id, x = find_at(id, fac, &id);
    if (!x) return;
    if (!need_ids(id, *x)) return;  // the ids of a conversation a request uses are in memory before its plan reads them (a lost file frees it)
    const uint64_t base = bk(h, fac);
    split_forget(x->h);  // a request names it: it is not split meanwhile
    const model::Model::StateInfo si = m_.state_info(x->h);
    join_incomplete_ = false;
    if (si.split && join_now) {  // its quarter slots come back first: whole numbers before anything reads or writes them
      const RestoreAsk ra = restore_ask(h.session_id, base, need, &prompt);
      const bool mine = ra.found && ra.src_id == id;
      int64_t budget = m_.state_info(x->h).slots;  // not the request's conversation (a stop's spill): every slot, a number each
      if (mine) {
        // the count of this join (a read-back's rule): the lists that must come back, then what the admission found spare for
        // bookmarks (its own bookmarks' numbers included), read once in the pass that makes the reservation; each command carries what is left, never 0 (0 is no limit)
        MemCache mc;
        const int s = x->slot >= 0 ? x->slot : pick_slot(keep, need, h.session_id, base, &prompt);
        int64_t have = 0, want = 0;
        mem_sums(s >= 0 ? s : 0, need, keep, h.session_id, base, &mc, &have, &want, &prompt);
        if (s < 0) have = want;
        budget = model::rules::join_budget(ra.split, ra.bookmarks, have, want);
        give_back((int)std::min<int64_t>(budget, (int64_t)ra.split + ra.bookmarks) + ra.on_disk, {}, h.session_id, base, -1, 0, "a conversation joined", false);
      }
      join_back(*x, mine ? ra.upto : INT_MAX, id, budget);
      join_incomplete_ = mine && lists_split(x->h, ra.upto) > 0;  // the request is not begun on a conversation still split below `upto`
    }
    if (!join_now) return;  // a deferred return's units read the SSD's lists too (join_unit); its bookmarks and the rest are this call's, from pf_begin
    if (!si.on_disk) return;
    const double t0 = now_s();
    int64_t cap = 0, room = (int64_t)m_.state_restore_chunks(x->h), asked = 0;  // 0: no limit; asked: the lists the count said can come back
    int upto = 0;           // > 0: the read-back is commands of at most J pieces below this position (--kv-split cache); 0: one command, as with --kv-split off
    bool nothing_below = false;  // a fork at position 0 reads nothing: its base's pieces all lie above it
    {
      MemCache mc;
      const int s = x->slot >= 0 ? x->slot : pick_slot(keep, need, h.session_id, base, &prompt);
      int64_t have = 0, want = 0;
      mem_sums(s >= 0 ? s : 0, need, keep, h.session_id, base, &mc, &have, &want, &prompt);
      const int back = mc.ask.on_disk;  // the lists below the need (a fork: below its fork point) that must come back
      asked = back;
      if (s < 0) have = want;           // no slot to count against: the lists only
      cap = model::rules::restore_budget(back, have, want);
      if (cap > 0) room = std::min(room, cap);
      if (o_.kv_split) {  // no command of a return handles more than J pieces: the request's own point, or all of it for a conversation no request names
        const bool mine = mc.ask.found && mc.ask.src_id == id;
        upto = mine ? mc.ask.upto : INT_MAX;
        if (upto <= 0) upto = 0, nothing_below = true;
      }
    }
    // room for what comes back: idle lines let their conversations go first (they stay in the pool, now evictable), as
    // room_for does for a request's own need, so that the read-back does not find the pool held by idle lines. What
    // running requests hold is left: the read-back takes what fits (and never what a running request was reserved)
    if (o_.imager && upto > 0 && m_.state_unspill_left(x->h, upto) == 0) return;  // nothing at or below the request's point is only on the SSD: the request asks once, not again for a tail it does not read
    give_back(room, {}, h.session_id, base, -1, 0, "a conversation read back", false);
    int d_in = 0, d_disk = 0;
    m_.state_list_numbers(x->h, INT_MAX, &d_in, &d_disk);
    const uint64_t sum_cuts0 = m_.state_sum_cuts();
    const double t_rb = now_s();
    int commands = 0;
    if (upto > 0) {  // units of at most J pieces; each carries what is left of the count (the numbers it may take), and the last one of a
                     // spent count says so (a < 0) so that the core cuts where the count ended, as one command did
      const uint64_t taken0 = m_.split_stats().unspill_taken;
      int64_t left = cap;
      for (; commands < 100000;) {
        const int todo = m_.state_unspill_left(x->h, upto);
        if (todo == 0) break;
        const int32_t a = cap > 0 ? (left > 0 ? (int32_t)std::min<int64_t>(left, 1 << 30) : -1) : 0;
        state({{kStUnspill, x->h, a, upto}});
        ++commands;
        m_.drain();
        left = cap - (int64_t)(m_.split_stats().unspill_taken - taken0);
        if (a < 0 || m_.state_unspill_left(x->h, upto) >= todo) break;  // the count ran out (and was cut), or nothing moved
      }
    } else if (!nothing_below) {
      state({{kStUnspill, x->h, (int32_t)cap, 0}});
      m_.drain();
      commands = 1;
    }
    {  // the count paid for these lists: a read-back that is refused is a count that disagrees with the core's (rules::spare)
      int a_in = 0, a_disk = 0;
      m_.state_list_numbers(x->h, INT_MAX, &a_in, &a_disk);
      jc_.learn_disk(1e3 * (now_s() - t_rb), std::max(0, a_in - d_in));  // the per-piece time of a read from the SSD, the lists that came back
      const int64_t want_back = std::min<int64_t>(asked, d_disk);
      if (model::rules::readback_refused(cap, want_back, std::max(0, a_in - d_in), m_.state_sum_cuts() != sum_cuts0))
        std::printf("serve: BUG: a read-back the count paid for was refused (budget %lld, %lld lists asked, %d back)\n", (long long)cap,
                    (long long)want_back, std::max(0, a_in - d_in));
    }
    if (commands > 1) std::printf("serve: session %016llx read back in %d commands of at most %d pieces\n", (unsigned long long)id, commands, o_.kv_join_pieces);
    const model::Model::StateInfo back = m_.state_info(x->h);  // what came back: the room decides how much
    std::printf("serve: session %016llx back from disk: %.1f MiB of %.1f MiB, %zu positions, restart point %d of %d, in %.1f ms%s\n",
                (unsigned long long)id, back.bytes / 1048576.0, si.disk_bytes / 1048576.0, x->len(), back.restart,
                si.restart, 1e3 * (now_s() - t0), cap > 0 ? (" (budget " + std::to_string(cap) + " numbers)").c_str() : "");
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
    note_token();
    for (int k = 0; k < nt; ++k) {
      const fe::TopLogprob x{(uint32_t)o.top_id[row][k], o.top_lp[row][k]};
      f.add(&x);
    }
    io_.send(rq.conn, f.take());
  }

  // `tail`: the history entry's epoch and size (a request that ran); none: no DoneTail in the frame
  void done(const Request& rq, uint32_t reason, uint32_t cached, uint32_t sent, double t_start, double t_first,
            std::string err = "", const fe::DoneTail* tail = nullptr) {
    // cut short by the server's stop, an answer is an error to its client, not a finished one
    if (reason == fe::kCancelled && Io::stopping(rq)) reason = fe::kError, err = "the server is stopping";
    const double t_end = now_s();
    if (t_first <= 0) t_first = t_end;
    img_ask_.ended(rq.h.session_id, rq.h.req_id);
    fe::DoneHeader d{};
    d.req_id = rq.h.req_id;
    d.reason = reason;
    d.prompt_tokens = (uint32_t)rq.len();
    d.cached_tokens = cached;
    d.completion_tokens = sent;
    d.queue_us = us(t_start - rq.t_recv);
    d.prefill_us = us(t_first - t_start);
    d.decode_us = us(t_end - t_first);
    d.error_len = (uint32_t)err.size();
    io_.send(rq.conn, frames::done_frame(rq.conn->version, d, tail, err));
    if (!err.empty()) std::printf("serve: request %llu: %s\n", (unsigned long long)rq.h.req_id, err.c_str());
  }

  // kFlagPrefix: the request's prompt built from the history entry its epoch names (prefix.h), once. False when it cannot be (DONE
  // kPrefixGone: nothing is done, and the front-end sends the whole prompt).
  bool build_prompt(Request& rq) {
    const prefix::Ask a{rq.h.session_id, rq.h.base_session_id, rq.epoch, rq.n_prefix};
    const int fac = factor_for((int64_t)rq.n_prefix + (int64_t)rq.prompt.size());
    std::vector<int32_t> whole;
    const prefix::Gone g = prefix::build<Session>(
        a, fac, rq.prompt, &whole, [&](uint64_t id, int f, uint64_t* key) { return find_at(id, f, key); },
        [&](uint64_t key, Session& x) { return need_ids(key, x); });
    if (g != prefix::Gone::kNone) {
      done(rq, fe::kPrefixGone, 0, 0, now_s(), 0, prefix::gone_text(g));
      Io::forget(rq);
      return false;
    }
    if (g_log_detail)
      std::printf("serve: req %llu: a prompt of %zu ids: %u of the history of %016llx and %zu of its own\n", (unsigned long long)rq.h.req_id,
                whole.size(), rq.n_prefix, (unsigned long long)(rq.h.base_session_id ? rq.h.base_session_id : rq.h.session_id), rq.prompt.size());
    rq.prompt = std::move(whole);
    rq.h.n_prompt = (uint32_t)rq.prompt.size();  // DONE and the log count the whole prompt
    rq.built = true;
    return true;
  }

  // A submit: held while its session is in flight (a session's requests run one at a time); waiting while
  // no slot can be had (every slot's session in flight, and no time-slicing); else started.
  uint64_t rescued_req_ = 0;  // the waiting request the idle engine last took off the queue
  // `front`: the request comes off the head of the waiting queue (nothing waits ahead of it).
  void admit(Request&& rq, bool front = false) {
    if (Io::cancelled(rq)) {  // its client left (or the server stops) while it queued: nothing is made for it
      done(rq, fe::kCancelled, 0, 0, now_s(), 0);
      return Io::forget(rq);
    }
    if (rq.h.flags & fe::kFlagPixels) {  // pixels.h: a frame the engine will not run, or a request the encoder cannot take
      const std::string why = pixel_refusal(rq);
      if (!why.empty()) {
        done(rq, fe::kError, 0, 0, now_s(), 0, why);
        return Io::forget(rq);
      }
    }
    if (flying_.count(rq.h.session_id)) return held_.push_back(std::move(rq));
    if (vit_gated(rq)) return wait_.push_back(std::move(rq));  // an image of another request is between its blocks: it waits for the last block
    if (!rq.built && !build_prompt(rq)) return;  // kFlagPrefix: the prompt, once, from the history as it is after the session's request in flight
    note_image_ask(rq);
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
    const auto blocked = [&] {
      return (!cs || cs->slot < 0 || !fits(cs->slot, need) || !mem_ok(cs->slot, need, {}, sid, base, nullptr, &rq.prompt)) && pick_slot({}, need, sid, base, &rq.prompt) < 0;
    };
    if (blocked()) {
      if (!(flying_.empty() && pfq_.empty() && rot_.empty() && (front || wait_.empty())) || imager_work()) {  // running requests hold the rest, or images are still to be made
                                                                                                          // that free what it needs (never answered "cannot hold" while they are): it waits
        const RestoreAsk ask = restore_ask(sid, base, need, &rq.prompt);
        std::printf("serve: req %llu waits for a slot: %d positions need %d numbers; its stored conversation holds %d in the pool "
                    "and needs %d back; %d free and unreserved, %lld more can be given back (the request's own conversation "
                    "excepted), of %d numbers in the pool\n",
                    (unsigned long long)rq.h.req_id, need, m_.slot_chunks_for(0, need), ask.in_pool, ask.on_disk, m_.slot_uncommitted(),
                    (long long)giveable({}, sid, base, false), m_.slot_chunks_total());
        return wait_.push_back(std::move(rq));
      }
      // nothing runs and nothing waits ahead: what the pool does not hold now it never will. It is made to fit, or answered.
      if (!rescue(rq, need, base, blocked)) return;
    }
    start(std::move(rq));
  }

  // A request that cannot be admitted in an idle engine (admit): first its own conversation's window bookmarks other than its
  // end state go, oldest first, until it fits (restart points for a turn that diverges: the request makes new ones, and a
  // divergent turn restarts further back or cold, slower and never wrong); then a fork is admitted without its base (read
  // cold); what still does not fit is answered at once with the numbers. false: answered.
  template <class Blocked>
  bool rescue(Request& rq, int need, uint64_t& base, Blocked&& blocked) {
    const uint64_t sid = rq.h.session_id;
    const unsigned long long req = (unsigned long long)rq.h.req_id;
    if (Session* own = find_at(sid, factor_for((int64_t)rq.prompt.size()))) {
      int dropped = 0;
      // only bookmarks that hold numbers, oldest first, and only when all of them together would make it fit
      const std::vector<std::pair<int, int>> sn = m_.state_snap_numbers(own->h);
      int s0 = own->slot;
      for (int t = 0; s0 < 0 && t < (int)slot_.size(); ++t)
        if (fits(t, need)) s0 = t;
      std::vector<size_t> plan;
      if (s0 >= 0) {
        MemCache mc;
        int64_t have = 0, want = 0;
        mem_sums(s0, need, {}, sid, base, &mc, &have, &want, &rq.prompt);
        std::vector<int> held;
        for (const auto& p : sn) held.push_back(p.second);
        plan = model::rules::rescue_drops(have, want, held);
      }
      for (const size_t i : plan) {
        const int pos = sn[i].first;
        if (!blocked()) break;
        state({{kStDropSnap, own->h, pos, 0}});
        own->enc_snaps.erase(std::remove(own->enc_snaps.begin(), own->enc_snaps.end(), pos), own->enc_snaps.end());
        if (own->pe_snap == pos) own->pe_snap = -1;
        ++dropped;
      }
      if (dropped)
        std::printf("serve: req %llu: %d window bookmarks of its conversation dropped to fit the pool (%d numbers)\n", req, dropped,
                    m_.slot_chunks_total());
    }
    if (blocked() && base) {
      rq.h.base_session_id = 0, base = 0;
      std::printf("serve: req %llu: its base does not fit the pool beside it: the request is read cold\n", req);
    }
    if (!blocked()) return true;
    char msg[200];
    std::snprintf(msg, sizeof msg, "the pool of %d numbers cannot hold this request: %d positions need %d numbers", m_.slot_chunks_total(), need,
                  m_.slot_chunks_for(0, need));
    std::printf("serve: req %llu: %s\n", req, msg);
    done(rq, fe::kError, 0, 0, now_s(), 0, msg);
    Io::forget(rq);
    return false;
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
    // the Gumbel-path bit rides in top_k (model.h gsample_top_k); the client's value never reaches the bit's own pattern
    f.sp.top_k = model::gsample_top_k(model::ops::client_top_k(h.top_k), h.temperature, h.top_p, (h.flags & fe::kFlagSeed) != 0);
    f.sp.n_top = (h.flags & fe::kFlagLogprobs) ? (int32_t)std::min<uint32_t>(h.n_top_logprobs, model::ops::kTopLogprobs) : 0;
    // a request that wants logprobs keeps the full head (the certified head gives the argmax, not the partition sum)
    if ((h.flags & fe::kFlagLogprobs) && f.sp.n_top < 1) f.sp.n_top = 1;
#ifdef DSGB10_DEV
    {  // DSGB10_FORCE_NTOP, measurement only (development builds):: every request's rows carry their top logprobs in SampleOut (the step log)
      static const int force_ntop = getenv("DSGB10_FORCE_NTOP") ? atoi(getenv("DSGB10_FORCE_NTOP")) : 0;
      if (force_ntop > 0) f.sp.n_top = std::min(force_ntop, (int)model::ops::kTopLogprobs);
    }
#endif
    f.sp.seed = (h.flags & fe::kFlagSeed) ? h.seed : rng_();
    f.sp.ablit = (h.flags & fe::kFlagAblit) ? 1.f : 0.f;  // the runtime abliteration on this request (--ablit)
    for (const fe::ImageSpan& sp : f.rq.spans) f.img.rows += sp.rows;
    flying_.insert(h.session_id);
    JoinAsk ja;
    const int rows = il_.on ? rows_to_compute(f, false, &ja) : 0;
    if (il_.on && (ja.deferred || !il::read_whole(il_, rows, img_ms(f, rows), run_edge_ms(f)))) {
      auto L = std::make_unique<Long>();  // a long prompt: read in pieces, between the decoders' windows
      L->f = std::move(f);
      L->rows = rows;
      L->ja = ja;
      L->joining = ja.deferred;
      pfq_.push_back(std::move(L));
      return;
    }
    const Begin b = go(f, false);
    if (b == Begin::Began) {  // it decodes in the next run, first
      f.t_paused = now_s();
      rot_.push_front(std::move(f));
      stalled_ = false;
    } else if (b == Begin::Waits) {  // back to the front of the queue it came from, asked again after its retry time
      flying_.erase(f.rq.h.session_id);
      wait_.push_front(std::move(f.rq));
      start_wait_.stuck(now_s());
    }
  }

  // The rows a request's prompt would compute: after the restart point its session (or base) offers now. A resume's
  // prompt is the request's and the tokens it sent.
  int rows_to_compute(const Flight& f, bool resume = false, JoinAsk* ja = nullptr) {
    const fe::SubmitHeader& h = f.rq.h;
    std::vector<int32_t> cont;
    if (resume) cont = f.rq.prompt, cont.insert(cont.end(), f.out.begin(), f.out.end());
    const std::vector<int32_t>& prompt = resume ? cont : f.rq.prompt;
    const int n = (int)prompt.size();
    forget_evicted();
    // a split conversation's return: back to back with the request's own work when it fits a decode step's gap (or nobody decodes),
    // else the request is read as a long prompt whose first units are the join commands
    bool join_now = true;
    if (o_.kv_split && ja) {
      const JoinAsk a = join_ask(h.session_id, f.fac, bk(h, f.fac), need_now(f), prompt);
      if (a.found && il::return_asks(a.back, a.exch, a.disk) && il::join_in_units(a.est_ms, o_.kv_join_whole_ms, run_edge_ms(f), il_.gap_ms, !rot_.empty()) &&
          (a.disk == 0 || il::return_fits(a.disk, return_room(h.session_id, bk(h, f.fac)))))
        *ja = a, ja->deferred = true, join_now = false;
      else if (a.found) *ja = a;
    }
    unspill(h, f.fac, need_now(f), {}, prompt, join_now);
    if (o_.kv_split && ja && join_incomplete_) {  // the back-to-back join stopped for want of a number: the rest are units, whatever the estimate said
      *ja = join_ask(h.session_id, f.fac, bk(h, f.fac), need_now(f), prompt);
      ja->deferred = true;
    }
    const Session* src = find_at(h.session_id, f.fac);
    if (!src && h.base_session_id) src = find_at(h.base_session_id, f.fac);
    if (!src) return n;
    const int p = std::min(lcp(src->hist, prompt), n - 1);
    return n - choose(*src, src->slot >= 0, p, n).q;
  }

  // The decoders' step (ms) for the piece size: a line through two measured points, by the number of decoding sessions (at most 16)
  double step_ms() const { return rot_.empty() ? 0.0 : 25.77 + 15.9 * (double)(std::min<size_t>(rot_.size(), 16) - 1); }

  // The device time of an image request's read while sessions decode (0: not an image request, or nobody decodes)
  // What read_whole adds to an image request's read: for a request with pixels the run's edge (measured: il::Edge), else one step
  double run_edge_ms(const Flight& f) const { return f.rq.pix.empty() ? step_ms() : edge_.value(step_ms()); }
  // The decoders' silence across the units of requests read in pieces beside them (il::Edge): the last TOKENS before a unit and the first
  // after it. `tok_at_` is the time of the last token sent; `pend_` the unit whose silence is open.
  il::Edge edge_;
  double tok_at_ = 0;
  struct Pend {
    uint64_t req = 0;
    double pre = 0, unit_ms = 0;
    int kind = -1;  // 0 an encode unit, 1 a row piece; -1 none
  } pend_;
  struct UnitStat {
    int n[2] = {0, 0};
    double longest_ms[2] = {0, 0}, silence_ms[2] = {0, 0};
    double edge_sum = 0;
    int edge_n = 0;
  };
  std::map<uint64_t, UnitStat> unit_stats_;
  // A unit of request `req` of unit_ms ran beside decoders: its silence is read at the next token
  void note_unit(uint64_t req, int kind, double unit_ms) {
    if (rot_.empty() || !tok_at_) return;
    unit_stats_[req].n[kind]++;
    unit_stats_[req].longest_ms[kind] = std::max(unit_stats_[req].longest_ms[kind], unit_ms);
    pend_ = {req, tok_at_, unit_ms, kind};
  }
  void note_token() {
    const double now = now_s();
    if (pend_.kind >= 0) {
      const double sil = 1e3 * (now - pend_.pre);
      UnitStat& u = unit_stats_[pend_.req];
      u.silence_ms[pend_.kind] = std::max(u.silence_ms[pend_.kind], sil);
      u.edge_sum += std::max(0.0, sil - pend_.unit_ms), u.edge_n++;
      edge_.observe(sil, pend_.unit_ms);
      pend_.kind = -1;
    }
    tok_at_ = now;
  }
  // one line a request that was read in pieces beside decoders (at its end)
  void unit_line(uint64_t req) {
    const auto it = unit_stats_.find(req);
    if (it == unit_stats_.end()) return;
    const UnitStat& u = it->second;
    if (g_log_detail && u.n[0] + u.n[1] > 0)
      std::printf("serve: req %llu read in pieces beside decoders: %d encode units (longest %.1f ms, longest silence across one %.1f ms), %d row pieces "
                  "(longest %.1f ms, longest silence across one %.1f ms), the run edge %.1f ms mean (now %.1f ms)\n", (unsigned long long)req, u.n[0],
                  u.longest_ms[0], u.silence_ms[0], u.n[1], u.longest_ms[1], u.silence_ms[1], u.edge_n ? u.edge_sum / u.edge_n : 0.0,
                  edge_.value(step_ms()));
    unit_stats_.erase(it);
  }
  // plus the encode of the images its rows need (a pixel request: the unwritten ones whose spans end after the restart position)
  double img_ms(const Flight& f, int rows, bool resume = false) const {
    if (f.img.rows <= 0 || rot_.empty()) return 0.0;
    double enc = 0;
    if (!f.rq.pix.empty()) {
      const int q = (int)(f.rq.prompt.size() + (resume ? f.out.size() : 0)) - rows;
      enc = vitsched::encode_ms(vit_imgs(f), q, enc_cost_);
    }
    return il::img_read_ms(il_cost_, rows, enc);
  }

  // One pass of the scheduler while long prompts exist (il::next): admit one request, begin the front long prompt,
  // read a piece of it, or give the decoders their window
  void il_pass() {
    Long& L = *pfq_.front();
    const il::Settings& st = il_;
    il::State s;
    s.decoders = !rot_.empty();
    s.owed_ms = owed_ms_;
    s.reader = L.begun || L.joining;  // a join in flight reads like a piece: a command, then the decoders' window
    s.queued = !L.begun && !L.joining;
    const bool picked = !L.begun && pick_slot({}, need_long(L), L.f.rq.h.session_id, bkf(L.f), &L.f.rq.prompt) >= 0;
    s.reader_fits = !L.begun && il::begin_fits(picked, L.jw, now_s());
    pf_waits_ = splitpace::pf_start_waits(!L.begun && !L.joining, picked, L.start_waited);  // the front waits for the pool (read by the idle imager, the age and the rides)
    Long* cand = nullptr;  // the second reader's candidate: the first queued one of at most push_rows rows that fits
    if (L.begun && !push_ && st.push_rows > 0)
      for (size_t i = 1; i < pfq_.size() && !cand; ++i) {
        Long& q = *pfq_[i];
        if (!q.begun && q.rows <= st.push_rows && il::may_begin_beside(L.f.img.rows > 0, q.f.img.rows > 0, q.joining) &&
            il::begin_fits(pick_slot({L.f.rq.h.session_id}, need_long(q), q.f.rq.h.session_id, bkf(q.f), &q.f.rq.prompt) >= 0, q.jw, now_s()))
          cand = &q;
      }
    s.push_fits = cand != nullptr;
    const bool waiting = !wait_.empty() && start_wait_.may_try(now_s()) && !vit_gated(wait_.front()) && window_ready(wait_.front()) &&
                         pick_slot({}, need_clamped(wait_.front()), wait_.front().h.session_id, bkr(wait_.front()), &wait_.front().prompt) >= 0;
    s.arrivals = !join_active() && (!ready_.empty() || waiting || io_.submit_waiting());  // a join is a store change: nothing is admitted meanwhile
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
        // a prompt that waits stays queued, never dropped: it has no DONE yet, and it is asked again after its retry time
        if (L.begun) {  // the second reader, beside the front one (whose slot it may not take)
          const Begin b = pf_begin(cand->f, cand->resume, {L.f.rq.h.session_id}, cand->r);
          if (b == Begin::Began) cand->begun = true, cand->start_waited = false, push_ = cand;
          else if (b == Begin::Ended) drop_long(cand);
          else cand->jw.stuck(t0), cand->start_waited = true;
        } else {
          const Begin b = pf_begin(L.f, L.resume, {}, L.r);
          if (b == Begin::Began) L.begun = true, L.start_waited = false, pf_waits_ = false;
          else if (b == Begin::Ended) drop_long(&L);  // it ended at its start
          else L.jw.stuck(t0), L.start_waited = true, pf_waits_ = true;
        }
        // the decoders have waited since their last run (the arrival, its admission, this setup): owed before a piece
        if (s.decoders) owed_ms_ += il::window_ms(st, 1e3 * (now_s() - std::max(dec_end_, t0 - 1.0)));
        break;
      }
      case il::Next::Piece: {
        if (L.joining) {  // the request's first units: the join, one command at a time, each followed by the decoders' window
          if (!L.jw.may_try(t0)) {  // no number a moment ago: arrivals, releases and the other commands go on while it waits
            idle_wait();
            break;
          }
          const int r = join_unit(L.ja, L.f.rq.h.req_id, L.f.rq.h.session_id, bkf(L.f));
          if (r == 0) L.joining = false, L.jw.live = false;  // every list below `upto` is whole: it begins as any queued prompt does, next pass
          else if (r == 3) {  // not units: the prompt begins as any queued one; its begin joins what is still slots and reads the rest in commands of at most J
            L.joining = false, L.jw.live = false, L.ja.deferred = false;
            std::printf("serve: req %llu: the return of session %016llx is read at its begin: %d lists on the SSD, the pool gives %lld without a running request's line\n",
                        (unsigned long long)L.f.rq.h.req_id, (unsigned long long)L.f.rq.h.session_id, lists_disk(L.ja.h, L.ja.upto),
                        (long long)return_room(L.f.rq.h.session_id, bkf(L.f)));
            std::fflush(stdout);
            break;
          } else if (r == 1) L.jw.moved();
          else L.jw.stuck(t0);
          if (s.decoders && r != 2) owed_ms_ += il::window_ms(st, 1e3 * (now_s() - t0));
          break;
        }
        Long& P = push_ ? *push_ : L;  // the second reader reads first
        Reader& r = P.r;
        const bool dec = !rot_.empty();
        il::Settings sr = st;  // the pieces are bounded by this request's capacity, not the standing one
        sr.prefill_rows = r.cap;
        const int R = il::adjust_piece(il::piece_rows(sr, il_cost_, step_ms(), dec), r.pos, r.n, r.ced, r.cap);
        const int next = il::piece_rows(sr, il_cost_, step_ms(), dec);
        const int before = r.pos;
        const bool ok = pf_piece(r, R, next);
        const double ms = 1e3 * (now_s() - t0);
        if (ok && r.pos > before) il_cost_.k = 0.8 * il_cost_.k + 0.2 * ms / (il_cost_.fixed_ms + il_cost_.per_row_ms * R);  // an encode unit moves no rows
        if (ok && r.pos > before) note_unit(P.f.rq.h.req_id, 1, ms);  // every request read in pieces: the line's second half (a text prompt is the control)
        if (dec) owed_ms_ += il::window_ms(st, ms);
        if (!ok || r.pos >= r.n) {
          if (pf_end(r)) {  // it decodes from here, first in the next run
            P.f.t_paused = now_s();
            rot_.push_front(std::move(P.f));
            stalled_ = false;  // a request that just began to decode is not behind an idle command
          }
          drop_long(&P);
        }
        break;
      }
      case il::Next::Decode:
        burst_ms_ = splitpace::run_window(pf_waits_, pf_reading(), owed_ms_, st.burst_max_ms);
        if (burst_ms_ == 0) owed_ms_ = 0;  // nothing is being read: nobody owes the decoders
        burst_end_ = 0;
        run();
        burst_ms_ = 0;
        owed_ms_ = splitpace::owed_left(owed_ms_, 1e3 * (now_s() - t0), true);
        break;
      case il::Next::Idle:
        idle_wait();  // the front long prompt waits for a slot: wait for a request or a release
        break;
    }
  }

  void idle_wait() {
    publish();  // a long prompt waits for a line: the loop's own publish is off while prompts are queued
    Event e;
    if (io_.pop(e, 50)) {
      if (e.release) {
        if (flying_.count(e.session)) later_.insert(e.session);
        else release(e.session);
      } else if (!e.rq.conn->dead) {
        admit(std::move(e.rq));
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
    const int64_t emib = std::min<int64_t>(c.engram_mib, o_.engram_max_mib);  // at most the block's address space (int32 on the ring)
    if (emib >= 0 && m_.engram_lend_enabled() && emib != engram_set_mib_) {  // the Engram cache's size, MiB
      engram_set_mib_ = engram_goal_mib_ = emib;
      ++engram_gen_;  // a size that failed is tried again for a new setting
      engram_next_ = 0;
      std::printf("serve: the Engram cache goes to %lld MiB at the next run boundary\n", (long long)emib);
    }
#ifdef DSGB10_TEST_FAULTS
    if (c.test_tenant_groups >= 0 && c.test_tenant_groups != tenant_set_groups_) {  // the dummy tenant 15, at a run boundary
      tenant_set_groups_ = c.test_tenant_groups;
      tenant_goal_mib_ = tenant_set_groups_ * (int64_t)(m_.group_bytes() >> 20);
      ++tenant_gen_;
      tenant_next_ = 0;
      std::printf("serve: tenant 15 goes to %lld groups at the next run boundary\n", (long long)tenant_set_groups_);
    }
#endif
    published_ = 0;  // the next STATS shows it
    std::printf("serve: interleave settings: %s, whole %d rows, push %d rows, gap %d ms, share %d %%, window at most %d ms\n",
                il_.on ? "on" : "off", il_.whole_rows, il_.push_rows, il_.gap_ms, il_.share, il_.burst_max_ms);
  }

  // A decode run's window: set at its first step; over when a long prompt waits and the window has passed
  // whether the run is ended for another request (splitpace::pause_for_others); `can_begin` is asked last, at most every 50 ms
  bool pause_wanted(bool batch, bool ends) {
    splitpace::PauseIn o;
    o.batch = batch, o.pf = !pfq_.empty(), o.pf_busy = splitpace::pfq_busy(!pfq_.empty(), pf_reading(), pf_waits_), o.burst_over = burst_over();
    o.submit = io_.submit_waiting(), o.admissible = o.submit ? admissible() : false, o.ends = ends, o.wait = !wait_.empty();
    if (splitpace::pause_for_others(o)) return true;
#ifndef PICO_OLD_RULES
    if (o.pf && o.pf_busy) return false;
    if (!o.wait && !(pf_waits_ && !pf_reading())) return false;
    return can_begin_ask();
#else
    return false;
#endif
  }
  // The waiter can begin now: the head of wait_ (its retry time reached, not image-gated, its window open) or the queue's front that waits, with admission's own count
  // (pick_slot: free, idle lines' reservations, clean pieces; no request in flight loses its line). Never larger than what the loop admits.
  bool can_begin_ask() {
    const double t = now_s();
    if (!splitpace::ask_due(t, cb_last_, cb_miss_until_)) return false;
    cb_last_ = t;
    uint64_t id = 0;
    bool ok = false;
    if (!wait_.empty()) {
      const Request& r = wait_.front();
      id = r.h.req_id;
      ok = start_wait_.may_try(t) && !vit_gated(r) && window_ready(r) && pick_slot({}, need_clamped(r), r.h.session_id, bkr(r), &r.prompt) >= 0;
    } else if (!pfq_.empty() && pf_waits_ && !pf_reading()) {
      Long& L = *pfq_.front();
      id = L.f.rq.h.req_id;
      ok = il::begin_fits(pick_slot({}, need_long(L), L.f.rq.h.session_id, bkf(L.f), &L.f.rq.prompt) >= 0, L.jw, t);
    }
    if (!ok || !id) return false;
    if (cb_granted_ == id) {  // a run was ended for it and it waits again: a miss
      std::printf("serve: req %llu: a run was ended for its start and it waits again\n", (unsigned long long)id);
      cb_granted_ = 0, cb_miss_until_ = t + 1.0;
      return false;
    }
    cb_granted_ = id;
    std::printf("serve: req %llu can begin: the run is ended for it\n", (unsigned long long)id);
    return true;
  }
  double cb_last_ = 0, cb_miss_until_ = 0;
  uint64_t cb_granted_ = 0;
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
    bool whole = false;  // go(): the prompt is prefilled in one run (the encode of its images is not cut to the gap)
    bool fail = false;   // the request ended in an error inside a piece (Flight::err says which)
    // The set of row-sized prefill buffers the prompt is read on (Cmd::rows_set) and the rows a piece may have at most: chosen at the
    // prompt's begin, one value for every piece (its look-ahead lookups, its CED state and its snapshots belong to one set).
    int rows_set = 0, cap = 0;
    const std::vector<int32_t>& prompt() const { return resume ? cont : f->rq.prompt; }
  };

  // ---- vision: images encoded in the engine (vit_sched.h; model/vit_cmd.cu) ----
  // Why a pixel request cannot run here, or empty: the frame's own refusal (pixels::parse checked every grid), no encoder, no work area.
  std::string pixel_refusal(const Request& rq) const {
    if (!rq.refuse.empty()) return rq.refuse;
    const int vs = m_.vision_state();
    if (vs == 0) return pixels::kNoEncoder;
    if (vs == 2) return "the image encoder has no work area";
    return "";
  }
  std::vector<vitsched::Img> vit_imgs(const Flight& f) const {
    std::vector<vitsched::Img> im(f.rq.spans.size());
    for (size_t i = 0; i < im.size(); ++i)
      im[i] = {(int)f.rq.spans[i].start, (int)f.rq.spans[i].rows, (int)f.rq.px.grids[i].n_h * (int)f.rq.px.grids[i].n_w, i < f.enc.size() && f.enc[i] != 0};
    return im;
  }
  // One range of blocks [b0, b1) of image i of the pixel request: the command, its time, the mark. 0: done; <0: the request ended in an
  // error (r.fail, Flight::err).
  int vit_encode(Reader& r, int i, int b0, int b1) {
    Flight& f = *r.f;
    const uint64_t req = f.rq.h.req_id;
    const pixels::Grid& pg = f.rq.px.grids[(size_t)i];
    const vit::Grid g = vit::make_grid(pg.n_h, pg.n_w);
    if (!enc_mark_.may_run(req, i, b0)) die("serve: BUG: request %llu image %d blocks [%d, %d) while image %d of request %llu is between its blocks (next %d)",
                                          (unsigned long long)req, i, b0, b1, enc_mark_.image, (unsigned long long)enc_mark_.req, enc_mark_.next);
    size_t row0 = 0;
    for (int k = 0; k < i; ++k) row0 += f.rq.spans[(size_t)k].rows;
    Cmd c{};
    c.op = kCmdVit;
    c.vit_key = f.img.key, c.vit_span = i, c.vit_row0 = (int32_t)row0, c.vit_nh = pg.n_h, c.vit_nw = pg.n_w, c.vit_b0 = b0, c.vit_b1 = b1;
    issue(c);
    const double t0 = now_s();
    wd_.started = t0;
    const int st = run_vit(m_, c, reinterpret_cast<const uint8_t*>(f.rq.pix.data()) + f.rq.px.off[(size_t)i]);
    wd_.started = 0;
    const double ms = 1e3 * (now_s() - t0);
    enc_cost_.observe(ms, g.patches(), b1 - b0);
    if (st != model::Model::kVitOk) {
      enc_mark_.clear();
      f.err = "the image encoder's four ranks disagree on the rows of image " + std::to_string(i + 1) + " of this request";
      r.fail = true;
      return -1;
    }
    enc_mark_.ran(req, i, b1);
    if (b1 >= vitsched::kBlocks) {
      f.enc[(size_t)i] = 1;
      if (g_log_detail)
        std::printf("serve: req %llu image %d of %zu encoded: %d patches, %d rows in %.1f ms (blocks [%d, %d) the last)\n", (unsigned long long)req, i + 1,
                  f.rq.spans.size(), g.patches(), g.rows(), ms, b0, b1);
    }
    return 0;
  }
  // The encode unit due before the rows piece [r.pos, r.pos + R) of a pixel request, if one is: 0 none (the rows go on), 1 a unit ran, -1
  // the request ended (cancelled between two commands, or an error). The budget of a unit is the gap less one decode step while sessions
  // decode, unless the prompt is prefilled in one run (go(): read_whole counted the encode in its time).
  int vit_step(Reader& r, int R) {
    Flight& f = *r.f;
    double budget = std::numeric_limits<double>::max();
    if (!r.whole && il_.on && !rot_.empty()) budget = std::max(1.0, (double)il_.gap_ms - edge_.value(step_ms()));
#ifdef DSGB10_TEST_FAULTS
    if (const char* b = std::getenv("DSGB10_VIT_BUDGET_MS")) budget = std::atof(b);  // tests: the encode cut as finely as asked (0.001: a block at a time)
#endif
    const vitsched::Unit u = vitsched::next_unit(vit_imgs(f), r.pos, R, budget, enc_cost_, enc_mark_, f.rq.h.req_id);
    if (u.kind == vitsched::Unit::Rows) return 0;
    if (u.kind == vitsched::Unit::Hold)
      die("serve: BUG: request %llu would encode while image %d of request %llu is between its blocks", (unsigned long long)f.rq.h.req_id,
          enc_mark_.image, (unsigned long long)enc_mark_.req);
#ifdef DSGB10_TEST_FAULTS
    if (std::getenv("DSGB10_VIT_MIDTEST") && u.count >= 2 && !vit_midtest_done_) {  // a second image starts while the first is between its blocks: the engine stops
      vit_midtest_done_ = true;
      (void)vit_encode(r, u.first, 0, 1);
      Cmd c{};
      c.op = kCmdVit;
      c.vit_key = f.img.key, c.vit_span = u.first + 1, c.vit_nh = f.rq.px.grids[(size_t)u.first + 1].n_h, c.vit_nw = f.rq.px.grids[(size_t)u.first + 1].n_w;
      c.vit_b0 = 0, c.vit_b1 = 1;
      issue(c);
      (void)run_vit(m_, c, reinterpret_cast<const uint8_t*>(f.rq.pix.data()) + f.rq.px.off[(size_t)u.first + 1]);
    }
#endif
    const double t_unit = now_s();
    for (int k = 0; k < u.count; ++k) {
      if (Io::cancelled(f.rq)) {  // a cancel is taken between commands, never inside one
        r.cancelled = true;
        return -1;
      }
      if (vit_encode(r, u.first + k, u.count == 1 ? u.b0 : 0, u.count == 1 ? u.b1 : vitsched::kBlocks) < 0) return -1;
    }
    if (!r.whole) note_unit(f.rq.h.req_id, 0, 1e3 * (now_s() - t_unit));
    return 1;
  }
  // A pixel request that may not start or resume now (vitsched::may_start): an image of another request is between its blocks.
  bool vit_gated(const Request& rq) const { return !rq.pix.empty() && !vitsched::may_start(enc_mark_, true, rq.h.req_id); }
  // The waiting queue's front is the next to start: a gated pixel request does not hold the ones behind it (a text request is never held by it).
  void vit_rotate_wait() {
    for (size_t i = 0; i < wait_.size() && vit_gated(wait_.front()); ++i) {
      wait_.push_back(std::move(wait_.front()));
      wait_.pop_front();
    }
  }
  vitsched::EncCost enc_cost_;
  vitsched::Mark enc_mark_;  // the one image between its blocks, across requests
#ifdef DSGB10_TEST_FAULTS
  bool vit_midtest_done_ = false;
#endif

  // Flight f's prompt run in a slot: the request's first (resume false), or the resume of a request whose
  // session lost its slot, which runs the prompt and the tokens sent so far (the session holds them: it
  // restarts at their last position, exact continuation, one row; `keep`: the sessions whose slots it may
  // not take). Either way it sends a token, the flight's next to feed, and leaves its session live in the
  // slot; false when that token (or a cancel) ended the request. The sampler's noise is a function of the
  // seed and the position, so a request run in pieces samples what a single run would. Here the pieces run
  // back to back: the whole-prompt path.
  // What pf_begin did: it began (the slot and the state are set), the request ended at its start (its DONE is sent), or it waits
  // (the pool cannot hold it yet: nothing of it is held, the caller keeps it and asks again).
  enum class Begin { Began, Ended, Waits };
  // go() gives the same answers: Began after the prompt is prefilled and its first token sent, Ended when that token (or a cancel) ended it.
  Begin go(Flight& f, bool resume, const std::set<uint64_t>& keep = {}) {
    Reader r;
    r.whole = true;
    if (const Begin b = pf_begin(f, resume, keep, r); b != Begin::Began) return b;
    while (r.pos < r.n)
      if (!pf_piece(r, std::min(r.cap, r.n - r.pos), r.cap)) break;
    return pf_end(r) ? Begin::Began : Begin::Ended;
  }

  // The prompt's setup: the slot, where the state comes from, the restart point q, the state operations and the
  // prompt's tokens on every rank. Ended when the request ended here (a resume cancelled, or past the context); Waits when no slot
  // or not enough memory is there for it now (what the running requests are owed leaves less than admission counted).
  Begin pf_begin(Flight& f, bool resume, const std::set<uint64_t>& keep, Reader& r) {
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
      return ended_at_start(f, fe::kError);
    }
    unspill(h, fac, need_now(f), keep, prompt);

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
    // pick_slot() gives one that can, so room_for() below finds the memory; where the count and the pool disagree the request waits.
    const int need = need_now(f);
    const uint64_t sid = h.session_id, base = bk(h, fac);
    int s = S ? S->slot : -1;
    if (s >= 0 && (!fits(s, need) || !mem_ok(s, need, keep, sid, base, nullptr, &prompt)))
      persist_slot(s, true, holds(sid, base)), s = -1;  // a small slot (the session moves to a big one), or memory only another slot has
    if (!S && src && src->slot >= 0) {
      if (!flying_.count(src_id) && fits(src->slot, need) && mem_ok(src->slot, need, keep, sid, base, nullptr, &prompt)) s = src->slot;
      else persist_slot(src->slot, false, holds(sid, base));
    }
    if (s < 0) s = pick_slot(keep, need, sid, base, &prompt);
    if (s < 0) return begin_waits(f, "no slot", -1, need, keep, sid, base, &prompt);
    const bool src_live = src && src->slot == s;
    // the session that holds slot s (unless it is the fork's own base) leaves the line first, in a command of its own:
    // its numbers then belong to a conversation no line holds, which room_for may evict for this request
    if (slot_[s] && slot_[s] != sid && !src_live) persist_slot(s, true, holds(sid, base));
    if (!room_for(s, need, keep, sid, base, &prompt)) return begin_waits(f, "no room", s, need, keep, sid, base, &prompt);
    wait_logged_ = 0;
    int p = src ? std::min(lcp(src->hist, prompt), n - 1) : 0;
    if (src && join_incomplete_) {  // the join stopped for want of a number: the restart is planned from what is whole, never past the first slot the store's guard would cut at
      p = model::rules::whole_below(p, [&](int u) { return lists_split(src->h, u); });
    }
    Plan pl = src ? choose(*src, src_live, p, n) : Plan{};
    if (src && join_incomplete_ && pl.how != kLive && pl.how != kCold && m_.state_info(src->h).slots > 0) pl = Plan{};  // a bookmark that may still be a quarter slot is not read
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
      if (g_log_detail)
        std::printf("serve: req %llu factor %d (window %d): restart: %s; reused %d (%s)\n", (unsigned long long)h.req_id, fac,
                  window_of(fac), from, q, kHow[pl.how]);
    }
    if (src && q > 0) {  // the cut at q copies the chunks the session shares with another (own_tail), a fork's partial piece
      // takes a number per list: those numbers must be there before the operations run (the model dies on a pool held by lines)
      const int64_t cow = (int64_t)m_.state_cow_numbers(src->h, S ? S->h : -1, q);
      // one count with the request's own reservation (room_for above): the copy's numbers must not come out of what the
      // reservation was admitted against
      if (cow > 0)
        give_back(room_n(s, need, keep, sid, base, &prompt) + cow, keep, sid, base, s, 0,
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
    // the request's prompt and a horizon of response are committed (growth), and the numbers its image rows will take
    v.push_back({kStReserve, (int32_t)img_extra(sid), s, need});
    f.upto = need;
    v.insert(v.end(), pv.begin(), pv.end());
    if (!S) {
      Session ns;
      ns.h = next_h_++;
      ns.factor = fac;
      ns.ep.value = new_epoch();
      S = &sess_.emplace(h.session_id, std::move(ns)).first->second;
      if (resume && g_log_detail)
        std::printf("serve: req %llu factor %d (window %d): a new session for its %d positions, read again (%s)\n",
                    (unsigned long long)h.req_id, fac, window_of(fac), n - q, kHow[pl.how]);
    }
    v.push_back({kStAttach, S->h, s, src ? std::max(q, 1) : 0});  // the slot holds S: its rows are S's chunk lists (the chunk pool, no copy); a restart brings back only what is below its point
    v.push_back({kStFactor, S->h, s, fac});  // the slot rotates with S's factor from its first step; the store tags S
    if (src) {
      // S's own state is cut to q; a fork's base's [0, q) is copied into S's numbers (from a base the slot held too)
      if (q > 0 && (!src_live || src != S)) v.push_back({kStLoad, src->h, q, 0});
      if (pl.how == kSnap) v.push_back({kStRestore, src->h, pl.snap, 0});
      if (pl.how == kEnd) v.push_back({kStRestoreEnd, src->h, 0, 0});
      if (src != S) v.push_back({kStCopySnaps, src->h, S->h, q}), fork_base_ = src->h;  // a fork: src's snapshots are S's too
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
    S->file_ok = false;  // the run changes the history: its ids file is stale (kept until the idle ids replace it)
    prefix::lose_below(&S->ep, (uint64_t)q, [&] { return new_epoch(); });  // the epoch's first site: the history is cut below what a DONE reported
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
      // copy drafting (model/copy.h) keeps the whole conversation's tokens, so a unit whose session came from the store or
      // another slot has the right history before pos0: the whole prompt goes
      c.pos0 = model::copy::mode() ? 0 : std::max(0, q - 3);
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
      if (fork_base_ >= 0) split_note(fork_base_), fork_base_ = -1;  // the fork's setup is done: its base, left alone above the fork point, is split again
      std::vector<int32_t> t(prompt.begin() + c.pos0, prompt.end());
      for (const fe::ImageSpan& sp : f.rq.spans)  // vision: image positions are dead to Engram (-1)
        for (int k = std::max<int>(sp.start, c.pos0); k < (int)(sp.start + sp.rows); ++k) t[k - c.pos0] = -1;
      bcast_payload(ring_, t.data(), t.size() * 4, 0);
      m_.set_tokens(c.pos0, c.n, t.data());
      if (!f.rq.spans.empty()) images(f, s, sid, base);
      tu.ops_ms = 1e3 * (now_s() - (resume ? t0 : f.t_start));
    }
    r.S = S, r.s = s, r.n = n, r.q = q, r.ced = ced, r.pos = q;
    r.rows_set = n - q > o_.prefill_rows ? pick_rows_set() : 0;  // a prompt that fits the standing buffers never holds the lent set
    r.cap = m_.prefill_rows_cap(r.rows_set);
    return Begin::Began;
  }
  Begin ended_at_start(Flight& f, uint32_t why) {
    finish(f, why);
    return Begin::Ended;
  }
  // pf_begin's third answer: the request stays as it was (a read-back and the line's detach are all it did), the count that let it
  // in disagrees with what the pool can give. One line a request; each retry is the caller's.
  Begin begin_waits(const Flight& f, const char* what, int s, int need, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base,
                    const std::vector<int32_t>* pr) {
    if (wait_logged_ != f.rq.h.req_id) {
      wait_logged_ = f.rq.h.req_id;
      int t = s;
      for (int u = 0; t < 0 && u < (int)slot_.size(); ++u)
        if (fits(u, need)) t = u;
      MemCache mc;
      int64_t have = 0, want = 0;
      mem_sums(t < 0 ? 0 : t, need, keep, sid, base, &mc, &have, &want, pr);
      std::printf("serve: req %llu waits at its start (%s): have %lld, want %lld numbers\n", (unsigned long long)f.rq.h.req_id, what,
                  (long long)have, (long long)want);
    }
    return Begin::Waits;
  }

  // The set of row-sized buffers a prompt that begins now is read on: the lent set when it is held whole on every rank and nobody has asked
  // it back, else the standing one. (Cmd::rows_set; the same value goes to every rank in the piece's command.)
  int pick_rows_set() const { return loan_goal_ == 1 && m_.prefill_rows_cap(1) > 0 ? 1 : 0; }

  // ---- the lent prefill row set (model/rows_lend.cu, rows_loan.h) ----
  // loan_goal_: 1 the set is wanted, 0 it is asked back, -1 not steered yet. A prompt being read on it holds it (loan_in_use).
  int loan_goal_ = -1;
  bool loan_forced_ = false;  // a consumer asked it back (give_back's tier 0): it goes whatever the marks say
  int64_t loan_mib_ = 0;  // the set's size, MiB (0: no second set)
  int loan_groups_ = 0;
  double loan_check_ = 0, loan_next_ = 0;
  model::lend::Backoff loan_backoff_;
  uint64_t loan_gen_ = 0;
  bool loan_enabled() const { return loan_mib_ > 0; }
  bool loan_held() const { return m_.tenant_group_count(model::lend::kTenantPrefill) > 0; }
  bool loan_in_use() const {
    for (const auto& q : pfq_)
      if (q->begun && q->r.rows_set == 1) return true;
    return false;
  }
  // One line on rank 0 whenever the loan's reason changes (the start, then every transition), so that a loan that is not held, or not
  // given back, says why: the free groups against the ones the policy needs, the readers it waits for.
  rowsloan::Why loan_why_ = rowsloan::Why::Off;
  bool loan_why_said_ = false, loan_busy_said_ = false;
  int loan_readers() const {
    int n = 0;
    for (const auto& q : pfq_) n += q->begun && q->r.rows_set == 1;
    return n;
  }
  void loan_say(const rowsloan::State& s) {
    const rowsloan::Why w = rowsloan::why(s);
    if (loan_why_said_ && w == loan_why_) return;
    loan_why_said_ = true, loan_why_ = w;
    using W = rowsloan::Why;
    switch (w) {
      case W::Off: return;
      case W::Changing: return;  // the change's own lines follow
      case W::HeldPlenty: std::printf("serve: prefill rows: held (%d groups), the pool has %d groups free (goes back under %d)\n", s.held, s.free_groups, s.marks.low_free); break;
      case W::HeldReading: std::printf("serve: prefill rows: the pool has %d groups free (under %d): waiting for %d prompt(s) reading on the second set before it goes back\n", s.free_groups, s.marks.low_free, loan_readers()); break;
      case W::HeldShort: std::printf("serve: prefill rows: the pool has %d groups free (under %d) and nothing reads on the second set: it goes back\n", s.free_groups, s.marks.low_free); break;
      case W::WaitBackoff: std::printf("serve: prefill rows: not held; the last ask failed and is not repeated yet\n"); break;
      case W::WaitTopBusy: std::printf("serve: prefill rows: not held; a number in the pool's top %d groups is in use (a take would evict): not asked\n", s.loan); break;
      case W::WaitFewFree: std::printf("serve: prefill rows: not held; the pool has %d groups free, %d needed (the set's %d plus %d): not asked\n", s.free_groups, s.loan + s.marks.high_extra, s.loan, s.marks.high_extra); break;
      case W::CanAsk: std::printf("serve: prefill rows: not held; %d groups free and the top %d groups unused: asked for\n", s.free_groups, s.loan); break;
    }
    std::fflush(stdout);
  }

  // Once a second at most: what the policy says now, and the goal it sets. The pool's questions are read only.
  void loan_policy(double t) {
    if (!loan_enabled() || t < loan_check_) return;
    loan_check_ = t + 1.0;
    m_.tenant_mark_idle(model::lend::kTenantPrefill, !loan_in_use());
    if (m_.tenant_active()) return;
    rowsloan::State s;
    s.enabled = true, s.loan = loan_groups_;
    s.marks.low_free = o_.loan_low_groups, s.marks.high_extra = o_.loan_high_extra;
    s.held = m_.tenant_group_count(model::lend::kTenantPrefill);
    s.in_use = loan_in_use();
    s.going_back = s.held > 0 && loan_goal_ == 0;  // asked back: the state is sticky (rows_loan.h)
    s.free_groups = (int)(m_.pool_free_numbers() / model::lend::kGroupNums);
    s.top_free = s.held == 0 && m_.pool_top_free(loan_groups_);
    s.backoff = loan_backoff_.blocked((uint64_t)loan_mib_, t, loan_gen_);
    loan_say(s);
    switch (rowsloan::decide(s)) {
      case rowsloan::Act::Ask:
        std::printf("serve: the pool has %d groups free and none of its top %d in use: the prefill's second set of row buffers is asked for\n",
                    s.free_groups, loan_groups_);
        loan_goal_ = 1, loan_next_ = 0;
        break;
      case rowsloan::Act::GiveBack:
        if (loan_goal_ != 0) {
          std::printf("serve: the pool has %d groups free (under %d): the prefill's second set of row buffers is going back%s\n", s.free_groups,
                      s.marks.low_free, s.in_use ? " (when the prompt reading on it has ended)" : "");
          loan_goal_ = 0, loan_next_ = 0, loan_backoff_ = model::lend::Backoff();
        }
        break;
      case rowsloan::Act::Withdraw:
        if (!loan_forced_) {
          std::printf("serve: the pool has %d groups free again (the set's %d plus %d): the prefill's second set stays\n", s.free_groups, s.loan,
                      s.marks.high_extra);
          loan_goal_ = 1;
        }
        break;
      case rowsloan::Act::Keep:  // an ask whose conditions went is withdrawn; a held loan keeps its goal
        if (s.held == 0) loan_goal_ = 0, loan_forced_ = false;
        else if (loan_goal_ < 0) loan_goal_ = 1;
        break;
    }
  }
  // Tier 0 of give_back: a consumer is short and the second set is held and unused: it goes back before any line is detached or any
  // conversation evicted. True when its return is under way (the consumer waits for the groups and asks again).
  bool loan_give_way(double t) {
    if (!loan_enabled() || !loan_held() || loan_in_use()) return false;
    if (m_.tenant_active()) return m_.tenant_change_id() == model::lend::kTenantPrefill;
    if (loan_backoff_.blocked(0, t, loan_gen_)) return false;
    std::printf("serve: a request needs the pool's memory: the prefill's second set of row buffers goes back first\n");
    loan_goal_ = 0, loan_next_ = 0, loan_forced_ = true;
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
    if (!f.rq.pix.empty())  // vision: an image the piece reaches is encoded first, as a unit of its own
      if (const int e = vit_step(r, R)) return e > 0;
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
    // DSGB10_REUSE_EXACT=1: a reused prompt's short remainder (at most --small-rows positions after its restart point) takes the
    // prefill path, not one decode step. The step sums in another order than the prefill, so a request sent again would leave
    // a state that is not its cold prefill's, and its reply could differ at a near-tie. A whole prompt of at most --small-rows
    // positions (its cold run steps it too) and a time-sliced request's resume keep the step. 0 / unset: as without it.
    static const int reuse_exact = [] {
      const char* e = std::getenv("DSGB10_REUSE_EXACT");
      const int v = e && *e ? std::atoi(e) : 0;
      if (v < 0 || v > 1 || (e && *e && !(e[0] >= '0' && e[0] <= '1' && e[1] == 0))) die("DSGB10_REUSE_EXACT=%s: 0 or 1", e);
      if (v) std::printf("serve: DSGB10_REUSE_EXACT=1: a reused prompt's remainder of at most --small-rows positions takes the prefill path\n");
      return v;
    }();
    const bool step_ok = resume || reuse_exact < 1 || n <= o_.small_rows;
    if (R <= o_.small_rows && !ced && !img && step_ok) {
      c.op = kCmdRows;
      std::copy(prompt.begin() + pos, prompt.begin() + pos + R, c.tokens);
      issue(c);
      const uint64_t k = run_rows(m_, c);
      first = m_.wait(k);
      tu.rows_ms += m_.last_gpu_ms();
      first_row = R - 1;
      S->w = std::max(S->w, pos + R + (m_.spec() ? model::copy::verify_reach(dspark::draft_rows(), (model::copy::mode() & model::copy::kRows) != 0, model::copy::copy_rows()) : 0));  // its draft's block keys
    } else {
      c.op = kCmdPrefill;
      const bool watch = memwatch::chunk_ride_due(mem_watch_on(), now_s(), watch_at_);  // the memory watch rides this chunk
      c.flags = (last ? kPfSample : 0) | (ced ? kPfCed : 0) | (watch ? kPfWatch : 0);
      const int rest = n - pos - R;
      c.next = rest > o_.small_rows || (ced && rest > 0) ? std::min(next_rows, rest) : 0;
      for (int i = 0; i < c.n_snaps; ++i)
        if (ced && c.snaps[i] < n) S->enc_snaps.push_back(c.snaps[i]);  // the replay covers only one at n
      issue(c);
      if (watch) mem_watch_note(watch_over());
      const uint64_t k = run_prefill(m_, c);
      if (watch) {  // every rank has read: rank 0's run_prefill returns after the all-gather
        watch_at_ = now_s();
        watch_line("in a prompt's chunk");
      }
      publish();  // the STATS frame is built while the device computes the chunk, not in the gap before the next one
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
      m_.set_images(0, nullptr, nullptr, 0);  // the view only: the rows stay in the store until the request ends
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
    if (cancelled || r.fail || Io::cancelled(f.rq)) {
      drain();
      return ended(f, r.fail ? fe::kError : fe::kCancelled);
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

  // Vision: the flight's spans to every rank for slot s's prefill chunks (kCmdImages), whatever runs between them. Its first
  // go takes the rows' numbers from the store (kStImgTake, the request's own conversation and its base pinned) and sends
  // the rows in pieces, straight from rank 0's copy, which is freed after the last; every later go (a resume included)
  // sends the spans only, and the ranks read the rows they hold. They are freed when the request ends (finish).
  void images(Flight& f, int s, uint64_t sid, uint64_t base) {
    const double t0 = now_s();
    Request& rq = f.rq;
    const imghold::Go g = imghold::on_go(f.img);
    Cmd c{};
    c.op = kCmdImages;
    c.slot = s;
    c.n = (int)rq.spans.size();
    if (g.take) {
      f.img.key = next_img_++;
      for (const int h : pins(sid, base)) c.ops[c.n_ops++] = {kStPin, h, 0, 0};
      c.ops[c.n_ops++] = {kStImgTake, f.img.key, (int32_t)f.img.rows, s};
      img_ask_.taken(sid);
    }
    const bool px = !rq.pix.empty();  // pixels: the rows are written by the encoder, one image at a time (vit_step), not sent
    c.rows = g.fill && !px ? 1 : 0;
    c.pos0 = f.img.key;
    c.vit_reg = g.take && px ? 1 : 0;
    issue(c);
    if (g.take && px) m_.vision_register(f.img.key, c.n), f.enc.assign(rq.spans.size(), 0);
    if (g.take) digest("image take");
    std::vector<int32_t> lh(2 * rq.spans.size());
    for (size_t i = 0; i < rq.spans.size(); ++i)
      lh[i] = (int32_t)rq.spans[i].start, lh[rq.spans.size() + i] = (int32_t)(rq.spans[i].start + rq.spans[i].rows);
    bcast_payload(ring_, lh.data(), lh.size() * 4, 0);
    if (g.fill && !px) {
      const size_t total = rq.img.size() / fe::kImageDim;
      for (size_t r0 = 0; r0 < total; r0 += model::Model::kImgStageRows) {
        const size_t n = std::min(model::Model::kImgStageRows, total - r0);
        uint16_t* piece = rq.img.data() + r0 * fe::kImageDim;
        bcast_payload(ring_, piece, n * fe::kImageDim * 2, 0);
        m_.image_fill(f.img.key, r0, n, piece);
      }
      std::vector<uint16_t>().swap(rq.img);  // rank 0 keeps no copy: the store has the rows
    }
    m_.set_images(c.n, lh.data(), lh.data() + c.n, f.img.key);
    if (g_log_detail)
      std::printf("serve: req %llu images: %d spans, %zu rows, %s in %.1f ms (%zu image numbers held)\n",
                (unsigned long long)rq.h.req_id, c.n, f.img.rows, g.fill ? (px ? "taken, to be encoded here" : "taken and sent to every rank") : "held", 1e3 * (now_s() - t0),
                m_.image_numbers());
  }
  int next_img_ = 1;  // the keys the ranks hold flights' image rows under

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
  std::vector<int> pins(uint64_t sid, uint64_t base, const std::set<uint64_t>* keep = nullptr, bool all_live = false, bool paused_too = true) const {
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
  // A session whose slot memory the request may take (room_for, mem_ok): not the request's own or its base, not in the
  // run (`keep`), and with no request in flight, or paused between turns. Counting paused ones keeps every admitted
  // request able to run: a flight that lost its slot finds one when its turn comes.
  // `paused_too` false: a paused request's line is not idle (a read-back unit's rule, reclaim::unit_may_detach).
  bool idle_id(uint64_t id, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, bool paused_too = true) const {
    if (id == sid || id == base || keep.count(id)) return false;
    // a request that is not running takes nothing from one that is, so the line of a paused request is not idle for any taker (PICO_OLD_RULES: it is, for a reservation)
    return paused_too ? reclaim::idle_for_taker(flying_.count(id) > 0, paused(id)) : reclaim::unit_may_detach(flying_.count(id) > 0, paused(id));
  }
  // A slot whose session no request uses and the request does not need: its memory can be given back (room_for)
  bool idle_slot(int t, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, bool paused_too = true) const {
    return idle_id(slot_[t], keep, sid, base, paused_too);
  }
  // What the request continues from, read at the moment of the count (never kept across passes: the base may be rewritten
  // meanwhile): its own conversation, else its fork's base. The lists that hold positions below `upto` (the request's need for
  // its own conversation; the fork point p = min(lcp(base history, prompt), n - 1), as the plan computes it, for a base), those
  // in the pool and those only on the disk, the bookmarks' numbers on the disk, and the copies a fork makes at p. mem_ok's want
  // takes the lists in the pool off (they become the slot's) and leaves the bookmarks out (they come back into what is spare:
  // unspill's budget).
  struct RestoreAsk {
    bool found = false, fork = false;
    uint64_t src_id = 0;           // the key of the conversation counted
    int in_pool = 0, on_disk = 0;  // its lists below `upto`
    int split = 0;                 // its lists below `upto` that are quarter slots: each needs a whole number to come back into
    int upto = 0;                  // the position the count and a join go to (the request's need; a fork's point)
    int64_t bookmarks = 0;         // its bookmarks' numbers on the disk
    int64_t cow = 0;               // a fork's copies at p
  };
  RestoreAsk restore_ask(uint64_t sid, uint64_t base, int need, const std::vector<int32_t>* pr) const {
    RestoreAsk a;
    auto it = sid ? sess_.find(sid) : sess_.end();
    int upto = need;
    if (it == sess_.end() && base && base != sid) {
      it = sess_.find(base);
      if (it != sess_.end()) {
        a.fork = true;
        const int p = pr && !pr->empty() ? std::min(lcp(it->second.hist, *pr), (int)pr->size() - 1) : 0;
        upto = std::max(0, p);
        a.cow = (int64_t)m_.state_cow_numbers(it->second.h, -1, p);
      }
    }
    if (it == sess_.end()) return a;
    a.found = true, a.src_id = it->first;
    a.upto = upto;
    m_.state_list_numbers(it->second.h, upto, &a.in_pool, &a.on_disk, &a.split);
    int all_in = 0, all_disk = 0;
    m_.state_list_numbers(it->second.h, INT_MAX, &all_in, &all_disk);
    a.bookmarks = (int64_t)m_.state_restore_chunks(it->second.h) - all_disk;
    return a;
  }
  // The credit slot s gives: the lists of the conversation the request continues that are in the pool. Not when that
  // conversation holds slot s: they are in the slot's commitment already (own).
  int64_t credit_of(int s, const RestoreAsk& a) const { return a.found && slot_[s] != a.src_id ? model::rules::list_credit(a.in_pool) : 0; }
  // The numbers slot s needs from the pool for the request (mem_ok): see pool_rules.h request_want.
  int64_t want_of(int s, int need, const RestoreAsk& a, int64_t img) const {
    return model::rules::request_want((int64_t)m_.slot_chunks_for(s, need), credit_of(s, a), a.on_disk + a.split, a.bookmarks, a.cow, img);
  }
  // What the reservation of slot s still needs from free and given-back numbers (room_for, the copy at the restart
  // point): the same count, read now, with the slot's own commitment taken off.
  int64_t room_n(int s, int need, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, const std::vector<int32_t>* pr) const {
    const RestoreAsk a = restore_ask(sid, base, need, pr);
    return model::rules::room_want((int64_t)m_.slot_chunks_for(s, need), credit_of(s, a), img_extra(sid)) - own(s, keep, sid, base);
  }
  // The numbers a request's image rows will take at its first go, counted in its want like a restore: admission and room_for
  // find them before the request is let in, and kStReserve reserves them until kStImgTake takes them.
  void note_image_ask(const Request& rq) {
    if (rq.spans.empty()) return;
    size_t rows = 0;
    for (const fe::ImageSpan& sp : rq.spans) rows += sp.rows;
    imghold::Hold h;
    h.rows = rows;
    img_ask_.note(rq.h.session_id, rq.h.req_id, imghold::admission_numbers(h));
  }
  int64_t img_extra(uint64_t sid) const { return img_ask_.extra(sid); }
  imghold::Asks img_ask_;  // session id -> numbers its admitted request's image rows will take (until taken or it ends)
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
  // pool, what the store holds under unused sessions: room_for, Model::slot_reserve) must cover it. The request's own
  // session and base on disk count as restored first (restore_ask, want_of). Without the pool: always.
  bool mem_ok(int s, int need, const std::set<uint64_t>& keep, uint64_t sid = 0, uint64_t base = 0, MemCache* mc = nullptr,
              const std::vector<int32_t>* pr = nullptr) const {
    MemCache tmp;
    MemCache& c = mc ? *mc : tmp;
    if (!c.ex) c.ask = restore_ask(sid, base, need, pr), c.img = img_extra(sid), c.ex = true;
    const int64_t want = want_of(s, need, c.ask, c.img);
    int64_t have = m_.slot_spare() + own(s, keep, sid, base);
#ifdef PICO_OLD_RULES
    have = idle_sum_have(s, keep, sid, base, have, c, want);
#else
    if (!o_.kv_split) have = idle_sum_have(s, keep, sid, base, have, c, want);  // --kv-split off: the same sum, with the spare signed: smaller by what the reservations exceed the free numbers, equal elsewhere
    else if (have < want) have = pool_have_for(s, keep, sid, base, have);
#endif
    return model::rules::admissible(have, want);
  }
  // The sum without the split tiers (--kv-split off, and PICO_OLD_RULES): each idle line's commitment, then what eviction gives back for the conversations no
  // line holds; it stops adding once `want` is covered. A number two idle lines name is counted by each.
  int64_t idle_sum_have(int s, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, int64_t have, MemCache& c, int64_t want) const {
    for (int t = 0; t < (int)slot_.size() && have < want; ++t)
      if (t != s && idle_slot(t, keep, sid, base)) have += m_.slot_committed(t);
    if (have < want) {
      if (!c.gv) c.giveable = giveable(keep, sid, base, false), c.gv = true;
      have += c.giveable;
    }
    return have;
  }
  // What the pool can give the request in slot s, counted once (pool_rules.h pool_have): the free numbers beside the reservations (`have`),
  // the reservations of the idle lines the request may take (they end with the detach), and what eviction gives back for the conversations
  // of those lines counted as if no line held them, a number two of them name only once. The conversations of the lines are
  // `lines`; the request's own and its base are pinned out as always.
  int64_t pool_have_for(int s, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, int64_t have, bool paused_too = true) const {
    int64_t reservations = 0;
    std::vector<int> lines;
    for (int t = 0; t < (int)slot_.size(); ++t)
      if (t != s && idle_slot(t, keep, sid, base, paused_too)) {
        reservations += m_.slot_unheld_reservation(t);
        const auto it = sess_.find(slot_[t]);
        if (it != sess_.end()) lines.push_back(it->second.h);
      }
    return model::rules::pool_have(have, reservations, (int64_t)m_.state_giveable_joint(pins(sid, base, &keep, false, paused_too), lines));
  }
  // What the pool gives a read-back unit: the sum admission makes (free and unreserved numbers, the reservations of the lines the unit may
  // detach, the store those lines' conversations free), less the lines of paused requests: a unit never takes a request's line.
  int64_t return_room(uint64_t sid, uint64_t base) const {
    return pool_have_for(-1, {}, sid, base, m_.slot_spare(), false);
  }
  // mem_ok's two sums whole (no early stop): what the pool can give the request in slot s, and what it needs
  void mem_sums(int s, int need, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, MemCache* mc, int64_t* have_out,
                int64_t* want_out, const std::vector<int32_t>* pr = nullptr) const {
    MemCache& c = *mc;
    if (!c.ex) c.ask = restore_ask(sid, base, need, pr), c.img = img_extra(sid), c.ex = true;
    const int64_t have = m_.slot_spare() + own(s, keep, sid, base);
#ifdef PICO_OLD_RULES
    *have_out = idle_sum_have(s, keep, sid, base, have, c, INT64_MAX);
#else
    *have_out = o_.kv_split ? pool_have_for(s, keep, sid, base, have) : idle_sum_have(s, keep, sid, base, have, c, INT64_MAX);
#endif
    *want_out = want_of(s, need, c.ask, c.img);
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
  // alone: what Model::slot_reserve does). false when only running requests hold the rest (the caller waits)
  bool room_for(int s, int need, const std::set<uint64_t>& keep, uint64_t sid, uint64_t base, const std::vector<int32_t>* pr = nullptr) {
    char why[48];
    std::snprintf(why, sizeof why, "a request of %d positions", need);
    return give_back(room_n(s, need, keep, sid, base, pr), keep, sid, base, s, 0, why);
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
                 const char* why, bool may_close = true, bool paused_too = true) {
    auto short_by = [&](int64_t w) {
      const int64_t u = m_.slot_spare();
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
    if (may_close && close_unused_window() && !short_by(n)) return true;
    if (may_close) engram_give_way();  // the last tier: the answer is still no; the group arrives and the consumer asks again
    return false;
  }
  // A slot for a session that holds none, among those that hold `need` positions (with big slots, the small
  // ones first): a free one; else the least recently used one whose session no request uses (its state goes
  // to the store); else, with time-slicing, the slot of the paused request whose turn comes last. Never the
  // slot of a session in `keep`. -1: none. (sid, base: the request's session and its base, whose stored state the
  // store keeps.)
  int pick_slot(const std::set<uint64_t>& keep, int need = 0, uint64_t sid = 0, uint64_t base = 0,
                const std::vector<int32_t>* pr = nullptr) const {
    const int ns = (int)slot_.size();
    const int big = m_.slot_ctx(0);
    MemCache mc;
    for (int pass = 0; pass < 2; ++pass)  // small slots first (big_slots), so that the big ones stay for long requests
      for (int s = 0; s < ns; ++s)
        if (!slot_[s] && fits(s, need) && (pass == 1 || m_.slot_ctx(s) < big) && mem_ok(s, need, keep, sid, base, &mc, pr)) return s;
    int best = -1;
    uint64_t used = UINT64_MAX;
    for (int s = 0; s < ns; ++s) {
      if (!fits(s, need) || !splitpace::slot_free_for_arrival(flying_.count(slot_[s]) > 0, run_ended_.count(slot_[s]) > 0) || keep.count(slot_[s]) || !mem_ok(s, need, keep, sid, base, &mc, pr)) continue;
      const auto it = sess_.find(slot_[s]);
      const uint64_t u = it == sess_.end() ? 0 : it->second.used;
      if (u < used) best = s, used = u;
    }
    if (best >= 0 || o_.slice <= 0) return best;
#ifndef PICO_OLD_RULES
    return best;  // a newcomer never takes the slot of a paused request (turn-slicing among requests in flight is theirs; the extra one waits for an end)
#endif
    for (auto it = rot_.rbegin(); it != rot_.rend(); ++it) {
      const auto x = sess_.find(it->rq.h.session_id);
      if (x != sess_.end() && x->second.slot >= 0 && !keep.count(x->first) && fits(x->second.slot, need) &&
          mem_ok(x->second.slot, need, keep, sid, base, &mc, pr))
        return x->second.slot;
    }
    return -1;
  }
  // whether a submit that waits may pause the requests decoding: it can have a slot
#ifdef PICO_OLD_RULES
  bool admissible() const { return o_.slice > 0 || pick_slot({}) >= 0; }
#else
  bool admissible() const { return pick_slot({}) >= 0; }  // with no slot to be had a waiter takes nothing, so its arrival ends no run
#endif

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
    int holders = 0;               // flights of this pass with a live line (the standstill's test)
    std::set<uint64_t> ext_ids;    // the requests put back for an extension (the one with the longest response is the one ended)
    bool start_waits = false;      // a flight of this pass waits at its start for the pool
    int pool_waits = 0;            // flights this pass put back because memory could not be had (the imager's reason, as a request in wait_ is)
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
          ++pool_waits;
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
      if (live) ++holders;
      if (!live && pick_slot(keep, need, f.rq.h.session_id, bkf(f), &f.rq.prompt) < 0) {
        ++pool_waits;
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
        const bool room = mem_ok(s, need, keep, f.rq.h.session_id, bkf(f), nullptr, &f.rq.prompt) && room_for(s, need, keep, f.rq.h.session_id, bkf(f), &f.rq.prompt);
        const int left = f.upto - (int)(f.rq.prompt.size() + f.out.size());
        if (!room && reclaim::ext_blocks(left)) {  // (b): only when its reservation is used up; else it runs on with what it holds
          ++ext_waits, ++pool_waits;
          ext_ids.insert(f.rq.h.req_id);
          later.push_back(std::move(f));
          continue;
        }
        if (room) {
          state({{kStReserve, 0, s, need}});
          std::printf("serve: req %llu grows: its reservation reaches %d positions (was %d)\n", (unsigned long long)f.rq.h.req_id,
                      need, f.upto);
          f.upto = need;
        }
      }
      if (!live && vit_gated(f.rq)) {  // a pixel request is not resumed while another's image is between its blocks: paused for this pass
        later.push_back(std::move(f));
        continue;
      }
      f.paused_s += now_s() - f.t_paused;
      // a resume with more rows to read than a prompt read whole: read in pieces like a long prompt, between windows
      JoinAsk ja;
      const int rows = !live && il_.on ? rows_to_compute(f, true, &ja) : 0;
      if (!live && il_.on && (ja.deferred || !il::read_whole(il_, rows, img_ms(f, rows, true), run_edge_ms(f)))) {
        std::printf("serve: req %llu resumes in pieces: %d rows to read again\n", (unsigned long long)f.rq.h.req_id, rows);
        auto L = std::make_unique<Long>();
        L->f = std::move(f);
        L->resume = true;
        L->rows = rows;
        L->ja = ja;
        L->joining = ja.deferred;
        pfq_.push_back(std::move(L));
        continue;
      }
      if (!live) {
        const Begin b = go(f, true, keep);
        if (b == Begin::Waits) {  // paused for the next pass, as a flight that waits for a slot
          f.t_paused = now_s();
          later.push_back(std::move(f));
          start_waits = true, ++pool_waits;
          continue;
        }
        if (b == Begin::Ended) continue;
      }
      keep.insert(f.rq.h.session_id);
      fl.push_back(std::move(f));
    }
    for (auto it = later.rbegin(); it != later.rend(); ++it) rot_.push_front(std::move(*it));
    pool_waits_ = pool_waits;  // read by the idle imager, its age and the rides (splitpace::rot_stalled, pool_waiter)
    stalled_ = splitpace::rot_stalled((int)fl.size(), pool_waits, pf_reading());
    // every flight of two passes in a row waited for an extension: nothing can end, pause or free what they wait for
    // (--slice 0, whose flights are never paused, cannot break it; the production setting can)
#ifdef PICO_OLD_RULES
    if (fl.empty() && ext_waits > 0 && (size_t)ext_waits == rot_.size() && ++ext_stall_ == 2)
      std::printf("serve: BUG: every flight waits for an extension (%d flights)\n", ext_waits);
    if (!fl.empty() || ext_waits == 0) ext_stall_ = 0;
    if (fl.empty() && start_waits && pfq_.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(20));  // nothing else flies: the pool is asked again shortly
#else
    (void)start_waits;
    {  // the standstill is a fact of the pass, not a count of passes; the line is the truth after 2 s; the loop rests whenever a pool wait took the whole pass
      const bool standstill = reclaim::ext_standstill((int)fl.size(), holders, ext_waits, imager_work(), join_active(), pf_reading());
      const double tn = now_s();
      if (!standstill) ext_since_ = 0, ext_noted_ = false;
      else {
        if (ext_since_ == 0) ext_since_ = tn;
        if (!ext_noted_ && tn - ext_since_ >= 2.0) {
          ext_noted_ = true;
          std::printf("serve: BUG: every flight waits for an extension (%d flights)\n", ext_waits);
        }
        if (reclaim::ext_end_due(tn - ext_since_, tn - ext_end_last_)) {  // the waiter with the longest response ends
          std::vector<int> outs;
          std::vector<size_t> at;
          for (size_t i = 0; i < rot_.size(); ++i)
            if (ext_ids.count(rot_[i].rq.h.req_id)) outs.push_back((int)rot_[i].out.size()), at.push_back(i);
          const int pick = reclaim::ext_pick(outs);
          if (pick >= 0) {
            Flight victim = std::move(rot_[at[(size_t)pick]]);
            rot_.erase(rot_.begin() + (long)at[(size_t)pick]);
            ext_end_last_ = tn;
            std::printf("serve: req %llu ends at %zu tokens (length): every request that holds memory waits to grow and nothing can be freed (%zu requests, %u numbers in the pool)\n",
                        (unsigned long long)victim.rq.h.req_id, victim.out.size(), outs.size(), (unsigned)m_.slot_chunks_total());
            finish(victim, fe::kLength);
          }
        }
      }
      if (fl.empty() && pool_waits > 0 && pfq_.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(20));  // nothing else flies: the pool is asked again shortly
    }
#endif
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
    run_ended_.clear();  // every flight of the run is finished or back in rot_: the freed slots are assigned from here by the real passes
    for (auto it = paused.rbegin(); it != paused.rend(); ++it) rot_.push_front(std::move(*it));
    dec_end_ = now_s();
  }

  // Session S's history for a run that feeds the flight's next token at pos: the positions before it (one
  // more, the token's own, when a run before fed it: it goes in again, with the same KV).
  void rejoin(Session& S, int pos, uint64_t req) {
    if ((int)S.hist.size() < pos)
      die("serve: req %llu: its session holds %zu positions, its next token goes at %d", (unsigned long long)req,
          S.hist.size(), pos);
    prefix::lose_below(&S.ep, (uint64_t)pos, [&] { return new_epoch(); });  // the epoch's second site
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
    room_for_bookmarks(reclaim::run_snaps(pos, c.max_new, o_.snap_every, dspark::draft_rows() + 2), {h.session_id}, h.session_id, bkf(f));
    const double t_dec = now_s();
    issue(c);
    std::vector<int32_t> samples;
    bool ctl = false, cancel = false, pause = false;
    uint32_t word = 0;  // rank 0's control word: bit 0 stops the response, kCtlWatch toggles for the watch, kCtlSplit for the imager's ride
    RideWatch rw;
    RideSplit rs;
    pause_why_ = lendpace::Why::None;
    const DecodeEnd e = decode_loop(m_, c, wd_, [&](int, const model::SampleOut& o, int row) {
      samples.push_back(o.row[row].token);
      if (f.tu.dec1_ms < 0) f.tu.dec1_ms = 1e3 * (now_s() - t_dec);
      if (cancel) return;
      if (Io::cancelled(f.rq)) {
        if (!ctl) m_.set_ctl(word |= 1);  // every rank stops after the step that reads it
        ctl = cancel = true;
        return;
      }
      tokens(f.rq, o, row);  // a paused request's last tokens too: the session holds them
      f.out.push_back(o.row[row].token);
      // a new request: this one pauses for its prefill; while long prompts are prefilled, at its window's end instead
      if (!ctl && pause_wanted(false, false)) {
        m_.set_ctl(word |= 1);
        ctl = pause = true;
      }
    }, [&] {  // once a step: the flips, the STATS frame and the reasons that pause a run (step_hook)
      if (step_hook(rw, rs, word, ctl, 1u)) ctl = pause = true;
    });
    if (word) m_.set_ctl(0);
    ride_watch_end(rw);
    ride_split_run_end(rs);
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
    const uint32_t end = cancel || (e.why == kEndCancel && !pause) ? fe::kCancelled
                         : e.why == kEndStop || stop_id(f, f.out.back()) ? fe::kStop
                         : e.why == kEndContext                          ? window_end(f)
                         : (int)f.out.size() >= max_new                  ? fe::kLength
                         : pause                                         ? kPaused
                                                                         : kTurn;
    if (end == kPaused) pause_note();
    return end;
  }
  // A decode that stopped at its limit: at its factor's window the response goes on at the next factor (cross(), before the
  // flight's next run) when it may generate more and the engine's context holds that window; else the context reason
  uint32_t window_end(Flight& f) {
    if ((int64_t)f.rq.prompt.size() + (int64_t)f.out.size() < cap_of(f.fac) || !crossable(f)) return fe::kContext;
    f.cross = true;
    return kTurn;
  }

  // The flights' run together, each in its own slot: a row of each per step (batch_loop), or with drafts
  // (spec_batch_loop) when the server has them and the batch is small enough for a draft each; with
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
        snaps += reclaim::run_snaps(e[k].pos0, e[k].max_new, o_.snap_every, dspark::draft_rows() + 2);
        run_ids.insert(fl[k].rq.h.session_id);
      }
      room_for_bookmarks(snaps, run_ids, 0, 0);
    }
    const double t_dec = now_s();
    issue(c);
    bcast_payload(ring_, e.data(), e.size() * sizeof(BatchEntry), 0);
    uint32_t ctl = 0;
    RideWatch rw;
    RideSplit rs;
    std::vector<std::vector<int32_t>> samples(n);
    pause_why_ = lendpace::Why::None;
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
        // a cancel ends like a stop or a length end (the contract above: its DONE goes out at once while the others go on): the DONE now, no token after it, its slot counted as freed
        // for the pause check below. The run-end loop sends no second DONE (done_sent); why[k] is kCancelled by cancel[k]. PICO_OLD_RULES: the DONE waited for the run's end.
#ifndef PICO_OLD_RULES
        ended[k] = 1;
        run_ended_.insert(f.rq.h.session_id);
        f.done_sent = true;
        const fe::DoneTail tail = report_of(f, (int64_t)f.rq.prompt.size() + (int64_t)f.out.size() - 1);
        done(f.rq, fe::kCancelled, (uint32_t)f.q, (uint32_t)f.out.size(), f.t_start, f.t_first, "", &tail);
        if (!(ctl & kCtlPause) && pause_wanted(true, true)) m_.set_ctl(ctl |= kCtlPause);
#endif
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
        run_ended_.insert(f.rq.h.session_id);
        f.done_sent = true;
        // the history ends with the last token fed: the prompt and the response but its last token (this run's ids join `hist` below)
        const fe::DoneTail tail = report_of(f, (int64_t)f.rq.prompt.size() + (int64_t)f.out.size() - 1);
        done(f.rq, stop ? fe::kStop : fe::kLength, (uint32_t)f.q, (uint32_t)f.out.size(), f.t_start, f.t_first, "", &tail);
      }
      if (!(ctl & kCtlPause) && pause_wanted(true, ends))
        m_.set_ctl(ctl |= kCtlPause);
    };
    const OnStep step = [&] {  // once a step: the flips, the STATS frame and the reasons that pause a run (step_hook)
      step_hook(rw, rs, ctl, (ctl & kCtlPause) != 0, kCtlPause);
    };
    std::vector<BatchEnd> end(n);
    std::vector<DecodeEnd> dend;
    SpecBatchStats st;
    if (drafts) {
      dend = spec_batch_loop(m_, c, e, wd_, cb, &st, step);
      for (int k = 0; k < n; ++k) end[k].sampled = dend[k].sampled, end[k].fed = dend[k].fed, end[k].why = dend[k].why;
    } else {
      end = batch_loop(m_, c, e, wd_, cb, step);
    }
    if (ctl) m_.set_ctl(0);
    ride_watch_end(rw);
    ride_split_run_end(rs);
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
    if (g_log_detail) {  // DSGB10_LOG_DETAIL: the run's line
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
    for (int k = 0; k < n; ++k)
      if (why[k] == kPaused) {
        pause_note();
        break;
      }
  }

  // Paused requests that their client cancelled, or that the server's stop ends, end now, not at their
  // turn; so do requests waiting for a slot.
  // The dashboard's STATS frame (docs/dashboard.md), at most every half second, from this thread, the only one that
  // touches the model: the chunk pool and its conversations, the disk tier and its counters since the start, the
  // lines, interleaved prefill. The reader answers the front-end's StatsReq with the latest one.
  // Called from the top of the loop while no prompt is queued, from step_hook once a step of a decoding run (in_run: after the next step was
  // queued), from pf_piece between a chunk's launch and its wait and from idle_wait, and, forced, when a tenant's change landed or failed
  // (lend_pace.h): a long response does not freeze it. It reads rank 0's state on the engine thread, nothing a step or a chunk computes.
  void publish(bool in_run = false, double now = -1) {
    const double t = now < 0 ? now_s() : now;
    if (!lendpace::publish_due(t, published_, publish_forced_, in_run)) return;
    publish_forced_ = false;
    published_ = t;
    const model::Model::TierStats ts = m_.tier_stats();
    const model::Model::PoolStats ps = m_.pool_stats();
    fe::StatsFrame sf{};
    sf.version = fe::kStatsVersion, sf.n_fields = fe::kStatsFields;
    sf.pool_numbers = ps.numbers, sf.pool_free = ps.free, sf.pool_held = ps.held, sf.pool_shared = ps.shared;
    sf.pool_reserved = ps.reserved, sf.number_bytes = ps.number_bytes;
    sf.image_numbers = ps.image_numbers;
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
    if (m_.engram_lend_enabled()) {
      const model::Model::EngramLendInfo li = m_.engram_lend_info();
      sf.engram_cache_bytes = li.slots * engram::kRowBytes, sf.engram_groups = li.groups;
      sf.engram_pending_mib = li.running ? li.pending_mib : (int64_t)li.mib != engram_goal_mib_ ? (uint64_t)engram_goal_mib_ : 0;
      sf.engram_lookups = li.lookups, sf.engram_hits = li.hits;
      sf.engram_lend_ok_total = li.ok_total, sf.engram_lend_failed_total = li.failed_total;
      sf.engram_huge_pct = li.huge_pct, sf.engram_evictions = li.evictions;
    }
    sf.lent_owed_groups = (uint64_t)std::max(0, m_.window_owed());
    {
      const MemWatchView mw = mem_watch_view();
      sf.store_numbers = o_.store_numbers, sf.mem_avail_min_bytes = mw.min_bytes, sf.mem_avail_low_count = mw.low_count;
      sf.ids_host_bytes = ids_bytes(), sf.ids_file_sessions = (uint64_t)std::max<int64_t>(0, ids_files_);
      for (int r = 0; r < 4; ++r) sf.mem_avail_bytes[r] = mw.avail[r];
    }
    for (const model::Model::TenantRow& r : m_.tenant_rows()) {
      if (r.id < 8) sf.tenant_groups[r.id] = (uint64_t)r.groups;
      else sf.tenant_models_groups += (uint64_t)r.groups;
    }
    {
      const model::Model::SplitStats sp = m_.split_stats();
      sf.kv_split = o_.kv_split ? 1 : 0, sf.conv_split = sp.conv_split, sf.quarter_slots = sp.quarter_slots, sf.quarter_numbers = sp.quarter_numbers;
      sf.ghost_numbers = sp.ghost_numbers, sf.split_pieces_total = sp.split_pieces, sf.split_deferred_total = sp.split_deferred;
      sf.revives_total = sp.revives, sf.joins_total = sp.joins, sf.join_bytes_total = sp.join_bytes, sf.join_ns_total = sp.join_ns;
      sf.join_max_ns = sp.join_max_ns, sf.lend_busy_total = lend_busy_total_, sf.join_short_total = sp.join_short;
      sf.split_level = (uint64_t)split_level_.level, sf.split_pressure_pm = (uint64_t)(split_level_.pressure * 1000 + 0.5);
      sf.split_age_now_s = (uint64_t)(split_level_.age_s + 0.5), sf.split_demote_pps = demote_pps();
      sf.l2_numbers = sp.quarter_numbers, sf.l2_dropped_total = l2_dropped_total_;
      sf.clean_numbers = sp.clean_numbers, sf.image_pending = sp.image_pending, sf.split_rides_total = sp.rides;
      sf.returns_read_total = sp.returns_read, sf.image_waits_total = sp.image_waits, sf.split_reader_waits_total = reader_waits_total_;
    }
    if (model::copy::mode() & model::copy::kAdapt) {
      sf.copy_adapt = 1, sf.copy_proposals_total = copy_proposals_total_, sf.copy_scored_total = copy_scored_total_;
      sf.copy_first_right_total = copy_first_right_total_, sf.copy_rows_verified_total = copy_rows_verified_total_;
      sf.copy_rows_accepted_total = copy_rows_accepted_total_, sf.copy_failing_requests_total = copy_failing_total_;
      for (int b = 0; b < 4; ++b) sf.copy_chance_pm[b] = ((uint64_t)copy_chance_[b] * 1000 + 32767) / 65535;
    }
    io_.publish_stats(Frame(fe::kStats).add(&sf).take());
  }
  double published_ = 0;
  bool publish_forced_ = false;  // a tenant's change landed or failed: the next publish is not rate-limited

  // The memory watch (mem_watch.h): every memwatch::kEveryS seconds, between commands, one kStMemWatch on every rank. The lines the
  // server itself knows to be over their cap are named in its report.
  void watch_tick() {
    const double t = now_s();
    if (!memwatch::due(mem_watch_on(), t, watch_at_)) return;
    if (memwatch::defer_to_chunk(!pfq_.empty() && pfq_.front()->begun, t, watch_at_)) return;  // the prompt's chunks carry it (kPfWatch)
    watch_at_ = t;
    mem_watch_note(watch_over());
    Cmd c{};  // not state(): no digest line from the server either
    c.op = kCmdState;
    c.n_ops = 1;
    c.ops[0] = StateOp{kStMemWatch, -1, 0, 0};
    issue(c);
    watch_line("as a command between turns");
  }
  double watch_at_ = 0;
  // DSGB10_STORE_CHECK=1: one line a reading that does not ride a decoding run, saying where it ran (tests/watch_rides_test.py prompt reads them)
  void watch_line(const char* where) {
    static const bool on = std::getenv("DSGB10_STORE_CHECK") && std::atoi(std::getenv("DSGB10_STORE_CHECK")) == 1;
    if (on) std::printf("serve: memory watch read %s\n", where);
  }
  // The lines the server itself knows to be over their cap, named in the watch's report.
  std::string watch_over() {
    std::string over;
    if (o_.ids_dir.size() && ids_bytes() > o_.ids_host_bytes) {
      char b[128];
      std::snprintf(b, sizeof b, "rank 0 server token ids %.2f GiB > %.2f GiB", ids_bytes() / 1073741824.0, o_.ids_host_bytes / 1073741824.0);
      over = b;
    }
    return over;
  }
  // The watch riding a decoding run (mem_watch.h): from a step callback, when the watch is due and no pause is being asked, rank 0 flips
  // kCtlWatch in its control word `word`; every rank reads the memory after the step that shows it, and the run goes on. The watch counts as
  // done when rank 0's reading count moved: only then its time is set and the keeper decides on the reading. A flip no step read (the run
  // ended first) leaves the watch due, for watch_tick.
  struct RideWatch {
    bool flipped = false;
    uint64_t seq = 0;  // the reading count at the flip
  };
  void ride_watch(RideWatch& r, uint32_t& word, bool pausing, double t) {
    if (!mem_watch_on()) return;
    ride_watch_end(r, t);
    if (!memwatch::ride_due(true, t, watch_at_, pausing, r.flipped)) return;
    mem_watch_note(watch_over());
    r.flipped = true, r.seq = mem_watch_view().seq;
    word ^= kCtlWatch;
    m_.set_ctl(word);
  }
  void ride_watch_end(RideWatch& r, double t = -1) {
    if (!r.flipped || mem_watch_view().seq == r.seq) return;
    r.flipped = false;
    watch_at_ = t < 0 ? now_s() : t;
  }

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

  // The epoch and the size a DONE of flight f tells the front-end: its kept session's entry at its factor, `n_hist` (-1: what the
  // entry holds) recorded as reported. Epoch 0, size 0: the session is not kept (no kFlagKeep, released, or none made).
  fe::DoneTail report_of(const Flight& f, int64_t n_hist) {
    fe::DoneTail t{};
    if (!(f.rq.h.flags & fe::kFlagKeep) || later_.count(f.rq.h.session_id)) return t;
    Session* S = find_at(f.rq.h.session_id, f.fac);
    if (!S) return t;
    const uint64_t n = n_hist >= 0 ? (uint64_t)n_hist : (uint64_t)S->hist.size();
    prefix::reported_at(&S->ep, n);
    t.epoch = S->ep.value, t.n_hist = (uint32_t)n;
    return t;
  }

  // Every way a request ends (finish): its image rows go back to the pool, and an image it left between two block ranges is no longer
  // mid-way, so the next request's image starts from its block 0.
  void end_images(Flight& f) {
    if (imghold::free_at_end(f.img)) {
      state({{kStImgFree, f.img.key, 0, 0}});
      f.img.key = 0;
    }
    enc_mark_.ended(f.rq.h.req_id);
    unit_line(f.rq.h.req_id);
    if (pend_.req == f.rq.h.req_id) pend_.kind = -1;
  }

  // DONE, then the session kept (its snapshots thinned) or freed, and the store held to its budget; the
  // requests held for the session go on.
  void finish(Flight& f, uint32_t why) {
    const fe::SubmitHeader& h = f.rq.h;
    if (!f.done_sent) {
      const fe::DoneTail tail = report_of(f, -1);
      done(f.rq, why, (uint32_t)f.q, (uint32_t)f.out.size(), f.t_start, f.t_first, f.err, &tail);
    }
    Io::forget(f.rq);
    end_images(f);
    const double t_end = now_s();
    flying_.erase(h.session_id);
    const bool released = later_.erase(h.session_id) > 0;  // the front-end let it go meanwhile
    forget_evicted();
    Session* S = (h.flags & fe::kFlagKeep) && !released ? find_at(h.session_id, f.fac) : nullptr;
    if (S && S->hist.size() < S->ep.reported) {  // a DONE told more than the history holds: its epoch is not true any more
      std::printf("serve: BUG: req %llu: the history holds %zu positions, its DONE reported %llu; the epoch changes\n", (unsigned long long)h.req_id,
                  S->hist.size(), (unsigned long long)S->ep.reported);
      prefix::lose_below(&S->ep, (uint64_t)S->hist.size(), [&] { return new_epoch(); });
    }
    if (S) thin(*S);
    win_retry_at_ = 0;
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
    if (S) S->t_fin = now_s();  // its idle age counts from here
    if (S) split_note(S->h);  // its split is a command of its own, between runs (split_tick)
    const model::Model::StateInfo si = S ? m_.state_info(S->h) : model::Model::StateInfo{};
    const DecodeEnd* de = &f.de;
    if (de->copy_proposals > 0) {  // copy drafting's learner: what the request's proposals came to (rank 0's counts; every rank learns the same)
      copy_proposals_total_ += (uint64_t)de->copy_proposals, copy_scored_total_ += (uint64_t)de->copy_scored;
      copy_first_right_total_ += (uint64_t)de->copy_first_right, copy_rows_verified_total_ += (uint64_t)de->copy_rows_verified;
      copy_rows_accepted_total_ += (uint64_t)de->copy_rows_accepted;
      if (de->copy_scored >= 16 && 2 * de->copy_first_right < de->copy_scored) ++copy_failing_total_;
      std::copy(de->copy_chance, de->copy_chance + 4, copy_chance_);
    }
    if (g_log_detail && de->vsteps) {
      char h67[48] = "";  // only with a longer block, so the default log line is as before
      if (dspark::draft_rows() > dspark::kDraftModel)
        std::snprintf(h67, sizeof h67, " 6..7: %d %d;", de->rows_hist[6], de->rows_hist[7]);
      std::printf("serve: req %llu drafts: %d verify steps, %.2f drafts verified and %.2f accepted per step, "
                  "steps verifying 0..5: %d %d %d %d %d %d;%s gate %.3f (%.3f-%.3f); GPU ms by rows:",
                  (unsigned long long)h.req_id, de->vsteps, (double)de->verified / de->vsteps,
                  (double)de->accepted / de->vsteps, de->rows_hist[0], de->rows_hist[1], de->rows_hist[2],
                  de->rows_hist[3], de->rows_hist[4], de->rows_hist[5], h67, de->gate_sum / std::max(1, de->gate_n),
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
    if (g_log_detail)
      std::printf("serve: req %llu turn: state ops %.1f ms (GPU %.1f), %d chunks: layers 0-19 %.1f ms, replay %.1f ms, "
                "head %.1f ms, draft windows %.1f ms, rows %.1f ms; first token to decode %.1f ms, first decode step "
                "%.1f ms\n",
                (unsigned long long)h.req_id, u.ops_ms, u.ops_gpu_ms, u.chunks, u.pf[0], u.pf[1], u.pf[2], u.pf[3],
                u.rows_ms, u.tail_ms, u.dec1_ms);
    if (g_log_detail && u.jn_on)
      std::printf("serve: req %llu joins: %d chunks: attention %.1f ms, MoE %.1f ms (%d pairs), floor %.3f ms a pair\n",
                  (unsigned long long)h.req_id, u.jn_chunks, u.jn[0], u.jn[1], u.jn_pairs,
                  u.jn[2] / std::max(1, u.jn_chunks));
    if (g_log_detail) {  // the chunk pool: the conversations no line holds, and what the lines hold and have reserved
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
  double ext_since_ = 0, ext_end_last_ = 0;  // when the standstill began; when a request was last ended for it
  bool ext_noted_ = false;
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
        if (factor_for((int64_t)r.len()) == factor) return true;
    return false;
  }
  // The pool's tenants change size through one queue (Model::tenant_lend): a tenant's goal is asked of the model at a run boundary
  // (kCmdLend begin) and the change finished when every rank's body has (kCmdLend poll, every 100 ms until then). The Engram cache's goal
  // is the setting, or lower while the cache gave way; a test build's dummy tenant 15 has CONFIG's test_tenant_groups. A refused or
  // failed size is not asked for again for five minutes unless the setting changes; a request that finds the pool's runners in the way
  // is asked again in a second. What a closed window or a finished change owes the pool is settled here too, whatever the Engram cache
  // does: while a change is in flight no window command and no settle runs (the window waits, ensure_window).
  // The tenant whose change the server would begin at time t (-1: none): the Engram cache's size is not its goal, else tenant 15's.
  int lend_begin_id(double t) {
    if (!m_.engram_lend_enabled() && !m_.tenant_other_enabled()) return -1;
    const int64_t cur = m_.engram_lend_enabled() ? (int64_t)m_.engram_lend_info().mib : engram_goal_mib_;
    if (cur != engram_goal_mib_ && t >= engram_next_ && !engram_backoff_.blocked((uint64_t)engram_goal_mib_, t, engram_gen_))
      return model::lend::kTenantEngram;
    loan_policy(t);
    if (loan_enabled() && loan_goal_ >= 0 && (loan_goal_ == 1) != loan_held() && (loan_goal_ == 1 || !loan_in_use()) && t >= loan_next_ &&
        !loan_backoff_.blocked(loan_goal_ ? (uint64_t)loan_mib_ : 0, t, loan_gen_))
      return model::lend::kTenantPrefill;
    if (tenant_goal_mib_ >= 0 && tenant_size_mib(15) != tenant_goal_mib_ && t >= tenant_next_ &&
        !tenant_backoff_.blocked((uint64_t)tenant_goal_mib_, t, tenant_gen_))
      return 15;
    return -1;
  }
  // ---- the split (--kv-split cache; split_pace.h; docs/engine.md) ----
  // A conversation whose request ended is split by commands of its own between runs, and comes back by commands of its own before
  // anything reads or writes it. The list holds store handles: a conversation leaves it when a request names it, or when a command
  // split all it could.
  std::vector<int> split_pending_;
  double split_retry_ = 0;
  bool split_waits_ = false;       // the last split command left pieces whole for want of a number
  double split_start_ = now_s();   // the server's start
  splitpace::Marks split_marks_;   // the spare numbers' marks and the ages (init_split_marks)
  splitpace::Level split_level_;   // the last pass's level (STATS)
  uint64_t l2_dropped_total_ = 0;  // conversations given back from L2 to the SSD
  std::deque<std::pair<double, int>> demoted_;  // (time, pieces) of the commands of the last 10 s: the demotion rate (STATS)
  uint64_t lend_busy_total_ = 0;   // window and tenant changes that answered busy (STATS)
  // copy drafting's learner (STATS): proposals made and scored, the first token right, rows verified and accepted, requests whose copies
  // failed, and the shared first-token chance by bucket (1/65535) as the last finished request's last step showed it
  uint64_t copy_proposals_total_ = 0, copy_scored_total_ = 0, copy_first_right_total_ = 0, copy_rows_verified_total_ = 0;
  uint64_t copy_rows_accepted_total_ = 0, copy_failing_total_ = 0;
  uint32_t copy_chance_[4] = {};
  bool join_incomplete_ = false;  // unspill's join left a list entry below `upto` a slot: the request waits as the units of its read
  int fork_base_ = -1;  // a fork's base, put back on the pending list when the fork's setup is done
  il::JoinCost jc_;                // the per-piece times of a return: taken back, exchanged, read from the SSD (interleave.h)
  JoinAsk join_ask(uint64_t sid, int fac, uint64_t base, int need, const std::vector<int32_t>& prompt) {
    JoinAsk a;
    uint64_t id = sid;
    Session* x = find_at(sid, fac, &id);
    if (!x && base) id = base, x = find_at(base, fac, &id);
    if (!x) return a;
    const RestoreAsk ra = restore_ask(sid, base, need, &prompt);
    if (!ra.found || ra.src_id != id) return a;
    a.found = true, a.h = x->h, a.key = id, a.upto = ra.upto, a.disk = ra.on_disk;
    m_.state_join_plan(x->h, ra.upto, &a.back, &a.exch);
    if (o_.imager) a.disk = std::max(0, a.disk - a.back);  // the ghosts are in `back` (taken back), not read from the SSD
    a.est_ms = jc_.est_ms(a.back, a.exch, a.disk);  // the pieces on the SSD are in the estimate, and units of the return like the rest (join_unit)
    return a;
  }
  // a request's join is in flight: no other request is admitted, no tenant change, settle or split command runs until it ends
  bool join_active() const {
    for (const auto& L : pfq_)
      if (L->joining && L->jw.live) return true;  // true only from a unit that made progress to the next
    return false;
  }
  void split_note(int h) {
    uint64_t key = 0;
    const Session* x = h > 0 ? session_of(h, &key) : nullptr;
    if (x && in_flight(key, *x)) return;  // a request holds it: finish() notes it again
    if (!(o_.kv_split || o_.imager) || h <= 0 || std::find(split_pending_.begin(), split_pending_.end(), h) != split_pending_.end()) return;
    split_pending_.push_back(h);
  }
  void split_forget(int h) {
    split_pending_.erase(std::remove(split_pending_.begin(), split_pending_.end(), h), split_pending_.end());
  }
  Session* session_of(int h, uint64_t* key) {
    for (auto& [k, x] : sess_)
      if (x.h == h) return *key = k, &x;
    return nullptr;
  }
  // The marks: --kv-split-high (default the room `off` keeps between requests, in numbers) and --kv-split-low (default half of it).
  void init_split_marks(int64_t room_numbers) {
    split_marks_.high = o_.kv_split_high > 0 ? o_.kv_split_high : room_numbers;
    split_marks_.low = o_.kv_split_low > 0 ? std::min(o_.kv_split_low, split_marks_.high) : split_marks_.high / 2;
    split_marks_.after_s = o_.kv_split_after_s, split_marks_.pass_ms = o_.kv_split_pass_ms;
  }
  // The pass: at the top of the loop, between decoding runs and never inside one. While the pool's pressure and the candidate's idle age
  // say so (split_pace.h): the least recently used listed conversation no request uses is demoted by one command (at most
  // --kv-split-budget pieces), or, under the low mark, the least recently used conversation holding quarter slots is given back to the
  // SSD (the slots have their images: no write). It goes on until the rule says nothing, a deferral holds the list, or the pass's time is
  // used. A conversation a request runs on leaves the list: finish() puts it back.
  void split_tick() {
    if (o_.imager) return imager_tick();
    if (!o_.kv_split || join_active()) return;
    const splitpace::Mode mode = (splitpace::Mode)o_.kv_split_mode;
    const bool flights = !flying_.empty() || !rot_.empty() || !pfq_.empty();
    splitpace::Level lv = splitpace::level(mode, m_.slot_spare(), split_marks_, flights);
    split_level_ = lv;
    if (lv.level == 0 || (lv.level == 1 && split_pending_.empty())) return;
    const double t0 = now_s();
    while (true) {
      const int64_t spare = m_.slot_spare();
      lv = splitpace::level(mode, spare, split_marks_, flights);
      split_level_ = lv;
      if (lv.level == 0) return;
      // the candidate for a demotion: the list's least recently used conversation no request is in flight on
      std::vector<splitpace::Cand> cands;
      for (size_t i = 0; i < split_pending_.size();) {
        uint64_t key = 0;
        Session* x = session_of(split_pending_[i], &key);
        if (!x || in_flight(key, *x)) {
          split_pending_.erase(split_pending_.begin() + i);
          continue;
        }
        cands.push_back({x->h, x->used, false});
        ++i;
      }
      const int h = splitpace::pick(cands);
      uint64_t key = 0;
      Session* x = h >= 0 ? session_of(h, &key) : nullptr;
      const double idle = x ? now_s() - std::max(x->t_fin, split_start_) : -1;  // a conversation no request finished since the start counts from the start
      int h2 = -1;  // the level 2 victim: the least recently used conversation holding quarter slots that no request uses
      if (lv.level == 2) {
        uint64_t u2 = 0;
        for (auto& [k, y] : sess_)
          if (!in_flight(k, y) && (h2 < 0 || y.used < u2) && m_.state_info(y.h).slots > 0) h2 = y.h, u2 = y.used;
      }
      const double t = now_s();
      const bool hold = splitpace::held(split_waits_, split_waits_ && m_.split_can_open(), t, split_retry_);
      const splitpace::Act act = splitpace::next(lv, h2 >= 0, idle, hold);
      if (act == splitpace::Act::kNothing) return;
      if (act == splitpace::Act::kDropL2) {
        drop_l2(h2, spare);
      } else {
        const int budget = std::max(1, o_.kv_split_budget);
        const model::Model::SplitStats a = m_.split_stats();
        const bool could_open = m_.split_can_open();
        {  // a command of its own: every session a request is in flight on is named live, or nothing is sent now
          Cmd c{};
          c.op = kCmdState;
          c.n_ops = 1;
          c.ops[0] = {kStSplit, h, budget, 0};
          if (!issue(c, true)) {
            split_retry_ = t + 5;
            return;
          }
          digest("state");
        }
        const model::Model::SplitStats b = m_.split_stats();
        const int now_split = (int)(b.split_pieces - a.split_pieces), deferred = (int)(b.split_deferred - a.split_deferred);
        if (now_split > 0) demoted_.push_back({t, now_split});
        if (now_split > 0 || deferred > 0)
          std::printf("serve: session %016llx split: %d pieces%s; %llu quarter slots in %llu numbers, %llu ghosts (level %d, pressure %.2f, idle %.0f s of %.0f s)\n",
                      (unsigned long long)key, now_split,
                      deferred ? (", " + std::to_string(deferred) + " left whole for want of a quarter number").c_str() : "",
                      (unsigned long long)b.quarter_slots, (unsigned long long)b.quarter_numbers, (unsigned long long)b.ghost_numbers, lv.level,
                      lv.pressure, idle, lv.age_s);
        const splitpace::Outcome o = splitpace::after(now_split, budget, deferred, t);
        if (o.done) split_pending_.erase(std::remove(split_pending_.begin(), split_pending_.end(), h), split_pending_.end());
        else split_retry_ = o.retry_at;
        // a command that deferred pieces waits for a number to open; one sent with a number open that split nothing waits the plain retry
        split_waits_ = deferred > 0 && !(could_open && now_split == 0);
      }
      if (!splitpace::go_on(1e3 * (now_s() - t0), lv.pass_ms)) return;
    }
  }
  // ---- the imager: conversations are imaged to each Spark's SSD as quarters and then marked, never in the way of a request.
  // Nothing between a prompt's chunks, at an admission, in finish() or in persist_slot: an idle command when nothing is in flight, queued,
  // waiting or arriving; inside a decoding run, rides (a bit of the step's control word, one host all-gather, the store's bookkeeping between two steps).
  // Work to do: a conversation with pieces that have no image, or image commands not settled yet (a settle marks).
  // A conversation is imaged once it has been idle for --kv-split-after-s (default 0, every mode); a request waiting for clean blocks
  // does not wait for the age.
  bool imager_aged(const Session& x) const {
    return splitpace::pool_waiter(!wait_.empty(), pool_waits_, pf_waits_) || o_.kv_split_after_s <= 0 || now_s() - std::max(x.t_fin, split_start_) >= o_.kv_split_after_s;
  }
  bool imager_work() {
    if (m_.image_pending() > 0) return true;
    for (const int h : split_pending_) {
      uint64_t key = 0;
      const Session* x = session_of(h, &key);
      if (x && !in_flight(key, *x) && imager_aged(*x) && m_.image_todo(h) > 0) return true;
    }
    return false;
  }
  // The conversation to image next: the least recently used one on the list that no request is in flight on, off its line first; those with nothing
  // left to queue leave the list (the settle that marks them is the work that remains). -1: none.
  int imager_pick() {
    int best = -1;
    bool best_off_line = false;
    uint64_t used = 0;
    for (size_t i = 0; i < split_pending_.size();) {
      uint64_t key = 0;
      Session* x = session_of(split_pending_[i], &key);
      if (!x || m_.image_todo(x->h) == 0) {
        split_pending_.erase(split_pending_.begin() + i);
        continue;
      }
      if (!in_flight(key, *x) && imager_aged(*x)) {
        const bool off_line = x->slot < 0;
        if (best < 0 || (off_line && !best_off_line) || (off_line == best_off_line && x->used < used)) best = x->h, best_off_line = off_line, used = x->used;
      }
      ++i;
    }
    return best;
  }
  // the sessions a request is in flight on, for a ride's `live` (false: they do not all fit in the list)
  bool imager_live(int32_t* live) const {
    int n = 0;
    for (const auto& [k, x] : sess_) {
      if (!in_flight(k, x)) continue;
      if (n >= model::kMaxSlots) return false;
      live[n++] = x.h;
    }
    return true;
  }
  // The idle pass: at the top of the loop, with nothing in flight, queued, waiting or arriving (splitpace::idle_ok, asked before each command),
  // ride commands back to back (--kv-split-budget pieces each) for at most 250 ms. Between a prompt's chunks, at an admission, in finish() and in
  // persist_slot it returns at once.
  void imager_tick() {
    if (rot_.empty()) stalled_ = false, pool_waits_ = 0;  // nothing in the rotation: no stall, no flight waits for the pool
    if (pfq_.empty()) pf_waits_ = false;
    {  // one line per request that waits at its start while nothing in flight can step; the stall line keeps its text and its own flag
      const bool aw = splitpace::all_wait(!rot_.empty(), stalled_, !pfq_.empty(), pf_reading(), pf_waits_);
      if (aw && pf_waits_ && !pfq_.empty() && !join_active() && pf_wait_noted_ != pfq_.front()->f.rq.h.req_id) {
        pf_wait_noted_ = pfq_.front()->f.rq.h.req_id;
        size_t convs = 0;
        for (const int h : split_pending_) convs += m_.image_todo(h) > 0;
        if (imager_work())
          std::printf("serve: req %llu waits at its start for the pool and nothing in flight can step (%zu paused, %zu waiting for a slot): the imager makes room (%llu image jobs queued, %zu conversations to image)\n",
                      (unsigned long long)pf_wait_noted_, rot_.size(), wait_.size(), (unsigned long long)m_.image_pending(), convs);
        else
          std::printf("serve: req %llu waits at its start for the pool and nothing in flight can step (%zu paused, %zu waiting for a slot): nothing can be imaged: it waits for a request to end\n",
                      (unsigned long long)pf_wait_noted_, rot_.size(), wait_.size());
      }
    }
    if (!stalled_) stall_noted_ = false;
    if (join_active() || !imager_work()) {
      if (stalled_ && !stall_noted_ && !join_active()) {
        stall_noted_ = true;
        std::printf("serve: every request in flight waits for the pool (%zu paused, %zu waiting for a slot): nothing can be imaged: they wait for a request to end\n", rot_.size(), wait_.size());
      }
      return;
    }
    if (stalled_ && !stall_noted_) {
      stall_noted_ = true;
      size_t convs = 0;
      for (const int h : split_pending_) convs += m_.image_todo(h) > 0;
      std::printf("serve: every request in flight waits for the pool (%zu paused, %zu waiting for a slot): the imager makes room (%llu image jobs queued, %zu conversations to image)\n",
                  rot_.size(), wait_.size(), (unsigned long long)m_.image_pending(), convs);
    }
    const auto idle = [&] {
      return splitpace::idle_ok(!flying_.empty(), !rot_.empty(), splitpace::pfq_busy(!pfq_.empty(), pf_reading(), pf_waits_), !ready_.empty(), !wait_.empty(), io_.submit_waiting(), true,
                                splitpace::all_wait(!rot_.empty(), stalled_, !pfq_.empty(), pf_reading(), pf_waits_));
    };
    if (pfq_.empty()) reader_wait_noted_ = false;
    if (splitpace::reader_wait(!rot_.empty(), pf_reading(), !wait_.empty(), true) && !reader_wait_noted_) {  // a prompt read alone: no decoder in the rotation (flying_ holds the prompt itself)
      reader_wait_noted_ = true;  // The rule: nothing between a prompt's chunks. The request waits for the reader's end, said once per reader
      ++reader_waits_total_;
      std::printf("serve: a request waits for unwritten quarters while a prompt is read alone (%llu image jobs queued, %zu conversations to image): it waits for the reader's end\n",
                  (unsigned long long)m_.image_pending(), split_pending_.size());
    }
    const double t0 = now_s();
    while (idle() && imager_work()) {
      const int h = imager_pick();
      const model::Model::SplitStats a = m_.split_stats();
      Cmd c{};
      c.op = kCmdState;
      c.n_ops = 1;
      c.ops[0] = {kStRide, h, std::max(1, o_.kv_split_budget), 0};
      if (!imager_live(c.live)) return;
      bcast(ring_, c);
      wd_.started = now_s();  // a ride is watched as a step is: a rank that waits in it for a partner ends with the watchdog's line
      apply_ops(m_, c);
      wd_.started = 0;
      digest("ride");
      const model::Model::SplitStats b = m_.split_stats();
      const int imaged = (int)(b.imaged - a.imaged), marked = (int)(b.split_pieces - a.split_pieces);
      if (imaged > 0 || marked > 0 || h > 0) {
        if (marked > 0) demoted_.push_back({now_s(), marked});
        uint64_t key = 0;
        if (h > 0) session_of(h, &key);
        // the counts are the settle's of the earlier commands; the conversation is the one this command queued
        char q[40] = "none";
        if (h > 0) std::snprintf(q, sizeof q, "%016llx", (unsigned long long)key);
        if (g_log_detail)
          std::printf("serve: split: settled: %d marked, %d imaged (idle); queued %s; %llu ghosts\n", marked, imaged, q, (unsigned long long)b.ghost_numbers);
      }
      if (1e3 * (now_s() - t0) >= 250) return;
      if (h < 0 && b.image_pending == a.image_pending && imaged == 0 && marked == 0) {  // a settle that could settle nothing yet: the image streams are still at it
        if (m_.image_pending() == 0) return;
      }
    }
  }
  // The ride: from a step callback, when one is due (splitpace::ride_due) and no pause is being asked, rank 0 asks for the request and flips
  // kCtlSplit in its control word `word`; every rank runs split_ride after the step that shows it, and the run goes on. A flip no step read leaves
  // the work pending for the next one.
  struct RideSplit {
    bool flipped = false;
    uint64_t count = 0;                 // the ride count at the flip
    uint64_t rides0 = 0, imaged0 = 0, marked0 = 0;  // at the run's start: the run's end line
    bool started = false;
  };
  double split_ride_at_ = 0, split_check_at_ = 0;
  int pool_waits_ = 0;        // flights the last pass of run() put back for the pool
  bool pf_waits_ = false;     // the front of pfq_ has not begun and waits for the pool (splitpace::pf_start_waits)
  uint64_t pf_wait_noted_ = 0;  // the request whose `waits at its start ... nothing in flight can step` line was printed last
  // some queued long prompt is being prefilled: begun, or its return's units are in flight (a command, then the decoders' window)
  bool pf_reading() const {
    for (const auto& L : pfq_)
      if (L->begun || L->joining) return true;
    return false;
  }
  bool stalled_ = false;      // the rotation cannot move (splitpace::rot_stalled)
  bool stall_noted_ = false;
  uint64_t reader_waits_total_ = 0;
  bool reader_wait_noted_ = false;
  void ride_split(RideSplit& r, uint32_t& word, bool pausing, double t) {
    if (!o_.imager) return;
    if (!r.started) {
      const model::Model::SplitStats a = m_.split_stats();
      r.rides0 = a.rides, r.imaged0 = a.imaged, r.marked0 = a.split_pieces, r.started = true;
    }
    ride_split_end(r, t);
    if (r.flipped || pausing || t - split_ride_at_ < o_.kv_split_ride_ms / 1e3 || t - split_check_at_ < 0.05 || !imager_work()) return;
    split_check_at_ = t;  // the readings below walk the store's records: at most every 50 ms, not every token row
    RideReq q;
    if (!imager_live(q.live)) return;
    std::vector<int> live;
    for (const int32_t h : q.live)
      if (h > 0) live.push_back(h);
    // the blocks that can be given back without a write: the spare numbers, and the clean pieces of the conversations no request runs on
    const int64_t ready = m_.slot_spare() + (int64_t)m_.state_giveable(live);
    const bool waiter = splitpace::pool_waiter(!wait_.empty(), pool_waits_, pf_waits_);
    if (!splitpace::ride_due((splitpace::Mode)o_.kv_split_mode, ready, split_marks_, true, t, split_ride_at_, pausing, r.flipped, (double)o_.kv_split_ride_ms, waiter)) return;
    q.h = imager_pick();
    q.budget = splitpace::ride_pieces(ready, split_marks_, o_.kv_split_ride_pieces, waiter);
    split_ride_set(q);
    r.flipped = true, r.count = split_ride_count();
    word ^= kCtlSplit;
    m_.set_ctl(word);
  }
  void ride_split_end(RideSplit& r, double t = -1) {
    if (!r.flipped || split_ride_count() == r.count) return;
    r.flipped = false;
    split_ride_at_ = t < 0 ? now_s() : t;
  }
  // the run's end: one line, when the run carried rides
  void ride_split_run_end(RideSplit& r) {
    if (!o_.imager || !r.started) return;
    ride_split_end(r);
    const model::Model::SplitStats b = m_.split_stats();
    if (b.rides > r.rides0) {
      uint64_t longest = 0;
      split_ride_take_longest(&longest);
      std::printf("serve: split rides in the run: %llu rides, %llu imaged, %llu marked, longest %.2f ms\n", (unsigned long long)(b.rides - r.rides0),
                  (unsigned long long)(b.imaged - r.imaged0), (unsigned long long)(b.split_pieces - r.marked0), longest / 1e6);
    }
  }
  // A conversation holding quarter slots goes to the SSD: what `evict` does for its victim (a line it holds goes first, then kStSpill).
  void drop_l2(int h, int64_t spare_before) {
    uint64_t key = 0;
    Session* x = session_of(h, &key);
    if (!x) return;
    if (x->slot >= 0) persist_slot(x->slot, true);
    state({{kStSpill, h, 0, 0}});
    split_forget(h);
    forget_evicted();
    ++l2_dropped_total_;
    const int64_t spare = m_.slot_spare();
    std::printf("serve: session %016llx dropped from L2: %lld numbers back (spare %lld, low %lld)\n", (unsigned long long)key,
                (long long)(spare - spare_before), (long long)spare, (long long)split_marks_.low);
  }
  // pieces demoted a second over the last 10 s
  uint64_t demote_pps() {
    const double t = now_s();
    while (!demoted_.empty() && t - demoted_.front().first > 10) demoted_.pop_front();
    int n = 0;
    for (const auto& d : demoted_) n += d.second;
    return (uint64_t)(n / 10);
  }
  // The lists of conversation h below `upto` that are still quarter slots (the bookmarks are pf_begin's own call)
  int lists_split(int h, int upto) const {
    int in = 0, disk = 0, sp = 0;
    m_.state_list_numbers(h, upto, &in, &disk, &sp);
    return sp;
  }
  // One unit of a deferred return: after `give_back` says the pieces' numbers are there (as admission counted them: the idle lines'
  // reservations are still reserved until it does), one join command carrying exactly the lists still split below `upto`. 0: every list
  // below `upto` is whole; 1: it made progress; 2: no number yet (the request waits, nothing is cut, nothing else is held up); 3: the pool
  // does not give the SSD's pieces without a running request's line, so they are not units (the request's begin reads them).
  int lists_disk(int h, int upto) const {
    int in = 0, disk = 0, sp = 0;
    m_.state_list_numbers(h, upto, &in, &disk, &sp);
    return disk;
  }
  int join_unit(const JoinAsk& ja, uint64_t req_id, uint64_t sid, uint64_t base) {
    const int rest = lists_split(ja.h, ja.upto);
    const int disk = lists_disk(ja.h, ja.upto);
    const il::ReturnUnit ru = il::return_unit(disk, rest, o_.kv_join_pieces);
    if (ru.read > 0) {  // the pieces on the SSD first, at most J in a command: a read-back of lists below `upto` (the bookmarks are pf_begin's)
      if (!il::return_fits(disk, return_room(sid, base))) return 3;  // asked before every unit: the room is what is left of the lists
      if (!give_back(ru.read, {}, sid, base, -1, 0, "a conversation read back", false, false)) return 2;
      const double t0 = now_s();
      state({{kStUnspill, ja.h, (int32_t)ru.read, ja.upto}});  // the count: the lists this command brings, never 0 (0 is no limit)
      m_.drain();
      const int left = lists_disk(ja.h, ja.upto);
      jc_.learn_disk(1e3 * (now_s() - t0), std::max(0, disk - left));
      if (left == 0 && lists_split(ja.h, ja.upto) == 0) return 0;
      return left < disk ? 1 : 2;
    }
    if (rest == 0) {
      std::printf("serve: req %llu: the return of session %016llx to position %d is whole\n", (unsigned long long)req_id, (unsigned long long)ja.key, ja.upto);
      return 0;
    }
    if (!give_back(std::min(rest, o_.kv_join_pieces), {}, sid, base, -1, 0, "a conversation joined", false)) return 2;
    const model::Model::SplitStats a = m_.split_stats();
    const double t0 = now_s();
    const int prev = m_.state_info(ja.h).slots;
    state({{kStJoin, ja.h, ja.upto, (int32_t)rest}});  // the count: what admission counted, never 0 (0 is no limit)
    const int cur = m_.state_info(ja.h).slots;
    const model::Model::SplitStats b = m_.split_stats();
    jc_.learn_join(1e3 * (now_s() - t0), (int)(b.revives - a.revives), (int)(b.exchanged - a.exchanged));
    if (lists_split(ja.h, ja.upto) == 0) return 0;
    return cur < prev ? 1 : 2;
  }
  // The conversation's pieces below `upto` that are quarter slots come back whole (kStJoin commands of at most --kv-join-pieces pieces,
  // each checked on the device and agreed by the four ranks), within `budget` numbers in all (restore_budget, read once: each command
  // carries what is left, and none is sent with nothing left: 0 on the wire is no limit; PICO_OLD_RULES sends 0). Before anything reads or
  // writes them: the core's own join inside an operation is a guard that cuts and says so. Returns the quarter slots it still names.
  int join_back(Session& x, int upto, uint64_t key, int64_t budget) {
    const model::Model::SplitStats a = m_.split_stats();
    const double t0 = now_s();
    int prev = m_.state_info(x.h).slots, commands = 0;
    int64_t left = budget;
    for (; commands < 100000 && prev > 0; ++commands) {
      const int64_t count = model::rules::join_command_count(left);
      if (count < 0) break;  // nothing left to take: no command (0 on the wire is no limit)
      state({{kStJoin, x.h, upto, (int32_t)std::min<int64_t>(count, 1 << 30)}});
      const int cur = m_.state_info(x.h).slots;
      left = budget - (int64_t)(m_.split_stats().join_taken - a.join_taken);
      if (cur >= prev) break;  // nothing below `upto`, or no number (join_short): the request is planned from what is whole
      prev = cur;
    }
    const model::Model::SplitStats b = m_.split_stats();
    if (b.joins > a.joins) {
      const double ms = 1e3 * (now_s() - t0);
      const uint64_t pieces = (b.revives - a.revives) + (b.exchanged - a.exchanged);
      jc_.learn_join(ms, (int)(b.revives - a.revives), (int)(b.exchanged - a.exchanged));
      std::printf("serve: session %016llx joined to position %d: %llu pieces back (%llu taken back from their ghosts, %llu exchanged) in %d commands, %.1f ms%s\n",
                  (unsigned long long)key, upto, (unsigned long long)pieces, (unsigned long long)(b.revives - a.revives),
                  (unsigned long long)(b.exchanged - a.exchanged), commands, ms, b.join_short > a.join_short ? "; stopped for want of a number" : "");
    }
    return m_.state_info(x.h).slots;
  }
  // A decoding run asks to pause after its step when a CONFIG is waiting or a tenant's change is due to begin or to be polled
  // (lend_pace.h): the batch of a few long responses is one run for minutes, and the change is a command between runs.
  bool lend_pause_due(double now = -1, lendpace::Why* why = nullptr) {
    lendpace::State st;
    const double t = now < 0 ? now_s() : now;
    st.config_waiting = io_.config_pending();
    st.change_active = m_.tenant_active();
    st.begin_wanted = !st.change_active && lend_begin_id(t) >= 0;
    // the settle lend_tick would run: the same gate and interval (a pause that no tick answers would repeat)
    st.settle_due = (m_.engram_lend_enabled() || m_.tenant_other_enabled()) && m_.window_owed() > 0 && t - win_settled_ >= 5 && m_.window_may_run();
    st.watch_due = memwatch::due(mem_watch_on(), t, watch_at_);
    st.now = t, st.poll_at = engram_poll_at_;
    if (why) *why = lendpace::pause_why(st);
    return lendpace::pause_due(st);  // no split or drop command ends a run: they go between runs (split_tick)
  }
  // A decoding run's bookkeeping on rank 0, once a step: cmd.cpp calls it at the end of the loop's turn, after the next step was queued and the
  // ride done, so it runs while the device computes that step and not in the token callback (which runs for every row). The flips of the watch
  // and of the imager's ride, the STATS frame at its limit, and the reasons that pause the run (lend_pause_due); one clock read for all of them.
  // `pausing`: a pause or cancel is already asked. `pause_bit`: the control word's bit this run pauses with (1 for a single run, kCtlPause for a
  // batch). True when it asked for the pause itself (a reason is kept for pause_note). The control word is read when a step's sample kernel runs,
  // so a bit set here, after the step was queued, is read by the step that is running or the next one.
  bool step_hook(RideWatch& rw, RideSplit& rs, uint32_t& word, bool pausing, uint32_t pause_bit) {
    const double t = now_s();
    ride_watch(rw, word, pausing, t);
    ride_split(rs, word, pausing, t);
    publish(true, t);
    if (pausing) return false;
    lendpace::Why why = lendpace::Why::None;
    if (!lend_pause_due(t, &why)) return false;
    m_.set_ctl(word |= pause_bit);
    pause_why_ = why;
    return true;
  }
  // The reason the run that just ended paused, when it was a tenant's change, a setting or owed groups: every such pause costs each decoder a
  // whole step, so the log names who asked. One line after the run's own, none for a pause an arrival or a burst's end asked.
  lendpace::Why pause_why_ = lendpace::Why::None;
  void pause_note() {
    if (pause_why_ == lendpace::Why::None) return;
    std::printf("serve: the run was paused for %s\n", lendpace::why_text(pause_why_));
    pause_why_ = lendpace::Why::None;
  }
  void lend_tick() {
    if (join_active()) return;  // a join in flight is a store change in the tenants' one queue: they wait at most its length
    if (!m_.engram_lend_enabled() && !m_.tenant_other_enabled()) return;
    const double t = now_s();
    if (m_.window_owed() > 0 && t - win_settled_ >= 5) settle_windows();
    Cmd c{};
    c.op = kCmdLend;
    int id = model::lend::kTenantEngram;
    if (m_.tenant_active()) {
      id = m_.tenant_change_id();
      if (t < engram_poll_at_) return;
      engram_poll_at_ = t + 0.1;
      c.rows = model::Model::kLendPoll;
    } else {
      id = lend_begin_id(t);
      if (id < 0) return;
      c.rows = model::Model::kLendBegin;
      c.pos0 = (int32_t)(id == model::lend::kTenantEngram ? engram_goal_mib_ : id == model::lend::kTenantPrefill ? (loan_goal_ == 1 ? loan_mib_ : 0) : tenant_goal_mib_);
    }
    c.tenant = id;
    const bool engram = id == model::lend::kTenantEngram, loan = id == model::lend::kTenantPrefill;
    int64_t loan_goal_mib = loan_goal_ == 1 ? loan_mib_ : 0;
    int64_t& goal = engram ? engram_goal_mib_ : loan ? loan_goal_mib : tenant_goal_mib_;
    double& next = engram ? engram_next_ : loan ? loan_next_ : tenant_next_;
    model::lend::Backoff& backoff = engram ? engram_backoff_ : loan ? loan_backoff_ : tenant_backoff_;
    const uint64_t gen = engram ? engram_gen_ : loan ? loan_gen_ : tenant_gen_;
    wd_.started = now_s();
    issue(c);
    const int st = run_lend(m_, c);
    wd_.started = 0;
    digest(engram ? "engram lend" : "tenant lend");
    forget_evicted();
    using M = model::Model;
    if (loan && st != M::kLendBusy && st != M::kLendPending) loan_busy_said_ = false;
    if (loan && (st == M::kLendNoRoom || st == M::kLendSame))
      std::printf("serve: prefill rows: the change to %lld MiB was refused (%s); not asked again for 300 s\n", (long long)goal, st == M::kLendNoRoom ? "the pool cannot lend it" : "already that size");
    if (st == M::kLendDone || st == M::kLendFailed) {
      publish_forced_ = true;  // STATS shows the change at once
      publish();
    }
    if (st == M::kLendPending) {
      engram_poll_at_ = now_s() + 0.05;
    } else if (st == M::kLendDone) {
      win_retry_at_ = 0;  // groups came back to the pool: a consumer that waited for them asks again at once
    } else if (st == M::kLendBusy) {
      ++lend_busy_total_;  // STATS: a tenant change answered busy
      if (loan && !loan_busy_said_) {
        loan_busy_said_ = true;
        std::printf("serve: prefill rows: the pool answered busy (a running request holds numbers the cut takes); asked again every second\n");
      }
      next = now_s() + 1.0;  // running requests hold what the cut takes
    } else if (st == M::kLendSame || st == M::kLendNoRoom || st == M::kLendFailed) {
      backoff.fail((uint64_t)goal, now_s(), gen);
      if (st == M::kLendFailed) win_retry_at_ = 0;  // its groups are back in the pool, or owed
      if (engram && st == M::kLendSame)
        std::printf("serve: the Engram cache stays at %llu MiB (the pool cannot lend more; %lld MiB asked for)\n", (unsigned long long)m_.engram_lend_info().mib,
                    (long long)engram_goal_mib_);
    }
  }
  // Tenant `id`'s size now, MiB (0 when it has none)
  int64_t tenant_size_mib(int id) const {
    for (const model::Model::TenantRow& r : m_.tenant_rows())
      if (r.id == id) return (int64_t)(r.size_bytes >> 20);
    return 0;
  }
  // A consumer is short, no line has anything left and no window can be closed: the Engram cache gives up one group, when its
  // floor (--engram-cache-min-mib) is below its size. The cache is smaller until the setting changes; what it gives arrives at
  // the change's poll, which wakes the consumer. With the floor at the set size (the default) the cache never gives way.
  void engram_give_way() {
    // a tenant ahead of the Engram cache in the order of giving way (an idle work area, a test build's dummy block) gives its group first
    const int victim = m_.tenant_victim();
    if (victim == 15 && !m_.tenant_active() && tenant_goal_mib_ >= 0) {
      const int64_t cur = tenant_size_mib(15), gb = (int64_t)(m_.group_bytes() >> 20);
      if (cur >= gb) {
        std::printf("serve: a request needs the pool's memory: tenant 15 gives up a group, %lld -> %lld MiB\n", (long long)cur, (long long)(cur - gb));
        tenant_goal_mib_ = cur - gb;
        tenant_next_ = 0;
        tenant_backoff_ = model::lend::Backoff();
        return;
      }
    }
    if (!m_.engram_lend_enabled() || o_.engram_min_mib < 0) return;
    const model::Model::EngramLendInfo li = m_.engram_lend_info();
    if (li.running || li.groups < 1 || (int64_t)li.mib <= o_.engram_min_mib) return;
    const int64_t to = std::max<int64_t>(o_.engram_min_mib, (int64_t)m_.engram_mib_for_groups((int)li.groups - 1));
    if (to >= (int64_t)li.mib) return;
    std::printf("serve: a request needs the pool's memory: the Engram cache gives up a group, %llu -> %lld MiB\n", (unsigned long long)li.mib, (long long)to);
    engram_goal_mib_ = to;
    engram_next_ = 0;
    engram_backoff_ = model::lend::Backoff();
  }
  // The window to give back when a consumer is short and no line has anything left: the least recently used open one that
  // no request needs (a paused flight is in rot_, so its window is never closed under it). Closes it, settles what the
  // close could not give back at once, and says whether a window was closed (the caller asks the pool again).
  bool close_unused_window() {
    if (!m_.window_lazy() || !m_.window_may_run()) return false;
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
    if (m_.window_owed() <= 0 || !m_.window_may_run()) return;
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
    return fac <= kBaseFactor || m_.window_is_open(fac) || (m_.window_may_run() && now_s() >= win_retry_at_);
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
    if (!m_.window_may_run()) return kWinWait;  // a tenant's change is in flight: no window command runs until it ends (lend_tick wakes the wait)
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
      if (plan.image_numbers > 0) {  // image rows never move: the window waits for those flights to end (finish() ends the wait)
        win_retry_at_ = now_s() + 1.0;
        if (now_s() - win_note_ > 10) {
          win_note_ = now_s();
          std::printf("serve: req %llu waits for the window of factor %d: %d image flights hold %lld image numbers it needs\n",
                      (unsigned long long)req, fac, plan.image_flights, (long long)plan.image_numbers);
        }
        return kWinWait;
      }
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
    if (!m_.window_lazy() || !flying_.empty() || !rot_.empty() || !pfq_.empty() || !m_.window_may_run()) return;
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
  // the conversations' ids files; no index to protect without --keep-gib
  idsfile::Keeper keeper_{o_.ids_dir, o_.keep_bytes != 0 && !o_.keep_index.empty(), (uint64_t)(unix_now() * 1e3)};
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
    JoinAsk ja;           // the return of its conversation (or base) that its first units are
    bool joining = false; // the join commands are not all done: the prompt is not begun until they are
    il::JoinWait jw;      // a unit that found no number waits (the request holds nothing up meanwhile)
    bool start_waited = false;  // its begin answered Waits (a prompt waiting at its start waits for the pool as a request in wait_ does)
  };
  // The positions a queued long prompt needs a slot for: a first run's prompt and horizon, a resume's from where it is
  int need_long(const Long& L) const { return L.resume ? need_now(L.f) : need_clamped(L.f.rq); }
  uint64_t wait_logged_ = 0;  // the request whose "waits at its start" line was printed last (0 once a start succeeds)
  il::JoinWait start_wait_;   // a request that waited at its start is asked again after its retry time
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
  std::set<uint64_t> run_ended_;  // sessions of flights that ended inside the run in progress (DONE sent): their slots are free for an arrival's question; cleared when the run's flights are finished
  std::set<uint64_t> flying_, later_;  // the sessions in flight; RELEASEs of them, applied at their end
  double win_used_[3] = {};   // when each factor's window was last used (index: win_level)
  int64_t engram_set_mib_ = 0, engram_goal_mib_ = 0;  // the Engram cache's size: the setting, and what the server steers to (lower after it gave way)
  uint64_t engram_gen_ = 1;                          // changes with the setting: a failed size is tried again
  double engram_poll_at_ = 0, engram_next_ = 0;
  model::lend::Backoff engram_backoff_;
  int64_t tenant_goal_mib_ = -1, tenant_set_groups_ = -1;  // a test build's dummy tenant 15: the size the server steers to (-1: none), and CONFIG's groups
  uint64_t tenant_gen_ = 1;
  double tenant_next_ = 0;
  model::lend::Backoff tenant_backoff_;
  double win_retry_at_ = 0;   // a window refused for lack of memory is not asked for again before this
  double win_note_ = 0, win_settled_ = 0;
  int win_fails_ = 0;
  int next_h_ = 1;
  uint64_t clock_ = 0;
  std::mt19937_64 rng_{std::random_device{}()};
};

}  // namespace

int serve(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o) {
  if (std::max(o.prefill_rows, o.prefill_rows_max) / o.snap_every > kMaxSnaps) die("serve: more than %d snapshots per prefill chunk", kMaxSnaps);
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
  if (!o.ids_dir.empty()) {  // the token ids' files: the directory made (what an earlier run left is removed after the kept index is read)
    std::string why;
    if (!idsfile::ensure_dir(o.ids_dir, &why)) die("serve: the ids directory: %s", why.c_str());
  }
  Io io(o.socket, Frame(fe::kHelloEngine).add(&hi).add(o.name.data(), o.name.size()).take(), o.frame_stage_bytes);
  std::printf("serve: listening on %s (context %d, session store %.2f GiB, kept between requests %.2f GiB)\n",
              o.socket.c_str(), o.max_ctx, m.state_capacity() / 1073741824.0, budget / 1073741824.0);
  Server s(m, ring, wd, o, io, budget);
  s.load_kept();
  s.sweep_ids();
  s.loop();
  if (g_prof) CUDA_CHECK(cudaProfilerStop());
  std::printf("serve: stopped\n");
  return 0;
}

}  // namespace srv
