// dsgb10_rank: the engine process. Every box runs it with the same arguments (engine/server/run4.sh);
// rank 0 decides and sends commands, ranks 1-3 follow (docs/engine.md). In decode every rank drives
// itself from the tokens each step's sampling hands to all of them.
//
//   dsgb10_rank [--fleet FILE] [--rank R] --mode prefill|tf|generate|bench|decode|batch [options]
//     --fleet FILE: the fleet file (fleet/homelab.conf; default: fleet.conf beside this executable, where
//                run4.sh and service.sh install theirs): the ring's devices and addresses, this box's
//                rank by its hostname unless --rank, and this rank's files: --weights, --hashmap,
//                --shard1 and --shard14 (in RANK<r>_ENGRAM) and --disk-dir default to its values. In
//                these paths, given or the fleet's, {rank} is the rank and a leading ~/ this box's home:
//                --weights '~/w/rank{rank}.dsw' names each rank's own file. Also this rank's Engram
//                read batch (RANK<r>_ENGRAM_READ_BATCH, 1 when absent)
//     prefill    the golden sequence in chunks of --chunk rows, every layer compared (--inject: each
//                layer fed the golden input), logits at every position
//     tf         teacher-forced decode: the first --prompt tokens as prefill, then steps of --k rows
//     generate   greedy (or --temp) generation of --gen tokens after --prompt tokens, against golden
//     bench      step times: the golden tokens' first --ctx positions as prefill, then for each k of
//                --ks (e.g. 1,6) --steps teacher-forced steps of k rows, no checks (graph replays
//                unless --no-graphs); the hash of every chosen token and its logit compares runs;
//                --bench-stride S: then as many steps of k unrelated rows, as other sessions' would
//                be: tokens from scattered golden positions, the rows S positions apart (S >= 128:
//                apart in the window too), then the same with row r's router logits rotated 48 r
//                experts (rows that share few experts), and each kind's distinct experts per layer;
//                --bench-groups G,...: then, for each G dividing k (1 < G < k), steps of k / G runs of
//                G consecutive rows (the verify rows of k / G sessions: docs/batching.md), each run
//                from its own scattered golden position, the runs S apart; natural, then each run's
//                routing rotated 48 experts further than the run before. A k past 8 (up to 16) runs
//                the grouped kinds only, as a batch of k / G sessions (--slots at least that; session g
//                in slot g, G consecutive positions per step, the sessions S positions apart): one
//                session's step holds at most model::kMaxSessRows rows
//     decode     end-to-end decode speed: --ctx golden positions as prefill, then --steps greedy (or
//                --temp, --top-p, --top-k) tokens through the self-driving decode loop
//     batch      sessions decoding together (docs/batching.md; --big): --batch 1024,16384c,1024
//                prefills each of these golden contexts (c: with CED) in a session slot of its own
//                (--slots; the last slot first), then decodes --steps tokens of each in one batch loop.
//                Each session's tokens hash is its decode-mode run's at that context. --batch-tf
//                [--batch-rows G]: instead, each session's golden continuation teacher-forced, G
//                (1..8) consecutive rows of each per step for --steps steps (a step of at most 16 rows in
//                all); its samples hash is the same whatever the other sessions and G, and tf mode's
//                over the same positions (--big --prompt C --k G --limit C + G * steps)
//     serve      rank 0 serves the front-end (docs/protocol.md) on --socket (default
//                /tmp/dsgb10-engine.sock): sessions with exact prefix reuse (--snap-every positions
//                between window snapshots, thinned with the distance from a session's end unless
//                --snap-all; every conversation's caches in one chunk pool, sized at startup by
//                --pool-free-gib G (leave G GiB of device memory and of available host memory free on every
//                Spark) and/or --pool-tokens N (a pool that holds N positions of conversation), with both
//                --pool-pick smaller (default) or larger; every rank takes the smallest rank's count. After
//                sizing the store's table is printed (model/pool_rules.h): with --pool-free-gib the store is
//                shortened on every rank by what the lines outside it may still take (cap less what they hold at the read:
//                the glibc heap, rank 0's front-end --front-mib (no default: a --pool-free-gib start refuses without it), the
//                image process outside the engine (the encoder container, --encoder-mib 1260; with --vision-weights the media
//                worker, --media-mib 512), the output queue, request frames --frame-stage-mib (400: a larger
//                frame is answered DONE too-large and not read), the server's token ids --ids-host-mib (1024: past it
//                idle conversations' ids go to <disk-dir>/ids), kernel drift --pool-slack-gib (0.75)), so that G GiB stay
//                free; --front-now-mib and --encoder-now-mib are what the two processes hold at the read (default 0: the
//                whole cap comes off the store). The store is a whole number of groups of 64 numbers; the start is refused
//                when it cannot hold a conversation of --max-ctx, the Engram cache's groups, the largest image request and
//                the largest window, or when one of the engine's own lines holds more than its cap at the read;
//                LRU,
//                the evicted spilled to a file of --disk-gib (default 32; 0: none) in --disk-dir (default the
//                fleet file's RANK<r>_SPILL), prefill in
//                chunks of --prefill-rows (default 8192 with CED, else 4096 as in the other modes),
//                sampled decode. --keep-gib N: a planned stop (SIGTERM) keeps the sessions for the
//                next start (docs/engine.md, "Kept sessions"): those in the file, and those in device
//                memory, the most recently used first, up to N GiB written at the stop (default 0:
//                nothing is kept), none unused for --keep-days (30; while serving too, a mop-up); --disk-tag T names the files
//                rank<r>.T.spill and .index (rank 0's server.T.index), so that runs of another tag
//                never touch them (the service's: service). --slots N: up to N requests decode
//                together, a row each per step (one alone: with drafts). --slice N: a request that
//                comes while others decode pauses them and is prefilled; while more requests decode
//                than there are slots, they take turns of N tokens (default 64; 0: no turns, and a
//                request that would need the slot of another in flight waits for one to free).
//                --engram-vram off|on|require (default off): the Engram row cache's first slots (touched by the CPU only)
//                go into the GB10's display reservation (~2 GiB a headless Spark never uses), which the pool's sizing
//                does not count; off never opens the card (ordinary memory, as without the option); on falls back
//                to ordinary memory with one start line when the card cannot give it; require stops;
//                --engram-vram-card (default /dev/dri/card0), --engram-vram-mib N (the most it asks for, default 2032).
//                --engram-cache-mib N (serve; default 4096) is the row cache's slots in MiB, the display reservation's included.
//                The cache's block is host memory taken from the chunk pool by whole groups right after the pool is made, and
//                the size can change while serving (the dashboard's setting, or CONFIG engram_mib): a size past what the pool can
//                lend is cut to what fits and a failed hand-over leaves the cache smaller, never refuses a start.
//                --engram-cache-max-mib N (default 16384): the address space the block is reserved over (the most a size can reach).
//                --engram-cache-min-mib N (default: the set size, so the cache never gives way): the smallest the cache is
//                made when the pool is short for a request and no window can be closed. --lend-null-mib N: the null control's
//                spread of MemAvailable in MiB, the hand-over proof's tolerance when above its floor of 256.
//                --pscan off|ext|all (default off: the ordinary graphs and buffers): the decode index scan split by positions over the four
//                ranks (each scans its quarter of every session's keys, one all-gather and one merge give the whole scan's picks, bit
//                for bit): ext splits the step graphs of factors 32 and 64 (windows past 1,048,576 positions), all factor 16's too (a
//                measuring arm); a step of more than 16 rows scans whole. Adds 0.94 MiB of candidate buffers a rank.
//                --kv-split off|tiered|max (default off; cache is read as max): every write to the disk tier, in every mode, is each Spark's own
//                quarter of a piece (a bookmark's slice), made in the background on a second stream and never waited for. tiered and max
//                then mark a conversation that is on no line and has no request in flight: its blocks are free where they lie (ghosts),
//                host bookkeeping only; a return takes them back when nobody used them, else the four Sparks read their quarters and
//                exchange them. off marks nothing (a conversation holds its blocks until a take evicts them, clean). Images are queued when
//                nothing is in flight (--kv-split-budget P (16) pieces a command) and inside a decoding run by rides at most every
//                --kv-split-ride-ms (250), --kv-split-ride-pieces (4) pieces each (tiered and off only while the ready blocks are under
//                --kv-split-high, default the room off keeps between requests; --kv-split-low, default half of it, raises a ride to 64
//                pieces); never between a prompt's chunks. --kv-split-after-s S (0): the idle age before a conversation is imaged, not asked while a request waits for clean blocks;
//                --kv-split-pass-ms N (50): the longest a pass of split commands runs before the loop goes on, with requests in flight at the
//                highest pressure (down to a fifth of it as the pressure falls; with nothing in flight, 250 ms);
//                --kv-join-pieces J (64): the pieces one return command brings back;
//                --kv-join-whole-ms M (250): a return estimated at most M ms (or with nobody decoding) goes back to back, above it the
//                request's first units are its return's commands, each followed by the decoders' window (0: always units beside decoders).
//                --ssd-flush-s S (30): the least time between two checkpoints while nothing runs and the store's images changed; without the imager also the
//                idle age at which a conversation's pieces are written to the disk tier (0: none; with the imager, which writes in the background, none).
//                --window-idle-s S: with --max-ctx past 1,048,576 the graphs and score buffers of factors 32 and 64 are made when a
//                request first needs them, out of the chunk pool. A window is freed when a consumer of the pool is short and no
//                request needs it; with S > 0 also after S seconds in which no request of its factor existed (default 0:
//                not by time).
//                --extend on|off: Extend context (default on). On: a request runs at the smallest YaRN factor whose window
//                holds its prompt, up to --max-ctx, and a response that reaches its window goes on at the next factor with its
//                conversation read again there (exact). Off: only the 1,048,576-token window is served: a longer prompt is
//                refused and a response that reaches the window ends with reason context, as with --max-ctx 1048576. The
//                dashboard's SETTINGS card changes it live.
//                --grow-ahead N: a request reserves its prompt and N positions of response (default 131072,
//                at most its max_new); a response reaches past them a horizon at a time, when the pool can hold it.
//                Interleaved prefill (docs/interleave.md): a prompt with more than --interleave-whole-rows
//                rows to compute (default 32768; the rows after its restart point, not the whole prompt)
//                is read in pieces while others decode: after each piece the decoders get a window, so
//                that they get --interleave-share % of the time (default 40) and wait at most
//                --interleave-gap-ms between steps (default 600: the piece size follows from it;
//                --interleave-rows N fixes it), a window at most --interleave-burst-max-ms (1000).
//                With nobody decoding a long prompt reads in --prefill-rows pieces, or in --prefill-rows-max pieces
//                (0: off, the default) while the pool lends the second set of row buffers they need (tenant 4: taken at the
//                start, given back when the pool has fewer than --prefill-loan-low-groups free groups (2) and nothing reads on it,
//                asked for again when it has the set's groups plus --prefill-loan-high-groups (8) free and its top groups
//                hold no number in use). Long prompts
//                wait in a queue holding nothing; one with at most --interleave-push-rows rows to compute
//                (default 131072; 0: none) reads beside the one being read, first. The dashboard (and a
//                CONFIG frame) changes these settings live.
//                --no-interleave: every prompt read whole. --exact reads every prompt whole.
//                The pool's chunk numbers are made and mapped once, before serving; a line points at its
//                conversation's numbers, a conversation takes them lowest free first as it grows and gives
//                them back when it ends (docs/engine.md, "The chunk pool")
//   generate --loop: the generation runs through the decode loop (same tokens as without)
//   --spec: DSpark drafts (mtp.* loaded) in the decode loop (decode, generate --loop; serve runs them
//     unless --no-spec: they change the speed, never the tokens), at most
//     --spec-drafts per step, verified while their survival >= --spec-thresh (default: lambda * dC from
//     --spec-cost C1,dC,Cdraft ms); --spec-gate ratio: the prefix of drafts with the most expected tokens
//     per ms, by --spec-row-ms (the ms of a step verifying 0..5 drafts, its draft pass included);
//     --spec-gate marginal: draft j while survival[j - 1] >= lambda * its row's ms by --spec-row-ms;
//     --spec-gate thresh: the default above; --mode spectf: the draft module teacher-forced on the golden run (a
//     step per position after the first --prompt - 1, drafting from the golden next token), its leading
//     drafts that equal the golden tokens against engine/dspark's reference; with --big the prompt goes
//     through the prefill path and its seeding of the windows; --draft-fp8: the drafter's head GEMV on an
//     FP8 copy of the vocab head, 0.75 ms less per step (the drafts' arithmetic changes, never the tokens;
//     serve's default, --draft-bf16 opts out); --draft-head4: that head as MXFP4 instead (made from the FP8
//     copy, which goes), --draft-markov8: the Markov head as FP8 (docs/engine.md, "The draft pass's
//     approximations": bundle B's formats, the study's simulated drafts bit for bit; off by default)
//   --node-stats-port N (serve; 9760): the dashboard's node sampler (server/node_stats.h) on this rank's first link
//     address (the fleet's NEXT_IP0): asleep until rank 0's front-end asks; 0: none
//   --ablit FILE: the runtime abliteration (model/ablit.h, made by tools/ablit/make_ablit.py; every rank reads the same
//     file): a request whose SUBMIT carries the ablit flag runs with the edit on its rows; the others are untouched
//   --tree-probe FILE (spectf; EXPERIMENT, docs/engine.md "Draft trees"): each step also runs the drafter's
//     probe along the golden tokens after the bonus (dspark::ProbeOut), a JSON line per step into FILE, with
//     the target's own top logprobs for the bonus's position (and with --seq-list, the sequence);
//     --tree-sib N: at every Nth step i, the routed experts of the probe's 4 most likely tokens for
//     position i + 2 (an eager step each after step i + 1, as a tree's siblings there would route) and of
//     the golden rows i + 2 .. i + 5 (one eager step, a chain's rows), a JSON line each
//   --head-raw: the vocab head's GEMVs on the bf16 head as loaded, rather than on its exact coding (every
//     tile's exponents as 4-bit offsets, 3/4 of the bytes: model::Model::code_head; the same bits)
//   --no-l2pf: no L2 prefetch in the decode step's windows (model::Options::l2pf, kernels/l2pf.cu: while a
//     collective or the attention leaves DRAM idle, a side stream prefetches the next weights; the same bits)
//   --setenv NAME=VALUE: set in every rank's environment before anything else (experiments' knobs)
//   --mode serve with none of these named runs the served configuration (server/served.h: the flags and DSGB10_* settings
//     it takes as defaults; --no-draft-head4 and --no-draft-markov8 turn the two drafters off, the other switches take their off values)
//   --profile-from-req N (serve): request N's first command opens nsys's capture range on every rank
//     (--capture-range=cudaProfilerApi) until the server stops; builds with -DDSGB10_NVTX add NVTX ranges
//     per turn, chunk, layer and part (model/nvtx.h)
//   --wave-probe MASK --wave-probe-at POS (with --big): in the prefill chunk at POS, for each layer L of
//     bit L, time layer L's FFN beside a second attention side (model/wave_probe.cu); a second ring on
//     the next port, its proxy on core 6. The run's outputs are not the model's.
//   --ced: --big prefills of at least 128 positions run CED (docs/ced.md): the encoder over every
//   position, the decoder replaying the last 128. serve runs CED by default (turns of at least 128 new
//   positions); --exact makes every turn exact (the bit-level checks)
//   --slots N: N session slots (docs/batching.md): live sessions' states side by side, every step through
//   the attention kernels' slot tables (serve: N requests decode together); --slot S: the golden modes run
//   in slot S (the same bits in any slot)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "comm/ring.h"
#include "common/alloc_seal.h"
#include "common/byte_account.h"
#include "common/check.h"
#include "common/heap.h"
#include "common/keepfile.h"
#include "dspark/dspark.h"
#include "engram/display.h"
#include "engram/engram.h"
#include "model/copy.h"
#include "model/exact_rule.h"
#include "model/img_rows.h"
#include "model/lend.h"
#include "model/model.h"
#include "model/spec.h"
#include "model/state.h"
#include "model/pool_rules.h"
#include "server/bounds.h"
#include "server/node_stats.h"
#include "server/cmd.h"
#include "server/golden.h"
#include "server/served.h"
#include "common/last_words.h"
#include "server/serve.h"
#include "vision/vit_device.h"
#include "weights/loader.h"

namespace {

using namespace srv;

// Bench: the distinct routed experts per layer of a step, from rank 0's route_idx dumps, and how often
// each expert of each layer was routed to (rows: the rows seen per layer).
struct RouteSink : model::Sink {
  double distinct = 0;
  int n = 0;
  std::vector<std::vector<int>> freq = std::vector<std::vector<int>>(model::kLayers, std::vector<int>(model::kExperts));
  std::vector<int> rows = std::vector<int>(model::kLayers);
  void fill(int, const char*, int, int, void*, size_t) override {}
  void tensor(int L, const char* name, int, int R, const void* data, size_t) override {
    if (std::strcmp(name, "route_idx") != 0 || L < 0 || L >= model::kLayers) return;
    const auto* e = static_cast<const int32_t*>(data);
    distinct += (double)std::set<int32_t>(e, e + (size_t)R * model::kTopK).size();
    ++n;
    for (int i = 0; i < R * model::kTopK; ++i)
      if (e[i] >= 0 && e[i] < model::kExperts) ++freq[L][e[i]];
    rows[L] += R;
  }
  void logits(int, int, const float*) override {}
  // the distinct experts per layer that B rows routed independently with these frequencies would reach
  double independent(int B) const {
    double s = 0;
    int nl = 0;
    for (int L = 0; L < model::kLayers; ++L) {
      if (!rows[L]) continue;
      for (int x : freq[L]) s += 1.0 - std::pow(1.0 - (double)x / rows[L], B);
      ++nl;
    }
    return nl ? s / nl : 0;
  }
};

// EXPERIMENT (--tree-sib): each layer's routed experts in one eager step, [R][kTopK] per layer
struct ExpertSink : model::Sink {
  std::vector<std::vector<int32_t>> e = std::vector<std::vector<int32_t>>(model::kLayers);
  void fill(int, const char*, int, int, void*, size_t) override {}
  void tensor(int L, const char* name, int, int R, const void* data, size_t) override {
    if (std::strcmp(name, "route_idx") != 0 || L < 0 || L >= model::kLayers) return;
    const auto* p = static_cast<const int32_t*>(data);
    e[L].assign(p, p + (size_t)R * model::kTopK);
  }
  void logits(int, int, const float*) override {}
};

// EXPERIMENT (--seq-list; docs/engine.md "Prompt-lookup drafts"): sequences of their own in one process, a
// line "DIR PROMPT [N]" each: DIR's tokens.npy, its first PROMPT tokens the prompt; N: generate's most
// tokens, spectf's end (0: the whole sequence)
struct SeqLine {
  std::string dir;
  int prompt = 0, n = 0;
  std::vector<int32_t> toks;
};

std::vector<SeqLine> read_seq_list(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) die("cannot read %s", path.c_str());
  std::vector<SeqLine> v;
  char line[8192], dir[4096];
  while (std::fgets(line, sizeof line, f)) {
    SeqLine s;
    if (line[0] == '#' || std::sscanf(line, "%4095s %d %d", dir, &s.prompt, &s.n) < 2) continue;
    s.dir = dir;
    golden::Golden g(s.dir);
    for (int p = 0; p < g.length(); ++p) s.toks.push_back(g.token(p));
    v.push_back(std::move(s));
  }
  std::fclose(f);
  if (v.empty()) die("%s: no sequences", path.c_str());
  return v;
}

// EXPERIMENT (--props): rows to route, a line "SEQ ANCHOR t1 .. tk" each: the anchor's own token, then t1 .. tk
using Props = std::map<std::pair<int, int>, std::vector<std::vector<int32_t>>>;
Props read_props(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (!f) die("cannot read %s", path.c_str());
  Props m;
  char line[8192];
  while (std::fgets(line, sizeof line, f)) {
    char* q = line;
    char* e;
    const long sq = std::strtol(q, &e, 10);
    if (e == q) continue;
    q = e;
    const long anchor = std::strtol(q, &e, 10);
    if (e == q) continue;
    std::vector<int32_t> t;
    for (q = e;; q = e) {
      const long x = std::strtol(q, &e, 10);
      if (e == q) break;
      t.push_back((int32_t)x);
    }
    m[{(int)sq, (int)anchor}].push_back(t);
  }
  std::fclose(f);
  return m;
}

// Free bytes in dir (made if missing; mode 0700: the sessions' state is what conversations left), this
// rank's file (base.spill) counted as free: it is either reused or made again.
uint64_t spill_free(const std::string& dir, const std::string& base) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  struct stat s;
  if (stat(dir.c_str(), &s) == 0 && s.st_uid == getuid() && (s.st_mode & 077)) chmod(dir.c_str(), 0700);
  struct statvfs v;
  if (statvfs(dir.c_str(), &v) != 0) return 0;
  uint64_t fr = (uint64_t)v.f_bavail * v.f_frsize;
  if (lstat((base + ".spill").c_str(), &s) == 0 && S_ISREG(s.st_mode)) fr += (uint64_t)s.st_blocks * 512;
  return fr;
}

// dir, or a directory above it, holds .git: a repository or a worktree, where no session's state goes
bool in_git_tree(const std::string& dir) {
  std::error_code ec;
  for (std::filesystem::path p = std::filesystem::absolute(dir, ec); !ec && !p.empty(); p = p.parent_path()) {
    if (std::filesystem::exists(p / ".git", ec)) return true;
    if (p == p.parent_path()) break;
  }
  return false;
}

struct Args {
  std::string fleet;  // the fleet file ("": fleet.conf beside this executable)
  int rank = -1;
  std::string mode = "prefill";
  std::string golden = "/data/dsgb10/golden/prose";
  std::string weights, hashmap, shard[2];
  bool inject = false, inject_sub = false, no_engram = false, quiet = false, graphs = true, per_position = false;
  bool profile = false;  // decode: the decode loop is nsys's capture range (--capture-range=cudaProfilerApi)
  bool loop = false, big = false, ced = false, exact = false, graph_sweep = false;
  int ahead = 1, prefill_rows = 0;  // 0: 8192 for serve with CED, else 4096
  int loan_low_groups = 2, loan_high_extra = 8;  // serve: the lent set goes back under this many free groups; it is asked for at its own groups plus this many
  int prefill_rows_max = 0;         // serve: > prefill_rows: chunks of this many rows on a second set of buffers lent by the pool (0: off)
  int wave_probe = 0, wave_probe_at = -1;
  int profile_at = -1;  // --big: the prefill chunk at this position is nsys's capture range
  uint64_t profile_from_req = 0;  // serve: request N's first command opens it on every rank (0: none)
  int chunk = 8, k = 1, prompt = -1, limit = -1, gen = 96, max_ctx = 65536, threads = 4, layers = model::kLayers, first = 0;
  int ctx = 1024, steps = 64, onset = -1, bench_stride = 0;
  std::vector<int> ks{1}, ctxs;  // ctxs: decode and bench at each of these contexts in turn (else ctx)
  std::vector<int> bench_groups;  // bench: runs of consecutive rows (--bench-groups)
  std::string batch;             // --mode batch: the sessions' contexts
  bool batch_tf = false;         // --mode batch: teacher-forced, batch_rows rows of each session per step
  int batch_rows = 1;
  std::string tree_probe;        // --mode spectf, EXPERIMENT: the probe's records
  int tree_sib = 0;
  std::string seq_list, route_file, props;  // EXPERIMENT (spectf, generate --loop): prompt-lookup drafts
  int route_every = 0, chain_every = 0;
  int step_timeout_s = 120, engram_cache_mib = 4096, engram_cache_max_mib = 16384, engram_cache_min_mib = -1, lend_null_mib = 0;
  std::string engram_vram = "off";                   // the row cache's first slots in the display reservation: off | on | require
  std::string engram_vram_card = "/dev/dri/card0";   // its DRM device
  std::string engram_vram_sysfs = "/sys";            // where the card's outputs are read (tests point it at a fake tree)
  int engram_vram_mib = 2032;                        // the most it asks for (the display reservation of a GB10 holds 2,032 MiB)
  int engram_read_batch = 1;  // the fleet file's RANK<r>_ENGRAM_READ_BATCH
  float temp = 0.f, top_p = 1.f;
  int top_k = 0;
  uint32_t seed = 1;
  std::string socket = "/tmp/dsgb10-engine.sock";
  double pool_free_gib = -1;    // serve: the chunk pool leaves this many GiB free on every Spark (< 0: not given)
  double pool_slack_gib = 0.75;  // serve: kernel drift, a line outside the store that nothing enforces (slab the engine does not own)
  int front_mib = 0;            // serve, rank 0: what the front-end process may hold (dsfront's --front-mib: PICO_FRONT_MIB), a line outside the store; 0 refuses the start
  int front_now_mib = 0;        // serve, rank 0: what it holds at the start (its cgroup's memory.current); 0: unread, the whole cap comes off the store
  int encoder_mib = 1260;       // serve, rank 0: what the image encoder process holds (device and host), a line outside the store
  int media_mib = 512;          // serve, rank 0, with --vision-weights: the media worker's container cap, a line outside the store
  int encoder_now_mib = 0;      // serve, rank 0: what it holds at the start; 0: unread
  int frame_stage_mib = 400;    // serve, rank 0: a frame larger than this is not read (DONE too-large); the request frames line is twice this (payload and the request's rows)
  int ids_host_mib = 1024;      // serve, rank 0: the server's token ids in memory; past it idle conversations' ids go to files under the disk directory
  int64_t pool_tokens = -1;      // serve: the chunk pool holds this many positions of conversation (< 0: not given)
  std::string pool_pick = "smaller";  // serve: with both, the smaller pool or the larger
  double disk_gib = 32;  // serve: the store's disk tier (0: none)
  std::string disk_dir;  // its directory (default: the fleet's RANK<r>_SPILL)
  std::string disk_tag;  // its files' tag: rank<r>.<tag>.spill (none: rank<r>.spill)
  bool extend = true;  // serve: --extend on|off (Extend context)
  double keep_gib = 0, keep_days = 30;  // kept sessions: written at a planned stop, at most; their age
  int snap_every = 1024, small_rows = 8;
  int slots = 1, slot = 0;
  int copy_mib = 512;  // --copy-mib N: copy drafting's arena a rank (model::Options::copy_mib; used only with DSGB10_COPY / DSGB10_COPY_DUMP)
  int big_slots = 0;  // --big-slots B: slots [0, B) hold --max-ctx, the others --max-ctx / 8 (model::Options::big_slots)  // session slots (model::Options::slots); the golden modes' slot
  int bgraph_bodies = 0;      // DIAGNOSTICS (spec.cu, the batch graph): switch bodies to capture (0: kMaxRows)
  bool bgraph_noif = false;   // DIAGNOSTICS: the entries' accepts without if nodes
  bool bgraph_check = false;  // DIAGNOSTICS: the one-session draft of every entry after the batch
                              // step's pass over them, and the words that differ, printed by --mode batch
  int slice = 64;  // serve: time-slicing's turns, in tokens (0: none)
  int grow_ahead = 131072;  // serve: positions of response a reservation reaches past where it is
  int ssd_flush_s = 30;     // serve: seconds between checkpoints while idle; without the imager also the idle age for writing a conversation's pieces
  bool kv_split = false;    // serve: --kv-split tiered|max: a stored conversation is split into quarter slots (docs/engine.md)
  int kv_split_mode = 0;    // splitpace::Mode
  double kv_split_after_s = 0;  // --kv-split-after-s: the idle age before a conversation is imaged (default 0, every mode)
  int64_t kv_split_high = 0, kv_split_low = 0;
  int kv_split_pass_ms = 50;  // --kv-split-pass-ms: the longest a pass of split commands runs with requests in flight
  model::PScan pscan = model::PScan::Off;  // --pscan off|ext|all (model::Options::pscan)
  int kv_split_budget = 16, kv_join_pieces = 64, kv_join_whole_ms = 250;
  int kv_split_ride_ms = 250, kv_split_ride_pieces = 4;  // the imager's rides inside a decoding run (split_pace.h)  // the split command's size, the join command's size
  int window_idle_s = reclaim::kWindowIdleS;  // serve past 1M: a factor's window freed after this many seconds with no request of it (0: not by time)
  il::Settings il;  // serve: interleaved prefill's settings (--interleave-*)
  bool snap_all = false;
  size_t ring_cap = 4u << 20;  // the ring's per-rank collective size (prefill's bulk collectives post cap / 4)
  bool spec = false, no_spec = false, draft_fp8 = false, draft_bf16 = false, draft_head4 = false, draft_markov8 = false;
  bool head_raw = false, no_l2pf = false;
  std::string draft_weights;  // --draft-weights FILE: the drafter (mtp.*) from this rank file instead of --weights
  std::string vision_weights;  // --vision-weights FILE: this rank's quarter of the image encoder (dsw_convert --vision), loaded before the store is sized
  std::string ablit;          // --ablit FILE: the runtime abliteration (model/ablit.h); requests opt in per request
  int node_stats_port = 9760;  // serve: the dashboard's node sampler on this rank's first link address (0: none)
  std::string node_stats_addr, node_stats_rails[2];  // from the fleet file: that address, the rails' devices
  int spec_drafts = -1;  // -1: the one-session pass's rows (dspark::draft_rows(), DSGB10_DRAFT_MAX; 5 by default)
  float spec_thresh = -1.f;
  float spec_cost[3] = {19.4f, 3.6f, 3.3f};
  int spec_gate = model::ops::kGateThresh;
  float spec_row_ms[model::ops::kDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f, 41.70f, 44.70f};
};

Args parse(int argc, char** argv) {
  Args a;
  // a serve start takes the served settings (served.h) for every flag it does not name
  std::vector<std::string> in(argv + 1, argv + argc);
  std::vector<std::string> av = served::flags_for(in);
  av.insert(av.end(), in.begin(), in.end());
  const int n_av = (int)av.size();
  for (int i = 0; i < n_av; ++i) {
    const std::string s = av[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= n_av) die("missing value for %s", s.c_str());
      return av[++i];
    };
    if (s == "--rank") a.rank = std::atoi(val().c_str());
    else if (s == "--fleet") a.fleet = comm::fleet_path(val());  // a leading ~/ is $HOME, as in the paths below
    else if (s == "--mode") a.mode = val();
    else if (s == "--golden") a.golden = val();
    else if (s == "--weights") a.weights = val();
    else if (s == "--draft-weights") a.draft_weights = val();
    else if (s == "--vision-weights") a.vision_weights = val();
    else if (s == "--hashmap") a.hashmap = val();
    else if (s == "--shard1") a.shard[0] = val();
    else if (s == "--shard14") a.shard[1] = val();
    else if (s == "--inject") a.inject = true;
    else if (s == "--inject-sub") a.inject = a.inject_sub = true;
    else if (s == "--no-engram") a.no_engram = true;
    else if (s == "--quiet") a.quiet = true;
    else if (s == "--profile") a.profile = true;
    else if (s == "--profile-at") a.profile_at = std::atoi(val().c_str());
    else if (s == "--profile-from-req") a.profile_from_req = std::strtoull(val().c_str(), nullptr, 10);
    else if (s == "--wave-probe") a.wave_probe = std::atoi(val().c_str());
    else if (s == "--wave-probe-at") a.wave_probe_at = std::atoi(val().c_str());
    else if (s == "--chunk") a.chunk = std::atoi(val().c_str());
    else if (s == "--k") a.k = std::atoi(val().c_str());
    else if (s == "--prompt") a.prompt = std::atoi(val().c_str());
    else if (s == "--limit") a.limit = std::atoi(val().c_str());
    else if (s == "--gen") a.gen = std::atoi(val().c_str());
    else if (s == "--max-ctx") a.max_ctx = std::atoi(val().c_str());
    else if (s == "--threads") a.threads = std::atoi(val().c_str());
    else if (s == "--layers") a.layers = std::atoi(val().c_str());
    else if (s == "--first") a.first = std::atoi(val().c_str());
    else if (s == "--step-timeout") a.step_timeout_s = std::atoi(val().c_str());
    else if (s == "--engram-cache-mib") a.engram_cache_mib = std::atoi(val().c_str());
    else if (s == "--engram-cache-max-mib") a.engram_cache_max_mib = std::atoi(val().c_str());
    else if (s == "--engram-cache-min-mib") a.engram_cache_min_mib = std::atoi(val().c_str());
    else if (s == "--lend-null-mib") a.lend_null_mib = std::atoi(val().c_str());
    else if (s == "--engram-vram") {
      a.engram_vram = val();
      if (a.engram_vram != "off" && a.engram_vram != "on" && a.engram_vram != "require") die("--engram-vram: off, on or require");
    }
    else if (s == "--engram-vram-card") a.engram_vram_card = val();
    else if (s == "--engram-vram-sysfs") a.engram_vram_sysfs = val();
    else if (s == "--engram-vram-mib") a.engram_vram_mib = std::atoi(val().c_str());
    else if (s == "--temp") a.temp = std::atof(val().c_str());
    else if (s == "--seed") a.seed = (uint32_t)std::atoll(val().c_str());
    else if (s == "--ctx") a.ctx = std::atoi(val().c_str());
    else if (s == "--steps") a.steps = std::atoi(val().c_str());
    else if (s == "--bench-stride") a.bench_stride = std::atoi(val().c_str());
    else if (s == "--ks" || s == "--ctxs" || s == "--bench-groups") {
      std::vector<int>& list = s == "--ks" ? a.ks : s == "--ctxs" ? a.ctxs : a.bench_groups;
      list.clear();
      const std::string v = val();
      for (const char* q = v.c_str(); *q;) {
        list.push_back(std::atoi(q));
        while (*q && *q != ',') ++q;
        if (*q) ++q;
      }
    }
    else if (s == "--no-graphs") a.graphs = false;
    else if (s == "--onset") a.onset = std::atoi(val().c_str());
    else if (s == "--per-position") a.per_position = true;
    else if (s == "--loop") a.loop = true;
    else if (s == "--ahead") a.ahead = std::atoi(val().c_str());
    else if (s == "--big") a.big = true;
    else if (s == "--graph-sweep") a.graph_sweep = true;
    else if (s == "--ced") a.ced = true;
    else if (s == "--exact") a.exact = true;
    else if (s == "--prefill-rows") a.prefill_rows = std::atoi(val().c_str());
    else if (s == "--prefill-rows-max") a.prefill_rows_max = std::atoi(val().c_str());
    else if (s == "--prefill-loan-low-groups") a.loan_low_groups = std::atoi(val().c_str());
    else if (s == "--prefill-loan-high-groups") a.loan_high_extra = std::atoi(val().c_str());
    else if (s == "--top-p") a.top_p = std::atof(val().c_str());
    else if (s == "--top-k") a.top_k = std::atoi(val().c_str());
    else if (s == "--socket") a.socket = val();
    else if (s == "--pool-free-gib") a.pool_free_gib = std::atof(val().c_str());
    else if (s == "--pool-slack-gib") a.pool_slack_gib = std::atof(val().c_str());
    else if (s == "--front-mib") a.front_mib = std::atoi(val().c_str());
    else if (s == "--front-now-mib") a.front_now_mib = std::atoi(val().c_str());
    else if (s == "--encoder-now-mib") a.encoder_now_mib = std::atoi(val().c_str());
    else if (s == "--frame-stage-mib") a.frame_stage_mib = std::atoi(val().c_str());
    else if (s == "--ids-host-mib") a.ids_host_mib = std::atoi(val().c_str());
    else if (s == "--encoder-mib") a.encoder_mib = std::atoi(val().c_str());
    else if (s == "--media-mib") a.media_mib = std::atoi(val().c_str());
    else if (s == "--pool-tokens") a.pool_tokens = std::atoll(val().c_str());
    else if (s == "--pool-pick") a.pool_pick = val();
    else if (s == "--slot-gib" || s == "--session-gib" || s == "--session-margin-gib" || s == "--no-vmm")
      die("%s is gone: the chunk pool's size is --pool-free-gib G and/or --pool-tokens N (--pool-pick smaller|larger; "
          "docs/engine.md, \"The chunk pool\")", s.c_str());
    else if (s == "--disk-gib") a.disk_gib = std::atof(val().c_str());
    else if (s == "--disk-dir") a.disk_dir = val();
    else if (s == "--disk-tag") a.disk_tag = val();
    else if (s == "--keep-gib") a.keep_gib = std::atof(val().c_str());
    else if (s == "--keep-days") a.keep_days = std::atof(val().c_str());
    else if (s == "--extend") {
      const std::string v = val();
      if (v != "on" && v != "off") die("--extend is on or off");
      a.extend = v == "on";
    }
    else if (s == "--snap-every") a.snap_every = std::atoi(val().c_str());
    else if (s == "--slots") a.slots = std::atoi(val().c_str());
    else if (s == "--big-slots") a.big_slots = std::atoi(val().c_str());
    else if (s == "--copy-mib") a.copy_mib = std::atoi(val().c_str());
    else if (s == "--bgraph-bodies") a.bgraph_bodies = std::atoi(val().c_str());
    else if (s == "--bgraph-noif") a.bgraph_noif = true;
    else if (s == "--bgraph-check") a.bgraph_check = true;
    else if (s == "--slot") a.slot = std::atoi(val().c_str());
    else if (s == "--batch") a.batch = val();
    else if (s == "--batch-tf") a.batch_tf = true;
    else if (s == "--batch-rows") a.batch_rows = std::atoi(val().c_str());
    else if (s == "--slice") a.slice = std::atoi(val().c_str());
    else if (s == "--grow-ahead") a.grow_ahead = std::atoi(val().c_str());
    else if (s == "--ssd-flush-s") a.ssd_flush_s = std::atoi(val().c_str());
    else if (s == "--kv-split") {
      const std::string v = val();
      if (v != "off" && v != "tiered" && v != "max" && v != "cache") die("--kv-split is off, tiered or max, not %s", v.c_str());
      a.kv_split_mode = v == "off" ? 0 : v == "tiered" ? 1 : 2;  // cache is max
      a.kv_split = a.kv_split_mode != 0;
    }
    else if (s == "--kv-split-after-s") a.kv_split_after_s = std::max(0.0, std::atof(val().c_str()));
    else if (s == "--kv-split-high") a.kv_split_high = std::max<int64_t>(0, std::atoll(val().c_str()));
    else if (s == "--kv-split-low") a.kv_split_low = std::max<int64_t>(0, std::atoll(val().c_str()));
    else if (s == "--kv-split-pass-ms") a.kv_split_pass_ms = std::max(1, std::atoi(val().c_str()));
    else if (s == "--pscan") {
      const std::string v = val();
      if (v != "off" && v != "ext" && v != "all") die("--pscan is off, ext or all, not %s", v.c_str());
      a.pscan = v == "off" ? model::PScan::Off : v == "ext" ? model::PScan::Ext : model::PScan::All;
    }
    else if (s == "--kv-split-budget") a.kv_split_budget = std::max(1, std::atoi(val().c_str()));
    else if (s == "--kv-split-ride-ms") a.kv_split_ride_ms = std::max(1, std::atoi(val().c_str()));
    else if (s == "--kv-split-ride-pieces") a.kv_split_ride_pieces = std::max(1, std::atoi(val().c_str()));
    else if (s == "--kv-join-pieces") a.kv_join_pieces = std::max(1, std::atoi(val().c_str()));
    else if (s == "--kv-join-whole-ms") a.kv_join_whole_ms = std::max(0, std::atoi(val().c_str()));
    else if (s == "--window-idle-s") a.window_idle_s = std::atoi(val().c_str());
    else if (s == "--interleave") a.il.on = true;
    else if (s == "--no-interleave") a.il.on = false;
    else if (s == "--interleave-whole-rows") a.il.whole_rows = std::atoi(val().c_str());
    else if (s == "--interleave-push-rows") a.il.push_rows = std::atoi(val().c_str());
    else if (s == "--interleave-gap-ms") a.il.gap_ms = std::atoi(val().c_str());
    else if (s == "--interleave-share") a.il.share = std::atoi(val().c_str());
    else if (s == "--interleave-burst-max-ms") a.il.burst_max_ms = std::atoi(val().c_str());
    else if (s == "--interleave-rows") a.il.rows = std::atoi(val().c_str());
    else if (s == "--snap-all") a.snap_all = true;
    else if (s == "--small-rows") a.small_rows = std::atoi(val().c_str());
    else if (s == "--ring-cap") a.ring_cap = std::strtoull(val().c_str(), nullptr, 10);
    else if (s == "--spec") a.spec = true;
    else if (s == "--no-spec") a.no_spec = true;
    else if (s == "--draft-fp8") a.draft_fp8 = true;
    else if (s == "--draft-bf16") a.draft_bf16 = true;
    else if (s == "--draft-head4") a.draft_head4 = true;
    else if (s == "--no-draft-head4") a.draft_head4 = false;
    else if (s == "--draft-markov8") a.draft_markov8 = true;
    else if (s == "--no-draft-markov8") a.draft_markov8 = false;
    else if (s == "--ablit") a.ablit = val();
    else if (s == "--node-stats-port") a.node_stats_port = std::atoi(val().c_str());
    else if (s == "--tree-probe") a.tree_probe = val();
    else if (s == "--tree-sib") a.tree_sib = std::atoi(val().c_str());
    else if (s == "--seq-list") a.seq_list = val();
    else if (s == "--route-file") a.route_file = val();
    else if (s == "--props") a.props = val();
    else if (s == "--route-every") a.route_every = std::atoi(val().c_str());
    else if (s == "--chain-every") a.chain_every = std::atoi(val().c_str());
    else if (s == "--head-raw") a.head_raw = true;
    else if (s == "--no-l2pf") a.no_l2pf = true;
    else if (s == "--setenv") {
      const std::string v = val();
      const size_t eq = v.find('=');
      if (eq == std::string::npos || eq == 0) die("--setenv takes NAME=VALUE");
      setenv(v.substr(0, eq).c_str(), v.substr(eq + 1).c_str(), 1);
    }
    else if (s == "--spec-drafts") a.spec_drafts = std::atoi(val().c_str());
    else if (s == "--spec-thresh") a.spec_thresh = std::atof(val().c_str());
    else if (s == "--spec-cost") {
      const std::string v = val();
      if (std::sscanf(v.c_str(), "%f,%f,%f", &a.spec_cost[0], &a.spec_cost[1], &a.spec_cost[2]) != 3)
        die("--spec-cost takes C1,dC,Cdraft (ms)");
    }
    else if (s == "--spec-gate") {
      const std::string v = val();
      if (v == "thresh") a.spec_gate = model::ops::kGateThresh;
      else if (v == "ratio") a.spec_gate = model::ops::kGateRatio;
      else if (v == "marginal") a.spec_gate = model::ops::kGateMarginal;
      else die("--spec-gate takes thresh, ratio or marginal");
    }
    else if (s == "--spec-row-ms") {
      const std::string v = val();
      float* r = a.spec_row_ms;
      if (std::sscanf(v.c_str(), "%f,%f,%f,%f,%f,%f", &r[0], &r[1], &r[2], &r[3], &r[4], &r[5]) != 6)
        die("--spec-row-ms takes the ms of a step verifying 0..5 drafts, six of them");
    }
    else die("unknown argument %s", s.c_str());
  }
  if (!a.prefill_rows) a.prefill_rows = a.mode == "serve" && !a.exact ? 8192 : 4096;
  if (a.prefill_rows_max < 0 || (a.prefill_rows_max > 0 && a.prefill_rows_max < a.prefill_rows))
    die("--prefill-rows-max takes 0 or a number of rows of at least --prefill-rows (%d)", a.prefill_rows);
  if (a.loan_low_groups < 0 || a.loan_high_extra < 0) die("--prefill-loan-low-groups and --prefill-loan-high-groups take 0 or more");
  if (a.prefill_rows_max == a.prefill_rows || a.mode != "serve" || a.exact) a.prefill_rows_max = 0;  // no second set: the engine as it is
  if (a.disk_tag.size() > 32 || a.disk_tag.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
    die("--disk-tag: at most 32 of a-z, 0-9, _ and -");
  if (a.keep_gib < 0 || !(a.keep_days > 0)) die("--keep-gib must be >= 0 and --keep-days > 0");
  if (a.slice < 0) die("--slice must be >= 0");
  if (a.grow_ahead < 64) die("--grow-ahead must be >= 64");
  if (a.il.share < 10 || a.il.share > 90) die("--interleave-share must be 10..90 (%% of the time decoders get)");
  if (a.il.gap_ms < 100) die("--interleave-gap-ms must be >= 100");
  if (a.il.burst_max_ms < 50) die("--interleave-burst-max-ms must be >= 50");
  if (a.il.rows < 0 || a.il.whole_rows < 0 || a.il.push_rows < 0) die("--interleave-rows, -whole-rows and -push-rows must be >= 0");
  // this box's rank and files: the fleet file's, unless given
  const comm::Fleet& fl = comm::fleet(a.fleet);
  if (a.rank < 0) a.rank = comm::rank_from_hostname(a.fleet);
  die_set_rank(a.rank);
  lastwords::install(a.rank);  // the signal, rank, time and call stack of a crash (common/last_words.h)
  if (a.rank < 0)
    die("cannot tell this box's rank: its hostname is no RANK<r>_HOSTNAME in the fleet file %s; pass --rank",
        fl.file.c_str());
  if (a.rank > 3) die("--rank %d: the ring's ranks are 0-3", a.rank);
  const comm::RankAddr& me = fl.rank[a.rank];
  a.node_stats_addr = me.next_ip[0], a.node_stats_rails[0] = fl.next_dev[0], a.node_stats_rails[1] = fl.next_dev[1];
  const std::string r = std::to_string(a.rank);
  // a path flag's value, else the fleet's: {rank} is this rank, a leading ~/ this box's home (the same
  // arguments then name each rank's own files)
  auto path = [&](std::string& v, const std::string& value, const char* key, const char* flag) {
    v = comm::fleet_path(v.empty() ? value : v, a.rank);
    if (v.empty()) die("no %s, and the fleet file %s has no RANK%d_%s", flag, fl.file.c_str(), a.rank, key);
  };
  path(a.weights, me.weights, "WEIGHTS", "--weights");
  if (!a.draft_weights.empty()) a.draft_weights = comm::fleet_path(a.draft_weights, a.rank);
  if (!a.vision_weights.empty()) a.vision_weights = comm::fleet_path(a.vision_weights, a.rank);
  if (!a.no_engram) {
    path(a.hashmap, me.hashmap, "HASHMAP", "--hashmap");
    std::string dir = me.engram;
    while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
    path(a.shard[0], dir.empty() ? "" : dir + "/engram-l1-r" + r + "of4.bin", "ENGRAM", "--shard1");
    path(a.shard[1], dir.empty() ? "" : dir + "/engram-l14-r" + r + "of4.bin", "ENGRAM", "--shard14");
  }
  if (a.mode == "serve") path(a.disk_dir, me.spill, "SPILL", "--disk-dir");
  a.engram_read_batch = me.engram_read_batch;
  // steps of one session hold at most kMaxSessRows rows (config.h); a step of up to kMaxRows takes several sessions
  if (a.chunk < 1 || a.chunk > model::kMaxSessRows || a.k < 1 || a.k > model::kMaxSessRows)
    die("--chunk and --k must be 1..%d (one session's rows in a step)", model::kMaxSessRows);
  if (a.small_rows < 0 || a.small_rows > model::kMaxSessRows)
    die("--small-rows must be 0..%d (a small turn runs as one step of one session)", model::kMaxSessRows);
  if (a.batch_rows < 1 || a.batch_rows > (getenv("DSGB10_TF_SESSROWS") ? model::kMaxRows : model::kMaxSessRows))
    die("--batch-rows must be 1..%d (one session's rows in a step)", model::kMaxSessRows);
  if (a.mode == "serve") a.spec = !a.no_spec;  // the drafts change the speed, never the tokens
  if (a.mode == "serve") a.draft_fp8 = !a.draft_bf16;  // nor does the drafter's FP8 head
  if (a.draft_fp8 && a.draft_bf16) die("--draft-fp8 or --draft-bf16");
  if (a.draft_head4 && a.draft_bf16) die("--draft-head4 or --draft-bf16");
  if (a.slots < 1 || a.slots > model::kMaxSlots || a.slot < 0 || a.slot >= a.slots)
    die("--slots %d --slot %d: 1 to %d slots, a slot below their number", a.slots, a.slot, model::kMaxSlots);
  if (a.mode == "serve" && a.pool_free_gib < 0 && a.pool_tokens < 0)
    die("serve: the chunk pool's size: --pool-free-gib G (leave G GiB free on every Spark) and/or --pool-tokens N "
        "(a pool of N positions of conversation)");
  if (a.pool_slack_gib < 0 || a.front_mib < 0 || a.encoder_mib < 0 || a.media_mib < 0 || a.front_now_mib < 0 || a.encoder_now_mib < 0 || a.ids_host_mib < 0 ||
      a.frame_stage_mib < 1)
    die("--pool-slack-gib, --front-mib, --encoder-mib, --media-mib, --front-now-mib, --encoder-now-mib and --ids-host-mib must be >= 0, --frame-stage-mib >= 1");
  // the memory left to the host (--pool-free-gib) is sized beside the front-end's cap: without one the store cannot be sized (a start
  // by --pool-tokens alone leaves nothing aside and needs none)
  if (a.mode == "serve" && a.pool_free_gib >= 0)
    if (const char* why = model::rules::front_cap_refusal(a.front_mib)) die("serve: %s", why);
  if (a.pool_pick != "smaller" && a.pool_pick != "larger") die("--pool-pick smaller or larger");
  if (a.mode == "spectf") a.spec = true;
  if ((!a.tree_probe.empty() || a.tree_sib) && (a.mode != "spectf" || a.tree_probe.empty() || a.tree_sib < 0))
    die("--tree-probe FILE [--tree-sib N]: --mode spectf only");
  if (!a.seq_list.empty() && !(a.mode == "spectf" || (a.mode == "generate" && a.loop && a.big)))
    die("--seq-list FILE: --mode spectf, or generate --loop --big");
  if ((!a.route_file.empty() || !a.props.empty() || a.route_every || a.chain_every) &&
      (a.mode != "spectf" || a.route_file.empty() || a.route_every < 0 || a.route_every > model::kMaxSessRows ||
       a.chain_every < 0))
    die("--route-file FILE [--route-every 1..%d] [--chain-every N] [--props FILE]: --mode spectf only",
        model::kMaxSessRows);
  // the served environment (served.h): a variable the start's environment or a --setenv has set keeps its value
  served::env_for(a.mode == "serve", [](const char* name, const char* value) { setenv(name, value, 0); });
  if (a.spec_drafts < 0) a.spec_drafts = dspark::draft_rows();  // after every --setenv
  if (a.spec_drafts > model::ops::kDrafts) die("--spec-drafts must be 0..%d", model::ops::kDrafts);
  if (a.spec && (a.mode == "decode" || a.mode == "generate") && !a.big)
    die("--spec: the prompt goes through the prefill path (--big), which fills the draft module's windows");
  for (int k : a.ks) {
    if (k < 1 || k > model::kMaxRows) die("--ks: %d is not 1..%d", k, model::kMaxRows);
    if (k <= model::kMaxSessRows) continue;
    // past one session's rows: runs of several sessions only (bench), each a slot of its own
    bool ok = false;
    for (int G : a.bench_groups) ok |= G > 1 && G <= model::kMaxSessRows && G < k && k % G == 0 && k / G <= a.slots;
    if (a.mode != "bench" || a.bench_stride <= 0 || !ok)
      die("--ks %d: more than %d rows take runs of several sessions: --mode bench --bench-stride S --bench-groups G "
          "with G <= %d dividing %d, and --slots at least %d / G", k, model::kMaxSessRows, model::kMaxSessRows, k, k);
  }
  return a;
}

// What a kept session's state depends on besides the store's layout (state.cu checks that): this
// binary's bytes, the weights' and the Engram files' paths, sizes and times, and the options that change
// what a request computes. A start whose id differs from the stop's starts clean (a new build does).
uint64_t run_id(const Args& a) {
  uint64_t h = keepfile::fnv("dsgb10 kept sessions", 20);
  const int fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  if (fd < 0) die("run id: /proc/self/exe: %s", std::strerror(errno));
  std::vector<uint8_t> buf(1 << 20);
  for (ssize_t r; (r = read(fd, buf.data(), buf.size())) > 0;) h = keepfile::fnv(buf.data(), (size_t)r, h);
  close(fd);
  auto file = [&](const std::string& f) {
    struct stat s{};
    const int64_t v[3] = {stat(f.c_str(), &s) == 0 ? (int64_t)s.st_size : -1, (int64_t)s.st_mtim.tv_sec,
                          (int64_t)s.st_mtim.tv_nsec};
    h = keepfile::fnv(f.data(), f.size(), h);
    h = keepfile::fnv(v, sizeof v, h);
  };
  file(a.weights);
  if (!a.draft_weights.empty()) file(a.draft_weights);
  if (!a.no_engram) file(a.hashmap), file(a.shard[0]), file(a.shard[1]);
  // --ablit: the abliteration file's bytes, as an ablit session's state depends on them; not its time,
  // which every start's copy to the workers renews
  if (!a.ablit.empty()) {
    const int af = open(a.ablit.c_str(), O_RDONLY | O_CLOEXEC);
    if (af < 0) die("run id: %s: %s", a.ablit.c_str(), std::strerror(errno));
    for (ssize_t r; (r = read(af, buf.data(), buf.size())) > 0;) h = keepfile::fnv(buf.data(), (size_t)r, h);
    close(af);
  }
  const int64_t o[] = {a.max_ctx, a.exact, a.spec, a.draft_fp8, a.prefill_rows, a.small_rows, a.snap_every,
                       a.no_engram, a.layers, a.first, (int64_t)a.ring_cap, a.draft_head4, a.draft_markov8,
                       !a.ablit.empty()};
  return keepfile::fnv(o, sizeof o, h);
}

// /proc/meminfo's MemAvailable (free pages and the page cache the kernel can reclaim), bytes
size_t mem_available() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t kb = 0;
  while (std::fgets(line, sizeof line, f))
    if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
  std::fclose(f);
  return kb << 10;
}

// This process's resident host memory, bytes
size_t resident() {
  FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  size_t pages = 0, rss = 0;
  if (std::fscanf(f, "%zu %zu", &pages, &rss) != 2) rss = 0;
  std::fclose(f);
  return rss * (size_t)sysconf(_SC_PAGESIZE);
}

void mem_report(int rank, const char* when) {
  size_t dfree = 0, dtotal = 0;
  CUDA_CHECK(cudaMemGetInfo(&dfree, &dtotal));
  std::printf("rank %d: %s; free device memory %.2f GiB, available host memory %.2f GiB, resident %.2f GiB\n", rank,
              when, dfree / 1073741824.0, mem_available() / 1073741824.0, resident() / 1073741824.0);
}

}  // namespace

int main(int argc, char** argv) {
  heap::limit_arenas();
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const Args a = parse(argc, argv);
  {  // the exact shared steps' settings (model/exact_rule.h): a value outside its range stops the start
    char msg[160];
    if (model::exact::env_error(msg, sizeof msg)) die("%s", msg);
    if (model::copy::run_env_error(msg, sizeof msg)) die("%s", msg);  // the long copy steps' settings (model/copy.h)
  }
  const double t0 = now_s();
  std::printf("rank %d (%s): the fleet file %s; weights %s", a.rank, comm::fleet(a.fleet).rank[a.rank].name.c_str(),
              comm::fleet(a.fleet).file.c_str(), a.weights.c_str());
  if (!a.no_engram) std::printf("; Engram reads in batches of %d", a.engram_read_batch);
  std::printf("\n");
  // 1. CUDA, and every allocation but the weights' arena, before any file is read (docs/findings.md)
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  model::Options mo;
  mo.max_ctx = a.max_ctx;
  mo.n_layers = a.layers;
  mo.first_layer = a.first;
  mo.graphs = a.graphs;
  mo.prefill_rows = a.prefill_rows;
  mo.prefill_rows_max = a.prefill_rows_max;
  mo.wave_probe = a.wave_probe, mo.wave_probe_at = a.wave_probe_at;
  mo.draft_fp8 = a.draft_fp8;
  mo.draft_head4 = a.draft_head4, mo.draft_markov8 = a.draft_markov8;
  mo.ablit = a.ablit;
  mo.lend_null_mib = a.lend_null_mib;
  mo.tree_probe = !a.tree_probe.empty();
  mo.slots = a.slots;
  mo.big_slots = a.big_slots;
  mo.copy_mib = a.copy_mib;
  mo.pool_one_line = a.mode == "serve";  // it grows to the chosen size before serving (step 6)
  mo.kv_split = a.mode == "serve" && a.kv_split;
#ifndef PICO_OLD_RULES
  mo.imager = a.mode == "serve" && a.disk_gib > 0;  // every write to the disk tier is each Spark's own quarter, in the background (every mode)
#endif
  mo.kv_split_mode = a.kv_split_mode;
  mo.pscan = a.pscan;
  mo.kv_join_pieces = a.kv_join_pieces;
  mo.bgraph_bodies = a.bgraph_bodies, mo.bgraph_noif = a.bgraph_noif, mo.bgraph_check = a.bgraph_check;
  mo.l2pf = !a.no_l2pf;
  model::Model m(a.rank, mo);
  m.use_slot(a.slot);
  // 2. the ring (waits for the other three ranks)
  comm::Options ro;
  ro.fleet = a.fleet;
  ro.rank = a.rank;
  if (a.ring_cap) ro.cap = a.ring_cap;
  comm::Ring ring(ro);
  std::unique_ptr<comm::Ring> ring2;
  if (a.wave_probe) {
    comm::Options r2 = ro;
    r2.port = ro.port + 1, r2.proxy_cpu = 6;
    ring2 = std::make_unique<comm::Ring>(r2);
  }
  std::printf("rank %d: ring up after %.1f s\n", a.rank, now_s() - t0);
  // 3. weights (the DSpark blocks with --spec)
  dsw::WeightFile wf(a.weights, a.rank, comm::kRanks);  // refuses another rank's file
  dsw::LoadOptions lo;
  lo.threads = a.threads;
  // when the prefill path's gathered weights are made at start (step 5), this rank's own slice of them is loaded
  // straight into its slot of the gathered buffers and decode reads it there: the arena does not hold it too.
  // DSGB10_NO_GATHER_ALIAS=1: the arena keeps them (as without the prefill); DSGB10_ALIAS_CHECK=1: each aliased tensor's
  // CRC is checked at its slot right after the load.
  const bool prefill_at_start = a.mode == "serve" || (a.big && a.mode != "prefill" && a.mode != "bench");
  const bool alias_gathered = prefill_at_start && !std::getenv("DSGB10_NO_GATHER_ALIAS");
  if (alias_gathered) {
    m.reserve_gathered(!a.no_engram);
    lo.dest = [&m](const dsw::DeviceTensor& t) { return m.gathered_dest(t.name, t.bytes); };
    const char* chk = std::getenv("DSGB10_ALIAS_CHECK");
    lo.verify_dest = chk && *chk && std::strcmp(chk, "0") != 0;
  }
  lo.select = [&a](const dsw::DeviceTensor& t) {
    if (t.name.rfind("mtp.", 0) == 0) return a.spec && a.draft_weights.empty();
    if (t.name.rfind("layers.", 0) == 0) {
      const int L = std::atoi(t.name.c_str() + 7);
      return L >= a.first && L < a.layers;
    }
    return true;
  };
  const dsw::LoadStats ls = wf.load(lo);
  std::printf("rank %d: %.2f GiB of weights in %.1f s (%.2f GB/s)\n", a.rank, ls.bytes / 1073741824.0, ls.seconds, ls.gbps);
  if (alias_gathered) {
    size_t n = 0, bytes = 0;
    m.aliased(&n, &bytes);
    std::printf("rank %d: %zu weight tensors (%.2f MiB) loaded into this rank's slot of the whole buffers they are gathered into%s\n",
                a.rank, n, bytes / 1048576.0, lo.verify_dest ? ", CRC checked there" : "");
  }
  m.bind(wf, ring);
  if (ring2) m.bind_ring2(*ring2);
  // --draft-weights: another drafter's mtp.* (same shapes and formats) from its own rank file; embed and head
  // stay the model's. Drafts change the speed, never the tokens.
  std::unique_ptr<dsw::WeightFile> dwf;
  if (a.spec && !a.draft_weights.empty()) {
    dwf = std::make_unique<dsw::WeightFile>(a.draft_weights, a.rank, comm::kRanks);
    dsw::LoadOptions dlo;
    dlo.threads = a.threads;
    dlo.select = [](const dsw::DeviceTensor& t) { return t.name.rfind("mtp.", 0) == 0; };
    const dsw::LoadStats ds = dwf->load(dlo);
    std::printf("rank %d: drafter %s: %.2f GiB in %.1f s\n", a.rank, a.draft_weights.c_str(), ds.bytes / 1073741824.0,
                ds.seconds);
  }
  if (a.spec) m.bind_spec(wf, dwf.get());
  if (!a.head_raw) {  // after bind_spec: DSpark's FP8 head comes from the raw one
    const double t1 = now_s();
    int raw = 0;
    const size_t hb = m.code_head(&raw);
    std::printf("rank %d: the head coded in %.2f s: %.1f MB in place of %.1f (%d tiles raw)\n", a.rank, now_s() - t1,
                hb / 1e6, (double)model::kVocabPart * model::kDim * 2 / 1e6, raw);
  }
  // 3a. copy drafting's settings (DSGB10_COPY*, the table file's bytes) must be one set on the four ranks: a rank that differs verifies
  // other rows than the others and the ranks stop at the next collective. Every rank makes this call, whatever its own settings.
  {
    const model::copy::Settings mine = model::copy::settings();
    model::copy::Settings all[comm::kRanks];
    ring.allgather_host(&mine, all, sizeof mine, comm::kCallCopyAgree);
    for (int r = 0; r < comm::kRanks; ++r)
      if (std::memcmp(&all[r], &mine, sizeof mine) != 0)
        die("rank %d: copy drafting settings differ between the ranks (mode bits %u %u %u %u, shortest match %u %u %u %u, table file hash "
            "%016llx %016llx %016llx %016llx): set DSGB10_COPY, DSGB10_COPY_DUMP, DSGB10_COPY_ADAPT, DSGB10_COPY_MIN and DSGB10_COPY_TABLE alike "
            "on every Spark (and DSGB10_COPY_ROWS %u %u %u %u, DSGB10_COPY_ENTER %u %u %u %u)", a.rank, all[0].mode, all[1].mode, all[2].mode, all[3].mode, all[0].min_match, all[1].min_match, all[2].min_match,
            all[3].min_match, (unsigned long long)all[0].table_hash, (unsigned long long)all[1].table_hash, (unsigned long long)all[2].table_hash,
            (unsigned long long)all[3].table_hash, all[0].rows, all[1].rows, all[2].rows, all[3].rows, all[0].enter, all[1].enter,
            all[2].enter, all[3].enter);
  }
  // 3a'. the Gumbel sampler's settings (DSGB10_GSAMPLE, DSGB10_GCOUPLE, DSGB10_GCOUPLE_CF) must be one set on the four ranks: a rank with
  // another A, B computes another survival and chooses other rows, and the ranks stop at the next collective. Every rank makes this call.
  {
    const model::ops::GSettings mine = model::gsample_settings();
    model::ops::GSettings all[comm::kRanks];
    ring.allgather_host(&mine, all, sizeof mine, comm::kCallGSampleAgree);
    for (int r = 0; r < comm::kRanks; ++r)
      if (std::memcmp(&all[r], &mine, sizeof mine) != 0)
        die("rank %d: the Gumbel sampler settings differ between the ranks (mode %d %d %d %d, coupled %d %d %d %d, Td[0] %g %g %g %g, gate A,B "
            "%g,%g %g,%g %g,%g %g,%g): set DSGB10_GSAMPLE, DSGB10_GCOUPLE and DSGB10_GCOUPLE_CF alike on every Spark", a.rank, all[0].mode,
            all[1].mode, all[2].mode, all[3].mode, all[0].coupled, all[1].coupled, all[2].coupled, all[3].coupled, all[0].td[0], all[1].td[0],
            all[2].td[0], all[3].td[0], all[0].cf_a, all[0].cf_b, all[1].cf_a, all[1].cf_b, all[2].cf_a, all[2].cf_b, all[3].cf_a, all[3].cf_b);
  }
  // 3b. --vision-weights: this rank's quarter of the image encoder (dsw_convert --vision), into device memory before the
  // store is sized, so the pool's offer is taken from what is left. The file is checked against the 266 tensors this rank
  // must hold, and the four ranks compare a digest of their files before any byte is read: a file of another rank, another
  // cut or another checkpoint ends the start on every rank (the same flag, and so the same collective, on every rank).
  std::unique_ptr<vit::VisionWeights> vision;
  if (!a.vision_weights.empty()) {
    vision = std::make_unique<vit::VisionWeights>(a.vision_weights, a.rank);  // refuses another rank's file, or one that is not the rank's vision weights
    const uint64_t mine = vision->digest();
    uint64_t all[comm::kRanks];
    ring.allgather_host(&mine, all, sizeof mine, comm::kCallMain0 + 0);
    for (int r = 0; r < comm::kRanks; ++r)
      if (all[r] != mine)
        die("rank %d: the vision weight files are not one set: digests of ranks 0-3 %016llx %016llx %016llx %016llx "
            "(cut them together, from one checkpoint, with dsw_convert --vision)", a.rank, (unsigned long long)all[0],
            (unsigned long long)all[1], (unsigned long long)all[2], (unsigned long long)all[3]);
    const dsw::LoadStats vs = vision->load(a.threads);
    std::printf("rank %d: vision weights %s: %d tensors, %.2f MiB in %.2f s (%.2f GB/s), %.2f MiB on the device with the "
                "padded patch embedding; set digest %016llx\n", a.rank, vision->path().c_str(), vision->tensors(),
                vs.bytes / 1048576.0, vs.seconds, vs.gbps, vision->device_bytes() / 1048576.0, (unsigned long long)mine);
    const vit::VisionWeights::Held& vh = vision->held();
    std::printf("rank %d: vision weights held after the load: %.2f MiB on the device (the weights), pinned staging %.2f MiB "
                "at most and %llu B still held; readings around the load: device free fell by %.2f MiB, available host "
                "memory by %.2f MiB\n", a.rank, vision->device_bytes() / 1048576.0, vh.staging_peak / 1048576.0,
                (unsigned long long)vh.staging_held, vh.device_fell / 1048576.0, vh.host_fell / 1048576.0);    // the encoder in the engine: this rank's quarter, the pinned staging buffer for pixels (made here, so the sizing read below already
    // counts it) and tenant 3, the work area; the area itself is taken by the hand-over below
    m.set_vision(&vision->weights());
    std::printf("rank %d: vision encoder: %.1f MiB of pinned staging for pixels; work area to take from the pool: %llu MiB\n", a.rank,
                model::Model::kVitStageBytes / 1048576.0, (unsigned long long)model::Model::vision_area_mib());
  }
  // 4. Engram
  std::unique_ptr<engram::HashMap> emap;
  engram::Display display;  // before the store: released after it (munmap, GEM_CLOSE, close)
  std::unique_ptr<engram::Store> store;
  if (!a.no_engram) {
    emap = std::make_unique<engram::HashMap>(engram::HashMap::load(a.hashmap));
    engram::StoreConfig sc;
    sc.rank = a.rank;
    sc.shard[0] = a.shard[0];
    sc.shard[1] = a.shard[1];
    sc.cpu = 9;
    // serve: the cache starts empty over address space for --engram-cache-max-mib and is made by the hand-over from the chunk
    // pool (Model::engram_lend, after the pool is made); the other modes keep a fixed cache made here
    const bool lend = a.mode == "serve" && a.engram_cache_mib > 0 && a.engram_cache_max_mib > 0;
    const size_t want_bytes = (size_t)a.engram_cache_mib << 20;
    if (lend) sc.cache_max_bytes = (size_t)a.engram_cache_max_mib << 20;
    else sc.cache_bytes = want_bytes;
    // the row cache's slots are only ever touched by the CPU: its first slots go into the display reservation (a
    // headless Spark never uses it; engram/display.h), the rest stay in ordinary memory, and the pool's sizing (step 6)
    // counts only the ordinary part. Each rank makes the largest buffer it is given (at most the cache's size); every
    // rank uses the smallest rank's size, or none when any rank has none.
    if (a.engram_vram != "off" && want_bytes > 0) {
      const bool ok = display.create(a.engram_vram_card, a.engram_vram_sysfs,
                                     std::min(want_bytes, (size_t)a.engram_vram_mib << 20));
      const uint64_t mine = ok ? display.size() : 0;
      uint64_t sizes[4];
      ring.allgather_host(&mine, sizes, sizeof mine, comm::kCallMain0 + 1);
      const uint64_t least = std::min(std::min(sizes[0], sizes[1]), std::min(sizes[2], sizes[3]));
      if (least == 0) {
        const std::string why = ok ? "another rank has none" : display.why();
        display.destroy();
        std::printf("rank %d: engram row cache: display reservation not used (%s); all %.0f MiB in ordinary memory\n",
                    a.rank, why.c_str(), want_bytes / 1048576.0);
        if (a.engram_vram == "require") die("--engram-vram require: the display reservation is not available on every rank");
      } else {
        display.limit(least);
        sc.display_region = display.data();
        sc.display_bytes = display.capacity();
      }
    }
    sc.read_batch = a.engram_read_batch;  // the fleet file's, for this rank's SSD
    store = std::make_unique<engram::Store>(sc, *emap);
    m.set_engram(emap.get(), store.get());
    if (display.active())
      std::printf("rank %d: engram row cache: %.0f MiB of its %.0f MiB slots in the display reservation (of %.0f MiB; the "
                  "smallest size over the ranks), %.0f MiB in ordinary memory\n", a.rank,
                  store->cache_display_bytes() / 1048576.0, want_bytes / 1048576.0, display.capacity() / 1048576.0,
                  (want_bytes - std::min(want_bytes, store->cache_display_bytes())) / 1048576.0);
  }
  // 5. the prefill path's buffers and gathered weights now, not on the first request (serve, and the
  // modes that prefill through it with --big); every rank's free device memory afterwards goes to
  // rank 0's log
  if (prefill_at_start) {
    const uint64_t mine = m.prefill_init(ring);
    uint64_t all[4];
    ring.allgather_host(&mine, all, sizeof mine, comm::kCallMain0 + 2);
    if (a.rank == 0)
      std::printf("free device memory after the prefill buffers, ranks 0-3: %.2f %.2f %.2f %.2f GiB\n",
                  all[0] / 1073741824.0, all[1] / 1073741824.0, all[2] / 1073741824.0, all[3] / 1073741824.0);
  }
  // 6. serve: nothing takes device memory after startup (DESIGN.md's hardening list: CUDA takes only free
  // pages, and the page cache grows into them). The spec step's graphs of every context bucket first, then
  // the disk tier, then the chunk pool to its size: each rank offers a count (--pool-free-gib, --pool-tokens)
  // and every rank takes the smallest (below).
  const std::string disk_dir = a.disk_dir;  // serve: from the fleet file unless given
  const std::string disk_tag = a.disk_tag.empty() ? "" : "." + a.disk_tag;
  const std::string disk_base = disk_dir + "/rank" + std::to_string(a.rank) + disk_tag;
  uint64_t store_numbers = 0;  // serve: the store's size, for STATS
  if (a.mode == "serve") {
    if (a.spec) m.spec_graphs();
    const double gib = 1073741824.0;
    // the store's disk tier first, as its staging is pinned host memory: each rank offers its directory's
    // free space less 32 GiB (its own file counted as free), at most --disk-gib; every rank takes the
    // smallest offer (under 1 GiB: none). Kept sessions: when every rank's index of an earlier planned
    // stop checks out (state_kept_check) with the same generation, and its file fits every offer, every
    // rank opens that file as it is; anything else starts clean and removes the kept files, as does
    // --keep-gib 0 (nothing kept, nothing taken back).
    const bool disk = a.disk_gib > 0 && !in_git_tree(disk_dir);
    if (a.disk_gib > 0 && !disk) std::printf("rank %d: no disk tier: %s is in a git tree\n", a.rank, disk_dir.c_str());
    if (disk) {
      const uint64_t fr = spill_free(disk_dir, disk_base), margin = 32ull << 30;
      size_t kbytes = 0;
      std::string why;
      const double tk0 = now_s();
      const uint64_t id = run_id(a);
      const double tk1 = now_s();
      const uint64_t gen = a.keep_gib > 0 ? m.state_kept_check(disk_base, id, a.keep_days * 86400, &kbytes, &why) : 0;
      const double tk2 = now_s();
      if (!why.empty()) std::printf("rank %d: kept sessions: none reused (%s)\n", a.rank, why.c_str());
      const uint64_t mine[3] = {std::min<uint64_t>((uint64_t)(a.disk_gib * gib), fr > margin ? fr - margin : 0), gen,
                                kbytes};
      uint64_t all[4][3];
      ring.allgather_host(mine, all, sizeof mine, comm::kCallMain0 + 3);
      uint64_t bytes = mine[0];
      bool reuse = gen != 0;
      for (const auto& x : all) bytes = std::min(bytes, x[0]), reuse = reuse && x[1] == gen && x[2] == kbytes;
      reuse = reuse && kbytes <= bytes;
      if (reuse) bytes = kbytes;
      if (bytes >= (1ull << 30)) m.state_disk(disk_base, bytes, 512ull << 20, reuse, id);
      std::printf("session store's disk tier: %.1f GiB in %s (at most %.1f; free space less 32 GiB, ranks 0-3: "
                  "%.0f %.0f %.0f %.0f GiB)\n", m.state_disk_capacity() / gib,
                  m.state_disk_capacity() ? m.state_disk_path().c_str() : "(none)", a.disk_gib, all[0][0] / gib,
                  all[1][0] / gib, all[2][0] / gib, all[3][0] / gib);
      if (a.rank == 0)
        std::printf("kept sessions: %s (generations of ranks 0-3: %016llx %016llx %016llx %016llx; the run id %016llx "
                    "in %.1f ms, this rank's index checked in %.1f ms)\n",
                    reuse && m.state_disk_capacity() ? "the files of the last planned stop are reused" : "none, a clean start",
                    (unsigned long long)all[0][1], (unsigned long long)all[1][1], (unsigned long long)all[2][1],
                    (unsigned long long)all[3][1], (unsigned long long)id, 1e3 * (tk1 - tk0), 1e3 * (tk2 - tk1));
    }
    if (!m.state_disk_capacity()) {  // no disk tier: nothing kept stays
      std::error_code ec;
      std::filesystem::remove(disk_base + ".spill", ec);
      std::filesystem::remove(disk_base + ".index", ec);
      std::filesystem::remove(disk_base + ".index.new", ec);
    }
    // the chunk pool's size (docs/engine.md, "The chunk pool"): each rank's count from what it would leave free
    // (--pool-free-gib: of device memory and of available host memory, the Engram cache still to grow counted as
    // used) and/or from positions (--pool-tokens: 5 numbers every 16,384), the smaller or the larger of the two
    // (--pool-pick); every rank takes the smallest rank's count. The pool so far (one line) counts in it.
    m.split_prepare();  // with --kv-split cache: the split's buffers, before the read, so that it is already short by them
    size_t dfree = 0, dtotal = 0;
    CUDA_CHECK(cudaMemGetInfo(&dfree, &dtotal));
    const size_t avail = mem_available();
    // the Engram cache still to grow, in ordinary memory: its slots in the display reservation are not counted. A cache made
    // by the hand-over (Model::engram_lend, below) is carved out of the pool after it is made: not counted here
    const bool carved = store && store->cache_max_bytes() > 0;
    const size_t cache = a.no_engram || carved ? 0 : ((size_t)a.engram_cache_mib << 20) - (store ? store->cache_display_bytes() : 0);
    const model::Model::PoolStats p0 = m.pool_stats();
    const uint64_t img = p0.number_bytes, have = p0.numbers;
    // this rank's ledger (pool_rules.h): the lines it holds or may grow to outside the store, each with its cap and what it holds at
    // this read; those only rank 0 holds are rank 0's, from the options naming their sizes. The store is sized beside them.
    namespace rules = model::rules;
    rules::Ledger ledger;
    ledger.set(rules::kHeap, heap::kCap, heap::in_use(), true);
    ledger.set(rules::kEngramCache, cache, 0, true);
    // state_reserve's gather buffer (model/state.cu), made after this read: its size as the allocation computes it (the larger of one
    // number's image and the end and window snapshot blocks with the draft module's, in 64 KiB grains)
    const uint64_t gather_bytes = model::detail::Pool::round(std::max<size_t>({(size_t)img, model::detail::kEndBytes + model::detail::kSpecEndBytes, model::detail::kSnapBytes + model::detail::kSpecSnapBytes}));
    ledger.set(rules::kGather, gather_bytes, 0, true);
    const uint64_t split_bytes = rules::split_buffer_bytes(a.kv_split || mo.imager) + rules::imager_buffer_bytes(mo.imager, gather_bytes);
    ledger.set(rules::kSplit, split_bytes, split_bytes, true);  // made before the read below (split_prepare)
    ledger.set(rules::kDrift, (uint64_t)(a.pool_slack_gib * gib), 0, false);
    if (a.rank == 0) {
      ledger.set(rules::kFront, (uint64_t)a.front_mib << 20, (uint64_t)a.front_now_mib << 20, true);
      // the image process beside the engine: with --vision-weights the engine encodes images itself (tenant 3 holds its memory
      // inside the store) and the process is the media worker, capped by its container at --media-mib; without them it is the
      // encoder container, capped at --encoder-mib
      ledger.set(rules::kEncoder, rules::image_process_cap(vision != nullptr, a.encoder_mib, a.media_mib), (uint64_t)a.encoder_now_mib << 20, false);  // host part by its container, device part not
      ledger.image_what = vision ? 1 : 2;
      ledger.set(rules::kOutQueue, bounds::kOutCap, 0, true);
      ledger.set(rules::kFrames, 2 * ((uint64_t)a.frame_stage_mib << 20), 0, true);  // a frame's payload and the request's copy of its rows
      ledger.set(rules::kIds, (uint64_t)a.ids_host_mib << 20, 0, !disk_dir.empty());  // empty at the start: nothing held yet
    }
    const uint64_t outside = rules::outside_sum(ledger);
    uint64_t by_free = UINT64_MAX, by_tokens = UINT64_MAX;
    if (a.pool_free_gib >= 0) {
      const size_t keep = (size_t)(a.pool_free_gib * gib);
      by_free = model::rules::store_offer(have, dfree, avail, keep, outside, cache, img);
    }
    if (a.pool_tokens >= 0) by_tokens = ((uint64_t)a.pool_tokens * 5 + 16383) / 16384;
    const uint64_t want = a.pool_free_gib < 0 ? by_tokens
                          : a.pool_tokens < 0 ? by_free
                          : a.pool_pick == "larger" ? std::max(by_free, by_tokens) : std::min(by_free, by_tokens);
    struct Mine {
      uint64_t dfree, avail, offer;
      rules::Ledger ledger;
    } mine{dfree, avail, std::max(want, have), ledger};
    Mine all[model::rules::kRanks];
    alloc_seal::open();  // the sizing read is done: from here every allocation call is in a scope of its class or counted as outside
    ring.allgather_host(&mine, all, sizeof mine, comm::kCallMain0 + 4);
    uint64_t pool = mine.offer;
    for (const auto& r : all) pool = std::min(pool, r.offer);
    store_numbers = pool;
    m.state_reserve((size_t)(pool - have) * img);
    m.split_zone_update();  // with --kv-split cache: the lend zone, one collective, before the first command
    if (a.rank == 0 && a.kv_split && !mo.imager)
      std::printf("kv split (%s): a conversation is split between decoding runs, never inside one: each Spark keeps a quarter; a returning "
                  "conversation takes its numbers back when nobody used them, else the Sparks exchange quarters; split commands of at most %d "
                  "pieces, one return command of at most %d pieces; the split's buffers %.1f MiB on every rank (the start table's line)\n",
                  a.kv_split_mode == 1 ? "tiered" : "max", a.kv_split_budget, a.kv_join_pieces, model::rules::split_buffer_bytes(true) / 1048576.0);
    if (a.rank == 0) {
      std::printf("chunk pool: %llu numbers of %.2f MiB (%.2f GiB, %.1fM positions) on every rank (", (unsigned long long)pool,
                  img / 1048576.0, pool * (double)img / gib, pool * 16384.0 / 5 / 1e6);
      if (a.pool_free_gib >= 0) std::printf("--pool-free-gib %.1f: %llu on this rank", a.pool_free_gib, (unsigned long long)by_free);
      if (a.pool_free_gib >= 0 && a.pool_tokens >= 0) std::printf(", ");
      if (a.pool_tokens >= 0) std::printf("--pool-tokens %lld: %llu", (long long)a.pool_tokens, (unsigned long long)by_tokens);
      if (a.pool_free_gib >= 0 && a.pool_tokens >= 0) std::printf(", the %s", a.pool_pick.c_str());
      std::printf("; free device / available host memory before it, ranks 0-3:");
      for (const auto& r : all) std::printf(" %.2f / %.2f", r.dfree / gib, r.avail / gib);
      std::printf(" GiB, the Engram cache's %.1f to grow)\n", cache / gib);
      if (vision && a.pool_free_gib >= 0) {  // loaded before the reads above, so they are already lower by its bytes
        const uint64_t vb = vision->device_bytes();
        std::printf("vision weights: %llu bytes on every rank (%.2f numbers of %.2f MiB)\n", (unsigned long long)vb,
                    vb / (double)img, img / 1048576.0);
      }
    }
    // the table and the start check: with --pool-free-gib the store must hold one conversation of --max-ctx, the Engram cache's groups
    // (as many as the lender can give: a larger cache is cut, never a refusal), one largest image request's rows and the largest window
    // a request may open; and none of the engine's own lines may hold more than its cap
    rules::Ledger ledgers[rules::kRanks];
    rules::StoreRead rd;
    uint64_t projected[rules::kRanks];
    for (int r = 0; r < rules::kRanks; ++r) {
      ledgers[r] = all[r].ledger;
      rd.dfree[r] = all[r].dfree, rd.avail[r] = all[r].avail;
      projected[r] = rules::projected_available(all[r].avail, (pool - have) * img, ledgers[r]);
    }
    const uint64_t whole = ((uint64_t)a.max_ctx * 5 + 16383) / 16384 + 2;  // one conversation of --max-ctx (5 numbers per 16,384 positions) and the guard
    uint64_t engram_groups = 0;
    if (carved) {
      const uint64_t gb = (uint64_t)model::lend::kGroupNums * img;
      engram_groups = std::min<uint64_t>(model::lend::sizes((uint64_t)a.engram_cache_mib, store->cache_display_bytes(), engram::kRowBytes, gb).groups,
                                         model::lend::max_total_groups(pool, whole));
    }
    const uint64_t window_groups = a.max_ctx > 65536 * 16 ? (uint64_t)m.window_groups_estimate(64) : 0;  // the largest window a request may open
    const uint64_t image_nums = model::imgrows::numbers_for(((uint64_t)a.frame_stage_mib << 20) / model::imgrows::kRowBytes);
    const uint64_t need_min = rules::need_min_numbers(whole, engram_groups, image_nums, window_groups);
    const size_t keep_bytes = a.pool_free_gib >= 0 ? (size_t)(a.pool_free_gib * gib) : 0;
    const rules::StoreCheck chk = rules::store_check(ledgers, pool, need_min);
    if (a.rank == 0) {
      std::printf("%s", rules::store_table(ledgers, rd, have, cache, pool, img, keep_bytes, engram_groups, window_groups, chk).c_str());
      if (!rules::unequal_lines(ledgers).empty()) std::printf("memory ledger: a line differs between ranks (marked above)\n");
      for (int r = 0; r < rules::kRanks; ++r)
        if (projected[r] < rules::kProjectedFloor)
          std::printf("WARNING: rank %d is projected to have %.2f GiB of memory available once the pool, the Engram cache and "
                      "the fixed lines are taken (below %.0f GiB)\n", r, projected[r] / gib, rules::kProjectedFloor / gib);
    }
    if (a.pool_free_gib >= 0 && chk.refuse) {
      if (chk.why == rules::Refuse::OwnLineOver)
        die("the start is refused: rank %d's own line %s holds %.2f GiB at the read, more than its cap of %.2f GiB (table above)", chk.rank,
            rules::kLines[chk.line].name, ledgers[chk.rank].now[chk.line] / gib, ledgers[chk.rank].cap[chk.line] / gib);
      die("the start is refused: the store of %llu numbers cannot hold one conversation of --max-ctx %d, the Engram cache, one largest image request "
          "and the largest window (%llu numbers; table above; lower --front-mib, --encoder-mib, --frame-stage-mib, --ids-host-mib or --pool-slack-gib, "
          "or --pool-free-gib)", (unsigned long long)chk.store, a.max_ctx, (unsigned long long)chk.need);
    }
    srv::mem_watch_init(&ring, keep_bytes);
    srv::split_ride_init(&ring);  // the ride's host all-gather: without it every rank rode alone
    // the Engram cache from the pool's groups (model/tenant.cu): the pool is made whole, so the groups that leave are real memory.
    // A size the pool cannot lend is cut to what fits, a failed hand-over leaves the cache empty and the engine serving; the
    // server asks again (a start is never refused by the cache)
    if (carved && m.engram_lend_enabled()) {
      int st = m.engram_lend(model::Model::kLendBegin, a.engram_cache_mib);
      while (st == model::Model::kLendPending) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        st = m.engram_lend(model::Model::kLendPoll, 0);
      }
      const model::Model::EngramLendInfo li = m.engram_lend_info();
      std::printf("rank %d: engram row cache: %llu MiB asked for, %llu MiB made (%llu groups of the pool, %.0f MiB resident in the block, "
                  "%llu %% in huge pages)%s\n", a.rank, (unsigned long long)a.engram_cache_mib, (unsigned long long)li.mib,
                  (unsigned long long)li.groups, li.block_bytes / 1048576.0, (unsigned long long)li.huge_pct,
                  st == model::Model::kLendDone ? "" : li.reason[0] ? (std::string("; the hand-over failed: ") + li.reason).c_str() : "; not made now");
    }
    // the image encoder's work area (model/vit_area.cu): tenant 3, device memory, two groups of the pool, taken once, here, before any
    // request and before kept conversations are read back, and kept while the engine runs. A take that fails on any rank fails on all:
    // the engine serves text and answers a pixel request with an error until the next start
    if (vision) {
      int st = m.tenant_lend(model::lend::kTenantWork, model::Model::kLendBegin, (int64_t)model::Model::vision_area_mib());
      while (st == model::Model::kLendPending) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        st = m.tenant_lend(model::lend::kTenantWork, model::Model::kLendPoll, 0);
      }
      std::printf("rank %d: vision work area: %llu MiB asked for, %.0f MiB held on this rank (status %d): %s\n", a.rank,
                  (unsigned long long)model::Model::vision_area_mib(), m.vision_area_held() / 1048576.0, st,
                  m.vision_state() == 1 ? "the engine encodes images" : "the engine has no image encoder until the next start");
    }
    // the prefill's second set of row buffers (model/rows_lend.cu): tenant 4, device and pinned host memory, whole groups of the pool, taken
    // here while the pool is empty so that no prompt waits for it; the server gives it back under pressure and asks again when room returns
    // (serve.cpp loan_policy). A take that fails on any rank fails on all: every chunk is --prefill-rows rows.
    if (m.prefill_loan_mib() > 0) {
      int st = m.tenant_lend(model::lend::kTenantPrefill, model::Model::kLendBegin, m.prefill_loan_mib());
      while (st == model::Model::kLendPending) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        st = m.tenant_lend(model::lend::kTenantPrefill, model::Model::kLendPoll, 0);
      }
      std::printf("rank %d: prefill rows: %lld MiB asked for (chunks of up to %d rows), status %d: %s\n", a.rank,
                  (long long)m.prefill_loan_mib(), a.prefill_rows_max, st,
                  st == model::Model::kLendDone ? "the second set is made" : "chunks stay at the standing rows for now");
    }
    if (a.rank == 0)  // the pool's tenants (lend.h's table), one line each
      for (const model::Model::TenantRow& t : m.tenant_rows())
        std::printf("store tenant %d, %s (%s, %s): %d groups of %.0f MiB (%.0f MiB held by its body), %d in flight\n", t.id, model::lend::tenant_name(t.id),
                    t.kind == model::lend::kKindHost ? "host memory" : "device memory",
                    t.standing == model::lend::kRequired ? "required" : t.standing == model::lend::kCache ? "cache" : "on demand", t.groups,
                    m.group_bytes() / 1048576.0, t.size_bytes / 1048576.0, t.pending);
  }
  ring.barrier();
  if (a.mode == "serve" && a.node_stats_port > 0) {  // the dashboard's node sampler (server/node_stats.h)
    size_t dfree = 0, dtotal = 0;
    CUDA_CHECK(cudaMemGetInfo(&dfree, &dtotal));
    node_stats::Memory nm;
    std::error_code ec;
    nm.device_bytes = dtotal;
    nm.weights_bytes = std::filesystem::file_size(a.weights, ec);
    if (ec) nm.weights_bytes = 0;
    if (vision) nm.weights_bytes += vision->device_bytes();
    nm.kv_bytes = m.state_capacity();
    const uint64_t used = dtotal - dfree;
    nm.work_bytes = used > nm.weights_bytes + nm.kv_bytes ? used - nm.weights_bytes - nm.kv_bytes : 0;
    node_stats::start(a.rank, a.node_stats_addr, a.node_stats_port, nm, a.node_stats_rails);
  }
  pin_self(15);
  mem_report(a.rank, ("ready after " + std::to_string((int)(now_s() - t0)) + " s").c_str());
  {
    char line[256];
    alloc_seal::ready_line(line, sizeof line);
    std::printf("%s\n", line);
  }
#ifdef DSGB10_TEST_FAULTS
  // DSGB10_ALLOC_OUTSIDE=rank:N: on rank N, 30 s after ready, one unscoped 1 MiB cudaMalloc, so that a test sees the seal count a call
  // that no scope names
  if (const char* ao = std::getenv("DSGB10_ALLOC_OUTSIDE")) {
    int fr = -1;
    if (std::sscanf(ao, "rank:%d", &fr) == 1 && fr == a.rank)
      std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        cudaSetDevice(0);
        void* p = nullptr;
        if (cudaMalloc(&p, 1u << 20) == cudaSuccess) cudaFree(p);
        else cudaGetLastError();
      }).detach();
  }
#endif
  if (a.mode == "serve") {  // the byte account at ready, by rank (common/byte_account.h): the unexplained remainder is its own row
    size_t dfree = 0, dtotal = 0;
    CUDA_CHECK(cudaMemGetInfo(&dfree, &dtotal));
    byte_account::Sample mine = byte_account::read_system(dtotal - dfree), all[model::rules::kRanks];
    ring.allgather_host(&mine, all, sizeof mine, comm::kCallMain0 + 5);
    if (a.rank == 0) std::printf("%s", byte_account::table(all, model::rules::kRanks).c_str());
  }

  Watchdog wd(a.step_timeout_s, ring);
  if (a.rank != 0) {  // follow rank 0
    follow(m, ring, wd, a.rank);
    mem_report(a.rank, "done");
    return 0;
  }

  if (a.mode == "serve") {
    ServeOptions so;
    so.socket = a.socket;
    so.max_ctx = a.max_ctx;
    so.prefill_rows = a.prefill_rows;
    so.prefill_rows_max = a.prefill_rows_max;
    so.loan_low_groups = a.loan_low_groups, so.loan_high_extra = a.loan_high_extra;
    so.small_rows = a.small_rows;
    so.snap_every = a.snap_every;
    so.slice = a.slice;
    so.grow_ahead = a.grow_ahead;
    so.ssd_flush_s = a.ssd_flush_s;
    so.kv_split = a.kv_split;
    so.kv_split_mode = a.kv_split_mode;
    so.kv_split_after_s = a.kv_split_after_s;
    so.kv_split_high = a.kv_split_high;
    so.kv_split_low = a.kv_split_low;
    so.kv_split_pass_ms = a.kv_split_pass_ms;
    so.kv_split_budget = a.kv_split_budget;
    so.imager = mo.imager;
    so.kv_split_ride_ms = a.kv_split_ride_ms, so.kv_split_ride_pieces = a.kv_split_ride_pieces;
    so.kv_join_whole_ms = a.kv_join_whole_ms;
    so.kv_join_pieces = a.kv_join_pieces;
    so.window_idle_s = a.window_idle_s;
    so.engram_mib = a.engram_cache_mib;
    so.engram_min_mib = a.engram_cache_min_mib < 0 ? a.engram_cache_mib : a.engram_cache_min_mib;
    so.engram_max_mib = a.engram_cache_max_mib;
    so.interleave = a.il;
    so.interleave.prefill_rows = a.prefill_rows;
    if (a.exact) so.interleave.on = false;  // --exact reads every prompt whole
    if (so.interleave.whole_rows < std::max(a.prefill_rows, a.prefill_rows_max)) so.interleave.whole_rows = std::max(a.prefill_rows, a.prefill_rows_max);
    so.snap_thin = !a.snap_all;
    so.keep_index = disk_dir + "/server" + disk_tag + ".index";
    so.keep_bytes = std::min(m.state_disk_capacity(), (size_t)(a.keep_gib * 1073741824.0));  // none without a file
    so.keep_age_s = a.keep_days * 86400;
    so.frame_stage_bytes = (uint64_t)a.frame_stage_mib << 20;
    so.ids_dir = disk_dir.empty() ? "" : disk_dir + "/ids";
    so.ids_host_bytes = (uint64_t)a.ids_host_mib << 20;
    so.store_numbers = store_numbers;
    so.extend = a.extend;
    so.ced = !a.exact;
    so.spec = a.spec;
    so.spec_drafts = a.spec_drafts;
    so.spec_thresh = a.spec_thresh;
    std::copy(a.spec_cost, a.spec_cost + 3, so.spec_cost);
    so.spec_gate = a.spec_gate;
    so.profile_from_req = a.profile_from_req;
    std::copy(a.spec_row_ms, a.spec_row_ms + model::ops::kDrafts + 1, so.spec_row_ms);
    const int rc = serve(m, ring, wd, so);
    mem_report(a.rank, "stopped");
    return rc;
  }

  // ---- rank 0: drive ----
  int32_t sampled[model::kMaxRows];
  golden::Golden g(a.golden);
  if (a.onset >= 0) g.set_split(a.onset);
  g.set_per_position(a.per_position);
  const int L = a.limit > 0 ? std::min(a.limit, g.length()) : g.length();
  const int prompt = a.prompt > 0 ? a.prompt : a.mode == "prefill" ? L : 39;
  const uint32_t check = model::kFlagDump | model::kFlagLogits | (a.inject ? model::kFlagInject : 0) |
                         (a.inject_sub ? model::kFlagInjectSub : 0);
  double t_pref = 0, t_dec = 0, t_bcast = 0;
  int n_pref = 0, n_dec = 0;
  uint64_t chosen_hash = 1469598103934665603ull;  // FNV-1a over every row's chosen token (graphs vs eager)
  auto run = [&](int pos0, int R, const int32_t* toks, uint32_t flags) {
    Cmd c{};
    c.op = kCmdStep;
    c.rows = R;
    c.pos0 = pos0;
    c.flags = flags;
    for (int r = 0; r < R; ++r) c.tokens[r] = toks[r];
    c.sp.temperature = a.temp;
    c.sp.seed = a.seed;
    const double tb = now_s();
    bcast(ring, c);
    const double ts = now_s();
    t_bcast = ts - tb;
    wd.started = ts;
    m.step(to_ctl(c), sampled, &g);
    wd.started = 0;
    const double t = now_s() - ts;
    for (int r = 0; r < R; ++r) {
      float v = m.last_logit(r);
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      chosen_hash = (chosen_hash ^ (uint32_t)sampled[r]) * 1099511628211ull;
      chosen_hash = (chosen_hash ^ bits) * 1099511628211ull;
    }
    return t;
  };
  // The decode loop from rank 0: tok_0 at position pos0, n tokens; times[i] when tok_{i+1} arrived
  // (step_times: when each step's first token did).
  std::vector<double> step_times;
  auto run_decode = [&](int pos0, int32_t tok0, int n, std::vector<int32_t>& out, std::vector<double>& times) {
    Cmd c{};
    c.op = kCmdDecode;
    c.pos0 = pos0;
    c.tokens[0] = tok0;
    c.sp.temperature = a.temp;
    c.sp.top_p = a.top_p;
    c.sp.top_k = model::gsample_top_k(a.top_k, a.temp, a.top_p, true);  // the tools' seed is a client's
    c.sp.seed = a.seed;
    c.max_new = n;
    c.limit = a.max_ctx;
    c.ahead = a.ahead;
    c.spec = a.spec;
    c.spec_drafts = a.spec_drafts;
    c.spec_thresh = a.spec_thresh;
    std::copy(a.spec_cost, a.spec_cost + 3, c.spec_cost);
    c.spec_gate = a.spec_gate;
    std::copy(a.spec_row_ms, a.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    if (a.profile) c.flags |= model::kFlagProfile;
    if (!a.seq_list.empty()) c.n_stop = 1, c.stop[0] = 1;  // EXPERIMENT (--seq-list): a sequence ends at EOS
    bcast(ring, c);
    step_times.clear();
    if (a.profile) CUDA_CHECK(cudaProfilerStart());
    const DecodeEnd e = decode_loop(m, c, wd, [&](int, const model::SampleOut& o, int row) {
      times.push_back(now_s());
      if (row == 0) step_times.push_back(times.back());
      out.push_back(o.row[row].token);
      uint32_t bits;
      std::memcpy(&bits, &o.row[row].value, 4);
      chosen_hash = (chosen_hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
      chosen_hash = (chosen_hash ^ bits) * 1099511628211ull;
    });
    if (a.profile) CUDA_CHECK(cudaProfilerStop());
    return e;
  };
  auto spec_report = [&](const DecodeEnd& e) {
    if (!e.vsteps) return;
    std::printf("  drafts: %d verify steps, %.3f drafts verified and %.3f accepted per step (%.1f%%), "
                "%.3f tokens per step; steps verifying 0..5 drafts: %d %d %d %d %d %d\n",
                e.vsteps, (double)e.verified / e.vsteps, (double)e.accepted / e.vsteps,
                e.verified ? 100.0 * e.accepted / e.verified : 0.0, 1.0 + (double)e.accepted / e.vsteps, e.rows_hist[0],
                e.rows_hist[1], e.rows_hist[2], e.rows_hist[3], e.rows_hist[4], e.rows_hist[5]);
    if (e.rows_hist[6] + e.rows_hist[7]) std::printf("  steps verifying 6, 7 drafts: %d %d\n", e.rows_hist[6], e.rows_hist[7]);
    std::printf("  verify-length gate (%s): mean %.3f, min %.3f, max %.3f over %d steps\n",
                a.spec_gate == model::ops::kGateRatio      ? "ratio; lambda, tokens per ms"
                : a.spec_gate == model::ops::kGateMarginal ? "marginal; lambda, tokens per ms"
                                                           : "lambda * dC",
                e.gate_sum / std::max(1, e.gate_n), e.gate_min, e.gate_max, e.gate_n);
    std::printf("  spec step GPU ms (rows, draft pass) by rows:");
    for (int r = 1; r <= model::kMaxRows; ++r)
      if (e.gpu_n[r]) std::printf(" %d: %.2f (%d)", r, e.gpu_ms[r] / e.gpu_n[r], e.gpu_n[r]);
    std::printf("\n");
  };
  // Prefill positions [p0, p0 + n) through the many-row path, in chunks of --prefill-rows; returns the
  // last row's sample.
  bool ced_now = a.ced;  // big_prefill's CED (--mode batch: the session's)
  model::SampleOut prompt_out{};  // big_prefill's last sample (tf hashes it when it teacher-forces nothing)
  auto big_prefill = [&](int p0, int n, const int32_t* t) {
    Cmd c{};
    c.op = kCmdTokens;
    c.pos0 = std::max(0, p0 - 3);
    c.n = p0 + n - c.pos0;
    bcast(ring, c);
    std::vector<int32_t> v(t - (p0 - c.pos0), t + n);
    bcast_payload(ring, v.data(), v.size() * 4, 0);
    m.set_tokens(c.pos0, c.n, v.data());
    int32_t last = -1;
    for (int q = p0; q < p0 + n;) {
      Cmd k{};
      k.op = kCmdPrefill;
      k.pos0 = q;
      k.rows = std::min(a.prefill_rows, p0 + n - q);
      k.next = std::min(a.prefill_rows, p0 + n - q - k.rows);
      k.flags = (q + k.rows == p0 + n ? kPfSample : 0) | (ced_now && n >= 128 ? kPfCed : 0);
      k.sp.temperature = a.temp;
      k.sp.seed = a.seed;
      k.sp.n_top = model::ops::kTopLogprobs;
      const bool prof = q == a.profile_at;  // rank 0's (nsys wraps it)
      if (prof) CUDA_CHECK(cudaProfilerStart());
      const double tb = now_s();
      bcast(ring, k);
      wd.started = now_s();
      const uint64_t s = run_prefill(m, k);
      if (s != ~0ull) {
        const model::SampleOut& o = m.wait(s);
        last = o.row[0].token;
        prompt_out = o;
        std::printf("first token (position %d%s): %d, logprob %.5f; top logprobs:", q + k.rows - 1,
                    (k.flags & kPfCed) ? ", CED" : "", last, o.row[0].logprob);
        for (int j = 0; j < o.n_top; ++j) std::printf(" %d:%.5f", o.top_id[0][j], o.top_lp[0][j]);
        std::printf("\n");
      }
      m.drain();
      if (prof) CUDA_CHECK(cudaProfilerStop());
      wd.started = 0;
      const double dt = now_s() - tb;
      t_pref += dt;
      ++n_pref;
      if (!a.quiet) std::printf("prefill [%d, %d): %.1f ms, %.0f tok/s\n", q, q + k.rows, 1e3 * dt, k.rows / dt);
      q += k.rows;
    }
    return last;
  };
  std::vector<int32_t> toks(L);
  for (int p = 0; p < L; ++p) toks[p] = g.token(p);
  // decode and bench run at each of --ctxs in turn (ascending), each reached from where the last ended
  const std::vector<int> ctxs = a.ctxs.empty() ? std::vector<int>{a.ctx} : a.ctxs;
  // positions [p, ctx) of the golden tokens: through the prefill path (--big) or 8-row steps; returns
  // the last row's sample
  auto reach = [&](int p, int ctx) {
    t_pref = 0;
    n_pref = 0;
    if (a.big) return big_prefill(p, ctx - p, &toks[p]);
    for (int q = p; q < ctx; q += a.chunk) {
      t_pref += run(q, std::min(a.chunk, ctx - q), &toks[q], 0);
      if (!a.quiet && ++n_pref % 512 == 0) std::printf("prefill %d/%d positions\n", q + a.chunk, ctx);
    }
    n_pref = (ctx - p + a.chunk - 1) / a.chunk;
    return (int32_t)sampled[(ctx - 1 - p) % a.chunk];
  };

  if (a.mode == "decode") {
    for (size_t c = 0, p = 0; c < ctxs.size(); ++c) {
      const int ctx = ctxs[c];
      if (ctx <= (int)p || ctx + a.steps >= L)
        die("decode: context %d after %zu positions, %d steps, %d golden tokens", ctx, p, a.steps, L);
      const int32_t first = reach((int)p, ctx);
      std::printf("prefill: %d positions in %.2f s (%.0f tok/s); first token %d\n", ctx - (int)p, t_pref,
                  (ctx - (int)p) / t_pref, first);
      std::vector<int32_t> out;
      std::vector<double> times;
      const double t0d = now_s();
      const DecodeEnd e = run_decode(ctx, first, a.steps, out, times);
      if (c + 1 == ctxs.size()) bcast(ring, Cmd{});  // stop
      const int n = (int)times.size(), skip = std::min(8, n / 4);  // the first steps capture graphs
      std::printf("decode: context %d, %d tokens (%s, T %.2f top-p %.2f top-k %d%s), %d positions fed, end %u; "
                  "tokens hash %016llx\n",
                  ctx, n, a.graphs ? "graphs" : "eager", a.temp, a.top_p, a.top_k, a.spec ? ", DSpark drafts" : "",
                  e.fed, e.why, (unsigned long long)chosen_hash);
      spec_report(e);
      std::printf("  first token after %.2f ms; then %.3f ms per token (%.2f tok/s) over the last %d\n",
                  1e3 * (times[0] - t0d), 1e3 * (times[n - 1] - times[skip]) / (n - 1 - skip),
                  (n - 1 - skip) / (times[n - 1] - times[skip]), n - 1 - skip);
      {  // per step: with drafts a step emits several tokens at once
        const std::vector<double>& t = a.spec ? step_times : times;
        const int ns = (int)t.size(), sk = std::min(8, ns / 4);
        std::vector<double> d;
        for (int i = sk + 1; i < ns; ++i) d.push_back(1e3 * (t[i] - t[i - 1]));
        std::sort(d.begin(), d.end());
        if (!d.empty())
          std::printf("  per %s (ms): median %.3f min %.3f p90 %.3f max %.3f\n", a.spec ? "step" : "token",
                      d[d.size() / 2], d[0], d[d.size() * 9 / 10], d.back());
      }
      int same = 0;
      while (same < n && ctx + 1 + same < L && out[same] == toks[ctx + 1 + same]) ++same;
      std::printf("  the first %d tokens equal the golden continuation\n", same);
      {
        std::vector<double> late, lau;
        for (size_t i = skip; i + 1 < g_t_launch.size(); ++i) {
          late.push_back(1e3 * (g_t_late[i] - g_t_wait[i]));
          lau.push_back(1e3 * (g_t_launch[i] - g_t_late[i]));
        }
        std::sort(late.begin(), late.end());
        std::sort(lau.begin(), lau.end());
        if (!late.empty())
          std::printf("  host: sample -> lookups submitted %.3f ms (p90 %.3f), -> next launched %.3f ms (p90 %.3f, max %.3f)\n",
                      late[late.size() / 2], late[late.size() * 9 / 10], lau[lau.size() / 2], lau[lau.size() * 9 / 10],
                      lau.back());
      }
      std::printf("generated:");
      for (int t : out) std::printf(" %d", t);
      std::printf("\n");
      // the history now holds what the loop fed (tok_0 .. tok_{fed-1}): the next context goes on from it
      for (int i = 0; i < e.fed; ++i) toks[ctx + i] = i ? out[i - 1] : first;
      p = ctx + e.fed;
    }
    return 0;
  }

  if (a.mode == "batch") {
    struct Sess {
      int ctx = 0, slot = 0;
      bool ced = false;
      int32_t first = 0;
      uint64_t hash = 1469598103934665603ull;  // as decode mode's chosen_hash over its decode
      std::vector<int32_t> out;
      std::vector<uint32_t> bits;  // exact: each sample's logit bits (DSGB10_TF_DUMP)
    };
    std::vector<Sess> ss;
    for (const char* q = a.batch.c_str(); *q;) {
      Sess s;
      s.ctx = std::atoi(q);
      while (*q && *q != ',') s.ced |= *q++ == 'c';
      if (*q) ++q;
      ss.push_back(s);
    }
    if (!a.big || ss.empty() || (int)ss.size() > a.slots)
      die("batch: %zu sessions (--batch) in %d slots (--slots), through the prefill path (--big)", ss.size(), a.slots);
    const int per = a.batch_tf ? a.steps * a.batch_rows : a.steps;  // positions each session goes on by
    if (a.batch_tf && (int)ss.size() * a.batch_rows > model::kMaxRows)
      die("batch: %zu sessions of %d rows a step: at most %d rows", ss.size(), a.batch_rows, model::kMaxRows);
    if (a.spec && (int)ss.size() > m.spec_batch_rows() / 2)  // a draft each at least
      die("batch: %zu sessions with drafts (--spec): at most %d", ss.size(), m.spec_batch_rows() / 2);
    for (size_t k = 0; k < ss.size(); ++k) {  // each session's prompt in its slot
      Sess& s = ss[k];
      s.slot = a.slots - 1 - (int)k;
      if (s.ctx < 128 || s.ctx + per >= L) die("batch: context %d and %d positions, %d golden tokens", s.ctx, per, L);
      Cmd c{};
      c.op = kCmdState;
      c.slot = s.slot;
      bcast(ring, c);
      m.use_slot(s.slot);
      ced_now = s.ced;
      t_pref = 0, n_pref = 0;
      s.first = big_prefill(0, s.ctx, toks.data());
      std::printf("batch: session %zu in slot %d: context %d%s, prefill %.2f s, first token %d\n", k, s.slot, s.ctx,
                  s.ced ? " (CED)" : "", t_pref, s.first);
    }
    if (a.batch_tf) {
      // The teacher-forced batch check (docs/batching.md): every session's golden continuation teacher-forced, --batch-rows rows of
      // each per step (kCmdBatchTf). Each row computes what it computes alone, so a session's samples hash is
      // the same whatever the other sessions and the rows per step, and tf mode's over the same positions
      // (--big --prompt C --k G --limit C + G * steps: the same (token, logit) folds in the same order).
      std::vector<BatchEntry> e(ss.size());
      std::vector<std::vector<int32_t>> t(ss.size());
      for (size_t k = 0; k < ss.size(); ++k) {
        BatchEntry& x = e[k];
        x.slot = ss[k].slot, x.pos0 = ss[k].ctx, x.max_new = per, x.limit = a.max_ctx;
        x.sp.temperature = a.temp, x.sp.top_p = a.top_p, x.sp.top_k = model::gsample_top_k(a.top_k, a.temp, a.top_p, true), x.sp.seed = a.seed;
        t[k].assign(toks.begin() + ss[k].ctx, toks.begin() + ss[k].ctx + per);
      }
      Cmd c{};
      c.op = kCmdBatchTf;
      c.n = (int)e.size();
      c.rows = a.batch_rows;
      bcast(ring, c);
      bcast_tf(ring, c, e, t, 0);
      const std::vector<float> ms = batch_tf_loop(m, c, e, t, wd, [&](int k, const model::SampleOut& o, int row) {
        Sess& s = ss[k];
        s.out.push_back(o.row[row].token);  // the sample at its position: the golden token after it when equal
        uint32_t bits;
        std::memcpy(&bits, &o.row[row].value, 4);
        s.bits.push_back(bits);
        s.hash = (s.hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
        s.hash = (s.hash ^ bits) * 1099511628211ull;
      });
      bcast(ring, Cmd{});  // stop
      if (const char* dp = getenv("DSGB10_TF_DUMP")) {  // every session's samples, one line each
        FILE* f = std::fopen(dp, "w");
        for (size_t k = 0; f && k < ss.size(); ++k) {
          std::fprintf(f, "%zu %d", k, ss[k].ctx);
          for (size_t i = 0; i < ss[k].out.size(); ++i) std::fprintf(f, " %d:%08x", ss[k].out[i], ss[k].bits[i]);
          std::fprintf(f, "\n");
        }
        if (f) std::fclose(f);
      }
      for (size_t k = 0; k < ss.size(); ++k) {
        const Sess& s = ss[k];
        int same = 0;
        for (size_t i = 0; i < s.out.size(); ++i) same += s.out[i] == toks[s.ctx + 1 + i];
        std::printf("batch-tf: session %zu (slot %d, context %d%s): %zu rows in steps of %d, samples hash %016llx; %d of "
                    "them equal the golden next token\n",
                    k, s.slot, s.ctx, s.ced ? ", CED" : "", s.out.size(), a.batch_rows, (unsigned long long)s.hash, same);
      }
      const int skip = std::min(3, (int)ms.size() / 2);  // the first steps run eagerly or capture
      std::vector<double> d(ms.begin() + skip, ms.end());
      std::sort(d.begin(), d.end());
      if (!d.empty())
        std::printf("batch-tf: %zu steps of %zu sessions, %d rows each: gpu (ms) median %.3f min %.3f p90 %.3f max %.3f\n",
                    ms.size(), ss.size(), (int)ss.size() * a.batch_rows, d[d.size() / 2], d[0], d[d.size() * 9 / 10],
                    d.back());
      return 0;
    }
    std::vector<BatchEntry> e(ss.size());
    for (size_t k = 0; k < ss.size(); ++k) {
      BatchEntry& x = e[k];
      x.slot = ss[k].slot, x.pos0 = ss[k].ctx, x.token = ss[k].first;
      x.max_new = a.steps, x.limit = a.max_ctx;
      x.sp.temperature = a.temp, x.sp.top_p = a.top_p, x.sp.top_k = model::gsample_top_k(a.top_k, a.temp, a.top_p, true), x.sp.seed = a.seed;
    }
    Cmd c{};
    c.op = a.spec ? kCmdSpecBatch : kCmdBatch;  // with --spec: drafts across the sessions, every one to its end
    c.n = (int)e.size();
    if (a.spec) {
      c.spec = 1;
      c.spec_drafts = a.spec_drafts;
      c.spec_thresh = a.spec_thresh;
      std::copy(a.spec_cost, a.spec_cost + 3, c.spec_cost);
      c.spec_gate = a.spec_gate;
      std::copy(a.spec_row_ms, a.spec_row_ms + model::ops::kDrafts + 1, c.spec_row_ms);
    }
    bcast(ring, c);
    bcast_payload(ring, e.data(), e.size() * sizeof(BatchEntry), 0);
    const OnBatch cb = [&](int k, const model::SampleOut& o, int row) {
      Sess& s = ss[k];
      s.out.push_back(o.row[row].token);
      uint32_t bits;
      std::memcpy(&bits, &o.row[row].value, 4);
      s.hash = (s.hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
      s.hash = (s.hash ^ bits) * 1099511628211ull;
    };
    std::vector<BatchEnd> ends(ss.size());
    std::vector<DecodeEnd> dends;
    SpecBatchStats st;
    if (a.spec) {
      dends = spec_batch_loop(m, c, e, wd, cb, &st);
      for (size_t k = 0; k < ss.size(); ++k)
        ends[k].sampled = dends[k].sampled, ends[k].fed = dends[k].fed, ends[k].why = dends[k].why;
    } else {
      ends = batch_loop(m, c, e, wd, cb);
    }
    bcast(ring, Cmd{});  // stop
    for (size_t k = 0; k < ss.size(); ++k) {
      const Sess& s = ss[k];
      int same = 0;
      while (same < (int)s.out.size() && s.ctx + 1 + same < L && s.out[same] == toks[s.ctx + 1 + same]) ++same;
      std::printf("batch: session %zu (slot %d, context %d%s): %zu tokens, %d positions fed, end %u; tokens hash "
                  "%016llx; the first %d tokens equal the golden continuation\n",
                  k, s.slot, s.ctx, s.ced ? ", CED" : "", s.out.size(), ends[k].fed, ends[k].why,
                  (unsigned long long)s.hash, same);
      if (a.spec) spec_report(dends[k]);
    }
    // the steps' times, by when each one's samples came (the first ones run eagerly and capture graphs)
    const std::vector<double>& t = g_t_wait;
    const int ns = (int)t.size(), skip = std::min(8, ns / 4);
    if (ns - 1 - skip > 0) {
      const double ms = 1e3 * (t[ns - 1] - t[skip]) / (ns - 1 - skip);
      std::vector<double> d;
      for (int i = skip + 1; i < ns; ++i) d.push_back(1e3 * (t[i] - t[i - 1]));
      std::sort(d.begin(), d.end());
      if (!a.spec) {
        std::printf("batch: %d steps of %zu rows: %.3f ms per step (median %.3f, p90 %.3f, max %.3f), %.1f tok/s in "
                    "all, %.1f per session\n",
                    ns, ss.size(), ms, d[d.size() / 2], d[d.size() * 9 / 10], d.back(), 1e3 * ss.size() / ms, 1e3 / ms);
      } else {
        const double tps = st.steps ? (double)st.emitted / st.steps : 0.0;  // tokens per step, every session's
        std::printf("batch: %d steps of %.2f rows (%zu sessions with drafts): %.3f ms per step (median %.3f, p90 %.3f, "
                    "max %.3f), %.2f tokens per step, %.1f tok/s in all, %.1f per session; spec step GPU ms by rows:",
                    ns, st.steps ? (double)st.rows / st.steps : 0.0, ss.size(), ms, d[d.size() / 2],
                    d[d.size() * 9 / 10], d.back(), tps, 1e3 * tps / ms, 1e3 * tps / ms / (double)ss.size());
        for (int r = 1; r <= model::kMaxRows; ++r)
          if (st.gpu_n[r]) std::printf(" %d: %.2f (%d)", r, st.gpu_ms[r] / st.gpu_n[r], st.gpu_n[r]);
        std::printf("\n");
      }
    }
    if (a.spec && a.bgraph_check)  // the passes' drafts against each entry's one-session draft
      std::printf("batch: draft check: %u words of the entries' drafts and confidence differ from their one-session "
                  "drafts (0: the pass over the entries is bit for bit each entry's own)\n",
                  m.spec_batch_mismatches());
    return 0;
  }

  if (a.mode == "spectf") {
    // engine/dspark's reference run: step i (i >= the prompt's last position) adds position i to the
    // windows and drafts from the golden token at i + 1; d_j is for position i + 1 + j.
    // EXPERIMENT (--seq-list): the list's sequences one after another, each from position 0
    std::vector<SeqLine> seqs;
    if (a.seq_list.empty()) seqs.push_back(SeqLine{a.golden, prompt, L, toks});
    else seqs = read_seq_list(a.seq_list);
    const int32_t* tk = nullptr;  // the sequence's tokens
    auto rows = [&](int pos0, int R, int force, const int32_t* probe = nullptr) {
      Cmd c{};
      c.op = kCmdRows;
      c.pos0 = pos0;
      c.rows = R;
      std::copy(tk + pos0, tk + pos0 + R, c.tokens);
      c.spec_force = force;
      if (probe) std::copy(probe, probe + model::ops::kDrafts, c.spec_probe);
      if (probe) c.sp.n_top = model::ops::kTopLogprobs;  // the target's own top logprobs too (the draft gate study)
      bcast(ring, c);
      wd.started = now_s();
      const model::SampleOut& o = m.wait(run_rows(m, c));
      wd.started = 0;
      return o;
    };
    // EXPERIMENT (--tree-probe, --tree-sib): the probe's records, and the routed experts of eager steps
    std::FILE* tp = nullptr;
    if (!a.tree_probe.empty() && !(tp = std::fopen(a.tree_probe.c_str(), "w")))
      die("spectf: cannot write %s", a.tree_probe.c_str());
    auto ints = [](const int32_t* v, int k) {
      std::string s = "[";
      for (int j = 0; j < k; ++j) s += (j ? "," : "") + std::to_string(v[j]);
      return s + "]";
    };
    auto flts = [](const float* v, int k) {
      std::string s = "[";
      char b[32];
      for (int j = 0; j < k; ++j) std::snprintf(b, sizeof b, "%s%.6g", j ? "," : "", v[j]), s += b;
      return s + "]";
    };
    auto route = [&](int pos0, int R, const int32_t* t, ExpertSink& es) {  // an eager step's routed experts
      Cmd c{};
      c.op = kCmdStep;
      c.rows = R;
      c.pos0 = pos0;
      c.flags = model::kFlagDump;
      std::copy(t, t + R, c.tokens);
      bcast(ring, c);
      wd.started = now_s();
      m.step(to_ctl(c), sampled, &es);
      wd.started = 0;
    };
    auto experts = [&](int pos0, int R, const int32_t* t) {  // the same, per layer, as JSON
      ExpertSink es;
      route(pos0, R, t, es);
      std::string s = "[";
      for (int l = 0; l < model::kLayers; ++l) s += (l ? "," : "") + ints(es.e[l].data(), (int)es.e[l].size());
      return s + "]";
    };
    // EXPERIMENT (--route-file; docs/engine.md "Prompt-lookup drafts"): the routed experts of eager steps on
    // the way, a record each: int32 {sequence, kind, anchor, pos0, R}, the R rows' tokens, then int16 experts
    // [kLayers][R][kTopK] (-1: a layer without). Kinds: 0 the sequence's own tokens from the anchor on
    // (--route-every R: R rows at every R-th anchor), 1 the anchor's token and the step's drafts (--chain-every
    // N: at every N-th anchor), 2 the anchor's token and a --props line's tokens. Rows past 8 go in further
    // steps behind the first ones (a row computes the same in a step of any size).
    std::FILE* rf = nullptr;
    if (!a.route_file.empty() && !(rf = std::fopen(a.route_file.c_str(), "wb")))
      die("spectf: cannot write %s", a.route_file.c_str());
    const Props props = a.props.empty() ? Props{} : read_props(a.props);
    double t_route = 0;
    int n_route = 0;
    auto route_rec = [&](int sq, int kind, int anchor, int R, const int32_t* t) {
      // rows at [anchor, anchor + R) read the positions before the anchor; what they write past the history
      // must leave the carries' slot of anchor - 1 alone, for the next spectf step (16 slots: R <= 15)
      if (R < 1 || R > 15) die("spectf: %d rows to route at %d (1..15)", R, anchor);
      for (int q = 0; q < R; q += model::kMaxSessRows) {  // steps of one session: at most kMaxSessRows rows
        const int nr = std::min(model::kMaxSessRows, R - q);
        ExpertSink es;
        const double t0 = now_s();
        route(anchor + q, nr, t + q, es);
        t_route += now_s() - t0, ++n_route;
        const int32_t h[5] = {sq, kind, anchor, anchor + q, nr};
        std::fwrite(h, 4, 5, rf);
        std::fwrite(t + q, 4, nr, rf);
        std::vector<int16_t> e((size_t)model::kLayers * nr * model::kTopK, -1);
        for (int l = 0; l < model::kLayers; ++l)
          if ((int)es.e[l].size() == nr * model::kTopK)
            for (int j = 0; j < nr * model::kTopK; ++j) e[(size_t)l * nr * model::kTopK + j] = (int16_t)es.e[l][j];
        std::fwrite(e.data(), 2, e.size(), rf);
      }
    };
    int sib_at = -1;  // the step whose probe's 4 most likely tokens for position sib_at + 2 are routed next
    int32_t sib[4] = {};
    int hist[model::ops::kDrafts + 1] = {}, n = 0, first = -1;
    double conf[model::ops::kDrafts] = {}, rate[model::ops::kDrafts] = {};
    int runs[model::ops::kDrafts] = {};
    for (size_t sq = 0; sq < seqs.size(); ++sq) {
      tk = seqs[sq].toks.data();
      const int SL = std::min(seqs[sq].n > 0 ? seqs[sq].n : INT32_MAX, (int)seqs[sq].toks.size());
      first = seqs[sq].prompt - 1;
      if (first < 1 || first + 2 > SL) die("spectf: --prompt %d does not fit the %d golden tokens", first + 1, SL);
      char pre[32] = "";
      if (!a.seq_list.empty()) std::snprintf(pre, sizeof pre, "seq %zu ", sq);
      if (a.big)  // the prompt through the prefill path: its seeding of the windows
        big_prefill(0, first, tk);
      else  // spec steps of one session: at most kMaxSessRows rows each
        for (int p = 0; p < first; p += model::kMaxSessRows) rows(p, std::min(model::kMaxSessRows, first - p), -1);
      sib_at = -1;
      for (int i = first; i + 1 < SL; ++i, ++n) {
        int32_t path[model::ops::kDrafts];
        for (int j = 0; j < model::ops::kDrafts; ++j) path[j] = i + 2 + j < SL ? tk[i + 2 + j] : -1;
        const model::SampleOut& o = rows(i, 1, tk[i + 1], tp ? path : nullptr);
        int32_t drafts[model::ops::kDrafts];  // o's slot is reused 4 steps on
        std::copy(o.drafts, o.drafts + model::ops::kDrafts, drafts);
        if (tp) {
          dspark::ProbeOut p;
          static_assert(sizeof p == sizeof o.probe, "ProbeOut");
          std::memcpy(&p, o.probe, sizeof p);
          std::string id = "[", pr = "[";
          for (int j = 0; j < dspark::draft_rows(); ++j) {
            id += (j ? "," : "") + ints(p.id[j], dspark::kProbeK);
            pr += (j ? "," : "") + flts(p.p[j], dspark::kProbeK);
          }
          // with --seq-list, the sequence; tid / tlp: the target's own most likely ids for position i + 1 and their
          // logprobs (T = 1, before top-k / top-p), which the draft gate study turns into P(a sample takes d_j)
          const std::string sqf = a.seq_list.empty() ? "" : "\"seq\":" + std::to_string(sq) + ",";
          std::fprintf(tp, "{%s\"i\":%d,\"bonus\":%d,\"sample\":%d,\"d\":%s,\"s\":%s,\"path\":%s,\"id\":%s],"
                       "\"p\":%s],\"pp\":%s,\"rank\":%s,\"conf\":%s,\"lse\":%s,\"tid\":%s,\"tlp\":%s}\n",
                       sqf.c_str(), i, tk[i + 1], o.row[0].token, ints(o.drafts, 5).c_str(), flts(o.survival, 5).c_str(),
                       ints(path, 5).c_str(), id.c_str(), pr.c_str(), flts(p.p_path, 5).c_str(), ints(p.rank, 5).c_str(),
                       flts(p.conf, 5).c_str(), flts(p.lse, 5).c_str(), ints(o.top_id[0], o.n_top).c_str(),
                       flts(o.top_lp[0], o.n_top).c_str());
        }
        int acc = 0;
        while (acc < dspark::draft_rows() && i + 2 + acc < SL && o.drafts[acc] == tk[i + 2 + acc]) ++acc;
        std::printf("%sspectf %d: drafts %d %d %d %d %d, survival %.4f %.4f %.4f %.4f %.4f, %d equal\n", pre, i,
                    o.drafts[0], o.drafts[1], o.drafts[2], o.drafts[3], o.drafts[4], o.survival[0], o.survival[1],
                    o.survival[2], o.survival[3], o.survival[4], acc);
        ++hist[acc];
        for (int j = 0; j < dspark::draft_rows() && j <= acc && i + 2 + j < SL; ++j) {  // P(d_j+1 | d_1..d_j accepted)
          const double prev = j ? o.survival[j - 1] : 1.0;
          conf[j] += prev > 0 ? o.survival[j] / prev : 0.0;
          rate[j] += j < acc;
          ++runs[j];
        }
        if (tp && sib_at == i - 1 && i + 4 < SL) {  // position i + 1 has its KV of the golden token: route there
          std::string s = "[";
          for (int k = 0; k < 4; ++k) s += (k ? "," : "") + experts(i + 1, 1, &sib[k]);
          const std::string ch = experts(i + 1, 4, &tk[i + 1]);
          std::fprintf(tp, "{\"sib_at\":%d,\"pos\":%d,\"tokens\":%s,\"golden\":%s,\"sib\":%s],\"chain\":%s}\n", sib_at,
                       i + 1, ints(sib, 4).c_str(), ints(&tk[i + 1], 4).c_str(), s.c_str(), ch.c_str());
        }
        sib_at = -1;
        if (tp && a.tree_sib > 0 && (i - first) % a.tree_sib == 0) {
          sib_at = i;
          dspark::ProbeOut p;
          std::memcpy(&p, o.probe, sizeof p);
          std::copy(p.id[0], p.id[0] + 4, sib);
        }
        if (rf) {  // the next step's anchor i + 1: rows from there (the history before it is the sequence's)
          const int A = i + 1;
          if (a.route_every > 0 && (i - first) % a.route_every == 0)
            route_rec((int)sq, 0, A, std::min(a.route_every, SL - A), tk + A);
          if (a.chain_every > 0 && (i - first) % a.chain_every == 0) {
            int32_t r[1 + model::ops::kDrafts] = {tk[A]};
            std::copy(drafts, drafts + dspark::draft_rows(), r + 1);
            route_rec((int)sq, 1, A, 1 + dspark::draft_rows(), r);
          }
          const auto it = props.find({(int)sq, A});
          if (it != props.end())
            for (const std::vector<int32_t>& p : it->second) {
              std::vector<int32_t> r{tk[A]};
              r.insert(r.end(), p.begin(), p.end());
              route_rec((int)sq, 2, A, (int)r.size(), r.data());
            }
        }
      }
      if (!a.seq_list.empty()) std::printf("seq %zu: steps from position %d to %d\n", sq, first, SL - 2);
    }
    bcast(ring, Cmd{});  // stop
    if (tp) std::fclose(tp);
    if (rf) {
      std::fclose(rf);
      std::printf("spectf: %d eager steps routed, %.1f ms each\n", n_route, n_route ? 1e3 * t_route / n_route : 0.0);
    }
    std::printf("spectf: %d steps from position %d; leading drafts equal to the golden tokens 0..5: %d %d %d %d %d %d "
                "(%.3f per step; engine/dspark's reference on prose: 47 31 12 3 2 1, 0.80)\n",
                n, first, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5],
                (double)(hist[1] + 2 * hist[2] + 3 * hist[3] + 4 * hist[4] + 5 * hist[5]) / n);
    for (int j = 0; j < 3; ++j)
      if (runs[j])
        std::printf("  d%d: mean confidence %.3f, accepted %.3f, of %d steps%s\n", j + 1, conf[j] / runs[j],
                    rate[j] / runs[j], runs[j], j == 1 ? " with d1 accepted" : j ? " with d1..d2 accepted" : "");
    return 0;
  }

  if (a.mode == "bench") {
    int need = ctxs.back();
    for (int k : a.ks) {
      const bool multi = k > model::kMaxSessRows;  // runs of several sessions only
      if (!multi) need += k * (a.bench_stride > 0 ? 3 * a.steps + 18 : a.steps);
      for (int G : a.bench_groups)
        if (a.bench_stride > 0 && G > 1 && G < k && k % G == 0 && (!multi || (G <= model::kMaxSessRows && k / G <= a.slots)))
          need += k * (2 * a.steps + 12);
    }
    // the rotated kinds: row r's (or run r's) router logits rotated kRouteShift experts further than the one
    // before (model.h set_route_shift), kExperts / kMaxSessRows, whatever kMaxRows
    constexpr int kRouteShift = model::kExperts / model::kMaxSessRows;
    if (ctxs[0] < 1 || a.steps < 4 || need > L)
      die("bench: context %d plus %d steps per k do not fit the %d golden tokens", ctxs.back(), a.steps, L);
    auto stat = [](std::vector<double> v) {  // the first 3 steps run eagerly or capture
      v.erase(v.begin(), v.begin() + 3);
      std::sort(v.begin(), v.end());
      double s = 0;
      for (double x : v) s += x;
      char b[128];
      std::snprintf(b, sizeof b, "median %.3f mean %.3f min %.3f p90 %.3f", v[v.size() / 2], s / v.size(), v[0],
                    v[v.size() * 9 / 10]);
      return std::string(b);
    };
    for (size_t c = 0, p = 0; c < ctxs.size(); ++c) {
      const int ctx = ctxs[c];
      if (ctx < (int)p) die("bench: context %d after %zu positions (--ctxs ascend by the steps at least)", ctx, p);
      n_pref = 0;
      if (ctx > (int)p) reach((int)p, ctx);
      std::vector<std::vector<double>> host(a.ks.size()), gpu(a.ks.size()), bc(a.ks.size()), ugpu(a.ks.size()),
          xgpu(a.ks.size());
      RouteSink route_all;  // every consecutive row routed in this context's eager steps
      p = ctx;
      for (size_t j = 0; j < a.ks.size(); ++j) {
        if (a.ks[j] > model::kMaxSessRows) {
          // A step of 9..16 rows holds runs of several sessions only (one session's at most kMaxSessRows: the
          // one-session kinds below are skipped), through Model::launch_batch (kCmdBatchTf) with a session slot per
          // run: session g in slot g, G consecutive positions per step from base - (n - 1 - g) * S (S = --bench-stride:
          // the sessions as far apart as a one-session step's runs), its tokens from its own scattered golden
          // position (run_tok, as the runs below); natural, then each session's routing rotated kRouteShift experts
          // further than the one before (rank 0's, as the kinds below); then a few eager steps of each kind for
          // their distinct experts. The slots past the context's hold no state (their rows attend to empty windows
          // and caches: the same work), and every slot's Engram history is the golden tokens (set_tokens), so that
          // each session's rows hash their n-grams from it.
          const int k = a.ks[j];
          std::printf("  k %d: past one session's %d rows; runs of several sessions only\n", k, model::kMaxSessRows);
          const int base = (int)p;
          for (int G : a.bench_groups) {
            if (G <= 1 || G > model::kMaxSessRows || G >= k || k % G || k / G > a.slots) continue;
            const int n = k / G;
            if (base < (n - 1) * a.bench_stride)
              die("bench: k %d in %d runs %d apart needs a context of at least %d", k, n, a.bench_stride, (n - 1) * a.bench_stride);
            for (int s = 0; s < n; ++s) {  // the sessions' Engram histories: the golden tokens, in every slot
              Cmd c{};
              c.op = kCmdTokens;
              c.slot = s;
              c.pos0 = 0;
              c.n = L;
              bcast(ring, c);
              bcast_payload(ring, toks.data(), toks.size() * 4, 0);
              m.use_slot(s);
              m.set_tokens(0, L, toks.data());
            }
            {
              Cmd c{};
              c.op = kCmdState;
              c.slot = a.slot;
              bcast(ring, c);
              m.use_slot(a.slot);
            }
            auto run_tok = [&](size_t q, int r) {
              const uint64_t g = (uint64_t)(r / G);
              return toks[(size_t)(((q + g) * 2654435761ull + g * 40503ull) % (uint64_t)(L - G)) + r % G];
            };
            // one step: the sessions' step i (their positions go on by G each), tokens drawn at q; eager with its
            // routed experts into rs when given; its GPU ms
            auto tf_step = [&](size_t q, int i, RouteSink* rs) {
              Cmd c{};
              c.op = kCmdBatchTf;
              c.n = n;
              c.rows = G;
              c.flags = rs ? model::kFlagDump : 0;
              std::vector<BatchEntry> e(n);
              std::vector<std::vector<int32_t>> t(n);
              for (int g = 0; g < n; ++g) {
                e[g].slot = g;
                e[g].pos0 = base - (n - 1 - g) * a.bench_stride + i * G;
                e[g].max_new = G;
                e[g].limit = a.max_ctx;
                e[g].sp.temperature = a.temp, e[g].sp.seed = a.seed;
                for (int r = 0; r < G; ++r) t[g].push_back(run_tok(q, g * G + r));
              }
              bcast(ring, c);
              bcast_tf(ring, c, e, t, 0);
              const std::vector<float> ms =
                  batch_tf_loop(m, c, e, t, wd, [&](int, const model::SampleOut& o, int row) {
                    uint32_t bits;
                    std::memcpy(&bits, &o.row[row].value, 4);
                    chosen_hash = (chosen_hash ^ (uint32_t)o.row[row].token) * 1099511628211ull;
                    chosen_hash = (chosen_hash ^ bits) * 1099511628211ull;
                  }, rs);
              return ms.empty() ? 0.0 : (double)ms[0];
            };
            std::vector<double> gg[2];
            RouteSink gs[2];
            int i = 0;
            for (int v = 0; v < 2; ++v) {
              m.set_route_shift(v ? kRouteShift : 0, G);
              for (int s = 0; s < a.steps; ++s, ++i, p += k) gg[v].push_back(tf_step(p, i, nullptr));
              for (int s = 0; s < 6; ++s, ++i, p += k) tf_step(p, i, a.rank == 0 ? &gs[v] : nullptr);
            }
            m.set_route_shift(0);
            std::printf("  k %d in %d runs of %d rows %d apart (a session a slot), gpu (ms): natural %s, %.1f experts per "
                        "layer; rotated %s, %.1f\n",
                        k, n, G, a.bench_stride, stat(gg[0]).c_str(), gs[0].distinct / std::max(1, gs[0].n),
                        stat(gg[1]).c_str(), gs[1].distinct / std::max(1, gs[1].n));
          }
          continue;
        }
        for (int i = 0; i < a.steps; ++i, p += a.ks[j]) {
          // nsys --capture-range=cudaProfilerApi: the first context's replays from the first k's on (tools:
          // kprof drops each k's first 3 steps, which run eagerly or capture)
          const bool start = c == 0 && j == 0 && i == 3;
          if (start) CUDA_CHECK(cudaProfilerStart());
          host[j].push_back(1e3 * run((int)p, a.ks[j], &toks[p], start ? model::kFlagProfile : 0));
          gpu[j].push_back(m.last_gpu_ms());
          bc[j].push_back(1e3 * t_bcast);
        }
        if (a.bench_stride <= 0) continue;
        // the same row count, unrelated rows: golden tokens from scattered positions, the rows
        // bench_stride positions apart (every rank draws the same); then the same with each row's router
        // logits rotated kRouteShift experts further than the last row's, so the rows share few experts
        auto scattered = [&](size_t q, int r) {
          return toks[(size_t)(((uint64_t)(q + r) * 2654435761ull + (uint64_t)r * 40503ull) % (uint64_t)L)];
        };
        m.set_pos_stride(a.bench_stride);
        for (int v = 1; v <= 2; ++v) {
          m.set_route_shift(v == 2 ? kRouteShift : 0);
          for (int i = 0; i < a.steps; ++i, p += a.ks[j]) {
            int32_t u[model::kMaxRows];
            for (int r = 0; r < a.ks[j]; ++r) u[r] = scattered(p, r);
            run((int)p, a.ks[j], u, 0);
            (v == 1 ? ugpu : xgpu)[j].push_back(m.last_gpu_ms());
          }
        }
        // then a few eager steps of each kind for their routing: distinct experts per layer
        RouteSink rs[3];
        for (int v = 0; v < 3; ++v) {
          m.set_pos_stride(v ? a.bench_stride : 1);
          m.set_route_shift(v == 2 ? kRouteShift : 0);
          for (int i = 0; i < 6; ++i, p += a.ks[j]) {
            Cmd c{};
            c.op = kCmdStep;
            c.rows = a.ks[j];
            c.pos0 = (int)p;
            c.flags = model::kFlagDump;
            for (int r = 0; r < a.ks[j]; ++r) c.tokens[r] = v ? scattered(p, r) : toks[p + r];
            bcast(ring, c);
            m.step(to_ctl(c), sampled, a.rank == 0 ? &rs[v] : nullptr);
          }
        }
        m.set_pos_stride(1);
        m.set_route_shift(0);
        for (int L = 0; L < model::kLayers; ++L) {
          for (int e = 0; e < model::kExperts; ++e) route_all.freq[L][e] += rs[0].freq[L][e];
          route_all.rows[L] += rs[0].rows[L];
        }
        if (a.rank == 0)
          std::printf("  k %d routed experts per layer (distinct, of %d): consecutive %.1f, unrelated %.1f, rotated %.1f\n",
                      a.ks[j], a.ks[j] * model::kTopK, rs[0].distinct / std::max(1, rs[0].n),
                      rs[1].distinct / std::max(1, rs[1].n), rs[2].distinct / std::max(1, rs[2].n));
        // then runs of G consecutive rows: k / G sessions' verify rows, each run from its own scattered golden
        // position, the runs bench_stride apart; natural, then each run's routing rotated kRouteShift experts
        // further than the run before (sessions that share few experts); a few eager steps of each for their
        // distinct experts
        for (int G : a.bench_groups) {
          if (G <= 1 || G >= a.ks[j] || a.ks[j] % G) continue;
          auto run_tok = [&](size_t q, int r) {
            const uint64_t g = (uint64_t)(r / G);
            return toks[(size_t)(((q + g) * 2654435761ull + g * 40503ull) % (uint64_t)(L - G)) + r % G];
          };
          std::vector<double> gg[2];
          RouteSink gs[2];
          m.set_pos_stride(a.bench_stride, G);
          for (int v = 0; v < 2; ++v) {
            m.set_route_shift(v ? kRouteShift : 0, G);
            int32_t u[model::kMaxRows];
            for (int i = 0; i < a.steps; ++i, p += a.ks[j]) {
              for (int r = 0; r < a.ks[j]; ++r) u[r] = run_tok(p, r);
              run((int)p, a.ks[j], u, 0);
              gg[v].push_back(m.last_gpu_ms());
            }
            for (int i = 0; i < 6; ++i, p += a.ks[j]) {
              Cmd c{};
              c.op = kCmdStep;
              c.rows = a.ks[j];
              c.pos0 = (int)p;
              c.flags = model::kFlagDump;
              for (int r = 0; r < a.ks[j]; ++r) c.tokens[r] = run_tok(p, r);
              bcast(ring, c);
              m.step(to_ctl(c), sampled, &gs[v]);
            }
          }
          m.set_pos_stride(1);
          m.set_route_shift(0);
          std::printf("  k %d in %d runs of %d rows %d apart, gpu (ms): natural %s, %.1f experts per layer; rotated %s, "
                      "%.1f\n",
                      a.ks[j], a.ks[j] / G, G, a.bench_stride, stat(gg[0]).c_str(), gs[0].distinct / std::max(1, gs[0].n),
                      stat(gg[1]).c_str(), gs[1].distinct / std::max(1, gs[1].n));
        }
      }
      if (a.bench_stride > 0 && a.rank == 0) {
        // rows routed independently, each as often as the consecutive rows above routed to each expert
        std::printf("  independent rows by the experts' frequencies over %d rows, distinct per layer:", route_all.rows[0]);
        for (int B = 1; B <= 6; ++B) std::printf(" %d: %.1f", B, route_all.independent(B));
        std::printf("\n");
      }
      if (c == 0) CUDA_CHECK(cudaProfilerStop());
      if (c + 1 == ctxs.size() && !a.graph_sweep) bcast(ring, Cmd{});  // stop
      std::printf("bench: layers [%d, %d), %s, context %d, %d steps per k; chosen tokens hash %016llx\n", a.first,
                  a.layers, a.graphs ? "graphs" : "eager", ctx, a.steps, (unsigned long long)chosen_hash);
      if (n_pref && a.big)
        std::printf("  prefill: %d chunks of up to %d rows, %.2f s\n", n_pref, a.prefill_rows, t_pref);
      else if (n_pref)
        std::printf("  prefill: %d chunks of %d rows, %.2f ms each\n", n_pref, a.chunk, 1e3 * t_pref / n_pref);
      for (size_t j = 0; j < a.ks.size(); ++j) {
        if (host[j].empty()) continue;  // a k past kMaxSessRows: its runs of several sessions are printed above
        std::printf("  k %d step, host (ms): %s\n", a.ks[j], stat(host[j]).c_str());
        std::printf("  k %d step, gpu  (ms): %s\n", a.ks[j], stat(gpu[j]).c_str());
        std::printf("  k %d bcast      (ms): %s\n", a.ks[j], stat(bc[j]).c_str());
        if (!ugpu[j].empty())
          std::printf("  k %d unrelated rows %d apart, gpu (ms): %s\n", a.ks[j], a.bench_stride, stat(ugpu[j]).c_str());
        if (!xgpu[j].empty())
          std::printf("  k %d the same, routing rotated, gpu (ms): %s\n", a.ks[j], stat(xgpu[j]).c_str());
      }
    }
    if (a.graph_sweep) {
      // a step ending at the last position of every context bucket up to --max-ctx, for each k: each one
      // without a graph yet captures and instantiates it, as a decode loop does on entering a bucket. The
      // rows overwrite positions the benches have passed (their state is not used again).
      size_t free0 = 0, free1 = 0, total = 0;
      const int g0 = m.graph_count();
      CUDA_CHECK(cudaMemGetInfo(&free0, &total));
      for (int k : a.ks) {
        if (k > model::kMaxSessRows) continue;  // one session's steps only
        std::vector<double> t;
        for (int b = model::kCtxBucket; b <= a.max_ctx; b += model::kCtxBucket) {
          const int n0 = m.graph_count();
          const double dt = run(b - k, k, &toks[b - k], 0);
          if (m.graph_count() > n0) t.push_back(1e3 * dt);
        }
        std::sort(t.begin(), t.end());
        if (!t.empty())
          std::printf("graph sweep: k %d, %zu new graphs; a step that captures one takes %.1f ms median, %.1f max\n", k,
                      t.size(), t[t.size() / 2], t.back());
      }
      CUDA_CHECK(cudaMemGetInfo(&free1, &total));
      bcast(ring, Cmd{});  // stop
      const int ng = m.graph_count() - g0;
      std::printf("graph sweep: %d graphs in all; device memory free %.0f -> %.0f MiB, %.2f MiB per new graph\n",
                  m.graph_count(), free0 / 1048576.0, free1 / 1048576.0,
                  ng ? ((double)free0 - (double)free1) / 1048576.0 / ng : 0.0);
    }
    return 0;
  }

  // EXPERIMENT (--seq-list, generate --loop; docs/engine.md "Prompt-lookup drafts"): the list's sequences one
  // after another, each prompt from position 0 through the prefill path, then up to N tokens through the
  // decode loop (with --spec, today's drafts: their speed on that output), ending at EOS
  if (a.mode == "generate" && !a.seq_list.empty()) {
    const std::vector<SeqLine> seqs = read_seq_list(a.seq_list);
    for (size_t sq = 0; sq < seqs.size(); ++sq) {
      const SeqLine& s = seqs[sq];
      if (s.prompt < 1 || s.prompt > (int)s.toks.size() || s.n < 1 || s.prompt + s.n + 8 > a.max_ctx)
        die("--seq-list: %s: prompt %d of %zu tokens, %d to generate, context %d", s.dir.c_str(), s.prompt,
            s.toks.size(), s.n, a.max_ctx);
      t_pref = 0, n_pref = 0;
      const int32_t tok0 = big_prefill(0, s.prompt, s.toks.data());
      const double tp0 = t_pref;
      std::vector<int32_t> out{tok0}, rest;
      std::vector<double> times;
      chosen_hash = 1469598103934665603ull;
      const double td = now_s();
      const DecodeEnd e = run_decode(s.prompt, tok0, s.n - 1, rest, times);
      const double dt = now_s() - td;
      out.insert(out.end(), rest.begin(), rest.end());
      const int nt = (int)times.size(), skip = std::min(8, nt / 4);
      std::printf("seq %zu (%s): prompt %d in %.2f s; %zu tokens in %.2f s, %.1f tok/s after the first %d, end %u; "
                  "tokens hash %016llx\n",
                  sq, s.dir.c_str(), s.prompt, tp0, out.size(), dt,
                  nt - 1 - skip > 0 ? (nt - 1 - skip) / (times[nt - 1] - times[skip]) : 0.0, skip, e.why,
                  (unsigned long long)chosen_hash);
      spec_report(e);
      std::printf("seq %zu generated:", sq);
      for (int t : out) std::printf(" %d", t);
      std::printf("\n");
      std::fflush(stdout);
    }
    bcast(ring, Cmd{});  // stop
    return 0;
  }

  // prompt / prefill in chunks
  const uint32_t pflags = a.mode == "generate" ? model::kFlagLogits : check;
  if (a.big && a.mode != "prefill") {
    sampled[(prompt - 1) % a.chunk] = big_prefill(0, prompt, toks.data());
    std::printf("prompt: %d positions through the prefill path in %.2f s\n", prompt, t_pref);
  }
  for (int p = 0; p < prompt && !(a.big && a.mode != "prefill"); p += a.chunk) {
    const int R = std::min(a.chunk, prompt - p);
    uint32_t f = pflags;  // dump and inject only where the golden run saved the positions
    bool any = false, all = true;
    for (int r = 0; r < R; ++r) any |= g.saved(p + r), all &= g.saved(p + r);
    if (!any) f &= ~model::kFlagDump;
    if (!all) f &= ~(model::kFlagInject | model::kFlagInjectSub);
    t_pref += run(p, R, &toks[p], f);
    ++n_pref;
    if (!a.quiet && (n_pref % 256 == 0 || p + R == prompt))
      std::printf("prefill %d/%d positions, %.1f ms per chunk\n", p + R, prompt, 1e3 * t_pref / n_pref);
  }
  int first_miss = -1;
  if (a.mode == "tf") {
    for (int p = prompt; p < L; p += a.k) {
      const int R = std::min(a.k, L - p);
      t_dec += run(p, R, &toks[p], check);
      ++n_dec;
    }
    if (a.big && n_dec == 0) {
      // Nothing teacher-forced (--limit = --prompt, as long28k's CED check): the hash covers the prompt's
      // sample from the prefill path instead: its token, logit and logprob, then its top logprobs' ids and
      // values, so the hash depends on what the engine computed. A run with steps hashes them alone.
      const model::SampleOut& o = prompt_out;
      const int nt = std::min(o.n_top, model::ops::kTopLogprobs);
      const auto fold = [&](uint32_t x) { chosen_hash = (chosen_hash ^ x) * 1099511628211ull; };
      const auto bits = [](float v) {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return b;
      };
      fold((uint32_t)o.row[0].token);
      fold(bits(o.row[0].value));
      fold(bits(o.row[0].logprob));
      for (int j = 0; j < nt; ++j) fold((uint32_t)o.top_id[0][j]), fold(bits(o.top_lp[0][j]));
      // equal hashes: the same sample after the prompt, bit for bit
      std::printf("teacher-forced: 0 positions; the prompt's sample (token, logit, logprob, top %d logprobs) hash "
                  "%016llx\n", nt, (unsigned long long)chosen_hash);
    } else {
      // equal hashes: the same state after the prompt, bit for bit (e.g. prefill chunks of another size)
      std::printf("teacher-forced: %d positions, chosen tokens and logits hash %016llx\n", L - prompt,
                  (unsigned long long)chosen_hash);
    }
  } else if (a.mode == "generate" && a.loop) {
    // the same generation through the decode loop: tok_0 is the prompt's sample, the loop samples the rest
    std::vector<int32_t> out{sampled[(prompt - 1) % a.chunk]}, rest;
    std::vector<double> times;
    const double td = now_s();
    spec_report(run_decode(prompt, out[0], a.gen - 1, rest, times));
    t_dec = now_s() - td;
    n_dec = (int)rest.size();
    out.insert(out.end(), rest.begin(), rest.end());
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    int same = 0;
    while (same < (int)out.size() && prompt + same < L && out[same] == toks[prompt + same]) ++same;
    std::printf("generation equals the golden (SGLang) tokens for %d of %zu; tokens hash %016llx\n", same,
                out.size(), (unsigned long long)chosen_hash);
  } else if (a.mode == "generate") {
    int next = sampled[(prompt - 1) % a.chunk];
    std::vector<int32_t> out;
    for (int i = 0; i < a.gen && prompt + i < a.max_ctx; ++i) {
      out.push_back(next);
      const int p = prompt + i;
      if (first_miss < 0 && p < L && next != toks[p]) first_miss = p;
      t_dec += run(p, 1, &next, model::kFlagLogits);
      ++n_dec;
      next = sampled[0];
    }
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    int same = 0;
    while (same < (int)out.size() && prompt + same < L && out[same] == toks[prompt + same]) ++same;
    std::printf("generation equals the golden (SGLang) tokens for %d of %zu; first difference at position %d\n", same,
                out.size(), first_miss);
  } else if (a.mode != "prefill") {
    die("unknown mode %s", a.mode.c_str());
  }
  bcast(ring, Cmd{});  // stop
  if (n_pref) std::printf("prefill: %d chunks, %.2f ms each\n", n_pref, 1e3 * t_pref / n_pref);
  if (n_dec) std::printf("decode: %d steps, %.2f ms each\n", n_dec, 1e3 * t_dec / n_dec);
  if (a.mode != "generate") g.report(stdout);
  if (a.mode != "generate" && a.onset >= 0) g.report(stdout, true);
  g.report_logits(stdout, a.mode == "tf" ? prompt : 0);
  g.report_cmp(stdout, 0, L - 1);
  g.report_cmp(stdout, std::min(prompt, 39) - 1, L - 1);
  if (a.onset >= 0) {  // windows of 4096 positions, split at --onset; then everything from the onset on
    std::vector<int> e;
    for (int p = 0; p < L; p += 4096) {
      if (a.onset > p && a.onset < p + 4096) e.push_back(p), p = a.onset - 4096;
      else e.push_back(p);
    }
    e.push_back(L);
    g.report_windows(stdout, e);
    g.report_windows(stdout, {0, a.onset, L});
    g.report_windows(stdout, {0, 128, 256, 512, 1024, 2048, 4096});
  }
  if (a.per_position) g.report_positions(stdout);
  if (a.mode == "generate" && first_miss >= 0) {
    for (const golden::LogitRec& r : g.logit_recs())
      if (r.pos == first_miss - 1)
        std::printf("at the first difference: ours %d (margin %.3f), golden %d (golden margin %.3f)\n", r.ours,
                    r.our_margin, r.gold, r.gold_margin);
  }
  return 0;
}
