// Rank 0 as a server (docs/protocol.md): the front-end's Unix socket, sessions that keep their state
// across requests with exact prefix reuse, and each request run on every rank through cmd.h's commands.
#pragma once
#include <cstddef>
#include <string>

#include "comm/ring.h"
#include "model/model.h"
#include "server/cmd.h"
#include "server/interleave.h"
#include "server/reclaim.h"

namespace srv {

struct ServeOptions {
  std::string socket;              // the front-end's socket (mode 0600)
  int max_ctx = 65536;             // prompt + output positions per request
  int prefill_rows = 4096;         // rows per prefill chunk
  int engram_mib = 0;              // the Engram row cache's size (MiB of slots) the server keeps asking the model for (0: no cache hand-over)
  int engram_min_mib = 0;          // the smallest it is made to give way to a request (the set size: it never gives way)
  int engram_max_mib = 16384;      // the largest a CONFIG may set (--engram-cache-max-mib: the block's address space)
  int small_rows = 8;              // a remainder of at most this many rows runs as one step (graphs)
  int snap_every = 1024;           // a window snapshot every this many positions (exact reuse)
  bool snap_thin = true;           // keep them dense near a session's end only (serve.cpp keep_snap); false
                                   // (--snap-all): every one kept
  bool ced = true;                 // CED prefill (docs/ced.md) for turns of at least 128 new positions
  int slice = 64;                  // time-slicing (docs/batching.md, B1): a new request pauses the requests
                                   // decoding and is prefilled; while more of them decode than there are
                                   // slots (Model::slots), they take turns of this many tokens. 0: no turns
                                   // (a request that would need the slot of another in flight waits)
  int ssd_flush_s = 30;            // seconds between idle checkpoints; without the imager also the idle age at which a conversation's pieces are written (0: none)
  int grow_ahead = 131072;         // growth (the chunk pool): a response's reservation reaches this many positions past
                                   // where it is, and goes further at its next run (serve.cpp need_now)
  // Kept sessions (docs/engine.md): a planned stop (SIGTERM) spills the sessions in device memory to the
  // disk tier, the most recently used first, up to keep_bytes (0: nothing is kept), and writes every
  // kept session's history to keep_index (beside the disk tier's files); the next start takes them back
  // when the store reused that stop's files. Sessions unused for keep_age_s are not kept.
  std::string keep_index;
  size_t keep_bytes = 0;
  double keep_age_s = 30 * 86400.0;  // also while serving: an idle conversation unused this long is dropped
  // Extend context: on serves up to max_ctx, a YaRN factor per request, a conversation that outgrows its window read again
  // at the next factor; off serves the 1,048,576-token window only (as --max-ctx 1048576 does). Live: CONFIG extend.
  bool extend = true;
  // The graphs and score buffers of factors 32 and 64 are made when a request first needs them, out of the chunk pool
  // (Model::window_open). They are freed when a consumer of the pool is short and no request needs the window (reclaim.h),
  // and, when this is not 0, after this many seconds that no request of the factor existed (0: not by time)
  int window_idle_s = reclaim::kWindowIdleS;
  std::string name = "deepseek-v4.1-flash";
  // DSpark drafts in decode (the model bound them): Cmd's spec_drafts, spec_thresh, spec_cost, spec_gate,
  // spec_row_ms
  bool spec = false;
  int spec_drafts = 5;
  float spec_thresh = -1.f;
  float spec_cost[3] = {19.4f, 3.6f, 3.3f};
  int spec_gate = model::ops::kGateThresh;
  float spec_row_ms[model::ops::kDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f};
  // --profile-from-req N: request N's first command opens nsys's capture range on every rank
  // (model::kFlagProfile; nsys --capture-range=cudaProfilerApi), until the server stops. 0: never.
  uint64_t profile_from_req = 0;
  // Interleaved prefill (server/interleave.h, docs/interleave.md): a prompt with more than interleave.whole_rows rows
  // to compute is read in pieces while others decode; interleave.on false: every prompt is prefilled whole
  il::Settings interleave;
};

// Serves until SIGINT / SIGTERM, then stops the other ranks (kCmdStop). Rank 0 only.
int serve(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o);

}  // namespace srv
