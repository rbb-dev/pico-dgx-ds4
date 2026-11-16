# The engine: this repository's changes

pico-dgx-ds4's [engine.md](../ds41f-4xgx10-switchless/docs/engine.md) describes the engine this repository builds on; this page
describes what this repository changed or added in it.

Contents: [Start-up](#start-up), [Sessions and the store](#sessions-and-the-store) (lines, the chunk pool, bookmarks and
forks, admission and memory given back, the disk tier, the split cache), [Long prompts](#long-prompts),
[Context past 1M](#context-past-1m-a-yarn-factor-per-request), [The decode index scan split over the ranks](#the-decode-index-scan-split-over-the-ranks---pscan),
[The Engram row cache in the display reservation](#the-engram-row-cache-in-the-display-reservation),
[Kept sessions and crash checkpoints](#kept-sessions-and-crash-checkpoints), [Formats](#formats), [Tests](#tests).

The whole memory account (what lives in the store, how its size is chosen, its tenants, what stays free) is
[store.md](store.md); the SSD side of the cache is [split.md](split.md). The served settings are
[`deploy/pico.env.example`](../deploy/pico.env.example).

## Start-up

Serve mode makes the disk tier first: each rank offers its spill directory's free space less 32 GiB, at most `--disk-gib`
(default 32; 512 as served), and every rank makes a file of the smallest offer. Then it makes the chunk pool: each rank offers a
count of chunks from `--pool-free-gib` (its free device memory and its available host memory, the Engram row cache still to grow
counted as used, every user of memory outside the store at its cap) and/or `--pool-tokens`; one all-gather, and every rank takes
the smallest. Rank 0 logs the size and every rank's free memory. `--slot-gib`, `--session-gib`, `--session-margin-gib` and
`--no-vmm` are refused at start: the store's size comes only from `--pool-free-gib` and `--pool-tokens`.

The memory a rank holds beside the weights, as served:

| What | Size | Where |
|---|---|---|
| the chunk pool (the store) | chosen at start ([store.md](store.md)) | every conversation's compressed caches and index keys, 2,916,352 B a chunk, the same count on every rank |
| the Engram row cache | `--engram-cache-mib` (2,032 as served) | host memory lent by the store; its first slots, up to 2,032 MiB, in the display reservation with `--engram-vram on` ([below](#the-engram-row-cache-in-the-display-reservation)) |
| the image encoder's weights and work area | a quarter of the tower; two groups of the store | [vision.md](vision.md), [media.md](media.md) |
| the lines | 5.8 MiB a line | [below](#lines) |

## Sessions and the store

Every conversation is a session of the store ([state.cu](../ds41f-4xgx10-switchless/engine/model/state.cu)). Its state is
its window rings and carries, which live in the line that holds it, and its compressed caches and index keys, which are its
own chunks of the pool whether a line holds it or not. A session holds its lists of chunks (positions [0, saved) of its
caches); window bookmarks (each layer's ring slots of positions [s - 128, s), with the drafter's windows: 2.9 MB in 2 chunks) at
the end of its last prompt and every `--snap-every` = 1,024 positions, thinned further back; and an end bookmark (every layer's
whole ring and the carries, with the drafter's windows: 6.1 MB in 3 chunks). The engine's log and STATS call a chunk a
"number".

**A conversation on a line is its only copy.** The line writes into the conversation's own chunks, so saving a turn
(`persist_ops`) is bookkeeping and the end bookmark; when a line lets it go, nothing is copied.

**What a restart needs.** A restart at the end bookmark needs the caches up to its position (`saved >= end_pos`); one at a
window bookmark s needs them up to its restart position (s, or s - 1 for an odd s). `StateInfo::restart` is the latest of
these a session holds, whether its pieces are in the pool or on the disk tier.

### Lines

`--slots N` (1 by default, up to 64; 32 as served) lays out the per-line state N times: the window rings, the carries, the
drafter's windows and Engram's history (5.8 MiB a line: rings 40 layers x (131,072 + 4,096) B, carries 0.25 MiB, the
drafter's windows 0.39 MiB; Engram's history is host memory, 4 B a position). A line holds one conversation at a time and no
cache memory of its own. A request takes its conversation's line; else a free one; else the least recently used one whose
conversation no request uses; else, with turns, the line of the request whose turn comes last. A batch step takes 16
conversations (`ops::kMaxSess`), so a run takes at most 16 requests and the other live ones take turns (`--slice`, 64
tokens). The dashboard shows one row a line. A prompt with more rows to compute than `--interleave-whole-rows` is processed in
pieces between the decoding sessions' steps ([interleave.md](interleave.md)).

### The chunk pool

([chunks.h](../ds41f-4xgx10-switchless/engine/model/chunks.h), [vmm.h](../ds41f-4xgx10-switchless/engine/model/vmm.h),
`--pool-free-gib`, `--pool-tokens`, `--pool-pick`.) The four cache sources' arrays (compressed KV, its scales, index keys,
their scales) are four pool-wide arrays cut into slices of 8,192 entries; a chunk is one slice of each: 2 MiB of KV, 256 KiB
of scales, 512 KiB of keys and 32 KiB of key scales, 2,916,352 B on each rank. A chunk holds 16,384 positions of a half-rate
source (layers 2, 8, 14) or 8,192 of layer 20, so a conversation takes 5 chunks every 16,384 positions (about 890 B a
position a rank). Chunk 0 is the guard: never taken, every list entry past a list's end names it, and it reads zero (0xFF
with `DSGB10_VMM_POISON=1`).

- **Its size, once.** Every chunk is made and mapped before serving (`cuMemCreate` costs 39 ms a GiB on a GB10, measured), and
  nothing is made, mapped or unmapped while serving, except whole groups of 64 chunks lent to the store's tenants. Each rank
  offers a count from what it would leave free (`--pool-free-gib G`) and/or from positions (`--pool-tokens N`: N x 5 / 16,384
  chunks), rounded down to whole groups; with both, `--pool-pick smaller` (the default) or `larger`. Every rank takes the
  smallest rank's count, so every rank has the same chunks. The start's log says `chunk pool: N numbers of 2.78 MiB (G GiB,
  P positions) on every rank (...)`, followed by the table of the users of memory outside the store ([store.md](store.md#the-sizing-rule)).
- **Lists.** A conversation holds one list of chunks per source: its entry e of a source is the pool entry
  `(list[e >> 13] << 13) | (e & 8191)`. A line's rows of the device table (`lists_d`: one row a unit and source, 136 entries
  at 1M) are its conversation's lists, padded with the guard; kernels find a row through the step's slot table, so captured
  graphs keep their addresses. Index scans and cache writes look entries up one by one; the selection writes each key pick as
  a pool entry, so attention reads the picks as given and one lookup serves the 4 to 6 layers that share a top-k list.
  Measured: a decode step 0.03 % longer, prefill within 0.6 %.
- **Taking and giving.** Chunks come lowest free first, the same on every rank: rank 0 decides and sends each command with
  its state operations, every rank applies them in the same order, and the decisions read only the allocator and the store's
  records, never timing or free memory (the allocator keeps an order-free digest of the taken set). A line's conversation
  grows its lists 4,096 positions ahead of what the next step needs (`ensure_mapped`); a list entry is a 4-byte copy queued
  before the step that reads it.
- **Lines point; nothing copies.** `kStAttach` makes a line's rows the conversation's lists (a row write, after waiting on the
  stream so no queued copy reads a row being rewritten); a line's release points its rows at the guard and the conversation
  keeps its chunks. A conversation that comes back to another line costs no copy of its caches: only its rings and carries
  come back, from its end bookmark.
- **A chunk given back.** Every device access to a chunk is queued on the model's stream or a stream joined to it at each
  step, so a chunk given back can be taken again at once. With poison on, a chunk given back holds 0xFD and a taken one 0xFF
  until written.

### Bookmarks, forks and growth

- **Bookmarks.** A window bookmark takes 2 chunks and the end bookmark 3, used as raw bytes in their KV slices. A bookmark
  takes only chunks that leave every line's reservation covered, counting what is free and what evicting can give back
  (`take_extra`); with none to spare it is not taken, and a later restart starts at an earlier one.
- **Forks by reference.** Every chunk has a count of the lists and bookmarks naming it (`refs`). A fork takes its base's
  chunks wholly below its restart point by count and copies only the entries below the restart point of the one chunk holding
  it (kilobytes to 2.8 MiB); window bookmarks at or below it are shared the same way. Nothing writes a chunk another list
  names: a cut of a conversation's own history, a drop and a cold start first make the chunks from the cut on its own
  (`own_tail`), and a bookmark written again takes fresh chunks when it is shared. A base can be freed before its forks; its
  chunks stay while any fork names them.
- **Growth.** A request reserves its prompt and a horizon of output, `--grow-ahead` positions (131,072 by default, at most its
  `max_new`), not every token it may generate (a request that names no `max_tokens` would otherwise reserve a whole 1M window,
  324 chunks). An output that reaches its horizon ends its run and its reservation goes a horizon further at its next run when
  the pool can hold it; else it waits in the turns' queue like a paused request, and its line's memory counts as idle for the
  others. An output alone is split at most once every 131,072 tokens. Six 1M outputs on a small pool decode in one batch
  (`growth_test.py`).

### Admission and memory given back on demand

A request is admitted when the chunks it needs (from 0, and what a read-back from the disk tier takes) fit what is free less
the other reservations not yet held, plus its line's own, plus what idle lines' conversations hold, plus the chunks only
evictable conversations name (`state_giveable`: a shared chunk counts when every list naming it is evictable). `make_room` then
lets idle lines go; their conversations stay in the pool and become evictable.

Every chunk is free, named by a running request, or cache. A request is running while it is in flight and not paused between
turns (`--slice`); a queued long prompt is running from its admission. Cache is everything else: stored conversations,
conversations kept on an idle or paused line, their bookmarks, their images in the disk tier's file, an open window no request
needs. Cache gives way in this order: free chunks; stored conversations, least recently used first (the last piece to the
disk tier, or dropped whole without one); the conversation on an idle or paused line (the line is detached: its state saved
with an end bookmark, then it is a stored conversation); and, for a request's growth, a copy or a window, a window no request
needs. A window bookmark or a read-back from the disk tier never closes a window: it takes what the steps before give.
Running requests are never taken from: a consumer short only of what they hold waits, and the log names the requests. Nothing
is freed on a timer; memory goes back when a consumer asks.

- `Server::give_back` runs before every consumer of chunks that a line could be holding: a request's slot memory
  (`make_room`), the extension of an output's reservation, the window bookmarks of a prefill piece or of a decode run, the
  prompt's end bookmark, the copy a restart point makes of chunks a session shares, a conversation read back from the disk
  tier. A read-back takes only what no running request has reserved (`take_restore`).
- A finished request's reservation is zeroed with its last save (`kStUnreserve`), so a line that stays attached holds the
  chunks it maps and reserves nothing; a paused output keeps its reservation until its line is detached. With nothing in
  flight, a line reserving chunks it does not hold is logged as `BUG: idle line holds a reservation`.
- `Cmd::live` names the conversations of running requests only: those on idle and paused lines are evictable like stored ones.
- A window is made from a plan (`Model::window_plan`): the sessions holding a chunk the window takes are detached when idle or
  paused, the pool gives back what is still short, then one command goes out. While a running request holds what the window
  needs, no command is sent: the request waits, retrying when a request ends or crosses and at least once a second.
- When every request that holds memory waits to grow and nothing can be freed, the engine logs `serve: BUG: every flight waits
  for an extension` and, once the standstill has lasted 2 seconds, ends the request with the longest output so far with reason
  `length` (README, [Requests that wait for memory](../README.md#requests-that-wait-for-memory)).
- A client that left: the front-end cancels the request, every queue is swept (waiting for a line, held for its session, let go
  to run) and admission looks first, so a cancelled request makes no window and takes no line.
- `DSGB10_STORE_CHECK=1` prints the store's digest after each window, state and reservation command; the four ranks' lines
  must agree.

### The disk tier, piece by piece

Every piece of a conversation, a cache entry (one chunk's four slices) or a bookmark, can have an image in the disk tier's
file (unless `--disk-gib 0`). Each rank writes its own to its own disk: `rank<r>.spill` in `--disk-dir`, by default the fleet
file's `RANK<r>_SPILL`. The file's mechanism (one file a rank, `O_DIRECT`, pinned staging and a writer thread, kept across a
planned stop) is pico-dgx-ds4's; what changed is what it stores and when:

- **An image is written once while its content holds.** A turn's save writes every full entry (a chunk whose 8,192 entries
  are all written) and the window bookmarks below it that have none; a conversation idle for `--ssd-flush-s` (30 s) gets the
  rest written: its tail entry and its end bookmark. Eviction writes a piece only when it has no image. A read-back keeps the
  image, so a conversation squeezed out again writes nothing it wrote before. The holders of a shared chunk share its image. An
  image goes stale only where the conversation writes again: a cut, a drop and a cold start clear the images from the cut on,
  a turn's save clears the image of an entry it wrote into, and a bookmark taken again drops its image.
- **Eviction, the last piece first.** The least recently used conversation that no line holds and the command does not use
  gives up one piece at a time: its end bookmark, then window bookmarks from the latest, then cache entries from the latest
  position down. A piece with an image gives its chunk back at once. With no room on the disk the piece is dropped and the
  conversation is cut to below it. Beginnings, shared ones above all, stay in memory longest.
- **Read-back.** A request whose conversation (or fork base) has a piece only on the disk brings it back before it plans its
  restart: cache entries from the first position, then the end bookmark, then window bookmarks from the latest, as far as the
  pool has room (a conversation is cut to below the first piece that finds none), each image checked against its checksum (a
  piece that comes back different cuts the conversation to below it).
- **Its room** is a pool of 64 KiB blocks; every rank runs the same operations on it, so every image lands at the same
  offsets. When it is full, the least recently used conversations give up their images (one that needs them is dropped:
  `state: session … dropped from the disk tier`). Writes go through one gather buffer into the pinned staging; reads through
  two 8 MiB pinned buffers, after the writes queued before them.
- **Size and expiry while serving.** The dashboard's SSD CACHE SIZE (`CONFIG` `ssd_gib`) makes the limit smaller inside the
  file made at start, or back up to it, applied when no request is in flight or paused: a smaller one gives up the images past
  the new end, every rank alike. A conversation unused for `--keep-days` (30; FORGET UNUSED AFTER, `CONFIG` `keep_days`) is
  dropped while nothing runs.

### The split cache

`--kv-split off|tiered|max` (`tiered` as served). Every write to the SSD tier, in every mode, is each Spark's own quarter of a
piece, made in the background on a second stream and never waited for: a decode step, a prefill chunk or an admission shares
no stream with it. A piece is plain (no image), clean (its quarter image is on the SSD and its chunk still holds the same
bytes) or, for `tiered` and `max`, a ghost: the conversation says "on the SSD", the chunk is free and its bytes are intact. A
return takes a ghost back with no read when nobody used it; a ghost something took is read from the four SSDs and exchanged
(about 0.46 ms a piece, measured on a test build of the split cache). An eviction never writes: it takes clean pieces only,
and admission counts only chunks whose images exist. Images are queued by an idle command when nothing is in flight, and inside
a decoding run by rides (a bit of the step's control word, bit 29, one host all-gather and the store's bookkeeping between
two steps). The modes, options, log lines and STATS fields are in [split.md](split.md).

## Long prompts

A long prompt is processed 16,384 rows at a pass with `--prefill-rows-max 16384` (as served), on a second set of row buffers that
the store lends while it is free; the set goes back first when the store needs the room (`--prefill-loan-low-groups`, 2), and
is asked for again only when the set's groups plus `--prefill-loan-high-groups` (8) are free. Without it a pass processes 8,192
rows (`--prefill-rows`). The rules and the measurements: README, [Long prompts processed 16,384 rows at a time](../README.md#long-prompts-processed-16384-rows-at-a-time).
A prompt with more rows to compute than `--interleave-whole-rows` is processed in pieces between the decoding sessions' steps
([interleave.md](interleave.md)).

## Context past 1M: a YaRN factor per request

The checkpoint's rotary scaling is YaRN with a factor of 16, which holds 1,048,576 positions. With `--max-ctx 4194304` (as
served) the engine serves up to 4,194,304 positions by choosing the factor per request: 16 up to 1,048,576 tokens, 32 up to
2,097,152, 64 up to 4,194,304. With `--max-ctx` at or below 1,048,576 every request is factor 16 and none of the paths below is
taken. The setting `extend` turns the growth past 1,048,576 on and off while the engine serves (the start flag `--extend
on|off`, on by default; the dashboard's EXTEND CONTEXT; `CONFIG` / `STATS`, [protocol.md](protocol.md#context-past-1m)): off,
the window is 1,048,576 (a longer prompt is refused, an output that reaches it ends with reason `context`).

**How the factor is chosen.** A request starts at the smallest factor whose window holds its prompt (`factor_for` in
`serve.cpp`; the front-end computes the same function from the prompt's ids, `session.rs`). Memory grows in chunks as the
request writes. A session has one factor for its whole life, and every match (a continuation, a restart's longest common
prefix, a fork, a read-back from the disk tier, the front-end's prefix index) looks only at sessions of the request's factor,
so state written at one factor is never read at another: rotary is applied before a key is stored, so keys cached at one
factor are wrong at another.

**Crossing a boundary.** A conversation holds one primary session, plus alternates of other factors that rank 0 keeps under
internal keys.

- *A turn whose prompt outgrows its factor* gets a new session, processed cold at the next factor (the front-end names a new
  session id). The old session stays as an alternate for siblings (a fork, a regenerate) whose prompts are under the boundary;
  with a disk tier it leaves the pool at once and a sibling reads it back.
- *An output that reaches its window* pauses at the window's last position; the prompt and the output so far are processed again at
  the next factor (in pieces when other requests are decoding), and decoding goes on from the sampled token. An output past
  4,194,304 ends with reason `context`. Rank 0 logs `req N crosses from factor A to B at P positions`.
- *Exactness.* Both paths process the tokens again at the new factor, so the result is the cold Prompt Processing of the same prompt bit for bit
  (tokens and log-probabilities: `factor_test.py`). Nothing is approximated.

**What it costs.** A crossing costs one read of the whole conversation at the new factor, once per boundary: 115 s for
1,056,768 tokens at factor 32 and 255.8 s for 2,105,344 at factor 64 (four Sparks, one conversation, the exact path of
`needles_boundary.py`, measured on an earlier build that processed prompts 8,192 rows a pass). Cold prompts on the served build,
through the front-end with the dashboard open: 999,999 tokens in 97.3 s, 2,050,000 in 225.0 s and 4,149,999 in 552.6 s, then
105.5, 105.0 and 95.8 t/s of decode (README, [Context to 4M](../README.md#context-to-4m)).

**Batching and graphs.** A step never mixes factors: `run()` takes the requests of the first one's factor and leaves the others
at the front of the rotation; each slot carries its factor (the state operation `kStFactor`). The captured graphs are keyed by
(rows, context, factor), because the captured kernels' parameters hold the factor.

**Rotary values and selection.** Factor 16 reads the rotary table built at start; factors 32 and 64 compute each cos and sin in
the kernels from (position, factor), by the code that builds the table (`rope_calc.cuh`), so no table of 0.5 or 1 GiB a rank is
spent per factor. The kernels' factor-16 values equal the table's bit for bit (`rope_factor_test`). The dense index scan has two
instantiations: at or below 1,048,576 keys the standard path; above it a wide path that lists the selected blocks in segments
of 65,536 and flushes its packed 16-bit histogram counters every 64 chunks, since a counter holds 65,535 and a 4M scan can
overflow it (`sel_tests`: selection against a host top-k at 256K, 1M, 2M and 4M).

**A window opens when a request needs it.** The engine starts with what factor 16's window needs: its step and batch step
graphs and the index scans' score buffers sized for 1,048,576 keys, so the pool of a `--max-ctx 4194304` start has the size of
a 1M start's at the same `--pool-free-gib`. The graphs of factors 32 and 64 and their score buffers (at 4M: decode's 144 MiB and
the prefill's 1.13 GiB; at 2M 72 MiB and 0.56 GiB; each factor's graphs about 1.1 to 1.3 GiB, measured: 3.56 GiB for the three
factors against 1.29 GiB for factor 16) are made when a request first needs the factor. Rank 0 sends a window command
(`kCmdWindow`) before the request's first step, and every rank does the same (`Model::window_open`, `window.cu`):

1. Each rank sizes the window (its buffers, and 1.25 times the graphs factor 16 took at start), and the largest figure decides
   how many groups of 64 chunks the pool gives up (64 chunks is the smallest set whose slices of all four arrays are whole
   2 MiB allocations, 178 MiB).
2. The chunks above the pool's new end are made free: the conversations holding them, least recently used first, give up their
   pieces; a line, a pinned or a live conversation holding one stops it (the request waits, the pool is not touched).
3. The allocator drops the chunks (`chunks::Alloc::shrink`), the arrays unmap their tail (`vmm::Pool::shrink`: the addresses of
   what stays mapped do not change, so factor 16's captured graphs stay valid), and the physical allocations go back to the
   driver (`Pool::trim`) for the new buffers and graphs.
4. Each rank makes the buffers and captures the factor's graphs, measures what that took, and every rank gives back the groups
   the smallest measurement did not need.

A rank with too little device memory undoes it on every rank (the request waits; three failures in a row refuse it); a pool
that could not hold one conversation of the window refuses the request (`kWindowNoRoom`). A window is closed when a consumer of
the pool is short and no request needs it, and by time only with `--window-idle-s S` (default 0: never by time); the pool then
makes the chunks again. Opening costs the capture time (about 1 to 1.6 s of CPU for three factors, measured) plus the evictions
it needs. The log has `window: factor F open ...` and `window: factor F closed ...`.

## The decode index scan split over the ranks (`--pscan`)

A decode step's index scans (layers 2, 8, 14 and 20) read every key on every rank. With `--pscan ext` the step graphs of
factors 32 and 64 split each scan by positions instead: rank `r` scans the `r`-th quarter of every session's own keys
(`index_dense`'s `part` / `nparts`, in 64-key units), its 512 top candidates (layer 20 also its 2,048 pool candidates) go into the
rank's slot of a gathered buffer, one `allgather_multi` brings the four ranks' lists, and `index_merge2` reads them in place,
rank-major, into the picks the whole scan would write: the same positions, keys and pool, bit for bit. `--pscan all` (as
served) splits factor 16's graphs too; `off` (the default) makes no part buffers and launches no merge kernels. A step of more
than 16 rows scans whole. The decision reads only the option, the step's factor and its row count, so the four ranks make the
same collectives in the same order. Cost: 0.94 MiB of candidate buffers a rank. Measured on an idle GPU, one rank's part plus
the merge against the whole scan: layer 20 at 1M 349 to 130 us and at 4M 1,440 to 410 us (one row); in the engine, rank 0's
step time at 1M, 3 % shorter at 2 rows a step and equal at 6 rows; the picks equal in 72 of 72 cases, tokens and log-probabilities equal to the whole scan
in 8 of 8 turns (README, [Kernels](../README.md#kernels)).

## The Engram row cache in the display reservation

A GB10 sets aside about 2 GiB of system memory for its display; a headless Spark never uses it and `MemAvailable` does not count
it. A DRM dumb buffer made on `/dev/dri/card0` (`CREATE_DUMB`: width 4096, 32 bits a pixel, 2,032 MiB at most, lower in 16 MiB
steps) is that memory, and `MAP_DUMB` plus `mmap` give the CPU a mapping of it (`engram/display.{h,cpp}`).

Only memory the CPU alone touches may live there. Measured on a Spark: the CPU mapping is write-combined (writes post, reads go
to the device uncached and are far slower than ordinary memory), a GPU kernel reading the buffer through
`cuImportExternalMemory` reads it uncached too, so no weights or KV go there, and `O_DIRECT` transfers into or out of the mapping
fail with `EFAULT`, so the disk tier's staging cannot use it. What fits is the Engram hot-row cache's slot array (`RowCache` in
`engram/rowcache.h`): an insert writes a row, a hit reads one, and only the CPU worker touches it. The cache's first slots, up to
the reservation's size, are in the mapping and the rest in ordinary memory, under one slot numbering; the CLOCK index stays in
ordinary memory. The pool's sizing counts only the ordinary part of the cache as still to grow.

`--engram-vram off|on|require` (default off; `on` as served). With `on` or `require`, each rank opens the card before the store
is made, refuses when a display output is connected, makes the largest buffer it is given up to the cache's size, and every rank
uses the smallest rank's size. When any rank cannot, all keep the cache in ordinary memory with one start line saying why;
`require` stops instead. The start line of a placement: `engram row cache: N MiB of its M MiB slots in the display reservation
..., K MiB in ordinary memory`. `--engram-vram-card`, `--engram-vram-mib` and `--engram-vram-sysfs` are for tests. The mapping and
the buffer are released at exit.

Measured on the four Sparks through rank 0's socket, freshly started, one conversation of real text, 4 lines and `--pool-free-gib
10`, with `on` and with `off`: the pool held 8,337 and 7,575 chunks (+762, +2.07 GiB, +2.5M positions a rank), the memory left to
the host kept in both; prompts of 24K, 1M, 2M and 4M processed and decoded within 0.6 % of the `off` run; outputs bit-equal
(`server/tests/engram_vram_test.py`).

## Kept sessions and crash checkpoints

A planned stop keeps the sessions for the next start of the same build, as pico-dgx-ds4 does
([Kept sessions](../ds41f-4xgx10-switchless/docs/engine.md#kept-sessions)). This engine also comes back from a crash.

**Checkpoints while serving.** When nothing runs, the disk tier's images changed since the last checkpoint, and `--ssd-flush-s`
(30 s) has passed since it, rank 0's server saves every idle line's conversation as a planned stop does (its caches to its
history's end and its end bookmark; the line keeps it) and writes its pieces, then sends `kStCheckpoint` with a new generation:
every rank waits for its writes, `fdatasync`s its file and writes its index, naming every conversation as far as its pieces
from position 0 on have images; then the server writes the histories of those conversations with the same generation (`serve:
checkpoint: …`). A turn's full entries are written at its end and an idle conversation's tail and end bookmark after
`--ssd-flush-s`, so a checkpoint holds every conversation idle that long whole. An image the latest index names keeps its file
space until the next index is durable, so no write lands where an index on disk points. A crash comes back to the last
checkpoint as to a planned stop: a conversation that went on after it restarts at its checkpointed position, and the turn since
is read again from the history the client sends. The indexes stay after a start: they are the latest checkpoint until the next.

**What the indexes hold.** The server's index, `server.<tag>.index`, names each session's id, its restart ranges, when it was
last used, its epoch and the history size its last DONE reported, and the count and sum of its ids file
(`<disk-dir>/ids/<handle>.<seq>.ids`: the token ids live in that file, not in the index). A file an index names is never
changed: every write of a conversation's ids is a new file, and a file no index names is removed only after the next index is
on the disk.

**The front-end's table.** dsfront saves its session table while serving and at its own stop (`--table-file`;
`run/front.table` under the units): each session's cut points, with no text and no ids
([protocol.md](protocol.md#a-request-with-only-its-suffix)), by which a restarted dsfront sends a conversation's next turn to
the same session. It saves when the table has changed, at most every `--table-save-s` seconds (30; 0: only at the stop), whole
or not at all (a temporary file, synced, renamed over it, the directory synced), off the request path.

**What starts clean.** A new build (the run id hashes the binary, the weights' and Engram files' paths, sizes and times, the
options that change what a request computes, and with `--ablit` the abliteration file's bytes); a crash before the run's first
checkpoint or between the ranks' indexes and the server's (the generations differ); a stop that failed partway; a damaged or
foreign file, or one older than `--keep-days` (30); `--keep-gib 0`. A block that comes back different from its checksum
(`state: session …: ERROR: … differ from what was written`) cuts its conversation below it on every rank alike: each rank's
first bad position is shared and every rank cuts at the smallest, so no turn continues from a wrong state.

## Formats

| File | Format | Another version |
|---|---|---|
| each rank's kept index on its disk tier | `DSGB10K5`: a factor (16, 32 or 64) per session, every image a rank's quarter image with a trailer (a cache entry's or a window bookmark's slice 786,432 B, the end bookmark's 1,572,864 B) | an index of another layout (`DSGB10K3`, `DSGB10K4`) is not read: the engine starts clean and says so, once; the index write is agreed by the four ranks (all adopt or none) |
| rank 0's server index | `DSGB10S3`: factor and owning conversation, the epoch and the ids file's count and sum per session | another version is refused the same way; a session whose factor the ranks' store does not confirm, or whose window exceeds `--max-ctx`, is dropped at load |
| the front-end's session table | `DSFT0004`: per session its cuts (bytes of text, ids, hash), its index keys, the engine's epoch and history size, a factor word; no text and no ids | `DSFT0003`, `DSFT0002` and `DSFT0001` are read once and become cuts, with no epoch (the first request carries its whole prompt); a factor other than 16, 32 or 64 refuses the load |

Images of the disk tier carry no factor: they are named by their session. `HELLO`'s `max_context` is the engine's `--max-ctx`,
which the front-end reads for `/v1/models` and the output-room clamp.

## Tests

`engine/model/tests`: `chunks_test` (a million random operations against a reference model; four ranks replaying one stream
agree, one skipping an operation differs), `vmm_test`, `vmm_probe.cu`, `store_core_test.cpp`, `lend_test.cpp`.
`engine/server/tests`: `fork_test.py` (the same requests twice, the second rolling the base back inside the shared chunks and
freeing it; a build without copy on write, `-DDSGB10_POOL_NOCOW`, fails it), `growth_test.py`, `stats_test.py`, `squeeze_test.py`
(under pressure with poison on), `kept_test.py` (across a planned restart), `kept_factor_test.py`, `lines_test.py` (32 lines),
`room_test.py`, `pool_soak.py`, `factor_test.py` (the factor, the exact crossing, the crossing mid-output), `lazy_window_test.py`
(a 4M start's pool equals a 1M start's; the window opens, closes back exactly and reopens), `needles_boundary.py` (real text with
a sentence to find before, under, at and after 1,048,576, and 2,097,152 with `--also-2m`: 4 of 4 at each boundary),
`engram_vram_test.py`; `engine/attn/tests`: `rope_factor_test.cu`, `sel_tests.cu`; `bit_probe.py` against the recorded build
with and without poison.
