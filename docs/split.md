# The split cache

An idle conversation's cache is written to the SSDs as each Spark's own quarter, in the background, and the memory it held is
marked free without being copied or cleared. A conversation that comes back before anyone took its memory takes it back with
no read; otherwise the four Sparks read their quarters from their own SSDs and exchange them. This page describes the pieces,
the three modes, the options, what the engine logs and what it measures. The user's view is in the
[README](../README.md#the-ssd-tier-the-split-cache-and-crash-checkpoints); the tier's file, staging and writer are pico-dgx-ds4's
(pico-dgx-ds4's [engine.md](../ds41f-4xgx10-switchless/docs/engine.md#kept-sessions)); [engine.md](engine.md#the-disk-tier-piece-by-piece) says what this engine writes to it.

## Why quarters

The cache arrays are replicated on all four Sparks (tensor parallelism does not shard them), so every Spark holds every
conversation whole. A rank's quarter of a piece is every fourth entry of its cache block for the codes and every fourth run of
512 for the keys. Written to its own SSD, the four quarters together are one copy of the piece, kept on four drives; any
Spark's memory for it can be freed, and the four read in parallel to get it back. RAM still holds a conversation whole or not
at all: the four-fold saving is on the SSDs, where each Spark keeps a quarter.

## The pieces

A conversation's **piece** is a cache entry (one chunk's four slices) or a bookmark (a window bookmark's slice, or the end
bookmark's). Its image on a Spark is that Spark's quarter plus a trailer with checksums: 786,432 B for a cache entry or a window
bookmark, 1,572,864 B for the end bookmark.

| State | Meaning |
|---|---|
| plain | the conversation holds the chunk; no image yet |
| clean | the chunk is held and its quarter image is on every Spark's SSD (at least in staging), proven on all four |
| ghost | the conversation says "on the SSD"; the chunk is free and stamped; its bytes are intact and linked to the image |
| out | "on the SSD" with no chunk linked |

- **Imaging** (plain to clean) is the write-behind, in quarters, off every request's path.
- **The mark** (clean to ghost) is the split. It is host bookkeeping: no device call, no stream, no copy. It marks every clean
  piece of a conversation that is on no line and has no request in flight. To the rest of the engine a marked conversation is
  a conversation on the SSD whose chunks are free.
- **A take** (ghost to out) is the allocator handing the chunk to someone else: plain free chunks go first, then the oldest
  ghost. The take cuts the ghost's link to its image, voids a queued image job for it and dissolves a bookmark's ghost group.
- **The return** names the same chunk again when the entry's image still names a ghost nobody took (no read, no exchange,
  no device call); otherwise it reads the quarters and exchanges them, bookmarks included.
- An image goes stale only where the conversation writes again (a cut, a drop, a cold start, a turn's write into an entry, a
  bookmark taken again), and exactly those images are cleared. Holders of a shared chunk share its image.

## Modes

`--kv-split off|tiered|max` (default `off`; `cache` is a name for `max`). Every mode writes the quarter images whenever there
is an SSD tier (`--disk-gib` above 0); the modes differ in marking and in when images are made.

| | images | marks | rides inside a decoding run |
|---|---|---|---|
| `off` | written | nothing: a conversation keeps its chunks until a take evicts them, clean | only while the chunks that can be given back without a write are under `--kv-split-high` |
| `tiered` | written | conversations on no line with nothing in flight | the same |
| `max` | written | the same | whenever images are pending |

In every mode a request that waits for clean chunks makes the imager ride: at an engine with nothing else running it is not
refused while images are still to be made, and its rides carry 64 pieces each.

## How the imager runs

- **Idle:** when nothing is in flight, queued, waiting or arriving, an idle command queues up to `--kv-split-budget` pieces
  (16), after the conversation has been idle `--kv-split-after-s` (0: at once).
- **In a run:** a **ride**. Rank 0 flips one bit of the step's control word; every rank sees it at the end of the same step, one
  host all-gather agrees the budget, and the store's bookkeeping happens between two steps, at most every
  `--kv-split-ride-ms` (250) and `--kv-split-ride-pieces` (4) pieces, 64 when the chunks that can be given back are under
  `--kv-split-low` (default: half of `--kv-split-high`).
- **Never:** between a prompt's chunks, at an admission, at a request's end or when a line is let go. A request that waits for
  unwritten quarters while a prompt is processed alone waits for its end (`split_reader_waits_total`, with one log line
  each).
- **The streams.** Images are made on a second stream and never waited for by a decode step, a prefill chunk or an admission.
  An image command's first call makes the second stream wait for the first stream's queued
  steps, one way.
- **The budget is agreed.** It is counted in units of 786,432 B (the end bookmark's image is two units) and every Spark queues
  exactly the same images. Each rank settles up to the smallest reading of the four image streams, so nothing is marked early
  anywhere. A ride inside a run marks no piece that several conversations share; an idle command does.

## Eviction and admission

An eviction never writes: it takes clean pieces only, last piece first (the end bookmark, then window bookmarks from the
latest, then cache entries from the latest position down), and admission counts only chunks whose images exist. A piece with
an image returns its chunk at once. A take that still finds only plain pieces (a broken count) makes the piece's image, waits
for it, counts it (`image_waits_total`, expected 0) and takes.

## Options

| Option | Default | |
|---|---|---|
| `--kv-split off\|tiered\|max` | off | the mode |
| `--kv-split-after-s S` | 0 | idle age before a conversation is imaged |
| `--kv-split-high N` | the larger of 256 MiB and 1/16 of the store, in chunks (printed on the start line) | above it `tiered` and `off` do not ride |
| `--kv-split-low N` | half of high | under it a ride queues 64 pieces |
| `--kv-split-budget N` | 16 | pieces an idle image command queues |
| `--kv-split-ride-ms N` | 250 | at most one ride this often |
| `--kv-split-ride-pieces N` | 4 | pieces a ride queues |
| `--kv-split-pass-ms N` | 50 | the longest one pass of split commands runs with requests in flight at full pressure (down to a fifth of it as the pressure falls; 250 ms with none in flight) |
| `--kv-join-pieces N` | 64 | pieces one return command brings back |
| `--kv-join-whole-ms N` | 250 | a return estimated at most this long (or with nobody decoding) goes back to back; above it, in units |
| `--disk-gib G` | 32 | the tier's size at most; 0 turns the tier and the imager off |
| `--ssd-flush-s S` | 30 | the least time between two checkpoints while nothing runs and the store's images changed (an idle conversation's pieces are written by the imager) |


## What it logs and reports

- At the start: `serve: kv split MODE: ...` with the idle command's size, the ride's pieces and spacing, the high mark, and
  whether pieces are marked free where they lie or evicted clean; the split's buffers are one of the lines outside the store.
- Per run: `serve: split rides in the run: R rides, I imaged, M marked, longest X ms`.
- STATS (and so the dashboard's tooltips): `ghost_numbers`, `split_pieces_total` (pieces marked), `revives_total` (ghosts taken
  back with no read), `returns_read_total` (returns that read the SSDs), `conv_split`, `split_deferred_total`,
  `clean_numbers`, `image_pending`, `split_rides_total`, `image_waits_total` and `split_reader_waits_total`.
- The dashboard: CONVERSATIONS counts conversations whose piece is only on the SSDs; CONTEXT CACHE's tooltip says how many GiB of
  free chunks still hold a ghost's bytes; the CACHE chart and CACHE HITS tile split prompt tokens into memory and SSD, and
  the CACHE chart's hover box says how many pieces were taken back from memory with no read.

## Formats

The kept index is `DSGB10K5`: every image is a rank's quarter image with a trailer, with no flag byte and no whole image. A
kept index of another layout (`K3`, `K4`) is not read; the engine starts clean and says so, once. The index write is agreed by
the four ranks (all adopt or none). Images an index names are held until the next index is durable.

## Measured

On four GB10 with `--kv-split tiered`, on a test build of the split cache: a conversation of 64 pieces came back from the SSDs
in 28.7 to 29.4 ms, about 0.46 ms a piece (the SSD quarters read, exchanged and checked, in one command), resuming at its last
position. A conversation evicted and read back is bit-equal at its restart point.

## Tests

`engine/model/tests/store_core_test.cpp` (the core: imaging, settling, marking, the cut on take and on write, the return, each
rule with a control that fails without it), `engine/server/tests/split_pace_test.cpp`, `mem_watch_test.cpp`,
`split_return_selftest.py` and `split_wait_selftest.py` (source checks that the request paths reach no ride, mark or join),
`split_test.py` (GPU cases: the imager, the return, the wait, the reader), `quarter_test.cu` (a quarter made on a second
stream beside a busy one is byte-equal to the as-built image), `disk_test.py`, `fork_test.py`, `squeeze_test.py`.
