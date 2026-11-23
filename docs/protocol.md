# The front-end protocol: this repository's changes

pico-dgx-ds4's [protocol.md](../ds41f-4xgx10-switchless/docs/protocol.md) describes the socket between the front-end (`dsfront`) and
rank 0 of the engine this repository builds on: the transport, the frames and how a session is reused. This page describes the
frames and fields this repository added or changed. The layouts in code are `engine/common/frontend_protocol.h` and
`frontend/src/protocol.rs`.

## Frames added or changed

| Code | Frame | Direction | This repository's part |
|---|---|---|---|
| 0x02 | `SUBMIT` | front-end to rank 0 | after the `SubmitHeader`, `{u64 epoch; u32 n_prefix; u32 0}` when `flags & PREFIX` ([below](#a-request-with-only-its-suffix)); then the stop ids and the prompt as before |
| 0x05 | `STATS_REQ` | front-end to rank 0 | `{u32 0; u32 0}`: send one `STATS` ([the dashboard's frames](#the-dashboards-frames)) |
| 0x06 | `CONFIG` | front-end to rank 0 | `{u32 version; u32 n_fields}`, `i64 field[n_fields]`: the settings that change while serving |
| 0x81 | `HELLO` | rank 0 to front-end | `u64 keep_positions` after `name_len` |
| 0x83 | `DONE` | rank 0 to front-end | `DoneTail {u64 epoch; u32 n_hist; u32 0}` after the `DoneHeader` when the request ran, before the error text |
| 0x84 | `PROGRESS` | rank 0 to front-end | `{u64 req_id; u32 computed; u32 total}` |
| 0x85 | `STATS` | rank 0 to front-end | `{u32 version; u32 n_fields}`, `u64 field[n_fields]` |

In the `SubmitHeader`, `flags` adds 8 IMAGES, 16 ABLIT, 32 PIXELS (the images as pixels: [media.md](media.md#the-request-frame-flag-32-pixels))
and 64 PREFIX to pico-dgx-ds4's 1 SEED, 2 KEEP and 4 LOGPROBS; with PREFIX, `n_prompt` counts the suffix (any number; the built prompt
needs one id). In the `DoneHeader`, `reason` adds 5 TOO_LARGE and 6 PREFIX_GONE; `prompt_tokens` is the whole prompt's count
(with PREFIX, `n_prefix + n_prompt`); `queue_us` includes a long prompt's wait in the queue, and `prefill_us` a long prompt's
pieces and the decode windows between them ([interleave.md](interleave.md)).

## Requests that end before they run

- A `SUBMIT` frame larger than the engine's `--frame-stage-mib` is not read into memory: its bytes are skipped, the connection
  stays up, and the request ends with `DONE{reason = TOO_LARGE}` and a message naming the limit, with no `TOKENS`. The front-end
  answers 413.
- A `SUBMIT` whose prompt cannot be built from its epoch and count (a request with `PREFIX`) ends with
  `DONE{reason = PREFIX_GONE}` and a one-line reason, with no `TOKENS`; nothing was done, and the front-end sends the whole
  prompt.

## The dashboard's frames

Protocol version 4. These are for the dashboard (`/dashboard`, [dashboard.md](dashboard.md)) and cost the
engine nothing while nobody watches.

- `PROGRESS` goes out after each prefill chunk of a request, the last chunk included. `total` is the
  number of prompt tokens this request has to compute (the prompt without the positions reused from a
  session or restored from the disk tier), `computed` how many of them are done. The front-end takes the
  change in `computed` as the prefill rate. A request that reuses the whole prompt sends none.
- `STATS_REQ` asks for one `STATS`. The front-end sends it once a second while the dashboard has been requested in
  the last 10 s, and never otherwise. `STATS` is `u32 version` (2), `u32 n_fields` (100), then `n_fields` u64 in this
  order. The chunk pool ([engine.md](engine.md#the-chunk-pool); rank 0's counts, every rank's being the same):
  `pool_numbers`, `pool_free`, `pool_held` (named by a conversation's caches or bookmarks), `pool_shared` (named by
  more than one: shared beginnings), `pool_reserved` (kept for the running responses' horizons, not yet held),
  `number_bytes` (one number on each rank), `pool_positions` (positions of conversation the whole pool holds: 5
  numbers every 16,384); the conversations: `conv_on_lines`, `conv_in_memory` (idle, in the pool), `conv_on_disk`,
  `conv_sharing` (naming a shared number); `lines`; the disk tier: `disk_cap_bytes` (0: no disk tier),
  `disk_used_bytes`, `spills_total`, `spill_bytes_total`, `spills_pending`, `restores_total`, `restore_bytes_total`,
  `restored_tokens_total`, `restore_wait_ns_total`, `restore_wait_max_ns`; then interleaved prefill's
  ([interleave.md](interleave.md)): `il_on`, `il_whole_rows`, `il_push_rows`, `il_gap_ms`, `il_share`,
  `il_burst_max_ms` (the settings it runs), `il_reading` (long prompts being processed, 0 to 2) and `il_queued` (long
  prompts waiting); then `keep_days` and `extend` (CONFIG's; 0 or 1), `ctx_limit` (the longest context a request may
  use at the time of the frame: the start window, at most 1,048,576 with `extend` 0) and `ranks` (the Sparks the
  engine runs on, each holding the same pool and disk tier); `image_numbers` (numbers the running requests' image rows
  hold, 204 rows each, freed when each request ends: [vision.md](vision.md)); then the Engram row cache, a borrower of
  the pool's memory (rank 0's; its size is every rank's): `engram_cache_bytes` (its slots, the display reservation's
  included), `engram_groups` (groups of the pool its block holds), `engram_pending_mib` (the size a change not yet landed
  goes to; 0 none), `engram_lookups`, `engram_hits`, `engram_lend_ok_total`, `engram_lend_failed_total` (a failed change
  leaves the old size), `lent_owed_groups` (groups given back that the pool could not yet make into numbers),
  `engram_huge_pct` (huge pages in its block at the last change's end) and `engram_evictions`; then the store's sizing
  rule and the memory watch (rank 0's): `store_numbers` (the store's size at the start, a whole number of groups of 64),
  `mem_avail_min_bytes` (the smallest of the four ranks' MemAvailable at the watch's last reading, every 10 s; 0 none yet),
  `mem_avail_low_count` (new lows under the memory `--pool-free-gib` leaves to the host, reported since the start),
  `ids_host_bytes` (the server's token ids held in memory) and `ids_file_sessions` (conversations whose ids are in an ids
  file under the disk directory); then `mem_avail_bytes[4]` (each rank's MemAvailable at that watch reading, rank order); then the tenants' groups (`tenant_groups[0..7]`, `tenant_models_groups`: [store.md](store.md)), then the split cache and the imager: `kv_split`, `conv_split`, `quarter_slots`, `quarter_numbers`, `ghost_numbers`, `split_pieces_total`, `split_deferred_total`, `revives_total`, `joins_total`, `join_bytes_total`, `join_ns_total`, `join_max_ns`, `lend_busy_total`, `join_short_total`, `split_level`, `split_pressure_pm`, `split_age_now_s`, `split_demote_pps`, `l2_numbers`, `l2_dropped_total`, `clean_numbers`, `image_pending`, `split_rides_total`, `returns_read_total`, `image_waits_total` and `split_reader_waits_total`, then copy drafting's learning (`DSGB10_COPY_ADAPT`; rank 0's counts, every Spark learns the same): `copy_adapt` (1 when the engine learns, else 0 and the rest are 0), `copy_proposals_total`, `copy_scored_total`, `copy_first_right_total`, `copy_rows_verified_total`, `copy_rows_accepted_total`, `copy_chance_pm[4]` (the chance a copied token is right after a match of 8 to 15, 16 to 31, 32 to 63 and 64 or more tokens, in thousandths) and `copy_failing_requests_total` (requests that ended with 16 or more proposals scored and the first copied token right in under half of them) (the order is `frontend/src/protocol.rs`, `EngineStats::fields`). Counters count from the engine's start. A front-end
  reads as many fields as it knows and takes missing ones as 0; within a version fields are only ever appended, and a
  front-end reads only its own version. An `il_share` of 0 means an engine without interleaved prefill.
- `HELLO`'s `keep_positions` (version 4) is what the engine can keep of conversations: its pool and its disk tier, 5
  numbers every 16,384 positions. The front-end's session table keeps conversations up to that many positions (at
  least 16M ids and 256 sessions), so a conversation is matched for as long as the engine holds its state.
- `CONFIG` changes interleaved prefill's settings and the disk tier's while the engine serves: `u32 version` (3),
  `u32 n_fields` (10), then `n_fields` i64 in this order: `on` (0 or 1), `whole_rows`, `push_rows`, `gap_ms`, `share`,
  `burst_max_ms`, `ssd_gib` (the disk tier's size, at most the file made at the engine's start, applied when no
  request is in flight: smaller drops the images past the new end, every rank alike), `keep_days` (a conversation
  unused this long is dropped), `extend` (0 or 1: whether a conversation may grow past 1,048,576 positions, see
  "Context past 1M"), `engram_mib` (the Engram row cache's size in MiB, the display reservation's included; groups of
  the pool are handed to it or taken back at a run boundary, and a size larger than the pool can lend is cut). A frame
  of 9 fields, of version 2 (8 fields) or 1 is accepted. The engine's struct has 15 fields: after `engram_mib`, four reserved
  for a per-Spark Engram size (not read) and one read only by fault builds; the front-end sends 10. A field below 0, or one not sent, keeps its
  value; the engine checks each against its flag's range and ignores one outside it, applies the rest before its next
  scheduling decision, and the next `STATS` shows what it runs. The settings last until the engine restarts. The
  front-end sends it only for a signed-in dashboard's change ([dashboard.md](dashboard.md)); an engine without it
  skips the frame.

## The ablit flag

`SubmitHeader.flags & ABLIT` asks for the abliterated weights path for this request; without it the request
runs the plain path. The flag is the request's, not the connection's: both kinds of request share one
connection and one batch. A session belongs to one flag: the front-end names a session in a `SUBMIT`
(`session_id`, `base_session_id`) only for requests of the flag the session was created with, so rank 0
never has to continue or fork across the two (a prompt whose text matches a session of the other flag
starts a new session, cold). `DONE` and the rest of the protocol do not change with the flag.

## Context past 1M

A request runs at the YaRN factor its prompt needs (16, 32 or 64;
[engine.md](engine.md#context-past-1m-a-yarn-factor-per-request)). With `CONFIG` / `STATS` `extend` 1 (the start flag
`--extend on|off`, on by default) a conversation grows past 1,048,576 positions up to `--max-ctx`, and at 1,048,576
and 2,097,152 it is processed again at the next factor, the prompt's turn or the response that reached the window alike;
the front-end's frames are the same at every factor. With `extend` 0 the window is 1,048,576: a longer prompt gets
`DONE` with an error (`a prompt of N tokens does not fit the context of 1048576`) and no prefill, and a response that
reaches the window ends with reason `context`. The setting is read when a request is admitted. `HELLO`'s
`max_context` is the engine's `--max-ctx`, up to 4,194,304, whatever `extend` is.

## A request with only its suffix

This belongs to protocol version 5 (`kVersionPrefix`). The front-end says its version in `HELLO`; the engine accepts 4 and 5. The engine's own
`HELLO` goes out at the connect, in version 4; on a front-end `HELLO` of version 5 it sends a second `HELLO` answering 5. On a connection of
version 4 nothing changes: `DONE` is the 56 bytes and the error text, and a `SUBMIT` with `PREFIX` is a malformed frame (the engine closes
the connection, as it does for any malformed frame). On a connection of version 5 every `DONE` of a request that ran carries the 16-byte
block, and `PREFIX` is allowed.

The engine keeps each kept conversation's token ids (rank 0's memory, and an ids file when they pass `--ids-host-mib`), so a request
need not send them. With `flags & PREFIX` the `SubmitHeader` is followed by `{u64 epoch; u32 n_prefix; u32 0}` (before the stop ids), `n_prompt`
counts the suffix, and the engine builds the prompt once, when the request first gets past the wait for its session's request in
flight: ids `[0, n_prefix)` of the history entry, then the suffix. The entry is the session's, or with a `base_session_id` the base's
(a fork: the epoch and the count are the base's), at the factor of the built prompt (a conversation has one entry per factor, each with
its own history). The frame is `72 + 16 + 4 * (n_stop_ids + n_prompt)` bytes (plus an image part, which is unchanged).

**The front-end's side** (`frontend/src/session.rs`). dsfront keeps, per session, no text and no ids: it keeps cuts, the places where
tokenisation restarts (the edges of added tokens), each as the bytes of text before it, the ids before it and a hash of that text, with
the epoch and history size of the last `DONE` of the session. A request is rendered, hashed in one pass outside the table's lock, and
matched to the session whose latest cut it shares (the index finds the candidates by the hash of the text's first 64, 256, 1,024 ...
bytes); only the text after that cut is tokenised and sent, with the session's epoch and the cut's id count. At its `DONE` the session's
cuts above the cut the request continued from are dropped (the engine has rewritten what they described), then the cuts of the suffix and
of the response are added, those whose ids are in the history the `DONE` reported. A request ends without a `DONE` (its client left): the
session is left as it was; a session that never finished a turn is released. A `DONE` with reason `PREFIX_GONE` has run nothing: the
front-end tokenises the whole prompt and sends it again with no prefix, and the session's cuts come from that request. A request with
images, a session with no epoch (a table read from an older file, a connection of version 4) and a prompt with no ids before its first cut
carry their whole prompt. `--no-prefix` says version 4 to the engine, so that every request does.

Every `DONE` of a request that ran (not one cancelled in the queue, rejected, too large or prefix-gone) carries, directly after its
56 bytes and before any error text, `DoneTail {u64 epoch; u32 n_hist; u32 0}`: the entry's epoch and the positions it reports. A reader
finds it when the frame is `56 + 16 + error_len` bytes. Epoch 0 and `n_hist` 0 say the session was not kept (no `KEEP`, or released).

The epoch belongs to an entry. It becomes a new random value when, and only when, the history loses or rewrites a position below
`n_hist` of the last `DONE`: when a request restarts the session below it, and when a run's history is cut below it. Appends never
change it, and a cut of the stored state (a snapshot, a read-back) does not: the ids are still true. A restart keeps each entry's
epoch and `n_hist` (the kept index). The request is `PREFIX_GONE` when there is no such entry, the epoch differs, `n_prefix` is more
than the entry reported, or its ids are in a file that cannot be read. Rejected drafts and the last sampled token of a response are not in
the history, so never in a built prompt.

## Requests with images

A request with images (`FLAG_IMAGES`, [vision.md](vision.md)) is matched to a session as a text request is (pico-dgx-ds4's
[protocol.md](../ds41f-4xgx10-switchless/docs/protocol.md#sessions-and-exact-prefix-reuse)), with `KEEP`, and may name a
base. Its text writes each image's placeholder once per row, so the text spells the ids, and every image row has
the same id. The engine's LCP is by id alone and cannot tell two images apart, so the front-end only names a
session whose images inside the text it shares with the prompt are the prompt's own (same place, rows and content
hash); an image that differs rules the session out. The SUBMIT always carries every image's span and rows, the
reused part's too: the engine may restart from a state inside an image's span and injects the rows from there on.
