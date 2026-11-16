# The dashboard

`dsfront serve` serves a status page at `/dashboard` on its own listen addresses. The ENGINE strip across the top holds nine
tiles, for all the engine's Sparks together: FIRST TOKEN, CONTEXT CACHE, MAX CONTEXT, CONVERSATIONS, SHARED PREFIXES, ENGRAM
CACHE, LENT OUT, SSD CACHE and CACHE HITS ([below](#the-tiles-and-the-cache-chart)); LINES (the requests
in flight) sits beside the DECODE and PREFILL charts of the last 15 minutes and the CACHE chart; NODE HEALTH is a table, one row a Spark;
ENGINE SETTINGS (interleaved prefill's settings, [interleave.md](interleave.md), and the SSD tier's size and
expiry, which it can change) sits beside COOLING AND CLOCK, there with `--fan-tool`: the fan profile and GPU clock limit of the Sparks, which it
can change ([below](#the-fans)). The header shows the decode and prefill rates and names the served models: `{name}`
runs on the plain weights path and `{name}-ablit` on the abliterated one, by name only
([protocol.md](protocol.md#the-ablit-flag)).

**Sign-in.** The routes below take the front-end's API key (`--api-key-file`), as `/v1` does: `Authorization: Bearer
KEY`, compared in constant time; without it they answer 401. The page itself loads without the key and asks for it
once; the browser keeps it (localStorage) until SIGN OUT in the header, with no time limit, and sends it with every
call. A key that stops working (401) brings the sign-in back. A front-end without `--api-key-file` leaves the routes
open. The routes answer only requests whose `Host` is `localhost`, one of the listen addresses or a name given with
`--dashboard-host` (a proxy's), so a page that points a name of its own at this address gets 403.

The page polls the status and the requests every second, the series every 5 s and the memory ledger every 30 s.
While nothing has asked for a dashboard route in the last 10 s the front-end asks the engine for no `STATS`
and polls no sampler.

## Routes (`/dashboard/api/v1/...`, JSON, no caching)

| route | answer |
|---|---|
| `GET status` | `{at_ms, model_name, ablit: {alias}, engine: {connected, connected_s, name, uptime_s}, health, sessions, interleave, node_health, engram, memory, copy, cache}` |
| `GET requests` | `{in_flight: [{id, state, prompt_tokens, generated_tokens, key_label, age_s, kv_tokens, computed, to_compute, cached_tokens, endpoint, priority, ablit}], recent: [{ttft_s, prompt_tokens, cached_tokens, ablit}]}` |
| `GET series?name=decode_tps\|prefill_tps&seconds=900&step=10` | `{points, peaks, step_s, last_ms}` |
| `GET series?name=cache&seconds=1800&step=10` | the CACHE chart's buckets: `{points, step_s, last_ms, before}`, a point per bucket holding the prompt tokens that arrived in it split into memory, SSD and computed, the SSD pieces read and the time requests waited, and `before` the buckets ahead of the first point, so the first running rate has its whole window |
| `GET ledger` | `{ranks: [{rank, device_bytes}], entries: [{rank, feature_id: weights\|store\|work, device_bytes}]}` |
| `GET settings` | `{on, whole_rows, push_rows, gap_ms, share, burst_max_ms, reading, queued, ssd_gib, keep_days, extend, engram_mib}`: the settings the engine runs and its long prompts being processed and waiting (from `STATS`); null before the first `STATS` or from an engine without interleaved prefill |
| `POST settings` | a JSON object of the settings to change, e.g. `{"share": 60}`: `on` (true or false), `share` (10 to 90), `gap_ms` (100 to 60,000), `burst_max_ms` (50 to 60,000), `whole_rows` (8,192 to the context; the engine also ignores one below its `--prefill-rows`), `push_rows` (0 to the context), `ssd_gib` (1 to 131,071; the engine takes at most its disk tier's size at its start), `keep_days` (1 to 3,650), `extend` (true or false: whether a conversation may grow past 1,048,576 tokens, read again at 1M and 2M), `engram_mib` (0 to 131,072: the Engram row cache's size in MiB, the display reservation's included; the engine clamps it to its `--engram-cache-max-mib`, applies it at a run boundary and cuts a size the pool cannot lend; the page offers 2,032, 4,096 and 6,912). Anything else, an empty object, or a value out of range is refused with 400 and nothing is sent; else one `CONFIG` frame goes to the engine and the answer is `{sent: ...}`; `GET settings` shows the change within a second. The body goes as `Content-Type: application/json`; any other type is 415, so a page on another site cannot send a change |
| `GET fans` | `{at_ms, tool, profiles: [{id, label, group}], wanted, mixed, clocks, clock, clock_mixed, pinned, nodes: [{rank, host, ok, stale, fallback, profile, requested, profile_error, state, floor_rpm, hottest_c, rpm: [fan1, fan2], guard, gpu_clock: {limit, at, ok, error} \| null, age_s, error}]}`: what each Spark's fan daemon reports, through the fan tool. `wanted` is the profile all four Sparks are asked for (null when they differ), `mixed` is true when the daemons run different profiles, `clocks` are the GPU clock limits on offer, `clock` is the limit all reachable Sparks recorded (null when they differ; `clock_mixed` says so), `pinned` is the reason a timing window holds the fans (null when free). A node is `stale` when its reading is older than 10 s or came from the kernel because the daemon is not answering. Without `--fan-tool`: `{tool: false, profiles: [], nodes: []}`. 502 when the tool fails |
| `POST fans` | exactly one key: `{"profile": "<id>"}` with an id of `profiles` (`stock`, `medium`, `performance` (the page's default), `max`, `hold-4500`, `hold-6300`, `hold-9000`), or `{"gpu_clock": "<MHz>"}` with a string of `clocks` (`2400`, `2200`, `2000`, `1800`; never `unlocked`, a number, or any other value); two keys, any other key or a name outside the lists is 400 with the lists. 200 `{sent: {profile} or {gpu_clock}, nodes: [{rank, ok, error}]}` when at least one Spark took it (a Spark that did not is named), 409 when the fans are pinned (the reason in the message), 502 when no Spark took it or the tool failed, 503 without `--fan-tool`. The key and the Host check are the same as every route's, the JSON content type as for `POST settings` |

- `status.health`: `{max_batch_size` (HELLO `max_sessions`: an advisory count, not the lines; the page draws one
  row per line, `status.sessions.lines`)`, live: {connections, waiting, decode_tokens_per_second,
  prefill_tokens_per_second, prompt_pass: {computed, total, seconds} | null}, spill: {state: on|off|unknown},
  prompt_cache: {spill: [{stored_bytes, cap_bytes, points, writes_committed, written_bytes, writes_pending}],
  restores_begun, disk_hits, restored_tokens, restore_waits, restore_wait_ns, restore_wait_max_ns}}`.
  `connections` is the requests in flight, `waiting` those that have not started decoding. The decode rate is the token ids
  received over the last 3 s, the prefill rate the `PROGRESS` change over the same window. `prompt_pass` is the
  oldest request that has reported prefill progress (`seconds` since it was submitted). `prompt_cache` is
  null, and `spill.state` `unknown`, until the first `STATS` arrives.
- `status.sessions` (null until the first `STATS`): the chunk pool every conversation's caches live in, and the
  conversations. `lines` = `lines`, `usable_window` = `STATS` `ctx_limit` (the longest context a request may use at that
  moment: it follows `extend`; HELLO `max_context` from an engine that does not send it), `ranks` = `STATS` `ranks`;
  `pool: {numbers, free, held, shared, reserved, number_bytes, positions}` = `STATS`' `pool_*` and `number_bytes`;
  `conversations: {on_lines, in_memory, on_disk, sharing}` = `STATS`' `conv_*`. The ENGINE strip's tiles are described [below](#the-tiles-and-the-cache-chart). The header shows `live`'s decode
  and prefill rates; the charts draw their history. NODE HEALTH's memory split counts the whole pool as its KV part (the
  ledger's `store` is only the store's share of it).
- SSD tier from `STATS`: `stored_bytes` = `disk_used_bytes`, `cap_bytes` = `disk_cap_bytes`, `points` =
  `conv_on_disk`, `writes_committed` = `spills_total`, `written_bytes` = `spill_bytes_total`,
  `writes_pending` = `spills_pending`, `restores_begun` = `disk_hits` = `restore_waits` = `restores_total`,
  `restored_tokens` = `restored_tokens_total`, `restore_wait_ns` = `restore_wait_ns_total`,
  `restore_wait_max_ns` as sent. `spill.state` is `on` when `disk_cap_bytes > 0`.
- A request's `state` is `queued` until the first `PROGRESS`, `prefill` until the first token, then
  `decode`. `computed` and `to_compute` are its last `PROGRESS` (0 before one); `cached_tokens` is
  `prompt_tokens - to_compute` once a `PROGRESS` came, all of `prompt_tokens` when it decodes without one (the whole
  prompt reused), null while it is queued. LINES shows each request's own state, progress and prefill rate: two long
  prompts can be processed at once. `key_label` is the request's `user` field (32 printable characters at most), else `api`.
  `kv_tokens` is the prompt plus output while decoding, the tokens computed so far while prefilling, 0
  while queued. `endpoint` is `/v1/chat/completions`, `priority` `normal`. `recent` holds the last 200 requests
  that did not end in an error, newest first; `ttft_s = (queue_us + prefill_us) / 1e6` from `DONE`.
- Series: 90 buckets of 10 s with the mean and peak of the once-a-second readings; `step` is rounded to a
  multiple of 10 s; a point with no reading is null; `last_ms` is the end of the newest bucket.
- `status.interleave`: the same object as `GET settings`, with `ssd_gib` (the disk tier's current size), `keep_days`, `extend` (true or false) and `engram_mib` (the Engram cache's size now) (the ENGINE SETTINGS box reads it with the status).
- `status.engram` (null from an engine that sends no Engram fields): `{cache_bytes, size_mib, groups, pending_mib (the size a change under way goes to, 0 none), lookups, hits, hit_rate_1m (hits over lookups in the last 60 s of STATS, null without lookups), evictions, lend_ok, lend_failed, owed_groups, huge_pct}`; `status.sessions.pool.image_numbers` is the chunks the running requests' images hold. The ENGRAM CACHE tile of the ENGINE strip reads them.
- `status.copy` (null when the engine does not learn from its copy drafts: copy drafting off, or `DSGB10_COPY_ADAPT=0`): `{proposals, scored, first_right, rows_verified, rows_accepted, chance_pm (four numbers in thousandths: the chance a copied token is right after a match of 8, 16, 32 and 64 tokens), failing_requests}`. The ENGINE SETTINGS box shows it as one line under its settings: `Copy drafts: learning from 3,188 outcomes. A copied token is right 88 %, 92 %, 94 %, 98 % of the time after a match of 8, 16, 32, 64 tokens. 91 % of copied rows accepted. 0 requests with failing copies.` The line is absent when the engine does not learn.
- `status.node_health`: `{ranks: [sample + {rank, age_ms, at_ms}], zone_names}`, one entry per `--node-stats`
  address in order; a rank whose sampler has not answered is `{rank, age_ms: null, at_ms: null}`. `age_ms` is the
  time since that sample arrived; `at_ms` is its arrival as Unix milliseconds, the same on every poll, so the page
  counts a sample once and takes the rails' rates between two samples' own times. `ledger` comes from each sample's
  `memory`.

## The tiles and the CACHE chart

The ENGINE strip is nine tiles, one a figure (a value with its unit, a short qualifier, the detail in the tile's tip), for all the
engine's Sparks together (`ranks`):

| Tile | Shows |
|---|---|
| FIRST TOKEN | the median time to first token of the `recent` requests and the slowest; they start over when the front-end restarts |
| CONTEXT CACHE | the chunks taken of the whole store (held by anything: conversations and their bookmarks, the images of running requests, the lines' own chunks, the split cache's pieces), and the tokens they hold of the tokens the pool holds |
| MAX CONTEXT | the longest context one request may use now; it follows Extend context |
| CONVERSATIONS | conversations whose cache is in the pool (running, or waiting for their next turn), and those with a piece only on the SSD cache |
| SHARED PREFIXES | the memory kept once for the beginnings conversations share, and how many share one |
| ENGRAM CACHE | the Engram row cache's size on each Spark, the groups it borrows from the store, and its hit rate over the last minute (`status.engram`) |
| LENT OUT | what the store lent to each borrower, on all the Sparks: the Engram cache, the images, the windows past 1M (`status.memory`, the engine's tenant table) |
| SSD CACHE | the saved conversation caches on each Spark's SSD, in GiB a Spark, of the size the cache may use; the four Sparks together; the bytes written since the engine started are in the tip, because they wear the drive |
| CACHE HITS | of the prompt tokens that arrived in the last 15 minutes, the share found in memory and the share read back from the SSDs (`status.cache`) |

The **CACHE chart** beside DECODE and PREFILL draws, from `series?name=cache`: three **running rates** on the left scale, in per
cent (the share of prompt tokens already in memory, the share read back from the SSDs, the share of Engram rows found). A rate
is a ratio of sums over the last 15 minutes of requests ending at its point, never a mean of buckets' shares, and a bucket with
no request carries the value before it, so a quiet engine's lines hold still. The **context cache's fill** is a level, drawn
stepped and dashed in its own colour on the same scale: per cent of the store taken, the CONTEXT CACHE tile's figure. Faint
bars on the right scale are each bucket's prompt tokens split into memory, SSD and computed. The right ends of the memory and
SSD lines are the CACHE HITS tile's two figures.

On the three charts a hover box shows the bucket's values and stays inside its plot: to the right of the hairline when it fits,
else to its left. DECODE and PREFILL show the bucket's average and peak (idle seconds count as 0); CACHE shows the three rates,
the fill with its bytes, when the last request was, the bucket's prompt tokens, and what was read from the SSDs and how long
requests waited. Tests: `frontend/tools/dashboard_smoke.py` and the front-end's cargo tests.

## The node sampler

`--node-stats HOST:PORT,HOST:PORT,...` names each rank's sampler in rank order. While the dashboard is
watched the front-end connects to each every 2 s, sends `GET\n` and reads one JSON line (at most 1 MiB, 1.5 s
in all); the sampler closes the connection. The default port for the engine's sampler is 9760. The line:

```
{"rank": 0, "mem_available_bytes": N, "mem_total_bytes": N, "zones_c": [C, ...], "zone_names": ["...", ...],
 "gpu": {"temp_c": C, "util_pct": P, "clock_mhz": N, "clock_max_mhz": N, "reasons": ["..."], "power_w": W},
 "load1": X, "cpus": N, "cpu_busy_pct": P,
 "nvme": {"temp_c": C, "wear_pct": P, "root_free_bytes": N, "root_total_bytes": N, "available_spare_pct": P,
          "spare_threshold_pct": P, "media_errors": N, "critical_warning": N, "data_units_written": N,
          "data_unit_bytes": N, "power_on_hours": N, "unsafe_shutdowns": N},
 "rails": [{"rx_bytes": N, "tx_bytes": N, "errors": N, "discards": N}, {...}],
 "memory": {"device_bytes": N, "weights_bytes": N, "work_bytes": N, "kv_bytes": N}}
```

`rails` are the two RoCE port functions in order (the page shows bits per second from the change in the
byte counters between samples). A value the host does not give may be null or 0. `dsfront mock-node-stats
--port P --rank R` serves a moving sample for tests.

## The fans

`--fan-tool PATH` names the tool the COOLING AND CLOCK box runs: `dgx-fan-fleet` (from [rbb-dev/dgx-spark-fan-control](https://github.com/rbb-dev/dgx-spark-fan-control), built on dgx-spark-fan-control by christopherowen, whose daemon sets the fans on each Spark; its `docs/fleet.md`), or any executable with the same two
commands. dsfront runs it directly, with no shell, no input and fixed arguments: `PATH status --json` (at most every 4 s
however many pages are open, and only while the dashboard is watched) and `PATH set --json PROFILE` (only for a profile
of the fixed list above) and `PATH clock set --json LIMIT` (only a limit of the list above); the request is checked before the tool is started. Each call is bounded in time and output size.
`status --json` prints `{pinned, mixed, profile, nodes: [{rank, host, ok, stale, age_s, error, status}]}`, one `status`
per Spark as that Spark's daemon wrote it (`{v, at, profile, requested, profile_error, state, floor_rpm, hottest_c,
filtered_c, rpm, zero_rpm_guard, driver}`; `fallback: true` when the Spark's daemon was not answering and the tool read the
kernel itself). `set --json` prints `{profile, pinned, nodes: [{rank, ok, error}]}` and exits 0 (every Spark took it), 1
(some did not) or 3 (pinned: nothing was written); `clock set --json` prints `{gpu_clock, pinned, nodes: [{rank, ok, error}]}` with the same exit codes. Each Spark's
`gpu_clock` in the status is the record the clock tool keeps there (`DGX_FAN_CLOCK_STATE`). The clock limit lasts until the next start (the start helper in `deploy/` sets it again from `GPU_CLOCK`) and a reboot clears it; the page says so. Installing the tool, the settings and the profiles are in [operations.md](operations.md#fans-and-the-gpu-clock); the clock tool the Sparks need is `ds41f-4xgx10-switchless/tools/gpu-clock` ([setup.md](setup.md#9-fans-and-the-gpu-clock)).

The page and this front-end never touch a fan, a sensor or ssh: the tool writes one profile name into a small file on each
Spark and each Spark's own daemon sets the fans from it. A front-end, a tool or an ssh that is down leaves every fan as
it was. `frontend/tools/fake_fan_tool.py` is the stand-in the smoke test runs. Each change is one line in the journal
(`fan: profile=... ok=N/N peer=...` or `fan: gpu_clock=... ok=N/N peer=...`; the API has one shared key, so the peer address is the identity there is). The page shows a COOLING AND CLOCK box
(a profile list and a GPU clock limit list for all four Sparks at once, what is in force, a warning when the Sparks differ or one does not answer,
and the pin's reason) and FANS columns in NODE HEALTH's table (each fan's RPM, then the profile; the temperatures stay
under TEMP.). Without `--fan-tool` both are absent.

## The engine's side

- `PROGRESS`: rank 0 sends one after each prefill chunk of a request's first run: the prompt tokens it has computed
  and those it computes in all (after the session's reuse).
- `STATS`: rank 0's main loop, the only thread that touches the model, publishes a frame at most every half second
  (Model::tier_stats: the store's capacity and use, its sessions in memory and on disk, the disk tier's file, and the
  sessions written to it and read back since the start, with the positions restored and the time requests waited for
  them; the slots and their contexts). The socket's reader answers each `StatsReq` with the latest one at once.
- The node sampler (server/node_stats.h): every rank in serve mode listens on its first link address (the fleet's
  `NEXT_IP0`), port `--node-stats-port` (9760; 0: none), asleep in `accept` until asked, on the small cores. It reads
  `/proc/meminfo`, `/proc/loadavg`, `/proc/stat`, the thermal zones, NVML (by `dlopen`), the NVMe's SMART page (the
  hwmon temperature when the ioctl is refused, as it is without root), `statvfs("/")` and the rails' InfiniBand
  counters (the fleet's `NEXT_DEV0` / `NEXT_DEV1`). `memory`: the device's total, the rank's weight file, the session
  store, and the rest of what was in use when serving began. For four ranks:
  `--node-stats ADDR0:9760,ADDR1:9760,ADDR2:9760,ADDR3:9760` on the front-end.

## Tests without an engine

`frontend/tools/dashboard_smoke.py` starts `dsfront mock-engine`, four `mock-node-stats` and `dsfront serve`, and checks
every route, the page, the two model names (and that no switch route exists), a request's states, STATS only while
watched, the FANS routes through `frontend/tools/fake_fan_tool.py` (the key, every profile set and read back, the refusals, a pin, a Spark
down, no tool, no engine), and the stream's timing with and without the page polling.
