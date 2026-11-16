# Operating it

Running the four-Spark engine day to day: start, stop, restart, status, upgrade, rollback, logs, what the engine's own log lines mean,
and the fans and the clock. [`setup.md`](setup.md) builds the installation this page runs. The units and settings are in
[`deploy/`](../deploy/README.md). Commands run on rank 0 as the login that owns `~/dsgb10`.

## Start, stop, restart, status

Three user units: `pico-engine` (rank 0, and ranks 1-3 on the workers through `pico-ctl.sh`), `pico-front` (`dsfront`: the API and the
dashboard) and `pico-media` (the media worker).

| To | Command | What happens |
|---|---|---|
| start everything | `systemctl --user start pico-media pico-engine pico-front` | `prepare` sets the GPU clock (`tools/gpu-clock`, in the `ds41f-4xgx10-switchless` folder), copies the engine, `fleet.conf`, `cfmlp_cml.bin` and `ablit.bin` to the workers, drops the page cache of every model file and then every Spark's clean cache, checks `MIN_AVAIL_GIB`, starts ranks 1-3; rank 0 loads; `ready` waits for its socket |
| stop the engine | `systemctl --user stop pico-engine` | rank 0 ends the request in flight, writes the kept sessions (up to `KEEP_GIB`) and stops ranks 1-3; `reap` ends what is left on the workers. `pico-front` stays up and holds new requests for `HOLD_S`, then answers 503 with `Retry-After` |
| restart the engine | `systemctl --user restart pico-engine` | a stop and a start; the kept sessions come back if the binary, the weights and the engine arguments are the same |
| restart the front-end alone | `systemctl --user restart pico-front` | the engine is untouched; the session table (`run/front.table`) keeps the conversations |
| status | `~/dsgb10/bin/pico-ctl.sh status`, `curl -s http://127.0.0.1:8000/health` | the units' state; `/health` answers `{"status": "ok"}` without the key and whole with it |
| stop everything | `systemctl --user stop pico-front pico-engine pico-media` | |

A start takes a few minutes (`journalctl --user -u pico-engine -f`). `pico-engine` restarts itself with a growing delay (10 s, up to
an hour) when rank 0 dies; the delay clears after a start that served. A failed start prints the reason in the journal (the
table below). To keep the units across a reboot: `systemctl --user enable ...` and `sudo loginctl enable-linger <user>` ([setup.md](setup.md#1-what-you-need)).

**Without systemd.** `engine/server/service.sh start | stop | status | restart | forget` runs the same engine under a supervisor
process, with its settings in the environment (the script's header lists them), and the logs in `~/dsgb10/service/logs`. Use one
way on a set of Sparks.

## The ring lend

Development jobs that need all four Sparks (`engine/server/run4.sh`, `ring.sh`, tests, benchmarks) take the ring lock
(`RING_LOCK` in the fleet file). With `service.sh`, the service lends the ring to a waiting job once no request has come for
`DSGB10_YIELD_IDLE_S` (600 s) and takes it back when the job ends; a request that arrives during a lend stops the job unless it ran with
`PREEMPT=0` (pico-dgx-ds4's [engine.md, Borrowing the ring](../ds41f-4xgx10-switchless/docs/engine.md#borrowing-the-ring)). **Under the systemd units nothing lends:** stop
`pico-engine` (the units hold no lock), run the job, and start it again. A job that runs while the engine serves finds the GPUs and
memory taken and fails its start.

## Upgrade

1. On rank 0 in the repository: update the sources, rebuild (`cmake --build engine/build -j4`; `(cd frontend && cargo build --release -j4)`).
2. Keep what runs: `cp ~/dsgb10/bin/dsgb10_rank ~/dsgb10/bin/dsgb10_rank.prev; cp ~/dsgb10/bin/dsfront ~/dsgb10/bin/dsfront.prev; cp ~/dsgb10/pico.env ~/dsgb10/pico.env.prev`.
3. Put the rebuilt binaries and units in place: `../deploy/pico-ctl.sh install "$PWD"` (from `ds41f-4xgx10-switchless`). It enables and starts nothing.
4. A front-end change alone: `systemctl --user restart pico-front`. An engine or settings change: `systemctl --user restart pico-engine`
   (the front-end holds requests meanwhile). The first start copies the rebuilt engine to the workers.
5. Check the start lines, `/health`, one request, and `MemAvailable` on every Spark (more than `--pool-free-gib` at an idle
   engine: [store.md](store.md#what-stays-free)).

What an upgrade changes for stored data:

- **Kept sessions belong to the engine that wrote them.** A rebuilt engine binary, other weights or other engine arguments read none of
  them: the conversations come back cold once, with correct output (`not a kept index` in the log is this, not damage). The
  front-end's session table is kept across the front-end's restart and re-sends whole prompts when the engine reports `PREFIX_GONE`.
- **Rank files have a version.** An engine refuses a file of another version; cut the files again with `dsw_convert` ([setup.md, 5.1](setup.md#5-the-files-each-spark-needs)),
  and the vision files with `dsw_convert --vision`. Engram shards and the hash map change only when the checkpoint does.
- **A changed media worker** needs another image tag in `MEDIA_IMAGE` and `systemctl --user restart pico-media`.
- **An added or changed engine argument or `DSGB10_*` setting** goes in `pico.env`; change the Spark-wide files (`copy-T.txt`, `ablit.bin`) on every Spark
  at the same path before the restart.

## Rollback

Restoring the three kept copies (step 2 of the upgrade) rolls back:

```bash
systemctl --user stop pico-front pico-engine
cp ~/dsgb10/bin/dsgb10_rank.prev ~/dsgb10/bin/dsgb10_rank
cp ~/dsgb10/bin/dsfront.prev ~/dsgb10/bin/dsfront
cp ~/dsgb10/pico.env.prev ~/dsgb10/pico.env
systemctl --user start pico-engine pico-front
```

`prepare` copies rank 0's engine binary to the workers at every start, so rank 0's copy is the only one to restore. The kept
sessions are read only if the binary that starts wrote them with the same settings. Rank files of another version need the engine
that reads that version, or cutting again.

## Logs

| Where | What |
|---|---|
| `journalctl --user -u pico-engine` | rank 0's output: the start tables, `chunk pool:`, `memory:` lines, `serve:` lines, `ready after N s` |
| `journalctl --user -u pico-front` | the front-end: requests refused or held, fan and clock changes |
| `~/dsgb10/logs/rank.<r>.log` (ranks 1-3, on each worker) | that rank's output; each start moves the last log to `rank.<r>.<end time>.log` and keeps the last 20 |
| `docker logs pico-media` | the media worker |
| `~/dsgb10/service/logs` | only with `service.sh`: `service.log`, `rank.0.log`, `front.log`, rotated |

A crashed rank's log is the one to read first: rank 0 reports `socket read: peer closed` when a worker died.

## What a log line means

| Line (or answer) | Meaning | What to do |
|---|---|---|
| `ready after N s`, `serve: listening on ...` | the engine serves | nothing |
| `rank N: the store's kernels ran once` | that rank warmed its kernels at the start | a missing line on a rank means that rank did not finish its start; read its log |
| `memory: ...` (rank 0, at each new low) | a Spark's available memory went below `--pool-free-gib` while serving | find what else uses memory on that Spark; raise `--pool-free-gib` to keep more free. A low under 2 GiB is the warning to act on |
| `the start is refused: ... more than its cap` | one of the engine's own memory lines already holds more than its cap | raise the cap (`--front-mib`, `--media-mib`, `--ids-host-mib`) or stop what holds it |
| `the start is refused: the store of N numbers cannot hold one conversation of --max-ctx ...` | not enough memory for the window, the Engram cache and the largest image request together | lower `--max-ctx`, `--engram-cache-mib` or `--slots`, or free memory |
| `serve: BUG: every flight waits for an extension` | every request that holds memory waits to grow and nothing can be freed | the engine ends the request with the longest output so far (`serve: req N ends at T tokens (length)...`) and carries on; a store that small needs more `--pool-free-gib` headroom or fewer lines |
| `serve: BUG: a read-back the count paid for was refused` | a bookkeeping disagreement in the store | the engine carries on; keep the log and report it |
| `not a kept index`, a start with empty sessions | the kept sessions were written by another binary, weights or arguments | expected after an upgrade |
| `PREFIX_GONE` (a DONE reason, engine to front-end) | the engine restarted or does not hold the prefix | the front-end sends the whole prompt once; nothing to do |
| `DONE too-large` / HTTP 413 | a request frame over `--frame-stage-mib` (400) or an image over `--front-image-mib` | send a smaller request or raise the cap (and `--pool-free-gib` follows) |
| `this engine has no image encoder` | the engine started without `--vision-weights`, or `store: work area ... failed ...; the engine has no image encoder until the next start` | add the setting and restart; free memory for the work area |
| HTTP 429 with `Retry-After` on an image | rows of images in flight are over `--max-image-rows-in-flight` (80,000) | retry after the header's seconds |
| `engram row cache: display reservation not used (...)` | `--engram-vram on` could not open `/dev/dri/card0` (owner, or a screen is connected) | fix the owner, or use `--engram-vram off`; the cache runs in ordinary memory |
| `copy: DSGB10_COPY_TABLE=... cannot read it on this rank` | `copy-T.txt` is missing on that Spark | put it at the same path on every Spark |
| `copy drafting settings differ between the ranks` | a Spark has other `DSGB10_COPY*` settings, or another table file, than the others | set the same on every Spark (the engine stops at its start and names each rank's values) |
| `DSGB10_COPY_ADAPT cannot run with DSGB10_RELAX` | `DSGB10_COPY_ADAPT=1` is set by name beside a relaxed accept | unset `DSGB10_RELAX` (and `DSGB10_RELAX_P`), or `DSGB10_COPY_ADAPT` (with a relaxed accept the learner is off by itself) |
| `spec: DSGB10_CF_MLP=...: N words, not M` | the confidence network file is not the shipped `cfmlp_cml.bin` | reinstall it from `tools/stack4/` |
| `ablit: <path>: <reason>` | the abliteration file is unreadable or not made for this checkpoint | remake it with `tools/ablit/make_ablit.py` |
| `comm: cannot read the fleet file ...`, `comm: this host, X, is no RANK<r>_HOSTNAME`, `comm: on <dev>, ...'s GID ...` | the fleet file is missing, a hostname differs, or a rail's GID does not match the file's address | fix the fleet file ([setup.md, 2](setup.md#2-the-network)) |
| `pico-ctl: MemFree under N GiB after the cache drop on <spark>` | another process holds memory on that Spark | stop it; `pgrep -x dsgb10_rank` for a stray rank |
| HTTP 503 with `Retry-After` | the engine was down longer than `HOLD_S` | wait; check the engine's journal |
| HTTP 401 | no key, or a different key | send `Authorization: Bearer <key>` |
| a response ends with `length` far below the request's `max_tokens` | the context limit was reached, or the standstill line above | raise `--max-ctx` or lower the prompt |


## Fans and the GPU clock

**The clock.** Every start sets `GPU_CLOCK` (2,200 MHz in the example) on all four Sparks through `tools/gpu-clock` (in the `ds41f-4xgx10-switchless` folder; `GPU_CLOCK_TOOL`):
`gpu-clock LIMIT [ssh-target ...]` locks this Spark and each target to 2,400, 2,200, 2,000 or 1,800 MHz (never unlocked) and records
`{"limit","at","ok","error"}` in `GPU_CLOCK_STATE` on the Spark it runs on (the targets are set, not recorded; at a start that is rank 0). The dashboard's COOLING AND CLOCK box runs the same tool through `dgx-fan-fleet`
and changes the limit live until the next start, which applies `pico.env`'s value again; `dgx-fan-fleet clock set 2000` does the same from a
shell. A reboot clears a lock. Check on every Spark with `nvidia-smi --query-gpu=clocks.sm --format=csv` while it works. The sudo lines the
tool and the helper need are in [setup.md, section 7](setup.md#7-sudo-the-services-and-the-ring-check).

**The fans.** One profile applies to the fans of all four Sparks: `stock` (the machine's own controller alone), `medium`, `performance` (the
dashboard's default), `max`, or `hold-4500`, `hold-6300`, `hold-9000` (a held floor that rises with heat). No profile cools less than stock.
The dgx-spark-fan-control project's service on each Spark reads a profile name from `/var/lib/dgx-fan-control/profile` every two seconds;
`dgx-fan-fleet` on rank 0 writes that file on each Spark over ssh:

```bash
dgx-fan-fleet status                        # each Spark's profile, state, both fans' RPM, hottest zone
dgx-fan-fleet set performance               # all four
dgx-fan-fleet check performance             # exit 0 only if every Spark runs it, with fresh readings (for scripts)
dgx-fan-fleet pin max --why "timed run"     # refuse every change until unpin; the dashboard refuses too
dgx-fan-fleet unpin
```

The fan tool is [rbb-dev/dgx-spark-fan-control](https://github.com/rbb-dev/dgx-spark-fan-control), built on dgx-spark-fan-control by
christopherowen (its daemon sets the fans on each Spark); its `docs/installation.md` and `docs/fleet.md` cover the install and the tool's
contract. If the front-end, the tool or ssh fails, every fan stays as it was; if a Spark's fan service stops
(`sudo systemctl disable --now dgx-fan-control`), the Spark's own controller takes over. Without `--fan-tool` the dashboard has no COOLING AND
CLOCK box. The dashboard's side (the routes, what the box shows) is in [dashboard.md](dashboard.md#the-fans).
