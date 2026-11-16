<h1 align="center">DeepSeek-V4.1-Flash on four DGX Sparks</h1>

<p align="center"><b>One memory store, contexts that grow to 4M on their own, images and video inside the engine, a conversation cache on the SSDs, and long prompts that do not stall other users: on the pico-dgx-ds4 engine</b></p>

This repository builds on **[pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4)** by Pico (Patrick Moorhead), which builds
on [dsgx10](https://github.com/blaesus/dsgx10) by Andy Shu: DeepSeek-V4.1-Flash-1017 on four NVIDIA DGX Sparks (GB10), tensor
parallel over four, linked through their ConnectX-7 ports and served through an OpenAI-compatible front-end. The engine, its
kernels, the RoCE ring, the DSpark drafter, multi-user batching, the 1M-token context and the disk tier come from pico-dgx-ds4 and from dsgx10
([Built on](#built-on)). What this repository adds is under [What you get](#what-you-get); the code of both is in
`ds41f-4xgx10-switchless/`. The folder keeps upstream's name: "switchless" is dsgx10's cabled-ring configuration, and
the same engine runs on one switch with `DSGB10_RING_DIRECT=2`, as served here.

To run it: [What you need](#what-you-need), then [Running it](#running-it) and the step-by-step
[setup](docs/setup.md). [`README-pico.md`](README-pico.md) is pico-dgx-ds4's README; its settings and
numbers describe pico-dgx-ds4's own setup.

**Terms.** A **line** is one session slot (`--slots`): a conversation whose state stays on the GPU between turns. **The store**
is the memory the engine takes at start for conversations: one pool of equal **chunks** (the engine's start lines call them
`numbers`), from which it also lends whole groups to its other users of memory (the Engram row cache, the image encoder's work
area, the larger context windows). A **piece** is one chunk's contents. **Prompt Processing (PP)** is prefill and **Token
Generation (TG)** is decode, in t/s (tokens a second). Paths are relative to `ds41f-4xgx10-switchless/`, except `docs/`,
`tools/bench/bench.py`, `deploy/` and `network/`, which are at the repository root. [`docs/`](docs/README.md) holds this
repository's documents; pico-dgx-ds4's are in `ds41f-4xgx10-switchless/docs/`.

## What you get

A four-Spark DeepSeek-V4.1-Flash-1017 server with an OpenAI-compatible API, contexts to 4M tokens, images and video, and one
page to watch and tune it. Each item links to the section that explains it.

- **One memory store, sized at start.** You choose how much memory each Spark leaves to the host (`--pool-free-gib`). At start
  the engine sets aside the most its other users of memory may take, makes one store from the rest, the same size on all four
  Sparks, and keeps that size while it runs. Every conversation's cache, its restart points and the images' rows live in that
  store as equal chunks: forks share chunks, nothing fragments, and a running request is never taken from
  ([The memory store](#the-memory-store)).
- **Contexts that grow on their own, to 4M.** A request runs in the smallest window that holds it: 1M, 2M or 4M tokens (YaRN).
  When a conversation outgrows its window, even in the middle of a response, the engine moves it to the next window and carries
  on; the request is never refused for its length
  ([Context to 4M](#context-to-4m)).
- **Images and video inside the engine.** The model's own vision encoder runs in the engine, split over the four Sparks like
  the rest of the model. Send up to 100 images, or a video clip sampled into frames, in an ordinary chat request; a small media
  worker with no GPU and no model turns them into pixels. A conversation that sends the same images again processes only its
  new text ([Images and clips](#images-and-clips)).
- **A conversation cache on the SSDs.** Idle conversations are written to the SSDs in the background, each Spark keeping one
  quarter of each on its own drive, so their memory can be reused without losing them. A conversation that returns comes back
  from memory if nothing used its space, otherwise from the four SSDs at once. Idle conversations are also checkpointed, so after a
  crash they come back (and, with pico-dgx-ds4's kept sessions, after a planned restart of the same build)
  ([The SSD tier](#the-ssd-tier-the-split-cache-and-crash-checkpoints)).
- **Long prompts that do not stall other users.** A long prompt is processed in pieces between the decoding users' steps, so
  they keep generating while it runs ([Interleaved prefill](#interleaved-prefill)). A request that needs memory waits for it
  instead of taking it from a running one ([Requests that wait for memory](#requests-that-wait-for-memory)).
- **Long prompts processed 16,384 rows at a time.** A long cold prompt is processed 16,384 rows at a pass on a second set of
  prefill buffers that the store lends while it is free; the set goes back first when the store needs the room, and the engine
  processes 8,192 rows at a pass while it is given back ([Long prompts processed 16,384 rows at a time](#long-prompts-processed-16384-rows-at-a-time)).
- **Token ids kept once, in the engine.** The front-end sends only the new part of a prompt; the engine already holds the rest
  ([Token ids kept once](#token-ids-kept-once)).
- **The Engram row cache, resized live.** Its memory is lent by the store, it can sit in the display memory the headless Sparks
  leave unused, and its size changes from the dashboard while serving ([The Engram row cache](#the-engram-row-cache)).
- **More tokens per step.** Besides the model's own draft heads, the engine copies text the conversation already contains (an
  agent rewriting a file repeats most of it), and when several users share a step, pico-dgx-ds4's confidence network also
  decides how many of each user's drafted tokens are worth checking. Both are on in the served settings and off by default in the code ([Drafting](#drafting)).
- **Runtime abliteration.** A second model name with the `-ablit` suffix, served from the same weights: the published
  abliteration edit is applied at run time, only to the requests that ask for that name
  ([Runtime abliteration](#runtime-abliteration)).
- **Leaner weights, faster kernels.** The embedding is cut by vocabulary over the four Sparks, gathered weights load in place
  and the rotary table is computed on the fly: the engine keeps one copy of each weight, so about 2 GiB a Spark of the
  memory is the store's. A faster prefill up kernel, pico-dgx-ds4's balanced reduce-scatter (`DSGB10_RING_RSBAL`) and prefill tuning
  (`DSGB10_PF_TUNE`) switched on in the served settings, and smaller decode changes ([Weights and copies](#weights-and-copies), [Kernels](#kernels)).
- **A dashboard with fans and GPU clock.** Live speed and cache charts, every line, every Spark's health, the settings that
  change while serving, and the fan profile and GPU clock limit of all four Sparks
  ([Dashboard, fans and GPU clock](#dashboard-fans-and-gpu-clock)).
- **The network, reproducible.** The switch's configuration and each Spark's flow control at boot, with the tests that show
  no drops under load ([`NETWORK.md`](NETWORK.md)).

![The dashboard while 16 users run a coding bench and one longer request: DECODE 337 t/s with 17 of 32 lines decoding, the nine engine tiles, the DECODE, PREFILL and CACHE charts, NODE HEALTH with each Spark's memory, temperatures, fans, GPU clock and rails, the ENGINE SETTINGS and the COOLING AND CLOCK box](assets/dashboard.jpg)

## At a glance

Measured on four Dell Pro Max GB10 systems on one switch, from a clean boot, GPU clock limited to 2,200 MHz, with the settings
of [Running it](#running-it), with `tools/bench/bench.py`. One user: the median of
27 chat and 27 code requests; 16 and 32 users at once, each with a coding prompt of 16 different ones (at 32 users each prompt
twice), the median of 3 runs; 16 users sending the same 3 prompts repeated. Every request starts
with a random prefix, so no cache helps. 512 output tokens a request unless a row says otherwise. The bench's spread between
runs is about 4 to 5 %.

| | |
|---|---|
| **Model** | DeepSeek-V4.1-Flash-1017 with its vision encoder (images and video), tensor parallel over four Sparks, served as `deepseek-v4.1-flash-1017` and `deepseek-v4.1-flash-1017-ablit` |
| **One user, chat / code (TG)** | 83.3 / 146.3 t/s (on a second clean boot of the same build) |
| **16 users at once (TG)** | 354.0 t/s in total (16 different prompts), about 22 per user; 388.9 t/s with 16 copies of 3 prompts |
| **32 users at once (TG)** | 328.5 t/s in total; with 2,048-token outputs 1 / 16 / 32 streams give 78.2 / 390.3 / 328.6 t/s |
| **One cold prompt (PP)** | 28K tokens in 2.8 s (10,339 t/s), 228K in 20.6 s (11,041 t/s), 778K in 74.4 s (10,449 t/s) |
| **Long prompts beside decoders** | a cold prompt of about 170K tokens beside 4 decoders: its first token after 53.7 s, the decoders' longest silence while answering 0.6 s, short prompts' first token 0.6 s median; one of about 785K tokens: 139.9 s |
| **Agent edit sessions** | 119.6 t/s with thinking high, 121.7 with thinking off; 33 of 36 tasks passed both ways |
| **Context** | 1M, growing to 2M and 4M as a conversation needs it (`--max-ctx 4194304`) |
| **Lines** | 32 (`--slots 32`), each able to grow to the whole context |
| **The store** | one pool of equal chunks on every Spark, made once at start from the memory `--pool-free-gib` does not leave to the host ([The memory store](#the-memory-store)) |

## What you need

- **Four NVIDIA GB10 systems** (DGX Spark class; the numbers here ran on Dell Pro Max GB10, 121 GB of memory each, 4 TB of
  NVMe each: a rank takes about 90 GiB of memory, its weight file about 71.4 GiB, the SSD tier whatever `--disk-gib` allows) with
  one ConnectX-7 port each, ranks 0 to 3. Rank 0 serves the API and reaches the other three over ssh with a key.
- **One 200G switch** forwarding RoCE v2 with priority flow control, one cable to each Spark; two rails per Spark
  (`mlx5_0` and `mlx5_2`). The configuration, the Spark's addresses and flow control, and the tests: [`NETWORK.md`](NETWORK.md).
  The engine's cabled-ring mode is pico-dgx-ds4's; leave `DSGB10_RING_DIRECT=2` out on a ring.
- **The checkpoint:** [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, cut into per-rank files (version 3) with `dsw_convert`; the front-end
  needs `--encoding 1017` and the checkpoint's `tokenizer.json`. For images, each rank's `vision.rank{rank}.dsw`; for the
  `-ablit` name, the abliteration file in this repository (`ds41f-4xgx10-switchless/tools/ablit/ablit.bin`).
- **Software:** DGX OS or Ubuntu 24.04 with NVIDIA driver 580 and CUDA 13.0, a C++20 compiler (g++ 13.3), CMake 3.28 or later,
  a Rust toolchain, Python 3.12 and, for the media worker, Docker. The measured Sparks run DGX OS 7.6 with kernel 7.0 and
  driver 580.178.04 ([`NETWORK.md`](NETWORK.md#3-the-mikrotik-crs812-measured-here-from-its-factory-state)). The full list is in
  [setup, section 1](docs/setup.md).
- **The settings that matter first:** `--max-ctx`, `--slots`, `--pool-free-gib` with `--front-mib` on both the engine and the
  front-end, and the front-end's `--encoding 1017` with `--tokenizer` ([Running it](#running-it)).

## Contents

- [Running it](#running-it)
- [The memory store](#the-memory-store)
- [Long prompts processed 16,384 rows at a time](#long-prompts-processed-16384-rows-at-a-time)
- [Context to 4M](#context-to-4m)
- [The SSD tier, the split cache and crash checkpoints](#the-ssd-tier-the-split-cache-and-crash-checkpoints)
- [Images and clips](#images-and-clips)
- [The Engram row cache](#the-engram-row-cache)
- [Drafting](#drafting)
- [Interleaved prefill](#interleaved-prefill)
- [Requests that wait for memory](#requests-that-wait-for-memory)
- [Runtime abliteration](#runtime-abliteration)
- [Dashboard, fans and GPU clock](#dashboard-fans-and-gpu-clock)
- Inside the engine: [The shared KV chunk pool](#the-shared-kv-chunk-pool), [Token ids kept once](#token-ids-kept-once),
  [Weights and copies](#weights-and-copies), [Kernels](#kernels)
- [Tests and tools](#tests-and-tools)
- [pico-dgx-ds4's design records](#pico-dgx-ds4s-design-records)
- [Built on](#built-on)
- [License](#license)

## Running it

[`setup`](docs/setup.md) takes four Sparks from a clone to the settings below: the network
([`NETWORK.md`](NETWORK.md)), the build, the files each Spark needs, the settings, the services, the media worker, the fans and
the clock, the first start and its checks. [`operations`](docs/operations.md) is the guide for the running
service: start, stop, upgrade, rollback, logs. The files of a served installation are in [`deploy/`](deploy/) (the settings
file, the fleet file, the `pico-engine`, `pico-front` and `pico-media` systemd user units and `pico-ctl.sh`, which the units call)
and the GPU clock tool in [`ds41f-4xgx10-switchless/tools/gpu-clock`](ds41f-4xgx10-switchless/tools/gpu-clock). `README-pico.md` describes pico-dgx-ds4's own setup for a cabled ring; its
run block uses `DSGB10_OEA`, the lossy expert re-route that this repository's served settings leave out.

The engine runs on all four Sparks and the front-end, `dsfront`, on rank 0. Two ways start and supervise them: the units of
`deploy/` (the served way), or `engine/server/service.sh start|stop|status|restart` without systemd. Use one, never both on a set
of Sparks. `service.sh` takes the engine's arguments from `DSGB10_ENGINE_ARGS` and the front-end's more arguments from
`DSGB10_FRONT_ARGS` (the front-end's API key from the file named by `DSGB10_KEY_FILE`); it starts the engine with
`--mode serve --disk-tag service --keep-gib` (`DSGB10_KEEP_GIB`, 4 by default) and does not start the media worker. Without
`DSGB10_ENGINE_ARGS` it starts the engine with its own default, `--max-ctx 262144 --pool-free-gib 10 --front-mib 840`, so a 4M
window needs the settings below. `service.sh` lends the ring to a waiting test job after `DSGB10_YIELD_IDLE_S` seconds
(600) without a request and holds requests meanwhile (`DSGB10_HOLD_S`); its header lists every setting.
`engine/server/run4.sh` is a development launcher for a ring job (fleet file, ring lock).

### As served

The engine's arguments (`ENGINE_ARGS` of `deploy/pico.env.example`; every rank gets the same). The speeds in
[At a glance](#at-a-glance) were measured with this line. The units add `--mode serve`, `--socket`,
`--disk-tag service` and `--keep-gib` (from `KEEP_GIB`, 64 here); with `service.sh` leave `--mode serve` and `--disk-tag`
out and give the kept-session size as `DSGB10_KEEP_GIB=64`. In `pico.env` write full paths (systemd does not expand `~`;
the engine expands a leading `~/` in `--weights` itself). The GPU clock limit of 2,200 MHz is `GPU_CLOCK` in `pico.env`,
applied at every start, or set with `ds41f-4xgx10-switchless/tools/gpu-clock`.

```
--max-ctx 4194304 --slots 32 --big-slots 0 --pool-free-gib 6.3 \
  --engram-vram on --engram-cache-mib 2032 --disk-gib 512 --front-mib 840 --media-mib 512 --pscan all --kv-split tiered \
  --prefill-rows-max 16384 \
  --draft-head4 --draft-markov8 --ablit /home/<user>/dsgb10/ablit.bin --vision-weights /home/<user>/dsgb10/vision/vision.rank{rank}.dsw \
  --weights ~/dsgb10/weights/rank{rank}.dsw \
  --setenv DSGB10_COPY=1 --setenv DSGB10_BUDGET=1 --setenv DSGB10_DRAFT1=6 \
  --setenv DSGB10_WIDE=64 --setenv DSGB10_WIDE_COST=24,2.2,57,1.1 --setenv DSGB10_HEAD4_C1=1 \
  --setenv DSGB10_CF_MLP=/home/<user>/dsgb10/bin/cfmlp_cml.bin --setenv DSGB10_CF_ALPHA=1.0 --setenv DSGB10_BATCH_CF=1 \
  --setenv DSGB10_BATCH_TRIM=1 --setenv DSGB10_BATCH_TRIM_COST=19,2.9 --setenv DSGB10_CERT=1 --setenv DSGB10_RING_DIRECT=2 \
  --setenv DSGB10_RING_RSBAL=1 --setenv DSGB10_PF_TUNE=hc,small,down
```

Of these, `--big-slots`, `--draft-head4`, `--draft-markov8`, `DSGB10_BUDGET`, `DSGB10_DRAFT1`, `DSGB10_WIDE`,
`DSGB10_WIDE_COST`, `DSGB10_HEAD4_C1`, `DSGB10_CF_MLP`, `DSGB10_CF_ALPHA`, `DSGB10_CERT` and `DSGB10_RING_DIRECT` are pico-dgx-ds4's
options, and the code behind `DSGB10_RING_RSBAL` and `DSGB10_PF_TUNE` is pico-dgx-ds4's too; the others are this repository's.

| In that line | What it does |
|---|---|
| `--max-ctx 4194304` | the largest window, 4M tokens (1M, 2M and 4M open on demand) |
| `--slots 32 --big-slots 0` | 32 lines, every one able to grow to the whole context |
| `--pool-free-gib 6.3` | the engine leaves 6.3 GiB to the host on every Spark, counting every outside user of memory at its cap ([Memory left to the host](#memory-left-to-the-host---pool-free-gib)) |
| `--engram-vram on` | the Engram cache's first slots sit in the display reservation; with 2,032 MiB the whole cache is there, and a larger size puts its first 2,032 MiB there (the engine's user must own `/dev/dri/card0` and no screen may be connected; `off` otherwise) |
| `--engram-cache-mib 2032` | the Engram cache's size at start |
| `--disk-gib 512` | the SSD tier's size at most (each Spark offers its free space less 32 GiB) |
| `--front-mib 840` | the front-end process's memory cap, one of the outside users the store is sized around |
| `--media-mib 512` | the media worker's container cap, another of them |
| `--pscan all` | the decode index scan split by positions over the four ranks, in every window |
| `--kv-split tiered` | the split cache's mode |
| `--prefill-rows-max 16384` | long prompts are processed 16,384 rows at a pass on a second set of row buffers lent by the store while it is free; 0 processes 8,192 rows at a pass only ([Long prompts processed 16,384 rows at a time](#long-prompts-processed-16384-rows-at-a-time)) |
| `--draft-head4`, `--draft-markov8` | the drafter's head as 4-bit and its Markov head as FP8 |
| `--ablit FILE` | the abliteration file the `-ablit` name serves (in this repository: `tools/ablit/ablit.bin`) |
| `--vision-weights FILE` | each rank's quarter of the image encoder (made by `dsw_convert --vision`) |
| `--weights FILE` | each rank's weight file (version 3) |
| `--setenv NAME=VALUE` | sets an environment variable of the engine process on every rank |
| `DSGB10_COPY=1` | copy drafting on |
| `DSGB10_COPY_ADAPT=0` (not in the line above) | the learning copy drafter is on whenever copy drafting is: the table is its starting values, and the engine counts how often copies of each length are accepted and uses its own counts; each conversation also gets a quick correction. `0` turns it off (the table alone decides). With `DSGB10_RELAX` or `DSGB10_RELAX_P` set it stays off unless set to `1`, and then the start is refused. Every Spark must have the same copy settings and table file: a start where they differ stops |
| `DSGB10_COPY_TABLE=FILE` (not in the line above) | optional: a copy-drafting survival table fitted from real traffic by `tools/copy/fit.py`, the same path on every Spark; a start whose file cannot be read on one rank stops. This repository does not ship one: without the setting the engine uses its built-in table, which is what At a glance, the long-context table and the learner comparison were measured with; the first two rows of [Drafting](#drafting) and the interleave table were measured on an earlier build with a table fitted from real traffic |
| `DSGB10_BUDGET=1` | every session may draft up to the rows left after one for each, so drafts are shared across sessions (lossless) |
| `DSGB10_DRAFT1=6` | batch steps of 6 or more sessions draft them in one pass |
| `DSGB10_WIDE=64`, `DSGB10_WIDE_COST=24,2.2,57,1.1` | steps of 17 to 64 rows run through the prefill GEMMs, and the cost model that takes the wide step only when the drafts pay for it |
| `DSGB10_HEAD4_C1=1` | the 4-bit draft head serves one-user drafts only; batches use the FP8 copy |
| `DSGB10_CF_MLP=FILE`, `DSGB10_CF_ALPHA=1.0` | the small network that decides how many drafts a step verifies (`tools/stack4/cfmlp_cml.bin`, copied to the same path on every Spark; a start with the variable set and the file missing or of the wrong size stops), and the scale of its threshold |
| `DSGB10_BATCH_CF=1`, `DSGB10_BATCH_TRIM=1`, `DSGB10_BATCH_TRIM_COST=19,2.9` | the draft count for several sessions ([Drafting](#drafting)) |
| `DSGB10_CERT=1` | greedy rows read a coarse head and the exact head only where the winner can lie (the certified head; sampled rows too, [Kernels](#kernels)) |
| `DSGB10_RING_DIRECT=2` | on one switch every rank writes to every other rank itself, with no relays; leave it out on a cabled ring |
| `DSGB10_RING_RSBAL=1` | pico-dgx-ds4's balanced reduce-scatter: the prefill exchange's reduce-scatter spreads each neighbour's share of the data evenly over the card's two PCIe functions. pico-dgx-ds4's code, switched on in the served settings ([Kernels](#kernels)) |
| `DSGB10_PF_TUNE=hc,small,down` | pico-dgx-ds4's prefill tuning: the prefill kernels' shapes per row count (the split of the hyper-connection step for passes of up to 344 rows a Spark, small-row passes, and pico-dgx-ds4's column-group down kernel from 4,096 rows). pico-dgx-ds4's code, switched on in the served settings ([Kernels](#kernels)) |

`DSGB10_OEA` and `DSGB10_RELAX*` are pico-dgx-ds4's lossy speed settings; they are off unless set and the served settings leave them
out; with either of `DSGB10_RELAX` and `DSGB10_RELAX_P` set the copy drafting learner is off.

### The front-end

```
dsfront serve --listen 127.0.0.1:18890 --api-key-file KEYFILE --engine-socket /path/engine.sock --table-file TABLE \
  --tokenizer CHECKPOINT/tokenizer.json --encoding 1017 --model-name deepseek-v4.1-flash-1017 \
  --thinking --effort 75 --temperature 1.0 --top-p 0.95 --max-tokens 1048576 --front-mib 840 \
  --media-addr 127.0.0.1:18951 \
  --node-stats RANK0:9760,RANK1:9760,RANK2:9760,RANK3:9760 --dashboard-host NAME --fan-tool PATH
```

`--tokenizer` is the checkpoint's `tokenizer.json` (its default path exists only on the machine the front-end was written on,
so a start without it fails); `--encoding 1017` is required for DeepSeek-V4.1-Flash-1017 (the default, `v4.1`, is the chat
format of DeepSeek-V4.1-Flash); `--model-name` is the name `/v1/models` lists (default `deepseek-v4.1-flash-uncensored`; the same
name with `-ablit` is served beside it); thinking is off unless a request or `--thinking` turns it on, and `--effort` (75) is
its default effort. `--api-key-file` makes `/v1` and the dashboard require the key (without it requests are not authenticated),
`--listen` is the address and port (repeat it), `--engine-socket` is rank 0's socket, `--table-file` is where the session
table is kept across a restart (`--table-save-s` and `--table-days` do nothing without it), and `--front-mib` is the cap the
front-end watches itself against: the kernel's resident count of the `dsfront` process, read every 10 seconds (0: not
watched). Give it the same value as the engine's `--front-mib`, which should be what the front-end holds at most; the 840 here
is the cap the measurements of this README ran with. A start with `--pool-free-gib` is refused without an engine `--front-mib`.
`--media-addr` is the media worker's address: the worker script `tools/vision/media_worker.py` listens on 18950 unless given
`--port`, and the container's command uses 18951, which is what the served line names; `service.sh` does not start it.

### The code's defaults and the other options

The defaults in this table are the code's, which differ from the served line above where the table says so. Every other
engine option, including the test and benchmark modes, is in pico-dgx-ds4's [`engine.md`](ds41f-4xgx10-switchless/docs/engine.md); what this
repository changed in the engine is [`docs/engine.md`](docs/engine.md).

| Flag or variable | |
|---|---|
| `--slots N` (default 1; the code allows up to 64, served and tested at 32) | lines; with the store, `--big-slots 0` (every line can grow to the whole context) |
| `--pool-free-gib G` (unset), `--pool-tokens N` (unset), `--pool-pick smaller\|larger` (smaller) | the store's size: what leaves G GiB of device and of available host memory to the host on every Spark with every outside user of memory at its cap, or a store of N positions, or the smaller or larger of both ([The memory store](#the-memory-store)) |
| `--front-mib N` (0; engine and front-end: an engine start with `--pool-free-gib` is refused at 0, a front-end at 0 does not watch itself), `--media-mib N` (512), `--encoder-mib N` (1,260), `--frame-stage-mib N` (400), `--ids-host-mib N` (1,024), `--pool-slack-gib G` (0.75) | the most each user of memory outside the store may take |
| `--prefill-rows N` (8,192), `--prefill-rows-max N` (0: off), `--prefill-loan-low-groups N` (2), `--prefill-loan-high-groups N` (8) | the rows a prefill pass takes with the standing buffers; the rows a pass takes on the lent set (0 or equal to `--prefill-rows`: no lent set; served 16,384); the free groups under which the lent set goes back, and how many free groups above the set's own are needed before it is asked for again ([Long prompts processed 16,384 rows at a time](#long-prompts-processed-16384-rows-at-a-time)) |
| `--grow-ahead N` (131,072) | the output horizon a request reserves |
| `--slice N` (64) | tokens a request runs before the next waiting one takes a turn (0: no time-slicing) |
| `--kv-split off\|tiered\|max` (off), `--kv-split-high N`, `--kv-split-low N`, `--kv-split-after-s S` (0), `--kv-split-budget N` (16), `--kv-split-ride-ms N` (250), `--kv-split-ride-pieces N` (4) | the split cache ([The SSD tier, the split cache](#the-ssd-tier-the-split-cache-and-crash-checkpoints)) |
| `--disk-gib G` (32; 0: no tier), `--ssd-flush-s S` (30), `--keep-days D` (30), `--keep-gib G` (0; 64 as served) | the SSD tier: its size at most (each rank offers its free space less 32 GiB), the spacing of crash checkpoints, how long an unused conversation is kept, how much a planned stop keeps; size and expiry live on the dashboard |
| `--table-file PATH`, `--table-save-s S` (30), `--table-days N` (7) (front-end) | the session table's file, kept across a restart so the front-end finds its conversations; how often a changed table is saved, and how long an unused session is kept. The last two do nothing without `--table-file` |
| `--max-ctx N` (65,536; 4,194,304 as served), `--extend on\|off` (on), `--window-idle-s S` (0) | the window: to 4M, whether a conversation may go past 1M (live on the dashboard), after how many seconds with no request of its factor a larger window's memory goes back to the store (0: only when the store needs it) |
| `--engram-cache-mib N` (4,096), `--engram-cache-max-mib N` (16,384), `--engram-cache-min-mib N` (the start size), `--engram-vram off\|on\|require` (off) | the Engram row cache: its size at start, the most and the least it holds, and whether its first slots sit in the display reservation (the engine's user must own the card) |
| `--vision-weights FILE` (engine) | the image encoder's weights; without them the engine answers "this engine has no image encoder" |
| `--media-addr ADDR:PORT`, `--vision engine\|container` (engine), `--vision-addr ADDR:PORT` (front-end) | the media worker the front-end takes pixels from (without `--media-addr` image requests are refused); `container` with `--vision-addr` is the container encoder |
| `--snap-every N` (1,024) | positions between window bookmarks |
| `--kv-split-pass-ms N` (50) | the longest a pass of split commands runs while requests are in flight, at the highest pressure; down to a fifth of it as the pressure falls, 250 ms with nothing in flight |
| `--kv-join-pieces N` (64), `--kv-join-whole-ms N` (250) | the size of one return command, and the estimated time under which a return goes back to back |
| `--copy-mib N` (512) | copy drafting's memory on every Spark |
| `--engram-vram-card PATH` (`/dev/dri/card0`), `--engram-vram-mib N` (2,032) | the DRM device the Engram cache's reservation is made on, and the most it asks for |
| `--listen ADDR:PORT` (127.0.0.1:18890), `--api-key-file PATH`, `--engine-socket PATH`, `--top-p P` (0.95), `--temperature T` (1.0), `--effort N` (75) (front-end; the engine's own `--top-p` is a different option, default 1.0) | where the front-end listens (repeat for more), the bearer key it requires, rank 0's socket, and the sampling and effort defaults for a request that names none |
| `--tokenizer PATH` (no usable default), `--encoding v4.1\|1017` (v4.1), `--model-name NAME` (`deepseek-v4.1-flash-uncensored`), `--thinking` / `--no-thinking` (off) (front-end) | the checkpoint's `tokenizer.json`; the chat encoding (the 1017 checkpoint needs `1017`); the served name, with `NAME-ablit` beside it; whether thinking is on when a request does not say |
| `--hold-s N` (600), `--no-loop-guard`, `--tokenize-threads N` (8), `--dump-dir DIR`, `--admin-socket PATH` (front-end) | how long a request waits while the engine restarts, then is answered 503 with `Retry-After: 60`; the repetition guard off; tokenizer threads; request dumps for debugging; the supervisor's socket (`service.sh` uses it) |
| `--max-image-rows N` (40,000), `--max-image-rows-in-flight N` (80,000), `--front-image-mib N` (400), `--max-body-mb N` (front-end) | the image limits |
| `--no-prefix` (front-end) | every request sends its whole prompt (token ids kept once, off) |
| `--pscan off\|ext\|all` (off) | the decode index scan split by positions over the four ranks: `ext` in the 2M and 4M windows, `all` in the 1M window too |
| `DSGB10_BATCH_CF=1`, `DSGB10_BATCH_TRIM=1`, `DSGB10_BATCH_TRIM_COST=a,b` | the draft count for several sessions: the MLP's survival in batch steps, the trim within 16 rows and its step cost, a + b x rows ms (19, 2.9 measured for batch steps) |
| `--node-stats-port N` (engine, 9760), `--node-stats ADDR:PORT,...` (front-end) | each rank's node sampler and where the dashboard reads them; without the second the node-health panel is empty |
| `DSGB10_COPY=1`, `DSGB10_COPY_TABLE=FILE`, `DSGB10_COPY_DUMP=1`, `DSGB10_COPY_ADAPT=0` | copy drafting: rows on, the survival table (the same file on every Spark; the learner's starting values), the dump mode that logs proposals without adding rows, the learner off (it is on with copy drafting) |
| `--fan-tool PATH` (front-end) | the dashboard's COOLING AND CLOCK box runs this tool, which is `dgx-fan-fleet` (`status --json`, `set --json PROFILE`, `clock set --json LIMIT`); without it the box is absent |
| `--dashboard-host NAME` (front-end) | a host name the dashboard answers to, besides its own addresses and localhost |
| `--interleave-share P` (40), `--interleave-gap-ms N` (600), `--interleave-whole-rows N` (32,768), `--interleave-push-rows N` (131,072), `--interleave-burst-max-ms N` (1,000), `--no-interleave` | interleaved prefill; all live on the dashboard ([`docs/interleave.md`](docs/interleave.md#settings)) |
| `--ablit FILE` | the abliteration the `-ablit` name serves; the front-end's `--no-ablit-alias` leaves the name out |
| `--max-tokens N` (front-end; 32,768; 1,048,576 as served) | the output cap for a request that sets none; the default cuts long outputs |
| `DSGB10_NO_GATHER_ALIAS=1`, `DSGB10_ALIAS_CHECK=1` | gathered weights: keep the separate copies, or check every slot's checksum in place |
| `DSGB10_STORE_CHECK=1` | prints the store's digest after each window, state and reservation command; the four ranks' lines must agree |
| `DSGB10_VMM_POISON=1` | test mode: freed chunks are filled, so a stale read shows |
| `PrefillTuning::upk` (`kUpKernel`) | the prefill up kernel: 1 the producer/consumer one (default), 0 `k_moe_up_p` |
| `DSK_PRE2=<mask>` | the decode GEMVs' second weight batch before the wait: 1 wq_a + wkv, 2 the router gate; 3 (default) both, 0 neither |
| `DSGB10_CERT_SAMPLED=0` | with `DSGB10_CERT=1` (pico-dgx-ds4's): sampled rows keep the full head (by default they take the certified one) |

| `--front-now-mib N`, `--encoder-now-mib N` (0) | what the front-end and the image process hold when the engine reads memory at start; 0 takes their whole caps off the store |
| `DSGB10_CKPT_STOP=1` | test only: the engine stops between the ranks' indexes and the server's at a crash checkpoint, to test the recovery |

Other `DSGB10_*` names in the code are test dumps, traces and fault injection (`*_DUMP`, `*_TRACE`, `*_FAIL`, `*_POISON`, `*_BREAK`, `*_SHORT`, `*_SLOW`, `*_OUTSIDE`, `*_TAG`, `PF_*`); the documents in `docs/` and pico-dgx-ds4's `engine.md` name the ones they use.

## The memory store

The GB10 has one memory for the CPU and the GPU, so everything competes for the same bytes: the weights, the conversations'
caches, the images, the Engram row cache, the front-end, the kernel's own page cache. The engine takes the memory that is
not weights as one **store**: a pool of equal chunks, made once at start, the same size on every Spark. The whole account is
in [`docs/store.md`](docs/store.md).

What lives in the store:

- **Conversations' caches.** The compressed KV and index keys of every conversation, in chunks, with a list per
  conversation. Forks share their base's chunks.
- **Restart points.** The snapshots a conversation restarts from (two chunks each, three for the last one).
- **Images.** A request's image rows are taken from the store when it starts, filled as the image is encoded, and given back
  when the request ends, however it ends.
- **Memory lent to the engine's other users.** Whole groups of 64 chunks (178 MiB) lent to the larger context windows
  ([Context to 4M](#context-to-4m)), the Engram row cache ([below](#the-engram-row-cache)) and the image encoder's work area
  ([Images and clips](#images-and-clips)). The start prints one line for each. A borrower gives its groups back when the
  store needs them, and every Spark checks that the memory really moved.

So the Engram cache, the images and the larger windows take their memory from the store and never from what is left to the
host. Everything in the store that no running request uses is cache and gives way to whatever needs room, in one order; a
running request is never taken from ([Requests that wait for memory](#requests-that-wait-for-memory)).

### Memory left to the host (`--pool-free-gib`)

`--pool-free-gib G` is how much memory the engine leaves to the host when it starts. It is a reservation made once, at
start: the store keeps its size while the engine runs, and the engine neither hands memory back nor takes more later.

At start, on every Spark, the engine reads the free memory, takes away G and the most each of its other users of memory may
take, and makes the store from the rest. The Spark with the least decides, rounded down to whole groups, so all four hold the
same store. These are the only users of memory outside the store, each with the most it may take; they can grow while the
engine runs, up to these caps:

| Outside the store | Up to | Set by |
|---|---|---|
| the engine's own heap | 1 GiB | fixed |
| the front-end process (rank 0) | what you give it | `--front-mib`, on the engine and the front-end both (a start with `--pool-free-gib` needs it) |
| the media worker (rank 0) | 512 MiB | `--media-mib` |
| the front-end's output queue (rank 0) | 64 MiB | fixed |
| request frames (rank 0) | 800 MiB | `--frame-stage-mib` (400, twice; a larger request is answered `too-large`) |
| the server's token ids (rank 0) | 1 GiB | `--ids-host-mib` (past it, idle conversations' ids go to files and come back with the conversation) |
| buffers of the split cache and of the state transfer | computed from the chunk size | fixed |
| the kernel's own growth | 0.75 GiB, not enforced | `--pool-slack-gib` |

So with every one of these at its cap, a Spark keeps G free; while they hold less, which is usual, it keeps more. What a
given cluster keeps depends on its machines and on what else runs on them, so the start prints a table: each line with its
cap and what it holds, the store's size, and what each Spark would keep with the store full and every line at its cap. The
start is refused, with the reason, when the store could not hold one conversation of the full window together with the
Engram cache, the largest image request and the largest window, or when one of the engine's own lines already holds more
than its cap; it warns when a Spark would be left under 2 GiB. While serving, each Spark reads its available memory every 10
seconds and rank 0 logs `memory: rank R has X GiB available, under the G GiB left to the host; ...` at each new low under G.

Without `--pool-free-gib`, `--pool-tokens N` sizes the store by a number of positions instead, and leaves nothing aside by
this rule (no `--front-mib` needed).

## Long prompts processed 16,384 rows at a time

A long cold prompt is processed a number of rows at a time, each a pass over the whole model (a row is one token of the prompt). The
engine's standing prefill buffers hold 8,192 rows. With `--prefill-rows-max 16384` the engine also keeps a second, complete set of row buffers for 16,384 rows, made in
memory the store lends while it is free, and processes a long prompt on it 16,384 rows at a time. A pass of 16,384 rows keeps
its window restart points at the same spacing as before (a pass may hold up to 16 of them, `kMaxSnaps`). The lent set is a
tenant of the store ([Tenants](docs/store.md#tenants): device memory, on demand, the first to give
way), so what each Spark leaves to the host does not change. Rules:

- **A second set, never a grown one.** The standing 8,192-row buffers are made at start and never touched. The lent set is a
  second, complete copy of the row-sized buffers (device arrays and the pinned host buffers of the Engram lookups): 23 groups
  of the store, about 4,065 MiB a Spark. While it is held the standing set idles. No buffer is ever grown or remade in place,
  because the Engram lookups of the next pass write into them between commands.
- **One set for a request, the same on all four Sparks.** Rank 0 picks the set when a prompt's processing starts and keeps it for
  every piece of that request. Prompts with more than 8,192 rows to process use the lent set while it is held; the others use the
  standing one.
- **Taken only when it moves nothing.** After start, the lent set is asked for when the store's top 23 groups (the groups at
  the end of the store, which a take would have to empty) hold nothing in use and the free groups are at least the set's 23
  plus the high mark (`--prefill-loan-high-groups`, 8), so 8 stay free after the take. A conversation is never evicted for it.
- **The first to give way.** When the store's free groups fall under the low mark, the set goes back at the first moment no
  prompt is processed on it, before any cached conversation is evicted for room; a prompt being processed when it is asked back
  finishes at 16,384 rows a pass first. Once the set is going back it is withdrawn only if the free groups rise to the set's groups plus the
  high mark again, so a store hovering at the low mark does not flap between held and given back.
- **A failed allocation is undone everywhere.** If one Spark cannot make the set, every Spark frees what it made, the groups go
  back to the store, Prompt Processing stays at 8,192 rows a pass and the engine does not ask again for 300 s.
- **Off is the engine as it was.** `--prefill-rows-max 0` (or a value equal to `--prefill-rows`) makes no tenant and no
  second set. A kept session saved with one setting is still valid with the other, because outputs are bit-equal at either.

| Option | Default | |
|---|---|---|
| `--prefill-rows-max N` | 0 (off); served 16,384 | the rows a pass takes on the lent set |
| `--prefill-loan-low-groups N` | 2 | the free groups under which the set goes back |
| `--prefill-loan-high-groups N` | 8 | the free groups above the set's own that are needed before it is asked for again |

Measured on the four Sparks, GPU clock limited to 2,200 MHz, the dashboard open, the served settings; the same build with and
without the loan, each from a clean boot, one run each. Single cold prompts: 9 to 10 % faster from 28K to 1M tokens (1M: 9,357 to
10,276 t/s), 8.6 % at 2.05M (8,391 to 9,112) and 7.2 % at 4.15M (7,005 to 7,509). Token Generation after the prompt: equal
after 1M and 2.05M (104.7 and 105.5, 103.6 and 105.0 t/s, without and with); after 4.15M 103.6 without and 95.8 with, so not
shown to be equal. Beside 4 decoders: a prompt of about 785K tokens got its first token 7.4 % sooner (151.1 to 139.9 s), one of
about 170K tokens 5.3 % later (51.0 to 53.7 s). Bit for bit:
with the loan off, on, and on with the served settings, two cold 150,000-token prompts of 200 tokens each equal the 8,192-row
record (`bit_probe.py`).

Tests, `engine/server/tests`: `prefill_loan_test.py` with the modes `chunks` (a cold prompt processed in the expected number of
passes of 16,384 rows and the start line's 23 groups), `cycle` (conversations fill the store until the set goes back with none of
them evicted first, then are released and the set returns), `asked` (the set is asked back while a prompt is processed, also with the
request cancelled mid-prompt), `fault` (one Spark's allocation refused: every Spark at 8,192 rows a pass, the groups back) and `digests`
(with `DSGB10_STORE_CHECK=1` the four ranks' store digests agree, the loan's commands included); `rows_loan_test` (host, no GPU)
checks the policy as a pure function over its table of cases and fails against a policy that asks whenever numbers are free.

## Context to 4M

A conversation's context grows on its own, from 1M to 2M to 4M tokens, in the middle of the conversation or even of a
response, and the call is not rejected: when the conversation outgrows its window the engine moves it to the next window and
carries on. The rest of this section is how that works.

DeepSeek-V4.1-Flash's rotary scaling is YaRN with a factor of 16, which holds 1,048,576 positions. YaRN's frequency
interpolation depends on the factor, and rotary is applied before a key is stored (the write path runs `kv_norm`, RoPE,
then FP8), so keys cached at one factor are wrong at another. A conversation's cache is therefore valid for one factor
only, and a crossing processes the conversation again instead of re-rotating its keys, which keeps the result equal to a cold
start. One running engine serves 2M and 4M by choosing the factor per request. With `--max-ctx` at or below 1,048,576
every request runs at factor 16 and none of this code path is taken. Reference: [`docs/engine.md`](docs/engine.md#context-past-1m-a-yarn-factor-per-request).

- **A factor per request.** A request runs at the smallest factor whose window holds it: 16 to 1,048,576 positions, 32
  to 2,097,152, 64 to 4,194,304. The factor is part of every cache match key (a continuation, a longest common prefix, a
  fork, a read-back from the SSD, the front-end's prefix index) and captured graphs are keyed by (rows, context, factor). A step
  never mixes factors, so requests of different factors take turns: each run starts with the first waiting request of
  another factor than the last run's (`turns_test`).
- **Recompute at a boundary.** A conversation whose next prompt does not fit its window gets a new session, processed from
  the start at the next factor. The replaced session stays as a resident alternate, kept at its own factor, so forks and regenerations
  whose prompts are still under the boundary match it; with a disk tier it leaves the pool at once. An output that
  reaches its window pauses at the last position, the prompt and the output so far are processed again at the next factor (in pieces when other requests decode), and decoding continues from
  the sampled token. Both paths process the tokens again, so the result equals a cold start on the same prompt bit for bit
  (tokens and log-probabilities); nothing is approximated. An output past 4,194,304 ends with reason `context`.
- **Rotary values computed in the kernels.** Factor 16 reads the table built at start. Factors 32 and 64 compute each cos
  and sin from (position, factor) with the code that builds the table (`rope_calc.cuh`), which saves tables of 0.5 and
  1 GiB per rank. The kernels' factor-16 values equal the table's bit for bit (`rope_factor_test`; `rope_kernels_test`:
  computed equals table).
- **Selection above 1M.** The dense index scan has two instantiations. At or below 1,048,576 keys it is the scan of a
  1M-window build: the same instructions and bits. Above that a wide path lists selected blocks in segments of 65,536 and
  flushes its packed 16-bit histogram counters every 64 chunks, because a counter holds 65,535 and a 4M scan can
  overflow it (`sel_tests`: selection against a host top-k at 256K, 1M, 2M and 4M).
- **Windows opened on demand.** The engine starts with what factor 16 needs: step graphs and index-scan score buffers
  sized for 1,048,576 keys, so the pool of a `--max-ctx 4194304` start equals that of a 1M start at the same
  `--pool-free-gib`. The graphs and score buffers of factors 32 and 64 (about a GiB of graphs for each, and score buffers
  that grow with the window) are created when a request first needs the factor. Rank 0 broadcasts a window command before the request's first step and every rank sizes the window alike.
  The pool then gives up whole groups of 64 chunks (178 MiB, the smallest set whose slices of all four arrays are whole
  2 MiB physical allocations) from its end: the conversations holding them evict their pieces, least recently used first; the
  allocator and the arrays shrink without moving anything that stays mapped, so factor 16's captured graphs remain valid;
  the freed physical allocations go back to the driver for the new buffers. Idle and paused lines holding one of those
  chunks are detached first; a running request holding one makes the window wait for it. If any rank lacks the memory
  the open is undone on every rank. A window's graphs and buffers are freed, and the pool re-creates its chunks, when
  something is short of chunks and no request needs the window (also by time with `--window-idle-s`, off by default).
- **Extend context.** `--extend on|off` (live from the dashboard's EXTEND CONTEXT checkbox) turns growth past 1M on or
  off; off keeps the 1M window and refuses longer requests.

Measured on four GB10 through the front-end, the dashboard open, the engine as served (`--max-ctx 4194304`): one conversation,
a cold prompt of Python source asking for a long answer, then 256 tokens of greedy output.

| Prompt | Window | Prompt Processing, cold | Token Generation after it |
|---|---|---|---|
| 1M tokens | 1M (factor 16) | 97.3 s (10,276 t/s) | 105.5 t/s |
| 2.05M tokens | 2M (factor 32) | 225.0 s (9,112 t/s) | 105.0 t/s |
| 4.15M tokens | 4M (factor 64) | 552.6 s (7,509 t/s) | 95.8 t/s |

A crossing costs one Prompt Processing pass over the whole conversation at the new factor; through rank 0's socket, one
conversation of real text took about 115 s to cross from 1M into 2M and about 256 s to cross from 2M into 4M (measured on the earlier build, before prompts were processed 16,384 rows at a time). At or under 1M, outputs
are bit-equal to a 1M-window build, PP is equal and the TG median differs by -0.6 %, inside
the bench's spread. A 1.1M request at factor 32 is bit-equal to a build with whole-window tables.

Tests, `engine/server/tests`: `factor_test.py` (a 1.1M request at factor 32; a crossing equals a cold start bit for bit; a
crossing in mid-output; extend off refuses past 1M), `kept_factor_test.py` (two phases around a restart: a 1.1M
conversation at factor 32 and a 20K one at factor 16 each restart from their kept state, equal bit for bit),
`lazy_window_test.py` (a 4M start's pool equals a 1M start's; the window opens, closes back exactly and reopens),
`needles_boundary.py` (needle retrieval across the 1M and 2M boundaries).

## The SSD tier, the split cache and crash checkpoints

The disk tier as a mechanism is pico-dgx-ds4's: each rank writes its own file on its own NVMe, opened `O_DIRECT`, sized to the
smallest rank's offer of free space less 32 GiB and at most `--disk-gib` (32; 0 turns the tier off), with pinned staging
and a writer thread; kept sessions across a planned stop (`--keep-gib`, 0: none) are pico-dgx-ds4's too. What the tier stores here
is the piece, and it is written as each Spark's own quarter. Reference:
[`docs/split.md`](docs/split.md) and [`docs/engine.md`](docs/engine.md#the-disk-tier-piece-by-piece)
(the disk tier, piece by piece).

### Quarter images, written in the background

A piece is a cache entry (one chunk's four slices) or a bookmark (window or end). Each Spark writes only its own quarter of
a piece, on a second
CUDA stream, and never waits for it. A decode step, a prefill chunk or an admission shares no stream with it, and nothing
is written on a request's path. The images are queued when nothing is in flight (an idle command of `--kv-split-budget`
pieces, 16) and, inside a decoding run, by a ride: one bit of the step's control word, one host all-gather and the store's
bookkeeping between two steps, at most every `--kv-split-ride-ms` (250) and `--kv-split-ride-pieces` (4) pieces, 64 when
the store is short. Nothing is done between a prompt's chunks, at an admission, at a request's end or when a line is let go.
The four Sparks agree each command's budget and queue the same images, and each settles only up to the slowest of the four
image streams, so nothing is marked early anywhere.

A piece is **plain** (no image yet), **clean** (its quarter image is on the SSD and the chunk still holds the same bytes) or,
in the split modes, a **ghost** (the conversation says "on the SSD"; the chunk is free and its bytes are intact). A piece
goes stale only where its conversation writes again (a cut, a drop, a cold start, a turn's write into an entry, a bookmark
taken again), and exactly those images are cleared. Each image carries a checksum, checked at read-back; a mismatch cuts
the conversation below it.

### The split: a mark

`--kv-split off|tiered|max` (default `off`; `cache` is a name for `max`).

- **`off`** writes the images but marks nothing: a conversation keeps its chunks until a take evicts them, and a chunk
  with an image is given back with no write.
- **`tiered`** and **`max`** also **mark** a conversation that is on no line and has no request in flight: every clean
  piece becomes a ghost. The mark is host bookkeeping, with no device call, no copy and no stream. To the rest of the
  engine a marked conversation is a conversation on the SSD whose chunks are free. `tiered` rides only while the chunks that
  can be given back without a write are under `--kv-split-high` (default: the larger of 256 MiB and 1/16 of the store, in chunks,
  printed on the start line), so on a store that is not short a run's steps are untouched; `off` images by the same rule and marks nothing; `max` rides
  whenever images are pending.
  In every mode a request that waits for clean chunks makes the imager ride. `--kv-split-after-s` (default 0) is the idle age
  before a conversation is imaged; `--kv-split-low` (default: half of the high mark) is the mark under which a ride queues 64 pieces.
- **A return.** A ghost nobody took comes back with no read and no exchange: the conversation names the same chunk again.
  A ghost something took is read back: the four Sparks each read their quarter from their own SSD and exchange them (on a test
  build of the split cache, 64 pieces in 28.7 to 29.4 ms, about 0.46 ms a piece). RAM holds a conversation whole or not at all; the
  four-fold saving is on the SSDs, where each Spark keeps a quarter.
- **Eviction never writes.** Under pressure the allocator takes clean pieces only, last piece first (the end bookmark, then
  window bookmarks from the latest, then cache entries from the latest position down), and admission counts only chunks
  whose images exist. A take that still finds only plain pieces writes the piece's image first and waits for it
  (`image_waits_total` in STATS, which stays 0).

Tests, `engine/server/tests`: `split_test.py` (cases for the imager, the return, the wait, the reader), `split_pace_test.cpp`,
`split_return_selftest.py`, `split_wait_selftest.py`; `engine/model/tests`: `store_core_test.cpp` (the host core, with its
controls), `quarter_test.cu` and `quarter_map_test.cpp` (a quarter made on a second stream beside a busy one is byte-equal to
the as-built image).

### The SSD tier's size and expiry

The size limit and `--keep-days` (30) change live from the dashboard's ENGINE SETTINGS box. A larger size grows every
Spark's file while the engine idles, as far as each Spark's free space less 32 GiB allows (the ranks agree on the smallest
offer, so all four stay alike); a smaller size gives up the images past the new end and cuts the file, so the disk's space
comes back at once. Measured on four GB10: a live grow from 512 to 768 GiB, with the file sizes read back on each Spark. The
space is a pool of 64 KiB blocks and every rank runs the same operations on it, so images land at the same offsets.

### Read-back

A request whose conversation or fork base has a piece only on disk brings it back before it plans its restart: entries from
the first position, then the end bookmark, then window bookmarks from the latest, as far as the store has room. A
conversation coming back from the SSD is admitted on what it needs (the lists not already in the store or the lists on the
disk, whichever is more), not on a sum that counted its lists twice; the read-back stops when the store is full, and a
waiting request is rescued in three steps (its own window bookmarks, a fork without its base, an error with the numbers).
Measured on four GB10: a conversation evicted and read back is bit-equal at its restart point. With a 256 GiB tier, read-back ran
at 4.0 to 4.7 GB/s, so a 1M-token session (about 0.9 GB a rank) reads back in about 0.2 s, estimated from the rate, where
Prompt Processing of a cold 1M-token prompt takes about 97 s ([Context to 4M](#context-to-4m)). With `--keep-gib` 64, the next turn of a 64K-token
conversation after a planned restart takes 0.23 s instead of 6.55 s cold, bit-identical (`kept_test.py`, from pico-dgx-ds4's engine:
four turns across the stop and start). Kept sessions belong to the engine that wrote them: a start with another engine
binary, other weights or other options, paths included, reads none of them and starts empty. The kept index is
`DSGB10K5`: an index of another layout is not read and the engine starts clean. Tests, `engine/server/tests`: `ssd_test.py`,
`disk_test.py`, `disk_full_test.py`, `squeeze_test.py` and `fork_test.py` with `DSGB10_VMM_POISON=1`.

### Crash checkpoints

While nothing runs and the tier has changed (at most every `--ssd-flush-s`, 30 seconds), the engine saves the idle lines'
conversations as a planned stop does: every rank writes its index and the server its histories, under one generation, and
the images an index names are held until the next. A damaged piece cuts its conversation at the same position on every
rank (an allgather of the first bad position). The front-end saves its conversation table when it changes
(`--table-save-s`, 30; atomic, off the request path; a busy conversation is saved as its last finished turn).

Measured on four GB10 by killing rank 0 (SIGKILL) and letting the service manager restart it: every conversation idle at the last checkpoint answers bit-equal to the reference, and
after SIGKILL of the front-end the next turn reuses its cache. A checkpoint runs only while idle. In this repository, `kept_test.py` (pico-dgx-ds4's) runs its two phases around a planned stop
and compares the turns bit for bit, and the front-end's table has cargo tests.

## Images and clips

The engine can take images and video in a chat request. The model's own vision encoder runs inside the engine, split over the
four Sparks like the rest of the model, and what it makes goes straight into the conversation's chunks of the store, so nothing
has to cross the fabric afterwards. A small **media worker** opens the images and the video clips and turns them into pixels; it
holds no model and uses no GPU. To turn this on, start the engine with `--vision-weights` and give `dsfront serve`
`--media-addr` (the media worker); without `--media-addr` image requests are refused. Reference: [`docs/vision.md`](docs/vision.md)
(where each part of vision runs and the path of an image or a clip from the client to the first token) and [`docs/media.md`](docs/media.md)
(the media worker, the request frame and the engine's encoder).

```
client --image_url / base64 / video_url--> dsfront --> media worker (127.0.0.1, container, no model, no GPU)
                                                        <-- padded pixels, rows, hash and grid per image --
dsfront --one request: the prompt, the pixels--> engine rank 0 --> ranks 1-3
engine: the tower on four ranks (bf16 tensor-core GEMM, dense attention) --> rows in the request's store chunks
```

- **Pixels in the request.** `dsfront serve --media-addr HOST:PORT` (with `--vision engine`, the default) fetches and decodes
  the images through the media worker and sends their pixels in the request. The engine checks the frame before it keeps
  anything (at most 1,024 rows an image and 9,216 patches); an image over the worker's pixel limit is answered 413 and
  never reaches the engine. An engine started without `--vision-weights` answers "this engine has no image encoder".
- **The encoder's weights.** `--vision-weights FILE` (each rank's own quarter of the tower, made by
  `dsw_convert --vision` from the checkpoint; the four files are checked to be one set). They are loaded before the store is
  sized, so the memory left to the host accounts for them.
- **The work area.** The encoder's buffers are a tenant of the store: device memory, two groups (356 MiB), taken once at
  start and held while the engine runs. If any Spark cannot give it, all four serve text and answer a pixel request with an
  error until the next start.
- **Paced beside the decoders.** An image is encoded in units sized from the measured step, so a decode step is never held
  longer than the interleave gap (`--interleave-gap-ms`, 600). One image's blocks never interleave with another's; a text
  request is never held. With nobody decoding, images are encoded back to back. Four ranks compare a sum of each image's
  rows, and a difference ends the request with an error.
- **Held for a resume.** A request's rows stay in the store while it runs, so a resume (a paused request, a restart inside an
  image) reads the rows it holds and encodes nothing again.
- **Limits.** Up to 100 images across all messages of a request, clips' frames included; `--max-image-rows` (40,000 rows;
  `dsfront serve` sends its own with each request and the worker's `--max-rows` is the same default, the lower applies) and `--max-image-rows-in-flight` (80,000 across requests; a request that does not
  fit is answered 429 with `Retry-After`); the worker reads images up to 40,000,000 pixels and fetches up to 32 MiB each
  (`--max-image-pixels`, `--max-bytes`); `--front-image-mib` (400) counts the pixel bytes a request holds in the front-end;
  the JSON body limit is 64 MiB (`--max-body-mb`).
- **Clips.** `video_url` parts (a public http(s) URL or a base64 data URL) are sampled to up to 32 frames by default
  (`max_frames` 1 to 100), uniformly from the start through the end, each frame at most 768x768 with its timestamp, in a
  separate process with a 60-second limit (`--video-timeout-s`); clips are limited to 128 MiB (`--max-video-bytes`). Frames
  count as images against the 100 and the row cap. Audio is ignored.
- **The worker's container.** `tools/vision/Dockerfile.media-worker` (build line in its header; the setup's media-worker section runs it): Python, Pillow, numpy and PyAV pinned by hash, no
  PyTorch, running as an unprivileged user on 127.0.0.1; image decoders and the video demuxer stay out of the engine's
  process. Its memory cap is `--media-mib` (512) on the engine's start table.
- **Image conversations.** Images are keyed by content hash and place in the prompt, and the encoder's kind is part of the
  hash. A session is reused only if every image in the shared prefix is identical (a changed image is not reused), so a later
  turn of an image conversation processes only its new tokens.
- **The container encoder.** `--vision container --vision-addr HOST:PORT` uses pico-dgx-ds4's encoder container (`tools/vision/vision_encoder.py`,
  PyTorch) instead; its cap is `--encoder-mib` (1,260).

Measured on four GB10 through the front-end's API, eight users decoding and one image request among them, medians of five
runs, the engine's encoder against the encoder container (two builds of this engine; [`docs/vision.md`](docs/vision.md#measured)
says which): one 768 x 768 image's first token after 0.443 s against 0.737 s; 32 frames of 768 x 768 sent as one request, first
token after 10.20 s against 12.56 s, with the users' longest pause 0.58 s against 2.01 s; for one image their longest pause was
0.51 s against 0.33 s, inside the 600 ms gap. Every image the four ranks encoded was byte-equal to the unsplit run. On one Spark
with the engine stopped, a rank's share of the 32-block tower took 12.8 ms for a small image and 141.8 ms for a large one (1,521
to 9,189 patches, `bench_vit`). On an earlier build that encoded with the container, the second turn of an image conversation
took nearly all of its prompt from cache and started in 0.25 s (the first turn, cold: 1.40 s; through the front-end).

Tests: `engine/vision/tests` (`vit_test`: the split tower against the unsplit run in bytes, with ten deliberately wrong builds;
`vit_ring_test`, `vit_gather_test`, `bench_vit`), `engine/server/tests/vit_engine_test.py`, `vit_sched_test.cpp`,
`pixels_test.cpp`, `pixel_client.py`, `image_store_test.py`, `img_rows_test.cpp` (`engine/model/tests`),
`tools/vision/test_media_worker.py` (the grid and the pixels equal the checkpoint's reference), `test_video_frames.py`,
`test_video_errors.py`, `test_encoder_limits.py`, `smoke_100.py` and `smoke_video.py` against a running endpoint (they need the front-end's `--media-addr` and its API key).

## The Engram row cache

The model's two Engram tables are n-gram embedding tables, sharded across the ranks and kept on each rank's NVMe
(pico-dgx-ds4's [`engram.md`](ds41f-4xgx10-switchless/docs/engram.md)). A hot-row cache (`RowCache`, `engram/rowcache.h`; 264-byte rows)
keeps recently used rows in host memory; an insert writes a row, a hit reads one, and only the CPU worker touches it.

**A tenant of the store.** The store is sized without the cache. At start each Spark lends the cache whole groups of 64
chunks (178 MiB each), and the cache's block is committed and populated over those same bytes as plain host memory, so the
bytes the cache holds are bytes the store gave up, never memory taken from what the host keeps. A proof on every
Spark compares the pool's handles and the box's available memory before and after (the tolerance is the larger of a
measured null and 256 MiB); a failed hand-over leaves the cache empty and the engine serving, and is tried again.

- `--engram-cache-mib N` (4,096): the size asked for at start. `--engram-cache-max-mib` (16,384) is the largest the engine
  will ever hold; `--engram-cache-min-mib` (default: the start size) is the floor the cache gives way down to.
  `--lend-null-mib` (0) is the proof's null reading, in MiB, for tests.
- **Live size.** The dashboard's ENGRAM CACHE SIZE (2,032, 4,096 or 6,912 MiB) resizes it while the engine serves. The
  store lends or takes groups back at a run boundary, when no step is running. A size the store cannot lend is cut to what
  fits (the room keeps one conversation of the longest window `--max-ctx` allows, plus each allowed window's groups); a
  size that failed is not tried again for five minutes unless the setting changes.
- The ENGRAM CACHE and LENT OUT tiles of the dashboard show the size, the groups it borrows, the hit rate over the last
  minute and the evictions.

**The display reservation.** The GB10 has no separate video memory: CPU and GPU share one pool of system memory. This option
uses the firmware's display reservation, about 2 GiB of system memory set aside for a display that a headless Spark never
uses and that `MemAvailable` does not count. A DRM dumb buffer created on `/dev/dri/card0` (`CREATE_DUMB`, width 4096, 32
bits per pixel, at most 2,032 MiB, lower in 16 MiB steps) is that memory, and `MAP_DUMB` plus `mmap` give the CPU a mapping
of it (`engine/engram/display.{h,cpp}`).

What this memory can hold was measured on a Spark. The CPU mapping is write-combined: writes post, reads go to the device
uncached and are far slower than ordinary memory. A GPU kernel reading the buffer through `cuImportExternalMemory`
reads it uncached too, so no weights or KV go there. `O_DIRECT` transfers into or out of the mapping fail with `EFAULT`,
so the disk tier's staging cannot use it. Only memory the CPU alone touches fits: the cache's slot array. With
`--engram-vram on` the cache's first slots, up to the reservation's size, are in the mapping and the rest in ordinary
memory under one slot numbering; the CLOCK index stays in ordinary memory.

`--engram-vram off|on|require`: off (the default) never opens the card. Otherwise, before the store is made, each rank
opens the card, refuses when a display output is connected (a screen uses the reservation), makes the largest buffer it
is given up to the cache's size, and all ranks use the smallest size. If any rank cannot, all keep the cache in ordinary
memory and print one start line saying why; `require` stops instead. The engine's user must own `/dev/dri/card0`. The
buffer and mapping are released at exit.

Measured on four GB10 through rank 0's socket after an engine start, one conversation, real text, 4 lines, with `on` and
with `off`: PP and TG at 24K, 1M, 2M and 4M within 0.6 % of off, so the slower reads of the write-combined
mapping do not show; outputs are bit-equal. With `--engram-cache-mib 2032` and `--engram-vram on` the whole cache sits in
the reservation and one group (178 MiB) leaves the store for it: the start line reads "Engram cache 0 -> 2032
MiB: 1 groups (178 MiB) left the pool", the block 134 MiB resident and 100 % in huge pages.
Tests: `engine/engram/tools/rowcache_test.cpp` (the split cache equals an all-ordinary cache through inserts, hits and
evictions; the refusals), `engine/server/tests/engram_vram_test.py` (placement, bit equality),
`engine/server/tests/engram_lend_test.py` and `engine/model/tests/lend_test.cpp` (the hand-over, its proof, the resize).

Each pending set of Engram look-ahead lookups carries its unit and the token epoch it was hashed from, so only its own
conversation uses it, also when its prompt is cancelled between chunks (`lookahead_test.py`). Each unit's token history for
the n-gram hash is kept as a window: the prompt in flight whole, then trimmed at command boundaries to its last 4,096 ids.

## Drafting

The engine checks several tokens in one step when it can guess them in advance. Every guess is verified by the model before
it counts, so the output is the model's own. The guesses come from three places:

- **The model's draft heads** (DeepSeek's DSpark drafter, as dsgx10 runs it), with pico-dgx-ds4's confidence network, a small
  learned model that estimates how likely each drafted token is to be accepted; that estimate decides how many drafts a step
  checks.
- **Copy drafting.** Coding-agent output repeats text already in the conversation: edits quote the old code and tool calls
  repeat paths. In an offline count over 40 recorded coding-agent sessions (users' data, not shipped; 4.68M output tokens), 31.1 % of the output tokens sat in runs of 8
  or more that a copy of the conversation predicts. When the conversation's last 8 tokens appeared earlier, the tokens that
  followed their latest earlier appearance are offered as up to 7 extra drafts after the draft heads' own. How many of them are
  worth checking comes from a small table of how often copies of each length turn out right: the engine's built-in table, or a
  fitted one named with `DSGB10_COPY_TABLE` (`tools/copy/fit.py`). The table is only the starting point (`DSGB10_COPY_ADAPT=0`
  makes it the whole rule; it is off by itself beside a relaxed accept, `DSGB10_RELAX` or `DSGB10_RELAX_P`): the engine counts, on every Spark alike, how many copied tokens were accepted after matches of each length
  and follows its own counts, and each conversation gets a quick correction on top, so one whose copies keep failing does
  not make the engine trust everyone else's copies less. The matcher runs on the GPU inside the step and is
  identical on every Spark. Off unless `DSGB10_COPY` is set; `--copy-mib` (512 MiB a rank) is its working memory, counted by
  the store's sizing.
- **Drafting for several users at once.** When several users share a step, the same learned estimate decides how many of
  each user's drafts to check: drafts are added in order of expected accepted tokens per millisecond of added step time, and
  the rest are dropped (`DSGB10_BATCH_CF=1`, `DSGB10_BATCH_TRIM=1`, with the step-time cost `DSGB10_BATCH_TRIM_COST`; off
  unless set).

Reference: [`docs/batching.md`](docs/batching.md#copy-drafting).

Measured on four GB10 from a clean boot, GPU clock limited to 2,200 MHz, the dashboard open, the engine as served. The first
two rows were measured on an earlier build with a copy table fitted from real traffic: the first row's decode rate read from
rank 0's own log (`tools/engine_rate.py`), the second row's the bench client's totals, the mean of each case's runs; the third
on a later build with the built-in table, timed by the bench client, so its agent-edit speeds do not compare with the first
row's.

| | With | Without |
|---|---|---|
| Copy drafting: agent edit sessions one at a time (`bench/editbench.py`, pico-dgx-ds4's bench; temperature 1.0) | 141.1 t/s, 3.63 drafts accepted a step | 136.2 t/s, 3.34 drafts accepted a step |
| Drafting for several users: 2 to 5 sessions on three sets of coding prompts (`tools/gate/bench_seeded.py`, temperature 1.0) | on average +0.8 % over twelve combinations of session count and prompt set, inside the bench's spread; from 5.3 % slower to 9.6 % faster | |
| Copy drafting that learns from the traffic, against the table alone: one user, 16 users, 16 copy-friendly users, agent edit sessions (same build, a clean boot each) | every bench within 1.5 % of the table alone (one user 83.1 / 145.8 against 82.9 / 146.6 t/s; agent edit sessions 116.5 against 116.0) | |

The first row's difference (3.6 %) and the second row's average are inside the 4 to 5 % spread between two boots of this bench. Prompts with nothing earlier to copy run as
without copy drafting. With greedy decoding one session's output is bit-equal with and without either feature; sampled output
keeps the model's distribution, since every draft is verified. With several users drafting together, the output depends on
how positions are grouped into steps (pico-dgx-ds4's [`stack4.md`](ds41f-4xgx10-switchless/docs/stack4.md)
and [`wide64.md`](ds41f-4xgx10-switchless/docs/wide64.md)), as it does for any batch. Tests, `engine/model/tests`:
`copy_match_test` (the matcher against a plain reference), `copy_kernel_test` (device equal to host, bit for bit),
`spec_batch_test`.

## Interleaved prefill

Prompt Processing in one piece stops every decoding session for its whole duration: about 20 s for a cold 200K-token prompt
(the interleave tests below, synthetic token ids, four GB10). Interleaved prefill processes a prompt with more than `--interleave-whole-rows`
(32,768) rows to compute, counted after what its session, fork base or SSD tier gives back, in pieces between the decoding
sessions' windows. Reference:
[`docs/interleave.md`](docs/interleave.md).

- **Piece size follows the gap.** A piece of R rows takes about k x (76 ms + 0.092 ms x R) on the device (k a running
  correction from timed pieces). The piece is the largest multiple of 1,024 rows whose time plus one decode step fits
  `--interleave-gap-ms` (600): 5,120 rows with one session decoding, 4,096 with three.
- **Window follows the share.** After a piece of T ms the decoders get T x share / (100 - share) ms of decode
  (`--interleave-share` 40, so 0.67 x T), at most `--interleave-burst-max-ms` (1,000).
- **Queue.** Long prompts wait in a first-come queue holding no slot and no memory. A queued prompt with at most
  `--interleave-push-rows` (131,072) rows to compute starts beside the one being processed and processes its pieces first, so a 60K
  prompt does not wait behind a 300K one. With nobody decoding, pieces are 16,384 rows while the lent set is held (8,192 otherwise, while it is given back: [Long prompts processed 16,384 rows at a time](#long-prompts-processed-16384-rows-at-a-time)), back to back, as fast as one piece. A resume that must process more than the one-piece limit is processed in pieces too, each conversation's image set
  stays with its slot, and a long prompt's wait is reported as queue time.
- **Live settings.** All parameters change from the dashboard without a restart.
- **The decode time the decoders are due is settled by every run.** After a piece the decoders are due a window of decode time. A run of decoders
  pays that debt down whether the interleaver gave it a window or the loop started it with no long prompt waiting, never
  below zero, so a debt does not carry into the next window; while a long prompt only waits for the store and nobody is being processed,
  the decoders get no window at all.

Measured on four GB10 with `--slots 32` and synthetic token ids, through rank 0's socket; the last row through the front-end on
the served build:

| Case | In one piece (`--no-interleave`) | Interleaved |
|---|---|---|
| One session decoding beside a cold 200K prompt | stopped for 20.2 s | 79.3 t/s, longest silence 0.60 s |
| The 200K prompt itself | 20.2 s | 36.6 s |
| A 60K prompt behind a 300K prompt | 57.5 s | 11.9 s |
| Three decoders and a cold 200K prompt, with a 4K prompt arriving every second | not measured | 42.8 / 36.6 / 29.0 t/s, silences at most 0.80 s, time to first token of the 4K prompts 0.97 to 1.12 s, the 200K prompt 62.0 s |
| Decoders' share set live to 20 % / 80 % (`config_test.py`) | not measured | 47.0 / 159.4 t/s; the 200K prompt takes 35.7 s and 94.2 s |
| One prompt alone | the reference | within 0.2 % of `--no-interleave` (393K / 200K cold) |
| Through the front-end with the dashboard open, real text: four sessions generating code, a short prompt every 2 s, and a cold prompt of about 170K, then about 785K tokens | not measured | the short prompts' first token 0.6 / 0.84 s median, 1.08 / 1.79 s slowest; the long prompt's own first token 53.7 / 139.9 s |

A prompt processed beside decoders (interleave) and then decoded alone has the tokens and log-probabilities of its run alone, with
cancels, a second long prompt and two image prompts. Sessions decoding together in one batch do not reproduce their runs alone
exactly. The cause is the batch path, where the logits depend on how positions are grouped into steps; it is the same with
interleave off and on every build. On synthetic text the difference is in the last bits (the largest log-probability
difference 0.0032, token ids equal). On real text a request alone and in a batch of three agreed token for token until token 42
of 200, where a near tie went the other way; while the tokens still agreed, log-probabilities differed by up to 0.36 (a
150,000-token prompt beside four decoding sessions).
Tests, `engine/server/tests`: `overlap_test.py`, `stream_test.py`, `cancel_test.py`, `push_test.py`, `queue_test.py`,
`config_test.py`, `image_overlap.py` (its negative control fails as it should); `interleave_test` covers the scheduling
decisions as pure functions without a GPU.

## Requests that wait for memory

A request is **running** while it is in flight and not paused between turns (`--slice`, 64 tokens); a queued long prompt is
running from its admission. Everything else the store holds is cache. The rules for a request that needs more memory than is
free:

- **Nothing running is taken from.** A growth, a copy at a restart point, a window or a read-back that is short only of
  what running requests hold waits, and the log names those requests. A bookmark or a read-back from the SSD never closes a
  window and takes only what the steps before give.
- **Cache gives way first**, in one order: free chunks; stored conversations, least recently used first (their pieces go
  back to the allocator; a piece whose quarter is already on the SSDs returns its chunk at once, and with no SSD tier the
  pieces are freed); then the conversations on idle or paused lines (the line is detached and the conversation is stored);
  then, for a request's growth, a copy or a window, a window no request needs.
- **A request let in that finds no line or no room waits at its start and is asked again.** It never stops the engine.
- **A response's reservation.** A response that reaches its reservation ends its run and the reservation goes a horizon
  further at the next run (`--grow-ahead`, 131,072 positions). When the store cannot extend it, the request waits as a paused
  one does. A refused extension puts a request back only when its reservation is used up; until then it runs on with what
  it holds.
- **When every request that holds memory is waiting to grow** and nothing can be freed (nothing is being imaged, no
  conversation is coming back, no prompt is being processed), the engine logs `every flight waits for an extension` and, once the
  standstill has lasted 2 seconds, ends the request with the longest output so far with reason `length`. Both log lines carry the prefix `serve:`, the first
  with `BUG:` (`serve: BUG: every flight waits for an extension`; the engine ends one request and carries on) and the second
  `serve: req N ends at T tokens (length): every request that holds memory waits to grow and nothing can be freed`. At most one
  request ends in 2 seconds; the others get their tokens again at once.
- **A client that left** ends its request wherever it waits (for a line, for its session, to run, or for memory), and a
  cancelled request makes no window and takes no line.

Measured on four GB10, with a store too small for four conversations of about 355K tokens each (second turns reusing their
whole first turn) to all grow: the engine ended one request with reason `length` within 5 seconds, and the other three
carried on at once. Tests: `engine/server/tests/reclaim_test.cpp`, `pool_rules_test.cpp`, `resv_test.cpp`, `admit_test.cpp` (host);
`split_test.py --arm imager --event ext` (GPU).

## Runtime abliteration

A published abliteration edits two weights of each layer, `attn.wo_b` and `ffn.shared_experts.w2`, as
`W' = W - s Q Q^T W`. The engine loads the abliteration file (`--ablit FILE`; it is in this repository as
`tools/ablit/ablit.bin`, made by `tools/ablit/make_ablit.py` from the edit by mrfakename) and applies the same edit to those weights'
outputs, `y' = y - s Q (Q^T y)`, only on the requests that ask for it. The weights stay the checkpoint's, a request for the stock
name gets exactly the checkpoint's output, and both variants run in the same batch from one set of rank files.
The front-end serves the edited variant as a second model beside the stock one, its name with the `-ablit` suffix
(`GET /v1/models` lists both, and a request picks one with its `model` field); how it is served, including
`--no-ablit-alias`, is in [`docs/ablit.md`](docs/ablit.md#serving-it). The speeds of
[At a glance](#at-a-glance) were measured with the edit loaded and the stock model requested.

## Dashboard, fans and GPU clock

The image under [What you get](#what-you-get) is this page with 16 users generating.

`/dashboard` on the front-end shows the engine's decode (TG) and prefill (PP) rates in its header and, below it:

- **The ENGINE strip, nine tiles**, each a value, a short qualifier and, on hover, its detail. FIRST TOKEN (the median and the
  slowest time to first token of recent requests); CONTEXT CACHE (the chunks taken of the whole store, the tokens they hold
  of the tokens the pool holds); MAX CONTEXT (the longest context one request may use now: it follows Extend context);
  CONVERSATIONS (in memory, and on the SSD only); SHARED PREFIXES (the memory kept once for beginnings conversations share);
  ENGRAM CACHE (its size on each Spark, the groups it borrows from the store and its hit rate over the last minute); LENT OUT
  (what the store lent to each borrower: the Engram cache, the images, the windows past 1M); SSD CACHE (used of its size on
  each Spark, and the four together; the bytes written since the engine started are in the tip, because they wear the drive);
  CACHE HITS (of the prompt tokens that arrived in the last 15 minutes, the share found in memory and the share read back
  from the SSDs).
- **The CACHE chart** beside DECODE and PREFILL. Three lines on the left scale, in per cent: the share of prompt tokens that
  were already in memory, the share read back from the SSDs, and the share of Engram rows found. Each is a **running rate**:
  a ratio of sums over the last 15 minutes of requests ending at that point, never a mean of buckets' shares, so a quiet
  engine's lines hold still instead of falling to zero. The stepped dashed line is the **context cache's fill as a level**,
  per cent of the store (the CONTEXT CACHE tile's figure). Faint bars on the right scale are the prompt tokens that arrived
  in each 10 seconds, split into memory, SSD and computed.
- **Hover boxes** on the three charts show the bucket's values (for DECODE and PREFILL the bucket's average and peak, idle
  seconds counted as 0; for CACHE the three running rates, the fill with its bytes, when the last request was, the bucket's prompt tokens split into memory, SSD and computed, and what was read from the SSDs and how long requests waited). A box sits to the right of
  the hairline when it fits inside the plot and to the left when it does not, never outside it, and wraps inside a phone's
  width.
- **LINES**, one row per line with its phase, rate and progress; **NODE HEALTH**, a table with one row per Spark (memory split
  into weights, work and the store, temperatures, GPU, SSD wear, the two rails, fans); **ENGINE SETTINGS**, the settings that
  change live: SPLIT LONG PROMPTS, TIME FOR OTHERS, MAX PAUSE FOR OTHERS, JUMP THE QUEUE UP TO, NO SPLIT UP TO (interleaved
  prefill), SSD CACHE SIZE, FORGET UNUSED AFTER, ENGRAM CACHE SIZE and EXTEND CONTEXT. A change applies at once and lasts
  until the engine restarts; the start flags set it again.

Every data route takes the front-end's API key (compared in constant time, as `/v1` is); open
`http://<rank 0 address>:<port>/dashboard`, enter the key from `--api-key-file` (made with `(umask 077; openssl rand -hex 24 > ~/api-key)`),
and the page keeps it in the browser until sign-out; a front-end started without a key leaves it open. The speeds of
[At a glance](#at-a-glance) were measured with the page open. Reference:
[`docs/dashboard.md`](docs/dashboard.md); test `frontend/tools/dashboard_smoke.py`.

The COOLING AND CLOCK box drives an external tool (`--fan-tool PATH`: `dgx-fan-fleet`, from
[rbb-dev/dgx-spark-fan-control](https://github.com/rbb-dev/dgx-spark-fan-control), built on dgx-spark-fan-control by
christopherowen; its `docs/fleet.md`) with `status --json`, `set --json PROFILE` and `clock set --json LIMIT`. The
front-end only runs the tool, with fixed arguments and no shell.

- **Fans.** One profile applies to the fans of all four Sparks: Stock (Dell's embedded controller alone), Medium,
  Performance (the page's default), Max, or a held speed of 4,500, 6,300 or 9,000 RPM that rises with heat. No profile cools less than Stock.
  The tool writes one profile name to a small file on each Spark, and each Spark's own fan daemon (dgx-spark-fan-control,
  outside this repository) sets the fans from it. A failure of the front-end, the tool or ssh leaves every fan as
  it was. Each Spark's row shows both fans' RPM and the profile.
- **GPU clock.** The same box sets every Spark's GPU clock limit to 2,400, 2,200, 2,000 or 1,800 MHz, never unlocked (the
  page, the route and the helper each refuse it). It runs the clock tool named by `DGX_FAN_CLOCK_TOOL`, which `dgx-fan-fleet`
  calls as `TOOL LIMIT [ssh-target ...]`; this repository ships one, [`ds41f-4xgx10-switchless/tools/gpu-clock`](ds41f-4xgx10-switchless/tools/gpu-clock) (the four limits, never
  unlocked, its state file `$GPU_CLOCK_STATE`). A lock lasts until the Spark reboots. The engine does not set a limit at its
  start: `deploy/pico-ctl.sh` sets the configured `GPU_CLOCK` at every start it makes, and `service.sh` sets none.

Neither the fan profile nor the clock limit changes the engine's settings, and without `--fan-tool` the box is absent.
Each Spark's memory split counts the whole store as its KV part, from the status's pool size, and the maximum context
follows Extend context (the engine reports its live limit and ranks in STATS). Tests:
`frontend/tools/dashboard_smoke.py` with `frontend/tools/fake_fan_tool.py` (each limit set and read back; refusals never
reach the tool), cargo tests.

## Inside the engine

The sections below are internals: how the store is laid out, how token ids are kept, how many copies of the weights there are, and the kernels.

### The shared KV chunk pool

Every conversation's compressed caches live in one chunk pool per rank, made once at start: the core of the store
([The memory store](#the-memory-store)). Memory goes to a
conversation in equal-size chunks as it grows, so a slot count does not cap the conversations, a conversation that moves
to another line or goes idle copies no cache, and forks share their base's chunks. Reference:
[`docs/engine.md`](docs/engine.md#the-chunk-pool).

#### Layout

The state that grows with context is the compressed KV cache, and four layers keep one: layers 2, 8 and 14 compress at
ratio 2 (one entry for every two positions) and layer 20 at ratio 1. A **source** is one such layer's stream of entries.
Each entry holds a compressed KV vector, its scales, an index key for the sparse-attention selector and the key's scale
(356 B). The pool holds one pool-wide array for each of these four kinds of data, cut into slices of 8,192 entries, and
a chunk is one slice of each of the four arrays. The compressed-KV slice is 8,192 entries x 256 B = 2 MiB, one physical
allocation of the CUDA virtual memory API; the other three slices are 1/8, 1/4 and 1/64 of that, so only the KV slice
fills a whole allocation, and memory goes back to the driver in groups of 64 chunks (see Context to 4M):

| Slice | Size on each rank |
|---|---|
| compressed KV | 2 MiB |
| KV scales | 256 KiB |
| index keys | 512 KiB |
| index-key scales | 32 KiB |
| **one chunk** | **2,916,352 B (2.78 MiB)** |

A slice of 8,192 entries covers 16,384 positions of a ratio-2 source and 8,192 positions of the ratio-1 source. A
conversation therefore takes 3 chunks (one for each of layers 2, 8 and 14) plus 2 (layer 20), 5 chunks per 16,384
positions: 5 x 2,916,352 B / 16,384 = about 890 B per position per rank, for the four layers together. Window bookmarks
(2 chunks each, one every `--snap-every` = 1,024 positions, thinned further back) and the end bookmark (3 chunks) are
also pool chunks, used as raw bytes. Chunk 0 is a guard: it is never allocated, every list entry past a list's end names
it, and it reads zero (0xFF under `DSGB10_VMM_POISON=1`).

#### Creation and size

Every chunk is created and mapped before the engine serves, through the CUDA virtual memory API (`cuMemCreate`
physical allocations mapped at fixed addresses; backend in `engine/model/vmm.h`). Nothing is
created, mapped or unmapped on the serving path, so captured CUDA graphs keep their addresses and conversations
arriving and leaving change only the allocator's free list. The exceptions are the store's tenants, which take and give back
whole groups of 64 chunks ([The memory store](#the-memory-store)).

Each rank offers a count from what it can give and still leave `--pool-free-gib G` GiB of device memory and of
available host memory free, with every line outside the store at its cap (the table in [Memory left to the
host](#memory-left-to-the-host---pool-free-gib)), and/or from a position target (`--pool-tokens N`: N x 5 / 16,384 chunks); with both,
`--pool-pick smaller` (the default, so every Spark keeps what `--pool-free-gib` leaves) or `larger` decides. The count is rounded down to
whole groups of 64 chunks, and all ranks take the smallest offer (an allgather at start), so every rank holds the same
chunks.

#### Lists and the per-line table

A conversation holds one list of chunk ids per source. Its entry `e` of a source is the pool entry
`(list[e >> 13] << 13) | (e & 8191)`. A line owns the sliding-window rings (one per layer, 40 layers, 256 positions
deep, FP8) and the compressor carry state. The pool holds the four compressed caches, the only per-position state, which
all 40 layers read. A window bookmark is a snapshot of the ring slots at a position (every `--snap-every` positions); the
end bookmark holds the whole rings and carries. The pool's arrays are replicated on every rank, since tensor parallelism
does not shard them, so every per-position cost below is per rank.

A line's rows of a device table (`lists_d`: one row per unit and source, 136
entries at 1M) are its conversation's lists, padded with the guard; kernels find a row through the step's slot table.
Index scans and cache writes look entries up one at a time. The selector writes each key pick as a pool entry, so
attention reads the picks as given and one lookup serves the 4 to 6 layers that share a top-k list. Measured cost of the
indirection: a decode step 0.03 % longer, PP within 0.6 %.

Attaching a conversation to a line writes that line's rows (after waiting on the stream, so no queued copy reads a row
being rewritten). Releasing the line points the rows back at the guard and the conversation keeps its chunks. A
conversation that moves to another line, goes idle or returns copies no cache: only its window rings and carries (5.8 MiB
per line) come back, from its end bookmark.

#### Allocation and eviction

Chunks are taken lowest id first, a deterministic order every rank reproduces. Rank 0 decides and sends each command
with its state operations (up to 12: take, give, share, evict; operations, not chunk ids) to all ranks; every rank
applies the same operations in the same order and derives the ids itself. Decisions read only the allocator and the
store's records, never timing or free memory, so all ranks allocate identically without exchanging allocation messages.
`chunks_test` checks this: it replays one operation stream on four allocator instances, which agree, while an instance
that skips an operation differs. A line's conversation grows its lists 4,096 positions ahead of what the next step
needs, as a 4-byte list-entry copy queued before the step that reads it.

Everything in the pool that no running request uses is cache, and it gives way to whatever needs chunks: a request's
line, its growth, a copy on write, a bookmark, a read-back from the SSD or a window. Free chunks go first. Then the least recently
used conversation that no line holds and the current command does not use is evicted, one at a time and the same one on
every rank: its pieces give their chunks back, last piece first, so the beginnings other turns share stay resident
longest. With an SSD tier, a piece whose quarter image is already written returns its chunk at once and nothing is written
on the request's path ([The SSD tier and the split cache](#the-ssd-tier-the-split-cache-and-crash-checkpoints)); with no
tier the pieces are freed. Then the conversations on idle or paused lines are detached, least recently used
first, and evicted the same way. A request, its growth, a copy on write or a window that is still short then closes a
window no request needs; a bookmark or a read-back from the SSD takes only what the steps before give. Something short only of
what running requests hold waits, and the log names those requests ([Requests that wait for memory](#requests-that-wait-for-memory); `reclaim_test`, `pool_rules_test`).

#### Reservation horizon

A request reserves its prompt plus an output horizon of `--grow-ahead` positions (131,072 by default, capped by its
`max_new`), not every token it may generate; a request that names no `max_tokens` would otherwise reserve a whole 1M
window. An output that reaches its horizon ends its run and extends the reservation by another horizon at
its next run if the pool can hold it; otherwise it waits in the turns queue like a paused request and its line's memory
counts as idle for the others. A reservation ends with its request: a finished request's conversation stays on its line
as cache, with nothing reserved for it (`resv_test`). Bookmarks (a window bookmark takes 2 chunks, the end bookmark 3) take only room that
leaves every line's reservation covered, counting free chunks and what eviction can return; with none to spare a
bookmark is skipped and a later restart begins from an earlier one, so a line's lists are never left short. Six 1M-token
outputs on a small pool decode in one batch (`growth_test.py`): a request holds its horizon, not its whole window.

#### Forks by reference, copy on write

Every chunk has a count of the lists and bookmarks that name it. A fork takes its base's chunks wholly below its
restart point by count and copies only the one chunk that holds the restart point (kilobytes to 2.78 MiB); window
bookmarks at or below it are shared the same way. Nothing writes a chunk another list names: a cut of a conversation's
own history, a drop and a cold start first make the chunks from the cut on private (`own_tail`), and a bookmark written
again takes fresh chunks when it is shared. A base can be freed before its forks; its chunks stay while any fork names
them. A fork of a 40K-token conversation and its base together hold about 60 % of the chunks two full copies would.

#### Why it does not fragment

Every allocation unit has the same size, so any free chunk satisfies any request and there is no hole to search for. A
conversation's list maps its positions to chunks anywhere in the pool, so it never needs contiguous space and can grow
while one chunk is free. External fragmentation cannot occur. Internal waste is bounded by one partly filled chunk per
list, and a request's reservation grows 4,096 positions ahead inside it. The backing memory is created once and neither returned to nor
requested from the driver during serving, except when a tenant takes or gives back whole groups.

#### Lines

A line holds only its window rings and carries (5.8 MiB of real memory per rank), so lines are cheap and `--slots` can go to
64 (served and tested at 32). One step decodes 16 lines; the rest take fair turns. With 32 live sessions, all resume from their slots with no disk
read, and 24 can decode together (`lines_test.py`); 32 users together, each with a different request, reached 328.5 t/s
(the served build, the median of 3 runs, [At a glance](#at-a-glance)). A cancel reaches its own request with any number of lines:
the batch's cancel bits index its entries, not its lines (`batch_test.py`). A client that leaves ends its request
wherever it waits: for a line, for its session or to run; the front-end notices it gone while the engine sends nothing
(queued, waiting for memory, in a long Prompt Processing pass), not only at the next failed send (`cancel_test.cpp` for the queues and
the cancel marks, `server_cancel_test.rs` for the front-end).

#### Checks

Outputs are bit-equal to a build with fixed per-line slot memory, the reference of the exactness tests.
`DSGB10_VMM_POISON=1` fills freed chunks so a stale read shows.

Properties each with a test: a fork continuing in its base's line keeps that line's memory, bit-identical to a build with
per-line slot memory (`serve_test.py --exact`); a request takes its own line only through the pool check, and one that
cannot fit moves to a line that can (`room_test.py`); a session returning from disk is counted in admission, so a request
that does not fit waits; a paused request whose state went to disk counts as releasable, so it gets a line back under
time-slicing; a squeezed session store keeps every agent's restart point, each agent resuming at its end (`squeeze_test.py`).
`engine/model/tests`: `chunks_test.cpp`, `vmm_test.cpp` (host), `vmm_probe.cu` (graphs replay over new maps; 400 map and
unmap rounds beside a captured graph, 0 wrong values). `engine/server/tests`: `fork_test.py` (the same requests twice, the
second with the base rolled back inside the shared chunks and freed; a build without copy on write,
`-DDSGB10_POOL_NOCOW`, fails it), `growth_test.py`, `squeeze_test.py`, `stats_test.py`, `lines_test.py`, `pool_soak.py`.

### Token ids kept once

The engine keeps a conversation's token ids once. The front-end does not send a whole prompt on every turn: it keeps, for
each conversation, a table of **cut points** (the byte offset in the text, the id count and a 128-bit hash; no text and no
ids), sends only the text after the last cut it can prove, and names the engine's epoch and count for the rest. The engine
builds the prompt from its own ids. When the epoch or the count does not hold (the conversation was evicted, cut or the
engine restarted) the engine answers "prefix gone" and the front-end sends the whole prompt once. The epoch changes only
when a conversation's history loses or rewrites a position, so a continuing conversation keeps it.

- Host memory for the ids is one of the outside users the store is sized around: `--ids-host-mib` (1,024). Past it, the least recently used idle
  conversations' ids go to files under the disk tier's directory (`ids/`) and come back before a request uses the
  conversation. A file that does not check out (length, sum) is treated as a lost conversation.
- A conversation's index entry in the kept index names its ids file by count and sum, so the ids survive a restart.
- `dsfront serve --no-prefix` turns the suffix protocol off (every request sends its whole prompt).
- The front-end's session table is smaller for it: token end offsets are a mark every 32 tokens (8.7 B a token, measured live). `--front-table-mib` caps it: 0 derives the cap from `--front-mib` (the cap divided by
  1.1, less 107 MiB and `--front-image-mib`) and, with `--front-mib` also 0, sets no limit.

Measured: the kept index of 97 conversations of 200K ids each is 10 KB (with the ids in it, 77.6 MB); through the front-end a four-turn conversation answers equal with and without the suffix, turn
by turn. Tests: `engine/server/tests/prefix_test.cpp` (49
rows), `ids_file_test.cpp`, `prefix_test.py`; the front-end's `session.rs` and `server_prefix_test.rs`.

### Weights and copies

The model's weights are the checkpoint's; the engine keeps one copy of each where it can.

- **Gathered weights load into place.** A rank loads its own quarter of `wq_a`, `wkv`, `wo_b`, the Engram `wkv` and the
  router gate straight into its slot of the whole buffer they are gathered into, so the gather copies nothing it already
  holds. Every rank prints a line saying so and checks each slot's checksum there. `DSGB10_NO_GATHER_ALIAS=1` turns it off; `DSGB10_ALIAS_CHECK=1` checks every
  slot in place.
- **The router projection** that decode reads is the rank's own quarter, which is the weight file's tensor itself; the whole
  is built once for prefill, in one pair of buffers reused by every layer, then freed.
- **The plain rotary table is computed**, not stored (the same function the longer windows already used); the start line says
  "plain RoPE computed, no table: 0.25 GiB a rank not taken".
- **The embedding is cut by vocabulary.** Each rank keeps a quarter of the embedding and of the drafter's Markov embedding
  (token ids `[32,320 r, 32,320 (r+1))`); a token's row is written by its owner and summed over the ranks in one all-reduce
  (the others write bf16 -0.0, so the sum is exact); prefill's rows travel in the all-to-all it already makes. The rank files
  are version 3: an engine refuses a file of another version, so the weights are cut again with `dsw_convert`.
- **Vision weights** are loaded without the loader's staging.

Measured on four GB10 at start with the same settings, against the same build with the copies kept (`DSGB10_NO_GATHER_ALIAS=1`):
the first two items give the store 0.75 GiB a rank, the embedding cut 0.98 GiB; outputs are bit-equal and the store's digests equal on the four ranks. Tests: `fused_tests`, `dsw_dest_test`, `rope_factor_test`, `embed_shard_test`, `argmax_row_test`,
`alloc_audit_test`.

### Kernels

The prefill and attention kernels are checked bit for bit against their reference kernels. The certified head for
sampled rows is checked by running the full head beside it: its winner matched on every row compared. Detail and the
bit-check design: [`docs/kernel-trims.md`](docs/kernel-trims.md).

| Change | Mechanism | Measured | Check |
|---|---|---|---|
| Prefill MoE up kernel, producer/consumer | One warp moves each chunk of activations, weights and scale words into a two-stage shared-memory ring (`cp.async`, completion counted on `mbarrier`s); four warps only compute, in `k_moe_up_p`'s MMA order. No block-wide barrier in the loop, no weights held ahead in registers (238 to 155 registers), full tiles without branches between MMAs | One Spark, engine stopped, GPU 2,190 MHz (`bench_kernels`): up kernel 7.2 % shorter than `k_moe_up_p` at 8,192 rows (7.20 to 6.68 ms), 8.5 % at one row; a cold 1M-token prompt 102.9 against 105.1 s, within the run-to-run spread of 4 to 5 %; decode does not run this kernel | `bench_kernels upws`: the whole MoE against `k_moe_up_p` bit for bit, 1 to 8,192 rows, uniform and skewed routes, raw and coded scales; `serve_test --exact` |
| Prefill MoE down kernels | Weight operands expanded first; an activation quad's MMAs issued together | One Spark, engine stopped, GPU 2,190 MHz: routed down 6.424 to 6.349 ms at 8,192 rows (1.2 % shorter) | `serve_test --exact`, `batch_test` |
| Prefill MoE down kernel with a mover warp | Selected from 4,096 rows when `DSGB10_PF_TUNE` has no `down` item (`k_moe_down_ws`; `DSGB10_PF_DWS=0` selects the previous down kernel): one warp moves the next tile of weights and activations while the other warps compute, as the up kernel does. With the served `DSGB10_PF_TUNE=hc,small,down`, pico-dgx-ds4's column-group down kernel takes those sizes instead, so this kernel does not run in the served settings | Measured without the `down` item: the kernel 16 % faster at 8,192 rows; in the engine 0.7 % faster, because the exchange after the down kernel (the rows are summed and exchanged between the Sparks) takes longer than the kernel's saving, so it sets the pace of each layer | `bench_feed --check`: bit-identical |
| pico-dgx-ds4's balanced reduce-scatter and prefill tuning, switched on in the served settings | `DSGB10_RING_RSBAL=1` is pico-dgx-ds4's balanced reduce-scatter: the prefill exchange spreads each neighbour's share of the data evenly over the card's two PCIe functions. `DSGB10_PF_TUNE=hc,small,down` is pico-dgx-ds4's prefill tuning: kernel shapes per row count (the split of the hyper-connection step for passes of up to 344 rows a Spark, small-row passes, and pico-dgx-ds4's column-group down kernel from 4,096 rows). Both are pico-dgx-ds4's code | Tuning alone, against the same binary without it, from a clean boot: 16 users +4.0 % with different prompts (334.0 to 347.2 t/s), +3.7 % with the same 3 prompts repeated (377.1 to 391.1). Balanced reduce-scatter alone: equal at 16 users, a long cold prompt 2.3 % shorter (engine clock, not a clean boot) | `bit_probe.py`: bit-equal; the engine's start line names the tuning in force |
| Decode attention prefetch | The next block's keys are gathered while the current block is scored | One Spark, engine stopped, GPU 2,190 MHz (`bench_attn`): attention call time 27 % shorter with one key-range split of a row's attention, 10 % with ten | `slot_tests`, `fused_tests`, `serve_test --exact` |
| Decode GEMVs, second weight batch before the dependency wait | After a collective, `wq_a` + `wkv` and the router gate (2 to 8 rows) request their second weight batch before waiting on the preceding kernel (programmatic dependent launch, PDL), so both register buffers fill while the collective runs; a template flag on kernels of their own, applied only where `ptxas` spills nothing; `DSK_PRE2` mask. The technique follows pico-dgx-ds4's note on a second weight batch (`ds41f-4xgx10-switchless/docs/engine.md`) | Four GB10, GPU clock 2,200 MHz, a decode step at one user 0.7 to 0.9 % shorter (4 rows 2.1 %, 3 rows 1.2 %, 2 rows 0.8 %) | `bit_probe.py`: 1,600 tokens' ids, log-probabilities and top two bit-equal with and without; a build that reverses a split-K sum differs in 8 of 8 prompts |
| Certified LM head for sampled rows | The certified head (pico-dgx-ds4's `DSGB10_CERT=1`: a coarse FP8 read of the whole head with a rigorous error bound, then the exact head only where the winner can lie) is extended to sampled rows. For a row sampled at temperature up to 1 (above 1 the scores flatten, more of the head survives, and the full head is kept) with no top-k, top-p or log-probabilities (the request sets top_p 1; the front-end's default top_p, 0.95, keeps the full head), the logit bounds become bounds of the sampler's Gumbel score `l/T - log(-log(u))` (the sampler's own hash of seed, position and id), so the exact head runs only on the row groups that could hold the winner | Four GB10, GPU clock 2,200 MHz, a decode step at one user, sampled at temperature 1.0, 0.7 % shorter (1 row 1.5 %) | check mode with the full head beside it on every rank: 0 mismatches over every row compared |
| Decode index scan split by positions (`--pscan ext\|all`, default off) | With `ext` each rank scans a quarter of the keys of the four scanning layers (2, 8, 14, 20) with the same dense kernel, the four ranks' candidates are gathered in place in one collective a layer and merged into the picks the whole scan would make; only in the 2M and 4M windows. `all` does it in the 1M window too | One rank's part plus the merge against the whole scan, on an idle GPU: layer 20 at 1M 349 to 130 us (1 row), at 4M 1,440 to 410 us (1 row); in the engine (`all` as served, rank 0's step time at 1M): 3 % shorter at 2 rows a step (22.94 to 22.26 ms), equal at 6 rows and under 64K | Picks equal in 72 of 72 cases; tokens and log-probabilities equal to the whole scan in 8 of 8 turns |

Nsight Compute on `k_moe_up_p`, the one-role up kernel (selectable with `PrefillTuning::upk` 0), shows a latency-bound
kernel: no eligible warp in 67 % of cycles, occupancy 16.7 %, two block-wide barriers per chunk. The producer warp removes
the barriers and the register-held weight prefetch.

Correctness: `bench_kernels` bit checks use signed inputs and check the output's spread, so a kernel fed a wrong scale
cannot compare equal (a wrong scale byte differs in 99.99 % of outputs).

## Tests and tools

- `engine/server/tests/`: `squeeze_test.py`, `room_test.py`, `lines_test.py`, `fork_test.py`, `growth_test.py`,
  `stats_test.py`, `pool_soak.py` (every chunk taken and given back, with `DSGB10_VMM_POISON=1`), `free_at_start_test.py` (the
  memory `--pool-free-gib` leaves, under load), `prefill_loan_test.py` (the lent prefill set: chunks, cycle, asked, fault, digests), `split_test.py`, `disk_test.py`, `prefix_test.py`, `vit_engine_test.py`,
  `image_store_test.py`, `hot_path_selftest.py` (the once-a-step bookkeeping).
- `engine/server/tests/*.cpp` (host, no GPU): `pool_rules_test`, `reclaim_test`, `resv_test`, `admit_test`, `mem_watch_test`,
  `split_pace_test`, `interleave_test`, `prefix_test`, `ids_file_test`, `pixels_test`, `vit_sched_test`, `tenant_test`, `rows_loan_test` (the lent prefill set's policy).
- `engine/model/tests/`: `chunks_test.cpp`, `vmm_test.cpp` (host), `vmm_probe.cu` (GPU), `store_core_test.cpp` (the store's
  bookkeeping, replayed against traces recorded on the Sparks), `lend_test.cpp`, `quarter_test.cu`.
- `engine/vision/tests/`: the split image encoder against the unsplit run, byte for byte.
- `engine/kernels/tests/bench_kernels.cu`: `bench_kernels upws` (env `PF_UPK`, `PF_M`, `PF_ROUNDS`) times the up kernel's
  shapes against `k_moe_up_p` after comparing the whole MoE's output with it bit for bit.
- `tools/gate/`: `collect.sh` (a dump of the draft count's inputs per prompt set and session count), `gate_fit.py` (the survivals' calibration,
  each prompt set held out in turn; `test_gate_fit.py`), `compare_dumps.py` (two runs' output hashes), `bench_seeded.py`
  (pico-dgx-ds4's `tools/bench/bench.py` with each request's prefix and seed fixed; `collect.sh` runs it, another with `BENCH`).
- `engine/server/tests/bit_probe.py`: two engine builds or settings compared bit for bit on the decode path (prompts cut
  from a tokenized text, greedy with drafts; every token's id, logprob and top two).
- `engine/engram/tools/rowcache_test.cpp` and `engine/server/tests/engram_vram_test.py`: the split Engram row cache and
  its placement.
- `tools/vision/`: the media worker and its tests, `smoke_100.py`, `smoke_video.py` (against a running endpoint: they need `--media-addr` and the API key).
- `tools/copy/fit.py`: fits the copy-drafting survival table from the dump mode's output.

## pico-dgx-ds4's design records

`ds41f-4xgx10-switchless/docs/` holds pico-dgx-ds4's own records of the engine this one builds on: `c1fast.md`, `drafter2.md`,
`stack4.md`, `stack5.md`, `wide64.md`, `ced.md`, `engram.md`, `findings.md`, `experiments.md`, `batching.md`, `engine.md`,
`layout.md`, `operations.md`, `protocol.md`, `ring.md`, `vision.md` and `review/`. Each describes pico-dgx-ds4 as it stood when
it was written, with its own settings, checkpoints, numbers and paths (some of the settings it names are lossy and are left out
of this repository's served settings). This repository's documents are in [`docs/`](docs/README.md): pages written for it, and
for each subject of those records that it changed, a page of what it changed or added (`batching.md`, `engine.md`,
`layout.md`, `protocol.md`, `ring.md`), or a whole page of its own (`operations.md`, `vision.md`). Where a record and this
README differ, this README describes this repository. The engine README in that folder is pico-dgx-ds4's, kept as it is
there except for the rows this repository changed.

## Built on

- **[dsgx10](https://github.com/blaesus/dsgx10) by Andy Shu** (MIT): the engine, its kernels, the RoCE ring and the drafter.
- **[pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4)** by Pico (Patrick Moorhead): multi-user serving, the 1M
  context and the wide step, on top of dsgx10. See [`README-pico.md`](README-pico.md) for both and for the ideas it
  drew on (DeepSeek's DSpark, Carryover Drafting, certhead, CSV-Decode, FR-Spec, TVP).
- **Pull requests #4 to #6 to pico-dgx-ds4 by [MiaAI-Lab](https://github.com/MiaAI-Lab)** (merged upstream): thinking options (`reasoning_effort` validated, `thinking: {"type": ...}` read, `--thinking` for a default of on), up to 100 images in one chat request, and video
  input through timestamped frame sampling.
- **The coding agent pi** (MIT): its system prompt, tool definitions and tool texts in the agentic edit bench; `NOTICE`
  lists this and the other third-party material.
- **DeepSeek**: DeepSeek-V4.1-Flash and its DSpark drafter; reference code and chat encoding under MIT (see
  [`NOTICE`](NOTICE)).
- **[DeepSeek-V4.1-Flash-1017-abliterated](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated)** by
  mrfakename (MIT): the published abliteration edit (`abliteration-edit.npz`) that `tools/ablit/make_ablit.py` turns into the file
  the engine applies at run time, `tools/ablit/ablit.bin`.
- **[dgx-spark-fan-control](https://github.com/christopherowen/dgx-spark-fan-control)** by christopherowen: the fan driver and
  daemon that apply the dashboard's fan profiles on each Spark (outside this repository).

## License

MIT: [`LICENSE`](LICENSE), unchanged, and [`NOTICE`](NOTICE), which lists the third-party material, including the
abliteration file.
