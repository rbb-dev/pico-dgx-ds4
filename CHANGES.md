# Changes in this repository

This repository builds on [pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4) by Pico (Patrick Moorhead), from its
commit `7c123ad`, which builds on [dsgx10](https://github.com/blaesus/dsgx10) by Andy Shu. Below is every change this repository
makes to it, as the code here has it. pico-dgx-ds4's own change list is [`CHANGES-pico.md`](CHANGES-pico.md).

The code is in `ds41f-4xgx10-switchless/`. How each change works is in [`docs/`](docs/README.md); the [README](README.md) is the
user's view.

**How the numbers were measured.** Four GB10 Sparks, tensor parallel over four, one ConnectX-7 port each on one 200G switch,
serving DeepSeek-V4.1-Flash-1017 with the settings of [`deploy/pico.env.example`](deploy/pico.env.example) (a 4,194,304-token
context) and the GPU clock limited to 2,200 MHz. Speeds of the served build come from a clean boot of all four Sparks. Any
other number is the change's own before and after, and the
entry says what it was compared with. Two boots of the same build differ by up to about 5 %, so a smaller difference is
reported as equal. Prompt Processing (PP) and Token Generation (TG) are in tokens a second (t/s).

Contents: [The memory store](#the-memory-store) · [Context to 4M](#context-to-4m) ·
[The conversation cache](#the-conversation-cache) · [Prompts](#prompts) · [Weights and kernels](#weights-and-kernels) ·
[Drafting](#drafting) · [Images and video](#images-and-video) · [The dashboard](#the-dashboard) ·
[Serving and operating](#serving-and-operating) · [Fixes](#fixes)

## The memory store

[`docs/store.md`](docs/store.md) explains the whole store.

**One memory store.** At start, all the memory the weights leave on a Spark, less what you keep for the host, becomes one store
of equal chunks, the same on all four Sparks. It holds every conversation's cache, its restart points and its image rows. A
conversation that moves between a running line and the store copies nothing, and a fork shares its base's chunks.
- Setting: `--pool-free-gib` (served: 6.3 GiB kept for the host on every Spark).
- Measured: the served store is 25.38 GiB on every Spark, 30.6 million cached positions. Against the same engine with fixed
  memory per line: TG equal, outputs bit-equal; a fork of a 40K-token conversation and its base hold 58 chunks, where two copies
  would hold about 98.

**What the host keeps, checked at every start.** The engine counts every other user of memory at its cap (its own heap, the
front-end, the media worker, request frames, the server's token ids), sizes the store from what is left after
`--pool-free-gib`, and prints the account at every start. It refuses to start when the store could not hold one conversation of
the full window, and while serving it logs any Spark whose free memory falls under the setting.
- Settings: `--front-mib` (served: 840), `--media-mib` (512), `--frame-stage-mib` (400: a larger request is refused).
- Measured: the served build at ready keeps 10.1 GiB of host memory free on rank 0.

**Up to 64 conversations live at once.** A line (`--slots`) holds only a conversation's window state, 5.8 MiB a Spark, and points
at its chunks in the store, so every line can grow to the whole context. One step decodes 16 lines; the others take turns.
- Setting: `--slots` (served: 32; at most 64).
- Measured: 32 users together 328.5 t/s, 16 users 354.0 t/s.

**Cache gives way; a running request is never cut.** Everything no running request uses is cache, and it gives way in one order:
free chunks, then stored conversations least recently used first, then idle lines. A request reserves its prompt and a stretch
of output, not its whole window. A request that finds no memory waits; if every growing request is stuck for 2 s, the one with
the longest output ends with reason `length` and the others go on.
- Setting: `--grow-ahead` (131,072 tokens of output a reservation covers).
- Measured: with a store too small for four 355K-token conversations to grow, one ended with `length` within 5 s and the other
  three went on at once.

**Tenants of the store.** The store lends whole groups of 178 MiB to the engine's other users of memory: the 2M and 4M windows,
the Engram row cache, the image encoder's work area and the buffers for 16,384-row prompt passes. A loan that fails on any Spark
is undone on all four and never stops the engine.

**The Engram row cache, sized while serving.** The cache lives in memory the store lends, so it never takes from what the host
keeps. The dashboard's ENGRAM CACHE SIZE changes it between runs.
- Settings: `--engram-cache-mib` (served: 2,032); live: 2,032, 4,096 or 6,912 MiB.
- Measured: a resize lands in 1.2 s on an idle engine; outputs bit-equal at every size; PP and TG within 1 %.

**The Engram row cache in the display memory.** A GB10 sets aside about 2 GiB for a display, unused on a Spark with no screen.
The cache's slots live there, so they cost the store almost nothing.
- Setting: `--engram-vram on` (served; refused when a screen is connected).
- Measured: PP and TG within 0.6 % from 24K to 4M tokens, outputs bit-equal; the served 2,032 MiB cache takes one 178 MiB group
  from the store.

**Token ids kept once, in the engine.** The engine keeps each conversation's token ids. The front-end sends only the new part of
a prompt and keeps a small table of where earlier prompts ended (no text and no ids); when the engine no longer has the
conversation, the front-end sends it whole once.
- Setting: front-end `--no-prefix` turns it off.
- Measured: the saved index of 97 conversations of 200K ids is 10 KB, where it was 77.6 MB; a four-turn conversation gives the
  same outputs with and without.

**A smaller front-end session table, with a cap.** The table keeps a token mark every 32 tokens, is capped by `--front-mib`,
and is saved off the request path.
- Measured: 8.7 bytes a token on the serving front-end, where it was 22.8.

**Host memory bounded.** The Engram token history is kept to the prompt in flight plus its last 4,096 ids, the engine's heap
arenas are capped and trimmed, and a front-end connection with more than 64 MiB queued is closed.
- Measured: PP and TG within 0.4 %, outputs bit-equal.

**The store's rules in one testable class.** All the store's rules live in `store_core`, which runs without a GPU. The engine can
record every store operation on the Sparks, and the recordings replay against the class anywhere; with `DSGB10_STORE_CHECK=1`
the four ranks compare a digest of the store after every command.
- Measured: 20 recordings from the four Sparks replay equal; PP and TG unchanged (an 8K prompt 968 against 981 ms).

**Memory readings without a pause.** The memory reading every 10 s rides the decode step instead of pausing it, and rank 0's
bookkeeping for each request runs once a step, after the next step is queued.
- Measured: the decoders' longest gap before a long prompt 0.30 s, where the reading's pause gave 0.41 s.

## Context to 4M

[`docs/engine.md`](docs/engine.md) and the README's [Context to 4M](README.md#context-to-4m).

**Contexts that grow to 4M on their own.** Each request runs in the smallest window that holds it: 1M, 2M or 4M tokens, with the
matching YaRN factor. A conversation that outgrows its window is read again once at the next size and goes on, with the same
output as a cold start, and a request is never refused for its length. The dashboard's "Extend context" switches growth past 1M
on or off.
- Settings: `--max-ctx` (served: 4,194,304), `--extend on|off` (on).
- Measured, the served build, one cold prompt and then 256 tokens of output:

| Prompt | Prompt Processing | TG after it |
|---|---|---|
| 1M tokens | 97.3 s (10,276 t/s) | 105.5 t/s |
| 2.05M tokens | 225.0 s (9,112 t/s) | 105.0 t/s |
| 4.15M tokens | 552.6 s (7,509 t/s) | 95.8 t/s |

  Under 1M, against a build with a 1M window only: outputs bit-equal, speed equal.

**Larger windows only when needed.** A start reserves only what the 1M window needs; the 2M and 4M windows are made from the
store when a request first needs them and given back when the store is short.
- Measured: a 4M start's store is as large as a 1M start's; a 1.57M-token request opened the 2M window in 0.43 s.

**Window sizes take turns.** A step runs one window size, so each run starts with the first waiting request of another size.
- Tested on a host model of the run loop (`turns_test`): a 1M request beside two 4M requests runs in 50 of 100 runs, where it
  ran in none.

## The conversation cache

The SSD tier and keeping conversations across a planned stop are pico-dgx-ds4's; these are this repository's changes to them.
[`docs/split.md`](docs/split.md) explains the whole cache.

**The SSD tier piece by piece.** Each piece of a conversation is written to the SSDs once, in the background and off every
request's path; eviction takes the last piece first, and a piece already on disk leaves memory without a write.
- Settings: `--disk-gib` (served: 512), `--keep-days` (30).
- Measured: a 60K-token conversation evicted and read back twice: 117 MiB came back in 25.2 and 26.0 ms, bit-equal; the second
  eviction wrote nothing.

**A quarter on each SSD, and the split cache.** Each Spark writes its own quarter of every piece to its own SSD. With the split
cache on, an idle conversation's written pieces are marked free where they lie, with no copy: if it returns before the memory is
taken it costs nothing, otherwise the four Sparks read their quarters and exchange them. Memory holds a conversation whole or
not at all.
- Setting: `--kv-split off|tiered|max` (served: tiered).
- Measured on a test build of the split cache: a conversation of 64 pieces came back from the four SSDs in 28.7 to 29.4 ms
  (about 0.46 ms a piece), resuming at its last position.

**Read-backs admitted on what they need.** A conversation coming back from the SSDs is admitted on the memory it actually lacks.
- Measured: requests that had waited without end under the old count are admitted; 20 read-backs in a row each equal a cold start.

**The SSD tier's size and expiry changed while serving.** From the dashboard; a smaller size gives the disk space back at once.
- Measured: grown from 512 to 768 GiB while serving.

**Crash checkpoints.** While the engine is idle it saves the idle conversations as a planned stop does, so a crash comes back to
the last checkpoint; the front-end saves its session table the same way.
- Setting: `--ssd-flush-s` (30: the least time between two checkpoints).
- Measured: after rank 0 was killed and restarted, every conversation idle at the last checkpoint gave bit-equal outputs.

**Kept conversations in the served settings.** pico-dgx-ds4's keeping of conversations across a planned stop is switched on.
- Setting: `KEEP_GIB` (served: 64).
- Measured: the next turn of a 64K-token conversation after a planned restart: first token after 0.23 s, where the same prompt
  cold took 6.55 s. A new engine binary cannot read the kept conversations, so an upgrade starts them cold.

**A request sent again gets its first reply, bit for bit.** When a prompt the engine holds comes again (a retry, a regenerate),
the positions after its restart point run as one decode step when there are at most `--small-rows` of them; the step adds
in another order than the prefill, so the state is not the cold one and a near-tie can change the reply. With the switch they
take the prefill path. Based on pico-dgx-ds4's a40eeb8 by Pico.
- Setting: `DSGB10_REUSE_EXACT=1` (off unless set).
- Measured: a re-sent prompt's first reply bit-equal to the first send's (without the switch 32 of 32 log probabilities differed,
  with it 0); its first token 10 to 20 ms later (median at 2K tokens 51 against 72 ms, 8K 47 against 57 ms, 32K 56 against 73 ms);
  first sends unchanged.

## Prompts

**Interleaved prefill.** A long prompt is processed in pieces between the decoding users' steps, so they keep receiving tokens;
a short prompt that arrives meanwhile starts beside it and goes first. All of it changes live on the dashboard.
[`docs/interleave.md`](docs/interleave.md)
- Settings: `--interleave-gap-ms` (600: the longest a decoding user waits), `--interleave-share` (40), `--no-interleave`.
- Measured with synthetic token ids through rank 0's socket: one user decoding beside a cold 200K-token prompt keeps 79.3 t/s with a longest silence of 0.60 s; without it the
  user stops for 20.2 s. In the served build, beside 4 decoding users the longest silence was 0.6 s.

**Long prompts processed 16,384 rows at a pass.** A second set of prefill buffers for 16,384-row passes is made from store memory
that is free, never by moving a conversation, and is the first thing given back when the store runs short; without it the
engine processes 8,192 rows a pass.
- Setting: `--prefill-rows-max` (served: 16,384; 0 off).
- Measured against the same build without it, a clean boot each. Prompt Processing of a single cold prompt is 9 to 10 % faster
  from 28K to 1M tokens, 8.6 % at 2.05M and 7.2 % at 4.15M:

| | Without | With |
|---|---|---|
| 1M-token prompt | 106.9 s (9,357 t/s) | 97.3 s (10,276 t/s) |
| 2.05M-token prompt | 244.3 s (8,391 t/s) | 225.0 s (9,112 t/s) |
| 4.15M-token prompt | 592.5 s (7,005 t/s) | 552.6 s (7,509 t/s) |
| A prompt of about 785K tokens beside 4 decoding users, first token | 151.1 s | 139.9 s |
| A prompt of about 170K tokens beside 4 decoding users, first token | 51.0 s | 53.7 s |

  TG at 1, 16 and 32 users equal; after the 4.15M prompt 95.8 against 103.6 t/s (one run each). Outputs bit-equal.

**pico-dgx-ds4's balanced reduce-scatter and prefill tuning, switched on.** `DSGB10_RING_RSBAL=1` spreads the prompt exchange's reduce-scatter
evenly over the network card's two PCIe functions (its two rails); `DSGB10_PF_TUNE=hc,small,down` picks the prefill kernels' shapes by row count. Both are pico-dgx-ds4's code
(pico-dgx-ds4 `f2bafa4` and `a40eeb8`), each in its own commit authored by Pico; this repository switches them on.
[`docs/ring.md`](docs/ring.md)
- Measured, one setting at a time on the same build: the tuning, 16 users 4.0 % faster (334.0 to 347.2 t/s); the balanced
  reduce-scatter, equal at 16 users, and a long prompt 2.3 % shorter on the engine's own clock (not a clean boot). Outputs
  bit-equal.

**A clock for the prompt passes' waits.** `DSGB10_PF_STALL=1` times how long each prompt pass waits at its two joins with the
exchange stream (after the attention's all-to-all and after the MoE's last column group) on layers 0 to 19, and rank 0 prints one
line after each request. Off by default; when off no timing event is made. [`docs/ring.md`](docs/ring.md)

## Weights and kernels

**Weights loaded into place.** Each Spark loads its own quarter of the gathered weights straight into its slot of the whole
buffer, so the separate copies are gone.
- Measured: 0.75 GiB more store on every Spark; outputs bit-equal.

**The embedding cut four ways.** Each Spark keeps a quarter of the embedding table, by vocabulary. The rank files are version 3:
cut them again with `dsw_convert`.
- Measured: 0.98 GiB more free memory on every Spark; outputs equal.

**Rotary tables computed, not stored.** Each rotation is computed from its position and factor, so the table and its copies are
gone, and the 2M and 4M windows need no tables of their own (0.5 and 1 GiB a Spark).
- Measured: 0.25 GiB a Spark not taken; the computed values equal the table's bit for bit.

**The decode index scan split four ways.** Each Spark scans a quarter of a conversation's index keys and the results are merged.
- Setting: `--pscan off|ext|all` (served: all).
- Measured (rank 0's step time at 1M tokens): 3 % shorter at 2 rows a step (22.94 to 22.26 ms), equal at 6 rows and under 64K;
  the picks equal the whole scan's in 72 of 72 cases.

**Prefill MoE up kernel with a mover warp.** One warp moves the data and four compute. [`docs/kernel-trims.md`](docs/kernel-trims.md)
- Measured: the kernel 7.2 % shorter at 8,192 rows; bit-identical.

**Prefill MoE down kernel with a mover warp.** The same for the down kernel. It is not used in the served settings, where pico-dgx-ds4's
tuning picks pico-dgx-ds4's column-group down kernel, which measured better.
- Measured, without that tuning: the kernel 16 % shorter at 8,192 rows, Prompt Processing 0.7 % faster; bit-identical.

**Smaller loops in the prefill MoE kernels.** The down kernels expand the weights first, and the 4-bit expand reads its half of a
word directly.
- Measured: the routed down kernel 1.2 % shorter; bit-identical.

**Decode attention prefetch.** The next block's keys are fetched while the current block is scored.
- Measured: decode attention 10 to 27 % shorter on one Spark; bit-identical.

**Decode matrix-vector kernels fetch earlier.** The kernels right after an exchange (`wq_a` with `wkv`, and the router's) request
their second batch of weights while the exchange is still running.
- Measured: a decode step at one user 0.7 to 0.9 % shorter; outputs bit-equal.

**The certified head for sampled requests.** pico-dgx-ds4's certified LM head (it reads a coarse copy of the head and computes the exact
one only where the winner is in doubt) also serves requests sampled at temperature up to 1 with no top-k, top-p or log
probabilities. With the served default of top_p 0.95, requests use it only when they set top_p to 1.
- Measured: a decode step at one user 0.7 % shorter; checked against the full head on 43,041 rows, 0 differences.

**Wide steps: the shared expert beside the router, a lighter all-reduce, the head on rows normalised once.** Three switches for
steps of 17 to 64 rows, each off unless set, each giving the same bits as without it. `DSGB10_WIDE_SH=1` runs the shared expert on
the side stream while the gate, the router and the routed experts' first half run. `DSGB10_WIDE_AR=1` makes the attention's and
the FFN's all-reduce a reduce-scatter and an all-gather (each Spark sums one quarter) in place of every Spark receiving the other
three partials whole; `2` also rounds the FFN's reduced quarter to bf16 before the all-gather, so it carries 2 bytes an element in
place of 4. `DSGB10_HEADG=2` normalises the rows once for the wide head in place of in every CTA of the head kernel. Based on
pico-dgx-ds4's a40eeb8 / f2bafa4 by Pico. [`docs/batching.md`](docs/batching.md#wide-steps-three-switches)
- Setting: `DSGB10_WIDE_SH=1`, `DSGB10_WIDE_AR=1|2`, `DSGB10_HEADG=2` (on every Spark).
- Measured: 16 users 6.0 % faster, 16 users on three shared prompts 7.1 %, 32 users 4.9 %; 2 to 5 users unchanged (within 0.3 %);
  each switch and all three bit-equal to none (the samples hashes at 18, 32 and 64 rows).

**Exact shared steps: each row of a step of 9 to 64 rows computes what it computes alone.** A step shared by several requests
sums a row's values in another order than a step of 1 to 8 rows does, so a greedy or seeded request beside others got other bits
than it gets alone. `DSGB10_NORM_EXACT=1` makes the RMSNorm launches add a row's squares in the fused kernels' order.
`DSGB10_WIDE_EXACT=95` runs the projections, the gate, the MoE and the Engram's GEMV of a step of 17 to 64 rows through row-group
kernels: each weight is read once, and each row's outputs are those of a call of at most 16 rows. `DSGB10_WIDE_EXACT_MT=2` sets
wo_b's tile. With `DSGB10_OEA` set, a step of one session's rows is then never re-routed. Every default is the behaviour
without them. Based on pico-dgx-ds4's `0580fe6` by Pico (its kernels and tests; the copy drafting of that commit is not taken).
[`docs/batching.md`](docs/batching.md#exact-shared-steps)
- Setting: `DSGB10_NORM_EXACT=1`, `DSGB10_WIDE_EXACT=95`, `DSGB10_WIDE_EXACT_MT=2` (on every Spark; a value outside its range stops the start).
- Measured: in the teacher-forced tool every shape of 12 to 64 rows gives each session its hash alone; through the service a request
  beside others gave its tokens alone (200 of 200) and its log probabilities (0 of 200 different), the abliteration included. With
  the draft gate's cost fits of the plain bodies: 16 users 8.5 % slower, 32 users 7.3 %, 2 to 5 users 1.0 to 1.7 %.

## Drafting

Every drafted token is checked by the model before it counts, so the output is the model's own. [`docs/batching.md`](docs/batching.md)

**Copy drafting.** When a conversation's last 8 tokens appeared earlier in it, the tokens that followed are offered as extra drafts.
Agents editing code repeat much of what they read, so these drafts are often right.
- Setting: `DSGB10_COPY=1` (served: on).
- Measured on an earlier build with a table fitted from real traffic, on rank 0's own clock: agent edit sessions 3.6 % faster
  (inside the spread), one user and 16 users unchanged; outputs bit-equal.

**Copy drafting that learns from the traffic.** How many copied drafts a step checks is learned from which ones were accepted,
for each match length and for each conversation, the same on every Spark.
- Setting: on with copy drafting (`DSGB10_COPY_ADAPT=0` turns it off).
- Measured: equal to the built-in table alone on every bench (within 1.5 %), with no table to fit.

**Drafting for several users.** In steps shared by several users, pico-dgx-ds4's confidence network estimates each draft's chance, and
drafts are kept in order of expected accepted tokens per millisecond of step time.
- Settings: `DSGB10_BATCH_CF=1`, `DSGB10_BATCH_TRIM=1` (served: on).
- Measured: 2 to 5 users at temperature 1.0, on average 0.8 % faster (inside the spread), from 5.3 % slower to 9.6 % faster.

**Drafts chosen with the request's noise.** A sampled request without a client seed draws through the Gumbel-max sampler, which
gives the same distribution as before, and the drafter's Markov chain picks each draft with the noise of the position that will
verify it, so a draft is accepted whenever the sampler would pick that token. Greedy requests and requests with a seed are
untouched. The settings are checked to be one set on the four Sparks at the start.
- Settings: `DSGB10_GSAMPLE=1`, `DSGB10_GCOUPLE=0.8` (off unless set; the certified head keeps serving a flagged request that samples
  with the temperature alone).
- Measured: sampled at temperature 1.0 and top_p 0.95 with the same seeded requests in both runs (`DSGB10_GSAMPLE=2`), the coupling
  against none: one user 80.0 against 80.9 tok/s, 16 users within 0.5 %, drafts accepted per verify step 2.19 against 2.20 (no
  speed difference). The Gumbel-max draws equal their replay on 8,360 of 8,360 rows.

## Images and video

[`docs/vision.md`](docs/vision.md) explains the whole change, with the path of an image and a clip; [`docs/media.md`](docs/media.md)
the media worker.

**The image encoder inside the engine, split four ways.** Each Spark holds a quarter of the vision tower's weights and runs its share of every image over the ring, in units
that fit between the decoding users' steps.
- Setting: `--vision-weights` (each Spark's quarter, made by `dsw_convert --vision`).
- Measured through the API with 8 users decoding (medians of five), against pico-dgx-ds4's separate encoder (two builds:
  [`docs/vision.md`](docs/vision.md#measured) says which):

| | Separate encoder | Inside the engine |
|---|---|---|
| One image, first token | 0.737 s | 0.443 s |
| 32 video frames, first token | 12.56 s | 10.20 s |
| Users' longest pause during the 32 frames | 2.01 s | 0.58 s |
| Users' longest pause during one image | 0.33 s | 0.51 s |

  Every image's rows equal the unsplit encoder's, byte for byte.

**The separate encoder no longer started.** With the engine encoding, rank 0 runs no encoder container (`--vision container`
still selects it).
- Result: rank 0's memory counted outside the store is the media worker's 512 MiB cap instead of the container's 1,260 MiB cap.

**Pixels in the request, read by a media worker.** A small worker with no model and no GPU (Pillow, numpy and PyAV) fetches and
decodes images and clips; the front-end sends their pixels to the engine, which checks them before it keeps anything.
- Settings: front-end `--media-addr` (served: 127.0.0.1:18951); engine `--media-mib` (512).
- Measured: the worker's pixels equal the checkpoint's reference bit for bit on 11 images in 4 formats.

**Image rows in the store.** A request's image rows live in store chunks on every Spark from its first pass until it ends, so a
resumed request encodes nothing again.
- Measured: first token equal (+0.5 %); a cancel or a client that leaves gives every chunk back.

**Smaller loading of the vision weights.** The loader sizes its staging to its reads.
- Measured: loading the vision weights costs the store 104 chunks, where it cost 292.

**Image conversations reuse their cache.** Images are keyed by content and place, so a later turn processes only its new tokens.
- Measured (with the separate encoder): the second turn of an image conversation answered in 0.25 s, the first in 1.40 s.

## The dashboard

`/dashboard` on the front-end. [`docs/dashboard.md`](docs/dashboard.md)

**A live status page.** The decode and prefill rates; nine tiles (first token, context cache, max context, conversations, shared
prefixes, Engram cache, lent memory, SSD cache, cache hits); decode, prefill and cache charts; one row a request with its phase
and progress; one row a Spark with memory, temperatures, GPU, SSD wear, both links and fans; and the settings that change live.
- Measured with the first version of the page on an earlier build: one user 79.3 t/s with it open, 79.8 with it closed.

**Sign-in.** Every data route needs the front-end's API key; the page asks once and keeps it until SIGN OUT. Settings changes
must be JSON, so another site's form cannot change them.

**COOLING AND CLOCK.** One fan profile for all four Sparks, and each Spark's GPU clock limit at 2,400, 2,200, 2,000 or 1,800 MHz,
never unlocked. Needs the external tool `dgx-fan-fleet` (`--fan-tool`).

## Serving and operating

**Runtime abliteration as a second model name.** The engine serves `NAME-ablit` beside `NAME` from the same weights: the published
abliteration edit (by mrfakename, for this checkpoint) is applied at run time, only to the requests that ask for that name, and both
names run in one batch. The file is in this repository: `ds41f-4xgx10-switchless/tools/ablit/ablit.bin`.
[`docs/ablit.md`](docs/ablit.md)
- Settings: engine `--ablit FILE`; front-end `--no-ablit-alias` serves the plain name only.
- Measured: the plain name's outputs are unchanged; 16 users at the same speed with and without the edit (321.9 t/s).

**The served front-end settings.** For a request that sets none: thinking on at effort 75, temperature 1.0, top_p 0.95, and an
output cap of 1,048,576 tokens (the code's default of 32,768 cuts long outputs).

**A served installation in `deploy/`.** The settings file, the systemd units for the engine, front-end and media worker, and
`pico-ctl.sh`, which drops every Spark's page cache before a start, refuses a start when a Spark has too little memory free, and
sets the GPU clock limit on every Spark. [`docs/setup.md`](docs/setup.md) goes from a clone to a running service;
[`docs/operations.md`](docs/operations.md) covers start, stop, upgrade and rollback.

**The network, reproducible.** The switch's configuration (RoCE with flow control on one priority) and each Spark's flow control at
boot. [`NETWORK.md`](NETWORK.md)
- Measured: three Sparks sending to one at line rate, engine stopped: 0 drops, 0 retransmissions.

## Fixes

Fixes to code this repository builds on; each was shown failing before the fix.

| Fix | Before | After |
|---|---|---|
| A request that ends inside a batch gets its end at once | beside two long outputs, a 20-token answer came back after 12.1 s | 0.35 s |
| A client that leaves ends its request wherever it waits | a queued or non-streaming request ran to its end for nobody | it ends at once, streaming or not |
| Engram look-ahead used only by its own conversation | a prompt cancelled between pieces could give another conversation a wrong output (4 of 5 in the test) | 5 of 5 equal |
| A full session store keeps every restart point | under a squeezed store, agents resumed from nothing or an old point | every agent resumes at its end |
| Batch cancel marks index the batch's entries | a request on line 31 would have paused the whole batch | marks only its own entry |
| The abliterated name applies in steps shared with other drafting sessions | beside another drafting session, a `NAME-ablit` request ran the published checkpoint from its second token on | the same tokens as alone, in either order (the batch spec steps carry each session's abliteration strength) |
