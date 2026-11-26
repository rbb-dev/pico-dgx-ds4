# Batching and drafting: this repository's changes

pico-dgx-ds4's [batching.md](../ds41f-4xgx10-switchless/docs/batching.md) describes how the engine this repository builds on runs
several sessions in one step; this page describes what this repository changed or added in it. The served settings are
[`deploy/pico.env.example`](../deploy/pico.env.example).

## A request that ends inside a batch

A request whose output ends while others in its batch go on gets its DONE at once, from the loop's callback (the front-end
completes a response only on DONE). The run goes on for the others and finishes that request's session at the run's end; a new
request for the same session waits for that, and its arrival pauses the run. Measured on the four Sparks through the front-end:
a 20-token response beside two long generations returns in 0.35 s.

## Long prompts beside decoding sessions

A long prompt is processed in pieces between the decoding sessions' steps, and the decoders get a set share of the time between
the pieces: [interleave.md](interleave.md).

## The batch draft count's survival

A one-session step decides how many drafts to verify from the confidence MLP's survival (the drafter head's logits, the
top-1 margins and log p over the whole vocabulary, a hidden layer of 64), which is calibrated. A batch step's draft count
(`k_spec_gate_batch_all`) reads the drafter head's survival unless `DSGB10_BATCH_CF=1`, and that survival runs low at
depths 3-5: on 5,151 batch entries of 2-5 coding sessions it predicted 0.817 / 0.784 / 0.727 where 0.829 / 0.806 / 0.764
landed. The MLP's log-loss is lower at every depth, on every prompt set and at every session count
(`tools/gate/gate_fit.py`, each set held out in turn).

Three switches, each off unless set:

| switch | what it does |
|---|---|
| `DSGB10_BATCH_CF=1` (with `DSGB10_CF_MLP`) | a batch pass runs the MLP over its entries' rows (`cf_scan_kernel`, one all-gather of 16 bytes a row, `cf_mlp_batch_kernel` on the one-session MLP's device function) and the batch step's draft count reads that survival |
| `DSGB10_BATCH_TRIM=1`, `DSGB10_BATCH_TRIM_COST=a,b` | the tokens-per-ms trim of the wide steps also within 16 rows, with a step costing a + b x rows ms (19, 2.9 measured for batch steps) |
| `DSGB10_BATCH_CAL=a1,b1,..,a5,b5` | each depth's chance given the one before through sigmoid(a x logit + b) |

`DSGB10_GATE_DUMP=1|2` with `DSGB10_GATE_DUMP_FILE` writes one CSV line per verified batch entry (the drafts verified and
accepted, the three survivals, the MLP's inputs); 2 verifies every draft the row budget allows, so depths the draft count would
drop are observed too. `tools/gate/collect.sh` runs the sessions and cuts the lines per run; `compare_dumps.py` compares
two runs' response hashes.

**Measured** on an earlier build of this engine (four Sparks, each setting from a clean boot of
all four, GPU clock 2,200 MHz, the dashboard open; that build's served settings, with a copy table fitted from real traffic,
with the MLP survival and the trim at 19, 2.9 against the same build with both switches off;
`tools/gate/bench_seeded.py distinct N`, 2-5 sessions on three sets of coding prompts, temperature 1.0, 3 runs for each set and session count,
2 for 2 sessions on set 0 with the switches on, where a third run overlapped another request and is left out; the runs for
each set and session count agree within 2 %, except 4 sessions on set 3: 3.4 % and 5.2 %). Total t/s, off -> on:

| sessions | set 0 | set 3 | set 6 |
|---|---|---|---|
| 2 | 154.5 -> 153.2 (-0.9 %) | 167.1 -> 183.2 (+9.6 %) | 169.6 -> 160.7 (-5.3 %) |
| 3 | 186.3 -> 180.6 (-3.1 %) | 184.1 -> 188.8 (+2.6 %) | 184.5 -> 182.0 (-1.4 %) |
| 4 | 196.3 -> 194.8 (-0.7 %) | 187.0 -> 193.4 (+3.4 %) | 189.2 -> 194.4 (+2.7 %) |
| 5 | 200.8 -> 202.8 (+1.0 %) | 203.1 -> 203.2 (+0.0 %) | 195.1 -> 198.1 (+1.5 %) |

On average +0.8 % across the twelve combinations of session count and prompt set; they range from -5.3 % (set 6, two sessions) to +9.6 % (set 3, two sessions). Each
setting is one boot, so a change includes boot-to-boot variation: between two boots of builds that differ only in copy
drafting, which these prompts barely use, this bench moved -3.6 % to +4.7 % (mean +0.1 %).

**Exactness.** One session never runs a batch pass: its responses are bit-equal with the switches on (greedy, two prompts).
With every switch off the batch steps are bit-equal at 2, 3 and 5 sessions to a build without these code paths. With the
switches on, the batch steps' verified tokens are the target model's, but a step's rows are grouped differently and this
engine's batch logits depend on that grouping (as for the row budget and the wide steps): greedy responses of 2-3 sessions
differed from the run without the switches in 9 of 10 cases and were equal in 10 of 10 at 5 sessions, while the same build
run twice gave the same responses (all but one 8-token greeting).

## Sampled requests: the Gumbel path and coupled drafts

A sampled request without a client seed can draw through the Gumbel-max sampler, and its drafts can be chosen with the request's
own noise. Both are off unless set (every rank must have the same environment: a start where the settings differ stops).

| switch | what it does |
|---|---|
| `DSGB10_GSAMPLE=1` | a sampled request that sent no seed, and a sampled request without top-p or top-k, is on the Gumbel path. A request with a seed and a top-p or top-k keeps the sampler and the replies it had |
| `DSGB10_GSAMPLE=2` | every sampled request is on it; a request with a seed, top-p or top-k then gets another text for its seed than without the switch, the same one every time from then on (for tests) |
| `DSGB10_GCOUPLE=Td[,Td2,..]` | sessions on the Gumbel path have the drafter's Markov chain pick each draft as the largest of its logit / (T x Td) plus the request's noise at the position that will verify it, Td in (0, 4] per depth (the last repeats). Needs the switch above to have any effect |
| `DSGB10_GCOUPLE_CF=A,B` | with coupling, the confidence network's logit of a coupled session becomes A x logit + B (default 1,0: unchanged) |

On the path, a truncated row (top-p or top-k) takes the Gumbel-max winner among the same kept tokens as the walk over one
uniform: token i wins with probability exp(l_i / T) divided by the sum over the kept tokens, so the distribution is the same and
the draw for a given seed is another one. An untruncated row already drew this way. The noise is a hash of the seed, the position
and the token id, so a draft chosen with the noise of its verify row is accepted whenever the sampler's own pick is that token.

**Exactness.** Greedy requests are never on the path, and the requests a mode leaves off it are bit-equal to a build without the
switches. For a request on the path, drafts never change what a one-session run emits: with a given noise the same tokens come
out with any drafts; only how many are accepted changes. In a step shared by several sessions the tokens are the target's under
that step's grouping, as for the other batch switches, and the coupling adds one dependence: which sessions' rows share a step
depends on the coupled drafts, which depend on the noise of the very positions the step verifies. Each token is still the
sampler's exact pick from a valid evaluation of the target, but the choice among the valid evaluations (the groupings whose logits
differ by the batch non-invariance above) is no longer independent of that position's noise. No direction of bias follows from
the code. A shared step with a session on the path is therefore not validated by comparing its text with another run's; only steps
with no session on the path are.

## Copy drafting

A second source of drafts for the speculative steps. Agent turns copy: a tool's input repeats text the conversation
already holds (an edit's old text, a path, a block of code that was read). The tokens that followed the **latest
earlier occurrence** of the conversation's last 8 tokens are appended to the DSpark chain's drafts as extra rows of the
same step, and the target model verifies them exactly as it verifies the chain's. Only which drafts are verified
changes, never a token.

**Offline, on recorded transcripts (no engine, not a speed measurement):** 31 % of the assistant's output tokens sit
inside right copy runs of 8 or more tokens (tool inputs 36 %, an Edit's old text 72 %, prose 3 %); the line-number prefix
of a Read result makes no difference to that.

**Measured** on an earlier build of this engine (four Sparks, each setting from a clean boot, GPU
clock 2,200 MHz, the dashboard open, that build's served settings with copy drafting on and with it off; pico-dgx-ds4's agentic edit bench `bench/editbench.py` at temperature 1.0, thinking off
and high; rates from rank 0's own per-request lines): one session 141.1 against 136.2 t/s (+3.6 %), 3.63 against
3.34 drafts accepted a step; one user (chat 83.8 against 83.4, code 146.4 against 146.4 t/s) and 16 users (336.9
against 335.6 t/s) unchanged. The table for these runs was fitted from a
sampled dump run of the same bench: a copy's first token right 87 / 92 / 93 / 98 % of the time for matches of 8-15 /
16-31 / 32-63 / 64+ tokens.

**Where it runs.** On the device, inside the step graph, on all four ranks, because the rows of a step are chosen on the
device while the host is already a step ahead. Each session unit keeps a mirror of its token history (int32 by position)
and a direct-mapped table keyed by a hash of 8 consecutive tokens, holding the end of the latest occurrence. A spec step
appends the tokens it emitted (the rows it accepted, then the bonus) and indexes the grams that end before the bonus; the
gram that ends at the bonus waits for the next step, so the lookup of the bare tail finds the occurrence before it, never
itself. A lookup takes the last 8 tokens of the history followed by the chain's kept drafts, reads the table, **compares
the 8 tokens at the candidate** (a collision or a stale entry is a missed copy, never a wrong one), extends the match
backward by up to 64 tokens for its length, and proposes the tokens after the source: at most 7 drafts in all, never
past the known history, ending at a token that is not an id (an image position holds -1). Every input is an integer every
rank already holds (sampled tokens, the chain's draft ids, the prompt's broadcast), visited in a fixed order, so the
four ranks choose the same rows with no traffic added. Copy drafting asks the front-end for the whole prompt on every
request (the history before the cached prefix may belong to another conversation).

**How many copy rows are kept.** Copy rows go after the chain's, never instead of it: on recorded agent transcripts replayed offline, copy
rows alone, or either source alone, verified fewer tokens a step than both together. Row j after the chain's k kept drafts
has the survival `chain_survival[k-1] x T[bucket][j]`, `bucket` the match length (8-15, 16-31, 32-63, 64+) and `T` the
chance that the first j copy tokens are all accepted. A one-session step then keeps copy rows by the step's own rule
(threshold, ratio or marginal; the step-time table `--spec-row-ms` is extended to 6 and 7 drafts by its last difference). A
batch step lists each entry as chain rows then copy rows (survival non-increasing), so the row budget and the tokens-per-ms
trim treat a copy row as any draft and drop copy rows first; batch copy rows need the row budget (`DSGB10_BUDGET=1`). `T`
has a built-in prior, derived from a greedy replay, that runs high under sampling. The built-in table is the default (no file is
shipped); `tools/copy/fit.py` makes a fitted one from the dump's data (below, `python3 tools/copy/fit.py DUMP.csv -o copy-T.txt`), and
`DSGB10_COPY_TABLE` names it. The same file must be at the same path on every Spark: a Spark without it stops the engine at its start
(`copy: DSGB10_COPY_TABLE=... cannot read it on this rank`), and neither the start helper nor the install copies it.

**Learning from the traffic** (on with copy drafting; `DSGB10_COPY_ADAPT=0` turns it off). A table fitted once says nothing about a
different model, sampling setting or client. So the table is the starting point and the engine keeps counts. One step after copy rows are proposed the
engine knows how many of them were accepted, and the token it would have used after them is the sampler's own: so every proposal
whose draft rows all landed is scored on the copied tokens that landed and on the first one that was not checked, whether
or not the step checked it (a copy chance that has fallen low keeps getting data, so it can climb back). The counts are kept
for each match-length bucket and for the first, the second and the later copied tokens; the starting value counts as 64
outcomes, and a count holds its last 512 to 1,024 outcomes. Each conversation also keeps how often its own copies fail compared
with what the counts expected of the same proposals, and scales the chances by that ratio (between a quarter and 8 times), so a
conversation whose copies fail stops spending rows on them quickly without lowering the chances for the others, and one that stops
failing gets them back. Until the first outcome the rows chosen are exactly the table's. Nothing learned is kept across a start. The
counts are integers updated in the same order on every Spark, so the four Sparks' chances are the same without anything sent. Only
which rows are checked changes, never the tokens. It cannot run with `DSGB10_RELAX` (a relaxed accept would make the outcomes read
low), and every Spark must have the same copy settings and the same table file: the engine compares them at its start and stops if
they differ. `tools/copy/fit.py` stays the offline way to check what the engine learned: it scores a dump of the same run.

Measured with the built-in table, learning on against the table alone (the served build's settings otherwise, a clean boot
each): every bench within 1.5 %: one user 83.1 / 145.8 against 82.9 / 146.6 t/s (chat /
code), 16 users with different prompts 338.4 against 336.3, 16 users with the same 3 prompts 371.9 against 377.6, agent edit
sessions 116.5 against 116.0 (thinking high) and 120.4 against 121.3 (thinking off). The served settings run the learner.

**Switches** (each off unless set; every rank must have the same environment):

| switch | what it does |
|---|---|
| `DSGB10_COPY=1` | copy rows on, chosen by the survival rule above |
| `DSGB10_COPY_ADAPT=0` | the survival rule reads the table alone; without it (the default) it reads chances learned from the traffic, the table being their starting values (above), whenever `DSGB10_COPY` or the dump mode is on; a relaxed accept (`DSGB10_RELAX`, `DSGB10_RELAX_P`) keeps the learner off unless it is set to `1` |
| `DSGB10_COPY_DUMP=1` | the dump mode: the matcher runs and logs, adds no rows (responses and step rows equal a run without it) |
| `DSGB10_COPY_DUMP=2` | also verifies every proposed row (data for depths the rule would drop; implies rows on) |
| `DSGB10_COPY_DUMP_FILE=FILE` | the dump's CSV, one line per step and entry (rank 0's server): the step's emitted tokens, the chain's kept drafts, the proposal (source end, match length, tokens); `DSGB10_COPY_TAG` marks the run |
| `DSGB10_COPY_MIN=N` | the shortest match a proposal needs (8 to 72; default 8) |
| `DSGB10_COPY_TABLE=FILE` | T: 4 lines (match buckets) of 7 numbers in (0, 1], `tools/copy/fit.py` writes it; unset: the built-in table |
| `--copy-mib N` | the arena a rank for the histories and tables (default 512): 4 bytes a position for the mirror and a table of 4 entries a position at most; past the arena the tables shrink (older entries are displaced: a missed copy), then the mirrors, then a unit gets none. Never a failure |

With a switch set every spec step also prints, once a request ends, `copy: digest H after N steps` on each rank: the combined hash
of the rows every step chose, equal on the four ranks. With the learner on each rank also prints `copy: learn H after N steps`,
the same kind of hash over every outcome the counts took, equal on the four ranks too, and rank 0 prints per request
`serve: req N copy: P proposals made, S scored, first token right F; V copy rows verified, A accepted; this conversation misses X times the
average` (a proposal is scored a step later; a proposal whose step did not verify rows is not scored)

**What "exact" means here.** One session: the responses with `DSGB10_COPY=1` equal the responses without it at every row
count up to 8 (the rows use the decode kernels, whose per-row arithmetic does not depend on the row count): checked greedy,
39 edit turns from an empty cache; sampled with a fixed seed it holds by construction (the draw is keyed by seed and
position); no test covers it. A batch step keeps the grouping dependence described above: copy rows change how a step's
rows are grouped, so responses of 2 or 3 sessions can differ from the run without them, and text equality does not validate
a batch change; the tokens each step emits are the target's under that step's grouping. With every switch off, steps carry
no copy rows.

**Cost, as an estimate (not timed).** Per step: one hash, one table load, one tail load, up to 64 backward compares and one
load of at most 7 tokens, serial in the one thread of the kernel that picks the drafts (`k_spec_next`, and `k_spec_gate_batch_all` for a batch) (about four dependent global loads and a loop of
independent ones: a few microseconds inside a step of tens of milliseconds); the host adds nothing. Memory: 4 bytes a
position for the mirror and 16 for a table at its full size (a unit of 1M positions: 20 MiB; 16 such units: 320 MiB of the
512 MiB default, computed from `plan_units` in `engine/model/copy.h`).

## Wide steps: three switches

Three switches for steps of 17 to 64 rows (`R > kDecodeRows`), each off unless set, every one giving the same bits as the step
without it; they are read when the engine starts, so every Spark needs the same settings. Based on pico-dgx-ds4's a40eeb8 and
f2bafa4 by Pico. Each prints one line at its first use (`wide steps: ...`) so a log shows it was in force.

- `DSGB10_WIDE_SH=1`: the shared expert (`moe_prefill` phase 2, its w1 + w3 SwiGLU GEMM and its w2 GEMM) is forked onto the
  side stream `sb` right after `ffn_norm`. It needs only the normed, quantised input, so it runs beside the gate GEMM,
  `router_topk` and phases 1 and 4 (bucketing, routed up) on the main stream; the stream joins before phases 8 and 16 (routed
  down, combine). The kernels and their inputs are the same. The abliteration's shared-expert term reads the shared expert's
  activations after the join.
- `DSGB10_WIDE_AR=1`: both all-reduces of a wide step are `comm::allreduce_bulk_bf16` (attention, bf16 partials) and
  `allreduce_bulk_f32` followed by `f32_to_bf16` (FFN, fp32 partials), then `attn::hc_post` and `attn::hc_pre`
  (`ops::allreduce_hc_bulk`), in place of `ops::allreduce_hc`, in which every Spark receives the other three partials whole. Each
  quarter is summed on one Spark in rank order and all-gathered: the same sums in the same order, the same rounding to bf16.
  `DSGB10_WIDE_AR=2`: the FFN's reduced quarter is rounded to bf16 on the Spark that sums it, before the all-gather
  (`allreduce_bulk_f32_bf16`), so the gather carries 2 bytes an element in place of 4 and the separate conversion is gone.
- `DSGB10_HEADG=2` (`1` is the default, `0` the 16-row groups): the wide head's rows are normalised once into `hfin` by
  `k_gemv16_prenorm` (one CTA per group of 16 rows, the head kernel's thread count, so each row's squares add in the same order)
  and the row-group head kernel loads them. Not in a run that dumps `final_hidden`, which uses `hfin` for the dump.

Tests (GPU rows): `kernels/tests/test_kernels.cu headrg` (the prenormalised head against the head that
normalises in every CTA and against calls of 16 rows, bit for bit, 17 to 64 rows, coded and plain);
`model/tests/coll_tests.cu wide` on the four Sparks (`ops::allreduce_hc_bulk`, modes 1 and 2, against `ops::allreduce_hc` at 17,
32 and 64 rows); `engine/server/tests/wide_test.sh` (the teacher-forced batch mode at 18, 32 and 64 rows with each switch and all
three: the samples hashes against the run with none, and each run's median step time); all run on the four Sparks and pass. Speed: 16
users 6.0 % faster, 32 users 4.9 %, 2 to 5 users unchanged.

## Long copy steps

Copy drafting appends at most 7 drafts to the bonus token of a session's step (8 rows). A request that decodes alone and rewrites text
it holds can verify far more in one step. With `DSGB10_COPY_ROWS=32` or `64` (and `DSGB10_COPY=1`) the one-session step graph holds one
more body for each of the buckets of 32 and 64 rows, captured with the exact kernels of [Exact shared steps](#exact-shared-steps) in
force whatever `DSGB10_WIDE_EXACT` says, and a step of 29 to 32 or 57 to 64 rows takes its bucket's body; steps of 1 to 8 rows keep
their bodies (certified or plain). With `8`, the graph is the one without long steps, node for node. Each body ends with its own sampler
launch (the exact ones' grid is 64 blocks), the draft module accepts a long step's last 8 final rows first and the earlier ones in a
loop of chunks of 8, and the three ratio-2 compressor carries, which keep a call's last 16 rows by position, get the step's last 16
final rows back from the projections the body saved (a rejected row's projection must not stay in them).

The rows are chosen on the device, in `k_spec_next`, by a cursor kept per unit (`model/copy_run.h`): `cur`, the mirror index of the
token predicted for the next position, and `len`, the tokens that matched behind it. After each step the tokens it emitted (its
accepted drafts, then the bonus) walk the cursor while each equals the source's token; with no cursor the matcher looks up the bare
tail. A ladder of three rungs (7, 31, 63 drafts) climbs by one after a step whose drafts were all accepted and whose tokens all
walked the cursor, and enters from the first rung only when the match is at least `DSGB10_COPY_ENTER << fails` long; any other step
sets it to the first rung. A long step not accepted whole adds one to `fails` (at most 3); one accepted whole clears it. A long
step's drafts are the source's tokens alone (no chain drafts, no short proposal), cut at the first token that is not a real id and
fitted to a body that exists (`copy_fit`: never 9 to 16 rows, never a bucket without an exact body). The learner's record of a
long step is void, so its counts keep describing proposals of at most 7 rows. The cursor carries the position it was left at: a step
run by another graph (the unit rode a batch step) does not match, and the cursor is dropped. A request's first step starts it over.

Long steps run for a request that decodes alone; in shared steps copy rows are at most 7 as before. The host's bounds
(`verify_reach`) widen to 63 rows, the draft rate and the step-cost model are not fed by long steps, and each request logs one line
beside the digests: `copy: long steps N (whole W), drafts D, accepted A`.

Settings (on every Spark; a value outside its range stops the start): `DSGB10_COPY_ROWS` 8 (no long steps), 32 or 64 (the default; 32: the 32-row body,
31 drafts; 64: both bodies, 63 drafts), `DSGB10_COPY_ENTER` 8 to 72, default 32. Memory: one more body each (a graph of the one-session
step is about 33 MiB a body, the start prints the figure) and 0.75 MiB of saved projections. Test-only: `DSGB10_TF_SOLO=1` with
`DSGB10_TF_SESSROWS=1` runs the tool's teacher-forced given steps of more than 16 rows as an exact body. Based on pico-dgx-ds4's
`0580fe6` by Pico.

Tests: `copy_run_test` (host: the cursor, the ladder, `copy_fit`, the settings, each with its wrong-rule control); GPU rows in `engine/server/tests/long_copy_gpu.sh` (the kernels' `rg` section, the tool's hash at 1, 8, 32 and 64 rows, the service's
rewrites with the setting unset and on, the speed workloads); the kernels, the tool's hashes and the greedy rewrites run on the four
Sparks and pass. Speed: rewrites of 1K, 4K and 8K tokens 47 %, 41 % and 37 % shorter.

## Exact shared steps

A request alone steps 1 to 8 rows at a time. Beside others it rides steps of 9 to 64 rows, where a row's values are added in
other orders (the RMSNorm launches' thread count, the 16-row kernels, and for 17 to 64 rows the prefill GEMMs and `moe_prefill`),
so a greedy or seeded request got other bits than it gets alone. Three settings make a row of a shared step compute with the
arithmetic of the 1 to 8-row kernels. Each defaults to the behaviour without it (the step graphs are then the same ones, node for
node: every branch is guarded by the setting and the new kernel arguments default to the old call); each is read at start, a
value outside its range stops the start (`model/exact_rule.h`), and each prints one line when on (`exact shared steps: ...`). Every
Spark needs the same settings. Based on pico-dgx-ds4's `0580fe6` by Pico.

- `DSGB10_NORM_EXACT=1`: the three separate RMSNorm launches of a layer (attention input, query, FFN input) of a step of 9 to 16
  rows, and of 17 to 64 rows when `DSGB10_WIDE_EXACT` is set, use the thread count of the fused kernels of a step of at most 8
  rows (`32 * wn * wk` of their `GemvCfg`) in place of D / 8. A row's squares then add in the same order. With a `DSGB10_OEA`
  set, a step of one session's rows is not re-routed whenever any exact setting is on.
- `DSGB10_WIDE_EXACT=bits` (0 to 95, bit 32 unused): the families of a 17 to 64-row step that run through the decode kernels'
  sums: 1 wq_a + wkv and wq_b, 2 wo_a + wo_b, 4 the router gate, 8 the MoE, 16 the Engram's GEMV. Without bit 64 each family runs
  the decode kernel in groups of 16 rows (the weights are read once per group). With bit 64 the row-group kernels
  (`gemv_fp8_rows`, `gemv_woa_rows`, `gemv_bf16_rows`, `moe_rows`) read each weight once for all the rows and give each row the
  outputs of a call of at most 16 rows. 95 is the whole exact wide body. With the abliteration on, bit 8 needs bit 64 (the
  shared expert's activations are then read where `moe_rows` leaves them).
- `DSGB10_WIDE_EXACT_MT=1|2`: wo_b's row-group kernel takes 1 or 2 tiles of 8 weight rows a warp; the same bits. `DSGB10_WIDE_EXACT_GQ`
  (0, 2, 4, plus 16) sets how many groups of 16 row slots `moe_rows` keeps in registers a pass (0: by the step's rows).

Test-only settings: `DSGB10_R16_EXACT=bits` (0 to 63) runs the families of a 9 to 16-row step in two groups of 8 rows (bit 32 the
head); `DSGB10_TF_SESSROWS=1` lets the tool's teacher-forced batch mode (`--batch-tf --batch-rows`) put up to 64 rows of one
session in a step.

Tests (GPU rows): `kernels/tests/test_kernels.cu rg` (every row-group kernel against calls of 8 rows, bit for
bit, at 9, 16, 17, 32, 48 and 64 rows; its control is a path that must differ); `model/tests/exact_rule_test` (host: the ranges and
the refusals); the GPU rows run on the four Sparks and pass. Speed, with the draft gate's cost fits of the plain bodies: 16 users
8.5 % slower, 32 users 7.3 %, 2 to 5 users 1.0 to 1.7 %.
