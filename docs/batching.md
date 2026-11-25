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
