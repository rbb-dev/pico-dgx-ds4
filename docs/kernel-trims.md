# Kernel changes

Six changes to the prefill and decode kernels: the prefill MoE up kernel as producer and consumers, the down kernels, the down
kernel with a mover warp, decode attention key prefetch, the decode GEMVs' second weight batch, and the certified head for
sampled rows. None changes an operand, an accumulation order or a rounding, so every output is bit for bit the reference
kernel's, except the certified head, which keeps every token but not every logit (section 5). Each stands alone: the down kernels are section 1, decode attention
section 2, the up kernel section 3, the decode GEMVs' second weight batch section 4, the certified head for sampled
rows section 5, the prefill MoE down kernel with a mover warp section 6.

## Measured on a GB10 (one Spark, engine stopped, GPU at 2,190 MHz; the reference kernels and these alternated twice)

| bench | reference kernels | these kernels | |
|---|---|---|---|
| `bench_attn --detail`: attention layer 20, 1 / 2 / 5 / 10 splits | 71.4-73.4 / 40.7-41.1 / 24.2 / 20.1 us | 52.9 / 30.8-30.9 / 20.1-20.2 / 18.1 us | -27 / -25 / -17 / -10 % |
| `bench_attn --detail`: per-layer attention (layers 20, 21-23, 24-) | 20.0-20.1 us | 18.0-18.1 us | -10 % |
| `PF_M=8192 bench_kernels prefill`: routed down (+combine) | 6.424 / 6.424 / 6.425 / 6.503 ms | 6.349 / 6.361 ms | -1.2 % |
| `PF_M=8192 bench_kernels prefill`: routed up (not changed by sections 1-2) | 7.345-7.363 ms | the same | |
| `bench_chain --rows 1` / `--rows 6`, a decode step | 18.13-18.27 / 36.41-36.70 ms | 18.11-18.21 / 36.70 ms | inside the bench's spread |

`Reference kernels` in the tables are pico-dgx-ds4's kernels, built without these changes; `these kernels` are
this repository's.

Instruction counts, reference to these kernels (`cuobjdump -sass` of the built engine): `k_moe_down_p` 3,464-4,016 to 2,872-3,560 per variant
(`MOV` about 1,170 to about 470), `k_moe_down_g` 3,568-4,288 to 2,936-3,656 (`MOV` about 1,190 to 400-550),
`attn_kernel` 6,240 to 6,808 (the prefetch's loads), registers 247; `LOCAL = 0` in every changed kernel.

## 1. Prefill MoE down and fused down: loop interchange, uniform `mt`

`engine/kernels/prefill.cu`: the `compute` lambda of `k_moe_down_p` and the unrolled step loop of `k_moe_down_g`.

How it works. For each 32-k block (`u`, `blk`) the four weight operand pairs and their scale bytes are expanded once
(12 registers), then each m16 tile loads its activation quad once (four 32-bit shared loads straight into the quad
registers) and issues its four MMAs; the tile loop exits through a `break` on `mi >= mt`, uniform per CTA. One
activation quad (4 registers) is live at a time; with the weight pairs outermost, four quads (16 registers) would stay
live across them, and the compiler would rebuild the quad before every MMA (four `MOV`) and re-derive `mi < mt` after
it (`PLOP3`).

Why it is exact. An accumulator `acc[mi][j]` receives exactly one MMA per (`u`, `blk`) step, in step order, with the
same A fragment (the same shared-memory bytes), the same expanded B pair, the same A and B scale bytes. The
accumulators are independent of each other, so changing the order in which they are issued changes no value. The
shared-memory addresses are the ones the two `uint2` loads read (`+0`, `+4`, and the same at row `r + 8`).
A comment block in each kernel states this.

## 2. Decode attention: the next block's keys are fetched during this block's math

`engine/attn/sparse_attn.cu`, `attn_kernel`.

How it works. `fetch(bk)` holds the loop's gather (positions, then every key's codes and scale), and the loop is
software-pipelined: the first live block is fetched before the loop; each iteration waits at the barrier, writes
`kvalid` from the saved positions, dequantises the fetched codes into `ks` (arithmetic untouched), then calls `fetch`
for the next live block, whose loads are in flight during the scores, softmax and PV. The skip test for window blocks
past the row's window is in `skipped` / `first_live`, the same predicate as the reference's, so the same blocks run in
the same order. The codes of a compressed block (8 bytes a lane) are held in `.x`, `.y` of the same `uint4` array as a
window block's 16 bytes.

Why it is exact. Only the time at which loads are issued moves. The bytes read are the same (`win.kv`, `win.sc`,
`ckv.kv`, `ckv.sc`, `a.topk`; all read-only inside the kernel and complete before `pdl_wait`), the dequantisation
code is the same statements on them, and the QK, max, sum and PV code is not touched, so keys, scores, maxima, sums,
the 64-key block order and the split order are the reference kernel's. `kvalid` is written after the top barrier, so
the previous block's reads of it are done; `my_kp` is only overwritten after `kvalid` has taken it.

Registers: 247 (one CTA per SM by shared memory, about 86 KB), `LOCAL = 0`.

## 3. Prefill MoE up kernel: one warp moves the data, the others compute (`k_moe_up_ws`)

`engine/kernels/prefill.cu`: `k_moe_up_ws<C, KU, S, H, CS>`, chosen by `PrefillTuning::upk` (`kUpKernel` = 1 by
default; 0 runs `k_moe_up_p`). `engine/kernels/tests/bench_kernels.cu`: the `upws` mode.

Why. Nsight Compute on `k_moe_up_p` at 8,192 rows: no eligible warp in 67 % of cycles, 2 warps a scheduler (238
registers a thread), and per instruction the 4-bit unpack waiting on the weights' global loads (long scoreboard 39 %
of its samples), the MMA waiting on the unpack (wait 51 %), the two block-wide barriers a chunk stalling every warp.
The kernel runs at 79 TFLOPS, 37 % of the bench's own `mma.sync` e4m3 peak (214 TFLOPS); the dense FP8 GEMMs reach
68-74 %. In it every warp does everything: its weights go straight to registers one chunk ahead (80 registers held), so
the compiler has no room to unpack ahead of the MMAs.

What it does. A CTA is C consumer warps and one producer warp. The producer moves each chunk of 128 k (the tile's
activation rows and their scales, the consumers' w1 / w3 rows and scale words, or under the 1-bit scale code the
code words) global -> shared into a ring of 2 stages with `cp.async`, and an `mbarrier` per stage counts the copies
(`cp.async.mbarrier.arrive.noinc`); the consumers wait on the stage, compute it, and release it on a second
`mbarrier`. There is no block-wide barrier in the k loop and no weight held in registers ahead of use. Each consumer
warp computes the same n16 group of w1 and w3 that a warp of `k_moe_up_p` computes, with the same MMAs in the same
order into each accumulator and the same epilogue: the outputs are the same bits. A chunk whose 4 m16 tiles all have
rows (most tiles of a long chunk) runs a copy of the loop without a branch between the MMAs, so the next unit's loads
and unpacking interleave with this one's MMAs; a waiting thread sleeps in the barrier (`try_wait` with a suspend hint).

Shapes measured (`bench_kernels upws`, M = 8,192, the code path, uniform routes, median of 11 interleaved rounds,
against `k_moe_up_p` in the same process):

| shape: consumer warps C, 64-k steps a stage KU, stages S | without the branch-free full tiles | with them |
|---|---|---|
| C4 KU2 S2 (2 CTAs an SM, 154-156 registers) **the default** | -5.1 to -6.0 % | **-7.2 %** (6,679 against 7,200 us) |
| C6 KU2 S2 (2 CTAs an SM, 128 registers) | -6.5 to -6.9 % | -4.5 % |
| C12 KU2 S2 (1 CTA an SM, 128 registers) | -6.8 to -7.2 % | -2.3 % |

The default at the other sizes (median of 11): M = 1 -8.5 %, 64 -1.1 %, 2,000 -1.4 %. The C12 shape loses at M = 1
(+26 %): 3 column tiles a tile give too few CTAs to stream the weights.

Exactness. `bench_kernels upws` runs the whole MoE with each shape and compares `y` with `k_moe_up_p`'s bit for bit:
uniform and skewed routes, the scales raw and from their 1-bit code (synthetic corners: experts with none to all of
their 32-groups raw), M = 1, 37, 200, 2,000, 4,097, 8,192: identical in all 144 cases; `serve_test --exact` on the
engine gives the same lines and logprobs as with `k_moe_up_p`. The bench's inputs are signed, with activation scales of
2^-8 .. 2^-5 and the shared expert zero: with `dev_fill`'s values (all positive) every SwiGLU input lands in its clamp
and `y` is the same whatever the up kernel computes. Each run prints `y`'s spread (finite, distinct values) and fails when it is flat; a build fed the
wrong activation scale byte differs in 99.99 % of the outputs.

The MMAs take whole scale words: PTX's byte selectors (`{byte-id, thread-id}`) pick the 32-k unit's byte, which
ptxas places in a uniform register (24 shifts and masks and 16 byte loads fewer a chunk; the time is the same).

End to end (four GB10, 32 lines, the GPU at its 2,200 MHz cap) a cold 1M-token prompt is processed in 102.9 s with this
kernel and in 105.1 s with `upk` 0, within the run-to-run spread of 4 to 5 %; decode does not run this kernel.

Why the default shape: larger stages or a third stage cost an SM its second CTA; with 12 consumers the MMAs throttle
on the tensor pipe (math 63 % of their samples) while the kernel runs at 86 TFLOPS.

## 4. Decode GEMVs: the second weight batch before the wait (`DSK_PRE2`)

`engine/kernels/gemv_fp8.cu`, `gemv16.cu` (a template flag `PRE` on `k_gemv_fp8` and `k_gemv16`), `dsk.h`
(`pre2_mask()`). The technique follows a note in pico-dgx-ds4's [engine.md](../ds41f-4xgx10-switchless/docs/engine.md#what-remains-for-15-ms-at-k--1); the kernels
and measurements here are this repository's.

Why. A classic decode GEMV streams its weights through two register buffers: the first batch is requested before the
kernel waits on the one before it (PDL), the next ones after the wait, one buffer ahead of the math. The normed GEMVs
start right after a collective, while DRAM is idle, and each warp has more than one batch to load: wq_a + wkv 10
64-byte steps a warp (3 batches), the router gate 20 (5). From two rows on they run the classic kernels (at one row a
persistent kernel fills a ring before the wait already).

What it does. With `PRE`, the second buffer is requested before the wait too; the loop then skips that one load. The
same loads, the same MMAs in the same order: the outputs are the same bits. Up to 8 rows wq_a + wkv (FP8) and the router
gate (bf16) launch it (wq_b has 2-3 steps a warp and already loads them all before the wait). Only the kernels that take
it without a register spill have it: `cuobjdump --dump-resource-usage` gives the three `PRE` kernels 117-127 registers
and no stack, and leaves every other GEMV kernel's registers and stack as they are. wo_a, whose CTAs run beside the
attention kernel, keeps one batch before the wait. `DSK_PRE2=<mask>`: 1 the FP8 GEMVs, 2 the gate; default 3, 0 none.

Measured in the engine (four Sparks, GPU at 2,200 MHz, one user, 27 chat and 27 code requests at temperature 1.0 with
a fixed prefix and seed each, so every run takes the same steps): the GPU's ms a step by rows from the engine's
own request lines, the build with `DSK_PRE2=0` and with the default (two runs each):

| rows | `DSK_PRE2=0` | default | |
|---|---|---|---|
| 1 | 19.35 / 19.42 | 19.35 / 19.44 | |
| 2 | 22.35 / 22.37 | 22.17 / 22.21 | -0.8 % |
| 3 | 25.46 / 25.49 | 25.13 / 25.19 | -1.2 % |
| 4 | 28.49 / 28.49 | 27.86 / 27.89 | -2.1 % |
| 5 | 30.50 / 30.50 | 30.38 / 30.39 | -0.4 % |
| 6 | 33.75 / 33.75 | 33.65 / 33.67 | -0.3 % |
| a step | 26.16 / 26.19 | 25.94 / 25.98 | -0.7 to -0.9 % |

`bench_chain` (one Spark, collective stand-ins
at the direct ring's times `--ar-us 9.9 --arf-us 13.3 --ag-us 11`): the FP8 GEMVs -0.13 / -0.12 ms and the gate
-0.08 / -0.10 ms at 2 / 4 rows.

Exactness. `engine/server/tests/bit_probe.py`: 8 prompts of 2,048 tokens of real text, greedy with drafts, 200 tokens
each, every token's id, logprob and top two through rank 0's socket: the same bits with it on and off (two runs off
the same bits too). The bit check sees a rounding change: a build that only reverses `k_gemv_fp8`'s split-K sum order
differs in all 8 prompts (from the 3rd to the 99th token).

## 5. The certified head for sampled rows

`engine/model/model.cu` (`k_cert_lo`, `k_cert_mask`, `k_cert_check`), `spec.cu` (`k_spec_in`), `impl.h`. The
certified LM head (pico-dgx-ds4's, [stack4.md](../ds41f-4xgx10-switchless/docs/stack4.md), lever 3, `DSGB10_CERT=1`) reads the drafter's FP8 copy of the head for a
coarse logit of every vocabulary row, bounds each row's error rigorously, and runs the exact head kernel only on the
row groups whose upper bound reaches the best lower bound over the ranks: the argmax, and so the token, is the full
head's. This section extends it to sampled rows, which otherwise read the whole head (about 1 ms a step).

What it does. A row sampled at a temperature T up to 1 with no top-k, top-p or top logprobs takes the Gumbel-max
winner of `l / T - log(-log(u))`, `u` hashed from the session's seed, the row's position and the id (`ops::sample`).
That score is monotone in `l`, so the logit bounds give bounds of the score: `k_cert_lo` takes the best lower bound of
the score, `k_cert_mask` keeps a group when a row's upper score bound reaches it (each with another 1e-3 of slack for
the roundings of the division and the noise, which the cert kernels compute with the sampler's hash). The winner always
survives, and its logit comes from the same exact kernel, so the token is the one the full head gives. `k_spec_in`
takes the certified bodies for such rows; `DSGB10_CERT_SAMPLED=0` keeps the full head for them. Above T = 1 the
scores flatten and more of the head survives, so those rows keep the full head.

Exactness. `DSGB10_CERT_CHECK=1` (pico-dgx-ds4's check mode, extended: for a sampled row it compares the scores' winners)
runs the full head beside the certified one on every rank: 27 chat and 27 code requests sampled at T = 1.0 and 18
greedy ones, every row checked on each of the four ranks, 0 mismatches; the largest |coarse - exact|
was 7.5 % of the bound. The sampled bench (fixed prefix and seed per request) takes the same steps with it as
without it, so the same texts.

Measured in the engine (four Sparks, one user, the GPU's ms a step by rows from the engine's request lines, with
section 4 in both):

| rows | full head | certified | |
|---|---|---|---|
| 1 | 19.44 | 19.14 | -1.5 % |
| 2 | 22.21 | 22.01 | -0.9 % |
| 3 | 25.19 | 25.01 | -0.7 % |
| 4 | 27.89 | 27.75 | -0.5 % |
| 5 | 30.39 | 30.27 | -0.4 % |
| 6 | 33.67 | 33.52 | -0.4 % |
| a step | 25.98 | 25.79 | -0.7 % |

A step re-reads 45-78 of the head's 2,020 row groups exactly
on top of the coarse read.

## 6. Prefill MoE down kernel with a mover warp (`k_moe_down_ws`)

`engine/kernels/prefill.cu`: `k_moe_down_ws`, selected from 4,096 rows when `DSGB10_PF_TUNE` has no `down` item; `DSGB10_PF_DWS=0` selects the previous down kernel.
With the served `DSGB10_PF_TUNE=hc,small,down`, pico-dgx-ds4's column-group down kernel (`k_moe_down_g`) takes every pass of 4,096 rows or more instead, so
the kernel of this section does not run in the served settings.
`engine/kernels/tests/bench_feed.cu` times it and checks it.

What it does. As the up kernel of section 3 does, one warp moves the next tile of weights and activations into shared memory
while the other warps compute on the tile that has arrived, so the loads of one tile overlap the multiply-accumulates of the one
before. No operand, accumulation order or rounding changes.

Measured, without the `down` item of `DSGB10_PF_TUNE`. As a kernel it is 16 % faster at 8,192 rows. In the engine the gain is 0.7 %, because the exchange after the down kernel (the rows are summed and exchanged between the Sparks) takes longer than the kernel's saving, so it sets the pace of each layer.

Exactness. `bench_feed --check`: the output is bit-identical to the previous down kernel's.

## Commands to reproduce (on a Spark, engine stopped)

From `ds41f-4xgx10-switchless/` on a Spark, under the ring lock the docs use (`engine/server/ring.sh`) for anything that
needs the ring. Check `nvidia-smi` first (the GPU is shared).

Build (flags are the repo's: `-O3 -lineinfo`, sm_121a, CUDA 13.0):

    cmake -S engine -B engine/build && cmake --build engine/build -j4
    cmake -S engine -B engine/build-attn -DCMAKE_BUILD_TYPE=Release && cmake --build engine/build-attn -j 16

Registers and spills (`LOCAL` must be 0):

    cuobjdump -res-usage engine/build/server/dsgb10_rank | c++filt | grep -A1 -E "k_moe_up_p|k_moe_down_p|k_moe_down_g|k_moe_up<|k_moe_down<|k_moe_up_r16|k_moe_down_r16|attn_kernel"

SASS counts. Dump once, split per function, count instructions, `MOV`, `PLOP3`, `QMMA`, `HMMA`, `LDS`:

    cuobjdump -sass engine/build/server/dsgb10_rank > rank.sass
    awk '/Function :/{f=$3} /^[ \t]+\/\*[0-9a-f]+\*\//{n[f]++; if($0~/ MOV /)m[f]++; if($0~/PLOP3/)p[f]++; if($0~/QMMA/)q[f]++; if($0~/HMMA/)h[f]++}
         END{for(f in n) printf "%s\t%d\tMOV %d\tPLOP3 %d\tQMMA %d\tHMMA %d\n", f, n[f], m[f], p[f], q[f], h[f]}' rank.sass \
      | while IFS=$'\t' read f rest; do printf '%s\t%s\n' "$(echo "$f" | c++filt)" "$rest"; done \
      | grep -E "k_moe_up_p|k_moe_down_p|k_moe_down_g|k_moe_up<1, 4, 4, 0, 1>|k_moe_down<4, 1|k_moe_down<16, 4|k_moe_down_r16|attn_kernel" | sort

(Instruction lines of `cuobjdump -sass` start with `/*0000*/`; the awk counts those. `QMMA` lines count each MMA; the per-MMA
figure is instructions in the loop divided by MMAs: take total over MMAs only as a rough check and compare totals
of the same function in the two builds.) For the reference column take the same dump from a build of the reference
kernels in a separate clean directory.
What to compare: the totals and `MOV` of `k_moe_down_p` and `k_moe_down_g` (the table above), and `attn_kernel`
(more loads, `LOCAL = 0`).

Exactness (all bit for bit):

    engine/build/kernels/test_kernels                       # all
    engine/build/kernels/test_kernels "" p_moe              # the prefill MoE variants against the reference, bit for bit (section 1)
    engine/build/kernels/test_kernels "" prefill            # p_fp8, p_16, p_moe
    engine/build/kernels/test_kernels "" moe                # decode MoE (a control: no section changes it)
    engine/build/kernels/test_kernels "" rows16moe          # the _r16 builds against 8-row calls
    engine/build/kernels/test_kernels "" mcode              # coded against raw scales
    engine/build-attn/attn/slot_tests                       # sparse_attn, several sessions in shuffled slots, vs each alone
    engine/build-attn/attn/attn_tests DATA/l20
    engine/build-attn/attn/attn_tests DATA/long32k    # long decode, compressed blocks
    engine/build-attn/attn/fused_tests                      # index_dense and fused kernels (not touched; run for the sparse_attn calls)

(`DATA` is where the test data was written; it is generated as `engine/attn/README.md` shows, `engine/attn/tests/gen.sh gen_attn.py ...`; `filter` names
and `l2`/`l20`/`long32k` case names are the README's.) Then the engine's golden hashes, from `golden/README.md` ("Checking a build"),
on the built `dsgb10_rank`: `golden/refhash.sh` (the eight runs: tf, k1, nol2pf, fp8, bf16, temp, ced16, ced28). They must print
`842278b28d87dad0` (tf), `febc7dc54ee40a39` (k1, nol2pf, fp8, bf16), `a8f7a08f8cd1c5e8` (temp), `81f772ab79a950e4` (ced16),
`8f08a153d415c5c8` (ced28). ced16 and ced28 exercise prefill MoE and the attention kernel.

Kernel timing (run the reference build once first, to learn this Spark's run-to-run spread on each bench; the reference times
in these comments are another run's than the tables above):

    # sections 1 and 3: prefill MoE down and up at 8192 rows (reference: up 7.38 ms standalone, down 6.61; each run
    # measures its own reference, which moves between runs: 7.20 to 7.38 ms for up)
    PF_M="8192" PF_ROUNDS=11 engine/build/kernels/bench_kernels prefill
    # a control: decode MoE kernel GB/s (up 215 GB/s, down 224-228) and the scale code
    engine/build/kernels/bench_kernels moe
    engine/build/kernels/bench_kernels mscale
    # decode step at one and six rows (reference: 18.96 ms at one row; 36-37 ms at six)
    engine/build/kernels/bench_chain --rows 1 --rounds 30
    engine/build/kernels/bench_chain --rows 6 --rounds 30
    # section 2: attention class at 1k, 16k, 128k and 250k context (reference: kernel 12.3 us x 40 at 128k)
    engine/build-attn/attn/bench_attn --detail 1024 16384 131072 262144

(`bench_chain` takes `--rows R --rounds N`; 30 rounds, medians. `bench_kernels` benches rotate
more than 256 MB of weights so nothing is cached. If a path above differs in the build tree, the README's list in
`engine/kernels/README.md` "Build, test, bench" is authoritative.)

Optional, engine stopped: `ncu` metrics `smsp__issue_active.avg.pct_of_peak_sustained_active`,
`smsp__warps_eligible.avg.per_cycle_active`, `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` on
`k_moe_up_p` and `attn_kernel`.

Section 1 is `prefill.cu`, section 2 is `sparse_attn.cu`; neither calls code of the other.
