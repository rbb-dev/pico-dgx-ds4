#!/bin/bash
# Long copy steps (DSGB10_COPY_ROWS, model/copy_run.h) on the Sparks: the GPU rows G2, G3 and G4's commands, run by hand on
# the four Sparks (not by the build): the engine stopped for G2, running for G3 and G4. Each section says what it
# proves and which settings it needs; the settings go on every rank (the start refuses ranks that differ).
#
# usage: long_copy_gpu.sh g1 | g2 | g3 | g4a | g4b | g4c | g4d | g4e      (each prints the commands it runs)
# environment: ENGINE_BIN (the built engine binary), SOCK (rank 0's socket), TOKENS (an int32 .npy of real text), OUT (a work folder)
set -u
ENGINE_BIN=${ENGINE_BIN:-build/dsgb10_rank}
SOCK=${SOCK:-/tmp/dsgb10-engine.sock}
TOKENS=${TOKENS:-docs-real.npy}
OUT=${OUT:-long-copy-out}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
run() { echo "+ $*"; "$@"; }

case "${1:-}" in
g1)
  # G1 (one Spark, engine stopped): every row-group kernel against 8-row calls, bit for bit, at 9 / 16 / 17 / 32 / 48 / 64 rows.
  # Proves: the exact bodies' arithmetic. Settings: none. Pass: every line ok, the CONTROL line ok, "all passed".
  run build/test_kernels "" rg
  ;;
g2)
  # G2 (four Sparks, the tool, no service; engine stopped): one session teacher-forced at R rows a step, R x S constant. Proves: a long
  # step's exact body (DSGB10_TF_SOLO: a given step of more than 16 rows runs as the one-session graph's exact bodies do) gives the
  # samples hash (tokens and logit bits) of the same positions at R = 1, at a short and a long context, with the tiered split, and with
  # the abliteration. Settings: --setenv DSGB10_TF_SESSROWS=1 (a session may hold more than 8 rows of a step) and DSGB10_TF_SOLO=1 on
  # every rank. Pass: the hash of R = 32 and R = 64 equals R = 1's, at both contexts. Control (must differ, or the run proves nothing):
  # R = 32 without DSGB10_TF_SOLO (the plain wide body). The golden folder: a real-text token file (tokens.npy) and a one-element
  # positions.npy / record_positions.npy; a step holds at most 16 sessions. Needs: RUN4 (the tool's run4.sh), GOLDEN, WEIGHTS, FLEET.
  RUN4=${RUN4:?run4.sh of the tool} GOLDEN=${GOLDEN:?golden folder} WEIGHTS=${WEIGHTS:-~/dsgb10/weights/rank{rank}.dsw}
  one() {  # NAME CONTEXT R EXTRA...: one run; its hash line into NAME.hash
    local name=$1 ctx=$2 r=$3 s=$((256 / $3)); shift 3
    echo "+ run4.sh ... --batch $ctx --batch-rows $r --steps $s $*"
    FLEET=${FLEET:?fleet.conf} timeout 1800 bash "$RUN4" "$ENGINE_BIN" --mode batch --golden "$GOLDEN" --big --max-ctx $((ctx + 512)) --batch-tf \
      --weights "$WEIGHTS" --slots 1 --batch "$ctx" --batch-rows "$r" --steps "$s" --kv-split tiered \
      --setenv DSGB10_TF_SESSROWS=1 "$@" > "$OUT/g2-$name.log" 2>&1
    sed -n 's/^batch-tf: session [0-9]* (slot [0-9]*, context \([0-9]*\).*samples hash \([0-9a-f]*\);.*/\1 \2/p' "$OUT/g2-$name.log" | tee "$OUT/g2-$name.hash"
  }
  for ctx in 1024 262144; do
    one "r1-$ctx" "$ctx" 1
    for r in 8 32 64; do one "r$r-$ctx" "$ctx" "$r" --setenv DSGB10_TF_SOLO=1; done
    one "control32-$ctx" "$ctx" 32
  done
  echo "compare: the hash of r32-C and r64-C must equal r1-C's; control32-C's must differ. Then the same with --ablit FILE on every run"
  echo "(the exact body reads the shared expert's activations through moe_rows_shared; ablit_batch_test.py needs its own check of the replies)"
  ;;
g3)
  # G3 (the service, one request at a time, no front-end connected): rewrites of 1K / 4K / 8K-token passages. Proves: ids and
  # log-probability bits equal with DSGB10_COPY_ROWS unset and 64. Settings: engine A: DSGB10_COPY=1 (rows on), COPY_ROWS unset; engine B:
  # DSGB10_COPY=1 DSGB10_COPY_ROWS=64, nothing else different. Pass: every prompt BIT-EQUAL, greedy and seeded-sampled (1.0, 0.95);
  # engine B's log shows "copy: long steps N (whole W), ..." with N > 0 and W < N at least once (a long step rejected in part), and
  # the "copy: digest" lines equal on the four ranks.
  for L in 1024 4096 8192; do
    run python3 "$HERE/long_copy_prompts.py" --tokens "$TOKENS" --out "$OUT/copy-$L.npy" --n 4 --len "$L"
  done
  echo "start the engine with COPY_ROWS unset (A), then: for each L and each mode (greedy | --temperature 1.0 --top-p 0.95 --seed 7):"
  echo "  python3 $HERE/bit_probe.py --socket $SOCK --tokens $OUT/copy-L.npy --out $OUT/A-L-mode.json --n 4 --len L --new 400 [mode]"
  echo "restart with DSGB10_COPY_ROWS=64 (B) and run the same lines with --out $OUT/B-L-mode.json, then:"
  echo "  python3 $HERE/bit_probe.py --compare $OUT/A-L-mode.json $OUT/B-L-mode.json"
  ;;
g4a)
  # G4 (a): speed, file rewrites, one session: the prompts of G3 (1K / 4K / 8K) on the build with COPY_ROWS unset (control, once) and 64
  # (arm). Proves: the long steps' gain. Clean boots, the dashboard open. Records tok/s of each reply.
  echo "bit_probe.py's decode_us per prompt, from the A and B json of g3: tokens / decode_us"
  ;;
g4b)
  # G4 (b): the edit bench, one session. Settings: as (a).
  echo "python3 ds41f-4xgx10-switchless/bench/editbench.py --socket $SOCK   # control: COPY_ROWS unset; arm: 64"
  ;;
g4c)
  # G4 (c): the 3-prompt chat / code bench: nothing to copy, so the result must stay inside the +-4-5 % floor (the unset graph costs nothing).
  echo "the 3-prompt chat and code bench; control: COPY_ROWS unset; arm: 64"
  ;;
g4d)
  # G4 (d): the edit bench with 3 sessions (--edit-sessions 3): expected unchanged, long rows do not run in shared steps.
  echo "python3 ds41f-4xgx10-switchless/bench/editbench.py --socket $SOCK --edit-sessions 3   # control and arm as (a)"
  ;;
g4e)
  # G4 (e): an arrival during a long run: time to the first token of a second request against the same arrival with COPY_ROWS unset.
  # Proves: the cost to an arrival (design estimate 0.26 s against 0.08 s), to be read before DSGB10_COPY_ROWS=64 is served.
  echo "start a rewrite (g3's 8K prompt, --new 1500), 2 s in send a short request; record its time to first token; control: COPY_ROWS unset"
  ;;
block1)
  # The 7-draft block (DSGB10_DRAFT_MAX=7, DSGB10_DRAFT_DYN=L, DSGB10_CF_TAIL=A,B; docs/engine.md): the block runs, and its rows 6-7 by
  # source. Settings on every rank: DSGB10_COPY=1 DSGB10_COPY_ROWS=64 (the cursor the rule reads), DSGB10_DRAFT_MAX=7, DSGB10_DRAFT_DYN=5,
  # DSGB10_STEP_LOG=$OUT/block-steplog.jsonl and DSGB10_COPY_DUMP=$OUT/block-copydump (the log then carries "cp":[chain rows kept, copy rows
  # offered] for the step after it). Start lines to see: "dspark: block rows 7 (one session, DSGB10_DRAFT_MAX)", "spec: draft pass by step
  # (DSGB10_DRAFT_DYN=5): 7 rows after 5 accepted drafts and no live copy cursor, else 5". Then the code prompts of the 3-prompt bench and the
  # edit bench, one request at a time; block_rows.py prints, for draft rows 6 and 7, how often each source (chain / copy) offered one and
  # how often it was accepted. Pass: the chain's rows 6-7 are offered (> 0) in steps that follow a step that accepted 5 or more drafts;
  # the copy rows 6-7 keep appearing (> 0) on the edit bench; the block is worth keeping only if the code prompts gain more than the 4-5 %
  # spread between runs.
  run python3 "$HERE/block_rows.py" "$OUT/block-steplog.jsonl"
  ;;
block2)
  # The 7-draft block unset: this build with DSGB10_DRAFT_MAX / _DYN / _CF_TAIL all unset against g3's results of the long copy
  # steps' build (A-L-mode.json, COPY_ROWS unset; this path's engine code is that build's with the settings unset), the same prompts. Start lines: no "dspark: block rows" line and no "draft pass by step" line; the "steps verifying
  # 0..5" line of the request log has no "6..7:" part. Pass: bit_probe --compare equal for every prompt and mode, greedy and sampled.
  for L in 1024 4096 8192; do
    echo "python3 $HERE/bit_probe.py --socket $SOCK --tokens $OUT/copy-$L.npy --out $OUT/U-$L-mode.json --n 4 --len $L --new 400 [mode]"
    echo "python3 $HERE/bit_probe.py --compare $OUT/A-$L-mode.json $OUT/U-$L-mode.json"
  done
  ;;
block3)
  # The block's own exactness: DSGB10_DRAFT_MAX=7 alone (no DYN), same prompts: the verified tokens must equal the unset run's (every draft
  # is verified; rows of at most 8 take the certified bodies, whose bits do not depend on the row count). Pass: ids equal, greedy and
  # sampled; logprob bits are reported (equal expected). Control: none needed beyond block2; the unset run is its reference.
  for L in 1024 4096 8192; do
    echo "python3 $HERE/bit_probe.py --socket $SOCK --tokens $OUT/copy-$L.npy --out $OUT/B7-$L-mode.json --n 4 --len $L --new 400 [mode]"
    echo "python3 $HERE/bit_probe.py --compare $OUT/U-$L-mode.json $OUT/B7-$L-mode.json"
  done
  ;;
*)
  echo "usage: $0 g1 | g2 | g3 | g4a | g4b | g4c | g4d | g4e | block1 | block2 | block3"
  exit 2
  ;;
esac
