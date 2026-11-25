#!/bin/bash
# The wide-step switches on the four Sparks: DSGB10_WIDE_SH=1 (the shared expert on the side stream), DSGB10_WIDE_AR=1 and 2
# (the all-reduce as reduce-scatter + all-gather; 2: the FFN's gather in bf16) and DSGB10_HEADG=2 (the head on rows
# normalised once), each alone and all three together, against the same engine with none of them.
#
# What it proves: every switch gives the same bits. The teacher-forced batch mode (--mode batch --batch-tf) runs R rows a
# step (R = sessions x --batch-rows, R x steps constant at 1,088) and prints each session's samples hash, which does not
# depend on the step's shape (docs/batching.md); a switch on must print the hash the switch-off run printed, for every
# session. It also proves each switch was in force: the engine's start line for it is in the log of a run with it on and in
# no log of a run with it off, so an engine that ignores a switch cannot pass as equal. And it prints each run's median
# step time (the line "gpu (ms) median") with its difference to the switch-off run of the same R.
#
# What it does not prove: a speed. One run of each configuration is a reading, not a measurement; the spread between runs of one
# build on these Sparks is 4-5 %, so a difference under it is none. Each switch is read when the engine starts (a static of
# model.cu), so on and off are two launches, one engine start each.
#
# Run by hand on the four Sparks (not by the build). Exit 0: every hash equal, every switch seen; 1: a difference or a switch not seen; 2: the
# premise was not reached (no hash line: the run did not get to its steps).
#
#   BIN=build/server/dsgb10_rank GOLDEN=<the golden data directory> engine/server/tests/wide_test.sh [R ...]   (default 18 32 64; R = sessions x rows, at most 16 sessions of at most 8 rows)
#   env: CONFIGS="base sh ar1 ar2 headg2 all"  OUT=<log directory>  MAXCTX=8192  RUN4=<run4.sh>
#        ARGS="--weights ~/dsgb10/weights3/rank{rank}.dsw ...": engine arguments every run takes (the weights, as the service has)
# The golden directory needs only tokens.npy (a real-text token file), positions.npy and record_positions.npy: the hashes are
# compared between configurations, never with reference outputs. The sessions start at distinct contexts (1024 + 128 k), so
# no two rows of a step are the same computation and a row mixed up with another one shows.
set -u
D=$(cd "$(dirname "$0")/.." && pwd)
RUN4=${RUN4:-$D/../comm/run4.sh}
BIN=${BIN:-} GOLDEN=${GOLDEN:-}
[ -n "$BIN" ] && [ -n "$GOLDEN" ] || {
  echo "usage: BIN=<dsgb10_rank> GOLDEN=<golden dir> $0 [R ...]"
  exit 2
}
OUT=${OUT:-${TMPDIR:-/tmp}/wide-test.$$}
MAXCTX=${MAXCTX:-8192}
ARGS=${ARGS:-}
CONFIGS=${CONFIGS:-base sh ar1 ar2 headg2 all}
ROWS=("$@")
[ ${#ROWS[@]} -gt 0 ] || ROWS=(18 32 64)
mkdir -p "$OUT"
fail=0 premise=0

# the settings of a configuration, and the start lines that must show (a ';'-separated list) when it is on
setenvs() {
  case $1 in
    base) echo "" ;;
    sh) echo "--setenv DSGB10_WIDE_SH=1" ;;
    ar1) echo "--setenv DSGB10_WIDE_AR=1" ;;
    ar2) echo "--setenv DSGB10_WIDE_AR=2" ;;
    headg2) echo "--setenv DSGB10_HEADG=2" ;;
    all) echo "--setenv DSGB10_WIDE_SH=1 --setenv DSGB10_WIDE_AR=2 --setenv DSGB10_HEADG=2" ;;
    *) echo "unknown configuration $1" >&2; exit 2 ;;
  esac
}
expected() {
  case $1 in
    base) echo "" ;;
    sh) echo "(DSGB10_WIDE_SH)" ;;
    ar1) echo "(DSGB10_WIDE_AR=1)" ;;
    ar2) echo "(DSGB10_WIDE_AR=2)" ;;
    headg2) echo "(DSGB10_HEADG=2)" ;;
    all) echo "(DSGB10_WIDE_SH);(DSGB10_WIDE_AR=2);(DSGB10_HEADG=2)" ;;
  esac
}
SWITCH_LINES=("(DSGB10_WIDE_SH)" "(DSGB10_WIDE_AR=" "(DSGB10_HEADG=2)")

# sessions x rows a step for R rows: the largest G <= 8 (model::kMaxSessRows) that divides R
shape() {
  local r=$1 g
  for g in 8 7 6 5 4 3 2 1; do
    if ((r % g == 0)) && ((r / g <= 16)); then  # at most ops::kMaxSess (16) sessions in a step
      echo "$((r / g)) $g"
      return
    fi
  done
}

# the hash lines of a log, one "session hash" a line, sorted
hashes() { sed -n 's/^batch-tf: session \([0-9]*\) .*samples hash \([0-9a-f]*\);.*/\1 \2/p' "$1" | sort -n; }
median() { sed -n 's/^batch-tf: .*gpu (ms) median \([0-9.]*\) .*/\1/p' "$1" | tail -1; }

for R in "${ROWS[@]}"; do
  read -r N G < <(shape "$R")
  S=$((1088 / R))
  [ $((S * R)) -ge 1088 ] || S=$((S + 1))
  list=$(for k in $(seq 0 $((N - 1))); do printf '%d,' $((1024 + 128 * k)); done)
  list=${list%,}
  base_ms="" base_hash=""
  for cfg in $CONFIGS; do
    log=$OUT/rows$R-$cfg.log
    echo "== $R rows ($N sessions x $G rows, $S steps), $cfg"
    # shellcheck disable=SC2046
    timeout 3000 bash "$RUN4" "$BIN" --mode batch --golden "$GOLDEN" --big --slots "$N" --max-ctx "$MAXCTX" \
      --batch "$list" --batch-tf --batch-rows "$G" --steps "$S" $ARGS $(setenvs "$cfg") > "$log" 2>&1
    rc=$?
    h=$(hashes "$log")
    ms=$(median "$log")
    if [ -z "$h" ] || [ -z "$ms" ]; then
      echo "  PREMISE NOT REACHED: no hash or step-time line in $log (exit $rc)"
      premise=1
      continue
    fi
    [ "$rc" -eq 0 ] || {
      echo "  FAIL: the run exited $rc"
      fail=1
    }
    # a switch's start line shows in the log of a configuration that has it and in no other
    want=$(expected "$cfg")
    for line in "${SWITCH_LINES[@]}"; do
      case "$want" in *"$line"*) continue ;; esac
      if grep -qF -- "$line" "$log"; then
        echo "  FAIL: the start line containing '$line' is present with $cfg, which does not set it"
        fail=1
      fi
    done
    # and each one it sets, with the very value (WIDE_AR=1 against 2)
    IFS=';' read -ra wants <<< "$want"
    for w in "${wants[@]}"; do
      [ -z "$w" ] || grep -qF -- "$w" "$log" || {
        echo "  FAIL: the start line containing '$w' is missing with $cfg"
        fail=1
      }
    done
    if [ "$cfg" = base ]; then
      base_ms=$ms base_hash=$h
      echo "  step median $ms ms (the switch-off run)"
      continue
    fi
    if [ -z "$base_hash" ]; then
      echo "  no switch-off run of $R rows to compare with (CONFIGS needs base first)"
      premise=1
      continue
    fi
    if [ "$h" = "$base_hash" ]; then
      echo "  ok: every session's samples hash equals the switch-off run's"
    else
      echo "  FAIL: samples hashes differ from the switch-off run:"
      diff <(echo "$base_hash") <(echo "$h") | sed 's/^/    /'
      fail=1
    fi
    pct=$(awk -v a="$ms" -v b="$base_ms" 'BEGIN { printf "%+.1f", (a - b) / b * 100 }')
    echo "  step median $ms ms against $base_ms ms switched off: $pct % (one reading each)"
  done
done
[ $fail -eq 0 ] || exit 1
[ $premise -eq 0 ] || exit 2
echo "wide_test: every hash equal, every switch seen"
