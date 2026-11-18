#!/usr/bin/env bash
# Data collection for the batch gate's calibration: 2..5 sessions on three prompt sets with the engine's dump on (correctness-style: no timing).
# Usage: collect.sh URL OUTDIR   env: BENCH (default: bench_seeded.py beside this script), DUMP_FILE (engine dump), DUMP_HOST (ssh target if remote), OFFSETS, SESSIONS, BENCH_TEMP (default 0)
# Output: OUTDIR/set<offset>_n<sessions>.csv (dump lines) and .shas (text hashes for compare_dumps.py). Fit: python3 gate_fit.py OUTDIR/*.csv
set -euo pipefail
URL=${1:?URL}; OUT=${2:?OUTDIR}
BENCH=${BENCH:-$(dirname "$0")/bench_seeded.py}
DUMP_FILE=${DUMP_FILE:?the engine dump file (DSGB10_GATE_DUMP_FILE)}
OFFSETS=${OFFSETS:-"0 3 6"}; SESSIONS=${SESSIONS:-"2 3 4 5"}
mkdir -p "$OUT"
sh() { if [ -n "${DUMP_HOST:-}" ]; then ssh "$DUMP_HOST" "$@"; else bash -c "$*"; fi; }
size() { sh "stat -c %s '$DUMP_FILE' 2>/dev/null || echo 0"; }
for off in $OFFSETS; do
  for n in $SESSIONS; do
    rm -f "$OUT/set${off}_n${n}.shas"
    a=$(size)
    BENCH_OFFSET=$off BENCH_TEMP=${BENCH_TEMP:-0} BENCH_DUMP="$OUT/set${off}_n${n}.shas" \
      python3 "$BENCH" "$URL" distinct "$n" >/dev/null
    b=$(size)
    { sh "head -n 1 '$DUMP_FILE'"; sh "tail -c +$((a + 1)) '$DUMP_FILE' | head -c $((b - a))"; } >"$OUT/set${off}_n${n}.csv"
    echo "set$off sessions $n: $(($(wc -l <"$OUT/set${off}_n${n}.csv") - 1)) lines"
  done
done
