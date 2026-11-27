#!/usr/bin/env bash
# The Gumbel sampler's gates on the four Sparks (DSGB10_GSAMPLE, DSGB10_GCOUPLE): the client side of G0 .. G3 and G5,
# run by hand (not by the build). Every gate is run on a clean boot of all four Sparks (clock 2,200 MHz), the dashboard open and signed in, a drain
# before each timed run; the engine is started with the settings the gate names (the engine's arguments: --setenv NAME=VALUE), and
# nothing else of the engine changes between two arms.
#
#   gsample_gates.sh g0 DIR            record: the build's own files in DIR (run on the build before the sampler and on this one, both
#                                      with the two variables unset; then `gsample_gates.sh compare DIR_A DIR_B`)
#   gsample_gates.sh g1 DIR            record under DSGB10_GSAMPLE=1 DSGB10_GCOUPLE=0.8, then `compare` against G0's DIR for the files
#                                      that must be equal, and `drafts DIR_A DIR_B` for the line that must differ
#   gsample_gates.sh g2                the certified head under G1's settings, DSGB10_CERT_CHECK=1 (a development build,
#                                      -DDSGB10_DEV=ON): prints the cert: lines' counters
#   gsample_gates.sh g3 DIR            the truncated pick (DSGB10_GSAMPLE=2, DSGB10_FORCE_NTOP=20, DSGB10_STEP_LOG=FILE; the replay
#                                      needs a development build, -DDSGB10_DEV=ON): the replay and
#                                      the counts, each with its control (a GSAMPLE=0 log; against T = 0.7)
#   gsample_gates.sh g5 ARM DIR        the speed arms on the seeded edit bench, 1, 3 and 8 sessions, 3 repeats: ARM A is
#                                      DSGB10_GSAMPLE=2, ARM B is DSGB10_GSAMPLE=2 DSGB10_GCOUPLE=0.8
#   gsample_gates.sh compare A B       G0 / G1 equality of two record directories
#   gsample_gates.sh drafts A B        the mean drafts accepted per step of the temperature-only seeded file, A against B
#
# What each proves:
#  G0  this build with both variables unset is the build before it: bit_probe's greedy (8 prompts x 200 tokens) and seeded sampled
#      (T 1.0 top-p 0.95; T 1.0 top-p 1.0) files equal (ids, log-probabilities, top two), and bench_seeded's greedy batch dumps at
#      2, 3, 5 and 16 sessions equal by compare_dumps.py. The batch files are taken twice on one build first (`g0 DIR` then `g0 DIR2`
#      on the same build): what differs between two runs of one build is the test's own floor, and only a request that is stable there counts.
#  G1  under DSGB10_GSAMPLE=1 DSGB10_GCOUPLE=0.8: G0's greedy and seeded-truncated files equal again; the seeded temperature-only file
#      equal AND its drafts-accepted-per-step different from G0's (the same would mean coupling did not run and the equality proves
#      nothing); greedy 2, 3, 5, 16 sessions equal by compare_dumps.py.
#  G2  a temperature-only request under G1's settings moves the cert: line's steps counter by its step count (flat: the certified
#      head lost to the Gumbel bit); DSGB10_CERT_CHECK=1: 0 mismatches over 10,000 rows or more with coupled drafts.
#  G3  gsample_replay.py rows: 6,000 rows or more, 0 different, and its control (a GSAMPLE=0 log) sees the walk; then 7,200 one-token
#      unseeded replies against the renormalised kept probabilities (p > 0.001), and against T = 0.7 rejected.
#  G5  speed on agent sessions as served (T 1.0, top-p 0.95): the edit bench with a client seed per task and turn, so that arm A
#      and arm B emit the same tokens at one session and drafts accepted per step is an exact count on identical work.
#
# Settings it reads (environment; the defaults in brackets):
#   SOCKET   rank 0's engine socket, forwarded to this box [/tmp/dsgb10-engine.sock]; no front-end connected during G0 / G1 / G3 counts
#   TOKENS   an int32 .npy of real text for bit_probe
#   URL      the chat endpoint (.../v1/chat/completions) for the batch dumps; KEY_FILE the key; MODEL the served name
#   LOG_CMD  a command that prints rank 0's engine log (for the drafts and cert: lines)
#   SPEC, REPO, BENCH_DIR  the edit bench's task file, checkout and directory [bench/edit_bench.json, the repository, bench/]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ENGINE=$(cd "$HERE/../.." && pwd)
TOOLS=$ENGINE/../tools/gate
BENCH_DIR=${BENCH_DIR:-$ENGINE/../bench}
SOCKET=${SOCKET:-/tmp/dsgb10-engine.sock}
TOKENS=${TOKENS:-}
URL=${URL:-}
KEY_FILE=${KEY_FILE:-}
MODEL=${MODEL:-deepseek-v4.1-flash}
LOG_CMD=${LOG_CMD:-}
SPEC=${SPEC:-$BENCH_DIR/edit_bench.json}
REPO=${REPO:-$(cd "$ENGINE/../.." && pwd)}
N=${N:-8}; LEN=${LEN:-2048}; NEW=${NEW:-200}

say() { printf '%s\n' "$*"; }
probe() { # NAME ARGS...: bit_probe into $DIR/NAME.json
  local name=$1; shift
  say "bit_probe $name: $*"
  python3 "$HERE/bit_probe.py" --socket "$SOCKET" --tokens "$TOKENS" --n "$N" --len "$LEN" --new "$NEW" --out "$DIR/$name.json" "$@"
}
batch() { # NAME SESSIONS TEMP: bench_seeded distinct dumps
  local name=$1 n=$2 temp=$3
  say "bench_seeded $name: $n sessions, temperature $temp"
  rm -f "$DIR/$name.shas"
  DSGB10_KEY_FILE=$KEY_FILE DSGB10_MODEL=$MODEL BENCH_TEMP=$temp BENCH_DUMP="$DIR/$name.shas" python3 "$TOOLS/bench_seeded.py" "$URL" distinct "$n" > /dev/null
}
record() {
  DIR=$1
  mkdir -p "$DIR"
  probe greedy --temperature 0
  probe seeded-topp --temperature 1.0 --top-p 0.95 --seed 1000
  probe seeded-temp --temperature 1.0 --top-p 1.0 --seed 2000
  for n in 2 3 5 16; do batch "greedy-$n" "$n" 0; done
  if [ -n "$LOG_CMD" ]; then bash -c "$LOG_CMD" > "$DIR/rank0.log"; fi
}
compare() {
  local a=$1 b=$2 bad=0
  for f in greedy seeded-topp seeded-temp; do
    say "== $f"; python3 "$HERE/bit_probe.py" --compare "$a/$f.json" "$b/$f.json" || bad=1
  done
  for n in 2 3 5 16; do
    say "== greedy-$n"; python3 "$TOOLS/compare_dumps.py" "$a/greedy-$n.shas" "$b/greedy-$n.shas" || bad=1
  done
  return $bad
}
accepted_mean() { # LOG: the mean of "accepted per step" over the log's request lines
  grep -o 'drafts verified and [0-9.]* accepted per step' "$1" | awk '{ s += $4; n++ } END { if (n) printf "%.4f (%d requests)\n", s / n, n; else print "none" }'
}
case ${1:-} in
  g0) record "${2:?DIR}";;
  g1) record "${2:?DIR}";;
  compare) compare "${2:?A}" "${3:?B}";;
  drafts)
    a=$(accepted_mean "${2:?A}/rank0.log"); b=$(accepted_mean "${3:?B}/rank0.log")
    say "drafts accepted per step: A $a, B $b"
    [ "$a" != "$b" ] || { say "G1: the same line: coupling did not run"; exit 1; };;
  g2)
    say "send the seeded temperature-only request (bit_probe --temperature 1.0 --top-p 1.0 --seed 2000) and read the cert: lines"
    mark=$(bash -c "$LOG_CMD" | grep -c '^cert:' || true)
    DIR=$(mktemp -d); probe cert --temperature 1.0 --top-p 1.0 --seed 2000
    bash -c "$LOG_CMD" | grep '^cert:' | tail -n +$((mark + 1))
    say "G2: the steps counters of those lines must add up to the requests' steps (a flat counter: the certified head was lost); mismatches 0"
    rm -rf "$DIR";;
  g3)
    DIR=${2:?DIR}; mkdir -p "$DIR"
    say "rows: STEP_LOG of a seeded request (DSGB10_GSAMPLE=2 DSGB10_FORCE_NTOP=20 DSGB10_STEP_LOG=FILE); then the control log of a GSAMPLE=0 run"
    say "  python3 $HERE/gsample_replay.py rows STEP_LOG --seed S --temperature 1.0 --top-p 0.95 --min-rows 6000"
    say "  python3 $HERE/gsample_replay.py rows CONTROL_LOG --seed S --control"
    say "counts: DSGB10_GSAMPLE=1, unseeded"
    python3 "$HERE/gsample_replay.py" counts --socket "$SOCKET" --tokens "$TOKENS" --n 7200;;
  g5)
    arm=${2:?ARM A or B}; DIR=${3:?DIR}
    for sess in 1 3 8; do for rep in 1 2 3; do
      out="$DIR/arm$arm-s$sess-r$rep"
      say "edit bench arm $arm: $sess sessions, repeat $rep"
      python3 "$BENCH_DIR/bench.py" --url "${URL%/chat/completions}" --key-file "$KEY_FILE" --model "$MODEL" --modes edit --edit-spec "$SPEC" \
        --edit-repo "$REPO" --edit-thinking off,high --edit-repeats 1 --edit-temperature 1.0 --edit-seed $((100000 * rep)) \
        --edit-sessions "$sess" --edit-out "$out"
    done; done;;
  *) sed -n 2,19p "$0"; exit 2;;
esac
