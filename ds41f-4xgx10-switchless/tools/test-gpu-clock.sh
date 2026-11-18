#!/usr/bin/env bash
# gpu-clock with ssh, sudo and nvidia-smi stood in: this machine and three ssh targets. Checks: only the four limits are
# applied (anything else exit 2, nothing run, no state file); each machine gets "sudo -n nvidia-smi -lgc 0,LIMIT", judged by
# its exit code and its words; a machine that fails is named and the others are still set; the exit says whether every
# machine took the limit; the JSON state file holds limit, at, ok and error; the clock is never reset.
#   bash tools/test-gpu-clock.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
C="$HERE/gpu-clock"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"
cat > "$T/bin/sudo" <<'EOS'
#!/usr/bin/env bash
printf '%s %s\n' "${FAKE_HOST:-here}" "$*" >> "$LOGDIR/sudo.argv"
[[ ${1:-} == -n ]] && shift
exec "$@"
EOS
cat > "$T/bin/ssh" <<'EOS'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$LOGDIR/ssh.argv"
while (( $# )); do case $1 in -o) shift 2;; -n) shift;; *) break;; esac; done
target=$1; shift
[[ " ${FAIL_SSH:-} " == *" $target "* ]] && { echo "ssh: connect to host $target: Connection refused" >&2; exit 255; }
FAKE_HOST=$target exec bash -c "$*"
EOS
cat > "$T/bin/nvidia-smi" <<'EOS'
#!/usr/bin/env bash
host=${FAKE_HOST:-here}
printf '%s %s\n' "$host" "$*" >> "$LOGDIR/smi.argv"
if [[ -f $LOGDIR/smi.$host ]]; then read -r rc text < "$LOGDIR/smi.$host"; echo "$text"; exit "$rc"; fi
case $1 in
  -lgc) echo "GPU clocks set to \"(gpuClkMin 0, gpuClkMax ${2#0,})\" for GPU 0000000F:01:00.0"; echo "All done.";;
  *) echo "bad args: $*"; exit 2;;
esac
EOS
chmod +x "$T/bin/"*
fails=0
ok()  { echo "ok:   $1"; }
bad() { echo "FAIL: $1"; fails=$((fails + 1)); }
run() {
  rm -rf "$T/log" "$T/st"; mkdir -p "$T/log" "$T/st"
  for h in "${ANSWERS[@]:-}"; do [[ -n $h ]] && printf '%s\n' "${h#*=}" > "$T/log/smi.${h%%=*}"; done
  LOGDIR="$T/log" PATH="$T/bin:$PATH" GPU_CLOCK_STATE="$T/st/gpu-clock.json" bash "$C" "$@" > "$T/out" 2>&1
  echo $? > "$T/rc"; touch "$T/log/smi.argv" "$T/log/ssh.argv" "$T/log/sudo.argv"
}
rc() { cat "$T/rc"; }
state() { cat "$T/st/gpu-clock.json" 2>/dev/null; }
# the state file against a python expression over its fields: v["limit"], v["ok"], v["error"], v["at"]
state_is() { python3 - "$(state)" "$1" <<'EOP'
import json, sys, time
v = json.loads(sys.argv[1])
assert eval(sys.argv[2]), v
EOP
}

ANSWERS=(); run
[[ $(rc) == 2 && ! -s $T/log/smi.argv && ! -e $T/st/gpu-clock.json ]] && ok "no limit: usage error, nothing run" || bad "no limit: rc $(rc): $(cat "$T/out")"
for l in 2300 unlocked 0 reset; do
  run "$l" w1
  [[ $(rc) == 2 && ! -s $T/log/smi.argv && ! -s $T/log/ssh.argv && ! -e $T/st/gpu-clock.json ]] && grep -q '2400 2200 2000 1800' "$T/out" &&
    ok "limit $l: refused (exit 2), the four limits named, nothing run" || bad "$l: rc $(rc): $(cat "$T/out")"
done

for l in 2400 2200 2000 1800; do
  run "$l" w1 w2 w3
  [[ $(rc) == 0 && $(grep -c "clocks set to $l MHz" "$T/out") == 4 && $(grep -c -- " -lgc 0,$l\$" "$T/log/smi.argv") == 4 ]] &&
    ok "$l: four machines set, exit 0, one line each" || bad "$l: rc $(rc): $(cat "$T/out") $(cat "$T/log/smi.argv")"
done
[[ $(grep -c -- '-rgc' "$T/log/smi.argv") == 0 ]] && ok "the clock is never reset" || bad "reset: $(cat "$T/log/smi.argv")"
[[ $(sed -n 1p "$T/out" | cut -d: -f1) == "$(hostname -s)" && $(grep -c . "$T/log/ssh.argv") == 3 ]] &&
  ok "this machine first and without ssh; three ssh targets" || bad "order: $(cat "$T/out") / $(cat "$T/log/ssh.argv")"
[[ $(grep -c -- '-n nvidia-smi' "$T/log/sudo.argv") == 4 ]] && ok "through sudo -n" || bad "sudo: $(cat "$T/log/sudo.argv")"
state_is 'v["limit"] == 1800 and v["ok"] is True and v["error"] is None and abs(v["at"] - time.time()) < 60' &&
  ok "the state file: limit (a number), at, ok true, no error" || bad "state: $(state)"

ANSWERS=("w2=1 Failed to set clocks: Not Supported"); run 2200 w1 w2 w3
[[ $(rc) == 1 ]] && grep -q '^w2: FAILED.*Not Supported' "$T/out" && [[ $(grep -c 'clocks set to 2200' "$T/out") == 3 ]] &&
  ok "a driver that refuses: named with its words, exit 1, the other three set" || bad "refused: rc $(rc): $(cat "$T/out")"
state_is 'v["limit"] == 2200 and v["ok"] is False and "w2" in v["error"] and "Not Supported" in v["error"]' &&
  ok "the failure recorded (ok false, the error)" || bad "failed state: $(state)"

ANSWERS=("w1=0 something unexpected"); run 2200 w1 w2
[[ $(rc) == 1 ]] && grep -q '^w1: FAILED' "$T/out" && ok "exit 0 but no 'clocks set': failed (the output is judged too)" || bad "odd output: rc $(rc): $(cat "$T/out")"
ANSWERS=("w1=0 GPU clocks set to (0, 2200). Warning: ignored Error: not supported"); run 2200 w1
[[ $(rc) == 1 ]] && ok "an error word in the output: failed" || bad "error word: rc $(rc): $(cat "$T/out")"

ANSWERS=(); FAIL_SSH=w3 run 2200 w1 w2 w3
[[ $(rc) == 1 ]] && grep -q '^w3: FAILED' "$T/out" && [[ $(grep -c 'clocks set to 2200' "$T/out") == 3 ]] &&
  ok "a machine ssh cannot reach: named, exit 1, the other three set" || bad "ssh down: rc $(rc): $(cat "$T/out")"

ANSWERS=(); run 2200
mkdir -p "$T/ro"; chmod 500 "$T/ro"
LOGDIR="$T/log" PATH="$T/bin:$PATH" GPU_CLOCK_STATE="$T/ro/x/gpu-clock.json" bash "$C" 2200 > "$T/out" 2>&1
[[ $? == 1 || $(id -u) == 0 ]] && ok "an unwritable state file: said, exit 1" || bad "unwritable: $(cat "$T/out")"

echo "test-gpu-clock: $fails failed"
exit $((fails > 0))
