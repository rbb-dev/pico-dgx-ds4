#!/usr/bin/env bash
# The helper the systemd units call (deploy/README.md). pico-engine.service runs rank 0 in the foreground as its main
# process; this script is its ExecStartPre (prepare: the GPU clock, the workers' files, the page cache, the memory check,
# ranks 1-3), its ExecStartPost (ready: waits for the engine's socket) and its ExecStopPost (reap: what is left of ranks
# 1-3). install puts the built binaries and the units in place and enables and starts nothing.
#
#   pico-ctl.sh prepare | ready | reap | install SRC_TREE | status
#
# Settings: ~/dsgb10/pico.env (pico.env.example). Every file lives under ~/dsgb10 on every Spark.
set -euo pipefail
PICO=${PICO:-$HOME/dsgb10}
BIN=$PICO/bin RUN=$PICO/run LOGS=$PICO/logs
# pico.env in systemd's EnvironmentFile syntax (KEY=value, unquoted, spaces kept): read as systemd reads it, never sourced
if [ -r "$PICO/pico.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ $line =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"; fi
  done < "$PICO/pico.env"
fi
export FLEET=$BIN/fleet.conf
NAME=dsgb10_rank
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=10)
log() { echo "pico-ctl: $*"; }
die() { echo "pico-ctl: $*" >&2; exit 1; }

load_fleet() {
  # shellcheck source=/dev/null
  . "$BIN/fleet.sh"
  fleet_load "$FLEET" || die "the fleet file $FLEET"
}

# the engine's arguments, the same on every rank (rank 0's ExecStart passes these and --rank 0)
engine_args() { echo "--mode serve --socket $RUN/engine.sock --disk-tag service --keep-gib ${KEEP_GIB:-64} ${ENGINE_ARGS:-}"; }

# shell text for a box: wait up to $1 s for its ranks to exit, then pkill -x, 10 s more, then -9 (the process name exactly)
reap_text() {
  echo "for i in \$(seq $((2 * $1))); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -x $NAME; for i in \$(seq 20); do pgrep -x $NAME > /dev/null || exit 0; sleep 0.5; done; pkill -9 -x $NAME"
}

prepare() {
  load_fleet
  local w r a low=
  # at boot the workers may still be starting: wait up to 10 minutes for each to answer ssh, so that the first start
  # does not fail and back off
  for w in "${FLEET_WORKERS[@]}"; do
    for a in $(seq 1 60); do "${SSH[@]}" "$w" true 2> /dev/null && break; sleep 10; done
  done
  mkdir -p "$RUN" "$LOGS"
  # the GPU clock limit on all four Sparks (GPU_CLOCK, MHz; never unlocked). GPU_CLOCK_TOOL, if set, is run as
  # "TOOL LIMIT [ssh targets...]" (the contract of DGX_FAN_CLOCK_TOOL); otherwise each Spark gets nvidia-smi's lock.
  if [ -n "${GPU_CLOCK_TOOL:-}" ]; then
    "$GPU_CLOCK_TOOL" "${GPU_CLOCK:-2200}" "${FLEET_WORKERS[@]}" || die "the GPU clock (${GPU_CLOCK:-2200}) was not set on every Spark"
  else
    sudo -n nvidia-smi -lgc "0,${GPU_CLOCK:-2200}" > /dev/null || die "the GPU clock was not set on this Spark"
    for w in "${FLEET_WORKERS[@]}"; do
      "${SSH[@]}" "$w" "sudo -n nvidia-smi -lgc 0,${GPU_CLOCK:-2200}" > /dev/null || die "the GPU clock was not set on $w"
    done
  fi
  # the engine binary, the fleet file, the confidence network's weights and the abliteration file on the workers, at the paths
  # rank 0 has them (small files; every rank reads them). The rank files, Engram shards, vision files and the copy table
  # are NOT copied here: docs/setup.md puts them on every Spark before the first start
  for w in "${FLEET_WORKERS[@]}"; do
    "${SSH[@]}" "$w" "mkdir -p dsgb10/bin dsgb10/logs dsgb10/run"
    scp -q -o BatchMode=yes -o ConnectTimeout=10 "$BIN/cfmlp_cml.bin" "$w:dsgb10/bin/cfmlp_cml.bin"
    [ ! -r "$PICO/ablit.bin" ] || scp -q -o BatchMode=yes -o ConnectTimeout=10 "$PICO/ablit.bin" "$w:dsgb10/ablit.bin"
    scp -q -o BatchMode=yes -o ConnectTimeout=10 "$BIN/$NAME" "$w:dsgb10/bin/$NAME.new"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "cat > dsgb10/bin/fleet.conf.new && mv -f dsgb10/bin/$NAME.new dsgb10/bin/$NAME && mv -f dsgb10/bin/fleet.conf.new dsgb10/bin/fleet.conf" < "$BIN/fleet.conf"
  done
  # strays out on every box
  bash -c "$(reap_text 0)"
  for w in "${FLEET_WORKERS[@]}"; do "${SSH[@]}" "$w" "$(reap_text 0)"; done
  # the page cache of every model file on every box dropped (every rank's weights and Engram shards under ~/dsgb10,
  # rank 0 holds them all after a conversion, and the checkpoint's shards): on GB10 a CUDA allocation sees only free
  # pages, and the session store is sized by the rank with the least
  local files="'$PICO/weights/*.dsw' '$PICO/engram/*.bin' '${MODEL_SNAPSHOT:-/nonexistent}/*.safetensors'"
  bash -c "python3 '$BIN/evict_cache.py' $(fleet_model_files 0) $files"
  r=1
  for w in "${FLEET_WORKERS[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "python3 - $(fleet_model_files $r) $files" < "$BIN/evict_cache.py"
    r=$((r + 1))
  done
  # then every box's whole page cache (clean pages only): builds,
  # profiles and tools leave GiBs cached that the files above do not cover. It goes on if the drop is refused.
  local drop='sync; echo 3 > /proc/sys/vm/drop_caches'
  sudo -n sh -c "$drop" || log "the page-cache drop was refused here (going on)"
  for w in "${FLEET_WORKERS[@]}"; do
    "${SSH[@]}" "$w" "sudo -n sh -c '$drop'" || log "the page-cache drop was refused on $w (going on)"
  done
  # then the memory a rank needs on every box: MemFree, what the GPU can take (MemAvailable counts the cache too)
  a=$(awk '/^MemFree:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
  [ "${a:-0}" -ge "${MIN_AVAIL_GIB:-100}" ] || low="rank 0 ${a} GiB"
  for w in "${FLEET_WORKERS[@]}"; do
    a=$("${SSH[@]}" "$w" "awk '/^MemFree:/ { printf \"%d\", \$2 / 1048576 }' /proc/meminfo")
    [ "${a:-0}" -ge "${MIN_AVAIL_GIB:-100}" ] || low="$low${low:+, }$w ${a:-?} GiB"
  done
  [ -z "$low" ] || die "MemFree under ${MIN_AVAIL_GIB:-100} GiB after the cache drop on $low (another model loaded, or other files cached?)"
  # ranks 1-3, then the unit's ExecStart runs rank 0. Only the rank goes to the background, every descriptor of it away
  # from ssh's (an & over the whole && chain keeps a shell holding ssh's output: ssh then waits for the rank to end)
  rm -f "$RUN/engine.sock"
  r=1
  for w in "${FLEET_WORKERS[@]}"; do
    # every earlier run's log is kept under its own end time (the last 20 of them), so no start can take a crashed run's log away
    "${SSH[@]}" "$w" "cd dsgb10 && f=logs/rank.$r.log && { [ ! -s \$f ] || mv -f \$f logs/rank.$r.\$(date -r \$f +%Y%m%d-%H%M%S).log; } && { ls -t logs/rank.$r.2*.log 2>/dev/null | tail -n +21 | xargs -r rm -f; } && { setsid nohup bin/$NAME --rank $r $(engine_args) >> \$f 2>&1 < /dev/null & }" ||
      { bash -c "$(reap_text 0)"; die "starting rank $r on $w failed"; }
    r=$((r + 1))
  done
  log "ranks 1-3 started; rank 0 next: bin/$NAME --rank 0 $(engine_args)"
}

# the engine serves once rank 0's socket is there (dsfront connects to it); rank 0 must still be running
ready() {
  local pid i
  pid=$(systemctl --user show -p MainPID --value pico-engine.service)
  for ((i = 0; i < 3600; i++)); do
    [ -S "$RUN/engine.sock" ] && { log "rank 0 (pid $pid) listens on $RUN/engine.sock"; return 0; }
    kill -0 "$pid" 2> /dev/null || die "rank 0 exited before it served"
    sleep 0.5
  done
  die "rank 0 did not listen within 30 minutes"
}

# after rank 0 ended (it stops ranks 1-3 itself on a planned stop): what is left, on every box
reap() {
  load_fleet
  local w
  for w in "${FLEET_WORKERS[@]}"; do "${SSH[@]}" "$w" "$(reap_text 10)" || true; done
  bash -c "$(reap_text 0)" || true
  rm -f "$RUN/engine.sock"
}

# the built binaries (SRC_TREE: the ds41f-4xgx10-switchless folder after its builds), the confidence network, the clock tool
# (tools/gpu-clock in that folder) and the units into place (the abliteration file, ablit.bin, is put in ~/dsgb10 on rank 0 by hand; prepare copies it to the workers)
do_install() {
  local src=${1:?install SRC_TREE}
  mkdir -p "$BIN" "$RUN" "$LOGS" "$HOME/.config/systemd/user"
  install -m 755 "$src/engine/build/server/$NAME" "$BIN/$NAME"
  install -m 755 "$src/frontend/target/release/dsfront" "$BIN/dsfront"
  install -m 644 "$src/engine/server/fleet.sh" "$BIN/fleet.sh"
  install -m 644 "$src/engine/server/tools/evict_cache.py" "$BIN/evict_cache.py"
  install -m 644 "$src/tools/stack4/cfmlp_cml.bin" "$BIN/cfmlp_cml.bin"
  [ -r "$BIN/fleet.conf" ] || die "put your fleet file at $BIN/fleet.conf (deploy/fleet.conf.example)"
  [ -r "$src/tools/gpu-clock" ] || die "the clock tool $src/tools/gpu-clock is missing"
  install -m 755 "$src/tools/gpu-clock" "$BIN/gpu-clock"
  install -m 755 "$0" "$BIN/pico-ctl.sh"
  local here; here=$(cd "$(dirname "$0")" && pwd)
  install -m 644 "$here/pico-engine.service" "$HOME/.config/systemd/user/pico-engine.service"
  install -m 644 "$here/pico-front.service" "$HOME/.config/systemd/user/pico-front.service"
  install -m 644 "$here/pico-media.service" "$HOME/.config/systemd/user/pico-media.service"
  systemctl --user daemon-reload
  log "installed in $BIN and ~/.config/systemd/user (enabled: none, started: none)"
}

status() {
  systemctl --user --no-pager status pico-engine.service pico-front.service pico-media.service | sed -n '1,40p'
}

case "${1:-}" in
  prepare) prepare ;;
  ready) ready ;;
  reap) reap ;;
  install) shift; do_install "$@" ;;
  status) status ;;
  *) die "prepare | ready | reap | install SRC_TREE | status" ;;
esac
