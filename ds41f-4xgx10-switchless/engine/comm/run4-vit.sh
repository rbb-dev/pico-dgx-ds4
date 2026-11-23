#!/bin/bash
# Run vit_ring_test (or another ring binary) on all four boxes: ranks 1-3 detached on the workers, rank 0 here in the
# foreground, as run4.sh does; the engine must be stopped on all four. The differences: a stale copy of the binary
# on a worker is stopped by its process id (found by its exact name), not with pkill; the timeout is 1200 s; and
# the weight files are checked on every box before anything starts (--weights FMT as vit_ring_test reads it, default
# ~/dsgb10/weights/vision.rank{rank}.dsw), so a missing file stops the run at once and says which box.
# usage: FLEET=FILE engine/comm/run4-vit.sh <binary> [vit_ring_test args...]
#   e.g. FLEET=~/dsgb10/bin/fleet.conf engine/comm/run4-vit.sh build/vision/vit_ring_test --model ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
set -u
. "$(dirname "$0")/../server/fleet.sh"
fleet_load "$FLEET" || exit 2
BIN=$1
shift
NAME=$(basename "$BIN")
BD=$FLEET_BIN_DIR LD=$FLEET_LOG_DIR  # as written, for the workers' shells (a leading ~/ is theirs)
LBD=$(fleet_local "$BD") LLD=$(fleet_local "$LD")
mkdir -p "$LBD" "$LLD"
# the weight files, when the arguments name them (or by default): rank r's file on box r
W="~/dsgb10/weights/vision.rank{rank}.dsw"
prev=""
for a in "$@"; do [ "$prev" = --weights ] && W=$a; prev=$a; done
SYN=0
for a in "$@"; do [ "$a" = --synthetic ] && SYN=1; done
if [ "$SYN" = 0 ]; then
  FW0=${W//\{rank\}/0}
  [ -f "${FW0/#\~/$HOME}" ] || { echo "run4-vit: rank 0's weight file ${FW0} is missing here"; exit 2; }
  r=1
  for w in "${FLEET_WORKERS[@]}"; do
    Fr=${W//\{rank\}/$r}
    ssh "$w" "[ -f ${Fr/#\~/\$HOME} ]" < /dev/null || { echo "run4-vit: rank $r's weight file $Fr is missing on $w"; exit 2; }
    r=$((r + 1))
  done
fi
cp "$BIN" "$LBD/$NAME"
cp "$FLEET" "$LBD/fleet.conf"
for w in "${FLEET_WORKERS[@]}"; do scp -q "$BIN" "$w:$(fleet_rel "$BD")/$NAME"; done
ARGS="$*"
r=1
for w in "${FLEET_WORKERS[@]}"; do
  ssh "$w" "cat > $BD/fleet.conf; for p in \$(pgrep -x $NAME); do kill \$p; done; mkdir -p $LD; setsid nohup timeout 1200 $BD/$NAME --rank $r $ARGS > $LD/$NAME.$r.log 2>&1 < /dev/null &" < "$FLEET"
  r=$((r + 1))
done
timeout 1200 "$LBD/$NAME" --rank 0 "$@" 2>&1 | tee "$LLD/$NAME.0.log"
rc=${PIPESTATUS[0]}
sleep 2
r=1
for w in "${FLEET_WORKERS[@]}"; do
  echo "--- rank $r ($(fleet_get "RANK${r}_NAME"))"; ssh "$w" "tail -n 4 $LD/$NAME.$r.log"
  r=$((r + 1))
done
exit "$rc"
