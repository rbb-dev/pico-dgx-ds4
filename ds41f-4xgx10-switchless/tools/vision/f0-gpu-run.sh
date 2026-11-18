#!/bin/bash
# The encoder's exactness run, on rank 0's Spark with the engine and dsfront stopped (the same discipline as the null-control gate):
# the reference's dumps in the pico-vision container, the encoder's operations on the same tensors with vit_teacher, the
# nine positive controls, and vit_compare.py's verdict. The engine and dsfront come back at the end whatever happens.
#
#   f0-gpu-run.sh TOOLS_DIR VIT_TEACHER [WORK_DIR]
#     TOOLS_DIR     a folder holding dump_ref.py, vit_compare.py and make_test_images.py (this folder's files)
#     VIT_TEACHER   the binary built on a Spark (engine/build/vision/vit_teacher), copied to rank 0's Spark
#     WORK_DIR      default $HOME/dsgb10/f0 (results land under it; needs about 12 GiB free)
#
# Exit code: vit_compare's (0 = every judged check passed and every control was detected).
set -u
P=$HOME/dsgb10
TOOLS=${1:?TOOLS_DIR}; TEACHER=${2:?VIT_TEACHER}; W=${3:-$P/f0}
TV=$P/src/ds41f-4xgx10-switchless/tools/vision      # the container's mounted tools folder
A="$P/bin/dsfront admin --socket $P/run/admin.sock"
STEMS_OPS=noise_544,noise_768                         # whole operation dumps (1,521 and 3,025 patches)
OP_BLOCKS=0,1,2,7,16,31
CONTROLS="swap-heads rope-axes norm-swap qkv-order gate-up unfold-window unfold-order pad-grey delimiters"
say() { echo "$(date +%T) $*" | tee -a "$W/run.log"; }
mkdir -p "$W"
restore() {
  rm -f "$TV/dump_ref.py" "$TV/vit_compare.py" "$TV/make_test_images.py"
  docker exec pico-vision rm -rf /tmp/f0 /tmp/f0-images >/dev/null 2>&1
  systemctl --user start pico-engine
  for i in $(seq 1 240); do [ -S $P/run/engine.sock ] && systemctl --user is-active --quiet pico-engine && break; sleep 1; done
  systemctl --user start pico-front
  for i in $(seq 1 60); do journalctl --user -u pico-front --since "-2 min" -o cat | grep -q "engine: connected" && break; sleep 2; done
  say "tool files removed from the mounted folder ($(ls $TV | grep -c -E 'dump_ref|vit_compare|make_test_images') left); serving back: $(systemctl --user is-active pico-engine pico-front | tr '\n' ' ')"
}
MS=$(grep ^MODEL_SNAPSHOT= $P/pico.env | cut -d= -f2)
[ -n "$MS" ] || { say "STOP: no MODEL_SNAPSHOT in pico.env"; exit 1; }
for f in dump_ref.py vit_compare.py make_test_images.py; do [ -f "$TOOLS/$f" ] || { say "STOP: $TOOLS/$f missing"; exit 1; }; done
[ -x "$TEACHER" ] || { say "STOP: $TEACHER is not an executable"; exit 1; }
docker ps --format '{{.Names}}' | grep -q '^pico-vision$' || { say "STOP: pico-vision is not running"; exit 1; }
free_gib=$(df --output=avail -BG "$W" | tail -1 | tr -dc 0-9); [ "${free_gib:-0}" -ge 12 ] || { say "STOP: $free_gib GiB free under $W, 12 needed"; exit 1; }
r=busy; for i in $(seq 1 60); do r=$($A "hold 3" 2>&1); [ "$r" = ok ] && break; sleep 2; done; say "hold: $r"
trap restore EXIT
systemctl --user stop pico-front pico-engine; say "dsfront and engine stopped"
rm -rf "$W/dumps" "$W/ours" "$W/images"

# 1. the reference's dumps, in the container: images, then the four attention modes, operation dumps, floors, gemm data
cp "$TOOLS/dump_ref.py" "$TOOLS/vit_compare.py" "$TOOLS/make_test_images.py" "$TV/"
docker exec pico-vision python3 /app/make_test_images.py /tmp/f0-images > "$W/images.log" 2>&1 || { say "STOP: make_test_images failed: $(tail -2 $W/images.log)"; exit 1; }
docker exec pico-vision bash -c 'mkdir -p /tmp/f0-images/cards && mv /tmp/f0-images/card_*.png /tmp/f0-images/cases.json /tmp/f0-images/cards/'
say "images made: $(docker exec pico-vision ls /tmp/f0-images | tr '\n' ' ')"
timeout 3600 docker exec pico-vision python3 /app/dump_ref.py --model "$MS" --images /tmp/f0-images --out /tmp/f0/dumps \
  --attn ref,shim,mem,flash --full noise_544,noise_768,strip_42x42882 --ops $STEMS_OPS --op-blocks $OP_BLOCKS \
  --floors --gemm-random > "$W/dump_ref.log" 2>&1
say "dump_ref exit $?: $(tail -1 "$W/dump_ref.log")"
mkdir -p "$W/images" && docker cp pico-vision:/tmp/f0-images/. "$W/images/" >/dev/null
docker cp pico-vision:/tmp/f0/dumps "$W/dumps" >/dev/null && say "dumps copied out: $(du -sh $W/dumps | cut -f1)"

# 2. the encoder's operations on the reference's tensors, its free-running tower, the weights, vit::gemm on random data
ALL_STEMS=$(ls "$W/dumps/ops" | grep -v MANIFEST | tr '\n' ',' | sed 's/,$//')
T="$TEACHER --model $MS --dumps $W/dumps"
$T --out $W/ours --mode weights --stems $STEMS_OPS > "$W/teacher.log" 2>&1
$T --out $W/ours --mode gemm --stems $STEMS_OPS >> "$W/teacher.log" 2>&1
$T --out $W/ours --mode teacher --stems $STEMS_OPS --op-blocks $OP_BLOCKS >> "$W/teacher.log" 2>&1
$T --out $W/ours --mode free --stems "$ALL_STEMS" --full noise_544,noise_768,strip_42x42882 >> "$W/teacher.log" 2>&1
say "vit_teacher exit $?: $(tail -1 "$W/teacher.log")"

# 3. the positive controls (a wrong build each): free-running images for the errors in blocks 0-2 and the rotary table,
#    the copy-type operations of block 0 for the byte test, on both stems (noise_768's grid of 55 x 55 is padded to 57 x 57:
#    a wrong pad value shows only there)
CTL=""
for c in $CONTROLS; do
  $T --out $W/ours/control/$c --mode free --stems $STEMS_OPS --full $STEMS_OPS --control $c >> "$W/teacher.log" 2>&1
  $T --out $W/ours/control/$c --mode teacher --stems $STEMS_OPS --op-blocks 0 --control $c >> "$W/teacher.log" 2>&1
  CTL="$CTL --control $c=/tmp/f0/ours/control/$c"
done
say "controls run: $(ls $W/ours/control | tr '\n' ' ')"

# 4. the verdict, in the container (numpy is there): dumps stay in the container, ours goes in
docker cp "$W/ours" pico-vision:/tmp/f0/ours >/dev/null
docker exec pico-vision python3 /app/vit_compare.py --dumps /tmp/f0/dumps --ours /tmp/f0/ours $CTL --out /tmp/f0/report.json \
  > "$W/compare.log" 2>&1
rc=$?
docker cp pico-vision:/tmp/f0/report.json "$W/report.json" >/dev/null 2>&1
say "vit_compare exit $rc: $(tail -1 "$W/compare.log")"
tail -40 "$W/compare.log"
exit $rc
