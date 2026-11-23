#!/bin/bash
# The media worker's memory high-water: one fresh container per row, so each peak is that row's alone.
# The worker runs with the unit's flags (read-only root, 192m /tmp, no GPU) under --memory 512m (and the same
# --memory-swap: no swap to hide behind). The test inputs come from a second container (the same image, no memory
# cap) so they do not count against the worker. Run it on a Spark that is not serving, never on rank 0's Spark
# while an engine or a measurement run is up.
#
#   bash media_highwater.sh [TREE] [IMAGE] [ROWS]
#
# TREE  the folder with tools/vision/ and ref/inference/config.json (default: this script's own checkout, two folders up)
# IMAGE the media worker's image (default pico-media-f5:rev4)
# ROWS  which rows (default "a b b2 c1 c2 d e")
#
# What is read, and from where: the container's own cgroup on the HOST (memory.peak, and a sampler every 0.1 s of
# memory.current and memory.stat), so a reading survives an OOM kill. The page cache a row was charged is `file`
# less `shmem` at the sample of highest memory.current. A row whose worker was killed, or whose memory.peak could
# not be read, prints FAIL on every line it prints and is compared with nothing.
#
# Rows (512m, --max-image-pixels 40000000; PASS = not OOM-killed, the worker answered as the row says, memory.peak
# at most 480 MiB; the samples of every row are kept in the run's folder, ~/dsgb10/logs/media-highwater-<time>):
#   a   one 40 MP RGBA image
#   b   the request at the worker's bound in its worst order: the 40 MP image by URL first, then 25 inline pages
#       of the 4 MP test image (a body of 69.6 MiB; the URL needs media_worker_highwater_wrapper.py)
#   b2  a frame one byte over the worker's bound (72 MiB): the error frame in words; memory.peak read right after it
#       (before anything else is sent) within 8 MiB of the idle peak; then a small image on the same connection
#       answered (its peak printed, not judged)
#   c1  a 47 MiB clip inline in the request
#   c2  a 128 MiB clip by URL, served by the client container on 127.0.0.1 (the worker is started through
#       media_worker_highwater_wrapper.py, which replaces only the public-address rule)
#   d   three requests of (a) and the clip of (c2) at once on four connections, all four answered; also the peak
#       less the page cache within 16 MiB of the largest of (a) and (c2) alone (measured first if not in ROWS)
#   e   row (d)'s four requests five times over on one worker: the same peak line as (d), and memory.current one
#       second after the last reply within 16 MiB of its value before the first request
# The control: bash media_highwater.sh TREE pico-media-f5:rev3 "d" prints FAIL on every line (the worker is killed).
set -u
TREE=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
IMAGE=${2:-pico-media-f5:rev4}
ROWS=${3:-a b b2 c1 c2 d e}
CFG=$TREE/ref/inference/config.json
HELPER=$TREE/tools/vision/media_worker_highwater.py
WRAPPER=$TREE/tools/vision/media_worker_highwater_wrapper.py
PORT=18951
LIMIT_MIB=480
BOUND=$((72 << 20))
OUT=$HOME/dsgb10/logs/media-highwater-$(date +%H%M%S); mkdir -p "$OUT"
[ -f "$CFG" ] && [ -f "$HELPER" ] && [ -f "$WRAPPER" ] || { echo "missing $CFG, $HELPER or $WRAPPER"; exit 2; }

# set by arm(): PEAK (memory.peak, MiB; -1 unreadable), CACHE (page cache charged at the highest sample, MiB),
# IDLE (idle memory.peak), CUR0 / CUR1 (memory.current before the first request / one second after the last reply),
# OK (1 = worker alive, peak readable, answered ok, peak at most the limit), RESULT (the helper's line)
PEAK=-1; CACHE=0; IDLE=0; CUR0=0; CUR1=0; OK=0; RESULT=
arm() {  # NAME WRAPPED(0|1) KIND SIZE [helper args]
  local name=$1 wrapped=$2 kind=$3 size=$4; shift 4
  local cname=pico-media-highwater-$$ tag=$(echo "$name" | tr -c "a-z0-9\n" _ | cut -c1-24)
  local run=(docker run -d --name "$cname" --net=host --read-only --tmpfs /tmp:rw,size=192m --cap-drop=ALL --security-opt=no-new-privileges
             --memory 512m --memory-swap 512m -v "$CFG":/model/inference/config.json:ro)
  PEAK=-1; CACHE=0; IDLE=0; CUR0=0; CUR1=0; OK=0; RESULT=
  if [ "$wrapped" = 1 ]; then
    "${run[@]}" --entrypoint python3 -v "$WRAPPER":/h/w.py:ro "$IMAGE" /h/w.py --host 127.0.0.1 --port $PORT --max-image-pixels 40000000 >/dev/null || { echo "$name: worker did not start: FAIL"; return; }
  else
    "${run[@]}" "$IMAGE" --host 127.0.0.1 --port $PORT --max-image-pixels 40000000 >/dev/null || { echo "$name: worker did not start: FAIL"; return; }
  fi
  for _ in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 0.2; done
  local id cg; id=$(docker inspect -f '{{.Id}}' "$cname")
  cg=$(ls -d /sys/fs/cgroup/system.slice/docker-$id.scope 2>/dev/null || find /sys/fs/cgroup -maxdepth 4 -type d -name "*$id*" | head -1)
  if [ -z "$cg" ] || [ ! -r "$cg/memory.peak" ]; then echo "$name $IMAGE: the container's cgroup is not readable from the host: FAIL"; docker rm -f "$cname" >/dev/null; return; fi
  local f=$OUT/$tag.samples
  ( while [ -r "$cg/memory.current" ]; do
      printf '%s cur %d %s\n' "$(date +%s.%N | cut -c1-14)" $(( $(cat "$cg/memory.current") >> 20 )) \
        "$(awk '$1~/^(anon|file|shmem|sock|kernel|file_mapped)$/{printf "%s %d ", $1, $2/1048576}' "$cg/memory.stat")"
      sleep 0.1; done > "$f" 2>/dev/null ) &
  local sp=$!
  sleep 0.5
  IDLE=$(( $(cat "$cg/memory.peak") >> 20 )); CUR0=$(( $(cat "$cg/memory.current") >> 20 ))
  MID=-1
  if [ "$kind" = bound ]; then  # the helper stops after the error frame: memory.peak is read there, then it goes on
    local sync=$OUT/$tag.sync; rm -rf "$sync"; mkdir -p "$sync"; chmod 777 "$sync"
    docker run --rm --net=host --user "$(id -u)" --entrypoint python3 -v "$HELPER":/h/c.py:ro -v "$sync":/sync "$IMAGE" /h/c.py "$kind" "$size" --port $PORT --sync /sync > "$sync/out" 2>&1 &
    local hp=$!
    for _ in $(seq 1 1200); do [ -e "$sync/refused" ] && break; sleep 0.1; done
    [ -e "$sync/refused" ] && MID=$(( $(cat "$cg/memory.peak" 2>/dev/null || echo -1048576) >> 20 ))
    touch "$sync/go"; wait $hp
    RESULT=$(tail -1 "$sync/out")
  else
    RESULT=$(docker run --rm --net=host --entrypoint python3 -v "$HELPER":/h/c.py:ro "$IMAGE" /h/c.py "$kind" "$size" --port $PORT "$@" 2>&1 | tail -1)
  fi
  sleep 1
  local oom; oom=$(docker inspect -f '{{.State.OOMKilled}}' "$cname")
  local pk; pk=$(cat "$cg/memory.peak" 2>/dev/null); local cur; cur=$(cat "$cg/memory.current" 2>/dev/null)
  [ -n "$cur" ] && CUR1=$(( cur >> 20 ))
  docker logs "$cname" > "$OUT/$tag.worker.log" 2>&1
  [ -n "$pk" ] && PEAK=$(( pk >> 20 ))
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  # the page cache charged at the sample of highest memory.current
  CACHE=$(sort -k3,3n "$f" | tail -1 | awk '{for(i=1;i<=NF;i++){if($i=="file")f=$(i+1); if($i=="shmem")s=$(i+1)} print f-s}')
  CACHE=${CACHE:-0}
  local verdict=PASS
  if [ "$oom" != false ] || [ "$PEAK" -lt 0 ] || ! docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; then
    verdict="FAIL (worker killed or memory.peak unreadable)"; PEAK=-1
  elif [[ "$RESULT" != *'"result": "ok"'* ]]; then verdict="FAIL (not answered as the row says)"
  elif [ "$PEAK" -gt $LIMIT_MIB ]; then verdict="FAIL (peak over $LIMIT_MIB MiB)"; fi
  [[ "$verdict" == PASS ]] && OK=1
  printf '%s %s: idle peak %d MiB, peak %d MiB, page cache charged %d MiB, memory.current %d -> %d MiB, OOMKilled=%s | %s | %s\n' \
    "$name" "$IMAGE" "$IDLE" "$PEAK" "$CACHE" "$CUR0" "$CUR1" "$oom" "$RESULT" "$verdict"
  docker rm -f "$cname" >/dev/null
}

less_cache() { echo $(( $1 - $2 )); }  # PEAK CACHE

ALONE_A=; ALONE_C2=
alone() {  # the largest of (a) and (c2) alone, peaks less their page cache; sets LARGEST (-1 when either failed)
  [ -n "$ALONE_A" ] || { arm "row a (alone, for the comparison)" 0 image 40000000; ALONE_A=$([ $OK = 1 ] && less_cache $PEAK $CACHE || echo -1); }
  [ -n "$ALONE_C2" ] || { arm "row c2 (alone, for the comparison)" 1 clipurl 128; ALONE_C2=$([ $OK = 1 ] && less_cache $PEAK $CACHE || echo -1); }
  if [ "$ALONE_A" -lt 0 ] || [ "$ALONE_C2" -lt 0 ]; then LARGEST=-1; else LARGEST=$(( ALONE_A > ALONE_C2 ? ALONE_A : ALONE_C2 )); fi
}

compare() {  # LABEL: the peak less page cache of the row just run against the largest alone + 16
  if [ $OK != 1 ] || [ "$LARGEST" -lt 0 ]; then echo "$1: peak less page cache not compared (a row failed or was killed): FAIL"; return; fi
  local nc; nc=$(less_cache $PEAK $CACHE)
  if [ "$nc" -le $((LARGEST + 16)) ]; then echo "$1: peak less page cache $nc MiB within 16 MiB of the largest alone ($LARGEST MiB): PASS"
  else echo "$1: peak less page cache $nc MiB is over the largest alone ($LARGEST MiB) + 16 MiB: FAIL"; fi
}

echo "image $IMAGE on $(hostname), $(date -Is); samples in $OUT"
for row in $ROWS; do
  case $row in
    a)  arm "row a (40 MP image)" 0 image 40000000; ALONE_A=$([ $OK = 1 ] && less_cache $PEAK $CACHE || echo -1) ;;
    b)  arm "row b (40 MP image by URL, then 25 inline pages)" 1 doc 25 ;;
    b2) arm "row b2 (a frame one byte over the bound)" 0 bound $BOUND
        if [ $OK != 1 ]; then echo "row b2: peak against the idle peak not compared (a row failed or was killed): FAIL"
        elif [ "$MID" -lt 0 ]; then echo "row b2: memory.peak right after the error frame could not be read: FAIL"
        elif [ "$MID" -le $((IDLE + 8)) ]; then echo "row b2: memory.peak right after the error frame $MID MiB, within 8 MiB of the idle peak ($IDLE MiB): PASS (the small image answered after it: peak $PEAK MiB, not judged)"
        else echo "row b2: memory.peak right after the error frame $MID MiB is over the idle peak ($IDLE MiB) + 8 MiB: FAIL"; fi ;;
    c1) arm "row c1 (47 MiB clip inline)" 0 clip 47 ;;
    c2) arm "row c2 (128 MiB clip by URL)" 1 clipurl 128; ALONE_C2=$([ $OK = 1 ] && less_cache $PEAK $CACHE || echo -1) ;;
    d)  alone
        arm "row d (3 images + URL clip at once)" 1 mixed 3
        compare "row d" ;;
    e)  alone
        arm "row e (row d five times over on one worker)" 1 mixed 3 --rounds 5
        compare "row e"
        if [ $OK != 1 ]; then echo "row e: memory.current after the last reply not compared (a row failed or was killed): FAIL"
        elif [ "$CUR1" -le $((CUR0 + 16)) ]; then echo "row e: memory.current $CUR1 MiB one second after the last reply, within 16 MiB of $CUR0 MiB before the first request: PASS"
        else echo "row e: memory.current $CUR1 MiB one second after the last reply is over $CUR0 MiB + 16 MiB: FAIL"; fi ;;
    *)  echo "unknown row $row" ;;
  esac
done
echo DONE
