# Setting up four Sparks

From four GB10 machines, a switch and the model checkpoint to the engine as served: the four ranks, the front-end with its
dashboard and sign-in, images and clips, the abliterated model name, kept sessions and the SSD tier, up to a first answer. Commands
run as the same login on all four Sparks, from this repository's `ds41f-4xgx10-switchless` folder on rank 0 unless a step says
otherwise (`../deploy` and `../tools/bench` are the repository root's). [operations.md](operations.md) says how to run it afterwards.

Contents: [1 What you need](#1-what-you-need), [2 The network](#2-the-network), [3 The checkpoint](#3-the-checkpoint),
[4 Build and install](#4-build-and-install), [5 The files each Spark needs](#5-the-files-each-spark-needs), [6 The settings](#6-the-settings),
[7 Sudo, the services and the ring check](#7-sudo-the-services-and-the-ring-check), [8 The media worker](#8-the-media-worker),
[9 Fans and the GPU clock](#9-fans-and-the-gpu-clock), [10 First start and checks](#10-first-start-and-checks),
[11 Where things go wrong first](#11-where-things-go-wrong-first).

The folder layout (the same on every Spark except where `bin/` says otherwise) (`~` is the login's home; the examples call the login `<user>`):

```
~/dsgb10/bin/        every Spark: dsgb10_rank, fleet.conf, cfmlp_cml.bin; rank 0 also fleet.sh, evict_cache.py, pico-ctl.sh, gpu-clock, dsfront
~/dsgb10/weights/    rank<r>.dsw            this Spark's weight file (one file a Spark)
~/dsgb10/vision/     vision.rank<r>.dsw     this Spark's quarter of the image encoder
~/dsgb10/engram/     engram-l1-r<r>of4.bin, engram-l14-r<r>of4.bin, engram_hashmap.bin
~/dsgb10/spill/      the SSD tier and the kept sessions (the engine fills it)
~/dsgb10/logs/       rank.<r>.log on each worker
~/dsgb10/ablit.bin   the abliteration file (from this repository, on rank 0; the start helper copies it to the workers)
rank 0 only:  ~/dsgb10/pico.env, api-key, run/ (sockets, the front-end's session table), model/ (the checkpoint)
```

## 1. What you need

- **Four GB10 machines** (121 GB of memory each; a rank takes about 90 GiB of it), ranks 0 to 3. Rank 0 serves the API and reaches
  the other three over ssh with a key and no password (as the login the engine runs as: `ssh <rank1-lan-address> true` must work
  from rank 0).
- **One switch** that forwards 200 Gb/s RoCE v2 with priority flow control, one 200G cable to each Spark ([section 2](#2-the-network)).
- **Disk, per Spark:** its weight file (about 71.4 GiB, computed; `python3 engine/weights/toc.py FILE` prints a file's exact size), its two Engram shards (about 50.6 GB together), its vision file (233 MiB) and room for the
  SSD tier (`--disk-gib`, 512 in the served line; the engine offers at most the Spark's free space less 32 GiB). **On rank 0 also:** the
  checkpoint (about 510 GB) and room to cut all four ranks' files (four of about 71.4 GiB) before they are copied out.
- **Software on every Spark:** NVIDIA driver 580 and CUDA 13.0 (`/usr/local/cuda-13.0`), `rdma-core` with `libibverbs-dev`, Python 3.12,
  `openssl`, `rsync`, `curl`. **On rank 0 also:** CMake 3.28 or later, g++ 13.3 (C++20), the Rust toolchain (`cargo`), Docker (run by the login without sudo: membership of the `docker` group), the
  Hugging Face command line (`hf`, from `pip install -U huggingface_hub`), and the Python packages `numpy` (for `make_ablit.py` and the image test)
  and `Pillow` (for the image tests). The measured machines ran DGX OS 7.6 (kernel 7.0, driver
  580.178.04); [NETWORK.md](../NETWORK.md#3-the-mikrotik-crs812-measured-here-from-its-factory-state) lists them.
- **The kernel setting `kho=off`.** On kernel 7.0.0-1019-nvidia the RDMA memory registration (`ibv_reg_mr`) fails without it
  ([findings.md](../ds41f-4xgx10-switchless/docs/findings.md#hardware-measured-here)). Check it with
  `grep -o kho=off /proc/cmdline`; if it is not set and the engine's start fails in `ibv_reg_mr`, add it (`kho=off` on
  `GRUB_CMDLINE_LINUX_DEFAULT` in a file in `/etc/default/grub.d/`, `sudo update-grub`, reboot).
- **Passwordless sudo** for two commands on every Spark ([section 7](#7-sudo-the-services-and-the-ring-check)), and user services that start
  at boot: `sudo loginctl enable-linger <user>`.
- **For `--engram-vram on`** (in the served line): the login owns `/dev/dri/card0` on every Spark and no screen is connected. Add the
  login to the group that owns the card (`ls -l /dev/dri/card0`, usually `video`) and log in again. A Spark that cannot open it keeps the Engram cache in ordinary memory and
  prints one start line saying why (`--engram-vram off` says it on purpose).
- **No other model on the GPUs.** A start refuses when a Spark has less than `MIN_AVAIL_GIB` (100) of free memory after the cache drop.

## 2. The network

Follow [NETWORK.md](../NETWORK.md) for the switch, each Spark's addresses and its flow control, and [`network/`](../network/) for the
files it names. It ends in two checks that must pass before the engine runs: a full-size ping with the don't-fragment bit on both rails
between every pair (`ping -M do -s 8972 <peer-rail-address>`), and `dcb pfc show dev <rail>` giving `3:on` after a reboot. Each Spark
has one port and two rails (`mlx5_0`, `mlx5_2`), so the fleet file below names the same two devices and addresses as "next" and "previous".

Copy [`deploy/fleet.conf.example`](../deploy/fleet.conf.example) to `~/dsgb10/fleet.conf` on rank 0 and fill in every `<...>`: each
Spark's `hostname` (the engine finds its rank by it), its ssh address (ranks 1-3), and its rail A and rail B addresses. `GID_INDEX` is the
index whose type is `RoCE v2` for the rail's IPv4 address (`cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/3`). Every key is
described in [the fleet file step of pico-dgx-ds4's engine README](../ds41f-4xgx10-switchless/README.md#how-to-use). The engine checks each GID against these addresses at its
start and stops at a mismatch, naming the device and the file.

## 3. The checkpoint

The served checkpoint is [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The confidence network that decides how many drafts a step checks (`tools/stack4/cfmlp_cml.bin`) was fitted on it; on another checkpoint the
output stays the model's own but the speed gain can differ. Download it on rank 0 only:

```bash
mkdir -p ~/dsgb10/model
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ~/dsgb10/model
```

(`hf download REPO --revision REV --local-dir DIR` is the Hugging Face command line's syntax; this repository does not run it, so check
`hf download --help` on your install.) The front-end reads `~/dsgb10/model/tokenizer.json`, the media worker
`~/dsgb10/model/inference/config.json`, and the converters the safetensors shards. The engine never reads the checkpoint; it reads the
files of [section 5](#5-the-files-each-spark-needs).

## 4. Build and install

On rank 0, from `ds41f-4xgx10-switchless`, for `sm_121a` with CUDA 13.0's `nvcc`. A build with many jobs takes a lot of memory and an
out-of-memory stop can hang a GB10, so keep `-j` modest:

```bash
cmake -S engine -B engine/build && cmake --build engine/build -j4
(cd frontend && cargo build --release -j4)
```

This makes `engine/build/server/dsgb10_rank` (the engine), `engine/build/weights/{dsw_convert,dsw_verify,dsw_load}`,
`engine/build/engram/engram_pack` and `frontend/target/release/dsfront`. Make the folders on every Spark, then install:

```bash
mkdir -p ~/dsgb10/{bin,logs,run,spill,weights,vision,engram}
for w in <rank1-lan-address> <rank2-lan-address> <rank3-lan-address>; do ssh $w 'mkdir -p ~/dsgb10/{bin,logs,run,spill,weights,vision,engram}'; done
cp ~/dsgb10/fleet.conf ~/dsgb10/bin/fleet.conf
../deploy/pico-ctl.sh install "$PWD"      # binaries, confidence network, clock tool to ~/dsgb10/bin; units to ~/.config/systemd/user; starts nothing
```

The first start copies the engine binary, `fleet.conf`, `cfmlp_cml.bin` and `ablit.bin` to the workers by itself; everything else in
[section 5](#5-the-files-each-spark-needs) you place on each Spark yourself.

## 5. The files each Spark needs

Everything is cut on rank 0 and copied out. Copy over the fabric (the rail A address of each Spark) with `rsync -a` or `scp`; a copy over
the LAN is slow (check the fabric's rate first: [NETWORK.md, section 6](../NETWORK.md#6-checking-it)). Use these variables:
`B=engine/build CKPT=~/dsgb10/model OUT=~/dsgb10/cut` and make `$OUT` on a disk with room.

| File | Made by | Where | Placed by |
|---|---|---|---|
| `rank<r>.dsw` | `dsw_convert` | `~/dsgb10/weights/` of Spark r | you |
| `vision.rank<r>.dsw` | `dsw_convert --vision` | `~/dsgb10/vision/` of Spark r | you |
| `engram-l1-r<r>of4.bin`, `engram-l14-r<r>of4.bin` | `engram_pack` | `~/dsgb10/engram/` of Spark r | you |
| `engram_hashmap.bin` | `make_hashmap_stdlib.py` | `~/dsgb10/engram/` of every Spark | you |
| `ablit.bin` | this repository (`tools/ablit/ablit.bin`) | `~/dsgb10/` of rank 0 | the start helper copies it to the workers |
| `cfmlp_cml.bin` | in the repository (`tools/stack4/`) | `~/dsgb10/bin/` of every Spark | `pico-ctl.sh install` on rank 0, the start helper on the workers |
| `copy-T.txt` (optional) | `tools/copy/fit.py` | the same path on every Spark | you |
| the SSD tier, the kept sessions | the engine | `~/dsgb10/spill/` | nothing to place |

**5.1 The rank files** (format version 3: the engine refuses a file of another version, so after an upgrade that changes the version, cut
them again):

```bash
$B/weights/dsw_convert --ckpt $CKPT --rank 0 --plan         # the partition and the size a rank holds, no file written
for r in 0 1 2 3; do
  $B/weights/dsw_convert --ckpt $CKPT --rank $r --out $OUT/rank$r.dsw      # tiles on the GPU
  $B/weights/dsw_verify  --ckpt $CKPT --file $OUT/rank$r.dsw --threads 4   # every byte compared against the checkpoint
done
```

Put `rank<r>.dsw` in `~/dsgb10/weights/` of Spark `r`. On each Spark check what arrived (reads the file once, checks every tensor's CRC on the
GPU and that the file belongs to that rank; copy `dsw_load` over with the file): `dsw_load ~/dsgb10/weights/rank<r>.dsw --verify-crc --rank <r>`.
The layout is in [layout.md](layout.md). Nothing is re-quantised: the engine converts the checkpoint's FP8 and FP4 layouts only.

**5.2 The vision files** (a quarter of the image encoder on each rank, 233 MiB; needs no GPU):

```bash
for r in 0 1 2 3; do
  $B/weights/dsw_convert --vision --ckpt $CKPT --rank $r --out $OUT/vision.rank$r.dsw
  $B/weights/dsw_verify  --vision --ckpt $CKPT --file $OUT/vision.rank$r.dsw --threads 4
done
```

Put `vision.rank<r>.dsw` in `~/dsgb10/vision/` of Spark `r`. The ranks check at the start that the four files are one set
([media.md](media.md#the-engines-encoder)).

**5.3 The Engram shards and the hash map.** Each Spark holds its own two shards and the one hash map:

```bash
for r in 0 1 2 3; do $B/engram/engram_pack $CKPT $OUT $r; done        # engram-l1-r<r>of4.bin and engram-l14-r<r>of4.bin
python3 engine/engram/py/make_hashmap_stdlib.py $CKPT $OUT/engram_hashmap.bin
```

Put both shards of rank `r` and `engram_hashmap.bin` in `~/dsgb10/engram/` of Spark `r`. The engine checks each shard's header and the
map's constants at its start and names a bad file ([engram.md](../ds41f-4xgx10-switchless/docs/engram.md)).

**5.4 The abliteration file** (`ablit.bin`, serves the `-ablit` model name; [ablit.md](ablit.md)). It is in this repository,
made for this checkpoint: copy it to rank 0.

```bash
cp tools/ablit/ablit.bin ~/dsgb10/ablit.bin
sha256sum ~/dsgb10/ablit.bin    # 42b85416c5c61dae4f1ec6450c86634a1dffc271297a80e414304342f2d3fdd4
```

Keep it on rank 0 only. Leave `--ablit` out of the settings to serve the checkpoint's own behaviour alone.

**5.5 The confidence network.** `tools/stack4/cfmlp_cml.bin` is in the repository ([stack4.md](../ds41f-4xgx10-switchless/docs/stack4.md)); `pico-ctl.sh install` puts it in
`~/dsgb10/bin/` and `prepare` copies it to the workers. The engine stops at its start if the file named by `DSGB10_CF_MLP` is missing on a Spark.

**5.6 The copy table** (optional). Copy drafting uses the engine's built-in table when `DSGB10_COPY_TABLE` is not set, and the served line
does not set it; the built-in table runs high under sampling. `tools/copy/fit.py` fits one from dumped data (`DSGB10_COPY_DUMP=1`, see
[batching.md](batching.md)); add `--setenv DSGB10_COPY_TABLE=<path>` to `ENGINE_ARGS` and put the file at the same path on **every**
Spark: a Spark without it stops the engine at its start (`copy: DSGB10_COPY_TABLE=... cannot read it on this rank`), and neither
`install` nor `prepare` copies it. The engine follows the traffic by itself and the table is only its starting values
(`DSGB10_COPY_ADAPT=0` turns that off); every Spark must have the same copy settings and table file (the engine compares them at its start and stops if they differ).

**5.7 The API key** (rank 0): `(umask 077; openssl rand -hex 24 > ~/dsgb10/api-key)`. The front-end reads it from the file; it is never on a
command line. Every request needs it as `Authorization: Bearer <key>`, and the dashboard asks for it once.

## 6. The settings

Copy [`deploy/pico.env.example`](../deploy/pico.env.example) to `~/dsgb10/pico.env` on rank 0 and replace every `<...>`. The file is read by
systemd's rules (`KEY=value`, no quotes, no `~` expansion except in the engine's own `--weights`) and by `pico-ctl.sh` the same way. The
engine line is the one the speed numbers in the README were measured with; the README's [As served](../README.md#as-served) says what each part
does. The ones you will most likely change:

| Setting | Meaning | Leave out or change when |
|---|---|---|
| `--max-ctx 4194304` | the largest context (1M, 2M and 4M open on demand) | `1048576` for a 1M-only engine |
| `--slots 32 --big-slots 0` | 32 lines, each able to grow to the whole context | fewer lines use less memory |
| `--pool-free-gib 6.3` | the memory the engine leaves to the host on every Spark at start, with every other user at its cap ([store.md](store.md#what-stays-free)) | raise it to leave more |
| `--engram-vram on`, `--engram-cache-mib 2032` | the Engram cache sits in the display reservation | `off` without the card |
| `--ablit`, `--vision-weights` | the abliterated name; images and clips | leave out to serve without |
| `DSGB10_RING_DIRECT=2` | every rank writes to every other rank (one switch) | leave out on a cabled ring |
| `--prefill-rows-max 16384` | long prompts are processed 16,384 rows at a pass on a second set of row buffers the store lends while it is free ([README](../README.md#long-prompts-processed-16384-rows-at-a-time)) | `0` reads 8,192 rows at a pass only |
| `DSGB10_RING_RSBAL=1`, `DSGB10_PF_TUNE=hc,small,down` | pico-dgx-ds4's balanced reduce-scatter and prefill tuning, switched on in the served settings | leave out for the plain prefill and reduce-scatter |
| `KEEP_GIB=64` | sessions a planned stop keeps | `0` keeps none |
| `GPU_CLOCK=2200` | the GPU clock limit set at every start | `2400`, `2000` or `1800`; never unlocked |
| `--effort 75`, `--thinking`, `--max-tokens` in `FRONT_ARGS` | the front-end's defaults for a request that names none | per your users |

`--front-mib` must have the same number in `ENGINE_ARGS` and `FRONT_ARGS` (a start with `--pool-free-gib` is refused without it).
`--media-mib` must match the container's `--memory` in `pico-media.service`. The front-end listens on loopback and on `LAN_ADDR`; repeat
`--listen` for another address and never use `0.0.0.0`.

## 7. Sudo, the services and the ring check

**Sudo.** `pico-ctl.sh prepare` needs two commands without a password on every Spark: the GPU clock lock and the page-cache drop. The
narrowest lines for a login `<user>` (put them in `/etc/sudoers.d/pico-dgx` with `sudo visudo -f /etc/sudoers.d/pico-dgx`, and check the
program paths with `command -v nvidia-smi sh`; review them against your own policy before use):

```
<user> ALL=(root) NOPASSWD: /usr/bin/nvidia-smi -lgc 0\,2400, /usr/bin/nvidia-smi -lgc 0\,2200, /usr/bin/nvidia-smi -lgc 0\,2000, /usr/bin/nvidia-smi -lgc 0\,1800
<user> ALL=(root) NOPASSWD: /bin/sh -c sync; echo 3 > /proc/sys/vm/drop_caches
```

The comma in `0,2200` is escaped because sudoers reads a comma as a list separator. A refused page-cache drop is logged and the start goes
on; a refused clock lock stops the start. Check with `sudo -n -l` that the login lists both, and with `sudo -n nvidia-smi -lgc 0,2200`.

**The ring check.** With the engine stopped, run the engine's own transport test on all four Sparks (each rank prints `passed`), once as
the fleet file describes and once in the direct mode the served line uses:

```bash
FLEET=~/dsgb10/fleet.conf engine/server/ring.sh engine/comm/run4.sh engine/build/comm/ring_test
DSGB10_RING_DIRECT=2 FLEET=~/dsgb10/fleet.conf engine/server/ring.sh engine/comm/run4.sh engine/build/comm/ring_test
```

`DSGB10_RING_DIRECT` is read by `ring_test` itself (`engine/comm/ring.cpp`: 0 off, 1 direct, 2 direct and merged).

**The units**, on rank 0:

```bash
systemctl --user daemon-reload
systemctl --user enable pico-media pico-engine pico-front
```

`pico-engine.service` runs rank 0 in the foreground and starts ranks 1-3 through `pico-ctl.sh prepare`; `pico-front.service` runs `dsfront`
and holds requests while the engine restarts; `pico-media.service` is [section 8](#8-the-media-worker). `engine/server/service.sh` is a second
way to run the engine without systemd (pico-dgx-ds4's [engine.md](../ds41f-4xgx10-switchless/docs/engine.md#as-a-service-servicesh)); use one way on a set of Sparks.

## 8. The media worker

The worker reads images and clips (Pillow, PyAV; no model, no GPU) and hands the pixels to the front-end, so image parsers stay out of the
engine ([media.md](media.md)). Build its image on rank 0 (the base image is pinned to its `linux/arm64` manifest) and name the tag in
`MEDIA_IMAGE`:

```bash
(cd tools/vision && docker build -f Dockerfile.media-worker -t pico-media:rev1 .)
```

`pico-media.service` runs it with host networking, no GPU, no capabilities, a read-only root, 512 MiB with no swap, and the checkpoint's
`inference/config.json` as its only mount. The front-end reaches it at `127.0.0.1:18951` (`--media-addr` in `FRONT_ARGS`); without that
argument image requests are refused.

## 9. Fans and the GPU clock

**The clock.** The measurements ran with the GPU limited to 2,200 MHz on all four Sparks. `pico-ctl.sh prepare` sets `GPU_CLOCK` on all four
at every start through `tools/gpu-clock` (in this folder; `GPU_CLOCK_TOOL` in `pico.env`): `gpu-clock LIMIT [ssh-target ...]` locks this Spark and each
target to 2,400, 2,200, 2,000 or 1,800 MHz (never unlocked), prints one line a Spark and records `{"limit","at","ok","error"}` in the file
named by `GPU_CLOCK_STATE` on the Spark it runs on (the targets are set, not recorded; at a start that is rank 0). Without `GPU_CLOCK_TOOL` the helper calls `sudo -n nvidia-smi -lgc 0,LIMIT` itself and nothing is
recorded. A lock lasts until the Spark reboots. Check on every Spark under load with `nvidia-smi --query-gpu=clocks.sm --format=csv`.

**The fans and the dashboard's COOLING AND CLOCK box.** Install the dgx-spark-fan-control project's driver and service on every Spark by its
installation guide (a kernel module signed with a key you enrol, which needs the console once) and its `dgx-fan-fleet` on rank 0:

```bash
sudo install -m 0755 tools/dgx-fan-fleet /usr/local/bin/dgx-fan-fleet      # from that project's checkout
dgx-fan-fleet status                                                         # every Spark answers
```

The box needs `--fan-tool /usr/local/bin/dgx-fan-fleet` in `FRONT_ARGS` and, in `pico.env`, `DGX_FAN_NODES` (the other Sparks as ssh
targets), `DGX_FAN_CLOCK_TOOL` (the same `gpu-clock`) and `DGX_FAN_CLOCK_STATE` (the same state file). Without them the box is absent and
the fans stay under each Spark's own controller. Profiles and commands: [operations.md](operations.md#fans-and-the-gpu-clock).

## 10. First start and checks

```bash
systemctl --user start pico-media pico-engine pico-front
journalctl --user -u pico-engine -f          # the start; ready within a few minutes
```

A good start shows, in order: `pico-ctl: ranks 1-3 started`; on every rank `the store's kernels ran once`; the start tables (the memory
lines and `chunk pool:`, [store.md](store.md)); `vision work area: ... the engine encodes images` on every rank; the Engram cache line;
`ready after N s` (19 s on the measured machines after a restart); and `serve: listening on ...`. Then:

```bash
K=$(cat ~/dsgb10/api-key)
curl -s http://127.0.0.1:8000/health                                       # {"status": "ok"} without the key, whole with it
curl -s -H "Authorization: Bearer $K" http://127.0.0.1:8000/v1/models      # the served name and the name with -ablit
curl -s -H "Authorization: Bearer $K" -H 'Content-Type: application/json' http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"Say pong"}],"max_tokens":2000}'
```

- **Memory.** On every Spark `awk '/^MemAvailable:/ {print $2/1048576 " GiB"}' /proc/meminfo` shows more than the `--pool-free-gib` you set at an idle engine, because the
  outside users of memory are not at their caps then ([store.md](store.md#what-stays-free)).
- **The `-ablit` name** answers with `"model":"deepseek-v4.1-flash-1017-ablit"`; without `--ablit` on the engine it runs as the plain name.
- **Images:** `KEYFILE=~/dsgb10/api-key MODEL=deepseek-v4.1-flash-1017 python3 tools/vision/imgtest.py http://127.0.0.1:8000/v1/chat/completions IMGDIR OUT.json`
  (the images and `cases.json` come from `python3 tools/vision/make_test_images.py IMGDIR`; it also fetches one public image by URL);
  clips: `tools/vision/smoke_video.py` ([media.md](media.md#tests)).
- **The dashboard:** open `http://<rank0-lan-address>:8000/dashboard`; the page asks for the API key once and keeps it in the browser until
  SIGN OUT. The tiles move under load; NODE HEALTH has a row for each Spark. `frontend/tools/dashboard_smoke.py` checks the routes without an
  engine ([dashboard.md](dashboard.md)).
- **Speed:** `DSGB10_KEY_FILE=~/dsgb10/api-key python3 ../tools/bench/bench.py http://127.0.0.1:8000/v1/chat/completions c1 9` (one user)
  and `... distinct 16` (16 users, different prompts), with the dashboard open, as the README's numbers were measured. The README's
  one-user figures are medians of 27 requests of each kind after a clean boot of all four Sparks; 9 is a quicker check.
- **Correctness:** `python3 engine/server/tests/serve_test.py --ced --socket ~/dsgb10/run/engine.sock --tokens <tokens.npy>` against the running
  engine, with `<tokens.npy>` made as [`golden/README.md`](../ds41f-4xgx10-switchless/golden/README.md) describes. The reference hashes (`golden/refhash.sh`, engine
  stopped) need the golden data of that page, which the repository does not ship. The reference hashes in pico-dgx-ds4's engine README
  belong to another checkpoint revision; `golden/README.md` says which apply to the served one.
- **A planned stop keeps sessions:** `systemctl --user restart pico-engine`, then a second turn of a long conversation answers from the cache
  (`usage.prompt_tokens_details.cached_tokens`).

## 11. Where things go wrong first

| Symptom | Cause and fix |
|---|---|
| `pico-ctl: MemFree under 100 GiB ... on <spark>` | another model, a leftover rank or cached files; `pgrep -x dsgb10_rank`, stop what holds memory, retry. Do not lower `MIN_AVAIL_GIB` below what a rank needs (about 90 GiB). |
| `pico-ctl: the GPU clock was not set on <spark>` | the sudoers lines of [section 7](#7-sudo-the-services-and-the-ring-check) are missing on that Spark. |
| the start stops with `comm: ... GID ... but the fleet file ... has ...` | `GID_INDEX` or an address in the fleet file does not match the rail's IPv4 address ([section 2](#2-the-network)). |
| `comm: this host, X, is no RANK<r>_HOSTNAME` | `hostname` on that Spark differs from the fleet file's. |
| `ibv_reg_mr` fails | the kernel lacks `kho=off` ([section 1](#1-what-you-need)). |
| a rank stops at its start naming a file | that Spark lacks its rank file, vision file, shard, hash map, `ablit.bin`, `cfmlp_cml.bin` or a copy table at the path in `pico.env` or the fleet file. |
| `the start is refused: the store ... cannot hold one conversation of --max-ctx` | lower `--max-ctx`, `--slots` or `--engram-cache-mib`, or free memory ([README, memory store](../README.md#the-memory-store)). |
| image requests answer "this engine has no image encoder" | the engine started without `--vision-weights`, or the start line `vision work area ... the engine has no image encoder` is there. |
| image requests are refused by the front-end | no `--media-addr` in `FRONT_ARGS`, or `pico-media` is not running (`docker logs pico-media`). |
| requests answer 401 | the key sent is not the key file's content. |
| requests wait, then 503 | the engine is restarting for longer than `HOLD_S`; `journalctl --user -u pico-engine`. |
| the dashboard's COOLING AND CLOCK box is missing | no `--fan-tool` in `FRONT_ARGS`, or the tool's settings are missing ([section 9](#9-fans-and-the-gpu-clock)). |
| throughput is low from the first request | a GPU not at its clock (`nvidia-smi --query-gpu=clocks.sm --format=csv` on all four), a rail at x2 ([NETWORK.md, section 4](../NETWORK.md#4-each-spark)), or drops on the fabric ([NETWORK.md, section 6](../NETWORK.md#6-checking-it)). |
