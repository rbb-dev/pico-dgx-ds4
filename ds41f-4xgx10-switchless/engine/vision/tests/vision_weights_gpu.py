#!/usr/bin/env python3
"""Checks of the vision weight files that need the four Sparks (docs/vision.md). Not run by the builder.
This script only reads logs; the commands that make them are below, to be run with the engine stopped.

1. Cut the four vision weight files once, on rank 0's Spark, from the checkpoint's shards (the tower is in shards 1 and 2; the
   rest of the checkpoint is left out; it reads 970,536,960 bytes and writes 4 x 244 MB; no GPU is touched):

     CK=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
     mkdir -p ~/dsgb10/vision-cut
     for r in 0 1 2 3; do
       build/weights/dsw_convert --vision --ckpt $CK --rank $r --out ~/dsgb10/vision-cut/vision.rank$r.dsw --threads 4 --mem-gib 2
     done                                  # prints: 266 tensors a rank, coverage ok
     for r in 0 1 2 3; do
       build/weights/dsw_verify --vision --ckpt $CK --file ~/dsgb10/vision-cut/vision.rank$r.dsw
     done                                  # every byte against the checkpoint, "0 bad"
     docker run --rm -v $CK:/models:ro -v ~/dsgb10/vision-cut:/dsw:ro -v REPO:/work IMAGE_WITH_TORCH \\
         python3 /work/engine/weights/check_dsw.py /models /dsw/vision.rank0.dsw    # and ranks 1-3: "vision file: all 266 ..."

2. Place them: rank r's file to ~/dsgb10/weights/vision.rank<r>.dsw on rank r's Spark (over the fabric addresses; rank 0
   keeps its own).

3. Starts, each through the tool the other engine tests use (pico-ctl prepare, the double
   reset): the same arguments, the same settings, once without and once with
     --vision-weights '~/dsgb10/weights/vision.rank{rank}.dsw'
   keep rank 0's log of each: LOG_WITHOUT, LOG_WITH. (--pool-free-gib must be in the settings: the offer line needs it.)

     python3 vision_weights_gpu.py offer LOG_WITH [LOG_WITHOUT]
         rank 0 loaded the file (266 tensors; its "vision weights ... set digest" line; the engine itself ends the
         start on every rank when the four digests differ, so a start that got this far had one value), the start line "vision weights: ... lower ... by: a b c d numbers"
         gives 84 or 85 on every device-bound rank (the weights are 244,002,816 B in the arena + 1,311,232 B of padded
         patch embedding and table = 84.12 numbers of 2,916,352 B), and LOG_WITHOUT has no vision line.

4. Text goldens unchanged: the golden modes of docs/engine.md "Golden check" run with the flag and without it:
     engine/server/run4.sh --mode prefill --inject --vision-weights '~/dsgb10/weights/vision.rank{rank}.dsw'
     engine/server/run4.sh --mode decode --golden /data/dsgb10/golden/long --ctx 1024 --steps 256 --big --vision-weights ...
   and the same without the flag; then
     python3 vision_weights_gpu.py golden LOG_A LOG_B
         every line that carries a hash, a token or a logit error (the lines that start with "golden", "layer",
         "hash", "token", "max") is identical in the two logs.

5. One rank's file swapped: start with rank 1's file named vision.rank2.dsw on box 2 (a copy of the file, in a scratch
   directory, passed as --vision-weights '/path/to/scratch/vision.rank2.dsw' for rank 2 only is not possible with one
   flag for all ranks: put the swapped file at ~/dsgb10/weights/vision.rank2.dsw on box 2 for this start, put it back
   after). The start must refuse; keep rank 2's log:

     python3 vision_weights_gpu.py swap LOG_RANK2
         the log says the file was cut for rank 1 of 4 but this process is rank 2 of 4, and has no "chunk pool" line.
     A mixed set that is not a rank swap (a file from another checkpoint, same rank) is refused by the digest vote on
     every rank: its message is "the vision weight files are not one set" (check with the same subcommand: `vote`).

   `python3 vision_weights_gpu.py selftest` checks this script's parsers against lines in the engine's own format.
"""
import re
import sys

DIGEST = re.compile(r"rank (\d): vision weights \S+: (\d+) tensors, ([\d.]+) MiB .*set digest ([0-9a-f]{16})")
OFFER = re.compile(r"vision weights: (\d+) bytes on every rank \(([\d.]+) numbers of ([\d.]+) MiB\).* by:((?: \d+)+) numbers")
SWAP = re.compile(r"cut for rank (\d) of 4, but this process is rank (\d) of 4")


def read(p):
    with open(p, errors="replace") as f:
        return f.read()


def offer(with_log, without_log=None):
    t = read(with_log)
    ok = True
    digs = DIGEST.findall(t)
    print(f"digest lines: {len(digs)}")
    for r, n, mib, d in digs:
        print(f"  rank {r}: {n} tensors, {mib} MiB, digest {d}")
        ok &= n == "266"
    if digs:
        ok &= len({d for _, _, _, d in digs}) == 1  # one line (rank 0's) in rank 0's log; more if the logs are joined
    else:
        print("FAIL no vision weights line"), sys.exit(1)
    m = OFFER.search(t)
    if not m:
        print("FAIL no 'vision weights: ... lower ... by:' line (start with --pool-free-gib and --vision-weights)")
        sys.exit(1)
    nums = [int(x) for x in m.group(4).split()]
    print(f"offers lower by {nums} numbers ({m.group(1)} bytes = {m.group(2)} numbers of {m.group(3)} MiB)")
    for n in nums:
        ok &= n in (84, 85) or n == 0  # 0: a host-bound rank (its offer is not bound by device memory)
    ok &= sum(1 for n in nums if n) >= 1
    if without_log and DIGEST.search(read(without_log)):
        print("FAIL the log without the flag has a vision line")
        ok = False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


KEEP = re.compile(r"^(golden|layer|hash|token|max|first|prefill|decode|tf)\b", re.I)


def golden(a, b):
    la = [l.rstrip() for l in read(a).splitlines() if KEEP.match(l)]
    lb = [l.rstrip() for l in read(b).splitlines() if KEEP.match(l)]
    # a time or a rate varies from run to run; hashes, tokens and errors do not
    strip = lambda l: re.sub(r"\d+(\.\d+)? ?(ms|s|tok/s|GB/s|MiB|GiB)\b", "T", l)
    la, lb = [strip(l) for l in la], [strip(l) for l in lb]
    print(f"{len(la)} and {len(lb)} result lines")
    if not la or la != lb:
        for x, y in zip(la, lb):
            if x != y:
                print("DIFF", x, "|", y)
                break
        print("FAIL")
        sys.exit(1)
    print("PASS")


def swap(log):
    t = read(log)
    m = SWAP.search(t)
    refused = m is not None and "chunk pool:" not in t
    print("refused:", m.group(0) if m else None)
    print("PASS" if refused else "FAIL")
    sys.exit(0 if refused else 1)


def vote(log):
    t = read(log)
    ok = "the vision weight files are not one set" in t and "chunk pool:" not in t
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def selftest():
    import tempfile, os
    d = tempfile.mkdtemp()
    w = os.path.join(d, "with.log")
    open(w, "w").write(
        "rank 0: vision weights /h/vision.rank0.dsw: 266 tensors, 232.70 MiB in 0.09 s (2.70 GB/s), 233.95 MiB on the device with the padded patch embedding; set digest 00112233445566ff\n"
        + "".join(f"rank {r}: vision weights /h/vision.rank{r}.dsw: 266 tensors, 232.70 MiB in 0.09 s (2.70 GB/s), 233.95 MiB on the device with the padded patch embedding; set digest 00112233445566ff\n" for r in (1, 2, 3))
        + "vision weights: 245314048 bytes on every rank (84.12 numbers of 2.78 MiB); each rank's own offer is lower than without them, at the same readings, by: 84 84 85 84 numbers\n")
    o = os.path.join(d, "without.log")
    open(o, "w").write("chunk pool: 7608 numbers\n")
    try:
        offer(w, o)
    except SystemExit as e:
        assert e.code == 0, "offer parser"
    s = os.path.join(d, "swap.log")
    open(s, "w").write("/h/vision.rank2.dsw: the file was cut for rank 1 of 4, but this process is rank 2 of 4 (a weight file serves only the rank it was cut for)\n")
    try:
        swap(s)
    except SystemExit as e:
        assert e.code == 0, "swap parser"
    print("selftest ok")


if __name__ == "__main__":
    c = sys.argv[1:2]
    if c == ["offer"] and len(sys.argv) in (3, 4): offer(*sys.argv[2:])
    elif c == ["golden"] and len(sys.argv) == 4: golden(*sys.argv[2:])
    elif c == ["swap"] and len(sys.argv) == 3: swap(sys.argv[2])
    elif c == ["vote"] and len(sys.argv) == 3: vote(sys.argv[2])
    elif c == ["selftest"]: selftest()
    else: print(__doc__), sys.exit(2)
