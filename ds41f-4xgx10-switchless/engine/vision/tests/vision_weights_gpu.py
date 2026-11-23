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

3. Three starts on one boot, each through the tool the other engine tests use (the engine's own start script, or the double reset), the same arguments and settings, --pool-free-gib among them: two WITHOUT the flag
   (their difference per rank is the null spread s) and then one WITH
     --vision-weights '~/dsgb10/weights/vision.rank{rank}.dsw'
   keep rank 0's log of each (LOG_NO_A, LOG_NO_B, LOG_WITH) and, for the with-flag start, the workers' logs.

     python3 vision_weights_gpu.py offer LOG_NO_A LOG_NO_B LOG_WITH [WORKER_LOG ...]
         rank 0's "chunk pool:" line carries every rank's free device / available host memory at the pool's sizing
         read. Per rank and per reading the flag-less mean minus the with-flag reading must equal the weights' device
         bytes (the "MiB on the device" of the with-flag log's vision line, 234.0 MiB when it is not there) within
         s + 10.24 MiB (s = |A - B| of the two flag-less starts; 10.24 MiB = the two-decimal GiB resolution). Exit 2
         (instrument invalid) when s is not under the weights' bytes: the three starts then cannot see them.
         The with-flag log must have the vision line (266 tensors, one digest) and no flag-less log may. Each
         "vision weights held after the load" line found (rank 0's log and the WORKER_LOGs) must say 0 B of staging
         still held; its readings around the load are printed.

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
POOL = re.compile(r"free device / available host memory before it, ranks 0-3:((?: [\d.]+ / [\d.]+){4}) GiB")
HELD = re.compile(r"rank (\d): vision weights held after the load: ([\d.]+) MiB on the device .*pinned staging ([\d.]+) MiB at most and (\d+) B still held; "
                  r"readings around the load: device free fell by (-?[\d.]+) MiB, available host memory by (-?[\d.]+) MiB")
DEVMIB = re.compile(r"([\d.]+) MiB on the device with the padded patch embedding")
RES = 10.24  # MiB: a reading is GiB with two decimals; two readings' difference carries it
WEIGHTS_MIB = 234.0  # when the with-flag log does not say
SWAP = re.compile(r"cut for rank (\d) of 4, but this process is rank (\d) of 4")


def read(p):
    with open(p, errors="replace") as f:
        return f.read()


def readings(path):
    """rank 0's pool line: [(device GiB, host GiB)] for ranks 0-3, or None."""
    m = POOL.search(read(path))
    if not m:
        return None
    return [(float(x), float(y)) for x, y in re.findall(r"([\d.]+) / ([\d.]+)", m.group(1))]


def judge(no_a, no_b, with_, expect_mib):
    """Per rank and reading: (name, rank, mean_without, with, lower_by_mib, s_mib, ok). Returns (rows, valid)."""
    rows, valid = [], True
    for k, name in enumerate(("device", "host")):
        for r in range(4):
            a, b, w = no_a[r][k], no_b[r][k], with_[r][k]
            s = abs(a - b) * 1024
            lower = ((a + b) / 2 - w) * 1024
            valid &= s < expect_mib
            rows.append((name, r, (a + b) / 2, w, lower, s, abs(lower - expect_mib) <= s + RES))
    return rows, valid


def offer(no_a, no_b, with_log, *workers):
    ok = True
    ra, rb, rw = readings(no_a), readings(no_b), readings(with_log)
    for p, r in ((no_a, ra), (no_b, rb), (with_log, rw)):
        if r is None:
            print(f"FAIL no 'chunk pool: ... free device / available host memory before it' line in {p}")
            sys.exit(1)
    t = read(with_log)
    digs = DIGEST.findall(t)
    print(f"digest lines: {len(digs)}")
    for r, n, mib, d in digs:
        print(f"  rank {r}: {n} tensors, {mib} MiB, digest {d}")
        ok &= n == "266"
    if not digs:
        print("FAIL no vision weights line in the with-flag log"), sys.exit(1)
    ok &= len({d for _, _, _, d in digs}) == 1
    for p in (no_a, no_b):
        if DIGEST.search(read(p)):
            print(f"FAIL {p}: a log without the flag has a vision line")
            ok = False
    m = DEVMIB.search(t)
    expect_mib = float(m.group(1)) if m else WEIGHTS_MIB
    print(f"the weights' device bytes: {expect_mib} MiB ({'the log' if m else 'default'})")
    rows, valid = judge(ra, rb, rw, expect_mib)
    for name, r, mean, w, lower, s, good in rows:
        print(f"  {name:6} rank {r}: without {mean:.3f} GiB (mean), with {w:.2f} GiB: lower by {lower:7.1f} MiB, "
              f"null spread {s:5.1f} MiB  {'ok' if good else 'OUT'}")
        ok &= good
    held = HELD.findall(t)
    for p in workers:
        held += HELD.findall(read(p))
    for r, dev, peak, still, dfell, hfell in held:
        print(f"  rank {r}: staging {peak} MiB at most, {still} B still held; around the load device free fell {dfell} MiB, host {hfell} MiB")
        ok &= still == "0"
    if not held:
        print("FAIL no 'vision weights held after the load' line (rank 0's log or a worker log)")
        ok = False
    if not valid:
        print("INVALID the null spread is not under the weights' bytes: the three starts cannot see the weights; repeat the starts")
        sys.exit(2)
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

    def pool(vals):
        return ("chunk pool: 8359 numbers of 2.78 MiB (22.67 GiB, 7.1M positions) on every rank (--pool-free-gib 8.0: 8359 on this rank; "
                "free device / available host memory before it, ranks 0-3: " + " ".join(f"{a:.2f} / {b:.2f}" for a, b in vals) +
                " GiB, the Engram cache's 6.0 to grow)\n")

    def vline(r, still=0):
        return (f"rank {r}: vision weights /h/vision.rank{r}.dsw: 266 tensors, 232.70 MiB in 0.09 s (2.70 GB/s), 232.80 MiB on the device "
                f"with the padded patch embedding; set digest 00112233445566ff\n"
                f"rank {r}: vision weights held after the load: 232.80 MiB on the device (the weights), pinned staging 64.00 MiB at most "
                f"and {still} B still held; readings around the load: device free fell by 232.90 MiB, available host memory by 233.10 MiB\n")

    def mk(name, vals, vision=False, still=0):
        p = os.path.join(d, name)
        open(p, "w").write((vline(0, still) if vision else "") + pool(vals))
        return p

    base = [(30.00, 31.00), (31.30, 31.50), (31.50, 31.60), (31.10, 31.40)]
    base2 = [(30.02, 31.01), (31.29, 31.52), (31.52, 31.58), (31.12, 31.38)]
    lo = lambda vals, gib: [(a - gib, b - gib) for a, b in vals]
    A, B = mk("a.log", base), mk("b.log", base2)

    def run(*args):
        try:
            offer(*args)
        except SystemExit as e:
            return e.code

    import io, contextlib
    def quiet(*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(*args)

    # the weights' 0.2273 GiB lower on every rank: passes
    assert quiet(A, B, mk("w1.log", lo(base, 0.2273), True)) == 0, "a passing set"
    # a worker log that says staging is still held fails
    wk = os.path.join(d, "worker.log"); open(wk, "w").write(vline(1, still=1).split("\n", 1)[1])
    assert quiet(A, B, mk("w1b.log", lo(base, 0.2273), True), wk) == 1, "staging held"
    # 0.6 GiB lower on rank 0 (and the others near it): fails, not invalid
    assert quiet(A, B, mk("w2.log", lo(base, 0.60), True)) == 1, "0.6 GiB lower"
    # uneven lowering: lower by 0.80 / 0.47 / 0.68 / 0.38 GiB
    assert quiet(A, B, mk("w3.log", [(a - x, b - x) for (a, b), x in zip(base, (0.80, 0.47, 0.68, 0.38))], True)) == 1, "uneven lowering"
    # no lowering at all: fails
    assert quiet(A, B, mk("w4.log", base, True)) == 1, "nothing lower"
    # a null spread larger than the weights: the instrument cannot judge: exit 2, whatever the with-flag start shows
    big = [(a - 0.5, b - 0.5) for a, b in base2]
    assert quiet(A, mk("b2.log", big), mk("w5.log", lo(base, 0.2273), True)) == 2, "null spread larger than the effect"
    # the log without the flag has a vision line, or the with-flag log has none: fail
    assert quiet(mk("a2.log", base, True), B, mk("w6.log", lo(base, 0.2273), True)) == 1, "flag-less log with a vision line"
    assert quiet(A, B, mk("w7.log", lo(base, 0.2273), False)) == 1, "no vision line"
    s = os.path.join(d, "swap.log")
    open(s, "w").write("/h/vision.rank2.dsw: the file was cut for rank 1 of 4, but this process is rank 2 of 4 (a weight file serves only the rank it was cut for)\n")
    assert quiet_swap(swap, s) == 0, "swap parser"
    print("selftest ok")


def quiet_swap(fn, arg):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            fn(arg)
        except SystemExit as e:
            return e.code


if __name__ == "__main__":
    c = sys.argv[1:2]
    if c == ["offer"] and len(sys.argv) >= 5: offer(*sys.argv[2:])
    elif c == ["golden"] and len(sys.argv) == 4: golden(*sys.argv[2:])
    elif c == ["swap"] and len(sys.argv) == 3: swap(sys.argv[2])
    elif c == ["vote"] and len(sys.argv) == 3: vote(sys.argv[2])
    elif c == ["selftest"]: selftest()
    else: print(__doc__), sys.exit(2)
