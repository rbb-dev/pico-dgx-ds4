#!/usr/bin/env python3
"""Host test of store_trace_check.py `same` on a real case: four ranks' traces of a run whose give-back failed on rank 3
(testdata/tenant_fail_back: the other ranks tried to grow the pool inside the command, rolled it back, and all four recorded the
debt). `same` must say SAME for the four, and must still say DIFFERENT for a digest that differs after the command and for an
attempt that does not return to the digest it started from."""
import gzip
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "store_trace_check.py")
DATA = os.path.join(HERE, "testdata", "tenant_fail_back")
bad = 0


def ok(cond, what):
    global bad
    print("%s %s" % ("ok  " if cond else "FAIL", what))
    bad += 0 if cond else 1


def same(paths):
    p = subprocess.run([sys.executable, CHECK, "same"] + paths, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    files = [os.path.join(DATA, "store.rank%d.1.t1.gz" % r) for r in range(4)]
    rc, out = same(files)
    ok(rc == 0 and "SAME" in out and "rolled-back" in out, "the four ranks of the failed give-back are the same, rank 0-2's rolled-back attempt left out")
    texts = [gzip.open(f, "rt").read().split("\n") for f in files]
    with tempfile.TemporaryDirectory() as tmp:
        def write(name, lines):
            p = os.path.join(tmp, name)
            open(p, "w").write("\n".join(lines))
            return p

        # a digest that differs after the command: rank 1's digest after the settle-free line 'out=26,12' (the debt recorded)
        t1 = list(texts[1])
        k = next(i for i, l in enumerate(t1) if l.startswith("D ") and "out=26,12," in l)
        t1[k] = t1[k].replace(" d=", " d=0", 1)
        rc, out = same([files[0], write("r1.t1", t1), files[2], files[3]])
        ok(rc == 1 and "DIFFERENT" in out, "a digest that differs after the command still fails")
        # an attempt that does not return to the digest it started from is not dropped
        t0 = list(texts[0])
        k = next(i for i, l in enumerate(t0) if l.startswith("E ") and " alloc_shrink " in l) + 1
        t0[k] = t0[k].replace(" d=", " d=1", 1)
        rc, out = same([write("r0.t1", t0), files[3]])
        ok(rc == 1 and "DIFFERENT" in out, "an attempt that does not return to its digest is not left out")
        # rank 3 itself with a wrong digest after the command
        t3 = list(texts[3])
        k = next(i for i, l in enumerate(t3) if l.startswith("D ") and "out=26,12," in l)
        t3[k] = t3[k].replace(" w=", " w=0", 1)
        rc, out = same([files[0], write("r3.t1", t3)])
        ok(rc == 1 and "DIFFERENT" in out, "a wide digest that differs after the command fails")
    print("%s store_trace_check_test: %d failed" % ("FAIL" if bad else "PASS", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
