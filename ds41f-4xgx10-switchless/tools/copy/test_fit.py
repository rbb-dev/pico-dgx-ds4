#!/usr/bin/env python3
"""tools/copy/fit.py on a synthetic dump with known answers (python3 test_fit.py)."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = "tag,step,entry,pos0,emitted_n,emitted,next_pos0,chain_k,chain,src,match_len,m,copy\n"


def line(step, pos0, emitted, next_pos0, chain, src, ml, copy):
    return "t,%d,0,%d,%d,%s,%d,%d,%s,%d,%d,%d,%s\n" % (
        step, pos0, len(emitted), ";".join(map(str, emitted)), next_pos0, len(chain), ";".join(map(str, chain)), src, ml,
        len(copy), ";".join(map(str, copy)))


rows = [HEADER]
# four proposals of bucket 0 (match 8): the first token right in 3 of 4, the second right in 2 of 4 (a miss at depth 0
# kills the rest of that proposal)
step = 0
for i in range(4):
    pos0 = 100 + 100 * i
    # emitted token j sits at pos0 + 1 + j (the first is the bonus); the proposal made after it, with no chain drafts,
    # covers pos0 + 2 .., so copy token c is right when it equals emitted[1 + c]
    emitted = [1000 * (i + 1) + j for j in range(8)]
    want = emitted[1:4]
    copy = list(want)
    if i == 0:
        copy[0] += 1  # wrong at depth 0
    if i == 1:
        copy[1] += 1  # wrong at depth 1
    rows.append(line(step, pos0, emitted, pos0 + 1, [], 40, 8, copy))
    step += 1
# one with a chain that did not land: not scored
rows.append(line(step, 900, [5, 6, 7, 8], 901, [99], 40, 8, [1, 2, 3]))
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "dump.csv")
    open(p, "w").write("".join(rows))
    out = subprocess.run([sys.executable, os.path.join(HERE, "fit.py"), p, "--min-n", "1"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    t = [[float(x) for x in l.split()] for l in out.stdout.strip().splitlines()]
    assert len(t) == 4 and all(len(r) == 7 for r in t), t
    # bucket 0: 4 scored; depth 0 right in 3 (i = 1, 2, 3), depth 1 right in 2 (i = 2, 3: i = 0 died at depth 0)
    assert abs(t[0][0] - 0.75) < 1e-3, t[0]
    assert abs(t[0][1] - 0.5) < 1e-3, t[0]
    assert "4 proposals scored" in out.stderr, out.stderr
    # buckets without data keep the prior
    assert abs(t[1][0] - 0.93) < 1e-3 and abs(t[3][1] - 0.93 * 0.94) < 1e-3, t
    # non-increasing along the rows
    assert all(r[j] >= r[j + 1] - 1e-9 for r in t for j in range(6)), t


# two requests of one entry that use the same positions: each proposal is scored against its own request's tokens
def run(rows):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dump.csv")
        open(p, "w").write("".join(rows))
        out = subprocess.run([sys.executable, os.path.join(HERE, "fit.py"), p, "--min-n", "1"], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return [[float(x) for x in l.split()] for l in out.stdout.strip().splitlines()], out.stderr


xa = [100 + j for j in range(8)]
xb = [200 + j for j in range(8)]
rows = [HEADER,
        line(0, 10, xa, 11, [], 40, 8, xa[1:4]),  # request 1: its copy is right
        line(0, 10, xb, 11, [], 40, 8, [7, 7, 7])]  # request 2, the same positions and step: its copy is wrong
t, err = run(rows)
assert "2 proposals scored" in err and abs(t[0][0] - 0.5) < 1e-3, (err, t[0])
# a new request whose positions are above the last one's next rows is told by its lower step number (a batch dump)
rows = [HEADER,
        line(5, 10, xa, 11, [], 40, 8, xa[1:4]),
        line(0, 50, xb, 51, [], 40, 8, xb[1:4])]
t, err = run(rows)
assert "2 proposals scored" in err and abs(t[0][0] - 1.0) < 1e-3, (err, t[0])
print("fit.py: ok")
