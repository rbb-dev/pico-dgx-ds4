#!/usr/bin/env python3
"""Fit copy drafting's gate table T from the probe's dump (DSGB10_COPY_DUMP=1 with DSGB10_COPY_DUMP_FILE).

Writes T[bucket][j] = P(copy tokens 1..j+1 all right | match-length bucket, chain landed): one line per bucket, 7 numbers, the file DSGB10_COPY_TABLE reads. A bucket with fewer than --min-n scored proposals keeps the prior's row.
"""
import argparse
import collections
import csv
import sys

BUCKETS = ((8, 16), (16, 32), (32, 64), (64, 1 << 30))
ROWS = 7
PRIOR_Q = (0.90, 0.93, 0.96, 0.93)
PRIOR_C = (0.91, 0.93, 0.95, 0.94)


def bucket_of(length):
    for b, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return b
    return 0


def ints(field):
    return [int(x) for x in field.split(";")] if field else []


def read(path):
    """Per (tag, entry, request): positions -> emitted token, and the proposals in step order.

    A request of an entry starts again at its own positions, so a line whose pos0 is below the previous line's next_pos0 (a step
    continues where the one before chose its next rows), or whose step is lower, begins a new request: its tokens are kept apart.
    """
    streams = collections.defaultdict(dict)
    props = collections.defaultdict(list)
    seg = collections.defaultdict(int)
    last = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            ent = (r["tag"], r["entry"])
            pos0, step = int(r["pos0"]), int(r["step"])
            if ent in last and (pos0 < last[ent][0] or step < last[ent][1]):
                seg[ent] += 1
            last[ent] = (int(r["next_pos0"]), step)
            key = ent + (seg[ent],)
            for j, t in enumerate(ints(r["emitted"])):
                streams[key][pos0 + j + 1] = t
            m = int(r["m"])
            if m > 0:
                props[key].append((int(r["next_pos0"]), ints(r["chain"]), int(r["match_len"]), ints(r["copy"])))
    return streams, props


def score(streams, props):
    """right[b][j]: proposals of bucket b whose first j + 1 copy tokens were right; total[b][j]: those scored to depth j."""
    right = [[0] * ROWS for _ in BUCKETS]
    total = [[0] * ROWS for _ in BUCKETS]
    for key, plist in props.items():
        st = streams[key]
        for next_pos0, chain, length, copy in plist:
            k = len(chain)
            if any(st.get(next_pos0 + 1 + i) != t for i, t in enumerate(chain)):
                continue  # the chain's drafts did not all land: T is defined given that they did
            b = bucket_of(length)
            alive = True
            for j, t in enumerate(copy[:ROWS]):
                got = st.get(next_pos0 + k + 1 + j)
                if got is None:
                    break
                total[b][j] += 1
                alive = alive and got == t
                if alive:
                    right[b][j] += 1
    return right, total


def fit(right, total, min_n):
    table = []
    for b in range(len(BUCKETS)):
        prior, p = [], PRIOR_Q[b]
        for _ in range(ROWS):
            prior.append(p)
            p *= PRIOR_C[b]
        if total[b][0] < min_n:
            table.append(prior)
            continue
        row, prev = [], 1.0
        for j in range(ROWS):
            v = right[b][j] / total[b][j] if total[b][j] >= max(1, min_n // 4) else None
            if v is None:  # too few reach this depth: continue the last value by the prior's per-token ratio
                v = prev * PRIOR_C[b]
            v = min(v, prev)  # non-increasing along the rows
            row.append(max(v, 0.001))
            prev = v
        table.append(row)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="+", help="probe CSV files")
    ap.add_argument("-o", "--out", default="-", help="the table file (default: stdout)")
    ap.add_argument("--min-n", type=int, default=200, help="scored proposals a bucket needs before it leaves the prior")
    a = ap.parse_args()
    right = [[0] * ROWS for _ in BUCKETS]
    total = [[0] * ROWS for _ in BUCKETS]
    for path in a.dump:
        r, t = score(*read(path))
        for b in range(len(BUCKETS)):
            for j in range(ROWS):
                right[b][j] += r[b][j]
                total[b][j] += t[b][j]
    table = fit(right, total, a.min_n)
    for b, (lo, hi) in enumerate(BUCKETS):
        print("bucket %d (match %d-%s): %d proposals scored; first token right %s" % (
            b, lo, "" if hi > 1000 else hi - 1, total[b][0],
            "%.3f" % (right[b][0] / total[b][0]) if total[b][0] else "-"), file=sys.stderr)
    text = "".join(" ".join("%.4f" % v for v in row) + "\n" for row in table)
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
