#!/usr/bin/env python3
"""Rows 6 and 7 of the one-session steps by source, from a DSGB10_STEP_LOG written with DSGB10_COPY_DUMP on.

Each line of the log is one step: i (index), R (rows of the step), n (tokens it emitted: its accepted drafts and one more), nt (the rows
the NEXT step verifies: bonus then drafts) and cp = [k, m]: of that next step's drafts the first k come from the draft pass (the chain,
or its block), the copy rows follow them. A step's draft j (1-based) is chain when j <= k of the step before, else copy; it is accepted
when j <= n - 1. Prints, for draft rows 6 and 7: how many steps offered one, from which source, and how many of them were accepted.
Also printed, as information: the steps a long copy step took over from a chain of 6 or more rows (the cursor was found after the
draft pass, which the block's rule cannot see).
usage: block_rows.py STEP_LOG
"""
import json
import sys


def analyse(lines):
    steps = [json.loads(l) for l in lines if l.strip()]
    offered = {(j, src): 0 for j in (6, 7) for src in ("chain", "copy")}
    accepted = dict(offered)
    block_with_long = 0
    for prev, cur in zip(steps, steps[1:]):
        if "cp" not in prev:
            raise SystemExit("the log has no 'cp' field: start with DSGB10_COPY_DUMP set")
        k = prev["cp"][0]
        rows = cur["R"]
        if rows > 8:
            if k >= 6:
                block_with_long += 1
            continue
        for j in (6, 7):
            if rows - 1 < j:
                continue
            src = "chain" if j <= k else "copy"
            offered[(j, src)] += 1
            if j <= cur["n"] - 1:
                accepted[(j, src)] += 1
    return offered, accepted, block_with_long


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    with open(sys.argv[1]) as f:
        offered, accepted, bwl = analyse(f)
    for j in (6, 7):
        for src in ("chain", "copy"):
            o, a = offered[(j, src)], accepted[(j, src)]
            print("draft row %d from the %-5s: offered %6d, accepted %6d (%s)" % (j, src, o, a, "%.1f %%" % (100.0 * a / o) if o else "-"))
    print("long copy steps that replaced a chain of 6 or more rows: %d" % bwl)


if __name__ == "__main__":
    main()
