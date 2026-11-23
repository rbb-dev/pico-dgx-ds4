#!/usr/bin/env python3
"""Host checks of the store's trace files (model/store_trace.h).

  store_trace_check.py same RANKFILES...    the four ranks' files of one run are equal after the `geom rank=` field
  store_trace_check.py coverage FILES...    a table of entry and answer kinds with their counts over the files, and the
                                            kinds with none (exit 1 when there are any)

`same` leaves out what differs by nature: the rank, each rank's own `base=` path and `run_id=` (a hash of that rank's
binary and its own weight files), the clock's value in `A now=`, and the
`V` lines (rank 0's server takes the evicted list at its own moments). A rank's own rolled-back attempt is left out too: inside one
command a rank that tried to grow the pool (`alloc_grow`) and went back (`alloc_shrink`) because another rank could not, and whose
digest after the shrink is the digest before the grow, made a record pair the others do not have. It is that rank's own, and the
record numbers after it are compared by order, not by value. A digest that differs after the command still fails. Everything else
must be equal, line by line.
Files named *.gz are read through gzip. The kinds come from the table in store_trace.h, so a kind added there is a kind
checked here.
"""
import gzip
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "..", "model", "store_trace.h")


def load_kinds():
    text = open(HEADER, encoding="utf-8").read()
    kinds = re.findall(r'^\s*X\((\w+), "([^"]*)", "([^"]*)"\)', text, re.M)
    ans_line = re.search(r"#define STORE_TRACE_ANSWERS\(X\)(.*)", text)
    answers = re.findall(r"X\((\w+)\)", ans_line.group(1)) if ans_line else []
    if not kinds or not answers:
        sys.exit("store_trace_check: cannot read the kinds from " + HEADER)
    return [k[0] for k in kinds], answers


def read_lines(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        text = f.read()
    if not text.endswith("\n"):
        sys.exit("%s: truncated (the last line is not complete)" % path)
    lines = text.split("\n")[:-1]
    if not lines or lines[0] != "pico-store-trace 1":
        sys.exit("%s: not a version 1 store trace" % path)
    return lines


def canon(line):
    """The line with the parts a comparison across ranks leaves out replaced; None for a line left out whole."""
    if line.startswith("V "):
        return None
    if line.startswith("geom "):
        line = re.sub(r" ?\brank=\d+", "", line)
    elif line.startswith("E "):
        line = re.sub(r" base=\S+", " base=*", line)
        line = re.sub(r" run_id=\S+", " run_id=*", line)
    elif line.startswith("A ") and re.match(r"A \d+ now=", line):
        line = re.sub(r"now=\S+", "now=*", line)
    return line


GROW_RE = re.compile(r"E \d+ alloc_grow ")
SHRINK_RE = re.compile(r"E \d+ alloc_shrink ")
D_RE = re.compile(r"D \d+ d=(\w+)")


def drop_rolled_back(lines):
    """The lines without the rank's rolled-back attempts (E grow, D, E shrink, D whose digest is the digest before the grow), the
    E, D, A and R record numbers renumbered by order; (lines, attempts dropped)."""
    out, i, last_d, dropped = [], 0, None, 0
    while i < len(lines):
        if i + 3 < len(lines) and GROW_RE.match(lines[i]) and lines[i + 1].startswith("D ") and SHRINK_RE.match(lines[i + 2]):
            back = D_RE.match(lines[i + 3])
            if back and last_d is not None and back.group(1) == last_d:
                i += 4
                dropped += 1
                continue
        m = D_RE.match(lines[i])
        if m:
            last_d = m.group(1)
        out.append(lines[i])
        i += 1
    # renumber by order: an E line takes the next number, the D and A lines of that entry follow it
    new, mapping = [], {}
    for l in out:
        m = re.match(r"([EDAR]) (\d+)(.*)$", l, re.S)
        if not m:
            new.append(l)
        elif m.group(1) == "E":
            mapping[m.group(2)] = len(mapping) + 1
            new.append("E %d%s" % (mapping[m.group(2)], m.group(3)))
        else:
            new.append("%s %s%s" % (m.group(1), mapping.get(m.group(2), m.group(2)), m.group(3)))
    return new, dropped


def cmd_same(paths):
    if len(paths) < 2:
        sys.exit("same: give at least two files")
    docs, own = [], []
    for p in paths:
        d, n = drop_rolled_back([c for c in map(canon, read_lines(p)) if c is not None])
        docs.append(d)
        own.append(n)
    for p, n in zip(paths, own):
        if n:
            print("note: %s: %d rolled-back attempt(s) of its own left out" % (p, n))
    ref = docs[0]
    bad = 0
    for p, d in zip(paths[1:], docs[1:]):
        for i in range(min(len(ref), len(d))):
            if ref[i] != d[i]:
                print("DIFFERENT %s vs %s at line %d:\n  %s\n  %s" % (paths[0], p, i + 1, ref[i], d[i]))
                bad += 1
                break
        else:
            if len(ref) != len(d):
                print("DIFFERENT %s vs %s: %d lines against %d (equal up to the shorter)" % (paths[0], p, len(ref), len(d)))
                bad += 1
    if bad:
        return 1
    print("SAME: %d files, %d lines each (rank, base path, run id, clock values and take_evicted lines left out)" % (len(paths), len(ref)))
    return 0


def cmd_coverage(paths):
    kinds, answers = load_kinds()
    ek, ak = Counter(), Counter()
    stray = evicted = records = 0
    for p in paths:
        for line in read_lines(p):
            t = line.split(" ")
            if t[0] == "E" and len(t) >= 3:
                ek[t[2]] += 1
                records += 1
            elif t[0] == "A" and len(t) >= 3:
                if t[1] == "0":
                    stray += 1
                ak[t[2].split("=", 1)[0]] += 1
            elif t[0] == "V":
                evicted += 1
    print("%d files, %d entries, %d take_evicted lines" % (len(paths), records, evicted))
    print("%-16s %8s" % ("entry kind", "records"))
    for k in kinds:
        print("%-16s %8d%s" % (k, ek[k], "" if ek[k] else "   <-- none"))
    print("%-16s %8s" % ("answer kind", "lines"))
    for a in answers:
        print("%-16s %8d%s" % (a, ak[a], "" if ak[a] else "   <-- none"))
    unknown = sorted(set(ek) - set(kinds)) + sorted(set(ak) - set(answers))
    missing = [k for k in kinds if not ek[k]] + [a for a in answers if not ak[a]]
    if unknown:
        print("kinds in the files that the header does not name:", ", ".join(unknown))
    if stray:
        print("%d answers taken outside any entry (A 0 lines): a replay refuses them" % stray)
    print("kinds with none:", ", ".join(missing) if missing else "(none)")
    return 1 if missing or unknown or stray else 0


def main(argv):
    if len(argv) < 3 or argv[1] not in ("same", "coverage"):
        sys.exit(__doc__)
    return cmd_same(argv[2:]) if argv[1] == "same" else cmd_coverage(argv[2:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
