#!/usr/bin/env python3
"""Greedy outputs bit-equal with the gate's flags on and off: compare two BENCH_DUMP files (bench_seeded.py's request
index, prompt start and text hash per line) from runs with BENCH_TEMP=0. Run one arm with the flags off, the other on, same
sessions and BENCH_OFFSET; the verified tokens cannot depend on which drafts were verified, so every hash must match.
Usage: compare_dumps.py OFF.shas ON.shas   (exit 1 on any difference)"""
import json, sys


def load(p):
    d = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            d[r["prompt"]] = (r["tokens"], r["sha"])  # distinct mode gives every request idx 0: the prompt is the key
    return d


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    bad = [i for i in sorted(set(a) | set(b)) if a.get(i) != b.get(i)]
    print("%d requests, %d differ%s" % (len(set(a) | set(b)), len(bad), (": " + "; ".join(bad)) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
