#!/usr/bin/env python3
"""Interleaved prefill, the second reader: a session decodes (A); a long cold prompt (B) is being read; a shorter long prompt
(C, at most --interleave-push-rows rows) arrives and reads beside B. Checks that C's first token comes before B's, that B's
and C's first tokens (with logprobs) equal each read alone, and that A never waits longer than --max-gap-s.

  push_test.py --socket S --tokens T.npy [--b-len 300000] [--c-len 60000] [--max-gap-s 0.8]
"""
import argparse
import struct
import sys
import time

from serve_test import CANCEL, DONE, PROGRESS, TOKENS, Engine, load_npy_i32


def first(o):
    return (o['ids'][:1], o['lp'][:1], o['top'][:1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--b-len', type=int, default=300000)
    ap.add_argument('--c-len', type=int, default=60000)
    ap.add_argument('--max-gap-s', type=float, default=0.8)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    S = 0x7075736800000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    pb, pc = T[0:a.b_len], T[a.b_len:a.b_len + a.c_len]
    b1 = e.run(S + 10, pb, max_new=4, n_top=2, keep=False)
    c1 = e.run(S + 11, pc, max_new=4, n_top=2, keep=False)
    ra = e.submit(S + 1, T[350000:352000], max_new=40000, keep=False, stop=())[0]
    rb = rc = None
    got = {}
    t0 = {}
    tf = {}
    last_a, gap = None, 0.0
    done = set()
    while True:
        t, p = e.frame()
        now = time.time()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            if r == ra:
                if last_a is not None and rb is not None and (rb not in tf or rc is None or rc not in tf):
                    gap = max(gap, now - last_a)
                last_a = now
                if rb is None:
                    rb = e.submit(S + 2, pb, max_new=4, n_top=2, keep=False)[0]
                    t0[rb] = now
                continue
            o = got.setdefault(r, {'ids': [], 'lp': [], 'top': []})
            o['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
            off = 16 + 4 * n
            if fl & 1:
                o['lp'] += struct.unpack('<%df' % n, p[off:off + 4 * n])
                off += 4 * n
            if fl & 2:
                pairs = struct.unpack('<' + 'If' * (n * 2), p[off:off + 16 * n])
                for i in range(n):
                    o['top'].append(list(zip(pairs[4 * i::2][:2], pairs[4 * i + 1::2][:2])))
            if r not in tf:
                tf[r] = now
        elif t == PROGRESS:
            r, rows, total = struct.unpack('<QII', p[:16])
            if r == rb and rc is None and rows >= 0.25 * total:
                rc = e.submit(S + 3, pc, max_new=4, n_top=2, keep=False)[0]
                t0[rc] = now
        elif t == DONE:
            r = struct.unpack('<Q', p[:8])[0]
            done.add(r)
            if rb in done and rc in done and ra not in done:
                e.send(CANCEL, struct.pack('<Q', ra))
            if ra in done:
                break
    tb, tc = tf[rb] - t0[rb], tf[rc] - t0[rc]
    print('B (%d rows) first token after %.1f s (alone %.1f s); C (%d rows, arrived at 25 %% of B) after %.1f s '
          '(alone %.1f s), %.1f s before B\'s; A\'s largest gap %.2f s' % (
              len(pb), tb, b1['prefill_us'] / 1e6, len(pc), tc, c1['prefill_us'] / 1e6, tf[rb] - tf[rc], gap))
    res = {'C\'s first token before B\'s (it read beside B)': tf[rc] < tf[rb],
           'B\'s first token equals B alone': first(got[rb]) == first(b1),
           'C\'s first token equals C alone': first(got[rc]) == first(c1),
           'A never waits more than %.1f s' % a.max_gap_s: gap <= a.max_gap_s}
    for name, ok in res.items():
        print('%s: %s' % ('ok  ' if ok else 'FAIL', name))
    print('push_test: %d of %d checks pass' % (sum(res.values()), len(res)))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
