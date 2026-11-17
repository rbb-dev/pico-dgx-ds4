#!/usr/bin/env python3
"""Interleaved prefill, a long prompt's times in its DONE: a session decodes (A); a long cold prompt (B) is read in pieces; a
second one (C) waits behind it (the second reader off: --interleave-push-rows 0, set live, put back at the end). C's wait is
queue time (DONE queue_us) and its prefill time is its own read only (prefill_us). B, read at once, has almost no queue time.

  queue_test.py --socket S --tokens T.npy [--b-len 100000] [--c-len 40000]
"""
import argparse
import struct
import sys
import time

from serve_test import CANCEL, DONE, PROGRESS, REASONS, TOKENS, Engine, load_npy_i32

CONFIG = 0x06


def config(e, **kw):
    order = ['on', 'whole_rows', 'push_rows', 'gap_ms', 'share', 'burst_max_ms']
    e.send(CONFIG, struct.pack('<II', 1, 6) + struct.pack('<6q', *[kw.get(k, -1) for k in order]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--b-len', type=int, default=100000)
    ap.add_argument('--c-len', type=int, default=40000)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    S = 0x7175657500000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    config(e, push_rows=0)
    ra = e.submit(S + 1, T[0:2000], max_new=20000, keep=False, stop=())[0]
    rb = rc = None
    times = {}
    while True:
        t, p = e.frame()
        if t == TOKENS:
            r = struct.unpack('<Q', p[:8])[0]
            if r == ra and rb is None:
                rb = e.submit(S + 2, T[100000:100000 + a.b_len], max_new=2, keep=False)[0]
                rc = e.submit(S + 3, T[310000:310000 + a.c_len], max_new=2, keep=False)[0]
        elif t == DONE:
            (r, reason, _, _, _, q_us, p_us, _) = struct.unpack('<QIIIIQQQ', p[:48])
            times[r] = (REASONS[reason], q_us / 1e6, p_us / 1e6)
            if r in (rb, rc) and rb in times and rc in times:
                e.send(CANCEL, struct.pack('<Q', ra))
            if r == ra:
                break
        elif t != PROGRESS:
            raise RuntimeError('frame %#x' % t)
    config(e, push_rows=131072)
    (wb, qb, pb), (wc, qc, pc) = times[rb], times[rc]
    print('B (%d tokens, read at once): %s, queue %.2f s, prefill %.2f s' % (a.b_len, wb, qb, pb))
    print('C (%d tokens, waits behind B): %s, queue %.2f s, prefill %.2f s' % (a.c_len, wc, qc, pc))
    res = {'both finish': wb == 'length' and wc == 'length',
           'B, read at once, has under 1 s of queue time': qb < 1.0,
           'C\'s wait behind B is queue time (at least 80 % of B\'s read)': qc >= 0.8 * pb,
           'C\'s prefill time is its own read only (shorter than its wait behind the longer B)': 0 < pc < qc}
    for name, ok in res.items():
        print('%s: %s' % ('ok  ' if ok else 'FAIL', name))
    print('queue_test: %d of %d checks pass' % (sum(res.values()), len(res)))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
