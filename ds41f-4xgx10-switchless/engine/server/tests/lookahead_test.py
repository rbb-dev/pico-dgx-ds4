#!/usr/bin/env python3
"""Engram look-ahead after a cancel: a prompt cancelled between chunks leaves its next chunk's lookups behind
(prefill_lookup); they must never be taken by another conversation's chunk at the same position and size, nor by a later
prompt of the same slot. A session continued from row P must equal the same prompt read cold, and the cancelled prompt
again keeps its P rows. Through rank 0's socket (dsfront stopped).

  lookahead_test.py --socket S --tokens T.npy        exit 0 when every check passes
"""
import argparse
import struct
import sys
import time

from serve_test import CANCEL, DONE, PROGRESS, REASONS, TOKENS, Engine, load_npy_i32, show


def collect_cancel(e, req, cancel_at):
    """collect, sending CANCEL once a Progress frame reports at least cancel_at rows; returns (DONE fields, last rows)"""
    rid, _ = req
    last, sent = 0, False
    while True:
        t, p = e.frame()
        if t == PROGRESS:
            r, done, _total = struct.unpack('<QII', p[:16])
            if r == rid:
                last = done
                if not sent and done >= cancel_at:
                    e.send(CANCEL, struct.pack('<Q', rid))
                    sent = True
        elif t == DONE:
            (_r, reason, _pt, cached, _comp, _q, _p, _d, _el, _) = struct.unpack('<QIIIIQQQII', p[:56])
            return {'reason': REASONS[reason], 'cached': cached}, last
        elif t != TOKENS:
            raise RuntimeError('frame type %#x' % t)


def same(name, a, b):
    eq = a['ids'] == b['ids'] and a['lp'] == b['lp'] and a['top'] == b['top']
    n = 0
    while n < min(len(a['ids']), len(b['ids'])) and a['ids'][n] == b['ids'][n]:
        n += 1
    print('%s: %s (tokens equal %d/%d)' % (name, 'BIT-EQUAL' if eq else 'DIFFER', n, max(len(a['ids']), len(b['ids']))))
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--cancel-at', type=int, default=16384)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    if len(T) < 200000:
        sys.exit('lookahead_test: the token file holds %d tokens, 200,000 needed' % len(T))
    e = Engine(a.socket)
    S = 0x6c6f6f6b00000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    tb, tx = T[0:120000], T[150000:150000 + 49152]
    for k in range(1, 7):
        e.run(S + k, tb[0:8192 * k], max_new=1)
    x, last = collect_cancel(e, e.submit(S + 100, tx, max_new=8), a.cancel_at)
    print('X: %s after %d of %d rows' % (x['reason'], last, len(tx)))
    res = {'X was cancelled after a chunk, with a chunk still to come':
           x['reason'] == 'cancelled' and last % 8192 == 0 and 8192 <= last <= 6 * 8192 and len(tx) - last >= 8192}
    if res['X was cancelled after a chunk, with a chunk still to come']:
        k = last // 8192
        p = tb[0:8192 * k] + tb[8192 * k:8192 * k + 12000]
        y = e.run(S + k, p, max_new=32, n_top=2)
        show('B%d continued from %d' % (k, last), y)
        c = e.run(S + 200, p, max_new=32, n_top=2, keep=False)
        show('  the same prompt cold', c)
        res['the continuation starts at the left-over lookups\' position'] = y['cached'] == last
        res['another conversation never takes a cancelled prompt\'s lookups'] = same('continued vs cold', y, c)
    x2 = e.run(S + 100, tx, max_new=32, n_top=2)
    show('X again on its session', x2)
    xc = e.run(S + 201, tx, max_new=32, n_top=2, keep=False)
    show('  X cold', xc)
    res['a cancelled prompt keeps the rows it computed'] = x2['cached'] == last
    res['and continues exactly'] = same('X again vs cold', x2, xc)
    for s in list(range(1, 7)) + [100]:
        e.release(S + s)
    for name, ok in res.items():
        print('%s: %s' % ('ok  ' if ok else 'FAIL', name))
    print('lookahead_test: %d of %d checks pass' % (sum(res.values()), len(res)))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
