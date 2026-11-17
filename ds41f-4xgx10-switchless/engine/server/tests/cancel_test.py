#!/usr/bin/env python3
"""Interleaved prefill, cancels: a session (A) decodes throughout while a long prompt (B) is read in pieces. Checks that a
waiting second prompt (C) cancelled at once gets its DONE and counts its wait as queue time, that B cancelled at 40 % of
its read ends within a piece and a window while A goes on, and that B again on its own session keeps its rows and equals
B read alone. The second reader is off for the test (--interleave-push-rows 0, set live, put back at the end).

  cancel_test.py --socket S --tokens T.npy        exit 0 when every check passes
"""
import argparse
import struct
import sys
import time

from serve_test import CANCEL, DONE, PROGRESS, REASONS, TOKENS, Engine, load_npy_i32, show

CONFIG = 0x06


def config(e, **kw):
    order = ['on', 'whole_rows', 'push_rows', 'gap_ms', 'share', 'burst_max_ms']
    e.send(CONFIG, struct.pack('<II', 1, 6) + struct.pack('<6q', *[kw.get(k, -1) for k in order]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--b-len', type=int, default=200000)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    S = 0x63616e6300000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    pa, pb, pc = T[0:2000], T[100000:100000 + a.b_len], T[310000:310000 + 80000]
    b_alone = e.run(S + 9, pb, max_new=8, n_top=2, keep=False)
    show('B alone', b_alone)
    config(e, push_rows=0)  # C (80K rows) would otherwise read beside B as the second reader
    ra = e.submit(S + 1, pa, max_new=20000, keep=False, stop=())[0]
    rb = rc = None
    t_c_cancel = t_b_cancel = None
    c_at = 0.0
    done = {}
    times = {}
    last_a = None
    gaps = []
    b_rows = 0
    while True:
        t, p = e.frame()
        now = time.time()
        if rc is not None and t_c_cancel is None and now >= c_at:
            e.send(CANCEL, struct.pack('<Q', rc))
            t_c_cancel = now
        if t == TOKENS:
            r = struct.unpack('<Q', p[:8])[0]
            if r == ra:
                if last_a is not None and rb is not None and rb not in done:
                    gaps.append(now - last_a)
                last_a = now
                if rb is None:
                    rb = e.submit(S + 2, pb, max_new=8, keep=True)[0]
        elif t == PROGRESS:
            r, rows, total = struct.unpack('<QII', p[:16])
            if r == rb:
                b_rows = rows
                if rc is None and rows >= 8192:
                    rc = e.submit(S + 3, pc, max_new=8, keep=False)[0]
                    c_at = now + 1.0  # admitted and queued behind B by then; the cancel goes from the frame loop
                if t_b_cancel is None and rows >= 0.4 * total:
                    e.send(CANCEL, struct.pack('<Q', rb))
                    t_b_cancel = time.time()
        elif t == DONE:
            (r, reason, _, _, _, q_us, p_us, _) = struct.unpack('<QIIIIQQQ', p[:48])
            done[r] = (REASONS[reason], now)
            times[r] = (q_us / 1e6, p_us / 1e6)
            if r == rb:
                e.send(CANCEL, struct.pack('<Q', ra))  # A has done its part
            if r == ra:
                break
    res = {}
    c_ms = 1e3 * (done[rc][1] - t_c_cancel) if rc in done and t_c_cancel else -1
    b_ms = 1e3 * (done[rb][1] - t_b_cancel) if rb in done and t_b_cancel else -1
    print('C (queued) cancelled: %s, its DONE %.0f ms after the cancel' % (done.get(rc, ('none',))[0], c_ms))
    print('B cancelled at %d rows: %s, its DONE %.0f ms after the cancel; A\'s largest gap during B %.2f s' % (
        b_rows, done.get(rb, ('none',))[0], b_ms, max(gaps) if gaps else -1))
    res['a queued long prompt\'s cancel ends it within 1 s'] = done.get(rc, ('',))[0] == 'cancelled' and 0 <= c_ms <= 1000
    res['the reading prompt\'s cancel ends it within 2 s'] = done.get(rb, ('',))[0] == 'cancelled' and 0 <= b_ms <= 2000
    res['the decoder never waits more than 1 s meanwhile'] = bool(gaps) and max(gaps) <= 1.0
    qc, pc = times.get(rc, (-1, -1))
    print('C\'s DONE: queue %.2f s, prefill %.2f s' % (qc, pc))
    res['C\'s wait in the queue is queue time, not prefill (it was never read)'] = qc >= 0.8 and pc < 0.05
    config(e, push_rows=131072)
    b2 = e.run(S + 2, pb, max_new=8, n_top=2, keep=False)
    show('B again on its session', b2)
    res['B keeps the rows it read'] = b2['cached'] >= b_rows - 1024 and b2['cached'] > 0
    res['and its first token equals B alone'] = (b2['ids'][:1] == b_alone['ids'][:1] and b2['lp'][:1] == b_alone['lp'][:1]
                                                and b2['top'][:1] == b_alone['top'][:1])
    for name, ok in res.items():
        print('%s: %s' % ('ok  ' if ok else 'FAIL', name))
    print('cancel_test: %d of %d checks pass' % (sum(res.values()), len(res)))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
