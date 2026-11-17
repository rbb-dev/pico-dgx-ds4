#!/usr/bin/env python3
"""Interleaved prefill under load, on one connection to rank 0's socket: three sessions decode long responses; a long cold
prompt (B) arrives; while it is read a short prompt arrives every second. Checks that B finishes (admissions and B's
pieces take turns, so a stream of short prompts cannot starve it), that no decoding session waits longer than
--max-gap-s, and that every short prompt gets its first token within --max-short-s. Reports each decoder's tokens per
second over B's read and B's time against B alone.

  stream_test.py --socket S --tokens T.npy [--decoders 3] [--shorts 15] [--max-gap-s 1.0] [--max-short-s 2.5]
"""
import argparse
import struct
import sys
import time

from serve_test import CANCEL, DONE, PROGRESS, REASONS, TOKENS, Engine, load_npy_i32, show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--decoders', type=int, default=3)
    ap.add_argument('--shorts', type=int, default=15)
    ap.add_argument('--b-len', type=int, default=200000)
    ap.add_argument('--max-gap-s', type=float, default=1.0)
    ap.add_argument('--max-short-s', type=float, default=2.5)
    ap.add_argument('--max-b-ratio', type=float, default=3.5)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    S = 0x7374726d00000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    pb = T[100000:100000 + a.b_len]
    b_alone = e.run(S + 99, pb, max_new=8, keep=False)
    show('B alone', b_alone)
    dec = [e.submit(S + 1 + i, T[i * 3000:i * 3000 + 2000], max_new=40000, keep=False, stop=())[0] for i in range(a.decoders)]
    last = {r: None for r in dec}
    gaps = {r: 0.0 for r in dec}
    count = {r: 0 for r in dec}
    started = set()
    rb = None
    t_b = t_b_first = None
    shorts = {}  # request -> [submit time, first token time]
    next_short = None
    n_sent = 0
    done = {}
    stopping = False
    while True:
        t, p = e.frame()
        now = time.time()
        if rb is not None and t_b_first is None and next_short is not None and now >= next_short and n_sent < a.shorts:
            r = e.submit(S + 200 + n_sent, T[200000 + n_sent * 5000:200000 + n_sent * 5000 + 4000], max_new=16, keep=False)[0]
            shorts[r] = [now, None]
            n_sent += 1
            next_short = now + 1.0
        if t == TOKENS:
            r, n = struct.unpack('<QI', p[:12])
            if r in last:
                if last[r] is not None and rb is not None and t_b_first is None:
                    gaps[r] = max(gaps[r], now - last[r])
                if rb is not None and t_b_first is None:
                    count[r] += n
                last[r] = now
                started.add(r)
                if rb is None and len(started) == len(dec):
                    rb = e.submit(S + 50, pb, max_new=8, keep=False)[0]
                    t_b = now
                    next_short = now + 1.0
            elif r == rb and t_b_first is None:
                t_b_first = now
            elif r in shorts and shorts[r][1] is None:
                shorts[r][1] = now
        elif t == DONE:
            r, reason = struct.unpack('<QI', p[:12])
            done[r] = REASONS[reason]
            if t_b_first is not None and not stopping and all(s in done for s in shorts) and rb in done:
                stopping = True
                for d in dec:
                    e.send(CANCEL, struct.pack('<Q', d))
            if stopping and all(d in done for d in dec):
                break
        elif t != PROGRESS:
            raise RuntimeError('frame %#x' % t)
    b_s = t_b_first - t_b
    tb_alone = b_alone['prefill_us'] / 1e6
    rates = [count[r] / max(1e-9, b_s) for r in dec]
    short_s = [s[1] - s[0] for s in shorts.values() if s[1] is not None]
    print('B: first token %.1f s together, %.1f s alone (x%.2f); %d short prompts sent during it' % (b_s, tb_alone, b_s / tb_alone, n_sent))
    print('decoders during B: %s tok/s; largest gaps %s s' % (', '.join('%.1f' % x for x in rates),
                                                               ', '.join('%.2f' % gaps[r] for r in dec)))
    print('short prompts, first token after: %s s' % ', '.join('%.2f' % x for x in short_s))
    res = {'B finishes (no starving behind the stream) within x%.1f of alone' % a.max_b_ratio: done.get(rb) == 'length' and b_s <= a.max_b_ratio * tb_alone,
           'no decoder waits more than %.1f s' % a.max_gap_s: max(gaps.values()) <= a.max_gap_s,
           'every short prompt answers within %.1f s' % a.max_short_s: len(short_s) == n_sent and max(short_s) <= a.max_short_s}
    for name, ok in res.items():
        print('%s: %s' % ('ok  ' if ok else 'FAIL', name))
    print('stream_test: %d of %d checks pass' % (sum(res.values()), len(res)))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
