#!/usr/bin/env python3
"""Interleaved prefill and resumes: more sessions decode than there are slots (--slots 2), taking turns, with a small
store (--session-gib small, --disk-gib 0). Checks that every session's reply equals the same session decoding alone
(greedy). Through rank 0's socket. It aims at a resume that reads its whole context again in pieces (more rows than
--interleave-whole-rows); with the shared session store a paused session's state is kept and the resume continues
from it, so whether a run reached that branch shows only in the engine's log ("resumes in pieces").

  resume_test.py --socket S --tokens T.npy [--ctx 40000] [--new 400]      exit 0 when every response is bit-equal
"""
import argparse
import struct
import sys
import time

from serve_test import DONE, PROGRESS, REASONS, TOKENS, Engine, load_npy_i32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--ctx', type=int, default=40000)
    ap.add_argument('--new', type=int, default=400)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    S = 0x7265736d00000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    prompts = [T[i * 50000:i * 50000 + a.ctx] for i in range(3)]
    alone = [e.run(S + 10 + i, p, max_new=a.new, n_top=2, keep=False, stop=()) for i, p in enumerate(prompts)]
    reqs = {e.submit(S + 1 + i, p, max_new=a.new, n_top=2, keep=False, stop=())[0]: i for i, p in enumerate(prompts)}
    out = {i: {'ids': [], 'lp': [], 'top': []} for i in range(3)}
    done = set()
    while len(done) < 3:
        t, p = e.frame()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            o = out[reqs[r]]
            o['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
            off = 16 + 4 * n
            if fl & 1:
                o['lp'] += struct.unpack('<%df' % n, p[off:off + 4 * n])
                off += 4 * n
            if fl & 2:
                pairs = struct.unpack('<' + 'If' * (n * 2), p[off:off + 16 * n])
                for i in range(n):
                    o['top'].append(list(zip(pairs[4 * i::2][:2], pairs[4 * i + 1::2][:2])))
        elif t == DONE:
            r, reason = struct.unpack('<QI', p[:12])
            done.add(r)
            print('session %d: %s, %d tokens' % (reqs[r], REASONS[reason], len(out[reqs[r]]['ids'])))
        elif t != PROGRESS:
            raise RuntimeError('frame %#x' % t)
    ok = True
    for i in range(3):
        eq = out[i]['ids'] == alone[i]['ids'] and out[i]['lp'] == alone[i]['lp'] and out[i]['top'] == alone[i]['top']
        n = 0
        while n < min(len(out[i]['ids']), len(alone[i]['ids'])) and out[i]['ids'][n] == alone[i]['ids'][n]:
            n += 1
        print('session %d together vs alone: %s (tokens equal %d/%d)' % (i, 'BIT-EQUAL' if eq else 'DIFFER', n, len(alone[i]['ids'])))
        ok = ok and eq
    print('resume_test: %s' % ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
