#!/usr/bin/env python3
"""A request whose own slot cannot get the memory it needs moves to a slot that can (docs/engine.md, "The chunk pool"); the
engine never stops for it. Session X holds a slot with a short turn's memory, three long requests commit most of the pool,
X's next turn asks for more than its slot and the free chunks can hold: it must run (state saved, read back in the slot it
gets, history reused) while the long requests go on and the engine stays up.

  dsgb10_rank --mode serve --slots 4 --big-slots 0 --pool-tokens 1808793 ...
  python3 engine/server/tests/room_test.py [--socket S] [--tokens tokens.npy] [--m 500000]
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import CANCEL, DONE, REASONS, TOKENS, Engine, load_npy_i32  # noqa: E402


def chunks_for(need, ctx):
    """Model::Impl::slot_chunks_for (model/model.cu): four kv sources (the first three at ratio 2), four arrays of
    256, 32, 64 and 4 bytes an entry, mapped to the request's last position + 4096 lookahead + 64 slack entries."""
    upto = min(need - 1 + 4096 + 64, ctx - 1)
    n = 0
    for k in range(4):
        e = upto // (2 if k < 3 else 1) + 1 + 64
        n += sum(-(-e * b // (2 << 20)) for b in (256, 32, 64, 4))
    return n


class Flights:
    """Several requests in flight on one connection: each frame goes to its request."""

    def __init__(self, e):
        self.e, self.out = e, {}

    def submit(self, session, prompt, max_new, stop=(1,)):
        rid, _ = self.e.submit(session, prompt, max_new=max_new, stop=stop)
        self.out[rid] = {'ids': [], 'done': None}
        return rid

    def cancel(self, rid):
        self.e.send(CANCEL, struct.pack('<Q', rid))

    def pump(self, until):
        while not until():
            t, p = self.e.frame()
            o = self.out[struct.unpack('<Q', p[:8])[0]]
            if t == TOKENS:
                n = struct.unpack('<QII', p[:16])[1]
                o['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
            elif t == DONE:
                (_, reason, pt, cached, comp, _q, _p, _d, el, _) = struct.unpack('<QIIIIQQQII', p[:56])
                o['done'] = dict(reason=REASONS[reason], prompt=pt, cached=cached, completion=comp,
                                 error=p[len(p) - el:].decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--m', type=int, default=500000, help='positions the long requests may generate')
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket)
    ctx = e.max_ctx
    cy, cxs, cxb = chunks_for(1000 + a.m, ctx), chunks_for(1008, ctx), chunks_for(1058 + a.m, ctx)
    print('engine: %s, context %d; chunks: long request %d, X short %d, X long %d; the pool must hold %d to %d chunks'
          % (e.name, ctx, cy, cxs, cxb, max(3 * cy + cxs, 2 * cy + cxb), 3 * cy + cxb - 1))
    F = Flights(e)
    S = 0x7007000000000000
    X = S + 1
    ok = []

    px = T[0:1000]  # X's short turn: it stays in its slot, holding a short turn's memory
    r = F.submit(X, px, 8)
    F.pump(lambda: F.out[r]['done'])
    hist = px + list(F.out[r]['ids'])
    print('X short turn: %s, %d tokens' % (F.out[r]['done']['reason'], len(F.out[r]['ids'])))

    ys = [F.submit(S + 2 + i, T[20000 * (i + 1):20000 * (i + 1) + 1000], a.m, stop=()) for i in range(3)]  # no stop id: they run until cancelled
    F.pump(lambda: all(len(F.out[y]['ids']) >= 32 for y in ys))  # the three decode together
    print('three long requests decoding')

    rx = F.submit(X, hist + T[90000:90050], a.m)  # X's long turn: more than its slot and the free chunks hold
    try:
        F.pump(lambda: len(F.out[rx]['ids']) >= 16 or F.out[rx]['done'])
        for q in [rx] + ys:
            if not F.out[q]['done']:
                F.cancel(q)
        F.pump(lambda: all(F.out[q]['done'] for q in [rx] + ys))
        r = F.submit(S + 9, T[150000:150300], 8)  # the engine still answers
        F.pump(lambda: F.out[r]['done'])
    except EOFError as err:
        print('FAIL: the engine stopped (%s): its log names the reason' % err)
        return 1

    dx = F.out[rx]['done']
    print('X long turn: %s, %d tokens, cached %d of %d (its history %d), error %r'
          % (dx['reason'], len(F.out[rx]['ids']), dx['cached'], dx['prompt'], len(hist), dx['error']))
    ok.append(len(F.out[rx]['ids']) >= 16 and not dx['error'])
    ok.append(dx['cached'] >= len(hist) - 128)  # its state came back from the store in the slot it got
    for y in ys:
        d = F.out[y]['done']
        print('long request %d: %s, %d tokens, error %r' % (y, d['reason'], len(F.out[y]['ids']), d['error']))
        ok.append(d['reason'] == 'cancelled' and not d['error'])
    d = F.out[r]['done']
    print('a new session after: %s, %d tokens' % (d['reason'], len(F.out[r]['ids'])))
    ok.append(not d['error'] and len(F.out[r]['ids']) > 0)
    print('room: %s (%d of %d checks)' % ('PASS' if all(ok) else 'FAIL', sum(ok), len(ok)))
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
