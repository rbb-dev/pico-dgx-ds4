#!/usr/bin/env python3
"""A squeezed session store keeps every restored session's restart point (docs/engine.md, the store).

Long sessions reloaded into slots keep their stored pages in a small store; a session spilled and read
back into what is left must still resume at its end (or at its newest snapshot when the end does not
fit), never at 0 and never at an old snapshot. The same requests that left 6 agents of 698k positions
at `reused 0 (cold)` and `reused 196608 (snapshot)` with --session-gib 3.

Start the server so that two sessions of --n positions fit the store's budget (pool less 256 MiB or 1/16),
and a third read back does not fit beside them unless the live sessions' pages are released:
a session of n positions holds about n * 896 B * 1.14 (n = 300000: 290 MiB), so with --slots 4 and
--session-gib 0.85 the budget holds two. Pass/fail does not depend on hitting that window exactly: the
checks below hold at any size (a store that fits everything just never squeezes); rank 0's log shows
whether it did (`state: no room for`, `is live in a slot`, `restart point N of the M spilled`).

  dsgb10_rank --mode serve --slots 4 --session-gib 0.85 ...
  python3 engine/server/tests/squeeze_test.py [--socket S] [--tokens tokens.npy] [--n 300000] [--agents 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, compare, load_npy_i32, show  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', type=int, default=300000, help='positions of an agent\'s first turn')
    ap.add_argument('--agents', type=int, default=4)
    ap.add_argument('--chats', type=int, default=12, help='short chats between the turns')
    ap.add_argument('--tolerance', type=int, default=2048, help='positions a resume may fall short of the history')
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    TT = T + T  # sessions longer than the golden sequence: slices at different offsets share no prefix

    def seq(off, n):
        return TT[off:off + n]

    e = Engine(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    S = 0x5a9e000000000000
    ok = []
    hist, first = {}, {}
    for i in range(a.agents):  # the long first turns
        sid = S + 1 + i
        p = seq(7919 * (i + 1), a.n)
        o = e.run(sid, p, max_new=8, n_top=2)
        show('agent %d first turn' % i, o)
        hist[sid] = p + list(o['ids'])
        first[sid] = o
    for k in range(a.chats):  # short chats: they take slots, and the server spills the sessions they displace
        o = e.run(S + 0x1000 + k, seq(100003 * (k + 1) % len(T), 2000), max_new=2)
        e.release(S + 0x1000 + k)
    for i, sid in enumerate(sorted(hist)):  # the second turns
        p = hist[sid] + seq(12345 + 1000 * i, 40)
        o = e.run(sid, p, max_new=8, n_top=2)
        show('agent %d second turn' % i, o)
        floor = len(hist[sid]) - a.tolerance
        good = o['cached'] >= floor
        print('  reused %d of a history of %d: %s' % (o['cached'], len(hist[sid]),
              ('END' if o['cached'] >= len(hist[sid]) - 1 else 'NEWEST SNAPSHOT') if good else
              'FAIL: below %d (a restart point was lost)' % floor))
        ok.append(good)
        if i == 0:
            c = e.run(S + 0xfff, p, max_new=8, n_top=2, keep=False)
            ok.append(compare('agent 0 vs cold', o, c))
    for sid in hist:
        e.release(sid)
    print('squeeze_test: %d of %d checks pass' % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
