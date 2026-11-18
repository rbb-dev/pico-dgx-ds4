#!/usr/bin/env python3
"""A chunk number's link to its disk image must be cut whenever the number's content changes (docs/engine.md, Serving:
"A copy goes stale only where the conversation writes again"). The number's image (Store::ndisk) is adopted by the next
holder that writes an image of the number, so a link left behind on a number that was written in place gives that holder
the old content: it reads back different from what it computed, with a valid checksum.

The sequence, through rank 0's socket (serve_test.py's client), on a pool so small that the store keeps nothing between
requests (every stored conversation is spilled when a request ends):
  1. A, a long conversation: its full pieces get images at the turn's save.
  2. B, a fork of A at a piece edge: B shares A's pieces wholly below the restart point q.
  3. Fillers in other sessions: A loses its line, then is spilled whole (its images stay, its numbers go back: B alone
     holds the shared numbers, B's line is untouched).
  4. B again, from a prompt that leaves A's text inside the last shared piece of the first three kv sources: B restarts
     inside that piece and writes it in place (the number is no longer shared).
  5. Fillers until B is spilled, then B's next turn (read back from the disk), compared bit for bit with the same prompt
     prefilled cold in a fresh session. The prompt's restart point is past the piece, so the read-back entries are used.
With --log the run proves each step from rank 0's log and the server's own counts (STATS: shared numbers, conversations on
disk, images written, read-backs); a step that did not happen ends the run with exit 2 ("did not reach the path"), never 0.
Exit 0: every step ran and the turn is bit-equal to cold. Exit 1: it ran and the turn differs (or a piece failed its
checksum).

Engine settings (engine-test.sh takes them as one string; the context is lowered to what the pool holds):
  MAXCTX=200000 engine-test.sh BIN "--slots 2 --big-slots 0 --pool-tokens 250000 --disk-gib 1" image_link_test.py \\
      --log ~/dsgb10/logs/engine-image_link_test.log [--n 60000 --q 49152]
- --slots 2: B keeps one line while the fillers cycle through the other and take A's.
- --pool-tokens 250000 (76 numbers, under 256 MiB): the store's budget between requests is zero, so every request's end
  spills the conversations no line holds.
- --disk-gib 1 (the smallest tier): images of A, B and the fillers fit with room to spare.

  python3 engine/server/tests/image_link_test.py --socket S --tokens T.npy --log engine.log [--n 60000] [--q 49152]
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402
from stats_test import stats  # noqa: E402

S0 = 0x696d676c696e6b00  # 'imglink'


def sid(k):
    return S0 + k


def turn(e, name, session, prompt, **kw):
    o = e.run(session, prompt, max_new=24, n_top=2, **kw)
    show(name, o)
    if o['error']:
        raise NotReached('%s: the engine answered with an error: %s' % (name, o['error']))
    return o


def filler(e, k, T, keep):
    """A short request in a session of its own: it takes a line, and its end spills what no line holds."""
    off = 100000 + 3000 * k
    return turn(e, 'filler %d' % k, sid(100 + k), T[off:off + 2000], keep=keep)


def run(a):
    T = [int(x) for x in load_npy_i32(a.tokens)]
    assert len(T) >= 110000 + a.n, 'the token file is too short'
    e = Engine(a.socket)
    log = Log(a.log)
    A, B = sid(1), sid(2)
    print('engine: %s, context %d' % (e.name, e.max_ctx))

    s = stats(e)[2]
    need(s['lines'] == 2, 'the engine runs with 2 lines (%d)' % s['lines'])
    need(s['disk_cap_bytes'] > 0, 'the engine has a disk tier (%.2f GiB)' % (s['disk_cap_bytes'] / 2**30))
    need(s['pool_free'] * 16384 // 5 < 300000, 'the pool is small (%d positions): the store keeps nothing between requests'
         % s['pool_positions'])
    spills0 = s['spills_total']

    print('1. A: %d positions' % a.n)
    turn(e, 'A cold %d' % a.n, A, T[:a.n])
    s = stats(e)[2]
    # full pieces below the saved length: three half-rate sources and one full-rate source (its pieces hold half as many positions)
    full = 3 * (a.n // PIECE) + a.n // (PIECE // 2)
    need(s['spills_total'] - spills0 >= full,
         "A's full pieces have images: %d images written (at least %d)" % (s['spills_total'] - spills0, full))

    print('2. B, a fork of A at %d' % a.q)
    p1 = T[:a.q] + T[a.n + 1000:a.n + 1300]
    b1 = turn(e, 'B fork at %d' % a.q, B, p1, base=A)
    m = log.wait_for(r'state: session \d+ forks \d+ at (\d+): (\d+) numbers shared', 'the fork')
    to, shared = int(m.group(1)), int(m.group(2))
    j = to // PIECE
    need(j >= 1 and shared > 0, 'B restarted at %d and shares %d numbers with A (%d whole pieces of sources 0-2)' % (to, shared, j))
    s = stats(e)[2]
    need(s['pool_shared'] > 0 and s['conv_sharing'] == 2, '%d numbers shared by %d conversations' % (s['pool_shared'], s['conv_sharing']))

    print('3. fillers until A is spilled')
    for k in range(1, 4):
        filler(e, k, T, True)
        if re.search(r'serve: spill session %016x ' % A, log.text()):
            break
    log.wait_for(r'serve: spill session %016x \(' % A, 'the spill of A')
    log.wait_for(r'state: session \d+ spilled to disk', 'the spill')
    s = stats(e)[2]
    need(s['pool_shared'] == 0, 'A is out of the pool while B holds its numbers: nothing is shared any more (%d)' % s['pool_shared'])
    need(s['conv_on_disk'] >= 1, '%d conversation(s) only on the disk tier' % s['conv_on_disk'])
    need(not re.search(r'serve: spill session %016x ' % B, log.text()), 'B was not spilled meanwhile')

    print('4. B again, rewound inside the last shared piece')
    pos = (j - 1) * PIECE + 8000  # inside piece j-1 of sources 0-2 (and inside source 3's pieces 2j-2 and 2j-1): all shared
    keep_to = j * PIECE + 1200  # long enough that the piece is complete and gets its image at the save
    p2 = T[:pos] + T[a.n + 2000:a.n + 2000 + keep_to - pos]
    b2 = turn(e, 'B rewound to %d' % pos, B, p2)
    need(b2['cached'] <= pos < to, 'B restarted at %d, inside the shared piece (common prefix %d, fork point %d)' % (b2['cached'], pos, to))

    print('5. fillers until B is spilled, then its next turn from the disk')
    restores0 = stats(e)[2]['restores_total']
    for k in range(4, 7):
        filler(e, k, T, True)
        if re.search(r'serve: spill session %016x ' % B, log.text()):
            break
    log.wait_for(r'serve: spill session %016x \(' % B, 'the spill of B')
    p3 = p2 + T[a.n + 9000:a.n + 9400]  # B's prompt again, and 400 new positions: the restart is at the prompt's end
    b3 = turn(e, 'B from disk', B, p3)
    log.wait_for(r'serve: session %016x back from disk' % B, 'the read-back of B')
    need(not re.search(r'state: session \d+: ERROR', log.text()), 'every piece read back matched its checksum')
    need(stats(e)[2]['restores_total'] > restores0, 'the server counted the read-back')
    need(b3['cached'] > j * PIECE, 'B restarted at %d, past the piece (%d): its entries came from the disk' % (b3['cached'], j * PIECE))
    need(not re.search(r'dropped from the disk tier|evicted to make room', log.text()), 'no conversation was dropped for room')

    print('verdict: the same prompt prefilled cold in a fresh session')
    c = turn(e, 'same prompt cold', sid(50), p3, keep=False)
    for k in range(1, 7):
        e.release(sid(100 + k))
    for x in (A, B):
        e.release(x)
    d = next((i for i, (p, q) in enumerate(zip(b3['ids'], c['ids'])) if p != q), None)
    ok = b3['ids'] == c['ids'] and b3['lp'] == c['lp'] and b3['top'] == c['top']
    print('  %s: B after the read-back vs cold: %d tokens%s' % (
        'EQUAL' if ok else 'DIFFER', len(b3['ids']), '' if ok else ', first token difference at %s (lp %s vs %s)' % (
            d, b3['lp'][:1], c['lp'][:1])))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', type=int, default=60000, help="A's prompt")
    ap.add_argument('--q', type=int, default=49152, help="B's fork point: a multiple of 16,384 (a piece edge of every source)")
    ap.add_argument('--log', required=True, help="rank 0's log from the engine's start (engine-test.sh keeps it)")
    a = ap.parse_args()
    try:
        rc = run(a)
    except NotReached as x:
        print('image_link_test: the test did not reach the path: %s' % x)
        return 2
    print('image_link_test: %s' % ('passed' if rc == 0 else 'FAILED'))
    return rc


if __name__ == '__main__':
    sys.exit(main())
