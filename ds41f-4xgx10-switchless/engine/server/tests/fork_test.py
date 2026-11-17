#!/usr/bin/env python3
"""Forks by reference (docs/engine.md, "The chunk pool"): a fork takes its base's chunk numbers wholly below its restart
point by count and copies only the one holding that point; nothing writes a number another list still names. Run R and
run T (A rolled back and freed before the forks' next turns) must give bit-equal next turns for B and C; with --log the
forks must show numbers shared.

  python3 engine/server/tests/fork_test.py --socket S --tokens T.npy [--log engine.log]     exit 0 when every check passes
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402
from slice_test import Mux  # noqa: E402

S0 = 0x666f726b00000000  # 'fork'
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def turn(e, name, sid, prompt, **kw):
    o = e.run(sid, prompt, max_new=24, n_top=2, **kw)
    show(name, o)
    check(not o['error'], '%s: no error' % name)
    return o


def scenario(e, T, a, run, rollback):
    """One run's requests in sessions of its own; returns the forks' next turns."""
    base = S0 + (run << 16)
    A, B, C = base + 1, base + 2, base + 3
    turn(e, '%s A cold %d' % (run, a.n), A, T[:a.n])
    fb = turn(e, '%s B fork at %d' % (run, a.q_aligned), B, T[:a.q_aligned] + T[a.n + 1000:a.n + 1300], base=A)
    fc = turn(e, '%s C fork at %d' % (run, a.q_unaligned), C, T[:a.q_unaligned] + T[a.n + 2000:a.n + 2300], base=A)
    if rollback:
        turn(e, '%s A back to %d' % (run, a.cut), A, T[:a.cut] + T[a.n + 3000:a.n + 3300])
        e.release(A)
    hb = T[:a.q_aligned] + T[a.n + 1000:a.n + 1300] + list(fb['ids'])
    hc = T[:a.q_unaligned] + T[a.n + 2000:a.n + 2300] + list(fc['ids'])
    nb = turn(e, '%s B next' % run, B, hb + T[a.n + 4000:a.n + 4040])
    nc = turn(e, '%s C next' % run, C, hc + T[a.n + 5000:a.n + 5040])
    for s in (A, B, C):
        e.release(s)
    return {'B fork': fb, 'C fork': fc, 'B next': nb, 'C next': nc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', type=int, default=60000, help="base A's first prompt")
    ap.add_argument('--q-aligned', type=int, default=49152, help='fork B: a multiple of 16,384 (a chunk edge of every source)')
    ap.add_argument('--q-unaligned', type=int, default=41000, help='fork C: inside a chunk')
    ap.add_argument('--cut', type=int, default=20000, help="A's rollback: inside numbers B and C share")
    ap.add_argument('--forks', type=int, default=8)
    ap.add_argument('--log', default='', help="rank 0's log (engine-test.sh keeps it): the forks' shared numbers")
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    assert len(T) > a.n + 6000
    e = Engine(a.socket)
    print('run R: the forks with their base untouched')
    r = scenario(e, T, a, 1, False)
    print('run T: the base goes back inside the shared numbers and is freed before the forks go on')
    t = scenario(e, T, a, 2, True)
    for k in ('B fork', 'C fork', 'B next', 'C next'):
        x, y = r[k], t[k]
        ok = x['ids'] == y['ids'] and x['lp'] == y['lp'] and x['cached'] == y['cached']
        d = next((i for i, (p, q) in enumerate(zip(x['ids'], y['ids'])) if p != q), None)
        check(ok, '%s: run T equals run R (%d tokens, reused %d vs %d)%s' % (
            k, len(x['ids']), x['cached'], y['cached'], '' if ok else ': first token difference at %s' % d))
    check(0 < r['B fork']['cached'] <= a.q_aligned and 0 < r['C fork']['cached'] <= a.q_unaligned,
          'the forks restarted inside their base (reused %d and %d)' % (r['B fork']['cached'], r['C fork']['cached']))

    print('%d forks of one base together' % a.forks)
    m = Mux(a.socket)
    base = S0 + (3 << 16)
    m.run(base, T[:30000], max_new=8)
    rids = [m.start('fork %d' % i, base + 1 + i, T[:30000] + T[a.n + 100 * i:a.n + 100 * i + 200], base=base, max_new=48)
            for i in range(a.forks)]
    m.pump(lambda: all(m.recs[x]['done'] for x in rids))
    for x in rids:
        rec = m.recs[x]
        check(rec['done']['reason'] == 'length' and not rec['done']['error'] and len(rec['ids']) == 48,
              '%s: %d tokens, %s, reused %d%s' % (rec['name'], len(rec['ids']), rec['done']['reason'], rec['done']['cached'],
                                                  ', ' + rec['done']['error'] if rec['done']['error'] else ''))
    for i in range(a.forks + 1):
        m.release(base + i)
    after = m.run(base + 100, T[1000:1500], max_new=4)
    check(not after['error'] and len(after['ids']) == 4, 'a request after them is answered')

    if a.log:
        time.sleep(1)
        lines = [l for l in open(a.log, errors='replace') if re.search(r'forks \d+ at \d+: \d+ numbers shared', l)]
        shared = [int(re.search(r'(\d+) numbers shared', l).group(1)) for l in lines]
        check(len(lines) >= 4 + a.forks and sum(shared) > 0,
              "rank 0's log: %d forks, %d numbers shared in all (none would mean the sharing never ran)" % (len(lines), sum(shared)))
    print('fork_test: %s' % ('%d failed' % fails if fails else 'all checks pass'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
