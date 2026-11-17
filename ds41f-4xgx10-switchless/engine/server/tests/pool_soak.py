#!/usr/bin/env python3
"""The chunk pool soaked (docs/engine.md, "The chunk pool"): conversations made, forked and revisited until the positions read
add up to --passes times the pool, on a small pool with the disk tier and DSGB10_VMM_POISON=1 (a number holds 0xFF when taken
and 0xFD when given back). A revisit that restarts at the reference's position must equal the reference bit for bit.
Through rank 0's socket.

  python3 engine/server/tests/pool_soak.py --socket S --tokens T.npy --pool-positions P [--passes 3] [--log engine.log]
  exit 0 when every comparable revisit is equal, at least --min-checks of them, and no request ended in an error
"""
import argparse
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32  # noqa: E402

S0 = 0x736f616b00000000  # 'soak'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--pool-positions', type=int, required=True, help="positions the pool holds (its start line)")
    ap.add_argument('--passes', type=float, default=3)
    ap.add_argument('--min-len', type=int, default=4000)
    ap.add_argument('--max-len', type=int, default=80000)
    ap.add_argument('--min-checks', type=int, default=40)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--log', default='', help="rank 0's log: spills, read-backs and drops counted")
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    rng = random.Random(a.seed)
    e = Engine(a.socket)
    conv = {}  # sid -> {'prompt': [...], 'ref': (ids, lp, cached)}
    read = 0   # positions read (prefilled), all requests
    stats = {'made': 0, 'forks': 0, 'revisits': 0, 'equal': 0, 'differ': 0, 'cold': 0, 'errors': 0, 'freed': 0}
    differ = []
    next_sid = [S0 + 1]

    def ask(sid, prompt, **kw):
        nonlocal read
        o = e.run(sid, prompt, max_new=12, n_top=1, **kw)
        read += len(prompt) - o['cached']
        if o['error']:
            stats['errors'] += 1
            print('  ERROR %x: %s' % (sid, o['error']))
        return o

    def make(prompt, base=0):
        sid = next_sid[0]
        next_sid[0] += 1
        ask(sid, prompt, base=base) if base else ask(sid, prompt)
        r = ask(sid, prompt)
        conv[sid] = {'prompt': prompt, 'ref': (r['ids'], r['lp'], r['cached'])}
        return sid

    t0 = time.time()
    target = a.passes * a.pool_positions
    while read < target:
        k = rng.random()
        if not conv or k < 0.35:  # a new conversation
            n = rng.randint(a.min_len, a.max_len)
            off = rng.randint(0, len(T) - n - 1)
            make(T[off:off + n])
            stats['made'] += 1
        elif k < 0.55:  # a fork of one: part of it by reference, then its own tokens
            b = rng.choice(list(conv))
            p = conv[b]['prompt']
            cut = rng.randint(min(1024, len(p) - 1), len(p) - 1)
            off = rng.randint(0, len(T) - 3001)
            make(p[:cut] + T[off:off + rng.randint(200, 3000)], base=b)
            stats['forks'] += 1
        elif k < 0.6 and len(conv) > 4:  # a base freed while its forks may live
            s = rng.choice(list(conv))
            e.release(s)
            del conv[s]
            stats['freed'] += 1
        else:  # a revisit: the reference's prompt again
            s = rng.choice(list(conv))
            ids, lp, cached = conv[s]['ref']
            o = ask(s, conv[s]['prompt'])
            stats['revisits'] += 1
            if o['cached'] != cached:
                stats['cold'] += 1
            elif o['ids'] == ids and o['lp'] == lp:
                stats['equal'] += 1
            else:
                stats['differ'] += 1
                differ.append('%x: reused %d, %s vs %s' % (s, cached, o['ids'][:6], ids[:6]))
        if (stats['made'] + stats['forks'] + stats['revisits']) % 20 == 0:
            print('  %.1f passes (%d positions read), %d conversations live; %s; %.0f s' % (
                read / a.pool_positions, read, len(conv), stats, time.time() - t0))
            sys.stdout.flush()
    for s in list(conv):
        e.release(s)
    print('pool_soak: %.2f passes through the pool (%d positions read) in %.0f s; %s' % (read / a.pool_positions, read, time.time() - t0, stats))
    for d in differ[:10]:
        print('  DIFFER %s' % d)
    if a.log:
        time.sleep(1)
        lines = open(a.log, errors='replace').read()
        print("  rank 0's log: %d spilled to disk, %d back from disk, %d dropped from the disk tier, %d blocks differing on read-back" % (
            len(re.findall(r'spilled to disk', lines)), len(re.findall(r'back from disk:', lines)),
            len(re.findall(r'dropped from the disk tier', lines)), len(re.findall(r'differ from what was spilled', lines))))
    ok = stats['differ'] == 0 and stats['errors'] == 0 and stats['equal'] >= a.min_checks and read >= target
    print('pool_soak: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
