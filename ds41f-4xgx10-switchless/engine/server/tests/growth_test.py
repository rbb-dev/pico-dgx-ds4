#!/usr/bin/env python3
"""Growth (docs/engine.md, "The chunk pool"): a request reserves its prompt and a horizon of response (--grow-ahead), not every
token it may generate, and a response goes past its horizon one horizon at a time. (a) --n responses with max_new = the context
decode together on a small pool; (b) one response of --long tokens on a small --grow-ahead ends at its length and the log
shows its reservation growing. Start the server with e.g. --slots 8 --big-slots 0 --pool-tokens 1507328 --grow-ahead 256.

  python3 engine/server/tests/growth_test.py --socket S --tokens T.npy --log engine.log [--grow-ahead 256]
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import load_npy_i32  # noqa: E402
from slice_test import Mux  # noqa: E402

S0 = 0x67726f7700000000  # 'grow'
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--tokens-each', type=int, default=200)
    ap.add_argument('--ctx', type=int, default=1048576, help='max_new of (a): the context')
    ap.add_argument('--long', type=int, default=1500)
    ap.add_argument('--grow-ahead', type=int, default=256, help="the server's --grow-ahead")
    ap.add_argument('--log', default='', help="rank 0's log (engine-test.sh keeps it)")
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    m = Mux(a.socket)

    print('(a) %d replies of max_new %d together on a small pool' % (a.n, a.ctx))
    rids = [m.start('r%d' % i, S0 + 1 + i, T[5000 * i:5000 * i + 2000], max_new=a.ctx) for i in range(a.n)]
    done = set()

    def enough():
        for x in rids:
            if x not in done and m.ntok(x) >= a.tokens_each:
                m.cancel(x)
                done.add(x)
        return all(m.recs[x]['done'] for x in rids)
    m.pump(enough, timeout=900)
    firsts = [m.recs[x]['t_tok'][0] for x in rids if m.recs[x]['t_tok']]
    ends = [m.recs[x]['done']['t'] for x in rids]
    for x in rids:
        r = m.recs[x]
        check(len(r['ids']) >= a.tokens_each and not r['done']['error'],
              '%s: %d tokens, %s%s' % (r['name'], len(r['ids']), r['done']['reason'],
                                       ', ' + r['done']['error'] if r['done']['error'] else ''))
    print('    the last first token %.1f s after the submit, the first end at %.1f s' % (
        max(firsts) - min(m.recs[x]['t_submit'] for x in rids) if firsts else -1, min(ends) - min(m.recs[x]['t_submit'] for x in rids)))
    if a.log:
        time.sleep(1)
        lines = open(a.log, errors='replace').read().splitlines()
        batches = [int(x) for l in lines for x in re.findall(r'batch of (\d+)', l)]
        taken = sum(1 for l in lines if re.search(r'gives its \d+ chunks back to the pool', l))
        check(max(batches, default=1) == a.n and taken == 0,
              "rank 0's log: the largest batch %d of %d, %d times a reply's memory taken for another's" % (
                  max(batches, default=1), a.n, taken))
    for i in range(a.n):
        m.release(S0 + 1 + i)

    print('(b) one reply of %d tokens past a horizon of %d positions' % (a.long, a.grow_ahead))
    o = m.run(S0 + 100, T[100000:101000], max_new=a.long, stop=())
    check(len(o['ids']) == a.long and o['reason'] == 'length' and not o['error'],
          '%d tokens, %s%s' % (len(o['ids']), o['reason'], ', ' + o['error'] if o['error'] else ''))
    m.release(S0 + 100)
    if a.log:
        time.sleep(1)
        grows = sum(1 for l in open(a.log, errors='replace') if re.search(r'grows: its reservation reaches', l))
        want = a.long // a.grow_ahead // 2
        check(grows >= want, "rank 0's log: the reservation grew %d times (at least %d)" % (grows, want))
    print('growth_test: %s' % ('%d failed' % fails if fails else 'all checks pass'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
