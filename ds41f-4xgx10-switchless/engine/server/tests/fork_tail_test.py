#!/usr/bin/env python3
"""A fork's read-back is bounded by its fork point: a base that is partly split in memory and partly
on the SSD, and a twin of it entirely on the SSD (the control). A fork at position p whose need leaves no spare numbers: the base's pieces
BELOW p come back (the join to p, the unspill to p's budget), its tail and its bookmarks above p are untouched (the log says "joined to
position p" and the base's STATS conv_on_disk / quarter_slots stay), the fork equals the same prompt prefilled cold, and the base's next
turn restarts at its history's end. Through rank 0's socket (serve_test.py's client); the engine runs --kv-split cache with a disk tier.

  python3 engine/server/tests/fork_tail_test.py --socket S --tokens T.npy --log engine.log [--n 200000] [--p 60000]
  settings: MAXCTX=300000 --slots 2 --big-slots 0 --pool-tokens 330000 --disk-gib 32 --kv-split cache

Steps (each proved from the log and STATS; a step that did not happen ends the run with exit 2, never 0):
  1. the base B of --n positions, kept; a filler of the same size takes the pool: B's quarters go to the disk tier (conv_on_disk 1);
  2. a turn on B joins it back in part: a request at --p whose need leaves one number spare;
  3. the fork at p: its reply equals the cold prefill of its prompt; the log shows "joined to position <= p"; B's pieces above p are still
     only on the disk tier (STATS conv_on_disk still 1 with B's slots 0);
  4. B's next turn: restart at its history's end (`cached` = its history), no bad piece in the log.
Written in step C; not run.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_proof import Log, NotReached, need  # noqa: E402
from serve_test import Engine, load_npy_i32  # noqa: E402
from stats_test import stats  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    fails += not cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', required=True)
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--p', type=int, default=60000)
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    e = Engine(a.socket)
    log = Log(a.log)
    B, F, FILL = 0x666f726b01, 0x666f726b02, 0x666f726b03
    try:
        base = T[:a.n]
        o = e.run(B, base, max_new=8)
        hist = base + list(o['ids'])
        e.run(FILL, T[a.n:2 * a.n], max_new=2)
        _, _, s = stats(e)
        need(s['conv_on_disk'] >= 1, "the base's pieces are only on the disk tier: %d conversations there" % s['conv_on_disk'])
        fork = hist[:a.p] + T[2 * a.n:2 * a.n + 400]
        t0 = log.mark()
        of = e.run(F, fork, max_new=8, base=B)
        log.wait_for(r'joined to position (\d+)|back from disk', 'the fork\'s read-back or join', since=t0)
        _, _, s2 = stats(e)
        cold = e.run(0x666f726b09, fork, max_new=8)
        check(of['ids'] == cold['ids'], "the fork equals the same prompt prefilled cold (%s)" % ('equal' if of['ids'] == cold['ids'] else 'DIFFERS'))
        check(s2['conv_on_disk'] >= 1, "the base's tail above %d is still only on the disk tier (conv_on_disk %d)" % (a.p, s2['conv_on_disk']))
        ob = e.run(B, hist + T[3 * a.n:3 * a.n + 40], max_new=8)
        check(ob['cached'] >= len(hist) - 1, "the base's next turn restarts at its history's end (cached %d of %d)" % (ob['cached'], len(hist)))
    except NotReached as x:
        print('fork_tail_test: did not reach the path: %s' % x)
        return 2
    print('fork_tail_test: %s' % ('passed' if not fails else 'FAILED'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
