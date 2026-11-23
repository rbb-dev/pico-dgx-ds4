#!/usr/bin/env python3
"""A base conversation whose read-back is cut (docs/engine.md, Serving: "a read-back that ends at its budget cuts the base above it").
A conversation spilled whole to the disk tier is read back with a budget (kStUnspill's count: the lists below the request's fork
point that must come back, then what the admission found spare). A fork from it whose own need takes the pool leaves little spare:
the read-back ends at its budget below the base's end, and the store cuts the base there (unspill -> cut_at): its lists above the
cut, its end bookmark and the bookmarks above the cut are gone, in the pool and on the disk. Then the base's own next turn restarts
below the cut and must equal the same prompt read cold, bit for bit.
The sequence, through rank 0's socket (serve_test.py's client):
  1. A, 40,000 positions, a window bookmark every 8,192 positions (--snap-every 8192) and one at its prompt's end.
  2. A filler (not kept): the store keeps nothing between requests, so A, which no line holds, is spilled whole at its end.
  3. F, a fork of A at 32,768 (a piece edge, with a bookmark of A at it), its prompt long enough that its reservation takes (nearly)
     all of the pool: the base's read-back budget is the 10 lists below 32,768 and the spare; the lists are read in position order, so
     it ends inside the 4 lists that start at 32,768 and the base is cut at 32,768. F then reads cold (the bookmarks came
     back for nothing: the budget was spent on lists) and must equal the same prompt prefilled in a fresh session.
  4. A's next turn: its history and 300 positions. Its restart point is at most 32,768 (the cut), and it must equal cold.
With --log the run proves each step from rank 0's log and STATS (the whole spill of A; the read-back of F's base with its budget and
a restart point below the end it had; the base's next turn restarting at or below the cut); a step that did not happen ends the run
with exit 2 ("did not reach the path"), never 0. Exit 0: every step ran and F and A's next turn equal cold. Exit 1: it ran and a
turn differs.

The arithmetic (the engine's constants: 8,192 entries a number, a lookahead of 4,096 positions, 64 entries of slack; a window
bookmark 2 numbers, an end bookmark 3):
  --pool-tokens 104000 asks for 32 numbers, but one line of --max-ctx 100000 takes 34 (numbers_for(0, 99,999): the engine's pool is the
     larger): 34 numbers = 95 MiB, the store's budget between requests 0 (under 256 MiB).
  A, 40,024 positions with its response: its lists 3 + 3 + 3 + 5 = 14 numbers (reservation 15); its bookmarks at 8,192, 16,384,
     24,576, 32,768 and 40,000 = 5 -> 10 numbers; its end bookmark 3: 27 numbers, within 31.
  F, a fork at p = 32,768: lists below p: 2 + 2 + 2 + 4 = 10, on the disk; its copies at p: 0 (p is a piece edge). F's prompt is sized
     at run time from STATS: the largest reservation the pool admits that the context holds (--max-ctx 100000: 99,xxx positions, 34
     numbers = the whole pool, spare 0), budget = 10 + spare; the lists of A at or above p are 15 - 10 = 5, so a spare under 5 cuts
     the read-back inside them (the lists are read in position order, at 32,768 four of them start together). The room for F then:
     34 less the 10 lists of the base in the pool = 24 <= 34 - 10. When no size fits (spare >= 5, or none admitted) the run ends
     with exit 2 and the numbers.
Engine settings (engine-test.sh takes them as one string; the context is lowered to what F needs):
  MAXCTX=100000 engine-test.sh BIN "--slots 2 --big-slots 0 --pool-tokens 104000 --disk-gib 4 --snap-every 8192" fork_cut_test.py \\
      --log ~/dsgb10/logs/engine-fork_cut_test.log

  python3 engine/server/tests/fork_cut_test.py --socket S --tokens T.npy --log engine.log
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_proof import END_NUMBERS, SNAP_NUMBERS, ceil_div, Log, NotReached, back_re, need, reservation, spill_re, stored_lists  # noqa: E402
from serve_test import Engine, load_npy_i32, same, show  # noqa: E402
from stats_test import stats  # noqa: E402

S0 = 0x666f726b63757400  # 'forkcut'
N = 40000          # A's prompt
P = 32768          # the fork point: a piece edge of every source, and a bookmark of A (--snap-every 8192)
EVERY = 8192
POOL_TOKENS = 104000


def lists_below(pos):
    """Lists whose first position is below `pos` (the four sources: 2 positions an entry, 8,192 entries, on sources 0-2)."""
    return sum(ceil_div(pos, 8192 * (2 if k < 3 else 1)) for k in range(4))


def turn(e, name, session, prompt, **kw):
    o = e.run(session, prompt, max_new=24, n_top=2, **kw)
    show(name, o)
    if o['error']:
        raise NotReached('%s: the engine answered with an error: %s' % (name, o['error']))
    return o


def run(a):
    T = [int(x) for x in load_npy_i32(a.tokens)]
    assert len(T) >= 175000, 'the token file is too short'
    e = Engine(a.socket)
    log = Log(a.log)
    A, F = S0 + 1, S0 + 2
    print('engine: %s, context %d' % (e.name, e.max_ctx))

    s = stats(e)[2]
    total = s['pool_numbers']
    need(s['lines'] == 2, 'the engine runs with 2 lines (%d)' % s['lines'])
    need(s['disk_cap_bytes'] > 0, 'the engine has a disk tier (%.2f GiB)' % (s['disk_cap_bytes'] / 2**30))
    need(total * s['number_bytes'] < 256 * 2**20, 'the pool is %d numbers = %.0f MiB, under 256 MiB: the store keeps nothing between requests'
         % (total, total * s['number_bytes'] / 2**20))
    snaps = [x for x in range(EVERY, N, EVERY)] + [N]
    a_numbers = stored_lists(N + 24) + SNAP_NUMBERS * len(snaps) + END_NUMBERS
    need(a_numbers + 4 <= total, 'A (%d lists + %d bookmarks of %d + end %d = %d numbers) fits the pool of %d beside a filler line'
         % (stored_lists(N + 24), len(snaps), SNAP_NUMBERS, END_NUMBERS, a_numbers, total))
    # F's size from the engine's own numbers: its reservation takes as much of the pool as the context allows (the least spare), and
    # the spare must be smaller than the lists of A above the fork point, so that the read-back cannot reach them all
    back = lists_below(P)
    above = reservation(N + 24) - back  # A's lists at or above the fork point: all of them are on the disk after the spill
    fits = [(reservation(n), n) for n in range(P + 1000, e.max_ctx + 1) if reservation(n) <= total]
    if not fits:
        raise NotReached('no fork of A fits the pool of %d numbers: even %d positions need %d' % (total, P + 1000, reservation(P + 1000)))
    most, fork_need = max(fits)[0], min(n for r, n in fits if r == max(fits)[0])
    spare = total - most
    fork_len = fork_need - 24
    budget = back + spare
    need(spare < above, "F of %d positions (the context %d holds it) reserves %d of the pool's %d numbers: spare %d < the %d lists of A "
         "at or above the fork point %d (%d lists below it): budget %d ends inside them" % (fork_len, e.max_ctx, most, total, spare, above, P, back, budget))
    spills0 = s['spills_total']

    print('1. A: %d positions' % N)
    a1 = turn(e, 'A cold %d' % N, A, T[:N])
    hist = N + len(a1['ids'])
    mark = log.mark()

    print('2. two fillers that keep their sessions: the first holds the other line, so the second takes A\'s (the least recently used')
    print('   idle line) and A, which no line holds then, is spilled whole at its end')
    turn(e, 'filler 1', S0 + 100, T[100000:102000], keep=True)
    turn(e, 'filler 2', S0 + 101, T[103000:105000], keep=True)
    # a stored conversation leaves this pool piece by piece during the next prefill (the end-of-request spill then skips it):
    # that A had pieces on the disk is proved by F's read-back of A below (the engine reads back only what is on the disk)
    s = stats(e)[2]
    need(s['conv_on_disk'] >= 1 and s['spills_total'] > spills0, '%d conversation(s) on the disk tier, %d images written' % (s['conv_on_disk'], s['spills_total'] - spills0))

    print('3. F, a fork of A at %d, with a prompt that takes the pool' % P)
    mark = log.mark()
    pf = T[:P] + T[100000 + 5000:100000 + 5000 + fork_len - P]
    f1 = turn(e, 'F fork at %d' % P, F, pf, base=A)
    b = log.wait_for(back_re(A), "the read-back of A", since=mark)
    got, of, pos, rp, of_rp = float(b.group(1)), float(b.group(2)), int(b.group(3)), int(b.group(4)), int(b.group(5))
    need(b.group(6) is not None, "the read-back of A carried a budget (%s numbers; the test's estimate %d)" % (b.group(6), budget))
    need(got < of, 'the read-back brought back %.1f MiB of the %.1f MiB on the disk: it ended at its budget' % (got, of))
    need(rp < of_rp, "A's restart point fell from %d to %d: the base was cut" % (of_rp, rp))
    need(rp <= P, 'the cut is at or below the fork point (%d <= %d)' % (rp, P))
    c1 = turn(e, 'F cold', S0 + 0xfff, pf, keep=False)
    ok1 = same('F vs cold', f1, c1)

    print("4. A's next turn: its history and 300 positions")
    h = T[:N] + list(a1['ids'])
    p2 = h + T[N + 5000:N + 5300]
    a2 = turn(e, 'A +300', A, p2)
    need(a2['cached'] <= P, 'A restarted at %d, at or below the cut (%d)' % (a2['cached'], P))
    c2 = turn(e, 'A +300 cold', S0 + 0xffe, p2, keep=False)
    ok2 = same('A +300 vs cold', a2, c2)
    for x in (A, F, S0 + 100, S0 + 101):
        e.release(x)
    return 0 if ok1 and ok2 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', required=True, help="rank 0's log from the engine's start (engine-test.sh keeps it)")
    a = ap.parse_args()
    try:
        rc = run(a)
    except NotReached as x:
        print('fork_cut_test: the test did not reach the path: %s' % x)
        return 2
    print('fork_cut_test: %s' % ('passed' if rc == 0 else 'FAILED'))
    return rc


if __name__ == '__main__':
    sys.exit(main())
