#!/usr/bin/env python3
"""Many lines on the shared pool (docs/engine.md, "The chunk pool" and "Scheduling: time-slicing and batching"): 32 sessions
live at once, each in a slot of its own, and more requests decoding than one batch step takes (16 entries) all get tokens.
(a) every session's first turn; (b) second turns resume from their slots with nothing going through the store; (c) --decode
sessions decode together and each reaches --tokens-each tokens before --limit seconds, then all cancel cleanly.

  dsgb10_rank --mode serve --slots 32 --big-slots 0 --max-ctx 1048576 ... > engine.log
  python3 engine/server/tests/lines_test.py --log engine.log [--socket S] [--tokens tokens.npy] [--n 65536]
      [--decode 24] [--tokens-each 64] [--limit 600] [--tolerance 128]

--n 16384 --limit 300 is the quick form. --lines is the server's --slots (32 by default; 64 works the same).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from budget_test import Log, Stalled, pump  # noqa: E402
from room_test import Flights  # noqa: E402
from serve_test import Engine, load_npy_i32  # noqa: E402

BAD = ('no slot for request', 'no slot memory')
MOVES = ('back from disk', 'spilled to disk')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', default='', help="the engine's stdout, written to a file (rank 0); required")
    ap.add_argument('--lines', type=int, default=32, help="the server's --slots: sessions live at once")
    ap.add_argument('--n', type=int, default=65536, help='positions of a session\'s first turn')
    ap.add_argument('--decode', type=int, default=24, help='requests decoding together in (c); more than 16')
    ap.add_argument('--tokens-each', type=int, default=64, help='tokens every request of (c) must reach')
    ap.add_argument('--limit', type=int, default=600, help='seconds a phase may take before it is a failure')
    ap.add_argument('--tolerance', type=int, default=128, help='positions a resume may fall short of the history')
    a = ap.parse_args()
    if not a.log or not os.path.exists(a.log):
        print("FAIL: --log FILE (the engine's stdout) is required: the checks that no session left its slot read it")
        return 1
    if a.decode > a.lines:
        print('FAIL: --decode %d is more than --lines %d' % (a.decode, a.lines))
        return 1
    T = load_npy_i32(a.tokens)
    TT = T + T
    e = Engine(a.socket)
    log = Log(a.log)
    log0 = os.path.getsize(a.log)  # this run's part of the log
    F = Flights(e)
    S = 0x7008000000000000
    ids = [S + i for i in range(a.lines)]
    ok = []

    def prompt(i):
        # a slice of the golden tokens from an offset of its own, its first token made unique: no two sessions share a prefix
        off = (i * 7919 + 1000) % max(1, len(T) - a.n)
        p = TT[off:off + a.n]
        p[0] = 1000 + i
        return p

    def check(name, cond, extra=''):
        ok.append(bool(cond))
        print('%s: %s%s' % ('ok  ' if cond else 'FAIL', name, extra), flush=True)

    t0 = time.time()
    hist = {}
    try:
        # (a) every session's first turn, all submitted at once
        print('(a) %d sessions of %d positions, first turns' % (a.lines, a.n), flush=True)
        log.since()
        rid = {}
        for i, s in enumerate(ids):
            rid[i] = F.submit(s, prompt(i), 4, stop=())
        pump(F, lambda: all(F.out[r]['done'] for r in rid.values()), 'first turns', limit=a.limit * 3)
        bad = [i for i, r in rid.items() if F.out[r]['done']['error'] or len(F.out[r]['ids']) < 4]
        check('(a) every first turn answered, %d tokens each' % 4, not bad, ' (failed: %s)' % bad if bad else '')
        for i, r in rid.items():
            hist[i] = prompt(i) + list(F.out[r]['ids'])
        print('    %.0f s; the sessions are live in %d slots' % (time.time() - t0, a.lines), flush=True)

        # (b) second turns: from the slots, nothing through the store
        print('(b) every session\'s second turn', flush=True)
        log.since()
        rid = {}
        for i, s in enumerate(ids):
            rid[i] = F.submit(s, hist[i] + T[5000 + 16 * i:5000 + 16 * i + 16], 8, stop=())
        pump(F, lambda: all(F.out[r]['done'] for r in rid.values()), 'second turns', limit=a.limit)
        short = [(i, F.out[r]['done']['cached'], len(hist[i])) for i, r in rid.items()
                 if F.out[r]['done']['cached'] < len(hist[i]) - a.tolerance]
        errs = [i for i, r in rid.items() if F.out[r]['done']['error'] or len(F.out[r]['ids']) < 8]
        check('(b) every second turn resumed from its slot (cached >= history - %d)' % a.tolerance, not short,
              ' (short: %s)' % short[:4] if short else '')
        check('(b) no error, 8 tokens each', not errs, ' (failed: %s)' % errs if errs else '')
        txt = log.since()
        moved = {m: txt.count(m) for m in MOVES}
        gave = txt.count(' back to the pool for a request')
        check('(b) no session left its slot: log has %d back from disk, %d spilled to disk, %d slots gave their chunks back' %
              (moved[MOVES[0]], moved[MOVES[1]], gave), not any(moved.values()) and not gave)
        for i, r in rid.items():
            hist[i] = hist[i] + T[5000 + 16 * i:5000 + 16 * i + 16] + list(F.out[r]['ids'])

        # (c) more requests decoding than one batch step holds
        print('(c) %d requests decoding together (a run holds 16)' % a.decode, flush=True)
        log.since()
        who = list(range(a.decode))
        rid = {i: F.submit(ids[i], hist[i] + T[9000 + 8 * i:9000 + 8 * i + 8], 1 << 20, stop=()) for i in who}
        try:
            pump(F, lambda: all(len(F.out[r]['ids']) >= a.tokens_each or F.out[r]['done'] for r in rid.values()),
                 'decoding together', limit=a.limit)
        except Stalled:
            ok.append(False)
            print('FAIL: (c) not every request reached %d tokens in %d s: %s' % (
                a.tokens_each, a.limit, sorted(len(F.out[r]['ids']) for r in rid.values())), flush=True)
        else:
            counts = sorted(len(F.out[r]['ids']) for r in rid.values())
            check('(c) every one of %d requests reached %d tokens (least %d, most %d)' % (
                a.decode, a.tokens_each, counts[0], counts[-1]), counts[0] >= a.tokens_each)
        early = [i for i, r in rid.items() if F.out[r]['done']]
        check('(c) none ended by itself', not early, ' (ended: %s)' % early if early else '')
        for r in rid.values():
            if not F.out[r]['done']:
                F.cancel(r)
        pump(F, lambda: all(F.out[r]['done'] for r in rid.values()), 'cancelling', limit=a.limit)
        notc = [i for i, r in rid.items() if F.out[r]['done']['reason'] != 'cancelled' or F.out[r]['done']['error']]
        check('(c) every request ended cancelled, no error', not notc, ' (not: %s)' % notc if notc else '')
        r = F.submit(S + 1000, T[150000:150300], 8)  # the engine still answers
        pump(F, lambda: F.out[r]['done'], 'a new session after', limit=a.limit)
        check('(c) a new session after is answered', not F.out[r]['done']['error'] and len(F.out[r]['ids']) > 0)
    except Stalled as err:
        print('FAIL: stalled in %s' % err)
        ok.append(False)
    except EOFError as err:
        print('FAIL: the engine stopped (%s): its log names the reason' % err)
        return 1
    with open(a.log, 'rb') as f:
        f.seek(log0)
        txt = f.read().decode(errors='replace')
    bad = [b for b in BAD if b in txt]
    check('the log holds none of %s' % list(BAD), not bad, ' (found: %s)' % bad if bad else '')
    print('lines: %s (%d of %d checks), %.0f s' % ('PASS' if all(ok) else 'FAIL', sum(ok), len(ok), time.time() - t0))
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
