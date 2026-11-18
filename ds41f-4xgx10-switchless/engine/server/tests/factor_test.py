#!/usr/bin/env python3
"""The YaRN factor per request, through rank 0's socket, on an engine started with --max-ctx 2097152 or more (4194304 for
the full range) and --log: (0) HELLO and a short prompt at factor 16; (a) a prompt past 1M at factor 32; (b) a conversation
continued past 1M becomes a new session at factor 32, equals a cold run bit for bit, and its sibling reuses the old
session; (c) a response that reaches its window crosses and completes; (d) Extend context off refuses a prompt over the window.

  python3 engine/server/tests/factor_test.py --socket S --tokens T.npy --log engine.log [--only 0,a,b,c,d]
"""
import argparse
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402

S0 = 0x6663743000000000  # 'fct0'
W16 = 1048576
CONFIG, STATS_REQ, STATS = 0x06, 0x05, 0x85
EXTEND = 31  # STATS field index of `extend`
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def tile(T, n, shift=0):
    """n ids from the token file, repeated to length: the same text at a different shift is a different prompt"""
    return [T[(i + shift) % len(T)] for i in range(n)]


def log_line(path, pattern, wait_s=10.0):
    """A line of rank 0's log matching pattern (buffered writes: waits up to wait_s)."""
    pat = re.compile(pattern)
    t_end = time.time() + wait_s
    while True:
        with open(path, errors='replace') as f:
            if any(pat.search(line) for line in f):
                return True
        if time.time() >= t_end:
            return False
        time.sleep(0.5)


def log_has(path, rid, text, wait_s=10.0):
    """rank 0's log has a line of request rid containing text: read again for up to wait_s, as the engine's output
    reaches the file in blocks (a request that ends quickly can end before its lines are written)"""
    pat = re.compile(r'serve: req %d\b.*%s' % (rid, text))
    t_end = time.time() + wait_s
    while True:
        with open(path, errors='replace') as f:
            if any(pat.search(line) for line in f):
                return True
        if time.time() >= t_end:
            return False
        time.sleep(0.5)


def set_extend(e, on):
    """CONFIG extend (the other fields not sent), then STATS until it shows the value (published at most twice a second)"""
    e.send(CONFIG, struct.pack('<II', 3, 9) + struct.pack('<9q', *([-1] * 8 + [1 if on else 0])))
    t_end = time.time() + 10
    while time.time() < t_end:
        time.sleep(0.7)
        e.send(STATS_REQ, struct.pack('<II', 0, 0))
        while True:
            t, p = e.frame()
            if t == STATS:
                break
        n = struct.unpack('<I', p[4:8])[0]
        if n > EXTEND and struct.unpack('<Q', p[8 + 8 * EXTEND:16 + 8 * EXTEND])[0] == (1 if on else 0):
            return True
    return False


def submit(e, *a, **k):
    rq = e.submit(*a, **k)
    return rq[0], e.collect(rq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', required=True, help="rank 0's log (the factor lines)")
    ap.add_argument('--only', default='0,a,b,c,d')
    ap.add_argument('--long', type=int, default=1100000, help="(a)'s prompt, past 1M")
    ap.add_argument('--first', type=int, default=1040000, help="(b)'s first turn, under 1M")
    ap.add_argument('--second', type=int, default=1100000, help="(b)'s second turn, past 1M")
    ap.add_argument('--near', type=int, default=1046000, help="(c)'s prompt, shortly under 1M")
    ap.add_argument('--over', type=int, default=1050000, help="(d)'s prompt, past 1M")
    ap.add_argument('--reply', type=int, default=6000, help="(c)'s max_new: takes it past 1,048,576")
    a = ap.parse_args()
    only = set(a.only.split(','))
    T = [int(x) for x in load_npy_i32(a.tokens)]
    e = Engine(a.socket)

    if '0' in only:
        print('(0) the window and a short prompt')
        check(e.max_ctx > W16, 'HELLO max_context %d is past 1,048,576 (start the engine with --max-ctx 2097152 or more)' % e.max_ctx)
        rid, o = submit(e, S0 + 1, T[:3000], max_new=8, n_top=1)
        show('short prompt', o)
        check(o['reason'] in ('stop', 'length') and not o['error'], 'it answers: %s' % o['reason'])
        check(log_has(a.log, rid, r'factor 16 \(window 1048576\)'), 'rank 0 log: factor 16')

    if 'a' in only:
        print('(a) a prompt of %d positions runs at factor 32' % a.long)
        rid, o = submit(e, S0 + 2, tile(T, a.long, 1), max_new=8, n_top=1)
        show('long prompt', o)
        check(not o['error'] and o['completion'] > 0, 'it answers: %s' % o['reason'])
        check(log_has(a.log, rid, r'factor 32 \(window 2097152\)'), 'rank 0 log: factor 32')

    if 'b' in only:
        print('(b) a conversation continued past 1M')
        B = S0 + 3
        P1 = tile(T, a.first, 7)
        rid1, o1 = submit(e, B, P1, max_new=16, n_top=1)
        show('first turn', o1)
        check(log_has(a.log, rid1, r'factor 16 '), 'the first turn ran at factor 16')
        P2 = P1 + list(o1['ids']) + tile(T, a.second - len(P1) - len(o1['ids']), 99)
        rid2, o2 = submit(e, B, P2, max_new=16, n_top=1)
        show('second turn (past 1M)', o2)
        check(o2['cached'] == 0, 'a new session read whole at the new factor (cached %d)' % o2['cached'])
        check(log_has(a.log, rid2, r'factor 32 '), 'the second turn ran at factor 32')
        check(log_line(a.log, r'serve: conversation %016x goes on past its factor-16 window: that copy .* to the disk tier' % B),
              'the factor-16 copy it left went to the disk tier at the crossing')
        rid3, o3 = submit(e, S0 + 4, P2, max_new=16, n_top=1)  # cold, another session, no base
        show('the same prompt, cold', o3)
        check(o3['cached'] == 0 and log_has(a.log, rid3, r'factor 32 '), 'cold at factor 32')
        check(list(o2['ids']) == list(o3['ids']) and list(o2['lp']) == list(o3['lp']),
              'the crossing turn equals the cold run: %d tokens and log-probabilities bit for bit' % len(o2['ids']))
        # a sibling whose prompt still fits the window matches the factor-16 session the conversation left
        P3 = P1 + list(o1['ids']) + tile(T, 100, 5)
        rid4, o4 = submit(e, S0 + 5, P3, base=B, max_new=8, n_top=1)
        show('sibling under 1M (base = the conversation)', o4)
        check(o4['cached'] >= a.first - 2048 and log_has(a.log, rid4, r'factor 16 '),
              'the sibling reused the old factor-16 session (read back from the disk tier): cached %d of %d' % (o4['cached'], len(P3)))

    if 'c' in only:
        print('(c) a reply that reaches its window (prompt %d, max_new %d)' % (a.near, a.reply))
        rid, o = submit(e, S0 + 6, tile(T, a.near, 3), max_new=a.reply, stop=(), n_top=0)
        show('mid-reply crossing', o)
        check(not o['error'] and o['reason'] == 'length', 'reason %s, error "%s"' % (o['reason'], o['error']))
        check(o['completion'] == a.reply, 'all %d tokens sent (%d)' % (a.reply, o['completion']))
        check(a.near + o['completion'] > W16, 'positions went past 1,048,576 (%d)' % (a.near + o['completion']))
        check(log_has(a.log, rid, r'crosses from factor 16 to 32'), 'rank 0 log: the request crossed')
        check(log_has(a.log, rid, r'factor 32 '), 'rank 0 log: it was read again at factor 32')

    if 'd' in only:
        print('(d) Extend context off: a prompt of %d positions is refused' % a.over)
        check(set_extend(e, False), 'STATS shows extend 0')
        try:
            t0 = time.time()
            rid, o = submit(e, S0 + 7, tile(T, a.over, 11), max_new=8, n_top=1)
            show('prompt over the window', o)
            check(o['error'] != '' and o['reason'] == 'error', 'refused: reason %s, error "%s"' % (o['reason'], o['error']))
            check('does not fit the context of %d' % W16 in o['error'], 'the window it names is %d' % W16)
            check(o['completion'] == 0 and o['cached'] == 0, 'no prefill: %d tokens, cached %d' % (o['completion'], o['cached']))
            check(time.time() - t0 < 60, 'it came back at once (%.1f s)' % (time.time() - t0))
        finally:
            check(set_extend(e, True), 'STATS shows extend 1 again')

    print('%s (%d failures)' % ('FAIL' if fails else 'PASS', fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
