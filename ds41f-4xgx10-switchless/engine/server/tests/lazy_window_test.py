#!/usr/bin/env python3
"""The window of a YaRN factor past 1M is made when a request first needs it, out of the chunk pool, and given back (docs/engine.md,
"Context past 1M"): (s) the 4M start offers the 1M start's pool less the plain rotary tables; (a)-(c) no window line for
short prompts, a window opens for a long one and shrinks the pool by whole groups of 64 numbers; (d) it closes after
--idle-s and the pool returns exactly; (e) it opens again. Needs the engine alone (the pool is compared before and after).

  python3 engine/server/tests/lazy_window_test.py --socket S --tokens T.npy --log engine.log [--log-1m engine1m.log]
         [--only s,a,b,c,d,e]

(f) needs a fault build (DSGB10_TEST_FAULTS) started with DSGB10_LEND_FAIL=rank:R,step:win: the first window's memory check fails on
that rank after its buffers were made; the buffers are freed on every rank (compare the ranks' used memory before and after with
nvidia-smi: it returns to the start's), the pool is whole, and the next long prompt opens the window.
"""
import argparse
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402

S0 = 0x6c7a773000000000  # 'lzw0'
W16 = 1048576
STATS_REQ, STATS = 0x05, 0x85
NUMBER_GIB = 2916352 / 1073741824.0
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def tile(T, n, shift=0):
    return [T[(i + shift) % len(T)] for i in range(n)]


def submit(e, *a, **k):
    rq = e.submit(*a, **k)
    return rq[0], e.collect(rq)


def window_out(log, n_before):
    """(numbers out, groups, numbers above the last whole group) from the latest "window: factor 32 open" line after
    n_before, or None"""
    for line in reversed(lines(log)[n_before:]):
        m = re.search(r'window: factor 32 open in [0-9.]+ s: (\d+) numbers .* out of the pool: (\d+) groups of 64 and (\d+) above', line)
        if m:
            return tuple(int(x) for x in m.groups())
    return None


def pool_numbers(e):
    """STATS' pool_numbers: the engine publishes at most every half second, so wait past that for a fresh one"""
    time.sleep(0.8)
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == STATS:
            return struct.unpack('<Q', p[8:16])[0]


def lines(path):
    with open(path, errors='replace') as f:
        return f.read().splitlines()


def start_figure(path):
    """the pool's count this start offered by --pool-free-gib: 'chunk pool: ... (--pool-free-gib G: N on this rank'"""
    for l in lines(path):
        m = re.search(r'chunk pool: .*--pool-free-gib [0-9.]+: (\d+) on this rank', l)
        if m:
            return int(m.group(1))
    return None


def wait_line(path, pat, since, wait_s):
    """a log line matching pat after line index `since` (the log reaches the file in blocks): its index, or None"""
    rx = re.compile(pat)
    t_end = time.time() + wait_s
    while True:
        ls = lines(path)
        for i in range(since, len(ls)):
            if rx.search(ls[i]):
                return i
        if time.time() >= t_end:
            return None
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', required=True, help="the 4M engine's rank 0 log")
    ap.add_argument('--log-1m', default='', help="the log of a 1M start with the same --pool-free-gib (for (s))")
    ap.add_argument('--only', default='s,a,b,c,d,e')
    ap.add_argument('--long', type=int, default=1100000)
    ap.add_argument('--short', type=int, default=3000)
    ap.add_argument('--tol-gib', type=float, default=1.6)
    ap.add_argument('--max-window-gib', type=float, default=4.0)
    ap.add_argument('--idle-s', type=int, default=90)
    ap.add_argument('--no-reopen', action='store_true')
    a = ap.parse_args()
    only = set(a.only.split(','))
    T = [int(x) for x in load_npy_i32(a.tokens)]
    e = Engine(a.socket)
    check(e.max_ctx > W16, 'HELLO max_context %d is past 1,048,576 (start with --max-ctx 4194304)' % e.max_ctx)

    if 's' in only:
        print('(s) the start')
        ls = lines(a.log)
        check(any(re.search(r'spec: 1 step graph .*factors up to 16', l) for l in ls),
              'one step graph, factors up to 16 (the log line of spec_graphs)')
        check(not any('window: factor' in l for l in ls[:next((i for i, l in enumerate(ls) if 'listening' in l), len(ls))]),
              'no window line before serving')
        n4 = start_figure(a.log)
        check(n4 is not None, 'the 4M log has the pool line: %s' % n4)
        if a.log_1m:
            n1 = start_figure(a.log_1m)
            check(n1 is not None, 'the 1M log has the pool line: %s' % n1)
            if n4 and n1:
                diff = (n1 - n4) * NUMBER_GIB
                print('  the 1M start offers %d numbers, the 4M start %d: %.2f GiB apart' % (n1, n4, diff))
                check(diff <= a.tol_gib, 'the 4M start gives up at most %.1f GiB against the 1M start (%.2f)' % (a.tol_gib, diff))
                check(n4 <= n1 + 8, 'and offers no more than it')
        else:
            print('  (no --log-1m: the comparison with a 1M start is skipped)')

    N0 = pool_numbers(e)
    print('  the pool: %d numbers (%.2f GiB)' % (N0, N0 * NUMBER_GIB))

    if 'a' in only:
        print('(a) a short prompt: factor 16, no window')
        n_before = len(lines(a.log))
        rid, o = submit(e, S0 + 1, T[:a.short], max_new=8, n_top=1)
        show('short prompt', o)
        check(not o['error'] and o['completion'] > 0, 'it answers: %s' % o['reason'])
        time.sleep(2)
        check(not any('window:' in l for l in lines(a.log)[n_before:]), 'no window line for it')
        check(pool_numbers(e) == N0, 'the pool is as at the start')

    if 'b' in only:
        print('(b) a prompt of %d positions opens factor 32' % a.long)
        n_before = len(lines(a.log))
        rid, o = submit(e, S0 + 2, tile(T, a.long, 1), max_new=8, n_top=1)
        show('long prompt', o)
        check(not o['error'] and o['completion'] > 0, 'it answers: %s' % o['reason'])
        i_open = wait_line(a.log, r'window: factor 32 open', n_before, 15)
        i_fac = wait_line(a.log, r'serve: req %d factor 32 \(window 2097152\)' % rid, n_before, 15)
        check(i_open is not None, 'rank 0 log: window: factor 32 open')
        check(i_open is not None and i_fac is not None and i_open < i_fac, 'the window opened before the request ran at factor 32')
        N1 = pool_numbers(e)
        took = N0 - N1
        print('  the pool while the window is open: %d numbers, %d fewer (%.2f GiB)' % (N1, took, took * NUMBER_GIB))
        out = window_out(a.log, n_before)
        check(took > 0 and took * NUMBER_GIB <= a.max_window_gib, 'at most %.1f GiB out of the pool' % a.max_window_gib)
        check(out is not None and out[0] == took and out[0] == out[1] * 64 + out[2] and out[2] < 64,
              "the engine's own count agrees: %s numbers out (groups of 64 and the top numbers short of a group)" % (out,))

    if 'c' in only:
        print('(c) a prompt under 1M while the window is open')
        n_before = len(lines(a.log))
        rid, o = submit(e, S0 + 3, tile(T, a.short, 17), max_new=8, n_top=1)
        show('short prompt', o)
        check(not o['error'] and o['completion'] > 0, 'it answers: %s' % o['reason'])
        check(not any('window: factor' in l for l in lines(a.log)[n_before:]), 'no window line for it')

    if 'd' in only:
        print('(d) the window closes after %d s idle and the pool is its old size' % a.idle_s)
        n_before = len(lines(a.log))
        i_close = wait_line(a.log, r'window: factor 32 closed', n_before, a.idle_s + 30)
        check(i_close is not None, 'rank 0 log: window: factor 32 closed')
        check(pool_numbers(e) == N0, 'the pool is %d numbers again (%d)' % (N0, pool_numbers(e)))

    if 'e' in only and not a.no_reopen:
        print('(e) the window opens again')
        n_before = len(lines(a.log))
        rid, o = submit(e, S0 + 4, tile(T, a.long, 5), max_new=8, n_top=1)
        show('long prompt again', o)
        check(not o['error'] and o['completion'] > 0, 'it answers: %s' % o['reason'])
        check(wait_line(a.log, r'window: factor 32 open', n_before, 15) is not None, 'rank 0 log: window: factor 32 open again')
        N2 = pool_numbers(e)
        out = window_out(a.log, n_before)
        check(N2 < N0 and out is not None and out[0] == N0 - N2,
              'the pool is smaller while it is open by what the engine took (%d fewer, the engine: %s)' % (N0 - N2, out))
        i_close = wait_line(a.log, r'window: factor 32 closed', n_before, a.idle_s + 30)
        check(i_close is not None, 'and it closes again')
        check(pool_numbers(e) == N0, 'the pool is %d numbers again' % N0)

    if 'f' in only:
        print('(f) a window whose memory check fails on one rank')
        n_before = len(lines(a.log))
        rid, o = submit(e, S0 + 5, tile(T, a.long, 9), max_new=8, n_top=1)
        show('long prompt, first try', o)
        check(wait_line(a.log, r'window: factor 32: too little memory on a rank', n_before, 15) is not None,
              'rank 0 log: the window did not open (too little memory on a rank)')
        check(pool_numbers(e) == N0, 'the pool is %d numbers again' % N0)
        n_before = len(lines(a.log))
        rid, o = submit(e, S0 + 6, tile(T, a.long, 11), max_new=8, n_top=1)
        show('long prompt, second try', o)
        check(not o['error'] and o['completion'] > 0, 'it answers: %s' % o['reason'])
        check(wait_line(a.log, r'window: factor 32 open', n_before, 15) is not None, 'rank 0 log: the window opens')

    print('%s (%d failures)' % ('FAIL' if fails else 'PASS', fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
