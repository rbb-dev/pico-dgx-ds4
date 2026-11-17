#!/usr/bin/env python3
"""The dashboard's frames from the engine (docs/protocol.md, version 4): HELLO's keep_positions and STATS version 2, the
chunk pool and its conversations, read off rank 0's socket while the engine does things the counts must show.

  (a) at the start: version 2 and 30 fields; the pool's numbers as its startup line says (--numbers), all free, every one
      of number_bytes (one chunk image), holding numbers x 16,384 / 5 positions; the lines (--lines); HELLO's
      keep_positions = (the pool's numbers + the disk tier's bytes / number_bytes) x 16,384 / 5.
  (b) after a request: a conversation (on a line or idle in memory) and the numbers it holds, free + held = numbers.
  (c) after a fork of it: numbers shared, two conversations sharing.
  (d) after both are released: nothing held, nothing shared, no conversation.

  python3 engine/server/tests/stats_test.py --socket S --tokens T.npy --numbers N --lines L
"""
import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32  # noqa: E402

STATS_REQ, STATS = 0x05, 0x85
NAMES = ['pool_numbers', 'pool_free', 'pool_held', 'pool_shared', 'pool_reserved', 'number_bytes', 'pool_positions',
         'conv_on_lines', 'conv_in_memory', 'conv_on_disk', 'conv_sharing', 'lines',
         'disk_cap_bytes', 'disk_used_bytes', 'spills_total', 'spill_bytes_total', 'spills_pending', 'restores_total',
         'restore_bytes_total', 'restored_tokens_total', 'restore_wait_ns_total', 'restore_wait_max_ns',
         'il_on', 'il_whole_rows', 'il_push_rows', 'il_gap_ms', 'il_share', 'il_burst_max_ms', 'il_reading', 'il_queued']
S0 = 0x7374617400000000  # 'stat'
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def stats(e):
    """One STATS: the engine publishes at most every half second, so wait past that for a fresh one."""
    time.sleep(0.6)
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == STATS:
            ver, n = struct.unpack('<II', p[:8])
            v = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])
            return ver, n, dict(zip(NAMES, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--numbers', type=int, required=True, help="the pool's numbers (its startup line)")
    ap.add_argument('--lines', type=int, required=True)
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    e = Engine(a.socket)

    print('(a) at the start')
    ver, n, s = stats(e)
    check(e.version == 4 and ver == 2 and n == 30, 'protocol %d, STATS version %d with %d fields' % (e.version, ver, n))
    check(s['pool_numbers'] == a.numbers and s['pool_free'] == a.numbers and s['pool_held'] == 0,
          'the pool: %d numbers, %d free, %d held' % (s['pool_numbers'], s['pool_free'], s['pool_held']))
    check(s['number_bytes'] == 2916352 and s['pool_positions'] == a.numbers * 16384 // 5,
          'a number is %d bytes; the pool holds %d positions' % (s['number_bytes'], s['pool_positions']))
    check(s['lines'] == a.lines and s['conv_on_lines'] + s['conv_in_memory'] + s['conv_on_disk'] == 0,
          '%d lines, no conversation' % s['lines'])
    keep = (a.numbers + s['disk_cap_bytes'] // s['number_bytes']) * 16384 // 5 if s['number_bytes'] else -1
    check(e.keep_positions == keep, "HELLO's keep_positions %d = the pool's and the disk tier's positions %d" % (e.keep_positions, keep))

    print('(b) after a request')
    A, B = S0 + 1, S0 + 2
    o = e.run(A, T[:40000], max_new=8)
    check(not o['error'], 'A: %d tokens' % len(o['ids']))
    _, _, s = stats(e)
    held_a = s['pool_held']
    check(s['conv_on_lines'] + s['conv_in_memory'] == 1 and held_a > 0 and s['pool_free'] + held_a == a.numbers,
          'one conversation (%d on a line, %d in memory), %d numbers held, %d free' % (
              s['conv_on_lines'], s['conv_in_memory'], held_a, s['pool_free']))
    check(s['pool_shared'] == 0 and s['conv_sharing'] == 0, 'nothing shared yet')

    print('(c) after a fork of it')
    o = e.run(B, T[:36000] + T[100000:100300], max_new=8, base=A)
    check(not o['error'] and 0 < o['cached'] <= 36000, 'B, a fork of A: %d tokens, reused %d' % (len(o['ids']), o['cached']))
    _, _, s = stats(e)
    check(s['pool_shared'] > 0 and s['conv_sharing'] == 2,
          '%d numbers shared, %d conversations sharing' % (s['pool_shared'], s['conv_sharing']))
    check(s['pool_free'] + s['pool_held'] == a.numbers and s['pool_held'] < 2 * held_a + 16,
          'the fork holds less than a copy would: %d held for both (A alone %d)' % (s['pool_held'], held_a))

    print('(d) after both are released')
    e.release(A)
    e.release(B)
    _, _, s = stats(e)
    check(s['pool_held'] == 0 and s['pool_shared'] == 0 and s['conv_on_lines'] + s['conv_in_memory'] == 0,
          'nothing held (%d), nothing shared (%d), no conversation' % (s['pool_held'], s['pool_shared']))
    print('stats_test: %s' % ('%d failed' % fails if fails else 'all checks pass'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
