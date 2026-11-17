#!/usr/bin/env python3
"""The SSD tier per piece (docs/engine.md, "The disk tier, per piece"): images written once, the last piece evicted first,
copies kept after a read-back. (a) write-behind at a turn's save; (a2) idle flush after --flush-s; (b) a squeezed A restarts
at the same position and answers bit-equal after read-back; (c) again after the read-back; (d) the disk tier's size live
through a CONFIG frame. Through rank 0's socket, on a small pool with a disk tier.

  python3 engine/server/tests/ssd_test.py --socket S --tokens T.npy [--log engine.log]
  e.g. --slots 4 --big-slots 0 --pool-tokens 1507328 --disk-gib 32: four 300K fillers take every line and do not fit
  beside A (with fewer fillers than lines A can keep its line through a round: nothing then reads it back)
"""
import argparse
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32  # noqa: E402
from stats_test import NAMES, STATS, STATS_REQ  # noqa: E402

S0 = 0x7373647400000000  # 'ssdt'
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def stats(e):
    time.sleep(0.6)
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == STATS:
            _, n = struct.unpack('<II', p[:8])
            return dict(zip(NAMES, struct.unpack('<%dQ' % n, p[8:8 + 8 * n])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', type=int, default=60000, help="A's prompt")
    ap.add_argument('--fillers', type=int, default=4, help="the engine's lines (--slots)")
    ap.add_argument('--filler-n', type=int, default=300000)
    ap.add_argument('--flush-s', type=int, default=0, help="the server's --ssd-flush-s (0: not checked)")
    ap.add_argument('--log', default='')
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    e = Engine(a.socket)
    A = S0 + 1
    P = T[:a.n]

    print('(a) write-behind')
    s0 = stats(e)
    e.run(A, P, max_new=12)
    ref = e.run(A, P, max_new=12, n_top=1)
    s1 = stats(e)
    img = s1['number_bytes']
    check(s1['spills_total'] > s0['spills_total'] and s1['spill_bytes_total'] - s0['spill_bytes_total'] >= 8 * img and s1['conv_on_disk'] == 0,
          'A\'s turn wrote %d images (%.1f MiB) with no conversation on the disk' % (
              s1['spills_total'] - s0['spills_total'], (s1['spill_bytes_total'] - s0['spill_bytes_total']) / 2 ** 20))

    def squeeze(round_):
        for i in range(a.fillers):
            off = 50000 + (round_ * a.fillers + i) * 7919 % max(1, len(T) - a.filler_n - 1)
            o = e.run(S0 + 100 * (round_ + 1) + i, T[off:off + a.filler_n], max_new=4)
            check(not o['error'], 'filler %d.%d: %d tokens' % (round_, i, len(o['ids'])))
        s = stats(e)
        print('    pool: %d of %d held, %d conversations on the disk; %d images written in all' % (
            s['pool_held'], s['pool_numbers'], s['conv_on_disk'], s['spills_total']))
        return s

    if a.flush_s > 0:
        print('(a2) idle flush')
        time.sleep(a.flush_s + 2)
        s1b = stats(e)
        check(s1b['spills_total'] > s1['spills_total'],
              'after %d s idle, A\'s tail and bookmarks written: %d more images (%.1f MiB)' % (
                  a.flush_s + 2, s1b['spills_total'] - s1['spills_total'], (s1b['spill_bytes_total'] - s1['spill_bytes_total']) / 2 ** 20))
    for r in (1, 2):
        print('(%s) squeezed, then A again' % 'bc'[r - 1])
        s = squeeze(r)
        check(s['conv_on_disk'] >= 1, 'a conversation is on the disk (A gave up pieces)')
        o = e.run(A, P, max_new=12, n_top=1)
        s2 = stats(e)
        check(s2['restores_total'] > s['restores_total'], 'A was read back from the disk (%d read-backs)' % (s2['restores_total'] - s['restores_total']))
        same = o['ids'] == ref['ids'] and o['lp'] == ref['lp'] and o['cached'] == ref['cached']
        check(same, 'A after the read-back: restarted at %d (reference %d), %s' % (
            o['cached'], ref['cached'], 'bit-equal' if same else 'DIFFERS: %s vs %s' % (o['ids'][:6], ref['ids'][:6])))
        for i in range(a.fillers):
            e.release(S0 + 100 * r + i)
    print('(d) the size live')
    cap0 = stats(e)['disk_cap_bytes']

    def config(gib):
        e.send(0x06, struct.pack('<II', 2, 8) + struct.pack('<8q', -1, -1, -1, -1, -1, -1, gib, -1))
        time.sleep(1.5)
        return stats(e)['disk_cap_bytes']
    small = config(1)
    back = config(cap0 >> 30)
    o = e.run(S0 + 900, T[200000:202000], max_new=4)
    check(small == 1 << 30 and back == cap0 and not o['error'],
          'the disk tier %.1f GiB -> 1 GiB (%.1f) -> back to %.1f GiB; a request after it: %d tokens' % (
              cap0 / 2 ** 30, small / 2 ** 30, back / 2 ** 30, len(o['ids'])))
    e.release(S0 + 900)
    e.release(A)
    if a.log:
        time.sleep(1)
        backs = [float(m) for m in re.findall(r'state: session \d+ back from disk: ([0-9.]+) MiB', open(a.log, errors='replace').read())]
        print("  rank 0's read-backs: %s MiB" % backs)
    print('ssd_test: %s' % ('%d failed' % fails if fails else 'all checks pass'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
