#!/usr/bin/env python3
"""A request frame larger than --frame-stage-mib is not read: the engine skips its bytes, answers DONE with
the reason too_large and the limit in the text, and the connection serves the next request. Through rank 0's socket, one connection
(the engine serves ONE front-end connection at a time: a second replaces the first and cancels its requests).

  frame_cap_test.py --socket S --tokens T.npy [--frame-stage-mib 400] [--nodes local,...]

What it does, in order:
  1. a request of 36 x 1,100 image rows (387 MiB, under the cap) runs: the cap does not touch a frame under it;
  2. a request of --over-spans (46) x 1,100 rows (about 494 MiB, over the cap): DONE too_large, its request id, "--frame-stage-mib" in
     the text; no TOKENS for it;
  3. a text request straight after it on the same connection runs and answers (the stream stayed in step: the skipped bytes were
     exactly the frame's);
  4. a frame of another type over the cap is not built here (the front-end sends only SUBMIT frames this large);
  5. with --nodes, every box's MemAvailable before and after the over-the-cap request: it did not take the frame into memory
     (rank 0 within --spread GiB, default 0.3: a 494 MiB frame read whole would be 1 GiB with the request's copy of its rows).
Exit 0 pass, 1 fail, 2 not reached.

Engine settings: the release's start with the default --frame-stage-mib 400 (the line "store: ..." and the table's "request frames
being read" row show twice the cap: the payload and the request's copy of its rows), --front-mib set (a --pool-free-gib start is refused without it).
RED on a build without the frame cap: that build reads the 494 MiB frame whole (its largest frame is kMaxFrame = 100 x 1,100 rows) and prefills it.
"""
import argparse
import os
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_store_test import Conn, S0 as IMG_S0, collect, image_submit, layout, NotReached, need  # noqa: E402
from serve_test import DONE, SUBMIT, TOKENS, REASONS, load_npy_i32  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def meminfo(nodes):
    v = []
    for n in nodes:
        r = subprocess.run(['grep', 'MemAvailable', '/proc/meminfo'] if n == 'local' else ['ssh', '-o', 'BatchMode=yes', n, 'grep MemAvailable /proc/meminfo'],
                           capture_output=True, text=True, timeout=30)
        v.append(int(r.stdout.split()[1]) / 1048576.0)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--frame-stage-mib', type=int, default=400, dest='cap')
    ap.add_argument('--under-spans', type=int, default=36, dest='under')
    ap.add_argument('--over-spans', type=int, default=46, dest='over')
    ap.add_argument('--nodes', default='')
    ap.add_argument('--spread', type=float, default=0.3)
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    assert len(T) >= 100000, 'the token file is too short'
    e = Conn(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    nodes = a.nodes.split(',') if a.nodes else []
    try:
        rows_under, rows_over = a.under * 1100, a.over * 1100
        need(rows_under * 10240 < a.cap << 20 < rows_over * 10240, 'the two frames straddle the cap: %.0f MiB under, %.0f MiB over %d MiB' % (
            rows_under * 10240 / 2 ** 20, rows_over * 10240 / 2 ** 20, a.cap))
        text, spans = layout(T, 2000, a.under, 1100, 1200)
        o = collect(e, image_submit(e, IMG_S0 + 0x700, text, spans, 3, max_new=8, keep=False))
        check(not o['error'] and o['reason'] in ('stop', 'length'), 'a frame under the cap runs: %s %s' % (o['reason'], o['error']))
        m0 = meminfo(nodes) if nodes else []
        text, spans = layout(T, 2000, a.over, 1100, 1200)
        req = image_submit(e, IMG_S0 + 0x701, text, spans, 4, max_new=8, keep=False)
        got_tokens, done = 0, None
        while done is None:
            t, p = e.frame()
            if t == TOKENS and struct.unpack('<Q', p[:8])[0] == req[0]:
                got_tokens += 1
            elif t == DONE and struct.unpack('<Q', p[:8])[0] == req[0]:
                (r, reason, pt, cached, comp, q, pf, d, el, _) = struct.unpack('<QIIIIQQQII', p[:56])
                done = (REASONS[reason], p[len(p) - el:].decode())
        check(done[0] == 'too_large', 'a frame over the cap ends with DONE too_large (got %s)' % done[0])
        check('--frame-stage-mib' in done[1] and str(a.cap) in done[1], 'the limit is in the text: %s' % done[1])
        check(got_tokens == 0, 'and no TOKENS came for it')
        m1 = meminfo(nodes) if nodes else []
        for n, x, y in zip(nodes, m0, m1):
            check(x - y <= a.spread, '%s MemAvailable %.2f -> %.2f GiB: the frame was not taken into memory (spread %.2f)' % (n, x, y, a.spread))
        # the connection serves the next request: the skip left the stream in step
        t0 = time.time()
        o = collect(e, image_submit(e, IMG_S0 + 0x702, *layout(T, 2000, 1, 382, 3000), 5, max_new=8, keep=False))
        check(not o['error'] and o['reason'] in ('stop', 'length'), 'the next request on the same connection runs (%.1f s): %s %s' % (time.time() - t0, o['reason'], o['error']))
        rid = e.submit(IMG_S0 + 0x703, T[0:3000], max_new=8, keep=False, stop=())
        o = e.collect(rid)
        check(not o['error'] and o['completion'] > 0, 'and a text request: %d tokens' % o['completion'])
    except NotReached as x:
        print('frame_cap_test: NOT REACHED (%s)' % x)
        sys.exit(2)
    print('frame_cap_test: %s' % ('FAIL' if fails else 'PASS'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
