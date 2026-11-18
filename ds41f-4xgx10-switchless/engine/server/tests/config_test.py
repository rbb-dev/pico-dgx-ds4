#!/usr/bin/env python3
"""Interleaved prefill's live settings (CONFIG frames) through rank 0's socket (dsfront stopped): a CONFIG frame changes
what the next STATS frame shows; a field out of range or not sent keeps its value; the share takes effect at once.

  config_test.py --socket S --tokens T.npy        exit 0 when every check passes
"""
import argparse
import struct
import sys
import time

from serve_test import DONE, PROGRESS, TOKENS, CANCEL, Engine, load_npy_i32

CONFIG, STATS_REQ, STATS = 0x06, 0x05, 0x85
IL = ['on', 'whole_rows', 'push_rows', 'gap_ms', 'share', 'burst_max_ms', 'reading', 'queued']
EXTEND = 31  # STATS field index of `extend` (field 32 of 34; docs/protocol.md)
CTX_LIMIT, RANKS = 32, 33  # the window a request may use now, and the ranks (fields 33 and 34)


def config(e, **kw):
    order = ['on', 'whole_rows', 'push_rows', 'gap_ms', 'share', 'burst_max_ms', 'ssd_gib', 'keep_days', 'extend']
    e.send(CONFIG, struct.pack('<II', 3, 9) + struct.pack('<9q', *[kw.get(k, -1) for k in order]))


def config_v2(e, **kw):
    """A version 2 frame (8 fields, no extend): the engine keeps the extend setting it has."""
    order = ['on', 'whole_rows', 'push_rows', 'gap_ms', 'share', 'burst_max_ms', 'ssd_gib', 'keep_days']
    e.send(CONFIG, struct.pack('<II', 2, 8) + struct.pack('<8q', *[kw.get(k, -1) for k in order]))


def stats(e):
    time.sleep(0.7)  # the engine publishes at most twice a second; a CONFIG makes the next one current
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == STATS:
            version, n = struct.unpack('<II', p[:8])
            v = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])
            lo = 22 if version >= 2 else 19  # interleave's fields: after the pool's in version 2 (docs/protocol.md)
            if n < lo + 8:
                return {}
            d = dict(zip(IL, v[lo:lo + 8]))
            if n > EXTEND:
                d['extend'] = v[EXTEND]
            if n > RANKS:
                d['ctx_limit'], d['ranks'] = v[CTX_LIMIT], v[RANKS]
            return d


def decoder_rate(e, S, T, share):
    """a decoder's tokens a second while a 200K prompt is read, at the given share"""
    config(e, share=share)
    ra = e.submit(S, T[0:2000], max_new=20000, keep=False, stop=())[0]
    rb = None
    n_during = 0
    t_b = t_end = None
    while True:
        t, p = e.frame()
        if t == TOKENS:
            r, n = struct.unpack('<QI', p[:12])
            if r == ra:
                if rb is None:
                    rb = e.submit(S + 1, T[100000:300000], max_new=2, keep=False)[0]
                    t_b = time.time()
                elif t_end is None:
                    n_during += n
            elif r == rb and t_end is None:
                t_end = time.time()
                e.send(CANCEL, struct.pack('<Q', ra))
        elif t == DONE:
            r = struct.unpack('<Q', p[:8])[0]
            if r == ra:
                return n_during / max(1e-9, t_end - t_b), t_end - t_b
        elif t != PROGRESS and t != STATS:
            raise RuntimeError('frame %#x' % t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    res = {}
    s0 = stats(e)
    print('at start: %s' % s0)
    res['the start flags are reported (share 40, gap 600, on)'] = s0.get('share') == 40 and s0.get('gap_ms') == 600 and s0.get('on') == 1
    config(e, share=70, gap_ms=450, push_rows=65536)
    s1 = stats(e)
    print('after share 70, gap 450, push 65536: %s' % s1)
    res['a CONFIG frame changes what STATS shows'] = (s1.get('share'), s1.get('gap_ms'), s1.get('push_rows')) == (70, 450, 65536)
    res['fields not sent keep their values'] = s1.get('whole_rows') == s0.get('whole_rows') and s1.get('on') == 1
    config(e, share=95, gap_ms=10, whole_rows=100)
    s2 = stats(e)
    print('after out-of-range share 95, gap 10, whole 100: %s' % s2)
    res['values out of range are ignored'] = (s2.get('share'), s2.get('gap_ms'), s2.get('whole_rows')) == (70, 450, s0.get('whole_rows'))
    res['Extend context is on at the start (STATS field 32: 0 off, 1 on; the start flags here leave it at its default)'] = s0.get('extend') == 1
    res['at the start a request may use the start window (STATS ctx_limit = HELLO max_context), on 4 ranks'] = s0.get('ctx_limit') == e.max_ctx and s0.get('ranks') == 4
    config(e, extend=0)
    sx = stats(e)
    res['a CONFIG frame turns it off, the other settings stay'] = (
        sx.get('extend') == 0 and sx.get('share') == s1.get('share') and sx.get('gap_ms') == s1.get('gap_ms'))
    res['with it off a request may use at most 1,048,576 tokens (STATS ctx_limit)'] = sx.get('ctx_limit') == min(e.max_ctx, 1048576)
    config(e, extend=2)
    config(e, share=-1)
    res['an extend value out of range, or not sent, is ignored'] = stats(e).get('extend') == 0
    config_v2(e, share=70)
    res['a version 2 frame keeps it'] = stats(e).get('extend') == 0
    config(e, extend=1)
    sb = stats(e)
    res['and it goes back on, the window with it'] = sb.get('extend') == 1 and sb.get('ctx_limit') == e.max_ctx
    S = 0x636f6e6600000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    r20, w20 = decoder_rate(e, S + 10, T, 20)
    r80, w80 = decoder_rate(e, S + 20, T, 80)
    print('decoder during a 200K read: share 20 -> %.1f tok/s (read %.1f s); share 80 -> %.1f tok/s (read %.1f s)' % (r20, w20, r80, w80))
    res['the share takes effect at once (share 80 gives the decoder at least twice share 20)'] = r80 >= 2 * r20 and w80 > w20
    config(e, share=40, gap_ms=600, push_rows=131072)
    for name, ok in res.items():
        print('%s: %s' % ('ok  ' if ok else 'FAIL', name))
    print('config_test: %d of %d checks pass' % (sum(res.values()), len(res)))
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
