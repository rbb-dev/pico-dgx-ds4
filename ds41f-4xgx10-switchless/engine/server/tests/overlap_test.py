#!/usr/bin/env python3
"""Interleaved prefill, the overlap case: a session (A) decodes a long response while a long cold prompt (B) arrives. Reports A's
largest gap between token frames while B is read, A's tokens per second over B's read and B's time to first token; A's
tokens up to B's first token and B's first token with its logprobs must equal each run alone. With --max-gap-s the gap
is a check. Through rank 0's socket (dsfront stopped).

  overlap_test.py --socket S --tokens T.npy [--b-len 200000] [--a-new 1500] [--max-gap-s 0.8]
"""
import argparse
import struct
import sys
import time

from serve_test import DONE, PROGRESS, REASONS, TOKENS, Engine, load_npy_i32


def run_together(e, a_req, b_submit, b_after_tokens):
    """collect A's frames; once A has b_after_tokens tokens, submit B; collect both to their DONE"""
    (ra, nta) = a_req
    times = []  # (time, A tokens so far) at each of A's token frames
    b_req = None
    t_b = None
    pending = {ra}
    res = {ra: {'ids': [], 'lp': [], 'top': []}}
    ntop = {ra: nta}
    while pending:
        t, p = e.frame()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            o = res[r]
            o['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
            off = 16 + 4 * n
            if fl & 1:
                o['lp'] += struct.unpack('<%df' % n, p[off:off + 4 * n])
                off += 4 * n
            if fl & 2:
                k = ntop[r]
                pairs = struct.unpack('<' + 'If' * (n * k), p[off:off + 8 * n * k])
                for i in range(n):
                    o['top'].append(list(zip(pairs[2 * k * i::2][:k], pairs[2 * k * i + 1::2][:k])))
            if r == ra:
                times.append((time.time(), len(o['ids'])))
                if b_req is None and len(o['ids']) >= b_after_tokens:
                    b_req = b_submit()
                    t_b = time.time()
                    res[b_req[0]] = {'ids': [], 'lp': [], 'top': []}
                    ntop[b_req[0]] = b_req[1]
                    pending.add(b_req[0])
            elif 't_first' not in res[r]:
                res[r]['t_first'] = time.time() - t_b
        elif t == DONE:
            (r, reason, _pt, cached, _c, _q, p_us, _d, _el, _) = struct.unpack('<QIIIIQQQII', p[:56])
            res[r].update(reason=REASONS[reason], cached=cached, prefill_us=p_us, t_done=time.time())
            pending.discard(r)
        elif t != PROGRESS:
            raise RuntimeError('frame %#x' % t)
    return res[ra], res[b_req[0]], times, t_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--b-len', type=int, default=200000)
    ap.add_argument('--a-new', type=int, default=1500)
    ap.add_argument('--max-gap-s', type=float, default=0)
    a = ap.parse_args()
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    S = 0x6f766c7000000000 | (int(time.time() * 1000) & 0xffffffff) << 8
    pa, pb = T[0:2000], T[100000:100000 + a.b_len]
    stop = ()
    # alone first: A, then B (fresh sessions, nothing kept)
    a1 = e.run(S + 1, pa, max_new=a.a_new, n_top=2, keep=False, stop=stop)
    b1 = e.run(S + 2, pb, max_new=16, n_top=2, keep=False)
    a_rate = len(a1['ids']) / max(1e-9, a1['decode_us'] / 1e6)
    print('alone: A %d tokens at %.1f tok/s, B prefill %.1f s' % (len(a1['ids']), a_rate, b1['prefill_us'] / 1e6))
    # together: A decoding, B submitted once A has 100 tokens
    ra = e.submit(S + 3, pa, max_new=a.a_new, n_top=2, keep=False, stop=stop)
    a2, b2, times, t_b = run_together(e, ra, lambda: e.submit(S + 4, pb, max_new=16, n_top=2, keep=False), 100)
    t_b_end = t_b + b2['t_first']
    during = [(t, k) for t, k in times if t_b <= t <= t_b_end]
    gaps = [y[0] - x[0] for x, y in zip(times, times[1:]) if y[0] > t_b and x[0] < t_b_end]
    gap = max(gaps) if gaps else (t_b_end - t_b)
    toks = (during[-1][1] - during[0][1]) if len(during) > 1 else 0
    rate = toks / max(1e-9, b2['t_first'])
    print('together: B first token after %.1f s (alone %.1f s, %+.0f %%); A during B: %d tokens, %.1f tok/s (%.0f %% of '
          'alone); A largest gap %.2f s' % (b2['t_first'], b1['prefill_us'] / 1e6,
                                            100 * (b2['t_first'] / (b1['prefill_us'] / 1e6) - 1), toks, rate,
                                            100 * rate / max(1e-9, a_rate), gap))
    k = max([n for t, n in times if t <= t_b_end] or [0])  # A's tokens sent by B's first token
    ok_a = a2['ids'][:k] == a1['ids'][:k] and a2['lp'][:k] == a1['lp'][:k] and a2['top'][:k] == a1['top'][:k]
    ok_b = b2['ids'][:1] == b1['ids'][:1] and b2['lp'][:1] == b1['lp'][:1] and b2['top'][:1] == b1['top'][:1]
    full = lambda x, y: x['ids'] == y['ids'] and x['lp'] == y['lp'] and x['top'] == y['top']
    print('A up to B\'s first token (%d tokens) vs alone: %s; B\'s first token vs alone: %s; whole replies (batch after): '
          'A %s, B %s' % (k, 'BIT-EQUAL' if ok_a else 'DIFFER', 'BIT-EQUAL' if ok_b else 'DIFFER',
                          'equal' if full(a2, a1) else 'differ', 'equal' if full(b2, b1) else 'differ'))
    good = ok_a and ok_b and (a.max_gap_s <= 0 or gap <= a.max_gap_s)
    if a.max_gap_s > 0:
        print('largest gap %.2f s %s %.2f s' % (gap, '<=' if gap <= a.max_gap_s else '>', a.max_gap_s))
    print('overlap_test: %s' % ('PASS' if good else 'FAIL'))
    sys.exit(0 if good else 1)


if __name__ == '__main__':
    main()
