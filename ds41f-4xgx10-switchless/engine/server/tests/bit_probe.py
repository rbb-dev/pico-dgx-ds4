#!/usr/bin/env python3
"""Bit probe: two engine builds or settings compared bit for bit on the decode / verify path: prompts cut from a token file
through rank 0's socket (serve_test.py's client), one session at a time, greedy, drafts on; every token's id, logprob and
top two are recorded. Each prompt is prefilled cold. Run with no front-end connected.

  bit_probe.py --socket S --tokens T.npy --out A.json [--n 8] [--len 2048] [--new 200]
  bit_probe.py --compare A.json B.json      exit 0 when every token, logprob and top-two pair is the same
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compare(fa, fb):
    a, b = json.load(open(fa)), json.load(open(fb))
    bad = 0
    for x, y in zip(a, b):
        n = min(len(x['ids']), len(y['ids']))
        same = 0
        while same < n and x['ids'][same] == y['ids'][same]:
            same += 1
        lp_diff = sum(1 for p, q in zip(x['lp'][:same], y['lp'][:same]) if p != q)
        lp_max = max((abs(p - q) for p, q in zip(x['lp'][:same], y['lp'][:same])), default=0.0)
        top_diff = sum(1 for p, q in zip(x['top'][:same], y['top'][:same]) if p != q)
        equal = x['ids'] == y['ids'] and x['lp'] == y['lp'] and x['top'] == y['top']
        bad += not equal
        print('prompt %d: %s  tokens %d/%d equal, logprobs differing %d (max |diff| %.3g), top-two differing %d, '
              'cached %d / %d' % (x['i'], 'BIT-EQUAL' if equal else 'DIFFER', same, max(len(x['ids']), len(y['ids'])),
                                  lp_diff, lp_max, top_diff, x['cached'], y['cached']))
    print('%d of %d prompts bit-equal (%d tokens)' % (len(a) - bad, len(a), sum(len(x['ids']) for x in a)))
    return bad == 0 and len(a) == len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='', help='an int32 .npy of token ids (real text)')
    ap.add_argument('--out', default='')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--len', type=int, default=2048)
    ap.add_argument('--new', type=int, default=200)
    ap.add_argument('--compare', nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    from serve_test import Engine, load_npy_i32, show  # noqa: E402
    if not args.tokens or not args.out:
        ap.error('--tokens and --out (or --compare A B)')
    toks = load_npy_i32(args.tokens)
    stride = len(toks) // args.n
    e = Engine(args.socket)
    base = 0x6269740000000000 | (int(time.time() * 1000) & 0xffffffffff) << 8  # 'bit', then fresh ids per run
    out = []
    for i in range(args.n):
        o = e.run(base + i, toks[i * stride:i * stride + args.len], max_new=args.new, n_top=2, keep=False)
        show('probe %d' % i, o)
        out.append({'i': i, 'ids': list(o['ids']), 'lp': list(o['lp']), 'top': [[list(p) for p in t] for t in o['top']],
                    'cached': o['cached'], 'reason': o['reason'], 'completion': o['completion'],
                    'decode_us': o['decode_us']})
    json.dump(out, open(args.out, 'w'))


if __name__ == '__main__':
    main()
