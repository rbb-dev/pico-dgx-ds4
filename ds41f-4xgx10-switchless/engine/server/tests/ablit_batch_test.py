#!/usr/bin/env python3
"""The runtime abliteration in every step kind, through rank 0's socket. Run with no front-end connected, on an engine
started with --ablit and without the batch switches (DSGB10_BATCH_CF, DSGB10_BATCH_TRIM, DSGB10_COPY, DSGB10_BUDGET), so
that a request decoding beside another gives the tokens it gives alone (docs/batching.md, "Exactness").

A request that asks for the abliteration (kFlagAblit) must give, beside another drafting session, the tokens it gives
alone; so must the session beside it, which does not ask (the control: it shows what this engine's settings allow). The
log-probabilities are reported, not checked: where a batch step's rows are grouped otherwise than a one-session step's,
their last bits differ for every request, the plain one included. Both orders are run, so the asking request is the
step's first entry once and its second once. Null control: the same request alone with and without the abliteration must
differ, or the test cannot see the edit on this prompt.

  python3 engine/server/tests/ablit_batch_test.py --socket S [--tokens T.npy]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_test import Mux  # noqa: E402
from serve_test import load_npy_i32  # noqa: E402

S0 = 0x61626c7400000000  # 'ablt'

fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    if not cond:
        fails += 1


def same(name, rec, ref):
    d = next((i for i, (a, b) in enumerate(zip(rec['ids'], ref['ids'])) if a != b), None)
    check(rec['ids'] == ref['ids'], '%s: its tokens equal its run alone (%d tokens)%s' % (
        name, len(rec['ids']), '' if rec['ids'] == ref['ids'] else ': first difference at token %s of %d' % (d, len(ref['ids']))))
    nd = sum(1 for a, b in zip(rec['lp'], ref['lp']) if a != b)
    md = max((abs(a - b) for a, b in zip(rec['lp'], ref['lp'])), default=0.0)
    print('  %s: log-probabilities %d of %d differ in their bits from its run alone (largest %.3g)' % (name, nd, len(ref['lp']), md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Mux(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    P = dict(prompt=T[:1000], max_new=300, n_top=0)
    Q = dict(prompt=T[40000:42000], max_new=260, n_top=0)
    sid = [S0]

    def run(*reqs):  # (name, kwargs, ablit): submitted at once, each in a fresh session; returns their records
        rids = []
        for name, kw, ab in reqs:
            sid[0] += 1
            k = dict(kw)
            rids.append(e.start(name, sid[0], k.pop('prompt'), ablit=ab, **k))
        e.pump(lambda: e.finished(*rids))
        return [e.recs[r] for r in rids]

    print('1. alone: P with and without the abliteration, Q with and without')
    p_ab, = run(('P+ablit', P, True))
    p_pl, = run(('P', P, False))
    q_ab, = run(('Q+ablit', Q, True))
    q_pl, = run(('Q', Q, False))
    for x in (p_ab, p_pl, q_ab, q_pl):
        print('  %s: %d tokens, %s' % (x['name'], len(x['ids']), x['done']['reason']))
    check(p_ab['ids'] != p_pl['ids'] or p_ab['lp'] != p_pl['lp'],
          'null control: P with the abliteration differs from P without it (the test can see the edit)')
    check(q_ab['ids'] != q_pl['ids'] or q_ab['lp'] != q_pl['lp'],
          'null control: Q with the abliteration differs from Q without it')

    print('2. together, submitted at once: P with the abliteration (first entry), Q without')
    p2, q2 = run(('P+ablit', P, True), ('Q', Q, False))
    same('P+ablit beside Q', p2, p_ab)
    same('Q beside P+ablit', q2, q_pl)

    print('3. together, submitted at once: P without (first entry), Q with the abliteration')
    p3, q3 = run(('P', P, False), ('Q+ablit', Q, True))
    same('P beside Q+ablit', p3, p_pl)
    same('Q+ablit beside P', q3, q_ab)

    print('PASS' if fails == 0 else 'FAIL: %d check(s)' % fails)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
