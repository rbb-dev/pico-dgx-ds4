#!/usr/bin/env python3
"""Kept sessions keep their YaRN factor (docs/engine.md "Kept sessions", "Checkpoints while serving"): phase before makes A at
factor 32 and B at factor 16 and saves their histories; phase after (on the restarted or the same engine) continues both
from their kept state at the same factors; --compare checks two runs' after-turns equal bit for bit. Run on an engine
started with --max-ctx 4194304 and --log; the after phase numbers its requests from 1001.

  python3 engine/server/tests/kept_factor_test.py --socket S --tokens T.npy --log engine.log --phase before --state st.json
  python3 engine/server/tests/kept_factor_test.py --socket S --tokens T.npy --log engine.log --phase after --state st.json --out test.json
  python3 engine/server/tests/kept_factor_test.py --compare ref.json test.json
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402

S0 = 0x6b66616300000000  # 'kfac'
A, B = S0 + 1, S0 + 2
AFTER_RID = 1001
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def tile(T, n, shift=0):
    """n ids from the token file, repeated to length: the same text at a different shift is a different prompt"""
    return [T[(i + shift) % len(T)] for i in range(n)]


def log_has(path, rid, text, wait_s=15.0):
    """rank 0's log has a line of request rid containing text: read again for up to wait_s, as the engine's output
    reaches the file in blocks"""
    pat = re.compile(r'serve: req %d\b.*%s' % (rid, text))
    t_end = time.time() + wait_s
    while True:
        with open(path, errors='replace') as f:
            if any(pat.search(line) for line in f):
                return True
        if time.time() >= t_end:
            return False
        time.sleep(0.5)


def turn(e, name, sid, prompt, rec, **kw):
    rq = e.submit(sid, prompt, max_new=16, n_top=1, **kw)
    o = e.collect(rq)
    o['rid'] = rq[0]
    show(name, o)
    rec[name] = {'ids': list(o['ids']), 'lp': list(o['lp']), 'cached': o['cached'], 'prompt': o['prompt'],
                 'reason': o['reason'], 'ttft': o['t_first'] if o['t_first'] is not None else -1.0}
    return o


def before(e, T, n, log):
    rec = {}
    pa = tile(T, n, 1)
    pb = T[100000:120000]
    a = turn(e, 'A cold %d' % n, A, pa, rec)
    check(not a['error'] and a['completion'] > 0, 'A answers: %s' % a['reason'])
    check(log_has(log, a['rid'], r'factor 32 \(window 2097152\)'), "rank 0 log: A at factor 32")
    b = turn(e, 'B cold 20000', B, pb, rec)
    check(not b['error'] and b['completion'] > 0, 'B answers: %s' % b['reason'])
    check(log_has(log, b['rid'], r'factor 16 \(window 1048576\)'), 'rank 0 log: B at factor 16')
    return rec, {'A': pa + list(a['ids']), 'B': pb + list(b['ids'])}


def after(e, T, hist, log):
    rec = {}
    e.next_id = AFTER_RID
    a = turn(e, 'A +300 (after)', A, hist['A'] + T[500000:500300], rec)
    check(not a['error'] and a['completion'] > 0, 'A answers: %s' % a['reason'])
    check(a['cached'] > 1000000, 'A restarted from its kept state: cached %d of %d (> 1,000,000)' % (a['cached'], a['prompt']))
    check(log_has(log, a['rid'], r'factor 32 \(window 2097152\)'), 'rank 0 log: A at factor 32')
    b = turn(e, 'B +40 (after)', B, hist['B'] + T[125000:125040], rec)
    check(not b['error'] and b['completion'] > 0, 'B answers: %s' % b['reason'])
    check(b['cached'] > 19000, 'B reused its kept state: cached %d of %d (> 19,000)' % (b['cached'], b['prompt']))
    check(log_has(log, b['rid'], r'factor 16 \(window 1048576\)'), 'rank 0 log: B at factor 16')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', default='', help="rank 0's log (the factor lines); required by both phases")
    ap.add_argument('--n', type=int, default=1100000, help="A's first prompt, past 1M (factor 32)")
    ap.add_argument('--phase', choices=['before', 'after'])
    ap.add_argument('--state', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--compare', nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(f)) for f in a.compare)
        keys = ('ids', 'lp', 'cached', 'prompt', 'reason')
        ok = 0
        for name in x:
            eq = name in y and all(x[name][k] == y[name][k] for k in keys)
            ok += eq
            print('  %-20s %s' % (name, 'EQUAL' if eq else 'DIFFER: %s' % {
                k: (str(x[name][k])[:60], str(y.get(name, {}).get(k))[:60]) for k in keys if y.get(name, {}).get(k) != x[name][k]}))
        print('kept_factor_test --compare: %d of %d turns equal' % (ok, len(x)))
        return 0 if ok == len(x) and set(y) == set(x) else 1
    if not a.phase or not a.log or not a.state:
        ap.error('--phase, --log and --state are required')
    T = [int(v) for v in load_npy_i32(a.tokens)]
    e = Engine(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    check(e.max_ctx >= 2097152, 'HELLO max_context %d holds factor 32 (start the engine with --max-ctx 4194304)' % e.max_ctx)
    if a.phase == 'before':
        rec, hist = before(e, T, a.n, a.log)
        json.dump(hist, open(a.state, 'w'))
    else:
        rec = after(e, T, json.load(open(a.state)), a.log)
    if a.out:
        json.dump(rec, open(a.out, 'w'))
    print('%s kept_factor_test --phase %s (%d failures)' % ('FAIL' if fails else 'PASS', a.phase, fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
