#!/usr/bin/env python3
"""Needles across the YaRN factor boundary (docs/engine.md, "Context past 1M"): real text with a sentence "The vault code for
NAME is NNNNNN." planted before, just under, at and after a boundary (1,048,576 tokens; 2,097,152 with --also-2m), and a
question at the end asking for each code; through rank 0's socket with the prompt built from token ids. A needle is found
when a line of the answer holds its name and code; exit 1 when the exact arm misses one the control arm found.

  python3 engine/server/tests/needles_boundary.py --socket S --tokenizer .../tokenizer.json [--tokens docs-real.npy]
        [--also-2m] [--out needles.json] [--release]
  python3 engine/server/tests/needles_boundary.py --selftest        (the instrument's checks, no engine, no tokenizer)

Needs python with `tokenizers`; the engine must run with --max-ctx 2097152 or
more (4194304 with --also-2m, whose 2M scenario needs room past 2,097,152 + the question).
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import Engine, load_npy_i32, show  # noqa: E402

S0 = 0x6e6565646c000000  # 'needl'
B1M, B2M = 1048576, 2097152
CLASSES = ['before', 'under', 'at', 'after']
NAMES = {B1M: ['Alpha', 'Beta', 'Gamma', 'Delta'], B2M: ['Epsilon', 'Zeta', 'Eta', 'Theta']}


class Tok:
    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            sys.exit('needles_boundary: needs the tokenizers library (pip install tokenizers) and --tokenizer tokenizer.json')
        self.t = Tokenizer.from_file(path)

    def enc(self, s):
        return self.t.encode(s, add_special_tokens=False).ids

    def dec(self, ids):
        return self.t.decode(list(ids), skip_special_tokens=False)


def tile(T, start, n):
    out = []
    i = start % len(T)
    while len(out) < n:
        k = min(n - len(out), len(T) - i)
        out += T[i:i + k]
        i = 0
    return out


def sentence(name, code):
    return '\nThe vault code for %s is %s.\n' % (name, code)


def question(names):
    return '\n\nList the vault codes for %s, one per line as name: code.\n%s:' % (', '.join(names[:-1]) + ' and ' + names[-1],
                                                                              names[0])


def plan(B, Lp, n):
    """needle start positions by class for a prompt of n tokens (the 'at' start is fixed up by the caller to centre it)"""
    return {'before': B // 2, 'under': Lp - 100, 'at': B, 'after': B + 2048}


def build(T, tok, B, gap, extra, seed):
    """-> (C, Lp, needles [(class, name, code, start)]); C ends with the question"""
    Lp = B - gap
    n = Lp + extra
    C = tile(T, 12345 + B % 1000, n)
    names = NAMES[B]
    q = tok.enc(question(names))
    pos = plan(B, Lp, n)
    needles = []
    for k, cls in enumerate(CLASSES):
        code = '%06d' % ((seed * 7919 + k * 104729 + B // 4096 * 31) % 900000 + 100000)
        ids = tok.enc(sentence(names[k], code))
        start = pos[cls] - (len(ids) // 2 if cls == 'at' else 0)
        assert start + len(ids) < n - len(q) - 64, 'needle %s does not fit (raise --extra)' % cls
        C[start:start + len(ids)] = ids
        needles.append((cls, names[k], code, start))
    C[n - len(q):] = q
    return C, Lp, needles


def build_control(T, tok, needles, n=32768):
    """the same sentences at 25/50/60/80 % of a short prompt, the same question"""
    C = tile(T, 777, n)
    q = tok.enc(question([nm for _, nm, _, _ in needles]))
    for (cls, name, code, _), frac in zip(needles, (0.25, 0.5, 0.6, 0.8)):
        ids = tok.enc(sentence(name, code))
        s = int(n * frac)
        C[s:s + len(ids)] = ids
    C[n - len(q):] = q
    return C


def found(text, needles):
    """name -> True when some line of text holds the name and its code (text starts with the question's own 'Name:')"""
    out = {}
    for _, name, code, _ in needles:
        out[name] = any(name.lower() in ln.lower() and code in ln for ln in text.split('\n'))
    return out


def arm(e, name, session, prompt, n_new, **kw):
    t0 = time.time()
    o = e.collect(e.submit(session, prompt, max_new=n_new, **kw))
    show(name, o)
    print('%-28s wall %.1f s' % ('', time.time() - t0))
    return o


def answer_text(tok, names, o):
    return names[0] + ':' + tok.dec(o['ids'])


def scenario(e, tok, T, B, a, results):
    sid = S0 + (B // B1M) * 16
    C, Lp, needles = build(T, tok, B, a.gap, a.extra, a.seed)
    names = [nm for _, nm, _, _ in needles]
    print('\n== boundary %d: prompt %d tokens, boundary - gap %d; needles %s' % (
        B, len(C), Lp, ', '.join('%s %s at %d (%s)' % (nm, cd, st, c) for c, nm, cd, st in needles)))
    if e.max_ctx <= len(C):
        sys.exit('needles_boundary: the engine serves %d positions: no room for the %d-token prompt' % (e.max_ctx, len(C)))
    ctrl = arm(e, 'control %d' % 32768, sid + 1, build_control(T, tok, needles), a.n_new, keep=False)
    f_ctrl = found(answer_text(tok, names, ctrl), needles)
    run = {'control': f_ctrl}
    bad = []
    o = arm(e, 'exact (fresh, cold)', sid + 3, C, a.n_new)
    if o['cached'] != 0:
        bad.append('exact reused %d positions: it is not the exact path' % o['cached'])
    run['exact'] = found(answer_text(tok, names, o), needles)
    run['exact_answer'] = answer_text(tok, names, o)
    if a.release:
        for k in (1, 3):
            e.release(sid + k)
    results[B] = {'needles': needles, 'run': run, 'bad': bad}
    return bad


def report(results, arms):
    print('\nboundary  position  class   needle   arm        found')
    rows = []
    ok = True
    for B, r in results.items():
        run = r['run']
        for cls, name, code, st in r['needles']:
            for which in ['control'] + list(arms):
                f = run[which][name]
                rows.append((B, st, cls, which, f))
                print('%-9d %-9d %-7s %-8s %-10s %s' % (B, st, cls, name, which, 'yes' if f else 'NO'))
        lost = [nm for nm, f in run['control'].items() if f and 'exact' in run and not run['exact'][nm]]
        ok &= not lost and not r['bad']
        for b in r['bad']:
            print('INVALID: ' + b)
        s = 'boundary %d: control finds %d of 4' % (B, sum(run['control'].values()))
        for which in arms:
            s += ', %s %d of 4' % (which, sum(run[which].values()))
        s += '; exact misses %s the control found' % (lost or 'none') if 'exact' in run else ''
        print(s)
    return ok, rows


def selftest():
    class Fake:
        def enc(self, s):
            return [3 + b for b in s.encode()]

        def dec(self, ids):
            return bytes(i - 3 for i in ids).decode()
    tok, T = Fake(), list(range(1000, 6000))
    C, Lp, needles = build(T, tok, B1M, 4096, 12288, 3)
    assert len(C) == B1M - 4096 + 12288 and Lp == B1M - 4096
    for cls, name, code, st in needles:
        ids = tok.enc(sentence(name, code))
        assert C[st:st + len(ids)] == ids, cls
    at = next(st for c, _, _, st in needles if c == 'at')
    sl = len(tok.enc(sentence('Gamma', '123456')))
    assert at < B1M < at + sl, 'the at needle straddles the boundary'
    assert [st for _, _, _, st in needles][:2] == [B1M // 2, Lp - 100]
    assert tok.dec(C[-len(tok.enc(question(NAMES[B1M]))):]) == question(NAMES[B1M])
    ctl = build_control(T, tok, needles)
    assert len(ctl) == 32768
    txt = 'Alpha: %s\nBeta: %s\nGamma - no idea\nDelta: %s, wrong' % (needles[0][2], needles[1][2], '000000')
    f = found(txt, needles)
    assert f == {'Alpha': True, 'Beta': True, 'Gamma': False, 'Delta': False}, f
    assert not found('%s %s' % (needles[0][2], needles[1][2]), needles)['Alpha'], 'a code with no name on the line is not found'
    res = {B1M: {'needles': needles, 'run': {'control': f, 'exact': f}, 'bad': []}}
    ok, rows = report(res, ['exact'])
    assert ok and len(rows) == 8
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/docs-real/tokens.npy',
                    help='real text as int32 token ids, tiled to length')
    ap.add_argument('--tokenizer', default='/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json')
    ap.add_argument('--also-2m', action='store_true', help='also the 2,097,152 boundary (factor 32 -> 64)')
    ap.add_argument('--only-2m', action='store_true', help='only the 2,097,152 boundary')
    ap.add_argument('--gap', type=int, default=4096, help="the 'under' needle sits this far below the boundary")
    ap.add_argument('--extra', type=int, default=12288, help='tokens of the prompt past boundary - gap (crosses the boundary)')
    ap.add_argument('--n-new', type=int, default=128, help='greedy answer tokens')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='')
    ap.add_argument('--release', action='store_true', help='release the sessions at the end')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    a.arms = ['exact']
    tok = Tok(a.tokenizer)
    T = [int(x) for x in load_npy_i32(a.tokens)]
    print('token file %d ids; arms %s' % (len(T), a.arms))
    e = Engine(a.socket)
    if e.max_ctx <= B1M:
        sys.exit('needles_boundary: the engine serves %d positions: start it with --max-ctx 2097152 or more' % e.max_ctx)
    bounds = [B2M] if a.only_2m else [B1M] + ([B2M] if a.also_2m else [])
    results = {}
    for B in bounds:
        scenario(e, tok, T, B, a, results)
    ok, rows = report(results, a.arms)
    tot = {w: sum(1 for r in rows if r[3] == w and r[4]) for w in ['control'] + a.arms}
    n = len(results) * 4
    print('\nSUMMARY: %s  (found of %d: %s)' % ('PASS' if ok else 'FAIL', n, ', '.join('%s %d' % kv for kv in tot.items())))
    if a.out:
        json.dump({'results': {str(B): {'needles': r['needles'], 'found': r['run'], 'bad': r['bad']} for B, r in results.items()},
                   'ok': ok}, open(a.out, 'w'), default=list)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
