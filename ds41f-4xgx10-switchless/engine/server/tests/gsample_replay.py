#!/usr/bin/env python3
"""G3: the Gumbel sampler's truncated pick on the engine, replayed on the host, and its distribution (DSGB10_GSAMPLE).

  rows LOG --seed S [--temperature 1.0] [--top-p 0.95] [--min-rows 6000] [--control]
      LOG is a DSGB10_STEP_LOG file of ONE seeded request (DSGB10_GSAMPLE=2, DSGB10_FORCE_NTOP=20: every row carries its top 20 ids and
      log-probabilities at the temperature). For each row: the kept set is the leading tokens of the top 20 whose probabilities
      first reach top-p (rows whose top 20 hold less than top-p, or whose cut lies within 1e-3 of it, are counted and skipped), the
      noise is g(seed, position, id) of the sampler (model/samp_rule.h), the expected pick the kept token with the largest
      log-probability + noise. A row is equal (the engine's token is the expected one), ambiguous (the two best scores are within
      2e-3, the log's precision) or different. Without --control, different must be 0 and at least --min-rows rows must have
      been checked. With --control (a log of a DSGB10_GSAMPLE=0 run: the walk of one uniform) the replay must report many
      different rows (at least a fifth of those checked): the check can see the sampler it tests.
  counts --socket S --tokens T.npy [--n 7200] [--len 128] [--temperature 1.0] [--top-p 0.95] [--control-temperature 0.7]
      n one-token unseeded replies to one prompt (DSGB10_GSAMPLE=1): the counts of each token against the kept probabilities,
      renormalised (chi-square, p > 0.001; a token outside the kept set fails). Control: the same counts against the kept
      probabilities of --control-temperature, which must be rejected (p < 0.001).
  selftest
      the replay's own check on rows made here: Gumbel picks give 0 different rows, walk picks many.

Exit 0 when every check holds.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

M64 = (1 << 64) - 1


def mix64(x):
    x &= M64
    x ^= x >> 31
    x = (x * 0xBF58476D1CE4E5B9) & M64
    x ^= x >> 29
    x = (x * 0x94D049BB133111EB) & M64
    x ^= x >> 32
    return x


def unit(h):
    # ((float)(h >> 40) + 0.5f) * (1 / 16777216): float32 arithmetic, as the device's
    return float((np.float32(h >> 40) + np.float32(0.5)) * np.float32(1.0 / 16777216.0))


def gumbel(seed, pos, tok):
    """samp_rule.h's gumbel_noise(seed, pos, id)"""
    base = mix64(seed) ^ (((pos & 0xFFFFFFFF) * 0x9E3779B97F4A7C15) & M64)
    u = unit(mix64((base + tok * 0xD1B54A32D192ED03) & M64))
    return -math.log(-math.log(u))


def kept_of(tl, top_p):
    """the leading tokens of the top-n (log-probabilities tl, descending) whose probabilities first reach top_p: (keep, edge)
    or None when the top n hold less than top_p or the cut lies within 1e-3 of it"""
    cum = 0.0
    for i, x in enumerate(tl):
        cum += math.exp(x)
        if cum >= top_p:
            edge = min(cum - top_p, top_p - (cum - math.exp(x)))
            return None if edge < 1e-3 else i + 1
    return None


def classify(seed, pos, ids, tl, top_p, got):
    keep = kept_of(tl, top_p)
    if keep is None:
        return 'skipped'
    sc = sorted(((tl[i] + gumbel(seed, pos, ids[i]), ids[i]) for i in range(keep)), reverse=True)
    if got not in ids[:keep]:
        return 'different'
    if sc[0][1] == got:
        return 'equal'
    if keep > 1 and sc[0][0] - [s for s, t in sc if t == got][0] <= 2e-3:
        return 'ambiguous'
    return 'different'


def replay(log, seed, top_p, min_rows=0, control=False):
    import json
    n = {'equal': 0, 'ambiguous': 0, 'different': 0, 'skipped': 0}
    for line in open(log):
        r = json.loads(line)
        if 'ti' not in r:
            continue
        for j in range(r['R']):
            n[classify(seed, r['p'] + j, r['ti'][j], r['tl'][j], top_p, r['out'][j])] += 1
    checked = n['equal'] + n['ambiguous'] + n['different']
    print('rows: %d checked (%d equal, %d ambiguous, %d different), %d skipped' % (checked, n['equal'], n['ambiguous'], n['different'], n['skipped']))
    if control:
        ok = checked > 0 and n['different'] >= checked / 5
        print('control: %s (different %.1f%% of the rows checked, at least 20%% wanted)' % ('SEES THE SAMPLER' if ok else 'BLIND', 100.0 * n['different'] / max(checked, 1)))
        return ok
    ok = n['different'] == 0 and checked >= min_rows
    print('replay: %s (different must be 0; at least %d rows must be checked)' % ('PASS' if ok else 'FAIL', min_rows))
    return ok


def gamma_q(a, x):
    """the regularised upper incomplete gamma Q(a, x)"""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > 1e-300 else 1e-300
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi_square(counts, probs):
    """counts and probabilities of the kept tokens (probs sum to 1): the p-value, small expected counts pooled"""
    n = sum(counts)
    chi, bins, pe, po = 0.0, 0, 0.0, 0
    for c, p in zip(counts, probs):
        e = p * n
        if e < 5:
            pe += e
            po += c
        else:
            chi += (c - e) ** 2 / e
            bins += 1
    if pe >= 5:
        chi += (po - pe) ** 2 / pe
        bins += 1
    return gamma_q((bins - 1) / 2.0, chi / 2.0) if bins >= 2 else 1.0


def kept_probs(tl, top_p, temperature_scale=1.0):
    """kept token indices and their renormalised probabilities; tl at the request's temperature, scaled by temperature_scale to
    ask what another temperature (1 / T' over 1 / T) would keep and draw"""
    x = [v * temperature_scale for v in tl]
    z = max(x)
    p = [math.exp(v - z) for v in x]
    s = sum(p)
    p = [v / s for v in p]
    cum, keep = 0.0, len(p)
    for i, v in enumerate(p):
        cum += v
        if cum >= top_p:
            keep = i + 1
            break
    t = sum(p[:keep])
    return keep, [v / t for v in p[:keep]]


def counts_main(a):
    from serve_test import Engine, load_npy_i32
    toks = load_npy_i32(a.tokens)[:a.len]
    e = Engine(a.socket)
    base = 0x6773000000000000 | (int(time.time() * 1000) & 0xffffffffff) << 8
    first = e.run(base, toks, max_new=1, n_top=20, temp=a.temperature, top_p=a.top_p, keep=False)
    ids = [t for t, _ in first['top'][0]]
    tl = [v for _, v in first['top'][0]]
    # the engine's log-probabilities of the top 20 are the full vocabulary's at the request's temperature
    keep, probs = kept_probs(tl, a.top_p)
    counts = {}
    for i in range(a.n):
        o = e.run(base + 1 + i, toks, max_new=1, temp=a.temperature, top_p=a.top_p, keep=False)
        counts[o['ids'][0]] = counts.get(o['ids'][0], 0) + 1
    outside = sum(c for t, c in counts.items() if t not in ids[:keep])
    c = [counts.get(t, 0) for t in ids[:keep]]
    p = chi_square(c, probs)
    ok = outside == 0 and p > 1e-3
    print('counts: %d replies, kept %d tokens, %d outside the kept set, chi-square p %.4g: %s' % (a.n, keep, outside, p, 'PASS' if ok else 'FAIL'))
    ck, cp = kept_probs(tl, a.top_p, a.temperature / a.control_temperature)
    pc = chi_square([counts.get(t, 0) for t in ids[:ck]], cp) if ck == keep else 0.0
    okc = pc < 1e-3
    print('control: the same counts against temperature %g: p %.4g: %s' % (a.control_temperature, pc, 'REJECTED' if okc else 'NOT REJECTED (blind)'))
    return ok and okc


def selftest():
    import json
    import random
    import tempfile
    rng = random.Random(7)
    seed, top_p = 12345, 0.95
    rows_g, rows_w = [], []
    for step in range(400):
        for log, pick in ((rows_g, 'gumbel'), (rows_w, 'walk')):
            R = 3
            r = {'i': step, 'p': 1000 + 3 * step, 'R': R, 'out': [], 'ti': [], 'tl': []}
            for j in range(R):
                ids = rng.sample(range(129280), 20)
                z = sorted((rng.gauss(0, 1.2) for _ in range(20)), reverse=True)
                m = math.log(sum(math.exp(v) for v in z)) + rng.uniform(0, 0.02)
                tl = [round(v - m, 3) for v in z]
                keep = kept_of(tl, top_p)
                if keep is None:
                    keep = 5
                if pick == 'gumbel':
                    got = max(range(keep), key=lambda i: tl[i] + gumbel(seed, r['p'] + j, ids[i]))
                else:
                    u = (mix64(mix64(seed ^ 0x5851F42D4C957F2D) + r['p'] + j) >> 40) / 16777216.0
                    tot = sum(math.exp(tl[i]) for i in range(keep))
                    acc, got = 0.0, keep - 1
                    for i in range(keep):
                        acc += math.exp(tl[i])
                        if acc > u * tot:
                            got = i
                            break
                r['out'].append(ids[got])
                r['ti'].append(ids)
                r['tl'].append(tl)
            log.append(r)
    ok = True
    for name, rows, control in (('gumbel', rows_g, False), ('walk', rows_w, True)):
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        print('selftest %s picks:' % name)
        ok = replay(f.name, seed, top_p, min_rows=0 if control else 800, control=control) and ok
        os.unlink(f.name)
    # the chi-square: a draw from the probabilities passes, a draw from other ones does not
    probs = [0.5, 0.25, 0.15, 0.1]
    draws = [rng.choices(range(4), probs)[0] for _ in range(20000)]
    c = [draws.count(i) for i in range(4)]
    p_ok, p_bad = chi_square(c, probs), chi_square(c, [0.4, 0.3, 0.2, 0.1])
    good = p_ok > 1e-3 and p_bad < 1e-3
    print('selftest chi-square: p of the true probabilities %.3g (must be above 0.001), of wrong ones %.3g (below): %s' % (p_ok, p_bad, 'PASS' if good else 'FAIL'))
    return ok and good


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('rows')
    r.add_argument('log')
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--temperature', type=float, default=1.0)
    r.add_argument('--top-p', type=float, default=0.95)
    r.add_argument('--min-rows', type=int, default=6000)
    r.add_argument('--control', action='store_true')
    c = sub.add_parser('counts')
    c.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    c.add_argument('--tokens', required=True)
    c.add_argument('--n', type=int, default=7200)
    c.add_argument('--len', type=int, default=128)
    c.add_argument('--temperature', type=float, default=1.0)
    c.add_argument('--top-p', type=float, default=0.95)
    c.add_argument('--control-temperature', type=float, default=0.7)
    sub.add_parser('selftest')
    a = ap.parse_args()
    if a.cmd == 'rows':
        ok = replay(a.log, a.seed, a.top_p, a.min_rows, a.control)
    elif a.cmd == 'counts':
        ok = counts_main(a)
    else:
        ok = selftest()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
