#!/usr/bin/env python3
"""The session store's disk tier (docs/engine.md, Serving), through rank 0's socket (serve_test.py's
client). A session's turns, each after a filler request in another session; the same requests run on
two servers:
  - one whose store keeps everything (the default store, --disk-gib 0): the session waits in the store;
  - one whose pool holds one long session at a time (--pool-tokens 110000), with a disk
    tier: the filler's request spills the session, and its next turn reads it back.
--out writes each request's tokens, logprobs and restart point; --compare A B checks them equal, bit
for bit: a session back from disk continues exactly as if it had stayed. The turns: a long one (CED
replays the decoder), a short one (an exact continuation), a regenerate, a divergence far back (a window
snapshot), and a fork of the spilled session. The divergence and the fork restart inside the prompt
and add 300 positions, so they are also checked against the same prompt prefilled cold (CED: bit for
bit) in the same run. The +300 turn is compared with cold too, for information only: its history holds
decoded positions, whose roundings a cold prefill does not reproduce (docs/engine.md, Exact reuse).

  python3 engine/server/tests/disk_test.py --out store.json     # against a store-only server
  python3 engine/server/tests/disk_test.py --out disk.json      # against the spilling server
  python3 engine/server/tests/disk_test.py --compare store.json disk.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_proof import END_NUMBERS, SNAP_NUMBERS, Log, NotReached, back_re, need, reservation, spill_re  # noqa: E402
from serve_test import Engine, load_npy_i32, same, show  # noqa: E402
from stats_test import stats  # noqa: E402


SNAP_EVERY = 4096    # --snap-every of both servers
FILLER_NUMBERS = 4   # the filler line: 2,000 positions, 1 + 1 + 1 + 1 numbers


class Proof:
    """The premise of the compare arm, read from rank 0's log: what each compared turn must have been preceded and followed by."""

    def __init__(self, log):
        self.log = log
        self.mark = None  # where the log stood when the conversation's last turn ended
        self.hist = {}    # session -> positions its history holds
        self.whole = 0    # compared turns preceded by a whole spill of their conversation

    def after_turn(self, session, o, prompt_len):
        self.hist[session] = prompt_len + len(o['ids'])
        self.mark = self.log.mark()

    def before(self, name, key, expect_continuation):
        """Called after the turn: the conversation's pieces went to the disk since its last turn and came back whole: its read-back
        line, something read (the store's line before it), restart point N of N, every checksum equal. In a small pool a stored
        conversation leaves piece by piece during the next request's prefill, and the end-of-request spill skips one already
        partly on the disk, so a whole spill ("serve: spill session ...") is counted here, not required; done() requires one."""
        m = re.search(spill_re(key), self.log.text(self.mark), re.M)
        if m and int(m.group(2)) == self.hist[key]:
            self.whole += 1
            print('  log: %s (whole)' % m.group(0).strip())
        b = self.log.wait_for(back_re(key), 'the read-back before "%s"' % name, since=self.mark)
        text = self.log.text(self.mark)
        reads = re.findall(r'state: session \d+ back from disk: ([\d.]+) MiB in', text[:text.find(b.group(0))])
        need(bool(reads) and float(reads[-1]) > 0, '"%s": pieces of the conversation came back from the disk (%s MiB read)' % (
            name, reads[-1] if reads else 'nothing'))
        need(int(b.group(4)) == int(b.group(5)), '"%s": every piece came back (restart point %s of %s)' % (name, b.group(4), b.group(5)))
        need(not re.search(r'state: session \d+: ERROR', text), '"%s": every piece read back matched its checksum' % name)
        return int(b.group(5))

    def done(self):
        need(self.whole >= 1, 'a whole spill came before at least one compared turn (%d)' % self.whole)


def run(e, T, n, proof):
    S = 0x5d15000000000000 + (n << 20)
    F = S + 0x80000
    k = [0]
    res, rec = [], {}

    def filler():  # another session's short request: the server parks (and may spill) the one before
        k[0] += 1
        f = e.run(F + k[0], T[200000 + 100 * k[0]:202000 + 100 * k[0]], max_new=2)
        e.release(F + k[0])
        return f

    def turn(name, sid, prompt, cold_check, spilled=None, continuation=False, **kw):
        filler()
        before = proof.hist.get(spilled) if proof and spilled else None
        o = e.run(sid, prompt, max_new=24, n_top=2, **kw)
        show(name, o)
        if o['error']:
            raise NotReached('"%s": the engine answered with an error: %s' % (name, o['error']))
        if proof:
            if spilled:
                proof.before(name, spilled, continuation)
                if continuation:
                    need(o['cached'] == before, '"%s" restarted at the history\'s end: cached %d, history %d' % (name, o['cached'], before))
            if sid == A:
                proof.after_turn(A, o, len(prompt))
        rec[name] = {'ids': o['ids'], 'lp': o['lp'], 'cached': o['cached'], 'prompt': o['prompt']}
        if cold_check:
            c = e.run(S + 0xfff, prompt, max_new=24, n_top=2, keep=False)
            eq = same(name + ' vs cold' + (' (information)' if cold_check == 'info' else ''), o, c)
            if cold_check != 'info':
                res.append(eq)
        return o

    A = S + 1
    t1 = turn('cold %d' % n, A, T[:n], False)
    h = T[:n] + list(t1['ids'])
    p2 = h + T[n + 5000:n + 5300]
    t2 = turn('+300 (CED)', A, p2, 'info', spilled=A, continuation=True)
    p3 = p2 + list(t2['ids']) + T[n + 7000:n + 7040]
    turn('+40 (exact)', A, p3, False, spilled=A, continuation=True)
    turn('regenerate', A, p3, False, spilled=A)
    q = n - 8000
    turn('diverge %d back' % (n - q), A, T[:q] + T[n + 9000:n + 9300], True, spilled=A)
    q = n - 15000 if n > 20000 else n // 2
    turn('fork at %d' % q, A + 1, T[:q] + T[n + 11000:n + 11300], True, spilled=A, base=A)
    e.release(A)
    e.release(A + 1)
    return res, rec


def compare(files):
    x, y = (json.load(open(f)) for f in files)
    ok = 0
    for name in x:
        eq = name in y and x[name] == y[name]
        ok += eq
        print('  %-34s %s' % (name, 'EQUAL' if eq else 'DIFFER: %s' % (
            {k: (x[name][k][:6], y.get(name, {}).get(k, [])[:6]) if isinstance(x[name][k], list) else (x[name][k], y.get(name, {}).get(k))
             for k in x[name] if y.get(name, {}).get(k) != x[name][k]})))
    print('disk_test --compare: %d of %d requests equal' % (ok, len(x)))
    return 0 if ok == len(x) == len(y) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--n', default='20000,64000', help='session lengths')
    ap.add_argument('--out', default='')
    ap.add_argument('--compare', nargs=2, default=None)
    ap.add_argument('--arm', choices=('compare', 'tight'), default='compare')
    ap.add_argument('--log', default='', help="rank 0's log from the engine's start: the premise's proof, needed on the spilling server")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    proof = None
    try:
        if a.arm == 'compare':
            s = stats(e)[2]
            if s['disk_cap_bytes'] > 0:
                if not a.log:
                    raise NotReached('the spilling server needs --log (the premise is proved from rank 0\'s log)')
                need(s['lines'] == 1, 'the engine runs with 1 line, so the filler takes the conversation\'s line (%d)' % s['lines'])
                # serve.cpp serve(): cap = m.state_ceiling() (model state.cu: the pool's takeable numbers x chunk_image; STATS carries
                # them as pool_numbers x number_bytes, which is that value for a store with no disk-only tier), room = max(256 MiB,
                # cap / 16), budget = cap > room ? cap - room : 0; evict() spills while state_bytes() > budget. The store keeps nothing
                # between requests when the budget is under one number.
                cap = s['pool_numbers'] * s['number_bytes']
                budget = cap - max(256 * 2**20, cap // 16) if cap > max(256 * 2**20, cap // 16) else 0
                need(budget < s['number_bytes'],
                     'the pool is %d numbers = %.1f MiB: the store keeps %d B between requests (cap %d - max(256 MiB, cap / 16)), under one '
                     'number (%d B), so no stored conversation stays' % (s['pool_numbers'], cap / 2**20, budget, cap, s['number_bytes']))
                longest = max(int(x) for x in a.n.split(','))
                snaps = (longest - 1) // SNAP_EVERY + 1  # the prefill's, at every --snap-every, and one at the prompt's end
                floor = reservation(longest + 436) + SNAP_NUMBERS * snaps + END_NUMBERS + FILLER_NUMBERS
                need(s['pool_numbers'] >= floor,
                     'the pool holds the longest conversation whole with its bookmarks beside the filler: %d numbers >= %d '
                     '(reservation %d + %d bookmarks of %d + end %d + filler %d)' % (
                         s['pool_numbers'], floor, reservation(longest + 436), snaps, SNAP_NUMBERS, END_NUMBERS, FILLER_NUMBERS))
                proof = Proof(Log(a.log))
            else:
                print('  (a store-only server: nothing to prove, the requests are recorded)')
        res, rec = [], {}
        for n in (int(x) for x in a.n.split(',')):
            r, c = run(e, T, n, proof)
            res += r
            rec.update({'%d: %s' % (n, k): v for k, v in c.items()})
        if proof:
            proof.done()
    except NotReached as x:
        print('disk_test: the test did not reach the path: %s' % x)
        return 2
    if a.out:
        json.dump(rec, open(a.out, 'w'))
    print('disk_test: %d of %d checks against cold pass; %d requests recorded' % (sum(res), len(res), len(rec)))
    return 0 if all(res) else 1


if __name__ == '__main__':
    sys.exit(main())
