#!/usr/bin/env python3
"""The session store and the slots on one memory budget (docs/engine.md, "Slot memory, mapped on demand": the
store is a range of the slot pool's chunks). An idle machine gives its memory to the stored sessions, a busy one
to the live requests, and the engine never stops for the change.

Three phases, against one server (rank 0's socket, like room_test.py and squeeze_test.py):

 (a) idle growth. --sessions first turns of --n positions each, one after the other, then every session's second
     turn. The sessions the slots let go move into the store, which grows past what a fixed --session-gib would hold
     (the log's `pool:` lines name its blocks); every second turn must resume from RAM: cached >= history -
     --tolerance, and the log holds no `back from disk` line and no `spilled to disk` line in this phase.
 (b) busy shrink. One long request per slot, each committing the whole context (prompt + --m tokens), so that the
     slots' commitments leave less than the store holds: the store must give chunks back (`state: the store gives N
     chunks ... back to the slots`), the requests decode, and then the sessions of (a) resume exactly: cached >=
     history - --tolerance (now from the disk tier: `back from disk` lines are counted and printed); each is also
     compared with the same prompt prefilled cold, for information only (decoded positions round differently).
 (d) a session on disk resumed while live requests commit most of the pool (the restore maps chunks the request's slot
     decision must have counted): one long request per slot but one, then every session of (a) resumed one at a time
     through the same connection. Each resume must run (or wait for a slot and then run): cached >= history -
     --tolerance, no error, the engine up, and the engine's log holds no `no slot for request`, `no slot memory` and
     `do not fit the pool` line anywhere (read from the whole file at the end).
 (e) more long requests than slots (--slots + 2, time-slicing): every request keeps getting tokens (16, then 64 more each);
     a paused request whose state went to the store or the disk gets a slot back.
     The long requests carry no stop id: random-id prompts can draw one, and a request that stops inside a batch is
     only reported done when that batch's run ends.
 (c) the engine never stops: a short request after the long ones, and the socket stays open through every phase (an
     EOF names the phase and fails the run).
 Then the fragmentation gate (the panel's M8): --churn rounds of mixed turns (random sessions, random lengths,
     releases, forks) in the same engine; the last `pool:` line gives the chunks the store has mapped against the bytes
     its blocks hold. The bound expected: mapped <= 1.25 x live + 8 chunks once the idle chunks are trimmed, because the
     store's allocator is first fit from the bottom of its range (a freed block's hole is the first place the next block
     of that size goes, and every block is at least one chunk: a hole smaller than a chunk cannot keep a chunk mapped
     alone, so the waste is at most one chunk's worth per hole between blocks, and in practice per few blocks).
     The worst case by that argument is 1.6; the gate fails above --frag-limit (default 1.5, on mapped less idle) and
     prints the ratio either way.

Start the server so that the store's growth shows and the long requests squeeze it (the chunks, from the engine's
log: `model: slot memory: N chunks` and `state: the store shares the slot memory: ... in all`):

  dsgb10_rank --mode serve --max-ctx 131072 --slots 4 --big-slots 0 --slot-gib 0.5 --session-gib 0.25 --disk-gib 4 ... \\
      > engine.log
  python3 engine/server/tests/budget_test.py --log engine.log   (required: the sharing checks read the log; without it the
                                                                 test fails at once, it does not skip them)
  python3 engine/server/tests/budget_test.py --log engine.log [--socket S] [--tokens tokens.npy] \\
      [--session-gib 0.25] [--sessions 8] [--n 16000] [--churn 60]

That pool is 256 + 128 = 384 chunks (0.75 GiB). A first turn of 16,000 positions leaves ~53.6 MiB in the store (pages,
window snapshots, the end snapshot), so 8 sessions hold ~430 MiB (12 would not fit beside the slots' commitments and the second turns would thrash the disk tier), more than the 0.25 GiB a fixed store would be, and
well inside the pool beside the four slots; four long requests commit 71 chunks each (the context's worth: 284 of 384), leaving
100 chunks, less than the stored sessions hold; so the idle sessions leave the store for the disk tier. Other sizes work: --sessions and --n set how much the store grows, --slots (the server's)
how many long requests run. --log is the file the engine's stdout goes to (it is line buffered). The phase (d) case
for a tighter pool: --slot-gib 0.25 --session-gib 0.25 (256 chunks), the same client flags.
"""
import argparse
import os
import random
import re
import select
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from room_test import Flights, chunks_for  # noqa: E402
from serve_test import Engine, compare, load_npy_i32, show  # noqa: E402

POOL = re.compile(r'serve: req (\d+) pool: store (\d+) chunks mapped \(([\d.]+) MiB, (\d+) idle\) under ([\d.]+) MiB of '
                  r'blocks; slots committed (\d+) of (\d+) chunks; the store has given back (\d+) chunks in (\d+) times')


class Log:
    """The engine's log file, read in slices: what was written since the last call."""

    def __init__(self, path):
        self.path = path
        self.at = os.path.getsize(path) if path else 0

    def since(self):
        if not self.path:
            return ''
        time.sleep(1.5)  # a request's DONE goes out before its last log lines are written
        with open(self.path, 'rb') as f:
            f.seek(self.at)
            d = f.read()
        self.at += len(d)
        return d.decode(errors='replace')

    def whole(self):
        if not self.path:
            return ''
        with open(self.path, 'rb') as f:
            return f.read().decode(errors='replace')


class Stalled(Exception):
    pass


def pump(F, until, what, limit=300):
    """Flights.pump with progress (each request's token count every 10 s, flushed) and a deadline: past `limit` seconds
    the counts are printed and Stalled is raised instead of waiting for ever."""
    t0 = last = time.time()
    once = [0]

    def one():
        once[0] += 1
        return once[0] > 1

    def counts():
        return {rid & 0xff: (len(o['ids']), 'done' if o['done'] else '') for rid, o in F.out.items()}

    while not until():
        now = time.time()
        if now - t0 > limit:
            print('STALLED in %s after %d s: tokens per request %s' % (what, limit, counts()), flush=True)
            raise Stalled(what)
        if now - last >= 10:
            print('  %s: %d s, tokens per request %s' % (what, now - t0, counts()), flush=True)
            last = now
        if not F.e.buf and not select.select([F.e.s], [], [], 1.0)[0]:
            continue
        once[0] = 0
        F.pump(one)  # exactly one frame


def pools(text):
    """The `pool:` lines of a log slice: (mapped chunks, idle chunks, live MiB, committed chunks, total chunks)."""
    return [(int(m[2]), int(m[4]), float(m[5]), int(m[6]), int(m[7])) for m in POOL.finditer(text)]


def report_pool(name, text):
    p = pools(text)
    if not p:
        print('%s: no `pool:` line in the log slice (is the store on the slot pool? --no-vmm has none)' % name)
        return None
    mapped, idle, live, committed, total = p[-1]
    print('%s: store mapped %d chunks (%.0f MiB, %d idle) under %.1f MiB of blocks; slots committed %d of %d chunks; '
          'peak of blocks %.1f MiB, peak mapped %d chunks' % (name, mapped, mapped * 2.0, idle, live, committed, total,
                                                                max(x[2] for x in p), max(x[0] for x in p)))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', default='', help='the engine\'s stdout, written to a file (rank 0)')
    ap.add_argument('--session-gib', type=float, default=0.25, help='the server\'s --session-gib: what a fixed store would hold')
    ap.add_argument('--sessions', type=int, default=8, help='sessions of (a): each stores ~53.6 MiB after a 16,000-position first turn; keep them inside the pool less the slots\' commitments')
    ap.add_argument('--n', type=int, default=16000, help='positions of a session\'s first turn')
    ap.add_argument('--tolerance', type=int, default=2048, help='positions a resume may fall short of the history')
    ap.add_argument('--m', type=int, default=0, help='positions a long request may generate (0: to the context)')
    ap.add_argument('--slots', type=int, default=0, help='long requests in (b): the server\'s --slots (0: ask nothing, use 4)')
    ap.add_argument('--churn', type=int, default=60, help='rounds of the fragmentation run (0: skip it)')
    ap.add_argument('--frag-limit', type=float, default=1.5, help='mapped (less idle) / live bytes the run may end at')
    ap.add_argument('--seed', type=int, default=20261116)
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    TT = T + T

    def seq(off, n):
        return TT[off % len(T):off % len(T) + n]

    if not a.log or not os.path.exists(a.log):
        print('FAIL: --log FILE (the engine\'s stdout) is required: the checks that the store shares the slot memory, grows, '
              'gives chunks back and never ends in `no slot` read it; this test does not skip them')
        return 1
    log = Log(a.log)
    e = Engine(a.socket)
    ctx = e.max_ctx
    nslots = a.slots or 4
    m = a.m or max(16, ctx - 1100)
    print('engine: %s, context %d; %d long requests of %d + %d positions commit %d chunks each' %
          (e.name, ctx, nslots, 1000, m, chunks_for(1000 + m, ctx)))
    whole = log.whole()
    shared = re.search(r'the store shares the slot memory: .* (\d+) chunks \(([\d.]+) GiB\) in all', whole)
    if not shared:
        print('FAIL: the log does not say that the store shares the slot memory (is this the build with it? --no-vmm?)')
        return 1
    if shared:
        print('pool: %s chunks (%s GiB) in all; a fixed store would hold %.2f GiB' % (shared[1], shared[2], a.session_gib))
    ok = []
    S = 0x6b0d000000000000
    hist = {}
    phase = 'start'
    try:
        # ---- (a) idle growth
        phase = 'a: first turns'
        log.since()
        for i in range(a.sessions):
            sid = S + 1 + i
            p = seq(7919 * (i + 1), a.n)
            o = e.run(sid, p, max_new=4, n_top=2)
            show('session %d first turn' % i, o)
            hist[sid] = p + list(o['ids'])
            ok.append(not o['error'])
        phase = 'a: second turns'
        for i, sid in enumerate(sorted(hist)):
            p = hist[sid] + seq(12345 + 1000 * i, 16)
            o = e.run(sid, p, max_new=4, n_top=2)
            show('session %d second turn' % i, o)
            good = o['cached'] >= len(hist[sid]) - a.tolerance and not o['error']
            if not good:
                print('  FAIL: cached %d of a history of %d' % (o['cached'], len(hist[sid])))
            ok.append(good)
            hist[sid] = p + list(o['ids'])
        txt = log.since()
        p = report_pool('after idle growth', txt)
        if a.log:
            disk_in, disk_out = txt.count('spilled to disk'), txt.count('back from disk')
            print('  log: %d sessions spilled to disk, %d read back from disk, %d evicted to make room' %
                  (disk_in, disk_out, txt.count('evicted to make room')))
            ok.append(disk_in == 0 and disk_out == 0)
            if p:
                peak = max(x[2] for x in p)
                grew = peak > a.session_gib * 1024
                print('  the store held up to %.1f MiB of blocks against %.1f MiB of a fixed store: %s' %
                      (peak, a.session_gib * 1024, 'GREW PAST IT' if grew else 'FAIL: it never grew past it (raise --sessions or --n)'))
                ok.append(grew)
        else:
            print('  (no --log: the idle-growth checks on the log are skipped)')

        # ---- (b) busy shrink
        phase = 'b: long requests'
        F = Flights(e)
        ys = [F.submit(S + 0x100 + i, seq(20000 * (i + 1), 1000), m, stop=()) for i in range(nslots)]  # no stop id: until cancelled
        pump(F, lambda: all(len(F.out[y]['ids']) >= 16 or F.out[y]['done'] for y in ys), 'b: long requests')
        print('%d long requests decoding: %s' % (nslots, [len(F.out[y]['ids']) for y in ys]))
        for y in ys:
            if not F.out[y]['done']:
                F.cancel(y)
        pump(F, lambda: all(F.out[y]['done'] for y in ys), 'b: cancels')
        for y in ys:
            d = F.out[y]['done']
            ok.append(not d['error'] and len(F.out[y]['ids']) >= 16)
            print('long request %d: %s, %d tokens, error %r' % (y & 0xff, d['reason'], len(F.out[y]['ids']), d['error']))
        txt = log.since()
        report_pool('after the long requests', txt)
        if a.log:
            gives = re.findall(r'state: the store gives (\d+) chunks \(([\d.]+) MiB\) back to the slots', txt)
            print('  log: the store gave chunks back %d times (%d chunks), %d slots gave theirs to the pool' %
                  (len(gives), sum(int(g[0]) for g in gives), txt.count('gives its')))
            ok.append(len(gives) > 0)

        phase = 'b: resumes'
        outs = {}
        for i, sid in enumerate(sorted(hist)):
            p = hist[sid] + seq(54321 + 1000 * i, 16)
            o = e.run(sid, p, max_new=4, n_top=2)
            show('session %d after the squeeze' % i, o)
            good = o['cached'] >= len(hist[sid]) - a.tolerance and not o['error']
            if not good:
                print('  FAIL: cached %d of a history of %d (a restart point was lost)' % (o['cached'], len(hist[sid])))
            ok.append(good)
            outs[sid] = (p, o)
            hist[sid] = p + list(o['ids'])
        txt = log.since()
        report_pool('after the resumes', txt)
        if a.log:
            print('  log: %d sessions back from disk, %d spilled to disk' % (txt.count('back from disk'), txt.count('spilled to disk')))
        # every resumed session against the same prompt prefilled cold, for information only: these histories hold
        # decoded positions, whose roundings a cold prefill does not reproduce (docs/engine.md, Exact reuse); the exact
        # check of a session that went to disk and back is disk_test.py's A/B (bit for bit against one that stayed)
        for k, sid in enumerate(sorted(outs)):
            p, o = outs[sid]
            c = e.run(S + 0xfff0 + k, p, max_new=4, n_top=2, keep=False)
            compare('session %d vs cold (information)' % k, o, c)

        # ---- (d) a session on disk resumed while live requests commit most of the pool
        phase = 'd: resumes under load'
        G = Flights(e)
        ls = [G.submit(S + 0x200 + i, seq(30000 * (i + 1), 1000), m, stop=()) for i in range(max(1, nslots - 1))]
        pump(G, lambda: all(len(G.out[y]['ids']) >= 16 or G.out[y]['done'] for y in ls), 'd: long requests')
        print('%d long requests decoding: %s' % (len(ls), [len(G.out[y]['ids']) for y in ls]))
        for i, sid in enumerate(sorted(hist)):
            p = hist[sid] + seq(98765 + 1000 * i, 16)
            r = G.submit(sid, p, 4)
            pump(G, lambda: G.out[r]['done'], 'd: resume %d' % i)
            d = G.out[r]['done']
            good = d['cached'] >= len(hist[sid]) - a.tolerance and not d['error']
            print('session %d resumed under load: %s, cached %d of %d, error %r%s' %
                  (i, d['reason'], d['cached'], len(hist[sid]), d['error'], '' if good else '  FAIL'))
            ok.append(good)
            hist[sid] = p + list(G.out[r]['ids'])
        for y in ls:
            if not G.out[y]['done']:
                G.cancel(y)
        pump(G, lambda: all(G.out[y]['done'] for y in ls), 'd: cancels')
        txt = log.since()
        report_pool('after the resumes under load', txt)
        print('  log: %d sessions back from disk' % txt.count('back from disk'))

        # ---- (e) more long requests than slots: time-slicing takes turns, and a paused request whose state went to the
        # store (or the disk) must get its slot back: every one keeps getting tokens (none waits for ever)
        phase = 'e: more requests than slots'
        H = Flights(e)
        hs = [H.submit(S + 0x300 + i, seq(40000 + 3000 * i, 1000), m, stop=()) for i in range(nslots + 2)]
        pump(H, lambda: all(len(H.out[y]['ids']) >= 16 for y in hs), 'e: every request 16 tokens')
        first = {y: len(H.out[y]['ids']) for y in hs}
        pump(H, lambda: all(len(H.out[y]['ids']) >= first[y] + 64 for y in hs), 'e: every request 64 more')
        print('%d long requests on %d slots, tokens: %s' % (len(hs), nslots, [len(H.out[y]['ids']) for y in hs]))
        for y in hs:
            H.cancel(y)
        pump(H, lambda: all(H.out[y]['done'] for y in hs), 'e: cancels')
        ok.append(all(not H.out[y]['done']['error'] for y in hs))

        # ---- (c) the engine still answers
        phase = 'c: a short request'
        r = e.run(S + 0x9000, seq(150000, 300), max_new=8)
        show('a new session after', r)
        ok.append(not r['error'] and len(r['ids']) > 0)

        # ---- the fragmentation gate
        if a.churn > 0:
            phase = 'churn'
            rnd = random.Random(a.seed)
            live = list(sorted(hist))
            nxt = 0x2000
            for k in range(a.churn):
                what = rnd.random()
                o = None
                room = ctx - 4000  # a session that long starts over rather than overrun the context
                if what < 0.55 and live:  # a turn of an existing session, a few hundred to a few thousand positions longer
                    sid = rnd.choice(live)
                    if len(hist[sid]) < room:
                        p = hist[sid] + seq(rnd.randrange(len(T)), rnd.choice([40, 300, 2000]))
                        o = e.run(sid, p, max_new=rnd.choice([2, 8]))
                        hist[sid] = p + list(o['ids'])
                elif what < 0.8:  # a new session of random length
                    nxt += 1
                    sid = S + nxt
                    p = seq(rnd.randrange(len(T)), min(room, rnd.choice([1000, 6000, a.n, 3 * a.n])))
                    o = e.run(sid, p, max_new=4)
                    hist[sid] = p + list(o['ids'])
                    live.append(sid)
                elif what < 0.9 and live:  # a fork of an existing session
                    base = rnd.choice(live)
                    nxt += 1
                    sid = S + nxt
                    cut = max(64, len(hist[base]) - rnd.choice([100, 3000]))
                    p = hist[base][:cut] + seq(rnd.randrange(len(T)), 200)
                    o = e.collect(e.submit(sid, p, max_new=4, base=base))
                    hist[sid] = p + list(o['ids'])
                    live.append(sid)
                elif live:  # a release
                    sid = live.pop(rnd.randrange(len(live)))
                    e.release(sid)
                    hist.pop(sid, None)
                if o is not None:
                    ok.append(not o['error'])
            txt = log.since()
            p = report_pool('after the churn', txt)
            if a.log and p:
                mapped, idle, blocks, committed, total = p[-1]
                ratio = (mapped - idle) * 2.0 / max(1e-9, blocks)
                bound = 1.25 * blocks / 2.0 + 8
                print('fragmentation: %d chunks in use by blocks (%d mapped, %d idle) for %.1f MiB of blocks: x%.2f '
                      '(expected <= %.0f chunks, x1.25 + 8 chunks; the gate x%.2f)' %
                      (mapped - idle, mapped, idle, blocks, ratio, bound, a.frag_limit))
                ok.append(ratio <= a.frag_limit or blocks < 64)  # a store of a few blocks says nothing
            elif a.log:
                print('  (no `pool:` line after the churn: the gate cannot be read)')
    except Stalled as err:
        print('FAIL: phase %s stalled: an admitted request got no tokens within the deadline (see the counts above)' % err)
        return 1
    except EOFError as err:
        print('FAIL: the engine stopped in phase %s (%s): its log names the reason' % (phase, err))
        return 1
    bad = [w for w in ('no slot for request', 'no slot memory', 'do not fit the pool', 'cannot map positions')
           if w in log.whole()]
    print('log: %s' % ('no `%s` line' % '`, `'.join(('no slot for request', 'no slot memory', 'do not fit the pool'))
                       if not bad else 'FAIL: it holds %s' % bad))
    ok.append(not bad)
    print('budget_test: %d of %d checks pass' % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
