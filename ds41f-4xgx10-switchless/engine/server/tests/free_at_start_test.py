#!/usr/bin/env python3
"""Memory left to the host: on every Spark MemAvailable stays at the memory the engine was told to leave free
(--pool-free-gib), within the instrument's own spread, whatever the lines outside the store do up to their caps. The instrument is
NOT the engine: /proc/meminfo read over ssh on every box every 10 s (the engine's own mem_avail_min_bytes is printed beside it for
comparison and checked for no new lows, but a pass or a fail is the independent reading's).

  free_at_start_test.py --socket S --tokens T.npy [--nodes local,USER@HOST1,USER@HOST2,USER@HOST3 | --fleet FILE] CASE ...

  Nodes: rank 0's Spark is `local` (it cannot ssh to itself: run this test ON rank 0's box); the workers as user@host, in rank order.
  Without --nodes they are read from the fleet file (--fleet, default ~/dsgb10/bin/fleet.conf): `local`, then RANK1_SSH, RANK2_SSH, RANK3_SSH.

Cases (each exits 0 on pass, 1 on a failure, 2 when the run did not reach what it tests):
  null    idle for --seconds (default 600) with the engine up: the readings' spread per box (max - min). Writes --null-file
          (default free-at-start-null.json). Run this FIRST: every other case takes its spread from it (or from --spread), and a null
          that never moves would make the other cases' passes meaningless.
  bench   the ordinary set: --rounds (3) rounds of 8 requests of 16K tokens decoding together, new conversations each round.
  ids     the server's token ids driven to their cap: --ids-sessions (40) conversations of --ids-len (262,144) tokens, kept. Needs
          the engine started with --ids-host-mib 32 (the real 1,024 MiB is 268M positions: hours of prefill). Proves: STATS
          ids_file_sessions > 0 and ids_host_bytes within the cap plus one conversation, then the FIRST conversation continued
          and its cached tokens reach back over the whole prefix (its ids came back from the file; a lost history would recompute).
  ids (continued)  The first conversation is continued after its ids went to the file; its reply (200 greedy tokens) and cached count must equal the
          same continuation on a CONTROL start with the cap off: start the engine with a large --ids-host-mib (4096), run `ids --ids-control C.json`;
          then start with --ids-host-mib 32 and run `ids --ids-ref C.json` (it also checks the engine's ids-file count fell by one while the
          continuation ran).
  frame   request frames of the cap: 2 requests of 36 x 1,100 image rows (387 MiB, under the default --frame-stage-mib 400).
  heap    the glibc heap by a checkpoint of 64 conversations (--heap-sessions, 20K tokens each), then --ssd-flush-s + 60 s idle
          for the checkpoint to be written.
  front   dsfront's table past its budget: --front URL --key K posts --front-sessions distinct long conversations through dsfront,
          which owns the engine's one connection: run this case ALONE (no other case, no STATS), with dsfront started with the
          release's PICO_FRONT_MIB.
  all     null (if --null-file is missing), bench, ids, frame, heap in turn, the verdict over the whole run ("the sum at the end").

Verdict per case: every box's lowest reading >= (--keep-gib - spread). A reading under it exits 1 with the box, the time and the value.
The sampler also prints each box's lowest reading, its end value and its start value.

Engine settings (the release's, with the three that make the cases reachable; every start the double reset and the quiet lock):
  start as production does (the service's prepare and start) with ENGINE_ARGS holding --pool-free-gib 10, PICO_FRONT_MIB set
  (a --pool-free-gib start is refused without --front-mib), and for the ids case --ids-host-mib 32; for the frame case the default
  --frame-stage-mib 400. The engine's start line "store: ..." gives the numbers this test checks the readings against.
  python3 engine/server/tests/free_at_start_test.py --socket ~/dsgb10/run/engine.sock --tokens TOKENS.npy \
      --keep-gib 10 null
  python3 engine/server/tests/free_at_start_test.py ... all
The null control's own value sets the spread every other case uses: a case run with --spread 0 and no null file is refused.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import struct  # noqa: E402
from image_store_test import Conn, S0 as IMG_S0, big, collect as img_collect, NotReached, need  # noqa: E402
from serve_test import DONE, TOKENS, Engine, load_npy_i32  # noqa: E402
from stats_test import stats  # noqa: E402

fails = 0
SESS = 0x70726f6d00000000  # 'prom'


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


class Sampler:
    """/proc/meminfo MemAvailable of every node every `every` seconds, GiB, on its own thread. 'local' reads this box's file."""

    def __init__(self, nodes, every=10.0):
        self.nodes, self.every = nodes, every
        self.rows = {n: [] for n in nodes}
        self.stop_ev = threading.Event()
        self.th = None

    def read(self, n):
        cmd = ['grep', 'MemAvailable', '/proc/meminfo'] if n == 'local' else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', n, 'grep MemAvailable /proc/meminfo']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return int(r.stdout.split()[1]) / 1048576.0

    def once(self):
        t = time.time()
        for n in self.nodes:
            try:
                self.rows[n].append((t, self.read(n)))
            except Exception as x:  # a missed reading is not a reading: say so, never count it as one
                print('  sampler: %s unreadable (%s)' % (n, x))

    def loop(self):
        while not self.stop_ev.is_set():
            self.once()
            self.stop_ev.wait(self.every)

    def start(self):
        self.stop_ev.clear()
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()

    def stop(self):
        self.stop_ev.set()
        self.th.join()
        self.once()  # the end value


def spread_of(s):
    out = {}
    for n, r in s.rows.items():
        v = [x for _, x in r]
        out[n] = (max(v) - min(v)) if v else None
    return out


def verdict(name, s, keep, spread):
    need(all(len(r) >= 2 for r in s.rows.values()), '%s: every box was read at least twice' % name)
    print('  %s readings (GiB, MemAvailable over ssh): box lowest / start / end' % name)
    low = keep - spread
    for n, r in s.rows.items():
        v = [x for _, x in r]
        print('    %-24s %.2f / %.2f / %.2f' % (n, min(v), v[0], v[-1]))
        t_bad = [(t, x) for t, x in r if x < low]
        check(not t_bad, '%s %s: no reading under keep - spread = %.2f - %.2f = %.2f GiB%s' % (
            name, n, keep, spread, low, '' if not t_bad else ' (lowest %.2f at +%.0f s)' % (min(x for _, x in t_bad), t_bad[0][0] - r[0][0])))


def stats_keep(e):
    """One STATS on this connection while a request runs: the request's own frames that arrive first are kept, in order, for run_many."""
    from stats_test import NAMES, STATS_REQ, STATS
    time.sleep(0.6)
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = Engine.frame(e)  # the socket, not the kept queue (Conn.numbers does the same)
        if t == STATS:
            n = struct.unpack('<II', p[:8])[1]
            return dict(zip(NAMES, struct.unpack('<%dQ' % n, p[8:8 + 8 * n])))
        e.q.append((t, p))


def first_tokens(e, rid):
    """Read frames until the first TOKENS frame of request `rid` has arrived; every frame read is kept, in order, for run_many. The request is
    admitted by then: its conversation's ids are back in memory, and flush_idle cannot run while it flies."""
    while True:
        t, p = Engine.frame(e)
        e.q.append((t, p))
        if t == TOKENS and struct.unpack('<Q', p[:8])[0] == rid:
            return


def continuation(e, session, prompt, max_new=200):
    """The ids-file count before the submit, mid-flight (after the first TOKENS frame; the STATS reply is read among the TOKENS frames that keep
    arriving, which are kept) and after DONE; with the request's outcome."""
    before = stats(e)[2]['ids_file_sessions']
    sub = e.submit(session, prompt, max_new=max_new, keep=True, stop=())
    first_tokens(e, sub[0])
    mid = stats_keep(e)['ids_file_sessions']
    o = list(run_many(e, [sub]).values())[0]
    after = stats(e)[2]['ids_file_sessions']
    print('  ids files: before submit %d, mid-flight %d, after DONE %d' % (before, mid, after))
    return o, before, mid, after


def engine_view(e):
    s = stats(e)[2]
    print('  engine (its own reading, not the instrument): store_numbers %d, mem_avail_min %.2f GiB, new lows %d, ids in memory %.1f MiB, ids files %d' % (
        s['store_numbers'], s['mem_avail_min_bytes'] / 2 ** 30, s['mem_avail_low_count'], s['ids_host_bytes'] / 2 ** 20, s['ids_file_sessions']))
    return s


def run_many(e, subs):
    """submit(...) returns (rid, n_top): run them all together, return {rid: out}"""
    pending = {r[0] for r in subs}
    out = {r: {'ids': [], 'cached': 0, 'reason': None, 'error': ''} for r in pending}
    while pending:
        t, p = e.frame()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            if r in out:
                out[r]['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
        elif t == DONE:
            (r, reason, pt, cached, comp, q, pf, d, el, _) = struct.unpack('<QIIIIQQQII', p[:56])
            if r in out:
                out[r].update(reason=reason, cached=cached, prompt=pt, error=p[len(p) - el:].decode())
                pending.discard(r)
    return out


# ---- the cases -------------------------------------------------------------------------------------------------------------

def case_null(a, e, T, s):
    print('null control: the engine idle for %d s' % a.seconds)
    time.sleep(a.seconds)


def case_bench(a, e, T, s):
    for rd in range(a.rounds):
        subs = [e.submit(SESS + 100 * rd + i, T[i * 17000:i * 17000 + 16000], max_new=256, keep=False, stop=()) for i in range(8)]
        out = run_many(e, subs)
        bad = [o for o in out.values() if o['error']]
        check(not bad, 'bench round %d: 8 requests of 16K ran (%s)' % (rd + 1, bad[0]['error'] if bad else 'no errors'))


def case_ids(a, e, T, s):
    s0 = stats(e)[2]
    need(s0['ids_host_bytes'] >= 0 and 'ids_file_sessions' in s0, 'the engine reports the ids fields: it is a build with the ids file')
    control = bool(a.ids_control)  # the control start: the cap off (--ids-host-mib large), nothing is expected out of memory
    cap = a.ids_host_mib * 2 ** 20
    L = a.ids_len
    for i in range(a.ids_sessions):
        out = run_many(e, [e.submit(SESS + 0x1000 + i, T[(i * 997) % 5000:(i * 997) % 5000 + L], max_new=4, keep=True, stop=())])
        o = list(out.values())[0]
        if o['error']:
            check(False, 'ids: conversation %d of %d (%d tokens) ran: %s' % (i + 1, a.ids_sessions, L, o['error']))
        if i % 8 == 7:
            time.sleep(35)  # flush_idle runs while nothing does: the conversations' ids go out between requests
            engine_view(e)
    time.sleep(40)
    v = engine_view(e)
    if not control:
        need(a.ids_sessions * L * 4 > cap, 'the conversations hold %.0f MiB of ids, over the cap %d MiB' % (a.ids_sessions * L * 4 / 2 ** 20, a.ids_host_mib))
        check(v['ids_file_sessions'] > 0, 'conversations\' ids are in files (%d)' % v['ids_file_sessions'])
        check(v['ids_host_bytes'] <= cap + L * 4 * 2, 'ids in memory %.1f MiB: within the cap %d MiB plus the conversations on lines' % (v['ids_host_bytes'] / 2 ** 20, a.ids_host_mib))
    else:
        check(v['ids_file_sessions'] == 0, 'control: no ids are in files (%d)' % v['ids_file_sessions'])
    # the first conversation (the least recently used: its ids went out first) continues AFTER its ids went to the file. Greedy, 200 new tokens.
    # Once its first tokens have arrived (flush_idle cannot run while it flies), the engine's count of conversations with ids in a file is one lower than before it: its file was read
    # back. Its reply and the tokens reused are the same as the continuation on a control start with the cap off (--ids-control writes them there,
    # --ids-ref compares them here).
    prompt = T[0:L] + T[L:L + 64]
    o, f0, mid, after = continuation(e, SESS + 0x1000, prompt)
    check(not o['error'], 'the first conversation continued without an error')
    if not control:
        check(mid == f0 - 1, 'while it ran, ids files %d -> %d: its ids came back from the file' % (f0, mid))
    check(o['cached'] >= L - 256, 'its %d tokens were reused (cached %d)' % (L, o['cached']))
    rec = {'ids': list(o['ids']), 'cached': o['cached'], 'ids_len': L, 'sessions': a.ids_sessions}
    if control:
        json.dump(rec, open(a.ids_control, 'w'))
        print('  control: %d tokens, cached %d, written to %s' % (len(rec['ids']), rec['cached'], a.ids_control))
    if a.ids_ref:
        ref = json.load(open(a.ids_ref))
        need(ref['ids_len'] == L and ref['sessions'] == a.ids_sessions, 'the control ran the same conversations')
        check(rec['ids'] == ref['ids'], 'the reply equals the control\'s (cap off): %d tokens' % len(rec['ids']))
        check(rec['cached'] == ref['cached'], 'the tokens reused equal the control\'s: %d against %d' % (rec['cached'], ref['cached']))


def case_frame(a, e, T, s):
    for k in range(2):
        req, rows = big(e, T, IMG_S0 + 0x500 + k, 36, max_new=8)
        o = img_collect(e, req)
        check(not o['error'] and o['reason'] in ('stop', 'length'), 'frame %d of %d rows (%.0f MiB) ran: %s %s' % (k + 1, rows, rows * 10240 / 2 ** 20, o['reason'], o['error']))


def case_heap(a, e, T, s):
    subs = []
    for i in range(a.heap_sessions):
        o = list(run_many(e, [e.submit(SESS + 0x2000 + i, T[(i * 311) % 9000:(i * 311) % 9000 + 20000], max_new=8, keep=True, stop=())]).values())[0]
        subs.append(o)
    check(not any(o['error'] for o in subs), 'heap: %d conversations of 20K tokens ran' % a.heap_sessions)
    print('  idle %d s for the checkpoint' % (a.ssd_flush_s + 60))
    time.sleep(a.ssd_flush_s + 60)


def case_front(a, e, T, s):
    import urllib.request
    need(bool(a.front), '--front URL is given')
    words = ' '.join('w%d' % i for i in range(40000))
    for i in range(a.front_sessions):
        body = json.dumps({'messages': [{'role': 'user', 'content': 'conversation %d ' % i + words}], 'max_tokens': 4, 'stream': False}).encode()
        rq = urllib.request.Request(a.front + '/v1/chat/completions', body, {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (a.key or '')})
        try:
            urllib.request.urlopen(rq, timeout=900).read()
        except Exception as x:
            print('  front: request %d: %s' % (i, x))
    time.sleep(60)  # the governor reads every 10 s


def nodes_of(a):
    """`local` for rank 0, then the workers: --nodes as given, else RANK<r>_SSH of the fleet file (KEY=value lines)."""
    if a.nodes:
        return a.nodes.split(',')
    kv = {}
    with open(os.path.expanduser(a.fleet)) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                kv[k.strip()] = v.strip()
    workers = [kv.get('RANK%d_SSH' % r, '') for r in (1, 2, 3)]
    if not all(workers):
        sys.exit('free_at_start_test: %s names no RANK1_SSH, RANK2_SSH and RANK3_SSH: pass --nodes' % a.fleet)
    return ['local'] + workers


CASES = {'null': case_null, 'bench': case_bench, 'ids': case_ids, 'frame': case_frame, 'heap': case_heap, 'front': case_front}


def run_case(a, name, e, T, spread):
    s = Sampler(nodes_of(a), a.every)
    s.once()
    s.start()
    try:
        CASES[name](a, e, T, s)
    finally:
        s.stop()
    if name == 'null':
        sp = spread_of(s)
        print('  null spread per box (GiB): %s' % json.dumps({k: round(v, 3) for k, v in sp.items()}))
        worst = max(v for v in sp.values() if v is not None)
        json.dump({'spread_gib': worst, 'per_box': sp, 'seconds': a.seconds}, open(a.null_file, 'w'))
        print('  wrote %s: spread %.3f GiB' % (a.null_file, worst))
        need(worst > 0, 'the instrument moved at all in %d s (a flat null cannot show a drop)' % a.seconds)
        return s
    verdict(name, s, a.keep_gib, spread)
    if e is not None:
        v = engine_view(e)
        check(v['mem_avail_low_count'] == 0, '%s: the engine reported no new low under --pool-free-gib' % name)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='')
    ap.add_argument('--tokens', default='')
    ap.add_argument('--nodes', default='', help='local,user@host,... in rank order; default: read from --fleet')
    ap.add_argument('--fleet', default='~/dsgb10/bin/fleet.conf', help='the fleet file: RANK1_SSH, RANK2_SSH, RANK3_SSH name the workers')
    ap.add_argument('--keep-gib', type=float, default=10.0, dest='keep_gib')
    ap.add_argument('--spread', type=float, default=None, help='GiB; default: the null file')
    ap.add_argument('--null-file', default='free-at-start-null.json', dest='null_file')
    ap.add_argument('--seconds', type=int, default=600)
    ap.add_argument('--every', type=float, default=10.0)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--ids-sessions', type=int, default=40, dest='ids_sessions')
    ap.add_argument('--ids-len', type=int, default=262144, dest='ids_len')
    ap.add_argument('--ids-control', dest='ids_control', help='control start (cap off): write the continuation\'s reply and cached count here')
    ap.add_argument('--ids-ref', dest='ids_ref', help='capped start: compare the continuation with this control file')
    ap.add_argument('--ids-host-mib', type=int, default=32, dest='ids_host_mib', help='what the engine was started with')
    ap.add_argument('--heap-sessions', type=int, default=64, dest='heap_sessions')
    ap.add_argument('--ssd-flush-s', type=int, default=30, dest='ssd_flush_s')
    ap.add_argument('--front')
    ap.add_argument('--key')
    ap.add_argument('--front-sessions', type=int, default=120, dest='front_sessions')
    ap.add_argument('cmds', nargs='+', choices=list(CASES) + ['all'])
    a = ap.parse_args()
    cases = ['null', 'bench', 'ids', 'frame', 'heap'] if a.cmds == ['all'] else a.cmds
    e = T = None
    if cases != ['front']:
        T = [int(x) for x in load_npy_i32(a.tokens)]
        assert len(T) >= 100000, 'the token file is too short'
        e = Conn(a.socket)
        print('engine: %s, context %d' % (e.name, e.max_ctx))
    try:
        for name in cases:
            spread = a.spread
            if name != 'null':
                if spread is None and os.path.exists(a.null_file):
                    spread = json.load(open(a.null_file))['spread_gib']
                if spread is None or spread == 0:
                    print('free_at_start_test: no null control (%s) and no --spread: run the null case first' % a.null_file)
                    sys.exit(2)
            elif name == 'null' and a.cmds == ['all'] and os.path.exists(a.null_file):
                continue
            print('== case %s' % name)
            run_case(a, name, e, T, spread)
    except NotReached as x:
        print('free_at_start_test: NOT REACHED (%s)' % x)
        sys.exit(2)
    print('free_at_start_test %s: %s' % (' '.join(a.cmds), 'FAIL' if fails else 'PASS'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
