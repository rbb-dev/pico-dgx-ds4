#!/usr/bin/env python3
"""The prefill's second set of row buffers, lent by the pool (model/rows_lend.cu, server/rows_loan.h), from outside, on the four
Sparks. The engine is started by the caller (rank 0's socket, dsfront stopped) with --prefill-rows 8192 --prefill-rows-max 16384
(the loan: tenant 4, 23 groups). Every check prints ok / FAIL; the exit status is 0 only when all hold. Chunks are counted from the
engine's own PROGRESS frames (one after every prefill chunk of a request), the loan from STATS' tenant_groups[4], the pool's pressure
from a client that fills it with kept conversations. Bit equality of the replies is bit_probe.py's: run it with --len 150000 on the
8,192-row record (an engine without --prefill-rows-max) and on this engine and compare (--compare).

  Marks for the cycle / asked modes: engine-test.sh starts the engine with KEEP_GIB=0, so kept sessions go to the SSD tier and the
  pool settles near 6 groups free on a 37-group pool (--pool-tokens 7760000) instead of filling. The default marks (go back under 2
  free groups, ask at the set's 23 + 8) are then never crossed. Start the engine for these modes with
  --prefill-loan-low-groups 8 --prefill-loan-high-groups 8: the fill crosses 8 free, the release gives about 32.7 >= 23 + 8 = 31.
  The going back is sticky (server/rows_loan.h): once under the low mark it completes when nothing reads on the set and is
  withdrawn only if the free groups reach the ask condition.

  chunks    one cold prompt of --len tokens (default 150000), read on the lent set: PROGRESS frames == --expect (10 at 16,384 rows,
            19 at 8,192), the loan held (tenant_groups[4] == --groups, 23), the start table's line in --log:
            "store tenant 4, prefill rows (device memory, on demand): 23 groups"
              prefill_loan_test.py chunks --socket S --tokens T.npy [--len 150000] [--expect 10] [--groups 23] [--log rank0.log]
  pressure  conversations of --fill-len tokens (kept, one session each) are added until the loan has gone back (tenant_groups[4]
            == 0, within --window s of the last one): none of them was evicted before it went (each, asked again, is answered from
            the cache: cached == its prompt), the pool's free numbers fell under the low mark (2 groups) first, and the next cold
            prompt of --len tokens is read in --expect chunks (19 at 8,192 rows). The sessions stay for `room`
              prefill_loan_test.py pressure --socket S --tokens T.npy [--fill-len 50000] [--max-sessions 60] [--batch 4] [--expect 19]
  cycle     `pressure` and then `room` on one connection (the sessions of the first are released by the second), --expect is the
            first part's (19); the second part expects 10
              prefill_loan_test.py cycle --socket S --tokens T.npy [--fill-len 50000] [--max-sessions 60] [--batch 4]
  room      the sessions of `pressure` released (--sessions N, ids 7000..): the loan comes back (tenant_groups[4] == --groups) within --window s
            (the pool's free numbers above the set plus 8 groups, then the ask), and the next cold prompt reads in --expect (10)
              prefill_loan_test.py room --socket S --tokens T.npy --sessions N [--expect 10]
  asked     a prompt of --len tokens (default 400000) is read on the lent set and, while it is read, the pool is pressed (sessions
            as in `pressure`): every chunk but the last has 16,384 rows (PROGRESS deltas), the prompt ends on the set, and only then
            does the loan go back. With --cancel-after-chunks N the request is cancelled after N chunks (a lookup for the next chunk
            is in flight): the engine goes on serving (a short request is answered), and the loan goes back without a crash
              prefill_loan_test.py asked --socket S --tokens T.npy [--len 400000] [--cancel-after-chunks 3]
  fault     (the injection fires once, at the start's take; the engine then does not ask again for 300 s, so the prompt reads at 8,192
            rows; follow it with bit_probe.py --n 2 --len 150000 --new 64 --out F.json and --compare against the 8,192-row record)
            after a fault build (-DDSGB10_TEST_FAULTS=ON, DSGB10_PFLOAN_FAIL=rank:R in the engine's environment: rank R's twentieth
            allocation of the lent set is refused): the loan is not held (tenant_groups[4] == 0, the groups back in the pool), the
            log says the change failed, and a cold prompt of --len tokens reads in --expect chunks (19)
              prefill_loan_test.py fault --socket S --tokens T.npy [--len 150000] [--expect 19] [--log rank0.log]
  digests   DSGB10_STORE_CHECK=1: the four ranks' "store-check" digest lines are equal line by line (the loan's commands included)
              prefill_loan_test.py digests --logs rank0.log rank1.log rank2.log rank3.log
"""
import argparse
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import CANCEL, DONE, PROGRESS, TOKENS, Engine, load_npy_i32  # noqa: E402

STATS_REQ, STATS = 0x05, 0x85
RELEASE = 0x04
# STATS: tenant_groups[8] starts at the 55th u64 after the header (common/frontend_protocol.h offsetof(tenant_groups) == 8 + 8 * 54)
TENANT0, LOAN = 54, 4
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def stats(e):
    time.sleep(0.6)  # the engine publishes at most twice a second
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == STATS:
            _, n = struct.unpack('<II', p[:8])
            v = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])
            if n < TENANT0 + 8:
                raise SystemExit('the engine reports %d STATS fields: no tenant table' % n)
            return v


def loan_groups(e):
    return stats(e)[TENANT0 + LOAN]


def wait_loan(e, groups, window):
    t0 = time.time()
    while time.time() - t0 < window:
        if loan_groups(e) == groups:
            return time.time() - t0
    return None


def read_prompt(e, session, prompt, max_new=8, cancel_after_chunks=None, keep=True):
    """submit and collect, counting the PROGRESS frames: (chunks, rows of each chunk, the DONE's fields)"""
    rid = e.submit(session, prompt, max_new=max_new, keep=keep)[0]
    done_rows, last, cancelled = [], 0, False
    while True:
        t, p = e.frame()
        if t == PROGRESS:
            r, d, total = struct.unpack('<QII', p[:16])
            if r == rid:
                done_rows.append(d - last)
                last = d
                if cancel_after_chunks and not cancelled and len(done_rows) >= cancel_after_chunks:
                    e.send(CANCEL, struct.pack('<Q', rid))
                    cancelled = True
        elif t == DONE:
            r, reason, pt, cached = struct.unpack('<QIII', p[:20])
            if r == rid:
                return done_rows, dict(prompt=pt, cached=cached, reason=reason)
        elif t == TOKENS:
            pass


def cold_prompt(toks, n, offset=0):
    return [int(x) for x in toks[offset:offset + n]]


def log_has(path, pat):
    return bool(path) and any(re.search(pat, ln) for ln in open(path, errors='replace'))


def fill(e, toks, args, sessions, stop_when_back):
    """kept conversations of --fill-len tokens, one session each, in batches of --batch submitted together (the lines of a batch hold
    numbers together), until the loan is back (or --max-sessions)"""
    while len(sessions) < args.max_sessions:
        rids = {}
        for k in range(args.batch):
            sid = 7000 + len(sessions)
            prompt = cold_prompt(toks, args.fill_len, (len(sessions) * args.fill_len) % max(1, len(toks) - args.fill_len))
            rids[e.submit(sid, prompt, max_new=4)[0]] = 1
            sessions.append((sid, prompt))
        while rids:
            t, p = e.frame()
            if t == DONE:
                rids.pop(struct.unpack('<Q', p[:8])[0], None)
        if stop_when_back and loan_groups(e) == 0:
            return True
    return False


def run_room(e, toks, args):
    for i in range(args.sessions):
        e.send(RELEASE, struct.pack('<Q', 7000 + i))
    got = wait_loan(e, args.groups, args.window)
    check(got is not None, 'the loan came back (%d groups) after the sessions were released%s' % (args.groups, '' if got is None else ' in %.0f s' % got))
    rows, d = read_prompt(e, 3, cold_prompt(toks, args.len, 54321))
    exp = args.expect or 10
    check(len(rows) == exp, 'a cold prompt of %d tokens reads in %d chunks (%s)' % (args.len, exp, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['chunks', 'pressure', 'room', 'cycle', 'asked', 'fault', 'digests'])
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='')
    ap.add_argument('--len', type=int, default=150000)
    ap.add_argument('--expect', type=int, default=0)
    ap.add_argument('--groups', type=int, default=0, help='the loan the engine took at its start (0: read from STATS at the start; the set is 23 groups of 178 MiB at 16,384 rows)')
    ap.add_argument('--log', default='')
    ap.add_argument('--logs', nargs='*', default=[])
    ap.add_argument('--fill-len', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max-sessions', type=int, default=60)
    ap.add_argument('--window', type=float, default=120)
    ap.add_argument('--sessions', type=int, default=0)
    ap.add_argument('--cancel-after-chunks', type=int, default=0)
    args = ap.parse_args()
    if args.mode == 'digests':
        rows = [[ln.split('digest')[-1].strip() for ln in open(p, errors='replace') if 'store-check:' in ln and 'digest' in ln] for p in args.logs]
        check(len(rows) == 4 and len(set(map(len, rows))) == 1 and rows[0], 'four ranks, the same number of digest lines (%s)' % [len(r) for r in rows])
        check(all(len(set(c)) == 1 for c in zip(*rows)), 'every digest line equal on the four ranks')
        sys.exit(1 if fails else 0)
    toks = load_npy_i32(args.tokens)
    e = Engine(args.socket)
    if not args.groups and args.mode not in ('room', 'fault'):
        args.groups = loan_groups(e)
        check(args.groups > 0, 'the engine holds the loan at its start (%d groups)' % args.groups)
    if args.mode == 'chunks':
        check(loan_groups(e) == args.groups, 'the loan is held at the start: %d groups' % args.groups)
        check(not args.log or log_has(args.log, r'store tenant 4, prefill rows \(device memory, on demand\): %d groups' % args.groups),
              "the start table's line for tenant 4 shows %d groups" % args.groups)
        rows, d = read_prompt(e, 1, cold_prompt(toks, args.len))
        exp = args.expect or 10
        check(len(rows) == exp, 'a cold prompt of %d tokens read in %d chunks (%s)' % (args.len, exp, rows))
        check(max(rows) <= 16384 and rows[0] == 16384, 'the first chunk has 16,384 rows (%d)' % rows[0])
    elif args.mode in ('pressure', 'cycle'):
        sessions = []
        check(loan_groups(e) == args.groups, 'the loan is held before the pressure')
        back = fill(e, toks, args, sessions, True)
        check(back, 'the loan went back under the pool filling (tenant_groups[4] == 0 after %d sessions)' % len(sessions))
        check(wait_loan(e, 0, args.window) is not None, 'tenant 4 holds no group')
        hit = [read_prompt(e, sid, p, keep=True)[1]['cached'] >= len(p) - 1 for sid, p in sessions]
        check(all(hit), 'no conversation was evicted before the loan went (%d of %d answered from the cache or the SSD tier, to within one snapshot spacing)' % (sum(hit), len(hit)))
        rows, d = read_prompt(e, 2, cold_prompt(toks, args.len, 12345))
        exp = args.expect or 19
        check(len(rows) == exp, 'the next cold prompt of %d tokens reads in %d chunks (%s)' % (args.len, exp, rows))
        print('sessions kept: %d (ids 7000..)' % len(sessions))
        if args.mode == 'cycle':  # the sessions released: the loan comes back, 10 chunks again
            args.sessions, args.expect = len(sessions), 10
            run_room(e, toks, args)
    elif args.mode == 'room':
        run_room(e, toks, args)
    elif args.mode == 'asked':
        n = args.len if args.len != 150000 else 400000
        check(loan_groups(e) == args.groups, 'the loan is held')
        rid = e.submit(9, cold_prompt(toks, n), max_new=8)[0]
        press, rows, last, cancelled, pending = [], [], 0, False, set()
        while True:
            t, p = e.frame()
            if t == PROGRESS:
                r, d, total = struct.unpack('<QII', p[:16])
                if r != rid:
                    continue
                rows.append(d - last)
                last = d
                if not press:  # from the first chunk on the pool is pressed: other conversations are submitted beside the long prompt
                    for i in range(args.max_sessions):
                        q = cold_prompt(toks, args.fill_len, (i * args.fill_len) % max(1, len(toks) - args.fill_len))
                        pending.add(e.submit(7000 + i, q, max_new=4)[0])
                        press.append(7000 + i)
                if args.cancel_after_chunks and not cancelled and len(rows) >= args.cancel_after_chunks:
                    e.send(CANCEL, struct.pack('<Q', rid))
                    cancelled = True
            elif t == DONE:
                r = struct.unpack('<Q', p[:8])[0]
                if r == rid:
                    break
                pending.discard(r)
        if not args.cancel_after_chunks:
            check(all(r == 16384 for r in rows[:-1]), 'every chunk of the prompt but the last had 16,384 rows (%s)' % rows)
            check(len(rows) == -(-n // 16384), 'the prompt read in %d chunks' % (-(-n // 16384)))
        else:
            check(len(rows) >= args.cancel_after_chunks, 'the request was cancelled after %d chunks (%s)' % (args.cancel_after_chunks, rows))
        while pending:  # the pressing requests end
            t, p = e.frame()
            if t == DONE:
                pending.discard(struct.unpack('<Q', p[:8])[0])
        r2, d2 = read_prompt(e, 10, cold_prompt(toks, 3000, 777), max_new=8, keep=False)
        check(d2['prompt'] > 0, 'the engine goes on serving')
        check(wait_loan(e, 0, args.window) is not None, 'the loan went back once the prompt had ended (the pool was pressed)')
        for sid in press:
            e.send(RELEASE, struct.pack('<Q', sid))
    elif args.mode == 'fault':
        check(loan_groups(e) == 0, 'the loan is not held (the groups are back in the pool)')
        check(not args.log or log_has(args.log, r'prefill rows .*failed|chunks stay at the standing rows'), 'the log says the change failed')
        rows, d = read_prompt(e, 1, cold_prompt(toks, args.len))
        exp = args.expect or 19
        check(len(rows) == exp, 'a cold prompt of %d tokens reads in %d chunks (%s)' % (args.len, exp, rows))
    print('%s prefill_loan_test %s: %d failed' % ('FAIL' if fails else 'PASS', args.mode, fails))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
