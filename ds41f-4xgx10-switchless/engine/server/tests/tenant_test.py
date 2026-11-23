#!/usr/bin/env python3
"""The pool's tenants from outside, on the four Sparks (model/tenant.cu, lend.h, tenant.h): a dummy device tenant, id 15, takes
groups from the chunk pool and gives them back while sessions decode. The host tests (tenant_test, lend_test) cover the rules; this
is the engine's proof that a Device tenant's hand-over really moves memory, that every rank takes the same decisions, and that a
failure on one rank leaves a good engine.

WHAT IT NEEDS. A FAULT BUILD (cmake -DDSGB10_TEST_FAULTS=ON): the dummy tenant is compiled only there, and it is driven by CONFIG
field 14, `test_tenant_groups` (the groups it holds; a normal build never reads it). The engine is started by the caller (rank 0's
socket, dsfront stopped; the engine serves one front-end connection at a time, so this script's one connection carries the
sessions, the STATS and the probe request), on all four Sparks with DSGB10_STORE_CHECK=1 in the environment (the four ranks'
digests are compared from the logs) and --lend-null-mib set from `engram_lend_test.py null`. The pool's settings are the engine's
usual ones (--pool-free-gib, the 1M window); the groups are 178 MiB each, so 12 groups are 2 GiB of device memory.

  grow    CONFIG test_tenant_groups 2, then 12, then 0, while --sessions requests decode (default 8): each size must land
          (STATS tenant_models_groups = the groups asked; the dummy is the only model tenant), the owed groups stay 0, the pool's
          numbers fall by 64 a group and come back whole at 0; MemAvailable on every Spark (ssh: an instrument independent of the
          engine) within --tol-mib of before at each landed size (the memory moved from the pool to the block: net flat); the
          longest gap between a session's tokens in each change window against a control window with no change (reported, and
          not above the control's longest gap + --gap-slack-ms); the engine answers a request after.
            tenant_test.py grow --socket S --tokens T.npy --hosts ... [--sizes 2,12,0] [--sessions 8]
  fail    the same with a failure injected on one rank (the engine started with DSGB10_LEND_FAIL=rank:R,step:B|C|back,tenant:15
          on that rank only): the first CONFIG must NOT land (tenant_models_groups stays at its old value, the owed groups 0 or
          returning within --settle s), the engine serves (a request is answered), and a second CONFIG to another size lands (the
          fault fires once). `back` needs a grow that lands first (--sizes 2,12 shows it) and fails the give-back of a shrink.
            tenant_test.py fail --socket S --tokens T.npy --step B|C|back --mib-groups 12 [--hosts ...]
  cache-off  the same as grow, for an engine started with the Engram cache off (--engram-cache-mib 0): the tick still runs for
          a tenant that is not the cache, the groups land. Run it as `grow` against that engine; this name is only the label.
  check-log  after any of them: the four ranks' logs: every tenant-15 lend command's digest equal across the ranks line by line,
          every "tenant table" line says "adds up", the dummy block's pattern "intact" at every read-back and never "BROKEN";
          --expect fail (B and C steps): rank 0 printed "tenant 15 ... failed"; --expect owed (the back step): rank 0 printed the
          shrink's groups "owed to the pool" and no failed line (--expect-fail is refused for a log that shows only a debt).
            tenant_test.py check-log --logs rank0.log rank1.log rank2.log rank3.log [--expect fail|owed]
Every check prints ok / FAIL; the exit status is 0 only when all hold. The GPU regression set that goes with this build
(engram_lend_test.py: identity, resize, check-log; factor_test.py; lazy_window_test.py, each with DSGB10_STORE_CHECK=1) is run
by the main session against the normal build.
"""
import argparse
import os
import re
import select
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engram_lend_test as el  # noqa: E402
from serve_test import CANCEL, DONE, TOKENS, Engine, load_npy_i32  # noqa: E402

CONFIG, STATS_REQ, STATS = 0x06, 0x05, 0x85
# STATS field indexes (common/frontend_protocol.h)
POOL_NUMBERS, ENGRAM_GROUPS, OWED, TENANT0, MODELS = 0, 36, 42, 54, 62
CHANGE_WINDOW = 300
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def config_groups(e, groups):
    """CONFIG with every field -1 (kept) but index 14, test_tenant_groups"""
    e.send(CONFIG, struct.pack('<II', 3, 15) + struct.pack('<15q', *([-1] * 14 + [groups])))


class Conn:
    """One connection for the sessions, STATS and the probe request (the engine serves one front-end at a time)."""

    def __init__(self, sock, tokens, sessions):
        self.T = list(load_npy_i32(tokens))
        self.e = Engine(sock)
        self.last, self.gaps, self.got = {}, [], {}
        self.probe = {'rid': None, 'done': False, 't0': 0.0, 'first': None, 'done_at': None}
        self.post = {}  # each session's token arrivals since the probe's DONE
        self.last_ttft = None
        self.rids = [self.e.submit(0x74656e00 + k, self.T[k * 4096:(k + 1) * 4096], max_new=100000, keep=False, stop=())[0] for k in range(sessions)]
        self.last = {r: None for r in self.rids}
        self.post = {r: 0 for r in self.rids}

    def handle(self, t, p):
        now = time.time()
        if t == TOKENS:
            r = struct.unpack('<Q', p[:8])[0]
            if r in self.last:
                if self.last[r] is not None:
                    self.gaps.append((now, now - self.last[r]))
                self.last[r] = now
                if self.probe['done_at'] is not None:
                    self.post[r] += 1
            elif r == self.probe['rid'] and self.probe['first'] is None:
                self.probe['first'] = now - self.probe['t0']
        elif t == DONE:
            if struct.unpack('<Q', p[:8])[0] == self.probe['rid']:
                self.probe['done'] = True
                self.probe['done_at'] = now
        elif t == STATS:
            n = struct.unpack('<II', p[:8])[1]
            if n < MODELS + 1:
                raise SystemExit('the engine reports %d STATS fields: it has no tenants table' % n)
            self.got['s'] = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])

    def pump(self, until, stop=lambda: False):
        while time.time() < until and not stop():
            if select.select([self.e.s], [], [], max(0.0, min(0.2, until - time.time())))[0]:
                self.handle(*self.e.frame())

    def stats(self):
        self.got.pop('s', None)
        self.e.send(STATS_REQ, struct.pack('<II', 0, 0))
        self.pump(time.time() + 10, lambda: 's' in self.got)
        return self.got.get('s')

    def wait_models(self, groups, timeout):
        """until tenant_models_groups == groups; the STATS then, or None"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.pump(time.time() + 1.0)
            s = self.stats()
            if s and s[MODELS] == groups:
                return s
        return None

    def answer(self, tag, n=16):
        """The first-token probe (a 2,048-token prompt: the interleaver stops the decoders while it prefills), then until every
        session has had two token arrivals after its DONE: the probe's silence ends before the next window starts, so no window holds
        a gap that is the probe's. self.last_ttft is the probe's time to its first token."""
        self.probe.update(rid=None, done=False, done_at=None, first=None, t0=time.time())
        self.post = {r: 0 for r in self.post}
        self.probe['rid'] = self.e.submit(tag, self.T[9 * 2048:10 * 2048], max_new=n, keep=False)[0]
        self.pump(time.time() + 120, lambda: self.probe['done'])
        self.pump(time.time() + 120, lambda: all(v >= 2 for v in self.post.values()))
        self.probe['done_at'] = None
        self.last_ttft = self.probe['first']
        return self.probe['done']

    def close(self):
        for r in self.rids:
            self.e.send(CANCEL, struct.pack('<Q', r))


def cmd_grow(a):
    c = Conn(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)  # the sessions settle
    s0 = c.stats()
    mem0 = el.meminfo(a.hosts) if a.hosts else None
    check(s0[MODELS] == 0 and s0[OWED] == 0, 'the dummy tenant holds nothing at the start (models %d groups, owed %d)' % (s0[MODELS], s0[OWED]))
    c.pump(time.time() + 2)
    check(c.answer(0x74656e60), 'the control window\'s probe was answered (prefill %.0f ms)' % ((c.last_ttft or -1) * 1000))
    pre_ms = (c.last_ttft or -1) * 1000
    t = time.time()
    c.pump(t + a.control)
    ctrl = max((g for ts, g in c.gaps if t <= ts <= time.time()), default=0)
    print('control window %.0f s: longest gap between a session\'s tokens %.0f ms' % (a.control, ctrl * 1000))
    for groups in [int(x) for x in a.sizes.split(',')]:
        t = time.time()
        config_groups(c.e, groups)
        s = c.wait_models(groups, CHANGE_WINDOW)
        t1 = time.time()
        check(s is not None, '%d groups landed while %d sessions decode (%.1f s)' % (groups, a.sessions, t1 - t))
        if s is None:
            break
        check(s[OWED] == 0, '%d groups: nothing is owed' % groups)
        check(s[TENANT0 + 1] == s0[TENANT0 + 1], '%d groups: the Engram cache\'s groups are as they were (%d)' % (groups, s[TENANT0 + 1]))
        check(s[POOL_NUMBERS] == s0[POOL_NUMBERS] - 64 * groups, '%d groups: the pool lost exactly %d numbers (%d -> %d)' %
              (groups, 64 * groups, s0[POOL_NUMBERS], s[POOL_NUMBERS]))
        g = max((g for ts, g in c.gaps if t <= ts <= t1), default=0)
        check(g <= ctrl + a.gap_slack_ms / 1000.0, '%d groups: the longest gap %.0f ms, the control\'s %.0f ms (the probe before this window: prefill %.0f ms)' %
              (groups, g * 1000, ctrl * 1000, pre_ms))
        if mem0:
            mem1 = el.meminfo(a.hosts)
            for k, h in enumerate(a.hosts):
                check(abs(mem1[k] - mem0[k]) <= a.tol_mib, '%d groups: %s MemAvailable within %.0f MiB of before (%.0f -> %.0f)' % (groups, h, a.tol_mib, mem0[k], mem1[k]))
        c.answer(0x74656e61 + groups)  # a probe after each window, and two tokens of every session, so its silence is in none
        pre_ms = (c.last_ttft or -1) * 1000
    s = c.stats()
    check(s[POOL_NUMBERS] == s0[POOL_NUMBERS] - 64 * s[MODELS] and s[OWED] == 0, 'the pool is whole again (or short exactly the groups still held), nothing owed')
    check(c.answer(0x74656e70), 'the engine answers a request after')
    c.close()
    return 0


def cmd_fail(a):
    c = Conn(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)
    s0 = c.stats()
    before = s0[MODELS]
    if a.step == 'back':  # the grow lands, the shrink's give-back is the failure
        config_groups(c.e, a.mib_groups)
        s = c.wait_models(a.mib_groups, CHANGE_WINDOW)
        check(s is not None, 'the grow to %d groups landed' % a.mib_groups)
        before = a.mib_groups
        config_groups(c.e, 0)
        c.pump(time.time() + a.wait)
        s = c.stats()
        check(s[MODELS] == 0, 'the shrink landed (its give-back failed once: the groups owed, then settled)')
        t0 = time.time()
        while s[OWED] > 0 and time.time() - t0 < a.settle:
            c.pump(time.time() + 1.0)
            s = c.stats()
        check(s[OWED] == 0, 'the owed groups came back within %d s' % a.settle)
    else:
        config_groups(c.e, a.mib_groups)
        c.pump(time.time() + a.wait)
        s = c.stats()
        check(s[MODELS] == before, 'the change did not land (tenant_models_groups %d -> %d): the fault rolled it back' % (before, s[MODELS]))
        check(s[OWED] == 0 or s[POOL_NUMBERS] == s0[POOL_NUMBERS] - 64 * s[OWED], 'the groups are back in the pool, or owed (owed %d)' % s[OWED])
        check(s[POOL_NUMBERS] == s0[POOL_NUMBERS] or s[OWED] > 0, 'the pool has its numbers (%d, was %d)' % (s[POOL_NUMBERS], s0[POOL_NUMBERS]))
        check(c.answer(0x74656e71), 'the engine serves after the fault')
        config_groups(c.e, max(1, a.mib_groups // 2))  # a new setting is tried at once; the fault fired once
        s = c.wait_models(max(1, a.mib_groups // 2), CHANGE_WINDOW)
        check(s is not None, 'a second size lands (the fault fires once)')
    check(c.answer(0x74656e72), 'the engine answers a request at the end')
    c.close()
    return 0


def cmd_check_log(a):
    texts = [open(p, errors='replace').read() for p in a.logs]
    lines = [re.findall(r'store-check: rank \d lend command of tenant 15 \(phase (\d), (\d+) MiB\) status (\d) digest ([0-9a-f]+)', t) for t in texts]
    check(all(lines) or a.expect_fail or a.expect == 'fail', 'every rank logged tenant-15 lend commands (%s)' % ' '.join(str(len(x)) for x in lines))
    check(all(x == lines[0] for x in lines), 'the four ranks\' tenant-15 commands and digests are equal line by line')
    for k, t in enumerate(texts):
        tab = re.findall(r'tenant table (adds up|BROKEN)', t)
        check(tab and all(x == 'adds up' for x in tab), 'rank %d: the tenant table added up at every command (%d checked)' % (k, len(tab)))
        check('pattern BROKEN' not in t, 'rank %d: the dummy block\'s pattern was never broken (%d read-backs intact)' % (k, t.count('pattern intact')))
    failed = re.search(r'store: tenant 15 \d+ -> \d+ groups failed', texts[0]) is not None
    owed = re.search(r'store: tenant 15 \d+ -> \d+ groups: \d+ groups owed to the pool', texts[0]) is not None
    expect = 'fail' if a.expect_fail else a.expect
    if expect == 'fail':
        check(failed, 'rank 0 logged a failed tenant-15 change')
        check(not (owed and not failed), 'this log is not a back step\'s (its debt was owed, not failed): use --expect owed for the back step')
    elif expect == 'owed':
        check(owed, 'rank 0 logged the shrink\'s groups owed to the pool (the failed give-back of the back step)')
        check(not failed, 'and no failed tenant-15 change: the owed groups were settled, not a failure')
    else:
        check(not failed, 'rank 0 logged no failed tenant-15 change')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--socket', required=True)
        p.add_argument('--tokens', required=True)
        p.add_argument('--hosts', nargs='*', default=[])
        p.add_argument('--sessions', type=int, default=8)
        p.add_argument('--tol-mib', type=float, default=256)
    for name in ('grow', 'cache-off'):
        p = sub.add_parser(name)
        common(p)
        p.add_argument('--sizes', default='2,12,0')
        p.add_argument('--control', type=int, default=60)
        p.add_argument('--gap-slack-ms', type=float, default=0)
    p = sub.add_parser('fail')
    common(p)
    p.add_argument('--step', choices=['B', 'C', 'back'], required=True)
    p.add_argument('--mib-groups', type=int, default=12)  # the groups the failing change asks for
    p.add_argument('--wait', type=int, default=40)
    p.add_argument('--settle', type=int, default=30)
    p = sub.add_parser('check-log')
    p.add_argument('--logs', nargs=4, required=True)
    p.add_argument('--expect-fail', action='store_true', help='B and C steps: the change failed (same as --expect fail)')
    p.add_argument('--expect', choices=['none', 'fail', 'owed'], default='none',
                   help='fail: B or C step; owed: the back step (the failed give-back leaves the groups owed, then settled; no failed line)')
    a = ap.parse_args()
    rc = {'grow': cmd_grow, 'cache-off': cmd_grow, 'fail': cmd_fail, 'check-log': cmd_check_log}[a.cmd](a)
    print('%s tenant_test %s: %d failed' % ('FAIL' if fails else 'PASS', a.cmd, fails))
    return 1 if fails else rc


if __name__ == '__main__':
    sys.exit(main())
