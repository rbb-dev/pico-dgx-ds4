#!/usr/bin/env python3
"""The Engram row cache handed over from the chunk pool (model/tenant.cu), from outside, on the four Sparks. The engine is started
by the caller (rank 0's socket, dsfront stopped); the host tests (lend_test, engram_block_test) cover the rules. Every check
prints ok / FAIL; the exit status is 0 only when all hold. MemAvailable is read over ssh from /proc/meminfo on each Spark: an
instrument independent of the engine's own counters.

  null     the null control: MemAvailable on every Spark every --interval s for --rounds rounds, the engine idle; prints the
           spread (max - min) per Spark, the proof's tolerance is max(spread, 256 MiB) (the engine takes --lend-null-mib)
             engram_lend_test.py null --hosts user@rank0-host ... [--rounds 12] [--interval 10]
  flat     MemAvailable is flat from ready on while requests run (a fall by the cache's ordinary part during the first
           requests fails it): the fall over --seconds against the start, per Spark
             engram_lend_test.py flat --socket S --tokens T.npy --hosts ... [--seconds 600] [--tol-mib 256]
  resize   CONFIG engram_mib through --sizes in turn; each change must land (STATS engram_cache_bytes, engram_groups, no pending
           change, ok total up, failed total unchanged) with MemAvailable on every Spark moved by what the change's arithmetic
           says, within the tolerance: the groups that left or came back to the pool (--group-mib each) minus the change in the
           block's bytes (the slots above the display reservation, --display-mib, and the index); whole groups move, so the
           cache's own change is not what MemAvailable shows. The time it took, and the STATS the engine reports (rank 0's log
           carries the proof's lines: check-log)
             engram_lend_test.py resize --socket S --hosts ... --sizes 2048,6912,4096 [--tol-mib 256] [--group-mib 178] [--display-mib 2032]
  load     the same sizes while --sessions requests decode; the longest gap between a session's tokens in each change's window,
           against the longest gap of a window of the same length with no change (the control), and the first request's TTFT
           after each change; not above the control's longest gap (+ --gap-slack-ms)
           The control window is as long as the longest change window (--window, 300 s), so both windows look for a maximum over
           the same time
             engram_lend_test.py load --socket S --tokens T.npy --sizes 4096,2048,6912,4096 --sessions 8 [--window 300]
  identity replies equal whatever the cache's size: the probe (bit_probe.py) at the starting size, at 4 MiB (the working set
           overflows it: STATS evictions must rise), grown back to the start size, shrunk to 4 MiB again; every pair must be
           bit-equal. With --break the engine was started from a build with -DDSGB10_TEST_FAULTS=ON and DSGB10_ENGRAM_BREAK=
           <layer>:<row> naming a row the prompts read (the log shows "test break"): the control, which must DIFFER (exit 1 from
           the compare) so that the probe is known to see a flipped row
             engram_lend_test.py identity --socket S --tokens T.npy [--small-mib 4] [--n 8] [--len 2048] [--new 200] [--out-dir D]
  check-log  rank logs: every rank made the cache at the start (its line), rank 0's lend lines, no "failed" unless --expect-fail,
           the four ranks' store-check digests (DSGB10_STORE_CHECK=1) equal line by line
             engram_lend_test.py check-log --logs rank0.log rank1.log rank2.log rank3.log [--expect-fail]
  fault    after a change was made to fail by DSGB10_LEND_FAIL=rank:R,step:B|C,skip:1 or rank:R,step:back (a
           -DDSGB10_TEST_FAULTS=ON build, set in the engine's environment; skip:1 leaves the start's hand-over alone, the first
           CONFIG fails): CONFIG to --mib, then the engine's refusal is waited for (failed total +1; the change itself stays pending for the
           engine's back-off, 300 s, and is not waited for) and the cache must stay at its old size (B, C); for back the grow
           lands, the CONFIG back to the starting size is the shrink whose give-back fails, the groups owed and then returned
           within --settle s; the engine serving (a request is answered), and a probe equal to the one before (--before A.json
           from bit_probe.py)
             engram_lend_test.py fault --socket S --tokens T.npy --mib 6912 --step B|C|back [--before A.json] [--settle 30]
  timing   decode and prefill at each size for the timing arm: --sizes 4096,2032,6912; one decode arm (1 and 3 streams) and one
           128K prefill per size, rates printed; compare with the S1 build's record of the same arm (the 4-5 % floor)
             engram_lend_test.py timing --socket S --tokens T.npy --sizes 4096,2032,6912
"""
import argparse
import json
import os
import re
import select
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import CANCEL, DONE, PROGRESS, TOKENS, Engine, load_npy_i32  # noqa: E402

CONFIG, STATS_REQ, STATS = 0x06, 0x05, 0x85
# STATS field indexes of the Engram fields (common/frontend_protocol.h, after `ranks`)
F = dict(cache_bytes=35, groups=36, pending_mib=37, lookups=38, hits=39, ok=40, failed=41, owed=42, huge=43, evictions=44)
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


CHANGE_WINDOW = 300  # the longest a change may take to land (wait_landed, load's windows), and the control window's length
REFUSAL_WINDOW = 60  # the longest the engine may take to refuse a change made to fail (wait_refused)
ASKED = [None]  # the size the last CONFIG asked for


def config_mib(e, mib):
    ASKED[0] = mib
    e.send(CONFIG, struct.pack('<II', 3, 10) + struct.pack('<10q', *([-1] * 9 + [mib])))


def stats(e):
    time.sleep(0.6)  # the engine publishes at most twice a second; a CONFIG makes the next one current
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == STATS:
            version, n = struct.unpack('<II', p[:8])
            v = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])
            if n <= F['evictions']:
                raise SystemExit('the engine reports %d STATS fields: it has no Engram hand-over' % n)
            return {k: v[i] for k, i in F.items()}


def meminfo(hosts):
    """MemAvailable in MiB on each host (ssh; one number per host)"""
    out = []
    for h in hosts:
        # 'local': this box's own /proc/meminfo (rank 0's Spark cannot ssh to itself)
        r = subprocess.run(['awk', '/MemAvailable/ {print $2}', '/proc/meminfo'] if h == 'local' else
                           ['ssh', '-o', 'BatchMode=yes', h, "awk '/MemAvailable/ {print $2}' /proc/meminfo"], capture_output=True, text=True, timeout=30)
        out.append(int(r.stdout.split()[0]) / 1024.0)
    return out


def landed(s, before):
    """a change finished (ok or failed total moved), or the size asked for is the size already (the engine counts no change
    for it), and none is pending; the cache's bytes are whole slots, within 1 MiB of the MiB asked"""
    same = ASKED[0] is not None and abs(s['cache_bytes'] / 1048576.0 - ASKED[0]) <= 1.0
    return s['pending_mib'] == 0 and (s['ok'] + s['failed'] > before['ok'] + before['failed'] or same)


def wait_landed(e, before, timeout=CHANGE_WINDOW):
    """until landed(); returns (seconds, STATS)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = stats(e)
        if landed(s, before):
            return time.time() - t0, s
    raise SystemExit('the change did not land in %d s' % timeout)


def prompt_of(T, i, n):
    return T[i * n:(i + 1) * n]


def cmd_null(a):
    rounds = []
    for r in range(a.rounds):
        rounds.append(meminfo(a.hosts))
        if r + 1 < a.rounds:
            time.sleep(a.interval)
    for i, h in enumerate(a.hosts):
        col = [x[i] for x in rounds]
        print('%s: MemAvailable %.0f .. %.0f MiB, spread %.1f MiB' % (h, min(col), max(col), max(col) - min(col)))
    spread = max(max(x[i] for x in rounds) - min(x[i] for x in rounds) for i in range(len(a.hosts)))
    print('tolerance = max(spread %.1f, 256) = %.1f MiB' % (spread, max(spread, 256)))
    return 0


def cmd_flat(a):
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    base = meminfo(a.hosts)
    t0 = time.time()
    worst = [0.0] * len(a.hosts)
    i = 0
    while time.time() - t0 < a.seconds:
        o = e.run(0x666c6174 + i, prompt_of(T, i % 48, 8192), max_new=64, keep=False)  # 48 distinct prompts: the cache keeps growing
        check(not o['error'], 'request %d answered' % i) if o['error'] else None
        now = meminfo(a.hosts)
        for k in range(len(a.hosts)):
            worst[k] = max(worst[k], base[k] - now[k])
        i += 1
    for k, h in enumerate(a.hosts):
        check(worst[k] <= a.tol_mib, '%s: MemAvailable fell by at most %.0f MiB over %d requests (tolerance %.0f)' % (h, worst[k], i, a.tol_mib))
    return 0


def block_mib(cache_bytes, display_mib):
    """the block's MiB for a cache of `cache_bytes`: the slots above the display reservation and the index (keys, reference
    bits, a table of the power of two at least twice the slots, 4 bytes an entry)"""
    slots = cache_bytes // 264
    if not slots:
        return 0.0
    table = 1
    while table < slots * 2:
        table <<= 1
    return max(0.0, cache_bytes / 1048576.0 - display_mib) + (slots * 9 + table * 4) / 1048576.0


def cmd_resize(a):
    e = Engine(a.socket)
    s = stats(e)
    print('at start: %s' % s)
    for mib in [int(x) for x in a.sizes.split(',')]:
        before_mem = meminfo(a.hosts)
        config_mib(e, mib)
        sec, s2 = wait_landed(e, s)
        after_mem = meminfo(a.hosts)
        got = s2['cache_bytes'] / 1048576.0
        print('-> %d MiB landed in %.1f s: cache %.0f MiB, %d groups, huge pages %d %%, ok %d failed %d owed %d' %
              (mib, sec, got, s2['groups'], s2['huge'], s2['ok'], s2['failed'], s2['owed']))
        check(s2['failed'] == s['failed'], '%d MiB: no change failed' % mib)
        check(abs(got - mib) <= 1.0 or got < mib, '%d MiB: the cache is %.0f MiB (a size the pool cannot lend is cut, never past the setting)' % (mib, got))
        # whole groups move: the groups that changed hands, less the change in the block's bytes
        want = (s2['groups'] - s['groups']) * a.group_mib - (block_mib(s2['cache_bytes'], a.display_mib) - block_mib(s['cache_bytes'], a.display_mib))
        for k, h in enumerate(a.hosts):
            d = after_mem[k] - before_mem[k]
            check(abs(d - want) <= a.tol_mib, '%s: MemAvailable moved by %+.0f MiB, the arithmetic says %+.0f (%+d groups of %.0f MiB less the block) (within %.0f)' %
                  (h, d, want, s2['groups'] - s['groups'], a.group_mib, a.tol_mib))
        check(s2['owed'] == 0, '%d MiB: no groups owed after it' % mib)
        s = s2
    return 0


def cmd_load(a):
    """One connection carries the sessions, STATS and the first-token probe, so every frame goes through one handler: the
    sessions' token gaps are recorded whatever else is read (a STATS that dropped them would stretch the gaps it hid), and a
    frame is read only when the socket has one (select), never cut by a timeout. The control window polls STATS the same way
    the change windows do."""
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    s = stats(e)
    rids = [e.submit(0x6c6f6100 + k, prompt_of(T, k, 4096), max_new=100000, keep=False, stop=())[0] for k in range(a.sessions)]
    last = {r: None for r in rids}
    gaps = []  # (time, gap) of every session's token arrivals
    marks = []  # (label, t_start, t_end)
    pr = {'rid': None, 't0': 0.0, 'first': None, 'done': False, 'done_at': None}
    got = {}
    post = {r: 0 for r in rids}  # each session's token arrivals since the probe's DONE

    def handle(t, p):
        now = time.time()
        if t == TOKENS:
            r = struct.unpack('<Q', p[:8])[0]
            if r in last:
                if last[r] is not None:
                    gaps.append((now, now - last[r]))
                last[r] = now
                if pr['done_at'] is not None:
                    post[r] += 1
            elif r == pr['rid'] and pr['first'] is None:
                pr['first'] = now - pr['t0']
        elif t == DONE:
            if struct.unpack('<Q', p[:8])[0] == pr['rid']:
                pr['done'] = True
                pr['done_at'] = now
        elif t == STATS:
            n = struct.unpack('<II', p[:8])[1]
            v = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])
            got['s'] = {k: v[i] for k, i in F.items()}

    def pump(until, stop=lambda: False):
        while time.time() < until and not stop():
            if select.select([e.s], [], [], max(0.0, min(0.2, until - time.time())))[0]:
                handle(*e.frame())

    def stats_live():
        got.pop('s', None)
        e.send(STATS_REQ, struct.pack('<II', 0, 0))
        pump(time.time() + 10, lambda: 's' in got)
        return got.get('s')

    def window(seconds):  # the sessions decode, STATS asked once a second (as while a change lands)
        end = time.time() + seconds
        while time.time() < end:
            pump(min(end, time.time() + 1.0))
            stats_live()

    window(20)  # the sessions settle
    t = time.time()
    window(a.window)
    marks.append(('control (no change)', t, time.time()))
    for mib in [int(x) for x in a.sizes.split(',')]:
        config_mib(e, mib)
        t = time.time()
        done = None
        while time.time() < t + CHANGE_WINDOW and done is None:
            pump(time.time() + 1.0)
            s2 = stats_live()
            if s2 and landed(s2, s):
                done = s2
        marks.append(('-> %d MiB' % mib, t, time.time()))
        check(done is not None, '%d MiB landed while %d sessions decode' % (mib, a.sessions))
        if done:
            s = done
        pr.update(rid=None, t0=time.time(), first=None, done=False)
        pr['rid'] = e.submit(0x70726f62 + mib, prompt_of(T, 9, 2048), max_new=1, keep=False)[0]
        pump(time.time() + 120, lambda: pr['done'])
        ttft.append((mib, pr['first']))
    ctrl = max((g for ts, g in gaps if marks[0][1] <= ts <= marks[0][2]), default=0)
    print('control window %.0f s: longest gap between a session\'s tokens %.0f ms' % (a.window, ctrl * 1000))
    for label, t0, t1 in marks[1:]:
        g = max((g for ts, g in gaps if t0 <= ts <= t1), default=0)
        check(g <= ctrl + a.gap_slack_ms / 1000.0, '%s (%.0f s): the longest gap %.0f ms, the control\'s %.0f ms' % (label, t1 - t0, g * 1000, ctrl * 1000))
    for mib, tt in ttft:
        print('first request after %d MiB: TTFT %.2f s' % (mib, tt if tt is not None else -1))
    for r in rids:
        e.send(CANCEL, struct.pack('<Q', r))
    return 0


def probe(a, out, extra=()):
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, 'bit_probe.py'), '--socket', a.socket, '--tokens', a.tokens, '--out', out,
           '--n', str(a.n), '--len', str(a.len), '--new', str(a.new)] + list(extra)
    return subprocess.run(cmd).returncode == 0


def compare(x, y):
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run([sys.executable, os.path.join(here, 'bit_probe.py'), '--compare', x, y]).returncode == 0


def cmd_identity(a):
    os.makedirs(a.out_dir, exist_ok=True)
    e = Engine(a.socket)
    s = stats(e)
    start_mib = mib_asked(s['cache_bytes'])
    print('start: cache %d MiB, evictions %d' % (start_mib, s['evictions']))
    arms = {}

    def arm(name, mib=None):
        nonlocal s, e
        if mib is not None:
            config_mib(e, mib)
            _, s = wait_landed(e, s)
        out = os.path.join(a.out_dir, name + '.json')
        ev0 = s['evictions']
        at = mib_asked(s['cache_bytes'])
        e.s.close()  # the probe connects on its own, and the engine serves one front-end connection at a time
        ok = probe(a, out)
        e = Engine(a.socket)
        check(ok, 'probe %s at %d MiB' % (name, at))
        s = stats(e)
        arms[name] = out
        return s['evictions'] - ev0

    arm('A_start')
    ev = arm('B_small', a.small_mib)
    check(ev > 0, 'the %d MiB arm evicted rows (%d): the working set overflows it' % (a.small_mib, ev))
    arm('C_grown', start_mib)
    arm('D_shrunk', a.small_mib)
    for n in ('B_small', 'C_grown', 'D_shrunk'):
        ok = compare(arms['A_start'], arms[n])
        if a.brk:
            check(not ok, 'control: A_start against %s DIFFERS (a flipped cached row is seen by the probe)' % n)
        else:
            check(ok, 'A_start against %s: bit-equal' % n)
    return 0


def cmd_check_log(a):
    texts = [open(p, errors='replace').read() for p in a.logs]
    for r, t in enumerate(texts):
        m = re.search(r'rank %d: engram row cache: (\d+) MiB asked for, (\d+) MiB made \((\d+) groups of the pool, ([\d.]+) MiB resident in the block, (\d+) %% in huge pages\)' % r, t)
        check(m is not None, 'rank %d made its cache at the start: %s' % (r, m.group(0) if m else 'no line'))
    t0 = texts[0]
    changes = re.findall(r'store: Engram cache (\d+) -> (\d+) MiB[^\n]*', t0)
    print('rank 0: %d changes' % len(changes))
    for line in re.findall(r'store: Engram cache [^\n]*', t0):
        print('   ' + line)
    failed = re.findall(r'store: Engram cache [^\n]*failed[^\n]*', t0)
    check(bool(failed) == a.expect_fail, 'failed hand-overs %s' % ('expected and present' if a.expect_fail else 'none'))
    # the proof's three readings are inside the engine (tenant.cu): a change that landed passed them on every rank
    check(not re.search(r'did not add up', ''.join(texts)) or a.expect_fail, 'no proof reported a mismatch')
    # the same digest on the same line on every rank
    lines = [re.findall(r'store-check: rank \d lend command \(phase (\d), (\d+) MiB\) status (\d) digest ([0-9a-f]+)', t) for t in texts]
    if all(lines):
        n = min(len(x) for x in lines)
        check(all(x[:n] == lines[0][:n] for x in lines), 'the four ranks agree on %d lend commands (phase, size, status, digest)' % n)
    else:
        print('  (no store-check lines: start the engine with DSGB10_STORE_CHECK=1 for the digest comparison)')
    return 0


def cmd_fault(a):
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    s0 = stats(e)
    mem0 = meminfo(a.hosts) if a.hosts else None
    config_mib(e, a.mib)
    # B and C: the engine refuses the change and keeps the old size, and the same size is not tried again for its back-off, so
    # the change never lands: its refusal is what is waited for. back: the grow lands
    sec, s1 = wait_refused(e, s0) if a.step in ('B', 'C') else wait_landed(e, s0)
    print('after the injected %s fault: cache %d MiB (was %d), failed %d -> %d, owed %d, groups %d, pending %d MiB' %
          (a.step, mib_asked(s1['cache_bytes']), mib_asked(s0['cache_bytes']), s0['failed'], s1['failed'], s1['owed'], s1['groups'], s1['pending_mib']))
    if a.step in ('B', 'C'):
        check(s1['failed'] == s0['failed'] + 1, 'the change failed once (failed total %d -> %d)' % (s0['failed'], s1['failed']))
        check(s1['cache_bytes'] == s0['cache_bytes'], 'the cache stays at its old size (%d MiB)' % mib_asked(s0['cache_bytes']))
        check(s1['groups'] == s0['groups'] or s1['owed'] > 0, 'the groups are back in the pool (or owed)')
    else:
        # the grow made no give-back; the shrink to the starting size is the give-back whose pool grow fails once: the groups
        # are owed, then settle
        check(s1['failed'] == s0['failed'] and s1['cache_bytes'] > s0['cache_bytes'], 'the grow landed (cache %d MiB)' % mib_asked(s1['cache_bytes']))
        config_mib(e, mib_asked(s0['cache_bytes']))
        sec, s2 = wait_landed(e, s1)
        print('after the shrink (the give-back fails): cache %d MiB, ok %d failed %d owed %d' % (mib_asked(s2['cache_bytes']), s2['ok'], s2['failed'], s2['owed']))
        check(s2['cache_bytes'] == s0['cache_bytes'], 'the cache is back at its starting size (%d MiB)' % mib_asked(s0['cache_bytes']))
        check(s2['owed'] > 0 or s2['groups'] == s0['groups'], 'the failed give-back left the groups owed (%d) or the pool took them back at once' % s2['owed'])
        t0 = time.time()
        while s2['owed'] > 0 and time.time() - t0 < a.settle:
            time.sleep(1)
            s2 = stats(e)
        check(s2['owed'] == 0, 'the owed groups came back within %d s' % a.settle)
        s1 = s2
    o = e.run(0x66617574, prompt_of(T, 0, 2048), max_new=16, keep=False)
    check(not o['error'] and len(o['ids']) > 0, 'the engine serves after the fault')
    if a.before:
        out = os.path.join(os.path.dirname(os.path.abspath(a.before)), 'after_fault.json')
        check(probe(a, out) and compare(a.before, out), 'the probe after the fault is bit-equal to the one before')
    if mem0 and a.step in ('B', 'C'):
        mem1 = meminfo(a.hosts)
        for k, h in enumerate(a.hosts):
            check(abs(mem1[k] - mem0[k]) <= a.tol_mib, '%s: MemAvailable within %.0f MiB of before' % (h, a.tol_mib))
    return 0


def cmd_timing(a):
    """each size its own token slices (no arm reads rows an earlier arm cached), the hit rate over the arm alone"""
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    s = stats(e)
    pf = 120000  # 3 arms x (3 x 4,096 decode prompts + 120,000 prefill) fit the 400K-token file
    for arm, mib in enumerate(int(x) for x in a.sizes.split(',')):
        config_mib(e, mib)
        _, s = wait_landed(e, s)
        base = arm * (3 * 4096 + pf)
        rates = []
        for streams in (1, 3):
            rids = [e.submit(0x74696d00 + 16 * arm + 4 * streams + k, T[base + k * 4096:base + (k + 1) * 4096], max_new=a.decode_tokens,
                             keep=False, stop=())[0] for k in range(streams)]
            n, t0 = 0, None
            done = set()
            while len(done) < streams:
                t, p = e.frame()
                if t == TOKENS:
                    r, c = struct.unpack('<QI', p[:12])
                    if t0 is None:
                        t0 = time.time()
                    n += c
                elif t == DONE:
                    done.add(struct.unpack('<Q', p[:8])[0])
            rates.append(n / max(1e-9, time.time() - t0))
        r = e.submit(0x74696d99 + arm, T[base + 3 * 4096:base + 3 * 4096 + pf], max_new=1, keep=False)
        o = e.collect(r)
        s2 = stats(e)
        look, hit = s2['lookups'] - s['lookups'], s2['hits'] - s['hits']
        print('%d MiB (cache %d MiB): decode 1 stream %.1f tok/s, 3 streams %.1f tok/s aggregate, %dK prefill TTFT %.2f s, '
              'hit rate over the arm %.1f %% (%d of %d rows), evictions %d' %
              (mib, mib_asked(s2['cache_bytes']), rates[0], rates[1], pf // 1000, o['t_first'], 100.0 * hit / max(1, look), hit, look,
               s2['evictions'] - s['evictions']))
        s = s2
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, sock=True, hosts=False, tokens=False):
        if sock:
            p.add_argument('--socket', required=True)
        if tokens:
            p.add_argument('--tokens', required=True)
        if hosts:
            p.add_argument('--hosts', nargs='*', default=[])
    p = sub.add_parser('null')
    common(p, sock=False, hosts=True)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--interval', type=float, default=10)
    p = sub.add_parser('flat')
    common(p, hosts=True, tokens=True)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--tol-mib', type=float, default=256)
    p = sub.add_parser('resize')
    common(p, hosts=True)
    p.add_argument('--sizes', default='2048,6912,4096')
    p.add_argument('--tol-mib', type=float, default=256)
    p.add_argument('--group-mib', type=float, default=178.0)  # 64 numbers of the pool: what a group is worth in MemAvailable
    p.add_argument('--display-mib', type=float, default=2032)  # the cache's slots that sit in the display reservation, not the block
    p = sub.add_parser('load')
    common(p, tokens=True)
    p.add_argument('--sizes', default='4096,2048,6912,4096')
    p.add_argument('--sessions', type=int, default=8)
    p.add_argument('--window', type=int, default=CHANGE_WINDOW)
    p.add_argument('--gap-slack-ms', type=float, default=0)
    p = sub.add_parser('identity')
    common(p, tokens=True)
    p.add_argument('--small-mib', type=int, default=4)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--len', type=int, default=2048)
    p.add_argument('--new', type=int, default=200)
    p.add_argument('--out-dir', default='identity-out')
    p.add_argument('--break', dest='brk', action='store_true')
    p = sub.add_parser('check-log')
    p.add_argument('--logs', nargs=4, required=True)
    p.add_argument('--expect-fail', action='store_true')
    p = sub.add_parser('fault')
    common(p, hosts=True, tokens=True)
    p.add_argument('--mib', type=int, required=True)
    p.add_argument('--step', choices=['B', 'C', 'back'], required=True)
    p.add_argument('--before', default='')
    p.add_argument('--settle', type=int, default=30)
    p.add_argument('--tol-mib', type=float, default=256)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--len', type=int, default=2048)
    p.add_argument('--new', type=int, default=200)
    p = sub.add_parser('timing')
    common(p, tokens=True)
    p.add_argument('--sizes', default='4096,2032,6912')
    p.add_argument('--decode-tokens', type=int, default=400)
    a = ap.parse_args()
    rc = {'null': cmd_null, 'flat': cmd_flat, 'resize': cmd_resize, 'load': cmd_load, 'identity': cmd_identity,
          'check-log': cmd_check_log, 'fault': cmd_fault, 'timing': cmd_timing}[a.cmd](a)
    print('%s: %d check(s) failed' % ('FAIL' if fails else 'PASS', fails))
    return 1 if fails else rc


if __name__ == '__main__':
    sys.exit(main())
