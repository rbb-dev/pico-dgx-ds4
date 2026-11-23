#!/usr/bin/env python3
"""Judge of a run that stops the engine when the ranks' spill lines differ (written, not run). The run itself, by the caller (the window's
script), on the production settings file (`--kv-split tiered`, the production pool), the front-end up, the window's own disk tag, KEEP_GIB=0, DSGB10_STORE_CHECK=1 and the store
trace on four ranks, automatic restart off:
    a decoding run of 8 users with no image set against the front-end (--front, http://127.0.0.1:8000 where the front-end runs beside the engine), twice back to back, then a third time after 30 s idle
then this script on the four ranks' logs of the whole row (rank 0's journal and the workers' logs, in rank order):
    python3 s2_row.py LOG0 LOG1 LOG2 LOG3 [--pid-before P --pid-after P] [--apport-before N --apport-after N]
Premises (exit 2 when not reached, before anything is judged): rank 0 has a `serve: split: settled:` line with marked above 0 and a ghost count above 0 between the runs;
the COUNT AND THE BYTES of the `spill:` lines since the start are equal on the four ranks (the fault: 23 lines on rank 0, none on ranks 1 to 3); a `back from disk` line of a
marked conversation. Pass: that line reads `restart point N of the N it had` with `0 pieces checked`; `returns_read` 0 (no read-back line with bytes); the four ranks' `store-check:`
lines equal line by line (the rank number removed); 0 BUG, ERROR, `differ`, `out of step`; the engine's pid unchanged; no new line in /var/log/apport.log.
"""
import argparse
import re
import sys

fails = 0


def check(c, what):
    global fails
    print('  %s: %s' % ('ok' if c else 'FAIL', what))
    fails += not c


class NotReached(Exception):
    pass


def need(c, what):
    if not c:
        raise NotReached(what)
    print('  premise ok: %s' % what)


def spill_lines(text):
    """(count, MiB) of the writer's `spill:` lines: `spill: 3.0 MiB to staging in ...`."""
    n, mib = 0, 0.0
    for m in re.finditer(r'^spill: ([\d.]+) MiB to staging', text, re.M):
        n += 1
        mib += float(m.group(1))
    return n, round(mib, 1)


def store_check(text):
    return [re.sub(r'rank \d+', 'rank N', ln) for ln in text.splitlines() if ln.startswith('store-check:')]


def judge(logs, pid_before=None, pid_after=None, apport_before=None, apport_after=None):
    sp = [spill_lines(t) for t in logs]
    need(all(s == sp[0] for s in sp) and sp[0][0] > 0, 'the count and bytes of `spill:` lines are equal on the four ranks: %s' % sp)
    m = re.findall(r'serve: split: settled: (\d+) marked, (\d+) imaged \(idle\); queued \S+; (\d+) ghosts', logs[0])
    need(any(int(a) > 0 for a, _, _ in m) and any(int(g) > 0 for _, _, g in m), 'rank 0 has a split line with marked and ghosts above 0 (%d lines)' % len(m))
    backs = re.findall(r'state: session \d+ back from disk: ([\d.]+) MiB in [\d.]+ ms \(.*?(\d+) pieces checked, \w+\); restart point (-?\d+) of the (-?\d+) it had', logs[0])
    need(len(backs) > 0, 'a `back from disk` line of a marked conversation (%d lines)' % len(backs))
    check(all(a == b and int(a) > 0 for _, _, a, b in backs), 'restart point N of the N it had: %s' % backs[-1:])
    check(any(c == '0' for _, c, _, _ in backs), 'a return with 0 pieces checked (every ghost taken back, nothing read)')
    sc = [store_check(t) for t in logs]
    check(all(s == sc[0] for s in sc) and len(sc[0]) > 0, 'the four ranks\' store-check lines equal line by line (%d lines)' % len(sc[0]))
    for pat in ('BUG', 'ERROR', 'differ', 'out of step'):
        n = sum(len(re.findall(pat, t)) for t in logs)
        check(n == 0, '%d lines %s on the four ranks' % (n, pat))
    if pid_before is not None:
        check(pid_before == pid_after, 'the engine\'s pid unchanged (%s -> %s)' % (pid_before, pid_after))
    if apport_before is not None:
        check(apport_before == apport_after, 'no new line in /var/log/apport.log (%s -> %s)' % (apport_before, apport_after))


def selftest():
    a = 'spill: 3.0 MiB to staging in 0.1 ms\nspill: 3.0 MiB to staging in 0.1 ms\nstore-check: rank 0 ride digest 1\n'
    assert spill_lines(a) == (2, 6.0)
    assert store_check(a) == ['store-check: rank N ride digest 1']
    try:
        judge([a, '', '', ''])
        return 1
    except NotReached as e:
        print('the fault\'s shape (spill lines on rank 0 only) is NOT REACHED at the second premise: %s' % e)
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='*')
    ap.add_argument('--pid-before'), ap.add_argument('--pid-after')
    ap.add_argument('--apport-before', type=int), ap.add_argument('--apport-after', type=int)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    logs = [open(f, errors='replace').read() for f in a.logs]
    if len(logs) != 4:
        print('four logs, rank order')
        return 2
    try:
        judge(logs, a.pid_before, a.pid_after, a.apport_before, a.apport_after)
    except NotReached as e:
        print('did not reach the path: %s' % e)
        return 2
    print('%d check(s) failed' % fails if fails else 'all checks passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
