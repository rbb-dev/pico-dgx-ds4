#!/usr/bin/env python3
"""The memory watch rides the decoding run: rank 0 toggles kCtlWatch of its control word, every rank reads
its memory after the step that shows the other value, and no run ends for it. GPU test, written and not run.

Two arms (exit 0 pass, 1 a failure, 2 the run did not reach what it tests):
  count   the correctness window (counts, no timing). Start the engine with the serving settings and the watch on (--pool-free-gib as
          production), the front end stopped, the dashboard open. Note rank 0's log size, then run ONE decode run of the seeded workload:
              a decoding run of 8 users on the seeded workload (--set none: no image set)
          Then `count --log rank0.log --from-offset N` reads the log lines written since. It passes when the first wave's batch lines equal the
          control's (rank 0's "serve: batch of 8" lines in order): steps 215 / 33 / 4 / 28 / 13 / 20 / 22 / 14 and tokens 3,113 / 463 / 34 /
          399 / 159 / 263 / 295 / 175 (the 27 of 27 runs of the nine other builds), and no batch line of 8 whose entries end "paused" is
          followed by neither an admission, a CONFIG nor a change (a pause for the watch, the rule of revision 4). Before the first
          comparison the run must be 12+ s long (the watch was due at least once): it prints how many watch readings rank 0 made.
  sixty   8 replies that do not end (the connection's own submits, max_new 100000, as tenant_test's Conn) for --seconds (60) beside the
          watch; STATS' four MemAvailable readings change at least every --every-s (11) seconds (STATS read each second). Then the arm
          cancels the eight requests and waits up to 15 s for the batch line of their end (a batch line is written when its run ends), and
          judges the log since --from-offset: exactly one batch line of 8, its steps x its median close to the arm's length, every entry
          ending "cancelled", none "paused". The bit-equal probe and the four equal digests (DSGB10_STORE_CHECK=1) are the caller's, after
          this arm (a probe is an admission). The neutrality row (N1) with the watch off is the caller's too.
  prompt  "a prompt read alone". What it proves: while one cold prompt of --prompt-tokens
          (463,939) is read alone, the memory watch is read inside the chunks' commands and no command of its own stands between two
          chunks. Settings: the serving start with the watch on (--pool-free-gib as production), DSGB10_STORE_CHECK=1 on every rank (the
          watch then prints `serve: memory watch read in a prompt's chunk` or `... as a command between turns` on rank 0), the front end
          stopped, nothing else running. It passes when, inside the prompt (rank 0's log between the size before the submit and the size at
          DONE): STATS' four mem_avail_bytes changed at least 3 times, the log holds at least 3 "in a prompt's chunk" lines and none "as a
          command between turns", and the last `store digest` of every rank-log given (--rank-log R=PATH, at least two) is equal.
          Premise (exit 2 when not reached): the prompt ran at least 30 s (the watch's period is 10 s) and the turn line says 8+ chunks.
Settings needed: all four Sparks, a build with the watch riding the run (rank 0's log on this Spark: run on rank 0's Spark), the engine the only front-end connection,
a quiet window. `count` needs only the log and the window's times; `sixty` needs --socket and --tokens (a token file as serve_test's).
  watch_rides_test.py count --log ~/dsgb10/run/rank0.log --from-offset N
  watch_rides_test.py sixty --socket S --tokens T.npy --log ~/dsgb10/run/rank0.log --from-offset N [--seconds 60] [--every-s 11]
  watch_rides_test.py prompt --socket S --tokens T.npy --log ~/dsgb10/run/rank0.log --rank-log 1=PATH --rank-log 2=PATH ... [--prompt-tokens 463939]
A comparison is made on two logs, one of the build and one of the control (the step lines of each run's
run's batch lines).
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

fails = 0
CONTROL_STEPS = [215, 33, 4, 28, 13, 20, 22, 14]
CONTROL_TOKENS = [3113, 463, 34, 399, 159, 263, 295, 175]
BATCH = re.compile(r'serve: batch of (\d+): (\d+) steps in ([\d.]+) ms, ([\d.]+) ms a step.*?req, tokens, end: (.*?)(?:; drafts|$)')


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def lines_since(path, offset):
    with open(os.path.expanduser(path), 'rb') as f:
        f.seek(offset)
        return f.read().decode('utf-8', 'replace').splitlines()


def batches(lines):
    """(line index, entries, steps, [(req, tokens, end)]) for every batch line"""
    out = []
    for i, ln in enumerate(lines):
        m = BATCH.search(ln)
        if m:
            out.append((i, int(m.group(1)), int(m.group(2)), re.findall(r'(\d+) (\d+) (\w+)', m.group(5)), float(m.group(3)), float(m.group(4))))
    return out


def arm_count(a):
    lines = lines_since(a.log, a.from_offset)
    b = batches(lines)
    watches = sum(1 for ln in lines if 'memory: rank' in ln)
    print('  %d batch lines since the offset' % len(b))
    wave = [x for x in b if x[1] == 8][:len(CONTROL_STEPS)]
    if len(wave) < len(CONTROL_STEPS):
        print('  the run did not reach the control\'s first wave (%d batch lines of 8, %d wanted)' % (len(wave), len(CONTROL_STEPS)))
        return 2
    steps = [x[2] for x in wave]
    toks = [sum(int(t) for _, t, _ in x[3]) for x in wave]
    check(steps == CONTROL_STEPS, 'the opening batch lines hold the control\'s steps %s (got %s)' % (CONTROL_STEPS, steps))
    check(toks == CONTROL_TOKENS, 'and its tokens %s (got %s)' % (CONTROL_TOKENS, toks))
    # a batch line of 8 ending in "paused" only for an admission, a CONFIG or a change: an admission shows as the next run's line with its
    # own prefill/submit text, a CONFIG or a change as its own line; a pause for the watch has none of them before the next batch line.
    bad = 0
    for k, (i, n, st, ends, _tot, _med) in enumerate(b):
        if n != 8 or not any(e == 'paused' for _, _, e in ends):
            continue
        j = b[k + 1][0] if k + 1 < len(b) else len(lines)
        between = lines[i + 1:j]
        if not any(re.search(r'config|lend command|tenant|admit|submit|prefill|serve: request', x, re.I) for x in between):
            bad += 1
            print('    batch line %d paused with nothing after it: %s' % (i, lines[i][:140]))
    check(bad == 0, 'no batch line of 8 "paused" without an admission, a CONFIG or a change after it (%d found)' % bad)
    print('  (the run was long enough for %d "memory: rank" lines; the watch readings themselves print only on a new low)' % watches)
    return 1 if fails else 0


def arm_sixty(a):
    from tenant_test import Conn  # noqa: E402
    from image_store_test import need  # noqa: E402
    from stats_test import NAMES  # noqa: E402
    idx = [NAMES.index('mem_avail_bytes_%d' % r) for r in range(4)]
    c = Conn(a.socket, a.tokens, 8)
    need(c.stats() is not None, 'the engine answers STATS')
    t0 = time.time()
    last, since, worst = None, t0, 0.0
    while time.time() - t0 < a.seconds:
        c.pump(time.time() + 1.0)
        s = c.stats()
        if not s:
            continue
        cur = tuple(s[i] for i in idx)
        if cur != last:
            last, since = cur, time.time()
        worst = max(worst, time.time() - since)
    check(worst <= a.every_s, 'STATS\' four readings changed at least every %.0f s (longest unchanged stretch %.1f s)' % (a.every_s, worst))
    t_arm = time.time() - t0
    import struct  # noqa: E402
    from serve_test import CANCEL  # noqa: E402
    for r in c.rids:
        c.e.send(CANCEL, struct.pack('<Q', r))
    n8 = []
    t_end = time.time() + 15
    while time.time() < t_end:
        c.pump(time.time() + 0.5)
        n8 = [x for x in batches(lines_since(a.log, a.from_offset)) if x[1] == 8]
        if n8:
            break
    time.sleep(1)
    n8 = [x for x in batches(lines_since(a.log, a.from_offset)) if x[1] == 8]
    print('  batch lines of 8 since the offset: %d' % len(n8))
    check(len(n8) == 1, 'exactly one batch line of 8 for the arm (no run ended for the watch)')
    if n8:
        _i, _n, steps, ends, tot_ms, med_ms = n8[0]
        print('  that line: %d steps in %.1f ms (%.2f ms a step median); the arm ran %.1f s' % (steps, tot_ms, med_ms, t_arm))
        check(abs(tot_ms / 1e3 - t_arm) <= 0.1 * t_arm + 3 and abs(steps * med_ms / 1e3 - t_arm) <= 0.25 * t_arm + 3,
              'it covers the arm (its time, and its steps x its median, close to the arm\'s length)')
        check(len(ends) == 8 and all(e == 'cancelled' for _, _, e in ends), 'every entry ends "cancelled"')
        check(not any(e == 'paused' for _, _, e in ends), 'none paused')
    return 1 if fails else 0


def arm_prompt(a):
    import random
    from serve_test import Engine, load_npy_i32  # noqa: E402
    from stats_test import NAMES, S0, frames_during, turn_chunks  # noqa: E402
    log0 = os.path.expanduser(a.log)
    off0 = os.path.getsize(log0)
    base = [int(x) for x in load_npy_i32(a.tokens)]
    prompt = [base[i % len(base)] for i in range(a.prompt_tokens)]
    prompt[0] = random.randrange(1000, 100000)  # cold: nothing of it is in the store
    e = Engine(a.socket)
    rid, _ = e.submit(S0 + 0x5b5b, prompt, max_new=1, keep=False)
    t0 = time.time()
    fr = frames_during(e, rid, 900)
    took = time.time() - t0
    chunks = turn_chunks(a.log, off0, rid)  # the turn line is written as the request ends: waited for
    lines = lines_since(a.log, off0)
    print('  the prompt ran %.1f s; turn line chunks %s; %d STATS frames' % (took, chunks, len(fr)))
    if took < 30 or not chunks or chunks[-1] < 8:
        print('  the premise: a prompt of at least 30 s and 8 chunks')
        return 2
    idx = [NAMES.index('mem_avail_bytes_%d' % r) for r in range(4)]
    changes, last = 0, None
    for _t, v in fr:
        cur = tuple(v[i] for i in idx)
        changes += last is not None and cur != last
        last = cur
    check(changes >= 3, 'STATS\' four readings changed %d times inside the prompt (3 wanted)' % changes)
    inchunk = sum("memory watch read in a prompt's chunk" in x for x in lines)
    cmd = sum('memory watch read as a command between turns' in x for x in lines)
    check(inchunk >= 3 and cmd == 0, "rank 0's log: %d reads in a chunk (3 wanted), %d as a command between turns (none wanted)" % (inchunk, cmd))
    digests = {}
    digests[0] = re.findall(r'store digest \d+ [^:]*: ([0-9a-f]{16})', '\n'.join(lines_since(a.log, 0)))
    for spec in a.rank_log:
        r, path = spec.split('=', 1)
        with open(os.path.expanduser(path), errors='replace') as f:
            digests[int(r)] = re.findall(r'store digest \d+ [^:]*: ([0-9a-f]{16})', f.read())
    if len(digests) < 2 or any(not d for d in digests.values()):
        print('  the digests need at least two rank logs with store digest lines (DSGB10_STORE_CHECK=1)')
        return 2
    ends = {r: d[-1] for r, d in digests.items()}
    check(len(set(ends.values())) == 1, 'the last store digest of ranks %s is equal: %s' % (sorted(ends), sorted(set(ends.values()))))
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='arm', required=True)
    for name in ('count', 'sixty', 'prompt'):
        s = sub.add_parser(name)
        s.add_argument('--log', required=True)
        s.add_argument('--from-offset', type=int, default=0)
        if name == 'prompt':
            s.add_argument('--socket', required=True)
            s.add_argument('--tokens', required=True)
            s.add_argument('--prompt-tokens', type=int, default=463939)
            s.add_argument('--rank-log', action='append', default=[], metavar='R=PATH')
        if name == 'sixty':
            s.add_argument('--socket', required=True)
            s.add_argument('--tokens', required=True)
            s.add_argument('--seconds', type=float, default=60)
            s.add_argument('--every-s', type=float, default=11)
    a = p.parse_args()
    rc = arm_count(a) if a.arm == 'count' else arm_prompt(a) if a.arm == 'prompt' else arm_sixty(a)
    print('watch_rides_test %s: %s' % (a.arm, 'passed' if rc == 0 else 'FAILED' if rc == 1 else 'not reached'))
    sys.exit(rc)


if __name__ == '__main__':
    main()
