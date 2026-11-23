#!/usr/bin/env python3
"""Host self-test of split_test.py's wait for the split (no GPU, no sockets): wait_split_quiet against a fake Decoders whose STATS
give the pieces split once a second, on a clock that jumps on every pump.
  rise     0 x6, 64, 128 x4: with rise_from=0 the wait returns the 128 record, not the quiet before the split began
  never    the count never rises: NotReached when `seconds` have passed
  python3 split_wait_selftest.py [FILE]     FILE is the split_test.py to check (default: the one beside this file)
"""
import importlib.util
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'split_test.py')
sys.path.insert(0, HERE)
fails = 0


def report(case, ok, what):
    global fails
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', case, what))
    fails += not ok


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t


class Fake:
    """the Decoders' pump and poll_stats as the wait uses them: each pump moves the clock to its deadline, each poll answers the next record"""
    def __init__(self, clock, counts):
        self.clock, self.counts, self.i = clock, list(counts), 0

    def pump(self, until, *a):
        self.clock.t = max(self.clock.t, until)

    def poll_stats(self):
        c = self.counts[min(self.i, len(self.counts) - 1)]
        self.i += 1
        return {'split_pieces_total': c}


def load():
    spec = importlib.util.spec_from_file_location('split_test_under_test', FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wait(mod, d, seconds):
    # a wait without rise_from (the function before it) is asked the same question without the argument
    if 'rise_from' in inspect.signature(mod.wait_split_quiet).parameters:
        return mod.wait_split_quiet(d, seconds, rise_from=0)
    return mod.wait_split_quiet(d, seconds)


def case_rise(mod):
    clock = Clock()
    mod.time = clock
    d = Fake(clock, [0, 0, 0, 0, 0, 0, 64, 128, 128, 128, 128])
    try:
        s = wait(mod, d, 120)
    except Exception as x:  # a function without rise_from raises TypeError
        report('rise', False, 'raised %s: %s' % (type(x).__name__, x))
        return
    report('rise', s['split_pieces_total'] == 128, 'returned the record with %d pieces (want 128)' % s['split_pieces_total'])


def case_never(mod):
    clock = Clock()
    mod.time = clock
    d = Fake(clock, [0])
    try:
        wait(mod, d, 20)
        report('never', False, 'returned for a count that never rose')
    except mod.NotReached as x:
        report('never', 18 <= clock.t <= 24, 'NotReached at %.0f s (seconds 20): %s' % (clock.t, x))
    except Exception as x:
        report('never', False, 'raised %s: %s' % (type(x).__name__, x))


def case_kick(mod):
    clock = Clock()
    mod.time = clock
    d = Fake(clock, [0] * 4 + [64, 64, 64, 64, 64, 64])
    kicked = []
    try:
        s = mod.wait_split_quiet(d, 120, rise_from=0, kick=lambda: kicked.append(clock.t))
    except Exception as x:
        report('kick', False, 'raised %s: %s' % (type(x).__name__, x))
        return
    report('kick', len(kicked) == 1 and kicked[0] >= 2.0 and s['split_pieces_total'] == 64, 'the kick came once, %.0f s after the start, and the split was then waited for' % (kicked[0] if kicked else -1))


m = load()
case_kick(load())
case_rise(m)
case_never(load())
print('FAILED' if fails else 'all passed')
sys.exit(1 if fails else 0)
