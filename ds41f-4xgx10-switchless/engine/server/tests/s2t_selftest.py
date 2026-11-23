#!/usr/bin/env python3
"""Host self-test of the Engram hand-over test's and the image store test's own checks, against simulated engines (no GPU, no
sockets). Each case runs the real test code with its engine-facing calls replaced:
  fault B   the engine refuses the change at once and keeps it pending for its back-off (300 s): the test must check the
            refusal and the unchanged size, not wait for a landing (a clock that jumps on every sleep makes a wait visible)
  fault C   the same for step C
  back      the starting size is the 4096 MiB the start asked for, which the cache holds as whole 264-byte slots (a hair under
            4096 MiB): the size back after the grow and the shrink is compared like with like
  leak      a rank's MemAvailable falls with the test holding nothing more: the image encoder container's device memory (nvidia-smi
            stub, cgroup not pico-engine) is taken out, and the engine's own must not grow
  python3 s2t_selftest.py [DIR]     DIR holds the two tests (default: this file's folder)
"""
import importlib.util
import os
import sys

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)
fails = 0


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DIR, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def report(case, ok, what):
    global fails
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', case, what))
    fails += not ok


MIB = 1048576
ROW = 264


def held(mib):  # the bytes of a cache asked for `mib` MiB: whole slots
    return (mib * MIB // ROW) * ROW


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Sim:
    """the engine's hand-over as the test sees it through STATS"""
    def __init__(self, start_mib, fail):
        self.cache = held(start_mib)
        self.groups = 14
        self.ok, self.failed, self.owed, self.pending = 1, 0, 0, 0
        self.fail = fail
        self.goal = None

    def config(self, mib):
        if mib == self.cache // MIB or abs(self.cache / MIB - mib) <= 0:
            return
        if self.fail:
            self.failed += 1
            self.pending = mib  # waiting for the back-off
        else:
            self.cache = held(mib)
            self.ok += 1
            self.pending = 0

    def stats(self):
        return dict(cache_bytes=self.cache, groups=self.groups, pending_mib=self.pending, lookups=0, hits=0, ok=self.ok,
                    failed=self.failed, owed=self.owed, huge=0, evictions=0)


def run_fault(case, step):
    mod = load('engram_lend_test')
    clock = Clock()
    mod.time = clock
    sim = Sim(4096, fail=step in ('B', 'C'))
    mod.stats = lambda e: (clock.sleep(0.6), sim.stats())[1]
    mod.config_mib = lambda e, mib: (mod.ASKED.__setitem__(0, mib), sim.config(mib))[0]
    mod.Engine = lambda path: type('E', (), {'run': lambda self, *a, **k: {'error': '', 'ids': [1]}})()
    mod.load_npy_i32 = lambda p: list(range(4096))
    mod.meminfo = lambda hosts: [10000.0 for _ in hosts]
    mod.probe = lambda a, out: True
    mod.compare = lambda x, y: True
    args = type('A', (), dict(socket='s', tokens='t', mib=6912, step=step, before='', settle=30, hosts=[], tol_mib=256))()
    mod.fails = 0
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.cmd_fault(args)
        out, err = buf.getvalue(), ''
    except SystemExit as x:
        out, err = buf.getvalue(), str(x)
    ok = not err and mod.fails == 0
    report(case, ok, 'no failed check, no timeout (simulated %.0f s)%s%s' % (clock.t, '; ' + err if err else '', '; %d checks failed' % mod.fails if mod.fails else ''))
    if not ok:
        print(out)


def run_back():
    mod = load('engram_lend_test')
    clock = Clock()
    mod.time = clock
    sim = Sim(4096, fail=False)
    mod.stats = lambda e: (clock.sleep(0.6), sim.stats())[1]
    mod.config_mib = lambda e, mib: (mod.ASKED.__setitem__(0, mib), sim.config(mib))[0]
    mod.Engine = lambda path: type('E', (), {'run': lambda self, *a, **k: {'error': '', 'ids': [1]}})()
    mod.load_npy_i32 = lambda p: list(range(4096))
    mod.meminfo = lambda hosts: []
    args = type('A', (), dict(socket='s', tokens='t', mib=6912, step='back', before='', settle=30, hosts=[], tol_mib=256))()
    mod.fails = 0
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            mod.cmd_fault(args)
        except SystemExit as x:
            buf.write(str(x))
    bad = [l for l in buf.getvalue().splitlines() if 'FAIL' in l]
    report('fault back', mod.fails == 0, 'the size back is the starting size%s' % (': ' + bad[0].strip() if bad else ''))


def run_leak(case, want_fail=False, cgroup=4136, vmrss=3973, gpu=98770, worker_fall=0.0):
    """A rank's MemAvailable falls 14.24 -> 13.22 GiB with the engine's terms flat (cgroup 4,136 -> 4,142 MiB, VmRSS 3,973 ->
    3,983, GPU 98,770 -> 98,772), the store back: passes. An engine term past the tolerance, or a worker's fall, fails."""
    mod = load('image_store_test')
    snaps = iter([dict(cgroup=4136, vmrss=3973, gpu_engine=98770, gpu_encoder=100, client=500.0, meminfo={'MemFree': 1, 'Cached': 2}),
                  dict(cgroup=cgroup + 6, vmrss=vmrss + 10, gpu_engine=gpu + 2, gpu_encoder=1189, client=520.0, meminfo={'MemFree': 1, 'Cached': 3})])
    mod.local_snapshot = lambda: next(snaps)
    mem = iter([[14.24, 11.31], [13.22, 11.31 - worker_fall]])
    mod.mem_available = lambda nodes: next(mem)
    mod.need_s3 = lambda e: {'pool_free': 8245, 'image_numbers': 0}
    mod.big = lambda e, T, s, n=36, **k: ((1, 2, 3), 39600)
    mod.collect = lambda e, req: {'reason': 'length', 'ids': [1] * 8, 'prefill_us': 1e7, 'error': ''}
    mod.stats = lambda e: (0, 0, {'image_numbers': 0, 'pool_free': 8245})
    mod.time = type('T', (), {'sleep': staticmethod(lambda s: None), 'time': staticmethod(lambda: 0.0)})
    seen = iter([False, True])
    e = type('E', (), {'done_seen': lambda self, rid: next(seen), 'numbers': lambda self: 195})()
    args = type('A', (), dict(nodes='local,user@n2', rows_spans=36, spread=0.15, engine_tol_mib=256))()
    mod.fails = 0
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.t_leak(args, e, list(range(100)))
    bad = [l.strip() for l in buf.getvalue().splitlines() if 'FAIL' in l]
    report(case, (mod.fails > 0) == want_fail, (bad[0] if bad else 'no failed check'))


run_fault('fault B', 'B')
run_fault('fault C', 'C')
run_back()
run_leak('leak: MemAvailable of one rank -1.02 GiB, engine flat')
run_leak('leak, engine cgroup grows 400 MiB', True, cgroup=4536)
run_leak('leak, rank 0 VmRSS grows 400 MiB', True, vmrss=4373)
run_leak('leak, engine GPU memory grows 400 MiB', True, gpu=99170)
run_leak('leak, a worker falls 1 GiB', True, worker_fall=1.0)
print('FAILED' if fails else 'all passed')
sys.exit(1 if fails else 0)
