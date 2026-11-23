#!/usr/bin/env python3
"""Host self-test of split_test.py's judgement of a return from the SSD (no GPU, no sockets): ssd_return_checks, gap_verdict and
gave_back_lines against the numbers of a run whose 1M conversation came back cold.
  cold      TTFT 370.7 s fails; the read-back line `restart point -1 of -1` or none fails
  twice     13,120 MiB exchanged against 600 pieces split (1,251 MiB) fails; a return of 565 MiB for 271 pieces passes
  good      TTFT 4 s, restart point 1000000 of 1000000, 565 MiB for 271 pieces split: all three pass
  gap       one of the test's decoders' lines given back in the window makes the gap not judged (and not failed); a filler's does not
  nopause   serve.cpp holds no split_pause_due, split_ended_ or kv_split_every, and lend_pause_due names no split
  runs      a journal sample with a `split:` line between two 6-step `batch of 16` lines counts one run ended for a split, an `off` sample none
  python3 split_return_selftest.py [FILE [SERVE]]     FILE is the split_test.py to check (default: the one beside this file), SERVE the serve.cpp
"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JOURNALS = os.environ.get('PICO_JOURNALS', os.path.join(HERE, 'journals'))  # recorded journals, when present: the checks that read them are skipped without
FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'split_test.py')
SERVE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'serve.cpp')
sys.path.insert(0, HERE)
fails = 0
MIB = 1 << 20


def report(case, ok, what):
    global fails
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', case, what))
    fails += not ok


spec = importlib.util.spec_from_file_location('split_test_under_test', FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SPLIT = m  # the module under test (m is used again below)

oks = lambda r: [ok for ok, _ in r]
cold = m.ssd_return_checks(370.7, 30, (-1, -1), 13120 * MIB, 271)
report('cold', oks(cold)[0] is False, 'TTFT 370.7 s fails')
report('cold', oks(cold)[1] is False, 'restart point -1 of -1 fails (equal, but nothing to restart from)')
report('cold', m.ssd_return_checks(4, 30, None, 565 * MIB, 271)[1][0] is False, 'no read-back line fails')
report('cold', m.ssd_return_checks(4, 30, (-1, 1000000), 565 * MIB, 271)[1][0] is False, 'restart point -1 of 1000000 fails')
twice = m.ssd_return_checks(4, 30, (1000000, 1000000), 13120 * MIB, 600)
report('twice', oks(twice) == [True, True, False], '13,120 MiB exchanged against 600 pieces split fails only the third check: %s' % oks(twice))
good = m.ssd_return_checks(4, 30, (1000000, 1000000), 565 * MIB, 271)
report('good', all(oks(good)), 'TTFT 4 s, 1000000 of 1000000, 565 MiB for 271 pieces: all pass')
report('good', m.ssd_return_checks(4, 30, (1000000, 1000000), 271 * 3 * 729088, 271)[2][0], 'exactly the quarters of the pieces split passes')

DEC = {0x73706400 + k for k in range(8)}
line = 'serve: slot %d (session %016x) gives its 10 chunks back to the pool for a conversation read back\n'
text = ''.join(line % (k, 0x73706400 + k) for k in (0, 3)) + line % (9, 0x73706c01)  # two decoders, one filler
report('gap', m.gave_back_lines(text, DEC) == 2, 'two decoders\' lines counted, the filler\'s not: %d' % m.gave_back_lines(text, DEC))
report('gap', m.gave_back_lines(line % (9, 0x73706c01), DEC) == 0, 'a filler\'s line alone: 0')
j, ok, what = m.gap_verdict(217.086, 0.259, 250, 2)
report('gap', (j, ok) == (False, True) and 'not judged' in what, 'decoder lines given back: not judged, not failed: %s' % what)
j, ok, what = m.gap_verdict(217.086, 0.259, 250, 0)
report('gap', (j, ok) == (True, False), 'none given back: judged, and 217,086 ms fails')
j, ok, what = m.gap_verdict(0.4, 0.259, 250, 0)
report('gap', (j, ok) == (True, True), 'none given back, 400 ms against 259 + 250: passes')
# the premise of the bookmarks (WR4A: 1,000,000 positions, 36 window snapshots, 348 slots)
report('split', m.split_wanted(1000000) == 3 * 62 + 123, 'a 1M conversation: 309 lists (3 x 62 + 123): %d' % m.split_wanted(1000000))
ok, what = m.bookmarks_premise(345, 1000000, 36)
report('split', ok and '345 slots, 309 lists + 36 bookmarks' in what, 'bookmarks_premise(345, 1M, 36) is reached: %s' % what)
ok, what = m.bookmarks_premise(290, 1000000, 36)
report('split', not ok, 'bookmarks_premise(290, 1M, 36) is not reached: %s' % what)
report('split', not m.bookmarks_premise(345, 1000000, None)[0], 'no req line in the log: not reached')
req = ('serve: req 9 session 0000000073706fe8: prompt 1000000, reused 0 (cold), prefill 335387.6 ms (2982 tok/s), 1 tokens in 3.2 ms (0.0 tok/s), length; '
       'store 0.0 MiB; session 1000000 positions, 1059.7 MiB, 36 window snapshots\n')
report('split', m.window_snapshots('x\n' + req, 0x73706fe8, 1000000) == 36 and m.window_snapshots(req, 0x73706fe9, 1000000) is None, 'the snapshots of the conversation\'s own req line: 36; another session: none')
# decoders ended before the return: the gap is never "ok" on an empty window
j, ok, what = m.gap_verdict(0.0, 0.259, 250, 0, 8, 8, 0)
report('ended', (j, ok) == (False, True) and '8 of 8 decoders had ended' in what, 'eight decoders ended: not judged: %s' % what)
j, ok, what = m.gap_verdict(0.0, 0.259, 250, 0, 0, 8, 0)
report('ended', j is False and 'no decoder token' in what, 'no token gap in the window: not judged: %s' % what)
j, ok, what = m.gap_verdict(0.4, 0.259, 250, 0, 3, 8, 40)
report('ended', (j, ok) == (True, True), 'three of eight ended, 40 gaps in the window: judged, passes')
# every BUG line counts
log = 'serve: BUG: idle line holds a reservation (103 numbers reserved, nothing in flight)\nserve: ok\n'
report('bug', m.bug_lines(log) == 1 and m.bug_lines('serve: ok\n') == 0, 'a log with one BUG: idle line line counts 1, a clean one 0')
# no split command ends a decoding run: the pause path is gone
src = open(SERVE, errors='replace').read()
for word in ('split_pause_due', 'split_ended_', 'kv_split_every'):
    report('nopause', word not in src, '%s: %d hits in serve.cpp' % (word, src.count(word)))
i = src.index('bool lend_pause_due(')  # its arguments vary by build: the name and the bracket are enough
ends = [j for j in (src.find('bool step_hook(', i), src.find('void lend_tick()', i)) if j > 0]  # its body ends at the next function
body = src[i:min(ends)]
code = '\n'.join(ln.split('//')[0] for ln in body.splitlines())  # comments aside
report('nopause', 'split' not in code, 'lend_pause_due names no split in its code')
# a journal sample: 6-step runs of 16 with a split between them (the 16-user run of a cache start), against `off`'s one long run
cache = ('serve: batch of 16: 6 steps in 841.3 ms\nserve: session 0000000073706c01 split: 4 pieces; 12 quarter slots in 3 numbers, 0 ghosts\n'
         'serve: spill: 3.0 MiB to staging in 0.29 ms\nserve: batch of 16: 7 steps in 905.0 ms\nserve: batch of 16: 6 steps in 840.0 ms\n'
         'serve: session 0000000073706c02 split: 1 pieces; 13 quarter slots in 4 numbers, 0 ghosts\nserve: batch of 16: 6 steps in 842.0 ms\n')
off = 'serve: batch of 16: 154 steps in 21801.1 ms\nserve: spill: 3.0 MiB to staging in 0.29 ms\nserve: batch of 16: 156 steps in 22000.0 ms\n'
report('runs', m.runs_ended_for_split(cache) == 2 and m.runs_ended_for_split(off) == 0, 'the cache sample counts %d runs ended for a split, the off sample %d' % (
    m.runs_ended_for_split(cache), m.runs_ended_for_split(off)))
# 5.14: nothing between a prompt's chunks, at an admission, in finish() or in persist_slot: those functions reach no ride, split or join command
def body_of(text, head):
    i = text.index(head)
    depth, j = 0, text.index('{', i)
    k = j
    while True:
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[j:k + 1]
        k += 1
for head in ('void il_pass()', 'void admit(Request&& rq', 'void finish(Flight& f', 'void persist_slot(int s', 'Begin pf_begin('):
    b = '\n'.join(ln.split('//')[0] for ln in body_of(src, head).splitlines())
    hits = [w for w in ('kStRide', 'kStSplit', 'kStJoin', 'kStFlush', 'imager_tick', 'ride_split(') if w in b]
    report('ride', not hits, '%s reaches no ride, split, join or flush command (found %s)' % (head, hits or 'none'))
i = src.index('void split_tick()')
b = src[i:src.index('void imager_tick()', i)] if 'void imager_tick()' in src[i:] else ''
# the loop's top asks idle_ok before any command: imager_tick returns at once unless nothing is in flight
it = body_of(src, 'void imager_tick()')
report('ride', 'idle_ok' in it and it.index('idle_ok') < it.index('kStRide'), 'imager_tick asks idle_ok before it sends a ride command')
# the device's image calls name no model stream: image_quarter and image_bookmark queue on sg
dev = open(os.path.join(HERE, '..', '..', 'model', 'state.cu'), errors='replace').read()
for head in ('uint64_t image_quarter(', 'uint64_t image_bookmark('):
    b = '\n'.join(ln.split('//')[0] for ln in body_of(dev, head).splitlines())
    report('stream', 'm.st' not in b and 'sg' in b, '%s names no m.st and queues on sg' % head.split('(')[0].split()[-1])
# A2: the only wait between the two streams is sg's on a marker of m.st; no cudaStreamSynchronize(sg) outside image_wait, disk_grow, disk_shrink
code = '\n'.join(ln.split('//')[0] for ln in dev.splitlines())
report('stream', 'cudaStreamWaitEvent(m.st' not in code, 'no cudaStreamWaitEvent(m.st: the model\'s stream waits for nothing')
sync_sg = [i for i in range(len(code)) if code.startswith('cudaStreamSynchronize(sg)', i)]
bad = [i for i in sync_sg if not any(f in code[max(0, i - 1500):i] for f in ('void image_wait()', 'bool disk_grow(', 'bool disk_shrink(', 'void warm_kernels('))]
report('stream', not bad, 'cudaStreamSynchronize(sg) only in image_wait, disk_grow, disk_shrink and the start warm_kernels (%d outside)' % len(bad))
# rescue( is reached only behind !imager_work(): the admit path waits while images can free what the request needs
i = src.index('void admit(Request&& rq')
ab = body_of(src, 'void admit(Request&& rq')
report('rescue', 'imager_work()' in ab and ab.index('imager_work()') < ab.index('rescue(rq'), 'admit asks imager_work() before it reaches rescue')
li = src.index('rescued_req_ = wait_.front().h.req_id')
report('rescue', '!imager_work()' in src[li - 400:li], 'the loop\'s rescue branch is behind !imager_work()')
# --kv-split-after-s is read by imager_pick/imager_work (imager_aged), its default is 0
report('after', 'kv_split_after_s' in body_of(src, 'bool imager_aged(') , 'imager_aged reads kv_split_after_s')
mainsrc = open(os.path.join(HERE, '..', 'main.cpp'), errors='replace').read()
report('after', 'double kv_split_after_s = 0;' in mainsrc, 'main.cpp: --kv-split-after-s defaults to 0')
# the ride's ring is connected beside the watch's, and a ride without it dies on an engine that has its ring
cmdsrc = open(os.path.join(HERE, '..', 'cmd.cpp'), errors='replace').read()
m = mainsrc
report('wire', 'split_ride_init(&ring)' in m and m.index('split_ride_init(&ring)') - m.index('mem_watch_init(&ring') < 400 and m.index('split_ride_init(&ring)') > m.index('mem_watch_init(&ring'),
       'main.cpp calls split_ride_init(&ring) beside mem_watch_init(&ring, ...)')
report('wire', 'has no ring for the ride' in cmdsrc and 'has_host_ring()' in body_of(cmdsrc, 'void split_ride_run('), 'split_ride_run dies when the engine has a ring and the ride has none')
report('wire', 'all[0] = mine;  // no ring: one rank (a test)' not in cmdsrc, 'the silent one-rank branch is gone')
# the dead code is gone: Model::state_image, state_settle, StoreCore::clean_idle_numbers
allsrc = ''.join(open(os.path.join(HERE, '..', '..', 'model', f), errors='replace').read() for f in ('model.h', 'state.cu', 'store_core.h'))
report('dead', 'state_image(' not in allsrc and 'state_settle(' not in allsrc and 'clean_idle_numbers' not in allsrc, 'no Model::state_image, state_settle or clean_idle_numbers')
# every host all-gather call names itself (an id other than the default)
import glob
unnamed = []
for f in glob.glob(os.path.join(HERE, '..', '*.cpp')) + glob.glob(os.path.join(HERE, '..', '..', 'model', '*.cu')):
    txt = open(f, errors='replace').read()
    for mm in re.finditer(r'allgather_host\(([^;]*)\);', txt):
        args = mm.group(1)
        if args.count(',') < 3:
            unnamed.append(os.path.basename(f) + ': ' + args[:60])
report('sites', not unnamed, 'every allgather_host call carries its id (%s)' % (unnamed or 'none unnamed'))
imager_mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location('split_test_b', FILE))
importlib.util.spec_from_file_location('split_test_b', FILE).loader.exec_module(imager_mod)
# row b's expectation, offline, on recorded numbers: the lists' pieces and 36 window bookmarks of a conversation still on its line are
# 345 x 786,432 = 271,319,040 B written; the end bookmark's 1,572,864 B comes only when a line lets its conversation go into the store. Row b judges the first (no release: a
# client RELEASE deletes the conversation); the full figure 272,891,904 is for a conversation that left its line.
report('rowb', 345 * 786432 == 271319040 and imager_mod.imager_expected_bytes(1000000, 36) == 271319040 and imager_mod.imager_expected_bytes(1000000, 36, True) == 272891904,
       'row b\'s expectation for a conversation on its line is the recorded 271,319,040 B (345 x 786,432); with the end bookmark 272,891,904')
ts = open(FILE, errors='replace').read()
bi = ts[ts.index("elif a.event == 'b':"):ts.index('else:  # qi / qw / qd / qs')]
report('rowb', 'e.release(' not in bi and 'imager_expected_bytes(a.size, snaps0)' in bi and 'imager_expected_bytes(a.size, snaps, True)' not in bi, 'event b releases nothing and judges no end bookmark')
uses = [ln for ln in ts.splitlines() if 'imager_expected_bytes(' in ln and 'def ' not in ln]
report('rowb', len(uses) == 2 and all(', True)' not in ln for ln in uses), 'every use of imager_expected_bytes (%d lines, the two calls of event b) counts no end bookmark' % len(uses))
# the imager rows read the shared connection through one demultiplexer. Two requests' frames interleaved on one connection (a decoder's request 9 and the
# return turn 10): the demultiplexer routes each frame to its request; today's Engine.collect asserts on the first frame that is not its own (AssertionError (9, 10)).
import socket
import time
import struct


def _frames(items):
    out = b''
    for t, p in items:
        out += struct.pack('<IHH', len(p), t, 0) + p
    return out


def _tok(r, ids):
    return struct.pack('<QII', r, len(ids), 0) + struct.pack('<%dI' % len(ids), *ids)


def _done(r, cached):
    return struct.pack('<QIIIIQQQII', r, 0, 100, cached, 3, 0, 0, 0, 0, 0)


def _engine(frames):
    a, b = socket.socketpair()
    e = imager_mod.Engine.__new__(imager_mod.Engine)
    e.s, e.buf, e.next_id = a, b'', 10
    b.sendall(_frames(frames))
    return e, a, b


TK, DN = imager_mod.TOKENS, imager_mod.DONE
inter = [(TK, _tok(9, [1, 2])), (TK, _tok(10, [7])), (DN, _done(9, 0)), (TK, _tok(10, [8, 9])), (DN, _done(10, 5))]
e, a, b = _engine(inter)
dd = imager_mod.Decoders(e, [], 0, 0)
o = imager_mod.demux_turn(dd, 0x1234, [1, 2, 3])
report('demux', o['ids'] == [7, 8, 9] and o['cached'] == 5, 'interleaved frames of requests 9 and 10: the turn (10) gets its ids %s and cached %s, request 9\'s frames go to its own' % (o['ids'], o['cached']))
e2, a2, b2 = _engine(inter)
try:
    e2.collect((10, 0))
    red = False
except AssertionError:
    red = True
report('demux', red, 'RED on today\'s Engine.collect: it asserts on request 9\'s frame while collecting request 10 (a recorded crash)')
# the exchange row sends fillers until the free numbers are gone. On a recorded run (engine-exch.log: free and unreserved 479 to 667 of 785, ghosts 384) one filler of
# 131,072 positions could not take a ghost; the row now needs this many fillers (it reads STATS before each and stops when ghosts fell).
rec = os.path.join(JOURNALS, 'engine-exch.log')
if os.path.exists(rec):
    frees = [int(x) for x in re.findall(r'(\d+) free and unreserved', open(rec, errors='replace').read())]
    need_f = imager_mod.exchange_fillers(max(frees), 0, 384, 131072 // 2900)
    report('exch', need_f > 1, 'recorded free %d..%d of 785, ghosts 384: %d fillers are needed before one takes a ghost; the row sent 1' % (min(frees), max(frees), need_f))
else:
    print('SKIP exch: %s not here' % rec)
report('exch', imager_mod.exchange_fillers(100, 0, 100, 45) == 1 and imager_mod.exchange_fillers(400, 20, 100, 45) == 7, 'exchange_fillers: no plain free numbers: 1; 280 plain at 45 a filler: 7')
# run() sets the stall from the two functions; the idle tick, the age and the ride read it; each ride lambda is watched; every store kernel is in the start's table
rb = body_of(src, 'void run()')
report('stall', 'splitpace::rot_stalled(' in rb and 'pool_waits_ = pool_waits' in rb and rb.count('++pool_waits') + rb.count('++ext_waits, ++pool_waits') + rb.count('start_waits = true, ++pool_waits') >= 4,
       'run() counts the flights put back for the pool (window, no slot, extension, start) and sets pool_waits_ and stalled_ from rot_stalled')
report('stall', 'stalled_' in body_of(src, 'void imager_tick()'), 'imager_tick\'s idle_ok names stalled_')
report('stall', 'pool_waiter(' in body_of(src, 'bool imager_aged(') and 'pool_waiter(' in body_of(src, 'void ride_split('), 'imager_aged and ride_split call pool_waiter')
rides = re.findall(r'auto ride = \[&\]\(uint32_t ctl\) \{(.*?)\n  \};', cmdsrc, re.S)
report('watch', len(rides) == 4 and all(r.count('wd.started = now_s()') >= 2 and 'split_ride_run' in r and 'wd.started = 0' in r for r in rides), 'each of the four ride lambdas sets wd.started around the ride (%d lambdas)' % len(rides))
qsrc = open(os.path.join(HERE, '..', '..', 'model', 'quarter.cu'), errors='replace').read()
wb = body_of(qsrc, 'int warm(')
launched = set(re.findall(r'(\w+)<<<', qsrc))
miss = sorted(k for k in launched if '"%s"' % k not in wb)
report('warm', not miss, 'every kernel launched in quarter.cu (%s) is in the start\'s table (missing %s)' % (', '.join(sorted(launched)), miss or 'none'))
devsrc = open(os.path.join(HERE, '..', '..', 'model', 'state.cu'), errors='replace').read()
report('warm', 'block_sum' in body_of(devsrc, 'void warm_kernels()') and 'warm_kernels();' in body_of(devsrc, 'void prepare()'), 'block_sum is launched once by warm_kernels, which prepare() calls')
# Python binds the LAST definition of a name, and the text searches above find the first: a top-level function, class or assignment defined twice in split_test.py (a stale
# copy of a section left behind by an edit) runs the stale one unseen. The check parses the file with ast.
import ast
import collections


def duplicate_names(text):
    c = collections.Counter()
    for n in ast.parse(text).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            c[n.name] += 1
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                for x in ast.walk(tg):
                    if isinstance(x, ast.Name):
                        c[x.id] += 1
    return sorted(k for k, v in c.items() if v > 1)


report('dup', duplicate_names('def f():\n    pass\n\n\ndef f():\n    pass\nX = 1\n') == ['f'], 'control: the check finds a function defined twice in a sample')
dups = duplicate_names(ts)
report('dup', not dups, 'split_test.py defines every top-level name once (twice: %s)' % (dups or 'none'))
# a name a function reads that nothing defines is a NameError at the line that reads it, on the GPU, hours into a window (cmd_imager read `given_back`, which only cmd_gap
# assigns). Every function's code object, nested ones included: every name read with LOAD_GLOBAL or LOAD_NAME must be a module global or a builtin.
import builtins
import dis
import types


def undefined_globals(module):
    seen = set()

    def walk(code, where):
        for ins in dis.get_instructions(code):
            if ins.opname in ('LOAD_GLOBAL', 'LOAD_NAME') and ins.argval not in vars(module) and not hasattr(builtins, ins.argval):
                seen.add((ins.argval, where))
        for c in code.co_consts:
            if isinstance(c, types.CodeType):
                walk(c, c.co_name)
    walk(compile(open(module.__file__).read(), module.__file__, 'exec'), '<module>')
    return sorted(seen)


class _Mod:  # a sample: the module's globals and its source
    pass


import tempfile
with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tf:
    tf.write('def f():\n    return missing_name + 1\n\n\ndef g():\n    def h():\n        return also_missing\n    return h\n')
sample = types.ModuleType('sample')
sample.__file__ = tf.name
exec(compile(open(tf.name).read(), tf.name, 'exec'), vars(sample))
report('names', [n for n, _ in undefined_globals(sample)] == ['also_missing', 'missing_name'], 'control: the check finds a name read in a function and in a nested one that nothing defines')
und = undefined_globals(imager_mod)
report('names', not und, 'split_test.py reads no undefined name in any function (undefined: %s)' % (und or 'none'))
# a premise or a control read after the thing it measures already happened. Row a2n's order, by fakes that record the calls: its mark and first reading come before any
# window, the event window reads STATS until the ghosts, and the control window comes after the premises (a recorded run: the mark was taken after a 20 s and a 60 s window, so the rides had
# imaged everything unseen: `rides rose (0)`).
import contextlib
import io
import types as _types


class _Stop(Exception):
    pass


def a2n_calls(path):
    spec2 = importlib.util.spec_from_file_location('split_test_order', path)
    mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)
    calls = []

    class FLog:
        def __init__(self, path):
            pass

        def mark(self):
            calls.append('mark')
            return 0

        def text(self, since=None):
            return ''

    class FDec:
        stall_guard = None

        def __init__(self, e, T, sessions, new):
            self.last, self.ended, self.gaps, self.watch = {1: 1.0}, {}, [], {}
            self.polls = 0

        def pump(self, until, stop=lambda: False):
            calls.append('pump')

        def window(self, seconds):
            calls.append('window')
            raise _Stop()

        def poll_stats(self):
            self.polls += 1
            calls.append('stats')
            return {'split_rides_total': self.polls, 'ghost_numbers': 0 if self.polls < 2 else 400, 'image_pending': 0, 'kv_split': 1}

        def longest(self, a, b):
            return 0.0

    class FEng:
        def __init__(self, sock):
            pass

        def run(self, *a, **k):
            return {'error': '', 'prompt': 1, 'ids': [1]}

    mod.Engine, mod.Decoders, mod.Log = FEng, FDec, FLog
    mod.load_npy_i32 = lambda f: [1, 2, 3]
    args = _types.SimpleNamespace(event='a2n', tokens='x', log='x', size=1000000, sessions=8, decode_new=70000, window=1.0, ssd_fillers=0, socket='x', out=None, ref=None, fill=0)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            mod.cmd_imager(args)
        except _Stop:
            pass
    return calls


def a2n_order_ok(calls):
    return 'window' in calls and 'mark' in calls and calls.index('mark') < calls.index('window') and calls[:calls.index('window')].count('stats') >= 2


import os as _os
cur = a2n_calls(FILE)
report('order', a2n_order_ok(cur), 'row a2n: the mark and the first reading come before any window, the event window polls STATS, the control window comes after (calls: %s)' % ' '.join(cur))
before2 = '/tmp/split_test.earlier2.py'
if _os.path.exists(before2):
    old_calls = a2n_calls(before2)
    report('order', not a2n_order_ok(old_calls), 'control: the file as it was (the mark after the windows) is caught: %s' % ' '.join(old_calls))
# row back's "nothing was read" from the store's own line, on the two real lines of a recorded return: the store's line gives 0 MiB and 0 pieces,
# the server's summary (1068 MiB of the conversation brought back, in 0.5 ms: a revive) is not bytes read; the restart point is the server line's (back_re).
STORE_LINE = 'state: session 1 back from disk: 0.0 MiB in 0.1 ms (0.00 GB/s; 0.0 ms waiting for writes; 0 pieces checked, 0000000000000000); restart point 1000000 of the 1000000 it had'
SERVE_LINE = 'serve: session 00000000737070e8 back from disk: 1068.0 MiB of 260.2 MiB, 1000000 positions, restart point 1000000 of 1000000, in 0.5 ms (budget 619 numbers)'
both = STORE_LINE + '\n' + SERVE_LINE + '\n'
report('backline', imager_mod.store_reads(both) == [(0.0, 0)], 'the store line gives 0.0 MiB and 0 pieces checked, and the server line is not counted: %s' % imager_mod.store_reads(both))
old_rx = re.compile(r'back from disk: ([\d.]+) MiB')
report('backline', any(float(x.group(1)) > 0 for x in old_rx.finditer(both)), 'control: the old regex `back from disk: X MiB` takes the server line (1068.0 MiB) for bytes read, and failed the row')
mb2 = re.search(imager_mod.back_re(0x737070e8), both)
report('backline', bool(mb2) and mb2.group(4) == mb2.group(5) == '1000000', 'the restart point is read from the server line (back_re): 1000000 of 1000000')
# the stall guard excuses a decoder whose line the engine gave to a request that needs the pool (decoder 1 waited for the 1M request by design), and still stops on one that was
# not given back, and on all of them silent. Fake log and decoders, the three cases.
def guard_case(path, silent, given_back_k):
    spec3 = importlib.util.spec_from_file_location('split_test_guard', path)
    mod = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(mod)
    text = 'serve: slot 0 (session %016x) gives its 45 chunks back to the pool for a request of 1000001 positions\n' % (0x73706400 + given_back_k) if given_back_k is not None else 'serve: ok\n'

    class GLog:
        def mark(self):
            return len(text)

        def text(self, since=None):
            return text

    d = mod.Decoders.__new__(mod.Decoders)
    now = time.time()
    d.last = {1: now - (12 if 0 in silent else 1), 2: now - (12 if 1 in silent else 1), 3: now - (12 if 2 in silent else 1)}
    d.ended = {}
    d.stall_guard, d.guard_mark, d.excused = (10.0, GLog()), 0, ()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            d._check_stall()
        return False
    except mod.DecodersStopped:
        return True


report('guard', not guard_case(FILE, {0}, 0), '(a) one decoder given back and silent 12 s, the others stepping: no stop')
report('guard', guard_case(FILE, {0}, None), '(b) a decoder not given back, silent 12 s: stop')
report('guard', guard_case(FILE, {0, 1, 2}, 0), '(c) every decoder silent 12 s, one given back: stop (the engine is not stepping)')
report('guard', guard_case(FILE, {0}, 1), '(b2) a silent decoder whose neighbour was given back, not itself: stop')
if _os.path.exists('/tmp/split_test.earlier4.py'):
    report('guard', guard_case('/tmp/split_test.earlier4.py', {0}, 0), 'control: the file as it was stops on (a) (the false stop)')
# the source rows of the earlier rules
ilb = body_of(src, 'void il_pass()')
report('pfwait', 'pf_start_waits(' in ilb and 'pf_waits_ =' in ilb and ilb.count('start_waited = true') >= 2, 'il_pass sets pf_waits_ from pf_start_waits and both Waits answers set start_waited')
rb2 = body_of(src, 'void run()')
report('pfwait', 'rot_stalled(' in rb2 and 'pf_reading()' in rb2[rb2.index('rot_stalled('):rb2.index('rot_stalled(') + 120] and '!pfq_.empty()' not in rb2[rb2.index('stalled_ = splitpace::rot_stalled('):rb2.index('stalled_ = splitpace::rot_stalled(') + 120],
       'run()\'s rot_stalled names pf_reading() and not !pfq_.empty()')
itb = body_of(src, 'void imager_tick()')
report('pfwait', 'pfq_busy(' in itb and 'all_wait(' in itb, 'imager_tick names pfq_busy( and all_wait(')
report('pfwait', 'pf_waits_' in body_of(src, 'bool imager_aged(') and 'pf_waits_' in body_of(src, 'void ride_split('), 'imager_aged and ride_split pass pf_waits_ to pool_waiter')
report('pfwait', 'reader_wait(!rot_.empty()' in itb, 'reader_wait( is called with !rot_.empty() (the prompt is in flying_: the counter could never rise)')
# the tests' own instrument: a gap counts only when it BEGAN inside the window
dd = imager_mod.Decoders.__new__(imager_mod.Decoders)
dd.gaps = [(10.0, 0.6), (10.5, 0.1)]
report('instrument', abs(dd.longest(9.9, 20.0) - 0.1) < 1e-9, 'Decoders.longest over gaps [(10.0, 0.6), (10.5, 0.1)] from 9.9 is 0.1 (the 0.6 gap began at 9.4, before the window; as built 0.6): %s' % dd.longest(9.9, 20.0))
# qd's premise counts decoders' sessions only (three lines named fillers giving back their own slots for window bookmarks)
fx = ('serve: slot 4 (session 0000000073706e5f) gives its 110 chunks back to the pool for window bookmarks\n'
      'serve: slot 5 (session 0000000073706e60) gives its 110 chunks back to the pool for window bookmarks\n'
      'serve: slot 4 (session 0000000073706e61) gives its 110 chunks back to the pool for window bookmarks\n'
      'serve: slot 0 (session 0000000073706400) gives its 45 chunks back to the pool for a request of 1000001 positions\n')
report('qdgb', imager_mod.decoder_give_backs(fx, 4) == 1, 'the premise counts 1 (the decoder\'s), not 4: %d' % imager_mod.decoder_give_backs(fx, 4))
report('qdgb', len(re.findall(r'gives its \d+ chunks back', fx)) == 4, 'control: the old unbound check counts 4 and refused the row')
# ---- the rows on the file as it was (/tmp/split_test.earlier8.py) and as it is now
QS_TEXT = "serve: req 8 waits for a slot: 350001 positions need 110 numbers; its stored conversation holds 0 in the pool and needs 0 back; 84 free and unreserved, 0 more can be given back (the request's own conversation excepted), of 428 numbers in the pool\nstore-check: rank 0 state command (6 ops, first 14) digest 28909dd9427868c6\nspill: 9.8 MiB to staging in 1.32 ms on the stream (7.7 GB/s; checksums before it 0.00 ms), then to the file in 2.1 ms (4.80 GB/s)\ncopy: digest 3b228071 after 3 batch steps\nserve: batch of 4: 3 steps in 168.7 ms, 66.99 ms a step (median); req, tokens, end: 2 5 paused, 1 6 paused, 3 4 paused, 4 5 paused; drafts: 6.67 tokens and 12.00 rows per step, 12 of 12 drafts accepted; GPU ms by rows:\nstore-check: rank 0 ride digest 189279d16e39d8ef\ncopy: digest eeca1866 after 3 batch steps\nserve: split rides in the run: 1 rides, 12 imaged, 12 marked, longest 1.29 ms\nserve: batch of 4: 3 steps in 162.0 ms, 63.93 ms a step (median); req, tokens, end: 2 7 paused, 1 4 paused, 3 5 paused, 4 4 paused; drafts: 6.67 tokens and 12.00 rows per step, 12 of 12 drafts accepted; GPU ms by rows:\nserve: slot 0 (session 0000000073706401) gives its 10 chunks back to the pool for a request of 350001 positions\nstore-check: rank 0 state command (3 ops, first 1) digest ae33ec3c1ea71b05\nserve: slot 1 (session 0000000073706400) gives its 10 chunks back to the pool for a request of 350001 positions\nstore-check: rank 0 state command (3 ops, first 1) digest 4a69040b681a6017\nserve: slot 2 (session 0000000073706402) gives its 10 chunks back to the pool for a request of 350001 positions\nstore-check: rank 0 state command (3 ops, first 1) digest a0913db8f721094f\nserve: slot 3 (session 0000000073706403) gives its 10 chunks back to the pool for a request of 350001 positions\nstore-check: rank 0 state command (3 ops, first 1) digest f0ddb42a8605f427\nserve: req 8 waits at its start (no room): have 100, want 110 numbers\nstore-check: rank 0 state command (6 ops, first 14) digest 26301e44d35aa0a1\nspill: 48.0 MiB to staging in 7.90 ms on the stream (6.4 GB/s; checksums before it 0.00 ms), then to the file in 10.9 ms (4.61 GB/s)\nstore-check: rank 0 state command (6 ops, first 14) digest ff0550a5afebe1c6\nstore-check: rank 0 state command (6 ops, first 14) digest fde4c34c1f0a4368\nstore-check: rank 0 state command (6 ops, first 14) digest 5298409ffaeef422\nstore-check: rank 0 ride digest 9eedc9a6eb3b9ca6\ncopy: digest 01f9213a after 7 batch steps\nserve: split rides in the run: 1 rides, 64 imaged, 64 marked, longest 1.72 ms\nserve: batch of 4: 7 steps in 418.1 ms, 64.50 ms a step (median); req, tokens, end: 2 29 paused, 1 25 paused, 3 23 paused, 4 7 paused; drafts: 12.00 tokens and 14.29 rows per step, 60 of 60 drafts accepted; GPU ms by rows: 16: 63.85 (2)\nserve: req 8 factor 16 (window 1048576): restart: no base named, and this engine does not hold the session; reused 0 (cold)\n"
QI_TEXT = 'serve: req 3 waits at its start (no room): have 82, want 110 numbers\nserve: req 3 waits at its start for the pool and nothing in flight can step (0 paused, 0 waiting for a slot): the imager makes room (0 image jobs queued, 2 conversations to image)\nstore-check: rank 0 ride digest 13b30df7c6aeb5fb\nserve: store digest 3 ride: 13b30df7c6aeb5fb\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e5f; 0 ghosts\nstore-check: rank 0 ride digest f88c5237f4c0a866\nserve: store digest 4 ride: f88c5237f4c0a866\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e5f; 0 ghosts\nstore-check: rank 0 ride digest 437cfdadd7d91789\nserve: store digest 5 ride: 437cfdadd7d91789\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e5f; 0 ghosts\nstore-check: rank 0 ride digest c3a6badab4ecb803\nserve: store digest 6 ride: c3a6badab4ecb803\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e5f; 16 ghosts\nstore-check: rank 0 ride digest af5e125a6ad08e38\nserve: store digest 7 ride: af5e125a6ad08e38\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e5f; 16 ghosts\nstore-check: rank 0 ride digest a6826c1804ddcd99\nserve: store digest 8 ride: a6826c1804ddcd99\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e5f; 16 ghosts\nstore-check: rank 0 ride digest 0240e7061890738e\nserve: store digest 9 ride: 0240e7061890738e\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e5f; 32 ghosts\nstore-check: rank 0 ride digest bc8c5b63c9068f0f\nserve: store digest 10 ride: bc8c5b63c9068f0f\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e5f; 32 ghosts\nspill: 12.0 MiB to staging in 1.01 ms on the stream (12.4 GB/s; checksums before it 0.00 ms), then to the file in 3.0 ms (4.26 GB/s)\nstore-check: rank 0 ride digest 3d55d8f07ad15e6b\nserve: store digest 11 ride: 3d55d8f07ad15e6b\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e5f; 48 ghosts\nstore-check: rank 0 ride digest 23eedae15f378166\nserve: store digest 12 ride: 23eedae15f378166\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e60; 64 ghosts\nstore-check: rank 0 ride digest fecb0661884a0711\nserve: store digest 13 ride: fecb0661884a0711\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e60; 64 ghosts\nstore-check: rank 0 ride digest e0831bf819618a18\nserve: store digest 14 ride: e0831bf819618a18\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e60; 80 ghosts\nstore-check: rank 0 ride digest 93836bde3a60ba43\nserve: store digest 15 ride: 93836bde3a60ba43\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e60; 80 ghosts\nspill: 12.0 MiB to staging in 0.99 ms on the stream (12.7 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.43 GB/s)\nstore-check: rank 0 ride digest f5d904ed723b7df3\nserve: store digest 16 ride: f5d904ed723b7df3\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e60; 96 ghosts\nstore-check: rank 0 ride digest a7387230400aba78\nserve: store digest 17 ride: a7387230400aba78\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e60; 96 ghosts\nstore-check: rank 0 ride digest d4b5d24b0d840044\nserve: store digest 18 ride: d4b5d24b0d840044\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e60; 115 ghosts\nspill: 12.0 MiB to staging in 1.01 ms on the stream (12.5 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.41 GB/s)\nstore-check: rank 0 ride digest 5f12e38761473eb9\nserve: store digest 19 ride: 5f12e38761473eb9\nserve: split: settled: 0 marked, 0 imaged (idle); queued 0000000073706e60; 115 ghosts\nspill: 12.0 MiB to staging in 1.00 ms on the stream (12.6 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.37 GB/s)\nstore-check: rank 0 ride digest 2b75de2c5204fdb5\nserve: store digest 20 ride: 2b75de2c5204fdb5\nserve: split: settled: 16 marked, 16 imaged (idle); queued 0000000073706e60; 147 ghosts\nstore-check: rank 0 ride digest 24e4d49fc19a7b79\nserve: store digest 21 ride: 24e4d49fc19a7b79\nserve: split: settled: 28 marked, 28 imaged (idle); queued none; 188 ghosts\nstore-check: rank 0 ride digest 24e4d49fc19a7b79\nserve: store digest 22 ride: 24e4d49fc19a7b79\nstore-check: rank 0 ride digest 5c4fed81e9e26bb4\nserve: store digest 23 ride: 5c4fed81e9e26bb4\nserve: split: settled: 16 marked, 16 imaged (idle); queued none; 204 ghosts\nspill: 12.0 MiB to staging in 0.98 ms on the stream (12.8 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.48 GB/s)\nstore-check: rank 0 ride digest 5c4fed81e9e26bb4\nserve: store digest 24 ride: 5c4fed81e9e26bb4\nstore-check: rank 0 ride digest 5c4fed81e9e26bb4\nserve: store digest 25 ride: 5c4fed81e9e26bb4\nstore-check: rank 0 ride digest 5c4fed81e9e26bb4\nserve: store digest 26 ride: 5c4fed81e9e26bb4\nstore-check: rank 0 ride digest d12b89328c72e977\nserve: store digest 27 ride: d12b89328c72e977\nserve: split: settled: 16 marked, 16 imaged (idle); queued none; 220 ghosts\nstore-check: rank 0 ride digest d12b89328c72e977\nserve: store digest 28 ride: d12b89328c72e977\nstore-check: rank 0 ride digest d12b89328c72e977\nserve: store digest 29 ride: d12b89328c72e977\nstore-check: rank 0 ride digest d12b89328c72e977\nserve: store digest 30 ride: d12b89328c72e977\nstore-check: rank 0 ride digest b9d9152a6b8783be\nserve: store digest 31 ride: b9d9152a6b8783be\nserve: split: settled: 16 marked, 16 imaged (idle); queued none; 236 ghosts\nspill: 12.0 MiB to staging in 0.97 ms on the stream (12.9 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.40 GB/s)\nstore-check: rank 0 ride digest b9d9152a6b8783be\nserve: store digest 32 ride: b9d9152a6b8783be\nstore-check: rank 0 ride digest b9d9152a6b8783be\nserve: store digest 33 ride: b9d9152a6b8783be\nstore-check: rank 0 ride digest 9264fdefc53e058e\nserve: store digest 34 ride: 9264fdefc53e058e\nserve: split: settled: 16 marked, 16 imaged (idle); queued none; 252 ghosts\nstore-check: rank 0 ride digest 9264fdefc53e058e\nserve: store digest 35 ride: 9264fdefc53e058e\nstore-check: rank 0 ride digest 9264fdefc53e058e\nserve: store digest 36 ride: 9264fdefc53e058e\nstore-check: rank 0 ride digest 9264fdefc53e058e\nserve: store digest 37 ride: 9264fdefc53e058e\nstore-check: rank 0 ride digest 30bcd6ddac884f6c\nserve: store digest 38 ride: 30bcd6ddac884f6c\nserve: split: settled: 16 marked, 16 imaged (idle); queued none; 268 ghosts\nstore-check: rank 0 ride digest 30bcd6ddac884f6c\nserve: store digest 39 ride: 30bcd6ddac884f6c\nspill: 12.0 MiB to staging in 1.08 ms on the stream (11.7 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.51 GB/s)\nstore-check: rank 0 ride digest 30bcd6ddac884f6c\nserve: store digest 40 ride: 30bcd6ddac884f6c\nstore-check: rank 0 ride digest 1f9dd8f6271ba699\nserve: store digest 41 ride: 1f9dd8f6271ba699\nserve: split: settled: 16 marked, 16 imaged (idle); queued none; 287 ghosts\nstore-check: rank 0 ride digest 1f9dd8f6271ba699\nserve: store digest 42 ride: 1f9dd8f6271ba699\nstore-check: rank 0 ride digest 1f9dd8f6271ba699\nserve: store digest 43 ride: 1f9dd8f6271ba699\nstore-check: rank 0 ride digest 1f9dd8f6271ba699\nserve: store digest 44 ride: 1f9dd8f6271ba699\nstore-check: rank 0 ride digest 0c7c243ebc4da739\nserve: store digest 45 ride: 0c7c243ebc4da739\nserve: split: settled: 16 marked, 16 imaged (idle); queued none; 319 ghosts\nspill: 12.0 MiB to staging in 1.29 ms on the stream (9.8 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.56 GB/s)\nstore-check: rank 0 ride digest 0c7c243ebc4da739\nserve: store digest 46 ride: 0c7c243ebc4da739\nstore-check: rank 0 ride digest 0c7c243ebc4da739\nserve: store digest 47 ride: 0c7c243ebc4da739\nstore-check: rank 0 ride digest f7ac0e9400e5416c\nserve: store digest 48 ride: f7ac0e9400e5416c\nserve: split: settled: 12 marked, 12 imaged (idle); queued none; 344 ghosts\nspill: 9.8 MiB to staging in 0.97 ms on the stream (10.6 GB/s; checksums before it 0.00 ms), then to the file in 1.8 ms (5.70 GB/s)\nspill: 12.0 MiB to staging in 1.01 ms on the stream (12.4 GB/s; checksums before it 0.00 ms), then to the file in 2.2 ms (5.73 GB/s)\nspill: 12.0 MiB to staging in 1.01 ms on the stream (12.4 GB/s; checksums before it 0.00 ms), then to the file in 2.2 ms (5.64 GB/s)\nspill: 12.0 MiB to staging in 0.98 ms on the stream (12.8 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.53 GB/s)\nspill: 12.0 MiB to staging in 1.03 ms on the stream (12.2 GB/s; checksums before it 0.00 ms), then to the file in 2.2 ms (5.76 GB/s)\nspill: 12.0 MiB to staging in 0.98 ms on the stream (12.9 GB/s; checksums before it 0.00 ms), then to the file in 2.3 ms (5.59 GB/s)\nspill: 12.0 MiB to staging in 0.99 ms on the stream (12.8 GB/s; checksums before it 0.00 ms), then to the file in 2.2 ms (5.71 GB/s)\nspill: 12.0 MiB to staging in 1.43 ms on the stream (8.8 GB/s; checksums before it 0.00 ms), then to the file in 2.2 ms (5.70 GB/s)\nspill: 12.0 MiB to staging in 1.53 ms on the stream (8.2 GB/s; checksums before it 0.00 ms), then to the file in 2.2 ms (5.75 GB/s)\nspill: 9.8 MiB to staging in 1.10 ms on the stream (9.3 GB/s; checksums before it 0.00 ms), then to the file in 1.8 ms (5.82 GB/s)\nserve: req 3 factor 16 (window 1048576): restart: no base named, and this engine does not hold the session; reused 0 (cold)\n'
OLD8 = '/tmp/split_test.earlier8.py'
old8 = None
if _os.path.exists(OLD8):
    old8 = importlib.util.module_from_spec(importlib.util.spec_from_file_location('split_test_old8', OLD8))
    importlib.util.spec_from_file_location('split_test_old8', OLD8).loader.exec_module(old8)
gp = [0.110, 0.0002, 0.0002, 0.0002] * 20
report('step', imager_mod.step_median(gp) == 0.110, 'twenty steps of gaps [0.110, 0.0002 x3]: step_median is 0.110: %s' % imager_mod.step_median(gp))
report('step', imager_mod.median(gp) == 0.0002, 'control: the old median of the same list is 0.0002 (the 0.0 ms of the earlier run)')
report('step', imager_mod.steps_agree(123.0, 123.6) and not imager_mod.steps_agree(0.2, 123.6), 'the cross-check: 123.0 ms agrees with the engine\'s 123.60 ms, 0.2 ms does not')
g1 = {'clean_numbers': 381, 'ghost_numbers': 0}
report('g1', imager_mod.imaged_numbers(g1) == 381, 'imaged_numbers of g1\'s recorded STATS (clean 381, ghosts 0) is 381')
rp = ts[ts.index('def rides_phase('):ts.index('def cmd_imager(')]
report('g1', 'imaged_numbers' in rp and "a.event == 'g'" in rp and 'ghost_numbers' in rp, 'rides_phase reads imaged_numbers for event g and ghost_numbers for a2n')
if old8 is not None:
    report('g1', 'imaged_numbers' not in open(OLD8).read(), 'control: the file as it was reads ghosts alone (0 for g1: the NOT REACHED)')
ci = ts[ts.index('def cmd_imager('):]
calls = [mm for mm in re.finditer(r'imager_turn\(([^\n]*)\)', ci)]
bad_calls = [c.group(0) for c in calls if "'ref'" not in c.group(0) and not c.group(0).rstrip().endswith('d)') and ', d)' not in c.group(0)]
report('a2nret', not bad_calls, 'every imager_turn( call of cmd_imager but event ref\'s passes d (%s)' % (bad_calls or 'none'))
ia = ci[ci.index("if a.event == 'a2n':"):ci.index("elif a.event in ('back'")]
report('a2nret', ia.index('rides_phase(') < ia.index('CANCEL') < ia.index('submit_watch(') and "ghost_numbers" in ia[ia.index('submit_watch('):], 'in event a2n a CANCEL and the wait for its DONE stand between rides_phase( and the first filler\'s submit_watch(; the fillers\' loop reads ghost_numbers')
if old8 is not None:
    oi = open(OLD8).read()
    oia = oi[oi.index("if a.event == 'a2n':", oi.index('def cmd_imager(')):]
    report('a2nret', "imager_turn(e, T, sid, hist, 'the return')" in oia, 'control: the file as it was calls imager_turn without d in a2n')
ride_pieces, idle_lines = imager_mod.carrier(QS_TEXT)
report('carrier', ride_pieces == 76 and idle_lines == 0, 'qs req 8\'s lines: %d pieces by rides, %d idle lines (76 and 0)' % (ride_pieces, idle_lines))
rp2, il2 = imager_mod.carrier(QI_TEXT)
report('carrier', il2 == 27, 'qi\'s lines: %d idle `split: settled:` lines (27)' % il2)
report('carrier', 'serve: split: settled:' not in QS_TEXT, 'control: the old test\'s `serve: split: settled:` in text is false on qs\'s lines (its FAIL)')
report('stall', imager_mod.stall_lines(QI_TEXT) == (1, 0), 'stall_lines on qi\'s lines: one `makes room`, no `nothing`: %s' % (imager_mod.stall_lines(QI_TEXT),))
nothing = 'serve: req 9 waits at its start for the pool and nothing in flight can step (0 paused, 0 waiting for a slot): nothing can be imaged: it waits for a request to end\n'
report('stall', imager_mod.stall_lines(nothing) == (0, 1), 'a line with the second ending counts one `nothing`: %s' % (imager_mod.stall_lines(nothing),))
qb = ci[ci.index("else:  # qi / qw / qd / qs"):]
report('qs', 'stall_b' not in qb[qb.index("if a.event == 'qs':"):qb.index("if a.event in ('qd', 'qs'):\n            print('  free")] and '180' in qb, 'event qs has no check that names a stall line, and its limit is 180')
if old8 is not None:
    report('qs', "a stall line (" in oi and " is in the log" in oi, 'control: the file as it was checks a stall line in qs')
# one decoder prints no `batch of` line: its step is its `tokens in X ms` over its `N verify steps`
G1 = ('serve: req 2 drafts: 1658 verify steps, 6.55 drafts verified and 6.40 accepted per step, steps verifying 0..5: 11 20 20 26 26 1555; gate 0.567 (0.154-0.601); GPU ms by rows: 1: 19.62 (10)\n'
      'serve: req 2 session 0000000073706400: prompt 4096, reused 0 (cold), prefill 544.8 ms (7519 tok/s), 12249 tokens in 60105.8 ms (203.9 tok/s), cancelled; store 0.0 MiB; session 0 positions, 0.0 MiB, 0 window snapshots\n')
ms, stepsrc = imager_mod.engine_step(G1)
report('g1step', abs(ms - 36.3) < 0.05 and 'verify steps' in stepsrc and imager_mod.steps_agree(37.0, ms), 'the two real lines give %.1f ms (source: %s) and agree with the test\'s 37.0' % (ms, stepsrc))
report('g1step', imager_mod.median([float(x) for x in imager_mod.STEP_LINE.findall(G1)]) == 0.0, 'control: the old reader (batch of N lines only) reads 0.0 ms from them')
# + both callbacks call pause_wanted; give_back is one pass; no taker takes a paused request's line or slot
cb_one = body_of(src, 'void pause_wanted(') if False else None
report('pause', src.count('pause_wanted(') >= 3 and '!wait_.empty()) && (ends || admissible())' not in src, 'both callbacks call pause_wanted and neither names `!wait_.empty()) && (ends || admissible())`')
report('pause', 'run_window(' in body_of(src, 'void il_pass()'), 'il_pass names run_window(')
gb = body_of(src, 'bool give_back(')
report('pausedslot', gb.count('detach_until(') == 1 and 'flight_lines_go' not in src, 'give_back is one pass of detach_until and names no flight_lines_go (never the lines of requests in flight)')
report('pausedslot', 'idle_for_taker(' in body_of(src, 'bool idle_id('), 'idle_id uses reclaim::idle_for_taker')
ps = body_of(src, 'int pick_slot(')
report('pausedslot', 'PICO_OLD_RULES' in ps and ps.index('return best;  // a newcomer never takes') < ps.index('for (auto it = rot_.rbegin()'), 'pick_slot never takes the slot of a paused request (the rot_ pass is old rules only)')
report('pausedslot', 'pick_slot({}) >= 0; }  // with no slot to be had' in src, 'admissible() has no --slice term under the new rules')
rd = open(os.path.join(HERE, '..', 'reclaim.h'), errors='replace').read()
report('pausedslot', 'inline bool idle_for_taker(' in rd, 'reclaim.h defines idle_for_taker')
# a cancel inside a batch run ends like a stop: DONE at once, done_sent, counted as an end for the pause check (a recorded run: no DONE for 10 s)
cbb = src[src.index('const OnBatch cb = '):src.index('std::vector<BatchEnd> end(n);')]
cc = cbb[cbb.index('if (Io::cancelled(f.rq))'):cbb.index('const int32_t t = o.row[row].token;')]
report('cancel', 'done(f.rq, fe::kCancelled' in cc and 'f.done_sent = true' in cc and 'ended[k] = 1' in cc and 'pause_wanted(true, true)' in cc,
       'the batch callback\'s cancel branch sends the DONE at once (kCancelled, done_sent, ended[k]) and asks the pause check as an end')
pk = body_of(src, 'int pick_slot(')
report('endedslot', 'slot_free_for_arrival(' in pk and src.count('run_ended_.insert(') == 2 and 'run_ended_.clear()' in src, 'pick_slot asks slot_free_for_arrival; both mid-run ends (stop/length and cancel) record their session; the run\'s end clears it')
# the readers on the recorded journals
_R = JOURNALS
_exch = os.path.join(_R, 'exch-journal.txt')
_qs = os.path.join(_R, 'engine-qs.log')
if os.path.exists(_exch):
    et = open(_exch, errors='replace').read()
    n_runs = imager_mod.runs_between(et, r'serve: req 11 waits for a slot', r'back from disk|joined to position|serve: req 11 factor')
    report('runs', n_runs == 11, 'runs_between over exch\'s journal, req 11\'s wait line to its first move: %d runs (11: the row\'s RED, by record)' % n_runs)
else:
    print('SKIP runs: %s not here' % _exch)
one = 'serve: req 5 waits for a slot: x\nserve: batch of 8: 3 steps in 288.0 ms, 96.0 ms a step (median); req, tokens, end: 1 5 paused; drafts: 17.67 tokens\nstate: session 1 back from disk: 0.0 MiB\n'
report('runs', imager_mod.runs_between(one, r'serve: req 5 waits for a slot', r'back from disk') == 1, 'over a text with one run: 1')
report('runs', imager_mod.runs_between('nothing', r'serve: req 5 waits', r'back from disk') == -1, 'no wait line: -1')
if os.path.exists(_qs):
    qt = open(_qs, errors='replace').read()
    gb = imager_mod.decoder_give_backs_before_start(qt, 7, 4)
    report('qsgb', gb == 4, 'decoder_give_backs_before_start over engine-qs.log, req 7\'s arrival to its start: %d (4)' % gb)
    report('qsgb', imager_mod.giveback_violations(qt, 7, 4) == 4, 'giveback_violations on req 7 as built: 4 (each decoder\'s give-back is followed by a wait line, not by the factor line)')
    report('qsgb', imager_mod.decoders_wait_at_start(qt, 4) == [4], 'a decoder that `waits at its start` in the record (req 4): %s' % imager_mod.decoders_wait_at_start(qt, 4))
    n7 = imager_mod.runs_between(qt, r'serve: req 7 waits at its start', r'serve: req 7 factor')
    report('qsgb', n7 == 1, 'batch lines between req 7\'s `waits at its start` and its start: %d (at most 1)' % n7)
else:
    print('SKIP qsgb: %s not here' % _qs)
bi = imager_mod.batch_info('serve: batch of 8: 3 steps in 288.0 ms, 96.0 ms a step (median); req, tokens, end: 9 15 paused, 8 10 paused; drafts: 17.67 tokens and 3 rows per step\n')
report('runs', bi == [(3, 96.0, 17.67, True)], 'batch_info reads steps, ms a step, tokens a step and the `paused` ends: %s' % bi)
# the rows' new checks are in the test, named by their source (each RED on /tmp/split_test.earlier10.py)
B10 = '/tmp/split_test.earlier10.py'
tsx = open(FILE, errors='replace').read()
et2 = tsx[tsx.index('def exchange_turn('):tsx.index('def cmd_imager(')]
need_et = ['can begin: the run is ended for it', 'ride digest', 'within 5 %', '0.95', '3.48', 'MISS_LINE', 'decoder_give_backs(seg']
report('exchrow', all(x in et2 for x in need_et), 'row exchange checks: can-begin line, ride-digest premise, tokens/ms a step within 5 %%, tokens/s 0.95, 3.48 s, miss lines, decoder give-backs (missing: %s)' % [x for x in need_et if x not in et2])
qsx = tsx[tsx.index("if a.event == 'qs':\n            seg_s"):tsx.index("if a.event in ('qd', 'qs'):\n            print('  free")]
report('exchrow', all(x in qsx for x in ('giveback_violations(', 'decoders_wait_at_start(', 'runs_between(', 'MISS_LINE')), 'row qs: give-back violations, decoders that wait at their start, at most one batch line, miss lines')
cix = tsx[tsx.index("if a.event == 'a2n':", tsx.index('def cmd_imager(')):]
report('exchrow', "within 1 s of the CANCEL" in cix and 'tf - ts_f <= 2.0' in cix and 'at most 3 steps' in cix, 'row a2n: the cancel\'s DONE within 1 s, the first filler\'s factor line within 2 s of its submit with exactly one batch line of at most 3 steps')
if os.path.exists(B10):
    ob = open(B10, errors='replace').read()
    report('exchrow', 'def exchange_turn(' not in ob and 'giveback_violations' not in ob and 'within 1 s of the CANCEL' not in ob and 'can begin' not in ob, 'control: the file as it was has none of them (RED)')
# three journal counts in every row (one helper), and a2n's three steps before its fillers
JT = ('serve: BUG: every flight waits for an extension (3 flights)\n'
      'serve: every request in flight waits for the pool (4 paused, 0 waiting for a slot): nothing can be imaged: they wait for a request to end\n'
      'serve: every request in flight waits for the pool (4 paused, 0 waiting for a slot): the imager makes room (0 image jobs queued, 2 conversations to image)\n'
      'serve: req 5 grows: its reservation reaches 12000 positions (was 8000)\n'
      'serve: req 6 grows: its reservation reaches 12000 positions (was 8000)\n')
report('counts', imager_mod.journal_counts(JT) == (1, 1, 2), 'journal_counts on a short text: (1 BUG, 1 `nothing can be imaged`, 2 grows; the `makes room` stall line is not counted): %s' % (imager_mod.journal_counts(JT),))
report('counts', imager_mod.journal_counts('serve: ok\n') == (0, 0, 0), 'a clean text counts (0, 0, 0)')
imager_mod.check_failed = None
fails_before = imager_mod.fails
with contextlib.redirect_stdout(io.StringIO()):
    imager_mod.row_counts(JT, 'x')
report('counts', imager_mod.fails - fails_before == 2, 'row_counts fails both of the first two counts on that text (2 failures: %d); grows alone is not a failure' % (imager_mod.fails - fails_before))
imager_mod.fails = fails_before
with contextlib.redirect_stdout(io.StringIO()):
    imager_mod.row_counts('serve: req 5 grows: its reservation reaches 1 positions (was 0)\n', 'x')
report('counts', imager_mod.fails == fails_before, 'a text with grows only is not a failure (printed for the reader)')
cs = open(FILE, errors='replace').read()
ci3 = cs[cs.index("if a.event == 'a2n':", cs.index('def cmd_imager(')):]
report('a2n3', all(x in ci3 for x in ('rid9 = d.add_decoder', '(i) ', '(ii) ', '(iii) ', 'rid10 = d.add_decoder', 'tk >= 0.95 * ev_ctl')), 'a2n has the three steps: ninth request left 30 s, cancel with the ninth\'s factor within 1 s, tenth after a second cancel')
report('a2n3', 'row_counts(' in cs[cs.index('def cmd_imager('):], 'every row ends with row_counts')
if os.path.exists('/tmp/split_test.earlier11.py'):
    ob11 = open('/tmp/split_test.earlier11.py', errors='replace').read()
    report('a2n3', 'journal_counts' not in ob11 and 'add_decoder' not in ob11, 'control: the file as it was has neither the helper nor the steps (RED)')
# the standstill and the longest-reply end in run(); the row ext
rb8 = body_of(src, 'void run()')
report('ext8', 'reclaim::ext_standstill(' in rb8 and 'reclaim::ext_blocks(left)' in rb8 and 'reclaim::ext_pick(' in rb8 and 'reclaim::ext_end_due(' in rb8 and 'finish(victim, fe::kLength)' in rb8,
       'run() asks ext_standstill, ext_blocks(left), ext_pick, ext_end_due and ends the pick with finish(victim, fe::kLength)')
report('ext8', 'ends at %zu tokens (length): every request that holds memory waits to grow and nothing can be freed' in rb8 and 'sleep_for(std::chrono::milliseconds(20))' in rb8 and 'pool_waits > 0 && pfq_.empty()' in rb8,
       'the new line, and the 20 ms rest after any pool wait that took the pass')
report('ext8', "def cmd_ext(" in open(FILE, errors='replace').read() and "'ext'" in open(FILE, errors='replace').read(), 'split_test.py has the row ext')
if os.path.exists('/tmp/split_test.earlier12.py'):
    report('ext8', 'cmd_ext' not in open('/tmp/split_test.earlier12.py', errors='replace').read(), 'control: the file as it was has no row ext (RED)')
# the decoders' owed time is settled by every decode run
lp9 = body_of(src, 'void loop()')
il9 = body_of(src, 'void il_pass()')
report('owed', 'owed_left(' in lp9[lp9.index('} else if (!rot_.empty()) {'):] and 'owed_left(owed_ms_' in il9 and 'std::max(0.0, owed_ms_ -' not in il9,
       'loop()\'s run branch names owed_left(; il_pass names it and no longer std::max(0.0, owed_ms_ -')
_x = os.path.join(_R, 'owed-exch.log')
if os.path.exists(_x):
    xt = open(_x, errors='replace').read()
    n9 = imager_mod.runs_between(xt, r'serve: req 11 waits for a slot', r'back from disk|serve: req 11 factor ')
    report('owed', n9 == 2, 'runs_between over the owed-time journal, req 11\'s wait line to its first move: %d (2: the row\'s RED, by record)' % n9)
else:
    print('SKIP owed: %s not here' % _x)
ts9 = open(FILE, errors='replace').read()
report('owedrows', all(x in ts9 for x in ("'(ii) exactly one `batch of` line", 'premise: the run the `can begin` line ends has at least 10 steps', '(iv) tokens over (steps - 2)', 'journal\'s own stamp', 'premise output: the four second turns')) and 'at least 0.95 of the control window\'s (retired' not in ts9 and "the decoders\\' tokens a second in the wait" not in ts9,
       'owed-time rows: exch (ii) one run line, the 10-step premise, (iv) tokens over steps-2, journal stamps, ext prints the admission lines; the tokens-a-second line is retired')
_ex = os.path.join(_R, 'owed-exch.log')
if os.path.exists(_ex):
    xe = open(_ex, errors='replace').read()
    segw = xe[xe.index('serve: req 11 waits for a slot'):xe.index('back from disk', xe.index('serve: req 11 waits for a slot'))]
    rr = imager_mod.run_tokens(segw)
    report('owedrows', [x[0] for x in rr] == [23, 5], 'run_tokens over exch\'s wait: runs of 23 and 5 steps (on 2 runs, so (ii) is RED by record): %s' % rr)
J1 = ('Nov 23 20:55:01.123456 spark-1 dsgb10_rank[1234]: serve: req 11 waits for a slot: x\n'
      'Nov 23 20:55:03.500000 spark-1 dsgb10_rank[1234]: serve: req 11 can begin: the run is ended for it\n')
U1 = 'serve: req 11 waits for a slot: x\nserve: req 11 can begin: the run is ended for it\n'
a_ = imager_mod.line_stamp(J1, r'serve: req 11 waits for a slot')
b_ = imager_mod.line_stamp(J1, r'serve: req 11 can begin')
report('stamp', a_ is not None and abs(a_ - (20 * 3600 + 55 * 60 + 1.123456)) < 1e-6 and abs(b_ - a_ - 2.376544) < 1e-6, 'line_stamp reads short-precise stamps: %s, %s' % (a_, b_))
report('stamp', imager_mod.line_stamp(U1, r'can begin') is None, 'an unstamped (-o cat) text gives None')
fb = imager_mod.stamp_of(U1, r'serve: req 11 can begin', 0, runner=lambda cmd: J1 if cmd[:4] == ['journalctl', '--user', '-u', 'pico-engine'] and '-o' in cmd and 'short-precise' in cmd else '')
report('stamp', fb is not None and abs(fb - (20 * 3600 + 55 * 60 + 3.5)) < 1e-6, 'stamp_of falls back to the journal (journalctl --user -u pico-engine -o short-precise) when the text has none (RED before: no fallback): %s' % fb)
report('stamp', imager_mod.stamp_of(U1, r'can begin', 0, runner=lambda cmd: '') is None, 'and None when even the journal gives none (the check keeps its message)')
# row ext's sizing from the code, against a recorded journal: lists 105 and 110, each turn's own need 4, req 8's `360321 positions need 114`
poss9 = [16384 * c - 400 for c in (21, 21, 21, 22)]
ls9, ws9, rs9 = imager_mod.ext_numbers(poss9, 1, 16, 256, 20000)
report('extsize', ls9 == [105, 105, 105, 110] and sum(ls9) == 425, 'list numbers 105, 105, 105, 110 (the 425 the row counted; req 8 held 14 + 96 = 110): %s' % ls9)
report('extsize', ws9 == [4, 4, 4, 4], 'each second turn\'s own need beyond its list is 4 (the journal\'s `have 0, want 4`): %s' % ws9)
report('extsize', imager_mod.chunks_for(360321) == 114 and imager_mod.chunks_for(poss9[0] + 1) == 109, 'chunks_for(360321) is 114 (req 8\'s `need 114 numbers`) and a first turn\'s line 109 (req 1\'s `lines 109`)')
tk9, tg9 = imager_mod.ext_pool_tokens_lists(poss9)
report('extsize', sum(rs9) == 441 and tg9 == 443 and (tk9 * 5 + 16383) // 16384 == 443 and ((tk9 - 1) * 5 + 16383) // 16384 == 442, 'the pool the four turns need is 441 numbers (the recorded pool, 428, was 13 short: RED); with 2 free 443 = --pool-tokens %d' % tk9)
report('extsize', (1400000 * 5 + 16383) // 16384 == 428, '--pool-tokens 1400000 gives 428 numbers (the recorded start line; server/main.cpp:1075)')
# row ext revised: the bookmarks the row had not counted
report('extsize2', imager_mod.snap_count(343664) == 30 and imager_mod.snap_count(360048) == 30 and imager_mod.snap_count(355443) == 30, 'snap_count is 30 at 343,664, 360,048 (the `30 window snapshots` of a recorded run) and 355,443')
report('extsize2', 443 - 109 - 60 == 274 and 443 - 218 - 120 == 105 and 105 < imager_mod.chunks_for(343664 + 1), 'by record: 443 - 109 - 60 = 274 free (line 153) and 443 - 218 - 120 = 105 (line 254): 105 free against the third line\'s 109 (the RED: the old pool held 425 + 16, not the bookmarks)')
cf = {n: imager_mod.chunks_for(n) for n in (355444, 355716, 356161, 356225, 364417, 372545)}
report('extsize2', cf == {355444: 110, 355716: 110, 356161: 113, 356225: 114, 364417: 115, 372545: 118}, 'chunks_for: 110 at 355,444 and 355,716, 113 at 356,161, 114 at 356,225, 115 at 364,417, 118 at 372,545: %s' % cf)
tk2, tg2 = imager_mod.ext_pool_tokens()
report('extsize2', (tk2, tg2) == (2251162, 688) and (2251161 * 5 + 16383) // 16384 == 687, 'ext_pool_tokens() = (2251162, 688) and 2251161 gives 687: %s' % ((tk2, tg2),))
report('extsize2', 355460 % 1024 == 132 and (355460 // 1024 + 1) * 1024 == 356352, '355,460 % 1024 = 132 and the next multiple is 356,352')
cx = open(FILE, errors='replace').read()
cx = cx[cx.index('def cmd_ext('):cx.index('def cmd_imager(')]
report('extsize2', all(x in cx for x in ('P = 355443', '60000', 'd.pump(t0 + 720', 'premise after phase 1', 'for the next start: --pool-tokens', 'reason == 1', 'as many ids received')) and 'same turn' not in cx, 'cmd_ext: P 355,443, max_new 60,000, 720 s, the phase-1 premise from `req 4 pool:` with the next --pool-tokens, reason length and the id count; the ids-equal-alone line is retired')
pl = imager_mod.POOL_LINE.findall('serve: req 4 pool: 688 chunk numbers; stored sessions hold 0 (0.0 MiB), the lines 440 with their reservations, 8 free and unreserved\n')
report('extsize2', pl == [('4', '688', '0', '440', '8')], 'POOL_LINE reads `req 4 pool:` (688, stored 0, lines 440, free 8): %s' % pl)
# the engine prints a request's `pool:` line as it ends, beside its DONE: the premise waits for the fourth line before it judges
# (a recorded run judged req 3's line: req 4's stood in the journal 22:50:22.154, the instant its DONE came back)
def ext_waits_for_four(src):
    c = src[src.index('def cmd_ext('):src.index('def cmd_imager(')]
    w = c.find('len(POOL_LINE.findall(log.text(0))) < 4')
    return w >= 0 and w < c.index('txt = log.text(0)')
report('extwait', ext_waits_for_four(open(FILE, errors='replace').read()), 'cmd_ext waits for four `pool:` lines in the log before it reads the premise')
_run = os.path.join(os.path.dirname(os.path.abspath(FILE)), 'split_test.py.earlier')
if os.path.exists(_run):
    report('extwait', not ext_waits_for_four(open(_run, errors='replace').read()), 'control: the earlier version reads the premise at once (RED)')
# row ext's journal checks: the engine's lines of one standstill (as a recorded run printed them), each count once and read in its place
STALL_W = 'serve: every request in flight waits for the pool (4 paused, 0 waiting for a slot): nothing can be imaged: they wait for a request to end'
BUG_W = 'serve: BUG: every flight waits for an extension (4 flights)'
ENDS_W = 'serve: req 8 ends at 8858 tokens (length): every request that holds memory waits to grow and nothing can be freed (4 requests, 688 numbers in the pool)'
GR = 'serve: req %d grows: its reservation reaches 356230 positions (was 355973)'
EXT_GOOD = '\n'.join(['copy: digest a4783d91 after 10 steps (request at 364061)', STALL_W, BUG_W, ENDS_W, 'store-check: rank 0 state command (8 ops, first 9) digest 7ad07aa41c6ff0a9',
                     GR % 6, 'store-check: rank 0 state command (1 ops, first 14) digest b8c8231f553854e2', GR % 5, GR % 7, 'spill: 6.8 MiB to staging']) + '\n'


def ext_run(text, mode='b', four=(5, 6, 7, 8)):
    """(checks failed, printed text) of ext_counts over `text`; a build without ext_counts fails."""
    if not hasattr(SPLIT, 'ext_counts'):
        return 99, 'no ext_counts'
    f0 = SPLIT.fails
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SPLIT.ext_counts(text, mode, four)
    n = SPLIT.fails - f0
    SPLIT.fails = f0
    return n, buf.getvalue()


import contextlib
import io
n, out = ext_run(EXT_GOOD)
report('extcounts', n == 0 and out.count('  ok:') == 6, 'the standstill as recorded (stall, BUG, ends, then the grows of 6, 5, 7): six ok, none FAIL (%d failed)' % n)
EXT_CASES = [
    ('a second BUG line', EXT_GOOD.replace(BUG_W + '\n', BUG_W + '\n' + BUG_W + '\n')),
    ('the stall line ending `the imager makes room`', EXT_GOOD.replace('nothing can be imaged: they wait for a request to end', 'the imager makes room')),
    ('a `grows:` line between the stall line and the `ends at` line', EXT_GOOD.replace(BUG_W + '\n', GR % 6 + '\n' + BUG_W + '\n')),
    ('a `batch of` line between the stall line and the `ends at` line', EXT_GOOD.replace(BUG_W + '\n', 'serve: batch of 16 steps\n' + BUG_W + '\n')),
    ('no stall line', EXT_GOOD.replace(STALL_W + '\n', '')),
    ('a second `ends at` line', EXT_GOOD.replace(ENDS_W + '\n', ENDS_W + '\n' + ENDS_W + '\n')),
    ('no `grows:` line of request 5 after the end', EXT_GOOD.replace(GR % 5 + '\n', '')),
    ('the BUG line after the `ends at` line', EXT_GOOD.replace(BUG_W + '\n' + ENDS_W + '\n', ENDS_W + '\n' + BUG_W + '\n')),
]
for what, txt in EXT_CASES:
    n, out = ext_run(txt)
    report('extcounts', n >= 1 and 'FAIL' in out, 'RED: %s fails (%d check(s))' % (what, n))
n, out = ext_run(EXT_GOOD.replace(ENDS_W + '\n', '').replace(GR % 5 + '\n', ''), 'a')
report('extcounts', n == 0, 'mode a needs no `ends at` line and no grows after it: checks 1, 2, 4, 5 only (%d failed)' % n)
# the RED by record: row_counts, the helper row ext used, on the same text fails twice (the two FAIL lines of a recorded run)
f0 = SPLIT.fails
with contextlib.redirect_stdout(io.StringIO()):
    SPLIT.row_counts(EXT_GOOD, 'ext')
n_rc = SPLIT.fails - f0
SPLIT.fails = f0
report('extcounts', n_rc == 2, 'row_counts on the same text fails twice (the BUG line 1, the stall line 1): %d' % n_rc)
cxs = open(FILE, errors='replace').read()
cxs = cxs[cxs.index('def cmd_ext('):cxs.index('def cmd_imager(')]
report('extcounts', 'ext_counts(log.text(m0), a.ext_mode, rids)' in cxs and "row_counts(log.text(m0), 'ext')" not in cxs, 'cmd_ext ends with ext_counts, not row_counts')
report('extcounts', hasattr(SPLIT, 'ext_counts') and "'--ext-counts'" in open(FILE, errors='replace').read(), 'split_test.py has ext_counts and the --ext-counts option')
report('extcounts', "row_counts(log.text(mk), a.event)" in open(FILE, errors='replace').read(), 'row_counts stays for every other row')
_before = os.path.join(os.path.dirname(os.path.abspath(FILE)), 'split_test.py.before-extc')
if os.path.exists(_before):
    bs = open(_before, errors='replace').read()
    bx = bs[bs.index('def cmd_ext('):bs.index('def cmd_imager(')]
    report('extcounts', 'def ext_counts(' not in bs and "'--ext-counts'" not in bs and "row_counts(log.text(m0), 'ext')" in bx, 'control: the file as it was has no ext_counts and no --ext-counts, and row ext ends with row_counts (RED)')
print('FAILED' if fails else 'all passed')
sys.exit(1 if fails else 0)
