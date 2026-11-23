#!/usr/bin/env python3
"""A request's image rows are store numbers (docs/vision.md; model/img_rows.h): taken once at the request's first go, read
by every later go (a resume sends no row), given back when the request ends however it ends. Through rank 0's socket
(serve_test.py's client), with synthetic rows: the engine only needs bf16 [rows][5120] and spans of image ids.

Subcommands (each exits 0 on pass, 1 on a failure, 2 when the run did not reach the path it tests):

  replies  --save REF.json | --ref REF.json
      Image requests of several shapes (one image; four images; a 1,100-row image in a 20K prompt; a second image turn on the
      same session straight after the first), greedy with top-2 logprobs. Run once on a build with --save, once on
      the build to compare with --ref: every token id and every logprob must be equal (the rows are the same bytes, read from the store).
  leak     [--nodes n1,n2,n3,n4] [--rows-spans 36]
      A request of 36 x 1,100 = 39,600 rows: STATS image_numbers rises to ceil(rows/204) = 195 while it prefills and is 0 after
      it ends; pool_free is back to its value before; with --nodes, every rank's MemAvailable (/proc/meminfo over ssh) is back
      within --spread GiB (a lasting fall fails the test). The test's own request payload
      is freed before the nodes are read. The workers are judged by MemAvailable; the box the client runs on (rank 0's Spark, 'local') by the
      engine's own terms (its cgroup's memory.current, rank 0's VmRSS, the engine's GPU memory: each within --engine-tol-mib, 256, of
      its value before) and the store's (image_numbers 0, pool_free back); its MemAvailable, /proc/meminfo's terms and the client's
      VmRSS are printed as information only.
  cancel
      The same request ended three ways while it prefills: a CANCEL; a client that closes its socket; a RELEASE of its session.
      After each, image_numbers is 0 (a client that left is found at the next prefill piece: --wait seconds).
  resume   [--log ENGINE.log]
      More sessions than slots (start with --slots 2): an image request (700 rows) decodes in turns beside two text decoders and
      loses its line; its reply must equal the same request alone. With --log the run proves it reached a resume that sent no
      row ("images: ... held") and exits 2 when it did not.
  pair
      Two image requests at once on different sessions beside a text decoder: each reply equals its alone run, image_numbers is
      the sum of both while both run and 0 after.
  ttft     [--ref TTFT.json | --save TTFT.json]
      The first-token time of one 382-row image prompt (8K tokens), 5 repeats: median and spread, to compare against the
      installed build's (not slower beyond the 5 % floor).

Engine settings (engine-test.sh takes them as one string; as image_link_test.py and resume_test.py):
  replies / leak / cancel / pair / ttft:  engine-test.sh BIN "--slots 4 --big-slots 0 --pool-tokens 1507328 --disk-gib 0" image_store_test.py ...
  resume:  engine-test.sh BIN "--slots 2 --big-slots 0 --pool-tokens 400000 --disk-gib 0" image_store_test.py resume --log ~/dsgb10/logs/...

  python3 engine/server/tests/image_store_test.py --socket S --tokens T.npy replies --save ref-installed.json     (one build)
  python3 engine/server/tests/image_store_test.py --socket S --tokens T.npy replies --ref ref-installed.json     (the build to compare)
  python3 engine/server/tests/image_store_test.py --socket S --tokens T.npy leak --nodes HOST1,HOST2,HOST3,HOST4
  python3 engine/server/tests/image_store_test.py --socket S --tokens T.npy cancel
  python3 engine/server/tests/image_store_test.py --socket S --tokens T.npy resume --log ENGINE.log
  python3 engine/server/tests/image_store_test.py --socket S --tokens T.npy pair
  python3 engine/server/tests/image_store_test.py --socket S --tokens T.npy ttft --save ttft-s3.json
"""
import argparse
import json
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import CANCEL, DONE, PROGRESS, SUBMIT, TOKENS, F_KEEP, F_LOGPROBS, REASONS, Engine, load_npy_i32  # noqa: E402
from stats_test import NAMES, STATS, STATS_REQ, stats  # noqa: E402


class Conn(Engine):
    """One front-end connection. The engine serves one at a time: a second connection replaces the first and cancels its
    requests, so STATS is asked on this one, and the request frames that arrive before its answer are kept, in order, for
    the next frame() calls."""

    def __init__(self, path):
        self.q = deque()  # before Engine's handshake, which reads its first frame through frame()
        super().__init__(path)

    def frame(self):
        return self.q.popleft() if self.q else super().frame()

    def numbers(self):
        """image_numbers now, from a fresh STATS (the engine publishes at most twice a second)"""
        time.sleep(0.6)
        self.send(STATS_REQ, struct.pack('<II', 0, 0))
        while True:
            t, p = super().frame()
            if t == STATS:
                n = struct.unpack('<II', p[:8])[1]
                return dict(zip(NAMES, struct.unpack('<%dQ' % n, p[8:8 + 8 * n]))).get('image_numbers')
            self.q.append((t, p))

    def done_seen(self, rid):
        return any(t == DONE and struct.unpack('<Q', p[:8])[0] == rid for t, p in self.q)

F_IMAGES = 8
IMAGE_ID = 129264
DIM = 5120
PER_NUMBER = 204  # whole rows in one store number (2 MiB of a number's first array / 10,240 B)
S0 = 0x696d6773746f0000  # 'imgsto'
fails = 0


class NotReached(Exception):
    pass


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def need(cond, what):
    print('  %s: %s' % ('ok' if cond else 'NOT REACHED', what))
    sys.stdout.flush()
    if not cond:
        raise NotReached(what)


def numbers_for(rows):
    return (rows + PER_NUMBER - 1) // PER_NUMBER


def rows_bytes(n, seed):
    """n rows of bf16 [5120]: a pattern per seed, each row a shifted slice of it (distinct rows, built fast)."""
    import random
    rnd = random.Random(seed)
    one = bytearray()
    for _ in range(DIM + 4096):  # bf16 = the top half of a float32 in [-0.5, 0.5]
        v = struct.unpack('<I', struct.pack('<f', rnd.uniform(-0.5, 0.5)))[0] >> 16
        one += struct.pack('<H', v)
    base = bytes(one)
    return b''.join(base[2 * ((r * 7) % 4096):2 * ((r * 7) % 4096) + 2 * DIM] for r in range(n))


def image_submit(e, session, text, spans, seed, max_new=24, keep=True, base=0):
    """text: token ids; spans: [(position, rows)] in the prompt where image ids go. The prompt is text[0:p0] + ids + text..."""
    prompt, pos, ti = [], 0, 0
    sp = []
    for (p_at, rows) in spans:
        take = p_at - len(prompt)
        prompt += text[ti:ti + take]
        ti += take
        sp.append((len(prompt), rows))
        prompt += [IMAGE_ID] * rows
    prompt += text[ti:ti + 64]
    rid = e.next_id
    e.next_id += 1
    flags = (F_KEEP if keep else 0) | F_LOGPROBS | F_IMAGES
    hdr = struct.pack('<QQQQffIIIIIIII', rid, session, base, 0, 0.0, 1.0, 0, max_new, flags, 2, 0, 1, len(prompt), 0)
    body = hdr + struct.pack('<I', 1) + struct.pack('<%dI' % len(prompt), *prompt)
    body += struct.pack('<I', len(sp)) + b''.join(struct.pack('<IIQ', s, r, 0) for s, r in sp)
    total = sum(r for _, r in sp)
    body += rows_bytes(total, seed)
    e.send(SUBMIT, body)
    return rid, 2, len(prompt)


def layout(T, off, n_images, rows, gap=1500):
    """Text ids and spans for n_images images of `rows` rows with `gap` text tokens before and between them."""
    text = T[off:off + gap * (n_images + 2) + 64]
    return text, [(gap * (i + 1) + rows * i, rows) for i in range(n_images)]


def collect(e, req, on_frame=None):
    rid, n_top, _ = req
    out = {'ids': [], 'lp': [], 'top': [], 'progress': 0}
    while True:
        t, p = e.frame()
        if on_frame:
            on_frame(t)
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            if r != rid:
                continue
            out['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
            off = 16 + 4 * n
            if fl & 1:
                out['lp'] += struct.unpack('<%df' % n, p[off:off + 4 * n])
                off += 4 * n
            if fl & 2:
                pairs = struct.unpack('<' + 'If' * (n * n_top), p[off:off + 8 * n * n_top])
                for i in range(n):
                    out['top'].append(list(zip(pairs[2 * n_top * i::2][:n_top], pairs[2 * n_top * i + 1::2][:n_top])))
        elif t == PROGRESS:
            out['progress'] += 1
        elif t == DONE:
            (r, reason, pt, cached, comp, q_us, p_us, d_us, el, _) = struct.unpack('<QIIIIQQQII', p[:56])
            if r != rid:
                continue
            out.update(reason=REASONS[reason], prompt=pt, prefill_us=p_us, error=p[len(p) - el:].decode())
            return out


def run_image(e, session, text, spans, seed, **kw):
    return collect(e, image_submit(e, session, text, spans, seed, **kw))


def image_numbers(e):
    return e.numbers()


def mem_available(nodes):
    v = []
    for n in nodes:
        # 'local': this box's own /proc/meminfo (rank 0's Spark cannot ssh to itself)
        r = subprocess.run(['grep', 'MemAvailable', '/proc/meminfo'] if n == 'local' else
                           ['ssh', '-o', 'BatchMode=yes', n, 'grep MemAvailable /proc/meminfo'], capture_output=True, text=True, timeout=30)
        v.append(int(r.stdout.split()[1]) / 1048576.0)
    return v


def need_s3(e):
    s = stats(e)[2]
    need('image_numbers' in s, 'the engine reports image_numbers: it is the S3 build')
    need(s['image_numbers'] == 0, 'no image rows are held at the start')
    return s


# ---- replies --------------------------------------------------------------------------------------------------------------

def t_replies(a, e, T):
    cases = {}
    text, spans = layout(T, 1000, 1, 382, 3000)
    cases['one image'] = run_image(e, S0 + 1, text, spans, 1, keep=False)
    text, spans = layout(T, 9000, 4, 300, 800)
    cases['four images'] = run_image(e, S0 + 2, text, spans, 2, keep=False)
    text, spans = layout(T, 20000, 1, 1100, 9000)
    cases['1,100 rows in 20K'] = run_image(e, S0 + 3, text, spans, 3, keep=False)
    text, spans = layout(T, 40000, 1, 500, 2000)  # two turns on one session, back to back
    cases['session turn 1'] = run_image(e, S0 + 4, text, spans, 4, keep=True)
    text2 = text + T[50000:50400]
    cases['session turn 2'] = run_image(e, S0 + 4, text2, spans, 4, keep=True)
    for k, o in cases.items():
        print('  %-20s %s, %d tokens, prefill %.0f ms %s' % (k, o['reason'], len(o['ids']), o['prefill_us'] / 1e3, o['error']))
        need(not o['error'], '%s ran without an error' % k)
    if a.save:
        json.dump({k: {'ids': o['ids'], 'lp': o['lp'], 'top': o['top']} for k, o in cases.items()}, open(a.save, 'w'))
        print('  saved %s: run the same on the S3 build with --ref' % a.save)
        return
    ref = json.load(open(a.ref))
    for k, o in cases.items():
        r = ref[k]
        same = o['ids'] == r['ids'] and [round(x, 9) for x in o['lp']] == [round(x, 9) for x in r['lp']]
        check(same, '%s: ids and logprobs equal the installed build\'s (%d tokens)' % (k, len(r['ids'])))


# ---- leak -----------------------------------------------------------------------------------------------------------------

def big(e, T, session, n_spans=36, seed=9, max_new=8):
    text, spans = layout(T, 2000, n_spans, 1100, 1200)
    rows = sum(r for _, r in spans)
    return image_submit(e, session, text, spans, seed, max_new=max_new, keep=False), rows


def t_leak(a, e, T):
    nodes = a.nodes.split(',') if a.nodes else []
    s0 = need_s3(e)
    m0 = mem_available(nodes) if nodes else []
    req, rows = big(e, T, S0 + 10, a.rows_spans)
    want = numbers_for(rows)
    peak = 0
    while not e.done_seen(req[0]):  # STATS on the request's own connection while it runs
        peak = max(peak, e.numbers() or 0)
    o = collect(e, req)
    print('  request: %s, %d tokens, prefill %.1f s, %d rows' % (o['reason'], len(o['ids']), o['prefill_us'] / 1e6, rows))
    need(not o['error'], 'the request ran')
    check(peak == want, 'while it ran the ranks held %d image numbers (ceil(%d / 204) = %d)' % (peak, rows, want))
    time.sleep(1.5)
    s1 = stats(e)[2]
    check(s1['image_numbers'] == 0, 'after it ended image_numbers is %d' % s1['image_numbers'])
    check(s1['pool_free'] == s0['pool_free'], 'pool_free is back: %d (was %d)' % (s1['pool_free'], s0['pool_free']))
    if nodes:
        del req, o  # the client's own payload is freed before anything is read
        let_go()
        m1 = mem_available(nodes)
        if p0 is not None:
            # the box the client runs on (a 405 MB payload is built there) is judged by the engine's own terms, never by its
            # MemAvailable: the client, and the image encoder container's allocator, are in that number
            p1 = local_snapshot()
            for name, x, y, ok in engine_terms_ok(p0, p1, a.engine_tol_mib):
                check(ok, '%s %s -> %s MiB (tolerance +%d MiB)' % (name, '%.0f' % x if x is not None else 'unread', '%.0f' % y if y is not None else 'unread', a.engine_tol_mib))
            print('  information, not a verdict on this box (the client runs on it):')
            print('    image encoder container GPU memory %d -> %d MiB; the client\'s own VmRSS %.0f -> %.0f MiB' %
                  (p0['gpu_encoder'], p1['gpu_encoder'], p0['client'], p1['client']))
            print('    MemAvailable %.2f -> %.2f GiB' % (m0[nodes.index('local')], m1[nodes.index('local')]))
            for k in MEMINFO_TERMS[1:]:
                print('    %-16s %d -> %d %s' % (k, p0['meminfo'].get(k, 0), p1['meminfo'].get(k, 0), '' if k.startswith('Huge') else 'kB'))
        for n, x, y in zip(nodes, m0, m1):
            if n != 'local':
                check(leak_ok(n, x, y, a.spread, 0.0), '%s MemAvailable %.2f -> %.2f GiB (spread %.2f)' % (n, x, y, a.spread))


# ---- cancel ---------------------------------------------------------------------------------------------------------------

def wait_numbers(e, want, seconds):
    end = time.time() + seconds
    n = e.numbers()
    while n != want and time.time() < end:
        n = e.numbers()
    return n


def t_cancel(a, e, T):
    """each request runs long enough (2,000 tokens) that its end comes from the cancel, not from its length"""
    need_s3(e)
    # a CANCEL
    req, rows = big(e, T, S0 + 20, max_new=2000)
    need(wait_numbers(e, numbers_for(rows), 60) == numbers_for(rows), 'the request holds its %d numbers' % numbers_for(rows))
    e.send(CANCEL, struct.pack('<Q', req[0]))
    o = collect(e, req)
    check(o['reason'] == 'cancelled', 'ended: %s' % o['reason'])
    check(wait_numbers(e, 0, a.wait) == 0, 'after a CANCEL: image_numbers 0')
    # a client that left: the front-end connection closes, and the engine ends the requests it carried
    req2, rows = big(e, T, S0 + 21, max_new=2000)
    need(wait_numbers(e, numbers_for(rows), 60) == numbers_for(rows), 'the second request holds its numbers')
    e.s.close()
    e = Conn(a.socket)
    check(wait_numbers(e, 0, a.wait) == 0, 'after the client closed its connection: image_numbers 0')
    # a RELEASE of its session while it runs (applied at its end)
    req3, rows = big(e, T, S0 + 22, max_new=200)
    need(wait_numbers(e, numbers_for(rows), 60) == numbers_for(rows), 'the third request holds its numbers')
    e.release(S0 + 22)
    o = collect(e, req3)
    check(not o['error'], 'a RELEASE while it runs: the request still ends: %s' % o['reason'])
    check(wait_numbers(e, 0, a.wait) == 0, 'after a RELEASE of its session: image_numbers 0')


# ---- resume ---------------------------------------------------------------------------------------------------------------

def t_resume(a, e, T):
    need_s3(e)
    log_off = os.path.getsize(a.log) if a.log else 0
    text, spans = layout(T, 3000, 1, 700, 2500)
    alone = run_image(e, S0 + 30, text, spans, 5, max_new=a.new, keep=False)
    need(not alone['error'], 'the request alone ran (%d tokens)' % len(alone['ids']))
    req = image_submit(e, S0 + 31, text, spans, 5, max_new=a.new, keep=False)
    dec = [e.submit(S0 + 32 + i, T[60000 + 20000 * i:60000 + 20000 * i + 12000], max_new=a.new, keep=False, stop=()) for i in range(2)]
    ids = {req[0]: 'image'}
    ids.update({d[0]: 'text%d' % i for i, d in enumerate(dec)})
    out = {'image': {'ids': [], 'lp': [], 'top': []}}
    done = set()
    while len(done) < 3:
        t, p = e.frame()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            if ids[r] == 'image':
                out['image']['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
                off = 16 + 4 * n
                if fl & 1:
                    out['image']['lp'] += struct.unpack('<%df' % n, p[off:off + 4 * n])
        elif t == DONE:
            done.add(struct.unpack('<Q', p[:8])[0])
    check(out['image']['ids'] == alone['ids'], 'the image request beside two decoders equals its alone run (%d tokens)' % len(alone['ids']))
    check(out['image']['lp'] == alone['lp'], 'and so do its logprobs')
    time.sleep(1.5)
    check(image_numbers(e) == 0, 'all ended: image_numbers 0')
    if a.log:
        with open(a.log, 'rb') as f:
            f.seek(log_off)
            txt = f.read().decode(errors='replace')
        held = re.findall(r'images: \d+ spans, \d+ rows, held in', txt)
        need(len(held) >= 1, 'rank 0 logged %d go(es) that held the rows and sent none (a resume)' % len(held))
        print('  log: %d "held" lines, %d "taken and sent" lines' % (len(held), len(re.findall(r'taken and sent', txt))))


# ---- pair -----------------------------------------------------------------------------------------------------------------

def t_pair(a, e, T):
    need_s3(e)
    tx, sx = layout(T, 5000, 1, 600, 6000)
    ty, sy = layout(T, 30000, 1, 900, 3000)
    ax = run_image(e, S0 + 40, tx, sx, 6, keep=False)
    ay = run_image(e, S0 + 41, ty, sy, 7, keep=False)
    peak = 0
    dec = e.submit(S0 + 42, T[70000:72000], max_new=200, keep=False, stop=())
    rx = image_submit(e, S0 + 43, tx, sx, 6, keep=False)
    ry = image_submit(e, S0 + 44, ty, sy, 7, keep=False)
    res, pending = {}, {rx[0]: 'x', ry[0]: 'y', dec[0]: 'd'}
    outs = {'x': {'ids': [], 'lp': []}, 'y': {'ids': [], 'lp': []}, 'd': {'ids': [], 'lp': []}}
    while pending:
        if not e.q:  # nothing kept: ask STATS on this connection (request frames that come first are kept)
            peak = max(peak, e.numbers() or 0)
        t, p = e.frame()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            k = pending.get(r)
            if k:
                outs[k]['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
                if fl & 1:
                    outs[k]['lp'] += struct.unpack('<%df' % n, p[16 + 4 * n:16 + 8 * n])
        elif t == DONE:
            pending.pop(struct.unpack('<Q', p[:8])[0], None)
    check(outs['x']['ids'] == ax['ids'] and outs['x']['lp'] == ax['lp'], 'x together equals x alone')
    check(outs['y']['ids'] == ay['ids'] and outs['y']['lp'] == ay['lp'], 'y together equals y alone')
    want = numbers_for(600) + numbers_for(900)
    check(peak >= want - numbers_for(900) and peak <= want, 'image_numbers peaked at %d (both together: %d)' % (peak, want))
    time.sleep(1.5)
    check(image_numbers(e) == 0, 'all ended: image_numbers 0')


# ---- ttft -----------------------------------------------------------------------------------------------------------------

def t_ttft(a, e, T):
    ts = []
    for i in range(5):
        text, spans = layout(T, 7000 + 9000 * i, 1, 382, 3500)
        o = run_image(e, S0 + 50 + i, text, spans, 10 + i, max_new=2, keep=False)
        ts.append(o['prefill_us'] / 1e3)
    med = statistics.median(ts)
    print('  prefill (first token) ms: %s; median %.1f, spread %.1f' % (' '.join('%.0f' % t for t in ts), med, max(ts) - min(ts)))
    if a.save:
        json.dump({'ms': ts}, open(a.save, 'w'))
    if a.ref:
        r = statistics.median(json.load(open(a.ref))['ms'])
        check(med <= r * 1.05, 'median %.1f ms against the reference %.1f ms (floor 5 %%)' % (med, r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('cmd', choices=['replies', 'leak', 'cancel', 'resume', 'pair', 'ttft'])
    ap.add_argument('--save')
    ap.add_argument('--ref')
    ap.add_argument('--nodes', default='')
    ap.add_argument('--spread', type=float, default=0.15)
    ap.add_argument('--engine-tol-mib', type=float, default=256, dest='engine_tol_mib')
    ap.add_argument('--rows-spans', type=int, default=36, dest='rows_spans')
    ap.add_argument('--wait', type=float, default=60)
    ap.add_argument('--new', type=int, default=400)
    ap.add_argument('--log')
    a = ap.parse_args()
    T = [int(x) for x in load_npy_i32(a.tokens)]
    assert len(T) >= 100000, 'the token file is too short'
    e = Conn(a.socket)
    print('engine: %s, context %d' % (e.name, e.max_ctx))
    if a.cmd == 'replies' and not (a.save or a.ref):
        ap.error('replies needs --save or --ref')
    try:
        {'replies': t_replies, 'leak': t_leak, 'cancel': t_cancel, 'resume': t_resume, 'pair': t_pair, 'ttft': t_ttft}[a.cmd](a, e, T)
    except NotReached as x:
        print('image_store_test %s: NOT REACHED (%s)' % (a.cmd, x))
        sys.exit(2)
    print('image_store_test %s: %s' % (a.cmd, 'FAIL' if fails else 'PASS'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
