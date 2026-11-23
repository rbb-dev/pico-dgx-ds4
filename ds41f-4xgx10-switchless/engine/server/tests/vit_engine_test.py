#!/usr/bin/env python3
"""The image encoder in the engine from outside, on the four Sparks (model/vit_area.cu, model/vit_cmd.cu, server/vit_sched.h, serve.cpp's
vit_step): pixel requests through pixel_client.py's frame, the encoded rows read back from every rank, the scheduler's units beside
decoding sessions, the work area at the start. The host tests (vit_sched_test, interleave_test) cover the rules; this is the engine's proof.

WHAT IT NEEDS. A FAULT BUILD (cmake -DDSGB10_TEST_FAULTS=ON) with the engine started on all four Sparks, rank 0's socket named, dsfront
stopped (the engine serves one front-end connection at a time, so this script's one connection carries the decoding sessions, STATS and the
pixel requests), the usual production settings (--vision-weights, --pool-free-gib, the 1M window), DSGB10_STORE_CHECK=1 in every rank's
environment, and for the byte tests DSGB10_VIT_DUMP=DIR (a directory every Spark has; each rank writes the rows it encoded) and
DSGB10_VIT_POISON=1. The images are the reference run's as `.u8` files (588 * H * W bytes, FILE:HxW as pixel_client.py), and --ref DIR holds each image's
unsplit span rows, `<basename of the image file>.span.bin`, written by `vit_ring_test --only correct --raw FILE:HxW ... --dump-spans DIR`
(the unsplit run on rank 0's whole weights: F3a's test). --hosts names rank 0..3's ssh targets ('local' for this machine) so the dumps are read.

  bytes     every image of --image alone, then all of them in one request: every rank's rows equal the others' and the unsplit run's;
            --sessions 8 runs it beside 8 decoding sessions (the longest gap between their tokens printed).         [vit_engine_bytes]
  ranges    a clip of --n copies of --clip beside --sessions decoders: with DSGB10_VIT_BUDGET_MS=0.001 in the engine's environment every
            image goes in 32 one-block commands with the decoders' steps (and a second request's row pieces, --filler N text tokens)
            between; without it the same clip in the scheduler's own pieces. Each run checks the rows against the unsplit run's, so the
            two runs are equal to each other.                                                                        [vit_engine_ranges]
  two       two requests at once (--clip image X and --clip2 image Y, --n each): their images run one at a time, every flight's rows equal
            to its image's alone. --expect-mid: the engine was started with DSGB10_VIT_MIDTEST=1, which starts a second image while the
            first is between its blocks: the engine must stop (EOF) and the logs say why (check-log --expect mid).     [vit_two]
            --hold: the arm for the rule that a pixel request is neither started nor resumed while another request's image is between
            its blocks (the engine has DSGB10_VIT_BUDGET_MS=0.001, --sessions 8): --clip x --n is the long request read a block at a time;
            while it is between two ranges a single-image request (--image) is sent, and --paused pixel requests that were decoding
            before it (started first, --fillers text sessions beside them so they are paused and resumed) are in flight. The engine
            lives; each reply equals the same request run alone first (greedy); the single request's wait (its DONE queue time) is
            printed and must be under --wait-max-s (the first image's remaining time: its 32 ranges and the decoders' windows).
  cancel    cancels a clip after each of --delays seconds (the largest image's command, after image k, between two block ranges when the
            engine has DSGB10_VIT_BUDGET_MS=0.001), then a single image of --image on the same connection: its rows are right every time;
            --vote: the engine has DSGB10_VIT_FAIL=rank:R,at:vote: the first request is answered with the ranks-disagree error, the
            next is right.                                                                                              [vit_cancel]
  resume    turn 1 of a kept session with --image, then turn 2 (the same prompt and more) with --fillers other sessions beside it, until the
            engine's log (--log rank0.log) shows the session read again from the start; turn 2's reply is compared with the same turn run
            alone (greedy): the image is encoded after the resume line.                                               [vit_resume]
  arena     the start: with --logs (the four ranks') the tenant line "work area (device memory, required): 2 groups", the hand-over's proof,
            the four ranks' lines equal; STATS tenant_groups[3] = 2 and the pool exactly 128 numbers under the pool of the start line;
            --fault room (a pool sized so that the room is one group: the area, which needs two, is refused whole: 0 groups and no block on
            every rank) or --fault alloc (the engine started with DSGB10_VIT_FAIL=rank:R,at:alloc): every rank at 0 groups, a text request equal to
            --text-ref (saved by --save-text on a normal start), a pixel request answered with the error and the connection alive. [vit_arena]
  clip      the arm of the pass line: --clip beside --sessions 8, --runs 5: the decoders' longest silence inside the request (median at
            most the gap, no run above 0.70 s), the clip's first token against --clip-first-s (14.979), the single --single image's
            engine-side time to its first token against --single-first-s (0.737).                                        [section 3's arm]
  check-log after any of them: --logs rank0.log .. rank3.log: no BUG line, equal store digests on the four ranks line by line, the
            expected fault lines.
Every check prints ok / FAIL; the exit status is 0 only when all hold. Each test says in its output how many numbers the requests' rows and
conversations hold against the pool's free numbers before each step (STATS 34, 2, 1). Not run: written for the main session's window.
"""
import argparse
import hashlib
import os
import re
import select
import statistics
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tenant_test as tt  # noqa: E402
from pixel_client import IMAGE_ID, F_PIXELS, pad8, rows_of  # noqa: E402
from serve_test import CANCEL, DONE, F_KEEP, REASONS, SUBMIT, TOKENS, load_npy_i32  # noqa: E402

POOL_FREE, POOL_HELD, IMAGE_NUMBERS, TENANT3 = 1, 2, 34, tt.TENANT0 + 3
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def sh(host, cmd):
    argv = ['bash', '-c', cmd] if host == 'local' else ['ssh', '-o', 'BatchMode=yes', host, cmd]
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout


class Image:
    def __init__(self, spec):
        f, hw = spec.rsplit(':', 1)
        self.path, (self.h, self.w) = f, (int(x) for x in hw.split('x'))
        self.name = os.path.basename(f)
        self.px = open(f, 'rb').read()
        assert len(self.px) == 588 * self.h * self.w, (self.name, len(self.px))
        self.span_rows = rows_of(self.h, self.w)
        self.hash = int.from_bytes(hashlib.sha256(self.px).digest()[:8], 'little')  # a distinct image, a distinct hash: no KV of another image is reused


class Rig(tt.Conn):
    """tenant_test's connection (decoding sessions, STATS) that also sends pixel requests and keeps each request's answer."""

    def __init__(self, sock, tokens, sessions):
        self.req = {}
        self._run, self._n = int(time.time()) & 0xFFFF, 0
        super().__init__(sock, tokens, sessions)

    def fresh(self, tag=0):
        """A session id never used on this engine start: a second run of the same call (alone, then beside decoders) continues no kept session."""
        self._n += 1
        return (self._run << 40) | (self._n << 16) | (tag & 0xFFFF)

    def handle(self, t, p):
        if t == TOKENS:
            r, n, _ = struct.unpack('<QII', p[:16])
            q = self.req.get(r)
            if q is not None:
                q['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
                if q['first'] is None:
                    q['first'] = time.time() - q['t0']
        elif t == DONE:
            r, reason, pt, cached, comp, qus, pus, dus, el, _ = struct.unpack('<QIIIIQQQII', p[:56])
            q = self.req.get(r)
            if q is not None:
                q.update(done=True, reason=REASONS[reason], error=p[len(p) - el:].decode(), prefill_s=pus / 1e6, queue_s=qus / 1e6, t_done=time.time())
        super().handle(t, p)

    def pixels(self, images, session, text, max_new=8, keep=True, fresh=True, stop=(1,), tail=None):
        """A pixel request: the text ids, then each image's span of IMAGE_ID ids (as pixel_client.pixel_submit); returns its id."""
        session = self.fresh(session) if fresh else session
        rid = self.e.next_id
        self.e.next_id += 1
        prompt, spans = list(text), b''
        for im in images:
            spans += struct.pack('<IIQ', len(prompt), im.span_rows, im.hash)
            prompt += [IMAGE_ID] * im.span_rows
        # the engine reads a span only with a text token after it (serve.cpp's admission: "bad image span"), as a chat prompt has
        prompt += list(tail) if tail is not None else list(text[-4:]) if len(text) >= 4 else [1, 2, 3, 4]  # tail: the text after the images (a chat prompt's)
        hdr = struct.pack('<QQQQffIIIIIIII', rid, session, 0, 0, 0.0, 1.0, 0, max_new, (F_KEEP if keep else 0) | F_PIXELS, 0, 0, len(stop), len(prompt), 0)
        grids = b''.join(struct.pack('<HH', im.h, im.w) for im in images)
        grids += b'\0' * (pad8(len(grids)) - len(grids))
        data = b''.join(im.px + b'\0' * (pad8(len(im.px)) - len(im.px)) for im in images)
        self.e.send(SUBMIT, hdr + struct.pack('<%dI' % len(stop), *stop) + struct.pack('<%dI' % len(prompt), *prompt) + struct.pack('<I', len(images)) + spans + grids + data)
        self.req[rid] = dict(t0=time.time(), first=None, done=False, ids=[], reason=None, error='')
        return rid

    def text(self, session, ids, max_new=16, keep=False, stop=(1,)):
        rid = self.e.submit(self.fresh(session), ids, max_new=max_new, keep=keep, stop=stop)[0]
        self.req[rid] = dict(t0=time.time(), first=None, done=False, ids=[], reason=None, error='')
        return rid

    def wait(self, rid, timeout=300):
        self.pump(time.time() + timeout, lambda: self.req[rid]['done'])
        return self.req[rid]

    def numbers(self, tag):
        s = self.stats()
        if s:
            print('  [%s] pool free %d, held by conversations %d, image rows %d numbers, work area %d groups' % (tag, s[POOL_FREE], s[POOL_HELD], s[IMAGE_NUMBERS], s[TENANT3]))
        return s

    def longest_gap(self, t0, t1):
        return max((g for ts, g in self.gaps if t0 <= ts <= t1), default=0.0)


# ---- the dumps (DSGB10_VIT_DUMP) ----

def dump_clear(hosts, d):
    for h in hosts:
        sh(h, 'rm -f %s/rank*-key*-span*.bin' % d)


def dump_read(hosts, d):
    """{key: {span: [bytes of rank 0..3]}}"""
    out = {}
    for r, h in enumerate(hosts):
        for name in sh(h, 'ls %s' % d).decode().split():
            m = re.fullmatch(r'rank(\d)-key(\d+)-span(\d+)\.bin', name)
            if m and int(m.group(1)) == r:
                out.setdefault(int(m.group(2)), {}).setdefault(int(m.group(3)), [None] * 4)[r] = sh(h, 'cat %s/%s' % (d, name))
    return out


def ref_of(refdir, im):
    return open(os.path.join(refdir, im.name + '.span.bin'), 'rb').read()


def check_flight(flight, images, refdir, tag):
    """flight: {span: [4 byte strings]}; images in span order"""
    check(sorted(flight) == list(range(len(images))), '%s: every span was written (%s)' % (tag, sorted(flight)))
    for s, im in enumerate(images):
        rows = flight.get(s)
        if rows is None or None in rows:
            check(False, '%s: image %d: a rank wrote nothing' % (tag, s))
            continue
        check(all(b == rows[0] for b in rows), '%s: image %d: the four ranks hold equal bytes' % (tag, s))
        check(rows[0] == ref_of(refdir, im), '%s: image %d: equal to the unsplit run (%d bytes)' % (tag, s, len(rows[0])))


def ok_done(q, tag):
    check(q['done'] and q['reason'] in ('stop', 'length') and not q['error'], '%s: answered (%s %r)' % (tag, q['reason'], q['error']))


# ---- the tests ----

def cmd_bytes(a):
    c = Rig(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)
    images = [Image(s) for s in a.image]
    for k, im in enumerate(images + [None]):
        dump_clear(a.hosts, a.dump)
        use = [im] if im else images
        c.numbers('before %s' % (im.name if im else 'all images'))
        t0 = time.time()
        q = c.wait(c.pixels(use, 0x76650000 + k, c.T[:200]))
        t1 = time.time()
        ok_done(q, im.name if im else 'all images')
        flights = dump_read(a.hosts, a.dump)
        check(len(flights) == 1, 'one flight wrote rows (%s)' % sorted(flights))
        for fl in flights.values():
            check_flight(fl, use, a.ref, im.name if im else 'all images')
        if a.sessions:
            print('  decoders: longest gap %.0f ms inside the request (%.0f ms)' % (1000 * c.longest_gap(t0, t1), 1000 * (t1 - t0)))
    c.close()
    return 0


def cmd_ranges(a):
    c = Rig(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)
    clip = Image(a.clip)
    dump_clear(a.hosts, a.dump)
    c.numbers('before')
    t0 = time.time()
    rid = c.pixels([clip] * a.n, 0x76670000, c.T[:200])
    filler = c.text(0x76670001, c.T[10 * 2048:10 * 2048 + a.filler]) if a.filler else None  # another request's row pieces between the encode's commands
    q = c.wait(rid, a.timeout_s or 20.0 * a.n)  # one-block commands: 32 per image, each between the decoders' windows
    t1 = time.time()
    ok_done(q, 'the clip of %d images' % a.n)
    if filler:
        c.wait(filler)
    c.numbers('after')
    flights = dump_read(a.hosts, a.dump)
    check(len(flights) == 1, 'one flight wrote rows')
    for fl in flights.values():
        check_flight(fl, [clip] * a.n, a.ref, 'the clip')
    print('  decoders: longest gap %.0f ms inside the request (%.0f ms); the clip\'s first token %.2f s, engine-side %.2f s' %
          (1000 * c.longest_gap(t0, t1), 1000 * (t1 - t0), q['first'] or -1, q.get('prefill_s', -1)))
    c.close()
    return 0


def cmd_hold(a):
    c = Rig(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)
    big, one = Image(a.clip), Image(a.image[0])
    text = c.T[:200]
    # controls: each request alone, greedy
    ctl_one = c.wait(c.pixels([one], 0x76680001, text, max_new=16))
    ok_done(ctl_one, 'the single image alone')
    ctl_p = [c.wait(c.pixels([one], 0x76680100 + k, c.T[400 + 300 * k:700 + 300 * k], max_new=a.tokens_out)) for k in range(a.paused)]
    for k, q in enumerate(ctl_p):
        ok_done(q, 'paused-request %d alone' % k)
    c.numbers('before')
    # the paused requests decode first, fillers beside them so their sessions lose their lines
    ps = [c.pixels([one], 0x76680200 + k, c.T[400 + 300 * k:700 + 300 * k], max_new=a.tokens_out) for k in range(a.paused)]
    fl = [c.text(0x76680300 + k, c.T[(12 + k) * 2048:(13 + k) * 2048], max_new=a.tokens_out) for k in range(a.fillers)]
    c.pump(time.time() + 3)
    rb = c.pixels([big] * a.n, 0x76680400, text, max_new=8)
    c.pump(time.time() + 0.3)  # the long request is between two ranges of its first image
    r1 = c.pixels([one], 0x76680401, text, max_new=16)
    try:
        q1, qb = c.wait(r1, a.timeout_s or 20.0 * a.n), c.wait(rb, a.timeout_s or 20.0 * a.n)
        qps = [c.wait(r) for r in ps]
        for r in fl:
            c.wait(r)
    except EOFError:
        check(False, "the engine lives: it closed the connection (a pixel request met another's image between its blocks)")
        return 0
    ok_done(q1, 'the single image sent between two ranges')
    ok_done(qb, 'the long request')
    check(q1['ids'] == ctl_one['ids'], 'its reply equals the same request alone (%d tokens)' % len(q1['ids']))
    for k, q in enumerate(qps):
        ok_done(q, 'paused request %d' % k)
        check(q['ids'] == ctl_p[k]['ids'], 'paused request %d: its reply equals the same request alone (%d tokens)' % (k, len(q['ids'])))
    wait_s = q1.get('queue_s')
    print('  the single image waited %s s (engine queue time) behind the first image' % ('%.2f' % wait_s if wait_s is not None else 'no DONE'))
    check(wait_s is not None and wait_s <= a.wait_max_s, "its wait is under the first image's remaining time (%.1f s)" % a.wait_max_s)
    c.numbers('after')
    c.close()
    return 0


def cmd_two(a):
    if a.hold:
        return cmd_hold(a)
    c = Rig(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)
    x, y = Image(a.clip), Image(a.clip2)
    dump_clear(a.hosts, a.dump)
    c.numbers('before')
    try:
        ra = c.pixels([x] * a.n, 0x76740000, c.T[:200])
        rb = c.pixels([y] * a.n, 0x76740001, c.T[400:600])
        qa, qb = c.wait(ra, 20.0 * a.n), c.wait(rb, 20.0 * a.n)
    except (EOFError, BrokenPipeError, ConnectionResetError):  # the engine's stop is the expected end of the mid arm
        check(a.expect_mid, 'the engine closed the connection: expected only with --expect-mid')
        print('  the engine stopped: check-log --expect mid names why')
        return 0
    check(not a.expect_mid, 'the engine did not stop: DSGB10_VIT_MIDTEST=1 should have made the second image start mid-way')
    ok_done(qa, 'request A')
    ok_done(qb, 'request B')
    flights = dump_read(a.hosts, a.dump)
    check(len(flights) == 2, 'two flights wrote rows (%s)' % sorted(flights))
    want = {'A': (x, a.n), 'B': (y, a.n)}
    seen = set()
    for key, fl in sorted(flights.items()):
        for name, (im, n) in want.items():
            if name not in seen and all(rows and None not in rows and rows[0] == ref_of(a.ref, im) for rows in fl.values()) and len(fl) == n:
                seen.add(name)
                check_flight(fl, [im] * n, a.ref, 'flight %d (request %s)' % (key, name))
                break
    check(seen == {'A', 'B'}, 'each flight\'s rows equal its own image\'s, one flight each (%s)' % sorted(seen))
    c.numbers('after')
    c.close()
    return 0


def cmd_cancel(a):
    c = Rig(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)
    clip, one = Image(a.clip), Image(a.image[0])
    if a.vote:
        dump_clear(a.hosts, a.dump)
        q = c.wait(c.pixels([one], 0x76630000, c.T[:200]))
        check(q['reason'] == 'error' and 'disagree' in q['error'], 'the vote made to differ ends the request with an error (%r), the connection lives' % q['error'])
        dump_clear(a.hosts, a.dump)
        q = c.wait(c.pixels([one], 0x76630001, c.T[:200]))
        ok_done(q, 'the next request (the fault fires once)')
        for fl in dump_read(a.hosts, a.dump).values():
            check_flight(fl, [one], a.ref, 'the next request')
        s = c.numbers('after the vote')
        check(s[IMAGE_NUMBERS] == 0, 'the failed request\'s rows were freed (image numbers %d)' % s[IMAGE_NUMBERS])
        c.close()
        return 0
    for k, delay in enumerate(float(x) for x in a.delays.split(',')):
        c.numbers('before the cancel at %.2f s' % delay)
        rid = c.pixels([clip] * a.n, 0x76630100 + 2 * k, c.T[:200], keep=False)  # not kept: its rows are the only image numbers it can hold
        c.pump(time.time() + delay)
        c.e.send(CANCEL, struct.pack('<Q', rid))
        q = c.wait(rid)
        print('  cancel at %.2f s: %s (%d tokens)' % (delay, q['reason'], len(q['ids'])))
        check(q['reason'] in ('cancelled', 'stop', 'length'), 'the cancelled request ended (%s)' % q['reason'])
        s = c.numbers('after the cancel')
        t_end = time.time() + 2.0  # STATS are published on a cadence: the freed rows show at the next reading, within 2 s
        while s[IMAGE_NUMBERS] != 0 and time.time() < t_end:
            c.pump(time.time() + 0.25)
            s = c.stats()
        check(s[IMAGE_NUMBERS] == 0, 'its rows were freed (image numbers %d, read for up to 2 s: the STATS cadence)' % s[IMAGE_NUMBERS])
        dump_clear(a.hosts, a.dump)
        q = c.wait(c.pixels([one], 0x76630101 + 2 * k, c.T[:200]))
        ok_done(q, 'the next request after a cancel at %.2f s' % delay)
        for fl in dump_read(a.hosts, a.dump).values():
            check_flight(fl, [one], a.ref, 'the next request\'s image (after a cancel at %.2f s)' % delay)
    c.close()
    return 0


def cmd_resume(a):
    """Arm B of checkpoint the image BEFORE the text. Turn 1 = 64 ids, the image, 2,936 ids; turn 2 = the same and 400 more, so its restart
    lands past the image (E: a bookmark at 2,048 or later; the premise line decides) and nothing is encoded in its first run. Beside fillers it is
    evicted and read again from the start: the image is then encoded at that resume, into numbers taken at its first go."""
    im = Image(a.image[0])
    log = a.log
    img_end = 64 + im.span_rows

    def read_log(n0):
        return open(log, errors='replace').read()[n0:]

    def turn2(c, sid, fillers):
        sid = c.fresh(sid)  # the same session for the two turns of one call, a new one for each call
        c.numbers('before turn 1 (session %x)' % sid)
        q1 = c.wait(c.pixels([im], sid, c.T[:64], max_new=8, fresh=False, stop=(), tail=c.T[64:3000]))
        ok_done(q1, 'session %x turn 1' % sid)
        c.numbers('before turn 2')
        n0 = len(open(log, errors='replace').read())
        dump_clear(a.hosts, a.dump)
        rid = c.pixels([im], sid, c.T[:64], max_new=a.tokens_out, fresh=False, stop=(), tail=c.T[64:3400])  # no stop ids: it runs to --tokens-out
        frids = [c.text(0x76720000 + 16 * sid % 4096 + k, c.T[(11 + k) * 2048:(12 + k) * 2048], max_new=a.tokens_out, stop=()) for k in range(fillers)]
        c.pump(time.time() + 5)
        c.numbers('during turn 2: the pixel session and %d fillers decoding' % fillers)
        q = c.wait(rid)
        for fr in frids:
            c.wait(fr)
        c.numbers('after turn 2')
        time.sleep(1)
        return q, rid, read_log(n0), dump_read(a.hosts, a.dump)

    c = Rig(a.socket, a.tokens, 0)
    g0 = len(open(log, errors='replace').read())
    alone, ra, la, fa = turn2(c, 0x76720001, 0)
    ok_done(alone, 'the unpaused turn 2')
    paused, rp, lp, fp = turn2(c, 0x76720002, a.fillers)
    ok_done(paused, 'the turn 2 beside %d fillers' % a.fillers)
    full = read_log(g0)

    def reused(text, rid):
        m = re.search(r'serve: req %d factor \d+ \(window \d+\): restart: [^\n]*?reused (\d+) \(([a-z ]+)\)' % rid, text)
        return (int(m.group(1)), m.group(2)) if m else None

    # premises, each printed; exit 2 when one fails
    ok = True
    for name, rid, text in (('the turn alone', ra, la), ('the turn beside the fillers', rp, lp)):
        ru = reused(text, rid)
        enc_before = re.search(r'serve: req %d image 1 of 1 encoded' % rid, text)
        print('  P1 %s (request %d): restart %s, the image ends at %d' % (name, rid, ru, img_end))
        if ru is None or ru[0] < img_end:
            print('  NOT REACHED (P1): its restart is not at or past the image\'s end')
            ok = False
    if fa:
        print('  NOT REACHED (P1): the turn alone wrote image rows (%s): it encoded in its first run' % sorted(fa))
        ok = False
    if re.search(r'serve: req %d image 1 of 1 encoded' % ra, la):
        print('  NOT REACHED (P1): the turn alone has an "image 1 of 1 encoded" line')
        ok = False
    read_again = re.search(r'serve: req %d factor \d+ \(window \d+\): a new session for its \d+ positions, read again' % rp, lp)
    if read_again is None:
        print('  NOT REACHED (P2): no "read again" line of the pixel request\'s own id %d: the fillers did not evict it in this pool' % rp)
        ok = False
    else:
        before = lp[:read_again.start()]
        if re.search(r'serve: req %d image 1 of 1 encoded' % rp, before):
            print('  NOT REACHED (P1): an "image 1 of 1 encoded" line of request %d precedes its first read-again' % rp)
            ok = False
    if not ok:
        c.close()
        return 2
    # pass
    after = lp[read_again.start():]
    held = re.search(r'serve: req %d images: [^\n]*held' % rp, after)
    encs = [m.start() for m in re.finditer(r'serve: req %d image 1 of 1 encoded' % rp, lp)]
    check(held is not None, '(a) after the read-again line the engine prints "req %d images: ... held"' % rp)
    check(len(encs) == 1 and held is not None and encs[0] > read_again.start() + held.start(),
          '(a) then exactly one "image 1 of 1 encoded" of request %d in the whole turn, after the held line (%d found)' % (rp, len(encs)))
    check(bool(fp) and len(fp) == 1, '(b) the dump of the turn: one flight (%s)' % sorted(fp))
    for fl in fp.values():
        check_flight(fl, [im], a.ref, 'the resumed turn (the resume\'s own write)')
    check(not fa, '(c) the turn alone wrote nothing')
    nxt = re.search(r'serve: req (\d+) images: [^\n]*\((\d+) image numbers held\)', full[full.index('serve: req %d ' % ra):].split('serve: req %d images' % ra, 1)[-1]) if ('serve: req %d images' % ra) in full else None
    nums = [int(m.group(2)) for m in re.finditer(r'serve: req (\d+) images: [^\n]*\((\d+) image numbers held\)', full)]
    print('  image numbers held on each images line of the run: %s' % nums)
    if nxt:
        check(int(nxt.group(2)) == 2, '(c) the first images line after the turn alone says 2 image numbers held (its numbers are back), not 4 (%s)' % nxt.group(2))
    check('BUG' not in full, '(c) no BUG line in this run\'s log')
    # pico is not batch-invariant (ENGINE-FACTS): printed, not judged
    print('  turn 2 reply, unpaused: %s' % list(alone['ids'][:24]))
    print('  turn 2 reply, beside the fillers: %s (equal: %s; printed, not judged)' % (list(paused['ids'][:24]), paused['ids'] == alone['ids']))
    c.close()
    return 0


def cmd_arena(a):
    texts = [open(p, errors='replace').read() for p in a.logs] if a.logs else []
    fault = a.fault in ('alloc', 'room')
    if texts:
        line = re.search(r'store tenant 3, work area \(device memory, required\): (\d+) groups of ([\d.]+) MiB', texts[0])
        check(line is not None and int(line.group(1)) == (0 if fault else 2), 'rank 0\'s tenant line: work area (device memory, required): %s groups' % (line.group(1) if line else '?'))
        if not fault:
            check(re.search(r'store: work area 0 -> 2 groups', texts[0]) is not None, 'the hand-over\'s proof is printed (rank 0: "store: work area 0 -> 2 groups")')
        if a.fault == 'room':
            print("  premise: the pool's room for the area, from the start table: %s" % re.findall(r'^chunk pool: [^\n]*', texts[0], re.M)[:1])
        got = [re.search(r'rank %d: vision work area: .*\(status (\d+)\): (.*)' % r, t) for r, t in enumerate(texts)]
        check(all(got), 'every rank printed its work area line')
        if all(got):
            want = 'the engine has no image encoder until the next start' if fault else 'the engine encodes images'
            check(all(g.group(2) == want for g in got), 'on every rank: %s' % want)
            check(len({g.group(1) for g in got}) == 1, 'the four ranks report the same status (%s)' % got[0].group(1))
        # a start's hand-over writes no "lend command of tenant" lines (none does): the four ranks' equality is check-log's store digests (traces SAME)
        pool = re.search(r'^chunk pool: (\d+) numbers', texts[0], re.M)  # the start line, not model's per-slot 'model: chunk pool' line
        eg = re.search(r'store citizens: Engram cache \d+ groups \((\d+) numbers\)', texts[0])
        engram_numbers = int(eg.group(1)) if eg else 0  # the other tenant lent at the same hand-over
    c = Rig(a.socket, a.tokens, 0)
    s = c.stats()
    check(s[TENANT3] == (0 if fault else 2), 'STATS tenant_groups[3] = %d' % s[TENANT3])
    if texts and pool and not fault:
        check(s[0] == int(pool.group(1)) - engram_numbers - 128, 'the pool is the start line\'s %d less the Engram cache\'s %d and the area\'s 128: %d (STATS %d)' % (int(pool.group(1)), engram_numbers, int(pool.group(1)) - engram_numbers - 128, s[0]))
    if a.pool_noflag:
        print('  the pool of a start without --vision-weights: %d; this start\'s: %d; the weights\' own cost and the area\'s 128: %d' % (a.pool_noflag, s[0], a.pool_noflag - s[0]))
    if a.save_text:
        q = c.wait(c.text(0x76610000, c.T[:500], max_new=32))
        open(a.save_text, 'w').write(' '.join(map(str, q['ids'])))
        print('  saved %d tokens of a text request to %s' % (len(q['ids']), a.save_text))
    if fault:
        q = c.wait(c.text(0x76610001, c.T[:500], max_new=32))
        ref = [int(x) for x in open(a.text_ref).read().split()]
        check(q['ids'] == ref, 'a text request\'s reply equals the control\'s (%d tokens)' % len(ref))
        q = c.wait(c.pixels([Image(a.image[0])], 0x76610002, c.T[:200]))
        check(q['reason'] == 'error' and 'work area' in q['error'], 'a pixel request is answered with the error (%r)' % q['error'])
        q = c.wait(c.text(0x76610003, c.T[:500], max_new=8))
        check(q['reason'] in ('stop', 'length'), 'the connection is alive: the next text request runs')
    return 0


UNIT_LINE = re.compile(r'serve: req (\d+) read in pieces beside decoders: (\d+) encode units \(longest ([\d.]+) ms, longest silence across one ([\d.]+) ms\), '
                       r'(\d+) row pieces \(longest ([\d.]+) ms, longest silence across one ([\d.]+) ms\)')


def unit_line(log, rid, start_re):
    """rank 0's per-request line for request rid in this start's log (waits up to 10 s for it to be written): (encode units, encode silence ms,
    row pieces, row silence ms) or None"""
    for _ in range(20):
        text = last_start(open(log, errors='replace').read(), start_re)
        for m in UNIT_LINE.finditer(text):
            if int(m.group(1)) == rid:
                return int(m.group(2)), float(m.group(4)), int(m.group(5)), float(m.group(7))
        time.sleep(0.5)
    return None


def cmd_clip(a):
    """The clip row: from the engine's own line per request (rank 0's log), (a) the longest silence across an encode unit <= the gap,
    (b) the client's longest silence = the larger of the line's two figures within 20 ms, (c) the clip's first token, (d) the single image,
    (e) a control: a 40,000-token text prompt beside the same decoders, its row pieces' silence printed beside the clip's, not judged."""
    clip, one = Image(a.clip), Image(a.single)
    c = Rig(a.socket, a.tokens, a.sessions)
    c.pump(time.time() + 20)
    first, single, enc_sil, row_sil = [], [], [], []
    for k in range(a.runs):
        t0 = time.time()
        rid = c.pixels([clip] * a.n, 0x76630000 + k, c.T[:200])
        q = c.wait(rid)
        t1 = time.time()
        ok_done(q, 'clip run %d' % (k + 1))
        client = c.longest_gap(t0, t1) * 1000.0
        first.append(q['first'])
        ln = unit_line(a.log, rid, a.start_re)
        if ln is None:
            check(False, 'clip run %d (request %d): rank 0 printed no "read in pieces beside decoders" line' % (k + 1, rid))
        else:
            en, es, rn, rs = ln
            enc_sil.append(es), row_sil.append(rs)
            check(es <= a.gap_ms, 'run %d: the longest silence across an encode unit %.0f ms (%d units) <= the gap %d ms' % (k + 1, es, en, a.gap_ms))
            big = max(es, rs)
            check(abs(client - big) <= 20.0, 'run %d: the client\'s longest silence %.0f ms equals the larger of the line\'s two figures %.0f ms (encode %.0f, rows %.0f) within 20 ms' % (k + 1, client, big, es, rs))
        c.pump(time.time() + 5)
        q1 = c.wait(c.pixels([one], 0x76640000 + k, c.T[:200]))
        single.append(q1['prefill_s'])
        c.pump(time.time() + 5)
        print('  run %d: client silence %.0f ms, clip first token %.3f s, single image engine-side %.3f s' % (k + 1, client, first[-1], single[-1]))
    check(statistics.median(first) <= a.clip_first_s, 'the clip\'s first token: median %.3f s <= %.3f s (F4a\'s own)' % (statistics.median(first), a.clip_first_s))
    check(statistics.median(single) <= a.single_first_s, 'one image beside decoders, request read to first token: median %.3f s <= %.3f s (the container\'s)' % (statistics.median(single), a.single_first_s))
    # (e) the control: the head's silence for text prefill pieces beside the same decoders, printed, not judged
    if len(c.T) >= 40000:
        rid = c.text(0x76650000, c.T[:40000], max_new=8, stop=())
        c.wait(rid, 600)
        ln = unit_line(a.log, rid, a.start_re)
        mine = '%.0f ms (median of the clip runs\' row pieces %.0f ms)' % (ln[3], statistics.median(row_sil)) if ln and row_sil else 'no line'
        print('  control, a 40,000-token text prompt beside the same decoders: the longest silence across a row piece %s' % mine)
    else:
        print('  control skipped: the tokens file holds %d ids, 40,000 are needed' % len(c.T))
    c.close()
    return 0


def last_start(text, start_re):
    """the lines of the file's last start (the rank logs hold every start): from the last line matching start_re"""
    at = [m.start() for m in re.finditer(start_re, text, re.M)]
    return text[at[-1]:] if at else text


def cmd_check_log(a):
    texts = [last_start(open(p, errors='replace').read(), a.start_re) for p in a.logs]
    for r, t in enumerate(texts):
        check('BUG' not in t, 'rank %d: no BUG line' % r)
    # every rank prints "store-check: rank R <what> digest X" (rank 0's server's "store digest N" lines are its own): compare those, rank R left out
    dig = [[re.sub(r'^store-check: rank \d ', '', l) for l in t.splitlines() if l.startswith('store-check: rank ')] for t in texts]
    check(all(d == dig[0] for d in dig) and bool(dig[0]), 'the four ranks\' store-check lines are equal line by line (%s)' % ' '.join(str(len(d)) for d in dig))
    if a.expect == 'mid':
        check(any('is between its blocks' in t for t in texts), 'the engine named the image that was between its blocks when a second started')
    if a.expect == 'vote':
        check('the four ranks\' rows differ' in texts[0], 'rank 0 printed the vote\'s sums')
        check(any('vision fault injected at vote' in t for t in texts), 'the fault fired once')
    if a.expect == 'room':
        check(not any('vision fault injected' in t for t in texts), 'no fault was injected: the room was short')
        check('the engine has no image encoder until the next start' in texts[0], 'the engine said it has no image encoder')
    if a.expect == 'alloc':
        check(any('vision fault injected at alloc' in t for t in texts), 'the allocation fault fired')
        check('the engine has no image encoder until the next start' in texts[0], 'the engine said it has no image encoder')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, dump=True):
        p.add_argument('--socket', required=True)
        p.add_argument('--tokens', required=True)
        p.add_argument('--sessions', type=int, default=0)
        p.add_argument('--image', action='append', default=[], help='FILE:HxW')
        if dump:
            p.add_argument('--hosts', nargs=4, default=['local'] * 4)
            p.add_argument('--dump', default='/tmp/vit-dump')
            p.add_argument('--ref', required=True)
    p = sub.add_parser('bytes')
    common(p)
    p = sub.add_parser('ranges')
    common(p)
    p.add_argument('--clip', required=True)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--filler', type=int, default=0, help='tokens of a second, text request beside the clip')
    p.add_argument('--timeout-s', type=float, default=0, help='wait for the clip; 0: 20 s an image (one-block commands)')
    p = sub.add_parser('two')
    common(p)
    p.add_argument('--clip', required=True)
    p.add_argument('--clip2', required=True)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--expect-mid', action='store_true')
    p.add_argument('--hold', action='store_true')
    p.add_argument('--paused', type=int, default=2)
    p.add_argument('--fillers', type=int, default=6)
    p.add_argument('--tokens-out', type=int, default=300)
    p.add_argument('--wait-max-s', type=float, default=3.0)
    p.add_argument('--timeout-s', type=float, default=0, help='wait for a long request; 0: 20 s an image (one-block commands)')
    p = sub.add_parser('cancel')
    common(p)
    p.add_argument('--clip', required=True)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--delays', default='0.05,0.3,1.0,2.0')
    p.add_argument('--vote', action='store_true')
    p = sub.add_parser('resume')
    common(p)
    p.add_argument('--log', help='rank 0\'s log, read for the resume line')
    p.add_argument('--fillers', type=int, default=6)
    p.add_argument('--tokens-out', type=int, default=2000)
    p = sub.add_parser('arena')
    common(p, dump=False)
    p.add_argument('--logs', nargs=4)
    p.add_argument('--fault', choices=['alloc', 'room'])
    p.add_argument('--pool-noflag', type=int, default=0)
    p.add_argument('--save-text')
    p.add_argument('--text-ref')
    p = sub.add_parser('clip')
    common(p, dump=False)
    p.add_argument('--clip', required=True)
    p.add_argument('--single', required=True)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--gap-ms', type=int, default=600)
    p.add_argument('--clip-first-s', type=float, default=14.979)
    p.add_argument('--single-first-s', type=float, default=0.737)
    p.add_argument('--log', required=True, help="rank 0's log of this start (the engine's per-request line is read from it)")
    p.add_argument('--start-re', default=r'^rank \d: vision encoder: ')
    p = sub.add_parser('check-log')
    p.add_argument('--logs', nargs=4, required=True)
    p.add_argument('--expect', choices=['none', 'mid', 'vote', 'alloc', 'room'], default='none')
    p.add_argument('--start-re', default=r'^rank \d: vision encoder: ', help='the line every rank prints once at a start')
    a = ap.parse_args()
    if a.cmd in ('bytes', 'cancel', 'resume', 'arena') and not a.image and not (a.cmd == 'arena' and not a.fault):
        ap.error('--image FILE:HxW is needed')
    rc = {'bytes': cmd_bytes, 'ranges': cmd_ranges, 'two': cmd_two, 'cancel': cmd_cancel, 'resume': cmd_resume, 'arena': cmd_arena,
          'clip': cmd_clip, 'check-log': cmd_check_log}[a.cmd](a)
    print('%s vit_engine_test %s: %d failed' % ('FAIL' if fails else 'PASS', a.cmd, fails))
    return 1 if fails else rc


if __name__ == '__main__':
    sys.exit(main())
