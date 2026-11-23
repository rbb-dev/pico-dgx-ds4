#!/usr/bin/env python3
"""The cache conversation split (--kv-split cache), from outside, on the four Sparks. The engine is started by
the caller (rank 0's socket; dsfront stopped). Every check prints ok / FAIL; exit 0 only when all hold, 2 when the arm did not
reach the path it is there for (the premise is proven from rank 0's log and STATS before anything is judged: a run that did not
reach it proves nothing).

  load     G-K3 (gate 8). Today's read-back beside decoders, the control of the SSD rows below. It runs on a build WITHOUT the split:
           the head's binary, --kv-split absent. A conversation of --size positions is made (keep), pushed to the SSD tier by a
           second conversation of the same size (the pool holds one of them and the decoders, not two), then --sessions requests
           decode; a control window of --window s with the same STATS polling, then the conversation's next turn (it reads the
           whole conversation back, one command); the longest gap between a decoding session's tokens in the control window and in
           the window of the return, the engine's "back from disk" line (MiB, ms, GB/s) and the return turn's time to its first
           token. No pass line: it is the record `off` leaves, and says whether today's one-command read-back already breaks the
           gap setting (E: 0.9 s at 4M). Settings (the caller's; MAXCTX >= --size + 2,048):
             MAXCTX=4200000 engine-test.sh BIN "--slots 12 --big-slots 0 --disk-gib 600 --pool-tokens P" split_test.py --arm load ...
           with P = --size + (--sessions x (4,096 + --decode-new)) + 65,536, so that two conversations of --size do not fit.
             split_test.py --arm load --socket S --tokens T.npy --log ~/dsgb10/logs/engine-split_test.log --size 1000000 [--sessions 8]
             (run again with --size 4000000)
  gap      Item 8's rows (cache build, --kv-split cache), each beside --sessions decoders: --event NAME is one of
             split      a conversation of --size positions finishes (its request ends) and is split by the server's commands
             back       the same conversation returns 2 s later (every ghost taken back)
             exchange   it returns after a filler took the ghosts (the Sparks exchange quarters); --size 131072, 1000000, 4000000
             ssd        it returns from the SSD (its quarters were evicted to the disk tier as images, K4, then read back through the exchange;
                      needs the disk tier: --disk-gib; the premise is STATS conv_on_disk and join_bytes_total rising on the return)
             grow       the decoders keep growing on a pool full of split cache (the step time against the same run with --kv-split off)
           For each: the longest gap between a decoding neighbour's tokens in a window with the event, at most the control window's
           longest + --gap-slack-ms (250: the tenants' line); the control runs the same probe, and the STATS inside the
           run (join_ns_total, join_max_ns, split_pieces_total) are printed. A row over the line: J, the budget or the pace is lowered
           and the row run again; still over at J = 8 goes back to the reader.
             split_test.py --arm gap --event split --socket S --tokens T.npy --log L --size 1000000 [--sessions 8]
           ssd also needs --log (rank 0's engine log) and judges, besides the gap: the return's first token within --ttft-max (30 s), the
           server's read-back line of the conversation `restart point R of R` (R above 0), and no piece crossed the ring twice (the bytes
           exchanged in the return at most the pieces split since the event began x 3 x 729,088); --ssd-fillers N (1) conversations push
           it out of the pool. After the split one premise line says whether the window bookmarks were split too (`slots, lists + bookmarks`;
           the bookmarks from the conversation's own `req` line in --log): --need-split-bookmarks makes fewer slots NOT REACHED. Any log line BUG
           fails. The gap is judged only when a decoder ran through the return's window (--decode-new keeps them alive) and no decoder's line
           was given back in it; else it prints `the gap is not judged` and does not fail, and an empty window never passes.
  tiers    Row L of `--kv-split tiered` (needs --log; the start sets --kv-split-after-s 20 and marks so that A and B of --tier-size fit above
           the high mark, C lands between the marks and the fillers reach under the low mark): A and B idle 25 s split nothing at level 0;
           after C one conversation (A) is demoted, level 1, pressure above 0; C returns with no join, A with one; fillers until level 2: the
           first line is `dropped from L2`, before any `split:`, the dropped conversation reads back whole. Prints the pool and the level
           at each step. `--arm gap` after the DONE of a conversation beside decoders sends one 1-token request without keep when the split
           has not started 2 s later (a finished flight finishes at the run's end or the next request) and prints that it did.
             split_test.py --arm tiers --socket S --tokens T.npy --log L --after-s 20
  tokens   the added row: the decoders' tokens per second while a 1M-position conversation's backlog is split at the 500 ms pace,
           against the same window without it (--kv-split off, or no backlog). Over 10 % lost FAILS: the pace is raised.
             split_test.py --arm tokens --socket S --tokens T.npy --size 1000000 --window 120
  equal    8.5's reference = off: the same turns on two servers (--out FILE per server) compared (--compare A B): tokens, logprobs,
           `cached` and restart point equal, every second turn: return after 2 s, after a filler took the ghosts, 1M positions, a fork
           of a split base, a fork whose base runs, a half-split base, a regenerate inside a shared piece, a request between two
           split commands. The store never short; conversations <= --slots.
             split_test.py --arm equal --socket S --tokens T.npy --out off.json   |  --out cache.json   then   --compare off.json cache.json
  forkrun  the second way in, from outside: a fork in flight must never lose the pieces it shares with its idle base. A base of
           --size positions (default 200,000) is made and left idle until the split goes quiet; then a FORK of it (its first
           --fork-shared positions, default 150,000, plus a new tail of 64) decodes --fork-new tokens (default 400: at least 3 s,
           checked) while the base stays idle and the server's split commands go on. Run with --kv-split cache (premise: STATS
           kv_split 1 and split_pieces_total above 0 after the base's wait) and again with --kv-split off (the twin); --out writes the
           fork's token ids, `--arm forkrun --compare cache.json off.json` compares them (equal: PASS). With --log the engine's
           `joined inside an operation` and `differ from their sums` lines are counted (both must be 0). One connection, as gap.
             split_test.py --arm forkrun --socket S --tokens T.npy --log L --out cache.json   (engine with --kv-split cache)
             split_test.py --arm forkrun --socket S --tokens T.npy --out off.json             (engine without it)
             split_test.py --arm forkrun --compare cache.json off.json
  long     P-scan's row (PS-4; `--pscan ext` against `--pscan off`, one binary): what the decode index scan split over the ranks must
           leave unchanged. For each size of --sizes (default 1200000,2500000: the 2M and 4M windows), for each count of --sessions
           (default 1,4): N conversations of that size with DIFFERENT prompts (offsets), each greedy for --long-new (24) tokens with
           logprobs kept; then every conversation's return turn (its history + 16 new prompt tokens + --long-new tokens), all N
           submitted together, so that they decode in the same steps and read their cached long contexts through the slot table
           (N = 1: no table). One connection (frames of all sessions read as they come). Written per turn to --out: ids, cached, lp.
           PROVES, once --compare A B finds no DIFFER (A from `--pscan off`, B from `--pscan ext`): tokens, logprobs and `cached` equal
           per turn, in the 2M and 4M windows, with one session and with several. The premise is checked here: the return turns of
           N > 1 overlapped (every session's first token before any session's last), else exit 2 (the path was not reached).
           Settings (the engine's): --max-ctx 4194304, the serving pool (--pool-tokens or the default), --slots >= max(--sessions) + 1,
           --pscan ext, then the same binary and settings with --pscan off; both need rank 0's startup line for the split (the
           caller greps it). Expected run time (E from M: prefill 2,946 tok/s at 1M): about 1.75 h an arm: 1 + 4 conversations a size,
           5 x 1.2M + 5 x 2.5M = 18.5M positions, ~6,300 s of prefill; the turns' decode is minutes).
             split_test.py --arm long --socket S --tokens T.npy --sizes 1200000,2500000 --sessions 1,4 --out ext.json   (--pscan ext)
             split_test.py --arm long --socket S --tokens T.npy --sizes 1200000,2500000 --sessions 1,4 --out off.json   (--pscan off)
             split_test.py --arm long --compare off.json ext.json
  imager   the imager's GPU rows A2n, G, B and Q (written, not run: for the first window after the build). --event NAME:
             a2n   (--kv-split max --kv-split-after-s 2, the pool of WR5 A2, --slots 8): the conversation of --size FIRST with no decoder, then the
                   --sessions decoders at once (--decode-new 70000); the log mark is taken after their first token. PREMISE (exit 2): split_rides_total
                   rose and ghost_numbers reached 300 with NO `serve: batch of` line and no pause line since the mark. Then --ssd-fillers conversations
                   push it out and the return: restart point N of N, no BUG, image_waits_total 0, returns_read_total up, ids equal --ref (the
                   reference start's --out: the same two turns on a start that is never short).
             back / exchange / ssd   (--kv-split max): the returns of the corrected design (STATS has no quarter_slots or joins_total any more):
                   back: 2 s later, revives_total up by the pieces, returns_read_total unchanged, no `back from disk` line with bytes; exchange: --filler
                   conversations took the ghosts (premise: ghost_numbers fell), returns_read_total up; ssd: fillers until conv_on_disk >= 1. Ids equal --ref.
             g     the gap: --sessions decoders beside a conversation of --size being imaged and marked for --window s with work pending: the
                   longest gap of a decoding neighbour at most the control window's + --gap-slack-ms (expected +0), and every run's end line
                   `split rides in the run: ... longest X ms` under the control window's median gap (one step; --step-ms is gone).
             b     the bytes: a conversation of --size made and left idle until the images are settled: spill_bytes_total rose by
                   (lists + window bookmarks) x 786,432 + 1,572,864 on rank 0 (the units from split_wanted and the log's window snapshots), STATS
                   image_waits_total 0, and no whole-image line in the log. Run once per mode (--kv-split off, tiered, max); the caller compares
                   with the standing build's recorded run of the same arm (27.0 % of its bytes).
             q     the queue (a small pool, --kv-split-after-s 30 so that idle conversations stay plain): --fillers conversations of --size
                   fill the pool, then a request of --size that needs their blocks: premise printed (image_pending and clean_numbers at its
                   arrival), judged: image_waits_total 0, the decoders' longest gap inside the standing line, its first token printed.
             split_test.py --arm imager --event a2n --socket S --tokens T.npy --log L --size 1000000 --sessions 8 --decode-new 70000
  nolog    no rank 0 log line says a split conversation was joined inside an operation: the server joins by kStJoin first.
             split_test.py --arm nolog --log L
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
from path_proof import Log, NotReached, back_re, need, spill_re  # noqa: E402
from serve_test import CANCEL, DONE, TOKENS, Engine, load_npy_i32  # noqa: E402
from stats_test import NAMES, STATS, STATS_REQ  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def tokens_of(T, n, offset):
    """n prompt tokens from the synthetic text, starting at `offset` (distinct conversations start at distinct places)."""
    out = []
    while len(out) < n:
        k = (offset + len(out)) % len(T)
        out += T[k:k + (n - len(out))]
    return out[:n]


class Decoders:
    """One connection carries the decoding sessions, STATS and the probes' requests, so every frame goes through one handler: the
    sessions' token gaps are recorded whatever else is read (a STATS that dropped them would stretch the gaps it hid). A frame is read
    only when the socket has one (select), never cut by a timeout."""

    def __init__(self, e, T, sessions, new):
        self.e = e
        self.gaps = []     # (time, gap) of every session's token arrivals
        self.tokens = 0    # token arrivals counted (rows of TOKENS frames carry n tokens: the count is the sum of n)
        self.tok_times = []  # (time, n)
        self.last = {}
        self.ended = {}    # decoding session's rid -> the time its request ended
        self.stats = None
        self.watch = {}    # rid -> {'first': t, 'done': t}
        for k in range(sessions):
            rid = e.submit(0x73706400 + k, tokens_of(T, 4096, 5000 * k), max_new=new, keep=False, stop=())[0]
            self.last[rid] = None

    def handle(self, t, p):
        now = time.time()
        if t == DONE and struct.unpack('<Q', p[:8])[0] in self.last:
            self.ended[struct.unpack('<Q', p[:8])[0]] = now  # a decoding session's request ended
        if t == TOKENS:
            r, n = struct.unpack('<QI', p[:12])
            if r in self.last:
                if self.last[r] is not None:
                    self.gaps.append((now, now - self.last[r]))
                self.last[r] = now
                self.tok_times.append((now, n))
            elif r in self.watch:
                w = self.watch[r]
                if w.get('first') is None:
                    w['first'] = now
                w.setdefault('ids', []).extend(struct.unpack('<%dI' % n, p[16:16 + 4 * n]))
        elif t == DONE:
            r = struct.unpack('<Q', p[:8])[0]
            if r in self.watch:
                self.watch[r]['done'] = now
                self.watch[r]['frame'] = p
        elif t == STATS:
            ver, n = struct.unpack('<II', p[:8])
            v = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])
            self.stats = dict(zip(NAMES, v))

    stall_guard = None   # set by the imager rows once the decoders stream: (seconds, log) -> DecodersStopped when one that has not ended gave no token that long

    guard_mark = 0       # the log position when the guard was armed: a `gives its N chunks back` line after it excuses that decoder
    excused = ()

    def _check_stall(self):
        if self.stall_guard is None:
            return
        secs, log = self.stall_guard
        now = time.time()
        live = [r for r, t in self.last.items() if r not in self.ended and t is not None]
        stale = [r for r in live if now - self.last[r] > secs]
        if not stale:
            return
        tail = log.text(max(0, log.mark() - 600)).strip().splitlines()[-1:] if log else []
        if len(stale) == len(live):  # every decoder silent: the engine is not stepping at all, given back or not
            raise DecodersStopped('the decoders stopped: every decoder gave no token for %.0f s (last journal line: %s); the window takes the capture before its stop' % (secs, tail))
        text = log.text(self.guard_mark) if log else ''
        for r in stale:
            k = list(self.last).index(r)
            if gave_back_lines(text, {0x73706400 + k}):  # its line went to a request that needs the pool: it waits by design, not a stop
                if r not in self.excused:
                    self.excused = tuple(self.excused) + (r,)
                    print('  decoder %d waits for the pool (its line was given back): not a stop' % k)
                continue
            raise DecodersStopped('the decoders stopped: request %d gave no token for %.0f s and its line was not given back (last journal line: %s); the window takes the capture before its stop' % (r, now - self.last[r], tail))

    def _buffered(self):
        """A whole frame is already in the engine client's buffer (a read of the socket takes everything that has arrived, so select would not see it)."""
        b = self.e.buf
        return len(b) >= 8 and len(b) >= 8 + struct.unpack('<I', b[:4])[0]

    def add_decoder(self, T, k, new):
        """A decoder like the others (session 0x73706400 + k, 4,096 positions, --decode-new), added after the start; returns its request id."""
        rid = self.e.submit(0x73706400 + k, tokens_of(T, 4096, 5000 * k), max_new=new, keep=False, stop=())[0]
        self.last[rid] = None
        return rid

    def pump(self, until, stop=lambda: False):
        while time.time() < until and not stop():
            self._check_stall()
            if self._buffered() or select.select([self.e.s], [], [], max(0.0, min(0.2, until - time.time())))[0]:
                self.handle(*self.e.frame())

    def poll_stats(self):
        self.stats = None
        self.e.send(STATS_REQ, struct.pack('<II', 0, 0))
        self.pump(time.time() + 10, lambda: self.stats is not None)
        return self.stats

    def window(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.pump(min(end, time.time() + 1.0))
            self.poll_stats()

    def longest(self, t0, t1):
        # a gap is stamped at its end: it counts only when it BEGAN inside the window (ts - g >= t0), else a silence that began in the previous piece of a long prompt is counted
        return max((g for ts, g in self.gaps if ts - g >= t0 and ts <= t1), default=0.0)

    def tok_per_s(self, t0, t1):
        return sum(n for ts, n in self.tok_times if t0 <= ts <= t1) / max(1e-9, t1 - t0)

    def submit_watch(self, session, prompt, max_new, keep=True):
        rid = self.e.submit(session, prompt, max_new=max_new, keep=keep, stop=())[0]
        self.watch[rid] = {'t0': time.time()}
        return rid

    def finish(self):
        for r in self.last:
            self.e.send(CANCEL, struct.pack('<Q', r))


class DecodersStopped(Exception):
    pass


def make_conversation(e, T, session, size, offset):
    """A kept conversation of `size` positions: one request that prefills the prompt and stops after one token."""
    o = e.run(session, tokens_of(T, size, offset), max_new=1, keep=True, stop=())
    check(o['reason'] in ('stop', 'length', 'max_new') or o['error'] == '', 'conversation %016x made: %d prompt positions, error %r' % (session, o['prompt'], o['error']))
    return o


def cmd_load(a):
    T = list(load_npy_i32(a.tokens))
    log = Log(a.log)
    e = Engine(a.socket)
    sid = 0x73706c00 + a.size // 1000
    # the conversation, then a second one of the same size pushes it to the disk tier (the pool holds one of them)
    o1 = make_conversation(e, T, sid, a.size, 0)
    hist = list(tokens_of(T, a.size, 0)) + list(o1['ids'])
    make_conversation(e, T, sid + 1, a.size, 13579)
    m = log.wait_for(spill_re(sid), 'the conversation pushed to the disk tier', seconds=120)
    need(m is not None, 'serve: spill session %016x' % sid)
    d = Decoders(e, T, a.sessions, a.decode_new)
    d.window(20)  # the sessions settle
    s0 = d.poll_stats()
    need(s0 is not None and s0['conv_on_disk'] >= 1, 'STATS: %d conversation(s) only on the disk tier' % (s0['conv_on_disk'] if s0 else -1))
    t = time.time()
    d.window(a.window)
    ctrl = d.longest(t, time.time())
    print('control window %.0f s, %d sessions: longest gap between a session\'s tokens %.0f ms' % (a.window, a.sessions, ctrl * 1e3))
    ret = hist + tokens_of(T, 16, 777)
    t0 = time.time()
    rid = d.submit_watch(sid, ret, 1)
    d.pump(time.time() + 3600, lambda: 'done' in d.watch[rid])
    d.pump(time.time() + 5)
    t1 = time.time()
    need('done' in d.watch[rid], 'the return turn finished')
    mb = log.wait_for(back_re(sid), 'the read-back line', seconds=15)
    mib, whole, pos, rp, rp0 = float(mb.group(1)), float(mb.group(2)), int(mb.group(3)), int(mb.group(4)), int(mb.group(5))
    ms = re.search(r'in ([\d.]+) ms', mb.group(0))
    ms = float(ms.group(1)) if ms else 0.0
    ev = d.longest(t0, t1)
    print('the return: %d positions, back from the disk %.0f MiB in %.0f ms (%.2f GB/s), restart point %d of %d, TTFT %.2f s' % (
        pos, mib, ms, mib * 1.048576 / max(ms, 1e-9), rp, rp0, (d.watch[rid]['first'] or t1) - t0))
    print('the window of the return (%.1f s): longest gap %.0f ms, the control\'s %.0f ms, difference %+.0f ms (no pass line: the record `off` leaves)' % (
        t1 - t0, ev * 1e3, ctrl * 1e3, (ev - ctrl) * 1e3))
    check(rp == rp0, 'nothing of the conversation was lost on the way (restart point %d of %d)' % (rp, rp0))
    d.finish()
    return 0


def cmd_nolog(a):
    text = open(a.log, errors='replace').read()
    bad = re.findall(r'state: session \d+: joined inside an operation', text)  # the guard's line: whenever it joins anything, cut or not
    check(not bad, 'no "joined inside an operation" line (%d found)' % len(bad))
    return 0


def runs_ended_for_split(text):
    """The decoding runs a split command stood between, in a journal or log sample: each group of `split:` lines between two
    `batch of` lines is one run ended and begun again for it (a run that nothing ended has none: `off`'s one long batch)."""
    prev_batch, pending, n = False, 0, 0
    for ln in text.splitlines():
        if 'serve: batch of ' in ln:
            if prev_batch and pending:
                n += 1
            prev_batch, pending = True, 0
        elif ' split: ' in ln and 'pieces' in ln:
            pending += 1
    return n


def kick_split(d, T):
    """A finished flight beside decoders that do not stop finishes (and its conversation is listed) at the run's end or the next
    request: one 1-token request without keep is that next request."""
    rid = d.submit_watch(0x73706ff0, tokens_of(T, 16, 31337), 1, keep=False)
    d.pump(time.time() + 120, lambda: 'done' in d.watch[rid])
    print('  the split had not started 2 s after the DONE: one 1-token request without keep was sent (a finished flight\'s finish() waits for a run\'s end or the next request)')


def wait_split_quiet(d, seconds=120, rise_from=None, kick=None):
    """STATS until the split has gone quiet: the pieces split stop rising for 3 s (the server's commands are 500 ms apart).
    With rise_from (the count before the event): first until the count rises above it, so a split the server deferred (it
    tries again 5 s later when no number could be opened) is waited for, not taken for quiet."""
    t_end = time.time() + seconds
    t_beg, kicked = time.time(), False
    while rise_from is not None and time.time() < t_end:
        d.pump(time.time() + 1.0)
        s = d.poll_stats()
        if s is not None and s['split_pieces_total'] > rise_from:
            break
        if kick is not None and not kicked and time.time() - t_beg >= 2.0:  # no rise 2 s after the DONE
            kicked = True
            kick()
    last, since = None, time.time()
    while time.time() < t_end:
        d.pump(time.time() + 1.0)
        s = d.poll_stats()
        if s is None:
            continue
        if last != s['split_pieces_total']:
            last, since = s['split_pieces_total'], time.time()
        elif time.time() - since >= 3:
            return s
    raise NotReached('the split did not go quiet in %d s' % seconds)


def need_cache(s):
    need(s is not None and s['kv_split'] == 1, 'STATS: kv_split 1 (the engine was started with --kv-split cache)')


QUARTERS_OF_A_PIECE = 3 * 729088  # the other three ranks' quarters of one piece, in bytes (a piece that crossed the ring)
GAVE_BACK = re.compile(r'serve: slot \d+ \(session ([0-9a-f]{16})\) gives its \d+ chunks back to the pool')


def decoder_give_backs(text, sessions):
    """Lines of `text` that say a DECODER's line (session 0x73706400 + k) was given back; fillers' give-backs do not count."""
    return gave_back_lines(text, {0x73706400 + k for k in range(sessions)})


def gave_back_lines(text, sids):
    """How many lines of the log say a line of one of `sids` was given back to the pool."""
    return sum(1 for m in GAVE_BACK.finditer(text) if int(m.group(1), 16) in sids)


def ssd_return_checks(ttft, ttft_max, back, exchanged, split_pieces):
    """The three checks of a return from the SSD, (ok, text) each: the first token in time; the server's read-back line of the
    conversation names a restart point and it is the whole one (a conversation computed cold says `-1 of -1` or no line); no piece
    crossed the ring twice (the bytes exchanged at most the quarters of the pieces split since the event began). `back` is
    (restart point, restart point it had) or None."""
    out = [(ttft <= ttft_max, 'TTFT %.1f s (at most %.0f s)' % (ttft, ttft_max))]
    if back is None:
        out.append((False, 'the server\'s read-back line of the conversation: none in the log'))
    else:
        out.append((back[0] == back[1] and back[0] > 0, 'the server\'s read-back line: restart point %d of %d' % back))
    cap = split_pieces * QUARTERS_OF_A_PIECE
    out.append((exchanged <= cap, 'no piece crossed the ring twice: %d MiB exchanged, at most %d MiB (%d pieces split)' % (
        exchanged >> 20, cap >> 20, split_pieces)))
    return out


def split_wanted(n):
    """The list pieces of a conversation of n prompt positions that the split quarters: 3 lists of 16,384-position chunks and one of 8,192."""
    return 3 * -(-n // 16384) + -(-n // 8192)


def window_snapshots(text, sid, prompt):
    """The window snapshots of the conversation's own `req` line (its first request, the prompt of `prompt` positions); None: no such line."""
    m = re.search(r'serve: req \d+ session %016x: prompt %d, .*?(\d+) window snapshots' % (sid, prompt), text)
    return int(m.group(1)) if m else None


def bookmarks_premise(slots, n, bookmarks):
    """(ok, text): the quarter slots hold the conversation's lists and its window bookmarks (the end bookmark is not counted here: it
    is a further three), so a split bookmark's eviction is in the row. bookmarks None: the req line is not in the log."""
    if bookmarks is None:
        return False, 'the window bookmarks are split too: the conversation\'s req line is not in the log (no count of its window snapshots)'
    lists = split_wanted(n)
    return slots >= lists + bookmarks, 'the window bookmarks are split too: %d slots, %d lists + %d bookmarks' % (slots, lists, bookmarks)


def bug_lines(text):
    """Lines of the log that say BUG, of any kind."""
    return sum(1 for ln in text.splitlines() if 'BUG' in ln)


def gap_verdict(g, ctrl, slack_ms, given_back, ended=0, total=0, gaps=1):
    """(judged, ok, text): the longest gap against the control's + slack, judged only when a decoder ran through the return's window
    (`ended` of `total` had ended before it, `gaps` token gaps were recorded in it: none, and the longest gap of an empty window is
    0 ms, which proves nothing) and no decoder's line was given back in it (the pool then did not hold the return beside the
    decoders and the gap measures the pool)."""
    if total and ended >= total:
        return False, True, 'the gap is not judged: %d of %d decoders had ended before the return' % (ended, total)
    if gaps == 0:
        return False, True, 'the gap is not judged: no decoder token in the window (%d of %d decoders had ended before the return)' % (ended, total)
    if given_back:
        return False, True, 'the gap is not judged: %d decoder lines given back (the pool does not hold the return beside the decoders): longest gap %.0f ms' % (
            given_back, g * 1e3)
    return True, g <= ctrl + slack_ms / 1000.0, 'the longest gap %.0f ms, the control\'s %.0f ms + %.0f ms' % (g * 1e3, ctrl * 1e3, slack_ms)


def cmd_gap(a):
    """Item 8's rows. One event per run, beside --sessions decoders; the control window is the same length with the same STATS polling."""
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    d = Decoders(e, T, a.sessions, a.decode_new)
    log = Log(a.log) if a.event == 'ssd' and a.log else None
    if a.event == 'ssd':
        need(log is not None, 'the ssd row reads the engine log: --log')
    given_back = 0
    d.window(20)
    s0 = d.poll_stats()
    if a.event != 'grow' or a.cache:
        need_cache(s0)
    sid = 0x73706c00 + a.size // 1000
    t = time.time()
    d.window(a.window)
    ctrl = d.longest(t, time.time())
    print('control window %.0f s: longest gap %.0f ms' % (a.window, ctrl * 1e3))
    ev0, ev1 = 0.0, 0.0
    if a.event == 'grow':
        # the decoders keep growing on a pool full of split cache: --fill conversations of --size, split, then the window again
        for k in range(a.fill):
            d.submit_watch(sid + 1 + k, tokens_of(T, a.size, 1000 * (k + 1)), 1)
            d.pump(time.time() + 3600, lambda: all('done' in v for v in d.watch.values()))
        s1 = wait_split_quiet(d) if a.cache else d.poll_stats()
        if a.cache:
            need(s1['quarter_slots'] > 0, 'the pool holds %d quarter slots (split cache)' % s1['quarter_slots'])
        t = time.time()
        d.window(a.window)
        ev0, ev1 = t, time.time()
        print('step time: %.1f tok/s over the window beside %d filled conversations (compare with the --kv-split off run)' % (d.tok_per_s(ev0, ev1), a.fill))
    else:
        s1 = d.poll_stats()
        t_ev = time.time()
        if a.event == 'split':
            rid = d.submit_watch(sid, tokens_of(T, a.size, 0), 1)
            d.pump(time.time() + 3600, lambda: 'done' in d.watch[rid])
            s2 = wait_split_quiet(d, rise_from=s1['split_pieces_total'], kick=(lambda: kick_split(d, T)) if a.sessions > 0 else None)
            need(s2['split_pieces_total'] > s1['split_pieces_total'], 'the finished conversation was split: %d pieces' % (s2['split_pieces_total'] - s1['split_pieces_total']))
            # the event is the split, from the conversation's end; its prefill beside the decoders is the interleaver's (no pass line here)
            print('its prefill beside the decoders (%.1f s): longest gap %.0f ms (the interleaver\'s, not the split\'s)' % (
                d.watch[rid]['done'] - t_ev, d.longest(t_ev, d.watch[rid]['done']) * 1e3))
            ev0, ev1 = d.watch[rid]['done'], time.time()
        else:
            # on the decoders' connection: every frame of theirs keeps coming while this request runs
            p0 = tokens_of(T, a.size, 0)
            r0 = d.submit_watch(sid, p0, 1)
            d.pump(time.time() + 3600, lambda: 'done' in d.watch[r0])
            hist = p0 + list(d.watch[r0].get('ids', []))
            need(len(hist) == len(p0) + 1, 'the conversation was made: %d prompt ids + %d new' % (len(p0), len(hist) - len(p0)))
            s2 = wait_split_quiet(d, rise_from=s1['split_pieces_total'], kick=(lambda: kick_split(d, T)) if a.sessions > 0 else None)
            print('pieces split and deferred since the event began: %d, %d' % (
                s2['split_pieces_total'] - s1['split_pieces_total'], s2['split_deferred_total'] - s1['split_deferred_total']))
            need(s2['quarter_slots'] > 0, 'the conversation is split: %d slots' % s2['quarter_slots'])
            if a.event == 'ssd':
                ok, what = bookmarks_premise(s2['quarter_slots'], len(p0), window_snapshots(log.text(0), sid, len(p0)))
                if a.need_split_bookmarks:
                    need(ok, what)
                else:
                    print('  %s: %s' % ('ok' if ok else 'note', what if ok else what + ': a split bookmark\'s eviction is not in this row'))
            if a.event == 'back':
                d.pump(time.time() + 2)  # 2 s later: every ghost is still there
            elif a.event == 'exchange':
                f = d.submit_watch(sid + 1, tokens_of(T, a.filler, 7777), 1)  # a filler takes the ghosts
                d.pump(time.time() + 3600, lambda: 'done' in d.watch[f])
                wait_split_quiet(d)
            elif a.event == 'ssd':
                # conversations of the same size: at least --ssd-fillers, then more (up to --ssd-fillers-max) while no conversation
                # is only on the disk tier yet (split conversations hold a quarter each, so a pool that holds the return needs several)
                k = 0
                while k < max(a.ssd_fillers, a.ssd_fillers_max):
                    if k >= a.ssd_fillers:
                        sk = d.poll_stats()
                        if sk is not None and sk['conv_on_disk'] >= 1:
                            break
                    f = d.submit_watch(sid + 1 + k, tokens_of(T, a.size, 4242 + 1000 * k), 1)
                    d.pump(time.time() + 3600, lambda f=f: 'done' in d.watch[f])
                    wait_split_quiet(d)
                    k += 1
                print('fillers of %d ids: %d' % (a.size, k))
            s3 = d.poll_stats()
            if a.event == 'ssd':
                need(s3['conv_on_disk'] >= 1, 'the split conversation is only on the disk tier (its quarters evicted as images): %d on the disk' % s3['conv_on_disk'])
            t0 = time.time()
            mark = log.mark() if log else None
            rid = d.submit_watch(sid, hist + tokens_of(T, 16, 777), 1)
            d.pump(time.time() + 3600, lambda: 'done' in d.watch[rid])
            d.pump(time.time() + 5)
            s4 = d.poll_stats()
            ev0, ev1 = t0, time.time()
            need(a.event == 'ssd' or s4['joins_total'] > s3['joins_total'], 'the return was a join: %d commands, %d pieces taken back, %d MiB from the other Sparks' % (
                s4['joins_total'] - s3['joins_total'], s4['revives_total'] - s3['revives_total'], (s4['join_bytes_total'] - s3['join_bytes_total']) >> 20))
            if a.event == 'back':
                need(s4['join_bytes_total'] == s3['join_bytes_total'], 'nothing crossed the ring: every ghost was taken back')
            if a.event in ('exchange', 'ssd'):
                need(s4['join_bytes_total'] > s3['join_bytes_total'], 'the Sparks exchanged quarters')
            ttft = (d.watch[rid]['first'] or ev1) - t0
            print('join_max_ns %.1f ms, join_ns_total %.1f ms, TTFT %.2f s' % (s4['join_max_ns'] / 1e6, s4['join_ns_total'] / 1e6, ttft))
            if a.event == 'ssd':
                text = log.text(mark)
                mb = list(re.finditer(back_re(sid), text, re.M))
                back = (int(mb[-1].group(4)), int(mb[-1].group(5))) if mb else None
                for ok, what in ssd_return_checks(ttft, a.ttft_max, back, s4['join_bytes_total'] - s3['join_bytes_total'],
                                                  s3['split_pieces_total'] - s1['split_pieces_total']):
                    check(ok, what)
                given_back = gave_back_lines(text, {0x73706400 + k for k in range(a.sessions)})
                check(bug_lines(log.text(0)) == 0, 'lines BUG of any kind in the log: %d' % bug_lines(log.text(0)))
    g = d.longest(ev0, ev1)
    judged, ok, what = gap_verdict(g, ctrl, a.gap_slack_ms, given_back, sum(1 for t in d.ended.values() if t < ev0), len(d.last),
                                   sum(1 for ts, _ in d.gaps if ev0 <= ts <= ev1))
    if judged:
        check(ok, '%s (%.1f s): %s' % (a.event, ev1 - ev0, what))
    else:
        print('  %s (%.1f s): %s' % (a.event, ev1 - ev0, what))
    d.finish()
    return 0


STORE_BACK = re.compile(r'^state: session \d+ back from disk: ([\d.]+) MiB in [\d.]+ ms \(.*?(\d+) pieces checked', re.M)


def store_reads(text):
    """(MiB read, pieces checked) of each STORE line `state: session N back from disk: ...`: what the return really read. The server's own summary line
    (`serve: session ... back from disk: X MiB of Y MiB ...`, back_re) counts the conversation brought back, a revive included, and says the restart point: it is not bytes read."""
    return [(float(m.group(1)), int(m.group(2))) for m in STORE_BACK.finditer(text)]


BATCH_LINE = re.compile(r'^serve: batch of \d+: ', re.M)
RIDE_END = re.compile(r'serve: split rides in the run: (\d+) rides, (\d+) imaged, (\d+) marked, longest ([0-9.]+) ms')
IMAGE_BYTES, END_IMAGE_BYTES = 786432, 1572864


def imager_expected_bytes(size, window_snaps, end=False):
    """What rank 0 writes for a conversation of `size` positions once it is imaged: its lists' pieces and window bookmarks at 786,432 B each, and, only
    when `end`, the end bookmark's 1,572,864 B. The end bookmark is taken only when a line lets its conversation go into the store (serve.cpp persist_slot ->
    persist_ops -> kStSnapEnd -> end_take); a conversation still on its line has none."""
    return (split_wanted(size) + window_snaps) * IMAGE_BYTES + (END_IMAGE_BYTES if end else 0)


def imager_turn(e, T, sid, hist, tag, d=None):
    """The conversation's return turn: its history + 16 new tokens, greedy 24 tokens (ids kept for the equality with the reference). With `d` (the
    Decoders object that owns the connection) the turn goes through its demultiplexer: every frame is routed to its request id, never assumed in order, so the
    decoders streaming on the same connection cannot be taken for the turn's response (an AssertionError on frame ids 9 and 10)."""
    if d is not None:
        o = demux_turn(d, sid, hist + tokens_of(T, 16, 777))
        print('  %s: first token %.2f s, %d ids, cached %d' % (tag, o['t_first'] or -1, len(o['ids']), o['cached']))
        return o
    o = e.run(sid, hist + tokens_of(T, 16, 777), max_new=24, n_top=2, keep=True, stop=())
    print('  %s: first token %.2f s, %d ids, cached %d' % (tag, o['t_first'] or -1, len(o['ids']), o['cached']))
    return o


def demux_turn(d, sid, prompt, new=24):
    """One request on the shared connection through Decoders.handle (it routes TOKENS and DONE frames by request id)."""
    t0 = time.time()
    rid = d.submit_watch(sid, prompt, new, keep=True)
    d.pump(time.time() + 3600, lambda: 'done' in d.watch[rid])
    w = d.watch[rid]
    r, reason, pt, cached, comp = struct.unpack('<QIIII', w['frame'][:24])
    return {'ids': list(w.get('ids', [])), 'cached': cached, 'prompt': pt, 'completion': comp, 't_first': (w['first'] - t0) if w.get('first') else None}


def exchange_fillers(free, reserved, ghosts, per_filler):
    """Fillers the exchange row must send before one has to take a ghost: the engine uses plain free numbers first (a take is the lowest plain free number,
    then the oldest ghost), so the plain free numbers (free less reserved less ghosts) are spent first. One more filler takes the ghosts."""
    plain = max(0, free - reserved - ghosts)
    return plain // max(1, per_filler) + 1


def imager_equal(a, tag, o):
    """--out writes the return's ids; --ref FILE (the reference start's) must hold the same ids."""
    if a.out:
        json.dump({tag: {'ids': o['ids'], 'cached': o['cached']}}, open(a.out, 'w'))
    if a.ref:
        r = json.load(open(a.ref)).get(tag)
        need(r is not None, 'the reference file holds %s' % tag)
        check(r['ids'] == o['ids'], '%s: the ids equal the reference\'s (%d ids)' % (tag, len(o['ids'])))


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0.0


def step_gaps(gaps):
    """The gaps of at least 5 ms: the server sends one TOKENS frame per token (serve.cpp:1432-1446), so the frames of one step arrive together and their gaps are ~0; the
    smallest step is 18.9 ms (M 5.15 A)."""
    return [g for g in gaps if g >= 0.005]


def step_median(gaps):
    return median(step_gaps(gaps))


def imaged_numbers(s):
    """Numbers whose quarter image is on the SSD: clean (held) plus ghosts (marked). A conversation that keeps its line is imaged and never marked."""
    return s['clean_numbers'] + s['ghost_numbers']


BATCH_FULL = re.compile(r'^serve: batch of (\d+): (\d+) steps in ([\d.]+) ms, ([\d.]+) ms a step \(median\); req, tokens, end: (.*?); drafts: ([\d.]+) tokens', re.M)
MISS_LINE = re.compile(r'^serve: req \d+: a run was ended for its start and it waits again', re.M)


EXT_BUG = re.compile(r'serve: BUG: every flight waits for an extension')
STALL_NOTHING = re.compile(r'serve: every request in flight waits for the pool [^\n]*nothing can be imaged')
GROWS_LINE = re.compile(r'serve: req \d+ grows: its reservation reaches')


def journal_counts(text):
    """(BUG: every flight waits for an extension, `every request in flight waits for the pool ... nothing can be imaged`, `grows: its reservation reaches`) over a row's span.
    The first two must be 0; the third is the premise that the row stays outside the standstill rule (not 0 is printed for the reader, not a failure)."""
    return len(EXT_BUG.findall(text)), len(STALL_NOTHING.findall(text)), len(GROWS_LINE.findall(text))


def row_counts(text, label):
    """One helper for every row: print the three counts, require the first two 0."""
    b, n, g = journal_counts(text)
    print('  %s: journal counts: BUG every flight waits for an extension %d, `waits for the pool ... nothing can be imaged` %d, grows %d (premise: outside the standstill rule%s)' % (
        label, b, n, g, '' if not g else '; NOT 0: for the reader'))
    check(b == 0, '%s: 0 lines `BUG: every flight waits for an extension` (%d)' % (label, b))
    check(n == 0, '%s: 0 lines `every request in flight waits for the pool ... nothing can be imaged` (%d)' % (label, n))
    return b, n, g


def runs_between(text, wait_mark, first_move):
    """`serve: batch of` lines between the first match of `wait_mark` (a regex: the request's wait line) and the first match of `first_move` after it (a regex: its first move: a
    `back from disk` or `joined` line of its session, or its `serve: req N factor` line). -1 when the wait line is absent; the whole rest of the text when no move follows."""
    w = re.search(wait_mark, text, re.M)
    if w is None:
        return -1
    m = re.compile(first_move, re.M).search(text, w.end())
    return len(re.findall(r'serve: batch of \d+: ', text[w.end():m.start() if m else len(text)]))  # unanchored: a journal's lines carry a time and a process prefix


def batch_info(text):
    """Each `serve: batch of N` line as (steps, ms a step, tokens a step, every end `paused`)."""
    out = []
    for m in BATCH_FULL.finditer(text):
        ends = re.findall(r'\d+ \d+ (\w+)', m.group(5))
        out.append((int(m.group(2)), float(m.group(4)), float(m.group(6)), bool(ends) and all(x == 'paused' for x in ends)))
    return out


def decoder_give_backs_before_start(text, req, sessions):
    """Decoders' give-back lines between the previous request's end (its `serve: req M pool:` line) and request `req`'s `serve: req N factor` line (its start): a request's arrival is not
    logged, its first effect is the give-backs that precede its wait line."""
    st = re.search(r'^serve: req %d factor ' % req, text, re.M)
    end = st.start() if st else len(text)
    pre = text[:end]
    pools = list(re.finditer(r'^serve: req \d+ pool:', pre, re.M))
    seg = pre[pools[-1].end():] if pools else pre
    return decoder_give_backs(seg, sessions)


def giveback_violations(text, req, sessions):
    """Give-back lines naming a decoder's session (a request in flight) between request `req`'s arrival and start, other than the one that its `factor` line follows with no wait line between."""
    st = re.search(r'^serve: req %d factor ' % req, text, re.M)
    pre = text[:st.start()] if st else text
    pools = list(re.finditer(r'^serve: req \d+ pool:', pre, re.M))
    off = pools[-1].end() if pools else 0
    bad = 0
    ids = {0x73706400 + k for k in range(sessions)}
    for m in GAVE_BACK.finditer(pre, off):
        if int(m.group(1), 16) not in ids:
            continue
        nxt = re.compile(r'^serve: req %d (waits for a slot|waits at its start|factor )' % req, re.M).search(text, m.end())
        if nxt is None or not nxt.group(1).startswith('factor'):
            bad += 1
    return bad


def decoders_wait_at_start(text, sessions):
    """Decoders' requests (ids 1..sessions, as Decoders submits them first) that wait at their start."""
    return [int(x) for x in re.findall(r'^serve: req (\d+) waits at its start', text, re.M) if int(x) <= sessions]


STEP_LINE = re.compile(r'^serve: batch of \d+: \d+ steps in [\d.]+ ms, ([\d.]+) ms a step \(median\)', re.M)


DRAFTS_LINE = re.compile(r'^serve: req (\d+) drafts: (\d+) verify steps', re.M)
TOKENS_LINE = re.compile(r'^serve: req (\d+) session [0-9a-f]+: .*? (\d+) tokens in ([\d.]+) ms', re.M)


def engine_step(text):
    """(ms a step, source): the run's own median from its `serve: batch of N` lines; with none (one decoder prints none) the decoder's own `tokens in X ms` divided by its
    `N verify steps`; (0.0, 'none') when neither exists."""
    b = [float(x) for x in STEP_LINE.findall(text)]
    if b:
        return median(b), 'batch of N lines'
    dr = {m.group(1): int(m.group(2)) for m in DRAFTS_LINE.finditer(text)}
    for m in TOKENS_LINE.finditer(text):
        if dr.get(m.group(1)):
            return float(m.group(3)) / dr[m.group(1)], 'req %s: tokens in %s ms / %d verify steps' % (m.group(1), m.group(3), dr[m.group(1)])
    return 0.0, 'none'


def engine_step_ms(text):
    return engine_step(text)[0]


def steps_agree(test_ms, engine_ms, tol=0.20):
    return engine_ms > 0 and abs(test_ms - engine_ms) <= tol * engine_ms


RIDE_LINE = re.compile(r'^serve: split rides in the run: (\d+) rides, (\d+) imaged, (\d+) marked, longest ([0-9.]+) ms', re.M)
SETTLED_LINE = re.compile(r'^serve: split: settled:', re.M)
STALL_LINE = re.compile(r'^serve: req \d+ waits at its start for the pool and nothing in flight can step .*?: (the imager makes room|nothing can be imaged)|'
                        r'^serve: every request in flight waits for the pool .*?: (the imager makes room|nothing can be imaged)', re.M)


def carrier(text):
    """(pieces imaged by rides, idle `split: settled:` lines) in `text` (the log between a request's first wait line and its start): what made the room."""
    return sum(int(m.group(2)) for m in RIDE_LINE.finditer(text)), len(SETTLED_LINE.findall(text))


def stall_lines(text):
    """(lines ending `the imager makes room`, lines ending `nothing can be imaged`) (the stall lines of a rotation that cannot move and of a prompt that waits at its start)."""
    ms = [m.group(1) or m.group(2) for m in STALL_LINE.finditer(text)]
    return sum(1 for x in ms if x.startswith('the imager')), sum(1 for x in ms if x.startswith('nothing'))


REQ_START = re.compile(r'^serve: req (\d+) factor ', re.M)


def rides_phase(a, d, log, mark, s1):
    """Rows a2n and g: the event window from the mark until ghost_numbers is at least 300 and image_pending 0 (at most --window s), the premises (rides rose; ghosts;
    no `serve: batch of`, no pause line and no request start line since the mark: the rides rode one run and no request was read), then the control window of --window s in the
    same run (premise: image_pending 0 before it and no ride across it). Judged: the event window's longest gap (a gap counts when it began inside the window) at most the control's
    longest plus its median (no step was lost: a run that is ended costs a whole step). Returns the control's longest and median (s) and the event window's end."""
    t_ev0 = time.time()
    t_end = t_ev0 + a.window
    s = s1
    marks = imaged_numbers if a.event == 'g' else (lambda x: x['ghost_numbers'])  # a2n's question is marks made inside a run; g's is the rides' cost (a lone decoder keeps its line: imaged, never marked)
    while time.time() < t_end and (marks(s) < 300 or s['image_pending'] > 0):
        d.pump(time.time() + 1.0)
        s = d.poll_stats()
    t_ev1 = time.time()
    text = log.text(mark)
    batches, pauses, starts = len(BATCH_LINE.findall(text)), len(re.findall(r'pause', text, re.I)), len(REQ_START.findall(text))
    need(s['split_rides_total'] > s1['split_rides_total'], 'premise: rides rose (%d)' % (s['split_rides_total'] - s1['split_rides_total']))
    need(marks(s) >= 300, 'premise: %s reached %d (300 wanted)' % ('imaged_numbers (clean + ghosts)' if a.event == 'g' else 'ghost_numbers', marks(s)))
    need(batches == 0 and pauses == 0 and starts == 0, 'premise: no `serve: batch of` line, no pause line and no request start line since the mark (%d, %d, %d): the marks rode one run' % (batches, pauses, starts))
    s_pre = d.poll_stats()
    t0 = time.time()
    d.window(a.window)
    t1 = time.time()
    s_post = d.poll_stats()
    gaps = [g for ts, g in d.gaps if ts - g >= t0 and ts <= t1]
    ctrl, med = max(gaps, default=0.0), step_median(gaps)
    ctrl_tok = d.tok_per_s(t0, t1)
    print('control window %.0f s: longest gap %.0f ms, step %.1f ms (%d step gaps of %d gaps); image_pending %d before, rides %d -> %d' % (
        a.window, ctrl * 1e3, med * 1e3, len(step_gaps(gaps)), len(gaps), s_pre['image_pending'], s_pre['split_rides_total'], s_post['split_rides_total']))
    need(s_pre['image_pending'] == 0 and s_post['split_rides_total'] == s_pre['split_rides_total'], 'the control window had nothing to image: image_pending 0 and no ride during it')
    ev = d.longest(t_ev0, t_ev1)
    check(ev <= ctrl + med, 'event window (%.1f s): longest gap %.0f ms at most the control\'s %.0f ms + its median %.1f ms (no step lost)' % (t_ev1 - t_ev0, ev * 1e3, ctrl * 1e3, med * 1e3))
    return {'ctrl': ctrl, 'med': med, 'ev0': t1, 'tok_s': ctrl_tok}


STAMP = re.compile(r'^(?:\w{3} +\d+ )?(\d\d):(\d\d):(\d\d\.\d+)\b', re.M)


def line_stamp(text, pattern):
    """Seconds of the day of the first line matching `pattern` (a regex over the line), from the journal's own stamp (`-o short-precise`); None when the lines carry none."""
    m = re.search(r'^(.*?(?:%s).*)$' % pattern, text, re.M)
    if not m:
        return None
    t = STAMP.search(m.group(1))
    return int(t.group(1)) * 3600 + int(t.group(2)) * 60 + float(t.group(3)) if t else None


def journal_text(since_epoch, runner=None):
    """The engine's journal since `since_epoch` with stamps (`journalctl --user -u pico-engine -o short-precise`), read where the test runs (beside the engine); '' when it cannot be read."""
    since = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since_epoch))
    cmd = ['journalctl', '--user', '-u', 'pico-engine', '--since', since, '-o', 'short-precise', '--no-pager']
    try:
        if runner:
            return runner(cmd)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ''


def stamp_of(text, pattern, since_epoch, runner=None):
    """line_stamp over the text the row reads; when that text carries no stamps (the window's follower writes `-o cat`), over the journal itself."""
    t = line_stamp(text, pattern)
    if t is None:
        t = line_stamp(journal_text(since_epoch, runner), pattern)
    return t


def run_tokens(text):
    """(steps, tokens) of each `serve: batch of` line: the tokens its `req, tokens, end:` list names, summed."""
    out = []
    for m in BATCH_FULL.finditer(text):
        out.append((int(m.group(2)), sum(int(x) for x in re.findall(r'\d+ (\d+) \w+', m.group(5)))))
    return out


def exchange_turn(a, d, log, sid, hist, T, ctrl, ctrl_tok_s, ctrl_batches, ctrl_text=''):
    """Row exchange's return: the request's wait line, its first move and the run between them, read from the journal as they happen, through the decoders' handler."""
    mk = log.mark()
    t0 = time.time()
    rid = d.submit_watch(sid, hist + tokens_of(T, 16, 777), 24, keep=True)
    t_wait = t_move = t_cb = None
    req = None
    wait_re = re.compile(r'^serve: req (\d+) (?:waits for a slot|waits at its start)', re.M)
    t_end = t0 + 180
    while time.time() < t_end and 'done' not in d.watch[rid]:
        d.pump(time.time() + 0.05)
        txt = log.text(mk)
        if t_wait is None:
            m = wait_re.search(txt)
            if m:
                t_wait, req = time.time(), int(m.group(1))
        if req is not None:
            if t_cb is None and re.search(r'^serve: req %d can begin: the run is ended for it' % req, txt, re.M):
                t_cb = time.time()
            if t_move is None and re.search(r'^serve: req %d factor |^state: session \d+ back from disk|joined to position' % req, txt, re.M):
                t_move = time.time()
    check('done' in d.watch[rid], 'the return ended (DONE) within 180 s of its submit')
    txt = log.text(mk)
    need(req is not None, 'premise: the return has a wait line (`waits for a slot` or `waits at its start`)')
    w = wait_re.search(txt)
    mv = re.compile(r'^serve: req %d factor |^state: session \d+ back from disk|joined to position' % req, re.M).search(txt, w.end())
    seg = txt[w.end():mv.start() if mv else len(txt)]
    rides = len(re.findall(r'ride digest', seg))
    need(rides >= 2, 'premise: at least two `ride digest` lines between the wait line and the first move (%d): the wait spans runs' % rides)
    cb = re.search(r'^serve: req %d can begin: the run is ended for it' % req, seg, re.M)
    batches_before_cb = len(re.findall(r'serve: batch of \d+: ', seg[:cb.start()])) if cb else len(re.findall(r'serve: batch of \d+: ', seg))
    bi = batch_info(seg)
    print('  premise: wait line of req %d, %d ride digest lines before the first move, %d `batch of` lines, can begin %s' % (req, rides, len(bi), 'seen' if cb else 'absent'))
    # the run that `can begin` ends has at least 10 steps (premise); (i) none before it; (ii) exactly one `batch of` line in all (on a build without the owed-time rule, --no-owed-time: at most one more, of at most
    # --interleave-burst-max-ms + 2 steps), every end `paused`; (iii) the first move within 0.5 s of `can begin` and (vi) within 3.48 s of the wait line, both by the journal's stamps (the test's clock
    # judges only times of 2 s or more); (iv) the run against the control run; (v) printed: GPU ms by rows.
    rt = run_tokens(seg)
    need(bool(rt) and rt[0][0] >= 10, 'premise: the run the `can begin` line ends has at least 10 steps (%s)' % (rt[0][0] if rt else 'no run'))
    check(cb is not None and batches_before_cb == 0, '(i) no `serve: batch of` line between the wait line and the `can begin` line (%d)' % batches_before_cb)
    if getattr(a, 'no_owed_time', False):
        cap = a.burst_max_ms / max(1.0, bi[0][1]) + 2 if bi else 0
        check(len(bi) >= 1 and len(bi) <= 2 and all(x[3] for x in bi) and (len(bi) == 1 or bi[1][0] <= cap),
              '(ii, build without the owed-time rule) at most one more `batch of` line, of at most burst-max + 2 steps, every end `paused` (%d lines)' % len(bi))
    else:
        check(len(bi) == 1 and all(x[3] for x in bi), '(ii) exactly one `batch of` line between the wait line and the first move, every end `paused` (%d; on the earlier build by record: 2)' % len(bi))
    if bi and ctrl_batches:
        c_ms = median([x[1] for x in ctrl_batches])
        check(abs(bi[0][1] - c_ms) <= 0.05 * c_ms, '(iv) that run\'s ms a step %.1f within 5 %% of the control run\'s %.1f' % (bi[0][1], c_ms))
        c_rt = run_tokens(ctrl_text) if ctrl_text else []
        if rt and ctrl_tok_s and c_rt:
            have = rt[0][1] / max(1, rt[0][0] - 2)
            want = 0.95 * ctrl_tok_s * (c_ms / 1e3)
            check(have >= want, '(iv) tokens over (steps - 2): %.2f at least 0.95 of the control window\'s tokens a second x the control run\'s median step = %.2f' % (have, want))
        else:
            check(False, '(iv) the control run is not readable: the premise fails')
    gpu = re.findall(r'GPU ms by rows: ([^\n]*)', seg)
    print('  (v) the run\'s GPU ms by rows (printed for the reader): %s; the control run\'s: %s' % (gpu[0][:120] if gpu else 'none', (re.findall(r'GPU ms by rows: ([^\n]*)', ctrl_text) or ['none'])[0][:120] if ctrl_text else 'none'))
    st_w, st_c, st_m = stamp_of(txt, r'serve: req %d (?:waits for a slot|waits at its start)' % req, t0 - 5), stamp_of(txt, r'serve: req %d can begin: the run is ended for it' % req, t0 - 5), \
        stamp_of(txt, r'serve: req %d factor |state: session \d+ back from disk|joined to position' % req, t0 - 5)
    if st_w is not None and st_c is not None and st_m is not None:
        check(st_m - st_c <= 0.5, '(iii) the first move %.3f s after the `can begin` line by the journal (at most 0.5 s)' % (st_m - st_c))
        check(st_m - st_w <= 3.48, '(vi) the first move %.3f s after the wait line by the journal (at most 3.48 s)' % (st_m - st_w))
    else:
        check(False, '(iii, vi) the journal lines carry no stamps (neither the text the row reads nor `journalctl --user -u pico-engine -o short-precise` gave any): not judged by the test\'s clock under 2 s; read from the window script\'s capture')
    if t_wait is not None and t_move is not None:
        print('  longest decoder gap in the wait %.0f ms beside the control\'s %.0f ms' % (d.longest(t_wait, t_move) * 1e3, ctrl * 1e3))
    check(decoder_give_backs(seg, a.sessions) == 0, 'no `gives its` line naming a decoder\'s session in the wait (%d)' % decoder_give_backs(seg, a.sessions))
    check(len(MISS_LINE.findall(txt)) == 0, '0 lines `a run was ended for its start and it waits again` (%d)' % len(MISS_LINE.findall(txt)))
    w = d.watch[rid]
    return {'ids': list(w.get('ids', [])), 'cached': struct.unpack('<QIIII', w['frame'][:24])[3] if 'frame' in w else 0, 't_first': (w['first'] - t0) if w.get('first') else None, 'error': ''}


EXT_LINE = re.compile(r'^serve: req (\d+) ends at (\d+) tokens \(length\): every request that holds memory waits to grow and nothing can be freed', re.M)
EXT_BUG_LINE = re.compile(r'^serve: BUG: every flight waits for an extension', re.M)


# Row ext's sizing, from the engine's own code (each number with its source; engine tree = this repository's engine/):
#   chunk entries per number 8192 (model/store_geom.h:13, kChunkShift 13); four kv lists, three at 2 positions an entry and one at 1 (model/store_core.cpp:86 ent_pos, :93 ratio);
#   lookahead 4096 (model/store_geom.h:15) and slack 64 (model/model.cu:606) past a request's last position (model/model.cu:625-628 slot_chunks_for, :615-623 numbers_for);
#   a stored conversation's lists below `saved` positions: ceil(ceil(saved / ratio) / 8192) a list (checked: 343,664 -> 105 and 360,048 -> 110, the two counts a recorded journal printed);
#   a request's need = prompt + min(max_new, --grow-ahead) (server/serve.cpp:2538-2541 need_clamped); its want = max(slot_chunks_for(need) - lists in the pool, back) (server/serve.cpp:2634-2638
#   credit_of/want_of, model/pool_rules.h:126-133 request_want); --pool-tokens N gives ceil(N x 5 / 16384) numbers (server/main.cpp:1075).
CHUNK_ENTRIES, MAP_LOOKAHEAD, MAP_SLACK = 8192, 4096, 64


def list_numbers(pos):
    """Numbers a stored conversation of `pos` positions holds in its four lists."""
    return sum(-(-(-(-pos // (2 if k < 3 else 1))) // CHUNK_ENTRIES) for k in range(4))


def chunks_for(need):
    """model/model.cu:625 slot_chunks_for(s, need): numbers a line reserves for `need` positions."""
    upto = need - 1 + MAP_LOOKAHEAD + MAP_SLACK
    return sum(-(-((upto // (2 if k < 3 else 1)) + 1 + MAP_SLACK) // CHUNK_ENTRIES) for k in range(4))


def ext_numbers(poss, hist_extra, new_ids, grow_ahead, max_new):
    """(lists, want, reservations): the four lists, each second turn's own need beyond its list (chunks_for(need) - list), and the four lines' whole commitments (the pool the four turns
    need to admit with reuse: lists + wants). `hist_extra`: ids the first turn added to the history; `new_ids`: the second turn's added ids."""
    lists = [list_numbers(p) for p in poss]
    needs = [p + hist_extra + new_ids + min(max_new, grow_ahead) for p in poss]
    wants = [chunks_for(n) - l for n, l in zip(needs, lists)]
    return lists, wants, [chunks_for(n) for n in needs]


def ext_pool_tokens_lists(poss, grow_ahead=256, max_new=20000, free=2):
    """The --pool-tokens that leaves `free` numbers (under one extension's) after the four second turns are admitted: pool numbers = sum of the four commitments + free."""
    _, _, resv = ext_numbers(poss, 1, 16, grow_ahead, max_new)
    target = sum(resv) + free
    t = target * 16384 // 5
    while (t * 5 + 16383) // 16384 < target:
        t += 1
    while ((t - 1) * 5 + 16383) // 16384 >= target:
        t -= 1
    return t, target


def keep_snap(s, E, every):
    """server/serve.cpp:556-562 keep_snap: whether the window snapshot at position s stays for a history of E positions."""
    d, dense = E - s, 8 * every
    if d < dense:
        return True
    g, lim = 2 * every, 2 * dense
    while d >= lim:
        g, lim = g * 2, lim * 2
    return s % g == 0


def snap_count(n, every=1024):
    """The window bookmarks a kept conversation of `n` positions holds: those of keep_snap (taken at serve.cpp:2379-2382) plus the one at the prompt's end (:2466-2473). 30 at 343,664, 360,048 and 355,443."""
    return sum(1 for s_ in range(every, n + 1, every) if keep_snap(s_, n, every)) + 1


def ext_pool_tokens(P=355443, free=8):
    """Item 22 E 3 (revised): the --pool-tokens for four kept conversations of P positions: 4 lines of chunks_for(P + 1) + 4 x 2 numbers x snap_count(P) bookmarks + `free` numbers
    (server/main.cpp:1075: numbers = ceil(N x 5 / 16384)). Returns (tokens, numbers)."""
    target = 4 * chunks_for(P + 1) + 4 * 2 * snap_count(P) + free
    t = target * 16384 // 5
    while (t * 5 + 16383) // 16384 < target:
        t += 1
    while ((t - 1) * 5 + 16383) // 16384 >= target:
        t -= 1
    return t, target


POOL_LINE = re.compile(r'^serve: req (\d+) pool: (\d+) chunk numbers; stored sessions hold (\d+) [^,]*,? ?(?:\([^)]*\))?[^,]*, the lines (\d+) with their reservations, (\d+) free and unreserved', re.M)


EXT_STALL_WHOLE = re.compile(r'^serve: every request in flight waits for the pool \(4 paused, 0 waiting for a slot\): nothing can be imaged: they wait for a request to end$', re.M)
EXT_GROWS = re.compile(r'^serve: req (\d+) grows:', re.M)
EXT_BATCH = re.compile(r'^serve: batch of', re.M)


def _eol(text, m):
    """The whole line a match begins."""
    e = text.find('\n', m.start())
    return text[m.start():e if e >= 0 else len(text)]


def ext_others_grown(text, four):
    """The requests of `four` (the ids) other than the one the `ends at` line names that have a `serve: req N grows:` line after that line; (victim or None, set of them)."""
    em = EXT_LINE.search(text)
    if em is None:
        return None, set()
    after = text[em.end():]
    return int(em.group(1)), {int(r) for r in EXT_GROWS.findall(after)} & (set(four) - {int(em.group(1))})


def ext_counts(text, mode, four):
    """Row ext's journal checks, in place of row_counts: one standstill is made on purpose, so each of its lines is counted ONCE and read in its place.
    `text` is the log from the second turns' submit; `four` the ids of the four requests; `mode` b (the engine ends the longest reply) or a (the test cancels one: no `ends at` line).
    1 the stall line once (either ending) and it is the standstill's; 2 the BUG line once, `(4 flights)`; 3 (b) the `ends at` line once, `(4 requests,`; 4 the order;
    5 nothing moved between the stall line and the `ends at` line (a: the BUG line): no `grows:` and no `batch of` line; 6 (b) a `grows:` line of each of the other three after the `ends at` line."""
    sm = list(STALL_LINE.finditer(text))
    bm = list(EXT_BUG_LINE.finditer(text))
    em = list(EXT_LINE.finditer(text))
    whole = EXT_STALL_WHOLE.findall(text)
    n_grows = len(GROWS_LINE.findall(text))
    print('  ext: journal counts: BUG %d, stall %d, ends %d, grows %d (row ext is inside the standstill rule on purpose)' % (len(bm), len(sm), len(em), n_grows))
    check(len(sm) == 1 and len(whole) == 1, 'ext: the stall line once, and it is the standstill\'s (%d stall line(s) of either kind, %d the whole standstill line)' % (len(sm), len(whole)))
    check(len(bm) == 1 and _eol(text, bm[0]).endswith('(4 flights)'), 'ext: the standstill line once over the row (%d, %s)' % (len(bm), '(4 flights)' if bm and _eol(text, bm[0]).endswith('(4 flights)') else 'not `(4 flights)`'))
    if mode == 'b':
        check(len(em) == 1 and '(4 requests,' in _eol(text, em[0]), 'ext: one request ended for the standstill (%d, %s)' % (len(em), '(4 requests,' if em and '(4 requests,' in _eol(text, em[0]) else 'not `(4 requests,`'))
    end = em[0].start() if mode == 'b' and em else (bm[0].start() if bm else -1)
    order = bool(sm) and bool(bm) and sm[0].start() < bm[0].start() and (mode != 'b' or (bool(em) and bm[0].start() < em[0].start()))
    check(order, 'ext: the order: the stall line, then the BUG line%s' % (', then the `ends at` line' if mode == 'b' else ''))
    if sm and end > sm[0].start():
        seg = text[sm[0].end():end]
        g, bt = len(EXT_GROWS.findall(seg)), len(EXT_BATCH.findall(seg))
    else:
        g = bt = -1
    check(g == 0 and bt == 0, 'ext: nothing moved inside the standstill (%d `grows:` lines and %d `batch of` lines between the stall line and the %s)' % (g, bt, '`ends at` line' if mode == 'b' else 'BUG line'))
    if mode == 'b':
        victim, got = ext_others_grown(text, four)
        want = sorted(set(four) - {victim}) if victim is not None else []
        check(bool(want) and got == set(want), 'ext: the standstill ends by the engine\'s own lines: a `grows:` line after the `ends at` line of each of the other three (%s of %s)' % (sorted(got), want))


def cmd_ext(a, e, T, log):
    """Row ext: start `--slots 4 --big-slots 0 --pool-tokens 2251162 --disk-gib 256 --kv-split off --grow-ahead 256` (688 numbers). Phase 1: four kept
    conversations of P = 355,443 positions one after another (max_new 1); premise read from the fourth's `serve: req N pool:` line. Phase 2: their second turns at once (max_new 60,000);
    the standstill line waited for up to 720 s. Mode a: the test cancels one; mode b: the engine ends the longest reply (exactly one DONE `length`, its id count the line's M)."""
    P = 355443
    d = Decoders(e, T, 0, 0)
    sids = [0x73706ea0 + k for k in range(4)]
    hist = []
    for k in range(4):
        rid = d.submit_watch(sids[k], tokens_of(T, P, 1000 * k), 1, keep=True)
        d.pump(time.time() + 3600, lambda rid=rid: 'done' in d.watch[rid])
        hist.append(tokens_of(T, P, 1000 * k) + list(d.watch[rid].get('ids', [])))
    # the engine prints a request's `pool:` line as the request ends, beside its DONE: the fourth line may not be in the log yet
    t_pl = time.time() + 15
    while len(POOL_LINE.findall(log.text(0))) < 4 and time.time() < t_pl:
        time.sleep(0.2)
    txt = log.text(0)
    start_pool = re.findall(r'chunk pool: (\d+) numbers', txt)
    pls = POOL_LINE.findall(txt)
    tok, target = ext_pool_tokens(P)
    lines_want = 4 * chunks_for(P + 1)
    print('  c check from the code: the four lines %d (chunks_for(%d) = %d each), the bookmarks 4 x 2 x %d = %d, free 8: %d numbers = --pool-tokens %d; the start says %s' % (
        lines_want, P + 1, chunks_for(P + 1), snap_count(P), 8 * snap_count(P), target, tok, start_pool[-1:] or 'no pool line'))
    for ln in re.findall(r'^serve: req \d+ pool: [^\n]*', txt, re.M)[-4:]:
        print('    ' + ln[:230])
    ok1 = bool(pls) and int(pls[-1][1]) == target and int(pls[-1][2]) == 0 and int(pls[-1][3]) == lines_want and 4 <= int(pls[-1][4]) <= 20 and not GAVE_BACK.findall(txt)
    if not ok1:
        pool_n, free_n = (int(pls[-1][1]), int(pls[-1][4])) if pls else (0, 0)
        nt, nn = ext_pool_tokens(P, 8)
        nxt = ((pool_n - free_n + 8) * 16384 + 4) // 5 if pool_n else nt
        print('  premise (phase 1) missed: pool/stored/lines/free = %s, `gives its` lines %d; for the next start: --pool-tokens %d (pool - free + 8 = %d numbers)' % (
            pls[-1][1:] if pls else 'no pool line', len(GAVE_BACK.findall(txt)), nxt, pool_n - free_n + 8))
    need(ok1, 'premise after phase 1: `req 4 pool:` reads %d chunk numbers, stored sessions hold 0, the lines %d, free 4 to 20, no `gives its` line' % (target, lines_want))
    m0 = log.mark()
    rids = [d.submit_watch(sids[k], hist[k] + tokens_of(T, 16, 777 + k), 60000, keep=True) for k in range(4)]
    t0 = time.time()
    d.pump(t0 + 720, lambda: bool(EXT_BUG_LINE.search(log.text(m0))))
    seg = log.text(m0)
    t_line = time.time()
    adm = re.findall(r'^serve: req \d+ (?:waits[^\n]*|[^\n]*positions need \d+ numbers[^\n]*)', seg, re.M)
    print('  premise output: the four second turns\' admission lines: %d lines' % len(adm))
    for ln in adm[:16]:
        print('    ' + ln[:200])
    print('  factor lines: %s' % re.findall(r'^serve: req \d+ factor [^\n]*', seg, re.M)[:8])
    grows = re.findall(r'^serve: req (\d+) grows: its reservation reaches (\d+)', seg, re.M)
    passed = sorted({int(r_) for r_, v in grows if int(v) > 356160})
    print('  free after phase 1 %s; %.0f s from the submit to the standstill line; %d `grows:` lines; requests whose reservation passed 356,160: %s' % (
        pls[-1][4] if pls else '?', t_line - t0, len(grows), passed))
    need(len(re.findall(r'^serve: req \d+ factor .*reused [1-9]', seg, re.M)) == 4 and all(d.watch[r].get('first') for r in rids), 'premise: four `factor` lines with reused above 0 and four first tokens')
    need(bool(EXT_BUG_LINE.search(seg)), 'premise: the standstill line `BUG: every flight waits for an extension` within 720 s of the submit')
    need(not GAVE_BACK.findall(seg), 'premise: no `gives its` line between the second turns\' submit and the standstill line (%d)' % len(GAVE_BACK.findall(seg)))
    check(len(EXT_BUG_LINE.findall(seg)) == 1, 'the standstill line once (%d)' % len(EXT_BUG_LINE.findall(seg)))
    if a.ext_mode == 'a':
        c1 = rids[0]
        e.send(CANCEL, struct.pack('<Q', c1))
        d.pump(time.time() + 1, lambda: 'done' in d.watch[c1])
        check('done' in d.watch[c1], '(a) the cancelled request\'s DONE within 1 s')
        mk = {r: len(d.watch[r].get('ids', [])) for r in rids[1:]}
        d.pump(time.time() + 10, lambda: all(len(d.watch[r].get('ids', [])) > mk[r] for r in rids[1:]))
        check(all(len(d.watch[r].get('ids', [])) > mk[r] for r in rids[1:]) and 'grows: its reservation reaches' in log.text(m0), '(a) the other three give tokens again within 10 s (`grows:` lines present)')
    else:
        d.pump(time.time() + 5, lambda: bool(EXT_LINE.search(log.text(m0))))
        em = EXT_LINE.search(log.text(m0))
        check(em is not None, '(b) within 5 s of the standstill line one request ends with the new line (%s)' % (em.group(0)[:80] if em else 'none'))
        if em:
            victim, m_tok = int(em.group(1)), int(em.group(2))
            d.pump(time.time() + 3, lambda: sum('done' in d.watch[r] for r in rids) >= 1)
            ended = [r for r in rids if 'done' in d.watch[r]]
            check(len(ended) == 1, '(b) exactly one request has a DONE (%d)' % len(ended))
            if ended:
                reason = struct.unpack('<QIIII', d.watch[ended[0]]['frame'][:24])[1]
                n_ids = len(d.watch[ended[0]].get('ids', []))
                check(reason == 1 and ended[0] == victim, '(b) the DONE is the line\'s request %d with reason length (reason %d, request %d)' % (victim, reason, ended[0]))
                check(n_ids == m_tok, '(b) as many ids received as the line\'s %d tokens (%d)' % (m_tok, n_ids))
            rest = [r for r in rids if r not in ended]
            mk = {r: len(d.watch[r].get('ids', [])) for r in rest}
            t_r = time.time()
            d.pump(t_r + 10, lambda: all(len(d.watch[r].get('ids', [])) > mk[r] for r in rest))
            check(all(len(d.watch[r].get('ids', [])) > mk[r] for r in rest), '(b) the other %d give tokens again within 10 s (%.1f s)' % (len(rest), time.time() - t_r))
    check(d.poll_stats()['image_waits_total'] == 0, 'image_waits_total 0')
    if a.ext_mode == 'b':
        # the three others move again only through their `grows:` lines, a moment after the `ends at` line: wait for them before the count
        d.pump(time.time() + 10, lambda: len(ext_others_grown(log.text(m0), rids)[1]) == 3)
    ext_counts(log.text(m0), a.ext_mode, rids)
    d.finish()
    return 0


def cmd_imager(a):
    """The imager's GPU rows. See the module's docstring. Every row prints its premise and exits 2
    (NotReached) when it is not reached. The events that need the pool short (a2n, back, exchange, ssd, q) say the start they need."""
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    log = Log(a.log) if a.log else None
    need(log is not None, 'the imager rows read the engine log: --log')
    sid = 0x73706d00 + a.size // 1000
    if a.event == 'ext':
        return cmd_ext(a, e, T, log)
    if a.event == 'ref':
        # the reference for a2n, back and exchange (one turn): the conversation alone, no decoder, no window, no premise about rides, ghosts or marks; run on the
        # no-tier start (--disk-gib 0: no imager) as eqr's second command; --out holds the ids under the three tags
        o0 = e.run(sid, tokens_of(T, a.size, 0), max_new=1, keep=True, stop=())
        check(o0['error'] == '', 'conversation %016x made: %d prompt positions, error %r' % (sid, o0['prompt'], o0['error']))
        time.sleep(2)
        r = imager_turn(e, T, sid, tokens_of(T, a.size, 0) + list(o0['ids']), 'ref')
        if a.out:
            json.dump({t: {'ids': r['ids'], 'cached': r['cached']} for t in ('a2n', 'back', 'exchange')}, open(a.out, 'w'))
        return 0
    if a.event == 'g' and a.decode_new < 200000:
        a.decode_new = 200000  # the decoders must outlive the row (at one user the decoder ended after 103 s, before the 1M conversation was read)
    if a.event in ('a2n', 'g', 'back', 'exchange', 'ssd'):
        # the conversation first, with no decoder: its prefill is read alone (the interleaver's own business, not the imager's)
        o0 = e.run(sid, tokens_of(T, a.size, 0), max_new=1, keep=True, stop=())
        check(o0['error'] == '', 'conversation %016x made: %d prompt positions, error %r' % (sid, o0['prompt'], o0['error']))
        hist = tokens_of(T, a.size, 0) + list(o0['ids'])
    d = Decoders(e, T, a.sessions, a.decode_new)
    given_back = 0  # the decoders' lines given back to the pool during the row (set below where the row expects none)
    if a.sessions > 0:  # every row with decoders fails (never "not reached") when tokens stop for 10 s: the window's script then takes the capture
        d.pump(time.time() + 60, lambda: all(t is not None for t in d.last.values()))
        d.stall_guard, d.guard_mark = (10.0, log), log.mark()
    ctrl = 0.0
    ctrl_text, ctrl_batches, ctrl_tok_s = '', [], 0.0
    if a.event in ('a2n', 'g'):
        # a2n and g take their mark and its first reading right after the decoders' first tokens, before any window: with --kv-split-after-s 2 the rides begin as soon as the
        # decoders run, so a window taken first would let them image everything unseen (`rides rose (0)` in a recorded run) and would not be a control either. Its control window
        # comes after the event window, when nothing is left to image.
        mark = log.mark()
        s1 = d.poll_stats()
        s0 = s1
        need(s0 is not None and 'split_rides_total' in s0, 'STATS carries the imager\'s fields (a build with the imager)')
    else:
        d.window(20)
        s0 = d.poll_stats()
        need(s0 is not None and 'split_rides_total' in s0, 'STATS carries the imager\'s fields (a build with the imager)')
        t = time.time()
        cm0 = log.mark()
        d.window(a.window)
        ctrl = d.longest(t, time.time())
        ctrl_tok_s = d.tok_per_s(t, time.time())
        ctrl_text = log.text(cm0)
        ctrl_batches = batch_info(ctrl_text)  # the control run's own lines: tokens a step and ms a step
        print('control window %.0f s: longest gap %.0f ms, %.1f tokens/s' % (a.window, ctrl * 1e3, ctrl_tok_s))
    ev0, ev1 = time.time(), time.time()
    if a.event == 'a2n':
        # the premise: the marks ride a run that never ended (rides_phase); then the fillers and the return, whose window holds a long prompt's pieces: printed, not judged
        ev_ctl = rides_phase(a, d, log, mark, s1)
        ctrl = ev_ctl['ctrl']
        ev0 = time.time()
        # one decoder is cancelled: --slots 8 holds the eight decoders and the prompt being read needs the ninth slot (a decoder without a slot is paused, not stopped)
        # the paused-slot test: with the eight decoders on --slots 8 a ninth request takes nothing from them and waits; a cancel frees its slot at once; an arrival after an end starts
        # within a second. Three steps before the fillers.
        m9 = log.mark()
        rid9 = d.add_decoder(T, 8, a.decode_new)
        t9 = time.time()
        d.pump(t9 + 30)
        seg9 = log.text(m9)
        check(not re.search(r'^serve: req %d (factor |waits)' % rid9, seg9, re.M), '(i) the ninth request (req %d) has no `factor` line and no wait line in 30 s' % rid9)
        check(not BATCH_LINE.findall(seg9), '(i) no `serve: batch of` line in the 30 s (%d)' % len(BATCH_LINE.findall(seg9)))
        check(not GAVE_BACK.findall(seg9), '(i) no `gives its` line (%d)' % len(GAVE_BACK.findall(seg9)))
        tk = d.tok_per_s(t9, t9 + 30)
        check(tk >= 0.95 * ev_ctl['tok_s'], '(i) the decoders\' tokens a second %.1f at least 0.95 of the control window\'s %.1f' % (tk, ev_ctl['tok_s']))
        live9 = [r for r in d.last if r not in d.ended and r != rid9]
        c1 = live9[0]
        e.send(CANCEL, struct.pack('<Q', c1))
        t_c1 = time.time()
        d.pump(t_c1 + 1, lambda: c1 in d.ended)
        check(c1 in d.ended, '(ii) the cancelled decoder\'s (req %d) DONE came within 1 s of the CANCEL' % c1)
        t_done1 = time.time()
        tf9 = None
        while time.time() < t_done1 + 1 and tf9 is None:
            d.pump(time.time() + 0.01)
            if re.search(r'^serve: req %d factor ' % rid9, log.text(m9), re.M):
                tf9 = time.time()
        check(tf9 is not None, '(ii) the ninth\'s `factor` line within 1 s of that DONE (%s)' % ('%.0f ms' % (1e3 * (tf9 - t_done1)) if tf9 else 'none'))
        seg9 = log.text(m9)
        bl = re.findall(r'^serve: batch of 8: .*?end: ([^;]*);', seg9, re.M)
        ends9 = re.findall(r'(\d+) \d+ (\w+)', bl[0]) if len(bl) == 1 else []
        check(len(BATCH_LINE.findall(seg9)) == 1 and len(bl) == 1 and any(int(r_) == c1 and w_ == 'cancelled' for r_, w_ in ends9) and all(w_ == 'paused' for r_, w_ in ends9 if int(r_) != c1),
              '(ii) exactly one `batch of 8` line between them, the others ended `paused`, the cancelled request %d named' % c1)
        d.pump(time.time() + 10, lambda: d.last.get(rid9) is not None)
        print('  the ninth\'s wait from its send to its first token: %s' % ('%.2f s' % (d.last[rid9] - t9) if d.last.get(rid9) else 'no token in 10 s'))
        need(d.last.get(rid9) is not None, 'the ninth decodes (the next run has begun)')
        c2 = [r for r in d.last if r not in d.ended and r not in (rid9,)][0]
        e.send(CANCEL, struct.pack('<Q', c2))
        d.pump(time.time() + 1, lambda: c2 in d.ended)
        check(c2 in d.ended, '(iii) one more decoder (req %d) cancelled: its DONE within 1 s' % c2)
        m10 = log.mark()
        t10 = time.time()
        rid10 = d.add_decoder(T, 9, a.decode_new)
        tf10 = None
        while time.time() < t10 + 1 and tf10 is None:
            d.pump(time.time() + 0.01)
            if re.search(r'^serve: req %d factor ' % rid10, log.text(m10), re.M):
                tf10 = time.time()
        check(tf10 is not None, '(iii) the tenth request\'s `factor` line within 1 s of its send (%s)' % ('%.0f ms' % (1e3 * (tf10 - t10)) if tf10 else 'none'))
        seg_all = log.text(m9)
        check(not GAVE_BACK.findall(seg_all), 'throughout: no decoder\'s line given back (%d)' % len(GAVE_BACK.findall(seg_all)))
        check(len(MISS_LINE.findall(seg_all)) == 0, 'throughout: 0 miss lines (%d)' % len(MISS_LINE.findall(seg_all)))
        check(d.poll_stats()['image_waits_total'] == 0, 'throughout: image_waits_total 0')
        # the fillers need a slot too: one more decoder is cancelled
        first_rid = [r for r in d.last if r not in d.ended][0]
        e.send(CANCEL, struct.pack('<Q', first_rid))
        t_c = time.time() + 1  # the cancel ends like a stop (DONE at once): within 1 s
        d.pump(t_c, lambda: first_rid in d.ended)
        need(first_rid in d.ended, 'a slot is free for the fillers and the return: the cancelled decoder\'s DONE came within 1 s of the CANCEL')
        g0 = d.poll_stats()['ghost_numbers']
        mk2 = log.mark()
        fk = 0
        while fk < a.ssd_fillers:
            mkf = log.mark()
            ts_f = time.time()
            f = d.submit_watch(sid + 1 + fk, tokens_of(T, a.size, 4242 + 1000 * fk), 1)
            if fk == 0:  # the freed slot: the first filler's `factor` line within one step plus 50 ms of its submit
                tf = None
                while time.time() < ts_f + 5 and tf is None:
                    d.pump(time.time() + 0.01)
                    if re.search(r'^serve: req \d+ factor ', log.text(mkf), re.M):
                        tf = time.time()
                check(tf is not None and tf - ts_f <= 2.0, 'the first filler\'s `serve: req N factor` line within 2 s of its submit by the test\'s clock' % (
                    '%.0f ms' % (1e3 * (tf - ts_f)) if tf is not None else 'none in 5 s'))
                segf = log.text(mkf)
                nb = [x for x in run_tokens(segf[:re.search(r'^serve: req \d+ factor ', segf, re.M).start()])] if tf is not None else []
                check(len(nb) == 1 and nb[0][0] <= 3, 'exactly one `batch of` line between the cancelled decoder\'s end and the filler\'s `factor` line, of at most 3 steps (%s)' % (nb,))
            d.pump(time.time() + 3600, lambda f=f: 'done' in d.watch[f])
            fk += 1
            if g0 - d.poll_stats()['ghost_numbers'] >= 64:
                break
        d.pump(time.time() + 5)
        s2 = d.poll_stats()
        need(g0 - s2['ghost_numbers'] >= 64, 'the fillers took ghosts: %d -> %d (at least 64 in at most %d fillers)' % (g0, s2['ghost_numbers'], a.ssd_fillers))
        # the first run ends when the first filler is admitted: its end lines are read now (row g8's judgement, completed, and the step cross-check)
        txt2 = log.text(mark)
        ends = [float(m.group(4)) for m in RIDE_LINE.finditer(txt2)]
        need(len(ends) > 0, 'a `split rides in the run` line since the mark')
        check(max(ends) < ev_ctl['med'] * 1e3, 'the longest ride of any run %.2f ms, under the step (%.1f ms)' % (max(ends), ev_ctl['med'] * 1e3))
        eng = engine_step_ms(txt2)
        need(steps_agree(ev_ctl['med'] * 1e3, eng), 'the test\'s step %.1f ms agrees within 20 %% with the engine\'s `ms a step` %.1f ms' % (ev_ctl['med'] * 1e3, eng))
        need(s2['returns_read_total'] >= 0 and s2['clean_numbers'] >= 0, 'STATS read after the fillers')
        r = imager_turn(e, T, sid, hist, 'the return', d)  # through the decoders' handler: the decoders stream on this connection
        s3 = d.poll_stats()
        check(bug_lines(log.text(0)) == 0, 'lines BUG in the log: %d' % bug_lines(log.text(0)))
        check(s3['image_waits_total'] == 0, 'image_waits_total 0 (a take never waited for an image): %d' % s3['image_waits_total'])
        check(s3['returns_read_total'] > s2['returns_read_total'], 'the return read pieces from the SSD: returns_read_total up by %d' % (s3['returns_read_total'] - s2['returns_read_total']))
        mb = list(re.finditer(back_re(sid), log.text(mark), re.M))
        check(bool(mb) and mb[-1].group(4) == mb[-1].group(5) and int(mb[-1].group(4)) > 0, 'restart point N of N: %s' % (mb[-1].group(0)[-60:] if mb else 'no read-back line'))
        imager_equal(a, 'a2n', r)
        ev1 = time.time()
    elif a.event in ('back', 'exchange', 'ssd'):
        # back: 2 s later, every ghost still there. exchange: fillers took the ghosts. ssd: fillers until the conversation is only on the disk tier.
        mark = log.mark()
        need(d.poll_stats()['kv_split'] == 1, 'STATS: kv_split 1 (--kv-split tiered or max: ghosts exist)')
        t_end = time.time() + a.window
        s = d.poll_stats()
        while time.time() < t_end and s['ghost_numbers'] == 0:
            d.pump(time.time() + 1.0)
            s = d.poll_stats()
        need(s['ghost_numbers'] > 0, 'premise: the conversation was marked (ghost_numbers %d)' % s['ghost_numbers'])
        pieces = split_wanted(a.size)
        if a.event == 'back':
            d.pump(time.time() + 2)
        else:
            k = 0
            # exchange: fillers of --size positions (a take uses plain free numbers first, so a small filler touches no ghost), repeated until ghost_numbers fell by at
            # least 64, at most three; the free numbers and the ghosts are printed before each. ssd: fillers until conv_on_disk >= 1.
            limit = max(a.ssd_fillers, a.ssd_fillers_max, 1) if a.event == 'ssd' else 3
            while k < limit:
                if a.event == 'ssd' and k >= a.ssd_fillers and d.poll_stats()['conv_on_disk'] >= 1:
                    break
                if a.event == 'exchange':
                    sk = d.poll_stats()
                    print('  filler %d of %d positions: free %d, reserved %d, ghosts %d' % (k + 1, a.size, sk['pool_free'], sk['pool_reserved'], sk['ghost_numbers']))
                    if s['ghost_numbers'] - sk['ghost_numbers'] >= 64:
                        break  # the ghosts fell
                f = d.submit_watch(sid + 1 + k, tokens_of(T, a.size, 4242 + 1000 * k), 1)
                d.pump(time.time() + 3600, lambda f=f: 'done' in d.watch[f])
                k += 1
            print('fillers: %d' % k)
        s3 = d.poll_stats()
        if a.event == 'ssd':
            need(s3['conv_on_disk'] >= 1, 'premise: the conversation is only on the disk tier (conv_on_disk %d)' % s3['conv_on_disk'])
        if a.event == 'exchange':
            need(s['ghost_numbers'] - s3['ghost_numbers'] >= 64, 'premise: the fillers took ghosts, at least 64 (%d -> %d)' % (s['ghost_numbers'], s3['ghost_numbers']))
        t0 = time.time()
        if a.event == 'exchange':
            r = exchange_turn(a, d, log, sid, hist, T, ctrl, ctrl_tok_s, ctrl_batches, ctrl_text)
        else:
            r = imager_turn(e, T, sid, hist, a.event, d)  # through the demultiplexer: the decoders stream on this connection
        s4 = d.poll_stats()
        ttft = (r['t_first'] or 0)
        if a.event == 'back':
            check(s4['revives_total'] - s3['revives_total'] >= pieces // 2, 'revives_total up by %d (the conversation\'s %d pieces)' % (s4['revives_total'] - s3['revives_total'], pieces))
            check(s4['returns_read_total'] == s3['returns_read_total'], 'returns_read_total unchanged: nothing was read')
            reads = store_reads(log.text(mark))
            check(bool(reads) and all(mib == 0.0 and pieces == 0 for mib, pieces in reads), 'nothing was read: every store `back from disk` line says 0.0 MiB and 0 pieces checked (%s)' % reads)
        else:
            check(s4['returns_read_total'] > s3['returns_read_total'], 'returns_read_total up by %d' % (s4['returns_read_total'] - s3['returns_read_total']))
        check(s4['image_waits_total'] == 0, 'image_waits_total 0: %d' % s4['image_waits_total'])
        check(bug_lines(log.text(0)) == 0, 'lines BUG in the log: %d' % bug_lines(log.text(0)))
        imager_equal(a, a.event, r)
        print('first token %.2f s (compare with the control twin: at most its + 250 ms)' % ttft)
        ev0, ev1 = t0, time.time()
    elif a.event == 'g':
        # g is a2n's first half (rides alone): rides_phase, then the decoders are cancelled and the run's end line is read; the rides' longest is judged against the control's step
        ev_ctl = rides_phase(a, d, log, mark, s1)
        ctrl = ev_ctl['ctrl']
        d.stall_guard = None
        d.finish()  # the decoders outlive the row and are cancelled here: a run's end line is printed when the run ends
        ends = []
        t_end = time.time() + 10
        while time.time() < t_end and not ends:
            d.pump(time.time() + 1.0)
            ends = [float(m.group(4)) for m in RIDE_END.finditer(log.text(mark))]
        need(len(ends) > 0, 'premise: a `split rides in the run` line after the cancel: %d' % len(ends))
        check(max(ends) < ev_ctl['med'] * 1e3, 'the longest ride of any run %.2f ms, under the control window\'s step (%.1f ms)' % (max(ends), ev_ctl['med'] * 1e3))
        eng, src = engine_step(log.text(mark))
        need(steps_agree(ev_ctl['med'] * 1e3, eng), 'the test\'s step %.1f ms agrees within 20 %% with the engine\'s %.1f ms (source: %s)' % (ev_ctl['med'] * 1e3, eng, src))
        s_end = d.poll_stats()
        check(s_end['image_waits_total'] == 0, 'image_waits_total 0: %d' % s_end['image_waits_total'])
        ev0, ev1 = ev_ctl['ev0'], ev_ctl['ev0']  # judged inside rides_phase: nothing left for the final verdict
    elif a.event == 'b':
        mark = log.mark()
        s1 = d.poll_stats()
        rid = d.submit_watch(sid, tokens_of(T, a.size, 0), 1)
        d.pump(time.time() + 3600, lambda: 'done' in d.watch[rid])
        # The conversation stays on its line: no end bookmark exists (it is taken only when a line lets its conversation go into the store, persist_slot ->
        # end_take; a client RELEASE deletes the conversation and takes none). The row judges the lists' pieces and window bookmarks; the end bookmark's image
        # is proven by the rows whose conversations leave their lines kept (s2: end_take and `restart point N of the N`; a2n, back, exchange).
        s = d.poll_stats()
        t_end = time.time() + a.window
        snaps0 = window_snapshots(log.text(mark), sid, a.size) or 0
        want0 = imager_expected_bytes(a.size, snaps0)
        while time.time() < t_end and (s['image_pending'] > 0 or s['spill_bytes_total'] - s1['spill_bytes_total'] < want0):
            d.pump(time.time() + 1.0)
            s = d.poll_stats()
        d.pump(time.time() + 5)
        s = d.poll_stats()
        text = log.text(mark)
        snaps = window_snapshots(text, sid, a.size) or 0
        want = imager_expected_bytes(a.size, snaps)
        got = s['spill_bytes_total'] - s1['spill_bytes_total']
        check(got == want, 'spill_bytes_total rose by %d B, wanted (%d lists + %d window bookmarks) x 786,432 = %d (the conversation is on its line: no end bookmark)' % (got, split_wanted(a.size), snaps, want))
        check(s['image_waits_total'] == 0, 'image_waits_total 0: %d' % s['image_waits_total'])
        check(not re.search(r'whole image', text), 'no whole-image line in the log')
        ev0, ev1 = time.time(), time.time()
    else:  # qi / qw / qd / qs: a request that needs plain pieces' blocks waits for their images and is admitted; never refused, never a bookmark dropped to fit
        mark = log.mark()
        need_n = a.size // 2900  # numbers the request needs (E from M: 100,000 positions = 34)
        nfill = 0
        if a.event == 'qd':  # fillers of --size until the free numbers are under the request's need plus 40, at most 12
            while nfill < 12 and d.poll_stats()['pool_free'] >= need_n + 40:
                f = d.submit_watch(sid + 1 + nfill, tokens_of(T, a.size, 1000 * (nfill + 1)), 1)
                d.pump(time.time() + 3600, lambda f=f: 'done' in d.watch[f])
                nfill += 1
        else:
            for nfill in range(a.fill):
                f = d.submit_watch(sid + 1 + nfill, tokens_of(T, a.size, 1000 * (nfill + 1)), 1)
                d.pump(time.time() + 3600, lambda f=f: 'done' in d.watch[f])
            nfill = a.fill
        sq = d.poll_stats()
        print('fillers: %d; free %d, clean %d, image_pending %d, need about %d numbers' % (nfill, sq['pool_free'], sq['clean_numbers'], sq['image_pending'], need_n))
        if a.event in ('qi', 'qw'):
            need(sq['pool_free'] + sq['clean_numbers'] < need_n + 8, 'premise: free %d plus clean %d are under its need (about %d numbers)' % (sq['pool_free'], sq['clean_numbers'], need_n))
        if a.event == 'qd':
            gtxt = log.text(mark)
            dgb = decoder_give_backs(gtxt, a.sessions)
            print('  give-backs since the mark: decoders %d, others (fillers, for their window bookmarks) %d' % (dgb, len(GAVE_BACK.findall(gtxt)) - dgb))
            need(dgb == 0, 'premise: no decoder gave its chunks back (else the pool was too small for the row)')
        msub = log.mark()
        t0 = time.time()
        r = d.submit_watch(sid + 100, tokens_of(T, a.size, 31337), 1)
        wait_re = re.compile(r'^serve: req %d (waits for a slot|waits at its start)' % r, re.M)
        start_re = re.compile(r'^serve: req %d factor ' % r, re.M)
        stall_b = re.compile(r'^serve: req %d waits at its start for the pool and nothing in flight can step' % r, re.M)
        t_wait = t_start = None
        t_end = t0 + 180  # no DONE after 180 s fails the row (it does not wait for ever)
        while time.time() < t_end and 'done' not in d.watch[r]:
            d.pump(time.time() + 0.2)
            txt = log.text(mark)
            if t_wait is None and wait_re.search(txt):
                t_wait = time.time()
            if t_start is None and start_re.search(txt):
                t_start = time.time()
        text = log.text(mark)
        s2 = d.poll_stats()
        check('done' in d.watch[r], 'the request ended (DONE) within 180 s of its submit')
        wm = wait_re.search(text)
        if a.event in ('qi', 'qw', 'qd', 'qs'):
            need(wm is not None, 'premise: the log has a wait line for the request (`waits for a slot` or `waits at its start`)')
            print('  wait lines: %s' % sorted({m.group(1) for m in wait_re.finditer(text)}))
        if a.event in ('qi', 'qw', 'qd', 'qs') and wm is not None:
            sm = start_re.search(text, wm.end())
            between = text[wm.end():sm.start()] if sm is not None else text[wm.end():]
            ride_pieces, idle_lines = carrier(between)
            print('  carrier between the wait line and the start: %d pieces imaged by rides, %d idle `split: settled:` lines' % (ride_pieces, idle_lines))
            need(ride_pieces > 0 or idle_lines > 0, 'the room came from the imager (rides or idle commands)')
            if a.event in ('qi', 'qw'):  # an idle command prints a settled line; a ride prints none (its pieces are in the run's end line)
                check(idle_lines > 0, 'a `serve: split: settled:` line between the wait line and the request\'s `serve: req N factor` line')
            lim = 5 if a.event in ('qi', 'qw') else 30
            check(t_wait is not None and t_start is not None and t_start - t_wait <= lim, 'the request\'s start came within %d s of its wait line (%s s)' % (
                lim, '%.1f' % (t_start - t_wait) if t_wait is not None and t_start is not None else 'no start line'))
        mk_ok, mk_nothing = stall_lines(text)
        print('  stall lines: %d `the imager makes room`, %d `nothing can be imaged` (0 is the expected reading in qs, qd)' % (mk_ok, mk_nothing))
        check(mk_nothing == 0, 'no `nothing can be imaged` ending')
        if a.event == 'qi':
            check(stall_b.search(text) is not None, 'the stall line of a prompt waiting at its start: `waits at its start for the pool and nothing in flight can step`')
        if a.event == 'qs':
            seg_s = log.text(msub)
            req_n = int(next(iter(re.findall(r'^serve: req (\d+) (?:waits for a slot|waits at its start)', seg_s, re.M)), 0)) or 0
            check(giveback_violations(seg_s, req_n, a.sessions) == 0 if req_n else True, 'no `gives its N chunks back` line names a session with a request in flight unless the request\'s `factor` line follows with no wait line between (%d)' % (giveback_violations(seg_s, req_n, a.sessions) if req_n else 0))
            check(not decoders_wait_at_start(seg_s, a.sessions), 'no decoder `waits at its start` (%s)' % decoders_wait_at_start(seg_s, a.sessions))
            check(not req_n or runs_between(seg_s, r'^serve: req %d waits at its start' % req_n, r'^serve: req %d factor ' % req_n) <= 1, 'at most one `batch of` line between the request\'s `waits at its start` and its start (%d)' % (runs_between(seg_s, r'^serve: req %d waits at its start' % req_n, r'^serve: req %d factor ' % req_n) if req_n else 0))
            check(len(MISS_LINE.findall(seg_s)) == 0, '0 miss lines (%d)' % len(MISS_LINE.findall(seg_s)))
        if a.event in ('qd', 'qs'):
            print('  free %d plus clean %d against its need about %d numbers (printed, no gate)' % (sq['pool_free'], sq['clean_numbers'], need_n))
            check(all(r2 in d.ended or d.last[r2] is not None and time.time() - d.last[r2] < 10 for r2 in d.last), 'each decoder ended or still gave tokens at the row\'s end')
        if a.event == 'qd':
            check(s2['split_rides_total'] > sq['split_rides_total'], 'admission after rides: split_rides_total rose %d -> %d' % (sq['split_rides_total'], s2['split_rides_total']))
            print('  `serve: batch of` lines between the wait line and the start (printed, not judged): %d' % (len(BATCH_LINE.findall(text[wm.end():])) if wm else -1))
        check(s2['image_waits_total'] == 0, 'image_waits_total 0: %d' % s2['image_waits_total'])
        check('cannot hold this request' not in text, 'no `cannot hold this request` line')
        check('dropped to fit the pool' not in text, 'no bookmark `dropped to fit the pool` line')
        print('its first token %.2f s after submit; `no room for a bookmark` lines %d; reader waits (at most) %d' % (
            ((d.watch[r].get('first') or time.time()) - t0), text.count('no room for a bookmark'), s2['split_reader_waits_total']))
        ev0, ev1 = t0, time.time()
    mk = locals().get('mark')
    if a.sessions > 0 and mk is not None and a.event not in ('qs', 'qi', 'qw'):
        # a decoder whose line was given back to the pool in the row is not a clean gap measurement (cmd_gap's rule): counted from the log, as cmd_gap does, with the
        # decoders' session ids (Decoders submits 0x73706400 + k)
        given_back = gave_back_lines(log.text(mk), {0x73706400 + k for k in range(a.sessions)})
        if given_back:
            print('  decoder lines given back to the pool during the row: %d' % given_back)
    if mk is not None:
        row_counts(log.text(mk), a.event)
    g = d.longest(ev0, ev1)
    pieces = re.findall(r'^serve: req \d+ .*read in pieces.*$|^serve: req \d+ .*row pieces \(longest.*$', log.text(mk), re.M)[-1:] if mk is not None else []
    if a.event == 'g':
        pass  # judged inside rides_phase and above
    elif a.event in ('back', 'exchange', 'ssd'):
        # a return beside decoders is the request's own path, placed by the interleaver: its neighbours' longest gap that BEGAN in the return's window is judged against the
        # engine's own setting (--interleave-gap-ms), printed beside the control's (the bound "control + 250 ms" was a line for a join sent whole)
        check(g * 1e3 <= a.interleave_gap_ms, '%s (%.1f s): the neighbours\' longest gap %.0f ms at most --interleave-gap-ms %.0f (control window %.0f ms)' % (a.event, ev1 - ev0, g * 1e3, a.interleave_gap_ms, ctrl * 1e3))
    elif a.event in ('a2n', 'qi', 'qw', 'qd', 'qs'):
        # the window holds a long prompt's pieces (that gap is not this row's): printed with the prompt's own `read in pieces` line, never judged
        print('  %s (%.1f s): longest gap %.0f ms (not judged: the window holds a long prompt\'s pieces); its line: %s' % (a.event, ev1 - ev0, g * 1e3, pieces[0][:200] if pieces else 'none'))
    else:
        judged, ok, what = gap_verdict(g, ctrl, a.gap_slack_ms, given_back, sum(1 for t in d.ended.values() if t < ev0), len(d.last),
                                       sum(1 for ts, _ in d.gaps if ev0 <= ts <= ev1))
        if judged:
            check(ok, '%s (%.1f s): %s' % (a.event, ev1 - ev0, what))
        else:
            print('  %s (%.1f s): %s' % (a.event, ev1 - ev0, what))
    d.finish()
    return 0


DROPPED = re.compile(r'serve: session ([0-9a-f]{16}) dropped from L2: (\d+) numbers back')
SPLIT_LINE = re.compile(r'serve: session ([0-9a-f]{16}) split: ')
JOINED = re.compile(r'serve: session ([0-9a-f]{16}) joined to position')
REQ_LINE = r'serve: req \d+ session %016x: prompt (\d+), reused (\d+) \((\w+)\)'


def tiers_stats_line(label, s):
    return '  %s: pool free %d reserved %d of %d; split_level %d, pressure %d pm, age asked %d s, %d pieces/s, L2 %d numbers, %d dropped, %d pieces split' % (
        label, s['pool_free'], s['pool_reserved'], s['pool_numbers'], s['split_level'], s['split_pressure_pm'], s['split_age_now_s'],
        s['split_demote_pps'], s['l2_numbers'], s['l2_dropped_total'], s['split_pieces_total'])


def cmd_tiers(a):
    """Row L, `--kv-split tiered` (needs --log): the two factors and the order. The caller's start sets the marks so that A and B (--tier-size
    each) fit above the high mark, C lands between the marks and the fillers reach under the low mark; --after-s is the start's
    --kv-split-after-s (20). The arm prints the pool and the level at each step."""
    T = list(load_npy_i32(a.tokens))
    log = Log(a.log)
    e = Engine(a.socket)
    d = Decoders(e, T, 0, a.decode_new)
    base = 0x73706c40
    A, B, C = base, base + 1, base + 2
    hist = {}

    def make(sid, off):
        o = make_conversation(e, T, sid, a.tier_size, off)
        hist[sid] = list(tokens_of(T, a.tier_size, off)) + list(o['ids'])

    def turn(sid):
        mk = log.mark()
        o = e.run(sid, hist[sid] + tokens_of(T, 16, 777), max_new=1, keep=True, stop=())
        hist[sid] = hist[sid] + tokens_of(T, 16, 777) + list(o['ids'])
        time.sleep(1.0)
        return log.text(mk)

    s0 = d.poll_stats()
    need(s0 is not None and s0['kv_split'] == 1, 'STATS: kv_split 1 (the engine was started with --kv-split tiered)')
    print(tiers_stats_line('at the start', s0))
    mk0 = log.mark()
    make(A, 1000)
    make(B, 2000)
    d.pump(time.time() + a.after_s + 5)  # A and B idle past the base age
    s1 = d.poll_stats()
    print(tiers_stats_line('A and B idle %d s' % (a.after_s + 5), s1))
    need(s1['split_level'] == 0, 'the pool is above the high mark with A and B (split_level 0)')
    check(s1['split_pieces_total'] == s0['split_pieces_total'] and s1['conv_split'] == 0, 'nothing is split above the high mark, whatever the age (pieces %d, conversations %d)' % (
        s1['split_pieces_total'] - s0['split_pieces_total'], s1['conv_split']))
    make(C, 3000)
    t_end = time.time() + 90
    s2 = d.poll_stats()
    while s2['split_pieces_total'] <= s1['split_pieces_total'] and time.time() < t_end:
        d.pump(time.time() + 1.0)
        s2 = d.poll_stats()
    print(tiers_stats_line('after C', s2))
    need(s2['split_pieces_total'] > s1['split_pieces_total'], 'a conversation was demoted after C: %d pieces' % (s2['split_pieces_total'] - s1['split_pieces_total']))
    check(s2['split_level'] == 1 and s2['split_pressure_pm'] > 0 and s2['split_age_now_s'] < a.after_s, 'level 1 between the marks: pressure %d pm above 0, age asked %d s under %d' % (
        s2['split_pressure_pm'], s2['split_age_now_s'], a.after_s))
    check(s2['split_demote_pps'] > 0, 'pieces demoted a second above 0: %d' % s2['split_demote_pps'])
    named = {m.group(1) for m in SPLIT_LINE.finditer(log.text(mk0))}
    check(named == {'%016x' % A}, 'every split: line names A only: %s' % sorted(named))
    # C is whole: its next turn needs no join; A's does
    tc = turn(C)
    mc = re.search(REQ_LINE % C, tc)
    check(mc is not None and mc.group(3) == 'end' and not JOINED.search(tc), 'C\'s next turn: reused %s (%s), no joined line' % (mc.group(2) if mc else '?', mc.group(3) if mc else 'no req line'))
    ta = turn(A)
    ma = re.search(REQ_LINE % A, ta)
    check(any(m.group(1) == '%016x' % A for m in JOINED.finditer(ta)) and ma is not None and ma.group(3) == 'end',
          'A\'s next turn: a joined to position line and reused %s (%s)' % (ma.group(2) if ma else '?', ma.group(3) if ma else 'no req line'))
    # fillers until under the low mark: L2 is dropped first
    dropped0 = d.poll_stats()['l2_dropped_total']
    seg, k = '', 0
    s3 = d.poll_stats()
    while s3['split_level'] != 2 and k < a.tier_fillers_max:
        mk = log.mark()
        make(base + 10 + k, 5000 + 1000 * k)
        k += 1
        d.pump(time.time() + 2.0)
        s3 = d.poll_stats()
        seg = log.text(mk)
    print(tiers_stats_line('after %d fillers' % k, s3))
    need(s3['split_level'] == 2 or s3['l2_dropped_total'] > dropped0, 'the pool reached the low mark (split_level %d, %d dropped)' % (s3['split_level'], s3['l2_dropped_total'] - dropped0))
    ev = [(m.start(), 'drop', m.group(1)) for m in DROPPED.finditer(seg)] + [(m.start(), 'split', m.group(1)) for m in SPLIT_LINE.finditer(seg)]
    ev.sort()
    check(bool(ev) and ev[0][1] == 'drop', 'the first line under the low mark is a drop from L2, before any split: %s' % [x[1] for x in ev[:3]])
    check(s3['l2_dropped_total'] > dropped0, 'l2_dropped_total rose: %d' % (s3['l2_dropped_total'] - dropped0))
    victim = next((int(x[2], 16) for x in ev if x[1] == 'drop'), None)
    if victim in hist:
        tv = turn(victim)
        mb = list(re.finditer(back_re(victim), tv, re.M))
        check(bool(mb) and int(mb[-1].group(4)) == int(mb[-1].group(5)) and int(mb[-1].group(4)) > 0, 'the dropped conversation\'s next turn reads back: %s' % (
            'restart point %s of %s' % (mb[-1].group(4), mb[-1].group(5)) if mb else 'no read-back line'))
    check(bug_lines(log.text(0)) == 0, 'lines BUG of any kind in the log: %d' % bug_lines(log.text(0)))
    d.finish()
    return 0


def cmd_tokens(a):
    """the added row: the decoders' tokens per second while a conversation's backlog is split at the pace, against the same window without it."""
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    d = Decoders(e, T, a.sessions, a.decode_new)
    d.window(20)
    need_cache(d.poll_stats())
    t = time.time()
    d.window(a.window)
    base = d.tok_per_s(t, time.time())
    print('without the backlog: %.1f tokens/s over %.0f s (%d sessions)' % (base, a.window, a.sessions))
    sid = 0x73706c00 + a.size // 1000
    s1 = d.poll_stats()
    rid = d.submit_watch(sid, tokens_of(T, a.size, 0), 1)
    d.pump(time.time() + 3600, lambda: 'done' in d.watch[rid])
    t = time.time()
    d.window(a.window)  # the backlog is split in this window (500 ms apart, 64 pieces a command)
    s2 = d.poll_stats()
    with_ = d.tok_per_s(t, time.time())
    need(s2['split_pieces_total'] > s1['split_pieces_total'], 'the window held the split: %d pieces' % (s2['split_pieces_total'] - s1['split_pieces_total']))
    print('with the backlog splitting: %.1f tokens/s' % with_)
    check(with_ >= 0.9 * base, 'the decoders lost %.1f %% of their tokens per second (the line is 10 %%: over it the pace is raised)' % (100 * (1 - with_ / max(base, 1e-9))))
    d.finish()
    return 0


def cmd_equal(a):
    """8.5's reference: the same turns on an off server and a cache server, compared (tokens, logprobs, cached, restart point)."""
    if a.compare:
        x, y = [json.load(open(p)) for p in a.compare]
        bad = 0
        for k in sorted(set(x) | set(y)):
            same = k in x and k in y and x[k]['ids'] == y[k]['ids'] and x[k]['cached'] == y[k]['cached'] and x[k]['lp'] == y[k]['lp']
            print('  %s: %s' % ('ok' if same else 'DIFFER', k))
            bad += not same
        return 1 if bad else 0
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    out = {}

    def turn(name, session, prompt, **k):
        o = e.run(session, prompt, max_new=24, stop=(), n_top=None, **k)
        out[name] = {'ids': list(o['ids']), 'cached': o['cached'], 'lp': list(o['lp'])}
        print('  %s: prompt %d, cached %d' % (name, o['prompt'], o['cached']))
        return o

    def idle(seconds):
        time.sleep(seconds)  # the engine splits between requests; on an off server nothing happens

    for size in (16384, 131072, a.size):
        sid = 0x73706c10 + size // 1024
        p = tokens_of(T, size, size)
        o = turn('%d/1' % size, sid, p)
        h = p + list(o['ids'])
        idle(2)
        o = turn('%d/2 return after 2 s' % size, sid, h + tokens_of(T, 16, 5))
        h = h + tokens_of(T, 16, 5) + list(o['ids'])
        filler = 0x73706c80 + size // 1024
        turn('%d/filler' % size, filler, tokens_of(T, 20000, 99))
        idle(2)
        o = turn('%d/3 after a filler' % size, sid, h + tokens_of(T, 16, 6))
        turn('%d/4 regenerate' % size, sid, h + tokens_of(T, 16, 6))
        turn('%d/5 fork of the base' % size, sid + 0x1000, h + tokens_of(T, 32, 7), base=sid)
    if a.out:
        json.dump(out, open(a.out, 'w'))
    return 0


def run_together(e, jobs, new):
    """jobs: [(name, session, prompt)], all submitted before any frame is read, so that their decoding overlaps. Returns {name: ids / cached /
    lp / t_first / t_done}; one connection, frames dispatched by request id."""
    reqs = {}
    for name, session, prompt in jobs:
        rid = e.submit(session, prompt, max_new=new, stop=(), n_top=1, keep=True)[0]
        reqs[rid] = name
    res = {n: {'ids': [], 'lp': [], 'cached': 0, 't_first': None, 't_done': None} for n in reqs.values()}
    left = set(reqs)
    while left:
        t, p = e.frame()
        if t == TOKENS:
            r, n, fl = struct.unpack('<QII', p[:16])
            o = res[reqs[r]]
            o['ids'] += struct.unpack('<%dI' % n, p[16:16 + 4 * n])
            if fl & 1:
                o['lp'] += struct.unpack('<%df' % n, p[16 + 4 * n:16 + 8 * n])
            if o['t_first'] is None:
                o['t_first'] = time.time()
        elif t == DONE:
            r, reason, pt, cached = struct.unpack('<QIII', p[:20])
            res[reqs[r]].update(cached=cached, t_done=time.time())
            left.discard(r)
    return res


def cmd_long(a):
    """P-scan's row: N long conversations (different prompts), then their return turns together; see the header."""
    if a.compare:
        x, y = [json.load(open(p)) for p in a.compare]
        bad = 0
        for k in sorted(set(x) | set(y)):
            same = k in x and k in y and x[k]['ids'] == y[k]['ids'] and x[k]['cached'] == y[k]['cached'] and x[k]['lp'] == y[k]['lp']
            print('  %s: %s' % ('ok' if same else 'DIFFER', k))
            bad += not same
        print('%d turns compared, %d differ' % (len(set(x) | set(y)), bad))
        return 1 if bad else 0
    T = list(load_npy_i32(a.tokens))
    e = Engine(a.socket)
    out = {}
    sizes = [int(v) for v in a.sizes.split(',')]
    serial = 0
    for size in sizes:
        for n in a.sessions_list:
            sids = [0x73706c00000 + (size // 1000) * 1000 + serial * 16 + k for k in range(n)]
            serial += 1
            prompts = [tokens_of(T, size, 7919 * (serial * 8 + k) + size) for k in range(n)]
            t0 = time.time()
            first = run_together(e, [('%d/s%d/%d first' % (size, n, k), sids[k], prompts[k]) for k in range(n)], a.long_new)
            print('  size %d, %d session(s): first turns in %.0f s' % (size, n, time.time() - t0))
            hist = [prompts[k] + first['%d/s%d/%d first' % (size, n, k)]['ids'] for k in range(n)]
            ret = run_together(e, [('%d/s%d/%d return' % (size, n, k), sids[k], hist[k] + tokens_of(T, 16, 31 * k + 5)) for k in range(n)],
                               a.long_new)
            for res in (first, ret):
                for k, v in res.items():
                    out[k] = {'ids': list(v['ids']), 'cached': v['cached'], 'lp': list(v['lp'])}
                    print('  %s: %d tokens, cached %d' % (k, len(v['ids']), v['cached']))
            check(all(len(v['ids']) == a.long_new for v in ret.values()), 'the return turns decoded %d tokens each' % a.long_new)
            check(all(v['cached'] > 0 for v in ret.values()), 'the return turns read their cached contexts (cached > 0)')
            if n > 1:
                firsts = [v['t_first'] for v in ret.values()]
                dones = [v['t_done'] for v in ret.values()]
                need(max(firsts) < min(dones), 'the %d return turns decoded together (every first token before any last)' % n)
    if a.out:
        json.dump(out, open(a.out, 'w'))
    return 0


def cmd_forkrun(a):
    """The fork row: the fork decodes for seconds beside its idle, split base; its ids must equal the off twin's."""
    if a.compare:
        x, y = [json.load(open(p)) for p in a.compare]
        same = x['fork']['ids'] == y['fork']['ids'] and len(x['fork']['ids']) > 0
        print('  %s: the fork decoded %d / %d tokens' % ('ok' if same else 'DIFFER', len(x['fork']['ids']), len(y['fork']['ids'])))
        return 0 if same else 1
    T = list(load_npy_i32(a.tokens))
    size = a.size if a.size != 1000000 else 200000
    e = Engine(a.socket)
    d = Decoders(e, T, 0, 0)  # no decoding neighbours: the fork is the only request, the frames still go through one handler
    base_id, fork_id = 0x73706f01, 0x73706f02
    prompt = tokens_of(T, size, 777)
    o = e.run(base_id, prompt, max_new=1, keep=True, stop=())
    check(o['error'] == '', 'base %016x made: %d positions' % (base_id, o['prompt']))
    s0 = d.poll_stats()
    cache = s0 is not None and s0['kv_split'] == 1
    if cache:
        s0 = wait_split_quiet(d)
        need(s0['split_pieces_total'] > 0, 'the base was split (split_pieces_total %d)' % s0['split_pieces_total'])
    else:
        d.pump(time.time() + 3)
    shared = min(a.fork_shared, size)
    fprompt = prompt[:shared] + tokens_of(T, 64, 7)
    t0 = time.time()
    rid = d.submit_watch(fork_id, fprompt, a.fork_new, keep=False)
    d.pump(time.time() + 600, lambda: 'done' in d.watch[rid])
    w = d.watch[rid]
    need('done' in w, 'the fork finished')
    s1 = d.poll_stats()
    dur = w['done'] - (w.get('first') or t0)
    ids = list(w.get('ids', []))
    check(len(ids) > 0, 'the fork decoded %d tokens in %.1f s' % (len(ids), dur))
    need(dur >= 3.0, 'the fork decoded for at least 3 s (%.1f s): raise --fork-new' % dur)
    if cache and s1:
        print('  split_pieces_total %d -> %d, join_bytes_total %d -> %d (the base\'s split commands ran beside the fork)' % (
            s0['split_pieces_total'], s1['split_pieces_total'], s0.get('join_bytes_total', 0), s1.get('join_bytes_total', 0)))
    if a.log:
        text = open(a.log, errors='replace').read()
        inside = re.findall(r'state: session \d+: joined inside an operation', text)
        sums = re.findall(r'differ from their sums', text)
        check(not inside, '"joined inside an operation" lines: %d' % len(inside))
        check(not sums, '"differ from their sums" lines: %d' % len(sums))
    if a.out:
        json.dump({'fork': {'ids': ids, 'cached': 0, 'lp': []}}, open(a.out, 'w'))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=['load', 'gap', 'tokens', 'equal', 'nolog', 'forkrun', 'long', 'tiers', 'imager'], default='load')
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log')
    ap.add_argument('--size', type=int, default=1000000)
    ap.add_argument('--sessions', default='8', help='the decoders (load, gap, tokens); long: a comma list of session counts, e.g. 1,4')
    ap.add_argument('--sizes', default='1200000,2500000', help='long arm: prompt sizes, comma separated')
    ap.add_argument('--long-new', type=int, default=24, help='long arm: tokens each turn decodes')
    ap.add_argument('--decode-new', type=int, default=20000,
                    help='tokens each decoding session makes: they must run through the return (the ssd event with fillers: 70000 keeps them alive about 18 minutes beyond the start, E); the gap is not judged when they ended first')
    ap.add_argument('--window', type=float, default=60)
    ap.add_argument('--gap-slack-ms', type=float, default=250)
    ap.add_argument('--event', choices=['split', 'back', 'exchange', 'ssd', 'grow', 'a2n', 'g', 'b', 'q', 'ref', 'qi', 'qw', 'qd', 'qs', 'ext'])
    ap.add_argument('--interleave-burst-max-ms', dest='burst_max_ms', type=float, default=1000.0, help='the engine\'s --interleave-burst-max-ms (the exch row\'s allowance on a build without the owed-time rule)')
    ap.add_argument('--ext-grow-ahead', type=int, default=256, help='ext row: the engine\'s --grow-ahead (the sizing mirrors server/serve.cpp need_clamped)')
    ap.add_argument('--ext-pool-tokens', action='store_true', help='print the --pool-tokens the ext row needs and exit')
    ap.add_argument('--no-owed-time', action='store_true', help='exch on a build without the owed-time rule: allows one more run line')
    ap.add_argument('--ext-counts', metavar='LOG', help='apply row ext\'s journal checks (mode b) to a recorded log and exit: 1 on a FAIL; the four requests are those of the last four `factor` lines before the stall line')
    ap.add_argument('--ext-mode', choices=['a', 'b'], default='b', help='ext row: a (the test cancels one request) or b (the engine ends the longest reply by itself)')
    ap.add_argument('--ref', help='imager arm: the reference start\'s --out file (the return\'s ids must equal it)')
    ap.add_argument('--interleave-gap-ms', type=float, default=600, help='imager back/exchange rows: the engine\'s own setting (--interleave-gap-ms) a return\'s neighbour gap is judged against')
    ap.add_argument('--filler', type=int, default=131072)
    ap.add_argument('--fill', type=int, default=4)
    ap.add_argument('--ssd-fillers', type=int, default=1, help='ssd event: conversations of --size made after the first, so that it leaves the pool')
    ap.add_argument('--tier-size', type=int, default=131072, help='tiers arm: positions of each conversation (A, B, C and the fillers)')
    ap.add_argument('--after-s', type=float, default=20, help='tiers arm: the start\'s --kv-split-after-s')
    ap.add_argument('--tier-fillers-max', type=int, default=40, help='tiers arm: fillers made at most, until the pool is under the low mark')
    ap.add_argument('--need-split-bookmarks', action='store_true', help='ssd event: the window bookmarks must be split too (slots >= lists + bookmarks), else NOT REACHED')
    ap.add_argument('--ssd-fillers-max', type=int, default=0, help='ssd event: more fillers, up to this many, while no conversation is only on the disk tier (0: exactly --ssd-fillers)')
    ap.add_argument('--ttft-max', type=float, default=30, help='ssd event: the return\'s time to its first token, at most (s)')
    ap.add_argument('--cache', action='store_true', help='grow arm: the engine runs --kv-split cache (without: the off run to compare with)')
    ap.add_argument('--out')
    ap.add_argument('--fork-shared', type=int, default=150000)
    ap.add_argument('--fork-new', type=int, default=400)
    ap.add_argument('--compare', nargs=2)
    a = ap.parse_args()
    if a.ext_pool_tokens:
        tk, tg = ext_pool_tokens()
        print('--pool-tokens %d (%d numbers)' % (tk, tg))
        sys.exit(0)
    if a.ext_counts:
        txt = open(a.ext_counts, errors='replace').read()
        sm = STALL_LINE.search(txt)
        four = [int(r) for r in REQ_START.findall(txt[:sm.start() if sm else len(txt)])[-4:]]
        ext_counts(txt, 'b', four)
        print('%d check(s) failed' % fails if fails else 'all checks passed')
        sys.exit(1 if fails else 0)
    a.sessions_list = [int(v) for v in str(a.sessions).split(',')]
    a.sessions = a.sessions_list[0]
    try:
        rc = {'load': cmd_load, 'gap': cmd_gap, 'tokens': cmd_tokens, 'equal': cmd_equal, 'nolog': cmd_nolog, 'forkrun': cmd_forkrun, 'long': cmd_long, 'tiers': cmd_tiers, 'imager': cmd_imager}[a.arm](a)
    except NotReached as x:
        print('did not reach the path: %s' % x)
        return 2
    except DecodersStopped as x:
        print('  FAIL: %s' % x)
        return 1
    print('%d check(s) failed' % fails if fails else 'all checks passed')
    return rc or (1 if fails else 0)


if __name__ == '__main__':
    sys.exit(main())
