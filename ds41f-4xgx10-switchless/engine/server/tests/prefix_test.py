#!/usr/bin/env python3
"""A request that carries only its prompt's suffix (kFlagPrefix, docs/protocol.md "A request with only its suffix"), through the
engine's own socket, without the front-end: the engine builds the prompt from its copy of the conversation's history, the epoch
and the count the request names say whether that copy is the one the client knows.

  python3 engine/server/tests/prefix_test.py --socket S --tokens tokens.npy main            # the three sizes, with and without the flag
  python3 engine/server/tests/prefix_test.py --socket S --tokens tokens.npy rewind          # an edit inside the history, its undo, a stale epoch
  python3 engine/server/tests/prefix_test.py --socket S --tokens tokens.npy leaves          # a client that leaves mid-reply, then returns
  python3 engine/server/tests/prefix_test.py --socket S --tokens tokens.npy evict           # a conversation released between turns
  python3 engine/server/tests/prefix_test.py --socket S --tokens tokens.npy ids --log-cmd 'ssh HOST cat RANK0_LOG'            # a conversation whose ids are in their file
  python3 engine/server/tests/prefix_test.py --socket S --tokens tokens.npy restart --restart-cmd 'ssh HOST COMMAND_THAT_RESTARTS_THE_ENGINE'
  python3 engine/server/tests/prefix_test.py --socket S --tokens tokens.npy all --restart-cmd ...   # every arm in turn (ids and restart only with their flags)

Every arm compares replies bit for bit (greedy; ids and first logprob) and says what it proves in its header line. A comparison that
fails prints what differed: the first 8 ids of each reply, the first logprob of each, and both `cached` counts.

Which reference, and why. A cold prefill does not reproduce the roundings of positions that were DECODED (docs/engine.md, "Exact reuse":
the engine's restarts from stored state are not bit-equal to a cold read of the same prompt; they are equal to another restart from the
same point). So:
  - A TWIN is a second conversation brought to the same state by the same turns, whose next request is the whole prompt without
    kFlagPrefix (the main arm's "without"). It restarts from the same stored state, so ids, first logprobs AND `cached` must be equal.
    Every comparison of a prefix request that restarts from a history holding decoded positions uses a twin.
  - A COLD run (the same prompt whole in a fresh, unkept session) is used only where the restart point is below every decoded
    position (an edit at 3000 of a history whose decoded positions start at 6000) or where nothing is stored (a released conversation);
    the check's text says which.
The engine's settings each arm needs:

  main     any engine; max_ctx >= the largest --sizes entry plus 2,000 (default sizes 16384, 131072, 1000000: the engine's window of
           1,048,576, so tokens.npy holds at least 1,010,000 ids). Proves: the replies and the `cached` counts of a prefix request equal
           the whole-prompt request's, and its frame is the suffix's size (72 + 16 + 4 x (stops + suffix) bytes), at each size.
  rewind   any engine. Proves: n_prefix below the history's end builds (an edit inside the history is a cut of the entry: the ids below
           it are still true), equals a twin's whole-prompt request and a cold run (the cut is below the decoded positions); the
           epoch then changes, so the epoch of the DONE before is kPrefixGone, and the new DONE's epoch builds a next prompt that
           restarts from decoded positions: reference = a twin (cold would differ in roundings); a second cut to 3000: twin and cold.
  leaves   any engine. Proves: after a client leaves mid-reply (the connection closes) the next connection's prefix request is either
           built (the entry's epoch survived the cancelled run) or kPrefixGone followed by a whole resend. Reference: a twin brought
           through the same turns and cancelled (CANCEL frame after 3 ids) at the same point; the two cancelled DONEs' n_hist must be
           equal (else the arm says it is inconclusive and fails: run again), then ids, first logprob and cached equal. A second
           part closes the connection (the real "client leaves"); the point of the cut is not observable there, so it checks only
           that the returning request succeeds.
  evict    any engine. Proves: a released conversation is kPrefixGone (no such entry), a whole resend works and equals cold (reference
           = cold: nothing is stored for it, so the resend is a cold read), and its new DONE teaches the new epoch.
  ids      an engine started with --ids-host-mib 32 and --disk-gib > 0 (the cap is 32 MiB of ids = 8M positions; this arm makes
           conversations of --ids-conv positions until the sum passes it). Proves: a conversation whose ids went to their file
           (STATS ids_file_sessions > 0 for it) builds a prefix request after the file is read back, and the reply equals a twin's (same
           turns, next request whole, no flag; the history holds decoded positions, so cold is not the reference).
  Both of those two arms also check the engine's log line "its ids read back from their file" for the conversation (--log-cmd, e.g.
  'ssh HOST cat RANK0_LOG'): the ids' path through their file, which is also the GPU proof of the ids file.
  restart  an engine under a planned-stop service with kept sessions (--keep-gib > 0, --disk-gib > 0) and --restart-cmd, a command that
           stops it (SIGTERM, so keep() runs) and starts it again, returning when the socket is back. Proves: the kept index carries
           each entry's epoch and size, so the DONE's epoch of before the restart builds after it (no ids in the index, the ids
           file named), and the reply equals a twin's: the twin went through the same turns and the same planned stop, so both restart from
           their kept state (reference = twin; cached must be equal).
"""
import argparse
import shlex
import struct
import subprocess
import sys
import time

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from serve_test import Engine, load_npy_i32  # noqa: E402

V = 5  # the protocol version with kFlagPrefix and the DoneTail (fe::kVersionPrefix)

FAILS = 0
STOP = (1,)
READ_BACK = 'its ids read back from their file'  # the engine's line at every read-back of an ids file


def engine_log(a):
    """The engine's log text, from --log-cmd (a command printing it)."""
    if not a.log_cmd:
        return None
    return subprocess.run(shlex.split(a.log_cmd), check=True, capture_output=True, text=True).stdout


def read_backs(a, session):
    """How many times the engine's log says session's ids were read back from their file (None: no --log-cmd)."""
    t = engine_log(a)
    return None if t is None else sum(1 for ln in t.splitlines() if READ_BACK in ln and '%016x' % session in ln)


def check(ok, what):
    global FAILS
    print('%s %s' % ('ok  ' if ok else 'FAIL', what))
    sys.stdout.flush()
    if not ok:
        FAILS += 1
    return ok


def same(a, b):
    """Bit for bit: the same ids and the same logprobs."""
    return a['ids'] == b['ids'] and a['lp'] == b['lp']


def diff_text(a, b):
    """What differed between two replies."""
    lp = lambda r: ('%.6f' % r['lp'][0]) if r['lp'] else '-'  # noqa: E731
    return 'first 8 ids %s vs %s; first logprob %s vs %s; cached %s vs %s' % (a['ids'][:8], b['ids'][:8], lp(a), lp(b), a['cached'], b['cached'])


def cmp_check(what, a, b, cached=False):
    """The replies equal bit for bit (and, with cached, the reused counts); on FAIL says what differed."""
    ok = same(a, b) and (not cached or a['cached'] == b['cached'])
    check(ok, what)
    if not ok:
        print('     differs: ' + diff_text(a, b))
    return ok


def frame_bytes(n_stop, n_prompt, prefix):
    return 72 + (16 if prefix else 0) + 4 * (n_stop + n_prompt)


class Conv:
    """One conversation the way a front-end holds it: its whole prompt so far, and the epoch and size of its last DONE."""

    def __init__(self, e, session, T, off=0):
        self.e, self.session, self.T, self.off = e, session, T, off
        self.prompt = []   # the ids the engine's history begins with, as far as the client knows: prompt + reply
        self.epoch = 0
        self.n_hist = 0

    def turn(self, new_ids, use_prefix, max_new=16, cancel_after=None, keep=True):
        """The next request: the prompt so far, the reply's ids, then new_ids. With use_prefix and an epoch: only the suffix."""
        prompt = self.prompt + list(new_ids)
        if use_prefix and self.epoch and self.n_hist:
            n = min(self.n_hist, len(self.prompt))
            r = self.e.run(self.session, prompt[n:], max_new=max_new, n_top=2, keep=keep, prefix=(self.epoch, n),
                           cancel_after=cancel_after)
            r['frame'] = self.e.last_frame_bytes
            r['n_prefix'] = n
            if r['reason'] == 'prefix_gone':   # the front-end's fallback: the whole prompt
                r2 = self.e.run(self.session, prompt, max_new=max_new, n_top=2, keep=keep, cancel_after=cancel_after)
                r2['frame'], r2['gone_first'] = self.e.last_frame_bytes, True
                r = r2
        else:
            r = self.e.run(self.session, prompt, max_new=max_new, n_top=2, keep=keep, cancel_after=cancel_after)
            r['frame'] = self.e.last_frame_bytes
        r['prompt_ids'] = prompt
        if r['has_tail'] and r['reason'] in ('stop', 'length'):
            self.prompt, self.epoch, self.n_hist = prompt + list(r['ids']), r['epoch'], r['n_hist']
        return r


def cold(e, session, prompt, max_new=16):
    """The same prompt run whole in a fresh session that is not kept."""
    return e.run(session, prompt, max_new=max_new, n_top=2, keep=False)


def arm_main(e, T, a):
    print('== main: 3 turns at each size, with and without kFlagPrefix: replies bit-equal, cached equal, the frame falls to the suffix')
    base = 0x5a00000000000000
    for si, size in enumerate(a.sizes):
        step = 400
        first = size - 2 * (step + 20) - 10
        if len(T) < size + 10:
            check(False, 'tokens.npy holds %d ids, %d needed' % (len(T), size + 10))
            continue
        whole, pfx = Conv(e, base + 2 * si, T), Conv(e, base + 2 * si + 1, T)
        news = [T[:first]] + [T[first + k * step:first + (k + 1) * step] for k in range(2)]
        for t in range(3):
            ra = whole.turn(news[t], False)
            rb = pfx.turn(news[t], True)
            print('  size %-8d turn %d: whole %6.1f ms, frame %9d B, cached %7d | prefix %6.1f ms, frame %9d B, cached %7d, n_prefix %s' % (
                size, t + 1, ra['prefill_us'] / 1e3, ra['frame'], ra['cached'], rb['prefill_us'] / 1e3, rb['frame'],
                rb['cached'], rb.get('n_prefix', '-')))
            check(ra['reason'] in ('stop', 'length') and rb['reason'] in ('stop', 'length'), 'size %d turn %d: both ended normally' % (size, t + 1))
            cmp_check('size %d turn %d: replies bit-equal (%d ids), cached equal (twin: the same turns, whole prompt)' % (size, t + 1, len(ra['ids'])), ra, rb, True)
            check(ra['prompt'] == rb['prompt'], 'size %d turn %d: the prompt %d equal' % (size, t + 1, ra['prompt']))
            if t > 0:
                n_suffix = rb['frame'] // 4 - (72 + 16) // 4 - len(STOP)
                check(rb['frame'] < ra['frame'] and rb['frame'] == frame_bytes(len(STOP), n_suffix, True) and n_suffix < 2 * (step + 20),
                      'size %d turn %d: the frame fell from %d to %d bytes, the suffix (%d ids)' % (size, t + 1, ra['frame'], rb['frame'], n_suffix))
            check(rb['has_tail'] and ra['has_tail'] and rb['n_hist'] == ra['n_hist'], 'size %d turn %d: DONE tells the same size (%d)' % (
                size, t + 1, rb['n_hist']))
        e.release(whole.session)
        e.release(pfx.session)


def arm_rewind(e, T, a):
    print('== rewind: an edit inside the history, a stale epoch, its undo (twin = same turns, whole prompt)')
    S = 0x5a10000000000000
    c, tw = Conv(e, S + 1, T), Conv(e, S + 5, T)
    c.turn(T[:6000], True)
    tw.turn(T[:6000], False)   # the twin: the same first turn (6,000 prefilled ids, then decoded positions)
    e1, n1 = c.epoch, c.n_hist
    # an edit inside the history: the prompt keeps its first 3000 ids and goes on with other text
    edited = c.prompt[:3000] + T[9000:9500]
    pre = e.run(S + 1, edited[3000:], max_new=16, n_top=2, prefix=(e1, 3000))
    check(pre['reason'] in ('stop', 'length') and pre['has_tail'], 'n_prefix 3000 below the history (%d): built, the epoch of the DONE before' % n1)
    ref = cold(e, S + 2, edited)
    cmp_check('the edited prompt equals the same prompt cold (%d ids; cold is the reference here: the restart point is at most 3000, below every decoded position, which start at 6000)' % len(edited), pre, ref)
    twe = e.run(S + 5, edited, max_new=16, n_top=2)
    cmp_check('and equals the twin\'s whole-prompt request (ids, first logprob, cached)', pre, twe, True)
    check(pre['epoch'] != e1 and pre['epoch'] != 0, 'the cut below the last DONE\'s size gave the entry a new epoch')
    old = e.run(S + 1, edited[3000:], max_new=16, n_top=2, prefix=(e1, 3000))
    check(old['reason'] == 'prefix_gone' and not old['has_tail'], 'the epoch of the DONE before the edit is kPrefixGone, no DoneTail (%s)' % old['error'])
    # the new DONE's epoch builds; this prompt restarts from a history that holds the decoded reply of `pre`: the twin is the reference
    new_prompt = edited + pre['ids'] + T[9500:9600]
    nxt = e.run(S + 1, new_prompt[pre['n_hist']:], max_new=16, n_top=2, prefix=(pre['epoch'], pre['n_hist']))
    twn = e.run(S + 5, edited + twe['ids'] + T[9500:9600], max_new=16, n_top=2)
    check(nxt['reason'] in ('stop', 'length'), 'the new DONE\'s epoch builds the next prompt (%s)' % nxt['reason'])
    cmp_check('equal to the twin (reference = twin: the restart is from decoded positions, a cold read would differ in roundings)', nxt, twn, True)
    undo = c.prompt[:3000] + T[:200]
    u = e.run(S + 1, undo[3000:], max_new=16, n_top=2, prefix=(nxt['epoch'], 3000)) if nxt['has_tail'] else None
    refu = cold(e, S + 4, undo)
    twu = e.run(S + 5, undo, max_new=16, n_top=2)
    check(u is not None and u['reason'] in ('stop', 'length'), 'a second cut to 3000 builds')
    if u is not None:
        cmp_check('and equals cold (the ids below 3000 are true; cold is the reference: the restart point is below every decoded position)', u, refu)
        cmp_check('and equals the twin (ids, first logprob, cached)', u, twu, True)
    for k in (1, 2, 4, 5):
        e.release(S + k)


def arm_leaves(e, T, a):
    print('== leaves: a client that leaves mid-reply, then returns (twin = same turns, cancelled at the same point, whole prompt)')
    S = 0x5a20000000000000
    c, tw = Conv(e, S + 1, T), Conv(e, S + 3, T)
    c.turn(T[:4000], True)
    tw.turn(T[:4000], False)
    epoch, n_hist = c.epoch, c.n_hist
    second = T[7000:7300]
    # part A: both conversations get a second request that is cancelled (a CANCEL frame after 3 ids); the DONE says where the history ended
    ca = e.run(S + 1, c.prompt[n_hist:] + second, max_new=200, n_top=2, prefix=(epoch, n_hist), cancel_after=3)
    cb = e.run(S + 3, tw.prompt + second, max_new=200, n_top=2, cancel_after=3)
    check(ca['reason'] == 'cancelled' and cb['reason'] == 'cancelled', 'both second requests were cancelled mid-reply (%s, %s)' % (ca['reason'], cb['reason']))
    same_point = ca['n_hist'] == cb['n_hist']
    check(same_point, 'the twin was cancelled at the same point (history ended at %s and %s ids)' % (ca['n_hist'], cb['n_hist']))
    prompt = c.prompt + second + T[8000:8100]
    r = e.run(S + 1, prompt[n_hist:], max_new=16, n_top=2, prefix=(epoch, n_hist))
    resent = False
    if r['reason'] == 'prefix_gone':
        resent = True
        r = e.run(S + 1, prompt, max_new=16, n_top=2)
        print('  the returning request was kPrefixGone (the cancelled run cut the entry); whole resend')
    else:
        print('  the returning request was built (epoch kept through the cancelled run)')
    rt = e.run(S + 3, tw.prompt + second + T[8000:8100], max_new=16, n_top=2)
    check(r['reason'] in ('stop', 'length'), 'the returning request succeeds (%s)' % ('after a whole resend' if resent else 'built'))
    if same_point:
        cmp_check('its reply equals the twin\'s (ids, first logprob, cached; reference = twin: the restart is from decoded positions)', r, rt, True)
    else:
        print('     inconclusive: the twin was cancelled at another point; run the arm again')
    e.release(S + 1)
    e.release(S + 3)
    # part B: the real leave: the connection closes after 3 ids. The point of the cut is not observable, so no reference: the
    # returning request must succeed, built or after a whole resend
    c2 = Conv(e, S + 4, T)
    c2.turn(T[:4000], True)
    epoch2, n2 = c2.epoch, c2.n_hist
    e.submit(S + 4, c2.prompt[n2:] + second, max_new=200, keep=True, prefix=(epoch2, n2))
    got = 0
    while got < 3:
        t, p = e.frame()
        if t == 0x82:
            got += struct.unpack('<I', p[8:12])[0]
    e.s.close()
    time.sleep(2.0)   # the engine cancels the run and keeps what it held
    e2 = Engine(a.socket, V)
    prompt2 = c2.prompt + second + T[8000:8100]
    r2 = e2.run(S + 4, prompt2[n2:], max_new=16, n_top=2, prefix=(epoch2, n2))
    if r2['reason'] == 'prefix_gone':
        r2 = e2.run(S + 4, prompt2, max_new=16, n_top=2)
    check(r2['reason'] in ('stop', 'length') and r2['ids'], 'after the connection closed mid-reply the returning request succeeds (no reference: the cut point is not observable)')
    e2.release(S + 4)
    return e2


def arm_evict(e, T, a):
    print('== evict: a conversation released between turns')
    S = 0x5a30000000000000
    c = Conv(e, S + 1, T)
    c.turn(T[:4000], True)
    epoch, n_hist = c.epoch, c.n_hist
    e.release(S + 1)
    time.sleep(1.0)
    prompt = c.prompt + T[7000:7200]
    r = e.run(S + 1, prompt[n_hist:], max_new=16, n_top=2, prefix=(epoch, n_hist))
    check(r['reason'] == 'prefix_gone' and not r['has_tail'], 'an unknown session is kPrefixGone (%s)' % r['error'])
    r = e.run(S + 1, prompt, max_new=16, n_top=2)
    ref = cold(e, S + 2, prompt)
    cmp_check('the whole resend is cold (nothing is stored for the released conversation; reference = cold) and equals cold', r, ref)
    check(r['cached'] == 0, 'its cached count is 0')
    check(r['has_tail'] and r['epoch'] not in (0, epoch), 'its DONE teaches a new epoch (%x, not %x)' % (r['epoch'], epoch))
    e.release(S + 1)
    e.release(S + 2)


def stats(e):
    e.send(0x05, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == 0x85:
            ver, n = struct.unpack('<II', p[:8])
            return struct.unpack('<%dQ' % n, p[8:8 + 8 * n])


def arm_ids(e, T, a):
    print('== ids: a conversation whose ids went to their file (engine --ids-host-mib 32)')
    S = 0x5a40000000000000
    first = Conv(e, S + 1, T)
    first.turn(T[:4000], True)
    twin = Conv(e, S + 2, T)
    twin.turn(T[:4000], False)   # the twin: the same first turn, so its next whole request restarts from the same stored state
    k = 0
    while stats(e)[49] == 0 and k < 40:   # ids_file_sessions: conversations whose ids are in a file now
        k += 1
        big = Conv(e, S + 100 + k, T)
        big.turn((T * 2)[:a.ids_conv], False, max_new=4)
        time.sleep(0.5)
    n_files = stats(e)[49]
    check(n_files > 0, 'the ids cap sent conversations\' ids to files (STATS ids_file_sessions %d after %d conversations of %d ids)' % (n_files, k, a.ids_conv))
    time.sleep(2.0)   # idle: ids_to_files runs between requests
    before = read_backs(a, S + 1)
    r = first.turn(T[7000:7200], True)
    after = read_backs(a, S + 1)
    check(after is not None and before is not None and after > before,
          'the engine\'s log has a read-back line for the conversation\'s ids file (%s -> %s; needs --log-cmd)' % (before, after))
    rt = twin.turn(T[7000:7200], False)
    check(r['reason'] in ('stop', 'length') and 'gone_first' not in r, 'the first conversation, its ids in a file or back, builds by prefix')
    cmp_check('and its reply equals the twin\'s (reference = twin: the restart is from decoded positions; ids, first logprob, cached)', r, rt, True)
    for j in range(1, k + 1):
        e.release(S + 100 + j)
    e.release(S + 1)
    e.release(S + 2)


def arm_restart(e, T, a):
    print('== restart: a planned stop between turns; the kept index carries the epoch')
    if not a.restart_cmd:
        check(False, 'the restart arm needs --restart-cmd')
        return e
    S = 0x5a50000000000000
    c = Conv(e, S + 1, T)
    c.turn(T[:6000], True)
    twin = Conv(e, S + 2, T)
    twin.turn(T[:6000], False)   # the twin goes through the same planned stop: both reuse their kept state after it
    epoch, n_hist = c.epoch, c.n_hist
    e.s.close()
    print('  restarting: %s' % a.restart_cmd)
    subprocess.run(shlex.split(a.restart_cmd), check=True)
    for _ in range(120):
        try:
            e = Engine(a.socket, V)
            break
        except OSError:
            time.sleep(5)
    prompt = c.prompt + T[7000:7300]
    before = read_backs(a, S + 1)
    r = e.run(S + 1, prompt[n_hist:], max_new=16, n_top=2, prefix=(epoch, n_hist))
    after = read_backs(a, S + 1)
    check(after is not None and before is not None and after > before,
          'the engine\'s log has a read-back line for the conversation\'s ids file after the restart (%s -> %s; needs --log-cmd)' % (before, after))
    check(r['reason'] in ('stop', 'length'), 'after the restart the epoch of the DONE before it builds the prompt (%s)' % r['reason'])
    rt = e.run(S + 2, twin.prompt + T[7000:7300], max_new=16, n_top=2)
    cmp_check('the reply equals the twin\'s, which restarted from its kept state too (reference = twin; ids, first logprob, cached)', r, rt, True)
    check(r['cached'] > 0, 'the kept state was reused (cached %d)' % r['cached'])
    e.release(S + 1)
    e.release(S + 2)
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arm', choices=['main', 'rewind', 'leaves', 'evict', 'ids', 'restart', 'all'])
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[16384, 131072, 1000000])
    ap.add_argument('--ids-conv', type=int, default=300000, help='positions of each conversation of the ids arm')
    ap.add_argument('--restart-cmd', default='')
    ap.add_argument('--log-cmd', default='', help='a command that prints the engine\'s log (the ids and restart arms check its read-back line)')
    a = ap.parse_args()
    T = load_npy_i32(a.tokens)
    e = Engine(a.socket, V)
    arms = {'main': arm_main, 'rewind': arm_rewind, 'leaves': arm_leaves, 'evict': arm_evict, 'ids': arm_ids, 'restart': arm_restart}
    order = ['main', 'rewind', 'leaves', 'evict', 'ids'] + (['restart'] if a.restart_cmd else []) if a.arm == 'all' else [a.arm]
    for name in order:
        r = arms[name](e, T, a)
        if isinstance(r, Engine):  # an arm that had to reconnect
            e = r
    print('prefix_test %s' % ('FAILED (%d)' % FAILS if FAILS else 'passed'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
