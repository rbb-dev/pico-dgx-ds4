#!/usr/bin/env python3
"""A full disk tier (docs/engine.md, Serving): when the file has no free space for an image, the least recently used conversation that
no command uses gives its images up (a conversation only on the disk is dropped whole: "dropped from the disk tier to make room"; one
whose pieces are in the pool loses its images alone). The sequence, through rank 0's socket (serve_test.py's client), on a pool so
small that two conversations do not fit and the smallest disk tier:
  1. Conversations of 60,000 positions, one after the other, each in a session of its own and kept. With two lines the third one's
     request takes the least recently used line; the first conversation's pieces then leave the pool piece by piece, as the next
     request's prefill needs numbers (store_core.cpp take_nums -> evict_step -> evict_piece, each image written to the file). The
     server's own spill at a request's end (serve.cpp evict) does not run for them: it skips a conversation with pieces already on the
     disk. So no "serve: spill session" line is to be expected; the images accumulate until the file is full, and the next image finds no
     free space (disk_take): the least recently used conversation is dropped whole. The run adds conversations until the log says
     "dropped from the disk tier to make room" (at most --conversations), and proves the file's fill from STATS (disk bytes in use
     against the cap) before it claims the drop.
  2. The log shows conversations "evicted to make room" (the dropped ones' ids) and STATS the file near full.
  3. A turn on a dropped conversation: the engine no longer holds it, so the turn is cold (`cached` 0) and equals the same prompt
     prefilled in a fresh session.
  4. A turn on a survivor, a conversation that was not dropped and no line holds (not one of the last two): its prompt diverges 8,000
     positions back; it comes back from the disk ("back from disk" in the log) and must equal the same prompt prefilled cold (CED: bit
     for bit). Its window bookmarks may have been dropped for want of room in this pool, so it may restart at 0; the read-back and the
     equality are what is proved.
  5. A fork of another survivor with the file still full: its own images need room beside the base's, and the base, whose command
     pins it, must not be the one that gives up its images; the fork equals cold, and the base's next turn comes back from the disk.
With --log the run proves each step from rank 0's log and STATS; a step that did not happen ends the run with exit 2 ("did not reach
the path"), never 0. Exit 0: every step ran and every turn equals cold. Exit 1: it ran and a turn differs.

The arithmetic (the engine's constants: 8,192 entries a number, a number 2,916,352 B, a window bookmark 2 numbers, an end bookmark 3;
a snapshot every --snap-every 4096 positions, thinned by serve.cpp keep_snap):
  --pool-tokens 250000 = 77 numbers (STATS says so), 214 MiB, the store's budget between requests 0.
  one conversation of 60,000 positions (60,024 with its response): lists 3 x 4 + 8 = 20 numbers (reservation 20); the cold prefill's
     bookmarks at 4,096 .. 57,344 = 14 and one at its prompt's end = 15 -> 30 numbers (a recorded run: several are refused for want of room,
     "no room for a bookmark of session N at 40960 / 45056 / 49152"); end bookmark 3: 20 + 30 + 3 = 53 numbers at the peak, within 77.
  what a conversation leaves on the disk is between its 20 lists (every bookmark refused) and its whole record of 47 images (20 lists +
     12 bookmarks + end 3): 20 x 2.78 MiB = 55.6 MiB at the least. --disk-gib 1 = 1,024 MiB = 18.4 conversations at 55.6 MiB, 7.8 at
     130.7 MiB: the default of 24 conversations passes the file in every case; the run stops at the first drop.
Engine settings (engine-test.sh takes them as one string):
  MAXCTX=100000 engine-test.sh BIN "--slots 2 --big-slots 0 --pool-tokens 250000 --disk-gib 1 --snap-every 4096" disk_full_test.py \\
      --log ~/dsgb10/logs/engine-disk_full_test.log

  python3 engine/server/tests/disk_full_test.py --socket S --tokens T.npy --log engine.log [--conversations 24]
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_proof import END_NUMBERS, SNAP_NUMBERS, Log, NotReached, back_re, need, reservation, snaps_kept, spill_re, stored_lists  # noqa: E402
from serve_test import Engine, load_npy_i32, same, show  # noqa: E402
from stats_test import stats  # noqa: E402

S0 = 0x6469736b66756c00  # 'diskful'
N = 60000
EVERY = 4096


def sid(k):
    return S0 + k


def offset(k):
    return (k * 17000) % 120000  # the token file holds 270,000: a window of 60,000 from here and 400 more past 150,000 are inside it


def turn(e, name, session, prompt, **kw):
    o = e.run(session, prompt, max_new=24, n_top=2, **kw)
    show(name, o)
    if o['error']:
        raise NotReached('%s: the engine answered with an error: %s' % (name, o['error']))
    return o


def run(a):
    T = [int(x) for x in load_npy_i32(a.tokens)]
    assert len(T) >= 200000, 'the token file is too short'
    e = Engine(a.socket)
    log = Log(a.log)
    print('engine: %s, context %d' % (e.name, e.max_ctx))

    s = stats(e)[2]
    total = s['pool_numbers']
    need(s['lines'] == 2, 'the engine runs with 2 lines (%d)' % s['lines'])
    need(0 < s['disk_cap_bytes'] <= 2**30, 'the disk tier is %.2f GiB (at most 1)' % (s['disk_cap_bytes'] / 2**30))
    need(total * s['number_bytes'] < 256 * 2**20, 'the pool is %d numbers = %.0f MiB, under 256 MiB: the store keeps nothing between requests'
         % (total, total * s['number_bytes'] / 2**20))
    peak = reservation(N + 24) + SNAP_NUMBERS * ((N - 1) // EVERY + 1) + END_NUMBERS
    need(peak <= total, 'a conversation of %d positions needs %d numbers at its peak, within the pool of %d' % (N, peak, total))
    kept = snaps_kept(N + 24, EVERY, N)
    record = stored_lists(N + 24) + SNAP_NUMBERS * len(kept) + END_NUMBERS
    per = record * s['number_bytes']
    least = stored_lists(N + 24) * s['number_bytes']
    need(least * a.conversations > s['disk_cap_bytes'],
         'a conversation leaves %d to %d images on the disk (%d lists; the whole record adds %d bookmarks of %d and an end of %d) = %.1f to %.1f MiB: '
         '%d of them pass the file (%.0f MiB) even at the least' % (
             stored_lists(N + 24), record, stored_lists(N + 24), len(kept), SNAP_NUMBERS, END_NUMBERS, least / 2**20, per / 2**20,
             a.conversations, s['disk_cap_bytes'] / 2**20))

    print('1. conversations of %d positions, until the log says one was dropped for room (at most %d)' % (N, a.conversations))
    prompts = {}
    hist = {}
    peak = 0
    made = 0
    for k in range(1, a.conversations + 1):
        prompts[k] = T[offset(k):offset(k) + N]
        o = turn(e, 'C%d cold' % k, sid(k), prompts[k])
        hist[k] = list(prompts[k]) + list(o['ids'])
        made = k
        s = stats(e)[2]
        peak = max(peak, s['disk_used_bytes'])
        print('  disk: %.0f of %.0f MiB in use' % (s['disk_used_bytes'] / 2**20, s['disk_cap_bytes'] / 2**20))
        if re.search(r'state: session \d+ dropped from the disk tier to make room', log.text()):
            break

    print('2. the file filled: conversations were dropped for room')
    need(peak * 10 >= s['disk_cap_bytes'] * 9, 'the file held %.0f MiB of its %.0f MiB at the most: the next image found no free space' % (
        peak / 2**20, s['disk_cap_bytes'] / 2**20))
    need(bool(re.search(r'state: session \d+ dropped from the disk tier to make room', log.text())),
         'the log says "dropped from the disk tier to make room" (after %d conversations)' % made)
    dropped = [int(x, 16) for x in re.findall(r'serve: session ([0-9a-f]{16}) evicted to make room', log.text())]
    dropped = [d for d in dict.fromkeys(dropped) if S0 < d <= S0 + made]
    need(len(dropped) >= 1, '%d conversation(s) dropped: %s' % (len(dropped), ', '.join('C%d' % (d - S0) for d in dropped)))
    survivors = [sid(k) for k in range(1, made - 1) if sid(k) not in dropped]  # the last two hold the lines
    need(len(survivors) >= 2, '%d survivors no line holds: %s' % (len(survivors), ', '.join('C%d' % (d - S0) for d in survivors)))

    print('3. a turn on a dropped conversation: cold')
    kd = dropped[0] - S0
    p = hist[kd] + T[150000:150300]
    d1 = turn(e, 'C%d (dropped) +300' % kd, sid(kd), p)
    need(d1['cached'] == 0, 'the dropped conversation C%d starts cold (cached %d)' % (kd, d1['cached']))
    c1 = turn(e, 'same prompt cold', sid(200), p, keep=False)
    ok = [same('dropped conversation vs cold', d1, c1)]

    def on_disk():  # the survivors not dropped since (each request of a full file drops the least recently used), oldest first
        gone = set(int(x, 16) for x in re.findall(r'serve: session ([0-9a-f]{16}) evicted to make room', log.text()))
        return [d for d in survivors if d not in gone]

    print('4. a turn on a survivor: back from the disk')
    alive = on_disk()
    need(len(alive) >= 2, '%d survivors still on the disk after step 3: %s' % (len(alive), ', '.join('C%d' % (d - S0) for d in alive)))
    ks = alive[-1] - S0  # the most recently used: the requests before its turn take the least recently used first
    mark = log.mark()
    q = N - 8000
    p = prompts[ks][:q] + T[150500:150800]
    s1 = turn(e, 'C%d (survivor) diverge %d back' % (ks, N - q), sid(ks), p)
    b = log.wait_for(back_re(sid(ks)), 'the read-back of the survivor C%d' % ks, since=mark)
    need(not re.search(r'state: session \d+: ERROR', log.text(mark)), 'every piece read back matched its checksum')
    print('  it restarted at %d' % s1['cached'])
    c2 = turn(e, 'same prompt cold', sid(201), p, keep=False)
    ok.append(same('survivor vs cold', s1, c2))

    print('5. a fork of another survivor with the file still full')
    alive = [d for d in on_disk() if d != sid(ks)]
    need(len(alive) >= 1, '%d other survivor(s) still on the disk for the fork' % len(alive))
    kb = alive[-1] - S0
    mark = log.mark()
    s = stats(e)[2]
    need(s['disk_used_bytes'] * 10 >= s['disk_cap_bytes'] * 7, 'the file is still near full: %.0f of %.0f MiB' % (s['disk_used_bytes'] / 2**20, s['disk_cap_bytes'] / 2**20))
    q = 32768
    p = prompts[kb][:q] + T[151000:151400]
    f1 = turn(e, 'fork of C%d at %d' % (kb, q), sid(300), p, base=sid(kb))
    log.wait_for(back_re(sid(kb)), 'the read-back of the base C%d' % kb, since=mark)
    c3 = turn(e, 'same prompt cold', sid(202), p, keep=False)
    ok.append(same('fork vs cold', f1, c3))
    mark = log.mark()
    b2 = turn(e, "C%d's next turn" % kb, sid(kb), prompts[kb][:N - 8000] + T[151500:151800])
    log.wait_for(back_re(sid(kb)), "the read-back of the base for its own next turn", since=mark)
    c4 = turn(e, 'same prompt cold', sid(203), prompts[kb][:N - 8000] + T[151500:151800], keep=False)
    ok.append(same("the base's next turn vs cold", b2, c4))
    for k in list(range(1, made + 1)) + [300, 200, 201, 202, 203]:
        e.release(sid(k))
    return 0 if all(ok) else 1


def run_flush(a):
    """ARM flush (the test `disk-victim-any` has waited for since an earlier run).
    A pool that KEEPS conversations (it holds about 10 of 60,000 positions: nothing is squeezed out), --ssd-flush-s 2 and the smallest disk
    tier. Each conversation is made, left idle, and its idle flush (2 s later) writes its pieces to the file; when the file is full the
    flush of the newest conversation needs space and the least recently used idle conversation gives its images up (it keeps its pieces
    in the pool: only a conversation already on the disk is dropped whole). The conversation being flushed and a conversation a request
    runs on never give up their images (rules::disk_victim_ok). Premise (proved from STATS and rank 0's log, else exit 2): the file's use
    reaches 90 % of its cap while no conversation is only on the disk and no "dropped from the disk tier" line exists.
    Pass: (1) the newest conversation's next turn finds its pieces (`cached` = its history less one: its images were not taken while it
    was flushed, and its pieces never left the pool); (2) the oldest conversation, whose images went, still has its pieces in the pool:
    its next turn reuses its history (`cached` the same) and equals the same prompt prefilled cold; (3) no conversation was dropped whole.
    Settings (engine-test.sh takes them as one string; the pool is --pool-tokens 1800000 = 549 numbers, 10 conversations of about 53):
      MAXCTX=100000 engine-test.sh BIN "--slots 2 --big-slots 0 --pool-tokens 1800000 --disk-gib 1 --snap-every 4096 --ssd-flush-s 2" \\
          disk_full_test.py --arm flush --log ~/dsgb10/logs/engine-disk_full_test.log
    With --kv-split cache a conversation's flush is a quarter of the space, so the file fills after about four times as many: give
    --conversations 40 and a pool of 3,400,000 tokens (--pool-tokens), and the same checks hold (the images given up are quarter images).
    Not run in step C's build."""
    T = [int(x) for x in load_npy_i32(a.tokens)]
    assert len(T) >= 200000, 'the token file is too short'
    e = Engine(a.socket)
    log = Log(a.log)
    s = stats(e)[2]
    need(0 < s['disk_cap_bytes'] <= 2**30, 'the disk tier is %.2f GiB (at most 1)' % (s['disk_cap_bytes'] / 2**30))
    need(s['pool_numbers'] * s['number_bytes'] > 1500 * 2**20, 'the pool is %d numbers = %.0f MiB: it keeps conversations between requests' % (
        s['pool_numbers'], s['pool_numbers'] * s['number_bytes'] / 2**20))
    prompts, hist = {}, {}
    made = 0
    for k in range(1, a.conversations + 1):
        prompts[k] = T[offset(k):offset(k) + N]
        o = turn(e, 'C%d cold' % k, sid(k), prompts[k])
        hist[k] = list(prompts[k]) + list(o['ids'])
        made = k
        time.sleep(4)  # its idle flush (--ssd-flush-s 2) runs and the checkpoint after it
        s = stats(e)[2]
        print('  disk: %.0f of %.0f MiB in use, %d on the disk only, %d in memory' % (
            s['disk_used_bytes'] / 2**20, s['disk_cap_bytes'] / 2**20, s['conv_on_disk'], s['conv_in_memory'] + s['conv_on_lines']))
        if s['disk_used_bytes'] * 10 >= s['disk_cap_bytes'] * 9:
            break
    need(s['disk_used_bytes'] * 10 >= s['disk_cap_bytes'] * 9, 'the file is %.0f of %.0f MiB full after %d conversations' % (
        s['disk_used_bytes'] / 2**20, s['disk_cap_bytes'] / 2**20, made))
    need(made >= 3, 'at least 3 conversations were made (%d)' % made)
    time.sleep(4)
    s = stats(e)[2]
    need(s['conv_on_disk'] == 0 and not re.search(r'dropped from the disk tier', log.text()),
         'no conversation is only on the disk and none was dropped whole: the pool kept them all (%d on the disk only)' % s['conv_on_disk'])
    ok = []
    last = made
    o1 = turn(e, "C%d's next turn (the newest, just flushed)" % last, sid(last), hist[last] + T[150000:150040])
    ok.append(o1['cached'] >= len(hist[last]) - 1)
    print('  %s: the newest conversation kept its images and pieces: cached %d of %d' % ('ok' if ok[-1] else 'FAIL', o1['cached'], len(hist[last])))
    p = hist[1] + T[150100:150140]
    o2 = turn(e, "C1's next turn (the oldest: its images were given up)", sid(1), p)
    ok.append(o2['cached'] >= len(hist[1]) - 1)
    print('  %s: the oldest conversation kept its pieces in the pool: cached %d of %d' % ('ok' if ok[-1] else 'FAIL', o2['cached'], len(hist[1])))
    c = turn(e, 'same prompt cold', sid(200), p, keep=False)
    ok.append(same("the oldest's turn vs cold", o2, c))
    for k in list(range(1, made + 1)) + [200]:
        e.release(sid(k))
    return 0 if all(ok) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--log', required=True, help="rank 0's log from the engine's start (engine-test.sh keeps it)")
    ap.add_argument('--conversations', type=int, default=24)
    ap.add_argument('--arm', choices=['full', 'flush'], default='full')
    a = ap.parse_args()
    try:
        rc = run_flush(a) if a.arm == 'flush' else run(a)
    except NotReached as x:
        print('disk_full_test: the test did not reach the path: %s' % x)
        return 2
    print('disk_full_test: %s' % ('passed' if rc == 0 else 'FAILED'))
    return rc


if __name__ == '__main__':
    sys.exit(main())
