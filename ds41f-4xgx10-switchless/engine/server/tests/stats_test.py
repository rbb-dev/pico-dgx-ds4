#!/usr/bin/env python3
"""The dashboard's frames from the engine (docs/protocol.md, version 4): HELLO's keep_positions and STATS version 2 (35
fields), read off rank 0's socket while the engine does things the counts must show: (a) at the start the pool is all free
and keep_positions = (numbers + disk bytes / number_bytes) x 16,384 / 5; (b) after a request free + held = numbers; (c) after
a fork numbers are shared; (d) after release nothing is held.

  python3 engine/server/tests/stats_test.py --socket S --tokens T.npy --numbers N --lines L

Arms (GPU rows, written and not run; exit 0 pass, 1 a failure, 2 the premise was not reached):
  --arm inrun     "frames inside a run". What it proves: the STATS frame is built once a step from the step hook, so a run that lasts
                   publishes all along: 16 decoding users for --seconds (20), the page's reader asking every 0.25 s, see at least 30 distinct
                   frames, all inside one `batch of 16` line of the log. Settings: the serving start, the page's connection the only one
                   (this arm is it, with its 16 sessions of 4,096 tokens as tenant_test's Conn), the front end stopped. Premise (exit 2 when
                   not reached): some number of the frames moved between the first and the last (pool_free, pool_held, engram_lookups,
                   engram_hits or a MemAvailable reading), and after the arm cancels its sessions exactly one `batch of 16` line since
                   --from-offset covers the arm (steps x median within 25 % of its length).
  --arm inprompt   "frames inside a prompt". What it proves: the STATS frame is built while a chunk computes, so a prompt read alone
                   still publishes: the page's reader (STATS asked every 0.25 s) sees at least one distinct frame per two chunks with
                   il_reading 1. Settings: the serving start, the page's connection the only one (this arm is it), the front end stopped;
                   a cold prompt of --prompt-tokens (463,939) alone, a new session, tokens tiled from --tokens with a random first token.
                   Premise (exit 2 when not reached): the request's `turn:` line in --log (since --from-offset) says N chunks, N >= 8,
                   and the prompt ran at least 10 s.
"""

import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_test import DONE, Engine, load_npy_i32  # noqa: E402

STATS_REQ, STATS = 0x05, 0x85
NAMES = ['pool_numbers', 'pool_free', 'pool_held', 'pool_shared', 'pool_reserved', 'number_bytes', 'pool_positions',
         'conv_on_lines', 'conv_in_memory', 'conv_on_disk', 'conv_sharing', 'lines',
         'disk_cap_bytes', 'disk_used_bytes', 'spills_total', 'spill_bytes_total', 'spills_pending', 'restores_total',
         'restore_bytes_total', 'restored_tokens_total', 'restore_wait_ns_total', 'restore_wait_max_ns',
         'il_on', 'il_whole_rows', 'il_push_rows', 'il_gap_ms', 'il_share', 'il_burst_max_ms', 'il_reading', 'il_queued',
         'keep_days', 'extend', 'ctx_limit', 'ranks', 'image_numbers',
         'engram_cache_bytes', 'engram_groups', 'engram_pending_mib', 'engram_lookups', 'engram_hits', 'engram_lend_ok_total',
         'engram_lend_failed_total', 'lent_owed_groups', 'engram_huge_pct', 'engram_evictions',
         'store_numbers', 'mem_avail_min_bytes', 'mem_avail_low_count', 'ids_host_bytes', 'ids_file_sessions',
         'mem_avail_bytes_0', 'mem_avail_bytes_1', 'mem_avail_bytes_2', 'mem_avail_bytes_3',
         'tenant_groups_0', 'tenant_groups_1', 'tenant_groups_2', 'tenant_groups_3', 'tenant_groups_4', 'tenant_groups_5',
         'tenant_groups_6', 'tenant_groups_7', 'tenant_models_groups',
         'kv_split', 'conv_split', 'quarter_slots', 'quarter_numbers', 'ghost_numbers', 'split_pieces_total', 'split_deferred_total',
         'revives_total', 'joins_total', 'join_bytes_total', 'join_ns_total', 'join_max_ns', 'lend_busy_total', 'join_short_total',
         'split_level', 'split_pressure_pm', 'split_age_now_s', 'split_demote_pps', 'l2_numbers', 'l2_dropped_total',
         'clean_numbers', 'image_pending', 'split_rides_total', 'returns_read_total', 'image_waits_total', 'split_reader_waits_total',
         'copy_adapt', 'copy_proposals_total', 'copy_scored_total', 'copy_first_right_total', 'copy_rows_verified_total',
         'copy_rows_accepted_total', 'copy_chance_pm_0', 'copy_chance_pm_1', 'copy_chance_pm_2', 'copy_chance_pm_3',
         'copy_failing_requests_total']
S0 = 0x7374617400000000  # 'stat'
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def stats(e):
    """One STATS: the engine publishes at most every half second, so wait past that for a fresh one."""
    time.sleep(0.6)
    e.send(STATS_REQ, struct.pack('<II', 0, 0))
    while True:
        t, p = e.frame()
        if t == STATS:
            ver, n = struct.unpack('<II', p[:8])
            v = struct.unpack('<%dQ' % n, p[8:8 + 8 * n])
            return ver, n, dict(zip(NAMES, v))


STATS_ASK_S = 0.25  # the page's reader



def turn_chunks(path, offset, rid=None, wait_s=15):
    """The chunk counts of the `turn:` lines in the log since `offset` (the request's own first when `rid` is given). The engine
    writes a request's turn line as the request ends, beside its DONE, so the line may reach the log after the reply: it is
    waited for, up to `wait_s`."""
    import re
    end = time.time() + wait_s
    while True:
        with open(os.path.expanduser(path), 'rb') as f:
            f.seek(offset)
            text = f.read().decode('utf-8', 'replace')
        m = (re.findall(r'req %d turn: .*?(\d+) chunks' % rid, text) if rid is not None else []) or re.findall(r'turn: .*?(\d+) chunks', text)
        if m or time.time() >= end:
            return [int(x) for x in m]
        time.sleep(0.2)

def frames_during(e, rid, max_s):
    """STATS asked every 0.25 s on the one connection while request `rid` runs to its DONE (or max_s passes): the list of (time, fields as a
    tuple) in the order answered. The engine answers with the last frame it published, at most every half second."""
    import select
    out, t0, nxt, done = [], time.time(), 0.0, False
    while not done and time.time() - t0 < max_s:
        if time.time() >= nxt:
            e.send(STATS_REQ, struct.pack('<II', 0, 0))
            nxt = time.time() + STATS_ASK_S
        if select.select([e.s], [], [], 0.05)[0]:
            t, p = e.frame()
            if t == STATS:
                n = struct.unpack('<II', p[:8])[1]
                out.append((time.time() - t0, struct.unpack('<%dQ' % n, p[8:8 + 8 * n])))
            elif t == DONE:
                if struct.unpack('<Q', p[:8])[0] == rid:
                    done = True
    return out


def arm_inprompt(a):
    import random
    import re
    if not a.log:
        print('NOT REACHED: inprompt needs --log (rank 0\'s log: the premise reads the turn line)')
        return 2
    base = [int(x) for x in load_npy_i32(a.tokens)]
    e = Engine(a.socket)
    prompt = [base[i % len(base)] for i in range(a.prompt_tokens)]
    prompt[0] = random.randrange(1000, 100000)  # a cold prompt: nothing of it is in the store
    rid, _ = e.submit(S0 + 0x5a5a, prompt, max_new=1, keep=False)
    t0 = time.time()
    fr = frames_during(e, rid, 900)
    took = time.time() - t0
    m = turn_chunks(a.log, a.from_offset, rid)
    chunks = m[-1] if m else 0
    print('  the prompt ran %.1f s; the log\'s turn line says %d chunks; %d frames answered' % (took, chunks, len(fr)))
    if chunks < 8 or took < 10:
        print('stats_test inprompt: not reached (the premise: a prompt of at least 8 chunks and 10 s, one turn line in the log)')
        return 2
    idx = NAMES.index('il_reading')
    distinct, last = 0, None
    for _t, v in fr:
        if v != last and v[idx] == 1:
            distinct += 1
        last = v
    check(distinct * 2 >= chunks, 'at least one distinct frame per two chunks with il_reading 1: %d frames for %d chunks' % (distinct, chunks))
    print('stats_test inprompt: %s' % ('%d failed' % fails if fails else 'all checks pass'))
    return 1 if fails else 0


def arm_inrun(a):
    from tenant_test import Conn  # noqa: E402
    from watch_rides_test import batches, lines_since  # noqa: E402
    if not a.log:
        print('NOT REACHED: inrun needs --log (rank 0\'s log: the premise reads the batch line)')
        return 2
    c = Conn(a.socket, a.tokens, 16)
    t0 = time.time()
    c.pump(t0 + 3.0)  # the 16 sessions' prompts are read, the run is going
    t0 = time.time()
    fr, last = [], None
    while time.time() - t0 < a.seconds:
        c.pump(time.time() + STATS_ASK_S)
        s = c.stats()
        if s:
            fr.append(s)
    t_arm = time.time() - t0
    c.close()
    n16 = []
    t_end = time.time() + 20
    while time.time() < t_end:
        c.pump(time.time() + 0.5)
        n16 = [x for x in batches(lines_since(a.log, a.from_offset)) if x[1] == 16]
        if n16:
            break
    time.sleep(1)
    n16 = [x for x in batches(lines_since(a.log, a.from_offset)) if x[1] == 16]
    idx = [NAMES.index(k) for k in ('pool_free', 'pool_held', 'engram_lookups', 'engram_hits', 'mem_avail_bytes_0', 'mem_avail_bytes_1', 'mem_avail_bytes_2', 'mem_avail_bytes_3')]
    moved = bool(fr) and any(fr[0][i] != fr[-1][i] for i in idx)
    distinct = sum(1 for i in range(1, len(fr)) if fr[i] != fr[i - 1])
    print('  %d frames in %.1f s, %d distinct from the one before; %d batch line(s) of 16 since the offset' % (len(fr), t_arm, distinct, len(n16)))
    if not moved or len(n16) != 1:
        print('stats_test inrun: not reached (the premise: a counter moved over the arm, and one batch line of 16 covers it)')
        return 2
    _i, _n, steps, _ends, tot_ms, med_ms = n16[0]
    if abs(steps * med_ms / 1e3 - t_arm) > 0.25 * t_arm + 4:
        print('stats_test inrun: not reached (the batch line covers %.1f s, the arm ran %.1f s)' % (steps * med_ms / 1e3, t_arm))
        return 2
    check(distinct >= 30, 'at least 30 distinct frames inside one batch of 16: %d' % distinct)
    print('stats_test inrun: %s' % ('%d failed' % fails if fails else 'all checks pass'))
    return 1 if fails else 0


ARMS = {'inprompt': arm_inprompt, 'inrun': arm_inrun}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--numbers', type=int, help="the pool's numbers (its startup line)")
    ap.add_argument('--lines', type=int)
    ap.add_argument('--arm', choices=('inprompt', 'inrun'), help='a 5.15 arm instead of the frame checks (see the top)')
    ap.add_argument('--log', help='rank 0\'s log (the arms\' premises)')
    ap.add_argument('--from-offset', type=int, default=0, help='the log\'s size when the arm starts')
    ap.add_argument('--prompt-tokens', type=int, default=463939)
    ap.add_argument('--seconds', type=float, default=20)
    a = ap.parse_args()
    if a.arm:
        return ARMS[a.arm](a)
    if a.numbers is None or a.lines is None:
        ap.error('--numbers and --lines are required without --arm')
    T = [int(x) for x in load_npy_i32(a.tokens)]
    e = Engine(a.socket)

    print('(a) at the start')
    ver, n, s = stats(e)
    check(e.version == 4 and ver == 2 and n == len(NAMES), 'protocol %d, STATS version %d with %d fields' % (e.version, ver, n))
    check(s['kv_split'] in (0, 1) and s['quarter_slots'] == 0 and s['conv_split'] == 0 and s['join_ns_total'] == 0 and s['split_pieces_total'] == 0,
          'the split counters at the start: kv_split %d, no slot, nothing split or joined' % s['kv_split'])
    check(s['ctx_limit'] == e.max_ctx and s['ranks'] == 4,
          'a request may use %d tokens (HELLO max_context %d), on %d ranks' % (s['ctx_limit'], e.max_ctx, s['ranks']))
    check(s['image_numbers'] == 0, 'no image rows held at the start')
    check(s['extend'] == 1, 'extend (Extend context) is %d: on at the default start' % s['extend'])
    check(s['pool_numbers'] == a.numbers and s['pool_free'] == a.numbers and s['pool_held'] == 0,
          'the pool: %d numbers, %d free, %d held' % (s['pool_numbers'], s['pool_free'], s['pool_held']))
    check(s['number_bytes'] == 2916352 and s['pool_positions'] == a.numbers * 16384 // 5,
          'a number is %d bytes; the pool holds %d positions' % (s['number_bytes'], s['pool_positions']))
    check(s['lines'] == a.lines and s['conv_on_lines'] + s['conv_in_memory'] + s['conv_on_disk'] == 0,
          '%d lines, no conversation' % s['lines'])
    keep = (a.numbers + s['disk_cap_bytes'] // s['number_bytes']) * 16384 // 5 if s['number_bytes'] else -1
    check(e.keep_positions == keep, "HELLO's keep_positions %d = the pool's and the disk tier's positions %d" % (e.keep_positions, keep))

    print('(b) after a request')
    A, B = S0 + 1, S0 + 2
    o = e.run(A, T[:40000], max_new=8)
    check(not o['error'], 'A: %d tokens' % len(o['ids']))
    _, _, s = stats(e)
    held_a = s['pool_held']
    check(s['conv_on_lines'] + s['conv_in_memory'] == 1 and held_a > 0 and s['pool_free'] + held_a == a.numbers,
          'one conversation (%d on a line, %d in memory), %d numbers held, %d free' % (
              s['conv_on_lines'], s['conv_in_memory'], held_a, s['pool_free']))
    check(s['pool_shared'] == 0 and s['conv_sharing'] == 0, 'nothing shared yet')

    print('(c) after a fork of it')
    o = e.run(B, T[:36000] + T[100000:100300], max_new=8, base=A)
    check(not o['error'] and 0 < o['cached'] <= 36000, 'B, a fork of A: %d tokens, reused %d' % (len(o['ids']), o['cached']))
    _, _, s = stats(e)
    check(s['pool_shared'] > 0 and s['conv_sharing'] == 2,
          '%d numbers shared, %d conversations sharing' % (s['pool_shared'], s['conv_sharing']))
    check(s['pool_free'] + s['pool_held'] == a.numbers and s['pool_held'] < 2 * held_a + 16,
          'the fork holds less than a copy would: %d held for both (A alone %d)' % (s['pool_held'], held_a))

    print('(d) after both are released')
    e.release(A)
    e.release(B)
    _, _, s = stats(e)
    check(s['pool_held'] == 0 and s['pool_shared'] == 0 and s['conv_on_lines'] + s['conv_in_memory'] == 0,
          'nothing held (%d), nothing shared (%d), no conversation' % (s['pool_held'], s['pool_shared']))
    print('stats_test: %s' % ('%d failed' % fails if fails else 'all checks pass'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
