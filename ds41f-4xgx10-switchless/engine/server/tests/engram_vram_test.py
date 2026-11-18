#!/usr/bin/env python3
"""The Engram row cache in the display reservation (docs/engine.md, "The Engram row cache in the display reservation") from
outside: the same prompt (24K tokens) asked of an engine started with --engram-vram on and one with off. check passes when
the on log shows slots in the reservation, the pool is larger by the placed MiB (within --slack numbers), and the greedy
outputs are bit-equal.

  1. python3 engram_vram_test.py record --socket S --tokens T.npy --out response-on.json     (engine with --engram-vram on)
  2. python3 engram_vram_test.py record --socket S --tokens T.npy --out response-off.json    (engine with --engram-vram off)
  3. python3 engram_vram_test.py check --log-on rank0-on.log --log-off rank0-off.log --response-on response-on.json --response-off response-off.json
     [--cache-mib 4096]
Exit 0 when every check holds."""
import argparse
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NUMBER_MIB = 2916352 / 1048576.0
fails = 0


def check(cond, what):
    global fails
    print('  %s: %s' % ('ok' if cond else 'FAIL', what))
    sys.stdout.flush()
    fails += not cond


def record(a):
    from serve_test import Engine, load_npy_i32
    prompt = [int(x) for x in load_npy_i32(a.tokens)][:a.prompt_len]
    e = Engine(a.socket)
    o = e.run(0x65767200 + a.salt, prompt, max_new=a.max_new, n_top=1, keep=False)
    if o['error']:
        print('ERROR: ' + o['error'])
        return 1
    bits = [struct.unpack('<I', struct.pack('<f', x))[0] for x in o['lp']]  # the floats as bits: the comparison is of bits
    json.dump({'ids': list(o['ids']), 'lp_bits': bits, 'prompt': len(prompt), 'cached': o['cached']}, open(a.out, 'w'))
    print('recorded %d tokens of a %d-token prompt (cached %d) in %s' % (len(o['ids']), len(prompt), o['cached'], a.out))
    return 0


def pool_numbers(text):
    """(numbers, MiB a number, the room it was sized from in GiB) from the pool's sizing line ("chunk pool: N numbers of X MiB
    (...) on every rank (... free device / available host memory before it, ranks 0-3: d0 / h0 d1 / h1 ... GiB, the Engram
    cache's C to grow)"): the room is the smallest rank's min(device, host - C), what --pool-free-gib sizes from"""
    m = re.search(r'chunk pool: (\d+) numbers of ([\d.]+) MiB \(.*?\) on every rank \((.*)', text)
    if not m:
        return None, None, None
    r = re.search(r'ranks 0-3: ([\d. /]+) GiB, the Engram cache\'s ([\d.]+) to grow', m.group(3))
    room = None
    if r:
        v = [float(x) for x in r.group(1).replace('/', ' ').split()]
        cache = float(r.group(2))
        room = min(min(v[i], v[i + 1] - cache) for i in range(0, len(v) - 1, 2))
    return int(m.group(1)), float(m.group(2)), room


def verify(a):
    on, off = open(a.log_on).read(), open(a.log_off).read()
    print('the start lines')
    pat = r'rank 0: engram row cache: (\d+) MiB of its (\d+) MiB slots in the display reservation'
    m = re.search(pat, on)
    nolines = re.search(r'rank 0: engram row cache: display reservation not used[^\n]*', on)
    check(m is not None, "the on log has rank 0's placement line: %s" % (m.group(0) if m else 'missing (%s)' % (nolines.group(0) if nolines else 'no line')))
    check(re.search(pat, off) is None, 'the off log has no placement line')
    placed = int(m.group(1)) if m else 0
    check(0 < placed <= (int(m.group(2)) if m else 0) and (not m or int(m.group(2)) == a.cache_mib),
          'placed %d MiB of a %d MiB cache' % (placed, a.cache_mib))
    n_on, img_on, room_on = pool_numbers(on)
    n_off, img_off, room_off = pool_numbers(off)
    check(n_on is not None and n_off is not None, "both logs have the pool's sizing line (%s, %s numbers)" % (n_on, n_off))
    if n_on is not None and n_off is not None:
        grew = n_on - n_off
        # the room each start sized from (it moves a little run to run with free memory): the placed MiB come back in it
        want = (room_on - room_off) * 1024 / NUMBER_MIB if room_on is not None and room_off is not None else placed / NUMBER_MIB
        check(grew > 0 and abs(grew - want) <= a.slack,
              'the pool grew by %d numbers; the room it was sized from grew by %.1f numbers (placed %d MiB = %.1f; slack %d)' % (
                  grew, want, placed, placed / NUMBER_MIB, a.slack))
        check(img_on == img_off, 'a number is the same size (%s, %s MiB)' % (img_on, img_off))
    print('the replies')
    ro, rf = json.load(open(a.reply_on)), json.load(open(a.reply_off))
    check(ro['prompt'] == rf['prompt'] and ro['prompt'] >= 20000, 'the same prompt, %d tokens' % ro['prompt'])
    check(ro['cached'] == rf['cached'] == 0, 'both cold (cached %d, %d)' % (ro['cached'], rf['cached']))
    check(len(ro['ids']) > 0 and ro['ids'] == rf['ids'], 'the same %d token ids' % len(ro['ids']))
    check(ro['lp_bits'] == rf['lp_bits'], 'the same log-probability bits')
    print('PASS' if not fails else 'FAIL')
    return 1 if fails else 0


def main():
    # one flat parser, options in any order (engine-test.sh puts --socket and --tokens before the command)
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['record', 'check'])
    ap.add_argument('--socket', default='/tmp/dsgb10-engine.sock')
    ap.add_argument('--tokens', default='/data/dsgb10/golden/synth270k/tokens.npy')
    ap.add_argument('--prompt-len', type=int, default=24576)
    ap.add_argument('--max-new', type=int, default=24)
    ap.add_argument('--salt', type=int, default=0)
    ap.add_argument('--out')
    ap.add_argument('--log-on')
    ap.add_argument('--log-off')
    ap.add_argument('--reply-on')
    ap.add_argument('--reply-off')
    ap.add_argument('--cache-mib', type=int, default=4096, help="the engine's --engram-cache-mib")
    ap.add_argument('--slack', type=int, default=8)
    a = ap.parse_args()
    need = ['out'] if a.cmd == 'record' else ['log_on', 'log_off', 'reply_on', 'reply_off']
    missing = [n for n in need if getattr(a, n) is None]
    if missing:
        ap.error('%s needs %s' % (a.cmd, ', '.join('--' + n.replace('_', '-') for n in missing)))
    return record(a) if a.cmd == 'record' else verify(a)


if __name__ == '__main__':
    sys.exit(main())
