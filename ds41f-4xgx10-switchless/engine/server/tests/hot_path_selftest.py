#!/usr/bin/env python3
"""Source scan of what stands on a request's path (host only, no GPU, no sockets). Each row reads a function's
body (comments and string literals blanked, braces matched) in server/serve.cpp, cmd.cpp or cmd.h and says where a call stands:
  V3  pf_piece publishes the STATS frame between run_prefill( and m_.drain() (the device computes the chunk meanwhile) and nowhere else;
      loop()'s publish is guarded by an empty prompt queue; idle_wait publishes (a long prompt waiting for a line)
  V4  watch_tick returns while a prompt is being read (pfq_.front()->begun) unless the watch is two periods late; pf_piece asks
      chunk_ride_due and sets kPfWatch; run_prefill calls mem_watch_run after m.prefill( and only under kPfWatch; cmd.h names kPfWatch = 4
  V2  the two step callbacks (decode_one's, decode_batch's) hold no publish(, lend_pause_due(, ride_watch( or ride_split( and one now_s()
      (the first token's dec1_ms); each of the four loops calls step() exactly once, on the line after its turn's last ride(wctl);
      step_hook reads the clock once
  python3 hot_path_selftest.py [--rows V3,V4,V2] [SERVE [CMD_CPP [CMD_H]]]     (default: all rows, the files beside this one)
Exit 0 all rows pass, 1 a row fails.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
rows = {'V3', 'V4', 'V2'}
if args and args[0] == '--rows':
    rows = set(args[1].split(','))
    args = args[2:]
SERVE = args[0] if len(args) > 0 else os.path.join(HERE, '..', 'serve.cpp')
CMDCPP = args[1] if len(args) > 1 else os.path.join(HERE, '..', 'cmd.cpp')
CMDH = args[2] if len(args) > 2 else os.path.join(HERE, '..', 'cmd.h')
fails = 0


def report(case, ok, what):
    global fails
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', case, what))
    fails += not ok


def blank(src):
    """comments, string and character literals made spaces (newlines kept), so braces and words in them count for nothing"""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
        elif src.startswith('/*', i):
            j = src.find('*/', i)
            j = n if j < 0 else j + 2
            out.append(re.sub(r'[^\n]', ' ', src[i:j]))
            i = j
        elif c in '"\'':
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(c + re.sub(r'[^\n]', ' ', src[i + 1:j - 1]) + c)
            i = j
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def match(src, i, open_c, close_c):
    depth = 0
    for j in range(i, len(src)):
        if src[j] == open_c:
            depth += 1
        elif src[j] == close_c:
            depth -= 1
            if depth == 0:
                return j
    return -1


def body(src, anchor):
    """the text between the braces of the function or lambda whose head matches `anchor` (a regex up to just before its parameter list)"""
    m = re.search(anchor, src)
    if not m:
        return None
    p = src.index('(', m.end())
    q = match(src, p, '(', ')')
    b = src.index('{', q)
    e = match(src, b, '{', '}')
    return src[b + 1:e]


serve = blank(open(SERVE, errors='replace').read())
cmdcpp = blank(open(CMDCPP, errors='replace').read())
cmdh = blank(open(CMDH, errors='replace').read())


def have(case, text, what):
    if text is None:
        report(case, False, what + ': the function is not in the file')
    return text is not None


def count(pat, text):
    return len(re.findall(pat, text))


if 'V3' in rows:
    pf = body(serve, r'bool pf_piece\b')
    if have('V3', pf, 'pf_piece'):
        pubs = [m.start() for m in re.finditer(r'\bpublish\(', pf)]
        a, d = pf.find('run_prefill('), pf.find('m_.drain()')
        report('V3', len(pubs) == 1 and a >= 0 and d > a and a < pubs[0] < d,
               'pf_piece: %d publish( call(s); the one stands %s run_prefill( and m_.drain()' % (len(pubs), 'between' if pubs and a < pubs[0] < d else 'NOT between'))
    lp = body(serve, r'void loop\b')
    if have('V3', lp, 'loop'):
        bare = count(r'(?m)^\s*publish\(\);', lp)
        guarded = count(r'if \(pfq_\.empty\(\)\) publish\(\);', lp)
        report('V3', bare == 0 and guarded == 1, 'loop(): %d unguarded publish(), %d under `if (pfq_.empty())`' % (bare, guarded))
    iw = body(serve, r'void idle_wait\b')
    if have('V3', iw, 'idle_wait'):
        report('V3', 'publish(' in iw, 'idle_wait publishes: %d call(s)' % count(r'\bpublish\(', iw))

if 'V4' in rows:
    wt = body(serve, r'void watch_tick\b')
    if have('V4', wt, 'watch_tick'):
        report('V4', 'pfq_.front()->begun' in wt and 'defer_to_chunk(' in wt,
               'watch_tick holds the guard (a prompt being read, unless the watch is two periods late): pfq_.front()->begun %d, defer_to_chunk %d' %
               (count(r'pfq_\.front\(\)->begun', wt), count(r'defer_to_chunk\(', wt)))
    pf = body(serve, r'bool pf_piece\b')
    if have('V4', pf, 'pf_piece'):
        report('V4', 'chunk_ride_due(' in pf and 'kPfWatch' in pf and re.search(r'run_prefill\([^;]*;[\s\S]*watch_at_\s*=', pf) is not None,
               'pf_piece asks chunk_ride_due(, sets kPfWatch and sets watch_at_ after run_prefill')
    rp = body(cmdcpp, r'uint64_t run_prefill\b')
    if have('V4', rp, 'run_prefill'):
        a, w = rp.find('m.prefill('), rp.find('mem_watch_run(')
        report('V4', a >= 0 and w > a and re.search(r'if \(c\.flags & kPfWatch\)\s*mem_watch_run\(', rp) is not None,
               'run_prefill: mem_watch_run( %s m.prefill( and under `c.flags & kPfWatch`' % ('after' if a >= 0 and w > a else 'NOT after'))
    report('V4', re.search(r'kPfWatch\s*=\s*4\b', cmdh) is not None, 'cmd.h names kPfWatch = 4 beside kPfSample and kPfCed')

CALLBACKS = (('decode_one callback', serve, r'decode_loop\(m_, c, wd_, \[&\]'), ('decode_batch callback', serve, r'const OnBatch cb = \[&\]'))
if 'V2' in rows:
    for name, src, anchor in CALLBACKS:
        b = body(src, anchor)
        if not have('V2', b, name):
            continue
        hits = {w: count(r'\b%s\(' % w, b) for w in ('publish', 'lend_pause_due', 'ride_watch', 'ride_split')}
        report('V2', not any(hits.values()), '%s: %s' % (name, ', '.join('%s %d' % kv for kv in hits.items())))
        report('V2', count(r'\bnow_s\(\)', b) == 1, '%s: %d now_s() (the first token\'s dec1_ms is the one)' % (name, count(r'\bnow_s\(\)', b)))
    for name, anchor in (('spec_loop', r'DecodeEnd spec_loop\b'), ('decode_loop', r'DecodeEnd decode_loop\b'),
                         ('batch_loop', r'std::vector<BatchEnd> batch_loop\b'), ('spec_batch_loop', r'std::vector<DecodeEnd> spec_batch_loop\b')):
        b = body(cmdcpp, anchor)
        if not have('V2', b, name):
            continue
        n = count(r'\bstep\(\)', b)
        lines = b.split('\n')
        last = max((i for i, ln in enumerate(lines) if 'ride(wctl);' in ln), default=-1)
        nxt = lines[last + 1].strip() if 0 <= last < len(lines) - 1 else ''
        report('V2', n == 1 and nxt == 'if (step) step();', '%s: %d step() call(s); the line after its last ride(wctl) is `%s`' % (name, n, nxt))
    sh = body(serve, r'(?:void|bool) step_hook\b')
    if have('V2', sh, 'step_hook'):
        report('V2', count(r'\bnow_s\(\)', sh) == 1, 'step_hook: %d now_s()' % count(r'\bnow_s\(\)', sh))

print('hot_path_selftest: %s' % ('%d failed' % fails if fails else 'all rows pass'))
sys.exit(1 if fails else 0)
