#!/usr/bin/env python3
"""ring.cpp's segment arithmetic (split_at, rs_split, direct_segs, relay_seg, transcribed) checked by
enumeration, from the audit in docs/ring.md's send-buffer race section: for every post size to 4 MiB,
every sender and several slow-link shares, each rank gets exactly what it needs of the post, once,
16-byte aligned, and each relay forwards the part its neighbour sent it as A. Then, for direct mode's reduce-scatter
posts (post_direct, post_opp), the routes of Options::rs_bal 0, 1 and 2: each rank still gets exactly its quarter once,
each data write has its flag behind it on the same domain, and the bytes a post puts on each domain are (0.5, 2.5),
(1.5, 1.5) and (1.5, 1.5) quarters to the 64-byte rounding of split_at. ~10 s:
  python3 engine/comm/tests/seg_model.py [SLOW_LINK]     (default 0, ring_config.h's kSlowLink)"""
import sys
K, SLOW = 4, 0  # kRanks; kSlowLink (ring_config.h)

def split_at(n): return (n // 2) & ~63
def rs_split(sender, q, pct):
    part = (q * pct // 100) & ~63
    next_slow = sender == SLOW or (sender + 1) % K == SLOW
    return q - part if next_slow else part
def direct_segs(rs, sender, to_next, n, pct):
    if rs:
        q = n // 4; hq = rs_split(sender, q, pct); opp = (sender + 2) % K * q
        if to_next: return (opp + hq, q - hq), ((sender + 1) % K * q, q)
        return (opp, hq), ((sender + K - 1) % K * q, q)
    h = split_at(n)
    return ((h, n - h), (0, h)) if to_next else ((0, h), (h, n - h))
def relay_seg(rs, rank, to_prev, n, pct):
    if to_prev: return direct_segs(rs, (rank + 1) % K, False, n, pct)[0]
    return direct_segs(rs, (rank + K - 1) % K, True, n, pct)[0]

def cover(segs, lo, hi):  # the segments tile [lo, hi) exactly (empty ones skipped), 16-byte aligned
    segs = sorted(s for s in segs if s[1])
    x = lo
    for off, ln in segs:
        if off != x or off % 16 or ln % 16 or ln < 0: return False
        x = off + ln
    return x == hi

# --- direct mode, reduce-scatter post of n >= split_domains_from: which domain carries which bytes (rs_bal) ---
def rs_route(sender, n, bal):
    """The data writes of one post, (domain, receiver, offset, length) plus the domains' flags, as post_direct and
    post_opp issue them: a neighbour's quarter (a.len is 0 in direct mode), then the opposite rank's quarter."""
    q = n // 4
    writes, flags = [], []
    for to_next in (True, False):
        rcv = (sender + 1) % K if to_next else (sender + K - 1) % K
        b = direct_segs(True, sender, to_next, n, 50)[1]
        if bal:
            h = split_at(b[1]) if bal == 1 else (b[1] if to_next else 0)
        else:
            h = 0
        chains = {0: [(b[0], h)], 1: [(b[0] + h, b[1] - h)]}
        for d in (0, 1):
            for off, ln in chains[d]:
                writes.append((d, rcv, off, ln))
            flags.append((d, rcv))
    opp, oq = (sender + 2) % K, q
    h = split_at(oq)
    writes += [(0, opp, opp * q, h), (1, opp, opp * q + h, oq - h)]
    flags += [(0, opp), (1, opp)]
    return writes, flags

def check_routes(model_bal=None):
    """model_bal: the rs_bal the model is built with for each asserted route (None = the route asked for);
    a model that ignores the switch (0) fails the totals of routes 1 and 2."""
    bad = 0
    want = {0: (1, 5), 1: (3, 3), 2: (3, 3)}  # in half quarters: domain 0, domain 1
    for bal in (0, 1, 2):
        for n in range(4096, (4 << 20) + 1, 64 * 61):  # n a multiple of 64 from split_domains_from
            n -= n % 64
            q = n // 4
            for s in range(K):
                writes, flags = rs_route(s, n, bal if model_bal is None else model_bal)
                tot = [sum(w[3] for w in writes if w[0] == d) for d in (0, 1)]
                # each receiver gets its own quarter once (neighbours) / the sender's share of it (opposite)
                ok = True
                for r in range(K):
                    if r == s: continue
                    got = sorted((w[2], w[3]) for w in writes if w[1] == r and w[3])
                    lo = r * q if r != (s + 2) % K else (s + 2) % K * q
                    ok &= cover(got, lo, lo + q)
                ok &= sorted(flags) == sorted((d, r) for r in range(K) if r != s for d in (0, 1))
                for d, r, off, ln in writes: ok &= off % 16 == 0 and ln % 16 == 0
                # totals: within the 64-byte rounding of split_at (three halvings: 3 x 64 B) of want x q / 2
                for d in (0, 1):
                    ok &= abs(tot[d] - want[bal][d] * q // 2) <= 3 * 64
                ok &= tot[0] + tot[1] == 3 * q
                if not ok:
                    bad += 1
                    if bad <= 5: print('BAD route', bal, n, s, tot, 'want', [w * q // 2 for w in want[bal]])
    print('rs_bal routes 0/1/2: %d bad' % bad)
    return bad

def main():
    bad = 0; cases = 0
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    slow = int(args[0]) if args else 0
    global SLOW; SLOW = slow
    cap = 4 << 20
    for rs in (False, True):
        step = 64 if rs else 16
        pcts = (50, 0, 25, 37, 75, 100) if rs else (50,)
        for pct in pcts:
            for n in range(step, cap + 1, step):
                q = n // 4
                for s in range(K):
                    cases += 1
                    nx, pv, op = (s + 1) % K, (s + K - 1) % K, (s + 2) % K
                    an, bn = direct_segs(rs, s, True, n, pct)   # to next
                    ap, bp = direct_segs(rs, s, False, n, pct)  # to prev
                    r_nx = relay_seg(rs, nx, False, n, pct)  # next relays s's A on to its next (= op)
                    r_pv = relay_seg(rs, pv, True, n, pct)   # prev relays s's A on to its prev (= op)
                    ok = r_nx == an and r_pv == ap
                    if rs:  # each rank gets its own quarter of s's contribution
                        ok &= cover([bn], nx * q, nx * q + q) and cover([bp], pv * q, pv * q + q)
                        ok &= cover([r_nx, r_pv], op * q, op * q + q)
                        ok &= cover([an], an[0], an[0] + an[1]) and an[0] >= op * q and an[0] + an[1] <= op * q + q
                    else:  # each rank gets all of it
                        ok &= cover([an, bn], 0, n) and cover([ap, bp], 0, n) and cover([r_nx, r_pv], 0, n)
                    if not ok:
                        bad += 1
                        if bad <= 5: print('BAD', 'rs' if rs else 'ae', pct, n, s, an, bn, ap, bp, r_nx, r_pv)
    bad += check_routes(None if '--red' not in sys.argv else 0)
    print('slow link %d: %d cases (all-exchange n = 16 B .. 4 MiB step 16; reduce-scatter n = 64 B .. 4 MiB step 64, '
          'slow_pct 0/25/37/50/75/100; 4 senders): %d bad' % (slow, cases, bad))
    return bad

if __name__ == '__main__':
    sys.exit(1 if main() else 0)
