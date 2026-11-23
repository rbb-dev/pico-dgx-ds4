"""What the store tests share (disk_test.py, image_link_test.py, fork_cut_test.py, disk_full_test.py): the engine's own arithmetic
for chunk numbers, and the proof that a test reached the path it is there for, read from rank 0's log and from STATS.

A test that did not reach its path ends with exit 2 and the numbers ("did not reach the path"), never with a verdict about the
engine: a run that compared nothing proves nothing.
"""
import os
import re
import sys
import time

# ---- the engine's constants (model/store_geom.h, model.cu numbers_for, serve.cpp) --------------------------------------------
ENTRIES = 8192          # entries in one chunk number (attn.h kChunkEntries)
LOOKAHEAD = 4096        # positions a line's lists grow past what a step needs (impl.h kMapLookahead)
SLACK = 64              # entries past a position's own (model.cu kMapSlackEntries)
NUMBER_BYTES = 2916352  # one chunk number across the four arrays (STATS number_bytes)
SNAP_NUMBERS = 2        # a window bookmark: 2,906,112 B with the draft module, 2 numbers of 2 MiB
END_NUMBERS = 3         # the end bookmark: 6,074,368 B, 3 numbers
PIECE = 16384           # positions in one cache piece of kv sources 0-2 (source 3's holds half)


def ceil_div(a, b):
    return -(-a // b)


def lists_for(upto):
    """Numbers a line's lists hold to reach position `upto`: the four kv sources, 2 positions an entry on sources 0-2 (model.cu numbers_for)."""
    return sum(ceil_div(upto // (2 if k < 3 else 1) + 1 + SLACK, ENTRIES) for k in range(4))


def reservation(need):
    """Numbers the reservation of a request of `need` positions asks of the pool (Model::slot_chunks_for)."""
    return lists_for(need - 1 + LOOKAHEAD + SLACK)


def stored_lists(positions):
    """Numbers a stored conversation's lists hold for `positions` (store_core.cpp nums_for_pos, summed over the sources)."""
    return sum(ceil_div((positions - 1) // (2 if k < 3 else 1) + 1 + SLACK, ENTRIES) for k in range(4))


def pool_numbers(tokens):
    """Numbers `--pool-tokens tokens` makes (5 numbers every 16,384 positions, rounded up); one of them is the guard."""
    return (tokens * 5 + 16383) // 16384


def keep_snap(s, end, every):
    """serve.cpp keep_snap: whether the window bookmark at s stays when the history ends at `end`."""
    d, dense = end - s, 8 * every
    if d < dense:
        return True
    g, lim = 2 * every, 2 * dense
    while d >= lim:
        g, lim = g * 2, lim * 2
    return s % g == 0


def snaps_kept(end, every, prompt_end=None):
    """The window bookmarks a conversation of `end` positions keeps: those at multiples of `every` that keep_snap keeps, and the
    one at its last prompt's end."""
    ks = [s for s in range(every, end, every) if s >= 128 and keep_snap(s, end, every)]
    if prompt_end and prompt_end % every and prompt_end not in ks:
        ks.append(prompt_end)
    return ks


# ---- the proof ---------------------------------------------------------------------------------------------------------------

class NotReached(Exception):
    pass


class Log:
    """Rank 0's log from where the run started."""

    def __init__(self, path):
        self.path = path
        self.off = os.path.getsize(path)

    def mark(self):
        return os.path.getsize(self.path)

    def text(self, since=None):
        with open(self.path, 'rb') as f:
            f.seek(self.off if since is None else since)
            return f.read().decode(errors='replace')

    def wait_for(self, pattern, what, seconds=15, since=None):
        """The first match of `pattern` in the log so far (the engine writes it a moment after the reply)."""
        end = time.time() + seconds
        while True:
            m = re.search(pattern, self.text(since), re.M)
            if m or time.time() > end:
                break
            time.sleep(0.5)
        if not m:
            raise NotReached('rank 0 log: no line for %s (/%s/)' % (what, pattern))
        print('  log: %s' % m.group(0).strip())
        return m


def need(cond, what):
    print('  %s: %s' % ('ok' if cond else 'NOT REACHED', what))
    sys.stdout.flush()
    if not cond:
        raise NotReached(what)


def spill_re(sid):
    return r'serve: spill session %016x \(([\d.]+) MiB, (\d+) positions\)' % sid


def back_re(sid):
    return (r'serve: session %016x back from disk: ([\d.]+) MiB of ([\d.]+) MiB, (\d+) positions, restart point (-?\d+) of (-?\d+), '
            r'in [\d.]+ ms(?: \(budget (\d+) numbers\))?' % sid)
