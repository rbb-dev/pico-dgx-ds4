#!/usr/bin/env python3
"""Prompts that make a model repeat text, for bit_probe.py (long copy steps, DSGB10_COPY_ROWS): from a real-text token file, n prompts of
LEN tokens each, every one a passage X of LEN - HEAD tokens followed by the first HEAD tokens of X again. A model that sees a passage
start over carries on repeating it, so its reply copies X for hundreds of tokens (long copy steps), and where it stops copying a long
step is rejected in part. bit_probe.py cuts prompt i at i * (file length // n), so the file written here holds n segments of STRIDE
tokens (the prompt, then the rest of the segment unused).

  long_copy_prompts.py --tokens docs-real.npy --out copy-1k.npy --n 4 --len 1024 [--head 64]

Proves: nothing by itself. Used by long_copy_gpu.sh (G3): the same file is sent with DSGB10_COPY_ROWS unset and 64, and bit_probe.py
--compare must say every prompt BIT-EQUAL.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True, help='an int32 .npy of real text')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--len', type=int, default=1024, help='prompt tokens')
    ap.add_argument('--head', type=int, default=64, help='tokens of the passage repeated at the end of the prompt')
    a = ap.parse_args()
    src = np.load(a.tokens).astype(np.int32).reshape(-1)
    body = a.len - a.head
    if body < 128 or a.head < 8 or len(src) < a.n * body:
        sys.exit('need --len - --head >= 128, --head >= 8 and a file of at least n * (len - head) tokens')
    stride = a.len + 16  # the segment: the prompt, then a gap
    out = np.zeros(a.n * stride, dtype=np.int32)
    for i in range(a.n):
        x = src[i * body:(i + 1) * body]
        out[i * stride:i * stride + a.len] = np.concatenate([x, x[:a.head]])
    np.save(a.out, out)
    print('%s: %d prompts of %d tokens (%d + the first %d again), stride %d' % (a.out, a.n, a.len, body, a.head, stride))


if __name__ == '__main__':
    main()
