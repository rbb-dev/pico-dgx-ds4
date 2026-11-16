#!/usr/bin/env python3
"""Write a tokenizer.json for running `dsfront` against its mock engine where the checkpoint's is not at hand.

    make_test_tokenizer.py OUT.json

Byte-level, no merges, ids 0 and 1 = BOS and EOS, the chat encoding's added tokens; it is the file
`dsfront::tokenizer::synthetic` builds in the unit tests. Outputs and prompts round-trip byte for byte;
token counts are bytes, not the model's.
"""
import json
import sys

ADDED = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
         "<think>", "</think>", "<｜tool▁outputs▁begin｜>"]


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def main():
    out = sys.argv[1]
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: len(ADDED) + b for b in range(256)}
    added = [{"id": i, "content": t, "special": True, "normalized": False} for i, t in enumerate(ADDED)]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"added_tokens": added, "model": {"type": "BPE", "vocab": vocab, "merges": []}}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
