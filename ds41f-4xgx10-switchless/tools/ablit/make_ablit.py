#!/usr/bin/env python3
"""The runtime abliteration file (--ablit) from a published low-rank edit (mrfakename's abliteration-edit.npz) and the checkpoint.

  make_ablit.py --edit abliteration-edit.npz --ckpt DIR --out ablit.bin
  make_ablit.py --edit abliteration-edit.npz --hf deepseek-ai/DeepSeek-V4.1-Flash-1017@REV --out ablit.bin

File, little-endian: 64-byte header (magic DSGB10AB, u32 version 1, layers 40, dim 5120, inter 2304, k, 0, the edit's sha256),
Q f32 [dim][k], scales f32 [layers][2], G f32 [layers][inter][k] with G = W2^T Q from the FP8 W2 and its scales.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

import numpy as np

DIM, INTER, LAYERS = 5120, 2304, 40
MAGIC = b"DSGB10AB"


def e4m3_table():
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if b >> 7 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        if e == 0:
            v = m / 8.0 * 2.0 ** -6
        elif e == 0xF and m == 7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = s * v
    return t


E4M3 = e4m3_table()
E8M0 = np.array([np.nan if b == 255 else 2.0 ** (b - 127) for b in range(256)], dtype=np.float64)


class Local:
    def __init__(self, root):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.heads = {}

    def header(self, shard):
        if shard not in self.heads:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.heads[shard] = (json.loads(f.read(n)), 8 + n)
        return self.heads[shard]

    def read(self, shard, start, length):
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(start)
            return f.read(length)


class Hub(Local):
    def __init__(self, spec):
        repo, rev = spec.split("@", 1)
        self.base = f"https://huggingface.co/{repo}/resolve/{rev}/"
        self.map = json.loads(self.get("model.safetensors.index.json"))["weight_map"]
        self.heads = {}

    def get(self, name, rng=None):
        req = urllib.request.Request(self.base + name, headers={"Range": "bytes=%d-%d" % rng} if rng else {})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    def header(self, shard):
        if shard not in self.heads:
            n = struct.unpack("<Q", self.get(shard, (0, 7)))[0]
            self.heads[shard] = (json.loads(self.get(shard, (8, 8 + n - 1))), 8 + n)
        return self.heads[shard]

    def read(self, shard, start, length):
        return self.get(shard, (start, start + length - 1))


def tensor(src, name):
    shard = src.map[name]
    head, base = src.header(shard)
    meta = head[name]
    a, b = meta["data_offsets"]
    return np.frombuffer(src.read(shard, base + a, b - a), dtype=np.uint8), meta["shape"], meta["dtype"]


def w2_dequant(src, L):
    w, ws, wd = tensor(src, f"layers.{L}.ffn.shared_experts.w2.weight")
    s, ss, sd = tensor(src, f"layers.{L}.ffn.shared_experts.w2.scale")
    if wd != "F8_E4M3" or sd != "F8_E8M0" or ws != [DIM, INTER] or len(ss) != 2 or DIM % ss[0] or INTER % ss[1]:
        sys.exit(f"layer {L}: unexpected w2 {wd} {ws} / scale {sd} {ss}")
    br, bc = DIM // ss[0], INTER // ss[1]  # the scale blocks: the shape ratio
    W = E4M3[w].reshape(DIM, INTER)
    S = np.repeat(np.repeat(E8M0[s].reshape(ss), br, axis=0), bc, axis=1)
    return W * S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--hf")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = open(a.edit, "rb").read()
    z = np.load(a.edit, allow_pickle=False)
    Q = z["Q"].astype(np.float64)
    scales = z["scales"].astype(np.float32)
    k = Q.shape[1]
    if Q.shape != (DIM, k) or scales.shape != (LAYERS, 2):
        sys.exit(f"edit: Q {Q.shape}, scales {scales.shape}")
    err = np.abs(Q.T @ Q - np.eye(k)).max()
    if err > 1e-5:
        sys.exit(f"edit: Q is not orthonormal (max |Q^T Q - I| = {err:.2e})")

    src = Local(a.ckpt) if a.ckpt else Hub(a.hf)
    G = np.zeros((LAYERS, INTER, k), dtype=np.float32)
    for L in range(LAYERS):
        if scales[L, 1] == 0.0:
            continue  # the shared expert is not edited in this layer: G stays zero, the engine skips it
        W = w2_dequant(src, L)
        G[L] = (W.T @ Q).astype(np.float32)
        print(f"layer {L:2d}: shexp s {scales[L, 1]:.4f}  |G| {np.linalg.norm(G[L]):.4f}", flush=True)

    head = MAGIC + struct.pack("<6I", 1, LAYERS, DIM, INTER, k, 0) + hashlib.sha256(raw).digest()
    assert len(head) == 64
    with open(a.out + ".tmp", "wb") as f:
        f.write(head)
        f.write(Q.astype(np.float32).tobytes())
        f.write(scales.tobytes())
        f.write(G.tobytes())
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: {os.path.getsize(a.out)} bytes, sha256 {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
