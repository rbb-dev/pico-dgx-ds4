#!/usr/bin/env python3
"""Per-layer activations of the vision tower and aligner, and every operation's input and output, saved as .npy files.

Runs the checkpoint's own reference code (inference/vision.py, inference/image_processor.py) on the GPU, with forward
hooks, for every image. Free-running points (every attention mode; the points the encoder's free-running run is held to):

    patch_embed            the patch embedding                                   [n, 1024]
    blockNN.attn_add       x after block NN's attention add (NN = 00..31)       [n, 1024]
    blockNN.mlp_add        x after block NN's MLP add                           [n, 1024]
    norm                   the final norm                                        [n, 1024]
    unfold                 the aligner's 3x3 unfolded input                      [m, 9216]
    g                      gelu(w1(unfold))                                      [m, 5120]
    aligner                the aligner's output                                  [m, 5120]
    span                   the span rows (delimiters and aligner rows)           [rows, 5120]

Four attention modes, the same weights (--attn takes a comma list):
    ref    the reference unchanged (scaled_dot_product_attention on 3D strided views: PyTorch's MATH backend)
    shim   what the running encoder does (tools/vision/vision_encoder.py: the same function on a contiguous 4D view,
           which takes the flash kernel)
    mem    the same 4D view, the memory-efficient backend forced
    flash  the same 4D view, the flash backend forced
shim, mem and flash against ref are the null pairs vit_compare.py takes its lines from.

Operation dumps (--ops STEMS, mode ref only, whole tensors, saved under OUT/ops/<stem>/): the input and the output of every
operation of the blocks named by --op-blocks, and of the operations outside the blocks:

    patches_in [n,588]  rope_cos, rope_sin [n,32] float32  (and pixels [H,W,3] u8 for every image, ops or not)
    blockNN.x_in  .norm1_out  .qkv (wqkv output, before the rotation)  .q_rope  .k_rope  (the attention's inputs)
    .attn_o (its output)  .wo_out  .attn_add  .norm2_out  .w1_out  .swiglu_out (w2's input)  .w2_out  .mlp_add
    patch_embed  norm_in  norm  unfold  a1_out (aligner w1, before gelu)  g  aligner  span

--floors measures, per operation of those dumps, what a correct implementation can be asked for: for rmsnorm, rope, swiglu
and gelu torch's own bf16 result against the float64 answer; for attention the largest spread between the reference's
attention and each other mode on the same q, k, v (OUT/floors.json). --gemm-random writes random data of each matrix
product's shape with cuBLAS's output (OUT/gemm_random/<shape>/{a,w,b,out}.npy) and the copy-type weights
(OUT/weights/<name>.npy) as the reference holds them.

Files are bf16, stored as the raw 16 bits in a uint16 array (numpy has no bf16). Whole tensors for the images named in
--full (and, by default, for any image of at most --full-patches patches); for the others every 16th row plus a SHA-256
of the whole tensor's bytes. Layout:

    OUT/MODE/MANIFEST.json
    OUT/MODE/<image stem>/<point>.npy
    OUT/ops/MANIFEST.json, OUT/ops/<image stem>/<point>.npy, OUT/floors.json, OUT/weights/, OUT/gemm_random/

Run it inside the container (the GPU and torch are there; vit_compare.py is imported for its statistics, copy it beside):
    docker exec pico-vision python3 /app/dump_ref.py --model MODEL_SNAPSHOT --images DIR --out /tmp/f0 \
        --attn ref,shim,mem,flash --ops noise_544,noise_768 --floors --gemm-random
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

STRIDE = 16
MODES = ("ref", "shim", "mem", "flash")
DEFAULT_OP_BLOCKS = (0, 1, 2, 7, 16, 31)
BLOCK_OP_NAMES = ("x_in", "norm1_out", "qkv", "q_rope", "k_rope", "attn_o", "wo_out", "attn_add", "norm2_out", "w1_out",
                  "swiglu_out", "w2_out", "mlp_add")
GLOBAL_OP_NAMES = ("patches_in", "rope_cos", "rope_sin", "pixels", "patch_embed", "norm_in", "norm", "unfold", "a1_out", "g",
                   "aligner", "span")
FREE_NAMES = {"patch_embed", "norm", "unfold", "g", "aligner", "span"}
_BLOCK_RE = re.compile(r"block(\d+)\.(.*)$")
# the matrix products' shapes (N, K as the engine holds them; patch_embed's K is padded to 640), bias or not
GEMM_SHAPES = OrderedDict([("patch_embed", (1024, 640, True)), ("wqkv", (3072, 1024, True)), ("wo", (1024, 1024, True)),
                           ("w1", (5632, 1024, False)), ("w2", (1024, 2816, False)), ("aligner1", (5120, 9216, True)),
                           ("aligner2", (5120, 5120, True))])


# ---------------------------------------------------------------------------------------------------------------
# numeric helpers (numpy only: usable on any machine for the comparison)
# ---------------------------------------------------------------------------------------------------------------


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def to_bf16_np(f):
    """float32 -> bf16 bits, round to nearest even (the finite values the tower holds)."""
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def pixel_table_np():
    """The 256 bf16 values of a pixel byte, the reference's own formula in fp32: ((v / 255) - 0.5) / 0.5, rounded once."""
    v = np.arange(256, dtype=np.float32)
    return to_bf16_np((v / np.float32(255) - np.float32(0.5)) / np.float32(0.5))


def ulp_key(u16):
    """bf16 bit patterns as integers that grow with the value, so the difference of two keys is the number of
    representable bf16 steps between them (-0 and +0 are the same key)."""
    k = (u16 & 0x7FFF).astype(np.int32)
    return np.where((u16 & 0x8000) != 0, -k, k)


def metrics(cand, ref):
    """Distance of candidate rows from reference rows (both uint16 bf16 bits, [rows, cols])."""
    a = bf16_to_f32(cand).astype(np.float64)
    b = bf16_to_f32(ref).astype(np.float64)
    nonfinite = int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum())
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    dn, an, bn = np.linalg.norm(a - b, axis=1), np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    rel = np.where(bn > 0, dn / np.maximum(bn, 1e-30), dn)  # relative error of the row (L2)
    cos = np.where((an > 0) & (bn > 0), (a * b).sum(1) / np.maximum(an * bn, 1e-30), np.where(dn == 0, 1.0, 0.0))
    ulps = np.abs(ulp_key(cand) - ulp_key(ref))
    return {"max_rel_row_err": float(rel.max()), "min_row_cos": float(cos.min()),
            "share_over_1_step": float((ulps > 1).mean()), "share_not_equal": float((ulps > 0).mean()),
            "max_steps": int(ulps.max()), "rows": int(a.shape[0]), "nonfinite": nonfinite}


def aggregate(ms):
    """Over the images: the worst of each distance."""
    return {"max_rel_row_err": max(m["max_rel_row_err"] for m in ms), "min_row_cos": min(m["min_row_cos"] for m in ms),
            "share_over_1_step": max(m["share_over_1_step"] for m in ms),
            "share_not_equal": max(m["share_not_equal"] for m in ms), "max_steps": max(m["max_steps"] for m in ms),
            "nonfinite": sum(m["nonfinite"] for m in ms), "images": len(ms)}


# ---------------------------------------------------------------------------------------------------------------
# sinks: where a captured tensor goes. A sink says which points it takes (accepts); wants_ops tells the run to hook the
# operations of the blocks too, which costs nothing for the plain free-running dumps.
# ---------------------------------------------------------------------------------------------------------------


def is_free_point(name):
    m = _BLOCK_RE.match(name)
    if m:
        return m.group(2) in ("attn_add", "mlp_add")
    return name in FREE_NAMES or name == "patch_embed"


def to_np(t):
    """A tensor as numpy: float32 stays float32 (the rope tables), everything else is bf16 bits."""
    import torch
    t = t.detach().contiguous()
    if t.dtype == torch.float32:
        return t.cpu().numpy()
    return t.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)


def to_u16(t):
    return to_np(t)


def select(u, full):
    return u if full else u[::STRIDE]


class SaveSink:
    wants_ops = False

    def __init__(self, root, stem, full):
        self.dir, self.full, self.points = os.path.join(root, stem), full, OrderedDict()
        os.makedirs(self.dir, exist_ok=True)

    def accepts(self, name):
        return is_free_point(name)

    def put(self, name, t):
        if not self.accepts(name):
            return
        u = to_u16(t)
        sel = np.ascontiguousarray(select(u, self.full))
        np.save(os.path.join(self.dir, name + ".npy"), sel)
        self.points[name] = {"shape": list(u.shape), "saved_rows": int(sel.shape[0]), "stride": 1 if self.full else STRIDE,
                             "sha256_whole": hashlib.sha256(u.tobytes()).hexdigest()}


class OpsSink:
    """Whole tensors of every operation's input and output, for the blocks in op_blocks and for the operations outside."""
    wants_ops = True

    def __init__(self, root, stem, op_blocks):
        self.root, self.stem, self.op_blocks = root, stem, set(op_blocks)
        self.dir = os.path.join(root, stem)
        os.makedirs(self.dir, exist_ok=True)
        self.points = OrderedDict()

    def accepts(self, name):
        m = _BLOCK_RE.match(name)
        if m:
            return int(m.group(1)) in self.op_blocks and m.group(2) in BLOCK_OP_NAMES
        return name in GLOBAL_OP_NAMES

    def put(self, name, t):
        if self.accepts(name):
            self.put_array(name, to_np(t))

    def put_array(self, name, arr):
        np.save(os.path.join(self.dir, name + ".npy"), np.ascontiguousarray(arr))
        self.points[name] = list(arr.shape)


class CompareSink:
    """Compares each captured tensor with a saved dump of the same image; keeps only the distances."""
    wants_ops = False

    def __init__(self, root, stem, full, into):
        self.dir, self.full, self.into = os.path.join(root, stem), full, into

    def accepts(self, name):
        return is_free_point(name)

    def put(self, name, t):
        if not self.accepts(name):
            return
        ref = np.load(os.path.join(self.dir, name + ".npy"), mmap_mode="r")
        sel = select(to_u16(t), self.full)
        if sel.shape != ref.shape:
            raise RuntimeError("%s: shape %s against the dump's %s" % (name, sel.shape, ref.shape))
        self.into.setdefault(name, []).append(metrics(sel, np.asarray(ref)))


class DictSink:
    """Keeps free-running points in memory (tests)."""
    wants_ops = False

    def __init__(self):
        self.d = {}

    def accepts(self, name):
        return is_free_point(name)

    def put(self, name, t):
        if self.accepts(name):
            self.d[name] = to_u16(t)


class Tee:
    def __init__(self, *sinks):
        self.sinks = sinks

    @property
    def wants_ops(self):
        return any(s.wants_ops for s in self.sinks)

    def accepts(self, name):
        return any(s.accepts(name) for s in self.sinks)

    def put(self, name, t):
        for s in self.sinks:
            s.put(name, t)

    def put_array(self, name, arr):
        for s in self.sinks:
            if hasattr(s, "put_array") and s.accepts(name):
                s.put_array(name, arr)


# ---------------------------------------------------------------------------------------------------------------
# the attention modes
# ---------------------------------------------------------------------------------------------------------------


class AttnShim:
    """Stands in for `F` in the reference's vision module: every function is torch.nn.functional's, except
    scaled_dot_product_attention, which runs in the chosen mode and, when a recorder is set, reports (q, k, o) as the
    operation's input and output. The reference mode passes the tensors on exactly as received (3D strided views)."""

    def __init__(self, F, mode, torch, record=None):
        self.F, self.mode, self.torch, self.record = F, mode, torch, record

    def __getattr__(self, n):
        return getattr(self.F, n)

    def sdpa(self, q, k, v, *a, **kw):
        F, mode = self.F, self.mode
        if mode == "ref":
            return F.scaled_dot_product_attention(q, k, v, *a, **kw)
        if q.dim() == 3:
            q4, k4, v4 = q.unsqueeze(0).contiguous(), k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous()
        else:
            q4, k4, v4 = q.contiguous(), k.contiguous(), v.contiguous()
        if mode == "shim":
            o = F.scaled_dot_product_attention(q4, k4, v4, *a, **kw)
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            be = SDPBackend.EFFICIENT_ATTENTION if mode == "mem" else SDPBackend.FLASH_ATTENTION
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(q4, k4, v4, *a, **kw)
        return o[0] if q.dim() == 3 else o

    def scaled_dot_product_attention(self, q, k, v, *a, **kw):
        o = self.sdpa(q, k, v, *a, **kw)
        if self.record is not None:
            self.record(q, k, v, o)
        return o


# ---------------------------------------------------------------------------------------------------------------
# the model (the same loading and the same shim as tools/vision/vision_encoder.py)
# ---------------------------------------------------------------------------------------------------------------


class Model:
    def __init__(self, model_dir, device="cuda"):
        import torch
        import torch.nn.functional as F
        sys.path.insert(0, os.path.join(model_dir, "inference"))
        import image_processor as ip
        import vision as vmod
        from safetensors import safe_open

        cfg = json.load(open(os.path.join(model_dir, "inference", "config.json")))
        args = SimpleNamespace(**cfg)
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        want = [k for k in idx if k.startswith(("vision.", "aligner.", "image_"))]
        sd = {}
        for f in sorted({idx[k] for k in want}):
            with safe_open(os.path.join(model_dir, f), framework="pt", device="cpu") as h:
                for k in want:
                    if idx[k] == f:
                        sd[k] = h.get_tensor(k)
        self._build(torch, F, ip, vmod, args, device, sd)

    def _build(self, torch, F, ip, vmod, args, device, sd):
        self.torch, self.F, self.ip, self.vmod, self.args, self.device = torch, F, ip, vmod, args, device
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(device)
        self.vit, self.aligner = vmod.ViT(args), vmod.Aligner(args)
        self.vit.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("vision.")}, strict=True)
        self.aligner.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("aligner.")}, strict=True)
        self.vit.eval(), self.aligner.eval()
        self.delim = {k: sd[k].to(device, torch.bfloat16) for k in ("image_start", "image_newline", "image_end")}

    @classmethod
    def synthetic(cls, seed=0, layers=3, device="cpu"):
        """A tiny random ViT of the reference's own classes (tests): the same code paths, no checkpoint."""
        import torch
        import torch.nn.functional as F
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "..", "ref", "inference"))
        import image_processor as ip
        import vision as vmod
        args = SimpleNamespace(vision_dim=64, vision_n_heads=4, vision_inter_dim=48, vision_patch_size=14,
                               vision_downsample_ratio=3, vision_n_layers=layers, vision_rope_theta=10000,
                               vision_max_wh_ratio=None, vision_min_pixels=0, vision_max_n_token=1024, dim=80)
        torch.manual_seed(seed)
        m = cls.__new__(cls)
        m.torch, m.F, m.ip, m.vmod, m.args, m.device = torch, F, ip, vmod, args, device
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(device)
        m.vit, m.aligner = vmod.ViT(args), vmod.Aligner(args)
        for mod in list(m.vit.modules()) + list(m.aligner.modules()):
            if isinstance(mod, vmod.RMSNorm):
                mod.weight.data = 1 + 0.1 * torch.randn(mod.weight.shape, dtype=torch.float32)
        m.vit.eval(), m.aligner.eval()
        m.delim = {k: torch.randn(args.dim, dtype=torch.bfloat16) for k in ("image_start", "image_newline", "image_end")}
        return m

    # -- images -------------------------------------------------------------------------------------------------

    def pixels_of(self, raw):
        """The u8 image the reference feeds its patches from (image_processor.load_image's own steps up to the patching)."""
        import io
        from PIL import Image, ImageOps
        a = self.args
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
        _, _, bh, bw = self.ip.plan_image_grid(image.width, image.height, a)
        if a.vision_max_wh_ratio is not None and image.width >= a.vision_max_wh_ratio * image.height:
            image = image.resize((bw, bh))
        else:
            image = ImageOps.pad(image, (bw, bh), color=(127, 127, 127))
        return np.asarray(image, dtype=np.uint8)

    def prepare_pixels(self, pix):
        """(patches [n,3,14,14] bf16, token types, (n_h, n_w, lh, lw), pixels) from a u8 image of whole patches."""
        torch, p = self.torch, self.args.vision_patch_size
        h, w = pix.shape[0] // p, pix.shape[1] // p
        r = self.args.vision_downsample_ratio
        lh, lw = -(-h // r), -(-w // r)
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(pix, dtype=np.float32)).permute(2, 0, 1) / 255
            x = ((x - 0.5) / 0.5).to(torch.bfloat16)
            patches = x.reshape(3, h, p, w, p).permute(1, 3, 0, 2, 4).reshape(h * w, 3, p, p)
            types = self.ip.image_token_types(lh, lw)
        return patches, types, (h, w, lh, lw), np.asarray(pix, dtype=np.uint8)

    def prepare(self, path):
        torch = self.torch
        raw = open(path, "rb").read()
        pix = self.pixels_of(raw)
        prepared = self.prepare_pixels(pix)
        # the instrument check: the reference's own load_image makes the same patches from the same bytes
        with torch.inference_mode():
            patches, h, w, lh, lw = self.ip.load_image({"data": raw}, self.args)
        if not (torch.equal(patches.cpu(), prepared[0].cpu()) and (h, w, lh, lw) == tuple(prepared[2])):
            raise RuntimeError("%s: the pixels rebuilt here do not give image_processor.load_image's patches" % path)
        return prepared

    # -- one forward pass ---------------------------------------------------------------------------------------

    def _shim(self, attn, record=None):
        return AttnShim(self.F, attn, self.torch, record)

    def forward_plain(self, prepared, attn):
        """The same forward without any hook: the free-running points as bf16 bits (tests: hooks change nothing)."""
        torch = self.torch
        patches, types, (h, w, lh, lw) = prepared[:3]
        self.vmod.F = self._shim(attn)
        out = {}
        try:
            with torch.inference_mode():
                x = self.vit.patch_embed(patches.to(self.device))
                out["patch_embed"] = to_u16(x)
                cos, sin = self.vmod.get_vision_cos_sin(h, w, self.vit.rope_dim, self.vit.rope_theta)
                for i, b in enumerate(self.vit.blocks):
                    x = b(x, cos, sin)
                    out["block%02d.mlp_add" % i] = to_u16(x)
                out["aligner"] = to_u16(self.aligner(self.vit.norm(x), h, w))
        finally:
            self.vmod.F = self.F
        return out

    def run(self, prepared, attn, sink):
        """One forward pass of the tower and aligner with the hooks; every point goes to the sink."""
        torch, ip = self.torch, self.ip
        patches, types, (h, w, lh, lw) = prepared[:3]
        pix = prepared[3] if len(prepared) > 3 else None
        ops = getattr(sink, "wants_ops", False)
        cur = [0]
        hooks, held = [], {}

        def attn_record(q, k, v, o):  # q, k, v, o as the reference's attention sees them: [heads, n, head_dim]
            i = cur[0]
            n = q.shape[-2]
            flat = lambda t: t.reshape(-1, n, t.shape[-1]).transpose(0, 1).reshape(n, -1)
            sink.put("block%02d.q_rope" % i, flat(q))
            sink.put("block%02d.k_rope" % i, flat(k))
            sink.put("block%02d.attn_o" % i, flat(o))

        self.vmod.F = self._shim(attn, attn_record if ops else None)

        def reg(h_):
            hooks.append(h_)

        def done_block(i):
            def hook(m, args, o):  # returns nothing: a forward hook that returns a value would replace the output
                sink.put("block%02d.mlp_add" % i, o)
                held.pop(i, None)
            return hook

        def pre_block(i):
            def hook(m, args):
                held[i] = args[0]
                cur[0] = i
                if ops:
                    sink.put("block%02d.x_in" % i, args[0])
            return hook

        def out_hook(name):
            return lambda m, a_, o: sink.put(name, o)

        reg(self.vit.patch_embed.register_forward_hook(lambda m, i, o: sink.put("patch_embed", o)))
        if ops:
            reg(self.vit.patch_embed.register_forward_pre_hook(lambda m, args: sink.put("patches_in", args[0].flatten(1))))
            reg(self.vit.norm.register_forward_pre_hook(lambda m, args: sink.put("norm_in", args[0])))
            reg(self.aligner.w1.register_forward_hook(out_hook("a1_out")))
        for i, b in enumerate(self.vit.blocks):
            reg(b.register_forward_pre_hook(pre_block(i)))
            reg(b.attn.register_forward_hook(lambda m, args, o, i=i: sink.put("block%02d.attn_add" % i, held[i] + o)))
            reg(b.register_forward_hook(done_block(i)))
            if ops:
                reg(b.norm1.register_forward_hook(out_hook("block%02d.norm1_out" % i)))
                reg(b.attn.wqkv.register_forward_hook(out_hook("block%02d.qkv" % i)))
                reg(b.attn.wo.register_forward_hook(out_hook("block%02d.wo_out" % i)))
                reg(b.norm2.register_forward_hook(out_hook("block%02d.norm2_out" % i)))
                reg(b.mlp.w1.register_forward_hook(out_hook("block%02d.w1_out" % i)))
                reg(b.mlp.w2.register_forward_pre_hook(lambda m, args, i=i: sink.put("block%02d.swiglu_out" % i, args[0])))
                reg(b.mlp.w2.register_forward_hook(out_hook("block%02d.w2_out" % i)))
        reg(self.vit.norm.register_forward_hook(lambda m, i, o: sink.put("norm", o)))
        reg(self.aligner.w1.register_forward_pre_hook(lambda m, args: sink.put("unfold", args[0])))
        reg(self.aligner.w1.register_forward_hook(lambda m, i, o: sink.put("g", self.F.gelu(o))))
        try:
            with torch.inference_mode():
                feats = self.aligner(self.vit(patches.to(self.device), h, w), h, w)
                sink.put("aligner", feats)
                t = types.to(self.device)
                span = torch.empty((t.numel(), feats.shape[1]), dtype=torch.bfloat16, device=self.device)
                span[t == ip.IMAGE_START] = self.delim["image_start"]
                span[t == ip.IMAGE_END] = self.delim["image_end"]
                span[t == ip.IMAGE_NEW_LINE] = self.delim["image_newline"]
                span[t == ip.IMAGE] = feats.to(torch.bfloat16)
                sink.put("span", span)
                if ops:
                    cos, sin = self.vmod.get_vision_cos_sin(h, w, self.vit.rope_dim, self.vit.rope_theta)
                    sink.put("rope_cos", cos[:, 0, :])
                    sink.put("rope_sin", sin[:, 0, :])
                    if pix is not None and hasattr(sink, "put_array"):
                        sink.put_array("pixels", pix)
            if self.device == "cuda":
                torch.cuda.synchronize()
        finally:
            for hk in hooks:
                hk.remove()
            self.vmod.F = self.F
        return {"patches": int(h * w), "rows": int(types.numel()), "grid": [h, w, lh, lw]}

    # -- what a correct implementation can be asked for ---------------------------------------------------------

    def weights_dump(self, out):
        """The copy-type weights as the reference holds them (bf16 bits): norms, delimiters, the pixel table."""
        d = os.path.join(out, "weights")
        os.makedirs(d, exist_ok=True)
        w = lambda name, t: np.save(os.path.join(d, name + ".npy"), to_u16(t.detach().to(self.torch.bfloat16)))
        for i, b in enumerate(self.vit.blocks):
            w("norm1_%02d" % i, b.norm1.weight)
            w("norm2_%02d" % i, b.norm2.weight)
        w("norm_final", self.vit.norm.weight)
        w("start", self.delim["image_start"])
        w("newline", self.delim["image_newline"])
        w("end", self.delim["image_end"])
        np.save(os.path.join(d, "pixel_table.npy"), pixel_table_np())

    def floors(self, ops, stem, op_blocks, attn_modes=("shim", "mem", "flash")):
        """From the saved operation dumps of one image: for rmsnorm, rope, swiglu and gelu torch's bf16 result against the
        float64 answer; for attention each mode against the reference's attention on the same q, k, v. Returns
        {"point": {point: stats}, "attn": {point: {mode: stats}}}; stats as vit_compare.stats."""
        import vit_compare as vc
        torch = self.torch
        root = getattr(ops, "dir", None) or os.path.join(ops, stem)
        L = lambda name: np.load(os.path.join(root, name + ".npy"))
        D = lambda name: torch.from_numpy(bf16_to_f32(L(name))).to(self.device).double()
        back = lambda t: to_u16(t.float().to(torch.bfloat16))
        n_heads = self.args.vision_n_heads
        hd = self.args.vision_dim // n_heads
        out = {"point": {}, "attn": {}}

        def rms(x, w):
            x = x.double()
            return w.double() * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6))

        def put(name, got_name, want):
            out["point"][name] = vc.stats(L(got_name), back(want))

        def rope(q, cos, sin):
            q = q.view(q.shape[0], n_heads, hd)
            x1, x2 = q.chunk(2, dim=-1)
            c, s = cos.unsqueeze(1), sin.unsqueeze(1)
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).reshape(q.shape[0], -1)

        cos = torch.from_numpy(L("rope_cos")).to(self.device).double()
        sin = torch.from_numpy(L("rope_sin")).to(self.device).double()
        for b in op_blocks:
            p = "block%02d." % b
            blk = self.vit.blocks[b]
            put(p + "norm1_out", p + "norm1_out", rms(D(p + "x_in"), blk.norm1.weight))
            put(p + "norm2_out", p + "norm2_out", rms(D(p + "attn_add"), blk.norm2.weight))
            qkv = D(p + "qkv")
            dm = n_heads * hd
            put(p + "q_rope", p + "q_rope", rope(qkv[:, :dm], cos, sin))
            put(p + "k_rope", p + "k_rope", rope(qkv[:, dm:2 * dm], cos, sin))
            w1 = D(p + "w1_out")
            gate, up = w1.chunk(2, dim=-1)
            put(p + "swiglu_out", p + "swiglu_out", torch.nn.functional.silu(gate) * up)
            # attention: the same q, k, v in every mode
            n = qkv.shape[0]
            q = torch.from_numpy(bf16_to_f32(L(p + "q_rope"))).to(self.device).to(torch.bfloat16).view(n, n_heads, hd).transpose(0, 1)
            k = torch.from_numpy(bf16_to_f32(L(p + "k_rope"))).to(self.device).to(torch.bfloat16).view(n, n_heads, hd).transpose(0, 1)
            v = torch.from_numpy(bf16_to_f32(L(p + "qkv")[:, 2 * dm:])).to(self.device).to(torch.bfloat16).view(n, n_heads, hd).transpose(0, 1)
            ref_o = L(p + "attn_o")
            per = {}
            with torch.inference_mode():
                for mode in attn_modes:
                    if mode in ("mem", "flash") and self.device == "cpu":
                        continue  # the fused backends exist on the GPU only
                    o = self._shim(mode).sdpa(q, k, v)
                    per[mode] = vc.stats(to_u16(o.transpose(0, 1).reshape(n, -1)), ref_o)
            out["attn"][p + "attn_o"] = per
        put("norm", "norm", rms(D("norm_in"), self.vit.norm.weight))
        put("g", "g", torch.nn.functional.gelu(D("a1_out")))
        return out

    def gemm_random(self, out, rows=512, seed=1):
        """Random data of every matrix product's shape and cuBLAS's (torch's) result, for the floor of vit::gemm: written to
        OUT/gemm_random/<shape>/{a,w,b,out}.npy (b absent for the products without a bias). The weights are uniform in
        +-sqrt(3 / K) as the engine's tests draw them, the inputs unit-scale; patch_embed's K is the padded 640."""
        torch = self.torch
        rng = np.random.default_rng(seed)
        done = {}
        for name, (N, K, bias) in GEMM_SHAPES.items():
            k_real = 588 if name == "patch_embed" else K
            a = np.zeros((rows, K), np.float32)
            a[:, :k_real] = rng.standard_normal((rows, k_real))
            w = np.zeros((N, K), np.float32)
            w[:, :k_real] = (rng.random((N, k_real), dtype=np.float32) * 2 - 1) * np.sqrt(3.0 / k_real)
            b = (rng.standard_normal(N).astype(np.float32) * 0.05) if bias else None
            ta, tw = (torch.from_numpy(x).to(self.device).to(torch.bfloat16) for x in (a, w))
            tb = torch.from_numpy(b).to(self.device).to(torch.bfloat16) if bias else None
            with torch.inference_mode():
                o = torch.nn.functional.linear(ta, tw, tb)
            d = os.path.join(out, "gemm_random", name)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "a.npy"), to_u16(ta))
            np.save(os.path.join(d, "w.npy"), to_u16(tw))
            if bias:
                np.save(os.path.join(d, "b.npy"), to_u16(tb))
            np.save(os.path.join(d, "out.npy"), to_u16(o))
            done[name] = [rows, N, K]
        return done


def list_images(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.endswith((".png", ".jpg", ".jpeg")) and not f.startswith("frame_"))
    return [path]


def is_full(stem, patches, full_names, full_patches):
    return stem in full_names or patches <= full_patches


def merge_floors(path, stem, fl):
    cur = json.load(open(path)) if os.path.exists(path) else {"point": {}, "attn": {}}
    cur["point"][stem], cur["attn"][stem] = fl["point"], fl["attn"]
    json.dump(cur, open(path, "w"), indent=1)


def dump_tree(m, items, out, modes, full_names=(), full_patches=4096, ops_stems=(), op_blocks=DEFAULT_OP_BLOCKS,
              floors=False, gemm_rows=None, model_name=""):
    """Everything the docstring lists, for `items` = [(stem, prepared)]: the free-running points of every mode, the operation
    dumps and floors of the stems in ops_stems (mode ref), the weights and the random gemm data (gemm_rows rows)."""
    ops_stems, ops_manifest = set(ops_stems), {}
    if ops_stems or gemm_rows:
        m.weights_dump(out)
    if gemm_rows:
        print("gemm_random:", m.gemm_random(out, rows=gemm_rows), flush=True)
    for mode in modes:
        root = os.path.join(out, mode)
        manifest = {"attn": mode, "model": model_name, "stride": STRIDE, "images": OrderedDict()}
        for stem, prep in items:
            n = int(prep[2][0] * prep[2][1])
            full = is_full(stem, n, set(full_names), full_patches)
            sink = SaveSink(root, stem, full)
            ops = OpsSink(os.path.join(out, "ops"), stem, op_blocks) if mode == "ref" and stem in ops_stems else None
            t1 = time.time()
            info = m.run(prep, mode, Tee(sink, ops) if ops else sink)
            if mode == modes[0] and len(prep) > 3:  # the u8 image every image's free run starts from (vit_teacher reads it)
                os.makedirs(os.path.join(out, "ops", stem), exist_ok=True)
                np.save(os.path.join(out, "ops", stem, "pixels.npy"), prep[3])
            info.update(full=full, points=sink.points)
            manifest["images"][stem] = info
            if ops:
                ops_manifest[stem] = {"op_blocks": list(op_blocks), "n": n, "grid": info["grid"], "points": list(ops.points)}
                if floors:
                    merge_floors(os.path.join(out, "floors.json"), stem, m.floors(ops, stem, op_blocks))
            print("%s %s: %d patches, %d rows, %s, %d points%s in %.1f s" % (
                mode, stem, info["patches"], info["rows"], "whole" if full else "every %dth row" % STRIDE, len(sink.points),
                (" + %d operation points" % len(ops.points)) if ops else "", time.time() - t1), flush=True)
        with open(os.path.join(root, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print("MANIFEST.json written in", root, flush=True)
    if ops_manifest:
        with open(os.path.join(out, "ops", "MANIFEST.json"), "w") as f:
            json.dump(ops_manifest, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MODEL_SNAPSHOT", "/model"), help="the checkpoint folder (has inference/)")
    p.add_argument("--images", required=True, help="a folder (frame_*.png skipped) or one file")
    p.add_argument("--out", required=True)
    p.add_argument("--attn", required=True, help="comma list of: " + ", ".join(MODES))
    p.add_argument("--full", default="", help="comma list of image stems saved whole whatever their size")
    p.add_argument("--full-patches", type=int, default=4096, help="images of at most this many patches are saved whole")
    p.add_argument("--ops", default="", help="comma list of image stems whose operations are dumped whole (mode ref)")
    p.add_argument("--op-blocks", default=",".join(str(b) for b in DEFAULT_OP_BLOCKS))
    p.add_argument("--floors", action="store_true", help="measure the floors of the dumped operations (needs --ops)")
    p.add_argument("--gemm-random", action="store_true", help="random data of every product's shape with cuBLAS's output")
    a = p.parse_args()
    modes = [x for x in a.attn.split(",") if x]
    for m_ in modes:
        if m_ not in MODES:
            p.error("unknown attention mode " + m_)
    t0 = time.time()
    m = Model(a.model)
    print("loaded in %.1f s" % (time.time() - t0), flush=True)
    items = [(os.path.splitext(os.path.basename(path))[0], m.prepare(path)) for path in list_images(a.images)]
    dump_tree(m, items, a.out, modes, set(x for x in a.full.split(",") if x), a.full_patches,
              [x for x in a.ops.split(",") if x], [int(x) for x in a.op_blocks.split(",") if x], a.floors,
              512 if a.gemm_random else None, a.model)


if __name__ == "__main__":
    main()
