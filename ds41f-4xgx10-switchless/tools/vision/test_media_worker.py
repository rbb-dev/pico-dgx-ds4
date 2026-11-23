"""CPU tests of the media worker against the checkpoint's reference code; no model, no GPU, no network.
The reference side needs torch (any 2.x CPU wheel); the worker itself must not import it.

    python3 -m unittest discover -s tools/vision -p test_media_worker.py -v

grid_port: the worker's ported grid functions give the reference's answers over a sweep of sizes.
pixels_equal_reference: the worker's padded u8 image, through the engine's 256-entry pixel table and its
patch order (engine/vision/vit_weights.cpp `pixel_table`, engine/vision/vit_kernels.cu `k_patchify`, both
ported below), is the reference's `load_image` patches bit for bit.
"""
import base64
import importlib.util
import io
import itertools
import json
import os
import random
import socket
import struct
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from test_encoder_limits import ask_over_the_wire
from test_encoder_limits import image as stand_in_image
from test_encoder_limits import load_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.environ.get("VISION_REF", os.path.join(HERE, "..", "..", "ref"))


def reference():
    """The checkpoint's image_processor.py, loaded from its file (it imports torch at its line 19)."""
    spec = importlib.util.spec_from_file_location("reference_image_processor", os.path.join(REF, "inference", "image_processor.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config(**over):
    with open(os.path.join(REF, "inference", "config.json")) as f:
        cfg = json.load(f)
    cfg.update(over)
    return SimpleNamespace(**cfg)


def worker():
    """The worker as production imports it: a real config directory (the reference's inference/config.json)."""
    sys.modules.pop("media_worker", None)
    argv, sys.argv = sys.argv, ["media_worker.py", "--model", REF]
    sys.path.insert(0, HERE)
    try:
        import media_worker
    finally:
        sys.argv = argv
        sys.path.remove(HERE)
    return media_worker


# ---- the engine's pixel path, ported -----------------------------------------------------------------------------

def to_bf16_bits(f32):
    """float32 -> bf16 bits, round to nearest even (vit_weights.cpp `to_bf16`; the inputs are finite)."""
    u = np.asarray(f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def pixel_table():
    """vit_weights.cpp `pixel_table`: bf16((float(v) / 255.0f - 0.5f) / 0.5f), every step in float32."""
    x = np.arange(256, dtype=np.float32) / np.float32(255.0)
    return to_bf16_bits((x - np.float32(0.5)) / np.float32(0.5))


def engine_patchify(px, nh, nw, table):
    """vit_kernels.cu `k_patchify`: u8 [nh*14][nw*14][3] -> bf16 bits [n][640], (channel, row, column) order,
    columns 588..639 zero. Index arithmetic as the kernel's."""
    n = nh * nw
    i = np.arange(n * 640)
    p, k = i // 640, i % 640
    ok = k < 588
    c, rem = k // 196, k % 196
    y = (p // nw) * 14 + rem // 14
    x = (p % nw) * 14 + rem % 14
    flat = px.reshape(-1)
    idx = np.where(ok, (y * nw * 14 + x) * 3 + c, 0)
    return np.where(ok, table[flat[idx]], 0).astype(np.uint16).reshape(n, 640)


# ---- tests --------------------------------------------------------------------------------------------------------

SIZES = sorted(set(
    list(itertools.product([1, 2, 13, 14, 15, 28, 42, 100, 307, 544, 543, 545, 1000, 4096], repeat=2))
    + [(1, 100000), (100000, 1), (3, 20000), (20000, 3), (1, 1), (14, 100000), (100000, 14), (40000, 40000), (6300, 6300),
       (9999, 41), (41, 9999), (30000, 20), (20, 30000), (36, 5000), (5000, 36)]
    + [(random.Random(7).randint(1, 8000), random.Random(8).randint(1, 8000)) for _ in range(1)]
    + [(r.randint(1, 6000), r.randint(1, 6000)) for r in [random.Random(11)] for _ in range(1500)]
    + [(r.randint(1, 60), r.randint(1000, 30000)) for r in [random.Random(12)] for _ in range(300)]
    + [(r.randint(1000, 30000), r.randint(1, 60)) for r in [random.Random(13)] for _ in range(300)]
))


class GridPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.mw = reference(), worker()

    def test_plan_over_the_size_sweep(self):
        for over in ({}, {"vision_max_wh_ratio": 4}, {"vision_max_wh_ratio": 200.0}, {"vision_max_n_token": 64},
                     {"vision_max_n_token": 5}):
            args = config(**over)
            for w, h in SIZES:
                self.assertEqual(self.mw.plan_image_grid(w, h, args), self.ref.plan_image_grid(w, h, args), (over, w, h))

    def test_the_helpers_one_by_one(self):
        r = random.Random(5)
        for _ in range(3000):
            h, w = r.randint(1, 30000), r.randint(1, 30000)
            budget, p, d = r.choice([1024, 64, 7, 3000]), r.choice([14, 16]), r.choice([2, 3])
            if budget >= 4:
                self.assertEqual(self.mw.solve_resize_ratio(h, w, p, d, budget), self.ref.solve_resize_ratio(h, w, p, d, budget), (h, w, budget))
            bh, bw = -(-h // p) * p, -(-w // p) * p
            self.assertEqual(self.mw.llm_grid(bh, bw, p, d), self.ref.llm_grid(bh, bw, p, d))
            self.assertEqual(self.mw.num_image_tokens(h % 97 + 1, w % 89 + 1), self.ref.num_image_tokens(h % 97 + 1, w % 89 + 1))
            if budget >= 4:
                self.assertEqual(self.mw.safe_resize(h, w, bh, bw, p, d, budget), self.ref.safe_resize(h, w, bh, bw, p, d, budget))

    def test_strips_collapse_to_one_row_or_one_column(self):
        args = config()
        for w, h in ((100000, 1), (1, 100000), (20000, 3)):
            got = self.mw.plan_image_grid(w, h, args)
            self.assertEqual(got, self.ref.plan_image_grid(w, h, args))
            self.assertTrue(got[0] == 1 or got[1] == 1, (w, h, got))

    def test_a_wrong_port_is_seen(self):
        """Positive control: the sweep detects a port with one changed constant."""
        args = config()
        real = self.mw.solve_resize_ratio
        self.mw.solve_resize_ratio = lambda h, w, p, d, m: real(h, w, p, d, m + 1)
        try:
            self.assertTrue(any(self.mw.plan_image_grid(w, h, args) != self.ref.plan_image_grid(w, h, args) for w, h in SIZES))
        finally:
            self.mw.solve_resize_ratio = real


def pictures():
    r = np.random.RandomState(3)
    out = {}
    out["noise_301x517"] = Image.fromarray(r.randint(0, 256, (517, 301, 3), dtype=np.uint8))
    out["noise_700x700"] = Image.fromarray(r.randint(0, 256, (700, 700, 3), dtype=np.uint8))
    out["tiny_5x7"] = Image.fromarray(r.randint(0, 256, (7, 5, 3), dtype=np.uint8))
    out["large_2200x1700"] = Image.fromarray(r.randint(0, 256, (1700, 2200, 3), dtype=np.uint8))
    out["tall_strip_20x3000"] = Image.fromarray(r.randint(0, 256, (3000, 20, 3), dtype=np.uint8))
    out["wide_strip_3000x20"] = Image.fromarray(r.randint(0, 256, (20, 3000, 3), dtype=np.uint8))
    out["gradient_640x480"] = Image.fromarray(np.tile(np.arange(640, dtype=np.uint8) * 0 + np.linspace(0, 255, 640).astype(np.uint8), (480, 1))[..., None].repeat(3, 2))
    out["gray_L_333x222"] = Image.fromarray(r.randint(0, 256, (222, 333), dtype=np.uint8), "L")
    out["rgba_400x300"] = Image.fromarray(r.randint(0, 256, (300, 400, 4), dtype=np.uint8), "RGBA")
    out["palette_P_256x256"] = Image.fromarray(r.randint(0, 256, (256, 256), dtype=np.uint8)).convert("P")
    out["all_256_values"] = Image.fromarray(np.arange(256 * 3 * 4, dtype=np.uint32).reshape(32, 32, 3).astype(np.uint8))
    return out


def png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


class PixelsEqualReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.mw = reference(), worker()
        cls.table = pixel_table()

    def test_the_table_is_the_reference_arithmetic_for_all_256_values(self):
        x = torch.arange(256, dtype=torch.float32) / 255
        want = ((x - 0.5) / 0.5).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        self.assertTrue(np.array_equal(self.table, want))
        self.assertEqual((self.table[0], self.table[255]), (0xBF80, 0x3F80))

    def check(self, raw, args):
        patches, nh, nw, lh, lw = self.ref.load_image({"data": raw}, args)
        px, rows, grid = self.mw.measure(raw, args)
        self.assertEqual(grid, (nh, nw, lh, lw))
        self.assertEqual(rows, self.ref.image_token_types(lh, lw).numel())
        self.assertEqual((px.dtype, px.shape), (np.uint8, (nh * 14, nw * 14, 3)))
        want = patches.reshape(nh * nw, 588).view(torch.int16).numpy().view(np.uint16)
        got = engine_patchify(px, nh, nw, self.table)
        self.assertTrue(np.array_equal(got[:, :588], want), "patches differ")
        self.assertFalse(got[:, 588:].any(), "the padding columns are not zero")

    def test_every_picture(self):
        args = config()
        for name, im in pictures().items():
            with self.subTest(name):
                self.check(png(im), args)

    def test_other_formats(self):
        args = config()
        im = pictures()["noise_301x517"]
        for fmt in ("JPEG", "BMP", "WEBP", "GIF"):
            b = io.BytesIO()
            im.convert("RGB").save(b, fmt)
            with self.subTest(fmt):
                self.check(b.getvalue(), args)

    def test_the_resize_branch_for_extreme_ratios(self):
        args = config(vision_max_wh_ratio=4)
        r = np.random.RandomState(9)
        for w, h in ((900, 100), (500, 500), (130, 600)):
            with self.subTest((w, h)):
                self.check(png(Image.fromarray(r.randint(0, 256, (h, w, 3), dtype=np.uint8))), args)

    def test_a_wrong_table_is_seen(self):
        """Positive control: one table entry off by one bit fails the comparison."""
        args = config()
        raw = png(pictures()["all_256_values"])
        bad = self.table.copy()
        bad[128] ^= 1
        patches, nh, nw, *_ = self.ref.load_image({"data": raw}, args)
        px, _, _ = self.mw.measure(raw, args)
        want = patches.reshape(nh * nw, 588).view(torch.int16).numpy().view(np.uint16)
        self.assertFalse(np.array_equal(engine_patchify(px, nh, nw, bad)[:, :588], want))


RUN_DUMPS = os.environ.get("VISION_REF_DUMPS", "")


@unittest.skipUnless(bool(RUN_DUMPS) and os.path.isdir(os.path.join(RUN_DUMPS, "dumps", "ops")), "the reference run's images and dumps are not at VISION_REF_DUMPS")
class F0StemsTests(unittest.TestCase):
    """The reference's own tensors from the reference run (the pico-vision container): `pixels` (the padded u8 image, every
    stem) and `patches_in` (its bf16 patches as u16 bits, the two operation stems), against the worker's output."""

    @classmethod
    def setUpClass(cls):
        cls.mw, cls.table = worker(), pixel_table()

    def test_padded_pixels_are_the_references_for_every_stem(self):
        stems = sorted(d for d in os.listdir(os.path.join(RUN_DUMPS, "dumps", "ops")) if os.path.exists(os.path.join(RUN_DUMPS, "dumps", "ops", d, "pixels.npy")))
        self.assertGreaterEqual(len(stems), 7)
        for stem in stems:
            with self.subTest(stem):
                with open(os.path.join(RUN_DUMPS, "images", stem + ".png"), "rb") as f:
                    px, rows, grid = self.mw.measure(f.read())
                want = np.load(os.path.join(RUN_DUMPS, "dumps", "ops", stem, "pixels.npy"))
                self.assertEqual((px.shape, px.dtype), (want.shape, want.dtype))
                self.assertTrue(np.array_equal(px, want))

    def test_patches_through_the_engines_table_are_the_references(self):
        for stem in ("noise_544", "noise_768"):
            with self.subTest(stem):
                with open(os.path.join(RUN_DUMPS, "images", stem + ".png"), "rb") as f:
                    px, rows, (nh, nw, lh, lw) = self.mw.measure(f.read())
                want = np.load(os.path.join(RUN_DUMPS, "dumps", "ops", stem, "patches_in.npy"))
                got = engine_patchify(px, nh, nw, self.table)
                self.assertEqual(want.shape, (nh * nw, 588))
                self.assertTrue(np.array_equal(got[:, :588], want))
                self.assertFalse(got[:, 588:].any())


class ImagePixelLimitTests(unittest.TestCase):
    """--max-image-pixels means what it says: an image over it is refused from its header, before any decode."""

    @classmethod
    def setUpClass(cls):
        cls.mw = worker()

    def setUp(self):
        self.saved = (self.mw.a.max_image_pixels, Image.MAX_IMAGE_PIXELS)
        self.mw.a.max_image_pixels = Image.MAX_IMAGE_PIXELS = 2_000_000

    def tearDown(self):
        self.mw.a.max_image_pixels, Image.MAX_IMAGE_PIXELS = self.saved

    def png_of(self, w, h):
        return png(Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)))

    def test_at_the_limit_is_taken_and_over_it_is_refused(self):
        px, rows, grid = self.mw.measure(self.png_of(2000, 1000))  # 2,000,000 px
        self.assertGreater(rows, 0)
        for w, h in ((2000, 1001), (1500, 1500), (3000, 1000)):  # between 1x and 2x PIL's limit, and over 2x
            with self.assertRaisesRegex(Exception, r"exceeds limit of 2000000 pixels"):
                self.mw.measure(self.png_of(w, h))

    def test_the_refusal_decodes_nothing(self):
        raw = self.png_of(1500, 1500)
        calls = []
        real = Image.Image.convert
        Image.Image.convert = lambda self, *a, **k: calls.append(1) or real(self, *a, **k)
        try:
            with self.assertRaises(Exception):
                self.mw.plan(raw)
        finally:
            Image.Image.convert = real
        self.assertEqual(calls, [])

    def test_the_reply_is_an_error_with_the_message(self):
        with self.assertRaisesRegex(Exception, "exceeds limit of 2000000 pixels"):
            list(self.mw.handle({"images": [{"data": base64.b64encode(self.png_of(1500, 1500)).decode()}]}))


class WorkerShapeTests(unittest.TestCase):
    def test_no_torch_in_the_worker(self):
        code = ("import sys; sys.argv=['media_worker.py','--model',%r]; sys.path.insert(0,%r); import media_worker; "
                "bad=[m for m in sys.modules if m.split('.')[0] in ('torch','vision','image_processor')]; "
                "print(bad); sys.exit(1 if bad else 0)") % (REF, HERE)
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=120, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "[]"), r.stderr[-400:])

    def test_the_reply_carries_rows_hash_grid_and_the_padded_image(self):
        mw = worker()
        raw = png(pictures()["noise_301x517"])
        srv = mw.S(("127.0.0.1", 0), mw.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            recs, data, last = ask_over_the_wire(mw, srv.server_address, {"images": [{"data": base64.b64encode(raw).decode()}] * 2})
        finally:
            srv.shutdown()
            srv.server_close()
        px, rows, grid = mw.measure(raw, config())
        self.assertEqual(last, {"ok": True, "images": 2})
        import hashlib
        h = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
        self.assertEqual(len(recs), 2)
        for rec in recs:
            self.assertEqual((rec["rows"], rec["hash"], tuple(rec["grid"])), (rows, h, grid))
        self.assertEqual(data, px.tobytes() * 2)


if __name__ == "__main__":
    unittest.main()
