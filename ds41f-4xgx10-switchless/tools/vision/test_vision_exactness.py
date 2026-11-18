"""CPU tests of the vision exactness tools (make_test_images.py, dump_ref.py's operation dumps, vit_compare.py).

Needs numpy, Pillow and (for the dump tool) torch on the CPU; no GPU, no checkpoint. Run:
    python3 -m unittest test_vision_exactness -v
The compare logic runs on synthetic dump trees written here in the layout vit_compare.py documents; the dump tool's
hooks run on a tiny random ViT of the reference's own classes (inference/vision.py).
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vit_compare as vc  # noqa: E402

BLOCKS = 32
COLS = 64
OP_BLOCKS = [0, 1, 2, 7, 16]
STEM = "synth"
N, M = 48, 6   # patches, aligner rows of the synthetic image


def to_bf16_bits(f):
    """float32 -> bf16 bits (round to nearest even), as uint16."""
    f = np.ascontiguousarray(f, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def from_bf16_bits(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def noisy(rng, ref_u16, rel, heavy_rows=0, heavy_rel=1.0):
    """ref plus a relative perturbation per element (rel), a few rows with a big one (the late-tower tails)."""
    x = from_bf16_bits(ref_u16)
    d = rng.standard_normal(x.shape).astype(np.float32) * rel * np.abs(x).mean()
    y = x + d
    for r in rng.choice(x.shape[0], heavy_rows, replace=False) if heavy_rows else []:
        y[r] += rng.standard_normal(x.shape[1]).astype(np.float32) * heavy_rel * np.abs(x[r]).mean()
    return to_bf16_bits(y)


def rand_bf16(rng, shape, scale=1.0):
    return to_bf16_bits(rng.standard_normal(shape).astype(np.float32) * scale)


def save(root, rel, arr):
    p = os.path.join(root, rel + ".npy")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, arr)


FREE_POINTS = ["patch_embed"] + ["block%02d.%s" % (i, k) for i in range(BLOCKS) for k in ("attn_add", "mlp_add")] + ["norm"]


def block_ops(i):
    """(point, kind, shape) of a block's teacher-forced dumps; the same names vit_compare.BLOCK_OPS holds."""
    p = "block%02d." % i
    return [(p + "x_in", None, (N, COLS)), (p + "norm1_out", "point", (N, COLS)), (p + "qkv", "gemm", (N, 3 * COLS)),
            (p + "q_rope", "point", (N, COLS)), (p + "k_rope", "point", (N, COLS)), (p + "attn_o", "attn", (N, COLS)),
            (p + "wo_out", "gemm", (N, COLS)), (p + "attn_add", "exact", (N, COLS)), (p + "norm2_out", "point", (N, COLS)),
            (p + "w1_out", "gemm", (N, 2 * COLS)), (p + "swiglu_out", "point", (N, COLS)), (p + "w2_out", "gemm", (N, COLS)),
            (p + "mlp_add", "exact", (N, COLS))]


GLOBAL_OPS = [("patch_embed", "gemm", (N, COLS)), ("norm", "point", (N, COLS)), ("a1_out", "gemm", (M, COLS)),
              ("g", "point", (M, COLS)), ("aligner", "gemm", (M, COLS))]
COPY_OPS = [("unfold", (M, 9 * COLS)), ("span", (M + 3, COLS))]
GEMM_SHAPES = ["patch_embed", "wqkv", "wo", "w1", "w2", "aligner1", "aligner2"]
WEIGHT_NAMES = ["pixel_table", "start", "newline", "end", "norm_final"] + ["norm1_%02d" % i for i in range(BLOCKS)] \
    + ["norm2_%02d" % i for i in range(BLOCKS)]


class Synth:
    """A synthetic dump tree (what dump_ref.py writes in the container) and a matching tree of ours (what vit_teacher
    writes), under a temp folder. `ours_rel` is how far ours is from the reference at the teacher-forced points;
    the null modes sit `null_rel` from the reference at the free-running points."""

    def __init__(self, root, null_rel=1e-2, ours_rel=0.5e-2, op_rel=1.5e-3, seed=7):
        self.root, self.rng = root, np.random.default_rng(seed)
        self.dumps, self.ours = os.path.join(root, "dumps"), os.path.join(root, "ours")
        rng = self.rng
        manifest = {"stride": 16, "modes": ["ref", "shim", "mem", "flash"], "images": {STEM: {
            "n": N, "grid": [6, 8, 2, 3], "full": True, "op_blocks": OP_BLOCKS, "points": {}}}}
        # free-running dumps: ref, three nulls, ours
        self.free_ref = {p: rand_bf16(rng, (N, COLS)) for p in FREE_POINTS}
        for mode, rel in (("ref", 0), ("shim", null_rel), ("mem", 0.7 * null_rel), ("flash", 0.9 * null_rel)):
            for p, a in self.free_ref.items():
                save(self.dumps, "%s/%s/%s" % (mode, STEM, p), a if mode == "ref" else noisy(rng, a, rel, heavy_rows=1, heavy_rel=1.0))
        for p, a in self.free_ref.items():
            save(self.ours, "free/%s/%s" % (STEM, p), noisy(rng, a, ours_rel, heavy_rows=1, heavy_rel=1.0))
        for mode in manifest["modes"]:
            os.makedirs(os.path.join(self.dumps, mode), exist_ok=True)
            json.dump({"attn": mode, "images": {STEM: {"full": True}}}, open(os.path.join(self.dumps, mode, "MANIFEST.json"), "w"))
        # teacher-forced dumps
        self.ops_ref, floors_point, floors_attn = {}, {}, {}
        ops = list(GLOBAL_OPS)
        for i in OP_BLOCKS:
            ops += [(p, k, s) for p, k, s in block_ops(i)]
        for p, kind, shape in ops:
            ref = rand_bf16(rng, shape)
            self.ops_ref[p] = ref
            save(self.dumps, "ops/%s/%s" % (STEM, p), ref)
            if kind is None:
                continue
            if kind == "exact":
                ours = ref.copy()
            else:
                ours = noisy(rng, ref, op_rel)
            save(self.ours, "teacher/%s/%s" % (STEM, p), ours)
            if kind == "point":
                floors_point[p] = vc.stats(noisy(rng, ref, op_rel), ref)
            if kind == "attn":
                floors_attn[p] = {m: vc.stats(noisy(rng, ref, op_rel), ref) for m in ("shim", "mem", "flash")}
        for p, shape in COPY_OPS:
            ref = rand_bf16(rng, shape)
            save(self.dumps, "ops/%s/%s" % (STEM, p), ref)
            save(self.ours, "teacher/%s/%s" % (STEM, p), ref.copy())
        for name in ("rope_cos", "rope_sin"):
            ref = rng.standard_normal((N, 32)).astype(np.float32)
            save(self.dumps, "ops/%s/%s" % (STEM, name), ref)
            save(self.ours, "teacher/%s/%s" % (STEM, name), ref.copy())
        patches = rand_bf16(rng, (N, 588))
        save(self.dumps, "ops/%s/patches_in" % STEM, patches)
        save(self.ours, "teacher/%s/patches" % STEM, np.concatenate([patches, np.zeros((N, 52), np.uint16)], axis=1))
        json.dump({"point": {STEM: floors_point}, "attn": {STEM: floors_attn}}, open(os.path.join(self.dumps, "floors.json"), "w"))
        json.dump({STEM: {"op_blocks": OP_BLOCKS, "n": N}}, open(os.path.join(self.dumps, "ops", "MANIFEST.json"), "w"))
        # weights
        for name in WEIGHT_NAMES:
            w = rand_bf16(rng, (1024,) if name.startswith("norm") else (256,))
            save(self.dumps, "weights/" + name, w)
            save(self.ours, "weights/" + name, w.copy())
        # random gemm data: cuBLAS output and ours
        for shp in GEMM_SHAPES:
            a, w, b = rand_bf16(rng, (16, 64)), rand_bf16(rng, (32, 64), 0.2), rand_bf16(rng, (32,), 0.05)
            out = rand_bf16(rng, (16, 32))
            for nm, arr in (("a", a), ("w", w), ("b", b), ("out", out)):
                save(self.dumps, "gemm_random/%s/%s" % (shp, nm), arr)
            save(self.ours, "gemm_random/%s/out" % shp, noisy(rng, out, op_rel))

    def edit(self, side, rel, fn):
        """Replace a saved array in place by fn(array); returns the path."""
        p = os.path.join(self.dumps if side == "dumps" else self.ours, rel + ".npy")
        a = np.load(p)
        np.save(p, fn(a))
        return p


def shuffled(a):
    return np.roll(a, 1, axis=0) if a.dtype != np.float32 else np.roll(a, 1, axis=0)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="f0test")
        self.s = Synth(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_compare(self, **kw):
        return vc.run(self.s.dumps, self.s.ours, **kw)


class StatsTests(unittest.TestCase):
    def test_identical_is_zero_and_rotated_is_large(self):
        rng = np.random.default_rng(1)
        a = rand_bf16(rng, (100, 64))
        st = vc.stats(a, a)
        self.assertEqual(st["rel_frob"], 0.0)
        self.assertEqual(st["med_row_rel"], 0.0)
        self.assertEqual(st["share_not_equal"], 0.0)
        self.assertAlmostEqual(st["cos_p01"], 1.0)
        rot = vc.stats(np.roll(a, 1, axis=0), a)
        self.assertGreater(rot["med_row_rel"], 0.5)
        self.assertGreater(rot["rel_frob"], 0.5)
        self.assertGreater(rot["share_over_1_step"], 0.5)

    def test_robust_to_one_heavy_row_where_the_max_is_not(self):
        rng = np.random.default_rng(2)
        a = rand_bf16(rng, (200, 64))
        b = a.copy()
        b[3] = rand_bf16(rng, (64,))  # one row replaced
        st = vc.stats(b, a)
        self.assertLess(st["med_row_rel"], 1e-6)
        self.assertLess(st["rel_frob"], 0.15)
        self.assertGreater(st["max_row_rel"], 0.5)  # recorded, not judged

    def test_float32_tables_are_compared_by_bits(self):
        a = np.random.default_rng(3).standard_normal((8, 32)).astype(np.float32)
        self.assertTrue(vc.bytes_equal(a, a.copy()))
        b = a.copy()
        b.view(np.uint32)[2, 5] ^= 1
        self.assertFalse(vc.bytes_equal(b, a))


class GreenTests(Base):
    def test_null_passes(self):
        """GREEN: ours = the encoder's own mode (one of the nulls) against the reference passes every part."""
        for p in self.s.free_ref:
            shutil.copy(os.path.join(self.s.dumps, "shim", STEM, p + ".npy"), os.path.join(self.s.ours, "free", STEM, p + ".npy"))
        rep = self.run_compare()
        self.assertTrue(rep.ok, "\n".join(rep.failures))

    def test_correct_synthetic_passes_all_three_parts(self):
        rep = self.run_compare()
        self.assertTrue(rep.ok, "\n".join(rep.failures))
        kinds = {r.part for r in rep.results}
        self.assertEqual(kinds, {"instrument", "teacher", "free"})

    def test_later_points_are_recorded_not_judged(self):
        p = os.path.join(self.s.ours, "free", STEM, "block20.mlp_add.npy")
        np.save(p, shuffled(np.load(p)))  # a ruined late point
        rep = self.run_compare()
        self.assertTrue(rep.ok, "\n".join(rep.failures))
        rec = [r for r in rep.results if r.name.endswith("block20.mlp_add")]
        self.assertTrue(rec and not rec[0].judged)
        self.assertGreater(rec[0].stats["med_row_rel"], 0.3)  # but it is in the record


class RedTests(Base):
    def test_free_running_rows_shuffled_fails_at_each_judged_point(self):
        """RED: one tensor's rows shuffled in ours, the compare run must fail (and name the point)."""
        for point in ("patch_embed", "block00.attn_add", "block03.mlp_add", "block07.mlp_add"):
            with self.subTest(point=point):
                s = Synth(tempfile.mkdtemp(prefix="f0red"))
                try:
                    p = os.path.join(s.ours, "free", STEM, point + ".npy")
                    np.save(p, shuffled(np.load(p)))
                    rep = vc.run(s.dumps, s.ours)
                    self.assertFalse(rep.ok)
                    self.assertTrue(any(point in f for f in rep.failures), rep.failures)
                finally:
                    shutil.rmtree(s.root, ignore_errors=True)

    def test_shuffled_reference_dump_fails_too(self):
        """The same on the other side: the reference's dump with one tensor's rows shuffled."""
        p = os.path.join(self.s.dumps, "ref", STEM, "block01.mlp_add.npy")
        np.save(p, shuffled(np.load(p)))
        rep = self.run_compare()
        self.assertFalse(rep.ok)

    def test_teacher_forced_shuffled_output_fails_for_every_kind(self):
        for point in ("block00.qkv", "block01.norm1_out", "block02.attn_o", "block07.attn_add", "patch_embed", "g"):
            with self.subTest(point=point):
                s = Synth(tempfile.mkdtemp(prefix="f0red"))
                try:
                    p = os.path.join(s.ours, "teacher", STEM, point + ".npy")
                    np.save(p, shuffled(np.load(p)))
                    rep = vc.run(s.dumps, s.ours)
                    self.assertFalse(rep.ok)
                    self.assertTrue(any(point in f for f in rep.failures), rep.failures)
                finally:
                    shutil.rmtree(s.root, ignore_errors=True)

    def test_copy_ops_need_every_bit(self):
        for point in ("unfold", "span", "rope_cos", "rope_sin"):
            with self.subTest(point=point):
                s = Synth(tempfile.mkdtemp(prefix="f0red"))
                try:
                    p = os.path.join(s.ours, "teacher", STEM, point + ".npy")
                    a = np.load(p)
                    a.view(np.uint16 if a.dtype == np.uint16 else np.uint32).reshape(-1)[7] ^= 1  # one bit
                    np.save(p, a)
                    rep = vc.run(s.dumps, s.ours)
                    self.assertFalse(rep.ok)
                    self.assertTrue(any(point in f for f in rep.failures), rep.failures)
                finally:
                    shutil.rmtree(s.root, ignore_errors=True)

    def test_patchify_tail_must_be_zero_and_head_equal(self):
        p = os.path.join(self.s.ours, "teacher", STEM, "patches.npy")
        a = np.load(p)
        a[4, 600] = 1
        np.save(p, a)
        self.assertFalse(self.run_compare().ok)

    def test_norm_weight_one_bit_fails(self):
        p = os.path.join(self.s.ours, "weights", "norm2_17.npy")
        a = np.load(p)
        a[100] ^= 1
        np.save(p, a)
        rep = self.run_compare()
        self.assertFalse(rep.ok)
        self.assertTrue(any("norm2_17" in f for f in rep.failures))

    def test_missing_file_fails_never_skips(self):
        os.remove(os.path.join(self.s.ours, "teacher", STEM, "block16.w2_out.npy"))
        rep = self.run_compare()
        self.assertFalse(rep.ok)
        self.assertTrue(any("missing" in f and "block16.w2_out" in f for f in rep.failures))

    def test_gemm_beyond_its_random_data_floor_fails(self):
        s = self.s
        # ours off the reference by 20x the floor's noise: above any margin
        p = os.path.join(s.ours, "teacher", STEM, "block02.w1_out.npy")
        rng = np.random.default_rng(5)
        np.save(p, noisy(rng, np.load(os.path.join(s.dumps, "ops", STEM, "block02.w1_out.npy")), 3e-2))
        rep = self.run_compare()
        self.assertFalse(rep.ok)
        self.assertTrue(any("block02.w1_out" in f for f in rep.failures))

    def test_attention_judged_against_the_largest_pair(self):
        # a floor from three pairs: the line is the largest; ours at the largest pair's noise passes, at 20x fails
        rep = self.run_compare()
        self.assertTrue(rep.ok)
        p = os.path.join(self.s.ours, "teacher", STEM, "block07.attn_o.npy")
        rng = np.random.default_rng(6)
        np.save(p, noisy(rng, np.load(os.path.join(self.s.dumps, "ops", STEM, "block07.attn_o.npy")), 3e-2))
        self.assertFalse(self.run_compare().ok)

    def test_nonfinite_fails(self):
        p = os.path.join(self.s.ours, "teacher", STEM, "block00.wo_out.npy")
        a = np.load(p)
        a[0, 0] = 0x7FC0
        np.save(p, a)
        self.assertFalse(self.run_compare().ok)


class ControlTests(Base):
    """Positive controls: ours with a deliberate error must stand >= 10x the null in blocks 0-2 (or fail the byte
    test); a control that slips under makes the suite fail: the line is too loose."""

    def make_control(self, name, rel, rows_shuffled=False, points=None):
        d = os.path.join(self.tmp, "control_" + name)
        for p in points or FREE_POINTS:
            a = np.load(os.path.join(self.s.dumps, "ref", STEM, p + ".npy"))
            if p == "patch_embed" or p.startswith(("block00", "block01", "block02")):
                a = shuffled(a) if rows_shuffled else noisy(np.random.default_rng(9), a, rel)
            save(d, "free/%s/%s" % (STEM, p), a)
        return d

    def test_a_visible_control_is_detected(self):
        d = self.make_control("norm-swap", 0.0, rows_shuffled=True)
        rep = self.run_compare(controls={"norm-swap": d})
        self.assertTrue(rep.ok, "\n".join(rep.failures))
        c = [r for r in rep.results if r.part == "control"]
        self.assertTrue(c and c[0].ok)

    def test_a_control_under_ten_times_the_null_fails_the_suite(self):
        d = self.make_control("gate-up", 0.2e-2)   # as near as ours: under the null
        rep = self.run_compare(controls={"gate-up": d})
        self.assertFalse(rep.ok)
        self.assertTrue(any("too loose" in f and "gate-up" in f for f in rep.failures), rep.failures)

    def test_a_control_with_nothing_in_blocks_0_to_2_cannot_pass(self):
        d = os.path.join(self.tmp, "control_none")
        for p in FREE_POINTS:  # error only at block 20: invisible to the judged range
            a = np.load(os.path.join(self.s.dumps, "ref", STEM, p + ".npy"))
            save(d, "free/%s/%s" % (STEM, p), shuffled(a) if p.startswith("block20") else a)
        rep = self.run_compare(controls={"qkv-order": d})
        self.assertFalse(rep.ok)

    def test_a_copy_control_is_detected_by_the_byte_test(self):
        d = os.path.join(self.tmp, "control_unfold")
        a = np.load(os.path.join(self.s.dumps, "ops", STEM, "unfold.npy"))
        save(d, "teacher/%s/unfold" % STEM, shuffled(a))
        rep = self.run_compare(controls={"unfold-window": d})
        self.assertTrue(rep.ok, "\n".join(rep.failures))

    def test_a_copy_control_that_equals_the_reference_fails(self):
        d = os.path.join(self.tmp, "control_unfold_same")
        a = np.load(os.path.join(self.s.dumps, "ops", STEM, "unfold.npy"))
        save(d, "teacher/%s/unfold" % STEM, a)
        rep = self.run_compare(controls={"unfold-window": d})
        self.assertFalse(rep.ok)

    def test_unknown_control_name_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_compare(controls={"no-such-error": self.tmp})


class CliTests(Base):
    def test_exit_codes(self):
        self.assertEqual(vc.main(["--dumps", self.s.dumps, "--ours", self.s.ours, "--out", os.path.join(self.tmp, "r.json")]), 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "r.json")))
        p = os.path.join(self.s.ours, "free", STEM, "patch_embed.npy")
        np.save(p, shuffled(np.load(p)))
        self.assertNotEqual(vc.main(["--dumps", self.s.dumps, "--ours", self.s.ours]), 0)


class ImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import make_test_images as mti
        cls.mti = mti
        cls.tmp = tempfile.mkdtemp(prefix="f0img")
        cls.names = mti.make_all(cls.tmp, seed=20261121)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_fixed_sizes_and_kinds(self):
        from PIL import Image
        want = {"noise_544": (544, 544), "noise_768": (768, 768), "gradient_768": (768, 768),
                "checker_770x1022": (770, 1022), "square_1302": (1302, 1302),
                "strip_42x42882": (42882, 42), "tall_42x42882": (42, 42882)}
        for stem, size in want.items():
            self.assertIn(stem, self.names)
            self.assertEqual(Image.open(os.path.join(self.tmp, stem + ".png")).size, size, stem)

    def test_grids_match_the_plan_the_design_states(self):
        """The reference's own planner: 1,521 / 3,025 / 4,015 patches and the 3 x 3,063 strip."""
        import torch  # noqa: F401
        sys.path.insert(0, os.path.join(HERE, "..", "..", "ref", "inference"))
        import image_processor as ip
        from types import SimpleNamespace
        args = SimpleNamespace(**json.load(open(os.path.join(HERE, "..", "..", "ref", "inference", "config.json"))))
        from PIL import Image
        got = {}
        for stem in ("noise_544", "noise_768", "checker_770x1022", "strip_42x42882", "tall_42x42882"):
            w, h = Image.open(os.path.join(self.tmp, stem + ".png")).size
            lh, lw, bh, bw = ip.plan_image_grid(w, h, args)
            got[stem] = (bh // 14, bw // 14)
        self.assertEqual(got["noise_544"], (39, 39))
        self.assertEqual(got["noise_768"], (55, 55))
        self.assertEqual(got["checker_770x1022"], (73, 55))
        self.assertEqual(got["strip_42x42882"], (3, 3063))
        self.assertEqual(got["tall_42x42882"], (1533, 3))  # a column is capped at 511 token rows

    def test_same_seed_same_bytes_other_seed_other_noise(self):
        other = tempfile.mkdtemp(prefix="f0img2")
        try:
            self.mti.make_all(other, seed=20261121)
            for stem in ("noise_544", "checker_770x1022", "card_01_digits"):
                a = hashlib.sha256(open(os.path.join(self.tmp, stem + ".png"), "rb").read()).hexdigest()
                b = hashlib.sha256(open(os.path.join(other, stem + ".png"), "rb").read()).hexdigest()
                self.assertEqual(a, b, stem)
            c = tempfile.mkdtemp(prefix="f0img3")
            try:
                self.mti.make_all(c, seed=1)
                a = open(os.path.join(self.tmp, "noise_544.png"), "rb").read()
                b = open(os.path.join(c, "noise_544.png"), "rb").read()
                self.assertNotEqual(a, b)
            finally:
                shutil.rmtree(c, ignore_errors=True)
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_ten_cards_with_answers_in_the_cases_file(self):
        cases = json.load(open(os.path.join(self.tmp, "cases.json")))
        self.assertEqual(len(cases), 10)
        for name, question, answers in cases:
            self.assertTrue(os.path.exists(os.path.join(self.tmp, name)), name)
            self.assertTrue(question and answers)
            for a in answers:
                self.assertTrue(isinstance(a, (str, list)))
        # the same shape as tools/vision/cases.json (imgtest.py reads that)
        old = json.load(open(os.path.join(HERE, "cases.json")))
        self.assertEqual(type(old[0][2]), type(cases[0][2]))

    def test_the_cards_are_not_blank_and_differ(self):
        from PIL import Image
        seen = set()
        for name, _, _ in json.load(open(os.path.join(self.tmp, "cases.json"))):
            im = np.asarray(Image.open(os.path.join(self.tmp, name)).convert("L"))
            self.assertGreater(im.std(), 10, name)
            seen.add(hashlib.sha256(im.tobytes()).hexdigest())
        self.assertEqual(len(seen), 10)

    def test_the_digits_card_really_shows_its_digits(self):
        """Positive control of the generator: the answer on the digits card is drawn, not only written down."""
        from PIL import Image
        cases = {n: a for n, _, a in json.load(open(os.path.join(self.tmp, "cases.json")))}
        name = [n for n in cases if "digits" in n][0]
        im = np.asarray(Image.open(os.path.join(self.tmp, name)).convert("L"))
        self.assertGreater((im < 100).mean(), 0.01)  # dark glyph pixels exist


class DumpToolTests(unittest.TestCase):
    """dump_ref.py's operation dumps, on a tiny random ViT built from the reference's own classes (CPU)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch needed")
        import dump_ref as dr
        cls.dr = dr

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="f0dump")
        self.m = self.dr.Model.synthetic(seed=3, layers=3, device="cpu")
        # an image's patches [n, 3, 14, 14] of a 6 x 5 grid, as the reference makes them
        g = np.random.default_rng(4)
        self.pix = g.integers(0, 256, size=(6 * 14, 5 * 14, 3), dtype=np.uint8)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dump(self, attn="ref", op_blocks=(0, 2)):
        sink = self.dr.SaveSink(os.path.join(self.tmp, attn), "img", True)
        ops = self.dr.OpsSink(os.path.join(self.tmp, "ops"), "img", op_blocks)
        self.m.run(self.m.prepare_pixels(self.pix), attn, self.dr.Tee(sink, ops))
        return sink, ops

    def load(self, rel):
        return np.load(os.path.join(self.tmp, rel + ".npy"))

    def test_hooks_do_not_change_the_forward(self):
        sink, _ = self.dump("ref")
        plain = self.m.forward_plain(self.m.prepare_pixels(self.pix), "ref")
        self.assertTrue(np.array_equal(self.load("ref/img/aligner"), plain["aligner"]))
        self.assertTrue(np.array_equal(self.load("ref/img/block02.mlp_add"), plain["block02.mlp_add"]))

    def test_every_operation_input_and_output_is_dumped(self):
        self.dump("ref")
        for p in ("patches_in", "rope_cos", "rope_sin", "patch_embed", "norm_in", "norm", "unfold", "a1_out", "g", "aligner",
                  "span", "pixels"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, "ops", "img", p + ".npy")), p)
        for b in (0, 2):
            for p in ("x_in", "norm1_out", "qkv", "q_rope", "k_rope", "attn_o", "wo_out", "attn_add", "norm2_out", "w1_out",
                      "swiglu_out", "w2_out", "mlp_add"):
                self.assertTrue(os.path.exists(os.path.join(self.tmp, "ops", "img", "block%02d.%s.npy" % (b, p))), (b, p))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "ops", "img", "block01.qkv.npy")))  # only the asked blocks

    def test_the_dumps_chain_exactly(self):
        """Each operation's output is the next one's input, bit for bit: the dumps are the model's own tensors."""
        self.dump("ref")
        L = lambda p: self.load("ops/img/" + p)
        U = self.dr.bf16_to_f32
        # attn_add = x_in + wo_out  (bf16 add, one rounding)
        for b in (0, 2):
            x, wo, aa = L("block%02d.x_in" % b), L("block%02d.wo_out" % b), L("block%02d.attn_add" % b)
            want = self.dr.to_bf16_np(U(x) + U(wo))
            self.assertTrue(np.array_equal(want, aa), b)
            w2, mlp = L("block%02d.w2_out" % b), L("block%02d.mlp_add" % b)
            self.assertTrue(np.array_equal(self.dr.to_bf16_np(U(aa) + U(w2)), mlp), b)
        # block 2's input is block 1's output: not dumped; but the free-running point of block 1 equals it
        self.assertTrue(np.array_equal(L("block02.x_in"), self.load("ref/img/block01.mlp_add")))
        # the free point equals the op dump
        self.assertTrue(np.array_equal(L("block00.mlp_add"), self.load("ref/img/block00.mlp_add")))
        self.assertTrue(np.array_equal(L("patch_embed"), self.load("ref/img/patch_embed")))
        # final norm input is the last block's output
        self.assertTrue(np.array_equal(L("norm_in"), self.load("ref/img/block02.mlp_add")))
        # the rope tables are float32 and the reference's own
        self.assertEqual(L("rope_cos").dtype, np.float32)
        self.assertEqual(L("rope_cos").shape[1], self.m.args.vision_dim // self.m.args.vision_n_heads // 2)

    def test_q_rope_k_rope_are_the_attention_inputs(self):
        """Recompute the attention from the dumped q, k, v in float64: it matches attn_o to bf16 rounding."""
        self.dump("ref")
        L = lambda p: self.load("ops/img/" + p)
        U = self.dr.bf16_to_f32
        H, D = self.m.args.vision_n_heads, self.m.args.vision_dim // self.m.args.vision_n_heads
        q, k = U(L("block00.q_rope")).astype(np.float64), U(L("block00.k_rope")).astype(np.float64)
        v = U(L("block00.qkv"))[:, 2 * H * D:].astype(np.float64)
        n = q.shape[0]
        o = np.zeros((n, H * D))
        for h in range(H):
            s = q[:, h * D:(h + 1) * D] @ k[:, h * D:(h + 1) * D].T / np.sqrt(D)
            s = np.exp(s - s.max(1, keepdims=True))
            o[:, h * D:(h + 1) * D] = (s / s.sum(1, keepdims=True)) @ v[:, h * D:(h + 1) * D]
        got = U(L("block00.attn_o")).astype(np.float64)
        rel = np.linalg.norm(got - o) / np.linalg.norm(o)
        self.assertLess(rel, 1e-2)

    def test_swiglu_in_out_and_norm_pairs(self):
        self.dump("ref")
        L = lambda p: self.load("ops/img/" + p)
        U = self.dr.bf16_to_f32
        w1 = U(L("block00.w1_out")).astype(np.float64)
        inter = w1.shape[1] // 2
        gate, up = w1[:, :inter], w1[:, inter:]
        want = gate / (1 + np.exp(-gate)) * up
        got = U(L("block00.swiglu_out")).astype(np.float64)
        self.assertLess(np.linalg.norm(got - want) / np.linalg.norm(want), 1e-2)

    def test_pixels_dumped_equal_the_input_and_make_the_patches(self):
        """The u8 image the harness needs is the one the reference's patches came from (instrument check)."""
        self.dump("ref")
        pix = self.load("ops/img/pixels")
        self.assertTrue(np.array_equal(pix, self.pix))
        patches = self.load("ops/img/patches_in")   # [n, 588]
        t = self.dr.pixel_table_np()
        n_h, n_w = self.pix.shape[0] // 14, self.pix.shape[1] // 14
        want = t[self.pix].reshape(n_h, 14, n_w, 14, 3).transpose(0, 2, 4, 1, 3).reshape(n_h * n_w, 588)
        self.assertTrue(np.array_equal(want, patches))

    def test_attention_modes_other_than_ref_leave_the_op_dump_alone(self):
        self.dump("ref")
        before = hashlib.sha256(open(os.path.join(self.tmp, "ops", "img", "block00.attn_o.npy"), "rb").read()).hexdigest()
        sink = self.dr.SaveSink(os.path.join(self.tmp, "shim"), "img", True)
        self.m.run(self.m.prepare_pixels(self.pix), "shim", sink)  # no OpsSink: free-running points only
        after = hashlib.sha256(open(os.path.join(self.tmp, "ops", "img", "block00.attn_o.npy"), "rb").read()).hexdigest()
        self.assertEqual(before, after)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "shim", "img", "block00.mlp_add.npy")))

    def test_floors_are_measured_for_point_ops_and_attention(self):
        sink, ops = self.dump("ref")
        fl = self.m.floors(ops, "img", op_blocks=(0, 2), attn_modes=("shim",))
        self.assertIn("block00.norm1_out", fl["point"])
        self.assertIn("block02.swiglu_out", fl["point"])
        self.assertIn("g", fl["point"])
        self.assertIn("block00.attn_o", fl["attn"])
        self.assertIn("shim", fl["attn"]["block00.attn_o"])
        for v in fl["point"].values():
            self.assertLess(v["rel_frob"], 0.05)  # torch's op against the float64 answer: rounding-sized

    def test_gemm_random_cases_carry_cublas_output(self):
        out = self.m.gemm_random(self.tmp, rows=16, seed=1)
        d = os.path.join(self.tmp, "gemm_random", "patch_embed")
        for nm in ("a", "w", "b", "out"):
            self.assertTrue(os.path.exists(os.path.join(d, nm + ".npy")), nm)
        a = np.load(os.path.join(d, "a.npy"))
        self.assertEqual(a.shape[1], 640)  # patch_embed's K, padded as the engine holds it
        self.assertEqual(set(out), {"patch_embed", "wqkv", "wo", "w1", "w2", "aligner1", "aligner2"})


class EndToEndTests(unittest.TestCase):
    """The names and layout dump_ref.py writes are the ones vit_compare.py reads: a real dump tree of the tiny model, with
    'ours' made from it (the shim's free-running points as the encoder's; the operations equal to the reference's)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch needed")
        import dump_ref as dr
        cls.dr = dr
        cls.tmp = tempfile.mkdtemp(prefix="f0e2e")
        m = dr.Model.synthetic(seed=5, layers=8, device="cpu")  # the judged range is blocks 0-7
        pix = np.random.default_rng(8).integers(0, 256, size=(6 * 14, 8 * 14, 3), dtype=np.uint8)
        items = [("img", m.prepare_pixels(pix))]
        cls.dumps = os.path.join(cls.tmp, "dumps")
        dr.dump_tree(m, items, cls.dumps, ["ref", "shim"], full_names=["img"], ops_stems=["img"], op_blocks=[0, 1, 2],
                     floors=True, gemm_rows=8, model_name="synthetic")
        cls.items = items

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make_ours(self, name):
        ours = os.path.join(self.tmp, name)
        shutil.rmtree(ours, ignore_errors=True)
        d = self.dumps
        for root, _, files in os.walk(os.path.join(d, "shim", "img")):
            for f in files:
                if f.endswith(".npy"):
                    os.makedirs(os.path.join(ours, "free", "img"), exist_ok=True)
                    shutil.copy(os.path.join(root, f), os.path.join(ours, "free", "img", f))
        os.makedirs(os.path.join(ours, "teacher", "img"))
        for f in os.listdir(os.path.join(d, "ops", "img")):
            if f.endswith(".npy") and f not in ("pixels.npy", "patches_in.npy"):
                shutil.copy(os.path.join(d, "ops", "img", f), os.path.join(ours, "teacher", "img", f))
        pin = np.load(os.path.join(d, "ops", "img", "patches_in.npy"))
        np.save(os.path.join(ours, "teacher", "img", "patches.npy"), np.concatenate([pin, np.zeros((pin.shape[0], 52), np.uint16)], 1))
        shutil.copytree(os.path.join(d, "weights"), os.path.join(ours, "weights"))
        for shp in os.listdir(os.path.join(d, "gemm_random")):
            os.makedirs(os.path.join(ours, "gemm_random", shp))
            shutil.copy(os.path.join(d, "gemm_random", shp, "out.npy"), os.path.join(ours, "gemm_random", shp, "out.npy"))
        return ours

    def test_the_tree_has_the_layout_the_compare_reads(self):
        for rel in ("ref/MANIFEST.json", "shim/MANIFEST.json", "ops/MANIFEST.json", "floors.json", "weights/norm1_00.npy",
                    "weights/pixel_table.npy", "gemm_random/wqkv/out.npy", "ops/img/block00.attn_o.npy", "ref/img/patch_embed.npy"):
            self.assertTrue(os.path.exists(os.path.join(self.dumps, rel)), rel)

    def test_the_pixel_table_is_the_reference_formula(self):
        """The table in the weights dump, applied to the u8 pixels, gives the reference's patches bit for bit."""
        t = np.load(os.path.join(self.dumps, "weights", "pixel_table.npy"))
        self.assertEqual(t.shape, (256,))
        self.assertTrue(np.array_equal(t, self.dr.pixel_table_np()))
        pix = self.items[0][1][3]
        patches = np.load(os.path.join(self.dumps, "ops", "img", "patches_in.npy"))
        n_h, n_w = pix.shape[0] // 14, pix.shape[1] // 14
        want = t[pix].reshape(n_h, 14, n_w, 14, 3).transpose(0, 2, 4, 1, 3).reshape(n_h * n_w, 588)
        self.assertTrue(np.array_equal(want, patches))

    def test_ours_equal_to_the_reference_passes_every_part(self):
        ours = self.make_ours("ours_ok")
        rep = vc.run(self.dumps, ours)
        self.assertTrue(rep.ok, "\n".join(rep.failures))
        self.assertEqual({r.part for r in rep.results}, {"instrument", "teacher", "free"})
        judged = [r for r in rep.results if r.judged]
        self.assertGreater(len(judged), 60)

    def test_a_wrong_unfold_and_a_shuffled_operation_are_caught_in_the_real_layout(self):
        ours = self.make_ours("ours_bad")
        for point in ("unfold", "block01.qkv", "block02.w2_out"):
            p = os.path.join(ours, "teacher", "img", point + ".npy")
            np.save(p, np.roll(np.load(p), 1, axis=0))
        rep = vc.run(self.dumps, ours)
        self.assertFalse(rep.ok)
        for point in ("unfold", "block01.qkv", "block02.w2_out"):
            self.assertTrue(any(point in f for f in rep.failures), (point, rep.failures))

    def test_free_points_shuffled_fail_in_the_real_layout(self):
        ours = self.make_ours("ours_free")
        p = os.path.join(ours, "free", "img", "block01.mlp_add.npy")
        np.save(p, np.roll(np.load(p), 1, axis=0))
        self.assertFalse(vc.run(self.dumps, ours).ok)


if __name__ == "__main__":
    unittest.main()
