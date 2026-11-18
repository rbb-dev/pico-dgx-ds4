#!/usr/bin/env python3
"""Holds the image encoder's kernels (vit::) to the model's own math: in three parts.

Inputs are two trees of .npy files (bf16 as raw uint16; float32 tables as float32; pixels as uint8):

  DUMPS   written by dump_ref.py in the pico-vision container (the reference's own code, on a GPU)
    ref|shim|mem|flash/MANIFEST.json, <mode>/<stem>/<point>.npy   free-running points of the four attention modes
    ops/MANIFEST.json, ops/<stem>/<point>.npy                     every operation's input and output (mode ref, whole)
    floors.json                                                   per operation, measured on the reference
    weights/<name>.npy                                            the copy-type weights as the reference holds them
    gemm_random/<shape>/{a,w,b,out}.npy                           random data and cuBLAS's output, patch_embed's shape etc.
  OURS    written by vit_teacher (engine/vision/tests) on a GPU
    free/<stem>/<point>.npy        the encoder's own free-running run
    teacher/<stem>/<point>.npy     each single operation fed the reference's input, output named as the dump's point
    weights/<name>.npy             the same weights from the checkpoint reader
    gemm_random/<shape>/out.npy    vit::gemm on the same random data

Part 1, teacher-forced, one operation at a time (errors do not accumulate):
  copy-type (patchify, rope table, unfold, span, the pixel table, delimiter rows, norm weights) and the adds: every bit;
  matrix products: within a floor measured as vit::gemm against cuBLAS on random data of that product's shape;
  rmsnorm, rope, swiglu, gelu: within the floor of torch's own bf16 operation against the float64 answer;
  attention: within the largest distance among the reference's attention and the encoder's mode, memory-efficient and
  flash attention on the same q, k, v.
Part 2, free-running, on robust statistics (row-median relative error, 1st-percentile row cosine, relative Frobenius norm),
  judged at patch_embed and blocks 0-7 against FREE_MARGIN x the largest of the null pairs (shim, mem, flash each against
  the reference); later points are recorded, not judged.
Part 3, positive controls (a deliberate error each): a control must stand >= CONTROL_FACTOR x the null in patch_embed or
  blocks 0-2 on a robust statistic, or differ in a copy-type operation's bytes; one that does not makes the suite fail
  ("the line is too loose"). Split-only controls live in the byte test against the unsplit run, not here.

Before it judges, the instrument is checked: a dump against itself is 0 apart; the same dump with its rows rotated is not.
Exit code 0 only when everything judged passes and every control is detected.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# the plan's constants
# ---------------------------------------------------------------------------------------------------------------

TEACHER_MARGIN = 3.0      # x the measured floor of the operation
FREE_MARGIN = 2.0         # x the largest null pair
CONTROL_FACTOR = 10.0     # a control must stand this far above the null
JUDGED_BLOCKS = 8         # free-running points judged: patch_embed and blocks 0..7
CONTROL_BLOCKS = 3        # positive controls: patch_embed and blocks 0..2
# One bf16 rounding of an element is ~2^-9 of its size; two independent roundings differ by ~1.6e-3 relative. A floor
# that measured 0 (identical operations) still leaves room for that.
# A null this far from the reference at a judged point means a dump is broken, not that two correct runs differ
# (measured: about 1e-2 at these points).
NULL_CEILING = 0.1
ABS_MIN = {"rel_frob": 4e-3, "med_row_rel": 4e-3, "cos_dist": 2e-5}
JUDGED_STATS = ("med_row_rel", "cos_dist", "rel_frob")
CONTROL_STATS = ("med_row_rel", "rel_frob")
NULL_MODES = ("shim", "mem", "flash")

# (point suffix, kind): kinds exact (every bit), gemm:<shape>, point, attn; None = input only
BLOCK_OPS = [("x_in", None), ("norm1_out", "point"), ("qkv", "gemm:wqkv"), ("q_rope", "point"), ("k_rope", "point"),
             ("attn_o", "attn"), ("wo_out", "gemm:wo"), ("attn_add", "exact"), ("norm2_out", "point"),
             ("w1_out", "gemm:w1"), ("swiglu_out", "point"), ("w2_out", "gemm:w2"), ("mlp_add", "exact")]
GLOBAL_OPS = [("patch_embed", "gemm:patch_embed"), ("norm", "point"), ("a1_out", "gemm:aligner1"), ("g", "point"),
              ("aligner", "gemm:aligner2")]
COPY_OPS = ("unfold", "span")          # bf16 copies
TABLE_OPS = ("rope_cos", "rope_sin")   # float32 tables
GEMM_SHAPES = ("patch_embed", "wqkv", "wo", "w1", "w2", "aligner1", "aligner2")

# the controls and where each is seen: free-running blocks 0-2 (and the byte test where a copy-type op is involved)
CONTROLS = ("swap-heads", "rope-axes", "norm-swap", "qkv-order", "gate-up", "unfold-window", "unfold-order", "pad-grey",
            "delimiters")


# ---------------------------------------------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------------------------------------------


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def ulp_key(u16):
    k = (u16 & 0x7FFF).astype(np.int32)
    return np.where((u16 & 0x8000) != 0, -k, k)


def bytes_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def stats(cand, ref):
    """Distance of candidate rows from reference rows (uint16 bf16 bits, [rows, cols]): robust statistics first.
    rel_frob: ||c - r||_F / ||r||_F. med_row_rel: median over rows of ||c_i - r_i|| / ||r_i||. cos_p01: the cosine 99 % of
    the rows reach (the 1st percentile of the row cosines); cos_dist = 1 - cos_p01. max_row_rel and min_row_cos are
    recorded, never judged."""
    cand, ref = np.asarray(cand), np.asarray(ref)
    if cand.shape != ref.shape:
        raise ValueError("shape %s against %s" % (cand.shape, ref.shape))
    a2, b2 = cand.reshape(cand.shape[0], -1), ref.reshape(ref.shape[0], -1)
    a, b = bf16_to_f32(a2).astype(np.float64), bf16_to_f32(b2).astype(np.float64)
    nonfinite = int((~np.isfinite(a)).sum())
    ref_nonfinite = int((~np.isfinite(b)).sum())
    a, b = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    dn, an, bn = np.linalg.norm(a - b, axis=1), np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    rel = np.where(bn > 0, dn / np.maximum(bn, 1e-30), dn)
    cos = np.where((an > 0) & (bn > 0), (a * b).sum(1) / np.maximum(an * bn, 1e-30), np.where(dn == 0, 1.0, 0.0))
    ulps = np.abs(ulp_key(a2) - ulp_key(b2))
    bf = np.linalg.norm(b)
    frob = float(np.linalg.norm(a - b) / bf) if bf > 0 else float(np.linalg.norm(a - b))
    cos_p01 = float(np.percentile(cos, 1))
    return {"rel_frob": frob, "med_row_rel": float(np.median(rel)), "cos_p01": cos_p01, "cos_dist": max(0.0, 1.0 - cos_p01),
            "max_row_rel": float(rel.max()), "min_row_cos": float(cos.min()),
            "share_over_1_step": float((ulps > 1).mean()), "share_not_equal": float((ulps > 0).mean()),
            "max_steps": int(ulps.max()), "rows": int(a.shape[0]), "nonfinite": nonfinite, "ref_nonfinite": ref_nonfinite}


def limit(floor, margin):
    """The line for each judged statistic: margin x the floor, never under the rounding allowance."""
    return {s: max(margin * floor.get(s, 0.0), ABS_MIN[s]) for s in JUDGED_STATS}


def worst_of(stat_list):
    return {s: max(x[s] for x in stat_list) for s in JUDGED_STATS}


# ---------------------------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------------------------


class Result:
    def __init__(self, part, name, ok, judged=True, kind="", stats_=None, detail=""):
        self.part, self.name, self.ok, self.judged, self.kind, self.stats, self.detail = part, name, ok, judged, kind, stats_ or {}, detail

    def line(self):
        flag = "ok  " if self.ok else "FAIL"
        if not self.judged:
            flag = "rec "
        s = self.stats
        nums = ""
        if "rel_frob" in s:
            nums = " rel_frob %.3e med_row %.3e cos_dist %.3e" % (s["rel_frob"], s["med_row_rel"], s["cos_dist"])
        return "%s %-8s %-34s %-6s%s %s" % (flag, self.part, self.name, self.kind, nums, self.detail)


class Report:
    def __init__(self):
        self.results = []

    def add(self, *a, **k):
        r = Result(*a, **k)
        self.results.append(r)
        return r

    @property
    def failures(self):
        return ["%s %s: %s" % (r.part, r.name, r.detail or "failed") for r in self.results if r.judged and not r.ok]

    @property
    def ok(self):
        return not self.failures

    def lines(self):
        out = [r.line() for r in self.results]
        n = sum(1 for r in self.results if r.judged)
        out.append("")
        out.append("%d judged, %d failed, %d recorded" % (n, len(self.failures), len(self.results) - n))
        out.extend("FAILED: " + f for f in self.failures)
        out.append("RESULT: " + ("PASS" if self.ok else "FAIL"))
        return out

    def to_json(self):
        return {"ok": self.ok, "failures": self.failures,
                "results": [{"part": r.part, "name": r.name, "ok": r.ok, "judged": r.judged, "kind": r.kind,
                             "stats": r.stats, "detail": r.detail} for r in self.results]}


# ---------------------------------------------------------------------------------------------------------------
# reading the trees
# ---------------------------------------------------------------------------------------------------------------


class Missing(Exception):
    pass


def load(root, rel):
    p = os.path.join(root, rel + ".npy")
    if not os.path.exists(p):
        raise Missing(rel)
    return np.load(p)


def load_json(root, rel, default=None):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        if default is not None:
            return default
        raise Missing(rel)
    return json.load(open(p))


def free_points_judged():
    pts = ["patch_embed"]
    for i in range(JUDGED_BLOCKS):
        pts += ["block%02d.attn_add" % i, "block%02d.mlp_add" % i]
    return pts


def control_points():
    pts = ["patch_embed"]
    for i in range(CONTROL_BLOCKS):
        pts += ["block%02d.attn_add" % i, "block%02d.mlp_add" % i]
    return pts


def list_free_points(root, stem):
    d = os.path.join(root, "free", stem)
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy")) if os.path.isdir(d) else []


def ordered(points):
    def key(p):
        m = re.match(r"block(\d+)\.(.*)", p)
        if m:
            return (1, int(m.group(1)), 0 if m.group(2) == "attn_add" else 1)
        return (0 if p == "patch_embed" else 2, 0, 0)
    return sorted(points, key=key)


# ---------------------------------------------------------------------------------------------------------------
# the parts
# ---------------------------------------------------------------------------------------------------------------


def instrument_check(dumps, rep):
    """A dump against itself is 0 apart; the same dump with its rows rotated is far apart."""
    man = load_json(dumps, "ref/MANIFEST.json")
    for stem in man["images"]:
        for point in ("block07.mlp_add", "block00.mlp_add", "patch_embed"):
            p = os.path.join(dumps, "ref", stem, point + ".npy")
            if os.path.exists(p):
                x = np.load(p)
                same, rolled = stats(x, x), stats(np.roll(x, 1, axis=0), x)
                ok = same["share_not_equal"] == 0 and same["rel_frob"] == 0 and rolled["med_row_rel"] > 0.05 \
                    and rolled["share_over_1_step"] > 0.05
                rep.add("instrument", "%s/%s" % (stem, point), ok, kind="check", stats_=rolled,
                        detail="identical -> 0; rows rotated -> med_row %.3f, %.1f%% over 1 step" % (
                            rolled["med_row_rel"], 100 * rolled["share_over_1_step"]))
                return
    rep.add("instrument", "none", False, detail="instrument check could not run: no dump found")


def gemm_floors(dumps, ours, rep):
    """vit::gemm against cuBLAS on random data of each product's shape: the floor of every matrix product."""
    out = {}
    for shp in GEMM_SHAPES:
        try:
            ref = load(dumps, "gemm_random/%s/out" % shp)
            got = load(ours, "gemm_random/%s/out" % shp)
        except Missing as e:
            rep.add("teacher", "gemm_random/%s" % shp, False, kind="floor", detail="missing %s" % e)
            continue
        st = stats(got, ref)
        out[shp] = st
        rep.add("teacher", "gemm_random/%s" % shp, st["nonfinite"] == 0, judged=True, kind="floor", stats_=st,
                detail="vit::gemm against cuBLAS (the floor of this product)")
    return out


def check_teacher(dumps, ours, rep, margin=TEACHER_MARGIN):
    gf = gemm_floors(dumps, ours, rep)
    floors = load_json(dumps, "floors.json", default={"point": {}, "attn": {}})
    man = load_json(dumps, "ops/MANIFEST.json")
    # weights: every bit
    wdir = os.path.join(dumps, "weights")
    for f in sorted(os.listdir(wdir)) if os.path.isdir(wdir) else []:
        name = f[:-4]
        try:
            ok = bytes_equal(load(ours, "weights/" + name), load(dumps, "weights/" + name))
            rep.add("teacher", "weights/" + name, ok, kind="copy", detail="" if ok else "bytes differ from the reference's")
        except Missing as e:
            rep.add("teacher", "weights/" + name, False, kind="copy", detail="missing %s" % e)
    for stem, info in man.items():
        ops = [(p, k) for p, k in GLOBAL_OPS]
        for b in info["op_blocks"]:
            ops += [("block%02d.%s" % (b, s), k) for s, k in BLOCK_OPS if k]
        # copy-type
        for point in COPY_OPS + TABLE_OPS:
            _teacher_copy(dumps, ours, stem, point, point, rep)
        _teacher_patches(dumps, ours, stem, rep)
        for point, kind in ops:
            name = "%s/%s" % (stem, point)
            try:
                ref, got = load(dumps, "ops/%s/%s" % (stem, point)), load(ours, "teacher/%s/%s" % (stem, point))
            except Missing as e:
                rep.add("teacher", name, False, kind=kind.split(":")[0], detail="missing %s" % e)
                continue
            if ref.shape != got.shape:
                rep.add("teacher", name, False, kind=kind.split(":")[0], detail="shape %s against %s" % (got.shape, ref.shape))
                continue
            if kind == "exact":
                ok = bytes_equal(got, ref)
                st = stats(got, ref)
                rep.add("teacher", name, ok, kind="exact", stats_=st, detail="" if ok else "bytes differ (%.2e of elements)" % st["share_not_equal"])
                continue
            st = stats(got, ref)
            fl, why = _floor_for(kind, stem, point, gf, floors)
            if fl is None:
                rep.add("teacher", name, False, kind=kind.split(":")[0], stats_=st, detail=why)
                continue
            lim = limit(fl, margin)
            bad = [s for s in JUDGED_STATS if st[s] > lim[s]]
            if st["nonfinite"] > st["ref_nonfinite"]:
                bad.append("non-finite values")
            rep.add("teacher", name, not bad, kind=kind.split(":")[0], stats_=st,
                    detail="" if not bad else "over the line: " + ", ".join(
                        "%s %.3e > %.3e" % (s, st[s], lim[s]) if s in lim else s for s in bad))


def _floor_for(kind, stem, point, gf, floors):
    if kind.startswith("gemm:"):
        shp = kind.split(":")[1]
        if shp not in gf:
            return None, "no floor: gemm_random/%s missing" % shp
        return gf[shp], ""
    if kind == "point":
        f = floors.get("point", {}).get(stem, {}).get(point)
        return (f, "") if f else (None, "no floor measured for %s in floors.json" % point)
    if kind == "attn":
        pairs = floors.get("attn", {}).get(stem, {}).get(point)
        return (worst_of(list(pairs.values())), "") if pairs else (None, "no floor measured for %s in floors.json" % point)
    return None, "unknown kind " + kind


def _teacher_copy(dumps, ours, stem, point, ours_point, rep):
    name = "%s/%s" % (stem, point)
    try:
        ref, got = load(dumps, "ops/%s/%s" % (stem, point)), load(ours, "teacher/%s/%s" % (stem, ours_point))
    except Missing as e:
        rep.add("teacher", name, False, kind="copy", detail="missing %s" % e)
        return
    ok = bytes_equal(got, ref)
    rep.add("teacher", name, ok, kind="copy", detail="" if ok else "bytes differ from the reference's")


def _teacher_patches(dumps, ours, stem, rep):
    """patchify's output is [n][640]: the 588 values of a patch equal the reference's, the padding is zero."""
    name = "%s/patches" % stem
    try:
        ref, got = load(dumps, "ops/%s/patches_in" % stem), load(ours, "teacher/%s/patches" % stem)
    except Missing as e:
        rep.add("teacher", name, False, kind="copy", detail="missing %s" % e)
        return
    k = ref.shape[1]
    ok = got.shape[0] == ref.shape[0] and got.shape[1] >= k and bytes_equal(got[:, :k], ref) and not got[:, k:].any()
    rep.add("teacher", name, ok, kind="copy", detail="" if ok else "patch values differ from the reference's, or the padding is not zero")


def free_lines(dumps, stem, points, modes):
    """Null distances per judged point: the statistics of each null mode against ref, and their largest."""
    out = {}
    for p in points:
        ref = load(dumps, "ref/%s/%s" % (stem, p))
        per = {}
        for m in modes:
            try:
                per[m] = stats(load(dumps, "%s/%s/%s" % (m, stem, p)), ref)
            except Missing:
                pass
        out[p] = per
    return out


def check_free(dumps, ours, rep, gf=None, margin=FREE_MARGIN):
    man = load_json(dumps, "ref/MANIFEST.json")
    gf = gf or {}
    modes = [m for m in NULL_MODES if os.path.exists(os.path.join(dumps, m, "MANIFEST.json"))]
    if not modes:
        rep.add("free", "nulls", False, detail="no null mode dumped (shim, mem, flash)")
        return {}
    judged, nulls = free_points_judged(), {}
    for stem in man["images"]:
        try:
            nl = free_lines(dumps, stem, judged, modes)
        except Missing as e:
            rep.add("free", stem, False, detail="missing %s" % e)
            continue
        nulls[stem] = nl
        have = list_free_points(ours, stem)
        for p in ordered(set(have) | set(judged)):
            name = "%s/%s" % (stem, p)
            is_judged = p in judged
            try:
                got, ref = load(ours, "free/%s/%s" % (stem, p)), load(dumps, "ref/%s/%s" % (stem, p))
            except Missing as e:
                if is_judged:
                    rep.add("free", name, False, detail="missing %s" % e)
                continue
            if got.shape != ref.shape:
                rep.add("free", name, False, judged=is_judged, detail="shape %s against %s" % (got.shape, ref.shape))
                continue
            st = stats(got, ref)
            if not is_judged:
                rep.add("free", name, True, judged=False, kind="stat", stats_=st, detail="recorded")
                continue
            per = nl[p]
            broke = [m for m, v in per.items() if v["med_row_rel"] > NULL_CEILING]
            if broke:
                rep.add("free", name, False, kind="null",
                        detail="the null %s is %.3f from the reference (ceiling %.2f): a dump is broken" % (
                            broke[0], per[broke[0]]["med_row_rel"], NULL_CEILING))
                continue
            floor = worst_of(list(per.values())) if per else {s: 0.0 for s in JUDGED_STATS}
            lim = limit(floor, margin)
            if p == "patch_embed" and "patch_embed" in gf:   # null 0 there: the gemm's own floor stands in
                lim = {s: max(lim[s], limit(gf["patch_embed"], TEACHER_MARGIN)[s]) for s in JUDGED_STATS}
            bad = [s for s in JUDGED_STATS if st[s] > lim[s]]
            if st["nonfinite"] > st["ref_nonfinite"]:
                bad.append("non-finite values")
            rep.add("free", name, not bad, kind="stat", stats_=st,
                    detail="" if not bad else "over the line: " + ", ".join(
                        "%s %.3e > %.3e" % (s, st[s], lim[s]) if s in lim else s for s in bad))
    return nulls


def check_controls(dumps, nulls, controls, rep):
    man = load_json(dumps, "ref/MANIFEST.json")
    ops_man = load_json(dumps, "ops/MANIFEST.json", default={})
    pts = control_points()
    for name, d in sorted(controls.items()):
        best, where = 0.0, ""
        for stem in man["images"]:
            for p in pts:
                try:
                    got, ref = load(d, "free/%s/%s" % (stem, p)), load(dumps, "ref/%s/%s" % (stem, p))
                except Missing:
                    continue
                st = stats(got, ref)
                per = nulls.get(stem, {}).get(p, {})
                null = worst_of(list(per.values())) if per else {s: 0.0 for s in JUDGED_STATS}
                for s in CONTROL_STATS:
                    ratio = st[s] / max(null[s], ABS_MIN[s])
                    if ratio > best:
                        best, where = ratio, "%s/%s %s" % (stem, p, s)
        byte_hit = ""
        for stem in ops_man:
            for point in COPY_OPS + TABLE_OPS:
                try:
                    got, ref = load(d, "teacher/%s/%s" % (stem, point)), load(dumps, "ops/%s/%s" % (stem, point))
                except Missing:
                    continue
                if not bytes_equal(got, ref):
                    byte_hit = "%s/%s" % (stem, point)
        if best >= CONTROL_FACTOR:
            rep.add("control", name, True, kind="free", detail="%.1f x the null at %s" % (best, where))
        elif byte_hit:
            rep.add("control", name, True, kind="bytes", detail="the copy-type operation %s differs in bytes" % byte_hit)
        else:
            rep.add("control", name, False,
                    detail="the line is too loose: %s stands %.2f x the null at best (needs %.0f x) and no copy-type "
                           "operation differs" % (name, best, CONTROL_FACTOR))


def run(dumps, ours, controls=None, margin=TEACHER_MARGIN, free_margin=FREE_MARGIN):
    controls = controls or {}
    for name in controls:
        if name not in CONTROLS:
            raise ValueError("unknown control %r; the list: %s" % (name, ", ".join(CONTROLS)))
    rep = Report()
    instrument_check(dumps, rep)
    try:
        check_teacher(dumps, ours, rep, margin)
    except Missing as e:
        rep.add("teacher", "dumps", False, detail="missing %s" % e)
    gf = {}
    for shp in GEMM_SHAPES:
        try:
            gf[shp] = stats(load(ours, "gemm_random/%s/out" % shp), load(dumps, "gemm_random/%s/out" % shp))
        except Missing:
            pass
    nulls = check_free(dumps, ours, rep, gf, free_margin)
    if controls:
        check_controls(dumps, nulls, controls, rep)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True)
    p.add_argument("--ours", required=True)
    p.add_argument("--control", action="append", default=[], metavar="NAME=DIR",
                   help="a wrong build's output tree (free/ and/or teacher/); NAME one of: " + ", ".join(CONTROLS))
    p.add_argument("--margin", type=float, default=TEACHER_MARGIN)
    p.add_argument("--free-margin", type=float, default=FREE_MARGIN)
    p.add_argument("--out", help="write the report as JSON")
    a = p.parse_args(argv)
    controls = dict(c.split("=", 1) for c in a.control)
    rep = run(a.dumps, a.ours, controls, a.margin, a.free_margin)
    print("\n".join(rep.lines()))
    if a.out:
        json.dump(rep.to_json(), open(a.out, "w"), indent=1)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
