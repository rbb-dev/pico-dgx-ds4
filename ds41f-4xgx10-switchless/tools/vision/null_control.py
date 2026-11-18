#!/usr/bin/env python3
"""The null control of the vision tower: how far apart two correct runs are, at every layer.

A later comparison (our kernels against the reference) may differ from the reference by no more than twice the distance
that two runs of the same weights already differ by. This tool measures that distance. For every test image it runs
the tower on the GPU four times and compares at every point (dump_ref.py lists them: the patch embedding, both adds of
all 32 blocks, the final norm, the unfold, g, the aligner, the span rows):

    ref1   the reference unchanged (MATH attention)                               saved to OUT/ref
    shim1  what the running encoder does (flash attention on a contiguous view)   saved to OUT/shim
           -> shim1 against ref1 is THE NULL DISTANCE (null_distance.json): the distance of today's encoder from the
              publisher's reference, which the later pass line is 2 x
    then the GPU state is disturbed (a large allocation and release, other image sizes run, the cache emptied, odd-sized
    blocks freed in between) and
    shim2  the encoder's mode again     -> shim2 against shim1: repeat distance, expected 0 bits
    ref2   the reference again          -> ref2 against ref1: repeat distance, expected 0 bits

Distances per point (the worst over the images): the largest relative error of a row (L2), the smallest row cosine, the
share of elements more than one bf16 step apart, the share not equal, the largest step count.

Before it reports, the instrument is checked: the same dump against itself must give 0, and the same dump with its rows
rotated must give a large distance. If either check fails the run stops.

Run it inside the container, with the engine stopped (a second copy of the weights and the MATH attention's scores need
several GiB of GPU memory):
    docker exec pico-vision python3 /app/null_control.py --model MODEL_SNAPSHOT --images /tmp/gv4-images --out /tmp/gv4
Outputs in OUT: ref/, shim/ (dumps), null_distance.json, pass_lines.json, null_report.txt.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_ref as dr  # noqa: E402

DEFAULT_FULL = "noise_544,noise_768,strip_42x42882"


def disturb(m, gib):
    """Change the GPU's state between two runs: allocator layout, caches, another stream of work."""
    torch = m.torch
    keep = []
    big = torch.empty(int(gib * 2**30) // 2, dtype=torch.bfloat16, device="cuda")
    big.normal_()
    for k in range(1, 40):  # odd-sized blocks, every other one freed again: a fragmented pool
        keep.append(torch.empty(1000003 * k, dtype=torch.bfloat16, device="cuda").fill_(k))
    del keep[::2]
    a = torch.randn(4097, 1031, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1031, 3071, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        (a @ b).sum().item()
    del big, keep, a, b
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def instrument_check(root, stems):
    """Zero for a dump against itself; a large distance for the same dump with its rows rotated."""
    for stem in stems:
        d = os.path.join(root, stem)
        for point in ("block16.mlp_add", "span"):
            f = os.path.join(d, point + ".npy")
            if not os.path.exists(f):
                continue
            x = np.load(f)
            same = dr.metrics(x, x)
            assert same["share_not_equal"] == 0 and same["max_rel_row_err"] == 0, ("identical rows are not zero apart", same)
            rolled = dr.metrics(np.roll(x, 1, axis=0), x)
            assert rolled["max_rel_row_err"] > 0.05 and rolled["share_over_1_step"] > 0.05, ("rotated rows not seen", rolled)
            print("instrument check %s/%s: identical -> 0; rows rotated -> max_rel %.3f, %.1f%% of elements over 1 step"
                  % (stem, point, rolled["max_rel_row_err"], 100 * rolled["share_over_1_step"]), flush=True)
            return
    raise SystemExit("instrument check could not run: no dump found")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("MODEL_SNAPSHOT", "/model"))
    p.add_argument("--images", required=True, help="a folder of images (frame_*.png skipped)")
    p.add_argument("--out", required=True)
    p.add_argument("--full", default=DEFAULT_FULL, help="image stems saved and compared whole; the others by every 16th row")
    p.add_argument("--full-patches", type=int, default=4096)
    p.add_argument("--disturb-gib", type=float, default=2.0)
    p.add_argument("--no-disturb", action="store_true", help="skip the disturbance (not the design's control)")
    a = p.parse_args()

    t0 = time.time()
    m = dr.Model(a.model)
    print("loaded in %.1f s" % (time.time() - t0), flush=True)
    names = set(x for x in a.full.split(",") if x)
    paths = dr.list_images(a.images)
    roots = {k: os.path.join(a.out, k) for k in ("ref", "shim")}
    manifests = {"ref": {"attn": "ref", "stride": dr.STRIDE, "images": {}}, "shim": {"attn": "shim", "stride": dr.STRIDE, "images": {}}}
    cmp = {"null": {}, "repeat_shim": {}, "repeat_ref": {}}  # image -> point -> [metrics]
    order, stems = [], []
    small = None
    for path in paths:  # the smallest image runs in the disturbance
        prep = m.prepare(path)
        if small is None or prep[2][0] * prep[2][1] < small[0]:
            small = (prep[2][0] * prep[2][1], prep)
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        prep = m.prepare(path)
        n = int(prep[2][0] * prep[2][1])
        full = dr.is_full(stem, n, names, a.full_patches)
        stems.append(stem)
        t1 = time.time()
        # ref1: saved
        s_ref = dr.SaveSink(roots["ref"], stem, full)
        info = m.run(prep, "ref", s_ref)
        manifests["ref"]["images"][stem] = dict(info, full=full, points=s_ref.points)
        # shim1: saved, and compared with the dump of ref1 (the null distance)
        s_shim = dr.SaveSink(roots["shim"], stem, full)
        cmp["null"][stem] = {}
        m.run(prep, "shim", dr.Tee(s_shim, dr.CompareSink(roots["ref"], stem, full, cmp["null"][stem])))
        manifests["shim"]["images"][stem] = dict(info, full=full, points=s_shim.points)
        # disturb, then the repeats
        if not a.no_disturb:
            disturb(m, a.disturb_gib)
            m.run(small[1], "shim", dr.Tee())  # another image size between the runs
        cmp["repeat_shim"][stem] = {}
        m.run(prep, "shim", dr.CompareSink(roots["shim"], stem, full, cmp["repeat_shim"][stem]))
        cmp["repeat_ref"][stem] = {}
        m.run(prep, "ref", dr.CompareSink(roots["ref"], stem, full, cmp["repeat_ref"][stem]))
        order = list(s_ref.points)
        print("%s: %d patches, %d rows, %s, four runs in %.1f s" % (stem, n, info["rows"],
              "whole" if full else "every %dth row" % dr.STRIDE, time.time() - t1), flush=True)
    for k in ("ref", "shim"):
        with open(os.path.join(roots[k], "MANIFEST.json"), "w") as f:
            json.dump(manifests[k], f, indent=1)

    instrument_check(roots["ref"], stems)

    def agg(kind):
        return {pt: dr.aggregate([cmp[kind][s][pt][0] for s in stems]) for pt in order}
    null, rep_s, rep_r = agg("null"), agg("repeat_shim"), agg("repeat_ref")

    lines = ["%-18s %12s %12s %12s %12s %9s | repeat shim: not-equal  repeat ref: not-equal"
             % ("point", "max_rel_row", "min_row_cos", "over_1_step", "not_equal", "max_steps")]
    for pt in order:
        n_, s_, r_ = null[pt], rep_s[pt], rep_r[pt]
        lines.append("%-18s %12.3e %12.8f %12.3e %12.3e %9d | %22.3e %22.3e" % (
            pt, n_["max_rel_row_err"], n_["min_row_cos"], n_["share_over_1_step"], n_["share_not_equal"], n_["max_steps"],
            s_["share_not_equal"], r_["share_not_equal"]))
    worst = max(order, key=lambda q: null[q]["max_rel_row_err"])
    lines.append("")
    lines.append("null distance = shim against ref; worst point by row error: %s (%.3e)" % (worst, null[worst]["max_rel_row_err"]))
    lines.append("repeat of the encoder's mode after the disturbance: %s" % (
        "bit-identical at every point" if all(v["share_not_equal"] == 0 for v in rep_s.values()) else "NOT identical, see the table"))
    lines.append("repeat of the reference: %s" % (
        "bit-identical at every point" if all(v["share_not_equal"] == 0 for v in rep_r.values()) else "NOT identical, see the table"))
    if any(v["nonfinite"] for v in list(null.values()) + list(rep_s.values()) + list(rep_r.values())):
        lines.append("WARNING: non-finite values were seen; the distances are not meaningful")
    report = "\n".join(lines)
    print(report)

    # The pass line is 2 x the null distance at each point (the design). For the row cosine the distance is 1 - cos.
    pass_lines, zero = {}, []
    for pt in order:
        n_ = null[pt]
        pass_lines[pt] = {"max_rel_row_err": 2 * n_["max_rel_row_err"], "min_row_cos": 1 - 2 * (1 - n_["min_row_cos"]),
                          "share_over_1_step": 2 * n_["share_over_1_step"]}
        if n_["max_rel_row_err"] == 0 or n_["share_over_1_step"] == 0:
            zero.append(pt)
    if zero:
        report += ("\nNOTE: at %d points a null distance is 0 (e.g. %s): 2 x 0 is 0, so any difference there would fail. "
                   "The design gives no floor; decide one before the line is used." % (len(zero), zero[0]))
        print(report.splitlines()[-1])
    json.dump({"images": stems, "full": sorted(names), "null_distance": null, "repeat_shim": rep_s, "repeat_ref": rep_r,
               "per_image_null": {s: {pt: cmp["null"][s][pt][0] for pt in order} for s in stems}},
              open(os.path.join(a.out, "null_distance.json"), "w"), indent=1)
    json.dump(pass_lines, open(os.path.join(a.out, "pass_lines.json"), "w"), indent=1)
    open(os.path.join(a.out, "null_report.txt"), "w").write(report + "\n")
    print("written: null_distance.json pass_lines.json null_report.txt in", a.out)


if __name__ == "__main__":
    main()
