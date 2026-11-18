#!/usr/bin/env python3
"""Calibration of the batch gate's draft survival from DSGB10_GATE_DUMP data (engine/server/cmd.cpp writes the lines).

  gate_fit.py FILE.csv [FILE.csv ..] [--source ship|mlp]   report per prompt set and per number of sessions, and the fitted DSGB10_BATCH_CAL
Prompt set = the file name's part before the first '_' (set0_n3.csv -> set0), else the tag column. Fits are scored out of sample.
"""
import argparse, csv, math, os, sys
import numpy as np

D = 5
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def cond_from_surv(s):
    """Survival [N, D] -> the chance of each depth given the ones before it [N, D]."""
    s = np.asarray(s, dtype=np.float64)
    prev = np.concatenate([np.ones((s.shape[0], 1)), s[:, :-1]], axis=1)
    return s / np.maximum(prev, 1e-12)


def surv_from_cond(c):
    return np.cumprod(np.asarray(c, dtype=np.float64), axis=1)


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def ece(p, y, bins=10):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not len(y):
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(y))


def fit_logistic(x, y, iters=50, ridge=1e-6):
    """argmin of the log-loss of sigmoid(a * x + b) against y (Newton). Returns (a, b); (1, 0) without usable data."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 20 or y.min() == y.max() or np.ptp(x) < 1e-9:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = sigmoid(a * x + b)
        g = np.array([np.sum((p - y) * x) + ridge * (a - 1), np.sum(p - y)])
        w = p * (1 - p)
        h = np.array([[np.sum(w * x * x) + ridge, np.sum(w * x)], [np.sum(w * x), np.sum(w) + ridge]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


def apply_cal(s, cal):
    """Survival [N, D] through the correction cal = [(a1, b1) .. (a5, b5)], as the engine's gate applies it."""
    c = cond_from_surv(s)
    out = np.empty_like(c)
    for j in range(D):
        out[:, j] = sigmoid(cal[j][0] * logit(c[:, j]) + cal[j][1])
    return surv_from_cond(out)


def cond_samples(s, verified, accepted, j):
    """Depth j (1-based): the predicted chance and the outcome, over the lines where it was tested."""
    m = (verified >= j) & (accepted >= j - 1)
    c = cond_from_surv(s)[:, j - 1]
    return c[m], (accepted[m] >= j).astype(np.float64)


def fit_cal(s, verified, accepted):
    cal = []
    for j in range(1, D + 1):
        c, y = cond_samples(s, verified, accepted, j)
        cal.append(fit_logistic(logit(c), y))
    return cal


def surv_samples(s, verified, accepted, j):
    m = verified >= j
    return s[m, j - 1], (accepted[m] >= j).astype(np.float64)


def score(s, verified, accepted):
    """Per depth: lines, mean predicted / observed survival, log-loss and ECE of the survival, log-loss of the conditional."""
    rows = []
    for j in range(1, D + 1):
        p, y = surv_samples(s, verified, accepted, j)
        c, yc = cond_samples(s, verified, accepted, j)
        rows.append((len(y), float(p.mean()) if len(y) else float("nan"), float(y.mean()) if len(y) else float("nan"),
                     logloss(p, y), ece(p, y), logloss(c, yc)))
    return rows


def load(paths):
    cols = None
    data = {"tag": [], "step": [], "sessions": [], "verified": [], "accepted": [], "ship": [], "mlp": [], "used": [],
            "feat": []}
    for path in paths:
        stem = os.path.basename(path).rsplit(".", 1)[0]
        ftag = stem.split("_")[0] if "_" in stem else ""
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row.get("step") or row["step"] == "step":  # an empty line, or the header again (a cut that began at it)
                    continue
                data["tag"].append(ftag or row.get("tag", ""))
                data["step"].append(int(row["step"]))
                data["sessions"].append(int(row["sessions"]))
                data["verified"].append(int(row["verified"]))
                data["accepted"].append(int(row["accepted"]))
                data["ship"].append([float(row["ship%d" % j]) for j in range(1, D + 1)])
                data["mlp"].append([float(row["mlp%d" % j]) for j in range(1, D + 1)])
                data["used"].append([float(row["used%d" % j]) for j in range(1, D + 1)])
                data["feat"].append([float(row["f%d" % j]) for j in range(15)])
    out = {k: np.array(v) for k, v in data.items()}
    return out


def fmt(rows):
    return "  ".join("d%d n=%d pred %.3f obs %.3f ll %.3f ece %.3f" % ((i + 1,) + r[:3] + r[3:5]) for i, r in enumerate(rows))


def report(data, source, out=sys.stdout):
    tags = sorted(set(data["tag"]))
    have_mlp = bool((data["mlp"] > 0).any())
    srcs = [("ship", data["ship"])] + ([("mlp", data["mlp"])] if have_mlp else [])
    lines = []
    for name, s in srcs:
        # the fit is scored out of sample: leave one prompt set out, or even / odd steps with a single set
        fitted = np.empty_like(s)
        if len(tags) > 1:
            for t in tags:
                m = data["tag"] == t
                cal = fit_cal(s[~m], data["verified"][~m], data["accepted"][~m])
                fitted[m] = apply_cal(s[m], cal)
        else:
            ev = data["step"] % 2 == 0
            for m in (ev, ~ev):
                cal = fit_cal(s[~m], data["verified"][~m], data["accepted"][~m])
                fitted[m] = apply_cal(s[m], cal)
        for label, mask in [("all", np.ones(len(s), bool))] + [("set " + t, data["tag"] == t) for t in tags] + [
                ("sessions %d" % n, data["sessions"] == n) for n in sorted(set(data["sessions"]))]:
            if not mask.any():
                continue
            for kind, arr in (("raw", s), ("fitted", fitted)):
                sc = score(arr[mask], data["verified"][mask], data["accepted"][mask])
                lines.append("%-4s %-12s %-6s %s" % (name, label, kind, fmt(sc)))
    print("\n".join(lines), file=out)
    s = dict(srcs)[source]
    cal = fit_cal(s, data["verified"], data["accepted"])
    print("\nDSGB10_BATCH_CAL (fitted on all of %s's lines): %s" % (source, ",".join("%.4f,%.4f" % c for c in cal)),
          file=out)
    return cal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--source", choices=["ship", "mlp"], default="ship")
    a = ap.parse_args()
    data = load(a.files)
    if not len(data["step"]):
        sys.exit("no lines")
    report(data, a.source)


if __name__ == "__main__":
    main()
