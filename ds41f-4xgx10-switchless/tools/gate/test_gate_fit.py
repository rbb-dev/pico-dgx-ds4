#!/usr/bin/env python3
"""Tests of gate_fit's mathematics on synthetic data with a known calibration (python3 test_gate_fit.py)."""
import io, os, sys, tempfile, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_fit as g

TRUE_A = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
TRUE_B = np.array([0.5, 0.4, 0.3, 0.2, 0.1])


def synth(n, seed, with_mlp=False):
    """Lines whose true chance of depth j is sigmoid(A_j * logit(c_j) + B_j) for the head's stated chance c_j."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.2, 0.95, size=(n, g.D))
    true_c = g.sigmoid(TRUE_A * g.logit(c) + TRUE_B)
    accepted = np.zeros(n, int)
    alive = np.ones(n, bool)
    for j in range(g.D):
        ok = alive & (rng.uniform(size=n) < true_c[:, j])
        accepted += ok
        alive = ok
    verified = np.full(n, g.D)
    return g.surv_from_cond(c), verified, accepted, true_c


class Math(unittest.TestCase):
    def test_cond_surv_roundtrip(self):
        c = np.random.default_rng(1).uniform(0.1, 0.9, size=(50, g.D))
        np.testing.assert_allclose(g.cond_from_surv(g.surv_from_cond(c)), c, rtol=1e-9)

    def test_fit_recovers_known_calibration(self):
        s, v, a, _ = synth(60000, 2)
        cal = g.fit_cal(s, v, a)
        for j in range(g.D):
            self.assertAlmostEqual(cal[j][0], TRUE_A[j], delta=0.06, msg="slope depth %d" % (j + 1))
            self.assertAlmostEqual(cal[j][1], TRUE_B[j], delta=0.06, msg="offset depth %d" % (j + 1))

    def test_calibration_improves_scores_out_of_sample(self):
        s, v, a, _ = synth(40000, 3)
        cal = g.fit_cal(s, v, a)
        s2, v2, a2, _ = synth(40000, 4)
        before = g.score(s2, v2, a2)
        after = g.score(g.apply_cal(s2, cal), v2, a2)
        for j in range(g.D):
            self.assertLess(after[j][3], before[j][3])  # survival log-loss
            self.assertLess(after[j][4], before[j][4])  # survival ECE
            self.assertLess(after[j][5], before[j][5] + 1e-9)  # conditional log-loss

    def test_identity_when_already_calibrated(self):
        rng = np.random.default_rng(5)
        c = rng.uniform(0.2, 0.95, size=(80000, g.D))
        acc = np.zeros(len(c), int)
        alive = np.ones(len(c), bool)
        for j in range(g.D):
            alive = alive & (rng.uniform(size=len(c)) < c[:, j])
            acc += alive
        cal = g.fit_cal(g.surv_from_cond(c), np.full(len(c), g.D), acc)
        for a, b in cal:
            self.assertAlmostEqual(a, 1.0, delta=0.05)
            self.assertAlmostEqual(b, 0.0, delta=0.05)

    def test_ece_and_logloss_extremes(self):
        rng = np.random.default_rng(6)
        p = rng.uniform(size=200000)
        y = (rng.uniform(size=p.size) < p).astype(float)
        self.assertLess(g.ece(p, y), 0.01)
        self.assertGreater(g.ece(np.full(1000, 0.9), np.zeros(1000)), 0.89)
        self.assertAlmostEqual(g.logloss([0.5, 0.5], [0, 1]), np.log(2), places=9)

    def test_censoring_counts_tested_depths_only(self):
        s = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]] * 4)
        v = np.array([0, 1, 2, 5])
        a = np.array([0, 1, 1, 5])
        p, y = g.surv_samples(s, v, a, 2)  # verified >= 2: lines 3, 4
        self.assertEqual(len(y), 2)
        self.assertEqual(list(y), [0.0, 1.0])
        c, yc = g.cond_samples(s, v, a, 2)  # tested and the first accepted: lines 3 and 4 (line 3 accepted 1 of 2, so depth 1 landed)
        self.assertEqual(len(yc), 2)

    def test_engine_formula_matches(self):
        """apply_cal is the gate kernel's correction: sigmoid(a * logit(s_j / s_j-1) + b), multiplied along the depths."""
        s = np.array([[0.8, 0.6, 0.3, 0.2, 0.1]])
        cal = [(0.5, 0.2)] * 5
        out = g.apply_cal(s, cal)[0]
        prev, ref = 1.0, 1.0
        for j in range(5):
            cnd = s[0, j] / (s[0, j - 1] if j else 1.0)
            ref *= 1 / (1 + np.exp(-(0.5 * np.log(cnd / (1 - cnd)) + 0.2)))
            self.assertAlmostEqual(out[j], ref, places=9)


class Csv(unittest.TestCase):
    def test_load_and_report(self):
        s, v, a, _ = synth(3000, 7)
        hdr = "tag,step,entry,sessions,step_rows,verified,accepted," + ",".join(
            "%s%d" % (n, j) for n in ("ship", "mlp", "used") for j in range(1, 6)) + "," + ",".join("f%d" % i for i in range(15))
        with tempfile.TemporaryDirectory() as d:
            for t in ("setA", "setB"):
                with open(os.path.join(d, t + "_n3.csv"), "w") as f:
                    f.write(hdr + "\n")
                    for i in range(len(a)):
                        f.write("x,%d,0,%d,12,%d,%d," % (i, 2 + i % 3, v[i], a[i]) + ",".join("%.7g" % x for x in s[i]) + "," +
                                ",".join(["0"] * 5) + "," + ",".join("%.7g" % x for x in s[i]) + "," + ",".join(["0"] * 15) + "\n")
            data = g.load([os.path.join(d, "setA_n3.csv"), os.path.join(d, "setB_n3.csv")])
            self.assertEqual(sorted(set(data["tag"])), ["setA", "setB"])
            out = io.StringIO()
            cal = g.report(data, "ship", out)
            self.assertIn("DSGB10_BATCH_CAL", out.getvalue())
            self.assertEqual(len(cal), 5)
            self.assertAlmostEqual(cal[0][0], TRUE_A[0], delta=0.15)


if __name__ == "__main__":
    unittest.main()
