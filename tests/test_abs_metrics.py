#!/usr/bin/env python3
"""
test_abs_metrics.py — known-answer tests for the evaluation suite.

Every number you put in the Week 7 report comes out of src/abs_metrics.py, so
before trusting any of them, check each metric against a case where the right
answer is known in advance. If one of these fails, the metric is wrong, and
every model comparison built on it is wrong too.

These double as the clearest explanation of what each metric means: each test
builds a model with a known defect and checks the metric detects exactly that.

    python tests/test_abs_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import abs_metrics as M  # noqa: E402

RNG = np.random.default_rng(2026)
N = 200_000                       # large, so estimates sit tight on the truth
failures = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<62} got {got:+.4f}  want {want:+.4f}")
    if not ok:
        failures.append(name)


print("\nBRIER & LOG LOSS — the basics")
y = RNG.random(N) < 0.5
check("perfect predictions give Brier 0", M.brier(y, y.astype(float)), 0.0, 1e-12)
check("constant 0.5 on a coin flip gives Brier 0.25", M.brier(y, np.full(N, .5)), 0.25, 1e-12)
check("constant 0.5 gives log loss ln(2)", M.log_loss(y, np.full(N, .5)), np.log(2), 1e-9)
base = float(y.mean())
check("predicting the base rate gives Brier skill 0",
      M.brier_skill(y, np.full(N, base), base), 0.0, 1e-9)
check("perfect predictions give Brier skill 1",
      M.brier_skill(y, y.astype(float), base), 1.0, 1e-9)

print("\nCALIBRATION SLOPE — too extreme vs too timid")
z = RNG.normal(0, 1.5, N)
y = RNG.random(N) < expit(z)           # the truth is sigmoid(z)
check("a correct model has slope 1", M.calibration_slope(y, expit(z)), 1.0, 0.03)
check("predicting sigmoid(2z) — overconfident — has slope 0.5",
      M.calibration_slope(y, expit(2 * z)), 0.5, 0.02)
check("predicting sigmoid(z/2) — too timid — has slope 2",
      M.calibration_slope(y, expit(z / 2)), 2.0, 0.06)

print("\nCALIBRATION INTERCEPT — right on average?")
y = RNG.random(N) < expit(z + 0.8)     # the truth is shifted up by 0.8 logits
check("a model missing a +0.8 logit shift has intercept +0.8",
      M.calibration_intercept(y, expit(z)), 0.8, 0.03)
check("the corrected model has intercept 0",
      M.calibration_intercept(y, expit(z + 0.8)), 0.0, 0.03)
check("intercept is positive when the model UNDER-predicts",
      float(M.calibration_intercept(y, expit(z)) > 0), 1.0, 0)

print("\nECE — average gap between predicted and observed")
p = RNG.uniform(0.02, 0.98, N)
y = RNG.random(N) < p                  # perfectly calibrated by construction
check("a perfectly calibrated model has ECE near 0 at large n", M.ece(y, p), 0.0, 0.005)
y_off = RNG.random(N) < np.clip(p + 0.10, 0, 1)
check("a model everywhere 10 points too low has ECE near 0.10",
      M.ece(y_off, p), 0.10, 0.01)
small = RNG.uniform(0.1, 0.9, 800)
floor = M.ece_noise_floor(small)
print(f"        (with only 800 pitches, a PERFECT model still scores ECE ~{floor:.3f} — "
      f"that's the noise floor)")
check("the noise floor is above zero for a small sample", float(floor > 0.01), 1.0, 0)

print("\nAUC — ranking only")
y = RNG.random(N) < 0.5
check("random scores give AUC 0.5", M.auc(y, RNG.random(N)), 0.5, 0.01)
check("perfect ranking gives AUC 1", M.auc(y, y + RNG.random(N) * 0.1), 1.0, 1e-9)
p = expit(z)
y = RNG.random(N) < p
check("halving every prediction leaves AUC unchanged — it ignores scale",
      M.auc(y, p * 0.5) - M.auc(y, p), 0.0, 1e-12)
check("...while the halved predictions are badly miscalibrated (ECE > 0.15)",
      float(M.ece(y, p * 0.5) > 0.15), 1.0, 0)

print("\nMID-RANGE CHECK — calibration where the decisions are")
p = np.r_[RNG.uniform(0.0, 0.05, N // 2), RNG.uniform(0.2, 0.8, N // 2)]
y = RNG.random(len(p)) < p
mr = M.midrange_check(y, p)
check("a calibrated model: observed rate matches mean prediction in 20-80%",
      mr["obs_rate"] - mr["mean_pred"], 0.0, 0.01)
p_bad = np.where((p >= 0.2) & (p <= 0.8), np.clip(p + 0.15, 0, 0.8), p)
mr_bad = M.midrange_check(y, p_bad)
check("a model 15 points too high in the middle is caught (gap ~ -0.15)",
      mr_bad["obs_rate"] - mr_bad["mean_pred"], -0.15, 0.03)
check("...even though its overall ECE looks modest (< 0.10)",
      float(M.ece(y, p_bad) < 0.10), 1.0, 0)

print("\nRELIABILITY TABLE & BOOTSTRAP — plumbing")
p = RNG.uniform(0, 1, 5000)
y = RNG.random(5000) < p
tab = M.reliability_table(y, p)
check("bin counts add up to n", float(sum(r["n"] for r in tab)), 5000.0, 0)
check("every observed rate sits inside its own 95% interval",
      float(all(r["ci_lo"] <= r["obs_rate"] <= r["ci_hi"] for r in tab)), 1.0, 0)
games = RNG.integers(0, 120, 5000)
ci, diffs = M.cluster_bootstrap(y, {"a": p, "b": np.full(5000, .5)}, games, 0.5, B=200)
lo, hi = ci["a"]["brier"]
check("the true model's Brier lies inside its bootstrap interval",
      float(lo <= M.brier(y, p) <= hi), 1.0, 0)
mean_d, dlo, dhi = diffs[("a", "b")]["brier"]
check("the informed model beats a coin flip (Brier difference CI below 0)",
      float(dhi < 0), 1.0, 0)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL METRIC TESTS PASSED")
