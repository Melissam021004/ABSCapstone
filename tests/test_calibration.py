#!/usr/bin/env python3
"""
test_calibration.py — does src/abs_calibration.py do what Week 9 relies on?

Each check plants a known answer and asks the code to find it:

  1. Platt scaling recovers a miscalibration we built in (a = +0.4, b = 0.7)
  2. ...and leaves an already-calibrated model alone (a ≈ 0, b ≈ 1)
  3. isotonic output never decreases, stays inside [0.001, 0.999], and matches
     scikit-learn's own IsotonicRegression to the last decimal
  4. JSON round trip: a saved calibrator predicts exactly like the original
  5. cross-fitting is honest: changing one game's outcomes cannot change that
     game's own calibrated predictions
  6. on an overconfident model, cross-fitted Platt really lowers the Brier score
  7. the Week 9 rule picks "none" for a model that is already calibrated

    python tests/test_calibration.py        (a few seconds)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import abs_metrics as M                                     # noqa: E402
from src.abs_baselines import choose                                 # noqa: E402
from src.abs_calibration import ISO_CLIP, METHODS, Calibrator, cross_fit   # noqa: E402

RNG = np.random.default_rng(11)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def fake(n, a=0.0, b=1.0):
    """Model scores p, and outcomes drawn from the TRUE probability expit(a + b·logit(p))."""
    p = expit(RNG.normal(0, 2.2, n))
    y = (RNG.random(n) < expit(a + b * logit(p))).astype(int)
    return p, y


print("\nPlatt scaling")
p, y = fake(40000, a=0.4, b=0.7)
c = Calibrator("platt").fit(p, y)
check("recovers the planted intercept a = +0.40", abs(c.params["a"] - 0.4) < 0.06,
      f"a = {c.params['a']:+.3f}")
check("recovers the planted slope b = 0.70", abs(c.params["b"] - 0.7) < 0.04,
      f"b = {c.params['b']:.3f}")
p, y = fake(40000)
c = Calibrator("platt").fit(p, y)
check("leaves a calibrated model alone (a ≈ 0, b ≈ 1)",
      abs(c.params["a"]) < 0.06 and abs(c.params["b"] - 1) < 0.04,
      f"a = {c.params['a']:+.3f}, b = {c.params['b']:.3f}")

print("\nIsotonic regression")
p, y = fake(3000, a=-0.3, b=0.8)
c = Calibrator("isotonic").fit(p, y)
grid = np.linspace(0, 1, 2001)
out = c.predict(grid)
check("never decreases", bool(np.all(np.diff(out) >= -1e-12)))
check(f"outputs stay inside [{ISO_CLIP}, {1 - ISO_CLIP}]",
      out.min() >= ISO_CLIP - 1e-12 and out.max() <= 1 - ISO_CLIP + 1e-12,
      f"{out.min():.4f} – {out.max():.4f}")
from sklearn.isotonic import IsotonicRegression   # noqa: E402
sk = IsotonicRegression(y_min=ISO_CLIP, y_max=1 - ISO_CLIP, out_of_bounds="clip").fit(
    np.clip(p, 1e-6, 1 - 1e-6), y)
gap = float(np.max(np.abs(sk.predict(np.clip(grid, 1e-6, 1 - 1e-6)) - out)))
check("matches scikit-learn's IsotonicRegression exactly", gap < 1e-12, f"max gap {gap:.1e}")

print("\nSave / load")
for m_ in METHODS:
    c = Calibrator(m_).fit(p, y)
    c2 = Calibrator.from_json(c.to_json())
    gap = float(np.max(np.abs(c.predict(grid) - c2.predict(grid))))
    check(f"{m_}: JSON round trip gives identical predictions", gap == 0.0, f"max gap {gap:.1e}")

print("\nCross-fitting")
n_games = 300
games = np.repeat(np.arange(n_games), 5)
p, y = fake(len(games), a=0.5, b=0.6)
x1 = cross_fit(p, y, games, "platt", k=5, seed=3)
y2 = y.copy()
g0 = games == 17
y2[g0] = 1 - y2[g0]                                   # rewrite one game's outcomes
x2 = cross_fit(p, y2, games, "platt", k=5, seed=3)
check("a game's own predictions ignore its own outcomes",
      float(np.max(np.abs(x1[g0] - x2[g0]))) == 0.0)
check("...while other folds do move (the check can fail)",
      float(np.max(np.abs(x1[~g0] - x2[~g0]))) > 0)

p, y = fake(6000, a=0.0, b=0.55)                       # overconfident model
games = np.repeat(np.arange(1200), 5)
b_none = M.brier(y, p)
b_platt = M.brier(y, cross_fit(p, y, games, "platt", k=5, seed=1))
check("cross-fitted Platt lowers the Brier of an overconfident model", b_platt < b_none,
      f"{b_none:.4f} -> {b_platt:.4f}")

print("\nThe Week 9 rule on a model that's already calibrated")
p, y = fake(3000)
games = np.repeat(np.arange(600), 5)
preds = {f"x|{m_}": cross_fit(p, y, games, m_, k=5, seed=2) for m_ in METHODS}
point = {k: M.metric_suite(y, v, y.mean()) for k, v in preds.items()}
_, diffs = M.cluster_bootstrap(y, preds, games, y.mean(), B=300, seed=4)
best, pick, _ = choose(list(preds), point, diffs)
check("chooses 'none' (nothing to fix, so no extra parameters)", pick == "x|none",
      f"best by Brier: {best}, chosen: {pick}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL CALIBRATION CHECKS PASSED")
