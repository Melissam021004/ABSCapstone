#!/usr/bin/env python3
"""
test_abs_models.py — proves the hand-written models are correct.

The logistic regression and the GAM in src/abs_models.py are written out in
numpy rather than imported. That's only acceptable if they can be shown to give
the right answers, so each test here builds data where the right answer is
known, fits the model, and checks it recovers that answer.

    python tests/test_abs_models.py

If you have statsmodels installed, you can add one more independent check of
your own:

    import statsmodels.api as sm
    sm.Logit(y, X).fit().params   # should equal Logistic(...).fit(...).beta
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.abs_models import Design, Logistic, GAM, lr_test  # noqa: E402

RNG = np.random.default_rng(7)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


# ----------------------------------------------------------------- logistic
print("\nLOGISTIC — matches sklearn's unpenalised maximum likelihood")
n = 20_000
df = pd.DataFrame({"x1": RNG.normal(0, 1, n), "x2": RNG.normal(3, 2, n),
                   "g": RNG.choice(["a", "b", "c"], n, p=[.5, .3, .2])})
eta = -0.4 + 1.2 * df.x1 - 0.3 * df.x2 + np.select([df.g == "b", df.g == "c"], [0.7, -0.5], 0)
y = (RNG.random(n) < expit(eta)).astype(int)
m = Logistic(Design(numeric=["x1", "x2"], categorical=["g"])).fit(df, y)

from sklearn.linear_model import LogisticRegression  # noqa: E402
X, names = m.design.transform(df)
with warnings.catch_warnings():      # sklearn renamed this argument in 1.8
    warnings.simplefilter("ignore")
    sk = LogisticRegression(penalty=None, fit_intercept=False, tol=1e-12,
                            max_iter=5000).fit(X, y)
diff = float(np.max(np.abs(sk.coef_[0] - m.beta)))
check("coefficients equal sklearn's to 1e-5", diff < 1e-5, f"max diff {diff:.1e}")

true = {"(intercept)": -0.4, "x1": 1.2, "x2": -0.3, "g=b": 0.7, "g=c": -0.5}
t = m.table().set_index("term")
inside = all(t.loc[k, "coef"] - 3 * t.loc[k, "se"] <= v <= t.loc[k, "coef"] + 3 * t.loc[k, "se"]
             for k, v in true.items())
check("every true coefficient recovered within 3 standard errors", inside)
check("reference level is the most common one ('a')",
      "g=a" not in names and "g=b" in names)

print("\nLOGISTIC — the confidence intervals mean what they claim")
cover, sims = 0, 300
for _ in range(sims):
    d = pd.DataFrame({"x": RNG.normal(0, 1, 1500)})
    yy = (RNG.random(1500) < expit(0.3 + 0.8 * d.x)).astype(int)
    tt = Logistic(Design(numeric=["x"])).fit(d, yy).table().set_index("term")
    lo = tt.loc["x", "coef"] - 1.96 * tt.loc["x", "se"]
    hi = tt.loc["x", "coef"] + 1.96 * tt.loc["x", "se"]
    cover += lo <= 0.8 <= hi
rate = cover / sims
check("a 95% interval contains the true slope ~95% of the time",
      0.92 <= rate <= 0.98, f"{rate:.1%} of {sims} simulated datasets")

print("\nLIKELIHOOD-RATIO TEST — right false-alarm rate")
rej = 0
for _ in range(300):
    d = pd.DataFrame({"x": RNG.normal(0, 1, 1000), "junk": RNG.normal(0, 1, 1000)})
    yy = (RNG.random(1000) < expit(0.5 * d.x)).astype(int)
    a = Logistic(Design(numeric=["x"])).fit(d, yy)
    b = Logistic(Design(numeric=["x", "junk"])).fit(d, yy)
    rej += lr_test(a, b)[2] < 0.05
check("adding a useless variable 'passes' p < 0.05 only ~5% of the time",
      0.02 <= rej / 300 <= 0.09, f"{rej / 300:.1%}")
d = pd.DataFrame({"x": RNG.normal(0, 1, 3000), "z": RNG.normal(0, 1, 3000)})
yy = (RNG.random(3000) < expit(0.5 * d.x + 0.6 * d.z)).astype(int)
p = lr_test(Logistic(Design(numeric=["x"])).fit(d, yy),
            Logistic(Design(numeric=["x", "z"])).fit(d, yy))[2]
check("adding a real variable is detected", p < 1e-6, f"p = {p:.1e}")

print("\nLOGISTIC — robustness")
d = pd.DataFrame({"x": np.r_[RNG.normal(-3, .5, 300), RNG.normal(3, .5, 300)]})
yy = np.r_[np.zeros(300), np.ones(300)].astype(int)          # perfectly separated
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s = Logistic(Design(numeric=["x"])).fit(d, yy)
check("perfectly separated data doesn't crash, and says so", "separation" in s.note,
      s.note[:48])
new = pd.DataFrame({"x1": [0.0], "x2": [3.0], "g": ["never_seen"]})
check("an unseen category at prediction time is treated as the reference",
      abs(m.predict(new)[0] - m.predict(new.assign(g="a"))[0]) < 1e-12)

# A category with 50 rows that ALL stood: its coefficient wants to be -infinity.
d = pd.DataFrame({"x": RNG.normal(0, 1, 3000),
                  "g": np.r_[["common"] * 2950, ["doomed"] * 50]})
yy = (RNG.random(3000) < expit(0.8 * d.x)).astype(int)
yy[-50:] = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    q = Logistic(Design(numeric=["x"], categorical=["g"])).fit(d, yy)
t = q.table().set_index("term")
check("a category whose rows all went one way is caught and fit with a weak prior",
      "separation" in q.note and np.isfinite(t.loc["g=doomed", "coef"]),
      f"coef {t.loc['g=doomed', 'coef']:.1f}")
check("...without disturbing the well-identified slope",
      abs(t.loc["x", "coef"] - 0.8) < 0.15, f"x slope {t.loc['x', 'coef']:.2f}")
d2 = d.assign(g=np.r_[["common"] * 2985, ["rare"] * 15])
r = Logistic(Design(numeric=["x"], categorical=["g"])).fit(d2, yy)
check("a category with only 15 rows is folded into the reference level",
      r.design.folded.get("g") == ["rare"] and "g" in r.design.dropped)
pb, lo_b, hi_b = m.predict_band(df.head(200))
check("predict_band: the 95% band always contains the prediction",
      bool(np.all((lo_b <= pb) & (pb <= hi_b))))

# ----------------------------------------------------------------- GAM
print("\nGAM — recovers a known curve")
n = 8000
d = pd.DataFrame({"x": RNG.uniform(-3, 3, n)})
truth = lambda x: 2.0 * np.sin(1.5 * x)                     # noqa: E731
yy = (RNG.random(n) < expit(truth(d.x))).astype(int)
g = GAM(Design(), smooth=["x"]).fit(d, yy)
grid = np.linspace(-2.6, 2.6, 60)
f, lo, hi = g.curve("s(x)", grid)
fitted = f + g.beta[0]
err = float(np.max(np.abs(fitted - truth(grid))))
cov = float(np.mean((lo + g.beta[0] <= truth(grid)) & (truth(grid) <= hi + g.beta[0])))
check("fitted curve tracks 2·sin(1.5x) closely", err < 0.45, f"max error {err:.2f} logits")
check("the 95% band contains the true curve almost everywhere", cov >= 0.85,
      f"{cov:.0%} of grid points")
check("EDF shows real curvature was found", g.edf["s(x)"] > 4, f"EDF {g.edf['s(x)']:.1f}")

print("\nGAM — shrinks to a straight line when the truth is straight")
yy = (RNG.random(n) < expit(0.9 * d.x)).astype(int)
g2 = GAM(Design(), smooth=["x"]).fit(d, yy)
check("EDF near 1 for a truly linear effect", g2.edf["s(x)"] < 2.5,
      f"EDF {g2.edf['s(x)']:.2f}")

# The failure this guards against: the smoothing choice used to depend on how
# rows were split into cross-validation folds, so the same data could give a
# straight line on one computer and a wiggly curve on another. REML uses all
# rows at once, so reordering them must not change anything.
order = RNG.permutation(n)
g2b = GAM(Design(), smooth=["x"]).fit(d.iloc[order].reset_index(drop=True), yy[order])
check("same data in a different row order -> same smoothing and same curve",
      g2b.lam == g2.lam and abs(g2b.edf["s(x)"] - g2.edf["s(x)"]) < 1e-6,
      f"λ {g2.lam:g} vs {g2b.lam:g}")

print("\nGAM — a separate curve per group ('by' smooth)")
d = pd.DataFrame({"x": RNG.uniform(-2, 2, n), "grp": RNG.choice(["steep", "flat"], n)})
slope = np.where(d.grp == "steep", 2.5, 0.6)
yy = (RNG.random(n) < expit(slope * d.x)).astype(int)
g3 = GAM(Design(categorical=["grp"]), smooth=["x"], by={"x": "grp"}).fit(d, yy)
xs = np.array([-1.0, 1.0])
rise_steep = np.diff(g3.curve("s(x | grp=steep)", xs)[0])[0] / 2
rise_flat = np.diff(g3.curve("s(x | grp=flat)", xs)[0])[0] / 2
check("the 'steep' group's curve rises about 2.5 per unit", 1.9 < rise_steep < 3.1,
      f"{rise_steep:.2f}")
check("the 'flat' group's curve rises about 0.6 per unit", 0.3 < rise_flat < 0.9,
      f"{rise_flat:.2f}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL MODEL TESTS PASSED")
