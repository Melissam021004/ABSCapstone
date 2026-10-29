"""
abs_calibration.py — Week 9. Turn a model's scores into probabilities you can trust.

A model can rank challenges perfectly and still be wrong about HOW likely
each one is. Calibration is a small second model, fitted on data the first
model never saw, that maps "the model says p" to "what actually happens when
the model says p".

Three options, simplest first (the order matters: ties go to the simpler one):

    none       use the model's probabilities as they are             0 parameters
    platt      logit(p_new) = a + b * logit(p)                       2 parameters
               shifts (a) and stretches/squeezes (b) the log-odds.
               Fixes "right on average?" (a) and "too extreme?" (b) —
               exactly the two numbers the calibration intercept and
               slope measure. Smooth, keeps the ranking.
    isotonic   any non-decreasing step function of p                 one step per
               (pool-adjacent-violators). Can fix any monotone shape,  block of the
               but it is greedy: with ~1,500 challenges it can learn   data
               noise, and it turns p(s) into a staircase.

Every calibrator here is saved as plain JSON (no pickles), so the locked
production model loads on any machine and any library version.

    cal = Calibrator("platt").fit(p_august, y_august)
    p_new = cal.predict(p)
    Calibrator.from_json(cal.to_json())      # identical predictions

`cross_fit` gives HONEST calibrated predictions for the calibration set
itself: each game's challenges are calibrated by a calibrator that never saw
that game. That's how the three options are compared without a fourth data set.
"""

import json

import numpy as np
from scipy.special import expit

METHODS = ["none", "platt", "isotonic"]          # simplest first
ISO_CLIP = 0.001       # isotonic never says "impossible": outputs live in [0.001, 0.999]
EPS = 1e-6
PLATT_RIDGE = 1e-4     # a whisper of shrinkage on the slope so near-perfect separation
                       # can't send it to infinity; invisible otherwise


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _platt_fit(p, y, ridge=PLATT_RIDGE):
    """Newton–Raphson for  logit P(y=1) = a + b·logit(p), ridge on b only."""
    x = _logit(p)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    pen = np.array([0.0, ridge * len(y)])
    for _ in range(100):
        mu = expit(X @ beta)
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + np.diag(pen)
        g = X.T @ (y - mu) - pen * (beta - np.array([0.0, 1.0]))   # shrink b toward 1
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


class Calibrator:
    """One fitted calibration map. See the module docstring."""

    def __init__(self, method="platt"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.params = {}
        self.n_fit = 0

    # ------------------------------------------------------------ fit
    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        y = np.asarray(y, int)
        self.n_fit = int(len(y))
        if self.method == "platt":
            a, b = _platt_fit(p, y)
            self.params = {"a": a, "b": b}
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=ISO_CLIP, y_max=1 - ISO_CLIP, increasing=True,
                                     out_of_bounds="clip").fit(p, y)
            self.params = {"x": [float(v) for v in iso.X_thresholds_],
                           "y": [float(v) for v in iso.y_thresholds_]}
        return self

    # ------------------------------------------------------------ use
    def predict(self, p):
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        if self.method == "none":
            out = p
        elif self.method == "platt":
            out = expit(self.params["a"] + self.params["b"] * _logit(p))
        else:
            # Linear interpolation between the steps' corners, flat beyond the ends —
            # exactly what sklearn's IsotonicRegression.predict does with out_of_bounds="clip".
            out = np.interp(p, self.params["x"], self.params["y"])
        return np.clip(out, EPS, 1 - EPS)

    @property
    def n_params(self):
        if self.method == "none":
            return 0
        if self.method == "platt":
            return 2
        return len(set(self.params.get("y", [])))     # distinct step heights

    def describe(self):
        if self.method == "none":
            return "no calibration — the model's own probabilities"
        if self.method == "platt":
            a, b = self.params["a"], self.params["b"]
            return (f"Platt scaling: logit(p_new) = {a:+.3f} + {b:.3f} × logit(p) "
                    f"(fitted on {self.n_fit:,} challenges)")
        return (f"isotonic regression: {self.n_params} steps, outputs kept in "
                f"[{ISO_CLIP}, {1 - ISO_CLIP}] (fitted on {self.n_fit:,} challenges)")

    # ------------------------------------------------------------ save / load
    def to_json(self):
        return json.dumps({"method": self.method, "params": self.params, "n_fit": self.n_fit,
                           "iso_clip": ISO_CLIP})

    @classmethod
    def from_json(cls, text):
        d = json.loads(text) if isinstance(text, str) else text
        c = cls(d["method"])
        c.params, c.n_fit = d["params"], int(d.get("n_fit", 0))
        return c


def cross_fit(p, y, groups, method, k=5, seed=2026):
    """
    Out-of-fold calibrated predictions: split the calibration set into k folds
    of whole GAMES, fit the calibrator on k−1 folds, apply it to the held-out
    fold, rotate. Every challenge ends up calibrated by a map that never saw
    its game — so scoring these is an honest estimate of how the calibrator
    will do on new games.
    """
    from .abs_pipeline import grouped_folds
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    out = np.full(len(p), np.nan)
    if method == "none":
        return np.clip(p, EPS, 1 - EPS)
    for fit_idx, held in grouped_folds(groups, k=k, seed=seed):
        out[held] = Calibrator(method).fit(p[fit_idx], y[fit_idx]).predict(p[held])
    return out


__all__ = ["METHODS", "ISO_CLIP", "Calibrator", "cross_fit"]
