"""
abs_baselines.py — the Week 7 model ladder, shared by Weeks 7, 8 and 9.

Week 8 has to compare the boosted models against EXACTLY the Week 7 models,
fitted the same way on the same rows. Keeping the ladder here (instead of
inside the Week 7 script) means every week builds them with one piece of code.

    MODEL_SPECS   key -> (label, kind, blocks)
    fit_model     fit one ladder model on the training challenges
    choose        the pre-registered "simplest model within noise of the best" rule
"""

import pandas as pd

from .abs_features import block_spec
from .abs_models import GAM, Design, Logistic

ALL4 = ["location", "situation", "pitch", "people"]

# key -> (label, kind, blocks). Blocks are defined in src/abs_features.py.
MODEL_SPECS = {
    "M0": ("M0 · base rate", "logistic", []),
    "M1": ("M1 · distance", "logistic", None),          # special: one feature
    "M2": ("M2 · location", "logistic", ["location"]),
    "M3": ("M3 · + situation", "logistic", ["location", "situation"]),
    "M4": ("M4 · + pitch", "logistic", ["location", "situation", "pitch"]),
    "M5": ("M5 · + people", "logistic", ALL4),
    "GAM-loc": ("GAM · location", "gam", ["location"]),
    "GAM-full": ("GAM · everything", "gam", ALL4),
}
ORDER = list(MODEL_SPECS)
# Simplest first. Used by the baseline-to-beat rule.
CANDIDATES = ["M1", "M2", "GAM-loc", "M3", "M4", "M5", "GAM-full"]
LOCATION_ONLY = ["M1", "M2", "GAM-loc"]

# (bigger, smaller, what the comparison asks)
STEPS = [
    ("M2", "M1", "+ edge & call direction"),
    ("M3", "M2", "+ situation"),
    ("M4", "M3", "+ pitch"),
    ("M5", "M4", "+ people"),
    ("GAM-loc", "M2", "curve instead of line (location)"),
    ("GAM-full", "M5", "curve instead of line (everything)"),
    ("M5", "M2", "everything vs location only"),
]
LR_PAIRS = [("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M4", "M3"), ("M5", "M4"), ("M5", "M2")]

MIN_EDGE_ROWS_FOR_BY = 60      # fewer training challenges at an edge -> one shared curve


def gam_spec(blocks):
    """The GAM's parametric part: the blocks minus the straight-line distance."""
    s = block_spec(blocks)
    s["slopes_by"] = {k: v for k, v in s["slopes_by"].items() if k != "dist_against_call"}
    return s


def fit_model(key, tr):
    """Fit one ladder model on the training challenges `tr` (needs a `y` column)."""
    label, kind, blocks = MODEL_SPECS[key]
    if key == "M1":
        return Logistic(Design(numeric=["dist_against_call"])).fit(tr, tr["y"])
    if kind == "logistic":
        return Logistic(Design(**block_spec(blocks))).fit(tr, tr["y"])
    by_ok = tr["edge"].value_counts().min() >= MIN_EDGE_ROWS_FOR_BY and tr["edge"].nunique() > 1
    g = GAM(Design(**gam_spec(blocks)), smooth=["dist_against_call"],
            by={"dist_against_call": "edge"} if by_ok else None)
    g.fit(tr, tr["y"])
    if not by_ok:
        g.note = (g.note + "; " if g.note else "") + \
            "one shared distance curve (an edge had too few training challenges for its own)"
    return g


def n_params(model):
    """Coefficients for a logistic, effective degrees of freedom for a GAM."""
    if isinstance(model, GAM):
        return model.edf_total
    return float(model.k)


def choose(candidates, point, diffs):
    """
    The baseline-to-beat rule, written down before looking at the numbers:

      1. Find the candidate with the lowest validation Brier score.
      2. Walk the candidates from simplest to most complex. Pick the FIRST one
         whose Brier can't be told apart from the best — its paired bootstrap
         95% interval for (this model − best) reaches zero or below.

    A model that's better by an amount August can't confirm isn't better; the
    simpler model wins ties. This is the same idea as the "one-standard-error
    rule" in cross-validation.

    Returns (best, chosen, table).
    """
    best = min(candidates, key=lambda k: point[k]["brier"])
    rows, chosen = [], None
    for k in candidates:
        if k == best:
            mean_, lo_, hi_ = 0.0, 0.0, 0.0
        else:
            mean_, lo_, hi_ = diffs[(k, best)]["brier"]
        tie = (k == best) or (lo_ <= 0)
        rows.append({"model": k, "d_vs_best": mean_, "lo": lo_, "hi": hi_, "tie": tie})
        if chosen is None and tie:
            chosen = k
    return best, chosen, pd.DataFrame(rows)


def calibration_ok(ci, ref_point=None, point=None):
    """
    Is a model's calibration acceptable? Intercept interval must contain 0 and
    slope interval must contain 1 — or, when a reference model is given, the
    point estimates must be no further from ideal than the reference's.
    """
    lo_i, hi_i = ci["cal_intercept"]
    lo_s, hi_s = ci["cal_slope"]
    int_ok = lo_i <= 0 <= hi_i
    slope_ok = lo_s <= 1 <= hi_s
    if ref_point is not None and point is not None:
        int_ok = int_ok or abs(point["cal_intercept"]) <= abs(ref_point["cal_intercept"])
        slope_ok = slope_ok or abs(point["cal_slope"] - 1) <= abs(ref_point["cal_slope"] - 1)
    return bool(int_ok and slope_ok)


__all__ = ["ALL4", "MODEL_SPECS", "ORDER", "CANDIDATES", "LOCATION_ONLY", "STEPS",
           "LR_PAIRS", "MIN_EDGE_ROWS_FOR_BY", "gam_spec", "fit_model", "n_params",
           "choose", "calibration_ok"]
