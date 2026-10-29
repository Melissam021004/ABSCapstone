"""
abs_metrics.py — the evaluation suite for every probability model in the project.

Week 7 uses it for the baselines, Week 8 for the boosted models, Week 9 for
calibration. Keeping it in one module means every model in the capstone is
graded by literally the same code, which is the only way a comparison between
them means anything.

Everything here takes two arrays:
    y  — what happened (1 = overturned, 0 = call stood)
    p  — what the model said the probability was

and returns a number or a small table. Nothing here fits the model being
evaluated; it only scores predictions that already exist.

The functions, in the order the report presents them:

    brier, brier_skill      how far off the probabilities were, on average
    log_loss                the same, but brutal on confident mistakes
    calibration_intercept   is the model right ON AVERAGE?  (ideal 0)
    calibration_slope       are its predictions too extreme or too timid?  (ideal 1)
    ece                     average gap between predicted and observed, by bin
    auc                     can it rank overturned above stood?  (secondary)
    reliability_table       the data behind the reliability diagram
    midrange_check          calibration among the uncertain challenges only
    cluster_bootstrap       uncertainty on all of the above, resampling by game
"""

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

EPS = 1e-6


# ------------------------------------------------------------------ helpers
def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _irls(X, y, offset=None, max_iter=100, tol=1e-10):
    """
    Newton-Raphson for a small unpenalised logistic regression. Used here only
    for the one- and two-parameter calibration fits, so it is kept minimal.
    Returns coefficients, or NaNs if the fit can't be made (e.g. constant input).
    """
    n, k = X.shape
    off = np.zeros(n) if offset is None else offset
    beta = np.zeros(k)
    for _ in range(max_iter):
        mu = expit(X @ beta + off)
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None])
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.full(k, np.nan)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    if not np.all(np.isfinite(beta)):
        return np.full(k, np.nan)
    return beta


# ------------------------------------------------------------------ scores
def brier(y, p):
    """Mean squared error of the probabilities. 0 is perfect. Lower is better."""
    y = np.asarray(y, float)
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def brier_skill(y, p, p_ref):
    """
    Share of the reference model's Brier score that this model removes.

        BSS = 1 - Brier(model) / Brier(reference)

    The reference is "predict the training base rate for every pitch" — the
    best you can do without looking at the pitch at all. 0 means no better than
    that, 1 is perfect, negative means worse than not having a model.
    """
    ref = brier(y, np.full(len(y), p_ref))
    return float(1 - brier(y, p) / ref) if ref > 0 else np.nan


def log_loss(y, p):
    """
    Average negative log-likelihood. Lower is better.

    Like Brier, but a confident wrong answer (p = 0.99, call stood) costs
    enormously more than a hesitant one. A model feeding a decision process
    must not be confidently wrong, which is why it's reported alongside Brier.
    """
    y = np.asarray(y, float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_intercept(y, p):
    """
    Calibration-in-the-large. Fit  logit P(y=1) = a + 1·logit(p)  and report a.

    The slope is FIXED at 1 here; that's what makes `a` mean "is the model
    right on average". Ideal 0. Positive: the model under-predicts overturns.
    Negative: it over-predicts them.

    Not the same number as the intercept of the slope model below — reporting
    that one by mistake is a common error.
    """
    y = np.asarray(y, float)
    lp = logit(p)
    a = _irls(np.ones((len(y), 1)), y, offset=lp)
    return float(a[0])


def calibration_slope(y, p):
    """
    Fit  logit P(y=1) = a + b·logit(p)  and report b. Ideal 1.

        b < 1   predictions too EXTREME — the model is overconfident. When it
                says 90%, the truth is nearer 75%. The classic overfitting sign.
        b > 1   predictions too TIMID — the model hedges toward 50% more than
                the data warrant.
    """
    y = np.asarray(y, float)
    lp = logit(p)
    if np.std(lp) < 1e-9:
        return np.nan
    X = np.column_stack([np.ones(len(y)), lp])
    b = _irls(X, y)
    return float(b[1])


def _bins(p, n_bins, strategy):
    p = np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        return np.zeros(len(p), dtype=int), 1
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return idx, len(edges) - 1


def ece(y, p, n_bins=10, strategy="quantile"):
    """
    Expected calibration error: the count-weighted average of
    |observed rate − mean predicted| across bins. Ideal 0.

    Two things to know before quoting it:
      * It depends on the binning. We use 10 equal-COUNT bins, so every bin
        holds the same number of pitches. Say so when you report it.
      * It is biased upward on small samples: even a perfectly calibrated model
        scores above zero, because bin rates are noisy. `ece_noise_floor` below
        estimates how big that floor is for your sample, so you can tell real
        miscalibration from sampling noise.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx, nb = _bins(p, n_bins, strategy)
    tot = 0.0
    for b in range(nb):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))


def ece_noise_floor(p, n_bins=10, n_sim=300, seed=0):
    """
    The ECE a PERFECTLY calibrated model would get on a sample this size.
    Simulates outcomes as y ~ Bernoulli(p) — so the model is exactly right by
    construction — and averages the ECE that results. Anything your real ECE
    has beyond this is genuine miscalibration; anything below it is noise.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    return float(np.mean([ece(rng.random(len(p)) < p, p, n_bins) for _ in range(n_sim)]))


def auc(y, p):
    """
    Area under the ROC curve: the probability that a randomly chosen overturned
    challenge gets a higher predicted p than a randomly chosen one that stood.
    0.5 is a coin flip, 1.0 is perfect ranking.

    AUC only cares about ORDER. Multiply every prediction by 0.5 and AUC doesn't
    move — but every probability is now wrong. That's why it's the secondary
    metric here: the MDP needs the probabilities themselves to be right.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def wilson(k, n, z=1.96):
    """95% Wilson interval for a binomial proportion — honest at small n."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def reliability_table(y, p, n_bins=10, strategy="quantile"):
    """Per-bin mean prediction, observed rate, count and 95% interval."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx, nb = _bins(p, n_bins, strategy)
    rows = []
    for b in range(nb):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        k = int(y[m].sum())
        lo, hi = wilson(k, n)
        rows.append({"bin": b, "n": n, "mean_pred": float(p[m].mean()),
                     "obs_rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return rows


def midrange_check(y, p, lo=0.2, hi=0.8):
    """
    Calibration where decisions actually get made.

    Most challenges are easy: the pitch is clearly in or clearly out and every
    model says ~0% or ~100%. The MDP's challenge-or-hold decision only turns on
    the UNCERTAIN ones, so this asks: among the challenges the model put
    between `lo` and `hi`, what did it predict on average, and what share were
    actually overturned? Those two numbers should match.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = (p >= lo) & (p <= hi)
    n = int(m.sum())
    if not n:
        return {"n": 0, "mean_pred": np.nan, "obs_rate": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    k = int(y[m].sum())
    ci = wilson(k, n)
    return {"n": n, "mean_pred": float(p[m].mean()), "obs_rate": k / n,
            "ci_lo": ci[0], "ci_hi": ci[1]}


def metric_suite(y, p, p_ref):
    """Every headline metric for one model on one sample."""
    return {
        "brier": brier(y, p),
        "brier_skill": brier_skill(y, p, p_ref),
        "log_loss": log_loss(y, p),
        "cal_intercept": calibration_intercept(y, p),
        "cal_slope": calibration_slope(y, p),
        "ece": ece(y, p),
        "auc": auc(y, p),
    }


METRICS = ["brier", "brier_skill", "log_loss", "cal_intercept", "cal_slope", "ece", "auc"]
# For each metric, is a LOWER number better? (Used to orient comparisons.)
LOWER_IS_BETTER = {"brier": True, "brier_skill": False, "log_loss": True,
                   "cal_intercept": None, "cal_slope": None, "ece": True, "auc": False}
IDEAL = {"cal_intercept": 0.0, "cal_slope": 1.0}


# ------------------------------------------------------------------ bootstrap
def cluster_bootstrap(y, preds, groups, p_ref, B=1000, seed=0):
    """
    Confidence intervals for every metric, for every model, resampling GAMES.

    Why games and not rows: challenges in the same game share an umpire, a
    catcher, a park and a score. They are not independent draws, and treating
    them as independent makes the intervals too narrow — you'd conclude two
    models differ when the data can't actually tell them apart.

    `preds` is {model_name: predictions}. Every model is scored on the SAME
    resample each round, so differences between models are paired — that cuts
    out the shared noise and gives far tighter intervals on "is model A better
    than model B?" than comparing two separate intervals would.

    Returns {model: {metric: (lo, hi)}} and {(a, b): {metric: (mean, lo, hi)}}
    for every ordered pair, where the difference is a minus b.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    names = list(preds)
    P = {k: np.asarray(v, float) for k, v in preds.items()}

    draws = {k: {m: [] for m in METRICS} for k in names}
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([members[i] for i in pick])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        for k in names:
            s = metric_suite(yy, P[k][rows], p_ref)
            for m in METRICS:
                draws[k][m].append(s[m])

    ci = {k: {m: _pct(draws[k][m]) for m in METRICS} for k in names}
    diffs = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            diffs[(a, b)] = {}
            for m in METRICS:
                d = np.array(draws[a][m]) - np.array(draws[b][m])
                d = d[np.isfinite(d)]
                diffs[(a, b)][m] = ((float(d.mean()),) + _pct(d)) if len(d) else (np.nan,) * 3
    return ci, diffs


def _pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return (np.nan, np.nan)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
