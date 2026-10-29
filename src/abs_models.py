"""
abs_models.py — the two Week 7 baselines: logistic regression and a GAM.

Both are written out in numpy/scipy rather than imported from statsmodels or
pygam, for two reasons. First, they run on any machine with the standard stack
(pygam has had repeated compatibility problems with recent numpy/scipy releases). Second, every step is
visible: if a committee member asks how the smoothing was chosen or where an
interval came from, the answer is a few readable lines in this file.

Both are verified in tests/test_abs_models.py: the logistic against sklearn's
unpenalised fit to many decimal places, and the GAM against a curve whose true
shape is known in advance.

THE DESIGN SPEC
Both models take the same small spec describing how to turn a DataFrame into
numbers, so the logistic and the GAM can be compared on identical inputs:

    numeric       columns entered as-is (one coefficient each)
    categorical   columns one-hot encoded; the most common level is the
                  reference, so every other level reads as "compared with usual"
    smooth        (GAM only) columns entered as smooth curves
    by            (GAM only) {smooth_column: factor} — a separate curve for each
                  level of the factor, e.g. one distance curve per zone edge
    slopes_by     (logistic only) {numeric_column: factor} — a separate straight
                  line per level, the logistic version of the same idea
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import null_space
from scipy.special import expit
from scipy.stats import norm


# ============================================================ design matrices
class Design:
    """
    Learns encodings on the training data, then applies them unchanged.

    Categories seen fewer than `min_level_count` times in training are folded
    into the reference level: 12 rows can't support an effect of their own, and
    a level whose few rows all went the same way would send its coefficient to
    infinity. Columns with nothing to estimate (one level, or no variation) are
    dropped. Both are recorded in `folded` / `dropped` so the report can say so.
    """

    def __init__(self, numeric=(), categorical=(), slopes_by=None, min_level_count=30):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.slopes_by = dict(slopes_by or {})
        self.min_level_count = min_level_count
        self.levels = {}
        self.folded = {}
        self.dropped = []
        self.medians = {}

    def fit(self, df):
        self.dropped, self.folded, self.levels = [], {}, {}
        for c in self.categorical:
            vc = df[c].astype(str).value_counts()
            keep = [lv for lv in vc.index if vc[lv] >= self.min_level_count]
            rare = [lv for lv in vc.index if lv not in keep]
            if rare:
                self.folded[c] = rare
            if len(keep) < 2:
                self.dropped.append(c)       # one usable level: nothing to estimate
                continue
            self.levels[c] = keep            # first = most common = reference
        for c in list(self.slopes_by.values()):
            if c not in self.levels:
                vc = df[c].astype(str).value_counts()
                self.levels[c] = list(vc.index)
        for c in self.numeric + list(self.slopes_by):
            self.medians[c] = float(df[c].median()) if df[c].notna().any() else 0.0
            if df[c].notna().sum() == 0 or df[c].std(skipna=True) == 0:
                self.dropped.append(c)
        return self

    def transform(self, df):
        cols, names = [np.ones(len(df))], ["(intercept)"]
        for c in self.numeric:
            if c in self.dropped:
                continue
            cols.append(df[c].fillna(self.medians[c]).to_numpy(float))
            names.append(c)
        for num, fac in self.slopes_by.items():
            if num in self.dropped:
                continue
            x = df[num].fillna(self.medians[num]).to_numpy(float)
            lv = df[fac].astype(str).to_numpy()
            for level in self.levels[fac]:
                cols.append(x * (lv == level))
                names.append(f"{num} | {fac}={level}")
        for c in self.categorical:
            if c in self.dropped:
                continue
            v = df[c].astype(str).to_numpy()
            for level in self.levels[c][1:]:
                cols.append((v == level).astype(float))
                names.append(f"{c}={level}")
        return np.column_stack(cols), names


# ============================================================ logistic
class Logistic:
    """
    Maximum-likelihood logistic regression by Newton-Raphson (IRLS), with
    standard errors from the inverse Hessian — the same estimator statsmodels'
    Logit uses, so coefficients, intervals and log-likelihoods match it.

    If the data are separable (some feature or category predicts the outcome
    perfectly in training) the unpenalised estimate runs off to infinity and
    Newton's method never settles. When that happens the fit is redone with a
    weak prior — normal, sd 2.5 on the change in log-odds per standard
    deviation of each input — and `self.note` says so. A prior that weak barely
    moves well-identified coefficients; it only stops the runaway ones.
    """

    PRIOR_SD = 2.5

    def __init__(self, design):
        self.design = design
        self.note = ""

    def fit(self, df, y, max_iter=100, tol=1e-9):
        self.design.fit(df)
        X, self.names = self.design.transform(df)
        y = np.asarray(y, float)
        self.note = ""
        R = np.zeros((X.shape[1], X.shape[1]))
        beta, ok = self._newton(X, y, R, max_iter, tol)
        if not ok:
            sd = X.std(axis=0)
            sd[0] = 0.0                                   # never penalise the intercept
            R = np.diag((sd / self.PRIOR_SD) ** 2)
            beta, ok = self._newton(X, y, R, max_iter, tol)
            self.note = ("did not converge unpenalised (separation: something predicts "
                         "the outcome perfectly in training); refit with a weak prior")
            if not ok:
                self.note += " — and STILL did not converge; do not trust this fit"
        self.beta, self.prior = beta, R
        mu = expit(X @ beta)
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + R
        self.cov = np.linalg.pinv(H)
        self.se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        self.loglik = float(np.sum(y * np.log(np.clip(mu, 1e-12, 1))
                                   + (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))))
        self.n, self.k = len(y), len(beta)
        self.aic = 2 * self.k - 2 * self.loglik
        return self

    @staticmethod
    def _newton(X, y, R, max_iter, tol):
        """
        Newton-Raphson with step-halving (each step must not lower the
        penalised log-likelihood). Returns (beta, converged). Under separation
        the steps never shrink, so it returns converged=False after max_iter.
        """
        def pll(b):
            eta = X @ b
            return float(np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * b @ R @ b)

        beta = np.zeros(X.shape[1])
        cur = pll(beta)
        for _ in range(max_iter):
            mu = expit(X @ beta)
            w = mu * (1 - mu)
            H = X.T @ (X * w[:, None]) + R
            g = X.T @ (y - mu) - R @ beta
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return beta, False
            if not np.all(np.isfinite(step)):
                return beta, False
            t = 1.0
            while True:
                new = beta + t * step
                val = pll(new)
                if val >= cur - 1e-9 or t < 1e-4:
                    break
                t /= 2
            beta, cur = new, val
            if np.max(np.abs(t * step)) < tol:
                return beta, True
        return beta, False

    def predict(self, df):
        X, _ = self.design.transform(df)
        return expit(X @ self.beta)

    def predict_band(self, df):
        """Probability with a pointwise 95% band, from the coefficient covariance."""
        X, _ = self.design.transform(df)
        eta = X @ self.beta
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", X, self.cov, X), 0, None))
        return expit(eta), expit(eta - 1.96 * se), expit(eta + 1.96 * se)

    def table(self):
        """Coefficients, odds ratios and 95% Wald intervals."""
        z = self.beta / np.where(self.se > 0, self.se, np.nan)
        lo, hi = self.beta - 1.96 * self.se, self.beta + 1.96 * self.se
        return pd.DataFrame({
            "term": self.names, "coef": self.beta, "se": self.se, "z": z,
            "p_value": 2 * norm.sf(np.abs(z)),
            "odds_ratio": np.exp(self.beta),
            "or_lo": np.exp(lo), "or_hi": np.exp(hi)})


def lr_test(small, big):
    """
    Likelihood-ratio test between two NESTED logistic fits on the same rows.
    Answers: does the bigger model explain the TRAINING data better than chance
    would allow? It says nothing about prediction — that's what the validation
    metrics are for.
    """
    from scipy.stats import chi2
    stat = 2 * (big.loglik - small.loglik)
    df = big.k - small.k
    return float(stat), int(df), float(chi2.sf(stat, df)) if df > 0 else np.nan


# ============================================================ GAM
class _Smooth:
    """
    One smooth term: a cubic B-spline basis with a second-difference penalty
    (a "P-spline", Eilers & Marx 1996 — the construction pygam uses), and a
    sum-to-zero constraint so the curve can't drift into the intercept.
    """

    def __init__(self, col, n_knots=12, degree=3, by=None, level=None):
        self.col, self.n_knots, self.degree = col, n_knots, degree
        self.by, self.level = by, level

    def fit(self, df):
        x = df[self.col].to_numpy(float)
        mask = self._mask(df)
        xs = x[mask & np.isfinite(x)]
        lo, hi = np.percentile(xs, [0.5, 99.5]) if len(xs) > 20 else (xs.min(), xs.max())
        if hi <= lo:
            hi = lo + 1.0
        self.lo, self.hi = float(lo), float(hi)
        inner = np.linspace(lo, hi, self.n_knots)
        step = inner[1] - inner[0]
        self.t = np.concatenate([lo - step * np.arange(self.degree, 0, -1), inner,
                                 hi + step * np.arange(1, self.degree + 1)])
        B = self._raw(df)
        # Sum-to-zero over the rows this smooth applies to: reparameterise onto
        # the null space of the column means, so the curve averages to zero.
        cm = B[mask].mean(axis=0, keepdims=True)
        self.Z = null_space(cm)
        k = B.shape[1]
        D = np.diff(np.eye(k), n=2, axis=0)
        self.P = self.Z.T @ (D.T @ D) @ self.Z
        return self

    def _mask(self, df):
        if self.by is None:
            return np.ones(len(df), bool)
        return df[self.by].astype(str).to_numpy() == self.level

    def _raw(self, df, x=None):
        if x is None:
            x = df[self.col].to_numpy(float)
        # B-splines don't extrapolate sensibly; clamp to the fitted range.
        x = np.clip(np.where(np.isfinite(x), x, (self.lo + self.hi) / 2), self.lo, self.hi)
        k = len(self.t) - self.degree - 1
        B = np.column_stack([BSpline(self.t, np.eye(k)[j], self.degree, extrapolate=False)(x)
                             for j in range(k)])
        return np.nan_to_num(B)

    def basis(self, df):
        B = self._raw(df) @ self.Z
        if self.by is not None:
            B = B * self._mask(df)[:, None]
        return B

    def basis_at(self, xgrid):
        return self._raw(None, x=np.asarray(xgrid, float)) @ self.Z

    @property
    def name(self):
        return f"s({self.col})" if self.by is None else f"s({self.col} | {self.by}={self.level})"


class GAM:
    """
    Logistic GAM:   logit P(overturn) = parametric terms + Σ f_j(x_j)

    Each f_j is a P-spline. All smooths share one smoothing parameter λ, chosen
    by REML (restricted maximum likelihood, in the Laplace-approximate form
    R's mgcv uses for logistic models with method = "REML"). REML asks which
    amount of smoothing makes the training data most probable once the curve's
    own coefficients are averaged over. It uses all the training rows at once,
    so there are no folds, and the same data give the same λ on every machine.

    Why not cross-validation: an earlier version picked λ by 5-fold CV. When
    the true curve is nearly straight, the CV scores for λ values a million
    times apart differ by ~0.0002, so the "winner" depended on how games were
    split into folds and even on floating-point details of the computer —
    tests/test_abs_models.py caught it giving EDF 1.3 on one machine and 8.8 on
    another for the same data. REML is stable in exactly that situation.

    After fitting, for every smooth you get:
      * the curve f_j(x) with a 95% confidence band
      * its effective degrees of freedom (EDF): 1 means the data chose a
        straight line, larger means it found real curvature
    """

    def __init__(self, design, smooth=(), by=None, n_knots=12,
                 lambdas=tuple(10.0 ** np.linspace(-3, 6, 19))):
        self.design = design
        self.smooth_cols = list(smooth)
        self.by = dict(by or {})
        self.n_knots = n_knots
        self.lambdas = list(lambdas)
        self.note = ""

    def _build(self, df):
        Xp, names = self.design.transform(df)
        blocks, pens = [Xp], [np.zeros((Xp.shape[1], Xp.shape[1]))]
        self.spans = []
        start = Xp.shape[1]
        for s in self.smooths:
            B = s.basis(df)
            blocks.append(B)
            pens.append(s.P)
            self.spans.append((s, slice(start, start + B.shape[1])))
            start += B.shape[1]
        X = np.hstack(blocks)
        S = np.zeros((X.shape[1], X.shape[1]))
        i = 0
        for P in pens:
            S[i:i + P.shape[0], i:i + P.shape[0]] = P
            i += P.shape[0]
        return X, S, names

    @staticmethod
    def _pirls(X, y, S, lam, max_iter=60, tol=1e-8):
        beta = np.zeros(X.shape[1])
        Sl = lam * S
        Sl[1:, 1:] += 1e-8 * np.eye(X.shape[1] - 1)   # numerical floor only
        for _ in range(max_iter):
            mu = expit(X @ beta)
            w = mu * (1 - mu)
            H = X.T @ (X * w[:, None]) + Sl
            g = X.T @ (y - mu) - Sl @ beta
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            beta += step
            if np.max(np.abs(step)) < tol:
                break
        mu = expit(X @ beta)                            # curvature at the final fit
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None]) + Sl
        return beta, H

    def fit(self, df, y, groups=None, seed=None):
        """
        Fit on the training rows. `groups` and `seed` are accepted so older
        calls still work, but nothing uses them: REML needs no folds.
        """
        y = np.asarray(y, float)
        self.note = ""
        self.design.fit(df)
        self.smooths = []
        for c in self.smooth_cols:
            if c in self.by:
                for level in df[self.by[c]].astype(str).value_counts().index:
                    self.smooths.append(_Smooth(c, self.n_knots, by=self.by[c],
                                                level=level).fit(df))
            else:
                self.smooths.append(_Smooth(c, self.n_knots).fit(df))
        X, S, self.pnames = self._build(df)

        # --- choose λ by REML ------------------------------------------------
        # Laplace-approximate restricted log marginal likelihood (Wood 2011):
        #     l(β̂) − ½ λ β̂ᵀSβ̂ + ½ rank(S) log λ − ½ log|XᵀWX + λS|
        # Larger is better. The first two terms reward fitting the data; the
        # last two charge for the flexibility the curve uses. Constants that
        # don't depend on λ are dropped.
        rank_S = int(np.linalg.matrix_rank(S)) if np.any(S) else 0
        self.reml = []
        for lam in self.lambdas:
            b, H = self._pirls(X, y, S, lam)
            eta = X @ b
            ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
            sign, logdet = np.linalg.slogdet(H)
            crit = ll - 0.5 * lam * float(b @ S @ b) + 0.5 * rank_S * np.log(lam) - 0.5 * logdet
            self.reml.append((lam, float(crit) if sign > 0 and np.isfinite(crit) else -np.inf))
        self.lam = max(self.reml, key=lambda t: t[1])[0]

        # --- final fit on all training rows ----------------------------------
        self.beta, H = self._pirls(X, y, S, self.lam)
        self.V = np.linalg.pinv(H)                  # Bayesian covariance (mgcv's Vp)
        mu = expit(X @ self.beta)
        w = mu * (1 - mu)
        F = self.V @ (X.T @ (X * w[:, None]))       # influence matrix
        self.edf = {s.name: float(np.trace(F[sl, sl])) for s, sl in self.spans}
        self.edf_total = float(np.trace(F))

        # --- say so if the smoothing choice hit the end of the grid ------------
        if self.lam == self.lambdas[-1]:
            if max(self.edf.values(), default=1.0) < 1.1:
                self.note = ("every curve came out a straight line — λ is at the top of "
                             "the grid, but a bigger λ can't make a line straighter, so this "
                             "is a finding (the data showed no bend), not a problem")
            else:
                self.note = (f"chosen λ={self.lam:g} is the largest in the grid; "
                             "widen `lambdas` upward")
        elif self.lam == self.lambdas[0]:
            self.note = (f"chosen λ={self.lam:g} is the smallest in the grid — the curves "
                         "may be under-smoothed; widen `lambdas` downward")
        return self

    def predict(self, df):
        X, _, _ = self._build(df)
        return expit(X @ self.beta)

    def predict_band(self, df):
        """Probability with a pointwise 95% band, from the Bayesian covariance."""
        X, _, _ = self._build(df)
        eta = X @ self.beta
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", X, self.V, X), 0, None))
        return expit(eta), expit(eta - 1.96 * se), expit(eta + 1.96 * se)

    def curve(self, name, xgrid):
        """The smooth f(x) on the logit scale, with a 95% band."""
        for s, sl in self.spans:
            if s.name == name:
                B = s.basis_at(xgrid)
                f = B @ self.beta[sl]
                se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", B, self.V[sl, sl], B), 0, None))
                return f, f - 1.96 * se, f + 1.96 * se
        raise KeyError(name)

    def predict_on_grid(self, base_row, col, xgrid, set_cols=None):
        """
        Probability as `col` sweeps across `xgrid`, everything else held at
        `base_row` — the partial-dependence curve on the probability scale.
        Returns predictions and a 95% band from the full covariance.
        """
        g = pd.DataFrame([base_row] * len(xgrid))
        g[col] = xgrid
        for k, v in (set_cols or {}).items():
            g[k] = v
        X, _, _ = self._build(g)
        eta = X @ self.beta
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", X, self.V, X), 0, None))
        return expit(eta), expit(eta - 1.96 * se), expit(eta + 1.96 * se)
