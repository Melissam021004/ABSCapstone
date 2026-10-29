"""
abs_boosting.py — the Week 8 gradient-boosted overturn models, behind one interface.

Three backends, one way of using them:

    catboost   CatBoostClassifier. Categorical columns (umpire, catcher, pitcher,
               batter, pitch type...) are passed NATIVELY — CatBoost encodes them
               with ordered target statistics, which is why the plan picked it.
    lightgbm   LGBMClassifier. Categoricals as pandas 'category' columns, split
               natively by LightGBM.
    hgb        scikit-learn's HistGradientBoostingClassifier — the same family of
               algorithm, always installed. Used automatically when CatBoost or
               LightGBM isn't, and by the test suite.

    m = Booster("catboost", numeric=[...], categorical=[...], params={...})
    m.fit(train_df, y, es_mask)       # early-stops on the es_mask rows, then
                                      # refits on ALL training rows
    m.predict(df)                     # probabilities
    m.shap(df)                        # (SHAP values, base value) or None
    m.save(folder) ; Booster.load(folder)

HOW THE NUMBER OF TREES IS CHOSEN
    Boosting adds trees until told to stop; too many and it memorises the
    training months. Each fit (1) trains on the training window minus its last
    ~15% of games, (2) watches log loss on those held-back games and remembers
    the best tree count, then (3) refits on the WHOLE training window with that
    many trees. August is never used to fit anything — only to compare settings.

ONE THING WE KNOW FOR SURE
    `dist_against_call` gets a monotone constraint: a pitch further on the wrong
    side of the call can never be LESS likely to be overturned. All three
    libraries can enforce this, and it keeps p(s) physically sensible for the MDP.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKENDS = ("catboost", "lightgbm", "hgb")
MONOTONE_UP = ("dist_against_call",)
HGB_MAX_LEVELS = 250          # HistGradientBoosting allows at most 255 bins per feature


def available(backend):
    """(installed?, version or the reason it isn't)."""
    try:
        if backend == "catboost":
            import catboost
            return True, catboost.__version__
        if backend == "lightgbm":
            import lightgbm
            return True, lightgbm.__version__
        if backend == "hgb":
            import sklearn
            return True, sklearn.__version__
    except Exception as e:                      # ImportError, or a broken install
        return False, f"{type(e).__name__}: {e}"
    return False, "unknown backend"


INSTALL_HINT = {
    "catboost": "conda install -c conda-forge catboost      (or: pip install catboost)",
    "lightgbm": "conda install -c conda-forge lightgbm      (or: pip install lightgbm; "
                "on a Mac, pip may also need: brew install libomp)",
    "hgb": "pip install scikit-learn",
}


# ----------------------------------------------------------- search spaces
SEARCH_SPACE = {
    "catboost": {
        "depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "l2_leaf_reg": [1, 3, 5, 10, 20, 40],
        "random_strength": [0.5, 1, 2, 5],
        "bagging_temperature": [0, 0.5, 1, 2],
    },
    "lightgbm": {
        "num_leaves": [4, 7, 15, 31, 63],
        "learning_rate": [0.02, 0.03, 0.05, 0.1],
        "min_child_samples": [10, 20, 40, 80, 160],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
        "reg_lambda": [0, 1, 5, 20, 50],
        "cat_smooth": [5, 10, 20, 50],
    },
    "hgb": {
        "max_leaf_nodes": [4, 7, 15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_samples_leaf": [10, 20, 40, 80, 160],
        "l2_regularization": [0, 1, 5, 20, 50],
        "max_features": [0.6, 0.8, 1.0],
    },
}


def sample_configs(backend, n, seed=2026):
    """n distinct random settings from the backend's search space (seeded, so the
    same n gives the same list every time — the search is reproducible)."""
    rng = np.random.default_rng(seed)
    space = SEARCH_SPACE[backend]
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = json.dumps(cfg, sort_keys=True, default=float)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def _cat_strings(s):
    """
    Category values as clean strings, so 123, 123.0 and "123" are the same level.
    (An ID column read with a missing value becomes float — 123.0 — and would
    otherwise never match the 123 that predict(state) is given.)
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(str)
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        out = v.astype(object)
        whole = v.notna() & (v % 1 == 0)
        out[whole] = v[whole].astype("int64").astype(str)
        out[v.notna() & ~whole] = v[v.notna() & ~whole].astype(str)
        return out.where(v.notna(), "missing").astype(str)

    def key(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)) or x is pd.NA:
            return "missing"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return str(int(x)) if float(x).is_integer() else str(x)
        return str(x)
    return s.map(key).astype(str)


def _log_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ----------------------------------------------------------------- model
class Booster:
    """One gradient-boosted classifier; see the module docstring."""

    def __init__(self, backend, numeric, categorical, params=None, seed=2026,
                 max_rounds=3000, patience=100, monotone=True, n_jobs=4):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.params = dict(params or {})
        self.seed, self.max_rounds, self.patience = seed, max_rounds, patience
        self.monotone, self.n_jobs = monotone, n_jobs
        self.levels = {}
        self.monotone_note = ""
        self.best_iteration = None
        self.es_logloss = None
        self.fit_seconds = None
        self.model = None

    # ------------------------------------------------------------ data
    @property
    def features(self):
        return self.numeric + self.categorical

    def _learn_levels(self, df):
        for c in self.categorical:
            vc = _cat_strings(df[c]).value_counts()
            levels = list(vc.index)
            if self.backend == "hgb" and len(levels) > HGB_MAX_LEVELS:
                levels = levels[:HGB_MAX_LEVELS]          # rarest folded into "missing"
            self.levels[c] = [str(v) for v in levels]

    def _prep(self, df):
        X = pd.DataFrame(index=df.index)
        for c in self.numeric:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df.columns else np.nan
        for c in self.categorical:
            v = (_cat_strings(df[c]) if c in df.columns
                 else pd.Series("missing", index=df.index, dtype=object).astype(str))
            if self.backend == "catboost":
                X[c] = v.astype(object)                    # CatBoost wants plain str objects
            else:
                known = v.where(v.isin(self.levels[c]))    # unseen -> missing
                X[c] = pd.Categorical(known, categories=self.levels[c])
        return X[self.features]

    def _monotone_vector(self):
        return [1 if (self.monotone and f in MONOTONE_UP) else 0 for f in self.features]

    # ------------------------------------------------------------ fit
    def fit(self, df, y, es_mask=None, n_trees=None):
        """
        Early-stop on the rows where `es_mask` is True (the last ~15% of training
        games), then refit on every row with the best number of trees.

        Or pass n_trees to skip the early-stopping step and fit exactly that many
        trees on every row — Week 9 uses this to refit a locked model on folds of
        the training window with the SAME settings and tree count.
        """
        t0 = time.time()
        y = np.asarray(y, int)
        self._learn_levels(df)
        X = self._prep(df)
        fit_fn = {"catboost": self._fit_catboost, "lightgbm": self._fit_lightgbm,
                  "hgb": self._fit_hgb}[self.backend]
        if n_trees is not None:
            self.best_iteration, self.es_logloss = int(n_trees), None
            fit_fn(X, y, None, None, probe=False)
            self.fit_seconds = time.time() - t0
            return self
        es_mask = np.asarray(es_mask, bool)
        if es_mask.sum() < 30 or (~es_mask).sum() < 100:
            raise ValueError("early-stopping split too small — need >= 30 held-back rows")
        Xa, ya, Xb, yb = X[~es_mask], y[~es_mask], X[es_mask], y[es_mask]
        try:
            self.best_iteration, self.es_logloss = fit_fn(Xa, ya, Xb, yb, probe=True)
        except Exception as e:
            # A library version that rejects the monotone constraint shouldn't sink a
            # long tuning run: drop the constraint, say so loudly, keep going.
            if not (self.monotone and "monoton" in str(e).lower()):
                raise
            self.monotone = False
            self.monotone_note = f"monotone constraint dropped: {type(e).__name__}: {e}"
            print(f"    WARNING: {self.backend} {self.monotone_note}")
            self.best_iteration, self.es_logloss = fit_fn(Xa, ya, Xb, yb, probe=True)
        fit_fn(X, y, None, None, probe=False)
        self.fit_seconds = time.time() - t0
        return self

    def _fit_catboost(self, Xa, ya, Xb, yb, probe):
        from catboost import CatBoostClassifier, Pool
        # boosting_type="Plain": monotone constraints don't work with CatBoost's
        # "Ordered" scheme, which it would otherwise pick on a dataset this small.
        # Constraints by NAME ({"dist_against_call": 1}) rather than one entry per
        # column, so CatBoost never sees a constraint slot for a categorical column.
        mono = {f: 1 for f in self.numeric if self.monotone and f in MONOTONE_UP}
        base = dict(loss_function="Logloss", eval_metric="Logloss", random_seed=self.seed,
                    verbose=False, allow_writing_files=False, thread_count=self.n_jobs,
                    boosting_type="Plain", **self.params)
        if mono:
            base["monotone_constraints"] = mono
        if "bagging_temperature" in self.params:
            base["bootstrap_type"] = "Bayesian"      # the only bootstrap that setting applies to
        cats = self.categorical
        if probe:
            m = CatBoostClassifier(iterations=self.max_rounds, od_type="Iter",
                                   od_wait=self.patience, use_best_model=True, **base)
            m.fit(Pool(Xa, ya, cat_features=cats), eval_set=Pool(Xb, yb, cat_features=cats))
            best = int(m.get_best_iteration()) + 1
            p = m.predict_proba(Pool(Xb, cat_features=cats))[:, 1]
            return best, _log_loss(yb, p)
        self.model = CatBoostClassifier(iterations=max(self.best_iteration, 1), **base)
        self.model.fit(Pool(Xa, ya, cat_features=cats))
        return None

    def _fit_lightgbm(self, Xa, ya, Xb, yb, probe):
        import lightgbm as lgb
        params = dict(self.params)
        if params.get("subsample", 1.0) < 1.0:
            params.setdefault("subsample_freq", 1)    # subsample is ignored without it
        base = dict(objective="binary", random_state=self.seed, n_jobs=self.n_jobs,
                    verbose=-1, deterministic=True, force_row_wise=True,
                    monotone_constraints=self._monotone_vector(), **params)
        if probe:
            m = lgb.LGBMClassifier(n_estimators=self.max_rounds, **base)
            m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(self.patience, verbose=False)])
            best = int(m.best_iteration_ or self.max_rounds)
            p = m.predict_proba(Xb, num_iteration=best)[:, 1]
            return best, _log_loss(yb, p)
        self.model = lgb.LGBMClassifier(n_estimators=max(self.best_iteration, 1), **base)
        self.model.fit(Xa, ya)
        return None

    def _fit_hgb(self, Xa, ya, Xb, yb, probe):
        from sklearn.ensemble import HistGradientBoostingClassifier
        cst = {f: 1 for f in self.features if self.monotone and f in MONOTONE_UP}
        base = dict(categorical_features="from_dtype", early_stopping=False,
                    random_state=self.seed, monotonic_cst=cst or None, **self.params)
        if probe:
            cap = min(self.max_rounds, 1500)
            m = HistGradientBoostingClassifier(max_iter=cap, **base).fit(Xa, ya)
            losses = []
            for i, pr in enumerate(m.staged_predict_proba(Xb)):
                losses.append(_log_loss(yb, pr[:, 1]))
                if i - int(np.argmin(losses)) > self.patience:    # same patience rule
                    break
            best = int(np.argmin(losses)) + 1
            return best, float(losses[best - 1])
        self.model = HistGradientBoostingClassifier(max_iter=max(self.best_iteration, 1), **base)
        self.model.fit(Xa, ya)
        return None

    # ------------------------------------------------------------ use
    def predict(self, df):
        X = self._prep(df)
        if self.backend == "catboost":
            from catboost import Pool
            p = self.model.predict_proba(Pool(X, cat_features=self.categorical))[:, 1]
        elif self.backend == "lightgbm" and not hasattr(self.model, "predict_proba"):
            p = self.model.predict(X)                     # a loaded lgb.Booster
        else:
            p = self.model.predict_proba(X)[:, 1]
        return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)

    def shap(self, df):
        """
        SHAP values on the log-odds scale: one column per feature plus a base
        value, such that base + sum(row) = the model's log-odds for that row.
        CatBoost and LightGBM compute these natively (no `shap` package needed);
        HistGradientBoosting can't, so this returns None and the Week 8 script
        falls back to permutation importance.
        """
        X = self._prep(df)
        if self.backend == "catboost":
            from catboost import Pool
            v = np.asarray(self.model.get_feature_importance(
                Pool(X, cat_features=self.categorical), type="ShapValues"))
        elif self.backend == "lightgbm":
            booster = getattr(self.model, "booster_", self.model)
            v = np.asarray(booster.predict(X, pred_contrib=True))
        else:
            return None
        return pd.DataFrame(v[:, :-1], columns=self.features, index=df.index), float(v[0, -1])

    def log_odds(self, df):
        p = self.predict(df)
        return np.log(p / (1 - p))

    # ------------------------------------------------------------ save / load
    def meta(self):
        return {"backend": self.backend, "numeric": self.numeric,
                "categorical": self.categorical, "params": self.params,
                "seed": self.seed, "max_rounds": self.max_rounds, "patience": self.patience,
                "monotone": self.monotone, "monotone_note": self.monotone_note,
                "levels": self.levels,
                "best_iteration": self.best_iteration, "es_logloss": self.es_logloss}

    def save(self, folder):
        """Native model file + meta.json (feature lists and category levels)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.backend == "catboost":
            self.model.save_model(str(folder / "model.cbm"))
        elif self.backend == "lightgbm":
            booster = getattr(self.model, "booster_", self.model)
            booster.save_model(str(folder / "model.txt"))
        else:
            import pickle
            with open(folder / "model.pkl", "wb") as fh:
                pickle.dump(self.model, fh)
        (folder / "meta.json").write_text(json.dumps(self.meta(), indent=2, default=float))
        return folder

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        m = json.loads((folder / "meta.json").read_text())
        b = cls(m["backend"], m["numeric"], m["categorical"], m["params"], m["seed"],
                m["max_rounds"], m["patience"], m["monotone"])
        b.levels, b.best_iteration, b.es_logloss = m["levels"], m["best_iteration"], m["es_logloss"]
        b.monotone_note = m.get("monotone_note", "")
        if b.backend == "catboost":
            from catboost import CatBoostClassifier
            b.model = CatBoostClassifier()
            b.model.load_model(str(folder / "model.cbm"))
        elif b.backend == "lightgbm":
            import lightgbm as lgb
            b.model = lgb.Booster(model_file=str(folder / "model.txt"))
        else:
            import pickle
            with open(folder / "model.pkl", "rb") as fh:
                b.model = pickle.load(fh)
        return b
