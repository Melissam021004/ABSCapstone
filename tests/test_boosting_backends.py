#!/usr/bin/env python3
"""
test_boosting_backends.py — do CatBoost, LightGBM and HistGradientBoosting
work on THIS machine, through src/abs_boosting.py, the way Week 8 uses them?

Run this before the Week 8 script. It takes ~30 seconds and catches
installation problems and API mismatches on a tiny problem with a known
answer, instead of an hour into a tuning run.

For every backend that's installed it checks:
  1. it learns a planted signal (AUC well above chance)
  2. early stopping chooses fewer trees than the maximum
  3. the monotone constraint holds: more distance against the call never
     lowers the predicted overturn chance
  4. SHAP values add up: base value + sum of a row's SHAP = its log-odds
     (CatBoost and LightGBM only; HistGradientBoosting has no SHAP)
  5. save -> load gives identical predictions
  6. a category never seen in training doesn't crash prediction

A backend that isn't installed is SKIPPED with the install command.

    python tests/test_boosting_backends.py
"""

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.abs_boosting import BACKENDS, INSTALL_HINT, Booster, available  # noqa: E402

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(5)
failures, skipped = [], []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


n = 3000
df = pd.DataFrame({
    "dist_against_call": RNG.normal(0, 2, n),
    "noise": RNG.normal(0, 1, n),
    "team": RNG.choice(list("ABCDEFGH"), n),
    "game_order": np.arange(n),
})
eta = 1.5 * df.dist_against_call + np.where(df.team.isin(["A", "B"]), 0.8, 0.0)
y = (RNG.random(n) < expit(eta)).astype(int)
es = df.game_order >= int(0.85 * n)                 # last 15% held back for early stopping

PARAMS = {"catboost": {"depth": 4, "learning_rate": 0.1},
          "lightgbm": {"num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20},
          "hgb": {"max_leaf_nodes": 15, "learning_rate": 0.1}}

for backend in BACKENDS:
    ok, ver = available(backend)
    print(f"\n{backend.upper()} — {'version ' + ver if ok else 'NOT INSTALLED'}")
    if not ok:
        print(f"  SKIP  install with:  {INSTALL_HINT[backend]}")
        skipped.append(backend)
        continue
    try:
        m = Booster(backend, numeric=["dist_against_call", "noise"], categorical=["team"],
                    params=PARAMS[backend], max_rounds=2000, patience=50, n_jobs=2).fit(df, y, es)
        p = m.predict(df)
        auc = roc_auc_score(y, p)
        check("learns the planted signal", auc > 0.8 and 0 < p.min() and p.max() < 1,
              f"AUC {auc:.3f}")
        check("early stopping chose fewer trees than the maximum",
              1 <= m.best_iteration < 2000, f"{m.best_iteration} trees")

        grid = pd.DataFrame({"dist_against_call": np.linspace(-6, 6, 61), "noise": 0.0,
                             "team": "C", "game_order": 0})
        pg = m.predict(grid)
        check("monotone: further on the wrong side never lowers the chance",
              bool(np.all(np.diff(pg) >= -1e-9)), f"{pg[0]:.3f} -> {pg[-1]:.3f}")

        sh = m.shap(df.head(200))
        if backend == "hgb":
            check("no native SHAP (expected) -> returns None", sh is None)
        else:
            vals, base = sh
            recon = base + vals.sum(axis=1).to_numpy()
            gap = float(np.max(np.abs(recon - m.log_odds(df.head(200)))))
            check("SHAP adds up to the log-odds", gap < 1e-3, f"max gap {gap:.1e}")

        with tempfile.TemporaryDirectory() as tmp:
            m.save(tmp)
            m2 = Booster.load(tmp)
            gap = float(np.max(np.abs(m2.predict(df) - p)))
            check("save -> load gives identical predictions", gap < 1e-9, f"max gap {gap:.1e}")
            if backend != "hgb":
                sh2 = m2.shap(df.head(50))
                check("...and the loaded model still gives SHAP values", sh2 is not None)

        new = df.head(5).assign(team="NEVER_SEEN")
        pn = m.predict(new)
        check("an unseen category doesn't crash prediction", bool(np.all(np.isfinite(pn))))
    except Exception as e:                              # report, don't hide
        import traceback
        traceback.print_exc()
        check(f"{backend} runs end to end", False, f"{type(e).__name__}: {e}")

print()
if "catboost" in skipped or "lightgbm" in skipped:
    print("NOTE: Week 8 needs CatBoost and LightGBM for the plan's comparison. Without them it "
          "falls back to scikit-learn's HistGradientBoosting and says so in the report.")
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL INSTALLED BOOSTING BACKENDS PASSED"
      + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
