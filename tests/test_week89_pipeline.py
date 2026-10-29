#!/usr/bin/env python3
"""
test_week89_pipeline.py — do Weeks 8 and 9 find what we planted, and keep their promises?

Builds the same fake season as tests/test_week7_pipeline.py, plants two more
things, then runs Weeks 7, 8 and 9 on it end to end:

  PLANTED
    A. A LEAK. On challenged pitches the `batter` ID secretly encodes the
       result (IDs 1–50 were overturned, 51–100 stood) — the kind of thing a
       bad join does. Week 8's leak check must catch it.
    B. DRIFT. In August and September the odds of an overturn are e^−1.2 ≈ 0.3×
       what the same pitch got in March–July (say teams start challenging more
       marginal pitches). Models trained on March–July over-predict; Week 9's
       calibration on August must notice and correct it for September.

  PROMISED
    - Week 8 refits the Week 7 ladder EXACTLY (same rows, same predictions)
    - nothing in Weeks 7–8 touches September
    - the tuning search is reproducible (same seed, same settings)
    - Week 9 locks a bundle that loads, verifies and answers predict(state)
    - the test window opens once: a second --final-test and a re-lock refuse
    - the model card and the ≥5 subgroup tables exist

    python tests/test_week89_pipeline.py          (2–4 minutes)

It works in a temporary folder and never touches your data/, docs/ or models/.
Boosting runs on scikit-learn's HistGradientBoosting here so the test is the
same on every machine; tests/test_boosting_backends.py covers CatBoost and
LightGBM themselves.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.abs_features import nearest_edge                                # noqa: E402

spec = importlib.util.spec_from_file_location("w7test", ROOT / "tests" / "test_week7_pipeline.py")
W7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W7)

DRIFT = -1.2
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def plant(season):
    rng = np.random.default_rng(7)
    df = season.copy()
    ch = df["was_challenged"].to_numpy(bool)
    late = df["split"].isin(["validate", "test"]).to_numpy()
    # B. drift: redraw late outcomes from the same curve, shifted in log-odds
    # (the three planted mislinks, a foot from the edge, are left as they are)
    strike = df["called_strike"].to_numpy(bool)
    z = df["zone_dist_in"].to_numpy(float)
    d = np.where(strike, z, -z)                                  # inches against the call
    edge = nearest_edge(df["plate_x"], df["plate_z"], df["abs_zone_top"], df["abs_zone_bot"])
    shift = np.where(edge == "top", W7.TOP_SHIFT_IN, 0.0)        # Week 7's planted zone shift
    d_eff = np.where(strike, d - shift, d + shift)
    p = 0.06 + 0.88 * norm.cdf(d_eff / 0.6)                      # the Week 7 fixture's curve
    p = np.where(df["challenger_role"].astype(str).eq("pitcher"),
                 expit(logit(p) + np.log(0.5)), p)
    redraw = ch & late & (np.abs(z) < 11.9)
    p2 = expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + DRIFT)
    df.loc[redraw, "is_overturned"] = rng.random(redraw.sum()) < p2[redraw]
    # A. the leak: batter ID encodes the challenge result
    ov = df["is_overturned"].fillna(False).astype(bool).to_numpy()
    df["batter"] = rng.integers(1, 101, len(df))
    df.loc[ch & ov, "batter"] = rng.integers(1, 51, int((ch & ov).sum()))
    df.loc[ch & ~ov, "batter"] = rng.integers(51, 101, int((ch & ~ov).sum()))
    df["pitcher"] = rng.integers(500, 620, len(df))
    return df


def run(script, *args, env, expect_ok=True):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], env=env,
                       capture_output=True, text=True)
    if expect_ok and r.returncode:
        print(r.stdout[-4000:], r.stderr[-4000:])
    return r


def main():
    tmp = Path(tempfile.mkdtemp(prefix="w89_e2e_"))
    try:
        data = tmp / "data_demo_e2e"
        (data / "processed").mkdir(parents=True)
        print("\nBuilding a fake season with a planted leak and planted drift ...")
        season, _ = W7.build_season()
        season = plant(season)
        season.to_csv(data / "processed" / "cohort_2026.csv", index=False)
        n_ch = int(season["was_challenged"].sum())
        print(f"  {len(season):,} called pitches, {n_ch:,} challenges")
        env = dict(os.environ, ABS_DATA_DIR=str(data), ABS_DOCS_DIR=str(tmp / "docs"),
                   ABS_CONFIG_DIR=str(tmp / "config"), ABS_FIG_DIR=str(tmp / "figs"),
                   ABS_MODEL_DIR=str(tmp / "models"))
        docs, models = tmp / "docs", tmp / "models"

        print("Running Week 7 ...")
        r = run("15_week7_baseline.py", "--boot", "200", "--no-figures", env=env)
        check("Week 7 runs", r.returncode == 0)
        print("Running Week 8 (HistGradientBoosting, 3 settings per feature set) ...")
        r = run("16_week8_boosting.py", "--trials", "3", "--boot", "200", "--backends", "hgb",
                "--no-figures", env=env)
        check("Week 8 runs", r.returncode == 0)
        if r.returncode:
            return
        sel = json.loads((models / "week8" / "selection.json").read_text())
        p7 = pd.read_csv(data / "processed" / "week7_validate_predictions.csv")
        p8 = pd.read_csv(data / "processed" / "week8_validate_predictions.csv")

        print("\nWeek 8 — same comparison as Week 7")
        same_rows = (p7[["game_pk", "at_bat_number", "pitch_number"]].to_numpy() ==
                     p8[["game_pk", "at_bat_number", "pitch_number"]].to_numpy()).all()
        check("identical August rows in Weeks 7 and 8", bool(same_rows), f"{len(p8)} rows")
        gap = max(float(np.max(np.abs(p7[f"p_{k}"] - p8[f"p_{k}"])))
                  for k in ("M1", "M2", "M5", "GAM-loc"))
        check("the refit Week 7 ladder gives identical predictions", gap < 1e-9,
              f"max gap {gap:.1e}")
        check("Week 8 never scored September",
              pd.to_datetime(p8["game_date"]).dt.month.eq(8).all())
        log = pd.read_csv(docs / "week8_tuning_log.csv")
        check("every setting tried is logged", len(log) == 6, f"{len(log)} rows")
        from src.abs_boosting import sample_configs
        check("the search is reproducible (same seed -> same settings)",
              sample_configs("hgb", 5, 2026) == sample_configs("hgb", 5, 2026))

        print("\nWeek 8 — planted leak: batter ID encodes the result")
        lk = sel["leak_check"]
        check("the importance leak check flags it", lk["flagged"], lk["detail"][:90])
        check("the AUC-gap warning fires too", lk["auc_gap_warning"],
              f"everything − location = {lk['auc_gap_full_minus_location']:+.3f}")
        report = (docs / "WEEK8_REPORT.md").read_text()
        check("the report says so", "SUSPICIOUS" in report and "more than 0.02 AUC" in report)
        loc = sel["candidate_location"]
        check("the location-only candidate can't use the leak",
              sel["models"][loc]["feature_set"] == "location", loc)

        print("\nRunning Week 9 (calibrate, subgroups, lock) ...")
        r = run("17_week9_calibrate_lock.py", "--boot", "200", "--no-figures", env=env)
        check("Week 9 locks a model", r.returncode == 0)
        if r.returncode:
            return
        man = json.loads((models / "production" / "manifest.json").read_text())
        cal = pd.read_csv(docs / "week9_calibration.csv")

        print("\nWeek 9 — planted drift: August over-predicted by the March–July model")
        method = man["calibration"]["method"]
        check("a calibrator is chosen (not 'none')", method != "none", method)
        row = cal[(cal["feature_set"] == "location") & (cal["method"] == "none")].iloc[0]
        check("uncalibrated, the model over-predicts August (intercept < 0)",
              row["cal_intercept"] < -0.3, f"intercept {row['cal_intercept']:+.2f}")
        sub = pd.read_csv(docs / "week9_subgroups.csv")
        check("subgroup tables cover at least 5 groups", sub["group"].nunique() >= 5,
              f"{sub['group'].nunique()} groups, {len(sub)} slices")
        mo = pd.read_csv(docs / "week9_months.csv")
        check("months include out-of-fold training months and August",
              mo["window"].str.startswith("train").any() and (mo["window"] == "August").any(),
              ", ".join(mo["level"].str[3:]))

        print("\nWeek 9 — the locked bundle")
        sys.path.insert(0, str(ROOT))
        from src.abs_predict import load_production
        pm = load_production(models / "production")
        p = pm.predict({"plate_x": 0.80, "plate_z": 2.5, "batter_height_ft": 6.1,
                        "called_strike": True})
        check("loads, verifies its hashes and answers predict(state)", 0 < p < 1, f"{p:.3f}")
        check("the self-check reproduced August exactly", man["selfcheck_max_gap"] < 1e-9,
              f"{man['selfcheck_max_gap']:.1e}")
        check("...and spotted the 3 planted mislinks (distance ≠ location)",
              man["selfcheck_geometry_mismatches"] == 3,
              f"{man['selfcheck_geometry_mismatches']} found")
        card = (docs / "MODEL_CARD.md").read_text()
        need = ["## Intended use", "should NOT be used", "## Training data", "## Performance",
                "by subgroup", "## Known limitations", "## Versioning"]
        check("the model card has every section the plan asks for",
              all(n in card for n in need), ", ".join(n for n in need if n not in card) or "all")
        check("September is still sealed in the card", "sealed" in card)

        print("\nRunning --final-test ...")
        r = run("17_week9_calibrate_lock.py", "--final-test", "--boot", "200", "--no-figures",
                env=env)
        check("the final test runs", r.returncode == 0)
        if r.returncode:
            return
        tc = pd.read_csv(docs / "week9_test_comparison.csv")
        b = tc[tc["metric"] == "brier"].set_index("model")["value"]
        i = tc[tc["metric"] == "cal_intercept"].set_index("model")["value"]
        check("on September the calibrated p(s) beats its uncalibrated self",
              b["PROD"] < b["PROD-raw"], f"Brier {b['PROD-raw']:.4f} -> {b['PROD']:.4f}")
        check("...and its intercept moves toward 0", abs(i["PROD"]) < abs(i["PROD-raw"]),
              f"{i['PROD-raw']:+.2f} -> {i['PROD']:+.2f}")
        mo = pd.read_csv(docs / "week9_months.csv")
        check("September joins the month table", (mo["window"] == "September").any())
        card = (docs / "MODEL_CARD.md").read_text()
        check("the model card now has the test row", "September (test, opened once)" in card)

        print("\nThe test window opens once")
        r = run("17_week9_calibrate_lock.py", "--final-test", "--no-figures", env=env,
                expect_ok=False)
        check("a second --final-test refuses", r.returncode != 0 and "exactly once" in r.stdout
              + r.stderr)
        r = run("17_week9_calibrate_lock.py", "--no-figures", env=env, expect_ok=False)
        check("re-locking after the test refuses", r.returncode != 0 and
              "relock-after-test" in r.stdout + r.stderr)
        st = json.loads((models / "production" / "TEST_OPENED.json").read_text())
        check("the opening is stamped", st["times_opened"] == 1, st["opened"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("ALL WEEK 8–9 PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
