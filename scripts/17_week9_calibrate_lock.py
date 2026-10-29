#!/usr/bin/env python3
"""
17_week9_calibrate_lock.py — Week 9. Calibrate, check every slice, lock p(s), test once.

The plan: "Turn the winning model into a trustworthy probability, then freeze
it. After this week, p(s) is a fixed input to everything else."

    python scripts/17_week9_calibrate_lock.py                     # steps 0–6: calibrate → lock
    python scripts/17_week9_calibrate_lock.py --final-test        # step 7: open September ONCE
    python scripts/17_week9_calibrate_lock.py --production full   # lock the "everything" model

    0. PRE-FLIGHT  same cohort as Week 8? which candidates did Week 8 send?
    1. CALIBRATE   The plan's three-way split: train = March–July, CALIBRATE =
                   August, test = September. For each Week 8 candidate, three
                   options — none, Platt scaling, isotonic regression — are scored
                   on August by 5-fold cross-fitting over games, so every
                   challenge is scored by a calibrator that never saw its game.
                   Rule fixed in advance: the simplest option whose Brier can't be
                   told apart from the best one's.
    2. SUBGROUPS   calibration by challenger, count, challenges left, month,
                   batter hand, pitcher hand, edge, inning and leverage (9 groups)
    3. MONTHS      is p(s) stable across the season? Training months scored
                   out-of-fold (5 refits of the same model), August cross-fitted
    4. DECIDE      the production model, with its reasons written down
    5. LOCK        models/production/: model + calibrator (JSON) + participant
                   rates + manifest.json with a SHA-256 hash of every file +
                   requirements-lock.txt; previous locks kept in
                   models/production_history/; docs/MODEL_CARD.md
    6. FIGURES     figures/week9/

    --final-test   (once, after locking) scores the locked p(s) AND every Week 8
                   model on September — one table, the same subgroups — fills in
                   the model card's test section and stamps
                   models/production/TEST_OPENED.json. A second run refuses
                   unless --force-retest, and that is recorded too.

Reads   data/processed/cohort_2026.parquet, config/cohort.json,
        models/week8/selection.json and the Week 8 models
Writes  docs/WEEK9_REPORT.md, docs/MODEL_CARD.md, docs/week9_*.csv,
        models/production/, figures/week9/*.png
        (+ docs/WEEK9_TEST_RESULTS.md, docs/week9_test_*.csv with --final-test)
"""

import argparse
import datetime as dt
import json
import os
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "eda"))

from src import abs_metrics as M                                       # noqa: E402
from src.abs_baselines import MODEL_SPECS, choose, fit_model            # noqa: E402
from src.abs_boosting import BACKENDS, Booster                          # noqa: E402
from src.abs_calibration import METHODS, Calibrator, cross_fit          # noqa: E402
from src.abs_features import describe_call_check                        # noqa: E402
from src.abs_pipeline import grouped_folds, load_challenges             # noqa: E402
from src.abs_zone import zone_distance_in                               # noqa: E402
from src.abs_predict import (load_production, required_fields,          # noqa: E402
                             states_from_cohort, write_bundle)

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
CONFIG = Path(os.environ.get("ABS_CONFIG_DIR", "config"))
MODEL_DIR = Path(os.environ.get("ABS_MODEL_DIR", "models"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "week9"
W8 = MODEL_DIR / "week8"
PROD = MODEL_DIR / "production"
HISTORY = MODEL_DIR / "production_history"
STAMP = PROD / "TEST_OPENED.json"

METHOD_LABEL = {"none": "uncalibrated", "platt": "Platt scaling", "isotonic": "isotonic"}
SET_LABEL = {"location": "location only", "full": "everything"}
MIN_FLAG_N = 30          # a slice needs this many challenges before it can be flagged


def banner(msg):
    print(f"\n{msg}\n{'-' * len(msg)}")


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if abs(x) < 0.5 * 10 ** -d:
        x = 0.0                                  # no "-0.000"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fci(v, ci, d=3, sign=False):
    lo, hi = ci
    if lo is None or not np.isfinite(lo):
        return f(v, d, sign)
    return f"{f(v, d, sign)} [{f(lo, d, sign)}, {f(hi, d, sign)}]"


# =========================================================== Week 8 models
def load_week8():
    sel_path = W8 / "selection.json"
    if not sel_path.exists():
        raise SystemExit(f"  {sel_path} not found — run scripts/16_week8_boosting.py first.")
    sel = json.loads(sel_path.read_text())
    models = {}
    for k, m in sel["models"].items():
        if m["family"] in BACKENDS:
            models[k] = Booster.load(W8 / k)
        else:
            with open(W8 / f"{k}.pkl", "rb") as fh:
                models[k] = pickle.load(fh)
    return sel, models


def model_features(model):
    """The columns a fitted model reads (what predict(state) must be able to build)."""
    if isinstance(model, Booster):
        return list(model.features)
    d = model.design
    feats = d.numeric + d.categorical + list(d.slopes_by) + list(d.slopes_by.values())
    if hasattr(model, "smooth_cols"):
        feats += list(model.smooth_cols) + list(getattr(model, "by", {}).values())
    return list(dict.fromkeys(feats))


def refit_like(key, model, df):
    """The same model, same settings, fitted on different rows (for out-of-fold scores)."""
    if isinstance(model, Booster):
        b = Booster(model.backend, model.numeric, model.categorical, model.params,
                    seed=model.seed, max_rounds=model.max_rounds, patience=model.patience,
                    monotone=model.monotone)
        return b.fit(df, df["y"], n_trees=model.best_iteration)
    return fit_model(key, df)


# =========================================================== subgroups
def slices(df):
    """
    The subgroup each challenge belongs to, for 9 groupings. The plan asks for
    at least 5 (role, count, challenges remaining, month, handedness); edge,
    inning and leverage are added because the MDP's decisions differ there too.
    """
    out = {}
    role = df["challenger_role"].astype(str).str.lower()
    out["Challenger"] = role.where(role.isin(["batter", "catcher", "pitcher"]), "other")
    b = pd.to_numeric(df.get("balls"), errors="coerce")
    s = pd.to_numeric(df.get("strikes"), errors="coerce")
    out["Count"] = pd.Series(np.select(
        [(b == 3) & (s == 2), s == 2, b == 3], ["full count (3-2)", "two strikes (not full)",
                                                "three balls (not full)"], "other counts"),
        index=df.index)
    cl = pd.to_numeric(df.get("chal_left"), errors="coerce")
    out["Challenges left"] = pd.Series(np.select(
        [cl <= 0, cl == 1, cl == 2, cl >= 3], ["0 left", "1 left", "2 left", "3+ left"],
        "unknown"), index=df.index)
    out["Month"] = pd.to_datetime(df["game_date"]).dt.strftime("%m %b")
    out["Batter hand"] = "bats " + df["stand"].astype(str)
    out["Pitcher hand"] = "throws " + df["p_throws"].astype(str)
    out["Edge"] = df["edge"].astype(str) + " edge"
    inn = pd.to_numeric(df.get("inning"), errors="coerce")
    out["Inning"] = pd.Series(np.select([inn <= 3, inn <= 6, inn <= 9], ["1–3", "4–6", "7–9"],
                                        "extras"), index=df.index)
    li = pd.to_numeric(df.get("leverage_index"), errors="coerce") if "leverage_index" in df \
        else pd.Series(np.nan, index=df.index)
    out["Leverage"] = pd.Series(np.select([li < 0.85, li < 2.0, li >= 2.0],
                                          ["low (< 0.85)", "medium", "high (≥ 2)"], "unknown"),
                                index=df.index)
    return out


LEVEL_ORDER = ["batter", "catcher", "pitcher", "other",
               "other counts", "two strikes (not full)", "three balls (not full)",
               "full count (3-2)", "0 left", "1 left", "2 left", "3+ left",
               "side edge", "top edge", "bottom edge", "1–3", "4–6", "7–9", "extras",
               "low (< 0.85)", "medium", "high (≥ 2)"]


def _level_key(v):
    v = str(v)
    return (LEVEL_ORDER.index(v), v) if v in LEVEL_ORDER else (len(LEVEL_ORDER), v)


def subgroup_table(df, p, window):
    """Observed vs predicted overturn rate in every slice, with a 95% Wilson interval."""
    y = df["y"].to_numpy(int)
    p = np.asarray(p, float)
    rows = []
    for group, lv in slices(df).items():
        lv = lv.to_numpy()
        for level in sorted(pd.unique(lv), key=_level_key):
            m = lv == level
            n = int(m.sum())
            if n == 0 or str(level) in ("unknown", "bats nan", "throws nan", "nan edge"):
                continue
            k = int(y[m].sum())
            lo, hi = M.wilson(k, n)
            pred = float(p[m].mean())
            off = n >= MIN_FLAG_N and (pred < lo or pred > hi)
            rows.append({"window": window, "group": group, "level": str(level), "n": n,
                         "pred": pred, "obs": k / n, "lo": lo, "hi": hi,
                         "gap_pts": 100 * (k / n - pred), "flag": bool(off)})
    return pd.DataFrame(rows)


# =========================================================== main (lock)
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--production", choices=["location", "full"], default="location",
                    help="which Week 8 candidate becomes p(s) (default: location, the Week 7 "
                         "decision)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--final-test", action="store_true",
                    help="open the September test window ONCE and score the locked model")
    ap.add_argument("--force-retest", action="store_true",
                    help="re-open the test window anyway (recorded in TEST_OPENED.json)")
    ap.add_argument("--relock-after-test", action="store_true",
                    help="re-lock even though September was already opened (recorded; the "
                         "test results stop being an unbiased estimate)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    if args.final_test:
        return final_test(args)

    # ---------------------------------------------------------- 0. pre-flight
    banner("0. Pre-flight")
    if STAMP.exists() and not args.relock_after_test:
        st = json.loads(STAMP.read_text())
        raise SystemExit(
            f"  September was already opened on {st['opened'][:10]} for p(s) {st['version']}.\n"
            "  Re-locking now would let the test result influence the model. If you really "
            "must,\n  pass --relock-after-test — it is written into the new manifest and the "
            "model card.")
    data = load_challenges(DATA_DIR, CONFIG)
    tr, va = data["tr"], data["va"]
    sel, models = load_week8()
    for line in describe_call_check(data["call_check"]):
        print(f"  {line}")
    print(f"  cohort fingerprint {data['fingerprint']}"
          + (f"  ({data['fingerprint_note']})" if data["fingerprint_note"] else ""))
    if sel.get("cohort_fingerprint") != data["fingerprint"]:
        why = ("Week 8 ran before the cohort fingerprint was recorded (it says 'unknown')"
               if sel.get("cohort_fingerprint") in (None, "unknown") else
               f"The cohort changed since Week 8 (Week 8 used {sel.get('cohort_fingerprint')}, "
               f"now {data['fingerprint']})")
        raise SystemExit(f"  {why}.\n  Re-run scripts/16_week8_boosting.py (or "
                         "scripts/run_week89.py) so both weeks are checked against the same "
                         "cohort.")
    cands = {"location": sel["candidate_location"], "full": sel["candidate_full"]}
    labels = {k: m["label"] for k, m in sel["models"].items()}
    family = {k: m["family"] for k, m in sel["models"].items()}
    print(f"  challenges: train {len(tr):,} · August {len(va):,} (the calibration set) · "
          f"September {data['n_test']:,} (sealed)")
    for fs, k in cands.items():
        print(f"  Week 8 candidate ({SET_LABEL[fs]}): {labels[k]}")
    y = va["y"].to_numpy(int)
    games = va["game_pk"].to_numpy()
    p_ref = float(tr["y"].mean())

    # ---------------------------------------------------------- 1. calibrate
    banner(f"1. Calibrate on August ({args.folds}-fold cross-fitting over games)")
    raw = {fs: np.clip(models[k].predict(va), 1e-6, 1 - 1e-6) for fs, k in cands.items()}
    xfit, point, mid = {}, {}, {}
    for fs in cands:
        for m_ in METHODS:
            key = f"{fs}|{m_}"
            xfit[key] = cross_fit(raw[fs], y, games, m_, k=args.folds, seed=args.seed)
            point[key] = M.metric_suite(y, xfit[key], p_ref)
            point[key]["ece_floor"] = M.ece_noise_floor(xfit[key], n_sim=200, seed=args.seed)
            mid[key] = M.midrange_check(y, xfit[key])
    print(f"  bootstrapping {args.boot} resamples of {len(np.unique(games))} August games ...")
    ci, diffs = M.cluster_bootstrap(y, xfit, games, p_ref, B=args.boot, seed=args.seed)
    chosen, rules, final_cal = {}, {}, {}
    for fs in cands:
        keys = [f"{fs}|{m_}" for m_ in METHODS]
        best, pick, rule = choose(keys, point, diffs)
        chosen[fs], rules[fs] = pick.split("|")[1], rule
        final_cal[fs] = Calibrator(chosen[fs]).fit(raw[fs], y)
        for k_ in keys:
            s = point[k_]
            print(f"  {labels[cands[fs]]:<24} {METHOD_LABEL[k_.split('|')[1]]:<14} "
                  f"Brier {s['brier']:.4f}  log loss {s['log_loss']:.4f}  ECE {s['ece']:.3f}  "
                  f"slope {f(s['cal_slope'], 2)}" + ("   <- chosen" if k_ == pick else ""))
        print(f"    final calibrator: {final_cal[fs].describe()}")

    # ---------------------------------------------------------- 2–3. subgroups, months
    prod_fs = args.production
    pkey = cands[prod_fs]
    pmodel = models[pkey]
    pcal = final_cal[prod_fs]
    p_aug = xfit[f"{prod_fs}|{chosen[prod_fs]}"]
    banner(f"2. Subgroups — {labels[pkey]} + {METHOD_LABEL[chosen[prod_fs]]} (August, "
           "cross-fitted)")
    sub = subgroup_table(va, p_aug, "August")
    n_flag = int(sub["flag"].sum())
    n_test_lv = int((sub["n"] >= MIN_FLAG_N).sum())
    print(f"  {sub['group'].nunique()} groups, {len(sub)} slices; {n_flag} of {n_test_lv} "
          f"testable slices off by more than chance (~{0.05 * n_test_lv:.1f} expected by luck)")
    for r in sub[sub["flag"]].itertuples():
        print(f"    {r.group:<16} {r.level:<24} n={r.n:<5} predicted {r.pred:.1%}  "
              f"observed {r.obs:.1%} [{r.lo:.1%}, {r.hi:.1%}]")

    banner(f"3. Season drift — training months out-of-fold ({args.folds} refits)")
    oof = np.full(len(tr), np.nan)
    for fit_idx, held in grouped_folds(tr["game_pk"].to_numpy(), k=args.folds, seed=args.seed):
        mdl = refit_like(pkey, pmodel, tr.iloc[fit_idx].reset_index(drop=True))
        oof[held] = mdl.predict(tr.iloc[held])
    oof_cal = pcal.predict(oof)
    months = pd.concat([subgroup_table(tr, oof_cal, "train (out-of-fold)"),
                        sub.assign(window="August")])
    months = months[months["group"] == "Month"].reset_index(drop=True)
    for r in months.itertuples():
        print(f"  {r.level[3:]:<4} n={r.n:<5} predicted {r.pred:.1%}  observed {r.obs:.1%} "
              f"[{r.lo:.1%}, {r.hi:.1%}]  {'OFF' if r.flag else ''}")

    # ---------------------------------------------------------- 4. decide
    banner("4. Production model")
    s_p = point[f"{prod_fs}|{chosen[prod_fs]}"]
    s_n = point[f"{prod_fs}|none"]
    decision = decision_text(prod_fs, pkey, labels, chosen, rules, point, diffs, sel, n_flag,
                             n_test_lv)
    print("  " + decision["headline"])

    # ---------------------------------------------------------- 5. lock
    banner("5. Lock")
    feats = model_features(pmodel)
    manifest_extra = {
        "date": dt.date.today().strftime("%Y.%m.%d"),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model_key": pkey, "model_label": labels[pkey], "family": family[pkey],
        "feature_set": prod_fs, "features": feats, "state_fields": required_fields(feats),
        "calibration": {"method": chosen[prod_fs], "describe": pcal.describe(),
                        "n_fit": pcal.n_fit, "fitted_on": "August (the calibration window)"},
        "cohort_fingerprint": data["fingerprint"], "call_check": data["call_check"],
        "windows": window_info(tr, va, data["n_test"]),
        "metrics": {"august_crossfit": clean_metrics(s_p, mid[f"{prod_fs}|{chosen[prod_fs]}"]),
                    "august_crossfit_uncalibrated": clean_metrics(
                        s_n, mid[f"{prod_fs}|none"]),
                    "september": None},
        "test_window": {"opened": False},
        "week8": {"baseline_to_beat": sel["baseline_to_beat"],
                  "candidate_location": sel["candidate_location"],
                  "candidate_full": sel["candidate_full"],
                  "generated": sel.get("generated")},
        "decision": decision, "seed": args.seed,
        "relocked_after_test": bool(STAMP.exists()),
    }

    def save_model(folder):
        if isinstance(pmodel, Booster):
            pmodel.save(folder)
        else:
            with open(Path(folder) / "model.pkl", "wb") as fh:
                pickle.dump(pmodel, fh)

    old = None
    if PROD.exists():
        old = json.loads((PROD / "manifest.json").read_text()).get("version", "unknown") \
            if (PROD / "manifest.json").exists() else "unknown"
        HISTORY.mkdir(parents=True, exist_ok=True)
        if (HISTORY / old).exists():
            shutil.rmtree(HISTORY / old)
        shutil.move(str(PROD), str(HISTORY / old))
    manifest = write_bundle(PROD, model=pmodel, calibrator=pcal, rates=data["rates"],
                            league=data["league"], manifest_extra=manifest_extra,
                            model_files=save_model)
    if old is not None:
        if old == manifest["version"]:
            # same files -> same version: nothing about the model changed
            shutil.rmtree(HISTORY / old, ignore_errors=True)
            print(f"  re-locked: identical to the previous lock (same model files, same "
                  f"version)")
        else:
            print(f"  previous lock {old} kept in {HISTORY / old}")
    # Prove the bundle reproduces what we just scored: load it back through the
    # public interface and predict August from the RAW columns.
    pm = load_production(PROD)
    p_check = pm.predict_frame(states_from_cohort(va), strict=False)
    # Rows whose recorded distance disagrees with their own plate location can't
    # match (predict(state) recomputes distance from the location). There should
    # be none; any that exist are worth a look in the Week 5 table.
    geo = np.abs(zone_distance_in(va["plate_x"], va["plate_z"], va["abs_zone_top"],
                                  va["abs_zone_bot"]) - va["zone_dist_in"].to_numpy(float))
    consistent = ~(geo > 1e-6)
    gap = float(np.max(np.abs(p_check - pcal.predict(raw[prod_fs]))[consistent]))
    manifest["selfcheck_max_gap"] = gap
    manifest["selfcheck_geometry_mismatches"] = int((~consistent).sum())
    (PROD / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  locked p(s) {manifest['version']}: {labels[pkey]} + "
          f"{METHOD_LABEL[chosen[prod_fs]]}")
    print(f"  self-check: the reloaded bundle reproduces August to within {gap:.1e}")
    if (~consistent).any():
        bad = va.loc[~consistent, ["game_pk", "at_bat_number", "pitch_number", "zone_dist_in"]]
        print(f"  NOTE: {len(bad)} August challenge(s) have a recorded distance that doesn't "
              "match their plate location (skipped in the self-check):")
        for r in bad.head(5).itertuples():
            print(f"    game {r.game_pk} AB {r.at_bat_number} pitch {r.pitch_number}: "
                  f"recorded {r.zone_dist_in:+.1f} in.")
    if gap > 1e-6:
        print("  WARNING: the bundle's predictions differ from the in-memory model — "
              "check src/abs_predict.build_features against add_features.")

    # ---------------------------------------------------------- write docs
    DOCS.mkdir(parents=True, exist_ok=True)
    cal_rows = []
    for k_, s in point.items():
        fs, m_ = k_.split("|")
        d_b = diffs[(k_, f"{fs}|none")]["brier"] if m_ != "none" else (0.0, 0.0, 0.0)
        d_l = diffs[(k_, f"{fs}|none")]["log_loss"] if m_ != "none" else (0.0, 0.0, 0.0)
        cal_rows.append({"feature_set": fs, "model": cands[fs], "label": labels[cands[fs]],
                         "method": m_, "chosen": m_ == chosen[fs], **{m: s[m] for m in M.METRICS},
                         "ece_floor": s["ece_floor"],
                         **{f"{m}_lo": ci[k_][m][0] for m in M.METRICS},
                         **{f"{m}_hi": ci[k_][m][1] for m in M.METRICS},
                         "d_brier_vs_none": d_b[0], "d_brier_lo": d_b[1], "d_brier_hi": d_b[2],
                         "d_logloss_vs_none": d_l[0], "d_logloss_lo": d_l[1],
                         "d_logloss_hi": d_l[2], "mid_n": mid[k_]["n"],
                         "mid_pred": mid[k_]["mean_pred"], "mid_obs": mid[k_]["obs_rate"]})
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(DOCS / "week9_calibration.csv", index=False)
    sub.to_csv(DOCS / "week9_subgroups.csv", index=False)
    months.to_csv(DOCS / "week9_months.csv", index=False)
    rel = []
    for k_, p_ in xfit.items():
        for r in M.reliability_table(y, p_, n_bins=10, strategy="uniform"):
            rel.append({"key": k_, **r})
    pd.DataFrame(rel).to_csv(DOCS / "week9_reliability.csv", index=False)
    pd.DataFrame({"game_pk": va["game_pk"], "game_date": va["game_date"],
                  "at_bat_number": va["at_bat_number"], "pitch_number": va["pitch_number"],
                  "y": y, **{f"p_{k_.replace('|', '_')}": v for k_, v in xfit.items()}}).to_csv(
        DATA_DIR / "processed" / "week9_august_crossfit.csv", index=False)

    ctx = dict(data=data, tr=tr, va=va, sel=sel, labels=labels, cands=cands, chosen=chosen,
               rules=rules, point=point, ci=ci, diffs=diffs, mid=mid, final_cal=final_cal,
               sub=sub, months=months, prod_fs=prod_fs, pkey=pkey, manifest=manifest,
               decision=decision, n_flag=n_flag, n_test_lv=n_test_lv, args=args)
    (DOCS / "WEEK9_REPORT.md").write_text(build_report(ctx))
    card = build_model_card(manifest, sub, months, test=None)
    (DOCS / "MODEL_CARD.md").write_text(card)
    (PROD / "MODEL_CARD.md").write_text(card)
    for p_ in (DOCS / "WEEK9_REPORT.md", DOCS / "MODEL_CARD.md", DOCS / "week9_calibration.csv",
               DOCS / "week9_subgroups.csv", DOCS / "week9_months.csv", PROD / "manifest.json"):
        print(f"  {p_}")

    if not args.no_figures:
        banner("6. Figures")
        import week9_figures as W9
        W9.make_lock_figures(FIGS, DATA_DIR, va=va, xfit=xfit, point=point, ci=ci, diffs=diffs,
                             labels=labels, cands=cands, chosen=chosen, prod_fs=prod_fs,
                             sub=sub, months=months, pm=pm, tr=tr)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. p(s) {manifest['version']} is locked.")
    print("Next, ONCE:  python scripts/17_week9_calibrate_lock.py --final-test")


# =========================================================== helpers
def window_info(tr, va, n_test):
    return {"train": {"from": f"{tr['game_date'].min():%Y-%m-%d}",
                      "to": f"{tr['game_date'].max():%Y-%m-%d}", "challenges": int(len(tr))},
            "calibration": {"from": f"{va['game_date'].min():%Y-%m-%d}",
                            "to": f"{va['game_date'].max():%Y-%m-%d}",
                            "challenges": int(len(va))},
            "test": {"challenges": int(n_test)}}


def clean_metrics(s, mid):
    out = {k: (None if not np.isfinite(v) else float(v)) for k, v in s.items()}
    out["midrange"] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                       for k, v in mid.items()}
    return out


def decision_text(prod_fs, pkey, labels, chosen, rules, point, diffs, sel, n_flag, n_test_lv):
    m_ = chosen[prod_fs]
    k_, kn = f"{prod_fs}|{m_}", f"{prod_fs}|none"
    s, sn = point[k_], point[kn]
    if m_ == "none":
        why_cal = ("No calibrator earned its place: neither Platt scaling nor isotonic "
                   "regression improved the cross-fitted August Brier by more than the noise, "
                   "so the model's own probabilities are used (fewer moving parts).")
    else:
        d = diffs[(k_, kn)]["brier"]
        why_cal = (f"{METHOD_LABEL[m_].capitalize()} is the simplest option within noise of "
                   f"the best: cross-fitted August Brier {s['brier']:.4f} vs {sn['brier']:.4f} "
                   f"uncalibrated (Δ {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]); calibration "
                   f"slope {f(sn['cal_slope'], 2)} → {f(s['cal_slope'], 2)}.")
    why_model = (
        "Location only, per the Week 7 decision: ABS rules on the ball's position alone, the "
        "policy will be asked about pitches nobody challenged, and features describing who "
        "chose to challenge would make a policy that judges challenge decisions circular."
        if prod_fs == "location" else
        "Everything (chosen with --production full): the most accurate model for the "
        "challenges teams actually make. Caveat: its non-location effects were learned from "
        "the challenges teams chose, so it is not the right p(s) for pitches nobody challenged.")
    return {"headline": f"p(s) = {labels[pkey]} + {METHOD_LABEL[m_]}",
            "why_model": why_model,
            "why_candidate": (f"{labels[pkey]} is Week 8's {SET_LABEL[prod_fs]} candidate: the "
                              "simplest model whose August Brier can't be told apart from the "
                              "best model with the same features."),
            "why_calibration": why_cal,
            "subgroups": (f"{n_flag} of {n_test_lv} testable slices are off by more than "
                          f"chance (about {0.05 * n_test_lv:.1f} would be by luck alone).")}


# =========================================================== final test
def final_test(args):
    banner("7. FINAL TEST — opening September")
    if not (PROD / "manifest.json").exists():
        raise SystemExit("  Nothing is locked yet. Run the script without --final-test first: "
                         "the model has to be chosen BEFORE the test window is opened.")
    if STAMP.exists() and not args.force_retest:
        st = json.loads(STAMP.read_text())
        raise SystemExit(f"  September was already opened on {st['opened'][:10]} for p(s) "
                         f"{st['version']}. The test set is used exactly once.\n"
                         "  Results: docs/WEEK9_TEST_RESULTS.md. If the earlier run crashed "
                         "before writing them, or you\n  must re-open it anyway, pass "
                         "--force-retest — every opening is recorded in TEST_OPENED.json.")
    pm = load_production(PROD)
    manifest = pm.manifest
    data = load_challenges(DATA_DIR, CONFIG, include_test=True)
    if manifest.get("cohort_fingerprint") != data["fingerprint"]:
        if manifest.get("cohort_fingerprint") in (None, "unknown"):
            raise SystemExit(
                "  p(s) was locked before the cohort fingerprint was recorded (its manifest says "
                "'unknown'),\n  so nothing confirms September would be scored on the same "
                "cohort. Re-lock first:\n      python scripts/run_week89.py\n  then run "
                "--final-test. September is still sealed — nothing was opened.")
        raise SystemExit("  The cohort changed since p(s) was locked. Re-lock first.")
    tr, va, te = data["tr"], data["va"], data["te"]
    sel, models = load_week8()
    if sel.get("generated") != manifest.get("week8", {}).get("generated"):
        print("  NOTE: Week 8 was re-run after p(s) was locked. The locked model is unchanged; "
              "the Week 8 models scored alongside it are the newer ones.")
    labels = {k: m["label"] for k, m in sel["models"].items()}
    y = te["y"].to_numpy(int)
    games = te["game_pk"].to_numpy()
    p_ref = float(tr["y"].mean())
    print(f"  {len(te):,} September challenges in {len(np.unique(games))} games")

    preds = {k: np.clip(m.predict(te), 1e-6, 1 - 1e-6) for k, m in models.items()}
    preds["PROD"] = pm.predict_frame(states_from_cohort(te), strict=False)
    preds["PROD-raw"] = pm.predict_frame(states_from_cohort(te), strict=False, raw=True)
    labels["PROD"] = f"p(s) {manifest['version']} (locked)"
    labels["PROD-raw"] = f"{manifest['model_label']}, uncalibrated"
    point = {k: M.metric_suite(y, p, p_ref) for k, p in preds.items()}
    mid = {k: M.midrange_check(y, p) for k, p in preds.items()}
    print(f"  bootstrapping {args.boot} resamples ...")
    ci, diffs = M.cluster_bootstrap(y, preds, games, p_ref, B=args.boot, seed=args.seed)
    base = sel["baseline_to_beat"]

    sub_te = subgroup_table(te, preds["PROD"], "September")
    sub_aug = pd.read_csv(DOCS / "week9_subgroups.csv") if (DOCS / "week9_subgroups.csv").exists() \
        else pd.DataFrame()
    months = pd.read_csv(DOCS / "week9_months.csv") if (DOCS / "week9_months.csv").exists() \
        else pd.DataFrame()
    if len(months) and "window" in months:
        months = months[months["window"] != "September"]          # a forced re-test replaces it
    months = pd.concat([months, sub_te[sub_te["group"] == "Month"]], ignore_index=True)

    # stamp FIRST: once the numbers exist, the window counts as opened
    history = json.loads(STAMP.read_text()).get("history", []) if STAMP.exists() else []
    history.append({"opened": dt.datetime.now().isoformat(timespec="seconds"),
                    "version": manifest["version"], "forced": bool(STAMP.exists())})
    STAMP.write_text(json.dumps({"opened": history[0]["opened"], "version": history[0]["version"],
                                 "times_opened": len(history), "history": history}, indent=2))

    rows = []
    for k in preds:
        for m_ in M.METRICS:
            rows.append({"model": k, "label": labels[k], "metric": m_, "value": point[k][m_],
                         "ci_lo": ci[k][m_][0], "ci_hi": ci[k][m_][1]})
    pd.DataFrame(rows).to_csv(DOCS / "week9_test_comparison.csv", index=False)
    pd.DataFrame([{"model": k, "label": labels[k],
                   **{f"{m_}_{t}": v for m_ in ("brier", "log_loss")
                      for t, v in zip(("d", "lo", "hi"), diffs[(k, base)][m_])}}
                  for k in preds if k != base]).to_csv(
        DOCS / "week9_test_paired_vs_baseline.csv", index=False)
    sub_te.to_csv(DOCS / "week9_test_subgroups.csv", index=False)
    months.to_csv(DOCS / "week9_months.csv", index=False)
    out = te[["game_pk", "game_date", "at_bat_number", "pitch_number", "y"]].copy()
    for k, p in preds.items():
        out[f"p_{k}"] = p
    out.to_csv(DATA_DIR / "processed" / "week9_test_predictions.csv", index=False)

    s = point["PROD"]
    manifest["metrics"]["september"] = clean_metrics(s, mid["PROD"])
    manifest["test_window"] = {"opened": True, "opened_at": history[-1]["opened"],
                               "times_opened": len(history), "challenges": int(len(te)),
                               "from": f"{te['game_date'].min():%Y-%m-%d}",
                               "to": f"{te['game_date'].max():%Y-%m-%d}"}
    (PROD / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    report = build_test_report(dict(manifest=manifest, labels=labels, point=point, ci=ci,
                                    diffs=diffs, mid=mid, base=base, sel=sel, te=te,
                                    sub_te=sub_te, sub_aug=sub_aug, history=history,
                                    models=models, args=args))
    (DOCS / "WEEK9_TEST_RESULTS.md").write_text(report)
    card = build_model_card(manifest, sub_aug if len(sub_aug) else sub_te, months, test=sub_te)
    (DOCS / "MODEL_CARD.md").write_text(card)
    # the bundle's own copy of the card is outside the hashed files on purpose
    (PROD / "MODEL_CARD.md").write_text(card)
    print(f"  p(s) on September: Brier {fci(s['brier'], ci['PROD']['brier'], 4)}"
          f" · log loss {s['log_loss']:.4f} · slope {f(s['cal_slope'], 2)} · AUC {s['auc']:.3f}")
    if manifest["calibration"]["method"] == "none":
        print("  calibration: none was chosen in August, so p(s) is the model's own output")
    else:
        d = diffs[("PROD", "PROD-raw")]["brier"]
        print(f"  calibration on September: Δ Brier vs uncalibrated {d[0]:+.4f} [{d[1]:+.4f}, "
              f"{d[2]:+.4f}]")
    for p_ in (DOCS / "WEEK9_TEST_RESULTS.md", DOCS / "MODEL_CARD.md",
               DOCS / "week9_test_comparison.csv", STAMP):
        print(f"  {p_}")
    if not args.no_figures:
        import week9_figures as W9
        W9.make_test_figures(FIGS, DATA_DIR, te=te, preds=preds, point=point, ci=ci,
                             diffs=diffs, labels=labels, sel=sel, sub_te=sub_te,
                             sub_aug=sub_aug, months=months)
    print("\nThe test window has now been used. Its numbers go in the report as they are.")


# =========================================================== reports
def build_report(c):
    w_ = []
    w = w_.append
    point, ci, diffs, labels, cands = c["point"], c["ci"], c["diffs"], c["labels"], c["cands"]
    man, dec, prod_fs = c["manifest"], c["decision"], c["prod_fs"]
    demo = "demo" in str(DATA_DIR).lower()
    w("# Week 9 — Calibrate and Lock\n")
    w(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/17_week9_calibrate_lock.py` · "
      f"cohort `{c['data']['fingerprint']}` · {c['args'].folds}-fold cross-fitting · "
      f"{c['args'].boot} game-level bootstrap resamples\n")
    if demo:
        w("> **DEMO DATA — every number below comes from the synthetic fixture.**\n")
    w("> Read this next to `WEEK9.md`, which explains every section.\n")
    w(f"## The locked model\n")
    w(f"**p(s) {man['version']} = {man['model_label']} + {METHOD_LABEL[man['calibration']['method']]}**"
      f"  ·  needs: `{'`, `'.join(man['state_fields'])}`\n")
    w(f"- {dec['why_model']}")
    w(f"- {dec['why_candidate']}")
    w(f"- {dec['why_calibration']}")
    w(f"- Subgroups: {dec['subgroups']}")
    w(f"- September: **sealed**. Run `--final-test` once to score it.\n")

    w("## 0. Pre-flight\n")
    a, b = describe_call_check(c["data"]["call_check"])
    w(f"- Recorded call on challenged pitches: {a}; {b}.")
    w(f"- Same cohort as Week 8 (`{c['data']['fingerprint']}`) — checked.")
    w(f"- Train {len(c['tr']):,} challenges (March–July) · **calibrate** {len(c['va']):,} "
      f"(August) · test {c['data']['n_test']:,} (September, sealed).\n")

    w("## 1. Calibration — August, cross-fitted\n")
    w("Every number below is out-of-sample for the calibrator: each August game was "
      "calibrated by a map fitted on the other four-fifths of August's games.\n")
    w("| candidate | method | Brier [95%] | log loss | ECE (noise) | cal. intercept | "
      "cal. slope | 20–80% band: predicted → observed | Δ Brier vs uncalibrated [95%] |")
    w("|---|---|---|---:|---|---|---|---|---|")
    for fs, k in cands.items():
        for m_ in METHODS:
            key = f"{fs}|{m_}"
            s, cc, md = point[key], ci[key], c["mid"][key]
            star = " ★" if m_ == c["chosen"][fs] else ""
            band = (f"{md['mean_pred']:.0%} → {md['obs_rate']:.0%} (n={md['n']})" if md["n"]
                    else "—")
            if m_ == "none":
                dd = "—"
            else:
                d = diffs[(key, f"{fs}|none")]["brier"]
                dd = f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
            w(f"| {labels[k]} | {METHOD_LABEL[m_]}{star} | {fci(s['brier'], cc['brier'], 4)} | "
              f"{f(s['log_loss'], 4)} | {f(s['ece'], 3)} ({f(s['ece_floor'], 3)}) | "
              f"{fci(s['cal_intercept'], cc['cal_intercept'], 2, True)} | "
              f"{fci(s['cal_slope'], cc['cal_slope'], 2)} | {band} | {dd} |")
    w("\n★ = chosen by the rule fixed before looking: **the simplest method (none → Platt → "
      "isotonic) whose cross-fitted Brier can't be told apart from the best** (paired "
      "bootstrap over games). The final calibrator is then refitted on all of August:\n")
    for fs in cands:
        w(f"- {labels[cands[fs]]}: {c['final_cal'][fs].describe()}")
    w("")

    w("## 2. Subgroups — is p(s) right for every kind of challenge?\n")
    sub = c["sub"]
    w(f"{labels[c['pkey']]} + {METHOD_LABEL[c['chosen'][prod_fs]]}, August cross-fitted. "
      f"A slice is **flagged** when its observed rate's 95% interval excludes the mean "
      f"prediction (only slices with ≥ {MIN_FLAG_N} challenges). With "
      f"{c['n_test_lv']} testable slices, about {0.05 * c['n_test_lv']:.1f} would be flagged "
      f"by luck alone; **{c['n_flag']} were.**\n")
    w("| group | slice | n | predicted | observed [95%] | gap (pts) | |")
    w("|---|---|---:|---:|---|---:|---|")
    for r in sub.itertuples():
        w(f"| {r.group} | {r.level} | {r.n} | {r.pred:.1%} | {r.obs:.1%} [{r.lo:.1%}, "
          f"{r.hi:.1%}] | {r.gap_pts:+.1f} | {'⚠' if r.flag else ''} |")
    w("")

    w("## 3. Season drift — the same model, month by month\n")
    w("Training months are scored **out-of-fold**: the model was refitted 5 times with the "
      "same settings, each time leaving out a fifth of the training games, and each game was "
      "predicted by the fit that never saw it. August is cross-fitted as above.\n")
    w("| month | window | n | predicted | observed [95%] | gap (pts) |")
    w("|---|---|---:|---:|---|---:|")
    for r in c["months"].itertuples():
        w(f"| {r.level[3:]} | {r.window} | {r.n} | {r.pred:.1%} | {r.obs:.1%} [{r.lo:.1%}, "
          f"{r.hi:.1%}] | {r.gap_pts:+.1f} |")
    w("")

    w("## 4. The production decision\n")
    w(f"**{dec['headline']}.**\n")
    for k in ("why_model", "why_candidate", "why_calibration", "subgroups"):
        w(f"- {dec[k]}")
    w("\nTo lock the other candidate instead: `--production "
      f"{'full' if prod_fs == 'location' else 'location'}` (re-run before `--final-test`).\n")

    w("## 5. The locked bundle (`models/production/`)\n")
    w(f"- version **{man['version']}**, created {man['created']}")
    w(f"- model `{man['model_key']}` ({man['family']}), features: "
      f"`{'`, `'.join(man['features'])}`")
    w(f"- calibration: {man['calibration']['describe']}")
    w(f"- self-check: reloading the bundle through `src/abs_predict.py` and predicting August "
      f"from the raw columns reproduces the in-memory model to within "
      f"{man.get('selfcheck_max_gap', float('nan')):.1e}"
      + (f" ({man['selfcheck_geometry_mismatches']} challenge(s) skipped: their recorded "
         "distance doesn't match their plate location — worth checking in the Week 5 table)"
         if man.get("selfcheck_geometry_mismatches") else ""))
    w("- every file is hashed in `manifest.json`; `load_production()` refuses a bundle that "
      "changed. Library versions are pinned in `requirements-lock.txt`.\n")
    w("```python")
    w("from src.abs_predict import load_production")
    w("pm = load_production()")
    ex = {"plate_x": 0.85, "plate_z": 2.4, "batter_height_ft": 6.0, "called_strike": True}
    w(f"pm.predict({ex})")
    w("```\n")
    w("## 6. Figures (`figures/week9/`)\n")
    for name, desc in LOCK_FIGURES:
        w(f"- `{name}` — {desc}")
    w("\nThe model card is `docs/MODEL_CARD.md` (a copy travels inside the bundle).")
    return "\n".join(w_)


LOCK_FIGURES = [
    ("w9_01_calibration.png", "reliability before/after each calibration method (cross-fitted)"),
    ("w9_02_methods.png", "the three methods' scores and their difference from uncalibrated"),
    ("w9_03_subgroups.png", "observed − predicted for every slice"),
    ("w9_04_months.png", "season drift: out-of-fold training months, August (and September)"),
    ("w9_05_surface.png", "the locked p(s) over the strike zone, for a called strike and a "
                          "called ball"),
]


def build_test_report(c):
    w_ = []
    w = w_.append
    man, point, ci, diffs, labels = c["manifest"], c["point"], c["ci"], c["diffs"], c["labels"]
    demo = "demo" in str(DATA_DIR).lower()
    w("# Week 9 — The one look at September\n")
    w(f"Opened {c['history'][-1]['opened']} for p(s) **{man['version']}** · "
      f"{len(c['te']):,} challenges · {c['args'].boot} game-level bootstrap resamples\n")
    if demo:
        w("> **DEMO DATA — every number below comes from the synthetic fixture.**\n")
    if len(c["history"]) > 1:
        w(f"> ⚠ The test window has been opened **{len(c['history'])} times** "
          f"(see `models/production/TEST_OPENED.json`). Only the first look is unbiased.\n")
    w("Nothing was chosen after this point. These are the numbers that go in the report.\n")
    w("## 1. Every model, the same September games\n")
    w("| model | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | ECE | AUC |")
    w("|---|---|---:|---|---|---:|---:|")
    order = ["PROD", "PROD-raw"] + [k for k in c["models"]]
    for k in order:
        s, cc = point[k], ci[k]
        star = " ★" if k == c["base"] else ""
        bold = "**" if k == "PROD" else ""
        w(f"| {bold}{labels[k]}{bold}{star} | {fci(s['brier'], cc['brier'], 4)} | "
          f"{f(s['log_loss'], 3)} | {fci(s['cal_intercept'], cc['cal_intercept'], 2, True)} | "
          f"{fci(s['cal_slope'], cc['cal_slope'], 2)} | {f(s['ece'], 3)} | {f(s['auc'], 3)} |")
    w(f"\n★ = Week 8's baseline to beat.\n")
    w("## 2. Did the August conclusions hold?\n")
    d = diffs[("PROD", "PROD-raw")]
    if man["calibration"]["method"] == "none":
        w("- **Calibration:** August chose no calibrator, so the locked p(s) IS the model's own "
          f"output. Its September calibration slope is "
          f"{fci(point['PROD']['cal_slope'], ci['PROD']['cal_slope'], 2)} and intercept "
          f"{fci(point['PROD']['cal_intercept'], ci['PROD']['cal_intercept'], 2, True)} — if "
          "either interval misses its ideal (slope 1, intercept 0), "
          "say so in the report: it is the honest finding, and it is too late to change.")
    else:
        w(f"- **Calibration:** locked p(s) minus the same model uncalibrated: Δ Brier "
          f"{d['brier'][0]:+.4f} [{d['brier'][1]:+.4f}, {d['brier'][2]:+.4f}], Δ log loss "
          f"{d['log_loss'][0]:+.4f} [{d['log_loss'][1]:+.4f}, {d['log_loss'][2]:+.4f}]. "
          "Below zero = the calibrator chosen on August also helped on September.")
    cl, cf = c["sel"]["candidate_location"], c["sel"]["candidate_full"]
    if cf != cl and (cf, cl) in diffs:
        dd = diffs[(cf, cl)]["brier"]
        w(f"- **Everything vs location only:** {labels[cf]} minus {labels[cl]}: Δ Brier "
          f"{dd[0]:+.4f} [{dd[1]:+.4f}, {dd[2]:+.4f}].")
    boosted = [k for k in c["models"] if c["sel"]["models"][k]["family"] in BACKENDS]
    for k in boosted:
        dd = diffs[(k, c["base"])]["brier"]
        verdict = ("better" if dd[2] < 0 else "worse" if dd[1] > 0 else "can't tell apart")
        w(f"- {labels[k]} minus the baseline ({labels[c['base']]}): Δ Brier {dd[0]:+.4f} "
          f"[{dd[1]:+.4f}, {dd[2]:+.4f}] — {verdict}.")
    md = c["mid"]["PROD"]
    if md["n"]:
        w(f"- **Where decisions are made** (p(s) between 20% and 80%): predicted "
          f"{md['mean_pred']:.0%}, observed {md['obs_rate']:.0%} [{md['ci_lo']:.0%}, "
          f"{md['ci_hi']:.0%}], n = {md['n']}.")
    w("")
    w("## 3. Subgroups on September\n")
    sub = c["sub_te"]
    n_t = int((sub["n"] >= MIN_FLAG_N).sum())
    w(f"{int(sub['flag'].sum())} of {n_t} testable slices flagged (≈ {0.05 * n_t:.1f} expected "
      "by chance).\n")
    w("| group | slice | n | predicted | observed [95%] | gap (pts) | |")
    w("|---|---|---:|---:|---|---:|---|")
    for r in sub.itertuples():
        w(f"| {r.group} | {r.level} | {r.n} | {r.pred:.1%} | {r.obs:.1%} [{r.lo:.1%}, "
          f"{r.hi:.1%}] | {r.gap_pts:+.1f} | {'⚠' if r.flag else ''} |")
    w("\n## 4. Figures\n")
    w("- `figures/week9/w9_06_test.png` — every model on September, and the locked p(s)'s "
      "reliability diagram")
    w("- `figures/week9/w9_03_subgroups.png`, `w9_04_months.png` — redrawn with September added")
    return "\n".join(w_)


def build_model_card(man, sub, months, test=None):
    """docs/MODEL_CARD.md — the plan's sections, filled from the manifest."""
    w_ = []
    w = w_.append
    win = man["windows"]
    a = man["metrics"]["august_crossfit"]
    au = man["metrics"]["august_crossfit_uncalibrated"]
    se = man["metrics"].get("september")
    demo = "demo" in str(DATA_DIR).lower()
    w(f"# Model card — p(s), the ABS challenge overturn probability\n")
    if demo:
        w("> **DEMO DATA — this card was generated from the synthetic fixture.**\n")
    w(f"**Version** {man['version']} · locked {man['created']} · bundle `models/production/` · "
      f"content hash `{man['content_hash'][:16]}…`\n")
    w("## Intended use\n")
    w("p(s): the probability that ABS overturns the umpire's call **if the call is "
      "challenged**, given the state s of the pitch. It is the overturn input to the "
      "Q-function of the challenge MDP (Week 11):\n")
    w("    Q_challenge(s, r) = p(s)·V(s_overturned, r) + (1 − p(s))·V(s_stands, r − 1)\n")
    w("It is also used to measure decision regret (what a challenge was worth vs what the "
      "team did) and by the dashboard's simulator.\n")
    w("## What it should NOT be used for\n")
    w("- **Judging umpires.** p(s) is about challenged calls; it is not a measure of an "
      "umpire's accuracy.")
    w("- **Other seasons or rule sets** without re-validation (2026 is the first season of the "
      "challenge system; the zone definition or challenge rules may change).")
    w("- **Pitches that weren't taken** (swings), or anything other than the ABS rectangle "
      "defined below.")
    if man["feature_set"] == "full":
        w("- **Pitches nobody challenged**, as a 'what if' — this model's non-location effects "
          "were learned from the challenges teams chose to make.")
    w("- **As a guarantee for one pitch.** It is calibrated on average and by slice; any single "
      "prediction carries the measurement error of the pitch location (about an inch).\n")
    w("## Model\n")
    w(f"- **{man['model_label']}** (`{man['model_key']}`, family: {man['family']}), "
      f"feature set: {SET_LABEL.get(man['feature_set'], man['feature_set'])}")
    w(f"- Features: `{'`, `'.join(man['features'])}`")
    w(f"- Input state (`predict(state)`): `{'`, `'.join(man['state_fields'])}`")
    w(f"- Calibration: {man['calibration']['describe']}, fitted on {man['calibration']['fitted_on']}")
    w("- Strike zone: 17 in. wide; top 53.5% and bottom 27% of the batter's height; a strike "
      "if any part of the ball (radius 1.45 in.) touches it. Distance is measured from the "
      "ball's surface and signed against the call.\n")
    w("## Training data\n")
    w(f"- MLB 2026 regular season, every challenged pitch with a known result "
      f"(cohort `{man['cohort_fingerprint']}`).")
    w(f"- Train {win['train']['from']} → {win['train']['to']} ({win['train']['challenges']:,} "
      f"challenges) · calibrate {win['calibration']['from']} → {win['calibration']['to']} "
      f"({win['calibration']['challenges']:,}) · test: September ({win['test']['challenges']:,}).")
    cc = man.get("call_check") or {}
    if cc:
        w(f"- Call recording check: {cc.get('verdict', '—')}"
          + (f" ({cc.get('n_repaired', 0):,} challenged pitches corrected to the original call)"
             if cc.get("repair") else "") + ".")
    w(f"- Sources: Statcast (pitch location), MLB Stats API (challenges, umpires, heights), "
      "derived challenge inventory.\n")
    w("## Performance\n")
    w("| window | Brier | log loss | cal. intercept | cal. slope | ECE | AUC |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    w(f"| August, uncalibrated (cross-fitted) | {f(au['brier'], 4)} | {f(au['log_loss'], 4)} | "
      f"{f(au['cal_intercept'], 2, True)} | {f(au['cal_slope'], 2)} | {f(au['ece'], 3)} | "
      f"{f(au['auc'], 3)} |")
    w(f"| **August, calibrated (cross-fitted)** | {f(a['brier'], 4)} | {f(a['log_loss'], 4)} | "
      f"{f(a['cal_intercept'], 2, True)} | {f(a['cal_slope'], 2)} | {f(a['ece'], 3)} | "
      f"{f(a['auc'], 3)} |")
    if se:
        w(f"| **September (test, opened once)** | {f(se['brier'], 4)} | {f(se['log_loss'], 4)} | "
          f"{f(se['cal_intercept'], 2, True)} | {f(se['cal_slope'], 2)} | {f(se['ece'], 3)} | "
          f"{f(se['auc'], 3)} |")
    else:
        w("| September (test) | sealed | | | | | |")
    w("")
    w("## Performance by subgroup\n")
    for name, df in (("August (cross-fitted)", sub), ("September (test)", test)):
        if df is None or not len(df):
            continue
        n_t = int((df["n"] >= MIN_FLAG_N).sum())
        w(f"**{name}** — {int(df['flag'].sum())} of {n_t} testable slices flagged "
          f"(≈ {0.05 * n_t:.1f} by chance).\n")
        w("| group | slice | n | predicted | observed [95%] |")
        w("|---|---|---:|---:|---|")
        for r in df.itertuples():
            w(f"| {r.group} | {r.level} | {r.n} | {r.pred:.1%} | {r.obs:.1%} "
              f"[{r.lo:.1%}, {r.hi:.1%}]{' ⚠' if r.flag else ''} |")
        w("")
    w("## Known limitations\n")
    w("1. **Selection on unobservables.** Trained on challenged pitches only; teams challenge "
      "pitches they suspect were missed, so for an unchallenged pitch at the same location "
      "the true chance may be lower. ABS's ruling is never observed without a challenge.")
    w("2. **Zone geometry.** Heights are listed (rounded to the inch) rather than ABS-measured; "
      "a one-inch height error moves the top edge about half an inch.")
    w("3. **Measurement error** in Statcast's plate location vs the Hawk-Eye measurement ABS "
      "uses — the reason p(s) climbs over an inch or two instead of jumping at the edge.")
    w("4. **One season, one calibration month, one test month.** Intervals cover sampling "
      "noise, not year-to-year change.")
    w("5. **August was used twice** — to choose models (Weeks 7–8) and to fit the calibrator — "
      "so August scores are slightly optimistic; September is the honest number.\n")
    w("## Ethical and practical considerations\n")
    w("- Public data only; players appear as IDs in the model.")
    w("- A recommendation tool, not an automated decision: the policy built on p(s) advises "
      "whether a challenge is worth its option value.\n")
    w("## Versioning\n")
    w(f"- Libraries: " + ", ".join(f"{k} {v}" for k, v in man["versions"].items() if v))
    w("- Pinned in `models/production/requirements-lock.txt`; every file's SHA-256 is in "
      "`models/production/manifest.json`, checked on every load.")
    w("- Earlier locks are kept in `models/production_history/<version>/`.")
    tw = man.get("test_window", {})
    if tw.get("opened"):
        w(f"- Test window opened {tw.get('opened_at')} ({tw.get('times_opened', 1)} time(s)).")
    if man.get("relocked_after_test"):
        w("- ⚠ **This version was locked AFTER September had been opened** — its September "
          "score is not an unbiased estimate.")
    return "\n".join(w_)


if __name__ == "__main__":
    main()
