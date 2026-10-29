#!/usr/bin/env python3
"""
15_week7_baseline.py — Week 7. The baseline overturn models, graded on calibration first.

The question every model this term answers:

    Given that a challenge happened, what is the probability ABS overturns the call?

That probability is p(s) in your Q-function. Week 11's MDP will compare
p(s)·V(overturned) + (1−p(s))·V(stand, r−1) against V(stand, r), so p(s) has
to be an HONEST probability — not just good at ranking pitches. That's why
every model here is graded on calibration before it's graded on AUC.

WHAT IT FITS (all on the TRAINING window, March–July)
    M0   base rate            "every challenge has the league overturn rate"
    M1   distance only        how far the pitch sits on the wrong side of the call
    M2   location             + which edge (side/top/bottom), with its own slope
                                and its own 50% point per call direction
    M3   + situation          count, outs, inning, leverage, challenges left
    M4   + pitch              type, velocity, movement, handedness, batter height
    M5   + people             pitcher-vs-catcher, umpire and catcher accuracy
    GAM  location             M2 with the straight line in distance replaced by
                                a smooth curve per edge
    GAM  everything           M5, likewise

WHAT IT REPORTS (all on the VALIDATION window, August — the test window stays
sealed until the end of Week 8)
    Brier, Brier skill, log loss, calibration intercept and slope, ECE with its
    noise floor, a mid-range check, then AUC — each with a 95% interval from a
    bootstrap that resamples whole GAMES. Then: what each block adds (in-sample
    likelihood-ratio test next to out-of-sample change), the location model in
    inches, odds ratios, subgroup calibration, and which model is the baseline
    to beat.

    python scripts/15_week7_baseline.py
    python scripts/15_week7_baseline.py --boot 200          # faster while iterating
    ABS_DATA_DIR=data_demo python scripts/15_week7_baseline.py

Reads   data/processed/cohort_2026.parquet     (Week 6 — run 13_define_cohort.py)
        config/cohort.json                      (for the cohort fingerprint)
Writes  docs/WEEK7_BASELINE_REPORT.md           the report, numbers filled in
        docs/week7_metrics.csv                  every metric, every model, 95% CIs
        docs/week7_comparisons.csv              paired model-vs-model differences
        docs/week7_coefficients.csv             every logistic coefficient + odds ratio
        docs/week7_reliability.csv              the bins behind each reliability diagram
        figures/week7/*.png                     7 figures, white background
        models/week7_*.pkl                      fitted models (git-ignored)
        models/week7_feature_spec.json          features, levels, settings, decision
        data/processed/week7_validate_predictions.csv

Read WEEK7.md before reading the output — it explains every number.
"""

import argparse
import datetime as dt
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eda"))

from src import abs_metrics as M                                   # noqa: E402
from src.abs_features import (BLOCKS, UNITS, add_features,  # noqa: E402
                              challenge_rows, describe_call_check, load_cohort,
                              participant_rates, split_masks)
from src.abs_models import GAM, Logistic, lr_test                  # noqa: E402
from src.abs_pipeline import resolve_fingerprint                   # noqa: E402

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
CONFIG = Path(os.environ.get("ABS_CONFIG_DIR", "config"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "week7"
MODELS = Path(os.environ.get("ABS_MODEL_DIR", "models"))

# ---------------------------------------------------------------- the models
# The ladder (M0–M5, the two GAMs), the comparisons and the decision rule live
# in src/abs_baselines.py so Weeks 8 and 9 refit exactly the same models.
from src.abs_baselines import (CANDIDATES, LOCATION_ONLY, LR_PAIRS,  # noqa: E402
                               MODEL_SPECS, ORDER, STEPS, choose, fit_model,
                               n_params)


# ===================================================================== utils
def banner(msg):
    print(f"\n{msg}\n{'-' * len(msg)}")


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fci(v, ci, d=3, sign=False):
    lo, hi = ci
    if not np.isfinite(lo):
        return f(v, d, sign)
    return f"{f(v, d, sign)} [{f(lo, d, sign)}, {f(hi, d, sign)}]"


def pct(x, d=1):
    return "—" if not np.isfinite(x) else f"{100 * x:.{d}f}%"


def ensure_columns(df):
    """Every column a block asks for must exist; missing ones become all-NaN
    (the Design drops them and the report lists them as dropped)."""
    missing = []
    for b in BLOCKS.values():
        for c in b.get("numeric", []) + b.get("categorical", []):
            if c not in df.columns:
                df[c] = np.nan
                missing.append(c)
    return df, missing


def crossings(xs, p, level):
    """Every x where the curve p(x) passes through `level` (linear interpolation)."""
    d = np.asarray(p, float) - level
    i = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    out = [float(xs[j] - d[j] * (xs[j + 1] - xs[j]) / (d[j + 1] - d[j])) for j in i]
    out += [float(x) for x in xs[d == 0]]
    return sorted(out)


def crossing_near(xs, p, level, near=0.0):
    """The crossing of `level` closest to `near` (a GAM curve can wiggle back
    through a level far out in the tails, where there's little data)."""
    c = crossings(xs, p, level)
    return min(c, key=lambda v: abs(v - near)) if c else np.nan


def grid_frame(base, xs):
    g = pd.DataFrame([base] * len(xs))
    g["dist_against_call"] = xs
    return g


def typical_row(tr):
    """Median numerics / most common categories — the 'everything else held
    at typical values' row for partial-dependence curves."""
    row = {}
    for c in tr.columns:
        s = tr[c]
        if s.dtype.kind in "if":
            row[c] = float(s.median()) if s.notna().any() else 0.0
        elif s.dtype.kind == "b":
            row[c] = bool(s.mode().iloc[0]) if len(s) else False
        else:
            row[c] = s.mode().iloc[0] if s.notna().any() else ""
    return row


def edge_call_table(model, tr, edges, xs):
    """50% point, its 95% band, and the 25->75% transition width, for every
    edge x call direction. Works the same for the logistic and the GAM."""
    base = typical_row(tr)
    rows = []
    for e in edges:
        for cd, who in ((-1.0, "called strike (batter challenges)"),
                        (1.0, "called ball (fielder challenges)")):
            b = dict(base, edge=e, call_dir=cd, called_strike=(cd < 0))
            p, lo, hi = model.predict_band(grid_frame(b, xs))
            x50 = crossing_near(xs, p, 0.5)
            # 95% interval for the coin-flip point: where the band's edges cross
            # 50% on either side of it (the upper band crosses first).
            left = [c for c in crossings(xs, hi, 0.5) if c <= x50] if np.isfinite(x50) else []
            right = [c for c in crossings(xs, lo, 0.5) if c >= x50] if np.isfinite(x50) else []
            rows.append({
                "edge": e, "call": who, "call_dir": cd,
                "p50": x50,
                "p50_lo": max(left) if left else np.nan,
                "p50_hi": min(right) if right else np.nan,
                "width": (crossing_near(xs, p, 0.75, x50) - crossing_near(xs, p, 0.25, x50)
                          if np.isfinite(x50) else np.nan),
                "p_at_edge": float(np.interp(0.0, xs, p)),
            })
    return pd.DataFrame(rows)


def subgroups(df, p):
    """Observed vs predicted overturn rate within slices of the data."""
    def lev_tier(s):
        li = np.exp(s)
        return np.where(df["leverage_missing"], "unknown",
                        np.where(li < 0.85, "low (<0.85)",
                                 np.where(li <= 2.0, "medium", "high (>2)")))
    groups = [
        ("Who challenged", df["challenger_role"].str.lower()),
        ("Edge · call", df["edge"] + " · " + np.where(df["called_strike"], "strike", "ball")),
        ("Count", pd.Series(np.where(df["strikes"] >= 2, "2 strikes", "0–1 strikes"), index=df.index)),
        ("Challenges in hand", df["chal_left"].map(lambda v: f"{int(v)} left" if np.isfinite(v) else "?")),
        ("Inning", pd.cut(df["inning"], [0, 3, 6, 9, 99], labels=["1–3", "4–6", "7–9", "10+"]).astype(str)),
        ("Leverage", pd.Series(lev_tier(df["log_leverage"]), index=df.index)),
    ]
    order = {"Who challenged": ["batter", "catcher", "pitcher"],
             "Edge · call": [f"{e} · {c}" for e in ("side", "top", "bottom")
                             for c in ("strike", "ball")],
             "Count": ["0–1 strikes", "2 strikes"],
             "Challenges in hand": ["0 left", "1 left", "2 left", "3 left"],
             "Inning": ["1–3", "4–6", "7–9", "10+"],
             "Leverage": ["low (<0.85)", "medium", "high (>2)", "unknown"]}
    rows = []
    y = df["y"].to_numpy()
    for gname, lab in groups:
        lab = pd.Series(lab, index=df.index).astype(str)
        known = order.get(gname, [])
        levels = [v for v in known if v in set(lab)] + sorted(set(lab) - set(known))
        for level in levels:
            m = (lab == level).to_numpy()
            n = int(m.sum())
            if n < 15:
                continue
            k = int(y[m].sum())
            lo, hi = M.wilson(k, n)
            rows.append({"group": gname, "level": level, "n": n,
                         "pred": float(p[m].mean()), "obs": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def by_month(tr, va, p_tr, p_va):
    rows = []
    for name, d, p in (("train (in-sample)", tr, p_tr), ("validate", va, p_va)):
        mon = d["game_date"].dt.to_period("M").astype(str).to_numpy()
        y = d["y"].to_numpy()
        for mth in sorted(set(mon)):
            m = mon == mth
            n = int(m.sum())
            if n < 15:
                continue
            k = int(y[m].sum())
            lo, hi = M.wilson(k, n)
            rows.append({"window": name, "month": mth, "n": n,
                         "pred": float(p[m].mean()), "obs": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap resamples of validation games (default 1000)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ------------------------------------------------------------ 1. data
    banner("1. Load the frozen Week 6 cohort")
    raw = load_cohort(DATA_DIR)
    fingerprint, fp_note = resolve_fingerprint(raw, CONFIG)
    print(f"  {len(raw):,} called pitches · cohort fingerprint {fingerprint}")
    if fp_note:
        print(f"  ({fp_note})")

    df, call_check = add_features(raw, return_check=True)
    for line in describe_call_check(call_check):
        print(f"  {line}")
    tr_mask, va_mask, te_mask = split_masks(df)
    df, league, rates = participant_rates(df, tr_mask)
    df, missing_cols = ensure_columns(df)
    n_train_called = int((tr_mask & df["is_called_pitch"].to_numpy(bool)).sum())
    print(f"  umpire / catcher accuracy from {n_train_called:,} TRAINING called pitches "
          f"(league agreement with ABS: {league:.2%})")

    ch = challenge_rows(df)
    tr = ch[ch["split"] == "train"].reset_index(drop=True)
    va = ch[ch["split"] == "validate"].reset_index(drop=True)
    n_test = int((ch["split"] == "test").sum())
    del ch                                            # test outcomes go no further
    print(f"  challenges with a known outcome: train {len(tr):,} · validate {len(va):,} · "
          f"test {n_test:,} (sealed — not read)")
    if len(tr) < 200:
        raise SystemExit("  Fewer than 200 training challenges — too few to fit anything "
                         "meaningful. Check the Week 6 split dates.")
    if len(va) < 30:
        raise SystemExit("  Fewer than 30 validation challenges — August is empty. "
                         "Check the Week 6 split dates and that August data was pulled.")
    small_va = len(va) < 300
    p_ref = float(tr["y"].mean())
    print(f"  overturn rate: train {p_ref:.1%} · validate {va['y'].mean():.1%}")

    # ------------------------------------------------- 2. the sign trap
    banner("2. Check the sign of the distance feature")
    auc_raw = M.auc(tr["y"], tr["zone_dist_in"])
    auc_res = M.auc(tr["y"], tr["dist_against_call"])
    print(f"  AUC of zone_dist_in as-is      {auc_raw:.3f}   (signed relative to the zone)")
    print(f"  AUC of dist_against_call       {auc_res:.3f}   (signed relative to the call)")

    # ------------------------------------------------------- 3. fit
    banner("3. Fit every model on the training window")
    models = {}
    for key in ORDER:
        t1 = time.time()
        models[key] = fit_model(key, tr)
        m = models[key]
        extra = (f" · λ={m.lam:g}, EDF {m.edf_total:.1f}" if isinstance(m, GAM)
                 else f" · {m.k} coefficient{'s' if m.k != 1 else ''}")
        note = f"   NOTE: {m.note}" if m.note else ""
        print(f"  {MODEL_SPECS[key][0]:<20}{extra} ({time.time() - t1:.1f}s){note}")

    # ------------------------------------------------- 4. evaluate
    banner("4. Score every model on the validation window")
    preds_va = {k: np.clip(m.predict(va), 1e-6, 1 - 1e-6) for k, m in models.items()}
    preds_tr = {k: np.clip(m.predict(tr), 1e-6, 1 - 1e-6) for k, m in models.items()}
    point = {k: M.metric_suite(va["y"], p, p_ref) for k, p in preds_va.items()}
    floor = {k: M.ece_noise_floor(p, n_sim=300, seed=args.seed) for k, p in preds_va.items()}
    mid = {k: M.midrange_check(va["y"], p) for k, p in preds_va.items()}

    print(f"  bootstrapping {args.boot} resamples of {va['game_pk'].nunique()} validation games ...")
    t1 = time.time()
    ci, diffs = M.cluster_bootstrap(va["y"].to_numpy(), preds_va, va["game_pk"].to_numpy(),
                                    p_ref, B=args.boot, seed=args.seed)
    print(f"  done in {time.time() - t1:.0f}s")

    print(f"\n  {'model':<20}{'Brier':>8}{'logloss':>9}{'icpt':>8}{'slope':>7}{'ECE':>7}{'(floor)':>8}{'AUC':>7}")
    for k in ORDER:
        s = point[k]
        print(f"  {MODEL_SPECS[k][0]:<20}{s['brier']:>8.4f}{s['log_loss']:>9.4f}"
              f"{f(s['cal_intercept'], 2, True):>8}{f(s['cal_slope'], 2):>7}"
              f"{s['ece']:>7.3f}{floor[k]:>8.3f}{s['auc']:>7.3f}")

    # ---------------------------------------- 5. block contributions
    banner("5. What each block adds")
    lr = {}
    for big, small in LR_PAIRS:
        lr[(big, small)] = lr_test(models[small], models[big])
    step_rows = []
    for big, small, what in STEPS:
        d_b = diffs[(big, small)]["brier"]
        d_l = diffs[(big, small)]["log_loss"]
        d_a = diffs[(big, small)]["auc"]
        lr_stat, lr_df, lr_p = lr.get((big, small), (np.nan, np.nan, np.nan))
        if d_b[2] < 0:
            verdict = "better on new data"
        elif d_b[1] > 0:
            verdict = "worse on new data"
        else:
            verdict = "no detectable difference"
        # Brier decides; say so when log loss is clear-cut on its own.
        if d_l[2] < 0 and d_b[2] >= 0:
            verdict += " (log loss: better)"
        elif d_l[1] > 0 and d_b[1] <= 0:
            verdict += " (log loss: worse)"
        step_rows.append({
            "step": what, "bigger": big, "smaller": small,
            "params_added": n_params(models[big]) - n_params(models[small]),
            "lr_chi2": lr_stat, "lr_df": lr_df, "lr_p": lr_p,
            "d_brier": d_b[0], "d_brier_lo": d_b[1], "d_brier_hi": d_b[2],
            "d_logloss": d_l[0], "d_logloss_lo": d_l[1], "d_logloss_hi": d_l[2],
            "d_auc": d_a[0], "verdict": verdict})
        lr_txt = f"LR p={lr_p:.3g} (train)" if np.isfinite(lr_p) else "(not nested)"
        print(f"  {what:<38} ΔBrier {d_b[0]:+.4f} [{d_b[1]:+.4f}, {d_b[2]:+.4f}]  "
              f"{lr_txt:<22} -> {verdict}")
    steps = pd.DataFrame(step_rows)

    # ------------------------------------------- 6. location in inches
    banner("6. The location model in baseball language")
    edges = [e for e in ("side", "top", "bottom") if e in set(tr["edge"])]
    xs = np.linspace(-10, 10, 2001)
    loc_tab = {k: edge_call_table(models[k], tr, edges, xs) for k in ("M2", "GAM-loc")}
    for k in ("M2", "GAM-loc"):
        print(f"  {MODEL_SPECS[k][0]}")
        for _, r in loc_tab[k].iterrows():
            print(f"    {r['edge']:<7} {r['call']:<34} 50% point {f(r['p50'], 2, True):>6} in. "
                  f"[{f(r['p50_lo'], 2, True)}, {f(r['p50_hi'], 2, True)}] · "
                  f"25→75% over {f(r['width'], 2)} in.")
    gl = models["GAM-loc"]
    for name, e in gl.edf.items():
        print(f"  EDF {name}: {e:.2f}   (1 = straight line)")

    # ------------------------------------------------ 7. subgroups
    mdp_key = choose(LOCATION_ONLY, point, diffs)[1]
    sub = subgroups(va, preds_va[mdp_key])
    months = by_month(tr, va, preds_tr[mdp_key], preds_va[mdp_key])

    # ------------------------------------------ 8. baseline to beat
    banner("7. The baseline to beat")
    best_key, chosen_key, rule_rows = choose(CANDIDATES, point, diffs)
    print(f"  lowest validation Brier: {MODEL_SPECS[best_key][0]}")
    print(f"  BASELINE TO BEAT:        {MODEL_SPECS[chosen_key][0]}")
    print(f"  location-only candidate for the MDP's p(s): {MODEL_SPECS[mdp_key][0]}")

    # ---------------------------------------------- 9. leak check
    leak_flag = (point["M5"]["auc"] - point["M2"]["auc"] > 0.02) or \
                (point["GAM-full"]["auc"] - point["GAM-loc"]["auc"] > 0.02)

    # ------------------------------------------- 10. write everything
    banner("8. Write outputs")
    DOCS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    met_rows = []
    for k in ORDER:
        for m_ in M.METRICS:
            met_rows.append({"model": k, "label": MODEL_SPECS[k][0], "metric": m_,
                             "value": point[k][m_], "ci_lo": ci[k][m_][0], "ci_hi": ci[k][m_][1]})
        met_rows.append({"model": k, "label": MODEL_SPECS[k][0], "metric": "ece_noise_floor",
                         "value": floor[k], "ci_lo": np.nan, "ci_hi": np.nan})
        met_rows.append({"model": k, "label": MODEL_SPECS[k][0], "metric": "n_params_or_edf",
                         "value": n_params(models[k]), "ci_lo": np.nan, "ci_hi": np.nan})
    pd.DataFrame(met_rows).to_csv(DOCS / "week7_metrics.csv", index=False)

    comp_rows = []
    for (a, b), dd in diffs.items():
        for m_ in M.METRICS:
            mean_, lo_, hi_ = dd[m_]
            comp_rows.append({"model_a": a, "model_b": b, "metric": m_,
                              "a_minus_b": mean_, "ci_lo": lo_, "ci_hi": hi_})
    pd.DataFrame(comp_rows).to_csv(DOCS / "week7_comparisons.csv", index=False)

    coef = []
    for k in ORDER:
        if isinstance(models[k], Logistic):
            t = models[k].table()
            t.insert(0, "model", k)
            coef.append(t)
    coef = pd.concat(coef, ignore_index=True)
    coef.to_csv(DOCS / "week7_coefficients.csv", index=False)

    rel_rows = []
    for k in ORDER:
        for r in M.reliability_table(va["y"], preds_va[k], n_bins=10, strategy="uniform"):
            rel_rows.append(dict(model=k, **r))
    pd.DataFrame(rel_rows).to_csv(DOCS / "week7_reliability.csv", index=False)

    pd.concat([t.assign(model=kk) for kk, t in loc_tab.items()], ignore_index=True) \
        .to_csv(DOCS / "week7_location.csv", index=False)
    wrong = confidently_wrong(va, preds_va["M2"])
    wrong.to_csv(DOCS / "week7_confidently_wrong.csv", index=False)

    out = va[["game_pk", "game_date", "at_bat_number", "pitch_number", "y"]].copy()
    for k in ORDER:
        out[f"p_{k}"] = preds_va[k]
    pred_path = DATA_DIR / "processed" / "week7_validate_predictions.csv"
    out.to_csv(pred_path, index=False)

    for k, m in models.items():
        with open(MODELS / f"week7_{k}.pkl", "wb") as fh:
            pickle.dump(m, fh)
    with open(MODELS / "week7_participant_rates.pkl", "wb") as fh:
        pickle.dump({"league": league, "shrink": 200, "rates": rates}, fh)

    spec = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "cohort_fingerprint": fingerprint,
        "data_dir": str(DATA_DIR),
        "n_challenges": {"train": len(tr), "validate": len(va), "test_sealed": n_test},
        "blocks": BLOCKS,
        "models": {k: {"label": MODEL_SPECS[k][0], "kind": MODEL_SPECS[k][1],
                       "blocks": MODEL_SPECS[k][2] if k != "M1" else ["dist_against_call only"],
                       "terms": getattr(models[k], "names", None) or getattr(models[k], "pnames", None),
                       "folded_levels": models[k].design.folded,
                       "dropped_columns": models[k].design.dropped,
                       "note": models[k].note,
                       **({"lambda": models[k].lam, "edf": models[k].edf}
                          if isinstance(models[k], GAM) else {})}
                   for k in ORDER},
        "participant_rates": {"league_agreement": league, "shrink": 200,
                              "source": "training-window called pitches only"},
        "baseline_to_beat": chosen_key,
        "mdp_location_candidate": mdp_key,
        "auc_zone_dist_in_raw": auc_raw,
        "auc_dist_against_call": auc_res,
    }
    (MODELS / "week7_feature_spec.json").write_text(json.dumps(spec, indent=2, default=str))

    report = build_report(
        fingerprint=fingerprint, raw=raw, tr=tr, va=va, n_test=n_test, p_ref=p_ref,
        league=league, n_train_called=n_train_called, missing_cols=missing_cols,
        auc_raw=auc_raw, auc_res=auc_res, models=models, point=point, ci=ci,
        floor=floor, mid=mid, diffs=diffs, steps=steps, loc_tab=loc_tab,
        sub=sub, months=months, best_key=best_key, chosen_key=chosen_key,
        rule_rows=rule_rows, mdp_key=mdp_key, wrong=wrong, leak_flag=leak_flag,
        call_check=call_check,
        small_va=small_va, boot=args.boot)
    (DOCS / "WEEK7_BASELINE_REPORT.md").write_text(report)

    for p_ in (DOCS / "WEEK7_BASELINE_REPORT.md", DOCS / "week7_metrics.csv",
               DOCS / "week7_comparisons.csv", DOCS / "week7_coefficients.csv",
               DOCS / "week7_reliability.csv", DOCS / "week7_location.csv",
               DOCS / "week7_confidently_wrong.csv", pred_path,
               MODELS / "week7_feature_spec.json"):
        print(f"  {p_}")
    print(f"  {MODELS}/week7_*.pkl  ({len(models)} models + participant rates)")

    if not args.no_figures:
        banner("9. Figures")
        import week7_figures as W      # noqa: E402  (lives next to this script)
        W.make_all(FIGS, DATA_DIR, tr=tr, va=va, models=models, preds_va=preds_va,
                   point=point, ci=ci, floor=floor, steps=steps, loc_tab=loc_tab,
                   sub=sub, months=months, mdp_key=mdp_key, chosen_key=chosen_key,
                   auc_raw=auc_raw, auc_res=auc_res, labels={k: v[0] for k, v in MODEL_SPECS.items()},
                   xs=xs, edges=edges, base_row=typical_row(tr), term_unit=term_unit, order=ORDER)

    print(f"\nDone in {time.time() - t0:.0f}s. Read {DOCS / 'WEEK7_BASELINE_REPORT.md'} "
          f"with WEEK7.md open beside it.")


def confidently_wrong(va, p, n=10):
    """The n validation challenges with the largest log-loss penalty."""
    yv = va["y"].to_numpy()
    ll = -(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    top = np.argsort(-ll)[:n]
    rows = []
    for i in top:
        r = va.iloc[i]
        rows.append({"game_pk": r["game_pk"], "date": f"{r['game_date']:%Y-%m-%d}",
                     "at_bat_number": r["at_bat_number"], "pitch_number": r["pitch_number"],
                     "inning": r["inning"], "count": f"{int(r['balls'])}-{int(r['strikes'])}",
                     "challenger": r["challenger_role"],
                     "call": "strike" if r["called_strike"] else "ball", "edge": r["edge"],
                     "in_against_call": float(r["dist_against_call"]), "p_overturn": float(p[i]),
                     "result": "overturned" if yv[i] else "stood", "log_loss": float(ll[i])})
    return pd.DataFrame(rows)


# ================================================================== report
def build_report(**k):
    L = MODEL_SPECS
    tr, va, point, ci, models = k["tr"], k["va"], k["point"], k["ci"], k["models"]
    demo = "demo" in str(DATA_DIR).lower()
    out = []
    w = out.append

    w("# Week 7 — Baseline Overturn Model Report\n")
    w(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/15_week7_baseline.py` · "
      f"cohort `{k['fingerprint']}` · data `{DATA_DIR}` · {k['boot']} game-level bootstrap resamples\n")
    if demo:
        w("> **DEMO DATA — every number below comes from the synthetic fixture.** "
          "Re-run on the real cohort before quoting anything.\n")
    w("> Read this next to `WEEK7.md`, which explains what each number means and "
      "what to look for.\n")

    # 1 ----------------------------------------------------------------
    w("## 1. The question, and why calibration comes first\n")
    w("Every model here estimates **p(s): given that a challenge happened, the probability "
      "ABS overturns the call.** That probability is an input to the Week 11 MDP, which "
      "challenges when p(s)·V(overturned) + (1 − p(s))·V(stand, r − 1) beats V(stand, r).\n")
    w("**We care about calibration over discrimination because the output feeds an MDP.** "
      "The decision compares p(s) against a threshold set by the game state. A model that "
      "ranks pitches perfectly (high AUC) but says 60% when the truth is 40% will recommend "
      "challenges that lose value — and AUC cannot see that error, because AUC only checks "
      "the ORDER of the predictions, not their size. So the models are graded on Brier score, "
      "log loss, calibration intercept and slope, the reliability diagram and ECE first; "
      "ROC-AUC is reported last, as a secondary check.\n")

    # 2 ----------------------------------------------------------------
    w("## 2. Data\n")
    rows = []
    for name, d in (("train", tr), ("validate", va)):
        rows.append(f"| {name} | {d['game_date'].min():%b %d} – {d['game_date'].max():%b %d} | "
                    f"{len(d):,} | {d['game_pk'].nunique():,} | {d['y'].mean():.1%} |")
    w("| window | dates | challenges | games | overturned |")
    w("|---|---|---:|---:|---:|")
    w("\n".join(rows))
    w(f"| test | Sept → | {k['n_test']:,} | — | *sealed until the end of Week 8* |\n")
    gap = va["y"].mean() - k["p_ref"]
    if abs(gap) > 0.03:
        w(f"**The overturn rate moved {100 * gap:+.1f} points between training and August.** "
          "Part of any calibration-intercept error below is this shift, not the model. "
          "If the location mix of challenges didn't change, this is evidence teams' "
          "challenge behaviour changed — your adaptation question.\n")
    cc = k["call_check"]
    line_a, line_b = describe_call_check(cc)
    w(f"- **Which call is recorded on challenged pitches?** The {line_a}; {line_b}.")
    if cc["repair"]:
        w("  Statcast-style data can store the call *after* the ABS review. On an "
          "overturned pitch that makes the call look correct, which erases the distance "
          "signal and points \"challenges left\" at the wrong team. Every model below "
          "uses the umpire's original call, worked out from who challenged.")
    w(f"- Umpire and catcher accuracy come from {k['n_train_called']:,} **training-window** "
      f"called pitches only (league agreement with the ABS zone: {k['league']:.2%}), shrunk "
      "toward the league rate, and enter as percentage points above or below it.")
    lev_missing = float(tr["leverage_missing"].mean())
    w(f"- Leverage index was missing for {lev_missing:.1%} of training challenges; those were "
      "set to average leverage (log LI = 0).")
    folded = {kk: m.design.folded for kk, m in models.items() if m.design.folded}
    if folded:
        biggest = max(folded, key=lambda kk: len(models[kk].design.folded))
        w(f"- Rare categories (fewer than 30 training challenges) folded into the reference "
          f"level: {models[biggest].design.folded}.")
    dropped = sorted({c for m in models.values() for c in m.design.dropped})
    if dropped:
        w(f"- Dropped because they had no variation in training (nothing to estimate): "
          f"`{'`, `'.join(dropped)}`.")
    if k["missing_cols"]:
        w(f"- **Not in the cohort at all:** `{'`, `'.join(k['missing_cols'])}` — check the "
          "Week 5 join.")
    w("")

    # 3 ----------------------------------------------------------------
    w("## 3. The sign of the distance feature\n")
    w(f"Week 5's `zone_dist_in` is signed relative to the zone (+ outside, − inside). Used "
      f"as-is, its AUC for predicting an overturn is **{k['auc_raw']:.3f}** — barely better "
      f"than a coin flip — because a pitch outside the zone is likely overturned when it was "
      f"called a strike and almost never when it was called a ball, and pooling the two "
      f"cancels them out. Re-signed relative to the **call** (`dist_against_call`: + means "
      f"the pitch is on the wrong side of the edge for the call that was made), its AUC is "
      f"**{k['auc_res']:.3f}**. Every model below uses the re-signed version. "
      f"(Figure `w7_01_sign_trap.png`.)\n")

    # 4 ----------------------------------------------------------------
    w("## 4. Headline results — validation window\n")
    w("Lower is better for Brier, log loss and ECE. Ideal calibration intercept is 0, ideal "
      "slope is 1. Brackets are 95% intervals from resampling whole validation games. "
      "Params = coefficients (logistic) or effective degrees of freedom (GAM).\n")
    w("| model | params | Brier | Brier skill | log loss | cal. intercept | cal. slope | ECE (noise floor) | AUC |")
    w("|---|---:|---|---:|---|---|---|---|---|")
    for kk in ORDER:
        s, c = point[kk], ci[kk]
        w(f"| {L[kk][0]} | {n_params(models[kk]):.1f} | {fci(s['brier'], c['brier'], 4)} | "
          f"{f(s['brier_skill'], 3)} | {fci(s['log_loss'], c['log_loss'], 3)} | "
          f"{fci(s['cal_intercept'], c['cal_intercept'], 2, True)} | "
          f"{fci(s['cal_slope'], c['cal_slope'], 2)} | "
          f"{f(s['ece'], 3)} ({f(k['floor'][kk], 3)}) | {fci(s['auc'], c['auc'], 3)} |")
    w("")
    w("**Where decisions get made.** Most challenges are easy calls that every model gets "
      "right. The MDP's challenge-or-hold choice turns on the uncertain ones, so: among the "
      "validation challenges each model put between 20% and 80%, what did it predict on "
      "average, and how many were actually overturned?\n")
    w("| model | challenges in 20–80% | mean predicted | actually overturned [95%] |")
    w("|---|---:|---:|---|")
    for kk in ORDER[1:]:
        m_ = k["mid"][kk]
        if m_["n"]:
            w(f"| {L[kk][0]} | {m_['n']} | {pct(m_['mean_pred'])} | "
              f"{pct(m_['obs_rate'])} [{pct(m_['ci_lo'])}, {pct(m_['ci_hi'])}] |")
        else:
            w(f"| {L[kk][0]} | 0 | — | — |")
    w("")
    if k["small_va"]:
        w(f"> Only {len(va)} validation challenges: intervals are wide and small "
          "differences between models are not meaningful.\n")

    # 5 ----------------------------------------------------------------
    w("## 5. What each block adds\n")
    w("Two different questions, answered side by side. The **likelihood-ratio test** asks "
      "whether the bigger model fits the *training* months better than chance would allow. "
      "The **validation change** asks whether it predicts *August* better. A block can pass "
      "the first and fail the second — that's a pattern that didn't carry forward, and it's "
      "the second question that matters. Negative Δ = the bigger model is better.\n")
    w("| step | params added | LR test (train) | Δ Brier (validation) [95%] | Δ log loss [95%] | verdict |")
    w("|---|---:|---|---|---|---|")
    for _, r in k["steps"].iterrows():
        lrtxt = (f"χ²={r['lr_chi2']:.1f}, df={int(r['lr_df'])}, p={r['lr_p']:.2g}"
                 if np.isfinite(r["lr_p"]) else "not nested")
        pa = r["params_added"] if abs(r["params_added"]) >= 0.05 else 0.0
        w(f"| {r['step']} ({r['bigger']} vs {r['smaller']}) | {pa:+.1f} | {lrtxt} | "
          f"{f(r['d_brier'], 4, True)} [{f(r['d_brier_lo'], 4, True)}, {f(r['d_brier_hi'], 4, True)}] | "
          f"{f(r['d_logloss'], 4, True)} [{f(r['d_logloss_lo'], 4, True)}, {f(r['d_logloss_hi'], 4, True)}] | "
          f"{r['verdict']} |")
    m1m0 = k["diffs"][("M1", "M0")]["brier"]
    w(f"\nFor scale: distance alone (M1 vs M0) changes Brier by {m1m0[0]:+.4f}. "
      "Everything after that is fine-tuning.\n")

    # 6 ----------------------------------------------------------------
    w("## 6. The location model, in inches\n")
    w("For each edge and each direction of call: where the overturn chance crosses 50% "
      "(\"the coin-flip point\", in inches past our computed edge, + = on the wrong side of "
      "the call), and how many inches it takes to go from 25% to 75% (\"the blur\"). "
      "Everything else is held at typical values.\n")
    for kk in ("M2", "GAM-loc"):
        w(f"**{L[kk][0]}**\n")
        w("| edge | challenge | coin-flip point (in.) [95%] | 25→75% blur (in.) | chance right at our edge |")
        w("|---|---|---|---:|---:|")
        for _, r in k["loc_tab"][kk].iterrows():
            w(f"| {r['edge']} | {r['call']} | {f(r['p50'], 2, True)} [{f(r['p50_lo'], 2, True)}, "
              f"{f(r['p50_hi'], 2, True)}] | {f(r['width'], 2)} | {pct(r['p_at_edge'], 0)} |")
        if k["loc_tab"][kk][["p50", "p50_lo", "p50_hi"]].isna().any().any():
            w("\n— = the curve or its band never crosses 50% within ±10 inches: too little "
              "data at that edge to pin the point down.")
        w("")
    w("How to read the two rows for each edge (details in WEEK7.md):\n")
    w("- **Both coin-flip points below zero** → challenges succeed before the pitch even "
      "reaches our computed edge: the challengers know something the measured location "
      "doesn't (or the tracking puts the ball slightly differently from what ABS used).")
    w("- **The two points far apart, in opposite directions** → the ABS edge isn't where "
      "our height-based zone puts it. Half the gap is the size of the shift.\n")
    t = k["loc_tab"]["M2"]
    lines = []
    for e in t["edge"].unique():
        s_ = t[(t.edge == e) & (t.call_dir < 0)]["p50"].iloc[0]
        b_ = t[(t.edge == e) & (t.call_dir > 0)]["p50"].iloc[0]
        lines.append(f"| {e} | {f((s_ + b_) / 2, 2, True)} | {f((s_ - b_) / 2, 2, True)} |")
    w("Sign of the half-gap: **+** means ABS's zone reaches further out than ours at that "
      "edge (it calls strikes on pitches our zone calls balls); **−** means it stops short.\n")
    w("| edge (M2) | midpoint of the two coin-flip points | half the gap (strike − ball) |")
    w("|---|---:|---:|")
    w("\n".join(lines))
    gl = models["GAM-loc"]
    w(f"\n**GAM smoothness.** λ = {gl.lam:g}, chosen by REML (restricted maximum likelihood, "
      "as in R's mgcv) on the training months. Effective degrees of freedom per curve (1 = the data "
      "chose a straight line; more = real bends): " +
      ", ".join(f"{n.replace('s(dist_against_call | edge=', '').rstrip(')')}: {e:.1f}"
                for n, e in gl.edf.items()) + ".\n")

    # 7 ----------------------------------------------------------------
    w("## 7. Everything else, at a fixed location (M5)\n")
    w("Odds ratios from the full logistic model, per a meaningful unit. ABS only sees the "
      "ball, so **none of these can change how ABS rules a given pitch.** An effect here "
      "means the challenges that got made in that situation were different — a statement "
      "about who chose to challenge, not about the system. Read them as selection effects.\n")
    t5 = models["M5"].table().set_index("term")
    w("| term | per | odds ratio [95%] | p |")
    w("|---|---|---|---:|")
    for term in t5.index:
        if term == "(intercept)" or " | " in term or term.startswith("edge="):
            continue
        scale, per = term_unit(term, models["M5"].design)
        b, se = t5.loc[term, "coef"], t5.loc[term, "se"]
        orr = np.exp(scale * b)
        lo, hi = np.exp(scale * (b - 1.96 * se)), np.exp(scale * (b + 1.96 * se))
        w(f"| `{term}` | {per} | {orr:.2f} [{lo:.2f}, {hi:.2f}] | {t5.loc[term, 'p_value']:.2g} |")
    w("")

    # 8 ----------------------------------------------------------------
    w(f"## 8. Subgroup calibration ({L[k['mdp_key']][0]}, validation)\n")
    w("Overall calibration can hide a model that's too high for one kind of challenge and "
      "too low for another. Each row: what the model predicted on average for that slice, "
      "and what happened. A 95% interval that excludes the prediction is worth a look; with "
      "slices this small, most won't. Week 9 does this properly on more data.\n")
    w("| slice | n | predicted | observed [95%] | gap (pts) |")
    w("|---|---:|---:|---|---:|")
    for _, r in k["sub"].iterrows():
        flag = " ⚑" if (r["pred"] < r["lo"] or r["pred"] > r["hi"]) else ""
        w(f"| {r['group']}: {r['level']} | {r['n']} | {pct(r['pred'])} | {pct(r['obs'])} "
          f"[{pct(r['lo'])}, {pct(r['hi'])}] | {100 * (r['obs'] - r['pred']):+.1f}{flag} |")
    w("\n**By month** (training months are in-sample, so small gaps are expected there; a "
      "steady drift across months is the thing to look for — it would mean the same pitch "
      "location was overturned at a different rate as the season went on):\n")
    w("| month | window | n | predicted | observed [95%] |")
    w("|---|---|---:|---:|---|")
    for _, r in k["months"].iterrows():
        w(f"| {r['month']} | {r['window']} | {r['n']} | {pct(r['pred'])} | {pct(r['obs'])} "
          f"[{pct(r['lo'])}, {pct(r['hi'])}] |")
    w("")

    # 9 ----------------------------------------------------------------
    w("## 9. The baseline to beat\n")
    best, chosen = k["best_key"], k["chosen_key"]
    w("**Rule (fixed before looking):** take the model with the lowest validation Brier; "
      "then, going from simplest to most complex, choose the first model whose paired 95% "
      "interval for (its Brier − the best Brier) reaches zero. A gain August can't confirm "
      "doesn't count, and ties go to the simpler model.\n")
    w("| model | Brier − best [95%] | indistinguishable from best? |")
    w("|---|---|---|")
    for _, r in k["rule_rows"].iterrows():
        if r["model"] == best:
            w(f"| {L[r['model']][0]} | — (this is the best) | — |")
            continue
        w(f"| {L[r['model']][0]} | {f(r['d_vs_best'], 4, True)} [{f(r['lo'], 4, True)}, "
          f"{f(r['hi'], 4, True)}] | {'yes' if r['tie'] else 'no'} |")
    sc, cc = point[chosen], ci[chosen]
    w(f"\n### Baseline to beat: **{L[chosen][0]}**\n")
    w(f"Validation Brier {fci(sc['brier'], cc['brier'], 4)}, log loss "
      f"{fci(sc['log_loss'], cc['log_loss'], 3)}, calibration intercept "
      f"{fci(sc['cal_intercept'], cc['cal_intercept'], 2, True)}, slope "
      f"{fci(sc['cal_slope'], cc['cal_slope'], 2)}, AUC {f(sc['auc'], 3)}. "
      f"(Lowest Brier overall: {L[best][0]}.)\n")
    cal_ok = (cc["cal_intercept"][0] <= 0 <= cc["cal_intercept"][1]) and \
             (cc["cal_slope"][0] <= 1 <= cc["cal_slope"][1])
    if cal_ok:
        w("Its calibration intercept interval contains 0 and its slope interval contains 1: "
          "no detectable miscalibration on August.\n")
    else:
        w("**Its calibration is detectably off on August** (intercept interval excludes 0 or "
          "slope interval excludes 1). It's still the baseline to beat; Week 9's "
          "recalibration step exists to fix exactly this.\n")
    w(f"**A Week 8 model beats it** only if its paired-bootstrap 95% interval for "
      f"(its Brier − {chosen}'s Brier) lies entirely below zero on the same validation "
      "games, and its calibration intercept and slope are no worse.\n")
    w(f"**Location-only candidate for the MDP's p(s): {L[k['mdp_key']][0]}** (same rule, "
      "restricted to models that use only what ABS sees). Whether p(s) should also use the "
      "situation/pitch/people blocks is a decision for you and your supervisor — see "
      "WEEK7.md, \"Location-only or everything?\".\n")
    if k["leak_flag"]:
        w("> ⚠ **Leak check failed:** adding the non-location blocks raised AUC by more than "
          "0.02. Since ABS only sees the ball, that's implausible — look for a feature that "
          "encodes the challenge outcome before trusting anything above.\n")

    # 10 ---------------------------------------------------------------
    w("## 10. Most confidently wrong validation predictions — check by hand\n")
    w("The ten validation challenges M2 · location was most surprised by. M2 is used here "
      "on purpose: a straight line in log-odds is the model most shocked by a result far "
      "from the edge, so it makes data problems stand out (a flexible GAM can learn a small "
      "\"floor\" from a few bad records and hide them). Very confident misses are usually "
      "data problems — a challenge linked to the wrong pitch, a wrong height, a mislabelled "
      "outcome — and they are what drags log loss up. Look each one up on Baseball Savant.\n")
    w("| game_pk | date | inning | count | challenger | call | edge | in. against call | p(overturn) | result |")
    w("|---|---|---:|---|---|---|---|---:|---:|---|")
    for _, r in k["wrong"].iterrows():
        w(f"| {r['game_pk']} | {r['date']} | {r['inning']} | {r['count']} | {r['challenger']} | "
          f"{r['call']} | {r['edge']} | {r['in_against_call']:+.2f} | {r['p_overturn']:.3f} | "
          f"{r['result']} |")
    w("")

    # 11 ---------------------------------------------------------------
    notes = [f"- **{L[kk][0]}**: {m.note}" for kk, m in models.items() if m.note]
    w("## 11. Fitting notes\n")
    w("\n".join(notes) if notes else "- Every model converged without help.")
    w("- Intervals reflect how much each score would move with a different set of August "
      "games. They do not include the uncertainty from re-fitting the models.")
    w("- The test window was not read. It is used once, at the end of Week 8.\n")

    # 12 ---------------------------------------------------------------
    w("## 12. Figures (`figures/week7/`)\n")
    for name, desc in FIGURE_LIST:
        w(f"- `{name}` — {desc}")
    w("")
    return "\n".join(out)


FIGURE_LIST = [
    ("w7_01_sign_trap.png", "why the distance feature has to be re-signed relative to the call"),
    ("w7_02_reliability_baselines.png", "reliability diagrams for the two location-only baselines"),
    ("w7_03_reliability_all.png", "reliability diagram for every model, side by side"),
    ("w7_04_block_contributions.png", "what each block adds on August, with 95% intervals"),
    ("w7_05_distance_curves.png", "overturn chance vs inches past each edge — GAM curve vs logistic line"),
    ("w7_06_odds_ratios.png", "M5's odds ratios: location (what ABS sees) vs everything else"),
    ("w7_07_subgroup_calibration.png", "observed minus predicted, by slice and by month"),
]

NICE = {"pitch_group": "pitch", "stand": "batter hand", "p_throws": "pitcher hand", "edge": "edge"}


def term_unit(term, design):
    """(scale, 'per ...' label) to turn a coefficient into a readable odds ratio."""
    if " | " in term:
        num, rest = term.split(" | ")
        level = rest.split("=", 1)[1]
        if num == "dist_against_call":
            return 1.0, f"per inch past the {level} edge"
        if num == "call_dir":
            return 2.0, f"fielder vs batter challenge, {level} edge"
    if "=" in term:
        col, level = term.split("=", 1)
        ref = design.levels.get(col, ["?"])[0]
        if col == "edge":
            return 1.0, f"{level} edge vs {ref}, right at the edge"
        return 1.0, f"{NICE.get(col, col)} {level} (vs {ref})"
    scale, per = UNITS.get(term, (1.0, "per unit"))
    return float(scale), per


if __name__ == "__main__":
    main()
