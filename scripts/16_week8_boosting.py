#!/usr/bin/env python3
"""
16_week8_boosting.py — Week 8. Try to beat the baseline with gradient boosting.

The plan: "Beat the baseline with CatBoost and LightGBM — or show that you
can't, which is also a legitimate result." This script does both halves
honestly:

    0. PRE-FLIGHT   which call was recorded on challenged pitches (and repair
                    it), the test window stays sealed
    1. BASELINES    refit the whole Week 7 ladder on the (corrected) data and
                    re-apply the baseline-to-beat rule
    2. TUNE         CatBoost and LightGBM, each on two feature sets
                    ("location" = only what ABS sees, "full" = everything incl.
                    umpire / catcher / pitcher / batter IDs as native
                    categoricals). Seeded random search, every setting logged.
                    Settings are compared on AUGUST; the number of trees is
                    chosen inside the training window, so August never fits
                    anything.
    3. COMPARE      all four families x both feature sets on identical rows
                    and identical metrics, with game-level paired bootstrap
                    intervals — one table
    4. DECIDE       does any boosted model RELIABLY beat the baseline? which
                    models go forward to Week 9?
    5. EXPLAIN      SHAP values for the best boosted model in each set, plus
                    the plan's leak check: if an ID outranks location, suspect
                    a leak

    python scripts/16_week8_boosting.py
    python scripts/16_week8_boosting.py --trials 8 --boot 200     # quicker while iterating
    python scripts/16_week8_boosting.py --backends catboost       # just one library

If CatBoost or LightGBM isn't installed, the script says so and uses
scikit-learn's HistGradientBoosting in its place. Run
`python tests/test_boosting_backends.py` first — it checks both libraries work
on your machine in 30 seconds.

Reads   data/processed/cohort_2026.parquet, config/cohort.json
Writes  docs/WEEK8_REPORT.md                the report, numbers filled in
        docs/week8_tuning_log.csv           every setting tried, with its scores
        docs/week8_comparison.csv           every metric, every model, 95% CIs
        docs/week8_vs_baseline.csv          paired differences vs the baseline
        docs/week8_importance_*.csv         SHAP / permutation importances
        figures/week8/*.png                 6 figures, white background
        models/week8/                       every fitted model + selection.json
        data/processed/week8_validate_predictions.csv

The test window is NOT read here. Week 9's --final-test opens it once, after
calibration — the right order when Weeks 8 and 9 are due together.
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "eda"))

from src import abs_metrics as M                                          # noqa: E402
from src.abs_baselines import (CANDIDATES, LOCATION_ONLY, MODEL_SPECS,    # noqa: E402
                               ORDER, calibration_ok, choose, fit_model, n_params)
from src.abs_boosting import (BACKENDS, INSTALL_HINT, Booster, available,  # noqa: E402
                              sample_configs)
from src.abs_features import (BOOST_SETS, FEATURE_BLOCK, FEATURE_LABEL,  # noqa: E402
                              ID_FEATURES, describe_call_check)
from src.abs_pipeline import es_mask_by_date, load_challenges           # noqa: E402

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
CONFIG = Path(os.environ.get("ABS_CONFIG_DIR", "config"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "week8"
MODELS = Path(os.environ.get("ABS_MODEL_DIR", "models")) / "week8"

FAMILY_LABEL = {"catboost": "CatBoost", "lightgbm": "LightGBM", "hgb": "HistGB"}
SET_SHORT = {"location": "loc", "full": "full"}
SET_LABEL = {"location": "location", "full": "everything"}
# The plan's "four models", per feature set: logistic, GAM, CatBoost, LightGBM.
W7_PEERS = {"location": ("M2", "GAM-loc"), "full": ("M5", "GAM-full")}
# Week 7 models that use more than location, simplest first.
FULL_ONLY = [k for k in CANDIDATES if k not in LOCATION_ONLY]


def boost_key(backend, fset):
    return f"{backend}-{SET_SHORT[fset]}"


def comparison_rows(models, family, fsets, baseline):
    """The plan's 'four models' for each feature set — logistic, GAM and the
    boosted ones — plus the baseline if it isn't already among them (e.g. M3)."""
    show = []
    for fs in ("location", "full"):
        peers = [x for x in W7_PEERS[fs] if x in models]
        if baseline in models and fsets.get(baseline) == fs and baseline not in W7_PEERS[fs]:
            peers = [baseline] + peers
        show += peers + [x for x in models if family[x] in BACKENDS and fsets[x] == fs]
    return show


def boost_label(backend, fset):
    return f"{FAMILY_LABEL[backend]} · {SET_LABEL[fset]}"


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


# =========================================================== tuning
def tune(backend, fset, tr, va, es, trials, seed, max_rounds, log):
    """Seeded random search. Every setting is logged; the best by August log
    loss is kept (log loss is the boosting objective, so it's the smoothest
    signal to tune on; Brier stays the yardstick for comparing families)."""
    best, best_ll = None, np.inf
    cfgs = sample_configs(backend, trials, seed)
    for i, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        try:
            b = Booster(backend, **BOOST_SETS[fset], params=cfg, seed=seed,
                        max_rounds=max_rounds).fit(tr, tr["y"], es)
        except Exception as e:                            # log it, keep searching
            log.append({"backend": backend, "feature_set": fset, "trial": i,
                        "params": json.dumps(cfg), "error": f"{type(e).__name__}: {e}"})
            print(f"    trial {i:>2}/{len(cfgs)}  FAILED  {type(e).__name__}: {e}")
            continue
        p = b.predict(va)
        ll, br = M.log_loss(va["y"], p), M.brier(va["y"], p)
        log.append({"backend": backend, "feature_set": fset, "trial": i,
                    "params": json.dumps(cfg), "trees": b.best_iteration,
                    "early_stop_logloss": b.es_logloss, "august_logloss": ll,
                    "august_brier": br, "seconds": time.time() - t0, "error": ""})
        mark = ""
        if ll < best_ll:
            best, best_ll, mark = b, ll, "  <- best so far"
        print(f"    trial {i:>2}/{len(cfgs)}  trees {b.best_iteration:>5}  "
              f"August log loss {ll:.4f}  Brier {br:.4f}  ({time.time() - t0:.1f}s){mark}")
    return best


def permutation_importance(model, va, seed=2026, n_repeats=5):
    """Rise in August log loss when one feature's values are shuffled — the
    importance measure for HistGradientBoosting, which has no native SHAP."""
    rng = np.random.default_rng(seed)
    base = M.log_loss(va["y"], model.predict(va))
    rows = []
    for c in model.features:
        rises = []
        for _ in range(n_repeats):
            sh = va.copy()
            sh[c] = sh[c].to_numpy()[rng.permutation(len(sh))]
            rises.append(M.log_loss(va["y"], model.predict(sh)) - base)
        rows.append({"feature": c, "importance": float(np.mean(rises)),
                     "sd": float(np.std(rises))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def importance(model, va):
    """SHAP when the library has it, permutation importance otherwise."""
    sh = model.shap(va)
    if sh is None:
        return permutation_importance(model, va), "permutation", None
    vals, base = sh
    imp = vals.abs().mean().sort_values(ascending=False)
    df = pd.DataFrame({"feature": imp.index, "importance": imp.to_numpy()})
    return df, "shap", vals


def leak_check(imp):
    """The plan's rule: if an identity feature outranks pitch location,
    something is leaking (ABS cannot see who anyone is)."""
    ranks = {f_: i for i, f_ in enumerate(imp["feature"])}
    loc = [ranks[x] for x in ranks if FEATURE_BLOCK.get(x) == "location"]
    ids = [ranks[x] for x in ranks if x in ID_FEATURES]
    if not loc or not ids:
        return False, ""
    top_loc, top_id = min(loc), min(ids)
    name = lambda i: FEATURE_LABEL.get(imp["feature"][i], imp["feature"][i])   # noqa: E731
    if top_id < top_loc:
        return True, (f"`{name(top_id)}` ranks #{top_id + 1}, above the best location "
                      f"feature (`{name(top_loc)}`, #{top_loc + 1})")
    return False, (f"best location feature `{name(top_loc)}` ranks #{top_loc + 1}; "
                   f"best ID `{name(top_id)}` #{top_id + 1}")


# =========================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--trials", type=int, default=24, help="settings tried per library per feature set")
    ap.add_argument("--backends", nargs="*", default=None,
                    help=f"subset of {BACKENDS} (default: catboost + lightgbm if installed)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    warnings.filterwarnings("ignore")

    # ---------------------------------------------------------- 0. pre-flight
    banner("0. Pre-flight")
    data = load_challenges(DATA_DIR, CONFIG)
    tr, va, n_test = data["tr"], data["va"], data["n_test"]
    for line in describe_call_check(data["call_check"]):
        print(f"  {line}")
    print(f"  challenges: train {len(tr):,} · August {len(va):,} · September {n_test:,} (sealed)")
    if len(tr) < 300 or len(va) < 50:
        raise SystemExit("  Too few challenges to tune boosted models. Check the Week 6 split.")
    p_ref = float(tr["y"].mean())

    if args.backends:
        backends = []
        for b in args.backends:
            ok, ver = available(b)
            if not ok:
                raise SystemExit(f"  {b} is not installed: {ver}\n  install: {INSTALL_HINT[b]}")
            backends.append(b)
    else:
        backends = [b for b in ("catboost", "lightgbm") if available(b)[0]]
        missing = [b for b in ("catboost", "lightgbm") if b not in backends]
        for b in missing:
            print(f"  NOTE: {b} is not installed — install with: {INSTALL_HINT[b]}")
        if len(backends) < 2:
            backends.append("hgb")
            print("  Using scikit-learn's HistGradientBoosting in its place "
                  "(same algorithm family; recorded in the report).")
    versions = {b: available(b)[1] for b in backends}
    print("  boosting libraries: " + ", ".join(f"{b} {v}" for b, v in versions.items()))

    # ---------------------------------------------------------- 1. baselines
    banner("1. Refit the Week 7 ladder on the corrected data")
    models, labels, family, fsets = {}, {}, {}, {}
    for k in ORDER:
        models[k] = fit_model(k, tr)
        labels[k] = MODEL_SPECS[k][0]
        family[k] = MODEL_SPECS[k][1]
        fsets[k] = "location" if k in ("M0", "M1", "M2", "GAM-loc") else "full"
    prior = None
    spec7 = Path(os.environ.get("ABS_MODEL_DIR", "models")) / "week7_feature_spec.json"
    if spec7.exists():
        try:
            prior = json.loads(spec7.read_text()).get("baseline_to_beat")
        except Exception:
            prior = None

    # ---------------------------------------------------------- 2. tune
    banner(f"2. Tune {', '.join(FAMILY_LABEL[b] for b in backends)} "
           f"({args.trials} settings each, per feature set)")
    es = es_mask_by_date(tr)
    print(f"  trees chosen on the last {es.mean():.0%} of training games "
          f"({tr.loc[es, 'game_date'].min():%b %d} – {tr['game_date'].max():%b %d}); "
          f"settings compared on August")
    log = []
    for b in backends:
        for fs in ("location", "full"):
            k = boost_key(b, fs)
            print(f"\n  {boost_label(b, fs)}")
            best = tune(b, fs, tr, va, es, args.trials, args.seed, args.max_rounds, log)
            if best is None:
                print("    every setting failed — skipped")
                continue
            models[k], labels[k], family[k], fsets[k] = best, boost_label(b, fs), b, fs
    boosted = [k for k in models if family[k] in BACKENDS]
    log = pd.DataFrame(log)

    # ---------------------------------------------------------- 3. compare
    banner("3. Score everything on August")
    keys = list(models)
    preds = {k: np.clip(models[k].predict(va), 1e-6, 1 - 1e-6) for k in keys}
    point = {k: M.metric_suite(va["y"], p, p_ref) for k, p in preds.items()}
    floor = {k: M.ece_noise_floor(p, n_sim=200, seed=args.seed) for k, p in preds.items()}
    mid = {k: M.midrange_check(va["y"], p) for k, p in preds.items()}
    print(f"  bootstrapping {args.boot} resamples of {va['game_pk'].nunique()} August games "
          f"for {len(keys)} models ...")
    t1 = time.time()
    ci, diffs = M.cluster_bootstrap(va["y"].to_numpy(), preds, va["game_pk"].to_numpy(),
                                    p_ref, B=args.boot, seed=args.seed)
    print(f"  done in {time.time() - t1:.0f}s")

    # ---------------------------------------------------------- 4. decide
    banner("4. Decide")
    best7, baseline, rule7 = choose(CANDIDATES, point, diffs)
    loc_boost = [k for k in boosted if fsets[k] == "location"]
    full_boost = [k for k in boosted if fsets[k] == "full"]
    order_all = CANDIDATES + loc_boost + full_boost
    best_all, winner, rule_all = choose(order_all, point, diffs)
    best_loc, cand_loc, rule_loc = choose(LOCATION_ONLY + loc_boost, point, diffs)
    best_full, cand_full, rule_full = choose(FULL_ONLY + full_boost, point, diffs)
    vs = []
    for k in boosted:
        d_b = diffs[(k, baseline)]["brier"]
        d_l = diffs[(k, baseline)]["log_loss"]
        cal = calibration_ok(ci[k], point[baseline], point[k])
        vs.append({"model": k, "label": labels[k], "d_brier": d_b[0], "d_brier_lo": d_b[1],
                   "d_brier_hi": d_b[2], "d_logloss": d_l[0], "d_logloss_lo": d_l[1],
                   "d_logloss_hi": d_l[2], "calibration_ok": cal,
                   "beats_baseline": bool(d_b[2] < 0 and cal)})
    vs = pd.DataFrame(vs)
    print(f"  baseline to beat (Week 7 rule, corrected data): {labels[baseline]}"
          + (f"   [Week 7 said: {MODEL_SPECS.get(prior, (prior,))[0]}]" if prior and prior != baseline else ""))
    for r in vs.itertuples():
        print(f"  {r.label:<26} ΔBrier vs baseline {r.d_brier:+.4f} [{r.d_brier_lo:+.4f}, "
              f"{r.d_brier_hi:+.4f}]  calibration {'ok' if r.calibration_ok else 'WORSE'}  -> "
              f"{'BEATS the baseline' if r.beats_baseline else 'does not beat it'}")
    print(f"  overall winner (simplest within noise of the best): {labels[winner]}")
    print(f"  location-only candidate for p(s): {labels[cand_loc]}")
    print(f"  candidate that uses everything:   {labels[cand_full]}")

    # ---------------------------------------------------------- 5. explain
    banner("5. Explain the best boosted models")
    imps = {}
    for fs, ks in (("location", loc_boost), ("full", full_boost)):
        if not ks:
            continue
        kbest = min(ks, key=lambda k: point[k]["brier"])
        imp, kind, shap_vals = importance(models[kbest], va)
        leak, leak_txt = leak_check(imp) if fs == "full" else (False, "")
        imps[fs] = {"key": kbest, "imp": imp, "kind": kind, "shap": shap_vals,
                    "leak": leak, "leak_txt": leak_txt}
        top = ", ".join(f"{FEATURE_LABEL.get(x, x)}" for x in imp["feature"][:5])
        print(f"  {labels[kbest]} ({kind}): top features — {top}")
        if fs == "full":
            print(f"  leak check: {'SUSPICIOUS — ' if leak else 'ok — '}{leak_txt}")

    loc_all = [x for x in models if fsets[x] == "location" and x != "M0"]
    full_all = [x for x in models if fsets[x] == "full"]
    auc_gap = (max(point[x]["auc"] for x in full_all) - max(point[x]["auc"] for x in loc_all)
               if loc_all and full_all else float("nan"))
    leak = {"flagged": bool(imps.get("full", {}).get("leak", False)),
            "detail": imps.get("full", {}).get("leak_txt", ""),
            "auc_gap_full_minus_location": auc_gap,
            "auc_gap_warning": bool(auc_gap > 0.02)}
    print(f"  AUC, everything minus location only: {auc_gap:+.3f}"
          + ("   <- more than 0.02: check for a leak" if auc_gap > 0.02 else ""))

    # ---------------------------------------------------------- 6. write
    banner("6. Write outputs")
    DOCS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    log.to_csv(DOCS / "week8_tuning_log.csv", index=False)
    rows = []
    for k in keys:
        for m_ in M.METRICS:
            rows.append({"model": k, "label": labels[k], "family": family[k],
                         "feature_set": fsets[k], "metric": m_, "value": point[k][m_],
                         "ci_lo": ci[k][m_][0], "ci_hi": ci[k][m_][1]})
        rows.append({"model": k, "label": labels[k], "family": family[k], "feature_set": fsets[k],
                     "metric": "ece_noise_floor", "value": floor[k], "ci_lo": np.nan, "ci_hi": np.nan})
    pd.DataFrame(rows).to_csv(DOCS / "week8_comparison.csv", index=False)
    vs.to_csv(DOCS / "week8_vs_baseline.csv", index=False)
    pd.DataFrame([{"model": k, "label": labels[k],
                   **{f"{m_}_{t}": v for m_ in ("brier", "log_loss")
                      for t, v in zip(("d", "lo", "hi"), diffs[(k, baseline)][m_])}}
                  for k in keys if k != baseline]).to_csv(
        DOCS / "week8_paired_vs_baseline.csv", index=False)
    for fs, d in imps.items():
        d["imp"].assign(label=d["imp"]["feature"].map(FEATURE_LABEL),
                        block=d["imp"]["feature"].map(FEATURE_BLOCK)).to_csv(
            DOCS / f"week8_importance_{fs}.csv", index=False)

    out = va[["game_pk", "game_date", "at_bat_number", "pitch_number", "y"]].copy()
    for k in keys:
        out[f"p_{k}"] = preds[k]
    pred_path = DATA_DIR / "processed" / "week8_validate_predictions.csv"
    out.to_csv(pred_path, index=False)

    saved = {}
    for k in keys:
        if family[k] in BACKENDS:
            models[k].save(MODELS / k)
            saved[k] = str(MODELS / k)
        else:
            with open(MODELS / f"{k}.pkl", "wb") as fh:
                pickle.dump(models[k], fh)
            saved[k] = str(MODELS / f"{k}.pkl")
    selection = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "cohort_fingerprint": data["fingerprint"],
        "call_check": data["call_check"],
        "baseline_to_beat": baseline, "week7_reported_baseline": prior,
        "winner_all": winner, "candidate_location": cand_loc, "candidate_full": cand_full,
        "boosting_versions": versions, "leak_check": leak,
        "importance_models": {fs: d["key"] for fs, d in imps.items()},
        "importance_kind": {fs: d["kind"] for fs, d in imps.items()},
        "models": {k: {"label": labels[k], "family": family[k], "feature_set": fsets[k],
                       "path": saved[k],
                       "params": (models[k].params if family[k] in BACKENDS else None),
                       "trees": (models[k].best_iteration if family[k] in BACKENDS else None),
                       "monotone": (models[k].monotone if family[k] in BACKENDS else None),
                       "august_brier": point[k]["brier"], "august_logloss": point[k]["log_loss"]}
                   for k in keys},
    }
    (MODELS / "selection.json").write_text(json.dumps(selection, indent=2, default=str))

    report = build_report(data=data, tr=tr, va=va, n_test=n_test, models=models, labels=labels,
                          family=family, fsets=fsets, point=point, ci=ci, diffs=diffs,
                          floor=floor, mid=mid, baseline=baseline, prior=prior, winner=winner,
                          cand_loc=cand_loc, cand_full=cand_full, rule_all=rule_all,
                          rule_loc=rule_loc, rule_full=rule_full, best_loc=best_loc,
                          best_full=best_full, vs=vs, log=log, imps=imps,
                          backends=backends, versions=versions, es=es, args=args, best_all=best_all)
    (DOCS / "WEEK8_REPORT.md").write_text(report)
    for p_ in (DOCS / "WEEK8_REPORT.md", DOCS / "week8_tuning_log.csv",
               DOCS / "week8_comparison.csv", DOCS / "week8_vs_baseline.csv", pred_path,
               MODELS / "selection.json"):
        print(f"  {p_}")

    if not args.no_figures:
        banner("7. Figures")
        import week8_figures as W8
        W8.make_all(FIGS, DATA_DIR, tr=tr, va=va, models=models, labels=labels, family=family,
                    fsets=fsets, preds=preds, point=point, ci=ci, diffs=diffs, log=log,
                    baseline=baseline, imps=imps, boosted=boosted, floor=floor, vs=vs,
                    show=comparison_rows(models, family, fsets, baseline), peers=W7_PEERS)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Read {DOCS / 'WEEK8_REPORT.md'} "
          f"with WEEK8.md open beside it.")


# =========================================================== report
def build_report(**k):
    va, point, ci, labels, models = k["va"], k["point"], k["ci"], k["labels"], k["models"]
    fsets, family, diffs = k["fsets"], k["family"], k["diffs"]
    baseline, winner, cand_loc = k["baseline"], k["winner"], k["cand_loc"]
    demo = "demo" in str(DATA_DIR).lower()
    out = []
    w = out.append
    w("# Week 8 — Gradient Boosting Report\n")
    w(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/16_week8_boosting.py` · cohort "
      f"`{k['data']['fingerprint']}` · {k['args'].trials} settings per library per feature set · "
      f"{k['args'].boot} game-level bootstrap resamples\n")
    if demo:
        w("> **DEMO DATA — every number below comes from the synthetic fixture.**\n")
    w("> Read this next to `WEEK8.md`, which explains every section.\n")

    # 0 ---------------------------------------------------------------
    w("## 0. Pre-flight\n")
    a, b = describe_call_check(k["data"]["call_check"])
    w(f"- **Recorded call on challenged pitches:** {a}; {b}.")
    if k["data"]["call_check"]["repair"]:
        w("  Every model in this report — including the refit Week 7 ladder — uses the umpire's "
          "original call. Week 7 results computed before this repair are superseded by section 1.")
    w(f"- Challenges: train {len(k['tr']):,} · August {len(va):,} · September {k['n_test']:,} "
      "(**sealed** — opened once, by Week 9's `--final-test`).")
    w(f"- Boosting libraries: " + ", ".join(f"{FAMILY_LABEL[b_]} {v}" for b_, v in k["versions"].items()) + ".")
    if "hgb" in k["backends"]:
        w("  **HistGradientBoosting stands in for a library that isn't installed** — install "
          "CatBoost and LightGBM and re-run for the plan's comparison.")
    es = k["es"]
    w(f"- Number of trees chosen on the last {es.mean():.0%} of training games "
      f"({k['tr'].loc[es, 'game_date'].min():%b %d} – {k['tr']['game_date'].max():%b %d}), then "
      "every model refit on the whole training window. August only compares settings.\n")

    # 1 ---------------------------------------------------------------
    w("## 1. The Week 7 ladder, refit\n")
    w("| model | params | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | AUC |")
    w("|---|---:|---|---:|---|---|---:|")
    for kk in ORDER:
        s, c = point[kk], ci[kk]
        star = " ★" if kk == baseline else ""
        w(f"| {labels[kk]}{star} | {n_params(models[kk]):.1f} | {fci(s['brier'], c['brier'], 4)} | "
          f"{f(s['log_loss'], 3)} | {fci(s['cal_intercept'], c['cal_intercept'], 2, True)} | "
          f"{fci(s['cal_slope'], c['cal_slope'], 2)} | {f(s['auc'], 3)} |")
    w(f"\n★ **Baseline to beat: {labels[baseline]}** (the Week 7 rule: simplest model whose "
      "Brier can't be told apart from the best).")
    if k["prior"] and k["prior"] != baseline:
        w(f"Week 7 originally chose **{MODEL_SPECS.get(k['prior'], (k['prior'],))[0]}**; refit on "
          f"the corrected data the rule now chooses **{labels[baseline]}**. The baseline every "
          "boosted model is measured against below is the corrected one.")
    w("")

    # 2 ---------------------------------------------------------------
    w("## 2. Tuning\n")
    lg = k["log"]
    w("| library · feature set | settings tried | best August log loss | median | worst | "
      "best setting | trees |")
    w("|---|---:|---:|---:|---:|---|---:|")
    for kk in [x for x in models if family[x] in BACKENDS]:
        sub = lg[(lg["backend"] == family[kk]) & (lg["feature_set"] == fsets[kk])]
        ok = sub[sub["error"].fillna("") == ""] if "error" in sub else sub
        params = ", ".join(f"{p}={v}" for p, v in models[kk].params.items())
        w(f"| {labels[kk]} | {len(ok)} | {ok['august_logloss'].min():.4f} | "
          f"{ok['august_logloss'].median():.4f} | {ok['august_logloss'].max():.4f} | {params} | "
          f"{models[kk].best_iteration} |")
    dropped = [labels[x] for x in models if family[x] in BACKENDS and not models[x].monotone]
    if dropped:
        w(f"\n> ⚠ The monotone constraint on distance was **rejected by the library** for: "
          f"{', '.join(dropped)}. Those models were fitted without it (see the console log). "
          "Upgrading the library usually fixes this.")
    n_err = int((lg["error"].fillna("") != "").sum()) if len(lg) and "error" in lg else 0
    w(f"\nEvery setting is in `docs/week8_tuning_log.csv`" +
      (f" ({n_err} failed and are logged with the error)." if n_err else ".") +
      " The search is seeded: the same `--trials` gives the same settings every run. "
      "**Best − median** is what tuning bought; a small gap means the defaults were already "
      "fine and the data, not the settings, limits the model. Choosing the best of many "
      "settings on August flatters August a little — September (Week 9) is the clean check.\n")

    # 3 ---------------------------------------------------------------
    w("## 3. One comparison table (August)\n")
    w("Same rows, same metrics, same bootstrap for every model. Lower is better for Brier, log "
      "loss and ECE; ideal intercept 0, slope 1.\n")
    w("| feature set | model | Brier [95%] | Brier skill | log loss [95%] | cal. intercept | "
      "cal. slope | ECE (floor) | AUC |")
    w("|---|---|---|---:|---|---|---|---|---:|")
    for kk in comparison_rows(models, family, fsets, baseline):
        s, c = point[kk], ci[kk]
        star = " ★" if kk == baseline else ""
        w(f"| {SET_LABEL[fsets[kk]]} | {labels[kk]}{star} | {fci(s['brier'], c['brier'], 4)} | "
          f"{f(s['brier_skill'], 3)} | {fci(s['log_loss'], c['log_loss'], 3)} | "
          f"{fci(s['cal_intercept'], c['cal_intercept'], 2, True)} | "
          f"{fci(s['cal_slope'], c['cal_slope'], 2)} | {f(s['ece'], 3)} ({f(k['floor'][kk], 3)}) | "
          f"{f(s['auc'], 3)} |")
    w("")

    # 4 ---------------------------------------------------------------
    w("## 4. Did boosting beat the baseline?\n")
    w(f"A boosted model **beats** {labels[baseline]} only if its paired 95% interval for "
      "(its Brier − baseline Brier) lies entirely below zero on the same August games **and** "
      "its calibration is no worse (rule fixed in Week 7).\n")
    w("| model | Δ Brier vs baseline [95%] | Δ log loss [95%] | calibration | verdict |")
    w("|---|---|---|---|---|")
    for r in k["vs"].itertuples():
        w(f"| {r.label} | {f(r.d_brier, 4, True)} [{f(r.d_brier_lo, 4, True)}, "
          f"{f(r.d_brier_hi, 4, True)}] | {f(r.d_logloss, 4, True)} [{f(r.d_logloss_lo, 4, True)}, "
          f"{f(r.d_logloss_hi, 4, True)}] | {'ok' if r.calibration_ok else 'worse'} | "
          f"{'**beats the baseline**' if r.beats_baseline else 'does not beat it'} |")
    nb = int(k["vs"]["beats_baseline"].sum()) if len(k["vs"]) else 0
    w("")
    if nb:
        w(f"**{nb} boosted model(s) reliably beat the baseline.**")
    else:
        w("**No boosted model reliably beat the baseline.** That is a legitimate result: the "
          "simpler, interpretable model is as good as the flexible one on new data, so it keeps "
          "the job.")
    w(f"\nOverall winner by the same rule over every model (simplest within noise of the best "
      f"{labels[k['best_all']]}): **{labels[winner]}**.\n")
    w("| model | Brier − best [95%] | within noise of the best? |")
    w("|---|---|---|")
    for r in k["rule_all"].itertuples():
        if r.model == k["best_all"]:
            w(f"| {labels[r.model]} | — (this is the best) | — |")
        else:
            w(f"| {labels[r.model]} | {f(r.d_vs_best, 4, True)} [{f(r.lo, 4, True)}, "
              f"{f(r.hi, 4, True)}] | {'yes' if r.tie else 'no'} |")
    w("")

    # 5 ---------------------------------------------------------------
    w("## 5. Location only, or everything?\n")
    loc_all = [x for x in models if fsets[x] == "location" and x != "M0"]
    full_all = [x for x in models if fsets[x] == "full"]
    bl = max(loc_all, key=lambda x: point[x]["auc"])
    bf = max(full_all, key=lambda x: point[x]["auc"])
    gap = point[bf]["auc"] - point[bl]["auc"]
    w(f"Highest August AUC using location only: **{f(point[bl]['auc'], 3)}** ({labels[bl]}); "
      f"using everything: **{f(point[bf]['auc'], 3)}** ({labels[bf]}). Gap: {f(gap, 3, True)}.\n")
    if gap > 0.02:
        w("> ⚠ **The non-location features add more than 0.02 AUC.** ABS only sees the ball, so "
          "the extra features can only help by describing *which* challenges teams choose — or by "
          "leaking the outcome. Check section 6's leak test and the feature list before believing "
          "it.\n")
    else:
        w("The plan's alarm (\"AUC above ~0.95 → hunt for a leak\") needs adapting here: ABS is a "
          "machine applying a fixed rectangle, so location alone *should* separate overturns very "
          "well, and a high AUC from location is physics, not a leak. The leak signal is the "
          "**gap**: features ABS can't see adding a lot on top of location. That gap is small.\n")
    for name, rule, cand, best in (("location-only", k["rule_loc"], cand_loc, k["best_loc"]),
                                   ("everything", k["rule_full"], k["cand_full"], k["best_full"])):
        w(f"**Candidate using {name}: {labels[cand]}** (simplest within noise of the best, "
          f"{labels[best]}).\n")
        w("| model | Brier − best [95%] | within noise? |")
        w("|---|---|---|")
        for r in rule.itertuples():
            if r.model == best:
                w(f"| {labels[r.model]} | — (the best) | — |")
            else:
                w(f"| {labels[r.model]} | {f(r.d_vs_best, 4, True)} [{f(r.lo, 4, True)}, "
                  f"{f(r.hi, 4, True)}] | {'yes' if r.tie else 'no'} |")
        w("")

    # 6 ---------------------------------------------------------------
    w("## 6. What the best boosted models learned\n")
    for fs, d in k["imps"].items():
        kind = "mean |SHAP| (log-odds)" if d["kind"] == "shap" else "rise in August log loss when shuffled"
        w(f"**{labels[d['key']]}** — {kind}\n")
        w("| rank | feature | block | importance |")
        w("|---:|---|---|---:|")
        for i, r in d["imp"].head(12).iterrows():
            w(f"| {i + 1} | {FEATURE_LABEL.get(r.feature, r.feature)} | "
              f"{FEATURE_BLOCK.get(r.feature, '?')} | {r.importance:.4f} |")
        if fs == "full":
            w(f"\n**Leak check:** {'⚠ SUSPICIOUS — ' if d['leak'] else 'passed — '}{d['leak_txt']}.")
        w("")

    # 7 ---------------------------------------------------------------
    w("## 7. Going to Week 9\n")
    w(f"- **Location-only candidate: {labels[cand_loc]}** — the default for the MDP's p(s) "
      "(Week 7's decision: ABS rules on location alone, and the policy will be asked about "
      "pitches nobody challenged).")
    w(f"- **Candidate using everything: {labels[k['cand_full']]}** — the instrument for studying "
      "who challenges well.")
    w("- Both are saved in `models/week8/` and named in `models/week8/selection.json`. Week 9 "
      "calibrates them on August, checks subgroups, locks the production model, and only then "
      "opens September — once — for every model at the same time.\n")
    w("## 8. Figures (`figures/week8/`)\n")
    for name, desc in FIGURES:
        w(f"- `{name}` — {desc}")
    w("")
    return "\n".join(out)


FIGURES = [
    ("w8_01_tuning.png", "every setting tried: how much tuning moved the August log loss"),
    ("w8_02_comparison.png", "the one comparison table as a picture, and the Δ vs the baseline"),
    ("w8_03_reliability.png", "reliability diagrams for the four families × two feature sets"),
    ("w8_04_importance.png", "what the best boosted models rely on (SHAP), coloured by block"),
    ("w8_05_shap_distance.png", "the learned distance effect, read straight off the SHAP values"),
    ("w8_06_curves.png", "overturn chance vs inches past each edge: GAM vs the best boosted model"),
]


if __name__ == "__main__":
    main()
