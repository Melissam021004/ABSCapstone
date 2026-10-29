#!/usr/bin/env python3
"""
18_export_model_dashboard.py — put the Week 7–9 modelling results on the dashboard.

The dashboard is a static site (see dashboard/README.md), so everything the
new "Overturn Model" and "Progress" tabs show has to arrive as one file the
browser loads with a <script> tag. This script gathers it from what Weeks 8
and 9 already wrote — it fits nothing and re-scores nothing:

    models/week8/selection.json            baseline, candidates, leak check
    docs/week8_comparison.csv              every model's August scores + 95% CIs
    docs/week8_vs_baseline.csv             paired differences vs the baseline
    docs/week8_importance_*.csv            what the boosted models rely on
    models/production/manifest.json        the locked p(s)
    docs/week9_calibration.csv, _reliability.csv, _subgroups.csv, _months.csv
    docs/week9_test_*.csv                  only after --final-test

...plus the p(s) surface itself: the locked model is loaded through
src/abs_predict.py (hashes checked) and evaluated on a grid over the plate
for three batter heights, so the page can show — and you can hover — the
exact probability the MDP will use.

    python scripts/18_export_model_dashboard.py
    ABS_DATA_DIR=data_demo ABS_DOCS_DIR=docs_demo ABS_MODEL_DIR=models_demo \\
        python scripts/18_export_model_dashboard.py

Output: dashboard/model_data.js  (then open dashboard/index.html)
Missing pieces (e.g. Week 9 not run yet) are left out and the page says so.
"""

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
MODEL_DIR = Path(os.environ.get("ABS_MODEL_DIR", "models"))
OUT = Path(os.environ.get("ABS_DASHBOARD_DIR", "dashboard"))
HEIGHTS_IN = (69, 73, 77)                 # 5'9", 6'1", 6'5"


def num(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def read(path):
    return pd.read_csv(path) if Path(path).exists() else None


def attach_paired(rows, path):
    """Add each model's paired Δ Brier vs the baseline (same games, same resamples)."""
    t = read(path)
    if t is None:
        return rows
    d = {r.model: r for r in t.itertuples()}
    for m in rows:
        r = d.get(m["model"])
        if r is not None:
            m.update(pd_brier=num(r.brier_d, 5), pd_lo=num(r.brier_lo, 5),
                     pd_hi=num(r.brier_hi, 5))
    return rows


def wide(comp):
    """week8/9 long comparison table -> one row per model."""
    rows = {}
    for r in comp.itertuples():
        d = rows.setdefault(r.model, {"model": r.model, "label": r.label})
        for c in ("family", "feature_set"):
            if hasattr(r, c):
                d[c] = getattr(r, c)
        d[r.metric] = num(r.value)
        d[r.metric + "_lo"] = num(r.ci_lo)
        d[r.metric + "_hi"] = num(r.ci_hi)
    return list(rows.values())


# ------------------------------------------------------------------ week 8
def week8():
    sel_p = MODEL_DIR / "week8" / "selection.json"
    comp = read(DOCS / "week8_comparison.csv")
    if not sel_p.exists() or comp is None:
        return None
    sel = json.loads(sel_p.read_text())
    vs = read(DOCS / "week8_vs_baseline.csv")
    vsd = {r.model: r for r in vs.itertuples()} if vs is not None else {}
    models = []
    for m in attach_paired(wide(comp), DOCS / "week8_paired_vs_baseline.csv"):
        k = m["model"]
        v = vsd.get(k)
        m.update(params=None)
        if v is not None:
            m.update(d_brier=num(v.d_brier, 5), d_lo=num(v.d_brier_lo, 5),
                     d_hi=num(v.d_brier_hi, 5), calibration_ok=bool(v.calibration_ok),
                     beats=bool(v.beats_baseline))
        models.append(m)
    imp = {}
    for fs in ("location", "full"):
        t = read(DOCS / f"week8_importance_{fs}.csv")
        if t is not None:
            key = sel.get("importance_models", {}).get(fs)
            imp[fs] = {"model": key, "label": sel["models"].get(key, {}).get("label", key),
                       "kind": sel.get("importance_kind", {}).get(fs, "shap"),
                       "rows": [{"feature": r.feature,
                                 "label": r.label if isinstance(r.label, str) else r.feature,
                                 "block": r.block if isinstance(r.block, str) else "other",
                                 "importance": num(r.importance, 6)}
                                for r in t.head(14).itertuples()]}
    tun = []
    log = read(DOCS / "week8_tuning_log.csv")
    if log is not None and len(log):
        okl = log[log["error"].fillna("") == ""] if "error" in log else log
        for (b, fs), g in okl.groupby(["backend", "feature_set"]):
            tun.append({"library": b, "feature_set": fs, "n": int(len(g)),
                        "best": num(g["august_logloss"].min()),
                        "median": num(g["august_logloss"].median()),
                        "worst": num(g["august_logloss"].max())})
    labels = {k: v["label"] for k, v in sel["models"].items()}
    return {"baseline": sel["baseline_to_beat"], "baseline_label": labels[sel["baseline_to_beat"]],
            "week7_baseline": sel.get("week7_reported_baseline"),
            "candidate_location": sel["candidate_location"],
            "candidate_full": sel["candidate_full"], "winner": sel.get("winner_all"),
            "labels": labels, "leak": sel.get("leak_check"), "versions": sel.get("boosting_versions"),
            "call_check": {k: sel.get("call_check", {}).get(k) for k in
                           ("verdict", "repair", "n_repaired", "n_challenges")},
            "models": models, "importance": imp, "tuning": tun,
            "n_boosted": sum(1 for v in sel["models"].values()
                             if v["family"] in ("catboost", "lightgbm", "hgb")),
            "n_beat": sum(1 for m in models if m.get("beats"))}


# ------------------------------------------------------------------ week 9
def surface(pm):
    from src.abs_zone import zone_from_height
    out = {"heights": [], "xs": [round(float(x), 3) for x in np.linspace(-1.3, 1.3, 53)]}
    for h_in in HEIGHTS_IN:
        h = h_in / 12
        top, bot = zone_from_height(h)
        zs = np.linspace(max(0.3, bot - 0.7), top + 0.7, 53)
        gx, gz = np.meshgrid(out["xs"], zs)
        entry = {"inches": h_in, "label": f"{h_in // 12}'{h_in % 12}\"",
                 "top": round(float(top), 4), "bot": round(float(bot), 4),
                 "zs": [round(float(z), 3) for z in zs]}
        for name, cs in (("strike", True), ("ball", False)):
            st = pd.DataFrame({"plate_x": gx.ravel(), "plate_z": gz.ravel(),
                               "batter_height_ft": h, "called_strike": cs})
            p = pm.predict_frame(st, strict=False).reshape(gz.shape)
            entry[name] = [[round(float(v), 4) for v in row] for row in p]
        out["heights"].append(entry)
    return out


def week9():
    mpath = MODEL_DIR / "production" / "manifest.json"
    if not mpath.exists():
        return None
    from src.abs_predict import load_production
    pm = load_production(MODEL_DIR / "production")
    man = pm.manifest
    cal = read(DOCS / "week9_calibration.csv")
    rel = read(DOCS / "week9_reliability.csv")
    sub = read(DOCS / "week9_subgroups.csv")
    mo = read(DOCS / "week9_months.csv")
    tsub = read(DOCS / "week9_test_subgroups.csv")
    tcomp = read(DOCS / "week9_test_comparison.csv")
    fs = man["feature_set"]
    method = man["calibration"]["method"]

    def rel_rows(key):
        if rel is None:
            return []
        r = rel[rel["key"] == key]
        return [{"mean_pred": num(x.mean_pred), "obs": num(x.obs_rate), "n": int(x.n),
                 "lo": num(x.ci_lo), "hi": num(x.ci_hi)} for x in r.itertuples()]

    cal_rows = []
    if cal is not None:
        for r in cal.itertuples():
            cal_rows.append({"feature_set": r.feature_set, "label": r.label, "method": r.method,
                             "chosen": bool(r.chosen), "brier": num(r.brier),
                             "brier_lo": num(r.brier_lo), "brier_hi": num(r.brier_hi),
                             "log_loss": num(r.log_loss), "ece": num(r.ece),
                             "cal_slope": num(r.cal_slope, 3),
                             "cal_intercept": num(r.cal_intercept, 3),
                             "d_brier": num(r.d_brier_vs_none, 5), "d_lo": num(r.d_brier_lo, 5),
                             "d_hi": num(r.d_brier_hi, 5)})

    def sub_rows(t):
        if t is None:
            return []
        return [{"window": r.window, "group": r.group, "level": r.level, "n": int(r.n),
                 "pred": num(r.pred), "obs": num(r.obs), "lo": num(r.lo), "hi": num(r.hi),
                 "flag": bool(r.flag)} for r in t.itertuples()]

    test = {"opened": bool(man.get("test_window", {}).get("opened"))}
    if test["opened"] and tcomp is not None:
        test.update(opened_at=man["test_window"].get("opened_at"),
                    times_opened=man["test_window"].get("times_opened", 1),
                    n=man["test_window"].get("challenges"),
                    comparison=attach_paired(wide(tcomp),
                                             DOCS / "week9_test_paired_vs_baseline.csv"))
        tp = DATA_DIR / "processed" / "week9_test_predictions.csv"
        if tp.exists():
            from src import abs_metrics as M
            t = pd.read_csv(tp)
            test["reliability"] = [{"mean_pred": num(x["mean_pred"]), "obs": num(x["obs_rate"]),
                                    "n": int(x["n"]), "lo": num(x["ci_lo"]), "hi": num(x["ci_hi"])}
                                   for x in M.reliability_table(t["y"], t["p_PROD"], n_bins=10,
                                                                strategy="uniform")]
    return {"version": man["version"], "created": man["created"], "model_key": man["model_key"],
            "model_label": man["model_label"], "family": man["family"], "feature_set": fs,
            "features": man["features"], "state_fields": man["state_fields"],
            "calibration": man["calibration"], "windows": man["windows"],
            "metrics": man["metrics"], "decision": man.get("decision"),
            "content_hash": man["content_hash"][:16],
            "relocked_after_test": man.get("relocked_after_test", False),
            "calibration_table": cal_rows,
            "reliability": {"uncalibrated": rel_rows(f"{fs}|none"),
                            "calibrated": rel_rows(f"{fs}|{method}")},
            "subgroups": sub_rows(sub) + sub_rows(tsub), "months": sub_rows(mo),
            "test": test, "surface": surface(pm)}


# ------------------------------------------------------------------ progress
FALL = [
    (3, "Sep 14–18", "Data-feasibility spike", "Feasibility report, source inventory, risk decision"),
    (4, "Sep 21–25", "Ingestion pipeline", "Versioned, resumable ingestion from six public sources"),
    (5, "Sep 28–Oct 2", "Join everything", "Analysis-ready pitch table, data dictionary, QA report"),
    (6, "Oct 5–9", "Define the cohort", "EDA and a frozen cohort with a temporal split"),
    (7, "Oct 12–16", "Baseline overturn model", "Logistic + GAM, calibration-first evaluation"),
    (8, "Oct 19–23", "Gradient boosting", "CatBoost & LightGBM vs the baseline, SHAP, leak check"),
    (9, "Oct 26–30", "Calibrate and lock", "Calibrated p(s), subgroup checks, model card, predict()"),
    (10, "Nov 2–6", "Win-probability component", "Validated V(s) and call-outcome transitions"),
    (11, "Nov 9–13", "The MDP", "MDP spec, solver, unit tests, first threshold policy"),
    (12, "Nov 16–20", "Dashboard v0", "Dashboard v0 and interim research memo"),
    (13, "Nov 23–27", "Thanksgiving", "—"),
    (14, "Nov 30–Dec 4", "Consolidate", "Fall research package to supervisor"),
    (15, "Dec 7–11", "Buffer", "Revisions only if required"),
]


def progress(w8, w9):
    res = {3: "Challenge-to-pitch join confirmed feasible; sources and risks logged",
           4: "Scripts 01–07 pull Statcast, challenges, leaderboards, umpires, WP, bios",
           5: "One row per pitch from six sources; automated QA assertions",
           6: "Inclusion rules as code; train Mar–Jul · validate Aug · test Sep"}
    done = {3, 4, 5, 6}
    if w8:
        done.add(7)
        res[7] = (f"Baseline to beat: {w8['baseline_label']}"
                  + (" (refit on the corrected call data)" if w8["call_check"].get("repair")
                     else ""))
        done.add(8)
        lk = w8.get("leak") or {}
        res[8] = (f"{w8['n_beat']} of {w8['n_boosted']} boosted models reliably beat the "
                  f"baseline · leak check {'FLAGGED' if lk.get('flagged') else 'passed'}")
    else:
        res[7] = "Baseline models fitted (see WEEK7_BASELINE_REPORT.md)"
        done.add(7)
    if w9:
        done.add(9)
        se = (w9.get("metrics") or {}).get("september")
        how = {"none": "no calibration needed", "platt": "Platt scaling",
               "isotonic": "isotonic calibration"}.get(w9["calibration"]["method"], "")
        res[9] = (f"p(s) {w9['version']} locked: {w9['model_label']} + {how}"
                  + (f" · September Brier {se['brier']:.4f}" if se else " · test sealed"))
    rows, now_set = [], False
    for wk, dates, title, deliv in FALL:
        if wk in done:
            st = "done"
        elif not now_set and wk != 13:
            st, now_set = "next", True
        else:
            st = "planned"
        rows.append({"week": wk, "dates": dates, "title": title, "deliverable": deliv,
                     "status": st, "result": res.get(wk, "")})
    return rows


def main():
    w8, w9 = week8(), week9()
    demo = "demo" in str(DATA_DIR).lower() or "demo" in str(MODEL_DIR).lower()
    payload = {"meta": {"built": dt.date.today().isoformat(), "is_demo": demo,
                        "has_week8": w8 is not None, "has_week9": w9 is not None,
                        "test_opened": bool(w9 and w9["test"]["opened"])},
               "week8": w8, "week9": w9, "progress": progress(w8, w9)}
    OUT.mkdir(parents=True, exist_ok=True)
    js = ("/* generated by scripts/18_export_model_dashboard.py — do not edit by hand */\n"
          "window.ABS_MODEL = " + json.dumps(payload, separators=(",", ":"), default=str) + ";\n")
    (OUT / "model_data.js").write_text(js)
    kb = len(js) / 1024
    print(f"  {OUT / 'model_data.js'}  ({kb:,.0f} KB)")
    print(f"  Week 8: {'yes' if w8 else 'not found'} · Week 9: {'yes' if w9 else 'not found'} · "
          f"test window: {'opened' if payload['meta']['test_opened'] else 'sealed'}")
    if demo:
        print("  DEMO data — the page will say so.")


if __name__ == "__main__":
    main()
