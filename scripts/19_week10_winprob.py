#!/usr/bin/env python3
"""
19_week10_winprob.py — Week 10. V(s): win probability, run expectancy, call transitions.

The plan: "Build V(s) — the probability of eventually winning from a given game
state. This is the other half of your Q-function, and it's what turns 'the call
was wrong' into 'the call was worth something.'"

    python scripts/19_week10_winprob.py                          # 2026 only
    python scripts/19_week10_winprob.py --seasons 2024,2025,2026 # more history (pull with
                                                                 # script 01 --season first)
    python scripts/19_week10_winprob.py --no-gbm                 # skip the boosted version

    0. PRE-FLIGHT   every pitch -> a game state; who won each game; which games count
    1. THE STATE    its exact dimensions (plan step 1)
    2. RUNS         run expectancy (RE24 / RE288) and run distributions — the engine of
                    the structural model, and the inning-level view of a call
    3. EMPIRICAL    win-expectancy table with cell counts; thin cells (plan step 2)
    4. MODELS       the structural (Markov) model and the boosted one, each scored only
                    on games it never saw — 5 folds of whole games (plan step 3)
    5. VALIDATE     Week 9's calibration metrics, agreement with the table where it is
                    well populated, published sanity values, and MLB's own win
                    probability when the data has it (plan step 4)
    6. LOCK         the production V(s): models/winprob/ with a hash of every file
    7. TRANSITIONS  from every (outs, bases, count): where the game goes after a called
                    ball and after a called strike — "stands" vs "overturned" (plan step 5)
    8. FIGURES      figures/week10/

Reads   data/raw/statcast/statcast_<season>_*.parquet (every pitch, not only the cohort)
        data/raw/winprob/winprob_2026.parquet (MLB's win probability, if script 06 ran)
Writes  docs/WEEK10_REPORT.md, docs/week10_*.csv, models/winprob/ (+ models/winprob_history/),
        figures/week10/
"""

import argparse
import datetime as dt
import json
import os
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
from src.abs_baselines import choose                                   # noqa: E402
from src.abs_boosting import available                                 # noqa: E402
from src.abs_calibration import METHODS, Calibrator, cross_fit         # noqa: E402
from src.abs_features import read_any                                  # noqa: E402
from src.abs_gamestate import (BASE_NAMES, COUNT_NAMES, FIT_INNINGS,  # noqa: E402
                               available_seasons, build_states, describe_sources,
                               describe_states, load_pitches, pa_starts)
from src.abs_pipeline import grouped_folds                             # noqa: E402
from src.abs_winprob import (DMAX, N_INN, SANITY, WELL_POPULATED,     # noqa: E402
                             CallTransitions, MarkovWP, RunModel, agreement, empirical_table,
                             fit_gbm, game_bootstrap, gbm_predict, load_winprob, parse_states,
                             write_bundle)

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
MIN_GAMES = 300          # below this, V(s) can't be learned (a season is ~2,430 games)
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
MODEL_DIR = Path(os.environ.get("ABS_MODEL_DIR", "models"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "week10"
WP_DIR = MODEL_DIR / "winprob"
HISTORY = MODEL_DIR / "winprob_history"

LABEL = {"markov": "Structural (Markov)", "markov_raw": "Structural, uncalibrated",
         "gbm": "Boosted", "base": "Base rate (home team always)",
         "mlb_statcast": "MLB — Statcast home_win_exp", "mlb_api": "MLB — Stats API win probability"}
METHOD_LABEL = {"none": "uncalibrated", "platt": "Platt scaling", "isotonic": "isotonic"}
# contexts for the "what a call is worth" examples
CONTEXTS = [("tied, 3rd inning, top", 3, "top", 0), ("tied, 6th inning, bottom", 6, "bot", 0),
            ("tied, 9th inning, bottom", 9, "bot", 0), ("home up 1, 9th inning, top", 9, "top", 1),
            ("home up 4, 7th inning, bottom", 7, "bot", 4), ("home down 6, 5th inning, top", 5, "top", -6)]


def banner(msg):
    print(f"\n{msg}\n{'-' * len(msg)}")


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if abs(x) < 0.5 * 10 ** -d:
        x = 0.0
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fci(v, ci, d=3, sign=False):
    lo, hi = ci
    if lo is None or not np.isfinite(lo):
        return f(v, d, sign)
    return f"{f(v, d, sign)} [{f(lo, d, sign)}, {f(hi, d, sign)}]"


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def is_demo(path):
    return "demo" in str(path).lower()


def pick_backend(choice):
    if choice != "auto":
        ok, why = available(choice)
        if not ok:
            raise SystemExit(f"  {choice} isn't available here: {why}")
        return choice
    for b in ("lightgbm", "catboost", "hgb"):
        if available(b)[0]:
            return b
    return None


# =========================================================== MLB's own numbers
def mlb_references(ps, demo):
    """MLB's published win probability at the same plate-appearance starts, if present."""
    refs = {}
    if demo:
        return refs, "demo data: MLB's win-probability columns are placeholders, so no comparison"
    if "home_win_exp" in ps.columns:
        v = pd.to_numeric(ps["home_win_exp"], errors="coerce")
        if v.notna().mean() > 0.5:
            v = v / 100 if v.max() > 1.5 else v
            refs["mlb_statcast"] = v.to_numpy(float)
    wpf = read_any(DATA_DIR / "raw" / "winprob" / "winprob_2026.parquet")
    if wpf is not None and {"game_pk", "at_bat_index", "home_win_prob"} <= set(wpf.columns):
        w = wpf[["game_pk", "at_bat_index", "home_win_prob"]].drop_duplicates(
            ["game_pk", "at_bat_index"])
        m = ps[["game_pk", "at_bat_number"]].assign(at_bat_index=ps["at_bat_number"] - 1).merge(
            w, on=["game_pk", "at_bat_index"], how="left")
        v = pd.to_numeric(m["home_win_prob"], errors="coerce")
        if v.notna().mean() > 0.5:
            refs["mlb_api"] = (v / 100 if v.max() > 1.5 else v).to_numpy(float)
    return refs, ""


# =========================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seasons", default="2026",
                    help="comma-separated seasons, or 'all' for every season pulled")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-gbm", action="store_true", help="skip the boosted version")
    ap.add_argument("--gbm-backend", default="auto", choices=["auto", "lightgbm", "catboost", "hgb"])
    ap.add_argument("--gbm-max-rows", type=int, default=300000,
                    help="pitches sampled per boosted fit (rows within a game are near-copies)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--allow-small", action="store_true",
                    help=f"run even with fewer than {MIN_GAMES} usable games (testing only)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    demo = is_demo(DATA_DIR)

    # ------------------------------------------------------------ 0. pre-flight
    banner("0. Pre-flight")
    seasons = (available_seasons(DATA_DIR) if args.seasons == "all"
               else [int(s) for s in args.seasons.split(",") if s.strip()])
    raw, missing = load_pitches(DATA_DIR, seasons)
    if missing:
        print(f"  no files for season(s) {missing} — using {sorted(set(seasons) - set(missing))}")
    for line in describe_sources(raw.attrs.get("sources", [])):
        print(f"  {line}")
    p, games = build_states(raw)
    info = describe_states(p, games)
    if info["n_usable"] < MIN_GAMES and not args.allow_small:
        found = "\n".join("    " + l for l in describe_sources(raw.attrs.get("sources", [])))
        raise SystemExit(
            f"\n  Only {info['n_usable']:,} usable games were found — a season has about 2,430, "
            f"and V(s) needs at least {MIN_GAMES:,}.\n{found}\n"
            "  Check data/raw/statcast/: it should hold one file per month of the season\n"
            "  (statcast_2026_2026-03.parquet ... statcast_2026_2026-09.parquet). If the full\n"
            "  pull never finished, re-run  python scripts/01_statcast_pitches_2026.py  (it\n"
            "  resumes where it stopped), then run Week 10 again.")
    print(f"  seasons {info['seasons']} · {info['n_pitches']:,} pitches · {info['n_pa']:,} plate "
          f"appearances · {info['n_games']:,} games"
          + (f" ({info['rows_dropped']:,} rows with no count or score dropped)"
             if info["rows_dropped"] else ""))
    print(f"  usable games {info['n_usable']:,} (a winner, 9+ innings); "
          + (f"left out {info['n_excluded']:,}: " + "; ".join(f"{v} {k}" for k, v in
                                                            info["excluded_why"].items())
             if info["n_excluded"] else "none left out"))
    print(f"  home teams won {pct(info['home_win_rate'])} · {info['runs_per_team_game']:.2f} runs "
          f"per team per game · {pct(info['extra_inning_share'])} went to extra innings")
    print(f"  called pitches: the recorded call matches what the count did next on "
          f"{pct(info['call_agreement'], 2)} (the rest are overturned challenges recorded with "
          "the original call)" if info["call_agreement"] < 0.999 else
          f"  called pitches: the recorded call matches what the count did next on "
          f"{pct(info['call_agreement'], 2)}")
    if demo:
        print("  DEMO DATA — every number below is from a simulated season")

    # ------------------------------------------------------------ 1. the state
    banner("1. The state")
    dims = {"innings": "1–9 + extras (10+, every extra inning alike)", "halves": 2, "outs": 3,
            "bases": 8, "counts": 12, "score_diff": "−30..+30 (61)"}
    n_states = N_INN * 2 * 3 * 8 * 12 * (2 * DMAX + 1)
    print(f"  {N_INN} innings × 2 halves × 3 outs × 8 bases × 12 counts × {2 * DMAX + 1} home "
          f"leads = {n_states:,} states")
    print(f"  empirical table: {N_INN} × 2 × 3 × 8 × 21 leads (−10..+10) = "
          f"{N_INN * 2 * 3 * 8 * 21:,} cells at plate-appearance starts; × 12 counts = "
          f"{N_INN * 2 * 3 * 8 * 21 * 12:,} with the count")

    # ------------------------------------------------------------ 2. runs
    banner("2. Runs — the structural model's engine")
    fit = p[p["fit_row"]]
    rm = RunModel().fit(fit)
    re24 = rm.re_table()
    ps_fit = pa_starts(fit)
    emp_re24 = (ps_fit.groupby(["bases", "outs"])["runs_rest_inning"].agg(["mean", "size"])
                .reset_index())
    emp_re24["model"] = [re24.iloc[b, o] for b, o in zip(emp_re24["bases"], emp_re24["outs"])]
    re_mae = float(np.average((emp_re24["mean"] - emp_re24["model"]).abs(), weights=emp_re24["size"]))
    print(f"  plate-appearance chain from {rm.n_fit:,} pitches (innings 1–{FIT_INNINGS}), "
          f"solved in {rm.iterations} sweeps")
    print(f"  run expectancy, bases empty / 0 out: {re24.iloc[0, 0]:.3f}; bases loaded / 0 out: "
          f"{re24.iloc[7, 0]:.3f}; vs the plain averages: weighted mean |difference| {re_mae:.3f} runs")
    emp_re288 = (fit.groupby(["outs", "bases", "count"])["runs_rest_inning"]
                 .agg(["mean", "size"]).reset_index())
    w_side = rm.n_cell.sum((1, 2, 3))
    re_pool = (rm.RE * (w_side / w_side.sum())[:, None, None, None]).sum(0)
    emp_re288["model"] = re_pool[emp_re288["outs"], emp_re288["bases"], emp_re288["count"]]
    big = emp_re288["size"] >= 200
    re288_mae = float(np.average((emp_re288.loc[big, "mean"] - emp_re288.loc[big, "model"]).abs(),
                                 weights=emp_re288.loc[big, "size"]))
    print(f"  RE288 (with the count) vs the plain averages in cells with 200+ pitches: weighted "
          f"mean |difference| {re288_mae:.3f} runs")
    # Statcast's own pitch run values, if present
    dre = None
    if "delta_run_exp" in p.columns and pd.to_numeric(p["delta_run_exp"], errors="coerce").notna().mean() > 0.5:
        q = fit.copy()
        nxt_re = np.where(q["next_absorb"], 0.0,
                          rm.RE[q["half"], q["next_outs"].clip(0, 2), q["next_bases"], q["next_count"]])
        q["dre_ours"] = q["runs_next"] + nxt_re - rm.RE[q["half"], q["outs"], q["bases"], q["count"]]
        s = pd.to_numeric(q["delta_run_exp"], errors="coerce")
        ok = s.notna()
        r_ = float(np.corrcoef(q.loc[ok, "dre_ours"], s[ok])[0, 1])
        dre = {"r": r_, "mae": float((q.loc[ok, "dre_ours"] - s[ok]).abs().mean()), "n": int(ok.sum())}
        print(f"  vs Statcast's own run value of each pitch (delta_run_exp): r = {r_:.3f}, mean "
              f"|difference| {dre['mae']:.3f} runs over {dre['n']:,} pitches")

    # ------------------------------------------------------------ 3. empirical table
    banner("3. Empirical win expectancy (the plan's step 2)")
    use = p[p["usable"].astype(bool)]
    ps = pa_starts(use)
    emp = empirical_table(ps)
    emp_c = empirical_table(use, with_count=True)
    cover = float(emp.loc[emp["well_populated"], "n"].sum() / emp["n"].sum())
    thin_share = float(emp.loc[emp["thin"], "n"].sum() / emp["n"].sum())
    print(f"  {len(emp):,} of {N_INN * 2 * 3 * 8 * 21:,} cells ever happened; "
          f"{int(emp['well_populated'].sum()):,} have {WELL_POPULATED}+ games and hold "
          f"{pct(cover)} of plate appearances")
    print(f"  thin cells (< 30 games): {int(emp['thin'].sum()):,} — {pct(thin_share)} of plate "
          f"appearances; with the count: {int(emp_c['thin'].sum()):,} of {len(emp_c):,} cells thin")

    # ------------------------------------------------------------ 4. models, out of fold
    banner(f"4. Models — each game scored by a model that never saw it ({args.folds} folds)")
    y = ps["home_win"].to_numpy(int)
    gid = ps["game_pk"].to_numpy()
    fold_rows = grouped_folds(gid, k=args.folds, seed=args.seed)
    fold_of_game = {}
    for j, (_, held) in enumerate(fold_rows):
        for gk in np.unique(gid[held]):
            fold_of_game[gk] = j
    p_fold = p["game_pk"].map(fold_of_game)
    s_ps = parse_states(ps.assign(half=ps["half"], score_diff=ps["diff"]).rename(
        columns={"outs_when_up": "_o"}))
    v_oof = np.full(len(ps), np.nan)
    g_oof = np.full(len(ps), np.nan)
    backend = None if args.no_gbm else pick_backend(args.gbm_backend)
    if not args.no_gbm and backend is None:
        print("  no boosting library available — the boosted version is skipped")
    gbm_trees = []
    for j, (_, held) in enumerate(fold_rows):
        tj = time.time()
        train_mask = p_fold.ne(j) & p_fold.notna()
        rm_j = RunModel().fit(p[train_mask & p["fit_row"]])
        mk_j = MarkovWP(rm_j)
        v_oof[held] = mk_j.v_table[s_ps["i"][held], s_ps["h"][held], s_ps["o"][held],
                                   s_ps["b"][held], s_ps["c"][held],
                                   np.clip(s_ps["d"][held], -DMAX, DMAX) + DMAX]
        msg = f"  fold {j + 1}: structural fitted on {int(train_mask.sum()):,} pitches"
        if backend:
            gm = fit_gbm(p[train_mask & p["usable"].astype(bool)], backend, seed=args.seed,
                         max_rows=args.gbm_max_rows)
            g_oof[held] = gbm_predict(gm, ps.iloc[held])
            gbm_trees.append(gm.best_iteration)
            msg += f"; boosted ({backend}) {gm.best_iteration} trees"
        print(msg + f" ({time.time() - tj:.0f}s)")
    # calibration of the structural model, Week 9's way
    cal_preds = {m: cross_fit(v_oof, y, gid, m, k=args.folds, seed=args.seed) for m in METHODS}
    c_point, c_ci, c_diffs = game_bootstrap(y, cal_preds, gid, B=args.boot, seed=args.seed,
                                            calibration=False)
    _, cal_method, cal_tab = choose(METHODS, c_point, c_diffs)
    v_cal = cal_preds[cal_method]
    print(f"  calibration of the structural model (cross-fitted, Week 9's rule): "
          f"{METHOD_LABEL[cal_method]}")
    for _, r in cal_tab.iterrows():
        print(f"    {METHOD_LABEL[r['model']]:<14} Brier − best {r['d_vs_best']:+.5f} "
              f"[{r['lo']:+.5f}, {r['hi']:+.5f}]{'  <- chosen' if r['model'] == cal_method else ''}")

    # ------------------------------------------------------------ 5. validate
    banner("5. Validate")
    refs, ref_note = mlb_references(ps, demo)
    if ref_note:
        print(f"  ({ref_note})")
    preds = {"markov": v_cal, "markov_raw": v_oof}
    if backend:
        preds["gbm"] = g_oof
    preds["base"] = np.full(len(y), float(y.mean()))
    for k, v in refs.items():
        ok = np.isfinite(v)
        if ok.mean() > 0.95:
            preds[k] = np.where(ok, v, np.nanmean(v))
    if cal_method == "none":
        preds.pop("markov_raw")                 # identical to the production model
    point, ci, diffs = game_bootstrap(y, preds, gid, B=args.boot, seed=args.seed)
    for k in preds:
        s = point[k]
        point[k]["ece"] = M.ece(y, preds[k])
        point[k]["auc"] = M.auc(y, preds[k])
        if k == "base":                          # a constant has no slope
            for m in ("cal_intercept", "cal_slope"):
                point[k][m], ci[k][m] = np.nan, (np.nan, np.nan)
        print(f"  {LABEL[k]:<34} Brier {s['brier']:.4f}  log loss {s['log_loss']:.4f}  "
              f"intercept {f(s['cal_intercept'], 2, True)}  slope {f(s['cal_slope'], 2)}  "
              f"AUC {point[k]['auc']:.3f}")
    verdict = None
    if backend:
        d = diffs[("gbm", "markov")]["brier"]
        verdict = ("better" if d[2] < 0 else "worse" if d[1] > 0 else "within noise")
        print(f"  boosted minus structural: Δ Brier {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}] -> the "
              f"boosted model is {verdict}")
    rel = {k: M.reliability_table(y, preds[k], n_bins=10) for k in preds if k != "base"}
    # where the season is decided: by inning group and by size of the lead
    grp = pd.DataFrame({"y": y, "v": v_cal, "inning": ps["inning"].to_numpy(),
                        "lead": ps["diff"].abs().to_numpy()})
    grp["inning_group"] = pd.cut(grp["inning"], [0, 3, 6, 9, 99], labels=["1–3", "4–6", "7–9", "extras"])
    grp["lead_group"] = pd.cut(grp["lead"], [-1, 0, 2, 4, 99], labels=["tied", "1–2", "3–4", "5+"])
    by_group = []
    for col in ("inning_group", "lead_group"):
        for lv, gg in grp.groupby(col, observed=True):
            lo, hi = M.wilson(int(gg["y"].sum()), len(gg))
            by_group.append({"group": "inning" if col == "inning_group" else "home lead (abs.)",
                             "level": str(lv), "n": len(gg), "pred": gg["v"].mean(),
                             "obs": gg["y"].mean(), "lo": lo, "hi": hi})
    by_group = pd.DataFrame(by_group)
    # agreement with the empirical table
    ps_m = ps.assign(markov=v_cal, gbm=g_oof if backend else np.nan)
    tab = empirical_table(ps_m)
    st = ps_m.copy()
    st["inning_c"] = np.minimum(st["inning"], N_INN)
    st["diff_c"] = st["diff"].clip(-10, 10)
    means = st.groupby(["inning_c", "half", "outs", "bases", "diff_c"])[["markov", "gbm"]].mean()
    tab = tab.merge(means.reset_index(), on=["inning_c", "half", "outs", "bases", "diff_c"])
    agree = {}
    worst = {}
    for k in (["markov", "gbm"] if backend else ["markov"]):
        t_, a_ = agreement(tab, k)
        agree[k] = a_
        worst[k] = t_.reindex(t_["z"].abs().sort_values(ascending=False).index).head(8)
        print(f"  {LABEL[k]} vs the empirical table, {a_['cells']} cells with {WELL_POPULATED}+ "
              f"games ({pct(a_['share_rows'])} of plate appearances): mean |gap| "
              f"{a_['mae'] * 100:.2f} pts; {pct(a_['share_beyond_2se'])} beyond 2 SE "
              "(≈5% expected)")

    # ------------------------------------------------------------ 6. lock
    banner("6. Lock the production V(s)")
    rm_full = rm
    mk = MarkovWP(rm_full)
    calls = CallTransitions().fit(fit)
    calibrator = Calibrator(cal_method).fit(v_oof, y)
    sanity = []
    for name, stt, lo, hi in SANITY:
        s_ = parse_states(stt)
        raw_v = mk.v_table[s_["i"][0], s_["h"][0], s_["o"][0], s_["b"][0], s_["c"][0],
                           s_["d"][0] + DMAX]
        v = float(calibrator.predict(np.array([raw_v]))[0])
        sanity.append({"state": name, "value": v, "lo": lo, "hi": hi, "ok": lo <= v <= hi})
    gbm_full = None
    if backend:
        gbm_full = fit_gbm(use, backend, seed=args.seed, max_rows=args.gbm_max_rows)
        for r_ in sanity:
            stt = next(s for n, s, _, _ in SANITY if n == r_["state"])
            row = pd.DataFrame([{"inning": stt["inning"], "half": 1 if stt["half"] == "bot" else 0,
                                 "outs": stt["outs"], "bases": stt["bases"],
                                 "diff": stt["score_diff"], "balls": 0, "strikes": 0}])
            r_["gbm"] = float(gbm_predict(gbm_full, row)[0])
            r_["gbm_ok"] = bool(r_["lo"] <= r_["gbm"] <= r_["hi"])
    n_ok = sum(r["ok"] for r in sanity)
    n_ok_gbm = sum(r.get("gbm_ok", False) for r in sanity) if backend else None
    print(f"  sanity values: {n_ok} of {len(sanity)} inside the published ranges"
          + (f" (the boosted model: {n_ok_gbm} of {len(sanity)})" if backend else ""))
    for r in sanity:
        print(f"    {'ok  ' if r['ok'] else 'MISS'} {r['value']:.3f}  [{r['lo']:.2f}–{r['hi']:.2f}]  "
              f"{r['state']}")
    old = None
    if (WP_DIR / "manifest.json").exists():
        old = json.loads((WP_DIR / "manifest.json").read_text()).get("version")
        HISTORY.mkdir(parents=True, exist_ok=True)
        if (HISTORY / old).exists():
            shutil.rmtree(HISTORY / old)
        shutil.move(str(WP_DIR), str(HISTORY / old))
    elif WP_DIR.exists():
        shutil.rmtree(WP_DIR)
    now = dt.datetime.now()
    manifest = write_bundle(WP_DIR, markov=mk, run_model=rm_full, calls=calls,
                            calibrator=calibrator, manifest_extra={
        "date": now.strftime("%Y.%m.%d"), "created": now.isoformat(timespec="seconds"),
        "seasons": info["seasons"], "n_games": info["n_usable"], "n_pa": info["n_pa"],
        "n_pitches_fit": int(rm_full.n_fit), "fit_innings": f"1–{FIT_INNINGS}",
        "calibration": {"method": cal_method, "params": calibrator.params},
        "out_of_fold": {k: {m: (float(point[k][m]) if np.isfinite(point[k][m]) else None)
                            for m in ("brier", "log_loss", "cal_intercept", "cal_slope")}
                        for k in preds},
        "sanity": sanity, "demo": demo, "data_dir": str(DATA_DIR)})
    if old is not None:
        if old == manifest["version"]:
            shutil.rmtree(HISTORY / old, ignore_errors=True)
            print("  re-locked: identical to the previous V(s)")
        else:
            print(f"  previous V(s) {old} kept in {HISTORY / old}")
    wp = load_winprob(WP_DIR)
    check = wp.frame(ps.assign(score_diff=ps["diff"]))
    direct = calibrator.predict(mk.v_table[s_ps["i"], s_ps["h"], s_ps["o"], s_ps["b"], s_ps["c"],
                                           np.clip(s_ps["d"], -DMAX, DMAX) + DMAX])
    self_check = float(np.max(np.abs(check - direct)))
    print(f"  locked V(s) {manifest['version']} -> {WP_DIR}")
    print(f"  self-check: the reloaded bundle reproduces every plate-appearance start to within "
          f"{self_check:.1e}")
    print(f"  pre-game home win probability: {wp.info()['pregame_home_wp']:.3f}")

    # ------------------------------------------------------------ 7. transitions
    banner("7. Call transitions — stands vs overturned (the plan's step 5)")
    ttab = calls.table()
    long_rows = []
    for o in range(3):
        for b in range(8):
            for c in range(12):
                for kk in (0, 1):
                    dec, w = calls.cells[int(calls.cell_id(o, b, c, kk))]
                    for (eo, eb, ec, r), ww in zip(dec, w):
                        long_rows.append({"outs": o, "bases": BASE_NAMES[b], "count": COUNT_NAMES[c],
                                          "call": "strike" if kk else "ball",
                                          "next_outs": "inning over" if eo >= 3 else eo,
                                          "next_bases": "" if eo >= 3 else BASE_NAMES[eb],
                                          "next_count": "" if eo >= 3 else COUNT_NAMES[ec],
                                          "runs": r, "prob": ww})
    long_t = pd.DataFrame(long_rows)
    share_rule = float((ttab["p_rulebook"] * ttab["n"]).sum() / max(ttab["n"].sum(), 1))
    print(f"  {calls.n_fit:,} called pitches; {pct(share_rule)} went exactly where the rulebook "
          "says (count moves, ball four = walk, strike three = out); the rest had a steal, a "
          "wild pitch or a runner thrown out on the same pitch")
    examples = []
    for cname, inn, half, lead in CONTEXTS:
        for o in range(3):
            for c, (balls, strikes) in enumerate([(b_, s_) for b_ in range(4) for s_ in range(3)]):
                for call in ("strike", "ball"):
                    stt = {"inning": inn, "half": half, "outs": o, "bases": 0, "score_diff": lead,
                           "balls": balls, "strikes": strikes}
                    cv = wp.call_value(stt, call).iloc[0]
                    examples.append({"context": cname, "outs": o, "count": COUNT_NAMES[c],
                                     "call": call, "wp_home_stands": cv["wp_stands"],
                                     "wp_home_overturned": cv["wp_overturned"],
                                     "dwp_challenger": cv["dwp_challenger"],
                                     "druns_batting": cv["druns_batting"]})
    ex = pd.DataFrame(examples)
    k3 = ex[(ex["count"] == "1-2") & (ex["call"] == "strike") & (ex["outs"] == 2)]
    for _, r in k3.iterrows():
        print(f"  strike three called with 2 out, bases empty, {r['context']}: overturning it is "
              f"worth {100 * r['dwp_challenger']:+.1f} pts of win probability and "
              f"{r['druns_batting']:+.3f} runs to the batting team")

    # ------------------------------------------------------------ write
    DOCS.mkdir(parents=True, exist_ok=True)
    emp.to_csv(DOCS / "week10_empirical_we.csv", index=False)
    emp_c.to_csv(DOCS / "week10_empirical_we_by_count.csv", index=False)
    re_long = []
    for side, sname in ((0, "away batting (top)"), (1, "home batting (bottom)"), (None, "both")):
        for o in range(3):
            for b in range(8):
                for c in range(12):
                    if side is None:
                        val = re_pool[o, b, c]
                        p1 = 1 - (rm.R[:, o, b, c, 0] * (w_side / w_side.sum())).sum()
                    else:
                        val, p1 = rm.RE[side, o, b, c], 1 - rm.R[side, o, b, c, 0]
                    re_long.append({"batting": sname, "outs": o, "bases": BASE_NAMES[b],
                                    "count": COUNT_NAMES[c], "exp_runs": val, "p_score": p1,
                                    "n_pitches": int(rm.n_cell[:, o, b, c].sum() if side is None
                                                     else rm.n_cell[side, o, b, c])})
    pd.DataFrame(re_long).to_csv(DOCS / "week10_run_expectancy.csv", index=False)
    ttab.to_csv(DOCS / "week10_call_transitions.csv", index=False)
    long_t.to_csv(DOCS / "week10_call_transitions_long.csv", index=False)
    ex.to_csv(DOCS / "week10_call_value_examples.csv", index=False)
    pd.DataFrame([{"model": LABEL[k], **{m: point[k][m] for m in point[k]},
                   **{f"{m}_lo": ci[k][m][0] for m in ci[k]},
                   **{f"{m}_hi": ci[k][m][1] for m in ci[k]}} for k in preds]).to_csv(
        DOCS / "week10_validation.csv", index=False)
    pd.DataFrame(sanity).to_csv(DOCS / "week10_sanity.csv", index=False)
    tab.to_csv(DOCS / "week10_empirical_vs_models.csv", index=False)

    k = dict(info=info, demo=demo, rm=rm, re24=re24, emp_re24=emp_re24, re_mae=re_mae,
             re288_mae=re288_mae, dre=dre, emp=emp, emp_c=emp_c, cover=cover,
             thin_share=thin_share, cal_method=cal_method, cal_tab=cal_tab, point=point, ci=ci,
             diffs=diffs, preds=list(preds), verdict=verdict, backend=backend,
             gbm_trees=gbm_trees, agree=agree, worst=worst, sanity=sanity, n_ok=n_ok,
             n_ok_gbm=n_ok_gbm,
             manifest=manifest, self_check=self_check, wp=wp, ttab=ttab, share_rule=share_rule,
             ex=ex, by_group=by_group, rel=rel, ref_note=ref_note, args=args, mk=mk,
             re_pool=re_pool, folds=args.folds, calls=calls, agree_table=tab)
    if not args.no_figures:
        banner("8. Figures")
        import week10_figures as W10
        W10.make_all(FIGS, DATA_DIR, k=k, ps=ps, y=y, preds=preds)
    (DOCS / "WEEK10_REPORT.md").write_text(report(k))
    print(f"\n  report -> {DOCS / 'WEEK10_REPORT.md'}   ({time.time() - t0:.0f}s)")


# =========================================================== report
def report(k):
    w_ = []
    w = w_.append
    info, man = k["info"], k["manifest"]
    w("# Week 10 — Win Probability Component\n")
    w(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/19_week10_winprob.py` · seasons "
      f"{', '.join(map(str, info['seasons']))} · {info['n_usable']:,} games · "
      f"{k['folds']}-fold out-of-game scoring · {k['args'].boot} game-level bootstrap resamples\n")
    if k["demo"]:
        w("> **DEMO DATA** — a simulated season. The method is real; the numbers are not.\n")
    w("> Read this next to `WEEK10.md`, which explains every section.\n")
    w("## The locked V(s)\n")
    w(f"**V(s) {man['version']} = structural (Markov) model + "
      f"{METHOD_LABEL[k['cal_method']]}** · input: inning, half, outs, runners, count, score\n")
    p_ = k["point"]["markov"]
    w(f"- Out-of-fold (every game scored by a model that never saw it): Brier "
      f"{fci(p_['brier'], k['ci']['markov']['brier'], 4)}, calibration intercept "
      f"{fci(p_['cal_intercept'], k['ci']['markov']['cal_intercept'], 2, True)}, slope "
      f"{fci(p_['cal_slope'], k['ci']['markov']['cal_slope'], 2)}.")
    if k["verdict"]:
        d = k["diffs"][("gbm", "markov")]["brier"]
        w(f"- The boosted model (the plan's model version): Δ Brier vs structural {d[0]:+.5f} "
          f"[{d[1]:+.5f}, {d[2]:+.5f}] — **{k['verdict']}**.")
    w(f"- Published sanity values: **{k['n_ok']} of {len(k['sanity'])}** inside their ranges"
      + (f" (the boosted model: {k['n_ok_gbm']} of {len(k['sanity'])})." if k["n_ok_gbm"] is not None
         else "."))
    w(f"- Pre-game home win probability {k['wp'].info()['pregame_home_wp']:.3f}; home teams "
      f"actually won {pct(info['home_win_rate'])}.\n")

    w("## 0. Pre-flight\n")
    left = ("none left out" if not info["n_excluded"] else
            f"left out {info['n_excluded']:,}: " + "; ".join(
                f"{v} {k}" for k, v in info["excluded_why"].items()))
    w(f"- {info['n_pitches']:,} pitches, {info['n_pa']:,} plate appearances, {info['n_games']:,} "
      f"games. Usable for win probability: {info['n_usable']:,} (a winner and 9+ innings); "
      f"{left}.")
    w(f"- Run environment: {info['runs_per_team_game']:.2f} runs per team per game; home teams won "
      f"{pct(info['home_win_rate'])}; {pct(info['extra_inning_share'])} of games went to extra "
      "innings.")
    w(f"- Called pitches: the recorded call matches what the count did next on "
      f"{pct(info['call_agreement'], 2)}. Transitions use what the count did (the *effective* "
      "call), so they are right whichever call Statcast recorded.\n")

    w("## 1. The state\n")
    w(f"| part | values |\n|---|---|\n| inning | 1–9, and extras (10+) as one repeating inning |\n"
      f"| half | top (away bats), bottom (home bats) |\n| outs | 0, 1, 2 |\n"
      f"| runners | 8: empty, 1B, 2B, 1B 2B, 3B, 1B 3B, 2B 3B, loaded |\n"
      f"| count | 12: 0-0 … 3-2 |\n| home lead | −{DMAX} … +{DMAX} |\n")
    w(f"{N_INN} × 2 × 3 × 8 × 12 × {2 * DMAX + 1} = **{N_INN * 2 * 3 * 8 * 12 * (2 * DMAX + 1):,} "
      "states** in V(s). The empirical table uses the home lead clipped to −10…+10: "
      f"{N_INN * 2 * 3 * 8 * 21:,} cells at the start of a plate appearance, "
      f"{N_INN * 2 * 3 * 8 * 21 * 12:,} with the count.\n")

    w("## 2. Runs — the inning-level view\n")
    w(f"The structural model's engine: how plate appearances move outs, runners and runs, "
      f"learned from {k['rm'].n_fit:,} pitches in innings 1–{FIT_INNINGS}, then chained to the "
      "third out. Expected runs from each state to the end of the inning (RE24, count 0-0):\n")
    re24 = k["re24"]
    w("| runners | 0 out | 1 out | 2 out |\n|---|---:|---:|---:|")
    for b, name in enumerate(BASE_NAMES):
        w(f"| {name} | {re24.iloc[b, 0]:.2f} | {re24.iloc[b, 1]:.2f} | {re24.iloc[b, 2]:.2f} |")
    sp = k["rm"].score_prob_table()
    w("\nChance of scoring at least once more this inning:\n")
    w("| runners | 0 out | 1 out | 2 out |\n|---|---:|---:|---:|")
    for b, name in enumerate(BASE_NAMES):
        w(f"| {name} | {pct(sp.iloc[b, 0], 0)} | {pct(sp.iloc[b, 1], 0)} | {pct(sp.iloc[b, 2], 0)} |")
    w(f"\n- Against the plain averages (mean runs to the end of the inning from each state): "
      f"weighted mean |difference| {k['re_mae']:.3f} runs for RE24 and {k['re288_mae']:.3f} for "
      "RE288 (cells with 200+ pitches). Small = the chain reproduces the data.")
    if k["dre"]:
        w(f"- Against Statcast's own run value of each pitch (`delta_run_exp`): correlation "
          f"{k['dre']['r']:.3f}, mean |difference| {k['dre']['mae']:.3f} runs over "
          f"{k['dre']['n']:,} pitches.")
    re_c = k["re_pool"]
    w("\nRun expectancy by count, bases empty (the count is where a challenge acts):\n")
    w("| outs | " + " | ".join(COUNT_NAMES) + " |\n|---|" + "---:|" * 12)
    for o in range(3):
        w(f"| {o} | " + " | ".join(f"{re_c[o, 0, c]:.3f}" for c in range(12)) + " |")
    w("\nFull tables: `docs/week10_run_expectancy.csv` (by batting side, every outs × runners × "
      "count, with the chance of scoring and the number of pitches).\n")

    w("## 3. Empirical win expectancy (the plan's step 2)\n")
    emp = k["emp"]
    w(f"- {len(emp):,} of {N_INN * 2 * 3 * 8 * 21:,} cells ever happened. "
      f"{int(emp['well_populated'].sum()):,} have {WELL_POPULATED}+ games and hold "
      f"{pct(k['cover'])} of all plate appearances.")
    w(f"- Thin cells (< 30 games): {int(emp['thin'].sum()):,}, holding {pct(k['thin_share'])} of "
      f"plate appearances. With the count, {int(k['emp_c']['thin'].sum()):,} of "
      f"{len(k['emp_c']):,} cells are thin — the reason the count comes from the structural "
      "model, not from the table.")
    w("- Counts are of games, not plate appearances: rows from one game share one outcome.")
    w("- Tables: `docs/week10_empirical_we.csv`, `docs/week10_empirical_we_by_count.csv`.\n")

    w("## 4. Models (the plan's step 3)\n")
    w("Both models are scored only on games they never saw: the games are split into "
      f"{k['folds']} folds; each fold is predicted by models fitted on the other four.\n")
    w(f"- **Structural (Markov)** — plate-appearance transitions (smoothed toward a structural "
      f"prior where thin; home and away separately), chained to runs per inning, worked back "
      "from the end of the game with the 2026 rules (no bottom 9th when the home team leads, "
      "walk-offs, the automatic runner in extras). Never sees who won.")
    if k["backend"]:
        trees = k["gbm_trees"]
        w(f"- **Boosted** — {k['backend']}, one fixed setting, monotone in the home lead, early "
          f"stopping on the last 15% of training games ({min(trees)}–{max(trees)} trees across "
          "folds). Learns from who won.")
    w(f"- Calibration of the structural model, cross-fitted Week 9's way:\n")
    w("| method | Brier − best [95%] | |\n|---|---|---|")
    for _, r in k["cal_tab"].iterrows():
        w(f"| {METHOD_LABEL[r['model']]} | {r['d_vs_best']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}] | "
          f"{'★ chosen' if r['model'] == k['cal_method'] else ''} |")
    w("")

    w("## 5. Validation (the plan's step 4)\n")
    w("### 5.1 Calibration, out of fold — Week 9's metrics\n")
    w("| model | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | ECE | AUC |")
    w("|---|---|---:|---|---|---:|---:|")
    for kk in k["preds"]:
        s, c = k["point"][kk], k["ci"][kk]
        w(f"| {LABEL[kk]} | {fci(s['brier'], c['brier'], 4)} | {f(s['log_loss'], 4)} | "
          f"{fci(s['cal_intercept'], c['cal_intercept'], 2, True)} | "
          f"{fci(s['cal_slope'], c['cal_slope'], 2)} | {f(s['ece'], 3)} | {f(s['auc'], 3)} |")
    if k["ref_note"]:
        w(f"\n*{k['ref_note']}.*")
    if any(x.startswith("mlb_") for x in k["preds"]):
        w("\n*MLB's rows are MLB's own published win probability on the same plate "
          "appearances. If a value was recorded after the plate appearance rather than before "
          "it, it knows something V(s) doesn't, so a lower Brier for it is not a fair loss — "
          "compare the calibration columns instead.*")
    w("\nThe base rate (always the home team's season win rate) is the floor: a win-probability "
      "model's Brier is mostly the noise of baseball, so the gaps between real models are "
      "small. Read the intercept and slope: those are what a decision threshold feels.\n")
    w("### 5.2 By inning and by lead\n")
    w("| group | slice | n | predicted | observed [95%] |\n|---|---|---:|---:|---|")
    for _, r in k["by_group"].iterrows():
        w(f"| {r['group']} | {r['level']} | {r['n']:,} | {pct(r['pred'])} | {pct(r['obs'])} "
          f"[{pct(r['lo'])}, {pct(r['hi'])}] |")
    w("\n### 5.3 Agreement with the empirical table\n")
    for kk, a in k["agree"].items():
        w(f"- **{LABEL[kk]}**: {a['cells']} cells with {WELL_POPULATED}+ games "
          f"({pct(a['share_rows'])} of plate appearances); mean |model − table| "
          f"{a['mae'] * 100:.2f} points; {pct(a['share_beyond_2se'])} of cells beyond 2 standard "
          "errors (about 5% expected if the model is right).")
    for kk, t_ in k["worst"].items():
        w(f"\nLargest gaps, {LABEL[kk]} (z = gap in standard errors):\n")
        w("| inning | half | outs | runners | home lead | games | table | model | z |\n"
          "|---:|---|---:|---|---:|---:|---:|---:|---:|")
        for _, r in t_.iterrows():
            inn = "10+" if r["inning_c"] >= N_INN else int(r["inning_c"])
            w(f"| {inn} | {'bot' if r['half'] else 'top'} | {int(r['outs'])} | "
              f"{BASE_NAMES[int(r['bases'])]} | {int(r['diff_c']):+d} | {int(r['n_games'])} | "
              f"{pct(r['rate'])} | {pct(r[kk])} | {r['z']:+.1f} |")
    w("\n### 5.4 Published sanity values\n")
    has_gbm = any("gbm" in r for r in k["sanity"])
    w("| state | V(s) | published range | " + ("boosted | " if has_gbm else "") + "|")
    w("|---|---:|---|" + ("---:|" if has_gbm else "") + "---|")
    for r in k["sanity"]:
        extra = (f" {pct(r['gbm'])}{'' if r.get('gbm_ok', True) else ' ✗'} |" if has_gbm else "")
        w(f"| {r['state']} | {pct(r['value'])} | {pct(r['lo'], 0)}–{pct(r['hi'], 0)} |{extra} "
          f"{'ok' if r['ok'] else '**MISS**'} |")
    w("\nRanges from the published win-expectancy tables (Tango; FanGraphs) for a modern run "
      "environment, kept wide on purpose: they catch a broken model, not a run environment that "
      "differs a little from the 2010s. ✗ = a boosted value outside its range — usually a "
      "rare late-game state, where a model that learns only from game outcomes has few games "
      "to learn from.\n")

    w("## 6. The locked bundle (`models/winprob/`)\n")
    w(f"- version **{man['version']}**, created {man['created']}")
    w(f"- structural model fitted on {man['n_pitches_fit']:,} pitches (innings {man['fit_innings']}) "
      f"from {man['n_games']:,} games; calibration: {METHOD_LABEL[k['cal_method']]}")
    w(f"- self-check: reloading through `src/abs_winprob.py` reproduces every plate-appearance "
      f"start to within {k['self_check']:.1e}")
    w("- every file hashed in `manifest.json`; `load_winprob()` refuses a bundle that changed\n")
    w("```python\nfrom src.abs_winprob import load_winprob\nwp = load_winprob()\n"
      "wp.home({'inning': 9, 'half': 'bot', 'outs': 1, 'on_3b': True, 'home_score': 3, "
      "'away_score': 3})\nwp.call_value({'inning': 7, 'half': 'top', 'outs': 2, 'home_score': 2, "
      "'away_score': 2,\n               'balls': 1, 'strikes': 2}, call='strike')\n```\n")

    w("## 7. Call transitions — stands vs overturned (the plan's step 5)\n")
    w(f"- From {k['calls'].n_fit:,} called pitches, every (outs × runners × count × call) cell: "
      f"where the game went next. {pct(k['share_rule'])} went exactly where the rulebook says; "
      "the rest had a steal, a wild pitch or a runner thrown out on the same pitch — which is "
      "why the table comes from the data.")
    w("- A challenge's two branches: **stands** = the umpire's call; **overturned** = the other "
      "call from the same state. `docs/week10_call_transitions.csv` (one row per cell, with n) "
      "and `docs/week10_call_transitions_long.csv` (every next state with its probability).\n")
    ex = k["ex"]
    w("What overturning a called strike three is worth (bases empty, 1-2 count), by situation:\n")
    w("| situation | outs | win prob. if it stands | if overturned | gain for the batting team | "
      "runs |\n|---|---:|---:|---:|---:|---:|")
    for _, r in ex[(ex["count"] == "1-2") & (ex["call"] == "strike")].iterrows():
        bat_home = "bottom" in r["context"]
        ws = r["wp_home_stands"] if bat_home else 1 - r["wp_home_stands"]
        wo = r["wp_home_overturned"] if bat_home else 1 - r["wp_home_overturned"]
        w(f"| {r['context']} | {r['outs']} | {pct(ws)} | {pct(wo)} | "
          f"{100 * r['dwp_challenger']:+.1f} pts | {r['druns_batting']:+.3f} |")
    w("\nThe same call is worth very different amounts — that is RQ2 in one table. Every "
      "situation × count × call is in `docs/week10_call_value_examples.csv`; "
      "`scripts/20_week10_challenge_value.py` values every 2026 challenge with it.\n")

    w("## 8. Figures (`figures/week10/`)\n")
    w("- `w10_01_runs.png` — run expectancy and the chance of scoring, by runners and outs")
    w("- `w10_02_wp_curves.png` — home win probability through the game, by score")
    w("- `w10_03_calibration.png` — reliability, out of fold: structural vs boosted")
    w("- `w10_04_agreement.png` — model vs empirical table, cell by cell")
    w("- `w10_05_call_value.png` — what overturning strike three is worth, inning by inning")
    w("- `w10_06_count_value.png` — runs an overturn is worth, by count and outs")
    return "\n".join(w_)


if __name__ == "__main__":
    main()
