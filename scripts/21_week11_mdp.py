#!/usr/bin/env python3
"""
21_week11_mdp.py — Week 11. The challenge decision as a Markov decision process.

Builds the pitch-by-pitch game, works out V(s, r) for every state by backward
induction, and answers the questions from the weekly plan and from your professor:

    1. THE GAME        a pitch-level model of 2026 (every pitch, every count), checked
                       against the season's run expectancy and Week 10
    2. LEVERAGE        Tom Tango's Leverage Index from our own V(s), checked against
                       the numbers Tango and FanGraphs published
    3. V(s, r)         both teams' challenges in the state, both teams choosing well:
                       what a challenge is worth, inning by inning (win probability)
    4. RUNS            the same game played for run differential — a challenge priced
                       in runs, next to Statcast's 0.2-run price
    5. THRESHOLDS      p* = H / (Δ + H): the confidence a team needs, by situation,
                       by leverage and by challenges left (the "money figure")
    6. RUN LEVELS      at-bat, inning and game: how each measure of run production
                       turns into win probability (Tango: runs early, one run late)
    7. POLICIES        the optimal rule vs never / p ≥ 0.5 / Statcast's breakeven /
                       breakeven × LI / play-for-runs — wins lost per season
    8. REAL 2026       a preview of RQ3: the season's decisions against the thresholds
    9. UMPIRES         does an inaccurate plate umpire change what a challenge is worth?

    python scripts/21_week11_mdp.py
    python scripts/21_week11_mdp.py --sims 5000      # faster simulations

Reads   models/production (p(s), Week 9), models/winprob (V(s), Week 10), the cohort,
        data/raw/statcast (every pitch)
Writes  models/mdp/ (the solved MDP), docs/WEEK11_REPORT.md, docs/week11_*.csv,
        figures/week11/w11_01..06
"""

import argparse
import datetime as dt
import json
import os
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

from src.abs_features import add_features, describe_call_check, load_cohort   # noqa: E402
from src.abs_gamestate import (BASE_NAMES, N_BASES, N_COUNTS,                  # noqa: E402
                               N_OUTS, build_states, describe_sources, load_pitches)
from src.abs_leverage import Leverage                                         # noqa: E402
from src.abs_mdp import (D, N_INN, NEVER, NEXT_BALL, NEXT_STRIKE,             # noqa: E402
                         WALK_BASES, WALK_RUNS, Breakeven, Constant, FromSolution,
                         GameMDP, PitchModel, Simulator, flip_runs, save_solution,
                         solve_half, wp_loss)
from src.abs_pipeline import resolve_fingerprint                              # noqa: E402
from src.abs_predict import load_production                                   # noqa: E402
from src.abs_value import KEY, umpire_table, valuation_table                  # noqa: E402
from src.abs_winprob import load_winprob                                      # noqa: E402

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
CONFIG = Path(os.environ.get("ABS_CONFIG_DIR", "config"))
MODELS = Path(os.environ.get("ABS_MODEL_DIR", "models"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "week11"
GAMES = 162
SAVANT_COST = 0.2                     # runs: Statcast's price of a failed challenge
CLEMENS = {                           # RE288 run value of flipping a call (FanGraphs, Apr 1 2026)
    ("3B", 1, "0-0"): 0.12, ("3B", 1, "1-0"): 0.15, ("3B", 1, "2-0"): 0.20,
    ("3B", 1, "3-0"): 0.12, ("3B", 1, "3-2"): 0.84, ("empty", 2, "3-2"): 0.23,
}
TIER_LABEL = {0: "most accurate third", 1: "middle third", 2: "least accurate third"}
CONTEXTS = ["innings 1–6", "7th or later, within 1 run", "7th or later, 2+ runs apart"]


# ------------------------------------------------------------------ formatting
def banner(msg):
    print(f"\n{msg}\n{'-' * len(msg)}")


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def pts(x, d=2):
    """Win probability in percentage points."""
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f} pts"


def num(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def md_table(df, cols=None, fmt=None, rename=None):
    df = df if cols is None else df[cols]
    fmt = fmt or {}
    rename = rename or {}
    out = ["| " + " | ".join(str(rename.get(c, c)) for c in df.columns) + " |",
           "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append(num(v, 3))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def half_name(i, s):
    return ("extras" if i >= N_INN else f"{i}") + (" top" if s == 0 else " bot")


def is_demo(path):
    return "demo" in str(path).lower()


# ------------------------------------------------------------------ helpers
def run_tables(rm):
    """Per (side, outs, bases, count): P(the plate appearance ends without an out),
    P(at least one more run this inning) and RE288 — all from Week 10's run model."""
    from src.abs_winprob import _end_decode
    T = rm.T                                             # (2, 3, 8, 12, end)
    n_end = T.shape[-1]
    eo = np.array([_end_decode(e)[0] for e in range(n_end)])
    no_out = np.zeros(T.shape[:-1])
    for o in range(N_OUTS):
        no_out[:, o] = T[:, o][..., eo == o].sum(-1)
    p1 = 1 - rm.R[..., 0]
    return no_out, p1, rm.RE


def flip_generic(X, after_walk, after_k):
    """Value after a BALL minus after a STRIKE, for any per-state table X (…, 3, 8, 12)."""
    out = np.zeros(X.shape)
    for o in range(N_OUTS):
        for b in range(N_BASES):
            for c in range(N_COUNTS):
                xb = X[..., o, b, NEXT_BALL[c]] if NEXT_BALL[c] >= 0 else after_walk(X, o, b)
                xs = X[..., o, b, NEXT_STRIKE[c]] if NEXT_STRIKE[c] >= 0 else after_k(X, o, b)
                out[..., o, b, c] = xb - xs
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--sims", type=int, default=20000, help="simulated games per policy")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t_start = time.time()
    demo = is_demo(DATA_DIR)
    R = {}                                               # everything the report needs

    # ================================================================ 0. inputs
    banner("0. Inputs")
    pmodel = load_production()
    wp = load_winprob()
    raw = load_cohort(DATA_DIR)
    fingerprint, fp_note = resolve_fingerprint(raw, CONFIG)
    cohort, check = add_features(raw, return_check=True)
    for line in describe_call_check(check):
        print(f"  {line}")
    R["versions"] = {"p": pmodel.manifest.get("version"), "v": wp.version, "cohort": fingerprint}
    print(f"  p(s) {R['versions']['p']} · V(s) {wp.version} · cohort {fingerprint}"
          + (f" ({fp_note})" if fp_note else ""))
    val = valuation_table(cohort, pmodel, wp)
    rcols = [c for c in ("challenges_remaining_batting", "challenges_remaining_fielding")
             if c in cohort.columns]
    val = val.merge(cohort[KEY + rcols].drop_duplicates(KEY), on=KEY, how="left")
    raw_pitches = load_pitches(DATA_DIR, (2026,))[0]
    for i, line in enumerate(describe_sources(raw_pitches.attrs.get("sources", []))):
        print(("  every pitch, " if i == 0 else "  ") + line)
    pitches, games = build_states(raw_pitches)
    st = pitches[KEY + ["outs", "bases", "count", "diff", "half"]]
    val = val.merge(st, on=KEY, how="left")
    ok_state = val[["outs", "bases", "count", "diff"]].notna().all(axis=1)
    val = val[ok_state].copy()
    for c in ("outs", "bases", "count", "diff", "half"):
        val[c] = val[c].astype(int)
    val["r_bat"] = np.clip(pd.to_numeric(val.get("challenges_remaining_batting"), errors="coerce")
                           .fillna(0), 0, 2).astype(int)
    val["r_fld"] = np.clip(pd.to_numeric(val.get("challenges_remaining_fielding"), errors="coerce")
                           .fillna(0), 0, 2).astype(int)
    n_games = int(val["game_pk"].nunique())
    R["n_called"], R["n_games"] = len(val), n_games
    print(f"  {len(val):,} called pitches in {n_games:,} games carry a game state, p(s) and "
          f"both teams' challenges left")
    if demo:
        print("  DEMO DATA — every number below is from a simulated season")

    # ================================================================ 1. the game
    banner("1. The pitch-level game")
    called = pd.DataFrame({"count": val["count"], "strike": val["call_strike"].astype(int),
                           "p": val["p_s"]})
    pm = PitchModel.fit(pitches, called)
    pm_played = PitchModel.fit(pitches, called, calls="final")
    info = pm.info
    print(f"  {info['n_pitches']:,} pitches: {info['n_in_play']:,} balls in play, "
          f"{info['n_strikeouts']:,} strikeouts, {info['n_walks_hbp']:,} walks/HBP, "
          f"{info['n_runner_moves']:,} runner moves between pitches "
          f"({pct(info['runner_move_rate'])} of pitches)")
    if info.get("n_other_dropped"):
        print(f"  {info['n_other_dropped']:,} pitches with other descriptions left out "
              f"({', '.join(info['other_descriptions'][:5])})")
    # run expectancy check
    f = pitches[pitches["fit_row"]]
    emp = f.groupby(["half", "outs", "bases", "count"])["runs_rest_inning"].agg(["mean", "size"])
    re_rows = []
    re_models = {}
    for name, model in (("as played", pm_played), ("umpire only", pm)):
        re_models[name] = np.stack([
            solve_half(model, s, np.zeros((1, 3, 3)), objective="runs", keep=False)["V"][..., 0, 0, 0]
            * (1 if s == 1 else -1) for s in (0, 1)])
    E = np.full((2, N_OUTS, N_BASES, N_COUNTS), np.nan)
    W = np.zeros((2, N_OUTS, N_BASES, N_COUNTS))
    for (h, o, b, c), r in emp.iterrows():
        E[h, o, b, c], W[h, o, b, c] = r["mean"], r["size"]
    w = W / W.sum()
    for name, X in (("pitch model — as played", re_models["as played"]),
                    ("Week 10 run model", wp.rm.RE),
                    ("pitch model — umpire calls only", re_models["umpire only"])):
        err = X - E
        re_rows.append({"model": name, "mean_error": float(np.nansum(w * err)),
                        "mean_abs_error": float(np.nansum(w * np.abs(err))),
                        "re24_empty_0": float((X[:, 0, 0, 0] * W[:, 0, 0, 0]).sum() / W[:, 0, 0, 0].sum()),
                        "re24_loaded_0": float((X[:, 0, 7, 0] * W[:, 0, 7, 0]).sum()
                                               / max(W[:, 0, 7, 0].sum(), 1))})
    re_rows.append({"model": "the season (observed)", "mean_error": 0.0, "mean_abs_error": 0.0,
                    "re24_empty_0": float(np.nansum(E[:, 0, 0, 0] * W[:, 0, 0, 0]) / W[:, 0, 0, 0].sum()),
                    "re24_loaded_0": float(np.nansum(E[:, 0, 7, 0] * W[:, 0, 7, 0]) / max(W[:, 0, 7, 0].sum(), 1))})
    re_check = pd.DataFrame(re_rows)
    challenge_runs = float(np.nansum(w * (re_models["as played"] - re_models["umpire only"])))
    for _, r in re_check.iterrows():
        print(f"  {r['model']:<33} RE24 empty/0 out {r['re24_empty_0']:.3f}, loaded/0 out "
              f"{r['re24_loaded_0']:.3f}; error vs the season {r['mean_error']:+.4f} "
              f"(typical {r['mean_abs_error']:.4f})")
    print(f"  the 2026 challenges added {challenge_runs:+.4f} runs per state to the batting team's "
          f"inning (as played − umpire calls only)")
    # the win probability the MDP's game gives (as played, nobody challenging) vs Week 10's V(s)
    from src.abs_winprob import DMAX
    sol_check = GameMDP(pm_played).solve(policies={"home": Constant(NEVER), "away": Constant(NEVER)})
    q = pitches[pitches["usable"].fillna(False).astype(bool)]
    q = q[q["diff"].abs() <= D]
    if len(q) > 250000:
        q = q.sample(250000, random_state=args.seed)
    ii = np.minimum(q["inning"].to_numpy(int), N_INN)
    v_mdp = sol_check.lookup("V", ii, q["half"].to_numpy(), q["outs"].to_numpy(), q["bases"].to_numpy(),
                             q["count"].to_numpy(), q["diff"].to_numpy(), 0, 0)
    v_w10 = wp.v[ii - 1, q["half"].to_numpy(int), q["outs"].to_numpy(int), q["bases"].to_numpy(int),
                 q["count"].to_numpy(int), np.clip(q["diff"].to_numpy(int), -DMAX, DMAX) + DMAX]
    y = q["home_win"].to_numpy(float)
    okv = np.isfinite(v_mdp) & np.isfinite(v_w10) & np.isfinite(y)
    v_check = {"n": int(okv.sum()), "mean_abs_pts": float(100 * np.abs(v_mdp - v_w10)[okv].mean()),
               "p95_abs_pts": float(100 * np.percentile(np.abs(v_mdp - v_w10)[okv], 95)),
               "corr": float(np.corrcoef(v_mdp[okv], v_w10[okv])[0, 1]),
               "brier_mdp": float(np.mean((v_mdp[okv] - y[okv]) ** 2)),
               "brier_w10": float(np.mean((v_w10[okv] - y[okv]) ** 2)),
               "pregame_mdp": sol_check.pregame(0, 0)}
    print(f"  its win probability vs Week 10's V(s) on {v_check['n']:,} real pitches: mean difference "
          f"{v_check['mean_abs_pts']:.2f} pts (95% within {v_check['p95_abs_pts']:.2f}), correlation "
          f"{v_check['corr']:.5f}; Brier {v_check['brier_mdp']:.4f} vs {v_check['brier_w10']:.4f}")
    R["v_check"] = v_check
    wrong_pg = float(val["p_s"].sum() / n_games)
    clear_pg = float((val["p_s"] >= 0.9).sum() / n_games)
    by_call = val.groupby("call_strike")["p_s"].agg(["sum", "size"])
    print(f"  mistakes on offer: {wrong_pg:.2f} expected wrong calls per game ({clear_pg:.2f} with "
          f"p(s) ≥ 90%); {pct(by_call.loc[True, 'sum'] / by_call.loc[True, 'size'])} of called "
          f"strikes and {pct(by_call.loc[False, 'sum'] / by_call.loc[False, 'size'])} of called balls")
    R.update(pm_info=info, re_check=re_check, challenge_runs=challenge_runs,
             wrong_pg=wrong_pg, clear_pg=clear_pg)
    ev_tab = pm.summary()

    # ================================================================ 2. leverage
    banner("2. Leverage — Tango's LI from our own V(s)")
    lev = Leverage.fit(wp, pitches)
    li_inn = lev.by_inning()
    li_9 = lev.ninth()
    bands = lev.bands()
    top_states = lev.top_states(8)
    shares = dict(zip(bands["band"], bands["share_of_PAs"]))
    mx_key = top_states.iloc[0]["key"]
    li_max = float(lev.pa_li.max())
    print(f"  {len(lev.pa_li):,} plate appearances; LI averages 1 by construction; "
          f"{pct(shares['below 1'])} are below 1 (FanGraphs: about 60%), "
          f"{pct(shares['high (> 2)'])} above 2 (about 10%)")
    for _, r in li_inn.iterrows():
        pub = r["li_2013_published"]
        print(f"    inning {int(r['inning']):>2}{'+' if r['inning'] == 10 else ' '}: LI {r['li_2026']:.2f}"
              f"  (2013, Tango's LI: {num(pub, 2)})")
    for _, r in li_9.iterrows():
        print(f"  {r['half']} 9th: {r['li_2026_all']:.2f} (2013: {r['li_2013_all']:.2f}); its first PA "
              f"{r['li_2026_first_pa']:.2f} (2013 first play: {r['li_2013_first_play']:.2f})")
    print(f"  highest LI seen: {li_max:.2f} — inning {mx_key[0]}, {'bottom' if mx_key[1] else 'top'}, "
          f"{mx_key[2]} out, {BASE_NAMES[mx_key[3]]}, home lead {mx_key[4]:+d}")
    li_grid = lev.mdp_grid(D)
    R.update(li_inn=li_inn, li_9=li_9, bands=bands, li_max=li_max, li_max_key=mx_key,
             top_states=top_states)

    # ================================================================ 3. V(s, r)
    banner("3. V(s, r) — both teams' challenges, both teams choosing well")
    game = GameMDP(pm)
    sol = game.solve()
    pre = sol.start[(1, 0)][D]                           # (r_home, r_away), tied, first pitch
    v_all = ((pre[2, 2] - pre[0, 2]) + (pre[2, 0] - pre[2, 2])) / 2
    v_first = ((pre[1, 2] - pre[0, 2]) + (pre[2, 0] - pre[2, 1])) / 2
    v_second = ((pre[2, 2] - pre[1, 2]) + (pre[2, 1] - pre[2, 2])) / 2
    print(f"  solved {len(sol.halves)} half-innings × 54,432 states in {sol.seconds:.1f}s; "
          f"home team wins {pct(sol.pregame())} with both sides choosing well "
          f"({pct(pre[0, 0])} if neither had a challenge)")
    print(f"  at the first pitch, a team's two challenges are worth {pts(v_all)} of win probability "
          f"= {v_all * GAMES:.1f} wins a season; the first {pts(v_first)}, the second {pts(v_second)}")
    hold_rows = []
    sol_r = GameMDP(pm, "runs").solve()
    for i in range(1, N_INN + 1):
        for s in (0, 1):
            for team in ("home", "away"):
                role = "batting" if (team == "home") == (s == 1) else "fielding"
                row = {"inning": i, "half": s, "half_label": half_name(i, s), "team": team, "role": role}
                for tag, so in (("win", sol), ("runs", sol_r)):
                    row[f"H1_{tag}"] = so.hold_value(i, s, team, 0, 1, 1)
                    row[f"H2_{tag}"] = so.hold_value(i, s, team, 0, 2, 2)
                hold_rows.append(row)
    hold = pd.DataFrame(hold_rows)
    hv = hold[hold["role"] == "fielding"]                 # the team in the field at the half's start
    for i in (1, 4, 7, 9, N_INN):
        for s in (0, 1):
            r = hv[(hv["inning"] == i) & (hv["half"] == s)]
            if len(r):
                r = r.iloc[0]
                print(f"    start of {half_name(i, s):>10}, tied: last challenge {pts(r['H1_win'])} "
                      f"({r['H1_runs']:.3f} runs), second {pts(r['H2_win'])} ({r['H2_runs']:.3f} runs)")
    rpw = hold.loc[(hold["inning"] <= 6), "H1_runs"].sum() / max(hold.loc[(hold["inning"] <= 6), "H1_win"].sum(), 1e-12)
    print(f"  runs per win implied by the two currencies (innings 1–6): {rpw:.1f} "
          f"(the usual rule of thumb: about 10)")
    R.update(pregame=sol.pregame(), pregame_none=float(pre[0, 0]), v_all=v_all, v_first=v_first,
             v_second=v_second, hold=hold, rpw=rpw, extra_x=sol.extra, solve_seconds=sol.seconds,
             pregame_runs=sol_r.pregame())

    # ================================================================ 4. runs
    banner("4. Runs — the same game played for run differential")
    h1 = hold[(hold["inning"] == 1) & (hold["half"] == 0) & (hold["role"] == "fielding")].iloc[0]
    print(f"  at the first pitch a challenge is worth {h1['H1_runs']:.2f} runs (the last one) and "
          f"{h1['H2_runs']:.2f} runs (the second); Statcast prices a failed challenge at "
          f"{SAVANT_COST:.1f} runs, FanGraphs' early estimate of an unspent one was ≈ 0.1")
    R.update(h1_runs_start=h1["H1_runs"], h2_runs_start=h1["H2_runs"])
    # save the solved model now, so nothing below can lose it
    out = MODELS / "mdp"
    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "p_version": R["versions"]["p"],
            "v_version": wp.version, "cohort": fingerprint, "demo": demo,
            "pregame_home": sol.pregame(), "value_two_challenges": v_all, "value_first": v_first,
            "value_second": v_second, "pitch_model": {k: v for k, v in info.items() if k != "other_descriptions"}}
    save_solution(sol, out, meta)
    save_solution(sol_r, out, meta)
    np.savez_compressed(out / "pitch_model.npz", **pm.to_arrays())
    np.savez_compressed(out / "leverage.npz", li=lev.li.astype(np.float32), denom=lev.denom)
    (out / "manifest.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"  saved the solved MDP to {out}/ (load it with src.abs_mdp.load_mdp())")

    # ================================================================ 5. thresholds
    banner("5. Thresholds — the confidence a challenge needs")
    rm = wp.rm
    flip = flip_runs(rm.RE)                              # (2, 3, 8, 12)
    situations = [
        ("1st, top, tied, bases empty, 0 out, 0-0 — batter, 2 left", 1, 0, 0, 0, 0, 0, 1, 2, 2),
        ("1st, top, tied, bases empty, 0 out, 0-0 — batter, 1 left", 1, 0, 0, 0, 0, 0, 1, 1, 2),
        ("1st, bot, home down 8, bases empty, 0 out, 0-0 — batter, 1 left (blowout)", 1, 1, 0, 0, 0, -8, 1, 1, 2),
        ("4th, top, tied, bases loaded, 2 out, 3-2 — batter, 1 left", 4, 0, 2, 7, 11, 0, 1, 1, 2),
        ("5th, bot, tied, runner on 1st, 1 out, 2-2 — catcher, 1 left", 5, 1, 1, 1, 8, 0, 0, 1, 2),
        ("7th, top, away up 1, bases loaded, 2 out, 3-2 — catcher, 1 left", 7, 0, 2, 7, 11, -1, 0, 1, 2),
        ("9th, bot, tied, runner on 3rd, 1 out, 3-2 — batter, 1 left", 9, 1, 1, 4, 11, 0, 1, 1, 1),
        ("9th, bot, tied, runner on 3rd, 1 out, 3-2 — catcher, 1 left", 9, 1, 1, 4, 11, 0, 0, 1, 1),
        ("9th, top, away down 1, bases empty, 2 out, 0-2 — batter, 1 left (strike three ends it)", 9, 0, 2, 0, 2, 1, 1, 1, 1),
        ("extras, top, tied, runner on 2nd, 0 out, 1-1 — batter, 1 left", 10, 0, 0, 2, 4, 0, 1, 1, 1),
        ("8th, top, home up 6, bases empty, 1 out, 1-1 — batter, 1 left", 8, 0, 1, 0, 4, 6, 1, 1, 2),
    ]
    ex = []
    for name, i, s, o, b, c, d, call, r_dec, r_oth in situations:
        rb, rf = (r_dec, r_oth) if call == 1 else (r_oth, r_dec)
        t, H, Dl = (float(sol.lookup(k, i, s, o, b, c, d, rb, rf)) for k in
                    (("thr_S", "H_S", "D_S") if call == 1 else ("thr_B", "H_B", "D_B")))
        li = float(lev.li[min(i, N_INN) - 1, s, o, b, int(np.clip(d, -30, 30)) + 30])
        rv = float(flip[s, o, b, c])
        ex.append({"situation": name, "call": "strike" if call == 1 else "ball", "p_star": t,
                   "swing_pts": 100 * Dl, "hold_pts": 100 * H, "LI": li, "run_value": rv,
                   "savant_p": SAVANT_COST / (SAVANT_COST + rv), "_s": s, "_o": o, "_b": b, "_c": c})
    ex = pd.DataFrame(ex)
    for _, r in ex.iterrows():
        print(f"  {r['situation']:<74} p* {num(r['p_star'], 2):>5}  (Δ {r['swing_pts']:.2f} pts, "
              f"H {r['hold_pts']:.2f} pts, LI {r['LI']:.1f}; Statcast {num(r['savant_p'], 2)})")
    # every real opportunity
    opp = val[val["opportunity"] & np.isfinite(val["p_s"])].copy()
    strike = opp["call_strike"].to_numpy(bool)
    args_ = (opp["inning"].to_numpy(), opp["half"].to_numpy(), opp["outs"].to_numpy(),
             opp["bases"].to_numpy(), opp["count"].to_numpy(), opp["diff"].to_numpy(),
             opp["r_bat"].to_numpy(), opp["r_fld"].to_numpy())
    for fld, (a, b_) in {"p_star": ("thr_S", "thr_B"), "hold": ("H_S", "H_B"),
                         "swing": ("D_S", "D_B")}.items():
        opp[fld] = np.where(strike, sol.lookup(a, *args_), sol.lookup(b_, *args_))
    opp["r_self"] = np.where(strike, opp["r_bat"], opp["r_fld"])
    opp["li"] = lev.for_states(opp.assign(inning=opp["inning"]))
    opp["run_value"] = flip[opp["half"], opp["outs"], opp["bases"], opp["count"]]
    opp["savant_p"] = SAVANT_COST / (SAVANT_COST + np.maximum(opp["run_value"], 1e-9))
    opp["savant_li_p"] = SAVANT_COST / (SAVANT_COST + np.maximum(opp["run_value"] * opp["li"], 1e-9))
    opp = opp[opp["r_self"] >= 1]
    fin = np.isfinite(opp["p_star"])
    li_edges = np.array([0, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.5, 7.0, 20])
    curve = []
    for r_self in (1, 2):
        d_ = opp[fin & (opp["r_self"] == r_self)]
        bins = np.clip(np.searchsorted(li_edges, d_["li"], side="right") - 1, 0, len(li_edges) - 2)
        for j in range(len(li_edges) - 1):
            x = d_[bins == j]
            if len(x) < 30:
                continue
            curve.append({"r_left": r_self, "li_lo": li_edges[j], "li_hi": li_edges[j + 1],
                          "li_mid": float(x["li"].median()), "n": len(x),
                          "p_star_median": float(x["p_star"].median()),
                          "p_star_q25": float(x["p_star"].quantile(0.25)),
                          "p_star_q75": float(x["p_star"].quantile(0.75)),
                          "savant_median": float(x["savant_p"].median()),
                          "savant_li_median": float(x["savant_li_p"].median())})
    curve = pd.DataFrame(curve)
    by_inn = (opp[fin].assign(inn=np.minimum(opp.loc[fin, "inning"], 10))
              .groupby(["inn", "r_self"])["p_star"].median().unstack())
    for r_self in (1, 2):
        lo_li = curve[(curve["r_left"] == r_self)].head(1)
        hi_li = curve[(curve["r_left"] == r_self)].tail(1)
        if len(lo_li) and len(hi_li):
            print(f"  {r_self} left: median p* {lo_li['p_star_median'].iloc[0]:.2f} at LI ≈ "
                  f"{lo_li['li_mid'].iloc[0]:.1f} → {hi_li['p_star_median'].iloc[0]:.2f} at LI ≈ "
                  f"{hi_li['li_mid'].iloc[0]:.1f}")
    corr_li = float(np.corrcoef(np.log(opp.loc[fin, "li"].clip(1e-3)), opp.loc[fin, "p_star"].clip(0, 1))[0, 1])
    print(f"  across {int(fin.sum()):,} real opportunities, p* falls as LI rises "
          f"(correlation with log LI {corr_li:+.2f})")
    R.update(examples=ex, curve=curve, p_by_inning=by_inn, corr_li=corr_li, n_opp=int(fin.sum()))

    # ================================================================ 6. run levels
    banner("6. Run production at three levels — and what turns it into wins")
    no_out, p1, RE = run_tables(rm)

    def walk_noout(X, o, b):
        return 1.0

    def k_noout(X, o, b):
        return 0.0

    def walk_p1(X, o, b):
        return 1.0 if WALK_RUNS[b] else X[..., o, WALK_BASES[b], 0]

    def k_p1(X, o, b):
        return 0.0 if o == 2 else X[..., o + 1, b, 0]

    f_noout = flip_generic(no_out, walk_noout, k_noout)
    f_p1 = flip_generic(p1, walk_p1, k_p1)
    ex_idx = (ex["_s"].to_numpy(int), ex["_o"].to_numpy(int), ex["_b"].to_numpy(int), ex["_c"].to_numpy(int))
    ex["at_bat_safe_pts"] = 100 * f_noout[ex_idx]
    ex["inning_score1_pts"] = 100 * f_p1[ex_idx]
    ex = ex.drop(columns=["_s", "_o", "_b", "_c"])
    R["examples"] = ex
    print("  the ladder for the example situations (ball minus strike): "
          + "; ".join(f"#{k + 1}: safe {r['at_bat_safe_pts']:+.0f} pts, runs {r['run_value']:+.2f}, "
                      f"score ≥1 {r['inning_score1_pts']:+.0f} pts, win {r['swing_pts']:+.1f} pts"
                      for k, r in ex.head(4).iterrows()))
    idx = (opp["half"].to_numpy(), opp["outs"].to_numpy(), opp["bases"].to_numpy(), opp["count"].to_numpy())
    opp["f_noout"], opp["f_p1"], opp["f_runs"] = f_noout[idx], f_p1[idx], flip[idx]
    late_close = (opp["inning"] >= 7) & (opp["diff"].abs() <= 1)
    ctx = np.where(opp["inning"] <= 6, "innings 1–6",
                   np.where(late_close, "7th or later, within 1 run", "7th or later, 2+ runs apart"))
    opp["context"] = ctx
    b9_tied = (opp["inning"] >= 9) & (opp["half"] == 1) & (opp["diff"] == 0)
    ladder = []
    for name in CONTEXTS + ["bottom 9th or later, tied", "all"]:
        d_ = opp if name == "all" else (opp[b9_tied] if name.startswith("bottom 9th") else
                                        opp[opp["context"] == name])
        y = d_["swing"].to_numpy()
        row = {"context": name, "n": len(d_), "swing_pts": 100 * float(np.mean(y)),
               "at_bat_no_out": float(d_["f_noout"].mean()), "runs_RE288": float(d_["f_runs"].mean()),
               "inning_score_1plus": float(d_["f_p1"].mean()), "LI": float(d_["li"].mean())}
        for key, xcol in (("r2_runs", "f_runs"), ("r2_score1", "f_p1"), ("r2_noout", "f_noout")):
            for tag, x in (("", d_[xcol].to_numpy()), ("_x_LI", d_[xcol].to_numpy() * d_["li"].to_numpy())):
                okk = np.isfinite(x) & np.isfinite(y)
                row[key + tag] = float(np.corrcoef(x[okk], y[okk])[0, 1] ** 2) if okk.sum() > 10 else np.nan
        ladder.append(row)
    ladder = pd.DataFrame(ladder)
    for _, r in ladder.iterrows():
        print(f"  {r['context']:<28} n={int(r['n']):>7,}: share of the WP swing explained (R²) by — runs "
              f"{r['r2_runs']:.2f} (×LI {r['r2_runs_x_LI']:.2f}); P(score ≥1) {r['r2_score1']:.2f} "
              f"(×LI {r['r2_score1_x_LI']:.2f}); P(batter safe) {r['r2_noout']:.2f} (×LI {r['r2_noout_x_LI']:.2f})")
    # FanGraphs/Clemens check on RE288 flips
    w_side = rm.n_cell.sum((1, 2, 3))
    flip_pooled = (flip * (w_side / w_side.sum())[:, None, None, None]).sum(0)
    clem = []
    for (bname, o, cname), pub in CLEMENS.items():
        b = BASE_NAMES.index(bname)
        c = 3 * int(cname[0]) + int(cname[2])
        clem.append({"runners": bname, "outs": o, "count": cname, "ours_2026": float(flip_pooled[o, b, c]),
                     "fangraphs_published": pub})
    clem = pd.DataFrame(clem)
    print("  RE288 run value of a flip, ours vs FanGraphs (Clemens, Apr 2026): "
          + "; ".join(f"{r['runners']} {r['outs']} out {r['count']} {r['ours_2026']:.2f} vs {r['fangraphs_published']:.2f}"
                      for _, r in clem.iterrows()))
    # per-run conversion
    early = opp[opp["context"] == "innings 1–6"]
    wpr = float((early["swing"] / np.maximum(early["f_runs"] * early["li"], 1e-9)).median())
    print(f"  in innings 1–6 one run of call value × LI is worth {pts(wpr)} of win probability "
          f"(≈ 1 win per {1 / max(wpr, 1e-9):.1f} runs)")
    R.update(ladder=ladder, clemens=clem, wpr=wpr)

    # ================================================================ 7. policies
    banner("7. Challenge rules compared — wins a season, against a team that plays well")
    policies = [("optimal (this MDP)", FromSolution(sol)),
                ("play for runs (run-differential MDP)", FromSolution(sol_r, "runs")),
                ("Statcast breakeven × LI", Breakeven(flip, SAVANT_COST, leverage=li_grid)),
                ("Statcast breakeven (0.2 runs)", Breakeven(flip, SAVANT_COST)),
                ("challenge when p(s) ≥ 0.5", Constant(0.5)),
                ("never challenge", Constant(NEVER))]
    pol_rows = []
    for name, pol in policies:
        t0 = time.time()
        loss = {"loss": 0.0, "home_loss": 0.0, "away_loss": 0.0} if name.startswith("optimal") \
            else wp_loss(game, sol, pol)
        sim = Simulator(pm, game, {"home": pol, "away": pol}, seed=args.seed).run(args.sims)
        pol_rows.append({"policy": name, "wp_lost_per_game": loss["loss"],
                         "wins_lost_per_season": loss["loss"] * GAMES,
                         "home_loss": loss["home_loss"], "away_loss": loss["away_loss"],
                         "challenges_per_team_game": sim["challenges_per_team_game"],
                         "success_rate": sim["success_rate"],
                         "overturns_per_team_game": sim["overturns_per_team_game"],
                         "wrong_calls_missed_per_team_game": sim["wrong_missed_per_team_game"],
                         "wrong_calls_per_team_game": sim["wrong_calls_per_team_game"],
                         "left_at_end": sim["left_at_end"], "sim_home_win": sim["home_win"],
                         "sim_se": sim["se_home_win"]})
        print(f"  {name:<38} loses {pts(loss['loss'])} a game = {loss['loss'] * GAMES:5.2f} wins a "
              f"season; {sim['challenges_per_team_game']:.2f} challenges a game, "
              f"{pct(sim['success_rate'], 0)} overturned  ({time.time() - t0:.0f}s)")
    pols = pd.DataFrame(pol_rows)
    sim_opt = pol_rows[0]
    z = (sim_opt["sim_home_win"] - sol.pregame()) / max(sim_opt["sim_se"], 1e-9)
    print(f"  check: {args.sims:,} simulated games with the optimal rule — home team won "
          f"{pct(sim_opt['sim_home_win'], 2)} vs {pct(sol.pregame(), 2)} from backward induction "
          f"(z = {z:+.1f})")
    ch_real = val[val["challenged"] & np.isfinite(val["p_s"])]
    real_use = {"challenges_per_team_game": len(ch_real) / (2 * n_games),
                "success_rate": float(ch_real["overturned"].mean()) if len(ch_real) else np.nan,
                "overturns_per_team_game": float(ch_real["overturned"].sum()) / (2 * n_games),
                "wrong_calls_per_team_game": float(val.loc[val["opportunity"], "p_s"].sum()) / (2 * n_games)}
    print(f"  2026 as played: {real_use['challenges_per_team_game']:.2f} challenges a team-game, "
          f"{pct(real_use['success_rate'], 0)} overturned, {real_use['overturns_per_team_game']:.2f} "
          f"overturns of {real_use['wrong_calls_per_team_game']:.2f} expected wrong calls")
    R.update(policies=pols, real_use=real_use, sim_z=z)

    # ================================================================ 8. real decisions
    banner("8. The 2026 decisions against the thresholds (a preview of RQ3)")
    p_ = opp["p_s"].to_numpy()
    H_, Dl_ = opp["hold"].to_numpy(), opp["swing"].to_numpy()
    model_ch = fin.to_numpy() & (p_ >= opp["p_star"].to_numpy())
    did = opp["challenged"].to_numpy(bool)
    gain = p_ * (Dl_ + H_) - H_                          # Q_challenge − Q_hold, for the decider
    regret = np.where(did, np.maximum(-gain, 0), np.maximum(gain, 0))
    regret = np.where(np.isfinite(regret), regret, 0.0)
    opp["model_challenge"], opp["regret"] = model_ch, regret
    opp["savant_challenge"] = p_ >= opp["savant_p"].to_numpy()
    tab = pd.crosstab(pd.Series(did, name="challenged"), pd.Series(model_ch, name="model says"))
    both = int(tab.loc[True, True]) if (True in tab.index and True in tab.columns) else 0
    agree = float((did == model_ch).mean())
    print(f"  {len(opp):,} opportunities: teams challenged {int(did.sum()):,}, the thresholds say "
          f"{int(model_ch.sum()):,}; both {both:,}")
    over_ch = did & ~model_ch
    under = ~did & model_ch
    print(f"  challenges below the bar: {int(over_ch.sum()):,} (cost {regret[over_ch].sum():.1f} wins); "
          f"chances passed up above it: {int(under.sum()):,} (cost {regret[under].sum():.1f} wins)")
    clear_missed = int((~did & (p_ >= 0.9)).sum())
    print(f"  calls with p(s) ≥ 90% left alone with a challenge in hand: {clear_missed:,}")
    n_teams = max(int(pd.Series(np.where(opp["chall_home"], opp["home_team"], opp["away_team"])).nunique()), 1)
    print(f"  total {regret.sum():.1f} wins = {regret.sum() / n_teams:.2f} per team over the season, "
          f"if every decision had been made with p(s) known (the full-information benchmark)")
    sav = opp["savant_challenge"].to_numpy()
    either = sav | model_ch
    sav_agree = float((sav[either] == model_ch[either]).mean()) if either.any() else np.nan
    made_ok = float(model_ch[did].mean()) if did.any() else np.nan
    taken = float(did[model_ch].mean()) if model_ch.any() else np.nan
    print(f"  of the challenges teams made, {pct(made_ok)} cleared the bar; of the chances above the "
          f"bar, teams took {pct(taken)}")
    print(f"  where either rule would challenge, Statcast's breakeven and the thresholds agree "
          f"{pct(sav_agree)} of the time (sav only {int((sav & ~model_ch).sum()):,}, thresholds only "
          f"{int((model_ch & ~sav).sum()):,})")
    role = opp["role"].str.lower()
    by_role = []
    def dec_row(name, d_, opps=True):
        ch_, mc_ = d_["challenged"].to_numpy(bool), d_["model_challenge"].to_numpy(bool)
        return {"group": name, "opportunities": len(d_) if opps else np.nan, "challenged": int(ch_.sum()),
                "above_bar": int(mc_.sum()) if opps else np.nan,
                "made_above_bar": float(mc_[ch_].mean()) if ch_.any() else np.nan,
                "above_bar_taken": float(ch_[mc_].mean()) if (opps and mc_.any()) else np.nan,
                "regret_wins": float(d_["regret"].sum()),
                "clear_missed": int((~ch_ & (d_["p_s"].to_numpy() >= 0.9)).sum()) if opps else np.nan}

    by_role.append(dec_row("batters' side (called strikes)", opp[opp["call_strike"]]))
    by_role.append(dec_row("fielders' side (called balls)", opp[~opp["call_strike"]]))
    for name in ("batter", "catcher", "pitcher"):
        d_ = opp[opp["challenged"] & (role == name)]
        if len(d_):
            by_role.append(dec_row(f"challenges by {name}s", d_, opps=False))
    if "split" in opp.columns and opp["split"].astype(str).str.len().gt(0).any():
        for name, d_ in opp.groupby(opp["split"].astype(str)):
            if name and name != "nan":
                by_role.append(dec_row(f"cohort split: {name}", d_))
    by_role = pd.DataFrame(by_role)
    team = np.where(opp["chall_home"], opp["home_team"], opp["away_team"])
    by_team = (opp.assign(team=team).groupby("team")
               .agg(opportunities=("p_s", "size"), challenged=("challenged", "sum"),
                    model_says=("model_challenge", "sum"), regret_wins=("regret", "sum"))
               .sort_values("regret_wins").reset_index())
    pb = np.clip(np.searchsorted(np.linspace(0, 1, 21), p_, side="right") - 1, 0, 19)
    dcurve = (pd.DataFrame({"bin": pb, "actual": did, "model": model_ch,
                            "savant": opp["savant_challenge"].to_numpy()})
              .groupby("bin").agg(n=("actual", "size"), actual=("actual", "mean"),
                                  model=("model", "mean"), savant=("savant", "mean")).reset_index())
    dcurve["p_lo"], dcurve["p_hi"] = dcurve["bin"] / 20, (dcurve["bin"] + 1) / 20
    R.update(decisions={"n": len(opp), "did": int(did.sum()), "model": int(model_ch.sum()), "both": both,
                        "agree": agree, "over": int(over_ch.sum()), "over_cost": float(regret[over_ch].sum()),
                        "under": int(under.sum()), "under_cost": float(regret[under].sum()),
                        "clear_missed": clear_missed, "total": float(regret.sum()),
                        "per_team": float(regret.sum() / n_teams), "made_ok": made_ok, "taken": taken,
                        "savant_agree": sav_agree, "savant_only": int((sav & ~model_ch).sum()),
                        "model_only": int((model_ch & ~sav).sum())},
             by_role=by_role, by_team=by_team, dcurve=dcurve)

    # ================================================================ 9. umpires
    banner("9. Umpires — does a less accurate plate umpire change the math?")
    ump, _ = umpire_table(val)
    ump_ok = ump[ump["games"] >= 10].copy()
    tiers = []
    if len(ump_ok) >= 3:
        ump_ok["tier"] = pd.qcut(ump_ok["excess_per_100"].rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
        tier_of = dict(zip(ump_ok["umpire_id"], ump_ok["tier"]))
        val["tier"] = val["umpire_id"].map(tier_of)
        for t in (0, 1, 2):
            d_ = val[val["tier"] == t]
            if not len(d_):
                continue
            cal = pd.DataFrame({"count": d_["count"], "strike": d_["call_strike"].astype(int), "p": d_["p_s"]})
            pm_t = pm.with_p_supply(*PitchModel.p_supply(cal))
            sol_t = GameMDP(pm_t).solve()
            pre_t = sol_t.start[(1, 0)][D]
            tiers.append({"tier": TIER_LABEL[t], "umpires": int((ump_ok["tier"] == t).sum()),
                          "games": int(d_["game_pk"].nunique()),
                          "wrong_per_game": float(d_["p_s"].sum() / d_["game_pk"].nunique()),
                          "first_challenge_pts": 100 * ((pre_t[1, 2] - pre_t[0, 2]) + (pre_t[2, 0] - pre_t[2, 1])) / 2,
                          "second_challenge_pts": 100 * ((pre_t[2, 2] - pre_t[1, 2]) + (pre_t[2, 1] - pre_t[2, 2])) / 2,
                          "p_star_example": float(sol_t.lookup("thr_S", 1, 0, 0, 0, 0, 0, 1, 2))})
            r = tiers[-1]
            print(f"  {r['tier']:<22} {r['wrong_per_game']:.2f} wrong calls a game → first challenge worth "
                  f"{r['first_challenge_pts']:.2f} pts, second {r['second_challenge_pts']:.2f} pts; "
                  f"1st-inning bar with 1 left {num(r['p_star_example'], 2)}")
    tiers = pd.DataFrame(tiers)
    R.update(tiers=tiers)

    # ================================================================ save
    banner("Saving")
    out = MODELS / "mdp"
    DOCS.mkdir(parents=True, exist_ok=True)
    csvs = {"week11_pitch_model_events.csv": ev_tab, "week11_run_expectancy_check.csv": re_check,
            "week11_leverage_by_inning.csv": li_inn, "week11_leverage_ninth.csv": li_9,
            "week11_leverage_bands.csv": bands, "week11_hold_value.csv": hold,
            "week11_threshold_examples.csv": ex, "week11_threshold_curve.csv": curve,
            "week11_run_levels.csv": ladder, "week11_flip_vs_fangraphs.csv": clem,
            "week11_policies.csv": pols, "week11_real_decisions_by_side.csv": by_role,
            "week11_real_decisions_by_team.csv": by_team, "week11_decision_curve.csv": dcurve,
            "week11_umpire_tiers.csv": tiers}
    # (week11_summary.json below feeds the dashboard's Progress tab)
    for name, df in csvs.items():
        df.to_csv(DOCS / name, index=False)
    d = R["decisions"]
    summary = {"pregame_home": sol.pregame(), "value_two": v_all, "value_first": v_first,
               "value_second": v_second, "wins_two_per_season": v_all * GAMES, "rpw": rpw,
               "li_below_1": shares["below 1"], "li_above_2": shares["high (> 2)"], "li_max": li_max,
               "policies": {r["policy"]: r["wins_lost_per_season"] for _, r in pols.iterrows()},
               "decisions": d, "tests_passed": os.environ.get("ABS_W11_TESTS", ""), "demo": demo}
    (DOCS / "week11_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"  models → {out}/  ·  tables → {DOCS}/week11_*.csv")

    if not args.no_figures:
        import week11_figures as W11
        FIGS.mkdir(parents=True, exist_ok=True)
        W11.make_all(FIGS, DATA_DIR, R=R, li_pa=lev.pa_li, opp=opp[["li", "p_star", "r_self", "savant_p",
                                                                       "swing", "f_runs", "f_p1", "context"]])
        print(f"  figures → {FIGS}/")

    write_report(R, demo, DOCS / "WEEK11_REPORT.md", args)
    print(f"\nWEEK11_REPORT.md written ({time.time() - t_start:.0f}s in all)")


# =================================================================== the report
def write_report(R, demo, path, args):
    v = R["versions"]
    L = []
    add = L.append
    add("# Week 11 report — the challenge decision as an MDP")
    add("")
    add(f"*Generated {dt.date.today().isoformat()} by `scripts/21_week11_mdp.py`. p(s) {v['p']} · "
        f"V(s) {v['v']} · cohort {v['cohort']}. Read it with WEEK11.md beside it.*")
    if demo:
        add("")
        add("> **DEMO DATA.** Every number here comes from a simulated season, not 2026.")
    add("")
    add("## 1. The pitch-level game")
    i = R["pm_info"]
    add(f"Every pitch of the season ({i['n_pitches']:,}) sets the odds of what the next pitch does at "
        f"each count; {i['n_in_play']:,} balls in play, {i['n_strikeouts']:,} strikeouts and "
        f"{i['n_walks_hbp']:,} walks/HBP set where the runners end up; {i['n_runner_moves']:,} runner "
        f"moves between pitches ({pct(i['runner_move_rate'])} of pitches) are in too.")
    add("")
    add(md_table(R["re_check"], fmt={"mean_error": lambda x: f"{x:+.4f}", "mean_abs_error": lambda x: f"{x:.4f}",
                                     "re24_empty_0": lambda x: f"{x:.3f}", "re24_loaded_0": lambda x: f"{x:.3f}"},
                 rename={"mean_error": "error vs the season (runs)", "mean_abs_error": "typical error",
                         "re24_empty_0": "RE24 bases empty, 0 out", "re24_loaded_0": "RE24 bases loaded, 0 out"}))
    add("")
    vc = R["v_check"]
    add(f"Win probability from the same game (as played, nobody challenging) against Week 10's locked V(s), "
        f"on {vc['n']:,} real pitches: mean difference **{vc['mean_abs_pts']:.2f} pts** (95% of pitches within "
        f"{vc['p95_abs_pts']:.2f}), correlation {vc['corr']:.5f}, Brier score {vc['brier_mdp']:.4f} vs "
        f"{vc['brier_w10']:.4f}.")
    add("")
    add(f"The 2026 challenges added **{R['challenge_runs']:+.4f} runs** per state to the batting team's "
        f"inning (as played − umpire calls only). Mistakes on offer: **{R['wrong_pg']:.2f}** expected "
        f"wrong calls a game, {R['clear_pg']:.2f} of them with p(s) ≥ 90%.")
    add("")
    add("## 2. Leverage — Tango's LI from our V(s)")
    b = dict(zip(R["bands"]["band"], R["bands"]["share_of_PAs"]))
    add(f"Below 1: **{pct(b['below 1'])}** of plate appearances (FanGraphs: about 60%). Above 2: "
        f"**{pct(b['high (> 2)'])}** (about 10%). Highest: **{R['li_max']:.2f}**.")
    add("")
    add(md_table(R["li_inn"], ["inning", "li_2026", "li_2013_published", "n_pa"],
                 fmt={"li_2026": lambda x: f"{x:.2f}", "li_2013_published": lambda x: num(x, 2),
                      "inning": lambda x: "10+" if x == 10 else str(int(x)), "n_pa": lambda x: f"{int(x):,}"},
                 rename={"li_2026": "average LI (ours)", "li_2013_published": "2013, Tango's LI (published)",
                         "n_pa": "plate appearances"}))
    add("")
    add(md_table(R["li_9"], ["half", "li_2026_all", "li_2013_all", "li_2026_first_pa", "li_2013_first_play"],
                 fmt={c: (lambda x: f"{x:.2f}") for c in ("li_2026_all", "li_2013_all",
                                                          "li_2026_first_pa", "li_2013_first_play")},
                 rename={"half": "9th inning", "li_2026_all": "LI, ours", "li_2013_all": "2013 published",
                         "li_2026_first_pa": "first PA, ours", "li_2013_first_play": "first play, 2013"}))
    add("")
    add("## 3. V(s, r) — what a challenge is worth")
    add(f"Home team wins **{pct(R['pregame'])}** with both teams choosing well ({pct(R['pregame_none'])} "
        f"with no challenges). Solved in {R['solve_seconds']:.1f}s.")
    add("")
    add(f"At the first pitch a team's two challenges are worth **{pts(R['v_all'])}** "
        f"(= {R['v_all'] * GAMES:.1f} wins a season): the first **{pts(R['v_first'])}**, the second "
        f"**{pts(R['v_second'])}**.")
    add("")
    h = R["hold"][R["hold"]["role"] == "fielding"].copy()
    h["H1 (pts)"] = 100 * h["H1_win"]
    h["H2 (pts)"] = 100 * h["H2_win"]
    h["H1 (runs)"], h["H2 (runs)"] = h["H1_runs"], h["H2_runs"]
    add("Value of holding at the start of each half, tied game, for the team in the field "
        "(H1 = the last challenge, H2 = the second):")
    add("")
    add(md_table(h[["half_label", "H1 (pts)", "H2 (pts)", "H1 (runs)", "H2 (runs)"]],
                 fmt={**{c: (lambda x: f"{x:.2f}") for c in ("H1 (pts)", "H2 (pts)")},
                      **{c: (lambda x: f"{x:.3f}") for c in ("H1 (runs)", "H2 (runs)")}},
                 rename={"half_label": "start of"}))
    add("")
    add(f"Runs per win implied by the two currencies (innings 1–6): **{R['rpw']:.1f}** (rule of thumb ≈ 10).")
    add("")
    add("## 4. Runs")
    add(f"At the first pitch: last challenge **{R['h1_runs_start']:.2f} runs**, second "
        f"**{R['h2_runs_start']:.2f} runs**. Statcast prices a failed challenge at {SAVANT_COST} runs.")
    add("")
    add("## 5. Thresholds")
    ex = R["examples"].copy()
    add(md_table(ex, ["situation", "call", "p_star", "swing_pts", "hold_pts", "LI", "run_value", "savant_p"],
                 fmt={"p_star": lambda x: num(x, 2), "swing_pts": lambda x: f"{x:.2f}",
                      "hold_pts": lambda x: f"{x:.2f}", "LI": lambda x: f"{x:.1f}",
                      "run_value": lambda x: f"{x:.3f}", "savant_p": lambda x: num(x, 2)},
                 rename={"p_star": "p* (MDP)", "swing_pts": "Δ swing (pts)", "hold_pts": "H hold (pts)",
                         "run_value": "RE288 flip (runs)", "savant_p": "Statcast breakeven"}))
    add("")
    add("The same situations, one level at a time (ball minus strike, for the batting team): the batter's "
        "chance of not making an out, expected runs to the end of the inning, the chance of scoring at "
        "least once more, and win probability.")
    add("")
    add(md_table(ex, ["situation", "at_bat_safe_pts", "run_value", "inning_score1_pts", "swing_pts", "LI"],
                 fmt={"at_bat_safe_pts": lambda x: f"{x:+.1f} pts", "run_value": lambda x: f"{x:+.3f} runs",
                      "inning_score1_pts": lambda x: f"{x:+.1f} pts", "swing_pts": lambda x: f"{x:.2f} pts",
                      "LI": lambda x: f"{x:.1f}"},
                 rename={"at_bat_safe_pts": "at-bat: batter safe", "run_value": "inning: expected runs",
                         "inning_score1_pts": "inning: scores ≥ 1", "swing_pts": "game: win probability"}))
    add("")
    add(f"Across {R['n_opp']:,} real opportunities p* falls as leverage rises (correlation with log LI "
        f"{R['corr_li']:+.2f}). Figure w11_01.")
    add("")
    add(md_table(R["curve"], ["r_left", "li_lo", "li_hi", "n", "p_star_median", "p_star_q25", "p_star_q75",
                              "savant_median", "savant_li_median"],
                 fmt={"r_left": lambda x: str(int(x)), "li_lo": lambda x: f"{x:.2f}", "li_hi": lambda x: f"{x:.2f}",
                      "n": lambda x: f"{int(x):,}", **{c: (lambda x: f"{x:.2f}") for c in
                      ("p_star_median", "p_star_q25", "p_star_q75", "savant_median", "savant_li_median")}},
                 rename={"r_left": "challenges left", "li_lo": "LI from", "li_hi": "LI to", "n": "opportunities",
                         "p_star_median": "median p*", "p_star_q25": "p* 25th pct", "p_star_q75": "p* 75th pct",
                         "savant_median": "Statcast breakeven", "savant_li_median": "breakeven × LI"}))
    add("")
    add("## 6. Run production at three levels")
    add("Average flip (ball minus strike) at each level, by game situation:")
    add("")
    add(md_table(R["ladder"], ["context", "n", "at_bat_no_out", "runs_RE288", "inning_score_1plus", "swing_pts", "LI"],
                 fmt={"n": lambda x: f"{int(x):,}", "at_bat_no_out": lambda x: f"{100 * x:+.1f} pts",
                      "runs_RE288": lambda x: f"{x:+.3f}", "inning_score_1plus": lambda x: f"{100 * x:+.1f} pts",
                      "swing_pts": lambda x: f"{x:.2f} pts", "LI": lambda x: f"{x:.2f}"},
                 rename={"n": "opportunities", "at_bat_no_out": "at-bat: batter safe",
                         "runs_RE288": "inning: expected runs", "inning_score_1plus": "inning: scores ≥ 1",
                         "swing_pts": "game: win probability", "LI": "average LI"}))
    add("")
    add("How much of the win-probability swing each run measure explains (R²), alone and × LI:")
    add("")
    add(md_table(R["ladder"], ["context", "r2_noout", "r2_noout_x_LI", "r2_runs", "r2_runs_x_LI", "r2_score1",
                               "r2_score1_x_LI"],
                 fmt={c: (lambda x: f"{x:.2f}") for c in ("r2_noout", "r2_noout_x_LI", "r2_runs", "r2_runs_x_LI",
                                                          "r2_score1", "r2_score1_x_LI")},
                 rename={"r2_noout": "batter safe", "r2_noout_x_LI": "batter safe × LI",
                         "r2_runs": "expected runs", "r2_runs_x_LI": "expected runs × LI",
                         "r2_score1": "scores ≥ 1", "r2_score1_x_LI": "scores ≥ 1 × LI"}))
    add("")
    add(f"One run of call value × LI ≈ **{pts(R['wpr'])}** of win probability in innings 1–6 "
        f"(1 win ≈ {1 / max(R['wpr'], 1e-9):.1f} runs).")
    add("")
    add("RE288 run value of flipping a call, ours vs FanGraphs (Clemens, April 2026):")
    add("")
    add(md_table(R["clemens"], fmt={"ours_2026": lambda x: f"{x:.2f}", "fangraphs_published": lambda x: f"{x:.2f}"},
                 rename={"ours_2026": "ours", "fangraphs_published": "FanGraphs (published)"}))
    add("")
    add("## 7. Challenge rules compared")
    p = R["policies"]
    add(md_table(p, ["policy", "wins_lost_per_season", "challenges_per_team_game", "success_rate",
                     "overturns_per_team_game", "wrong_calls_missed_per_team_game", "left_at_end"],
                 fmt={"wins_lost_per_season": lambda x: f"{x:.2f}", "success_rate": lambda x: pct(x, 0),
                      "challenges_per_team_game": lambda x: f"{x:.2f}", "overturns_per_team_game": lambda x: f"{x:.2f}",
                      "wrong_calls_missed_per_team_game": lambda x: f"{x:.2f}", "left_at_end": lambda x: f"{x:.2f}"},
                 rename={"wins_lost_per_season": "wins lost / 162", "challenges_per_team_game": "challenges / game",
                         "success_rate": "overturned", "overturns_per_team_game": "overturns / game",
                         "wrong_calls_missed_per_team_game": "wrong calls left standing / game",
                         "left_at_end": "challenges left at the end"}))
    add("")
    ru = R["real_use"]
    add(f"2026 as played: {ru['challenges_per_team_game']:.2f} challenges a team-game, "
        f"{pct(ru['success_rate'], 0)} overturned, {ru['overturns_per_team_game']:.2f} overturns of "
        f"{ru['wrong_calls_per_team_game']:.2f} expected wrong calls. Simulation check: z = {R['sim_z']:+.1f}.")
    add("")
    add("## 8. The 2026 decisions (preview of RQ3)")
    d = R["decisions"]
    add(f"{d['n']:,} opportunities; teams challenged {d['did']:,}, the thresholds say {d['model']:,}, "
        f"both {d['both']:,}. Of the challenges teams made, **{pct(d['made_ok'])}** cleared the bar; of "
        f"the chances above the bar, teams took **{pct(d['taken'])}**. Below-the-bar challenges: "
        f"{d['over']:,} ({d['over_cost']:.1f} wins). Passed-up chances above it: {d['under']:,} "
        f"({d['under_cost']:.1f} wins). Calls with p(s) ≥ 90% left alone: {d['clear_missed']:,}. "
        f"Total gap to the full-information benchmark: **{d['total']:.1f} wins** "
        f"({d['per_team']:.2f} per team). Where either rule would challenge, Statcast's breakeven and the "
        f"thresholds agree {pct(d['savant_agree'])} of the time (Statcast only {d['savant_only']:,}; "
        f"thresholds only {d['model_only']:,}).")
    add("")
    intf = lambda x: "—" if not np.isfinite(x) else f"{int(x):,}"   # noqa: E731
    add(md_table(R["by_role"], fmt={"opportunities": intf, "challenged": intf, "above_bar": intf,
                                    "made_above_bar": lambda x: pct(x), "above_bar_taken": lambda x: pct(x),
                                    "regret_wins": lambda x: f"{x:.1f}", "clear_missed": intf},
                 rename={"above_bar": "above the bar", "made_above_bar": "challenges above the bar",
                         "above_bar_taken": "above-bar chances taken", "regret_wins": "gap (wins)",
                         "clear_missed": "p ≥ 90% left alone"}))
    add("")
    add("## 9. Umpire tiers")
    if len(R["tiers"]):
        add(md_table(R["tiers"], fmt={"wrong_per_game": lambda x: f"{x:.2f}",
                                      "first_challenge_pts": lambda x: f"{x:.2f}",
                                      "second_challenge_pts": lambda x: f"{x:.2f}",
                                      "p_star_example": lambda x: num(x, 2)},
                     rename={"tier": "plate umpire", "wrong_per_game": "expected wrong calls / game",
                             "first_challenge_pts": "first challenge (pts)",
                             "second_challenge_pts": "second challenge (pts)",
                             "p_star_example": "bar, 1st inning 0-0, 1 left"}))
    else:
        add("Not enough umpires with 10+ games.")
    add("")
    add("## 10. What Spring adds")
    for line in SPRING:
        add(f"- {line}")
    add("")
    add("Figures: `figures/week11/w11_01_thresholds.png` (the money figure), `w11_02_hold_value`, "
        "`w11_03_leverage`, `w11_04_run_levels`, `w11_05_policies`, `w11_06_real_decisions`.")
    path.write_text("\n".join(L) + "\n")


SPRING = [
    "**Who is batting and pitching.** V(s, r) is an average-team model, like every published win "
    "expectancy table. Spring conditions the pitch model on the batter, pitcher and catcher (a 2-strike "
    "count means more against a contact hitter).",
    "**What players can see.** p(s) is what Hawk-Eye knows; a catcher has two seconds and a view. The "
    "optimal policy here is the full-information benchmark. Spring adds a perception model (p seen with "
    "noise, by role and by distance from the zone) so the thresholds become rules a player can use.",
    "**The September test.** Evaluate the policies on the held-out month (RQ3) with p(s) from the lock, "
    "and the regret by team with game-level bootstrap intervals.",
    "**Umpire-specific supply of mistakes** for the game in progress (we only compare thirds here), and a "
    "calibrated p(s) (the lock is uncalibrated; thresholds are only as good as the probabilities fed to them).",
    "**Score cap and extras.** The lead is capped at ±10 (a 4·10⁻⁶ effect on win probability) and every "
    "extra inning is treated alike; position players pitching are left out of the decision.",
]


if __name__ == "__main__":
    main()
