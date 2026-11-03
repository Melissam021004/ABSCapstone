#!/usr/bin/env python3
"""
20_week10_challenge_value.py — Week 10, part 2. What every 2026 challenge was worth.

Script 19 built and locked V(s). This script applies it — with Week 9's locked
p(s) — to every called pitch of the season, and answers the questions you and
your professor raised:

    1. WORTH      what a call is worth in win probability and in runs, and the most
                  valuable overturns of the season
    2. INNING     does an overturn give the batting team more runs before the third
                  out? (overturned vs upheld challenges: what happened vs predicted)
    3. GAME       the same for who won the game
    4. AHEAD      how much is still at stake later in the game — the raw material of
                  "the value of holding a challenge" (Week 11 turns it into a policy)
    5. SEASON     wins each team added with challenges, and wins left on the table
    6. SERIES     challenges that swung a series
    7. PLAYERS    do some players challenge better? does WHO challenges change what
                  ABS rules? (if not, it stays out of p(s); it belongs in the policy)
    8. UMPIRES    do some umpires miss more? does it change the overturn chance, or
                  how many chances a team gets?

    python scripts/20_week10_challenge_value.py
    python scripts/20_week10_challenge_value.py --perm 199     # faster permutation tests

Reads   the cohort (Week 6), models/production (p(s), Week 9), models/winprob (V(s),
        script 19), data/raw/statcast (runs and results after each pitch)
Writes  docs/WEEK10_VALUE_REPORT.md, docs/week10_value_*.csv, figures/week10/w10_07..12
"""

import argparse
import datetime as dt
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
from src.abs_gamestate import BASE_NAMES, COUNT_NAMES, build_states, load_pitches  # noqa: E402
from src.abs_pipeline import resolve_fingerprint                     # noqa: E402
from src.abs_predict import load_production                          # noqa: E402
from src.abs_value import (KEY, challenger_table, cluster_signal,    # noqa: E402
                           game_mean_ci, league_totals, natural_experiment, series_effects,
                           series_table, split_half, summarise_ahead, team_season,
                           umpire_table, valuation_table, value_ahead)
from src.abs_winprob import load_winprob                             # noqa: E402

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
CONFIG = Path(os.environ.get("ABS_CONFIG_DIR", "config"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "week10"
ROLE_ORDER = ["batter", "catcher", "pitcher"]
TIER_LABEL = {0: "most accurate third", 1: "middle third", 2: "least accurate third"}


def banner(msg):
    print(f"\n{msg}\n{'-' * len(msg)}")


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def pts(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f} pts"


def is_demo(path):
    return "demo" in str(path).lower()


def half_label(i):
    if i >= 18:
        return "extras"
    return f"{i // 2 + 1} {'top' if i % 2 == 0 else 'bot'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=999, help="permutations per skill test")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    demo = is_demo(DATA_DIR)

    # ------------------------------------------------------------ 0. inputs
    banner("0. Inputs")
    pm = load_production()
    wp = load_winprob()
    raw = load_cohort(DATA_DIR)
    fingerprint, fp_note = resolve_fingerprint(raw, CONFIG)
    cohort, check = add_features(raw, return_check=True)
    for line in describe_call_check(check):
        print(f"  {line}")
    print(f"  p(s) {pm.manifest.get('version')} · V(s) {wp.version} · cohort {fingerprint}"
          + (f" ({fp_note})" if fp_note else ""))
    val = valuation_table(cohort, pm, wp)
    pitches, games = build_states(load_pitches(DATA_DIR, (2026,))[0])
    extra = pitches[KEY + ["runs_rest_inning", "home_win"]]
    val = val.merge(extra, on=KEY, how="left")
    val["chall_won"] = np.where(val["chall_home"], val["home_win"], 1 - val["home_win"])
    names = {}
    if "challenger_name" in cohort.columns:
        names = (cohort.dropna(subset=["challenger_id"]).drop_duplicates("challenger_id")
                 .set_index("challenger_id")["challenger_name"].to_dict())
    ok = np.isfinite(val["p_s"])
    ch = val[val["challenged"] & ok].copy()
    print(f"  {len(val):,} called pitches valued ({int((~ok).sum()):,} outside p(s)'s range "
          f"skipped); {int((val['opportunity'] & ok).sum()):,} were challenge opportunities; "
          f"{len(ch):,} were challenged")
    if demo:
        print("  DEMO DATA — every number below is from a simulated season")

    # ------------------------------------------------------------ 1. worth
    banner("1. What a call is worth")
    opp = val[val["opportunity"] & ok]
    wrong = opp[opp["p_s"] >= 0.5]
    ovr = ch[ch["overturned"]]
    dist = []
    for name, d in (("every opportunity", opp), ("wrong calls (p(s) ≥ 50%)", wrong),
                    ("challenged", ch), ("overturned", ovr)):
        x = d["dwp"].to_numpy()
        dist.append({"set": name, "n": len(d), "mean": float(np.mean(x)) if len(x) else np.nan,
                     "median": float(np.median(x)) if len(x) else np.nan,
                     "p90": float(np.percentile(x, 90)) if len(x) else np.nan,
                     "runs_mean": float(d["druns_chall"].mean()) if len(d) else np.nan})
        print(f"  {name:<26} n={len(d):>7,}  mean {pts(dist[-1]['mean'], 2)}  median "
              f"{pts(dist[-1]['median'], 2)}  90th pct {pts(dist[-1]['p90'], 2)}  runs "
              f"{dist[-1]['runs_mean']:+.3f}")
    dist = pd.DataFrame(dist)
    top = ovr.sort_values("dwp", ascending=False).head(10).copy()
    top["challenger"] = top["challenger_id"].map(names).fillna("")
    top_cols = KEY + ["game_date", "inning", "bat_home", "home_team", "away_team", "chall_team",
                      "role", "challenger", "call_strike", "dwp", "druns_chall",
                      "wp_stands_chall", "wp_over_chall"]
    st_cols = cohort.set_index(KEY)[["outs_when_up", "balls", "strikes", "home_score",
                                     "away_score", "on_1b", "on_2b", "on_3b"]]
    top = top.join(st_cols, on=KEY)

    # ------------------------------------------------------------ 2. inning
    banner("2. Inning level — runs after overturned vs upheld challenges")
    ch_inn = ch[(ch["inning"] <= 8) & ch["runs_rest_inning"].notna()]
    nat_runs = natural_experiment(ch_inn, "runs_rest_inning", "re_over", "re_stands",
                                  B=args.boot, seed=args.seed)
    for _, r in nat_runs.iterrows():
        print(f"  {r['side']:<9} n={r['n']:>5,}: runs to the end of the inning — after overturned "
              f"{r['actual_overturned']:.3f}, after upheld {r['actual_upheld']:.3f}; gap "
              f"{r['actual_diff']:+.3f} [{r['actual_diff_lo']:+.3f}, {r['actual_diff_hi']:+.3f}]; "
              f"model predicted {r['pred_diff']:+.3f}")

    # ------------------------------------------------------------ 3. game
    banner("3. Game level — who won after overturned vs upheld challenges")
    ch_g = ch[ch["chall_won"].notna()]
    nat_wins = natural_experiment(ch_g, "chall_won", "wp_over_chall", "wp_stands_chall",
                                  B=args.boot, seed=args.seed)
    for _, r in nat_wins.iterrows():
        print(f"  {r['side']:<9} n={r['n']:>5,}: challenging team won {pct(r['actual_overturned'])} "
              f"after overturned vs {pct(r['actual_upheld'])} after upheld (gap "
              f"{pts(r['actual_diff'])}); predicted gap {pts(r['pred_diff'])}")
    realized, exp_made = float(ch["gain"].sum()), float(ch["ev"].sum())
    print(f"  the {len(ch):,} challenges were worth {exp_made:.1f} wins in expectation (p(s) × "
          f"value) and delivered {realized:.1f}")

    # ------------------------------------------------------------ 8 first: umpire tiers
    ump, vq = umpire_table(val)
    ump_ok = ump[ump["games"] >= 10].copy()
    if len(ump_ok) >= 3:
        ump_ok["tier"] = pd.qcut(ump_ok["excess_per_100"].rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
    else:
        ump_ok["tier"] = 1
    tier_of_ump = dict(zip(ump_ok["umpire_id"], ump_ok["tier"]))
    game_tier = val.drop_duplicates("game_pk").set_index("game_pk")["umpire_id"].map(tier_of_ump)

    # ------------------------------------------------------------ 4. ahead
    banner("4. Value still ahead — what holding a challenge could be worth")
    team_tier = {(g, t): game_tier.get(g) for g, t in val[["game_pk", "chall_team"]]
                 .drop_duplicates().itertuples(index=False)}
    ahead = value_ahead(val[ok], team_tier=team_tier)
    a_all = summarise_ahead(ahead)
    a_tier = summarise_ahead(ahead.dropna(subset=["tier"]), by="tier")
    now_val = (wrong.groupby("half_index")["dwp"].mean().rename("dwp_now").reset_index())
    a_all = a_all.merge(now_val, on="half_index", how="left")
    for h in (0, 8, 16, 17):
        r = a_all[a_all["half_index"] == h]
        if len(r):
            r = r.iloc[0]
            print(f"  start of {half_label(h):>6}: still at stake {pts(r['stakes'], 2)}; best single "
                  f"wrong call still to come {pts(r['best1'], 2)}; best two {pts(r['best2'], 2)}; "
                  f"a wrong call right now is worth {pts(r['dwp_now'], 2)}")

    # ------------------------------------------------------------ 5. season
    banner("5. Season level — wins added and left on the table")
    teams = team_season(val[ok])
    lg = league_totals(val[ok])
    print(f"  wrong calls moved {lg['wins_swung_by_wrong_calls']:.1f} wins across the league "
          f"({lg['expected_wrong_calls']:,.0f} expected wrong calls); overturns won back "
          f"{lg['wins_recovered']:.1f} ({pct(lg['share_recovered'])})")
    t3 = teams.head(3)
    for _, r in t3.iterrows():
        print(f"  {r['team']}: +{r['wins_added']:.2f} wins from {int(r['overturned'])} overturns; "
              f"{r['left_on_table']:.2f} left on the table")

    # ------------------------------------------------------------ 6. series
    banner("6. Series level")
    ser = series_table(games[games["usable"]])
    s_games, s_series = series_effects(val[ok], ser)
    n_series = int(ser["series_id"].nunique())
    swung = s_series[s_series["shift"].abs() >= 0.05]
    print(f"  {n_series:,} series; challenges moved the expected series result of a team by 5+ "
          f"points in {swung['series_id'].nunique():,} of them; total expected series results "
          f"changed: {s_series['shift'].abs().sum() / 2:.1f}")

    # ------------------------------------------------------------ 7. players
    banner("7. Players — who challenges well, and does it matter to ABS?")
    roles = []
    for role in ROLE_ORDER:
        d = ch[ch["role"].str.lower() == role]
        if not len(d):
            continue
        lo, hi = wilson_ci(int(d["overturned"].sum()), len(d))
        roles.append({"role": role, "n": len(d), "rate": d["overturned"].mean(), "lo": lo, "hi": hi,
                      "expected": d["p_s"].mean(), "mean_p": d["p_s"].mean(),
                      "oae_per_100": 100 * (d["overturned"].mean() - d["p_s"].mean()),
                      "value_per_chal": d["gain"].mean(), "ev_per_chal": d["ev"].mean()})
    roles = pd.DataFrame(roles)
    for side in ("batting", "fielding"):
        d = opp[(opp["side"] == side) & (opp["p_s"] >= 0.9)]
        rate = d["challenged"].mean() if len(d) else np.nan
        roles.loc[len(roles)] = {"role": f"{side} side — clearly wrong calls challenged",
                                 "n": len(d), "rate": rate}
    for _, r in roles.iterrows():
        if r["role"] in ROLE_ORDER:
            print(f"  {r['role'] + 's':<9} {int(r['n']):>6,} challenges, {pct(r['rate'])} overturned; "
                  f"the pitches they chose averaged p(s) {pct(r['mean_p'])}")
        else:
            print(f"  {r['role']}: {pct(r['rate'])} of {int(r['n']):,}")
    chal = challenger_table(ch)
    chal["name"] = chal["challenger_id"].map(names).fillna("")
    ch_r = ch[ch["challenger_id"].notna() & ch["role"].isin(ROLE_ORDER)]
    resid = ch_r["overturned"].astype(float) - ch_r["p_s"]
    t_out = cluster_signal(resid, ch_r["challenger_id"], ch_r["role"], n_perm=args.perm, seed=args.seed)
    t_sel = cluster_signal(ch_r["p_s"], ch_r["challenger_id"], ch_r["role"], n_perm=args.perm,
                           seed=args.seed)
    rel_sel = split_half(ch_r["p_s"], ch_r["challenger_id"], ch_r["game_date"])
    rel_out = split_half(resid, ch_r["challenger_id"], ch_r["game_date"])
    print(f"  given the pitch, does WHO challenged change the outcome? permutation p = {t_out[1]:.3f}; "
          f"size of real differences ±{100 * t_out[3]:.1f} pts (reliability {fmt_rel(rel_out)})")
    print(f"  do challengers differ in WHICH pitches they challenge? permutation p = {t_sel[1]:.3f}; "
          f"size ±{100 * t_sel[3]:.1f} pts of p(s) (reliability {fmt_rel(rel_sel)})")

    # ------------------------------------------------------------ 8. umpires
    banner("8. Umpires — who misses more, and what it changes")
    t_ump = cluster_signal(vq["excess"], vq["umpire_id"], n_perm=args.perm, seed=args.seed,
                           units=vq["game_pk"], min_n=200)
    rel_ump = split_half_games(vq)
    ch_u = ch[ch["umpire_id"].notna()]
    t_ump_out = cluster_signal(ch_u["overturned"].astype(float) - ch_u["p_s"], ch_u["umpire_id"],
                               n_perm=args.perm, seed=args.seed)
    tiers = (ump_ok.groupby("tier").agg(umpires=("umpire_id", "size"), games=("games", "sum"),
                                        wrong_per_game=("wrong_per_game", "mean"),
                                        excess_per_100=("excess_per_100", "mean"),
                                        stakes_per_game=("stakes_per_game", "mean"),
                                        challengeable_per_game=("challengeable_per_game", "mean"))
             .reset_index())
    print(f"  do umpires differ in how often ABS would reverse them (for the same difficulty)? "
          f"permutation p = {t_ump[1]:.3f}; size ±{100 * t_ump[3]:.2f} wrong calls per 100; "
          f"game-split reliability {fmt_rel(rel_ump)}")
    print(f"  given the pitch, does the UMPIRE change the overturn chance? p = {t_ump_out[1]:.3f}; "
          f"size ±{100 * t_ump_out[3]:.1f} pts")
    for _, r in tiers.iterrows():
        print(f"  {TIER_LABEL[int(r['tier'])]:<22} {r['wrong_per_game']:.2f} wrong calls a game; "
              f"{pts(r['stakes_per_game'], 2)} of win probability at stake a game")

    # ------------------------------------------------------------ write
    DOCS.mkdir(parents=True, exist_ok=True)
    keep = KEY + ["game_date", "inning", "bat_home", "home_team", "away_team", "chall_team",
                  "side", "role", "challenger_id", "umpire", "call_strike", "rem", "p_s", "dwp",
                  "druns_chall", "ev", "overturned", "gain", "runs_gain", "runs_rest_inning",
                  "chall_won", "split"]
    ch[keep].to_csv(DOCS / "week10_value_challenges.csv", index=False)
    teams.to_csv(DOCS / "week10_value_team_season.csv", index=False)
    a_all.to_csv(DOCS / "week10_value_ahead.csv", index=False)
    a_tier.to_csv(DOCS / "week10_value_ahead_by_umpire_tier.csv", index=False)
    s_series.to_csv(DOCS / "week10_value_series.csv", index=False)
    chal.to_csv(DOCS / "week10_value_challengers.csv", index=False)
    ump.to_csv(DOCS / "week10_value_umpires.csv", index=False)
    pd.concat([nat_runs.assign(outcome="runs to the end of the inning (innings 1–8)"),
               nat_wins.assign(outcome="challenging team won")]).to_csv(
        DOCS / "week10_value_natural_experiments.csv", index=False)
    import json as _json
    (DOCS / "week10_value_summary.json").write_text(_json.dumps({
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "p_s_version": pm.manifest.get("version"), "v_version": wp.version,
        **{k_: (float(v_) if isinstance(v_, (int, float, np.floating, np.integer)) else v_)
           for k_, v_ in lg.items()},
        "best_team": teams.iloc[0]["team"] if len(teams) else None,
        "best_team_wins": float(teams.iloc[0]["wins_added"]) if len(teams) else None,
        "p_who_given_pitch": t_out[1], "p_which_pitches": t_sel[1], "p_umpires": t_ump[1],
        "demo": demo}, indent=1))

    k = dict(demo=demo, pm=pm, wp=wp, fingerprint=fingerprint, check=check, val=val, ch=ch,
             opp=opp, dist=dist, top=top, nat_runs=nat_runs, nat_wins=nat_wins,
             realized=realized, exp_made=exp_made, a_all=a_all, a_tier=a_tier, teams=teams,
             lg=lg, n_series=n_series, s_series=s_series, s_games=s_games, roles=roles,
             chal=chal, t_out=t_out, t_sel=t_sel, rel_sel=rel_sel, rel_out=rel_out, ump=ump,
             ump_ok=ump_ok, t_ump=t_ump, rel_ump=rel_ump, t_ump_out=t_ump_out, tiers=tiers,
             args=args, n_skipped=int((~ok).sum()))
    if not args.no_figures:
        banner("Figures")
        import week10_figures as W10
        W10.make_value_figures(FIGS, DATA_DIR, k=k)
    (DOCS / "WEEK10_VALUE_REPORT.md").write_text(report(k))
    print(f"\n  report -> {DOCS / 'WEEK10_VALUE_REPORT.md'}   ({time.time() - t0:.0f}s)")


def wilson_ci(k, n):
    from src.abs_metrics import wilson
    return wilson(k, n)


def fmt_rel(rel):
    r, n = rel
    return "—" if not np.isfinite(r) else f"{r:.2f} over {n} clusters"


def split_half_games(vq):
    """Umpire accuracy, odd vs even games (in date order): is it a stable trait?"""
    g = vq.groupby(["umpire_id", "game_pk"]).agg(ex=("excess", "sum"), n=("excess", "size"),
                                                  date=("game_date", "first")).reset_index()
    g = g.sort_values(["umpire_id", "date", "game_pk"])
    g["half"] = g.groupby("umpire_id").cumcount() % 2
    h = g.groupby(["umpire_id", "half"]).agg(ex=("ex", "sum"), n=("n", "sum")).unstack()
    h = h.dropna()
    h = h[(h[("n", 0)] >= 500) & (h[("n", 1)] >= 500)]
    if len(h) < 5:
        return np.nan, int(len(h))
    a, b = h[("ex", 0)] / h[("n", 0)], h[("ex", 1)] / h[("n", 1)]
    r = float(np.corrcoef(a, b)[0, 1])
    return float(2 * r / (1 + r)), int(len(h))


# =========================================================== report
def report(k):
    w_ = []
    w = w_.append
    lg, ch = k["lg"], k["ch"]
    w("# Week 10 — What the 2026 challenges were worth\n")
    w(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/20_week10_challenge_value.py` · "
      f"p(s) {k['pm'].manifest.get('version')} · V(s) {k['wp'].version} · cohort "
      f"`{k['fingerprint']}` · {k['args'].boot} game-level bootstrap resamples · "
      f"{k['args'].perm} permutations per skill test\n")
    if k["demo"]:
        w("> **DEMO DATA** — a simulated season with planted differences between umpires and "
          "between challengers. The method is real; the numbers are not.\n")
    w("> Read this next to `WEEK10.md` (Part B), which explains every section.\n")
    w("## In one paragraph\n")
    t = k["teams"]
    w(f"Umpires' wrong calls moved **{lg['wins_swung_by_wrong_calls']:.1f} wins** across the "
      f"league ({lg['expected_wrong_calls']:,.0f} expected wrong calls on "
      f"{lg['called_pitches']:,} called pitches). Overturned challenges won back "
      f"**{lg['wins_recovered']:.1f}** of them ({pct(lg['share_recovered'])}). The "
      f"{len(ch):,} challenges were worth {k['exp_made']:.1f} wins in expectation and delivered "
      f"{k['realized']:.1f}. The best team added {t['wins_added'].max():.2f} wins with challenges; "
      f"the median team {t['wins_added'].median():.2f}.\n")

    w("## 0. Inputs\n")
    for line in describe_call_check(k["check"]):
        w(f"- {line}")
    w(f"- {len(k['val']):,} called pitches valued with p(s) and V(s) "
      f"({k['n_skipped']:,} with a location outside p(s)'s range skipped); "
      f"{int((k['val']['opportunity'] & np.isfinite(k['val']['p_s'])).sum()):,} challenge "
      "opportunities (a call against a side "
      f"with a challenge left, worked out from the umpire's ORIGINAL call); {len(ch):,} "
      "challenges.\n")

    w("## 1. What a call is worth\n")
    w("Win probability the challenging side gains if the call is overturned (`dwp`), and runs "
      "(to the end of the inning) for the same side:\n")
    w("| set | n | mean | median | 90th percentile | runs (mean) |\n|---|---:|---:|---:|---:|---:|")
    for _, r in k["dist"].iterrows():
        w(f"| {r['set']} | {r['n']:,} | {pts(r['mean'], 2)} | {pts(r['median'], 2)} | "
          f"{pts(r['p90'], 2)} | {r['runs_mean']:+.3f} |")
    w("\nMost calls are worth little; a few are worth a lot. The ten most valuable overturns:\n")
    w("| date | inning | score (home–away) | count | outs | challenging team | role | gain |\n"
      "|---|---|---|---|---:|---|---|---:|")
    for _, r in k["top"].iterrows():
        inn = f"{int(r['inning'])} {'bot' if r['bat_home'] else 'top'}"
        who = f"{r['chall_team']}" + (f" ({r['challenger']})" if r.get("challenger") else "")
        w(f"| {pd.Timestamp(r['game_date']):%b %d} | {inn} | {int(r['home_score'])}–"
          f"{int(r['away_score'])} | {int(r['balls'])}-{int(r['strikes'])} | "
          f"{int(r['outs_when_up'])} | {who} | {r['role']} | {pts(r['dwp'])} |")
    w("\nEvery challenge with its value: `docs/week10_value_challenges.csv`.\n")

    w("## 2. Inning level — more time before the third out\n")
    w("An overturned strike keeps an at-bat (or an inning) alive; an overturned ball takes "
      "a baserunner away. Whether a challenge is overturned is decided by where the ball was, "
      "not by the game situation, so overturned and upheld challenges are close to a natural "
      "experiment. Runs the batting team scored from the challenged pitch to the end of the "
      "inning (innings 1–8; walk-offs cut innings short):\n")
    w("| challenging side | n | after overturned | after upheld | gap [95%] | model's predicted "
      "gap |\n|---|---:|---:|---:|---|---:|")
    for _, r in k["nat_runs"].iterrows():
        w(f"| {r['side']} | {r['n']:,} | {r['actual_overturned']:.3f} | {r['actual_upheld']:.3f} | "
          f"{r['actual_diff']:+.3f} [{r['actual_diff_lo']:+.3f}, {r['actual_diff_hi']:+.3f}] | "
          f"{r['pred_diff']:+.3f} |")
    w("\nRead: on a batting-side challenge the gap should be positive (an overturned strike "
      "helps the batting team score); on a fielding-side challenge negative (an overturned "
      "ball takes runs away). If the observed gap's interval covers the model's prediction, "
      "the run-expectancy engine behind V(s) is right about what a challenge does to an "
      "inning.\n")

    w("## 3. Game level\n")
    w("| challenging side | n | won after overturned | won after upheld | gap [95%] | predicted |\n"
      "|---|---:|---:|---:|---|---:|")
    for _, r in k["nat_wins"].iterrows():
        w(f"| {r['side']} | {r['n']:,} | {pct(r['actual_overturned'])} | {pct(r['actual_upheld'])} | "
          f"{pts(r['actual_diff'])} [{pts(r['actual_diff_lo'])}, {pts(r['actual_diff_hi'])}] | "
          f"{pts(r['pred_diff'])} |")
    w("\nThe game-level gap is a few points at most and games are noisy, so expect a wide "
      "interval; the check is that it doesn't contradict the prediction.\n")

    w("## 4. Value still ahead — the raw material of \"holding a challenge\"\n")
    w("At the start of each half-inning, averaged over every game and team: how much win "
      "probability is still at stake in the calls that will go against that team later.\n")
    w("| start of | stakes still ahead | best single wrong call ahead | best two | a wrong call "
      "now is worth |\n|---|---:|---:|---:|---:|")
    for _, r in k["a_all"].iterrows():
        if int(r["half_index"]) in (0, 2, 4, 6, 8, 10, 12, 14, 16, 17):
            w(f"| {half_label(int(r['half_index']))} | {pts(r['stakes'], 2)} | {pts(r['best1'], 2)} | "
              f"{pts(r['best2'], 2)} | {pts(r['dwp_now'], 2)} |")
    w("\n- **best single wrong call ahead** is what ONE challenge saved for later could win with "
      "perfect foresight — an upper bound on the value of holding it. A real team doesn't know "
      "which call will be the big one, and a failed challenge is lost; Week 11's backward "
      "induction computes the real value, V(s, r) − V(s, r − 1), below this ceiling.")
    w("- Compare it with **a wrong call now**: early in the game the best call still to come is "
      "worth several times a typical call now — the case for holding. Late, the two converge, "
      "and the last innings are where a challenge in hand is worth spending.\n")
    a_t = k["a_tier"]
    if len(a_t) and a_t["tier"].nunique() > 1:
        w("By the umpire's accuracy (thirds of umpires, adjusted for how hard their pitches were):\n")
        w("| start of | " + " | ".join(TIER_LABEL[int(t_)] for t_ in sorted(a_t["tier"].unique()))
          + " |\n|---|" + "---:|" * a_t["tier"].nunique())
        for h in (0, 8, 16):
            vals = [a_t[(a_t["half_index"] == h) & (a_t["tier"] == t_)]["best1"]
                    for t_ in sorted(a_t["tier"].unique())]
            w(f"| {half_label(h)} | " + " | ".join(pts(v.iloc[0], 2) if len(v) else "—"
                                                    for v in vals) + " |")
        w("\n(best single wrong call ahead) A less accurate umpire supplies more chances later, "
          "so a challenge in hand is worth more with him behind the plate.\n")

    w("## 5. Season level\n")
    w("| team | games | challenges | overturned | success | wins added | runs added | left on the "
      "table | lost when out | umpire net |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        w(f"| {r['team']} | {r['games']} | {r['challenges']} | {r['overturned']} | "
          f"{pct(r['success_rate'], 0)} | {r['wins_added']:+.2f} | {r['runs_added']:+.1f} | "
          f"{r['left_on_table']:.2f} | {r['lost_when_out']:.2f} | {r['umpire_net']:+.2f} |")
    w("\n- **wins added**: win probability gained by the team's overturned challenges.")
    w("- **left on the table**: expected wins in wrong calls against the team that it could have "
      "challenged (it had one left) and didn't.")
    w("- **lost when out**: expected wins in wrong calls against it after it had run out.")
    w("- **umpire net**: wrong calls that stood in its favour minus those that stood against it.\n")

    w("## 6. Series level\n")
    ss = k["s_series"]
    w(f"{k['n_series']:,} series. A challenge's win probability only changes a series result when "
      "its game decided the series (flip that game and the team wins, splits or loses the series "
      "instead). Summed over those deciding games, challenges changed the expected series "
      f"results by **{ss['shift'].abs().sum() / 2:.1f} series** across the season; "
      f"{int((ss['shift'].abs() >= 0.05).sum() // 2)} series moved by 5+ points.\n")
    big = ss.reindex(ss["shift"].sort_values(ascending=False).index).head(8)
    w("| series | team | games | net challenge value | value in deciding games |\n"
      "|---:|---|---:|---:|---:|")
    for _, r in big.iterrows():
        w(f"| {int(r['series_id'])} | {r['team']} | {int(r['series_len'])} | {pts(r['net_gain'])} | "
          f"{pts(r['shift'])} |")
    w("\nSeason level and series level are the same numbers added up differently: Week 11's "
      "policy will be scored the same way (wins per season), and Spring's simulator replays "
      "whole seasons under it.\n")

    w("## 7. Players — who challenges well, and does it matter to ABS?\n")
    w("| role | challenges | overturned [95%] | mean p(s) of the pitches chosen | above expected "
      "(per 100) | win prob. per challenge |\n|---|---:|---|---:|---:|---:|")
    for _, r in k["roles"].iterrows():
        if r["role"] in ROLE_ORDER:
            w(f"| {r['role']} | {int(r['n']):,} | {pct(r['rate'])} [{pct(r['lo'])}, {pct(r['hi'])}] | "
              f"{pct(r['mean_p'])} | {r['oae_per_100']:+.1f} | {pts(r['value_per_chal'], 2)} |")
    for _, r in k["roles"].iterrows():
        if r["role"] not in ROLE_ORDER:
            w(f"\n- {r['role']} (p(s) ≥ 90%): {pct(r['rate'])} of {int(r['n']):,}")
    t_out, t_sel = k["t_out"], k["t_sel"]
    w(f"\n**Two different questions:**\n")
    w(f"1. *Given the pitch, does WHO challenged change what ABS rules?* Permutation test on "
      f"overturned − p(s), shuffling challengers within role: **p = {t_out[1]:.3f}**; size of "
      f"real differences ±{100 * t_out[3]:.1f} points (split-half reliability "
      f"{fmt_rel(k['rel_out'])}). ABS sees only the ball, so this should be no or tiny — and "
      "then **player accuracy does not belong in p(s)**: the location model already says what "
      "the challenger's identity could.")
    w(f"2. *Do challengers differ in WHICH pitches they challenge?* Permutation test on the p(s) "
      f"of the pitches they chose: **p = {t_sel[1]:.3f}**; size ±{100 * t_sel[3]:.1f} points of "
      f"p(s) (split-half reliability {fmt_rel(k['rel_sel'])}). This is the real meaning of \"a "
      "more accurate challenger\": better judgment about which calls are wrong. It belongs in "
      "the **decision**, not in p(s) — see WEEK10.md, \"Where each idea goes\".\n")
    c = k["chal"][k["chal"]["n"] >= 10].sort_values("rate_shrunk", ascending=False)
    if len(c):
        w("Challengers with 10+ challenges, best and worst by shrunk success rate (pulled toward "
          "their role's rate in proportion to how few challenges they have):\n")
        w("| challenger | role | challenges | overturned | shrunk rate | mean p(s) chosen | above "
          "expected |\n|---|---|---:|---:|---:|---:|---:|")
        for _, r in pd.concat([c.head(5), c.tail(5)]).drop_duplicates().iterrows():
            nm = r["name"] or str(int(r["challenger_id"]))
            w(f"| {nm} | {r['role']} | {int(r['n'])} | {int(r['overturned'])} | "
              f"{pct(r['rate_shrunk'])} | {pct(r['mean_p'])} | {r['oae']:+.1f} |")
        w("")

    w("## 8. Umpires — who misses more, and what it changes\n")
    w(f"- Do umpires differ in how often ABS would reverse them, for pitches of the same "
      f"difficulty? Permutation test (whole games shuffled between umpires): **p = "
      f"{k['t_ump'][1]:.3f}**; size ±{100 * k['t_ump'][3]:.2f} wrong calls per 100 called "
      f"pitches; reliability across odd and even games {fmt_rel(k['rel_ump'])}.")
    tu = k["t_ump_out"]
    w(f"- Given the pitch, does the umpire change the chance a challenge succeeds? **p = "
      f"{tu[1]:.3f}**; size ±{100 * tu[3]:.1f} points.")
    if tu[1] < 0.05:
        w("  ABS doesn't know who the umpire was, so a real effect here can only come from our "
          "measurement: near the edge, Statcast's location and the batter's listed height are "
          "a little off from what ABS used, so a call that LOOKS right by our numbers is more "
          "often actually wrong behind an umpire who misses more. Its size is what matters: if "
          "it is a point or two, p(s) stays location-only; Spring's robustness week can test "
          "adding umpire accuracy to p(s).\n")
    else:
        w("  ABS doesn't know who the umpire was, and the data agree: umpire accuracy doesn't "
          "belong in p(s).\n")
    w("| umpires | count | wrong calls a game | vs an average umpire (per 100 calls) | win prob. "
      "at stake a game | of it challengeable |\n|---|---:|---:|---:|---:|---:|")
    for _, r in k["tiers"].iterrows():
        w(f"| {TIER_LABEL[int(r['tier'])]} | {int(r['umpires'])} | {r['wrong_per_game']:.2f} | "
          f"{r['excess_per_100']:+.2f} | {pts(r['stakes_per_game'], 2)} | "
          f"{pts(r['challengeable_per_game'], 2)} |")
    w("\nUmpire accuracy changes **how many chances** a team gets, not whether a challenge "
      "succeeds. That is exactly where it enters the decision: the value of holding a "
      "challenge depends on how many wrong calls are still likely to come (section 4).\n")

    w("## 9. Where each idea goes\n")
    w("| idea | what Week 10 did | where it is finished |\n|---|---|---|")
    w("| value of holding a challenge | the stakes still ahead and their perfect-foresight "
      "ceiling (section 4) | **Week 11** — V(s, r) − V(s, r − 1) from backward induction |")
    w("| inning-level effect | runs an overturn is worth (RE288), checked against what happened "
      "(section 2) | done; Week 11 can optimise runs or wins |")
    w("| series and season | wins added, left on the table, series swung (sections 5–6) | "
      "**Spring, Feb 19** — replay seasons under the optimal policy |")
    w("| player accuracy | tested: in p(s)? which pitches? (section 7) | **Spring, Feb 26 (RQ3)** "
      "— decision regret per player; optionally Week 11's \"realistic information\" variant |")
    w("| umpire accuracy | tested; opportunity supply by umpire (sections 4, 8) | **Week 11** — "
      "umpire-specific chance of future wrong calls in the MDP; **Spring, Mar 19 (RQ4)** — "
      "did umpires change during the season? |")
    w("\n## Figures (`figures/week10/`)\n")
    w("- `w10_07_opportunities.png` — every opportunity: chance it was wrong vs what it was worth")
    w("- `w10_08_value_ahead.png` — the stakes still ahead, by inning and by umpire")
    w("- `w10_09_inning_experiment.png` — runs after overturned vs upheld challenges vs predicted")
    w("- `w10_10_team_wins.png` — wins added and left on the table, by team")
    w("- `w10_11_challengers.png` — challengers: the pitches they choose vs how often they win")
    w("- `w10_12_umpires.png` — umpires: wrong calls per game against the chances they create")
    return "\n".join(w_)


if __name__ == "__main__":
    main()
