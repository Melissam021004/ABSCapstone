#!/usr/bin/env python3
"""
make_demo_data.py — synthetic stand-ins for the outputs of scripts 03-06.

Purpose: let you see exactly what the slide figures will look like BEFORE your
real pulls finish. Run this, run the four EDA scripts, and you have your deck's
figures today — then re-run the EDA scripts against real data and the same PNGs
regenerate with the real numbers.

The values are plausible but INVENTED. Every figure built from them is stamped
"DEMO DATA" so a demo chart can never accidentally end up in your presentation.

    python scripts/eda/make_demo_data.py
    python scripts/eda/make_demo_data.py --out data_demo   # keep it away from real data

Default output root is `data_demo/` precisely so it cannot overwrite a real pull.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260924)

FIRST = ["Cal", "Adley", "Will", "Sean", "Alejandro", "Logan", "Patrick", "Yainer",
         "J.T.", "Gabriel", "Austin", "Jonah", "Tyler", "Carson", "Francisco",
         "Bo", "Corbin", "Elly", "Gunnar", "Riley", "Marcus", "Jose", "Julio",
         "Kyle", "Matt", "Luis", "Rafael", "Bobby", "Anthony", "Ketel"]
LAST = ["Raleigh", "Rutschman", "Smith", "Murphy", "Kirk", "OHoppe", "Bailey",
        "Diaz", "Realmuto", "Moreno", "Wells", "Heim", "Stephenson", "Kelly",
        "Alvarez", "Bichette", "Carroll", "De La Cruz", "Henderson", "Greene",
        "Semien", "Ramirez", "Rodriguez", "Tucker", "Olson", "Arraez",
        "Devers", "Witt", "Volpe", "Marte"]
TEAMS = ["NYY", "BOS", "TOR", "BAL", "TBR", "CLE", "MIN", "DET", "KCR", "CHW",
         "HOU", "SEA", "TEX", "LAA", "ATH", "ATL", "PHI", "NYM", "WSN", "MIA",
         "MIL", "CHC", "STL", "CIN", "PIT", "LAD", "SDP", "SFG", "ARI", "COL"]
UMPS = ["Pat Hoberg", "Will Little", "Mark Wegner", "Tripp Gibson", "Lance Barksdale",
        "Angel Hernandez", "Doug Eddings", "Laz Diaz", "CB Bucknor", "Dan Iassogna",
        "Jordan Baker", "Adam Hamari", "Ryan Additon", "Nic Lentz", "Edwin Moscoso",
        "Chris Guccione", "Todd Tichenor", "John Tumpane", "Alan Porter", "Ben May",
        "Quinn Wolcott", "Chad Whitson", "Sean Barber", "Jansen Visconti",
        "Erich Bacchus", "Shane Livensparger", "Brennan Miller", "Nestor Ceja",
        "Emil Jimenez", "Malachi Moore", "Clint Vondrak", "Alex Tosi",
        "Ryan Wills", "Mike Estabrook", "Chris Segal", "Jeremie Rehak",
        "Roberto Ortiz", "Charlie Ramos", "Derek Thomas", "Brian Walsh"]


def write_table(df, path):
    """Write parquet if pyarrow is available, otherwise CSV. The EDA scripts
    read either, so a missing pyarrow never blocks the preview."""
    path = Path(path)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def names(n):
    return [f"{RNG.choice(FIRST)} {RNG.choice(LAST)}" for _ in range(n)]


def leaderboards(root):
    d = root / "raw" / "savant"
    d.mkdir(parents=True, exist_ok=True)
    # MLB has reported catchers are the most successful challengers.
    spec = {"catcher": (72, 0.55, 26), "batter": (210, 0.44, 9), "pitcher": (150, 0.49, 7)}
    for role, (n, base_rate, mean_chal) in spec.items():
        chal = np.maximum(1, RNG.poisson(mean_chal, n))
        rate = np.clip(RNG.normal(base_rate, 0.13, n), 0.05, 0.95)
        won = RNG.binomial(chal, rate)
        exp_chal = np.maximum(0.5, chal * RNG.normal(1.0, 0.22, n))
        df = pd.DataFrame({
            "entity_id": RNG.integers(400000, 700000, n),
            "player_name": names(n),
            "team": RNG.choice(TEAMS, n),
            "n_challenges": chal,
            "n_challenges_won": won,
            "challenge_win_pct": np.round(won / chal, 4),
            "expected_challenges": np.round(exp_chal, 2),
            "challenges_vs_expected": np.round(chal - exp_chal, 2),
            "net_runs_gained": np.round(RNG.normal(0.4, 2.1, n), 2),
        })
        for mode in ("for", "against"):
            out = df if mode == "for" else df.assign(
                n_challenges=np.maximum(1, RNG.poisson(mean_chal * 0.9, n)))
            out.to_csv(d / f"abs_leaderboard_{role}_{mode}_mlb_2026.csv", index=False)

    for kind, n in (("team-summary", 30), ("catching-team", 30)):
        chal = RNG.integers(180, 340, n)
        won = (chal * np.clip(RNG.normal(0.49, 0.05, n), 0.3, 0.7)).astype(int)
        df = pd.DataFrame({
            "team": TEAMS[:n], "n_challenges": chal, "n_challenges_won": won,
            "challenge_win_pct": np.round(won / chal, 4),
            "expected_challenges": np.round(chal * RNG.normal(1, 0.1, n), 1),
            "net_runs_gained": np.round(RNG.normal(0, 6, n), 2),
        })
        for mode in ("for", "against"):
            df.to_csv(d / f"abs_leaderboard_{kind}_{mode}_mlb_2026.csv", index=False)
    print(f"  leaderboards -> {d}")


def framing(root):
    d = root / "raw" / "savant"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for season, spread in ((2024, 1.0), (2025, 1.0), (2026, 0.78)):
        n = 68
        called = RNG.integers(2400, 7200, n)
        rv = RNG.normal(0, 5.4 * spread, n)
        frames.append(pd.DataFrame({
            "player_id": RNG.integers(400000, 700000, n),
            "player_name": names(n),
            "team": RNG.choice(TEAMS, n),
            "n_called_pitches": called,
            "strike_rate": np.round(np.clip(RNG.normal(0.503, 0.017, n), 0.44, 0.57), 4),
            "runs_extra_strikes": np.round(rv, 2),
            "season": season,
        }))
    allf = pd.concat(frames, ignore_index=True)
    write_table(allf, d / "catcher_framing_all_years.parquet")
    for s, g in allf.groupby("season"):
        g.to_csv(d / f"catcher_framing_{s}.csv", index=False)
    print(f"  catcher framing -> {d}")


def umpires(root):
    d = root / "raw" / "umpires"
    d.mkdir(parents=True, exist_ok=True)
    n = len(UMPS)
    games = RNG.integers(14, 32, n)
    called = games * RNG.integers(138, 158, n)
    acc = np.clip(RNG.normal(0.931, 0.011, n), 0.88, 0.965)
    err = 1 - acc
    tilt = RNG.normal(0.5, 0.12, n)            # share of errors that are missed strikes
    ump = pd.DataFrame({
        "hp_umpire": UMPS,
        "hp_umpire_id": RNG.integers(400000, 500000, n),
        "n_called": called, "n_games": games,
        "accuracy": np.round(acc, 4),
        "missed_strike_rate": np.round(err * tilt, 4),
        "missed_ball_rate": np.round(err * (1 - tilt), 4),
        "mean_abs_dist_in": np.round(RNG.normal(4.6, 0.4, n), 2),
        "n_borderline": (called * RNG.normal(0.14, 0.02, n)).astype(int),
        "borderline_accuracy": np.round(np.clip(RNG.normal(0.792, 0.031, n), 0.68, 0.89), 4),
    }).sort_values("accuracy", ascending=False)
    write_table(ump, d / "umpire_accuracy_derived_2026.parquet")

    rows = []
    for r in ump.itertuples():
        for _ in range(r.n_games):
            rows.append({"game_pk": int(RNG.integers(800000, 810000)),
                         "hp_umpire": r.hp_umpire, "hp_umpire_id": r.hp_umpire_id,
                         "n_called": int(RNG.integers(128, 168)),
                         "accuracy": float(np.clip(RNG.normal(r.accuracy, 0.021), 0.82, 1.0))})
    write_table(pd.DataFrame(rows), d / "umpire_game_accuracy_2026.parquet")
    print(f"  umpires -> {d}")


def winprob(root):
    d = root / "raw" / "winprob"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in range(240):
        pk = 800000 + g
        n_inn = 9 if RNG.random() > 0.09 else int(RNG.integers(10, 13))
        wp = 50.0
        ab = 0
        for inn in range(1, n_inn + 1):
            for half in ("top", "bottom"):
                for _ in range(int(RNG.integers(3, 7))):
                    # leverage climbs late and is right-skewed
                    li = float(RNG.lognormal(np.log(0.62 + 0.085 * inn), 0.62))
                    li = min(li, 6.0)
                    wp = float(np.clip(wp + RNG.normal(0, 3.1 * min(li, 3)), 1.5, 98.5))
                    rows.append({
                        "game_pk": pk, "at_bat_index": ab, "inning": inn,
                        "half_inning": half, "outs": int(RNG.integers(0, 3)),
                        "balls": 0, "strikes": 0,
                        "home_score": int(RNG.integers(0, 8)),
                        "away_score": int(RNG.integers(0, 8)),
                        "home_win_prob": round(wp, 2),
                        "away_win_prob": round(100 - wp, 2),
                        "home_win_prob_added": round(float(RNG.normal(0, 2.4)), 2),
                        "leverage_index": round(li, 3),
                        "batter_id": int(RNG.integers(400000, 700000)),
                        "pitcher_id": int(RNG.integers(400000, 700000)),
                        "event": "Single",
                    })
                    ab += 1
    df = pd.DataFrame(rows)
    df["bat_win_prob"] = np.where(df["half_inning"] == "bottom",
                                  df["home_win_prob"], df["away_win_prob"])
    write_table(df, d / "winprob_2026.parquet")
    print(f"  win probability -> {d}  ({len(df):,} plate appearances)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_demo",
                    help="output root (default data_demo/, kept separate from real data)")
    args = ap.parse_args()
    root = Path(args.out)
    print(f"Generating DEMO data under {root}/ — these numbers are invented.\n")
    leaderboards(root)
    framing(root)
    umpires(root)
    winprob(root)
    print(f"""
Done. Preview the figures with:

    ABS_DATA_DIR={root} python scripts/eda/eda_00_overview.py
    ABS_DATA_DIR={root} python scripts/eda/eda_03_leaderboards.py
    ABS_DATA_DIR={root} python scripts/eda/eda_04_framing.py
    ABS_DATA_DIR={root} python scripts/eda/eda_05_umpires.py
    ABS_DATA_DIR={root} python scripts/eda/eda_06_winprob.py

Every figure built from this data carries a "DEMO DATA" stamp.""")


if __name__ == "__main__":
    main()
