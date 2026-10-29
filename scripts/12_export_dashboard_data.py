#!/usr/bin/env python3
"""
12_export_dashboard_data.py — turn the Week 5 table into the dashboard's data file.

The dashboard is a static website: no server, no database. Everything it shows
has to arrive as one file the browser downloads once. A full season is ~700,000
pitches, which is far too much to ship, so this script does the aggregating in
Python and exports only what the page needs.

Two kinds of data come out:

  1. EVERY CHALLENGE, one row each. A season has roughly 8,000 of them, which is
     small enough to ship whole — so the dashboard can filter them any way you
     like, in the browser, instantly. This is what makes the page feel like
     Tableau rather than a set of pictures.

  2. A PRE-AGGREGATED CUBE for everything that needs the pitch-level denominator
     (challenge rates, opportunity counts). Counting 700,000 rows in a browser is
     slow and shipping them is impossible, so we pre-count them here, grouped by
     the four dimensions the filters use. That is ~300 rows instead of 700,000,
     and filtering still produces exactly correct rates.

Output is a .js file rather than .json on purpose: a <script> tag works when you
double-click index.html locally, while fetch() of a .json file is blocked by the
browser's file:// security rules. One less thing to go wrong.

    python scripts/12_export_dashboard_data.py
    ABS_DATA_DIR=data_demo python scripts/12_export_dashboard_data.py

Output: dashboard/data.js
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
PROCESSED = DATA_DIR / "processed"
OUT = Path(os.environ.get("ABS_DASHBOARD_DIR", "dashboard"))

INNING_BUCKETS = [(1, 3, "1-3"), (4, 6, "4-6"), (7, 9, "7-9"), (10, 99, "Extras")]


def read_any(path):
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    csv = p.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def jnum(v, nd=4):
    """JSON-safe number: NaN/inf become None, floats get rounded."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), nd)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def jstr(v, default=""):
    """JSON-safe string: a missing value in a pandas object column is NaN, not
    None, and NaN is truthy — so `v or default` silently keeps the NaN."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return str(v)


def clean(obj):
    """
    Walk the whole payload and make it strictly JSON-compliant.

    Guarding each field individually is how NaN keeps slipping through: real
    data has gaps in columns the synthetic fixture always fills, and json.dumps
    with allow_nan=False then fails on the one field that was missed. Sanitising
    the finished structure removes the whole class of bug.
    """
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    try:
        if pd.isna(obj):            # pd.NA, pd.NaT
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)                 # Timestamps and anything else exotic


def inning_bucket(inn):
    for lo, hi, label in INNING_BUCKETS:
        if lo <= inn <= hi:
            return label
    return "Extras"


def build_challenges(df, names):
    """One compact record per challenge. Short keys keep the file small."""
    ch = df[df["was_challenged"]].copy()
    if ch.empty:
        return []
    ch["month"] = pd.to_datetime(ch["game_date"]).dt.strftime("%b")
    ch["inn_b"] = ch["inning"].apply(inning_bucket)
    ch["team"] = np.where(ch["challenging_side"] == "home",
                          ch.get("home_team", ""), ch.get("away_team", ""))
    rem = np.where(ch["challenging_side"] == ch["batting_side"],
                   ch["challenges_remaining_batting"],
                   ch["challenges_remaining_fielding"])
    ch["rem"] = rem

    out = []
    for r in ch.itertuples():
        out.append({
            "g": int(r.game_pk),
            "dt": str(r.game_date)[:10],
            "inn": jnum(r.inning), "ib": jstr(r.inn_b, "?"),
            "half": "T" if str(r.inning_topbot).lower().startswith("top") else "B",
            "b": jnum(r.balls), "s": jnum(r.strikes),
            "role": jstr(r.challenger_role, "unknown"),
            "side": jstr(r.challenging_side, "unknown"),
            "team": jstr(r.team, ""),
            "mo": jstr(r.month, "?"),
            "rem": int(r.rem) if pd.notna(r.rem) else None,
            "ov": bool(r.is_overturned) if pd.notna(r.is_overturned) else None,
            "d": jnum(r.zone_dist_in, 2),
            "px": jnum(r.plate_x, 3), "pz": jnum(r.plate_z, 3),
            "zt": jnum(r.abs_zone_top, 3), "zb": jnum(r.abs_zone_bot, 3),
            "who": names.get(int(r.challenger_id), "") if pd.notna(r.challenger_id) else "",
            "ump": jstr(r.hp_umpire, ""),
            "lev": jnum(getattr(r, "leverage_index", None), 2),
        })
    return out


def build_cube(df):
    """
    Opportunity and challenge counts grouped by the four filter dimensions.

    This is the trick that makes correct rates possible without shipping every
    pitch: group once in pandas, and the browser re-sums whichever slices the
    filters leave. Rates stay exact because we keep the numerator AND the
    denominator in every cell.
    """
    opp = df[df["is_challenge_opportunity"]].copy()
    if opp.empty:
        return []
    opp["mo"] = pd.to_datetime(opp["game_date"]).dt.strftime("%b")
    opp["ib"] = opp["inning"].apply(inning_bucket)

    # An opportunity belongs to whichever side the call went against.
    opp["role_group"] = np.where(opp["called_strike"], "batter", "fielding")
    opp["rem"] = np.where(opp["called_strike"],
                          opp["challenges_remaining_batting"],
                          opp["challenges_remaining_fielding"])

    g = opp.groupby(["role_group", "mo", "ib", "rem", "balls", "strikes"],
                    observed=True).agg(
        opp=("was_challenged", "size"),
        chal=("was_challenged", "sum")).reset_index()

    return [{"rg": r.role_group, "mo": r.mo, "ib": r.ib, "rem": int(r.rem),
             "b": int(r.balls), "s": int(r.strikes),
             "o": int(r.opp), "c": int(r.chal)} for r in g.itertuples()]


def build_players(df, names):
    """Leaderboards. One table per role, plus umpires."""
    ch = df[df["was_challenged"] & df["is_overturned"].notna()].copy()
    out = {}
    if ch.empty:
        return {"batter": [], "catcher": [], "pitcher": [], "umpire": []}

    for role in ("batter", "catcher", "pitcher"):
        sub = ch[ch["challenger_role"] == role]
        if sub.empty:
            out[role] = []
            continue
        g = sub.groupby("challenger_id").agg(
            n=("is_overturned", "size"),
            won=("is_overturned", "sum"),
            avg_dist=("zone_dist_in", lambda s: s.abs().mean())).reset_index()
        g["rate"] = g["won"] / g["n"]
        g = g.sort_values(["n", "rate"], ascending=False)
        out[role] = [{"name": names.get(int(r.challenger_id), f"#{int(r.challenger_id)}"),
                      "id": int(r.challenger_id), "n": int(r.n), "won": int(r.won),
                      "rate": jnum(r.rate), "dist": jnum(r.avg_dist, 2)}
                     for r in g.itertuples()]

    # Umpires: challenges faced, plus their own accuracy if script 05 ran.
    if "hp_umpire" in ch.columns:
        g = ch.groupby("hp_umpire").agg(
            n=("is_overturned", "size"), won=("is_overturned", "sum")).reset_index()
        g["rate"] = g["won"] / g["n"]
        acc = (df.groupby("hp_umpire")["ump_accuracy"].first()
               if "ump_accuracy" in df.columns else None)
        g = g.sort_values("n", ascending=False)
        out["umpire"] = [{"name": r.hp_umpire, "n": int(r.n), "won": int(r.won),
                          "rate": jnum(r.rate),
                          "acc": jnum(acc.get(r.hp_umpire) if acc is not None else None)}
                         for r in g.itertuples()]
    else:
        out["umpire"] = []
    return out


def build_inventory(df):
    d = df[df["inning"] <= 9]
    rows = []
    for inn, g in d.groupby("inning"):
        both = pd.concat([g["challenges_remaining_batting"],
                          g["challenges_remaining_fielding"]])
        vc = both.value_counts(normalize=True)
        rows.append({"inn": int(inn), "two": jnum(vc.get(2, 0), 4),
                     "one": jnum(vc.get(1, 0), 4), "zero": jnum(vc.get(0, 0), 4)})
    return sorted(rows, key=lambda r: r["inn"])


def build_distance_curve(df):
    ch = df[df["was_challenged"] & df["is_overturned"].notna()].copy()
    if ch.empty:
        return []
    ch["absd"] = ch["zone_dist_in"].abs()
    edges = [0, 0.5, 1, 1.5, 2, 3, 4, 6, np.inf]
    labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "4-6", "6+"]
    ch["bucket"] = pd.cut(ch["absd"], edges, labels=labels, right=False)
    g = ch.groupby("bucket", observed=True).agg(
        n=("is_overturned", "size"), rate=("is_overturned", "mean")).reset_index()
    return [{"bucket": str(r.bucket), "n": int(r.n), "rate": jnum(r.rate)}
            for r in g.itertuples() if r.n >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default=str(PROCESSED / "challenge_inventory_2026.parquet"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = read_any(args.table)
    if df is None:
        sys.exit(f"No joined table at {args.table}.\n"
                 f"Run: python scripts/run_week5.py   (or --demo for a test season)")

    for c in ("was_challenged", "is_called_pitch", "is_challenge_opportunity",
              "is_challenge_opportunity_batting", "is_challenge_opportunity_fielding",
              "is_reasonable_pitch", "abs_strike", "would_be_overturned",
              "called_strike"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    print(f"Reading {len(df):,} pitches x {len(df.columns)} columns")

    # name lookup for challengers
    names = {}
    bios = read_any(DATA_DIR / "raw" / "players" / "player_bios_2026.parquet")
    if bios is not None and {"player_id", "full_name"} <= set(bios.columns):
        names = dict(zip(bios["player_id"].astype(int), bios["full_name"]))
    if "challenger_name" in df.columns and "challenger_id" in df.columns:
        sub = df[df["challenger_id"].notna() & df["challenger_name"].notna()]
        for i, n in zip(sub["challenger_id"].astype(int), sub["challenger_name"]):
            names.setdefault(i, n)

    is_demo = "demo" in str(DATA_DIR).lower()
    challenges = build_challenges(df, names)
    cube = build_cube(df)
    players = build_players(df, names)

    data = {
        "meta": {
            "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "season": 2026,
            "is_demo": is_demo,
            "n_pitches": int(len(df)),
            "n_called": int(df["is_called_pitch"].sum()),
            "n_opportunities": int(df["is_challenge_opportunity"].sum()),
            "n_reasonable": int(df["is_reasonable_pitch"].sum()),
            "n_challenges": int(df["was_challenged"].sum()),
            "n_games": int(df["game_pk"].nunique()),
            "date_min": str(df["game_date"].min())[:10],
            "date_max": str(df["game_date"].max())[:10],
            "overturn_rate": jnum(df.loc[df["was_challenged"], "is_overturned"].mean()),
            "n_columns": int(len(df.columns)),
        },
        "funnel": [
            {"stage": "All pitches", "n": int(len(df))},
            {"stage": "Called pitches", "n": int(df["is_called_pitch"].sum())},
            {"stage": "Challenge opportunities",
             "n": int(df["is_challenge_opportunity"].sum())},
            {"stage": "Reasonable pitches", "n": int(df["is_reasonable_pitch"].sum())},
            {"stage": "Challenged", "n": int(df["was_challenged"].sum())},
        ],
        "challenges": challenges,
        "cube": cube,
        "players": players,
        "inventory": build_inventory(df),
        "distance_curve": build_distance_curve(df),
        "months": sorted(pd.to_datetime(df["game_date"]).dt.strftime("%b").unique(),
                         key=lambda m: dt.datetime.strptime(m, "%b").month),
        "teams": sorted(set(df.get("home_team", pd.Series(dtype=str)).dropna())
                        | set(df.get("away_team", pd.Series(dtype=str)).dropna())),
    }

    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.js"
    # allow_nan=False stays on deliberately: clean() should make it impossible
    # to reach, and if it ever does we want a loud failure, not a file the
    # browser silently refuses to parse.
    payload = json.dumps(clean(data), separators=(",", ":"), allow_nan=False)
    path.write_text(
        "// Generated by scripts/12_export_dashboard_data.py — do not edit by hand.\n"
        "// Re-run that script to refresh. Loaded by index.html before app.js.\n"
        f"window.ABS_DATA = {payload};\n")

    kb = path.stat().st_size / 1024
    print(f"\n  challenges exported : {len(challenges):,} rows")
    print(f"  cube cells          : {len(cube):,} (instead of "
          f"{data['meta']['n_opportunities']:,} opportunity rows)")
    print(f"  players             : " + ", ".join(
        f"{k} {len(v)}" for k, v in players.items()))
    print(f"\n  {path}  ({kb:,.0f} KB)")
    if is_demo:
        print("\n  Built from DEMO data — the dashboard will show a banner saying so.")
    src_n = int(df["was_challenged"].sum())
    dropped = src_n - len(challenges)
    if dropped > 0:
        print(f"\n  NOTE: {dropped:,} challenge(s) in the table had no usable "
              f"pitch key and are not in the export.")
    if kb > 2000:
        print(f"\n  NOTE: data.js is {kb/1024:.1f} MB. GitHub Pages gzips it, so "
              f"visitors download roughly a quarter of that, but if it keeps "
              f"growing the cheapest trim is dropping `zt`/`zb` from the "
              f"challenge records (used only by the zone-position chart).")


if __name__ == "__main__":
    main()
