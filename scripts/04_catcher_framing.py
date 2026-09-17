#!/usr/bin/env python3
"""
04_catcher_framing.py — Statcast catcher framing leaderboard.

Why your project needs this: framing and challenging are the same skill viewed
twice. Savant's own ABS documentation says so explicitly — when a catcher gets
a ball flipped to a strike on challenge, he is credited both for the challenge
(+1.00) and for the framing attempt that preceded it.

That matters two ways:
  - As a MODEL FEATURE. A good framer gets more borderline calls in his favour
    to begin with, so his challenge opportunities are a different mix.
  - As an RQ4 OUTCOME. "Do catchers frame less now that they can challenge?"
    is a framing-metric question over time.

CAUTION for RQ4: 2026 framing numbers are themselves affected by challenges, so
season-over-season comparisons are not clean. Note that as a limitation, and
prefer within-2026 changes (e.g. framing behaviour when 0 vs 2 challenges remain)
which are identified off within-season variation instead.

    python 04_catcher_framing.py
    python 04_catcher_framing.py --years 2023 2024 2025 2026   # RQ4 trend

Output: data/raw/savant/catcher_framing_<year>.csv
        data/raw/savant/catcher_framing_all_years.parquet
"""

import argparse
import io
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
OUT_DIR = DATA_DIR / "raw" / "savant"

URL = "https://baseballsavant.mlb.com/leaderboard/catcher-framing"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")


def fetch_direct(session, year, min_called="q"):
    """Same parameter block pybaseball.statcast_catcher_framing() uses."""
    params = {
        "type": "catcher",
        "year": year,
        "seasonStart": year,
        "seasonEnd": year,
        "team": "",
        "min": min_called,
        "sortColumn": "rv_tot",
        "sortDirection": "desc",
        "csv": "true",
    }
    r = session.get(URL, params=params, timeout=90)
    r.raise_for_status()
    text = r.content.decode("utf-8", "replace")
    if "<html" in text[:200].lower():
        raise ValueError("got HTML instead of CSV")
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    # Savant appends a trailing blank column on some leaderboards.
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def fetch_pybaseball(year, min_called="q"):
    from pybaseball import statcast_catcher_framing
    return statcast_catcher_framing(year, min_called_p=min_called)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2026])
    ap.add_argument("--min-called", default="q",
                    help="'q' for qualified, or an integer pitch minimum")
    ap.add_argument("--method", choices=["auto", "pybaseball", "direct"], default="auto")
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Referer": URL})

    frames = []
    for year in args.years:
        print(f"  {year}", end="  ", flush=True)
        df = None

        if args.method in ("auto", "pybaseball"):
            try:
                df = fetch_pybaseball(year, args.min_called)
            except Exception as e:
                print(f"pybaseball failed ({type(e).__name__})", end=" -> ")
                df = None

        if df is None or (hasattr(df, "empty") and df.empty):
            try:
                df = fetch_direct(session, year, args.min_called)
            except Exception as e:
                print(f"direct failed too: {type(e).__name__}: {e}")
                continue

        if df is None or df.empty:
            print("no data")
            continue

        df["season"] = year
        df.to_csv(OUT_DIR / f"catcher_framing_{year}.csv", index=False)
        frames.append(df)
        print(f"{len(df)} catchers, {len(df.columns)} cols")
        time.sleep(args.sleep)

    if not frames:
        raise SystemExit(
            "No framing data.\n"
            "Fallback: open the leaderboard in a browser, use its Download CSV "
            "button, and save into data/raw/savant/.\n"
            "Second fallback: you can compute your own framing proxy from the "
            "Statcast pull — model P(called strike | location, count) and credit "
            "each catcher the residual. More work, but fully reproducible, which "
            "is one of your stated novelty claims.")

    allf = pd.concat(frames, ignore_index=True)
    out = OUT_DIR / "catcher_framing_all_years.parquet"
    allf.to_parquet(out, index=False)

    print(f"\nSaved {len(allf)} catcher-seasons -> {out}")
    print(f"Columns: {', '.join(allf.columns.astype(str))}")

    id_col = next((c for c in ("player_id", "catcher", "entity_id", "id")
                   if c in allf.columns), None)
    if id_col:
        print(f"\nJoin key for the pitch table: {id_col} == Statcast 'fielder_2'")
    else:
        print("\nNo obvious player-id column — you may have to join on name. "
              "If so, build a name->MLBAM id crosswalk with "
              "pybaseball.playerid_lookup and keep it in data/interim/.")

    if len(args.years) > 1:
        rv = next((c for c in allf.columns if "runs_extra_strikes" in c.lower()
                   or c.lower() in ("rv_tot", "runs_total")), None)
        if rv:
            print("\nFraming run value by season (RQ4 trend):")
            print(allf.groupby("season")[rv].agg(["mean", "std", "count"]).to_string())
            print("\nReminder: a 2026 drop is NOT clean evidence that catchers "
                  "changed behaviour — the metric's definition now interacts with "
                  "challenges. Say so in the limitations section.")


if __name__ == "__main__":
    main()
