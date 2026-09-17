#!/usr/bin/env python3
"""
03_savant_abs_leaderboards.py — Baseball Savant ABS challenge leaderboards.

These are AGGREGATES (player-season / team-season), not pitch-level data.
Two things they're good for:

  1. VALIDATION. Your pitch-level challenge counts from script 02 should add up
     to Savant's totals. If they don't, your extraction is wrong. Do this check
     in Week 5 — it is the cheapest QA you will ever run.
  2. BENCHMARKING. Savant publishes "expected challenges" and "overturns vs
     expected", which is the descriptive baseline your proposal says you are
     improving on. You need their numbers to argue you beat them.

Do NOT feed Savant's expected-challenge metrics into your model as features —
they are built from the outcome you are predicting. That is the leak your
Week 8 notes warn about.

    python 03_savant_abs_leaderboards.py
    python 03_savant_abs_leaderboards.py --year 2026 --level aaa

Output: data/raw/savant/abs_leaderboard_<type>_<mode>_<level>_<year>.csv
        data/raw/savant/abs_leaderboard_summary.txt
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

URL = "https://baseballsavant.mlb.com/leaderboard/abs-challenges"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

# The views the ABS leaderboard exposes.
CHALLENGE_TYPES = ["batter", "catcher", "pitcher", "team-summary", "catching-team"]
# 'for'    = challenges this player/team made
# 'against'= challenges made against them
DATA_MODES = ["for", "against"]


def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "text/csv,application/csv,text/html;q=0.9,*/*;q=0.8",
        "Referer": URL,
    })
    return s


def fetch(session, year, challenge_type, mode, level, game_type, retries=3):
    params = {
        "gameType": game_type,
        "year": year,
        "challengeType": challenge_type,
        "level": level,
        "minChal": 1,
        "minOppChal": 0,
        "dataMode": mode,
        "page": 0,
        "pageSize": 2000,      # ask for everything in one page
        "csv": "true",
    }
    for attempt in range(retries):
        try:
            r = session.get(URL, params=params, timeout=90)
            if r.status_code != 200:
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(5 * (2 ** attempt))
                    continue
                return None, f"HTTP {r.status_code}"
            text = r.content.decode("utf-8", "replace")
            if text.lstrip().lower().startswith("<!doctype") or "<html" in text[:200].lower():
                return None, "got HTML, not CSV (csv=true may not be supported here)"
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            return (df, "ok") if not df.empty else (None, "empty")
        except Exception as e:
            if attempt == retries - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(5 * (2 ** attempt))
    return None, "exhausted retries"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--level", default="mlb", choices=["mlb", "aaa"])
    ap.add_argument("--game-type", default="regular",
                    choices=["regular", "spring", "postseason"])
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    lines, got = [], 0

    print(f"Savant ABS leaderboards — {args.year} {args.level} {args.game_type}\n")
    for ct in CHALLENGE_TYPES:
        for mode in DATA_MODES:
            print(f"  {ct:<15} {mode:<8}", end="  ", flush=True)
            df, status = fetch(session, args.year, ct, mode, args.level, args.game_type)
            if df is None:
                print(f"skip ({status})")
                lines.append(f"{ct}/{mode}: FAILED — {status}")
                time.sleep(args.sleep)
                continue

            name = f"abs_leaderboard_{ct}_{mode}_{args.level}_{args.year}.csv"
            df.to_csv(OUT_DIR / name, index=False)
            got += 1
            print(f"{len(df):>5} rows, {len(df.columns)} cols -> {name}")
            lines.append(f"{ct}/{mode}: {len(df)} rows, {len(df.columns)} cols")
            lines.append(f"    columns: {', '.join(df.columns.astype(str))}")
            time.sleep(args.sleep)

    (OUT_DIR / "abs_leaderboard_summary.txt").write_text("\n".join(lines))

    print(f"\n{got}/{len(CHALLENGE_TYPES)*len(DATA_MODES)} leaderboards saved to {OUT_DIR}")

    if got == 0:
        print("""
Nothing downloaded. The ABS leaderboard may not honour csv=true.
Fallbacks, in order of effort:
  1. Open the leaderboard in a browser and use its own "Download CSV" button,
     then drop the file into data/raw/savant/ by hand. Note the date you did it
     in your manifest — a hand-pulled file still needs provenance.
  2. Watch the Network tab while the page loads; the table is populated from a
     JSON call. Copy that URL into this script.
  3. Skip it. These are validation aggregates, not model inputs. Your pitch-level
     pipeline (scripts 01, 02) does not depend on them.""")
        return

    # Quick validation hook for Week 5.
    tm = OUT_DIR / f"abs_leaderboard_team-summary_for_{args.level}_{args.year}.csv"
    if tm.exists():
        df = pd.read_csv(tm)
        col = next((c for c in df.columns
                    if "chal" in c.lower() and df[c].dtype.kind in "if"
                    and "pct" not in c.lower() and "rate" not in c.lower()), None)
        if col:
            print(f"\nValidation target: Savant total '{col}' = {df[col].sum():,.0f}")
            print("Compare against len(abs_challenges_2026.parquet) from script 02.")
            print("A gap over ~2% means your extraction is missing challenge events.")


if __name__ == "__main__":
    main()
