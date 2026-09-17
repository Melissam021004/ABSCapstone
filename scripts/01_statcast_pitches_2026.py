#!/usr/bin/env python3
"""
01_statcast_pitches_2026.py — every 2026 MLB Statcast pitch.

This is the backbone table. Everything else joins onto it.

Two independent paths to the same data:
  --method pybaseball   pybaseball.statcast()  (convenient; sometimes 403s)
  --method direct       raw Savant CSV endpoint with a browser User-Agent
  --method auto         try pybaseball, fall back to direct  [default]

The pull is CHUNKED and RESUMABLE. Each chunk is cached to disk; re-running
skips chunks already downloaded. A full season is ~700-750k pitches and takes
1-3 hours, so start it in the morning and let it run.

    # Week 3: one day, to inspect columns
    python 01_statcast_pitches_2026.py --start 2026-04-15 --end 2026-04-15

    # Week 4: the whole season
    python 01_statcast_pitches_2026.py

    # resume after an interruption (just run it again)
    python 01_statcast_pitches_2026.py

Output:
    data/raw/statcast/chunks/statcast_<start>_<end>.parquet   (cache)
    data/raw/statcast/statcast_2026_<YYYY-MM>.parquet         (monthly)
    data/raw/statcast/statcast_2026_columns.txt
"""

import argparse
import datetime as dt
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
OUT_DIR = DATA_DIR / "raw" / "statcast"
CHUNK_DIR = OUT_DIR / "chunks"

# 2026 regular season. Adjust if you want spring training or the postseason.
SEASON_START = "2026-03-25"
SEASON_END = "2026-09-28"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"


def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "text/csv,application/csv,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://baseballsavant.mlb.com/statcast_search",
    })
    return s


def savant_params(start, end, game_type="R|"):
    """
    Mirrors the parameter block Baseball Savant's own search form submits.
    Most of these must be present-but-empty or the endpoint returns nothing.
    game_type: 'R|' regular, 'R|PO|' + postseason, 'S|' spring.
    """
    return {
        "all": "true",
        "hfPT": "", "hfAB": "", "hfGT": game_type, "hfPR": "", "hfZ": "",
        "hfStadium": "", "hfBBL": "", "hfNewZones": "", "hfC": "",
        "hfSea": "2026|", "hfSit": "", "hfOuts": "", "hfOpponent": "",
        "pitcher_throws": "", "batter_stands": "", "hfSA": "",
        "player_type": "pitcher",
        "game_date_gt": start,
        "game_date_lt": end,
        "hfInfield": "", "team": "", "position": "", "hfOutfield": "",
        "hfRO": "", "home_road": "", "hfFlag": "", "hfPull": "",
        "metric_1": "", "hfInn": "", "min_pitches": "0", "min_results": "0",
        "group_by": "name", "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc", "min_pas": "0",
        "type": "details",
    }


def fetch_direct(session, start, end, retries=4):
    """Pull one chunk straight from the Savant CSV endpoint."""
    last = None
    for attempt in range(retries):
        try:
            r = session.get(SAVANT_CSV, params=savant_params(start, end), timeout=180)
            if r.status_code == 200:
                if not r.content or len(r.content) < 200:
                    return pd.DataFrame()
                df = pd.read_csv(io.StringIO(r.content.decode("utf-8", "replace")),
                                 low_memory=False)
                # Savant sometimes repeats the header inside the body.
                if "game_date" in df.columns:
                    df = df[df["game_date"] != "game_date"]
                return df
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 5 * (2 ** attempt)
                print(f"      HTTP {r.status_code}, retrying in {wait}s")
                time.sleep(wait)
                last = f"HTTP {r.status_code}"
                continue
            return None if r.status_code == 403 else pd.DataFrame()
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(5 * (2 ** attempt))
    print(f"      giving up on {start}..{end} ({last})")
    return pd.DataFrame()


def fetch_pybaseball(start, end):
    from pybaseball import statcast
    return statcast(start_dt=start, end_dt=end, verbose=False)


def date_chunks(start, end, days):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    while s <= e:
        c_end = min(s + dt.timedelta(days=days - 1), e)
        yield s.isoformat(), c_end.isoformat()
        s = c_end + dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=SEASON_START)
    ap.add_argument("--end", default=SEASON_END)
    ap.add_argument("--method", choices=["auto", "pybaseball", "direct"], default="auto")
    ap.add_argument("--chunk-days", type=int, default=3,
                    help="days per request; smaller = more robust, slower")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--game-type", default="R|",
                    help="'R|' regular, 'R|PO|' +postseason, 'S|' spring")
    ap.add_argument("--force", action="store_true", help="re-download cached chunks")
    args = ap.parse_args()

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    chunks = list(date_chunks(args.start, args.end, args.chunk_days))
    print(f"Statcast pull: {args.start} .. {args.end}  ({len(chunks)} chunks, "
          f"{args.chunk_days}d each, method={args.method})")

    use_pyb = args.method in ("auto", "pybaseball")
    if use_pyb:
        try:
            import pybaseball  # noqa: F401
        except ImportError:
            if args.method == "pybaseball":
                sys.exit("pybaseball not installed:  pip install pybaseball")
            print("  pybaseball not installed -> using direct method")
            use_pyb = False

    total, failures = 0, []
    for i, (cs, ce) in enumerate(chunks, 1):
        path = CHUNK_DIR / f"statcast_{cs}_{ce}.parquet"
        if path.exists() and not args.force:
            n = len(pd.read_parquet(path, columns=["game_pk"]))
            total += n
            print(f"  [{i:>3}/{len(chunks)}] {cs}..{ce}  cached ({n:,})")
            continue

        print(f"  [{i:>3}/{len(chunks)}] {cs}..{ce}", end="  ", flush=True)
        df = None
        if use_pyb:
            try:
                df = fetch_pybaseball(cs, ce)
            except Exception as e:
                print(f"pybaseball failed ({type(e).__name__})", end=" -> ")
                df = None
                if args.method == "auto":
                    use_pyb = False  # don't keep retrying a broken path

        if df is None:
            df = fetch_direct(session, cs, ce)
            if df is None:
                print("403 from Savant.")
                print("      Savant is blocking you. Options: wait a few minutes, "
                      "raise --sleep, switch networks, or lower --chunk-days.")
                failures.append((cs, ce))
                time.sleep(30)
                continue

        if df is None or df.empty:
            print("0 rows (off day?)")
            time.sleep(args.sleep)
            continue

        df.to_parquet(path, index=False)
        total += len(df)
        print(f"{len(df):,} pitches")
        time.sleep(args.sleep)

    print(f"\nDownloaded {total:,} pitches across {len(chunks)} chunks.")
    if failures:
        print(f"{len(failures)} chunks failed — re-run this script to retry them:")
        for cs, ce in failures[:10]:
            print(f"    {cs} .. {ce}")
        return

    # ---------------- consolidate into monthly parquet ----------------
    print("\nConsolidating into monthly files...")
    files = sorted(CHUNK_DIR.glob("statcast_*.parquet"))
    if not files:
        sys.exit("No chunks on disk.")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df["game_date"] = pd.to_datetime(df["game_date"])
    before = len(df)
    key = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in df.columns]
    if len(key) == 3:
        df = df.drop_duplicates(subset=key, keep="last")
    df = df.sort_values(key or ["game_date"]).reset_index(drop=True)

    for month, g in df.groupby(df["game_date"].dt.to_period("M")):
        out = OUT_DIR / f"statcast_2026_{month}.parquet"
        g.to_parquet(out, index=False)
        print(f"  {out.name}: {len(g):,} pitches")

    (OUT_DIR / "statcast_2026_columns.txt").write_text(
        "\n".join(sorted(df.columns.tolist())))

    print(f"\nTotal: {len(df):,} pitches  ({before - len(df):,} duplicates dropped)")
    print(f"Games: {df['game_pk'].nunique():,}")
    print(f"Dates: {df['game_date'].min():%Y-%m-%d} .. {df['game_date'].max():%Y-%m-%d}")
    print(f"Columns: {len(df.columns)}")

    # The Week 3 question, answered against the full season.
    kw = ("abs", "challeng", "review", "overturn", "robo", "hawkeye")
    hits = [c for c in df.columns if any(k in c.lower() for k in kw)]
    print("\nABS-related columns:", ", ".join(hits) if hits else
          "NONE — challenge outcomes must come from script 02 and be joined on "
          "game_pk + at_bat_number + pitch_number.")

    if "description" in df.columns:
        called = df["description"].isin(["called_strike", "ball", "blocked_ball"]).sum()
        print(f"Called pitches (the challenge-opportunity pool): {called:,} "
              f"({called/len(df):.1%})")


if __name__ == "__main__":
    main()
