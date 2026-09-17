#!/usr/bin/env python3
"""
07_player_bios.py — player biographical data, mainly BATTER HEIGHT.

Do not skip this one. The 2026 ABS zone is defined off the batter's height:

    top    = 53.5% of the batter's height
    bottom = 27%   of the batter's height
    width  = 17 inches (the plate), and any part of the ball touching it is a strike

Statcast's `sz_top` / `sz_bot` are a DIFFERENT zone — they vary with stance and
are set per pitch by the tracking system. If you compute distance-from-the-edge
using Statcast's zone, you are measuring the wrong zone, and the error will
correlate with batter height, stance, and count. That is exactly the kind of
quiet bias that survives every check except the one nobody runs.

So: pull heights, build the real ABS zone, and keep Statcast's version beside it
as a robustness check.

    python 07_player_bios.py

Output: data/raw/players/player_bios_2026.parquet
        data/raw/players/abs_zone_by_player.csv
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
STATCAST_DIR = DATA_DIR / "raw" / "statcast"
OUT_DIR = DATA_DIR / "raw" / "players"

API = "https://statsapi.mlb.com/api/v1"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

ZONE_TOP_PCT = 0.535
ZONE_BOT_PCT = 0.27

HEIGHT_RE = re.compile(r"(\d+)\s*'\s*(\d+(?:\.\d+)?)?")


def parse_height(s):
    """'6\\' 2\"' -> 6.1667 feet. Returns None if unparseable."""
    if not s or not isinstance(s, str):
        return None
    m = HEIGHT_RE.search(s)
    if not m:
        return None
    feet = int(m.group(1))
    inches = float(m.group(2)) if m.group(2) else 0.0
    return feet + inches / 12.0


def collect_player_ids():
    files = sorted(STATCAST_DIR.glob("statcast_2026_*.parquet"))
    if not files:
        raise SystemExit(
            f"No Statcast files in {STATCAST_DIR}. Run 01_statcast_pitches_2026.py first.")
    ids = set()
    for f in files:
        cols = [c for c in ("batter", "pitcher", "fielder_2") if c in
                pd.read_parquet(f).columns]
        df = pd.read_parquet(f, columns=cols)
        for c in cols:
            ids.update(df[c].dropna().astype("int64").unique().tolist())
    return sorted(ids)


def fetch_people(session, ids, batch=100, sleep=0.4):
    out = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        try:
            r = session.get(f"{API}/people",
                            params={"personIds": ",".join(map(str, chunk))}, timeout=60)
            r.raise_for_status()
            people = r.json().get("people", [])
        except Exception as e:
            print(f"  batch {i//batch + 1} failed: {type(e).__name__}")
            time.sleep(3)
            continue

        for p in people:
            out.append({
                "player_id": p.get("id"),
                "full_name": p.get("fullName"),
                "height_raw": p.get("height"),
                "height_ft": parse_height(p.get("height")),
                "weight_lb": p.get("weight"),
                "bat_side": (p.get("batSide") or {}).get("code"),
                "pitch_hand": (p.get("pitchHand") or {}).get("code"),
                "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
                "birth_date": p.get("birthDate"),
                "mlb_debut_date": p.get("mlbDebutDate"),
            })
        print(f"  [{min(i+batch, len(ids))}/{len(ids)}] players fetched")
        time.sleep(sleep)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "player_bios_2026.parquet"
    if out_path.exists() and not args.force:
        df = pd.read_parquet(out_path)
        print(f"Already have {len(df):,} players at {out_path} (use --force to refresh)")
    else:
        print("Collecting player ids from the Statcast pull...")
        ids = collect_player_ids()
        print(f"  {len(ids):,} unique players")

        session = requests.Session()
        session.headers.update({"User-Agent": UA, "Accept": "application/json"})
        df = fetch_people(session, ids, sleep=args.sleep)
        if df.empty:
            raise SystemExit("No player data returned.")
        df.to_parquet(out_path, index=False)
        print(f"\nSaved {len(df):,} players -> {out_path}")

    # ---- the thing this script exists for ----
    df["abs_zone_top_ft"] = df["height_ft"] * ZONE_TOP_PCT
    df["abs_zone_bot_ft"] = df["height_ft"] * ZONE_BOT_PCT
    df["abs_zone_height_ft"] = df["abs_zone_top_ft"] - df["abs_zone_bot_ft"]

    zone_cols = ["player_id", "full_name", "height_raw", "height_ft",
                 "abs_zone_top_ft", "abs_zone_bot_ft", "abs_zone_height_ft",
                 "bat_side", "primary_position"]
    df[zone_cols].to_csv(OUT_DIR / "abs_zone_by_player.csv", index=False)

    missing = df["height_ft"].isna().sum()
    print(f"\nHeight parsed for {len(df)-missing:,}/{len(df):,} players")
    if missing:
        print(f"  {missing} unparsed — inspect: "
              f"{df.loc[df['height_ft'].isna(), 'height_raw'].dropna().unique()[:5]}")

    h = df["height_ft"].dropna()
    print(f"\nHeights: min {h.min():.2f} ft, median {h.median():.2f} ft, "
          f"max {h.max():.2f} ft")
    print(f"ABS zone top:    {df['abs_zone_top_ft'].min():.2f} .. "
          f"{df['abs_zone_top_ft'].max():.2f} ft")
    print(f"ABS zone bottom: {df['abs_zone_bot_ft'].min():.2f} .. "
          f"{df['abs_zone_bot_ft'].max():.2f} ft")
    print(f"ABS zone height: {df['abs_zone_height_ft'].mean():.2f} ft mean "
          f"({df['abs_zone_height_ft'].mean()*12:.1f} inches)")

    print("""
SANITY CHECK
  A 6'0" batter gets a zone from 1.62 ft to 3.21 ft — about 19 inches tall.
  If your numbers look nothing like that, the height parsing is wrong.

WHY THIS MATTERS
  The ABS zone is height-based and fixed for a given batter. Statcast's zone
  moves with stance. Compare the two in Week 6: where they disagree is precisely
  where umpire calls and ABS rulings diverge — which is to say, where the
  challenges are. That comparison is worth a figure in the interim memo.""")


if __name__ == "__main__":
    main()
