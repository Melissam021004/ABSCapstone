#!/usr/bin/env python3
"""
05_umpire_data.py — home-plate umpire accuracy, two ways.

Your proposal links umpscorecards.com. That site is a rendered web app with no
documented public API and its own terms of use, so treat it as a nice-to-have.
The primary path here computes umpire accuracy DIRECTLY FROM YOUR OWN STATCAST
PULL, which is better for this project on three counts:

  - reproducible (one of your three stated novelty claims)
  - pitch-level, so it joins to your model table instead of sitting at game level
  - consistent with the ABS zone you actually model, rather than someone else's

Modes:
    --mode derive    compute per-umpire accuracy from data/raw/statcast  [default]
    --mode scrape    try umpscorecards.com
    --mode both

    python 05_umpire_data.py
    python 05_umpire_data.py --mode both

Output: data/raw/umpires/umpire_accuracy_derived_2026.parquet
        data/raw/umpires/umpire_game_accuracy_2026.parquet
        data/raw/umpires/umpscorecards_games.csv        (scrape mode, if it works)
"""

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
STATCAST_DIR = DATA_DIR / "raw" / "statcast"
CHALLENGE_DIR = DATA_DIR / "raw" / "challenges"
OUT_DIR = DATA_DIR / "raw" / "umpires"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

# ---- 2026 ABS zone constants -------------------------------------------------
PLATE_HALF_WIDTH_FT = 8.5 / 12          # 17-inch plate, measured from centre
BALL_RADIUS_FT = (2.9 / 2) / 12         # any part of the ball touching = strike
ZONE_TOP_PCT = 0.535                    # 53.5% of batter height
ZONE_BOT_PCT = 0.27                     # 27% of batter height


def zone_distance_ft(plate_x, plate_z, top, bot):
    """
    Signed distance from the BALL'S SURFACE to the rectangular ABS zone, in feet.
      < 0  ball overlaps the zone  -> ABS strike
      > 0  ball misses the zone    -> ABS ball
    Magnitude is how badly it missed / how far inside it was.
    """
    plate_x = np.asarray(plate_x, dtype=float)
    plate_z = np.asarray(plate_z, dtype=float)
    top = np.asarray(top, dtype=float)
    bot = np.asarray(bot, dtype=float)

    dx = np.abs(plate_x) - PLATE_HALF_WIDTH_FT      # >0 outside horizontally
    dz = np.maximum(bot - plate_z, plate_z - top)   # >0 outside vertically

    inside = (dx <= 0) & (dz <= 0)
    corner = (dx > 0) & (dz > 0)

    d = np.where(
        inside, np.maximum(dx, dz),                          # negative
        np.where(corner, np.hypot(np.maximum(dx, 0), np.maximum(dz, 0)),
                 np.maximum(dx, dz)),
    )
    return d - BALL_RADIUS_FT


def load_statcast():
    files = sorted(STATCAST_DIR.glob("statcast_2026_*.parquet"))
    if not files:
        raise SystemExit(
            f"No Statcast files in {STATCAST_DIR}. Run 01_statcast_pitches_2026.py first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  loaded {len(df):,} pitches from {len(files)} files")
    return df


def derive_umpire_accuracy(heights_path=None):
    """Per-umpire called-strike accuracy against the ABS zone."""
    print("\nDeriving umpire accuracy from Statcast")
    sc = load_statcast()

    games_f = CHALLENGE_DIR / "games_2026.parquet"
    if not games_f.exists():
        raise SystemExit(
            f"Missing {games_f}. Run 02_abs_challenges_statsapi.py first — it "
            "writes the game index with home-plate umpires attached.")
    games = pd.read_parquet(games_f)[
        ["game_pk", "game_date", "hp_umpire", "hp_umpire_id", "home_team", "away_team"]]

    called = sc[sc["description"].isin(["called_strike", "ball", "blocked_ball"])].copy()
    print(f"  {len(called):,} called pitches")

    called = called.merge(games.drop(columns=["game_date"]), on="game_pk", how="left")
    missing = called["hp_umpire"].isna().sum()
    if missing:
        print(f"  WARNING: {missing:,} called pitches have no umpire attached")

    # --- ABS zone from batter height, with a documented fallback ---
    used_heights = False
    if heights_path and Path(heights_path).exists():
        h = pd.read_parquet(heights_path)
        if {"player_id", "height_ft"} <= set(h.columns):
            called = called.merge(
                h[["player_id", "height_ft"]].rename(
                    columns={"player_id": "batter", "height_ft": "batter_height_ft"}),
                on="batter", how="left")
            used_heights = called["batter_height_ft"].notna().mean() > 0.5
            print(f"  batter heights matched for "
                  f"{called['batter_height_ft'].notna().mean():.1%} of pitches")

    if used_heights:
        called["abs_zone_top"] = called["batter_height_ft"] * ZONE_TOP_PCT
        called["abs_zone_bot"] = called["batter_height_ft"] * ZONE_BOT_PCT
        zone_source = "batter height (53.5% / 27%)"
    else:
        called["abs_zone_top"] = called["sz_top"]
        called["abs_zone_bot"] = called["sz_bot"]
        zone_source = "Statcast sz_top/sz_bot  [APPROXIMATION]"
        print("  NOTE: falling back to Statcast's zone. Run 07_player_bios.py and "
              "pass --heights to use the real ABS zone definition.")

    called["zone_dist_ft"] = zone_distance_ft(
        called["plate_x"], called["plate_z"], called["abs_zone_top"], called["abs_zone_bot"])
    called["zone_dist_in"] = called["zone_dist_ft"] * 12
    called["abs_strike"] = called["zone_dist_ft"] < 0
    called["called_strike"] = called["description"] == "called_strike"
    called["correct_call"] = called["abs_strike"] == called["called_strike"]
    # Directional errors: what each side would want to challenge.
    called["missed_strike"] = called["abs_strike"] & ~called["called_strike"]   # pitcher/catcher
    called["missed_ball"] = ~called["abs_strike"] & called["called_strike"]     # batter

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ump = called.groupby(["hp_umpire", "hp_umpire_id"], dropna=True).agg(
        n_called=("correct_call", "size"),
        n_games=("game_pk", "nunique"),
        accuracy=("correct_call", "mean"),
        missed_strike_rate=("missed_strike", "mean"),
        missed_ball_rate=("missed_ball", "mean"),
        mean_abs_dist_in=("zone_dist_in", lambda s: s.abs().mean()),
    ).reset_index().sort_values("accuracy", ascending=False)

    # Borderline pitches are where challenges actually live.
    edge = called[called["zone_dist_in"].abs() <= 2.0]
    edge_acc = edge.groupby(["hp_umpire", "hp_umpire_id"], dropna=True).agg(
        n_borderline=("correct_call", "size"),
        borderline_accuracy=("correct_call", "mean"),
    ).reset_index()
    ump = ump.merge(edge_acc, on=["hp_umpire", "hp_umpire_id"], how="left")

    ump.to_parquet(OUT_DIR / "umpire_accuracy_derived_2026.parquet", index=False)

    game = called.groupby(["game_pk", "hp_umpire", "hp_umpire_id"], dropna=True).agg(
        n_called=("correct_call", "size"),
        accuracy=("correct_call", "mean"),
        n_missed_strikes=("missed_strike", "sum"),
        n_missed_balls=("missed_ball", "sum"),
    ).reset_index()
    game.to_parquet(OUT_DIR / "umpire_game_accuracy_2026.parquet", index=False)

    print(f"\n  zone definition: {zone_source}")
    print(f"  {len(ump)} umpires, {len(game):,} umpire-games")
    print(f"  league-wide called-pitch accuracy: {called['correct_call'].mean():.2%}")
    print(f"  borderline (within 2 in) accuracy: {edge['correct_call'].mean():.2%}")
    print(f"  missed strikes: {called['missed_strike'].mean():.2%}   "
          f"missed balls: {called['missed_ball'].mean():.2%}")

    print("\n  Most accurate:")
    print(ump.head(5)[["hp_umpire", "n_called", "accuracy", "borderline_accuracy"]]
          .to_string(index=False))
    print("\n  Least accurate:")
    print(ump.tail(5)[["hp_umpire", "n_called", "accuracy", "borderline_accuracy"]]
          .to_string(index=False))

    print("\n  SANITY CHECK: published MLB umpire accuracy sits around 92-94% on all "
          "called pitches. If your number is far off, your zone definition is wrong "
          "— that is a bug, not a finding.")
    return ump


def scrape_umpscorecards():
    """Best-effort. The site has no documented API; endpoints may change."""
    print("\nTrying umpscorecards.com")
    print("  Check their terms of use before relying on this in a published paper.")
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/csv, */*",
                      "Referer": "https://umpscorecards.com/data/games"})

    candidates = [
        "https://umpscorecards.com/api/games",
        "https://umpscorecards.com/api/v1/games",
        "https://umpscorecards.com/data/api/games",
        "https://api.umpscorecards.com/games",
        "https://umpscorecards.com/single_umpire_data.csv",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for url in candidates:
        try:
            r = s.get(url, params={"season": 2026}, timeout=30)
            print(f"  {url} -> HTTP {r.status_code}")
            if r.status_code != 200:
                continue
            ct = r.headers.get("Content-Type", "")
            if "json" in ct:
                js = r.json()
                data = js.get("data", js) if isinstance(js, dict) else js
                df = pd.json_normalize(data)
            elif "csv" in ct or url.endswith(".csv"):
                df = pd.read_csv(io.StringIO(r.text))
            else:
                continue
            if df.empty:
                continue
            out = OUT_DIR / "umpscorecards_games.csv"
            df.to_csv(out, index=False)
            print(f"  SUCCESS: {len(df)} rows -> {out}")
            print(f"  columns: {', '.join(df.columns.astype(str)[:25])}")
            return df
        except Exception as e:
            print(f"  {url} -> {type(e).__name__}")
        time.sleep(1.5)

    print("""
  No endpoint responded. This is expected — the site renders client-side.
  Options:
    1. Use --mode derive (the default). It gives you the same construct at
       pitch level and it is reproducible, which your proposal argues for.
    2. Open umpscorecards.com in a browser, watch the Network tab, find the
       request that populates the table, and paste that URL into `candidates`.
    3. Email them. They have historically shared data for academic work — and
       a documented data-use permission is worth having in an appendix.""")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["derive", "scrape", "both"], default="derive")
    ap.add_argument("--heights",
                    default=str(DATA_DIR / "raw" / "players" / "player_bios_2026.parquet"),
                    help="output of 07_player_bios.py — enables the true ABS zone")
    args = ap.parse_args()

    if args.mode in ("derive", "both"):
        derive_umpire_accuracy(args.heights)
    if args.mode in ("scrape", "both"):
        scrape_umpscorecards()


if __name__ == "__main__":
    main()
