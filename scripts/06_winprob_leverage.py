#!/usr/bin/env python3
"""
06_winprob_leverage.py — win probability and leverage index per plate appearance.

This feeds two things:
  - RQ2's V(s): the value of the game state you are choosing between.
  - The leverage covariate the whole project uses to separate "likely to succeed"
    from "worth succeeding at".

GRAIN WARNING, and it matters: this endpoint reports one row per PLATE
APPEARANCE (atBatIndex), not per pitch. A challenge happens mid-PA, so this
gives you the state entering the PA, not the state at the pitch.

Three ways to get to pitch level, in order of preference:
  1. Statcast may already carry `home_win_exp` / `bat_win_exp` /
     `delta_home_win_exp` per pitch. Script 01 tells you whether it does. If it
     does, use it and keep this file as a cross-check.
  2. Build your own count-aware win-probability model in Week 10 — which your
     timeline already schedules, and which you need anyway for the transition
     tables.
  3. Use PA-level values as a leverage proxy only, and say so in limitations.

Resumable: one cached JSON per game.

    python 06_winprob_leverage.py --start 2026-04-15 --end 2026-04-15
    python 06_winprob_leverage.py                 # whole season

Output: data/raw/winprob/games/<game_pk>.json
        data/raw/winprob/winprob_2026.parquet
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
OUT_DIR = DATA_DIR / "raw" / "winprob"
GAME_DIR = OUT_DIR / "games"
CHALLENGE_DIR = DATA_DIR / "raw" / "challenges"

API = "https://statsapi.mlb.com/api/v1"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")


def get_json(session, url, params=None, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (2 ** attempt))
                continue
            return None
        except Exception:
            time.sleep(3 * (2 ** attempt))
    return None


def parse_winprob(game_pk, payload):
    rows = []
    plays = payload if isinstance(payload, list) else payload.get("plays", payload)
    if not isinstance(plays, list):
        return rows

    for p in plays:
        about = p.get("about", {}) or {}
        result = p.get("result", {}) or {}
        matchup = p.get("matchup", {}) or {}
        count = p.get("count", {}) or {}
        rows.append({
            "game_pk": game_pk,
            "at_bat_index": p.get("atBatIndex", about.get("atBatIndex")),
            "inning": about.get("inning"),
            "half_inning": (about.get("halfInning") or "").lower(),
            "outs": count.get("outs"),
            "balls": count.get("balls"),
            "strikes": count.get("strikes"),
            "home_score": result.get("homeScore", about.get("homeScore")),
            "away_score": result.get("awayScore", about.get("awayScore")),
            "home_win_prob": p.get("homeTeamWinProbability"),
            "away_win_prob": p.get("awayTeamWinProbability"),
            "home_win_prob_added": p.get("homeTeamWinProbabilityAdded"),
            "leverage_index": p.get("leverageIndex"),
            "batter_id": (matchup.get("batter") or {}).get("id"),
            "pitcher_id": (matchup.get("pitcher") or {}).get("id"),
            "event": result.get("event"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="unused if --games-file exists")
    ap.add_argument("--end", default=None)
    ap.add_argument("--games-file",
                    default=str(CHALLENGE_DIR / "games_2026.parquet"),
                    help="game index written by 02_abs_challenges_statsapi.py")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gf = Path(args.games_file)
    if not gf.exists():
        raise SystemExit(
            f"Missing {gf}. Run 02_abs_challenges_statsapi.py first — it builds "
            "the game index this script iterates over.")

    games = pd.read_parquet(gf)
    games["game_date"] = pd.to_datetime(games["game_date"])
    if args.start:
        games = games[games["game_date"] >= pd.Timestamp(args.start)]
    if args.end:
        games = games[games["game_date"] <= pd.Timestamp(args.end)]
    if games.empty:
        raise SystemExit("No games in that window.")

    GAME_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept": "application/json"})

    print(f"Win probability for {len(games):,} games")
    all_rows, failed = [], []
    n = len(games)
    for i, pk in enumerate(games["game_pk"].tolist(), 1):
        cache = GAME_DIR / f"{pk}.json"
        if cache.exists() and not args.force:
            rows = json.loads(cache.read_text())
        else:
            payload = get_json(session, f"{API}/game/{pk}/winProbability")
            if payload is None:
                failed.append(pk)
                rows = []
            else:
                rows = parse_winprob(pk, payload)
                cache.write_text(json.dumps(rows))
            time.sleep(args.sleep)
        all_rows.extend(rows)
        if i % 200 == 0 or i == n:
            print(f"  [{i}/{n}] {len(all_rows):,} plate appearances")

    if not all_rows:
        raise SystemExit("Nothing returned. Inspect one cached JSON by hand.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["game_pk", "at_bat_index"]).reset_index(drop=True)

    # Value from the batting team's point of view — that is the side deciding
    # whether a batter's challenge is worth it.
    df["bat_win_prob"] = df.apply(
        lambda r: r["home_win_prob"] if r["half_inning"] == "bottom"
        else r["away_win_prob"], axis=1)

    out = OUT_DIR / "winprob_2026.parquet"
    df.to_parquet(out, index=False)

    print(f"\nSaved {len(df):,} plate appearances across "
          f"{df['game_pk'].nunique():,} games -> {out}")
    if failed:
        print(f"{len(failed)} games failed; re-run to retry. e.g. {failed[:5]}")

    li = df["leverage_index"].dropna()
    print(f"\nLeverage index: n={len(li):,}  mean={li.mean():.3f}  median={li.median():.3f}")
    print(f"  high leverage (LI > 1.5): {(li > 1.5).mean():.1%} of PAs")
    print(f"  low leverage  (LI < 0.5): {(li < 0.5).mean():.1%} of PAs")
    print("  SANITY: mean LI should sit near 1.0 by construction. If it doesn't, "
          "check that you are not double-counting extra innings.")

    wp = df["home_win_prob"].dropna()
    print(f"\nHome win probability: mean={wp.mean():.1f} (expect ~50-54 on a 0-100 "
          f"scale, or ~0.5 if it comes back as a proportion — check the units)")

    print("\nJoin key for the pitch table:  game_pk + at_bat_index")
    print("Statcast's at_bat_number is 1-BASED; this at_bat_index is 0-BASED. "
          "Subtract 1 from Statcast before joining.")


if __name__ == "__main__":
    main()
