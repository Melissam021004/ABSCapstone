#!/usr/bin/env python3
"""
02_abs_challenges_statsapi.py — every ABS challenge event of 2026.

The MLB Stats API is the authoritative source for WHO challenged and WHETHER
the call was overturned. Baseball Savant publishes aggregates; the Stats API
publishes the events. Your model target (`was_overturned`) comes from here.

Resumable: one cached JSON per game. Re-run to pick up where it stopped.

    # Week 3: one day
    python 02_abs_challenges_statsapi.py --start 2026-04-15 --end 2026-04-15

    # Week 4: the whole season
    python 02_abs_challenges_statsapi.py

Output:
    data/raw/challenges/games/<game_pk>.json        (cache)
    data/raw/challenges/abs_challenges_2026.parquet (tidy events)
    data/raw/challenges/games_2026.parquet          (game index + umpires)

2026 rules encoded downstream (script 08), recorded here for reference:
  - 2 challenges per team to start the game
  - a SUCCESSFUL challenge is retained; only a failed one is consumed
  - only the batter, catcher, or pitcher may challenge
  - in extra innings a team that has run out gets one more that inning
  - no challenges while a position player is pitching
"""

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
OUT_DIR = DATA_DIR / "raw" / "challenges"
GAME_DIR = OUT_DIR / "games"

API = "https://statsapi.mlb.com/api/v1"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

SEASON_START = "2026-03-25"
SEASON_END = "2026-09-28"

CHALLENGE_WORDS = ("challenge", "challenged", "challenges", "abs review",
                   "automated ball-strike")


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json"})
    return s


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


def fetch_schedule(session, start, end, game_types="R"):
    """Final games in the window, with the home-plate umpire attached."""
    out = []
    cur = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    while cur <= end_d:
        stop = min(cur + dt.timedelta(days=13), end_d)
        js = get_json(session, f"{API}/schedule", {
            "sportId": 1, "startDate": cur.isoformat(), "endDate": stop.isoformat(),
            "gameType": game_types, "hydrate": "officials,linescore,team",
        })
        for d in (js or {}).get("dates", []):
            for g in d.get("games", []):
                st = g.get("status", {}).get("abstractGameState")
                if st not in ("Final", "Game Over"):
                    continue
                hp = hp_id = None
                for o in g.get("officials", []) or []:
                    if o.get("officialType") == "Home Plate":
                        hp = (o.get("official") or {}).get("fullName")
                        hp_id = (o.get("official") or {}).get("id")
                        break
                ls = g.get("linescore", {}) or {}
                out.append({
                    "game_pk": g["gamePk"],
                    "game_date": d["date"],
                    "game_type": g.get("gameType"),
                    "home_team": (g.get("teams", {}).get("home", {})
                                  .get("team", {}).get("abbreviation")),
                    "away_team": (g.get("teams", {}).get("away", {})
                                  .get("team", {}).get("abbreviation")),
                    "home_team_id": (g.get("teams", {}).get("home", {})
                                     .get("team", {}).get("id")),
                    "away_team_id": (g.get("teams", {}).get("away", {})
                                     .get("team", {}).get("id")),
                    "hp_umpire": hp,
                    "hp_umpire_id": hp_id,
                    "n_innings": ls.get("currentInning"),
                    "venue_id": (g.get("venue") or {}).get("id"),
                })
        cur = stop + dt.timedelta(days=1)
        time.sleep(0.4)
    return pd.DataFrame(out).drop_duplicates(subset=["game_pk"])


def looks_like_challenge(event):
    det = event.get("details", {}) or {}
    if det.get("hasReview"):
        return True
    desc = (det.get("description") or "").lower()
    return any(w in desc for w in CHALLENGE_WORDS)


def extract_challenges(game_pk, pbp, game_meta, catcher_by_half=None):
    """Pull one tidy row per challenge event out of a playByPlay payload."""
    rows = []
    for play in pbp.get("allPlays", []) or []:
        about = play.get("about", {}) or {}
        matchup = play.get("matchup", {}) or {}
        ab_index = about.get("atBatIndex")
        inning = about.get("inning")
        half = (about.get("halfInning") or "").lower()   # 'top' / 'bottom'

        batter = matchup.get("batter", {}) or {}
        pitcher = matchup.get("pitcher", {}) or {}

        events = play.get("playEvents", []) or []
        for ev_i, ev in enumerate(events):
            if not looks_like_challenge(ev):
                continue

            det = ev.get("details", {}) or {}
            rd = ev.get("reviewDetails", {}) or {}
            cnt = ev.get("count", {}) or {}
            pd_ = ev.get("pitchData", {}) or {}
            coord = pd_.get("coordinates", {}) or {}

            challenger = rd.get("player") or {}
            challenger_id = challenger.get("id")
            challenger_name = challenger.get("fullName")

            # Only the batter, catcher, or pitcher may challenge, so anyone who
            # is neither the batter nor the pitcher must be the catcher.
            if challenger_id is None:
                role = None
            elif challenger_id == batter.get("id"):
                role = "batter"
            elif challenger_id == pitcher.get("id"):
                role = "pitcher"
            else:
                role = "catcher"

            # Batting side challenges; everyone else is the fielding side.
            if role == "batter":
                challenging_side = "away" if half == "top" else "home"
            elif role in ("pitcher", "catcher"):
                challenging_side = "home" if half == "top" else "away"
            else:
                challenging_side = None

            # The count on an event is AFTER the pitch resolves; reconstruct
            # the count the decision was actually made under.
            prev = events[ev_i - 1] if ev_i > 0 else None
            prev_cnt = (prev or {}).get("count", {}) or {}
            balls_before = prev_cnt.get("balls", 0) if prev else 0
            strikes_before = prev_cnt.get("strikes", 0) if prev else 0

            rows.append({
                "game_pk": game_pk,
                "game_date": game_meta.get("game_date"),
                "home_team": game_meta.get("home_team"),
                "away_team": game_meta.get("away_team"),
                "hp_umpire": game_meta.get("hp_umpire"),
                "hp_umpire_id": game_meta.get("hp_umpire_id"),

                "at_bat_index": ab_index,          # 0-based (Statcast is 1-based)
                "play_event_index": ev_i,
                "pitch_number": ev.get("pitchNumber"),
                "inning": inning,
                "half_inning": half,
                "is_extra_inning": bool(inning and inning > 9),

                "balls_before": balls_before,
                "strikes_before": strikes_before,
                "balls_after": cnt.get("balls"),
                "strikes_after": cnt.get("strikes"),
                "outs": cnt.get("outs"),

                "batter_id": batter.get("id"),
                "batter_name": batter.get("fullName"),
                "pitcher_id": pitcher.get("id"),
                "pitcher_name": pitcher.get("fullName"),

                "challenger_id": challenger_id,
                "challenger_name": challenger_name,
                "challenger_role": role,
                "challenging_side": challenging_side,

                "call_code": (det.get("call") or {}).get("code"),
                "call_desc": (det.get("call") or {}).get("description"),
                "is_overturned": rd.get("isOverturned"),
                "review_type": rd.get("reviewType"),
                "challenge_team_id": rd.get("challengeTeamId"),
                "description": det.get("description"),
                "has_review_flag": bool(det.get("hasReview")),

                # Stats API pitch location — a useful cross-check on Statcast.
                "api_px": coord.get("pX"),
                "api_pz": coord.get("pZ"),
                "api_sz_top": pd_.get("strikeZoneTop"),
                "api_sz_bot": pd_.get("strikeZoneBottom"),
                "start_time": about.get("startTime"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=SEASON_START)
    ap.add_argument("--end", default=SEASON_END)
    ap.add_argument("--game-types", default="R",
                    help="R regular, 'R,P' + postseason, S spring")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    GAME_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    print(f"Schedule: {args.start} .. {args.end}")
    games = fetch_schedule(session, args.start, args.end, args.game_types)
    if games.empty:
        raise SystemExit("No final games found in that window.")
    games.to_parquet(OUT_DIR / "games_2026.parquet", index=False)
    print(f"  {len(games):,} final games")

    missing_ump = games["hp_umpire"].isna().sum()
    if missing_ump:
        print(f"  WARNING: {missing_ump} games missing a home-plate umpire")

    all_rows = []
    n = len(games)
    for i, meta in enumerate(games.to_dict("records"), 1):
        pk = meta["game_pk"]
        cache = GAME_DIR / f"{pk}.json"

        if cache.exists() and not args.force:
            rows = json.loads(cache.read_text())
        else:
            pbp = get_json(session, f"{API}/game/{pk}/playByPlay")
            if pbp is None:
                print(f"  [{i}/{n}] {pk}  FAILED — re-run to retry")
                continue
            rows = extract_challenges(pk, pbp, meta)
            cache.write_text(json.dumps(rows))
            time.sleep(args.sleep)

        all_rows.extend(rows)
        if i % 100 == 0 or i == n:
            print(f"  [{i}/{n}] games processed, {len(all_rows):,} challenges so far")

    if not all_rows:
        raise SystemExit("No challenges extracted. Inspect a cached game JSON.")

    df = pd.DataFrame(all_rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_pk", "at_bat_index", "play_event_index"]) \
           .reset_index(drop=True)
    # Chronological order within a game — script 08 needs this to walk the
    # challenge inventory down.
    df["challenge_seq_in_game"] = df.groupby("game_pk").cumcount() + 1

    out = OUT_DIR / "abs_challenges_2026.parquet"
    df.to_parquet(out, index=False)

    # ------------------------------ QA ------------------------------
    print("\n" + "=" * 62)
    print(f"{len(df):,} challenges across {df['game_pk'].nunique():,} games "
          f"({len(df)/max(df['game_pk'].nunique(),1):.2f} per game)")

    ovr = df["is_overturned"]
    known = ovr.notna().sum()
    print(f"\nOutcome known for {known:,}/{len(df):,} challenges")
    if known:
        print(f"Overall overturn rate: {ovr.mean():.1%}")

    print("\nBy challenger role:")
    by_role = df.groupby("challenger_role", dropna=False).agg(
        n=("is_overturned", "size"), overturn_rate=("is_overturned", "mean"))
    print(by_role.to_string())

    print("\nBy inning (top 10):")
    print(df.groupby("inning").agg(
        n=("is_overturned", "size"),
        overturn_rate=("is_overturned", "mean")).head(10).to_string())

    print("\nBy count before the pitch:")
    cnt = df.groupby(["balls_before", "strikes_before"]).size() \
            .sort_values(ascending=False).head(8)
    print(cnt.to_string())

    # Data-quality flags worth knowing before Week 5.
    print("\nData quality:")
    for col, label in [("challenger_name", "challenger identity"),
                       ("challenger_role", "challenger role"),
                       ("is_overturned", "outcome (MODEL TARGET)"),
                       ("pitch_number", "pitch number (join key)"),
                       ("hp_umpire", "home-plate umpire")]:
        miss = df[col].isna().sum()
        status = "ok" if miss == 0 else "CHECK"
        print(f"  [{status:>5}] {label}: {miss:,} missing")

    over_two = (df.groupby(["game_pk", "challenging_side"]).size() > 4).sum()
    if over_two:
        print(f"  [CHECK] {over_two} game-sides with >4 challenges — expected only "
              f"in extra innings. Verify against the box score.")

    print(f"\nSaved: {out}")
    print("=" * 62)


if __name__ == "__main__":
    main()
