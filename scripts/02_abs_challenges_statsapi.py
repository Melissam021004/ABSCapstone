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

    # re-extract every game with the November 2026 extractor (see below)
    python 02_abs_challenges_statsapi.py --force

Where a review is recorded (November 2026 fix): the Stats API puts a review on
the pitch itself, on a separate event after the pitch, or on the plate
appearance as a whole (where a review of strike three or ball four can land).
The first version of this script looked only at the pitch, and its season total
came out about a quarter below MLB's official count of 10,557 challenges. It now
looks in all three places, keeps only ABS challenges (see keep_abs_reviews), and
saves each game's raw feed so the next re-extraction needs no internet.

Output:
    data/raw/challenges/games/<game_pk>.json        (cache of extracted rows)
    data/raw/challenges/pbp/<game_pk>.json.gz       (raw playByPlay feed)
    data/raw/challenges/abs_challenges_2026.parquet (tidy events)
    data/raw/challenges/games_2026.parquet          (game index + umpires)
    data/raw/challenges/review_records_left_out.csv (reviews that aren't ABS challenges)

2026 rules encoded downstream (script 08), recorded here for reference:
  - 2 challenges per team to start the game
  - a SUCCESSFUL challenge is retained; only a failed one is consumed
  - only the batter, catcher, or pitcher may challenge
  - in extra innings a team that has run out gets one more that inning
  - no challenges while a position player is pitching
"""

import argparse
import datetime as dt
import gzip
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

PBP_DIR = OUT_DIR / "pbp"
EXTRACTOR_VERSION = 2              # 2 = looks on the pitch, a separate event and the play
CALLED_CODES = {"B", "C", "*B"}    # called ball, called strike, ball in the dirt

# MLB's official totals for the 2026 regular season (Baseball-Reference's ABS
# challenge analysis, from MLB data; the AP reported the same totals Oct 2, 2026).
OFFICIAL = {
    "games": 2429, "challenges": 10557, "overturned": 5685,
    "per_game": 10557 / 2429, "overturn_rate": 5685 / 10557,
    "role": {"catcher": (5609, 3285), "batter": (4766, 2328), "pitcher": (182, 72)},
    "role_rate": {"catcher": 3285 / 5609, "batter": 2328 / 4766, "pitcher": 72 / 182},
    "source": "Baseball-Reference ABS challenge analysis, Nov 2026",
}
# Team-games by number of challenges (same source): 4,858 team-games.
OFFICIAL_TEAM_GAME = {0: 402, 1: 1147, 2: 1592, 3: 999, 4: 472, 5: 167, 6: 55, 7: 18, 8: 3, 9: 3}


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


def _is_pitch(ev):
    return bool(ev.get("isPitch")) or ev.get("pitchNumber") is not None


def _reviews(rd):
    """A reviewDetails dict plus any reviews nested in its `additionalReviews`."""
    if not isinstance(rd, dict) or not rd:
        return []
    out = [rd]
    for extra in rd.get("additionalReviews") or []:
        if isinstance(extra, dict) and extra:
            out.append(extra)
    return out


def _pitch_at_or_before(events, j):
    """Index of the last pitch at or before event j: the pitch a review refers to."""
    for k in range(min(j, len(events) - 1), -1, -1):
        if _is_pitch(events[k]):
            return k
    return None


def find_reviews(play):
    """
    Every review recorded in one plate appearance, with the pitch it refers to.

    The Stats API can record a review in three places. Versions of this script
    before November 2026 looked only in the first one:
      pitch_event       on the pitch itself
      separate_event    on an event of its own after the pitch -> the last pitch before it
      plate_appearance  on the play (the plate appearance) itself -> its last pitch.
                        A review of the pitch that ENDED the at-bat (strike three,
                        ball four) can land here.
    An event flagged by `hasReview` or by its description but carrying no
    reviewDetails is kept too (its source gets "_flag"), as before.

    Returns dicts: pitch (event index of the reviewed pitch, or None), event (index
    of the event carrying the review, None when it sits on the play), review, source.
    """
    events = play.get("playEvents", []) or []
    found = []
    for j, ev in enumerate(events):
        revs = _reviews(ev.get("reviewDetails"))
        if not revs and not looks_like_challenge(ev):
            continue
        on_pitch = _is_pitch(ev)
        src = "pitch_event" if on_pitch else "separate_event"
        k = j if on_pitch else _pitch_at_or_before(events, j)
        for rd in revs or [{}]:
            found.append({"pitch": k, "event": j, "review": rd,
                          "source": src if revs else src + "_flag"})
    if events:
        k = _pitch_at_or_before(events, len(events) - 1)
        for rd in _reviews(play.get("reviewDetails")):
            found.append({"pitch": k, "event": None, "review": rd,
                          "source": "plate_appearance"})
    return found


def extract_challenges(game_pk, pbp, game_meta, catcher_by_half=None):
    """One tidy row per review record in a playByPlay payload. keep_abs_reviews()
    then keeps the ABS challenges among them."""
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
        for f in find_reviews(play):
            k, j, rd, source = f["pitch"], f["event"], f["review"], f["source"]
            # The reviewed pitch supplies the pitch facts; the carrier (the event
            # or play the review was recorded on) supplies the text and the flag.
            pev = events[k] if k is not None else (events[j] if j is not None else {})
            det = pev.get("details", {}) or {}
            cnt = pev.get("count", {}) or {}
            pd_ = pev.get("pitchData", {}) or {}
            coord = pd_.get("coordinates", {}) or {}
            if j is not None:
                cdet = events[j].get("details", {}) or {}
                desc, flag = cdet.get("description"), bool(cdet.get("hasReview"))
            else:
                desc = (play.get("result", {}) or {}).get("description")
                flag = bool(about.get("hasReview"))

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
            prev = events[k - 1] if (k is not None and k > 0) else None
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
                "play_event_index": j if j is not None else k,
                "pitch_number": pev.get("pitchNumber") if k is not None else None,
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
                "description": desc,
                "has_review_flag": flag,
                "review_source": source,           # where the review was recorded
                "reviewed_event_index": k,         # the pitch it refers to

                # Stats API pitch location — a useful cross-check on Statcast.
                "api_px": coord.get("pX"),
                "api_pz": coord.get("pZ"),
                "api_sz_top": pd_.get("strikeZoneTop"),
                "api_sz_bot": pd_.get("strikeZoneBottom"),
                "start_time": about.get("startTime"),
            })
    return rows


SOURCE_RANK = {"pitch_event": 0, "separate_event": 1, "plate_appearance": 2,
               "pitch_event_flag": 3, "separate_event_flag": 4}


def keep_abs_reviews(df):
    """
    Keep the ABS challenges among the review records, one per pitch.

      * Rows from the old extractor (no `review_source`): kept exactly as before.
      * Every other row needs a challenger — the batter, catcher or pitcher. A
        review with no challenger is someone else's replay review (a manager's
        challenge of a stolen base on that pitch, say), not an ABS challenge.
        On her full 2026 run, that left out 119 records and brought the role
        totals to within a few of MLB's (Nov 16, 2026).
      * On the pitch event: kept, with that one condition.
      * Found elsewhere (a separate event, or the plate appearance): kept only if
        the pitch it refers to was a called ball or strike, and its reviewType is
        one that player challenges recorded on called pitches use. That keeps a
        manager's replay review of a tag play or a catch out of the ball-strike
        challenges.
      * Separate events flagged only by their text: left out (no outcome, no
        challenger), but returned with the other left-out rows so you can look.
      * The same challenge reported twice (on the pitch and on the play): kept
        once, preferring the record on the pitch.

    Returns (kept rows, left-out rows, the ABS review types learned).
    """
    if "review_source" not in df.columns:
        return df, df.iloc[0:0], set()
    legacy = df["review_source"].isna()
    src = df["review_source"].fillna("old extractor")
    called = df["call_code"].isin(CALLED_CODES)
    known = df["is_overturned"].notna()
    has_challenger = df["challenger_role"].notna()
    abs_types = set(df.loc[src.eq("pitch_event") & called & known & has_challenger, "review_type"]
                    .dropna().astype(str))
    rtype = df["review_type"]
    type_ok = (rtype.notna() & rtype.astype(str).isin(abs_types)) if abs_types else \
        pd.Series(True, index=df.index)
    elsewhere = src.isin(["separate_event", "plate_appearance"])
    keep = legacy | (has_challenger &
                     (src.str.startswith("pitch_event") | (elsewhere & called & type_ok)))
    kept = df[keep].copy()
    kept["_rank"] = kept["review_source"].map(SOURCE_RANK).fillna(-1)   # legacy first
    keyed = kept["pitch_number"].notna() & ~kept["review_source"].isna()
    one = (kept[keyed].sort_values("_rank", kind="stable")
           .drop_duplicates(["game_pk", "at_bat_index", "pitch_number"], keep="first"))
    out = pd.concat([kept[~keyed], one]).drop(columns="_rank").sort_index()
    left_out = df[~df.index.isin(out.index)]
    return out, left_out, abs_types


def load_cached(cache, raw, args):
    """
    Rows for one game from the caches, or None if the game must be (re-)extracted.
    Returns (rows, from_old_extractor).
    """
    if args.force or args.redownload or not cache.exists():
        return None, False
    data = json.loads(cache.read_text())
    if isinstance(data, dict) and data.get("extractor") == EXTRACTOR_VERSION:
        return data.get("rows", []), False
    if raw.exists():
        return None, False                    # old rows, but the feed is saved: redo it
    return (data if isinstance(data, list) else data.get("rows", [])), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=SEASON_START)
    ap.add_argument("--end", default=SEASON_END)
    ap.add_argument("--game-types", default="R",
                    help="R regular, 'R,P' + postseason, S spring")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--force", action="store_true",
                    help="re-extract every game: from the saved raw feed when there is "
                         "one, otherwise download it again")
    ap.add_argument("--redownload", action="store_true",
                    help="download every game again even if its raw feed is saved")
    args = ap.parse_args()

    GAME_DIR.mkdir(parents=True, exist_ok=True)
    PBP_DIR.mkdir(parents=True, exist_ok=True)
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
    n_done = n_old = 0
    for i, meta in enumerate(games.to_dict("records"), 1):
        pk = meta["game_pk"]
        cache = GAME_DIR / f"{pk}.json"
        raw = PBP_DIR / f"{pk}.json.gz"

        rows, old = load_cached(cache, raw, args)
        if rows is None:
            pbp = None
            if raw.exists() and not args.redownload:
                pbp = json.loads(gzip.decompress(raw.read_bytes()))
            if pbp is None:
                pbp = get_json(session, f"{API}/game/{pk}/playByPlay")
                if pbp is None:
                    print(f"  [{i}/{n}] {pk}  FAILED — re-run to retry")
                    continue
                raw.write_bytes(gzip.compress(json.dumps(pbp).encode()))
                time.sleep(args.sleep)
            rows = extract_challenges(pk, pbp, meta)
            cache.write_text(json.dumps({"extractor": EXTRACTOR_VERSION, "rows": rows}))
        n_done += 1
        n_old += old
        for r in rows:
            r.setdefault("review_source", None)    # old-extractor rows
        all_rows.extend(rows)
        if i % 100 == 0 or i == n:
            print(f"  [{i}/{n}] games processed, {len(all_rows):,} review records so far")

    if not all_rows:
        raise SystemExit("No challenges extracted. Inspect a cached game JSON.")

    found = pd.DataFrame(all_rows)
    df, left_out, abs_types = keep_abs_reviews(found)
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_pk", "at_bat_index", "play_event_index"]) \
           .reset_index(drop=True)
    # Chronological order within a game — script 08 needs this to walk the
    # challenge inventory down.
    df["challenge_seq_in_game"] = df.groupby("game_pk").cumcount() + 1

    out = OUT_DIR / "abs_challenges_2026.parquet"
    df.to_parquet(out, index=False)
    if len(left_out):
        left_out.to_csv(OUT_DIR / "review_records_left_out.csv", index=False)

    # ------------------------------ QA ------------------------------
    print("\n" + "=" * 62)
    print(f"{len(df):,} challenges across {n_done:,} games "
          f"({len(df)/max(n_done, 1):.2f} per game)")

    src = found["review_source"].fillna("old extractor")
    print("\nWhere the reviews were recorded (before keeping the ABS challenges):")
    print(src.value_counts().to_string())
    if (src != "old extractor").any():
        print(f"ABS review types (learned from reviews on called pitches): "
              f"{', '.join(sorted(abs_types)) or 'none recorded'}")
        if not abs_types:
            print("  [CHECK] no reviewType on pitch-event reviews, so reviews found on a "
                  "separate event or the plate appearance were kept on the called-pitch "
                  "and challenger tests alone. Spot-check a few on video.")
        if len(left_out):
            print(f"Left out {len(left_out):,} records (other replay reviews, text-only flags, "
                  f"duplicates) -> {OUT_DIR / 'review_records_left_out.csv'}")
        kept_src = df["review_source"].fillna("old extractor").value_counts()
        print("Challenges kept, by where they were recorded: "
              + ", ".join(f"{k} {v:,}" for k, v in kept_src.items()))
    if n_old:
        print(f"\n  [CHECK] {n_old:,} games used rows cached by the OLD extractor, which missed "
              "reviews recorded on a separate event or on the plate appearance (e.g. on strike "
              "three or ball four). Re-run with --force: it downloads each of those games once "
              "more (about 20–30 minutes for a season) and saves the raw feed, so it's the last "
              "time.")

    ovr = df["is_overturned"]
    known = ovr.notna().sum()
    print(f"\nOutcome known for {known:,}/{len(df):,} challenges")
    if known:
        print(f"Overall overturn rate: {ovr.astype(float).mean():.1%}")

    print("\nBy challenger role:")
    by_role = df.groupby("challenger_role", dropna=False).agg(
        n=("is_overturned", "size"),
        overturn_rate=("is_overturned", lambda s: s.astype(float).mean()))
    print(by_role.to_string())

    # MLB's official numbers for the 2026 regular season. Your extraction should
    # match them within a couple of percent.
    if set(str(args.game_types).split(",")) == {"R"}:
        per_game = len(df) / max(n_done, 1)
        print(f"\nMLB's official 2026 regular season: {OFFICIAL['challenges']:,} challenges in "
              f"{OFFICIAL['games']:,} games ({OFFICIAL['per_game']:.2f} a game), "
              f"{OFFICIAL['overturn_rate']:.1%} overturned — catchers "
              f"{OFFICIAL['role_rate']['catcher']:.1%}, batters {OFFICIAL['role_rate']['batter']:.1%}, "
              f"pitchers {OFFICIAL['role_rate']['pitcher']:.1%} ({OFFICIAL['source']}).")
        ok = per_game >= 0.98 * OFFICIAL["per_game"]
        print(f"  [{'ok' if ok else 'CHECK':>5}] this extraction: {per_game:.2f} a game"
              + ("" if ok else " — challenges are missing; see scripts/24_challenge_count_check.py"))

    print("\nBy inning (top 10):")
    print(df.groupby("inning").agg(
        n=("is_overturned", "size"),
        overturn_rate=("is_overturned", lambda s: s.astype(float).mean())).head(10).to_string())

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

    # The rule: in innings 1–9 a team can fail at most twice (a successful
    # challenge is kept, so a team CAN make 5+ challenges — MLB had 246 such
    # team-games in 2026 — but never a third failure in regulation).
    sided = df.dropna(subset=["challenging_side"])
    failed = sided["is_overturned"].map({True: True, False: False}).eq(False)
    reg = sided[failed & (pd.to_numeric(sided["inning"], errors="coerce") <= 9)]
    over_budget = int((reg.groupby(["game_pk", "challenging_side"]).size() > 2).sum())
    print(f"  [{'ok' if over_budget == 0 else 'CHECK':>5}] game-sides with more than 2 failed "
          f"challenges in innings 1–9 (the rules allow 2): {over_budget:,}")

    # Challenges per team-game against MLB's distribution (team-games with none
    # count as 0).
    if set(str(args.game_types).split(",")) == {"R"}:
        per_side = sided.groupby(["game_pk", "challenging_side"]).size()
        dist = per_side.value_counts().sort_index()
        dist.loc[0] = 2 * n_done - len(per_side)
        top = max(int(dist.index.max()), max(OFFICIAL_TEAM_GAME))
        tab = pd.DataFrame({"this extraction": dist.reindex(range(top + 1), fill_value=0),
                            "MLB": pd.Series(OFFICIAL_TEAM_GAME).reindex(range(top + 1), fill_value=0)})
        tab.index.name = "challenges in a team-game"
        print("\nChallenges per team-game, against MLB's distribution (Baseball-Reference):")
        print(tab.to_string())

    print(f"\nSaved: {out}")
    print("=" * 62)


if __name__ == "__main__":
    main()
