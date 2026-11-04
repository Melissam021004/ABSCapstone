#!/usr/bin/env python3
"""
00_env_check.py — Week 3 data-feasibility spike.

Checks that every data source in the capstone proposal is reachable and returns
what it is supposed to return, then writes a report you can paste into the
Week 3 deliverable.

Run this FIRST. Nothing else is worth debugging until this passes.

    python 00_env_check.py
    python 00_env_check.py --date 2026-04-15      # probe a specific game day

Output: data/raw/_feasibility/env_check_<timestamp>.json  + a printed table.
"""

import argparse
import datetime as dt
import importlib
import json
import os
import platform
import sys
from pathlib import Path

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
OUT_DIR = DATA_DIR / "raw" / "_feasibility"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Every source named in the Fall proposal / Capstone outline.
SOURCES = {
    "statsapi_schedule": "https://statsapi.mlb.com/api/v1/schedule",
    "statsapi_playbyplay": "https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay",
    "statsapi_winprob": "https://statsapi.mlb.com/api/v1/game/{game_pk}/winProbability",
    "statsapi_people": "https://statsapi.mlb.com/api/v1/people",
    "savant_statcast_csv": "https://baseballsavant.mlb.com/statcast_search/csv",
    "savant_abs_leaderboard": "https://baseballsavant.mlb.com/leaderboard/abs-challenges",
    "savant_catcher_framing": "https://baseballsavant.mlb.com/leaderboard/catcher-framing",
    "savant_abs_dashboard": "https://baseballsavant.mlb.com/abs",
    "umpscorecards": "https://umpscorecards.com/data/games",
    "abs_scoreboard_mirror": "https://absscoreboard.github.io/abs-data/data/abs-challenges.json",
}

results = []


def record(name, ok, detail, note=""):
    results.append({"check": name, "ok": bool(ok), "detail": str(detail), "note": note})
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: {detail}")
    if note:
        print(f"         -> {note}")


# --------------------------------------------------------------------------
# 1. Environment
# --------------------------------------------------------------------------
def check_environment():
    print("\n1. ENVIRONMENT")
    record("python_version", sys.version_info >= (3, 9), platform.python_version(),
           "" if sys.version_info >= (3, 9) else "pybaseball needs Python >= 3.9")

    required = {
        "pandas": "dataframes",
        "numpy": "math",
        "requests": "HTTP",
        "pyarrow": "parquet output",
        "pybaseball": "Statcast convenience wrapper",
    }
    optional = {
        "catboost": "Week 8 model",
        "lightgbm": "Week 8 model",
        "sklearn": "Weeks 7-10",
        "statsmodels": "Week 7 logistic",
        "pygam": "Week 7 GAM",
        "dash": "Week 12 dashboard",
        "plotly": "Week 12 dashboard",
    }

    for mod, why in required.items():
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "unknown")
            record(f"pkg:{mod}", True, v)
        except ImportError:
            record(f"pkg:{mod}", False, "NOT INSTALLED",
                   f"needed for {why}  ->  pip install {mod}")

    print("\n   Optional (needed later in the semester):")
    for mod, why in optional.items():
        try:
            m = importlib.import_module(mod)
            print(f"     ok      {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"     missing {mod}  (needed for {why})")


# --------------------------------------------------------------------------
# 2. Network reachability
# --------------------------------------------------------------------------
def check_network():
    print("\n2. SOURCE REACHABILITY")
    try:
        import requests
    except ImportError:
        record("network", False, "requests not installed", "cannot test sources")
        return

    s = requests.Session()
    s.headers.update({"User-Agent": UA})

    simple = {
        "statsapi_schedule": (SOURCES["statsapi_schedule"],
                              {"sportId": 1, "date": "2026-04-15"}),
        "savant_abs_leaderboard": (SOURCES["savant_abs_leaderboard"], {"year": 2026}),
        "savant_catcher_framing": (SOURCES["savant_catcher_framing"], {"year": 2026}),
        "savant_abs_dashboard": (SOURCES["savant_abs_dashboard"], {}),
        "umpscorecards": (SOURCES["umpscorecards"], {}),
        "abs_scoreboard_mirror": (SOURCES["abs_scoreboard_mirror"], {}),
    }

    for name, (url, params) in simple.items():
        try:
            r = s.get(url, params=params, timeout=30)
            ok = r.status_code == 200
            note = ""
            if r.status_code == 403:
                note = "403 = blocked. Try a real browser User-Agent, or your campus network."
            record(name, ok, f"HTTP {r.status_code} ({len(r.content):,} bytes)", note)
        except Exception as e:
            record(name, False, type(e).__name__, str(e)[:160])


# --------------------------------------------------------------------------
# 3. The questions Week 3 actually has to answer
# --------------------------------------------------------------------------
def check_statcast_columns(date):
    """THE key question: does the 2026 Statcast export carry ABS/challenge columns?"""
    print("\n3. STATCAST PITCH-LEVEL EXPORT")
    try:
        import pybaseball
        from pybaseball import statcast
    except ImportError:
        record("statcast_pull", False, "pybaseball not installed", "pip install pybaseball")
        return None

    try:
        df = statcast(start_dt=date, end_dt=date, verbose=False)
    except Exception as e:
        record("statcast_pull", False, type(e).__name__,
               f"{str(e)[:160]}  |  if 403, see 01_statcast_pitches_2026.py --method direct")
        return None

    if df is None or df.empty:
        record("statcast_pull", False, "0 rows",
               f"No games on {date}? Try a mid-season date.")
        return None

    record("statcast_pull", True, f"{len(df):,} pitches, {len(df.columns)} columns")

    cols = sorted(df.columns.tolist())
    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "statcast_columns.txt").write_text("\n".join(cols))

    # Hunt for anything ABS-related. This determines your whole join strategy.
    kw = ("abs", "challeng", "review", "overturn", "robo", "hawkeye", "hawk_eye")
    hits = [c for c in cols if any(k in c.lower() for k in kw)]
    record(
        "statcast_has_abs_columns",
        bool(hits),
        ", ".join(hits) if hits else "none found",
        "ABS fields ship with Statcast -> the join in script 02 is a cross-check, not a "
        "dependency." if hits
        else "No ABS columns. Challenge outcomes MUST come from the Stats API (script 02) "
             "and be joined on game_pk + at_bat_number + pitch_number.",
    )

    # Columns the modeling weeks depend on.
    needed = [
        "game_pk", "game_date", "at_bat_number", "pitch_number", "inning",
        "inning_topbot", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
        "batter", "pitcher", "fielder_2", "stand", "p_throws", "pitch_type",
        "release_speed", "release_pos_x", "release_pos_z", "pfx_x", "pfx_z",
        "plate_x", "plate_z", "sz_top", "sz_bot", "description", "type", "zone",
        "home_score", "away_score",
    ]
    missing = [c for c in needed if c not in df.columns]
    record("statcast_required_columns", not missing,
           f"{len(needed) - len(missing)}/{len(needed)} present",
           f"MISSING: {missing}" if missing else "")

    # Win-probability columns come free with Statcast in recent seasons.
    wp = [c for c in cols if "win_exp" in c.lower() or "run_exp" in c.lower()]
    record("statcast_win_expectancy", bool(wp), ", ".join(wp) if wp else "none",
           "Week 10 can start from these instead of the Stats API winProbability endpoint."
           if wp else "Use script 06 (statsapi winProbability) for Week 10.")

    called = df[df["description"].isin(["called_strike", "ball", "blocked_ball"])] \
        if "description" in df.columns else df.iloc[0:0]
    record("called_pitch_share", len(called) > 0,
           f"{len(called):,} called pitches ({len(called)/max(len(df),1):.1%} of all pitches)",
           "Savant says ~50% of pitches are challenge opportunities; this is the pool.")

    return df


def check_challenges(date, statcast_df):
    """Can we get challenge events, and do they join to pitches?"""
    print("\n4. ABS CHALLENGE EVENTS (MLB Stats API)")
    try:
        import requests
    except ImportError:
        return

    s = requests.Session()
    s.headers.update({"User-Agent": UA})

    try:
        r = s.get(SOURCES["statsapi_schedule"],
                  params={"sportId": 1, "startDate": date, "endDate": date,
                          "gameType": "R", "hydrate": "officials"}, timeout=30)
        r.raise_for_status()
        sched = r.json()
    except Exception as e:
        record("schedule_fetch", False, type(e).__name__, str(e)[:160])
        return

    games = [g for d in sched.get("dates", []) for g in d.get("games", [])]
    final = [g for g in games if g.get("status", {}).get("abstractGameState") in
             ("Final", "Game Over")]
    record("schedule_games", bool(final), f"{len(final)} final games on {date}")
    if not final:
        return

    # Umpire hydration — needed for RQ4 and the umpire covariate.
    with_ump = 0
    for g in final:
        for o in g.get("officials", []):
            if o.get("officialType") == "Home Plate":
                with_ump += 1
                break
    record("umpire_hydrate", with_ump == len(final),
           f"{with_ump}/{len(final)} games have a home-plate umpire",
           "" if with_ump == len(final)
           else "hydrate=officials is not populating; check the schedule params.")

    # Pull challenges from a few games.
    sample = final[: min(5, len(final))]
    challenges = []
    for g in sample:
        pk = g["gamePk"]
        try:
            pbp = s.get(SOURCES["statsapi_playbyplay"].format(game_pk=pk), timeout=30).json()
        except Exception as e:
            record(f"pbp_{pk}", False, type(e).__name__, str(e)[:120])
            continue
        for play in pbp.get("allPlays", []):
            ab = play.get("about", {}).get("atBatIndex")
            for ev in play.get("playEvents", []):
                det = ev.get("details", {}) or {}
                desc = (det.get("description") or "")
                if det.get("hasReview") or "challenge" in desc.lower():
                    rd = ev.get("reviewDetails", {}) or {}
                    challenges.append({
                        "game_pk": pk,
                        "at_bat_index": ab,
                        "pitch_number": ev.get("pitchNumber"),
                        "challenger": (rd.get("player") or {}).get("fullName"),
                        "overturned": rd.get("isOverturned"),
                        "desc": desc[:80],
                    })

    record("challenge_events", bool(challenges),
           f"{len(challenges)} challenges in {len(sample)} games "
           f"({len(challenges)/max(len(sample),1):.1f}/game)",
           "MLB averages roughly 2-4 challenges per game." if challenges
           else "No challenges found. Check that playEvents carry hasReview/reviewDetails.")

    if challenges:
        have_challenger = sum(1 for c in challenges if c["challenger"])
        have_result = sum(1 for c in challenges if c["overturned"] is not None)
        record("challenge_has_challenger", have_challenger == len(challenges),
               f"{have_challenger}/{len(challenges)} have a challenger name",
               "" if have_challenger == len(challenges)
               else "Missing challenger identity -> you cannot split by batter/catcher/pitcher.")
        record("challenge_has_result", have_result == len(challenges),
               f"{have_result}/{len(challenges)} have isOverturned",
               "" if have_result == len(challenges)
               else "Missing outcome -> this is your MODEL TARGET. Must be fixed.")

    # ---- THE JOIN TEST. This is the make-or-break check for Week 3. ----
    if challenges and statcast_df is not None and not statcast_df.empty:
        sc = statcast_df
        pks = {c["game_pk"] for c in challenges}
        sc = sc[sc["game_pk"].isin(pks)]
        # Statcast at_bat_number is 1-based per game; Stats API atBatIndex is 0-based.
        keys = set(zip(sc["game_pk"], sc["at_bat_number"] - 1, sc["pitch_number"]))
        matched = sum(
            1 for c in challenges
            if (c["game_pk"], c["at_bat_index"], c["pitch_number"]) in keys
        )
        rate = matched / len(challenges)
        record("challenge_to_pitch_join", rate >= 0.95,
               f"{matched}/{len(challenges)} matched ({rate:.1%})",
               "Join key works: game_pk + (atBatIndex) + pitchNumber." if rate >= 0.95
               else "JOIN PROBLEM. Check the at_bat_number off-by-one and whether "
                    "pitchNumber counts pickoffs. Solve this before Week 5.")

    return challenges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-04-15",
                    help="a game day to probe (YYYY-MM-DD)")
    args = ap.parse_args()

    print("=" * 74)
    print("ABS CAPSTONE — WEEK 3 DATA FEASIBILITY CHECK")
    print(f"probe date: {args.date}   run at: {dt.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 74)

    check_environment()
    check_network()
    sc = check_statcast_columns(args.date)
    check_challenges(args.date, sc)

    n_ok = sum(1 for r in results if r["ok"])
    print("\n" + "=" * 74)
    print(f"SUMMARY: {n_ok}/{len(results)} checks passed")
    fails = [r for r in results if not r["ok"]]
    if fails:
        print("\nFailures to resolve before Week 4:")
        for r in fails:
            print(f"  - {r['check']}: {r['detail']}")
            if r["note"]:
                print(f"      {r['note']}")
    else:
        print("All checks passed. You are clear to build the pipeline (Week 4).")
    print("=" * 74)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"env_check_{stamp}.json"
    path.write_text(json.dumps(
        {"run_at": dt.datetime.now().isoformat(), "probe_date": args.date,
         "python": platform.python_version(), "results": results}, indent=2))
    print(f"\nReport saved: {path}")
    print("Paste the summary table into your Week 3 feasibility report.")


if __name__ == "__main__":
    main()
