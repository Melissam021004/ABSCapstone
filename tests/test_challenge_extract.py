#!/usr/bin/env python3
"""
test_challenge_extract.py — does script 02 find every ABS challenge in a game feed?

The Stats API records a review in one of three places: on the pitch, on a
separate event after the pitch, or on the plate appearance as a whole (where a
review of strike three or ball four can land). The first extractor looked only at
the pitch, and the 2026 season total came out about a quarter below MLB's official
10,557. These tests build small fake playByPlay feeds with each case — plus the
look-alikes that must NOT count (a manager's replay review of a stolen base, a
review of a ball in play, text-only flags, the same review reported twice) — and
check what scripts/02 keeps.

    python tests/test_challenge_extract.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("abs02", ROOT / "scripts" / "02_abs_challenges_statsapi.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BATTER, PITCHER, CATCHER, RUNNER = 100, 200, 300, 400
failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def pitch(n, code, balls, strikes, review=None, has_review=False):
    ev = {"isPitch": True, "pitchNumber": n, "type": "pitch",
          "count": {"balls": balls, "strikes": strikes, "outs": 0},
          "details": {"call": {"code": code, "description": code},
                      "description": {"B": "Ball", "C": "Called Strike", "X": "In play, out(s)"}[code],
                      "hasReview": has_review},
          "pitchData": {"coordinates": {"pX": 0.1, "pZ": 2.5},
                        "strikeZoneTop": 3.4, "strikeZoneBottom": 1.6}}
    if review:
        ev["reviewDetails"] = review
    return ev


def action(desc, balls, strikes, review=None, has_review=False):
    ev = {"isPitch": False, "type": "action",
          "count": {"balls": balls, "strikes": strikes, "outs": 0},
          "details": {"description": desc, "hasReview": has_review}}
    if review:
        ev["reviewDetails"] = review
    return ev


def rev(overturned, who, rtype="MJ", extra=None):
    r = {"isOverturned": overturned, "inProgress": False, "reviewType": rtype,
         "challengeTeamId": 1}
    if who is not None:
        r["player"] = {"id": who, "fullName": f"Player {who}"}
    if extra:
        r["additionalReviews"] = extra
    return r


def play(i, events, review=None, has_review=False, result="Result."):
    p = {"about": {"atBatIndex": i, "inning": 3, "halfInning": "top", "hasReview": has_review},
         "matchup": {"batter": {"id": BATTER, "fullName": "The Batter"},
                     "pitcher": {"id": PITCHER, "fullName": "The Pitcher"}},
         "result": {"description": result},
         "playEvents": events}
    if review:
        p["reviewDetails"] = review
    return p


plays = [
    # 0. on the pitch itself (what the first extractor found)
    play(0, [pitch(1, "B", 1, 0), pitch(2, "C", 2, 0, rev(True, BATTER), True), pitch(3, "X", 2, 0)]),
    # 1. on a separate event after the pitch -> the pitch before it
    play(1, [pitch(1, "C", 0, 1), pitch(2, "B", 1, 1),
             action("Catcher challenged the call: call stands", 1, 2, rev(False, CATCHER), True),
             pitch(3, "X", 1, 2)]),
    # 2. on the plate appearance: a failed challenge of strike three
    play(2, [pitch(1, "C", 0, 1), pitch(2, "C", 0, 2), pitch(3, "C", 0, 3)],
         review=rev(False, BATTER), has_review=True, result="The Batter called out on strikes."),
    # 3. a manager's replay review of a stolen base, after a called ball: NOT an ABS challenge
    play(3, [pitch(1, "B", 1, 0),
             action("Stolen base attempt; call overturned", 1, 0, rev(True, RUNNER, rtype="MA"), True),
             pitch(2, "X", 1, 0)]),
    # 4. the same challenge reported on the pitch AND on the play: count it once
    play(4, [pitch(1, "C", 0, 1), pitch(2, "C", 0, 2), pitch(3, "C", 0, 3, rev(False, BATTER), True)],
         review=rev(False, BATTER), has_review=True),
    # 5. a ball-four challenge on the play, with a second (manager) review nested inside it
    play(5, [pitch(1, "B", 1, 0), pitch(2, "B", 2, 0), pitch(3, "B", 3, 0), pitch(4, "B", 4, 0)],
         review=rev(False, CATCHER, extra=[rev(True, None, rtype="MA")]), has_review=True),
    # 6. a manager review of a ball in play, on the play: NOT an ABS challenge
    play(6, [pitch(1, "C", 0, 1), pitch(2, "X", 0, 1)], review=rev(True, None, rtype="MB"), has_review=True),
    # 7. a separate event flagged only by its text: left out (no outcome, no challenger)
    play(7, [pitch(1, "B", 1, 0), action("ABS challenge system check", 1, 0), pitch(2, "X", 1, 0)]),
    # 8. the pitch flagged hasReview with no details, the details on the play: keep the one with details
    play(8, [pitch(1, "C", 0, 1), pitch(2, "C", 0, 2), pitch(3, "C", 0, 3, has_review=True)],
         review=rev(True, BATTER), has_review=True),
    # 9. a review on a called-pitch event with NO challenger (a manager's review of a steal on
    #    that pitch): not an ABS challenge, and its type must not be learned as one —
    #    otherwise play 3's stolen-base review (same type) would slip in
    play(9, [pitch(1, "B", 1, 0, rev(True, None, rtype="MA"), True), pitch(2, "X", 1, 0)]),
]
pbp = {"allPlays": plays}
meta = {"game_date": "2026-06-01", "home_team": "HOM", "away_team": "AWY",
        "hp_umpire": "Ump", "hp_umpire_id": 9}

print("Extracting from a fake game feed")
rows = m.extract_challenges(777, pbp, meta)
found = pd.DataFrame(rows)
kept, left, types = m.keep_abs_reviews(found)
k = kept.set_index("at_bat_index")

check("ABS review types are learned from reviews on called pitches", types == {"MJ"}, str(types))
check("six ABS challenges kept (one per challenged pitch)", len(kept) == 6,
      f"{len(kept)}: {kept[['at_bat_index', 'pitch_number', 'review_source']].values.tolist()}")
check("plays 0, 1, 2, 4, 5, 8 kept", sorted(k.index) == [0, 1, 2, 4, 5, 8], str(sorted(k.index)))

r = k.loc[0]
check("on the pitch: found exactly as before", r["review_source"] == "pitch_event" and r["pitch_number"] == 2
      and r["challenger_role"] == "batter" and r["is_overturned"] is True
      and (r["balls_before"], r["strikes_before"]) == (1, 0) and r["call_code"] == "C")
r = k.loc[1]
check("on a separate event: tied to the pitch before it, with that pitch's count and call",
      r["review_source"] == "separate_event" and r["pitch_number"] == 2 and r["challenger_role"] == "catcher"
      and r["is_overturned"] is False and (r["balls_before"], r["strikes_before"]) == (0, 1)
      and r["call_code"] == "B" and r["challenging_side"] == "home", str(r.to_dict()))
r = k.loc[2]
check("on the plate appearance: tied to the last pitch (strike three)",
      r["review_source"] == "plate_appearance" and r["pitch_number"] == 3 and r["challenger_role"] == "batter"
      and r["is_overturned"] is False and (r["balls_before"], r["strikes_before"]) == (0, 2)
      and r["description"] == "The Batter called out on strikes." and r["challenging_side"] == "away")
check("reported twice: kept once, the record on the pitch", k.loc[4]["review_source"] == "pitch_event")
check("ball four on the play kept; the manager review nested inside it left out",
      k.loc[5]["review_source"] == "plate_appearance" and k.loc[5]["pitch_number"] == 4
      and k.loc[5]["challenger_role"] == "catcher"
      and ((left["at_bat_index"] == 5) & (left["review_type"] == "MA")).sum() == 1)
check("a text-only flag on the pitch loses to the full record on the play",
      k.loc[8]["review_source"] == "plate_appearance" and k.loc[8]["is_overturned"] is True
      and ((left["at_bat_index"] == 8) & (left["review_source"] == "pitch_event_flag")).sum() == 1)
check("a manager's stolen-base review is left out (wrong review type)",
      ((left["at_bat_index"] == 3) & (left["review_type"] == "MA")).sum() == 1 and 3 not in k.index)
check("a review of a ball in play is left out (not a called pitch)", 6 not in k.index
      and ((left["at_bat_index"] == 6) & (left["call_code"] == "X")).sum() == 1)
check("a text-only separate event is left out", 7 not in k.index
      and ((left["at_bat_index"] == 7) & (left["review_source"] == "separate_event_flag")).sum() == 1)
check("a review with no challenger is left out, even on a called pitch", 9 not in k.index
      and ((left["at_bat_index"] == 9) & (left["review_source"] == "pitch_event")).sum() == 1)
check("…and its review type isn't learned as an ABS type (so play 3 stays out)", "MA" not in types)
check("every kept row carries the pitch number script 08 joins on", kept["pitch_number"].notna().all())

print("\nRows from the old extractor")
old = found.copy()
old["review_source"] = None
o_kept, o_left, _ = m.keep_abs_reviews(old)
check("are kept exactly as they were (the cache from before the fix reproduces old results)",
      len(o_kept) == len(old) and len(o_left) == 0)
check("a table without the new column passes through untouched",
      len(m.keep_abs_reviews(found.drop(columns="review_source"))[0]) == len(found))

print("\nThe per-game cache")
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    cache, raw = d / "1.json", d / "1.json.gz"
    a = SimpleNamespace(force=False, redownload=False)
    cache.write_text(json.dumps([{"x": 1}]))
    rows_, old_ = m.load_cached(cache, raw, a)
    check("an old-format cache is used, and reported as old", rows_ == [{"x": 1}] and old_ is True)
    raw.write_bytes(b"x")
    check("an old-format cache with the raw feed saved is re-extracted", m.load_cached(cache, raw, a)[0] is None)
    cache.write_text(json.dumps({"extractor": m.EXTRACTOR_VERSION, "rows": [{"y": 2}]}))
    check("a new-format cache is used as is", m.load_cached(cache, raw, a) == ([{"y": 2}], False))
    check("--force re-extracts", m.load_cached(cache, raw, SimpleNamespace(force=True, redownload=False))[0] is None)

print("\nThe whole script on one fake game (no internet: the API calls are replaced)")
try:
    import pyarrow  # noqa: F401  (script 02 writes parquet, as in requirements.txt)
    read_out = pd.read_parquet
except ImportError:
    print("  (pyarrow isn't installed here, so this check writes the parquet files as CSV)")
    pd.DataFrame.to_parquet = lambda self, path, index=False, **k: self.to_csv(path, index=index)
    read_out = pd.read_csv
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    m.OUT_DIR, m.GAME_DIR, m.PBP_DIR = d, d / "games", d / "pbp"
    calls = []
    m.fetch_schedule = lambda *a, **k: pd.DataFrame([{**meta, "game_pk": 777, "game_type": "R"}])
    m.get_json = lambda s, url, params=None: calls.append(url) or json.loads(json.dumps(pbp))
    argv = sys.argv
    import contextlib
    import io
    try:
        for args_ in ([], [], ["--force"]):
            sys.argv = ["02"] + args_
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                m.main()
        out = read_out(d / "abs_challenges_2026.parquet")
    finally:
        sys.argv = argv
    check("downloads the game once, then works from the caches (even with --force)", len(calls) == 1, str(calls))
    check("saves the raw feed", (d / "pbp" / "777.json.gz").exists())
    check("writes the six ABS challenges, in game order", len(out) == 6
          and out["challenge_seq_in_game"].tolist() == [1, 2, 3, 4, 5, 6]
          and out["at_bat_index"].tolist() == [0, 1, 2, 4, 5, 8])
    check("lists the left-out reviews for inspection", (d / "review_records_left_out.csv").exists())
    check("compares the rate with MLB's official 4.35 a game", "official 2026 regular season" in buf.getvalue())
    check("checks the two-failures rule in innings 1–9", "more than 2 failed challenges in innings 1–9 "
          "(the rules allow 2): 0" in buf.getvalue(), buf.getvalue()[-900:])
    check("compares challenges per team-game with MLB's table",
          "Challenges per team-game, against MLB's distribution" in buf.getvalue())

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("All challenge-extraction checks pass.")
