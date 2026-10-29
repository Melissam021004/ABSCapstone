#!/usr/bin/env python3
"""
11_verify_challenges_remaining.py — the twenty minutes that saves November.

`challenges_remaining` is derived, not published. It is the state variable r in
your MDP, every policy result depends on it, and it will run without erroring
whether it is right or wrong. Your plan's Week 5 warning is specifically about
this field.

This script makes checking it cheap. For a few games it prints a chronological
challenge-by-challenge ledger showing what the inventory should be before and
after each challenge, and what the built table actually says — then gives you
the MLB.com and Savant links to confirm against the real play-by-play.

It also recomputes the inventory a SECOND time, independently, straight from the
challenge event list. Two implementations agreeing is weak evidence; two
implementations disagreeing is proof something is broken. The script tells you
which you have, and then you still open one game in a browser, because only that
catches an error both implementations share.

    python scripts/11_verify_challenges_remaining.py
    python scripts/11_verify_challenges_remaining.py --games 5
    python scripts/11_verify_challenges_remaining.py --game-pk 800017

Output: console ledger + docs/VERIFICATION_CHALLENGES_REMAINING.md
"""

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
PROCESSED = DATA_DIR / "processed"
RAW = DATA_DIR / "raw"
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))

START_CHALLENGES = 2


def read_any(path):
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    csv = p.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def independent_walk(game_challenges, max_inning):
    """
    A second, deliberately simple implementation of the inventory rules, built
    only from the challenge list. Written differently from script 08 on purpose:
    if both agree, they at least don't share a transcription slip.

    Rules: start at 2 per side; a failed challenge costs one; a successful one
    costs nothing; entering an extra inning, a side sitting at zero gets one.
    """
    rem = {"home": START_CHALLENGES, "away": START_CHALLENGES}
    granted = {"home": 0, "away": 0}
    seen_inning = None
    out = []

    # Real Stats API data has gaps the synthetic fixture never produces. Some
    # review events carry no pitchNumber (the review is logged as its own event
    # rather than on the pitch), and a few carry no challenger, which leaves the
    # side unknown. Neither is an error — but neither can be forced into an int.
    def safe_int(v):
        return None if v is None or pd.isna(v) else int(v)

    for r in game_challenges.sort_values(
            ["inning", "at_bat_index", "pitch_number"]).itertuples():
        inn = safe_int(r.inning)
        if inn is None:
            continue                      # cannot place it in the game at all
        if seen_inning is None:
            seen_inning = inn
        while seen_inning < inn:
            seen_inning += 1
            if seen_inning > 9:
                for side in ("home", "away"):
                    if rem[side] == 0:
                        rem[side] = 1
                        granted[side] += 1

        side = r.challenging_side if r.challenging_side in rem else None
        overturned = bool(r.is_overturned) if pd.notna(r.is_overturned) else None

        if side is None:
            # Without a side we cannot decrement anything. Record it so it shows
            # up in the ledger rather than vanishing, but leave the walk alone.
            before = after = None
        else:
            before = rem[side]
            if overturned is False:
                rem[side] = max(rem[side] - 1, 0)
            after = rem[side]

        b = safe_int(getattr(r, "balls_before", None))
        s = safe_int(getattr(r, "strikes_before", None))

        out.append({
            "seq": len(out) + 1,
            "inning": inn,
            "half": r.half_inning,
            "count": f"{b}-{s}" if b is not None and s is not None else "?",
            "challenger": r.challenger_name if pd.notna(r.challenger_name) else None,
            "role": r.challenger_role if pd.notna(r.challenger_role) else "unknown",
            "side": side or "unknown",
            "result": "OVERTURNED" if overturned else
                      ("stands" if overturned is False else "unknown"),
            "expected_before": before,
            "expected_after": after,
            "at_bat_index": safe_int(r.at_bat_index),
            "pitch_number": safe_int(r.pitch_number),
        })

    # extra-inning grants that happen after the last challenge
    while seen_inning is not None and seen_inning < max_inning:
        seen_inning += 1
        if seen_inning > 9:
            for side in ("home", "away"):
                if rem[side] == 0:
                    rem[side] = 1
                    granted[side] += 1

    return pd.DataFrame(out), rem, granted


def table_inventory(pitches, side, at_bat_index, pitch_number, when):
    """
    What the BUILT table says the inventory is, for one side, immediately
    before or after a given pitch.
    """
    # No key, no lookup. A challenge missing its pitch number or at-bat index
    # cannot be matched to a pitch — by anyone, including script 08's join.
    if (at_bat_index is None or pitch_number is None
            or (isinstance(at_bat_index, float) and pd.isna(at_bat_index))
            or (isinstance(pitch_number, float) and pd.isna(pitch_number))
            or side not in ("home", "away")):
        return np.nan
    g = pitches.sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)
    ab = at_bat_index + 1                       # Stats API is 0-based
    hit = g.index[(g["at_bat_number"] == ab) & (g["pitch_number"] == pitch_number)]
    if len(hit) == 0:
        return np.nan
    i = int(hit[0])
    if when == "after":
        nxt = g.iloc[i + 1:]
        if nxt.empty:
            return np.nan
        row = nxt.iloc[0]
    else:
        row = g.iloc[i]
    col = ("challenges_remaining_batting"
           if row["batting_side"] == side else "challenges_remaining_fielding")
    return row[col]


def verify_game(pk, inv, challenges, games, md):
    pitches = inv[inv["game_pk"] == pk]
    ch = challenges[challenges["game_pk"] == pk]
    meta = games[games["game_pk"] == pk].iloc[0] if games is not None and \
        (games["game_pk"] == pk).any() else None

    home = meta["home_team"] if meta is not None else pitches["batting_side"].iloc[0]
    away = meta["away_team"] if meta is not None else "?"
    date = str(meta["game_date"])[:10] if meta is not None else \
        str(pitches["game_date"].iloc[0])[:10]
    ump = meta["hp_umpire"] if meta is not None else "?"
    max_inning = int(pitches["inning"].max())

    head = f"Game {pk} — {away} @ {home}, {date}"
    print("\n" + "=" * 78)
    print(head)
    print(f"  HP umpire: {ump}   innings: {max_inning}   "
          f"pitches: {len(pitches):,}   challenges: {len(ch)}")
    print("  Verify against:")
    print(f"    Savant game feed  https://baseballsavant.mlb.com/gamefeed?gamePk={pk}")
    print(f"    MLB Gameday       https://www.mlb.com/gameday/{pk}")
    print(f"    Raw play-by-play  https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay")
    print("=" * 78)

    md.append(f"## {head}\n")
    md.append(f"- HP umpire: **{ump}** · {max_inning} innings · "
              f"{len(pitches):,} pitches · **{len(ch)} challenges**")
    md.append(f"- [Savant game feed](https://baseballsavant.mlb.com/gamefeed?gamePk={pk}) · "
              f"[MLB Gameday](https://www.mlb.com/gameday/{pk}) · "
              f"[raw play-by-play](https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay)\n")

    if ch.empty:
        print("  No challenges in this game — nothing to verify here.")
        md.append("No challenges in this game.\n")
        return True

    led, final_rem, granted = independent_walk(ch, max_inning)

    # compare the independent walk against the built table
    rows, mismatches, unjoinable = [], 0, 0
    for r in led.itertuples():
        got_before = table_inventory(pitches, r.side, r.at_bat_index,
                                     r.pitch_number, "before")
        got_after = table_inventory(pitches, r.side, r.at_bat_index,
                                    r.pitch_number, "after")
        # pandas turns a column of ints-and-None into float64 with NaN, so an
        # `is None` test here silently never fires. Check for both.
        gone = lambda v: v is None or (isinstance(v, float) and pd.isna(v))
        no_key = (gone(r.at_bat_index) or gone(r.pitch_number)
                  or r.side == "unknown" or gone(r.expected_before))
        if no_key:
            # Not a mismatch — there is nothing to compare against. Counting it
            # as a failure would raise a false alarm on every game that has one.
            unjoinable += 1
            verdict = "no key"
        else:
            ok_b = pd.isna(got_before) or int(got_before) == int(r.expected_before)
            ok_a = pd.isna(got_after) or int(got_after) == int(r.expected_after)
            if not (ok_b and ok_a):
                mismatches += 1
            verdict = "ok" if (ok_b and ok_a) else "MISMATCH"
        def s(v, dflt="?"):
            """DataFrame string fields arrive as NaN when missing, and NaN is
            truthy — so `v or default` keeps the NaN and the next slice fails."""
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return dflt
            return str(v)

        rows.append({
            "#": r.seq,
            "inn": f"{r.inning}{s(r.half, '?')[:1].upper()}",
            "count": s(r.count, "?"),
            "challenger": s(r.challenger, "(unknown)")[:20],
            "role": s(r.role, "unknown"),
            "side": s(r.side, "unknown"), "result": s(r.result, "unknown"),
            "before": "-" if gone(r.expected_before) else int(r.expected_before),
            "after": "-" if gone(r.expected_after) else int(r.expected_after),
            "table_before": "-" if pd.isna(got_before) else int(got_before),
            "table_after": "-" if pd.isna(got_after) else int(got_after),
            "match": verdict,
        })

    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    md.append("| # | Inn | Count | Challenger | Role | Side | Result | "
              "Expected before→after | Table before→after | |")
    md.append("|---:|---|---|---|---|---|---|---|---|---|")
    MARK = {"ok": "✅", "MISMATCH": "❌", "no key": "—"}
    for r in out.itertuples():
        md.append(f"| {r._1} | {r.inn} | {r.count} | {r.challenger} | {r.role} | "
                  f"{r.side} | {r.result} | {r.before} → {r.after} | "
                  f"{r.table_before} → {r.table_after} | "
                  f"{MARK.get(r.match, '❌')} |")
    md.append("")

    if unjoinable:
        msg = (f"{unjoinable} of {len(led)} challenges here carry no pitch number, "
               f"at-bat index or challenger, so they cannot be matched to a pitch "
               f"by any method. They are shown as 'no key' rather than as failures.")
        print(f"\n  NOTE: {msg}")
        md.append(f"> ℹ️ {msg}\n")

    print(f"\n  Final inventory: home {final_rem['home']}, away {final_rem['away']}")
    if granted["home"] or granted["away"]:
        print(f"  Extra-inning grants: home +{granted['home']}, away +{granted['away']}")
        md.append(f"- Extra-inning grants: home +{granted['home']}, "
                  f"away +{granted['away']}")
    md.append(f"- Final inventory: home **{final_rem['home']}**, "
              f"away **{final_rem['away']}**\n")

    if mismatches:
        print(f"\n  *** {mismatches} MISMATCH(ES) between the independent walk and "
              f"the built table. Script 08's derivation is wrong. ***")
        md.append(f"> ❌ **{mismatches} mismatches** between the independent walk and "
                  f"the built table. Fix `derive_challenge_state()` in script 08 "
                  f"before going further.\n")
    else:
        print("\n  Independent walk agrees with the built table on every challenge.")
        md.append("> ✅ The independent walk agrees with the built table on every "
                  "challenge in this game.\n")

    print("""
  NOW CHECK BY HAND — open the Savant game feed above and confirm:
    1. the number of challenges matches the count above
    2. each challenger's name and role matches
    3. each result (overturned / stands) matches
    4. a team that lost two challenges stops appearing after that
  Two implementations can share the same wrong assumption. Your eyes cannot.""")

    md.append("**Hand-check (open the Savant game feed):**\n")
    md.append(f"- [ ] Savant shows {len(ch)} challenges in this game")
    md.append("- [ ] Every challenger name and role matches")
    md.append("- [ ] Every result (overturned / stands) matches")
    md.append("- [ ] No side challenges again after losing two")
    md.append("- [ ] Extra-inning grants, if any, look right\n")
    return mismatches == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3,
                    help="how many games to verify (default 3)")
    ap.add_argument("--game-pk", type=int, action="append",
                    help="verify specific game_pk(s); repeatable")
    ap.add_argument("--pick", choices=["most-challenges", "random", "extra-innings"],
                    default="most-challenges",
                    help="how to choose games when --game-pk isn't given")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inv = read_any(PROCESSED / "challenge_inventory_2026.parquet")
    if inv is None:
        sys.exit("No built table. Run 08_build_challenge_inventory.py first.")
    challenges = read_any(RAW / "challenges" / "abs_challenges_2026.parquet")
    if challenges is None:
        sys.exit("No challenge events. Run 02_abs_challenges_statsapi.py first.")
    games = read_any(RAW / "challenges" / "games_2026.parquet")

    if "was_challenged" in inv.columns:
        inv["was_challenged"] = inv["was_challenged"].fillna(False).astype(bool)

    # choose games
    if args.game_pk:
        pks = list(args.game_pk)
    elif args.pick == "random":
        rng = np.random.default_rng(args.seed)
        pool = challenges["game_pk"].unique()
        pks = list(rng.choice(pool, size=min(args.games, len(pool)), replace=False))
    elif args.pick == "extra-innings":
        # the hardest case: extra-inning replenishment
        maxinn = inv.groupby("game_pk")["inning"].max()
        pool = maxinn[maxinn > 9].index
        pool = [p for p in pool if p in set(challenges["game_pk"])]
        pks = pool[: args.games]
        if not pks:
            print("No extra-inning games with challenges; falling back to "
                  "most-challenges.\n")
            pks = challenges["game_pk"].value_counts().head(args.games).index.tolist()
    else:
        pks = challenges["game_pk"].value_counts().head(args.games).index.tolist()

    md = [f"# Verification — `challenges_remaining`\n",
          f"Generated {dt.date.today().isoformat()} by "
          f"`scripts/11_verify_challenges_remaining.py`.\n",
          "`challenges_remaining` is derived rather than published, and it is the "
          "state variable r in the MDP. This document records the hand-verification "
          "the Week 5 plan requires.\n",
          "Each game below is checked two ways: an independent re-walk of the "
          "inventory rules straight from the challenge events, compared against the "
          "built table — and then a manual check against the real play-by-play, "
          "which is the only one that catches an assumption both implementations "
          "share.\n"]

    print("=" * 78)
    print(f"VERIFYING challenges_remaining ON {len(pks)} GAME(S)")
    print(f"selection: {'explicit --game-pk' if args.game_pk else args.pick}")
    print("=" * 78)

    all_ok = True
    for pk in pks:
        all_ok &= verify_game(int(pk), inv, challenges, games, md)

    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "VERIFICATION_CHALLENGES_REMAINING.md"

    md.append("---\n")
    md.append("## Sign-off\n")
    md.append("- [ ] I opened at least three of these games in the Savant game feed "
              "and confirmed every challenge, challenger and result by eye.")
    md.append("- [ ] Any mismatch above was investigated and resolved.")
    md.append(f"\nChecked by: ____________________   Date: ____________\n")
    out.write_text("\n".join(md))

    print("\n" + "=" * 78)
    if all_ok:
        print("Independent walk matches the built table on every game checked.")
    else:
        print("MISMATCHES FOUND — fix derive_challenge_state() in script 08.")
    print("This is NOT the whole check. Open the Savant links above and verify by")
    print("eye: two implementations can share the same wrong assumption.")
    print(f"\nWorksheet -> {out}")
    print("=" * 78)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
