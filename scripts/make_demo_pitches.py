#!/usr/bin/env python3
"""
make_demo_pitches.py — synthetic pitch-level data for testing the Week 5 chain.

Why this exists: scripts 08-11 cannot run until your full-season pulls finish,
and you should not be debugging a join for the first time against 700,000 real
rows. This generates a small, INTERNALLY CONSISTENT season so you can run the
whole Week 5 pipeline end to end in about a minute, see what every output looks
like, and confirm the QA assertions actually pass before the real data lands.

"Internally consistent" is the important word. The challenge events obey the
real 2026 rules — two per team, successful ones retained, extra-inning
replenishment, only batter/catcher/pitcher challenging, never with zero left —
and a challenge is overturned exactly when the umpire's call disagreed with the
ABS zone. So if script 08's derivation is correct, the QA assertions pass; if
you break the derivation, they fail. That makes this a test fixture, not just
a demo.

The games are real baseball, simplified (added for Week 10, which needs whole
games): runners on base, walks, strikeouts, hits, double plays, sacrifice
flies, steals and wild pitches; no bottom of the 9th when the home team leads;
walk-offs; extra innings with the automatic runner on second; teams of
different strength and a home-field edge. Statcast's `events`, `bat_score`,
`fld_score` and `post_*_score` columns are filled in, so win probability and
run expectancy can be estimated from it. A few differences are PLANTED —
umpires who miss more calls, players who challenge better — and written to
raw/_demo_truth.json so the Week 10 tests can check they're found.

    python scripts/make_demo_pitches.py              # -> data_demo/
    python scripts/make_demo_pitches.py --games 120
    python scripts/make_demo_pitches.py --recorded-call final   # post-review call
                                                     # recorded, as in real 2026 data

Everything is written under data_demo/ by default so it can never overwrite a
real pull. Figures and reports built from it are stamped as demo output.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261001)

PLATE_HALF_WIDTH_FT = 8.5 / 12
BALL_RADIUS_FT = (2.9 / 2) / 12
ZONE_TOP_PCT = 0.535
ZONE_BOT_PCT = 0.27
START_CHALLENGES = 2

TEAMS = ["NYY", "BOS", "TOR", "BAL", "TBR", "CLE", "MIN", "DET", "KCR", "CHW",
         "HOU", "SEA", "TEX", "LAA", "ATH", "ATL", "PHI", "NYM", "WSN", "MIA",
         "MIL", "CHC", "STL", "CIN", "PIT", "LAD", "SDP", "SFG", "ARI", "COL"]
UMPS = ["Pat Hoberg", "Will Little", "Mark Wegner", "Tripp Gibson", "Lance Barksdale",
        "Doug Eddings", "Dan Iassogna", "Jordan Baker", "Adam Hamari", "Nic Lentz",
        "Edwin Moscoso", "Chris Guccione", "Todd Tichenor", "John Tumpane", "Ben May",
        "Quinn Wolcott", "Sean Barber", "Erich Bacchus", "Alan Porter", "Ryan Wills"]
FIRST = ["Cal", "Adley", "Will", "Sean", "Logan", "Patrick", "Yainer", "Gabriel",
         "Austin", "Jonah", "Tyler", "Carson", "Bo", "Corbin", "Elly", "Gunnar",
         "Riley", "Marcus", "Jose", "Julio", "Kyle", "Matt", "Luis", "Rafael"]
LAST = ["Raleigh", "Rutschman", "Smith", "Murphy", "Kirk", "Bailey", "Diaz",
        "Moreno", "Wells", "Heim", "Stephenson", "Kelly", "Alvarez", "Bichette",
        "Carroll", "Henderson", "Greene", "Semien", "Ramirez", "Rodriguez",
        "Tucker", "Olson", "Arraez", "Devers"]
PITCH_TYPES = ["FF", "SL", "CH", "CU", "SI", "FC", "ST", "KC"]
PITCH_W = [0.34, 0.20, 0.12, 0.09, 0.14, 0.06, 0.04, 0.01]


def zone_distance_ft(plate_x, plate_z, top, bot):
    """Signed distance from the ball's surface to the ABS rectangle (feet).
    Negative => any part of the ball overlaps the zone => ABS strike."""
    plate_x = np.asarray(plate_x, float)
    plate_z = np.asarray(plate_z, float)
    top = np.asarray(top, float)
    bot = np.asarray(bot, float)
    dx = np.abs(plate_x) - PLATE_HALF_WIDTH_FT
    dz = np.maximum(bot - plate_z, plate_z - top)
    inside = (dx <= 0) & (dz <= 0)
    corner = (dx > 0) & (dz > 0)
    d = np.where(inside, np.maximum(dx, dz),
                 np.where(corner, np.hypot(np.maximum(dx, 0), np.maximum(dz, 0)),
                          np.maximum(dx, dz)))
    return d - BALL_RADIUS_FT


def write_table(df, path):
    """Parquet if pyarrow is installed, otherwise CSV. Readers handle both."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def build_people(n_players=420):
    ids = RNG.choice(np.arange(400000, 700000), size=n_players, replace=False)
    heights = np.round(RNG.normal(6.05, 0.19, n_players), 3)
    heights = np.clip(heights, 5.5, 6.75)
    pos = RNG.choice(["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"],
                     n_players, p=[.42, .08, .05, .06, .06, .07, .06, .06, .06, .08])
    return pd.DataFrame({
        "player_id": ids,
        "full_name": [f"{RNG.choice(FIRST)} {RNG.choice(LAST)}" for _ in range(n_players)],
        "height_raw": [f"{int(h)}' {int(round((h % 1) * 12))}\"" for h in heights],
        "height_ft": heights,
        "weight_lb": RNG.integers(170, 260, n_players),
        "bat_side": RNG.choice(["R", "L", "S"], n_players, p=[.58, .34, .08]),
        "pitch_hand": RNG.choice(["R", "L"], n_players, p=[.72, .28]),
        "primary_position": pos,
        "birth_date": "1997-01-01",
        "mlb_debut_date": "2020-04-01",
    })


# ----------------------------------------------------------------- the game
# Run environment, tuned so a simulated season looks like MLB 2023–25:
# about 4.5 runs per team per game, K% ≈ 22%, BB% ≈ 8.5%, HR ≈ 3% of plate
# appearances, ≈ 3.9 pitches per plate appearance, home teams winning ≈ 53%.
# main() prints the simulated values next to these targets.

# Chance the pitcher ATTACKS the zone, by count (balls, strikes). Behind in the
# count he has to throw strikes; ahead, he expands.
ATTACK = {(0, 0): .55, (1, 0): .56, (2, 0): .62, (3, 0): .74,
          (0, 1): .52, (1, 1): .54, (2, 1): .59, (3, 1): .67,
          (0, 2): .40, (1, 2): .44, (2, 2): .50, (3, 2): .60}
# Chance the batter swings, by count, at a pitch in / out of the ABS zone.
SWING_IN = {(0, 0): .52, (1, 0): .60, (2, 0): .62, (3, 0): .12,
            (0, 1): .70, (1, 1): .72, (2, 1): .74, (3, 1): .66,
            (0, 2): .84, (1, 2): .85, (2, 2): .86, (3, 2): .88}
SWING_OUT = {(0, 0): .18, (1, 0): .21, (2, 0): .20, (3, 0): .03,
             (0, 1): .30, (1, 1): .31, (2, 1): .30, (3, 1): .22,
             (0, 2): .42, (1, 2): .42, (2, 2): .43, (3, 2): .45}
CONTACT = {True: 0.875, False: 0.62}          # by "pitch in the zone?"
FOUL_SHARE = {True: 0.53, False: 0.60}       # of contact
HBP_RATE = 0.010                             # of taken pitches outside the zone
# Ball in play: chance of each hit type (the rest are outs), before the
# team-strength and contact-quality multipliers.
BIP = {"single": 0.206, "double": 0.068, "triple": 0.006, "home_run": 0.049,
       "field_error": 0.014}
OUT_CHASE_PENALTY = 0.72                     # contact outside the zone is weaker
HOME_BAT = 1.03                              # home batters hit a little better
STEAL_TRY_1B, STEAL_OK_1B = 0.030, 0.79      # per pitch, runner on 1st, 2nd open
STEAL_TRY_2B, STEAL_OK_2B = 0.004, 0.75      # per pitch, runner on 2nd, 3rd open
WILD_PITCH = 0.010                           # per taken ball with runners on
STARTER_INNINGS = 6
MAX_INNINGS = 25


def _sample_location(attack, mid, top, bot):
    """
    Where the pitch actually crosses the plate (ft) and its true distance to the
    ABS zone. An attacking pitch is aimed into the zone (and sometimes misses);
    otherwise the pitcher works the edges and beyond — drawn until it is outside.
    """
    if attack:
        tx, tz = float(RNG.normal(0, 0.46)), float(RNG.normal(mid, 0.42))
        return tx, tz, float(zone_distance_ft(tx, tz, top, bot))
    for _ in range(12):
        tx, tz = float(RNG.normal(0, 1.0)), float(RNG.normal(mid - 0.05, 0.95))
        d = float(zone_distance_ft(tx, tz, top, bot))
        if d > 0:
            break
    return tx, tz, d


def _force(bases, batter):
    """Walk / hit-by-pitch: every runner who is forced moves up one base."""
    r1, r2, r3 = bases
    runs = 0
    if r1 is not None:
        if r2 is not None:
            if r3 is not None:
                runs += 1
            r3 = r2
        r2 = r1
    r1 = batter
    return [r1, r2, r3], runs


def _advance_all(bases):
    """Wild pitch / passed ball / error: every runner moves up one base."""
    r1, r2, r3 = bases
    return [None, r1, r2], int(r3 is not None)


def _ball_in_play(bases, outs, batter, quality):
    """
    One ball in play -> (events, new bases, outs added, runs). Probabilities are
    MLB-like; `quality` scales the hit types (team strength, home field, contact
    quality). Runs that cross on a play that makes the third out don't count
    (true for force outs and batter outs, which is all this simulates).
    """
    probs = {k: v * quality for k, v in BIP.items()}
    r = RNG.random()
    acc, kind = 0.0, "out"
    for k, v in probs.items():
        acc += v
        if r < acc:
            kind = k
            break
    r1, r2, r3 = bases
    runs = 0
    if kind == "home_run":
        return "home_run", [None, None, None], 0, 1 + sum(b is not None for b in bases)
    if kind == "triple":
        return "triple", [None, None, batter], 0, sum(b is not None for b in bases)
    if kind == "double":
        runs = int(r3 is not None) + int(r2 is not None)
        new3 = None
        if r1 is not None:
            if RNG.random() < (0.55 if outs == 2 else 0.40):
                runs += 1
            else:
                new3 = r1
        return "double", [None, batter, new3], 0, runs
    if kind == "single":
        runs = int(r3 is not None)
        new2, new3 = None, None
        if r2 is not None:
            if RNG.random() < (0.78 if outs == 2 else 0.64):
                runs += 1
            else:
                new3 = r2
        if r1 is not None:
            if new3 is None and RNG.random() < 0.28:
                new3 = r1
            else:
                new2 = r1
        return "single", [batter, new2, new3], 0, runs
    if kind == "field_error":
        nb, runs = _advance_all(bases)
        nb[0] = batter
        return "field_error", nb, 0, runs

    # ---- an out ----
    kind = RNG.choice(["ground", "fly", "line", "pop"], p=[0.45, 0.35, 0.12, 0.08])
    if kind == "ground":
        if r1 is not None and outs <= 1:
            u = RNG.random()
            if u < 0.42:                                   # 6-4-3: batter and runner from 1st out
                after = outs + 2
                new = [None, r2, r3]
                if after < 3:
                    if r3 is not None and RNG.random() < 0.6:
                        runs, new[2] = 1, None
                    if new[1] is not None and new[2] is None and RNG.random() < 0.5:
                        new[2], new[1] = new[1], None
                return "grounded_into_double_play", new, 2, runs
            if u < 0.62:                                   # runner from 1st forced at 2nd
                after = outs + 1
                new = [batter, None, None]
                if r2 is not None:
                    new[2] = r2                            # forced to third
                if r3 is not None:
                    if r2 is not None or RNG.random() < 0.5:
                        runs = 1                           # forced home, or scores on the play
                    else:
                        new[2] = r3
                return "force_out", new, 1, (runs if after < 3 else 0)
        # batter thrown out at first; runners move up (forced ones always)
        after = outs + 1
        new2, new3 = r2, r3
        if r3 is not None and (r1 is not None and r2 is not None or RNG.random() < 0.68):
            runs, new3 = 1, None
        if r2 is not None and new3 is None and (r1 is not None or RNG.random() < 0.5):
            new3, new2 = r2, None
        if r1 is not None:
            new2 = r1
        return "field_out", [None, new2, new3], 1, (runs if after < 3 else 0)
    if kind == "fly":
        after = outs + 1
        new = [r1, r2, r3]
        if after < 3:
            if r3 is not None and RNG.random() < 0.78:
                runs, new[2] = 1, None
            if new[1] is not None and new[2] is None and RNG.random() < 0.25:
                new[2], new[1] = new[1], None
        return ("sac_fly" if runs else "field_out"), new, 1, runs
    return "field_out", [r1, r2, r3], 1, 0                  # line drive / pop-up


def simulate(n_games, people, start_date="2026-03-26", recorded_call="original"):
    """
    One pass that produces pitches and challenges together, so they agree.

    Real baseball, simplified: runners on base, outs, walks, strikeouts, hits,
    double plays, sacrifice flies, steals, wild pitches; nine innings, no bottom
    of the 9th when the home team leads, walk-offs, and extra innings with the
    automatic runner on second. Teams differ in strength, umpires in how often
    they miss, and challengers in how well they tell a wrong call from a right
    one — so the analyses that look for those differences have something real
    to find.

    recorded_call: "original" stores the umpire's call in `description` on
    challenged pitches; "final" stores the post-review call, as Statcast did in
    2026 (the call-recording check repairs either).
    """
    heights = dict(zip(people["player_id"], people["height_ft"]))
    # ABS rules on the height MLB measured for the system, not the listed height
    # the Stats API reports. They differ by about half an inch — which is why,
    # on real data, calls near the top and bottom of the zone are less certain
    # than calls near the sides (the sides are a fixed 17 inches).
    abs_offset = {pid: float(RNG.normal(0, 0.05)) for pid in people["player_id"]}
    SEASON_DAYS = 184
    catchers = people.loc[people["primary_position"] == "C", "player_id"].tolist()
    pitchers = people.loc[people["primary_position"] == "P", "player_id"].tolist()
    hitters = people.loc[people["primary_position"] != "P", "player_id"].tolist()
    names = dict(zip(people["player_id"], people["full_name"]))
    bats = dict(zip(people["player_id"], people["bat_side"]))
    throws = dict(zip(people["player_id"], people["pitch_hand"]))

    # Planted differences (the Week 10 analyses should find these):
    #   team strength — scales hits for (offense) and against (run prevention)
    #   umpire error  — some umpires miss more calls than others
    #   judgment      — some players challenge wrong calls more and right calls less
    off = {t: float(np.clip(RNG.normal(1.0, 0.07), 0.82, 1.18)) for t in TEAMS}
    prv = {t: float(np.clip(RNG.normal(1.0, 0.07), 0.82, 1.18)) for t in TEAMS}
    ue = np.exp(RNG.normal(0, 0.28, len(UMPS)))
    ump_err = dict(zip(UMPS, ue / ue.mean()))
    judgment = {pid: float(np.exp(RNG.normal(0, 0.35))) for pid in people["player_id"]}

    pitch_rows, chal_rows, game_rows = [], [], []
    d0 = dt.date.fromisoformat(start_date)
    # a schedule of SERIES: 2-4 games in a row between the same two teams, same park
    pairs = []
    while len(pairs) < n_games:
        h_, a_ = (str(x) for x in RNG.choice(TEAMS, 2, replace=False))
        pairs += [(h_, a_)] * int(RNG.choice([2, 3, 3, 3, 4]))

    for g in range(n_games):
        pk = 800000 + g
        home, away = pairs[g]
        # games of one series fall on consecutive days
        gdate = d0 + dt.timedelta(days=int(g * SEASON_DAYS / max(n_games, 1)))
        ump = str(RNG.choice(UMPS))
        ump_id = 400000 + UMPS.index(ump)
        err_mult = ump_err[ump]

        catcher_of = {"home": int(RNG.choice(catchers)), "away": int(RNG.choice(catchers))}
        pitcher_of = {"home": int(RNG.choice(pitchers)), "away": int(RNG.choice(pitchers))}
        lineup = {"home": [int(x) for x in RNG.choice(hitters, 9, replace=False)],
                  "away": [int(x) for x in RNG.choice(hitters, 9, replace=False)]}
        team = {"home": home, "away": away}
        allowed = {"home": 0, "away": 0}          # runs off the current pitcher
        rem = {"home": START_CHALLENGES, "away": START_CHALLENGES}
        ab = 0                      # 1-based at_bat_number after increment
        score = {"home": 0, "away": 0}
        chal_seq = 0
        spot = {"home": 0, "away": 0}
        game_over = False
        inning = 0

        while not game_over and inning < MAX_INNINGS:
            inning += 1
            if inning > 9:          # extra-inning replenishment (2026 rule)
                for side in ("home", "away"):
                    if rem[side] == 0:
                        rem[side] = 1
            for half in ("Top", "Bot"):
                if half == "Bot" and inning >= 9 and score["home"] > score["away"]:
                    game_over = True              # home team doesn't need its last at-bats
                    break
                bat_side = "away" if half == "Top" else "home"
                fld_side = "home" if half == "Top" else "away"
                order = lineup[bat_side]
                # a reliever from the 7th, or once the starter has been hit hard
                if inning == STARTER_INNINGS + 1 or (inning > STARTER_INNINGS and
                                                     RNG.random() < 0.6) \
                        or allowed[fld_side] >= 5:
                    pitcher_of[fld_side] = int(RNG.choice(pitchers))
                    allowed[fld_side] = 0
                pitcher = pitcher_of[fld_side]
                catcher = catcher_of[fld_side]
                p_hand = throws.get(pitcher, "R")
                quality_base = (off[team[bat_side]] * prv[team[fld_side]]
                                * (HOME_BAT if bat_side == "home" else 1.0))
                outs = 0
                bases = [None, None, None]
                if inning > 9:                  # automatic runner on second
                    bases[1] = order[(spot[bat_side] - 1) % 9]

                while outs < 3 and not game_over:
                    ab += 1
                    batter = order[spot[bat_side] % 9]
                    stand = bats.get(batter, "R")
                    if stand == "S":
                        stand = "L" if p_hand == "R" else "R"
                    b_h = heights.get(batter, 6.0)
                    top, bot = b_h * ZONE_TOP_PCT, b_h * ZONE_BOT_PCT
                    abs_h = b_h + abs_offset.get(batter, 0.0)
                    atop, abot = abs_h * ZONE_TOP_PCT, abs_h * ZONE_BOT_PCT
                    balls = strikes = 0
                    pitch_no = 0
                    pa_over = False
                    advance_spot = True

                    while not pa_over:
                        pitch_no += 1
                        # True location (what Hawk-Eye/ABS rules on)
                        attack = RNG.random() < ATTACK[(balls, strikes)]
                        tx, tz, true_dist = _sample_location(attack, (top + bot) / 2,
                                                             atop, abot)
                        abs_strike = true_dist < 0
                        # Statcast's recorded location carries a little
                        # measurement noise relative to the ABS ruling.
                        px = tx + float(RNG.normal(0, 0.02))
                        pz = tz + float(RNG.normal(0, 0.02))

                        bat_score, fld_score = score[bat_side], score[fld_side]
                        row = {
                            "game_pk": pk, "game_date": gdate.isoformat(), "game_type": "R",
                            "at_bat_number": ab, "pitch_number": pitch_no,
                            "inning": inning, "inning_topbot": half,
                            "balls": balls, "strikes": strikes, "outs_when_up": outs,
                            "on_1b": float(bases[0]) if bases[0] is not None else np.nan,
                            "on_2b": float(bases[1]) if bases[1] is not None else np.nan,
                            "on_3b": float(bases[2]) if bases[2] is not None else np.nan,
                            "batter": batter, "pitcher": pitcher, "fielder_2": catcher,
                            "stand": stand, "p_throws": p_hand,
                            "pitch_type": str(RNG.choice(PITCH_TYPES, p=PITCH_W)),
                            "release_speed": round(float(RNG.normal(92.5, 5.4)), 1),
                            "release_pos_x": round(float(RNG.normal(-1.6, 0.8)), 3),
                            "release_pos_z": round(float(RNG.normal(5.9, 0.35)), 3),
                            "pfx_x": round(float(RNG.normal(0, 0.8)), 3),
                            "pfx_z": round(float(RNG.normal(0.9, 0.7)), 3),
                            "plate_x": round(px, 4), "plate_z": round(pz, 4),
                            "sz_top": round(top + float(RNG.normal(0.11, 0.05)), 3),
                            "sz_bot": round(bot + float(RNG.normal(-0.07, 0.04)), 3),
                            "zone": int(RNG.integers(1, 15)),
                            "home_score": score["home"], "away_score": score["away"],
                            "bat_score": bat_score, "fld_score": fld_score,
                            "home_team": home, "away_team": away,
                            "events": None,
                        }

                        # ---- what the batter and umpire do ----
                        called = False
                        swung = RNG.random() < (SWING_IN if abs_strike else SWING_OUT)[(balls, strikes)]
                        if swung:
                            if RNG.random() >= CONTACT[abs_strike]:
                                desc, ptype = "swinging_strike", "S"
                            elif RNG.random() < FOUL_SHARE[abs_strike]:
                                desc, ptype = "foul", "S"
                            else:
                                desc, ptype = "hit_into_play", "X"
                            called_strike = False
                        elif not abs_strike and RNG.random() < HBP_RATE:
                            desc, ptype, called_strike = "hit_by_pitch", "B", False
                        else:
                            called = True
                            # Umpires miss more the closer the pitch is to the edge,
                            # and some umpires miss more than others.
                            p_err = float(np.clip(0.42 * err_mult * np.exp(-abs(true_dist) * 10.5),
                                                  0.004, 0.48))
                            called_strike = abs_strike != (RNG.random() < p_err)
                            desc = "called_strike" if called_strike else "ball"
                            ptype = "S" if called_strike else "B"
                        row["description"], row["type"] = desc, ptype
                        effective_strike = called_strike

                        # ---- challenge decision, obeying the real rules ----
                        if called:
                            # The call goes against the batter on a strike,
                            # against the fielding side on a ball.
                            side = bat_side if called_strike else fld_side
                            wrong = abs_strike != called_strike
                            if rem[side] > 0:
                                # Role is decided first, because judgment differs by
                                # role: MLB has reported catchers are the most
                                # successful challengers, and that only falls out of
                                # the data if catchers are modelled as better at
                                # telling a wrong call from a right one.
                                if side == bat_side:
                                    role, cid = "batter", batter
                                else:
                                    role = "catcher" if RNG.random() < 0.72 else "pitcher"
                                    cid = catcher if role == "catcher" else pitcher
                                p_wrong = {"batter": 0.17, "catcher": 0.30, "pitcher": 0.20}
                                p_right = {"batter": 0.0300, "catcher": 0.0040, "pitcher": 0.0065}
                                jd = judgment.get(cid, 1.0)
                                # nobody challenges an obvious call: the chance of
                                # challenging a CORRECT call fades with distance
                                p_chal = (min(p_wrong[role] * jd, 0.6) if wrong
                                          else p_right[role] / jd
                                          * float(np.exp(-max(abs(true_dist) - 0.1, 0) * 3.0)))
                                if abs(true_dist) < 0.12:
                                    p_chal += 0.03
                                # Bigger calls get challenged more: strike three,
                                # ball four, and close games late.
                                ends_pa = (called_strike and strikes == 2) or \
                                          (not called_strike and balls == 3)
                                diff = abs(score["home"] - score["away"])
                                mult = (1.6 if ends_pa else 1.0) * \
                                       (1.3 if inning >= 7 and diff <= 2 else 1.0) * \
                                       (0.6 if diff >= 6 else 1.0) * \
                                       (0.8 if rem[side] == 1 else 1.0)
                                if RNG.random() < min(p_chal * mult, 0.9):
                                    chal_seq += 1
                                    overturned = bool(wrong)
                                    chal_rows.append({
                                        "game_pk": pk, "game_date": gdate.isoformat(),
                                        "home_team": home, "away_team": away,
                                        "hp_umpire": ump, "hp_umpire_id": ump_id,
                                        "at_bat_index": ab - 1,    # 0-based, as the Stats API reports it
                                        "play_event_index": pitch_no - 1,
                                        "pitch_number": pitch_no,
                                        "inning": inning,
                                        "half_inning": half.lower().replace("bot", "bottom"),
                                        "is_extra_inning": inning > 9,
                                        "balls_before": balls, "strikes_before": strikes,
                                        "balls_after": balls, "strikes_after": strikes,
                                        "outs": outs,
                                        "batter_id": batter, "batter_name": names.get(batter),
                                        "pitcher_id": pitcher, "pitcher_name": names.get(pitcher),
                                        "challenger_id": cid,
                                        "challenger_name": names.get(cid),
                                        "challenger_role": role,
                                        "challenging_side": side,
                                        "call_code": "C" if called_strike else "B",
                                        "call_desc": "Called Strike" if called_strike else "Ball",
                                        "is_overturned": overturned,
                                        "review_type": "ABS",
                                        "challenge_team_id": 100 + TEAMS.index(home if side == "home" else away),
                                        "description": f"{role} challenged, "
                                                       f"{'overturned' if overturned else 'call stands'}",
                                        "has_review_flag": True,
                                        "api_px": round(tx, 4), "api_pz": round(tz, 4),
                                        "api_sz_top": round(top, 3), "api_sz_bot": round(bot, 3),
                                        "start_time": f"{gdate}T19:10:00Z",
                                        "challenge_seq_in_game": chal_seq,
                                    })
                                    if not overturned:
                                        rem[side] -= 1      # only failures cost
                                    else:
                                        # The call flips, so the count flips too.
                                        effective_strike = not called_strike
                                        if recorded_call == "final":
                                            row["description"] = ("called_strike" if effective_strike
                                                                  else "ball")
                                            row["type"] = "S" if effective_strike else "B"

                        # ---- the result of the pitch ----
                        runs = 0
                        if desc == "hit_into_play":
                            q = quality_base * (1.0 if abs_strike else OUT_CHASE_PENALTY)
                            ev, bases, o_add, runs = _ball_in_play(bases, outs, batter, q)
                            outs += o_add
                            row["events"], pa_over = ev, True
                        elif desc == "hit_by_pitch":
                            bases, runs = _force(bases, batter)
                            row["events"], pa_over = "hit_by_pitch", True
                        elif desc == "foul":
                            strikes = min(strikes + 1, 2)
                        elif desc == "swinging_strike" or (called and effective_strike):
                            strikes += 1
                        else:
                            balls += 1
                        if not pa_over and strikes >= 3:
                            outs += 1
                            row["events"], pa_over = "strikeout", True
                        elif not pa_over and balls >= 4:
                            bases, runs = _force(bases, batter)
                            row["events"], pa_over = "walk", True

                        # ---- running between pitches (the plate appearance goes on) ----
                        if not pa_over and outs < 3:
                            if (called and not effective_strike and any(b is not None for b in bases)
                                    and RNG.random() < WILD_PITCH):
                                bases, wp_runs = _advance_all(bases)
                                runs += wp_runs
                            elif bases[0] is not None and bases[1] is None and \
                                    RNG.random() < STEAL_TRY_1B:
                                if RNG.random() < STEAL_OK_1B:
                                    bases = [None, bases[0], bases[2]]
                                else:
                                    bases = [None, bases[1], bases[2]]
                                    outs += 1
                                    if outs >= 3:
                                        # caught stealing for the third out: the batter
                                        # leads off the next inning with a fresh count
                                        row["events"], pa_over = "caught_stealing_2b", True
                                        advance_spot = False
                            elif bases[1] is not None and bases[2] is None and \
                                    RNG.random() < STEAL_TRY_2B:
                                if RNG.random() < STEAL_OK_2B:
                                    bases = [bases[0], None, bases[1]]
                                else:
                                    bases = [bases[0], None, bases[2]]
                                    outs += 1
                                    if outs >= 3:
                                        row["events"], pa_over = "caught_stealing_3b", True
                                        advance_spot = False

                        # ---- walk-off: the game ends when the winning run scores ----
                        if (half == "Bot" and inning >= 9 and runs
                                and score["home"] + runs > score["away"]):
                            if row["events"] != "home_run":
                                runs = score["away"] + 1 - score["home"]
                            game_over, pa_over = True, True
                        score[bat_side] += runs
                        allowed[fld_side] += runs
                        row["post_home_score"], row["post_away_score"] = score["home"], score["away"]
                        row["post_bat_score"], row["post_fld_score"] = score[bat_side], score[fld_side]
                        pitch_rows.append(row)
                        if pitch_no >= 20 and not pa_over:      # safety valve, never hit in practice
                            outs += 1
                            row["events"], pa_over = "field_out", True
                    if advance_spot:
                        spot[bat_side] += 1
                if game_over:
                    break
            if not game_over and inning >= 9 and score["home"] != score["away"]:
                game_over = True

        game_rows.append({
            "game_pk": pk, "game_date": gdate.isoformat(), "game_type": "R",
            "home_team": home, "away_team": away,
            "home_team_id": 100 + TEAMS.index(home),
            "away_team_id": 100 + TEAMS.index(away),
            "hp_umpire": ump, "hp_umpire_id": ump_id,
            "n_innings": inning, "venue_id": 1000 + TEAMS.index(home),
            "home_final": score["home"], "away_final": score["away"],
        })

    truth = {"team_offense": off, "team_prevention": prv, "umpire_error": ump_err,
             "challenger_judgment": judgment}
    return pd.DataFrame(pitch_rows), pd.DataFrame(chal_rows), pd.DataFrame(game_rows), truth


def build_support(root, pitches, games, people):
    """Framing, umpire accuracy and win probability, consistent with the pitches."""
    # --- win probability, one row per plate appearance ---
    pa = pitches.groupby(["game_pk", "at_bat_number", "inning", "inning_topbot"],
                         as_index=False).size()
    li = np.clip(RNG.lognormal(np.log(0.62 + 0.085 * pa["inning"]), 0.60), 0.05, 6.0)
    wp = np.clip(RNG.normal(50, 18, len(pa)), 2, 98)
    winprob = pd.DataFrame({
        "game_pk": pa["game_pk"],
        "at_bat_index": pa["at_bat_number"] - 1,       # 0-based, matching the API
        "inning": pa["inning"],
        "half_inning": pa["inning_topbot"].str.lower().str.replace("bot", "bottom"),
        "outs": RNG.integers(0, 3, len(pa)),
        "balls": 0, "strikes": 0,
        "home_score": RNG.integers(0, 8, len(pa)),
        "away_score": RNG.integers(0, 8, len(pa)),
        "home_win_prob": np.round(wp, 2), "away_win_prob": np.round(100 - wp, 2),
        "home_win_prob_added": np.round(RNG.normal(0, 2.4, len(pa)), 2),
        "leverage_index": np.round(li, 3),
        "batter_id": 0, "pitcher_id": 0, "event": "Single",
    })
    winprob["bat_win_prob"] = np.where(winprob["half_inning"] == "bottom",
                                       winprob["home_win_prob"], winprob["away_win_prob"])
    write_table(winprob, root / "raw" / "winprob" / "winprob_2026.parquet")

    # --- catcher framing ---
    cs = pitches["fielder_2"].value_counts()
    fr = pd.DataFrame({
        "player_id": cs.index,
        "player_name": [f"Catcher {i}" for i in range(len(cs))],
        "team": RNG.choice(TEAMS, len(cs)),
        "n_called_pitches": cs.values,
        "strike_rate": np.round(np.clip(RNG.normal(0.503, 0.017, len(cs)), 0.44, 0.57), 4),
        "runs_extra_strikes": np.round(RNG.normal(0, 4.2, len(cs)), 2),
        "season": 2026,
    })
    write_table(fr, root / "raw" / "savant" / "catcher_framing_all_years.parquet")
    fr.to_csv(root / "raw" / "savant" / "catcher_framing_2026.csv", index=False)

    # --- umpire accuracy, derived the same way script 05 derives it ---
    called = pitches[pitches["description"].isin(["called_strike", "ball"])].merge(
        games[["game_pk", "hp_umpire", "hp_umpire_id"]], on="game_pk", how="left")
    h = dict(zip(people["player_id"], people["height_ft"]))
    bh = called["batter"].map(h).fillna(6.0)
    dist = zone_distance_ft(called["plate_x"], called["plate_z"],
                            bh * ZONE_TOP_PCT, bh * ZONE_BOT_PCT)
    called = called.assign(
        abs_strike=dist < 0,
        called_strike=called["description"] == "called_strike",
        dist_in=dist * 12)
    called["correct"] = called["abs_strike"] == called["called_strike"]
    called["missed_strike"] = called["abs_strike"] & ~called["called_strike"]
    called["missed_ball"] = ~called["abs_strike"] & called["called_strike"]

    ump = called.groupby(["hp_umpire", "hp_umpire_id"], as_index=False).agg(
        n_called=("correct", "size"), n_games=("game_pk", "nunique"),
        accuracy=("correct", "mean"),
        missed_strike_rate=("missed_strike", "mean"),
        missed_ball_rate=("missed_ball", "mean"),
        mean_abs_dist_in=("dist_in", lambda s: s.abs().mean()))
    edge = called[called["dist_in"].abs() <= 2.0]
    ump = ump.merge(edge.groupby(["hp_umpire", "hp_umpire_id"], as_index=False).agg(
        n_borderline=("correct", "size"), borderline_accuracy=("correct", "mean")),
        on=["hp_umpire", "hp_umpire_id"], how="left")
    write_table(ump, root / "raw" / "umpires" / "umpire_accuracy_derived_2026.parquet")

    gm = called.groupby(["game_pk", "hp_umpire", "hp_umpire_id"], as_index=False).agg(
        n_called=("correct", "size"), accuracy=("correct", "mean"),
        n_missed_strikes=("missed_strike", "sum"), n_missed_balls=("missed_ball", "sum"))
    write_table(gm, root / "raw" / "umpires" / "umpire_game_accuracy_2026.parquet")

    # --- ABS leaderboard stand-ins, so script 03's outputs exist too ---
    sav = root / "raw" / "savant"
    sav.mkdir(parents=True, exist_ok=True)
    return called


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=90)
    ap.add_argument("--out", default="data_demo")
    ap.add_argument("--recorded-call", choices=["original", "final"], default="original",
                    help="which call Statcast's description holds on challenged pitches")
    args = ap.parse_args()

    root = Path(args.out)
    print(f"Generating a synthetic {args.games}-game season under {root}/")
    print("These numbers are invented, but the ABS rules they obey are real.\n")

    people = build_people()
    pitches, challenges, games, truth = simulate(args.games, people,
                                                 recorded_call=args.recorded_call)

    pitches["game_date"] = pd.to_datetime(pitches["game_date"])
    challenges["game_date"] = pd.to_datetime(challenges["game_date"])

    for month, grp in pitches.groupby(pitches["game_date"].dt.to_period("M")):
        write_table(grp, root / "raw" / "statcast" / f"statcast_2026_{month}.parquet")
    write_table(challenges, root / "raw" / "challenges" / "abs_challenges_2026.parquet")
    write_table(games, root / "raw" / "challenges" / "games_2026.parquet")
    write_table(people, root / "raw" / "players" / "player_bios_2026.parquet")
    called = build_support(root, pitches, games, people)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "_demo_truth.json").write_text(json.dumps({
        "recorded_call": args.recorded_call,
        "umpire_error_multiplier": truth["umpire_error"],
        "team_offense": truth["team_offense"], "team_prevention": truth["team_prevention"],
        "challenger_judgment": {str(k): v for k, v in truth["challenger_judgment"].items()},
    }, indent=1))

    n_ch = len(challenges)
    print(f"  pitches        {len(pitches):>8,}  ({pitches['game_pk'].nunique()} games)")
    print(f"  called pitches {len(called):>8,}  ({len(called)/len(pitches):.1%})")
    print(f"  challenges     {n_ch:>8,}  ({n_ch/max(games.shape[0],1):.2f} per game)")
    print(f"  overturn rate  {challenges['is_overturned'].mean():>8.1%}")
    # Does the simulated season look like MLB? (targets: MLB 2023-25)
    pa_end = pitches[pitches["events"].notna()]
    n_pa = len(pa_end)
    ev = pa_end["events"].value_counts()
    runs_pg = (games["home_final"] + games["away_final"]).mean() / 2
    home_win = (games["home_final"] > games["away_final"]).mean()
    print(f"\n  realism check          simulated   MLB 2023-25")
    print(f"  runs / team / game     {runs_pg:>9.2f}   4.4-4.6")
    print(f"  strikeout %            {ev.get('strikeout', 0) / n_pa:>9.1%}   22-23%")
    print(f"  walk %                 {ev.get('walk', 0) / n_pa:>9.1%}   8-9%")
    print(f"  home run %             {ev.get('home_run', 0) / n_pa:>9.1%}   ~3%")
    print(f"  pitches / PA           {len(pitches) / n_pa:>9.2f}   ~3.9")
    print(f"  extra-inning games     {(games['n_innings'] > 9).mean():>9.1%}   ~8-9%")
    print(f"  home team wins         {home_win:>9.1%}   ~53%")
    print(f"  players        {len(people):>8,}")
    by_role = challenges.groupby("challenger_role")["is_overturned"].agg(["size", "mean"])
    print("\n  by challenger role:")
    print(by_role.to_string())

    # Prove the fixture itself obeys the rules before anyone builds on it.
    bad = 0
    for pk, g in challenges.groupby("game_pk"):
        for side, gs in g.groupby("challenging_side"):
            fails = (~gs["is_overturned"].astype(bool)).sum()
            cap = START_CHALLENGES + max(0, int(games.loc[games.game_pk == pk,
                                                          "n_innings"].iloc[0]) - 9)
            if fails > cap:
                bad += 1
    print(f"\n  rule check: {bad} game-sides exceed their failed-challenge budget "
          f"(expected 0)")

    print(f"""
Now run the Week 5 chain against it:

    ABS_DATA_DIR={root} python scripts/run_week5.py

or step by step:

    ABS_DATA_DIR={root} python scripts/08_build_challenge_inventory.py
    ABS_DATA_DIR={root} python scripts/09_data_dictionary.py
    ABS_DATA_DIR={root} python scripts/10_qa_assertions.py
    ABS_DATA_DIR={root} python scripts/11_verify_challenges_remaining.py""")


if __name__ == "__main__":
    main()
