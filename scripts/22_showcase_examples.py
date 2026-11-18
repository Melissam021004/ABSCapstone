#!/usr/bin/env python3
"""
22_showcase_examples.py — real 2026 pitches that show this semester's findings in action.

For the checkpoint presentation: finds the most extreme real examples of each
finding, describes every one in plain words (no jargon), and draws a slide-ready
figure for each — plus leaderboards, game win-probability stories, a "was it
challenged?" quiz, a glossary slide and a map of every challenge decision.
Every example carries a Baseball Savant link so you can find the replay.

  Categories (each a ranked list; the top few get a full slide):
    swing           the overturns that moved a team's chance of winning the most
    leverage        challenges made in the biggest moments (Tango's leverage)
    obvious_fixed   the most obvious mistakes that were challenged and fixed
    longshot        challenges with the lowest chance that still worked
    bad_challenge   challenges that weren't worth the risk (below the MDP's bar)
    missed_obvious  obvious mistakes nobody challenged, with a challenge in hand
    missed_costly   the most valuable challenges never made (above the bar)
    empty           costly wrong calls against a team with no challenges left
    contrast        same confidence, opposite answer: the bar moves with the game

    python scripts/22_showcase_examples.py
    python scripts/22_showcase_examples.py --per-category 5 --quiz 8
    python scripts/22_showcase_examples.py --video-links    # also look up direct replay links

Needs Weeks 9–11 (models/production, models/winprob, models/mdp). Reads the cohort,
every raw pitch, and data/raw/players/player_bios_2026 for names.
Writes  docs/SHOWCASE.md (presenter notes), docs/showcase_examples.csv,
        figures/showcase/ (slides, leaderboards, game stories, quiz, glossary, map)
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "eda"))

from src.abs_features import add_features, load_cohort            # noqa: E402
from src.abs_gamestate import build_states, load_pitches          # noqa: E402
from src.abs_leverage import Leverage                             # noqa: E402
from src.abs_mdp import load_mdp                                  # noqa: E402
from src.abs_predict import load_production                       # noqa: E402
from src.abs_value import KEY, valuation_table                    # noqa: E402
from src.abs_winprob import load_winprob                          # noqa: E402

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "showcase"
SAVANT_GAME = "https://baseballsavant.mlb.com/gamefeed?gamePk={pk}"
SAVANT_VIDEO = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
API = "https://statsapi.mlb.com/api/v1"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

BASES_TXT = ["bases empty", "runner on first", "runner on second", "runners on first and second",
             "runner on third", "runners on first and third", "runners on second and third",
             "bases loaded"]
OUTS_TXT = ["no outs", "one out", "two outs"]
CAT_ORDER = ["swing", "leverage", "obvious_fixed", "longshot", "bad_challenge",
             "missed_obvious", "missed_costly", "empty", "contrast"]
CATEGORIES = {
    "swing": ("The overturns that swung games the most",
              "A challenge won, the call flipped — and these flips moved the challenging team's "
              "chance of winning more than any others this season.",
              "dwp", "win-probability points gained by the challenging team"),
    "leverage": ("Challenges in the biggest moments",
                 "Leverage (Tom Tango's Leverage Index) measures how much a moment can swing a game: "
                 "1 is a typical plate appearance. These challenges came at the highest-leverage moments.",
                 "li", "leverage index of the moment (1 = typical)"),
    "obvious_fixed": ("The most obvious mistakes that were fixed",
                      "Called pitches that were wrong by the widest margin — and the team noticed, "
                      "challenged, and got the call reversed.",
                      "dist_against_call", "inches the ball was on the wrong side of the zone edge"),
    "longshot": ("Long shots that worked",
                 "Challenges that the overturn model gave the lowest chance — and that were reversed "
                 "anyway. (Low chance doesn't mean never.)",
                 "p_s", "chance the call would be reversed (the overturn model, p(s))"),
    "bad_challenge": ("Challenges that weren't worth the risk",
                      "By the MDP, these challenges risked a challenge worth more than what they could "
                      "win: the chance of a reversal was below the bar for that moment.",
                      "regret", "win-probability points given up by challenging"),
    "missed_obvious": ("Obvious mistakes nobody challenged",
                       "The call was clearly wrong, the team still had a challenge, the game was still "
                       "in the balance — and nobody challenged.",
                       "dist_against_call", "inches the ball was on the wrong side of the zone edge"),
    "missed_costly": ("The most valuable challenges never made",
                      "Moments where the chance of a reversal was well above the bar and the call "
                      "mattered — the biggest gains left on the table by the MDP's measure.",
                      "regret", "win-probability points left on the table"),
    "empty": ("When a team had nothing left",
              "Costly wrong calls against a team that had already used up its challenges — "
              "what holding a challenge for later is for.",
              "dwp", "win-probability points the wrong call cost"),
    "contrast": ("Same confidence, opposite answer",
                 "Two calls with about the same chance of being wrong. The bar to challenge depends on "
                 "the moment and on how many challenges are left, so the right answer flips.",
                 "p_s", "chance the call would be reversed"),
}


# =================================================================== plain words
def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def inning_txt(inning, half):
    return f"{'Top' if int(half) == 0 else 'Bottom'} of the {ordinal(inning)}"


def inning_short(inning, half):
    return f"{'top' if int(half) == 0 else 'bot'} {int(inning)}"


def score_txt(home, away, hs, as_):
    hs, as_ = int(hs), int(as_)
    if hs == as_:
        return f"tied {hs}–{as_}"
    return f"{home} leads {hs}–{as_}" if hs > as_ else f"{away} leads {as_}–{hs}"


def count_txt(b, s):
    b, s = int(b), int(s)
    return "full count (3-2)" if (b, s) == (3, 2) else f"{b}-{s} count"


def pct(x, d=0):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def pts(x, d=1, sign=True):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:{'+' if sign else ''}.{d}f} pts"


def abs_verdict(zone_dist_in):
    """What the tracked location says, from the signed distance between the ball's surface
    and the ABS zone (positive = a gap = ball; zero or negative = touching = strike)."""
    d = float(zone_dist_in)
    if not np.isfinite(d):
        return "Tracked location unknown", None
    if d > 0:
        return f"Tracked location: BALL — the whole ball missed the zone by {d:.1f} in.", "ball"
    return f"Tracked location: STRIKE — the ball caught {abs(d):.1f} in. of the zone.", "strike"


def describe(r):
    """Every plain-language string a slide or the notes needs, for one pitch."""
    call = "STRIKE" if r["call_strike"] else "BALL"
    team = r["chall_team"]
    who = r["role"].lower() if isinstance(r["role"], str) and r["role"] else ""
    cname = r.get("challenger_name") if isinstance(r.get("challenger_name"), str) else ""
    if r["challenged"]:
        whom = f"the {who}" + (f", {cname}" if cname else "") if who else (cname or "")
        happened = (f"{team} challenged ({whom}) — " +
                    ("OVERTURNED: the call was reversed." if r["overturned"] else "the call STOOD."))
    elif r["r_self"] >= 1:
        happened = (f"Not challenged — {team} still had {int(r['r_self'])} "
                    f"challenge{'s' if r['r_self'] > 1 else ''} left.")
    else:
        happened = f"Couldn't be challenged — {team} had no challenges left."
    bar = r["p_star"]
    if r["r_self"] >= 1 and np.isfinite(bar):
        model = ("Worth a challenge" if r["p_s"] >= bar else "Better to hold") + \
                f" — {pct(r['p_s'])} chance vs. a bar of {pct(bar)}"
    else:
        model = "No decision to make — no challenge left"
    verdict, _ = abs_verdict(r["zone_dist_in"])
    if 0.25 <= r["p_s"] <= 0.75:
        verdict += " Too close to call by eye — a challenge could go either way."
    return {
        "situation": f"{inning_txt(r['inning'], r['half'])} · {score_txt(r['home_team'], r['away_team'], r['home_score'], r['away_score'])}",
        "runners": f"{BASES_TXT[int(r['bases'])].capitalize()}, {OUTS_TXT[int(r['outs'])]} · {count_txt(r['balls'], r['strikes'])}",
        "matchup": f"{r['batter_name']} batting vs. {r['pitcher_name']}",
        "game": f"{pd.Timestamp(r['game_date']):%b %-d, %Y} · {r['away_team']} at {r['home_team']}",
        "call": f"Umpire called it a {call}",
        "verdict": verdict,
        "decider": f"{team} ({'batting' if r['side'] == 'batting' else 'in the field'})",
        "swing_txt": (f"{team}'s chance to win: {pct(r['wp_stands_chall'])} if the call stands, "
                      f"{pct(r['wp_over_chall'])} if reversed ({pts(r['dwp'])})"),
        "happened": happened,
        "model": model,
        "short": (f"{pd.Timestamp(r['game_date']):%b %-d} · {r['away_team']} @ {r['home_team']} · "
                  f"{inning_short(r['inning'], r['half'])} · {int(r['balls'])}-{int(r['strikes'])}"),
    }


# =================================================================== data
def player_names():
    """player_id -> full name, from script 07's bios (parquet or csv)."""
    for name in ("player_bios_2026.parquet", "player_bios_2026.csv"):
        p = DATA_DIR / "raw" / "players" / name
        if p.exists():
            b = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            idc = next((c for c in ("player_id", "id", "mlbam_id") if c in b.columns), None)
            nmc = next((c for c in ("full_name", "fullName", "name") if c in b.columns), None)
            if idc and nmc:
                return dict(zip(pd.to_numeric(b[idc], errors="coerce"), b[nmc].astype(str)))
    return {}


def build_table():
    """Every called pitch of 2026 with its state, p(s), values, leverage and the MDP's bar."""
    pmodel, wp = load_production(), load_winprob()
    raw = load_cohort(DATA_DIR)
    cohort = add_features(raw)
    val = valuation_table(cohort, pmodel, wp)
    want = ["plate_x", "plate_z", "abs_zone_top", "abs_zone_bot", "batter", "pitcher", "fielder_2",
            "stand", "p_throws", "pitch_type", "release_speed", "home_score", "away_score",
            "balls", "strikes", "challenger_name", "challenges_remaining_batting",
            "challenges_remaining_fielding", "zone_dist_in", "description", "events"]
    have = [c for c in want if c in cohort.columns and c not in val.columns]
    val = val.merge(cohort[KEY + have].drop_duplicates(KEY), on=KEY, how="left")
    raw_p = load_pitches(DATA_DIR, (2026,))[0]
    pitches, games = build_states(raw_p)
    st = pitches[KEY + ["outs", "bases", "count", "diff", "half"]]
    val = val.merge(st, on=KEY, how="inner")
    for c in ("outs", "bases", "count", "diff", "half", "inning"):
        val[c] = val[c].astype(int)
    for c, src in (("r_bat", "challenges_remaining_batting"), ("r_fld", "challenges_remaining_fielding")):
        val[c] = np.clip(pd.to_numeric(val.get(src), errors="coerce").fillna(0), 0, 2).astype(int)
    strike = val["call_strike"].to_numpy(bool)
    val["r_self"] = np.where(strike, val["r_bat"], val["r_fld"])

    sol = load_mdp()
    args_ = (val["inning"].to_numpy(), val["half"].to_numpy(), val["outs"].to_numpy(),
             val["bases"].to_numpy(), val["count"].to_numpy(), val["diff"].to_numpy(),
             val["r_bat"].to_numpy(), val["r_fld"].to_numpy())
    for fld, (a, b) in {"p_star": ("thr_S", "thr_B"), "hold": ("H_S", "H_B"),
                        "swing": ("D_S", "D_B")}.items():
        val[fld] = np.where(strike, sol.lookup(a, *args_), sol.lookup(b, *args_))
    lev = Leverage.fit(wp, pitches)
    val["li"] = lev.for_states(val)
    p, H, Dl = val["p_s"].to_numpy(), val["hold"].to_numpy(), val["swing"].to_numpy()
    gain = p * (Dl + H) - H
    can = val["r_self"].to_numpy() >= 1
    val["model_says"] = can & np.isfinite(val["p_star"]) & (p >= val["p_star"].to_numpy())
    did = val["challenged"].to_numpy(bool)
    val["regret"] = np.where(can, np.where(did, np.maximum(-gain, 0), np.maximum(gain, 0)), 0.0)
    names = player_names()
    for c, src in (("batter_name", "batter"), ("pitcher_name", "pitcher"), ("catcher_name", "fielder_2")):
        ids = pd.to_numeric(val.get(src), errors="coerce") if src in val.columns else pd.Series(np.nan, index=val.index)
        val[c] = ids.map(names).fillna("the batter" if c == "batter_name" else
                                       "the pitcher" if c == "pitcher_name" else "the catcher")
    return val, pitches, games, wp, lev


# =================================================================== picking examples
def pick(pool, sort_col, ascending, n, used, one_per_game=True):
    pool = pool[np.isfinite(pool[sort_col])].sort_values(sort_col, ascending=ascending)
    out, games = [], set()
    for idx, r in pool.iterrows():
        key = tuple(r[k] for k in KEY)
        if key in used or (one_per_game and r["game_pk"] in games):
            continue
        out.append(idx)
        used.add(key)
        games.add(r["game_pk"])
        if len(out) >= n:
            break
    return pool.loc[out]


def trustworthy(val, max_inches):
    """
    Rows safe to show as an example. Picking the most EXTREME pitches of a season also
    picks the noisiest numbers, so two guards:
      * Week 10's value of the call (dwp) and the MDP's swing (Δ) — built two different
        ways — must agree (within 1 point, or 25%), and the call must matter (dwp > 0).
        They disagree only in thin cells (a rare count with rare runners), where one
        season can't pin the value down.
      * the call can't be wrong by more than `max_inches` (the pitch that far on the wrong
        side of the edge): beyond that it's almost always a tracking or record-linking
        glitch, not a real call. (Correct calls far from the zone are fine.)
    """
    ok = (np.isfinite(val["p_s"]) & val["plate_x"].notna() & val["plate_z"].notna()
          & val["abs_zone_top"].notna())
    d, m = val["dwp"].to_numpy(float), val["swing"].to_numpy(float)
    agree = (d > 0) & (np.abs(d - m) <= np.maximum(0.01, 0.25 * np.abs(m)))
    sane = ~(val["dist_against_call"].to_numpy(float) > max_inches)      # wrong by more than that: a glitch
    return ok & agree & sane, {"total": int(ok.sum()), "disagree": int((ok & ~agree).sum()),
                               "too_far": int((ok & agree & ~sane).sum())}


def categories(val, n, used, keep):
    """{key: (slide rows, leaderboard rows)}."""
    v = val[keep]
    opp = v[v["opportunity"] & (v["r_self"] >= 1)]
    ch = opp[opp["challenged"]]
    pools = {
        "swing": (ch[ch["overturned"]], "dwp", False),
        "leverage": (ch[ch["dwp"] >= 0.01], "li", False),
        "obvious_fixed": (ch[ch["overturned"] & (ch["p_s"] >= 0.99)], "dist_against_call", False),
        "longshot": (ch[ch["overturned"]], "p_s", True),
        "bad_challenge": (ch[~ch["model_says"]], "regret", False),
        "missed_obvious": (opp[~opp["challenged"] & (opp["p_s"] >= 0.99) & (opp["dwp"] >= 0.01)],
                           "dist_against_call", False),
        "missed_costly": (opp[~opp["challenged"] & opp["model_says"]], "regret", False),
        "empty": (v[(v["r_self"] == 0) & (v["p_s"] >= 0.9) & ~v["challenged"]], "dwp", False),
    }
    out = {}
    for key, (pool, col, asc) in pools.items():
        board = pool[np.isfinite(pool[col])].sort_values(col, ascending=asc).head(10)
        out[key] = (pick(pool, col, asc, n, used), board)
    # the contrast pair: about the same chance of being wrong, opposite answers
    mid = opp[(opp["p_s"] >= 0.55) & (opp["p_s"] <= 0.85) & np.isfinite(opp["p_star"])]
    hold = mid[(mid["inning"] <= 3) & (mid["r_self"] == 1) & (mid["p_s"] < mid["p_star"])]
    rows = []
    if len(hold):
        a = hold.assign(gap=hold["p_star"] - hold["p_s"]).sort_values("gap", ascending=False).iloc[0]
        go = mid[(mid["inning"] >= 8) & (mid["diff"].abs() <= 1) & (mid["p_s"] >= mid["p_star"])]
        go = go[(go["p_s"] - a["p_s"]).abs() <= 0.05]
        if len(go):
            b = go.sort_values("p_star").iloc[0]
            rows = [a.name, b.name]
            used.update({tuple(a[k] for k in KEY), tuple(b[k] for k in KEY)})
    pair = val.loc[rows]
    out["contrast"] = (pair, pair)
    return out


def quiz_set(val, n, used, seed, keep):
    """Obvious mistakes that mattered, half challenged and half not, shuffled."""
    v = val[keep & val["opportunity"] & (val["r_self"] >= 1)]
    clear = v[(v["p_s"] >= 0.99) & (v["dwp"] >= 0.01)]
    k = n // 2
    a = pick(clear[~clear["challenged"]], "dist_against_call", False, n - k, used)
    b = pick(clear[clear["challenged"] & clear["overturned"]], "dist_against_call", False, k, used)
    q = pd.concat([a, b])
    return q.sample(frac=1.0, random_state=seed) if len(q) else q


# =================================================================== replay links
def video_links(rows):
    """{(game_pk, at_bat_number, pitch_number): playId} from the MLB Stats API (optional)."""
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json"})
    found = {}
    for pk in sorted(rows["game_pk"].unique()):
        try:
            js = s.get(f"{API}/game/{int(pk)}/playByPlay", timeout=15).json()
        except Exception as e:                                    # no internet, API change …
            print(f"  (couldn't reach the Stats API for game {pk}: {type(e).__name__}) — Savant game links only")
            return found
        for play in js.get("allPlays", []) or []:
            ab = (play.get("about") or {}).get("atBatIndex")
            for ev in play.get("playEvents", []) or []:
                if ev.get("isPitch") and ev.get("playId") and ab is not None:
                    found[(int(pk), int(ab) + 1, int(ev.get("pitchNumber") or 0))] = ev["playId"]
        time.sleep(0.3)
    return found


# =================================================================== game stories
def game_series(pitches, game_pk, wp):
    """Home win probability before every pitch of one game, plus the final result."""
    g = pitches[pitches["game_pk"] == game_pk].sort_values(["at_bat_number", "pitch_number"])
    g = g.assign(half=g["half"].astype(int))
    v = wp.frame(g[["inning", "half", "outs", "bases", "home_score", "away_score", "balls", "strikes"]])
    return g.reset_index(drop=True), np.asarray(v, float)


def biggest_swings(pitches, wp, n=10):
    """The biggest single-pitch changes in win probability of the season (any pitch)."""
    p = pitches[pitches["usable"].fillna(False).astype(bool)].sort_values(KEY).reset_index(drop=True)
    v = np.asarray(wp.frame(p[["inning", "half", "outs", "bases", "home_score", "away_score", "balls",
                               "strikes"]]), float)
    nxt = np.roll(v, -1)
    last = ~p["game_pk"].eq(p["game_pk"].shift(-1)).to_numpy()
    nxt[last] = p["home_win"].to_numpy(float)[last]
    bat = np.where(p["half"].to_numpy(int) == 1, 1.0, -1.0)
    p["wp_change_batting"] = (nxt - v) * bat
    p["wp_before_batting"] = np.where(bat > 0, v, 1 - v)
    top = p.reindex(p["wp_change_batting"].abs().sort_values(ascending=False).index).head(n)
    return top


# =================================================================== notes
NOT_CHALLENGED = ("missed_obvious", "missed_costly", "empty")
OFFICIAL_PER_GAME = 10557 / 2429      # MLB's 2026 regular season (Baseball-Reference, AP)


def coverage_warning(val, demo):
    """A warning for the notes if the data has clearly fewer challenges than MLB's official count."""
    if demo:
        return None
    n_ch, n_g = int(val["challenged"].sum()), int(val["game_pk"].nunique())
    rate = n_ch / max(n_g, 1)
    if rate >= 0.95 * OFFICIAL_PER_GAME:
        return None
    return (f"> **Check before you present.** The data has {n_ch:,} challenges in {n_g:,} games "
            f"({rate:.2f} a game); MLB's official 2026 total is 10,557 ({OFFICIAL_PER_GAME:.2f} a game). "
            "Some challenges are missing (see `scripts/24_challenge_count_check.py`), so a pitch listed here "
            "as \"not challenged\" may have been challenged, and a team shown with a challenge left may have "
            "had none. Confirm every \"not challenged\" example and every quiz answer on the replay: the "
            "broadcast shows each challenge and the challenges left.")


def write_notes(path, demo, cats, quiz, texts, links, n_opp, warning=None):
    L = ["# Checkpoint showcase — this semester's findings in real 2026 pitches", ""]
    if demo:
        L += ["> **DEMO DATA** — a simulated season; names, teams and games are not real.", ""]
    if warning:
        L += [warning, ""]
    L += ["*Generated by `scripts/22_showcase_examples.py`. Slides: `figures/showcase/`. Every example has "
          "a Baseball Savant link. A game link opens the game feed: go to the inning, find the at-bat by the "
          "batter's name, then the pitch number within that at-bat. Run with `--video-links` to get a direct "
          "link to each pitch's video instead.*", "",
          "## A running order for the talk", "",
          "1. `00_glossary` — the four ideas, in plain words.",
          "2. `02_biggest_swings_any_pitch` — for scale: what a big moment looks like (walk-off home runs), "
          "next to the biggest swing a single call produced.",
          "3. `01_decision_map` — every challenge opportunity of the season; point at the circled examples.",
          "4. The category slides (`03_…` to `10_…`): each leaderboard, then its #1 slide with the replay, then "
          "its game story (`20_story_…`) where there is one.",
          "5. The quiz (`30_quiz_…`): question slide, ask the room, answer slide.",
          "6. `12_first_challenge_value`, `13_first_challenge_one_game` and `14_first_challenge_all_games` "
          "(made by `scripts/25_first_challenge_value.py`; the explainer is `docs/FIRST_CHALLENGE_VALUE.md`): "
          "what a challenge is worth. The “about 6 points” is the value of holding one for the whole game, "
          "not of the first pitch, and it is why the bar starts high.",
          "7. `11_contrast_1` and `11_contrast_2` — same chance the call was wrong, opposite answers: the bar "
          "moves with the inning, the score and the challenges left. A good place to end.", "",
          "## How to talk about the numbers (for people who don't follow baseball)", "",
          "- **Chance the call was wrong, p(s):** from exactly where the ball crossed the plate. If the team "
          "challenges, ABS (the robot umpire) reverses the call with this probability.",
          "- **Chance to win:** a team's probability of winning from this exact moment — inning, score, "
          "runners, outs and count (Week 10's model). A **point** is one percentage point; 100 points of "
          "win probability = one win.",
          "- **Leverage:** how much this moment can swing the game. 1 is a typical plate appearance; "
          "above 2 is a big moment; the most tense moments of a season reach about 10 (Tom Tango's "
          "Leverage Index).",
          "- **The bar:** the confidence a team needs before a challenge is worth the risk. A failed "
          "challenge is lost, so the bar is high early in a game with one challenge left, and near zero "
          "in a tied 9th inning (Week 11's MDP).",
          "- **What a challenge is worth:** what holding it for the rest of the game is worth, compared with "
          "having none: every wrong call it could still fix, since a challenge that succeeds is kept. At the "
          "first pitch that is Week 11 §3's number, far more than the first pitch's own call (slides 12–14).", "",
          f"Across the season there were {n_opp:,} challenge opportunities (a called pitch went against a "
          "team that still had a challenge).", ""]
    for key in CAT_ORDER:
        if key not in cats:
            continue
        slides, board = cats[key]
        title, blurb, col, unit = CATEGORIES[key]
        L += [f"## {title}", "", blurb, ""]
        for i, (idx, r) in enumerate(slides.iterrows(), 1):
            t = texts[idx]
            L += [f"### {title} — #{i}: {t['short']}", "",
                  f"- **The moment:** {t['situation']}; {t['runners'].lower()}. {t['matchup']}. ({t['game']})",
                  f"- **The call:** {t['call']}. {t['verdict']}",
                  f"- **Chance it was wrong:** {pct(r['p_s'])}. **Leverage:** {r['li']:.1f}.",
                  f"- **What it was worth:** {t['swing_txt']}.",
                  f"- **What happened:** {t['happened']}",
                  f"- **The MDP:** {t['model']}.",
                  f"- **Replay:** {links.get(idx, SAVANT_GAME.format(pk=int(r['game_pk'])))} — inning "
                  f"{int(r['inning'])} ({'top' if r['half'] == 0 else 'bottom'}), at-bat #{int(r['at_bat_number'])}, "
                  f"pitch #{int(r['pitch_number'])}."]
            if key in NOT_CHALLENGED:
                L.append("- **Before you show it:** on the replay, confirm nobody challenged, and check the "
                         "challenges-left graphic matches what this slide says.")
            L.append("")
        if len(board):
            L += [f"Top {len(board)} ({unit}):", ""]
            for j, (idx, r) in enumerate(board.iterrows(), 1):
                val = r[col]
                shown = (pts(val) if col in ("dwp", "regret") else f"{val:.2f}" if col == "li"
                         else f"{val:.1f} in." if col == "dist_against_call" else pct(val, 1))
                L.append(f"{j}. {describe(r)['short']} — {shown}")
            L.append("")
    if len(quiz):
        L += ["## Quiz: was it challenged?", "",
              "Show the question slide (where the pitch crossed the plate and the moment), ask the room, "
              "then show the answer slide. All of these calls were clearly wrong. Check each answer on the "
              "replay before class.", ""]
        for i, (idx, r) in enumerate(quiz.iterrows(), 1):
            t = texts[idx]
            L += [f"{i}. {t['short']} — {t['verdict']} **Answer:** {t['happened']} "
                  f"({'cost ' + pts(r['dwp']) if not r['challenged'] else 'gained ' + pts(r['dwp'])} of win "
                  f"probability) · replay: {links.get(idx, SAVANT_GAME.format(pk=int(r['game_pk'])))}"]
        L.append("")
    path.write_text("\n".join(L) + "\n")


# =================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--per-category", type=int, default=3, help="slides per category")
    ap.add_argument("--quiz", type=int, default=6, help="quiz questions (half challenged)")
    ap.add_argument("--stories", type=int, default=6, help="game win-probability stories")
    ap.add_argument("--video-links", action="store_true", help="look up direct replay links (internet)")
    ap.add_argument("--max-inches", type=float, default=8.0,
                    help="ignore pitches further than this on the wrong side of the edge (glitches)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    demo = "demo" in str(DATA_DIR).lower()
    print("Finding this season's showcase pitches" + (" (DEMO DATA)" if demo else ""))
    val, pitches, games, wp, lev = build_table()
    n_opp = int((val["opportunity"] & (val["r_self"] >= 1)).sum())
    print(f"  {len(val):,} called pitches · {n_opp:,} challenge opportunities")
    keep, why = trustworthy(val, args.max_inches)
    print(f"  example pool: {int(keep.sum()):,} pitches — left out {why['disagree']:,} where Week 10's and "
          f"the MDP's value of the call disagree (thin data) and {why['too_far']:,} more than "
          f"{args.max_inches:g} in. off (tracking/linking glitches)")
    used = set()
    cats = categories(val, args.per_category, used, keep)
    quiz = quiz_set(val, args.quiz, used, args.seed, keep)
    chosen = pd.concat([s for s, _ in cats.values()] + [b for _, b in cats.values()] + [quiz])
    chosen = chosen[~chosen.index.duplicated()]
    texts = {idx: describe(r) for idx, r in chosen.iterrows()}

    links = {}
    if args.video_links:
        print("  looking up replay links on the MLB Stats API …")
        ids = video_links(chosen)
        for idx, r in chosen.iterrows():
            pid = ids.get((int(r["game_pk"]), int(r["at_bat_number"]), int(r["pitch_number"])))
            if pid:
                links[idx] = SAVANT_VIDEO.format(pid=pid)
        print(f"  direct replay links for {len(links)} of {len(chosen)} pitches")

    rows = []
    for key in CAT_ORDER:
        if key not in cats:
            continue
        slides, board = cats[key]
        for i, (idx, r) in enumerate(slides.iterrows(), 1):
            rows.append({"category": key, "rank": i, "slide": True, **{k: r[k] for k in KEY}, **texts[idx],
                         "p_s": r["p_s"], "swing_pts": 100 * r["dwp"], "li": r["li"], "bar": r["p_star"],
                         "hold_pts": 100 * r["hold"], "regret_pts": 100 * r["regret"],
                         "inches_wrong": r["dist_against_call"], "challenged": r["challenged"],
                         "overturned": r["overturned"], "challenges_left": r["r_self"],
                         "replay": links.get(idx, SAVANT_GAME.format(pk=int(r["game_pk"])))})
        print(f"  {CATEGORIES[key][0]:<45} {len(slides)} slides · {len(board)} on the leaderboard")
    for i, (idx, r) in enumerate(quiz.iterrows(), 1):
        rows.append({"category": "quiz", "rank": i, "slide": True, **{k: r[k] for k in KEY}, **texts[idx],
                     "p_s": r["p_s"], "swing_pts": 100 * r["dwp"], "li": r["li"], "bar": r["p_star"],
                     "hold_pts": 100 * r["hold"], "regret_pts": 100 * r["regret"],
                     "inches_wrong": r["dist_against_call"], "challenged": r["challenged"],
                     "overturned": r["overturned"], "challenges_left": r["r_self"],
                     "replay": links.get(idx, SAVANT_GAME.format(pk=int(r["game_pk"])))})
    table = pd.DataFrame(rows)
    DOCS.mkdir(parents=True, exist_ok=True)
    table.to_csv(DOCS / "showcase_examples.csv", index=False)
    warning = coverage_warning(val, demo)
    if warning:
        print("  WARNING: the data has fewer challenges than MLB's official count — confirm every "
              "'not challenged' example on the replay (details at the top of SHOWCASE.md)")
    write_notes(DOCS / "SHOWCASE.md", demo, cats, quiz, texts, links, n_opp, warning)
    print(f"  quiz: {len(quiz)} questions · notes → {DOCS / 'SHOWCASE.md'} · table → "
          f"{DOCS / 'showcase_examples.csv'}")

    import showcase_figures as F
    FIGS.mkdir(parents=True, exist_ok=True)
    stories = []
    for key in ("swing", "leverage", "missed_costly", "missed_obvious", "empty", "bad_challenge"):
        s = cats.get(key, (pd.DataFrame(), None))[0]
        if len(s):
            stories.append((key, s.iloc[0]))
    stories = stories[:args.stories]
    series = {int(r["game_pk"]): game_series(pitches, int(r["game_pk"]), wp) for _, r in stories}
    swings = biggest_swings(pitches, wp)
    opp_all = val[val["opportunity"] & (val["r_self"] >= 1) & np.isfinite(val["p_s"])]
    n = F.make_all(FIGS, DATA_DIR, cats=cats, quiz=quiz, texts=texts, stories=stories, series=series,
                   swings=swings, opp=opp_all, names_of=CATEGORIES, order=CAT_ORDER,
                   games=games, links=links, describe=describe)
    print(f"  {n} figures → {FIGS}/   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
