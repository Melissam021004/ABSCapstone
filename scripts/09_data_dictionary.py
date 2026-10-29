#!/usr/bin/env python3
"""
09_data_dictionary.py — Week 5 step 7. Generates docs/DATA_DICTIONARY.md.

Reads the analysis-ready table built by script 08 and documents every column:
type, source, definition, null rate, range or example values, and any known
gotcha. Columns it doesn't recognise are still listed, with their statistics
and a flag saying the definition needs filling in — so a new Statcast column
can never silently go undocumented.

The part that will save you in Week 8: every column is tagged with a MODELLING
ROLE.

    target        the thing you are predicting
    feature       safe to use as a model input
    LEAKAGE RISK  known only AFTER the challenge resolved — using it as a
                  feature will give you a beautiful model and a worthless one
    identifier    keys and names, for joining and inspection, not modelling
    metadata      provenance and bookkeeping

The Week 8 note in your plan says an AUC above ~0.95 means hunt for the leak.
This table is where you hunt.

    python scripts/09_data_dictionary.py
    ABS_DATA_DIR=data_demo python scripts/09_data_dictionary.py

Output: docs/DATA_DICTIONARY.md
        docs/data_dictionary.csv
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
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))

S_STATCAST = "Statcast (script 01)"
S_API = "MLB Stats API (script 02)"
S_DERIVED = "Derived (script 08)"
S_BIOS = "MLB Stats API people (script 07)"
S_WP = "MLB Stats API winProbability (script 06)"
S_FRAMING = "Savant catcher framing (script 04)"
S_UMP = "Derived umpire accuracy (script 05)"

# name -> (source, modelling role, definition, gotcha)
REGISTRY = {
    # ---------------- keys ----------------
    "game_pk": (S_STATCAST, "identifier", "MLB game identifier.", ""),
    "game_date": (S_STATCAST, "metadata", "Date the game was played.", ""),
    "at_bat_number": (S_STATCAST, "identifier",
                      "Plate appearance number within the game, 1-based.",
                      "The Stats API's atBatIndex is 0-BASED. Join as at_bat_number - 1."),
    "pitch_number": (S_STATCAST, "identifier",
                     "Pitch number within the plate appearance, 1-based.", ""),

    # ---------------- game state ----------------
    "inning": (S_STATCAST, "feature", "Inning number.", ""),
    "inning_topbot": (S_STATCAST, "feature", "Top or Bot half.",
                      "Top = away team batting."),
    "balls": (S_STATCAST, "feature", "Balls in the count BEFORE this pitch.", ""),
    "strikes": (S_STATCAST, "feature", "Strikes in the count BEFORE this pitch.", ""),
    "outs_when_up": (S_STATCAST, "feature", "Outs before this pitch.", ""),
    "on_1b": (S_STATCAST, "feature", "Runner id on first, null if empty.",
              "Null means empty base, not missing data."),
    "on_2b": (S_STATCAST, "feature", "Runner id on second, null if empty.", ""),
    "on_3b": (S_STATCAST, "feature", "Runner id on third, null if empty.", ""),
    "home_score": (S_STATCAST, "feature", "Home score before the pitch.", ""),
    "away_score": (S_STATCAST, "feature", "Away score before the pitch.", ""),
    "batting_side": (S_DERIVED, "feature", "'home' or 'away' — which side is batting.",
                     "Derived from inning_topbot."),
    "fielding_side": (S_DERIVED, "feature", "The side in the field.", ""),

    # ---------------- participants ----------------
    "batter": (S_STATCAST, "feature", "Batter MLBAM id.",
               "High-cardinality categorical. CatBoost handles natively; do not one-hot."),
    "pitcher": (S_STATCAST, "feature", "Pitcher MLBAM id.", ""),
    "fielder_2": (S_STATCAST, "feature", "Catcher MLBAM id.",
                  "This is how catcher framing metrics join on."),
    "stand": (S_STATCAST, "feature", "Batter handedness, L or R.", ""),
    "p_throws": (S_STATCAST, "feature", "Pitcher handedness, L or R.", ""),
    "hp_umpire": (S_API, "feature", "Home-plate umpire name.",
                  "From the schedule endpoint with hydrate=officials."),
    "hp_umpire_id": (S_API, "identifier", "Home-plate umpire id.", ""),
    "pitcher_pos": (S_BIOS, "metadata", "Pitcher's listed primary position.",
                    "Used only to detect position players pitching."),

    # ---------------- pitch characteristics ----------------
    "pitch_type": (S_STATCAST, "feature", "Pitch type code (FF, SL, CH, ...).", ""),
    "release_speed": (S_STATCAST, "feature", "Release velocity, mph.", ""),
    "release_pos_x": (S_STATCAST, "feature", "Horizontal release point, ft.", ""),
    "release_pos_z": (S_STATCAST, "feature", "Vertical release point, ft.", ""),
    "pfx_x": (S_STATCAST, "feature", "Horizontal movement, ft.", ""),
    "pfx_z": (S_STATCAST, "feature", "Vertical movement, ft.", ""),
    "plate_x": (S_STATCAST, "feature",
                "Horizontal location at the plate, ft; 0 is the plate's centre.",
                "Catcher's perspective. Sign flips meaning by handedness — use the "
                "absolute value or interact with `stand`."),
    "plate_z": (S_STATCAST, "feature", "Height at the plate, ft above the ground.", ""),
    "zone": (S_STATCAST, "feature", "Savant's 1-14 zone bucket.",
             "Savant's own zone grid, NOT the ABS zone. Coarse; prefer zone_dist_in."),
    "description": (S_STATCAST, "metadata", "Pitch result as called.",
                    "called_strike / ball / foul / swinging_strike / hit_into_play / ..."),
    "type": (S_STATCAST, "metadata", "S, B or X.", ""),
    "sz_top": (S_STATCAST, "metadata", "Statcast's zone top, ft.",
               "NOT the ABS zone. Stance-dependent and set per pitch. Using it for "
               "the ABS calculation biases every distance feature."),
    "sz_bot": (S_STATCAST, "metadata", "Statcast's zone bottom, ft.", "See sz_top."),

    # ---------------- ABS zone ----------------
    "batter_height_ft": (S_BIOS, "feature", "Listed batter height in feet.",
                         "Self-reported and rounded to the inch; a 1-inch error moves "
                         "the zone edge about half an inch."),
    "abs_zone_top": (S_DERIVED, "feature", "ABS zone top = 53.5% of batter height, ft.",
                     "Falls back to sz_top when height is missing — check abs_zone_source."),
    "abs_zone_bot": (S_DERIVED, "feature", "ABS zone bottom = 27% of batter height, ft.", ""),
    "abs_zone_source": (S_DERIVED, "metadata",
                        "'batter_height' (correct) or 'statcast_sz' (fallback).",
                        "Any row with 'statcast_sz' has an approximated zone. Report "
                        "the share in your limitations."),
    "zone_dist_ft": (S_DERIVED, "feature",
                     "Signed distance from the BALL'S SURFACE to the ABS zone, ft. "
                     "Negative = overlapping the zone = ABS strike.",
                     "The ball's 1.45-inch radius is already subtracted, because any "
                     "part of the ball touching the zone is a strike."),
    "zone_dist_in": (S_DERIVED, "feature", "Same, in inches. The headline feature.",
                     "Highly non-linear near zero — give it a spline or let a tree "
                     "split it."),
    "abs_strike": (S_DERIVED, "feature", "True if ABS would call this a strike.", ""),
    "sc_zone_dist_in": (S_DERIVED, "metadata",
                        "Distance using Statcast's zone instead, for robustness checks.",
                        "Comparison only. Do not model on it."),
    "zone_defs_disagree": (S_DERIVED, "metadata",
                           "True where the ABS zone and Statcast's zone disagree.",
                           "Where the challenges live. Good diagnostic, weak feature."),

    # ---------------- challenge inventory ----------------
    "challenges_remaining_batting": (
        S_DERIVED, "feature",
        "Challenges the batting side has BEFORE this pitch (0-2).",
        "DERIVED, NOT PUBLISHED. Walked forward per game: start at 2, a failed "
        "challenge costs one, a successful one is retained, and in extra innings a "
        "side already at zero is granted one for that inning. This is the MDP's "
        "state variable r — hand-verify it (script 11)."),
    "challenges_remaining_fielding": (
        S_DERIVED, "feature",
        "Challenges the fielding side has BEFORE this pitch (0-2).", "See above."),

    # ---------------- opportunities ----------------
    "is_called_pitch": (S_DERIVED, "metadata", "Taken pitch — called strike or ball.", ""),
    "called_strike": (S_DERIVED, "feature", "The umpire called this pitch a strike.", ""),
    "position_player_pitching": (S_DERIVED, "metadata",
                                 "A position player is on the mound.",
                                 "Challenges are not available in these at-bats."),
    "is_challenge_opportunity_batting": (
        S_DERIVED, "metadata",
        "The batting side could have challenged: called strike, a challenge in hand, "
        "no position player pitching.", "The model's denominator for batter behaviour."),
    "is_challenge_opportunity_fielding": (
        S_DERIVED, "metadata", "Same for the fielding side on a called ball.", ""),
    "is_challenge_opportunity": (S_DERIVED, "metadata",
                                 "Either side had an opportunity.",
                                 "Savant reports ~50% of pitches."),
    "is_reasonable_pitch": (S_DERIVED, "metadata",
                            "Opportunity that was either genuinely wrong or within 3 "
                            "inches of the edge.",
                            "A simplified stand-in for Savant's 'reasonable pitch' "
                            "(~5%); theirs also uses RE288 run value."),

    # ---------------- outcomes ----------------
    "was_challenged": (S_API, "target",
                       "A challenge was actually made on this pitch.",
                       "The target for the BEHAVIOUR model (did they challenge?). "
                       "Never a feature."),
    "is_overturned": (S_API, "target",
                      "The challenge succeeded and the call was reversed.",
                      "The target for the OVERTURN model p(s). Null on unchallenged "
                      "pitches by construction, not by data loss."),
    "would_be_overturned": (
        S_DERIVED, "target",
        "Counterfactual: the umpire's call disagrees with the ABS zone, so a "
        "challenge here would have succeeded.",
        "LEAKAGE RISK AS A FEATURE. This is what lets you label unchallenged "
        "opportunities, which is the whole point of the denominator — but it is an "
        "outcome, so it belongs on the left-hand side of the model only."),
    "challenger_role": (S_API, "LEAKAGE RISK",
                        "batter, catcher or pitcher.",
                        "Known only once a challenge happened. Fine for splitting "
                        "results by role; not an input to a model predicting whether "
                        "a challenge occurs."),
    "challenger_id": (S_API, "LEAKAGE RISK", "Challenger's MLBAM id.", "See challenger_role."),
    "challenger_name": (S_API, "identifier", "Challenger's name.", ""),
    "challenging_side": (S_API, "LEAKAGE RISK", "'home' or 'away'.", "See challenger_role."),
    "challenge_call_code": (S_API, "LEAKAGE RISK", "Original call code on a challenged pitch.", ""),
    "challenge_description": (S_API, "LEAKAGE RISK", "Free-text description of the review.", ""),

    # ---------------- context ----------------
    "leverage_index": (S_WP, "feature",
                       "Leverage index entering the plate appearance; 1.0 is average.",
                       "PLATE-APPEARANCE grain, not pitch. A challenge happens mid-PA, "
                       "so this is the state entering the PA."),
    "home_win_prob": (S_WP, "feature", "Home team win probability entering the PA.", ""),
    "away_win_prob": (S_WP, "feature", "Away team win probability entering the PA.", ""),
    "bat_win_prob": (S_WP, "feature", "Win probability for the batting side.",
                     "This is V(s) for the side deciding whether to challenge."),
    "home_win_prob_added": (S_WP, "metadata", "WPA of the completed plate appearance.",
                            "LEAKAGE RISK if used as a feature — it describes how the "
                            "PA ended."),
}

PREFIX_RULES = [
    ("catcher_", S_FRAMING, "feature",
     "Season framing metric for the catcher on this pitch.",
     "Season-level, so it is constant within a catcher-season and partly reflects "
     "the same 2026 challenges you are modelling."),
    ("ump_", S_UMP, "feature",
     "Season accuracy metric for the home-plate umpire.",
     "Derived from the same pitches you are modelling — mild circularity. Consider "
     "leave-one-game-out if it ranks high in Week 8."),
]

ROLE_ORDER = ["target", "LEAKAGE RISK", "feature", "identifier", "metadata", "UNDOCUMENTED"]


def describe(df, col):
    s = df[col]
    n = len(s)
    nn = int(s.notna().sum())
    out = {"column": col, "dtype": str(s.dtype),
           "non_null_pct": round(100 * nn / n, 2) if n else 0.0,
           "n_unique": int(s.nunique(dropna=True))}
    if s.dtype.kind in "ifc" and nn:
        out["min"] = round(float(s.min()), 4)
        out["max"] = round(float(s.max()), 4)
        out["mean"] = round(float(s.mean()), 4)
        out["example"] = ""
    elif s.dtype.kind == "b":
        out["min"] = out["max"] = out["mean"] = ""
        out["example"] = f"True {100*float(s.mean()):.1f}%" if nn else ""
    else:
        out["min"] = out["max"] = out["mean"] = ""
        vals = s.dropna().astype(str).unique()[:3]
        out["example"] = ", ".join(vals)
    return out


def lookup(col):
    if col in REGISTRY:
        return REGISTRY[col]
    for pre, src, role, defn, gotcha in PREFIX_RULES:
        if col.startswith(pre):
            return (src, role, defn, gotcha)
    return (S_STATCAST, "UNDOCUMENTED",
            "Not yet documented — a Statcast column this project does not use.",
            "Look it up in Savant's CSV documentation and add it to REGISTRY in "
            "scripts/09_data_dictionary.py if you start using it.")


def read_any(path):
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    csv = p.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default=str(PROCESSED / "challenge_inventory_2026.parquet"))
    args = ap.parse_args()

    df = read_any(args.table)
    if df is None:
        sys.exit(f"No table at {args.table}. Run 08_build_challenge_inventory.py first.")

    print(f"Documenting {len(df.columns)} columns over {len(df):,} rows")

    rows = []
    for col in df.columns:
        d = describe(df, col)
        src, role, defn, gotcha = lookup(col)
        d.update({"source": src, "modelling_role": role,
                  "definition": defn, "gotcha": gotcha})
        rows.append(d)
    dd = pd.DataFrame(rows)

    DOCS.mkdir(parents=True, exist_ok=True)
    dd.to_csv(DOCS / "data_dictionary.csv", index=False)

    counts = dd["modelling_role"].value_counts()
    undoc = dd[dd["modelling_role"] == "UNDOCUMENTED"]["column"].tolist()
    leaky = dd[dd["modelling_role"] == "LEAKAGE RISK"]["column"].tolist()
    targets = dd[dd["modelling_role"] == "target"]["column"].tolist()
    empty = dd[dd["non_null_pct"] == 0]["column"].tolist()
    sparse = dd[(dd["non_null_pct"] > 0) & (dd["non_null_pct"] < 95)]

    md = []
    md.append("# Data Dictionary — `challenge_inventory_2026`\n")
    md.append(f"Generated {dt.date.today().isoformat()} by "
              f"`scripts/09_data_dictionary.py`. Do not edit by hand — edit the "
              f"REGISTRY in that script and re-run.\n")
    md.append(f"**{len(df):,} rows × {len(df.columns)} columns.** "
              f"One row per pitch, keyed on `game_pk` + `at_bat_number` + "
              f"`pitch_number`.\n")

    md.append("## Modelling roles\n")
    md.append("| Role | Columns | Meaning |")
    md.append("|---|---:|---|")
    meanings = {
        "target": "What you predict. Never an input.",
        "LEAKAGE RISK": "Known only after the challenge resolved. Using these as "
                        "features inflates performance and invalidates the model.",
        "feature": "Safe model input — known at the moment of the decision.",
        "identifier": "Keys and names. For joining and inspection.",
        "metadata": "Provenance, flags and bookkeeping.",
        "UNDOCUMENTED": "Present in the table but not yet defined here.",
    }
    for r in ROLE_ORDER:
        if r in counts:
            md.append(f"| {'**' + r + '**' if r == 'LEAKAGE RISK' else r} "
                      f"| {counts[r]} | {meanings[r]} |")
    md.append("")

    if targets:
        md.append(f"**Targets:** {', '.join('`'+c+'`' for c in targets)}\n")
    if leaky:
        md.append("> **Before Week 7, drop these from any feature matrix:**  \n"
                  "> " + ", ".join(f"`{c}`" for c in leaky) + "\n>\n"
                  "> Each is known only once a challenge has happened and resolved. "
                  "This is the leak your Week 8 notes warn about.\n")

    md.append("## Data quality\n")
    if empty:
        md.append(f"- **{len(empty)} columns are entirely null:** "
                  + ", ".join(f"`{c}`" for c in empty)
                  + ". Decide in Week 6 whether to drop them.\n")
    if len(sparse):
        md.append(f"- **{len(sparse)} columns are under 95% populated.** "
                  "Each needs a Week 6 decision: drop the row, impute, or add a "
                  "missing-indicator.\n")
        md.append("| Column | Non-null % | Source |")
        md.append("|---|---:|---|")
        for r in sparse.sort_values("non_null_pct").itertuples():
            md.append(f"| `{r.column}` | {r.non_null_pct:.1f}% | {r.source} |")
        md.append("")
    if undoc:
        md.append(f"- **{len(undoc)} columns are undocumented** (Statcast columns this "
                  "project doesn't use): " + ", ".join(f"`{c}`" for c in undoc[:25])
                  + ("..." if len(undoc) > 25 else "") + "\n")
    if not empty and not len(sparse) and not undoc:
        md.append("- No empty columns, nothing under 95% populated, nothing "
                  "undocumented.\n")

    for src in dd["source"].drop_duplicates().tolist():
        grp = dd[dd["source"] == src].copy()
        grp["_o"] = grp["modelling_role"].apply(
            lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 99)
        grp = grp.sort_values(["_o", "column"])
        md.append(f"## {src}\n")
        md.append("| Column | Type | Role | Non-null | Range / example | Definition |")
        md.append("|---|---|---|---:|---|---|")
        for r in grp.itertuples():
            if r.min != "" and r.max != "":
                rng = f"{r.min} … {r.max}"
            else:
                rng = str(r.example)
            rng = rng.replace("|", "\\|")[:46]
            defn = r.definition.replace("|", "\\|")
            if r.gotcha:
                defn += f"<br>⚠ {r.gotcha.replace('|', chr(92)+'|')}"
            role = f"**{r.modelling_role}**" if r.modelling_role in (
                "target", "LEAKAGE RISK") else r.modelling_role
            md.append(f"| `{r.column}` | {r.dtype} | {role} | "
                      f"{r.non_null_pct:.0f}% | {rng} | {defn} |")
        md.append("")

    out = DOCS / "DATA_DICTIONARY.md"
    out.write_text("\n".join(md))

    print(f"\n  roles: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if leaky:
        print(f"  LEAKAGE RISK columns flagged: {len(leaky)} — listed in the doc")
    if undoc:
        print(f"  undocumented: {len(undoc)} (add them to REGISTRY if you use them)")
    print(f"\n  {out}")
    print(f"  {DOCS/'data_dictionary.csv'}")


if __name__ == "__main__":
    main()
