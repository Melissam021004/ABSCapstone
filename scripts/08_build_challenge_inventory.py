#!/usr/bin/env python3
"""
08_build_challenge_inventory.py — the analysis-ready table. (Week 5 deliverable.)

Joins every source into one row per called pitch, and derives the two fields
nobody publishes but your MDP cannot work without:

    challenges_remaining_batting / _fielding   the MDP's state variable r
    is_challenge_opportunity_batting / _fielding   the model's denominator

Run order: 01 -> 02 -> 07 -> (04, 05, 06 optional) -> 08

    python 08_build_challenge_inventory.py
    python 08_build_challenge_inventory.py --called-only    # smaller output

Output: data/processed/challenge_inventory_2026.parquet
        data/processed/challenge_events_2026.parquet
        data/processed/qa_report.txt

2026 rules implemented here:
  - each team starts with 2 challenges
  - a SUCCESSFUL challenge is retained; only a FAILED one is consumed
  - in extra innings, a team already at 0 is granted 1 for that inning
  - only the batter, catcher, or pitcher may challenge
  - no challenges while a position player is pitching
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
RAW = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

PLATE_HALF_WIDTH_FT = 8.5 / 12
BALL_RADIUS_FT = (2.9 / 2) / 12
ZONE_TOP_PCT = 0.535
ZONE_BOT_PCT = 0.27
START_CHALLENGES = 2

CALLED = ["called_strike", "ball", "blocked_ball"]

qa = []


def note(msg, ok=None):
    prefix = "" if ok is None else ("  [ok]    " if ok else "  [CHECK] ")
    line = f"{prefix}{msg}"
    print(line)
    qa.append(line)


# ---------------------------------------------------------------- zone math
def zone_distance_ft(plate_x, plate_z, top, bot):
    """Signed distance from the ball's surface to the ABS rectangle, in feet.
    Negative => some part of the ball overlaps the zone => ABS strike."""
    plate_x = np.asarray(plate_x, float)
    plate_z = np.asarray(plate_z, float)
    top = np.asarray(top, float)
    bot = np.asarray(bot, float)

    dx = np.abs(plate_x) - PLATE_HALF_WIDTH_FT
    dz = np.maximum(bot - plate_z, plate_z - top)

    inside = (dx <= 0) & (dz <= 0)
    corner = (dx > 0) & (dz > 0)
    d = np.where(inside, np.maximum(dx, dz),
                 np.where(corner,
                          np.hypot(np.maximum(dx, 0), np.maximum(dz, 0)),
                          np.maximum(dx, dz)))
    return d - BALL_RADIUS_FT


# ---------------------------------------------------------------- loading
def load_statcast():
    files = sorted((RAW / "statcast").glob("statcast_2026_*.parquet"))
    if not files:
        sys.exit(f"No Statcast files in {RAW/'statcast'}. Run script 01.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    note(f"Statcast: {len(df):,} pitches, {df['game_pk'].nunique():,} games")
    return df


def load_challenges():
    f = RAW / "challenges" / "abs_challenges_2026.parquet"
    if not f.exists():
        sys.exit(f"No challenge file at {f}. Run script 02.")
    df = pd.read_parquet(f)
    note(f"Challenges: {len(df):,} events, {df['game_pk'].nunique():,} games")
    return df


def maybe(path, label):
    if Path(path).exists():
        df = pd.read_parquet(path)
        note(f"{label}: {len(df):,} rows")
        return df
    note(f"{label}: not found at {path} — skipping", ok=False)
    return None


# ---------------------------------------------------------------- core logic
def derive_challenge_state(pitches, challenges):
    """
    Walk each game in order and carry the challenge inventory forward.

    The state attached to a pitch is the state BEFORE that pitch is decided,
    which is the state the challenge decision is actually made under.
    """
    # Statcast at_bat_number is 1-based; Stats API at_bat_index is 0-based.
    ch = challenges.copy()
    ch["at_bat_number"] = ch["at_bat_index"] + 1
    keep = ["game_pk", "at_bat_number", "pitch_number", "challenger_role",
            "challenger_id", "challenger_name", "challenging_side",
            "is_overturned", "call_code", "description", "hp_umpire",
            "hp_umpire_id"]
    ch = ch[[c for c in keep if c in ch.columns]].copy()

    # Statcast ALSO has a `description` column (the pitch result). Merging without
    # renaming silently produces description_x / description_y and breaks every
    # downstream reference. Same story for call_code.
    ch = ch.rename(columns={"description": "challenge_description",
                            "call_code": "challenge_call_code"})

    key = ["game_pk", "at_bat_number", "pitch_number"]
    dupes = ch.duplicated(subset=key).sum()
    if dupes:
        note(f"{dupes} duplicate challenge keys — keeping the first of each", ok=False)
        ch = ch.drop_duplicates(subset=key, keep="first")
    ch["was_challenged"] = True

    overlap = (set(ch.columns) & set(pitches.columns)) - set(key)
    if overlap:
        note(f"Column name collision on merge: {sorted(overlap)} — suffixing "
             f"the challenge copy with '_chal'", ok=False)

    df = pitches.merge(ch, on=key, how="left", validate="one_to_one",
                       suffixes=("", "_chal"))
    df["was_challenged"] = df["was_challenged"].fillna(False).astype(bool)
    df = df.reset_index(drop=True)

    matched = int(df["was_challenged"].sum())
    rate = matched / max(len(challenges), 1)
    note(f"Challenge->pitch join: {matched:,}/{len(challenges):,} matched ({rate:.1%})",
         ok=rate >= 0.95)
    if rate < 0.95:
        note("    Unmatched challenges usually mean (a) the at_bat_number "
             "off-by-one is wrong, (b) pitchNumber counts pickoffs/timeouts, or "
             "(c) the game is in one source but not the other. Fix before modelling.",
             ok=False)

    # batting side per pitch
    df["batting_side"] = np.where(df["inning_topbot"].str.lower().str.startswith("top"),
                                  "away", "home")
    df["fielding_side"] = np.where(df["batting_side"] == "away", "home", "away")

    # Walk pitches in true game order. `order` holds positional indices.
    order = np.lexsort((df["pitch_number"].values,
                        df["at_bat_number"].values,
                        df["game_pk"].values))

    rem_bat = np.full(len(df), START_CHALLENGES, dtype=np.int8)
    rem_fld = np.full(len(df), START_CHALLENGES, dtype=np.int8)

    g_pk = df["game_pk"].values
    innings = df["inning"].values
    bat_side = df["batting_side"].values
    challenged = df["was_challenged"].values
    overturned = df["is_overturned"].values if "is_overturned" in df.columns \
        else np.full(len(df), None, dtype=object)
    chal_side = df["challenging_side"].values if "challenging_side" in df.columns \
        else np.full(len(df), None, dtype=object)

    cur_game = None
    rem = {"home": START_CHALLENGES, "away": START_CHALLENGES}
    last_inning = None

    for i in order:
        pk, inn, bs = g_pk[i], innings[i], bat_side[i]

        if pk != cur_game:
            cur_game = pk
            rem = {"home": START_CHALLENGES, "away": START_CHALLENGES}
            last_inning = inn

        # Extra innings: a side already at zero is granted one for that inning.
        if inn != last_inning:
            if inn and inn > 9:
                for side in ("home", "away"):
                    if rem[side] == 0:
                        rem[side] = 1
            last_inning = inn

        fs = "home" if bs == "away" else "away"
        rem_bat[i] = rem[bs]
        rem_fld[i] = rem[fs]

        # Apply the outcome AFTER recording the pre-pitch state: a failed
        # challenge costs one, a successful challenge is retained.
        if challenged[i]:
            side = chal_side[i]
            ov = overturned[i]
            if side in ("home", "away") and not pd.isna(ov) and not bool(ov):
                rem[side] = max(rem[side] - 1, 0)

    df["challenges_remaining_batting"] = rem_bat
    df["challenges_remaining_fielding"] = rem_fld

    bad = ((df["challenges_remaining_batting"] < 0) |
           (df["challenges_remaining_batting"] > START_CHALLENGES)) & (df["inning"] <= 9)
    note(f"Challenges-remaining in [0,{START_CHALLENGES}] during regulation: "
         f"{(~bad).mean():.4%} of pitches", ok=not bad.any())
    return df


def add_abs_zone(df, bios):
    """Attach the true ABS zone and distance-from-edge."""
    if bios is not None and {"player_id", "height_ft"} <= set(bios.columns):
        h = bios[["player_id", "height_ft"]].rename(
            columns={"player_id": "batter", "height_ft": "batter_height_ft"})
        df = df.merge(h, on="batter", how="left")
        cover = df["batter_height_ft"].notna().mean()
        note(f"Batter heights matched: {cover:.1%}", ok=cover > 0.95)
    else:
        df["batter_height_ft"] = np.nan
        note("No batter heights — falling back to Statcast's zone.", ok=False)

    use_h = df["batter_height_ft"].notna()
    df["abs_zone_top"] = np.where(use_h, df["batter_height_ft"] * ZONE_TOP_PCT, df["sz_top"])
    df["abs_zone_bot"] = np.where(use_h, df["batter_height_ft"] * ZONE_BOT_PCT, df["sz_bot"])
    df["abs_zone_source"] = np.where(use_h, "batter_height", "statcast_sz")

    df["zone_dist_ft"] = zone_distance_ft(df["plate_x"], df["plate_z"],
                                          df["abs_zone_top"], df["abs_zone_bot"])
    df["zone_dist_in"] = df["zone_dist_ft"] * 12
    df["abs_strike"] = df["zone_dist_ft"] < 0

    # Keep Statcast's zone alongside it as a robustness check (Week 6 figure).
    df["sc_zone_dist_in"] = zone_distance_ft(
        df["plate_x"], df["plate_z"], df["sz_top"], df["sz_bot"]) * 12
    df["zone_defs_disagree"] = df["abs_strike"] != (df["sc_zone_dist_in"] < 0)

    note(f"ABS zone vs Statcast zone disagree on "
         f"{df['zone_defs_disagree'].mean():.2%} of called pitches "
         f"(this is where challenges live)")
    return df


def flag_opportunities(df, bios):
    """Savant's challenge-opportunity definition, applied per side."""
    df["is_called_pitch"] = df["description"].isin(CALLED)
    df["called_strike"] = df["description"] == "called_strike"

    # No challenges while a position player is pitching.
    if bios is not None and "primary_position" in bios.columns:
        pos = bios[["player_id", "primary_position"]].rename(
            columns={"player_id": "pitcher", "primary_position": "pitcher_pos"})
        df = df.merge(pos, on="pitcher", how="left")
        df["position_player_pitching"] = (
            df["pitcher_pos"].notna() & (df["pitcher_pos"] != "P"))
    else:
        df["position_player_pitching"] = False
        note("No primary-position data — cannot exclude position-player pitching.",
             ok=False)

    eligible = df["is_called_pitch"] & ~df["position_player_pitching"]

    # The call goes against the batter when it's a strike; against the
    # pitcher/catcher when it's a ball.
    df["is_challenge_opportunity_batting"] = (
        eligible & df["called_strike"] & (df["challenges_remaining_batting"] > 0))
    df["is_challenge_opportunity_fielding"] = (
        eligible & ~df["called_strike"] & (df["challenges_remaining_fielding"] > 0))
    df["is_challenge_opportunity"] = (df["is_challenge_opportunity_batting"] |
                                      df["is_challenge_opportunity_fielding"])

    # "Would the call be overturned?" — the counterfactual label for every
    # opportunity, challenged or not. This is what lets you model unchallenged
    # pitches, which is the whole point of the opportunity denominator.
    df["would_be_overturned"] = df["is_called_pitch"] & (df["abs_strike"] != df["called_strike"])

    # A cheap stand-in for Savant's "reasonable pitch". Their real definition
    # also uses RE288 run value; refine this in Week 6 if you need to match them.
    df["is_reasonable_pitch"] = (
        df["is_challenge_opportunity"] &
        (df["would_be_overturned"] | (df["zone_dist_in"].abs() <= 3.0)))

    share = df["is_challenge_opportunity"].mean()
    note(f"Challenge opportunities: {df['is_challenge_opportunity'].sum():,} "
         f"({share:.1%} of pitches) — Savant reports ~50%",
         ok=0.35 <= share <= 0.65)
    r_share = df["is_reasonable_pitch"].mean()
    note(f"'Reasonable' pitches: {df['is_reasonable_pitch'].sum():,} "
         f"({r_share:.1%}) — Savant reports ~5%", ok=0.02 <= r_share <= 0.12)
    return df


def attach_context(df, winprob, framing, umpires):
    if winprob is not None:
        wp = winprob.copy()
        wp["at_bat_number"] = wp["at_bat_index"] + 1
        cols = ["game_pk", "at_bat_number", "home_win_prob", "away_win_prob",
                "bat_win_prob", "home_win_prob_added", "leverage_index"]
        wp = wp[[c for c in cols if c in wp.columns]].drop_duplicates(
            subset=["game_pk", "at_bat_number"])
        df = df.merge(wp, on=["game_pk", "at_bat_number"], how="left")
        note(f"Win probability matched: {df['leverage_index'].notna().mean():.1%} "
             f"of pitches")

    if framing is not None:
        idc = next((c for c in ("player_id", "catcher", "entity_id", "id")
                    if c in framing.columns), None)
        if idc:
            keep = [idc] + [c for c in framing.columns
                            if "runs" in c.lower() or "strike_rate" in c.lower()
                            or "rv" in c.lower()][:6]
            fr = framing[keep].rename(columns={idc: "fielder_2"})
            fr.columns = ["fielder_2"] + [f"catcher_{c}" for c in fr.columns[1:]]
            fr = fr.drop_duplicates(subset=["fielder_2"])
            df = df.merge(fr, on="fielder_2", how="left")
            note("Catcher framing metrics attached")
        else:
            note("Framing file has no joinable player id — skipping", ok=False)

    if umpires is not None and "hp_umpire_id" in umpires.columns:
        keep = ["hp_umpire_id", "accuracy", "borderline_accuracy",
                "missed_strike_rate", "missed_ball_rate"]
        u = umpires[[c for c in keep if c in umpires.columns]].copy()
        u.columns = ["hp_umpire_id"] + [f"ump_{c}" for c in u.columns[1:]]
        # Use the umpire id from the game index, not the challenge join (which
        # is only populated on challenged pitches).
        if "hp_umpire_id" not in df.columns or df["hp_umpire_id"].isna().all():
            gi = RAW / "challenges" / "games_2026.parquet"
            if gi.exists():
                g = pd.read_parquet(gi)[["game_pk", "hp_umpire", "hp_umpire_id"]]
                df = df.drop(columns=[c for c in ("hp_umpire", "hp_umpire_id")
                                      if c in df.columns]).merge(g, on="game_pk", how="left")
        df = df.merge(u, on="hp_umpire_id", how="left")
        note("Umpire accuracy attached")
    return df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--called-only", action="store_true",
                    help="keep only called pitches (much smaller file)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING THE ANALYSIS-READY CHALLENGE INVENTORY")
    print("=" * 70 + "\n1. LOADING")

    pitches = load_statcast()
    challenges = load_challenges()
    bios = maybe(RAW / "players" / "player_bios_2026.parquet", "Player bios")
    winprob = maybe(RAW / "winprob" / "winprob_2026.parquet", "Win probability")
    framing = maybe(RAW / "savant" / "catcher_framing_all_years.parquet", "Catcher framing")
    umpires = maybe(RAW / "umpires" / "umpire_accuracy_derived_2026.parquet",
                    "Umpire accuracy")

    print("\n2. DERIVING CHALLENGE STATE")
    df = derive_challenge_state(pitches, challenges)

    print("\n3. ABS ZONE")
    df = add_abs_zone(df, bios)

    print("\n4. CHALLENGE OPPORTUNITIES")
    df = flag_opportunities(df, bios)

    print("\n5. CONTEXT")
    df = attach_context(df, winprob, framing, umpires)

    if args.called_only:
        before = len(df)
        df = df[df["is_called_pitch"]].reset_index(drop=True)
        note(f"Filtered to called pitches: {len(df):,} (from {before:,})")

    print("\n6. QA")
    n_ch = int(df["was_challenged"].sum())
    ovr = df.loc[df["was_challenged"], "is_overturned"]
    note(f"Challenged pitches in the table: {n_ch:,}")
    if len(ovr.dropna()):
        note(f"Overturn rate: {ovr.mean():.1%}", ok=0.30 <= ovr.mean() <= 0.70)

    by_role = df[df["was_challenged"]].groupby("challenger_role")["is_overturned"] \
                .agg(["size", "mean"])
    print("\n  Overturn rate by challenger role:")
    print(by_role.to_string())
    qa.append(by_role.to_string())

    if len(ovr.dropna()) and "catcher" in by_role.index and "batter" in by_role.index:
        note(f"Catchers beat batters: "
             f"{by_role.loc['catcher','mean']:.1%} vs {by_role.loc['batter','mean']:.1%}",
             ok=by_role.loc["catcher", "mean"] > by_role.loc["batter", "mean"])
        qa.append("    (MLB has publicly said catchers are the better challengers; "
                  "if your data disagrees, suspect your data.)")

    ch_rows = df[df["was_challenged"]]
    if len(ch_rows):
        acc = (ch_rows["would_be_overturned"] == ch_rows["is_overturned"]).mean()
        note(f"Your ABS zone predicts the actual challenge outcome "
             f"{acc:.1%} of the time", ok=acc >= 0.85)
        qa.append("    This is the single best test of your zone implementation. "
                  "Below ~85% means the zone constants or the ball-radius rule "
                  "are off, and every distance feature downstream is biased.")

    # Challenge rate on opportunities, by challenges remaining — the behavioural
    # signature the MDP is supposed to explain.
    opp = df[df["is_challenge_opportunity"]]
    if len(opp):
        tbl = opp.groupby("challenges_remaining_batting").agg(
            opportunities=("was_challenged", "size"),
            challenge_rate=("was_challenged", "mean")).round(5)
        print("\n  Challenge rate by challenges remaining (batting side):")
        print(tbl.to_string())
        qa.append(tbl.to_string())

    out = OUT_DIR / "challenge_inventory_2026.parquet"
    df.to_parquet(out, index=False)
    ch_out = OUT_DIR / "challenge_events_2026.parquet"
    df[df["was_challenged"]].to_parquet(ch_out, index=False)

    (OUT_DIR / "qa_report.txt").write_text("\n".join(qa))

    print(f"\n{'=' * 70}")
    print(f"Saved {len(df):,} rows x {len(df.columns)} cols -> {out}")
    print(f"Saved {n_ch:,} challenge events -> {ch_out}")
    print(f"QA report -> {OUT_DIR/'qa_report.txt'}")
    print("""
BEFORE YOU MODEL: hand-verify challenges_remaining on three real games.
Open the play-by-play on MLB.com, find each challenge, and confirm the
inventory in this table matches. It is derived, not published, and it is the
state variable your entire MDP rests on. Twenty minutes here saves November.""")
    print("=" * 70)


if __name__ == "__main__":
    main()
