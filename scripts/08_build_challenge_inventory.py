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

# What --spine-from-table drops from the existing table and rebuilds. The
# challenge columns and everything computed from kept inputs are always rebuilt.
# Columns that come from an optional raw file are rebuilt only when that file is
# there; otherwise the existing table's values are KEPT (a raw folder that has
# lost its player bios must not cost the table its batter heights).
REBUILT_ALWAYS = {
    "challenger_role", "challenger_id", "challenger_name", "challenging_side", "is_overturned",
    "challenge_call_code", "challenge_description", "was_challenged", "challenge_rekeyed",
    "batting_side", "fielding_side", "challenges_remaining_batting",
    "challenges_remaining_fielding", "abs_zone_top", "abs_zone_bot", "abs_zone_source",
    "zone_dist_ft", "zone_dist_in", "abs_strike", "sc_zone_dist_in", "zone_defs_disagree",
    "is_called_pitch", "called_strike", "would_be_overturned", "is_reasonable_pitch"}
REBUILT_ALWAYS_PREFIXES = ("is_challenge_opportunity",)
FROM_SOURCE = {                     # source -> (columns, column prefixes)
    "heights": ({"batter_height_ft"}, ()),
    "positions": ({"pitcher_pos", "position_player_pitching"}, ()),
    "winprob": ({"home_win_prob", "away_win_prob", "bat_win_prob", "home_win_prob_added",
                 "leverage_index"}, ()),
    "framing": (set(), ("catcher_",)),
    "umpires": (set(), ("ump_",)),
    "games": ({"hp_umpire", "hp_umpire_id"}, ()),
}

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
# Row-count ledger: Week 5 asks for row counts at each join step, and a silent
# row-count change across a join is the classic way a merge goes wrong.
ledger = []


def step(label, n_rows, n_cols=None):
    ledger.append({"step": label, "rows": int(n_rows),
                   "cols": int(n_cols) if n_cols is not None else None})


def read_any(path):
    """Read .parquet, or the .csv sibling when pyarrow isn't installed."""
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    csv = p.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    return None


def write_any(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def load_statcast():
    d = RAW / "statcast"
    files = sorted(d.glob("statcast_2026_*.parquet")) or sorted(d.glob("statcast_2026_*.csv"))
    if not files:
        sys.exit(f"No Statcast files in {d}. Run script 01 first.")
    frames = []
    for f in files:
        got = read_any(f)
        if got is not None:
            frames.append(got)
    if not frames:
        sys.exit(f"Statcast files in {d} could not be read (pyarrow missing?).")
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    note(f"Statcast: {len(df):,} pitches, {df['game_pk'].nunique():,} games")
    step("1. Statcast pitch spine", len(df), len(df.columns))
    return df


BOOL = {True: True, False: False, "True": True, "False": False, 1: True, 0: False,
        1.0: True, 0.0: False}


def zone_agreement(ch):
    """
    How often the reconstructed ABS zone reproduces what ABS ruled on challenged
    pitches. 2026 Statcast records the FINAL call on a challenged pitch (found in
    Weeks 8–9), so a zone check against the recorded call scores every overturned
    challenge as a miss. The ORIGINAL call follows from who challenged: the batting
    side only challenges strikes, the fielding side only balls.
    Returns (agreement, share of overturned challenges whose recorded call is still
    the original, rows used).
    """
    d = ch[ch["is_overturned"].notna()]
    orig = pd.Series(np.nan, index=d.index, dtype=object)
    if {"challenging_side", "batting_side"} <= set(d.columns):
        cs = d["challenging_side"].astype(str).str.lower()
        bs = d["batting_side"].astype(str).str.lower()
        known = cs.isin(["home", "away"]) & bs.isin(["home", "away"])
        orig[known] = (cs[known] == bs[known])
    if "challenger_role" in d.columns:
        role = d["challenger_role"].astype(str).str.lower()
        fill = orig.isna() & role.isin(["batter", "catcher", "pitcher"])
        orig[fill] = role[fill].eq("batter")
    d, orig = d[orig.notna()], orig[orig.notna()].astype(bool)
    if not len(d):
        return np.nan, np.nan, 0
    ov = d["is_overturned"].map(BOOL).fillna(False).astype(bool)
    agree = float(((d["abs_strike"].map(BOOL).fillna(False).astype(bool) != orig) == ov).mean())
    rec = d["called_strike"].map(BOOL).fillna(False).astype(bool)
    rec_orig = float((rec[ov] == orig[ov]).mean()) if ov.any() else np.nan
    return agree, rec_orig, len(d)


def table_games(path):
    """How many games the existing analysis-ready table covers (None if there isn't one)."""
    path = Path(path)
    try:
        if path.exists():
            return int(pd.read_parquet(path, columns=["game_pk"])["game_pk"].nunique())
        csv = path.with_suffix(".csv")
        if csv.exists():
            return int(pd.read_csv(csv, usecols=["game_pk"])["game_pk"].nunique())
    except Exception:
        return None
    return None


def rebuild_sources(bios, winprob, framing, umpires):
    """Which optional raw files are here to rebuild their columns from."""
    return {"heights": bios is not None and {"player_id", "height_ft"} <= set(bios.columns),
            "positions": bios is not None and {"player_id", "primary_position"} <= set(bios.columns),
            "winprob": winprob is not None,
            "framing": framing is not None,
            "umpires": umpires is not None and "hp_umpire_id" in umpires.columns,
            "games": any((RAW / "challenges" / f"games_2026{e}").exists()
                         for e in (".parquet", ".csv"))}


def backup_of(path):
    """The backup a --spine-from-table run keeps (challenge_inventory_2026_prev.*)."""
    for ext in (".parquet", ".csv"):
        p = path.with_name(Path(path).stem + "_prev" + ext)
        if p.exists():
            return p
    return None


def _columns_of(path):
    """Column names without reading the data (pyarrow's schema for parquet)."""
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            return list(pq.read_schema(path).names)
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _coverage(path, col):
    try:
        s = (pd.read_parquet(path, columns=[col]) if path.suffix == ".parquet"
             else pd.read_csv(path, usecols=[col]))[col]
        return float(s.notna().mean())
    except Exception:
        return None


def stop_if_table_lost_columns(prev, current, sources):
    """
    A rebuild that ran without the raw files some columns come from (before this
    was fixed) leaves a table missing them — and rebuilding from THAT table would
    carry the loss forward. If the backup still has them, stop and say how to restore.
    """
    bak = backup_of(Path(current))
    if bak is None:
        return
    bcols = _columns_of(bak)
    lost = [c for c in bcols if c not in prev.columns]
    h_now = float(prev["batter_height_ft"].notna().mean()) if "batter_height_ft" in prev.columns else 0.0
    h_bak = _coverage(bak, "batter_height_ft") if "batter_height_ft" in bcols else None
    no_heights = (not sources["heights"]) and h_bak is not None and h_bak > 0.95 and h_now < 0.5
    if lost or no_heights:
        what = ", ".join(lost[:8]) + (" …" if len(lost) > 8 else "")
        if no_heights:
            what = (what + "; " if what else "") + f"batter heights ({h_now:.0%} filled vs {h_bak:.0%})"
        sys.exit(f"""
STOPPED before rebuilding anything.
The table at {current} has lost columns that its backup still has:
  {what}
That happens when an earlier rebuild ran without the raw files those columns come
from. Restore the backup, then run this again — it now keeps those columns:

    cp {bak} {current}
""")


def load_spine_from_table(sources):
    """
    The pitch spine from the EXISTING analysis-ready table instead of the raw
    Statcast files. It drops the challenge columns and everything recomputed from
    kept inputs, and rebuilds them (steps 2–6). Columns from an optional raw file
    (batter heights, positions, win probability, framing, umpire accuracy) are
    rebuilt when the file is there and KEPT from the existing table when it isn't.
    This re-joins corrected challenge records (script 02, November 2026) without
    re-pulling a season of Statcast. The old table is copied to
    challenge_inventory_2026_prev.* before it's replaced.
    """
    path = OUT_DIR / "challenge_inventory_2026.parquet"
    prev = read_any(path)
    if prev is None:
        sys.exit(f"No existing table at {path} to reuse. Run without --spine-from-table.")
    current = path if path.exists() else path.with_suffix(".csv")
    stop_if_table_lost_columns(prev, current, sources)
    drop, kept = [], []
    for c in prev.columns:
        if c in REBUILT_ALWAYS or c.startswith(REBUILT_ALWAYS_PREFIXES):
            drop.append(c)
            continue
        for src, (cols, prefixes) in FROM_SOURCE.items():
            if c in cols or (prefixes and c.startswith(prefixes)):
                (drop if sources[src] else kept).append(c)
                break
    df = prev.drop(columns=drop)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    note(f"Pitch spine from the existing table: {len(df):,} pitches, "
         f"{df['game_pk'].nunique():,} games ({len(drop)} columns dropped to rebuild)")
    if kept:
        note(f"Kept from the existing table because their raw files aren't here: "
             f"{', '.join(kept)}")
    step("1. Statcast pitch spine (from the existing table)", len(df), len(df.columns))
    ledger.insert(0, ledger.pop())          # keep step 1 first in the join ledger
    return df


def rekey_by_count(ch, pitches):
    """
    Statcast and the Stats API number the pitches of an at-bat the same way almost
    always. Where they don't (an at-bat with an automatic ball or strike, say), a
    challenge joined on pitch_number lands on the wrong pitch — sometimes a swing.
    The count before a CALLED pitch can't repeat within an at-bat, so it's a second
    key: when the pitch a challenge joins to isn't a called pitch at the count the
    Stats API recorded, move the challenge to the called pitch in that at-bat that
    is. Returns the challenges (with `challenge_rekeyed`) and a summary.
    """
    out = ch.reset_index(drop=True).copy()
    out["challenge_rekeyed"] = False
    if not {"balls_before", "strikes_before"} <= set(out.columns):
        return out, None
    key = ["game_pk", "at_bat_number", "pitch_number"]
    p = pitches[key + ["balls", "strikes"]].copy()
    p["_called"] = pitches["description"].isin(CALLED).to_numpy()
    for c in key + ["balls", "strikes"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
    c = out[key + ["balls_before", "strikes_before"]].apply(pd.to_numeric, errors="coerce").astype(float)
    t = c.merge(p, on=key, how="left")
    ok = (t["_called"].eq(True) & t["balls"].eq(t["balls_before"]) &
          t["strikes"].eq(t["strikes_before"])).to_numpy()
    on_swing = (t["_called"].eq(False)).to_numpy()
    called = p[p["_called"]].drop_duplicates(["game_pk", "at_bat_number", "balls", "strikes"],
                                             keep=False)
    alt = c[~ok].reset_index().merge(
        called.rename(columns={"balls": "balls_before", "strikes": "strikes_before",
                               "pitch_number": "_alt"})[
            ["game_pk", "at_bat_number", "balls_before", "strikes_before", "_alt"]],
        on=["game_pk", "at_bat_number", "balls_before", "strikes_before"], how="left")
    alt = alt[alt["_alt"].notna()].set_index("index")["_alt"]
    taken = set(map(tuple, c[ok][key].to_numpy()))
    move = [i for i, a in alt.items()
            if (c.at[i, "game_pk"], c.at[i, "at_bat_number"], a) not in taken]
    dtype = out["pitch_number"].dtype
    for i in move:
        out.at[i, "pitch_number"] = alt[i]
    try:
        out["pitch_number"] = out["pitch_number"].astype(dtype)
    except (TypeError, ValueError):
        pass
    out.loc[move, "challenge_rekeyed"] = True
    summary = {"checked": len(out), "ok": int(ok.sum()), "moved": len(move),
               "moved_from_swing": int(on_swing[move].sum()) if move else 0,
               "left": int((~ok).sum()) - len(move)}
    return out, summary


def load_challenges():
    df = read_any(RAW / "challenges" / "abs_challenges_2026.parquet")
    if df is None:
        sys.exit(f"No challenge file in {RAW/'challenges'}. Run script 02 first.")
    note(f"Challenges: {len(df):,} events, {df['game_pk'].nunique():,} games")
    step("   challenge events (input)", len(df), len(df.columns))
    return df


def maybe(path, label):
    df = read_any(path)
    if df is not None:
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
            "hp_umpire_id", "balls_before", "strikes_before"]
    ch = ch[[c for c in keep if c in ch.columns]].copy()

    # Check each challenge lands on a called pitch at the count the Stats API
    # recorded; move the few that don't (see rekey_by_count).
    ch, rk = rekey_by_count(ch, pitches)
    if rk is not None:
        note(f"Pitch numbers checked against the count: {rk['ok']:,} of {rk['checked']:,} "
             f"challenges join to a called pitch at the Stats API's count; {rk['moved']:,} "
             f"moved to the called pitch with that count in the same at-bat "
             f"({rk['moved_from_swing']:,} of them had landed on a swing); {rk['left']:,} left "
             f"as they were", ok=rk["left"] <= 0.005 * rk["checked"])
    ch = ch.drop(columns=[c for c in ("balls_before", "strikes_before") if c in ch.columns])
    # A spine kept from the existing table may already carry the plate umpire
    # (when there's no game index to re-attach it from); don't add a second copy.
    ch = ch.drop(columns=[c for c in ("hp_umpire", "hp_umpire_id") if c in pitches.columns])

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
    if "challenge_rekeyed" in df.columns:
        df["challenge_rekeyed"] = df["challenge_rekeyed"].eq(True)
    df = df.reset_index(drop=True)

    matched = int(df["was_challenged"].sum())
    rate = matched / max(len(challenges), 1)

    # Split the misses into the two very different reasons they happen. A
    # challenge with no pitch number cannot be joined by ANY method — the Stats
    # API logged the review as its own event rather than on the pitch. Counting
    # those as join failures makes a healthy pipeline look broken.
    key_cols = [c for c in ("at_bat_number", "pitch_number") if c in challenges.columns
                or c == "at_bat_number"]
    src = challenges.copy()
    src["at_bat_number"] = src["at_bat_index"] + 1
    no_key = src[["at_bat_number", "pitch_number"]].isna().any(axis=1).sum()
    joinable = len(challenges) - int(no_key)
    adj_rate = matched / max(joinable, 1)

    note(f"Challenge->pitch join: {matched:,}/{len(challenges):,} matched ({rate:.1%})",
         ok=adj_rate >= 0.95)
    if no_key:
        note(f"    {int(no_key):,} challenge(s) carry no pitch number or at-bat "
             f"index, so they can't be joined. Excluding those, the "
             f"join rate is {adj_rate:.1%} of {joinable:,} joinable events. Re-run "
             f"script 02 with --force: since November 2026 it ties a review recorded "
             f"on its own event to the pitch it refers to.")
    if adj_rate < 0.95:
        note("    Among events that DO have keys, misses usually mean (a) the "
             "at_bat_number off-by-one is wrong, (b) pitchNumber counts "
             "pickoffs/timeouts, or (c) the game is in one source but not the "
             "other. Fix before modelling.", ok=False)

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
    step("2. + challenge outcomes & inventory", len(df), len(df.columns))

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
        df = df.drop(columns=["batter_height_ft"], errors="ignore").merge(h, on="batter", how="left")
        cover = df["batter_height_ft"].notna().mean()
        note(f"Batter heights matched: {cover:.1%}", ok=cover > 0.95)
    elif "batter_height_ft" in df.columns and df["batter_height_ft"].notna().any():
        # --spine-from-table with no bios file: the existing table's heights.
        cover = df["batter_height_ft"].notna().mean()
        note(f"Batter heights kept from the existing table: {cover:.1%} of pitches",
             ok=cover > 0.95)
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
    step("3. + ABS zone & distance", len(df), len(df.columns))
    return df


def flag_opportunities(df, bios):
    """Savant's challenge-opportunity definition, applied per side."""
    df["is_called_pitch"] = df["description"].isin(CALLED)
    df["called_strike"] = df["description"] == "called_strike"

    # No challenges while a position player is pitching.
    if bios is not None and "primary_position" in bios.columns:
        pos = bios[["player_id", "primary_position"]].rename(
            columns={"player_id": "pitcher", "primary_position": "pitcher_pos"})
        df = df.drop(columns=["pitcher_pos", "position_player_pitching"], errors="ignore")
        df = df.merge(pos, on="pitcher", how="left")
        df["position_player_pitching"] = (
            df["pitcher_pos"].notna() & (df["pitcher_pos"] != "P"))
    elif "position_player_pitching" in df.columns:
        # --spine-from-table with no bios file: the existing table's flags.
        m = {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
        df["position_player_pitching"] = df["position_player_pitching"].map(m).fillna(False).astype(bool)
        note(f"Position-player pitching kept from the existing table: "
             f"{int(df['position_player_pitching'].sum()):,} pitches")
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

    # Stand-in for Savant's "reasonable pitch". Theirs is: the call would be
    # overturned, OR it was correct but within 3 inches of the edge AND the
    # situation carries run value >= +0.30 on the RE288 scale, OR challenge
    # probability >= 20%.
    #
    # We have no RE288 table yet, so the run-value condition is proxied by the
    # count: a ball/strike matters most when it ends the plate appearance, i.e.
    # two strikes or three balls. Without that condition the flag lands near 20%
    # of pitches instead of Savant's ~5% — QA check REASONABLE_SHARE catches it.
    # Replace this with a real RE288 lookup in Week 10, when you build the
    # empirical run-expectancy tables anyway.
    high_value_count = (df["strikes"] >= 2) | (df["balls"] >= 3)
    df["is_reasonable_pitch"] = (
        df["is_challenge_opportunity"] &
        (df["would_be_overturned"] |
         ((df["zone_dist_in"].abs() <= 3.0) & high_value_count)))

    share = df["is_challenge_opportunity"].mean()
    note(f"Challenge opportunities: {df['is_challenge_opportunity'].sum():,} "
         f"({share:.1%} of pitches) — Savant reports ~50%",
         ok=0.35 <= share <= 0.65)
    r_share = df["is_reasonable_pitch"].mean()
    note(f"'Reasonable' pitches: {df['is_reasonable_pitch'].sum():,} "
         f"({r_share:.1%}) — Savant reports ~5%", ok=0.02 <= r_share <= 0.12)
    step("4. + opportunity flags", len(df), len(df.columns))
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

    # The home-plate umpire is a GAME-level attribute, so it must come from the
    # game index and apply to every pitch. The challenge merge also carries an
    # hp_umpire column, but only on challenged pitches (~2% of rows) — taking
    # that one would leave the umpire covariate almost entirely null, and the
    # model would quietly lose a variable the proposal commits to.
    games = maybe(RAW / "challenges" / "games_2026.parquet", "Game index")
    if games is not None and "hp_umpire_id" in games.columns:
        g = games[[c for c in ("game_pk", "hp_umpire", "hp_umpire_id")
                   if c in games.columns]].drop_duplicates(subset=["game_pk"])
        df = df.drop(columns=[c for c in ("hp_umpire", "hp_umpire_id")
                              if c in df.columns])
        df = df.merge(g, on="game_pk", how="left")
        note(f"Home-plate umpire attached to "
             f"{df['hp_umpire_id'].notna().mean():.1%} of pitches",
             ok=df["hp_umpire_id"].notna().mean() > 0.95)
    else:
        note("No game index — umpire covariate unavailable", ok=False)

    if umpires is not None and "hp_umpire_id" in umpires.columns \
            and "hp_umpire_id" in df.columns:
        keep = ["hp_umpire_id", "accuracy", "borderline_accuracy",
                "missed_strike_rate", "missed_ball_rate"]
        u = umpires[[c for c in keep if c in umpires.columns]].copy()
        u.columns = ["hp_umpire_id"] + [f"ump_{c}" for c in u.columns[1:]]
        u = u.drop_duplicates(subset=["hp_umpire_id"])
        df = df.merge(u, on="hp_umpire_id", how="left")
        note(f"Umpire accuracy matched: "
             f"{df['ump_accuracy'].notna().mean():.1%} of pitches",
             ok=df["ump_accuracy"].notna().mean() > 0.95)
    step("5. + win prob, framing, umpire", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--called-only", action="store_true",
                    help="keep only called pitches (much smaller file)")
    ap.add_argument("--spine-from-table", action="store_true",
                    help="take the pitches from the existing table instead of the raw "
                         "Statcast files (re-join new challenge records without a re-pull)")
    ap.add_argument("--allow-smaller", action="store_true",
                    help="rebuild from raw even if the raw files cover far fewer games "
                         "than the existing table")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING THE ANALYSIS-READY CHALLENGE INVENTORY")
    print("=" * 70 + "\n1. LOADING")

    target = OUT_DIR / "challenge_inventory_2026.parquet"
    if not args.spine_from_table:
        pitches = load_statcast()
        # Don't let a partial raw folder silently replace a full-season table.
        before = table_games(target)
        demo = "demo" in str(DATA_DIR).lower()
        if before and not demo and not args.allow_smaller and \
                pitches["game_pk"].nunique() < 0.9 * before:
            sys.exit(f"""
STOPPED before overwriting anything.
The raw Statcast files cover {pitches['game_pk'].nunique():,} games, but the existing
table ({target}) covers {before:,}. Rebuilding from these files would replace it with a
much smaller one.
  - To re-join new challenge records on the pitches you already have:
        python scripts/run_week5.py --spine-from-table
  - To rebuild from raw, finish the Statcast pull first (it resumes; a few hours):
        python scripts/01_statcast_pitches_2026.py
  - To overwrite anyway, add --allow-smaller.""")
    challenges = load_challenges()
    bios = maybe(RAW / "players" / "player_bios_2026.parquet", "Player bios")
    winprob = maybe(RAW / "winprob" / "winprob_2026.parquet", "Win probability")
    framing = maybe(RAW / "savant" / "catcher_framing_all_years.parquet", "Catcher framing")
    umpires = maybe(RAW / "umpires" / "umpire_accuracy_derived_2026.parquet",
                    "Umpire accuracy")
    if args.spine_from_table:
        pitches = load_spine_from_table(rebuild_sources(bios, winprob, framing, umpires))

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
        acc, rec_orig, _ = zone_agreement(ch_rows)
        note(f"Your ABS zone predicts the actual challenge outcome {acc:.1%} of the time "
             f"(against the ORIGINAL call, worked out from who challenged; the recorded call "
             f"is still the original on {rec_orig:.0%} of overturned challenges)",
             ok=acc >= 0.85)
        qa.append("    This is the single best test of your zone implementation. "
                  "Below ~85% means the zone constants or the ball-radius rule "
                  "are off, and every distance feature downstream is biased.")

    # Challenge rate on opportunities, by challenges remaining — the behavioural
    # signature the MDP is supposed to explain.
    # Group BATTING-side opportunities by the BATTING side's inventory. Pooling
    # all opportunities here would mix in pitches that are only the fielding
    # side's opportunity, and the rate would be meaningless.
    opp = df[df["is_challenge_opportunity_batting"]]
    if len(opp):
        challenged_by_batter = opp["was_challenged"] & (opp["challenger_role"] == "batter")
        tbl = opp.assign(_c=challenged_by_batter).groupby(
            "challenges_remaining_batting").agg(
            opportunities=("_c", "size"),
            challenge_rate=("_c", "mean")).round(5)
        print("\n  Batting-side challenge rate by challenges remaining:")
        print(tbl.to_string())
        qa.append(tbl.to_string())

    step("6. FINAL analysis-ready table", len(df), len(df.columns))

    if args.spine_from_table:              # keep the table being replaced
        import shutil
        for old in (target, target.with_suffix(".csv")):
            if old.exists():
                bak = old.with_name(old.stem + "_prev" + old.suffix)
                if bak.exists():           # never overwrite the first backup
                    note(f"A backup is already there and was left as it is: {bak}")
                else:
                    shutil.copy2(old, bak)
                    note(f"Previous table kept as {bak}")
    out = write_any(df, OUT_DIR / "challenge_inventory_2026.parquet")
    ch_out = write_any(df[df["was_challenged"]],
                       OUT_DIR / "challenge_events_2026.parquet")

    led = pd.DataFrame(ledger)
    led["row_change"] = led["rows"].diff().fillna(0).astype(int)
    led.to_csv(OUT_DIR / "join_ledger.csv", index=False)
    print("\n  JOIN LEDGER (rows must not change after step 1):")
    print("  " + led.to_string(index=False).replace("\n", "\n  "))
    qa.append("\nJOIN LEDGER\n" + led.to_string(index=False))

    spine = led.loc[led["step"].str.startswith("1."), "rows"].iloc[0]
    final = led.loc[led["step"].str.startswith("6."), "rows"].iloc[0]
    if not args.called_only:
        note(f"Row count preserved through every join: {spine:,} -> {final:,}",
             ok=spine == final)

    (OUT_DIR / "qa_report.txt").write_text("\n".join(qa))

    print(f"\n{'=' * 70}")
    print(f"Saved {len(df):,} rows x {len(df.columns)} cols -> {out}")
    print(f"Saved {n_ch:,} challenge events -> {ch_out}")
    print(f"Join ledger -> {OUT_DIR/'join_ledger.csv'}")
    print(f"QA notes    -> {OUT_DIR/'qa_report.txt'}")
    print("\nNext: 09_data_dictionary.py, then 10_qa_assertions.py, "
          "then 11_verify_challenges_remaining.py")
    print("""
BEFORE YOU MODEL: hand-verify challenges_remaining on three real games.
Open the play-by-play on MLB.com, find each challenge, and confirm the
inventory in this table matches. It is derived, not published, and it is the
state variable your entire MDP rests on. Twenty minutes here saves November.""")
    print("=" * 70)


if __name__ == "__main__":
    main()
