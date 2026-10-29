"""
abs_features.py — the one place features are built, for every model this term.

Weeks 7, 8 and 9 all import from here. That is deliberate: if Week 8's
CatBoost saw features computed even slightly differently from Week 7's
logistic regression, the comparison between them would be meaningless, and
nobody would notice. One builder, imported everywhere, makes that impossible.

THE TARGET
    is_overturned — did ABS reverse the call? Observed only on challenged
    pitches, so the models train on challenged pitches.

THE FEATURE THAT MATTERS MOST, AND THE TRAP IN IT
    `zone_dist_in` from Week 5 is signed relative to the ZONE: negative inside,
    positive outside. That is the wrong sign for this model. A pitch 2 inches
    outside is likely overturned if the umpire called it a strike — and almost
    certainly upheld if he called it a ball. Pool the two kinds of challenge
    and the raw distance cancels itself out (AUC ~0.5).

    `dist_against_call` re-signs it relative to the CALL: positive means the
    pitch sits on the wrong side of the edge for the call that was made.
    Overturn probability rises with it for every challenge, whoever made it.

THE BLOCKS
    Features are grouped by what they can tell you, not by data source:

    location   what ABS itself sees: how far the pitch is on the wrong side of
               the call, which edge it's near, and which way the call went
    situation  count, outs, inning, leverage, challenges left
    pitch      type, velocity, movement, handedness, batter height
    people     pitcher-vs-catcher challenges, and how accurate this umpire /
               this catcher's games have been

    Because ABS only sees the ball, anything outside `location` that still
    predicts overturns is telling you about WHO CHOSE TO CHALLENGE, not about
    how ABS rules. Keep that in mind when you read the results.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PLATE_HALF_WIDTH_FT = 8.5 / 12

PITCH_GROUP = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball", "FA": "fastball",
    "FT": "fastball",
    "SL": "breaking", "CU": "breaking", "KC": "breaking", "SV": "breaking",
    "ST": "breaking", "CS": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed",
    "KN": "offspeed", "EP": "offspeed",
}

# Nested blocks. Each model adds one block to the previous one.
#
# Location gets two kinds of per-edge term:
#   dist_against_call | edge   how fast the overturn chance climbs per inch,
#                              separately at the sides, the top and the bottom
#   call_dir | edge            whether the 50% point sits at a different place
#                              for called strikes than for called balls
# The second one exists because the ABS edge may not be exactly where our
# height-based zone puts it. If the true top of the zone is an inch lower than
# ours, a called strike needs to be an inch further "outside" before ABS agrees,
# and a called ball an inch less — the same shift shows up with opposite signs
# depending on the call. `call_dir` is +1 on called balls, -1 on called strikes,
# which is exactly the shape that shift takes.
BLOCKS = {
    "location":  dict(numeric=[],
                      categorical=["edge"],
                      slopes_by={"dist_against_call": "edge", "call_dir": "edge"}),
    "situation": dict(numeric=["balls", "strikes", "outs_when_up", "inning",
                               "log_leverage", "chal_left"],
                      categorical=[]),
    "pitch":     dict(numeric=["release_speed", "pfx_x", "pfx_z", "batter_height_ft"],
                      categorical=["pitch_group", "stand", "p_throws"]),
    "people":    dict(numeric=["pitcher_challenge", "ump_acc_pp", "catcher_acc_pp"],
                      categorical=[]),
}

# Feature sets for the Week 8 boosted models. Trees don't need hand-built
# interaction terms, so they get the raw ingredients too: where the pitch
# crossed (plate_x, plate_z) and the batter's height, from which they can work
# out the zone edges themselves. "location" = only what ABS sees; "full" adds
# the situation, the pitch and the people — including the IDs of the umpire,
# catcher, pitcher and batter as native categoricals (the reason the plan
# picked CatBoost).
_LOC_NUM = ["dist_against_call", "call_dir", "plate_x", "plate_z", "batter_height_ft"]
BOOST_SETS = {
    "location": dict(numeric=_LOC_NUM, categorical=["edge"]),
    "full": dict(numeric=_LOC_NUM + ["balls", "strikes", "outs_when_up", "inning",
                                     "log_leverage", "chal_left", "release_speed",
                                     "pfx_x", "pfx_z", "ump_acc_pp", "catcher_acc_pp"],
                 categorical=["edge", "pitch_type", "stand", "p_throws", "challenger_role",
                              "hp_umpire_id", "fielder_2", "pitcher", "batter"]),
}
ID_FEATURES = ("hp_umpire_id", "fielder_2", "pitcher", "batter")
FEATURE_BLOCK = {
    **{f: "location" for f in _LOC_NUM + ["edge"]},
    **{f: "situation" for f in ["balls", "strikes", "outs_when_up", "inning",
                                "log_leverage", "chal_left"]},
    **{f: "pitch" for f in ["release_speed", "pfx_x", "pfx_z", "pitch_type",
                            "stand", "p_throws"]},
    **{f: "people" for f in ["challenger_role", "ump_acc_pp", "catcher_acc_pp"]},
    **{f: "identity" for f in ID_FEATURES},
}
FEATURE_LABEL = {
    "dist_against_call": "inches against the call", "call_dir": "call (ball / strike)",
    "plate_x": "horizontal location", "plate_z": "vertical location",
    "batter_height_ft": "batter height", "edge": "nearest edge",
    "balls": "balls", "strikes": "strikes", "outs_when_up": "outs", "inning": "inning",
    "log_leverage": "leverage", "chal_left": "challenges left",
    "release_speed": "velocity", "pfx_x": "horizontal break", "pfx_z": "vertical break",
    "pitch_type": "pitch type", "stand": "batter hand", "p_throws": "pitcher hand",
    "challenger_role": "who challenged", "ump_acc_pp": "umpire accuracy",
    "catcher_acc_pp": "catcher-game accuracy", "hp_umpire_id": "umpire ID",
    "fielder_2": "catcher ID", "pitcher": "pitcher ID", "batter": "batter ID",
}

# Plain-English units for reading coefficients (used by the Week 7 report).
# term -> (multiply the coefficient by this, "per ..." label)
UNITS = {
    "balls": (1, "per extra ball in the count"),
    "strikes": (1, "per extra strike in the count"),
    "outs_when_up": (1, "per extra out"),
    "inning": (1, "per inning later"),
    "log_leverage": (np.log(2), "per doubling of leverage"),
    "chal_left": (1, "per extra challenge in hand"),
    "release_speed": (5, "per 5 mph faster"),
    "pfx_x": (0.5, "per 6 in. more horizontal break"),
    "pfx_z": (0.5, "per 6 in. more vertical break"),
    "batter_height_ft": (1 / 12, "per inch taller batter"),
    "pitcher_challenge": (1, "pitcher (vs catcher) made the challenge"),
    "ump_acc_pp": (1, "per point more accurate umpire"),
    "catcher_acc_pp": (1, "per point higher catcher-game accuracy"),
}


def read_any(path):
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    csv = p.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def _as_bool(s):
    """True/False from bools, 0/1 or the strings 'True'/'False' — keeping NaN.
    (A plain .astype(bool) turns the STRING 'False' into True.)"""
    if s.dtype == bool:
        return s
    m = {True: True, False: False, 1: True, 0: False, 1.0: True, 0.0: False,
         "true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    return s.map(lambda v: m.get(v.strip().lower() if isinstance(v, str) else v, np.nan)
                 if pd.notna(v) else np.nan)


def load_cohort(data_dir):
    """The frozen Week 6 cohort — called pitches with a `split` column."""
    df = read_any(Path(data_dir) / "processed" / "cohort_2026.parquet")
    if df is None:
        raise SystemExit(
            "No frozen cohort found. Week 7 trains on the Week 6 cohort so the "
            "split dates are fixed. Run:\n    python scripts/13_define_cohort.py")
    for c in ("was_challenged", "called_strike", "abs_strike", "is_called_pitch",
              "would_be_overturned", "is_challenge_opportunity"):
        if c in df.columns:
            df[c] = _as_bool(df[c]).fillna(False).astype(bool)
    if "is_overturned" in df.columns:
        df["is_overturned"] = _as_bool(df["is_overturned"])     # NaN = not challenged
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ------------------------------------------------- which call was recorded?
def implied_original_call(df):
    """
    The umpire's ORIGINAL call on each challenged pitch, worked out from WHO
    challenged rather than from the recorded call.

    A team can only challenge a call that went against it, so:
        the batting team challenged  ->  the original call was a STRIKE
        the fielding team challenged ->  the original call was a BALL
    Uses `challenging_side` vs `batting_side` when present, else the challenger's
    role (batter = strike; catcher or pitcher = ball). NaN where neither is known.
    """
    out = pd.Series(np.nan, index=df.index, dtype=object)
    if {"challenging_side", "batting_side"} <= set(df.columns):
        cs = df["challenging_side"].astype(str).str.lower()
        bs = df["batting_side"].astype(str).str.lower()
        known = df["challenging_side"].notna() & df["batting_side"].notna() & \
            cs.isin(["home", "away"]) & bs.isin(["home", "away"])
        out[known] = (cs[known] == bs[known])
    if "challenger_role" in df.columns:
        role = df["challenger_role"].astype(str).str.lower()
        fill = out.isna() & role.isin(["batter", "catcher", "pitcher"])
        out[fill] = role[fill].eq("batter")
    return out


def call_recording_check(df):
    """
    Did the data record the umpire's ORIGINAL call on challenged pitches, or the
    FINAL call after the ABS review? It matters enormously: with the final call,
    every overturned pitch looks like it was called correctly, so "distance
    against the call" stops predicting anything, and "challenges left" gets read
    off the wrong team (the impossible '0 left' challenges).

    Compares the recorded call with the call implied by who challenged:
      * upheld challenges     — both conventions agree, so this measures how
                                reliable the side/role data are (want ~100%)
      * overturned challenges — ~100% if the ORIGINAL call was recorded,
                                ~0% if the FINAL call was
    Returns a dict with the counts, both agreement rates, a verdict, and whether
    the call should be repaired (and the repaired values).
    """
    ch = df["was_challenged"].to_numpy(bool) & df["is_overturned"].notna().to_numpy()
    implied = implied_original_call(df)
    known = ch & implied.notna().to_numpy()
    rec = df["called_strike"].astype(bool)
    ov = _as_bool(df["is_overturned"]).fillna(False).astype(bool)
    up_m, ov_m = known & ~ov.to_numpy(), known & ov.to_numpy()
    agree = (rec == implied.astype(object).fillna(False).astype(bool)).to_numpy()
    n_up, n_ov = int(up_m.sum()), int(ov_m.sum())
    a_up = float(agree[up_m].mean()) if n_up else np.nan
    a_ov = float(agree[ov_m].mean()) if n_ov else np.nan

    res = {"n_challenges": int(ch.sum()), "n_with_side": int(known.sum()),
           "n_upheld": n_up, "n_overturned": n_ov,
           "agree_upheld": a_up, "agree_overturned": a_ov, "repair": False,
           "n_repaired": 0}
    if n_up < 20 or n_ov < 20:
        res["verdict"] = "too few challenges to tell — recorded call kept as-is"
    elif a_up < 0.95:
        res["verdict"] = (f"cannot verify: on UPHELD challenges the recorded call matches "
                          f"who challenged only {a_up:.0%} of the time, so the side/role data "
                          f"themselves look wrong — check the challenge-to-pitch join (Week 5). "
                          f"Recorded call kept as-is")
    elif a_ov >= 0.95:
        res["verdict"] = "ORIGINAL umpire call recorded — no repair needed"
    else:
        kind = "FINAL (post-review) call recorded" if a_ov <= 0.05 else \
               f"MIXED recording ({a_ov:.0%} of overturned challenges carry the original call)"
        res["verdict"] = f"{kind} — repaired from who challenged"
        res["repair"] = True
        fix = known & ~agree
        res["n_repaired"] = int(fix.sum())
        res["_fix_mask"] = fix
        res["_implied"] = implied
    return res


def describe_call_check(res):
    """Two printable lines for the console and the reports."""
    pct = lambda v: "—" if v != v else f"{v:.1%}"      # noqa: E731  (NaN-safe)
    return (f"recorded call matches who challenged on {pct(res['agree_upheld'])} of "
            f"{res['n_upheld']:,} upheld and {pct(res['agree_overturned'])} of "
            f"{res['n_overturned']:,} overturned challenges",
            f"verdict: {res['verdict']}"
            + (f" ({res['n_repaired']:,} challenged pitches corrected)" if res["repair"] else ""))


def nearest_edge(px, pz, top, bot):
    """
    Which edge of the ABS zone the pitch is closest to: a side (fixed at 17
    inches wide), the top, or the bottom (both set from the batter's height).

    This matters because the edges are known with different precision. The
    sides never move. The top and bottom come from listed height, which is
    rounded to the inch and may differ from the height MLB measured for ABS —
    and the ball crosses them at a downward angle, so where it's measured
    along the plate matters more. The model gets a separate curve for each.
    """
    px, pz = np.asarray(px, float), np.asarray(pz, float)
    d_side = np.abs(np.abs(px) - PLATE_HALF_WIDTH_FT)
    d_top = np.abs(pz - np.asarray(top, float))
    d_bot = np.abs(pz - np.asarray(bot, float))
    return np.where((d_side <= d_top) & (d_side <= d_bot), "side",
                    np.where(d_top <= d_bot, "top", "bottom"))


def add_features(df, return_check=False):
    """
    Every derived column the models use.

    First, makes sure `called_strike` is the umpire's ORIGINAL call on challenged
    pitches (see call_recording_check). The recorded value is kept in
    `called_strike_recorded`. Pass return_check=True to also get the check's
    result, so scripts can print the verdict.
    """
    df = df.copy()
    df["called_strike"] = _as_bool(df["called_strike"]).fillna(False).astype(bool)
    check = call_recording_check(df)
    df["called_strike_recorded"] = df["called_strike"]
    if check["repair"]:
        m = check["_fix_mask"]
        df.loc[m, "called_strike"] = check["_implied"][m].astype(bool).to_numpy()
    strike = df["called_strike"].to_numpy(bool)
    df["dist_against_call"] = df["zone_dist_in"] * np.where(strike, 1.0, -1.0)
    df["call_dir"] = np.where(strike, -1.0, 1.0)        # +1 called ball, -1 called strike
    df["edge"] = nearest_edge(df["plate_x"], df["plate_z"],
                              df["abs_zone_top"], df["abs_zone_bot"])

    li = df["leverage_index"] if "leverage_index" in df.columns else pd.Series(np.nan, index=df.index)
    df["leverage_missing"] = li.isna()
    df["log_leverage"] = np.log(li.clip(lower=0.05)).fillna(0.0)   # log(1) = 0 = average

    pt = df["pitch_type"] if "pitch_type" in df.columns else pd.Series("", index=df.index)
    df["pitch_group"] = pt.map(PITCH_GROUP).fillna("other")

    # Challenges the deciding side had: batting side on a called strike,
    # fielding side on a called ball. (Week 5 records these BEFORE the pitch,
    # so a failed challenge doesn't leak into its own row.)
    df["chal_left"] = np.where(strike, df["challenges_remaining_batting"],
                               df["challenges_remaining_fielding"]).astype(float)

    for c in ("stand", "p_throws", "challenger_role"):
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)
    role = df["challenger_role"] if "challenger_role" in df.columns else pd.Series("", index=df.index)
    # Batters can only challenge strikes and fielders only balls, so "batter vs
    # catcher" is already carried by call_dir. What's left to ask is whether a
    # PITCHER's challenge differs from a CATCHER's on the same kind of call.
    df["pitcher_challenge"] = role.str.lower().eq("pitcher").astype(float)
    if return_check:
        return df, {k: v for k, v in check.items() if not k.startswith("_")}
    return df


def participant_rates(df, train_mask, shrink=200):
    """
    How often each umpire's calls (and the calls in each catcher's games)
    agreed with the ABS zone, computed from TRAINING-window pitches only, in
    percentage points above (+) or below (−) the league rate.

    Why only training: a season-long accuracy figure would include August and
    September. Using it to predict August challenges hands the model information
    from the future — a quiet form of leakage that inflates validation scores.

    Why shrink: an umpire seen for 40 pitches has a noisy rate. Each rate is
    pulled toward the league average as if it came with `shrink` extra pitches
    at the average rate, so small samples lean on the average and large samples
    speak for themselves (empirical-Bayes shrinkage, in its simplest form).

    Why percentage points: raw accuracies all sit between about 0.91 and 0.95,
    so "per 1.0 of accuracy" is a meaningless unit and makes the fit unstable.
    "Per point above league average" is readable and well-behaved.
    """
    df = df.copy()
    tr = df[train_mask & df["is_called_pitch"].to_numpy(bool)]
    correct = (tr["called_strike"] == tr["abs_strike"]).astype(float)
    league = float(correct.mean()) if len(correct) else 0.93
    rates = {}
    for col, out in (("hp_umpire_id", "ump_acc_pp"), ("fielder_2", "catcher_acc_pp")):
        if col not in df.columns:
            df[out] = 0.0
            continue
        g = pd.DataFrame({"k": tr[col], "c": correct}).groupby("k")["c"].agg(["sum", "count"])
        rate = (g["sum"] + shrink * league) / (g["count"] + shrink)
        rates[out] = rate
        df[out] = 100 * (df[col].map(rate).fillna(league) - league)
    return df, league, rates


def split_masks(df):
    s = df["split"].astype(str)
    return s.eq("train").to_numpy(), s.eq("validate").to_numpy(), s.eq("test").to_numpy()


def challenge_rows(df):
    """Challenged pitches with a known outcome — the modelling population."""
    m = df["was_challenged"].to_numpy(bool) & df["is_overturned"].notna().to_numpy()
    out = df[m].copy()
    out["y"] = out["is_overturned"].astype(bool).astype(int)
    return out


def block_spec(names):
    """Merge several blocks into one Design spec."""
    spec = dict(numeric=[], categorical=[], slopes_by={})
    for n in names:
        b = BLOCKS[n]
        spec["numeric"] += b.get("numeric", [])
        spec["categorical"] += b.get("categorical", [])
        spec["slopes_by"].update(b.get("slopes_by", {}))
    return spec
