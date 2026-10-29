#!/usr/bin/env python3
"""
test_call_recording.py — does the pipeline notice WHICH call was recorded?

On a challenged pitch there are two calls: the umpire's original call, and the
final call after the ABS review. Models need the ORIGINAL one. Some data
sources store the final one, which silently wrecks every location feature
(an overturned pitch then looks like it was called correctly).

src/abs_features.call_recording_check works it out from who challenged —
a team can only challenge a call that went against it — and add_features
repairs the call when needed. Each test plants a known recording convention
and checks the verdict and the repair.

    python tests/test_call_recording.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.abs_features import _as_bool, add_features, call_recording_check  # noqa: E402

RNG = np.random.default_rng(11)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def season(n=4000, n_ch=1200):
    """Called pitches with the ORIGINAL call recorded, plus challenges."""
    top, bot = 3.3, 1.6
    df = pd.DataFrame({
        "plate_x": RNG.uniform(-1.2, 1.2, n), "plate_z": RNG.uniform(1.0, 4.0, n),
        "abs_zone_top": top, "abs_zone_bot": bot,
        "zone_dist_in": RNG.normal(0, 3, n),
        "called_strike": RNG.random(n) < 0.4,
        "was_challenged": False, "is_overturned": np.nan,
        "batting_side": RNG.choice(["home", "away"], n),
        "challenges_remaining_batting": RNG.integers(1, 3, n),
        "challenges_remaining_fielding": RNG.integers(1, 3, n),
        "challenger_role": None, "challenging_side": None,
        "stand": "R", "p_throws": "R", "pitch_type": "FF", "leverage_index": 1.0,
    })
    df["fielding_side"] = np.where(df.batting_side == "home", "away", "home")
    idx = RNG.choice(n, n_ch, replace=False)
    df.loc[idx, "was_challenged"] = True
    ov = RNG.random(n_ch) < 0.5
    df["is_overturned"] = df["is_overturned"].astype(object)
    df.loc[idx, "is_overturned"] = ov
    strike = df.loc[idx, "called_strike"].to_numpy()
    df.loc[idx, "challenging_side"] = np.where(strike, df.loc[idx, "batting_side"],
                                               df.loc[idx, "fielding_side"])
    df.loc[idx, "challenger_role"] = np.where(strike, "batter",
                                              RNG.choice(["catcher", "pitcher"], n_ch))
    return df, idx, ov


print("\nCALL RECORDING — the verdict for each planted convention")
base, idx, ov = season()
truth = add_features(base)

res = call_recording_check(base)
check("original call recorded -> no repair", not res["repair"] and "ORIGINAL" in res["verdict"],
      res["verdict"][:40])

final = base.copy()
flip = idx[ov]
final.loc[flip, "called_strike"] = ~final.loc[flip, "called_strike"].astype(bool)
fixed, res = add_features(final, return_check=True)
check("final (post-review) call recorded -> detected", "FINAL" in res["verdict"],
      f"{res['agree_overturned']:.0%} of overturned matched")
same = all(np.allclose(truth[c].astype(float), fixed[c].astype(float))
           for c in ("called_strike", "dist_against_call", "call_dir", "chal_left"))
check("...and repairing it restores every feature exactly", same)
check("...while keeping what was recorded in called_strike_recorded",
      bool((fixed["called_strike_recorded"] == final["called_strike"]).all()))

mixed = base.copy()
half = flip[: len(flip) // 2]
mixed.loc[half, "called_strike"] = ~mixed.loc[half, "called_strike"].astype(bool)
fixed_m, res = add_features(mixed, return_check=True)
check("mixed recording (half the overturns post-review) -> detected and repaired",
      "MIXED" in res["verdict"] and res["n_repaired"] == len(half),
      f"{res['n_repaired']} repaired")
check("...and the repaired call equals the original", bool(
    (fixed_m["called_strike"] == truth["called_strike"]).all()))

noisy = base.copy()
upheld = idx[~ov]
bad = upheld[: int(0.2 * len(upheld))]
noisy.loc[bad, "challenging_side"] = np.where(noisy.loc[bad, "challenging_side"] == "home",
                                              "away", "home")
noisy.loc[bad, "challenger_role"] = None            # force the side to decide
res = call_recording_check(noisy)
check("unreliable side data -> 'cannot verify', nothing changed",
      not res["repair"] and "cannot verify" in res["verdict"],
      f"{res['agree_upheld']:.0%} of upheld matched")

tiny = base.copy()
tiny.loc[idx[10:], "was_challenged"] = False
res = call_recording_check(tiny)
check("too few challenges -> no verdict, nothing changed", not res["repair"])

print("\nBOOLEANS — 'False' must not become True")
s = pd.Series(["True", "False", "true", "0", 1, 0.0, None, np.nan])
got = _as_bool(s).tolist()
check("strings, numbers and blanks parse correctly",
      got[:6] == [True, False, True, False, True, False] and pd.isna(got[6]) and pd.isna(got[7]),
      str(got))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL CALL-RECORDING TESTS PASSED")
