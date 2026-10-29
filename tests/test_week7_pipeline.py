#!/usr/bin/env python3
"""
test_week7_pipeline.py — does the Week 7 script find effects we planted on purpose?

The unit tests (test_abs_metrics.py, test_abs_models.py) check each piece on
its own. This one checks the WHOLE pipeline: it builds a fake season where the
right answers are known, runs scripts/15_week7_baseline.py on it, and checks
that the outputs say what they should. If the script ever stops finding these,
something in it broke — even if every individual piece still passes.

What gets planted:
  1. ABS's top edge sits 0.8 in. ABOVE where our height-based zone puts it.
     The sides and the bottom are exactly right.
  2. The overturn curve has a floor and a ceiling (6% and 94%): a straight
     line in log-odds can't bend like that, a GAM can.
  3. A pitcher's challenge has HALF the odds of a catcher's on a pitch at the
     same spot — a pure selection effect.
  4. Count, outs, inning, leverage, velocity, movement: no effect at all.
  5. One umpire is terrible — but only in August. A leak-free pipeline must
     not know that when it predicts August.
  6. Three August challenges are "mislinked": the pitch is a foot on the
     right side of the call, yet the record says overturned.

    python tests/test_week7_pipeline.py          (about 30 seconds)

It works in a temporary folder and never touches your data/, docs/ or models/.
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)
HW = 8.5 / 12                      # half the plate, ft
R_BALL = 1.45                      # ball radius, in.
TOP_SHIFT_IN = 0.8                 # planted: ABS top edge is this far above ours
BAD_UMP = 999
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- the season
def build_season(n_games=1500, pitches_per_game=60, challenges_per_game=4):
    dates = pd.date_range("2026-03-26", "2026-09-27", periods=n_games).normalize()
    umps = np.arange(1, 61)
    rows = []
    for g in range(n_games):
        date = dates[g]
        august = date.month == 8
        # The bad umpire works ~5% of games all season (so training sees him as
        # ordinary) and ~40% of August games, where he's bad.
        ump = BAD_UMP if RNG.random() < (0.4 if august else 0.05) else int(RNG.choice(umps))
        catcher = int(RNG.integers(100, 160))
        n_ch = challenges_per_game
        for i in range(pitches_per_game):
            challenged = i < n_ch
            h = RNG.normal(6.1, 0.2)
            top, bot = 0.535 * h, 0.27 * h
            if challenged:
                strike = RNG.random() < 0.5
                edge = RNG.choice(["side", "top", "bottom"], p=[0.56, 0.22, 0.22])
                z = float(np.clip(RNG.normal(0, 2.0), -4, 3.5))   # in., + = outside the zone
            else:
                strike = RNG.random() < 0.35
                edge = RNG.choice(["side", "top", "bottom"])
                z = float(RNG.normal(0, 4))
            out_ft = (R_BALL + z) / 12
            if edge == "side":
                px = RNG.choice([-1, 1]) * (HW + out_ft)
                pz = RNG.uniform(bot + 0.45, top - 0.45)
            elif edge == "top":
                px, pz = RNG.uniform(-0.3, 0.3), top + out_ft
            else:
                px, pz = RNG.uniform(-0.3, 0.3), bot - out_ft
            # umpire accuracy on ordinary pitches: 93%, the bad umpire 80% in August
            acc = 0.80 if ump == BAD_UMP and august else 0.93
            abs_strike = strike if RNG.random() < acc else (not strike)
            row = dict(game_pk=700000 + g, game_date=date.strftime("%Y-%m-%d"),
                       at_bat_number=i // 4 + 1, pitch_number=i % 4 + 1,
                       inning=int(RNG.integers(1, 10)) if RNG.random() > .03 else 10,
                       balls=int(RNG.integers(0, 4)), strikes=int(RNG.integers(0, 3)),
                       outs_when_up=int(RNG.integers(0, 3)),
                       plate_x=px, plate_z=pz, abs_zone_top=top, abs_zone_bot=bot,
                       batter_height_ft=h, zone_dist_in=z, called_strike=strike,
                       abs_strike=abs_strike, is_called_pitch=True,
                       hp_umpire_id=ump, fielder_2=catcher,
                       leverage_index=float(np.exp(RNG.normal(0, 0.6))),
                       release_speed=RNG.normal(93, 4), pfx_x=RNG.normal(0, .7),
                       pfx_z=RNG.normal(.8, .6),
                       pitch_type=RNG.choice(["FF", "SI", "SL", "CH", "CU", "FC"]),
                       stand=RNG.choice(["L", "R"]), p_throws=RNG.choice(["L", "R"], p=[.3, .7]),
                       challenges_remaining_batting=int(RNG.integers(1, 3)),
                       challenges_remaining_fielding=int(RNG.integers(1, 3)),
                       was_challenged=challenged, is_overturned=np.nan, challenger_role=np.nan)
            if challenged:
                role = "batter" if strike else ("pitcher" if RNG.random() < 0.25 else "catcher")
                d = z if strike else -z                                  # against the call
                shift = TOP_SHIFT_IN if edge == "top" else 0.0
                d_eff = d - shift if strike else d + shift
                p = 0.06 + 0.88 * norm.cdf(d_eff / 0.6)                  # floor and ceiling
                if role == "pitcher":
                    p = expit(logit(p) + np.log(0.5))                    # selection effect
                row.update(challenger_role=role, is_overturned=bool(RNG.random() < p))
            rows.append(row)
    df = pd.DataFrame(rows)
    gd = pd.to_datetime(df["game_date"])
    df["split"] = np.select([gd <= "2026-07-31", gd <= "2026-08-31"],
                            ["train", "validate"], "test")
    # three mislinked August challenges: a foot on the right side, "overturned"
    aug = df.index[(df["split"] == "validate") & df["was_challenged"]][[5, 50, 300]]
    df.loc[aug, "zone_dist_in"] = np.where(df.loc[aug, "called_strike"], -12.0, 12.0)
    df.loc[aug, "is_overturned"] = True
    return df, df.loc[aug, ["game_pk", "at_bat_number", "pitch_number"]]


# ---------------------------------------------------------------- run it
def main():
    tmp = Path(tempfile.mkdtemp(prefix="w7_e2e_"))
    try:
        data = tmp / "data_demo_e2e"
        (data / "processed").mkdir(parents=True)
        print("\nBuilding a fake season with planted effects ...")
        season, mislinked = build_season()
        season.to_csv(data / "processed" / "cohort_2026.csv", index=False)
        n_test = int((season["was_challenged"] & (season["split"] == "test")).sum())
        print(f"  {len(season):,} called pitches, {int(season['was_challenged'].sum()):,} challenges")

        env = dict(os.environ, ABS_DATA_DIR=str(data), ABS_DOCS_DIR=str(tmp / "docs"),
                   ABS_CONFIG_DIR=str(tmp / "config"), ABS_FIG_DIR=str(tmp / "figs"),
                   ABS_MODEL_DIR=str(tmp / "models"))
        print("Running scripts/15_week7_baseline.py on it ...")
        run = subprocess.run([sys.executable, str(ROOT / "scripts" / "15_week7_baseline.py"),
                              "--boot", "200", "--no-figures"],
                             env=env, capture_output=True, text=True)
        if run.returncode:
            print(run.stdout[-3000:], run.stderr[-3000:])
            check("the script runs to the end", False)
            return
        check("the script runs to the end", True)

        docs, models = tmp / "docs", tmp / "models"
        spec = json.loads((models / "week7_feature_spec.json").read_text())
        loc = pd.read_csv(docs / "week7_location.csv")
        comp = pd.read_csv(docs / "week7_comparisons.csv")
        coef = pd.read_csv(docs / "week7_coefficients.csv")
        wrong = pd.read_csv(docs / "week7_confidently_wrong.csv")
        preds = pd.read_csv(data / "processed" / "week7_validate_predictions.csv")
        with open(models / "week7_participant_rates.pkl", "rb") as fh:
            rates = pickle.load(fh)

        print("\nThe sign trap")
        check("raw zone distance is near-useless", spec["auc_zone_dist_in_raw"] < 0.65,
              f"AUC {spec['auc_zone_dist_in_raw']:.3f}")
        check("re-signed by the call, it's strong", spec["auc_dist_against_call"] > 0.80,
              f"AUC {spec['auc_dist_against_call']:.3f}")

        print("\nPlanted zone shift: top edge 0.8 in. bigger than ours, others exact")
        m2 = loc[loc["model"] == "M2"]
        for edge, lo, hi in (("top", 0.45, 1.15), ("side", -0.3, 0.3), ("bottom", -0.35, 0.35)):
            s_ = m2[(m2["edge"] == edge) & (m2["call_dir"] < 0)]["p50"].iloc[0]
            b_ = m2[(m2["edge"] == edge) & (m2["call_dir"] > 0)]["p50"].iloc[0]
            gap = (s_ - b_) / 2
            check(f"{edge}: half-gap between coin-flip points in [{lo:+.2f}, {hi:+.2f}]",
                  lo <= gap <= hi, f"{gap:+.2f} in.")

        print("\nPlanted floor/ceiling: the GAM should beat the straight line on August")
        r = comp[(comp["model_a"] == "GAM-loc") & (comp["model_b"] == "M2") & (comp["metric"] == "brier")]
        check("GAM-loc's Brier is reliably below M2's", float(r["ci_hi"].iloc[0]) < 0,
              f"Δ {float(r['a_minus_b'].iloc[0]):+.4f} [{float(r['ci_lo'].iloc[0]):+.4f}, "
              f"{float(r['ci_hi'].iloc[0]):+.4f}]")

        print("\nPlanted selection effect: pitcher challenges at half the odds")
        pc = coef[(coef["model"] == "M5") & (coef["term"] == "pitcher_challenge")].iloc[0]
        check("odds ratio recovered near 0.5", 0.3 <= pc["odds_ratio"] <= 0.8,
              f"OR {pc['odds_ratio']:.2f} [{pc['or_lo']:.2f}, {pc['or_hi']:.2f}]")
        check("...and its interval excludes 1", pc["or_hi"] < 1)

        print("\nPlanted nothing: situation and pitch blocks")
        for big, small in (("M3", "M2"), ("M4", "M3")):
            r = comp[(comp["model_a"] == big) & (comp["model_b"] == small) & (comp["metric"] == "brier")]
            v = float(r["a_minus_b"].iloc[0])
            check(f"{big} vs {small}: Brier barely moves", abs(v) < 0.002, f"Δ {v:+.4f}")

        print("\nLeakage: the umpire who's only bad in August")
        lg = rates["league"]
        seen = BAD_UMP in rates["rates"]["ump_acc_pp"].index
        pp = 100 * (rates["rates"]["ump_acc_pp"].get(BAD_UMP, lg) - lg)
        check("he worked training games, so the check means something", seen)
        check("his accuracy feature comes from training months only (≈ league)",
              abs(pp) < 1.0, f"{pp:+.2f} points vs league; a leak would show about −8")

        print("\nMislinked challenges")
        keys = set(map(tuple, mislinked.to_numpy()))
        top5 = set(map(tuple, wrong.head(5)[["game_pk", "at_bat_number", "pitch_number"]].to_numpy()))
        check("all three planted mislinks are in M2's top 5 'most surprised by'",
              keys <= top5, f"{len(keys & top5)} of 3 found")

        print("\nThe test window stays sealed")
        check("validation predictions are August-only",
              pd.to_datetime(preds["game_date"]).dt.month.eq(8).all())
        check("test challenges were counted, not used",
              spec["n_challenges"]["test_sealed"] == n_test,
              f"{spec['n_challenges']['test_sealed']} sealed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("ALL PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
