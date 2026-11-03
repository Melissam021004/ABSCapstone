#!/usr/bin/env python3
"""
test_week10_pipeline.py — Week 10 end to end, on a simulated season with planted answers.

The simulated season (scripts/make_demo_pitches.py) is real baseball, simplified,
with an MLB-like run environment — so the published win-expectancy values must
come out right — and with differences PLANTED for Week 10 to find: some umpires
miss more calls than others, and some players are better than others at telling
a wrong call from a right one.

  1. script 19 builds and locks V(s): every published sanity value in range, the
     plan's check (tie, bottom 9th, runner on 3rd, 1 out ≈ 80%+), calibrated out
     of fold, run expectancy matching the plain averages, the rulebook transitions
  2. the locked V(s) values calls the right way round
  3. script 20 values every challenge: overturned strikes really were followed by
     more runs (and about as many more as V(s) predicted); the planted umpire
     differences and the planted differences in players' judgment are found;
     series are 2–4 games; the stakes still ahead shrink as the game goes on

    python tests/test_week10_pipeline.py        (about a minute)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def run(script, env, *args):
    p = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"{script} failed")
    return p.stdout


tmp = Path(tempfile.mkdtemp(prefix="w10test_"))
try:
    env = dict(os.environ, ABS_DATA_DIR=str(tmp / "data"), ABS_DOCS_DIR=str(tmp / "docs"),
               ABS_CONFIG_DIR=str(tmp / "config"), ABS_MODEL_DIR=str(tmp / "models"),
               ABS_FIG_DIR=str(tmp / "figs"), MPLBACKEND="Agg")
    print("\nBuilding a simulated 1,500-game season ...")
    run("scripts/make_demo_pitches.py", env, "--games", "1500", "--out", str(tmp / "data"))
    run("scripts/08_build_challenge_inventory.py", env)
    run("scripts/13_define_cohort.py", env)
    truth = json.loads((tmp / "data" / "raw" / "_demo_truth.json").read_text())

    print("\nScript 19 — V(s)")
    out19 = run("scripts/19_week10_winprob.py", env, "--no-gbm", "--boot", "60", "--no-figures")
    man = json.loads((tmp / "models" / "winprob" / "manifest.json").read_text())
    san = pd.read_csv(tmp / "docs" / "week10_sanity.csv")
    check("V(s) is locked with a version and hashed files",
          man["version"].count("-") == 1 and len(man["files"]) >= 5)
    check("at least 10 of 11 published sanity values in range", int(san["ok"].sum()) >= 10,
          f"{int(san['ok'].sum())} of {len(san)}")
    plan = san[san["state"].str.contains("plan's check")].iloc[0]
    check("the plan's check: tie, bottom 9th, runner on 3rd, 1 out", 0.76 <= plan["value"] <= 0.90,
          f"{plan['value']:.3f}")
    oof = man["out_of_fold"]["markov"]
    check("calibrated out of fold (intercept near 0, slope near 1)",
          abs(oof["cal_intercept"]) < 0.15 and 0.85 < oof["cal_slope"] < 1.2,
          f"intercept {oof['cal_intercept']:+.3f}, slope {oof['cal_slope']:.3f}")
    re = pd.read_csv(tmp / "docs" / "week10_run_expectancy.csv")
    r0 = re[(re["batting"] == "both") & (re["outs"] == 0) & (re["bases"] == "empty")
            & (re["count"] == "0-0")]["exp_runs"].iloc[0]
    check("run expectancy, bases empty / 0 out, is MLB-like (0.40–0.62)", 0.40 <= r0 <= 0.62,
          f"{r0:.3f}")
    gap = float(out19.split("weighted mean |difference| ")[1].split(" runs")[0])
    check("the run chain matches the plain averages (RE24)", gap < 0.05, f"{gap:.3f} runs")
    tt = pd.read_csv(tmp / "docs" / "week10_call_transitions.csv")
    share = (tt["p_rulebook"] * tt["n"]).sum() / tt["n"].sum()
    check("called pitches go where the rulebook says ≥ 97% of the time", share > 0.97, f"{share:.3f}")
    k3 = tt[(tt["count"] == "1-2") & (tt["call"] == "strike") & (tt["outs"] == 2)]
    check("strike three with two out is recorded as ending the inning",
          k3["rulebook_next"].str.contains("inning over").all())

    print("\nThe locked V(s) values calls the right way round")
    os.environ["ABS_MODEL_DIR"] = str(tmp / "models")
    from src.abs_winprob import load_winprob                         # noqa: E402
    wp = load_winprob(tmp / "models" / "winprob")
    late = wp.call_value({"inning": 9, "half": "bot", "outs": 2, "bases": 0, "score_diff": 0,
                          "balls": 1, "strikes": 2}, "strike").iloc[0]
    early = wp.call_value({"inning": 1, "half": "top", "outs": 0, "bases": 0, "score_diff": 0,
                           "balls": 0, "strikes": 0}, "strike").iloc[0]
    blow = wp.call_value({"inning": 8, "half": "bot", "outs": 2, "bases": 0, "score_diff": 8,
                          "balls": 1, "strikes": 2}, "strike").iloc[0]
    check("a strike-three overturn late in a tie game is worth 3x+ a 0-0 call in the 1st",
          late["dwp_challenger"] > 3 * early["dwp_challenger"] > 0,
          f"{100 * late['dwp_challenger']:.2f} vs {100 * early['dwp_challenger']:.2f} pts")
    check("...and almost nothing in a blowout", blow["dwp_challenger"] < 0.003,
          f"{100 * blow['dwp_challenger']:.2f} pts")

    print("\nScript 20 — what the challenges were worth")
    # a locked p(s) for the season: Week 7's M1 (distance) is enough for this test
    from src.abs_baselines import fit_model                          # noqa: E402
    from src.abs_calibration import Calibrator                       # noqa: E402
    from src.abs_pipeline import load_challenges                     # noqa: E402
    from src.abs_predict import write_bundle                         # noqa: E402
    data = load_challenges(tmp / "data", tmp / "config")
    m1 = fit_model("M1", data["tr"])

    def save(folder):
        with open(Path(folder) / "model.pkl", "wb") as fh:
            pickle.dump(m1, fh)
    write_bundle(tmp / "models" / "production", model=m1, calibrator=Calibrator("none"),
                 rates=data["rates"], league=data["league"], model_files=save,
                 manifest_extra={"date": "2026.01.01", "family": "logistic",
                                 "model_label": "M1 · distance", "features": ["dist_against_call"],
                                 "calibration": {"method": "none"}})
    out20 = run("scripts/20_week10_challenge_value.py", env, "--boot", "200", "--perm", "99",
                "--no-figures")
    nat = pd.read_csv(tmp / "docs" / "week10_value_natural_experiments.csv")
    bat = nat[(nat["side"] == "batting") & nat["outcome"].str.startswith("runs")].iloc[0]
    check("overturned strikes were followed by more runs (interval reaches above 0)",
          bat["actual_diff_hi"] > 0,
          f"{bat['actual_diff']:+.3f} runs [{bat['actual_diff_lo']:+.3f}, {bat['actual_diff_hi']:+.3f}]")
    check("...about as many more as V(s) predicted (inside the 95% interval)",
          bat["actual_diff_lo"] <= bat["pred_diff"] <= bat["actual_diff_hi"],
          f"predicted {bat['pred_diff']:+.3f}")
    pu = float(out20.split("same difficulty)? permutation p = ")[1].split(";")[0])
    check("the planted umpire differences are found", pu < 0.05, f"p = {pu:.3f}")
    ps = float(out20.split("WHICH pitches they challenge? permutation p = ")[1].split(";")[0])
    check("the planted differences in players' judgment are found", ps < 0.05, f"p = {ps:.3f}")
    ser = pd.read_csv(tmp / "docs" / "week10_value_series.csv")
    med = ser["series_len"].median()
    check("series are 2–4 games", 2 <= med <= 4 and ser["series_len"].max() <= 6, f"median {med}")
    ah = pd.read_csv(tmp / "docs" / "week10_value_ahead.csv").set_index("half_index")
    check("the stakes still ahead shrink as the game goes on",
          ah.loc[0, "stakes"] > ah.loc[8, "stakes"] > ah.loc[16, "stakes"],
          f"{100 * ah.loc[0, 'stakes']:.1f} -> {100 * ah.loc[8, 'stakes']:.1f} -> "
          f"{100 * ah.loc[16, 'stakes']:.1f} pts")
    ts = pd.read_csv(tmp / "docs" / "week10_value_team_season.csv")
    check("every team's season is valued", len(ts) == 30 and (ts["wins_added"] >= 0).all())
    ump = pd.read_csv(tmp / "docs" / "week10_value_umpires.csv")
    planted = pd.Series(truth["umpire_error_multiplier"])
    ump["planted"] = ump["umpire"].map(planted)
    rho = ump[["excess_per_100", "planted"]].corr(method="spearman").iloc[0, 1]
    check("umpires ranked by extra wrong calls follow the planted error rates", rho > 0.6,
          f"Spearman ρ = {rho:.2f}")
    check("the report is written", (tmp / "docs" / "WEEK10_VALUE_REPORT.md").exists())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL WEEK 10 PIPELINE CHECKS PASSED")
