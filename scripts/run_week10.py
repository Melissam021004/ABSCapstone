#!/usr/bin/env python3
"""
run_week10.py — Week 10 in one command, tests first.

    python scripts/run_week10.py                          # tests -> V(s) -> challenge values
    python scripts/run_week10.py --demo                   # all of it on a simulated season
    python scripts/run_week10.py --seasons 2024,2025,2026 # more history for the win table
    python scripts/run_week10.py --no-gbm                 # skip the boosted win model (faster)

Order, and why:
  1. TESTS — known answers for the win-probability engine (a coin flip between
     equal teams, extra innings, the walk-off rule, the walk/strikeout rulebook),
     then the whole of Week 10 on a simulated season with planted answers.
  2. scripts/19_week10_winprob.py — builds, validates and LOCKS V(s) in
     models/winprob/. Doesn't touch p(s), the cohort or September.
  3. scripts/20_week10_challenge_value.py — values every 2026 challenge with the
     locked p(s) and V(s): inning, game, series, season, players, umpires.
  4. DASHBOARD — model_data.js and the single-file build, so the Progress tab
     shows Week 10 (real data only).

Needs: Week 9's locked p(s) in models/production/ (script 20 only) and the raw
Statcast pulls in data/raw/statcast/ (every pitch, with the score after each pitch).

--demo builds a simulated season under data_demo/ first — and, because script 20
needs a locked p(s) for that season, runs Weeks 7–9 on it in their quick mode
(run_week89.py --demo --quick). Everything stays in the *_demo folders.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    ("tests/test_winprob.py", "win probability: known answers"),
    ("tests/test_boosting_backends.py", "CatBoost / LightGBM work on this machine"),
    ("tests/test_week10_pipeline.py", "Week 10 finds planted answers (~1 min)"),
]
DEMO_ENV = {"ABS_DATA_DIR": "data_demo", "ABS_DOCS_DIR": "docs_demo",
            "ABS_CONFIG_DIR": "config_demo", "ABS_FIG_DIR": "figures_demo_run",
            "ABS_MODEL_DIR": "models_demo"}


def run(path, args=(), quiet=False):
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / path), *args], cwd=ROOT,
                          capture_output=quiet, text=True)
    if quiet and proc.returncode:
        print(proc.stdout[-4000:], proc.stderr[-4000:])
    return proc.returncode, time.time() - t0


def step(title):
    print("\n" + "─" * 78 + f"\n{title}\n" + "─" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--games", default="2430", help="demo season size")
    ap.add_argument("--seasons", default="2026")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--boot", default="500")
    ap.add_argument("--perm", default="999")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    print("=" * 78 + "\nWEEK 10 — WIN PROBABILITY AND WHAT CHALLENGES ARE WORTH\n" + "=" * 78)
    if not args.skip_tests:
        print("\nTESTS (stop here if any fails)")
        for path, what in TESTS:
            rc, secs = run(path, quiet=True)
            print(f"  [{' pass ' if rc == 0 else ' FAIL '}] {what:<46} {path}  ({secs:.0f}s)")
            if rc:
                sys.exit(f"\n{path} failed — fix that before trusting any Week 10 number.")

    if args.demo:
        os.environ.update(DEMO_ENV)
        step(f"DEMO — a simulated {args.games}-game season and a locked p(s) for it")
        rc, _ = run("scripts/run_week89.py", ["--demo", "--quick", "--skip-tests",
                                              "--games", args.games], quiet=True)
        if rc:
            sys.exit("\nThe demo season / p(s) build failed (see above).")
        print("  data_demo/ and models_demo/production/ ready")

    step("PART A — V(s): win probability, run expectancy, call transitions (script 19)")
    extra = ["--seasons", args.seasons, "--boot", args.boot] + (["--no-gbm"] if args.no_gbm else [])
    rc, _ = run("scripts/19_week10_winprob.py", extra)
    if rc:
        sys.exit(rc)
    step("PART B — what every challenge was worth (script 20)")
    rc, _ = run("scripts/20_week10_challenge_value.py", ["--boot", "1000", "--perm", args.perm])
    if rc:
        sys.exit(rc)
    if not args.demo:
        step("DASHBOARD — the Progress tab picks up Week 10")
        for path in ("scripts/18_export_model_dashboard.py", "scripts/14_build_single_file.py"):
            rc, _ = run(path)
            if rc:
                sys.exit(rc)
    docs = os.environ.get("ABS_DOCS_DIR", "docs")
    print("\n" + "=" * 78)
    print(f"Next: read {docs}/WEEK10_REPORT.md and {docs}/WEEK10_VALUE_REPORT.md with WEEK10.md "
          "beside them.")


if __name__ == "__main__":
    main()
