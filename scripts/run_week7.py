#!/usr/bin/env python3
"""
run_week7.py — the whole of Week 7 in one command.

Runs the three test files first and stops if any fails: every number in the
Week 7 report comes from code they check, so there's no point reading the report
if they don't pass. Then it runs the baseline script.

    python scripts/run_week7.py            # tests, then the real run on data/
    python scripts/run_week7.py --demo     # tests, then a synthetic 1,200-game
                                           # season end to end (~2 min)
    python scripts/run_week7.py --boot 200 # quicker bootstrap while iterating

--demo keeps everything it makes in separate *_demo folders (data_demo/,
docs_demo/, config_demo/, figures_demo_run/week7/, models_demo/), so it can never
overwrite your real cohort, report or models.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TESTS = [
    ("tests/test_abs_metrics.py", "metrics give known right answers"),
    ("tests/test_abs_models.py", "logistic + GAM recover known truths"),
    ("tests/test_week7_pipeline.py", "pipeline finds planted effects"),
]

DEMO_ENV = {"ABS_DATA_DIR": "data_demo", "ABS_DOCS_DIR": "docs_demo",
            "ABS_CONFIG_DIR": "config_demo", "ABS_FIG_DIR": "figures_demo_run",
            "ABS_MODEL_DIR": "models_demo"}


def run(path, args=(), quiet=False):
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / path), *args],
                          capture_output=quiet, text=True)
    if quiet and proc.returncode:
        print(proc.stdout[-4000:], proc.stderr[-4000:])
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--boot", default="1000")
    ap.add_argument("--games", default="1200", help="demo season size")
    args = ap.parse_args()

    print("=" * 78 + "\nWEEK 7 — BASELINE OVERTURN MODEL\n" + "=" * 78)
    print("\nTESTS (stop here if any fails)")
    for path, what in TESTS:
        rc, secs = run(path, quiet=True)
        print(f"  [{' pass ' if rc == 0 else ' FAIL '}] {what:<40} {path}  ({secs:.0f}s)")
        if rc:
            sys.exit(f"\n{path} failed — fix that before trusting any Week 7 number.")

    if args.demo:
        os.environ.update(DEMO_ENV)
        print(f"\nDEMO — building a synthetic {args.games}-game season under data_demo/")
        for path, extra in (("scripts/make_demo_pitches.py", ["--games", args.games,
                                                              "--out", "data_demo"]),
                            ("scripts/08_build_challenge_inventory.py", []),
                            ("scripts/13_define_cohort.py", [])):
            rc, secs = run(path, extra, quiet=True)
            print(f"  [{'  ok  ' if rc == 0 else ' FAIL '}] {path}  ({secs:.0f}s)")
            if rc:
                sys.exit(f"\n{path} failed.")

    print("\n" + "─" * 78)
    rc, _ = run("scripts/15_week7_baseline.py", ["--boot", args.boot])
    if rc:
        sys.exit(rc)
    docs = os.environ.get("ABS_DOCS_DIR", "docs")
    print("\n" + "=" * 78)
    print(f"Next: open {docs}/WEEK7_BASELINE_REPORT.md with WEEK7.md beside it,")
    print("and work through WEEK7.md Part 5 (reading the report in order).")


if __name__ == "__main__":
    main()
