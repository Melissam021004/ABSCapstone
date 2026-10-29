#!/usr/bin/env python3
"""
run_week89.py — Weeks 8 and 9 in one command, tests first.

    python scripts/run_week89.py                  # tests -> Week 7 refresh -> Week 8 -> Week 9 lock
                                                  #   -> dashboard (September stays sealed)
    python scripts/run_week89.py --final-test     # ...and then open September ONCE at the end
    python scripts/run_week89.py --demo           # everything, on a synthetic season (~5 min)
    python scripts/run_week89.py --quick          # 6 settings, 300 resamples: for iterating

Order, and why:
  1. TESTS — every number in the reports comes from code these check. The
     boosting smoke test runs first: it tells you in 30 seconds whether
     CatBoost and LightGBM work on this machine.
  2. WEEK 7 — re-run once, because the call-recording check now runs before
     any model is fitted (see WEEK8.md Part 0). Your Week 7 report is
     refreshed with the corrected data.
  3. WEEK 8 — tune and compare (the longest step: ~10–40 min with 24 settings).
  4. WEEK 9 — calibrate, check subgroups, lock p(s).
  5. --final-test — only when you pass it. The test month is used once; after
     that, re-locking needs --relock-after-test and is recorded. Once it has
     been opened, running this script again skips Weeks 7–9 and only
     refreshes the dashboard.
  6. DASHBOARD — refresh data.js and model_data.js and rebuild the
     single-file dist/abs-dashboard.html.

--demo keeps everything in *_demo folders (data_demo/, docs_demo/,
config_demo/, figures_demo_run/, models_demo/) so it can never touch your real
cohort, reports or models — and it opens the demo's test month automatically.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    ("tests/test_boosting_backends.py", "CatBoost / LightGBM work on this machine"),
    ("tests/test_abs_metrics.py", "metrics give known right answers"),
    ("tests/test_abs_models.py", "logistic + GAM recover known truths"),
    ("tests/test_call_recording.py", "original-vs-final call check"),
    ("tests/test_calibration.py", "Platt / isotonic / cross-fitting"),
    ("tests/test_predict_interface.py", "predict(state) == the pipeline"),
    ("tests/test_week7_pipeline.py", "Week 7 finds planted effects"),
    ("tests/test_week89_pipeline.py", "Weeks 8–9 find a planted leak and drift"),
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
    ap.add_argument("--final-test", action="store_true",
                    help="open the September test month once, after locking")
    ap.add_argument("--quick", action="store_true", help="6 settings, 300 bootstrap resamples")
    ap.add_argument("--trials", default=None)
    ap.add_argument("--boot", default=None)
    ap.add_argument("--production", choices=["location", "full"], default="location")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-week7", action="store_true")
    ap.add_argument("--games", default="1200", help="demo season size")
    args = ap.parse_args()
    trials = args.trials or ("6" if args.quick else "24")
    boot = args.boot or ("300" if args.quick else "1000")

    print("=" * 78 + "\nWEEKS 8–9 — GRADIENT BOOSTING, CALIBRATION, LOCKED p(s)\n" + "=" * 78)
    if not args.skip_tests:
        print("\nTESTS (stop here if any fails)")
        for path, what in TESTS:
            rc, secs = run(path, quiet=True)
            print(f"  [{' pass ' if rc == 0 else ' FAIL '}] {what:<44} {path}  ({secs:.0f}s)")
            if rc:
                sys.exit(f"\n{path} failed — fix that before trusting any Week 8–9 number.")

    if args.demo:
        os.environ.update(DEMO_ENV)
        # the demo is rebuilt from scratch, so its old lock (and test stamp) go too
        for d in ("production", "production_history", "week8"):
            shutil.rmtree(ROOT / "models_demo" / d, ignore_errors=True)
        print(f"\nDEMO — building a synthetic {args.games}-game season under data_demo/")
        for path, extra in (("scripts/make_demo_pitches.py", ["--games", args.games,
                                                              "--out", "data_demo"]),
                            ("scripts/08_build_challenge_inventory.py", []),
                            ("scripts/13_define_cohort.py", [])):
            rc, secs = run(path, extra, quiet=True)
            print(f"  [{'  ok  ' if rc == 0 else ' FAIL '}] {path}  ({secs:.0f}s)")
            if rc:
                sys.exit(f"\n{path} failed.")

    # Once September has been opened, the locked p(s) is final. Week 9 refuses
    # to re-lock after that (unless --relock-after-test), so re-running Weeks
    # 7–9 would spend 10–40 minutes on Week 8 and then stop at that refusal
    # without refreshing the dashboard. Skip straight to the dashboard instead.
    stamp = (ROOT / os.environ.get("ABS_MODEL_DIR", "models") / "production"
             / "TEST_OPENED.json")
    test_opened = stamp.exists() and not args.demo
    if test_opened:
        st = json.loads(stamp.read_text())
        print(f"\nSeptember was opened on {st.get('opened', '?')[:10]} for p(s) "
              f"{st.get('version', '?')} — that lock is final.")
        print("Skipping Weeks 7–9 and refreshing the dashboard only. (Re-locking now "
              "needs\n  python scripts/17_week9_calibrate_lock.py --relock-after-test\n"
              "and is recorded in the model card — see WEEK9.md.)")
        if args.final_test:
            print("--final-test ignored: the test month has already had its one look.")
    else:
        if not args.skip_week7:
            step("WEEK 7 — refresh the baselines with the call-recording check")
            rc, _ = run("scripts/15_week7_baseline.py", ["--boot", boot])
            if rc:
                sys.exit(rc)
        step(f"WEEK 8 — gradient boosting ({trials} settings per library per feature set)")
        rc, _ = run("scripts/16_week8_boosting.py", ["--trials", trials, "--boot", boot])
        if rc:
            sys.exit(rc)
        step("WEEK 9 — calibrate, check subgroups, lock p(s)")
        rc, _ = run("scripts/17_week9_calibrate_lock.py", ["--boot", boot,
                                                           "--production", args.production])
        if rc:
            sys.exit(rc)
        if args.final_test or args.demo:
            step("WEEK 9 — the one look at September")
            rc, _ = run("scripts/17_week9_calibrate_lock.py", ["--final-test", "--boot", boot])
            if rc:
                sys.exit(rc)

    if not args.demo:
        step("DASHBOARD")
        for path in ("scripts/12_export_dashboard_data.py", "scripts/18_export_model_dashboard.py",
                     "scripts/14_build_single_file.py"):
            rc, _ = run(path)
            if rc:
                sys.exit(rc)
    docs = os.environ.get("ABS_DOCS_DIR", "docs")
    print("\n" + "=" * 78)
    print(f"Next: read {docs}/WEEK8_REPORT.md with WEEK8.md beside it, then "
          f"{docs}/WEEK9_REPORT.md with WEEK9.md.")
    if not (args.final_test or args.demo or test_opened):
        print("September is still sealed. When the decision is final:")
        print("    python scripts/17_week9_calibrate_lock.py --final-test")
    print("Demo script for the checkpoint: docs/CHECKPOINT_DEMO.md")


if __name__ == "__main__":
    main()
