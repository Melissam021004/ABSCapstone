#!/usr/bin/env python3
"""
run_week11.py — Week 11 in one command, tests first.

    python scripts/run_week11.py                  # tests -> the MDP -> the dashboard
    python scripts/run_week11.py --demo           # all of it on a simulated season
    python scripts/run_week11.py --sims 5000      # faster simulations (default 20,000 games a rule)

Order, and why:
  1. TESTS — the answers that must hold in any baseball game (p = 1 always challenges,
     p = 0 never, r = 0 has no option, a tied 9th has a lower bar than a blowout, the
     Bellman equation, walk-offs, the extra-inning rule, Tango's leverage), on a toy
     game with no data. Plus Week 10's win-probability checks, which V(s, r) builds on.
  2. scripts/21_week11_mdp.py — the pitch-level game, Tango's LI, V(s, r) for every
     state, the thresholds, the run-production levels, the policies, the 2026
     decisions and the umpire tiers. Writes models/mdp/, docs/WEEK11_REPORT.md,
     docs/week11_*.csv and figures/week11/.
  3. DASHBOARD — model_data.js and the single-file build, so the Progress tab shows
     Week 11 (real data only).

Needs: Week 9's locked p(s) (models/production/), Week 10's locked V(s)
(models/winprob/) and the raw Statcast pulls. Nothing here changes either lock.

--demo uses the *_demo folders; if Week 10 hasn't been run on the demo season yet, it
runs run_week10.py --demo first (a simulated season, a quick p(s) and V(s) for it).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    ("tests/test_mdp.py", "the MDP: known answers on a toy game"),
    ("tests/test_winprob.py", "win probability (Week 10) still right"),
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
    return proc, time.time() - t0


def step(title):
    print("\n" + "─" * 78 + f"\n{title}\n" + "─" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--sims", default="20000")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    print("=" * 78 + "\nWEEK 11 — THE CHALLENGE DECISION AS A MARKOV DECISION PROCESS\n" + "=" * 78)
    passed = []
    if not args.skip_tests:
        print("\nTESTS (stop here if any fails)")
        for path, what in TESTS:
            proc, secs = run(path, quiet=True)
            ok = proc.returncode == 0
            n_ok = len(re.findall(r"^\s+ok\s", proc.stdout or "", flags=re.M))
            print(f"  [{' pass ' if ok else ' FAIL '}] {what:<46} {path}  ({secs:.0f}s, {n_ok} checks)")
            if not ok:
                sys.exit(f"\n{path} failed — fix that before trusting any Week 11 number.")
            if path.endswith("test_mdp.py"):
                passed.append(n_ok)
    if passed:
        os.environ["ABS_W11_TESTS"] = f"{passed[0]}/{passed[0]}"

    if args.demo:
        os.environ.update(DEMO_ENV)
        need = [ROOT / "models_demo" / "winprob" / "manifest.json",
                ROOT / "models_demo" / "production"]
        if not all(p.exists() for p in need):
            step("DEMO — no demo season with a locked p(s) and V(s) yet: running run_week10.py --demo")
            proc, _ = run("scripts/run_week10.py", ["--demo", "--skip-tests", "--no-gbm"], quiet=True)
            if proc.returncode:
                sys.exit("\nThe demo season / Week 10 build failed (see above).")
            print("  data_demo/, models_demo/production and models_demo/winprob ready")

    step("THE MDP — game model, leverage, V(s, r), thresholds, policies, 2026 decisions (script 21)")
    extra = ["--sims", args.sims] + (["--no-figures"] if args.no_figures else [])
    proc, _ = run("scripts/21_week11_mdp.py", extra)
    if proc.returncode:
        sys.exit(proc.returncode)
    if not args.demo:
        step("DASHBOARD — the Progress tab picks up Week 11")
        for path in ("scripts/18_export_model_dashboard.py", "scripts/14_build_single_file.py"):
            proc, _ = run(path)
            if proc.returncode:
                sys.exit(proc.returncode)
    docs = os.environ.get("ABS_DOCS_DIR", "docs")
    print("\n" + "=" * 78)
    print(f"Next: read {docs}/WEEK11_REPORT.md with WEEK11.md beside it; the spec is docs/MDP_SPEC.md.")


if __name__ == "__main__":
    main()
