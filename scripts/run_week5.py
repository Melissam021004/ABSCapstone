#!/usr/bin/env python3
"""
run_week5.py — the whole of Week 5 in one command.

Week 5's deliverable is "analysis-ready schema, data dictionary, QA report".
This runs the four scripts that produce them, in order, stopping at the first
real failure, and finishes by printing the plan's "Done when" checklist with
each box ticked or not based on what actually happened.

    python scripts/run_week5.py

    python scripts/run_week5.py --demo     # generate a synthetic season first
                                           # and run the whole chain on it

    python scripts/run_week5.py --strict   # QA warnings also stop the run

    python scripts/run_week5.py --spine-from-table
                                           # re-join new challenge records on the
                                           # pitches already in the table (no re-pull)

Steps:
    08  build the analysis-ready table   (plan steps 1-6)
    09  generate the data dictionary     (plan step 7)
    10  run the QA assertions            (plan step 8)
    11  verify challenges_remaining      (the plan's "watch out for")

Nothing here touches the network. If a prerequisite pull is missing, the step
says which script produces it and stops.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))

STEPS = [
    ("08_build_challenge_inventory.py", [],
     "Build the analysis-ready table", "plan steps 1-6", True),
    ("09_data_dictionary.py", [],
     "Generate the data dictionary", "plan step 7", True),
    ("10_qa_assertions.py", [],
     "Run the QA assertions", "plan step 8", True),
    ("11_verify_challenges_remaining.py", ["--games", "3"],
     "Verify challenges_remaining", "the plan's 'watch out for'", False),
]


def run(script, extra):
    cmd = [sys.executable, str(HERE / script)] + extra
    t0 = time.time()
    proc = subprocess.run(cmd)
    return proc.returncode, time.time() - t0


def prerequisites(data_dir):
    """
    Which upstream pulls exist. Takes data_dir as an argument rather than
    reading the module-level constant, because --demo changes ABS_DATA_DIR
    at runtime and a constant resolved at import time would miss it.
    """
    def exists(*parts):
        base = data_dir.joinpath(*parts[:-1])
        return [p for p in (base / f"{parts[-1]}{e}" for e in (".parquet", ".csv"))
                if p.exists()]

    sc = data_dir / "raw" / "statcast"
    need = {
        "Statcast pitches (script 01)":
            sorted(sc.glob("statcast_2026_*")) if sc.exists() else [],
        "ABS challenges (script 02)":
            exists("raw", "challenges", "abs_challenges_2026"),
    }
    optional = {
        "Player bios (script 07) — the ABS zone needs these":
            exists("raw", "players", "player_bios_2026"),
        "Win probability (script 06)":
            exists("raw", "winprob", "winprob_2026"),
        "Catcher framing (script 04)":
            exists("raw", "savant", "catcher_framing_all_years"),
        "Umpire accuracy (script 05)":
            exists("raw", "umpires", "umpire_accuracy_derived_2026"),
    }
    missing = [k for k, v in need.items() if not v]
    return missing, need, optional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="generate a synthetic season and run the chain on it")
    ap.add_argument("--strict", action="store_true",
                    help="QA warnings also fail the run")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip step 11 (not recommended)")
    ap.add_argument("--spine-from-table", action="store_true",
                    help="script 08 takes the pitches from the existing table instead of the "
                         "raw Statcast files (re-join new challenge records without a re-pull)")
    args = ap.parse_args()

    if args.demo:
        os.environ["ABS_DATA_DIR"] = "data_demo"
        os.environ.setdefault("ABS_DOCS_DIR", "docs_demo")
        print("DEMO MODE — generating a synthetic season under data_demo/\n")
        rc, _ = run("make_demo_pitches.py", ["--out", "data_demo"])
        if rc != 0:
            sys.exit("Fixture generation failed.")
        print()

    data_dir = Path(os.environ.get("ABS_DATA_DIR", "data"))
    docs_dir = Path(os.environ.get("ABS_DOCS_DIR", "docs"))

    print("=" * 78)
    print("WEEK 5 — JOIN EVERYTHING")
    print(f"data: {data_dir}    docs: {docs_dir}")
    print("=" * 78)

    missing, need, optional = prerequisites(data_dir)
    print("\nPREREQUISITES")
    for k, v in need.items():
        print(f"  [{'  ok   ' if v else 'MISSING'}] {k}")
    for k, v in optional.items():
        print(f"  [{'  ok   ' if v else '   -   '}] {k}  (optional)")
    if missing:
        print(f"\nCannot run Week 5 without: {', '.join(missing)}")
        print("Those come from Week 4. Either finish the pulls, or try the whole")
        print("chain on synthetic data first:\n\n    python scripts/run_week5.py --demo")
        sys.exit(1)

    results = []
    for script, extra, label, maps_to, blocking in STEPS:
        if args.skip_verify and script.startswith("11"):
            print(f"\n{'─' * 78}\nSKIPPED  {script}  ({label})")
            results.append((script, label, None, 0))
            continue
        if script.startswith("10") and args.strict:
            extra = extra + ["--strict"]
        if script.startswith("08") and args.spine_from_table:
            extra = extra + ["--spine-from-table"]

        print(f"\n{'─' * 78}")
        print(f"RUNNING  {script}   — {label}  ({maps_to})")
        print("─" * 78)
        rc, secs = run(script, extra)
        results.append((script, label, rc, secs))

        if rc != 0 and blocking:
            print(f"\n{'=' * 78}")
            print(f"STOPPED — {script} exited {rc}.")
            if script.startswith("10"):
                print(f"The QA assertions failed. Read {docs_dir/'QA_REPORT.md'} and")
                print(f"the offending rows in {docs_dir/'qa_failures'}/ before going on.")
                print("Week 6 builds on this table; a broken table poisons everything")
                print("downstream and you will not notice until November.")
            print("=" * 78)
            sys.exit(rc)

    # ---------------------------------------------------------------- summary
    print(f"\n{'=' * 78}")
    print("WEEK 5 — DONE WHEN")
    print("=" * 78)

    inv = any((data_dir / "processed").glob("challenge_inventory_2026.*")) \
        if (data_dir / "processed").exists() else False
    dd = (docs_dir / "DATA_DICTIONARY.md").exists()
    qa = (docs_dir / "QA_REPORT.md").exists()
    ver = (docs_dir / "VERIFICATION_CHALLENGES_REMAINING.md").exists()
    qa_rc = next((rc for s, _, rc, _ in results if s.startswith("10")), 1)
    ver_rc = next((rc for s, _, rc, _ in results if s.startswith("11")), 1)

    boxes = [
        (inv, "One analysis-ready parquet file, one row per pitch"),
        (ver and ver_rc == 0,
         "challenges_remaining derived and cross-checked "
         "(still open 3 games by hand)"),
        (qa and qa_rc == 0,
         "Challenge opportunity flags, counts near Savant's ~50% / ~5%"),
        (qa and qa_rc == 0, "ABS-zone distance computed from batter height"),
        (dd, "Data dictionary covers every column"),
        (qa and qa_rc == 0, "QA assertions run as a script and all pass"),
    ]
    for ok, text in boxes:
        print(f"  [{'x' if ok else ' '}] {text}")

    print(f"\nARTIFACTS")
    for p in [data_dir / "processed" / "challenge_inventory_2026.parquet",
              data_dir / "processed" / "challenge_inventory_2026.csv",
              data_dir / "processed" / "join_ledger.csv",
              docs_dir / "DATA_DICTIONARY.md",
              docs_dir / "QA_REPORT.md",
              docs_dir / "VERIFICATION_CHALLENGES_REMAINING.md"]:
        if p.exists():
            print(f"  {p}")

    print(f"\nTIMING")
    for s, label, rc, secs in results:
        state = "skipped" if rc is None else ("ok" if rc == 0 else f"exit {rc}")
        print(f"  {s:<38} {secs:6.1f}s  {state}")

    all_done = all(ok for ok, _ in boxes)
    print("\n" + "=" * 78)
    if all_done:
        print("Week 5 is mechanically complete.")
        print("""
The one box no script can tick for you: open three games in the Savant game
feed and verify the challenges by eye. The worksheet is in
docs/VERIFICATION_CHALLENGES_REMAINING.md with the links already in it.

Two implementations agreeing proves they share no typo. It does not prove they
share no wrong assumption. That is what your eyes are for, and it is the single
cheapest insurance in this project.""")
    else:
        print("Some boxes are not ticked — see the list above.")
    print("=" * 78)


if __name__ == "__main__":
    main()
