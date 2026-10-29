#!/usr/bin/env python3
"""
13_define_cohort.py — Week 6. Decide which rows the model trains on, and freeze it.

Week 5 built a table of every pitch. Week 6 answers a narrower question: which of
those rows is the model allowed to learn from, and which window of time does each
one belong to? After this week, "the cohort" means one specific file, and changing
it means re-running every week that comes after.

Three things happen here, all of them decisions rather than computations:

  1. INCLUSION RULES. Each rule is a function with a written justification, applied
     in order, with the row count logged after each one. Reviewers ask "how did you
     get from 700,000 pitches to your sample?" and this is the answer, as code.

  2. THE TEMPORAL SPLIT. Train on early months, validate on a later one, test on
     the last. NOT a random split: you are predicting future challenges from past
     ones, and a random split lets the model see September while training for
     September. The dates live in a config file so every later script reads the
     same ones.

  3. MISSINGNESS. For every column: how much is missing, and what you decided to
     do about it. Some nulls are not missing data at all — `is_overturned` is null
     on unchallenged pitches by construction — and the report separates those from
     real gaps.

    python scripts/13_define_cohort.py
    python scripts/13_define_cohort.py --dry-run     # report, write nothing
    ABS_DATA_DIR=data_demo python scripts/13_define_cohort.py

Output: data/processed/cohort_2026.parquet   (the frozen cohort)
        config/cohort.json                   (rules + split dates, machine-readable)
        docs/COHORT.md                       (the write-up)
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
PROCESSED = DATA_DIR / "processed"
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
CONFIG = Path(os.environ.get("ABS_CONFIG_DIR", "config"))

# ---------------------------------------------------------------------------
# EDIT THIS. Everything about the cohort is decided here and nowhere else.
# ---------------------------------------------------------------------------
SPLIT = {
    # Train on the first chunk of the season, validate on the next, test on the
    # last. Tune only the validation window while developing; the test window is
    # touched exactly once, at the very end of Week 8.
    "train": ("2026-03-01", "2026-07-31"),
    "validate": ("2026-08-01", "2026-08-31"),
    "test": ("2026-09-01", "2026-12-31"),
}

# Each rule: (name, function returning the mask of rows to KEEP, why)
RULES = [
    ("called_pitches_only",
     lambda d: d["is_called_pitch"],
     "A challenge can only follow a taken pitch. Swings are not challengeable, "
     "so they carry no information about the decision."),

    ("drop_position_player_pitching",
     lambda d: ~d["position_player_pitching"].fillna(False).astype(bool),
     "Challenges are not available while a position player pitches, so these "
     "rows have a structurally impossible outcome."),

    ("require_pitch_location",
     lambda d: d["plate_x"].notna() & d["plate_z"].notna(),
     "Without a location there is no distance-from-zone feature, which is the "
     "model's primary input."),

    ("require_abs_zone_from_height",
     lambda d: d["abs_zone_source"].eq("batter_height")
     if "abs_zone_source" in d.columns else d.index.to_series().notna(),
     "Rows falling back to Statcast's zone have a different zone definition. "
     "Keeping them would mix two measurement systems in one feature."),

    ("require_challenge_inventory",
     lambda d: d["challenges_remaining_batting"].notna()
     & d["challenges_remaining_fielding"].notna(),
     "The inventory is the MDP's state variable; a row without it cannot be "
     "used for either the behaviour model or the policy."),

    ("regular_season_only",
     lambda d: d["game_date"] <= pd.Timestamp("2026-10-01"),
     "Postseason challenge behaviour may differ and the sample is small. "
     "Revisit in Spring if you want a postseason transportability test."),
]

# How to treat each column's nulls. 'structural' means the null is meaningful and
# correct, not a gap. Anything not listed gets flagged for a decision.
MISSINGNESS = {
    "is_overturned": ("structural", "Null on unchallenged pitches by design."),
    "challenger_role": ("structural", "Only exists once a challenge happened."),
    "challenger_id": ("structural", "Same."),
    "challenger_name": ("structural", "Same."),
    "challenging_side": ("structural", "Same."),
    "challenge_call_code": ("structural", "Same."),
    "challenge_description": ("structural", "Same."),
    "on_1b": ("structural", "Null means the base is empty."),
    "on_2b": ("structural", "Null means the base is empty."),
    "on_3b": ("structural", "Null means the base is empty."),
    "leverage_index": ("impute_or_flag",
                       "Plate-appearance grain; a few pitches may not match. "
                       "Add a missing-indicator rather than dropping the row."),
    "catcher_strike_rate": ("impute_or_flag",
                            "Catchers below the qualifying threshold have no "
                            "season framing number. Flag, don't drop."),
    "catcher_runs_extra_strikes": ("impute_or_flag", "Same."),
    "ump_accuracy": ("impute_or_flag", "Umpires with few games. Flag, don't drop."),
    "ump_borderline_accuracy": ("impute_or_flag", "Same."),
    "ump_missed_strike_rate": ("impute_or_flag", "Same."),
    "ump_missed_ball_rate": ("impute_or_flag", "Same."),
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


def apply_rules(df):
    """Apply inclusion rules in order, logging the cost of each."""
    log = []
    kept = df
    start = len(kept)
    print(f"\n  {'rule':<34} {'kept':>10} {'dropped':>10}  {'% of start':>10}")
    print(f"  {'-' * 34} {'-' * 10} {'-' * 10}  {'-' * 10}")
    print(f"  {'(start)':<34} {start:>10,} {'':>10}  {100.0:>9.1f}%")

    for name, fn, why in RULES:
        try:
            mask = fn(kept)
        except Exception as e:
            print(f"  {name:<34} SKIPPED — {type(e).__name__}: {e}")
            log.append({"rule": name, "status": "skipped", "reason": str(e),
                        "justification": why})
            continue
        before = len(kept)
        kept = kept[mask.fillna(False).astype(bool)]
        dropped = before - len(kept)
        print(f"  {name:<34} {len(kept):>10,} {dropped:>10,}  "
              f"{100*len(kept)/max(start,1):>9.1f}%")
        log.append({"rule": name, "kept": int(len(kept)), "dropped": int(dropped),
                    "justification": why})
    return kept.reset_index(drop=True), log, start


def assign_split(df):
    """Label each row train / validate / test by date. Never randomly."""
    d = pd.to_datetime(df["game_date"])
    split = pd.Series("unassigned", index=df.index, dtype=object)
    for name, (lo, hi) in SPLIT.items():
        m = (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
        split[m] = name
    df = df.copy()
    df["split"] = split
    return df


def missingness_report(df):
    rows = []
    for c in df.columns:
        n_null = int(df[c].isna().sum())
        if n_null == 0:
            continue
        kind, why = MISSINGNESS.get(c, ("DECIDE", "Not yet classified — decide "
                                               "whether to drop, impute, or flag."))
        rows.append({"column": c, "n_null": n_null,
                     "pct_null": round(100 * n_null / len(df), 2),
                     "treatment": kind, "note": why})
    return pd.DataFrame(rows).sort_values("pct_null", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default=str(PROCESSED / "challenge_inventory_2026.parquet"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report but write nothing")
    args = ap.parse_args()

    df = read_any(args.table)
    if df is None:
        sys.exit(f"No joined table at {args.table}. Run scripts/run_week5.py first.")

    for c in ("is_called_pitch", "was_challenged", "is_challenge_opportunity",
              "position_player_pitching", "called_strike", "abs_strike",
              "would_be_overturned", "is_reasonable_pitch"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    df["game_date"] = pd.to_datetime(df["game_date"])

    print("=" * 76)
    print(f"WEEK 6 — DEFINING THE COHORT    ({len(df):,} pitches in)")
    print("=" * 76)

    print("\n1. INCLUSION RULES")
    cohort, rule_log, start = apply_rules(df)

    print("\n2. TEMPORAL SPLIT")
    cohort = assign_split(cohort)
    counts = cohort["split"].value_counts()
    for name in ("train", "validate", "test", "unassigned"):
        n = int(counts.get(name, 0))
        if name == "unassigned" and n == 0:
            continue
        lo, hi = SPLIT.get(name, ("", ""))
        ch = int(cohort.loc[cohort["split"] == name, "was_challenged"].sum())
        flag = "   <-- CHECK: rows outside every window" if name == "unassigned" else ""
        print(f"  {name:<12} {n:>9,} pitches  {ch:>7,} challenges   {lo} .. {hi}{flag}")

    train_ch = int(cohort.loc[cohort["split"] == "train", "was_challenged"].sum())
    if train_ch < 500:
        print(f"\n  NOTE: only {train_ch:,} challenges in the training window. "
              f"That is thin for a model with categorical features — consider "
              f"widening the train window or pooling AAA data.")

    print("\n3. MISSINGNESS")
    miss = missingness_report(cohort)
    if miss.empty:
        print("  No nulls anywhere in the cohort.")
    else:
        print(miss.to_string(index=False))
        undecided = miss[miss["treatment"] == "DECIDE"]
        if len(undecided):
            print(f"\n  {len(undecided)} column(s) need a decision. Add them to "
                  f"MISSINGNESS at the top of this script:")
            for c in undecided["column"]:
                print(f"    - {c}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    # ---------------------------------------------------------------- freeze
    out = write_any(cohort, PROCESSED / "cohort_2026.parquet")
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            cohort[["game_pk", "at_bat_number", "pitch_number"]],
            index=False).values.tobytes()).hexdigest()[:16]

    CONFIG.mkdir(parents=True, exist_ok=True)
    cfg = {
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_table": str(args.table),
        "rows_in": int(start), "rows_out": int(len(cohort)),
        "retention_pct": round(100 * len(cohort) / max(start, 1), 2),
        "cohort_fingerprint": digest,
        "split": SPLIT,
        "split_counts": {k: int(v) for k, v in counts.items()},
        "rules": rule_log,
        "missingness": miss.to_dict("records") if not miss.empty else [],
    }
    (CONFIG / "cohort.json").write_text(json.dumps(cfg, indent=2))

    DOCS.mkdir(parents=True, exist_ok=True)
    md = [f"# Analytic Cohort — 2026\n",
          f"Frozen {cfg['frozen_at'][:16]} by `scripts/13_define_cohort.py`. "
          f"Fingerprint `{digest}`.\n",
          f"**{len(cohort):,} rows** from {start:,} pitches "
          f"({cfg['retention_pct']}% retained), {int(cohort['was_challenged'].sum()):,} "
          f"of them challenged.\n",
          "Changing anything here means re-running Weeks 7 onward. If you do change "
          "it, note the date and the reason — the fingerprint above is how you prove "
          "which cohort produced which result.\n",
          "## Inclusion rules\n",
          "| # | Rule | Rows kept | Dropped | Why |",
          "|---:|---|---:|---:|---|"]
    for i, r in enumerate(rule_log, 1):
        if r.get("status") == "skipped":
            md.append(f"| {i} | `{r['rule']}` | skipped | — | {r['justification']} |")
        else:
            md.append(f"| {i} | `{r['rule']}` | {r['kept']:,} | {r['dropped']:,} | "
                      f"{r['justification']} |")

    md += ["", "## Temporal split\n",
           "Split on **time, not at random**. The model predicts future challenges "
           "from past ones; a random split would let it train on September while "
           "being tested on September.\n",
           "| Window | Dates | Pitches | Challenges |", "|---|---|---:|---:|"]
    for name in ("train", "validate", "test"):
        lo, hi = SPLIT[name]
        n = int(counts.get(name, 0))
        ch = int(cohort.loc[cohort["split"] == name, "was_challenged"].sum())
        md.append(f"| {name} | {lo} → {hi} | {n:,} | {ch:,} |")
    md += ["", "Tune on **validate** only. The **test** window is read exactly once, "
           "at the end of Week 8.\n"]

    md += ["## Missingness\n"]
    if miss.empty:
        md.append("No nulls in the cohort.\n")
    else:
        md += ["`structural` means the null is correct and meaningful, not a gap.\n",
               "| Column | Null % | Treatment | Note |", "|---|---:|---|---|"]
        for r in miss.itertuples():
            md.append(f"| `{r.column}` | {r.pct_null}% | {r.treatment} | {r.note} |")
        md.append("")

    (DOCS / "COHORT.md").write_text("\n".join(md))

    print(f"\n{'=' * 76}")
    print(f"  cohort      -> {out}  ({len(cohort):,} rows)")
    print(f"  config      -> {CONFIG/'cohort.json'}")
    print(f"  write-up    -> {DOCS/'COHORT.md'}")
    print(f"  fingerprint -> {digest}")
    print("""
Next: commit and tag this, so you can always point at the exact cohort behind a
result.

    git add config/cohort.json docs/COHORT.md
    git commit -m "Week 6: freeze analytic cohort"
    git tag v0.2-cohort""")
    print("=" * 76)


if __name__ == "__main__":
    main()
