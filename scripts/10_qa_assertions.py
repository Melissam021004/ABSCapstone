#!/usr/bin/env python3
"""
10_qa_assertions.py — Week 5 step 8. The table either passes or it doesn't.

Runs ~20 assertions against the analysis-ready table and exits non-zero if any
CRITICAL one fails, so you can wire it into a Makefile, a git hook or CI and
stop a broken table from reaching Week 6.

Two severities:

    CRITICAL  the table is wrong. Do not model on it. Exit code 1.
    WARN      the table is probably fine but a number is outside the range we
              expected. Look, decide, and write down why it's acceptable.

Every failure writes the offending rows to docs/qa_failures/<check_id>.csv, so
"13 rows violate the inventory bound" comes with the 13 rows attached.

    python scripts/10_qa_assertions.py
    python scripts/10_qa_assertions.py --strict      # WARN also fails the run
    ABS_DATA_DIR=data_demo python scripts/10_qa_assertions.py

Output: docs/QA_REPORT.md, docs/qa_failures/*.csv, exit code 0 or 1
"""

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
PROCESSED = DATA_DIR / "processed"
RAW = DATA_DIR / "raw"
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
FAIL_DIR = DOCS / "qa_failures"

START_CHALLENGES = 2
CRITICAL, WARN = "CRITICAL", "WARN"

results = []


def read_any(path):
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    csv = p.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def check(cid, severity, title, passed, detail, meaning="", offenders=None):
    """Record one assertion. `offenders` is a DataFrame of the bad rows."""
    saved = None
    if not passed and offenders is not None and len(offenders):
        FAIL_DIR.mkdir(parents=True, exist_ok=True)
        saved = FAIL_DIR / f"{cid}.csv"
        offenders.head(500).to_csv(saved, index=False)
    results.append({"id": cid, "severity": severity, "title": title,
                    "passed": bool(passed), "detail": detail,
                    "meaning": meaning, "offenders": str(saved) if saved else ""})
    mark = "PASS" if passed else ("FAIL" if severity == CRITICAL else "WARN")
    print(f"  [{mark:>4}] {cid:<26} {detail}")
    if not passed and meaning:
        print(f"         -> {meaning}")
    if saved:
        print(f"         -> offending rows: {saved}")


def skip(cid, reason):
    results.append({"id": cid, "severity": "SKIP", "title": cid, "passed": True,
                    "detail": reason, "meaning": "", "offenders": ""})
    print(f"  [SKIP] {cid:<26} {reason}")


def has(df, *cols):
    return all(c in df.columns for c in cols)


# --------------------------------------------------------------------------
def run_checks(df, raw_challenges):
    n = len(df)

    # ---- structural ----
    check("TABLE_NOT_EMPTY", CRITICAL, "Table has rows", n > 0,
          f"{n:,} rows", "Script 08 produced nothing. Check its inputs.")
    if n == 0:
        return

    if has(df, "game_pk", "at_bat_number", "pitch_number"):
        key = ["game_pk", "at_bat_number", "pitch_number"]
        dup = df[df.duplicated(subset=key, keep=False)]
        check("UNIQUE_PITCH_KEY", CRITICAL, "One row per pitch", len(dup) == 0,
              f"{len(dup):,} duplicate key rows",
              "The same pitch appears twice. A merge fanned out — every count and "
              "rate downstream is inflated.", dup.sort_values(key))
    else:
        skip("UNIQUE_PITCH_KEY", "key columns missing")

    if has(df, "balls", "strikes"):
        bad = df[(df["balls"] < 0) | (df["balls"] > 3) |
                 (df["strikes"] < 0) | (df["strikes"] > 2)]
        check("COUNT_IN_RANGE", CRITICAL, "Counts are legal", len(bad) == 0,
              f"{len(bad):,} rows outside balls 0-3 / strikes 0-2",
              "A count outside these bounds means the pre-pitch state is wrong.", bad)

    # ---- the challenge inventory: the MDP's state variable ----
    if has(df, "challenges_remaining_batting", "challenges_remaining_fielding"):
        lo = df[["challenges_remaining_batting", "challenges_remaining_fielding"]].min().min()
        hi = df[["challenges_remaining_batting", "challenges_remaining_fielding"]].max().max()
        bad = df[(df["challenges_remaining_batting"] < 0) |
                 (df["challenges_remaining_batting"] > START_CHALLENGES) |
                 (df["challenges_remaining_fielding"] < 0) |
                 (df["challenges_remaining_fielding"] > START_CHALLENGES)]
        check("INVENTORY_IN_BOUNDS", CRITICAL,
              f"Challenges remaining within 0-{START_CHALLENGES}", len(bad) == 0,
              f"observed range {lo}-{hi}, {len(bad):,} violations",
              "Teams start with 2 and the extra-inning grant only restores a side to "
              "1, so nothing can ever exceed 2. This is r in your MDP.", bad)

        if has(df, "inning"):
            first = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]) \
                      .groupby("game_pk").head(1)
            bad = first[(first["challenges_remaining_batting"] != START_CHALLENGES) |
                        (first["challenges_remaining_fielding"] != START_CHALLENGES)]
            check("INVENTORY_STARTS_AT_2", CRITICAL, "Every game starts at 2 each",
                  len(bad) == 0, f"{len(bad):,} games start wrong",
                  "The per-game reset is broken — inventory is carrying across games.",
                  bad)
    else:
        skip("INVENTORY_IN_BOUNDS", "challenges_remaining columns missing")

    # ---- opportunities ----
    if has(df, "is_challenge_opportunity_batting", "challenges_remaining_batting"):
        bad = df[df["is_challenge_opportunity_batting"] &
                 (df["challenges_remaining_batting"] <= 0)]
        bad2 = df[df["is_challenge_opportunity_fielding"] &
                  (df["challenges_remaining_fielding"] <= 0)]
        tot = len(bad) + len(bad2)
        check("NO_OPPORTUNITY_AT_ZERO", CRITICAL,
              "No opportunity for a side with zero challenges", tot == 0,
              f"{tot:,} violations",
              "An opportunity requires a challenge in hand. If this fires, the "
              "denominator for your behaviour model is wrong.",
              pd.concat([bad, bad2]))

    if has(df, "was_challenged", "challenging_side",
           "challenges_remaining_batting", "challenges_remaining_fielding"):
        ch = df[df["was_challenged"]].copy()
        if len(ch):
            rem = np.where(ch["challenging_side"] == ch["batting_side"],
                           ch["challenges_remaining_batting"],
                           ch["challenges_remaining_fielding"])
            bad = ch[rem <= 0]
            check("NO_CHALLENGE_AT_ZERO", CRITICAL,
                  "Nobody challenges with zero left", len(bad) == 0,
                  f"{len(bad):,} challenges made with no challenge remaining",
                  "Either the inventory walk is wrong or a challenge event was "
                  "mis-attributed to the wrong side.", bad)

    # ---- challenged pitches ----
    if has(df, "was_challenged"):
        ch = df[df["was_challenged"]]
        n_ch = len(ch)
        check("SOME_CHALLENGES", CRITICAL, "Challenges are present", n_ch > 0,
              f"{n_ch:,} challenged pitches",
              "No challenges matched onto pitches at all — the join failed.")

        if n_ch:
            if has(df, "is_called_pitch"):
                bad = ch[~ch["is_called_pitch"].astype(bool)]
                check("CHALLENGED_ARE_CALLED", CRITICAL,
                      "Challenged pitches are called pitches", len(bad) == 0,
                      f"{len(bad):,} challenges on swung-at pitches",
                      "You can only challenge a taken pitch. These are join errors — "
                      "most likely pitch_number is off by one.", bad)

            # These two are about DATA GAPS rather than logic errors. The Stats
            # API genuinely omits the challenger on a small number of reviews,
            # and nothing in the pipeline can recover it. Demanding zero would
            # halt the whole chain over rows that cannot be fixed — so they fail
            # critically only when the share is big enough to mean the
            # extraction itself is broken, and warn otherwise.
            if has(df, "challenger_role"):
                ok_roles = {"batter", "catcher", "pitcher"}
                bad = ch[~ch["challenger_role"].isin(ok_roles)]
                share = len(bad) / max(n_ch, 1)
                check("CHALLENGER_ROLE_VALID",
                      CRITICAL if share > 0.02 else WARN,
                      "Challenger is batter, catcher or pitcher", len(bad) == 0,
                      f"{len(bad):,} of {n_ch:,} ({share:.1%}) with a missing or "
                      f"invalid role",
                      "Only those three may challenge, so a null role means the "
                      "API payload had no reviewDetails.player. A few is normal "
                      "and they drop out of any role-split analysis; above 2% "
                      "suspect the extraction in script 02.", bad)

            if has(df, "is_overturned"):
                bad = ch[ch["is_overturned"].isna()]
                share = len(bad) / max(n_ch, 1)
                check("OUTCOME_PRESENT", CRITICAL if share > 0.01 else WARN,
                      "Every challenge has an outcome", len(bad) == 0,
                      f"{len(bad):,} of {n_ch:,} ({share:.1%}) missing is_overturned",
                      "is_overturned is your model target, so these rows cannot "
                      "train anything and are excluded from every rate on this "
                      "page. A handful is tolerable; more means script 02 is "
                      "mis-reading the review payload.", bad)

            # failed-challenge budget per game-side
            if has(df, "challenging_side", "inning", "is_overturned"):
                rows = []
                for (pk, side), g in ch.groupby(["game_pk", "challenging_side"]):
                    fails = int((~g["is_overturned"].fillna(True).astype(bool)).sum())
                    max_inn = int(df.loc[df["game_pk"] == pk, "inning"].max())
                    budget = START_CHALLENGES + max(0, max_inn - 9)
                    if fails > budget:
                        rows.append({"game_pk": pk, "side": side,
                                     "failed_challenges": fails, "budget": budget,
                                     "max_inning": max_inn})
                bad = pd.DataFrame(rows)
                check("FAILED_BUDGET", CRITICAL,
                      "No side exceeds its failed-challenge budget", len(bad) == 0,
                      f"{len(bad):,} game-sides over budget",
                      "A side can lose at most 2 in regulation plus 1 per extra "
                      "inning. Over budget means the inventory never decremented.",
                      bad)

            # the zone implementation's own test
            if has(df, "would_be_overturned", "is_overturned"):
                sub = ch[ch["is_overturned"].notna()]
                if len(sub):
                    agree = float((sub["would_be_overturned"].astype(bool) ==
                                   sub["is_overturned"].astype(bool)).mean())
                    mism = sub[sub["would_be_overturned"].astype(bool) !=
                               sub["is_overturned"].astype(bool)]
                    check("ZONE_PREDICTS_OUTCOME", CRITICAL,
                          "ABS zone reproduces actual challenge outcomes",
                          agree >= 0.85, f"{agree:.1%} agreement (need >= 85%)",
                          "Your reconstructed ABS zone disagrees with what ABS "
                          "actually ruled. The zone constants, the ball-radius rule "
                          "or the batter heights are wrong — and every distance "
                          "feature inherits the error.", mism)

    # ---- join completeness against the source file ----
    if raw_challenges is not None and has(df, "was_challenged"):
        src = len(raw_challenges)
        matched = int(df["was_challenged"].sum())
        rate = matched / src if src else 0
        check("JOIN_RATE", CRITICAL, "Challenge events matched onto pitches",
              rate >= 0.95, f"{matched:,}/{src:,} matched ({rate:.1%})",
              "Unmatched challenges are usually the at_bat_number off-by-one "
              "(Statcast 1-based vs Stats API 0-based) or pitch numbering that "
              "counts pickoffs.")
    else:
        skip("JOIN_RATE", "raw challenge file not found for comparison")

    # ---- coverage ----
    for cid, col, label, thresh in [
        ("UMPIRE_COVERAGE", "hp_umpire_id", "home-plate umpire", 0.95),
        ("HEIGHT_COVERAGE", "batter_height_ft", "batter height", 0.95),
        ("LEVERAGE_COVERAGE", "leverage_index", "leverage index", 0.90),
    ]:
        if col in df.columns:
            cov = float(df[col].notna().mean())
            check(cid, WARN, f"{label} coverage", cov >= thresh,
                  f"{cov:.1%} populated (want >= {thresh:.0%})",
                  f"A sparse {label} column means that covariate is mostly missing "
                  f"from the model.")
        else:
            skip(cid, f"{col} not in table")

    if "abs_zone_source" in df.columns:
        share = float((df["abs_zone_source"] == "batter_height").mean())
        check("ZONE_FROM_HEIGHT", WARN, "ABS zone built from batter height",
              share >= 0.95, f"{share:.1%} of rows use batter height",
              "Rows falling back to Statcast's sz_top/sz_bot have an approximated "
              "zone. Run 07_player_bios.py and rebuild.")

    # ---- plausibility against published numbers ----
    if has(df, "was_challenged", "is_overturned"):
        ch = df[df["was_challenged"]]
        if len(ch) and ch["is_overturned"].notna().any():
            rate = float(ch["is_overturned"].mean())
            check("OVERTURN_BAND", WARN, "Overturn rate is plausible",
                  0.25 <= rate <= 0.75, f"{rate:.1%} (expected roughly 40-60%)",
                  "Far outside this band suggests mis-parsed outcomes.")

            if "challenger_role" in ch.columns:
                by = ch.groupby("challenger_role")["is_overturned"].mean()
                if {"catcher", "batter"} <= set(by.index):
                    check("CATCHERS_BEAT_BATTERS", WARN,
                          "Catchers out-challenge batters",
                          by["catcher"] > by["batter"],
                          f"catcher {by['catcher']:.1%} vs batter {by['batter']:.1%}",
                          "MLB has publicly reported catchers are the most "
                          "successful challengers. If your data says otherwise, "
                          "suspect the data before the finding.")

    if "is_challenge_opportunity" in df.columns:
        share = float(df["is_challenge_opportunity"].mean())
        check("OPPORTUNITY_SHARE", WARN, "Opportunity share near Savant's ~50%",
              0.35 <= share <= 0.65, f"{share:.1%} of pitches",
              "Savant reports about half of 2026 pitches are challenge "
              "opportunities. A big gap means the opportunity rule is off.")

    if "is_reasonable_pitch" in df.columns:
        share = float(df["is_reasonable_pitch"].mean())
        check("REASONABLE_SHARE", WARN, "'Reasonable pitch' share near ~5%",
              0.02 <= share <= 0.12, f"{share:.1%} of pitches",
              "Ours is a simplified version of Savant's definition, so some drift "
              "is expected — but not a lot.")

    # ---- the relationship the whole model rests on ----
    if has(df, "zone_dist_in", "was_challenged", "is_overturned"):
        ch = df[df["was_challenged"] & df["is_overturned"].notna()].copy()
        if len(ch) >= 40:
            ch["bucket"] = pd.cut(ch["zone_dist_in"].abs(),
                                  [0, 1, 2, 4, 8, np.inf],
                                  labels=["0-1", "1-2", "2-4", "4-8", "8+"])
            by = ch.groupby("bucket", observed=True)["is_overturned"].mean().dropna()
            decreasing = bool(len(by) >= 3 and by.iloc[0] > by.iloc[-1])
            check("OVERTURN_FALLS_WITH_DISTANCE", WARN,
                  "Overturn rate falls as pitches move from the zone edge",
                  decreasing,
                  " | ".join(f"{i}in {v:.0%}" for i, v in by.items()),
                  "Calls far from the edge are obvious, so they should almost never "
                  "be overturned. If this doesn't hold, the distance feature is "
                  "wrong and Week 7's model has nothing to learn from.")

    if "leverage_index" in df.columns and df["leverage_index"].notna().any():
        m = float(df["leverage_index"].mean())
        check("LEVERAGE_MEAN", WARN, "Mean leverage index near 1.0",
              0.75 <= m <= 1.35, f"mean {m:.3f}",
              "Leverage is scaled so the average plate appearance is 1.0.")


# --------------------------------------------------------------------------
def write_report(df_rows, n_rows, n_cols, strict):
    DOCS.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(df_rows)
    res.to_csv(DOCS / "qa_results.csv", index=False)

    real = res[res["severity"] != "SKIP"]
    crit_fail = real[(real["severity"] == CRITICAL) & (~real["passed"])]
    warn_fail = real[(real["severity"] == WARN) & (~real["passed"])]
    n_pass = int(real["passed"].sum())

    if len(crit_fail):
        verdict, emoji = "FAILED", "❌"
    elif len(warn_fail):
        verdict = "FAILED (strict)" if strict else "PASSED WITH WARNINGS"
        emoji = "❌" if strict else "⚠️"
    else:
        verdict, emoji = "PASSED", "✅"

    md = [f"# QA Report — `challenge_inventory_2026`\n",
          f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by "
          f"`scripts/10_qa_assertions.py`.\n",
          f"## {emoji} {verdict}\n",
          f"**{n_rows:,} rows × {n_cols} columns.** "
          f"{n_pass}/{len(real)} assertions passed "
          f"({len(crit_fail)} critical failures, {len(warn_fail)} warnings).\n"]

    if len(crit_fail):
        md.append("## Critical failures — do not model on this table\n")
        for r in crit_fail.itertuples():
            md.append(f"### `{r.id}` — {r.title}\n")
            md.append(f"- **Result:** {r.detail}")
            md.append(f"- **Why it matters:** {r.meaning}")
            if r.offenders:
                md.append(f"- **Offending rows:** `{r.offenders}`")
            md.append("")

    if len(warn_fail):
        md.append("## Warnings — look, decide, write down why it's acceptable\n")
        for r in warn_fail.itertuples():
            md.append(f"### `{r.id}` — {r.title}\n")
            md.append(f"- **Result:** {r.detail}")
            md.append(f"- **Why it matters:** {r.meaning}")
            if r.offenders:
                md.append(f"- **Offending rows:** `{r.offenders}`")
            md.append("")

    md.append("## All assertions\n")
    md.append("| Check | Severity | Result | Detail |")
    md.append("|---|---|---|---|")
    for r in res.itertuples():
        if r.severity == "SKIP":
            status = "skipped"
        else:
            status = "pass" if r.passed else ("**FAIL**" if r.severity == CRITICAL
                                              else "*warn*")
        md.append(f"| `{r.id}` | {r.severity} | {status} | "
                  f"{str(r.detail).replace('|', chr(92)+'|')} |")
    md.append("")
    md.append("---\n")
    md.append("Re-run with `python scripts/10_qa_assertions.py`. "
              "Exit code is 1 on any critical failure, so this can gate a commit "
              "or a CI job.\n")

    (DOCS / "QA_REPORT.md").write_text("\n".join(md))
    return verdict, len(crit_fail), len(warn_fail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default=str(PROCESSED / "challenge_inventory_2026.parquet"))
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures too")
    args = ap.parse_args()

    df = read_any(args.table)
    if df is None:
        sys.exit(f"No table at {args.table}. Run 08_build_challenge_inventory.py first.")
    raw_ch = read_any(RAW / "challenges" / "abs_challenges_2026.parquet")

    for c in ("was_challenged", "is_called_pitch", "is_challenge_opportunity",
              "is_challenge_opportunity_batting", "is_challenge_opportunity_fielding",
              "is_reasonable_pitch", "abs_strike", "would_be_overturned"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    print("=" * 74)
    print(f"QA ASSERTIONS — {len(df):,} rows x {len(df.columns)} columns")
    print("=" * 74)
    run_checks(df, raw_ch)

    verdict, n_crit, n_warn = write_report(results, len(df), len(df.columns),
                                           args.strict)
    print("=" * 74)
    print(f"{verdict}   ({n_crit} critical, {n_warn} warnings)")
    print(f"Report -> {DOCS/'QA_REPORT.md'}")
    if n_crit or n_warn:
        print(f"Failing rows -> {FAIL_DIR}/")
    print("=" * 74)

    if n_crit or (args.strict and n_warn):
        sys.exit(1)


if __name__ == "__main__":
    main()
