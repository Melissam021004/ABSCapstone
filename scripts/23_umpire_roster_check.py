#!/usr/bin/env python3
"""
23_umpire_roster_check.py — why the analysis finds more plate umpires than MLB's staff.

MLB employs 76 full-time umpires, but the analysis counts every person who worked
home plate in a 2026 regular-season game. When a full-time umpire is on vacation,
injured or away, a Triple-A umpire from MLB's call-up list fills in (MLB.com, Aug
2025: 17 Triple-A umpires were on that list; fill-ins are needed "basically every
day"). Crews rotate positions, so a call-up who works a series usually takes the
plate too. This script shows the count and checks for data problems instead:

  1. every distinct home-plate umpire in the regular-season games, with plate games
  2. how many cleared the analysis cut (10+ plate games: Weeks 10–11's umpire tiers)
  3. data checks: one name under two ids, one id under two names, games with no
     plate umpire on record, non-regular-season games
  4. the split you'd expect if MLB's staff is 76: how many plate games the 76 busiest
     umpires worked vs everyone else — and, with --staff, the exact split from a list
     of the full-time umpires' names (one per line; copy them from MLB's umpire page)

    python scripts/23_umpire_roster_check.py
    python scripts/23_umpire_roster_check.py --staff docs/mlb_umpire_staff_2026.txt

Writes docs/UMPIRE_CHECK.md and docs/umpire_roster_check.csv.
"""

import argparse
import os
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
STAFF_SIZE = 76
MIN_GAMES = 10                     # the cut Weeks 10–11 use for umpire tiers


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.replace(".", " ").replace(",", " ").lower().split())


def load_games():
    for name in ("games_2026.parquet", "games_2026.csv"):
        p = DATA_DIR / "raw" / "challenges" / name
        if p.exists():
            g = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            return g, p
    raise SystemExit(f"No games file in {DATA_DIR / 'raw' / 'challenges'} — run scripts/02 first.")


def accuracy_by_umpire():
    """Week 10's difficulty-adjusted extra wrong calls per 100, per umpire (if the locks exist)."""
    try:
        from src.abs_features import add_features, load_cohort
        from src.abs_predict import load_production
        from src.abs_value import umpire_table, valuation_table
        from src.abs_winprob import load_winprob
        cohort = add_features(load_cohort(DATA_DIR))
        val = valuation_table(cohort, load_production(), load_winprob())
        ump, _ = umpire_table(val)
        return ump[["umpire_id", "games", "called", "wrong_per_game", "excess_per_100"]].rename(
            columns={"games": "games_with_called_pitches"})
    except Exception as e:                                   # noqa: BLE001
        print(f"  (accuracy columns skipped: {type(e).__name__}: {e})")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--staff", help="text file: full-time MLB umpires, one name per line")
    ap.add_argument("--no-accuracy", action="store_true", help="skip the accuracy columns (faster)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    demo = "demo" in str(DATA_DIR).lower()

    g, path = load_games()
    n_all = len(g)
    if "game_type" in g.columns:
        g = g[g["game_type"].astype(str).str.upper().eq("R")]
    g = g.drop_duplicates("game_pk")
    missing = g["hp_umpire_id"].isna() | g["hp_umpire"].isna()
    print(f"{path}: {n_all:,} rows, {len(g):,} regular-season games; {int(missing.sum())} with no plate umpire")
    gu = g[~missing].copy()
    gu["hp_umpire_id"] = pd.to_numeric(gu["hp_umpire_id"], errors="coerce").astype("Int64")
    per = (gu.groupby("hp_umpire_id")
             .agg(name=("hp_umpire", lambda s: s.mode().iat[0]), plate_games=("game_pk", "nunique"),
                  names_seen=("hp_umpire", lambda s: " | ".join(sorted(set(map(str, s))))),
                  first=("game_date", "min"), last=("game_date", "max"))
             .reset_index().sort_values("plate_games", ascending=False).reset_index(drop=True))
    per["rank"] = np.arange(1, len(per) + 1)
    n_ump = len(per)
    n_cut = int((per["plate_games"] >= MIN_GAMES).sum())

    # data checks
    per["key"] = per["name"].map(norm_name)
    dup_name = per[per.duplicated("key", keep=False)].sort_values("key")
    multi_name = per[per["names_seen"].str.contains(r"\|", regex=True)]

    # who is staff
    staff_note = ""
    if args.staff:
        names = [norm_name(x) for x in Path(args.staff).read_text().splitlines() if x.strip()]
        per["staff"] = per["key"].isin(set(names))
        unmatched = sorted(set(names) - set(per["key"]))
        staff_note = (f"{int(per['staff'].sum())} of the {len(names)} names in {args.staff} worked the plate; "
                      f"{len(unmatched)} didn't match a name in the data"
                      + (f" ({', '.join(unmatched[:8])}{' …' if len(unmatched) > 8 else ''})" if unmatched else ""))
    else:
        per["staff"] = per["rank"] <= STAFF_SIZE                 # a proxy: the 76 busiest
    if not args.no_accuracy:
        acc = accuracy_by_umpire()
        if acc is not None:
            acc["umpire_id"] = pd.to_numeric(acc["umpire_id"], errors="coerce").astype("Int64")
            per = per.merge(acc, left_on="hp_umpire_id", right_on="umpire_id", how="left").drop(columns="umpire_id")

    top, rest = per[per["staff"]], per[~per["staff"]]
    print(f"\nDistinct home-plate umpires in regular-season games: {n_ump}")
    print(f"  with {MIN_GAMES}+ plate games (the umpires Weeks 10–11 rank): {n_cut}")
    label = "full-time (from --staff)" if args.staff else f"the {STAFF_SIZE} busiest (a proxy for the full-time staff)"
    print(f"  {label}: {len(top)} umpires, {int(top['plate_games'].min()) if len(top) else 0}–"
          f"{int(top['plate_games'].max()) if len(top) else 0} plate games each "
          f"(median {top['plate_games'].median():.0f})")
    if len(rest):
        print(f"  everyone else: {len(rest)} umpires, {int(rest['plate_games'].min())}–"
              f"{int(rest['plate_games'].max())} plate games (median {rest['plate_games'].median():.0f}), "
              f"{int(rest['plate_games'].sum())} games in all ({rest['plate_games'].sum() / len(gu):.1%} of games)")
    if staff_note:
        print(f"  {staff_note}")
    print(f"\nData checks: {len(dup_name)} rows share a name with another id; {len(multi_name)} ids appear "
          f"under more than one spelling; {int(missing.sum())} games have no plate umpire")
    if len(dup_name):
        print(dup_name[["hp_umpire_id", "name", "plate_games"]].to_string(index=False))
    if "excess_per_100" in per.columns and len(rest) and per["excess_per_100"].notna().any():
        a = top["excess_per_100"].dropna()
        b = rest["excess_per_100"].dropna()
        print(f"\nAccuracy (extra wrong calls per 100 called pitches vs the league, adjusted for difficulty): "
              f"{'full-time' if args.staff else 'busiest ' + str(STAFF_SIZE)} {a.mean():+.2f} (n={len(a)}), "
              f"others {b.mean():+.2f} (n={len(b)})")

    DOCS.mkdir(parents=True, exist_ok=True)
    out = per.drop(columns=["key"])
    out.to_csv(DOCS / "umpire_roster_check.csv", index=False)
    L = ["# Umpire roster check", ""]
    if demo:
        L += ["> **DEMO DATA** — simulated umpires.", ""]
    L += [f"Source: `{path}` · regular-season games only · written by `scripts/23_umpire_roster_check.py`.", "",
          f"- **Distinct home-plate umpires:** {n_ump} (with {MIN_GAMES}+ plate games, the umpires Weeks 10–11 "
          f"rank: {n_cut}).",
          f"- **{label}:** {len(top)} umpires, median {top['plate_games'].median():.0f} plate games each.",
          (f"- **Everyone else:** {len(rest)} umpires, median {rest['plate_games'].median():.0f} plate games, "
           f"{rest['plate_games'].sum() / len(gu):.1%} of all games." if len(rest) else "- **Everyone else:** none."),
          f"- **Data checks:** {len(dup_name)} names under two ids, {len(multi_name)} ids under two spellings, "
          f"{int(missing.sum())} games with no plate umpire.", "",
          ("**Why the count is above 76.** " if n_ump > STAFF_SIZE else "**Context.** ") +
          "MLB's 76 full-time umpires are the staff; when one is on vacation, "
          "injured or away, a Triple-A umpire from MLB's call-up list fills in (MLB.com, August 2025: 17 umpires "
          "on that list, with fill-ins needed \"basically every day\"). Crews rotate positions, so a call-up "
          "who works a series usually takes the plate as well. Their calls are real calls that teams had to "
          "decide whether to challenge, so they belong in the analysis; the tiers already use only umpires "
          f"with {MIN_GAMES}+ plate games.", "",
          f"**For the write-up:** \"{n_ump} home-plate umpires worked 2026 regular-season games — MLB's "
          f"full-time staff plus Triple-A call-ups filling in; the umpire tiers use the {n_cut} with at least "
          f"{MIN_GAMES} plate games.\" Run with `--staff` (a text file of the {STAFF_SIZE} full-time names, one "
          "per line) to get the exact split.", "",
          "| rank | umpire | plate games | first | last |" + (" extra wrong /100 |" if "excess_per_100" in per.columns else ""),
          "|---|---|---|---|---|" + ("---|" if "excess_per_100" in per.columns else "")]
    for _, r in per.iterrows():
        row = f"| {r['rank']} | {r['name']} | {int(r['plate_games'])} | {r['first']} | {r['last']} |"
        if "excess_per_100" in per.columns:
            row += f" {r['excess_per_100']:+.2f} |" if pd.notna(r.get("excess_per_100")) else " — |"
        L.append(row)
    (DOCS / "UMPIRE_CHECK.md").write_text("\n".join(L) + "\n")
    print(f"\n→ {DOCS / 'UMPIRE_CHECK.md'} and {DOCS / 'umpire_roster_check.csv'}")


if __name__ == "__main__":
    main()
