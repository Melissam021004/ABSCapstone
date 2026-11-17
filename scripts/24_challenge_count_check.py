#!/usr/bin/env python3
"""
24_challenge_count_check.py — does the pipeline have every 2026 ABS challenge?

MLB's official totals for the 2026 regular season (Baseball-Reference's ABS
challenge analysis, from MLB data; the AP reported the same totals Oct 2, 2026):
    10,557 challenges in 2,429 games (4.35 a game), 5,685 overturned (53.9%)
    catchers 5,609 (58.6% overturned) · batters 4,766 (48.8%) · pitchers 182 (39.6%)

This script counts the challenges at each stage of the pipeline, by challenger
role and outcome, and shows where any shortfall happens:
    1. extracted by script 02   data/raw/challenges/abs_challenges_2026.parquet
    2. joined to a pitch        data/processed/challenge_inventory_2026.parquet
    3. in the analytic cohort   data/processed/cohort_2026.parquet
It then runs the test that points to the likely cause: if the overturn rate is
far HIGHER than usual when an upheld call would have ended the at-bat, and far
LOWER when an overturned call would have, the missing challenges are the ones
whose final call ended the plate appearance (strike three, ball four). Those are
the reviews the first version of script 02 never looked for.

    python scripts/24_challenge_count_check.py

Runs offline in a few seconds. Writes docs/CHALLENGE_COUNT_CHECK.md and
docs/challenge_count_check.csv.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.abs_features import _as_bool, implied_original_call, read_any  # noqa: E402

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
ROLES = ("catcher", "batter", "pitcher")
OFFICIAL = {"stage": "MLB official", "games": 2429, "challenges": 10557, "overturned": 5685,
            "catcher": 5609, "catcher_overturned": 3285, "batter": 4766, "batter_overturned": 2328,
            "pitcher": 182, "pitcher_overturned": 72}
OFFICIAL_MONTH = {3: 3.76, 4: 4.19, 5: 4.06, 6: 4.32, 7: 4.44, 8: 4.60, 9: 4.61}
SOURCE = ("Baseball-Reference, 2026 MLB ABS Challenge Analysis "
          "(https://www.baseball-reference.com/friv/abs-challenges.shtml); AP, Oct 2, 2026")


def regular(df):
    if df is None or "game_type" not in df.columns:
        return df
    return df[df["game_type"].astype(str).str.upper().eq("R")]


def counts(name, ch, n_games):
    """One stage's row: challenges, overturned, and both by role."""
    ov = _as_bool(ch["is_overturned"]) if "is_overturned" in ch.columns else pd.Series(np.nan, index=ch.index)
    role = ch["challenger_role"].astype(str).str.lower() if "challenger_role" in ch.columns else \
        pd.Series("", index=ch.index)
    row = {"stage": name, "games": int(n_games), "challenges": len(ch), "overturned": int(ov.eq(True).sum())}
    for r in ROLES:
        m = role.eq(r)
        row[r], row[f"{r}_overturned"] = int(m.sum()), int((m & ov.eq(True)).sum())
    return row


def ends_at_bat_test(ch):
    """
    Overturn rate by what the final call did to the at-bat. Needs the count before the
    pitch (Statcast's balls/strikes) and the original call (implied from who challenged).
      upheld_ends      an upheld call ends the at-bat (strike three stands, ball four stands)
      overturned_ends  an overturned call ends it (a strike becomes ball four, a ball strike three)
      neither          the at-bat goes on either way
    """
    need = {"balls", "strikes", "is_overturned"}
    if not need <= set(ch.columns):
        return None
    orig = implied_original_call(ch)
    ov = _as_bool(ch["is_overturned"])
    ok = orig.notna() & ov.notna()
    c = ch[ok]
    s = orig[ok].astype(bool).to_numpy()
    o = ov[ok].astype(bool).to_numpy()
    b, k = c["balls"].to_numpy(float), c["strikes"].to_numpy(float)
    upheld_ends = (s & (k == 2)) | (~s & (b == 3))
    over_ends = (s & (b == 3)) | (~s & (k == 2))
    group = np.where(upheld_ends & over_ends, "both", np.where(upheld_ends, "upheld_ends",
                     np.where(over_ends, "overturned_ends", "neither")))
    t = (pd.DataFrame({"group": group, "overturned": o})
         .groupby("group")["overturned"].agg(challenges="size", overturn_rate="mean"))
    final_ends = np.where(o, over_ends, upheld_ends)
    return t.reindex(["upheld_ends", "overturned_ends", "both", "neither"]).dropna(how="all"), \
        float(final_ends.mean())


def statcast_text_check(inv):
    """
    Plate appearances whose Statcast description mentions a challenge, vs the data.
    Descriptions also mention managers' replay challenges of plays (a tag, a force
    play, a catch), which aren't ABS challenges, so the ones with no challenge in the
    data are broken down by what the text says was challenged.
    """
    if inv is None or "des" not in inv.columns:
        return None
    key = ["game_pk", "at_bat_number"]
    txt = inv[inv["des"].astype(str).str.contains("challeng", case=False, na=False)]
    said = txt[key].drop_duplicates()
    if said.empty:
        return {"pas": 0}
    have = inv[_as_bool(inv["was_challenged"]).fillna(False).astype(bool)][key].drop_duplicates()
    m = said.merge(have.assign(h=1), on=key, how="left")
    miss = m[m["h"].isna()][key]
    des = txt.merge(miss, on=key)[key + ["des"]].drop_duplicates(key)["des"].astype(str)
    kinds = (des.str.extract(r"(?i)challenged\s*\(([^)]*)\)")[0].str.strip().str.lower()
             .fillna("(no reason in the text)"))
    return {"pas": len(said), "missing": len(miss), "kinds": kinds.value_counts().head(10),
            "examples": [d[:170] for d in des.head(3)]}


def savant_total():
    for p in sorted((DATA_DIR / "raw" / "savant").glob("abs_leaderboard_team-summary_for_*2026*.csv")):
        df = pd.read_csv(p)
        col = next((c for c in df.columns if "chal" in c.lower() and df[c].dtype.kind in "if"
                    and not any(w in c.lower() for w in ("pct", "rate", "exp"))), None)
        if col:
            return p.name, col, float(df[col].sum())
    return None


def missing_by_kind(T):
    """Official minus the last stage, by role and outcome (scaled to the same number of games)."""
    off, last = T.iloc[0], T.iloc[-1]
    scale = last["games"] / off["games"]
    out = []
    for r in ROLES:
        o_ov, o_fail = off[f"{r}_overturned"] * scale, (off[r] - off[f"{r}_overturned"]) * scale
        h_ov, h_fail = last[f"{r}_overturned"], last[r] - last[f"{r}_overturned"]
        out.append({"role": r, "overturned_missing": o_ov - h_ov, "failed_missing": o_fail - h_fail,
                    "overturned_kept": h_ov / o_ov if o_ov else np.nan,
                    "failed_kept": h_fail / o_fail if o_fail else np.nan})
    return pd.DataFrame(out).set_index("role")


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.1%}"


def main():
    warnings.filterwarnings("ignore")
    demo = "demo" in str(DATA_DIR).lower()
    print("Challenge count check" + (" (DEMO DATA)" if demo else ""))

    raw = read_any(DATA_DIR / "raw" / "challenges" / "abs_challenges_2026.parquet")
    games = regular(read_any(DATA_DIR / "raw" / "challenges" / "games_2026.parquet"))
    inv = regular(read_any(DATA_DIR / "processed" / "challenge_inventory_2026.parquet"))
    coh = read_any(DATA_DIR / "processed" / "cohort_2026.parquet")
    if raw is None and inv is None and coh is None:
        raise SystemExit(f"Nothing to check in {DATA_DIR} — run scripts 02, 08 and 13 first.")

    rows, notes = [OFFICIAL], []
    if raw is not None:
        if games is not None:
            raw = raw[raw["game_pk"].isin(set(games["game_pk"]))]
            n_g = games["game_pk"].nunique()
        else:
            n_g = raw["game_pk"].nunique()
        rows.append(counts("1. extracted (script 02)", raw, n_g))
        keyless = int(raw["pitch_number"].isna().sum()) if "pitch_number" in raw.columns else 0
        if keyless:
            notes.append(f"{keyless:,} extracted challenges have no pitch number, so script 08 can't join "
                         "them (the review was recorded on a separate event).")
        if "review_source" in raw.columns and raw["review_source"].notna().any():
            vc = raw["review_source"].value_counts()
            notes.append("Extracted by the November 2026 extractor; recorded on: "
                         + ", ".join(f"{k} {v:,}" for k, v in vc.items()) + ".")
        else:
            notes.append("Extracted by the FIRST version of script 02, which looked for reviews only "
                         "on the pitch event.")
    if inv is not None:
        ch = inv[_as_bool(inv["was_challenged"]).fillna(False).astype(bool)]
        rows.append(counts("2. joined to a pitch (script 08)", ch, inv["game_pk"].nunique()))
    if coh is not None:
        coh = regular(coh)
        cch = coh[_as_bool(coh["was_challenged"]).fillna(False).astype(bool)]
        rows.append(counts("3. in the cohort (script 13)", cch, coh["game_pk"].nunique()))
    T = pd.DataFrame(rows)
    T["per_game"] = T["challenges"] / T["games"].clip(lower=1)
    T["overturn_rate"] = T["overturned"] / T["challenges"].clip(lower=1)
    for r in ROLES:
        T[f"{r}_rate"] = T[f"{r}_overturned"] / T[r].clip(lower=1)
    T["share_of_official_rate"] = T["per_game"] / T.loc[0, "per_game"]

    last = T.iloc[-1]
    test_src = cch if coh is not None else (ch if inv is not None else None)
    ends = ends_at_bat_test(test_src) if test_src is not None else None
    text = statcast_text_check(inv)
    sav = savant_total()
    month = None
    if test_src is not None and "game_date" in test_src.columns:
        base = coh if coh is not None else inv
        mo = pd.to_datetime(base["game_date"]).dt.month
        g = base.assign(month=mo).groupby("month")["game_pk"].nunique()
        c = test_src.assign(month=pd.to_datetime(test_src["game_date"]).dt.month).groupby("month").size()
        month = pd.DataFrame({"games": g, "challenges": c}).fillna(0)
        month["per_game"] = month["challenges"] / month["games"].clip(lower=1)
        month["official_per_game"] = pd.Series(OFFICIAL_MONTH)
        month = month[month["games"] > 0]

    # ------------------------------------------------------------ print
    show = T[["stage", "games", "challenges", "per_game", "overturn_rate", "catcher", "catcher_rate",
              "batter", "batter_rate", "pitcher", "pitcher_rate"]]
    print(show.to_string(index=False, formatters={"per_game": "{:.2f}".format,
                                                  **{c: "{:.1%}".format for c in show.columns
                                                     if c.endswith("rate")}}))
    for n_ in notes:
        print(f"  {n_}")
    short = 1 - last["share_of_official_rate"]
    miss = missing_by_kind(T)
    if not demo:
        verdict = (f"{short:.0%} of the official challenge rate is missing by the last stage"
                   if short > 0.02 else "matches the official count (within 2%)")
        print(f"\n→ {verdict}.")
        print("\nMissing compared with MLB's totals, by role and outcome:")
        print(miss.to_string(formatters={"overturned_missing": "{:,.0f}".format,
                                         "failed_missing": "{:,.0f}".format,
                                         "overturned_kept": "{:.0%}".format, "failed_kept": "{:.0%}".format}))
    if ends is not None:
        t, share = ends
        print("\nOverturn rate by what the final call did to the at-bat:")
        print(t.to_string(formatters={"overturn_rate": "{:.1%}".format}))
        print(f"  share of challenges whose final call ended the at-bat: {share:.1%}")
    if text is not None:
        print(f"\nStatcast's play descriptions mention a challenge in {text['pas']:,} plate appearances"
              + (f"; {text['missing']:,} of them have no challenge in the data." if text.get("pas") else "."))
        if text.get("missing"):
            print("  what those descriptions say was challenged (managers' replay challenges of plays "
                  "aren't ABS challenges):")
            for k, v in text["kinds"].items():
                print(f"    {v:>5,}  {k}")
            for e in text["examples"]:
                print(f"    e.g. {e}")
    if sav:
        print(f"\nSavant team leaderboard ({sav[0]}, column '{sav[1]}'): {sav[2]:,.0f} challenges")
    if month is not None:
        print("\nChallenges per game by month:")
        print(month.to_string(formatters={"per_game": "{:.2f}".format, "official_per_game": "{:.2f}".format}))

    # ------------------------------------------------------------ write
    DOCS.mkdir(parents=True, exist_ok=True)
    T.to_csv(DOCS / "challenge_count_check.csv", index=False)
    L = ["# Challenge count check", ""]
    if demo:
        L += ["> **DEMO DATA** — a simulated season; it isn't expected to match MLB's totals.", ""]
    L += [f"Written by `scripts/24_challenge_count_check.py`. Official totals: {SOURCE}.", "",
          "| stage | games | challenges | a game | overturned | catchers | batters | pitchers |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        L.append(f"| {r['stage']} | {r['games']:,} | {r['challenges']:,} | {r['per_game']:.2f} | "
                 f"{pct(r['overturn_rate'])} | {r['catcher']:,} ({pct(r['catcher_rate'])}) | "
                 f"{r['batter']:,} ({pct(r['batter_rate'])}) | {r['pitcher']:,} ({pct(r['pitcher_rate'])}) |")
    L += [""] + [f"- {n_}" for n_ in notes]
    if not demo:
        L += ["", f"**By the last stage, {short:.0%} of the official challenge rate is missing.**"
              if short > 0.02 else "**The counts match MLB's (within 2%).**",
              "", "## Which challenges are missing?", "",
              "MLB's totals minus the data's, by challenger role and outcome:", "",
              "| role | overturned missing | failed missing | overturned kept | failed kept |",
              "|---|---|---|---|---|"]
        for r, x in miss.iterrows():
            L.append(f"| {r} | {x['overturned_missing']:,.0f} | {x['failed_missing']:,.0f} | "
                     f"{pct(x['overturned_kept'])} | {pct(x['failed_kept'])} |")
        L += ["", "With two strikes, a batter's failed challenge means strike three stands, and a catcher's "
              "successful one turns a ball into strike three. If those two kinds are the biggest gaps, the "
              "missing challenges are probably ones whose final call ended the at-bat: reviews the first "
              "extractor never looked for."]
    if ends is not None:
        t, share = ends
        L += ["", "### A second hint: the overturn rate by what the final call did to the at-bat", "",
              "If challenges whose final call ENDED the at-bat are missing, the first row's rate is far "
              "higher than the others (its failed challenges are gone) and the second row's far lower (its "
              "successful ones are gone). Teams are also choosier in some counts than others, so treat this "
              "as a hint; the re-extraction below is the real test.", "",
              "| what the final call would do | challenges | overturned |", "|---|---|---|"]
        lab = {"upheld_ends": "upheld call ends the at-bat", "overturned_ends": "overturned call ends it",
               "both": "either call ends it", "neither": "the at-bat goes on either way"}
        for g, r in t.iterrows():
            L.append(f"| {lab[g]} | {int(r['challenges']):,} | {pct(r['overturn_rate'])} |")
        L += ["", f"Share of challenges whose final call ended the at-bat: {pct(share)}."]
    if text is not None and text.get("pas"):
        L += ["", f"Statcast's play descriptions mention a challenge in {text['pas']:,} plate appearances; "
              f"{text['missing']:,} of them have no challenge in the data."]
        if text.get("missing"):
            L += ["", "What those descriptions say was challenged (managers' replay challenges of plays "
                  "— a tag, a force play, a catch — aren't ABS challenges, so they belong here):", "",
                  "| challenged | plate appearances |", "|---|---|"]
            L += [f"| {k} | {v:,} |" for k, v in text["kinds"].items()]
            L += [""] + [f"- e.g. {e}" for e in text["examples"]]
    if sav:
        L += ["", f"Savant's team leaderboard (`{sav[0]}`, column `{sav[1]}`) totals {sav[2]:,.0f} challenges."]
    if month is not None:
        L += ["", "| month | games | challenges a game | official |", "|---|---|---|---|"]
        for mo_, r in month.iterrows():
            off = r["official_per_game"]
            L.append(f"| {int(mo_)} | {int(r['games']):,} | {r['per_game']:.2f} | "
                     f"{'—' if pd.isna(off) else f'{off:.2f}'} |")
    L += ["", "## What depends on the count", "",
          "| result | how it depends |", "|---|---|",
          "| Challenges won back (Week 10) | directly: missing overturns are wins nobody gets credit for |",
          "| Challenges above the bar, above-bar chances taken, the full-information gap (Week 11 §8) | "
          "directly: a missing challenge looks like a chance passed up |",
          "| Challenger roles (Week 10 people) | directly: MLB has catchers 58.6%, batters 48.8% |",
          "| Challenges left, the MDP's state | a missing FAILED challenge leaves a team one challenge it "
          "didn't have |",
          "| The showcase's \"never challenged\" examples and quiz | some may have been challenged: confirm "
          "each on the replay |",
          "| p(s) | trained on challenged pitches; refit after the fix (location-only, so expect a small change) |",
          "| V(s), run expectancy, leverage | barely: built from every pitch, not from challenges |",
          "", "## The fix", "",
          "1. `python scripts/02_abs_challenges_statsapi.py --force` — the November 2026 extractor also "
          "reads reviews recorded on a separate event or on the plate appearance. It downloads each game "
          "once more (about 20–30 minutes) and saves the raw feed; after that, a run without `--force` "
          "re-filters from the saved records in about a minute. Its printout should show about 4.35 "
          "challenges a game, role totals near MLB's, and a per-team-game table like MLB's.",
          "2. Rebuild the table: `python scripts/run_week5.py --spine-from-table` (08–11 on the pitches "
          "already in your table; drop the flag once script 01 has pulled the full season again), then "
          "`python scripts/13_define_cohort.py` (a new cohort fingerprint).",
          "3. `python scripts/24_challenge_count_check.py` again: every stage should now be within ~2%.",
          "4. Re-run Weeks 10–11 and `22_showcase_examples.py`; refit p(s) (Weeks 7–9) as a new version — "
          "it was trained on the incomplete challenges — then re-run 10, 11 and 22 once more.",
          "5. NOTES.md: what was missing, why, and the new fingerprints."]
    (DOCS / "CHALLENGE_COUNT_CHECK.md").write_text("\n".join(L) + "\n")
    print(f"\n→ {DOCS / 'CHALLENGE_COUNT_CHECK.md'} and {DOCS / 'challenge_count_check.csv'}")


if __name__ == "__main__":
    main()
