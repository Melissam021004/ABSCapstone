#!/usr/bin/env python3
"""
eda_03_leaderboards.py — slide figures from the Savant ABS leaderboards (script 03).

Produces:
  fig03a_challenge_volume_vs_success.png   who challenges, how often, how well
  fig03b_leaderboard_coverage.png          what you actually extracted

Savant's column names are undocumented and differ between leaderboard views, so
this script DETECTS the columns it needs rather than hard-coding them. If a panel
can't find what it needs it says which column it wanted and skips, instead of
dying halfway through.

    python scripts/eda/eda_03_leaderboards.py
    ABS_DATA_DIR=data_demo python scripts/eda/eda_03_leaderboards.py   # preview
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_style as S

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
SRC = DATA_DIR / "raw" / "savant"
OUT = Path(os.environ.get("ABS_FIG_DIR", "figures"))
DEMO = S.is_demo(DATA_DIR)

ROLES = ["batter", "catcher", "pitcher"]          # 3 series -> all-pairs safe
ROLE_COLOR = dict(zip(ROLES, S.SERIES))

# Players below this are excluded from fig03a: with 1-2 challenges the
# success rate is 0% or 100% and tells you nothing.
MIN_CHALLENGES = int(os.environ.get("ABS_MIN_CHALLENGES", 5))


def load_role_files():
    out = {}
    for role in ROLES:
        f = SRC / f"abs_leaderboard_{role}_for_mlb_2026.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["challenger_role"] = role
            out[role] = df
    return out


# --------------------------------------------------------------------------
def fig_volume_vs_success(by_role):
    """Scatter: challenges attempted vs success rate, one mark per player."""
    if not by_role:
        print(S.missing_note(SRC / "abs_leaderboard_<role>_for_mlb_2026.csv"))
        return

    frames = []
    for role, df in by_role.items():
        n_col = S.find_col(df, r"^n_chal", r"chal.*total", r"^challenges$",
                           numeric=True, exclude=("exp", "pct", "rate", "won"))
        pct_col = S.find_col(df, r"win_pct", r"overturn.*pct", r"success.*rate",
                             r"pct.*won", numeric=True)
        won_col = S.find_col(df, r"won", r"overturn", numeric=True,
                             exclude=("pct", "rate", "exp"))
        name_col = S.find_col(df, r"name", r"player", numeric=False)

        if n_col is None:
            print(f"  [skip] {role}: no challenge-count column found")
            continue
        if pct_col is not None:
            rate = S.as_rate(df[pct_col])
        elif won_col is not None:
            rate = pd.to_numeric(df[won_col], errors="coerce") / \
                   pd.to_numeric(df[n_col], errors="coerce")
        else:
            print(f"  [skip] {role}: no success-rate or wins column found")
            continue

        frames.append(pd.DataFrame({
            "role": role,
            "name": df[name_col] if name_col else "",
            "challenges": pd.to_numeric(df[n_col], errors="coerce"),
            "success_rate": rate,
        }))

    if not frames:
        return
    d = pd.concat(frames, ignore_index=True).dropna(subset=["challenges", "success_rate"])
    d = d[d["challenges"] > 0]
    # Players with one or two challenges land at 0% or 100% and dominate the
    # plot with noise. A stated minimum is standard practice and defensible.
    n_all = len(d)
    d = d[d["challenges"] >= MIN_CHALLENGES]
    if d.empty:
        print(f"  [skip] no players with >= {MIN_CHALLENGES} challenges")
        return

    S.apply_style()
    fig, ax = S.new_fig()

    league = float(np.average(d["success_rate"], weights=d["challenges"]))
    ax.axhline(league, color=S.MUTED, lw=1.1, zorder=2)
    S.halo(ax.text(0.012, league + 0.018, f"league average {league:.0%}",
                   transform=ax.get_yaxis_transform(), color=S.INK_2,
                   fontsize=11, va="bottom", ha="left", zorder=6))

    for role in ROLES:
        g = d[d["role"] == role]
        if g.empty:
            continue
        ax.scatter(g["challenges"], g["success_rate"], s=52,
                   color=ROLE_COLOR[role], alpha=0.78, zorder=4,
                   edgecolors="white", linewidths=1.4,   # 2px surface ring
                   label=f"{role.capitalize()}  (n={len(g)})")

    # Direct-label the single busiest challenger per role. Never every point.
    for role in ROLES:
        g = d[d["role"] == role].nlargest(1, "challenges")
        for r in g.itertuples():
            if isinstance(r.name, str) and r.name:
                S.halo(ax.annotate(r.name, (r.challenges, r.success_rate),
                                   textcoords="offset points", xytext=(11, 6),
                                   fontsize=10.5, color=S.INK, zorder=8))

    means = d.groupby("role").apply(
        lambda g: np.average(g["success_rate"], weights=g["challenges"]))
    best = means.idxmax()

    S.titles(
        ax,
        f"{best.capitalize()}s are the most successful challengers "
        f"({means[best]:.0%} overturned)",
        f"{len(d):,} players with {MIN_CHALLENGES}+ challenges "
        f"(of {n_all:,}) · {int(d['challenges'].sum()):,} challenges · Savant, 2026",
        xlabel="Challenges attempted", ylabel="Share overturned")
    S.only_grid(ax, "y")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.set_xlim(0, d["challenges"].max() * 1.12)
    ax.legend(loc="upper right", ncol=1)
    S.footnote(fig, "Horizontal line is the challenge-weighted league average. "
                    "Values in fig03a_challenge_volume_vs_success_data.csv")
    S.demo_stamp(ax, DEMO)
    S.save(fig, OUT, "fig03a_challenge_volume_vs_success", d)


def fig_coverage():
    """What actually came out of script 03 — the extraction-provenance figure."""
    files = sorted(SRC.glob("abs_leaderboard_*_*_*_*.csv"))
    if not files:
        print(S.missing_note(SRC / "abs_leaderboard_*.csv"))
        return

    rows = []
    for f in files:
        parts = f.stem.replace("abs_leaderboard_", "").split("_")
        mode = parts[-3] if len(parts) >= 3 else "?"
        view = "_".join(parts[:-3]) if len(parts) >= 3 else f.stem
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        rows.append({"view": view, "mode": mode, "rows": len(df),
                     "columns": len(df.columns),
                     "label": f"{view}  ({mode})"})
    if not rows:
        return
    d = pd.DataFrame(rows).sort_values("rows")

    S.apply_style()
    fig, ax = S.new_fig(figsize=(12.0, 6.2))
    y = np.arange(len(d))
    ax.set_xlim(0, d["rows"].max() * 1.16)
    S.rounded_barh(ax, y, d["rows"].values, height=0.6, color=S.BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=11.5, color=S.INK)
    S.pad_cat_axis(ax, len(d), "y")

    for yy, v, c in zip(y, d["rows"].values, d["columns"].values):
        ax.text(v + d["rows"].max() * 0.012, yy, f"{v:,} rows · {c} cols",
                va="center", ha="left", fontsize=10.5, color=S.INK_2)

    S.titles(ax, f"{len(d)} ABS leaderboard views extracted, "
                 f"{int(d['rows'].sum()):,} rows total",
             "2026 regular season · 'for' = challenges made, "
             "'against' = challenges faced",
             xlabel="Rows extracted")
    S.only_grid(ax, "x")
    ax.tick_params(axis="y", length=0)
    S.footnote(fig, "Extracted from the JSON embedded in each leaderboard page; "
                    "Savant's csv=true parameter returns HTTP 500 on this endpoint.")
    S.demo_stamp(ax, DEMO)
    S.save(fig, OUT, "fig03b_leaderboard_coverage",
           d[["view", "mode", "rows", "columns"]])


def main():
    print(f"Reading {SRC}" + ("   [DEMO DATA]" if DEMO else ""))
    by_role = load_role_files()
    fig_volume_vs_success(by_role)
    fig_coverage()
    print(f"\nFigures -> {OUT}/")


if __name__ == "__main__":
    main()
