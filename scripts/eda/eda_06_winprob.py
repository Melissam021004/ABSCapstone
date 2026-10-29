#!/usr/bin/env python3
"""
eda_06_winprob.py — slide figures from the win probability pull (script 06).

Produces:
  fig06a_leverage_distribution.png   how concentrated leverage is
  fig06b_leverage_by_inning.png      where in the game the stakes actually sit

Why these two: your Q-function weighs a challenge's success probability against
how much the game state is worth. Leverage is the "how much is it worth" term.
Both figures make the case that challenge timing matters — most plate appearances
are low-stakes, and the high-stakes ones cluster late, which is precisely the
tension a finite-horizon MDP exists to resolve.

Grain caveat, stated on the figure: this endpoint reports one row per plate
appearance, not per pitch. A challenge happens mid-PA, so these are the stakes
entering the PA.

    python scripts/eda/eda_06_winprob.py
    ABS_DATA_DIR=data_demo python scripts/eda/eda_06_winprob.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_style as S

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
SRC = DATA_DIR / "raw" / "winprob"
OUT = Path(os.environ.get("ABS_FIG_DIR", "figures"))
DEMO = S.is_demo(DATA_DIR)

HIGH_LI = 1.5          # conventional "high leverage" threshold


def fig_leverage_distribution(df, li_col):
    g = pd.to_numeric(df[li_col], errors="coerce").dropna()
    g = g[(g >= 0) & (g < 8)]
    if g.empty:
        print("  [skip] fig06a: no usable leverage values")
        return

    S.apply_style()
    fig, ax = S.new_fig()
    bins = np.linspace(0, 4, 41)
    counts, edges, patches = ax.hist(g, bins=bins, color=S.SEQ[3], alpha=0.95,
                                     zorder=3, rwidth=0.82)
    # Emphasis: the high-leverage tail is the part of the story that matters.
    for p, left in zip(patches, edges[:-1]):
        if left >= HIGH_LI:
            p.set_facecolor(S.ORANGE)

    share_high = float((g >= HIGH_LI).mean())
    ax.axvline(1.0, color=S.INK, lw=1.6, zorder=5)
    S.halo(ax.text(1.0, counts.max() * 0.97, "  average PA = 1.0",
                   color=S.INK, fontsize=11.5, ha="left", va="top",
                   fontweight="semibold"))
    S.halo(ax.text(HIGH_LI + 0.08, counts.max() * 0.62,
                   f"high leverage (LI ≥ {HIGH_LI})\n{share_high:.0%} of plate appearances",
                   color=S.ORANGE, fontsize=11.5, ha="left", va="top",
                   fontweight="semibold"))

    S.titles(ax,
             f"Only {share_high:.0%} of plate appearances are high leverage",
             f"{len(g):,} plate appearances across "
             f"{df['game_pk'].nunique():,} games · MLB Stats API win probability",
             xlabel="Leverage index", ylabel="Plate appearances")
    S.only_grid(ax, "y")
    S.footnote(fig, "Leverage index is scaled so the average plate appearance = 1.0. "
                    "A challenge spent in a low-leverage spot is a challenge "
                    "unavailable later.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig06a_leverage_distribution",
           pd.DataFrame({"leverage_index": np.round(g.values, 3)}))


def fig_leverage_by_inning(df, li_col):
    if "inning" not in df.columns:
        print("  [skip] fig06b: no inning column")
        return
    d = df[["inning", li_col]].copy()
    d[li_col] = pd.to_numeric(d[li_col], errors="coerce")
    d = d.dropna()
    d = d[(d["inning"] >= 1) & (d["inning"] <= 12)]
    if d.empty:
        return

    g = d.groupby("inning")[li_col].agg(
        n="size", median="median",
        q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
        share_high=lambda s: (s >= HIGH_LI).mean()).reset_index()
    g = g[g["n"] >= 50]

    S.apply_style()
    fig, ax = S.new_fig()
    ax.fill_between(g["inning"], g["q25"], g["q75"], color=S.SEQ[1],
                    alpha=0.55, zorder=2, linewidth=0)
    ax.plot(g["inning"], g["median"], color=S.BLUE, lw=2.4, zorder=4,
            marker="o", markersize=8, markerfacecolor=S.BLUE,
            markeredgecolor="white", markeredgewidth=1.6)
    ax.axhline(1.0, color=S.MUTED, lw=1.1, zorder=3)
    S.halo(ax.text(g["inning"].min(), 1.02, "league average PA",
                   color=S.INK_2, fontsize=10.5, va="bottom", ha="left"))

    last = g.iloc[-1]
    first = g.iloc[0]
    S.halo(ax.annotate(f"{last['median']:.2f}",
                       (last["inning"], last["median"]),
                       textcoords="offset points", xytext=(10, 0),
                       fontsize=11.5, color=S.BLUE, fontweight="semibold",
                       va="center"))

    ax.set_xticks(g["inning"].astype(int))
    S.titles(ax,
             f"Median leverage rises from {first['median']:.2f} in the "
             f"{ordinal(int(first['inning']))} to {last['median']:.2f} by the "
             f"{ordinal(int(last['inning']))}",
             "Median leverage index per inning, shaded band = interquartile range",
             xlabel="Inning", ylabel="Leverage index")
    S.only_grid(ax, "y")
    ax.set_ylim(bottom=0)
    S.footnote(fig, "One row per plate appearance, not per pitch — a challenge "
                    "happens mid-PA, so these are the stakes entering the PA.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig06b_leverage_by_inning", g.round(4))


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"


def main():
    print(f"Reading {SRC}" + ("   [DEMO DATA]" if DEMO else ""))
    df = S.read_table(SRC / "winprob_2026.parquet")
    if df is None or df.empty:
        print(S.missing_note(SRC / "winprob_2026.parquet"))
        return
    li = S.find_col(df, r"leverage", numeric=True)
    if li is None:
        print(f"  [skip] no leverage column. Columns: {list(df.columns)}")
        return
    fig_leverage_distribution(df, li)
    fig_leverage_by_inning(df, li)
    print(f"\nFigures -> {OUT}/")


if __name__ == "__main__":
    main()
