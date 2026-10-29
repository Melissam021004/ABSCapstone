#!/usr/bin/env python3
"""
eda_05_umpires.py — slide figures from the derived umpire accuracy (script 05).

Produces:
  fig05a_umpire_accuracy.png    accuracy overall vs on borderline pitches
  fig05b_umpire_error_mix.png   which way each umpire misses

Why fig05b matters to the capstone: a challenge opportunity only exists when the
call went AGAINST you. An umpire who mostly misses strikes (calling a strike a
ball) manufactures opportunities for the pitching side; one who mostly misses
balls manufactures them for the batter. That asymmetry is a real feature for the
overturn model, and it is the kind of thing a committee finds interesting.

fig05a uses two panels rather than one chart with two y-axes. Overall accuracy
(~93%) and borderline accuracy (~79%) live on different scales, and putting them
on a shared axis would invent a comparison the data doesn't support.

    python scripts/eda/eda_05_umpires.py
    ABS_DATA_DIR=data_demo python scripts/eda/eda_05_umpires.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_style as S

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
SRC = DATA_DIR / "raw" / "umpires"
OUT = Path(os.environ.get("ABS_FIG_DIR", "figures"))
DEMO = S.is_demo(DATA_DIR)
TOP_N = 22


def fig_accuracy(df):
    acc = S.find_col(df, r"^accuracy$", r"overall.*acc", numeric=True)
    edge = S.find_col(df, r"borderline_acc", r"edge.*acc", numeric=True)
    if acc is None:
        print("  [skip] fig05a: no accuracy column")
        return

    panels = [(acc, "All called pitches", S.BLUE)]
    if edge is not None:
        panels.append((edge, "Borderline only (within 2 in of the zone)", S.ORANGE))

    S.apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(S.FULL[0], 5.6))
    axes = np.atleast_1d(axes)
    rows = []

    for ax, (col, label, color) in zip(axes, panels):
        g = S.as_rate(df[col]).dropna()
        if g.empty:
            continue
        lo, hi = g.min(), g.max()
        pad = max((hi - lo) * 0.08, 0.004)
        ax.hist(g, bins=np.linspace(lo - pad, hi + pad, 16), color=color,
                alpha=0.9, zorder=3, rwidth=0.8)
        m = g.mean()
        ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
        ax.axvline(m, color=S.INK, lw=1.8, zorder=5)
        S.halo(ax.text(m, ax.get_ylim()[1] * 0.98, f"  mean {m:.1%}",
                       color=S.INK, fontsize=11.5, ha="left", va="top",
                       fontweight="semibold"))
        ax.set_title(label, loc="left", fontsize=13.5)
        ax.set_xlabel("Share of calls matching the ABS zone")
        ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        S.only_grid(ax, "y")
        rows.append({"measure": label, "n_umpires": len(g),
                     "mean": round(float(m), 4), "sd": round(float(g.std()), 4),
                     "min": round(float(lo), 4), "max": round(float(hi), 4)})

    axes[0].set_ylabel("Umpires")
    head = rows[0]
    sub = (f"{head['n_umpires']} home-plate umpires · derived from Statcast "
           f"pitch locations against the 2026 ABS zone")
    if len(rows) > 1:
        title = (f"Umpires agree with ABS {rows[0]['mean']:.1%} of the time overall, "
                 f"but only {rows[1]['mean']:.1%} on borderline pitches")
    else:
        title = f"Umpires agree with the ABS zone {rows[0]['mean']:.1%} of the time"

    S.suptitles(fig, title, sub, top=1.14)
    S.footnote(fig, "Computed from our own pitch-level pull, not a third-party "
                    "scorecard, so the zone definition matches the one the model uses.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig05a_umpire_accuracy", pd.DataFrame(rows))


def fig_error_mix(df):
    ms = S.find_col(df, r"missed_strike", numeric=True)
    mb = S.find_col(df, r"missed_ball", numeric=True)
    name = S.find_col(df, r"hp_umpire$", r"umpire.*name", r"^umpire$", numeric=False)
    games = S.find_col(df, r"n_games", numeric=True)
    if ms is None or mb is None or name is None:
        print("  [skip] fig05b: needs missed_strike_rate, missed_ball_rate and a name")
        return

    d = df[[name, ms, mb] + ([games] if games else [])].dropna().copy()
    if games:
        d = d.nlargest(min(TOP_N, len(d)), games)
    d["tilt"] = d[ms] - d[mb]
    d = d.sort_values("tilt").reset_index(drop=True)

    S.apply_style()
    fig, ax = S.new_fig(figsize=(12.0, 7.8))
    y = np.arange(len(d))

    # Connector first, then the two endpoint markers on top.
    for yy, a, b in zip(y, d[mb].values, d[ms].values):
        ax.plot([a, b], [yy, yy], color=S.DEEMPH, lw=2.0, zorder=2,
                solid_capstyle="round")
    ax.scatter(d[mb], y, s=88, color=S.SEQ[2], zorder=4, edgecolors="white",
               linewidths=1.6, label="Missed ball (called a strike) — batter's opportunity")
    ax.scatter(d[ms], y, s=88, color=S.SEQ[5], zorder=4, edgecolors="white",
               linewidths=1.6, label="Missed strike (called a ball) — pitching side's opportunity")

    ax.set_yticks(y)
    ax.set_yticklabels(d[name], fontsize=10.5, color=S.INK)
    S.pad_cat_axis(ax, len(d), "y")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(lambda v, p: f"{v:.1%}")

    lean_p = d.iloc[-1]
    lean_b = d.iloc[0]
    # The legend owns the band directly under the title, so this figure carries
    # no subtitle — its provenance line moves into the footnote instead.
    S.titles(ax, "Umpires differ in which way they miss, not just how often\n",
             xlabel="Share of called pitches")
    S.only_grid(ax, "x")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, fontsize=10.5,
              handletextpad=0.4, columnspacing=1.8)
    S.footnote(fig,
               f"Top {len(d)} umpires by games worked, sorted by directional tilt. "
               f"{lean_p[name]} tilts most toward missed strikes "
               f"(+{lean_p['tilt']:.2%}); {lean_b[name]} most toward missed balls "
               f"({lean_b['tilt']:.2%}).")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig05b_umpire_error_mix",
           d.rename(columns={name: "umpire", ms: "missed_strike_rate",
                             mb: "missed_ball_rate"}))


def main():
    print(f"Reading {SRC}" + ("   [DEMO DATA]" if DEMO else ""))
    df = S.read_table(SRC / "umpire_accuracy_derived_2026.parquet")
    if df is None or df.empty:
        print(S.missing_note(SRC / "umpire_accuracy_derived_2026.parquet"))
        return
    fig_accuracy(df)
    fig_error_mix(df)
    print(f"\nFigures -> {OUT}/")


if __name__ == "__main__":
    main()
