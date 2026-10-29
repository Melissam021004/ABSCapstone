#!/usr/bin/env python3
"""
eda_04_framing.py — slide figures from the catcher framing pull (script 04).

Produces:
  fig04a_framing_distribution.png   how much framing skill varies across catchers
  fig04b_framing_leaders.png        who is best and worst at it

If you pulled more than one season, fig04a becomes small multiples by season,
which is the RQ4 "did catchers change?" view — with the caveat that Savant's
2026 framing metric now credits successful challenges, so a cross-season drop is
partly a definitional change, not purely a behavioural one. That caveat is
printed on the figure, because it is exactly the question an audience will ask.

    python scripts/eda/eda_04_framing.py
    ABS_DATA_DIR=data_demo python scripts/eda/eda_04_framing.py
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
TOP_N = 12


def load():
    df = S.read_table(SRC / "catcher_framing_all_years.parquet")
    if df is None:
        for f in sorted(SRC.glob("catcher_framing_*.csv")):
            df = pd.read_csv(f)
            break
    return df


def columns(df):
    rv = S.find_col(df, r"runs_extra_strikes", r"^rv_tot$", r"runs.*total",
                    r"framing.*run", r"\brv\b", numeric=True)
    name = S.find_col(df, r"player_name", r"\bname\b", r"catcher", numeric=False)
    n = S.find_col(df, r"n_called", r"called_p", r"n_pitch", numeric=True)
    season = S.find_col(df, r"season", r"^year$", numeric=True)
    return rv, name, n, season


def fig_distribution(df, rv, season):
    seasons = sorted(df[season].dropna().unique()) if season else []
    S.apply_style()

    if len(seasons) > 1:
        fig, axes = plt_subplots(len(seasons))
        lo, hi = df[rv].quantile([0.01, 0.99])
        rows = []
        # The panels share a y-axis, so set the limit ONCE after all three are
        # drawn. Expanding it inside the loop compounds, and the median labels
        # end up at visibly different heights.
        medians = []
        for ax, s in zip(axes, seasons):
            g = df[df[season] == s][rv].dropna()
            ax.hist(g, bins=np.linspace(lo, hi, 18), color=S.BLUE,
                    alpha=0.9, zorder=3, rwidth=0.8)
            ax.axvline(0, color=S.AXIS, lw=1.0, zorder=2)
            med = float(g.median())
            medians.append((ax, med))
            ax.axvline(med, color=S.ORANGE, lw=2.0, zorder=4)
            ax.set_title(f"{int(s)}", loc="left", fontsize=14)
            ax.set_xlabel("Framing runs")
            S.only_grid(ax, "y")
            rows.append({"season": int(s), "n_catchers": len(g),
                         "median_runs": round(med, 2),
                         "sd_runs": round(float(g.std()), 2),
                         "p10": round(float(g.quantile(.1)), 2),
                         "p90": round(float(g.quantile(.9)), 2)})

        top = max(ax.get_ylim()[1] for ax, _ in medians) * 1.16
        for ax, med in medians:
            ax.set_ylim(0, top)
            S.halo(ax.text(med, top * 0.97, f"median {med:+.1f}",
                           color=S.ORANGE, fontsize=10.5, ha="center",
                           va="top", fontweight="semibold"))
        axes[0].set_ylabel("Catchers")
        sds = {int(r["season"]): r["sd_runs"] for r in rows}
        newest = max(sds)
        S.suptitles(
            fig,
            f"Framing spread is {sds[newest]:.1f} runs in {newest}",
            f"Distribution of season framing run value · "
            f"{len(df):,} catcher-seasons · Baseball Savant",
            top=1.16)
        S.footnote(fig, "Caution: Savant's 2026 framing metric credits successful "
                        "challenges as well as the original call, so a cross-season "
                        "change is partly definitional, not purely behavioural.")
        S.demo_stamp(fig, DEMO)
        S.save(fig, OUT, "fig04a_framing_distribution", pd.DataFrame(rows))
        return

    # Single season
    g = df[rv].dropna()
    fig, ax = S.new_fig()
    lo, hi = g.quantile([0.005, 0.995])
    ax.hist(g, bins=np.linspace(lo, hi, 24), color=S.BLUE, alpha=0.9,
            zorder=3, rwidth=0.82)
    ax.axvline(0, color=S.AXIS, lw=1.0, zorder=2)
    med = g.median()
    ax.axvline(med, color=S.ORANGE, lw=2.2, zorder=4)
    S.halo(ax.text(med, ax.get_ylim()[1] * 0.96, f"  median {med:+.1f} runs",
                   color=S.ORANGE, fontsize=11.5, ha="left", va="top",
                   fontweight="semibold"))
    S.titles(ax,
             f"Framing value spans {g.min():+.0f} to {g.max():+.0f} runs "
             f"across {len(g)} catchers",
             "Season framing run value · Baseball Savant catcher framing leaderboard",
             xlabel="Framing runs (season)", ylabel="Catchers")
    S.only_grid(ax, "y")
    S.footnote(fig, "In 2026 this metric also credits successful ABS challenges, "
                    "so it blends framing skill with challenge skill.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig04a_framing_distribution",
           pd.DataFrame({"framing_runs": g.values}))


def plt_subplots(n):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(S.FULL[0], 5.4), sharey=True)
    return fig, (axes if n > 1 else [axes])


def fig_leaders(df, rv, name, season):
    if name is None:
        print("  [skip] fig04b: no catcher-name column found")
        return
    d = df.copy()
    if season and d[season].notna().any():
        newest = d[season].max()
        d = d[d[season] == newest]
        season_txt = f"{int(newest)}"
    else:
        season_txt = "2026"

    d = d[[name, rv]].dropna().sort_values(rv)
    if len(d) < 4:
        print("  [skip] fig04b: too few catchers")
        return
    k = min(TOP_N, len(d) // 2)
    sel = pd.concat([d.head(k), d.tail(k)]).reset_index(drop=True)

    S.apply_style()
    fig, ax = S.new_fig(figsize=(12.0, 7.4))
    y = np.arange(len(sel))
    vals = sel[rv].values
    colors = [S.POS if v >= 0 else S.NEG for v in vals]

    span = max(abs(vals).max(), 1e-9)
    ax.set_xlim(-span * 1.30, span * 1.30)
    for yy, v, c in zip(y, vals, colors):
        S.rounded_barh(ax, [yy], [v], height=0.66, color=c)
    ax.axvline(0, color=S.AXIS, lw=1.2, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(sel[name], fontsize=11, color=S.INK)
    S.pad_cat_axis(ax, len(sel), "y")
    ax.tick_params(axis="y", length=0)

    for yy, v in zip(y, vals):
        off = span * 0.03
        ax.text(v + (off if v >= 0 else -off), yy, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=10.5, color=S.INK_2)

    S.titles(ax, f"Best and worst {k} framers of {season_txt} span "
                 f"{vals.max() - vals.min():.0f} runs",
             f"Season framing run value · top and bottom {k} of {len(d)} catchers",
             xlabel="Framing runs (season)")
    S.only_grid(ax, "x")
    S.footnote(fig, "Blue = saved runs for the pitching staff, red = cost runs. "
                    "Zero line is league average.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig04b_framing_leaders",
           sel.rename(columns={name: "catcher", rv: "framing_runs"}))


def main():
    print(f"Reading {SRC}" + ("   [DEMO DATA]" if DEMO else ""))
    df = load()
    if df is None or df.empty:
        print(S.missing_note(SRC / "catcher_framing_all_years.parquet"))
        return
    rv, name, n, season = columns(df)
    if rv is None:
        print(f"  [skip] no framing run-value column. Columns present: "
              f"{list(df.columns)}")
        return
    df[rv] = pd.to_numeric(df[rv], errors="coerce")
    fig_distribution(df, rv, season)
    fig_leaders(df, rv, name, season)
    print(f"\nFigures -> {OUT}/")


if __name__ == "__main__":
    main()
