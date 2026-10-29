#!/usr/bin/env python3
"""
eda_week5_joined.py — slide figures from the Week 5 joined table.

Week 5 is "I joined six sources into one table", which sounds like nothing to
present. It isn't. Every figure here needs the join: none of them can be drawn
from any single source. That IS the presentation — the joining was the thing
that made these questions answerable.

Eight figures, WHITE background, 16:9, 200 DPI:

  fig_w5_01_sources_joined        what the join produced, and from where
  fig_w5_02_opportunity_funnel    how 21k pitches narrow to the ones that matter
  fig_w5_03_zone_edge_distance    challenged vs unchallenged, by distance to the edge
  fig_w5_04_overturn_vs_distance  the relationship Week 7's model will learn
  fig_w5_05_inventory_depletion   challenges remaining through a game
  fig_w5_06_challenge_rate_count  challenge rate by ball-strike count
  fig_w5_07_zone_map              challenged pitches on the strike zone
  fig_w5_08_zone_definitions      why the ABS zone needed batter heights

    python scripts/eda/eda_week5_joined.py
    ABS_DATA_DIR=data_demo python scripts/eda/eda_week5_joined.py
    python scripts/eda/eda_week5_joined.py --only 07
    python scripts/eda/eda_week5_joined.py --bg transparent

Each PNG is written with a `_data.csv` holding exactly what was plotted, so you
can quote a figure precisely in Q&A.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_style as S

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
PROCESSED = DATA_DIR / "processed"
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
OUT = Path(os.environ.get("ABS_FIG_DIR", "figures"))
DEMO = S.is_demo(DATA_DIR)

PLATE_HALF_WIDTH_FT = 8.5 / 12
BALL_RADIUS_FT = (2.9 / 2) / 12


def read_any(path):
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            pass
    csv = p.with_suffix(".csv")
    return pd.read_csv(csv) if csv.exists() else None


def need(df, *cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [skip] needs {', '.join(missing)}")
        return False
    return True


# =========================================================== 01 the join
def fig_sources(df):
    """Where the 77 columns came from. The Week 5 story in one chart."""
    dd = read_any(DOCS / "data_dictionary.csv")
    if dd is not None and {"column", "source"} <= set(dd.columns):
        counts = dd["source"].value_counts()
    else:
        # fall back to prefix rules if script 09 hasn't been run
        def src(c):
            if c.startswith("catcher_"):
                return "Savant catcher framing (04)"
            if c.startswith("ump_"):
                return "Derived umpire accuracy (05)"
            if c in ("leverage_index", "home_win_prob", "away_win_prob",
                     "bat_win_prob", "home_win_prob_added"):
                return "Stats API win probability (06)"
            if c.startswith(("challenge", "challenger", "is_overturned",
                             "was_challenged", "hp_umpire")):
                return "Stats API challenges (02)"
            if c.startswith(("abs_", "zone_dist", "challenges_remaining",
                             "is_challenge", "is_reasonable", "would_be",
                             "batting_side", "fielding_side", "sc_zone")):
                return "Derived in the join (08)"
            if c == "batter_height_ft":
                return "Stats API player bios (07)"
            return "Statcast (01)"
        counts = pd.Series([src(c) for c in df.columns]).value_counts()

    counts = counts.sort_values()
    S.apply_style("white" if ARGS.bg == "white" else ARGS.bg)
    fig, ax = S.new_fig(figsize=(12.0, 6.4))
    y = np.arange(len(counts))
    ax.set_xlim(0, counts.max() * 1.22)
    S.rounded_barh(ax, y, counts.values, height=0.6, color=S.BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=11.5, color=S.INK)
    S.pad_cat_axis(ax, len(counts), "y")
    ax.tick_params(axis="y", length=0)
    for yy, v in zip(y, counts.values):
        ax.text(v + counts.max() * 0.014, yy, f"{v}", va="center", ha="left",
                fontsize=11, color=S.INK_2)

    S.titles(ax,
             f"{len(df.columns)} columns, {len(counts)} sources, "
             f"one row per pitch",
             f"{len(df):,} pitches · every row keyed on game_pk + at_bat_number "
             f"+ pitch_number",
             xlabel="Columns contributed")
    S.only_grid(ax, "x")
    S.footnote(fig, "The row count is identical before and after every join — "
                    "verified by the join ledger, so no merge fanned out.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig_w5_01_sources_joined",
           counts.rename("columns").reset_index().rename(
               columns={"index": "source"}))


# =========================================================== 02 funnel
def fig_funnel(df):
    if not need(df, "is_called_pitch", "is_challenge_opportunity",
                "is_reasonable_pitch", "was_challenged"):
        return
    stages = [
        ("All pitches", len(df)),
        ("Called pitches\n(taken, not swung at)", int(df["is_called_pitch"].sum())),
        ("Challenge opportunities\n(call went against a side with one left)",
         int(df["is_challenge_opportunity"].sum())),
        ("'Reasonable' pitches\n(wrong, or close and high-value)",
         int(df["is_reasonable_pitch"].sum())),
        ("Actually challenged", int(df["was_challenged"].sum())),
    ]
    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]

    S.apply_style(ARGS.bg)
    fig, ax = S.new_fig(figsize=(12.0, 6.8))
    y = np.arange(len(vals))[::-1]
    # Ordinal ramp: darker as the funnel narrows. Stays above the light-surface
    # contrast floor by starting at step 250 of the blue ramp, not step 100.
    colors = [S.SEQ[2], S.SEQ[3], S.SEQ[4], S.SEQ[5], S.SEQ[6]]
    ax.set_xlim(0, max(vals) * 1.26)
    for yy, v, c in zip(y, vals, colors):
        S.rounded_barh(ax, [yy], [v], height=0.62, color=c)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=S.INK)
    S.pad_cat_axis(ax, len(vals), "y")
    ax.tick_params(axis="y", length=0)

    base = vals[0]
    for yy, v, prev in zip(y, vals, [None] + vals[:-1]):
        txt = f"{v:,}  ({v/base:.1%} of pitches)"
        if prev:
            txt += f"   ·  {v/prev:.0%} of the step above"
        ax.text(v + base * 0.015, yy, txt, va="center", ha="left",
                fontsize=10.5, color=S.INK_2)

    S.titles(ax,
             f"{vals[2]:,} challenge opportunities, {vals[-1]:,} challenges taken",
             "Each stage needs the join: the opportunity rule uses the derived "
             "challenge inventory, the ABS zone and the pitch result together",
             xlabel="Pitches")
    S.only_grid(ax, "x")
    S.footnote(fig, "Savant reports roughly 50% of 2026 pitches are challenge "
                    "opportunities and about 5% are 'reasonable' ones.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig_w5_02_opportunity_funnel",
           pd.DataFrame({"stage": [l.replace("\n", " ") for l in labels],
                         "pitches": vals,
                         "share_of_all": [round(v / base, 5) for v in vals]}))


# =========================================================== 03 distance
def fig_zone_edge_distance(df):
    if not need(df, "zone_dist_in", "is_called_pitch", "was_challenged"):
        return
    called = df[df["is_called_pitch"]].copy()
    called = called[called["zone_dist_in"].abs() <= 12]
    ch = called[called["was_challenged"]]
    un = called[~called["was_challenged"]]
    if len(ch) < 20:
        print("  [skip] too few challenges")
        return

    S.apply_style(ARGS.bg)
    fig, ax = S.new_fig()
    bins = np.linspace(-6, 12, 46)

    # Densities, not counts: unchallenged outnumbers challenged ~60:1, so raw
    # counts would render the challenged series as a flat line at zero.
    ax.hist(un["zone_dist_in"], bins=bins, density=True, color=S.DEEMPH,
            alpha=0.95, zorder=2, label=f"Not challenged  (n={len(un):,})")
    ax.hist(ch["zone_dist_in"], bins=bins, density=True, histtype="step",
            lw=2.4, color=S.ORANGE, zorder=4,
            label=f"Challenged  (n={len(ch):,})")

    ax.axvline(0, color=S.INK, lw=1.4, zorder=5)
    S.halo(ax.text(0.15, ax.get_ylim()[1] * 0.97, "edge of the ABS zone",
                   color=S.INK, fontsize=11, ha="left", va="top",
                   fontweight="semibold"))
    med = float(ch["zone_dist_in"].median())
    # Park the label in the empty left half rather than on top of the curve.
    S.halo(ax.text(-5.6, ax.get_ylim()[1] * 0.72,
                   f"median challenged pitch: {med:+.1f} in",
                   color=S.ORANGE, fontsize=11.5, ha="left", va="center",
                   fontweight="semibold"))

    S.titles(ax,
             "Challenges cluster within a couple of inches of the zone edge",
             f"Called pitches, {len(called):,} total · negative = the ball "
             f"overlapped the ABS zone",
             xlabel="Distance from the ABS zone edge (inches)",
             ylabel="Density")
    S.only_grid(ax, "y")
    ax.legend(loc="upper right")
    S.footnote(fig, "Needs three joined sources: Statcast locations, batter "
                    "heights for the zone, and the challenge events.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig_w5_03_zone_edge_distance",
           pd.DataFrame({"zone_dist_in": np.round(called["zone_dist_in"], 3),
                         "was_challenged": called["was_challenged"]}))


# =========================================================== 04 overturn curve
def fig_overturn_vs_distance(df):
    if not need(df, "zone_dist_in", "was_challenged", "is_overturned"):
        return
    ch = df[df["was_challenged"] & df["is_overturned"].notna()].copy()
    if len(ch) < 40:
        print("  [skip] too few resolved challenges")
        return
    ch["absd"] = ch["zone_dist_in"].abs()
    edges = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 100]
    labels = ["0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–4", "4–6", "6+"]
    ch["bucket"] = pd.cut(ch["absd"], edges, labels=labels, right=False)
    g = ch.groupby("bucket", observed=True).agg(
        n=("is_overturned", "size"),
        overturn_rate=("is_overturned", "mean")).reset_index()
    g = g[g["n"] >= 5]
    if len(g) < 3:
        print("  [skip] too few populated buckets")
        return

    S.apply_style(ARGS.bg)
    fig, ax = S.new_fig()
    x = np.arange(len(g))
    ax.plot(x, g["overturn_rate"], color=S.BLUE, lw=2.4, zorder=4, marker="o",
            markersize=9, markerfacecolor=S.BLUE, markeredgecolor=S.WHITE,
            markeredgewidth=1.8)
    overall = float(ch["is_overturned"].mean())
    ax.axhline(overall, color=S.MUTED, lw=1.1, zorder=3)
    S.halo(ax.text(len(g) - 0.5, overall + 0.02, f"overall {overall:.0%}",
                   color=S.INK_2, fontsize=10.5, ha="right", va="bottom"))

    for xx, r in zip(x, g.itertuples()):
        S.halo(ax.text(xx, r.overturn_rate + 0.035, f"{r.overturn_rate:.0%}",
                       fontsize=10.5, color=S.BLUE, ha="center",
                       fontweight="semibold"))
        # n sits just under the tick label; the axis title is pushed down to
        # make room (labelpad below), otherwise the two overlap.
        ax.text(xx, -0.075, f"n={r.n}", fontsize=9.5, color=S.MUTED,
                ha="center", va="top", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(g["bucket"], fontsize=11)
    ax.set_ylim(0, min(1.0, g["overturn_rate"].max() + 0.18))
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    S.titles(ax,
             "The closer the pitch, the more often the challenge wins",
             f"{len(ch):,} resolved challenges · this curve is what the Week 7 "
             f"overturn model has to reproduce",
             xlabel="Distance from the ABS zone edge (inches, absolute)",
             ylabel="Share overturned")
    ax.xaxis.labelpad = 34
    S.only_grid(ax, "y")
    S.footnote(fig, "If this curve were flat, the distance feature would carry no "
                    "signal and the model would have nothing to learn.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig_w5_04_overturn_vs_distance", g)


# =========================================================== 05 inventory
def fig_inventory(df):
    if not need(df, "inning", "challenges_remaining_batting",
                "challenges_remaining_fielding"):
        return
    d = df[df["inning"] <= 9].copy()
    rows = []
    for inn, g in d.groupby("inning"):
        both = pd.concat([g["challenges_remaining_batting"],
                          g["challenges_remaining_fielding"]])
        vc = both.value_counts(normalize=True)
        rows.append({"inning": int(inn),
                     "two": float(vc.get(2, 0)), "one": float(vc.get(1, 0)),
                     "zero": float(vc.get(0, 0))})
    t = pd.DataFrame(rows).sort_values("inning")
    if t.empty:
        return

    S.apply_style(ARGS.bg)
    fig, ax = S.new_fig()
    x = t["inning"].values
    # Ordinal: more challenges left = darker. Sequential ramp, one hue.
    ax.stackplot(x, t["two"], t["one"], t["zero"],
                 colors=[S.SEQ[5], S.SEQ[3], S.SEQ[1]], zorder=2,
                 labels=["2 remaining", "1 remaining", "0 remaining"],
                 edgecolor=S.WHITE, linewidth=1.5)

    # stackplot draws bottom-up, so "2 remaining" is the BOTTOM band and
    # "0 remaining" is the top one. Label them at their own band's centre.
    last = t.iloc[-1]
    S.halo(ax.text(x[-1] + 0.12, last["two"] / 2,
                   f"{last['two']:.0%}\nhave 2", fontsize=10.5, color=S.INK_2,
                   ha="left", va="center"))
    S.halo(ax.text(x[-1] + 0.12, last["two"] + last["one"] / 2,
                   f"{last['one']:.0%}\nhave 1", fontsize=10.5, color=S.INK_2,
                   ha="left", va="center"))
    S.halo(ax.text(x[-1] + 0.12, 1 - last["zero"] / 2,
                   f"{last['zero']:.0%}\nhave 0", fontsize=10.5, color=S.INK_2,
                   ha="left", va="center"))

    ax.set_xticks(x)
    ax.set_xlim(x.min(), x.max() + 1.0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    S.titles(ax,
             f"By the 9th, {t.iloc[-1]['zero']:.0%} of team-sides have no "
             f"challenge left\n",
             xlabel="Inning", ylabel="Share of pitches")
    S.only_grid(ax, "y")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.005))
    S.footnote(fig, "Share of pitches at each inventory level, both sides pooled. "
                    "challenges_remaining is the state variable r in the MDP — it "
                    "exists only because the join walks each game in order.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig_w5_05_inventory_depletion", t.round(4))


# =========================================================== 06 count heatmap
def fig_count_heatmap(df):
    if not need(df, "balls", "strikes", "is_challenge_opportunity",
                "was_challenged"):
        return
    opp = df[df["is_challenge_opportunity"]]
    if len(opp) < 100:
        print("  [skip] too few opportunities")
        return
    grid = opp.pivot_table(index="strikes", columns="balls",
                           values="was_challenged", aggfunc="mean")
    cnt = opp.pivot_table(index="strikes", columns="balls",
                          values="was_challenged", aggfunc="size")
    grid = grid.reindex(index=[0, 1, 2], columns=[0, 1, 2, 3])
    cnt = cnt.reindex(index=[0, 1, 2], columns=[0, 1, 2, 3])

    S.apply_style(ARGS.bg)
    fig, ax = S.new_fig(figsize=(10.5, 6.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("abs_blue", S.SEQ)
    vals = grid.values.astype(float)
    im = ax.imshow(vals, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(vals), vmax=np.nanmax(vals))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v, n = vals[i, j], cnt.values[i, j]
            if np.isnan(v):
                continue
            rel = (v - np.nanmin(vals)) / max(np.nanmax(vals) - np.nanmin(vals), 1e-9)
            ax.text(j, i - 0.10, f"{v:.1%}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=S.WHITE if rel > 0.55 else S.INK)
            ax.text(j, i + 0.22, f"n={int(n):,}", ha="center", va="center",
                    fontsize=9.5,
                    color=("#e8e8e8" if rel > 0.55 else S.INK_2))

    # 2px surface gap between cells, drawn as separators since imshow has none
    for j in range(1, grid.shape[1]):
        ax.axvline(j - 0.5, color=S.WHITE, lw=2.5, zorder=3)
    for i in range(1, grid.shape[0]):
        ax.axhline(i - 0.5, color=S.WHITE, lw=2.5, zorder=3)

    ax.set_xticks(range(4), [str(b) for b in range(4)])
    ax.set_yticks(range(3), [str(s) for s in range(3)])
    ax.set_xlabel("Balls", labelpad=10)
    ax.set_ylabel("Strikes", labelpad=10)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    best = np.unravel_index(np.nanargmax(vals), vals.shape)
    S.titles(ax,
             f"Challenge rate peaks at {best[1]}-{best[0]} "
             f"({vals[best]:.1%} of opportunities)",
             "Share of challenge opportunities actually challenged, by the count "
             "before the pitch")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=10, colors=S.MUTED)
    cb.ax.yaxis.set_major_formatter(lambda v, p: f"{v:.1%}")
    S.footnote(fig, "The count comes from Statcast, the opportunity flag from the "
                    "derived inventory — neither source answers this alone.")
    S.demo_stamp(fig, DEMO)
    out = grid.reset_index().melt(id_vars="strikes", var_name="balls",
                                  value_name="challenge_rate")
    S.save(fig, OUT, "fig_w5_06_challenge_rate_count", out.round(5))


# =========================================================== 07 zone map
def fig_zone_map(df):
    if not need(df, "plate_x", "plate_z", "was_challenged", "abs_zone_top",
                "abs_zone_bot"):
        return
    ch = df[df["was_challenged"] & df["is_overturned"].notna()].copy()
    if len(ch) < 30:
        print("  [skip] too few resolved challenges")
        return
    top = float(ch["abs_zone_top"].median())
    bot = float(ch["abs_zone_bot"].median())
    hw = PLATE_HALF_WIDTH_FT

    S.apply_style(ARGS.bg)
    fig, ax = S.new_fig(figsize=(9.6, 7.0))

    # The zone for a median-height batter, plus the ball-radius halo that makes
    # a pitch a strike when any part of the ball touches the zone.
    ax.add_patch(plt.Rectangle((-hw - BALL_RADIUS_FT, bot - BALL_RADIUS_FT),
                               2 * (hw + BALL_RADIUS_FT),
                               (top - bot) + 2 * BALL_RADIUS_FT,
                               fill=False, ec=S.MUTED, lw=1.2, ls="-", zorder=3))
    ax.add_patch(plt.Rectangle((-hw, bot), 2 * hw, top - bot,
                               fill=False, ec=S.INK, lw=2.0, zorder=4))

    stood = ch[~ch["is_overturned"].astype(bool)]
    over = ch[ch["is_overturned"].astype(bool)]
    ax.scatter(stood["plate_x"], stood["plate_z"], s=34, color=S.BLUE,
               alpha=0.62, zorder=5, edgecolors=S.WHITE, linewidths=0.8,
               label=f"Call stood  (n={len(stood):,})")
    ax.scatter(over["plate_x"], over["plate_z"], s=34, color=S.ORANGE,
               alpha=0.78, zorder=6, edgecolors=S.WHITE, linewidths=0.8,
               label=f"Overturned  (n={len(over):,})")

    # Anchor the annotations outside the point cloud, on the right.
    S.halo(ax.text(hw + 0.08, top + 0.04, "ABS zone", fontsize=11,
                   color=S.INK, ha="left", va="bottom", fontweight="semibold"))
    S.halo(ax.text(hw + BALL_RADIUS_FT + 0.08, bot - BALL_RADIUS_FT - 0.04,
                   "+ ball radius", fontsize=10, color=S.MUTED,
                   ha="left", va="top"))

    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(max(0, bot - 0.85), top + 0.85)
    ax.set_aspect("equal", adjustable="box")
    S.titles(ax, "Every challenged pitch, on the zone it was judged against\n",
             xlabel="Horizontal location (ft, catcher's view)",
             ylabel="Height (ft)")
    ax.grid(True, color=S.GRID, lw=0.8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    S.footnote(fig, f"{len(ch):,} resolved challenges. Each batter has their own "
                    f"zone (53.5% / 27% of height), so the rectangle is drawn for a "
                    f"median-height batter; distances are computed per batter.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig_w5_07_zone_map",
           ch[["plate_x", "plate_z", "is_overturned", "challenger_role"]].round(4))


# =========================================================== 08 zone definitions
def fig_zone_definitions(df):
    if not need(df, "batter_height_ft", "abs_zone_top", "abs_zone_bot",
                "sz_top", "sz_bot"):
        return
    d = df.dropna(subset=["batter_height_ft", "sz_top", "sz_bot"])
    if len(d) < 200:
        print("  [skip] too few rows with heights")
        return
    d = d.sample(min(4000, len(d)), random_state=0)

    S.apply_style(ARGS.bg)
    # Two panels, not one. The zone top sits near 3.2 ft and the bottom near
    # 1.6 ft, so a single axes spends half its height on empty air between them
    # and the cloud-vs-line contrast — the whole point — shrinks to nothing.
    fig, axes = plt.subplots(2, 1, figsize=(S.FULL[0], 7.2), sharex=True)
    hs = np.linspace(d["batter_height_ft"].min(), d["batter_height_ft"].max(), 50)

    for ax, sz_col, pct, label in [
            (axes[0], "sz_top", 0.535, "Top of the zone"),
            (axes[1], "sz_bot", 0.27, "Bottom of the zone")]:
        ax.scatter(d["batter_height_ft"], d[sz_col], s=10, color=S.DEEMPH,
                   alpha=0.40, zorder=2,
                   label="Statcast zone (stance-dependent)")
        ax.plot(hs, hs * pct, color=S.BLUE, lw=2.8, zorder=5,
                label="ABS zone (a fixed % of batter height)")
        ax.set_title(f"{label} — {pct:.1%} of batter height", loc="left",
                     fontsize=13.5)
        ax.set_ylabel("Height (ft)")
        S.only_grid(ax, "y")
        resid = (d[sz_col] - d["batter_height_ft"] * pct)
        S.halo(ax.text(0.995, 0.06,
                       f"Statcast sits {resid.mean():+.2f} ft from the ABS line "
                       f"on average, ±{resid.std():.2f}",
                       transform=ax.transAxes, fontsize=10.5, color=S.INK_2,
                       ha="right", va="bottom"))

    axes[1].set_xlabel("Batter height (ft)", labelpad=10)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2,
                   fontsize=10.5)
    fig.subplots_adjust(hspace=0.34)

    if "zone_defs_disagree" in df.columns and "is_called_pitch" in df.columns:
        dis = float(df.loc[df["is_called_pitch"], "zone_defs_disagree"].mean())
        sub = (f"The two definitions disagree on {dis:.1%} of called pitches — "
               f"which is exactly where challenges happen")
    else:
        sub = "Statcast's zone moves with stance; the ABS zone does not"

    S.suptitles(fig,
                "The ABS zone is a straight line in batter height. "
                "Statcast's is a cloud.", sub, top=1.26)
    S.footnote(fig, "This is why the join needed player bios. Using sz_top/sz_bot "
                    "for the ABS calculation biases every distance feature in a "
                    "way that correlates with batter height.")
    S.demo_stamp(fig, DEMO)
    S.save(fig, OUT, "fig_w5_08_zone_definitions",
           d[["batter_height_ft", "abs_zone_top", "abs_zone_bot",
              "sz_top", "sz_bot"]].round(4))


FIGS = {
    "01": ("sources joined", fig_sources),
    "02": ("opportunity funnel", fig_funnel),
    "03": ("zone-edge distance", fig_zone_edge_distance),
    "04": ("overturn vs distance", fig_overturn_vs_distance),
    "05": ("inventory depletion", fig_inventory),
    "06": ("challenge rate by count", fig_count_heatmap),
    "07": ("zone map", fig_zone_map),
    "08": ("zone definitions", fig_zone_definitions),
}


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default=str(PROCESSED / "challenge_inventory_2026.parquet"))
    ap.add_argument("--only", nargs="+", help="figure numbers, e.g. --only 03 07")
    ap.add_argument("--bg", default="white", choices=["white", "transparent"],
                    help="slide background (default white)")
    ARGS = ap.parse_args()

    df = read_any(ARGS.table)
    if df is None:
        sys.exit(f"No joined table at {ARGS.table}.\n"
                 f"Run: python scripts/run_week5.py   (or --demo to try it on "
                 f"synthetic data)")

    for c in ("was_challenged", "is_called_pitch", "is_challenge_opportunity",
              "is_challenge_opportunity_batting", "is_challenge_opportunity_fielding",
              "is_reasonable_pitch", "abs_strike", "would_be_overturned",
              "zone_defs_disagree"):
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    print(f"Week 5 figures from {ARGS.table}")
    print(f"  {len(df):,} rows x {len(df.columns)} cols · background: {ARGS.bg}"
          + ("   [DEMO DATA]" if DEMO else ""))

    which = ARGS.only or list(FIGS)
    for k in which:
        k = k.zfill(2)
        if k not in FIGS:
            print(f"  [skip] no figure {k}")
            continue
        label, fn = FIGS[k]
        print(f"\n{k} — {label}")
        try:
            fn(df)
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")

    print(f"\nFigures -> {OUT}/")


if __name__ == "__main__":
    main()
