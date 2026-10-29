"""
week7_figures.py — the seven Week 7 figures. Called by 15_week7_baseline.py;
not meant to be run on its own.

All figures: white background, 16:9, 200 DPI, each with a companion
*_data.csv holding the exact plotted numbers (so you can quote them in Q&A).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eda_style as S                       # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator  # noqa: E402
from src import abs_metrics as M            # noqa: E402
from src.abs_features import BLOCKS         # noqa: E402

CALL_COLORS = {"strike": S.ORANGE, "ball": S.BLUE}
CALL_NAMES = {"strike": "called strike — batter challenges",
              "ball": "called ball — fielder challenges"}


def _binned(x, y, edges, min_n=10):
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n >= min_n:
            k = int(y[m].sum())
            lo, hi = M.wilson(k, n)
            rows.append({"center": (edges[b] + edges[b + 1]) / 2, "n": n,
                         "rate": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=["center", "n", "rate", "lo", "hi"])


def _pct_axis(ax, axis="y"):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _strip(axh, p, color, annotate=True):
    """
    Histogram of predictions under a reliability diagram. Most challenges sit
    at ~0% or ~100%, which would flatten every other bar to nothing, so the
    two end bars are clipped (and labelled with their real counts) whenever
    they are much taller than everything in between.
    """
    counts, edges = np.histogram(np.clip(p, 0, 1), bins=np.linspace(0, 1, 21))
    mids = (edges[:-1] + edges[1:]) / 2
    inner = counts[1:-1].max() if len(counts) > 2 else counts.max()
    cap = max(inner * 2.4, 1)
    clipped = counts > cap
    axh.bar(mids, np.minimum(counts, cap), width=0.05, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    if clipped.any():
        axh.set_ylim(0, cap * 1.35)
        if annotate:
            for x, c in zip(mids[clipped], counts[clipped]):
                axh.text(x, cap * 1.02, f"{c:,}", ha="center", va="bottom", fontsize=8,
                         color=S.INK_2)
    axh.set_xlim(-0.02, 1.02)
    axh.set_xticks([0, .25, .5, .75, 1])
    _pct_axis(axh, "x")
    axh.grid(False)
    axh.set_yticks([])
    axh.spines["left"].set_visible(False)


# ------------------------------------------------------------------ fig 1
def fig_sign_trap(out, demo, tr, auc_raw, auc_res):
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, sharey=True)
    fig.subplots_adjust(top=0.80, wspace=0.12)
    edges = np.arange(-6.5, 7.0, 1.0)
    table = []
    for ax, col, auc_v, sub, xl in (
            (axes[0], "zone_dist_in", auc_raw, "As built in Week 5: signed by the ZONE",
             "inches outside the zone (+) / inside (−)"),
            (axes[1], "dist_against_call", auc_res, "Re-signed by the CALL",
             "inches on the wrong side of the edge for the call (+)")):
        for call, mask in (("strike", tr["called_strike"].to_numpy(bool)),
                           ("ball", ~tr["called_strike"].to_numpy(bool))):
            b = _binned(tr[col].to_numpy()[mask], tr["y"].to_numpy()[mask], edges)
            ax.plot(b["center"], b["rate"], "-o", color=CALL_COLORS[call], lw=2.2, ms=5.5,
                    zorder=3)
            table += [dict(panel=col, series=call, **r) for r in b.to_dict("records")]
        b = _binned(tr[col], tr["y"], edges)
        ax.plot(b["center"], b["rate"], color=S.INK_2, lw=1.6, ls=(0, (5, 3)), zorder=2)
        table += [dict(panel=col, series="pooled", **r) for r in b.to_dict("records")]
        ax.axvline(0, color=S.AXIS, lw=1.0, zorder=1)
        ax.set_xlim(-6.6, 6.6)
        ax.set_ylim(-0.03, 1.03)
        _pct_axis(ax)
        ax.set_xlabel(xl, labelpad=8)
        ax.set_title(sub, loc="left", fontsize=13.5)
        S.halo(ax.text(0.03, 0.95, f"AUC {auc_v:.2f}", transform=ax.transAxes,
                       fontsize=16, fontweight="bold", color=S.INK, va="top"))
    axes[0].set_ylabel("share of challenges overturned")
    handles = [Line2D([], [], color=S.ORANGE, lw=2.2, marker="o", ms=5.5, label=CALL_NAMES["strike"]),
               Line2D([], [], color=S.BLUE, lw=2.2, marker="o", ms=5.5, label=CALL_NAMES["ball"]),
               Line2D([], [], color=S.INK_2, lw=1.6, ls=(0, (5, 3)), label="both pooled")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=3,
               fontsize=10.5, frameon=False)
    S.suptitles(fig, "Distance only predicts overturns once it's signed by the call",
                "Training challenges, 1-inch bins (≥10 challenges each). Left: strikes and balls "
                "pull in opposite directions, and pooled they cancel out.", top=1.0)
    S.footnote(fig, "AUC = chance a random overturned challenge scores higher than a random "
                    "one that stood (0.5 = coin flip). Every Week 7 model uses dist_against_call.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w7_01_sign_trap", pd.DataFrame(table))


# ------------------------------------------------------------------ reliability
def _rel_panel(ax, axh, y, p, color, small=False):
    tab = pd.DataFrame(M.reliability_table(y, p, n_bins=10, strategy="uniform"))
    ax.plot([0, 1], [0, 1], color=S.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if len(tab):
        ax.plot(tab["mean_pred"], tab["obs_rate"], color=color, lw=1.1, alpha=0.55, zorder=2)
        nmax = tab["n"].max()
        for _, r in tab.iterrows():
            solid = r["n"] >= 20
            ax.plot([r["mean_pred"]] * 2, [r["ci_lo"], r["ci_hi"]], color=color,
                    lw=1.0 if small else 1.5, alpha=0.75, zorder=2)
            size = (10 if small else 24) + (60 if small else 150) * np.sqrt(r["n"] / nmax)
            ax.scatter(r["mean_pred"], r["obs_rate"], s=size, zorder=3, linewidths=1.3,
                       facecolor=color if solid else "white", edgecolor=color)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, .25, .5, .75, 1])
    _pct_axis(ax)
    ax.tick_params(labelbottom=False)
    _strip(axh, p, color, annotate=not small)
    return tab


def _stats_text(s, fl):
    return (f"Brier {s['brier']:.4f}   log loss {s['log_loss']:.3f}\n"
            f"intercept {s['cal_intercept']:+.2f}   slope {s['cal_slope']:.2f}\n"
            f"ECE {s['ece']:.3f} (noise ≈ {fl:.3f})   AUC {s['auc']:.3f}")


def fig_reliability_baselines(out, demo, va, preds, point, floor, labels, keys=("M2", "GAM-loc")):
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1], hspace=0.06, wspace=0.16)
    table = []
    for j, k in enumerate(keys):
        color = S.ORANGE if k.startswith("GAM") else S.BLUE
        ax = fig.add_subplot(gs[0, j])
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        tab = _rel_panel(ax, axh, va["y"], preds[k], color)
        table += [dict(model=k, **r) for r in tab.to_dict("records")]
        ax.set_title(labels[k], loc="left", fontsize=14)
        S.halo(ax.text(0.98, 0.03, _stats_text(point[k], floor[k]), transform=ax.transAxes,
                       ha="right", va="bottom", fontsize=10, color=S.INK_2, linespacing=1.5))
        if j == 0:
            ax.set_ylabel("actually overturned")
            axh.set_ylabel("challenges", labelpad=8)
        axh.set_xlabel("predicted chance of overturn (10 equal-width bins)")
    S.suptitles(fig, "Do the predicted probabilities come true?",
                f"Validation window ({len(va):,} challenges). Points on the dashed line = "
                "calibrated. Lines = 95% intervals; hollow = fewer than 20 challenges in the bin.",
                top=1.05)
    S.footnote(fig, "Bottom strips: how many challenges got each prediction (the end bars are "
                    "cut short and labelled with their real counts). Most challenges are easy "
                    "calls near 0% or 100%; the uncertain middle is where the MDP's decisions "
                    "are made and where the intervals are widest.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w7_02_reliability_baselines", pd.DataFrame(table))


def fig_reliability_all(out, demo, va, preds, point, floor, labels, order):
    fig = plt.figure(figsize=(12.0, 8.2))
    gs = fig.add_gridspec(5, 4, height_ratios=[3, 0.65, 0.75, 3, 0.65],
                          hspace=0.08, wspace=0.14, top=0.82, bottom=0.06)
    table = []
    for i, k in enumerate(order):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[3 * r, c])
        axh = fig.add_subplot(gs[3 * r + 1, c], sharex=ax)
        color = S.MUTED if k == "M0" else (S.ORANGE if k.startswith("GAM") else S.BLUE)
        tab = _rel_panel(ax, axh, va["y"], preds[k], color, small=True)
        table += [dict(model=k, **rr) for rr in tab.to_dict("records")]
        s = point[k]
        ax.set_title(labels[k], loc="left", fontsize=11.5, pad=16)
        slope = f"{s['cal_slope']:.2f}" if np.isfinite(s["cal_slope"]) else "—"
        ax.text(0, 1.015, f"Brier {s['brier']:.4f} · slope {slope}",
                transform=ax.transAxes, fontsize=9, color=S.INK_2, va="bottom")
        ax.tick_params(labelsize=8.5)
        axh.tick_params(labelsize=8.5)
        if c:
            ax.tick_params(labelleft=False)
    S.suptitles(fig, "Every model's reliability diagram, same validation challenges",
                "Blue = logistic, orange = GAM, gray = base rate. M0 is perfectly calibrated "
                "on average and useless — calibration alone isn't enough.", top=0.97)
    S.footnote(fig, "10 equal-width bins; lines are 95% intervals; hollow points have fewer "
                    "than 20 challenges. Strips below each panel show where the predictions fall "
                    "(end bars cut short).")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w7_03_reliability_all", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 4
def fig_block_contributions(out, demo, steps):
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, sharey=True,
                             gridspec_kw={"wspace": 0.12})
    d = steps.iloc[::-1].reset_index(drop=True)
    ys = np.arange(len(d))
    labels = [f"{r.step}\n({r.bigger} vs {r.smaller})" for r in d.itertuples()]

    def color_for(lo, hi):
        if hi < 0:
            return S.BLUE
        if lo > 0:
            return S.NEG
        return S.MUTED

    for ax, col, title in ((axes[0], "d_brier", "Δ Brier score"),
                           (axes[1], "d_logloss", "Δ log loss")):
        span = np.nanmax(np.abs(d[[f"{col}_lo", f"{col}_hi"]].to_numpy()))
        span = span if np.isfinite(span) and span > 0 else 1e-3
        for yy, r in zip(ys, d.itertuples()):
            lo, hi, v = getattr(r, f"{col}_lo"), getattr(r, f"{col}_hi"), getattr(r, col)
            c = color_for(lo, hi)
            ax.plot([lo, hi], [yy, yy], color=c, lw=3.0, solid_capstyle="round", zorder=2)
            ax.scatter([v], [yy], s=70, color=c, zorder=3, edgecolor="white", linewidth=1.2)
        ax.axvline(0, color=S.INK_2, lw=1.0, zorder=1)
        ax.set_xlim(-1.15 * span, 1.15 * span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f"{v:+.3f}" if abs(v) > 1e-12 else "0"))
        ax.tick_params(axis="x", labelsize=10)
        ax.set_title(title, loc="left", fontsize=13.5)
        ax.set_xlabel("← bigger model better        bigger model worse →", labelpad=8,
                      fontsize=10.5, color=S.MUTED)
        S.only_grid(ax, "x")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=10.5, color=S.INK_2)
    S.pad_cat_axis(axes[0], len(d))
    for yy, r in zip(ys, d.itertuples()):
        if np.isfinite(r.lr_p):
            axes[1].text(1.03, yy, f"LR p = {r.lr_p:.2g}\nin training",
                         transform=axes[1].get_yaxis_transform(),
                         fontsize=9.5, color=S.INK_2, va="center", linespacing=1.2)
    n_better = int((d["d_brier_hi"] < 0).sum())
    title = ("After distance, which blocks still help on August?" if n_better else
             "After distance, nothing else reliably improves August's predictions")
    S.suptitles(fig, title,
                "Paired change in each validation score when the bigger model replaces the "
                "smaller one, with 95% intervals from resampling games.", top=1.06)
    S.footnote(fig, "Blue = interval entirely below zero (reliably better); red = entirely above "
                    "(reliably worse); gray = can't tell. Right margin: the in-sample "
                    "likelihood-ratio test, for contrast — a small p there with a gray bar here "
                    "is a pattern in March–July that didn't carry into August.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w7_04_block_contributions", steps)


# ------------------------------------------------------------------ fig 5
def fig_distance_curves(out, demo, tr, models, edges, base_row):
    xs = np.linspace(-5, 5, 401)
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(2, len(edges), height_ratios=[4, 1], hspace=0.06, wspace=0.08,
                          top=0.80)
    table = []
    bins = np.arange(-5.5, 6.0, 1.0)
    hist_bins = np.arange(-5, 5.25, 0.5)
    ax0 = None
    for j, e in enumerate(edges):
        ax = fig.add_subplot(gs[0, j], sharey=ax0)
        ax0 = ax0 or ax
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        sub = tr[tr["edge"] == e]
        stack = []
        for call, cd in (("strike", -1.0), ("ball", 1.0)):
            color = CALL_COLORS[call]
            g = pd.DataFrame([dict(base_row, edge=e, call_dir=cd, called_strike=(cd < 0))] * len(xs))
            g["dist_against_call"] = xs
            p, lo, hi = models["GAM-loc"].predict_band(g)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.16, lw=0, zorder=1)
            ax.plot(xs, p, color=color, lw=2.4, zorder=3)
            pl = models["M2"].predict(g)
            ax.plot(xs, pl, color=color, lw=1.4, ls=(0, (4, 2.5)), zorder=4)
            m = sub["call_dir"].to_numpy() == cd
            b = _binned(sub["dist_against_call"].to_numpy()[m], sub["y"].to_numpy()[m], bins)
            ax.scatter(b["center"], b["rate"], s=16 + 3 * np.sqrt(b["n"]), facecolor="white",
                       edgecolor=color, linewidths=1.3, zorder=5)
            table += [dict(edge=e, call=call, x=x, gam=a, gam_lo=l_, gam_hi=h_, logistic=q)
                      for x, a, l_, h_, q in zip(xs[::10], p[::10], lo[::10], hi[::10], pl[::10])]
            dd = sub["dist_against_call"].to_numpy()[m]
            stack.append(dd[(dd >= -5) & (dd <= 5)])           # far-off challenges not shown
        axh.hist(stack, bins=hist_bins, color=[CALL_COLORS["strike"], CALL_COLORS["ball"]],
                 stacked=True, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=S.AXIS, lw=1.0, zorder=0)
        ax.axhline(0.5, color=S.AXIS, lw=1.0, zorder=0)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.02, 1.02)
        _pct_axis(ax)
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{e.capitalize()} edge", loc="left", fontsize=13.5)
        ax.text(1.0, 1.02, f"{len(sub):,} training challenges", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, color=S.MUTED)
        axh.grid(False)
        axh.set_yticks([])
        axh.spines["left"].set_visible(False)
        axh.set_xticks([-4, -2, 0, 2, 4])
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("chance of overturn")
        if j == len(edges) // 2:
            axh.set_xlabel("inches past our computed edge, against the call "
                           "(+ = wrong side for the call)", labelpad=8)
    handles = [Line2D([], [], color=S.ORANGE, lw=2.4, label=CALL_NAMES["strike"]),
               Line2D([], [], color=S.BLUE, lw=2.4, label=CALL_NAMES["ball"]),
               Line2D([], [], color=S.INK_2, lw=2.4, label="GAM curve (band = 95%)"),
               Line2D([], [], color=S.INK_2, lw=1.4, ls=(0, (4, 2.5)),
                      label="logistic M2 (straight line in log-odds)"),
               Line2D([], [], color=S.INK_2, marker="o", mfc="white", lw=0,
                      label="observed, 1-inch bins")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.935), ncol=3,
               fontsize=10, frameon=False)
    S.suptitles(fig, "How fast the overturn chance climbs past each edge",
                "Fitted on March–July. Where a curve crosses 50% is the coin-flip point; how "
                "steeply it climbs is how sharp that edge is.", top=1.02)
    S.footnote(fig, "The sides are fixed at 17 inches; the top and bottom come from listed "
                    "height, so expect them to be blurrier. If the orange and blue curves cross "
                    "50% on opposite sides of zero, the real ABS edge isn't where our zone puts it. "
                    "Bottom strips: where the training challenges were (stacked; beyond ±5 in. not shown).")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w7_05_distance_curves", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 6
def _block_of(term):
    base = term.split(" | ")[0].split("=")[0]
    for name, b in BLOCKS.items():
        if base in b.get("numeric", []) + b.get("categorical", []) + list(b.get("slopes_by", {})):
            return name
    return "other"


def fig_odds_ratios(out, demo, model, term_unit):
    t = model.table()
    t = t[t["term"] != "(intercept)"].copy()
    rows = []
    for r in t.itertuples():
        scale, per = term_unit(r.term, model.design)
        rows.append({"term": r.term, "label": per, "block": _block_of(r.term),
                     "odds": np.exp(scale * r.coef),
                     "lo": np.exp(scale * (r.coef - 1.96 * r.se)),
                     "hi": np.exp(scale * (r.coef + 1.96 * r.se)), "p": r.p_value})
    d = pd.DataFrame(rows)
    border = {b: i for i, b in enumerate(list(BLOCKS) + ["other"])}
    d["border"] = d["block"].map(border)
    d = d.sort_values(["border"], kind="stable").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=S.FULL,
                             gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.95})
    for ax, part, title in ((axes[0], d[d["block"] == "location"], "Location — what ABS sees"),
                            (axes[1], d[d["block"] != "location"],
                             "Everything else — who chose to challenge")):
        part = part.iloc[::-1].reset_index(drop=True)
        ys = np.arange(len(part))
        for yy, r in zip(ys, part.itertuples()):
            sig = (r.lo > 1) or (r.hi < 1)
            c = S.BLUE if sig else S.MUTED
            ax.plot([r.lo, r.hi], [yy, yy], color=c, lw=2.6, solid_capstyle="round", zorder=2)
            ax.scatter([r.odds], [yy], s=46, color=c, zorder=3, edgecolor="white", linewidth=1.0)
        # thin rules + names between blocks
        if len(part):
            blocks = part["block"].tolist()
            for i in range(1, len(blocks)):
                if blocks[i] != blocks[i - 1]:
                    ax.axhline(i - 0.5, color=S.GRID, lw=1.0, zorder=0)
            for b in dict.fromkeys(blocks):
                top_i = max(i for i, bb in enumerate(blocks) if bb == b)
                ax.text(1.0, top_i + 0.32, b, transform=ax.get_yaxis_transform(), ha="right",
                        va="bottom", fontsize=9, color=S.MUTED, style="italic")
        ax.axvline(1, color=S.INK_2, lw=1.0, zorder=1)
        ax.set_xscale("log")
        ax.set_yticks(ys)
        ax.set_yticklabels(part["label"], fontsize=9.5, color=S.INK_2)
        S.pad_cat_axis(ax, len(part))
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel("odds ratio (log scale; 1 = no effect)", labelpad=8)
        S.only_grid(ax, "x")
        lo_all = np.nanmin(part[["lo", "odds"]].to_numpy()) if len(part) else 0.5
        hi_all = np.nanmax(part[["hi", "odds"]].to_numpy()) if len(part) else 2
        lo_all, hi_all = min(lo_all, 0.8), max(hi_all, 1.25)
        ax.set_xlim(lo_all / 1.3, hi_all * 1.3)
        cand = np.array([0.01, 0.03, 0.1, 0.25, 0.5, 1, 2, 4, 10, 30, 100, 300, 1000,
                         3000, 10000, 30000, 100000])
        tk = cand[(cand >= lo_all / 1.3) & (cand <= hi_all * 1.3)]
        if len(tk) > 7:
            tk = tk[::2] if 1 in tk[::2] else tk[1::2]
        ax.xaxis.set_major_locator(FixedLocator(tk))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    S.suptitles(fig, "At a fixed location, does anything else move the odds of an overturn?",
                "Odds ratios from M5 (every block), fitted on March–July, with 95% intervals. "
                "Blue = interval excludes 1.", top=1.06)
    S.footnote(fig, "ABS only sees the ball, so the right-hand panel cannot be about how ABS "
                    "rules: an effect there means the challenges made in that situation were "
                    "different (selection). Units are chosen to be readable — see the labels.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w7_06_odds_ratios", d.drop(columns="border"))


# ------------------------------------------------------------------ fig 7
def fig_subgroups(out, demo, sub, months, label):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.4),
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.42})
    ax = axes[0]
    ys, labels, y = [], [], 0
    heads = []
    for g, part in sub.groupby("group", sort=False):
        heads.append((y, g))
        y += 1
        for r in part.itertuples():
            gap, lo, hi = r.obs - r.pred, r.lo - r.pred, r.hi - r.pred
            c = S.NEG if (lo > 0 or hi < 0) else S.BLUE
            ax.plot([100 * lo, 100 * hi], [-y, -y], color=c, lw=2.4, solid_capstyle="round")
            ax.scatter([100 * gap], [-y], s=40, color=c, zorder=3, edgecolor="white", linewidth=1)
            ys.append(-y)
            labels.append(f"{r.level}  (n={r.n})")
            y += 1
        y += 0.4
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, color=S.INK_2)
    for yy, g in heads:
        ax.text(-0.02, -yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, fontweight="semibold", color=S.INK)
    ax.axvline(0, color=S.INK_2, lw=1.0)
    ax.set_ylim(-y + 0.2, 0.8)
    ax.set_xlabel("observed − predicted overturn rate (percentage points)", labelpad=8)
    ax.set_title("Validation slices", loc="left", fontsize=13)
    S.only_grid(ax, "x")
    ax.tick_params(axis="y", length=0, pad=6)

    ax = axes[1]
    mo = months.reset_index(drop=True)
    for i, r in enumerate(mo.itertuples()):
        gap, lo, hi = r.obs - r.pred, r.lo - r.pred, r.hi - r.pred
        insample = r.window.startswith("train")
        c = S.MUTED if insample else S.ORANGE
        ax.plot([i, i], [100 * lo, 100 * hi], color=c, lw=2.4, solid_capstyle="round")
        ax.scatter([i], [100 * gap], s=46, zorder=3, linewidths=1.4,
                   facecolor="white" if insample else c, edgecolor=c)
    ax.axhline(0, color=S.INK_2, lw=1.0)
    ax.set_xticks(range(len(mo)))
    ax.set_xticklabels([pd.Period(m).strftime("%b") for m in mo["month"]])
    ax.set_ylabel("observed − predicted (points)")
    ax.set_title("By month", loc="left", fontsize=13)
    S.only_grid(ax, "y")
    ax.text(0.98, 0.02, "hollow gray = training months (in-sample)\n"
                        "filled orange = August (validation)",
            transform=ax.transAxes, fontsize=9.5, color=S.INK_2, va="bottom", ha="right")
    S.suptitles(fig, f"Is {label} right for every kind of challenge?",
                "Gap between what actually happened and what the model predicted, "
                "with 95% intervals. Zero = calibrated for that slice.", top=1.06)
    S.footnote(fig, "Red = interval excludes zero. Slices are small, so most intervals are wide — "
                    "Week 9 repeats this with more data. A steady drift across months would mean "
                    "the same location was overturned at different rates as the season went on.")
    S.demo_stamp(fig, demo)
    tab = pd.concat([sub.assign(panel="slices"), months.assign(panel="months")], ignore_index=True)
    S.save(fig, out, "w7_07_subgroup_calibration", tab)


# ------------------------------------------------------------------ all
def make_all(out, data_dir, *, tr, va, models, preds_va, point, ci, floor, steps, loc_tab,
             sub, months, mdp_key, chosen_key, auc_raw, auc_res, labels, xs, edges,
             base_row, term_unit, order):
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    fig_sign_trap(out, demo, tr, auc_raw, auc_res)
    fig_reliability_baselines(out, demo, va, preds_va, point, floor, labels)
    fig_reliability_all(out, demo, va, preds_va, point, floor, labels, order)
    fig_block_contributions(out, demo, steps)
    fig_distance_curves(out, demo, tr, models, edges, base_row)
    fig_odds_ratios(out, demo, models["M5"], term_unit)
    fig_subgroups(out, demo, sub, months, labels[mdp_key])
