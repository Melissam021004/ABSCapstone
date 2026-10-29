"""
week8_figures.py — the six Week 8 figures. Called by 16_week8_boosting.py;
not meant to be run on its own.

Same house style as Week 7: white background, 16:9, 200 DPI, and every PNG
has a companion *_data.csv holding the exact plotted numbers.

Colour follows the model FAMILY in figures 1–3: blue = logistic, orange =
GAM, aqua = gradient boosting (CatBoost circles, LightGBM diamonds,
HistGradientBoosting squares). Figure 4 colours features by BLOCK. Figures 5
and 6 colour by the CALL, exactly as Week 7 did (orange = called strike, the
batter challenged; blue = called ball, the fielder challenged).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging                              # noqa: E402

import eda_style as S                       # noqa: E402
import matplotlib.patches as mpatches       # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator   # noqa: E402
import week7_figures as W7                  # noqa: E402
from src.abs_features import FEATURE_BLOCK, FEATURE_LABEL               # noqa: E402
from src.abs_zone import BALL_RADIUS_FT, PLATE_HALF_WIDTH_FT, zone_from_height  # noqa: E402

BOOSTED = ("catboost", "lightgbm", "hgb")
FAMILY_COLOR = {"logistic": S.BLUE, "gam": S.ORANGE,
                "catboost": S.AQUA, "lightgbm": S.AQUA, "hgb": S.AQUA}
MARKER = {"logistic": "o", "gam": "o", "catboost": "o", "lightgbm": "D", "hgb": "s"}
LIB_NAME = {"catboost": "CatBoost", "lightgbm": "LightGBM", "hgb": "HistGB (stand-in)"}
SET_HEAD = {"location": "Location only", "full": "Everything"}
SET_TITLE = {"location": "Location only — what ABS sees",
             "full": "Everything — incl. umpire, catcher, pitcher & batter IDs"}
CALL_COLORS = W7.CALL_COLORS
CALL_NAMES = W7.CALL_NAMES
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # "semibold" fallbacks


def _color(k, family):
    return S.MUTED if k == "M0" else FAMILY_COLOR.get(family[k], S.INK_2)


def _fmt(d):
    return plt.FuncFormatter(lambda v, _: f"{v:.{d}f}")


def _typical_row(tr):
    """Median numerics / most common categories (same as the Week 7 script)."""
    row = {}
    for c in tr.columns:
        s = tr[c]
        if s.dtype.kind in "if":
            row[c] = float(s.median()) if s.notna().any() else 0.0
        elif s.dtype.kind == "b":
            row[c] = bool(s.mode().iloc[0]) if len(s) else False
        else:
            row[c] = s.mode().iloc[0] if s.notna().any() else ""
    return row


def _rows(groups):
    """y positions for grouped rows (top to bottom), with a header line per group."""
    ys, heads, y = {}, [], 0.0
    for name, keys in groups:
        if not keys:
            continue
        heads.append((-y, name))
        y += 1
        for k in keys:
            ys[k] = -y
            y += 1
        y += 0.35
    return ys, heads, -(y - 0.35)


def _heads(ax, heads):
    for yy, name in heads:
        ax.text(-0.015, yy, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=11, fontweight="semibold", color=S.INK)


# ------------------------------------------------------------------ fig 1
def fig_tuning(out, demo, log, point, labels, baseline, peers):
    ok = log.copy()
    if "error" in ok:
        ok = ok[ok["error"].fillna("") == ""]
    ok = ok[ok["august_logloss"].notna()] if "august_logloss" in ok else ok.iloc[0:0]
    if not len(ok):
        print("  [skip] w8_01_tuning — no successful tuning runs")
        return
    libs = [b for b in BOOSTED if (ok["backend"] == b).any()]
    sets = [s for s in ("location", "full") if (ok["feature_set"] == s).any()]
    groups = [(SET_HEAD[s], [(s, b) for b in libs if ((ok["backend"] == b) &
                                                       (ok["feature_set"] == s)).any()])
              for s in sets]
    ys, heads, ybot = _rows(groups)

    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(top=0.76, left=0.20, right=0.86)
    rng = np.random.default_rng(7)
    base_ll = point[baseline]["log_loss"]
    table, best_lls = [], []
    for (s, b), yy in ys.items():
        sub = ok[(ok["backend"] == b) & (ok["feature_set"] == s)].reset_index(drop=True)
        x = sub["august_logloss"].to_numpy(float)
        jit = rng.uniform(-0.17, 0.17, len(x))
        ib = int(np.argmin(x))
        rest = np.arange(len(x)) != ib
        ax.scatter(x[rest], yy + jit[rest], s=36, facecolor="white", edgecolor=S.AQUA,
                   linewidths=1.3, marker=MARKER[b], zorder=3)
        ax.scatter([x[ib]], [yy], s=120, color=S.AQUA, marker=MARKER[b], edgecolor="white",
                   linewidth=1.3, zorder=4)
        S.halo(ax.text(x[ib], yy + 0.24, f"{x[ib]:.4f}", ha="center", va="bottom",
                       fontsize=9.5, color=S.INK, zorder=5))
        best_lls.append(x[ib])
        rho = np.nan
        if "early_stop_logloss" in sub and sub["early_stop_logloss"].notna().sum() >= 4:
            rho = float(sub["early_stop_logloss"].corr(sub["august_logloss"], method="spearman"))
        ax.text(1.015, yy, f"{len(sub)} settings" + (f"\nρ = {rho:.2f}" if np.isfinite(rho) else ""),
                transform=ax.get_yaxis_transform(), fontsize=9.5, color=S.INK_2, va="center",
                linespacing=1.25)
        for i, r in sub.iterrows():
            table.append({"feature_set": s, "library": b, "trial": r.get("trial"),
                          "august_logloss": r["august_logloss"],
                          "early_stop_logloss": r.get("early_stop_logloss"),
                          "trees": r.get("trees"), "best": i == ib, "params": r.get("params"),
                          "spearman_rho_inner_vs_august": rho})
    # reference lines: the baseline everywhere, each set's GAM within its own group
    ax.axvline(base_ll, color=S.INK_2, lw=1.5, ls=(0, (5, 3)), zorder=2)
    for s in sets:
        gam = peers[s][1]
        rows_s = [yy for (ss, _), yy in ys.items() if ss == s]
        if gam in point and gam != baseline and rows_s:
            ax.plot([point[gam]["log_loss"]] * 2, [min(rows_s) - 0.45, max(rows_s) + 0.45],
                    color=S.ORANGE, lw=1.6, ls=(0, (1.2, 2.2)), zorder=2)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([LIB_NAME[b] for (_, b) in ys], fontsize=11, color=S.INK_2)
    _heads(ax, heads)
    ax.set_ylim(ybot - 0.6, 0.6)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.xaxis.set_major_formatter(_fmt(3))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("August log loss (lower = better)", labelpad=8)
    S.only_grid(ax, "x")
    m0 = MARKER[libs[0]]
    handles = [Line2D([], [], color=S.AQUA, marker=m0, mfc="white", lw=0, ms=7,
                      label="one setting tried"),
               Line2D([], [], color=S.AQUA, marker=m0, lw=0, ms=10,
                      label="best setting (its August log loss)"),
               Line2D([], [], color=S.INK_2, lw=1.5, ls=(0, (5, 3)),
                      label=f"baseline ★ {labels[baseline]}  ({base_ll:.4f})"),
               Line2D([], [], color=S.ORANGE, lw=1.6, ls=(0, (1.2, 2.2)),
                      label="the GAM with the same features")]
    if len(libs) > 1:
        handles += [Line2D([], [], color=S.AQUA, marker=MARKER[b], mfc="white", lw=0, ms=7,
                           label=LIB_NAME[b]) for b in libs]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.915), ncol=3,
               fontsize=10, frameon=False)
    below = sum(b_ < base_ll for b_ in best_lls)
    title = ("Tuned boosting gets under the baseline's August log loss" if below else
             "Even the best-tuned settings don't get under the baseline's log loss")
    S.suptitles(fig, title,
                "Every setting from the seeded random search, scored on August. Picking the best "
                "of many on August flatters it a little; Week 9's test month is the honest check.",
                top=1.0)
    S.footnote(fig, "ρ = rank agreement between each setting's score on the held-back last 15% "
                    "of training games and its August score. Near 1: both agree which settings are "
                    "good, so the August pick isn't luck. The number of trees for every setting was "
                    "chosen inside the training window; August never fitted anything.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w8_01_tuning", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 2
def _verdict(lo, hi):
    if hi < 0:
        return "reliably better", S.INK
    if lo > 0:
        return "reliably worse", S.INK
    return "can't tell apart", S.MUTED


def fig_comparison(out, demo, show, point, ci, diffs, labels, family, fsets, baseline, vs):
    groups = [(SET_HEAD[s], [k for k in show if fsets[k] == s]) for s in ("location", "full")]
    ys, heads, ybot = _rows(groups)
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, sharey=True,
                             gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1.05]})
    fig.subplots_adjust(top=0.78, left=0.21, right=0.85)
    table = []
    cal = {r.model: r.calibration_ok for r in vs.itertuples()} if len(vs) else {}
    for k, yy in ys.items():
        c, mk = _color(k, family), MARKER.get(family[k], "o")
        s, lo, hi = point[k]["brier"], *ci[k]["brier"]
        axes[0].plot([lo, hi], [yy, yy], color=c, lw=3.0, solid_capstyle="round", zorder=2)
        axes[0].scatter([s], [yy], s=80, color=c, marker=mk, zorder=3, edgecolor="white",
                        linewidth=1.2)
        row = {"model": k, "label": labels[k], "family": family[k], "feature_set": fsets[k],
               "brier": s, "brier_lo": lo, "brier_hi": hi}
        if k == baseline:
            S.halo(axes[1].text(0, yy, "★ the baseline", ha="center", va="center", fontsize=10,
                                color=S.INK, fontweight="semibold", zorder=4))
            row.update(d_brier=0.0, d_lo=0.0, d_hi=0.0, verdict="baseline")
        else:
            d, dlo, dhi = diffs[(k, baseline)]["brier"]
            axes[1].plot([dlo, dhi], [yy, yy], color=c, lw=3.0, solid_capstyle="round", zorder=2)
            axes[1].scatter([d], [yy], s=80, color=c, marker=mk, zorder=3, edgecolor="white",
                            linewidth=1.2)
            txt, tc = _verdict(dlo, dhi)
            if k in cal and dhi < 0 and not cal[k]:
                txt, tc = "better Brier,\nworse calibration", S.INK
            axes[1].text(1.02, yy, txt, transform=axes[1].get_yaxis_transform(), fontsize=9.5,
                         color=tc, va="center", linespacing=1.15)
            row.update(d_brier=d, d_lo=dlo, d_hi=dhi, verdict=txt.replace("\n", " "))
        table.append(row)
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels([labels[k] + (" ★" if k == baseline else "") for k in ys],
                            fontsize=10.5, color=S.INK_2)
    _heads(axes[0], heads)
    axes[0].set_ylim(ybot - 0.6, 0.6)
    for ax in axes:
        ax.tick_params(axis="y", length=0, pad=8)
        S.only_grid(ax, "x")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].xaxis.set_major_formatter(_fmt(3))
    axes[0].set_title("August Brier score [95%]", loc="left", fontsize=13.5)
    axes[0].set_xlabel("lower = better", labelpad=8, fontsize=10.5, color=S.MUTED)
    lim = [abs(v) for r in table for v in (r["d_lo"], r["d_hi"]) if np.isfinite(v)]
    span = max(lim) if lim else 1e-3
    span = span if span > 0 else 1e-3
    axes[1].set_xlim(-1.15 * span, 1.15 * span)
    axes[1].axvline(0, color=S.INK_2, lw=1.0, zorder=1)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:+.3f}" if abs(v) > 1e-12 else "0"))
    axes[1].set_title(f"minus the baseline, same games [95%]", loc="left", fontsize=13.5)
    axes[1].set_xlabel("← better than the baseline        worse →", labelpad=8, fontsize=10.5,
                       color=S.MUTED)
    beat = [r.label for r in vs.itertuples() if r.beats_baseline] if len(vs) else []
    if beat:
        title = (f"{beat[0]} reliably beats the baseline" if len(beat) == 1 else
                 f"{len(beat)} boosted models reliably beat the baseline")
    else:
        title = "No boosted model reliably beats the baseline"
    S.suptitles(fig, title,
                "Same August challenges and metric for every model. Intervals come from resampling "
                "whole games; the right panel pairs each model with the baseline on every resample.",
                top=1.0)
    S.footnote(fig, "Blue = logistic, orange = GAM, aqua = gradient boosting (○ CatBoost, ◇ "
                    "LightGBM, □ HistGB). 'Beats the baseline' needs the right-hand interval "
                    "entirely left of zero AND calibration no worse — the rule fixed in Week 7. "
                    "Overlapping intervals on the left can still separate on the right, because "
                    "pairing removes the noise the two models share.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w8_02_comparison", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 3
def fig_reliability(out, demo, va, preds, point, floor, labels, family, grid):
    grid = [(s, ks) for s, ks in grid if ks]
    ncol = max(len(ks) for _, ks in grid)
    nrow = len(grid)
    ratios = [3, 0.65]
    for _ in range(nrow - 1):
        ratios += [0.85, 3, 0.65]
    fig = plt.figure(figsize=(12.0, 4.4 * nrow))
    gs = fig.add_gridspec(len(ratios), ncol, height_ratios=ratios, hspace=0.08, wspace=0.14,
                          top=0.82 if nrow > 1 else 0.74, bottom=0.06)
    table = []
    for r, (s, ks) in enumerate(grid):
        for c, k in enumerate(ks):
            ax = fig.add_subplot(gs[3 * r, c])
            axh = fig.add_subplot(gs[3 * r + 1, c], sharex=ax)
            tab = W7._rel_panel(ax, axh, va["y"], preds[k], _color(k, family), small=True)
            table += [dict(model=k, feature_set=s, **rr) for rr in tab.to_dict("records")]
            st = point[k]
            ax.set_title(labels[k], loc="left", fontsize=11.5, pad=16)
            slope = f"{st['cal_slope']:.2f}" if np.isfinite(st["cal_slope"]) else "—"
            fl = f" (noise {floor[k]:.3f})" if floor and k in floor else ""
            ax.text(0, 1.015, f"Brier {st['brier']:.4f} · slope {slope} · ECE {st['ece']:.3f}{fl}",
                    transform=ax.transAxes, fontsize=8.5, color=S.INK_2, va="bottom")
            ax.tick_params(labelsize=8.5)
            axh.tick_params(labelsize=8.5)
            if c:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(f"{SET_HEAD[s]}\n\nactually overturned", fontsize=10)
    n_fam = len({family[k] for _, ks in grid for k in ks})
    S.suptitles(fig, f"Do the probabilities come true? {n_fam} model families, same August games",
                f"August ({len(va):,} challenges). Top row: location only. Bottom row: everything. "
                "Points on the dashed line = calibrated.", top=0.97 if nrow > 1 else 0.95)
    S.footnote(fig, "Blue = logistic, orange = GAM, aqua = gradient boosting. 10 equal-width bins; "
                    "lines are 95% intervals; hollow points have fewer than 20 challenges. Boosted "
                    "models often push predictions toward 0% and 100% harder than the data supports "
                    "— that shows up as a slope below 1 and is what Week 9's calibration fixes.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w8_03_reliability", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 4
BLOCK_COLOR = {"location": S.BLUE, "identity": S.ORANGE}


def _ratio_title(imps):
    """Title for the importance figure: the plan's leak rule, then how big the gap is."""
    full = imps.get("full")
    if full is None:
        return "What the location model relies on"
    imp = full["imp"]
    blk = imp["feature"].map(lambda f_: FEATURE_BLOCK.get(f_, "other"))
    loc = imp[blk == "location"]
    ids = imp[blk == "identity"]
    if full["leak"] and len(ids):
        return (f"⚠ {FEATURE_LABEL.get(ids['feature'].iloc[0], ids['feature'].iloc[0])} "
                "outranks every location feature — suspect a leak")
    if not len(loc):
        return "What the boosted models rely on"
    top_loc = FEATURE_LABEL.get(loc["feature"].iloc[0], loc["feature"].iloc[0])
    if not len(ids) or ids["importance"].iloc[0] <= 0:
        return f"Location does the work; no ID helps at all"
    ratio = loc["importance"].iloc[0] / ids["importance"].iloc[0]
    return (f"Location does the work: {top_loc} counts {ratio:,.0f}× more than any ID"
            if ratio >= 1.5 else f"IDs matter almost as much as {top_loc} — check for a leak")


def fig_importance(out, demo, imps, labels):
    """
    A dot plot on a LOG scale. Distance from the edge dwarfs everything else,
    so on an ordinary scale every other bar is a sliver and the leak check
    (where do the IDs rank?) can't be read. On a log scale each step on the
    axis is ×10, so 'distance counts 40× more than the umpire' is visible.
    """
    sets = [s for s in ("location", "full") if s in imps]
    if not sets:
        return
    nshow = {s: min(len(imps[s]["imp"]), 6 if s == "location" else 12) for s in sets}
    nmax = max(nshow.values())
    shown = pd.concat([imps[s]["imp"].head(nshow[s]) for s in sets])
    pos = shown["importance"][shown["importance"] > 0]
    vmax = float(pos.max()) if len(pos) else 1.0
    floor = float(pos.min()) / 3 if len(pos) else vmax * 1e-4
    floor = min(max(floor, vmax * 1e-6), vmax / 100)    # 2 to 6 decades on the axis
    kinds = {imps[s]["kind"] for s in sets}
    fig, axes = plt.subplots(1, len(sets), figsize=S.FULL, squeeze=False,
                             gridspec_kw={"wspace": 0.62})
    axes = axes[0]
    fig.subplots_adjust(top=0.78)
    table = []
    for ax, s in zip(axes, sets):
        d = imps[s]
        imp = d["imp"].head(nshow[s]).reset_index(drop=True)
        ys = nmax - 1 - np.arange(len(imp))
        for yy, r in zip(ys, imp.itertuples()):
            blk = FEATURE_BLOCK.get(r.feature, "other")
            col = BLOCK_COLOR.get(blk, S.DEEMPH)
            v = float(r.importance)
            ax.scatter([max(v, floor)], [yy], s=80, zorder=3, linewidths=1.4,
                       facecolor="white" if v <= 0 else col,
                       edgecolor=col if col != S.DEEMPH else S.MUTED)
            lab = (("0" if d["kind"] == "shap" else "≤ 0") if v <= 0 else f"{v:.3g}")
            S.halo(ax.text(max(v, floor) * 1.35, yy, lab, va="center", fontsize=9.5,
                           color=S.INK_2, zorder=4))
            table.append({"feature_set": s, "model": d["key"], "kind": d["kind"],
                          "rank": int(r.Index) + 1, "feature": r.feature,
                          "label": FEATURE_LABEL.get(r.feature, r.feature), "block": blk,
                          "importance": v})
        ax.set_xscale("log")
        ax.set_xlim(floor / 1.6, vmax * 6)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_yticks(ys)
        ax.set_yticklabels([FEATURE_LABEL.get(f_, f_) for f_ in imp["feature"]], fontsize=10.5,
                           color=S.INK_2)
        S.pad_cat_axis(ax, nmax, pad=0.7)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.grid(axis="y", color=S.GRID, lw=0.8)
        ax.grid(axis="x", which="major", color=S.GRID, lw=0.8)
        ax.set_title(labels[d["key"]], loc="left", fontsize=13.5)
        ax.set_xlabel(("mean |SHAP|, log-odds" if d["kind"] == "shap"
                       else "rise in August log loss when shuffled") + "  (log scale)", labelpad=8)
    handles = [Line2D([], [], color=S.BLUE, marker="o", lw=0, ms=8,
                      label="location — what ABS sees"),
               Line2D([], [], color=S.ORANGE, marker="o", lw=0, ms=8,
                      label="identity — who's involved (ABS can't see it)"),
               Line2D([], [], color=S.DEEMPH, marker="o", lw=0, ms=8, mec=S.MUTED,
                      label="situation, pitch, people")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.915), ncol=3,
               fontsize=10.5, frameon=False)
    what = ("SHAP splits every August prediction into one contribution per feature; the dot is "
            "the average size." if kinds == {"shap"} else
            "How much worse August gets when one feature's values are shuffled.")
    S.suptitles(fig, _ratio_title(imps), f"The best boosted model in each feature set. {what}",
                top=1.0)
    S.footnote(fig, "Log scale: each gridline step is ×10. The plan's leak test: ABS only sees "
                    "the ball, so if an umpire, catcher, pitcher or batter ID outranked the pitch's "
                    "location, the model would be learning something other than the zone (e.g. who "
                    "happened to win challenges in the training months) — a red flag, not a "
                    "discovery. Hollow dots: no measurable effect.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w8_04_importance", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 5
def _pd_curve(model, df, col, grid):
    """Centred partial dependence on the log-odds scale (HistGB fallback)."""
    vals = []
    for g in grid:
        dd = df.copy()
        dd[col] = g
        p = model.predict(dd)
        vals.append(float(np.mean(np.log(p / (1 - p)))))
    vals = np.array(vals)
    obs = np.clip(df[col].to_numpy(float), grid.min(), grid.max())
    return vals - np.mean(np.interp(obs, grid, vals))


def fig_shap_distance(out, demo, va, imps, models, labels):
    sets = [s for s in ("location", "full") if s in imps]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=S.FULL, sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.1})
    axes = axes[0]
    fig.subplots_adjust(top=0.78)
    x = va["dist_against_call"].to_numpy(float)
    strike = va["called_strike"].to_numpy(bool)
    show = (x >= -6) & (x <= 6)
    edges = np.arange(-6, 6.5, 0.5)
    table = []
    used_shap = used_pd = False
    for ax, s in zip(axes, sets):
        d = imps[s]
        sh = d.get("shap")
        if sh is not None and "dist_against_call" in sh.columns:
            yv = sh["dist_against_call"].to_numpy(float)
            for call, m in (("strike", strike), ("ball", ~strike)):
                mm = m & show
                ax.scatter(x[mm], yv[mm], s=14, color=CALL_COLORS[call], alpha=0.45, lw=0,
                           zorder=2)
            idx = np.digitize(x, edges) - 1
            cx, cy = [], []
            for b in range(len(edges) - 1):
                m = (idx == b) & show
                if m.sum() >= 5:
                    cx.append((edges[b] + edges[b + 1]) / 2)
                    cy.append(float(yv[m].mean()))
            ax.plot(cx, cy, color=S.INK, lw=2.2, zorder=3)
            used_shap = True
            table += [dict(feature_set=s, model=d["key"], kind="shap", dist_in=a, shap=b_,
                           call="strike" if st else "ball")
                      for a, b_, st, keep in zip(x, yv, strike, show) if keep]
            ax.set_title(labels[d["key"]], loc="left", fontsize=13.5)
        else:
            grid = np.linspace(-6, 6, 49)
            pdv = _pd_curve(models[d["key"]], va, "dist_against_call", grid)
            ax.plot(grid, pdv, color=S.INK, lw=2.4, zorder=3)
            table += [dict(feature_set=s, model=d["key"], kind="partial_dependence", dist_in=g,
                           shap=v) for g, v in zip(grid, pdv)]
            ax.set_title(labels[d["key"]], loc="left", fontsize=13.5)
            ax.text(0, 1.015, "partial dependence — this library has no SHAP",
                    transform=ax.transAxes, fontsize=9.5, color=S.MUTED, va="bottom")
            used_pd = True
        ax.axhline(0, color=S.AXIS, lw=1.0, zorder=1)
        ax.axvline(0, color=S.AXIS, lw=1.0, zorder=1)
        ax.set_xlim(-6.2, 6.2)
        ax.set_xlabel("inches on the wrong side of the edge for the call (+)", labelpad=8)
        S.only_grid(ax, "y")
    axes[0].set_ylabel("contribution to the log-odds of an overturn")
    handles = []
    if used_shap:
        handles += [Line2D([], [], color=S.ORANGE, marker="o", lw=0, ms=6, alpha=0.7,
                           label=CALL_NAMES["strike"]),
                    Line2D([], [], color=S.BLUE, marker="o", lw=0, ms=6, alpha=0.7,
                           label=CALL_NAMES["ball"]),
                    Line2D([], [], color=S.INK, lw=2.2, label="average in half-inch bins")]
    if used_pd:
        handles.append(Line2D([], [], color=S.INK, lw=2.4,
                              label="partial dependence: every August challenge moved to this "
                                    "distance, average log-odds"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.915), ncol=3,
               fontsize=10.5, frameon=False)
    sub = ("Each dot is an August challenge: how far its distance alone moved the prediction "
           "away from the model's average, in log-odds (+1 ≈ odds × 2.7)." if used_shap else
           "How the predicted log-odds change as distance changes, everything else as it was "
           "(+1 ≈ odds × 2.7).")
    S.suptitles(fig, "How the boosted models use distance from the edge", sub, top=1.0)
    S.footnote(fig, "Trees cut distance into intervals, so the effect moves in steps; the biggest "
                    "step should sit near zero, where the pitch crosses the edge. The monotone "
                    "constraint means no challenge's prediction can fall as its pitch moves further "
                    "onto the wrong side. Vertical spread at one distance = the same distance "
                    "counting differently at different edges. Beyond ±6 in. not shown.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w8_05_shap_distance", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 6
def _x50(xs, p):
    """Where a curve first crosses 50% (linear interpolation), NaN if it never does."""
    p = np.asarray(p, float)
    above = np.flatnonzero(p >= 0.5)
    if not len(above) or above[0] == 0:
        return np.nan
    i = above[0]
    return float(np.interp(0.5, [p[i - 1], p[i]], [xs[i - 1], xs[i]]))


def _edge_rows(base, edge, call_dir, d_in, height):
    """
    Pitch locations that really ARE `d_in` inches against the call at the
    middle of `edge`, for a batter of `height` feet — so the boosted model is
    asked about pitches that can exist (its plate_x / plate_z agree with the
    distance). Side edges come back twice, once on each side of the plate.
    """
    top, bot = zone_from_height(height)
    s_ft = np.asarray(d_in, float) * (1.0 if call_dir < 0 else -1.0) / 12.0   # + = outside zone
    n = len(s_ft)
    if edge == "side":
        xs = [PLATE_HALF_WIDTH_FT + BALL_RADIUS_FT + s_ft, -(PLATE_HALF_WIDTH_FT + BALL_RADIUS_FT + s_ft)]
        zs = [np.full(n, (top + bot) / 2)] * 2
    elif edge == "top":
        xs, zs = [np.zeros(n)], [top + BALL_RADIUS_FT + s_ft]
    else:
        xs, zs = [np.zeros(n)], [bot - BALL_RADIUS_FT - s_ft]
    frames = []
    for px, pz in zip(xs, zs):
        g = pd.DataFrame([dict(base, edge=edge, call_dir=call_dir,
                               called_strike=(call_dir < 0))] * n)
        g["dist_against_call"] = np.asarray(d_in, float)
        g["plate_x"], g["plate_z"] = px, pz
        g["batter_height_ft"] = height
        g["abs_zone_top"], g["abs_zone_bot"] = top, bot
        frames.append(g)
    return frames


def fig_curves(out, demo, tr, models, labels, gam_key, boost_key):
    edges = [e for e in ("side", "top", "bottom") if e in set(tr["edge"])]
    if gam_key not in models or boost_key not in models or not edges:
        return
    base = _typical_row(tr)
    height = float(tr["batter_height_ft"].median())
    xs = np.linspace(-5, 5, 401)
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(1, len(edges), wspace=0.08, top=0.78)
    bins = np.arange(-5.5, 6.0, 1.0)
    table, gaps = [], []
    ax0 = None
    for j, e in enumerate(edges):
        ax = fig.add_subplot(gs[0, j], sharey=ax0)
        ax0 = ax0 or ax
        sub = tr[tr["edge"] == e]
        for call, cd in (("strike", -1.0), ("ball", 1.0)):
            color = CALL_COLORS[call]
            g = pd.DataFrame([dict(base, edge=e, call_dir=cd, called_strike=(cd < 0))] * len(xs))
            g["dist_against_call"] = xs
            pg, lo, hi = models[gam_key].predict_band(g)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, lw=0, zorder=1)
            ax.plot(xs, pg, color=color, lw=2.4, zorder=3)
            pb = np.mean([models[boost_key].predict(fr) for fr in _edge_rows(base, e, cd, xs, height)],
                         axis=0)
            ax.plot(xs, pb, color=color, lw=1.6, ls=(0, (4, 2)), zorder=4)
            m = sub["call_dir"].to_numpy() == cd
            b = W7._binned(sub["dist_against_call"].to_numpy()[m], sub["y"].to_numpy()[m], bins)
            ax.scatter(b["center"], b["rate"], s=16 + 3 * np.sqrt(b["n"]), facecolor="white",
                       edgecolor=color, linewidths=1.3, zorder=5)
            x_g, x_b = _x50(xs, pg), _x50(xs, pb)
            if np.isfinite(x_g) and np.isfinite(x_b):
                gaps.append(abs(x_g - x_b))
            table += [dict(edge=e, call=call, inches=x_, gam=a, gam_lo=l_, gam_hi=h_, boosted=q)
                      for x_, a, l_, h_, q in zip(xs[::10], pg[::10], lo[::10], hi[::10], pb[::10])]
        ax.axvline(0, color=S.AXIS, lw=1.0, zorder=0)
        ax.axhline(0.5, color=S.AXIS, lw=1.0, zorder=0)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.02, 1.02)
        W7._pct_axis(ax)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_title(f"{e.capitalize()} edge", loc="left", fontsize=13.5)
        ax.text(1.0, 1.02, f"{len(sub):,} training challenges", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, color=S.MUTED)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("chance of overturn")
        if j == len(edges) // 2:
            ax.set_xlabel("inches past the edge, against the call (+ = wrong side for the call)",
                          labelpad=8)
    handles = [Line2D([], [], color=S.ORANGE, lw=2.4, label=CALL_NAMES["strike"]),
               Line2D([], [], color=S.BLUE, lw=2.4, label=CALL_NAMES["ball"]),
               Line2D([], [], color=S.INK_2, lw=2.4, label=f"{labels[gam_key]} (band = 95%)"),
               Line2D([], [], color=S.INK_2, lw=1.6, ls=(0, (4, 2)), label=labels[boost_key]),
               Line2D([], [], color=S.INK_2, marker="o", mfc="white", lw=0,
                      label="observed, 1-inch bins")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.915), ncol=3,
               fontsize=10, frameon=False)
    gap = max(gaps) if gaps else np.nan
    if not np.isfinite(gap):
        title = "Where boosting and the GAM put each edge"
    elif gap <= 0.5:
        title = "Boosting and the GAM put the edges in nearly the same place"
    else:
        title = f"Boosting and the GAM disagree about some edges — by up to {gap:.1f} in."
    gap_txt = f" Largest gap between their 50% points: {gap:.2f} in." if np.isfinite(gap) else ""
    S.suptitles(fig, title,
                f"Overturn chance as a pitch moves past the middle of each edge, for a "
                f"{height * 12:.0f}-inch batter. Fitted on March–July.{gap_txt}", top=1.0)
    S.footnote(fig, "The boosted curve is asked only about pitches that can exist: its horizontal "
                    "and vertical location are set to match each distance (side edges average both "
                    "sides of the plate). It moves in steps because trees cut distance into "
                    "intervals. Where the two curves agree, the flexible model found nothing the "
                    "GAM's smooth curve had missed.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w8_06_curves", pd.DataFrame(table))


# ------------------------------------------------------------------ all
def make_all(out, data_dir, *, tr, va, models, labels, family, fsets, preds, point, ci, diffs,
             log, baseline, imps, boosted, floor=None, vs=None, show=None, peers=None):
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    peers = peers or {"location": ("M2", "GAM-loc"), "full": ("M5", "GAM-full")}
    vs = vs if vs is not None else pd.DataFrame(columns=["model", "label", "beats_baseline",
                                                         "calibration_ok"])
    if show is None:
        show = []
        for s in ("location", "full"):
            show += list(peers[s]) + [k for k in boosted if fsets[k] == s]
        if baseline not in show:
            show = [baseline] + show
    fig_tuning(out, demo, log, point, labels, baseline, peers)
    fig_comparison(out, demo, show, point, ci, diffs, labels, family, fsets, baseline, vs)
    grid = [(s, list(peers[s]) + [k for k in boosted if fsets[k] == s]) for s in ("location", "full")]
    fig_reliability(out, demo, va, preds, point, floor, labels, family, grid)
    fig_importance(out, demo, imps, labels)
    fig_shap_distance(out, demo, va, imps, models, labels)
    loc_boost = imps.get("location", {}).get("key")
    if loc_boost is None:
        cands = [k for k in boosted if fsets[k] == "location"]
        loc_boost = min(cands, key=lambda k: point[k]["brier"]) if cands else None
    if loc_boost is not None:
        fig_curves(out, demo, tr, models, labels, peers["location"][1], loc_boost)
