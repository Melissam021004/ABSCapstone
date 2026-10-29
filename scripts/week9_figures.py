"""
week9_figures.py — the Week 9 figures. Called by 17_week9_calibrate_lock.py;
not meant to be run on its own.

Same house style as Weeks 7–8: white background, 16:9, 200 DPI, each PNG with
a companion *_data.csv of the exact plotted numbers.

Colour jobs in this set:
    calibration method   gray = uncalibrated, blue = Platt, orange = isotonic
    window               gray hollow = training months (out-of-fold),
                         blue = August (calibration), orange = September (test)
    probability surface  one blue ramp, light (0%) -> dark (100%)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eda_style as S                       # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402
from matplotlib.ticker import MaxNLocator   # noqa: E402
import week7_figures as W7                  # noqa: E402
from src.abs_calibration import METHODS     # noqa: E402
from src.abs_zone import BALL_RADIUS_FT, PLATE_HALF_WIDTH_FT, zone_from_height  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
METHOD_COLOR = {"none": S.MUTED, "platt": S.BLUE, "isotonic": S.ORANGE}
METHOD_LABEL = {"none": "uncalibrated", "platt": "Platt scaling", "isotonic": "isotonic"}
WINDOW_COLOR = {"August": S.BLUE, "September": S.ORANGE}
SET_LABEL = {"location": "location only", "full": "everything"}
SEQ_CMAP = LinearSegmentedColormap.from_list("abs_seq", ["#f4f8fd"] + S.SEQ)


# ------------------------------------------------------------------ fig 1
def fig_calibration(out, demo, va, xfit, point, labels, cands, chosen, prod_fs):
    k = cands[prod_fs]
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(2, 3, height_ratios=[4.2, 1], hspace=0.06, wspace=0.14, top=0.80)
    table = []
    for j, m_ in enumerate(METHODS):
        key = f"{prod_fs}|{m_}"
        ax = fig.add_subplot(gs[0, j])
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        tab = W7._rel_panel(ax, axh, va["y"], xfit[key], METHOD_COLOR[m_], small=True)
        table += [dict(method=m_, **r) for r in tab.to_dict("records")]
        s = point[key]
        star = "  ★ chosen" if m_ == chosen[prod_fs] else ""
        ax.set_title(METHOD_LABEL[m_].capitalize() + star, loc="left", fontsize=13.5, pad=18)
        slope = f"{s['cal_slope']:.2f}" if np.isfinite(s["cal_slope"]) else "—"
        ax.text(0, 1.015, f"Brier {s['brier']:.4f} · log loss {s['log_loss']:.3f} · "
                          f"slope {slope} · ECE {s['ece']:.3f}",
                transform=ax.transAxes, fontsize=9, color=S.INK_2, va="bottom")
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("actually overturned")
        axh.set_xlabel("predicted chance of overturn", fontsize=10)
    S.suptitles(fig, f"Calibrating {labels[k]}: before and after",
                f"August, cross-fitted: each game calibrated by a map fitted on the other games. "
                f"Points on the dashed line = calibrated.", top=1.0)
    S.footnote(fig, "10 equal-width bins; lines are 95% intervals; hollow points have fewer than "
                    "20 challenges. Platt can only shift and stretch the curve; isotonic can bend "
                    "it into any rising staircase — more flexible, and more able to learn "
                    "August's noise.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w9_01_calibration", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 2
def fig_methods(out, demo, point, ci, diffs, labels, cands, chosen):
    rows, heads, y = [], [], 0
    for fs, k in cands.items():
        heads.append((-y, f"{labels[k]}  ({SET_LABEL[fs]})"))
        y += 1
        for m_ in METHODS:
            rows.append((fs, m_, -y))
            y += 1
        y += 0.35
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, sharey=True,
                             gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1.05]})
    fig.subplots_adjust(top=0.80, left=0.22, right=0.86)
    table = []
    for fs, m_, yy in rows:
        key = f"{fs}|{m_}"
        c = METHOD_COLOR[m_]
        s, (lo, hi) = point[key]["brier"], ci[key]["brier"]
        axes[0].plot([lo, hi], [yy, yy], color=c, lw=3.0, solid_capstyle="round", zorder=2)
        axes[0].scatter([s], [yy], s=80, color=c, zorder=3, edgecolor="white", linewidth=1.2)
        rec = {"feature_set": fs, "method": m_, "brier": s, "lo": lo, "hi": hi,
               "chosen": m_ == chosen[fs]}
        if m_ == "none":
            S.halo(axes[1].text(0, yy, "reference", ha="center", va="center", fontsize=9.5,
                                color=S.INK_2))
        else:
            d, dlo, dhi = diffs[(key, f"{fs}|none")]["brier"]
            axes[1].plot([dlo, dhi], [yy, yy], color=c, lw=3.0, solid_capstyle="round", zorder=2)
            axes[1].scatter([d], [yy], s=80, color=c, zorder=3, edgecolor="white", linewidth=1.2)
            txt = ("reliably better" if dhi < 0 else "reliably worse" if dlo > 0
                   else "can't tell apart")
            axes[1].text(1.02, yy, txt, transform=axes[1].get_yaxis_transform(), fontsize=9.5,
                         color=S.INK if txt != "can't tell apart" else S.MUTED, va="center")
            rec.update(d=d, d_lo=dlo, d_hi=dhi, verdict=txt)
        table.append(rec)
    axes[0].set_yticks([r[2] for r in rows])
    axes[0].set_yticklabels([METHOD_LABEL[m_] + ("  ★" if m_ == chosen[fs] else "")
                             for fs, m_, _ in rows], fontsize=10.5, color=S.INK_2)
    for yy, name in heads:
        axes[0].text(-0.015, yy, name, transform=axes[0].get_yaxis_transform(), ha="right",
                     va="center", fontsize=10.5, fontweight="semibold", color=S.INK)
    axes[0].set_ylim(-(y - 0.35) - 0.6, 0.6)
    for ax in axes:
        ax.tick_params(axis="y", length=0, pad=8)
        S.only_grid(ax, "x")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    axes[0].set_title("Cross-fitted August Brier [95%]", loc="left", fontsize=13.5)
    axes[0].set_xlabel("lower = better", labelpad=8, fontsize=10.5, color=S.MUTED)
    span = max([abs(v) for r in table for v in (r.get("d_lo", 0), r.get("d_hi", 0))
                if np.isfinite(v)] + [1e-4])
    axes[1].set_xlim(-1.15 * span, 1.15 * span)
    axes[1].axvline(0, color=S.INK_2, lw=1.0, zorder=1)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:+.4f}" if abs(v) > 1e-12 else "0"))
    axes[1].set_title("minus uncalibrated, same games [95%]", loc="left", fontsize=13.5)
    axes[1].set_xlabel("← calibration helps        calibration hurts →", labelpad=8,
                       fontsize=10.5, color=S.MUTED)
    names = {fs: METHOD_LABEL[chosen[fs]] for fs in cands}
    title = ("Calibration earns its place" if any(v != "uncalibrated" for v in names.values())
             else "Neither calibrator beats the model's own probabilities")
    S.suptitles(fig, title,
                "★ = chosen by the rule fixed in advance: the simplest of uncalibrated → Platt → "
                "isotonic whose Brier can't be told apart from the best.", top=1.0)
    S.footnote(fig, "Every score is cross-fitted (5 folds of whole August games), so a calibrator "
                    "is never graded on the games it was fitted to. Intervals from resampling "
                    "games; the right panel pairs each method with 'uncalibrated' on every "
                    "resample.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w9_02_methods", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 3
def fig_subgroups(out, demo, sub, label):
    sub = sub.copy()
    windows = [w for w in ("August", "September") if (sub["window"] == w).any()]
    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    fig.subplots_adjust(top=0.84, left=0.30, right=0.97)
    ys, ylabels, heads, y = [], [], [], 0
    table = []
    off = {"August": 0.0, "September": 0.0}
    if len(windows) == 2:
        off = {"August": -0.17, "September": 0.17}
    base = sub[sub["window"] == windows[0]]
    for g, part in base.groupby("group", sort=False):
        heads.append((y, g))
        y += 1
        for r in part.itertuples():
            ys.append(-y)
            ylabels.append(r.level)
            for w in windows:
                rr = sub[(sub["window"] == w) & (sub["group"] == g) & (sub["level"] == r.level)]
                if not len(rr):
                    continue
                rr = rr.iloc[0]
                gap, lo, hi = rr.obs - rr.pred, rr.lo - rr.pred, rr.hi - rr.pred
                c = WINDOW_COLOR[w]
                yy = -y + off[w]
                ax.plot([100 * lo, 100 * hi], [yy, yy], color=c, lw=2.2, solid_capstyle="round",
                        alpha=1.0 if rr.n >= 30 else 0.45)
                ax.scatter([100 * gap], [yy], s=34, color=c if rr.flag else "white",
                           edgecolor=c, linewidth=1.4, zorder=3)
                table.append(dict(rr))
            ax.text(1.005, -y, f"n={int(r.n)}", transform=ax.get_yaxis_transform(), fontsize=8.5,
                    color=S.MUTED, va="center")
            y += 1
        y += 0.5
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels, fontsize=9, color=S.INK_2)
    for yy, g in heads:
        ax.text(-0.012, -yy, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, fontweight="semibold", color=S.INK)
    ax.axvline(0, color=S.INK_2, lw=1.0)
    ax.set_ylim(-y + 0.4, 0.8)
    lim = np.nanmax(np.abs(ax.get_xlim()))
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("observed − predicted overturn rate (percentage points)", labelpad=8)
    S.only_grid(ax, "x")
    ax.tick_params(axis="y", length=0, pad=6)
    handles = [Line2D([], [], color=WINDOW_COLOR[w], lw=2.2, marker="o", mfc="white", ms=6,
                      label=f"{w} ({'calibration, cross-fitted' if w == 'August' else 'test'})")
               for w in windows]
    handles.append(Line2D([], [], color=S.INK_2, lw=0, marker="o", ms=6,
                          label="filled = off by more than chance (n ≥ 30)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.905), ncol=3,
               fontsize=10, frameon=False)
    flagged = "; ".join(f"{w}: {int(sub[sub['window'] == w]['flag'].sum())} of "
                        f"{int((sub[sub['window'] == w]['n'] >= 30).sum())} testable slices "
                        f"flagged (≈ {0.05 * int((sub[sub['window'] == w]['n'] >= 30).sum()):.1f} "
                        "by chance)" for w in windows)
    S.suptitles(fig, "Is the locked p(s) right for every kind of challenge?",
                f"{label}. {flagged}. Zero = calibrated for that slice.", top=0.99)
    S.footnote(fig, "Gap = what actually happened minus what p(s) predicted, with the 95% "
                    "interval of the observed rate. Faded lines: fewer than 30 challenges (not "
                    "flagged). A slice that is off in BOTH windows is the one to worry about; one "
                    "that's off in only one is usually noise.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w9_03_subgroups", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 4
def fig_months(out, demo, months, label):
    mo = months.copy()
    mo["mkey"] = mo["level"].str[:2]
    mo = mo.sort_values(["mkey", "window"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(top=0.80)
    xs = {k: i for i, k in enumerate(sorted(mo["mkey"].unique()))}
    for r in mo.itertuples():
        i = xs[r.mkey]
        gap, lo, hi = r.obs - r.pred, r.lo - r.pred, r.hi - r.pred
        insample = str(r.window).startswith("train")
        c = S.MUTED if insample else WINDOW_COLOR.get(r.window, S.INK_2)
        ax.plot([i, i], [100 * lo, 100 * hi], color=c, lw=2.6, solid_capstyle="round")
        ax.scatter([i], [100 * gap], s=60, zorder=3, linewidths=1.5,
                   facecolor="white" if insample else c, edgecolor=c)
        ax.text(i, 100 * hi + 0.6, f"n={int(r.n)}", ha="center", va="bottom", fontsize=9,
                color=S.MUTED)
    ax.axhline(0, color=S.INK_2, lw=1.0)
    ax.set_xticks(list(xs.values()))
    lab = mo.drop_duplicates("mkey").set_index("mkey")["level"].str[3:]
    ax.set_xticklabels([lab[k] for k in xs])
    ax.set_ylabel("observed − predicted (percentage points)")
    S.only_grid(ax, "y")
    handles = [Line2D([], [], color=S.MUTED, lw=2.6, marker="o", mfc="white", ms=7,
                      label="training months — out-of-fold (5 refits)"),
               Line2D([], [], color=S.BLUE, lw=2.6, marker="o", ms=7,
                      label="August — calibration window, cross-fitted")]
    if (mo["window"] == "September").any():
        handles.append(Line2D([], [], color=S.ORANGE, lw=2.6, marker="o", ms=7,
                              label="September — the test window"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.915), ncol=3,
               fontsize=10, frameon=False)
    S.suptitles(fig, f"Does the locked p(s) hold up across the season?",
                "Gap between the overturn rate and p(s)'s average prediction, by month, with "
                "95% intervals. A steady drift would mean ABS — or who challenges — changed.",
                top=1.0)
    S.footnote(fig, "Out-of-fold: the model was refitted five times with the same settings, each "
                    "time leaving out a fifth of the training games, and every game was "
                    "predicted by the fit that never saw it — so the training months are graded "
                    "honestly too, not in-sample.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w9_04_months", mo.drop(columns="mkey"))


# ------------------------------------------------------------------ fig 5
def surface_grid(pm, height_ft, nx=81, nz=81):
    """p(s) over the plate for a batter of `height_ft`, for both calls."""
    top, bot = zone_from_height(height_ft)
    xs = np.linspace(-1.3, 1.3, nx)
    zs = np.linspace(max(0.3, bot - 0.7), top + 0.7, nz)
    gx, gz = np.meshgrid(xs, zs)
    out = {}
    for name, cs in (("strike", True), ("ball", False)):
        st = pd.DataFrame({"plate_x": gx.ravel(), "plate_z": gz.ravel(),
                           "batter_height_ft": height_ft, "called_strike": cs})
        out[name] = pm.predict_frame(st, strict=False).reshape(gz.shape)
    return xs, zs, out, (top, bot)


def fig_surface(out, demo, pm, tr):
    h = float(np.nanmedian(tr["batter_height_ft"]))
    xs, zs, P, (top, bot) = surface_grid(pm, h)
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(top=0.80, right=0.88)
    table = []
    for ax, name, title in ((axes[0], "strike", "Called STRIKE — the batter challenges"),
                            (axes[1], "ball", "Called BALL — the catcher or pitcher challenges")):
        im = ax.imshow(P[name], origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]],
                       cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="equal", interpolation="bilinear")
        cs = ax.contour(xs, zs, P[name], levels=[0.25, 0.5, 0.75], colors=[S.INK],
                        linewidths=[0.8, 1.6, 0.8], linestyles=["--", "-", "--"])
        ax.clabel(cs, fmt=lambda v: f"{v:.0%}", fontsize=9, inline=True)
        hw, r = PLATE_HALF_WIDTH_FT, BALL_RADIUS_FT
        ax.add_patch(plt.Rectangle((-hw, bot), 2 * hw, top - bot, fill=False, ec=S.ORANGE,
                                   lw=2.0, zorder=4))
        ax.add_patch(plt.Rectangle((-hw - r, bot - r), 2 * (hw + r), top - bot + 2 * r,
                                   fill=False, ec=S.ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=4))
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel("horizontal location, catcher's view (ft)", labelpad=8)
        ax.grid(False)
        for i in range(0, len(zs), 8):
            for j in range(0, len(xs), 8):
                table.append({"call": name, "plate_x": xs[j], "plate_z": zs[i],
                              "p": P[name][i, j]})
    axes[0].set_ylabel("height above the ground (ft)")
    cax = fig.add_axes([0.90, 0.18, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, .25, .5, .75, 1])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_visible(False)
    cax.set_title("p(overturn)", fontsize=10, color=S.INK_2, loc="left", pad=8)
    m = pm.manifest
    S.suptitles(fig, "The locked p(s): where a challenge wins",
                f"p(s) {m['version']} ({m['model_label']} + "
                f"{METHOD_LABEL.get(m['calibration']['method'], m['calibration']['method'])}), "
                f"for a {h * 12:.0f}-inch batter. Orange box = ABS zone; dashed = plus the "
                "ball's radius.", top=1.0)
    S.footnote(fig, "Solid contour = 50%: the coin-flip line. Left: a called strike is "
                    "overturned when the ball missed the zone, so p(s) rises OUTSIDE the box. "
                    "Right: a called ball is overturned when the ball touched it, so p(s) rises "
                    "INSIDE. How wide the band between 25% and 75% is shows how sharp the edges "
                    "are in practice.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w9_05_surface", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 6 (test)
def fig_test(out, demo, te, preds, point, ci, diffs, labels, sel):
    keys = ["PROD", "PROD-raw"] + [k for k in sel["models"] if k != "M0"]
    fam = {k: sel["models"][k]["family"] for k in sel["models"]}
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[4.2, 1], hspace=0.06,
                          wspace=0.55, top=0.80, left=0.25)
    ax = fig.add_subplot(gs[:, 0])
    ys = np.arange(len(keys))[::-1]
    table = []
    for yy, k in zip(ys, keys):
        if k.startswith("PROD"):
            c = S.INK if k == "PROD" else S.MUTED
        else:
            c = {"logistic": S.BLUE, "gam": S.ORANGE}.get(fam[k], S.AQUA)
        s, (lo, hi) = point[k]["brier"], ci[k]["brier"]
        ax.plot([lo, hi], [yy, yy], color=c, lw=3.0 if k == "PROD" else 2.4,
                solid_capstyle="round", zorder=2)
        ax.scatter([s], [yy], s=90 if k == "PROD" else 60, color=c, zorder=3, edgecolor="white",
                   linewidth=1.2)
        table.append({"model": k, "label": labels[k], "brier": s, "lo": lo, "hi": hi})
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[k] + (" ★" if k == sel["baseline_to_beat"] else "") for k in keys],
                       fontsize=10, color=S.INK_2)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    ax.get_yticklabels()[0].set_color(S.INK)
    S.pad_cat_axis(ax, len(keys), pad=0.7)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    S.only_grid(ax, "x")
    ax.set_title("September Brier [95%]", loc="left", fontsize=13.5)
    ax.set_xlabel("lower = better", labelpad=8, fontsize=10.5, color=S.MUTED)
    axr = fig.add_subplot(gs[0, 1])
    axh = fig.add_subplot(gs[1, 1], sharex=axr)
    tab = W7._rel_panel(axr, axh, te["y"], preds["PROD"], S.BLUE, small=True)
    s = point["PROD"]
    axr.set_title("Locked p(s) on September", loc="left", fontsize=13.5, pad=18)
    axr.text(0, 1.015, f"slope {s['cal_slope']:.2f} · intercept {s['cal_intercept']:+.2f} · "
                       f"ECE {s['ece']:.3f}", transform=axr.transAxes, fontsize=9,
             color=S.INK_2, va="bottom")
    axr.set_ylabel("actually overturned")
    axh.set_xlabel("p(s)", fontsize=10)
    base = sel["baseline_to_beat"]
    same = np.allclose(preds["PROD"], preds.get(base, np.full(len(te), np.nan)))
    if same:
        tail = f"The locked p(s) is the Week 8 baseline itself ({labels[base]}), uncalibrated."
    elif base in preds:
        d = diffs[("PROD", base)]["brier"]
        tail = (f"Locked p(s) minus the Week 8 baseline: Δ Brier {d[0]:+.4f} "
                f"[{d[1]:+.4f}, {d[2]:+.4f}].")
    else:
        tail = ""
    S.suptitles(fig, "The one look at September",
                f"{len(te):,} test challenges, never used for any choice. {tail}", top=1.0)
    S.footnote(fig, "Every Week 8 model is scored here too (uncalibrated), so the four-model "
                    "comparison also gets its single test-set look. Black = the locked p(s); "
                    "gray = the same model before calibration.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w9_06_test", pd.concat([pd.DataFrame(table),
                                              tab.assign(model="PROD")], ignore_index=True))


# ------------------------------------------------------------------ entry points
def make_lock_figures(out, data_dir, *, va, xfit, point, ci, diffs, labels, cands, chosen,
                      prod_fs, sub, months, pm, tr):
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    label = f"p(s) {pm.manifest['version']}"
    fig_calibration(out, demo, va, xfit, point, labels, cands, chosen, prod_fs)
    fig_methods(out, demo, point, ci, diffs, labels, cands, chosen)
    fig_subgroups(out, demo, sub, label)
    fig_months(out, demo, months, label)
    fig_surface(out, demo, pm, tr)


def make_test_figures(out, data_dir, *, te, preds, point, ci, diffs, labels, sel, sub_te,
                      sub_aug, months):
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    fig_test(out, demo, te, preds, point, ci, diffs, labels, sel)
    both = pd.concat([sub_aug, sub_te], ignore_index=True) if len(sub_aug) else sub_te
    label = labels["PROD"].replace(" (locked)", "")
    fig_subgroups(out, demo, both, label)
    fig_months(out, demo, months, label)
