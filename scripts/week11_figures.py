"""
week11_figures.py — the Week 11 figures. Called by 21_week11_mdp.py; not meant to
be run on its own.

Same house style as Weeks 7–10: white background, 16:9, 200 DPI, each PNG with a
companion *_data.csv of the exact plotted numbers (the table view).

Colour jobs in this set:
    challenges left / series      blue = 1 left (or the MDP), orange = 2 left (or a
                                  second series), aqua = a third
    reference rules               gray, dashed (Statcast's breakeven) / dotted (× LI)
    leverage bands                one blue ramp, light -> dark
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eda_style as S                       # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
R_COLOR = {1: S.BLUE, 2: S.ORANGE}
R_LABEL = {1: "1 challenge left", 2: "2 challenges left"}


def _pct(ax, axes=("y",)):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    if "y" in axes:
        ax.yaxis.set_major_formatter(fmt)
    if "x" in axes:
        ax.xaxis.set_major_formatter(fmt)


def _end_labels(ax, items, x, min_gap=0.05, fontsize=10):
    """Labels at the right end of lines, nudged apart so none overlap.
    items: [(y, text, color)]; min_gap in data units of the y axis."""
    items = sorted([it for it in items if np.isfinite(it[0])], key=lambda t: t[0])
    ys = [it[0] for it in items]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + min_gap)
    lo, hi = ax.get_ylim()
    over = ys[-1] - (hi - min_gap / 2) if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
        for k in range(len(ys) - 2, -1, -1):
            ys[k] = min(ys[k], ys[k + 1] - min_gap)
    for (y0, text, color), y in zip(items, ys):
        t = ax.text(x, y, text, color=color, fontsize=fontsize, va="center", ha="left")
        S.halo(t)


def _half_ticks(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)


# ------------------------------------------------------------------ fig 1: the money figure
def fig_thresholds(out, demo, curve, opp):
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(top=0.84, right=0.80)
    sav = None
    labels = []
    xmax = float(curve["li_mid"].max()) if len(curve) else 8.0
    for r in (2, 1):
        d = curve[curve["r_left"] == r].sort_values("li_mid")
        if not len(d):
            continue
        x = d["li_mid"].to_numpy()
        ax.fill_between(x, d["p_star_q25"], d["p_star_q75"], color=R_COLOR[r], alpha=0.14, lw=0)
        ax.plot(x, d["p_star_median"], color=R_COLOR[r], lw=2.4, marker="o", ms=5, zorder=4)
        labels.append((float(d["p_star_median"].iloc[-1]), R_LABEL[r], S.INK))
        if sav is None or len(d) > len(sav):
            sav = d
    if sav is not None:
        ax.plot(sav["li_mid"], sav["savant_median"], color=S.MUTED, lw=1.8, ls=(0, (5, 3)), zorder=3)
        ax.plot(sav["li_mid"], sav["savant_li_median"], color=S.MUTED, lw=1.8, ls=(0, (1, 2.2)), zorder=3)
        labels.append((float(sav["savant_median"].iloc[-1]), "Statcast breakeven (0.2-run cost)", S.MUTED))
        labels.append((float(sav["savant_li_median"].iloc[-1]), "Statcast breakeven × LI", S.MUTED))
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "8"])
    ax.axvline(1, color=S.AXIS, lw=1, zorder=1)
    ax.set_ylim(0, 1)
    _end_labels(ax, labels, xmax * 1.1, min_gap=0.055)
    _pct(ax)
    S.only_grid(ax, "y")
    ax.set_xlabel("Leverage Index of the plate appearance (Tango; 1 = average)")
    ax.set_ylabel("confidence needed to challenge, p*")
    falls = len(curve) and all(
        (lambda d: len(d) < 2 or d["p_star_median"].iloc[-1] < d["p_star_median"].iloc[0])(
            curve[curve["r_left"] == r].sort_values("li_mid")) for r in (1, 2))
    S.titles(ax, "The higher the leverage, the lower the bar to challenge" if falls
             else "How the bar to challenge moves with leverage",
             "Median p* = H / (Δ + H) over every 2026 challenge opportunity, by leverage; band = middle half")
    S.footnote(fig, "p* is the overturn probability at which challenging and holding are worth the "
                    "same win probability. Gray lines: Statcast's published rule, p ≥ 0.2 / (0.2 + run "
                    "value of the flip), and the same rule with the run value multiplied by LI.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w11_01_thresholds", curve)


# ------------------------------------------------------------------ fig 2: value of holding
def fig_hold(out, demo, hold):
    h = hold.groupby(["inning", "half", "half_label"], as_index=False)[["H1_win", "H2_win", "H1_runs",
                                                                     "H2_runs"]].mean()
    h = h.sort_values(["inning", "half"]).reset_index(drop=True)
    labels = [("X" if i >= 10 else str(i)) + ("▲" if s == 0 else "▼") for i, s in zip(h["inning"], h["half"])]
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"wspace": 0.22})
    fig.subplots_adjust(top=0.80)
    for ax, unit, scale, ylab in ((axes[0], "win", 100, "win-probability points"),
                                  (axes[1], "runs", 1, "runs")):
        for col, lab, color in ((f"H1_{unit}", "the last challenge (1 → 0)", S.BLUE),
                                (f"H2_{unit}", "the second challenge (2 → 1)", S.ORANGE)):
            ax.plot(range(len(h)), scale * h[col], color=color, lw=2.2, marker="o", ms=4, label=lab)
        _half_ticks(ax, labels)
        S.only_grid(ax, "y")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9.5)
    axes[0].set_title("Playing to win (the MDP)", loc="left", fontsize=13)
    axes[1].set_title("Playing for runs", loc="left", fontsize=13)
    S.suptitles(fig, "What a challenge in hand is worth, inning by inning",
                "V(s, r) − V(s, r − 1) at the start of each half-inning in a tied game, averaged over "
                "the two teams.", top=1.0)
    S.footnote(fig, "▲ top, ▼ bottom, X = any extra inning. A successful challenge is kept, so one "
                    "challenge can correct several calls — its value can exceed the single best wrong "
                    "call still to come (Week 10's ceiling). In extras a team with none left gets one back.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w11_02_hold_value", h.assign(label=labels))


# ------------------------------------------------------------------ fig 3: leverage
def fig_leverage(out, demo, li_inn, li_pa, bands):
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"wspace": 0.2, "width_ratios": [1.1, 1]})
    fig.subplots_adjust(top=0.80)
    ax = axes[0]
    x = np.arange(len(li_inn))
    ax.bar(x - 0.19, li_inn["li_2026"], width=0.36, color=S.BLUE, label="2026, our V(s)", zorder=3)
    ax.bar(x + 0.19, li_inn["li_2013_published"], width=0.36, color=S.SEQ[1],
           label="2013, Tango's LI (published)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([("10+" if i == 10 else str(int(i))) for i in li_inn["inning"]])
    ax.axhline(1, color=S.AXIS, lw=1)
    S.only_grid(ax, "y")
    ax.set_xlabel("inning")
    ax.set_ylabel("average LI of a plate appearance")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("Average leverage by inning", loc="left", fontsize=13)
    ax = axes[1]
    v = np.clip(np.asarray(li_pa, float), 0.01, 20)
    bins = np.logspace(np.log10(0.01), np.log10(20), 46)
    edges = [0.0, 0.85, 2.0, 21]
    shades = [S.SEQ[0], S.SEQ[2], S.SEQ[5]]
    share = dict(zip(bands["band"], bands["share_of_PAs"]))
    names = ["low (< 0.85)", "medium (0.85–2)", "high (> 2)"]
    for k in range(3):
        m = (v >= max(edges[k], 0.0)) & (v < edges[k + 1])
        ax.hist(v[m], bins=bins, color=shades[k], zorder=3,
                label=f"{names[k]}: {share.get(names[k], np.nan):.0%} of PAs")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 0.85, 2, 5, 10])
    ax.set_xticklabels(["≤0.01", "0.1", "0.85", "2", "5", "10"])
    S.only_grid(ax, "y")
    ax.set_xlabel("LI (log scale)")
    ax.set_ylabel("plate appearances")
    ax.set_title("How often each leverage comes up", loc="left", fontsize=13)
    S.suptitles(fig, "Tango's Leverage Index, rebuilt from our own win probability",
                "LI = expected win-probability swing of the plate appearance ÷ the average swing "
                "(Tango, 2006). Bands as on FanGraphs.", top=1.0)
    S.footnote(fig, "2013 averages: Studeman, The Hardball Times (Dec 9 2013), per play with Tango's LI. "
                    "Our extra innings start with a runner on second (the 2026 rule), so the 10th is "
                    "higher-leverage than in 2013.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w11_03_leverage", li_inn)


# ------------------------------------------------------------------ fig 4: run levels
def fig_run_levels(out, demo, ladder):
    measures = [("noout", "at-bat: chance the batter is safe", S.AQUA),
                ("runs", "inning: expected runs (RE288)", S.BLUE),
                ("score1", "inning: chance of scoring ≥ 1", S.ORANGE)]
    d = ladder[ladder["context"] != "all"].reset_index(drop=True)
    allrow = ladder[ladder["context"] == "all"]
    short = {"noout": "the batter's chance of reaching", "runs": "expected runs",
             "score1": "the chance of scoring once"}

    def best(ctx_mask):
        row = d[ctx_mask]
        if not len(row):
            return None
        vals = {k: float(row[f"r2_{k}"].iloc[0]) for k, _, _ in measures}
        vals = {k: v for k, v in vals.items() if np.isfinite(v)}
        return max(vals, key=vals.get) if vals else None

    b_early = best(d["context"].str.startswith("innings 1"))
    b_b9 = best(d["context"].str.startswith("bottom 9th"))
    if b_early and b_b9 and b_early != b_b9:
        title = (f"Early on, {short[b_early]} track a call's win value; in a tied bottom 9th, "
                 f"{short[b_b9]} does")
    else:
        title = "Which measure of run production tracks a call's win value"
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"wspace": 0.16, "width_ratios": [2.1, 1]})
    fig.subplots_adjust(top=0.80)
    ax = axes[0]
    x = np.arange(len(d))
    w = 0.26
    for k, (key, lab, color) in enumerate(measures):
        xs = x + (k - 1) * w
        vals = d[f"r2_{key}"].to_numpy()
        ax.bar(xs, vals, width=w * 0.92, color=color, label=lab, zorder=3)
        for xx, vv in zip(xs, vals):
            if np.isfinite(vv):
                t = ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=8.5, color=S.INK)
                S.halo(t)
    nice = {"innings 1–6": "innings 1–6", "7th or later, within 1 run": "7th+, within 1 run",
            "7th or later, 2+ runs apart": "7th+, 2+ runs apart", "bottom 9th or later, tied": "bottom 9th+, tied"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nice.get(c, c)}\n({int(n):,} pitches)" for c, n in zip(d["context"], d["n"])],
                       fontsize=9)
    ax.set_ylim(0, 1.12)
    S.only_grid(ax, "y")
    ax.set_ylabel("share of the win-probability swing explained (R²)")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=1)
    ax.set_title("In run terms, by game situation", loc="left", fontsize=13)
    ax = axes[1]
    if len(allrow):
        xs = np.arange(len(measures))
        raw = [float(allrow[f"r2_{k}"].iloc[0]) for k, _, _ in measures]
        lev = [float(allrow[f"r2_{k}_x_LI"].iloc[0]) for k, _, _ in measures]
        ax.bar(xs - 0.2, raw, width=0.38, color=S.SEQ[0], zorder=3, label="the run measure alone")
        ax.bar(xs + 0.2, lev, width=0.38, color=S.SEQ[5], zorder=3, label="× leverage (LI)")
        for xx, a, b in zip(xs, raw, lev):
            for xo, v in ((xx - 0.2, a), (xx + 0.2, b)):
                t = ax.text(xo, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
                S.halo(t)
        ax.set_xticks(xs)
        ax.set_xticklabels(["batter\nsafe", "expected\nruns", "score\n≥ 1"], fontsize=9.5)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_ylim(0, 1.12)
    S.only_grid(ax, "y")
    ax.tick_params(labelleft=False)
    ax.set_title("Leverage turns runs into wins", loc="left", fontsize=13)
    S.suptitles(fig, title, "R² between Δ — the win-probability swing of flipping the call, from V(s, r) — "
                            "and each run measure's flip, over 2026 challenge opportunities.", top=1.0)
    S.footnote(fig, "At-bat and inning EXPECTED runs are the same number (RE288 counts the rest of the "
                    "inning, the plate appearance included), so the at-bat level here is the batter's "
                    "chance of not making an out. Right panel: every opportunity, all situations together.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w11_04_run_levels", ladder)


# ------------------------------------------------------------------ fig 5: policies
def fig_policies(out, demo, pols, real_use):
    p = pols.reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"wspace": 1.05, "width_ratios": [1, 1]})
    fig.subplots_adjust(top=0.80, left=0.22, right=0.97)
    ax = axes[0]
    y = np.arange(len(p))[::-1]
    vals = p["wins_lost_per_season"].to_numpy()
    ax.set_xlim(0, max(np.nanmax(vals) * 1.25, 0.5))
    colors = [S.BLUE if i == 0 else S.DEEMPH for i in range(len(p))]
    for yy, vv, cc in zip(y, vals, colors):
        S.rounded_barh(ax, [yy], [max(vv, 0)], color=cc)
        t = ax.text(max(vv, 0) + ax.get_xlim()[1] * 0.015, yy, f"{vv:.2f}", va="center", fontsize=10)
        S.halo(t)
    ax.set_yticks(y)
    ax.set_yticklabels(p["policy"])
    S.pad_cat_axis(ax, len(p))
    S.only_grid(ax, "x")
    ax.set_xlabel("wins lost per 162 games")
    ax.set_title("Against an opponent who plays well", loc="left", fontsize=13)
    ax = axes[1]
    rows = [(r["policy"], r["challenges_per_team_game"], r["success_rate"], S.BLUE if k == 0 else S.DEEMPH)
            for k, (_, r) in enumerate(p.iterrows())]
    if real_use and np.isfinite(real_use.get("challenges_per_team_game", np.nan)):
        rows.append(("2026 as played", real_use["challenges_per_team_game"], real_use["success_rate"], S.ORANGE))
    y2 = np.arange(len(rows))[::-1]
    xmax = max(r[1] for r in rows) * 1.6 if rows else 1
    ax.set_xlim(0, max(xmax, 0.5))
    for yy, (name, n_ch, sr, color) in zip(y2, rows):
        S.rounded_barh(ax, [yy], [n_ch], color=color)
        lab = f"{n_ch:.2f}" + (f" · {sr:.0%} won" if np.isfinite(sr) else "")
        t = ax.text(n_ch + ax.get_xlim()[1] * 0.015, yy, lab, va="center", fontsize=9.5)
        S.halo(t)
    ax.set_yticks(y2)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    S.pad_cat_axis(ax, len(rows))
    S.only_grid(ax, "x")
    ax.set_xlabel("challenges per team per game")
    ax.set_title("How often each rule challenges", loc="left", fontsize=13)
    S.suptitles(fig, "The challenge rules, played out on the 2026 game",
                "Left: win probability a team gives up with each rule (×162). Right: simulated games with "
                "both teams on the rule, and the real season.", top=1.0)
    S.footnote(fig, "Every rule sees the same p(s) — Hawk-Eye's answer — so these are full-information "
                    "benchmarks; real players see the pitch for two seconds. 'Play for runs' uses the "
                    "thresholds of the run-differential MDP inside the game played to win.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w11_05_policies", p)


# ------------------------------------------------------------------ fig 6: real decisions
def fig_real(out, demo, dcurve):
    d = dcurve.copy()
    x = (d["p_lo"] + d["p_hi"]) / 2
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(top=0.84, right=0.82)
    series = [("model", "the thresholds (MDP)", S.BLUE, "-"),
              ("savant", "Statcast's breakeven", S.MUTED, (0, (5, 3))),
              ("actual", "2026 teams", S.ORANGE, "-")]
    labels = []
    for col, lab, color, ls in series:
        ax.plot(x, d[col], color=color, lw=2.4, ls=ls, marker="o", ms=4.5, zorder=4)
        labels.append((float(d[col].iloc[-1]), lab, S.INK))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    _end_labels(ax, labels, 1.015, min_gap=0.05)
    _pct(ax, ("x", "y"))
    S.only_grid(ax, "both")
    ax.set_xlabel("p(s): chance ABS reverses the call")
    ax.set_ylabel("share of opportunities challenged")
    hi = d[d["p_lo"] >= 0.5]
    gap = float((hi["model"] - hi["actual"]).mean()) if len(hi) else np.nan
    if np.isfinite(gap) and gap > 0.25:
        title = "Teams challenge far less often than the thresholds say they should"
    elif np.isfinite(gap) and gap < -0.1:
        title = "Teams challenge more often than the thresholds say they should"
    else:
        title = "How often teams challenged, against what the thresholds say"
    S.titles(ax, title,
             "Every 2026 challenge opportunity (a called pitch against a team with a challenge left), by p(s)")
    S.footnote(fig, "The gap at high p(s) is mostly what players cannot see in two seconds, not bad "
                    "strategy: the thresholds assume p(s) is known. The thresholds' curve is not a step at one "
                    "p because the bar moves with the inning, score, count and challenges left.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w11_06_real_decisions", d)


def make_all(out, data_dir, *, R, li_pa, opp):
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    fig_thresholds(out, demo, R["curve"], opp)
    fig_hold(out, demo, R["hold"])
    fig_leverage(out, demo, R["li_inn"], li_pa, R["bands"])
    fig_run_levels(out, demo, R["ladder"])
    fig_policies(out, demo, R["policies"], R["real_use"])
    fig_real(out, demo, R["dcurve"])
