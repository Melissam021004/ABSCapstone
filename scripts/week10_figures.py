"""
week10_figures.py — the Week 10 figures. Called by 19_week10_winprob.py and
20_week10_challenge_value.py; not meant to be run on its own.

Same house style as Weeks 7–9: white background, 16:9, 200 DPI, each PNG with a
companion *_data.csv of the exact plotted numbers (the table view).

Colour jobs in this set:
    magnitude (runs, win probability gained)   one blue ramp, light -> dark
    polarity (home lead: behind / tied / ahead) red <- gray -> blue
    model                                       blue = structural, orange = boosted
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
import week7_figures as W7                  # noqa: E402
from src.abs_gamestate import BASE_NAMES, COUNT_NAMES   # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
SEQ_CMAP = LinearSegmentedColormap.from_list("abs_seq", ["#f4f8fd"] + S.SEQ)
MODEL_COLOR = {"markov": S.BLUE, "gbm": S.ORANGE}
MODEL_LABEL = {"markov": "Structural (Markov)", "gbm": "Boosted"}
HALF_LABELS = [f"{i}{h}" for i in range(1, 10) for h in ("▲", "▼")]


def _pct(ax, axes=("y",)):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    if "y" in axes:
        ax.yaxis.set_major_formatter(fmt)
    if "x" in axes:
        ax.xaxis.set_major_formatter(fmt)


def _lead_color(d):
    """Polarity: home behind = red, tied = gray, ahead = blue; stronger = further from 0."""
    if d == 0:
        return S.MUTED
    base = S.POS if d > 0 else S.NEG
    return base


def _heat(ax, M, row_labels, col_labels, fmt, vmax=None, cbar_label=None):
    vmax = vmax if vmax is not None else float(np.nanmax(M)) or 1.0
    cmap = SEQ_CMAP.copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            dark = v / vmax > 0.55
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9,
                    color="white" if dark else S.INK)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    return im


# ------------------------------------------------------------------ fig 1
def fig_runs(out, demo, rm):
    re = rm.re_table().to_numpy()
    sp = rm.score_prob_table().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=S.FULL)
    fig.subplots_adjust(top=0.80)
    _heat(axes[0], re, BASE_NAMES, ["0 out", "1 out", "2 out"], lambda v: f"{v:.2f}")
    axes[0].set_title("Expected runs to the end of the inning", loc="left", fontsize=13.5)
    _heat(axes[1], sp, BASE_NAMES, ["0 out", "1 out", "2 out"], lambda v: f"{100 * v:.0f}%",
          vmax=1.0)
    axes[1].set_title("Chance of scoring at least once more", loc="left", fontsize=13.5)
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel("runners on base")
    S.suptitles(fig, "Run expectancy: what each base-out state is worth",
                "The structural model's engine, at the start of a plate appearance (count 0-0). "
                "Darker = more runs.", top=1.0)
    S.footnote(fig, "Plate-appearance transitions from innings 1–8, chained to the third out. "
                    "The count version (RE288) is in docs/week10_run_expectancy.csv.")
    S.demo_stamp(fig, demo)
    rows = [{"runners": BASE_NAMES[b], "outs": o, "exp_runs": re[b, o], "p_score": sp[b, o]}
            for b in range(8) for o in range(3)]
    S.save(fig, out, "w10_01_runs", pd.DataFrame(rows))


# ------------------------------------------------------------------ fig 2
def fig_wp_curves(out, demo, wp, emp):
    leads = [-3, -2, -1, 0, 1, 2, 3]
    fig, ax = S.new_fig()
    xs = np.arange(18)
    rows = []
    for d in leads:
        vals = []
        for i in range(9):
            for h in (0, 1):
                if i == 8 and h == 1 and d > 0:
                    vals.append(np.nan)             # never played: the home team already leads
                    continue
                v = wp.home({"inning": i + 1, "half": h, "outs": 0, "bases": 0, "score_diff": d})
                vals.append(v)
                rows.append({"inning": i + 1, "half": "top" if h == 0 else "bottom",
                             "home_lead": d, "home_wp": v})
        a = 0.45 + 0.55 * abs(d) / 3 if d else 1.0
        ax.plot(xs, vals, color=_lead_color(d), lw=2.0, alpha=a, zorder=3)
        last = int(np.flatnonzero(np.isfinite(vals))[-1])
        S.halo(ax.text(xs[last] + 0.25, vals[last], f"{d:+d}" if d else "tied", va="center",
                       fontsize=10, color=S.INK_2))
        # the empirical rate where the table is well populated
        e = emp[(emp["outs"] == 0) & (emp["bases"] == 0) & (emp["diff_c"] == d)
                & (emp["n_games"] >= 100) & (emp["inning_c"] <= 9)]
        if len(e):
            ex = (e["inning_c"] - 1) * 2 + e["half"]
            ax.scatter(ex, e["rate"], s=26, facecolor="white", edgecolor=_lead_color(d),
                       linewidths=1.2, zorder=4, alpha=a)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i + 1}\n{'top' if h == 0 else 'bot'}" for i in range(9) for h in (0, 1)],
                       fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.4, 18.2)
    _pct(ax, ("y",))
    S.only_grid(ax, "y")
    S.titles(ax, "Home win probability through the game, by score",
             "At the start of each half-inning (0 out, bases empty). Lines: V(s). Hollow points: "
             "the empirical rate where 100+ games were there.",
             xlabel="inning", ylabel="home team wins")
    S.footnote(fig, "Red = home team behind, blue = ahead, gray = tied; darker = bigger lead. "
                    "The lines fan out as innings run out: the same lead is worth more later. The "
                    "zigzag is the last at-bat: the home team's chances rise each time it comes "
                    "up. No bottom of the 9th when the home team leads.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_02_wp_curves", pd.DataFrame(rows))


# ------------------------------------------------------------------ fig 3
def fig_calibration(out, demo, y, preds, point):
    keys = [k for k in ("markov", "gbm") if k in preds]
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(2, len(keys), height_ratios=[4.2, 1], hspace=0.06, wspace=0.16, top=0.82)
    table = []
    for j, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        tab = W7._rel_panel(ax, axh, y, preds[k], MODEL_COLOR[k], small=True)
        table += [dict(model=k, **r) for r in tab.to_dict("records")]
        s = point[k]
        ax.set_title(MODEL_LABEL[k], loc="left", fontsize=13.5, pad=18)
        ax.text(0, 1.015, f"Brier {s['brier']:.4f} · slope {s['cal_slope']:.2f} · intercept "
                          f"{s['cal_intercept']:+.2f}", transform=ax.transAxes, fontsize=9,
                color=S.INK_2, va="bottom")
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("home team actually won")
        axh.set_xlabel("predicted home win probability", fontsize=10)
    S.suptitles(fig, "Is V(s) calibrated? Every game scored by a model that never saw it",
                "Start of every plate appearance, 5 folds of whole games. Points on the dashed "
                "line = right on average.", top=1.0)
    S.footnote(fig, "10 equal-width bins; lines are 95% intervals; hollow points have fewer than "
                    "20 plate appearances. The structural model never sees who won; the boosted "
                    "one learns from it.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_03_calibration", pd.DataFrame(table))


# ------------------------------------------------------------------ fig 4
def fig_agreement(out, demo, tab, keys):
    t = tab[tab["n_games"] >= 100]
    fig, axes = plt.subplots(1, len(keys), figsize=S.FULL, squeeze=False)
    fig.subplots_adjust(top=0.80)
    rows = []
    for j, k in enumerate(keys):
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=S.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        size = 10 + 140 * np.sqrt(t["n_games"] / max(t["n_games"].max(), 1))
        ax.scatter(t[k], t["rate"], s=size, color=MODEL_COLOR[k], alpha=0.55, linewidths=0.6,
                   edgecolor="white", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _pct(ax, ("x", "y"))
        ax.set_title(MODEL_LABEL[k], loc="left", fontsize=13.5)
        ax.set_xlabel("model's average prediction in the cell")
        if j == 0:
            ax.set_ylabel("empirical home win rate")
        else:
            ax.tick_params(labelleft=False)
        rows += [{"model": k, "inning": r.inning_c, "half": r.half, "outs": r.outs,
                  "bases": BASE_NAMES[int(r.bases)], "home_lead": r.diff_c, "games": r.n_games,
                  "empirical": r.rate, "model_pred": getattr(r, k)} for r in t.itertuples()]
    S.suptitles(fig, "Model vs the empirical table, cell by cell",
                "Cells (inning × half × outs × runners × lead) with 100+ games. On the line = "
                "agreement; dot size = games.", top=1.0)
    S.footnote(fig, "Scatter around the line shrinks with more games: a cell with 100 games has a "
                    "standard error of about 5 points on its own. The plan's test is that the "
                    "versions agree where data is plentiful.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_04_agreement", pd.DataFrame(rows))


# ------------------------------------------------------------------ fig 5
def fig_call_value(out, demo, wp):
    leads = [4, 3, 2, 1, 0, -1, -2, -3, -4]
    cols = [(i, h) for i in range(1, 11) for h in (0, 1)]
    M = np.zeros((len(leads), len(cols)))
    rows = []
    for a, d in enumerate(leads):
        for b, (i, h) in enumerate(cols):
            if d > 0 and ((h == 1 and i >= 9) or (h == 0 and i >= 10)):
                M[a, b] = np.nan     # can't happen: the game would already be over / extras start tied
                continue
            st = {"inning": i, "half": h, "outs": 2, "bases": 0, "score_diff": d,
                  "balls": 1, "strikes": 2}
            v = float(wp.call_value(st, "strike")["dwp_challenger"].iloc[0])
            M[a, b] = 100 * v
            rows.append({"home_lead": d, "inning": i if i < 10 else "10+",
                         "half": "top" if h == 0 else "bottom", "dwp_batting_pts": 100 * v})
    fig, ax = S.new_fig()
    _heat(ax, M, [f"{d:+d}" if d else "tied" for d in leads],
          [f"{i if i < 10 else '10+'}{'▲' if h == 0 else '▼'}" for i, h in cols],
          lambda v: f"{v:.1f}" if v >= 0.05 else "")
    ax.set_xlabel("inning (▲ top: away team bats · ▼ bottom: home team bats)")
    ax.set_ylabel("home lead")
    S.titles(ax, "What overturning a called strike three is worth",
             "Win-probability points gained by the batting team: 2 out, bases empty, 1-2 count. "
             "Darker = worth more.")
    S.footnote(fig, "If the call stands the inning is over; overturned, the count goes to 2-2. "
                    "The same call is worth almost nothing in a blowout and several points late "
                    "in a close game: that gap is why success rate is the wrong scorecard. Blank: "
                    "states that can't happen (the home team ahead in the bottom of the 9th or "
                    "later, or in the top of an extra inning, which starts tied).")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_05_call_value", pd.DataFrame(rows))


# ------------------------------------------------------------------ fig 6
def fig_count_value(out, demo, wp):
    fig, axes = plt.subplots(1, 2, figsize=S.FULL)
    fig.subplots_adjust(top=0.78)
    rows = []
    for j, (bases, title) in enumerate(((0, "Bases empty"), (7, "Bases loaded"))):
        M = np.zeros((3, 12))
        for o in range(3):
            for c in range(12):
                st = {"inning": 5, "half": "top", "outs": o, "bases": bases, "score_diff": 0,
                      "balls": c // 3, "strikes": c % 3}
                v = float(wp.call_value(st, "strike")["druns_batting"].iloc[0])
                M[o, c] = v
                rows.append({"runners": title, "outs": o, "count": COUNT_NAMES[c],
                             "runs_ball_minus_strike": v})
        _heat(axes[j], M, ["0 out", "1 out", "2 out"], COUNT_NAMES, lambda v: f"{v:.2f}",
              vmax=float(max(0.6, np.nanmax(M))) if j else 0.6)
        axes[j].set_title(title, loc="left", fontsize=13.5)
        axes[j].set_xlabel("count before the pitch")
        if j:
            axes[j].tick_params(labelleft=False)
    S.suptitles(fig, "The inning-level value of a challenge, by count",
                "Runs between a ball and a strike on the same pitch, from there to the end of the "
                "inning: what the batting team gains when a strike is overturned — and what the "
                "fielding team saves when a ball is.", top=1.0)
    S.footnote(fig, "The biggest cells are strike three (the at-bat goes on instead of an out) "
                    "and ball four (a walk becomes a full count). Same pitch, same inch: the count "
                    "and the runners decide what the call is worth. The two panels have their "
                    "own colour scales.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_06_count_value", pd.DataFrame(rows))


def make_all(out, data_dir, *, k, ps, y, preds):
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    fig_runs(out, demo, k["rm"])
    fig_wp_curves(out, demo, k["wp"], k["emp"])
    fig_calibration(out, demo, y, preds, k["point"])
    keys = [m for m in ("markov", "gbm") if m in k["agree"]]
    if k.get("agree_table") is not None and keys:
        fig_agreement(out, demo, k["agree_table"], keys)
    fig_call_value(out, demo, k["wp"])
    fig_count_value(out, demo, k["wp"])


# =================================================================== script 20
ROLE_COLOR = {"batter": S.BLUE, "catcher": S.ORANGE, "pitcher": S.AQUA}
TIER_COLOR = {0: S.SEQ[2], 1: S.SEQ[4], 2: S.SEQ[6]}
TIER_LABEL = {0: "most accurate third", 1: "middle third", 2: "least accurate third"}


def _p(p):
    return "p < 0.01" if p < 0.01 else f"p = {p:.2f}"


def _half_ticks(ax, n=18):
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{i // 2 + 1}\n{'top' if i % 2 == 0 else 'bot'}" for i in range(n)],
                       fontsize=8.5)


def fig_opportunities(out, demo, opp, ch):
    fig, ax = S.new_fig()
    o = opp[(opp["p_s"] >= 0.02) & ~opp["challenged"]]
    if len(o) > 20000:
        o = o.sample(20000, random_state=1)
    ax.scatter(o["p_s"], 100 * o["dwp"], s=6, color=S.DEEMPH, alpha=0.35, lw=0, zorder=1,
               label="not challenged")
    up = ch[~ch["overturned"]]
    ov = ch[ch["overturned"]]
    ax.scatter(up["p_s"], 100 * up["dwp"], s=14, facecolor="white", edgecolor=S.ORANGE, lw=0.9,
               zorder=3, label="challenged — call stood")
    ax.scatter(ov["p_s"], 100 * ov["dwp"], s=12, color=S.BLUE, alpha=0.7, lw=0, zorder=2,
               label="challenged — overturned")
    ymax = float(np.nanpercentile(100 * opp["dwp"], 99.8)) if len(opp) else 5
    ax.set_ylim(0, max(ymax, 1))
    ax.set_xlim(0, 1)
    _pct(ax, ("x",))
    S.only_grid(ax, "y")
    ax.legend(loc="upper left", frameon=False, markerscale=2.2)
    S.titles(ax, "Every challenge opportunity: how likely the call was wrong vs what it was worth",
             "x: p(s), the chance ABS reverses the call · y: win-probability points the "
             "challenging side gains if it does",
             xlabel="chance the call is overturned, p(s)",
             ylabel="win probability gained if overturned (points)")
    S.footnote(fig, "Opportunities with p(s) below 2% (clearly correct calls) not shown; up to "
                    "20,000 unchallenged ones sampled. Week 11 draws the line through this cloud: "
                    "challenge above it, hold below — and the line moves with the inning and the "
                    "challenges left.")
    S.demo_stamp(fig, demo)
    tab = pd.concat([ch.assign(group=np.where(ch["overturned"], "overturned", "stood")),
                     o.assign(group="not challenged")])[["group", "p_s", "dwp"]]
    S.save(fig, out, "w10_07_opportunities", tab)


def fig_value_ahead(out, demo, a_all, a_tier):
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"wspace": 0.18})
    fig.subplots_adjust(top=0.80)
    a = a_all[a_all["half_index"] < 18]
    ax = axes[0]
    series = [("stakes", "all wrong calls still to come", S.BLUE, "-"),
              ("best2", "best two still to come", S.ORANGE, "-"),
              ("best1", "best one still to come", S.AQUA, "-"),
              ("dwp_now", "a wrong call right now", S.INK_2, (0, (4, 3)))]
    for col, lab, color, ls in series:
        ax.plot(a["half_index"], 100 * a[col], color=color, lw=2, ls=ls, label=lab)
    _half_ticks(ax)
    S.only_grid(ax, "y")
    ax.set_ylabel("win-probability points (average game)")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_title("What is still at stake, by inning", loc="left", fontsize=13)
    ax = axes[1]
    for t_, d in a_tier[a_tier["half_index"] < 18].groupby("tier"):
        ax.plot(d["half_index"], 100 * d["best1"], color=TIER_COLOR[int(t_)], lw=2,
                label=TIER_LABEL[int(t_)])
    _half_ticks(ax)
    S.only_grid(ax, "y")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, title="umpire", title_fontsize=9.5)
    ax.set_title("Best single wrong call still to come, by umpire", loc="left", fontsize=13)
    S.suptitles(fig, "Why a challenge in hand is worth something",
                "Averaged over every game and team, at the start of each half-inning: the calls "
                "that will still go against that team.", top=1.0)
    S.footnote(fig, "\"Best one still to come\" is what one challenge saved for later could win "
                    "with perfect foresight — the ceiling on the value of holding it. Week 11's "
                    "backward induction computes the real value below this ceiling.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_08_value_ahead", pd.concat([a_all.assign(view="all"),
                                                      a_tier.assign(view="by umpire tier")]))


def fig_inning_experiment(out, demo, nat_runs, nat_wins):
    fig, axes = plt.subplots(1, 2, figsize=S.FULL, gridspec_kw={"wspace": 0.25})
    fig.subplots_adjust(top=0.80)
    rows = []
    for ax, nat, unit, title in ((axes[0], nat_runs, 1.0, "Runs to the end of the inning"),
                                 (axes[1], nat_wins, 100.0, "Challenging team won (points)")):
        d = nat[nat["side"].isin(["batting", "fielding"])].reset_index(drop=True)
        ys = np.arange(len(d))[::-1]
        for y_, (_, r) in zip(ys, d.iterrows()):
            ax.plot([unit * r["actual_diff_lo"], unit * r["actual_diff_hi"]], [y_, y_],
                    color=S.BLUE, lw=2.2, solid_capstyle="round")
            ax.scatter(unit * r["actual_diff"], y_, s=70, color=S.BLUE, zorder=3,
                       label="what happened (95% interval)" if y_ == ys[0] else None)
            ax.scatter(unit * r["pred_diff"], y_, s=90, marker="D", facecolor="white",
                       edgecolor=S.ORANGE, lw=1.6, zorder=4,
                       label="what V(s) predicted" if y_ == ys[0] else None)
            rows.append({"outcome": title, **r.to_dict()})
        ax.axvline(0, color=S.MUTED, lw=1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['side']} side\n(n = {int(r['n']):,})" for _, r in d.iterrows()])
        ax.set_ylim(-0.7, len(d) - 0.3)
        S.only_grid(ax, "x")
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel("after overturned − after upheld")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9.5)
    S.suptitles(fig, "Did overturned challenges change what happened next?",
                "Overturned vs upheld challenges — decided by where the ball was, not the "
                "situation, so the gap estimates what an overturn does.", top=1.0)
    S.footnote(fig, "Runs: innings 1–8 (walk-offs end innings early). Batting-side challenges "
                    "should raise runs (positive); fielding-side challenges should lower them "
                    "(negative). Agreement = the diamond inside the interval.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_09_inning_experiment", pd.DataFrame(rows))


def fig_team_wins(out, demo, teams):
    t = teams.sort_values("wins_added").reset_index(drop=True)
    fig, ax = S.new_fig(figsize=(12, max(6.75, 0.24 * len(t) + 2)))
    ys = np.arange(len(t))
    ax.hlines(ys, 0, t["wins_added"], color=S.GRID, lw=1, zorder=1)
    ax.scatter(t["wins_added"], ys, s=46, color=S.BLUE, zorder=3, label="wins added by overturns")
    ax.scatter(t["left_on_table"], ys, s=46, facecolor="white", edgecolor=S.ORANGE, lw=1.4,
               zorder=3, label="left on the table")
    ax.scatter(t["umpire_net"], ys, s=30, marker="s", color=S.AQUA, zorder=2,
               label="umpire net")
    ax.axvline(0, color=S.MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(t["team"], fontsize=9)
    ax.set_ylim(-0.8, len(t) + 2.2)              # headroom for the legend
    S.only_grid(ax, "x")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9.5)
    S.titles(ax, "What challenges were worth to each team over the season",
             "In wins (win probability added). Sorted by wins added.", xlabel="wins")
    S.footnote(fig, "Left on the table = expected value of wrong calls against the team that it "
                    "could have challenged (it had one left) and didn't — a ceiling: a team doesn't "
                    "know for sure which calls are wrong (Week 11). Umpire net = wrong calls that "
                    "stood in its favour minus those that stood against it.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_10_team_wins", teams)


def fig_challengers(out, demo, chal, t_out, t_sel):
    c = chal[chal["n"] >= 5]
    fig, ax = S.new_fig()
    ax.plot([0, 1], [0, 1], color=S.MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for role in ("batter", "catcher", "pitcher"):
        d = c[c["role"].str.lower() == role]
        if not len(d):
            continue
        ax.scatter(d["mean_p"], d["rate"], s=12 + 6 * np.sqrt(d["n"]) * 3, alpha=0.6,
                   color=ROLE_COLOR[role], edgecolor="white", lw=0.5, label=f"{role}s", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _pct(ax, ("x", "y"))
    ax.legend(loc="upper left", frameon=False)
    S.titles(ax, "A challenger's success rate is the pitches he chooses",
             f"Each dot a player with 5+ challenges (size = challenges). Who-challenged effect "
             f"given the pitch: {_p(t_out[1])}; differences in which pitches: {_p(t_sel[1])}.",
             xlabel="average p(s) of the pitches he challenged (his judgment)",
             ylabel="share of his challenges overturned")
    S.footnote(fig, "On the dashed line = overturned exactly as often as the pitches he picked "
                    "predict. ABS sees only the ball, so players sit on the line: their accuracy "
                    "is their choice of pitches — a decision skill (RQ3), not an input to p(s).")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_11_challengers", c)


def fig_umpires(out, demo, ump_ok, t_ump):
    u = ump_ok.sort_values("excess_per_100").reset_index(drop=True)
    fig, ax = S.new_fig()
    ys = np.arange(len(u))
    for t_, d in u.groupby("tier"):
        ax.scatter(d["excess_per_100"], d.index, s=20 + 2 * np.sqrt(d["games"]) * 4,
                   color=TIER_COLOR[int(t_)], label=TIER_LABEL[int(t_)], zorder=3)
    ax.axvline(0, color=S.MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(u["umpire"], fontsize=7 if len(u) > 40 else 9)
    S.only_grid(ax, "x")
    ax.legend(loc="lower right", frameon=False)
    S.titles(ax, "Umpires: calls ABS would reverse, beyond what the pitches' difficulty predicts",
             f"Per 100 called pitches, against an average umpire seeing the same pitches. "
             f"Differences beyond chance: {_p(t_ump[1])}.",
             xlabel="extra wrong calls per 100 called pitches")
    S.footnote(fig, "Umpires with 10+ games; dot size = games. More wrong calls means more "
                    "chances for a challenge later in the game — which is how umpire accuracy "
                    "enters the decision (Week 11), not through p(s).")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "w10_12_umpires", u)


def make_value_figures(out, data_dir, *, k):
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    fig_opportunities(out, demo, k["opp"], k["ch"])
    fig_value_ahead(out, demo, k["a_all"], k["a_tier"])
    fig_inning_experiment(out, demo, k["nat_runs"], k["nat_wins"])
    fig_team_wins(out, demo, k["teams"])
    fig_challengers(out, demo, k["chal"], k["t_out"], k["t_sel"])
    if len(k["ump_ok"]):
        fig_umpires(out, demo, k["ump_ok"], k["t_ump"])
