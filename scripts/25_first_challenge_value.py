#!/usr/bin/env python3
"""
25_first_challenge_value.py — what "the first challenge is worth about 6 points" means.

Week 11 §3 says that at the first pitch, a team's first challenge is worth about 6
percentage points of win probability (5.83 on the 2026 data). That is easy to read as
the value of challenging the first pitch. It isn't: reversing a 0-0 call in the 1st
inning is worth well under one point. It is the value of HAVING one challenge for the
whole game, compared with having none. That covers every wrong call still to come that
the challenge could fix, and it is kept each time it succeeds. Both teams use their
challenges well, and the other team still holds both of its own.

Three slides for the checkpoint, plus a one-page explainer:
  12_first_challenge_value     a team's chance to win at the first pitch with no, one or
                               two challenges, and why the bar at that pitch is so high
  13_first_challenge_one_game  one real 2026 game: every call against one team, what one
                               challenge used well does with each, and what it adds up to
  14_first_challenge_all_games the same tally for every team in every 2026 game: its
                               average is the model's number, and it comes from all nine
                               innings, not the first pitch

    python scripts/25_first_challenge_value.py
    python scripts/25_first_challenge_value.py --game 776543 --team NYY   # pick the game

"One challenge, used well" is the Week 11 policy with one challenge in hand. At each
call against the team, it challenges when p, the chance the call is wrong, clears the
bar for that moment, and keeps the challenge whenever the challenge succeeds. The tally
follows each real game as it was played. It uses expected values: a challenge at p = 90%
adds 0.9 × the call's value, and the team still holds the challenge afterwards 90% of
the time. Averaged over every game, the tally reproduces the model's number. That check
is independent: the model solves the game backward, and the tally walks the real games
forward.

Needs Weeks 9–11 (models/production, models/winprob, models/mdp). Runs in about a
minute. Writes figures/showcase/12–14_*.png (each with its data CSV),
docs/FIRST_CHALLENGE_VALUE.md and docs/first_challenge_team_games.csv.
"""

import argparse
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "eda"):
    sys.path.insert(0, str(p))

import eda_style as S                                    # noqa: E402
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402
from src.abs_mdp import D, N_INN, load_mdp               # noqa: E402

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
DOCS = Path(os.environ.get("ABS_DOCS_DIR", "docs"))
FIGS = Path(os.environ.get("ABS_FIG_DIR", "figures")) / "showcase"
GLITCH_IN = 8.0                     # same rule as script 22: wrong by more than this = a glitch
LIKELY = 0.5                        # p at or above this: the call is probably wrong
S_ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th", 9: "9th"}


def _s22():
    spec = importlib.util.spec_from_file_location("s22", ROOT / "scripts" / "22_showcase_examples.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pts(x, d=1, sign=False):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:{'+' if sign else ''}.{d}f}"


def pct(x, d=0):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def chance(p):
    """A probability for a label: never '100%' or '0%' for something that isn't certain."""
    if not np.isfinite(p):
        return "—"
    return ">99%" if p >= 0.995 else "<1%" if p < 0.005 else f"{100 * p:.0f}%"


def panel_title(ax, title, sub=None, size=14):
    """A panel's title and subtitle, stacked above the axes, left-aligned with it."""
    n = sub.count("\n") + 1 if sub else 0
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    va="bottom", ha="left", fontsize=11, color=S.INK_2, linespacing=1.25)
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 8 + n * 15 + (6 if sub else 0)),
                textcoords="offset points", va="bottom", ha="left", fontsize=size,
                fontweight="semibold", color=S.INK)


# ======================================================================= the model
def first_pitch(sol, val):
    """A team's chance to win at the first pitch with 0, 1 or 2 challenges (tied game, the
    other team holding both), and what a challenge of the first pitch wins and risks."""
    pre = sol.start[(1, 0)][D]                        # home win probability; [r_home, r_away]
    home = [float(pre[r, 2]) for r in (0, 1, 2)]
    away = [float(1 - pre[2, r]) for r in (0, 1, 2)]
    avg = [(h + a) / 2 for h, a in zip(home, away)]
    c00 = val.loc[(val["balls"] == 0) & (val["strikes"] == 0), "count"]
    c00 = int(c00.mode().iat[0]) if len(c00) else 0

    def look(field, r_bat, r_fld=2):                  # top of the 1st, 0 out, empty, 0-0, tied
        return float(np.asarray(sol.lookup(field, 1, 0, 0, 0, c00, 0, r_bat, r_fld)).ravel()[0])

    stake = {r: {"gain": look("D_S", r), "risk": look("H_S", r), "bar": look("thr_S", r)} for r in (1, 2)}
    return {"home": home, "away": away, "avg": avg, "v_first": avg[1] - avg[0],
            "v_second": avg[2] - avg[1], "v_all": avg[2] - avg[0], "stake": stake,
            "first_strike": look("D_S", 2), "first_ball": look("D_B", 2, 2)}


def drain(sol):
    """The value of a team's one challenge at the start of every half-inning of a tied game,
    with the other team holding both of its own."""
    rows = []
    for i in range(1, N_INN + 1):
        for s in (0, 1):
            st = sol.start[(i, s)][D]
            home, away = float(st[1, 2] - st[0, 2]), float(st[2, 0] - st[2, 1])
            rows.append({"inning": i, "half": "top" if s == 0 else "bottom",
                         "home_team": home, "away_team": away, "average_team": (home + away) / 2})
    return pd.DataFrame(rows)


# ======================================================================= the real games
def replay(val, sol):
    """
    Every call against every team in every game, with what one challenge used well (and two)
    would do with it. Returns the calls (with the decision, the chance the team still holds its
    challenge, and the expected gain) and one row per team-game.
    """
    v = val[np.isfinite(val["p_s"])].copy()
    if "dist_against_call" in v.columns:
        v = v[~(pd.to_numeric(v["dist_against_call"], errors="coerce") > GLITCH_IN)]
    strike = v["call_strike"].to_numpy(bool)
    base = tuple(v[c].to_numpy() for c in ("inning", "half", "outs", "bases", "count", "diff"))
    rb_act, rf_act = v["r_bat"].to_numpy(), v["r_fld"].to_numpy()
    for r in (1, 2):                                  # this team r, the other team as it really was
        rb, rf = np.where(strike, r, rb_act), np.where(strike, rf_act, r)
        v[f"bar{r}"] = np.where(strike, sol.lookup("thr_S", *base, rb, rf), sol.lookup("thr_B", *base, rb, rf))
        v[f"delta{r}"] = np.where(strike, sol.lookup("D_S", *base, rb, rf), sol.lookup("D_B", *base, rb, rf))
    v = v.sort_values(["game_pk", "chall_team", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    p, inn = v["p_s"].to_numpy(float), v["inning"].to_numpy(int)
    c1, c2 = p >= v["bar1"].to_numpy(), p >= v["bar2"].to_numpy()
    d1, d2 = v["delta1"].to_numpy(float), v["delta2"].to_numpy(float)
    hold_before, gain1, gain2 = np.zeros(len(v)), np.zeros(len(v)), np.zeros(len(v))
    grp = (v["game_pk"].astype(str) + "|" + v["chall_team"].astype(str)).to_numpy()
    starts = np.r_[0, np.flatnonzero(grp[1:] != grp[:-1]) + 1, len(v)]
    for a, b in zip(starts[:-1], starts[1:]):
        P1, Q2, Q1, last = 1.0, 1.0, 0.0, None
        for k in range(a, b):
            if inn[k] > 9 and inn[k] != last:         # 2026 rule: at zero, a team gets one in each extra inning
                P1, Q1 = 1.0, 1.0 - Q2
            last = inn[k]
            hold_before[k] = P1
            if c1[k]:
                gain1[k] = P1 * p[k] * d1[k]
                P1 *= p[k]
            gain2[k] = Q2 * c2[k] * p[k] * d2[k] + Q1 * c1[k] * p[k] * d1[k]
            Q2, Q1 = (Q2 * (p[k] if c2[k] else 1.0),
                      Q1 * (p[k] if c1[k] else 1.0) + (Q2 * (1 - p[k]) if c2[k] else 0.0))
    v["challenge1"], v["holds_before"], v["gain1"], v["gain2"] = c1, hold_before, gain1, gain2
    v["exp_challenges"] = c1 * hold_before            # expected challenges made, and won
    v["exp_overturns"] = v["exp_challenges"] * p
    ov = v["overturned"].fillna(False).astype(bool) & v["challenged"].fillna(False).astype(bool)
    v["actual_gain"] = np.where(ov, v["swing"].to_numpy(float), 0.0)
    wrong = p >= LIKELY
    tg = (v.assign(wrong=wrong, held_wrong=wrong & ~c1, inn_c=np.where(c1, inn, np.nan))
          .groupby(["game_pk", "chall_team"], sort=False)
          .agg(one=("gain1", "sum"), two=("gain2", "sum"), calls=("p_s", "size"),
               expected_wrong=("p_s", "sum"), wrong=("wrong", "sum"),
               policy_challenges=("challenge1", "sum"), expected_challenges=("exp_challenges", "sum"),
               expected_overturns=("exp_overturns", "sum"), held_wrong=("held_wrong", "sum"),
               first_inn=("inn_c", "min"), last_inn=("inn_c", "max"),
               actual_challenges=("challenged", "sum"), actual_gain=("actual_gain", "sum"))
          .reset_index().rename(columns={"chall_team": "team"}))
    return v, tg


def season_stats(val, v, tg):
    n = len(tg)
    inn = np.minimum(v["inning"].to_numpy(int), N_INN)
    by_inning = pd.Series(100 * v["gain1"].to_numpy()).groupby(inn).sum() / n
    first = v.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby("game_pk").head(1)
    return {"n": n, "mean1": 100 * tg["one"].mean(), "mean2": 100 * tg["two"].mean(),
            "median1": 100 * tg["one"].median(), "share10": float((tg["one"] >= 0.10).mean()),
            "expected_wrong": tg["expected_wrong"].mean(), "expected_challenges": tg["expected_challenges"].mean(),
            "expected_overturns": tg["expected_overturns"].mean(), "by_inning": by_inning,
            "first_pitch": 100 * first["gain1"].sum() / n,
            "extras_share": float((val.groupby("game_pk")["inning"].max() >= 10).mean())}


def pick_game(tg, game, team):
    """A typical game to show: one challenge used well adds about the season's average, with two to
    five challenges spread over the game, a probably-wrong call it rightly keeps the challenge for,
    and a team that really did challenge. Or the game you name."""
    if game is not None:
        t = tg[tg["game_pk"] == game]
        if team:
            t = t[t["team"].astype(str).str.upper() == str(team).upper()]
        if t.empty:
            raise SystemExit(f"No calls for game {game}" + (f" and team {team}" if team else "") + ".")
        return t.sort_values("one", ascending=False).iloc[0]
    mean = tg["one"].mean()
    ok = ((tg["policy_challenges"] >= 2) & (tg["policy_challenges"] <= 5) & (tg["held_wrong"] >= 1)
          & (tg["actual_challenges"] >= 1) & ((tg["last_inn"] - tg["first_inn"]) >= 3))
    cand = tg[ok] if ok.any() else tg
    cand = cand.assign(gap=(cand["one"] - mean).abs()).sort_values(["gap", "game_pk"])
    return cand.iloc[0]


# ======================================================================= slide 12
def fig_value(out, demo, fp):
    fig = plt.figure(figsize=S.FULL)
    gl = fig.add_axes([0.012, 0.17, 0.43, 0.53])
    gr = fig.add_axes([0.545, 0.17, 0.44, 0.53])
    # left: the same team with no, one, two challenges
    a = [100 * x for x in fp["avg"]]
    ys, h = [2.0, 1.0, 0.0], 0.40
    names = ["no challenges", "one challenge", "two challenges"]
    for k, y in enumerate(ys):
        gl.barh(y, a[0], height=h, color=S.DEEMPH, lw=0, zorder=3)
        if k >= 1:
            gl.barh(y, a[1] - a[0], left=a[0], height=h, color=S.BLUE, edgecolor="white", lw=1, zorder=3)
        if k >= 2:
            gl.barh(y, a[2] - a[1], left=a[1], height=h, color=S.SEQ[2], edgecolor="white", lw=1, zorder=3)
        gl.text(0, y + h / 2 + 0.07, names[k], fontsize=12, va="bottom", color=S.INK)
        S.halo(gl.text(a[k] + 0.9, y, f"{a[k]:.1f}%", va="center", fontsize=13, fontweight="semibold"))
    S.halo(gl.text((a[0] + a[1]) / 2, ys[1] + h / 2 + 0.07, f"+{a[1] - a[0]:.1f} first challenge",
                   ha="center", va="bottom", fontsize=11.5, fontweight="bold"))
    S.halo(gl.text(a[2], ys[2] + h / 2 + 0.07, f"+{a[2] - a[1]:.1f} second", ha="center", va="bottom",
                   fontsize=11, color=S.INK_2))
    gl.set_xlim(0, 64)
    gl.set_ylim(-0.45, 2.75)
    gl.set_yticks([])
    gl.set_xticks(range(0, 61, 10))
    gl.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    S.only_grid(gl, "x")
    gl.spines["left"].set_visible(False)
    gl.set_xlabel("the team's chance to win, at the first pitch")
    panel_title(gl, "The same team, three starting positions",
                "Tied game, the other team holding both of its challenges,\n"
                "both teams using theirs well")
    # right: what a challenge of the first pitch wins and risks
    st1, st2 = fp["stake"][1], fp["stake"][2]
    groups = [(2.05, "with its last challenge", st1), (0.45, "with both of its challenges", st2)]
    hb = 0.36
    xmax = 100 * st1["risk"] * 2.05
    for y0, name, st in groups:
        g_, r_ = 100 * st["gain"], 100 * st["risk"]
        gr.barh(y0 + 0.21, g_, height=hb, color=S.ORANGE, lw=0, zorder=3)
        gr.barh(y0 - 0.21, r_, height=hb, color=S.BLUE, lw=0, zorder=3)
        gr.text(0, y0 + 0.21 + hb / 2 + 0.08, name, fontsize=12, va="bottom", color=S.INK)
        S.halo(gr.text(g_ + 0.12, y0 + 0.21, f"+{g_:.2f} if reversed", va="center", fontsize=11))
        S.halo(gr.text(r_ + 0.12, y0 - 0.21, f"−{r_:.2f} if it stands", va="center", fontsize=11))
        xr = xmax * 0.70
        gr.text(xr, y0 + 0.13, f"bar {pct(st['bar'])}", fontsize=16, fontweight="bold", va="bottom", color=S.INK)
        gr.text(xr, y0 + 0.05, f"{r_:.2f} ÷ ({g_:.2f} + {r_:.2f})", fontsize=10.5, va="top", color=S.INK_2)
    gr.set_xlim(0, xmax)
    gr.set_ylim(-0.2, 2.9)
    gr.set_yticks([])
    ticks = [t for t in range(0, int(100 * st1["risk"]) + 2, 2)]
    gr.set_xticks(ticks)
    S.only_grid(gr, "x")
    gr.spines["left"].set_visible(False)
    gr.set_xlabel("points of win probability")
    gr.legend(handles=[Line2D([], [], color=S.ORANGE, lw=8, label="if the call is reversed, the team gains "
                              "this (and keeps the challenge)"),
                       Line2D([], [], color=S.BLUE, lw=8, label="if it stands, the challenge is gone: the team "
                              "loses what holding it was worth")],
              loc="upper left", bbox_to_anchor=(0, -0.17), frameon=False, fontsize=10, handlelength=1.2)
    panel_title(gr, "Why the bar at the first pitch is so high",
                "The visiting team's leadoff hitter takes a 0-0 called strike. Challenge\n"
                "only if p, the chance the call is wrong, ≥ loss ÷ (gain + loss)")
    S.suptitles(fig, "What “the first challenge is worth about 6 points” means",
                f"At the first pitch, a team with one challenge wins {pts(fp['v_first'])} percentage points more "
                f"often than a team with none. The first pitch's own call is worth only "
                f"{pts(fp['first_strike'])} if reversed:\nthe {pts(fp['v_first'])} is every wrong call still to "
                f"come, because a challenge that succeeds is kept. So challenging the first pitch with the last "
                f"one risks {pts(st1['risk'])} points to win {pts(st1['gain'])}.", top=0.95)
    S.footnote(fig, "Week 11's model. Points = percentage points of win probability; 100 points = one win. Left: "
                    "the average of the home and visiting teams. Right: the visiting team after the call stands, "
                    "so the loss differs a little from the left. The model's bar equals loss ÷ (gain + loss) "
                    "exactly.")
    S.demo_stamp(fig, demo)
    tab = pd.DataFrame({"what": [f"chance to win with {n_}" for n_ in names]
                        + ["first-pitch strike reversed, last challenge: gain", "… loss if it stands", "… bar",
                           "first-pitch strike reversed, two challenges: gain", "… loss if it stands", "… bar"],
                        "value": [round(x, 3) for x in a]
                        + [round(100 * st1["gain"], 3), round(100 * st1["risk"], 3), round(st1["bar"], 4),
                           round(100 * st2["gain"], 3), round(100 * st2["risk"], 3), round(st2["bar"], 4)]})
    S.save(fig, out, "12_first_challenge_value", tab)


# ======================================================================= slide 13
def callouts(ax, fig, items, fontsize=9.5, gap_in=0.06):
    """Puts each label in the lowest row of a band above the marks where it overlaps no label
    already there, to the right of its mark or, when the next mark is close, to the left.
    items: (x, y, text). Returns [(x, y, text, row, ha)] and the number of rows."""
    x0, x1 = ax.get_xlim()
    renderer = fig.canvas.get_renderer()
    ax_px = ax.get_window_extent(renderer=renderer).width

    def width(text):                                  # measured, in data units
        t_ = ax.text(0, 0, text, fontsize=fontsize, linespacing=1.2)
        w_ = t_.get_window_extent(renderer=renderer).width
        t_.remove()
        return w_ / ax_px * (x1 - x0)

    pad = gap_in * fig.dpi / ax_px * (x1 - x0)
    items = sorted(items, key=lambda r: r[0])
    rows, placed = [], []
    for j, (x, y, text) in enumerate(items):
        w = width(text)
        crowded = j + 1 < len(items) and items[j + 1][0] - x < w + pad
        k = 0
        while True:
            order = (("right", x - w), ("left", x)) if crowded and k == 0 else (("left", x), ("right", x - w))
            sides = [s for s in order if x0 <= s[1] and s[1] + w <= x1] or [("right", x - w)]
            busy = rows[k] if k < len(rows) else []
            fit = [s for s in sides if all(s[1] + w + pad < a or s[1] - pad > b for a, b in busy)]
            if fit:
                break
            k += 1
        ha, left = fit[0]
        while k >= len(rows):
            rows.append([])
        rows[k].append((left, left + w))
        placed.append((x, y, text, k, ha))
    return placed, max(len(rows), 1)


def fig_one_game(out, demo, val, v, row, mean_one, describe):
    pk, team = row["game_pk"], row["team"]
    g = val[val["game_pk"] == pk].sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)
    g["x"] = np.arange(len(g))
    t = v[(v["game_pk"] == pk) & (v["chall_team"] == team)].merge(
        g[["at_bat_number", "pitch_number", "x"]], on=["at_bat_number", "pitch_number"], how="left")
    t = t.sort_values("x").reset_index(drop=True)
    first = g.iloc[0]
    fig = plt.figure(figsize=S.FULL)
    top = fig.add_axes([0.08, 0.40, 0.75, 0.39])
    bot = fig.add_axes([0.08, 0.10, 0.75, 0.19], sharex=top)
    starts = g[g["half"] == 0].groupby("inning")["x"].min()
    for ax in (top, bot):
        for x_ in starts.to_numpy()[1:]:
            ax.axvline(x_ - 0.5, color=S.GRID, lw=1, zorder=1)
    top.set_xlim(-1, len(g))
    p, bar = t["p_s"].to_numpy(), t["bar1"].to_numpy()
    y = 100 * t["delta1"].to_numpy()
    chal = t["challenge1"].to_numpy(bool)
    keep = ~chal & (p >= LIKELY)
    rest = ~chal & ~keep
    act = t["challenged"].fillna(False).astype(bool).to_numpy()
    top.scatter(t.loc[rest & ~act, "x"], np.zeros((rest & ~act).sum()), marker="|", s=40, color=S.DEEMPH, zorder=2)
    for m_, kw in ((rest & act, dict(color=S.DEEMPH)),
                   (keep, dict(facecolor="white", edgecolor=S.ORANGE, linewidth=2)),
                   (chal, dict(color=S.BLUE))):
        top.vlines(t.loc[m_, "x"], 0, y[m_], color=S.DEEMPH, lw=1.2, zorder=2)
        top.scatter(t.loc[m_, "x"], y[m_], s=70, zorder=4, **kw)
    for i in np.flatnonzero(act):
        top.scatter([t.at[i, "x"]], [y[i]], s=260, facecolor="none", edgecolor=S.INK, linewidth=1.3, zorder=5)
        top.annotate("✓" if bool(t.at[i, "overturned"]) else "✗", (t.at[i, "x"], y[i]), xytext=(9, -1),
                     textcoords="offset points", fontsize=12, va="center", color=S.INK, zorder=6)
    items = []
    for i in np.flatnonzero(chal | keep | (rest & act)):
        if chal[i]:
            lab = f"p {chance(p[i])}, bar {pct(bar[i])}\nchallenge: +{100 * t.at[i, 'gain1']:.2f}"
        else:
            lab = f"p {chance(p[i])}, bar {pct(bar[i])}\nkeep it"
        items.append((t.at[i, "x"], y[i], lab))
    placed, nrows = callouts(top, fig, items)
    ymax = max(0.5, float(np.nanmax(y[chal | keep | (rest & act)])) if (chal | keep | (rest & act)).any() else 0.5)
    h_in = top.get_position().height * fig.get_figheight()
    row_in, base_in = 0.40, 0.12
    marks_in = h_in - (nrows * row_in + base_in) - 0.06
    per = ymax / marks_in
    top.set_ylim(-0.06 / 0.94 * marks_in * per * 0.1, ymax + (nrows * row_in + base_in) * per)
    for x_, y_, lab, k, ha in placed:
        ly = ymax + (base_in + k * row_in) * per
        top.plot([x_, x_], [y_, ly - 0.02 * per], color=S.MUTED, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
        S.halo(top.text(x_, ly, lab, ha=ha, va="bottom", fontsize=9.5, color=S.INK, linespacing=1.2))
    top.set_yticks([v_ for v_ in top.get_yticks() if 0 <= v_ <= ymax * 1.05])
    top.set_ylabel(f"points for {team}\nif reversed")
    S.only_grid(top, "y")
    plt.setp(top.get_xticklabels(), visible=False)
    top.tick_params(axis="x", length=0)
    handles = [Line2D([], [], marker="o", ls="", color=S.BLUE, ms=8, label="challenge it: p clears the bar"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=S.ORANGE, mew=2, ms=8,
                      label="keep it: p is 50% or more, but below the bar"),
               Line2D([], [], marker="|", ls="", color=S.DEEMPH, ms=11, mew=2,
                      label=f"probably right: p below 50% ({int((rest & ~act).sum())} calls)")]
    if act.any():
        handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=S.INK, ms=13, mew=1.3,
                              label=f"ringed: {team}'s real challenges (✓ overturned, ✗ stood)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.075, 0.795), ncol=2, frameon=False,
               fontsize=10, handletextpad=0.3, columnspacing=2.0)
    # bottom: the running total
    xs = np.r_[-1, t["x"].to_numpy(), len(g)]
    run = 100 * np.r_[0, np.cumsum(t["gain1"].to_numpy()), t["gain1"].sum()]
    bot.step(xs, run, where="post", color=S.BLUE, lw=2.4, zorder=3)
    bot.axhline(100 * mean_one, color=S.MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    S.halo(bot.text(0, 100 * mean_one, f"the average team-game: {100 * mean_one:.1f}", va="bottom",
                    fontsize=9.5, color=S.MUTED))
    S.halo(bot.annotate(f"{run[-1]:.1f} points", (len(g), run[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=12, fontweight="semibold", color=S.INK, annotation_clip=False))
    bot.set_ylim(0, max(run[-1], 100 * mean_one) * 1.35 + 0.3)
    mids = [(a_ + b_) / 2 for a_, b_ in zip(starts.to_numpy(), list(starts.to_numpy()[1:]) + [len(g)])]
    bot.set_xticks(mids)
    bot.set_xticklabels([str(int(i)) for i in starts.index])
    bot.tick_params(axis="x", length=0)
    bot.set_xlabel("inning")
    bot.set_ylabel("points won")
    S.only_grid(bot, "y")
    panel_title(bot, "Running total. Each challenge adds: chance the challenge is still in hand × p × points "
                     "if reversed", size=11.5)
    home, away = first["home_team"], first["away_team"]
    date = pd.Timestamp(first["game_date"])
    S.suptitles(fig, f"One real game: what one challenge, used well, would have done for {team}",
                f"{away} at {home}, {date:%B %-d, %Y}. Every call that went against {team}, in order: the height "
                f"is what reversing it would be worth to {team}, and p is the chance the call is wrong.\nOne "
                f"challenge, used well, is spent only when p clears the bar for that moment, and is kept whenever "
                f"it succeeds, so it can fix several calls in one game.", top=0.955)
    S.footnote(fig, f"Week 11's thresholds (the bars) with one challenge in hand. Calls that were probably right sit "
                    f"on the baseline. Expected values: a challenge at p = 90% adds 0.9 × "
                    f"its points and is still in hand afterwards 90% of the time. The game is followed as it was "
                    f"played. " + (f"The rings show what {team} really did, with both of its challenges."
                                   if act.any() else f"{team} made no challenges in this game."))
    S.demo_stamp(fig, demo)
    keepcols = ["inning", "half", "outs", "bases", "balls", "strikes", "call_strike", "p_s", "bar1", "delta1",
                "challenge1", "holds_before", "gain1", "challenged", "overturned", "actual_gain"]
    S.save(fig, out, "13_first_challenge_one_game", t[[c for c in keepcols if c in t.columns]])
    detail = []
    for i, r_ in t[~(rest & ~act)].iterrows():
        d = describe(r_)
        detail.append({"when": d["situation"], "runners": d["runners"],
                       "call": "strike" if r_["call_strike"] else "ball", "p": r_["p_s"], "bar": r_["bar1"],
                       "delta": r_["delta1"], "decision": "challenge" if r_["challenge1"] else "keep it",
                       "holds_before": r_["holds_before"], "gain": r_["gain1"],
                       "real": (("challenged: overturned" if r_["overturned"] else "challenged: stood")
                                if r_["challenged"] else "not challenged")})
    return pd.DataFrame(detail), int((rest & ~act).sum()), len(t)


# ======================================================================= slide 14
def fig_all_games(out, demo, tg, fp, row, st):
    vals = 100 * tg["one"].to_numpy()
    cap = 30
    counts, edges = np.histogram(np.minimum(vals, cap - 1e-9), bins=np.arange(0, cap + 1, 1))
    fig = plt.figure(figsize=S.FULL)
    gl = fig.add_axes([0.075, 0.15, 0.54, 0.56])
    gr = fig.add_axes([0.70, 0.15, 0.285, 0.56])
    gl.bar(edges[:-1] + 0.5, counts, width=0.86, color=S.BLUE, lw=0, zorder=3)
    mean, med, ex = st["mean1"], st["median1"], 100 * float(row["one"])
    ymax = counts.max() * 1.25
    gl.set_ylim(0, ymax)
    gl.axvline(mean, color=S.INK, lw=1.4, ls=(0, (4, 3)), zorder=4)
    near = abs(ex - mean) < 1.5
    S.halo(gl.text(mean + 0.4, ymax * 0.97, f"average: {mean:.1f} points\nthe model: {pts(fp['v_first'])}"
                   + (f"\nthe game on the last slide: {ex:.1f}" if near else ""),
                   va="top", fontsize=11.5, fontweight="semibold", linespacing=1.3))
    if not near:                                      # the label sits in the empty space over the tail
        k = min(int(ex), cap - 1)
        S.halo(gl.annotate(f"the game on the last slide: {ex:.1f}", (k + 0.5, counts[k]),
                           xytext=(min(cap - 9, max(mean + 6, 12)), ymax * 0.55), textcoords="data",
                           fontsize=10.5, va="center",
                           arrowprops=dict(arrowstyle="->", color=S.INK, lw=1.1, shrinkB=3)))
    gl.set_xlim(0, cap)
    gl.set_xticks(np.arange(0, cap + 1, 5))
    gl.set_xticklabels([f"{k}" for k in range(0, cap, 5)] + [f"{cap}+"])
    S.only_grid(gl, "y")
    gl.set_xlabel("points one challenge, used well, would have won the team in that game")
    gl.set_ylabel("team-games")
    panel_title(gl, "Game by game",
                f"{st['n']:,} team-games. Half recovered less than {med:.1f} points,\nand "
                f"{pct(st['share10'])} recovered 10 or more.")
    by = st["by_inning"]
    xs = np.arange(len(by))
    gr.bar(xs, by.to_numpy(), width=0.68, color=S.BLUE, lw=0, zorder=3)
    gr.set_xticks(xs)
    gr.set_xticklabels([str(int(i)) if i <= 9 else "X" for i in by.index])
    gr.tick_params(axis="x", length=0)
    nine = by[by.index <= 9]
    gr.set_ylim(0, max(by.max(), 0.1) * 1.45)
    S.only_grid(gr, "y")
    S.halo(gr.text(xs[0] - 0.3, nine.max() * 1.10, f"{nine.min():.2f} to {nine.max():.2f} points from each "
                   f"inning", fontsize=10.5, va="bottom", fontweight="semibold"))
    gr.set_xlabel("inning (X = extra innings)")
    gr.set_ylabel("points per team-game")
    panel_title(gr, "Where the average comes from",
                f"Every inning adds some. The game's first\npitch adds {st['first_pitch']:.2f}.")
    S.suptitles(fig, "Every team, every 2026 game: the average is the model's number",
                f"The one-game tally, done for every team in every game, averages {mean:.2f} points; the model "
                f"says {pts(fp['v_first'], 2)}. A challenge that succeeds is kept, so one challenge, used well,\n"
                f"would fix {st['expected_overturns']:.1f} of the {st['expected_wrong']:.1f} wrong calls a team "
                f"faces in a game, spread over all nine innings. Two used well: {st['mean2']:.2f} "
                f"(the model: {pts(fp['v_all'], 2)}).", top=0.95)
    S.footnote(fig, "The tally assumes the team knows p, the chance each call is wrong, and so does the model; "
                    "spring adds what a player can actually see. Values above 30 are in the last bar. "
                    f"{pct(st['extras_share'])} of games went to extra innings.")
    S.demo_stamp(fig, demo)
    tab = pd.concat([pd.DataFrame({"panel": "game by game", "from_points": edges[:-1], "value": counts}),
                     pd.DataFrame({"panel": "points per team-game by inning (10 = extras)",
                                   "from_points": by.index, "value": by.round(4).to_numpy()})])
    S.save(fig, out, "14_first_challenge_all_games", tab)


# ======================================================================= the explainer
def write_doc(path, demo, fp, dr, st, row, detail, n_right, n_calls):
    team = row["team"]
    s1, s2 = fp["stake"][1], fp["stake"][2]
    L = ["# What “the first challenge is worth about 6 points” means", ""]
    if demo:
        L += ["> **DEMO DATA** — a simulated season; teams and games are not real.", ""]
    L += ["*Written by `scripts/25_first_challenge_value.py`. Slides: `figures/showcase/12_first_challenge_value`, "
          "`13_first_challenge_one_game`, `14_first_challenge_all_games`.*", "",
          "## The number in one sentence", "",
          f"At the first pitch of a tied game, with the other team holding both of its challenges, a team with "
          f"**one** challenge wins **{pts(fp['v_first'])} percentage points** more often than a team with "
          f"**none**, if it uses it well. A second challenge adds only {pts(fp['v_second'])} more, so the two are "
          f"worth {pts(fp['v_all'])} together: {fp['v_all'] * 162:.1f} wins over 162 games.", "",
          "| the same team, at the first pitch | chance to win |", "|---|---|"]
    for lab, x in zip(("no challenges", "one challenge", "two challenges"), fp["avg"]):
        L.append(f"| {lab} | {pct(x, 1)} |")
    L += ["", "With two challenges each the teams are even, at 50%. Take one away and the team drops "
          f"{pts(fp['v_second'])} points; take both away and it drops {pts(fp['v_all'])}.", "",
          "## What it is not", "",
          f"It is **not** the value of challenging the first pitch. Reversing a 0-0 called strike on the leadoff "
          f"batter is worth only {pts(fp['first_strike'], 2)} points to the batting team, and reversing a 0-0 "
          f"called ball is worth {pts(fp['first_ball'], 2)} to the fielding team. The {pts(fp['v_first'])} points "
          f"is the value of having the challenge **for the whole game**: all the wrong calls still to come that "
          f"it could fix, each at the moment it comes. Across 2026, the game's first pitch contributed "
          f"{st['first_pitch']:.2f} points of it on average.", "",
          "## Why one challenge is worth so much", "",
          f"- About {2 * st['expected_wrong']:.0f} calls a game are wrong, about {st['expected_wrong']:.1f} "
          "against each team.",
          "- A challenge that succeeds is **kept**. It is lost only when it fails.",
          f"- So one challenge, used well, can fix several calls. Across 2026 it would have been used "
          f"{st['expected_challenges']:.1f} times a game and fixed {st['expected_overturns']:.1f} of the "
          f"{st['expected_wrong']:.1f} wrong calls against a team.",
          f"- That is also why a second challenge adds so little ({pts(fp['v_second'])}): a team that "
          "challenges only when it's very likely right rarely loses its first one.", "",
          "## Why the bar at the first pitch is so high", "",
          "A challenge risks the challenge itself. If the call is reversed, the team gains the call's value and "
          "keeps the challenge. If the call stands, the challenge is gone, and with it everything holding it "
          "was worth. So a challenge pays only if", "",
          "> p × gain ≥ (1 − p) × loss, that is, p ≥ loss ÷ (gain + loss)", "",
          "where p is the chance the call is wrong. For the visiting team's leadoff hitter taking a 0-0 called "
          "strike:", "",
          "| challenges in hand | gain if reversed | loss if it stands | bar = loss ÷ (gain + loss) |",
          "|---|---|---|---|",
          f"| one (its last) | {pts(s1['gain'], 2)} | {pts(s1['risk'], 2)} | {pct(s1['bar'])} |",
          f"| two | {pts(s2['gain'], 2)} | {pts(s2['risk'], 2)} | {pct(s2['bar'])} |", "",
          "With one challenge left, the team risks the whole value of holding it to gain less than one point, so "
          "it needs to be nearly certain. With two, failing costs only the second challenge, which is worth "
          "little, so the bar is much lower. As the game goes on there are fewer calls left to use the challenge "
          "on, holding it is worth less, and the bar falls. The value of the one challenge at the start of each "
          "inning of a tied game:", "",
          "| start of the | top | bottom |", "|---|---|---|"]
    for i in range(1, 10):
        a_ = dr[(dr["inning"] == i) & (dr["half"] == "top")]["average_team"].iat[0]
        b_ = dr[(dr["inning"] == i) & (dr["half"] == "bottom")]["average_team"].iat[0]
        L.append(f"| {S_ORD[i]} | {pts(a_, 2)} | {pts(b_, 2)} |")
    L += ["", "Extra innings restart the clock: a team that has run out gets one challenge back in each extra "
          "inning.", "",
          f"## One real game, call by call ({team})", "",
          f"{n_calls} calls went against {team}. {n_right} were probably right (p below 50%) and below the bar; "
          "the rest, in order:", "",
          "| inning and score | runners, outs, count | the call | p | the bar | worth if reversed | one challenge, "
          "used well | chance it's still in hand | adds | what really happened |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, d in detail.iterrows():
        L.append(f"| {d['when']} | {d['runners']} | {d['call']} | {chance(d['p'])} | {pct(d['bar'])} | "
                 f"{pts(d['delta'], 2)} | {d['decision']} | {pct(d['holds_before'])} | {pts(d['gain'], 2)} | "
                 f"{d['real']} |")
    L += ["", f"**Total: {pts(row['one'], 2)} points** with one challenge used well. The average team-game: "
          f"{st['mean1']:.2f}. Each “adds” is the chance the challenge is still in hand × p × the "
          "points if reversed. Each time the challenge is used, the chance it's still in hand is multiplied by "
          f"p, the chance it succeeds. {team}'s real challenges, made with two in hand, won "
          f"{pts(row['actual_gain'], 2)} points.", "",
          "## Every team in every game", "",
          f"Doing the same tally for all {st['n']:,} team-games of 2026, one challenge used well recovers "
          f"**{st['mean1']:.2f} points on average**; the model says **{pts(fp['v_first'], 2)}**. Two used well "
          f"recover {st['mean2']:.2f}; the model says {pts(fp['v_all'], 2)}. These are two independent "
          "calculations: the model solves the game backward from the last out, and the tally walks the real "
          "games forward, call by call.", "",
          f"Half of team-games recover less than {st['median1']:.1f} points and {pct(st['share10'])} recover 10 "
          "or more. By inning:", "",
          "| inning | points per team-game |", "|---|---|"]
    for i, x in st["by_inning"].items():
        L.append(f"| {S_ORD.get(int(i), 'extras')} | {x:.2f} |")
    L += ["", "## Fine print", "",
          "- Both the model and the tally assume the team knows p, the chance each call is wrong. A player sees "
          "less than that, so the spring perception model will lower the number a player could really capture.",
          "- The tally follows each game as it was played. After an overturn the game would have gone "
          "differently, but on average that washes out: the averages match the model.",
          "- The model's number is for a tied game with the other team holding both of its challenges, which is "
          "where every game starts.", ""]
    path.write_text("\n".join(L) + "\n")


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--game", type=int, default=None, help="game_pk of the game to show")
    ap.add_argument("--team", default=None, help="the team in that game (abbreviation)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    demo = "demo" in str(DATA_DIR).lower()
    print("What the first challenge is worth" + (" (DEMO DATA)" if demo else ""))
    m22 = _s22()
    val, *_ = m22.build_table()
    sol = load_mdp()
    fp, dr = first_pitch(sol, val), drain(sol)
    s1, s2 = fp["stake"][1], fp["stake"][2]
    print(f"  first pitch, tied, the other team holding both: chance to win with no / one / two challenges "
          f"{pct(fp['avg'][0], 1)} / {pct(fp['avg'][1], 1)} / {pct(fp['avg'][2], 1)} — the first is worth "
          f"{pts(fp['v_first'], 2)} points, the second {pts(fp['v_second'], 2)}")
    print(f"  a 0-0 strike on the leadoff hitter: reversing it is worth {pts(s1['gain'], 2)}; failing costs "
          f"{pts(s1['risk'], 2)} with the last challenge (bar {pct(s1['bar'])}), {pts(s2['risk'], 2)} with two "
          f"(bar {pct(s2['bar'])})")
    v, tg = replay(val, sol)
    st = season_stats(val, v, tg)
    print(f"  every 2026 team-game ({st['n']:,}): one challenge used well recovers {st['mean1']:.2f} points on "
          f"average (the model: {pts(fp['v_first'], 2)}); two, {st['mean2']:.2f} (the model: "
          f"{pts(fp['v_all'], 2)})")
    print(f"  one challenge used well: {st['expected_challenges']:.2f} challenges a game, fixing "
          f"{st['expected_overturns']:.2f} of {st['expected_wrong']:.2f} wrong calls; by inning "
          f"{st['by_inning'].min():.2f}–{st['by_inning'][st['by_inning'].index <= 9].max():.2f} points; the "
          f"game's first pitch {st['first_pitch']:.3f}")
    row = pick_game(tg, args.game, args.team)
    print(f"  the game shown: {row['game_pk']} ({row['team']}) — {100 * row['one']:.2f} points from "
          f"{int(row['policy_challenges'])} challenges used well; {int(row['actual_challenges'])} real "
          f"challenges won {100 * row['actual_gain']:.2f}")
    S.apply_style(bg="white")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_value(FIGS, demo, fp)
    detail, n_right, n_calls = fig_one_game(FIGS, demo, val, v, row, tg["one"].mean(), m22.describe)
    fig_all_games(FIGS, demo, tg, fp, row, st)
    DOCS.mkdir(parents=True, exist_ok=True)
    write_doc(DOCS / "FIRST_CHALLENGE_VALUE.md", demo, fp, dr, st, row, detail, n_right, n_calls)
    tg.to_csv(DOCS / "first_challenge_team_games.csv", index=False)
    print(f"  → {DOCS / 'FIRST_CHALLENGE_VALUE.md'}, {DOCS / 'first_challenge_team_games.csv'}  "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
