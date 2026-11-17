"""
showcase_figures.py — slide-ready figures for the checkpoint showcase. Called by
22_showcase_examples.py; not meant to be run on its own.

Written for an audience that doesn't follow baseball: every slide says what
happened in plain words, the numbers are labelled, and the strike-zone panel is
drawn to scale (the ABS zone from the batter's height, the ball 2.9 in. across).

Same house style as Weeks 7–11 (white, 16:9, 200 DPI, each PNG with its data CSV).
Colour jobs:
    the umpire's call on the pitch   blue = called BALL, orange = called STRIKE (always labelled)
    challenge outcome (map)          aqua = overturned, orange = call stood, gray = not challenged
    win probability (stories)        blue line; the what-if branch dashed orange
"""

import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eda_style as S                       # noqa: E402
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle   # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HALF_W = 8.5                                 # inches: half the plate
BALL_R = 1.45                                # inches: a baseball's radius
CALL_COLOR = {True: S.ORANGE, False: S.BLUE}
SHORT_CAT = {"swing": "biggest swing", "leverage": "biggest moment", "obvious_fixed": "most obvious fix",
             "longshot": "long shot", "bad_challenge": "not worth it", "missed_obvious": "obvious, missed",
             "missed_costly": "most costly miss", "empty": "nothing left", "contrast": "same chance"}


def _pct(v, d=0):
    return "—" if v is None or not np.isfinite(v) else f"{100 * v:.{d}f}%"


def _wrap(s, width):
    return "\n".join(textwrap.wrap(str(s), width=width))


def _text(ax, x, y, s, size=12, color=None, weight="normal", width=None, **kw):
    t = ax.text(x, y, _wrap(s, width) if width else s, fontsize=size, color=color or S.INK,
                fontweight=weight, va="top", ha=kw.pop("ha", "left"), transform=ax.transAxes, **kw)
    return t


# ------------------------------------------------------------------ the strike-zone panel
def zone_panel(ax, r, show_call=True, show_gap=True):
    x, z = 12 * float(r["plate_x"]), 12 * float(r["plate_z"])
    top, bot = 12 * float(r["abs_zone_top"]), 12 * float(r["abs_zone_bot"])
    ax.add_patch(Rectangle((-HALF_W, bot), 2 * HALF_W, top - bot, fc=S.SEQ[0], ec="none", alpha=0.55, zorder=1))
    ax.add_patch(Rectangle((-HALF_W, bot), 2 * HALF_W, top - bot, fill=False, ec=S.INK, lw=2.4, zorder=3))
    strike = bool(r["call_strike"])
    ax.add_patch(Circle((x, z), BALL_R, fc=CALL_COLOR[strike] if show_call else S.MUTED, ec=S.INK,
                        lw=1.3, zorder=6))
    # the gap (or overlap) between the ball's edge and the zone
    gap = float(r["zone_dist_in"]) if np.isfinite(r["zone_dist_in"]) else np.nan
    nx, nz = np.clip(x, -HALF_W, HALF_W), np.clip(z, bot, top)
    inside = (nx == x) and (nz == z)
    if inside:                                         # centre inside the zone: nearest edge
        cands = [(-HALF_W, z), (HALF_W, z), (x, bot), (x, top)]
        nx, nz = min(cands, key=lambda p: np.hypot(p[0] - x, p[1] - z))
    dx, dz = nx - x, nz - z
    dist = np.hypot(dx, dz)
    if show_gap and np.isfinite(gap) and gap <= 0 and not inside:     # the ball's edge overlaps the zone
        ux, uz = (x - nx) / max(dist, 1e-9), (z - nz) / max(dist, 1e-9)    # outward, away from the zone
        tx, tz = x + ux * (BALL_R + 1.5), z + uz * (BALL_R + 1.5)
        if abs(ux) >= abs(uz):
            ha, va = ("left" if ux > 0 else "right"), "center"
        else:
            ha, va = "center", ("bottom" if uz > 0 else "top")
        t = ax.text(tx, tz, f"{abs(gap):.1f} in. of the ball\ninside the zone", fontsize=10, ha=ha, va=va,
                    color=S.INK, zorder=8)
        S.halo(t)
    elif show_gap and np.isfinite(gap) and dist > 1e-6:
        ux, uz = dx / dist, dz / dist
        sx, sz = (x + ux * BALL_R, z + uz * BALL_R) if not inside else (x, z)
        ax.annotate("", xy=(nx, nz), xytext=(sx, sz), zorder=7,
                    arrowprops=dict(arrowstyle="-", color=S.INK, lw=1.4, linestyle=(0, (2, 1.5))))
        if inside:                       # label inside the zone, on the far side of the ball
            tx, tz = x - ux * (BALL_R + 1.2), z - uz * (BALL_R + 1.2)
            if abs(ux) >= abs(uz):
                ha, va = ("right" if ux > 0 else "left"), "center"
            else:
                ha, va = "center", ("top" if uz > 0 else "bottom")
            word = f"{abs(gap):.1f} in. inside"
        else:                            # label beside the middle of the gap
            tx, tz = (sx + nx) / 2 + (-uz) * 3.0, (sz + nz) / 2 + ux * 3.0
            ha, va = "center", "center"
            word = f"{gap:.1f} in. gap"
        t = ax.text(tx, tz, word, fontsize=10.5, ha=ha, va=va, color=S.INK, zorder=8)
        S.halo(t)
    xl = max(21.0, abs(x) + 7)
    ax.set_xlim(-xl, xl)
    ax.set_ylim(min(bot - 13, z - 8), max(top + 11, z + 8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, ax.get_ylim()[0] + 0.5, "catcher's view · drawn to scale", ha="center", va="bottom",
            fontsize=9, color=S.MUTED)
    # a small key in the top-left corner (outside the data, so it never sits on the ball)
    y0 = ax.get_ylim()[1]
    if show_call:
        ax.add_patch(Circle((-xl + 2.2, y0 + 3.4), 1.1, fc=CALL_COLOR[strike], ec=S.INK, lw=1, clip_on=False))
        ax.text(-xl + 4.2, y0 + 3.4, f"umpire called {'STRIKE' if strike else 'BALL'}",
                fontsize=11, va="center", color=S.INK, fontweight="semibold")
    ax.add_patch(Rectangle((-xl + 1.1, y0 + 0.2), 2.2, 1.5, fc=S.SEQ[0], ec=S.INK, lw=1.2, clip_on=False))
    ax.text(-xl + 4.2, y0 + 0.95, "ABS strike zone (from the batter's height)", fontsize=9.5, va="center",
            color=S.INK_2)


def _tiles(ax, r, show_bar=True):
    swing_lbl = (f"points for {r['chall_team']} if reversed\n"
                 f"(chance to win {_pct(r['wp_stands_chall'])}→{_pct(r['wp_over_chall'])})")
    items = [(_pct(r["p_s"]), "chance the call\nwas wrong, p(s)"),
             (f"{100 * r['dwp']:+.1f}", swing_lbl),
             (f"{r['li']:.1f}", "leverage\n(1 = typical moment)")]
    if show_bar:
        bar = r["p_star"] if (r["r_self"] >= 1 and np.isfinite(r["p_star"])) else np.nan
        items.append((_pct(bar) if np.isfinite(bar) else "—", "confidence needed\nto challenge (the bar)"))
    w = 1.0 / len(items)
    for k, (big, lab) in enumerate(items):
        x0 = k * w
        box = FancyBboxPatch((x0 + 0.004, 0.445), w - 0.014, 0.215, boxstyle="round,pad=0.006,rounding_size=0.02",
                             transform=ax.transAxes, fc="#f4f6f9", ec="none", zorder=0)
        ax.add_patch(box)
        ax.text(x0 + 0.02, 0.645, big, fontsize=22, fontweight="bold", color=S.INK, va="top",
                transform=ax.transAxes)
        ax.text(x0 + 0.02, 0.555, lab, fontsize=8.8, color=S.INK_2, va="top", transform=ax.transAxes,
                linespacing=1.25)


def card(out, demo, r, t, tag, name, mode="slide", link=None):
    """One slide for one pitch. mode: 'slide', 'question' (quiz) or 'answer' (quiz)."""
    fig = plt.figure(figsize=S.FULL)
    axz = fig.add_axes([0.02, 0.07, 0.36, 0.86])
    zone_panel(axz, r)
    ax = fig.add_axes([0.41, 0.05, 0.57, 0.9])
    ax.axis("off")
    _text(ax, 0, 1.0, tag.upper(), size=10.5, color=S.MUTED, weight="semibold")
    _text(ax, 0, 0.93, t["situation"], size=19, weight="bold", width=48)
    _text(ax, 0, 0.845, t["runners"], size=13.5)
    _text(ax, 0, 0.785, f"{t['matchup']} · {t['game']}", size=10.5, color=S.INK_2, width=92)
    _tiles(ax, r, show_bar=(mode != "question"))
    _text(ax, 0, 0.395, f"{t['call']}.  {t['verdict']}", size=11.5, width=82)
    if mode == "question":
        _text(ax, 0, 0.27, "Did they challenge?", size=26, weight="bold")
        left = int(r["r_self"])
        _text(ax, 0, 0.15, f"{r['chall_team']} had {left} challenge{'s' if left != 1 else ''} left.",
              size=13, color=S.INK_2)
    else:
        lead = "ANSWER: " if mode == "answer" else "What happened: "
        _text(ax, 0, 0.30, lead + t["happened"], size=12.5, weight="semibold", width=74)
        _text(ax, 0, 0.19, "The MDP: " + t["model"] + ".", size=12, width=78)
    foot = (f"game {int(r['game_pk'])} · inning {int(r['inning'])} {'top' if r['half'] == 0 else 'bottom'}, "
            f"at-bat #{int(r['at_bat_number'])}, pitch #{int(r['pitch_number'])} · replay: "
            f"{link or 'baseballsavant.mlb.com/gamefeed?gamePk=' + str(int(r['game_pk']))}")
    _text(ax, 0, 0.05, foot, size=8.5, color=S.MUTED, width=120)
    S.demo_stamp(fig, demo)
    cols = ["game_pk", "at_bat_number", "pitch_number", "inning", "half", "p_s", "dwp", "li", "p_star",
            "challenged", "overturned", "plate_x", "plate_z", "abs_zone_top", "abs_zone_bot", "zone_dist_in"]
    S.save(fig, out, name, pd.DataFrame([{c: r[c] for c in cols if c in r.index}]))


# ------------------------------------------------------------------ leaderboards
def leaderboard(out, demo, key, title, unit, col, board, describe, name):
    d = board.reset_index(drop=True)
    if not len(d):
        return False
    vals = d[col].to_numpy(float)
    show = (100 * vals if col in ("dwp", "regret") else 100 * vals if col == "p_s" else vals)
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(left=0.36, right=0.95, top=0.82, bottom=0.1)
    y = np.arange(len(d))[::-1]
    ax.set_xlim(0, max(np.nanmax(show) * 1.18, 1e-6))
    for k, (yy, v) in enumerate(zip(y, show)):
        S.rounded_barh(ax, [yy], [v], color=S.BLUE if k == 0 else S.SEQ[2])
        txt = (f"{v:+.1f} pts" if col in ("dwp", "regret") else f"{v:.1f}%" if col == "p_s"
               else f"{v:.2f}" if col == "li" else f"{v:.1f} in.")
        t = ax.text(v + ax.get_xlim()[1] * 0.012, yy, txt, va="center", fontsize=10)
        S.halo(t)
    labels = []
    for _, r in d.iterrows():
        s = describe(r)["short"]
        tag = (" · overturned" if r["overturned"] else " · stood") if r["challenged"] else ""
        labels.append(s + tag)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    S.pad_cat_axis(ax, len(d))
    S.only_grid(ax, "x")
    ax.set_xlabel({"dwp": "win-probability points", "regret": "win-probability points",
                   "p_s": "chance of a reversal (%)", "li": "leverage index",
                   "dist_against_call": "inches on the wrong side of the zone edge"}.get(col, col))
    S.titles(ax, title, f"Top {len(d)} of the season — {unit}")
    S.footnote(fig, "date · away @ home · half-inning · count (balls-strikes). The replay link for each is in "
                    "docs/SHOWCASE.md.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, name, d[[c for c in ("game_pk", "at_bat_number", "pitch_number", col, "challenged",
                                          "overturned") if c in d.columns]])
    return True


# ------------------------------------------------------------------ game stories
def story(out, demo, r, g, v, runs_away, runs_home, name, describe):
    """The home team's chance to win after every pitch of the game, with the showcase pitch and
    what would have happened had its call gone the other way."""
    n = len(g)
    if not n:
        return False
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(top=0.82, bottom=0.14, right=0.80)
    xs = np.arange(n + 1)
    home_v = np.r_[v, 1.0 if runs_home > runs_away else 0.0]
    ax.plot(xs, 100 * home_v, color=S.BLUE, lw=2.2, zorder=3)
    starts = g.groupby(["inning", "half"]).head(1)
    for idx, row in starts.iterrows():
        if row["half"] == 0:
            ax.axvline(idx, color=S.GRID, lw=1, zorder=1)
            ax.text(idx + 1, 101.5, str(int(row["inning"])), fontsize=9, color=S.MUTED, va="bottom")
    hit = g.index[(g["at_bat_number"] == r["at_bat_number"]) & (g["pitch_number"] == r["pitch_number"])]
    if len(hit):
        i = int(hit[0])
        home = bool(r["chall_home"])
        stands, over = r["wp_stands_chall"], r["wp_over_chall"]
        if not home:
            stands, over = 1 - stands, 1 - over
        happened = over if r["overturned"] else stands
        other = stands if r["overturned"] else over
        ax.scatter([i], [100 * home_v[i]], s=60, color=S.INK, zorder=6)
        ax.plot([i + 1, i + 1], [100 * happened, 100 * other], color=S.ORANGE, lw=2.4, ls=(0, (4, 2.5)), zorder=5)
        ax.scatter([i + 1], [100 * other], s=55, color=S.ORANGE, zorder=6, edgecolor="white", linewidth=1)
        lab = ax.annotate(f"this pitch: {_pct(home_v[i])} → {_pct(happened)}\n"
                          f"if the call had gone the other way: {_pct(other)}",
                          (i + 1, 100 * other), xytext=(14, 14 if other < 0.85 else -34),
                          textcoords="offset points", fontsize=10.5, color=S.INK,
                          arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1))
        S.halo(lab)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, n + 1)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0f}%"))
    S.only_grid(ax, "y")
    ax.set_ylabel(f"{r['home_team']} (home) chance to win")
    ax.set_xlabel("every pitch of the game, in order (numbers = innings)")
    t = describe(r)
    S.titles(ax, f"{r['away_team']} at {r['home_team']}, {pd.Timestamp(r['game_date']):%b %-d}: "
                 f"the game, pitch by pitch",
             f"{t['situation']} — {t['call'].lower()}; {t['happened']}")
    S.footnote(fig, f"Win probability from Week 10's model after every pitch. Final: {r['away_team']} "
                    f"{runs_away}, {r['home_team']} {runs_home}.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, name, pd.DataFrame({"pitch": xs, "home_win_prob": home_v}))
    return True


# ------------------------------------------------------------------ glossary
def glossary(out, demo):
    fig = plt.figure(figsize=S.FULL)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.8])
    ax.axis("off")
    items = [
        ("Chance the call was wrong — p(s)",
         "From exactly where the ball crossed the plate (and nothing else), the chance that ABS, the "
         "robot umpire, would reverse the call if a team challenged it. Week 9's model."),
        ("Chance to win",
         "A team's probability of winning from this exact moment: the inning, the score, the runners, "
         "the outs and the count. A point is one percentage point; 100 points add up to one win. Week 10."),
        ("Leverage",
         "How much this moment can swing the game. 1 is a typical plate appearance, above 2 is a big "
         "moment, and the tensest moments of a season reach about 10. Tom Tango's Leverage Index."),
        ("The bar",
         "The confidence a team needs before a challenge is worth the risk. A failed challenge is gone "
         "for the rest of the game, so the bar is high early with one left and near zero in a tied 9th. "
         "Week 11."),
    ]
    for k, (head, body) in enumerate(items):
        cx, cy = (k % 2) * 0.5, 0.5 - (k // 2) * 0.5
        ax.add_patch(FancyBboxPatch((cx + 0.01, cy + 0.02), 0.47, 0.44,
                                    boxstyle="round,pad=0.01,rounding_size=0.03", fc="#f4f6f9", ec="none",
                                    transform=ax.transAxes))
        ax.text(cx + 0.035, cy + 0.42, head, fontsize=15, fontweight="bold", color=S.INK, va="top",
                transform=ax.transAxes)
        ax.text(cx + 0.035, cy + 0.32, _wrap(body, 52), fontsize=11.5, color=S.INK_2, va="top",
                transform=ax.transAxes, linespacing=1.4)
    S.suptitles(fig, "Four ideas behind every slide", "The ABS challenge: a team has two; a successful "
                "challenge is kept, a failed one is lost.", top=0.97)
    S.demo_stamp(fig, demo)
    S.save(fig, out, "00_glossary")
    return True


# ------------------------------------------------------------------ the map of every decision
def decision_map(out, demo, opp, labelled, seed=0):
    d = opp.copy()
    d["y"] = np.clip(100 * d["dwp"].to_numpy(float), 0.01, None)
    rng = np.random.default_rng(seed)
    bg = d[~d["challenged"]]
    if len(bg) > 40000:
        bg = bg.iloc[rng.choice(len(bg), 40000, replace=False)]
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(top=0.82, right=0.97)
    ax.scatter(bg["p_s"], bg["y"], s=3, color=S.DEEMPH, alpha=0.35, lw=0, zorder=1,
               label=f"not challenged ({(~d['challenged']).sum():,})")
    ch = d[d["challenged"]]
    for ok, color, lab in ((True, S.AQUA, "challenged — overturned"), (False, S.ORANGE, "challenged — call stood")):
        x = ch[ch["overturned"] == ok]
        ax.scatter(x["p_s"], x["y"], s=8, color=color, alpha=0.6, lw=0, zorder=2, label=f"{lab} ({len(x):,})")
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.02)
    fig.canvas.draw()
    placed = []
    offsets = [(8, 6), (8, -16), (-8, 6), (-8, -16), (10, 22), (10, -32), (-10, 22), (-10, -32), (14, 40),
               (14, -50), (-14, 40), (-14, -50)]
    for key, r in labelled:
        yy = max(100 * r["dwp"], 0.01)
        ax.scatter([r["p_s"]], [yy], s=90, facecolor="none", edgecolor=S.INK, lw=1.6, zorder=5)
        for ox, oy in offsets:
            t = ax.annotate(SHORT_CAT.get(key, key), (r["p_s"], yy), xytext=(ox, oy), textcoords="offset points",
                            fontsize=9.5, color=S.INK, zorder=6, ha="left" if ox > 0 else "right",
                            arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.8) if abs(oy) > 16 else None)
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                S.halo(t)
                break
            t.remove()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
    ax.set_xlabel("chance the call was wrong, p(s)")
    ax.set_ylabel("win-probability points at stake (log scale)")
    S.only_grid(ax, "y")
    leg = ax.legend(loc="lower right", frameon=True, fontsize=9.5, markerscale=3)
    leg.get_frame().set_edgecolor("none")
    S.titles(ax, "Every 2026 challenge opportunity: how wrong the call likely was, and how much it mattered",
             "Each dot is a called pitch that went against a team holding a challenge; circled = examples in these slides")
    S.footnote(fig, "Top right = clearly wrong AND important. Gray dots up there are big, obvious mistakes "
                    "nobody challenged.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "01_decision_map", pd.DataFrame([{"example": SHORT_CAT.get(k, k), "game_pk": r["game_pk"],
                                                        "at_bat_number": r["at_bat_number"],
                                                        "pitch_number": r["pitch_number"], "p_s": r["p_s"],
                                                        "dwp": r["dwp"]} for k, r in labelled]))
    return True


# ------------------------------------------------------------------ for scale
def swings_chart(out, demo, swings, best_call, describe_any):
    d = swings.reset_index(drop=True)
    if not len(d):
        return False
    vals = 100 * d["wp_change_batting"].to_numpy(float)
    fig, ax = plt.subplots(figsize=S.FULL)
    fig.subplots_adjust(left=0.40, right=0.95, top=0.82, bottom=0.1)
    y = np.arange(len(d))[::-1]
    ax.set_xlim(min(0, vals.min() * 1.2), max(vals.max() * 1.2, (best_call or 0) * 1.3, 1))
    for yy, v in zip(y, vals):
        S.rounded_barh(ax, [yy], [v], color=S.BLUE if v >= 0 else S.ORANGE)
        t = ax.text(v + (1 if v >= 0 else -1) * ax.get_xlim()[1] * 0.012, yy, f"{v:+.0f} pts",
                    va="center", ha="left" if v >= 0 else "right", fontsize=10)
        S.halo(t)
    ax.set_yticks(y)
    ax.set_yticklabels([describe_any(r) for _, r in d.iterrows()], fontsize=9.5)
    S.pad_cat_axis(ax, len(d))
    S.only_grid(ax, "x")
    if best_call:
        ax.axvline(best_call, color=S.INK, lw=1.4, ls=(0, (4, 3)))
        t = ax.text(best_call, len(d) - 0.35, f" biggest overturned call: {best_call:+.0f} pts", fontsize=10,
                    va="bottom", color=S.INK)
        S.halo(t)
    ax.set_xlabel("change in the batting team's chance to win on that one pitch (points)")
    S.titles(ax, "For scale: the biggest single-pitch swings of the season",
             "Any pitch — mostly late home runs and big hits — next to the biggest swing from an overturned call")
    S.footnote(fig, "Win probability before and after each pitch, from Week 10's model; the last pitch of a "
                    "game counts the final result.")
    S.demo_stamp(fig, demo)
    S.save(fig, out, "02_biggest_swings_any_pitch",
           d[["game_pk", "at_bat_number", "pitch_number", "wp_change_batting", "events"]])
    return True


def make_all(out, data_dir, *, cats, quiz, texts, stories, series, swings, opp, names_of, order,
             games, links, describe):
    """Every showcase figure. Returns how many were written."""
    S.apply_style(bg="white")
    demo = S.is_demo(data_dir)
    n = 0
    n += glossary(out, demo)
    labelled = [(k, cats[k][0].iloc[0]) for k in order if k in cats and len(cats[k][0])]
    n += decision_map(out, demo, opp, labelled)
    best = 100 * float(cats["swing"][0]["dwp"].max()) if "swing" in cats and len(cats["swing"][0]) else None

    def describe_any(r):
        ev = str(r.get("events") or "").replace("_", " ")
        return (f"{pd.Timestamp(r['game_date']):%b %-d} · {r['away_team']} @ {r['home_team']} · "
                f"{'top' if r['half'] == 0 else 'bot'} {int(r['inning'])}"
                + (f" · {ev}" if ev and ev != "nan" else ""))
    n += swings_chart(out, demo, swings, best, describe_any)
    for ci, key in enumerate(order, 1):
        if key not in cats:
            continue
        slides, board = cats[key]
        title, blurb, col, unit = names_of[key]
        if key != "contrast":
            n += leaderboard(out, demo, key, title, unit, col, board, describe, f"{ci + 2:02d}_{key}_top10")
        for i, (idx, r) in enumerate(slides.iterrows(), 1):
            card(out, demo, r, texts[idx], f"{title} · #{i}", f"{ci + 2:02d}_{key}_{i}", link=links.get(idx))
            n += 1
    fin = games.set_index("game_pk")[["final_away", "final_home"]]
    for j, (key, r) in enumerate(stories, 1):
        g, v = series[int(r["game_pk"])]
        f = fin.loc[int(r["game_pk"])]
        n += story(out, demo, r, g, v, int(f["final_away"]), int(f["final_home"]),
                   f"20_story_{j}_{key}", describe)
    for i, (idx, r) in enumerate(quiz.iterrows(), 1):
        card(out, demo, r, texts[idx], f"Quiz #{i} — was it challenged?", f"30_quiz_{i}_question",
             mode="question", link=links.get(idx))
        card(out, demo, r, texts[idx], f"Quiz #{i} — the answer", f"30_quiz_{i}_answer", mode="answer",
             link=links.get(idx))
        n += 2
    return n
