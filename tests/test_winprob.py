#!/usr/bin/env python3
"""
test_winprob.py — does src/abs_winprob.py (and abs_value.py) get the known answers?

Each check plants an answer worked out by hand and asks the code to find it:

  1. Working back from the end of the game — with the same run distribution for
     both teams (a run in an inning with chance a, otherwise none):
       * the game is a coin flip before it starts (symmetry)
       * extra innings are a coin flip: x = 1/2
       * bottom 9th, home down 1   ->  a · x                (must tie, then extras)
       * top 9th, home up 1        ->  (1 − a) + a · (a + (1 − a) · x)
     and V never goes DOWN when the home team's lead goes up
  2. The run-expectancy chain recovers the exact answer for a toy game where
     every plate appearance is an out (probability q) or a walk
  3. The rulebook: ball four with the bases loaded scores a run; with runners on
     1st and 3rd it loads them; strike three with two out ends the inning
  4. The state parser takes Statcast's own column names and refuses impossible
     states (three outs, four balls, no score)
  5. The locked bundle: a saved V(s) reloads to the same numbers, refuses a file
     that changed, and values calls the right way round (an overturned strike
     three helps the batting team; an overturned ball four helps the fielders)
  6. The fast game bootstrap's Brier equals the direct calculation; identical
     models differ by exactly zero
  7. The skill tests find a planted difference between clusters and stay quiet
     when there is none; series are grouped and "deciding games" found correctly

    python tests/test_winprob.py        (a few seconds)
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import abs_metrics as M                                       # noqa: E402
from src.abs_calibration import Calibrator                             # noqa: E402
from src.abs_gamestate import N_BASES, N_COUNTS, N_OUTS                # noqa: E402
from src.abs_value import cluster_signal, series_effects, series_table, split_half  # noqa: E402
from src.abs_winprob import (DMAX, N_END, RMAX, CallTransitions, MarkovWP,   # noqa: E402
                             RunModel, WPStateError, game_bootstrap, load_winprob,
                             parse_states, write_bundle)

RNG = np.random.default_rng(10)
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def bernoulli_model(a):
    """A RunModel stand-in: every half-inning, from any state, scores 1 run with chance a."""
    rm = RunModel()
    f = np.zeros(RMAX + 1)
    f[0], f[1] = 1 - a, a
    rm.Z = np.broadcast_to(f, (2, N_OUTS, N_BASES, RMAX + 1)).copy()
    rm.R = np.broadcast_to(f, (2, N_OUTS, N_BASES, N_COUNTS, RMAX + 1)).copy()
    rm.RE = (rm.R * np.arange(RMAX + 1)).sum(-1)
    rm.n_cell = np.ones((2, N_OUTS, N_BASES, N_COUNTS))
    rm.T = np.zeros((2, N_OUTS, N_BASES, N_COUNTS, N_END))
    rm.n_fit = 0
    return rm


def V(mk, inning, half, d, outs=0, bases=0, c=0):
    i = min(inning, 10) - 1
    return float(mk.v_table[i, half, outs, bases, c, d + DMAX])


print("\nWorking back from the end of the game (known answers)")
a = 0.3
mk = MarkovWP(bernoulli_model(a))
check("symmetric teams: the game starts as a coin flip", abs(mk.pregame - 0.5) < 1e-12,
      f"{mk.pregame:.15f}")
check("extra innings are a coin flip too (x = 1/2)", abs(mk.x_extra - 0.5) < 1e-12,
      f"x = {mk.x_extra:.15f}")
v = V(mk, 9, 1, -1)
check("bottom 9th, home down 1 = a · x", abs(v - a * 0.5) < 1e-12, f"{v:.6f} vs {a * 0.5:.6f}")
v = V(mk, 9, 0, 1)
want = (1 - a) + a * (a + (1 - a) * 0.5)
check("top 9th, home up 1 = (1−a) + a·(a + (1−a)·x)", abs(v - want) < 1e-12,
      f"{v:.6f} vs {want:.6f}")
diffs = np.diff(mk.v_table, axis=-1)
check("V never falls when the home lead grows", diffs.min() > -1e-12, f"min step {diffs.min():.2e}")
check("every V is a probability", (mk.v_table.min() >= -1e-12) and (mk.v_table.max() <= 1 + 1e-12))

print("\nThe run-expectancy chain on a toy game (out with chance q, else a walk)")
q = 0.7
rows = []
for _ in range(20000):                      # half-innings
    outs, b, runs_total = 0, 0, 0
    seq = []
    while outs < 3:
        if RNG.random() < q:
            nb, r, no = b, 0, outs + 1
            cls = "OUT"
        else:
            nb, r, _ = 0, 0, 0
            eo, nb, ec, r = CallTransitions.rulebook(outs, b, 3 * 3 + 0, 0)   # ball four from 3-0
            no = outs
            cls = "BB"
        seq.append((outs, b, no, nb if no < 3 else 0, r, cls))
        outs, b = no, (nb if no < 3 else 0)
    for (o, bb, no, nb, r, cls) in seq:
        rows.append({"half": 0, "outs": o, "bases": bb, "count": 0, "pa_end_outs": no,
                     "pa_end_bases": nb, "runs_rest_pa": r, "pa_class": cls})
toy = pd.DataFrame(rows)
toy = pd.concat([toy, toy.assign(half=1)], ignore_index=True)
rm = RunModel().fit(toy)
# exact answer: solve the walk-or-out chain directly
RE = np.zeros((4, 8))
for _ in range(500):
    new = np.zeros_like(RE)
    for o in range(3):
        for b in range(8):
            _, nbw, _, rw = CallTransitions.rulebook(o, b, 9, 0)
            out_val = 0.0 if o + 1 >= 3 else RE[o + 1, b]
            new[o, b] = q * out_val + (1 - q) * (rw + RE[o, nbw])
    RE = new
seen = rm.n_cell[0, :, :, 0] > 0                  # walks only ever load the bases in order
err = np.abs(rm.RE[0, :, :, 0] - RE[:3])[seen].max()
check(f"RE24 matches the exact answer in every state the toy game reaches ({int(seen.sum())})",
      err < 0.06, f"max error {err:.4f} runs (sampling noise: the loaded-bases cells are rare)")

print("\nThe rulebook")
check("ball four with the bases loaded forces in a run",
      CallTransitions.rulebook(1, 7, 9, 0) == (1, 7, 0, 1))
check("ball four with runners on 1st and 3rd loads the bases, no run",
      CallTransitions.rulebook(0, 5, 9, 0) == (0, 7, 0, 0))
check("ball four with a runner on 2nd only puts the batter on 1st",
      CallTransitions.rulebook(0, 2, 11, 0) == (0, 3, 0, 0))
check("strike three with two out ends the inning", CallTransitions.rulebook(2, 3, 5, 1)[0] == 3)
check("strike two is just the next count", CallTransitions.rulebook(0, 0, 1, 1) == (0, 0, 2, 0))

print("\nThe state parser")
s = parse_states({"inning": 9, "inning_topbot": "Bot", "outs_when_up": 1, "on_3b": 660271.0,
                  "on_1b": np.nan, "home_score": 3, "away_score": 3, "balls": 1, "strikes": 2})
check("reads Statcast's names (inning_topbot, outs_when_up, runner ids)",
      s["h"][0] == 1 and s["o"][0] == 1 and s["b"][0] == 4 and s["d"][0] == 0 and s["c"][0] == 5)
s = parse_states({"inning": 12, "half": "top", "outs": 0, "on_2b": True, "score_diff": -2})
check("extras map to one inning; True means a runner; default count 0-0",
      s["i"][0] == 9 and s["b"][0] == 2 and s["c"][0] == 0)
for bad, why in (({"inning": 3, "half": "top", "outs": 3, "score_diff": 0}, "three outs"),
                 ({"inning": 3, "half": "top", "outs": 0, "score_diff": 0, "balls": 4}, "four balls"),
                 ({"inning": 3, "half": "top", "outs": 0}, "no score")):
    try:
        parse_states(bad)
        check(f"refuses {why}", False)
    except WPStateError:
        check(f"refuses {why}", True)

print("\nThe locked bundle")
rm = bernoulli_model(0.28)
mk = MarkovWP(rm)
empty = pd.DataFrame({c: pd.Series(dtype=float) for c in
                      ("outs", "bases", "count", "call_effective", "next_outs", "next_bases",
                       "next_count", "runs_next")}).assign(is_called=pd.Series(dtype=bool))
calls = CallTransitions().fit(empty)
tmp = Path(tempfile.mkdtemp())
try:
    man = write_bundle(tmp / "winprob", markov=mk, run_model=rm, calls=calls,
                       calibrator=Calibrator("none"), manifest_extra={"date": "2026.01.01"})
    wp = load_winprob(tmp / "winprob")
    st = {"inning": 7, "half": "bot", "outs": 1, "bases": 0, "score_diff": -1}
    check("reloads to the same numbers", abs(wp.home(st) - V(mk, 7, 1, -1, outs=1)) < 1e-6)
    k3 = wp.call_value({"inning": 9, "half": "bot", "outs": 2, "bases": 0, "score_diff": 0,
                        "balls": 1, "strikes": 2}, "strike").iloc[0]
    check("overturned strike three helps the batting team (win prob. and runs)",
          k3["dwp_challenger"] > 0 and k3["druns_batting"] > 0,
          f"{k3['dwp_challenger']:+.3f}, {k3['druns_batting']:+.3f} runs")
    b4 = wp.call_value({"inning": 9, "half": "top", "outs": 0, "bases": 7, "score_diff": 0,
                        "balls": 3, "strikes": 1}, "ball").iloc[0]
    check("overturned ball four helps the fielding team; the batting team loses runs",
          b4["dwp_challenger"] > 0 and b4["druns_batting"] < 0,
          f"{b4['dwp_challenger']:+.3f}, {b4['druns_batting']:+.3f} runs")
    both = wp.call_value(pd.DataFrame([{"inning": 4, "half": "top", "outs": 0, "bases": 1,
                                        "score_diff": 0, "balls": 2, "strikes": 2}] * 2),
                         ["strike", "ball"])
    check("a strike and a ball on the same pitch are mirror images",
          abs(both["wp_stands"].iloc[0] - both["wp_overturned"].iloc[1]) < 1e-12)
    np.save(tmp / "winprob" / "v_table.npy", np.zeros(3))
    try:
        load_winprob(tmp / "winprob")
        check("refuses a bundle whose files changed", False)
    except RuntimeError:
        check("refuses a bundle whose files changed", True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nThe fast game bootstrap")
n = 6000
games = np.repeat(np.arange(600), 10)
p = RNG.uniform(0.05, 0.95, n)
y = (RNG.random(n) < p).astype(int)
point, ci, d = game_bootstrap(y, {"a": p, "b": p.copy()}, games, B=100, seed=1)
check("Brier equals the direct calculation", abs(point["a"]["brier"] - M.brier(y, p)) < 1e-12)
check("identical models differ by exactly zero", d[("a", "b")]["brier"][0] == 0.0)
check("a calibrated model's slope interval contains 1",
      ci["a"]["cal_slope"][0] <= 1 <= ci["a"]["cal_slope"][1],
      f"[{ci['a']['cal_slope'][0]:.2f}, {ci['a']['cal_slope'][1]:.2f}]")

print("\nSkill tests and series")
lab = np.repeat(np.arange(40), 50)
eff = RNG.normal(0, 0.15, 40)[lab]
v = RNG.normal(0, 1, len(lab)) + eff * 4
_, p_eff, _, sd_eff = cluster_signal(v, lab, n_perm=199, seed=1)
check("finds a planted difference between 40 clusters", p_eff < 0.05, f"p = {p_eff:.3f}, sd {sd_eff:.2f}")
v0 = RNG.normal(0, 1, len(lab))
_, p0, _, sd0 = cluster_signal(v0, lab, n_perm=199, seed=2)
check("stays quiet when there is none", p0 > 0.05, f"p = {p0:.3f}")
rel, n_rel = split_half(v, lab, np.arange(len(lab)))
check("split-half reliability is high for a stable trait", rel > 0.5, f"{rel:.2f} over {n_rel}")
g = pd.DataFrame({"game_pk": [1, 2, 3, 4, 5], "home_team": ["A", "A", "A", "B", "B"],
                  "away_team": ["B", "B", "B", "A", "A"],
                  "game_date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03",
                                               "2026-06-10", "2026-06-11"]),
                  "home_win": [1, 0, 1, 1, 1]})
ser = series_table(g)
check("games are grouped into series (3 at A, then 2 at B)",
      sorted(ser.groupby("series_id").size().tolist()) == [2, 3])
val = pd.DataFrame({"game_pk": [1, 2], "chall_team": ["A", "B"], "gain": [0.05, 0.02]})
per_game, _ = series_effects(val, ser)
a_games = per_game[(per_game["team"] == "A") & (per_game["series_len"] == 3)]
check("a team that wins a series 2–1: its two wins decided it, its loss didn't",
      a_games.sort_values("game_pk")["pivotal"].tolist() == [True, False, True])
check("net challenge value = own overturns minus the opponent's",
      abs(a_games.sort_values("game_pk")["net_gain"].iloc[0] - 0.05) < 1e-12 and
      abs(a_games.sort_values("game_pk")["net_gain"].iloc[1] + 0.02) < 1e-12)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL WIN-PROBABILITY CHECKS PASSED")
