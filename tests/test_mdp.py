#!/usr/bin/env python3
"""
test_mdp.py — does the Week 11 challenge MDP (src/abs_mdp.py, src/abs_leverage.py)
get the answers that must hold in ANY baseball game?

Everything runs on the TOY game (PitchModel.toy(): every count alike, outs / singles
/ doubles / home runs, wrong calls planted at p(s) = 0.97 and 0.30), so the checks
need no data and take a few seconds each.

  The weekly plan's four
    1. p(s) = 1  →  always challenge (whenever the overturn helps at all)
    2. p(s) = 0  →  never challenge (whenever a challenge is worth keeping)
    3. r = 0     →  the option is not there
    4. the bar in a tied 9th with the winning run on 3rd is far LOWER than in a
       1st-inning blowout
  The Bellman equation itself
    5. at p* the team is exactly indifferent: Q_challenge(p*) = Q_hold, and the
       stored H and Δ are the differences of V they claim to be
    6. playing the solved policy as a FIXED rule gives back the same V; a game with
       nobody challenging between equal teams is a coin flip (0.5 exactly)
    7. more challenges never hurt: V rises in your own r and falls in the opponent's;
       the first challenge is worth more than the second (innings 1–8 — in the 9th
       the extra-inning rule can reverse it, see 9)
  Rules of the game
    8. walk-off: in the bottom of the 9th, a home lead is a win
    9. the 2026 extra-inning rule: a team with 0 challenges left gets one, so entering
       extras 0 and 1 are worth the same
   10. the simulator (pitch by pitch, random) wins as often as the backward
       induction says it should
  Leverage (Tango)
   11. LI averages exactly 1 over the plate appearances it was scaled on; a tied
       bottom of the 9th is several times the first pitch of the game, and a decided
       game is ≈ 0
   12. the Statcast/Tango breakeven: p* = 0.2 / (0.2 + run value); with LI = 1 the
       leverage-aware rule is the same rule
   13. the RE288 flip: ball four with the bases loaded forces in a run, and a ball is
       never worth less to the batter than a strike

    python tests/test_mdp.py           (about 30 seconds)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.abs_gamestate import N_BASES, N_COUNTS, N_OUTS                     # noqa: E402
from src.abs_leverage import Leverage                                         # noqa: E402
from src.abs_mdp import (D, NEVER, Breakeven, Constant, FromSolution,         # noqa: E402
                         GameMDP, PitchModel, Simulator, flip_runs)
from src.abs_winprob import N_END, MarkovWP, RunModel, _end_index             # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


pm = PitchModel.toy()
game = GameMDP(pm)
sol = game.solve()
print(f"toy game solved in {sol.seconds:.1f}s — home team wins {sol.pregame():.4f} with both "
      f"teams choosing well")

# ------------------------------------------------------------------ the plan's four
print("\nThe weekly plan's checks")
bad1 = bad0 = n_dec = 0
for (i, s), h in sol.halves.items():
    for t, H, Dl in (("thr_S", "H_S", "D_S"), ("thr_B", "H_B", "D_B")):
        T, Hh, Dd = h[t], h[H], h[Dl]
        av = np.isfinite(T)
        n_dec += int(av.sum())
        bad1 += int((av & (Dd > 1e-9) & (T > 1)).sum())
        bad0 += int((av & (Hh > 1e-12) & (T <= 0)).sum())
check("p(s) = 1: always challenge when the overturn helps", bad1 == 0,
      f"{bad1} of {n_dec:,} decision states would not")
check("p(s) = 0: never challenge when a challenge is worth keeping", bad0 == 0,
      f"{bad0} of {n_dec:,} decision states would")
r0 = all(np.all(~np.isfinite(h["thr_S"][..., 0, :])) and np.all(~np.isfinite(h["thr_B"][..., :, 0]))
         for h in sol.halves.values())
check("r = 0: no challenge left, no option", r0)
t9 = float(sol.lookup("thr_S", 9, 1, 1, 4, 11, 0, 1, 1))     # bottom 9, tie, runner on 3rd, 1 out, 3-2
t1 = float(sol.lookup("thr_S", 1, 1, 0, 0, 0, -8, 1, 2))     # bottom 1, home down 8, 0-0
check("tied 9th, winning run on 3rd: a far lower bar than a 1st-inning blowout",
      t9 < 0.2 and t1 > 0.5 and t9 < t1, f"p* = {t9:.3f} vs {t1:.3f}")

# ------------------------------------------------------------------ Bellman
print("\nThe Bellman equation")
h = sol.halves[(7, 0)]                                   # top of the 7th (no runner moves in the toy)
worst = 0.0
for (o, b, c, d, rb, rf) in [(0, 0, 0, D, 1, 1), (1, 3, 4, D - 1, 2, 1), (2, 5, 1, D + 2, 1, 2)]:
    V = h["V"]
    Sv, Bv = V[o, b, c + 1, d, rb, rf], V[o, b, c + 3, d, rb, rf]
    Sv_m = V[o, b, c + 1, d, rb - 1, rf]
    sg = -1                                              # the away team bats in the top
    p = h["thr_S"][o, b, c, d, rb, rf]
    q_ch, q_hold = p * Bv + (1 - p) * Sv_m, Sv
    worst = max(worst, abs(q_ch - q_hold), abs(h["H_S"][o, b, c, d, rb, rf] - sg * (Sv - Sv_m)),
                abs(h["D_S"][o, b, c, d, rb, rf] - sg * (Bv - Sv)))
check("indifferent at p*: Q_challenge(p*) = Q_hold; H and Δ are what they claim", worst < 1e-12,
      f"largest gap {worst:.1e}")
opt = FromSolution(sol)
again = game.solve(policies={"home": opt, "away": opt}).pregame()
check("the solved policy, played as a fixed rule, gives back the same V",
      abs(again - sol.pregame()) < 1e-10, f"{again:.12f} vs {sol.pregame():.12f}")
never = Constant(NEVER)
coin = game.solve(policies={"home": never, "away": never}).pregame()
check("equal teams, nobody challenges: a coin flip", abs(coin - 0.5) < 1e-5,
      f"{coin:.9f} — the 4e-6 left is the ±10-run cap on the lead; it vanishes as the cap widens")
mono = conc = 0.0
for (i, s), hh in sol.halves.items():
    V = hh["V"]
    sg = 1 if s == 1 else -1
    own = sg * (V[..., 1:, :] - V[..., :-1, :])          # batting team, its own challenges
    opp = sg * (V[..., :, 1:] - V[..., :, :-1])          # batting team, the opponent's
    mono = min(mono, own.min(), (-opp).min())
    if i <= 8:
        conc = min(conc, (own[..., 0, :] - own[..., 1, :]).min())
check("more challenges never hurt you, more for the opponent never help", mono > -1e-10,
      f"worst {mono:.1e}")
check("the first challenge is worth more than the second (innings 1–8)", conc > -1e-10,
      f"worst {conc:.1e}")

# ------------------------------------------------------------------ rules
print("\nRules of the game")
wo = sol.halves[(9, 1)]["V"][..., D + 1:, :, :]
check("walk-off: bottom of the 9th with the home team ahead is a win", np.allclose(wo, 1.0),
      f"min {wo.min():.6f}")
pre = GameMDP.replenish(sol.extra)
check("extras: 0 challenges left is worth the same as 1 (a team gets one back)",
      np.allclose(pre[0], pre[1]) and np.allclose(pre[:, 0], pre[:, 1]))
v_end9 = sol.halves[(9, 1)]["V"][2, 0, 2, D, :, 1]       # bottom 9, 2 out, 0-2, tied: home r = 0,1,2
check("…and so, late in a tied 9th, the SECOND challenge can be worth more than the first",
      (v_end9[2] - v_end9[1]) > (v_end9[1] - v_end9[0]),
      f"2nd {v_end9[2] - v_end9[1]:.5f} vs 1st {v_end9[1] - v_end9[0]:.5f}")
sim = Simulator(pm, game, {"home": opt, "away": opt}, seed=11).run(12000)
z = (sim["home_win"] - sol.pregame()) / sim["se_home_win"]
check("simulated games win as often as backward induction says", abs(z) < 3.5 and sim["unfinished"] == 0,
      f"{sim['home_win']:.4f} ± {sim['se_home_win']:.4f} vs {sol.pregame():.4f}")

# ------------------------------------------------------------------ leverage
print("\nTango's leverage index")
rm = RunModel()                                          # a toy plate-appearance model
rm.T = np.zeros((2, N_OUTS, N_BASES, N_COUNTS, N_END))
for o in range(N_OUTS):
    for b in range(N_BASES):
        def add(prob, eo, eb, r):
            rm.T[:, o, b, :, _end_index(eo, 0 if eo >= 3 else eb, r)] += prob
        add(0.70, o + 1, b, 0)
        add(0.09, o, [1, 3, 3, 7, 5, 7, 7, 7][b], 1 if b == 7 else 0)
        nb, runs = ((b << 1) | 1) & 7, int(b >= 4)
        add(0.14, o, nb, runs)
        add(0.07, o, 0, bin(b).count("1") + 1)
rm.n_cell = np.ones((2, N_OUTS, N_BASES, N_COUNTS))
rm._solve()
mk = MarkovWP(rm)
wp_like = SimpleNamespace(v=mk.v_table, e=mk.end_value, rm=rm)
grid = pd.DataFrame([(i, h_, o, b, d, True, True) for i in range(1, 10) for h_ in (0, 1)
                     for o in range(3) for b in range(8) for d in range(-3, 4)],
                    columns=["inning", "half", "outs", "bases", "diff", "pa_start", "usable"])
grid["game_pk"] = 0
lev = Leverage.fit(wp_like, grid)
check("LI averages exactly 1 over the plate appearances it was scaled on",
      abs(lev.pa_li.mean() - 1) < 1e-12, f"{lev.pa_li.mean():.12f}")
hi_ = float(lev.li[8, 1, 1, 4, 30])                      # bottom 9, tie, runner on 3rd, 1 out
lo_ = float(lev.li[8, 0, 0, 0, 30 + 9])                  # top 9, home up 9
mid = float(lev.li[0, 0, 0, 0, 30])                      # top 1, tie, bases empty
check("LI: tied bottom 9th, winning run on 3rd ≫ the first pitch of the game ≫ a decided game",
      hi_ > 3 * mid and lo_ < 0.05 * mid, f"{hi_:.2f} / {mid:.2f} / {lo_:.3f}")
check("LI is never negative", np.nanmin(lev.li) >= 0)

print("\nThe Statcast / Tango breakeven")
RE = rm.RE                                               # (2, 3, 8, 12)
fl = flip_runs(RE)
check("RE288 flip: ball four with the bases loaded forces in a run",
      fl[0, 0, 7, 9] > 1.0 - 1e-9, f"{fl[0, 0, 7, 9]:.3f} runs at 3-0 (this toy's counts are all alike)")
check("…and every flip helps the batter (a ball is never worse than a strike)", fl.min() > -1e-12)
be = Breakeven(fl, cost=0.2)
t = be.thresholds(1, 0, "bat", game)[1, 4, 11, D, 1, 1]
check("Statcast's confidence: p* = 0.2 / (0.2 + run value)", abs(t - 0.2 / (0.2 + fl[0, 1, 4, 11])) < 1e-12,
      f"{t:.4f}")
li_one = np.ones((10, 2, N_OUTS, N_BASES, 2 * D + 1))
be1 = Breakeven(fl, cost=0.2, leverage=li_one)
check("with LI = 1 everywhere, the leverage-aware rule is the same rule",
      np.allclose(be1.thresholds(5, 1, "fld", game), be.thresholds(5, 1, "fld", game)))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL MDP CHECKS PASSED")
