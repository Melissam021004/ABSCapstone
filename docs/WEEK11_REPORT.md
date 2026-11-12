# Week 11 report — the challenge decision as an MDP

*Generated 2026-11-11 by `scripts/21_week11_mdp.py`. p(s) 2026.10.29-b81183b4 · V(s) 2026.11.04-824ec8be · cohort 419af01ad9eee03f. Read it with WEEK11.md beside it.*

## 1. The pitch-level game
Every pitch of the season (716,792) sets the odds of what the next pitch does at each count; 112,807 balls in play, 36,366 strikeouts and 16,454 walks/HBP set where the runners end up; 4,746 runner moves between pitches (1.0% of pitches) are in too.

| model | error vs the season (runs) | typical error | RE24 bases empty, 0 out | RE24 bases loaded, 0 out |
|---|---|---|---|---|
| pitch model — as played | -0.0024 | 0.0208 | 0.501 | 2.411 |
| Week 10 run model | -0.0021 | 0.0170 | 0.501 | 2.422 |
| pitch model — umpire calls only | -0.0035 | 0.0211 | 0.500 | 2.407 |
| the season (observed) | +0.0000 | 0.0000 | 0.503 | 2.522 |

Win probability from the same game (as played, nobody challenging) against Week 10's locked V(s), on 250,000 real pitches: mean difference **0.12 pts** (95% of pitches within 0.42), correlation 0.99998, Brier score 0.1596 vs 0.1595.

The 2026 challenges added **+0.0011 runs** per state to the batting team's inning (as played − umpire calls only). Mistakes on offer: **9.62** expected wrong calls a game, 8.64 of them with p(s) ≥ 90%.

## 2. Leverage — Tango's LI from our V(s)
Below 1: **62.7%** of plate appearances (FanGraphs: about 60%). Above 2: **10.2%** (about 10%). Highest: **10.63**.

| inning | average LI (ours) | 2013, Tango's LI (published) | plate appearances |
|---|---|---|---|
| 1 | 0.96 | 0.97 | 20,803 |
| 2 | 0.96 | 0.95 | 20,391 |
| 3 | 0.95 | 0.94 | 20,689 |
| 4 | 0.96 | 0.95 | 20,603 |
| 5 | 0.95 | 0.94 | 20,858 |
| 6 | 0.95 | 0.95 | 20,829 |
| 7 | 0.93 | 0.96 | 20,823 |
| 8 | 0.93 | 0.97 | 21,103 |
| 9 | 1.18 | 1.21 | 15,792 |
| 10+ | 3.37 | 2.44 | 2,347 |

| 9th inning | LI, ours | 2013 published | first PA, ours | first play, 2013 |
|---|---|---|---|---|
| top | 0.90 | 0.92 | 0.79 | 0.87 |
| bot | 1.71 | 1.74 | 1.44 | 1.48 |

## 3. V(s, r) — what a challenge is worth
Home team wins **52.3%** with both teams choosing well (52.3% with no challenges). Solved in 2.2s.

At the first pitch a team's two challenges are worth **5.36 pts** (= 8.7 wins a season): the first **5.02 pts**, the second **0.34 pts**.

Value of holding at the start of each half, tied game, for the team in the field (H1 = the last challenge, H2 = the second):

| start of | H1 (pts) | H2 (pts) | H1 (runs) | H2 (runs) |
|---|---|---|---|---|
| 1 top | 5.04 | 0.34 | 0.570 | 0.038 |
| 1 bot | 4.75 | 0.32 | 0.536 | 0.036 |
| 2 top | 4.75 | 0.31 | 0.508 | 0.034 |
| 2 bot | 4.45 | 0.29 | 0.475 | 0.031 |
| 3 top | 4.43 | 0.29 | 0.446 | 0.029 |
| 3 bot | 4.13 | 0.27 | 0.413 | 0.027 |
| 4 top | 4.09 | 0.26 | 0.384 | 0.024 |
| 4 bot | 3.77 | 0.24 | 0.351 | 0.022 |
| 5 top | 3.70 | 0.23 | 0.321 | 0.020 |
| 5 bot | 3.37 | 0.21 | 0.288 | 0.018 |
| 6 top | 3.27 | 0.20 | 0.258 | 0.015 |
| 6 bot | 2.90 | 0.17 | 0.225 | 0.013 |
| 7 top | 2.75 | 0.16 | 0.195 | 0.011 |
| 7 bot | 2.32 | 0.13 | 0.162 | 0.009 |
| 8 top | 2.12 | 0.12 | 0.131 | 0.007 |
| 8 bot | 1.58 | 0.10 | 0.098 | 0.005 |
| 9 top | 1.26 | 0.09 | 0.066 | 0.003 |
| 9 bot | 0.53 | 0.07 | 0.033 | 0.001 |
| extras top | 1.53 | 0.08 | 0.087 | 0.004 |
| extras bot | 0.45 | 0.04 | 0.042 | 0.002 |

Runs per win implied by the two currencies (innings 1–6): **9.8** (rule of thumb ≈ 10).

## 4. Runs
At the first pitch: last challenge **0.57 runs**, second **0.04 runs**. Statcast prices a failed challenge at 0.2 runs.

## 5. Thresholds
| situation | call | p* (MDP) | Δ swing (pts) | H hold (pts) | LI | RE288 flip (runs) | Statcast breakeven |
|---|---|---|---|---|---|---|---|
| 1st, top, tied, bases empty, 0 out, 0-0 — batter, 2 left | strike | 0.34 | 0.66 | 0.33 | 0.8 | 0.071 | 0.74 |
| 1st, top, tied, bases empty, 0 out, 0-0 — batter, 1 left | strike | 0.88 | 0.66 | 4.96 | 0.8 | 0.071 | 0.74 |
| 1st, bot, home down 8, bases empty, 0 out, 0-0 — batter, 1 left (blowout) | strike | 0.85 | 0.21 | 1.20 | 0.3 | 0.080 | 0.71 |
| 4th, top, tied, bases loaded, 2 out, 3-2 — batter, 1 left | strike | 0.15 | 21.01 | 3.76 | 3.6 | 1.793 | 0.10 |
| 5th, bot, tied, runner on 1st, 1 out, 2-2 — catcher, 1 left | ball | 0.42 | 4.41 | 3.24 | 1.6 | 0.364 | 0.35 |
| 7th, top, away up 1, bases loaded, 2 out, 3-2 — catcher, 1 left | ball | 0.09 | 19.76 | 1.93 | 3.2 | 1.793 | 0.10 |
| 9th, bot, tied, runner on 3rd, 1 out, 3-2 — batter, 1 left | strike | 0.01 | 18.21 | 0.22 | 4.7 | 0.838 | 0.19 |
| 9th, bot, tied, runner on 3rd, 1 out, 3-2 — catcher, 1 left | ball | 0.02 | 18.21 | 0.38 | 4.7 | 0.838 | 0.19 |
| 9th, top, away down 1, bases empty, 2 out, 0-2 — batter, 1 left (strike three ends it) | strike | 0.00 | 2.61 | 0.00 | 1.4 | 0.069 | 0.74 |
| extras, top, tied, runner on 2nd, 0 out, 1-1 — batter, 1 left | strike | 0.30 | 3.25 | 1.41 | 3.1 | 0.128 | 0.61 |
| 8th, top, home up 6, bases empty, 1 out, 1-1 — batter, 1 left | strike | 0.51 | 0.08 | 0.08 | 0.1 | 0.086 | 0.70 |

The same situations, one level at a time (ball minus strike, for the batting team): the batter's chance of not making an out, expected runs to the end of the inning, the chance of scoring at least once more, and win probability.

| situation | at-bat: batter safe | inning: expected runs | inning: scores ≥ 1 | game: win probability | LI |
|---|---|---|---|---|---|
| 1st, top, tied, bases empty, 0 out, 0-0 — batter, 2 left | +10.5 pts | +0.071 runs | +3.2 pts | 0.66 pts | 0.8 |
| 1st, top, tied, bases empty, 0 out, 0-0 — batter, 1 left | +10.5 pts | +0.071 runs | +3.2 pts | 0.66 pts | 0.8 |
| 1st, bot, home down 8, bases empty, 0 out, 0-0 — batter, 1 left (blowout) | +11.2 pts | +0.080 runs | +3.7 pts | 0.21 pts | 0.3 |
| 4th, top, tied, bases loaded, 2 out, 3-2 — batter, 1 left | +100.0 pts | +1.793 runs | +100.0 pts | 21.01 pts | 3.6 |
| 5th, bot, tied, runner on 1st, 1 out, 2-2 — catcher, 1 left | +45.5 pts | +0.364 runs | +16.2 pts | 4.41 pts | 1.6 |
| 7th, top, away up 1, bases loaded, 2 out, 3-2 — catcher, 1 left | +100.0 pts | +1.793 runs | +100.0 pts | 19.76 pts | 3.2 |
| 9th, bot, tied, runner on 3rd, 1 out, 3-2 — batter, 1 left | +100.0 pts | +0.838 runs | +38.3 pts | 18.21 pts | 4.7 |
| 9th, bot, tied, runner on 3rd, 1 out, 3-2 — catcher, 1 left | +100.0 pts | +0.838 runs | +38.3 pts | 18.21 pts | 4.7 |
| 9th, top, away down 1, bases empty, 2 out, 0-2 — batter, 1 left (strike three ends it) | +22.0 pts | +0.069 runs | +4.7 pts | 2.61 pts | 1.4 |
| extras, top, tied, runner on 2nd, 0 out, 1-1 — batter, 1 left | +12.7 pts | +0.128 runs | +3.8 pts | 3.25 pts | 3.1 |
| 8th, top, home up 6, bases empty, 1 out, 1-1 — batter, 1 left | +17.9 pts | +0.086 runs | +4.5 pts | 0.08 pts | 0.1 |

Across 356,889 real opportunities p* falls as leverage rises (correlation with log LI -0.28). Figure w11_01.

| challenges left | LI from | LI to | opportunities | median p* | p* 25th pct | p* 75th pct | Statcast breakeven | breakeven × LI |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.25 | 19,660 | 0.77 | 0.56 | 0.88 | 0.70 | 0.97 |
| 1 | 0.25 | 0.40 | 6,800 | 0.76 | 0.61 | 0.85 | 0.70 | 0.88 |
| 1 | 0.40 | 0.60 | 9,262 | 0.75 | 0.59 | 0.85 | 0.71 | 0.83 |
| 1 | 0.60 | 0.80 | 7,717 | 0.72 | 0.57 | 0.83 | 0.70 | 0.76 |
| 1 | 0.80 | 1.00 | 7,349 | 0.72 | 0.57 | 0.81 | 0.67 | 0.69 |
| 1 | 1.00 | 1.30 | 7,948 | 0.67 | 0.52 | 0.78 | 0.66 | 0.63 |
| 1 | 1.30 | 1.70 | 7,513 | 0.60 | 0.42 | 0.73 | 0.64 | 0.55 |
| 1 | 1.70 | 2.20 | 5,928 | 0.51 | 0.31 | 0.66 | 0.60 | 0.45 |
| 1 | 2.20 | 3.00 | 4,468 | 0.41 | 0.25 | 0.57 | 0.56 | 0.34 |
| 1 | 3.00 | 4.50 | 3,609 | 0.30 | 0.18 | 0.42 | 0.53 | 0.24 |
| 1 | 4.50 | 7.00 | 1,546 | 0.17 | 0.10 | 0.24 | 0.50 | 0.16 |
| 1 | 7.00 | 20.00 | 166 | 0.11 | 0.07 | 0.14 | 0.46 | 0.10 |
| 2 | 0.00 | 0.25 | 36,199 | 0.17 | 0.08 | 0.31 | 0.70 | 0.96 |
| 2 | 0.25 | 0.40 | 19,427 | 0.21 | 0.11 | 0.35 | 0.72 | 0.89 |
| 2 | 0.40 | 0.60 | 38,055 | 0.25 | 0.14 | 0.38 | 0.74 | 0.86 |
| 2 | 0.60 | 0.80 | 37,320 | 0.22 | 0.13 | 0.32 | 0.70 | 0.77 |
| 2 | 0.80 | 1.00 | 45,295 | 0.20 | 0.12 | 0.29 | 0.67 | 0.70 |
| 2 | 1.00 | 1.30 | 34,223 | 0.15 | 0.09 | 0.23 | 0.66 | 0.62 |
| 2 | 1.30 | 1.70 | 28,900 | 0.12 | 0.07 | 0.20 | 0.62 | 0.53 |
| 2 | 1.70 | 2.20 | 19,717 | 0.10 | 0.05 | 0.16 | 0.55 | 0.39 |
| 2 | 2.20 | 3.00 | 9,667 | 0.05 | 0.03 | 0.09 | 0.51 | 0.30 |
| 2 | 3.00 | 4.50 | 4,686 | 0.03 | 0.01 | 0.05 | 0.50 | 0.22 |
| 2 | 4.50 | 7.00 | 1,307 | 0.01 | 0.01 | 0.02 | 0.49 | 0.16 |
| 2 | 7.00 | 20.00 | 127 | 0.00 | 0.00 | 0.01 | 0.46 | 0.09 |

## 6. Run production at three levels
Average flip (ball minus strike) at each level, by game situation:

| context | opportunities | at-bat: batter safe | inning: expected runs | inning: scores ≥ 1 | game: win probability | average LI |
|---|---|---|---|---|---|---|
| innings 1–6 | 247,123 | +22.3 pts | +0.147 | +6.5 pts | 1.27 pts | 0.94 |
| 7th or later, within 1 run | 35,703 | +22.4 pts | +0.154 | +6.6 pts | 2.69 pts | 2.10 |
| 7th or later, 2+ runs apart | 74,065 | +22.4 pts | +0.151 | +6.6 pts | 0.83 pts | 0.58 |
| bottom 9th or later, tied | 2,119 | +22.4 pts | +0.179 | +7.2 pts | 3.50 pts | 2.94 |
| all | 356,891 | +22.3 pts | +0.149 | +6.5 pts | 1.32 pts | 0.98 |

How much of the win-probability swing each run measure explains (R²), alone and × LI:

| context | batter safe | batter safe × LI | expected runs | expected runs × LI | scores ≥ 1 | scores ≥ 1 × LI |
|---|---|---|---|---|---|---|
| innings 1–6 | 0.44 | 0.87 | 0.73 | 0.80 | 0.67 | 0.74 |
| 7th or later, within 1 run | 0.39 | 0.81 | 0.68 | 0.81 | 0.67 | 0.77 |
| 7th or later, 2+ runs apart | 0.16 | 0.93 | 0.30 | 0.77 | 0.24 | 0.70 |
| bottom 9th or later, tied | 0.47 | 0.69 | 0.83 | 0.83 | 0.95 | 0.85 |
| all | 0.30 | 0.87 | 0.53 | 0.81 | 0.48 | 0.76 |

One run of call value × LI ≈ **10.47 pts** of win probability in innings 1–6 (1 win ≈ 9.6 runs).

RE288 run value of flipping a call, ours vs FanGraphs (Clemens, April 2026):

| runners | outs | count | ours | FanGraphs (published) |
|---|---|---|---|---|
| 3B | 1 | 0-0 | 0.14 | 0.12 |
| 3B | 1 | 1-0 | 0.08 | 0.15 |
| 3B | 1 | 2-0 | 0.12 | 0.20 |
| 3B | 1 | 3-0 | 0.18 | 0.12 |
| 3B | 1 | 3-2 | 0.81 | 0.84 |
| empty | 2 | 3-2 | 0.23 | 0.23 |

## 7. Challenge rules compared
| policy | wins lost / 162 | challenges / game | overturned | overturns / game | wrong calls left standing / game | challenges left at the end |
|---|---|---|---|---|---|---|
| optimal (this MDP) | 0.00 | 5.24 | 90% | 4.69 | 0.20 | 1.46 |
| play for runs (run-differential MDP) | 0.01 | 5.28 | 89% | 4.71 | 0.17 | 1.44 |
| Statcast breakeven × LI | 0.26 | 4.57 | 98% | 4.48 | 0.42 | 1.91 |
| Statcast breakeven (0.2 runs) | 0.31 | 4.66 | 98% | 4.57 | 0.34 | 1.92 |
| challenge when p(s) ≥ 0.5 | 0.29 | 4.76 | 98% | 4.64 | 0.26 | 1.89 |
| never challenge | 9.03 | 0.00 | — | 0.00 | 4.85 | 2.00 |

2026 as played: 1.63 challenges a team-game, 56% overturned, 0.91 overturns of 4.64 expected wrong calls. Simulation check: z = -0.2.

## 8. The 2026 decisions (preview of RQ3)
356,891 opportunities; teams challenged 7,897, the thresholds say 24,100, both 4,562. Of the challenges teams made, **57.8%** cleared the bar; of the chances above the bar, teams took **18.9%**. Below-the-bar challenges: 3,335 (12.8 wins). Passed-up chances above it: 19,538 (185.5 wins). Calls with p(s) ≥ 90% left alone: 15,995. Total gap to the full-information benchmark: **198.3 wins** (6.61 per team). Where either rule would challenge, Statcast's breakeven and the thresholds agree 87.6% of the time (Statcast only 149; thresholds only 2,855).

| group | opportunities | challenged | above the bar | challenges above the bar | above-bar chances taken | gap (wins) | p ≥ 90% left alone |
|---|---|---|---|---|---|---|---|
| batters' side (called strikes) | 110,813 | 3,461 | 13,102 | 59.3% | 15.7% | 109.6 | 9,355 |
| fielders' side (called balls) | 246,078 | 4,436 | 10,998 | 56.6% | 22.8% | 88.7 | 6,640 |
| challenges by batters | — | 3,460 | — | 59.3% | — | 6.0 | — |
| challenges by catchers | — | 4,278 | — | 57.4% | — | 6.3 | — |
| challenges by pitchers | — | 132 | — | 41.7% | — | 0.4 | — |
| cohort split: test | 52,727 | 1,243 | 3,569 | 58.3% | 20.3% | 27.8 | 2,304 |
| cohort split: train | 242,518 | 5,219 | 16,343 | 57.5% | 18.4% | 135.1 | 10,917 |
| cohort split: validate | 61,646 | 1,435 | 4,188 | 58.4% | 20.0% | 35.5 | 2,774 |

## 9. Umpire tiers
| plate umpire | umpires | games | expected wrong calls / game | first challenge (pts) | second challenge (pts) | bar, 1st inning 0-0, 1 left |
|---|---|---|---|---|---|---|
| most accurate third | 30 | 806 | 8.55 | 4.45 | 0.32 | 0.87 |
| middle third | 29 | 813 | 9.63 | 5.00 | 0.33 | 0.88 |
| least accurate third | 30 | 800 | 10.67 | 5.61 | 0.36 | 0.89 |

## 10. What Spring adds
- **Who is batting and pitching.** V(s, r) is an average-team model, like every published win expectancy table. Spring conditions the pitch model on the batter, pitcher and catcher (a 2-strike count means more against a contact hitter).
- **What players can see.** p(s) is what Hawk-Eye knows; a catcher has two seconds and a view. The optimal policy here is the full-information benchmark. Spring adds a perception model (p seen with noise, by role and by distance from the zone) so the thresholds become rules a player can use.
- **The September test.** Evaluate the policies on the held-out month (RQ3) with p(s) from the lock, and the regret by team with game-level bootstrap intervals.
- **Umpire-specific supply of mistakes** for the game in progress (we only compare thirds here), and a calibrated p(s) (the lock is uncalibrated; thresholds are only as good as the probabilities fed to them).
- **Score cap and extras.** The lead is capped at ±10 (a 4·10⁻⁶ effect on win probability) and every extra inning is treated alike; position players pitching are left out of the decision.

Figures: `figures/week11/w11_01_thresholds.png` (the money figure), `w11_02_hold_value`, `w11_03_leverage`, `w11_04_run_levels`, `w11_05_policies`, `w11_06_real_decisions`.
