# Week 10 — Win Probability Component

Generated 2026-11-04 09:14 by `scripts/19_week10_winprob.py` · seasons 2026 · 2,428 games · 5-fold out-of-game scoring · 500 game-level bootstrap resamples

> Read this next to `WEEK10.md`, which explains every section.

## The locked V(s)

**V(s) 2026.11.04-824ec8be = structural (Markov) model + uncalibrated** · input: inning, half, outs, runners, count, score

- Out-of-fold (every game scored by a model that never saw it): Brier 0.1585 [0.1534, 0.1629], calibration intercept -0.03 [-0.13, +0.06], slope 1.05 [0.99, 1.12].
- The boosted model (the plan's model version): Δ Brier vs structural +0.00056 [+0.00012, +0.00099] — **worse**.
- Published sanity values: **11 of 11** inside their ranges (the boosted model: 6 of 11).
- Pre-game home win probability 0.530; home teams actually won 52.9%.

## 0. Pre-flight

- 716,792 pitches, 184,288 plate appearances, 2,429 games. Usable for win probability: 2,428 (a winner and 9+ innings); left out 1: 1 shorter than 9 innings (shortened game).
- Run environment: 4.48 runs per team per game; home teams won 52.9%; 8.6% of games went to extra innings.
- Called pitches: the recorded call matches what the count did next on 100.00%. Transitions use what the count did (the *effective* call), so they are right whichever call Statcast recorded.

## 1. The state

| part | values |
|---|---|
| inning | 1–9, and extras (10+) as one repeating inning |
| half | top (away bats), bottom (home bats) |
| outs | 0, 1, 2 |
| runners | 8: empty, 1B, 2B, 1B 2B, 3B, 1B 3B, 2B 3B, loaded |
| count | 12: 0-0 … 3-2 |
| home lead | −30 … +30 |

10 × 2 × 3 × 8 × 12 × 61 = **351,360 states** in V(s). The empirical table uses the home lead clipped to −10…+10: 10,080 cells at the start of a plate appearance, 120,960 with the count.

## 2. Runs — the inning-level view

The structural model's engine: how plate appearances move outs, runners and runs, learned from 646,346 pitches in innings 1–8, then chained to the third out. Expected runs from each state to the end of the inning (RE24, count 0-0):

| runners | 0 out | 1 out | 2 out |
|---|---:|---:|---:|
| empty | 0.50 | 0.27 | 0.10 |
| 1B | 0.90 | 0.54 | 0.23 |
| 2B | 1.14 | 0.68 | 0.33 |
| 1B 2B | 1.55 | 0.93 | 0.46 |
| 3B | 1.37 | 0.95 | 0.38 |
| 1B 3B | 1.77 | 1.19 | 0.51 |
| 2B 3B | 2.01 | 1.41 | 0.54 |
| loaded | 2.42 | 1.66 | 0.77 |

Chance of scoring at least once more this inning:

| runners | 0 out | 1 out | 2 out |
|---|---:|---:|---:|
| empty | 27% | 16% | 7% |
| 1B | 42% | 27% | 13% |
| 2B | 62% | 40% | 22% |
| 1B 2B | 64% | 41% | 23% |
| 3B | 82% | 66% | 27% |
| 1B 3B | 85% | 65% | 27% |
| 2B 3B | 85% | 68% | 24% |
| loaded | 88% | 69% | 32% |

- Against the plain averages (mean runs to the end of the inning from each state): weighted mean |difference| 0.005 runs for RE24 and 0.012 for RE288 (cells with 200+ pitches). Small = the chain reproduces the data.
- Against Statcast's own run value of each pitch (`delta_run_exp`): correlation 0.851, mean |difference| 0.054 runs over 644,931 pitches.

Run expectancy by count, bases empty (the count is where a challenge acts):

| outs | 0-0 | 0-1 | 0-2 | 1-0 | 1-1 | 1-2 | 2-0 | 2-1 | 2-2 | 3-0 | 3-1 | 3-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.501 | 0.463 | 0.416 | 0.539 | 0.491 | 0.438 | 0.608 | 0.545 | 0.473 | 0.742 | 0.673 | 0.578 |
| 1 | 0.269 | 0.241 | 0.204 | 0.297 | 0.262 | 0.217 | 0.349 | 0.300 | 0.252 | 0.438 | 0.385 | 0.324 |
| 2 | 0.102 | 0.085 | 0.064 | 0.118 | 0.096 | 0.073 | 0.143 | 0.116 | 0.089 | 0.184 | 0.157 | 0.121 |

Full tables: `docs/week10_run_expectancy.csv` (by batting side, every outs × runners × count, with the chance of scoring and the number of pitches).

## 3. Empirical win expectancy (the plan's step 2)

- 6,855 of 10,080 cells ever happened. 421 have 100+ games and hold 51.3% of all plate appearances.
- Thin cells (< 30 games): 5,527, holding 22.3% of plate appearances. With the count, 54,505 of 59,296 cells are thin — the reason the count comes from the structural model, not from the table.
- Counts are of games, not plate appearances: rows from one game share one outcome.
- Tables: `docs/week10_empirical_we.csv`, `docs/week10_empirical_we_by_count.csv`.

## 4. Models (the plan's step 3)

Both models are scored only on games they never saw: the games are split into 5 folds; each fold is predicted by models fitted on the other four.

- **Structural (Markov)** — plate-appearance transitions (smoothed toward a structural prior where thin; home and away separately), chained to runs per inning, worked back from the end of the game with the 2026 rules (no bottom 9th when the home team leads, walk-offs, the automatic runner in extras). Never sees who won.
- **Boosted** — lightgbm, one fixed setting, monotone in the home lead, early stopping on the last 15% of training games (177–220 trees across folds). Learns from who won.
- Calibration of the structural model, cross-fitted Week 9's way:

| method | Brier − best [95%] | |
|---|---|---|
| uncalibrated | +0.00004 [-0.00024, +0.00030] | ★ chosen |
| Platt scaling | +0.00000 [+0.00000, +0.00000] |  |
| isotonic | +0.00012 [-0.00004, +0.00028] |  |

## 5. Validation (the plan's step 4)

### 5.1 Calibration, out of fold — Week 9's metrics

| model | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | ECE | AUC |
|---|---|---:|---|---|---:|---:|
| Structural (Markov) | 0.1585 [0.1534, 0.1629] | 0.4714 | -0.03 [-0.13, +0.06] | 1.05 [0.99, 1.12] | 0.011 | 0.848 |
| Boosted | 0.1590 [0.1537, 0.1639] | 0.4738 | +0.00 [-0.10, +0.09] | 0.98 [0.92, 1.04] | 0.006 | 0.846 |
| Base rate (home team always) | 0.2497 [0.2491, 0.2504] | 0.6926 | — | — | 0.000 | 0.500 |
| MLB — Statcast home_win_exp | 0.1586 [0.1534, 0.1633] | 0.4718 | +0.06 [-0.04, +0.15] | 1.07 [1.00, 1.14] | 0.015 | 0.848 |

*MLB's rows are MLB's own published win probability on the same plate appearances. If a value was recorded after the plate appearance rather than before it, it knows something V(s) doesn't, so a lower Brier for it is not a fair loss — compare the calibration columns instead.*

The base rate (always the home team's season win rate) is the floor: a win-probability model's Brier is mostly the noise of baseball, so the gaps between real models are small. Read the intercept and slope: those are what a decision threshold feels.

### 5.2 By inning and by lead

| group | slice | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| inning | 1–3 | 61,883 | 53.8% | 52.8% [52.4%, 53.2%] |
| inning | 4–6 | 62,290 | 53.6% | 53.0% [52.6%, 53.4%] |
| inning | 7–9 | 57,718 | 49.0% | 49.1% [48.7%, 49.5%] |
| inning | extras | 2,347 | 44.8% | 42.2% [40.2%, 44.2%] |
| home lead (abs.) | tied | 48,347 | 55.5% | 55.6% [55.1%, 56.0%] |
| home lead (abs.) | 1–2 | 73,940 | 53.1% | 52.0% [51.7%, 52.4%] |
| home lead (abs.) | 3–4 | 34,987 | 50.1% | 49.8% [49.3%, 50.4%] |
| home lead (abs.) | 5+ | 26,964 | 45.8% | 45.5% [45.0%, 46.1%] |

### 5.3 Agreement with the empirical table

- **Structural (Markov)**: 421 cells with 100+ games (51.3% of plate appearances); mean |model − table| 2.18 points; 4.3% of cells beyond 2 standard errors (about 5% expected if the model is right).
- **Boosted**: 421 cells with 100+ games (51.3% of plate appearances); mean |model − table| 1.63 points; 1.0% of cells beyond 2 standard errors (about 5% expected if the model is right).

Largest gaps, Structural (Markov) (z = gap in standard errors):

| inning | half | outs | runners | home lead | games | table | model | z |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 2 | bot | 1 | empty | -2 | 147 | 22.4% | 33.7% | -2.9 |
| 2 | bot | 0 | empty | -2 | 203 | 26.6% | 36.2% | -2.9 |
| 6 | top | 2 | empty | +2 | 145 | 92.4% | 84.6% | +2.6 |
| 6 | top | 0 | empty | +4 | 133 | 98.5% | 92.8% | +2.5 |
| 1 | bot | 0 | empty | -2 | 164 | 28.0% | 37.5% | -2.5 |
| 2 | top | 0 | empty | -2 | 138 | 22.5% | 32.4% | -2.5 |
| 3 | top | 0 | empty | -2 | 198 | 22.7% | 30.7% | -2.4 |
| 3 | bot | 2 | 1B | -1 | 119 | 31.9% | 42.7% | -2.4 |

Largest gaps, Boosted (z = gap in standard errors):

| inning | half | outs | runners | home lead | games | table | model | z |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 9 | top | 2 | empty | +1 | 146 | 98.6% | 91.1% | +3.2 |
| 9 | top | 1 | empty | +1 | 174 | 94.3% | 88.3% | +2.5 |
| 9 | bot | 2 | empty | -2 | 122 | 1.6% | 6.1% | -2.1 |
| 9 | top | 2 | empty | +2 | 139 | 99.3% | 95.8% | +2.0 |
| 3 | top | 1 | 1B | +0 | 227 | 57.3% | 50.6% | +2.0 |
| 9 | bot | 2 | empty | -1 | 123 | 5.7% | 11.4% | -2.0 |
| 8 | bot | 1 | empty | -2 | 158 | 7.6% | 12.7% | -1.9 |
| 5 | top | 1 | empty | +1 | 250 | 74.0% | 68.6% | +1.8 |

### 5.4 Published sanity values

| state | V(s) | published range | boosted | |
|---|---:|---|---:|---|
| Start of the game (top 1st, 0 out, bases empty, tied) | 53.0% | 51%–57% | 51.7% | ok |
| Bottom 9th, tied, runner on 3rd, 1 out — the plan's check | 83.6% | 78%–89% | 72.0% ✗ | ok |
| Bottom 9th, tied, bases empty, 0 out | 64.7% | 60%–71% | 59.7% ✗ | ok |
| Bottom 9th, tied, bases loaded, 0 out | 94.4% | 89%–97% | 88.7% ✗ | ok |
| Top 9th, home up 1, 0 out, bases empty | 84.5% | 79%–90% | 83.9% | ok |
| Top 9th, home up 3, 0 out, bases empty | 96.9% | 93%–99% | 97.6% | ok |
| Bottom 9th, home down 1, 2 out, bases empty | 4.7% | 2%–10% | 11.4% ✗ | ok |
| Bottom 9th, home down 1, 2 out, runner on 2nd | 15.0% | 8%–21% | 15.4% | ok |
| Top 10th, tied, automatic runner on 2nd, 0 out | 50.9% | 42%–60% | 48.6% | ok |
| Top 5th, tied, 0 out, bases empty | 52.2% | 50%–57% | 49.0% ✗ | ok |
| Bottom 8th, home up 2, 0 out, bases empty | 94.4% | 86%–96% | 93.6% | ok |

Ranges from the published win-expectancy tables (Tango; FanGraphs) for a modern run environment, kept wide on purpose: they catch a broken model, not a run environment that differs a little from the 2010s. ✗ = a boosted value outside its range — usually a rare late-game state, where a model that learns only from game outcomes has few games to learn from.

## 6. The locked bundle (`models/winprob/`)

- version **2026.11.04-824ec8be**, created 2026-11-04T09:14:27
- structural model fitted on 646,346 pitches (innings 1–8) from 2,428 games; calibration: uncalibrated
- self-check: reloading through `src/abs_winprob.py` reproduces every plate-appearance start to within 3.0e-08
- every file hashed in `manifest.json`; `load_winprob()` refuses a bundle that changed

```python
from src.abs_winprob import load_winprob
wp = load_winprob()
wp.home({'inning': 9, 'half': 'bot', 'outs': 1, 'on_3b': True, 'home_score': 3, 'away_score': 3})
wp.call_value({'inning': 7, 'half': 'top', 'outs': 2, 'home_score': 2, 'away_score': 2,
               'balls': 1, 'strikes': 2}, call='strike')
```

## 7. Call transitions — stands vs overturned (the plan's step 5)

- From 336,262 called pitches, every (outs × runners × count × call) cell: where the game went next. 98.6% went exactly where the rulebook says; the rest had a steal, a wild pitch or a runner thrown out on the same pitch — which is why the table comes from the data.
- A challenge's two branches: **stands** = the umpire's call; **overturned** = the other call from the same state. `docs/week10_call_transitions.csv` (one row per cell, with n) and `docs/week10_call_transitions_long.csv` (every next state with its probability).

What overturning a called strike three is worth (bases empty, 1-2 count), by situation:

| situation | outs | win prob. if it stands | if overturned | gain for the batting team | runs |
|---|---:|---:|---:|---:|---:|
| tied, 3rd inning, top | 0 | 45.0% | 47.1% | +2.2 pts | +0.200 |
| tied, 3rd inning, top | 1 | 43.2% | 44.7% | +1.6 pts | +0.140 |
| tied, 3rd inning, top | 2 | 42.0% | 43.0% | +1.0 pts | +0.084 |
| tied, 6th inning, bottom | 0 | 56.1% | 58.8% | +2.8 pts | +0.207 |
| tied, 6th inning, bottom | 1 | 53.5% | 55.8% | +2.4 pts | +0.161 |
| tied, 6th inning, bottom | 2 | 51.8% | 53.3% | +1.5 pts | +0.094 |
| tied, 9th inning, bottom | 0 | 59.1% | 63.6% | +4.6 pts | +0.207 |
| tied, 9th inning, bottom | 1 | 54.3% | 58.6% | +4.3 pts | +0.161 |
| tied, 9th inning, bottom | 2 | 50.9% | 53.9% | +3.0 pts | +0.094 |
| home up 1, 9th inning, top | 0 | 8.7% | 14.9% | +6.1 pts | +0.200 |
| home up 1, 9th inning, top | 1 | 3.5% | 8.0% | +4.5 pts | +0.140 |
| home up 1, 9th inning, top | 2 | 0.0% | 2.9% | +2.9 pts | +0.084 |
| home up 4, 7th inning, bottom | 0 | 97.2% | 97.4% | +0.2 pts | +0.207 |
| home up 4, 7th inning, bottom | 1 | 97.0% | 97.2% | +0.2 pts | +0.161 |
| home up 4, 7th inning, bottom | 2 | 96.8% | 97.0% | +0.1 pts | +0.094 |
| home down 6, 5th inning, top | 0 | 95.6% | 95.9% | +0.3 pts | +0.200 |
| home down 6, 5th inning, top | 1 | 95.4% | 95.6% | +0.2 pts | +0.140 |
| home down 6, 5th inning, top | 2 | 95.2% | 95.3% | +0.1 pts | +0.084 |

The same call is worth very different amounts — that is RQ2 in one table. Every situation × count × call is in `docs/week10_call_value_examples.csv`; `scripts/20_week10_challenge_value.py` values every 2026 challenge with it.

## 8. Figures (`figures/week10/`)

- `w10_01_runs.png` — run expectancy and the chance of scoring, by runners and outs
- `w10_02_wp_curves.png` — home win probability through the game, by score
- `w10_03_calibration.png` — reliability, out of fold: structural vs boosted
- `w10_04_agreement.png` — model vs empirical table, cell by cell
- `w10_05_call_value.png` — what overturning strike three is worth, inning by inning
- `w10_06_count_value.png` — runs an overturn is worth, by count and outs