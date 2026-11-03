# Week 10 — Win Probability Component

Generated 2026-10-30 21:41 by `scripts/19_week10_winprob.py` · seasons 2026 · 2,430 games · 5-fold out-of-game scoring · 500 game-level bootstrap resamples

> **DEMO DATA** — a simulated season. The method is real; the numbers are not.

> Read this next to `WEEK10.md`, which explains every section.

## The locked V(s)

**V(s) 2026.10.30-d4a59c02 = structural (Markov) model + uncalibrated** · input: inning, half, outs, runners, count, score

- Out-of-fold (every game scored by a model that never saw it): Brier 0.1624 [0.1576, 0.1672], calibration intercept +0.03 [-0.06, +0.11], slope 1.02 [0.96, 1.09].
- The boosted model (the plan's model version): Δ Brier vs structural +0.00097 [+0.00054, +0.00136] — **worse**.
- Published sanity values: **11 of 11** inside their ranges (the boosted model: 8 of 11).
- Pre-game home win probability 0.524; home teams actually won 52.9%.

## 0. Pre-flight

- 716,763 pitches, 186,686 plate appearances, 2,430 games. Usable for win probability: 2,430 (a winner and 9+ innings); none left out.
- Run environment: 4.50 runs per team per game; home teams won 52.9%; 10.7% of games went to extra innings.
- Called pitches: the recorded call matches what the count did next on 98.87%. Transitions use what the count did (the *effective* call), so they are right whichever call Statcast recorded.

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

The structural model's engine: how plate appearances move outs, runners and runs, learned from 644,577 pitches in innings 1–8, then chained to the third out. Expected runs from each state to the end of the inning (RE24, count 0-0):

| runners | 0 out | 1 out | 2 out |
|---|---:|---:|---:|
| empty | 0.50 | 0.27 | 0.10 |
| 1B | 0.88 | 0.53 | 0.22 |
| 2B | 1.08 | 0.67 | 0.31 |
| 1B 2B | 1.46 | 0.94 | 0.46 |
| 3B | 1.29 | 0.93 | 0.37 |
| 1B 3B | 1.66 | 1.11 | 0.47 |
| 2B 3B | 1.87 | 1.33 | 0.56 |
| loaded | 2.32 | 1.65 | 0.79 |

Chance of scoring at least once more this inning:

| runners | 0 out | 1 out | 2 out |
|---|---:|---:|---:|
| empty | 27% | 16% | 7% |
| 1B | 41% | 27% | 12% |
| 2B | 58% | 39% | 21% |
| 1B 2B | 60% | 42% | 23% |
| 3B | 81% | 65% | 26% |
| 1B 3B | 79% | 58% | 26% |
| 2B 3B | 85% | 67% | 25% |
| loaded | 86% | 69% | 33% |

- Against the plain averages (mean runs to the end of the inning from each state): weighted mean |difference| 0.006 runs for RE24 and 0.011 for RE288 (cells with 200+ pitches). Small = the chain reproduces the data.

Run expectancy by count, bases empty (the count is where a challenge acts):

| outs | 0-0 | 0-1 | 0-2 | 1-0 | 1-1 | 1-2 | 2-0 | 2-1 | 2-2 | 3-0 | 3-1 | 3-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.497 | 0.454 | 0.408 | 0.535 | 0.484 | 0.433 | 0.594 | 0.534 | 0.467 | 0.720 | 0.653 | 0.572 |
| 1 | 0.273 | 0.244 | 0.206 | 0.299 | 0.262 | 0.224 | 0.342 | 0.301 | 0.253 | 0.422 | 0.380 | 0.325 |
| 2 | 0.105 | 0.088 | 0.064 | 0.117 | 0.098 | 0.072 | 0.137 | 0.115 | 0.090 | 0.174 | 0.150 | 0.119 |

Full tables: `docs/week10_run_expectancy.csv` (by batting side, every outs × runners × count, with the chance of scoring and the number of pitches).

## 3. Empirical win expectancy (the plan's step 2)

- 6,846 of 10,080 cells ever happened. 436 have 100+ games and hold 52.4% of all plate appearances.
- Thin cells (< 30 games): 5,485, holding 20.9% of plate appearances. With the count, 54,025 of 58,846 cells are thin — the reason the count comes from the structural model, not from the table.
- Counts are of games, not plate appearances: rows from one game share one outcome.
- Tables: `docs/week10_empirical_we.csv`, `docs/week10_empirical_we_by_count.csv`.

## 4. Models (the plan's step 3)

Both models are scored only on games they never saw: the games are split into 5 folds; each fold is predicted by models fitted on the other four.

- **Structural (Markov)** — plate-appearance transitions (smoothed toward a structural prior where thin; home and away separately), chained to runs per inning, worked back from the end of the game with the 2026 rules (no bottom 9th when the home team leads, walk-offs, the automatic runner in extras). Never sees who won.
- **Boosted** — hgb, one fixed setting, monotone in the home lead, early stopping on the last 15% of training games (225–450 trees across folds). Learns from who won.
- Calibration of the structural model, cross-fitted Week 9's way:

| method | Brier − best [95%] | |
|---|---|---|
| uncalibrated | +0.00000 [+0.00000, +0.00000] | ★ chosen |
| Platt scaling | +0.00041 [+0.00010, +0.00073] |  |
| isotonic | +0.00057 [+0.00024, +0.00089] |  |

## 5. Validation (the plan's step 4)

### 5.1 Calibration, out of fold — Week 9's metrics

| model | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | ECE | AUC |
|---|---|---:|---|---|---:|---:|
| Structural (Markov) | 0.1624 [0.1576, 0.1672] | 0.4833 | +0.03 [-0.06, +0.11] | 1.02 [0.96, 1.09] | 0.006 | 0.840 |
| Boosted | 0.1634 [0.1581, 0.1683] | 0.4869 | +0.00 [-0.10, +0.09] | 0.95 [0.90, 1.01] | 0.008 | 0.838 |
| Base rate (home team always) | 0.2498 [0.2493, 0.2504] | 0.6927 | — | — | 0.000 | 0.500 |

*demo data: MLB's win-probability columns are placeholders, so no comparison.*

The base rate (always the home team's season win rate) is the floor: a win-probability model's Brier is mostly the noise of baseball, so the gaps between real models are small. Read the intercept and slope: those are what a decision threshold feels.

### 5.2 By inning and by lead

| group | slice | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| inning | 1–3 | 63,071 | 52.1% | 52.7% [52.4%, 53.1%] |
| inning | 4–6 | 63,010 | 52.2% | 53.0% [52.6%, 53.4%] |
| inning | 7–9 | 57,903 | 48.7% | 48.9% [48.5%, 49.3%] |
| inning | extras | 2,702 | 42.1% | 40.9% [39.1%, 42.8%] |
| home lead (abs.) | tied | 49,278 | 55.0% | 55.7% [55.3%, 56.1%] |
| home lead (abs.) | 1–2 | 76,716 | 52.0% | 52.7% [52.4%, 53.1%] |
| home lead (abs.) | 3–4 | 35,549 | 48.6% | 48.6% [48.1%, 49.2%] |
| home lead (abs.) | 5+ | 25,143 | 43.2% | 43.4% [42.7%, 44.0%] |

### 5.3 Agreement with the empirical table

- **Structural (Markov)**: 436 cells with 100+ games (52.4% of plate appearances); mean |model − table| 1.86 points; 1.8% of cells beyond 2 standard errors (about 5% expected if the model is right).
- **Boosted**: 436 cells with 100+ games (52.4% of plate appearances); mean |model − table| 1.49 points; 0.9% of cells beyond 2 standard errors (about 5% expected if the model is right).

Largest gaps, Structural (Markov) (z = gap in standard errors):

| inning | half | outs | runners | home lead | games | table | model | z |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 5 | bot | 2 | 1B | -1 | 103 | 54.4% | 38.6% | +3.3 |
| 4 | bot | 1 | empty | +1 | 251 | 77.3% | 68.3% | +3.1 |
| 4 | top | 1 | empty | +3 | 121 | 76.9% | 85.1% | -2.5 |
| 8 | bot | 0 | empty | +4 | 130 | 96.9% | 99.0% | -2.3 |
| 3 | bot | 2 | 1B | +0 | 195 | 62.1% | 54.4% | +2.1 |
| 8 | top | 1 | empty | +2 | 194 | 94.8% | 90.5% | +2.1 |
| 3 | top | 2 | 1B | +0 | 229 | 61.6% | 55.0% | +2.0 |
| 6 | bot | 1 | empty | +2 | 154 | 90.3% | 84.4% | +2.0 |

Largest gaps, Boosted (z = gap in standard errors):

| inning | half | outs | runners | home lead | games | table | model | z |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 5 | bot | 2 | 1B | -1 | 103 | 54.4% | 41.5% | +2.7 |
| 9 | top | 2 | empty | +1 | 130 | 99.2% | 94.3% | +2.4 |
| 2 | bot | 1 | 1B 2B | +0 | 102 | 52.0% | 62.3% | -2.1 |
| 2 | top | 0 | empty | +2 | 135 | 67.4% | 74.9% | -2.0 |
| 8 | top | 0 | empty | -1 | 251 | 27.5% | 22.3% | +2.0 |
| 9 | bot | 2 | empty | -1 | 128 | 3.9% | 8.8% | -2.0 |
| 1 | top | 2 | empty | -1 | 114 | 52.6% | 43.6% | +2.0 |
| 1 | top | 1 | 2B | +0 | 176 | 55.7% | 48.7% | +1.9 |

### 5.4 Published sanity values

| state | V(s) | published range | boosted | |
|---|---:|---|---:|---|
| Start of the game (top 1st, 0 out, bases empty, tied) | 52.4% | 51%–57% | 53.2% | ok |
| Bottom 9th, tied, runner on 3rd, 1 out — the plan's check | 82.7% | 78%–89% | 72.8% ✗ | ok |
| Bottom 9th, tied, bases empty, 0 out | 63.9% | 60%–71% | 62.9% | ok |
| Bottom 9th, tied, bases loaded, 0 out | 93.0% | 89%–97% | 81.2% ✗ | ok |
| Top 9th, home up 1, 0 out, bases empty | 84.3% | 79%–90% | 89.3% | ok |
| Top 9th, home up 3, 0 out, bases empty | 96.9% | 93%–99% | 96.6% | ok |
| Bottom 9th, home down 1, 2 out, bases empty | 4.7% | 2%–10% | 10.8% ✗ | ok |
| Bottom 9th, home down 1, 2 out, runner on 2nd | 14.0% | 8%–21% | 14.9% | ok |
| Top 10th, tied, automatic runner on 2nd, 0 out | 50.1% | 42%–60% | 49.0% | ok |
| Top 5th, tied, 0 out, bases empty | 51.8% | 50%–57% | 51.6% | ok |
| Bottom 8th, home up 2, 0 out, bases empty | 94.3% | 86%–96% | 95.2% | ok |

Ranges from the published win-expectancy tables (Tango; FanGraphs) for a modern run environment, kept wide on purpose: they catch a broken model, not a run environment that differs a little from the 2010s. ✗ = a boosted value outside its range — usually a rare late-game state, where a model that learns only from game outcomes has few games to learn from.

## 6. The locked bundle (`models/winprob/`)

- version **2026.10.30-d4a59c02**, created 2026-10-30T21:41:31
- structural model fitted on 644,577 pitches (innings 1–8) from 2,430 games; calibration: uncalibrated
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

- From 335,281 called pitches, every (outs × runners × count × call) cell: where the game went next. 99.1% went exactly where the rulebook says; the rest had a steal, a wild pitch or a runner thrown out on the same pitch — which is why the table comes from the data.
- A challenge's two branches: **stands** = the umpire's call; **overturned** = the other call from the same state. `docs/week10_call_transitions.csv` (one row per cell, with n) and `docs/week10_call_transitions_long.csv` (every next state with its probability).

What overturning a called strike three is worth (bases empty, 1-2 count), by situation:

| situation | outs | win prob. if it stands | if overturned | gain for the batting team | runs |
|---|---:|---:|---:|---:|---:|
| tied, 3rd inning, top | 0 | 45.5% | 47.5% | +2.0 pts | +0.186 |
| tied, 3rd inning, top | 1 | 43.7% | 45.3% | +1.6 pts | +0.146 |
| tied, 3rd inning, top | 2 | 42.5% | 43.6% | +1.1 pts | +0.092 |
| tied, 6th inning, bottom | 0 | 55.7% | 58.4% | +2.8 pts | +0.202 |
| tied, 6th inning, bottom | 1 | 53.1% | 55.3% | +2.2 pts | +0.151 |
| tied, 6th inning, bottom | 2 | 51.3% | 52.8% | +1.4 pts | +0.087 |
| tied, 9th inning, bottom | 0 | 58.5% | 63.1% | +4.6 pts | +0.202 |
| tied, 9th inning, bottom | 1 | 53.7% | 57.7% | +4.0 pts | +0.151 |
| tied, 9th inning, bottom | 2 | 50.1% | 53.0% | +2.9 pts | +0.087 |
| home up 1, 9th inning, top | 0 | 8.9% | 14.7% | +5.7 pts | +0.186 |
| home up 1, 9th inning, top | 1 | 3.6% | 8.4% | +4.8 pts | +0.146 |
| home up 1, 9th inning, top | 2 | 0.0% | 3.2% | +3.2 pts | +0.092 |
| home up 4, 7th inning, bottom | 0 | 97.2% | 97.4% | +0.2 pts | +0.202 |
| home up 4, 7th inning, bottom | 1 | 97.0% | 97.2% | +0.2 pts | +0.151 |
| home up 4, 7th inning, bottom | 2 | 96.8% | 97.0% | +0.1 pts | +0.087 |
| home down 6, 5th inning, top | 0 | 95.8% | 96.1% | +0.3 pts | +0.186 |
| home down 6, 5th inning, top | 1 | 95.6% | 95.8% | +0.2 pts | +0.146 |
| home down 6, 5th inning, top | 2 | 95.4% | 95.5% | +0.1 pts | +0.092 |

The same call is worth very different amounts — that is RQ2 in one table. Every situation × count × call is in `docs/week10_call_value_examples.csv`; `scripts/20_week10_challenge_value.py` values every 2026 challenge with it.

## 8. Figures (`figures/week10/`)

- `w10_01_runs.png` — run expectancy and the chance of scoring, by runners and outs
- `w10_02_wp_curves.png` — home win probability through the game, by score
- `w10_03_calibration.png` — reliability, out of fold: structural vs boosted
- `w10_04_agreement.png` — model vs empirical table, cell by cell
- `w10_05_call_value.png` — what overturning strike three is worth, inning by inning
- `w10_06_count_value.png` — runs an overturn is worth, by count and outs