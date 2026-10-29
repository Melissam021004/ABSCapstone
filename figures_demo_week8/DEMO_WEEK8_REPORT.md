# Week 8 — Gradient Boosting Report

Generated 2026-10-20 18:54 by `scripts/16_week8_boosting.py` · cohort `c380922df66bdf57` · 24 settings per library per feature set · 1000 game-level bootstrap resamples

> **DEMO DATA — every number below comes from the synthetic fixture.**

> Read this next to `WEEK8.md`, which explains every section.

## 0. Pre-flight

- **Recorded call on challenged pitches:** recorded call matches who challenged on 100.0% of 2,236 upheld and 100.0% of 2,427 overturned challenges; verdict: ORIGINAL umpire call recorded — no repair needed.
- Challenges: train 3,211 · August 793 · September 659 (**sealed** — opened once, by Week 9's `--final-test`).
- Boosting libraries: HistGB 1.9.1.
  **HistGradientBoosting stands in for a library that isn't installed** — install CatBoost and LightGBM and re-run for the plan's comparison.
- Number of trees chosen on the last 15% of training games (Jul 12 – Jul 31), then every model refit on the whole training window. August only compares settings.

## 1. The Week 7 ladder, refit

| model | params | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | AUC |
|---|---:|---|---:|---|---|---:|
| M0 · base rate | 1.0 | 0.2491 [0.2477, 0.2504] | 0.691 | +0.06 [-0.08, +0.21] | — | 0.500 |
| M1 · distance ★ | 2.0 | 0.0439 [0.0342, 0.0549] | 0.143 | -0.13 [-0.51, +0.24] | 0.87 [0.74, 1.07] | 0.987 |
| M2 · location | 9.0 | 0.0451 [0.0347, 0.0564] | 0.144 | -0.14 [-0.52, +0.24] | 0.84 [0.72, 1.01] | 0.986 |
| M3 · + situation | 15.0 | 0.0447 [0.0346, 0.0560] | 0.143 | -0.17 [-0.55, +0.22] | 0.85 [0.73, 1.02] | 0.987 |
| M4 · + pitch | 21.0 | 0.0451 [0.0348, 0.0564] | 0.145 | -0.18 [-0.56, +0.21] | 0.84 [0.72, 1.01] | 0.986 |
| M5 · + people | 24.0 | 0.0457 [0.0353, 0.0571] | 0.147 | -0.17 [-0.56, +0.24] | 0.82 [0.71, 0.98] | 0.986 |
| GAM · location | 9.3 | 0.0451 [0.0347, 0.0564] | 0.144 | -0.14 [-0.52, +0.24] | 0.84 [0.72, 1.01] | 0.986 |
| GAM · everything | 24.3 | 0.0458 [0.0353, 0.0572] | 0.147 | -0.17 [-0.56, +0.25] | 0.82 [0.71, 0.98] | 0.986 |

★ **Baseline to beat: M1 · distance** (the Week 7 rule: simplest model whose Brier can't be told apart from the best).

## 2. Tuning

| library · feature set | settings tried | best August log loss | median | worst | best setting | trees |
|---|---:|---:|---:|---:|---|---:|
| HistGB · location | 24 | 0.1492 | 0.1558 | 0.1600 | max_leaf_nodes=4, learning_rate=0.1, min_samples_leaf=80, l2_regularization=0, max_features=0.6 | 82 |
| HistGB · everything | 24 | 0.1695 | 0.1791 | 0.1953 | max_leaf_nodes=4, learning_rate=0.1, min_samples_leaf=80, l2_regularization=5, max_features=1.0 | 54 |

Every setting is in `docs/week8_tuning_log.csv`. The search is seeded: the same `--trials` gives the same settings every run. **Best − median** is what tuning bought; a small gap means the defaults were already fine and the data, not the settings, limits the model. Choosing the best of many settings on August flatters August a little — September (Week 9) is the clean check.

## 3. One comparison table (August)

Same rows, same metrics, same bootstrap for every model. Lower is better for Brier, log loss and ECE; ideal intercept 0, slope 1.

| feature set | model | Brier [95%] | Brier skill | log loss [95%] | cal. intercept | cal. slope | ECE (floor) | AUC |
|---|---|---|---:|---|---|---|---|---:|
| location | M1 · distance ★ | 0.0439 [0.0342, 0.0549] | 0.824 | 0.143 [0.112, 0.180] | -0.13 [-0.51, +0.24] | 0.87 [0.74, 1.07] | 0.013 (0.011) | 0.987 |
| location | M2 · location | 0.0451 [0.0347, 0.0564] | 0.819 | 0.144 [0.112, 0.180] | -0.14 [-0.52, +0.24] | 0.84 [0.72, 1.01] | 0.019 (0.010) | 0.986 |
| location | GAM · location | 0.0451 [0.0347, 0.0564] | 0.819 | 0.144 [0.112, 0.180] | -0.14 [-0.52, +0.24] | 0.84 [0.72, 1.01] | 0.019 (0.010) | 0.986 |
| location | HistGB · location | 0.0470 [0.0367, 0.0592] | 0.811 | 0.149 [0.117, 0.188] | -0.13 [-0.52, +0.28] | 0.87 [0.77, 1.01] | 0.016 (0.011) | 0.986 |
| everything | M5 · + people | 0.0457 [0.0353, 0.0571] | 0.816 | 0.147 [0.115, 0.183] | -0.17 [-0.56, +0.24] | 0.82 [0.71, 0.98] | 0.019 (0.010) | 0.986 |
| everything | GAM · everything | 0.0458 [0.0353, 0.0572] | 0.816 | 0.147 [0.115, 0.183] | -0.17 [-0.56, +0.25] | 0.82 [0.71, 0.98] | 0.020 (0.010) | 0.986 |
| everything | HistGB · everything | 0.0497 [0.0393, 0.0626] | 0.800 | 0.169 [0.135, 0.211] | -0.15 [-0.50, +0.23] | 0.94 [0.84, 1.07] | 0.019 (0.015) | 0.981 |

## 4. Did boosting beat the baseline?

A boosted model **beats** M1 · distance only if its paired 95% interval for (its Brier − baseline Brier) lies entirely below zero on the same August games **and** its calibration is no worse (rule fixed in Week 7).

| model | Δ Brier vs baseline [95%] | Δ log loss [95%] | calibration | verdict |
|---|---|---|---|---|
| HistGB · location | +0.0031 [+0.0008, +0.0057] | +0.0064 [-0.0027, +0.0142] | ok | does not beat it |
| HistGB · everything | +0.0059 [+0.0021, +0.0099] | +0.0266 [+0.0137, +0.0400] | ok | does not beat it |

**No boosted model reliably beat the baseline.** That is a legitimate result: the simpler, interpretable model is as good as the flexible one on new data, so it keeps the job.

Overall winner by the same rule over every model (simplest within noise of the best M1 · distance): **M1 · distance**.

| model | Brier − best [95%] | within noise of the best? |
|---|---|---|
| M1 · distance | — (this is the best) | — |
| M2 · location | +0.0011 [-0.0005, +0.0027] | yes |
| GAM · location | +0.0012 [-0.0005, +0.0027] | yes |
| M3 · + situation | +0.0008 [-0.0012, +0.0026] | yes |
| M4 · + pitch | +0.0012 [-0.0009, +0.0033] | yes |
| M5 · + people | +0.0018 [-0.0005, +0.0044] | yes |
| GAM · everything | +0.0018 [-0.0005, +0.0044] | yes |
| HistGB · location | +0.0031 [+0.0008, +0.0057] | no |
| HistGB · everything | +0.0059 [+0.0021, +0.0099] | no |

## 5. Location only, or everything?

Highest August AUC using location only: **0.987** (M1 · distance); using everything: **0.987** (M3 · + situation). Gap: +0.000.

The plan's alarm ("AUC above ~0.95 → hunt for a leak") needs adapting here: ABS is a machine applying a fixed rectangle, so location alone *should* separate overturns very well, and a high AUC from location is physics, not a leak. The leak signal is the **gap**: features ABS can't see adding a lot on top of location. That gap is small.

**Candidate using location-only: M1 · distance** (simplest within noise of the best, M1 · distance).

| model | Brier − best [95%] | within noise? |
|---|---|---|
| M1 · distance | — (the best) | — |
| M2 · location | +0.0011 [-0.0005, +0.0027] | yes |
| GAM · location | +0.0012 [-0.0005, +0.0027] | yes |
| HistGB · location | +0.0031 [+0.0008, +0.0057] | no |

**Candidate using everything: M3 · + situation** (simplest within noise of the best, M3 · + situation).

| model | Brier − best [95%] | within noise? |
|---|---|---|
| M3 · + situation | — (the best) | — |
| M4 · + pitch | +0.0004 [-0.0002, +0.0011] | yes |
| M5 · + people | +0.0010 [-0.0004, +0.0025] | yes |
| GAM · everything | +0.0011 [-0.0004, +0.0025] | yes |
| HistGB · everything | +0.0051 [+0.0009, +0.0098] | no |

## 6. What the best boosted models learned

**HistGB · location** — rise in August log loss when shuffled

| rank | feature | block | importance |
|---:|---|---|---:|
| 1 | inches against the call | location | 2.4513 |
| 2 | vertical location | location | 0.0017 |
| 3 | horizontal location | location | -0.0001 |
| 4 | nearest edge | location | -0.0001 |
| 5 | batter height | location | -0.0004 |
| 6 | call (ball / strike) | location | -0.0006 |

**HistGB · everything** — rise in August log loss when shuffled

| rank | feature | block | importance |
|---:|---|---|---:|
| 1 | inches against the call | location | 1.7680 |
| 2 | vertical location | location | 0.0004 |
| 3 | catcher ID | identity | 0.0003 |
| 4 | horizontal location | location | 0.0000 |
| 5 | call (ball / strike) | location | 0.0000 |
| 6 | balls | situation | 0.0000 |
| 7 | strikes | situation | 0.0000 |
| 8 | outs | situation | 0.0000 |
| 9 | batter height | location | 0.0000 |
| 10 | inning | situation | 0.0000 |
| 11 | leverage | situation | 0.0000 |
| 12 | velocity | pitch | 0.0000 |

**Leak check:** passed — best location feature `inches against the call` ranks #1; best ID `catcher ID` #3.

## 7. Going to Week 9

- **Location-only candidate: M1 · distance** — the default for the MDP's p(s) (Week 7's decision: ABS rules on location alone, and the policy will be asked about pitches nobody challenged).
- **Candidate using everything: M3 · + situation** — the instrument for studying who challenges well.
- Both are saved in `models/week8/` and named in `models/week8/selection.json`. Week 9 calibrates them on August, checks subgroups, locks the production model, and only then opens September — once — for every model at the same time.

## 8. Figures (`figures/week8/`)

- `w8_01_tuning.png` — every setting tried: how much tuning moved the August log loss
- `w8_02_comparison.png` — the one comparison table as a picture, and the Δ vs the baseline
- `w8_03_reliability.png` — reliability diagrams for the four families × two feature sets
- `w8_04_importance.png` — what the best boosted models rely on (SHAP), coloured by block
- `w8_05_shap_distance.png` — the learned distance effect, read straight off the SHAP values
- `w8_06_curves.png` — overturn chance vs inches past each edge: GAM vs the best boosted model
