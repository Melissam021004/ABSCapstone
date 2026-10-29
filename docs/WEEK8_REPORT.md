# Week 8 — Gradient Boosting Report

Generated 2026-10-29 16:00 by `scripts/16_week8_boosting.py` · cohort `419af01ad9eee03f` · 24 settings per library per feature set · 1000 game-level bootstrap resamples

> Read this next to `WEEK8.md`, which explains every section.

## 0. Pre-flight

- **Recorded call on challenged pitches:** recorded call matches who challenged on 100.0% of 3,453 upheld and 0.0% of 4,417 overturned challenges; verdict: FINAL (post-review) call recorded — repaired from who challenged (4,417 challenged pitches corrected).
  Every model in this report — including the refit Week 7 ladder — uses the umpire's original call. Week 7 results computed before this repair are superseded by section 1.
- Challenges: train 5,219 · August 1,435 · September 1,243 (**sealed** — opened once, by Week 9's `--final-test`).
- Boosting libraries: CatBoost 1.2.10, LightGBM 4.6.0.
- Number of trees chosen on the last 15% of training games (16% of its challenges, Jul 10 – Jul 31), then every model refit on the whole training window. August only compares settings.

## 1. The Week 7 ladder, refit

| model | params | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | AUC |
|---|---:|---|---:|---|---|---:|
| M0 · base rate | 1.0 | 0.2457 [0.2429, 0.2485] | 0.684 | +0.04 [-0.06, +0.15] | — | 0.500 |
| M1 · distance | 2.0 | 0.0207 [0.0174, 0.0244] | 0.091 | +0.02 [-0.14, +0.18] | 2.30 [1.45, 4.86] | 0.998 |
| M2 · location | 9.0 | 0.0218 [0.0179, 0.0261] | 0.093 | +0.03 [-0.13, +0.20] | 1.88 [1.30, 3.17] | 0.997 |
| M3 · + situation | 15.0 | 0.0218 [0.0178, 0.0261] | 0.093 | +0.03 [-0.14, +0.20] | 1.88 [1.30, 3.14] | 0.997 |
| M4 · + pitch | 23.0 | 0.0220 [0.0180, 0.0265] | 0.094 | +0.03 [-0.14, +0.20] | 1.86 [1.29, 3.06] | 0.997 |
| M5 · + people | 26.0 | 0.0222 [0.0182, 0.0266] | 0.094 | +0.03 [-0.14, +0.20] | 1.85 [1.29, 3.03] | 0.997 |
| GAM · location ★ | 14.8 | 0.0107 [0.0070, 0.0144] | 0.035 | -0.33 [-0.83, +0.17] | 0.98 [0.81, 1.36] | 0.999 |
| GAM · everything | 31.8 | 0.0114 [0.0077, 0.0155] | 0.038 | -0.20 [-0.75, +0.35] | 0.90 [0.75, 1.21] | 0.999 |

★ **Baseline to beat: GAM · location** (the Week 7 rule: simplest model whose Brier can't be told apart from the best).

## 2. Tuning

| library · feature set | settings tried | best August log loss | median | worst | best setting | trees |
|---|---:|---:|---:|---:|---|---:|
| CatBoost · location | 24 | 0.0369 | 0.0392 | 0.0443 | depth=5, learning_rate=0.05, l2_leaf_reg=5, random_strength=5, bagging_temperature=0.5 | 586 |
| CatBoost · everything | 24 | 0.0363 | 0.0392 | 0.0500 | depth=6, learning_rate=0.08, l2_leaf_reg=10, random_strength=1, bagging_temperature=0.5 | 760 |
| LightGBM · location | 24 | 0.0338 | 0.0424 | 0.0473 | num_leaves=15, learning_rate=0.05, min_child_samples=40, subsample=1.0, colsample_bytree=0.7, reg_lambda=0, cat_smooth=20 | 198 |
| LightGBM · everything | 24 | 0.0422 | 0.0498 | 0.0552 | num_leaves=15, learning_rate=0.05, min_child_samples=40, subsample=1.0, colsample_bytree=0.7, reg_lambda=0, cat_smooth=20 | 143 |

Every setting is in `docs/week8_tuning_log.csv`. The search is seeded: the same `--trials` gives the same settings every run. **Best − median** is what tuning bought; a small gap means the defaults were already fine and the data, not the settings, limits the model. Choosing the best of many settings on August flatters August a little — September (Week 9) is the clean check.

## 3. One comparison table (August)

Same rows, same metrics, same bootstrap for every model. Lower is better for Brier, log loss and ECE; ideal intercept 0, slope 1.

| feature set | model | Brier [95%] | Brier skill | log loss [95%] | cal. intercept | cal. slope | ECE (floor) | AUC |
|---|---|---|---:|---|---|---|---|---:|
| location | M2 · location | 0.0218 [0.0179, 0.0261] | 0.911 | 0.093 [0.075, 0.119] | +0.03 [-0.13, +0.20] | 1.88 [1.30, 3.17] | 0.031 (0.009) | 0.997 |
| location | GAM · location ★ | 0.0107 [0.0070, 0.0144] | 0.957 | 0.035 [0.024, 0.048] | -0.33 [-0.83, +0.17] | 0.98 [0.81, 1.36] | 0.003 (0.003) | 0.999 |
| location | CatBoost · location | 0.0106 [0.0065, 0.0150] | 0.957 | 0.037 [0.023, 0.052] | -0.33 [-0.87, +0.31] | 0.95 [0.85, 1.18] | 0.004 (0.003) | 0.999 |
| location | LightGBM · location | 0.0099 [0.0062, 0.0142] | 0.960 | 0.034 [0.021, 0.048] | -0.42 [-1.04, +0.29] | 0.93 [0.83, 1.20] | 0.004 (0.003) | 0.999 |
| everything | M5 · + people | 0.0222 [0.0182, 0.0266] | 0.910 | 0.094 [0.076, 0.119] | +0.03 [-0.14, +0.20] | 1.85 [1.29, 3.03] | 0.032 (0.009) | 0.997 |
| everything | GAM · everything | 0.0114 [0.0077, 0.0155] | 0.953 | 0.038 [0.025, 0.051] | -0.20 [-0.75, +0.35] | 0.90 [0.75, 1.21] | 0.002 (0.003) | 0.999 |
| everything | CatBoost · everything | 0.0105 [0.0063, 0.0150] | 0.957 | 0.036 [0.023, 0.051] | -0.39 [-0.91, +0.21] | 1.00 [0.90, 1.22] | 0.004 (0.004) | 0.999 |
| everything | LightGBM · everything | 0.0116 [0.0072, 0.0166] | 0.953 | 0.042 [0.028, 0.058] | -0.33 [-0.81, +0.20] | 1.03 [0.93, 1.24] | 0.007 (0.005) | 0.999 |

## 4. Did boosting beat the baseline?

A boosted model **beats** GAM · location only if its paired 95% interval for (its Brier − baseline Brier) lies entirely below zero on the same August games **and** its calibration is no worse (rule fixed in Week 7).

| model | Δ Brier vs baseline [95%] | Δ log loss [95%] | calibration | verdict |
|---|---|---|---|---|
| CatBoost · location | -0.0001 [-0.0016, +0.0015] | +0.0014 [-0.0077, +0.0114] | ok | does not beat it |
| CatBoost · everything | -0.0002 [-0.0013, +0.0011] | +0.0007 [-0.0071, +0.0082] | ok | does not beat it |
| LightGBM · location | -0.0007 [-0.0029, +0.0015] | -0.0015 [-0.0114, +0.0089] | ok | does not beat it |
| LightGBM · everything | +0.0009 [-0.0005, +0.0026] | +0.0067 [-0.0011, +0.0149] | ok | does not beat it |

**No boosted model reliably beat the baseline.** That is a legitimate result: the simpler, interpretable model is as good as the flexible one on new data, so it keeps the job.

Overall winner by the same rule over every model (simplest within noise of the best LightGBM · location): **GAM · location**.

| model | Brier − best [95%] | within noise of the best? |
|---|---|---|
| M1 · distance | +0.0108 [+0.0080, +0.0134] | no |
| M2 · location | +0.0118 [+0.0089, +0.0146] | no |
| GAM · location | +0.0007 [-0.0015, +0.0029] | yes |
| M3 · + situation | +0.0118 [+0.0089, +0.0146] | no |
| M4 · + pitch | +0.0121 [+0.0092, +0.0149] | no |
| M5 · + people | +0.0123 [+0.0093, +0.0151] | no |
| GAM · everything | +0.0015 [-0.0010, +0.0042] | yes |
| CatBoost · location | +0.0006 [-0.0010, +0.0022] | yes |
| LightGBM · location | — (this is the best) | — |
| CatBoost · everything | +0.0005 [-0.0016, +0.0028] | yes |
| LightGBM · everything | +0.0016 [-0.0005, +0.0040] | yes |

## 5. Location only, or everything?

Highest August AUC using location only: **0.999** (LightGBM · location); using everything: **0.999** (CatBoost · everything). Gap: +0.000.

The plan's alarm ("AUC above ~0.95 → hunt for a leak") needs adapting here: ABS is a machine applying a fixed rectangle, so location alone *should* separate overturns very well, and a high AUC from location is physics, not a leak. The leak signal is the **gap**: features ABS can't see adding a lot on top of location. That gap is small.

**Candidate using location-only: GAM · location** (simplest within noise of the best, LightGBM · location).

| model | Brier − best [95%] | within noise? |
|---|---|---|
| M1 · distance | +0.0108 [+0.0080, +0.0134] | no |
| M2 · location | +0.0118 [+0.0089, +0.0146] | no |
| GAM · location | +0.0007 [-0.0015, +0.0029] | yes |
| CatBoost · location | +0.0006 [-0.0010, +0.0022] | yes |
| LightGBM · location | — (the best) | — |

**Candidate using everything: GAM · everything** (simplest within noise of the best, CatBoost · everything).

| model | Brier − best [95%] | within noise? |
|---|---|---|
| M3 · + situation | +0.0113 [+0.0088, +0.0134] | no |
| M4 · + pitch | +0.0116 [+0.0091, +0.0137] | no |
| M5 · + people | +0.0118 [+0.0093, +0.0140] | no |
| GAM · everything | +0.0010 [-0.0006, +0.0024] | yes |
| CatBoost · everything | — (the best) | — |
| LightGBM · everything | +0.0011 [+0.0002, +0.0021] | no |

## 6. What the best boosted models learned

**LightGBM · location** — mean |SHAP| (log-odds)

| rank | feature | block | importance |
|---:|---|---|---:|
| 1 | inches against the call | location | 6.2692 |
| 2 | vertical location | location | 0.3685 |
| 3 | horizontal location | location | 0.3532 |
| 4 | call (ball / strike) | location | 0.3497 |
| 5 | batter height | location | 0.0745 |
| 6 | nearest edge | location | 0.0460 |

**CatBoost · everything** — mean |SHAP| (log-odds)

| rank | feature | block | importance |
|---:|---|---|---:|
| 1 | inches against the call | location | 4.1654 |
| 2 | who challenged | people | 0.5663 |
| 3 | nearest edge | location | 0.5489 |
| 4 | pitch type | pitch | 0.3723 |
| 5 | catcher ID | identity | 0.2920 |
| 6 | umpire ID | identity | 0.1519 |
| 7 | vertical location | location | 0.1239 |
| 8 | pitcher ID | identity | 0.1093 |
| 9 | horizontal location | location | 0.1031 |
| 10 | vertical break | pitch | 0.0834 |
| 11 | batter ID | identity | 0.0743 |
| 12 | catcher-game accuracy | people | 0.0656 |

**Leak check:** passed — best location feature `inches against the call` ranks #1; best ID `catcher ID` #5.

## 7. Going to Week 9

- **Location-only candidate: GAM · location** — the default for the MDP's p(s) (Week 7's decision: ABS rules on location alone, and the policy will be asked about pitches nobody challenged).
- **Candidate using everything: GAM · everything** — the instrument for studying who challenges well.
- Both are saved in `models/week8/` and named in `models/week8/selection.json`. Week 9 calibrates them on August, checks subgroups, locks the production model, and only then opens September — once — for every model at the same time.

## 8. Figures (`figures/week8/`)

- `w8_01_tuning.png` — every setting tried: how much tuning moved the August log loss
- `w8_02_comparison.png` — the one comparison table as a picture, and the Δ vs the baseline
- `w8_03_reliability.png` — reliability diagrams for the four families × two feature sets
- `w8_04_importance.png` — what the best boosted models rely on (SHAP), coloured by block
- `w8_05_shap_distance.png` — the learned distance effect, read straight off the SHAP values
- `w8_06_curves.png` — overturn chance vs inches past each edge: GAM vs the best boosted model
