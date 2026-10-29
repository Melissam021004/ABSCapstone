# Week 7 — Baseline Overturn Model Report

Generated 2026-10-13 18:45 by `scripts/15_week7_baseline.py` · cohort `c380922df66bdf57` · data `data_demo` · 1000 game-level bootstrap resamples

> **DEMO DATA — every number below comes from the synthetic fixture.** Re-run on the real cohort before quoting anything.

> Read this next to `WEEK7.md`, which explains what each number means and what to look for.

## 1. The question, and why calibration comes first

Every model here estimates **p(s): given that a challenge happened, the probability ABS overturns the call.** That probability is an input to the Week 11 MDP, which challenges when p(s)·V(overturned) + (1 − p(s))·V(stand, r − 1) beats V(stand, r).

**We care about calibration over discrimination because the output feeds an MDP.** The decision compares p(s) against a threshold set by the game state. A model that ranks pitches perfectly (high AUC) but says 60% when the truth is 40% will recommend challenges that lose value — and AUC cannot see that error, because AUC only checks the ORDER of the predictions, not their size. So the models are graded on Brier score, log loss, calibration intercept and slope, the reliability diagram and ECE first; ROC-AUC is reported last, as a secondary check.

## 2. Data

| window | dates | challenges | games | overturned |
|---|---|---:|---:|---:|
| train | Mar 26 – Jul 31 | 3,211 | 826 | 51.9% |
| validate | Aug 01 – Aug 31 | 793 | 201 | 53.5% |
| test | Sept → | 659 | — | *sealed until the end of Week 8* |

- Umpire and catcher accuracy come from 126,425 **training-window** called pitches only (league agreement with the ABS zone: 93.22%), shrunk toward the league rate, and enter as percentage points above or below it.
- Leverage index was missing for 0.0% of training challenges; those were set to average leverage (log LI = 0).
- Dropped because they had no variation in training (nothing to estimate): `p_throws`, `stand`.

## 3. The sign of the distance feature

Week 5's `zone_dist_in` is signed relative to the zone (+ outside, − inside). Used as-is, its AUC for predicting an overturn is **0.529** — barely better than a coin flip — because a pitch outside the zone is likely overturned when it was called a strike and almost never when it was called a ball, and pooling the two cancels them out. Re-signed relative to the **call** (`dist_against_call`: + means the pitch is on the wrong side of the edge for the call that was made), its AUC is **0.991**. Every model below uses the re-signed version. (Figure `w7_01_sign_trap.png`.)

## 4. Headline results — validation window

Lower is better for Brier, log loss and ECE. Ideal calibration intercept is 0, ideal slope is 1. Brackets are 95% intervals from resampling whole validation games. Params = coefficients (logistic) or effective degrees of freedom (GAM).

| model | params | Brier | Brier skill | log loss | cal. intercept | cal. slope | ECE (noise floor) | AUC |
|---|---:|---|---:|---|---|---|---|---|
| M0 · base rate | 1.0 | 0.2491 [0.2477, 0.2504] | 0.000 | 0.691 [0.689, 0.694] | +0.06 [-0.08, +0.21] | — | 0.016 (0.014) | 0.500 [0.500, 0.500] |
| M1 · distance | 2.0 | 0.0439 [0.0342, 0.0549] | 0.824 | 0.143 [0.112, 0.180] | -0.13 [-0.51, +0.24] | 0.87 [0.74, 1.07] | 0.013 (0.011) | 0.987 [0.981, 0.992] |
| M2 · location | 9.0 | 0.0451 [0.0347, 0.0564] | 0.819 | 0.144 [0.112, 0.180] | -0.14 [-0.52, +0.24] | 0.84 [0.72, 1.01] | 0.019 (0.010) | 0.986 [0.980, 0.992] |
| M3 · + situation | 15.0 | 0.0447 [0.0346, 0.0560] | 0.821 | 0.143 [0.111, 0.177] | -0.17 [-0.55, +0.22] | 0.85 [0.73, 1.02] | 0.018 (0.010) | 0.987 [0.981, 0.992] |
| M4 · + pitch | 21.0 | 0.0451 [0.0348, 0.0564] | 0.819 | 0.145 [0.113, 0.180] | -0.18 [-0.56, +0.21] | 0.84 [0.72, 1.01] | 0.013 (0.010) | 0.986 [0.980, 0.991] |
| M5 · + people | 24.0 | 0.0457 [0.0353, 0.0571] | 0.816 | 0.147 [0.115, 0.183] | -0.17 [-0.56, +0.24] | 0.82 [0.71, 0.98] | 0.019 (0.010) | 0.986 [0.980, 0.991] |
| GAM · location | 9.0 | 0.0451 [0.0347, 0.0564] | 0.819 | 0.144 [0.112, 0.180] | -0.14 [-0.52, +0.24] | 0.84 [0.72, 1.01] | 0.019 (0.010) | 0.986 [0.980, 0.992] |
| GAM · everything | 24.0 | 0.0457 [0.0353, 0.0571] | 0.816 | 0.147 [0.115, 0.183] | -0.17 [-0.56, +0.24] | 0.82 [0.71, 0.98] | 0.019 (0.010) | 0.986 [0.980, 0.991] |

**Where decisions get made.** Most challenges are easy calls that every model gets right. The MDP's challenge-or-hold choice turns on the uncertain ones, so: among the validation challenges each model put between 20% and 80%, what did it predict on average, and how many were actually overturned?

| model | challenges in 20–80% | mean predicted | actually overturned [95%] |
|---|---:|---:|---|
| M1 · distance | 83 | 54.7% | 56.6% [45.9%, 66.8%] |
| M2 · location | 83 | 55.3% | 60.2% [49.5%, 70.1%] |
| M3 · + situation | 82 | 55.3% | 59.8% [48.9%, 69.7%] |
| M4 · + pitch | 85 | 55.1% | 58.8% [48.2%, 68.7%] |
| M5 · + people | 82 | 53.3% | 58.5% [47.7%, 68.6%] |
| GAM · location | 83 | 55.3% | 60.2% [49.5%, 70.1%] |
| GAM · everything | 82 | 53.3% | 58.5% [47.7%, 68.6%] |

## 5. What each block adds

Two different questions, answered side by side. The **likelihood-ratio test** asks whether the bigger model fits the *training* months better than chance would allow. The **validation change** asks whether it predicts *August* better. A block can pass the first and fail the second — that's a pattern that didn't carry forward, and it's the second question that matters. Negative Δ = the bigger model is better.

| step | params added | LR test (train) | Δ Brier (validation) [95%] | Δ log loss [95%] | verdict |
|---|---:|---|---|---|---|
| + edge & call direction (M2 vs M1) | +7.0 | χ²=30.9, df=7, p=6.4e-05 | +0.0011 [-0.0005, +0.0027] | +0.0011 [-0.0070, +0.0078] | no detectable difference |
| + situation (M3 vs M2) | +6.0 | χ²=3.9, df=6, p=0.69 | -0.0004 [-0.0013, +0.0005] | -0.0008 [-0.0034, +0.0018] | no detectable difference |
| + pitch (M4 vs M3) | +6.0 | χ²=1.5, df=6, p=0.96 | +0.0004 [-0.0002, +0.0011] | +0.0020 [+0.0003, +0.0038] | no detectable difference (log loss: worse) |
| + people (M5 vs M4) | +3.0 | χ²=4.9, df=3, p=0.18 | +0.0006 [-0.0007, +0.0020] | +0.0019 [-0.0014, +0.0056] | no detectable difference |
| curve instead of line (location) (GAM-loc vs M2) | +0.0 | not nested | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | no detectable difference |
| curve instead of line (everything) (GAM-full vs M5) | +0.0 | not nested | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | no detectable difference |
| everything vs location only (M5 vs M2) | +15.0 | χ²=10.3, df=15, p=0.8 | +0.0007 [-0.0012, +0.0025] | +0.0032 [-0.0017, +0.0079] | no detectable difference |

For scale: distance alone (M1 vs M0) changes Brier by -0.2050. Everything after that is fine-tuning.

## 6. The location model, in inches

For each edge and each direction of call: where the overturn chance crosses 50% ("the coin-flip point", in inches past our computed edge, + = on the wrong side of the call), and how many inches it takes to go from 25% to 75% ("the blur"). Everything else is held at typical values.

**M2 · location**

| edge | challenge | coin-flip point (in.) [95%] | 25→75% blur (in.) | chance right at our edge |
|---|---|---|---:|---:|
| side | called strike (batter challenges) | -0.17 [-0.21, -0.12] | 0.29 | 78% |
| side | called ball (fielder challenges) | -0.13 [-0.18, -0.08] | 0.29 | 73% |
| top | called strike (batter challenges) | -0.25 [-0.37, -0.14] | 0.46 | 77% |
| top | called ball (fielder challenges) | -0.26 [-0.37, -0.16] | 0.46 | 78% |
| bottom | called strike (batter challenges) | -0.21 [-0.31, -0.11] | 0.38 | 77% |
| bottom | called ball (fielder challenges) | -0.03 [-0.13, +0.06] | 0.38 | 55% |

**GAM · location**

| edge | challenge | coin-flip point (in.) [95%] | 25→75% blur (in.) | chance right at our edge |
|---|---|---|---:|---:|
| side | called strike (batter challenges) | -0.17 [-0.21, -0.12] | 0.29 | 78% |
| side | called ball (fielder challenges) | -0.13 [-0.18, -0.08] | 0.29 | 73% |
| top | called strike (batter challenges) | -0.25 [-0.37, -0.14] | 0.46 | 77% |
| top | called ball (fielder challenges) | -0.26 [-0.37, -0.16] | 0.46 | 78% |
| bottom | called strike (batter challenges) | -0.21 [-0.31, -0.11] | 0.38 | 77% |
| bottom | called ball (fielder challenges) | -0.03 [-0.13, +0.06] | 0.38 | 55% |

How to read the two rows for each edge (details in WEEK7.md):

- **Both coin-flip points below zero** → challenges succeed before the pitch even reaches our computed edge: the challengers know something the measured location doesn't (or the tracking puts the ball slightly differently from what ABS used).
- **The two points far apart, in opposite directions** → the ABS edge isn't where our height-based zone puts it. Half the gap is the size of the shift.

Sign of the half-gap: **+** means ABS's zone reaches further out than ours at that edge (it calls strikes on pitches our zone calls balls); **−** means it stops short.

| edge (M2) | midpoint of the two coin-flip points | half the gap (strike − ball) |
|---|---:|---:|
| side | -0.15 | -0.02 |
| top | -0.26 | +0.01 |
| bottom | -0.12 | -0.09 |

**GAM smoothness.** λ = 10000, chosen by 5-fold cross-validation on the training months with folds grouped by game. Effective degrees of freedom per curve (1 = the data chose a straight line; more = real bends): side: 1.0, top: 1.0, bottom: 1.0.

## 7. Everything else, at a fixed location (M5)

Odds ratios from the full logistic model, per a meaningful unit. ABS only sees the ball, so **none of these can change how ABS rules a given pitch.** An effect here means the challenges that got made in that situation were different — a statement about who chose to challenge, not about the system. Read them as selection effects.

| term | per | odds ratio [95%] | p |
|---|---|---|---:|
| `balls` | per extra ball in the count | 0.95 [0.75, 1.20] | 0.67 |
| `strikes` | per extra strike in the count | 1.07 [0.83, 1.37] | 0.61 |
| `outs_when_up` | per extra out | 0.92 [0.73, 1.17] | 0.51 |
| `inning` | per inning later | 0.96 [0.89, 1.04] | 0.34 |
| `log_leverage` | per doubling of leverage | 0.88 [0.71, 1.09] | 0.24 |
| `chal_left` | per extra challenge in hand | 0.82 [0.54, 1.26] | 0.36 |
| `release_speed` | per 5 mph faster | 0.93 [0.78, 1.10] | 0.39 |
| `pfx_x` | per 6 in. more horizontal break | 0.96 [0.85, 1.09] | 0.54 |
| `pfx_z` | per 6 in. more vertical break | 1.02 [0.90, 1.17] | 0.72 |
| `batter_height_ft` | per inch taller batter | 1.00 [0.93, 1.08] | 0.94 |
| `pitcher_challenge` | pitcher (vs catcher) made the challenge | 0.53 [0.30, 0.95] | 0.033 |
| `ump_acc_pp` | per point more accurate umpire | 1.06 [0.61, 1.86] | 0.83 |
| `catcher_acc_pp` | per point higher catcher-game accuracy | 1.14 [0.70, 1.85] | 0.6 |
| `pitch_group=breaking` | pitch breaking (vs fastball) | 0.97 [0.64, 1.45] | 0.87 |
| `pitch_group=offspeed` | pitch offspeed (vs fastball) | 1.06 [0.57, 1.98] | 0.86 |

## 8. Subgroup calibration (M1 · distance, validation)

Overall calibration can hide a model that's too high for one kind of challenge and too low for another. Each row: what the model predicted on average for that slice, and what happened. A 95% interval that excludes the prediction is worth a look; with slices this small, most won't. Week 9 does this properly on more data.

| slice | n | predicted | observed [95%] | gap (pts) |
|---|---:|---:|---|---:|
| Who challenged: batter | 416 | 52.3% | 52.2% [47.4%, 56.9%] | -0.1 |
| Who challenged: catcher | 272 | 57.9% | 56.6% [50.7%, 62.4%] | -1.3 |
| Who challenged: pitcher | 105 | 50.5% | 50.5% [41.1%, 59.9%] | -0.0 |
| Edge · call: side · strike | 265 | 51.9% | 52.5% [46.4%, 58.4%] | +0.5 |
| Edge · call: side · ball | 219 | 63.1% | 61.2% [54.6%, 67.4%] | -1.9 |
| Edge · call: top · strike | 69 | 58.6% | 60.9% [49.1%, 71.5%] | +2.3 |
| Edge · call: top · ball | 84 | 49.3% | 48.8% [38.4%, 59.3%] | -0.5 |
| Edge · call: bottom · strike | 82 | 48.1% | 43.9% [33.7%, 54.7%] | -4.2 |
| Edge · call: bottom · ball | 74 | 41.8% | 43.2% [32.6%, 54.6%] | +1.4 |
| Count: 0–1 strikes | 570 | 52.3% | 50.9% [46.8%, 55.0%] | -1.4 |
| Count: 2 strikes | 223 | 58.3% | 60.1% [53.5%, 66.3%] | +1.8 |
| Challenges in hand: 1 left | 217 | 50.6% | 47.0% [40.5%, 53.6%] | -3.6 |
| Challenges in hand: 2 left | 576 | 55.2% | 55.9% [51.8%, 59.9%] | +0.7 |
| Inning: 1–3 | 276 | 56.2% | 56.9% [51.0%, 62.6%] | +0.7 |
| Inning: 4–6 | 241 | 53.0% | 52.3% [46.0%, 58.5%] | -0.7 |
| Inning: 7–9 | 256 | 52.8% | 51.6% [45.5%, 57.6%] | -1.3 |
| Inning: 10+ | 20 | 49.9% | 45.0% [25.8%, 65.8%] | -4.9 |
| Leverage: low (<0.85) | 310 | 55.1% | 56.1% [50.6%, 61.5%] | +1.1 |
| Leverage: medium | 367 | 52.5% | 51.8% [46.7%, 56.8%] | -0.7 |
| Leverage: high (>2) | 116 | 55.6% | 51.7% [42.7%, 60.6%] | -3.9 |

**By month** (training months are in-sample, so small gaps are expected there; a steady drift across months is the thing to look for — it would mean the same pitch location was overturned at a different rate as the season went on):

| month | window | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| 2026-03 | train (in-sample) | 155 | 46.8% | 43.9% [36.3%, 51.7%] |
| 2026-04 | train (in-sample) | 774 | 52.4% | 51.6% [48.0%, 55.1%] |
| 2026-05 | train (in-sample) | 761 | 53.4% | 53.1% [49.5%, 56.6%] |
| 2026-06 | train (in-sample) | 755 | 52.5% | 54.0% [50.5%, 57.6%] |
| 2026-07 | train (in-sample) | 766 | 50.2% | 50.4% [46.9%, 53.9%] |
| 2026-08 | validate | 793 | 54.0% | 53.5% [50.0%, 56.9%] |

## 9. The baseline to beat

**Rule (fixed before looking):** take the model with the lowest validation Brier; then, going from simplest to most complex, choose the first model whose paired 95% interval for (its Brier − the best Brier) reaches zero. A gain August can't confirm doesn't count, and ties go to the simpler model.

| model | Brier − best [95%] | indistinguishable from best? |
|---|---|---|
| M1 · distance | — (this is the best) | — |
| M2 · location | +0.0011 [-0.0005, +0.0027] | yes |
| GAM · location | +0.0011 [-0.0005, +0.0027] | yes |
| M3 · + situation | +0.0008 [-0.0012, +0.0026] | yes |
| M4 · + pitch | +0.0012 [-0.0009, +0.0033] | yes |
| M5 · + people | +0.0018 [-0.0005, +0.0044] | yes |
| GAM · everything | +0.0018 [-0.0005, +0.0044] | yes |

### Baseline to beat: **M1 · distance**

Validation Brier 0.0439 [0.0342, 0.0549], log loss 0.143 [0.112, 0.180], calibration intercept -0.13 [-0.51, +0.24], slope 0.87 [0.74, 1.07], AUC 0.987. (Lowest Brier overall: M1 · distance.)

Its calibration intercept interval contains 0 and its slope interval contains 1: no detectable miscalibration on August.

**A Week 8 model beats it** only if its paired-bootstrap 95% interval for (its Brier − M1's Brier) lies entirely below zero on the same validation games, and its calibration intercept and slope are no worse.

**Location-only candidate for the MDP's p(s): M1 · distance** (same rule, restricted to models that use only what ABS sees). Whether p(s) should also use the situation/pitch/people blocks is a decision for you and your supervisor — see WEEK7.md, "Location-only or everything?".

## 10. Most confidently wrong validation predictions — check by hand

The ten validation challenges M2 · location was most surprised by. M2 is used here on purpose: a straight line in log-odds is the model most shocked by a result far from the edge, so it makes data problems stand out (a flexible GAM can learn a small "floor" from a few bad records and hide them). Very confident misses are usually data problems — a challenge linked to the wrong pitch, a wrong height, a mislabelled outcome — and they are what drags log loss up. Look each one up on Baseball Savant.

| game_pk | date | inning | count | challenger | call | edge | in. against call | p(overturn) | result |
|---|---|---:|---|---|---|---|---:|---:|---|
| 801017 | 2026-08-28 | 7 | 1-0 | batter | strike | side | +0.50 | 0.994 | stood |
| 800940 | 2026-08-17 | 5 | 1-2 | batter | strike | side | +0.34 | 0.981 | stood |
| 800925 | 2026-08-14 | 3 | 2-1 | catcher | ball | top | +0.52 | 0.978 | stood |
| 800896 | 2026-08-10 | 1 | 1-0 | catcher | ball | top | -1.02 | 0.026 | overturned |
| 800926 | 2026-08-14 | 8 | 0-0 | catcher | ball | top | +0.48 | 0.972 | stood |
| 800954 | 2026-08-19 | 2 | 3-1 | catcher | ball | side | +0.22 | 0.936 | stood |
| 800917 | 2026-08-13 | 1 | 2-2 | catcher | ball | side | +0.19 | 0.921 | stood |
| 801001 | 2026-08-26 | 1 | 1-0 | batter | strike | side | -0.44 | 0.107 | overturned |
| 800913 | 2026-08-12 | 1 | 1-1 | pitcher | ball | side | +0.12 | 0.873 | stood |
| 800981 | 2026-08-23 | 5 | 0-2 | batter | strike | side | +0.08 | 0.872 | stood |

## 11. Fitting notes

- **GAM · location**: every curve came out a straight line — λ is at the top of the grid, but a bigger λ can't make a line straighter, so this is a finding (the data showed no bend), not a problem
- **GAM · everything**: every curve came out a straight line — λ is at the top of the grid, but a bigger λ can't make a line straighter, so this is a finding (the data showed no bend), not a problem
- Intervals reflect how much each score would move with a different set of August games. They do not include the uncertainty from re-fitting the models.
- The test window was not read. It is used once, at the end of Week 8.

## 12. Figures (`figures/week7/`)

- `w7_01_sign_trap.png` — why the distance feature has to be re-signed relative to the call
- `w7_02_reliability_baselines.png` — reliability diagrams for the two location-only baselines
- `w7_03_reliability_all.png` — reliability diagram for every model, side by side
- `w7_04_block_contributions.png` — what each block adds on August, with 95% intervals
- `w7_05_distance_curves.png` — overturn chance vs inches past each edge — GAM curve vs logistic line
- `w7_06_odds_ratios.png` — M5's odds ratios: location (what ABS sees) vs everything else
- `w7_07_subgroup_calibration.png` — observed minus predicted, by slice and by month
