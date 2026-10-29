# Week 7 — Baseline Overturn Model Report

Generated 2026-10-29 15:56 by `scripts/15_week7_baseline.py` · cohort `419af01ad9eee03f` · data `data` · 1000 game-level bootstrap resamples

> Read this next to `WEEK7.md`, which explains what each number means and what to look for.

## 1. The question, and why calibration comes first

Every model here estimates **p(s): given that a challenge happened, the probability ABS overturns the call.** That probability is an input to the Week 11 MDP, which challenges when p(s)·V(overturned) + (1 − p(s))·V(stand, r − 1) beats V(stand, r).

**We care about calibration over discrimination because the output feeds an MDP.** The decision compares p(s) against a threshold set by the game state. A model that ranks pitches perfectly (high AUC) but says 60% when the truth is 40% will recommend challenges that lose value — and AUC cannot see that error, because AUC only checks the ORDER of the predictions, not their size. So the models are graded on Brier score, log loss, calibration intercept and slope, the reliability diagram and ECE first; ROC-AUC is reported last, as a secondary check.

## 2. Data

| window | dates | challenges | games | overturned |
|---|---|---:|---:|---:|
| train | Mar 25 – Jul 31 | 5,219 | 1,580 | 55.6% |
| validate | Aug 01 – Aug 31 | 1,435 | 405 | 56.7% |
| test | Sept → | 1,243 | — | *sealed until the end of Week 8* |

- **Which call is recorded on challenged pitches?** The recorded call matches who challenged on 100.0% of 3,453 upheld and 0.0% of 4,417 overturned challenges; verdict: FINAL (post-review) call recorded — repaired from who challenged (4,417 challenged pitches corrected).
  Statcast-style data can store the call *after* the ABS review. On an overturned pitch that makes the call look correct, which erases the distance signal and points "challenges left" at the wrong team. Every model below uses the umpire's original call, worked out from who challenged.
- Umpire and catcher accuracy come from 252,016 **training-window** called pitches only (league agreement with the ABS zone: 93.88%), shrunk toward the league rate, and enter as percentage points above or below it.
- Leverage index was missing for 0.8% of training challenges; those were set to average leverage (log LI = 0).

## 3. The sign of the distance feature

Week 5's `zone_dist_in` is signed relative to the zone (+ outside, − inside). Used as-is, its AUC for predicting an overturn is **0.464** — barely better than a coin flip — because a pitch outside the zone is likely overturned when it was called a strike and almost never when it was called a ball, and pooling the two cancels them out. Re-signed relative to the **call** (`dist_against_call`: + means the pitch is on the wrong side of the edge for the call that was made), its AUC is **0.997**. Every model below uses the re-signed version. (Figure `w7_01_sign_trap.png`.)

## 4. Headline results — validation window

Lower is better for Brier, log loss and ECE. Ideal calibration intercept is 0, ideal slope is 1. Brackets are 95% intervals from resampling whole validation games. Params = coefficients (logistic) or effective degrees of freedom (GAM).

| model | params | Brier | Brier skill | log loss | cal. intercept | cal. slope | ECE (noise floor) | AUC |
|---|---:|---|---:|---|---|---|---|---|
| M0 · base rate | 1.0 | 0.2457 [0.2429, 0.2485] | 0.000 | 0.684 [0.679, 0.690] | +0.04 [-0.06, +0.15] | — | 0.011 (0.010) | 0.500 [0.500, 0.500] |
| M1 · distance | 2.0 | 0.0207 [0.0174, 0.0244] | 0.916 | 0.091 [0.074, 0.115] | +0.02 [-0.14, +0.18] | 2.30 [1.45, 4.86] | 0.035 (0.010) | 0.998 [0.995, 1.000] |
| M2 · location | 9.0 | 0.0218 [0.0179, 0.0261] | 0.911 | 0.093 [0.075, 0.119] | +0.03 [-0.13, +0.20] | 1.88 [1.30, 3.17] | 0.031 (0.009) | 0.997 [0.994, 0.999] |
| M3 · + situation | 15.0 | 0.0218 [0.0178, 0.0261] | 0.911 | 0.093 [0.075, 0.118] | +0.03 [-0.14, +0.20] | 1.88 [1.30, 3.14] | 0.031 (0.009) | 0.997 [0.994, 0.999] |
| M4 · + pitch | 23.0 | 0.0220 [0.0180, 0.0265] | 0.910 | 0.094 [0.076, 0.119] | +0.03 [-0.14, +0.20] | 1.86 [1.29, 3.06] | 0.031 (0.009) | 0.997 [0.994, 0.999] |
| M5 · + people | 26.0 | 0.0222 [0.0182, 0.0266] | 0.910 | 0.094 [0.076, 0.119] | +0.03 [-0.14, +0.20] | 1.85 [1.29, 3.03] | 0.032 (0.009) | 0.997 [0.994, 0.999] |
| GAM · location | 14.8 | 0.0107 [0.0070, 0.0144] | 0.957 | 0.035 [0.024, 0.048] | -0.33 [-0.83, +0.17] | 0.98 [0.81, 1.36] | 0.003 (0.003) | 0.999 [0.999, 1.000] |
| GAM · everything | 31.8 | 0.0114 [0.0077, 0.0155] | 0.953 | 0.038 [0.025, 0.051] | -0.20 [-0.75, +0.35] | 0.90 [0.75, 1.21] | 0.002 (0.002) | 0.999 [0.999, 1.000] |

**Where decisions get made.** Most challenges are easy calls that every model gets right. The MDP's challenge-or-hold choice turns on the uncertain ones, so: among the validation challenges each model put between 20% and 80%, what did it predict on average, and how many were actually overturned?

| model | challenges in 20–80% | mean predicted | actually overturned [95%] |
|---|---:|---:|---|
| M1 · distance | 184 | 49.7% | 47.8% [40.7%, 55.0%] |
| M2 · location | 172 | 51.5% | 52.3% [44.9%, 59.7%] |
| M3 · + situation | 176 | 51.4% | 52.3% [44.9%, 59.5%] |
| M4 · + pitch | 174 | 51.0% | 52.3% [44.9%, 59.6%] |
| M5 · + people | 173 | 51.2% | 52.6% [45.2%, 59.9%] |
| GAM · location | 41 | 52.5% | 48.8% [34.3%, 63.5%] |
| GAM · everything | 40 | 48.2% | 50.0% [35.2%, 64.8%] |

## 5. What each block adds

Two different questions, answered side by side. The **likelihood-ratio test** asks whether the bigger model fits the *training* months better than chance would allow. The **validation change** asks whether it predicts *August* better. A block can pass the first and fail the second — that's a pattern that didn't carry forward, and it's the second question that matters. Negative Δ = the bigger model is better.

| step | params added | LR test (train) | Δ Brier (validation) [95%] | Δ log loss [95%] | verdict |
|---|---:|---|---|---|---|
| + edge & call direction (M2 vs M1) | +7.0 | χ²=3.8, df=7, p=0.81 | +0.0010 [-0.0001, +0.0024] | +0.0020 [-0.0014, +0.0063] | no detectable difference |
| + situation (M3 vs M2) | +6.0 | χ²=1.2, df=6, p=0.98 | +0.0000 [-0.0002, +0.0002] | +0.0000 [-0.0006, +0.0006] | no detectable difference |
| + pitch (M4 vs M3) | +8.0 | χ²=1.5, df=8, p=0.99 | +0.0003 [-0.0000, +0.0006] | +0.0006 [-0.0001, +0.0014] | no detectable difference |
| + people (M5 vs M4) | +3.0 | χ²=-0.6, df=3, p=1 | +0.0002 [+0.0000, +0.0004] | +0.0005 [+0.0000, +0.0009] | worse on new data |
| curve instead of line (location) (GAM-loc vs M2) | +5.8 | not nested | -0.0111 [-0.0132, -0.0089] | -0.0578 [-0.0801, -0.0424] | better on new data |
| curve instead of line (everything) (GAM-full vs M5) | +5.8 | not nested | -0.0108 [-0.0130, -0.0085] | -0.0565 [-0.0781, -0.0405] | better on new data |
| everything vs location only (M5 vs M2) | +17.0 | χ²=2.0, df=17, p=1 | +0.0005 [+0.0000, +0.0009] | +0.0011 [-0.0001, +0.0022] | worse on new data |

For scale: distance alone (M1 vs M0) changes Brier by -0.2250. Everything after that is fine-tuning.

## 6. The location model, in inches

For each edge and each direction of call: where the overturn chance crosses 50% ("the coin-flip point", in inches past our computed edge, + = on the wrong side of the call), and how many inches it takes to go from 25% to 75% ("the blur"). Everything else is held at typical values.

**M2 · location**

| edge | challenge | coin-flip point (in.) [95%] | 25→75% blur (in.) | chance right at our edge |
|---|---|---|---:|---:|
| side | called strike (batter challenges) | -0.04 [-0.09, +0.02] | 0.44 | 54% |
| side | called ball (fielder challenges) | +0.02 [-0.02, +0.07] | 0.44 | 47% |
| top | called strike (batter challenges) | -0.05 [-0.12, +0.01] | 0.17 | 67% |
| top | called ball (fielder challenges) | +0.00 [-0.05, +0.06] | 0.17 | 48% |
| bottom | called strike (batter challenges) | -0.02 [-0.09, +0.04] | 0.51 | 52% |
| bottom | called ball (fielder challenges) | +0.02 [-0.04, +0.08] | 0.51 | 48% |

**GAM · location**

| edge | challenge | coin-flip point (in.) [95%] | 25→75% blur (in.) | chance right at our edge |
|---|---|---|---:|---:|
| side | called strike (batter challenges) | -0.00 [-0.02, +0.02] | 0.06 | 51% |
| side | called ball (fielder challenges) | +0.00 [-0.01, +0.02] | 0.06 | 47% |
| top | called strike (batter challenges) | -0.06 [-0.12, +0.01] | 0.17 | 68% |
| top | called ball (fielder challenges) | +0.00 [-0.06, +0.06] | 0.17 | 49% |
| bottom | called strike (batter challenges) | +0.01 [-0.02, +0.04] | 0.11 | 43% |
| bottom | called ball (fielder challenges) | +0.01 [-0.02, +0.04] | 0.11 | 46% |

How to read the two rows for each edge (details in WEEK7.md):

- **Both coin-flip points below zero** → challenges succeed before the pitch even reaches our computed edge: the challengers know something the measured location doesn't (or the tracking puts the ball slightly differently from what ABS used).
- **The two points far apart, in opposite directions** → the ABS edge isn't where our height-based zone puts it. Half the gap is the size of the shift.

Sign of the half-gap: **+** means ABS's zone reaches further out than ours at that edge (it calls strikes on pitches our zone calls balls); **−** means it stops short.

| edge (M2) | midpoint of the two coin-flip points | half the gap (strike − ball) |
|---|---:|---:|
| side | -0.01 | -0.03 |
| top | -0.02 | -0.03 |
| bottom | -0.00 | -0.02 |

**GAM smoothness.** λ = 0.001, chosen by REML (restricted maximum likelihood, as in R's mgcv) on the training months. Effective degrees of freedom per curve (1 = the data chose a straight line; more = real bends): side: 4.1, bottom: 2.9, top: 1.8.

## 7. Everything else, at a fixed location (M5)

Odds ratios from the full logistic model, per a meaningful unit. ABS only sees the ball, so **none of these can change how ABS rules a given pitch.** An effect here means the challenges that got made in that situation were different — a statement about who chose to challenge, not about the system. Read them as selection effects.

| term | per | odds ratio [95%] | p |
|---|---|---|---:|
| `balls` | per extra ball in the count | 0.94 [0.81, 1.09] | 0.41 |
| `strikes` | per extra strike in the count | 1.03 [0.86, 1.24] | 0.72 |
| `outs_when_up` | per extra out | 1.00 [0.86, 1.16] | 0.99 |
| `inning` | per inning later | 0.99 [0.94, 1.04] | 0.71 |
| `log_leverage` | per doubling of leverage | 1.03 [0.95, 1.12] | 0.45 |
| `chal_left` | per extra challenge in hand | 0.90 [0.65, 1.26] | 0.54 |
| `release_speed` | per 5 mph faster | 0.96 [0.79, 1.17] | 0.7 |
| `pfx_x` | per 6 in. more horizontal break | 1.03 [0.95, 1.11] | 0.52 |
| `pfx_z` | per 6 in. more vertical break | 1.03 [0.89, 1.18] | 0.72 |
| `batter_height_ft` | per inch taller batter | 0.98 [0.93, 1.03] | 0.39 |
| `pitcher_challenge` | pitcher (vs catcher) made the challenge | 0.92 [0.29, 2.90] | 0.88 |
| `ump_acc_pp` | per point more accurate umpire | 0.92 [0.77, 1.11] | 0.38 |
| `catcher_acc_pp` | per point higher catcher-game accuracy | 1.06 [0.79, 1.42] | 0.71 |
| `pitch_group=breaking` | pitch breaking (vs fastball) | 0.88 [0.52, 1.48] | 0.62 |
| `pitch_group=offspeed` | pitch offspeed (vs fastball) | 0.87 [0.52, 1.44] | 0.58 |
| `stand=L` | batter hand L (vs R) | 1.03 [0.80, 1.32] | 0.83 |
| `p_throws=L` | pitcher hand L (vs R) | 0.96 [0.71, 1.30] | 0.77 |

## 8. Subgroup calibration (GAM · location, validation)

Overall calibration can hide a model that's too high for one kind of challenge and too low for another. Each row: what the model predicted on average for that slice, and what happened. A 95% interval that excludes the prediction is worth a look; with slices this small, most won't. Week 9 does this properly on more data.

| slice | n | predicted | observed [95%] | gap (pts) |
|---|---:|---:|---|---:|
| Who challenged: batter | 611 | 61.7% | 61.0% [57.1%, 64.8%] | -0.6 |
| Who challenged: catcher | 793 | 53.9% | 53.7% [50.2%, 57.2%] | -0.1 |
| Who challenged: pitcher | 27 | 50.2% | 48.1% [30.7%, 66.0%] | -2.1 |
| Edge · call: side · strike | 314 | 63.3% | 63.1% [57.6%, 68.2%] | -0.2 |
| Edge · call: side · ball | 419 | 53.0% | 52.7% [48.0%, 57.5%] | -0.3 |
| Edge · call: top · strike | 91 | 68.4% | 67.0% [56.9%, 75.8%] | -1.4 |
| Edge · call: top · ball | 150 | 44.4% | 42.7% [35.0%, 50.7%] | -1.7 |
| Edge · call: bottom · strike | 206 | 56.2% | 55.3% [48.5%, 62.0%] | -0.9 |
| Edge · call: bottom · ball | 255 | 59.8% | 60.8% [54.7%, 66.6%] | +1.0 |
| Count: 0–1 strikes | 1245 | 58.8% | 58.5% [55.7%, 61.2%] | -0.4 |
| Count: 2 strikes | 190 | 44.8% | 44.7% [37.8%, 51.8%] | -0.1 |
| Challenges in hand: 1 left | 293 | 55.1% | 54.6% [48.9%, 60.2%] | -0.5 |
| Challenges in hand: 2 left | 1142 | 57.5% | 57.2% [54.3%, 60.0%] | -0.3 |
| Inning: 1–3 | 396 | 68.1% | 67.7% [62.9%, 72.1%] | -0.4 |
| Inning: 4–6 | 501 | 58.7% | 58.3% [53.9%, 62.5%] | -0.4 |
| Inning: 7–9 | 497 | 46.3% | 46.1% [41.7%, 50.5%] | -0.2 |
| Inning: 10+ | 41 | 59.0% | 58.5% [43.4%, 72.2%] | -0.5 |
| Leverage: low (<0.85) | 639 | 56.7% | 56.5% [52.6%, 60.3%] | -0.2 |
| Leverage: medium | 557 | 59.2% | 58.5% [54.4%, 62.5%] | -0.6 |
| Leverage: high (>2) | 227 | 51.9% | 52.0% [45.5%, 58.4%] | +0.1 |

**By month** (training months are in-sample, so small gaps are expected there; a steady drift across months is the thing to look for — it would mean the same pitch location was overturned at a different rate as the season went on):

| month | window | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| 2026-03 | train (in-sample) | 201 | 60.1% | 59.7% [52.8%, 66.2%] |
| 2026-04 | train (in-sample) | 1226 | 55.8% | 56.0% [53.2%, 58.7%] |
| 2026-05 | train (in-sample) | 1264 | 54.1% | 53.7% [51.0%, 56.5%] |
| 2026-06 | train (in-sample) | 1282 | 55.6% | 56.1% [53.4%, 58.8%] |
| 2026-07 | train (in-sample) | 1246 | 56.1% | 55.9% [53.2%, 58.7%] |
| 2026-08 | validate | 1435 | 57.0% | 56.7% [54.1%, 59.2%] |

## 9. The baseline to beat

**Rule (fixed before looking):** take the model with the lowest validation Brier; then, going from simplest to most complex, choose the first model whose paired 95% interval for (its Brier − the best Brier) reaches zero. A gain August can't confirm doesn't count, and ties go to the simpler model.

| model | Brier − best [95%] | indistinguishable from best? |
|---|---|---|
| M1 · distance | +0.0101 [+0.0078, +0.0122] | no |
| M2 · location | +0.0111 [+0.0089, +0.0132] | no |
| GAM · location | — (this is the best) | — |
| M3 · + situation | +0.0111 [+0.0089, +0.0132] | no |
| M4 · + pitch | +0.0114 [+0.0091, +0.0135] | no |
| M5 · + people | +0.0116 [+0.0092, +0.0138] | no |
| GAM · everything | +0.0008 [-0.0001, +0.0019] | yes |

### Baseline to beat: **GAM · location**

Validation Brier 0.0107 [0.0070, 0.0144], log loss 0.035 [0.024, 0.048], calibration intercept -0.33 [-0.83, +0.17], slope 0.98 [0.81, 1.36], AUC 0.999. (Lowest Brier overall: GAM · location.)

Its calibration intercept interval contains 0 and its slope interval contains 1: no detectable miscalibration on August.

**A Week 8 model beats it** only if its paired-bootstrap 95% interval for (its Brier − GAM-loc's Brier) lies entirely below zero on the same validation games, and its calibration intercept and slope are no worse.

**Location-only candidate for the MDP's p(s): GAM · location** (same rule, restricted to models that use only what ABS sees). Whether p(s) should also use the situation/pitch/people blocks is a decision for you and your supervisor — see WEEK7.md, "Location-only or everything?".

## 10. Most confidently wrong validation predictions — check by hand

The ten validation challenges M2 · location was most surprised by. M2 is used here on purpose: a straight line in log-odds is the model most shocked by a result far from the edge, so it makes data problems stand out (a flexible GAM can learn a small "floor" from a few bad records and hide them). Very confident misses are usually data problems — a challenge linked to the wrong pitch, a wrong height, a mislabelled outcome — and they are what drags log loss up. Look each one up on Baseball Savant.

| game_pk | date | inning | count | challenger | call | edge | in. against call | p(overturn) | result |
|---|---|---:|---|---|---|---|---:|---:|---|
| 823514 | 2026-08-08 | 5 | 1-2 | unknown | ball | bottom | -16.19 | 0.000 | overturned |
| 824071 | 2026-08-23 | 5 | 0-0 | catcher | ball | top | +0.21 | 0.941 | stood |
| 822689 | 2026-08-31 | 6 | 1-1 | catcher | ball | top | +0.16 | 0.889 | stood |
| 823751 | 2026-08-09 | 5 | 1-1 | catcher | ball | top | +0.15 | 0.874 | stood |
| 823515 | 2026-08-07 | 9 | 2-0 | catcher | ball | top | -0.11 | 0.179 | overturned |
| 823102 | 2026-08-08 | 7 | 0-1 | batter | strike | top | -0.16 | 0.207 | overturned |
| 823911 | 2026-08-21 | 3 | 0-0 | batter | strike | top | +0.04 | 0.784 | stood |
| 823516 | 2026-08-05 | 3 | 2-1 | catcher | ball | top | +0.07 | 0.705 | stood |
| 823010 | 2026-08-30 | 8 | 0-0 | batter | strike | top | +0.01 | 0.702 | stood |
| 823265 | 2026-08-10 | 6 | 2-2 | catcher | ball | top | +0.07 | 0.695 | stood |

## 11. Fitting notes

- **GAM · location**: chosen λ=0.001 is the smallest in the grid — REML wanted as little smoothing as the grid allows, i.e. a curve that bends very sharply (a near-step at the zone edge). The fit is as flexible as its knots let it be; widen `lambdas` downward or add knots near zero to let it bend further
- **GAM · everything**: chosen λ=0.001 is the smallest in the grid — REML wanted as little smoothing as the grid allows, i.e. a curve that bends very sharply (a near-step at the zone edge). The fit is as flexible as its knots let it be; widen `lambdas` downward or add knots near zero to let it bend further
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
