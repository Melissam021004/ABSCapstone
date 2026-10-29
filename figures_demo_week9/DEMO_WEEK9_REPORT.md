# Week 9 — Calibrate and Lock

Generated 2026-10-20 18:54 by `scripts/17_week9_calibrate_lock.py` · cohort `c380922df66bdf57` · 5-fold cross-fitting · 1000 game-level bootstrap resamples

> **DEMO DATA — every number below comes from the synthetic fixture.**

> Read this next to `WEEK9.md`, which explains every section.

## The locked model

**p(s) 2026.10.20-ff73ea58 = M1 · distance + uncalibrated**  ·  needs: `plate_x`, `plate_z`, `batter_height_ft`, `called_strike`

- Location only, per the Week 7 decision: ABS rules on the ball's position alone, the policy will be asked about pitches nobody challenged, and features describing who chose to challenge would make a policy that judges challenge decisions circular.
- M1 · distance is Week 8's location only candidate: the simplest model whose August Brier can't be told apart from the best model with the same features.
- No calibrator earned its place: neither Platt scaling nor isotonic regression improved the cross-fitted August Brier by more than the noise, so the model's own probabilities are used (fewer moving parts).
- Subgroups: 0 of 19 testable slices are off by more than chance (about 1.0 would be by luck alone).
- September: **sealed**. Run `--final-test` once to score it.

## 0. Pre-flight

- Recorded call on challenged pitches: recorded call matches who challenged on 100.0% of 2,236 upheld and 100.0% of 2,427 overturned challenges; verdict: ORIGINAL umpire call recorded — no repair needed.
- Same cohort as Week 8 (`c380922df66bdf57`) — checked.
- Train 3,211 challenges (March–July) · **calibrate** 793 (August) · test 659 (September, sealed).

## 1. Calibration — August, cross-fitted

Every number below is out-of-sample for the calibrator: each August game was calibrated by a map fitted on the other four-fifths of August's games.

| candidate | method | Brier [95%] | log loss | ECE (noise) | cal. intercept | cal. slope | 20–80% band: predicted → observed | Δ Brier vs uncalibrated [95%] |
|---|---|---|---:|---|---|---|---|---|
| M1 · distance | uncalibrated ★ | 0.0439 [0.0342, 0.0549] | 0.1429 | 0.013 (0.011) | -0.13 [-0.51, +0.24] | 0.87 [0.74, 1.07] | 55% → 57% (n=83) | — |
| M1 · distance | Platt scaling | 0.0442 [0.0349, 0.0550] | 0.1445 | 0.013 (0.012) | +0.01 [-0.33, +0.35] | 0.97 [0.82, 1.21] | 56% → 56% (n=94) | +0.0002 [-0.0007, +0.0010] |
| M1 · distance | isotonic | 0.0445 [0.0347, 0.0556] | 0.1513 | 0.016 (0.011) | -0.01 [-0.37, +0.38] | 0.85 [0.72, 1.06] | 57% → 55% (n=95) | +0.0005 [-0.0019, +0.0030] |
| M3 · + situation | uncalibrated ★ | 0.0447 [0.0346, 0.0560] | 0.1432 | 0.018 (0.010) | -0.17 [-0.55, +0.22] | 0.85 [0.73, 1.02] | 55% → 60% (n=82) | — |
| M3 · + situation | Platt scaling | 0.0449 [0.0355, 0.0556] | 0.1448 | 0.017 (0.012) | +0.00 [-0.33, +0.35] | 0.97 [0.83, 1.18] | 55% → 58% (n=101) | +0.0002 [-0.0010, +0.0011] |
| M3 · + situation | isotonic | 0.0447 [0.0353, 0.0557] | 0.1517 | 0.016 (0.011) | +0.00 [-0.36, +0.39] | 0.85 [0.73, 1.07] | 65% → 69% (n=100) | +0.0000 [-0.0034, +0.0032] |

★ = chosen by the rule fixed before looking: **the simplest method (none → Platt → isotonic) whose cross-fitted Brier can't be told apart from the best** (paired bootstrap over games). The final calibrator is then refitted on all of August:

- M1 · distance: no calibration — the model's own probabilities
- M3 · + situation: no calibration — the model's own probabilities

## 2. Subgroups — is p(s) right for every kind of challenge?

M1 · distance + uncalibrated, August cross-fitted. A slice is **flagged** when its observed rate's 95% interval excludes the mean prediction (only slices with ≥ 30 challenges). With 19 testable slices, about 1.0 would be flagged by luck alone; **0 were.**

| group | slice | n | predicted | observed [95%] | gap (pts) | |
|---|---|---:|---:|---|---:|---|
| Challenger | batter | 416 | 52.3% | 52.2% [47.4%, 56.9%] | -0.1 |  |
| Challenger | catcher | 272 | 57.9% | 56.6% [50.7%, 62.4%] | -1.3 |  |
| Challenger | pitcher | 105 | 50.5% | 50.5% [41.1%, 59.9%] | -0.0 |  |
| Count | other counts | 547 | 52.5% | 51.2% [47.0%, 55.4%] | -1.3 |  |
| Count | two strikes (not full) | 203 | 58.5% | 59.1% [52.2%, 65.6%] | +0.6 |  |
| Count | three balls (not full) | 23 | 48.1% | 43.5% [25.6%, 63.2%] | -4.6 |  |
| Count | full count (3-2) | 20 | 55.5% | 70.0% [48.1%, 85.5%] | +14.5 |  |
| Challenges left | 1 left | 217 | 50.6% | 47.0% [40.5%, 53.6%] | -3.6 |  |
| Challenges left | 2 left | 576 | 55.2% | 55.9% [51.8%, 59.9%] | +0.7 |  |
| Month | 08 Aug | 793 | 54.0% | 53.5% [50.0%, 56.9%] | -0.5 |  |
| Batter hand | bats R | 793 | 54.0% | 53.5% [50.0%, 56.9%] | -0.5 |  |
| Pitcher hand | throws R | 793 | 54.0% | 53.5% [50.0%, 56.9%] | -0.5 |  |
| Edge | side edge | 484 | 57.0% | 56.4% [52.0%, 60.8%] | -0.6 |  |
| Edge | top edge | 153 | 53.5% | 54.2% [46.3%, 61.9%] | +0.8 |  |
| Edge | bottom edge | 156 | 45.1% | 43.6% [36.1%, 51.4%] | -1.5 |  |
| Inning | 1–3 | 276 | 56.2% | 56.9% [51.0%, 62.6%] | +0.7 |  |
| Inning | 4–6 | 241 | 53.0% | 52.3% [46.0%, 58.5%] | -0.7 |  |
| Inning | 7–9 | 256 | 52.8% | 51.6% [45.5%, 57.6%] | -1.3 |  |
| Inning | extras | 20 | 49.9% | 45.0% [25.8%, 65.8%] | -4.9 |  |
| Leverage | low (< 0.85) | 310 | 55.1% | 56.1% [50.6%, 61.5%] | +1.1 |  |
| Leverage | medium | 367 | 52.5% | 51.8% [46.7%, 56.8%] | -0.7 |  |
| Leverage | high (≥ 2) | 116 | 55.6% | 51.7% [42.7%, 60.6%] | -3.9 |  |

## 3. Season drift — the same model, month by month

Training months are scored **out-of-fold**: the model was refitted 5 times with the same settings, each time leaving out a fifth of the training games, and each game was predicted by the fit that never saw it. August is cross-fitted as above.

| month | window | n | predicted | observed [95%] | gap (pts) |
|---|---|---:|---:|---|---:|
| Mar | train (out-of-fold) | 155 | 46.7% | 43.9% [36.3%, 51.7%] | -2.9 |
| Apr | train (out-of-fold) | 774 | 52.4% | 51.6% [48.0%, 55.1%] | -0.8 |
| May | train (out-of-fold) | 761 | 53.4% | 53.1% [49.5%, 56.6%] | -0.3 |
| Jun | train (out-of-fold) | 755 | 52.5% | 54.0% [50.5%, 57.6%] | +1.5 |
| Jul | train (out-of-fold) | 766 | 50.1% | 50.4% [46.9%, 53.9%] | +0.3 |
| Aug | August | 793 | 54.0% | 53.5% [50.0%, 56.9%] | -0.5 |

## 4. The production decision

**p(s) = M1 · distance + uncalibrated.**

- Location only, per the Week 7 decision: ABS rules on the ball's position alone, the policy will be asked about pitches nobody challenged, and features describing who chose to challenge would make a policy that judges challenge decisions circular.
- M1 · distance is Week 8's location only candidate: the simplest model whose August Brier can't be told apart from the best model with the same features.
- No calibrator earned its place: neither Platt scaling nor isotonic regression improved the cross-fitted August Brier by more than the noise, so the model's own probabilities are used (fewer moving parts).
- 0 of 19 testable slices are off by more than chance (about 1.0 would be by luck alone).

To lock the other candidate instead: `--production full` (re-run before `--final-test`).

## 5. The locked bundle (`models/production/`)

- version **2026.10.20-ff73ea58**, created 2026-10-20T18:54:45
- model `M1` (logistic), features: `dist_against_call`
- calibration: no calibration — the model's own probabilities
- self-check: reloading the bundle through `src/abs_predict.py` and predicting August from the raw columns reproduces the in-memory model to within 4.0e-15
- every file is hashed in `manifest.json`; `load_production()` refuses a bundle that changed. Library versions are pinned in `requirements-lock.txt`.

```python
from src.abs_predict import load_production
pm = load_production()
pm.predict({'plate_x': 0.85, 'plate_z': 2.4, 'batter_height_ft': 6.0, 'called_strike': True})
```

## 6. Figures (`figures/week9/`)

- `w9_01_calibration.png` — reliability before/after each calibration method (cross-fitted)
- `w9_02_methods.png` — the three methods' scores and their difference from uncalibrated
- `w9_03_subgroups.png` — observed − predicted for every slice
- `w9_04_months.png` — season drift: out-of-fold training months, August (and September)
- `w9_05_surface.png` — the locked p(s) over the strike zone, for a called strike and a called ball

The model card is `docs/MODEL_CARD.md` (a copy travels inside the bundle).