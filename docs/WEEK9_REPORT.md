# Week 9 — Calibrate and Lock

Generated 2026-10-29 16:01 by `scripts/17_week9_calibrate_lock.py` · cohort `419af01ad9eee03f` · 5-fold cross-fitting · 1000 game-level bootstrap resamples

> Read this next to `WEEK9.md`, which explains every section.

## The locked model

**p(s) 2026.10.29-b81183b4 = GAM · location + uncalibrated**  ·  needs: `plate_x`, `plate_z`, `batter_height_ft`, `called_strike`

- Location only, per the Week 7 decision: ABS rules on the ball's position alone, the policy will be asked about pitches nobody challenged, and features describing who chose to challenge would make a policy that judges challenge decisions circular.
- GAM · location is Week 8's location only candidate: the simplest model whose August Brier can't be told apart from the best model with the same features.
- No calibrator earned its place: neither Platt scaling nor isotonic regression improved the cross-fitted August Brier by more than the noise, so the model's own probabilities are used (fewer moving parts).
- Subgroups: 0 of 22 testable slices are off by more than chance (about 1.1 would be by luck alone).
- September: **sealed**. Run `--final-test` once to score it.

## 0. Pre-flight

- Recorded call on challenged pitches: recorded call matches who challenged on 100.0% of 3,453 upheld and 0.0% of 4,417 overturned challenges; verdict: FINAL (post-review) call recorded — repaired from who challenged (4,417 challenged pitches corrected).
- Same cohort as Week 8 (`419af01ad9eee03f`) — checked.
- Train 5,219 challenges (March–July) · **calibrate** 1,435 (August) · test 1,243 (September, sealed).

## 1. Calibration — August, cross-fitted

Every number below is out-of-sample for the calibrator: each August game was calibrated by a map fitted on the other four-fifths of August's games.

| candidate | method | Brier [95%] | log loss | ECE (noise) | cal. intercept | cal. slope | 20–80% band: predicted → observed | Δ Brier vs uncalibrated [95%] |
|---|---|---|---:|---|---|---|---|---|
| GAM · location | uncalibrated ★ | 0.0107 [0.0070, 0.0144] | 0.0354 | 0.003 (0.003) | -0.33 [-0.83, +0.17] | 0.98 [0.81, 1.36] | 52% → 49% (n=41) | — |
| GAM · location | Platt scaling | 0.0109 [0.0073, 0.0147] | 0.0361 | 0.001 (0.003) | -0.01 [-0.51, +0.50] | 0.96 [0.79, 1.34] | 52% → 54% (n=39) | +0.0002 [-0.0004, +0.0010] |
| GAM · location | isotonic | 0.0119 [0.0081, 0.0159] | 0.0419 | 0.000 (0.002) | +0.02 [-0.50, +0.59] | 0.91 [0.81, 1.20] | 49% → 49% (n=45) | +0.0012 [+0.0004, +0.0021] |
| GAM · everything | uncalibrated ★ | 0.0114 [0.0077, 0.0155] | 0.0376 | 0.002 (0.003) | -0.20 [-0.75, +0.35] | 0.90 [0.75, 1.21] | 48% → 50% (n=40) | — |
| GAM · everything | Platt scaling | 0.0116 [0.0079, 0.0157] | 0.0387 | 0.000 (0.003) | -0.01 [-0.51, +0.50] | 0.96 [0.80, 1.29] | 49% → 49% (n=43) | +0.0002 [-0.0003, +0.0007] |
| GAM · everything | isotonic | 0.0121 [0.0082, 0.0164] | 0.0435 | 0.000 (0.002) | +0.00 [-0.52, +0.55] | 0.92 [0.81, 1.19] | 54% → 53% (n=38) | +0.0006 [-0.0006, +0.0018] |

★ = chosen by the rule fixed before looking: **the simplest method (none → Platt → isotonic) whose cross-fitted Brier can't be told apart from the best** (paired bootstrap over games). The final calibrator is then refitted on all of August:

- GAM · location: no calibration — the model's own probabilities
- GAM · everything: no calibration — the model's own probabilities

## 2. Subgroups — is p(s) right for every kind of challenge?

GAM · location + uncalibrated, August cross-fitted. A slice is **flagged** when its observed rate's 95% interval excludes the mean prediction (only slices with ≥ 30 challenges). With 22 testable slices, about 1.1 would be flagged by luck alone; **0 were.**

| group | slice | n | predicted | observed [95%] | gap (pts) | |
|---|---|---:|---:|---|---:|---|
| Challenger | batter | 611 | 61.7% | 61.0% [57.1%, 64.8%] | -0.6 |  |
| Challenger | catcher | 793 | 53.9% | 53.7% [50.2%, 57.2%] | -0.1 |  |
| Challenger | pitcher | 27 | 50.2% | 48.1% [30.7%, 66.0%] | -2.1 |  |
| Challenger | other | 4 | 9.9% | 25.0% [4.6%, 69.9%] | +15.1 |  |
| Count | other counts | 1204 | 58.9% | 58.5% [55.7%, 61.2%] | -0.4 |  |
| Count | two strikes (not full) | 190 | 44.8% | 44.7% [37.8%, 51.8%] | -0.1 |  |
| Count | three balls (not full) | 41 | 57.5% | 58.5% [43.4%, 72.2%] | +1.1 |  |
| Challenges left | 1 left | 293 | 55.1% | 54.6% [48.9%, 60.2%] | -0.5 |  |
| Challenges left | 2 left | 1142 | 57.5% | 57.2% [54.3%, 60.0%] | -0.3 |  |
| Month | 08 Aug | 1435 | 57.0% | 56.7% [54.1%, 59.2%] | -0.3 |  |
| Batter hand | bats L | 739 | 58.3% | 58.3% [54.7%, 61.8%] | -0.0 |  |
| Batter hand | bats R | 696 | 55.6% | 54.9% [51.2%, 58.5%] | -0.7 |  |
| Pitcher hand | throws L | 434 | 56.7% | 56.0% [51.3%, 60.6%] | -0.7 |  |
| Pitcher hand | throws R | 1001 | 57.1% | 56.9% [53.9%, 60.0%] | -0.2 |  |
| Edge | side edge | 733 | 57.4% | 57.2% [53.6%, 60.7%] | -0.2 |  |
| Edge | top edge | 241 | 53.5% | 51.9% [45.6%, 58.1%] | -1.6 |  |
| Edge | bottom edge | 461 | 58.2% | 58.4% [53.8%, 62.8%] | +0.2 |  |
| Inning | 1–3 | 396 | 68.1% | 67.7% [62.9%, 72.1%] | -0.4 |  |
| Inning | 4–6 | 501 | 58.7% | 58.3% [53.9%, 62.5%] | -0.4 |  |
| Inning | 7–9 | 497 | 46.3% | 46.1% [41.7%, 50.5%] | -0.2 |  |
| Inning | extras | 41 | 59.0% | 58.5% [43.4%, 72.2%] | -0.5 |  |
| Leverage | low (< 0.85) | 639 | 56.7% | 56.5% [52.6%, 60.3%] | -0.2 |  |
| Leverage | medium | 557 | 59.2% | 58.5% [54.4%, 62.5%] | -0.6 |  |
| Leverage | high (≥ 2) | 227 | 51.9% | 52.0% [45.5%, 58.4%] | +0.1 |  |

## 3. Season drift — the same model, month by month

Training months are scored **out-of-fold**: the model was refitted 5 times with the same settings, each time leaving out a fifth of the training games, and each game was predicted by the fit that never saw it. August is cross-fitted as above.

| month | window | n | predicted | observed [95%] | gap (pts) |
|---|---|---:|---:|---|---:|
| Mar | train (out-of-fold) | 201 | 60.0% | 59.7% [52.8%, 66.2%] | -0.3 |
| Apr | train (out-of-fold) | 1226 | 55.8% | 56.0% [53.2%, 58.7%] | +0.1 |
| May | train (out-of-fold) | 1264 | 54.0% | 53.7% [51.0%, 56.5%] | -0.3 |
| Jun | train (out-of-fold) | 1282 | 55.7% | 56.1% [53.4%, 58.8%] | +0.4 |
| Jul | train (out-of-fold) | 1246 | 56.1% | 55.9% [53.2%, 58.7%] | -0.2 |
| Aug | August | 1435 | 57.0% | 56.7% [54.1%, 59.2%] | -0.3 |

## 4. The production decision

**p(s) = GAM · location + uncalibrated.**

- Location only, per the Week 7 decision: ABS rules on the ball's position alone, the policy will be asked about pitches nobody challenged, and features describing who chose to challenge would make a policy that judges challenge decisions circular.
- GAM · location is Week 8's location only candidate: the simplest model whose August Brier can't be told apart from the best model with the same features.
- No calibrator earned its place: neither Platt scaling nor isotonic regression improved the cross-fitted August Brier by more than the noise, so the model's own probabilities are used (fewer moving parts).
- 0 of 22 testable slices are off by more than chance (about 1.1 would be by luck alone).

To lock the other candidate instead: `--production full` (re-run before `--final-test`).

## 5. The locked bundle (`models/production/`)

- version **2026.10.29-b81183b4**, created 2026-10-29T16:01:08
- model `GAM-loc` (gam), features: `edge`, `call_dir`, `dist_against_call`
- calibration: no calibration — the model's own probabilities
- self-check: reloading the bundle through `src/abs_predict.py` and predicting August from the raw columns reproduces the in-memory model to within 0.0e+00
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