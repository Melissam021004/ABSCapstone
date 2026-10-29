# Week 9 — The one look at September

Opened 2026-10-29T16:03:39 for p(s) **2026.10.29-b81183b4** · 1,243 challenges · 1000 game-level bootstrap resamples

Nothing was chosen after this point. These are the numbers that go in the report.

## 1. Every model, the same September games

| model | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | ECE | AUC |
|---|---|---:|---|---|---:|---:|
| **p(s) 2026.10.29-b81183b4 (locked)** | 0.0087 [0.0055, 0.0127] | 0.040 | -0.11 [-0.68, +0.39] | 0.90 [0.64, 2.00] | 0.003 | 0.998 |
| GAM · location, uncalibrated | 0.0087 [0.0055, 0.0127] | 0.040 | -0.11 [-0.68, +0.39] | 0.90 [0.64, 2.00] | 0.003 | 0.998 |
| M0 · base rate | 0.2452 [0.2421, 0.2483] | 0.683 | +0.06 [-0.05, +0.17] | — | 0.015 | 0.500 |
| M1 · distance | 0.0220 [0.0190, 0.0255] | 0.100 | +0.01 [-0.16, +0.17] | 2.50 [1.51, 9.36] | 0.046 | 0.998 |
| M2 · location | 0.0230 [0.0195, 0.0271] | 0.101 | +0.02 [-0.17, +0.19] | 2.17 [1.37, 5.63] | 0.044 | 0.997 |
| M3 · + situation | 0.0231 [0.0197, 0.0272] | 0.101 | +0.02 [-0.18, +0.18] | 2.16 [1.37, 5.31] | 0.044 | 0.997 |
| M4 · + pitch | 0.0232 [0.0197, 0.0272] | 0.101 | +0.01 [-0.19, +0.18] | 2.16 [1.37, 5.08] | 0.044 | 0.997 |
| M5 · + people | 0.0234 [0.0198, 0.0275] | 0.102 | +0.01 [-0.19, +0.18] | 2.16 [1.37, 5.11] | 0.045 | 0.997 |
| GAM · location ★ | 0.0087 [0.0055, 0.0127] | 0.040 | -0.11 [-0.68, +0.39] | 0.90 [0.64, 2.00] | 0.003 | 0.998 |
| GAM · everything | 0.0086 [0.0053, 0.0127] | 0.040 | -0.04 [-0.62, +0.47] | 0.88 [0.63, 1.98] | 0.002 | 0.998 |
| CatBoost · location | 0.0067 [0.0035, 0.0105] | 0.028 | -0.11 [-0.68, +0.47] | 1.10 [0.91, 1.89] | 0.003 | 0.999 |
| CatBoost · everything | 0.0062 [0.0030, 0.0103] | 0.026 | -0.27 [-0.82, +0.28] | 1.16 [0.97, 2.05] | 0.004 | 1.000 |
| LightGBM · location | 0.0071 [0.0039, 0.0112] | 0.027 | -0.38 [-0.96, +0.23] | 1.09 [0.89, 2.16] | 0.004 | 0.999 |
| LightGBM · everything | 0.0059 [0.0030, 0.0096] | 0.026 | -0.07 [-0.54, +0.40] | 1.25 [1.05, 2.20] | 0.003 | 1.000 |

★ = Week 8's baseline to beat.

## 2. Did the August conclusions hold?

- **Calibration:** August chose no calibrator, so the locked p(s) IS the model's own output. Its September calibration slope is 0.90 [0.64, 2.00] and intercept -0.11 [-0.68, +0.39] — if either interval misses its ideal (slope 1, intercept 0), say so in the report: it is the honest finding, and it is too late to change.
- **Everything vs location only:** GAM · everything minus GAM · location: Δ Brier -0.0001 [-0.0010, +0.0007].
- CatBoost · location minus the baseline (GAM · location): Δ Brier -0.0020 [-0.0037, -0.0004] — better.
- CatBoost · everything minus the baseline (GAM · location): Δ Brier -0.0025 [-0.0041, -0.0010] — better.
- LightGBM · location minus the baseline (GAM · location): Δ Brier -0.0017 [-0.0040, +0.0003] — can't tell apart.
- LightGBM · everything minus the baseline (GAM · location): Δ Brier -0.0028 [-0.0043, -0.0016] — better.
- **Where decisions are made** (p(s) between 20% and 80%): predicted 50%, observed 51% [36%, 66%], n = 41.

## 3. Subgroups on September

0 of 22 testable slices flagged (≈ 1.1 expected by chance).

| group | slice | n | predicted | observed [95%] | gap (pts) | |
|---|---|---:|---:|---|---:|---|
| Challenger | batter | 511 | 59.4% | 59.3% [55.0%, 63.5%] | -0.1 |  |
| Challenger | catcher | 713 | 56.4% | 56.1% [52.4%, 59.7%] | -0.3 |  |
| Challenger | pitcher | 17 | 35.8% | 35.3% [17.3%, 58.7%] | -0.5 |  |
| Challenger | other | 2 | 0.0% | 50.0% [9.5%, 90.5%] | +50.0 |  |
| Count | other counts | 1012 | 60.3% | 60.1% [57.0%, 63.1%] | -0.2 |  |
| Count | two strikes (not full) | 177 | 39.7% | 40.1% [33.2%, 47.5%] | +0.4 |  |
| Count | three balls (not full) | 54 | 57.3% | 57.4% [44.2%, 69.7%] | +0.1 |  |
| Challenges left | 1 left | 233 | 53.0% | 52.4% [46.0%, 58.7%] | -0.7 |  |
| Challenges left | 2 left | 1010 | 58.2% | 58.2% [55.2%, 61.2%] | +0.0 |  |
| Month | 09 Sep | 1243 | 57.2% | 57.1% [54.4%, 59.8%] | -0.1 |  |
| Batter hand | bats L | 589 | 57.8% | 57.7% [53.7%, 61.7%] | -0.1 |  |
| Batter hand | bats R | 654 | 56.7% | 56.6% [52.7%, 60.3%] | -0.1 |  |
| Pitcher hand | throws L | 395 | 59.7% | 59.5% [54.6%, 64.2%] | -0.2 |  |
| Pitcher hand | throws R | 848 | 56.1% | 56.0% [52.7%, 59.3%] | -0.1 |  |
| Edge | side edge | 613 | 57.3% | 57.6% [53.6%, 61.4%] | +0.2 |  |
| Edge | top edge | 207 | 59.9% | 58.5% [51.6%, 65.0%] | -1.4 |  |
| Edge | bottom edge | 423 | 55.8% | 55.8% [51.0%, 60.5%] | +0.0 |  |
| Inning | 1–3 | 385 | 64.8% | 64.7% [59.8%, 69.3%] | -0.2 |  |
| Inning | 4–6 | 388 | 58.1% | 58.5% [53.5%, 63.3%] | +0.4 |  |
| Inning | 7–9 | 432 | 50.2% | 49.8% [45.1%, 54.5%] | -0.4 |  |
| Inning | extras | 38 | 51.1% | 50.0% [34.8%, 65.2%] | -1.1 |  |
| Leverage | low (< 0.85) | 570 | 59.8% | 59.6% [55.6%, 63.6%] | -0.1 |  |
| Leverage | medium | 484 | 56.6% | 56.6% [52.2%, 61.0%] | -0.0 |  |
| Leverage | high (≥ 2) | 176 | 48.0% | 47.7% [40.5%, 55.1%] | -0.3 |  |

## 4. Figures

- `figures/week9/w9_06_test.png` — every model on September, and the locked p(s)'s reliability diagram
- `figures/week9/w9_03_subgroups.png`, `w9_04_months.png` — redrawn with September added