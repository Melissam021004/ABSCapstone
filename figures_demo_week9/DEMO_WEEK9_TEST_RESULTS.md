# Week 9 — The one look at September

Opened 2026-10-20T18:55:14 for p(s) **2026.10.20-ff73ea58** · 659 challenges · 1000 game-level bootstrap resamples

> **DEMO DATA — every number below comes from the synthetic fixture.**

Nothing was chosen after this point. These are the numbers that go in the report.

## 1. Every model, the same September games

| model | Brier [95%] | log loss | cal. intercept [95%] | cal. slope [95%] | ECE | AUC |
|---|---|---:|---|---|---:|---:|
| **p(s) 2026.10.20-ff73ea58 (locked)** | 0.0360 [0.0263, 0.0461] | 0.118 | -0.03 [-0.40, +0.33] | 1.06 [0.87, 1.36] | 0.008 | 0.991 |
| M1 · distance, uncalibrated | 0.0360 [0.0263, 0.0461] | 0.118 | -0.03 [-0.40, +0.33] | 1.06 [0.87, 1.36] | 0.008 | 0.991 |
| M0 · base rate | 0.2499 [0.2483, 0.2515] | 0.693 | -0.02 [-0.20, +0.15] | — | 0.006 | 0.500 |
| M1 · distance ★ | 0.0360 [0.0263, 0.0461] | 0.118 | -0.03 [-0.40, +0.33] | 1.06 [0.87, 1.36] | 0.008 | 0.991 |
| M2 · location | 0.0364 [0.0267, 0.0473] | 0.120 | -0.01 [-0.41, +0.37] | 1.00 [0.82, 1.28] | 0.002 | 0.991 |
| M3 · + situation | 0.0356 [0.0261, 0.0459] | 0.118 | -0.03 [-0.43, +0.36] | 1.01 [0.83, 1.30] | 0.003 | 0.991 |
| M4 · + pitch | 0.0358 [0.0264, 0.0460] | 0.118 | -0.04 [-0.44, +0.34] | 1.01 [0.83, 1.30] | 0.003 | 0.991 |
| M5 · + people | 0.0358 [0.0265, 0.0462] | 0.119 | -0.08 [-0.49, +0.30] | 1.01 [0.83, 1.28] | 0.005 | 0.991 |
| GAM · location | 0.0364 [0.0267, 0.0473] | 0.120 | -0.01 [-0.41, +0.37] | 1.00 [0.82, 1.28] | 0.002 | 0.991 |
| GAM · everything | 0.0357 [0.0265, 0.0461] | 0.119 | -0.08 [-0.49, +0.30] | 1.01 [0.83, 1.28] | 0.005 | 0.991 |
| HistGB · location | 0.0370 [0.0267, 0.0480] | 0.127 | -0.04 [-0.43, +0.33] | 0.99 [0.84, 1.25] | 0.005 | 0.990 |
| HistGB · everything | 0.0414 [0.0300, 0.0540] | 0.145 | -0.12 [-0.49, +0.28] | 1.03 [0.91, 1.19] | 0.007 | 0.986 |

★ = Week 8's baseline to beat.

## 2. Did the August conclusions hold?

- **Calibration:** August chose no calibrator, so the locked p(s) IS the model's own output. Its September calibration slope is 1.06 [0.87, 1.36] and intercept -0.03 [-0.40, +0.33] — if either interval misses its ideal (slope 1, intercept 0), say so in the report: it is the honest finding, and it is too late to change.
- **Everything vs location only:** M3 · + situation minus M1 · distance: Δ Brier -0.0003 [-0.0031, +0.0023].
- HistGB · location minus the baseline (M1 · distance): Δ Brier +0.0010 [-0.0015, +0.0038] — can't tell apart.
- HistGB · everything minus the baseline (M1 · distance): Δ Brier +0.0054 [+0.0016, +0.0094] — worse.
- **Where decisions are made** (p(s) between 20% and 80%): predicted 55%, observed 57% [45%, 68%], n = 67.

## 3. Subgroups on September

0 of 19 testable slices flagged (≈ 1.0 expected by chance).

| group | slice | n | predicted | observed [95%] | gap (pts) | |
|---|---|---:|---:|---|---:|---|
| Challenger | batter | 345 | 51.2% | 51.0% [45.8%, 56.2%] | -0.2 |  |
| Challenger | catcher | 239 | 54.7% | 55.2% [48.9%, 61.4%] | +0.5 |  |
| Challenger | pitcher | 75 | 41.8% | 40.0% [29.7%, 51.3%] | -1.8 |  |
| Count | other counts | 450 | 54.5% | 54.2% [49.6%, 58.8%] | -0.3 |  |
| Count | two strikes (not full) | 167 | 42.4% | 43.1% [35.8%, 50.7%] | +0.7 |  |
| Count | three balls (not full) | 22 | 52.2% | 54.5% [34.7%, 73.1%] | +2.3 |  |
| Count | full count (3-2) | 20 | 55.0% | 50.0% [29.9%, 70.1%] | -5.0 |  |
| Challenges left | 1 left | 203 | 48.2% | 48.3% [41.5%, 55.1%] | +0.1 |  |
| Challenges left | 2 left | 456 | 52.8% | 52.6% [48.0%, 57.2%] | -0.2 |  |
| Month | 09 Sep | 659 | 51.4% | 51.3% [47.5%, 55.1%] | -0.1 |  |
| Batter hand | bats R | 659 | 51.4% | 51.3% [47.5%, 55.1%] | -0.1 |  |
| Pitcher hand | throws R | 659 | 51.4% | 51.3% [47.5%, 55.1%] | -0.1 |  |
| Edge | side edge | 367 | 54.6% | 53.4% [48.3%, 58.4%] | -1.2 |  |
| Edge | top edge | 156 | 48.1% | 48.1% [40.4%, 55.9%] | +0.0 |  |
| Edge | bottom edge | 136 | 46.5% | 49.3% [41.0%, 57.6%] | +2.8 |  |
| Inning | 1–3 | 242 | 51.5% | 53.3% [47.0%, 59.5%] | +1.8 |  |
| Inning | 4–6 | 214 | 51.8% | 50.9% [44.3%, 57.6%] | -0.8 |  |
| Inning | 7–9 | 180 | 50.3% | 48.3% [41.1%, 55.6%] | -2.0 |  |
| Inning | extras | 23 | 55.4% | 56.5% [36.8%, 74.4%] | +1.1 |  |
| Leverage | low (< 0.85) | 271 | 54.1% | 54.6% [48.7%, 60.4%] | +0.5 |  |
| Leverage | medium | 287 | 52.3% | 51.9% [46.1%, 57.6%] | -0.4 |  |
| Leverage | high (≥ 2) | 101 | 41.5% | 40.6% [31.5%, 50.3%] | -0.9 |  |

## 4. Figures

- `figures/week9/w9_06_test.png` — every model on September, and the locked p(s)'s reliability diagram
- `figures/week9/w9_03_subgroups.png`, `w9_04_months.png` — redrawn with September added