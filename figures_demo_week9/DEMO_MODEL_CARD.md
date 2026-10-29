# Model card — p(s), the ABS challenge overturn probability

> **DEMO DATA — this card was generated from the synthetic fixture.**

**Version** 2026.10.20-ff73ea58 · locked 2026-10-20T18:54:45 · bundle `models/production/` · content hash `ff73ea5861e7359d…`

## Intended use

p(s): the probability that ABS overturns the umpire's call **if the call is challenged**, given the state s of the pitch. It is the overturn input to the Q-function of the challenge MDP (Week 11):

    Q_challenge(s, r) = p(s)·V(s_overturned, r) + (1 − p(s))·V(s_stands, r − 1)

It is also used to measure decision regret (what a challenge was worth vs what the team did) and by the dashboard's simulator.

## What it should NOT be used for

- **Judging umpires.** p(s) is about challenged calls; it is not a measure of an umpire's accuracy.
- **Other seasons or rule sets** without re-validation (2026 is the first season of the challenge system; the zone definition or challenge rules may change).
- **Pitches that weren't taken** (swings), or anything other than the ABS rectangle defined below.
- **As a guarantee for one pitch.** It is calibrated on average and by slice; any single prediction carries the measurement error of the pitch location (about an inch).

## Model

- **M1 · distance** (`M1`, family: logistic), feature set: location only
- Features: `dist_against_call`
- Input state (`predict(state)`): `plate_x`, `plate_z`, `batter_height_ft`, `called_strike`
- Calibration: no calibration — the model's own probabilities, fitted on August (the calibration window)
- Strike zone: 17 in. wide; top 53.5% and bottom 27% of the batter's height; a strike if any part of the ball (radius 1.45 in.) touches it. Distance is measured from the ball's surface and signed against the call.

## Training data

- MLB 2026 regular season, every challenged pitch with a known result (cohort `c380922df66bdf57`).
- Train 2026-03-26 → 2026-07-31 (3,211 challenges) · calibrate 2026-08-01 → 2026-08-31 (793) · test: September (659).
- Call recording check: ORIGINAL umpire call recorded — no repair needed.
- Sources: Statcast (pitch location), MLB Stats API (challenges, umpires, heights), derived challenge inventory.

## Performance

| window | Brier | log loss | cal. intercept | cal. slope | ECE | AUC |
|---|---:|---:|---:|---:|---:|---:|
| August, uncalibrated (cross-fitted) | 0.0439 | 0.1429 | -0.13 | 0.87 | 0.013 | 0.987 |
| **August, calibrated (cross-fitted)** | 0.0439 | 0.1429 | -0.13 | 0.87 | 0.013 | 0.987 |
| **September (test, opened once)** | 0.0360 | 0.1177 | -0.03 | 1.06 | 0.008 | 0.991 |

## Performance by subgroup

**August (cross-fitted)** — 0 of 19 testable slices flagged (≈ 1.0 by chance).

| group | slice | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| Challenger | batter | 416 | 52.3% | 52.2% [47.4%, 56.9%] |
| Challenger | catcher | 272 | 57.9% | 56.6% [50.7%, 62.4%] |
| Challenger | pitcher | 105 | 50.5% | 50.5% [41.1%, 59.9%] |
| Count | other counts | 547 | 52.5% | 51.2% [47.0%, 55.4%] |
| Count | two strikes (not full) | 203 | 58.5% | 59.1% [52.2%, 65.6%] |
| Count | three balls (not full) | 23 | 48.1% | 43.5% [25.6%, 63.2%] |
| Count | full count (3-2) | 20 | 55.5% | 70.0% [48.1%, 85.5%] |
| Challenges left | 1 left | 217 | 50.6% | 47.0% [40.5%, 53.6%] |
| Challenges left | 2 left | 576 | 55.2% | 55.9% [51.8%, 59.9%] |
| Month | 08 Aug | 793 | 54.0% | 53.5% [50.0%, 56.9%] |
| Batter hand | bats R | 793 | 54.0% | 53.5% [50.0%, 56.9%] |
| Pitcher hand | throws R | 793 | 54.0% | 53.5% [50.0%, 56.9%] |
| Edge | side edge | 484 | 57.0% | 56.4% [52.0%, 60.8%] |
| Edge | top edge | 153 | 53.5% | 54.2% [46.3%, 61.9%] |
| Edge | bottom edge | 156 | 45.1% | 43.6% [36.1%, 51.4%] |
| Inning | 1–3 | 276 | 56.2% | 56.9% [51.0%, 62.6%] |
| Inning | 4–6 | 241 | 53.0% | 52.3% [46.0%, 58.5%] |
| Inning | 7–9 | 256 | 52.8% | 51.6% [45.5%, 57.6%] |
| Inning | extras | 20 | 49.9% | 45.0% [25.8%, 65.8%] |
| Leverage | low (< 0.85) | 310 | 55.1% | 56.1% [50.6%, 61.5%] |
| Leverage | medium | 367 | 52.5% | 51.8% [46.7%, 56.8%] |
| Leverage | high (≥ 2) | 116 | 55.6% | 51.7% [42.7%, 60.6%] |

**September (test)** — 0 of 19 testable slices flagged (≈ 1.0 by chance).

| group | slice | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| Challenger | batter | 345 | 51.2% | 51.0% [45.8%, 56.2%] |
| Challenger | catcher | 239 | 54.7% | 55.2% [48.9%, 61.4%] |
| Challenger | pitcher | 75 | 41.8% | 40.0% [29.7%, 51.3%] |
| Count | other counts | 450 | 54.5% | 54.2% [49.6%, 58.8%] |
| Count | two strikes (not full) | 167 | 42.4% | 43.1% [35.8%, 50.7%] |
| Count | three balls (not full) | 22 | 52.2% | 54.5% [34.7%, 73.1%] |
| Count | full count (3-2) | 20 | 55.0% | 50.0% [29.9%, 70.1%] |
| Challenges left | 1 left | 203 | 48.2% | 48.3% [41.5%, 55.1%] |
| Challenges left | 2 left | 456 | 52.8% | 52.6% [48.0%, 57.2%] |
| Month | 09 Sep | 659 | 51.4% | 51.3% [47.5%, 55.1%] |
| Batter hand | bats R | 659 | 51.4% | 51.3% [47.5%, 55.1%] |
| Pitcher hand | throws R | 659 | 51.4% | 51.3% [47.5%, 55.1%] |
| Edge | side edge | 367 | 54.6% | 53.4% [48.3%, 58.4%] |
| Edge | top edge | 156 | 48.1% | 48.1% [40.4%, 55.9%] |
| Edge | bottom edge | 136 | 46.5% | 49.3% [41.0%, 57.6%] |
| Inning | 1–3 | 242 | 51.5% | 53.3% [47.0%, 59.5%] |
| Inning | 4–6 | 214 | 51.8% | 50.9% [44.3%, 57.6%] |
| Inning | 7–9 | 180 | 50.3% | 48.3% [41.1%, 55.6%] |
| Inning | extras | 23 | 55.4% | 56.5% [36.8%, 74.4%] |
| Leverage | low (< 0.85) | 271 | 54.1% | 54.6% [48.7%, 60.4%] |
| Leverage | medium | 287 | 52.3% | 51.9% [46.1%, 57.6%] |
| Leverage | high (≥ 2) | 101 | 41.5% | 40.6% [31.5%, 50.3%] |

## Known limitations

1. **Selection on unobservables.** Trained on challenged pitches only; teams challenge pitches they suspect were missed, so for an unchallenged pitch at the same location the true chance may be lower. ABS's ruling is never observed without a challenge.
2. **Zone geometry.** Heights are listed (rounded to the inch) rather than ABS-measured; a one-inch height error moves the top edge about half an inch.
3. **Measurement error** in Statcast's plate location vs the Hawk-Eye measurement ABS uses — the reason p(s) climbs over an inch or two instead of jumping at the edge.
4. **One season, one calibration month, one test month.** Intervals cover sampling noise, not year-to-year change.
5. **August was used twice** — to choose models (Weeks 7–8) and to fit the calibrator — so August scores are slightly optimistic; September is the honest number.

## Ethical and practical considerations

- Public data only; players appear as IDs in the model.
- A recommendation tool, not an automated decision: the policy built on p(s) advises whether a challenge is worth its option value.

## Versioning

- Libraries: python 3.13.16, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, sklearn 1.9.1
- Pinned in `models/production/requirements-lock.txt`; every file's SHA-256 is in `models/production/manifest.json`, checked on every load.
- Earlier locks are kept in `models/production_history/<version>/`.
- Test window opened 2026-10-20T18:55:14 (1 time(s)).