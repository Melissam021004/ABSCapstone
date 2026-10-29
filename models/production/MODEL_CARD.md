# Model card — p(s), the ABS challenge overturn probability

**Version** 2026.10.29-b81183b4 · locked 2026-10-29T16:01:08 · bundle `models/production/` · content hash `b81183b4d5eb25a9…`

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

- **GAM · location** (`GAM-loc`, family: gam), feature set: location only
- Features: `edge`, `call_dir`, `dist_against_call`
- Input state (`predict(state)`): `plate_x`, `plate_z`, `batter_height_ft`, `called_strike`
- Calibration: no calibration — the model's own probabilities, fitted on August (the calibration window)
- Strike zone: 17 in. wide; top 53.5% and bottom 27% of the batter's height; a strike if any part of the ball (radius 1.45 in.) touches it. Distance is measured from the ball's surface and signed against the call.

## Training data

- MLB 2026 regular season, every challenged pitch with a known result (cohort `419af01ad9eee03f`).
- Train 2026-03-25 → 2026-07-31 (5,219 challenges) · calibrate 2026-08-01 → 2026-08-31 (1,435) · test: September (1,243).
- Call recording check: FINAL (post-review) call recorded — repaired from who challenged (4,417 challenged pitches corrected to the original call).
- Sources: Statcast (pitch location), MLB Stats API (challenges, umpires, heights), derived challenge inventory.

## Performance

| window | Brier | log loss | cal. intercept | cal. slope | ECE | AUC |
|---|---:|---:|---:|---:|---:|---:|
| August, uncalibrated (cross-fitted) | 0.0107 | 0.0354 | -0.33 | 0.98 | 0.003 | 0.999 |
| **August, calibrated (cross-fitted)** | 0.0107 | 0.0354 | -0.33 | 0.98 | 0.003 | 0.999 |
| **September (test, opened once)** | 0.0087 | 0.0405 | -0.11 | 0.90 | 0.003 | 0.998 |

## Performance by subgroup

**August (cross-fitted)** — 0 of 22 testable slices flagged (≈ 1.1 by chance).

| group | slice | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| Challenger | batter | 611 | 61.7% | 61.0% [57.1%, 64.8%] |
| Challenger | catcher | 793 | 53.9% | 53.7% [50.2%, 57.2%] |
| Challenger | pitcher | 27 | 50.2% | 48.1% [30.7%, 66.0%] |
| Challenger | other | 4 | 9.9% | 25.0% [4.6%, 69.9%] |
| Count | other counts | 1204 | 58.9% | 58.5% [55.7%, 61.2%] |
| Count | two strikes (not full) | 190 | 44.8% | 44.7% [37.8%, 51.8%] |
| Count | three balls (not full) | 41 | 57.5% | 58.5% [43.4%, 72.2%] |
| Challenges left | 1 left | 293 | 55.1% | 54.6% [48.9%, 60.2%] |
| Challenges left | 2 left | 1142 | 57.5% | 57.2% [54.3%, 60.0%] |
| Month | 08 Aug | 1435 | 57.0% | 56.7% [54.1%, 59.2%] |
| Batter hand | bats L | 739 | 58.3% | 58.3% [54.7%, 61.8%] |
| Batter hand | bats R | 696 | 55.6% | 54.9% [51.2%, 58.5%] |
| Pitcher hand | throws L | 434 | 56.7% | 56.0% [51.3%, 60.6%] |
| Pitcher hand | throws R | 1001 | 57.1% | 56.9% [53.9%, 60.0%] |
| Edge | side edge | 733 | 57.4% | 57.2% [53.6%, 60.7%] |
| Edge | top edge | 241 | 53.5% | 51.9% [45.6%, 58.1%] |
| Edge | bottom edge | 461 | 58.2% | 58.4% [53.8%, 62.8%] |
| Inning | 1–3 | 396 | 68.1% | 67.7% [62.9%, 72.1%] |
| Inning | 4–6 | 501 | 58.7% | 58.3% [53.9%, 62.5%] |
| Inning | 7–9 | 497 | 46.3% | 46.1% [41.7%, 50.5%] |
| Inning | extras | 41 | 59.0% | 58.5% [43.4%, 72.2%] |
| Leverage | low (< 0.85) | 639 | 56.7% | 56.5% [52.6%, 60.3%] |
| Leverage | medium | 557 | 59.2% | 58.5% [54.4%, 62.5%] |
| Leverage | high (≥ 2) | 227 | 51.9% | 52.0% [45.5%, 58.4%] |

**September (test)** — 0 of 22 testable slices flagged (≈ 1.1 by chance).

| group | slice | n | predicted | observed [95%] |
|---|---|---:|---:|---|
| Challenger | batter | 511 | 59.4% | 59.3% [55.0%, 63.5%] |
| Challenger | catcher | 713 | 56.4% | 56.1% [52.4%, 59.7%] |
| Challenger | pitcher | 17 | 35.8% | 35.3% [17.3%, 58.7%] |
| Challenger | other | 2 | 0.0% | 50.0% [9.5%, 90.5%] |
| Count | other counts | 1012 | 60.3% | 60.1% [57.0%, 63.1%] |
| Count | two strikes (not full) | 177 | 39.7% | 40.1% [33.2%, 47.5%] |
| Count | three balls (not full) | 54 | 57.3% | 57.4% [44.2%, 69.7%] |
| Challenges left | 1 left | 233 | 53.0% | 52.4% [46.0%, 58.7%] |
| Challenges left | 2 left | 1010 | 58.2% | 58.2% [55.2%, 61.2%] |
| Month | 09 Sep | 1243 | 57.2% | 57.1% [54.4%, 59.8%] |
| Batter hand | bats L | 589 | 57.8% | 57.7% [53.7%, 61.7%] |
| Batter hand | bats R | 654 | 56.7% | 56.6% [52.7%, 60.3%] |
| Pitcher hand | throws L | 395 | 59.7% | 59.5% [54.6%, 64.2%] |
| Pitcher hand | throws R | 848 | 56.1% | 56.0% [52.7%, 59.3%] |
| Edge | side edge | 613 | 57.3% | 57.6% [53.6%, 61.4%] |
| Edge | top edge | 207 | 59.9% | 58.5% [51.6%, 65.0%] |
| Edge | bottom edge | 423 | 55.8% | 55.8% [51.0%, 60.5%] |
| Inning | 1–3 | 385 | 64.8% | 64.7% [59.8%, 69.3%] |
| Inning | 4–6 | 388 | 58.1% | 58.5% [53.5%, 63.3%] |
| Inning | 7–9 | 432 | 50.2% | 49.8% [45.1%, 54.5%] |
| Inning | extras | 38 | 51.1% | 50.0% [34.8%, 65.2%] |
| Leverage | low (< 0.85) | 570 | 59.8% | 59.6% [55.6%, 63.6%] |
| Leverage | medium | 484 | 56.6% | 56.6% [52.2%, 61.0%] |
| Leverage | high (≥ 2) | 176 | 48.0% | 47.7% [40.5%, 55.1%] |

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

- Libraries: python 3.13.13, numpy 2.5.3, pandas 3.0.5, scipy 1.18.0, sklearn 1.9.1, catboost 1.2.10, lightgbm 4.6.0
- Pinned in `models/production/requirements-lock.txt`; every file's SHA-256 is in `models/production/manifest.json`, checked on every load.
- Earlier locks are kept in `models/production_history/<version>/`.
- Test window opened 2026-10-29T16:03:39 (1 time(s)).