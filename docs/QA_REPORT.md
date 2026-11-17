# QA Report — `challenge_inventory_2026`

Generated 2026-11-17 16:20 by `scripts/10_qa_assertions.py`.

## ⚠️ PASSED WITH WARNINGS

**720,209 rows × 167 columns.** 23/24 assertions passed (0 critical failures, 1 warnings).

## Warnings — look, decide, write down why it's acceptable

### `CHALLENGED_ARE_CALLED` — Challenged pitches are called pitches

- **Result:** 1 of 10,558 challenges on swung-at pitches (0.01%)
- **Why it matters:** You can only challenge a taken pitch. Many of these mean a join error (pitch_number off by one); a few are pitches Statcast and the Stats API record differently, and the cohort leaves them out.
- **Offending rows:** `docs/qa_failures/CHALLENGED_ARE_CALLED.csv`

## All assertions

| Check | Severity | Result | Detail |
|---|---|---|---|
| `TABLE_NOT_EMPTY` | CRITICAL | pass | 720,209 rows |
| `UNIQUE_PITCH_KEY` | CRITICAL | pass | 0 duplicate key rows |
| `COUNT_IN_RANGE` | CRITICAL | pass | 0 rows outside balls 0-3 / strikes 0-2 |
| `INVENTORY_IN_BOUNDS` | CRITICAL | pass | observed range 0-2, 0 violations |
| `INVENTORY_STARTS_AT_2` | CRITICAL | pass | 0 games start wrong |
| `NO_OPPORTUNITY_AT_ZERO` | CRITICAL | pass | 0 violations |
| `NO_CHALLENGE_AT_ZERO` | CRITICAL | pass | 0 challenges made with no challenge remaining |
| `SOME_CHALLENGES` | CRITICAL | pass | 10,558 challenged pitches |
| `CHALLENGED_ARE_CALLED` | WARN | *warn* | 1 of 10,558 challenges on swung-at pitches (0.01%) |
| `CHALLENGER_ROLE_VALID` | WARN | pass | 0 of 10,558 (0.0%) with a missing or invalid role |
| `OUTCOME_PRESENT` | WARN | pass | 0 of 10,558 (0.0%) missing is_overturned |
| `FAILED_BUDGET` | CRITICAL | pass | 0 game-sides over budget |
| `ZONE_PREDICTS_OUTCOME` | CRITICAL | pass | 98.8% agreement (need >= 85%; checked against the original call; the recorded call is still the original on 0% of overturned challenges) |
| `JOIN_RATE` | CRITICAL | pass | 10,558/10,558 matched (100.0%) |
| `UMPIRE_COVERAGE` | WARN | pass | 99.5% populated (want >= 95%) |
| `HEIGHT_COVERAGE` | WARN | pass | 99.9% populated (want >= 95%) |
| `LEVERAGE_COVERAGE` | WARN | pass | 98.2% populated (want >= 90%) |
| `ZONE_FROM_HEIGHT` | WARN | pass | 99.9% of rows use batter height |
| `OVERTURN_BAND` | WARN | pass | 53.9% (expected roughly 40-60%) |
| `CATCHERS_BEAT_BATTERS` | WARN | pass | catcher 58.6% vs batter 48.8% |
| `OPPORTUNITY_SHARE` | WARN | pass | 47.9% of pitches |
| `REASONABLE_SHARE` | WARN | pass | 4.8% of pitches |
| `OVERTURN_FALLS_WITH_DISTANCE` | WARN | pass | 0-1in 52% \| 1-2in 56% \| 2-4in 56% \| 4-8in 44% |
| `LEVERAGE_MEAN` | WARN | pass | mean 1.000 |

---

Re-run with `python scripts/10_qa_assertions.py`. Exit code is 1 on any critical failure, so this can gate a commit or a CI job.
