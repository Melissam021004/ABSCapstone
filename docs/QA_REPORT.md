# QA Report — `challenge_inventory_2026`

Generated 2026-10-08 13:53 by `scripts/10_qa_assertions.py`.

## ❌ FAILED

**720,209 rows × 166 columns.** 19/24 assertions passed (3 critical failures, 2 warnings).

## Critical failures — do not model on this table

### `NO_CHALLENGE_AT_ZERO` — Nobody challenges with zero left

- **Result:** 4 challenges made with no challenge remaining
- **Why it matters:** Either the inventory walk is wrong or a challenge event was mis-attributed to the wrong side.
- **Offending rows:** `docs/qa_failures/NO_CHALLENGE_AT_ZERO.csv`

### `CHALLENGED_ARE_CALLED` — Challenged pitches are called pitches

- **Result:** 97 challenges on swung-at pitches
- **Why it matters:** You can only challenge a taken pitch. These are join errors — most likely pitch_number is off by one.
- **Offending rows:** `docs/qa_failures/CHALLENGED_ARE_CALLED.csv`

### `ZONE_PREDICTS_OUTCOME` — ABS zone reproduces actual challenge outcomes

- **Result:** 44.1% agreement (need >= 85%)
- **Why it matters:** Your reconstructed ABS zone disagrees with what ABS actually ruled. The zone constants, the ball-radius rule or the batter heights are wrong — and every distance feature inherits the error.
- **Offending rows:** `docs/qa_failures/ZONE_PREDICTS_OUTCOME.csv`

## Warnings — look, decide, write down why it's acceptable

### `CHALLENGER_ROLE_VALID` — Challenger is batter, catcher or pitcher

- **Result:** 119 of 8,012 (1.5%) with a missing or invalid role
- **Why it matters:** Only those three may challenge, so a null role means the API payload had no reviewDetails.player. A few is normal and they drop out of any role-split analysis; above 2% suspect the extraction in script 02.
- **Offending rows:** `docs/qa_failures/CHALLENGER_ROLE_VALID.csv`

### `CATCHERS_BEAT_BATTERS` — Catchers out-challenge batters

- **Result:** catcher 55.3% vs batter 57.8%
- **Why it matters:** MLB has publicly reported catchers are the most successful challengers. If your data says otherwise, suspect the data before the finding.

## All assertions

| Check | Severity | Result | Detail |
|---|---|---|---|
| `TABLE_NOT_EMPTY` | CRITICAL | pass | 720,209 rows |
| `UNIQUE_PITCH_KEY` | CRITICAL | pass | 0 duplicate key rows |
| `COUNT_IN_RANGE` | CRITICAL | pass | 0 rows outside balls 0-3 / strikes 0-2 |
| `INVENTORY_IN_BOUNDS` | CRITICAL | pass | observed range 0-2, 0 violations |
| `INVENTORY_STARTS_AT_2` | CRITICAL | pass | 0 games start wrong |
| `NO_OPPORTUNITY_AT_ZERO` | CRITICAL | pass | 0 violations |
| `NO_CHALLENGE_AT_ZERO` | CRITICAL | **FAIL** | 4 challenges made with no challenge remaining |
| `SOME_CHALLENGES` | CRITICAL | pass | 8,012 challenged pitches |
| `CHALLENGED_ARE_CALLED` | CRITICAL | **FAIL** | 97 challenges on swung-at pitches |
| `CHALLENGER_ROLE_VALID` | WARN | *warn* | 119 of 8,012 (1.5%) with a missing or invalid role |
| `OUTCOME_PRESENT` | WARN | pass | 0 of 8,012 (0.0%) missing is_overturned |
| `FAILED_BUDGET` | CRITICAL | pass | 0 game-sides over budget |
| `ZONE_PREDICTS_OUTCOME` | CRITICAL | **FAIL** | 44.1% agreement (need >= 85%) |
| `JOIN_RATE` | CRITICAL | pass | 8,012/8,312 matched (96.4%) |
| `UMPIRE_COVERAGE` | WARN | pass | 99.5% populated (want >= 95%) |
| `HEIGHT_COVERAGE` | WARN | pass | 99.9% populated (want >= 95%) |
| `LEVERAGE_COVERAGE` | WARN | pass | 98.2% populated (want >= 90%) |
| `ZONE_FROM_HEIGHT` | WARN | pass | 99.9% of rows use batter height |
| `OVERTURN_BAND` | WARN | pass | 55.7% (expected roughly 40-60%) |
| `CATCHERS_BEAT_BATTERS` | WARN | *warn* | catcher 55.3% vs batter 57.8% |
| `OPPORTUNITY_SHARE` | WARN | pass | 49.8% of pitches |
| `REASONABLE_SHARE` | WARN | pass | 5.0% of pitches |
| `OVERTURN_FALLS_WITH_DISTANCE` | WARN | pass | 0-1in 52% \| 1-2in 59% \| 2-4in 61% \| 4-8in 47% \| 8+in 44% |
| `LEVERAGE_MEAN` | WARN | pass | mean 1.000 |

---

Re-run with `python scripts/10_qa_assertions.py`. Exit code is 1 on any critical failure, so this can gate a commit or a CI job.
