# Data Source Inventory

Week 3 deliverable. Fill in the "Verified" column as you confirm each one with
`00_env_check.py`, and keep this file updated — Week 14 lifts straight from it.

---

## Primary sources

### 1. Statcast pitch-level — Baseball Savant

- **URL:** `https://baseballsavant.mlb.com/statcast_search` (CSV at `/statcast_search/csv`)
- **Access:** `pybaseball.statcast()`, or direct HTTP with a browser User-Agent
- **Grain:** one row per pitch
- **Volume:** ~700–750k pitches for a full MLB season
- **Gives you:** pitch location (`plate_x`, `plate_z`), velocity, movement, release
  point, pitch type, count, outs, base state, score, batter/pitcher/catcher ids,
  the umpire's call (`description`), Statcast's own zone (`sz_top`, `sz_bot`), and
  in recent seasons run/win expectancy deltas
- **Does NOT give you:** who challenged, or whether the call was overturned
  — unless 2026 added ABS columns, which `00_env_check.py` tells you
- **Script:** `01_statcast_pitches_2026.py`
- **Risk:** rate limiting / 403s; data is silently revised after the fact
- **Verified:** ☐

### 2. ABS challenge events — MLB Stats API

- **URL:** `https://statsapi.mlb.com/api/v1/game/{gamePk}/playByPlay`
- **Access:** unauthenticated JSON
- **Grain:** one row per challenge event
- **Gives you:** challenger identity and role, `isOverturned` (**your model
  target**), inning, count, batter, pitcher, the original call
- **Also needed:** `/api/v1/schedule?...&hydrate=officials` for the home-plate umpire
- **Script:** `02_abs_challenges_statsapi.py`
- **Risk:** the Stats API is undocumented and unversioned in practice; field names
  can move between seasons. Note that MLB's GUMBO feed is known to flag only
  ~68% of challenges via `hasReview` alone, which is why the script also matches
  on description text.
- **Verified:** ☐

### 3. Player biographical data — MLB Stats API

- **URL:** `https://statsapi.mlb.com/api/v1/people?personIds=...`
- **Grain:** one row per player
- **Gives you:** **height** (required for the ABS zone), weight, bat side, throw
  hand, primary position (required to exclude position-player pitching)
- **Script:** `07_player_bios.py`
- **Risk:** listed heights are self-reported and rounded to the inch; a 1-inch
  error moves the zone by ~0.5 inch. Worth one sentence in limitations.
- **Verified:** ☐

---

## Supporting sources

### 4. ABS challenge leaderboards — Baseball Savant

- **URL:** `https://baseballsavant.mlb.com/leaderboard/abs-challenges`
- **Grain:** player-season and team-season aggregates
- **Gives you:** challenges, wins, win %, **expected challenges**, **overturns vs
  expected**, split by challenger role and by for/against
- **Use it for:** validating your event counts, and as the descriptive benchmark
  your novelty claim is measured against
- **Do NOT use it for:** model features — the expected metrics are built from the
  outcome you're predicting
- **Script:** `03_savant_abs_leaderboards.py`
- **Verified:** ☐

### 5. ABS dashboard + metrics documentation — Baseball Savant

- **URLs:** `https://baseballsavant.mlb.com/abs`,
  `https://baseballsavant.mlb.com/abs-metrics-documentation`
- **Use it for:** definitions you must match. The documentation defines
  *challenge opportunity* (a called pitch, ≥1 challenge left, call went against
  that side, no position player pitching — ~50% of 2026 pitches) and *reasonable
  pitch* (~5%). Script 08 implements both; the "reasonable" one is a simplified
  proxy since Savant's version also uses RE288 run value.
- **Verified:** ☐

### 6. Catcher framing — Baseball Savant

- **URL:** `https://baseballsavant.mlb.com/leaderboard/catcher-framing`
- **Access:** `pybaseball.statcast_catcher_framing()` or direct CSV
- **Grain:** catcher-season
- **Script:** `04_catcher_framing.py`
- **Caveat that matters for RQ4:** Savant's 2026 framing metric now credits
  successful challenges alongside the original framing attempt. So a
  2025-vs-2026 comparison is not measuring a clean behavioural change — the
  metric itself changed. Prefer within-2026 variation.
- **Verified:** ☐

### 7. Win probability and leverage — MLB Stats API

- **URL:** `https://statsapi.mlb.com/api/v1/game/{gamePk}/winProbability`
- **Grain:** plate appearance (**not** pitch)
- **Gives you:** `homeTeamWinProbability`, `homeTeamWinProbabilityAdded`,
  `leverageIndex`
- **Script:** `06_winprob_leverage.py`
- **Limitation:** a challenge happens mid-PA, so this is the state entering the
  PA. For pitch-level values use Statcast's win-expectancy columns if present, or
  build your own in Week 10 — which the timeline schedules anyway.
- **Verified:** ☐

### 8. Umpire accuracy

- **Preferred:** derived from your own Statcast pull against the ABS zone —
  reproducible, pitch-level, and consistent with the zone you model
- **Secondary:** `https://umpscorecards.com/data/games` — a client-rendered app
  with no documented API and its own terms of use. Check terms before relying on
  it in a published paper; they have historically been willing to share data for
  academic work, and a documented permission belongs in an appendix.
- **Script:** `05_umpire_data.py` (`--mode derive` is the default)
- **Verified:** ☐

---

## Cross-checks worth having

- **Baseball-Reference ABS page** (`baseball-reference.com/friv/abs-challenges.shtml`)
  — independent aggregates by role, inning, count, base state, leverage bracket,
  month, team, and umpire. No export, but excellent for eyeballing whether your
  numbers are sane.
- **ABSScoreboard/abs-data** (`github.com/ABSScoreboard/abs-data`) — a public
  GitHub mirror of Savant challenge data, refreshed three times daily as JSON at
  `absscoreboard.github.io/abs-data/data/abs-challenges.json`. Useful as a
  second opinion on your own extraction. Third-party, so verify before citing.
- **SABR Analytics 2026** — the MLB Statcast/Savant ABS session. Worth citing in
  the literature review.

---

## Risk register (Week 3 deliverable)

| # | Risk | Likelihood | Impact | Fallback |
|---|---|---|---|---|
| 1 | Savant rate-limits or blocks the bulk pull | Medium | High | Chunked + resumable pulls; longer sleeps; direct-HTTP fallback; pull early |
| 2 | Challenge↔pitch join rate is poor | Low | **Critical** | Diagnose the at_bat_number off-by-one and pitch-number counting in Week 3, not Week 5 |
| 3 | pybaseball breaks or goes unmaintained | Medium | Medium | Every script has a direct-HTTP path that doesn't import it |
| 4 | Challenges-remaining derivation is wrong | Medium | **Critical** | `test_logic.py` covers the state machine; hand-verify three real games |
| 5 | ABS zone approximation is off | Medium | High | Validate against actual challenge outcomes (script 08 reports the rate) |
| 6 | Single-season sample is too small for subgroup models | Medium | Medium | Pool with 2025 AAA/spring data, with a level flag and a transportability test |
| 7 | Statcast revises data mid-project | Low | Medium | Date-stamped snapshots + checksums from Week 4 |

---

## Manifest requirement

Every raw pull records: file name, source URL, pull timestamp, row count, and a
checksum, in `data/raw/MANIFEST.json`. Hand-downloaded files too. Without this you
cannot explain a number that moves between October and December — and something
will.
