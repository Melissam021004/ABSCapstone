# Week 10 — What the 2026 challenges were worth

Generated 2026-11-17 16:22 by `scripts/20_week10_challenge_value.py` · p(s) 2026.10.29-b81183b4 · V(s) 2026.11.17-824ec8be · cohort `419af01ad9eee03f` · 1000 game-level bootstrap resamples · 999 permutations per skill test

> Read this next to `WEEK10.md` (Part B), which explains every section.

## In one paragraph

Umpires' wrong calls moved **308.4 wins** across the league (24,083 expected wrong calls on 370,883 called pitches). Overturned challenges won back **102.2** of them (33.1%). The 10,535 challenges were worth 102.2 wins in expectation and delivered 102.2. The best team added 4.55 wins with challenges; the median team 3.33.

## 0. Inputs

- recorded call matches who challenged on 100.0% of 4,861 upheld and 0.0% of 5,674 overturned challenges
- verdict: FINAL (post-review) call recorded — repaired from who challenged (5,674 challenged pitches corrected)
- 370,909 called pitches valued with p(s) and V(s) (26 with a location outside p(s)'s range skipped); 345,141 challenge opportunities (a call against a side with a challenge left, worked out from the umpire's ORIGINAL call); 10,535 challenges.
- **Where p(s) is used.** p(s) learned from challenged pitches, which sit between -4.5 and +4.3 inches against the call (99% of them). Beyond that range its curve is held flat, so a call that is clearly right — a strike in the middle of the zone — would still get the small chance of being wrong it had at the edge of the range. There the ABS call isn't in doubt, so it is taken from the zone geometry instead: 203,826 called pitches further on the right side count as calls that stand, 91 further on the wrong side as calls ABS would reverse. Inside the range, the locked p(s) is used unchanged.

## 1. What a call is worth

Win probability the challenging side gains if the call is overturned (`dwp`), and runs (to the end of the inning) for the same side:

| set | n | mean | median | 90th percentile | runs (mean) |
|---|---:|---:|---:|---:|---:|
| every opportunity | 345,141 | +1.36 pts | +0.84 pts | +2.98 pts | +0.149 |
| wrong calls (p(s) ≥ 50%) | 22,332 | +1.27 pts | +0.79 pts | +2.75 pts | +0.139 |
| challenged | 10,535 | +2.05 pts | +1.13 pts | +4.90 pts | +0.209 |
| overturned | 5,674 | +1.80 pts | +1.04 pts | +4.28 pts | +0.187 |

Most calls are worth little; a few are worth a lot. The ten most valuable overturns:

| date | inning | score (home–away) | count | outs | challenging team | role | gain |
|---|---|---|---|---:|---|---|---:|
| Apr 16 | 9 top | 3–3 | 3-2 | 1 | CWS (Edgar Quero) | catcher | +36.6 pts |
| Jun 27 | 9 top | 7–6 | 3-2 | 0 | CIN (Elly De La Cruz) | batter | +27.1 pts |
| Jun 06 | 9 top | 6–5 | 3-1 | 2 | STL (Jimmy Crooks) | catcher | +27.1 pts |
| Jun 08 | 7 bot | 2–5 | 3-2 | 1 | SEA (Jhonny Pereda) | catcher | +25.8 pts |
| Aug 05 | 6 bot | 0–3 | 3-2 | 1 | STL (Iván Herrera) | catcher | +24.4 pts |
| Aug 09 | 9 bot | 3–4 | 3-2 | 1 | BOS (Caleb Durbin) | batter | +23.9 pts |
| May 03 | 5 top | 4–4 | 3-2 | 1 | ATL (Eli White) | batter | +21.5 pts |
| May 20 | 4 bot | 1–3 | 3-2 | 2 | COL (Kyle Karros) | batter | +21.2 pts |
| Jun 30 | 4 bot | 1–3 | 3-2 | 2 | HOU (Jose Altuve) | batter | +21.2 pts |
| Sep 17 | 10 bot | 3–4 | 1-2 | 1 | LAA (Mike Trout) | batter | +20.4 pts |

Every challenge with its value: `docs/week10_value_challenges.csv`.

## 2. Inning level — more time before the third out

An overturned strike keeps an at-bat (or an inning) alive; an overturned ball takes a baserunner away. Whether a challenge is overturned is decided by where the ball was, not by the game situation, so overturned and upheld challenges are close to a natural experiment. Runs the batting team scored from the challenged pitch to the end of the inning (innings 1–8; walk-offs cut innings short):

| challenging side | n | after overturned | after upheld | gap [95%] | model's predicted gap |
|---|---:|---:|---:|---|---:|
| batting | 3,991 | 0.553 | 0.378 | +0.175 [+0.115, +0.234] | +0.200 |
| fielding | 4,968 | 0.412 | 0.599 | -0.187 [-0.245, -0.128] | -0.210 |
| all | 8,959 | 0.469 | 0.489 | -0.020 [-0.062, +0.023] | -0.023 |

Read: on a batting-side challenge the gap should be positive (an overturned strike helps the batting team score); on a fielding-side challenge negative (an overturned ball takes runs away). If the observed gap's interval covers the model's prediction, the run-expectancy engine behind V(s) is right about what a challenge does to an inning.

## 3. Game level

| challenging side | n | won after overturned | won after upheld | gap [95%] | predicted |
|---|---:|---:|---:|---|---:|
| batting | 4,757 | 48.9% | 45.6% | +3.3 pts [+0.3 pts, +6.0 pts] | +3.2 pts |
| fielding | 5,778 | 50.5% | 52.6% | -2.1 pts [-4.7 pts, +0.5 pts] | -0.9 pts |
| all | 10,535 | 49.8% | 49.1% | +0.7 pts [-1.2 pts, +2.5 pts] | +1.3 pts |

The game-level gap is a few points at most and games are noisy, so expect a wide interval; the check is that it doesn't contradict the prediction.

## 4. Value still ahead — the raw material of "holding a challenge"

At the start of each half-inning, averaged over every game and team: how much win probability is still at stake in the calls that will go against that team later.

| start of | stakes still ahead | best single wrong call ahead | best two | a wrong call now is worth |
|---|---:|---:|---:|---:|
| 1 top | +6.32 pts | +2.90 pts | +4.26 pts | +1.27 pts |
| 2 top | +5.56 pts | +2.70 pts | +3.93 pts | +1.29 pts |
| 3 top | +4.86 pts | +2.51 pts | +3.59 pts | +1.19 pts |
| 4 top | +4.19 pts | +2.32 pts | +3.26 pts | +1.26 pts |
| 5 top | +3.52 pts | +2.09 pts | +2.88 pts | +1.19 pts |
| 6 top | +2.85 pts | +1.83 pts | +2.44 pts | +1.17 pts |
| 7 top | +2.18 pts | +1.50 pts | +1.94 pts | +1.24 pts |
| 8 top | +1.50 pts | +1.11 pts | +1.39 pts | +1.24 pts |
| 9 top | +0.85 pts | +0.69 pts | +0.81 pts | +1.33 pts |
| 9 bot | +0.90 pts | +0.75 pts | +0.87 pts | +2.02 pts |

- **best single wrong call ahead** is what ONE challenge saved for later could win with perfect foresight — an upper bound on the value of holding it. A real team doesn't know which call will be the big one, and a failed challenge is lost; Week 11's backward induction computes the real value, V(s, r) − V(s, r − 1), below this ceiling.
- Compare it with **a wrong call now**: early in the game the best call still to come is worth several times a typical call now — the case for holding. Late, the two converge, and the last innings are where a challenge in hand is worth spending.

By the umpire's accuracy (thirds of umpires, adjusted for how hard their pitches were):

| start of | most accurate third | middle third | least accurate third |
|---|---:|---:|---:|
| 1 top | +2.78 pts | +2.82 pts | +3.10 pts |
| 5 top | +1.98 pts | +2.02 pts | +2.29 pts |
| 9 top | +0.60 pts | +0.65 pts | +0.83 pts |

(best single wrong call ahead) A less accurate umpire supplies more chances later, so a challenge in hand is worth more with him behind the plate.

## 5. Season level

| team | games | challenges | overturned | success | wins added | runs added | left on the table | lost when out | umpire net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIN | 163 | 370 | 228 | 62% | +4.55 | +47.6 | 7.23 | 0.55 | -1.20 |
| MIN | 163 | 448 | 245 | 55% | +4.43 | +47.7 | 5.42 | 1.09 | +0.06 |
| ATH | 162 | 374 | 225 | 60% | +4.30 | +48.4 | 6.38 | 0.62 | -0.30 |
| CWS | 163 | 441 | 218 | 49% | +4.20 | +40.4 | 5.37 | 1.34 | +0.27 |
| MIA | 162 | 401 | 227 | 57% | +4.13 | +40.2 | 6.10 | 0.67 | -0.06 |
| LAA | 163 | 369 | 198 | 54% | +4.11 | +42.1 | 6.58 | 0.54 | -0.59 |
| COL | 163 | 384 | 209 | 54% | +3.81 | +41.9 | 4.80 | 0.88 | -0.24 |
| BOS | 164 | 336 | 184 | 55% | +3.60 | +38.8 | 7.13 | 0.62 | -1.01 |
| MIL | 163 | 381 | 201 | 53% | +3.58 | +34.8 | 7.01 | 1.01 | -0.17 |
| NYM | 162 | 384 | 197 | 51% | +3.56 | +35.6 | 6.15 | 1.04 | +0.00 |
| SD | 163 | 340 | 170 | 50% | +3.56 | +33.1 | 6.55 | 0.43 | +0.06 |
| TEX | 163 | 293 | 170 | 58% | +3.41 | +35.6 | 6.33 | 0.34 | -0.73 |
| BAL | 161 | 326 | 162 | 50% | +3.41 | +31.4 | 6.90 | 0.72 | -0.46 |
| NYY | 163 | 397 | 217 | 55% | +3.34 | +38.8 | 4.81 | 0.80 | +2.26 |
| SEA | 162 | 335 | 170 | 51% | +3.34 | +34.6 | 6.80 | 1.01 | -2.08 |
| CHC | 164 | 338 | 199 | 59% | +3.31 | +38.2 | 5.65 | 0.55 | +1.29 |
| AZ | 163 | 307 | 184 | 60% | +3.28 | +31.5 | 6.57 | 0.22 | -0.23 |
| CLE | 163 | 338 | 165 | 49% | +3.12 | +27.8 | 6.11 | 1.50 | +0.15 |
| STL | 162 | 285 | 141 | 49% | +3.11 | +28.7 | 6.48 | 0.70 | -0.10 |
| TOR | 162 | 340 | 188 | 55% | +3.08 | +34.6 | 5.60 | 0.85 | +1.63 |
| WSH | 162 | 338 | 171 | 51% | +3.03 | +30.7 | 6.29 | 1.08 | -0.53 |
| LAD | 163 | 296 | 160 | 54% | +2.98 | +30.2 | 6.01 | 0.46 | +0.38 |
| PHI | 163 | 333 | 191 | 57% | +2.98 | +32.8 | 5.95 | 0.67 | -0.25 |
| HOU | 163 | 385 | 201 | 52% | +2.98 | +32.0 | 4.97 | 1.31 | +0.72 |
| ATL | 164 | 303 | 161 | 53% | +2.97 | +31.3 | 5.43 | 0.78 | +0.06 |
| DET | 163 | 331 | 190 | 57% | +2.96 | +33.0 | 6.31 | 0.53 | +0.45 |
| KC | 163 | 332 | 184 | 55% | +2.96 | +30.7 | 5.82 | 0.73 | +0.29 |
| SF | 162 | 334 | 171 | 51% | +2.75 | +30.8 | 5.94 | 0.74 | +0.42 |
| PIT | 162 | 355 | 167 | 47% | +2.70 | +28.5 | 6.70 | 0.68 | +0.13 |
| TB | 163 | 341 | 180 | 53% | +2.63 | +27.7 | 5.51 | 0.87 | -0.21 |

- **wins added**: win probability gained by the team's overturned challenges.
- **left on the table**: expected wins in wrong calls against the team that it could have challenged (it had one left) and didn't.
- **lost when out**: expected wins in wrong calls against it after it had run out.
- **umpire net**: wrong calls that stood in its favour minus those that stood against it.

Sensitivity: with p(s) as it stands on every pitch — its curve held flat beyond the range of challenged pitches — the league totals would be 55,675 expected wrong calls (vs 24,083), 774.5 wins moved (vs 308.4) and 13.2% won back (vs 33.1%). The gap is the flat tail summed over hundreds of thousands of easy calls; the challenged pitches themselves are unaffected.

## 6. Series level

784 series. A challenge's win probability only changes a series result when its game decided the series (flip that game and the team wins, splits or loses the series instead). Summed over those deciding games, challenges changed the expected series results by **29.3 series** across the season; 212 series moved by 5+ points.

| series | team | games | net challenge value | value in deciding games |
|---:|---|---:|---:|---:|
| 557 | CIN | 3 | +44.8 pts | +45.4 pts |
| 98 | SEA | 4 | +28.4 pts | +28.4 pts |
| 161 | CIN | 3 | +30.5 pts | +27.6 pts |
| 235 | COL | 3 | +27.5 pts | +23.2 pts |
| 513 | MIA | 3 | +28.0 pts | +21.9 pts |
| 293 | AZ | 3 | +22.3 pts | +21.1 pts |
| 200 | CLE | 4 | +20.5 pts | +20.5 pts |
| 366 | LAA | 4 | +19.9 pts | +20.1 pts |

Season level and series level are the same numbers added up differently: Week 11's policy will be scored the same way (wins per season), and Spring's simulator replays whole seasons under it.

## 7. Players — who challenges well, and does it matter to ABS?

| role | challenges | overturned [95%] | mean p(s) of the pitches chosen | above expected (per 100) | win prob. per challenge |
|---|---:|---|---:|---:|---:|
| batter | 4,757 | 48.9% [47.5%, 50.3%] | 49.0% | -0.1 | +0.92 pts |
| catcher | 5,600 | 58.6% [57.3%, 59.8%] | 58.4% | +0.2 | +1.02 pts |
| pitcher | 178 | 39.3% [32.4%, 46.7%] | 39.2% | +0.1 | +0.73 pts |

- batting side — clearly wrong calls challenged (p(s) ≥ 90%): 20.0% of 11,157

- fielding side — clearly wrong calls challenged (p(s) ≥ 90%): 33.6% of 9,522

**Two different questions:**

1. *Given the pitch, does WHO challenged change what ABS rules?* Permutation test on overturned − p(s), shuffling challengers within role: **p = 0.170**; size of real differences ±0.6 points (split-half reliability 0.02 over 257 clusters). ABS sees only the ball, so this should be no or tiny — and then **player accuracy does not belong in p(s)**: the location model already says what the challenger's identity could.
2. *Do challengers differ in WHICH pitches they challenge?* Permutation test on the p(s) of the pitches they chose: **p = 0.006**; size ±4.4 points of p(s) (split-half reliability 0.21 over 257 clusters). This is the real meaning of "a more accurate challenger": better judgment about which calls are wrong. It belongs in the **decision**, not in p(s) — see WEEK10.md, "Where each idea goes".

Challengers with 10+ challenges, best and worst by shrunk success rate (pulled toward their role's rate in proportion to how few challenges they have):

| challenger | role | challenges | overturned | shrunk rate | mean p(s) chosen | above expected |
|---|---|---:|---:|---:|---:|---:|
| J.T. Realmuto | catcher | 86 | 70 | 68.4% | 79.8% | +1.4 |
| Tyler Stephenson | catcher | 136 | 93 | 63.9% | 68.9% | -0.8 |
| Hunter Goodman | catcher | 113 | 77 | 63.3% | 67.8% | +0.4 |
| Carson Kelly | catcher | 93 | 64 | 63.2% | 68.0% | +0.8 |
| Salvador Perez | catcher | 65 | 46 | 63.0% | 70.0% | +0.5 |
| Spencer Horwitz | batter | 19 | 5 | 48.0% | 30.5% | -0.8 |
| José Fermín | batter | 13 | 2 | 48.0% | 15.4% | -0.0 |
| Willi Castro | batter | 23 | 6 | 47.9% | 26.1% | -0.0 |
| Josh Naylor | batter | 26 | 7 | 47.8% | 27.0% | -0.0 |
| Ceddanne Rafaela | batter | 24 | 6 | 47.8% | 25.1% | -0.0 |

## 8. Umpires — who misses more, and what it changes

- 1,430 called pitches in 12 games have no home-plate umpire on record; they are left out of this section.
- Do umpires differ in how often ABS would reverse them, for pitches of the same difficulty? Permutation test (whole games shuffled between umpires): **p = 0.001**; size ±0.57 wrong calls per 100 called pitches; reliability across odd and even games 0.73 over 90 clusters.
- Given the pitch, does the umpire change the chance a challenge succeeds? **p = 0.790**; size ±0.0 points.
  ABS doesn't know who the umpire was, and the data agree: umpire accuracy doesn't belong in p(s).

| umpires | count | wrong calls a game | vs an average umpire (per 100 calls) | win prob. at stake a game | of it challengeable |
|---|---:|---:|---:|---:|---:|
| most accurate third | 30 | 8.71 | -0.76 | +11.38 pts | +10.50 pts |
| middle third | 29 | 9.90 | -0.01 | +12.39 pts | +11.51 pts |
| least accurate third | 30 | 11.09 | +0.78 | +14.19 pts | +13.10 pts |

Umpire accuracy changes **how many chances** a team gets, not whether a challenge succeeds. That is exactly where it enters the decision: the value of holding a challenge depends on how many wrong calls are still likely to come (section 4).

## 9. Where each idea goes

| idea | what Week 10 did | where it is finished |
|---|---|---|
| value of holding a challenge | the stakes still ahead and their perfect-foresight ceiling (section 4) | **Week 11** — V(s, r) − V(s, r − 1) from backward induction |
| inning-level effect | runs an overturn is worth (RE288), checked against what happened (section 2) | done; Week 11 can optimise runs or wins |
| series and season | wins added, left on the table, series swung (sections 5–6) | **Spring, Feb 19** — replay seasons under the optimal policy |
| player accuracy | tested: in p(s)? which pitches? (section 7) | **Spring, Feb 26 (RQ3)** — decision regret per player; optionally Week 11's "realistic information" variant |
| umpire accuracy | tested; opportunity supply by umpire (sections 4, 8) | **Week 11** — umpire-specific chance of future wrong calls in the MDP; **Spring, Mar 19 (RQ4)** — did umpires change during the season? |

## Figures (`figures/week10/`)

- `w10_07_opportunities.png` — every opportunity: chance it was wrong vs what it was worth
- `w10_08_value_ahead.png` — the stakes still ahead, by inning and by umpire
- `w10_09_inning_experiment.png` — runs after overturned vs upheld challenges vs predicted
- `w10_10_team_wins.png` — wins added and left on the table, by team
- `w10_11_challengers.png` — challengers: the pitches they choose vs how often they win
- `w10_12_umpires.png` — umpires: wrong calls per game against the chances they create