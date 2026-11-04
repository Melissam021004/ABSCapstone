# Week 10 — What the 2026 challenges were worth

Generated 2026-11-04 09:07 by `scripts/20_week10_challenge_value.py` · p(s) 2026.10.29-b81183b4 · V(s) 2026.11.04-824ec8be · cohort `419af01ad9eee03f` · 1000 game-level bootstrap resamples · 999 permutations per skill test

> Read this next to `WEEK10.md` (Part B), which explains every section.

## In one paragraph

Umpires' wrong calls moved **272.3 wins** across the league (23,428 expected wrong calls on 370,883 called pitches). Overturned challenges won back **57.5** of them (21.1%). The 7,897 challenges were worth 57.5 wins in expectation and delivered 57.5. The best team added 2.50 wins with challenges; the median team 1.91.

## 0. Inputs

- recorded call matches who challenged on 100.0% of 3,453 upheld and 0.0% of 4,417 overturned challenges
- verdict: FINAL (post-review) call recorded — repaired from who challenged (4,417 challenged pitches corrected)
- 370,909 called pitches valued with p(s) and V(s) (26 with a location outside p(s)'s range skipped); 358,321 challenge opportunities (a call against a side with a challenge left, worked out from the umpire's ORIGINAL call); 7,897 challenges.
- **Where p(s) is used.** p(s) learned from challenged pitches, which sit between -4.9 and +4.4 inches against the call (99% of them). Beyond that range its curve is held flat, so a call that is clearly right — a strike in the middle of the zone — would still get the small chance of being wrong it had at the edge of the range. There the ABS call isn't in doubt, so it is taken from the zone geometry instead: 192,150 called pitches further on the right side count as calls that stand, 70 further on the wrong side as calls ABS would reverse. Inside the range, the locked p(s) is used unchanged.

## 1. What a call is worth

Win probability the challenging side gains if the call is overturned (`dwp`), and runs (to the end of the inning) for the same side:

| set | n | mean | median | 90th percentile | runs (mean) |
|---|---:|---:|---:|---:|---:|
| every opportunity | 358,321 | +1.36 pts | +0.84 pts | +3.00 pts | +0.150 |
| wrong calls (p(s) ≥ 50%) | 21,995 | +1.14 pts | +0.76 pts | +2.42 pts | +0.126 |
| challenged | 7,897 | +1.40 pts | +0.90 pts | +3.08 pts | +0.141 |
| overturned | 4,424 | +1.30 pts | +0.86 pts | +2.88 pts | +0.133 |

Most calls are worth little; a few are worth a lot. The ten most valuable overturns:

| date | inning | score (home–away) | count | outs | challenging team | role | gain |
|---|---|---|---|---:|---|---|---:|
| Jun 06 | 9 top | 6–5 | 3-1 | 2 | STL (Jimmy Crooks) | catcher | +27.1 pts |
| Sep 17 | 10 bot | 3–4 | 1-2 | 1 | LAA (Mike Trout) | batter | +20.4 pts |
| May 31 | 10 top | 2–2 | 1-2 | 1 | AZ (Adrian Del Castillo) | batter | +15.2 pts |
| Jul 27 | 9 bot | 2–3 | 0-2 | 2 | WSH (Luis García Jr.) | batter | +13.9 pts |
| Sep 08 | 10 top | 2–2 | 2-2 | 1 | AZ (Nolan Arenado) | batter | +13.8 pts |
| Jun 01 | 10 top | 2–2 | 2-2 | 0 | NYM (Juan Soto) | batter | +13.0 pts |
| Sep 23 | 4 bot | 1–2 | 3-1 | 2 | SD (Ethan Salas) | catcher | +12.4 pts |
| Jun 06 | 9 top | 6–5 | 1-0 | 2 | CIN (Sal Stewart) | batter | +11.9 pts |
| Apr 04 | 9 top | 9–7 | 2-2 | 1 | MIA (Owen Caissie) | batter | +11.5 pts |
| Aug 17 | 10 top | 5–5 | 2-2 | 2 | CWS (Braden Montgomery) | batter | +11.4 pts |

Every challenge with its value: `docs/week10_value_challenges.csv`.

## 2. Inning level — more time before the third out

An overturned strike keeps an at-bat (or an inning) alive; an overturned ball takes a baserunner away. Whether a challenge is overturned is decided by where the ball was, not by the game situation, so overturned and upheld challenges are close to a natural experiment. Runs the batting team scored from the challenged pitch to the end of the inning (innings 1–8; walk-offs cut innings short):

| challenging side | n | after overturned | after upheld | gap [95%] | model's predicted gap |
|---|---:|---:|---:|---|---:|
| batting | 2,890 | 0.526 | 0.499 | +0.027 [-0.048, +0.105] | +0.053 |
| fielding | 3,774 | 0.497 | 0.519 | -0.022 [-0.096, +0.043] | -0.074 |
| all | 6,664 | 0.510 | 0.511 | -0.001 [-0.050, +0.049] | -0.019 |

Read: on a batting-side challenge the gap should be positive (an overturned strike helps the batting team score); on a fielding-side challenge negative (an overturned ball takes runs away). If the observed gap's interval covers the model's prediction, the run-expectancy engine behind V(s) is right about what a challenge does to an inning.

## 3. Game level

| challenging side | n | won after overturned | won after upheld | gap [95%] | predicted |
|---|---:|---:|---:|---|---:|
| batting | 3,461 | 48.9% | 45.4% | +3.5 pts [+0.2 pts, +7.1 pts] | +2.7 pts |
| fielding | 4,436 | 49.3% | 53.6% | -4.3 pts [-7.2 pts, -1.4 pts] | -2.4 pts |
| all | 7,897 | 49.1% | 50.2% | -1.0 pts [-3.2 pts, +1.0 pts] | -0.3 pts |

The game-level gap is a few points at most and games are noisy, so expect a wide interval; the check is that it doesn't contradict the prediction.

## 4. Value still ahead — the raw material of "holding a challenge"

At the start of each half-inning, averaged over every game and team: how much win probability is still at stake in the calls that will go against that team later.

| start of | stakes still ahead | best single wrong call ahead | best two | a wrong call now is worth |
|---|---:|---:|---:|---:|
| 1 top | +5.58 pts | +2.45 pts | +3.64 pts | +1.15 pts |
| 2 top | +4.91 pts | +2.29 pts | +3.37 pts | +1.16 pts |
| 3 top | +4.29 pts | +2.13 pts | +3.08 pts | +1.08 pts |
| 4 top | +3.68 pts | +1.97 pts | +2.79 pts | +1.11 pts |
| 5 top | +3.11 pts | +1.78 pts | +2.48 pts | +1.08 pts |
| 6 top | +2.50 pts | +1.55 pts | +2.09 pts | +1.08 pts |
| 7 top | +1.91 pts | +1.27 pts | +1.66 pts | +1.09 pts |
| 8 top | +1.32 pts | +0.96 pts | +1.19 pts | +1.11 pts |
| 9 top | +0.74 pts | +0.59 pts | +0.69 pts | +1.10 pts |
| 9 bot | +0.82 pts | +0.66 pts | +0.77 pts | +1.78 pts |

- **best single wrong call ahead** is what ONE challenge saved for later could win with perfect foresight — an upper bound on the value of holding it. A real team doesn't know which call will be the big one, and a failed challenge is lost; Week 11's backward induction computes the real value, V(s, r) − V(s, r − 1), below this ceiling.
- Compare it with **a wrong call now**: early in the game the best call still to come is worth several times a typical call now — the case for holding. Late, the two converge, and the last innings are where a challenge in hand is worth spending.

By the umpire's accuracy (thirds of umpires, adjusted for how hard their pitches were):

| start of | most accurate third | middle third | least accurate third |
|---|---:|---:|---:|
| 1 top | +2.30 pts | +2.46 pts | +2.58 pts |
| 5 top | +1.65 pts | +1.80 pts | +1.92 pts |
| 9 top | +0.50 pts | +0.63 pts | +0.65 pts |

(best single wrong call ahead) A less accurate umpire supplies more chances later, so a challenge in hand is worth more with him behind the plate.

## 5. Season level

| team | games | challenges | overturned | success | wins added | runs added | left on the table | lost when out | umpire net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIN | 163 | 278 | 170 | 61% | +2.50 | +24.3 | 7.98 | 0.12 | -1.15 |
| CWS | 163 | 347 | 176 | 51% | +2.32 | +23.3 | 6.16 | 0.83 | +0.28 |
| LAA | 163 | 261 | 148 | 57% | +2.27 | +21.2 | 7.16 | 0.30 | -0.70 |
| MIA | 162 | 295 | 174 | 59% | +2.27 | +21.7 | 6.79 | 0.35 | -0.20 |
| AZ | 163 | 245 | 151 | 62% | +2.23 | +18.2 | 6.94 | 0.20 | -0.31 |
| MIL | 163 | 289 | 155 | 54% | +2.23 | +21.0 | 7.81 | 0.50 | -0.13 |
| MIN | 163 | 326 | 170 | 52% | +2.18 | +22.7 | 6.11 | 0.75 | -0.02 |
| TEX | 163 | 216 | 136 | 63% | +2.17 | +22.6 | 6.78 | 0.12 | -0.71 |
| ATH | 162 | 264 | 162 | 61% | +2.11 | +23.6 | 7.04 | 0.26 | -0.30 |
| NYM | 162 | 293 | 156 | 53% | +2.11 | +21.4 | 6.97 | 0.47 | +0.03 |
| SD | 163 | 254 | 134 | 53% | +2.10 | +18.9 | 6.99 | 0.23 | +0.12 |
| WSH | 162 | 270 | 142 | 53% | +2.10 | +20.3 | 6.86 | 0.80 | -0.59 |
| COL | 163 | 285 | 158 | 55% | +1.98 | +20.3 | 5.29 | 0.58 | -0.14 |
| CHC | 164 | 253 | 154 | 61% | +1.95 | +19.9 | 6.13 | 0.39 | +1.38 |
| TOR | 162 | 276 | 151 | 55% | +1.93 | +20.3 | 5.96 | 0.73 | +1.69 |
| SF | 162 | 259 | 142 | 55% | +1.90 | +20.8 | 6.57 | 0.40 | +0.42 |
| HOU | 163 | 307 | 170 | 55% | +1.88 | +20.6 | 5.74 | 0.82 | +0.74 |
| KC | 163 | 252 | 152 | 60% | +1.85 | +19.1 | 6.54 | 0.28 | +0.29 |
| NYY | 163 | 299 | 173 | 58% | +1.80 | +21.7 | 5.66 | 0.32 | +2.26 |
| ATL | 164 | 221 | 126 | 57% | +1.76 | +17.1 | 6.17 | 0.39 | -0.11 |
| DET | 163 | 260 | 155 | 60% | +1.68 | +20.0 | 6.74 | 0.28 | +0.43 |
| BAL | 161 | 243 | 118 | 49% | +1.68 | +16.0 | 7.46 | 0.47 | -0.41 |
| BOS | 164 | 229 | 130 | 57% | +1.63 | +18.4 | 7.65 | 0.43 | -1.13 |
| CLE | 163 | 248 | 129 | 52% | +1.62 | +15.4 | 7.35 | 0.49 | +0.24 |
| PHI | 163 | 252 | 158 | 63% | +1.61 | +18.5 | 6.43 | 0.49 | -0.19 |
| STL | 162 | 202 | 107 | 53% | +1.61 | +15.4 | 7.27 | 0.12 | -0.03 |
| PIT | 162 | 270 | 133 | 49% | +1.55 | +16.8 | 7.29 | 0.38 | +0.17 |
| TB | 163 | 262 | 152 | 58% | +1.54 | +17.1 | 6.46 | 0.19 | -0.23 |
| SEA | 162 | 226 | 121 | 54% | +1.52 | +15.1 | 7.88 | 0.25 | -2.14 |
| LAD | 163 | 215 | 121 | 56% | +1.45 | +15.9 | 6.38 | 0.33 | +0.44 |

- **wins added**: win probability gained by the team's overturned challenges.
- **left on the table**: expected wins in wrong calls against the team that it could have challenged (it had one left) and didn't.
- **lost when out**: expected wins in wrong calls against it after it had run out.
- **umpire net**: wrong calls that stood in its favour minus those that stood against it.

Sensitivity: with p(s) as it stands on every pitch — its curve held flat beyond the range of challenged pitches — the league totals would be 54,485 expected wrong calls (vs 23,428), 731.5 wins moved (vs 272.3) and 7.9% won back (vs 21.1%). The gap is the flat tail summed over hundreds of thousands of easy calls; the challenged pitches themselves are unaffected.

## 6. Series level

784 series. A challenge's win probability only changes a series result when its game decided the series (flip that game and the team wins, splits or loses the series instead). Summed over those deciding games, challenges changed the expected series results by **16.7 series** across the season; 102 series moved by 5+ points.

| series | team | games | net challenge value | value in deciding games |
|---:|---|---:|---:|---:|
| 356 | LAA | 4 | +17.0 pts | +17.0 pts |
| 513 | MIA | 3 | +21.2 pts | +16.8 pts |
| 598 | SF | 4 | +12.5 pts | +15.6 pts |
| 386 | SD | 3 | +15.4 pts | +15.4 pts |
| 638 | HOU | 3 | +14.5 pts | +15.0 pts |
| 177 | CIN | 3 | +15.2 pts | +14.7 pts |
| 580 | SD | 3 | +14.1 pts | +14.5 pts |
| 784 | WSH | 3 | +13.2 pts | +14.4 pts |

Season level and series level are the same numbers added up differently: Week 11's policy will be scored the same way (wins per season), and Spring's simulator replays whole seasons under it.

## 7. Players — who challenges well, and does it matter to ABS?

| role | challenges | overturned [95%] | mean p(s) of the pitches chosen | above expected (per 100) | win prob. per challenge |
|---|---:|---|---:|---:|---:|
| batter | 3,460 | 57.8% [56.2%, 59.4%] | 57.8% | -0.0 | +0.86 pts |
| catcher | 4,278 | 55.3% [53.8%, 56.8%] | 55.4% | -0.1 | +0.63 pts |
| pitcher | 132 | 39.4% [31.5%, 47.9%] | 39.6% | -0.2 | +0.43 pts |

- batting side — clearly wrong calls challenged (p(s) ≥ 90%): 17.0% of 11,299

- fielding side — clearly wrong calls challenged (p(s) ≥ 90%): 25.7% of 8,988

**Two different questions:**

1. *Given the pitch, does WHO challenged change what ABS rules?* Permutation test on overturned − p(s), shuffling challengers within role: **p = 0.397**; size of real differences ±0.4 points (split-half reliability -0.07 over 213 clusters). ABS sees only the ball, so this should be no or tiny — and then **player accuracy does not belong in p(s)**: the location model already says what the challenger's identity could.
2. *Do challengers differ in WHICH pitches they challenge?* Permutation test on the p(s) of the pitches they chose: **p = 0.002**; size ±5.4 points of p(s) (split-half reliability 0.16 over 213 clusters). This is the real meaning of "a more accurate challenger": better judgment about which calls are wrong. It belongs in the **decision**, not in p(s) — see WEEK10.md, "Where each idea goes".

Challengers with 10+ challenges, best and worst by shrunk success rate (pulled toward their role's rate in proportion to how few challenges they have):

| challenger | role | challenges | overturned | shrunk rate | mean p(s) chosen | above expected |
|---|---|---:|---:|---:|---:|---:|
| J.T. Realmuto | catcher | 66 | 53 | 65.7% | 78.6% | +1.1 |
| Miguel Amaya | catcher | 59 | 41 | 60.8% | 70.7% | -0.7 |
| Salvador Perez | catcher | 51 | 36 | 60.7% | 70.5% | +0.1 |
| Tyler Stephenson | catcher | 111 | 72 | 60.5% | 65.2% | -0.4 |
| Hunter Goodman | catcher | 81 | 53 | 60.0% | 65.7% | -0.2 |
| Edgar Quero | catcher | 72 | 31 | 49.9% | 42.7% | +0.3 |
| Francisco Alvarez | catcher | 70 | 29 | 49.3% | 41.8% | -0.3 |
| Carlos Narváez | catcher | 60 | 24 | 49.3% | 41.3% | -0.8 |
| Brandon Valenzuela | catcher | 54 | 21 | 49.2% | 38.5% | +0.2 |
| Freddy Fermin | catcher | 53 | 19 | 48.2% | 37.3% | -0.8 |

## 8. Umpires — who misses more, and what it changes

- 1,430 called pitches in 12 games have no home-plate umpire on record; they are left out of this section.
- Do umpires differ in how often ABS would reverse them, for pitches of the same difficulty? Permutation test (whole games shuffled between umpires): **p = 0.001**; size ±0.54 wrong calls per 100 called pitches; reliability across odd and even games 0.71 over 90 clusters.
- Given the pitch, does the umpire change the chance a challenge succeeds? **p = 0.560**; size ±0.0 points.
  ABS doesn't know who the umpire was, and the data agree: umpire accuracy doesn't belong in p(s).

| umpires | count | wrong calls a game | vs an average umpire (per 100 calls) | win prob. at stake a game | of it challengeable |
|---|---:|---:|---:|---:|---:|
| most accurate third | 30 | 8.52 | -0.73 | +10.01 pts | +9.60 pts |
| middle third | 29 | 9.65 | +0.01 | +11.14 pts | +10.65 pts |
| least accurate third | 30 | 10.72 | +0.74 | +12.32 pts | +11.74 pts |

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