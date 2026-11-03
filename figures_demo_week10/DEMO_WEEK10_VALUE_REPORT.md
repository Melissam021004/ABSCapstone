# Week 10 — What the 2026 challenges were worth

Generated 2026-10-30 21:44 by `scripts/20_week10_challenge_value.py` · p(s) 2026.10.30-f4c02dc0 · V(s) 2026.10.30-d4a59c02 · cohort `140cf6b8b98e1757` · 1000 game-level bootstrap resamples · 999 permutations per skill test

> **DEMO DATA** — a simulated season with planted differences between umpires and between challengers. The method is real; the numbers are not.

> Read this next to `WEEK10.md` (Part B), which explains every section.

## In one paragraph

Umpires' wrong calls moved **303.3 wins** across the league (21,670 expected wrong calls on 372,816 called pitches). Overturned challenges won back **67.2** of them (22.2%). The 7,296 challenges were worth 67.4 wins in expectation and delivered 67.2. The best team added 3.17 wins with challenges; the median team 2.41.

## 0. Inputs

- recorded call matches who challenged on 100.0% of 3,063 upheld and 100.0% of 4,233 overturned challenges
- verdict: ORIGINAL umpire call recorded — no repair needed
- 372,817 called pitches valued with p(s) and V(s) (1 with a location outside p(s)'s range skipped); 356,013 challenge opportunities (a call against a side with a challenge left, worked out from the umpire's ORIGINAL call); 7,296 challenges.

## 1. What a call is worth

Win probability the challenging side gains if the call is overturned (`dwp`), and runs (to the end of the inning) for the same side:

| set | n | mean | median | 90th percentile | runs (mean) |
|---|---:|---:|---:|---:|---:|
| every opportunity | 356,013 | +1.41 pts | +0.88 pts | +3.08 pts | +0.152 |
| wrong calls (p(s) ≥ 50%) | 20,046 | +1.40 pts | +0.89 pts | +3.06 pts | +0.153 |
| challenged | 7,296 | +1.57 pts | +1.00 pts | +3.52 pts | +0.163 |
| overturned | 4,233 | +1.59 pts | +1.00 pts | +3.60 pts | +0.167 |

Most calls are worth little; a few are worth a lot. The ten most valuable overturns:

| date | inning | score (home–away) | count | outs | challenging team | role | gain |
|---|---|---|---|---:|---|---|---:|
| Jun 21 | 8 bot | 3–4 | 3-2 | 2 | TBR (Matt Semien) | pitcher | +20.9 pts |
| Jul 04 | 3 top | 1–1 | 3-2 | 2 | MIN (Luis Alvarez) | batter | +19.3 pts |
| Jul 16 | 9 top | 2–1 | 3-2 | 0 | SEA (Riley Ramirez) | batter | +18.0 pts |
| Apr 20 | 6 top | 0–0 | 3-2 | 0 | WSN (Logan Henderson) | batter | +17.5 pts |
| May 25 | 9 bot | 3–6 | 3-2 | 1 | BAL (Adley Arraez) | batter | +17.0 pts |
| Jun 11 | 8 top | 4–1 | 3-2 | 2 | PIT (Patrick Heim) | batter | +16.1 pts |
| Jun 09 | 5 bot | 1–2 | 3-2 | 1 | NYM (Matt Bichette) | batter | +15.8 pts |
| Apr 08 | 8 bot | 3–4 | 2-2 | 2 | HOU (Gabriel Bichette) | batter | +15.7 pts |
| May 10 | 7 top | 7–6 | 3-2 | 0 | CIN (Gunnar Smith) | batter | +15.6 pts |
| Aug 23 | 9 top | 2–2 | 3-2 | 1 | MIN (Matt Murphy) | batter | +15.4 pts |

Every challenge with its value: `docs/week10_value_challenges.csv`.

## 2. Inning level — more time before the third out

An overturned strike keeps an at-bat (or an inning) alive; an overturned ball takes a baserunner away. Whether a challenge is overturned is decided by where the ball was, not by the game situation, so overturned and upheld challenges are close to a natural experiment. Runs the batting team scored from the challenged pitch to the end of the inning (innings 1–8; walk-offs cut innings short):

| challenging side | n | after overturned | after upheld | gap [95%] | model's predicted gap |
|---|---:|---:|---:|---|---:|
| batting | 4,261 | 0.527 | 0.420 | +0.106 [+0.044, +0.168] | +0.120 |
| fielding | 2,310 | 0.373 | 0.539 | -0.166 [-0.240, -0.090] | -0.132 |
| all | 6,571 | 0.485 | 0.475 | +0.010 [-0.040, +0.058] | +0.025 |

Read: on a batting-side challenge the gap should be positive (an overturned strike helps the batting team score); on a fielding-side challenge negative (an overturned ball takes runs away). If the observed gap's interval covers the model's prediction, the run-expectancy engine behind V(s) is right about what a challenge does to an inning.

## 3. Game level

| challenging side | n | won after overturned | won after upheld | gap [95%] | predicted |
|---|---:|---:|---:|---|---:|
| batting | 4,742 | 51.2% | 49.9% | +1.3 pts [-1.7 pts, +4.4 pts] | +1.7 pts |
| fielding | 2,554 | 47.7% | 48.5% | -0.8 pts [-4.8 pts, +3.0 pts] | +1.1 pts |
| all | 7,296 | 50.3% | 49.3% | +1.0 pts [-1.3 pts, +3.3 pts] | +2.0 pts |

The game-level gap is a few points at most and games are noisy, so expect a wide interval; the check is that it doesn't contradict the prediction.

## 4. Value still ahead — the raw material of "holding a challenge"

At the start of each half-inning, averaged over every game and team: how much win probability is still at stake in the calls that will go against that team later.

| start of | stakes still ahead | best single wrong call ahead | best two | a wrong call now is worth |
|---|---:|---:|---:|---:|
| 1 top | +6.24 pts | +2.94 pts | +4.29 pts | +1.34 pts |
| 2 top | +5.57 pts | +2.78 pts | +4.00 pts | +1.38 pts |
| 3 top | +4.88 pts | +2.58 pts | +3.65 pts | +1.32 pts |
| 4 top | +4.20 pts | +2.36 pts | +3.28 pts | +1.29 pts |
| 5 top | +3.54 pts | +2.14 pts | +2.89 pts | +1.31 pts |
| 6 top | +2.86 pts | +1.84 pts | +2.43 pts | +1.28 pts |
| 7 top | +2.22 pts | +1.54 pts | +1.96 pts | +1.47 pts |
| 8 top | +1.55 pts | +1.17 pts | +1.42 pts | +1.41 pts |
| 9 top | +0.89 pts | +0.72 pts | +0.83 pts | +1.35 pts |
| 9 bot | +0.93 pts | +0.75 pts | +0.85 pts | +2.16 pts |

- **best single wrong call ahead** is what ONE challenge saved for later could win with perfect foresight — an upper bound on the value of holding it. A real team doesn't know which call will be the big one, and a failed challenge is lost; Week 11's backward induction computes the real value, V(s, r) − V(s, r − 1), below this ceiling.
- Compare it with **a wrong call now**: early in the game the best call still to come is worth several times a typical call now — the case for holding. Late, the two converge, and the last innings are where a challenge in hand is worth spending.

By the umpire's accuracy (thirds of umpires, adjusted for how hard their pitches were):

| start of | most accurate third | middle third | least accurate third |
|---|---:|---:|---:|
| 1 top | +2.66 pts | +2.94 pts | +3.24 pts |
| 5 top | +1.90 pts | +2.13 pts | +2.38 pts |
| 9 top | +0.60 pts | +0.69 pts | +0.87 pts |

(best single wrong call ahead) A less accurate umpire supplies more chances later, so a challenge in hand is worth more with him behind the plate.

## 5. Season level

| team | games | challenges | overturned | success | wins added | runs added | left on the table | lost when out | umpire net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLE | 183 | 284 | 169 | 60% | +3.17 | +32.1 | 9.02 | 0.37 | -0.62 |
| NYM | 196 | 276 | 156 | 57% | +2.77 | +30.2 | 9.26 | 0.92 | +0.22 |
| NYY | 178 | 259 | 156 | 60% | +2.65 | +28.1 | 8.36 | 0.30 | -0.59 |
| BOS | 183 | 293 | 164 | 56% | +2.63 | +27.6 | 8.14 | 0.74 | +0.48 |
| MIN | 172 | 261 | 146 | 56% | +2.59 | +26.6 | 8.03 | 0.55 | +0.07 |
| COL | 187 | 281 | 175 | 62% | +2.58 | +26.2 | 8.18 | 0.31 | +0.76 |
| WSN | 205 | 307 | 174 | 57% | +2.57 | +28.4 | 8.32 | 0.59 | +0.97 |
| CHC | 180 | 261 | 160 | 61% | +2.56 | +27.0 | 8.99 | 0.20 | -1.23 |
| CIN | 208 | 326 | 184 | 56% | +2.52 | +27.8 | 8.54 | 0.53 | +0.42 |
| MIL | 185 | 258 | 149 | 58% | +2.51 | +24.5 | 7.97 | 0.58 | +1.44 |
| MIA | 138 | 237 | 139 | 59% | +2.49 | +24.8 | 6.99 | 0.57 | -0.99 |
| SEA | 168 | 268 | 156 | 58% | +2.48 | +24.4 | 8.68 | 0.34 | -0.54 |
| DET | 184 | 263 | 165 | 63% | +2.47 | +24.8 | 9.05 | 0.22 | -0.78 |
| SDP | 135 | 211 | 136 | 64% | +2.46 | +22.9 | 6.24 | 0.30 | -0.08 |
| TBR | 152 | 236 | 134 | 57% | +2.44 | +24.5 | 6.44 | 0.51 | +0.71 |
| STL | 152 | 256 | 151 | 59% | +2.38 | +26.2 | 6.85 | 0.35 | +0.28 |
| HOU | 155 | 225 | 125 | 56% | +2.25 | +24.3 | 6.60 | 0.30 | +0.87 |
| CHW | 157 | 256 | 145 | 57% | +2.13 | +22.4 | 7.20 | 0.61 | -0.40 |
| TEX | 174 | 259 | 148 | 57% | +2.13 | +25.8 | 7.01 | 0.64 | -0.35 |
| PHI | 161 | 230 | 127 | 55% | +2.08 | +22.8 | 7.57 | 0.64 | -0.10 |
| PIT | 155 | 233 | 139 | 60% | +2.08 | +25.3 | 7.98 | 0.61 | -0.92 |
| TOR | 159 | 223 | 136 | 61% | +2.06 | +19.8 | 6.95 | 0.17 | +0.43 |
| SFG | 144 | 231 | 135 | 58% | +1.91 | +20.0 | 5.99 | 0.37 | +0.14 |
| BAL | 137 | 226 | 125 | 55% | +1.85 | +21.8 | 6.36 | 0.51 | -0.23 |
| KCR | 155 | 228 | 125 | 55% | +1.85 | +18.3 | 7.67 | 0.17 | -0.11 |
| LAD | 127 | 180 | 108 | 60% | +1.82 | +18.8 | 4.97 | 0.36 | +1.64 |
| ATL | 128 | 193 | 113 | 59% | +1.81 | +18.4 | 6.00 | 0.12 | +0.35 |
| ATH | 133 | 182 | 97 | 53% | +1.52 | +14.2 | 5.51 | 1.06 | -0.82 |
| ARI | 123 | 164 | 95 | 58% | +1.24 | +13.0 | 6.43 | 0.34 | +0.06 |
| LAA | 146 | 189 | 101 | 53% | +1.21 | +15.9 | 6.90 | 0.50 | -1.11 |

- **wins added**: win probability gained by the team's overturned challenges.
- **left on the table**: expected wins in wrong calls against the team that it could have challenged (it had one left) and didn't.
- **lost when out**: expected wins in wrong calls against it after it had run out.
- **umpire net**: wrong calls that stood in its favour minus those that stood against it.

## 6. Series level

804 series. A challenge's win probability only changes a series result when its game decided the series (flip that game and the team wins, splits or loses the series instead). Summed over those deciding games, challenges changed the expected series results by **22.2 series** across the season; 157 series moved by 5+ points.

| series | team | games | net challenge value | value in deciding games |
|---:|---|---:|---:|---:|
| 439 | MIL | 4 | +23.8 pts | +23.5 pts |
| 76 | BAL | 2 | +23.0 pts | +23.0 pts |
| 212 | CLE | 2 | +19.8 pts | +19.8 pts |
| 564 | SEA | 3 | +17.8 pts | +17.8 pts |
| 283 | BOS | 3 | +19.3 pts | +17.7 pts |
| 760 | LAD | 3 | +15.3 pts | +17.1 pts |
| 609 | SDP | 2 | +16.3 pts | +16.3 pts |
| 620 | SDP | 2 | +16.2 pts | +16.2 pts |

Season level and series level are the same numbers added up differently: Week 11's policy will be scored the same way (wins per season), and Spring's simulator replays whole seasons under it.

## 7. Players — who challenges well, and does it matter to ABS?

| role | challenges | overturned [95%] | mean p(s) of the pitches chosen | above expected (per 100) | win prob. per challenge |
|---|---:|---|---:|---:|---:|
| batter | 4,742 | 64.9% [63.5%, 66.2%] | 65.0% | -0.2 | +1.07 pts |
| catcher | 1,872 | 48.1% [45.8%, 50.3%] | 47.5% | +0.6 | +0.65 pts |
| pitcher | 682 | 37.7% [34.1%, 41.4%] | 38.6% | -0.9 | +0.62 pts |

- batting side — clearly wrong calls challenged (p(s) ≥ 90%): 21.9% of 13,218

- fielding side — clearly wrong calls challenged (p(s) ≥ 90%): 29.2% of 3,281

**Two different questions:**

1. *Given the pitch, does WHO challenged change what ABS rules?* Permutation test on overturned − p(s), shuffling challengers within role: **p = 0.124**; size of real differences ±1.1 points (split-half reliability -0.13 over 241 clusters). ABS sees only the ball, so this should be no or tiny — and then **player accuracy does not belong in p(s)**: the location model already says what the challenger's identity could.
2. *Do challengers differ in WHICH pitches they challenge?* Permutation test on the p(s) of the pitches they chose: **p = 0.001**; size ±11.2 points of p(s) (split-half reliability 0.65 over 241 clusters). This is the real meaning of "a more accurate challenger": better judgment about which calls are wrong. It belongs in the **decision**, not in p(s) — see WEEK10.md, "Where each idea goes".

Challengers with 10+ challenges, best and worst by shrunk success rate (pulled toward their role's rate in proportion to how few challenges they have):

| challenger | role | challenges | overturned | shrunk rate | mean p(s) chosen | above expected |
|---|---|---:|---:|---:|---:|---:|
| Gunnar Tucker | batter | 34 | 33 | 88.2% | 96.1% | +0.3 |
| Jose Greene | batter | 30 | 29 | 87.1% | 95.7% | +0.3 |
| Patrick Smith | batter | 29 | 28 | 86.8% | 96.1% | +0.1 |
| Luis Alvarez | batter | 19 | 19 | 85.8% | 97.5% | +0.5 |
| Will Smith | batter | 25 | 23 | 82.8% | 89.5% | +0.6 |
| Cal Ramirez | catcher | 52 | 16 | 35.2% | 39.2% | -4.4 |
| Rafael Henderson | pitcher | 10 | 3 | 34.6% | 21.7% | +0.8 |
| Jonah Wells | batter | 18 | 2 | 33.5% | 16.2% | -0.9 |
| Elly Diaz | catcher | 42 | 11 | 32.7% | 29.2% | -1.2 |
| Patrick Ramirez | catcher | 55 | 15 | 32.4% | 26.2% | +0.6 |

## 8. Umpires — who misses more, and what it changes

- Do umpires differ in how often ABS would reverse them, for pitches of the same difficulty? Permutation test (whole games shuffled between umpires): **p = 0.001**; size ±1.07 wrong calls per 100 called pitches; reliability across odd and even games 0.97 over 20 clusters.
- Given the pitch, does the umpire change the chance a challenge succeeds? **p = 0.371**; size ±0.2 points.
  ABS doesn't know who the umpire was, and the data agree: umpire accuracy doesn't belong in p(s).

| umpires | count | wrong calls a game | vs an average umpire (per 100 calls) | win prob. at stake a game | of it challengeable |
|---|---:|---:|---:|---:|---:|
| most accurate third | 7 | 7.20 | -1.17 | +10.21 pts | +9.74 pts |
| middle third | 6 | 8.99 | +0.03 | +12.40 pts | +11.81 pts |
| least accurate third | 7 | 10.57 | +1.18 | +14.80 pts | +14.16 pts |

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