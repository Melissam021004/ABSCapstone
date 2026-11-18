# What “the first challenge is worth about 6 points” means

*Written by `scripts/25_first_challenge_value.py`. Slides: `figures/showcase/12_first_challenge_value`, `13_first_challenge_one_game`, `14_first_challenge_all_games`.*

## The number in one sentence

At the first pitch of a tied game, with the other team holding both of its challenges, a team with **one** challenge wins **5.8 percentage points** more often than a team with **none**, if it uses it well. A second challenge adds only 0.3 more, so the two are worth 6.1 together: 9.9 wins over 162 games.

| the same team, at the first pitch | chance to win |
|---|---|
| no challenges | 43.9% |
| one challenge | 49.7% |
| two challenges | 50.0% |

With two challenges each the teams are even, at 50%. Take one away and the team drops 0.3 points; take both away and it drops 6.1.

## What it is not

It is **not** the value of challenging the first pitch. Reversing a 0-0 called strike on the leadoff batter is worth only 0.66 points to the batting team, and reversing a 0-0 called ball is worth 0.66 to the fielding team. The 5.8 points is the value of having the challenge **for the whole game**: all the wrong calls still to come that it could fix, each at the moment it comes. Across 2026, the game's first pitch contributed 0.02 points of it on average.

## Why one challenge is worth so much

- About 10 calls a game are wrong, about 4.9 against each team.
- A challenge that succeeds is **kept**. It is lost only when it fails.
- So one challenge, used well, can fix several calls. Across 2026 it would have been used 4.6 times a game and fixed 4.6 of the 4.9 wrong calls against a team.
- That is also why a second challenge adds so little (0.3): a team that challenges only when it's very likely right rarely loses its first one.

## Why the bar at the first pitch is so high

A challenge risks the challenge itself. If the call is reversed, the team gains the call's value and keeps the challenge. If the call stands, the challenge is gone, and with it everything holding it was worth. So a challenge pays only if

> p × gain ≥ (1 − p) × loss, that is, p ≥ loss ÷ (gain + loss)

where p is the chance the call is wrong. For the visiting team's leadoff hitter taking a 0-0 called strike:

| challenges in hand | gain if reversed | loss if it stands | bar = loss ÷ (gain + loss) |
|---|---|---|---|
| one (its last) | 0.66 | 5.76 | 90% |
| two | 0.66 | 0.30 | 31% |

With one challenge left, the team risks the whole value of holding it to gain less than one point, so it needs to be nearly certain. With two, failing costs only the second challenge, which is worth little, so the bar is much lower. As the game goes on there are fewer calls left to use the challenge on, holding it is worth less, and the bar falls. The value of the one challenge at the start of each inning of a tied game:

| start of the | top | bottom |
|---|---|---|
| 1st | 5.83 | 5.58 |
| 2nd | 5.49 | 5.23 |
| 3rd | 5.12 | 4.85 |
| 4th | 4.72 | 4.42 |
| 5th | 4.28 | 3.95 |
| 6th | 3.78 | 3.40 |
| 7th | 3.18 | 2.73 |
| 8th | 2.44 | 1.86 |
| 9th | 1.46 | 0.65 |

Extra innings restart the clock: a team that has run out gets one challenge back in each extra inning.

## One real game, call by call (NYM)

71 calls went against NYM. 65 were probably right (p below 50%) and below the bar; the rest, in order:

| inning and score | runners, outs, count | the call | p | the bar | worth if reversed | one challenge, used well | chance it's still in hand | adds | what really happened |
|---|---|---|---|---|---|---|---|---|---|
| Top of the 2nd · tied 0–0 | Bases empty, one out · 0-1 count | ball | >99% | 91% | 0.52 | challenge | 100% | 0.52 | not challenged |
| Top of the 4th · tied 0–0 | Bases empty, one out · full count (3-2) | ball | >99% | 48% | 5.03 | challenge | 100% | 5.03 | challenged: overturned |
| Top of the 5th · NYM leads 3–0 | Bases empty, no outs · full count (3-2) | ball | <1% | 35% | 5.93 | keep it | 100% | 0.00 | challenged: stood |
| Top of the 7th · NYM leads 4–1 | Bases empty, two outs · 0-1 count | ball | >99% | 82% | 0.24 | challenge | 100% | 0.24 | not challenged |
| Top of the 8th · NYM leads 6–1 | Bases empty, no outs · full count (3-2) | ball | <1% | 24% | 1.87 | keep it | 100% | 0.00 | challenged: stood |
| Bottom of the 8th · NYM leads 6–1 | Bases empty, two outs · 0-0 count | strike | 79% | 95% | 0.01 | keep it | 100% | 0.00 | not challenged |

**Total: 5.79 points** with one challenge used well. The average team-game: 5.77. Each “adds” is the chance the challenge is still in hand × p × the points if reversed. Each time the challenge is used, the chance it's still in hand is multiplied by p, the chance it succeeds. NYM's real challenges, made with two in hand, won 5.02 points.

## Every team in every game

Doing the same tally for all 4,858 team-games of 2026, one challenge used well recovers **5.77 points on average**; the model says **5.83**. Two used well recover 6.08; the model says 6.13. These are two independent calculations: the model solves the game backward from the last out, and the tally walks the real games forward, call by call.

Half of team-games recover less than 4.4 points and 15% recover 10 or more. By inning:

| inning | points per team-game |
|---|---|
| 1st | 0.70 |
| 2nd | 0.63 |
| 3rd | 0.62 |
| 4th | 0.62 |
| 5th | 0.61 |
| 6th | 0.61 |
| 7th | 0.62 |
| 8th | 0.61 |
| 9th | 0.58 |
| extras | 0.18 |

## Fine print

- Both the model and the tally assume the team knows p, the chance each call is wrong. A player sees less than that, so the spring perception model will lower the number a player could really capture.
- The tally follows each game as it was played. After an overturn the game would have gone differently, but on average that washes out: the averages match the model.
- The model's number is for a tied game with the other team holding both of its challenges, which is where every game starts.

