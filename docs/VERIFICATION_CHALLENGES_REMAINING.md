# Verification — `challenges_remaining`

Generated 2026-10-08 by `scripts/11_verify_challenges_remaining.py`.

`challenges_remaining` is derived rather than published, and it is the state variable r in the MDP. This document records the hand-verification the Week 5 plan requires.

Each game below is checked two ways: an independent re-walk of the inventory rules straight from the challenge events, compared against the built table — and then a manual check against the real play-by-play, which is the only one that catches an assumption both implementations share.

## Game 823178 — AZ @ SF, 2026-08-28

- HP umpire: **Alan Porter** · 9 innings · 344 pitches · **12 challenges**
- [Savant game feed](https://baseballsavant.mlb.com/gamefeed?gamePk=823178) · [MLB Gameday](https://www.mlb.com/gameday/823178) · [raw play-by-play](https://statsapi.mlb.com/api/v1/game/823178/playByPlay)

| # | Inn | Count | Challenger | Role | Side | Result | Expected before→after | Table before→after | |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 3T | 0-0 | Andrew Knizner | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 2 | 3T | 0-1 | Andrew Knizner | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 3 | 4T | 1-0 | Andrew Knizner | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 4 | 4B | 0-0 | James McCann | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 5 | 4B | 0-0 | James McCann | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 6 | 5T | 0-0 | Andrew Knizner | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 7 | 5B | 1-0 | James McCann | catcher | away | stands | 2 → 1 | 2 → 1 | ✅ |
| 8 | 5B | 0-0 | James McCann | catcher | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 9 | 5B | 1-1 | Drew Cavanaugh | batter | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 10 | 6T | 2-0 | Drew Cavanaugh | catcher | home | stands | 2 → 1 | 2 → 1 | ✅ |
| 11 | 9T | 1-0 | Nolan Arenado | batter | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 12 | 9B | 2-1 | Osleivis Basabe | batter | home | stands | 1 → 0 | 1 → 0 | ✅ |

- Final inventory: home **0**, away **1**

> ✅ The independent walk agrees with the built table on every challenge in this game.

**Hand-check (open the Savant game feed):**

- [ ] Savant shows 12 challenges in this game
- [ ] Every challenger name and role matches
- [ ] Every result (overturned / stands) matches
- [ ] No side challenges again after losing two
- [ ] Extra-inning grants, if any, look right

## Game 822774 — NYY @ TOR, 2026-08-16

- HP umpire: **Hunter Wendelstedt** · 10 innings · 293 pitches · **11 challenges**
- [Savant game feed](https://baseballsavant.mlb.com/gamefeed?gamePk=822774) · [MLB Gameday](https://www.mlb.com/gameday/822774) · [raw play-by-play](https://statsapi.mlb.com/api/v1/game/822774/playByPlay)

| # | Inn | Count | Challenger | Role | Side | Result | Expected before→after | Table before→after | |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 1T | 0-0 | Trent Grisham | batter | away | stands | 2 → 1 | 2 → 1 | ✅ |
| 2 | 3T | 0-0 | José Caballero | batter | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 3 | 3T | 1-0 | José Caballero | batter | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 4 | 6T | 0-2 | Luis García Jr. | batter | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 5 | 6B | 2-0 | Alejandro Kirk | batter | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 6 | 8T | 0-0 | Alejandro Kirk | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 7 | 8B | 2-0 | Daz Cameron | batter | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 8 | 8B | 2-1 | (unknown) | unknown | unknown | OVERTURNED | - → - | - → - | — |
| 9 | 9B | 2-1 | Ali Sánchez | catcher | away | stands | 1 → 0 | 1 → 0 | ✅ |
| 10 | 10B | 0-0 | Brett Bateman | batter | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 11 | 10B | 2-0 | Ali Sánchez | catcher | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |

> ℹ️ 1 of 11 challenges here carry no pitch number, at-bat index or challenger, so they cannot be matched to a pitch by any method. They are shown as 'no key' rather than as failures.

- Extra-inning grants: home +0, away +1
- Final inventory: home **2**, away **1**

> ✅ The independent walk agrees with the built table on every challenge in this game.

**Hand-check (open the Savant game feed):**

- [ ] Savant shows 11 challenges in this game
- [ ] Every challenger name and role matches
- [ ] Every result (overturned / stands) matches
- [ ] No side challenges again after losing two
- [ ] Extra-inning grants, if any, look right

## Game 823319 — COL @ SD, 2026-04-09

- HP umpire: **Nic Lentz** · 12 innings · 363 pitches · **10 challenges**
- [Savant game feed](https://baseballsavant.mlb.com/gamefeed?gamePk=823319) · [MLB Gameday](https://www.mlb.com/gameday/823319) · [raw play-by-play](https://statsapi.mlb.com/api/v1/game/823319/playByPlay)

| # | Inn | Count | Challenger | Role | Side | Result | Expected before→after | Table before→after | |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 2T | 0-0 | TJ Rumfield | batter | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 2 | 2B | 0-0 | Hunter Goodman | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 3 | 3T | 0-1 | Freddy Fermin | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 4 | 4B | 0-0 | Hunter Goodman | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 5 | 6T | 2-1 | Troy Johnston | batter | away | stands | 2 → 1 | 2 → 1 | ✅ |
| 6 | 7T | 1-0 | Willi Castro | batter | away | stands | 1 → 0 | 1 → 0 | ✅ |
| 7 | 8B | 0-1 | (unknown) | unknown | unknown | stands | - → - | - → - | — |
| 8 | 11T | 1-0 | Luis Campusano | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 9 | 11T | 2-1 | Luis Campusano | catcher | home | stands | 2 → 1 | 2 → 1 | ✅ |
| 10 | 11B | 0-0 | Brett Sullivan | catcher | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |

> ℹ️ 1 of 10 challenges here carry no pitch number, at-bat index or challenger, so they cannot be matched to a pitch by any method. They are shown as 'no key' rather than as failures.

- Extra-inning grants: home +0, away +1
- Final inventory: home **1**, away **1**

> ✅ The independent walk agrees with the built table on every challenge in this game.

**Hand-check (open the Savant game feed):**

- [ ] Savant shows 10 challenges in this game
- [ ] Every challenger name and role matches
- [ ] Every result (overturned / stands) matches
- [ ] No side challenges again after losing two
- [ ] Extra-inning grants, if any, look right

---

## Sign-off

- [ ] I opened at least three of these games in the Savant game feed and confirmed every challenge, challenger and result by eye.
- [ ] Any mismatch above was investigated and resolved.

Checked by: ____________________   Date: ____________
