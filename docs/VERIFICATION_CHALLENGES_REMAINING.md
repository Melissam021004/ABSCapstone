# Verification — `challenges_remaining`

Generated 2026-11-17 by `scripts/11_verify_challenges_remaining.py`.

`challenges_remaining` is derived rather than published, and it is the state variable r in the MDP. This document records the hand-verification the Week 5 plan requires.

Each game below is checked two ways: an independent re-walk of the inventory rules straight from the challenge events, compared against the built table — and then a manual check against the real play-by-play, which is the only one that catches an assumption both implementations share.

## Game 824998 — MIL @ ATH, 2026-06-08

- HP umpire: **Clint Vondrak** · 12 innings · 453 pitches · **16 challenges**
- [Savant game feed](https://baseballsavant.mlb.com/gamefeed?gamePk=824998) · [MLB Gameday](https://www.mlb.com/gameday/824998) · [raw play-by-play](https://statsapi.mlb.com/api/v1/game/824998/playByPlay)

| # | Inn | Count | Challenger | Role | Side | Result | Expected before→after | Table before→after | |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 1T | 0-1 | Jeffrey Springs | pitcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 2 | 3B | 2-1 | William Contreras | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 3 | 3B | 3-2 | William Contreras | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 4 | 6B | 1-2 | William Contreras | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 5 | 7T | 0-1 | Shea Langeliers | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 6 | 7B | 1-2 | Max Muncy | batter | home | stands | 2 → 1 | 2 → 1 | ✅ |
| 7 | 8B | 1-1 | William Contreras | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 8 | 9T | 1-1 | Shea Langeliers | catcher | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 9 | 9T | 2-2 | Shea Langeliers | catcher | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 10 | 9B | 0-0 | William Contreras | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 11 | 9B | 2-2 | William Contreras | catcher | away | stands | 2 → 1 | 2 → 1 | ✅ |
| 12 | 9B | 3-2 | Brent Rooker | batter | home | stands | 1 → 0 | 1 → 0 | ✅ |
| 13 | 10B | 0-0 | Jonah Heim | batter | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 14 | 11T | 2-2 | Luis Rengifo | batter | away | stands | 1 → 0 | 1 → 0 | ✅ |
| 15 | 11T | 1-2 | Shea Langeliers | catcher | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 16 | 12B | 0-1 | Jeff McNeil | batter | home | stands | 1 → 0 | 1 → 0 | ✅ |

- Extra-inning grants: home +1, away +1
- Final inventory: home **0**, away **1**

> ✅ The independent walk agrees with the built table on every challenge in this game.

**Hand-check (open the Savant game feed):**

- [ ] Savant shows 16 challenges in this game
- [ ] Every challenger name and role matches
- [ ] Every result (overturned / stands) matches
- [ ] No side challenges again after losing two
- [ ] Extra-inning grants, if any, look right

## Game 823178 — AZ @ SF, 2026-08-28

- HP umpire: **Alan Porter** · 9 innings · 344 pitches · **13 challenges**
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
| 12 | 9T | 1-2 | Tim Tawa | batter | away | stands | 1 → 0 | 1 → 0 | ✅ |
| 13 | 9B | 2-1 | Osleivis Basabe | batter | home | stands | 1 → 0 | 1 → 0 | ✅ |

- Final inventory: home **0**, away **0**

> ✅ The independent walk agrees with the built table on every challenge in this game.

**Hand-check (open the Savant game feed):**

- [ ] Savant shows 13 challenges in this game
- [ ] Every challenger name and role matches
- [ ] Every result (overturned / stands) matches
- [ ] No side challenges again after losing two
- [ ] Extra-inning grants, if any, look right

## Game 824599 — CHC @ CWS, 2026-05-17

- HP umpire: **Chris Segal** · 10 innings · 376 pitches · **12 challenges**
- [Savant game feed](https://baseballsavant.mlb.com/gamefeed?gamePk=824599) · [MLB Gameday](https://www.mlb.com/gameday/824599) · [raw play-by-play](https://statsapi.mlb.com/api/v1/game/824599/playByPlay)

| # | Inn | Count | Challenger | Role | Side | Result | Expected before→after | Table before→after | |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 1T | 0-0 | Alex Bregman | batter | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 2 | 3T | 3-2 | Edgar Quero | catcher | home | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 3 | 3T | 0-0 | Edgar Quero | catcher | home | stands | 2 → 1 | 2 → 1 | ✅ |
| 4 | 3B | 2-2 | Carson Kelly | catcher | away | OVERTURNED | 2 → 2 | 2 → 2 | ✅ |
| 5 | 4T | 0-0 | Edgar Quero | catcher | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 6 | 5T | 0-1 | Edgar Quero | catcher | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 7 | 7T | 0-0 | Edgar Quero | catcher | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 8 | 8T | 0-0 | Edgar Quero | catcher | home | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 9 | 8B | 1-0 | Miguel Amaya | catcher | away | stands | 2 → 1 | 2 → 1 | ✅ |
| 10 | 9T | 1-2 | Edgar Quero | catcher | home | stands | 1 → 0 | 1 → 0 | ✅ |
| 11 | 9B | 2-1 | Miguel Amaya | catcher | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |
| 12 | 10T | 1-2 | Seiya Suzuki | batter | away | OVERTURNED | 1 → 1 | 1 → 1 | ✅ |

- Extra-inning grants: home +1, away +0
- Final inventory: home **1**, away **1**

> ✅ The independent walk agrees with the built table on every challenge in this game.

**Hand-check (open the Savant game feed):**

- [ ] Savant shows 12 challenges in this game
- [ ] Every challenger name and role matches
- [ ] Every result (overturned / stands) matches
- [ ] No side challenges again after losing two
- [ ] Extra-inning grants, if any, look right

---

## Sign-off

- [ ] I opened at least three of these games in the Savant game feed and confirmed every challenge, challenger and result by eye.
- [ ] Any mismatch above was investigated and resolved.

Checked by: ____________________   Date: ____________
