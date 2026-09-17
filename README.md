# ABS Capstone — Data Pipeline

Ingestion scripts for *Optimal Challenge Policies, Decision Regret, and Behavioral
Adaptation under MLB's Automated Ball-Strike (ABS) System*.

See `WEEKLY_PLAN.md` for the Week 3–15 breakdown, and `docs/DATA_SOURCES.md` for
what each source gives you.

---

## Setup

```bash
conda create -n abs python=3.11 -y && conda activate abs
pip install pybaseball pandas numpy pyarrow requests
pip install scikit-learn statsmodels catboost lightgbm pygam   # Weeks 7–10
pip install dash plotly                                        # Week 12
```

Scripts write to `./data` by default. Override with `export ABS_DATA_DIR=/path/to/data`.

---

## Run order

```bash
python scripts/00_env_check.py                  # Week 3 — run this first
python scripts/01_statcast_pitches_2026.py      # ~1–3 hrs, resumable
python scripts/02_abs_challenges_statsapi.py    # ~1–2 hrs, resumable
python scripts/07_player_bios.py                # needs 01; gives batter heights
python scripts/03_savant_abs_leaderboards.py    # validation aggregates
python scripts/04_catcher_framing.py
python scripts/05_umpire_data.py                # needs 01, 02, 07
python scripts/06_winprob_leverage.py           # needs 02, ~1 hr, resumable
python scripts/08_build_challenge_inventory.py  # the analysis-ready table
python scripts/test_logic.py                    # no network needed
```

Scripts 01, 02, and 06 cache every chunk. If one dies, run it again — it picks up
where it left off. Start them in the morning.

`07` is not optional. The ABS zone is defined off batter height, and without it
`08` silently falls back to Statcast's zone, which is a different zone.

---

## What each script produces

| Script | Output | Grain |
|---|---|---|
| `00_env_check` | feasibility report JSON + printed table | — |
| `01_statcast_pitches_2026` | `raw/statcast/statcast_2026_<YYYY-MM>.parquet` | pitch |
| `02_abs_challenges_statsapi` | `raw/challenges/abs_challenges_2026.parquet` | challenge event |
| `03_savant_abs_leaderboards` | `raw/savant/abs_leaderboard_*.csv` | player/team-season |
| `04_catcher_framing` | `raw/savant/catcher_framing_*.csv` | catcher-season |
| `05_umpire_data` | `raw/umpires/umpire_accuracy_derived_2026.parquet` | umpire, umpire-game |
| `06_winprob_leverage` | `raw/winprob/winprob_2026.parquet` | plate appearance |
| `07_player_bios` | `raw/players/player_bios_2026.parquet` | player |
| `08_build_challenge_inventory` | `processed/challenge_inventory_2026.parquet` | **pitch** |

---

## The join keys

Getting these wrong is the most common way this pipeline breaks.

```
Statcast  <-> Stats API :  game_pk + (at_bat_number - 1 == at_bat_index) + pitch_number
                           ^^^ Statcast is 1-BASED, the Stats API is 0-BASED

Win probability         :  game_pk + at_bat_index        (plate-appearance grain)
Catcher framing         :  Statcast fielder_2 == player_id
Umpires                 :  game_pk -> hp_umpire_id       (from schedule hydrate=officials)
Player bios             :  batter / pitcher / fielder_2 == player_id
```

---

## The ABS zone (2026)

```
width   17 inches, centred on the plate
top     53.5% of the batter's height
bottom  27%   of the batter's height
strike  if ANY part of the ball touches the zone (so subtract the 1.45-inch radius)
```

**Do not use Statcast's `sz_top` / `sz_bot` for this.** Those describe a different,
stance-dependent zone. Using them biases every distance feature in a way that
correlates with batter height — and it will not show up in any check except the
one comparing your predicted overturns to the actual ones. Script 08 does that
check for you and prints the agreement rate; below ~85% means something is wrong.

## The challenge rules (2026)

```
2 challenges per team to start
successful challenge  -> retained (costs nothing)
failed challenge      -> consumed
extra innings         -> a side already at 0 gets 1 for that inning
who may challenge     -> batter, catcher, or pitcher only, within ~2 seconds
not available         -> while a position player is pitching
```

Challenges-remaining is **derived, not published**. Script 08 walks each game to
reconstruct it. `test_logic.py` covers the state machine, but the tests use
synthetic games — before Week 6, hand-check three real games against the
play-by-play on MLB.com. It is the state variable the whole MDP rests on.

---

## If a source blocks you

**Savant returns 403.** Raise `--sleep`, lower `--chunk-days`, wait a few minutes,
or try a different network. Script 01 falls back from pybaseball to a direct
request with a browser User-Agent automatically.

**pybaseball errors.** It has known open issues (FanGraphs 403s, maintenance
gaps). Every script here has a direct-HTTP fallback that does not depend on it.
Don't let a pybaseball outage block a week.

**Savant leaderboards won't export CSV.** Use the page's own Download CSV button
and drop the file into `data/raw/savant/` by hand — then record the date in your
manifest, because a hand-pulled file still needs provenance.

**umpscorecards.com has no API.** Expected. Use `05_umpire_data.py --mode derive`,
which computes the same construct from your own Statcast pull at pitch level.

---

## Three things worth knowing before you start

1. **Statcast silently revises past data.** Snapshot your raw pulls with a date
   stamp and checksums in Week 4, or you will not be able to explain why a number
   moved in November.

2. **Don't use Savant's expected-challenge or overturns-vs-expected metrics as
   model features.** They're built from the outcome you're predicting. Use them to
   benchmark against, not to train on.

3. **An AUC above ~0.95 in Week 8 means you have a leak, not a great model.**
   Check for any feature computed after the challenge resolved.
