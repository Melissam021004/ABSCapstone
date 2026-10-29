# Pipeline guide

Setup, run order, and data notes for *Optimal Challenge Policies, Decision Regret,
and Behavioral Adaptation under MLB's Automated Ball-Strike (ABS) System*. For the
project overview, goals and timeline, see the [main README](../README.md).

Run every command from the repository root; paths below are relative to it. See
`WEEKLY_PLAN.md` for the week-by-week plan, and `docs/DATA_SOURCES.md` for what
each source gives you.

---

## Setup

```bash
conda create -n abs python=3.11 -y && conda activate abs
pip install pybaseball pandas numpy pyarrow requests
pip install scipy scikit-learn matplotlib catboost lightgbm  # Weeks 7–10
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

Week 6 — define and freeze the analytic cohort (see `WEEK6.md`):

```bash
python scripts/13_define_cohort.py --dry-run   # inspect first
python scripts/13_define_cohort.py             # freeze + tag
```

Week 7 — the baseline overturn models, graded on calibration first
(see `WEEK7.md`, which explains every number):

```bash
python scripts/run_week7.py --demo     # whole week on a synthetic season, ~2 min
python scripts/run_week7.py            # 3 test files, then 15_week7_baseline.py
                                       # -> docs/WEEK7_BASELINE_REPORT.md + figures/week7/
```

Weeks 7–9 share one feature builder, one metric suite and one set of models in
`src/` (`abs_features.py`, `abs_metrics.py`, `abs_models.py`), so every model in
the project is built and graded by literally the same code. `figures_demo_week7/`
shows the Week 7 outputs on synthetic data.

The dashboard — a static site you can host free on GitHub Pages
(see `dashboard/README.md`):

```bash
python scripts/12_export_dashboard_data.py     # Week 5 table -> dashboard/data.js
open dashboard/index.html
python scripts/14_build_single_file.py         # one-file copy for email/archive
```

Week 5 — join everything, document it, prove it's right (see `scripts/WEEK5.md`):

```bash
python scripts/run_week5.py           # 08 -> 09 -> 10 -> 11, then the checklist
python scripts/run_week5.py --demo    # same chain on a synthetic season, ~3 sec
```

| Script | Produces |
|---|---|
| `09_data_dictionary.py` | `docs/DATA_DICTIONARY.md` — every column, with leakage-risk flags |
| `10_qa_assertions.py` | `docs/QA_REPORT.md` — 24 assertions, exit code 1 on failure |
| `11_verify_challenges_remaining.py` | `docs/VERIFICATION_CHALLENGES_REMAINING.md` |
| `make_demo_pitches.py` | a synthetic season that obeys the real ABS rules |

Presentation figures from the 03-06 pulls (see `scripts/eda/README.md`):

```bash
python scripts/eda/make_demo_data.py            # preview with fake data first
python scripts/eda/eda_00_overview.py           # + eda_03/04/05/06_*.py
```

`figures_demo/` holds a rendered set built from synthetic data, so you can see
the slide figures before your real pulls finish. Each carries a DEMO DATA badge.

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

```text
Statcast  <-> Stats API :  game_pk + (at_bat_number - 1 == at_bat_index) + pitch_number
                           ^^^ Statcast is 1-BASED, the Stats API is 0-BASED

Win probability         :  game_pk + at_bat_index        (plate-appearance grain)
Catcher framing         :  Statcast fielder_2 == player_id
Umpires                 :  game_pk -> hp_umpire_id       (from schedule hydrate=officials)
Player bios             :  batter / pitcher / fielder_2 == player_id
```

---

## The ABS zone (2026)

```text
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

```text
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

**Savant ABS leaderboard and `csv=true`.** Appending `&csv=true` works on most
Savant leaderboards but returns **HTTP 500** on `/leaderboard/abs-challenges`.
Script 03 therefore reads the rows out of the JSON embedded in the page's
`<script>` tags instead, and falls back to `csv=true` and a couple of candidate
API routes. Run it with `--debug` to save the raw HTML if it can't find the JSON.
Last resort: the page's own Download CSV button, saved into `data/raw/savant/` by
hand — record the date in your manifest, because a hand-pulled file still needs
provenance.

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

3. **For this target, judge leakage by what features add *beyond location*.**
   Pitch location alone legitimately pushes AUC into the 0.9s, because ABS rules
   on nothing else. A non-location feature that adds much on top of that is the
   red flag: check for anything computed after the challenge resolved.
