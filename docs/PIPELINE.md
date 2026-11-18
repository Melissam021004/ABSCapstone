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
pip install scipy scikit-learn matplotlib                     # Weeks 7–10
conda install -c conda-forge catboost lightgbm               # Week 8 (more reliable than pip on a Mac)
pip install dash plotly                                        # Week 12
```

Scripts write to `./data` by default. Override with `export ABS_DATA_DIR=/path/to/data`.

---

## Run order

```bash
python scripts/00_env_check.py                  # Week 3 — run this first
python scripts/01_statcast_pitches_2026.py      # ~1–3 hrs, resumable
python scripts/02_abs_challenges_statsapi.py    # ~1–2 hrs, resumable (--force re-extracts)
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

Weeks 8–9 — gradient boosting, then calibrate and lock p(s) (see `WEEK8.md` and
`WEEK9.md`; they're due together, so one runner does both):

```bash
python tests/test_boosting_backends.py          # 30 s: do CatBoost and LightGBM work here?
python scripts/run_week89.py --demo             # everything on a synthetic season, ~5 min
python scripts/run_week89.py                    # tests -> Week 7 refresh -> 16 -> 17 -> dashboard
python scripts/17_week9_calibrate_lock.py --final-test    # ONCE: the September test month
```

| Script | Produces |
|---|---|
| `16_week8_boosting.py` | `docs/WEEK8_REPORT.md`, `docs/week8_*.csv`, `figures/week8/`, `models/week8/` (+ `selection.json`) |
| `17_week9_calibrate_lock.py` | `docs/WEEK9_REPORT.md`, `docs/MODEL_CARD.md`, `docs/week9_*.csv`, `figures/week9/`, `models/production/` |
| `17_… --final-test` | `docs/WEEK9_TEST_RESULTS.md`, `docs/week9_test_*.csv`, `models/production/TEST_OPENED.json` |
| `18_export_model_dashboard.py` | `dashboard/model_data.js` (the Overturn Model and Progress tabs) |

Weeks 7–9 share one feature builder, one metric suite and one set of models in
`src/` (`abs_features.py`, `abs_metrics.py`, `abs_models.py`, `abs_baselines.py`),
so every model in the project is built and graded by literally the same code.
Week 8 adds `abs_boosting.py` and `abs_pipeline.py`; Week 9 adds
`abs_calibration.py` and `abs_predict.py` — **the only module Weeks 10–12 should
import for p(s)**: `from src.abs_predict import load_production`.
`figures_demo_week7/`, `figures_demo_week8/` and `figures_demo_week9/` show the
outputs on synthetic data. They (and the "what the demo shows" parts of
WEEK7–9.md) were made with the first, simpler demo season. Week 10 rebuilt the
demo season as real baseball (runners, scores, walk-offs), so re-running
`run_week89.py --demo` today gives different — equally synthetic — numbers (for
example, the demo baseline becomes M2 · location instead of M1). Your real-data
results don't change.

Week 10 — win probability V(s), and what every challenge was worth (see `WEEK10.md`):

```bash
python scripts/run_week10.py --demo             # everything on a simulated season, ~6 min
python scripts/run_week10.py                    # tests -> 19 -> 20 -> dashboard
python scripts/01_statcast_pitches_2026.py --season 2025   # optional: another season (hours)
python scripts/19_week10_winprob.py --seasons 2025,2026    # ...used for the win table
```

| Script | Produces |
|---|---|
| `19_week10_winprob.py` | `docs/WEEK10_REPORT.md`, `docs/week10_*.csv` (empirical table, run expectancy, call transitions, validation, sanity), `figures/week10/w10_01–06`, `models/winprob/` |
| `20_week10_challenge_value.py` | `docs/WEEK10_VALUE_REPORT.md`, `docs/week10_value_*.csv`, `figures/week10/w10_07–12` |

Week 10 adds `abs_gamestate.py` (pitches -> game states), `abs_winprob.py` (the
structural model, the call transitions and `load_winprob()` — **the only module
Weeks 11–12 should import for V(s)**) and `abs_value.py` (valuing challenges,
natural experiments, skill tests). It reads every pitch from the raw Statcast
pulls — it needs the score after each pitch (`post_home_score`, `post_away_score`),
the runners and the plate-appearance result (`events`).

Week 11 — the challenge decision as a Markov decision process (see `WEEK11.md` and
`docs/MDP_SPEC.md`):

```bash
python scripts/run_week11.py --demo             # everything on a simulated season, ~1.5 min
python scripts/run_week11.py                    # tests -> 21 -> dashboard
python tests/test_mdp.py                        # the MDP's known answers on a toy game (~10 s)
```

| Script | Produces |
|---|---|
| `21_week11_mdp.py` | `docs/WEEK11_REPORT.md`, `docs/week11_*.csv` (pitch model, leverage checks, value of holding, thresholds, run levels, policies, 2026 decisions, umpire tiers), `docs/week11_summary.json`, `figures/week11/w11_01–06`, `models/mdp/` |

Week 11 adds `abs_mdp.py` (the pitch-level game, the backward-induction solver,
policies and a simulator; `load_mdp()` re-solves the saved model in a few seconds)
and `abs_leverage.py` (Tango's Leverage Index from V(s)). It reads the same inputs
as Week 10 — the locked p(s), the locked V(s), the cohort and every raw pitch — and
changes none of them. `models/mdp/solution_*.npz` (~20 MB) stays out of git; the
small pitch model and manifests are committed.

The checkpoint showcase and two data checks (November 2026):

```bash
python scripts/24_challenge_count_check.py               # does the data have every 2026 challenge? (seconds, offline)
python scripts/22_showcase_examples.py                   # real-game examples + slides for the checkpoint (a few minutes)
python scripts/22_showcase_examples.py --video-links     # also look up each pitch's Savant video (internet)
python scripts/25_first_challenge_value.py               # what "the first challenge is worth ~6 points" means (about a minute)
python scripts/25_first_challenge_value.py --game 776543 --team NYY   # show a game you choose
python scripts/23_umpire_roster_check.py                 # why there are more plate umpires than MLB's 76
python scripts/23_umpire_roster_check.py --staff docs/mlb_umpire_staff_2026.txt   # exact split from a list of names
python tests/test_challenge_extract.py                   # script 02 finds a review wherever the feed records it
```

| Script | Produces |
|---|---|
| `22_showcase_examples.py` | `docs/SHOWCASE.md` (presenter notes: a glossary, a running order, every example in plain words with its replay link, the quiz answers), `docs/showcase_examples.csv`, `figures/showcase/` (about 55 slides, each with its data CSV) |
| `23_umpire_roster_check.py` | `docs/UMPIRE_CHECK.md`, `docs/umpire_roster_check.csv` |
| `24_challenge_count_check.py` | `docs/CHALLENGE_COUNT_CHECK.md`, `docs/challenge_count_check.csv` |
| `25_first_challenge_value.py` | `figures/showcase/12_first_challenge_value`, `13_first_challenge_one_game`, `14_first_challenge_all_games` (each with its data CSV), `docs/FIRST_CHALLENGE_VALUE.md` (the explainer, with the example game call by call), `docs/first_challenge_team_games.csv` |

Scripts 22 and 25 need Weeks 9–11 (`models/production`, `models/winprob`, `models/mdp`) and
read the cohort, every raw pitch and the player bios (for names). They change nothing upstream.

**What the first challenge is worth (script 25).** Week 11 §3's value of the first
challenge (5.83 points) is what holding one challenge for the whole game is worth,
compared with holding none. It is not the value of the first pitch, whose call is worth
under 1 point. Script 25 checks it a second, independent way: it walks every real game
forward, call by call, with one challenge used at Week 11's thresholds. The challenge is
kept with probability p each time it is used. The average over all team-games has to
match the MDP's number, and on the demo it does: 5.18 vs 5.16, and 5.76 vs 5.77 for two.

**The challenge count.** The first version of script 02 looked for a review only on the
pitch event. The Stats API can also record one on a separate event after the pitch, or on
the plate appearance itself, which is where a review of strike three or ball four can land.
The season came out at 7,897 challenges against MLB's official 10,557. The fixed extractor
looks in all three places and keeps only ABS challenges: a manager's replay review of a tag
play doesn't count. It also saves each game's raw feed, so the next re-extraction needs no
internet. It counts a review only when a batter, catcher or pitcher made it. On the full 2026
season it finds 10,558 challenges: batters 4,766, catchers 5,613, pitchers 179, against MLB's
4,766, 5,609 and 182. The team-games with 5+ challenges came out at 246, the same as MLB's.

To apply it:

```bash
python scripts/02_abs_challenges_statsapi.py --force      # once: downloads every game again (~20–30 min)
python scripts/run_week5.py --spine-from-table           # 08–11 on the pitches already in the table
python scripts/13_define_cohort.py                       # new cohort fingerprint
python scripts/24_challenge_count_check.py               # every stage within ~2% of MLB's count
```

Then re-run Weeks 10–11, and refit p(s) (Weeks 7–9) as a new version.

`--spine-from-table` makes script 08 take its pitches from the existing table instead of the raw
Statcast files. Use it while `data/raw/statcast/` holds less than a full season. It works like this:

- **Always rebuilt:** the challenge columns and everything computed from kept inputs (the ABS zone,
  opportunities).
- **Rebuilt or kept:** batter heights, positions, win probability, framing and umpire accuracy are
  rebuilt when their raw file is there, and kept from the table when it isn't.
- **Backup:** the first such run saves the old table as `challenge_inventory_2026_prev.*`, and later
  runs never overwrite it.
- **Safety stop:** it stops if the table has lost columns its backup still has, and prints the `cp`
  command to restore the backup.

With unchanged challenge records, it reproduces the table exactly (checked with every raw file
present and with all four optional files missing). Script 08 also stops rather than replace a
full-season table with one built from a partial raw folder (`--allow-smaller` overrides that).

Two checks changed in November 2026:

- **Script 08's pitch-number check.** Script 08 checks every challenge against the count the Stats
  API recorded. A challenge that lands on a swing, or on a called pitch at a different count, moves
  to the called pitch at that count in the same at-bat, where the two sources numbered the pitches
  differently. Those rows are marked `challenge_rekeyed`.
- **QA check ZONE_PREDICTS_OUTCOME.** It now compares the zone with the ORIGINAL call, worked out
  from who challenged. 2026 Statcast records the final call on a challenged pitch (Weeks 8–9), so
  against the recorded call every overturned challenge looked like a miss. Agreement collapsed to the
  share upheld (46%).

The dashboard — a static site you can host free on GitHub Pages
(see `dashboard/README.md`):

```bash
python scripts/12_export_dashboard_data.py     # Week 5 table -> dashboard/data.js
python scripts/18_export_model_dashboard.py    # Weeks 7–11 results -> dashboard/model_data.js
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
| `02_abs_challenges_statsapi` | `raw/challenges/abs_challenges_2026.parquet` (+ raw feeds in `raw/challenges/pbp/`) | challenge event |
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
