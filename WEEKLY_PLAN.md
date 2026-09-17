# Fall 2026 Capstone — Week-by-Week Action Plan

**Project:** Optimal Challenge Policies, Decision Regret, and Behavioral Adaptation under MLB's ABS System
**Author:** Melissa Martinez
**Covers:** Week 3 (Sept 14–18) through Week 15 (Dec 7–11)

---

## How to read this

Each week has four parts:

- **The point** — one sentence on what this week is actually for.
- **Do this** — numbered action items. Each one is a single sitting's work.
- **Done when** — the checklist you show your supervisor. If you can't tick every box, the week isn't finished.
- **Watch out for** — the thing that most commonly eats a week here.

Rough time estimates assume 10–12 hours/week. Weeks 5, 8, and 11 are the heavy ones — protect time for those.

**One rule that will save you in December:** every week, write a dated entry in `NOTES.md` saying what you decided and why. Weeks 14–15 are "write up what you did." If you keep that file, they take two days. If you don't, they take two weeks.

---

## Week 3 — Sept 14–18 · Data-Feasibility Spike

> **Deliverable:** Data feasibility report, source inventory, risk decision
> **You are here.** This week is the gate: it decides whether the project as proposed is buildable.

### The point

Before writing a real pipeline, prove that the five fields your whole project depends on actually exist and can be joined. You are not analyzing anything this week. You are answering: *can I get the data, and does it link up?*

### Do this

1. **Set up the repo and environment** (1 hr). Make the GitHub repo, create a conda env, `pip install pybaseball pandas pyarrow requests`, and commit a `requirements.txt` with pinned versions. Do this first so everything after it is versioned.

2. **Run `00_env_check.py`** (30 min). It tests every data source you linked in the proposal and prints a pass/fail table. Save the output — that table *is* half your source inventory deliverable.

3. **Pull one single day of Statcast data** (1 hr) with `01_statcast_pitches_2026.py --start 2026-04-15 --end 2026-04-15`. Then open it and print the full column list. **Write down whether any column mentions challenge, review, ABS, or overturn.** This is the single most important fact you will learn this week, because it determines whether challenge data comes free with Statcast or has to be joined in from a second source.

4. **Pull one day of challenge events** (1 hr) with `02_abs_challenges_statsapi.py --start 2026-04-15 --end 2026-04-15`. Count the challenges. Confirm you're getting challenger name, their role (batter/catcher/pitcher), and whether it was overturned.

5. **Prove the join works** (2 hrs). This is the crux. Take the challenges from step 4 and match them to pitches from step 3 on `game_pk` + at-bat + pitch number. **Report the match rate as a percentage.** Anything under ~95% means you have a join problem to solve before Week 5, not after.

6. **Answer the unchallenged-pitch question** (2 hrs). Your model needs *non-challenges* too — every pitch where someone could have challenged and didn't. Using the Savant definition (a called pitch, at least one challenge left, call went against that side), count how many challenge opportunities there are in that one day versus how many were actually challenged. Savant says ~50% of pitches are opportunities and ~5% are "reasonable" ones. Confirm you can reproduce a number in that neighborhood.

7. **Check the three supporting sources** (2 hrs). Run scripts `03`, `04`, and `05`. For each, note: does it return data, at what grain (player-season? game?), and can it be joined to a pitch? Umpire Scorecards in particular is game-level, not pitch-level — decide now whether you need it at all, since you can compute umpire accuracy yourself from Statcast.

8. **Write the feasibility report** (2 hrs). Two to three pages, no more. Use `templates/FEASIBILITY_REPORT.md`.

### Done when

- [ ] Repo exists, environment is reproducible from `requirements.txt`
- [ ] Source inventory table: source → what it gives → grain → how to access → works y/n
- [ ] Statcast column list saved to the repo, ABS-related columns explicitly named (or explicitly confirmed absent)
- [ ] Challenge↔pitch join match rate is a number in the report
- [ ] A defensible count of challenge opportunities for at least one game day
- [ ] Risk register: at least 3 risks, each with a fallback plan
- [ ] **Explicit go / go-with-changes / no-go recommendation**, signed off by your supervisor

### Watch out for

Spending the week making the data *pretty*. This week is throwaway code that answers questions. Ugly is fine. The deliverable is the report, not the notebook.

---

## Week 4 — Sept 21–25 · Ingestion Pipeline

> **Deliverable:** Versioned ingestion pipeline and repository structure

### The point

Turn last week's throwaway scripts into something you can re-run in March and get the same answer. "Reproducible" is one of your three novelty claims — this is the week you earn it.

### Do this

1. **Lock the repo structure** (1 hr):
   ```
   data/raw/          <- never edited by hand, never committed
   data/interim/
   data/processed/
   scripts/           <- the ingestion scripts
   notebooks/
   src/               <- functions you import
   docs/              <- data dictionary, model cards, reports
   ```
   Add a `.gitignore` that excludes `data/`. Raw data does not go in git.

2. **Run the full 2026 Statcast pull** (start it early — it takes hours). `01_statcast_pitches_2026.py` is resumable and writes one file per month, so start it Monday and let it run. Expect roughly 700k–750k pitches for a full season.

3. **Run the full challenge pull** for the whole season (`02`). Same deal — it's resumable and rate-limited.

4. **Run `03`, `04`, `05`, `06`, `07`** for the full season.

5. **Snapshot and freeze** (1 hr). Stamp every raw file with the date you pulled it. Write a `data/raw/MANIFEST.json` recording file name, pull date, row count, and a checksum. **Statcast silently revises past data** — without this you will not be able to explain why a number changed in November.

6. **Write `docs/DATA_SOURCES.md`** (2 hrs) — for each source: URL, what you pull, how often, license/terms, and what breaks if it goes away.

7. **Commit and tag** (30 min): `git tag v0.1-raw-snapshot`.

### Done when

- [ ] One command re-runs the whole ingestion from scratch
- [ ] Full-season raw files for all sources, with a manifest and checksums
- [ ] Row counts recorded and sanity-checked (~700k pitches, ~2 challenges/team/game as a ceiling)
- [ ] `docs/DATA_SOURCES.md` written
- [ ] Tagged commit

### Watch out for

Starting the big pulls on Thursday. Start them Monday. If a pull dies at hour four you need days of slack, not hours.

---

## Week 5 — Sept 28–Oct 2 · Join Everything · **HEAVY WEEK**

> **Deliverable:** Analysis-ready schema, data dictionary, QA report

### The point

Build the one table your entire project reads from. Every later week is "run a model on the Week 5 table." If this table is wrong, everything downstream is wrong and you won't find out until November.

### Do this

1. **Build the pitch spine** (2 hrs). Every 2026 called pitch, one row, keyed on `game_pk` + `at_bat_number` + `pitch_number`. Start from Statcast.

2. **Attach challenge outcomes** (3 hrs). Left-join the Stats API challenge events onto the spine. Every pitch gets `was_challenged` (0/1), and challenged ones additionally get `challenger_role`, `challenger_id`, and `was_overturned`. Verify the unmatched rate is near zero and investigate any misses.

3. **Derive challenges remaining** (3 hrs). Nobody publishes this per-pitch — you compute it. Walk each game in order: both teams start at 2, an unsuccessful challenge subtracts one, a successful one costs nothing, and in extra innings a team that has hit zero gets one more that inning. Every pitch needs `challenges_remaining_batting` and `challenges_remaining_fielding`.

4. **Flag challenge opportunities** (2 hrs). Mark each pitch as an opportunity for the batting side and/or the fielding side using the Savant rules: it was a called pitch (taken, not swung at), that side has a challenge left, the call went against them, and no position player is pitching.

5. **Attach the ABS zone and distance-from-edge** (2 hrs). The ABS zone is 17 inches wide, top at 53.5% of batter height, bottom at 27%, and *any part of the ball* touching it is a strike. So you need batter heights (script `07`) and you need to subtract the ball's radius when computing distance to the edge. **Do not just use Statcast's `sz_top`/`sz_bot`** — that's a different zone definition and using it will quietly bias your overturn model.

6. **Attach umpire, catcher, and game context** (2 hrs). Home plate umpire from the schedule hydrate, catcher from `fielder_2`, framing metrics from `04`, win probability and leverage from `06`.

7. **Write the data dictionary** (2 hrs). Every column: name, type, source, definition, and any known gotcha. `docs/DATA_DICTIONARY.md`.

8. **QA report** (2 hrs). Row counts at each join step, null rates per column, at least 5 assertions that must hold (e.g. challenges remaining is always 0–2 in regulation; overturn rate is in a plausible range; no pitch is an opportunity for a side with zero challenges left).

### Done when

- [ ] One analysis-ready parquet file, one row per pitch
- [ ] `challenges_remaining` derived and spot-checked by hand against 3 real box scores
- [ ] Challenge opportunity flags, with counts that roughly match Savant's ~50% / ~5% figures
- [ ] ABS-zone distance computed from batter height, not Statcast's zone
- [ ] Data dictionary covers every column
- [ ] QA assertions run as a script and all pass

### Watch out for

Trusting the challenges-remaining logic because it runs without erroring. Pick three games, open the play-by-play on MLB.com, and check by hand. This is the field most likely to be silently wrong, and it's the state variable your entire MDP is built on.

---

## Week 6 — Oct 5–9 · Define the Cohort

> **Deliverable:** EDA notebook and finalized analytic cohort

### The point

Decide exactly which rows the model trains on, and freeze that decision. Also: actually look at your data before you model it.

### Do this

1. **Write the inclusion rules** (2 hrs) as code, not prose. Regular season only? Drop position-player-pitching? Drop games with ABS technical failures? Each rule gets a line of code and a sentence of justification.

2. **Set the train/test split** (1 hr). **Split on time, not at random** — your proposal already commits to this, and it's correct, because you're predicting the future. Something like April–July train, August validate, September test. Write down the exact dates.

3. **Handle missingness** (2 hrs). Tabulate nulls per column. For each, decide: drop the row, impute, or add a missing-indicator. Justify in writing.

4. **EDA — the descriptives** (3 hrs): overturn rate overall and by challenger role; challenge rate by inning, by count, by challenges remaining; distance-from-zone-edge distribution for challenged vs. unchallenged pitches.

5. **EDA — the sanity checks** (2 hrs): does overturn rate fall as distance from the zone edge grows? Are catchers better challengers than batters (MLB has said they are)? If your data disagrees with published findings, your data is probably wrong.

6. **Freeze the cohort** (1 hr). Save it as its own file. Tag the commit. From here on, "the cohort" means this file.

### Done when

- [ ] Inclusion/exclusion rules are code, and the row count after each is logged
- [ ] Train/validate/test dates fixed in a config file
- [ ] Missingness decisions documented
- [ ] EDA notebook with at least 8 figures
- [ ] Your descriptives reproduce published MLB/Savant numbers within a reasonable margin
- [ ] Cohort file frozen and tagged

### Watch out for

Letting the cohort keep drifting. Once you freeze it, changing it means re-running every downstream week. Freeze it deliberately and note the date.

---

## Week 7 — Oct 12–16 · Baseline Overturn Model

> **Deliverable:** Baseline model report and calibration plots

### The point

Build the simplest model that could work, so you have something to beat and something interpretable to explain. **You are predicting: given that a challenge happened, would the call be overturned?**

### Do this

1. **Build the feature matrix** (2 hrs): distance from ABS zone edge (signed), plate_x/plate_z, batter height, count, pitch type, velocity, movement, handedness, catcher, umpire, inning, leverage.

2. **Fit the logistic regression** (2 hrs). Start with distance-from-edge alone — that one feature will do most of the work. Then add the rest. Report how much each block adds.

3. **Fit the GAM** (3 hrs) with smooth terms on the continuous variables, especially distance from edge, which is very non-linear near zero. `pygam` in Python, or `mgcv` in R via `rpy2` if you want the better implementation.

4. **Evaluate — and lead with calibration** (2 hrs). Brier score, log loss, calibration intercept and slope, reliability diagram, expected calibration error, then ROC-AUC. **Say this explicitly in the report:** you care about calibration over discrimination because the output feeds an MDP, and a miscalibrated probability produces a wrong policy even when it ranks cases correctly.

5. **Write the baseline report** (2 hrs) with a table of metrics and a reliability diagram per model.

### Done when

- [ ] Logistic and GAM both fit and evaluated on the temporal split
- [ ] Full calibration metric suite reported for both
- [ ] Reliability diagrams saved as figures
- [ ] Coefficients/partial-dependence interpreted in baseball language, not just statistics
- [ ] A clear statement of which is the baseline to beat

### Watch out for

Reporting only AUC. Your proposal explicitly commits to calibration metrics; a high-AUC, badly-calibrated model will wreck the MDP in Week 11.

---

## Week 8 — Oct 19–23 · Gradient Boosting · **HEAVY WEEK**

> **Deliverable:** Comparative predictive-model results

### The point

Beat the baseline with CatBoost and LightGBM — or show that you can't, which is also a legitimate result.

### Do this

1. **CatBoost** (3 hrs). Pass umpire, catcher, pitcher, batter, and pitch type as native categorical features — that's the reason you picked CatBoost, so use it. Do not one-hot encode them.

2. **LightGBM** (2 hrs) as the second boosting benchmark.

3. **Tune honestly** (3 hrs). Tune on the validation window only. The test window gets touched once, at the end. Log every configuration you try.

4. **Compare all four models** (2 hrs) on identical splits and identical metrics. One table.

5. **Feature importance** (2 hrs). SHAP values on the best model. Check the top features make baseball sense — if umpire identity outranks pitch location, something is leaking.

### Done when

- [ ] Four models trained on identical temporal splits
- [ ] One comparison table, calibration metrics included
- [ ] Hyperparameter search logged and reproducible (fixed seeds)
- [ ] SHAP plots for the best model
- [ ] Test set used exactly once

### Watch out for

Leakage. Any feature that encodes the outcome — including anything derived from the challenge result, or Savant's own expected-overturn metric — will give you a suspiciously good model. If AUC comes back above ~0.95, hunt for the leak before celebrating.

---

## Week 9 — Oct 26–30 · Calibrate and Lock

> **Deliverable:** Locked probability model and model card

### The point

Turn the winning model into a trustworthy probability, then freeze it. After this week, `p(s)` is a fixed input to everything else.

### Do this

1. **Calibrate** (3 hrs). Hold out a dedicated calibration set — separate from train and test — and fit isotonic regression and Platt scaling. Compare calibrated vs. uncalibrated on Brier, log loss, and ECE. Pick one and say why.

2. **Subgroup diagnostics** (3 hrs). Check calibration separately by challenger role, by count, by challenges remaining, by month, by handedness. A model that's well-calibrated overall but badly calibrated for batters specifically will distort the policy for batters.

3. **Choose the production model** (1 hr). Write the decision down with the reasoning.

4. **Write the model card** (3 hrs): intended use, training data and dates, features, performance overall and by subgroup, calibration method, known limitations, what it should *not* be used for.

5. **Serialize and version it** (1 hr). Save the fitted object, pin the library versions, and write a `predict()` function with a fixed input schema that Weeks 10–12 can import.

### Done when

- [ ] Calibration applied using a dedicated calibration set
- [ ] Subgroup calibration tables for at least 5 subgroups
- [ ] Production model chosen, justified, serialized, versioned
- [ ] Model card complete
- [ ] A stable `predict(state) -> probability` interface exists

### Watch out for

Calibrating on the test set. That invalidates your test results. Three-way split: train / calibrate / test.

---

## Week 10 — Nov 2–6 · Win Probability Component

> **Deliverable:** Validated win-probability component

### The point

Build `V(s)` — the probability of eventually winning from a given game state. This is the other half of your Q-function, and it's what turns "the call was wrong" into "the call was worth something."

### Do this

1. **Define the state** (2 hrs). Inning, half, score differential, outs, base state, count. That's roughly 24 base-out-count combinations × innings × score bins — write out the exact dimensions.

2. **Build the empirical table first** (3 hrs). For every state, what fraction of the time did that team go on to win? Use several seasons if you need the sample size — this is historical win expectancy and it doesn't depend on ABS. Note which cells are thin.

3. **Fit the model version** (3 hrs). Gradient boosting on state → win. Compare against the empirical table; they should broadly agree, and disagreement points at sparse cells.

4. **Validate** (2 hrs). Calibration again — same metrics as Week 9. Then sanity-check against known values: a tie game in the bottom of the 9th with a runner on third and one out should come out around 80%+ for the home team. If it doesn't, something's wrong.

5. **Build the transition tables** (2 hrs). This is what Week 11 needs: given the state and a called pitch, where does the state go if the call stands vs. if it's overturned? Ball-to-strike and strike-to-ball transitions, including walks and strikeouts. Get these from the empirical data, not from assumptions.

### Done when

- [ ] Empirical win-expectancy table with cell counts
- [ ] Model-based version, calibrated and validated
- [ ] Both agree on well-populated cells
- [ ] Transition tables for overturned vs. stands
- [ ] Sanity checks against published win-expectancy values pass
- [ ] Clean `V(state) -> probability` interface

### Watch out for

Thin cells. Extreme states (down 12 in the 2nd) have almost no data and will produce noisy values. Smooth them or flag them — and note it as a limitation.

---

## Week 11 — Nov 9–13 · The MDP · **HEAVY WEEK**

> **Deliverable:** MDP specification, unit tests, toy-policy results

### The point

Write down the decision problem formally and solve a small version of it. This is the intellectual core of the capstone; Spring is mostly scaling it up.

### Do this

1. **Write the formal spec first, on paper** (3 hrs). States, actions, transitions, rewards, horizon, terminal conditions. `docs/MDP_SPEC.md`. Do this before writing code — half the bugs in Spring will come from an ambiguity you didn't resolve here.

2. **Implement the Q-functions** (2 hrs) exactly as in your proposal:
   ```
   Q_challenge(s, r) = p(s)·V(s_overturned, r) + (1 − p(s))·V(s_stand, r − 1)
   Q_hold(s, r)      = V(s_stand, r)
   ```
   Challenge when `Q_challenge > Q_hold`. Note the asymmetry that makes this interesting: a successful challenge keeps the challenge, so only the failure branch decrements `r`.

3. **Build the backward-induction solver** (4 hrs). Finite horizon, decision epochs are challenge opportunities. Work backward from the end of the game.

4. **Write unit tests** (2 hrs). With `p(s) = 1` it should always challenge. With `p(s) = 0` it should never challenge. With `r = 0` challenging must be unavailable. A tied 9th-inning high-leverage opportunity should have a lower challenge threshold than a 1st-inning blowout.

5. **Solve a toy version** (2 hrs). One inning, coarse state space. Produce a threshold plot: the minimum `p(s)` that justifies challenging, as a function of leverage and challenges remaining. **That plot is the money figure of your fall semester** — put it in the interim memo and the poster.

### Done when

- [ ] Written MDP spec with every component defined
- [ ] Q-functions implemented and matching the proposal's formulas
- [ ] Backward-induction solver runs on a reduced state space
- [ ] All unit tests pass, including the degenerate cases
- [ ] Threshold plot produced and baseball-sane
- [ ] A clear written statement of what Spring needs to add

### Watch out for

Trying to solve the full state space. It's a toy policy this week — that's the deliverable. Scaling is Spring, Week 2.

---

## Week 12 — Nov 16–20 · Dashboard v0

> **Deliverable:** Dashboard v0 and fall interim research memo

### The point

Build the League Trends section only. The Optimal Challenge Explorer and Simulator are Spring. This week is about getting fluent with Dash, not about shipping the final product.

### Do this

1. **Scaffold the Dash app** (2 hrs). One file, a few callbacks, a filter bar (date range, team, challenger role).

2. **Build the League Trends tab** (4 hrs): challenge usage by week; challenge rate by inning; by count; success rate over the season; challenges preserved vs. exhausted; actual vs. expected challenge usage.

3. **Add a Model Results tab** (3 hrs): the reliability diagram, a calibration table, challenge success by context, and the Week 11 threshold plot.

4. **Write the interim memo** (4 hrs) — 5–8 pages: what you did, what you found, what's validated, what's still open, and the Spring plan. This is what your supervisor reads.

### Done when

- [ ] Dash app runs locally with no errors
- [ ] League Trends tab has at least 6 working charts
- [ ] Model Results tab shows calibration and the threshold plot
- [ ] Filters work and don't break on empty selections
- [ ] Interim memo delivered to supervisor
- [ ] `README.md` explains how to run the app

### Watch out for

Making it beautiful. It's v0 and its stated purpose is familiarity with Dash. Working and plain beats pretty and half-finished.

---

## Week 13 — Nov 23–27 · Thanksgiving

No deliverable. Rest.

If you want one low-effort win: re-run the ingestion pipeline end to end and confirm it still produces identical row counts. It catches silent breakage and costs one command.

---

## Week 14 — Nov 30–Dec 4 · Consolidate

> **Deliverable:** Fall research package and supervisor review

### The point

Assemble everything into one reviewable package. No new analysis.

### Do this

1. **Clean the repo** (3 hrs). Delete dead code, make sure every script runs top to bottom, write the README, pin dependencies.

2. **Assemble the research package** (4 hrs): abstract, data and methods, results (baseline, comparative, calibration, win probability, MDP), limitations, Spring plan. The pieces should mostly be lifted from the weekly deliverables you already wrote.

3. **Be explicit about limitations** (2 hrs). One MLB season. Derived rather than published challenges-remaining. Approximated ABS zone. Thin cells in win expectancy. Naming these yourself is a strength — a reviewer finding them for you is not.

4. **Write the Spring implementation plan** (2 hrs), mapped to the Spring timeline already in your outline.

5. **Send it to your supervisor early in the week**, so their comments land inside Week 15's buffer rather than after it.

### Done when

- [ ] Repo clean, documented, reproducible from scratch
- [ ] Research package assembled
- [ ] Limitations section written
- [ ] Spring plan written and mapped to the Spring timeline
- [ ] Supervisor has it with time to respond

### Watch out for

Starting new analysis. Anything that isn't done by Week 14 is a Spring item. Write it down as one and move on.

---

## Week 15 — Dec 7–11 · Buffer

> **Deliverable:** Revised fall package *only if required*

Explicitly no new feature work. Use it for supervisor comments, a data or model issue you deferred, or code cleanup and documentation.

If nothing needs fixing, take the week. You have a full Spring semester ahead of you, and the Spring timeline has no slack until March.

---

## The three things most likely to go wrong this semester

1. **The challenges-remaining field is wrong.** It's derived, not published, and it's the state variable the whole MDP rests on. Hand-verify it in Week 5 against real box scores.

2. **Your model is leaking.** Any feature computed after the challenge outcome — including Savant's own expected-overturn metrics — will inflate performance. Suspiciously good results in Week 8 mean hunt for the leak.

3. **You run out of November.** Weeks 10 and 11 are the hardest and land back to back right before Thanksgiving. If you're going to fall behind, do it in Week 6 or 12, not Week 10.
