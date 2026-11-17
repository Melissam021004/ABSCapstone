# Spring 2027 Capstone — Week-by-Week Action Plan

**Project:** Optimal Challenge Policies, Decision Regret, and Behavioral Adaptation under MLB's ABS System
**Author:** Melissa Martinez
**Covers:** Week 1 (Jan 18–22) through Week 15 (Apr 26–30), plus the weeks between the fall checkpoint and January

---

## How to read this

Same four parts as the fall plan:

- **The point** — one sentence on what this week is actually for.
- **Do this** — numbered action items. Each one is a single sitting's work.
- **Done when** — the checklist you show your supervisor. If you can't tick every box, the week isn't finished.
- **Watch out for** — the thing that most commonly eats a week here.

Rough time estimates assume 10–12 hours/week. **Weeks 5, 6 and 7 are the heavy ones, and they come back to back** (Feb 19, Feb 26, Mar 5). Weeks 3 and 4 are lighter than they look because fall Week 11 already built the solver. Use that slack to get Week 5 started.

**The dates are the proposal's.** Every milestone below is one from the Spring timeline in `README.md`, due on the Friday of its week (the poster is Monday, Apr 26). If your term's weeks fall differently, relabel the weeks, but don't move a milestone without telling your supervisor. Week 10 (Mar 22–26) is the only week with no milestone, so it's written as spring break. If your break is a different week, swap the two.

**The NOTES.md rule carries over, and it matters more now.** Weeks 13–15 are "write the report." Every number in it should already have a dated NOTES.md entry and a script that produces it.

**One new rule: the fall locks stay locked.** Every spring result is computed from the locked p(s), V(s) and cohort. After the Nov 17 re-run they are p(s) `2026.10.29-b81183b4`, V(s) `2026.11.17-824ec8be` (the same model as Nov 4, re-locked) and cohort `419af01ad9eee03f`. The cohort fingerprint stayed the same through the challenge fix because it hashes *which pitches* are in the cohort, not their challenge columns, so write in NOTES.md that the challenge records changed on Nov 17. The December p(s) refit gives p(s) a new version. A confirmed data error like the challenge undercount is the only good reason to break a lock. Even then it's never a quiet edit: it's a new version with a new hash, a NOTES.md entry, and a re-run of every number downstream.

**Naming.** Spring scripts continue the numbering (`25_…` onward), each with a runner and a report like the fall weeks. Name the reports after what they contain (`docs/POLICY_V1.md`, `docs/RQ3_REGRET.md`), not `WEEK6.md`: spring week numbers restart at 1 and would overwrite the fall's.

---

## Before spring · Nov 16 – Jan 15

The fall plan still has three weeks left, and two SABR dates fall over the break. One data fix also has to be finished before spring.

**The challenge fix (found Nov 13, confirmed Nov 16).** The first cohort had 7,897 challenges against MLB's official 10,557. The first version of script 02 looked for reviews only on the pitch event. The Stats API also records a review on the plate appearance itself, which is where a review of strike three or ball four lands, and on separate events. In the old cohort, not one challenge whose final call ended the at-bat survived. The fixed extractor finds 10,558: batters 4,766, catchers 5,613 and pitchers 179, against MLB's 4,766, 5,609 and 182. The team-games with 5+ challenges came out at 246, the same as MLB's.

There were two steps:
- **Done Nov 17:** the table and cohort were rebuilt (`run_week5.py --spine-from-table`, then `13_define_cohort.py`), followed by Weeks 10–11 and the showcase. Every stage now matches MLB's count, month by month.
- **December:** refit p(s). It was trained without a quarter of the challenges, so the refit gives a clean record. The Nov 17 run suggests it will change little, because the old p(s) is already calibrated on the full set: expected 102.2 wins vs delivered 102.2, and batters' chosen pitches 49.0% vs 48.9% overturned. First re-run `13_define_cohort.py`. Its regular-season rule now uses Statcast's `game_type` and drops 12 games that aren't regular-season games (1,430 called pitches, with no challenge records). That gives the cohort a new fingerprint for the refit.

| When | What | Notes |
|---|---|---|
| **Nov 16–20** · fall Week 12 | **Checkpoint.** Finish the rebuild (above), then run `python scripts/22_showcase_examples.py --video-links`. Pick the clips from `docs/SHOWCASE.md` and rehearse with the quiz | Still confirm each "not challenged" example on the replay: it's quick, and the broadcast graphics make good slides. If your supervisor still wants the interim memo, it's mostly `WEEK10.md`, `WEEK11.md` and the showcase notes |
| **Wed Nov 25, 11:59 p.m. MST** | **SABR abstract** to analytics-presentations@sabr.org (`docs/SABR_ABSTRACT_2027.md`) | It's the day before Thanksgiving. Send it Monday, Nov 23 |
| **Nov 30–Dec 4** · fall Week 14 | **Fall research package** to your supervisor, as planned | Add the checkpoint examples as an "in real games" section, and the umpire roster note (`docs/UMPIRE_CHECK.md`) to the data section. Describe the challenge fix in the data section: finding it by checking against MLB's official counts is a strength to show |
| **Dec 7–11** · fall Week 15 | Buffer, which the fall plan reserved for "a data or model issue you deferred." **Refit p(s) on the corrected challenges:** `13_define_cohort.py` (drops the 12 non-regular-season games), then `run_week7.py`, then `run_week89.py` for a new lock. Ask your supervisor before re-opening September with `--final-test`. Then re-run Weeks 10–11 and the showcase. If there's time left, **draft the analysis plan** (spring Week 2's deliverable) | The corrected numbers then go into the SABR deck in January. The analysis plan is the spring item that gains the most from early feedback |
| **December** | Ask your department about student travel funding for SABR | Finalists must register on Feb 19. Don't find out about funding then |
| **Dec 18** | SABR requests papers from general-track presenters | Student track: nothing until Jan 22 |
| **Winter break** | Optional: build the SABR student-track sample deck (due Jan 22) from the checkpoint slides. Start the Week 1 data pulls a few days before classes | Both run fine without you watching |

---

## Week 1 — Jan 18–22 · Re-baseline and Re-pull

> **Deliverable:** Fall release tagged and reproduced, full data re-pull running, analysis-plan draft
> Mon Jan 18 is MLK Day. SABR student-track sample deck (or general-track paper) due Fri Jan 22.

### The point

Start the semester from a state you can prove: the fall numbers reproduce exactly, and the long pulls are running before anything needs them.

### Do this

1. **Tag the fall release** (30 min). `git tag v1.0-fall` on the commit behind the fall research package. Every spring number gets compared against this tag.

2. **Reproduce the fall headline numbers** (1 hr). Re-run `run_week10.py` and `run_week11.py` and check them against the fall package: wrong calls per game, wins moved, what the first challenge is worth, and the share of above-bar chances taken. If anything moved, find out why before you start anything new. If the p(s) refit didn't happen in December, it comes first this week, because every other spring week depends on it. (The November numbers from before the challenge fix, for comparison, are 9.62 wrong calls a game, 272.3 wins moved, 5.02 points and 18.9%.)

3. **Finish the raw Statcast pull, and start it Monday** (1 hr to start, hours to run). Your `data/raw/statcast/` folder holds only 15 games, so since fall Week 10 the code has been falling back on the Week 5 pitch table. The April reproducibility package has to rebuild from raw, so re-run `01_statcast_pitches_2026.py` for the full season (it resumes from its chunks). Your raw folder has also lost the player bios, framing, umpire and win-probability pulls, so re-run `07`, `04`, `05` and `06` after it (07 and 05 need 01's games). The November rebuild kept those columns from the existing table, but the reproducibility package needs their raw files. Script 02 already saves each game's raw feed, so the challenges need no new pull. Then rebuild once from raw with `run_week5.py` (no `--spine-from-table`): it should match the December table apart from the Statcast revisions you diff in step 5. Update `data/raw/MANIFEST.json`.

4. **Test the Triple-A source on one day** (2 hrs). This is the same feasibility check as fall Week 3, for the Mar 5 transportability test. Can you get Triple-A pitch locations, calls and challenges? Start by pointing `02_abs_challenges_statsapi.py` at Triple-A games (the Stats API calls Triple-A sport id 11) and checking what the game feeds carry. If it works, start the full pull. If not, write down now that transportability drops out of the Mar 5 milestone, and tell your supervisor.

5. **Diff the new pull against the fall data** (2 hrs). Compare row counts by month and count how many pitches changed (Statcast revises old data). Then check the one that matters: did the location or call of any challenged pitch change? Write it up in `docs/DATA_REFRESH.md`.

6. **Draft the statistical analysis plan** (3 hrs), `docs/ANALYSIS_PLAN.md`. For RQ3 and RQ4, write down the exact question, the outcome, the sample, the test, and what result would count as a "yes". Write it before you look at any spring result. That's what makes Feb 26 and Mar 19 believable.

7. **SABR, if you submitted** (2–3 hrs). Student track: the sample deck is the checkpoint slides with the fall Week 10–11 results up front. General track: the paper.

### Done when

- [ ] `v1.0-fall` tagged, and the fall headline numbers reproduce exactly
- [ ] `24_challenge_count_check.py` shows every stage within ~2% of MLB's 10,557 challenges
- [ ] Full-season raw Statcast and challenge files in `data/raw/`, manifest updated
- [ ] Triple-A source tested: either the pull is running or there's a written decision to drop it
- [ ] `docs/DATA_REFRESH.md` lists what changed since the fall pull
- [ ] Analysis-plan draft sent to your supervisor
- [ ] SABR sample deck or paper submitted (if applicable)

### Watch out for

Treating every changed row as a problem. Statcast revises data, so small differences are expected. What matters is whether they touch the challenged pitches or the cohort.

---

## Week 2 — Jan 25–29 · Freeze the Data and the Plan

> **Deliverable:** Data refreshed and frozen; statistical analysis plan finalized (Jan 29)

### The point

Lock the two things every spring result rests on, the data and the questions, before you run a single spring analysis.

### Do this

1. **Decide what the refresh means for the cohort** (2 hrs). If the diff touched no challenged pitch and under 0.1% of called pitches, keep the fall cohort (`419af01ad9eee03f`) and footnote the diff. If it touched more, re-freeze it as cohort v2 and **re-score the locked p(s) and V(s) on it, but don't refit them**. Their held-out metrics should barely move. Either way, write the decision and the hash in NOTES.md.

2. **Rebuild from raw** (2 hrs). Delete `data/interim/` and `data/processed/` (keep `raw/`), run the pipeline end to end, and check that the cohort hash comes back. This tests the reproducibility claim for real, and now is a better time to find out than Apr 22.

3. **Read Lee et al. (2026) before finalizing RQ4** (1 hr). "When the Strike Zone Becomes Algorithmic" (arXiv:2609.25525, September 2026) studies how umpires' zones shift after overturns and which pitches players choose to challenge. Make your RQ4 tests extend it rather than repeat it, and cite it.

4. **Finalize `ANALYSIS_PLAN.md` with your supervisor** (3 hrs). Pin down:
   - primary vs secondary analyses (one primary per RQ)
   - the information set for RQ3: the realistic policy (Week 4) is primary, and full information is the upper bound
   - the September holdout: what's fit on March–August and what's only evaluated on September
   - the unit of resampling (the game) and the number of bootstrap draws
   - minimum samples: umpires with 10+ plate games (the fall cut), and players with enough opportunities (pick the number now)
   - multiple comparisons: with 30 teams and about 90 umpires, use false-discovery-rate control (Benjamini–Hochberg)
   - what would count as "behavior changed" for RQ4

5. **Freeze** (30 min). `git tag v1.1-data-freeze`, with the date and hashes in NOTES.md.

6. **Set up the spring structure** (1 hr): scripts `25_…` onward, a runner per milestone, and the report names from "How to read this."

### Done when

- [ ] Refresh decision recorded with the cohort hash
- [ ] Pipeline rebuilds from raw in one command and reproduces the hash
- [ ] `ANALYSIS_PLAN.md` signed off by your supervisor
- [ ] `v1.1-data-freeze` tagged

### Watch out for

Peeking. Once you've seen which teams look bad, every test you write will happen to find it. The plan comes first. That's its whole value.

---

## Week 3 — Feb 1–5 · End-to-End Policy Solver

> **Deliverable:** End-to-end policy solver (Feb 5)

### The point

Build one function that takes a pitch as recorded and returns the decision the policy makes, with the numbers behind it. Everything after this week calls it.

Fall Week 11 built the hard part: `load_mdp()` re-solves the game in seconds, and `Solution.lookup()` gives the bar for any state. What's missing is the plumbing from a pitch row to a decision, plus the inputs the fall model averaged over.

### Do this

1. **Wrap it** (2 hrs). Write `decide(pitches)`, which returns p, the bar, Q_challenge, Q_hold and the action for every challenge opportunity. It builds the state (inning, half, outs, runners, count, score, and both teams' challenges left), takes p from the locked model and takes the bar from the solved MDP. Vectorize it, because it has to run on the whole season.

2. **Check it against fall Week 11** (1 hr). Run it on every 2026 opportunity with league-average settings. It must reproduce §8 of the Week 11 report from the post-fix re-run exactly: the share of challenges above the bar, and the share of above-bar chances taken.

3. **Make the umpire an input** (3 hrs). The fall MDP assumes the league's supply of wrong calls (9.62 a game). Week 11 §9 showed that the supply changes what a challenge is worth: 4.45, 5.00 and 5.61 points across the three umpire tiers. `PitchModel.with_p_supply()` already takes a supply. Make `decide()` take the home-plate umpire, using the tiers (or shrunken per-umpire rates), each solved once and cached.

4. **Tests** (2 hrs). `decide()` should agree with `lookup()` on random states, and league-average settings should reproduce Week 11. Also check that the gap between umpire tiers shrinks as the game ends, since fewer future mistakes are left to save a challenge for.

5. **Start Week 5 now** (2–3 hrs). Write the replay loop for a single game (Week 5, step 1). February has three heavy weeks in a row, and this is how you make one of them lighter.

### Done when

- [ ] `decide()` runs on the full season
- [ ] It reproduces the Week 11 §8 numbers with league-average settings
- [ ] Umpire supply is an input, with tests
- [ ] A written list of what the solver still averages over (team quality, batter, pitcher). That's a paragraph of the limitations section
- [ ] One game replays end to end (the head start on Week 5)

### Watch out for

Rewriting fall Week 11. The solver is tested and verified on real data. Wrap it; don't rebuild it.

---

## Week 4 — Feb 8–12 · Optimal Policy v1 and the Heatmaps

> **Deliverable:** Optimal policy v1 and policy heatmaps (Feb 12)

### The point

Turn the solver into a policy a person can read, showing when and where to challenge as pictures. Then settle what a player can actually see.

### Do this

1. **Estimate how sharp a trained human's view of the pitch is** (3 hrs). The fall bar is compared with p(s), which comes from the tracked location to a fraction of an inch. Nobody on the field sees that. Use the umpires: the chance a call is wrong, as a function of the pitch's distance from the zone edge, is the error curve of a trained observer with the best view in the park. Fit it with a probit in distance, which gives the blur as σ in inches, with an interval. It uses no challenge decisions, so RQ3 can't become circular. Tennis has a literature on exactly this kind of perceptual error in line calls (Mather, 2008), and Lee et al. (2026) find that MLB players challenge on visible evidence more than on zone geometry. Both are worth citing here.

2. **Define the realistic policy** (2 hrs). p_seen is the overturn chance as a player with a σ-blurred view would estimate it. Policy v1 applies the same MDP bar to p_seen. Keep the full-information policy as the upper bound. The gap between the two is the value of a better view of the pitch, which makes a good poster line.

3. **The heatmaps** (4 hrs):
   - the bar by inning × score, one panel for each combination of challenges left (yours × theirs)
   - the bar by count × outs and runners, for a close, late game
   - "where to challenge" plate maps: the bar turned back into inches around the zone, for a called strike in the 2nd vs the 9th
   - each one for both policies

   Export every table as data (CSV/JSON). The dashboard's Explorer tab (spring Week 11) reads them.

4. **Sanity read** (1 hr). Show the heatmaps to someone who watches a lot of baseball and ask what looks wrong. Anything surprising is either a finding or a bug. Find out which before Friday.

5. **Write `docs/POLICY_V1.md`** (2 hrs): σ, both policies, the heatmaps, and what changed from the fall bar.

### Done when

- [ ] σ estimated with an interval, and p_seen defined
- [ ] Two policies, realistic and full information, from the one solver
- [ ] At least four heatmaps, baseball-sane, with their data exported for the dashboard
- [ ] `docs/POLICY_V1.md` written

### Watch out for

Estimating σ from the teams' own challenge decisions. That would excuse every bad decision as bad eyesight and shrink RQ3's regret to nothing. Use the umpires' error curve, then halve and double σ in Week 7.

---

## Week 5 — Feb 15–19 · Monte Carlo Game-Replay Simulator · **HEAVY WEEK**

> **Deliverable:** Monte Carlo game-replay simulator, benchmarked against simple policies (Feb 19)
> SABR finalists are notified Feb 19 and must register then. Mon Feb 15 is Presidents' Day at many schools.

### The point

Replay the real 2026 games under different challenge policies and count the wins. That turns "points of win probability" into wins, standings and playoff spots.

### Do this

1. **The replay loop** (3 hrs, started in Week 3). Walk through a game's real pitches. At each opportunity, ask the policy. While it agrees with what the team did, the real game stands. At the first pitch where it doesn't, the real game stops applying, and the rest is computed from that state.

2. **A behavioral policy** (2 hrs). This is a model of what teams actually did: the chance of a challenge given p_seen, the situation and the challenges left, fit on March–August. It plays the other team once the replay leaves the real game, and it's the "as teams played" benchmark.

3. **Validate** (2 hrs). Replaying "what teams did" must give back the real results, and replaying the behavioral policy should match real win rates and run scoring. If either is off, nothing after it can be trusted.

4. **Benchmark** (3 hrs). Replay each policy over all 2,429 games: the fall Week 11 rules (never, p ≥ 0.5, Statcast's breakeven, Statcast × leverage, play for runs), the realistic policy and the full-information policy. Do it two ways: one team switches (the front-office question), and every team switches. Report wins per 162 with game-bootstrap intervals, and compare with Week 11 §7.

5. **Standings** (2 hrs). Rebuild the 2026 standings under each policy. How many playoff spots or division titles change hands? Fall Week 10 deferred this season-level question to spring.

### Done when

- [ ] The replay reproduces the real results for "what teams did"
- [ ] Every policy benchmarked both ways, with intervals
- [ ] Standings under each policy
- [ ] `docs/SIMULATOR.md`: the method, the validation and the benchmark table

### Watch out for

Runtime. Simulate only what you need. Expected wins come exactly and quickly from policy evaluation: solve the game with both policies fixed and read the value at the state where the replay left the real game. That's how `wp_loss()` in `abs_mdp.py` did fall Week 11 §7. Save Monte Carlo (`Simulator`) for what averages can't give you: standings, playoff odds and spreads. Develop on 50 games, then run the full set overnight with saved seeds.

---

## Week 6 — Feb 22–26 · RQ3: Decision Regret · **HEAVY WEEK**

> **Deliverable:** Team and player decision-regret results (RQ3) (Feb 26)
> SABR: final abstract changes due Feb 26 (finalists only).

### The point

Answer RQ3: how far each team's and each player's decisions were from the optimal policy, in wins, with honest uncertainty.

### Do this

1. **Regret per opportunity** (2 hrs). It's the better of Q_challenge and Q_hold minus the Q of what they did, in points of win probability. It's zero when they chose right, and summed over a season it's in wins. The primary measure uses the realistic policy, and full information is the upper bound.

2. **Split the mistakes** (2 hrs) into challenges that shouldn't have been made (below the bar) and chances that should have been taken (above the bar, not challenged). Fall Week 11 suggests the second kind dominates: in the November data, teams took fewer than a fifth of their above-bar chances. Check how that changes with the full challenge count.

3. **Teams** (3 hrs). Compute season regret per team in wins, with game-bootstrap 95% intervals, and rank the teams. Then run the two tests from the analysis plan. First, do teams differ more than chance would predict? Use a permutation test that shuffles decisions between teams in similar situations. Second, does a team's March–August regret predict its September regret? If not, the ranking is luck.

4. **Players and roles** (3 hrs). Compare batters, catchers and pitchers as challengers. Then look at individual challengers above the minimum sample, shrunk toward their role's average (empirical Bayes). A raw leaderboard of small samples is mostly noise.

5. **Write `docs/RQ3_REGRET.md`** (2 hrs), with the three figures you'd put on the poster.

### Done when

- [ ] Regret per opportunity, team, role and player, for both the realistic and full-information policies
- [ ] An interval on every team and player number
- [ ] The two kinds of mistakes separated
- [ ] September out-of-sample check done
- [ ] RQ3 written up

### Watch out for

Ranking noise. A list of 30 teams always has a top and a bottom. The finding is whether the spread is bigger than chance, so say that before you show the ranking.

---

## Week 7 — Mar 1–5 · Robustness · **HEAVY WEEK**

> **Deliverable:** Robustness report: bootstrap uncertainty, calibration sensitivity, MLB-vs-AAA transportability (Mar 5)
> SABR final slides due Mar 5 (finalists).

### The point

Find out which conclusions survive when the inputs are wrong in plausible ways, and say so before a reviewer does.

### Do this

1. **Bootstrap the whole chain** (3 hrs + overnight). Resample games, refit p(s) and V(s) with their locked specifications, re-solve the MDP (seconds), and recompute the headline numbers: what the first challenge is worth, wins moved, the policy gap and team regret. Run 200 resamples overnight.

2. **Calibration sensitivity** (2 hrs). p(s) is locked uncalibrated because calibration didn't help in fall Week 9, but "didn't help" doesn't mean "can't be off." Bend it, p → logistic(a + b·logit p), across the range the Week 9 calibration fits allow. Then re-run the policy and the regret, and report what flips.

3. **σ and game-model sensitivity** (2 hrs). Halve and double the perception error. Then perturb the game model: scoring up and down 5%, and the supply of wrong calls up and down 10%. Re-solve each time. The bar and the regret ranking should move little. If they move a lot, that's a finding.

4. **MLB vs Triple-A** (3 hrs, if Week 1's source worked). Score Triple-A pitches with the locked p(s) and check calibration and discrimination there. If it transports, the policy applies to the minors too. If it doesn't, say what's different (zone definition, umpires, challenge rules).

5. **One table** (2 hrs): a row per headline conclusion, a column per check, and each cell marked holds, weakens or flips.

### Done when

- [ ] Bootstrap intervals on every headline number
- [ ] Calibration, σ and game-model sensitivity done
- [ ] Transportability result, or the written reason it was dropped (Week 1)
- [ ] The robustness table in `docs/ROBUSTNESS.md`
- [ ] SABR slides submitted (if a finalist)

### Watch out for

SABR landing on this week. If you're a finalist (you'll know Feb 19), the slides are the checkpoint deck plus the Week 5 benchmark. Build them the weekend of Feb 27–28, not on Thursday.

---

## Week 8 — Mar 8–12 · RQ4, Part 1: Selectivity over the Season

> **Deliverable:** How challenge selectivity changed over the season (RQ4) (Mar 12)
> SABR Analytics Conference, Phoenix, Mar 12–14.

### The point

Find out whether teams got choosier, or just better, about which calls to challenge as 2026 went on.

### Do this

1. **Measure selectivity three ways** (1 hr, as written in the analysis plan): the average p of challenged pitches, the share of above-bar chances taken, and regret per opportunity. Break each down by month (or two-week block) and by challenger role.

2. **Model the trend** (3 hrs). On all opportunities, model the chance of a challenge given p_seen, the situation and the time of season, with a p × time interaction. The interaction is the answer: it says whether teams got more selective with the pitch held fixed. Cluster standard errors by game.

3. **Learning or lineup?** (2 hrs). Did the same players change, or did the mix of who challenges change (say, teams telling batters to stop)? Split by role, and fit within-player trends for players with enough challenges.

4. **Why the success rate alone can't answer this** (1 hr). The overturn rate is the number in the news, but it goes up when teams challenge only the obvious calls, even if they're leaving more value on the table. One figure showing that is worth a lot in a talk.

5. **Write `docs/RQ4_SELECTIVITY.md`** (2 hrs).

### Done when

- [ ] Selectivity by month and role, all three measures
- [ ] Trend model with intervals, and the p × time answer stated in one sentence
- [ ] Learning vs lineup split
- [ ] RQ4 part 1 written up

### Watch out for

The conference. If you're presenting in Phoenix, finish by Wednesday, Mar 10, or agree on a Monday, Mar 15 date with your supervisor in February. Don't wait until that week to ask.

---

## Week 9 — Mar 15–19 · RQ4, Part 2: Umpires and Catchers; Freeze

> **Deliverable:** Catcher/umpire behavioral analysis; inferential analyses frozen (RQ4) (Mar 19)

### The point

Find out whether the people on the field changed once their mistakes could be overturned. Then stop analyzing.

### Do this

1. **Umpires over the season** (3 hrs). Take fall Week 10's difficulty-adjusted extra wrong calls per 100 called pitches by month, and fit a mixed model with umpire random effects and a season trend. Did umpires get more accurate, or call the edges differently, once they could be challenged? Lee et al. (2026) looked within games, after overturns. The season trend is yours. Separate the full-time staff from Triple-A call-ups with `scripts/23_umpire_roster_check.py --staff`, so a mid-season wave of call-ups doesn't look like a trend.

2. **Catchers** (3 hrs). For framing, track the called-strike rate on borderline pitches by catcher, by month. Did framing lose value once the worst stolen strikes could be challenged? Use only changes within 2026. Savant's 2026 framing metric credits successful challenges, so a 2025-vs-2026 comparison measures the metric's change, not behavior (see `docs/DATA_SOURCES.md`).

3. **Batters** (optional, 2 hrs). With challenges left, did batters take more borderline pitches?

4. **Freeze** (1 hr). `git tag v2.0-analysis-freeze`. From here on it's bug fixes only, each with a NOTES.md entry.

5. **Write `docs/RQ4_BEHAVIOR.md`** (2 hrs).

### Done when

- [ ] Umpire trend with intervals, with staff and call-ups separated
- [ ] Catcher framing trend
- [ ] RQ4 written up (both parts)
- [ ] `v2.0-analysis-freeze` tagged

### Watch out for

Mistaking the calendar for adaptation. Crews rotate, call-ups come and go, rosters expand in September, and pennant races change incentives. Compare like with like: difficulty-adjusted, with the same umpires and the same catchers.

---

## Week 10 — Mar 22–26 · Spring Break / Buffer

No deliverable. Rest, or catch up on whichever of Weeks 5–9 slipped.

If everything is on time, there's one low-effort win: rebuild from raw on a clean copy of the repo (Week 14's test, a month early). It catches silent breakage and costs one command.

---

## Week 11 — Mar 29–Apr 2 · Dashboard Beta

> **Deliverable:** Dashboard beta: policy explorer, team pages, game replay (Apr 2)

### The point

Put the spring results in the dashboard the way `dashboard/README.md` planned it: Python computes the answers, and the browser looks them up.

### Do this

1. **Export** (2 hrs). Extend `18_export_model_dashboard.py` to write the policy tables (both policies), the team results (regret and wins under each policy) and a pitch timeline for each game. Keep every file small: split the policy table by inning, round to 3 decimals, and write one file per game, loaded only when someone picks that game.

2. **Optimal Challenge Explorer** (3 hrs). Pick the inning, score, outs, runners, count and challenges left, and it shows the bar and a plate map of where a challenge is worth it. It's the Week 4 heatmaps, made clickable.

3. **Team pages** (2 hrs): each team's regret, both kinds of mistakes, and its wins under each policy.

4. **Game Replay** (3 hrs). Pick a game, and it shows the win-probability line with every challenge and missed chance marked, plus the what-if branch. It's the dashboard version of the checkpoint's game-story slides (`20_story_*`).

### Done when

- [ ] Three new tabs work, in light and dark mode
- [ ] No errors on empty selections
- [ ] The page still loads fast on a phone

### Watch out for

Shipping the whole season. The fall rule still holds: pre-aggregate, and load a game's pitches only when someone picks that game.

---

## Week 12 — Apr 5–9 · Dashboard Release Candidate

> **Deliverable:** Dashboard release candidate (Apr 9)

### The point

Make the dashboard work for someone who isn't you, including people who don't watch baseball.

### Do this

1. **Usability test** (3 hrs). Recruit three to five people, at least two of whom don't follow baseball. Give each one three tasks, like "find your team's most costly missed challenge" or "should a team challenge a called strike three with two outs in the 9th?" Watch, don't help, and write down where they get stuck.

2. **Fix what they hit** (4 hrs).

3. **Plain-language pass** (2 hrs). Put one sentence under every chart saying what it shows. Reuse the checkpoint glossary.

4. **Feature freeze** (30 min). `git tag v2.1-dashboard-rc`.

### Done when

- [ ] Usability notes written up, with what you changed
- [ ] Every chart has a plain-language line
- [ ] Release candidate tagged and live on GitHub Pages

### Watch out for

New tabs. Anything that isn't in the beta becomes a "future work" bullet in the report.

---

## Week 13 — Apr 12–16 · Write the Report

> **Deliverable:** Full report draft and poster draft to your supervisor (by Thu Apr 15)

### The point

Turn NOTES.md and the milestone docs into the report and the poster. No new analysis.

### Do this

1. **Outline from what you have** (2 hrs):
   - introduction and previous work: Tango's leverage and win expectancy, Statcast's breakeven and ABS metrics, MLB.com's 2026 challenge analyses, Lee et al. (2026), and the tennis challenge papers
   - data and cohort (fall Weeks 3–6)
   - p(s) (fall Weeks 7–9)
   - what a call is worth (fall Week 10)
   - the MDP (fall Week 11)
   - the spring results: policy v1, the simulator, RQ3, robustness, RQ4
   - the dashboard, limitations and future work

2. **Draft** (6 hrs). Most of it lifts from `WEEK*.md` and the spring docs. Write the limitations section yourself: one season, a derived challenges-left state, a perception model with a single σ, and players averaged over.

3. **Poster draft** (3 hrs). Use one figure per research question: the bar heatmap (RQ1), the value of a call (RQ2), team regret (RQ3) and selectivity over the season (RQ4). Put the checkpoint glossary in a corner.

4. **Send both by Thursday**, so the comments land in Week 14.

### Done when

- [ ] Complete draft, with every section present
- [ ] Every number traced to a script (keep a table)
- [ ] Poster draft done
- [ ] Both with your supervisor by Thursday

### Watch out for

Writing turning into analysis. Writing will surface something you wish you'd checked. Unless it's a bug, it goes in limitations.

---

## Week 14 — Apr 19–23 · Freeze and Package

> **Deliverable:** Final analysis frozen; reproducibility package (Apr 23)

### The point

Prove that someone else can rebuild every number from scratch, then lock it.

### Do this

1. **Clean-machine test, on Monday** (3 hrs). Use a fresh clone, a fresh environment from `requirements.txt` and raw data from the manifest, and run one command to every number and figure.

2. **The reproducibility package** (3 hrs): run instructions in the README, pinned requirements, the data manifest with checksums, the model cards (p(s), V(s) and the MDP), the analysis plan, and the table mapping numbers to scripts.

3. **Release** (1 hr). Make a GitHub release, `v3.0-final`. A Zenodo DOI is free and makes the code citable.

4. **Revise the report** with your supervisor's comments (3 hrs).

5. **Print the poster** (1 hr + lead time). Check the print shop's turnaround the week before. Most need two or three working days.

### Done when

- [ ] Clean-machine rebuild works
- [ ] Package complete and released
- [ ] Report revised
- [ ] Poster sent to print

### Watch out for

The clean-machine test failing on Thursday. Do it Monday.

---

## Week 15 — Apr 26–30 · Poster and Final Report

> **Deliverable:** Research poster presentation (Mon Apr 26); final report (Fri Apr 30)

1. **Rehearse three versions** of the poster pitch: 30 seconds, 2 minutes and 5 minutes. The 30-second one has to make sense to someone who doesn't watch baseball, and the checkpoint quiz makes a good opener.
2. **Present Monday.** Bring the dashboard on a laptop, with Game Replay open on your best game.
3. **Make the final edits and submit Friday.**
4. **Archive.** Push the final tag, check the dashboard link, and write a last NOTES.md entry.

---

## The three things most likely to go wrong this semester

1. **February is three heavy weeks in a row.** The simulator (Feb 19), RQ3 (Feb 26) and robustness (Mar 5) land back to back, with SABR deadlines on top. Weeks 3 and 4 have slack because fall Week 11 built the solver, so spend it on the replay loop. If you're going to fall behind, do it in Week 3 or 4, not Week 5.

2. **The comparison to "optimal" is unfair.** If RQ3 measures teams against a policy that knows the tracked location to a fraction of an inch, every team looks bad, and a reviewer will point that out before you do. The realistic policy (Week 4) is what makes RQ3 defensible. Don't cut it to save time.

3. **Noise reads as a finding.** With one season, 30 teams, hundreds of challengers and about 90 umpires, something will always look extreme. The analysis plan (Week 2), shrinkage, the September holdout and false-discovery control are what separate a result from a leaderboard of luck. They only work if you decide them before you look.
