# Challenge count check

Written by `scripts/24_challenge_count_check.py`. Official totals: Baseball-Reference, 2026 MLB ABS Challenge Analysis (https://www.baseball-reference.com/friv/abs-challenges.shtml); AP, Oct 2, 2026.

| stage | games | challenges | a game | overturned | catchers | batters | pitchers |
|---|---|---|---|---|---|---|---|
| MLB official | 2,429 | 10,557 | 4.35 | 53.9% | 5,609 (58.6%) | 4,766 (48.8%) | 182 (39.6%) |
| 1. extracted (script 02) | 2,430 | 10,558 | 4.34 | 53.9% | 5,613 (58.6%) | 4,766 (48.8%) | 179 (39.7%) |
| 2. joined to a pitch (script 08) | 2,429 | 10,558 | 4.35 | 53.9% | 5,613 (58.6%) | 4,766 (48.8%) | 179 (39.7%) |
| 3. in the cohort (script 13) | 2,429 | 10,535 | 4.34 | 53.9% | 5,600 (58.6%) | 4,757 (48.9%) | 178 (39.3%) |

- Extracted by the November 2026 extractor; recorded on: pitch_event 7,896, plate_appearance 2,662.

**The counts match MLB's (within 2%).**

## Which challenges are missing?

MLB's totals minus the data's, by challenger role and outcome:

| role | overturned missing | failed missing | overturned kept | failed kept |
|---|---|---|---|---|
| catcher | 6 | 3 | 99.8% | 99.9% |
| batter | 3 | 6 | 99.9% | 99.8% |
| pitcher | 2 | 2 | 97.2% | 98.2% |

With two strikes, a batter's failed challenge means strike three stands, and a catcher's successful one turns a ball into strike three. If those two kinds are the biggest gaps, the missing challenges are probably ones whose final call ended the at-bat: reviews the first extractor never looked for.

### A second hint: the overturn rate by what the final call did to the at-bat

If challenges whose final call ENDED the at-bat are missing, the first row's rate is far higher than the others (its failed challenges are gone) and the second row's far lower (its successful ones are gone). Teams are also choosier in some counts than others, so treat this as a hint; the re-extraction below is the real test.

| what the final call would do | challenges | overturned |
|---|---|---|
| upheld call ends the at-bat | 1,422 | 42.3% |
| overturned call ends it | 1,567 | 54.1% |
| either call ends it | 989 | 41.2% |
| the at-bat goes on either way | 6,557 | 58.2% |

Share of challenges whose final call ended the at-bat: 25.2%.

Statcast's play descriptions mention a challenge in 3,534 plate appearances; 839 of them have no challenge in the data.

What those descriptions say was challenged (managers' replay challenges of plays — a tag, a force play, a catch — aren't ABS challenges, so they belong here):

| challenged | plate appearances |
|---|---|
| play at 1st | 380 |
| tag play | 220 |
| hit by pitch | 77 |
| force play | 72 |
| catcher interference | 24 |
| catch or drop | 17 |
| home-plate collision | 10 |
| fair or foul in outfield | 8 |
| stadium boundary call | 8 |
| trap play | 5 |

- e.g. Mets challenged (fan interference), call on the field was overturned: Jacob Young flies out to right fielder Carson Benge in foul territory, on fan interference.
- e.g. Nationals challenged (home-plate collision), call on the field was upheld: Nationals challenged (tag play), call on the field was overturned: Jacob Young singles on a gro
- e.g. Rockies challenged (play at 1st), call on the field was overturned: Dylan Crews grounds out softly, catcher Brett Sullivan to first baseman TJ Rumfield. Jacob Young score

| month | games | challenges a game | official |
|---|---|---|---|
| 3 | 76 | 3.72 | 3.76 |
| 4 | 392 | 4.17 | 4.19 |
| 5 | 419 | 4.05 | 4.06 |
| 6 | 394 | 4.30 | 4.32 |
| 7 | 371 | 4.43 | 4.44 |
| 8 | 417 | 4.60 | 4.60 |
| 9 | 360 | 4.61 | 4.61 |

## What depends on the count

| result | how it depends |
|---|---|
| Challenges won back (Week 10) | directly: missing overturns are wins nobody gets credit for |
| Challenges above the bar, above-bar chances taken, the full-information gap (Week 11 §8) | directly: a missing challenge looks like a chance passed up |
| Challenger roles (Week 10 people) | directly: MLB has catchers 58.6%, batters 48.8% |
| Challenges left, the MDP's state | a missing FAILED challenge leaves a team one challenge it didn't have |
| The showcase's "never challenged" examples and quiz | some may have been challenged: confirm each on the replay |
| p(s) | trained on challenged pitches; refit after the fix (location-only, so expect a small change) |
| V(s), run expectancy, leverage | barely: built from every pitch, not from challenges |

## The fix

1. `python scripts/02_abs_challenges_statsapi.py --force` — the November 2026 extractor also reads reviews recorded on a separate event or on the plate appearance. It downloads each game once more (about 20–30 minutes) and saves the raw feed; after that, a run without `--force` re-filters from the saved records in about a minute. Its printout should show about 4.35 challenges a game, role totals near MLB's, and a per-team-game table like MLB's.
2. Rebuild the table: `python scripts/run_week5.py --spine-from-table` (08–11 on the pitches already in your table; drop the flag once script 01 has pulled the full season again), then `python scripts/13_define_cohort.py` (a new cohort fingerprint).
3. `python scripts/24_challenge_count_check.py` again: every stage should now be within ~2%.
4. Re-run Weeks 10–11 and `22_showcase_examples.py`; refit p(s) (Weeks 7–9) as a new version — it was trained on the incomplete challenges — then re-run 10, 11 and 22 once more.
5. NOTES.md: what was missing, why, and the new fingerprints.
