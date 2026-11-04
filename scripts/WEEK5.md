# Week 5 — Join Everything

**Deliverable:** analysis-ready schema, data dictionary, QA report.

```bash
python scripts/run_week5.py
```

That's the whole week's mechanical work. It takes a few minutes on real data.
Everything below explains what it does and what you still have to do yourself.

---

## Try it before your pulls finish

```bash
python scripts/run_week5.py --demo
```

Generates a synthetic 90-game season under `data_demo/` and runs the entire
chain on it in about three seconds. You see every output — the table, the
dictionary, the QA report, the verification worksheet — before the real data
exists.

The fixture isn't decorative. The challenge events in it obey the actual 2026
rules: two per team, successful ones retained, extra-inning replenishment, only
batter/catcher/pitcher challenging, never with zero left, and a challenge is
overturned exactly when the umpire disagreed with the ABS zone. So if script
08's derivation is right the assertions pass, and if you break it they fail.
That makes it a test fixture, not a demo.

---

## How the plan's eight steps map to scripts

| Plan step | Where it happens |
|---|---|
| 1. Build the pitch spine | `08` |
| 2. Attach challenge outcomes | `08` |
| 3. Derive challenges remaining | `08` |
| 4. Flag challenge opportunities | `08` |
| 5. Attach the ABS zone and distance | `08` |
| 6. Attach umpire, catcher, game context | `08` |
| 7. Write the data dictionary | `09` |
| 8. QA report | `10` |
| "Watch out for" — hand-verify the inventory | `11` |

Steps 1–6 were already in script 08 from Week 4. Weeks 5's new work is 09, 10,
11 and the orchestrator.

---

## The scripts

### `08_build_challenge_inventory.py` — the table

One row per pitch, keyed `game_pk` + `at_bat_number` + `pitch_number`. Now also
writes `data/processed/join_ledger.csv`, a row count after every join step. The
row count must not change after step 1; if it does, a merge fanned out, and the
run says so.

### `09_data_dictionary.py` — `docs/DATA_DICTIONARY.md`

Every column with its type, source, definition, null rate, range and gotchas.

The part that matters for Week 8: each column gets a **modelling role**.

- `target` — what you predict, never an input
- `LEAKAGE RISK` — known only *after* the challenge resolved
- `feature` — safe, known at the moment of the decision
- `identifier` / `metadata`

The doc prints an explicit "drop these before Week 7" list. That list is the leak
your Week 8 notes warn about, named in advance instead of discovered through a
suspiciously good AUC.

Columns it doesn't recognise are still listed and flagged `UNDOCUMENTED`, so a
new Statcast column can never silently go undefined. To document one, add it to
`REGISTRY` at the top of the script and re-run.

### `10_qa_assertions.py` — `docs/QA_REPORT.md`

24 assertions. **Exit code 1 on any critical failure**, so you can wire it into a
Makefile or a git hook and stop a broken table reaching Week 6.

Critical ones include: one row per pitch; inventory within 0–2; every game starts
at 2 each; nobody challenges or holds an opportunity with zero left; no side
exceeds its failed-challenge budget; every challenge has a role and an outcome;
challenged pitches are called pitches; ≥95% of challenge events joined; and the
reconstructed ABS zone reproduces ≥85% of actual challenge outcomes.

Warnings cover the plausibility checks — overturn rate band, opportunity share
near Savant's ~50%, "reasonable" near ~5%, catchers out-performing batters,
overturn rate falling with distance from the edge, coverage of umpire, height and
leverage.

Every failure writes the offending rows to `docs/qa_failures/<check_id>.csv`, so
"13 rows violate the inventory bound" arrives with the 13 rows attached.

I mutation-tested these: breaking the retention rule, duplicating rows, putting
the inventory out of range, breaking the per-game reset, inverting the zone,
nulling outcomes and corrupting counts are each caught by the matching assertion,
all with exit code 1.

### `11_verify_challenges_remaining.py` — the one that matters

Your plan's "watch out for" is specifically this field, and it's right to be.
`challenges_remaining` is derived, not published; it's the MDP's state variable
`r`; and it runs without erroring whether it's right or wrong.

For each of three games the script prints a challenge-by-challenge ledger —
inning, count, challenger, role, side, result, and the inventory before and after
— alongside what the built table actually says. It also re-walks the inventory a
**second time, independently**, from the challenge events alone.

Then it writes `docs/VERIFICATION_CHALLENGES_REMAINING.md` with the Savant and
MLB.com links already embedded and a sign-off checklist.

```bash
python scripts/11_verify_challenges_remaining.py --games 5
python scripts/11_verify_challenges_remaining.py --pick extra-innings   # hardest case
python scripts/11_verify_challenges_remaining.py --game-pk 800078
```

`--pick extra-innings` is worth running: the replenishment rule is the part of
the logic with the least test coverage in real data.

---

## What no script can do for you

Two implementations agreeing proves they share no typo. It does not prove they
share no wrong assumption — and if I misread the extra-inning rule, both
implementations are wrong together and both report green.

So open three games in the Savant game feed and check by eye: the challenge
count, each challenger and role, each result, and that a side which lost two
stops appearing. The worksheet has the links in it and a checklist to tick.

Twenty minutes. It is the cheapest insurance in this project, and it is the
difference between finding a broken state variable now and finding it in
November with the MDP already built on top of it.

---

## When the QA report comes back red

Read `docs/QA_REPORT.md` top down — critical failures first, each with what it
means and a CSV of the offending rows.

The two most likely on real data:

**`JOIN_RATE` below 95%.** Challenge events aren't matching pitches. Almost
always the at-bat off-by-one (Statcast's `at_bat_number` is 1-based, the Stats
API's `atBatIndex` is 0-based) or pitch numbering that counts pickoffs and
timeouts. Open `docs/qa_failures/` and compare a few unmatched events against the
raw play-by-play.

**`ZONE_PREDICTS_OUTCOME` below 85%.** Your reconstructed ABS zone disagrees with
what ABS actually ruled. Check that `07_player_bios.py` ran and heights are
attached (`abs_zone_source` should be `batter_height`, not `statcast_sz`), that
the constants are 53.5% / 27%, and that the ball's 1.45-inch radius is being
subtracted.

---

## Done when

```
[ ] One analysis-ready file, one row per pitch
[ ] challenges_remaining derived, cross-checked, AND hand-verified on 3 games
[ ] Opportunity flags near Savant's ~50% / ~5%
[ ] ABS-zone distance from batter height, not Statcast's zone
[ ] Data dictionary covers every column
[ ] QA assertions run as a script and all pass
```

`run_week5.py` prints this list with the boxes ticked, except the hand-check —
that one is yours.
