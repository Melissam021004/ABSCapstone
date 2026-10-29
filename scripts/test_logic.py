"""
test_logic.py — unit tests for the two pieces of logic you cannot eyeball.

Run this before you trust script 08, and again any time you touch the zone
constants or the challenge state machine:

    python scripts/test_logic.py

It checks, against hand-computed synthetic games:
  - the ABS zone geometry (17-inch plate, 53.5%/27% of batter height, ball
    radius counted, corner distances, monotonicity)
  - the challenge inventory (failed challenge costs one, successful is retained,
    floors at zero, extra-inning replenishment only when exhausted, resets
    between games)
  - the opportunity flags (called strike is the batter's opportunity, called
    ball is the fielding side's, exhausted sides get none)

Needs no network and no downloaded data.
"""
import sys, importlib.util, py_compile
from pathlib import Path
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent

# ---------- 1. every script must at least compile ----------
print("1. COMPILE CHECK")
for f in sorted(S.glob("*.py")):
    try:
        py_compile.compile(str(f), doraise=True)
        print(f"   ok    {f.name}")
    except Exception as e:
        print(f"   FAIL  {f.name}: {e}")
        sys.exit(1)

def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py",""), S/name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

b = load("08_build_challenge_inventory.py")

# ---------- 2. ABS zone geometry ----------
print("\n2. ABS ZONE GEOMETRY")
H = 6.0                      # a 6'0" batter
top, bot = H*0.535, H*0.27   # 3.21 ft, 1.62 ft
print(f"   6'0\" batter zone: {bot:.3f} .. {top:.3f} ft  ({(top-bot)*12:.1f} in tall)")
assert 1.55 < bot < 1.70 and 3.10 < top < 3.30, "zone bounds implausible"

cases = [
    # (plate_x, plate_z, expect_strike, label)
    (0.0,  2.4,  True,  "dead centre"),
    (0.70, 2.4,  True,  "inside the plate edge"),
    (0.82, 2.4,  True,  "ball clipping the outside corner"),
    (1.10, 2.4,  False, "clearly outside"),
    (0.0,  3.30, True,  "ball just above the top (radius overlaps)"),
    (0.0,  3.60, False, "clearly high"),
    (0.0,  1.55, True,  "ball just below the bottom (radius overlaps)"),
    (0.0,  1.20, False, "clearly low"),
]
ok = True
for px, pz, expect, label in cases:
    d = float(b.zone_distance_ft(px, pz, top, bot))
    got = d < 0
    flag = "ok  " if got == expect else "FAIL"
    if got != expect: ok = False
    print(f"   {flag} {label:<38} x={px:+.2f} z={pz:.2f}  dist={d*12:+6.2f} in  strike={got}")
assert ok, "zone geometry wrong"

# monotonicity: moving away from the zone must increase distance
xs = np.array([0.0, 0.5, 0.9, 1.3, 1.8])
ds = b.zone_distance_ft(xs, np.full(5, 2.4), np.full(5, top), np.full(5, bot))
assert np.all(np.diff(ds) > 0), "distance not monotone in |plate_x|"
print("   ok   distance increases monotonically away from the zone")

# corner case must exceed both edge distances
corner = float(b.zone_distance_ft(0.95, 3.45, top, bot))
side   = float(b.zone_distance_ft(0.95, 2.40, top, bot))
assert corner > side, "corner distance should exceed pure-side distance"
print(f"   ok   corner miss ({corner*12:.2f} in) > side miss ({side*12:.2f} in)")

# ---------- 3. challenge-inventory state machine ----------
print("\n3. CHALLENGE-REMAINING STATE MACHINE")

def make_game(game_pk, n_ab=12, innings=None):
    rows = []
    for ab in range(1, n_ab+1):
        inn = innings[ab-1] if innings else ((ab-1)//2 + 1)
        topbot = "Top" if ab % 2 == 1 else "Bot"
        for p in range(1, 4):
            rows.append(dict(game_pk=game_pk, at_bat_number=ab, pitch_number=p,
                             inning=inn, inning_topbot=topbot,
                             description="called_strike" if p == 3 else "ball",
                             plate_x=0.1, plate_z=2.4, sz_top=3.4, sz_bot=1.6,
                             batter=100+ab, pitcher=200, fielder_2=300))
    return pd.DataFrame(rows)

def run(pitches, challenges):
    b.qa.clear()
    return b.derive_challenge_state(pitches, challenges)

# Case A: away batter fails a challenge -> away drops 2 -> 1
pit = make_game(1)
ch = pd.DataFrame([dict(game_pk=1, at_bat_index=0, pitch_number=1,
                        challenger_role="batter", challenger_id=101,
                        challenger_name="X", challenging_side="away",
                        is_overturned=False, call_code="C", description="d",
                        hp_umpire="U", hp_umpire_id=9)])
out = run(pit, ch)
# ab1 is Top => away batting. Pitch 1 is the challenge; state BEFORE it = 2.
r1 = out[(out.at_bat_number==1)&(out.pitch_number==1)].iloc[0]
r2 = out[(out.at_bat_number==1)&(out.pitch_number==2)].iloc[0]
assert r1.challenges_remaining_batting == 2, r1.challenges_remaining_batting
assert r2.challenges_remaining_batting == 1, r2.challenges_remaining_batting
assert r2.challenges_remaining_fielding == 2, "home side must be untouched"
print("   ok   failed challenge consumes exactly one, and only for that side")

# Case B: successful challenge is RETAINED
ch2 = ch.copy(); ch2.loc[0, "is_overturned"] = True
out = run(pit, ch2)
r2 = out[(out.at_bat_number==1)&(out.pitch_number==2)].iloc[0]
assert r2.challenges_remaining_batting == 2, r2.challenges_remaining_batting
print("   ok   successful challenge is retained (no decrement)")

# Case C: two failures exhaust the side, and it floors at 0
ch3 = pd.DataFrame([
    dict(game_pk=1, at_bat_index=0, pitch_number=1, challenger_role="batter",
         challenger_id=101, challenger_name="X", challenging_side="away",
         is_overturned=False, call_code="C", description="d", hp_umpire="U", hp_umpire_id=9),
    dict(game_pk=1, at_bat_index=2, pitch_number=1, challenger_role="batter",
         challenger_id=103, challenger_name="Y", challenging_side="away",
         is_overturned=False, call_code="C", description="d", hp_umpire="U", hp_umpire_id=9),
])
out = run(pit, ch3)
after = out[(out.at_bat_number==3)&(out.pitch_number==2)].iloc[0]
assert after.challenges_remaining_batting == 0, after.challenges_remaining_batting
print("   ok   two failures exhaust the side (floors at 0)")

# Case D: extra innings replenish ONLY an exhausted side
pit_x = make_game(2, n_ab=24, innings=[(i//2)+1 for i in range(24)])
ch4 = pd.DataFrame([
    dict(game_pk=2, at_bat_index=0, pitch_number=1, challenger_role="batter",
         challenger_id=101, challenger_name="X", challenging_side="away",
         is_overturned=False, call_code="C", description="d", hp_umpire="U", hp_umpire_id=9),
    dict(game_pk=2, at_bat_index=2, pitch_number=1, challenger_role="batter",
         challenger_id=103, challenger_name="Y", challenging_side="away",
         is_overturned=False, call_code="C", description="d", hp_umpire="U", hp_umpire_id=9),
])
out = run(pit_x, ch4)
reg = out[(out.inning==9)&(out.inning_topbot=="Top")]
ext = out[(out.inning==10)&(out.inning_topbot=="Top")]
assert reg.challenges_remaining_batting.max() == 0, "away should be exhausted in regulation"
assert ext.challenges_remaining_batting.iloc[0] == 1, ext.challenges_remaining_batting.iloc[0]
ext_home = out[(out.inning==10)&(out.inning_topbot=="Bot")]
assert ext_home.challenges_remaining_batting.iloc[0] == 2, "home never used any; stays at 2"
print("   ok   extra innings grant +1 only to an exhausted side")

# Case E: state resets between games
pit_two = pd.concat([make_game(1), make_game(3)], ignore_index=True)
out = run(pit_two, ch3)
g3 = out[out.game_pk==3]
assert (g3.challenges_remaining_batting == 2).all(), "game 3 must start fresh"
print("   ok   inventory resets at each new game")

# ---------- 4. opportunity flags ----------
print("\n4. OPPORTUNITY FLAGS")
out = run(pit, ch)
bios = pd.DataFrame([{"player_id": i, "height_ft": 6.0, "primary_position": "P" if i==200 else "SS"}
                     for i in list(range(100,120))+[200,300]])
out = b.add_abs_zone(out, bios)
out = b.flag_opportunities(out, bios)
cs = out[out.description=="called_strike"].iloc[0]
bl = out[out.description=="ball"].iloc[0]
assert cs.is_challenge_opportunity_batting and not cs.is_challenge_opportunity_fielding, \
    "a called strike is the BATTER's opportunity"
assert bl.is_challenge_opportunity_fielding and not bl.is_challenge_opportunity_batting, \
    "a called ball is the FIELDING side's opportunity"
print("   ok   called strike -> batter's opportunity; called ball -> fielding side's")

exhausted = out[(out.challenges_remaining_batting==0) & (out.description=="called_strike")]
assert not exhausted.is_challenge_opportunity_batting.any(), \
    "a side with 0 challenges cannot have an opportunity"
print("   ok   exhausted sides get no opportunities")

# a real strike called a strike is NOT overturnable
assert not out[(out.description=="called_strike") & (out.abs_strike)].would_be_overturned.any()
print("   ok   correct calls are never flagged as overturnable")

print("\nALL TESTS PASSED")
