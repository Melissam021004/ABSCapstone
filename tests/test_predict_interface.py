#!/usr/bin/env python3
"""
test_predict_interface.py — is `predict(state) -> probability` safe to build Weeks 10–12 on?

The locked p(s) is called from code that knows nothing about the training
pipeline, starting from raw pitch locations. So the one thing that MUST be
true is: predict(state) gives exactly the probability the model gave during
training and scoring. This builds small bundles from the demo data with the
same `write_bundle` the Week 9 script uses, and checks:

  1. the zone geometry in src/abs_zone.py reproduces Week 5's zone_dist_in
  2. a location model: predict_frame from RAW columns == the model's own
     predictions on the pipeline's features, for every August challenge
  3. a boosted model that uses EVERYTHING (IDs as categories, umpire and
     catcher accuracy, count, leverage...): the same, to the last decimal
  4. predict(one dict) == the matching row of predict_frame
  5. the schema refuses bad states: missing fields, inches instead of feet,
     a call that isn't True/False
  6. a bundle whose files changed after locking refuses to load
  7. physics: on a called strike, moving the ball further off the plate never
     lowers p(s)

    python tests/test_predict_interface.py        (about 20 seconds)

Builds its own small season in a temporary folder (the Week 7 test's fixture),
so it runs anywhere. If your real cohort exists, check 1 also runs on it — it's
the one place the Week 5 zone geometry and predict(state) must agree on real data.
"""

import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.abs_baselines import fit_model                                   # noqa: E402
from src.abs_boosting import Booster                                      # noqa: E402
from src.abs_calibration import Calibrator                                # noqa: E402
from src.abs_features import BOOST_SETS, read_any                         # noqa: E402
from src.abs_pipeline import es_mask_by_date, load_challenges             # noqa: E402
from src.abs_predict import (StateError, load_production,                 # noqa: E402
                             states_from_cohort, write_bundle)
from src.abs_zone import zone_distance_in                                 # noqa: E402

warnings.filterwarnings("ignore")
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def bundle(folder, model, cal, data, family, feature_set, features):
    def save(f_):
        if isinstance(model, Booster):
            model.save(f_)
        else:
            with open(Path(f_) / "model.pkl", "wb") as fh:
                pickle.dump(model, fh)
    write_bundle(folder, model=model, calibrator=cal, rates=data["rates"], league=data["league"],
                 model_files=save,
                 manifest_extra={"date": "test", "model_key": "test", "model_label": "test",
                                 "family": family, "feature_set": feature_set,
                                 "features": features, "calibration": {"method": cal.method}})
    return load_production(folder)


tmp = Path(tempfile.mkdtemp(prefix="pred_"))
try:
    spec = importlib.util.spec_from_file_location("w7t", ROOT / "tests" / "test_week7_pipeline.py")
    w7t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w7t)
    season, mislinked = w7t.build_season(n_games=600)
    rng = np.random.default_rng(3)
    season["batter"] = rng.integers(400000, 400120, len(season))
    season["pitcher"] = rng.integers(500000, 500150, len(season))
    (tmp / "data" / "processed").mkdir(parents=True)
    season.to_csv(tmp / "data" / "processed" / "cohort_2026.csv", index=False)
    data = load_challenges(tmp / "data", tmp / "config")
    keys = set(map(tuple, mislinked.to_numpy()))
    va = data["va"]
    va = va[[k not in keys for k in map(tuple, va[["game_pk", "at_bat_number",
                                                   "pitch_number"]].to_numpy())]]
    va = va.reset_index(drop=True)                  # the 3 planted mislinks can't match
    tr = data["tr"]

    print("\nZone geometry")
    real = read_any(ROOT / "data" / "processed" / "cohort_2026.parquet")
    for name, df in (("your cohort", real), ("the test season", va)):
        if df is None or not {"plate_x", "plate_z", "abs_zone_top", "abs_zone_bot",
                              "zone_dist_in"} <= set(df.columns):
            print(f"  skip  {name}: not found")
            continue
        d = zone_distance_in(df["plate_x"], df["plate_z"], df["abs_zone_top"], df["abs_zone_bot"])
        gap = np.abs(d - pd.to_numeric(df["zone_dist_in"], errors="coerce").to_numpy(float))
        g = float(np.nanmax(gap)) if np.isfinite(gap).any() else float("nan")
        check(f"src/abs_zone.py reproduces Week 5's zone_dist_in on {name}", g < 1e-6,
              f"{int(np.isfinite(gap).sum()):,} pitches, max gap {g:.1e}")

    print("\nA location model (M2 + Platt)")
    m2 = fit_model("M2", tr)
    raw = m2.predict(va)
    cal = Calibrator("platt").fit(raw, va["y"])
    pm = bundle(tmp / "loc", m2, cal, data, "logistic", "location",
                ["dist_against_call", "edge", "call_dir"])
    p = pm.predict_frame(states_from_cohort(va))
    gap = float(np.max(np.abs(p - cal.predict(raw))))
    check("predict_frame from raw columns == the pipeline, every August challenge",
          gap < 1e-9, f"{len(va)} challenges, max gap {gap:.1e}")
    check("it only asks for the four core fields", pm.fields ==
          ["plate_x", "plate_z", "batter_height_ft", "called_strike"], str(pm.fields))
    row = states_from_cohort(va).iloc[7]
    state = {"plate_x": float(row.plate_x), "plate_z": float(row.plate_z),
             "batter_height_ft": float(row.batter_height_ft),
             "called_strike": bool(row.called_strike)}
    check("predict(dict) == the same row of predict_frame",
          abs(pm.predict(state) - p[7]) < 1e-12, f"{pm.predict(state):.6f}")

    print("\nA boosted model that uses everything (IDs, accuracies, count, leverage)")
    es = es_mask_by_date(tr)
    hb = Booster("hgb", **BOOST_SETS["full"], params={"max_leaf_nodes": 15,
                                                      "learning_rate": 0.1},
                 max_rounds=200, patience=30).fit(tr, tr["y"], es)
    raw_b = hb.predict(va)
    cal_b = Calibrator("isotonic").fit(raw_b, va["y"])
    pmb = bundle(tmp / "full", hb, cal_b, data, "hgb", "full", hb.features)
    pb = pmb.predict_frame(states_from_cohort(va))
    gap = float(np.max(np.abs(pb - cal_b.predict(raw_b))))
    check("predict_frame from raw columns == the pipeline, every August challenge",
          gap < 1e-9, f"max gap {gap:.1e}")
    ids = states_from_cohort(va).head(40).copy()
    as_float = ids.assign(**{c: ids[c].astype(float) for c in
                             ("hp_umpire_id", "fielder_2", "pitcher", "batter")})
    as_str = ids.assign(**{c: ids[c].astype(int).astype(str) for c in
                           ("hp_umpire_id", "fielder_2", "pitcher", "batter")})
    g2 = max(float(np.max(np.abs(pmb.predict_frame(as_float) - pb[:40]))),
             float(np.max(np.abs(pmb.predict_frame(as_str) - pb[:40]))))
    check("IDs given as 123, 123.0 or '123' all mean the same player", g2 < 1e-12,
          f"max gap {g2:.1e}")

    print("\nThe schema refuses bad states")
    def refuses(fn, needle):
        try:
            fn()
        except StateError as e:
            return needle in str(e), str(e)[:90]
        return False, "no error raised"
    ok, msg = refuses(lambda: pmb.predict(state), "missing field")
    check("a full model refuses a state without its extra fields (strict)", ok, msg)
    ok = np.isfinite(pmb.predict(state, strict=False))
    check("...and uses its training defaults when strict=False", bool(ok))
    ok, msg = refuses(lambda: pm.predict(dict(state, batter_height_ft=73)), "INCHES")
    check("height in inches is caught with a hint", ok, msg)
    ok, msg = refuses(lambda: pm.predict(dict(state, plate_x=10.2)), "INCHES")
    check("plate location in inches is caught with a hint", ok, msg)
    ok, msg = refuses(lambda: pm.predict(dict(state, called_strike="maybe")), "True/False")
    check("a call that isn't True/False is refused", ok, msg)
    ok, msg = refuses(lambda: pm.predict({k: v for k, v in state.items() if k != "plate_z"}),
                      "plate_z")
    check("a missing location is refused", ok, msg)
    check("'strike' / 'ball' are accepted for the call",
          abs(pm.predict(dict(state, called_strike="strike")) -
              pm.predict(dict(state, called_strike=True))) < 1e-12)

    print("\nLocked means locked")
    cj = tmp / "loc" / "calibrator.json"
    c = json.loads(cj.read_text())
    c["params"]["a"] += 0.01
    cj.write_text(json.dumps(c))
    try:
        load_production(tmp / "loc")
        check("a bundle edited after locking refuses to load", False, "it loaded")
    except RuntimeError as e:
        check("a bundle edited after locking refuses to load", "changed" in str(e),
              str(e)[:80])

    print("\nPhysics")
    # along the middle of the side edge (far enough out, the NEAREST edge becomes
    # the top or bottom one, and the model rightly switches curves there)
    xs = np.linspace(0.3, 1.2, 60)
    grid = pd.DataFrame({"plate_x": xs, "plate_z": 2.45, "batter_height_ft": 6.1,
                         "called_strike": True})
    pg = pm.predict_frame(grid)
    check("called strike: further off the plate never lowers p(s)",
          bool(np.all(np.diff(pg) >= -1e-12)), f"{pg[0]:.3f} -> {pg[-1]:.3f}")
    pg2 = pm.predict_frame(grid.assign(called_strike=False))
    check("called ball: further off the plate never raises p(s)",
          bool(np.all(np.diff(pg2) <= 1e-12)), f"{pg2[0]:.3f} -> {pg2[-1]:.3f}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PREDICT-INTERFACE CHECKS PASSED")
