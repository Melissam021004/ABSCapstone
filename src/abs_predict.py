"""
abs_predict.py — the locked overturn-probability model p(s), behind one stable call.

Weeks 10–12 (win probability, the MDP, the dashboard simulator) import THIS
module, never the training code. Whatever happens to the modelling scripts,
this interface stays the same:

    from src.abs_predict import load_production

    pm = load_production()                  # models/production/, every file hash checked
    pm.predict({"plate_x": 0.83,            # feet, catcher's view (Statcast convention)
                "plate_z": 3.41,            # feet above the ground
                "batter_height_ft": 6.08,   # the ABS zone is 27%–53.5% of this
                "called_strike": True})     # the umpire's ORIGINAL call
    # -> 0.62   chance ABS overturns the call if it's challenged

    pm.predict_frame(df)                    # many states at once (a DataFrame)
    pm.schema                               # the fixed input schema, field by field
    pm.info                                 # version, model, calibration, scores

    # or, for one-off use, the module-level shortcuts:
    from src.abs_predict import predict
    predict({...})

WHAT "LOCKED" MEANS
    models/production/manifest.json records a SHA-256 hash of every file in the
    bundle. load_production() recomputes them and refuses to load if anything
    changed — so the p(s) your Week 11 policy was solved with is provably the
    one the model card describes. It also compares the installed library
    versions with the ones the bundle was built with and warns on a mismatch.

THE STATE
    Four fields are always required (location + the call). A model that uses
    more than location needs the extra fields listed in `pm.schema`; by
    default a missing one is an error (strict=True), because silently filling
    in "typical" values would make p(s) quietly wrong. Pass strict=False to
    let the model use its training defaults instead.
"""

import hashlib
import json
import os
import pickle
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .abs_calibration import Calibrator
from .abs_features import PITCH_GROUP, _as_bool, nearest_edge
from .abs_zone import zone_distance_in, zone_from_height

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "abs-overturn-pmodel"

# ----------------------------------------------------------------- the schema
# field -> (kind, unit / allowed values, meaning). "core" fields are required by
# every model; the rest only by models whose features need them.
SCHEMA = {
    "plate_x": ("float", "ft", "horizontal location at the plate, catcher's view "
                "(Statcast plate_x; 0 = middle of the plate)"),
    "plate_z": ("float", "ft", "height of the pitch at the plate (Statcast plate_z)"),
    "batter_height_ft": ("float", "ft", "batter's height; the ABS zone runs from 27% to "
                         "53.5% of it. Or pass zone_top_ft and zone_bot_ft instead"),
    "called_strike": ("bool", "True/False", "the umpire's ORIGINAL call: True = called "
                      "strike (the batter may challenge), False = called ball (the "
                      "catcher or pitcher may)"),
    "balls": ("int", "0–3", "balls in the count before the pitch"),
    "strikes": ("int", "0–2", "strikes in the count before the pitch"),
    "outs": ("int", "0–2", "outs when the batter came up (Statcast outs_when_up)"),
    "inning": ("int", "1+", "inning"),
    "leverage_index": ("float", "> 0", "leverage index entering the plate appearance "
                       "(1 = average)"),
    "challenges_left": ("int", "0+", "challenges the deciding side has BEFORE this one"),
    "release_speed": ("float", "mph", "pitch velocity"),
    "pfx_x": ("float", "ft", "horizontal movement (Statcast pfx_x)"),
    "pfx_z": ("float", "ft", "vertical movement (Statcast pfx_z)"),
    "pitch_type": ("str", "FF, SL, CH, ...", "Statcast pitch type"),
    "stand": ("str", "L/R", "batter side"),
    "p_throws": ("str", "L/R", "pitcher hand"),
    "challenger_role": ("str", "batter/catcher/pitcher", "who would challenge"),
    "hp_umpire_id": ("id", "MLBAM id", "home-plate umpire"),
    "fielder_2": ("id", "MLBAM id", "catcher"),
    "pitcher": ("id", "MLBAM id", "pitcher"),
    "batter": ("id", "MLBAM id", "batter"),
}
CORE = ("plate_x", "plate_z", "batter_height_ft", "called_strike")
ALIASES = {"outs_when_up": "outs", "chal_left": "challenges_left"}

# Which raw fields each model feature is built from.
FEATURE_NEEDS = {
    "dist_against_call": CORE, "call_dir": ("called_strike",), "edge": CORE,
    "plate_x": ("plate_x",), "plate_z": ("plate_z",), "batter_height_ft": ("batter_height_ft",),
    "balls": ("balls",), "strikes": ("strikes",), "outs_when_up": ("outs",),
    "inning": ("inning",), "log_leverage": ("leverage_index",),
    "chal_left": ("challenges_left",), "release_speed": ("release_speed",),
    "pfx_x": ("pfx_x",), "pfx_z": ("pfx_z",), "pitch_type": ("pitch_type",),
    "pitch_group": ("pitch_type",), "stand": ("stand",), "p_throws": ("p_throws",),
    "challenger_role": ("challenger_role",), "pitcher_challenge": ("challenger_role",),
    "ump_acc_pp": ("hp_umpire_id",), "catcher_acc_pp": ("fielder_2",),
    "hp_umpire_id": ("hp_umpire_id",), "fielder_2": ("fielder_2",),
    "pitcher": ("pitcher",), "batter": ("batter",),
}
RANGES = {"plate_x": (-3.0, 3.0), "plate_z": (-1.5, 7.0), "batter_height_ft": (4.5, 7.6),
          "balls": (0, 3), "strikes": (0, 2), "outs": (0, 2), "inning": (1, 30),
          "leverage_index": (0.0, 20.0), "challenges_left": (0, 10),
          "release_speed": (40.0, 110.0), "pfx_x": (-4.0, 4.0), "pfx_z": (-4.0, 4.0)}


class StateError(ValueError):
    """A state that doesn't fit the schema (missing field, wrong unit, bad value)."""


# ----------------------------------------------------------------- features
def required_fields(features):
    need = set()
    for f_ in features:
        need |= set(FEATURE_NEEDS.get(f_, (f_,)))
    order = list(SCHEMA)
    return sorted(need, key=lambda x: order.index(x) if x in order else 99)


def _validate(df, need, strict):
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})
    if "batter_height_ft" not in df.columns and {"zone_top_ft", "zone_bot_ft"} <= set(df.columns):
        df["batter_height_ft"] = np.nan                       # zone given directly
    missing = [c for c in need if c not in df.columns]
    if missing and strict:
        raise StateError(f"missing field(s) {missing}. The production model needs: {need}. "
                         "Pass strict=False to use the model's training defaults instead.")
    for c in missing:
        df[c] = np.nan
    for c in ("plate_x", "plate_z", "called_strike"):
        if df[c].isna().any():
            raise StateError(f"'{c}' is required and can't be missing")
    geo = pd.to_numeric(df["batter_height_ft"], errors="coerce").notna()
    if {"zone_top_ft", "zone_bot_ft"} <= set(df.columns):
        geo |= (pd.to_numeric(df["zone_top_ft"], errors="coerce").notna() &
                pd.to_numeric(df["zone_bot_ft"], errors="coerce").notna())
    if not geo.all():
        raise StateError("every state needs batter_height_ft (or zone_top_ft + zone_bot_ft) "
                         "to place the ABS zone")
    for c, (lo, hi) in RANGES.items():
        if c not in df.columns or c not in need:      # only the fields this model reads
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        bad = v.notna() & ((v < lo) | (v > hi))
        if bad.any():
            hint = ""
            if c == "batter_height_ft" and (v[bad] > 40).all():
                hint = " — that looks like INCHES; pass feet (e.g. 6.08)"
            elif c in ("plate_x", "plate_z") and (v[bad].abs() < 80).all():
                hint = " — that looks like INCHES; Statcast locations are in feet"
            raise StateError(f"'{c}' = {v[bad].iloc[0]} is outside [{lo}, {hi}] {SCHEMA[c][1]}"
                             + hint)
    cs = df["called_strike"]
    if not pd.api.types.is_bool_dtype(cs):
        conv = cs.map(_call_value)            # (pandas 3 strings aren't dtype object)
        if conv.isna().any():
            raise StateError("'called_strike' must be True/False (or 'strike'/'ball', 1/0) — "
                             f"got {cs[conv.isna()].iloc[0]!r}")
        cs = conv.astype(bool)
    df["called_strike"] = cs.astype(bool)
    return df


_TRUE = {"true", "1", "1.0", "strike", "s", "called_strike", "yes"}
_FALSE = {"false", "0", "0.0", "ball", "b", "called_ball", "no"}


def _call_value(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)) and v in (0, 1):
        return bool(v)
    t = str(v).strip().lower()
    return True if t in _TRUE else False if t in _FALSE else None


def build_features(state_df, rates=None, league=None):
    """
    Raw states -> the exact columns the models were trained on, computed the
    same way as src/abs_features.add_features (tests/test_predict_interface.py
    checks they agree on every August challenge).
    """
    df = state_df.copy()
    strike = df["called_strike"].to_numpy(bool)
    if {"zone_top_ft", "zone_bot_ft"} <= set(df.columns):
        top = pd.to_numeric(df["zone_top_ft"], errors="coerce").to_numpy(float)
        bot = pd.to_numeric(df["zone_bot_ft"], errors="coerce").to_numpy(float)
        h = pd.to_numeric(df["batter_height_ft"], errors="coerce").to_numpy(float)
        th, bh = zone_from_height(h)
        top = np.where(np.isfinite(top), top, th)
        bot = np.where(np.isfinite(bot), bot, bh)
    else:
        top, bot = zone_from_height(pd.to_numeric(df["batter_height_ft"]).to_numpy(float))
    px = pd.to_numeric(df["plate_x"]).to_numpy(float)
    pz = pd.to_numeric(df["plate_z"]).to_numpy(float)
    df["abs_zone_top"], df["abs_zone_bot"] = top, bot
    df["zone_dist_in"] = zone_distance_in(px, pz, top, bot)
    df["dist_against_call"] = df["zone_dist_in"] * np.where(strike, 1.0, -1.0)
    df["call_dir"] = np.where(strike, -1.0, 1.0)
    df["edge"] = nearest_edge(px, pz, top, bot)
    if "outs" in df.columns:
        df["outs_when_up"] = df["outs"]
    li = pd.to_numeric(df["leverage_index"], errors="coerce") if "leverage_index" in df \
        else pd.Series(np.nan, index=df.index)
    df["log_leverage"] = np.log(li.clip(lower=0.05)).fillna(0.0)
    pt = df["pitch_type"] if "pitch_type" in df.columns else pd.Series(np.nan, index=df.index)
    df["pitch_group"] = pt.map(PITCH_GROUP).fillna("other")
    if "challenges_left" in df.columns:
        df["chal_left"] = pd.to_numeric(df["challenges_left"], errors="coerce").astype(float)
    for c in ("stand", "p_throws", "challenger_role"):
        df[c] = df[c].fillna("unknown").astype(str) if c in df.columns else "unknown"
    df["pitcher_challenge"] = df["challenger_role"].str.lower().eq("pitcher").astype(float)
    rates = rates or {}
    lg = league if league is not None else 0.93
    for col, out in (("hp_umpire_id", "ump_acc_pp"), ("fielder_2", "catcher_acc_pp")):
        table = rates.get(out, {})
        if col in df.columns and table:
            key = df[col].map(_id_key)
            df[out] = 100 * (key.map(table).astype(float).fillna(lg) - lg)
        else:
            df[out] = 0.0
    return df


def states_from_cohort(df):
    """
    Cohort / challenge rows (the Week 5–6 columns) -> the state schema. Passes
    the zone each row was actually judged against (abs_zone_top/bot), which
    covers the few batters with no listed height (Week 5 fell back to
    Statcast's zone for them).
    """
    out = df.copy()
    ren = {"abs_zone_top": "zone_top_ft", "abs_zone_bot": "zone_bot_ft",
           "outs_when_up": "outs", "chal_left": "challenges_left"}
    out = out.drop(columns=[v for v in ren.values() if v in out.columns])
    return out.rename(columns={k: v for k, v in ren.items() if k in out.columns})


def _id_key(v):
    """IDs arrive as 123, 123.0 or "123" — make them one key."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    try:
        f_ = float(v)
        return str(int(f_)) if f_.is_integer() else str(v)
    except (TypeError, ValueError):
        return str(v)


# ----------------------------------------------------------------- the bundle
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "catboost", "lightgbm"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def default_dir():
    return Path(os.environ.get("ABS_MODEL_DIR", ROOT / "models")) / "production"


class ProductionModel:
    """A loaded, verified p(s). Use predict / predict_frame."""

    def __init__(self, folder, manifest, model, calibrator, rates, league):
        self.folder, self.manifest = Path(folder), manifest
        self.model, self.calibrator = model, calibrator
        self.rates, self.league = rates, league
        self.features = manifest["features"]
        self.fields = required_fields(self.features)

    # ------------------------------------------------------------ info
    @property
    def schema(self):
        return {f_: {"kind": SCHEMA[f_][0], "unit": SCHEMA[f_][1], "meaning": SCHEMA[f_][2],
                     "required": True} for f_ in self.fields if f_ in SCHEMA}

    @property
    def info(self):
        keys = ("name", "version", "created", "model_label", "family", "feature_set",
                "calibration", "metrics", "test_window")
        return {k: self.manifest.get(k) for k in keys}

    def __repr__(self):
        m = self.manifest
        return (f"<p(s) {m['version']}: {m['model_label']} + {m['calibration']['method']} "
                f"calibration · needs {', '.join(self.fields)}>")

    # ------------------------------------------------------------ predict
    def predict_frame(self, states, strict=True, raw=False):
        """Probabilities for every row of a DataFrame of states (see `schema`).
        raw=True skips the calibration step (for diagnostics only)."""
        df = states.copy() if isinstance(states, pd.DataFrame) else pd.DataFrame(states)
        df = _validate(df, self.fields, strict)
        X = build_features(df, self.rates, self.league)
        p = np.asarray(self.model.predict(X), float)
        return p if raw else self.calibrator.predict(p)

    def predict(self, state, strict=True):
        """One state (a dict) -> the calibrated probability the call is overturned."""
        if not isinstance(state, dict):
            raise StateError("predict() takes one state as a dict; use predict_frame() for many")
        return float(self.predict_frame(pd.DataFrame([state]), strict=strict)[0])


def write_bundle(folder, *, model, calibrator, rates, league, manifest_extra, model_files):
    """
    Write a production bundle and its manifest (with file hashes). `model_files`
    is a callable that saves the model into folder/"model" and returns nothing.
    Used by scripts/17_week9_calibrate_lock.py; tests use it to build a tiny bundle.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model").mkdir(exist_ok=True)
    model_files(folder / "model")
    (folder / "calibrator.json").write_text(calibrator.to_json())
    rj = {"league": league,
          "ump_acc_pp": {_id_key(k): float(v) for k, v in dict(rates.get("ump_acc_pp", {})).items()},
          "catcher_acc_pp": {_id_key(k): float(v)
                             for k, v in dict(rates.get("catcher_acc_pp", {})).items()}}
    (folder / "participant_rates.json").write_text(json.dumps(rj))
    vers = library_versions()
    (folder / "requirements-lock.txt").write_text(
        "# Library versions this p(s) bundle was built and verified with.\n"
        "# Recreate the environment with:  pip install -r models/production/requirements-lock.txt\n"
        + "".join(f"{('scikit-learn' if k == 'sklearn' else k)}=={v}\n"
                  for k, v in vers.items() if v and k != "python")
        + f"# python {vers['python']}\n")
    files = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "TEST_OPENED.json", "MODEL_CARD.md"):
            files[str(p.relative_to(folder)).replace(os.sep, "/")] = _sha256(p)
    digest = hashlib.sha256("".join(f"{k}:{v}" for k, v in files.items()).encode()).hexdigest()
    manifest = {"name": BUNDLE_NAME, **manifest_extra, "files": files, "versions": vers,
                "content_hash": digest}
    manifest["version"] = f"{manifest_extra.get('date', 'v')}-{digest[:8]}"
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _load_model(folder, family):
    folder = Path(folder)
    if family in ("catboost", "lightgbm", "hgb"):
        from .abs_boosting import Booster
        return Booster.load(folder)
    with open(folder / "model.pkl", "rb") as fh:
        return pickle.load(fh)


def load_production(folder=None, verify=True):
    """Load models/production (or `folder`), checking every file's hash first."""
    folder = Path(folder) if folder else default_dir()
    mpath = folder / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(
            f"No production model at {folder}. Run scripts/17_week9_calibrate_lock.py first.")
    manifest = json.loads(mpath.read_text())
    if verify:
        bad = [rel for rel, h in manifest["files"].items()
               if not (folder / rel).exists() or _sha256(folder / rel) != h]
        if bad:
            raise RuntimeError(f"production bundle {manifest.get('version')} has changed since "
                               f"it was locked: {bad}. Re-lock it (Week 9 script) or restore "
                               "the files — refusing to serve an unverified p(s).")
        now = library_versions()
        diff = {k: (v, now.get(k)) for k, v in manifest.get("versions", {}).items()
                if v and k != "python" and now.get(k) != v}
        if diff:
            warnings.warn("p(s) bundle was built with different library versions: "
                          + ", ".join(f"{k} {a} (now {b})" for k, (a, b) in diff.items())
                          + ". Predictions are re-checked against the stored August scores "
                            "by tests/test_predict_interface.py.", stacklevel=2)
    model = _load_model(folder / "model", manifest["family"])
    cal = Calibrator.from_json((folder / "calibrator.json").read_text())
    rj = json.loads((folder / "participant_rates.json").read_text())
    rates = {k: rj.get(k, {}) for k in ("ump_acc_pp", "catcher_acc_pp")}
    return ProductionModel(folder, manifest, model, cal, rates, rj.get("league"))


_DEFAULT = {}


def predict(state, strict=True):
    """Module-level shortcut: the default production bundle, loaded once."""
    if "pm" not in _DEFAULT:
        _DEFAULT["pm"] = load_production()
    return _DEFAULT["pm"].predict(state, strict=strict)


def predict_frame(states, strict=True):
    if "pm" not in _DEFAULT:
        _DEFAULT["pm"] = load_production()
    return _DEFAULT["pm"].predict_frame(states, strict=strict)


__all__ = ["SCHEMA", "CORE", "StateError", "build_features", "required_fields",
           "states_from_cohort",
           "ProductionModel", "write_bundle", "load_production", "predict", "predict_frame",
           "library_versions"]
