"""
abs_pipeline.py — load the modelling data the same way in Weeks 8 and 9.

    data = load_challenges(DATA_DIR, CONFIG)               # train + August
    data = load_challenges(DATA_DIR, CONFIG, include_test=True)   # Week 9 --final-test ONLY

The September test window comes back as None unless include_test=True. Only
one place in the project asks for it, which makes an accidental peek hard.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .abs_features import (BLOCKS, BOOST_SETS, add_features, challenge_rows, load_cohort,
                           participant_rates, split_masks)


def _ensure_columns(df):
    """Every column a model asks for must exist; missing ones become all-NaN."""
    missing = []
    wanted = set()
    for b in BLOCKS.values():
        wanted |= set(b.get("numeric", []) + b.get("categorical", []))
    for s in BOOST_SETS.values():
        wanted |= set(s["numeric"] + s["categorical"])
    for c in sorted(wanted):
        if c not in df.columns:
            df[c] = np.nan
            missing.append(c)
    return df, missing


def cohort_fingerprint(df):
    """
    The Week 6 fingerprint, recomputed from the cohort itself: a hash of every
    row's (game, at-bat, pitch) key, in order — exactly how
    scripts/13_define_cohort.py computes the one it stores in config/cohort.json.
    """
    keys = df[["game_pk", "at_bat_number", "pitch_number"]]
    return hashlib.sha256(pd.util.hash_pandas_object(keys, index=False)
                          .values.tobytes()).hexdigest()[:16]


def resolve_fingerprint(raw, config_dir):
    """
    (fingerprint, note). Uses the one Week 6 stored in config/cohort.json when it
    exists and matches the cohort on disk; otherwise the one computed from the
    cohort, with a note saying why — so the "same cohort as last week" checks in
    Weeks 8 and 9 always compare something real.
    """
    computed = cohort_fingerprint(raw)
    cj = Path(config_dir) / "cohort.json"
    stored = json.loads(cj.read_text()).get("cohort_fingerprint") if cj.exists() else None
    if stored is None:
        return computed, (f"no {cj} — fingerprint computed from the cohort file; copy your "
                          "Week 6 config/cohort.json here to record the frozen one")
    if stored != computed:
        return computed, (f"WARNING: the cohort file on disk ({computed}) is not the one Week 6 "
                          f"froze ({stored}) — it changed after the freeze")
    return stored, ""


def load_challenges(data_dir, config_dir, include_test=False):
    raw = load_cohort(data_dir)
    fingerprint, fp_note = resolve_fingerprint(raw, config_dir)
    df, call_check = add_features(raw, return_check=True)
    tr_mask, _, _ = split_masks(df)
    df, league, rates = participant_rates(df, tr_mask)
    df, missing = _ensure_columns(df)
    ch = challenge_rows(df)
    out = {
        "tr": ch[ch["split"] == "train"].reset_index(drop=True),
        "va": ch[ch["split"] == "validate"].reset_index(drop=True),
        "n_test": int((ch["split"] == "test").sum()),
        "te": ch[ch["split"] == "test"].reset_index(drop=True) if include_test else None,
        "call_check": call_check, "league": league, "rates": rates,
        "fingerprint": fingerprint, "fingerprint_note": fp_note, "missing_cols": missing,
        "n_called": int(len(df)),
    }
    return out


def es_mask_by_date(tr, frac=0.15):
    """
    The early-stopping rows inside the training window: the last `frac` of its
    GAMES in date order. Temporal, like every other split in the project — the
    boosted models learn when to stop from games that come after the ones they
    were fitted on.
    """
    g = tr.groupby("game_pk")["game_date"].min().sort_values()
    cut = int(np.floor(len(g) * (1 - frac)))
    late = set(g.index[cut:])
    return tr["game_pk"].isin(late).to_numpy()


def grouped_folds(groups, k=5, seed=2026):
    """K folds that keep every game whole, assigned by a seeded shuffle (the
    same groups give the same folds on every machine)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = dict(zip(uniq[order], np.arange(len(uniq)) % k))
    f = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]
