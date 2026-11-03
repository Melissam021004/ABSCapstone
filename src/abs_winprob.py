"""
abs_winprob.py — Week 10. V(s): the probability the home team wins from a game state.

V(s) is the other half of the challenge decision. p(s) (Week 9) says how likely
the call is to be overturned; V says what the call is WORTH — the win
probability if it stands versus if it's overturned. Week 11's MDP compares them.

THREE VERSIONS, ONE DECISION (the plan's steps 2–4)

    empirical   for every state, the share of games the home team went on to win
                (EmpiricalWE). The ground truth where there is data; noise where
                there isn't (down 7 in the 3rd with the bases loaded and a 3-1 count).
    boosted     gradient boosting on state -> win (the plan's "model version"),
                built with Week 8's Booster and a monotone constraint (a bigger home
                lead can never lower the home team's chances).
    structural  a Markov model of the game (RunModel + MarkovWP) — the method
                behind the published win-expectancy tables. It learns how
                plate appearances move the outs, runners and score from ~180,000
                plate appearances a season, then COMPUTES the chance of winning by
                working back from the end of the game. It never sees who won.

Why the structural model is the one Week 11 uses (decided before looking):
the decision needs V at COUNTERFACTUAL states — the overturned branch of a 3-2
pitch with the bases loaded in the 8th — and it needs the effect of ONE pitch,
which is a percentage point or two. A model trained on game outcomes learns from
~2,430 independent results a season; a pitch's effect is too small to learn from
that. The structural model learns the same effect from every plate appearance.
The empirical table and the boosted model are how we check it — on calibration,
agreement in well-populated cells and the published sanity values.

THE STRUCTURAL MODEL IN THREE STEPS

  1. From every pitch with count c in outs/bases (o, b), record how its plate
     appearance ended: outs and bases afterward, runs scored. That table, smoothed
     where it's thin, is the plate appearance "transition" T (RunModel).
  2. Chain plate appearances until the third out: the distribution of runs the
     batting team scores in the rest of the half-inning from any (o, b, c). Its
     mean is run expectancy (RE24 at count 0-0, RE288 with the count).
  3. Work back from the end of the game (MarkovWP): the bottom of the 9th or later
     ends the moment the home team leads; the bottom of the 9th isn't played if
     the home team already leads; a tie after 9 goes to extra innings, which start
     with a runner on second (2026 rule) and repeat until someone wins. Home and
     away batting use their own transitions, which is where home-field advantage
     comes from.

THE CALL TRANSITIONS (the plan's step 5, CallTransitions)

    For a called pitch in (outs, bases, count) — where does the game go next if the
    call is a BALL, and where if it's a STRIKE? Read off the data: the count moves,
    ball four is a walk (forced runners move up, a run scores with the bases loaded),
    strike three is an out, and occasionally a runner steals or a wild pitch moves
    everyone up on the same pitch. "Stands" uses the umpire's call; "overturned" uses
    the other one.

Everything is saved as plain numbers (NumPy arrays + JSON), hashed in a manifest,
and loaded through `load_winprob()` — the same pattern as Week 9's p(s).

    from src.abs_winprob import load_winprob
    wp = load_winprob()
    wp.home({"inning": 9, "half": "bot", "outs": 1, "on_3b": True,
             "home_score": 3, "away_score": 3})                  # ≈ 0.83
    wp.call_value({... a called strike ..., "balls": 1, "strikes": 2}, call="strike")
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .abs_calibration import Calibrator
from .abs_gamestate import (BASE_NAMES, CLASSES, COUNT_NAMES, COUNTS, N_BASES, N_COUNTS,
                            N_OUTS, bases_code, count_index)

ROOT = Path(__file__).resolve().parents[1]
N_INN = 10              # innings 1..9 and "extras" (every extra inning is alike)
DMAX = 30               # home lead from −30 to +30
RMAX = 20               # runs in a half-inning (the far tail is lumped at 20)
RPA = 6                 # runs in one plate appearance / one pitch (lumped at 6)
N_END = 4 * N_BASES * (RPA + 1)                 # end of a PA: outs' (0-2, 3=over) × bases' × runs
N_NEXT = 4 * N_BASES * N_COUNTS * (RPA + 1)     # next pitch: outs' × bases' × count' × runs
M_STRUCT = 25.0         # pseudo-count: a thin (outs, bases, count) cell borrows from the prior
M_SIDE = 200.0          # pseudo-count: home and away cells borrow from the pooled cell
M_CALL = 5.0            # pseudo-count: a thin call cell borrows from the rulebook transition
EXTRA_START = 2         # bases code for "runner on second" — the automatic runner
BUNDLE_NAME = "V(s) home win probability"
EPS = 1e-6


class WPStateError(ValueError):
    """A game state that doesn't fit the schema."""


# =================================================================== helpers
def _end_index(eo, eb, r):
    eo = np.minimum(np.asarray(eo, int), 3)
    eb = np.where(eo >= 3, 0, np.asarray(eb, int))
    return (eo * N_BASES + eb) * (RPA + 1) + np.clip(np.asarray(r, int), 0, RPA)


def _end_decode(e):
    r = e % (RPA + 1)
    eb = (e // (RPA + 1)) % N_BASES
    eo = e // ((RPA + 1) * N_BASES)
    return int(eo), int(eb), int(r)


def _next_index(eo, eb, ec, r):
    eo = np.minimum(np.asarray(eo, int), 3)
    eb = np.where(eo >= 3, 0, np.asarray(eb, int))
    ec = np.where(eo >= 3, 0, np.asarray(ec, int))
    return ((eo * N_BASES + eb) * N_COUNTS + ec) * (RPA + 1) + np.clip(np.asarray(r, int), 0, RPA)


def _next_decode(e):
    r = e % (RPA + 1)
    ec = (e // (RPA + 1)) % N_COUNTS
    eb = (e // ((RPA + 1) * N_COUNTS)) % N_BASES
    eo = e // ((RPA + 1) * N_COUNTS * N_BASES)
    return int(eo), int(eb), int(ec), int(r)


def _shift(v, r):
    """Distribution over runs (last axis) shifted up by r runs; overflow lumped at RMAX."""
    if r == 0:
        return v
    out = np.zeros_like(v)
    out[..., r:] = v[..., :-r]
    out[..., -1] += v[..., -r:].sum(axis=-1)
    return out


def _normalise(a, axis=-1):
    s = a.sum(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, a / np.where(s > 0, s, 1), 0.0)


# =================================================================== run model
class RunModel:
    """
    Plate-appearance transitions -> runs to the end of the half-inning.

    T[side, o, b, c, e]: from a pitch with count c in outs/bases (o, b), the chance
    the plate appearance ends in outcome e = (outs', bases', runs). side 0 = the
    away team batting (top), 1 = the home team (bottom).

    Thin cells are smoothed toward a STRUCTURAL PRIOR: P(how the PA ends | count) ×
    P(where runners end up | outs, bases, how it ended) — e.g. "from 3-1, 21% of
    plate appearances end in a walk" × "a walk with runners on 1st and 2nd loads the
    bases". Both pieces are estimated from thousands of plate appearances, so the
    prior is solid even where a specific (outs, bases, count) cell is rare. Home and
    away cells are then pulled gently toward the pooled cell.
    """

    def fit(self, pitches):
        p = pitches
        side = p["half"].to_numpy(int)
        o, b, c = p["outs"].to_numpy(int), p["bases"].to_numpy(int), p["count"].to_numpy(int)
        e = _end_index(p["pa_end_outs"], p["pa_end_bases"], p["runs_rest_pa"])
        kc = p["pa_class"].fillna("OUT").map({k: i for i, k in enumerate(CLASSES)})
        k = kc.fillna(CLASSES.index("OUT")).to_numpy(int)
        N = np.zeros((2, N_OUTS, N_BASES, N_COUNTS, N_END))
        np.add.at(N, (side, o, b, c, e), 1.0)
        Nkc = np.zeros((N_COUNTS, len(CLASSES)))
        np.add.at(Nkc, (c, k), 1.0)
        Nobke = np.zeros((N_OUTS, N_BASES, len(CLASSES), N_END))
        np.add.at(Nobke, (o, b, k, e), 1.0)
        Pkc = _normalise(Nkc)
        Pobke = _normalise(Nobke)
        Q = _normalise(np.einsum("ck,obke->obce", Pkc, Pobke))
        # a cell the prior can't speak to either (never happens with real data)
        Q = np.where(Q.sum(-1, keepdims=True) > 0, Q,
                     _normalise(N.sum(axis=(0, 3), keepdims=False))[:, :, None, :])
        Npool = N.sum(0)
        P = (Npool + M_STRUCT * Q) / (Npool.sum(-1, keepdims=True) + M_STRUCT)
        self.T = (N + M_SIDE * P[None]) / (N.sum(-1, keepdims=True) + M_SIDE)
        self.n_cell = N.sum(-1)
        self.n_fit = int(len(p))
        self._solve()
        return self

    def _propagate(self, T, Z):
        """Run distribution to the end of the inning, given the one for PA starts (Z)."""
        shp = T.shape
        Tf = T.reshape(2, -1, N_END)
        out = np.zeros((2, Tf.shape[1], RMAX + 1))
        for e in np.flatnonzero(Tf.reshape(-1, N_END).sum(0) > 0):
            eo, eb, r = _end_decode(e)
            w = Tf[:, :, e]
            if eo >= 3:
                out[:, :, min(r, RMAX)] += w
            else:
                out += w[:, :, None] * _shift(Z[:, eo, eb, :], r)[:, None, :]
        return out.reshape(shp[:-1] + (RMAX + 1,))

    def _solve(self, tol=1e-13, max_iter=1000):
        T0 = self.T[:, :, :, 0, :]                       # plate-appearance starts (0-0)
        Z = np.zeros((2, N_OUTS, N_BASES, RMAX + 1))
        Z[..., 0] = 1.0
        for it in range(max_iter):
            Zn = self._propagate(T0, Z)
            if np.max(np.abs(Zn - Z)) < tol:
                Z = Zn
                break
            Z = Zn
        self.iterations = it + 1
        self.Z = Z                                        # (side, outs, bases, runs)
        self.R = self._propagate(self.T, Z)               # (side, outs, bases, count, runs)
        r = np.arange(RMAX + 1)
        self.RE = (self.R * r).sum(-1)                    # RE288 by side
        return self

    # ---------------------------------------------------------- summaries
    def re_table(self, side=None, count=0):
        """Expected runs to the end of the inning, 8 base states × 3 outs (RE24 at 0-0)."""
        if side is None:
            w = self.n_cell[:, :, :, count].sum((1, 2))
            R = (self.RE[:, :, :, count] * (w / w.sum())[:, None, None]).sum(0)
        else:
            R = self.RE[side, :, :, count]
        return pd.DataFrame(R.T, index=BASE_NAMES, columns=[f"{o} out" for o in range(3)])

    def score_prob_table(self, side=None, count=0):
        """Chance of scoring at least one more run this inning."""
        P1 = 1 - self.R[..., 0]
        if side is None:
            w = self.n_cell[:, :, :, count].sum((1, 2))
            P = (P1[:, :, :, count] * (w / w.sum())[:, None, None]).sum(0)
        else:
            P = P1[side, :, :, count]
        return pd.DataFrame(P.T, index=BASE_NAMES, columns=[f"{o} out" for o in range(3)])

    def to_arrays(self):
        return {"T": self.T.astype(np.float64), "Z": self.Z, "R": self.R, "RE": self.RE,
                "n_cell": self.n_cell}

    @classmethod
    def from_arrays(cls, a):
        m = cls()
        m.T, m.Z, m.R, m.RE, m.n_cell = a["T"], a["Z"], a["R"], a["RE"], a["n_cell"]
        return m


# =================================================================== call transitions
class CallTransitions:
    """
    For a CALLED pitch in (outs, bases, count): the distribution of the next state
    — outs', bases', count' and runs on the way — if the call is a ball (k=0) or a
    strike (k=1). Estimated from the effective call (what the count actually did).
    Thin cells borrow from the rulebook transition (the count moves; ball four is a
    walk; strike three is an out), so a rare cell still behaves like baseball.
    """

    @staticmethod
    def rulebook(o, b, c, k):
        """The no-surprises next state for call k: (outs', bases', count', runs)."""
        balls, strikes = COUNTS[c]
        if k == 1:                                       # strike
            if strikes < 2:
                return o, b, 3 * balls + strikes + 1, 0
            return (o + 1, b, 0, 0) if o + 1 < 3 else (3, 0, 0, 0)
        if balls < 3:                                    # ball
            return o, b, 3 * (balls + 1) + strikes, 0
        on1, on2, on3 = b & 1, (b >> 1) & 1, (b >> 2) & 1   # walk: forced runners move up
        runs = 0
        if on1:
            if on2:
                if on3:
                    runs = 1
                on3 = 1
            on2 = 1
        on1 = 1
        return o, on1 + 2 * on2 + 4 * on3, 0, runs

    def fit(self, pitches):
        cp = pitches[pitches["is_called"] & pitches["call_effective"].notna()]
        o, b, c = cp["outs"].to_numpy(int), cp["bases"].to_numpy(int), cp["count"].to_numpy(int)
        k = cp["call_effective"].to_numpy(int)
        nxt = _next_index(cp["next_outs"], cp["next_bases"], cp["next_count"], cp["runs_next"])
        N = np.zeros((N_OUTS, N_BASES, N_COUNTS, 2, N_NEXT), np.float64)
        np.add.at(N, (o, b, c, k, nxt), 1.0)
        prior = np.zeros_like(N)
        for oo in range(N_OUTS):
            for bb in range(N_BASES):
                for cc in range(N_COUNTS):
                    for kk in (0, 1):
                        eo, eb, ec, r = self.rulebook(oo, bb, cc, kk)
                        prior[oo, bb, cc, kk, int(_next_index(eo, eb, ec, r))] = 1.0
        n = N.sum(-1, keepdims=True)
        self.P = ((N + M_CALL * prior) / (n + M_CALL)).astype(np.float32)
        self.n = n[..., 0]
        self.n_fit = int(len(cp))
        self._index()
        return self

    def _index(self):
        """Non-zero next states per cell, for fast lookups."""
        self.cells = {}
        flat = self.P.reshape(-1, N_NEXT)
        for i, row in enumerate(flat):
            nz = np.flatnonzero(row > 1e-7)
            dec = np.array([_next_decode(e) for e in nz], int).reshape(-1, 4)
            self.cells[i] = (dec, row[nz].astype(float) / row[nz].sum())
        return self

    def cell_id(self, o, b, c, k):
        return ((np.asarray(o) * N_BASES + np.asarray(b)) * N_COUNTS + np.asarray(c)) * 2 + np.asarray(k)

    def table(self):
        """One row per (outs, bases, count, call) with the share of each kind of next state."""
        rows = []
        for o in range(N_OUTS):
            for b in range(N_BASES):
                for c in range(N_COUNTS):
                    for k in (0, 1):
                        dec, w = self.cells[int(self.cell_id(o, b, c, k))]
                        ro = self.rulebook(o, b, c, k)
                        rb = (dec[:, 0] == ro[0]) & (dec[:, 1] == ro[1]) & (dec[:, 2] == ro[2]) \
                            & (dec[:, 3] == ro[3])
                        rows.append({"outs": o, "bases": BASE_NAMES[b], "count": COUNT_NAMES[c],
                                     "call": "strike" if k else "ball",
                                     "n": int(self.n[o, b, c, k]),
                                     "rulebook_next": _describe_next(o, b, c, k, ro),
                                     "p_rulebook": float(w[rb].sum()),
                                     "p_other": float(1 - w[rb].sum()),
                                     "exp_runs_on_pitch": float((w * dec[:, 3]).sum())})
        return pd.DataFrame(rows)

    def to_arrays(self):
        return {"P": self.P, "n": self.n}

    @classmethod
    def from_arrays(cls, a):
        m = cls()
        m.P, m.n = a["P"].astype(np.float32), a["n"]
        return m._index()


def _describe_next(o, b, c, k, nxt):
    eo, eb, ec, r = nxt
    if k == 1 and COUNTS[c][1] == 2:
        return "strikeout" + (" — inning over" if eo == 3 else f" ({eo} out)")
    if k == 0 and COUNTS[c][0] == 3:
        return "walk" + (f", {r} run scores" if r else f" -> {BASE_NAMES[eb]}")
    return f"count {COUNT_NAMES[ec]}"


# =================================================================== Markov win probability
class MarkovWP:
    """
    Home win probability for every state, computed by working back from the end of
    the game with the run distributions of a RunModel. See the module docstring.

    v_table[i, h, o, b, c, d + DMAX]   i = inning 1..9 -> 0..8, extras -> 9; h = 0 top / 1 bottom
    end_value[i, h, d + DMAX + RMAX]   home win probability when that half-inning ends
                                       with home lead d (wider range: runs may push it out)
    """

    def __init__(self, run_model):
        self.rm = run_model
        Z = run_model.Z
        f_top, f_bot = Z[0, 0, 0], Z[1, 0, 0]                      # start of a regular half
        fx_top, fx_bot = Z[0, 0, EXTRA_START], Z[1, 0, EXTRA_START]  # extras: runner on 2nd
        D = np.arange(-DMAX, DMAX + 1)
        Dx = np.arange(-DMAX - RMAX, DMAX + RMAX + 1)
        runs = np.arange(RMAX + 1)

        def gt(f, k):        # P(runs > k), k may be negative
            k = np.asarray(k)
            cdf = np.concatenate([[0.0], np.cumsum(f)])
            return np.where(k < 0, 1.0, 1.0 - cdf[np.clip(k + 1, 0, RMAX + 1)])

        def eq(f, k):
            k = np.asarray(k)
            return np.where((k >= 0) & (k <= RMAX), f[np.clip(k, 0, RMAX)], 0.0)

        # ---- extras: x = P(home wins | top of an extra inning, tied)
        A = float((fx_top * gt(fx_bot, runs)).sum())
        B = float((fx_top * eq(fx_bot, runs)).sum())
        x = A / (1.0 - B)
        self.x_extra = x

        def clamp_lookup(arr, d):
            return arr[np.clip(d, -DMAX, DMAX) + DMAX]

        # ---- the end of each half-inning, as a function of the home lead d
        def end_bot_last(d):          # bottom of the 9th or later
            d = np.asarray(d)
            return np.where(d > 0, 1.0, np.where(d < 0, 0.0, x))

        # G_bot9[d]: start of the bottom of the 9th (only d <= 0 is reachable)
        def g_bot_last(f, d):
            d = np.asarray(d)
            return np.where(d > 0, 1.0, gt(f, -d) + x * eq(f, -d))

        G_bot = np.zeros((10, len(D)))        # index i-1 for innings 1..9; [9] = extras
        G_top = np.zeros((10, len(D)))
        G_bot[8] = g_bot_last(f_bot, D)
        G_bot[9] = g_bot_last(fx_bot, D)
        G_top[8] = np.array([(f_top * np.where(d - runs > 0, 1.0,
                                               clamp_lookup(G_bot[8], d - runs))).sum() for d in D])
        G_top[9] = np.array([(fx_top * np.where(d - runs > 0, 1.0,
                                                clamp_lookup(G_bot[9], d - runs))).sum() for d in D])
        for i in range(7, -1, -1):                       # innings 8 down to 1
            G_bot[i] = np.array([(f_bot * clamp_lookup(G_top[i + 1], d + runs)).sum() for d in D])
            G_top[i] = np.array([(f_top * clamp_lookup(G_bot[i], d - runs)).sum() for d in D])
        self.G_top, self.G_bot = G_top, G_bot

        # ---- end_value[i, h, d]: what the end of the half-inning is worth
        E = np.zeros((N_INN, 2, len(Dx)))
        for i in range(N_INN):
            # top ends -> the bottom half starts (or the game is over)
            if i >= 8:
                E[i, 0] = np.where(Dx > 0, 1.0, clamp_lookup(G_bot[i], Dx))
            else:
                E[i, 0] = clamp_lookup(G_bot[i], Dx)
            # bottom ends -> next inning starts (or the game is over / extras)
            if i >= 8:
                E[i, 1] = end_bot_last(Dx)
            else:
                E[i, 1] = clamp_lookup(G_top[i + 1], Dx)
        self.end_value = E

        # ---- v_table: every in-inning state
        R = run_model.R                                    # (side, o, b, c, runs)
        V = np.zeros((N_INN, 2, N_OUTS, N_BASES, N_COUNTS, len(D)))
        for h in (0, 1):
            pos = (D[:, None] + (-1 if h == 0 else 1) * runs[None, :]) + DMAX + RMAX   # (d, r)
            Rh = R[h]                                       # (o, b, c, runs)
            for i in range(N_INN):
                Ev = E[i, h][pos]                           # (d, r)
                V[i, h] = np.einsum("obcr,dr->obcd", Rh, Ev)
        self.v_table = V
        self.pregame = float(V[0, 0, 0, 0, 0, DMAX])


# =================================================================== state handling
_HALF = {"top": 0, "t": 0, "bot": 1, "bottom": 1, "b": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1,
         "false": 0, "true": 1}


def parse_states(states):
    """
    A dict or DataFrame of game states -> integer arrays (inning index, half, outs,
    bases, count, home lead). Accepts Statcast's own column names.

        inning                     1, 2, ... (10+ = extras)
        half / inning_topbot       "top"/"bot" (or "Top"/"Bot")
        outs / outs_when_up        0, 1, 2
        on_1b, on_2b, on_3b        True/False, or a runner id / empty (or `bases` 0..7)
        home_score + away_score    or `score_diff` (home − away)
        balls, strikes             optional (default 0-0, the start of a plate appearance)
    """
    df = pd.DataFrame([states]) if isinstance(states, dict) else pd.DataFrame(states).copy()
    if "half" not in df.columns:
        if "inning_topbot" in df.columns:
            df["half"] = df["inning_topbot"]
        else:
            raise WPStateError("need `half` ('top'/'bot') or `inning_topbot`")
    if "outs" not in df.columns and "outs_when_up" in df.columns:
        df["outs"] = df["outs_when_up"]
    for need in ("inning", "outs"):
        if need not in df.columns:
            raise WPStateError(f"missing field `{need}`")
    half = df["half"].map(lambda v: _HALF.get(str(v).strip().lower()) if not isinstance(v, (bool, np.bool_))
                          else int(v))
    if half.isna().any():
        raise WPStateError(f"`half` must be 'top' or 'bot' — got {df['half'][half.isna()].iloc[0]!r}")
    inning = pd.to_numeric(df["inning"], errors="coerce")
    if inning.isna().any() or (inning < 1).any():
        raise WPStateError("`inning` must be 1 or more")
    outs = pd.to_numeric(df["outs"], errors="coerce")
    if outs.isna().any() or (~outs.isin([0, 1, 2])).any():
        raise WPStateError("`outs` must be 0, 1 or 2 (a state with 3 outs is the next half-inning)")
    if "bases" in df.columns:
        bases = pd.to_numeric(df["bases"], errors="coerce")
        if bases.isna().any() or (~bases.between(0, 7)).any():
            raise WPStateError("`bases` must be 0..7")
        bases = bases.to_numpy(int)
    else:
        cols = [df[c] if c in df.columns else pd.Series([np.nan] * len(df)) for c in ("on_1b", "on_2b", "on_3b")]
        cols = [c.map(lambda v: v if not isinstance(v, (bool, np.bool_)) else (1 if v else np.nan))
                for c in cols]
        bases = bases_code(*cols)
    if "score_diff" in df.columns:
        diff = pd.to_numeric(df["score_diff"], errors="coerce")
    elif {"home_score", "away_score"} <= set(df.columns):
        diff = pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(df["away_score"], errors="coerce")
    else:
        raise WPStateError("need `home_score` and `away_score` (or `score_diff` = home − away)")
    if diff.isna().any():
        raise WPStateError("scores can't be missing")
    balls = pd.to_numeric(df["balls"], errors="coerce") if "balls" in df.columns else pd.Series(0, index=df.index)
    strikes = pd.to_numeric(df["strikes"], errors="coerce") if "strikes" in df.columns else pd.Series(0, index=df.index)
    if (~balls.isin([0, 1, 2, 3])).any() or (~strikes.isin([0, 1, 2])).any():
        raise WPStateError("`balls` must be 0-3 and `strikes` 0-2 (the count BEFORE the pitch)")
    return {"i": np.minimum(inning.to_numpy(int), N_INN) - 1, "h": half.to_numpy(int),
            "o": outs.to_numpy(int), "b": np.asarray(bases, int),
            "c": count_index(balls, strikes), "d": diff.to_numpy(int)}


# =================================================================== evaluation
def _w_logistic(x, y, w, offset=None, slope=True, iters=25):
    """Weighted logistic fit of y on x (slope) or intercept-only with an offset."""
    if slope:
        X = np.column_stack([np.ones_like(x), x])
        beta = np.array([0.0, 1.0])
    else:
        X = np.ones((len(x), 1))
        beta = np.array([0.0])
    off = np.zeros_like(x) if offset is None else offset
    for _ in range(iters):
        mu = expit(X @ beta + off)
        g = X.T @ (w * (y - mu))
        H = X.T @ (X * (w * mu * (1 - mu))[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def calibration_pair(y, p, w=None):
    """(intercept with slope fixed at 1, slope) — Week 7's definitions, with weights."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    lp = np.log(p / (1 - p))
    y = np.asarray(y, float)
    w = np.ones_like(lp) if w is None else w
    a = _w_logistic(lp, y, w, offset=lp, slope=False)[0]
    b = _w_logistic(lp, y, w, slope=True)[1]
    return float(a), float(b)


def game_bootstrap(y, preds, groups, B=500, seed=0, calibration=True):
    """
    Week 7's game-level bootstrap, made fast for hundreds of thousands of rows:
    Brier and log loss are sums over games, so a resample is a reweighting of
    per-game totals; the calibration intercept and slope are refitted with the same
    game weights. Every model is scored on the same resamples (paired differences).

    Returns point {model: {metric: value}}, ci {model: {metric: (lo, hi)}},
    diffs {(a, b): {metric: (mean, lo, hi)}} for brier and log_loss.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = len(uniq)
    n_g = np.bincount(inv, minlength=G).astype(float)
    names = list(preds)
    P = {k: np.clip(np.asarray(v, float), EPS, 1 - EPS) for k, v in preds.items()}
    sums = {}
    for k in names:
        se = (P[k] - y) ** 2
        ll = -(y * np.log(P[k]) + (1 - y) * np.log(1 - P[k]))
        sums[k] = (np.bincount(inv, se, G), np.bincount(inv, ll, G))
    point = {}
    for k in names:
        a, b = calibration_pair(y, P[k]) if calibration else (np.nan, np.nan)
        point[k] = {"brier": float(sums[k][0].sum() / n_g.sum()),
                    "log_loss": float(sums[k][1].sum() / n_g.sum()),
                    "cal_intercept": a, "cal_slope": b}
    draws = {k: {"brier": [], "log_loss": [], "cal_intercept": [], "cal_slope": []} for k in names}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        n = (cnt * n_g).sum()
        w_rows = cnt[inv] if calibration else None
        for k in names:
            draws[k]["brier"].append((cnt * sums[k][0]).sum() / n)
            draws[k]["log_loss"].append((cnt * sums[k][1]).sum() / n)
            if calibration:
                a, b = calibration_pair(y, P[k], w_rows)
                draws[k]["cal_intercept"].append(a)
                draws[k]["cal_slope"].append(b)

    def pct(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

    ci = {k: {m: pct(draws[k][m]) for m in draws[k]} for k in names}
    diffs = {}
    for a_ in names:
        for b_ in names:
            if a_ == b_:
                continue
            diffs[(a_, b_)] = {}
            for m in ("brier", "log_loss"):
                d = np.array(draws[a_][m]) - np.array(draws[b_][m])
                diffs[(a_, b_)][m] = (float(d.mean()),) + pct(d)
    return point, ci, diffs


# =================================================================== empirical table
EMP_DIFF = 10           # the empirical table clips the home lead to −10..+10
WELL_POPULATED = 100    # a cell needs this many GAMES before its rate is "well measured"
THIN = 30               # fewer games than this and a cell is flagged as thin


def state_columns(rows):
    """Integer state columns used by the tables and the boosted model."""
    out = pd.DataFrame(index=rows.index)
    out["inning_c"] = np.minimum(rows["inning"].astype(int), N_INN)
    out["half"] = rows["half"].astype(int)
    out["outs"] = rows["outs"].astype(int)
    b = rows["bases"].to_numpy(int)
    out["on1"], out["on2"], out["on3"] = b & 1, (b >> 1) & 1, (b >> 2) & 1
    out["bases"] = b
    out["diff_c"] = rows["diff"].clip(-EMP_DIFF, EMP_DIFF).astype(int)
    out["balls"], out["strikes"] = rows["balls"].astype(int), rows["strikes"].astype(int)
    return out


def empirical_table(rows, with_count=False):
    """
    The plan's step 2: for every state, the share of times the HOME team went on to
    win — with the number of plate appearances (or pitches) and, more honestly, the
    number of distinct GAMES behind each cell (rows from one game share one outcome).
    """
    st = state_columns(rows)
    st["home_win"] = rows["home_win"].to_numpy()
    st["game_pk"] = rows["game_pk"].to_numpy()
    key = ["inning_c", "half", "outs", "bases", "diff_c"] + (["balls", "strikes"] if with_count else [])
    t = st.groupby(key).agg(n=("home_win", "size"), n_games=("game_pk", "nunique"),
                            wins=("home_win", "sum")).reset_index()
    t["rate"] = t["wins"] / t["n"]
    t["thin"] = t["n_games"] < THIN
    t["well_populated"] = t["n_games"] >= WELL_POPULATED
    return t


def agreement(table, model_col, min_games=WELL_POPULATED):
    """
    Empirical rate vs a model's average prediction, cell by cell. In a cell with n
    games the empirical rate has a standard error of about sqrt(p(1-p)/n), so if the
    model is right about 5% of well-populated cells should miss by more than 2 SE.
    """
    t = table[table["n_games"] >= min_games].copy()
    p = t[model_col].clip(1e-4, 1 - 1e-4)
    t["se"] = np.sqrt(p * (1 - p) / t["n_games"])
    t["z"] = (t["rate"] - p) / t["se"]
    w = t["n"] / t["n"].sum()
    return t, {"cells": int(len(t)), "share_rows": float(t["n"].sum() / table["n"].sum()),
               "mae": float((w * (t["rate"] - p).abs()).sum()),
               "share_beyond_2se": float((t["z"].abs() > 2).mean()) if len(t) else np.nan}


# =================================================================== boosted version
GBM_FEATURES = ["inning_c", "half", "outs", "on1", "on2", "on3", "diff_c", "balls", "strikes"]
GBM_PARAMS = {
    "lightgbm": {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 400,
                 "subsample": 0.8, "colsample_bytree": 1.0, "reg_lambda": 5},
    "catboost": {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10},
    "hgb": {"max_leaf_nodes": 31, "learning_rate": 0.05, "min_samples_leaf": 400,
            "l2_regularization": 5},
}


def fit_gbm(rows, backend, seed=2026, max_rows=None):
    """
    Gradient boosting on state -> home win (the plan's "model version"). One fixed,
    sensible setting per library (no tuning: Week 10 isn't a boosting week), early
    stopping on the last 15% of the training games by date, and a monotone
    constraint: a bigger home lead can never lower the home team's chances.
    """
    from .abs_boosting import Booster
    from .abs_pipeline import es_mask_by_date
    rows = rows
    if max_rows and len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=seed).sort_index()
    X = state_columns(rows)
    X["game_pk"], X["game_date"] = rows["game_pk"].to_numpy(), rows["game_date"].to_numpy()
    es = es_mask_by_date(X)
    m = Booster(backend, numeric=GBM_FEATURES, categorical=[], params=GBM_PARAMS[backend],
                seed=seed, max_rounds=3000, patience=100, monotone=True,
                monotone_up=("diff_c",))
    m.fit(X, rows["home_win"].to_numpy(int), es)
    return m


def gbm_predict(model, rows):
    return model.predict(state_columns(rows))


# =================================================================== sanity checks
# Published win-expectancy values (Tango's tables, FanGraphs) put these states in
# these ranges for a modern run environment (~4.5 runs per team per game). The
# windows are generous on purpose: they catch a broken model, not a run environment
# that differs a little from the 2010s.
SANITY = [
    ("Start of the game (top 1st, 0 out, bases empty, tied)",
     dict(inning=1, half="top", outs=0, bases=0, score_diff=0), 0.51, 0.57),
    ("Bottom 9th, tied, runner on 3rd, 1 out — the plan's check",
     dict(inning=9, half="bot", outs=1, bases=4, score_diff=0), 0.78, 0.89),
    ("Bottom 9th, tied, bases empty, 0 out",
     dict(inning=9, half="bot", outs=0, bases=0, score_diff=0), 0.60, 0.71),
    ("Bottom 9th, tied, bases loaded, 0 out",
     dict(inning=9, half="bot", outs=0, bases=7, score_diff=0), 0.89, 0.97),
    ("Top 9th, home up 1, 0 out, bases empty",
     dict(inning=9, half="top", outs=0, bases=0, score_diff=1), 0.79, 0.90),
    ("Top 9th, home up 3, 0 out, bases empty",
     dict(inning=9, half="top", outs=0, bases=0, score_diff=3), 0.93, 0.99),
    ("Bottom 9th, home down 1, 2 out, bases empty",
     dict(inning=9, half="bot", outs=2, bases=0, score_diff=-1), 0.02, 0.10),
    ("Bottom 9th, home down 1, 2 out, runner on 2nd",
     dict(inning=9, half="bot", outs=2, bases=2, score_diff=-1), 0.08, 0.21),
    ("Top 10th, tied, automatic runner on 2nd, 0 out",
     dict(inning=10, half="top", outs=0, bases=2, score_diff=0), 0.42, 0.60),
    ("Top 5th, tied, 0 out, bases empty",
     dict(inning=5, half="top", outs=0, bases=0, score_diff=0), 0.50, 0.57),
    ("Bottom 8th, home up 2, 0 out, bases empty",
     dict(inning=8, half="bot", outs=0, bases=0, score_diff=2), 0.86, 0.96),
]


# =================================================================== the locked bundle
def default_dir():
    return Path(os.environ.get("ABS_MODEL_DIR", ROOT / "models")) / "winprob"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bundle(folder, *, markov, run_model, calls, calibrator, manifest_extra):
    """Save the production V(s) and a manifest with the SHA-256 of every file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "v_table.npy", markov.v_table.astype(np.float32))
    np.save(folder / "end_value.npy", markov.end_value.astype(np.float64))
    np.savez_compressed(folder / "run_model.npz", **run_model.to_arrays())
    np.savez_compressed(folder / "call_transitions.npz", **calls.to_arrays())
    (folder / "calibrator.json").write_text(calibrator.to_json())
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name not in ("manifest.json", "WP_CARD.md"):
            files[p.name] = _sha256(p)
    digest = hashlib.sha256("".join(f"{k}:{v}" for k, v in files.items()).encode()).hexdigest()
    manifest = {"name": BUNDLE_NAME, **manifest_extra, "files": files, "content_hash": digest,
                "dims": {"innings": N_INN, "halves": 2, "outs": N_OUTS, "bases": N_BASES,
                         "counts": N_COUNTS, "score_diff": [-DMAX, DMAX]},
                "x_extra": markov.x_extra, "pregame_home_wp_raw": markov.pregame}
    manifest["version"] = f"{manifest_extra.get('date', 'v')}-{digest[:8]}"
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


class WinProb:
    """The locked V(s). Build with load_winprob(); see the module docstring."""

    def __init__(self, folder, manifest, v_table, end_value, run_model, calls, calibrator):
        self.folder, self.manifest = Path(folder), manifest
        self.version = manifest.get("version", "?")
        self.v_raw, self.e_raw = v_table, end_value
        self.rm, self.calls, self.calibrator = run_model, calls, calibrator
        self.v = calibrator.predict(v_table.ravel()).reshape(v_table.shape)
        self.e = calibrator.predict(end_value.ravel()).reshape(end_value.shape)

    def __repr__(self):
        return f"<V(s) {self.version}: {self.calibrator.method} calibration>"

    # ---------------------------------------------------------- the basics
    def frame(self, states, raw=False):
        """Home win probability for every row of a DataFrame of states."""
        s = parse_states(states)
        V = self.v_raw if raw else self.v
        return V[s["i"], s["h"], s["o"], s["b"], s["c"], np.clip(s["d"], -DMAX, DMAX) + DMAX]

    def home(self, state):
        return float(self.frame(state)[0])

    def for_side(self, state, side):
        """side: 'home', 'away', 'batting' or 'fielding'."""
        s = parse_states(state)
        p = float(self.frame(state)[0])
        bat_home = bool(s["h"][0] == 1)
        home = {"home": True, "away": False, "batting": bat_home, "fielding": not bat_home}[side]
        return p if home else 1 - p

    def runs(self, states):
        """Expected runs the batting team scores from here to the end of the inning (RE288)."""
        s = parse_states(states)
        return self.rm.RE[s["h"], s["o"], s["b"], s["c"]]

    # ---------------------------------------------------------- after a call
    def _after(self, s, k):
        """Expected (home win probability, batting-team runs to inning end) after call k."""
        n = len(s["i"])
        wp, runs = np.zeros(n), np.zeros(n)
        cell = self.calls.cell_id(s["o"], s["b"], s["c"], k)
        sign = np.where(s["h"] == 1, 1, -1)
        for cid in np.unique(cell):
            rows = np.flatnonzero(cell == cid)
            dec, w = self.calls.cells[int(cid)]
            i, h, d = s["i"][rows], s["h"][rows], s["d"][rows]
            for (eo, eb, ec, r), ww in zip(dec, w):
                dn = d + sign[rows] * r
                if eo >= 3:
                    val = self.e[i, h, np.clip(dn, -DMAX - RMAX, DMAX + RMAX) + DMAX + RMAX]
                    rr = r
                else:
                    val = self.v[i, h, eo, eb, ec, np.clip(dn, -DMAX, DMAX) + DMAX]
                    rr = r + self.rm.RE[h, eo, eb, ec]
                wp[rows] += ww * val
                runs[rows] += ww * rr
        return wp, runs

    def call_value(self, states, call):
        """
        What a called pitch is worth. `call` is the umpire's call — 'strike'/'ball',
        True/False or 1/0, one value or one per row. Returns a DataFrame:

            wp_stands, wp_overturned   home win probability after each branch
            dwp_challenger             win probability the CHALLENGING side gains if it's
                                       overturned (batting side on a strike, fielding on a ball)
            re_stands, re_overturned   batting team's expected runs to the end of the inning
            druns_batting              runs the overturn is worth to the batting team
        """
        s = parse_states(states)
        k = np.broadcast_to(_call_array(call), s["i"].shape).astype(int)
        wp0, re0 = np.zeros(len(k)), np.zeros(len(k))
        wp1, re1 = np.zeros(len(k)), np.zeros(len(k))
        for kk in (0, 1):
            m = k == kk
            if m.any():
                sub = {key: v[m] for key, v in s.items()}
                a, ra = self._after(sub, kk)
                b, rb = self._after(sub, 1 - kk)
                wp0[m], re0[m], wp1[m], re1[m] = a, ra, b, rb
        bat_home = s["h"] == 1
        # strike called -> the batting side challenges; ball called -> the fielding side
        chall_home = np.where(k == 1, bat_home, ~bat_home)
        dwp_home = wp1 - wp0
        return pd.DataFrame({"wp_stands": wp0, "wp_overturned": wp1,
                             "dwp_challenger": np.where(chall_home, dwp_home, -dwp_home),
                             "re_stands": re0, "re_overturned": re1,
                             "druns_batting": re1 - re0})

    def info(self):
        m = self.manifest
        return {"version": self.version, "calibration": self.calibrator.describe(),
                "seasons": m.get("seasons"), "games": m.get("n_games"),
                "pregame_home_wp": float(self.v[0, 0, 0, 0, 0, DMAX])}


def _call_array(call):
    def one(v):
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("strike", "s", "called_strike", "true", "1"):
                return 1
            if t in ("ball", "b", "false", "0", "blocked_ball"):
                return 0
            raise WPStateError(f"call must be 'strike' or 'ball' — got {v!r}")
        return int(v)
    if np.ndim(call) == 0:
        return np.array(one(call))
    return np.array([one(v) for v in call])


def load_winprob(folder=None, verify=True):
    """Load the locked V(s), checking every file's hash first."""
    folder = Path(folder) if folder else default_dir()
    mpath = folder / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"No locked V(s) at {folder}. Run scripts/19_week10_winprob.py.")
    manifest = json.loads(mpath.read_text())
    if verify:
        bad = [rel for rel, h in manifest["files"].items()
               if not (folder / rel).exists() or _sha256(folder / rel) != h]
        if bad:
            raise RuntimeError(f"V(s) bundle {manifest.get('version')} has changed since it was "
                               f"locked: {bad}. Re-run scripts/19_week10_winprob.py.")
    rm = RunModel.from_arrays(dict(np.load(folder / "run_model.npz")))
    calls = CallTransitions.from_arrays(dict(np.load(folder / "call_transitions.npz")))
    cal = Calibrator.from_json((folder / "calibrator.json").read_text())
    return WinProb(folder, manifest, np.load(folder / "v_table.npy").astype(np.float64),
                   np.load(folder / "end_value.npy"), rm, calls, cal)


__all__ = ["N_INN", "DMAX", "RMAX", "RPA", "SANITY", "WPStateError", "RunModel",
           "state_columns", "empirical_table", "agreement", "GBM_FEATURES", "fit_gbm",
           "gbm_predict", "WELL_POPULATED", "THIN",
           "CallTransitions", "MarkovWP", "parse_states", "calibration_pair", "game_bootstrap",
           "write_bundle", "load_winprob", "WinProb", "default_dir"]
