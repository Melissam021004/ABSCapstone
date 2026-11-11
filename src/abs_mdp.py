"""
abs_mdp.py — Week 11: the challenge decision as a Markov decision process.

The full spec is docs/MDP_SPEC.md and WEEK11.md walks through every output.
In one paragraph:

A game is played pitch by pitch. When the umpire calls a pitch, the call is
either right or wrong, and p(s) — Week 9's locked overturn model — says how
likely ABS is to reverse it. If the call went against a team that still has a
challenge, that team chooses:

    Q_challenge(s, r) = p(s)·V(s_overturned, r) + (1 − p(s))·V(s_stand, r − 1)
    Q_hold(s, r)      = V(s_stand, r)

(a successful challenge is kept, so only the failure branch spends one).
V(s, r) is worked out backward from the end of the game for EVERY state, with
both teams' challenges in the state and both teams choosing well. The answer
has a closed form at every decision:

    challenge  ⇔  p(s) ≥ p* = H / (Δ + H)

    Δ = what the overturn is worth: the win-probability SWING of the call
    H = what holding the challenge is worth: V(s_stand, r) − V(s_stand, r − 1)

Two objectives share the machinery (Tango's two currencies):
    "win"  — maximise the chance of winning (the MDP in the proposal)
    "runs" — maximise run differential (run production, valued in runs)
The published Statcast/Tango breakeven rule, and a leverage-aware version of it,
are POLICIES that can be played inside the "win" model (Breakeven).

What the model leaves out (see the spec): who is batting or pitching (V is an
average-team model, like every published win-expectancy table), the two-second
decision window (p(s) is what Hawk-Eye knows — a full-information benchmark), and
challenges by a position player who is pitching.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .abs_gamestate import COUNTS, FIT_INNINGS, N_BASES, N_COUNTS, N_OUTS

# ------------------------------------------------------------------ dimensions
D = 10                         # home lead −10..+10 (beyond ±10 nothing is in doubt)
ND = 2 * D + 1
R0 = 2                         # challenges each team starts with
NR = R0 + 1                    # 0, 1 or 2 left
KMAX = 4                       # runs on one play (a grand slam is the most)
N_INN = 10                     # innings 1..9, and "extras" (every extra inning is alike)
EXTRA_START = 2                # bases code for the automatic runner on second
P_EDGES = np.array([0, .02, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60,
                    .65, .70, .75, .80, .85, .90, .95, .98, 1.0000001])
NK = len(P_EDGES) - 1          # p(s) bins for the called pitches
NEVER = np.inf                 # a threshold that means "never challenge"
NO = N_OUTS * N_BASES          # 24 (outs, bases) states
NX = 4 * N_BASES               # 32 = (outs' 0..3, bases') — outs' = 3 ends the inning

# ------------------------------------------------------------------ pitch events
EVENTS = ["CB", "CS", "SW", "FO", "IP", "HBP", "AB", "AS"]
EVENT_NAMES = {"CB": "called ball", "CS": "called strike", "SW": "swinging strike",
               "FO": "foul", "IP": "ball in play", "HBP": "hit by pitch",
               "AB": "automatic ball", "AS": "automatic strike"}
E = {k: i for i, k in enumerate(EVENTS)}
DESCRIPTION = {
    "ball": "CB", "blocked_ball": "CB",                         # balls a team can challenge
    "called_strike": "CS",                                      # strikes a team can challenge
    "swinging_strike": "SW", "swinging_strike_blocked": "SW", "foul_tip": "SW",
    "missed_bunt": "SW", "bunt_foul_tip": "SW", "foul_bunt": "SW", "swinging_pitchout": "SW",
    "foul": "FO", "foul_pitchout": "FO",
    "hit_into_play": "IP", "hit_into_play_no_out": "IP", "hit_into_play_score": "IP",
    "hit_by_pitch": "HBP",
    "pitchout": "AB", "automatic_ball": "AB", "intent_ball": "AB",  # balls nobody challenges
    "automatic_strike": "AS",                                        # strikes nobody challenges
}
BALLS = np.array([b for b, s in COUNTS])
STRIKES = np.array([s for b, s in COUNTS])
NEXT_BALL = np.array([c + 3 if BALLS[c] < 3 else -1 for c in range(N_COUNTS)])
NEXT_STRIKE = np.array([c + 1 if STRIKES[c] < 2 else -1 for c in range(N_COUNTS)])
# a walk or a hit batter: the batter takes first and every FORCED runner moves up
WALK_BASES = np.array([1, 3, 3, 7, 5, 7, 7, 7])
WALK_RUNS = np.array([0, 0, 0, 0, 0, 0, 0, 1])
M_SIDE = 200.0                 # a home/away cell borrows this many pitches from the pooled cell


def _norm(a, axis=-1):
    s = a.sum(axis=axis, keepdims=True)
    return np.divide(a, s, out=np.zeros_like(a, dtype=float), where=s > 0)


def p_bins(p):
    """The p(s) bin of each value."""
    return np.clip(np.searchsorted(P_EDGES, np.asarray(p, float), side="right") - 1, 0, NK - 1)


# =================================================================== the pitch model
class PitchModel:
    """
    How a plate appearance moves, one pitch at a time, estimated from every 2026 pitch.

      ev[side, c, e]        chance the next pitch at count c is event e (EVENTS);
                            side 0 = the away team bats (top), 1 = the home team (bottom)
      p_mass[call, c, j]    among CALLED pitches at count c with the umpire's ORIGINAL
                            call (0 ball, 1 strike), the share whose p(s) is in bin j —
                            the supply of challengeable mistakes
      p_mean[call, c, j]    the average p(s) inside that bin
      ip[side, o, b, o', b', k]   a ball in play from (o, b) leaves o' outs (3 = the
                            inning is over), bases b' and k runs scored
      q_run[side, o, b]     chance a runner moves between two pitches (steal, wild
                            pitch, pickoff, caught stealing …)
      run[side, o, b, o', b', k]  where that leaves the inning
    Walks, hit batters and strikeouts follow the rulebook.
    """

    def __init__(self):
        self.ev = self.p_mass = self.p_mean = self.ip = self.q_run = self.run = None
        self.kout = self.walk = None
        self.info = {}

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(cls, pitches, called, p_col="p", call_col="strike", m_side=M_SIDE, calls="original"):
        """
        pitches: abs_gamestate.build_states() output (every pitch of the season).
        called:  one row per CALLED pitch with `count`, the umpire's ORIGINAL call
                 (call_col: 1 strike / 0 ball) and p(s) (p_col).
        calls:   "original" — balls and strikes as the umpire called them (the game the
                 MDP plays: a call only changes if somebody challenges); "final" — as
                 recorded after the 2026 challenges (the season as it was played; used to
                 check the pitch model against what happened).
        """
        m = cls()
        p = pitches
        desc = p["description"].astype(object).where(p["description"].notna(), "")
        ev = desc.astype(str).str.lower().map(DESCRIPTION)
        keep = ev.notna().to_numpy()
        side = p["half"].to_numpy(int)
        cnt = p["count"].to_numpy(int)
        e_idx = ev.map(E).fillna(-1).to_numpy(int)
        if "call_effective" in p.columns and "is_called" in p.columns:
            # what the count actually did after a called pitch (2026 records a challenged
            # pitch's FINAL call; a demo season may record the original one)
            isc = p["is_called"].fillna(False).to_numpy(bool)
            eff = pd.to_numeric(p["call_effective"], errors="coerce").to_numpy(float)
            e_idx = np.where(isc & (eff == 1), E["CS"], np.where(isc & (eff == 0), E["CB"], e_idx))
        m.info.update(n_pitches=int(len(p)), n_other_dropped=int((~keep).sum()),
                      other_descriptions=sorted(desc[~keep].astype(str).unique().tolist())[:12])
        N = np.zeros((2, N_COUNTS, len(EVENTS)))
        np.add.at(N, (side[keep], cnt[keep], e_idx[keep]), 1.0)
        pooled = _norm(N.sum(0))
        evp = (N + m_side * pooled[None]) / (N.sum(-1, keepdims=True) + m_side)
        # In 2026 a challenged pitch is recorded with its FINAL call. How many pitches
        # were called is right either way; the split into balls and strikes comes from
        # the umpire's ORIGINAL calls (`called`).
        cc = called["count"].to_numpy(int)
        cs = pd.to_numeric(called[call_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(cs)
        share = np.array([cs[ok & (cc == c)].mean() if (ok & (cc == c)).any()
                          else evp[:, c, E["CS"]].sum() / max(evp[:, c, [E["CS"], E["CB"]]].sum(), 1e-12)
                          for c in range(N_COUNTS)])
        if calls == "original":
            called_tot = evp[:, :, E["CB"]] + evp[:, :, E["CS"]]
            evp[:, :, E["CS"]] = called_tot * share[None, :]
            evp[:, :, E["CB"]] = called_tot * (1 - share[None, :])
        m.info["calls"] = calls
        m.ev = _norm(evp)
        m.p_mass, m.p_mean = cls.p_supply(called, p_col, call_col)
        m.info["n_called"] = int(np.isfinite(pd.to_numeric(called[p_col], errors="coerce")).sum())

        # Balls in play and runner moves come from innings 1–8 of complete games (in the
        # 9th a walk-off cuts the play short), like Week 10's run model.
        if "fit_row" in p.columns:
            use = p["fit_row"].fillna(False).to_numpy(bool)
        else:
            use = p["inning"].to_numpy(int) <= FIT_INNINGS
        o, b = p["outs"].to_numpy(int), p["bases"].to_numpy(int)
        eo = np.minimum(p["pa_end_outs"].to_numpy(int), 3)
        eb = np.where(eo >= 3, 0, p["pa_end_bases"].to_numpy(int))
        k = np.clip(p["runs_rest_pa"].to_numpy(int), 0, KMAX)
        ipm = use & (e_idx == E["IP"])
        Nip = np.zeros((2, N_OUTS, N_BASES, 4, N_BASES, KMAX + 1))
        np.add.at(Nip, (side[ipm], o[ipm], b[ipm], eo[ipm], eb[ipm], k[ipm]), 1.0)
        m.ip = cls._shrink(Nip, m_side)
        m.info["n_in_play"] = int(ipm.sum())

        same_pa = (p["game_pk"].eq(p["game_pk"].shift(-1))
                   & p["at_bat_number"].eq(p["at_bat_number"].shift(-1))).to_numpy()
        no = np.minimum(p["next_outs"].to_numpy(int), 3)
        nb = np.where(no >= 3, 0, p["next_bases"].to_numpy(int))
        nr = np.clip(p["runs_next"].to_numpy(int), 0, KMAX)
        base = use & same_pa & keep
        moved = base & ((no != o) | (nb != b) | (nr != 0))
        Nq, Nm = np.zeros((2, N_OUTS, N_BASES)), np.zeros((2, N_OUTS, N_BASES))
        np.add.at(Nq, (side[base], o[base], b[base]), 1.0)
        np.add.at(Nm, (side[moved], o[moved], b[moved]), 1.0)
        poolq = Nm.sum(0) / np.maximum(Nq.sum(0), 1)
        m.q_run = (Nm + m_side * poolq[None]) / (Nq + m_side)
        Nr = np.zeros((2, N_OUTS, N_BASES, 4, N_BASES, KMAX + 1))
        np.add.at(Nr, (side[moved], o[moved], b[moved], no[moved], nb[moved], nr[moved]), 1.0)
        m.run = cls._shrink(Nr, m_side)
        empty = m.run.reshape(2, N_OUTS, N_BASES, -1).sum(-1) == 0
        for s_, o_, b_ in zip(*np.nonzero(empty)):          # never moved: nothing moves
            m.run[s_, o_, b_, o_, b_, 0] = 1.0
            m.q_run[s_, o_, b_] = 0.0
        m.info["runner_move_rate"] = float(Nm.sum() / max(Nq.sum(), 1))
        m.info["n_runner_moves"] = int(Nm.sum())

        # Strike three and ball four (or a hit batter) follow the rulebook — plus whatever
        # the runners did on that same pitch (a steal on strike three, a wild pitch on ball
        # four, a strike-'em-out-throw-'em-out double play), estimated from the season.
        last = ~same_pa
        strike_like = np.isin(e_idx, [E["CS"], E["SW"], E["AS"]])
        ball_like = np.isin(e_idx, [E["CB"], E["AB"]])
        kend = use & last & strike_like & (STRIKES[cnt] == 2)
        wend = use & last & ((ball_like & (BALLS[cnt] == 3)) | (e_idx == E["HBP"]))
        m.kout = cls._end_dist(side, o, b, eo, eb, k, kend, "K", m_side)
        m.walk = cls._end_dist(side, o, b, eo, eb, k, wend, "W", m_side)
        m.info["n_strikeouts"], m.info["n_walks_hbp"] = int(kend.sum()), int(wend.sum())
        return m

    @staticmethod
    def rulebook(kind):
        """The textbook end of a plate appearance: 'K' strike three, 'W' a free base."""
        R = np.zeros((N_OUTS, N_BASES, 4, N_BASES, KMAX + 1))
        for o in range(N_OUTS):
            for b in range(N_BASES):
                if kind == "K":
                    R[o, b, o + 1, b if o + 1 < 3 else 0, 0] = 1.0
                else:
                    R[o, b, o, WALK_BASES[b], WALK_RUNS[b]] = 1.0
        return R

    @classmethod
    def _end_dist(cls, side, o, b, eo, eb, k, mask, kind, m_side):
        N = np.zeros((2, N_OUTS, N_BASES, 4, N_BASES, KMAX + 1))
        np.add.at(N, (side[mask], o[mask], b[mask], eo[mask], eb[mask], k[mask]), 1.0)
        rule = cls.rulebook(kind)
        Nf = N.reshape(2, N_OUTS, N_BASES, -1)
        pool = Nf.sum(0)
        P0 = (pool + 50.0 * rule.reshape(N_OUTS, N_BASES, -1)) / (pool.sum(-1, keepdims=True) + 50.0)
        P = (Nf + m_side * P0[None]) / (Nf.sum(-1, keepdims=True) + m_side)
        return P.reshape(N.shape)

    @staticmethod
    def _shrink(N, m_side):
        shp = N.shape
        Nf = N.reshape(2, N_OUTS, N_BASES, -1)
        pool = _norm(Nf.sum(0))
        P = (Nf + m_side * pool[None]) / (Nf.sum(-1, keepdims=True) + m_side)
        P = np.where(Nf.sum((0, -1), keepdims=True) > 0, P, 0.0)
        return P.reshape(shp)

    @staticmethod
    def p_supply(called, p_col="p", call_col="strike"):
        """(p_mass, p_mean), each (2 calls, 12 counts, NK bins), from called pitches."""
        cc = called["count"].to_numpy(int)
        cs = pd.to_numeric(called[call_col], errors="coerce").to_numpy(float)
        pv = np.clip(pd.to_numeric(called[p_col], errors="coerce").to_numpy(float), 0, 1)
        ok = np.isfinite(pv) & np.isfinite(cs)
        mids = (P_EDGES[:-1] + np.minimum(P_EDGES[1:], 1.0)) / 2
        pm_, pb_ = np.zeros((2, N_COUNTS, NK)), np.tile(mids, (2, N_COUNTS, 1))
        for k in (0, 1):
            for c in range(N_COUNTS):
                v = pv[ok & (cc == c) & (cs == k)]
                if not len(v):
                    pm_[k, c, 0] = 1.0                       # no data: no mistakes
                    continue
                j = p_bins(v)
                n = np.bincount(j, minlength=NK).astype(float)
                pm_[k, c] = n / n.sum()
                pb_[k, c] = np.where(n > 0, np.bincount(j, weights=v, minlength=NK)
                                     / np.maximum(n, 1), mids)
        return pm_, pb_

    # ---------------------------------------------------------------- the toy version
    @classmethod
    def toy(cls, strike_supply=((0.01, 0.88), (0.30, 0.04), (0.97, 0.08)),
            ball_supply=((0.01, 0.96), (0.30, 0.02), (0.97, 0.02))):
        """
        A textbook game with no data behind it — every count alike, balls in play are
        outs (68%), singles (22%), doubles (6%) or home runs (4%) with every runner
        moving up as many bases as the batter, no stolen bases. Called strikes are wrong
        8% of the time with p(s) = 0.97 and borderline 4% of the time with p(s) = 0.30;
        called balls 2% and 2%. Used by the tests (answers that must hold in ANY
        baseball-like game) and as the reduced "toy" model of the weekly plan.
        """
        m = cls()
        ev = np.zeros((2, N_COUNTS, len(EVENTS)))
        for k, v in {"CB": .35, "CS": .17, "SW": .11, "FO": .17, "IP": .19, "HBP": .01}.items():
            ev[:, :, E[k]] = v
        m.ev = _norm(ev)
        ip = np.zeros((2, N_OUTS, N_BASES, 4, N_BASES, KMAX + 1))
        for o in range(N_OUTS):
            for b in range(N_BASES):
                ip[:, o, b, o + 1, b if o + 1 < 3 else 0, 0] += 0.68
                for prob, n in ((0.22, 1), (0.06, 2), (0.04, 4)):
                    nb, runs = 0, 0
                    for base in (1, 2, 3):
                        if b & (1 << (base - 1)):
                            if base + n >= 4:
                                runs += 1
                            else:
                                nb |= 1 << (base + n - 1)
                    if n >= 4:
                        runs += 1
                    else:
                        nb |= 1 << (n - 1)
                    ip[:, o, b, o, nb, min(runs, KMAX)] += prob
        m.ip = ip
        m.q_run = np.zeros((2, N_OUTS, N_BASES))
        m.run = np.zeros((2, N_OUTS, N_BASES, 4, N_BASES, KMAX + 1))
        for o in range(N_OUTS):
            for b in range(N_BASES):
                m.run[:, o, b, o, b, 0] = 1.0
        m.kout = np.broadcast_to(cls.rulebook("K"), (2,) + cls.rulebook("K").shape).copy()
        m.walk = np.broadcast_to(cls.rulebook("W"), (2,) + cls.rulebook("W").shape).copy()
        m.p_mass, m.p_mean = np.zeros((2, N_COUNTS, NK)), np.zeros((2, N_COUNTS, NK))
        mids = (P_EDGES[:-1] + np.minimum(P_EDGES[1:], 1.0)) / 2
        m.p_mean[:] = mids
        for call, supply in ((1, strike_supply), (0, ball_supply)):
            for p, share in supply:
                j = int(p_bins([p])[0])
                m.p_mass[call, :, j] += share
                m.p_mean[call, :, j] = p
        m.p_mass = _norm(m.p_mass)
        m.info = {"toy": True}
        return m

    # ---------------------------------------------------------------- variants
    def with_p_supply(self, p_mass, p_mean):
        """The same game with a different supply of mistakes (e.g. one umpire tier)."""
        m = PitchModel()
        m.ev, m.ip, m.q_run, m.run, m.info = self.ev, self.ip, self.q_run, self.run, dict(self.info)
        m.kout, m.walk = self.kout, self.walk
        m.p_mass, m.p_mean = np.asarray(p_mass, float), np.asarray(p_mean, float)
        return m

    # ---------------------------------------------------------------- persistence
    def to_arrays(self):
        return {"ev": self.ev, "p_mass": self.p_mass, "p_mean": self.p_mean, "ip": self.ip,
                "q_run": self.q_run, "run": self.run, "kout": self.kout, "walk": self.walk}

    @classmethod
    def from_arrays(cls, a, info=None):
        m = cls()
        for k in ("ev", "p_mass", "p_mean", "ip", "q_run", "run", "kout", "walk"):
            setattr(m, k, np.asarray(a[k], float))
        m.info = dict(info or {})
        return m

    def summary(self):
        """Per count: how often each event happens (both sides pooled)."""
        df = pd.DataFrame(self.ev.mean(0), columns=[EVENT_NAMES[e] for e in EVENTS])
        df.insert(0, "count", [f"{b}-{s}" for b, s in COUNTS])
        return df


# =================================================================== one half-inning
def _rminus(X, axis):
    """X with the deciding team's challenges one lower (index r → r − 1)."""
    Y = np.array(X, copy=True)
    if axis == -2:
        Y[..., 1:, :] = X[..., :-1, :]
    else:
        Y[..., 1:] = X[..., :-1]
    return Y


def _tails(mass, mean):
    """Σ_{j ≥ k} m_j and Σ_{j ≥ k} m_j·p_j for k = 0..NK (the last is 0)."""
    z = np.zeros(mass.shape[:-1] + (1,))
    M = np.concatenate([np.cumsum(mass[..., ::-1], -1)[..., ::-1], z], -1)
    S = np.concatenate([np.cumsum((mass * mean)[..., ::-1], -1)[..., ::-1], z], -1)
    return M, S


def solve_half(pm, side, T, objective="win", walkoff=False, thr=None, V0=None,
               tol=1e-9, max_outer=500, inner=3, keep=True):
    """
    Work out one half-inning backward from its third out.

    T       value when the half-inning ends, shape (nd, NR, NR) — home-oriented (home win
            probability, or home run margin), challenges in (batting, fielding) order
    side    0 top (the away team bats), 1 bottom
    thr     None: both teams choose optimally. Or (thr_bat, thr_fld), each (3, 8, 12, nd,
            NR, NR): the batting team challenges a called strike when p(s) ≥ thr_bat, the
            fielding team a called ball when p(s) ≥ thr_fld.
    V0      a starting guess (e.g. the last solution) — only speeds things up.

    Returns V (3 outs, 8 bases, 12 counts, nd, NR, NR) and, with keep, the decision at
    every called pitch, from the deciding team's side: thr_S/thr_B (p*), H_S/H_B (the
    value of holding the challenge), D_S/D_B (the swing Δ). _S = a called strike (the
    batting team decides), _B = a called ball (the fielding team decides).
    """
    win = objective == "win"
    nd = T.shape[0]
    sigma = 1 if side == 1 else -1
    wo = bool(walkoff and win)
    ev = pm.ev[side]
    def mats(A):
        Mk = [A[side][..., k].reshape(NO, NX) for k in range(KMAX + 1)]
        return Mk, [bool(np.any(x)) for x in Mk]

    (IPk, use_ip), (MVk, use_mv) = mats(pm.ip), mats(pm.run)
    (KOk, use_ko), (WKk, use_wk) = mats(pm.kout), mats(pm.walk)
    q = pm.q_run[side][:, :, None, None, None]
    if win:
        dv = np.arange(-D, D + 1)
        sidx = [np.clip(np.arange(ND) + sigma * k, 0, ND - 1) for k in range(KMAX + 1)]
        wmask = [(dv + sigma * k) > 0 for k in range(KMAX + 1)]

        def sh(X, k):
            if k == 0:
                return X
            Y = np.take(X, sidx[k], axis=-3)
            if wo:
                Y[..., wmask[k], :, :] = 1.0                  # a walk-off: the home team has won
            return Y
    else:
        def sh(X, k):
            return X + sigma * k if k else X

    Text = np.broadcast_to(T[None], (N_BASES,) + T.shape)

    def ext(Vc):                                               # outs 0..2 + "inning over"
        return np.concatenate([Vc, Text[None]], axis=0)

    def apply(Mk, used, X):                                    # X (4, 8, …) -> (3, 8, …)
        tail = X.shape[2:]
        acc = Mk[0] @ X.reshape(NX, -1)
        for k in range(1, KMAX + 1):
            if used[k]:
                acc = acc + Mk[k] @ sh(X, k).reshape(NX, -1)
        return acc.reshape((N_OUTS, N_BASES) + tail)

    def after_move(Vc):                                        # a runner may move first
        return (1 - q) * Vc + q * apply(MVk, use_mv, ext(Vc))

    pmS, pmB = pm.p_mean[1], pm.p_mean[0]
    MS, S1S = _tails(pm.p_mass[1], pm.p_mean[1])
    MB, S1B = _tails(pm.p_mass[0], pm.p_mean[0])
    none_bat = (np.arange(NR) == 0)[:, None]                   # broadcasts over (…, r_bat, r_fld)
    none_fld = (np.arange(NR) == 0)[None, :]
    dpos = (np.arange(-D, D + 1) > 0) if win else None

    def decide(Hold, Swing, none_mask, given):
        if given is None:
            den = Swing + Hold
            pos = den > 1e-12
            t = np.where(pos, np.maximum(Hold, 0.0) / np.where(pos, den, 1.0), NEVER)
        else:
            t = np.array(np.broadcast_to(given, Hold.shape), dtype=float)
        return np.where(none_mask, NEVER, t)

    shp = (N_OUTS, N_BASES, N_COUNTS, nd, NR, NR)
    if V0 is not None:
        V = np.array(V0, dtype=float, copy=True)
    else:
        V = np.empty(shp)
        V[:] = T[None, None, None] if win else 0.0
    if keep:
        thr_S, thr_B = np.full(shp, NEVER), np.full(shp, NEVER)
        H_S, H_B, D_S, D_B = (np.zeros(shp) for _ in range(4))
    it = 0
    for it in range(1, max_outer + 1):
        N0 = ext(V[:, :, 0])                                   # the next plate appearance
        I = apply(IPk, use_ip, N0)                             # ball in play
        Wk = apply(WKk, use_wk, N0)                            # ball four / hit batter
        K = apply(KOk, use_ko, N0)                             # strike three
        post = {}
        V_old0 = V[:, :, 0].copy()
        for c in range(N_COUNTS - 1, -1, -1):
            nb_, ns_ = NEXT_BALL[c], NEXT_STRIKE[c]
            Bv = post[nb_] if nb_ >= 0 else Wk
            Sv = post[ns_] if ns_ >= 0 else K
            # a called STRIKE — the batting team may challenge
            Sv_m = _rminus(Sv, -2)
            Hs, Ds = sigma * (Sv - Sv_m), sigma * (Bv - Sv)
            ts = decide(Hs, Ds, none_bat, None if thr is None else thr[0][:, :, c])
            ks = np.searchsorted(pmS[c], ts.ravel(), side="left").reshape(ts.shape)
            M, S1 = MS[c][ks], S1S[c][ks]
            ES = Sv * (1 - M) + Bv * S1 + Sv_m * (M - S1)
            # a called BALL — the fielding team may challenge
            Bv_m = _rminus(Bv, -1)
            Hb, Db = -sigma * (Bv - Bv_m), -sigma * (Sv - Bv)
            tb = decide(Hb, Db, none_fld, None if thr is None else thr[1][:, :, c])
            kb = np.searchsorted(pmB[c], tb.ravel(), side="left").reshape(tb.shape)
            M, S1 = MB[c][kb], S1B[c][kb]
            EB = Bv * (1 - M) + Sv * S1 + Bv_m * (M - S1)
            rest = ((ev[c, E["SW"]] + ev[c, E["AS"]]) * Sv + ev[c, E["IP"]] * I
                    + ev[c, E["HBP"]] * Wk + ev[c, E["AB"]] * Bv
                    + ev[c, E["CS"]] * ES + ev[c, E["CB"]] * EB)
            f = ev[c, E["FO"]]
            if ns_ >= 0:                                       # a foul is strike one or two
                Vc = rest + f * Sv
            else:                                              # two strikes: a foul changes nothing
                Vc = V[:, :, c]
                den = 1 - f * (1 - q)
                for _ in range(inner):
                    Vc = (rest + f * q * apply(MVk, use_mv, ext(Vc))) / den
            if wo:
                Vc = np.array(Vc, copy=True)
                Vc[..., dpos, :, :] = 1.0
            V[:, :, c] = Vc
            if c > 0:
                post[c] = after_move(Vc)
            if keep:
                thr_S[:, :, c], H_S[:, :, c], D_S[:, :, c] = ts, Hs, Ds
                thr_B[:, :, c], H_B[:, :, c], D_B[:, :, c] = tb, Hb, Db
        if np.max(np.abs(V[:, :, 0] - V_old0)) < tol:
            break
    out = {"V": V, "iterations": it}
    if keep:
        out.update(thr_S=thr_S, thr_B=thr_B, H_S=H_S, H_B=H_B, D_S=D_S, D_B=D_B)
    return out


# =================================================================== the whole game
def _swap(X, side):
    """(…, r_home, r_away) <-> (…, r_bat, r_fld): a swap in the top half, nothing in the bottom."""
    return X if side == 1 else np.swapaxes(X, -1, -2)


class Solution:
    """Everything the solver worked out, half-inning by half-inning."""

    FIELDS = ("V", "thr_S", "thr_B", "H_S", "H_B", "D_S", "D_B")

    def __init__(self, objective):
        self.objective = objective
        self.halves = {}           # (inning 1..10, side) -> arrays (r in batting/fielding order)
        self.start = {}            # (inning, side) -> (nd, r_home, r_away) at the half's start
        self.extra = None          # value at the start of an extra inning, tied, (r_home, r_away)
        self.seconds = 0.0
        self.info = {}

    @property
    def nd(self):
        return ND if self.objective == "win" else 1

    def pregame(self, r_home=R0, r_away=R0):
        st = self.start[(1, 0)]
        return float(st[D if self.objective == "win" else 0, r_home, r_away])

    def lookup(self, field, inning, side, o, b, c, d, r_bat, r_fld):
        """A stored array (V, thr_S, H_S, D_S, thr_B, …) at many states at once."""
        arrs = np.broadcast_arrays(*(np.asarray(x) for x in (inning, side, o, b, c, d, r_bat, r_fld)))
        inning, side, o, b, c, d, rb, rf = (a.astype(int) for a in arrs)
        ii = np.clip(inning, 1, N_INN)
        di = (np.clip(d, -D, D) + D) if self.objective == "win" else np.zeros_like(d)
        out = np.full(ii.shape, np.nan)
        for (i, s), h in self.halves.items():
            m = (ii == i) & (side == s)
            if m.any():
                out[m] = h[field][o[m], b[m], c[m], di[m], rb[m], rf[m]]
        return out

    def start_value(self, inning, side, d=0, r_home=R0, r_away=R0):
        st = self.start[(int(min(max(inning, 1), N_INN)), side)]
        di = (int(np.clip(d, -D, D)) + D) if self.objective == "win" else 0
        return float(st[di, r_home, r_away])

    def hold_value(self, inning, side, team, d=0, r_self=1, r_other=1):
        """V(s, r) − V(s, r − 1) at the START of a half-inning for `team` ('home'/'away'),
        in that team's own units (win probability, or runs)."""
        st = self.start[(int(min(max(inning, 1), N_INN)), side)]
        di = (int(np.clip(d, -D, D)) + D) if self.objective == "win" else 0
        if team == "home":
            return float(st[di, r_self, r_other] - st[di, r_self - 1, r_other])
        return float(st[di, r_other, r_self - 1] - st[di, r_other, r_self])


class GameMDP:
    """
    V(s, r) for a whole game, both teams' challenges in the state.

        GameMDP(pm).solve()                      both teams choose optimally (win)
        GameMDP(pm, "runs").solve()              both maximise run differential
        GameMDP(pm).solve(policies={"home": π, "away": π'})   play fixed rules (Policy)
    """

    def __init__(self, pm, objective="win", tol=1e-9, extra_tol=1e-8):
        if objective not in ("win", "runs"):
            raise ValueError("objective is 'win' or 'runs'")
        self.pm, self.objective = pm, objective
        self.tol, self.extra_tol = tol, extra_tol
        self.nd = ND if objective == "win" else 1

    # ------------------------------------------------------------ terminals
    def _after_top(self, i, start_bottom):
        """Three outs in the top of inning i: the bottom half starts (unless it isn't needed)."""
        T = np.array(start_bottom, copy=True)                  # (nd, r_home, r_away)
        if self.objective == "win" and i >= 9:
            T[np.arange(-D, D + 1) > 0] = 1.0                  # the home team leads: game over
        return T

    def _after_last_bottom(self, extra_pre):
        """Three outs in the bottom of the 9th or of an extra inning."""
        if self.objective == "runs":
            return np.zeros((1, NR, NR))
        dv = np.arange(-D, D + 1)
        T = np.zeros((ND, NR, NR))
        T[dv > 0] = 1.0
        T[dv == 0] = extra_pre                                 # tied: another inning
        return T

    @staticmethod
    def replenish(X):
        """Entering an extra inning, a team with no challenge left gets one (the 2026 rule)."""
        return np.array([[X[max(rh, 1), max(ra, 1)] for ra in range(NR)] for rh in range(NR)])

    def _half(self, i, side, T_ha, policies, walkoff, V0=None):
        thr = None
        if policies is not None:
            bat, fld = ("home", "away") if side == 1 else ("away", "home")
            thr = (policies[bat].thresholds(i, side, "bat", self),
                   policies[fld].thresholds(i, side, "fld", self))
        res = solve_half(self.pm, side, _swap(T_ha, side), objective=self.objective,
                         walkoff=walkoff, thr=thr, V0=V0, tol=self.tol)
        b0 = EXTRA_START if i == N_INN else 0
        return res, _swap(res["V"][0, b0, 0], side)

    # ------------------------------------------------------------ solve
    def solve(self, policies=None, max_extra=80):
        """Backward induction from the last out. policies=None: both teams optimal."""
        t0 = time.time()
        sol = Solution(self.objective)
        win = self.objective == "win"
        # extras: every extra inning is alike — find the value of starting one
        if win:
            X = np.full((NR, NR), 0.5)
            Vb = Vt = None
            for k in range(max_extra):
                rb_, sb = self._half(N_INN, 1, self._after_last_bottom(self.replenish(X)),
                                     policies, True, Vb)
                rt_, st = self._half(N_INN, 0, self._after_top(N_INN, sb), policies, False, Vt)
                Vb, Vt = rb_["V"], rt_["V"]
                Xn = st[D]
                done = np.max(np.abs(Xn - X)) < self.extra_tol
                X = Xn
                if done:
                    break
            sol.info["extra_iterations"] = k + 1
        else:                                                  # runs: one more inning, then stop
            rb_, sb = self._half(N_INN, 1, np.zeros((1, NR, NR)), policies, False)
            rt_, st = self._half(N_INN, 0, sb, policies, False)
            X = st[0]
        sol.halves[(N_INN, 1)], sol.halves[(N_INN, 0)] = rb_, rt_
        sol.start[(N_INN, 1)], sol.start[(N_INN, 0)] = sb, st
        sol.extra = X
        Tb = self._after_last_bottom(self.replenish(X) if win else None)
        for i in range(9, 0, -1):                             # the 9th back to the 1st
            rb_, sb = self._half(i, 1, Tb, policies, walkoff=(i >= 9))
            rt_, st = self._half(i, 0, self._after_top(i, sb), policies, False)
            sol.halves[(i, 1)], sol.halves[(i, 0)] = rb_, rt_
            sol.start[(i, 1)], sol.start[(i, 0)] = sb, st
            Tb = st
        sol.seconds = time.time() - t0
        return sol


# =================================================================== policies
class Policy:
    """A challenge rule: thresholds(i, side, role, game) gives an array (3, 8, 12, nd,
    NR, NR) in (r_bat, r_fld) order; the team challenges when p(s) ≥ threshold. role
    'bat' decides on called strikes, 'fld' on called balls."""

    name = "policy"

    def thresholds(self, i, side, role, game):
        raise NotImplementedError


class FromSolution(Policy):
    """The thresholds an MDP solution worked out (optimal for its own objective)."""

    def __init__(self, sol, name="optimal"):
        self.sol, self.name = sol, name

    def thresholds(self, i, side, role, game):
        t = self.sol.halves[(i, side)]["thr_S" if role == "bat" else "thr_B"]
        return np.broadcast_to(t, t.shape[:3] + (game.nd,) + t.shape[4:])


class Constant(Policy):
    """Challenge whenever p(s) ≥ a fixed number (np.inf: never challenge)."""

    def __init__(self, p, name=None):
        self.p = p
        self.name = name or ("never" if not np.isfinite(p) else f"p ≥ {p:.2f}")

    def thresholds(self, i, side, role, game):
        return np.full((N_OUTS, N_BASES, N_COUNTS, game.nd, NR, NR), self.p)


class Breakeven(Policy):
    """
    The published Statcast/Tango rule: challenge when p(s) ≥ cost / (cost + RV), with
    RV the run value of flipping the call (RE288) and a failed challenge priced at
    `cost` runs (0.2 on Baseball Savant). With `leverage` (Tango's LI at the plate
    appearance, shape (N_INN, 2, 3, 8, ND)) the run value is multiplied by LI — the
    same rule, made aware of the inning and the score.
    """

    def __init__(self, flip, cost=0.2, leverage=None, name=None):
        self.flip = np.asarray(flip, float)
        if self.flip.ndim == 3:
            self.flip = np.stack([self.flip, self.flip])
        self.cost, self.li = cost, leverage
        self.name = name or ("breakeven × LI" if leverage is not None else "Savant breakeven")

    def thresholds(self, i, side, role, game):
        rv = np.maximum(self.flip[side], 1e-9)                 # (3, 8, 12)
        if self.li is None:
            t = self.cost / (self.cost + rv)
            return np.broadcast_to(t[..., None, None, None], (N_OUTS, N_BASES, N_COUNTS, game.nd, NR, NR))
        li = self.li[min(i, N_INN) - 1, side]                  # (3, 8, ND)
        if game.nd == 1:
            li = li[..., D:D + 1]
        t = self.cost / (self.cost + rv[..., None] * np.maximum(li[:, :, None, :], 1e-9))
        return np.broadcast_to(t[..., None, None], (N_OUTS, N_BASES, N_COUNTS, game.nd, NR, NR))


def flip_runs(RE):
    """
    The RE288 run value of flipping a call at (outs, bases, count): expected runs to the
    end of the inning after a BALL minus after a STRIKE (ball four is a walk, strike
    three an out). RE: (…, 3, 8, 12) expected runs from each state, as in Week 10.
    """
    RE = np.asarray(RE, float)
    out = np.zeros(RE.shape)
    for o in range(N_OUTS):
        for b in range(N_BASES):
            for c in range(N_COUNTS):
                rb = (RE[..., o, b, NEXT_BALL[c]] if NEXT_BALL[c] >= 0
                      else WALK_RUNS[b] + RE[..., o, WALK_BASES[b], 0])
                rs = (RE[..., o, b, NEXT_STRIKE[c]] if NEXT_STRIKE[c] >= 0
                      else (0.0 if o == 2 else RE[..., o + 1, b, 0]))
                out[..., o, b, c] = rb - rs
    return out


def wp_loss(game, opt_sol, policy, opponent=None):
    """
    Win probability a team gives up per game by playing `policy` while the other team
    plays the optimal policy — at home, on the road, and the average of the two.
    """
    opp = opponent or FromSolution(opt_sol)
    p0 = opt_sol.pregame()
    h = game.solve(policies={"home": policy, "away": opp}).pregame()
    a = game.solve(policies={"home": opp, "away": policy}).pregame()
    return {"home_loss": p0 - h, "away_loss": a - p0, "loss": ((p0 - h) + (a - p0)) / 2}


# =================================================================== simulation
class Simulator:
    """
    Plays whole games pitch by pitch with the PitchModel and a challenge policy for each
    team. It is the check that the backward induction adds up (the home team's win rate
    must match V at the first pitch), and the answer to "how often would a team using
    this rule challenge, and how often would it win the challenge?"
    """

    def __init__(self, pm, game, policies, seed=0):
        self.pm, self.game, self.pol = pm, game, policies
        self.rng = np.random.default_rng(seed)
        self.thr = {}

    def _thr(self, i, side, role):
        key = (i, side, role)
        if key not in self.thr:
            team = ("home" if side == 1 else "away") if role == "bat" else ("away" if side == 1 else "home")
            self.thr[key] = np.asarray(self.pol[team].thresholds(i, side, role, self.game))
        return self.thr[key]

    def run(self, n_games=20000, max_pitches=1200):
        pm, rng, n = self.pm, self.rng, n_games
        inn, side = np.ones(n, int), np.zeros(n, int)
        outs, bases, cnt = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        lead = np.zeros(n, int)                                # home lead (not clipped)
        r = np.full((n, 2), R0)                                # [away, home]
        alive, won = np.ones(n, bool), np.zeros(n)
        st = {k: np.zeros((n, 2)) for k in ("challenges", "overturns", "opportunities",
                                            "wrong_calls", "wrong_missed")}
        by_inning = np.zeros((N_INN, 2))                       # challenges lost / won, by inning
        ev_cdf = np.cumsum(pm.ev, -1)
        pm_cdf = np.cumsum(pm.p_mass, -1)
        ip_cdf, run_cdf, k_cdf, walk_cdf = (np.cumsum(a.reshape(2, N_OUTS, N_BASES, -1), -1)
                                            for a in (pm.ip, pm.run, pm.kout, pm.walk))

        def draw(cdf_rows):
            u = rng.random(len(cdf_rows))
            return np.minimum((u[:, None] > cdf_rows).sum(1), cdf_rows.shape[1] - 1)

        def score(mask, k):
            lead[mask] += np.where(side[mask] == 1, k[mask], -k[mask])
            wo = mask & alive & (side == 1) & (inn >= 9) & (lead > 0)
            alive[wo], won[wo] = False, 1.0

        def jump(mask, cdf):
            """The games in `mask` move to a new (outs, bases), scoring k runs."""
            sel = np.flatnonzero(mask & alive)
            if not len(sel):
                return
            f = draw(cdf[side[sel], outs[sel], bases[sel]])
            eo, rem = np.divmod(f, N_BASES * (KMAX + 1))
            eb, kk = np.divmod(rem, KMAX + 1)
            outs[sel], bases[sel] = eo, eb
            k = np.zeros(n, int)
            k[sel] = kk
            m = np.zeros(n, bool)
            m[sel] = True
            score(m, k)

        def end_half(g):
            top, bot = g & (side == 0), g & (side == 1)
            over_t = top & (inn >= 9) & (lead > 0)
            alive[over_t], won[over_t] = False, 1.0
            over_b = bot & (inn >= 9) & (lead != 0)
            alive[over_b] = False
            won[over_b] = (lead[over_b] > 0).astype(float)
            cont_t, cont_b = top & ~over_t, bot & ~over_b
            side[cont_t] = 1
            side[cont_b] = 0
            inn[cont_b] += 1
            ext = cont_b & (inn >= 10)
            r[ext] = np.maximum(r[ext], 1)
            nxt = cont_t | cont_b
            outs[nxt], cnt[nxt] = 0, 0
            bases[nxt] = np.where(inn[nxt] >= 10, EXTRA_START, 0)

        for _ in range(max_pitches):
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            e = draw(ev_cdf[side[idx], cnt[idx]])
            ball, strike, ipl, hbp, foul = (np.zeros(n, bool) for _ in range(5))
            ball[idx[e == E["AB"]]] = True
            strike[idx[(e == E["AS"]) | (e == E["SW"])]] = True
            ipl[idx[e == E["IP"]]] = True
            hbp[idx[e == E["HBP"]]] = True
            foul[idx[e == E["FO"]]] = True
            for call, code in ((1, E["CS"]), (0, E["CB"])):
                sel = idx[e == code]
                if not len(sel):
                    continue
                j = draw(pm_cdf[call, cnt[sel]])
                p = pm.p_mean[call, cnt[sel], j]
                bat_t = side[sel]                              # column of the batting team in r
                dec_t = bat_t if call == 1 else 1 - bat_t
                rb, rf = r[sel, bat_t], r[sel, 1 - bat_t]
                mine = rb if call == 1 else rf
                ii = np.minimum(inn[sel], N_INN)
                thr = np.full(len(sel), NEVER)
                for i_, s_ in set(zip(ii.tolist(), side[sel].tolist())):
                    m = (ii == i_) & (side[sel] == s_)
                    Tt = self._thr(i_, s_, "bat" if call == 1 else "fld")
                    di = (np.clip(lead[sel][m], -D, D) + D) if Tt.shape[3] > 1 else np.zeros(m.sum(), int)
                    thr[m] = Tt[outs[sel][m], bases[sel][m], cnt[sel][m], di, rb[m], rf[m]]
                can = mine >= 1
                wrong = rng.random(len(sel)) < p               # ABS would reverse this call
                ch = can & (p >= thr)
                np.add.at(st["opportunities"], (sel[can], dec_t[can]), 1)
                np.add.at(st["wrong_calls"], (sel[wrong], dec_t[wrong]), 1)
                np.add.at(st["wrong_missed"], (sel[wrong & ~ch], dec_t[wrong & ~ch]), 1)
                np.add.at(st["challenges"], (sel[ch], dec_t[ch]), 1)
                ok = ch & wrong
                np.add.at(st["overturns"], (sel[ok], dec_t[ok]), 1)
                np.add.at(by_inning, (ii[ch] - 1, ok[ch].astype(int)), 1)
                lost = ch & ~wrong
                r[sel[lost], dec_t[lost]] -= 1
                is_strike = np.where(ok, 1 - call, call).astype(bool)
                strike[sel[is_strike]] = True
                ball[sel[~is_strike]] = True
            four = ball & (BALLS[cnt] == 3)
            moveb = ball & ~four
            cnt[moveb] = NEXT_BALL[cnt[moveb]]
            strike |= foul & (STRIKES[cnt] < 2)
            k3 = strike & (STRIKES[cnt] == 2)
            moves = strike & ~k3
            cnt[moves] = NEXT_STRIKE[cnt[moves]]
            walk = four | hbp
            jump(walk, walk_cdf)                               # ball four / hit batter
            jump(k3, k_cdf)                                    # strike three
            jump(ipl, ip_cdf)                                  # ball in play
            done_pa = walk | k3 | ipl
            cnt[done_pa] = 0
            go = np.flatnonzero(alive & ~done_pa & (outs < 3))
            if len(go):                                        # a runner moves before the next pitch
                go = go[rng.random(len(go)) < pm.q_run[side[go], outs[go], bases[go]]]
                mv = np.zeros(n, bool)
                mv[go] = True
                jump(mv, run_cdf)
            three = alive & (outs >= 3)
            if three.any():
                end_half(three)
        fin = ~alive
        out = {"games": n, "unfinished": int(alive.sum()),
               "home_win": float(won[fin].mean()) if fin.any() else np.nan}
        out["se_home_win"] = float(np.sqrt(out["home_win"] * (1 - out["home_win"]) / max(fin.sum(), 1)))
        for k, v in st.items():
            out[f"{k}_per_team_game"] = float(v.sum() / (2 * n))
            out[f"{k}_home"] = float(v[:, 1].mean())
            out[f"{k}_away"] = float(v[:, 0].mean())
        tot = st["challenges"].sum()
        out["success_rate"] = float(st["overturns"].sum() / tot) if tot else np.nan
        out["left_at_end"] = float(r.mean())
        out["by_inning"] = by_inning
        out["extra_innings_share"] = float((inn[fin] >= 10).mean()) if fin.any() else np.nan
        return out


# =================================================================== persistence
def save_solution(sol, folder, extra=None):
    """The solved model (float32 arrays + a manifest) for Spring and the dashboard."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for (i, s), h in sol.halves.items():
        for k in Solution.FIELDS:
            if k in h:
                arrays[f"{k}_{i}_{s}"] = np.where(np.isfinite(h[k]), h[k], 9.0).astype(np.float32)
        arrays[f"start_{i}_{s}"] = sol.start[(i, s)].astype(np.float32)
    path = folder / f"solution_{sol.objective}.npz"
    np.savez_compressed(path, **arrays)
    man = {"objective": sol.objective, "D": D, "R0": R0, "p_edges": P_EDGES.tolist(),
           "pregame_home": sol.pregame(), "seconds": round(sol.seconds, 2),
           "never_threshold_stored_as": 9.0, **sol.info, **(extra or {})}
    (folder / f"manifest_{sol.objective}.json").write_text(json.dumps(man, indent=1, default=float))
    return path


def load_solution(folder, objective="win"):
    folder = Path(folder)
    a = np.load(folder / f"solution_{objective}.npz")
    sol = Solution(objective)
    for key in a.files:
        name, i, s = key.rsplit("_", 2)
        i, s = int(i), int(s)
        if name == "start":
            sol.start[(i, s)] = a[key].astype(float)
            continue
        x = a[key].astype(float)
        if name.startswith("thr"):
            x = np.where(x >= 8.99, NEVER, x)
        sol.halves.setdefault((i, s), {})[name] = x
    sol.info = json.loads((folder / f"manifest_{objective}.json").read_text())
    return sol


def load_mdp(folder=None, objective="win"):
    """
    The solved MDP, rebuilt from the saved pitch model (models/mdp/pitch_model.npz) —
    solving takes a few seconds, so the 20 MB solution file never needs to go in git.

        sol = load_mdp()                     # V(s, r), thresholds, H and Δ for every state
        sol.lookup("thr_S", inning, half, outs, bases, count, home_lead, r_bat, r_fld)
    """
    import os
    folder = Path(folder) if folder else Path(os.environ.get("ABS_MODEL_DIR", "models")) / "mdp"
    path = folder / "pitch_model.npz"
    if not path.exists():
        raise FileNotFoundError(f"No pitch model at {path}. Run scripts/21_week11_mdp.py.")
    info = {}
    man = folder / "manifest.json"
    if man.exists():
        info = json.loads(man.read_text()).get("pitch_model", {})
    pm = PitchModel.from_arrays(dict(np.load(path)), info)
    return GameMDP(pm, objective).solve()


__all__ = ["PitchModel", "GameMDP", "Solution", "solve_half", "Policy", "FromSolution",
           "Constant", "Breakeven", "flip_runs", "wp_loss", "Simulator", "save_solution",
           "load_solution", "load_mdp", "p_bins", "D", "ND", "R0", "NR", "KMAX", "P_EDGES", "NK", "EVENTS",
           "EVENT_NAMES", "NEVER", "N_INN", "EXTRA_START", "NEXT_BALL", "NEXT_STRIKE",
           "WALK_BASES", "WALK_RUNS"]
