"""
abs_leverage.py — Week 11: Tom Tango's Leverage Index (LI), from our own V(s).

Tango built LI to answer "how much is riding on this plate appearance?" in one
number, scaled so the average plate appearance is 1 (Tango, "Crucial Situations",
The Hardball Times 2006; FanGraphs Library "Leverage Index"):

             Σ_e P(e | s) · |WP(after e) − WP(s)|
    LI(s) = ─────────────────────────────────────────────────────
             the same quantity, averaged over every plate appearance

e runs over the ways the plate appearance can end (outs, bases and runs after it),
WP is win probability. We compute it with Week 10's locked V(s) and its plate-
appearance model, and average over the real 2026 plate appearances, so LI = 1 is
"a typical 2026 plate appearance". FanGraphs' bands: low < 0.85, medium 0.85–2,
high > 2.

Why it matters for the challenge decision: the WP swing of a call, Δ, is (roughly)
LI × the run value of the call × wins per run. LI is therefore the single number
that tells a team how far to lower its bar — and the MDP's threshold p* = H/(Δ+H)
falls as LI rises. `call_leverage` gives the same normalisation for called
pitches (the swing of THIS call relative to the average call).
"""

import numpy as np
import pandas as pd

from .abs_gamestate import N_OUTS
from .abs_winprob import DMAX, N_INN, RMAX, _end_decode

BANDS = [(0.0, 0.85, "low (< 0.85)"), (0.85, 2.0, "medium (0.85–2)"), (2.0, np.inf, "high (> 2)")]

# Published reference points (see WEEK11.md, "Sources"):
#  * average LI by inning, Tango's LI on 2013 plays (Studeman, THT, Dec 9 2013)
#  * the first play of the top / bottom of the 9th (same article)
#  * the highest-leverage spot Tango names: bottom 9th, home team down one, runners on
#    second and third, one out — "over 10" (Tango, "Crucial Situations" archive, 2002)
PUBLISHED_BY_INNING_2013 = {1: 0.97, 2: 0.95, 3: 0.94, 4: 0.95, 5: 0.94, 6: 0.95, 7: 0.96,
                            8: 0.97, 9: 1.21, 10: 2.44}
PUBLISHED_TOP_BOT_9_2013 = {"top": 0.92, "bot": 1.74}
PUBLISHED_FIRST_PLAY_9_2013 = {"top": 0.87, "bot": 1.48}


def pa_swing(wp):
    """
    Expected |change in home win probability| over one plate appearance, from every
    plate-appearance start (count 0-0): array (N_INN, 2, 3, 8, 2·DMAX + 1) indexed
    [inning index, half, outs, bases, home lead + DMAX].
    """
    V = wp.v[:, :, :, :, 0, :]                    # (inning, half, outs, bases, lead)
    Ev = wp.e                                      # value when the half-inning ends
    T = wp.rm.T[:, :, :, 0, :]                     # (half, outs, bases, end)
    dv = np.arange(-DMAX, DMAX + 1)
    out = np.zeros(V.shape)
    n_end = T.shape[-1]
    for e in np.flatnonzero(T.reshape(-1, n_end).sum(0) > 0):
        eo, eb, r = (int(x) for x in _end_decode(e))
        for h in (0, 1):
            dn = dv + (1 if h == 1 else -1) * r
            if eo >= 3:
                after = Ev[:, h][:, np.clip(dn, -DMAX - RMAX, DMAX + RMAX) + DMAX + RMAX]
            else:
                after = V[:, h, eo, eb][:, np.clip(dn, -DMAX, DMAX) + DMAX]
            w = T[h, :, :, e]
            out[:, h] += w[None, :, :, None] * np.abs(after[:, None, None, :] - V[:, h])
    return out


def pa_start_index(pa):
    """(inning index, half, outs, bases, lead index) for plate-appearance rows."""
    i = np.minimum(pa["inning"].to_numpy(int), N_INN) - 1
    return (i, pa["half"].to_numpy(int), pa["outs"].to_numpy(int), pa["bases"].to_numpy(int),
            np.clip(pa["diff"].to_numpy(int), -DMAX, DMAX) + DMAX)


class Leverage:
    """
    Tango's LI for every state, normalised over the season's plate appearances.

        lev = Leverage.fit(wp, pitches)        # pitches from abs_gamestate.build_states
        lev.li[i, h, o, b, d + DMAX]           # LI at a plate-appearance start
        lev.for_states(df)                      # LI for rows with inning/half/outs/bases/diff
        lev.mdp_grid(D)                         # (N_INN, 2, 3, 8, 2D+1) for abs_mdp.Breakeven
    """

    def __init__(self, swing, denom, pa_li):
        self.swing, self.denom = swing, denom
        self.li = swing / denom
        self.pa_li = pa_li                         # LI of every real plate appearance

    @classmethod
    def fit(cls, wp, pitches):
        swing = pa_swing(wp)
        pa = pitches[pitches["pa_start"].astype(bool)]
        if "usable" in pa.columns:
            pa = pa[pa["usable"].fillna(False).astype(bool)]
        idx = pa_start_index(pa)
        s = swing[idx]
        denom = float(s.mean())
        out = cls(swing, denom, s / denom)
        out.pa = pa[["game_pk", "inning", "half", "outs", "bases", "diff"]].reset_index(drop=True)
        out.pa["li"] = out.pa_li
        return out

    def for_states(self, df):
        return self.li[pa_start_index(df)]

    def mdp_grid(self, D):
        return self.li[..., DMAX - D:DMAX + D + 1]

    # ------------------------------------------------------------ summaries
    def bands(self):
        x = self.pa_li
        rows = [{"band": name, "share_of_PAs": float(((x >= lo) & (x < hi)).mean())} for lo, hi, name in BANDS]
        rows.append({"band": "below 1", "share_of_PAs": float((x < 1).mean())})
        return pd.DataFrame(rows)

    def by_inning(self):
        """Average LI of the season's plate appearances by inning (10 = extras), with the
        2013 published averages beside them."""
        p = self.pa.copy()
        p["inn"] = np.minimum(p["inning"], 10)
        g = p.groupby("inn")["li"].agg(["mean", "size"]).rename(columns={"mean": "li_2026", "size": "n_pa"})
        g["li_2013_published"] = [PUBLISHED_BY_INNING_2013.get(int(i), np.nan) for i in g.index]
        return g.reset_index().rename(columns={"inn": "inning"})

    def ninth(self):
        """Top vs bottom of the 9th: every PA, and the first PA of the half."""
        p = self.pa[self.pa["inning"] == 9]
        rows = []
        for h, name in ((0, "top"), (1, "bot")):
            q = p[p["half"] == h]
            first = q.groupby("game_pk").head(1)
            rows.append({"half": name, "li_2026_all": float(q["li"].mean()) if len(q) else np.nan,
                         "li_2013_all": PUBLISHED_TOP_BOT_9_2013[name],
                         "li_2026_first_pa": float(first["li"].mean()) if len(first) else np.nan,
                         "li_2013_first_play": PUBLISHED_FIRST_PLAY_9_2013[name], "n_pa": len(q)})
        return pd.DataFrame(rows)

    def table(self, inning, half, lead, count_rows=None):
        """LI by bases (rows) × outs (columns) for one inning, half and home lead."""
        i = min(inning, N_INN) - 1
        t = self.li[i, half, :, :, int(np.clip(lead, -DMAX, DMAX)) + DMAX]
        from .abs_gamestate import BASE_NAMES
        return pd.DataFrame(t.T, index=BASE_NAMES, columns=[f"{o} out" for o in range(N_OUTS)])

    def top_states(self, n=10):
        """The highest-LI states that actually occurred in 2026."""
        p = self.pa.copy()
        p["key"] = list(zip(p["inning"].clip(upper=10), p["half"], p["outs"], p["bases"], p["diff"].clip(-DMAX, DMAX)))
        g = p.groupby("key")["li"].agg(["mean", "size"]).sort_values("mean", ascending=False).head(n)
        return g.reset_index()


def call_leverage(dwp, weights=None):
    """Swing of each call ÷ the average swing of calls: Tango's normalisation applied to
    a single pitch instead of a plate appearance."""
    d = np.abs(np.asarray(dwp, float))
    w = np.ones_like(d) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(d) & np.isfinite(w)
    return d / (np.sum(d[ok] * w[ok]) / np.sum(w[ok]))


__all__ = ["Leverage", "pa_swing", "call_leverage", "BANDS", "PUBLISHED_BY_INNING_2013",
           "PUBLISHED_TOP_BOT_9_2013", "PUBLISHED_FIRST_PLAY_9_2013"]
