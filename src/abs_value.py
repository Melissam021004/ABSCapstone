"""
abs_value.py — Week 10. What the 2026 challenges were worth, at every level.

Week 10's V(s) turns a call into a number: the win probability (and the runs)
it moves. This module applies that number, with Week 9's p(s), to every called
pitch of the season and asks the questions the challenge decision raises:

    INNING   how many runs does an overturn change before the third out? (the
             count, the outs and the runners decide it) — and do the runs that
             actually followed overturned vs upheld challenges match?
    GAME     how much win probability does each overturn move?
    AHEAD    how much is still at stake later in the game — the raw material of
             "the value of holding a challenge" that Week 11's MDP turns into a
             policy
    SERIES   did a challenge swing a series? (only when its game decided it)
    SEASON   wins each team added with challenges, and wins it left on the table
    PEOPLE   do some players challenge better — and does WHO challenges change
             what ABS rules? do some umpires miss more, and does that change how
             much a challenge in hand is worth?

Every value is "expected wins" in the win-probability-added sense: a challenge
that moves a team from 40% to 43% to win is worth 0.03 wins. Summed over a
season they are wins in the ordinary sense.

The two probabilities used throughout:
    p_s   p(s), Week 9's locked model: the chance ABS overturns this call if it is
          challenged (location only — ABS sees only the ball). For an unchallenged
          pitch it is the chance the call was wrong.
    dwp   V(s): win probability the CHALLENGING side gains if the call is
          overturned (the batting side on a called strike, the fielding side on a
          called ball).
So p_s × dwp is what challenging that pitch was worth on average, before
counting the cost of losing the challenge if it fails (Week 11 prices that).
"""

import numpy as np
import pandas as pd

from .abs_metrics import wilson

KEY = ["game_pk", "at_bat_number", "pitch_number"]
SIDE_LABEL = {"batting": "batting side (called strikes)", "fielding": "fielding side (called balls)"}


# =================================================================== valuation
def _in_range(df):
    ok = pd.Series(True, index=df.index)
    for c, lo, hi in (("plate_x", -5, 5), ("plate_z", -2.5, 8), ("abs_zone_top", 1, 6),
                      ("abs_zone_bot", 0.3, 4)):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            ok &= v.between(lo, hi)
    if "batter_height_ft" in df.columns:                 # missing is fine: the zone is given
        v = pd.to_numeric(df["batter_height_ft"], errors="coerce")
        ok &= v.isna() | v.between(4.5, 7.6)
    return ok


def valuation_table(cohort, pm, wp):
    """
    Every called pitch in the cohort, valued. `cohort` comes from add_features()
    (so `called_strike` is the umpire's ORIGINAL call); pm is the locked p(s),
    wp the locked V(s).
    """
    from .abs_predict import states_from_cohort
    c = cohort.reset_index(drop=True)
    ok = _in_range(c)
    p_s = np.full(len(c), np.nan)
    if ok.any():
        p_s[ok.to_numpy()] = pm.predict_frame(states_from_cohort(c[ok]), strict=False)
    strike = c["called_strike"].astype(bool).to_numpy()
    bat_home = c["inning_topbot"].astype(str).str.lower().str.startswith("bot").to_numpy()
    st = pd.DataFrame({"inning": c["inning"], "half": np.where(bat_home, "bot", "top"),
                       "outs": c["outs_when_up"], "on_1b": c["on_1b"], "on_2b": c["on_2b"],
                       "on_3b": c["on_3b"], "home_score": c["home_score"],
                       "away_score": c["away_score"], "balls": c["balls"], "strikes": c["strikes"]})
    cv = wp.call_value(st, strike)
    side = np.where(strike, "batting", "fielding")
    chall_home = np.where(strike, bat_home, ~bat_home)
    rem = np.where(strike, c["challenges_remaining_batting"], c["challenges_remaining_fielding"])
    ppp = c["position_player_pitching"].astype(bool).to_numpy() \
        if "position_player_pitching" in c.columns else np.zeros(len(c), bool)
    chal = c["was_challenged"].astype(bool).to_numpy()
    ovr = c["is_overturned"].map(lambda v: str(v).lower() in ("true", "1", "1.0")).to_numpy() & chal
    wps, wpo = cv["wp_stands"].to_numpy(), cv["wp_overturned"].to_numpy()
    druns_bat = cv["druns_batting"].to_numpy()
    out = pd.DataFrame({
        **{k: c[k].to_numpy() for k in KEY},
        "game_date": pd.to_datetime(c["game_date"]).to_numpy(),
        "inning": c["inning"].astype(int).to_numpy(), "bat_home": bat_home,
        "half_index": np.minimum((c["inning"].astype(int) - 1) * 2 + bat_home.astype(int), 18),
        "home_team": c["home_team"].to_numpy(), "away_team": c["away_team"].to_numpy(),
        "call_strike": strike, "side": side,
        "chall_team": np.where(chall_home, c["home_team"], c["away_team"]),
        "other_team": np.where(chall_home, c["away_team"], c["home_team"]),
        "chall_home": chall_home, "rem": rem, "opportunity": (rem > 0) & ~ppp,
        "p_s": p_s, "dwp": cv["dwp_challenger"].to_numpy(),
        "druns_chall": np.where(strike, druns_bat, -druns_bat),
        "wp_stands_chall": np.where(chall_home, wps, 1 - wps),
        "wp_over_chall": np.where(chall_home, wpo, 1 - wpo),
        "re_stands": cv["re_stands"].to_numpy(), "re_over": cv["re_overturned"].to_numpy(),
        "challenged": chal, "overturned": ovr,
        "role": c["challenger_role"].fillna("").astype(str).to_numpy()
        if "challenger_role" in c.columns else "",
        "challenger_id": c["challenger_id"].to_numpy() if "challenger_id" in c.columns else np.nan,
        "umpire_id": c["hp_umpire_id"].to_numpy() if "hp_umpire_id" in c.columns else np.nan,
        "umpire": c["hp_umpire"].to_numpy() if "hp_umpire" in c.columns else "",
        "abs_dist_in": pd.to_numeric(c["zone_dist_in"], errors="coerce").abs().to_numpy()
        if "zone_dist_in" in c.columns else np.nan,
        "split": c["split"].to_numpy() if "split" in c.columns else "",
    })
    out["ev"] = out["p_s"] * out["dwp"]
    out["gain"] = np.where(out["overturned"], out["dwp"], 0.0)
    out["runs_gain"] = np.where(out["overturned"], out["druns_chall"], 0.0)
    return out


# =================================================================== bootstrap helpers
def game_mean_ci(values, games, B=1000, seed=0):
    """Mean with a 95% interval from resampling games."""
    v = np.asarray(values, float)
    g = np.asarray(games)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if not len(v):
        return np.nan, (np.nan, np.nan)
    uniq, inv = np.unique(g, return_inverse=True)
    s, n = np.bincount(inv, v, len(uniq)), np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        draws.append((cnt * s).sum() / max((cnt * n).sum(), 1))
    return float(v.mean()), (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def game_diff_ci(values, groups_a, games, B=1000, seed=0):
    """mean(values | a) − mean(values | not a), with a game-level bootstrap interval."""
    v = np.asarray(values, float)
    a = np.asarray(groups_a, bool)
    g = np.asarray(games)
    ok = np.isfinite(v)
    v, a, g = v[ok], a[ok], g[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    G = len(uniq)
    sa, na = np.bincount(inv, v * a, G), np.bincount(inv, a.astype(float), G)
    sb, nb = np.bincount(inv, v * ~a, G), np.bincount(inv, (~a).astype(float), G)
    point = sa.sum() / max(na.sum(), 1) - sb.sum() / max(nb.sum(), 1)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, G, G), minlength=G)
        draws.append((cnt * sa).sum() / max((cnt * na).sum(), 1)
                     - (cnt * sb).sum() / max((cnt * nb).sum(), 1))
    return float(point), (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


# =================================================================== natural experiments
def natural_experiment(ch, actual, pred_over, pred_stand, B=1000, seed=0):
    """
    Challenges are close to a natural experiment: whether one is overturned is
    decided by where the ball was, not by the inning, the score or the runners.
    So the gap between what happened after OVERTURNED and after UPHELD challenges
    estimates what an overturn does — and the model's predictions for each branch
    should match what happened in each group.
    """
    rows = []
    for side in ("batting", "fielding", "all"):
        d = ch if side == "all" else ch[ch["side"] == side]
        if len(d) < 20 or d["overturned"].nunique() < 2:
            continue
        a = d[actual].to_numpy(float)
        ov = d["overturned"].to_numpy(bool)
        realized_pred = np.where(ov, d[pred_over], d[pred_stand])
        diff, dci = game_diff_ci(a, ov, d["game_pk"], B=B, seed=seed)
        pdiff = float(realized_pred[ov].mean() - realized_pred[~ov].mean())
        effect = float((d[pred_over] - d[pred_stand]).mean())
        resid, rci = game_mean_ci(a - realized_pred, d["game_pk"], B=B, seed=seed)
        rows.append({"side": side, "n": len(d), "n_overturned": int(ov.sum()),
                     "actual_overturned": float(a[ov].mean()), "actual_upheld": float(a[~ov].mean()),
                     "actual_diff": diff, "actual_diff_lo": dci[0], "actual_diff_hi": dci[1],
                     "pred_overturned": float(realized_pred[ov].mean()),
                     "pred_upheld": float(realized_pred[~ov].mean()), "pred_diff": pdiff,
                     "model_effect": effect, "resid": resid, "resid_lo": rci[0], "resid_hi": rci[1]})
    return pd.DataFrame(rows)


# =================================================================== value still ahead
def value_ahead(val, team_tier=None, max_half=18):
    """
    For every game and team, at the start of each half-inning: how much win
    probability is still at stake in the calls that will go AGAINST that team
    later in the game.

        stakes    sum of p_s × dwp over the remaining calls against the team —
                  the expected value of all the wrong calls still to come
        best1     the single most valuable wrong call still to come (dwp of the
                  best remaining call with p_s >= 0.5) — what ONE challenge saved
                  for later could win with perfect foresight
        best2     the two most valuable — what both challenges could win

    best1 is an upper bound on the value of holding a challenge: a real team
    doesn't know which call will be the big one, and a failed challenge is lost.
    Week 11's backward induction computes the real value; this is its ceiling.
    """
    v = val[np.isfinite(val["p_s"]) & np.isfinite(val["dwp"])]
    v = v.assign(wrong_val=np.where(v["p_s"] >= 0.5, v["dwp"], 0.0),
                 stake=v["p_s"] * v["dwp"])
    rows = []
    for (g, team), d in v.groupby(["game_pk", "chall_team"], sort=False):
        hi = d["half_index"].to_numpy()
        stake = np.bincount(hi, d["stake"].to_numpy(), max_half + 1)
        top = [[] for _ in range(max_half + 1)]
        for h, x in zip(hi, d["wrong_val"].to_numpy()):
            if x > 0:
                top[h].append(x)
        last = int(hi.max())
        acc_s, best = 0.0, []
        tier = team_tier.get((g, team)) if team_tier is not None else None
        for h in range(max_half, -1, -1):
            acc_s += stake[h]
            best = sorted(best + top[h], reverse=True)[:2]
            if h <= last:
                rows.append({"game_pk": g, "team": team, "half_index": h, "stakes": acc_s,
                             "best1": best[0] if best else 0.0,
                             "best2": sum(best[:2]), "tier": tier})
    return pd.DataFrame(rows)


def summarise_ahead(ahead, by=None):
    keys = ["half_index"] + ([by] if by else [])
    t = ahead.groupby(keys).agg(n=("stakes", "size"), stakes=("stakes", "mean"),
                                best1=("best1", "mean"), best2=("best2", "mean")).reset_index()
    return t


# =================================================================== team-season
def team_season(val):
    """Season totals per team, in wins (win probability added) and runs."""
    v = val[np.isfinite(val["p_s"]) & np.isfinite(val["dwp"])]
    rows = []
    for team, d in v.groupby("chall_team"):
        ch = d[d["challenged"]]
        stood = ~d["overturned"]
        against_opp = v[(v["other_team"] == team) & ~v["overturned"]]
        rows.append({
            "team": team, "games": int(d["game_pk"].nunique()),
            "challenges": int(len(ch)), "overturned": int(ch["overturned"].sum()),
            "failed": int((~ch["overturned"]).sum()),
            "success_rate": float(ch["overturned"].mean()) if len(ch) else np.nan,
            "wins_added": float(d["gain"].sum()),
            "runs_added": float(d["runs_gain"].sum()),
            "ev_of_challenges_made": float(ch["ev"].sum()),
            "left_on_table": float(d.loc[d["opportunity"] & ~d["challenged"], "ev"].sum()),
            "lost_when_out": float(d.loc[~d["opportunity"] & stood, "ev"].sum()),
            "lost_to_wrong_calls": float(d.loc[stood, "ev"].sum()),
            "gained_from_wrong_calls": float(against_opp["ev"].sum()),
        })
    t = pd.DataFrame(rows)
    t["umpire_net"] = t["gained_from_wrong_calls"] - t["lost_to_wrong_calls"]
    return t.sort_values("wins_added", ascending=False).reset_index(drop=True)


def league_totals(val):
    v = val[np.isfinite(val["p_s"]) & np.isfinite(val["dwp"])]
    swing = float(v["ev"].sum())
    rec = float(v["gain"].sum())
    return {"called_pitches": int(len(v)), "expected_wrong_calls": float(v["p_s"].sum()),
            "wins_swung_by_wrong_calls": swing, "wins_recovered": rec,
            "share_recovered": rec / swing if swing > 0 else np.nan,
            "overturns": int(v["overturned"].sum()), "challenges": int(v["challenged"].sum()),
            "games": int(v["game_pk"].nunique())}


# =================================================================== series
def series_table(games, gap_days=2):
    """
    Group games into series: consecutive games between the same two teams at the
    same park, no more than `gap_days` apart (doubleheaders stay together).
    """
    g = games[["game_pk", "game_date", "home_team", "away_team", "home_win"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["home_team", "away_team", "game_date", "game_pk"]).reset_index(drop=True)
    new = (g["home_team"].ne(g["home_team"].shift()) | g["away_team"].ne(g["away_team"].shift())
           | (g["game_date"] - g["game_date"].shift()).dt.days.gt(gap_days))
    g["series_id"] = new.cumsum()
    g["game_in_series"] = g.groupby("series_id").cumcount() + 1
    g["series_len"] = g.groupby("series_id")["game_pk"].transform("size")
    return g


def series_effects(val, series):
    """
    For each team in each series: its net challenge win probability in each game
    (its overturns minus its opponent's), and whether that game DECIDED the series
    for it (flipping its result would change the series result: won / split / lost).
    The expected change in the series result is the sum of net challenge value over
    the deciding games.
    """
    gain = val.groupby(["game_pk", "chall_team"])["gain"].sum().to_dict()
    total = val.groupby("game_pk")["gain"].sum().to_dict()
    rows = []
    for sid, d in series.groupby("series_id"):
        n = len(d)

        def result(w):
            return 1 if w * 2 > n else (0 if w * 2 == n else -1)
        for team, is_home in ((d["home_team"].iloc[0], True), (d["away_team"].iloc[0], False)):
            won = (d["home_win"] if is_home else 1 - d["home_win"]).astype(int).to_numpy()
            res = result(won.sum())
            for gk, w in zip(d["game_pk"], won):
                mine = gain.get((gk, team), 0.0)
                rows.append({"series_id": sid, "team": team, "game_pk": gk, "series_len": n,
                             "won_game": int(w), "series_result": res,
                             "pivotal": result(won.sum() - w + (1 - w)) != res,
                             "net_gain": float(mine - (total.get(gk, 0.0) - mine))})
    s = pd.DataFrame(rows)
    by_series = (s.assign(shift=s["net_gain"] * s["pivotal"])
                 .groupby(["series_id", "team"]).agg(series_len=("series_len", "first"),
                                                     result=("series_result", "first"),
                                                     net_gain=("net_gain", "sum"),
                                                     shift=("shift", "sum")).reset_index())
    return s, by_series


# =================================================================== people
def shrink_rates(k, n, groups):
    """Empirical-Bayes shrinkage of success rates toward each group's rate."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    out = np.full(len(k), np.nan)
    prior_m = np.full(len(k), np.nan)
    for g in np.unique(groups):
        m = np.asarray(groups) == g
        mu = k[m].sum() / max(n[m].sum(), 1)
        use = m & (n >= 3)
        r = k[use] / n[use]
        if use.sum() >= 5:
            var_obs = np.average((r - mu) ** 2, weights=n[use])
            tau2 = max(var_obs - mu * (1 - mu) / np.mean(n[use]), 1e-5)
            M = float(np.clip(mu * (1 - mu) / tau2 - 1, 5, 500))
        else:
            M = 50.0
        out[m] = (k[m] + M * mu) / (n[m] + M)
        prior_m[m] = M
    return out, prior_m


def cluster_signal(values, labels, strata=None, n_perm=999, seed=0, min_n=5, units=None):
    """
    Do the clusters (players, umpires) differ more than chance? The statistic is
    the between-cluster spread of the mean, sum n_j (mean_j − stratum mean)^2,
    over clusters with at least `min_n` rows. Its null distribution comes from
    shuffling the cluster labels — within strata (e.g. challenger role), and
    whole `units` at a time when rows come in groups (an umpire's pitches come in
    games, so games are shuffled between umpires, not single pitches).

    Returns (statistic, p-value, share of the spread beyond chance, sd): `sd` is
    the size of the real differences — the standard deviation of the clusters'
    true means around their stratum's mean, after removing what chance alone
    would produce (0 when the spread is no bigger than chance).
    """
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    lab = pd.factorize(pd.Series(labels).astype(str).to_numpy()[ok])[0]
    st = (np.zeros(ok.sum(), int) if strata is None
          else pd.factorize(pd.Series(strata).astype(str).to_numpy()[ok])[0])
    v = v[ok]
    if units is None:
        S_u, N_u, L_u, T_u = v, np.ones(len(v)), lab, st
    else:
        u = pd.factorize(pd.Series(np.asarray(units)[ok]))[0]
        S_u = np.bincount(u, v)
        N_u = np.bincount(u).astype(float)
        first = pd.Series(np.arange(len(u))).groupby(u).first().to_numpy()
        L_u, T_u = lab[first], st[first]
    nL, nS = int(lab.max()) + 1, int(st.max()) + 1
    m_s = np.bincount(T_u, S_u, nS) / np.maximum(np.bincount(T_u, N_u, nS), 1)
    s_of = np.arange(nS * nL) // nL

    def stat(L, with_n=False):
        key = T_u * nL + L
        S = np.bincount(key, S_u, nS * nL)
        N = np.bincount(key, N_u, nS * nL)
        m = N >= min_n
        if not m.any():
            return (np.nan, 0.0) if with_n else np.nan
        t = float((N[m] * (S[m] / N[m] - m_s[s_of[m]]) ** 2).sum())
        return (t, float(N[m].sum())) if with_n else t

    rng = np.random.default_rng(seed)
    obs, n_in = stat(L_u, with_n=True)
    groups = [np.flatnonzero(T_u == k) for k in range(nS)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        L = L_u.copy()
        for g in groups:
            L[g] = L_u[g][rng.permutation(len(g))]
        null[i] = stat(L)
    p = float((1 + np.sum(null >= obs)) / (1 + n_perm))
    excess = float(1 - np.nanmean(null) / obs) if obs and np.isfinite(obs) else np.nan
    sd = float(np.sqrt(max(obs - np.nanmean(null), 0.0) / n_in)) if n_in else np.nan
    return obs, p, excess, sd


def split_half(values, labels, order, min_n=10):
    """
    Reliability: split each cluster's rows into odd and even (in time order),
    correlate the two halves' means across clusters, and step it up to the full
    sample (Spearman–Brown). 0 = pure noise; 1 = a perfectly stable trait.
    """
    df = pd.DataFrame({"v": np.asarray(values, float), "l": pd.Series(labels).astype(str).to_numpy(),
                       "o": np.asarray(order)}).dropna(subset=["v"])
    df = df.sort_values(["l", "o"])
    df["half"] = df.groupby("l").cumcount() % 2
    g = df.groupby(["l", "half"])["v"].agg(["mean", "size"]).unstack()
    g = g[(g[("size", 0)] >= min_n // 2) & (g[("size", 1)] >= min_n // 2)]
    if len(g) < 5:
        return np.nan, int(len(g))
    r = float(np.corrcoef(g[("mean", 0)], g[("mean", 1)])[0, 1])
    return float(2 * r / (1 + r)) if r > -1 else np.nan, int(len(g))


def challenger_table(ch, min_n=1):
    """Per challenger: challenges, overturns, what the pitches they chose predicted."""
    d = ch[np.isfinite(ch["p_s"]) & ch["challenger_id"].notna()]
    t = d.groupby(["challenger_id", "role"]).agg(
        n=("overturned", "size"), overturned=("overturned", "sum"),
        expected=("p_s", "sum"), mean_p=("p_s", "mean"), value=("gain", "sum"),
        ev=("ev", "sum")).reset_index()
    t = t[t["n"] >= min_n]
    t["rate"] = t["overturned"] / t["n"]
    t["oae"] = t["overturned"] - t["expected"]
    t["rate_shrunk"], _ = shrink_rates(t["overturned"], t["n"], t["role"])
    return t.sort_values("n", ascending=False).reset_index(drop=True)


def difficulty_baseline(val, step=0.25, cap=6.0):
    """League chance a call is wrong at each distance from the zone edge (|in|)."""
    d = val[np.isfinite(val["p_s"]) & np.isfinite(val["abs_dist_in"])]
    b = np.minimum(np.floor(d["abs_dist_in"] / step), cap / step).astype(int)
    q = d.groupby(b)["p_s"].mean()
    allb = np.minimum(np.floor(val["abs_dist_in"].fillna(cap) / step), cap / step).astype(int)
    return allb.map(q).fillna(0.0).to_numpy()


def umpire_table(val):
    """Per umpire: how many calls ABS would reverse, adjusted for how hard the pitches were."""
    v = val[np.isfinite(val["p_s"])].copy()
    v["q"] = difficulty_baseline(v)
    v["excess"] = v["p_s"] - v["q"]
    v["ev_opp"] = np.where(v["opportunity"], v["ev"], 0.0)
    g = v.groupby(["umpire_id", "umpire"]).agg(
        games=("game_pk", "nunique"), called=("p_s", "size"), wrong=("p_s", "sum"),
        expected_avg_ump=("q", "sum"), stakes=("ev", "sum"),
        challengeable=("ev_opp", "sum"), overturns=("overturned", "sum")).reset_index()
    g["wrong_per_game"] = g["wrong"] / g["games"]
    g["excess_per_100"] = 100 * (g["wrong"] - g["expected_avg_ump"]) / g["called"]
    g["stakes_per_game"] = g["stakes"] / g["games"]
    g["challengeable_per_game"] = g["challengeable"] / g["games"]
    return g.sort_values("excess_per_100").reset_index(drop=True), v


__all__ = ["valuation_table", "game_mean_ci", "game_diff_ci", "natural_experiment",
           "value_ahead", "summarise_ahead", "team_season", "league_totals", "series_table",
           "series_effects", "shrink_rates", "cluster_signal", "split_half",
           "challenger_table", "difficulty_baseline", "umpire_table"]
