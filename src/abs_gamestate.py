"""
abs_gamestate.py — Week 10. Every pitch as a GAME STATE, and what happened next.

Win probability and run expectancy are both questions about game states: "from
here, how often does the home team win?" and "from here, how many more runs
does the batting team score this inning?" This module turns Statcast's pitch
rows into those states and records, for every pitch, where the game went next.

THE STATE (what V(s) is a function of)

    inning      1..9, and extras (10+) treated as one repeating inning
    half        top (away team bats) or bottom (home team bats)
    outs        0, 1, 2
    bases       8 combinations of runners: empty, 1B, 2B, 1B 2B, 3B, 1B 3B,
                2B 3B, loaded — coded 0..7 as 1·(1B) + 2·(2B) + 4·(3B)
    count       12 combinations: balls 0..3 × strikes 0..2 — coded 3·balls + strikes
    score diff  home runs − away runs, before the pitch

    10 innings × 2 halves × 3 outs × 8 bases × 12 counts × 61 score differences
    (−30..+30) = 351,360 states. Most never happen; the model still needs an
    answer for them, which is why Week 10 has a model and not only a table.

WHAT HAPPENED NEXT (three horizons, all read off the data, none assumed)

    next pitch          the state at the very next pitch — used for the CALL
                        TRANSITIONS: if this called strike stands, where does the
                        game go? if it's overturned, where?
    end of the PA       outs, bases and runs when this plate appearance is over —
                        used to build the run-expectancy chain (RE288)
    end of the inning   runs the batting team scored from this pitch to the end of
                        the half-inning — used to check that chain directly

The data used for estimating "what happens next" is innings 1–8: in the 9th
and later a walk-off can end the half-inning before three outs, which would
make the end of the inning look less productive than it is (the standard
choice in run-expectancy work).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .abs_features import read_any

N_OUTS, N_BASES, N_COUNTS = 3, 8, 12
COUNTS = [(b, s) for b in range(4) for s in range(3)]                 # index 3·b + s
BASE_NAMES = ["empty", "1B", "2B", "1B 2B", "3B", "1B 3B", "2B 3B", "loaded"]
COUNT_NAMES = [f"{b}-{s}" for b, s in COUNTS]
CALLED = {"called_strike": 1, "ball": 0, "blocked_ball": 0}            # recorded call: 1 = strike
FIT_INNINGS = 8                       # transitions and run expectancy come from innings 1–8

K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk"}
EVENT_CLASS = {
    "strikeout": "K", "strikeout_double_play": "K",
    "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
}
RUN_EVENTS = ("caught_stealing", "pickoff", "other_out", "stolen_base", "wild_pitch",
              "passed_ball", "truncated_pa", "runner_double_play")
CLASSES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT", "RUN"]

REQUIRED = ["game_pk", "game_date", "at_bat_number", "pitch_number", "inning",
            "inning_topbot", "outs_when_up", "balls", "strikes", "on_1b", "on_2b", "on_3b",
            "home_score", "away_score", "post_home_score", "post_away_score",
            "events", "description"]
OPTIONAL = ["home_team", "away_team", "game_type", "type", "bat_score", "fld_score",
            "post_bat_score", "home_win_exp", "bat_win_exp", "delta_run_exp",
            "delta_home_win_exp"]


def count_index(balls, strikes):
    return 3 * np.asarray(balls, int) + np.asarray(strikes, int)


def bases_code(on_1b, on_2b, on_3b):
    """Runner columns (an MLBAM id, or empty) -> 0..7. Also accepts True/False."""
    def occ(v):
        v = pd.Series(v)
        if v.dtype == bool:
            return v.to_numpy(int)
        return v.notna().to_numpy(int) & ~v.isin([False, 0, "False", "false", ""]).to_numpy()
    return (occ(on_1b) * 1 + occ(on_2b) * 2 + occ(on_3b) * 4).astype(int)


def event_class(events):
    ev = pd.Series(events, dtype=object)
    out = ev.map(EVENT_CLASS)
    is_run = ev.fillna("").astype(str).str.startswith(RUN_EVENTS)
    out = out.where(out.notna(), np.where(is_run, "RUN", "OUT"))
    return out.where(ev.notna(), None)


# ----------------------------------------------------------------- loading
def season_files(data_dir, season):
    d = Path(data_dir) / "raw" / "statcast"
    files = sorted(d.glob(f"statcast_{season}_*.parquet"))
    stems = {f.stem for f in files}
    files += [f for f in sorted(d.glob(f"statcast_{season}_*.csv")) if f.stem not in stems]
    return [f for f in files if not f.name.endswith("_columns.txt")]


def available_seasons(data_dir):
    d = Path(data_dir) / "raw" / "statcast"
    seasons = set()
    for f in list(d.glob("statcast_*_*.parquet")) + list(d.glob("statcast_*_*.csv")):
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            seasons.add(int(parts[1]))
    return sorted(seasons)


def load_pitches(data_dir, seasons=(2026,)):
    """
    Every regular-season pitch for the chosen seasons, straight from the raw
    Statcast pulls (script 01): Week 10 needs every pitch — swings, balls in play,
    the score after each pitch — not only the called pitches in the cohort.
    """
    frames, missing_files = [], []
    for season in seasons:
        files = season_files(data_dir, season)
        if not files:
            missing_files.append(season)
            continue
        for f in files:
            got = read_any(f)
            if got is None:
                got = pd.read_csv(f) if f.suffix == ".csv" else None
            if got is not None and len(got):
                got = got[[c for c in REQUIRED + OPTIONAL if c in got.columns]].copy()
                got["season"] = int(season)
                frames.append(got)
    if not frames:
        raise SystemExit(f"  No Statcast files for season(s) {list(seasons)} under "
                         f"{Path(data_dir) / 'raw' / 'statcast'}. Run script 01 first.")
    df = pd.concat(frames, ignore_index=True)
    lacking = [c for c in REQUIRED if c not in df.columns]
    if lacking:
        raise SystemExit(
            f"  The Statcast files are missing {lacking}. Week 10 needs the score after "
            "each pitch (post_home_score, post_away_score), the runners (on_1b/2b/3b) and "
            "the plate-appearance result (events) — every Savant CSV has them. Re-pull with "
            "script 01 if your files were trimmed.")
    if "game_type" in df.columns:
        df = df[df["game_type"].fillna("R").astype(str).str.upper().eq("R")]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = (df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="last")
            .sort_values(["game_pk", "at_bat_number", "pitch_number"])
            .reset_index(drop=True))
    return df, missing_files


# ----------------------------------------------------------------- states
def build_states(raw):
    """
    raw Statcast pitches -> (pitches, games).

    pitches: one row per pitch with the state before it (inning, half, outs,
    bases, count, score diff), the recorded and the EFFECTIVE call on called
    pitches, and three "what happened next" horizons (see the module docstring).
    games: one row per game with the final score, who won, and whether the game
    is usable for win probability (9+ innings, a winner).
    """
    df = raw.copy()
    essential = ["inning", "outs_when_up", "balls", "strikes", "home_score", "away_score"]
    for c in essential:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[essential].isna().any(axis=1)
    n_dropped = int(bad.sum())
    if n_dropped:                       # a handful of rows with no count or score: unusable
        df = df[~bad].reset_index(drop=True)
    df["half"] = df["inning_topbot"].astype(str).str.lower().str.startswith("bot").astype(int)
    df["outs"] = pd.to_numeric(df["outs_when_up"], errors="coerce").astype(int)
    df["bases"] = bases_code(df["on_1b"], df["on_2b"], df["on_3b"])
    df["balls"] = pd.to_numeric(df["balls"], errors="coerce").clip(0, 3).astype(int)
    df["strikes"] = pd.to_numeric(df["strikes"], errors="coerce").clip(0, 2).astype(int)
    df["count"] = count_index(df["balls"], df["strikes"])
    for c in ("home_score", "away_score", "post_home_score", "post_away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # a missing post-pitch score: the next pitch's pre-pitch score is the same number
    nxt = df.groupby("game_pk")[["home_score", "away_score"]].shift(-1)
    df["post_home_score"] = df["post_home_score"].fillna(nxt["home_score"]).fillna(df["home_score"])
    df["post_away_score"] = df["post_away_score"].fillna(nxt["away_score"]).fillna(df["away_score"])
    df["diff"] = (df["home_score"] - df["away_score"]).astype(int)
    bot = df["half"].eq(1)
    df["bat_runs"] = np.where(bot, df["home_score"], df["away_score"]).astype(int)
    df["bat_runs_post"] = np.where(bot, df["post_home_score"], df["post_away_score"]).astype(int)
    df["call_recorded"] = df["description"].map(CALLED)
    df["is_called"] = df["call_recorded"].notna()

    # ---- games: final score and winner
    g = df.groupby("game_pk").agg(
        season=("season", "first"), game_date=("game_date", "first"),
        home_team=("home_team", "first") if "home_team" in df.columns else ("season", "first"),
        away_team=("away_team", "first") if "away_team" in df.columns else ("season", "first"),
        final_home=("post_home_score", "max"), final_away=("post_away_score", "max"),
        last_inning=("inning", "max"), n_pitches=("pitch_number", "size"))
    g["home_win"] = (g["final_home"] > g["final_away"]).astype(int)
    tied = g["final_home"] == g["final_away"]
    short = g["last_inning"] < 9
    g["usable"] = ~tied & ~short
    g["why_not"] = np.where(tied, "no winner in the data (suspended or incomplete)",
                            np.where(short, "shorter than 9 innings (shortened game)", ""))
    games = g.reset_index()
    df = df.merge(games[["game_pk", "home_win", "usable"]], on="game_pk", how="left")

    # ---- keys for the plate appearance and the half-inning
    df["hi_key"] = (df["game_pk"].astype(np.int64) * 100 + df["inning"].astype(int) * 2
                    + df["half"])
    same_game = df["game_pk"].eq(df["game_pk"].shift(-1))
    same_pa = same_game & df["at_bat_number"].eq(df["at_bat_number"].shift(-1))
    same_hi = df["hi_key"].eq(df["hi_key"].shift(-1))

    # end of each half-inning: runs the batting team had when it ended, and
    # whether it ended normally (three outs) — a walk-off or a suspended game doesn't
    hi_end_runs = df.groupby("hi_key")["bat_runs_post"].transform("max")
    df["runs_rest_inning"] = (hi_end_runs - df["bat_runs"]).astype(int)

    # ---- the plate appearance: first pitch, and the state when it is over
    pa_first = ~(df["game_pk"].eq(df["game_pk"].shift(1))
                 & df["at_bat_number"].eq(df["at_bat_number"].shift(1)))
    df["pa_start"] = pa_first
    pa_idx = np.flatnonzero(pa_first.to_numpy())
    pa_id = np.cumsum(pa_first.to_numpy()) - 1                    # which PA each pitch is in
    # the next PA's first pitch, if it is in the same half-inning
    nxt_first = np.append(pa_idx[1:], -1)
    hk = df["hi_key"].to_numpy()
    same_hi_next_pa = np.zeros(len(pa_idx), bool)
    ok = nxt_first >= 0
    same_hi_next_pa[ok] = hk[nxt_first[ok]] == hk[pa_idx[ok]]
    outs_a, bases_a = df["outs"].to_numpy(), df["bases"].to_numpy()
    runs_a = df["bat_runs"].to_numpy()
    pa_end_outs = np.where(same_hi_next_pa, outs_a[np.where(ok, nxt_first, 0)], 3)
    pa_end_bases = np.where(same_hi_next_pa, bases_a[np.where(ok, nxt_first, 0)], 0)
    pa_end_runs = np.where(same_hi_next_pa, runs_a[np.where(ok, nxt_first, 0)],
                           hi_end_runs.to_numpy()[pa_idx])
    df["pa_end_outs"] = pa_end_outs[pa_id]
    df["pa_end_bases"] = pa_end_bases[pa_id]
    df["runs_rest_pa"] = (pa_end_runs[pa_id] - runs_a).astype(int)
    last_of_pa = ~same_pa
    ev_last = df["events"].where(last_of_pa)
    df["pa_events"] = ev_last.groupby(pa_id).transform("last")
    df["pa_class"] = event_class(df["pa_events"])

    # ---- the very next pitch (for the call transitions)
    n_outs = df["outs"].shift(-1)
    n_bases = df["bases"].shift(-1)
    n_count = df["count"].shift(-1)
    n_runs = df["bat_runs"].shift(-1)
    df["next_absorb"] = ~(same_pa | same_hi)
    df["next_pa_end"] = ~same_pa
    df["next_outs"] = np.where(df["next_absorb"], 3, n_outs.fillna(3)).astype(int)
    df["next_bases"] = np.where(df["next_absorb"], 0, n_bases.fillna(0)).astype(int)
    df["next_count"] = np.where(same_pa, n_count.fillna(0), 0).astype(int)
    df["runs_next"] = np.where(df["next_absorb"], hi_end_runs - df["bat_runs"],
                               n_runs.fillna(0) - df["bat_runs"]).astype(int)

    # ---- the EFFECTIVE call: what the count did next, not what was written down
    # (on real 2026 data the recorded call on a challenged pitch is the FINAL one;
    # a demo season may record the ORIGINAL one — the count tells the truth either way)
    nb, ns = df["balls"].shift(-1), df["strikes"].shift(-1)
    went_strike = same_pa & ns.eq(df["strikes"] + 1) & nb.eq(df["balls"])
    went_ball = same_pa & nb.eq(df["balls"] + 1) & ns.eq(df["strikes"])
    ended = ~same_pa
    eff = pd.Series(np.nan, index=df.index)
    eff[went_strike] = 1
    eff[went_ball] = 0
    eff[ended & df["events"].isin(K_EVENTS)] = 1
    eff[ended & df["events"].isin(BB_EVENTS)] = 0
    df["call_effective"] = eff.where(df["is_called"]).fillna(df["call_recorded"])
    df["fit_row"] = df["usable"].astype(bool) & (df["inning"] <= FIT_INNINGS)
    df.attrs["rows_dropped"] = n_dropped
    return df, games


def pa_starts(pitches):
    """One row per plate appearance: the state when it began (count 0-0)."""
    return pitches[pitches["pa_start"]].reset_index(drop=True)


def describe_states(pitches, games):
    """Numbers for the pre-flight and the report."""
    use = games[games["usable"]]
    called = pitches[pitches["is_called"]]
    agree = (called["call_effective"] == called["call_recorded"]).mean() if len(called) else np.nan
    per_game = use["final_home"] + use["final_away"]
    return {
        "seasons": sorted(int(s) for s in games["season"].unique()),
        "n_pitches": int(len(pitches)), "n_games": int(len(games)),
        "n_usable": int(len(use)), "n_excluded": int((~games["usable"]).sum()),
        "excluded_why": games.loc[~games["usable"], "why_not"].value_counts().to_dict(),
        "n_pa": int(pitches["pa_start"].sum()),
        "home_win_rate": float(use["home_win"].mean()) if len(use) else np.nan,
        "runs_per_team_game": float(per_game.mean() / 2) if len(use) else np.nan,
        "extra_inning_share": float((use["last_inning"] > 9).mean()) if len(use) else np.nan,
        "n_called": int(len(called)), "call_agreement": float(agree),
        "rows_dropped": int(pitches.attrs.get("rows_dropped", 0)),
    }


__all__ = ["N_OUTS", "N_BASES", "N_COUNTS", "COUNTS", "BASE_NAMES", "COUNT_NAMES", "CLASSES",
           "FIT_INNINGS", "count_index", "bases_code", "event_class", "available_seasons",
           "load_pitches", "build_states", "pa_starts", "describe_states"]
