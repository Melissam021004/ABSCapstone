#!/usr/bin/env python3
"""
03_savant_abs_leaderboards.py — Baseball Savant ABS challenge leaderboards.

These are AGGREGATES (player-season / team-season), not pitch-level data.
Two things they're good for:

  1. VALIDATION. Your pitch-level challenge counts from script 02 should add up
     to Savant's totals. If they don't, your extraction is wrong.
  2. BENCHMARKING. Savant publishes "expected challenges" and "overturns vs
     expected" — the descriptive baseline your novelty claim is measured against.

Do NOT feed Savant's expected-challenge metrics into your model as features.
They are built from the outcome you are predicting.

-------------------------------------------------------------------------------
WHY THIS SCRIPT WAS REWRITTEN

The obvious approach — appending `&csv=true`, which works on most other Savant
leaderboards — returns **HTTP 500** on the ABS leaderboard. It is not rate
limiting and it is not your network; the parameter is simply not supported here.

So this version fetches the normal HTML page and pulls the rows out of the JSON
that Savant embeds in a <script> tag to render the table. That JSON is the same
data the table shows.

Strategies are tried in order and each one reports exactly what happened:
    1. HTML page -> embedded JSON            (primary)
    2. csv=true                              (in case MLB fixes it)
    3. candidate JSON API endpoints          (long shot)
-------------------------------------------------------------------------------

    python 03_savant_abs_leaderboards.py
    python 03_savant_abs_leaderboards.py --debug        # dump what came back
    python 03_savant_abs_leaderboards.py --only catcher # one view, for testing

Output: data/raw/savant/abs_leaderboard_<type>_<mode>_<level>_<year>.csv
        data/raw/savant/abs_leaderboard_summary.txt
        data/raw/savant/_debug/    (with --debug)
"""

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
OUT_DIR = DATA_DIR / "raw" / "savant"
DEBUG_DIR = OUT_DIR / "_debug"

URL = "https://baseballsavant.mlb.com/leaderboard/abs-challenges"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

CHALLENGE_TYPES = ["batter", "catcher", "pitcher", "team-summary", "catching-team"]
DATA_MODES = ["for", "against"]

# Transient -> worth retrying. 500 is NOT here: on this endpoint it means the
# parameter combination is rejected, and retrying just wastes 15 seconds.
RETRY_STATUS = {429, 502, 503, 504}


def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,text/csv,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": URL,
    })
    return s


def base_params(year, challenge_type, mode, level, game_type):
    return {
        "gameType": game_type,
        "year": year,
        "challengeType": challenge_type,
        "level": level,
        "minChal": 1,
        "minOppChal": 0,
        "dataMode": mode,
        "page": 0,
        "pageSize": 1000,
    }


# ---------------------------------------------------------------- JSON digging
def _match_bracket(s, start):
    """Index of the bracket closing the one at `start`, string-literal aware."""
    open_ch = s[start]
    close_ch = {"[": "]", "{": "}"}[open_ch]
    depth, in_str, esc, quote = 0, False, False, None
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                in_str = False
            continue
        if c in "\"'":
            in_str, quote = True, c
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_embedded_json(html):
    """
    Find JSON arrays-of-objects that Savant embeds in <script> tags.
    Returns [(variable_name, parsed_rows), ...] best (longest) first.
    """
    found = []

    # var NAME = [...]  /  let/const NAME = {...}  /  NAME = [...]
    for m in re.finditer(r"(?:var|let|const)?\s*([A-Za-z_$][\w$]*)\s*=\s*([\[{])", html):
        name, start = m.group(1), m.start(2)
        end = _match_bracket(html, start)
        # _rows_from() does the real filtering; this only skips trivially short
        # matches so we aren't json-parsing every `x=[]` on the page.
        if end == -1 or end - start < 20:
            continue
        blob = html[start:end + 1]
        try:
            parsed = json.loads(blob)
        except Exception:
            continue
        rows = _rows_from(parsed)
        if rows:
            found.append((name, rows))

    # JSON.parse('...') / JSON.parse("...")
    for m in re.finditer(r"JSON\.parse\(\s*(['\"])", html):
        q = m.group(1)
        i = m.end(1)
        out, esc = [], False
        while i < len(html):
            c = html[i]
            if esc:
                out.append(c)
                esc = False
            elif c == "\\":
                out.append(c)
                esc = True
            elif c == q:
                break
            else:
                out.append(c)
            i += 1
        try:
            parsed = json.loads(json.loads(q + "".join(out) + q))
        except Exception:
            continue
        rows = _rows_from(parsed)
        if rows:
            found.append(("JSON.parse", rows))

    found.sort(key=lambda t: len(t[1]), reverse=True)
    return found


def _rows_from(parsed):
    """Coerce a parsed blob into a list of row dicts, or None."""
    if isinstance(parsed, list):
        # >= 1, not >= 2: a tightly filtered leaderboard can legitimately
        # return a single row, and silently dropping it would look like failure.
        if parsed and all(isinstance(r, dict) for r in parsed[:5]):
            return parsed
        return None
    if isinstance(parsed, dict):
        for key in ("data", "rows", "leaderboard", "results", "players", "leaders"):
            v = parsed.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        # one level deeper
        for v in parsed.values():
            if isinstance(v, dict):
                r = _rows_from(v)
                if r:
                    return r
    return None


# ---------------------------------------------------------------- strategies
def get(session, params, retries=3, timeout=90):
    """Returns (response, error_string). Reports the ACTUAL status code."""
    last = None
    for attempt in range(retries):
        try:
            r = session.get(URL, params=params, timeout=timeout)
            if r.status_code == 200:
                return r, None
            if r.status_code in RETRY_STATUS:
                last = f"HTTP {r.status_code}"
                time.sleep(4 * (2 ** attempt))
                continue
            snippet = r.text[:180].replace("\n", " ") if r.text else ""
            return None, f"HTTP {r.status_code}" + (f" — {snippet}" if snippet else "")
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(4 * (2 ** attempt))
    return None, f"{last} (after {retries} attempts)"


def try_html_json(session, params, debug_name=None):
    r, err = get(session, params)
    if r is None:
        return None, f"html: {err}"
    if debug_name:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        (DEBUG_DIR / f"{debug_name}.html").write_text(r.text, errors="replace")
    cands = extract_embedded_json(r.text)
    if not cands:
        return None, f"html: 200 OK ({len(r.content):,} bytes) but no embedded JSON found"
    name, rows = cands[0]
    return pd.DataFrame(rows), f"html: embedded JSON '{name}' ({len(rows)} rows)"


def try_csv(session, params):
    p = dict(params, csv="true")
    r, err = get(session, p)
    if r is None:
        return None, f"csv=true: {err}"
    text = r.content.decode("utf-8", "replace")
    if "<html" in text[:300].lower() or text.lstrip().startswith("<!"):
        return None, "csv=true: returned HTML, not CSV"
    try:
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception as e:
        return None, f"csv=true: unparseable ({type(e).__name__})"
    return (df, "csv=true: ok") if not df.empty else (None, "csv=true: empty")


def try_api(session, params):
    bases = [
        "https://baseballsavant.mlb.com/leaderboard/abs-challenges/api",
        "https://baseballsavant.mlb.com/api/leaderboard/abs-challenges",
    ]
    for base in bases:
        try:
            r = session.get(base, params=params, timeout=45)
            if r.status_code != 200:
                continue
            rows = _rows_from(r.json())
            if rows:
                return pd.DataFrame(rows), f"api: {base} ({len(rows)} rows)"
        except Exception:
            continue
    return None, "api: no candidate endpoint responded"


def fetch(session, year, ct, mode, level, game_type, debug=False):
    params = base_params(year, ct, mode, level, game_type)
    tag = f"{ct}_{mode}_{level}_{year}" if debug else None
    attempts = []
    for fn in (try_html_json, try_csv, try_api):
        df, msg = (fn(session, params, tag) if fn is try_html_json
                   else fn(session, params))
        attempts.append(msg)
        if df is not None and not df.empty:
            return df, attempts
    return None, attempts


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--level", default="mlb", choices=["mlb", "aaa"])
    ap.add_argument("--game-type", default="regular",
                    choices=["regular", "spring", "postseason"])
    ap.add_argument("--only", help="fetch a single challengeType (for testing)")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--debug", action="store_true",
                    help="save the raw HTML of each request to _debug/")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    types = [args.only] if args.only else CHALLENGE_TYPES

    lines, got, total = [], 0, 0
    print(f"Savant ABS leaderboards — {args.year} {args.level} {args.game_type}\n")

    for ct in types:
        for mode in DATA_MODES:
            total += 1
            print(f"  {ct:<15} {mode:<8}", end="  ", flush=True)
            df, attempts = fetch(session, args.year, ct, mode, args.level,
                                 args.game_type, args.debug)

            if df is None:
                print("FAILED")
                for a in attempts:
                    print(f"      {a}")
                lines.append(f"{ct}/{mode}: FAILED")
                lines += [f"    {a}" for a in attempts]
                time.sleep(args.sleep)
                continue

            name = f"abs_leaderboard_{ct}_{mode}_{args.level}_{args.year}.csv"
            df.to_csv(OUT_DIR / name, index=False)
            got += 1
            print(f"{len(df):>5} rows, {len(df.columns):>3} cols   [{attempts[-1]}]")
            lines.append(f"{ct}/{mode}: {len(df)} rows, {len(df.columns)} cols "
                         f"({attempts[-1]})")
            lines.append(f"    columns: {', '.join(df.columns.astype(str))}")
            time.sleep(args.sleep)

    (OUT_DIR / "abs_leaderboard_summary.txt").write_text("\n".join(lines))
    print(f"\n{got}/{total} leaderboards saved to {OUT_DIR}")

    if got == 0:
        print("""
Every strategy failed. Next step, and it takes about two minutes:

  1. Open this in Chrome:
       https://baseballsavant.mlb.com/leaderboard/abs-challenges?year=2026
  2. Open DevTools (Cmd+Opt+I) -> Network tab -> filter to Fetch/XHR
  3. Reload the page, then change a filter to force a refresh
  4. Find the request that returns the table rows. Right-click it ->
     Copy -> Copy as cURL, and paste that to me.

That gives us the real endpoint and we patch `try_api()` with it.

Or just skip this script. These are validation aggregates, not model inputs —
scripts 01, 02, and 08 do not depend on them, and the leaderboard has a working
"Download CSV" button you can click five times. Record the download date in your
manifest if you go that route; a hand-pulled file still needs provenance.""")
        return

    # Validation hook for Week 5.
    tm = OUT_DIR / f"abs_leaderboard_team-summary_for_{args.level}_{args.year}.csv"
    if tm.exists():
        df = pd.read_csv(tm)
        col = next((c for c in df.columns
                    if "chal" in c.lower() and df[c].dtype.kind in "if"
                    and not any(w in c.lower() for w in ("pct", "rate", "exp"))), None)
        if col:
            print(f"\nValidation target: Savant total '{col}' = {df[col].sum():,.0f}")
            print("Compare against len(abs_challenges_2026.parquet) from script 02.")
            print("A gap over ~2% means your extraction is missing challenge events.")


if __name__ == "__main__":
    main()
