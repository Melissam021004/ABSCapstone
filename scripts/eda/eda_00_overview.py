#!/usr/bin/env python3
"""
eda_00_overview.py — the opening slide: what came out of scripts 03-06.

Produces:
  fig00_extraction_overview.png

A handful of headline numbers is a KPI row of stat tiles, not a bar chart —
putting four quantities on different scales (rows, catchers, umpires, plate
appearances) into one bar chart would invent a comparison between them.

Tiles report what was extracted and go gray when a source is missing, so the
figure doubles as a status board while your pulls are still running.

    python scripts/eda/eda_00_overview.py
    ABS_DATA_DIR=data_demo python scripts/eda/eda_00_overview.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_style as S

DATA_DIR = Path(os.environ.get("ABS_DATA_DIR", "data"))
RAW = DATA_DIR / "raw"
OUT = Path(os.environ.get("ABS_FIG_DIR", "figures"))
DEMO = S.is_demo(DATA_DIR)


def fmt(n):
    if n is None:
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n/1_000:.0f}k"
    return f"{n:,}"


def gather():
    tiles = []

    # 03 — leaderboards
    files = sorted((RAW / "savant").glob("abs_leaderboard_*.csv"))
    rows = 0
    for f in files:
        try:
            rows += len(pd.read_csv(f))
        except Exception:
            pass
    tiles.append({
        "value": fmt(rows) if files else None,
        "label": "Leaderboard rows",
        "sub": f"{len(files)} Savant ABS views" if files else "script 03 not run",
        "ok": bool(files),
    })

    # 04 — catcher framing
    fr = S.read_table(RAW / "savant" / "catcher_framing_all_years.parquet")
    if fr is None:
        for f in sorted((RAW / "savant").glob("catcher_framing_*.csv")):
            fr = pd.read_csv(f)
            break
    if fr is not None and len(fr):
        season = S.find_col(fr, r"season", r"^year$", numeric=True)
        n_s = fr[season].nunique() if season else 1
        sub = f"{n_s} season{'s' if n_s != 1 else ''} pulled"
    else:
        sub = "script 04 not run"
    tiles.append({
        "value": fmt(len(fr)) if fr is not None and len(fr) else None,
        "label": "Catcher-seasons", "sub": sub,
        "ok": fr is not None and len(fr) > 0,
    })

    # 05 — umpires
    up = S.read_table(RAW / "umpires" / "umpire_accuracy_derived_2026.parquet")
    ug = S.read_table(RAW / "umpires" / "umpire_game_accuracy_2026.parquet")
    sub = f"{fmt(len(ug))} umpire-games" if ug is not None and len(ug) \
        else ("accuracy derived" if up is not None else "script 05 not run")
    tiles.append({
        "value": fmt(len(up)) if up is not None and len(up) else None,
        "label": "Home-plate umpires", "sub": sub,
        "ok": up is not None and len(up) > 0,
    })

    # 06 — win probability
    wp = S.read_table(RAW / "winprob" / "winprob_2026.parquet")
    if wp is not None and len(wp):
        games = wp["game_pk"].nunique() if "game_pk" in wp.columns else None
        sub = f"across {fmt(games)} games" if games else "win prob + leverage"
    else:
        sub = "script 06 not run"
    tiles.append({
        "value": fmt(len(wp)) if wp is not None and len(wp) else None,
        "label": "Plate appearances", "sub": sub,
        "ok": wp is not None and len(wp) > 0,
    })
    return tiles


def main():
    print(f"Reading {RAW}" + ("   [DEMO DATA]" if DEMO else ""))
    tiles = gather()

    S.apply_style()
    # One axes with manual tile placement. Four subplots leave matplotlib's
    # margins between them and the row drifts out of alignment with the title.
    fig, ax = plt.subplots(figsize=(S.FULL[0], 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n = len(tiles)
    step = 1.0 / n
    for i, t in enumerate(tiles):
        x = i * step
        color = S.BLUE if t["ok"] else S.DEEMPH
        ax.text(x, 0.66, t["value"] or "—", fontsize=52, fontweight="bold",
                color=color, ha="left", va="center")
        ax.text(x, 0.30, t["label"], fontsize=13.5, color=S.INK,
                ha="left", va="center", fontweight="semibold")
        ax.text(x, 0.12, t["sub"], fontsize=11, color=S.MUTED,
                ha="left", va="center")
        if i:   # hairline separator between tiles
            ax.plot([x - step * 0.06] * 2, [0.06, 0.86], color=S.GRID,
                    lw=1.0, zorder=1)

    n_ok = sum(1 for t in tiles if t["ok"])
    ax.text(0, 1.20, "ABS capstone — supporting data extracted", fontsize=19,
            fontweight="semibold", color=S.INK, ha="left", va="bottom")
    ax.text(0, 1.04,
            f"{n_ok} of {n} supporting sources pulled · "
            f"Baseball Savant and the MLB Stats API · 2026 season",
            fontsize=12, color=S.INK_2, ha="left", va="bottom")
    ax.text(0, -0.30, "Pitch-level Statcast and ABS challenge events (scripts 01-02) "
            "are reported separately.\nGray tiles are sources not yet pulled.",
            fontsize=9.5, color=S.MUTED, ha="left", va="top")
    S.demo_stamp(fig, DEMO)

    table = pd.DataFrame([{"source": t["label"], "value": t["value"],
                           "detail": t["sub"], "extracted": t["ok"]} for t in tiles])
    S.save(fig, OUT, "fig00_extraction_overview", table)
    print(f"\nFigures -> {OUT}/")


if __name__ == "__main__":
    main()
