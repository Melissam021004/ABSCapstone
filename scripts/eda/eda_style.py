"""
eda_style.py — shared chart styling for the EDA figures.

The one shared module in this project. Five chart scripts would otherwise carry
five copies of the same 80 lines of matplotlib setup, and changing a color would
mean editing all five.

Every figure is saved as a TRANSPARENT PNG at 200 DPI, 16:9, so it drops onto
whatever background your slide template uses.

The palette is validated for colorblind safety (adjacent and all-pairs CVD
separation, chroma, lightness band, contrast) against a light surface. If your
deck has a DARK background, these inks will not read — see DARK_MODE below.

Because a PNG has no tooltips, every figure also writes a companion .csv with the
exact plotted values. That is the "table view": it keeps the numbers reachable,
and it means you can quote a figure precisely when someone asks in Q&A.
"""

import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Text fonts, best first. Ones not installed on this machine are skipped, and
# DejaVu Sans (bundled with matplotlib) is the backstop for symbols.
_FONTS = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
# A listed font that isn't installed (Helvetica on Windows/Linux) makes
# matplotlib log "findfont: Font family ... not found" — expected, harmless,
# and the next font is used — so keep that out of the console.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ----------------------------------------------------------------- palette
# Validated categorical slots (first three are safe for scatter/all-pairs forms).
BLUE = "#2a78d6"     # slot 1
ORANGE = "#eb6834"   # slot 2
AQUA = "#1baf7a"     # slot 3
SERIES = [BLUE, ORANGE, AQUA]

# Diverging pair: warm/cool poles reading as opposite, neutral gray midpoint.
POS = BLUE
NEG = "#e34948"
MID = "#f0efec"

# Sequential blue ramp (light -> dark)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95"]

# Ink and chrome
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DEEMPH = "#c8c7c0"    # the "everything else" gray in emphasis charts

# If your slides are dark, flip this and re-validate before presenting.
DARK_MODE = False
if DARK_MODE:
    INK, INK_2 = "#ffffff", "#c3c2b7"
    GRID, AXIS, DEEMPH = "#2c2c2a", "#383835", "#4a4a47"
    BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
    SERIES = [BLUE, ORANGE, AQUA]
    POS, NEG, MID = BLUE, "#e66767", "#383835"

FULL = (12.0, 6.75)    # 16:9 -> 2400x1350 at 200 dpi
DPI = 200


# Background mode. The 03-06 figures default to transparent so they sit on any
# slide template; the Week 5 figures ask for solid white. Set once by
# apply_style() and read by save().
_BG = {"mode": "transparent", "color": "none"}
WHITE = "#ffffff"


def apply_style(bg="transparent"):
    """bg: 'transparent' or 'white' (any hex also works)."""
    if bg == "transparent":
        _BG.update(mode="transparent", color="none")
        face, transparent = "none", True
    else:
        color = WHITE if bg == "white" else bg
        _BG.update(mode="solid", color=color)
        face, transparent = color, False

    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.facecolor": face,
        "savefig.facecolor": face,
        "savefig.transparent": transparent,
        "axes.facecolor": face,
        # font.family as an explicit LIST turns on matplotlib's per-glyph
        # fallback (mpl >= 3.6): text is set in the first font that exists,
        # and any character that font lacks (★ ← → ≈ ≤ Δ ρ) is drawn from the
        # next one down. DejaVu Sans ships with matplotlib and has all of them,
        # so the "Glyph ... missing from font(s) Helvetica Neue" warnings and
        # the empty boxes they leave in the figures both go away. With the old
        # setting ("sans-serif") matplotlib picked ONE font and had no fallback.
        "font.family": list(_FONTS),
        "font.sans-serif": list(_FONTS) + ["sans-serif"],
        "font.size": 12,
        "text.color": INK,
        "axes.titlesize": 17,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlepad": 14,
        "axes.labelsize": 12,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",       # solid hairlines; dashed grids read as thresholds
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 11,
        "legend.labelcolor": INK_2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def new_fig(figsize=FULL):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def titles(ax, title, subtitle=None, xlabel=None, ylabel=None):
    """Title carries the finding; subtitle carries the provenance."""
    if subtitle:
        ax.set_title(f"{title}\n", loc="left")
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=11.5,
                color=INK_2, va="bottom", ha="left")
    else:
        ax.set_title(title, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)


def only_grid(ax, axis="y"):
    ax.grid(axis=axis, which="major")
    ax.grid(axis="x" if axis == "y" else "y", which="major", visible=False)


def footnote(fig, text, x=0.005):
    """
    Figure caption, wrapped so it never runs under the demo badge in the
    bottom-right corner. Long unwrapped captions were getting clipped.
    """
    import textwrap
    wrapped = "\n".join(textwrap.wrap(text, width=105))
    fig.text(x, -0.02, wrapped, fontsize=9.5, color=MUTED, ha="left", va="top")


def suptitles(fig, title, subtitle=None, x=0.005, top=1.13):
    """
    Title + subtitle for a MULTI-PANEL figure.

    Single-axes figures use titles(); those place the subtitle relative to the
    axes. On a multi-panel figure there is no single axes to anchor to, and
    stacking suptitle and a text call at hand-picked y values collides as soon
    as the figure height changes. This fixes the spacing in one place.
    """
    fig.text(x, top, title, fontsize=17, fontweight="semibold", color=INK,
             ha="left", va="bottom")
    if subtitle:
        fig.text(x, top - 0.075, subtitle, fontsize=11.5, color=INK_2,
                 ha="left", va="bottom")


def rounded_barh(ax, y, widths, height=0.62, color=BLUE, zorder=3, **kw):
    """
    Horizontal bars with rounded data-ends, anchored at the baseline.
    Falls back to plain bars if the rounding geometry misbehaves.
    """
    try:
        x0, x1 = ax.get_xlim()
        span = max(abs(x1 - x0), 1e-9)
        r = min(height * 0.30, span * 0.006)
        patches = []
        for yy, w in zip(y, widths):
            if pd.isna(w):
                continue
            left, width = (0, w) if w >= 0 else (w, -w)
            if width <= r * 2:          # too short to round without distortion
                p = mpatches.Rectangle((left, yy - height / 2), width, height,
                                       color=color, zorder=zorder, lw=0, **kw)
            else:
                p = mpatches.FancyBboxPatch(
                    (left + r, yy - height / 2 + r * 0.0), width - 2 * r, height,
                    boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=r),
                    mutation_aspect=1.0, color=color, zorder=zorder, lw=0, **kw)
            ax.add_patch(p)
            patches.append(p)
        return patches
    except Exception:
        return ax.barh(y, widths, height=height, color=color, zorder=zorder, lw=0)


def stat_tile(ax, value, label, sub=None, color=None):
    """A single headline number. The number IS the chart."""
    ax.axis("off")
    ax.text(0, 0.62, value, fontsize=44, fontweight="bold",
            color=color or INK, ha="left", va="center")
    ax.text(0, 0.24, label, fontsize=13, color=INK, ha="left", va="center",
            fontweight="semibold")
    if sub:
        ax.text(0, 0.06, sub, fontsize=10.5, color=MUTED, ha="left", va="center")


def save(fig, out_dir, name, table=None):
    """Write the PNG plus its table-view CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, transparent=(_BG['mode'] == 'transparent'),
                facecolor=_BG['color'])
    plt.close(fig)
    msg = f"  {png}"
    if table is not None and len(table):
        csv = out_dir / f"{name}_data.csv"
        table.to_csv(csv, index=False)
        msg += f"\n    table view -> {csv.name}"
    print(msg)
    return png


# ----------------------------------------------------------- column finding
def find_col(df, *patterns, numeric=None, exclude=()):
    """
    Locate a column by regex against lowercased names.

    Savant's column names are not documented and change between leaderboards,
    so every EDA script detects what it needs rather than hard-coding names.
    Returns None when nothing matches, and the caller skips that panel.
    """
    for pat in patterns:
        for c in df.columns:
            lc = str(c).lower()
            if not re.search(pat, lc):
                continue
            if any(e in lc for e in exclude):
                continue
            if numeric is True and df[c].dtype.kind not in "if":
                continue
            if numeric is False and df[c].dtype.kind in "if":
                continue
            return c
    return None


def as_rate(s):
    """Coerce a column that might be 0-1 or 0-100 into 0-1."""
    s = pd.to_numeric(s, errors="coerce")
    return s / 100.0 if s.dropna().max() and s.dropna().max() > 1.5 else s


def missing_note(path):
    return (f"  [skip] {path} not found — run the matching extraction script first.")


def read_table(path):
    """
    Read a .parquet, falling back to a .csv sibling.

    The extraction scripts write parquet, but pyarrow fails to install on some
    setups. Rather than have the whole EDA layer die on an ImportError, look for
    the CSV next to it. Returns None if neither is readable.
    """
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        try:
            return pd.read_parquet(p)
        except ImportError:
            print(f"  [warn] pyarrow unavailable — looking for {p.stem}.csv")
        except Exception as e:
            print(f"  [warn] could not read {p.name}: {type(e).__name__}")
    csv = p.with_suffix(".csv")
    if csv.exists():
        try:
            return pd.read_csv(csv)
        except Exception:
            return None
    return None


# ------------------------------------------------------------- demo marking
def is_demo(data_dir):
    """True when we're reading the synthetic fixtures, not a real pull."""
    return "demo" in str(data_dir).lower()


def demo_stamp(ax_or_fig, on=True):
    """
    Mark a demo figure unmistakably — as a corner badge, not a watermark.

    A big diagonal watermark sits on top of the marks and makes the chart hard
    to evaluate, which defeats the point of previewing it. A badge is just as
    unmissable and leaves the data readable.
    """
    if not on:
        return
    # Always figure-level, bottom-right, beside the footnote — the one place on
    # a chart guaranteed to be free of title, legend and marks.
    fig = getattr(ax_or_fig, "figure", ax_or_fig)
    fig.text(1.0, -0.02, "  DEMO DATA — NOT REAL  ", fontsize=10.5,
             fontweight="bold", color="white", ha="right", va="top", zorder=200,
             bbox=dict(boxstyle="round,pad=0.35", fc="#d03b3b", ec="none"))


def halo(artist, lw=2.6):
    """White outline so a direct label stays legible over a mark or gridline."""
    import matplotlib.patheffects as pe
    surface = WHITE if _BG["mode"] == "transparent" else _BG["color"]
    artist.set_path_effects([pe.withStroke(linewidth=lw, foreground=surface)])
    return artist


def pad_cat_axis(ax, n, axis="y", pad=0.85):
    """Breathing room so the first/last category isn't jammed against the frame."""
    if axis == "y":
        ax.set_ylim(-pad, n - 1 + pad)
    else:
        ax.set_xlim(-pad, n - 1 + pad)
