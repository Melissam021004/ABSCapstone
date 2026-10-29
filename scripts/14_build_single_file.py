#!/usr/bin/env python3
"""
14_build_single_file.py — fold the dashboard into one self-contained .html file.

The four-file version in dashboard/ is the one you edit and the one GitHub Pages
serves. This produces a single file with the CSS, JavaScript and data inlined,
which is useful when you need to hand the whole dashboard to someone as one
attachment — emailing it to your supervisor, dropping it on a USB stick for a
presentation machine, or archiving a snapshot alongside a report.

Open the output by double-clicking it. Nothing else is required.

    python scripts/14_build_single_file.py
    python scripts/14_build_single_file.py --out dist/abs-dashboard.html
    python scripts/14_build_single_file.py --fragment    # no <html>/<head> wrapper

Output: dist/abs-dashboard.html
"""

import argparse
import re
from pathlib import Path

SRC = Path("dashboard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default="dist/abs-dashboard.html")
    ap.add_argument("--fragment", action="store_true",
                    help="emit only title/style/body content, no document wrapper")
    args = ap.parse_args()

    src = Path(args.src)
    for f in ("index.html", "styles.css", "app.js", "data.js"):
        if not (src / f).exists():
            raise SystemExit(f"Missing {src/f}. Run 12_export_dashboard_data.py first.")
    if not (src / "model_data.js").exists():
        print(f"  NOTE: no {src / 'model_data.js'} — the Overturn Model tab will be empty. "
              "Run scripts/18_export_model_dashboard.py first.")

    html = (src / "index.html").read_text()
    css = (src / "styles.css").read_text()
    data = (src / "data.js").read_text()
    app = (src / "app.js").read_text()

    # swap the <link> to the stylesheet for the stylesheet itself
    html = html.replace('<link rel="stylesheet" href="styles.css">',
                        "<style>\n" + css + "\n</style>")
    # swap the two <script src> tags for their contents
    html = html.replace('<script src="data.js"></script>',
                        "<script>\n" + data + "\n</script>")
    html = html.replace('<script src="app.js"></script>',
                        "<script>\n" + app + "\n</script>")
    for extra in ("model_data.js", "model.js"):          # Weeks 7–9 tabs
        tag = f'<script src="{extra}"></script>'
        body = (src / extra).read_text() if (src / extra).exists() else ""
        html = html.replace(tag, "<script>\n" + body + "\n</script>" if body else "")

    if args.fragment:
        # Keep <title>, the font link, the style block and everything in <body>.
        title = re.search(r"<title>.*?</title>", html, re.S)
        fonts = re.findall(r'<link rel="stylesheet" href="https://fonts\.googleapis[^>]*>', html)
        style = re.search(r"<style>.*?</style>", html, re.S)
        body = re.search(r"<body>(.*)</body>", html, re.S)
        parts = []
        if title:
            parts.append(title.group(0))
        parts += fonts
        if style:
            parts.append(style.group(0))
        if body:
            parts.append(body.group(1).strip())
        html = "\n".join(parts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    kb = out.stat().st_size / 1024
    print(f"  {out}  ({kb:,.0f} KB)")
    print("  Self-contained — double-click to open, or attach it to an email.")


if __name__ == "__main__":
    main()
