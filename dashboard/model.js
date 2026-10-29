/* ===========================================================================
   ABS Challenge Analytics — the Overturn Model and Progress tabs
   ---------------------------------------------------------------------------
   Reads window.ABS_MODEL, written by scripts/18_export_model_dashboard.py from
   the Week 8 and Week 9 outputs, and draws with the same small SVG helpers as
   app.js (shared through window.ABS_UI). Nothing here fits or re-scores
   anything: every number on these tabs is one the pipeline already wrote.

   TO ADD A CHART: put a <div id="c-whatever"> in index.html, write a draw
   function below, and call it from renderModel().
   =========================================================================== */

(function () {
"use strict";

const U = window.ABS_UI;
const MD = window.ABS_MODEL;
if (!U) return;
const { $, mk, frame, txt, tok, showTip, hideTip, tipRow, legend, pct, dec, nf } = U;

const FAM = { logistic: "--series-1", gam: "--series-2", catboost: "--series-3",
              lightgbm: "--series-3", hgb: "--series-3" };
const famColor = (f) => tok(FAM[f] || "--ink-3");
const BOOSTED = ["catboost", "lightgbm", "hgb"];
const METHOD = { none: "uncalibrated", platt: "Platt scaling", isotonic: "isotonic" };
const SETNAME = { location: "Location only", full: "Everything" };
const view = { call: "strike", h: 1, win: "aug" };

const sgn = (v, d) => (v === null || v === undefined || !isFinite(v)) ? "—"
  : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d === undefined ? 4 : d);
const f4 = (v) => dec(v, 4);
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function rule(g, x1, y1, x2, y2, strong) {
  const a = { x1: x1, y1: y1, x2: x2, y2: y2 };
  if (strong) { a.stroke = tok("--ink-2"); a["stroke-width"] = 1.2; } else { a.class = "tick-line"; }
  return mk("line", a, g);
}
function note(host, msg) {
  host.innerHTML = "";
  host.appendChild(el("p", "note muted-note", msg));
}

/* ------------------------------------------------------------ colour ramp */
function hexRgb(h) {
  h = (h || "#000").replace("#", "");
  if (h.length === 3) h = h.split("").map(c => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rampFn() {
  const stops = ["--card", "--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5", "--seq-6",
                 "--seq-7"].map(t => hexRgb(tok(t)));
  return (p) => {
    const t = Math.max(0, Math.min(1, p)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(t)), u = t - i;
    const a = stops[i], b = stops[i + 1];
    return "rgb(" + a.map((v, k) => Math.round(v + (b[k] - v) * u)).join(",") + ")";
  };
}

/* ------------------------------------------------- the ABS zone (feet, in.) */
const HW = 8.5 / 12, RB = (2.9 / 2) / 12;
function zoneDistIn(x, z, top, bot) {
  const dx = Math.abs(x) - HW, dz = Math.max(bot - z, z - top);
  let d;
  if (dx <= 0 && dz <= 0) d = Math.max(dx, dz);
  else if (dx > 0 && dz > 0) d = Math.hypot(dx, dz);
  else d = Math.max(dx, dz);
  return (d - RB) * 12;
}

/* Marching squares: line segments where the grid crosses `level`. */
function contour(grid, xs, zs, level) {
  const segs = [];
  const P = (j, i) => [xs[j], zs[i]];
  const lerp = (p, q, va, vb) => {
    const t = (level - va) / ((vb - va) || 1e-12);
    return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
  };
  const CASES = { 1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 2], [0, 1]],
                  6: [[0, 2]], 7: [[3, 2]], 8: [[2, 3]], 9: [[0, 2]], 10: [[0, 3], [1, 2]],
                  11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[0, 3]] };
  for (let i = 0; i < zs.length - 1; i++) {
    for (let j = 0; j < xs.length - 1; j++) {
      const a = grid[i][j], b = grid[i][j + 1], c = grid[i + 1][j + 1], d = grid[i + 1][j];
      const idx = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) |
                  (d > level ? 8 : 0);
      if (!CASES[idx]) continue;
      const edge = [lerp(P(j, i), P(j + 1, i), a, b), lerp(P(j + 1, i), P(j + 1, i + 1), b, c),
                    lerp(P(j, i + 1), P(j + 1, i + 1), d, c), lerp(P(j, i), P(j, i + 1), a, d)];
      CASES[idx].forEach(([e1, e2]) => segs.push([edge[e1], edge[e2]]));
    }
  }
  return segs;
}

/* =========================================================================
   OVERTURN MODEL TAB
   ========================================================================= */
function modelStrip() {
  const host = $("#m-strip");
  host.innerHTML = "";
  const w9 = MD.week9, w8 = MD.week8;
  const items = [];
  if (w9) {
    const a = w9.metrics.august_crossfit, s = w9.metrics.september;
    const parts = w9.model_label.split(" · ");
    items.push({ k: "Locked p(s)", v: parts[0],
                 sub: (parts[1] || "") + " · " + METHOD[w9.calibration.method] });
    items.push({ k: "Brier · August", v: f4(a.brier), sub: "calibration month, cross-fitted" });
    items.push({ k: "Brier · September", v: s ? f4(s.brier) : "Sealed",
                 sub: s ? "test month, opened once" : "opened once, by --final-test" });
    items.push({ k: "Calibration slope", v: dec((s || a).cal_slope, 2),
                 sub: (s ? "September" : "August") + " · 1 = ideal" });
    const W = w9.windows;
    items.push({ k: "Challenges", v: nf(W.train.challenges + W.calibration.challenges +
                                         W.test.challenges),
                 sub: nf(W.train.challenges) + " train · " + nf(W.calibration.challenges) +
                      " cal · " + nf(W.test.challenges) + " test" });
  } else if (w8) {
    items.push({ k: "Baseline to beat", v: w8.baseline_label.split(" · ")[0],
                 sub: w8.baseline_label });
    items.push({ k: "Boosted models", v: w8.n_beat + " of " + w8.n_boosted,
                 sub: "reliably beat the baseline" });
  }
  items.forEach(it => {
    const d = el("div", "stat");
    d.append(el("span", "k", it.k), el("span", "v", it.v), el("span", "sub", it.sub));
    host.appendChild(d);
  });
}

function drawSurface() {
  const host = $("#c-surface");
  const S = MD.week9.surface, H = S.heights[view.h], grid = H[view.call];
  const xs = S.xs, zs = H.zs, nx = xs.length, nz = zs.length;
  const spanX = xs[nx - 1] - xs[0], spanZ = zs[nz - 1] - zs[0];
  const w = 540, m = { t: 10, r: 14, b: 38, l: 46 };
  const iw = w - m.l - m.r, ih = iw * spanZ / spanX;
  const f = frame(host, w, ih + m.t + m.b, m);
  const X = (x) => ((x - xs[0]) / spanX) * f.iw;
  const Y = (z) => f.ih - ((z - zs[0]) / spanZ) * f.ih;
  const cw = f.iw / (nx - 1), chh = f.ih / (nz - 1);
  const ramp = rampFn();
  for (let i = 0; i < nz; i++) {
    for (let j = 0; j < nx; j++) {
      mk("rect", { x: X(xs[j]) - cw / 2, y: Y(zs[i]) - chh / 2, width: cw + 0.6,
                   height: chh + 0.6, fill: ramp(grid[i][j]) }, f.g);
    }
  }
  // the 50% coin-flip line
  const segs = contour(grid, xs, zs, 0.5);
  if (segs.length) {
    mk("path", { d: segs.map(s => "M" + X(s[0][0]) + " " + Y(s[0][1]) + "L" + X(s[1][0]) + " " +
                                   Y(s[1][1])).join(""),
                 fill: "none", stroke: tok("--ink"), "stroke-width": 1.6,
                 "stroke-linecap": "round" }, f.g);
  }
  // the zone, and the zone plus the ball's radius
  mk("rect", { x: X(-HW - RB), y: Y(H.top + RB), width: X(HW + RB) - X(-HW - RB),
               height: Y(H.bot - RB) - Y(H.top + RB), fill: "none", stroke: tok("--series-2"),
               "stroke-width": 1, "stroke-dasharray": "4 3" }, f.g);
  mk("rect", { x: X(-HW), y: Y(H.top), width: X(HW) - X(-HW), height: Y(H.bot) - Y(H.top),
               fill: "none", stroke: tok("--series-2"), "stroke-width": 2.2 }, f.g);
  for (let gx = -1; gx <= 1.001; gx += 0.5) {
    txt(f.g, X(gx), f.ih + 15, dec(gx, 1), "axis-label", "middle");
  }
  for (let gz = Math.ceil(zs[0] * 2) / 2; gz <= zs[nz - 1]; gz += 0.5) {
    txt(f.g, -8, Y(gz), dec(gz, 1), "axis-label", "end");
  }
  txt(f.g, f.iw / 2, f.ih + 31, "feet from the middle of the plate (catcher's view)",
      "axis-label", "middle");
  const hit = mk("rect", { x: 0, y: 0, width: f.iw, height: f.ih, fill: "transparent" }, f.g);
  hit.addEventListener("pointermove", (e) => {
    const r = hit.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fz = 1 - (e.clientY - r.top) / r.height;
    const j = Math.max(0, Math.min(nx - 1, Math.round(fx * (nx - 1))));
    const i = Math.max(0, Math.min(nz - 1, Math.round(fz * (nz - 1))));
    const x = xs[j], z = zs[i], p = grid[i][j];
    const d = zoneDistIn(x, z, H.top, H.bot);
    const where = d < 0 ? Math.abs(d).toFixed(1) + " in. inside the zone"
                        : d.toFixed(1) + " in. outside the zone";
    showTip(e, tipRow(pct(p, 0) + " chance of overturn",
      (view.call === "strike" ? "called strike" : "called ball") + " · " + where +
      "<br>x " + sgn(x, 2).replace("−", "-") + " ft · height " + dec(z, 2) + " ft"));
  });
  hit.addEventListener("pointerleave", hideTip);
  const lg = $("#l-surface");
  lg.innerHTML = "";
  const bar = el("span", "rampbar");
  bar.style.background = "linear-gradient(90deg," +
    [0, .14, .28, .42, .57, .71, .85, 1].map(p => ramp(p)).join(",") + ")";
  const rs = el("span");
  rs.append(el("span", "rampend", "0%"), bar, el("span", "rampend", "100% overturned"));
  lg.appendChild(rs);
  const z1 = el("span"), i1 = el("i");
  i1.style.background = tok("--series-2");
  z1.append(i1, document.createTextNode("ABS zone (dashed: + ball radius)"));
  const c1 = el("span"), i2 = el("i");
  i2.style.background = tok("--ink");
  c1.append(i2, document.createTextNode("50% — the coin-flip line"));
  lg.append(z1, c1);
  $("#m-surface-src").textContent = "p(s) " + MD.week9.version + " · " + H.label +
    " batter · " + (view.call === "strike" ? "the batter challenges" :
                    "the catcher or pitcher challenges");
}

function compareRows() {
  const w8 = MD.week8;
  const sep = view.win === "sep" && MD.week9 && MD.week9.test.opened;
  const src = sep ? MD.week9.test.comparison : w8.models;
  const by = {};
  src.forEach(r => { by[r.model] = r; });
  const info = {};
  w8.models.forEach(r => { info[r.model] = r; });
  const peers = { location: ["M2", "GAM-loc"], full: ["M5", "GAM-full"] };
  const groups = [];
  if (sep && by.PROD) {
    groups.push({ name: "Locked p(s)", keys: ["PROD", "PROD-raw"] });
  }
  ["location", "full"].forEach(fs => {
    let keys = peers[fs].slice();
    const b = info[w8.baseline];
    if (b && b.feature_set === fs && keys.indexOf(w8.baseline) < 0) keys.unshift(w8.baseline);
    w8.models.filter(m => BOOSTED.indexOf(m.family) >= 0 && m.feature_set === fs)
      .forEach(m => keys.push(m.model));
    keys = keys.filter(k => by[k]);
    if (keys.length) groups.push({ name: SETNAME[fs], keys: keys });
  });
  return { by: by, info: info, groups: groups, sep: sep };
}

function drawCompare() {
  const host = $("#c-compare");
  const w8 = MD.week8;
  const C = compareRows();
  const rowH = 25, headH = 24, W = 620;
  let n = 0;
  C.groups.forEach(g => { n += g.keys.length; });
  const m = { t: 30, r: 92, b: 34, l: 158 };
  const H = n * rowH + C.groups.length * headH + m.t + m.b;
  const f = frame(host, W, H, m);
  const aw = 210, gap = 26, bw = f.iw - aw - gap;
  const rows = [];
  C.groups.forEach(g => g.keys.forEach(k => rows.push(C.by[k])));
  const lo = Math.min.apply(null, rows.map(r => r.brier_lo)),
        hi = Math.max.apply(null, rows.map(r => r.brier_hi));
  const pad = (hi - lo) * 0.06 || 0.001;
  const XA = (v) => ((v - (lo - pad)) / ((hi + pad) - (lo - pad))) * aw;
  const dvals = rows.filter(r => r.pd_lo !== undefined && r.pd_lo !== null)
                    .map(r => Math.max(Math.abs(r.pd_lo), Math.abs(r.pd_hi)));
  const span = (Math.max.apply(null, dvals.concat([0.0005])) || 0.001) * 1.12;
  const XB = (v) => aw + gap + ((v + span) / (2 * span)) * bw;
  // headings and axes
  txt(f.g, 0, -18, (C.sep ? "September" : "August") + " Brier [95%]", "val-label", "start")
    .setAttribute("fill", tok("--ink"));
  txt(f.g, aw + gap, -18, "minus the baseline ★, same games", "val-label", "start")
    .setAttribute("fill", tok("--ink"));
  const ticksA = [lo, (lo + hi) / 2, hi];
  ticksA.forEach(v => {
    mk("line", { x1: XA(v), y1: -4, x2: XA(v), y2: f.ih, class: "tick-line" }, f.g);
    txt(f.g, XA(v), f.ih + 14, dec(v, 3), "axis-label", "middle");
  });
  const dd = span < 0.002 ? 4 : 3;
  [-span * 0.8, 0, span * 0.8].forEach(v => {
    rule(f.g, XB(v), -4, XB(v), f.ih, v === 0);
    txt(f.g, XB(v), f.ih + 14, v === 0 ? "0" : sgn(v, dd), "axis-label", "middle");
  });
  txt(f.g, aw / 2, f.ih + 28, "lower = better", "axis-label", "middle");
  txt(f.g, aw + gap + bw / 2, f.ih + 28, "← better than ★   worse →", "axis-label", "middle");
  let y = 0;
  C.groups.forEach(g => {
    txt(f.g, -m.l + 4, y + headH / 2 + 2, g.name.toUpperCase(), "axis-label", "start")
      .setAttribute("fill", tok("--ink"));
    y += headH;
    g.keys.forEach(k => {
      const r = C.by[k], inf = C.info[k] || {};
      const fam = k === "PROD" ? "prod" : k === "PROD-raw" ? "raw" : inf.family;
      const col = k === "PROD" ? tok("--ink") : k === "PROD-raw" ? tok("--ink-3") : famColor(fam);
      const cy = y + rowH / 2;
      const lab = (r.label || k).replace(" (locked)", "") + (k === w8.baseline ? " ★" : "");
      txt(f.g, -10, cy, lab.length > 26 ? lab.slice(0, 25) + "…" : lab, "val-label", "end")
        .setAttribute("fill", k === "PROD" ? tok("--ink") : tok("--ink-2"));
      mk("line", { x1: XA(r.brier_lo), y1: cy, x2: XA(r.brier_hi), y2: cy, stroke: col,
                   "stroke-width": 3, "stroke-linecap": "round" }, f.g);
      mk(fam === "lightgbm" ? "rect" : "circle",
         fam === "lightgbm" ? { x: XA(r.brier) - 5, y: cy - 5, width: 10, height: 10, fill: col,
                                stroke: tok("--paper"), "stroke-width": 1.5,
                                transform: "rotate(45 " + XA(r.brier) + " " + cy + ")" }
                            : { cx: XA(r.brier), cy: cy, r: 5.5, fill: col,
                                stroke: tok("--paper"), "stroke-width": 1.5 }, f.g);
      let verdict = "";
      if (k === w8.baseline) {
        txt(f.g, XB(0), cy, "★ baseline", "axis-label", "middle")
          .setAttribute("fill", tok("--ink"));
      } else if (r.pd_brier !== undefined && r.pd_brier !== null) {
        mk("line", { x1: XB(r.pd_lo), y1: cy, x2: XB(r.pd_hi), y2: cy, stroke: col,
                     "stroke-width": 3, "stroke-linecap": "round" }, f.g);
        mk("circle", { cx: XB(r.pd_brier), cy: cy, r: 5, fill: col, stroke: tok("--paper"),
                       "stroke-width": 1.5 }, f.g);
        verdict = r.pd_hi < 0 ? "better" : r.pd_lo > 0 ? "worse" : "can't tell";
        if (!C.sep && inf.beats) verdict = "BEATS ★";
        if (!C.sep && BOOSTED.indexOf(inf.family) >= 0 && r.pd_hi < 0 && !inf.calibration_ok) {
          verdict = "better, worse cal.";
        }
        const t = txt(f.g, f.iw + 8, cy, verdict, "axis-label", "start");
        t.setAttribute("fill", verdict === "can't tell" ? tok("--ink-3") : tok("--ink"));
      }
      const hit = mk("rect", { x: -m.l, y: y, width: W, height: rowH, fill: "transparent" }, f.g);
      hit.addEventListener("pointermove", (e) => showTip(e,
        tipRow("Brier " + f4(r.brier), lab) +
        '<div class="tk">95% ' + f4(r.brier_lo) + "–" + f4(r.brier_hi) +
        (r.pd_brier !== undefined && r.pd_brier !== null && k !== w8.baseline
          ? " · vs ★ " + sgn(r.pd_brier) + " [" + sgn(r.pd_lo) + ", " + sgn(r.pd_hi) + "]" : "") +
        "<br>AUC " + dec(r.auc, 3) + " · slope " + dec(r.cal_slope, 2) +
        " · log loss " + dec(r.log_loss, 3) + "</div>"));
      hit.addEventListener("pointerleave", hideTip);
      y += rowH;
    });
  });
  legend($("#l-compare"), [
    { label: "logistic", color: tok("--series-1"), sq: true },
    { label: "GAM", color: tok("--series-2"), sq: true },
    { label: "gradient boosting", color: tok("--series-3"), sq: true }
  ].concat(C.sep ? [{ label: "locked p(s)", color: tok("--ink"), sq: true }] : []));
  const nb = w8.n_beat, nt = w8.n_boosted;
  $("#m-cmp-title").textContent = C.sep
    ? "September: every model, scored once on the test month"
    : (nb ? nb + " of " + nt + " boosted models reliably beat the baseline"
          : "No boosted model reliably beats the baseline");
  $("#m-cmp-src").textContent = (C.sep ? "Test month" : "Validation month") +
    " · 95% intervals from resampling whole games · Δ paired on the same resamples";
}

function relSeries() {
  const w9 = MD.week9, R = w9.reliability, out = [];
  const m = w9.calibration.method;
  if (m === "none") {
    out.push({ name: "August, cross-fitted", rows: R.uncalibrated, color: tok("--series-1"),
               filled: true });
  } else {
    out.push({ name: "August, before calibration", rows: R.uncalibrated, color: tok("--ink-3"),
               filled: false });
    out.push({ name: "August, " + METHOD[m] + " (cross-fitted)", rows: R.calibrated,
               color: tok("--series-1"), filled: true });
  }
  if (w9.test.opened && w9.test.reliability) {
    out.push({ name: "September (test)", rows: w9.test.reliability, color: tok("--series-2"),
               filled: true });
  }
  return out;
}

function drawReliability() {
  const host = $("#c-rel");
  const w = 440, h = 400;
  const f = frame(host, w, h, { t: 10, r: 14, b: 40, l: 46 });
  const X = (v) => v * f.iw, Y = (v) => f.ih - v * f.ih;
  [0, .25, .5, .75, 1].forEach(p => {
    mk("line", { x1: 0, y1: Y(p), x2: f.iw, y2: Y(p), class: "tick-line" }, f.g);
    mk("line", { x1: X(p), y1: 0, x2: X(p), y2: f.ih, class: "tick-line" }, f.g);
    txt(f.g, -8, Y(p), pct(p, 0), "axis-label", "end");
    txt(f.g, X(p), f.ih + 15, pct(p, 0), "axis-label", "middle");
  });
  mk("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: tok("--ink-3"),
               "stroke-width": 1, "stroke-dasharray": "4 3" }, f.g);
  txt(f.g, f.iw / 2, f.ih + 32, "predicted chance of overturn", "axis-label", "middle");
  const series = relSeries();
  const nmax = Math.max.apply(null, series.map(s => Math.max.apply(null,
    s.rows.map(r => r.n).concat([1]))));
  series.forEach((s, si) => {
    const off = (si - (series.length - 1) / 2) * 5;
    const pts = s.rows.filter(r => r.mean_pred !== null && r.obs !== null);
    if (pts.length > 1) {
      mk("path", { d: pts.map((r, i) => (i ? "L" : "M") + (X(r.mean_pred) + off) + " " +
                                       Y(r.obs)).join(""),
                   fill: "none", stroke: s.color, "stroke-width": 1.2, opacity: .55 }, f.g);
    }
    pts.forEach(r => {
      mk("line", { x1: X(r.mean_pred) + off, y1: Y(r.lo), x2: X(r.mean_pred) + off, y2: Y(r.hi),
                   stroke: s.color, "stroke-width": 1.5, opacity: .8 }, f.g);
      const c = mk("circle", { cx: X(r.mean_pred) + off, cy: Y(r.obs),
                               r: 3 + 7 * Math.sqrt(r.n / nmax),
                               fill: s.filled ? s.color : tok("--paper"), stroke: s.color,
                               "stroke-width": 1.6 }, f.g);
      c.addEventListener("pointermove", (e) => showTip(e,
        tipRow(pct(r.obs, 0) + " overturned", s.name + " · predicted " + pct(r.mean_pred, 0) +
               " · " + nf(r.n) + " challenges")));
      c.addEventListener("pointerleave", hideTip);
    });
  });
  legend($("#l-rel"), series.map(s => ({ label: s.name, color: s.color, sq: true })));
  const w9 = MD.week9, a = w9.metrics.august_crossfit, sp = w9.metrics.september;
  $("#m-rel-title").textContent = sp
    ? "On September, p(s) said " + pct(sp.midrange.mean_pred, 0) + " where it was unsure — " +
      pct(sp.midrange.obs_rate, 0) + " came true"
    : "Where p(s) is unsure (20–80%), it said " + pct(a.midrange.mean_pred, 0) + " and " +
      pct(a.midrange.obs_rate, 0) + " came true";
}

function drawCalTable() {
  const t = $("#t-cal");
  const w9 = MD.week9;
  t.innerHTML = "";
  const head = el("thead"), hr = el("tr");
  [["Method", ""], ["Brier", "num"], ["95%", "num"], ["Log loss", "num"], ["ECE", "num"],
   ["Slope", "num"], ["Δ Brier vs uncal.", "num"]]
    .forEach(([h, c]) => hr.appendChild(el("th", c, h)));
  head.appendChild(hr);
  t.appendChild(head);
  const tb = el("tbody");
  const order = [w9.feature_set].concat(["location", "full"].filter(x => x !== w9.feature_set));
  order.forEach(fs => {
    const rows = w9.calibration_table.filter(r => r.feature_set === fs);
    if (!rows.length) return;
    const gr = el("tr", "grouprow"), gd = el("td", "name",
      rows[0].label + (fs === w9.feature_set ? " — the locked model" : " — the alternative"));
    gd.colSpan = 7;
    gr.appendChild(gd);
    tb.appendChild(gr);
    rows.forEach(r => {
      const tr = el("tr");
      const ci = el("td", "num");
      ci.appendChild(el("span", "ci", f4(r.brier_lo) + "–" + f4(r.brier_hi)));
      tr.append(el("td", "", METHOD[r.method] + (r.chosen ? "  ★" : "")),
                el("td", "num", f4(r.brier)), ci, el("td", "num", dec(r.log_loss, 4)),
                el("td", "num", dec(r.ece, 3)), el("td", "num", dec(r.cal_slope, 2)),
                el("td", "num", r.method === "none" ? "—" : sgn(r.d_brier) +
                   (r.d_hi < 0 ? " ✓" : "")));
      if (r.chosen && fs === w9.feature_set) tr.className = "chosen";
      tr.title = r.method === "none" ? "" : "95% interval of the paired difference: " +
        sgn(r.d_lo) + " to " + sgn(r.d_hi);
      tb.appendChild(tr);
    });
  });
  t.appendChild(tb);
}

function drawSubgroups() {
  const host = $("#c-sub");
  const S = MD.week9.subgroups;
  const wins = ["August", "September"].filter(w => S.some(r => r.window === w));
  const base = S.filter(r => r.window === wins[0]);
  const groups = [];
  base.forEach(r => {
    let g = groups.find(x => x.name === r.group);
    if (!g) { g = { name: r.group, levels: [] }; groups.push(g); }
    if (g.levels.indexOf(r.level) < 0) g.levels.push(r.level);
  });
  const rowH = 19, headH = 22, W = 620;
  const nrow = groups.reduce((a, g) => a + g.levels.length, 0);
  const m = { t: 6, r: 50, b: 34, l: 168 };
  const f = frame(host, W, nrow * rowH + groups.length * headH + m.t + m.b, m);
  const gaps = S.map(r => Math.max(Math.abs(r.lo - r.pred), Math.abs(r.hi - r.pred)) * 100);
  const span = Math.min(60, Math.max(8, Math.max.apply(null, gaps))) * 1.05;
  const X = (v) => ((v + span) / (2 * span)) * f.iw;
  [-span * 0.8, -span * 0.4, 0, span * 0.4, span * 0.8].forEach(v => {
    rule(f.g, X(v), 0, X(v), f.ih, v === 0);
    txt(f.g, X(v), f.ih + 14, (v > 0 ? "+" : "") + v.toFixed(0), "axis-label", "middle");
  });
  txt(f.g, f.iw / 2, f.ih + 28, "observed − predicted (percentage points)", "axis-label",
      "middle");
  const colors = { August: tok("--series-1"), September: tok("--series-2") };
  let y = 0;
  groups.forEach(g => {
    txt(f.g, -m.l + 4, y + headH / 2 + 2, g.name.toUpperCase(), "axis-label", "start")
      .setAttribute("fill", tok("--ink"));
    y += headH;
    g.levels.forEach(lv => {
      const cy = y + rowH / 2;
      txt(f.g, -10, cy, lv, "axis-label", "end").setAttribute("fill", tok("--ink-2"));
      wins.forEach((w, wi) => {
        const r = S.find(x => x.window === w && x.group === g.name && x.level === lv);
        if (!r) return;
        const yy = cy + (wins.length > 1 ? (wi ? 3.5 : -3.5) : 0);
        const gap = (r.obs - r.pred) * 100, lo = (r.lo - r.pred) * 100, hi = (r.hi - r.pred) * 100;
        mk("line", { x1: X(Math.max(-span, lo)), y1: yy, x2: X(Math.min(span, hi)), y2: yy,
                     stroke: colors[w], "stroke-width": 2.4, "stroke-linecap": "round",
                     opacity: r.n >= 30 ? 1 : 0.4 }, f.g);
        mk("circle", { cx: X(Math.max(-span, Math.min(span, gap))), cy: yy, r: 3.6,
                       fill: r.flag ? colors[w] : tok("--paper"), stroke: colors[w],
                       "stroke-width": 1.5 }, f.g);
      });
      const hit = mk("rect", { x: -m.l, y: y, width: W, height: rowH, fill: "transparent" }, f.g);
      hit.addEventListener("pointermove", (e) => showTip(e, wins.map(w => {
        const r = S.find(x => x.window === w && x.group === g.name && x.level === lv);
        return r ? tipRow(pct(r.obs, 0) + " vs " + pct(r.pred, 0) + " predicted",
                          w + " · " + lv + " · n=" + nf(r.n) + (r.flag ? " · off" : "")) : "";
      }).join('<div style="height:5px"></div>')));
      hit.addEventListener("pointerleave", hideTip);
      y += rowH;
    });
  });
  legend($("#l-sub"), wins.map(w => ({ label: w + (w === "August" ? " (cross-fitted)" : " (test)"),
                                      color: colors[w], sq: true }))
    .concat([{ label: "filled dot = off by more than chance (n ≥ 30)", color: "transparent",
               sq: true }]));
  const nf_ = base.filter(r => r.flag).length, nt = base.filter(r => r.n >= 30).length;
  $("#m-sub-title").textContent = nf_ === 0
    ? "No slice of " + wins[0] + " is off by more than chance"
    : nf_ + " of " + nt + " slices are off by more than chance (≈" +
      (0.05 * nt).toFixed(1) + " expected by luck)";
}

function drawMonths() {
  const host = $("#c-months");
  const R = MD.week9.months.slice().sort((a, b) => a.level.localeCompare(b.level));
  if (!R.length) { note(host, "No month table yet."); return; }
  const keys = [];
  R.forEach(r => { const k = r.level.slice(0, 2); if (keys.indexOf(k) < 0) keys.push(k); });
  const f = frame(host, 620, 250, { t: 16, r: 10, b: 34, l: 46 });
  const gaps = R.map(r => Math.max(Math.abs(r.lo - r.pred), Math.abs(r.hi - r.pred)) * 100);
  const span = Math.max(6, Math.max.apply(null, gaps)) * 1.1;
  const Y = (v) => f.ih / 2 - (v / span) * (f.ih / 2);
  const step = f.iw / keys.length;
  [-span * 0.8, -span * 0.4, 0, span * 0.4, span * 0.8].forEach(v => {
    rule(f.g, 0, Y(v), f.iw, Y(v), v === 0);
    txt(f.g, -8, Y(v), (v > 0 ? "+" : "") + v.toFixed(0), "axis-label", "end");
  });
  R.forEach(r => {
    const i = keys.indexOf(r.level.slice(0, 2)), cx = i * step + step / 2;
    const train = r.window.indexOf("train") === 0;
    const col = train ? tok("--ink-3") : r.window === "August" ? tok("--series-1")
                                                                : tok("--series-2");
    const lo = (r.lo - r.pred) * 100, hi = (r.hi - r.pred) * 100, g = (r.obs - r.pred) * 100;
    mk("line", { x1: cx, y1: Y(lo), x2: cx, y2: Y(hi), stroke: col, "stroke-width": 3,
                 "stroke-linecap": "round" }, f.g);
    const c = mk("circle", { cx: cx, cy: Y(g), r: 5.5, fill: train ? tok("--paper") : col,
                             stroke: col, "stroke-width": 1.8 }, f.g);
    c.addEventListener("pointermove", (e) => showTip(e, tipRow(
      pct(r.obs, 1) + " vs " + pct(r.pred, 1) + " predicted",
      r.level.slice(3) + " · " + r.window + " · n=" + nf(r.n))));
    c.addEventListener("pointerleave", hideTip);
    txt(f.g, cx, f.ih + 15, r.level.slice(3), "axis-label", "middle");
  });
  txt(f.g, -38, -6, "pts", "axis-label", "start");
  legend($("#l-months"), [{ label: "training months, out-of-fold", color: tok("--ink-3"), sq: true },
                          { label: "August (calibration)", color: tok("--series-1"), sq: true }]
    .concat(R.some(r => r.window === "September")
      ? [{ label: "September (test)", color: tok("--series-2"), sq: true }] : []));
}

function drawImportance() {
  const host = $("#c-imp");
  const I = MD.week8 && MD.week8.importance;
  const d = I && (I.full || I.location);
  if (!d || !d.rows || !d.rows.length) { note(host, "No importance table yet."); return; }
  const rows = d.rows.slice(0, 10);
  const vals = rows.map(r => r.importance).filter(v => v > 0);
  const vmax = Math.max.apply(null, vals.concat([1e-9]));
  const vmin = Math.max(Math.min.apply(null, vals.concat([vmax])) / 3, vmax * 1e-6);
  const lmin = Math.log10(Math.min(vmin, vmax / 100)), lmax = Math.log10(vmax * 4);
  const f = frame(host, 620, rows.length * 26 + 46, { t: 6, r: 70, b: 34, l: 168 });
  const X = (v) => ((Math.log10(Math.max(v, Math.pow(10, lmin))) - lmin) / (lmax - lmin)) * f.iw;
  for (let e = Math.ceil(lmin); e <= Math.floor(lmax); e++) {
    mk("line", { x1: X(Math.pow(10, e)), y1: 0, x2: X(Math.pow(10, e)), y2: f.ih,
                 class: "tick-line" }, f.g);
    txt(f.g, X(Math.pow(10, e)), f.ih + 14, String(+Math.pow(10, e).toPrecision(1)),
        "axis-label", "middle");
  }
  txt(f.g, f.iw / 2, f.ih + 28, (d.kind === "shap" ? "mean |SHAP|, log-odds" :
      "rise in log loss when shuffled") + " (log scale: each gridline ×10)", "axis-label",
      "middle");
  const BC = { location: tok("--series-1"), identity: tok("--series-2") };
  rows.forEach((r, i) => {
    const cy = i * 26 + 13, col = BC[r.block] || tok("--ink-3");
    mk("line", { x1: 0, y1: cy, x2: f.iw, y2: cy, class: "tick-line" }, f.g);
    txt(f.g, -10, cy, r.label, "val-label", "end").setAttribute("fill", tok("--ink-2"));
    mk("circle", { cx: X(r.importance), cy: cy, r: 6, fill: r.importance > 0 ? col :
                   tok("--paper"), stroke: col, "stroke-width": 1.6 }, f.g);
    txt(f.g, X(r.importance) + 11, cy, r.importance > 0 ? String(+r.importance.toPrecision(3))
        : (d.kind === "shap" ? "0" : "≤ 0"), "axis-label", "start");
    txt(f.g, f.iw + 8, cy, r.block, "axis-label", "start");
  });
  legend($("#l-imp"), [{ label: "location — what ABS sees", color: BC.location, sq: true },
                       { label: "identity — umpire, catcher, pitcher, batter IDs",
                         color: BC.identity, sq: true },
                       { label: "situation, pitch, people", color: tok("--ink-3"), sq: true }]);
  const lk = MD.week8.leak || {};
  $("#m-imp-title").textContent = lk.flagged
    ? "⚠ An ID outranks location — the leak check flagged this model"
    : "Location does the work; the leak check passed";
  $("#m-imp-src").textContent = (d.label || "") + " · " + (lk.detail || "").replace(/`/g, "");
}

function drawDecision() {
  const host = $("#m-decision");
  const w9 = MD.week9;
  host.innerHTML = "";
  const D = w9.decision || {};
  host.appendChild(el("h4", "", D.headline || ("p(s) = " + w9.model_label)));
  const ul = el("ul");
  ["why_model", "why_candidate", "why_calibration", "subgroups"].forEach(k => {
    if (D[k]) ul.appendChild(el("li", "", D[k]));
  });
  host.appendChild(ul);
  host.appendChild(el("h4", "", "Using it (Weeks 10–12)"));
  const pre = el("pre", "code");
  pre.textContent = "from src.abs_predict import load_production\n" +
    "pm = load_production()      # hashes checked\n" +
    "pm.predict({\"plate_x\": 0.85, \"plate_z\": 2.40,\n" +
    "            \"batter_height_ft\": 6.0,\n" +
    "            \"called_strike\": True})";
  host.appendChild(pre);
  host.appendChild(el("p", "", "Version " + w9.version + " · content hash " + w9.content_hash +
    "… · fields: " + w9.state_fields.join(", ") + "." +
    (w9.test.opened ? " Test month opened " + String(w9.test.opened_at).slice(0, 10) +
      (w9.test.times_opened > 1 ? " (" + w9.test.times_opened + " times)" : " (once)") + "."
      : " Test month still sealed.")));
}

function initModelControls() {
  const hs = $("#m-height");
  hs.innerHTML = "";
  MD.week9.surface.heights.forEach((h, i) => {
    const b = el("button", "chip", h.label);
    b.setAttribute("aria-pressed", String(i === view.h));
    b.addEventListener("click", () => {
      view.h = i;
      [...hs.children].forEach((c, j) => c.setAttribute("aria-pressed", String(j === i)));
      drawSurface();
    });
    hs.appendChild(b);
  });
  [...$("#m-call").children].forEach(b => b.addEventListener("click", () => {
    view.call = b.dataset.call;
    [...$("#m-call").children].forEach(c =>
      c.setAttribute("aria-pressed", String(c === b)));
    drawSurface();
  }));
  const win = $("#m-window");
  if (!(MD.week9 && MD.week9.test.opened)) {
    win.hidden = true;
  } else {
    [...win.children].forEach(b => b.addEventListener("click", () => {
      view.win = b.dataset.win;
      [...win.children].forEach(c => c.setAttribute("aria-pressed", String(c === b)));
      drawCompare();
    }));
  }
}

function renderModel() {
  const miss = $("#m-missing");
  const hide = (cls, on) => document.querySelectorAll("#p-model ." + cls)
    .forEach(e => { e.hidden = on; });
  if (!MD || (!MD.week8 && !MD.week9)) {
    miss.hidden = false;
    $("#m-body").hidden = true;
    return;
  }
  modelStrip();
  hide("w8only", !MD.week8);
  hide("w9only", !MD.week9);
  $("#m-window").hidden = !(MD.week9 && MD.week9.test && MD.week9.test.opened);
  if (MD.week8) {
    drawCompare();
    drawImportance();
  }
  if (MD.week9) {
    drawSurface();
    drawReliability();
    drawCalTable();
    drawSubgroups();
    drawMonths();
    drawDecision();
  } else {
    miss.hidden = false;
    miss.innerHTML = "<b>Week 9 not run yet</b> &mdash; run " +
      "<code>scripts/17_week9_calibrate_lock.py</code>, then " +
      "<code>scripts/18_export_model_dashboard.py</code>.";
  }
}

/* =========================================================================
   PROGRESS TAB
   ========================================================================= */
function renderProgress() {
  const host = $("#pr-list");
  if (!MD) { note(host, "model_data.js did not load."); return; }
  host.innerHTML = "";
  const P = MD.progress;
  const done = P.filter(r => r.status === "done").length;
  const next = P.find(r => r.status === "next");
  const strip = $("#pr-strip");
  strip.innerHTML = "";
  const D = window.ABS_DATA || {};
  [{ k: "Fall weeks done", v: done + " of " + P.filter(r => r.week !== 13).length,
     sub: "Weeks 3–15 of the plan" },
   { k: "Up next", v: next ? "Week " + next.week : "—", sub: next ? next.title : "" },
   { k: "Pitches joined", v: D.meta ? U.compact(D.meta.n_pitches) : "—",
     sub: D.meta ? nf(D.meta.n_games) + " games" : "" },
   { k: "p(s)", v: MD.week9 ? "Locked" : MD.week8 ? "Week 8" : "—",
     sub: MD.week9 ? MD.week9.version : "" }].forEach(it => {
    const d = el("div", "stat");
    d.append(el("span", "k", it.k), el("span", "v", it.v), el("span", "sub", it.sub));
    strip.appendChild(d);
  });
  P.forEach(r => {
    const li = el("li", "tl-item tl-" + r.status);
    const left = el("div", "tl-when");
    left.append(el("span", "tl-week", "Week " + r.week), el("span", "tl-dates", r.dates));
    const body = el("div", "tl-body");
    const h = el("div", "tl-title");
    h.append(document.createTextNode(r.title + " "),
             el("span", "tl-pill", r.status === "done" ? "Done" : r.status === "next" ? "Next"
                                                                                    : "Planned"));
    body.append(h, el("div", "tl-deliv", r.deliverable));
    if (r.result) body.appendChild(el("div", "tl-result", r.result));
    li.append(left, body);
    host.appendChild(li);
  });
}

function renderAll() {
  try { renderModel(); } catch (e) { console.error("model tab:", e); }
  try { renderProgress(); } catch (e) { console.error("progress tab:", e); }
}

if (MD && MD.meta && MD.meta.is_demo && $("#demoBanner")) $("#demoBanner").hidden = false;
if (MD && MD.week9) initModelControls();
renderAll();
const tb = $("#themeBtn");
if (tb) tb.addEventListener("click", () => setTimeout(renderAll, 0));
})();
