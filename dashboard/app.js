/* ===========================================================================
   ABS Challenge Analytics — application code
   ---------------------------------------------------------------------------
   Plain JavaScript, no framework and no build step. Open index.html and it runs.

   HOW IT FITS TOGETHER
     data.js   exports window.ABS_DATA (written by scripts/12_export_dashboard_data.py)
     filters   produce a subset of ABS_DATA.challenges, plus a matching slice of
               the pre-aggregated opportunity cube
     render()  redraws every chart and table from that subset

   Charts are drawn as SVG by the small helpers in the CHART PRIMITIVES section.
   There is no charting library on purpose: you can read every line, and adding a
   chart means copying one of these functions rather than learning an API.

   TO ADD A CHART
     1. put a <div id="c-yourchart"></div> in index.html
     2. write a draw function below
     3. call it from render()
   =========================================================================== */

(function () {
"use strict";

const D = window.ABS_DATA;
const $  = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

if (!D) {
  document.querySelector("main").innerHTML =
    '<p class="prose" style="padding:40px 0">' +
    '<b>data.js did not load.</b> Run ' +
    '<code>python scripts/12_export_dashboard_data.py</code> and make sure ' +
    '<code>data.js</code> sits next to <code>index.html</code>.</p>';
  return;
}

/* --------------------------------------------------------------- formatting */
const nf   = (n) => (n === null || n === undefined || !isFinite(n)) ? "—" : n.toLocaleString("en-US");
const pct  = (n, d) => (n === null || n === undefined || !isFinite(n)) ? "—" : (n * 100).toFixed(d === undefined ? 1 : d) + "%";
const dec  = (n, d) => (n === null || n === undefined || !isFinite(n)) ? "—" : n.toFixed(d === undefined ? 1 : d);
const compact = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M"
                     : n >= 1e4 ? Math.round(n / 1e3) + "K" : nf(n);

/* Colours live in CSS so the theme switch works. Read them at draw time. */
function tok(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
const ROLE_TOKEN = { batter: "--series-1", catcher: "--series-2", pitcher: "--series-3" };
const roleColor = (r) => tok(ROLE_TOKEN[r] || "--deemph");

/* ------------------------------------------------------------------ tooltip */
const tipEl = $("#tip");
function showTip(e, html) {
  tipEl.innerHTML = html;                 // built only from our own data
  tipEl.classList.add("on");
  const pad = 14, r = tipEl.getBoundingClientRect();
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tipEl.style.left = Math.max(8, x) + "px";
  tipEl.style.top  = Math.max(8, y) + "px";
}
function hideTip() { tipEl.classList.remove("on"); }
function tipRow(value, key) {
  return '<span class="tv">' + value + '</span><span class="tk">' + key + "</span>";
}

/* =========================================================================
   CHART PRIMITIVES
   Each returns an <svg> sized by a viewBox, so it scales to its container.
   ========================================================================= */
const NS = "http://www.w3.org/2000/svg";
function mk(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function frame(host, w, h, m) {
  host.innerHTML = "";
  const svg = mk("svg", {
    viewBox: "0 0 " + w + " " + h, class: "chart",
    preserveAspectRatio: "xMidYMid meet", role: "img"
  }, host);
  const g = mk("g", { transform: "translate(" + m.l + "," + m.t + ")" }, svg);
  return { svg, g, iw: w - m.l - m.r, ih: h - m.t - m.b, m };
}
function txt(parent, x, y, s, cls, anchor) {
  const t = mk("text", {
    x: x, y: y, class: cls || "axis-label",
    "text-anchor": anchor || "start", "dominant-baseline": "middle"
  }, parent);
  t.textContent = s;
  return t;
}
/** Horizontal bars. rows: [{label, value, color?, note?}] */
function barsH(host, rows, opts) {
  opts = opts || {};
  const rowH = opts.rowH || 30, padL = opts.padL || 140;
  const h = rows.length * rowH + 26, w = 620;
  const f = frame(host, w, h, { t: 6, r: 96, b: 20, l: padL });
  const max = Math.max.apply(null, rows.map(r => r.value)) || 1;
  rows.forEach((r, i) => {
    const y = i * rowH, bh = Math.min(20, rowH - 10);
    txt(f.g, -10, y + rowH / 2, r.label, "val-label", "end")
      .setAttribute("fill", tok("--ink"));
    const bw = Math.max(1, (r.value / max) * f.iw);
    const rect = mk("rect", {
      x: 0, y: y + (rowH - bh) / 2, width: bw, height: bh,
      fill: r.color || tok("--series-1"), rx: 1
    }, f.g);
    txt(f.g, bw + 8, y + rowH / 2, r.note || nf(r.value), "val-label", "start");
    const hit = mk("rect", { x: 0, y: y, width: f.iw, height: rowH, fill: "transparent" }, f.g);
    const label = r.label, val = r.tip || (nf(r.value) + " challenges");
    hit.addEventListener("pointermove", (e) => showTip(e, tipRow(val, label)));
    hit.addEventListener("pointerleave", hideTip);
    rect.style.pointerEvents = "none";
  });
  return f;
}
/** Vertical columns. rows: [{label, value, note?}] */
function columns(host, rows, opts) {
  opts = opts || {};
  const w = 620, h = opts.h || 230;
  const f = frame(host, w, h, { t: 18, r: 8, b: 30, l: 42 });
  const max = Math.max.apply(null, rows.map(r => r.value)) || 1;
  const step = f.iw / rows.length, bw = Math.min(opts.maxBar || 34, step - 8);
  [0, .5, 1].forEach(p => {
    const y = f.ih - p * f.ih;
    mk("line", { x1: 0, y1: y, x2: f.iw, y2: y, class: "tick-line" }, f.g);
    txt(f.g, -8, y, opts.pctAxis ? pct(p * max, 0) : nf(Math.round(p * max)),
        "axis-label", "end");
  });
  rows.forEach((r, i) => {
    const x = i * step + (step - bw) / 2;
    const bh = Math.max(1, (r.value / max) * f.ih);
    mk("rect", { x: x, y: f.ih - bh, width: bw, height: bh,
                 fill: r.color || tok("--series-1"), rx: 1 }, f.g);
    txt(f.g, x + bw / 2, f.ih + 15, r.label, "axis-label", "middle");
    const hit = mk("rect", { x: i * step, y: 0, width: step, height: f.ih,
                             fill: "transparent" }, f.g);
    hit.addEventListener("pointermove",
      (e) => showTip(e, tipRow(r.note || nf(r.value), r.tipKey || r.label)));
    hit.addEventListener("pointerleave", hideTip);
  });
  return f;
}
/** Line with dots. pts: [{label, value, n}] — value is 0..1 */
function lineChart(host, pts) {
  const w = 620, h = 240;
  const f = frame(host, w, h, { t: 18, r: 22, b: 42, l: 42 });
  const max = Math.min(1, Math.max.apply(null, pts.map(p => p.value)) * 1.25 + .05);
  const X = (i) => pts.length < 2 ? f.iw / 2 : (i / (pts.length - 1)) * f.iw;
  const Y = (v) => f.ih - (v / max) * f.ih;
  [0, .5, 1].forEach(p => {
    const y = f.ih - p * f.ih;
    mk("line", { x1: 0, y1: y, x2: f.iw, y2: y, class: "tick-line" }, f.g);
    txt(f.g, -8, y, pct(p * max, 0), "axis-label", "end");
  });
  const d = pts.map((p, i) => (i ? "L" : "M") + X(i) + " " + Y(p.value)).join(" ");
  mk("path", { d: d, fill: "none", stroke: tok("--series-1"), "stroke-width": 2,
               "stroke-linejoin": "round", "stroke-linecap": "round" }, f.g);
  pts.forEach((p, i) => {
    mk("circle", { cx: X(i), cy: Y(p.value), r: 5, fill: tok("--series-1"),
                   stroke: tok("--paper"), "stroke-width": 2 }, f.g);
    txt(f.g, X(i), f.ih + 16, p.label, "axis-label", "middle");
    txt(f.g, X(i), f.ih + 31, "n=" + p.n, "axis-label", "middle")
      .setAttribute("fill", tok("--ink-3"));
    const hit = mk("circle", { cx: X(i), cy: Y(p.value), r: 16, fill: "transparent" }, f.g);
    hit.addEventListener("pointermove", (e) => showTip(e,
      tipRow(pct(p.value), p.label + " in · " + p.n + " challenges")));
    hit.addEventListener("pointerleave", hideTip);
  });
  return f;
}
/** Stacked area, 3 ordered bands. rows: [{x, a, b, c}] a is bottom. */
function stackArea(host, rows, colors, keys) {
  const w = 620, h = 250;
  const f = frame(host, w, h, { t: 10, r: 10, b: 30, l: 42 });
  const X = (i) => rows.length < 2 ? 0 : (i / (rows.length - 1)) * f.iw;
  const Y = (v) => f.ih - v * f.ih;
  [0, .5, 1].forEach(p => {
    const y = Y(p);
    mk("line", { x1: 0, y1: y, x2: f.iw, y2: y, class: "tick-line" }, f.g);
    txt(f.g, -8, y, pct(p, 0), "axis-label", "end");
  });
  let base = rows.map(() => 0);
  keys.forEach((k, ki) => {
    const top = rows.map((r, i) => base[i] + r[k]);
    let d = rows.map((r, i) => (i ? "L" : "M") + X(i) + " " + Y(top[i])).join(" ");
    for (let i = rows.length - 1; i >= 0; i--) d += " L" + X(i) + " " + Y(base[i]);
    mk("path", { d: d + " Z", fill: colors[ki], stroke: tok("--paper"),
                 "stroke-width": 1.5 }, f.g);
    base = top;
  });
  rows.forEach((r, i) => {
    txt(f.g, X(i), f.ih + 16, r.x, "axis-label", "middle");
    const hit = mk("rect", { x: X(i) - f.iw / rows.length / 2, y: 0,
                             width: f.iw / rows.length, height: f.ih,
                             fill: "transparent" }, f.g);
    hit.addEventListener("pointermove", (e) => showTip(e,
      keys.map((k, ki) => tipRow(pct(r[k]), labelFor(k))).join(
        '<div style="height:5px"></div>') +
      '<div class="tk" style="margin-top:4px">Inning ' + r.x + "</div>"));
    hit.addEventListener("pointerleave", hideTip);
  });
  function labelFor(k) { return { two: "2 remaining", one: "1 remaining", zero: "0 remaining" }[k]; }
  return f;
}
/** Heat grid. cells: [{row, col, value, n}] */
function heat(host, cells, rowLabels, colLabels, rowTitle, colTitle) {
  const w = 620, cw = 108, ch = 54;
  const h = rowLabels.length * ch + 60;
  const f = frame(host, w, h, { t: 22, r: 10, b: 30, l: 46 });
  const vals = cells.map(c => c.value).filter(v => v !== null && isFinite(v));
  const lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  const ramp = [1, 2, 3, 4, 5, 6, 7].map(i => tok("--seq-" + i));
  colLabels.forEach((cl, j) => txt(f.g, j * cw + cw / 2, -10, cl, "axis-label", "middle"));
  rowLabels.forEach((rl, i) => txt(f.g, -10, i * ch + ch / 2, rl, "axis-label", "end"));
  txt(f.g, colLabels.length * cw / 2, rowLabels.length * ch + 20, colTitle, "axis-label", "middle");
  const rt = txt(f.g, -34, rowLabels.length * ch / 2, rowTitle, "axis-label", "middle");
  rt.setAttribute("transform", "rotate(-90 -34 " + (rowLabels.length * ch / 2) + ")");
  cells.forEach(c => {
    const i = rowLabels.indexOf(c.row), j = colLabels.indexOf(c.col);
    if (i < 0 || j < 0) return;
    const rel = (hi - lo) < 1e-9 ? .5 : (c.value - lo) / (hi - lo);
    const fill = c.value === null ? tok("--deemph")
               : ramp[Math.min(6, Math.max(0, Math.round(rel * 6)))];
    mk("rect", { x: j * cw + 1, y: i * ch + 1, width: cw - 2, height: ch - 2,
                 fill: fill }, f.g);
    const t = txt(f.g, j * cw + cw / 2, i * ch + ch / 2 - 5,
                  c.value === null ? "—" : pct(c.value), "val-label", "middle");
    t.setAttribute("fill", rel > .55 ? "#ffffff" : tok("--ink"));
    t.setAttribute("font-size", "14");
    const t2 = txt(f.g, j * cw + cw / 2, i * ch + ch / 2 + 12,
                   "n=" + nf(c.n), "axis-label", "middle");
    t2.setAttribute("fill", rel > .55 ? "rgba(255,255,255,.8)" : tok("--ink-3"));
    const hit = mk("rect", { x: j * cw, y: i * ch, width: cw, height: ch,
                             fill: "transparent" }, f.g);
    hit.addEventListener("pointermove", (e) => showTip(e,
      tipRow(c.value === null ? "no data" : pct(c.value),
             c.col + "–" + c.row + " · " + nf(c.n) + " opportunities")));
    hit.addEventListener("pointerleave", hideTip);
  });
  return f;
}
/** Histogram. values: number[] */
function histogram(host, values, lo, hi, nbins, xlabel, opts) {
  opts = opts || {};
  const w = 620, h = 230;
  const f = frame(host, w, h, { t: 14, r: 10, b: 38, l: 42 });
  const bins = new Array(nbins).fill(0), bw = (hi - lo) / nbins;
  values.forEach(v => {
    if (v === null || !isFinite(v)) return;
    const i = Math.min(nbins - 1, Math.max(0, Math.floor((v - lo) / bw)));
    bins[i]++;
  });
  const max = Math.max.apply(null, bins) || 1;
  [0, .5, 1].forEach(p => {
    const y = f.ih - p * f.ih;
    mk("line", { x1: 0, y1: y, x2: f.iw, y2: y, class: "tick-line" }, f.g);
    txt(f.g, -8, y, nf(Math.round(p * max)), "axis-label", "end");
  });
  const step = f.iw / nbins;
  bins.forEach((n, i) => {
    const bh = (n / max) * f.ih;
    mk("rect", { x: i * step + 1, y: f.ih - bh, width: step - 2, height: Math.max(0, bh),
                 fill: tok("--series-1") }, f.g);
    const hit = mk("rect", { x: i * step, y: 0, width: step, height: f.ih,
                             fill: "transparent" }, f.g);
    const a = (lo + i * bw).toFixed(1), b = (lo + (i + 1) * bw).toFixed(1);
    hit.addEventListener("pointermove", (e) => showTip(e,
      tipRow(nf(n) + " challenges", a + " to " + b)));
    hit.addEventListener("pointerleave", hideTip);
  });
  // zero reference (suppressed when the caller marks its own edges)
  if (lo < 0 && hi > 0 && !opts.noZero) {
    const zx = ((0 - lo) / (hi - lo)) * f.iw;
    mk("line", { x1: zx, y1: 0, x2: zx, y2: f.ih, stroke: tok("--ink"),
                 "stroke-width": 1.5 }, f.g);
    txt(f.g, zx + 5, 10, "zone edge", "axis-label", "start")
      .setAttribute("fill", tok("--ink-2"));
  }
  [0, .25, .5, .75, 1].forEach(p => {
    txt(f.g, p * f.iw, f.ih + 16, dec(lo + p * (hi - lo), 1), "axis-label", "middle");
  });
  txt(f.g, f.iw / 2, f.ih + 32, xlabel, "axis-label", "middle");
  return f;
}

/* =========================================================================
   FILTERS
   ========================================================================= */
const state = { role: "all", month: "all", inning: "all", rem: "all",
                team: "all", outcome: "all" };

function filterChallenges() {
  return D.challenges.filter(c => {
    if (state.role !== "all" && c.role !== state.role) return false;
    if (state.month !== "all" && c.mo !== state.month) return false;
    if (state.inning !== "all" && c.ib !== state.inning) return false;
    if (state.rem !== "all" && String(c.rem) !== state.rem) return false;
    if (state.team !== "all" && c.team !== state.team) return false;
    if (state.outcome === "ov" && c.ov !== true) return false;
    if (state.outcome === "st" && c.ov !== false) return false;
    return true;
  });
}
/* The cube holds opportunity counts. A role filter maps onto it only coarsely:
   an opportunity belongs to a SIDE, and we only learn which fielder challenged
   after the fact — so catcher and pitcher share one denominator. */
function filterCube() {
  const rg = state.role === "all" ? null
           : (state.role === "batter" ? "batter" : "fielding");
  return D.cube.filter(c => {
    if (rg && c.rg !== rg) return false;
    if (state.month !== "all" && c.mo !== state.month) return false;
    if (state.inning !== "all" && c.ib !== state.inning) return false;
    if (state.rem !== "all" && String(c.rem) !== state.rem) return false;
    return true;
  });
}
function activeFilterCount() {
  return Object.keys(state).filter(k => state[k] !== "all").length;
}

/* =========================================================================
   RENDER
   ========================================================================= */
function statStrip(host, ch, cube) {
  const opp = cube.reduce((a, c) => a + c.o, 0);
  const resolved = ch.filter(c => c.ov !== null);
  const won = resolved.filter(c => c.ov).length;
  const items = [
    { k: "Challenges", v: compact(ch.length),
      sub: activeFilterCount() ? "in this selection" : "all season" },
    { k: "Overturn rate", v: resolved.length ? pct(won / resolved.length) : "—",
      sub: nf(won) + " of " + nf(resolved.length) + " resolved" },
    { k: "Opportunities", v: compact(opp),
      sub: opp ? pct(ch.length / opp, 2) + " challenged" : "—" },
    { k: "Median distance",
      v: resolved.length ? dec(median(ch.map(c => c.d)), 1) + '"' : "—",
      sub: "from the ABS zone edge" },
    { k: "Games", v: compact(new Set(ch.map(c => c.g)).size), sub: "with a challenge" }
  ];
  host.innerHTML = "";
  items.forEach(it => {
    const d = document.createElement("div");
    d.className = "stat";
    const k = document.createElement("span"); k.className = "k"; k.textContent = it.k;
    const v = document.createElement("span"); v.className = "v"; v.textContent = it.v;
    const s = document.createElement("span"); s.className = "sub"; s.textContent = it.sub;
    d.append(k, v, s); host.appendChild(d);
  });
}
function median(a) {
  const v = a.filter(x => x !== null && isFinite(x)).sort((x, y) => x - y);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function drawFunnel() {
  const rows = D.funnel.map((s, i) => ({
    label: s.stage, value: s.n,
    note: nf(s.n) + "  (" + pct(s.n / D.funnel[0].n, 1) + ")",
    color: tok("--seq-" + Math.min(7, 3 + i)),
    tip: nf(s.n) + " pitches"
  }));
  barsH($("#c-funnel"), rows, { padL: 150, rowH: 34 });
}

function drawDistanceCurve() {
  const host = $("#c-distance");
  if (!D.distance_curve.length) { host.innerHTML = empty(); return; }
  lineChart(host, D.distance_curve.map(b => ({
    label: b.bucket, value: b.rate, n: b.n })));
}

function drawInventory() {
  const host = $("#c-inventory");
  if (!D.inventory.length) { host.innerHTML = empty(); return; }
  stackArea(host, D.inventory.map(r => ({ x: r.inn, two: r.two, one: r.one, zero: r.zero })),
            [tok("--seq-7"), tok("--seq-5"), tok("--seq-3")], ["two", "one", "zero"]);
  legend($("#l-inventory"), [
    { label: "2 remaining", color: tok("--seq-7"), sq: true },
    { label: "1 remaining", color: tok("--seq-5"), sq: true },
    { label: "0 remaining", color: tok("--seq-3"), sq: true }
  ]);
}

function drawRole(ch) {
  const host = $("#c-role");
  const roles = ["batter", "catcher", "pitcher"];
  const rows = roles.map(r => {
    const sub = ch.filter(c => c.role === r);
    const res = sub.filter(c => c.ov !== null);
    const won = res.filter(c => c.ov).length;
    return {
      label: r.charAt(0).toUpperCase() + r.slice(1),
      value: sub.length, color: roleColor(r),
      note: res.length ? pct(won / res.length, 0) + " overturned" : "—",
      tip: nf(sub.length) + " challenges"
    };
  }).filter(r => r.value > 0);
  if (!rows.length) { host.innerHTML = empty(); return; }
  barsH(host, rows, { padL: 86, rowH: 38 });
}

function drawCountHeat(cube) {
  const host = $("#c-count");
  const agg = {};
  cube.forEach(c => {
    const k = c.b + "|" + c.s;
    if (!agg[k]) agg[k] = { o: 0, c: 0 };
    agg[k].o += c.o; agg[k].c += c.c;
  });
  const cells = [];
  for (let s = 0; s <= 2; s++) for (let b = 0; b <= 3; b++) {
    const a = agg[b + "|" + s];
    cells.push({ row: String(s), col: String(b),
                 value: a && a.o ? a.c / a.o : null, n: a ? a.o : 0 });
  }
  if (!cells.some(c => c.n)) { host.innerHTML = empty(); return; }
  heat(host, cells, ["0", "1", "2"], ["0", "1", "2", "3"], "Strikes", "Balls");
}

function drawInning(ch) {
  const host = $("#c-inning");
  const counts = {};
  ch.forEach(c => { counts[c.inn] = (counts[c.inn] || 0) + 1; });
  const innings = Object.keys(counts).map(Number).sort((a, b) => a - b);
  if (!innings.length) { host.innerHTML = empty(); return; }
  columns(host, innings.map(i => ({
    label: String(i), value: counts[i],
    note: nf(counts[i]) + " challenges", tipKey: "Inning " + i })));
}

function drawMonth(ch) {
  const host = $("#c-month");
  const counts = {};
  ch.forEach(c => { counts[c.mo] = (counts[c.mo] || 0) + 1; });
  const months = D.months.filter(m => counts[m]);
  if (!months.length) { host.innerHTML = empty(); return; }
  columns(host, months.map(m => ({
    label: m, value: counts[m], note: nf(counts[m]) + " challenges", tipKey: m })));
}

function drawDistHist(ch) {
  const host = $("#c-disthist");
  const v = ch.map(c => c.d).filter(d => d !== null && Math.abs(d) <= 12);
  if (v.length < 5) { host.innerHTML = empty(); return; }
  histogram(host, v, -6, 12, 36, "Distance from the ABS zone edge (inches)");
}

/* Strike zone scatter. Draws the zone for a median-height batter and plots each
   challenged pitch, coloured by whether the call was overturned. */
function drawZone(ch) {
  const host = $("#c-zone");
  let pts = ch.filter(c => c.px !== null && c.pz !== null && c.zt && c.zb);
  if (!pts.length) { host.innerHTML = empty(); return; }
  // A full season is ~11,000 challenges, and each point here costs two SVG
  // nodes (the mark plus a bigger invisible hover target). Past a few thousand
  // the browser starts to crawl, so take an evenly spaced sample instead —
  // deterministic, so the picture doesn't reshuffle every redraw.
  const CAP = 3500, total = pts.length;
  if (total > CAP) {
    const stride = total / CAP;
    pts = Array.from({ length: CAP }, (_, i) => pts[Math.floor(i * stride)]);
  }
  const top = median(pts.map(p => p.zt)), bot = median(pts.map(p => p.zb));
  const HW = 8.5 / 12, R = (2.9 / 2) / 12;
  const w = 480, h = 470;
  const f = frame(host, w, h, { t: 12, r: 12, b: 34, l: 42 });
  const xlo = -1.9, xhi = 1.9, ylo = Math.max(0, bot - .95), yhi = top + .95;
  const X = (v) => ((v - xlo) / (xhi - xlo)) * f.iw;
  const Y = (v) => f.ih - ((v - ylo) / (yhi - ylo)) * f.ih;

  for (let gx = -1.5; gx <= 1.5; gx += .5) {
    mk("line", { x1: X(gx), y1: 0, x2: X(gx), y2: f.ih, class: "tick-line" }, f.g);
    txt(f.g, X(gx), f.ih + 15, dec(gx, 1), "axis-label", "middle");
  }
  for (let gy = Math.ceil(ylo * 2) / 2; gy <= yhi; gy += .5) {
    mk("line", { x1: 0, y1: Y(gy), x2: f.iw, y2: Y(gy), class: "tick-line" }, f.g);
    txt(f.g, -8, Y(gy), dec(gy, 1), "axis-label", "end");
  }
  mk("rect", { x: X(-HW - R), y: Y(top + R), width: X(HW + R) - X(-HW - R),
               height: Y(bot - R) - Y(top + R), fill: "none",
               stroke: tok("--ink-3"), "stroke-width": 1 }, f.g);
  mk("rect", { x: X(-HW), y: Y(top), width: X(HW) - X(-HW),
               height: Y(bot) - Y(top), fill: "none",
               stroke: tok("--ink"), "stroke-width": 2 }, f.g);
  txt(f.g, X(HW + R) + 6, Y(top + R), "ABS zone", "val-label", "start")
    .setAttribute("fill", tok("--ink"));
  txt(f.g, X(HW + R) + 6, Y(bot - R), "+ ball radius", "axis-label", "start");

  pts.forEach(p => {
    const c = mk("circle", {
      cx: X(p.px), cy: Y(p.pz), r: 4.5,
      fill: p.ov ? tok("--series-2") : tok("--series-1"),
      "fill-opacity": .75, stroke: tok("--paper"), "stroke-width": 1
    }, f.g);
    const hit = mk("circle", { cx: X(p.px), cy: Y(p.pz), r: 12, fill: "transparent" }, f.g);
    hit.addEventListener("pointermove", (e) => showTip(e,
      tipRow(p.ov ? "Overturned" : "Call stood",
             (p.who || p.role) + " · " + p.b + "-" + p.s + " · " +
             p.half + p.inn + " · " + dec(p.d, 1) + " in")));
    hit.addEventListener("pointerleave", hideTip);
    c.style.pointerEvents = "none";
  });
  legend($("#l-zone"), [
    { label: "Call stood", color: tok("--series-1"), sq: true },
    { label: "Overturned", color: tok("--series-2"), sq: true }
  ].concat(total > CAP
    ? [{ label: "showing an even sample of " + nf(CAP) + " of " + nf(total),
         color: "transparent", sq: true }]
    : []));
}

/* Where in the batter's OWN zone the pitch sat: 0 = bottom edge, 1 = top edge.
   Challenges pile up at both edges, which is the argument for per-batter zones. */
function drawZoneCmp(ch) {
  const host = $("#c-zonecmp");
  const v = ch.filter(c => c.pz !== null && c.zt && c.zb && c.zt > c.zb)
              .map(c => (c.pz - c.zb) / (c.zt - c.zb))
              .filter(x => x > -1.2 && x < 2.2);
  if (v.length < 5) { host.innerHTML = empty(); return; }
  const f = histogram(host, v, -1, 2, 30,
                      "Height within this batter's own zone (0 = bottom, 1 = top)",
                      { noZero: true });
  // mark the two edges
  [[0, "bottom edge"], [1, "top edge"]].forEach(([val, label]) => {
    const x = ((val - (-1)) / 3) * f.iw;
    mk("line", { x1: x, y1: 0, x2: x, y2: f.ih, stroke: tok("--series-2"),
                 "stroke-width": 1.5 }, f.g);
    txt(f.g, x + 5, 22, label, "axis-label", "start")
      .setAttribute("fill", tok("--series-2"));
  });
}

function legend(host, items) {
  host.innerHTML = "";
  items.forEach(it => {
    const s = document.createElement("span");
    const i = document.createElement("i");
    i.className = it.sq ? "sq" : "";
    i.style.background = it.color;
    const t = document.createTextNode(it.label);
    s.append(i, t); host.appendChild(s);
  });
}
function empty() {
  return '<p class="note" style="padding:26px 0;color:var(--ink-3)">' +
         "Nothing matches these filters. Try widening the selection.</p>";
}

/* ----------------------------------------------------------------- tables */
let sortState = { key: "dt", dir: -1 };
function drawChallengeTable(ch) {
  const tb = $("#t-challenges tbody");
  const rows = ch.slice().sort((a, b) => {
    const k = sortState.key;
    let x = a[k], y = b[k];
    if (x === null || x === undefined) x = -Infinity;
    if (y === null || y === undefined) y = -Infinity;
    if (typeof x === "string") return sortState.dir * x.localeCompare(y);
    return sortState.dir * (x - y);
  });
  const MAX = 300;
  tb.innerHTML = "";
  rows.slice(0, MAX).forEach(c => {
    const tr = document.createElement("tr");
    const cells = [
      [c.dt, ""], [c.half + c.inn, ""], [c.b + "-" + c.s, ""],
      [c.who || "—", "name"], [c.role, "role"], [c.team || "—", ""],
      [dec(c.d, 1), "num"], [c.rem === null ? "—" : String(c.rem), "num"],
      [c.ov === null ? "—" : (c.ov ? "Overturned" : "Stood"), "res"],
      [c.ump || "—", ""]
    ];
    cells.forEach(([v, cls]) => {
      const td = document.createElement("td");
      if (cls === "role") {
        const key = document.createElement("span");
        key.className = "rolekey";
        key.style.background = roleColor(c.role);
        td.append(key, document.createTextNode(v));
      } else if (cls === "res") {
        const p = document.createElement("span");
        p.className = "pill " + (c.ov ? "ov" : "st");
        p.textContent = v;
        td.appendChild(c.ov === null ? document.createTextNode("—") : p);
      } else {
        td.className = cls;
        td.textContent = v;
      }
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  $("#tableNote").textContent = rows.length > MAX
    ? "Showing the first " + MAX + " of " + nf(rows.length) +
      " — narrow the filters to see the rest"
    : nf(rows.length) + " challenges";
}

const PLAYER_COLS = {
  default: [["name", "Player", ""], ["n", "Challenges", "num"],
            ["won", "Won", "num"], ["rate", "Win %", "num"],
            ["dist", "Avg dist (in)", "num"]],
  umpire: [["name", "Umpire", ""], ["n", "Challenges faced", "num"],
           ["won", "Overturned", "num"], ["rate", "Overturn %", "num"],
           ["acc", "Zone accuracy", "num"]]
};
const playerSort = {};
function drawPlayerTable(table, ch) {
  const role = table.dataset.role;
  const cols = PLAYER_COLS[role] || PLAYER_COLS.default;
  // Recompute from the filtered challenges so the tables obey the filter bar.
  const key = role === "umpire" ? "ump" : "who";
  const sub = role === "umpire" ? ch : ch.filter(c => c.role === role);
  const byName = {};
  sub.forEach(c => {
    const nm = c[key] || "—";
    if (!byName[nm]) byName[nm] = { name: nm, n: 0, won: 0, dsum: 0, dn: 0 };
    const o = byName[nm];
    o.n++;
    if (c.ov) o.won++;
    if (c.d !== null) { o.dsum += Math.abs(c.d); o.dn++; }
  });
  const base = D.players[role] || [];
  const accByName = {};
  base.forEach(p => { accByName[p.name] = p.acc; });
  let rows = Object.values(byName).map(o => ({
    name: o.name, n: o.n, won: o.won,
    rate: o.n ? o.won / o.n : null,
    dist: o.dn ? o.dsum / o.dn : null,
    acc: accByName[o.name] === undefined ? null : accByName[o.name]
  }));
  const st = playerSort[role] || (playerSort[role] = { key: "n", dir: -1 });
  rows.sort((a, b) => {
    let x = a[st.key], y = b[st.key];
    if (x === null || x === undefined) x = -Infinity;
    if (y === null || y === undefined) y = -Infinity;
    if (typeof x === "string") return st.dir * x.localeCompare(y);
    return st.dir * (x - y);
  });
  rows = rows.slice(0, 40);

  table.innerHTML = "";
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  cols.forEach(([k, label, cls]) => {
    const th = document.createElement("th");
    th.className = "sortable " + cls;
    th.textContent = label;
    if (st.key === k) {
      th.setAttribute("aria-sort", st.dir < 0 ? "descending" : "ascending");
      const a = document.createElement("span");
      a.className = "arrow"; a.textContent = st.dir < 0 ? "▼" : "▲";
      th.appendChild(a);
    }
    th.addEventListener("click", () => {
      if (st.key === k) st.dir *= -1; else { st.key = k; st.dir = -1; }
      drawPlayerTable(table, ch);
    });
    htr.appendChild(th);
  });
  thead.appendChild(htr); table.appendChild(thead);

  const tb = document.createElement("tbody");
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = cols.length; td.textContent = "No challenges match these filters.";
    td.style.color = "var(--ink-3)";
    tr.appendChild(td); tb.appendChild(tr);
  }
  rows.forEach(r => {
    const tr = document.createElement("tr");
    cols.forEach(([k, , cls]) => {
      const td = document.createElement("td");
      td.className = (k === "name" ? "name " : "") + cls;
      td.textContent = k === "rate" || k === "acc" ? pct(r[k])
                     : k === "dist" ? dec(r[k], 1)
                     : typeof r[k] === "number" ? nf(r[k]) : (r[k] || "—");
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  table.appendChild(tb);
}

/* ------------------------------------------------------------------ render */
function render() {
  const ch = filterChallenges();
  const cube = filterCube();
  $("#matchN").textContent = nf(ch.length);

  statStrip($("#statStrip"), ch, cube);
  statStrip($("#statStrip2"), ch, cube);
  drawFunnel();
  drawDistanceCurve();
  drawInventory();
  drawRole(ch);
  drawCountHeat(cube);
  drawInning(ch);
  drawMonth(ch);
  drawDistHist(ch);
  drawZone(ch);
  drawZoneCmp(ch);
  drawChallengeTable(ch);
  $$(".t-players").forEach(t => drawPlayerTable(t, ch));

  $("#src-count").textContent = state.role === "catcher" || state.role === "pitcher"
    ? "Opportunity denominators group catcher and pitcher together — only the "
      + "challenge counts are role-specific"
    : "Opportunity denominators group catcher and pitcher together";
}

/* ======================================================================= UI */
function initFilters() {
  const mo = $("#f-month");
  D.months.forEach(m => {
    const o = document.createElement("option"); o.value = m; o.textContent = m;
    mo.appendChild(o);
  });
  const tm = $("#f-team");
  (D.teams || []).forEach(t => {
    const o = document.createElement("option"); o.value = t; o.textContent = t;
    tm.appendChild(o);
  });
  $$("#f-role .chip").forEach(btn => {
    btn.addEventListener("click", () => {
      state.role = btn.dataset.role;
      $$("#f-role .chip").forEach(b =>
        b.setAttribute("aria-pressed", String(b === btn)));
      render();
    });
  });
  [["#f-month", "month"], ["#f-inning", "inning"], ["#f-rem", "rem"],
   ["#f-team", "team"], ["#f-outcome", "outcome"]].forEach(([sel, key]) => {
    $(sel).addEventListener("change", (e) => { state[key] = e.target.value; render(); });
  });
  $("#resetBtn").addEventListener("click", () => {
    Object.keys(state).forEach(k => state[k] = "all");
    $$("#f-role .chip").forEach(b =>
      b.setAttribute("aria-pressed", String(b.dataset.role === "all")));
    ["#f-month", "#f-inning", "#f-rem", "#f-team", "#f-outcome"]
      .forEach(s => { $(s).value = "all"; });
    render();
  });
  $$("#t-challenges th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.k;
      if (sortState.key === k) sortState.dir *= -1;
      else { sortState.key = k; sortState.dir = -1; }
      $$("#t-challenges th").forEach(h => h.removeAttribute("aria-sort"));
      th.setAttribute("aria-sort", sortState.dir < 0 ? "descending" : "ascending");
      drawChallengeTable(filterChallenges());
    });
  });
}

const FILTERED_TABS = ["overview", "challenges", "players", "zone"];
function selectTab(name) {
  $$(".tab").forEach(t => {
    const on = t.id === "tab-" + name;
    t.setAttribute("aria-selected", String(on));
  });
  $$('[role="tabpanel"]').forEach(p => { p.hidden = p.id !== "p-" + name; });
  $("#filterBar").hidden = FILTERED_TABS.indexOf(name) < 0;
  try { history.replaceState(null, "", "#" + name); } catch (e) { /* file:// */ }
  window.scrollTo({ top: 0, behavior: "instant" });
}
function initTabs() {
  $$(".tab").forEach(t => {
    t.addEventListener("click", () => selectTab(t.id.replace("tab-", "")));
  });
  const hash = (location.hash || "").replace("#", "");
  selectTab($("#tab-" + hash) ? hash : "overview");
  addEventListener("hashchange", () => {        // links like index.html#model
    const h = (location.hash || "").replace("#", "");
    if ($("#tab-" + h)) selectTab(h);
  });
}

function initTheme() {
  const btn = $("#themeBtn");
  let saved = null;
  try { saved = localStorage.getItem("abs-theme"); } catch (e) { /* blocked */ }
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  const label = () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark"
      || (!document.documentElement.getAttribute("data-theme")
          && matchMedia("(prefers-color-scheme: dark)").matches);
    btn.textContent = dark ? "Light" : "Dark";
  };
  label();
  btn.addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark"
      || (!document.documentElement.getAttribute("data-theme")
          && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("abs-theme", next); } catch (e) { /* blocked */ }
    label();
    render();                    // charts read colours from CSS, so redraw
  });
}

function initMeta() {
  const m = D.meta;
  $("#m-season").textContent = m.season;
  $("#m-games").textContent = nf(m.n_games);
  $("#m-pitches").textContent = compact(m.n_pitches);
  $("#m-built").textContent = m.built;
  if (m.is_demo) $("#demoBanner").hidden = false;
  $("#methods-build").textContent =
    "Data built " + m.built + " from " + nf(m.n_pitches) + " pitches across " +
    nf(m.n_games) + " games (" + m.date_min + " to " + m.date_max + "), " +
    m.n_columns + " columns.";
}

/* Shared with model.js, which draws the Overturn Model and Progress tabs with
   these same helpers. */
window.ABS_UI = { $, $$, mk, frame, txt, tok, showTip, hideTip, tipRow, legend,
                  pct, dec, nf, compact };

initMeta();
initFilters();
initTabs();
initTheme();
render();
addEventListener("resize", () => { hideTip(); });

})();
