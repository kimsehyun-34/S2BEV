/* Lab chart + DOM helpers. Plain SVG/Canvas, no dependencies, theme-aware (reads CSS custom properties at draw time).
 * Load order: this file, then engine.js, then the shell script that creates `Lab`. Exposes `LabCharts(Lab)` which
 * attaches helpers to the Lab object.
 */
function LabCharts(Lab) {
  'use strict';
  const SVG = 'http://www.w3.org/2000/svg';

  // ---------------------------------------------------------------- formatting
  const nf = (d) => new Intl.NumberFormat('ko-KR', { minimumFractionDigits: d, maximumFractionDigits: d });
  Lab.fmt = {
    int: (n) => (n == null || Number.isNaN(n) ? '–' : nf(0).format(Math.round(n))),
    num: (x, d = 2) => (x == null || Number.isNaN(x) ? '–' : nf(d).format(x)),
    pct: (x, d = 1) => (x == null || Number.isNaN(x) ? '–' : (x * 100).toFixed(d) + '%'),
    signed: (x, d = 3) => (x == null || Number.isNaN(x) ? '–' : (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d)),
    compact: (x) => {
      if (x == null || Number.isNaN(x)) return '–';
      const a = Math.abs(x);
      if (a >= 1e6) return (x / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
      if (a >= 1e4) return (x / 1e3).toFixed(0) + 'K';
      if (a >= 1e3) return (x / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
      if (a >= 100) return x.toFixed(0);
      if (a >= 10) return x.toFixed(1).replace(/\.0$/, '');
      return x.toFixed(2).replace(/0$/, '').replace(/\.0$/, '');
    },
  };

  // ---------------------------------------------------------------- DOM
  Lab.el = function (tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'html') node.innerHTML = v;       // only ever pass trusted, locally built strings
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return node;
  };
  Lab.svgEl = function (tag, attrs, ...kids) {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) node.setAttribute(k, v);
    for (const kid of kids.flat()) if (kid != null) node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return node;
  };
  Lab.css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  Lab.clear = (node) => { node.replaceChildren(); return node; };

  // ---------------------------------------------------------------- scales / ticks
  Lab.niceStep = function (span, count) {
    const raw = span / Math.max(1, count), mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  };
  Lab.ticks = function (min, max, count = 5) {
    if (!(max > min)) return [min];
    const step = Lab.niceStep(max - min, count), out = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  Lab.scaleLinear = function (domain, range) {
    const [d0, d1] = domain, [r0, r1] = range, k = (r1 - r0) / (d1 - d0 || 1);
    const f = (v) => r0 + (v - d0) * k;
    f.invert = (p) => d0 + (p - r0) / k;
    f.domain = domain; f.range = range;
    return f;
  };
  Lab.scaleBand = function (domain, range, pad = 0.25) {
    const n = domain.length, [r0, r1] = range, step = (r1 - r0) / Math.max(1, n), bw = step * (1 - pad);
    const f = (v) => r0 + domain.indexOf(v) * step + (step - bw) / 2;
    f.bandwidth = bw; f.step = step;
    return f;
  };
  Lab.quantile = function (sorted, q) {
    if (!sorted.length) return NaN;
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };

  // ---------------------------------------------------------------- tooltip (one shared node)
  let tipNode = null;
  function tip() {
    if (!tipNode) { tipNode = Lab.el('div', { class: 'tooltip', role: 'tooltip' }); tipNode.hidden = true; document.body.append(tipNode); }
    return tipNode;
  }
  Lab.tip = {
    show(html, x, y) {
      const n = tip(); n.innerHTML = html; n.hidden = false;
      const r = n.getBoundingClientRect(), pad = 12;
      let left = x + pad, top = y + pad;
      if (left + r.width > innerWidth - 8) left = x - r.width - pad;
      if (top + r.height > innerHeight - 8) top = y - r.height - pad;
      n.style.left = Math.max(8, left) + 'px'; n.style.top = Math.max(8, top) + 'px';
    },
    hide() { if (tipNode) tipNode.hidden = true; },
    attach(node, htmlFn) {
      node.addEventListener('pointermove', (e) => Lab.tip.show(htmlFn(e), e.clientX, e.clientY));
      node.addEventListener('pointerleave', () => Lab.tip.hide());
      node.addEventListener('focus', () => { const r = node.getBoundingClientRect(); Lab.tip.show(htmlFn(), r.left + r.width / 2, r.top); });
      node.addEventListener('blur', () => Lab.tip.hide());
    },
  };
  Lab.esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---------------------------------------------------------------- responsive chart scaffold
  // draw(ctx) is called now, on resize and on theme change. ctx = {svg, g, w, h, m, width, height}
  const live = new Set();
  Lab.redrawAll = () => { for (const fn of [...live]) fn(); };
  Lab.chart = function (container, { height = 280, margin = {}, aspect, draw, label = '' }) {
    const m = { t: 14, r: 16, b: 34, l: 48, ...margin };
    container.classList.add('chart');
    const svg = Lab.svgEl('svg', { role: 'img', 'aria-label': label, class: 'chart-svg' });
    container.replaceChildren(svg);
    function render() {
      if (!container.isConnected) { live.delete(render); return; }
      const width = Math.max(240, Math.round(container.clientWidth || container.getBoundingClientRect().width));
      if (!container.clientWidth) return;                    // hidden step: draw when shown
      const h = aspect ? Math.round(width / aspect) : height;
      svg.setAttribute('viewBox', `0 0 ${width} ${h}`); svg.setAttribute('width', width); svg.setAttribute('height', h);
      svg.replaceChildren();
      const g = Lab.svgEl('g', { transform: `translate(${m.l},${m.t})` });
      svg.append(g);
      draw({ svg, g, w: width - m.l - m.r, h: h - m.t - m.b, m, width, height: h });
    }
    live.add(render);
    if (typeof ResizeObserver === 'function') {
      let last = 0;
      new ResizeObserver(() => { const w = container.clientWidth; if (Math.abs(w - last) > 1) { last = w; render(); } }).observe(container);
    }
    render();
    return { redraw: render, svg };
  };

  Lab.axisBottom = function (g, scale, h, { ticks, format = Lab.fmt.compact, label, grid = false, gridH } = {}) {
    const vals = ticks || (scale.ticks ? scale.ticks() : Lab.ticks(scale.domain[0], scale.domain[1], 6));
    g.append(Lab.svgEl('line', { x1: scale.range[0], x2: scale.range[1], y1: h, y2: h, class: 'axis-line' }));
    for (const t of vals) {
      const x = scale(t);
      if (grid) g.append(Lab.svgEl('line', { x1: x, x2: x, y1: 0, y2: h, class: 'grid-line' }));
      g.append(Lab.svgEl('text', { x, y: h + 17, 'text-anchor': 'middle', class: 'tick' }, format(t)));
    }
    if (label) g.append(Lab.svgEl('text', { x: (scale.range[0] + scale.range[1]) / 2, y: h + 32, 'text-anchor': 'middle', class: 'axis-label' }, label));
  };
  Lab.axisLeft = function (g, scale, w, { ticks, format = Lab.fmt.compact, label, grid = true } = {}) {
    const vals = ticks || Lab.ticks(scale.domain[0], scale.domain[1], 5);
    for (const t of vals) {
      const y = scale(t);
      if (grid) g.append(Lab.svgEl('line', { x1: 0, x2: w, y1: y, y2: y, class: 'grid-line' }));
      g.append(Lab.svgEl('text', { x: -8, y: y + 4, 'text-anchor': 'end', class: 'tick' }, format(t)));
    }
    if (label) {
      const widest = Math.max(...vals.map((t) => String(format(t)).length), 1);        // keep the label clear of wide tick labels
      g.append(Lab.svgEl('text', { transform: `translate(${-Math.max(38, 16 + widest * 7)},${(scale.range[0] + scale.range[1]) / 2}) rotate(-90)`, 'text-anchor': 'middle', class: 'axis-label' }, label));
    }
  };

  // Legend: items = [{label, color, dash}]. Always rendered as real HTML so it wraps on phones.
  Lab.legend = function (items) {
    return Lab.el('div', { class: 'legend' }, items.map((it) => Lab.el('span', { class: 'legend-item' },
      Lab.el('i', { class: 'swatch', style: `background:${it.color}` }), it.label)));
  };

  // ---------------------------------------------------------------- chart kinds
  /* Horizontal/vertical bars. data = [{label, value, color?, note?, tip?}]  opts: {format, max, min, sort, height?, labelWidth?} */
  Lab.barsH = function (container, data, { format = Lab.fmt.compact, max, min = 0, labelWidth = 150, rowHeight = 26, color = 'var(--s1)', valueLabels = true, ariaLabel = '' } = {}) {
    const n = data.length, h = n * rowHeight + 12;
    return Lab.chart(container, {
      height: h, margin: { t: 4, r: valueLabels ? 64 : 16, b: 8, l: labelWidth }, label: ariaLabel,
      draw({ g, w, h: ph }) {
        const hi = max ?? Math.max(...data.map((d) => d.value), 1e-9), lo = Math.min(min, ...data.map((d) => d.value));
        const x = Lab.scaleLinear([lo, hi], [0, w]), band = ph / n, bar = Math.min(18, band * 0.62), zero = x(Math.max(lo, 0));
        g.append(Lab.svgEl('line', { x1: zero, x2: zero, y1: 0, y2: ph, class: 'axis-line' }));
        data.forEach((d, i) => {
          const cy = i * band + band / 2, x1 = x(d.value), left = Math.min(zero, x1), width = Math.max(1, Math.abs(x1 - zero));
          const row = Lab.svgEl('g', { tabindex: d.tip ? 0 : null });
          row.append(Lab.svgEl('text', { x: -10, y: cy + 4, 'text-anchor': 'end', class: 'tick tick-strong' }, d.label));
          row.append(Lab.svgEl('rect', { x: left, y: cy - bar / 2, width, height: bar, rx: 4, fill: d.color || color, class: 'mark' }));
          if (valueLabels) row.append(Lab.svgEl('text', { x: (d.value >= 0 ? left + width : left) + (d.value >= 0 ? 8 : -8), y: cy + 4, 'text-anchor': d.value >= 0 ? 'start' : 'end', class: 'value-label' }, format(d.value)));
          row.append(Lab.svgEl('rect', { x: -labelWidth, y: i * band, width: w + labelWidth + 60, height: band, fill: 'transparent' }));
          if (d.tip) Lab.tip.attach(row, () => d.tip);
          g.append(row);
        });
      },
    });
  };

  /* Overlaid / side-by-side histogram. groups = [{name, color, values}]  opts: {bins, domain, xLabel, normalize, clip:[lo,hi]} */
  Lab.hist = function (container, groups, { bins = 24, domain, xLabel = '', normalize = true, height = 240, format, ariaLabel = '' } = {}) {
    const all = groups.flatMap((g) => g.values).filter((v) => Number.isFinite(v));
    if (!all.length) { container.textContent = '표시할 값이 없습니다.'; return null; }
    const lo = domain ? domain[0] : Math.min(...all), hi = domain ? domain[1] : Math.max(...all), span = hi - lo || 1;
    const step = span / bins;
    const tables = groups.map((g) => {
      const c = new Array(bins).fill(0); let n = 0;
      for (const v of g.values) { if (!Number.isFinite(v) || v < lo || v > hi) continue; c[Math.min(bins - 1, Math.floor((v - lo) / step))]++; n++; }
      return { ...g, counts: c, n, shares: c.map((x) => (normalize && n ? x / n : x)) };
    });
    return Lab.chart(container, {
      height, margin: { l: 44, b: 38 }, label: ariaLabel,
      draw({ g, w, h }) {
        const top = Math.max(...tables.flatMap((t) => t.shares), 1e-9);
        const x = Lab.scaleLinear([lo, hi], [0, w]), y = Lab.scaleLinear([0, top * 1.08], [h, 0]);
        Lab.axisLeft(g, y, w, { format: normalize ? (v) => Math.round(v * 100) + '%' : Lab.fmt.compact });
        Lab.axisBottom(g, x, h, { format: format || Lab.fmt.compact, label: xLabel });
        const bw = (w / bins), inner = Math.max(1, (bw - 2) / (groups.length > 1 ? 1 : 1));
        tables.forEach((t, gi) => {
          t.shares.forEach((s, i) => {
            if (!s) return;
            const bx = x(lo + i * step), bh = h - y(s);
            const rect = Lab.svgEl('rect', { x: bx + 1, y: y(s), width: Math.max(1, bw - 2), height: bh, fill: t.color, opacity: groups.length > 1 ? 0.55 : 0.9, class: 'mark' });
            g.append(rect);
          });
        });
        const hit = Lab.svgEl('rect', { x: 0, y: 0, width: w, height: h, fill: 'transparent' });
        Lab.tip.attach(hit, (e) => {
          const r = hit.getBoundingClientRect(), i = Math.max(0, Math.min(bins - 1, Math.floor(((e ? e.clientX : r.left) - r.left) / r.width * bins)));
          const a = lo + i * step, b = a + step;
          return `<b>${Lab.fmt.compact(a)} ~ ${Lab.fmt.compact(b)}</b>` + tables.map((t) => `<div><i class="swatch" style="background:${t.color}"></i>${Lab.esc(t.name)} ${t.counts[i]}개 (${Lab.fmt.pct(t.n ? t.counts[i] / t.n : 0, 0)})</div>`).join('');
        });
        g.append(hit);
      },
    });
  };

  /* Canvas scatter. points = {x:[], y:[], c:[index into colors], label?:[]}  opts: {colors, xDomain, yDomain, xLabel, yLabel, xFormat, yFormat}
     colors: array of CSS colours or custom-property names ('--s1'), or a function returning such an array; resolved on every draw so the theme can change. */
  Lab.scatter = function (container, pts, { colors, names = [], xLabel = '', yLabel = '', xFormat, yFormat, height = 360, xDomain, yDomain, alpha = 0.45, radius = 3, tipFn, ariaLabel = '' } = {}) {
    container.classList.add('chart', 'chart-canvas');
    const canvas = Lab.el('canvas', { 'aria-label': ariaLabel, role: 'img' });
    const overlay = Lab.svgEl('svg', { class: 'chart-overlay' });
    container.replaceChildren(canvas, overlay);
    const m = { t: 12, r: 16, b: 40, l: 54 };
    let geo = null;
    function render() {
      if (!container.isConnected) { live.delete(render); return; }
      const width = container.clientWidth; if (!width) return;
      const yTicks = (() => { const ys0 = pts.y.filter(Number.isFinite); return ys0.length ? Lab.ticks(yDomain ? yDomain[0] : Math.min(...ys0), yDomain ? yDomain[1] : Math.max(...ys0), 5) : []; })();
      m.l = Math.max(54, 26 + 7 * Math.max(0, ...yTicks.map((t) => String((yFormat || Lab.fmt.compact)(t)).length)));   // room for wide tick labels + rotated title
      const dpr = window.devicePixelRatio || 1, h = height, pw = width - m.l - m.r, ph = h - m.t - m.b;
      canvas.width = width * dpr; canvas.height = h * dpr; canvas.style.width = width + 'px'; canvas.style.height = h + 'px';
      overlay.setAttribute('width', width); overlay.setAttribute('height', h); overlay.setAttribute('viewBox', `0 0 ${width} ${h}`); overlay.replaceChildren();
      const finite = (a) => a.filter(Number.isFinite);
      const xs = finite(pts.x), ys = finite(pts.y);
      const dx = xDomain || [Math.min(...xs), Math.max(...xs)], dy = yDomain || [Math.min(...ys), Math.max(...ys)];
      const padX = (dx[1] - dx[0]) * 0.03 || 1, padY = (dy[1] - dy[0]) * 0.05 || 1;
      const x = Lab.scaleLinear(xDomain || [dx[0] - padX, dx[1] + padX], [0, pw]), y = Lab.scaleLinear(yDomain || [dy[0] - padY, dy[1] + padY], [ph, 0]);
      const g = Lab.svgEl('g', { transform: `translate(${m.l},${m.t})` }); overlay.append(g);
      Lab.axisLeft(g, y, pw, { label: yLabel, format: yFormat }); Lab.axisBottom(g, x, ph, { label: xLabel, grid: true, format: xFormat });
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, h);
      ctx.save(); ctx.translate(m.l, m.t); ctx.beginPath(); ctx.rect(0, 0, pw, ph); ctx.clip();
      const px = new Float32Array(pts.x.length), py = new Float32Array(pts.x.length);
      const palette = (typeof colors === 'function' ? colors() : colors).map((c) => (c.startsWith('--') ? Lab.css(c) : c));
      for (let i = 0; i < pts.x.length; i++) {
        px[i] = x(pts.x[i]); py[i] = y(pts.y[i]);
        if (!Number.isFinite(px[i]) || !Number.isFinite(py[i])) continue;
        ctx.globalAlpha = alpha; ctx.fillStyle = palette[pts.c[i]] || palette[0];
        ctx.beginPath(); ctx.arc(px[i], py[i], radius, 0, 6.2832); ctx.fill();
      }
      ctx.restore();
      geo = { px, py, pw, ph };
    }
    const hover = Lab.svgEl('circle', { r: radius + 3, class: 'hover-ring', fill: 'none' });
    container.addEventListener('pointermove', (e) => {
      if (!geo) return;
      const r = container.getBoundingClientRect(), mx = e.clientX - r.left - m.l, my = e.clientY - r.top - m.t;
      let best = -1, bd = 14 * 14;
      for (let i = 0; i < geo.px.length; i++) { const dx = geo.px[i] - mx, dy = geo.py[i] - my, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
      overlay.querySelectorAll('.hover-ring').forEach((n) => n.remove());
      if (best < 0) { Lab.tip.hide(); return; }
      const ring = hover.cloneNode(); ring.setAttribute('cx', geo.px[best] + m.l); ring.setAttribute('cy', geo.py[best] + m.t); overlay.append(ring);
      Lab.tip.show(tipFn ? tipFn(best) : `x ${Lab.fmt.num(pts.x[best])} · y ${Lab.fmt.num(pts.y[best])}`, e.clientX, e.clientY);
    });
    container.addEventListener('pointerleave', () => { Lab.tip.hide(); overlay.querySelectorAll('.hover-ring').forEach((n) => n.remove()); });
    live.add(render);
    if (typeof ResizeObserver === 'function') new ResizeObserver(() => render()).observe(container);
    render();
    return { redraw: render };
  };

  /* Multi-series line chart. series = [{name, color, points:[[x,y]...], dash?}]  opts: {xLabel,yLabel,yDomain,xDomain,xFormat,yFormat} */
  Lab.lines = function (container, series, { xLabel = '', yLabel = '', xDomain, yDomain, xFormat, yFormat, height = 260, markers = true, ariaLabel = '' } = {}) {
    return Lab.chart(container, {
      height, margin: { l: 56, b: 40 }, label: ariaLabel,
      draw({ g, w, h }) {
        const xs = series.flatMap((s) => s.points.map((p) => p[0])), ys = series.flatMap((s) => s.points.map((p) => p[1])).filter(Number.isFinite);
        const dx = xDomain || [Math.min(...xs), Math.max(...xs)], dy = yDomain || [Math.min(...ys), Math.max(...ys)];
        const pad = (dy[1] - dy[0]) * 0.08 || 0.1;
        const x = Lab.scaleLinear(dx, [0, w]), y = Lab.scaleLinear(yDomain || [dy[0] - pad, dy[1] + pad], [h, 0]);
        Lab.axisLeft(g, y, w, { label: yLabel, format: yFormat }); Lab.axisBottom(g, x, h, { label: xLabel, format: xFormat });
        for (const s of series) {
          const d = s.points.filter((p) => Number.isFinite(p[1])).map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join('');
          g.append(Lab.svgEl('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dash || null }));
          if (markers) for (const p of s.points) if (Number.isFinite(p[1])) g.append(Lab.svgEl('circle', { cx: x(p[0]), cy: y(p[1]), r: 4, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2 }));
        }
        const hit = Lab.svgEl('rect', { x: 0, y: 0, width: w, height: h, fill: 'transparent' }); g.append(hit);
        Lab.tip.attach(hit, (e) => {
          const r = hit.getBoundingClientRect(), v = x.invert((e ? e.clientX : r.left) - r.left);
          const near = series[0].points.reduce((a, b) => (Math.abs(b[0] - v) < Math.abs(a[0] - v) ? b : a));
          return `<b>${(xFormat || Lab.fmt.compact)(near[0])}</b>` + series.map((s) => { const p = s.points.find((q) => q[0] === near[0]); return p ? `<div><i class="swatch" style="background:${s.color}"></i>${Lab.esc(s.name)} ${(yFormat || Lab.fmt.num)(p[1])}</div>` : ''; }).join('');
        });
      },
    });
  };

  // ---------------------------------------------------------------- table
  /* columns = [{key,label,align?,fmt?(v,row),sort?:true,title?}]  rows = [{...}] */
  Lab.table = function (container, columns, rows, { maxHeight, sortKey, sortDir = 1, caption = '', onRow } = {}) {
    let key = sortKey, dir = sortDir;
    const wrap = Lab.el('div', { class: 'scroll', style: maxHeight ? `max-height:${maxHeight}px` : null });
    function draw() {
      const data = key ? [...rows].sort((a, b) => ((a[key] > b[key]) - (a[key] < b[key])) * dir || 0) : rows;
      const head = Lab.el('tr', {}, columns.map((c) => {
        const th = Lab.el('th', { scope: 'col', class: c.align === 'right' ? 'num' : '', title: c.title || null, 'aria-sort': key === c.key ? (dir > 0 ? 'ascending' : 'descending') : null });
        if (c.sort === false) th.textContent = c.label;
        else { const b = Lab.el('button', { class: 'th-btn', type: 'button', onclick: () => { dir = key === c.key ? -dir : 1; key = c.key; draw(); } }, c.label, key === c.key ? (dir > 0 ? ' ▲' : ' ▼') : ''); th.append(b); }
        return th;
      }));
      const body = data.map((r) => {
        const tr = Lab.el('tr', onRow ? { onclick: () => onRow(r), tabindex: 0, class: 'clickable' } : {}, columns.map((c) => {
          const v = c.fmt ? c.fmt(r[c.key], r) : r[c.key];
          return Lab.el('td', { class: c.align === 'right' ? 'num' : '' }, v instanceof Node ? v : (v == null || v === '' ? '–' : v));
        }));
        return tr;
      });
      wrap.replaceChildren(Lab.el('table', { class: 'data' }, caption ? Lab.el('caption', { class: 'sr-only', text: caption }) : null, Lab.el('thead', {}, head), Lab.el('tbody', {}, body)));
    }
    draw();
    container.replaceChildren(wrap);
    return { redraw: draw };
  };

  // ---------------------------------------------------------------- small controls
  /* Segmented control: options=[{value,label}] */
  Lab.segmented = function (container, options, value, onChange, label = '') {
    const wrap = Lab.el('div', { class: 'seg', role: 'radiogroup', 'aria-label': label });
    const buttons = options.map((o) => Lab.el('button', { type: 'button', role: 'radio', 'aria-checked': String(o.value === value), 'data-value': o.value,
      onclick: () => { select(o.value); onChange(o.value); } }, o.label));
    function select(v) { buttons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === String(v)))); }
    wrap.append(...buttons); container.replaceChildren(wrap);
    return { select };
  };
  Lab.select = function (options, value, onChange, { id, label } = {}) {
    const s = Lab.el('select', { id, 'aria-label': label || null }, options.map((o) => Lab.el('option', { value: o.value, selected: String(o.value) === String(value) ? '' : null }, o.label)));
    s.addEventListener('change', () => onChange(s.value));
    return s;
  };
}
if (typeof module !== 'undefined' && module.exports) module.exports = LabCharts;
