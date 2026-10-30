/* LabEngine: runs the exported Random Forest in the browser (or Node) with no dependencies.
 *
 * A "row" is a plain object {inputName: value}. Missing numeric values are null/undefined/NaN.
 * Everything here mirrors sklearn: OneHot(handle_unknown="ignore") + median impute (+missing indicator) +
 * StandardScaler, features cast to float32, trees compare `x <= threshold`, forest averages leaf probabilities.
 */
(function (root) {
  'use strict';

  function decode(b64, Type) {
    const text = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
    return new Type(bytes.buffer);
  }

  const isMissing = (v) => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));
  const MISSING = '__missing__';

  function create(payload) {
    const pre = payload.model.pre, spec = payload.model.forest;
    const offsets = decode(spec.offsets, Int32Array), feature = decode(spec.feature, Int16Array);
    const threshold = decode(spec.threshold, Float64Array), left = decode(spec.left, Int16Array);
    const right = decode(spec.right, Int16Array), value = decode(spec.value, Float32Array);
    const nTrees = spec.n_trees, nFeatures = spec.n_features;

    // ---- column layout of the transformed matrix: [one-hot blocks][numeric][missing indicators]
    const catIndex = pre.categorical.map(() => new Map());
    let cursor = 0;
    const catStart = pre.categorical.map((c, k) => {
      const start = cursor;
      c.categories.forEach((name, i) => catIndex[k].set(name, i));
      cursor += c.categories.length;
      return start;
    });
    const numStart = cursor, nNumeric = pre.numeric.length, indStart = numStart + nNumeric;
    if (indStart + pre.indicator_of.length !== nFeatures) throw new Error('feature layout mismatch');
    // owner[j] = raw input a transformed column belongs to (for attributing contributions)
    const owner = new Array(nFeatures);
    pre.categorical.forEach((c, k) => { for (let i = 0; i < c.categories.length; i++) owner[catStart[k] + i] = c.name; });
    pre.numeric.forEach((name, i) => { owner[numStart + i] = name; });
    pre.indicator_of.forEach((idx, i) => { owner[indStart + i] = pre.numeric[idx]; });
    const ownerNames = Array.from(new Set(owner));

    function vectorize(row) {
      const x = new Float32Array(nFeatures);
      pre.categorical.forEach((c, k) => {
        const v = row[c.name];
        const hit = catIndex[k].get(isMissing(v) ? MISSING : String(v));
        if (hit !== undefined) x[catStart[k] + hit] = 1;
      });
      for (let i = 0; i < nNumeric; i++) {
        const v = row[pre.numeric[i]];
        const absent = isMissing(v);
        x[numStart + i] = Math.fround(((absent ? pre.median[i] : Number(v)) - pre.mean[i]) / pre.scale[i]);
      }
      pre.indicator_of.forEach((idx, i) => {
        // scaler arrays are indexed over [numeric..., indicators...]
        const flag = isMissing(row[pre.numeric[idx]]) ? 1 : 0;
        x[indStart + i] = Math.fround((flag - pre.mean[nNumeric + i]) / pre.scale[nNumeric + i]);
      });
      return x;
    }

    function predictVector(x) {
      let a = 0, b = 0, c = 0;
      for (let t = 0; t < nTrees; t++) {
        const base = offsets[t];
        let n = 0;
        while (feature[base + n] >= 0) n = x[feature[base + n]] <= threshold[base + n] ? left[base + n] : right[base + n];
        const v = (base + n) * 3;
        a += value[v]; b += value[v + 1]; c += value[v + 2];
      }
      return [a / nTrees, b / nTrees, c / nTrees];
    }

    // Per-input attribution of one prediction ("tree interpreter"): bias + sum(contributions) == proba exactly.
    function explainVector(x) {
      const byColumn = new Float64Array(nFeatures * 3);
      const bias = [0, 0, 0];
      for (let t = 0; t < nTrees; t++) {
        const base = offsets[t];
        let n = 0;
        bias[0] += value[base * 3]; bias[1] += value[base * 3 + 1]; bias[2] += value[base * 3 + 2];
        while (feature[base + n] >= 0) {
          const f = feature[base + n];
          const child = x[f] <= threshold[base + n] ? left[base + n] : right[base + n];
          const p = (base + n) * 3, q = (base + child) * 3;
          byColumn[f * 3] += value[q] - value[p];
          byColumn[f * 3 + 1] += value[q + 1] - value[p + 1];
          byColumn[f * 3 + 2] += value[q + 2] - value[p + 2];
          n = child;
        }
      }
      const byInput = new Map();
      for (let j = 0; j < nFeatures; j++) {
        const o = byColumn[j * 3], p = byColumn[j * 3 + 1], q = byColumn[j * 3 + 2];
        if (o === 0 && p === 0 && q === 0) continue;
        const acc = byInput.get(owner[j]) || [0, 0, 0];
        acc[0] += o / nTrees; acc[1] += p / nTrees; acc[2] += q / nTrees;
        byInput.set(owner[j], acc);
      }
      const proba = predictVector(x);
      return { proba, bias: bias.map((v) => v / nTrees), byInput };
    }

    const predict = (row) => predictVector(vectorize(row));

    // Contributions as a sorted list, largest effect on P(win) first (ties broken by size on all classes).
    function explain(row, top) {
      const e = explainVector(vectorize(row));
      const list = Array.from(e.byInput, ([name, c]) => ({ name, loss: c[0], draw: c[1], win: c[2] }));
      list.sort((u, v) => Math.abs(v.win) - Math.abs(u.win) || (Math.abs(v.loss) + Math.abs(v.draw)) - (Math.abs(u.loss) + Math.abs(u.draw)));
      return { proba: e.proba, bias: e.bias, contributions: top ? list.slice(0, top) : list };
    }

    // "What if this input took each of these values?" -> probabilities for every value.
    function sweep(row, name, values) {
      return values.map((v) => ({ value: v, proba: predict({ ...row, [name]: v }) }));
    }

    const elo = payload.elo_baseline;
    function eloBaseline(row) {
      const z = elo.columns.map((c, i) => ((isMissing(row[c]) ? 0 : Number(row[c])) - elo.mean[i]) / elo.scale[i]);
      const logits = elo.coef.map((w, k) => w.reduce((s, wi, i) => s + wi * z[i], elo.intercept[k]));
      const m = Math.max(...logits), e = logits.map((l) => Math.exp(l - m)), s = e[0] + e[1] + e[2];
      return e.map((v) => v / s);
    }

    return { vectorize, predict, predictVector, explain, sweep, eloBaseline, owner, ownerNames, nFeatures, nTrees, MISSING };
  }

  // ------------------------------------------------------------------ scenario construction (hypothetical fixtures)
  const DIFFS = ['elo', 'points_5', 'goals_for_20', 'goals_against_20', 'shots_20', 'sot_20', 'rest_days'];

  function parseFormation(f) {
    if (isMissing(f) || f === MISSING) return null;
    const parts = String(f).split('-').map(Number);
    return parts.every(Number.isFinite) && parts.length ? parts : null;
  }

  // Formation-dependent inputs. `snapshot` is scenario.snapshots[team]; `previous` is the previous formation (or MISSING).
  function applyFormation(row, snapshot, formation, previous) {
    const parts = parseFormation(formation);
    const prev = previous === undefined ? snapshot.last_formation : previous;
    const known = snapshot.formations_last20;
    const next = { ...row };
    next.formation = parts ? formation : MISSING;
    next.team_formation = `${row.team}|${next.formation}`;
    next.previous_formation = isMissing(prev) ? MISSING : prev;
    next.formation_known = parts ? 1 : 0;
    next.defenders = parts ? parts[0] : null;
    next.forwards = parts ? parts[parts.length - 1] : null;
    next.midfielders = parts ? parts.slice(1, -1).reduce((s, v) => s + v, 0) : null;
    next.formation_lines = parts ? parts.length : null;
    next.formation_used_last20 = parts && known.length ? known.filter((f) => f === formation).length / known.length : null;
    next.formation_switch = parts && next.previous_formation !== MISSING ? Number(formation !== next.previous_formation) : null;
    return next;
  }

  function recomputeDiffs(row) {
    const next = { ...row };
    for (const d of DIFFS) {
      const a = row[`team_${d}`], b = row[`opponent_${d}`];
      next[`diff_${d}`] = isMissing(a) || isMissing(b) ? null : a - b;
    }
    return next;
  }

  // Build every model input for "team plays opponent (home/away) with this formation" from the latest team states.
  function scenario(scn, { team, opponent, home, formation, previousFormation }) {
    const mine = scn.snapshots[team], theirs = scn.snapshots[opponent];
    if (!mine || !theirs) throw new Error(`No snapshot for ${!mine ? team : opponent}`);
    const row = { team, opponent, is_home: home ? 1 : 0 };
    for (const [k, v] of Object.entries(mine[home ? 'home' : 'away'])) row[`team_${k}`] = v;
    for (const [k, v] of Object.entries(theirs[home ? 'away' : 'home'])) row[`opponent_${k}`] = v;
    for (const k of scn.lineup_keys) { row[`team_lineup_${k}`] = mine.lineup[k]; row[`opponent_lineup_${k}`] = theirs.lineup[k]; }
    return applyFormation(recomputeDiffs(row), mine, formation, previousFormation);
  }

  // A held-out match as a plain row object.
  function heldoutRow(held, index) {
    const row = {};
    held.columns.forEach((c, i) => { row[c] = held.rows[index][i]; });
    return row;
  }

  const api = { create, scenario, applyFormation, recomputeDiffs, heldoutRow, parseFormation, isMissing, MISSING, DIFFS };
  root.LabEngine = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
