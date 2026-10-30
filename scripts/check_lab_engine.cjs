// Parity check: the JS engine must reproduce the Python model on every held-out row.
// node scripts/check_lab_engine.cjs <payload.json>   (payload = the "s4" object or the whole lab payload)
const fs = require('node:fs');
const path = require('node:path');
const Engine = require(path.join(__dirname, '..', 'epl_tactics', 'lab', 'engine.js'));

const raw = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const payload = raw.s4 || raw;
const engine = Engine.create(payload);
const held = payload.heldout;
let maxProba = 0, maxElo = 0, maxSum = 0, maxLeaf = 0;
for (let i = 0; i < held.rows.length; i++) {
  const row = Engine.heldoutRow(held, i);
  const p = engine.predict(row), e = engine.explain(row);
  for (let k = 0; k < 3; k++) {
    maxProba = Math.max(maxProba, Math.abs(p[k] - held.py_proba[i][k]));
    maxElo = Math.max(maxElo, Math.abs(engine.eloBaseline(row)[k] - held.py_elo[i][k]));
    const total = e.bias[k] + e.contributions.reduce((s, c) => s + [c.loss, c.draw, c.win][k], 0);
    maxSum = Math.max(maxSum, Math.abs(total - p[k]));
  }
  maxLeaf = Math.max(maxLeaf, Math.abs(p[0] + p[1] + p[2] - 1));
}
const result = { rows: held.rows.length, maxProbaDiff: maxProba, maxEloDiff: maxElo, maxContributionSumDiff: maxSum, maxProbaSumError: maxLeaf };
console.log(JSON.stringify(result));
if (maxProba > 2e-6 || maxElo > 2e-6 || maxSum > 1e-9 || maxLeaf > 1e-6) { console.error('PARITY FAILED'); process.exit(1); }
