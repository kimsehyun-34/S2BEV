// Build scenario rows with the browser engine's code path, for comparison with real feature rows.
// node scripts/lab_scenario_dump.cjs <in.json> <out.json>   in = {scenario, fixtures:[{team,opponent,home,formation,previousFormation}]}
const fs = require('node:fs');
const path = require('node:path');
const Engine = require(path.join(__dirname, '..', 'epl_tactics', 'lab', 'engine.js'));
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const rows = input.fixtures.map((f) => Engine.scenario(input.scenario, f));
fs.writeFileSync(process.argv[3], JSON.stringify(rows));
