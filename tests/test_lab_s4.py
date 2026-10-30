"""Invariants of the step-4 lab payload (epl_tactics/lab/export_s4.py) and its page fragment (s4.html)."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, MODEL = ROOT / "data" / "focused", ROOT / "artifacts" / "focused"
FRAGMENT = ROOT / "epl_tactics" / "lab" / "s4.html"
CTX = PAYLOAD = None


def setUpModule():
    global CTX, PAYLOAD
    if not (DATA / "matches.csv").exists() or not (MODEL / "metrics.json").exists():
        raise unittest.SkipTest("focused data/artifacts are not available")
    from epl_tactics.lab.context import load_context
    from epl_tactics.lab.export_s4 import build
    CTX = load_context(DATA, MODEL, fit=True)
    PAYLOAD = build(CTX)


def ordered_heldout() -> pd.DataFrame:
    return CTX.heldout.sort_values(["date", "match_id", "team"]).reset_index(drop=True)


class LabStep4ExportTests(unittest.TestCase):
    def test_train_and_heldout_split(self):
        model = PAYLOAD["model"]
        self.assertEqual(model["train_rows"], int(CTX.train_mask.sum()))
        self.assertEqual(model["train_rows"] + model["heldout_rows"], len(CTX.rows))
        self.assertLess(model["train_last_date"], model["heldout_first_date"])
        self.assertTrue((CTX.train.date < pd.Timestamp(model["heldout_first_date"])).all())
        self.assertTrue((CTX.heldout.season >= CTX.test_start).all())
        self.assertEqual(model["heldout_first_date"], str(CTX.rows[CTX.rows.season.eq(CTX.test_start)].date.min().date()))
        test = CTX.report["test"]["selected_with_formation"]
        self.assertEqual(model["heldout_rows"], test["team_appearances"])
        self.assertEqual(model["heldout_matches"], test["unique_matches"])
        self.assertEqual(model["spec"], CTX.spec)

    def test_heldout_rows_are_the_source_rows(self):
        from epl_tactics.focused_features import CATEGORIES
        held, frame = PAYLOAD["heldout"], ordered_heldout()
        columns = held["columns"]
        self.assertTrue(set(CATEGORIES) | set(CTX.numeric) <= set(columns))
        self.assertEqual(len(held["rows"]), len(frame))
        table = pd.DataFrame(held["rows"], columns=columns)
        for name in ["match_id", "team", "opponent", "formation", "previous_formation", "team_formation", "season"]:
            self.assertEqual(table[name].tolist(), frame[name].tolist(), name)
        self.assertEqual(table.date.tolist(), [str(d.date()) for d in frame.date])
        self.assertTrue((table.outcome.to_numpy() == frame.outcome.to_numpy()).all())
        self.assertTrue((table.goals_for.to_numpy() == frame.goals_for.to_numpy()).all())
        source = frame[CTX.numeric].astype(float).round(6).to_numpy()
        exported = table[CTX.numeric].astype(float).to_numpy()
        self.assertTrue(np.allclose(source, exported, equal_nan=True, atol=1e-9))
        # market columns (shown as "extra information the model lacks") are the source odds
        for name in ["market_loss", "market_draw", "market_win"]:
            self.assertTrue(np.allclose(table[name].astype(float), frame[name].astype(float).round(6), equal_nan=True))

    def test_python_probabilities_are_the_model_output(self):
        frame = ordered_heldout().copy()
        frame[CTX.numeric] = frame[CTX.numeric].round(6)
        expected = CTX.model.predict_proba(frame)
        proba = np.asarray(PAYLOAD["heldout"]["py_proba"])
        self.assertLess(np.abs(proba - expected).max(), 1e-8)
        self.assertLess(np.abs(proba.sum(1) - 1).max(), 1e-8)
        elo = np.asarray(PAYLOAD["heldout"]["py_elo"])
        self.assertLess(np.abs(elo.sum(1) - 1).max(), 1e-8)

    def test_elo_baseline_parameters_reproduce_its_probabilities(self):
        spec, table = PAYLOAD["elo_baseline"], pd.DataFrame(PAYLOAD["heldout"]["rows"], columns=PAYLOAD["heldout"]["columns"])
        z = (table[spec["columns"]].to_numpy(float) - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
        logits = z @ np.asarray(spec["coef"]).T + np.asarray(spec["intercept"])
        proba = np.exp(logits - logits.max(1, keepdims=True))
        proba /= proba.sum(1, keepdims=True)
        self.assertLess(np.abs(proba - np.asarray(PAYLOAD["heldout"]["py_elo"])).max(), 1e-8)

    def test_fixed_model_scores_and_fixture_weights(self):
        from epl_tactics.focused_model import cohort_metrics
        frame, proba = ordered_heldout(), np.asarray(PAYLOAD["heldout"]["py_proba"])
        scores = PAYLOAD["fixed_model_scores"]
        overall = cohort_metrics(frame, proba)
        for key in ["accuracy", "log_loss", "brier", "unique_matches", "team_appearances"]:
            self.assertAlmostEqual(scores["overall"][key], overall[key], places=12)
        weights = frame.fixture_weight.to_numpy()
        manual = np.sum(weights * (proba.argmax(1) == frame.outcome.to_numpy())) / weights.sum()
        self.assertAlmostEqual(scores["overall"]["accuracy"], manual, places=12)
        self.assertEqual(sorted(scores["by_season"]), sorted(frame.season.unique()))
        self.assertEqual(sum(v["team_appearances"] for v in scores["by_season"].values()), len(frame))
        # The page derives fixture weights as 0.5 when the opponent is also one of the six clubs.
        clubs = set(CTX.report["protocol"]["clubs"])
        rule = np.where(frame.opponent.isin(clubs), 0.5, 1.0)
        self.assertTrue(np.array_equal(rule, weights))

    def test_elo_baseline_and_market_scores(self):
        # The scoreboard and the "그래서?" sentence read these; they must be the pipeline's cohort metrics.
        from epl_tactics.focused_model import cohort_metrics
        frame, scores = ordered_heldout(), PAYLOAD["fixed_model_scores"]
        elo = np.asarray(PAYLOAD["heldout"]["py_elo"])
        for key, value in cohort_metrics(frame, elo).items():
            self.assertAlmostEqual(scores["elo"]["overall"][key], value, places=12, msg=key)
        self.assertEqual(sorted(scores["elo"]["by_season"]), sorted(scores["by_season"]))
        for season, block in scores["elo"]["by_season"].items():
            mask = frame.season.eq(season).to_numpy()
            self.assertAlmostEqual(block["accuracy"], cohort_metrics(frame[mask], elo[mask])["accuracy"], places=12)
            self.assertAlmostEqual(block["log_loss"], cohort_metrics(frame[mask], elo[mask])["log_loss"], places=12)
        market = frame[["market_loss", "market_draw", "market_win"]].to_numpy(float)
        ok = np.isfinite(market).all(1)
        self.assertEqual(scores["market"]["rows"], int(ok.sum()))
        self.assertAlmostEqual(scores["market"]["odds"]["log_loss"], cohort_metrics(frame[ok], market[ok])["log_loss"], places=12)
        proba = np.asarray(PAYLOAD["heldout"]["py_proba"])
        self.assertAlmostEqual(scores["market"]["model"]["accuracy"], cohort_metrics(frame[ok], proba[ok])["accuracy"], places=12)
        # The page summarises the comparison in words; the official (refit) comparison must not contradict it.
        versus = CTX.report["comparisons"]["selected_vs_elo"]["log_loss"]
        self.assertLessEqual(versus["ci95_low"], 0.0)

    def test_formation_history_reproduces_the_formation_inputs(self):
        # Per row, the known formations of the club's previous 20 matches: exactly what formation_used_last20 used.
        held, frame = PAYLOAD["heldout"], ordered_heldout()
        history = held["formation_history"]
        self.assertEqual(len(history), len(frame))
        for row, counts in zip(frame.to_dict("records"), history):
            self.assertLessEqual(sum(counts.values()), 20)
            self.assertNotIn("__missing__", counts)
            total, formation = sum(counts.values()), row["formation"]
            if formation == "__missing__" or total == 0:
                self.assertTrue(pd.isna(row["formation_used_last20"]), row["match_id"])
            else:
                self.assertAlmostEqual(counts.get(formation, 0) / total, row["formation_used_last20"], places=9, msg=row["match_id"])
            state = CTX.states[row["team"]]
            before = sum(d < row["date"] for d in state.dates)
            self.assertEqual(state.formations[before - 1] if before else "__missing__", row["previous_formation"])

    def test_made_up_fixture_lineups_use_recent_unique_matches(self):
        scenario, rows = PAYLOAD["scenario"], CTX.rows
        clubs = set(scenario["clubs"])
        for team, snap in scenario["snapshots"].items():
            mine = rows[rows.team.eq(team)][["match_id", "date", *[f"team_lineup_{k}" for k in scenario["lineup_keys"]]]]
            mine.columns = ["match_id", "date", *scenario["lineup_keys"]]
            theirs = rows[rows.opponent.eq(team)][["match_id", "date", *[f"opponent_lineup_{k}" for k in scenario["lineup_keys"]]]]
            theirs.columns = mine.columns
            every = pd.concat([mine, theirs])
            unique = every.drop_duplicates("match_id").sort_values(["date", "match_id"])
            # a derby's two rows describe the same lineup, so keeping one of them loses nothing
            pairs = every[every.match_id.duplicated(keep=False)].groupby("match_id")
            for _, group in list(pairs)[:20]:
                self.assertTrue(group[scenario["lineup_keys"]].nunique(dropna=False).le(1).all(), team)
            recent = unique.tail(10)
            window = snap["lineup_window"]
            self.assertEqual(window["matches"], len(recent))
            self.assertEqual(window["first"], str(recent.date.min().date()))
            self.assertEqual(window["last"], str(recent.date.max().date()))
            for key in scenario["lineup_keys"]:
                expected = recent[key].mean()
                if pd.isna(expected):
                    self.assertIsNone(snap["lineup"][key])
                else:
                    self.assertAlmostEqual(snap["lineup"][key], expected, places=5, msg=(team, key))
            if team not in clubs:   # other clubs only appear as the opponent of one of the six clubs
                self.assertTrue(mine.empty, team)
                self.assertTrue(set(recent.match_id) <= set(theirs.match_id), team)
        # the "older matches can be mixed in" caveat on the page is real for at least one current opponent
        oldest = min(s["lineup_window"]["first"] for s in scenario["snapshots"].values() if s["lineup_window"]["first"])
        self.assertLess(pd.Timestamp(oldest), pd.Timestamp(scenario["latest_match_date"]) - pd.Timedelta(days=365))

    def test_influence_is_an_exact_attribution_of_the_model(self):
        from epl_tactics.lab.export_s4 import ROUND, raw_input_owner, win_contributions
        frame = ordered_heldout().copy()
        frame[CTX.numeric] = frame[CTX.numeric].round(ROUND)
        contributions, bias = win_contributions(CTX.model, frame, raw_input_owner(PAYLOAD["model"]["pre"]))
        win = np.asarray(PAYLOAD["heldout"]["py_proba"])[:, 2]
        self.assertLess(np.abs(bias + contributions.sum(axis=1).to_numpy() - win).max(), 1e-8)
        influence = PAYLOAD["influence"]
        self.assertEqual(influence["rows"], len(frame))
        self.assertAlmostEqual(influence["bias_win"], bias, places=6)
        self.assertTrue(set(influence["by_input"]) <= set(CTX.numeric) | {"team", "opponent", "formation", "team_formation", "previous_formation"})
        means = contributions.abs().mean()
        for name, value in influence["by_input"].items():
            self.assertAlmostEqual(value, means[name], places=6, msg=name)
        for base in ["elo", "goals_for_20", "rest_days"]:
            combined = contributions[[f"team_{base}", f"opponent_{base}", f"diff_{base}"]].sum(axis=1).abs().mean()
            self.assertAlmostEqual(influence["pairs"][base], combined, places=6, msg=base)
        # The takeaway names the two inputs that move the forecast most.
        top = sorted(influence["by_input"], key=influence["by_input"].get, reverse=True)[:2]
        self.assertEqual(set(top), {"diff_elo", "diff_goals_for_20"})

    def test_influence_matches_the_browser_engine(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        from epl_tactics.lab.report import sanitize
        script = r"""
const fs = require('node:fs');
const Engine = require(process.argv[2]);
const p = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const e = Engine.create(p), H = p.heldout, acc = {}, pairs = {};
const bases = Object.keys(p.influence.pairs);
for (let i = 0; i < H.rows.length; i++) {
  const c = {};
  for (const x of e.explain(Engine.heldoutRow(H, i)).contributions) { c[x.name] = x.win; acc[x.name] = (acc[x.name] || 0) + Math.abs(x.win); }
  for (const b of bases) pairs[b] = (pairs[b] || 0) + Math.abs((c['team_' + b] || 0) + (c['opponent_' + b] || 0) + (c['diff_' + b] || 0));
}
const n = H.rows.length;
for (const k in acc) acc[k] /= n;
for (const k in pairs) pairs[k] /= n;
console.log(JSON.stringify({ byInput: acc, pairs }));
"""
        with tempfile.TemporaryDirectory() as folder:
            path, js = Path(folder) / "s4.json", Path(folder) / "influence.cjs"
            path.write_text(json.dumps(sanitize(PAYLOAD), allow_nan=False), encoding="utf-8")
            js.write_text(script, encoding="utf-8")
            done = subprocess.run([node, str(js), str(ROOT / "epl_tactics" / "lab" / "engine.js"), str(path)],
                                  capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        result = json.loads(done.stdout.strip().splitlines()[-1])
        influence = PAYLOAD["influence"]
        for name, value in influence["by_input"].items():
            self.assertAlmostEqual(result["byInput"].get(name, 0.0), value, delta=2e-6, msg=name)
        for base, value in influence["pairs"].items():
            self.assertAlmostEqual(result["pairs"][base], value, delta=2e-6, msg=base)

    def test_input_stats_match_training_rows(self):
        stats, train = PAYLOAD["stats"], CTX.train
        self.assertEqual(set(stats), set(CTX.numeric))
        for name, s in stats.items():
            series = pd.to_numeric(train[name], errors="coerce")
            self.assertAlmostEqual(s["missing"], float(series.isna().mean()), places=6, msg=name)
            if s["min"] is None:
                continue
            ordered = [s[k] for k in ["min", "p05", "p25", "median", "p75", "p95", "max"]]
            self.assertEqual(ordered, sorted(ordered), name)
        for name in ["team_elo", "opponent_goals_for_20", "team_rest_days", "diff_elo"]:
            valid = pd.to_numeric(train[name], errors="coerce").dropna()
            self.assertAlmostEqual(stats[name]["min"], valid.min(), places=5)
            self.assertAlmostEqual(stats[name]["max"], valid.max(), places=5)
            self.assertAlmostEqual(stats[name]["p05"], valid.quantile(.05), places=5)
            self.assertAlmostEqual(stats[name]["p95"], valid.quantile(.95), places=5)

    def test_scenario_block(self):
        scenario, league = PAYLOAD["scenario"], CTX.league
        self.assertEqual(scenario["clubs"], list(CTX.report["protocol"]["clubs"]))
        last = league.date.max()
        self.assertEqual(scenario["latest_match_date"], str(last.date()))
        self.assertEqual(scenario["nominal_kickoff"], str((last + pd.Timedelta(days=7)).date()))
        season = league[league.date.eq(last)].season.iloc[0]
        self.assertEqual(scenario["latest_season"], season)
        current = set(league[league.season.eq(season)][["home_team", "away_team"]].to_numpy().ravel())
        self.assertEqual(set(scenario["opponents"]), current)
        self.assertTrue(set(scenario["clubs"]) <= set(scenario["opponents"]))
        self.assertEqual(set(scenario["snapshots"]), current)
        for team, snap in scenario["snapshots"].items():
            state = CTX.states[team]
            self.assertAlmostEqual(snap["home"]["elo"], state.elo, places=5)
            self.assertEqual(set(snap["home"]), set(snap["away"]))
            self.assertEqual(snap["last_match"], str(state.dates[-1].date()))
            self.assertEqual(set(snap["lineup"]), set(scenario["lineup_keys"]))
            self.assertTrue(all(f != "__missing__" for f in snap["formations_last20"]))
            self.assertLessEqual(len(snap["formations_last20"]), 20)
            self.assertLessEqual(snap["lineup_window"]["matches"], 10)
        known = CTX.rows.formation.ne("__missing__")
        self.assertEqual(sum(f["matches"] for f in scenario["formations"]), int(known.sum()))
        self.assertEqual({f["formation"] for f in scenario["formations"]}, set(CTX.rows.formation[known]))
        # every model input of a made-up fixture is defined by the snapshot block
        keys = {f"team_{k}" for k in scenario["snapshots"]["Arsenal"]["home"]} | {f"diff_{k}" for k in scenario["diff_bases"]}
        keys |= {f"{side}_lineup_{k}" for side in ["team", "opponent"] for k in scenario["lineup_keys"]}
        keys |= {f"opponent_{k}" for k in scenario["snapshots"]["Arsenal"]["home"]} | set(scenario["formation_numeric"]) | {"is_home"}
        self.assertEqual(keys, set(CTX.numeric))

    def test_forest_layout(self):
        forest, pre = PAYLOAD["model"]["forest"], PAYLOAD["model"]["pre"]
        self.assertEqual(forest["n_trees"], CTX.spec["n_estimators"])
        offsets = np.frombuffer(base64.b64decode(forest["offsets"]), dtype="<i4")
        self.assertEqual(len(offsets), forest["n_trees"] + 1)
        self.assertEqual(int(offsets[-1]), forest["n_nodes"])
        self.assertEqual(forest["n_nodes"], sum(t.tree_.node_count for t in CTX.model.estimator_.estimators_))
        self.assertEqual(forest["n_features"], len(pre["feature_names"]))
        self.assertEqual(len(pre["numeric"]), len(CTX.numeric))
        self.assertEqual(len(pre["median"]), len(pre["numeric"]))
        self.assertEqual(len(pre["mean"]), len(pre["numeric"]) + len(pre["indicator_of"]))

    def test_payload_is_json_safe_and_compact(self):
        from epl_tactics.lab.report import sanitize
        text = json.dumps(sanitize(PAYLOAD), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.assertLess(len(text.encode("utf-8")), 3_200_000)

    def test_honesty_claims_on_the_page_hold(self):
        comparisons = CTX.report["comparisons"]
        # "포메이션 입력은 예측을 좋게 만든다는 증거가 없고(효과의 신뢰구간이 0을 포함)"
        for metric in ["accuracy", "log_loss"]:
            gain = comparisons["formation_gain"][metric]
            self.assertLessEqual(gain["ci95_low"], 0.0)
            self.assertGreaterEqual(gain["ci95_high"], 0.0)
            # "Elo 기준 모델보다 뚜렷하게 낫지도 않습니다"
            versus = comparisons["selected_vs_elo"][metric]
            self.assertLessEqual(versus["ci95_low"], 0.0)
            self.assertGreaterEqual(versus["ci95_high"], 0.0)
        # "시장(배당)은 … 모델이 따라잡지 못하는 것이 자연스럽습니다": on the same held-out rows the odds score better
        from epl_tactics.focused_model import cohort_metrics
        frame = ordered_heldout()
        market = frame[["market_loss", "market_draw", "market_win"]].to_numpy(float)
        ok = np.isfinite(market).all(1)
        odds = cohort_metrics(frame[ok], market[ok])
        mine = cohort_metrics(frame[ok], np.asarray(PAYLOAD["heldout"]["py_proba"])[ok])
        self.assertGreater(odds["accuracy"], mine["accuracy"])
        self.assertLess(odds["log_loss"], mine["log_loss"])

    def test_fragment_has_no_hard_coded_data_numbers(self):
        html = FRAGMENT.read_text(encoding="utf-8")
        self.assertEqual(html.count("<script"), 1)
        self.assertIn("Lab.register('s4'", html)
        ids = re.findall(r'\bid="([^"]+)"', html)
        self.assertTrue(ids and all(i.startswith("s4-") for i in ids), ids)
        model, scores = PAYLOAD["model"], PAYLOAD["fixed_model_scores"]
        acc = scores["overall"]["accuracy"]
        for needle in [str(model["heldout_rows"]), str(model["heldout_matches"]), str(model["train_rows"]),
                       model["train_last_date"], f"{acc * 100:.1f}", PAYLOAD["scenario"]["latest_match_date"],
                       f"{scores['elo']['overall']['accuracy'] * 100:.1f}%", f"{scores['elo']['overall']['log_loss']:.3f}",
                       f"{scores['market']['odds']['accuracy'] * 100:.1f}%", f"{PAYLOAD['influence']['bias_win'] * 100:.1f}%",
                       f"{PAYLOAD['influence']['pairs']['elo'] * 100:.1f}%p", str(round(PAYLOAD["stats"]["diff_elo"]["median"])) + "점"]:
            self.assertNotIn(needle, html)
        # The made-up-fixture box writes its own lineup sentence instead of echoing a one-size-fits-all note.
        self.assertNotIn("SC.note", html)

    def test_browser_engine_parity(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        from epl_tactics.lab.report import sanitize
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s4.json"
            path.write_text(json.dumps(sanitize(PAYLOAD), allow_nan=False), encoding="utf-8")
            done = subprocess.run([node, str(ROOT / "scripts" / "check_lab_engine.cjs"), str(path)],
                                  capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        result = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(result["rows"], PAYLOAD["model"]["heldout_rows"])
        self.assertLess(result["maxProbaDiff"], 2e-6)
        self.assertLess(result["maxContributionSumDiff"], 1e-9)


if __name__ == "__main__":
    unittest.main()
