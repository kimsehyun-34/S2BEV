"""Shared pieces of the lab page: input catalog, browser engine parity, hypothetical-fixture builder."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, MODEL = ROOT / "data" / "focused", ROOT / "artifacts" / "focused"
HAVE_DATA = (DATA / "matches.csv").exists() and (MODEL / "metrics.json").exists()
NODE = shutil.which("node")
_CTX = {}


def context():
    if "ctx" not in _CTX:
        from epl_tactics.lab.context import load_context
        _CTX["ctx"] = load_context(DATA, MODEL)
    return _CTX["ctx"]


class CatalogTests(unittest.TestCase):
    def test_every_model_input_has_a_korean_label_group_and_kind(self):
        from epl_tactics.focused_features import CATEGORIES
        from epl_tactics.lab.catalog import GROUP_LABEL, build_catalog
        numeric = json.loads((MODEL / "features.json").read_text(encoding="utf-8"))["numeric_features"] if HAVE_DATA else \
            ["team_elo", "opponent_elo", "diff_elo", "is_home", "team_points_5", "team_possession_coverage20",
             "team_venue_goals_for", "opponent_lineup_prior_sot_rate", "defenders", "formation_switch", "team_rest_days"]
        catalog = build_catalog(numeric)
        names = [entry["name"] for entry in catalog]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(set(CATEGORIES) <= set(names) and set(numeric) <= set(names))
        for entry in catalog:
            self.assertIn(entry["group"], GROUP_LABEL, entry)
            self.assertTrue(entry["label"] and entry["kind"], entry)
            self.assertFalse(entry["label"].isascii(), f"label should be Korean: {entry}")

    def test_unknown_column_is_rejected_instead_of_silently_unlabeled(self):
        from epl_tactics.lab.catalog import describe
        with self.assertRaises(KeyError):
            describe("team_mystery_7")


@unittest.skipUnless(HAVE_DATA and NODE, "needs data/focused, artifacts/focused and node")
class EngineParityTests(unittest.TestCase):
    def test_browser_engine_reproduces_python_model_on_every_held_out_row(self):
        from epl_tactics.lab import export_s4
        ctx = context()
        payload = export_s4.build(ctx)
        self.assertEqual(payload["model"]["heldout_rows"], len(payload["heldout"]["rows"]))
        self.assertTrue(payload["model"]["train_last_date"] < payload["model"]["heldout_first_date"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "s4.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            done = subprocess.run([NODE, str(ROOT / "scripts" / "check_lab_engine.cjs"), str(path)],
                                  capture_output=True, text=True, encoding="utf-8", timeout=300)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        result = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertLess(result["maxProbaDiff"], 2e-6)
        self.assertLess(result["maxContributionSumDiff"], 1e-9)

    def test_playground_model_matches_official_first_test_season_predictions(self):
        # Same frozen configuration + same training window => identical to the official 2024/25 forecasts.
        ctx = context()
        season = ctx.rows[ctx.rows.season.eq(ctx.test_start)]
        official = ctx.test_pred[ctx.test_pred.season.eq(ctx.test_start)].set_index(["match_id", "team"])
        official = official.loc[list(zip(season.match_id, season.team))]
        columns = [f"selected_with_formation_{c}" for c in ["p_loss", "p_draw", "p_win"]]
        np.testing.assert_allclose(ctx.model.predict_proba(season), official[columns].to_numpy(), atol=1e-9)


@unittest.skipUnless(HAVE_DATA and NODE, "needs data/focused, artifacts/focused and node")
class ScenarioBuilderTests(unittest.TestCase):
    """A made-up fixture must be featurised exactly like the real match it reproduces.

    History is cut just before a matchday; the snapshot taken there plus the browser builder must give the same
    inputs as the real rows of that matchday (lineup excepted: the real XI is unknown beforehand)."""

    def check_matchday(self, day: str):
        from epl_tactics.focused_features import build_focused_features
        from epl_tactics.lab import export_s4
        ctx = context()
        cutoff = pd.Timestamp(day)
        with tempfile.TemporaryDirectory() as temp:
            for name in ["matches.csv", "league_context.csv"]:
                frame = pd.read_csv(DATA / name, parse_dates=["date"])
                frame[frame.date < cutoff].to_csv(Path(temp) / name, index=False, date_format="%Y-%m-%d")
            _, _, _, states = build_focused_features(temp, return_states=True)
            real = ctx.rows[ctx.rows.date.eq(cutoff)]
            self.assertGreater(len(real), 0)
            fake = types.SimpleNamespace(states=states, rows=ctx.rows[ctx.rows.date < cutoff])
            teams = set(real.team) | set(real.opponent)
            snapshots = {t: export_s4.team_snapshot(fake, t, cutoff) for t in teams}
            fixtures = [{"team": r.team, "opponent": r.opponent, "home": bool(r.is_home), "formation": r.formation,
                         "previousFormation": r.previous_formation} for r in real.itertuples()]
            scenario = {"snapshots": snapshots, "lineup_keys": export_s4.LINEUP_KEYS}
            (Path(temp) / "in.json").write_text(json.dumps({"scenario": scenario, "fixtures": fixtures}), encoding="utf-8")
            subprocess.run([NODE, str(ROOT / "scripts" / "lab_scenario_dump.cjs"), str(Path(temp) / "in.json"), str(Path(temp) / "out.json")],
                           check=True, timeout=120)
            built = json.loads((Path(temp) / "out.json").read_text(encoding="utf-8"))
        compared = 0
        for row, made in zip(real.to_dict("records"), built):
            for name in [*ctx.numeric, "team", "opponent", "formation", "team_formation", "previous_formation"]:
                if "lineup" in name:
                    continue
                expected, got = row[name], made.get(name)
                if isinstance(expected, str):
                    self.assertEqual(got, expected, (name, row["match_id"]))
                elif pd.isna(expected):
                    self.assertIsNone(got, (name, row["match_id"]))
                else:
                    self.assertIsNotNone(got, (name, row["match_id"]))
                    self.assertAlmostEqual(got, expected, delta=2e-6 * max(1.0, abs(expected)), msg=f"{name} {row['match_id']}")
                compared += 1
        self.assertGreater(compared, 150 * len(real))

    def test_last_matchday_in_the_data(self):
        self.check_matchday(str(context().rows.date.max().date()))

    def test_a_mid_season_matchday(self):
        self.check_matchday("2025-02-22")


if __name__ == "__main__":
    unittest.main()
