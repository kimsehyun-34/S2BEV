from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from epl_tactics.benchmark import _score_predictions, run_benchmark, predict_fixture
from epl_tactics.data import validate_matches
from epl_tactics.estimators_v2 import match_predictions
from epl_tactics.features_v2 import FEATURES_V2, build_features_v2
from epl_tactics.model import load_model
from epl_tactics.recent_data import parse_summary
from epl_tactics.statistics import paired_week_interval
from test_tactics import synthetic_matches


def multi_season_data():
    frames = []
    for year in range(2019, 2027):
        frame = synthetic_matches(24)
        frame["date"] += pd.DateOffset(years=year - 2020, months=7)
        frame["match_id"] = str(year) + "_" + frame["match_id"]
        frame["season"] = f"{year}/{year+1}"
        for side in ["home", "away"]:
            frame[f"{side}_shots"] = 12.0
            frame[f"{side}_shots_on_target"] = 4.0
            frame[f"{side}_possession"] = 50.0
        frames.append(frame)
    return validate_matches(pd.concat(frames, ignore_index=True))


class ForwardFeatureTests(unittest.TestCase):
    def test_current_and_future_stats_never_change_past_features(self):
        frame = multi_season_data()
        cutoff = frame.date.unique()[10]
        changed = frame.copy()
        mask = changed.date >= cutoff
        changed.loc[mask, ["home_goals", "home_shots", "away_possession"]] = [13, 80, 2]
        before, _ = build_features_v2(frame)
        after, _ = build_features_v2(changed)
        pd.testing.assert_frame_equal(before.loc[before.date <= cutoff, FEATURES_V2],
                                      after.loc[after.date <= cutoff, FEATURES_V2])

    def test_history_survives_season_reset_but_season_games_reset(self):
        rows, _ = build_features_v2(multi_season_data())
        new_season = rows[rows.season == "2020/2021"]
        first_day = new_season[new_season.date == new_season.date.min()]
        self.assertTrue(first_day.team_season_games.eq(0).all())
        self.assertTrue(first_day.team_history_games.gt(0).all())

    def test_earlier_statistics_really_change_later_inputs(self):
        frame = multi_season_data()
        original, _ = build_features_v2(frame)
        frame.loc[0, "home_shots"] = 60
        changed, _ = build_features_v2(frame)
        mask = original.date > original.date.min()
        self.assertFalse(original.loc[mask, "team_shots_5"].equals(changed.loc[mask, "team_shots_5"]))


class EvaluationTests(unittest.TestCase):
    def test_two_views_produce_one_coherent_fixture(self):
        rows, _ = build_features_v2(synthetic_matches(1))
        raw = np.tile([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], (2, 1))
        result = match_predictions(rows, raw)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[["p_loss", "p_draw", "p_win"]], [[.15, .25, .60]] * 2)

    def test_uniform_accuracy_is_expected_random_not_argmax_tiebreak(self):
        rows, _ = build_features_v2(synthetic_matches(5))
        frame = match_predictions(rows, np.full((len(rows), 3), 1/3))
        self.assertAlmostEqual(_score_predictions({"uniform_random": frame})["uniform_random"]["accuracy"], 1/3)

    def test_paired_bootstrap_identical_models_have_zero_gain(self):
        rows, _ = build_features_v2(synthetic_matches(25))
        frame = match_predictions(rows, np.full((len(rows), 3), 1/3))
        proba = frame[["p_loss", "p_draw", "p_win"]].to_numpy()
        for metric in ["accuracy", "log_loss"]:
            result = paired_week_interval(frame, proba, proba, score=metric, repeats=100)
            self.assertEqual(result["improvement"], 0)
            self.assertEqual(result["ci_95_low"], 0)
            self.assertEqual(result["ci_95_high"], 0)

    def test_provider_diamond_tag_and_zero_forward_formation(self):
        sample = {
            "header": {"id": "test", "season": {"year": 2025}, "competitions": [{
                "date": "2025-08-20T19:00Z", "status": {"type": {"completed": True}},
                "competitors": [{"homeAway": side, "team": {"displayName": side}, "score": "0"}
                                for side in ["home", "away"]]}]},
            "rosters": [{"homeAway": "home", "formation": "3-4-3-d"},
                        {"homeAway": "away", "formation": "4-2-4-0"}],
        }
        parsed = parse_summary(sample)
        self.assertEqual(parsed["home_formation"], "3-4-3")
        self.assertEqual(parsed["home_formation_raw"], "3-4-3-d")
        self.assertEqual(parsed["away_formation"], "4-2-4-0")
        sample["header"]["competitions"][0]["status"]["type"]["completed"] = False
        with self.assertRaises(ValueError):
            parse_summary(sample)


class V2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "synthetic.csv"
        multi_season_data().to_csv(cls.data, index=False)
        small_grid = [{"family": "logistic", "C": .03}, {"family": "poisson", "alpha": .3}]
        with patch("epl_tactics.benchmark.SPECS", small_grid):
            cls.report = run_benchmark(cls.data, cls.root / "model", cls.root / "no-market.csv")
        cls.bundle = load_model(cls.root / "model" / "model.joblib")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_protocol_selection_and_test_are_separated(self):
        protocol = json.loads((self.root / "model" / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["validation_seasons"], ["2022/2023", "2023/2024"])
        self.assertEqual(protocol["test_seasons"], ["2024/2025", "2025/2026", "2026/2027"])
        predictions = pd.read_csv(self.root / "model" / "test_predictions.csv")
        self.assertTrue(predictions.match_id.is_unique)
        self.assertEqual(len(predictions), 3 * 48)
        self.assertFalse(self.report["tactical_recommendation_validated"])
        self.assertTrue(all(team["recommendation"] is None for team in self.report["support_audit"]["teams"]))

    def test_fixture_forecast_and_reversed_fixture_are_consistent(self):
        direct = predict_fixture(self.bundle, "Test A", "Test B", True, "4-3-3", "4-4-2")
        reverse = predict_fixture(self.bundle, "Test B", "Test A", False, "4-4-2", "4-3-3")
        self.assertAlmostEqual(direct["p_win"], reverse["p_loss"])
        self.assertAlmostEqual(direct["p_draw"], reverse["p_draw"])
        self.assertAlmostEqual(direct["p_loss"] + direct["p_draw"] + direct["p_win"], 1)
        self.assertIsNone(direct["recommended_formation"])

    def test_prediction_rejects_future_history_and_missing_announced_formations(self):
        with self.assertRaises(ValueError):
            predict_fixture(self.bundle, "Test A", "Test B", as_of="2020-01-01")
        bundle = deepcopy(self.bundle)
        bundle["report"]["deployment"]["uses_formation"] = True
        with self.assertRaises(ValueError):
            predict_fixture(bundle, "Test A", "Test B")

    def test_evaluated_protocol_cannot_be_silently_changed(self):
        with self.assertRaises(ValueError):
            run_benchmark(self.data, self.root / "model", self.root / "no-market.csv",
                          validation_seasons=["2021/2022", "2023/2024"])


if __name__ == "__main__":
    unittest.main()
