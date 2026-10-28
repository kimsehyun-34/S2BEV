from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from epl_tactics.data import iter_json_array, normalize_formation, validate_matches
from epl_tactics.features import FEATURES, build_features, chronological_split
from epl_tactics.model import load_model, train
from epl_tactics.recommend import recommend_all, recommend_match
from epl_tactics.report import write_report


def synthetic_matches(days: int = 50) -> pd.DataFrame:
    """Test fixtures only; never used to claim EPL predictive performance."""
    rows = []
    schedule = [[("Test A", "Test B"), ("Test C", "Test D")],
                [("Test A", "Test C"), ("Test D", "Test B")],
                [("Test D", "Test A"), ("Test B", "Test C")]]
    for day in range(days):
        for game, (home, away) in enumerate(schedule[day % 3]):
            hg, ag = [(1, 0), (1, 1), (0, 2)][(day + game) % 3]
            rows.append(dict(match_id=str(day * 2 + game), date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                             season="TEST", home_team=home, away_team=away,
                             home_formation=["4-3-3", "4-4-2"][day % 2],
                             away_formation=["4-3-3", "4-4-2"][(day + 1) % 2],
                             home_goals=hg, away_goals=ag))
    return validate_matches(pd.DataFrame(rows))


class DataTests(unittest.TestCase):
    def test_formation_normalization_and_validation(self):
        self.assertEqual(normalize_formation(4231), "4-2-3-1")
        self.assertEqual(normalize_formation("4-1-2-1-2"), "4-1-2-1-2")
        self.assertEqual(normalize_formation("4-2-4-0"), "4-2-4-0")
        for value in ["4-4-3", "4-0-6", "442junk", "1-4-4-2", None]:
            with self.assertRaises(ValueError):
                normalize_formation(value)

    def test_reject_bad_scores_and_duplicate_matches(self):
        frame = synthetic_matches(2)
        for value in [-1, 1.5, np.inf]:
            broken = frame.copy()
            broken["home_goals"] = broken["home_goals"].astype(float)
            broken.loc[0, "home_goals"] = value
            with self.assertRaises(ValueError):
                validate_matches(broken)
        with self.assertRaises(ValueError):
            validate_matches(pd.concat([frame, frame.iloc[:1]], ignore_index=True))

    def test_reject_double_appearance_same_day(self):
        frame = synthetic_matches(2)
        frame.loc[1, "home_team"] = frame.loc[0, "home_team"]
        with self.assertRaises(ValueError):
            validate_matches(frame)

    def test_reject_nat_dates(self):
        frame = synthetic_matches(2)
        frame["date"] = frame["date"].astype(str)
        frame.loc[0, "date"] = "NaT"
        with self.assertRaises(ValueError):
            validate_matches(frame)

    def test_incremental_utf8_and_early_exit(self):
        source = json.dumps([{"team": "서울", "type": {"name": "Starting XI"}},
                             {"team": "Arsenal"}], ensure_ascii=False).encode("utf-8")
        self.assertEqual(list(iter_json_array(iter(bytes([b]) for b in source)))[0]["team"], "서울")
        stream = iter_json_array(iter([source[:source.index(b'}, {') + 1]]))
        self.assertEqual(next(stream)["team"], "서울")
        with self.assertRaises(ValueError):
            list(iter_json_array(iter([b'[{"broken":'])))


class TemporalTests(unittest.TestCase):
    def test_same_day_and_future_results_do_not_leak(self):
        original = synthetic_matches(4)
        changed = original.copy()
        cutoff = original.date.unique()[1]
        changed.loc[changed.date >= cutoff, "home_goals"] = 19
        before, _ = build_features(original)
        after, _ = build_features(changed)
        pd.testing.assert_frame_equal(before.loc[before.date <= cutoff, FEATURES],
                                      after.loc[after.date <= cutoff, FEATURES])

    def test_previous_results_really_update_features(self):
        rows, _ = build_features(synthetic_matches(2))
        first = rows[(rows.team == "Test A") & (rows.date == rows.date.min())].iloc[0]
        second = rows[(rows.team == "Test A") & (rows.date == rows.date.max())].iloc[0]
        self.assertEqual(first.team_games_played, 0)
        self.assertEqual(second.team_games_played, 1)
        self.assertEqual(second.team_points_5, 3)
        self.assertEqual(second.team_rest_days, 1)

    def test_dates_and_both_sides_stay_in_one_split(self):
        rows, _ = build_features(synthetic_matches())
        train_rows, validation, test = chronological_split(rows)
        self.assertLess(train_rows.date.max(), validation.date.min())
        self.assertLess(validation.date.max(), test.date.min())
        self.assertFalse(set(train_rows.match_id) & set(test.match_id))
        for split in [train_rows, validation, test]:
            self.assertTrue(split.groupby("match_id").size().eq(2).all())

    def test_seasons_have_separate_history(self):
        early = synthetic_matches(2)
        later = early.copy()
        later["season"] = "TEST2"
        later["date"] += pd.Timedelta(days=365)
        later["match_id"] += "next"
        rows, _ = build_features(validate_matches(pd.concat([early, later], ignore_index=True)))
        start = rows[rows.season == "TEST2"].date.min()
        self.assertTrue(rows[(rows.season == "TEST2") & (rows.date == start)].team_games_played.eq(0).all())


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        csv = cls.root / "matches.csv"
        synthetic_matches().to_csv(csv, index=False)
        cls.metrics = train(csv, cls.root / "model")
        cls.bundle = load_model(cls.root / "model" / "model.joblib")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_probabilities_points_and_support(self):
        result = recommend_match(self.bundle, "Test A", "Test B")
        self.assertIsNone(result["recommended_formation"])
        self.assertEqual(result["status"], "not_validated")
        self.assertIn(result["exploratory_formation"], ["4-3-3", "4-4-2"])
        for row in result["candidates"]:
            self.assertAlmostEqual(row["p_loss"] + row["p_draw"] + row["p_win"], 1)
            self.assertAlmostEqual(row["expected_points"], row["p_draw"] + 3 * row["p_win"])
            self.assertTrue(0 <= row["expected_points"] <= 3)

    def test_insufficient_support_abstains(self):
        result = recommend_match(self.bundle, "Test A", "Test B", min_team_support=999)
        self.assertIsNone(result["recommended_formation"])
        self.assertEqual(result["comparison_status"], "insufficient_support")

    def test_single_formation_is_not_a_comparison(self):
        bundle = deepcopy(self.bundle)
        mask = (bundle["support"].team == "Test A") & (bundle["support"].formation == "4-3-3")
        bundle["support"].loc[mask, "matches"] = 0
        result = recommend_match(bundle, "Test A", "Test B")
        self.assertEqual(result["comparison_status"], "single_supported_formation")

    def test_reject_unknown_team_same_team_and_historical_date(self):
        for team, opponent, as_of in [("Unknown", "Test B", None), ("Test A", "Test A", None),
                                      ("Test A", "Test B", "2020-01-03")]:
            with self.assertRaises(ValueError):
                recommend_match(self.bundle, team, opponent, as_of=as_of)

    def test_report_all_teams_and_safe_json(self):
        results = recommend_all(self.bundle)
        self.assertEqual(len(results), 4)
        result = write_report(results, self.metrics, self.root / "reports")
        page = result.read_text(encoding="utf-8")
        self.assertNotIn("__PAYLOAD__", page)
        self.assertNotIn("__TABLE__", page)
        self.assertEqual(len(pd.read_csv(self.root / "reports" / "team_recommendations.csv")), 4)
        rankings = pd.read_csv(self.root / "reports" / "formation_rankings.csv")
        self.assertEqual(int(rankings.selected.sum()), 0)
        malicious = deepcopy(results)
        malicious[0]["team"] = '</script><img src=x onerror="alert(1)">'
        write_report(malicious, self.metrics, self.root / "escaped")
        escaped = (self.root / "escaped" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn(malicious[0]["team"], escaped)
        HTMLParser().feed(escaped)


if __name__ == "__main__":
    unittest.main()
