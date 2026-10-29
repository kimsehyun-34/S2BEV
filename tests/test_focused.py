from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from epl_tactics.focused_data import CLUBS, parse_facts
from epl_tactics.focused_features import build_focused_features
from epl_tactics.focused_model import (FocusedModel, cohort_metrics, feature_columns,
                                      specifications, tune, weighted_week_interval)


def fixture_frames():
    fixtures = []
    for year in [2017,2018,2019,2020]:
        for day in range(12):
            for index,(home,away) in enumerate([(CLUBS[0],CLUBS[1]),(CLUBS[2],"Other Club")]):
                hg,ag = [(2,0),(1,1),(0,1)][(day+index)%3]
                row = {"match_id":f"{year}_{day}_{index}","date":f"{year}-08-{day+1:02d}",
                       "season":f"{year}/{year+1}","home_team":home,"away_team":away,
                       "home_goals":hg,"away_goals":ag}
                for side in ["home","away"]:
                    row[f"{side}_shots"] = 10.0
                    row[f"{side}_sot"] = 4.0
                    row[f"{side}_formation"] = "4-3-3" if day%2 else "4-2-3-1"
                    row[f"{side}_players"] = json.dumps([{"id":f"{side}_{i}","starter":True,"appearance":1,
                                                           "goals":float(i==0),"assists":0.0,"shots":1.,"sot":1.} for i in range(11)])
                fixtures.append(row)
    return pd.DataFrame(fixtures)


class FocusedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frame = fixture_frames()
        self.save(self.frame)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, frame):
        frame.to_csv(self.root/"matches.csv",index=False)
        frame.drop(columns=[c for c in frame if "formation" in c or "players" in c]).to_csv(self.root/"league_context.csv",index=False)

    def test_only_selected_clubs_are_supervised_and_derbies_weight_once(self):
        rows,_,_ = build_focused_features(self.root)
        self.assertTrue(set(rows.team).issubset(CLUBS))
        self.assertNotIn("Other Club",set(rows.team))
        self.assertTrue(rows.groupby("match_id").fixture_weight.sum().eq(1).all())

    def test_future_current_match_stats_and_player_productivity_do_not_leak(self):
        before,numeric,_ = build_focused_features(self.root)
        changed = self.frame.copy()
        cutoff = "2018-08-05"
        mask = changed.date.ge(cutoff)
        changed.loc[mask,"home_goals"] = 20
        changed.loc[mask,"home_shots"] = 100
        changed.loc[mask,"home_players"] = changed.loc[mask,"home_players"].map(
            lambda raw:json.dumps([{**p,"goals":50.} for p in json.loads(raw)]))
        self.save(changed)
        after,_,_ = build_focused_features(self.root)
        columns = numeric+["team","opponent","formation","team_formation","previous_formation"]
        pd.testing.assert_frame_equal(before.loc[before.date<=cutoff,columns],after.loc[after.date<=cutoff,columns])

    def test_formation_ablation_removes_all_current_and_derived_formation_fields(self):
        _,numeric,_ = build_focused_features(self.root)
        categorical,numbers = feature_columns(numeric,False)
        self.assertEqual(categorical,["team","opponent"])
        self.assertFalse(any("formation" in name for name in numbers))
        self.assertNotIn("defenders",numbers)

    def test_unknown_previous_formation_is_not_a_switch(self):
        frame = self.frame.copy()
        frame["home_formation"] = np.where(frame.date.eq("2018-08-03"), None, frame.home_formation)
        self.save(frame)
        rows,_,_ = build_focused_features(self.root)
        after = rows[(rows.team == CLUBS[0]) & rows.previous_formation.eq("__missing__") & rows.formation_known.eq(1)]
        self.assertFalse(after.empty)
        self.assertTrue(after.formation_switch.isna().all())
        self.assertTrue(rows[rows.formation_known.eq(0)].formation_used_last20.isna().all())
        team = rows[rows.team == CLUBS[0]].sort_values("date").reset_index(drop=True)
        for i in range(1, len(team)):
            known = [f for f in team.formation[max(0, i-20):i] if f != "__missing__"]
            if team.formation_known[i] and known:
                self.assertAlmostEqual(team.formation_used_last20[i], sum(f == team.formation[i] for f in known)/len(known))

    def test_random_search_reproducible_and_not_only_two_fixed_leaf_sizes(self):
        first = specifications(32)
        self.assertEqual(first,specifications(32))
        rf = [s for s in first if s["family"]=="rf"]
        self.assertEqual(len(rf),32)
        self.assertGreater(len({s["max_depth"] for s in rf}),2)
        self.assertGreater(len({s["min_samples_leaf"] for s in rf}),2)

    def test_missing_features_train_and_only_forward_folds_used(self):
        rows,numeric,_ = build_focused_features(self.root)
        specs = [{"family":"rf","n_estimators":10,"max_depth":3,"min_samples_leaf":2}]
        result = tune(rows,numeric,["2018/2019","2019/2020"],specs,True,self.root)
        self.assertEqual(result["trials"],1)
        self.assertEqual([f["season"] for f in result["best"]["folds"]],["2018/2019","2019/2020"])
        model = FocusedModel(specs[0],numeric,True).fit(rows[rows.season<"2020/2021"])
        forecast = model.predict_proba(rows[rows.season=="2020/2021"])
        np.testing.assert_allclose(forecast.sum(1),1)

    def test_metric_and_intervals_do_not_count_derby_twice(self):
        rows,_,_ = build_focused_features(self.root)
        probability = np.full((len(rows),3),1/3)
        metrics = cohort_metrics(rows,probability)
        self.assertEqual(metrics["unique_matches"],len(self.frame))
        self.assertEqual(metrics["team_appearances"],len(self.frame)*3//2)
        interval = weighted_week_interval(rows,probability,probability,repeats=100)
        self.assertEqual(interval["improvement"],0)
        self.assertEqual(interval["ci95_low"],0)
        self.assertEqual(interval["ci95_high"],0)


if __name__ == "__main__":
    unittest.main()
