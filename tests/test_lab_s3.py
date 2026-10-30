"""Step 3 (model training) payload: every number checked against the pipeline artifacts / source data."""
from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, MODEL = ROOT / "data" / "focused", ROOT / "artifacts" / "focused"
CACHE: dict = {}
ROUND4 = 1e-4          # payload floats are rounded to 4 decimals


def setUpModule():
    if not (DATA / "matches.csv").exists() or not (MODEL / "metrics.json").exists():
        raise unittest.SkipTest("focused data/artifacts are not available")
    from epl_tactics.lab import export_s3
    from epl_tactics.lab.context import load_context
    ctx = load_context(DATA, MODEL, fit=True)
    CACHE.update(ctx=ctx, payload=export_s3.build(ctx), module=export_s3)


class LabStep3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.p, cls.mod = CACHE["ctx"], CACHE["payload"], CACHE["module"]
        from epl_tactics.focused_model import PROBS, cohort_metrics
        cls.PROBS, cls.cohort_metrics = PROBS, staticmethod(cohort_metrics)

    # ------------------------------------------------------------------ payload hygiene
    def test_payload_is_json_safe_and_small(self):
        from epl_tactics.lab.report import sanitize
        text = json.dumps(sanitize(self.p), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.assertLess(len(text.encode("utf-8")), 1.2e6)

    def test_untuned_spec_matches_pipeline_default(self):
        from epl_tactics import focused_model
        source = inspect.getsource(focused_model.run_focused)
        literal = re.search(r"default_rf\s*=\s*(\{[^}]*\})", source).group(1)
        self.assertEqual(ast.literal_eval(literal), self.mod.UNTUNED_RF)

    # ------------------------------------------------------------------ official headline = comparison.html numbers
    def test_headline_matches_official_test_predictions(self):
        test = self.ctx.test_pred
        proba = test[[f"selected_with_formation_{c}" for c in self.PROBS]].to_numpy()
        recomputed = self.cohort_metrics(test, proba)
        official = self.ctx.report["test"]["selected_with_formation"]
        h = self.p["headline"]
        self.assertEqual(h["rows"], len(test))
        self.assertEqual(h["rows"], 504)
        self.assertEqual(h["matches"], test.match_id.nunique())
        for key in ["accuracy", "log_loss", "brier"]:
            self.assertAlmostEqual(recomputed[key], official[key], places=9)
            self.assertAlmostEqual(h[key], official[key], delta=ROUND4)
        self.assertAlmostEqual(h["accuracy"], 0.5433, delta=ROUND4)        # what comparison.html reports
        self.assertAlmostEqual(h["log_loss"], 0.9897, delta=ROUND4)
        self.assertEqual(h["n_trees"], self.ctx.spec["n_estimators"])

    # ------------------------------------------------------------------ B. tree
    def test_tree_thresholds_are_in_raw_units(self):
        """For every shown numeric split, 'raw value <= raw threshold' sends exactly the same training rows left."""
        model = self.ctx.model
        tree = model.estimator_.estimators_[0].tree_
        x = np.asarray(model.processor_.transform(self.ctx.train), dtype=np.float32)   # sklearn trees split on float32
        columns = self.mod.transformed_columns(model.processor_)
        index = {c["name"]: k for k, c in enumerate(columns)}
        imputer = model.processor_.named_transformers_["numeric"].named_steps["impute"]
        numeric = list(model.processor_.transformers_[1][2])
        checked = 0
        for node in self.p["tree"]["nodes"]:
            if "question" not in node:
                continue
            path = node["id"]
            sk = 0
            for step in path:
                sk = tree.children_left[sk] if step == "L" else tree.children_right[sk]
            self.assertEqual(columns[tree.feature[sk]]["name"], node["feature"])
            self.assertAlmostEqual(node["scaled"], float(tree.threshold[sk]), places=3)
            entry = columns[index[node["feature"]]]
            if entry["kind"] != "numeric":
                continue
            raw = self.ctx.train[entry["source"]].astype(float)
            raw = raw.fillna(imputer.statistics_[numeric.index(entry["source"])]).to_numpy()
            sklearn_left = x[:, index[node["feature"]]] <= tree.threshold[sk]
            exact_raw = entry["mean"] + tree.threshold[sk] * entry["scale"]
            np.testing.assert_array_equal(sklearn_left, raw <= exact_raw + 1e-9)
            self.assertAlmostEqual(node["raw"], exact_raw, delta=ROUND4)
            checked += 1
        self.assertGreater(checked, 0)

    def test_tree_node_counts_and_shares(self):
        nodes = {n["id"]: n for n in self.p["tree"]["nodes"]}
        self.assertEqual(nodes[""]["n"], self.p["tree"]["root_distinct_rows"])
        self.assertLessEqual(self.p["tree"]["root_distinct_rows"], int(self.ctx.train_mask.sum()))
        self.assertEqual(self.p["tree"]["train_rows"], int(self.ctx.train_mask.sum()))
        for node in nodes.values():
            self.assertAlmostEqual(sum(node["share"]), 1.0, places=2)
            if "question" in node:
                self.assertEqual(nodes[node["id"] + "L"]["n"] + nodes[node["id"] + "R"]["n"], node["n"])
                self.assertIn(node["yes"], {"left", "right"})
        self.assertEqual(self.p["tree"]["candidates_per_question"],
                         int(self.ctx.spec["max_features"] * self.p["tree"]["n_columns"]))

    # ------------------------------------------------------------------ B. vote
    def test_vote_average_is_forest_and_official_prediction(self):
        vote = self.p["vote"]
        self.assertEqual(vote["season"], self.ctx.test_start)
        test = self.ctx.test_pred
        rows = self.ctx.rows
        for example in vote["examples"]:
            trees = np.array([example["trees"][k] for k in ["loss", "draw", "win"]]).T
            self.assertEqual(trees.shape, (self.ctx.spec["n_estimators"], 3))
            np.testing.assert_allclose(trees.mean(0), example["forest"], atol=2e-3)
            row = rows[rows.match_id.eq(example["match_id"]) & rows.team.eq(example["team"])]
            self.assertEqual(len(row), 1)
            np.testing.assert_allclose(self.ctx.model.predict_proba(row)[0], example["forest"], atol=1e-4)
            official = test[test.match_id.eq(example["match_id"]) & test.team.eq(example["team"])]
            np.testing.assert_allclose(official[[f"selected_with_formation_{c}" for c in self.PROBS]].to_numpy()[0],
                                       example["forest"], atol=1e-4)
            self.assertEqual(example["outcome"], int(row.outcome.iloc[0]))
        self.assertEqual(len({e["match_id"] for e in vote["examples"]}), len(vote["examples"]))

    def test_vote_draw_example_note(self):
        """'the only match in the test period where the model picked a draw' is computed from the official predictions."""
        vote, test = self.p["vote"], self.ctx.test_pred
        picks = test[[f"selected_with_formation_{c}" for c in self.PROBS]].to_numpy().argmax(1)
        self.assertEqual(vote["test_rows"], len(test))
        self.assertEqual(vote["test_draw_picks"], int((picks == 1).sum()))
        self.assertEqual(vote["test_draw_picks"], self.p["results"]["confusion"]["selected_with_formation"]["per_class"][1]["predicted"])
        for example in vote["examples"]:
            row = test.match_id.eq(example["match_id"]) & test.team.eq(example["team"])
            self.assertEqual(example["official_pick"], int(picks[row.to_numpy()][0]))
        draw = next(e for e in vote["examples"] if e["key"] == "draw")
        if draw["official_pick"] == 1 and vote["test_draw_picks"] == 1:
            only = test[picks == 1]
            self.assertEqual((str(only.match_id.iloc[0]), only.team.iloc[0]), (draw["match_id"], draw["team"]))

    def test_single_trees_vs_forest(self):
        vote = self.p["vote"]
        season_rows = self.ctx.rows[self.ctx.rows.season.eq(vote["season"])]
        self.assertEqual(vote["rows"], len(season_rows))
        official = self.ctx.report["test_by_season"][vote["season"]]["selected_with_formation"]
        self.assertAlmostEqual(vote["forest"]["accuracy"], official["accuracy"], delta=ROUND4)
        self.assertAlmostEqual(vote["forest"]["log_loss"], official["log_loss"], delta=ROUND4)
        acc, ll = np.array(vote["tree_accuracy"]), np.array(vote["tree_log_loss"])
        self.assertEqual(len(acc), self.ctx.spec["n_estimators"])
        self.assertEqual(vote["trees_beating_forest_accuracy"], int((acc > vote["forest"]["accuracy"] + 5e-5).sum()))
        self.assertEqual(vote["trees_beating_forest_log_loss"], int((ll < vote["forest"]["log_loss"] - 5e-5).sum()))
        self.assertGreater(np.median(ll), vote["forest"]["log_loss"])     # the page's "averaging beats one tree" claim
        self.assertEqual(vote["tree_predictions"], len(season_rows) * self.ctx.spec["n_estimators"])

    # ------------------------------------------------------------------ C. split + metric primer
    def test_split_sizes_follow_time_order(self):
        rows, split = self.ctx.rows, self.p["split"]
        self.assertEqual(split["seasons"], sorted(rows.season.unique()))
        for fold in split["folds"]:
            start = rows[rows.season.eq(fold["season"])].date.min()
            self.assertEqual(fold["train_rows"], int((rows.date < start).sum()))
            self.assertEqual(fold["val_rows"], int(rows.season.eq(fold["season"]).sum()))
            self.assertLess(fold["train_last_season"], fold["season"])
            self.assertLess(fold["season"], split["test_start"])
        self.assertEqual([f["season"] for f in split["folds"]], self.ctx.report["protocol"]["folds"])
        for test in split["tests"]:
            start = rows[rows.season.eq(test["season"])].date.min()
            self.assertEqual(test["train_rows"], int((rows.date < start).sum()))
        self.assertEqual(sum(t["rows"] for t in split["tests"]), len(self.ctx.test_pred))
        self.assertEqual(split["tests"][0]["train_rows"], int(self.ctx.train_mask.sum()))

    def test_test_seasons_refit_and_in_progress_flag(self):
        """The holdout card: each test season's model trains on every earlier season, earlier test seasons included;
        only the latest, unfinished season is flagged as in progress."""
        rows, tests = self.ctx.rows, self.p["split"]["tests"]
        for k, test in enumerate(tests):
            season_rows = rows[rows.season.eq(test["season"])]
            self.assertEqual(test["rows"], len(season_rows))
            self.assertEqual(test["last_date"], str(season_rows.date.max().date()))
            # earlier test seasons are part of this season's training rows
            self.assertEqual(test["train_rows"], tests[0]["train_rows"] + sum(t["rows"] for t in tests[:k]))
        self.assertGreater(len(tests), 1)
        self.assertEqual([t["in_progress"] for t in tests], [False] * (len(tests) - 1) + [True])
        self.assertEqual(tests[-1]["last_date"], str(rows.date.max().date()))
        self.assertLess(tests[-1]["rows"], tests[0]["rows"] / 2)

    def test_metric_examples(self):
        m = self.p["metrics"]
        self.assertAlmostEqual(m["uniform_log_loss"], math.log(3), places=4)
        test = self.ctx.test_pred
        for ex in m["examples"]:
            row = test[test.match_id.eq(ex["match_id"]) & test.team.eq(ex["team"])]
            p = row[[f"selected_with_formation_{c}" for c in self.PROBS]].to_numpy()[0]
            y = int(row.outcome.iloc[0])
            self.assertEqual(ex["outcome"], y)
            self.assertAlmostEqual(ex["p_actual"], p[y], delta=ROUND4)
            self.assertAlmostEqual(ex["log_loss"], -math.log(p[y]), delta=ROUND4)
            self.assertAlmostEqual(ex["brier"], float(((p - np.eye(3)[y]) ** 2).sum()), delta=ROUND4)
        self.assertEqual([ex["outcome"] for ex in m["examples"]], [2, 1])
        self.assertAlmostEqual(m["model_geo_p"], math.exp(-self.ctx.report["test"]["selected_with_formation"]["log_loss"]), delta=ROUND4)

    # ------------------------------------------------------------------ D. tuning
    def test_trials_and_selection(self):
        t = self.p["tuning"]
        for key, trials, sel in [("with", self.ctx.trials_with, "with_formation"), ("without", self.ctx.trials_without, "without_formation")]:
            variant = t["variants"][key]
            self.assertEqual(len(variant["trials"]), len(trials))
            rf = [x for x in trials if x["spec"]["family"] == "rf"]
            self.assertEqual(len(rf), self.ctx.report["protocol"]["rf_trials_per_variant"])
            best = min(trials, key=lambda x: x["validation"]["log_loss"])
            self.assertEqual(variant["best_trial"], best["trial"])
            self.assertEqual(variant["best_trial"], self.ctx.selection[sel]["best"]["trial"])
            for hp in variant["hyper"]:
                for value in hp["values"]:
                    losses = [x["validation"]["log_loss"] for x in rf if x["spec"][hp["name"]] == value["value"]]
                    self.assertEqual(value["n"], len(losses))
                    if losses:
                        self.assertAlmostEqual(value["median"], float(np.median(losses)), delta=ROUND4)
                best_value = next(v["value"] for v in hp["values"] if v["key"] == hp["best"])
                mine = [x["validation"]["log_loss"] for x in rf if x["spec"][hp["name"]] == best_value]
                others = [x["validation"]["log_loss"] for x in rf if x["spec"][hp["name"]] != best_value]
                # "separated" = every trial with the best value beat every trial without it (the page leans on this)
                self.assertEqual(hp["separated"], max(mine) < min(others))
            cw = variant["class_weight"]
            ranked = sorted(rf, key=lambda x: x["validation"]["log_loss"])
            self.assertTrue(all(x["spec"]["class_weight"] is None for x in ranked[:cw["leading_off"]]))
            if cw["leading_off"] < len(ranked):
                self.assertIsNotNone(ranked[cw["leading_off"]]["spec"]["class_weight"])
            self.assertEqual(cw["off"]["n"] + cw["on"]["n"], len(rf))
        best_spec = self.ctx.selection["with_formation"]["best"]["spec"]
        self.assertEqual(t["chosen"], {k: best_spec[k] for k in t["order"]})
        self.assertEqual(set(t["order"]), set(self.ctx.report["protocol"]["search_space"]))
        self.assertEqual(t["space"], {k: self.ctx.report["protocol"]["search_space"][k] for k in t["order"]})

    def test_variant_column_counts(self):
        """Tooltips turn 'max_features' shares into column counts per variant: the without-formation matrix is
        exactly the with-formation one minus the formation one-hot columns."""
        t = self.p["tuning"]
        with_cols, without_cols = t["variants"]["with"]["n_columns"], t["variants"]["without"]["n_columns"]
        self.assertEqual(with_cols, len(self.ctx.model.processor_.get_feature_names_out()))
        self.assertEqual(with_cols, t["n_columns"])
        self.assertEqual(with_cols - without_cols, self.p["importance"]["formation_columns"])
        self.assertLess(without_cols, with_cols)

    def test_formation_pairs(self):
        pairs, summary = self.p["tuning"]["pairs"], self.p["tuning"]["pairs_summary"]
        diffs = np.array([b["validation"]["log_loss"] - a["validation"]["log_loss"] for a, b in zip(self.ctx.trials_with, self.ctx.trials_without)])
        self.assertEqual(len(pairs), len(diffs))
        tie = self.mod.PAIR_TIE
        self.assertEqual(summary["with_better"], int((diffs > tie).sum()))
        self.assertEqual(summary["without_better"], int((diffs < -tie).sum()))
        self.assertEqual(summary["with_better"] + summary["without_better"] + summary["ties"], len(diffs))
        self.assertAlmostEqual(summary["max_abs_gap"], float(np.abs(diffs).max()), delta=ROUND4)
        # "formation adds no measurable value": neither side wins most pairs and every gap is small
        self.assertLess(abs(summary["with_better"] - summary["without_better"]), len(diffs) / 4)
        self.assertLess(summary["max_abs_gap"], 0.02)

    def test_forward_validation_reproduces_an_official_trial(self):
        """The routine used for the untuned reference reproduces the pipeline's own tune() numbers."""
        rf = [x for x in self.ctx.trials_with if x["spec"]["family"] == "rf"]
        cheapest = min(rf, key=lambda x: x["spec"]["n_estimators"] * (x["spec"]["max_depth"] or 40))
        folds, total = self.mod.forward_validation(self.ctx, cheapest["spec"])
        self.assertAlmostEqual(total["log_loss"], cheapest["validation"]["log_loss"], places=9)
        for mine, theirs in zip(folds, cheapest["folds"]):
            self.assertAlmostEqual(mine["log_loss"], theirs["log_loss"], places=9)
        untuned = self.p["tuning"]["untuned"]
        self.assertEqual(len(untuned["folds"]), len(self.ctx.report["protocol"]["folds"]))
        self.assertTrue(0.6 < untuned["log_loss"] < 1.1)

    def test_overfit_demo_uses_the_official_fold(self):
        demo = self.p["tuning"]["overfit"]
        best = self.ctx.selection["with_formation"]["best"]
        last = next(f for f in best["folds"] if f["season"] == demo["season"])
        small, chosen = demo["runs"]
        self.assertEqual(chosen["min_samples_leaf"], self.ctx.spec["min_samples_leaf"])
        self.assertAlmostEqual(chosen["val_log_loss"], last["log_loss"], delta=ROUND4)
        self.assertEqual(demo["train_rows"], last["train_team_rows"])
        # the page's sentence: tiny leaves fit the training games better but the unseen season worse
        self.assertGreater(small["train_accuracy"], chosen["train_accuracy"])
        self.assertGreater(small["val_log_loss"], chosen["val_log_loss"])

    # ------------------------------------------------------------------ E. results
    def test_scoreboard_and_comparisons(self):
        res, report = self.p["results"], self.ctx.report
        for row in res["scoreboard"]:
            self.assertAlmostEqual(row["accuracy"], report["test"][row["key"]]["accuracy"], delta=ROUND4)
            self.assertAlmostEqual(row["brier"], report["test"][row["key"]]["brier"], delta=ROUND4)
            self.assertEqual(row["rows"], len(self.ctx.test_pred))
            if row["key"] == "always_team_win":
                self.assertIsNone(row["log_loss"])            # a 0/1 forecast has no usable probabilities
            else:
                self.assertAlmostEqual(row["log_loss"], report["test"][row["key"]]["log_loss"], delta=ROUND4)
        for item in res["comparisons"]:
            for metric in ["accuracy", "log_loss"]:
                src = report["comparisons"][item["key"]][metric]
                self.assertAlmostEqual(item[metric]["value"], src["improvement"], delta=1e-5)
                self.assertAlmostEqual(item[metric]["low"], src["ci95_low"], delta=1e-5)
                self.assertAlmostEqual(item[metric]["high"], src["ci95_high"], delta=1e-5)
                self.assertEqual(item[metric]["includes_zero"], src["ci95_low"] <= 0 <= src["ci95_high"])
        by_key = {c["key"]: c for c in res["comparisons"]}
        self.assertFalse(by_key["selected_vs_always_win"]["log_loss"]["usable"])
        # honesty rules the page states in plain words
        self.assertTrue(by_key["formation_gain"]["log_loss"]["includes_zero"])
        self.assertTrue(by_key["formation_gain"]["accuracy"]["includes_zero"])
        self.assertFalse(by_key["selected_vs_always_win"]["accuracy"]["includes_zero"])
        self.assertTrue(by_key["selected_vs_elo"]["accuracy"]["includes_zero"])
        self.assertTrue(by_key["selected_vs_elo"]["log_loss"]["includes_zero"])

    def test_confusion_and_calibration_from_official_predictions(self):
        test = self.ctx.test_pred
        for key, block in self.p["results"]["confusion"].items():
            proba = test[[f"{key}_{c}" for c in self.PROBS]].to_numpy()
            table = pd.crosstab(test.outcome, pd.Series(proba.argmax(1), index=test.index)).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            np.testing.assert_array_equal(np.array(block["matrix"]), table.to_numpy())
            self.assertEqual(sum(map(sum, block["matrix"])), len(test))
            draw = block["per_class"][1]
            self.assertEqual(draw["predicted"], int((proba.argmax(1) == 1).sum()))
            self.assertEqual(draw["actual"], int(test.outcome.eq(1).sum()))
        # "draws are almost never predicted"
        self.assertLessEqual(self.p["results"]["confusion"]["selected_with_formation"]["per_class"][1]["predicted"], 0.02 * len(test))
        for key, bins in self.p["results"]["calibration"].items():
            self.assertEqual(sum(b["n"] for b in bins), len(test))
            self.assertEqual(sum(b["wins"] for b in bins), int(test.outcome.eq(2).sum()))

    def test_calibration_sentence_is_like_for_like(self):
        """Mean P(win) and the actual win share in the calibration sentence both count every club row once."""
        test, train = self.ctx.test_pred, self.ctx.train
        wr = self.p["results"]["win_rates"]
        mean_p = wr["mean_p_win"]["selected_with_formation"]
        self.assertAlmostEqual(mean_p, float(test["selected_with_formation_p_win"].mean()), delta=ROUND4)
        self.assertAlmostEqual(wr["test_rows_share"], float(test.outcome.eq(2).mean()), delta=ROUND4)
        self.assertAlmostEqual(wr["train_rows_share"], float(train.outcome.eq(2).mean()), delta=ROUND4)
        self.assertEqual(wr["test_rows"], len(test))
        # same rows as the bins: their n-weighted mean prediction is the same mean
        bins = self.p["results"]["calibration"]["selected_with_formation"]
        self.assertAlmostEqual(sum(b["mean_p"] * b["n"] for b in bins) / len(test), mean_p, delta=1e-3)
        self.assertAlmostEqual(sum(b["wins"] for b in bins) / len(test), wr["test_rows_share"], delta=ROUND4)
        # the page's reading: the model over-predicts wins, and the training rows won more often than the test rows
        self.assertGreater(mean_p, wr["test_rows_share"])
        self.assertGreater(wr["train_rows_share"], wr["test_rows_share"])

    def test_validation_vs_test_gap_is_pooled(self):
        """The gap card's test margin over 'always win' equals the official comparison (+4.4%p), not a per-season mean."""
        gap, report = self.p["results"]["gap"], self.ctx.report
        self.assertAlmostEqual(gap["test_model"], report["test"]["selected_with_formation"]["accuracy"], delta=ROUND4)
        self.assertAlmostEqual(gap["test_always"], report["test"]["always_team_win"]["accuracy"], delta=ROUND4)
        margin = report["comparisons"]["selected_vs_always_win"]["accuracy"]["improvement"]
        self.assertAlmostEqual(gap["test_model"] - gap["test_always"], margin, delta=2e-4)
        self.assertAlmostEqual(gap["val_model"], self.ctx.selection["with_formation"]["best"]["validation"]["accuracy"], delta=ROUND4)
        rows = self.ctx.rows
        val = rows[rows.season.isin(report["protocol"]["folds"])]
        self.assertAlmostEqual(gap["val_always"], float(np.average(val.outcome.eq(2), weights=val.fixture_weight)), delta=ROUND4)
        self.assertLess(gap["test_always"], gap["val_always"])          # "the strong clubs won less in the test seasons"

    def test_season_and_club_breakdowns(self):
        res, report = self.p["results"], self.ctx.report
        self.assertEqual([d["season"] for d in res["by_season"]], list(report["test_by_season"]))
        self.assertEqual(sum(d["selected_with_formation"]["rows"] for d in res["by_season"]), len(self.ctx.test_pred))
        self.assertEqual(sum(d["selected_with_formation"]["rows"] for d in res["by_team"]), len(self.ctx.test_pred))
        rows = self.ctx.rows
        for item in res["win_share"]:
            season = rows[rows.season.eq(item["season"])]
            self.assertAlmostEqual(item["always_win"], np.average(season.outcome.eq(2), weights=season.fixture_weight), delta=ROUND4)

    # ------------------------------------------------------------------ F. importance
    def test_importance_labels_and_groups(self):
        from epl_tactics.lab.catalog import describe
        imp = self.p["importance"]
        self.assertEqual(len(imp["permutation"]), len(self.ctx.report["validation_top_features"]))
        for item, src in zip(imp["permutation"], self.ctx.report["validation_top_features"]):
            self.assertEqual(item["feature"], src["feature"])
            # kept at 6 decimals so the page's 4-digit display is a single rounding of the source value
            self.assertAlmostEqual(item["importance"], src["validation_log_loss_importance"], delta=1e-6)
            self.assertEqual(f"{item['importance']:.4f}", f"{src['validation_log_loss_importance']:.4f}")
            if item["feature"].startswith("numeric__") and "missingindicator" not in item["feature"]:
                self.assertEqual(item["label"], describe(item["feature"][len("numeric__"):])["label"])
        self.assertEqual(imp["permutation_formation"], sum(i["group"] == "formation" for i in imp["permutation"]))
        shares = {g["group"]: g["share"] for g in imp["impurity_groups"]}
        self.assertAlmostEqual(sum(shares.values()), 1.0, delta=2e-3)
        self.assertEqual(sum(g["columns"] for g in imp["impurity_groups"]), imp["n_columns"])
        self.assertEqual(imp["n_columns"], len(self.ctx.model.processor_.get_feature_names_out()))
        self.assertLess(shares["formation"], shares["elo"])          # "formation inputs carry little"


if __name__ == "__main__":
    unittest.main()
