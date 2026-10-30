"""Step 1 (데이터) payload: every number on the page must agree with the source tables."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd

DATA = Path("data/focused")
MODEL = Path("artifacts/focused")
FRAGMENT = Path("epl_tactics/lab/s1.html")
HANGUL = re.compile(r"[가-힣]")


class LabStepOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DATA / "matches.csv").exists() or not (MODEL / "metrics.json").exists():
            raise unittest.SkipTest("focused data/artifacts not built")
        from epl_tactics.lab.context import load_context
        from epl_tactics.lab import export_s1
        from epl_tactics.lab.catalog import build_catalog
        from epl_tactics.lab.report import sanitize
        cls.m = export_s1
        cls.ctx = load_context(DATA, MODEL, fit=False)
        cls.catalog = build_catalog(cls.ctx.numeric)
        cls.payload = sanitize(export_s1.build(cls.ctx))
        # Independent reads of the source files (not via the context) for cross-checks.
        cls.matches = pd.read_csv(DATA / "matches.csv")
        cls.league = pd.read_csv(DATA / "league_context.csv")
        cls.coverage = pd.read_csv(DATA / "coverage.csv")
        cls.report = json.loads((MODEL / "metrics.json").read_text(encoding="utf-8"))
        cls.clubs = list(cls.report["protocol"]["clubs"])

    # ------------------------------------------------------------ labels / dictionaries
    def test_every_raw_column_has_a_korean_label(self):
        dictionary = self.m.raw_dictionary()
        self.assertEqual(set(self.matches.columns), set(dictionary), "raw dictionary must match matches.csv exactly")
        groups = {key for key, _, _ in self.m.RAW_GROUPS}
        columns = self.payload["raw"]["columns"]
        self.assertEqual([c["name"] for c in columns], list(self.matches.columns))
        for column in columns:
            self.assertTrue(HANGUL.search(column["label"]), column["name"])
            self.assertIn(column["group"], groups)
            self.assertIn(column["source"], self.m.SOURCE_LABEL)
            self.assertTrue(column["desc"])
        counts = {g["key"]: g["count"] for g in self.payload["raw"]["groups"]}
        self.assertEqual(sum(counts.values()), self.matches.shape[1])

    def test_raw_missing_and_zero_shares(self):
        by_name = {c["name"]: c for c in self.payload["raw"]["columns"]}
        for name in ["home_formation", "away_possession", "home_shots"]:
            self.assertAlmostEqual(by_name[name]["missing"], self.matches[name].isna().mean(), places=4)
        self.assertAlmostEqual(by_name["home_possession"]["zero"], self.matches.home_possession.eq(0).mean(), places=4)
        self.assertIsNone(by_name["event_id"]["zero"])          # identifiers have no meaningful zero share
        self.assertEqual(self.payload["raw"]["n_rows"], len(self.matches))
        self.assertEqual(self.payload["raw"]["n_cols"], self.matches.shape[1])

    def test_every_model_column_is_an_input_or_labelled_extra(self):
        rows = self.ctx.rows
        inputs = {e["name"] for e in self.catalog}
        extras = set(self.m.MODEL_EXTRA)
        self.assertFalse(inputs & extras)
        self.assertEqual(set(rows.columns), inputs | extras)
        model = self.payload["model"]
        self.assertEqual(model["n_cols"], rows.shape[1])
        self.assertEqual(model["roles"]["input_numeric"] + model["roles"]["input_category"] + model["roles"]["target"]
                         + len(model["roles"]["other"]), rows.shape[1])
        self.assertEqual(sum(model["dtypes"].values()), rows.shape[1])
        self.assertTrue(set(model["key_columns"]) <= set(rows.columns))

    # ------------------------------------------------------------ counts and the 7,680 -> 4,003 -> 4,608 flow
    def test_kpis_match_sources(self):
        k = self.payload["kpi"]
        self.assertEqual(k["unique_matches"], len(self.matches))
        self.assertEqual(k["unique_matches"], self.report["data"]["unique_matches"])
        self.assertEqual(k["rows"], len(self.ctx.rows))
        self.assertEqual(k["inputs"], k["numeric"] + k["categories"])
        self.assertEqual(k["numeric"], len(self.ctx.numeric))
        self.assertEqual(k["first_date"], str(pd.to_datetime(self.matches.date).min().date()))
        self.assertEqual(k["last_date"], str(pd.to_datetime(self.matches.date).max().date()))
        self.assertEqual(k["n_seasons"], self.matches.season.nunique())

    def test_train_and_test_rows_split_the_model_table(self):
        # The page says "learn from the rows before the first test season; the rest are unseen test rows".
        k, rows = self.payload["kpi"], self.ctx.rows
        test_start = self.report["protocol"]["test_start"]
        self.assertEqual(k["test_start"], test_start)
        self.assertEqual(k["train_rows"], int(rows.season.lt(test_start).sum()))
        self.assertEqual(k["train_rows"], int(np.asarray(self.ctx.train_mask).sum()))
        self.assertEqual(k["test_rows"], int(rows.season.ge(test_start).sum()))
        self.assertEqual(k["test_rows"], self.report["test"]["always_team_win"]["team_appearances"])
        self.assertEqual(k["train_rows"] + k["test_rows"], k["rows"])
        self.assertLess(k["train_rows"], k["rows"])
        self.assertEqual(k["refit_each_test_season"], self.report["protocol"]["test_refit"].lower().startswith("before each test season"))

    def test_flow_explains_match_and_row_counts(self):
        flow = self.payload["flow"]
        clubs = self.clubs
        involving = self.league.home_team.isin(clubs) | self.league.away_team.isin(clubs)
        both = self.league.home_team.isin(clubs) & self.league.away_team.isin(clubs)
        self.assertEqual(flow["league"], len(self.league))
        self.assertEqual(flow["league"], self.report["data"]["league_context_matches"])
        self.assertEqual(flow["involving"], int(involving.sum()))
        self.assertEqual(flow["involving"], len(self.matches))
        self.assertEqual(flow["two_club"], int(both.sum()))
        self.assertEqual(flow["other"] + flow["involving"], flow["league"])
        self.assertEqual(flow["one_club"] + 2 * flow["two_club"], flow["rows"])
        self.assertEqual(flow["per_club"] * len(clubs), flow["rows"])
        rows = self.ctx.rows
        sizes = rows.groupby("match_id").size()
        self.assertEqual(int(sizes.eq(2).sum()), flow["two_club"])
        self.assertAlmostEqual(flow["weight_sum"], flow["involving"])
        self.assertTrue(rows[rows.match_id.map(sizes).eq(2)].fixture_weight.eq(flow["derby_weight"]).all())
        self.assertEqual(flow["derby_weight"], 0.5)

    def test_sources_and_join_verification(self):
        src = self.payload["sources"]
        self.assertEqual(src["espn"]["verified"], int(self.matches.espn_verified.sum()))
        self.assertEqual(src["espn"]["unmatched"], int((~self.matches.espn_verified).sum()))
        mismatches = DATA / "source_mismatches.json"
        if mismatches.exists():
            self.assertEqual(src["espn"]["rejected"], len(json.loads(mismatches.read_text(encoding="utf-8"))))
        self.assertEqual(sum(o["matches"] for o in src["fd"]["odds"]), len(self.matches))
        self.assertEqual(src["fd"]["seasons"], self.league.season.nunique())

    def test_zero_coded_stats_are_counted_from_both_files(self):
        q = self.payload["sources"]["quality"]
        merged = self.matches.set_index("match_id")
        fd = self.league.set_index("match_id").reindex(merged.index)
        overwritten = ((merged.home_shots.eq(0) & fd.home_shots.gt(0)) | (merged.away_shots.eq(0) & fd.away_shots.gt(0)))
        self.assertEqual(q["shots_zero_overwritten"], int(overwritten.sum()))
        both_zero = self.matches.home_possession.eq(0) & self.matches.away_possession.eq(0)
        self.assertEqual(q["possession_both_zero"], int(both_zero.sum()))
        top = self.matches[both_zero].season.value_counts()
        self.assertEqual(q["possession_zero_top_season"], top.index[0])
        self.assertEqual(q["possession_zero_top_count"], int(top.iloc[0]))
        # The page says the pipeline uses those zeros as real values: a 5-match average possession of exactly 0%
        # is only possible if the zero-coded matches were averaged in.
        self.assertTrue(self.ctx.rows.team_possession_5.eq(0).any())
        # "Same team, both sources non-zero, counts still differ" is counted per side, as the page words it.
        differ, gaps = pd.Series(False, index=merged.index), []
        for side in ["home", "away"]:
            a, b = merged[f"{side}_shots"], fd[f"{side}_shots"]
            hit = a.gt(0) & b.gt(0) & a.ne(b)
            differ |= hit
            gaps.append((a - b).abs()[hit])
        self.assertEqual(q["shots_real_differ"], int(differ.sum()))
        self.assertAlmostEqual(q["shots_real_differ_median_gap"], float(pd.concat(gaps).median()))

    def test_stats_never_recorded_in_some_seasons_are_found(self):
        # Re-derive the "stored as 0 although not recorded" seasons from matches.csv alone.
        q = self.payload["sources"]["quality"]
        m = self.matches
        box = m.home_possession.gt(0) | m.away_possession.gt(0)
        found = {item["stat"]: item for item in q["structural"]}
        self.assertNotIn("red", found)                     # cards are genuinely 0 most of the time: not scanned
        self.assertNotIn("yellow", found)
        for stat in self.m.ZERO_SCAN:
            both0 = m[f"home_{stat}"].eq(0) & m[f"away_{stat}"].eq(0)
            judged = box & m[f"home_{stat}"].notna() & m[f"away_{stat}"].notna()
            almost, partial = [], {}
            for season, part in judged.groupby(m.season):
                if part.sum() < q["min_box"]:
                    continue
                share = both0[part.index[part]].mean()
                if share >= q["structural_share"]:
                    almost.append(season)
                elif share >= q["partial_share"]:
                    partial[season] = share
            if not almost and not partial:
                self.assertNotIn(stat, found)
                continue
            item = found[stat]
            self.assertEqual(item["almost"], almost, stat)
            self.assertEqual({p["season"]: p["share"] for p in item["partial"]}, {k: round(v, 4) for k, v in partial.items()})
            self.assertAlmostEqual(item["share_all"], both0.mean(), places=4)
            for name, share in item["inputs"].items():
                self.assertAlmostEqual(share, self.ctx.rows[name].eq(0).mean(), places=4)
        # The worst one is interceptions, which ESPN effectively did not record before 2018/19; its 5-match average
        # is 0 in more than half of the model rows (the page states both).
        worst = q["structural"][0]
        self.assertEqual(worst["stat"], "interceptions")
        self.assertEqual(worst["share_all"], max(item["share_all"] for item in q["structural"]))
        self.assertGreater(worst["rows_zero_5"], .5)
        self.assertIn("2013/2014", worst["almost"])         # the example derby's season: its 0 / 0 is a stored zero
        self.assertEqual(worst["almost_ranges"][0][0], min(worst["almost"]))
        self.assertEqual(worst["almost_ranges"][-1][1], max(worst["almost"]))
        # A printed range "A~B" claims every season from A to B: it must never bridge a season that was not
        # flagged (e.g. 2016/17, which could not be judged because its box scores are almost all empty).
        seasons = sorted(m.season.unique())
        for item in q["structural"]:
            covered = [s for a, b in item["almost_ranges"] for s in seasons[seasons.index(a):seasons.index(b) + 1]]
            self.assertEqual(covered, item["almost"], item["stat"])
        self.assertEqual(q["unjudged"], [u["season"] for u in q["unjudged_detail"]])
        possession_zero = m.home_possession.eq(0) & m.away_possession.eq(0)
        for u in q["unjudged_detail"]:
            in_season = m.season.eq(u["season"])
            self.assertLess(int(box[in_season].sum()), q["min_box"])
            self.assertEqual(u["matches"], int(in_season.sum()))
            self.assertEqual(u["box_zero"], int((possession_zero & in_season).sum()))
            self.assertGreater(u["box_zero"], u["matches"] / 2)   # the page says "most matches" are whole-box zeros
        # The coverage ("record availability") inputs count the stored zeros as recorded.
        rows = self.ctx.rows
        zero20 = rows.team_interceptions_20.eq(0)
        self.assertTrue(zero20.any())
        self.assertTrue(rows.loc[zero20, "team_interceptions_coverage20"].gt(0).all())
        self.assertTrue(q["worst_coverage_counts_zero"])

    # ------------------------------------------------------------ coverage
    def test_coverage_grid_matches_rows(self):
        cov = self.payload["coverage"]
        rows = self.ctx.rows
        self.assertEqual(sum(map(sum, cov["formation"])), int(rows.formation_known.sum()))
        self.assertEqual(sum(map(sum, cov["matches"])), len(rows))
        for ci, club in enumerate(self.clubs):
            for si, season in enumerate(cov["seasons"]):
                part = rows[rows.team.eq(club) & rows.season.eq(season)]
                self.assertEqual(cov["matches"][ci][si], len(part))
                self.assertEqual(cov["formation"][ci][si], int(part.formation_known.sum()))
        eleven = 0
        for record in self.matches.to_dict("records"):
            for side in ["home", "away"]:
                if record[f"{side}_team"] in self.clubs:
                    players = json.loads(record[f"{side}_players"]) if isinstance(record[f"{side}_players"], str) else []
                    eleven += sum(bool(p.get("starter")) for p in players) == 11
        self.assertEqual(sum(map(sum, cov["eleven"])), eleven)
        for season in cov["gap_seasons"]:
            self.assertEqual(int(rows[rows.season.eq(season)].formation_known.sum()), 0)
        for item in cov["partial_seasons"]:
            self.assertAlmostEqual(item["share"], rows[rows.season.eq(item["season"])].formation_known.mean(), places=4)
        for item in cov["short_seasons"]:
            self.assertTrue(rows[rows.season.eq(item["season"])].groupby("team").size().eq(item["matches"]).all())
        self.assertAlmostEqual(cov["formation_share"], rows.formation_known.mean(), places=4)

    # ------------------------------------------------------------ target
    def test_outcome_distribution(self):
        oc = self.payload["outcome"]
        rows = self.ctx.rows
        for club in self.clubs:
            for venue, flag in [("home", 1), ("away", 0)]:
                part = rows[rows.team.eq(club) & rows.is_home.eq(flag)]
                self.assertEqual(oc["by_club"][club][venue], [int(part.outcome.eq(k).sum()) for k in [0, 1, 2]])
        self.assertAlmostEqual(oc["row_share"][2], rows.outcome.eq(2).mean(), places=4)
        self.assertAlmostEqual(sum(oc["row_share"]), 1, places=3)
        self.assertGreater(oc["row_share"][2], .5)           # the page says "절반 넘게" only in that case (JS checks too)
        self.assertGreater(oc["home_win"], oc["away_win"])
        self.assertAlmostEqual(oc["always_win_test"], self.report["test"]["always_team_win"]["accuracy"], places=4)
        # "Always the club wins" accuracy = fixture-weighted win share. Over all seasons it is above one half, on the
        # test seasons it is lower (the page words both sentences from these two numbers).
        test = rows[rows.season.ge(self.report["protocol"]["test_start"])]
        self.assertAlmostEqual(oc["always_win_test"], np.average(test.outcome.eq(2), weights=test.fixture_weight), places=4)
        self.assertAlmostEqual(oc["weighted_share"][2], np.average(rows.outcome.eq(2), weights=rows.fixture_weight), places=4)
        self.assertGreater(oc["weighted_share"][2], .5)
        self.assertLess(oc["always_win_test"], oc["weighted_share"][2])
        # Outcome is from the row's club perspective: it agrees with the goals of that row.
        expected = np.select([rows.goals_for < rows.goals_against, rows.goals_for == rows.goals_against], [0, 1], 2)
        self.assertTrue((rows.outcome.to_numpy() == expected).all())

    # ------------------------------------------------------------ inputs and leakage
    def test_inputs_counts_groups_and_stats(self):
        inp = self.payload["inputs"]
        self.assertEqual(inp["n_total"], len(self.catalog))
        self.assertEqual(inp["n_total"], inp["n_numeric"] + inp["n_category"])
        self.assertEqual(sum(inp["group_counts"].values()), inp["n_total"])
        self.assertEqual(set(inp["stats"]), {e["name"] for e in self.catalog})
        rows = self.ctx.rows
        for name in ["team_elo", "opponent_lineup_returning", "formation_used_last20"]:
            self.assertAlmostEqual(inp["stats"][name]["miss"], rows[name].isna().mean(), places=4)
            self.assertAlmostEqual(inp["stats"][name]["q50"], rows[name].median(), places=3)
        self.assertAlmostEqual(inp["stats"]["formation"]["miss"], rows.formation.eq("__missing__").mean(), places=4)
        self.assertEqual(inp["stats"]["team"]["n"], len(self.clubs))

    def test_window_average_share(self):
        # "N% of the inputs are 5/10/20-match averages (or a difference of two averages)": the *_coverage20 shares
        # in the same groups are not averages and must not be counted. Re-derived from the input names alone.
        inp = self.payload["inputs"]
        names = [e["name"] for e in self.catalog]
        averages = [n for n in names if re.fullmatch(r"(team|opponent|diff)_[a-z_]+_(5|10|20)", n)]
        coverage = [n for n in names if n.endswith("_coverage20")]
        self.assertEqual(inp["window_averages"], len(averages))
        self.assertEqual(inp["window_coverage"], len(coverage))
        self.assertEqual(inp["window_inputs"], sum(inp["group_counts"][k] for k in ["form5", "form10", "form20"]))
        self.assertEqual(inp["window_averages"] + inp["window_coverage"], inp["window_inputs"])
        self.assertLess(inp["window_averages"] / inp["n_total"], .6)

    def test_leakage_rule(self):
        inp = self.payload["inputs"]
        inputs = {e["name"] for e in self.catalog}
        for forbidden in ["outcome", "goals_for", "goals_against", "market_win", "market_draw", "market_loss", "fixture_weight"]:
            self.assertNotIn(forbidden, inputs)
            self.assertIn(forbidden, inp["not_inputs"])
        same_match_raw = {c for c in self.matches.columns if c.startswith(("home_", "away_"))}
        self.assertFalse(inputs & same_match_raw)
        self.assertEqual([l["key"] for l in inp["leakage"]], self.ctx.feature_meta["same_match_excluded"])
        example = inp["leak_example"]
        raw = self.matches[self.matches.match_id.eq(self.payload["raw"]["example"]["match_id"])].iloc[0]
        row = self.ctx.rows[self.ctx.rows.match_id.eq(raw.match_id) & self.ctx.rows.is_home.eq(1)].iloc[0]
        for item in example["items"]:
            self.assertAlmostEqual(item["same"], raw[f"home_{item['stem']}"], places=3)
            self.assertAlmostEqual(item["pre"], row[item["input"]], places=3)
            self.assertIn(item["input"], inputs)

    # ------------------------------------------------------------ the worked example (one derby, two rows)
    def test_example_is_a_complete_derby_shown_as_two_mirrored_rows(self):
        ex = self.payload["raw"]["example"]
        self.assertIn(ex["home_team"], self.clubs)
        self.assertIn(ex["away_team"], self.clubs)
        raw = self.matches[self.matches.match_id.eq(ex["match_id"])].iloc[0]
        self.assertEqual((raw.home_goals, raw.away_goals), (ex["home_goals"], ex["away_goals"]))
        self.assertTrue(pd.notna(raw.home_formation) and pd.notna(raw.away_formation))
        self.assertEqual(ex["values"]["home_players"]["starters"], 11)
        home, away = self.payload["model"]["example"]
        self.assertEqual((home["team"], away["team"]), (ex["home_team"], ex["away_team"]))
        self.assertEqual((home["is_home"], away["is_home"]), (1, 0))
        self.assertEqual(home["opponent"], away["team"])
        self.assertEqual(home["outcome"] + away["outcome"], 2)
        self.assertEqual(home["fixture_weight"], .5)
        self.assertEqual(away["fixture_weight"], .5)
        self.assertAlmostEqual(home["diff_elo"], -away["diff_elo"], places=4)
        self.assertEqual(home["goals_for"], away["goals_against"])
        self.assertEqual(set(home), set(self.ctx.rows.columns))

    def test_pair_relations_describe_the_two_rows(self):
        model = self.payload["model"]
        home, away = model["example"]
        rel = model["relation"]
        self.assertEqual(set(rel), set(model["columns"]))
        self.assertEqual((rel["team"], rel["opponent"], rel["is_home"]), ("swap", "swap", "home_flip"))
        self.assertEqual(rel["diff_elo"], "neg")
        self.assertEqual(rel["match_id"], "same")
        # Each row holds its OWN club's formation: never labelled as a value shared by the match.
        for name in ["formation", "team_formation", "previous_formation", "defenders", "formation_known"]:
            self.assertIn(rel[name], {"own", "own_same"}, name)
            self.assertEqual(rel[name] == "own_same", home[name] == away[name])
        for name, code in rel.items():
            if code in {"swap", "swap_same"}:
                partner = self.m.swap_partner(name, model["columns"])
                self.assertEqual((home[name], away[name]), (away[partner], home[partner]), name)
                self.assertEqual(code == "swap_same", home[name] == away[name], name)
            if code == "same":
                self.assertEqual(home[name], away[name])
                self.assertTrue(name in self.m.MATCH_LEVEL, name)   # only match-level columns are "the same"
        flagged = {item["stat"] for item in self.payload["sources"]["quality"]["structural"]}
        for name in model["zero_coded"]:
            self.assertIn(re.sub(r"^(team|opponent)_|_(5|10|20)$", "", name), flagged)
            self.assertIn(0, (home[name], away[name]))

    def test_non_input_columns_and_their_uses(self):
        # The page says: outcome is the answer; date and fixture_weight only set training weights; nothing else is used.
        from epl_tactics.focused_model import train_weights
        extra = self.payload["model"]["extra"]
        self.assertEqual(extra["outcome"]["use"], "answer")
        self.assertEqual({n for n, e in extra.items() if e["use"] == "weight"}, {"date", "fixture_weight"})
        rows = self.ctx.rows[np.asarray(self.ctx.train_mask)]
        half_life = self.report["protocol"]["half_life_days"]
        age = (rows.date.max() - rows.date).dt.days.to_numpy()
        expected = rows.fixture_weight.to_numpy() * np.exp(-np.log(2) * age / half_life)
        self.assertTrue(np.allclose(train_weights(rows), expected))
        self.assertEqual(set(self.payload["inputs"]["not_inputs"]), set(extra))

    # ------------------------------------------------------------ payload + fragment hygiene
    def test_payload_is_strict_json_and_within_budget(self):
        text = json.dumps(self.payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.assertLess(len(text.encode("utf-8")), 600_000)

    def test_fragment_has_no_hard_coded_data_numbers(self):
        html = FRAGMENT.read_text(encoding="utf-8")
        self.assertIn("Lab.register('s1'", html)
        flow, kpi = self.payload["flow"], self.payload["kpi"]
        numbers = {flow["league"], flow["involving"], flow["rows"], flow["one_club"], flow["two_club"], flow["other"],
                   flow["per_club"], kpi["inputs"], kpi["numeric"], self.payload["sources"]["espn"]["rejected"]}
        for n in numbers:
            for token in {str(n), f"{n:,}"}:
                self.assertIsNone(re.search(rf"(?<![\d.,]){re.escape(token)}(?![\d.,])", html), token)


if __name__ == "__main__":
    unittest.main()
