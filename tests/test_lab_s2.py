"""Step 2 (preprocessing + distributions) payload: every shown number is checked against the source data."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, MODEL = ROOT / "data" / "focused", ROOT / "artifacts" / "focused"
CTX = PAYLOAD = None


def setUpModule():
    global CTX, PAYLOAD
    if not (DATA / "matches.csv").exists() or not (MODEL / "metrics.json").exists():
        return
    from epl_tactics.lab.context import load_context
    from epl_tactics.lab import export_s2
    from epl_tactics.lab.report import sanitize
    CTX = load_context(DATA, MODEL, fit=False)
    PAYLOAD = sanitize(export_s2.build(CTX))


class LabStep2Test(unittest.TestCase):
    def setUp(self):
        if PAYLOAD is None:
            self.skipTest("focused data/artifacts missing")
        self.p, self.rows = PAYLOAD, CTX.rows

    def col(self, name):
        return np.array([np.nan if v is None else v for v in self.p["cols"][name]], dtype=float)

    def test_payload_is_json_safe_and_within_budget(self):
        text = json.dumps(self.p, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.assertLess(len(text.encode("utf-8")), 2.5e6)
        n = len(self.rows)
        self.assertEqual(self.p["n_rows"], n)
        for key, values in {**self.p["rows"], **self.p["cols"]}.items():
            self.assertEqual(len(values), n, key)
        self.assertGreaterEqual(len(self.p["vars"]), 30)
        names = {c["name"]: c["label"] for c in __import__("epl_tactics.lab.catalog", fromlist=["x"]).build_catalog(CTX.numeric)}
        for var in self.p["vars"]:
            self.assertEqual(var["label"], names[var["name"]])

    def test_row_arrays_match_source(self):
        r, rows = self.p["rows"], self.rows
        dates = pd.Timestamp(self.p["base_date"]) + pd.to_timedelta(r["day"], unit="D")
        self.assertTrue((dates == rows.date.to_numpy()).all())
        self.assertEqual([self.p["clubs"][i] for i in r["club"]], rows.team.tolist())
        self.assertEqual([self.p["opponents"][i] for i in r["opp"]], rows.opponent.tolist())
        self.assertEqual(r["out"], rows.outcome.astype(int).tolist())
        self.assertEqual(r["gf"], rows.goals_for.astype(int).tolist())
        np.testing.assert_allclose(self.col("diff_elo"), rows.diff_elo.to_numpy(), atol=0.006)
        np.testing.assert_allclose(self.col("team_points_5"), rows.team_points_5.to_numpy(), atol=1e-4, equal_nan=True)
        # Training rows are the date prefix that the playground model learned from.
        self.assertEqual(self.p["n_train"], int(CTX.train_mask.sum()))
        self.assertTrue(CTX.train_mask[: self.p["n_train"]].all() and not CTX.train_mask[self.p["n_train"]:].any())

    def test_time_travel_recomputes_rolling_values_from_earlier_rows_only(self):
        """The demo rebuilds a row's rolling means from the club's previous rows; they must equal the real features."""
        r = self.p["rows"]
        day, club, out = np.array(r["day"]), np.array(r["club"]), np.array(r["out"])
        gf, ga, home = np.array(r["gf"]), np.array(r["ga"]), np.array(r["home"])
        points = np.array([0, 1, 3])[out]
        consts = self.p["pipeline"]["constants"]
        checked = 0
        for i in list(range(0, len(day), 17)) + [0, 1, len(day) - 1]:
            prev = np.flatnonzero((club == club[i]) & (day < day[i]))
            for w in consts["windows"]:
                win = prev[-w:]
                for stat, source in [("points", points), ("goals_for", gf), ("goals_against", ga)]:
                    expected = self.rows[f"team_{stat}_{w}"].iloc[i]
                    got = source[win].mean() if len(win) else np.nan
                    if np.isnan(expected):
                        self.assertEqual(len(win), 0)
                    else:
                        self.assertAlmostEqual(got, expected, places=9)
                    if f"team_{stat}_{w}" in self.p["cols"]:
                        shown = self.p["cols"][f"team_{stat}_{w}"][i]
                        self.assertTrue(shown is None if np.isnan(expected) else abs(shown - got) < 1e-4)
            venue = prev[home[prev] == home[i]][-consts["venue_n"]:]
            exp_v = self.rows.team_venue_points.iloc[i]
            self.assertTrue(np.isnan(exp_v) if not len(venue) else abs(points[venue].mean() - exp_v) < 1e-9)
            rest = min(day[i] - day[prev[-1]], consts["rest_cap"]) if len(prev) else consts["rest_default"]
            self.assertEqual(rest, self.rows.team_rest_days.iloc[i])
            # The row's own result is never in its inputs: it only appears in the *next* row's window.
            self.assertNotIn(i, prev)
            checked += 1
        self.assertGreater(checked, 250)

    def test_elo_constants_read_from_code_reproduce_the_feature(self):
        c = self.p["pipeline"]["constants"]
        self.assertEqual((c["elo_start"], c["elo_keep"], c["elo_home_edge"], c["elo_k"]), (1500.0, 0.75, 65.0, 20.0))
        elo, season, pre = {}, {}, {}
        league = CTX.league.sort_values(["date", "match_id"])
        for _, day in league.groupby("date", sort=True):
            updates = []
            for m in day.itertuples():
                for team in (m.home_team, m.away_team):
                    if season.get(team) != m.season:
                        elo[team] = c["elo_start"] + c["elo_keep"]*(elo.get(team, c["elo_start"]) - c["elo_start"])
                        season[team] = m.season
                pre[(m.match_id, m.home_team)], pre[(m.match_id, m.away_team)] = elo[m.home_team], elo[m.away_team]
                expect = 1/(1 + 10**((elo[m.away_team] - elo[m.home_team] - c["elo_home_edge"])/c["elo_scale"]))
                result = 1 if m.home_goals > m.away_goals else 0 if m.home_goals < m.away_goals else .5
                updates.append((m.home_team, m.away_team, c["elo_k"]*(result - expect)))
            for h, a, d in updates:
                elo[h] += d
                elo[a] -= d
        mine = np.array([pre[(m, t)] for m, t in zip(self.rows.match_id, self.rows.team)])
        theirs = np.array([pre[(m, t)] for m, t in zip(self.rows.match_id, self.rows.opponent)])
        np.testing.assert_allclose(mine, self.rows.team_elo.to_numpy(), atol=1e-6)
        np.testing.assert_allclose(theirs, self.rows.opponent_elo.to_numpy(), atol=1e-6)

    def test_pipeline_counts_match_sources(self):
        pl, m, rows = self.p["pipeline"], CTX.matches, self.rows
        self.assertEqual(pl["league_matches"], len(CTX.league))
        self.assertEqual(pl["matches"], len(m))
        self.assertEqual(pl["cohort_matches_in_league"], pl["matches"])
        self.assertEqual(pl["espn_verified"] + pl["unmatched"], pl["matches"])
        self.assertEqual(pl["rows"], len(rows))
        self.assertEqual(pl["rows"], pl["matches"] + pl["derby_matches"])
        self.assertEqual(pl["derby_rows"], 2*pl["derby_matches"])
        # ESPN '3-4-3-d' is merged into the plain shape, never kept as its own label.
        for side in ["home", "away"]:
            dmask = m[f"{side}_formation_raw"].astype(str).str.endswith("-d")
            self.assertTrue((m.loc[dmask, f"{side}_formation"] == m.loc[dmask, f"{side}_formation_raw"].str[:-2]).all())
        self.assertGreater(pl["formation_d_count"], 0)
        self.assertEqual(pl["formation_raw_kinds"] - pl["formation_clean_kinds"], len(pl["formation_d_labels"]))
        self.assertEqual(pl["formation_missing_rows"], int(rows.formation.eq("__missing__").sum()))
        self.assertEqual(pl["formation_missing_sides"], int(m.home_formation.isna().sum() + m.away_formation.isna().sum()))
        self.assertEqual(sum(g["n"] for g in pl["groups"]), pl["numeric"])
        # One-hot + numeric + 'was missing' indicators = every column the forest sees.
        self.assertEqual(pl["onehot_total"] + pl["numeric"] + pl["indicators"], pl["total_columns"])
        self.assertEqual(pl["onehot_total"], sum(rows[c][CTX.train_mask].nunique() for c in ["team", "opponent", "formation",
                                                                                           "team_formation", "previous_formation"]))
        self.assertEqual(pl["indicators"], int(CTX.train[CTX.numeric].isna().any().sum()))
        self.assertEqual(pl["missing_cells"], int(rows[CTX.numeric].isna().to_numpy().sum()))

    def test_real_imputer_and_scaler_parameters(self):
        train = CTX.train
        for var in self.p["vars"]:
            raw = train[var["name"]].to_numpy(dtype=float)
            self.assertAlmostEqual(var["median"], float(np.nanmedian(raw)), places=5)
            filled = np.where(np.isnan(raw), np.nanmedian(raw), raw)
            self.assertAlmostEqual(var["mean"], float(filled.mean()), places=4)
            self.assertAlmostEqual(var["scale"], float(filled.std()), places=4)
            self.assertEqual(var["train_missing"], int(np.isnan(raw).sum()))
            self.assertEqual("ind_mean" in var, bool(np.isnan(raw).any()))
            if "ind_mean" in var:
                self.assertAlmostEqual(var["ind_mean"], float(np.isnan(raw).mean()), places=5)

    def test_stats_correlations_and_season_means(self):
        rows = self.rows
        for name in ["diff_elo", "team_rest_days", "defenders"]:
            values = rows[name].to_numpy(dtype=float)
            for code, key in [(2, "win"), (1, "draw"), (0, "loss")]:
                v = values[(rows.outcome == code).to_numpy()]
                v = v[np.isfinite(v)]
                s = self.p["stats"][name][key]
                self.assertEqual(s["n"], len(v))
                self.assertAlmostEqual(s["mean"], v.mean(), places=3)
                q1, q3 = np.percentile(v, [25, 75])
                self.assertEqual(s["outliers"], int(((v < q1 - 1.5*(q3-q1)) | (v > q3 + 1.5*(q3-q1))).sum()))
            self.assertAlmostEqual(self.p["stats"][name]["all"]["miss_rate"], float(np.isnan(values).mean()), places=4)
        corr = self.p["corr"]
        matrix = np.array(corr["r"], dtype=float)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-9)
        score = rows.outcome.map({0: 0, 1: 1, 2: 3}).to_numpy(dtype=float)
        i = corr["names"].index("diff_elo")
        self.assertAlmostEqual(matrix[i, -1], np.corrcoef(rows.diff_elo, score)[0, 1], places=3)
        self.assertEqual(corr["outcome_rank"][0]["name"], max(corr["names"][:-1], key=lambda n: abs(matrix[corr["names"].index(n), -1])))
        self.assertLessEqual(corr["dup_coverage_pairs"], corr["all_dup_pairs"])
        for example in corr["dup_examples"]:
            self.assertGreaterEqual(example["r"], 0.95)
        score_s = pd.Series(score, index=rows.index)
        strongest = max(abs(rows[n].astype(float).corr(score_s)) for n in CTX.numeric if rows[n].std() > 0)
        self.assertAlmostEqual(corr["all_top"][0]["r"], strongest, places=3)
        means = self.p["season_stats"]["team_elo"]["mean"]
        expected = rows.groupby("season").team_elo.mean()
        np.testing.assert_allclose(means, [expected[s] for s in self.p["seasons"]], atol=1e-3)

    def test_missing_tables_agree_with_coverage_file(self):
        heat = {row["key"]: row for row in self.p["missing"]["heat"]}
        cov = CTX.coverage.groupby("season")[["matches", "formation_recorded"]].sum()
        for k, season in enumerate(self.p["seasons"]):
            rate = 1 - cov.loc[season, "formation_recorded"]/cov.loc[season, "matches"]
            self.assertAlmostEqual(heat["formation"]["rates"][k], rate, places=4)
        self.assertEqual(heat["formation"]["rates"][0], 1.0)  # 2006/07: no formation recorded at all
        groups = {g["key"]: g for g in self.p["missing"]["groups"]}
        self.assertEqual(groups["elo"]["rate"], 0.0)
        self.assertEqual(sum(g["cols"] for g in groups.values()), len(CTX.numeric))

    def test_weights_curve_and_shares(self):
        w = self.p["weights"]
        curve = dict((y, v) for y, v in w["curve"])
        self.assertAlmostEqual(curve[0.0], 1.0)
        self.assertAlmostEqual(w["marks"][0][1], 0.5, places=2)       # one half-life (1095 days) ~ three years
        self.assertAlmostEqual(w["marks"][1][1], 0.25, places=2)
        self.assertAlmostEqual(sum(w["share"]), 1.0, places=3)
        self.assertEqual(w["derby_weight"], 0.5)
        self.assertEqual(self.p["pipeline"]["half_life"], CTX.report["protocol"]["half_life_days"])
        self.assertEqual(sum(w["rows"]), int(CTX.train_mask.sum()))

    def test_placeholder_zero_stats_are_counted_from_both_sources(self):
        z, m = self.p["zeros"], CTX.matches
        flag = m.home_possession.eq(0) & m.away_possession.eq(0)
        self.assertEqual(z["matches"], int(flag.sum()))
        fd = CTX.league.set_index("match_id").loc[m.match_id[flag]]
        self.assertEqual(z["fd_has_shots"], int(((fd.home_shots + fd.away_shots) > 0).sum()))
        self.assertLessEqual(z["rows_in_window20"], z["rows"])

    def _flags(self):
        z = self.p["zeros"]
        return np.array([c == "1" for c in z["team_flag"]]), np.array([c == "1" for c in z["opp_flag"]])

    def test_placeholder_window_flags_match_an_independent_history_walk(self):
        """Per-row flags: the previous 20 league games of the club (team) / the opponent (opp) hold a placeholder match."""
        m, rows, z = CTX.matches, self.rows, self.p["zeros"]
        ids = set(m.match_id[m.home_possession.eq(0) & m.away_possession.eq(0)])
        history = {}
        for g in CTX.league.sort_values(["date", "match_id"]).itertuples():
            for club in (g.home_team, g.away_team):
                history.setdefault(club, []).append((g.date, g.match_id in ids))
        def walk(club, date):
            return any(hit for _, hit in [h for h in history[club] if h[0] < date][-z["window"]:])
        team, opp = self._flags()
        self.assertEqual(len(team), len(rows))
        for i in list(range(0, len(rows), 7)) + [len(rows) - 1]:
            self.assertEqual(team[i], walk(rows.team.iloc[i], rows.date.iloc[i]), i)
            self.assertEqual(opp[i], walk(rows.opponent.iloc[i], rows.date.iloc[i]), i)
        # The club's own window can also be read from its rows (a club plays every league game as a row).
        hit = rows.match_id.isin(ids)
        own = hit.groupby(rows.team).transform(lambda s: s.astype(int).shift(1, fill_value=0).rolling(20, min_periods=1).sum() > 0)
        self.assertTrue((own.to_numpy() == team).all())
        self.assertEqual(z["rows_in_window20"], int(team.sum()))
        self.assertEqual(z["rows_opp"], int(opp.sum()))
        self.assertEqual(z["rows_either"], int((team | opp).sum()))
        self.assertGreater(z["rows_either"], z["rows_in_window20"])   # opponent windows add affected rows

    def test_clean_statistics_leave_out_placeholder_rows(self):
        from epl_tactics.lab.export_s2 import zero_side
        team, opp = self._flags()
        rows, score = self.rows, self.rows.outcome.map({0: 0, 1: 1, 2: 3}).astype(float)
        affected = [v for v in self.p["vars"] if v["zero_side"]]
        self.assertTrue({"team_shots_20", "team_possession_20", "diff_sot_20", "opponent_possession_20"} <= {v["name"] for v in affected})
        for v in self.p["vars"]:
            self.assertEqual(v["zero_side"], zero_side(v["name"]))
            self.assertEqual(v["zero_flag"], v["zero_side"] is not None)
        for v in affected:
            hit = team if v["zero_side"] == "team" else opp if v["zero_side"] == "opp" else team | opp
            keep = ~hit
            values = rows[v["name"]].to_numpy(dtype=float)
            clean = self.p["stats"][v["name"]]["clean"]
            for code, key in [(2, "win"), (0, "loss")]:
                part = values[keep & (rows.outcome == code).to_numpy()]
                self.assertAlmostEqual(clean[key]["mean"], np.nanmean(part), places=3)
            self.assertEqual(clean["all"]["n"], int(np.isfinite(values[keep]).sum()))
            ss = self.p["season_stats"][v["name"]]
            for k, season in enumerate(self.p["seasons"]):
                part = values[keep & (rows.season == season).to_numpy()]
                n = int(np.isfinite(part).sum())
                self.assertEqual(ss["clean_n"][k], n)
                if n >= self.p["zeros"]["clean_min"]:
                    self.assertAlmostEqual(ss["clean_mean"][k], np.nanmean(part), places=3)
                else:
                    self.assertIsNone(ss["clean_mean"][k])
        # Correlation footnote: r with the outcome score on all rows and without affected rows.
        for item in self.p["corr"]["zero"]:
            side = zero_side(item["name"])
            keep = ~(team if side == "team" else opp if side == "opp" else team | opp)
            self.assertAlmostEqual(item["r_all"], rows[item["name"]].astype(float).corr(score), places=3)
            self.assertAlmostEqual(item["r_clean"], rows[item["name"]][keep].astype(float).corr(score[keep]), places=3)
            self.assertEqual(item["n_clean"], int(keep.sum()))
            # The takeaway names the strongest input over all rows; cleaning these inputs does not overturn it.
            self.assertLess(abs(item["r_clean"]), self.p["corr"]["all_top"][0]["r"])
        self.assertEqual({z["name"] for z in self.p["corr"]["zero"]}, {n for n in self.p["corr"]["names"] if zero_side(n)})

    def test_scatter_preset_with_placeholder_rows_is_recomputable_from_payload(self):
        """The page recomputes r in the browser from the shipped columns and flags; it must equal pandas on the source."""
        team, _ = self._flags()
        preset = next(p for p in self.p["presets"] if p["x"] == "team_shots_20")
        x, y = self.col(preset["x"]), self.col(preset["y"])
        ok = np.isfinite(x) & np.isfinite(y)
        src = self.rows[[preset["x"], preset["y"]]]
        self.assertAlmostEqual(np.corrcoef(x[ok], y[ok])[0, 1], src.corr().iloc[0, 1], places=3)
        keep = ok & ~team
        clean_src = src[~team].corr().iloc[0, 1]
        self.assertAlmostEqual(np.corrcoef(x[keep], y[keep])[0, 1], clean_src, places=3)
        # The page says the relation gets clearer without those rows: true by a wide margin.
        self.assertGreater(clean_src - src.corr().iloc[0, 1], 0.05)

    def test_missing_cells_by_group(self):
        """'Most missing by rate' and 'most missing by count' are different groups; the page computes both."""
        from epl_tactics.lab.catalog import build_catalog
        cat = {c["name"]: c["group"] for c in build_catalog(CTX.numeric)}
        miss = self.p["missing"]
        na = self.rows[CTX.numeric].isna()
        total = 0
        for g in miss["groups"]:
            cols = [n for n in CTX.numeric if cat[n] == g["key"]]
            self.assertEqual(g["cells"], int(na[cols].to_numpy().sum()))
            self.assertLessEqual(g["opp_espn_cells"], g["opp_cells"])
            self.assertLessEqual(g["opp_cells"], g["cells"])
            total += g["cells"]
        self.assertEqual(miss["total_cells"], total)
        self.assertEqual(miss["total_cells"], self.p["pipeline"]["missing_cells"])
        self.assertEqual(set(miss["espn_only"]), {"possession", "passes", "accurate_passes", "crosses", "long_balls",
                                                   "tackles", "interceptions", "saves", "blocked_shots"})
        by_rate = max(miss["groups"], key=lambda g: g["rate"])
        by_cells = max(miss["groups"], key=lambda g: g["cells"])
        if by_rate["key"] != by_cells["key"]:
            # so the old sentence "missing values are mostly formation records" would be false
            self.assertLess(by_rate["cells"], 0.5*miss["total_cells"])
        if by_cells["cells"]:
            espn = [c for c in CTX.numeric if cat[c] == by_cells["key"] and c.startswith("opponent_")
                    and any(c == f"opponent_{s}_{w}" for s in miss["espn_only"] for w in (5, 10, 20))]
            self.assertEqual(by_cells["opp_espn_cells"], int(na[espn].to_numpy().sum()))

    def test_unseen_categories_are_really_new(self):
        train, held = CTX.train, CTX.heldout
        for item in self.p["unseen"]:
            for value in item["values"]:
                self.assertNotIn(value, set(train[item["name"]]))
                self.assertIn(value, set(held[item["name"]]))


if __name__ == "__main__":
    unittest.main()
