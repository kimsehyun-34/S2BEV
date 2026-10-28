"""Predefined, season-forward experiment with an untouched recent-season evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import read_matches
from .estimators_v2 import SPECS, OutcomeEnsemble, fit_spec, match_predictions
from .features import FEATURES, build_features
from .features_v2 import build_features_v2
from .model import make_pipeline, metrics
from .statistics import legacy_audit, paired_week_interval

PROBABILITY_COLUMNS = ["p_loss", "p_draw", "p_win"]


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _proba(frame: pd.DataFrame) -> np.ndarray:
    return frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)


def data_provenance(data_path: str | Path, market_path: str | Path) -> dict:
    output = {}
    manifest = Path(data_path).with_suffix(".manifest.json")
    if manifest.exists():
        collection = json.loads(manifest.read_text(encoding="utf-8"))
        if collection.get("csv_sha256") != hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
            raise ValueError("Collection manifest checksum does not match the input CSV")
        output["collection"] = collection
    crosscheck = Path(market_path).with_name("crosscheck.json")
    if crosscheck.exists():
        output["crosscheck"] = json.loads(crosscheck.read_text(encoding="utf-8"))
    return output


def select_models(rows: pd.DataFrame, validation_seasons: list[str], formation: bool) -> tuple[dict, list[dict]]:
    records, predictions, truth_frame = [], [], None
    for spec in SPECS:
        fold_probabilities, fold_records, fold_frames = [], [], []
        for season in validation_seasons:
            cutoff = rows[rows.season == season].date.min()
            development = rows[rows.date < cutoff]
            validation = rows[rows.season == season]
            model = fit_spec(spec, development, formation)
            forecast = match_predictions(validation, model.predict_proba(validation))
            proba = _proba(forecast)
            fold_probabilities.append(proba)
            fold_frames.append(forecast)
            fold_records.append({"season": season, "matches": len(forecast),
                                 **metrics(forecast.outcome.to_numpy(), proba)})
        combined = pd.concat(fold_frames, ignore_index=True)
        probability = np.vstack(fold_probabilities)
        predictions.append(probability)
        truth_frame = combined
        records.append({"spec": spec, "with_formation": formation, "folds": fold_records,
                        "validation": metrics(combined.outcome.to_numpy(), probability)})
        print(f"Validation {'formation' if formation else 'no formation'} {spec}: logloss={records[-1]['validation']['log_loss']:.4f}", flush=True)
    ranked = sorted(range(len(records)), key=lambda i: records[i]["validation"]["log_loss"])
    finalists, families = [], set()
    for index in ranked:
        family = records[index]["spec"]["family"]
        if family not in families:
            finalists.append(index)
            families.add(family)
        if len(finalists) == 3:
            break
    combinations = [([index], [1.0]) for index in finalists]
    if len(finalists) >= 2:
        combinations += [(finalists[:2], [0.5, 0.5]), (finalists[:2], [0.75, 0.25])]
    if len(finalists) == 3:
        combinations += [(finalists, [1/3, 1/3, 1/3]), (finalists, [0.5, 0.25, 0.25])]
    blends = []
    for indices, weights in combinations:
        probability = sum(weight * predictions[index] for index, weight in zip(indices, weights))
        blends.append({"specs": [SPECS[index] for index in indices], "weights": weights,
                       "validation": metrics(truth_frame.outcome.to_numpy(), probability)})
    selected = min(blends, key=lambda blend: blend["validation"]["log_loss"])
    return selected, records + [{"blend_candidates": blends}]


def _score_predictions(predictions: dict[str, pd.DataFrame]) -> dict:
    output = {name: {"matches": len(frame), **metrics(frame.outcome.to_numpy(), _proba(frame))}
              for name, frame in predictions.items()}
    if "uniform_random" in output:
        # A uniform distribution's argmax is a deterministic tie-break, not random guessing.
        output["uniform_random"]["accuracy"] = 1 / 3
        output["uniform_random"]["accuracy_note"] = "Expected accuracy of uniform random guesses"
    return output


def support_audit(rows: pd.DataFrame) -> dict:
    latest_season = rows.iloc[-1].season
    active_teams = sorted(rows[rows.season == latest_season].team.unique())
    recent = rows[rows.date >= rows.date.max() - pd.Timedelta(days=730)]
    table = []
    for team in active_teams:
        team_rows = recent[recent.team == team]
        counts = team_rows.formation.value_counts()
        supported = counts[counts >= 10]
        table.append({"team": team, "appearances_last_730_days": len(team_rows),
                      "formations_observed": len(counts), "formations_with_at_least_10_appearances": len(supported),
                      "most_used_formation": str(counts.index[0]), "most_used_share": float(counts.iloc[0]/len(team_rows)),
                      "formation_counts": counts.to_dict(), "recommendation": None,
                      "reason": "No identified causal comparison or validated recommendation policy"})
    return {"season": latest_season, "window_days": 730, "support_floor": 10, "teams": table,
            "teams_with_two_supported_formations": sum(t["formations_with_at_least_10_appearances"] >= 2 for t in table),
            "note": "Counts are necessary overlap diagnostics, not proof of comparable opponents, coaches or squads."}


def _baseline_forecasts(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    home_training = train_rows[train_rows.is_home == 1]
    test = match_predictions(test_rows, np.full((len(test_rows), 3), 1/3))
    prior = np.bincount(home_training.outcome.to_numpy(dtype=int), minlength=3).astype(float)
    prior /= prior.sum()
    home_prior = test.copy()
    home_prior[PROBABILITY_COLUMNS] = np.tile(prior, (len(test), 1))
    # This baseline's hard prediction is the most frequent *home-view* outcome,
    # usually home win. Unlike the old mirrored class-prior, it retains home advantage.
    elo_features = ["is_home", "diff_elo"]
    elo = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=1, max_iter=1000))])
    elo.fit(train_rows[elo_features], train_rows.outcome)
    elo_forecast = match_predictions(test_rows, elo.predict_proba(test_rows[elo_features]))
    return {"uniform_random": test, "home_prior": home_prior, "elo_logistic": elo_forecast}


def run_benchmark(data_path: str | Path = "data/epl_recent.csv", output_dir: str | Path = "artifacts/v2",
                  market_path: str | Path = "data/football_data/market_probabilities.csv",
                  validation_seasons: list[str] | None = None, test_start_season: str = "2024/2025") -> dict:
    validation_seasons = validation_seasons or ["2022/2023", "2023/2024"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matches = read_matches(data_path)
    quality = data_provenance(data_path, market_path)
    seasons = sorted(matches.season.unique())
    if not all(season in seasons for season in validation_seasons) or test_start_season not in seasons:
        raise ValueError("Required validation/test seasons are absent")
    if max(validation_seasons) >= test_start_season:
        raise ValueError("Validation must precede the first test season")
    protocol = {
        "created_before_test_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "validation_seasons": validation_seasons, "test_seasons": [s for s in seasons if s >= test_start_season],
        "selection_metric": "Mean per-match multiclass log loss across two forward validation seasons",
        "specs": SPECS, "ensemble": "Three distinct-family validation finalists; fixed single/equal/top-heavy blends",
        "schedule": "Expanding training, refit at each season start; test parameters and weights frozen",
        "probabilities": "One match distribution: average home probabilities and reversed away probabilities",
        "features": "Strictly before match date; current-match shots, possession, goals, odds and records excluded",
        "hypothesis": "Does formation improve held-out match outcome prediction relative to its tuned ablation?",
        "formation_signal_rule": "Log-loss gain >= 0.01 and lower bound of paired calendar-week bootstrap 95% interval > 0",
        "causal_rule": "No observational predictive score alone authorizes an optimal-formation recommendation",
        "selection_warning": "2015/16 test was already seen and is audited only; never used for new selection or headline test",
    }
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and (output_dir / "metrics.json").exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        comparable = lambda p: {k: v for k, v in p.items() if k != "created_before_test_evaluation_utc"}
        if comparable(previous) != comparable(protocol):
            raise ValueError("Existing evaluated protocol differs. Use a new output directory; do not retune on this test.")
        protocol = previous
    _json(protocol_path, protocol)
    with threadpool_limits(limits=4):
        rows, states = build_features_v2(matches)
        old_rows, _ = build_features(matches)
        assert rows.match_id.tolist() == old_rows.match_id.astype(str).tolist()
        selected_no, selection_no = select_models(rows, validation_seasons, False)
        selected_yes, selection_yes = select_models(rows, validation_seasons, True)
        selection = {"without_formation": selected_no, "with_formation": selected_yes,
                     "details_without": selection_no, "details_with": selection_yes,
                     "frozen_before_test_utc": datetime.now(timezone.utc).isoformat()}
        _json(output_dir / "selection.json", selection)
        print("Model selection frozen. Beginning held-out season evaluation.", flush=True)
        by_model: dict[str, list[pd.DataFrame]] = {}
        per_season = {}
        for season in protocol["test_seasons"]:
            cutoff = rows[rows.season == season].date.min()
            development, test = rows[rows.date < cutoff], rows[rows.season == season]
            forecasts = _baseline_forecasts(development, test)
            # Same old algorithm and hyperparameters, retrained on the expanded data.
            old_development = old_rows[old_rows.date < cutoff]
            old_test = old_rows[old_rows.season == season]
            legacy_model = make_pipeline(min_samples_leaf=10, max_depth=7).fit(old_development[FEATURES], old_development.outcome)
            forecasts["legacy_rf_retrained"] = match_predictions(old_test, legacy_model.predict_proba(old_test[FEATURES]))
            for name, selected, with_formation in [("without_formation", selected_no, False),
                                                    ("with_formation", selected_yes, True)]:
                estimator = OutcomeEnsemble(selected, with_formation).fit(development)
                forecasts[name] = match_predictions(test, estimator.predict_proba(test))
            per_season[season] = _score_predictions(forecasts)
            for name, frame in forecasts.items():
                by_model.setdefault(name, []).append(frame)
            print(f"Finished locked evaluation for {season}: {len(forecasts['with_formation'])} matches", flush=True)
        forecasts = {name: pd.concat(frames, ignore_index=True) for name, frames in by_model.items()}
        primary = forecasts["with_formation"]
        comparisons = {}
        for name, reference in [("formation_vs_ablation", "without_formation"),
                                 ("upgraded_vs_legacy", "legacy_rf_retrained"),
                                 ("upgraded_vs_home_prior", "home_prior"), ("upgraded_vs_elo", "elo_logistic")]:
            comparisons[name] = {score: paired_week_interval(primary, _proba(primary), _proba(forecasts[reference]), score)
                                  for score in ["accuracy", "log_loss"]}
        market_report = {"available": False}
        if Path(market_path).exists():
            market = pd.read_csv(market_path)
            if not market.verified.fillna(False).all():
                raise ValueError("Independent match result cross-check contains unverified records")
            merged = primary.merge(market[["match_id", *PROBABILITY_COLUMNS]], on="match_id", suffixes=("", "_market"), validate="one_to_one")
            valid = merged[[f"{c}_market" for c in PROBABILITY_COLUMNS]].notna().all(axis=1)
            subset = merged[valid]
            market_probability = subset[[f"{c}_market" for c in PROBABILITY_COLUMNS]].to_numpy()
            market_report = {"available": True, "matches": len(subset),
                             "market": metrics(subset.outcome.to_numpy(), market_probability),
                             "with_formation_same_subset": metrics(subset.outcome.to_numpy(), subset[PROBABILITY_COLUMNS].to_numpy()),
                             "note": "Normalized pre-kickoff odds, extra information; reference rather than equal-input baseline"}
        signal = comparisons["formation_vs_ablation"]["log_loss"]
        signal_supported = signal["improvement"] >= 0.01 and signal["ci_95_low"] > 0
        deploy_with_formation = selected_yes["validation"]["log_loss"] < selected_no["validation"]["log_loss"]
        deployed_selection = selected_yes if deploy_with_formation else selected_no
        report = {
            "experiment": "v2_recent_season_forward", "model": "Validation-selected outcome ensemble",
            "data_quality": quality,
            "data": {"csv": str(data_path), "sha256": protocol["data_sha256"], "matches": len(matches),
                     "seasons": seasons, "trained_through": str(matches.date.max().date()),
                     "recommendation_season": seasons[-1], "test_matches": len(primary)},
            "selection": {"without_formation": selected_no, "with_formation": selected_yes},
            "test": _score_predictions(forecasts), "test_by_season": per_season,
            "comparisons": comparisons, "market_reference": market_report,
            "formation_signal_supported": bool(signal_supported),
            "tactical_recommendation_validated": False,
            "tactical_conclusion": "Optimal formation not identified from these observational match outcomes",
            "deployment": {"uses_formation": bool(deploy_with_formation),
                           "choice": "Frozen by forward-validation log loss, never test accuracy",
                           "purpose": "Match outcome probabilities only; no optimal-formation recommendation"},
            "support_audit": support_audit(rows), "legacy_audit": legacy_audit(), "protocol": protocol,
            "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
            "limitations": ["Match results never reveal what would have happened under a different formation.",
                            "Coach, player availability, selection policy and opponent response remain confounders.",
                            "Final lineup formations are observed retrospectively; no timestamped pre-kickoff snapshot is guaranteed.",
                            "COVID-era fixtures and repeated teams induce dependence beyond week clusters.",
                            "Current season contains few rounds and should not be treated as a full-season validation.",
                            "Market benchmark has richer information and is not a betting-profit evaluation."],
        }
        _json(output_dir / "metrics.json", report)
        export = primary[["match_id", "date", "season", "team", "opponent", "outcome"]].copy()
        for name, frame in forecasts.items():
            for column in PROBABILITY_COLUMNS:
                export[f"{name}_{column}"] = frame[column].to_numpy()
        export.to_csv(output_dir / "test_predictions.csv", index=False)
        _json(output_dir / "support_audit.json", report["support_audit"])
        # Final artifact is separate from all reported out-of-sample forecasts.
        deployed = OutcomeEnsemble(deployed_selection, deploy_with_formation).fit(rows)
        joblib.dump({"artifact_version": 2, "pipeline": deployed, "states": states,
                     "report": report, "tactical_recommendation_validated": False},
                    output_dir / "model.joblib", compress=3)
    return report


def predict_fixture(bundle: dict, team: str, opponent: str, is_home: bool = True,
                    formation: str | None = None, opponent_formation: str | None = None,
                    as_of: str | None = None) -> dict:
    from .data import normalize_formation
    from .features_v2 import context_v2
    if bundle.get("artifact_version") != 2:
        raise ValueError("Use the v2 outcome artifact")
    report = bundle["report"]
    active = [row["team"] for row in report["support_audit"]["teams"]]
    if team not in active or opponent not in active or team == opponent:
        raise ValueError(f"Use two different active teams: {', '.join(active)}")
    through = pd.Timestamp(report["data"]["trained_through"])
    date = pd.Timestamp(as_of).normalize() if as_of else through + pd.Timedelta(days=1)
    if date.tzinfo is not None or pd.isna(date) or date <= through:
        raise ValueError(f"Prediction date must be after {through.date()}")
    if report["deployment"]["uses_formation"] and (formation is None or opponent_formation is None):
        raise ValueError("This artifact requires both announced starting formations")
    own_formation = normalize_formation(formation) if formation else "__unknown__"
    opp_formation = normalize_formation(opponent_formation) if opponent_formation else "__unknown__"
    own, other = bundle["states"][team], bundle["states"][opponent]
    rows = pd.DataFrame([
        context_v2(team, opponent, own_formation, is_home, own, other, date),
        context_v2(opponent, team, opp_formation, not is_home, other, own, date),
    ])
    raw = bundle["pipeline"].predict_proba(rows)
    probability = (raw[0] + raw[1, ::-1]) / 2
    return {"team": team, "opponent": opponent, "venue": "home" if is_home else "away",
            "scenario_only": True, "formation": formation, "opponent_formation": opponent_formation,
            "as_of": str(date.date()), "trained_through": str(through.date()),
            "p_loss": float(probability[0]), "p_draw": float(probability[1]), "p_win": float(probability[2]),
            "expected_points": float(probability[1] + 3 * probability[2]),
            "recommended_formation": None, "tactical_status": "not_identified",
            "note": "Outcome forecast only. Changing the formation is not a validated tactical intervention."}
