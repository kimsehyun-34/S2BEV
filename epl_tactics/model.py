"""Random forest outcome model with forward-only model selection and evaluation."""

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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import read_matches
from .features import CATEGORICAL, NUMERIC, FEATURES, build_features, chronological_split

SEED = 42
ARTIFACT_VERSION = 1


def make_pipeline(min_samples_leaf: int = 5, max_depth: int = 10,
                  include_formation: bool = True) -> Pipeline:
    categories = CATEGORICAL if include_formation else ["team", "opponent"]
    processor = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
        ("numeric", "passthrough", NUMERIC),
    ])
    return Pipeline([
        ("features", processor),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
            max_features=0.8, random_state=SEED, n_jobs=-1,
        )),
    ])


def probabilities(estimator, features: pd.DataFrame) -> np.ndarray:
    """Always return columns in loss, draw, win order."""
    raw = estimator.predict_proba(features)
    classes = estimator.classes_
    output = np.zeros((len(features), 3), dtype=float)
    for index, label in enumerate(classes):
        output[:, int(label)] = raw[:, index]
    return output


def metrics(actual: np.ndarray, proba: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=int)
    points = np.array([0.0, 1.0, 3.0])
    return {
        "accuracy": float(accuracy_score(actual, proba.argmax(axis=1))),
        "log_loss": float(log_loss(actual, proba, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((proba - np.eye(3)[actual]) ** 2, axis=1))),
        "expected_points_mae": float(mean_absolute_error(points[actual], proba @ points)),
    }


def _period(frame: pd.DataFrame) -> dict:
    return {"first_date": str(frame.date.min().date()), "last_date": str(frame.date.max().date()),
            "matches": int(frame.match_id.nunique()), "team_rows": len(frame)}


def train(csv_path: str | Path, output_dir: str | Path) -> dict:
    matches = read_matches(csv_path)
    rows, states = build_features(matches)
    training, validation, test = chronological_split(rows)
    candidates = [dict(min_samples_leaf=5, max_depth=10), dict(min_samples_leaf=10, max_depth=7)]
    results = []
    for parameters in candidates:
        pipeline = make_pipeline(**parameters).fit(training[FEATURES], training.outcome)
        score = metrics(validation.outcome.to_numpy(), probabilities(pipeline, validation[FEATURES]))
        results.append({"parameters": parameters, "validation": score})
    selected = min(results, key=lambda result: result["validation"]["log_loss"])
    development = pd.concat([training, validation], ignore_index=True)
    full = make_pipeline(**selected["parameters"]).fit(development[FEATURES], development.outcome)
    # Tune the ablated comparator independently on the same validation period.
    ablation_results = []
    for parameters in candidates:
        estimator = make_pipeline(**parameters, include_formation=False).fit(
            training[FEATURES], training.outcome)
        score = metrics(validation.outcome.to_numpy(), probabilities(estimator, validation[FEATURES]))
        ablation_results.append({"parameters": parameters, "validation": score})
    ablation_selected = min(ablation_results, key=lambda result: result["validation"]["log_loss"])
    no_formation = make_pipeline(**ablation_selected["parameters"], include_formation=False).fit(
        development[FEATURES], development.outcome)
    dummy = DummyClassifier(strategy="prior").fit(development[NUMERIC], development.outcome)
    predictions = {
        "formation_model": probabilities(full, test[FEATURES]),
        "without_formation": probabilities(no_formation, test[FEATURES]),
        "class_prior": probabilities(dummy, test[NUMERIC]),
    }
    test_metrics = {name: metrics(test.outcome.to_numpy(), proba) for name, proba in predictions.items()}
    improvement = (test_metrics["without_formation"]["log_loss"]
                   - test_metrics["formation_model"]["log_loss"])
    latest_season = str(matches.iloc[-1].season)
    latest = rows[rows.season == latest_season]
    support = latest.groupby(["team", "formation"]).size().rename("matches").reset_index()
    league_support = rows.groupby("formation").size().to_dict()
    report = {
        "model": "RandomForestClassifier", "seed": SEED,
        "data": {"csv": str(csv_path), "sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                 "matches": len(matches), "team_rows": len(rows),
                 "seasons": sorted(matches.season.unique().tolist()),
                 "recommendation_season": latest_season, "teams": sorted(latest.team.unique().tolist()),
                 "trained_through": str(matches.date.max().date())},
        "split": {"train": _period(training), "validation": _period(validation), "test": _period(test)},
        "selection": results, "selected_parameters": selected["parameters"],
        "ablation_selection": ablation_results,
        "test": test_metrics,
        "formation_log_loss_improvement_vs_ablation": float(improvement),
        "formation_improves_test_log_loss": bool(improvement > 0),
        "method": "Daily walk-forward features; fixed 60/20/20 chronological date split. "
                  "Parameters selected only on validation. Final artifact refit on all observed matches.",
        "limitations": [
            "Observed associations do not identify the causal effect or true optimal formation.",
            "Test scores evaluate match outcomes for observed formations, not unplayed recommendations.",
            "Two rows per match are correlated; effective match count is half the team row count.",
            "One historical season; no current squads, injuries, player ability or in-match tactical changes.",
            "Opponent formation is not assumed known. Probabilities are not separately calibrated.",
            "Three appearances is only a support floor, not a guarantee of reliability.",
        ],
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    final = make_pipeline(**selected["parameters"]).fit(rows[FEATURES], rows.outcome)
    bundle = {"artifact_version": ARTIFACT_VERSION, "pipeline": final, "states": states,
              "support": support, "league_support": league_support, "report": report,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "model.joblib", compress=3)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    support.to_csv(output_dir / "formation_support.csv", index=False, encoding="utf-8-sig")
    test_output = test[["match_id", "date", "team", "opponent", "formation", "actual_points"]].copy()
    for name, proba in predictions.items():
        for index, label in enumerate(["loss", "draw", "win"]):
            test_output[f"{name}_p_{label}"] = proba[:, index]
    test_output.to_csv(output_dir / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
    return report


def load_model(path: str | Path) -> dict:
    # joblib/pickle is executable: load only artifacts you created or trust.
    bundle = joblib.load(path)
    if bundle.get("artifact_version") not in {ARTIFACT_VERSION, 2}:
        raise ValueError("Unsupported artifact version; retrain with this package")
    saved = bundle["report"]["environment"]["scikit_learn"]
    if saved != sklearn.__version__:
        raise ValueError(f"Artifact requires scikit-learn {saved}; installed {sklearn.__version__}. Retrain.")
    return bundle
