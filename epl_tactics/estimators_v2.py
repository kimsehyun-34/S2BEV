"""Small, predefined classifier/goal-model comparison; no new ML dependencies."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features_v2 import NUMERIC_V2

SPECS = [
    {"family": "logistic", "C": 0.03}, {"family": "logistic", "C": 0.3},
    {"family": "logistic", "C": 3.0},
    {"family": "hist_gradient", "leaves": 7}, {"family": "hist_gradient", "leaves": 15},
    {"family": "poisson", "alpha": 0.03}, {"family": "poisson", "alpha": 0.3},
    {"family": "random_forest", "leaf": 8}, {"family": "random_forest", "leaf": 16},
]


def weights_for(rows: pd.DataFrame) -> np.ndarray:
    days = (rows.date.max() - rows.date).dt.days.to_numpy()
    return np.exp(-np.log(2) * days / 730.0)


def processor(formation: bool, scaled: bool = False) -> ColumnTransformer:
    categories = ["team", "opponent"] + (["formation"] if formation else [])
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
        ("numeric", StandardScaler() if scaled else "passthrough", NUMERIC_V2),
    ])


class GoalModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, alpha: float, formation: bool):
        self.alpha, self.formation = alpha, formation

    def fit(self, rows: pd.DataFrame):
        self.scored_ = Pipeline([("features", processor(self.formation, True)),
                                 ("regressor", PoissonRegressor(alpha=self.alpha, max_iter=1500, tol=1e-7))])
        self.conceded_ = Pipeline([("features", processor(self.formation, True)),
                                   ("regressor", PoissonRegressor(alpha=self.alpha, max_iter=1500, tol=1e-7))])
        weights = weights_for(rows)
        self.scored_.fit(rows, rows.goals_for, regressor__sample_weight=weights)
        self.conceded_.fit(rows, rows.goals_against, regressor__sample_weight=weights)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        scored = np.clip(self.scored_.predict(rows), 0.05, 8)
        conceded = np.clip(self.conceded_.predict(rows), 0.05, 8)
        loss = skellam.cdf(-1, scored, conceded)
        draw = skellam.pmf(0, scored, conceded)
        output = np.column_stack([loss, draw, 1 - loss - draw])
        output = np.clip(output, 1e-9, 1)
        return output / output.sum(axis=1, keepdims=True)


def fit_spec(spec: dict, rows: pd.DataFrame, formation: bool):
    family = spec["family"]
    if family == "poisson":
        return GoalModel(spec["alpha"], formation).fit(rows)
    if family == "logistic":
        estimator = LogisticRegression(C=spec["C"], max_iter=2000, random_state=42)
    elif family == "hist_gradient":
        estimator = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.04, max_leaf_nodes=spec["leaves"],
            min_samples_leaf=30, l2_regularization=5.0, max_bins=63,
            early_stopping=False, random_state=42,
        )
    elif family == "random_forest":
        estimator = RandomForestClassifier(n_estimators=250, min_samples_leaf=spec["leaf"],
                                            max_depth=10, max_features=0.8, n_jobs=4, random_state=42)
    else:
        raise ValueError(f"Unknown family: {family}")
    result = Pipeline([("features", processor(formation, family == "logistic")), ("classifier", estimator)])
    result.fit(rows, rows.outcome, classifier__sample_weight=weights_for(rows))
    return result


class OutcomeEnsemble:
    classes_ = np.array([0, 1, 2])

    def __init__(self, selection: dict, formation: bool):
        self.selection, self.formation = selection, formation

    def fit(self, rows: pd.DataFrame):
        self.models_ = [fit_spec(spec, rows, self.formation) for spec in self.selection["specs"]]
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return sum(weight * model.predict_proba(rows)
                   for weight, model in zip(self.selection["weights"], self.models_))


def match_predictions(rows: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    """One coherent distribution per fixture, symmetrizing the two team views."""
    frame = rows[["match_id", "date", "season", "team", "opponent", "is_home", "outcome"]].copy()
    for i, label in enumerate(["loss", "draw", "win"]):
        frame[f"p_{label}"] = probability[:, i]
    home = frame[frame.is_home == 1].set_index("match_id").sort_index()
    away = frame[frame.is_home == 0].set_index("match_id").reindex(home.index)
    if len(home) * 2 != len(rows) or away.team.isna().any() or not home.team.eq(away.opponent).all():
        raise ValueError("Both home/away rows are required exactly once per match")
    columns = ["p_loss", "p_draw", "p_win"]
    home[columns] = (home[columns].to_numpy() + away[columns].to_numpy()[:, ::-1]) / 2
    return home.reset_index().drop(columns="is_home")
