"""Load-once inputs shared by every section of the lab report.

The playground model is the tuned, formation-aware Random Forest refit on everything *before* the
first test season, so every match from the test season onward is genuinely unseen by it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..focused_features import build_focused_features
from ..focused_model import FocusedModel


@dataclass
class LabContext:
    data_dir: Path
    model_dir: Path
    rows: pd.DataFrame                      # one row per (club, match) with every model input + outcome
    numeric: list[str]                      # numeric model inputs (215)
    feature_meta: dict
    states: dict                            # final per-team ClubHistory, used for hypothetical fixtures
    report: dict                            # artifacts/focused/metrics.json
    selection: dict                         # artifacts/focused/selection.json
    trials_with: list[dict]                 # all tuning trials, formation included
    trials_without: list[dict]
    coverage: pd.DataFrame                  # data/focused/coverage.csv
    matches: pd.DataFrame                   # data/focused/matches.csv (raw, one row per real match)
    league: pd.DataFrame                    # data/focused/league_context.csv
    test_pred: pd.DataFrame                 # artifacts/focused/test_predictions.csv (official expanding refit)
    test_start: str                         # first test season label, e.g. "2024/2025"
    spec: dict                              # selected RF hyperparameters
    train_mask: np.ndarray                  # rows used to fit the playground model
    model: FocusedModel | None = None       # fitted on rows[train_mask]
    extras: dict = field(default_factory=dict)

    @property
    def train(self) -> pd.DataFrame:
        return self.rows[self.train_mask]

    @property
    def heldout(self) -> pd.DataFrame:
        return self.rows[~self.train_mask]


def load_context(data_dir: str | Path = "data/focused", model_dir: str | Path = "artifacts/focused",
                 fit: bool = True) -> LabContext:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    rows, numeric, feature_meta, states = build_focused_features(data_dir, return_states=True)
    report = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((model_dir / "selection.json").read_text(encoding="utf-8"))
    test_start = report["protocol"]["test_start"]
    first_test_date = rows[rows.season.eq(test_start)].date.min()
    if pd.isna(first_test_date):
        raise ValueError(f"no feature rows for test season {test_start!r}; rebuild data/focused and artifacts/focused together")
    train_mask = (rows.date < first_test_date).to_numpy()
    spec = selection["with_formation"]["best"]["spec"]
    context = LabContext(
        data_dir=data_dir, model_dir=model_dir, rows=rows, numeric=numeric, feature_meta=feature_meta, states=states,
        report=report, selection=selection,
        trials_with=json.loads((model_dir / "trials_formation.json").read_text(encoding="utf-8")),
        trials_without=json.loads((model_dir / "trials_no_formation.json").read_text(encoding="utf-8")),
        coverage=pd.read_csv(data_dir / "coverage.csv"),
        matches=pd.read_csv(data_dir / "matches.csv", parse_dates=["date"]),
        league=pd.read_csv(data_dir / "league_context.csv", parse_dates=["date"]),
        test_pred=pd.read_csv(model_dir / "test_predictions.csv", parse_dates=["date"]),
        test_start=test_start, spec=spec, train_mask=train_mask)
    if fit:
        context.model = FocusedModel(spec, numeric, True).fit(rows[train_mask])
    return context
