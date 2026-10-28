"""Paired match/week uncertainty rather than treating mirrored team rows as IID."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_week_interval(frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray,
                         score: str = "log_loss", repeats: int = 3000, seed: int = 42) -> dict:
    truth = frame.outcome.to_numpy(dtype=int)
    if score == "accuracy":
        differences = (candidate.argmax(axis=1) == truth).astype(float) - (baseline.argmax(axis=1) == truth)
    elif score == "log_loss":
        differences = (-np.log(np.clip(baseline[np.arange(len(truth)), truth], 1e-12, 1))
                       + np.log(np.clip(candidate[np.arange(len(truth)), truth], 1e-12, 1)))
    else:
        raise ValueError("score must be accuracy or log_loss")
    # Positive means candidate is better for both metrics.
    clusters = pd.to_datetime(frame.date).dt.to_period("W-SUN").astype(str)
    groups = pd.DataFrame({"week": clusters.to_numpy(), "improvement": differences}).groupby("week").improvement.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repeats, len(groups)))
    samples = groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[draws].sum(axis=1)
    return {"improvement": float(np.mean(differences)), "ci_95_low": float(np.quantile(samples, 0.025)),
            "ci_95_high": float(np.quantile(samples, 0.975)), "weeks": len(groups), "matches": len(frame),
            "method": "Paired calendar-week cluster percentile bootstrap", "repeats": repeats,
            "limitation": "Approximate interval for this sample; does not cover model selection or long-term season drift."}


def legacy_audit(path="artifacts/test_predictions.csv", matches_path="data/epl_2015_16.csv") -> dict:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame.date)
    source = pd.read_csv(matches_path)
    source.match_id = source.match_id.astype(str)
    frame.match_id = frame.match_id.astype(str)
    home_lookup = source.set_index("match_id").home_team
    frame["home_row"] = frame.team.eq(frame.match_id.map(home_lookup))
    groups = []
    for match_id, game in frame.groupby("match_id"):
        candidate = game[[f"formation_model_p_{label}" for label in ["loss", "draw", "win"]]].to_numpy()
        baseline = game[[f"without_formation_p_{label}" for label in ["loss", "draw", "win"]]].to_numpy()
        truth = game.actual_points.map({0: 0, 1: 1, 3: 2}).to_numpy(dtype=int)
        groups.append({"match_id": match_id, "date": game.date.iloc[0],
                       "accuracy_gain": float(np.mean((candidate.argmax(1) == truth).astype(float)
                                                      - (baseline.argmax(1) == truth))),
                       "log_loss_gain": float(np.mean(np.log(candidate[np.arange(2), truth])
                                                      - np.log(baseline[np.arange(2), truth])))})
    grouped = pd.DataFrame(groups)
    rng = np.random.default_rng(42)
    weeks = grouped.date.dt.to_period("W-SUN").astype(str)
    intervals = {}
    for name in ["accuracy_gain", "log_loss_gain"]:
        per_week = grouped.groupby(weeks)[name].agg(["sum", "count"])
        draws = rng.integers(0, len(per_week), size=(10000, len(per_week)))
        gains = per_week["sum"].to_numpy()[draws].sum(1) / per_week["count"].to_numpy()[draws].sum(1)
        intervals[name] = {"value": float(grouped[name].mean()), "ci_95_low": float(np.quantile(gains, .025)),
                           "ci_95_high": float(np.quantile(gains, .975))}
    home = frame[frame.home_row]
    return {"uniform_random_expected_accuracy": 1 / 3, "always_home_accuracy": float(home.actual_points.eq(3).mean()),
            "old_reported_team_row_accuracy": float((frame[[f"formation_model_p_{x}" for x in ["loss", "draw", "win"]]].to_numpy().argmax(1)
                                                      == frame.actual_points.map({0: 0, 1: 1, 3: 2}).to_numpy()).mean()),
            "matches": len(home), "team_rows": len(frame), "paired_week_intervals": intervals,
            "interpretation": "52.5% is above uniform random, but only a small margin over always-home. "
                              "120 team rows are not 120 independent games. No validation of a formation policy."}
