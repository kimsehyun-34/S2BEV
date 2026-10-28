"""Strictly pre-date features; updates occur only after every match on a date."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CATEGORICAL = ["team", "opponent", "formation"]
STATE_FIELDS = ["points_5", "scored_5", "conceded_5", "elo", "rest_days", "games_played"]
NUMERIC = ["is_home"] + [f"{side}_{name}" for side in ["team", "opponent"] for name in STATE_FIELDS]
FEATURES = CATEGORICAL + NUMERIC


@dataclass
class TeamState:
    elo: float = 1500.0
    history: list[tuple[float, int, int]] = field(default_factory=list)
    last_date: pd.Timestamp | None = None

    def features(self, date: pd.Timestamp) -> dict[str, float]:
        recent = self.history[-5:]
        averages = np.mean(recent, axis=0) if recent else [1.0, 1.3, 1.3]
        rest = (date - self.last_date).days if self.last_date is not None else 7
        if rest < 0:
            raise ValueError("Prediction date precedes historical team state")
        return dict(zip(STATE_FIELDS, [*map(float, averages), self.elo,
                                      float(min(rest, 30)), float(len(self.history))]))


def context_row(team: str, opponent: str, formation: str, is_home: bool,
                team_state: TeamState, opponent_state: TeamState, date: pd.Timestamp) -> dict:
    row = {"team": team, "opponent": opponent, "formation": formation, "is_home": int(is_home)}
    for side, state in [("team", team_state), ("opponent", opponent_state)]:
        row.update({f"{side}_{key}": value for key, value in state.features(date).items()})
    return row


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], TeamState]]:
    states: dict[tuple[str, str], TeamState] = {}
    rows = []
    for date, day in matches.sort_values(["date", "match_id"]).groupby("date", sort=True):
        updates = []
        for match in day.itertuples(index=False):
            home = states.setdefault((match.season, match.home_team), TeamState())
            away = states.setdefault((match.season, match.away_team), TeamState())
            hg, ag = int(match.home_goals), int(match.away_goals)
            hp, ap = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)
            for team, opponent, formation, is_home, own, other, points in [
                (match.home_team, match.away_team, match.home_formation, True, home, away, hp),
                (match.away_team, match.home_team, match.away_formation, False, away, home, ap),
            ]:
                row = context_row(team, opponent, formation, is_home, own, other, date)
                row.update(match_id=match.match_id, date=date, season=match.season,
                           outcome={0: 0, 1: 1, 3: 2}[points], actual_points=points)
                rows.append(row)
            expected_home = 1 / (1 + 10 ** ((away.elo - home.elo - 65) / 400))
            actual_home = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
            delta = 20 * (actual_home - expected_home)
            updates.append((home, away, hp, ap, hg, ag, delta))
        for home, away, hp, ap, hg, ag, delta in updates:
            home.elo += delta
            away.elo -= delta
            home.history.append((hp, hg, ag))
            away.history.append((ap, ag, hg))
            home.last_date = away.last_date = date
    return pd.DataFrame(rows), states


def chronological_split(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(rows.date.unique())
    if len(dates) < 20 or rows.match_id.nunique() < 40:
        raise ValueError("Need at least 40 matches on 20 dates for train/validation/test evaluation")
    first, second = int(len(dates) * 0.6), int(len(dates) * 0.8)
    train = rows[rows.date < dates[first]].copy()
    validation = rows[(rows.date >= dates[first]) & (rows.date < dates[second])].copy()
    test = rows[rows.date >= dates[second]].copy()
    if set(train.outcome) != {0, 1, 2}:
        raise ValueError("Training period must contain wins, draws and losses")
    return train, validation, test
