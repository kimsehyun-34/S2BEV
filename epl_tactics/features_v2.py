"""Continuous, shrunk team histories with daily updates and no contemporaneous stats."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_STATS = ["points", "scored", "conceded", "shots", "shots_against", "sot", "sot_against", "possession"]
PRIORS = np.array([1.35, 1.4, 1.4, 12.5, 12.5, 4.3, 4.3, 50.0])
HISTORY_FIELDS = [f"{name}_{window}" for window in [5, 20] for name in BASE_STATS]
HISTORY_FIELDS += ["elo", "rest_days", "season_games", "history_games", "venue_scored", "venue_conceded"]
NUMERIC_V2 = ["is_home"] + [f"{side}_{name}" for side in ["team", "opponent"] for name in HISTORY_FIELDS]
NUMERIC_V2 += [f"diff_{name}" for name in HISTORY_FIELDS if name not in ["rest_days", "history_games"]]
CATEGORICAL_V2 = ["team", "opponent", "formation"]
FEATURES_V2 = CATEGORICAL_V2 + NUMERIC_V2


@dataclass
class History:
    elo: float = 1500.0
    records: list[np.ndarray] = field(default_factory=list)
    venues: list[bool] = field(default_factory=list)
    last_date: pd.Timestamp | None = None
    season: str | None = None
    season_games: int = 0

    def start_season(self, season: str):
        if self.season != season:
            if self.season is not None:
                self.elo = 1500 + 0.75 * (self.elo - 1500)
            self.season = season
            self.season_games = 0

    def features(self, date: pd.Timestamp, is_home: bool) -> dict:
        output = {}
        for window in [5, 20]:
            recent = self.records[-window:]
            # Five equivalent prior matches prevent extreme estimates at season/data start.
            average = (np.sum(recent, axis=0) + 5 * PRIORS) / (len(recent) + 5) if recent else PRIORS
            output.update({f"{name}_{window}": float(average[i]) for i, name in enumerate(BASE_STATS)})
        venue = [record for record, home in zip(self.records, self.venues) if home == is_home][-10:]
        venue_average = (np.sum(venue, axis=0) + 5 * PRIORS) / (len(venue) + 5) if venue else PRIORS
        rest = (date - self.last_date).days if self.last_date is not None else 7
        if rest < 0:
            raise ValueError("Cannot use future team history")
        output.update(elo=self.elo, rest_days=float(min(rest, 30)), season_games=float(self.season_games),
                      history_games=float(min(len(self.records), 100)), venue_scored=float(venue_average[1]),
                      venue_conceded=float(venue_average[2]))
        return output


def context_v2(team: str, opponent: str, formation: str, is_home: bool, own: History,
               other: History, date: pd.Timestamp) -> dict:
    row = dict(team=team, opponent=opponent, formation=formation, is_home=int(is_home))
    own_features, other_features = own.features(date, is_home), other.features(date, not is_home)
    for side, features in [("team", own_features), ("opponent", other_features)]:
        row.update({f"{side}_{key}": value for key, value in features.items()})
    for key in HISTORY_FIELDS:
        if f"diff_{key}" in NUMERIC_V2:
            row[f"diff_{key}"] = own_features[key] - other_features[key]
    return row


def _number(match, name: str, default: float) -> float:
    value = getattr(match, name, default)
    return float(value) if pd.notna(value) and np.isfinite(value) else default


def build_features_v2(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, History]]:
    states: dict[str, History] = {}
    rows = []
    for date, day in matches.sort_values(["date", "match_id"]).groupby("date", sort=True):
        updates = []
        for match in day.itertuples(index=False):
            home = states.setdefault(match.home_team, History())
            away = states.setdefault(match.away_team, History())
            home.start_season(match.season)
            away.start_season(match.season)
            hg, ag = int(match.home_goals), int(match.away_goals)
            hp, ap = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)
            for team, opponent, formation, is_home, own, other, points, gf, ga in [
                (match.home_team, match.away_team, match.home_formation, True, home, away, hp, hg, ag),
                (match.away_team, match.home_team, match.away_formation, False, away, home, ap, ag, hg),
            ]:
                row = context_v2(team, opponent, formation, is_home, own, other, date)
                row.update(match_id=str(match.match_id), date=date, season=match.season,
                           outcome={0: 0, 1: 1, 3: 2}[points], goals_for=gf, goals_against=ga)
                rows.append(row)
            expected = 1 / (1 + 10 ** ((away.elo - home.elo - 65) / 400))
            delta = 20 * ((1 if hg > ag else 0 if hg < ag else 0.5) - expected)
            hs, aws = _number(match, "home_shots", 12.5), _number(match, "away_shots", 12.5)
            hso, aso = _number(match, "home_shots_on_target", 4.3), _number(match, "away_shots_on_target", 4.3)
            hpos, apos = _number(match, "home_possession", 50), _number(match, "away_possession", 50)
            updates.append((home, away, delta, [hp, hg, ag, hs, aws, hso, aso, hpos],
                            [ap, ag, hg, aws, hs, aso, hso, apos]))
        # No match played on this date influences any other match's features on the date.
        for home, away, delta, hrecord, arecord in updates:
            home.elo += delta
            away.elo -= delta
            for state, record, venue in [(home, hrecord, True), (away, arecord, False)]:
                state.records.append(np.array(record, dtype=float))
                state.venues.append(venue)
                state.last_date = date
                state.season_games += 1
    return pd.DataFrame(rows), states
