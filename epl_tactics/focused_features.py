"""Pre-date process, schedule, formation and lineup features for the six-club cohort."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .focused_data import CLUBS, TEAM_STATS

HISTORY_STATS = ["points", "goals_for", "goals_against", "shots", "shots_against", "sot", "sot_against",
                 "possession", "corners", "fouls", "yellow", "red", "passes", "accurate_passes",
                 "crosses", "long_balls", "tackles", "interceptions", "saves", "blocked_shots"]
FORMATION_NUMERIC = ["formation_known", "defenders", "midfielders", "forwards", "formation_lines",
                     "formation_used_last20", "formation_switch"]
CATEGORIES = ["team", "opponent", "formation", "team_formation", "previous_formation"]


@dataclass
class ClubHistory:
    records: list[dict] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)
    venues: list[bool] = field(default_factory=list)
    formations: list[str] = field(default_factory=list)
    lineup: set[str] = field(default_factory=set)
    player_stats: dict[str, list[dict]] = field(default_factory=dict)
    elo: float = 1500.0
    season: str | None = None
    season_games: int = 0

    def predate(self, date: pd.Timestamp, home: bool) -> dict:
        output = {"elo": self.elo, "season_games": self.season_games, "history_games": len(self.records),
                  "rest_days": min((date-self.dates[-1]).days, 90) if self.dates else 7,
                  "games_last7": sum((date-d).days <= 7 for d in self.dates[-10:]),
                  "games_last14": sum((date-d).days <= 14 for d in self.dates[-10:])}
        for window in [5, 10, 20]:
            recent = self.records[-window:]
            for stat in HISTORY_STATS:
                values = [r[stat] for r in recent if pd.notna(r.get(stat))]
                output[f"{stat}_{window}"] = float(np.mean(values)) if values else np.nan
                if window == 20:
                    output[f"{stat}_coverage20"] = len(values)/max(1, len(recent))
        venue = [record for record, h in zip(self.records, self.venues) if h == home][-10:]
        for stat in ["points", "goals_for", "goals_against", "shots", "sot"]:
            values = [r[stat] for r in venue if pd.notna(r.get(stat))]
            output[f"venue_{stat}"] = float(np.mean(values)) if values else np.nan
        return output


def _players(raw) -> list[dict]:
    if not isinstance(raw, str) or not raw:
        return []
    return json.loads(raw)


def _lineup_features(players: list[dict], history: ClubHistory) -> dict:
    starting = [p for p in players if p.get("starter")]
    ids = {str(p["id"]) for p in starting}
    known = [history.player_stats[pid][-20:] for pid in ids if pid in history.player_stats]
    output = {"lineup_complete": int(len(ids) == 11), "lineup_size": len(ids),
              "lineup_returning": len(ids & history.lineup) if len(ids) == 11 and history.lineup else np.nan,
              "lineup_prior_players_known": len(known)}
    # These are previously observed appearances in collected fixtures, not career totals.
    output["lineup_observed_experience"] = float(np.mean([len(records) for records in known])) if known else np.nan
    for stat in ["goals", "assists", "shots", "sot"]:
        rates = []
        for records in known:
            values = [r.get(stat) for r in records if pd.notna(r.get(stat))]
            if values:
                rates.append((sum(values) + 0.1*5)/(len(values)+5))
        output[f"lineup_prior_{stat}_rate"] = float(np.mean(rates)) if rates else np.nan
    return output


def build_focused_features(data_dir: str | Path = "data/focused") -> tuple[pd.DataFrame, list[str], dict]:
    root = Path(data_dir)
    league = pd.read_csv(root / "league_context.csv")
    detailed = pd.read_csv(root / "matches.csv").set_index("match_id")
    for column in detailed.columns:
        if column in ["date", "season", "home_team", "away_team", "home_goals", "away_goals"]:
            continue
        mapped = league.match_id.map(detailed[column])
        league[column] = mapped.combine_first(league[column]) if column in league else mapped
    league.date = pd.to_datetime(league.date)
    states: dict[str, ClubHistory] = {}
    rows = []
    for date, day in league.sort_values(["date", "match_id"]).groupby("date", sort=True):
        updates = []
        for match in day.to_dict("records"):
            home = states.setdefault(match["home_team"], ClubHistory())
            away = states.setdefault(match["away_team"], ClubHistory())
            for state in [home, away]:
                if state.season != match["season"]:
                    state.elo = 1500 + 0.75*(state.elo-1500)
                    state.season_games = 0
                    state.season = match["season"]
            hg, ag = match["home_goals"], match["away_goals"]
            home_points, away_points = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)
            fixture_updates = []
            for side, other_side, own, other, points, gf, ga in [
                ("home", "away", home, away, home_points, hg, ag),
                ("away", "home", away, home, away_points, ag, hg),
            ]:
                team, opponent = match[f"{side}_team"], match[f"{other_side}_team"]
                players = _players(match.get(f"{side}_players"))
                opp_players = _players(match.get(f"{other_side}_players"))
                formation = match.get(f"{side}_formation")
                formation = str(formation) if pd.notna(formation) else "__missing__"
                if team in CLUBS:
                    row = {"match_id": match["match_id"], "date": date, "season": match["season"],
                           "team": team, "opponent": opponent, "formation": formation,
                           "team_formation": f"{team}|{formation}", "previous_formation": own.formations[-1] if own.formations else "__missing__",
                           "is_home": int(side == "home"), "outcome": {0:0, 1:1, 3:2}[points],
                           "goals_for": gf, "goals_against": ga}
                    own_context, other_context = own.predate(date, side == "home"), other.predate(date, side != "home")
                    for prefix, context in [("team", own_context), ("opponent", other_context)]:
                        row.update({f"{prefix}_{key}": value for key,value in context.items()})
                    for key in ["elo", "points_5", "goals_for_20", "goals_against_20", "shots_20", "sot_20", "rest_days"]:
                        row[f"diff_{key}"] = own_context[key] - other_context[key]
                    for prefix, lineup in [("team", _lineup_features(players, own)), ("opponent", _lineup_features(opp_players, other))]:
                        row.update({f"{prefix}_{key}": value for key,value in lineup.items()})
                    row["formation_known"] = int(formation != "__missing__")
                    parts = [int(p) for p in formation.split("-")] if row["formation_known"] else []
                    # Unknown past formations are not evidence of use or of a switch.
                    known_recent = [f for f in own.formations[-20:] if f != "__missing__"]
                    previous_known = bool(own.formations) and own.formations[-1] != "__missing__"
                    row.update(defenders=parts[0] if parts else np.nan, forwards=parts[-1] if parts else np.nan,
                               midfielders=sum(parts[1:-1]) if parts else np.nan, formation_lines=len(parts) if parts else np.nan,
                               formation_used_last20=sum(f == formation for f in known_recent)/len(known_recent)
                               if parts and known_recent else np.nan,
                               formation_switch=int(formation != own.formations[-1]) if parts and previous_known else np.nan)
                    for label, reverse_label in [("loss", "win"), ("draw", "draw"), ("win", "loss")]:
                        row[f"market_{label}"] = match.get(f"market_{label if side == 'home' else reverse_label}", np.nan)
                    rows.append(row)
                record = {"points": points, "goals_for": gf, "goals_against": ga,
                          "shots_against": match.get(f"{other_side}_shots", np.nan),
                          "sot_against": match.get(f"{other_side}_sot", np.nan)}
                record.update({name: match.get(f"{side}_{name}", np.nan) for name in TEAM_STATS})
                fixture_updates.append((own, record, side == "home", formation, players))
            expectation = 1/(1+10**((away.elo-home.elo-65)/400))
            delta = 20*((1 if hg>ag else 0 if hg<ag else .5)-expectation)
            updates.append((home, away, delta, fixture_updates))
        for home, away, delta, fixture_updates in updates:
            home.elo += delta
            away.elo -= delta
            for state, record, venue, formation, players in fixture_updates:
                state.records.append(record)
                state.dates.append(date)
                state.venues.append(venue)
                state.formations.append(formation)
                state.season_games += 1
                starting = {str(p["id"]) for p in players if p.get("starter")}
                if len(starting) == 11:
                    state.lineup = starting
                for player in players:
                    if player.get("appearance") and player["appearance"] > 0:
                        state.player_stats.setdefault(str(player["id"]), []).append(player)
    frame = pd.DataFrame(rows)
    frame["fixture_weight"] = 1/frame.groupby("match_id").match_id.transform("size")
    excluded = {"match_id", "date", "season", "outcome", "goals_for", "goals_against", "fixture_weight", *CATEGORIES,
                "market_loss", "market_draw", "market_win"}
    numeric = [column for column in frame.columns if column not in excluded]
    metadata = {"clubs": CLUBS, "supervised_team_rows": len(frame), "unique_matches": frame.match_id.nunique(),
                "numeric_features": numeric, "categorical_features": CATEGORIES,
                "missing_fraction": frame[numeric].isna().mean().to_dict(),
                "same_match_excluded": ["goals", "shots", "possession", "cards", "player match stats", "odds"],
                "lineup_note": "Current starting identities only; player productivity from earlier observed selected-club fixtures. No career/ability or injury claims.",
                "context_note": "All EPL opponent result histories are context, never supervised non-cohort team rows."}
    return frame, numeric, metadata
