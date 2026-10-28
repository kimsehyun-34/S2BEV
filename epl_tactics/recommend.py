"""Rank observed formation candidates while making data support explicit."""

from __future__ import annotations

import pandas as pd

from .features import FEATURES, context_row
from .model import probabilities


def _setup(bundle: dict, as_of: str | None) -> tuple[str, list[str], pd.Timestamp]:
    data = bundle["report"]["data"]
    through = pd.Timestamp(data["trained_through"])
    date = pd.Timestamp(as_of).normalize() if as_of else through + pd.Timedelta(days=7)
    if pd.isna(date) or date.tzinfo is not None or date <= through:
        raise ValueError(f"Prediction date must be a local date after {through.date()}; historical use requires retraining")
    return data["recommendation_season"], data["teams"], date


def _summarize(bundle: dict, frame: pd.DataFrame, team: str, date: pd.Timestamp,
               context: str, min_team_support: int, min_league_support: int) -> dict:
    if min_team_support < 1 or min_league_support < 1:
        raise ValueError("Support thresholds must be positive")
    proba = probabilities(bundle["pipeline"], frame[FEATURES])
    frame = frame.copy()
    for i, label in enumerate(["loss", "draw", "win"]):
        frame[f"p_{label}"] = proba[:, i]
    ranked = frame.groupby("formation")[["p_loss", "p_draw", "p_win"]].mean().reset_index()
    ranked["expected_points"] = ranked.p_draw + 3 * ranked.p_win
    support = bundle["support"].query("team == @team").set_index("formation")["matches"]
    ranked["team_matches"] = ranked.formation.map(support).fillna(0).astype(int)
    ranked["league_team_matches"] = ranked.formation.map(bundle["league_support"]).fillna(0).astype(int)
    ranked["supported"] = ((ranked.team_matches >= min_team_support)
                           & (ranked.league_team_matches >= min_league_support))
    ranked = ranked.sort_values(["expected_points", "formation"], ascending=[False, True]).reset_index(drop=True)
    eligible = ranked[ranked.supported]
    best = eligible.iloc[0].formation if len(eligible) else None
    gap = float(eligible.iloc[0].expected_points - eligible.iloc[1].expected_points) if len(eligible) > 1 else None
    status = ("insufficient_support" if not len(eligible) else "single_supported_formation"
              if len(eligible) == 1 else "near_tie" if gap < 0.03 else "candidate_recommendation")
    comparison_status = status
    exploratory_formation = best
    if not bundle["report"].get("tactical_recommendation_validated", False):
        best = None
        status = "not_validated"
    warnings = ["관측 데이터에 따른 후보 순위이며, 최적 전술이나 실제 승점 향상을 보장하지 않습니다."]
    if not bundle["report"]["formation_improves_test_log_loss"]:
        warnings.append("검증에서 포메이션 추가 모델이 포메이션 제외 모델보다 테스트 로그손실이 좋지 않았습니다.")
    if comparison_status == "single_supported_formation":
        warnings.append("사용 사례 기준을 충족한 대형이 하나뿐이어서 대형 간 비교 근거가 부족합니다.")
    if comparison_status == "near_tie":
        warnings.append("상위 두 대형의 기대 승점 차이가 0.03 미만으로 작습니다. 통계적 유의성을 뜻하지 않습니다.")
    through = pd.Timestamp(bundle["report"]["data"]["trained_through"])
    if (date - through).days > 60:
        warnings.append("학습 데이터 종료일에서 60일 이상 지난 예측입니다. 최신 데이터로 재학습해야 합니다.")
    return {"team": team, "season": bundle["report"]["data"]["recommendation_season"],
            "as_of": str(date.date()), "context": context, "recommended_formation": best,
            "status": status, "comparison_status": comparison_status,
            "exploratory_formation": exploratory_formation, "gap_to_next_supported": gap,
            "support_floor": {"team": min_team_support, "league_team_rows": min_league_support},
            "warnings": warnings, "candidates": ranked.to_dict(orient="records")}


def recommend_match(bundle: dict, team: str, opponent: str, is_home: bool = True,
                    as_of: str | None = None, min_team_support: int = 3,
                    min_league_support: int = 10) -> dict:
    season, teams, date = _setup(bundle, as_of)
    if team not in teams or opponent not in teams:
        raise ValueError(f"Unknown team for {season}; available: {', '.join(teams)}")
    if team == opponent:
        raise ValueError("A team cannot play itself")
    rows = [context_row(team, opponent, formation, is_home, bundle["states"][(season, team)],
                        bundle["states"][(season, opponent)], date)
            for formation in sorted(bundle["league_support"])]
    return _summarize(bundle, pd.DataFrame(rows), team, date,
                      f"vs {opponent}; {'home' if is_home else 'away'}", min_team_support, min_league_support)


def recommend_all(bundle: dict, as_of: str | None = None, min_team_support: int = 3,
                  min_league_support: int = 10) -> list[dict]:
    season, teams, date = _setup(bundle, as_of)
    results = []
    for team in teams:
        rows = [context_row(team, opponent, formation, is_home, bundle["states"][(season, team)],
                            bundle["states"][(season, opponent)], date)
                for opponent in teams if opponent != team
                for is_home in [False, True] for formation in sorted(bundle["league_support"])]
        context = f"Equal average over {len(teams) - 1} opponents and home/away; historical snapshot"
        results.append(_summarize(bundle, pd.DataFrame(rows), team, date, context,
                                  min_team_support, min_league_support))
    return results
