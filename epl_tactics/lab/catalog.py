"""Korean names, groups and short explanations for every model input column."""
from __future__ import annotations

from ..focused_features import CATEGORIES, FORMATION_NUMERIC, HISTORY_STATS

STAT_LABEL = {
    "points": ("승점", "점"), "goals_for": ("득점", "골"), "goals_against": ("실점", "골"),
    "shots": ("슈팅", "회"), "shots_against": ("허용 슈팅", "회"), "sot": ("유효슈팅", "회"),
    "sot_against": ("허용 유효슈팅", "회"), "possession": ("점유율", "%"), "corners": ("코너킥", "회"),
    "fouls": ("파울", "회"), "yellow": ("옐로카드", "장"), "red": ("레드카드", "장"), "passes": ("패스", "회"),
    "accurate_passes": ("정확한 패스", "회"), "crosses": ("크로스", "회"), "long_balls": ("롱볼", "회"),
    "tackles": ("태클", "회"), "interceptions": ("가로채기", "회"), "saves": ("선방", "회"),
    "blocked_shots": ("블록", "회"),
}
SIDE_LABEL = {"team": "우리 팀", "opponent": "상대 팀"}
GROUPS = [
    ("elo", "Elo 전력", "경기 전 두 팀의 종합 전력 점수와 그 차이"),
    ("form5", "최근 5경기", "직전 5경기의 평균 성적·기록"),
    ("form10", "최근 10경기", "직전 10경기의 평균 성적·기록"),
    ("form20", "최근 20경기", "직전 20경기의 평균 성적·기록과 기록 확보 비율"),
    ("venue", "홈/원정", "홈·원정 구분과 같은 장소 최근 10경기 성적"),
    ("schedule", "일정·경기 수", "휴식일, 최근 경기 밀도, 누적 경기 수"),
    ("lineup", "선발 11명", "발표된 선발 구성과 이전 경기에서 관측한 선수 기록"),
    ("formation", "포메이션", "선발 포메이션의 모양·사용 빈도·변경 여부"),
    ("identity", "팀·상대 이름", "팀 이름 자체(범주형)"),
]
GROUP_LABEL = {key: label for key, label, _ in GROUPS}
LINEUP_LABEL = {
    "complete": ("선발 11명 확보 여부", "0/1", "binary", "선발 명단이 11명 모두 기록되었는지(1=예)"),
    "size": ("확인된 선발 인원", "명", "count", "기록에서 확인된 선발 선수 수"),
    "returning": ("직전 경기 선발 유지 인원", "명", "count", "직전 경기 선발 11명 중 이번에도 선발인 선수 수"),
    "prior_players_known": ("과거 기록이 있는 선발 수", "명", "count", "이전 경기 기록이 이미 수집된 선발 선수 수"),
    "observed_experience": ("선발 평균 관측 출전 수", "경기", "rate", "선발 선수들의 이전 수집 경기 출전 수 평균(최대 20)"),
    "prior_goals_rate": ("선발 평균 득점률", "골/경기", "rate", "선발 선수들의 이전 경기당 득점률(소표본 보정)"),
    "prior_assists_rate": ("선발 평균 도움률", "개/경기", "rate", "선발 선수들의 이전 경기당 도움률(소표본 보정)"),
    "prior_shots_rate": ("선발 평균 슈팅률", "회/경기", "rate", "선발 선수들의 이전 경기당 슈팅률(소표본 보정)"),
    "prior_sot_rate": ("선발 평균 유효슈팅률", "회/경기", "rate", "선발 선수들의 이전 경기당 유효슈팅률(소표본 보정)"),
}
FORMATION_LABEL = {
    "formation_known": ("포메이션 기록 여부", "0/1", "binary", "이 경기 선발 포메이션이 기록되어 있는지(1=예)"),
    "defenders": ("수비수 수", "명", "count", "포메이션 첫 숫자(예: 4-2-3-1 → 4)"),
    "midfielders": ("미드필더 수", "명", "count", "포메이션 가운데 숫자의 합(예: 4-2-3-1 → 5)"),
    "forwards": ("공격수 수", "명", "count", "포메이션 마지막 숫자(예: 4-2-3-1 → 1)"),
    "formation_lines": ("포메이션 라인 수", "줄", "count", "포메이션 숫자 개수(4-2-3-1 → 4줄)"),
    "formation_used_last20": ("최근 20경기 같은 포메이션 사용률", "비율", "rate", "최근 20경기(기록 있는 것만) 중 이번과 같은 포메이션 비율"),
    "formation_switch": ("직전 경기 대비 포메이션 변경", "0/1", "binary", "직전 경기와 다른 포메이션이면 1"),
}
CATEGORY_LABEL = {
    "team": ("우리 팀", "분석 대상 6개 구단 중 하나"),
    "opponent": ("상대 팀", "상대 팀 이름"),
    "formation": ("선발 포메이션", "이 경기 우리 팀의 선발 포메이션(기록 없음은 __missing__)"),
    "team_formation": ("팀+포메이션 조합", "팀 이름과 포메이션을 합친 값"),
    "previous_formation": ("직전 경기 포메이션", "우리 팀 직전 경기의 선발 포메이션"),
}
SCHEDULE_LABEL = {
    "elo": ("Elo 전력 점수", "점", "rate", "경기 전까지의 승패로 갱신한 종합 전력 점수(시작값 1500, 시즌마다 1500 쪽으로 25% 수렴)"),
    "season_games": ("이번 시즌 치른 경기 수", "경기", "count", "경기 전까지 이번 시즌에 치른 리그 경기 수"),
    "history_games": ("수집된 누적 경기 수", "경기", "count", "수집 기간(2006/07~) 동안 경기 전까지 치른 경기 수"),
    "rest_days": ("직전 경기 후 휴식일", "일", "count", "직전 경기 이후 지난 날수(최대 90일, 기록 없으면 7일)"),
    "games_last7": ("최근 7일 경기 수", "경기", "count", "경기 전 7일 안에 치른 경기 수"),
    "games_last14": ("최근 14일 경기 수", "경기", "count", "경기 전 14일 안에 치른 경기 수"),
}
VENUE_STATS = {"points": "승점", "goals_for": "득점", "goals_against": "실점", "shots": "슈팅", "sot": "유효슈팅"}
DIFF_LABEL = {
    "elo": ("Elo 차이(우리−상대)", "점", "elo"), "points_5": ("최근 5경기 평균 승점 차이", "점", "form5"),
    "goals_for_20": ("최근 20경기 평균 득점 차이", "골", "form20"), "goals_against_20": ("최근 20경기 평균 실점 차이", "골", "form20"),
    "shots_20": ("최근 20경기 평균 슈팅 차이", "회", "form20"), "sot_20": ("최근 20경기 평균 유효슈팅 차이", "회", "form20"),
    "rest_days": ("휴식일 차이", "일", "schedule"),
}


def _entry(name, label, group, side, kind, unit="", desc=""):
    return {"name": name, "label": label, "group": group, "side": side, "kind": kind, "unit": unit, "desc": desc}


def describe(name: str) -> dict:
    """Catalog entry for one raw model input column; raises on an unknown column so new features are never unlabeled."""
    if name in CATEGORY_LABEL:
        label, desc = CATEGORY_LABEL[name]
        group = "formation" if name != "team" and name != "opponent" else "identity"
        return _entry(name, label, group, "match", "category", "", desc)
    if name == "is_home":
        return _entry(name, "홈 경기 여부", "venue", "match", "binary", "0/1", "1이면 홈, 0이면 원정")
    if name in FORMATION_LABEL:
        label, unit, kind, desc = FORMATION_LABEL[name]
        return _entry(name, label, "formation", "team", kind, unit, desc)
    if name.startswith("diff_"):
        label, unit, group = DIFF_LABEL[name[5:]]
        return _entry(name, label, group, "diff", "rate", unit, "우리 팀 값에서 상대 팀 값을 뺀 차이(양수면 우리 팀이 우세)")
    side, _, rest = name.partition("_")
    if side not in SIDE_LABEL:
        raise KeyError(f"No catalog entry for input column {name!r}")
    who = SIDE_LABEL[side]
    if rest.startswith("lineup_"):
        label, unit, kind, desc = LINEUP_LABEL[rest[7:]]
        return _entry(name, f"{who} · {label}", "lineup", side, kind, unit, desc)
    if rest in SCHEDULE_LABEL:
        label, unit, kind, desc = SCHEDULE_LABEL[rest]
        group = "elo" if rest == "elo" else "schedule"
        return _entry(name, f"{who} · {label}", group, side, kind, unit, desc)
    if rest.startswith("venue_"):
        stat = rest[6:]
        place = "홈 경기" if side == "team" else "같은 장소 경기"
        return _entry(name, f"{who} · 같은 장소(홈/원정) 최근 10경기 평균 {VENUE_STATS[stat]}", "venue", side, "rate",
                      STAT_LABEL[stat][1], "이번 경기와 같은 장소(홈이면 홈 경기만)의 최근 10경기 평균")
    if rest.endswith("_coverage20"):
        stat = rest[: -len("_coverage20")]
        return _entry(name, f"{who} · 최근 20경기 중 {STAT_LABEL[stat][0]} 기록 확보 비율", "form20", side, "rate", "비율",
                      "최근 20경기 중 이 항목이 실제로 기록된 경기의 비율(0~1)")
    stat_part, _, window = rest.rpartition("_")
    if stat_part in STAT_LABEL and window in {"5", "10", "20"}:
        label, unit = STAT_LABEL[stat_part]
        return _entry(name, f"{who} · 최근 {window}경기 평균 {label}", f"form{window}", side, "rate", unit,
                      f"경기 전까지 {who}의 최근 {window}경기 평균 {label}(기록이 있는 경기만)")
    raise KeyError(f"No catalog entry for input column {name!r}")


def build_catalog(numeric: list[str]) -> list[dict]:
    names = [*CATEGORIES, "is_home", *[n for n in numeric if n != "is_home"]]
    seen, result = set(), []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        result.append(describe(name))
    return result


__all__ = ["build_catalog", "describe", "GROUPS", "GROUP_LABEL", "STAT_LABEL", "FORMATION_NUMERIC", "HISTORY_STATS"]
