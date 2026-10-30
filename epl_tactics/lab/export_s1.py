"""Step 1 payload: where the data came from, what one row means, the target (패/무/승) and the model inputs.

Every number the page shows is computed here from the source tables (matches.csv, league_context.csv,
coverage.csv, the feature rows and metrics.json); the HTML fragment never types a data-dependent number.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ..focused_data import TEAM_STATS
from ..focused_features import CATEGORIES
from .catalog import GROUPS, build_catalog
from .context import LabContext

CODES = {"Arsenal": "ARS", "Chelsea": "CHE", "Liverpool": "LIV", "Manchester City": "MCI",
         "Manchester United": "MUN", "Tottenham Hotspur": "TOT"}

# ---------------------------------------------------------------- raw table dictionary (data/focused/matches.csv)
RAW_GROUPS = [
    ("id", "경기 식별", "어느 경기인지 알려 주는 값"),
    ("score", "점수", "최종 득점. 정답(패·무·승)은 이 두 숫자로 정해집니다"),
    ("events", "슈팅·카드·코너", "경기 중에 쌓인 기본 기록"),
    ("odds", "배당", "베팅 시장이 경기 전에 매긴 승·무·패 가능성"),
    ("teamstats", "팀 경기 기록", "ESPN이 집계한 점유율·패스·태클 등"),
    ("formation", "포메이션", "ESPN에 적힌 선발 포메이션"),
    ("players", "선수 명단", "선발·교체 선수와 선수별 그 경기 기록(JSON 글자)"),
]
SOURCE_LABEL = {"fd": "Football-Data", "espn": "ESPN", "both": "둘 다(ESPN 우선)", "made": "직접 계산"}
SIDE_LABEL = {"home": "홈 팀", "away": "원정 팀"}
# column -> (Korean label, group, source, short description)
RAW_SINGLE = {
    "date": ("경기 날짜", "id", "fd", "경기가 열린 날짜"),
    "season": ("시즌", "id", "fd", "8월에 시작해 이듬해 5월에 끝나는 시즌 이름"),
    "home_team": ("홈 팀", "id", "fd", "홈 팀 이름(두 출처의 표기를 하나로 맞춤)"),
    "away_team": ("원정 팀", "id", "fd", "원정 팀 이름(두 출처의 표기를 하나로 맞춤)"),
    "match_id": ("경기 번호", "id", "made", "시즌·홈 팀·원정 팀을 이어 만든 고유 번호"),
    "event_id": ("ESPN 경기 번호", "id", "espn", "ESPN에서 이 경기를 가리키는 번호"),
    "espn_verified": ("ESPN 대조 통과", "id", "made", "팀·득점·시즌·날짜가 두 출처에서 모두 같으면 참"),
    "market_win": ("배당 확률: 홈 승", "odds", "made", "배당률을 확률로 바꾼 값(세 확률의 합 = 1)"),
    "market_draw": ("배당 확률: 무승부", "odds", "made", "배당률을 확률로 바꾼 값(세 확률의 합 = 1)"),
    "market_loss": ("배당 확률: 홈 패(원정 승)", "odds", "made", "배당률을 확률로 바꾼 값(세 확률의 합 = 1)"),
    "market_columns": ("확률 계산에 쓴 배당 열", "odds", "made", "B365 = Bet365 배당, AvgC = 여러 업체의 마감 직전 평균 배당"),
}
RAW_PAIRED = {
    "goals": ("득점", "score", "fd", "최종 득점(ESPN 점수와 대조함)"),
    "shots": ("슈팅", "events", "both", "슈팅 수"),
    "sot": ("유효슈팅", "events", "both", "골문 안으로 향한 슈팅 수"),
    "corners": ("코너킥", "events", "both", "얻은 코너킥 수"),
    "fouls": ("파울", "events", "both", "범한 파울 수"),
    "yellow": ("옐로카드", "events", "both", "받은 경고 수"),
    "red": ("레드카드", "events", "both", "받은 퇴장 수"),
    "possession": ("점유율", "teamstats", "espn", "공을 가진 시간 비율(%)"),
    "passes": ("패스", "teamstats", "espn", "패스 시도 수"),
    "accurate_passes": ("정확한 패스", "teamstats", "espn", "동료에게 연결된 패스 수"),
    "crosses": ("크로스", "teamstats", "espn", "측면에서 올린 크로스 수"),
    "long_balls": ("롱볼", "teamstats", "espn", "길게 찬 패스 수"),
    "tackles": ("태클", "teamstats", "espn", "태클 수"),
    "interceptions": ("가로채기", "teamstats", "espn", "상대 패스를 끊어 낸 수"),
    "saves": ("선방", "teamstats", "espn", "골키퍼 선방 수"),
    "blocked_shots": ("블록", "teamstats", "espn", "몸으로 막아 낸 슈팅 수"),
    "formation": ("포메이션", "formation", "espn", "ESPN 표기를 ‘숫자-숫자’ 꼴로 정리한 값(없으면 빈 칸, 추측하지 않음)"),
    "formation_raw": ("포메이션 원문", "formation", "espn", "ESPN에 적힌 그대로의 표기"),
    "players": ("선수 명단", "players", "espn", "선수마다 {선수 번호, 선발 여부, 포지션, 득점, 도움, 슈팅, 유효슈팅, 출전}"),
}
ODDS_LABEL = {"B365": "Bet365 배당", "B365C": "Bet365 마감 배당", "Avg": "여러 업체 평균 배당",
              "AvgC": "여러 업체 마감 직전 평균 배당"}

# ---------------------------------------------------------------- model table: the columns that are NOT inputs
# column -> (Korean label, role, what it is used for, use code). None of these is a model input; `use` says what
# the pipeline does with it instead (focused_model.train_weights uses date + fixture_weight; outcome is the label).
MODEL_EXTRA = {
    "match_id": ("경기 번호", "other", "원시 표의 경기 번호와 같음(같은 경기의 두 줄을 묶는 열쇠)", "key"),
    "date": ("경기 날짜", "other", "시간 순서를 지키고, 학습할 때 최근 경기일수록 무게를 크게 주는 데 씀", "weight"),
    "season": ("시즌", "other", "시즌 단위로 검증·테스트 기간을 나누는 데 씀", "split"),
    "outcome": ("결과(정답)", "target", "우리 팀 기준 0 = 패, 1 = 무, 2 = 승. 모델이 맞히도록 배우는 답", "answer"),
    "goals_for": ("우리 팀 득점", "other", "결과를 만든 값이라 입력에서 제외", "result"),
    "goals_against": ("상대 팀 득점", "other", "결과를 만든 값이라 입력에서 제외", "result"),
    "fixture_weight": ("채점 가중치", "other", "보통 1, 6개 구단끼리 맞붙은 경기는 두 줄에 0.5씩. 채점과 학습 무게에 씀", "weight"),
    "market_loss": ("배당 확률: 우리 팀 패", "other", "비교용으로만 보관, 입력 아님", "compare"),
    "market_draw": ("배당 확률: 무승부", "other", "비교용으로만 보관, 입력 아님", "compare"),
    "market_win": ("배당 확률: 우리 팀 승", "other", "비교용으로만 보관, 입력 아님", "compare"),
}
# Columns that hold the row's own club's value (no team_/opponent_ partner) and columns that describe the match itself.
OWN_EXTRA = {"formation", "team_formation", "previous_formation"}
MATCH_LEVEL = {"match_id", "date", "season", "fixture_weight", "market_draw"}
SWAP_PAIRS = {"team": "opponent", "goals_for": "goals_against", "market_win": "market_loss"}
# Zero-coded stats: ESPN stored a missing team stat as 0. Cards are often genuinely 0, so they are not scanned.
ZERO_SCAN = [stat for stat in TEAM_STATS if stat not in {"yellow", "red"}]
STRUCTURAL_SHARE = 0.9      # a season where a stat is 0 for both teams in >= 90% of box-score matches = "not recorded"
PARTIAL_SHARE = 0.5         # >= 50% = "mostly not recorded"
MIN_BOX_MATCHES = 30        # seasons with fewer matches that have any ESPN box score cannot be judged
WINDOWS = (5, 10, 20)
KEY_MODEL_COLUMNS = ["match_id", "date", "team", "opponent", "is_home", "outcome", "goals_for", "goals_against",
                     "fixture_weight", "formation", "previous_formation", "team_elo", "opponent_elo", "diff_elo",
                     "team_points_5", "opponent_points_5", "team_shots_5", "team_lineup_returning",
                     "team_rest_days", "market_win"]

# features.json `same_match_excluded` key -> (what, why)
LEAK_TEXT = {
    "goals": ("그 경기의 득점·실점", "결과 그 자체입니다. 넣으면 답을 보고 문제를 푸는 셈입니다."),
    "shots": ("그 경기의 슈팅·유효슈팅·코너킥·파울", "경기가 끝나야 알 수 있는 값입니다."),
    "possession": ("그 경기의 점유율·패스·태클 같은 팀 기록", "경기 도중에 만들어지는 값입니다."),
    "cards": ("그 경기의 옐로·레드카드", "경기 도중에 나오고, 퇴장은 결과에 직접 영향을 줍니다."),
    "player match stats": ("선수별 그 경기 득점·도움·슈팅", "다음 경기부터 ‘이전 기록’으로만 씁니다."),
    "odds": ("배당(시장의 예측)", "경기 전에 공개되지만 다른 예측을 그대로 베끼는 셈이라 입력에서 빼고 비교용으로만 둡니다."),
}
# (same-match raw stem, pre-match model input) pairs for the concrete leakage example
LEAK_PAIRS = [("goals", "team_goals_for_5", "득점"), ("shots", "team_shots_5", "슈팅"),
              ("sot", "team_sot_5", "유효슈팅"), ("possession", "team_possession_5", "점유율")]
# Famous fixtures between two of the six clubs; the first one present with full formation + XI data is used.
EXAMPLE_PREFERENCES = [
    ("2014-04-27", "Liverpool", "Chelsea", "2013/14 시즌 막판 우승 경쟁 중에 열린 맞대결"),
    ("2014-03-22", "Chelsea", "Arsenal", "런던 팀끼리의 맞대결"),
    ("2016-05-02", "Chelsea", "Tottenham Hotspur", "런던 팀끼리의 맞대결"),
]


def r4(value):
    """JSON-safe rounded scalar."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return int(value) if value.is_integer() else round(value, 4)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def raw_dictionary() -> dict[str, dict]:
    out = {}
    for name, (label, group, source, desc) in RAW_SINGLE.items():
        out[name] = {"label": label, "group": group, "source": source, "desc": desc, "stem": name, "side": None}
    for stem, (label, group, source, desc) in RAW_PAIRED.items():
        for side, who in SIDE_LABEL.items():
            out[f"{side}_{stem}"] = {"label": f"{who} {label}", "group": group, "source": source, "desc": desc,
                                     "stem": stem, "side": side, "short": label}
    return out


def plain_type(name: str, series: pd.Series) -> str:
    if name.endswith("_players"):
        return "JSON 글자"
    if pd.api.types.is_datetime64_any_dtype(series) or name == "date":
        return "날짜"
    if pd.api.types.is_bool_dtype(series):
        return "참/거짓"
    if pd.api.types.is_integer_dtype(series):
        return "정수"
    if pd.api.types.is_float_dtype(series):
        return "실수"
    return "글자"


def _players(raw) -> list[dict]:
    return json.loads(raw) if isinstance(raw, str) and raw else []


def players_summary(raw) -> dict:
    players = _players(raw)
    starters = [p for p in players if p.get("starter")]
    sample = starters[0] if starters else (players[0] if players else None)
    return {"total": len(players), "starters": len(starters), "subs": len(players) - len(starters),
            "subs_used": sum(1 for p in players if not p.get("starter") and (p.get("appearance") or 0) > 0),
            "sample": {k: r4(v) for k, v in sample.items()} if sample else None}


def pick_example(ctx: LabContext, clubs: list[str]) -> dict:
    m = ctx.matches

    def complete(row) -> bool:
        return (row.home_team in clubs and row.away_team in clubs and pd.notna(row.home_formation)
                and pd.notna(row.away_formation)
                and sum(bool(p.get("starter")) for p in _players(row.home_players)) == 11
                and sum(bool(p.get("starter")) for p in _players(row.away_players)) == 11)

    for date, home, away, note in EXAMPLE_PREFERENCES:
        hit = m[(m.date == pd.Timestamp(date)) & m.home_team.eq(home) & m.away_team.eq(away)]
        if len(hit) and complete(hit.iloc[0]):
            return {"index": int(hit.index[0]), "note": note}
    # Fallback: the most one-sided derby with complete data (latest first on ties).
    derbies = m[m.home_team.isin(clubs) & m.away_team.isin(clubs)].copy()
    derbies["margin"] = (derbies.home_goals - derbies.away_goals).abs()
    for index, row in derbies.sort_values(["margin", "date"], ascending=False).iterrows():
        if complete(row):
            return {"index": int(index), "note": "6개 구단끼리의 맞대결"}
    raise ValueError("No complete derby between two of the clubs")


# ---------------------------------------------------------------- blocks
def kpis(ctx: LabContext, catalog: list[dict], clubs: list[str]) -> dict:
    rows = ctx.rows
    categories = [e["name"] for e in catalog if e["kind"] == "category"]
    train = np.asarray(ctx.train_mask, dtype=bool)
    return {"unique_matches": int(rows.match_id.nunique()), "rows": int(len(rows)), "inputs": len(catalog),
            "numeric": len(ctx.numeric), "categories": len(categories), "category_names": categories,
            "first_date": str(rows.date.min().date()), "last_date": str(rows.date.max().date()),
            "first_season": str(rows.season.min()), "last_season": str(rows.season.max()),
            "n_seasons": int(rows.season.nunique()), "clubs": len(clubs), "n_columns": int(rows.shape[1]),
            # Rows before the first test season train the model; the rest are the unseen test rows.
            "train_rows": int(train.sum()), "test_rows": int((~train).sum()), "test_start": ctx.test_start,
            # Official test scores refit before each test season on all earlier seasons (metrics.json protocol).
            "refit_each_test_season": "before each test season" in str(ctx.report["protocol"].get("test_refit", "")).lower()}


def _runs(selected: list[str], order: list[str]) -> list[list[str]]:
    """Compress seasons into [first, last] runs of seasons that are consecutive within `order` (ALL seasons, so a
    run never bridges a season that could not be judged: that season was not shown to be unrecorded)."""
    runs: list[list] = []
    for season in selected:
        index = order.index(season)
        if runs and index == runs[-1][2] + 1:
            runs[-1][1], runs[-1][2] = season, index
        else:
            runs.append([season, season, index])
    return [[first, last] for first, last, _ in runs]


def zero_coded_stats(ctx: LabContext) -> list[dict]:
    """Stats that ESPN never recorded in some seasons but which were stored as 0 instead of empty.

    A season is judged only on matches whose ESPN box score exists (some team has possession > 0), so a whole
    empty box score (counted separately) does not make every stat look unrecorded.
    """
    m, rows = ctx.matches, ctx.rows
    box = m.home_possession.gt(0) | m.away_possession.gt(0)
    seasons = sorted(m.season.unique())
    out = []
    for stat in ZERO_SCAN:
        home, away = m[f"home_{stat}"], m[f"away_{stat}"]
        both_zero = home.eq(0) & away.eq(0)
        judged = box & home.notna() & away.notna()
        almost, partial = [], []
        for season in seasons:
            part = judged & m.season.eq(season)
            if int(part.sum()) < MIN_BOX_MATCHES:
                continue
            share = float(both_zero[part].mean())
            if share >= STRUCTURAL_SHARE:
                almost.append(season)
            elif share >= PARTIAL_SHARE:
                partial.append({"season": season, "share": r4(share)})
        if not almost and not partial:
            continue
        inputs = {f"{side}_{stat}_{w}": r4(float(rows[f"{side}_{stat}_{w}"].eq(0).mean()))
                  for side in ["team", "opponent"] for w in WINDOWS if f"{side}_{stat}_{w}" in rows}
        out.append({"stat": stat, "label": RAW_PAIRED[stat][0], "almost": almost,
                    "almost_ranges": _runs(almost, seasons), "partial": partial,
                    "share_all": r4(float(both_zero.mean())), "rows_zero_5": inputs.get(f"team_{stat}_5"),
                    "inputs": inputs})
    out.sort(key=lambda item: -item["share_all"])
    return out


def data_quality(ctx: LabContext) -> dict:
    """Where the merged (ESPN-first) event counts disagree with Football-Data, and implausible zeros."""
    m = ctx.matches.set_index("match_id")
    fd = ctx.league.set_index("match_id").reindex(m.index)
    zero_over = pd.Series(False, index=m.index)
    differ = pd.Series(False, index=m.index)
    gaps = []
    for side in ["home", "away"]:
        merged, source = m[f"{side}_shots"], fd[f"{side}_shots"]
        zero_over |= merged.eq(0) & source.gt(0)
        # "Both sources have a real (non-zero) count for this team and the counts still differ."
        real = merged.gt(0) & source.gt(0) & merged.ne(source)
        differ |= real
        gaps.append((merged - source).abs()[real])
    gap_values = pd.concat(gaps)
    zero_seasons = m[zero_over].season.value_counts()
    possession_zero = m.home_possession.eq(0) & m.away_possession.eq(0)
    poss_seasons = m[possession_zero].season.value_counts()
    top_poss = str(poss_seasons.index[0]) if len(poss_seasons) else None
    rows = ctx.rows
    structural = zero_coded_stats(ctx)
    worst = structural[0]["stat"] if structural else None
    box = (m.home_possession.gt(0) | m.away_possession.gt(0)).groupby(m.season).sum()
    return {"matches": int(len(m)),
            "shots_zero_overwritten": int(zero_over.sum()),
            "shots_zero_top_season": str(zero_seasons.index[0]) if len(zero_seasons) else None,
            "shots_zero_top_count": int(zero_seasons.iloc[0]) if len(zero_seasons) else 0,
            "shots_real_differ": int(differ.sum()),
            "shots_real_differ_median_gap": r4(float(gap_values.median())) if len(gap_values) else None,
            "possession_both_zero": int(possession_zero.sum()),
            "possession_zero_top_season": top_poss,
            "possession_zero_top_count": int(poss_seasons.iloc[0]) if len(poss_seasons) else 0,
            "possession_zero_top_season_matches": int(m.season.eq(top_poss).sum()) if top_poss else 0,
            "structural": structural, "structural_share": STRUCTURAL_SHARE, "partial_share": PARTIAL_SHARE,
            "min_box": MIN_BOX_MATCHES, "unjudged": [str(s) for s, n in box.items() if n < MIN_BOX_MATCHES],
            # Why a season could not be judged: most of its matches have the whole ESPN box score stored as 0.
            "unjudged_detail": [{"season": str(s), "matches": int(m.season.eq(s).sum()),
                                 "box_zero": int((possession_zero & m.season.eq(s)).sum())}
                                for s, n in box.items() if n < MIN_BOX_MATCHES],
            # The "record availability" inputs count a stored 0 as recorded: rows whose 20-match average of the
            # worst stat is exactly 0 still report a positive coverage share.
            "worst_coverage_counts_zero": bool(rows.loc[rows[f"team_{worst}_20"].eq(0), f"team_{worst}_coverage20"].gt(0).all())
            if worst else None}


def sources(ctx: LabContext) -> dict:
    data, league, m = ctx.report["data"], ctx.league, ctx.matches
    odds = []
    for columns, group in m.groupby("market_columns"):
        prefix = str(columns).split(",")[0][:-1]
        odds.append({"columns": columns, "label": ODDS_LABEL.get(prefix, prefix), "matches": int(len(group)),
                     "first": str(group.season.min()), "last": str(group.season.max())})
    odds.sort(key=lambda item: item["first"])
    hosts = [urlparse(url).netloc for url in data.get("sources", [])]
    return {
        "fd": {"league_matches": int(len(league)), "seasons": int(league.season.nunique()),
               "first_season": str(league.season.min()), "last_season": str(league.season.max()),
               "full_season_matches": int(league.groupby("season").size().max()), "odds": odds,
               "host": hosts[0] if hosts else None},
        "espn": {"matches": int(len(m)), "verified": int(data["espn_verified_unique_matches"]),
                 "unmatched": int(data["unmatched_fixture_count"]), "rejected": int(data["rejected_espn_alternatives"]),
                 "host": hosts[1] if len(hosts) > 1 else None, "retrieved": str(data.get("retrieved_at_utc", ""))[:10],
                 "cutoff": data.get("cutoff")},
        "quality": data_quality(ctx),
    }


def flow(ctx: LabContext, clubs: list[str]) -> dict:
    league, rows = ctx.league, ctx.rows
    involving = league.home_team.isin(clubs) | league.away_team.isin(clubs)
    both = league.home_team.isin(clubs) & league.away_team.isin(clubs)
    per_club = rows.groupby("team").size()
    return {"league": int(len(league)), "involving": int(involving.sum()), "two_club": int(both.sum()),
            "one_club": int((involving & ~both).sum()), "other": int((~involving).sum()), "rows": int(len(rows)),
            "per_club": int(per_club.iloc[0]) if per_club.nunique() == 1 else None,
            "per_club_by_team": {team: int(n) for team, n in per_club.items()},
            "weight_sum": r4(float(rows.fixture_weight.sum())),
            "derby_weight": r4(float(rows.loc[rows.groupby("match_id").match_id.transform("size").eq(2), "fixture_weight"].iloc[0]))
            if both.any() else None}


def coverage(ctx: LabContext, clubs: list[str]) -> dict:
    cov = ctx.coverage
    seasons = sorted(cov.season.unique())

    def grid(column):
        return [[int(v) for v in cov[cov.team.eq(c)].set_index("season").reindex(seasons)[column].fillna(0)] for c in clubs]

    totals = cov.groupby("season")[["matches", "formation_recorded", "starting_eleven_recorded"]].sum().reindex(seasons)
    share = totals.formation_recorded / totals.matches
    full = int(cov.matches.max())
    short = [{"season": s, "matches": int(cov[cov.season.eq(s)].matches.max())} for s in seasons
             if cov[cov.season.eq(s)].matches.max() < full]
    return {"seasons": seasons, "matches": grid("matches"), "formation": grid("formation_recorded"),
            "eleven": grid("starting_eleven_recorded"), "full_season": full,
            "gap_seasons": [s for s in seasons if totals.loc[s, "formation_recorded"] == 0],
            "partial_seasons": [{"season": s, "share": r4(float(share.loc[s]))} for s in seasons if 0 < share.loc[s] < .9],
            "short_seasons": short,
            "formation_share": r4(float(totals.formation_recorded.sum() / totals.matches.sum())),
            "eleven_share": r4(float(totals.starting_eleven_recorded.sum() / totals.matches.sum())),
            "first_formation_season": next((s for s in seasons if totals.loc[s, "formation_recorded"] > 0), None)}


def raw_table(ctx: LabContext, example: dict) -> dict:
    m = ctx.matches
    dictionary = raw_dictionary()
    row = m.loc[example["index"]]
    columns, values = [], {}
    for name in m.columns:
        entry = dictionary[name]                       # KeyError = an unlabeled raw column (tested)
        series = m[name]
        empty = series.isna() | (series.astype(str).eq("[]") if name.endswith("_players") else False)
        # "Share of exact zeros" is only meaningful for measured quantities, not identifiers.
        numeric = (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
                   and entry["group"] != "id")
        columns.append({"name": name, **entry, "type": plain_type(name, series), "missing": r4(float(empty.mean())),
                        "zero": r4(float(series.eq(0).mean())) if numeric else None, "distinct": int(series.nunique())})
        values[name] = players_summary(row[name]) if name.endswith("_players") else r4(row[name])
    counts = Counter(c["group"] for c in columns)
    return {"path": "data/focused/matches.csv", "n_rows": int(len(m)), "n_cols": int(m.shape[1]),
            "groups": [{"key": k, "label": label, "desc": desc, "count": counts.get(k, 0)} for k, label, desc in RAW_GROUPS],
            "sources": SOURCE_LABEL, "columns": columns,
            "example": {"match_id": str(row.match_id), "date": str(row.date.date()), "season": str(row.season),
                        "home_team": row.home_team, "away_team": row.away_team, "home_goals": int(row.home_goals),
                        "away_goals": int(row.away_goals), "note": example["note"], "values": values}}


PLAIN_DTYPE = {"float64": "실수", "int64": "정수", "object": "글자", "datetime64[ns]": "날짜", "bool": "참/거짓"}


def swap_partner(name: str, columns) -> str | None:
    """The column that holds the other club's value for `name` (team_x <-> opponent_x, goals_for <-> goals_against …)."""
    reverse = {b: a for a, b in SWAP_PAIRS.items()}
    if name in SWAP_PAIRS or name in reverse:
        partner = SWAP_PAIRS.get(name) or reverse[name]
    elif name.startswith("team_"):
        partner = "opponent_" + name[5:]
    elif name.startswith("opponent_"):
        partner = "team_" + name[9:]
    else:
        return None
    return partner if partner in columns else None


def pair_relation(name: str, home: dict, away: dict, columns, own: set[str]) -> str:
    """How the home-club row and the away-club row of the same match relate in one column (a short code)."""
    a, b = home[name], away[name]
    same = a == b
    if name == "outcome":
        return "draw_both" if a == b == 1 else "outcome_flip" if a + b == 2 else ""
    if name == "is_home":
        return "home_flip" if a != b else ""
    partner = swap_partner(name, columns)
    if partner and home[partner] == b and away[partner] == a:
        return "swap_same" if same else "swap"
    if name.startswith("diff_") and isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == -b:
        return "neg_zero" if a == 0 else "neg"
    if name in own:
        return "own_same" if same else "own"
    if name in MATCH_LEVEL and same:
        return "same"
    return "same" if same else ""


def model_table(ctx: LabContext, catalog: list[dict], match_id: str, structural: list[dict]) -> dict:
    rows = ctx.rows
    inputs = {e["name"] for e in catalog}
    others = [n for n in rows.columns if n not in inputs and n != "outcome"]
    dtypes = Counter(PLAIN_DTYPE.get(str(t), str(t)) for t in rows.dtypes)
    pair = rows[rows.match_id.eq(match_id)].sort_values("is_home", ascending=False)
    views = [{name: r4(value) for name, value in record.items()} for record in pair.to_dict("records")]
    columns = list(rows.columns)
    own = OWN_EXTRA | {e["name"] for e in catalog if e["side"] == "team" and not swap_partner(e["name"], columns)}
    relation = {name: pair_relation(name, views[0], views[1], columns, own) for name in columns}
    # Window averages of a never-recorded stat that come out as exactly 0 in this example are stored zeros, not real.
    flagged = {item["stat"] for item in structural}
    zero_coded = [name for name in columns for stat in flagged for side in ["team", "opponent"] for w in WINDOWS
                  if name == f"{side}_{stat}_{w}" and 0 in (views[0][name], views[1][name])]
    return {"n_rows": int(len(rows)), "n_cols": int(rows.shape[1]), "columns": columns,
            "roles": {"input_numeric": len(ctx.numeric), "input_category": len([n for n in CATEGORIES if n in inputs]),
                      "target": 1, "other": others},
            "dtypes": dict(dtypes.most_common()),
            "extra": {name: {"label": label, "role": role, "desc": desc, "use": use}
                      for name, (label, role, desc, use) in MODEL_EXTRA.items()},
            "key_columns": [c for c in KEY_MODEL_COLUMNS if c in rows.columns], "example": views,
            "relation": relation, "zero_coded": zero_coded}


def outcomes(ctx: LabContext, clubs: list[str]) -> dict:
    rows = ctx.rows
    counts = rows.groupby(["team", "is_home", "outcome"]).size()

    def triple(team, home):
        return [int(counts.get((team, home, k), 0)) for k in [0, 1, 2]]

    weighted = rows.groupby("outcome").fixture_weight.sum().reindex([0, 1, 2], fill_value=0)
    always = ctx.report["test"]["always_team_win"]
    return {"by_club": {team: {"home": triple(team, 1), "away": triple(team, 0)} for team in clubs},
            "row_share": [r4(float(x)) for x in rows.outcome.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)],
            "weighted_share": [r4(float(x)) for x in weighted / weighted.sum()],
            "home_win": r4(float(rows[rows.is_home.eq(1)].outcome.eq(2).mean())),
            "away_win": r4(float(rows[rows.is_home.eq(0)].outcome.eq(2).mean())),
            "always_win_test": r4(float(always["accuracy"])), "test_rows": int(always["team_appearances"]),
            "test_matches": int(always["unique_matches"]), "test_start": ctx.test_start}


def inputs(ctx: LabContext, catalog: list[dict], example_row: pd.Series, match_id: str) -> dict:
    rows = ctx.rows
    stats = {}
    for entry in catalog:
        name, series = entry["name"], rows[entry["name"]]
        if entry["kind"] == "category":
            text = series.astype(str)
            missing = text.str.endswith("__missing__")
            counts = text[~missing].value_counts()
            stats[name] = {"miss": r4(float(missing.mean())), "top": str(counts.index[0]) if len(counts) else None,
                           "top_share": r4(float(counts.iloc[0] / len(series))) if len(counts) else None, "n": int(len(counts))}
        else:
            values = pd.to_numeric(series, errors="coerce")
            valid = values.dropna()
            q = valid.quantile([.1, .5, .9]).tolist() if len(valid) else [None] * 3
            stats[name] = {"miss": r4(float(values.isna().mean())), "q10": r4(q[0]), "q50": r4(q[1]), "q90": r4(q[2])}
            if entry["kind"] == "binary":
                stats[name]["one"] = r4(float(valid.eq(1).mean())) if len(valid) else None
    groups = Counter(e["group"] for e in catalog)
    # The form5/10/20 groups hold window averages (and differences of two averages) plus, in form20, the
    # "share of the last 20 matches where this stat was recorded" inputs, which are not averages.
    form = [e for e in catalog if e["group"] in {f"form{w}" for w in WINDOWS}]
    coverage_inputs = [e["name"] for e in form if e["name"].endswith("_coverage20")]
    home_view = rows[rows.match_id.eq(match_id) & rows.is_home.eq(1)].iloc[0]
    leak_example = {"team": home_view.team, "opponent": home_view.opponent, "date": str(home_view.date.date()),
                    "items": [{"stem": stem, "label": label, "same": r4(example_row[f"home_{stem}"]),
                               "input": column, "pre": r4(home_view[column])} for stem, column, label in LEAK_PAIRS]}
    return {"n_total": len(catalog), "n_numeric": len(ctx.numeric),
            "n_category": len([e for e in catalog if e["kind"] == "category"]),
            "group_counts": {key: int(groups.get(key, 0)) for key, _, _ in GROUPS}, "stats": stats,
            "window_inputs": len(form), "window_averages": len(form) - len(coverage_inputs),
            "window_coverage": len(coverage_inputs),
            "leakage": [{"key": key, "what": LEAK_TEXT[key][0], "why": LEAK_TEXT[key][1]}
                        for key in ctx.feature_meta["same_match_excluded"]],
            "not_inputs": [n for n in rows.columns if n not in {e["name"] for e in catalog}],
            "leak_example": leak_example}


def build(ctx: LabContext) -> dict:
    clubs = list(ctx.report["protocol"]["clubs"])
    catalog = build_catalog(ctx.numeric)
    example = pick_example(ctx, clubs)
    example_row = ctx.matches.loc[example["index"]]
    match_id = str(example_row.match_id)
    src = sources(ctx)
    return {
        "clubs": clubs, "codes": [CODES.get(c, c[:3].upper()) for c in clubs],
        "kpi": kpis(ctx, catalog, clubs),
        "sources": src,
        "flow": flow(ctx, clubs),
        "coverage": coverage(ctx, clubs),
        "raw": raw_table(ctx, example),
        "model": model_table(ctx, catalog, match_id, src["quality"]["structural"]),
        "outcome": outcomes(ctx, clubs),
        "inputs": inputs(ctx, catalog, example_row, match_id),
        "limits": {"first_date": str(ctx.rows.date.min().date()), "first_season": str(ctx.rows.season.min()),
                   "half_life_days": ctx.report["protocol"].get("half_life_days")},
    }
