"""Step 2 payload: preprocessing pipeline facts, the "time travel" demo data, missing-value tables,
real imputer/scaler parameters and distribution / scatter / correlation data for curated inputs.

Every number shown in s2.html comes from here and is computed from the source data or from the
fitted preprocessing of the playground model (``ctx.model.processor_``)."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import focused_features
from ..focused_data import CLUBS, HISTORICAL_NAMES, TEAM_STATS
from ..focused_features import CATEGORIES, HISTORY_STATS
from ..focused_model import preprocessing, train_weights
from .catalog import GROUPS, build_catalog
from .context import LabContext

# Curated numeric inputs for the pickers: (column, short Korean label for tight spots such as the heatmap).
CURATED = [
    ("team_elo", "우리 Elo"), ("opponent_elo", "상대 Elo"), ("diff_elo", "Elo 차이"),
    ("team_points_5", "우리 최근5 승점"), ("team_points_10", "우리 최근10 승점"), ("team_points_20", "우리 최근20 승점"),
    ("opponent_points_5", "상대 최근5 승점"), ("opponent_points_20", "상대 최근20 승점"), ("diff_points_5", "최근5 승점 차이"),
    ("team_goals_for_20", "우리 최근20 득점"), ("team_goals_against_20", "우리 최근20 실점"),
    ("opponent_goals_for_20", "상대 최근20 득점"), ("opponent_goals_against_20", "상대 최근20 실점"),
    ("diff_goals_for_20", "최근20 득점 차이"), ("diff_goals_against_20", "최근20 실점 차이"),
    ("team_shots_20", "우리 최근20 슈팅"), ("team_sot_20", "우리 최근20 유효슈팅"),
    ("diff_shots_20", "최근20 슈팅 차이"), ("diff_sot_20", "최근20 유효슈팅 차이"),
    ("team_possession_20", "우리 최근20 점유율"), ("opponent_possession_20", "상대 최근20 점유율"),
    ("opponent_possession_coverage20", "상대 ESPN 기록 비율"),
    ("team_venue_points", "우리 같은 장소 승점"), ("opponent_venue_points", "상대 같은 장소 승점"),
    ("team_rest_days", "우리 휴식일"), ("opponent_rest_days", "상대 휴식일"), ("diff_rest_days", "휴식일 차이"),
    ("team_games_last14", "우리 14일 경기 수"), ("team_season_games", "시즌 치른 경기 수"),
    ("team_lineup_returning", "선발 유지 인원"), ("team_lineup_observed_experience", "선발 관측 출전 수"),
    ("team_lineup_prior_goals_rate", "우리 선발 득점률"), ("opponent_lineup_prior_goals_rate", "상대 선발 득점률"),
    ("defenders", "수비수 수"), ("midfielders", "미드필더 수"), ("forwards", "공격수 수"),
    ("formation_used_last20", "같은 포메이션 사용률"), ("formation_switch", "포메이션 변경"),
]
# Extra columns only needed by the time-travel demo (rolling means it recomputes from earlier rows).
DEMO_EXTRA = ["team_goals_for_5", "team_goals_for_10", "team_goals_against_5", "team_goals_against_10"]
CORR = ["diff_elo", "team_elo", "opponent_elo", "is_home", "team_points_5", "team_points_20", "opponent_points_20",
        "diff_goals_for_20", "diff_goals_against_20", "diff_sot_20", "team_venue_points", "opponent_venue_points",
        "team_possession_20", "opponent_possession_coverage20", "diff_rest_days", "team_lineup_prior_goals_rate"]
SHORT_EXTRA = {"is_home": "홈 경기"}
PRESETS = [("diff_elo", "diff_goals_for_20", "Elo 차이 vs 최근 득점 차이"),
           ("team_points_5", "team_points_20", "최근 5경기 vs 20경기 승점"),
           ("team_elo", "opponent_elo", "우리 Elo vs 상대 Elo"),
           ("team_shots_20", "team_goals_for_20", "슈팅 vs 득점 (0 기록 주의)")]
# Inputs that carry ESPN's "0 instead of blank" placeholder rows, and which club's last-20 window decides it.
ZERO_PARTS = ("shots", "sot", "possession")
ZERO_WINDOW = 20
CLEAN_MIN = 30   # a season mean without the affected rows is shown only when at least this many rows remain
# Rows of the "missing by season" heatmap: (key, Korean label, how to flag a row as missing).
MISSING_ROWS = [
    ("formation", "포메이션 이름(우리 팀)", "formation_missing"),
    ("formation_switch", "포메이션 변경 여부", "formation_switch"),
    ("lineup_incomplete", "선발 11명 기록 불완전(우리 팀)", "lineup_incomplete"),
    ("team_lineup_returning", "직전 선발 유지 인원", "team_lineup_returning"),
    ("team_possession_5", "우리 최근 5경기 점유율(ESPN)", "team_possession_5"),
    ("opponent_possession_5", "상대 최근 5경기 점유율(ESPN)", "opponent_possession_5"),
    ("opponent_shots_5", "상대 최근 5경기 슈팅(Football-Data)", "opponent_shots_5"),
    ("team_points_5", "우리 최근 5경기 승점", "team_points_5"),
    ("team_venue_points", "같은 장소 최근 10경기 승점", "team_venue_points"),
]
OUTCOME_KEYS = [(2, "win"), (1, "draw"), (0, "loss")]
POINTS = {0: 0, 1: 1, 2: 3}


def _num(value, digits=4):
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _arr(values, digits=4):
    return [_num(v, digits) for v in np.asarray(values, dtype=float)]


def zero_side(name: str) -> str | None:
    """Which history window decides whether a rolling input can contain placeholder zeros: 'team', 'opp' or 'both'."""
    if not any(part in name for part in ZERO_PARTS):
        return None
    return {"team": "team", "opponent": "opp", "diff": "both"}.get(name.partition("_")[0])


def placeholder_ids(ctx: LabContext) -> set:
    """Matches where ESPN gave 0 % possession to both teams: a 'not recorded' placeholder, not a real value."""
    m = ctx.matches
    return set(m.match_id[m.home_possession.eq(0) & m.away_possession.eq(0)])


def window_flags(ctx: LabContext, ids: set, n: int = ZERO_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Per row: does the club's (team) / the opponent's (opp) previous-n league matches include a placeholder match?
    Histories cover every league match of a team (opponents' games against other clubs too), like ClubHistory."""
    league = ctx.league
    long = pd.concat([league[["date", "match_id", side]].rename(columns={side: "club"}) for side in ("home_team", "away_team")])
    long = long.sort_values(["date", "match_id"], kind="mergesort")
    long["hit"] = long.match_id.isin(ids).astype(int)
    lookup = {}
    for club, part in long.groupby("club", sort=False):
        lookup[club] = (part.date.to_numpy(), np.concatenate([[0], np.cumsum(part.hit.to_numpy())]))

    def flags(clubs: pd.Series) -> np.ndarray:
        out = np.zeros(len(clubs), dtype=bool)
        for k, (club, date) in enumerate(zip(clubs, ctx.rows.date.to_numpy())):
            dates, cs = lookup[club]
            i = int(np.searchsorted(dates, date, side="left"))   # matches strictly before this date
            out[k] = cs[i] - cs[max(0, i-n)] > 0
        return out

    return flags(ctx.rows.team), flags(ctx.rows.opponent)


def clean_mask(name: str, team_flag: np.ndarray, opp_flag: np.ndarray) -> np.ndarray:
    side = zero_side(name)
    if side is None:
        return np.ones(len(team_flag), dtype=bool)
    hit = team_flag if side == "team" else opp_flag if side == "opp" else team_flag | opp_flag
    return ~hit


def feature_constants() -> dict:
    """Read the Elo / schedule constants straight from focused_features.py so the page cannot drift from the code."""
    source = inspect.getsource(focused_features)

    def grab(pattern):
        match = re.search(pattern, source)
        if not match:
            raise ValueError(f"Constant not found in focused_features.py: {pattern}")
        return match.groups()

    start = float(grab(r"elo: float = ([\d.]+)")[0])
    base, keep, base2 = grab(r"state\.elo = ([\d.]+) ?\+ ?([\d.]+)\*\(state\.elo-([\d.]+)\)")
    home_edge, scale = grab(r"10\*\*\(\(away\.elo-home\.elo-([\d.]+)\)/([\d.]+)\)")
    k = grab(r"delta = ([\d.]+)\*\(")[0]
    rest_cap, rest_default = grab(r"min\(\(date-self\.dates\[-1\]\)\.days, (\d+)\) if self\.dates else (\d+)")
    windows = [int(w) for w in grab(r"for window in \[([\d, ]+)\]")[0].split(",")]
    venue_n = int(grab(r"if h == home\]\[-(\d+):\]")[0])
    if float(base) != start or float(base2) != start:
        raise ValueError("Elo season regression is not centred on the start rating")
    return {"elo_start": start, "elo_keep": float(keep), "elo_home_edge": float(home_edge), "elo_scale": float(scale),
            "elo_k": float(k), "rest_cap": int(rest_cap), "rest_default": int(rest_default), "windows": windows,
            "venue_n": venue_n, "history_stats": len(HISTORY_STATS)}


def _processor(ctx: LabContext):
    if ctx.model is not None:
        return ctx.model.processor_
    # Same deterministic preprocessing as the playground model (fit on the same training rows).
    return preprocessing(ctx.numeric, True).fit(ctx.train)


def _summary(values: np.ndarray) -> dict:
    total = len(values)
    v = values[np.isfinite(values)]
    if not len(v):
        return {"n": 0, "missing": total, "miss_rate": 1.0 if total else 0.0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5*iqr, q3 + 1.5*iqr
    inside = v[(v >= lo_fence) & (v <= hi_fence)]
    return {"n": int(len(v)), "missing": int(total-len(v)), "miss_rate": _num((total-len(v))/total),
            "mean": _num(v.mean()), "sd": _num(v.std(ddof=1) if len(v) > 1 else 0.0),
            "min": _num(v.min()), "q1": _num(q1), "median": _num(med), "q3": _num(q3), "max": _num(v.max()),
            "whisker_lo": _num(inside.min() if len(inside) else q1), "whisker_hi": _num(inside.max() if len(inside) else q3),
            "outliers": int(((v < lo_fence) | (v > hi_fence)).sum())}


def _renamed_names(ctx: LabContext) -> dict:
    """Football-Data spellings that were actually rewritten to the ESPN club name."""
    raw_dir = Path(ctx.data_dir) / "football_data"
    seen: set[str] = set()
    for path in sorted(raw_dir.glob("E0_*.csv")):
        frame = pd.read_csv(path, usecols=lambda c: c in {"HomeTeam", "AwayTeam"})
        seen |= set(frame.HomeTeam.dropna()) | set(frame.AwayTeam.dropna())
    used = set(ctx.league.home_team) | set(ctx.league.away_team)
    if seen:
        pairs = {k: v for k, v in HISTORICAL_NAMES.items() if k in seen and k != v}
        source = "raw"
    else:  # raw files not kept: fall back to the mapping entries whose target appears in the cleaned data
        pairs = {k: v for k, v in HISTORICAL_NAMES.items() if k != v and v in used}
        source = "mapping"
    ordered = sorted(pairs.items(), key=lambda kv: (kv[1] not in CLUBS, kv[1]))  # the six clubs first: familiar examples
    return {"count": len(pairs), "examples": [[k, v] for k, v in ordered[:4]], "source": source,
            "clean_names": len(used)}


def _pipeline(ctx: LabContext, proc, consts: dict) -> dict:
    rows, matches, league = ctx.rows, ctx.matches, ctx.league
    in_cohort = league.home_team.isin(CLUBS) | league.away_team.isin(CLUBS)
    raw_forms = pd.concat([matches.home_formation_raw, matches.away_formation_raw]).dropna().astype(str)
    clean_forms = pd.concat([matches.home_formation, matches.away_formation])
    d_suffix = raw_forms[raw_forms.str.endswith("-d")]
    per_match = rows.groupby("match_id").size()
    catalog = {c["name"]: c for c in build_catalog(ctx.numeric)}
    group_counts = {}
    for name in ctx.numeric:
        group_counts[catalog[name]["group"]] = group_counts.get(catalog[name]["group"], 0) + 1
    train = ctx.train
    miss_cells = int(rows[ctx.numeric].isna().to_numpy().sum())
    numbers = list(proc.transformers_[1][2])
    imputer = proc.named_transformers_["numeric"].named_steps["impute"]
    onehot = proc.named_transformers_["categories"]
    n_indicators = int(len(imputer.indicator_.features_)) if imputer.indicator_ is not None else 0
    weights = train_weights(train)
    derby = rows.fixture_weight.lt(1)
    return {
        "league_matches": int(len(league)), "league_seasons": int(league.season.nunique()),
        "cohort_matches_in_league": int(in_cohort.sum()), "matches": int(len(matches)),
        "espn_verified": int(matches.espn_verified.sum()),
        "unmatched": int(ctx.report["data"]["unmatched_fixture_count"]),
        "rejected_alternatives": int(ctx.report["data"]["rejected_espn_alternatives"]),
        "names": _renamed_names(ctx),
        "formation_raw_kinds": int(raw_forms.nunique()), "formation_clean_kinds": int(clean_forms.dropna().nunique()),
        "formation_d_count": int(len(d_suffix)), "formation_d_labels": sorted(d_suffix.unique().tolist()),
        "formation_sides": int(len(clean_forms)), "formation_missing_sides": int(clean_forms.isna().sum()),
        "formation_missing_rows": int(rows.formation.eq("__missing__").sum()),
        "rows": int(len(rows)), "rows_per_club": int(rows.groupby("team").size().iloc[0]),
        "clubs": int(rows.team.nunique()), "derby_matches": int((per_match == 2).sum()),
        "derby_rows": int(derby.sum()), "numeric": int(len(ctx.numeric)), "categorical": len(CATEGORIES),
        "groups": [{"key": k, "label": label, "n": group_counts.get(k, 0)} for k, label, _ in GROUPS if group_counts.get(k)],
        "missing_cells": miss_cells, "cells": int(len(rows)*len(ctx.numeric)),
        "cols_with_missing_train": int(train[ctx.numeric].isna().any().sum()),
        "cols_with_missing_all": int(rows[ctx.numeric].isna().any().sum()),
        "indicators": n_indicators, "scaled": len(numbers) + n_indicators,
        "onehot": [{"name": name, "label": catalog[name]["label"], "n": int(len(cats))}
                   for name, cats in zip(onehot.feature_names_in_, onehot.categories_)],
        "onehot_total": int(sum(len(c) for c in onehot.categories_)),
        "total_columns": int(len(proc.get_feature_names_out())),
        "train_rows": int(len(train)), "train_first": str(train.date.min().date()), "train_last": str(train.date.max().date()),
        "test_rows": int(len(rows)-len(train)), "half_life": int(ctx.report["protocol"]["half_life_days"]),
        "weight_total": _num(weights.sum(), 2),
        "constants": consts,
    }


def _unseen(ctx: LabContext) -> list[dict]:
    train, held = ctx.train, ctx.heldout
    out = []
    for name in CATEGORIES:
        new = sorted(set(held[name]) - set(train[name]))
        out.append({"name": name, "values": new[:4], "count": len(new), "rows": int(held[name].isin(new).sum())})
    return out


def _weights(ctx: LabContext) -> dict:
    train = ctx.train
    ref = train.date.max()
    years = np.round(np.arange(0, 18.01, 0.25), 2)
    synth = pd.DataFrame({"date": [ref - pd.Timedelta(days=int(round(y*365.25))) for y in years], "fixture_weight": 1.0})
    curve = train_weights(synth)
    w = train_weights(train)
    share = pd.Series(w, index=train.index).groupby(train.season).sum()/w.sum()
    seasons = sorted(train.season.unique())
    # train_weights measures age from the newest row, so the reference date must be inside the frame.
    marks = [[y, _num(curve[int(np.flatnonzero(years == y)[0])])] for y in (3, 6, 9)]
    last3 = seasons[-3:]
    return {"ref_date": str(ref.date()), "curve": [[float(y), _num(c)] for y, c in zip(years, curve)], "marks": marks,
            "seasons": seasons, "share": [_num(share[s]) for s in seasons],
            "rows": [int((train.season == s).sum()) for s in seasons],
            "last3": last3, "last3_share": _num(share[last3].sum()),
            "derby_weight": _num(train.fixture_weight.min())}


def _missing(ctx: LabContext, seasons: list[str]) -> dict:
    rows = ctx.rows
    catalog = {c["name"]: c for c in build_catalog(ctx.numeric)}
    # Stats that only ESPN reports (Football-Data's league file has no column for them).
    espn_only = [s for s in TEAM_STATS if f"home_{s}" not in ctx.league.columns]
    groups = []
    for key, label, _ in GROUPS:
        cols = [n for n in ctx.numeric if catalog[n]["group"] == key]
        if not cols:
            continue
        frame = rows[cols].isna()
        opp = [c for c in cols if c.startswith("opponent_")]
        opp_espn = [c for c in opp if any(c == f"opponent_{s}_{w}" for s in espn_only for w in (5, 10, 20))]
        groups.append({"key": key, "label": label, "cols": len(cols), "cols_missing": int(frame.any().sum()),
                       "rate": _num(frame.to_numpy().mean()), "cells": int(frame.to_numpy().sum()),
                       "opp_cells": int(frame[opp].to_numpy().sum()), "opp_espn_cells": int(frame[opp_espn].to_numpy().sum())})
    flags = {"formation_missing": rows.formation.eq("__missing__"), "lineup_incomplete": rows.team_lineup_complete.eq(0)}
    heat = []
    for key, label, how in MISSING_ROWS:
        flag = flags[how] if how in flags else rows[how].isna()
        by = flag.groupby(rows.season).mean()
        heat.append({"key": key, "label": label, "rates": [_num(by.get(s, np.nan)) for s in seasons],
                     "overall": _num(flag.mean()), "count": int(flag.sum())})
    from_big = rows.opponent.isin(CLUBS)
    cov = rows.opponent_possession_coverage20
    return {"groups": groups, "heat": heat, "season_rows": [int((rows.season == s).sum()) for s in seasons],
            "total_cells": int(sum(g["cells"] for g in groups)), "espn_only": espn_only,
            "formation_rows": _num(rows.formation.eq("__missing__").mean()),
            "espn_cover": {"big6_mean": _num(cov[from_big].mean()), "other_mean": _num(cov[~from_big].mean()),
                           "big6_rows": int(from_big.sum()), "other_rows": int((~from_big).sum()),
                           "other_poss5_missing": _num(rows.opponent_possession_5[~from_big].isna().mean()),
                           "big6_poss5_missing": _num(rows.opponent_possession_5[from_big].isna().mean())}}


def _placeholder_zeros(ctx: LabContext, ids: set, team_flag: np.ndarray, opp_flag: np.ndarray) -> dict:
    """ESPN sometimes reports team stats of a match it never recorded as 0 instead of blank (both sides 0 % possession).
    The current pipeline reads those zeros as real values; the page shows the size of that hidden-missing problem and
    ships per-row flags so charts can leave the affected rows out."""
    m, rows, league = ctx.matches, ctx.rows, ctx.league
    flag = m.match_id.isin(ids)
    per_season = m.season.value_counts()
    by = m[flag].season.value_counts().sort_values(ascending=False)
    fd = league.set_index("match_id")
    fd_shots = fd.loc[fd.index.intersection(list(ids)), ["home_shots", "away_shots"]]
    return {"matches": int(flag.sum()), "total": int(len(m)),
            "shots_zero_both": int((m.home_shots.eq(0) & m.away_shots.eq(0)).sum()),
            "top": [{"season": s, "n": int(n), "of": int(per_season[s])} for s, n in by.head(3).items()],
            "fd_has_shots": int((fd_shots.sum(axis=1) > 0).sum()),
            "window": ZERO_WINDOW, "clean_min": CLEAN_MIN,
            "rows_in_window20": int(team_flag.sum()), "rows_opp": int(opp_flag.sum()),
            "rows_either": int((team_flag | opp_flag).sum()), "rows": int(len(rows)),
            # one character per row ('1' = that side's last-20 window holds a placeholder match): compact and exact
            "team_flag": "".join("1" if x else "0" for x in team_flag),
            "opp_flag": "".join("1" if x else "0" for x in opp_flag)}


def _correlations(ctx: LabContext, short: dict, team_flag: np.ndarray, opp_flag: np.ndarray) -> dict:
    rows = ctx.rows
    score = rows.outcome.map(POINTS).astype(float)
    zero = []
    for name in CORR:
        if zero_side(name) is None:
            continue
        keep = clean_mask(name, team_flag, opp_flag)
        zero.append({"name": name, "short": short.get(name, name), "r_all": _num(rows[name].astype(float).corr(score), 3),
                     "r_clean": _num(rows[name][keep].astype(float).corr(score[keep]), 3), "n_clean": int(keep.sum())})
    frame = rows[CORR].astype(float).assign(outcome_score=score)
    matrix = frame.corr()
    names = [*CORR, "outcome_score"]
    with_outcome = matrix["outcome_score"].drop("outcome_score")
    order = with_outcome.abs().sort_values(ascending=False)
    pairs = []
    for i, a in enumerate(CORR):
        for b in CORR[i+1:]:
            pairs.append((abs(matrix.loc[a, b]), a, b, matrix.loc[a, b]))
    pairs.sort(reverse=True)
    # Whole-input view: how many of the 215 inputs are near-duplicates of another one?
    full = rows[ctx.numeric].astype(float).corr().abs().to_numpy()
    iu = np.triu_indices_from(full, k=1)
    vals = full[iu]
    finite = np.isfinite(vals)
    dup = finite & (vals >= 0.95)
    all_outcome = rows[ctx.numeric].astype(float).corrwith(score).abs().dropna().sort_values(ascending=False)
    catalog = {c["name"]: c for c in build_catalog(ctx.numeric)}
    dup_pairs = sorted(((vals[k], ctx.numeric[iu[0][k]], ctx.numeric[iu[1][k]]) for k in np.flatnonzero(dup)), reverse=True)
    examples, seen = [], set()
    coverage_pairs = sum(1 for _, a, b in dup_pairs if a.endswith("_coverage20") and b.endswith("_coverage20"))
    for v, a, b in dup_pairs:  # one example per family; coverage-vs-coverage copies are counted separately
        family = (a.split("_")[0], catalog[a]["group"], catalog[b]["group"])
        if family in seen or (a.endswith("_coverage20") and b.endswith("_coverage20")):
            continue
        seen.add(family)
        examples.append({"a": catalog[a]["label"], "b": catalog[b]["label"], "r": _num(v, 3)})
        if len(examples) == 4:
            break
    return {"names": names, "short": [short.get(n, SHORT_EXTRA.get(n, "결과 점수(승3·무1·패0)")) for n in names],
            "r": [[_num(matrix.loc[a, b], 3) for b in names] for a in names],
            "outcome_rank": [{"name": n, "short": short.get(n, SHORT_EXTRA.get(n, n)), "r": _num(with_outcome[n], 3)}
                             for n in order.index],
            "top_pairs": [{"a": a, "b": b, "r": _num(r, 3)} for _, a, b, r in pairs[:6]],
            "all_pairs": int(finite.sum()), "all_dup_pairs": int(dup.sum()),
            "all_dup_cols": int(len({c for _, a, b in dup_pairs for c in (a, b)})),
            "dup_examples": examples, "dup_coverage_pairs": coverage_pairs,
            "all_top": [{"name": n, "label": catalog[n]["label"], "r": _num(all_outcome[n], 3)} for n in all_outcome.index[:5]],
            "n_inputs": len(ctx.numeric), "zero": zero}


def build(ctx: LabContext) -> dict:
    rows = ctx.rows
    if not rows.date.is_monotonic_increasing:
        raise ValueError("rows must be sorted by date")
    n_train = int(ctx.train_mask.sum())
    if not ctx.train_mask[:n_train].all():
        raise ValueError("training rows must be a date prefix")
    proc = _processor(ctx)
    consts = feature_constants()
    catalog = {c["name"]: c for c in build_catalog(ctx.numeric)}
    seasons = sorted(rows.season.unique())
    season_idx = {s: i for i, s in enumerate(seasons)}
    opponents = sorted(rows.opponent.unique())
    opp_idx = {o: i for i, o in enumerate(opponents)}
    clubs = list(CLUBS)
    base = rows.date.min()
    short = dict(CURATED)

    numbers = list(proc.transformers_[1][2])
    pos = {n: i for i, n in enumerate(numbers)}
    imputer = proc.named_transformers_["numeric"].named_steps["impute"]
    scaler = proc.named_transformers_["numeric"].named_steps["scale"]
    ind = list(imputer.indicator_.features_) if imputer.indicator_ is not None else []

    ids = placeholder_ids(ctx)
    team_flag, opp_flag = window_flags(ctx, ids)
    season_arr = rows.season.to_numpy()

    columns = {}
    variables = []
    stats = {}
    season_stats = {}
    outcome = rows.outcome.to_numpy()
    for name, short_label in CURATED + [(n, n) for n in DEMO_EXTRA]:
        values = rows[name].to_numpy(dtype=float)
        digits = 2 if name.endswith("elo") else 4
        columns[name] = _arr(values, digits)
        if name in DEMO_EXTRA:
            continue
        finite = values[np.isfinite(values)]
        uniq = np.unique(finite)
        discrete = bool(np.allclose(uniq, np.round(uniq)) and len(uniq) <= 30)
        i = pos[name]
        entry = {"name": name, "short": short_label, "label": catalog[name]["label"], "group": catalog[name]["group"],
                 "unit": catalog[name]["unit"], "desc": catalog[name]["desc"], "discrete": discrete,
                 "lo": _num(finite.min() if discrete else np.percentile(finite, 0.5)),
                 "hi": _num(finite.max() if discrete else np.percentile(finite, 99.5)),
                 "median": _num(imputer.statistics_[i], 6), "mean": _num(scaler.mean_[i], 6), "scale": _num(scaler.scale_[i], 6),
                 "train_missing": int(rows[name][ctx.train_mask].isna().sum()),
                 "zero_flag": zero_side(name) is not None, "zero_side": zero_side(name)}
        if i in ind:
            k = len(numbers) + ind.index(i)
            entry.update(ind_mean=_num(scaler.mean_[k], 6), ind_scale=_num(scaler.scale_[k], 6))
        variables.append(entry)
        stats[name] = {"all": _summary(values), **{key: _summary(values[outcome == code]) for code, key in OUTCOME_KEYS}}
        by = pd.Series(values).groupby(rows.season.to_numpy())
        season_stats[name] = {"mean": [_num(by.get_group(s).mean()) for s in seasons],
                              "q1": [_num(by.get_group(s).quantile(.25)) for s in seasons],
                              "q3": [_num(by.get_group(s).quantile(.75)) for s in seasons],
                              "n": [int(by.get_group(s).notna().sum()) for s in seasons]}
        if entry["zero_side"]:
            keep = clean_mask(name, team_flag, opp_flag)
            v_keep, o_keep = values[keep], outcome[keep]
            stats[name]["clean"] = {"all": _summary(v_keep), **{key: _summary(v_keep[o_keep == code]) for code, key in OUTCOME_KEYS}}
            clean = pd.Series(v_keep).groupby(season_arr[keep])
            counts = clean.count()
            season_stats[name]["clean_n"] = [int(counts.get(s, 0)) for s in seasons]
            season_stats[name]["clean_mean"] = [_num(clean.get_group(s).mean()) if counts.get(s, 0) >= CLEAN_MIN else None
                                                for s in seasons]

    pipeline = _pipeline(ctx, proc, consts)
    return {
        "n_rows": int(len(rows)), "n_train": n_train, "test_start": ctx.test_start,
        "clubs": clubs, "seasons": seasons, "opponents": opponents, "base_date": str(base.date()),
        "rows": {"day": ((rows.date - base).dt.days).astype(int).tolist(),
                 "club": [clubs.index(t) for t in rows.team], "opp": [opp_idx[o] for o in rows.opponent],
                 "home": rows.is_home.astype(int).tolist(), "gf": rows.goals_for.astype(int).tolist(),
                 "ga": rows.goals_against.astype(int).tolist(), "out": rows.outcome.astype(int).tolist(),
                 "season": [season_idx[s] for s in rows.season]},
        "cols": columns, "vars": variables, "stats": stats, "season_stats": season_stats,
        "presets": [{"x": x, "y": y, "title": t} for x, y, t in PRESETS],
        "corr": _correlations(ctx, short, team_flag, opp_flag),
        "missing": _missing(ctx, seasons),
        "zeros": _placeholder_zeros(ctx, ids, team_flag, opp_flag),
        "pipeline": pipeline,
        "weights": _weights(ctx),
        "unseen": _unseen(ctx),
        "outcome_counts": {key: int((outcome == code).sum()) for code, key in OUTCOME_KEYS},
    }
