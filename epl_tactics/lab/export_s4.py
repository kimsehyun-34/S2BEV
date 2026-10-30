"""Step 4 payload: the tuned Random Forest and everything needed to run it in a browser.

The forest is exported node-by-node (not as a black box) so the page can reproduce sklearn's
`predict_proba` exactly and also attribute a prediction to individual inputs.
"""
from __future__ import annotations

import base64
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..focused_features import CATEGORIES, FORMATION_NUMERIC
from ..focused_model import train_weights
from .context import LabContext

ROUND = 6                       # decimals kept for exported raw inputs; parity is checked on the rounded values
DIFF_BASES = ["elo", "points_5", "goals_for_20", "goals_against_20", "shots_20", "sot_20", "rest_days"]
LINEUP_KEYS = ["complete", "size", "returning", "prior_players_known", "observed_experience",
               "prior_goals_rate", "prior_assists_rate", "prior_shots_rate", "prior_sot_rate"]
META_COLUMNS = ["match_id", "date", "season", "outcome", "goals_for", "goals_against",
                "market_loss", "market_draw", "market_win"]


def clean(value):
    """JSON-safe scalar: NaN/inf -> None, numpy numbers -> python, floats rounded."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if not math.isfinite(value) else round(value, ROUND)
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return value


def _b64(array, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def export_preprocessor(model) -> dict:
    """Parameters of ColumnTransformer(OneHot + median-impute/indicator + StandardScaler)."""
    transformer = model.processor_
    one_hot = transformer.named_transformers_["categories"]
    pipeline = transformer.named_transformers_["numeric"]
    imputer, scaler = pipeline.named_steps["impute"], pipeline.named_steps["scale"]
    category_columns, numeric_columns = transformer.transformers_[0][2], transformer.transformers_[1][2]
    if list(category_columns) != list(one_hot.feature_names_in_):
        raise ValueError("category column order differs from the one-hot encoder's fitted columns")
    indicators = imputer.indicator_.features_ if imputer.indicator_ is not None else np.array([], dtype=int)
    statistics = np.nan_to_num(np.asarray(imputer.statistics_, dtype=float), nan=0.0)
    names = [str(name) for name in transformer.get_feature_names_out()]
    return {
        "categorical": [{"name": column, "categories": [str(c) for c in cats]}
                        for column, cats in zip(category_columns, one_hot.categories_)],
        "numeric": list(numeric_columns), "median": statistics.tolist(),
        "indicator_of": [int(i) for i in indicators],
        "mean": np.asarray(scaler.mean_, dtype=float).tolist(), "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "feature_names": names,
    }


def export_forest(model) -> dict:
    forest = model.estimator_
    if [int(c) for c in forest.classes_] != [0, 1, 2]:
        raise ValueError(f"forest class order must be loss/draw/win (0, 1, 2), got {list(forest.classes_)}")
    offsets, feature, threshold, left, right, value = [0], [], [], [], [], []
    for estimator in forest.estimators_:
        tree = estimator.tree_
        node_value = tree.value[:, 0, :]
        node_value = node_value / node_value.sum(axis=1, keepdims=True)
        is_leaf = tree.children_left == -1
        feature.append(np.where(is_leaf, -1, tree.feature))
        threshold.append(np.where(is_leaf, 0.0, tree.threshold))
        left.append(tree.children_left)
        right.append(tree.children_right)
        value.append(node_value)
        offsets.append(offsets[-1] + tree.node_count)
    cat = np.concatenate
    if max(np.diff(offsets)) >= 32767:
        raise ValueError("a tree has too many nodes for the int16 child indices of the browser format")
    return {"n_trees": len(forest.estimators_), "n_nodes": int(offsets[-1]), "n_features": int(forest.n_features_in_),
            "offsets": _b64(offsets, "<i4"), "feature": _b64(cat(feature), "<i2"), "threshold": _b64(cat(threshold), "<f8"),
            "left": _b64(cat(left), "<i2"), "right": _b64(cat(right), "<i2"), "value": _b64(cat(value), "<f4")}


ELO_COLUMNS = ["is_home", "diff_elo"]


def fit_elo_baseline(ctx: LabContext) -> Pipeline:
    training = ctx.train
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=.3, max_iter=1000))])
    pipeline.fit(training[ELO_COLUMNS], training.outcome, model__sample_weight=train_weights(training))
    return pipeline


def export_elo_baseline(pipeline: Pipeline) -> dict:
    columns = ELO_COLUMNS
    scaler, logistic = pipeline.named_steps["scale"], pipeline.named_steps["model"]
    if [int(c) for c in logistic.classes_] != [0, 1, 2]:
        raise ValueError(f"baseline class order must be loss/draw/win (0, 1, 2), got {list(logistic.classes_)}")
    return {"columns": columns, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": logistic.coef_.tolist(), "intercept": logistic.intercept_.tolist(),
            "description": "Elo 차이와 홈 여부 두 개만 쓰는 로지스틱 회귀(비교 기준)"}


def input_stats(ctx: LabContext) -> dict:
    train = ctx.train
    stats = {}
    for name in ctx.numeric:
        series = pd.to_numeric(train[name], errors="coerce")
        valid = series.dropna()
        quantiles = valid.quantile([.05, .25, .5, .75, .95]).tolist() if len(valid) else [None] * 5
        stats[name] = {"min": clean(valid.min()) if len(valid) else None, "p05": clean(quantiles[0]), "p25": clean(quantiles[1]),
                       "median": clean(quantiles[2]), "p75": clean(quantiles[3]), "p95": clean(quantiles[4]),
                       "max": clean(valid.max()) if len(valid) else None, "mean": clean(valid.mean()) if len(valid) else None,
                       "std": clean(valid.std()) if len(valid) > 1 else None, "missing": clean(series.isna().mean())}
    return stats


def formation_history(ctx: LabContext, frame: pd.DataFrame) -> list[dict]:
    """Per row: how often each *known* formation was used in the club's previous 20 matches (all EPL matches).

    This is exactly what `formation_used_last20` was computed from, so the page can recompute that input when a
    user tries a different formation on a real match."""
    out = []
    for team, date in zip(frame.team, frame.date):
        state = ctx.states[team]
        before = int(np.searchsorted(np.asarray(state.dates, dtype="datetime64[ns]"), np.datetime64(date), side="left"))
        counts: dict[str, int] = {}
        for formation in state.formations[max(0, before - 20):before]:
            if formation != "__missing__":
                counts[formation] = counts.get(formation, 0) + 1
        out.append(counts)
    return out


def heldout_block(ctx: LabContext, elo: Pipeline) -> dict:
    """Matches the playground model never saw, with raw inputs and the Python model's own probabilities."""
    held = ctx.heldout.sort_values(["date", "match_id", "team"]).reset_index(drop=True).copy()
    held[ctx.numeric] = held[ctx.numeric].round(ROUND)
    probability = ctx.model.predict_proba(held)
    elo_probability = elo.predict_proba(held[ELO_COLUMNS])
    columns = [*META_COLUMNS, "team", "opponent", *[c for c in CATEGORIES if c not in {"team", "opponent"}], *ctx.numeric]
    rows = [[clean(row[column]) for column in columns] for row in held[columns].to_dict("records")]
    return {"columns": columns, "rows": rows, "py_proba": np.round(probability, 9).tolist(),
            "py_elo": np.round(elo_probability, 9).tolist(), "formation_history": formation_history(ctx, held)}


def recent_lineup_rows(ctx: LabContext, team: str, count: int = 10) -> pd.DataFrame:
    """The team's most recent *unique* matches that carry its lineup inputs.

    Lineups exist only for matches of the six clubs, so for any other club these are its matches against the six
    clubs (possibly years apart). A match between two of the six clubs appears in both clubs' rows; it is kept once."""
    rows = ctx.rows
    as_team = rows[rows.team.eq(team)][["match_id", "date", *[f"team_lineup_{k}" for k in LINEUP_KEYS]]]
    as_team.columns = ["match_id", "date", *LINEUP_KEYS]
    as_opponent = rows[rows.opponent.eq(team)][["match_id", "date", *[f"opponent_lineup_{k}" for k in LINEUP_KEYS]]]
    as_opponent.columns = ["match_id", "date", *LINEUP_KEYS]
    both = pd.concat([as_team, as_opponent]).drop_duplicates("match_id")
    return both.sort_values(["date", "match_id"]).tail(count)


def typical_lineup(ctx: LabContext, team: str, count: int = 10) -> dict:
    """Average lineup-derived inputs over a team's most recent matches that have lineup data."""
    recent = recent_lineup_rows(ctx, team, count)
    return {key: clean(recent[key].mean()) if recent[key].notna().any() else None for key in LINEUP_KEYS}


def team_snapshot(ctx: LabContext, team: str, kickoff: pd.Timestamp) -> dict:
    state = ctx.states[team]
    known = [f for f in state.formations[-20:] if f != "__missing__"]
    recent = recent_lineup_rows(ctx, team)
    return {"home": {key: clean(value) for key, value in state.predate(kickoff, True).items()},
            "away": {key: clean(value) for key, value in state.predate(kickoff, False).items()},
            "formations_last20": known, "last_formation": state.formations[-1] if state.formations else "__missing__",
            "lineup": typical_lineup(ctx, team), "last_match": str(state.dates[-1].date()) if state.dates else None,
            "lineup_window": {"matches": int(len(recent)),
                              "first": str(recent.date.min().date()) if len(recent) else None,
                              "last": str(recent.date.max().date()) if len(recent) else None}}


def scenario_block(ctx: LabContext) -> dict:
    last = ctx.league.date.max()
    kickoff = last + pd.Timedelta(days=7)
    season = ctx.league[ctx.league.date.eq(last)].season.iloc[0]
    current = sorted(set(ctx.league[ctx.league.season.eq(season)][["home_team", "away_team"]].to_numpy().ravel()))
    clubs = list(ctx.report["protocol"]["clubs"])
    snapshots = {team: team_snapshot(ctx, team, kickoff) for team in current if team in ctx.states}
    counts = ctx.rows[ctx.rows.formation.ne("__missing__")].formation.value_counts()
    return {"clubs": clubs, "opponents": [t for t in current if t in snapshots], "snapshots": snapshots,
            "latest_match_date": str(last.date()), "nominal_kickoff": str(kickoff.date()), "latest_season": season,
            "formations": [{"formation": f, "matches": int(n)} for f, n in counts.items()],
            "diff_bases": DIFF_BASES, "lineup_keys": LINEUP_KEYS, "formation_numeric": FORMATION_NUMERIC,
            "note": ("두 팀의 경기 전 상태(Elo·최근 성적·휴식일 등)는 가장 최근 경기까지의 기록입니다. 선발 11명 관련 값은 "
                     "실제 선발을 알 수 없어, 그 팀이 이 데이터에 나온 최근 경기(최대 10경기)의 선발 기록 평균으로 채웁니다. 이 "
                     "데이터에는 6개 구단이 치른 경기만 있어서, 6개 구단이 아닌 팀은 6개 구단과 만난 경기만 쓰이며 오래된 경기가 "
                     "섞일 수 있습니다.")}


def fixed_model_scores(ctx: LabContext, held: dict) -> dict:
    """Scores on the unseen rows of the *fixed* playground model (official reports refit each season), of the Elo
    baseline and, where odds exist, of the betting market -- all with the pipeline's fixture-weighted cohort metrics."""
    from ..focused_model import cohort_metrics
    frame = ctx.heldout.sort_values(["date", "match_id", "team"]).reset_index(drop=True)
    probability, elo = np.asarray(held["py_proba"]), np.asarray(held["py_elo"])
    result = {"overall": cohort_metrics(frame, probability), "by_season": {},
              "elo": {"overall": cohort_metrics(frame, elo), "by_season": {}}}
    for season in sorted(frame.season.unique()):
        mask = frame.season.eq(season).to_numpy()
        result["by_season"][season] = cohort_metrics(frame[mask], probability[mask])
        result["elo"]["by_season"][season] = cohort_metrics(frame[mask], elo[mask])
    market = frame[["market_loss", "market_draw", "market_win"]].to_numpy(float)
    ok = np.isfinite(market).all(1)
    if ok.any():
        result["market"] = {"rows": int(ok.sum()), "odds": cohort_metrics(frame[ok], market[ok]),
                            "model": cohort_metrics(frame[ok], probability[ok]), "elo": cohort_metrics(frame[ok], elo[ok])}
    return result


def raw_input_owner(pre: dict) -> list[str]:
    """Raw model input behind each transformed column: [one-hot blocks][numeric][missing indicators] (as engine.js)."""
    owner = [c["name"] for c in pre["categorical"] for _ in c["categories"]] + list(pre["numeric"])
    return owner + [pre["numeric"][i] for i in pre["indicator_of"]]


def win_contributions(model, frame: pd.DataFrame, owner: list[str]) -> tuple[pd.DataFrame, float]:
    """Tree-interpreter attribution of P(win): bias + row sum == predict_proba[:, win] for every row.

    Each split on a path moves the node's win share; the move is credited to the raw input behind the split column."""
    from scipy import sparse
    forest = model.estimator_
    x = np.asarray(model.processor_.transform(frame), dtype=np.float32)
    win = [int(c) for c in forest.classes_].index(2)
    total, bias = np.zeros((len(frame), x.shape[1])), 0.0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        share = tree.value[:, 0, :] / tree.value[:, 0, :].sum(axis=1, keepdims=True)
        share = share[:, win]
        parent = np.full(tree.node_count, -1)
        inner = np.flatnonzero(tree.children_left != -1)
        parent[tree.children_left[inner]] = inner
        parent[tree.children_right[inner]] = inner
        child = np.flatnonzero(parent >= 0)
        step = sparse.csr_matrix((share[child] - share[parent[child]], (child, tree.feature[parent[child]])),
                                 shape=(tree.node_count, x.shape[1]))
        total += (estimator.decision_path(x) @ step).toarray()
        bias += share[0]
    total /= len(forest.estimators_)
    by_input = pd.DataFrame(total, columns=owner).T.groupby(level=0, sort=False).sum().T
    return by_input, bias / len(forest.estimators_)


def influence_block(ctx: LabContext, pre: dict) -> dict:
    """How much each input moved P(win) on the unseen rows (mean |contribution|), alone and as team/opponent/diff pairs."""
    frame = ctx.heldout.sort_values(["date", "match_id", "team"]).reset_index(drop=True).copy()
    frame[ctx.numeric] = frame[ctx.numeric].round(ROUND)
    contributions, bias = win_contributions(ctx.model, frame, raw_input_owner(pre))
    pairs = {}
    for name in ctx.numeric:
        if name.startswith("team_") and f"opponent_{name[5:]}" in ctx.numeric:
            base = name[5:]
            members = [m for m in [name, f"opponent_{base}", f"diff_{base}"] if m in contributions]
            pairs[base] = round(float(contributions[members].sum(axis=1).abs().mean()), ROUND)
    return {"rows": int(len(frame)), "bias_win": round(float(bias), ROUND),
            "by_input": {name: round(float(v), ROUND) for name, v in contributions.abs().mean().items()},
            "pairs": pairs}


def build(ctx: LabContext) -> dict:
    elo = fit_elo_baseline(ctx)
    held = heldout_block(ctx, elo)
    preprocessor = export_preprocessor(ctx.model)
    return {
        "model": {"spec": ctx.spec, "pre": preprocessor, "forest": export_forest(ctx.model),
                  "train_rows": int(ctx.train_mask.sum()), "train_matches": int(ctx.train.match_id.nunique()),
                  "train_first_date": str(ctx.train.date.min().date()), "train_last_date": str(ctx.train.date.max().date()),
                  "heldout_first_date": str(ctx.heldout.date.min().date()), "heldout_last_date": str(ctx.heldout.date.max().date()),
                  "heldout_rows": int((~ctx.train_mask).sum()), "heldout_matches": int(ctx.heldout.match_id.nunique()),
                  "test_start": ctx.test_start},
        "elo_baseline": export_elo_baseline(elo),
        "stats": input_stats(ctx),
        "heldout": held,
        "scenario": scenario_block(ctx),
        "fixed_model_scores": fixed_model_scores(ctx, held),
        "influence": influence_block(ctx, preprocessor),
        "official_test": {name: ctx.report["test"][name] for name in ["selected_with_formation", "elo_logistic", "always_team_win", "untuned_rf"]},
    }
