"""Step 3 payload: how a Random Forest decides, how it was tuned, and how honestly it scores.

Official numbers (tuning trials, test scores, interval comparisons, the per-season-refit test
predictions) are read from the pipeline artifacts in `ctx`.  Three small things are recomputed here
with the pipeline's own functions, and are labelled as such on the page:

* the tree diagram and the "500 trees vote" demo use the playground model `ctx.model`.  It is trained on
  every season before the first test season, so on that first test season it is the *same* model as the
  official one (the tests check this), and the vote demo only uses matches from that season;
* the untuned default Random Forest on the three validation seasons (the official pipeline scored it
  only on the test seasons), using exactly the `tune()` protocol;
* an over-fitting illustration on the last validation fold (selected settings vs. tiny leaves).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..focused_data import CLUBS
from ..focused_model import PROBS, cohort_metrics, preprocessing, train_weights
from .catalog import GROUP_LABEL, GROUPS, describe
from .context import LabContext

# Same configuration as `default_rf` inside focused_model.run_focused (the "untuned" reference);
# tests/test_lab_s3.py checks the two stay identical.
UNTUNED_RF = {"family": "rf", "n_estimators": 350, "max_depth": 7, "min_samples_leaf": 10, "max_features": .8}
HYPER_ORDER = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features",
               "max_samples", "criterion", "class_weight"]
HYPER_TEXT = {
    "n_estimators": ("나무 개수", "숲에 심는 결정나무 수. 많을수록 평균이 안정되지만 계산이 느려집니다."),
    "max_depth": ("최대 깊이", "질문을 몇 단계까지 이어 갈 수 있는지. ‘제한 없음’이면 다른 조건이 멈출 때까지 갈라집니다."),
    "min_samples_leaf": ("잎 최소 크기", "마지막 칸(잎)에 남아야 하는 최소 학습 행 수. 클수록 나무가 소수 사례를 외우지 못합니다."),
    "min_samples_split": ("가르기 최소 크기", "한 칸을 더 나누려면 그 칸에 최소 몇 행이 있어야 하는지."),
    "max_features": ("질문 후보 비율", "질문 하나를 고를 때 살펴보는 입력 열의 비율. 작을수록 나무끼리 더 달라집니다."),
    "max_samples": ("나무마다 뽑는 행 비율", "나무마다 학습 행을 중복 허용으로 다시 뽑을 때의 표본 크기(전체 대비)."),
    "criterion": ("가르는 기준", "질문이 얼마나 좋은지 재는 공식. gini(지니 불순도)와 entropy(정보량) 두 가지."),
    "class_weight": ("결과별 균형", "드문 결과(무승부)에 더 큰 무게를 줄지 여부."),
}
MODEL_LABEL = {
    "selected_with_formation": "튜닝한 랜덤 포레스트(선택 모델)",
    "elo_logistic": "Elo 로지스틱(Elo 차이·홈 여부만)",
    "untuned_rf": "튜닝 전 랜덤 포레스트(기본 설정)",
    "team_venue_prior": "구단·홈/원정별 과거 결과 비율",
    "always_team_win": "항상 ‘우리 팀 승’",
    "uniform_random": "무작위(승·무·패 1/3씩)",
    "same_spec_without_formation": "같은 설정, 포메이션 입력 제외",
}
SCOREBOARD = ["selected_with_formation", "elo_logistic", "untuned_rf", "same_spec_without_formation",
              "team_venue_prior", "always_team_win", "uniform_random"]
COMPARISONS = [
    ("tuning_gain", "튜닝 효과", "튜닝한 RF", "튜닝 전 RF"),
    ("formation_gain", "포메이션 입력 효과", "포메이션 포함", "같은 설정, 포메이션 제외"),
    ("selected_vs_elo", "선택 모델 vs Elo 로지스틱", "선택 모델", "Elo 로지스틱"),
    ("selected_vs_always_win", "선택 모델 vs 항상 ‘우리 팀 승’", "선택 모델", "항상 ‘우리 팀 승’"),
]
CONFUSION_MODELS = ["selected_with_formation", "elo_logistic", "untuned_rf"]
OUTCOME_KEYS = ["loss", "draw", "win"]          # index = outcome code 0/1/2
PAIR_TIE = 5e-5                                 # with/without-formation log-loss gap that rounds to 0 at 4 decimals
# The permutation-importance routine in focused_model.run_focused (not stored in metrics.json);
# tests/test_lab_s3.py checks both numbers against that source.
PERM_PRESELECT, PERM_REPEATS = 30, 3
SEASON_END_MONTH = 5                            # an EPL season ends in May of its second year


def r(value, digits: int = 6):
    """Round a float for the payload; NaN/inf -> None.

    Six decimals by default: the page prints shares as 'xx.x%' and losses with up to 4 decimals, and a value stored
    with only 4 decimals would be rounded twice (0.568470 -> 0.5685 -> '56.9%' instead of '56.8%')."""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def metrics_of(block: dict) -> dict:
    return {"accuracy": r(block["accuracy"]), "log_loss": r(block["log_loss"]), "brier": r(block["brier"]),
            "rows": int(block["team_appearances"]), "matches": int(block["unique_matches"])}


# ------------------------------------------------------------------ transformed columns -> plain Korean
def transformed_columns(processor) -> list[dict]:
    """One entry per column of the fitted ColumnTransformer output, in order, with raw-unit scaling info."""
    one_hot = processor.named_transformers_["categories"]
    numeric_pipe = processor.named_transformers_["numeric"]
    imputer, scaler = numeric_pipe.named_steps["impute"], numeric_pipe.named_steps["scale"]
    numeric_columns = list(processor.transformers_[1][2])
    out = []
    for column, categories in zip(one_hot.feature_names_in_, one_hot.categories_):
        out += [{"kind": "onehot", "source": str(column), "category": str(c)} for c in categories]
    first_numeric = len(out)
    medians = np.asarray(imputer.statistics_, dtype=float)
    out += [{"kind": "numeric", "source": name, "median": medians[i]} for i, name in enumerate(numeric_columns)]
    indicators = imputer.indicator_.features_ if imputer.indicator_ is not None else []
    out += [{"kind": "missing", "source": numeric_columns[int(i)]} for i in indicators]
    names = [str(n) for n in processor.get_feature_names_out()]
    assert len(names) == len(out)
    for k, entry in enumerate(out):
        if entry["kind"] == "onehot":
            expected = f"categories__{entry['source']}_{entry['category']}"
        elif entry["kind"] == "numeric":
            expected = f"numeric__{entry['source']}"
        else:
            expected = f"numeric__missingindicator_{entry['source']}"
        assert names[k] == expected, (names[k], expected)
        entry["name"] = names[k]
        if k >= first_numeric:
            entry["mean"], entry["scale"] = float(scaler.mean_[k - first_numeric]), float(scaler.scale_[k - first_numeric])
    return out


def _category_text(column: str, category: str) -> str:
    return "기록 없음" if category == "__missing__" else category.replace("|", " · ")


def column_label(entry: dict) -> tuple[str, str]:
    """(Korean label, catalog group key) for one transformed column."""
    if entry["kind"] == "onehot":
        info = describe(entry["source"])
        return f"{info['label']} = {_category_text(entry['source'], entry['category'])}", info["group"]
    info = describe(entry["source"])
    if entry["kind"] == "missing":
        return f"{info['label']} · 비어 있음 표시", info["group"]
    return info["label"], info["group"]


def fmt_raw(value: float, kind: str) -> str:
    if kind == "count":
        return f"{math.floor(value + 1e-9):d}"
    magnitude = abs(value)
    digits = 0 if magnitude >= 100 else 1 if magnitude >= 10 else 2
    text = f"{value:.{digits}f}"
    return text.replace("-", "−")


def node_question(entry: dict, threshold: float) -> dict:
    """Plain-Korean question for a split `x <= threshold` (sklearn sends <= to the left child)."""
    source = entry["source"]
    info = describe(source)
    if entry["kind"] == "onehot":
        category = entry["category"]
        text = {"team": f"우리 팀이 {category}인가?", "opponent": f"상대 팀이 {category}인가?",
                "formation": f"선발 포메이션이 {_category_text(source, category)}인가?",
                "previous_formation": f"직전 경기 포메이션이 {_category_text(source, category)}인가?",
                "team_formation": f"팀·포메이션 조합이 {_category_text(source, category)}인가?"}[source]
        return {"q": text, "yes": "right", "raw": None, "unit": ""}
    raw = entry["mean"] + threshold * entry["scale"]
    if entry["kind"] == "missing":
        return {"q": f"{info['label']} 값이 비어 있는가?", "yes": "right", "raw": r(raw), "unit": ""}
    if source == "is_home":
        return {"q": "홈 경기인가?", "yes": "right", "raw": r(raw), "unit": ""}
    if info["kind"] == "binary":
        return {"q": f"{info['label']}: 예(1)인가?", "yes": "right", "raw": r(raw), "unit": ""}
    unit = info["unit"] if info["unit"] not in {"비율", "0/1"} else ""
    return {"q": f"{info['label']}: {fmt_raw(raw, info['kind'])}{unit} 이하인가?", "yes": "left", "raw": r(raw), "unit": unit}


# ------------------------------------------------------------------ B. one real tree + the vote
def tree_block(ctx: LabContext, columns: list[dict], depth_shown: int = 3) -> dict:
    forest = ctx.model.estimator_
    estimator = forest.estimators_[0]
    tree = estimator.tree_
    missing_share = ctx.train[ctx.numeric].isna().mean()
    nodes = []

    def visit(node: int, depth: int, path: str):
        value = tree.value[node, 0, :]
        value = value / value.sum()
        item = {"id": path, "depth": depth, "n": int(tree.n_node_samples[node]),
                "w": r(tree.weighted_n_node_samples[node], 2), "share": [r(v, 3) for v in value],
                "leaf": bool(tree.children_left[node] == -1)}
        if not item["leaf"] and depth < depth_shown:
            entry = columns[int(tree.feature[node])]
            question = node_question(entry, float(tree.threshold[node]))
            label, group = column_label(entry)
            item.update(question=question["q"], yes=question["yes"], feature=entry["name"], label=label, group=group,
                        raw=question["raw"], scaled=r(tree.threshold[node]))
            if entry["kind"] == "numeric" and missing_share.get(entry["source"], 0) > 0:
                median_raw = entry["median"]
                item["missing_note"] = {"share": r(missing_share[entry["source"]], 3), "median": r(median_raw),
                                        "median_text": fmt_raw(median_raw, describe(entry["source"])["kind"]),
                                        "goes": "left" if median_raw <= question["raw"] + 1e-12 else "right"}
            nodes.append(item)
            visit(int(tree.children_left[node]), depth + 1, path + "L")
            visit(int(tree.children_right[node]), depth + 1, path + "R")
        else:
            if not item["leaf"]:
                item["more"] = True          # the real tree keeps asking below this box
            nodes.append(item)

    visit(0, 0, "")
    depths = [e.get_depth() for e in forest.estimators_]
    leaves = [e.get_n_leaves() for e in forest.estimators_]
    return {"nodes": nodes, "shown_depth": depth_shown, "depth": int(estimator.get_depth()), "leaves": int(estimator.get_n_leaves()),
            "train_rows": int(ctx.train_mask.sum()), "root_distinct_rows": int(tree.n_node_samples[0]),
            "n_columns": len(columns), "max_features": ctx.spec["max_features"],
            "candidates_per_question": int(forest.estimators_[0].max_features_),   # what sklearn actually used
            "forest_depth_median": float(np.median(depths)), "forest_depth_range": [int(min(depths)), int(max(depths))],
            "forest_leaves_median": float(np.median(leaves)), "n_trees": len(forest.estimators_),
            "train_last_date": str(ctx.train.date.max().date())}


def per_tree_probabilities(ctx: LabContext, rows: pd.DataFrame) -> np.ndarray:
    """trees x rows x 3 (loss, draw, win) — the forest's predict_proba is exactly their mean."""
    forest = ctx.model.estimator_
    x = ctx.model.processor_.transform(rows)
    out = np.zeros((len(forest.estimators_), len(rows), 3))
    for k, estimator in enumerate(forest.estimators_):
        out[k][:, forest.classes_.astype(int)] = estimator.predict_proba(x)
    return out


def _match_label(row) -> dict:
    return {"match_id": str(row.match_id), "date": str(row.date.date()), "team": row.team, "opponent": row.opponent,
            "home": int(row.is_home), "gf": int(row.goals_for), "ga": int(row.goals_against), "outcome": int(row.outcome)}


def vote_block(ctx: LabContext) -> dict:
    """The forest on the first test season: per-tree votes for a few real matches + single tree vs forest."""
    season = ctx.test_start
    rows = ctx.rows[ctx.rows.season.eq(season)].sort_values(["date", "match_id", "team"]).reset_index(drop=True)
    per_tree = per_tree_probabilities(ctx, rows)
    forest = per_tree.mean(axis=0)
    truth = rows.outcome.to_numpy(dtype=int)
    predicted = forest.argmax(1)
    picks, used = [], set()

    def take(key, title, order):
        for index in order:
            if rows.match_id.iat[index] not in used:
                used.add(rows.match_id.iat[index])
                picks.append((key, title, int(index)))
                return

    spread = per_tree[:, :, 2].std(axis=0)
    take("split", "나무들의 의견이 가장 크게 갈린 경기", np.argsort(-spread))
    sure = np.where((truth == 2) & (predicted == 2), forest[:, 2], -1)
    take("sure", "자신 있게 맞힌 경기", np.argsort(-sure))
    upset = np.where(truth == 0, forest[:, 2], -1)
    take("upset", "‘승’을 크게 봤는데 진 경기", np.argsort(-upset))
    draw = np.where(truth == 1, forest[:, 1], -1)
    take("draw", "무승부로 끝난 경기", np.argsort(-draw))
    # The official (per-season refit) predictions over the whole test period: how often 'draw' was the pick.
    test = ctx.test_pred
    official = test[[f"selected_with_formation_{c}" for c in PROBS]].to_numpy()
    official_pick = {(str(m), t): int(p) for m, t, p in zip(test.match_id, test.team, official.argmax(1))}
    examples = []
    for key, title, index in picks:
        row = rows.iloc[index]
        examples.append({"key": key, "title": title, **_match_label(row),
                         "official_pick": official_pick[(str(row.match_id), row.team)],
                         "forest": [r(v) for v in forest[index]],
                         "trees": {name: [r(v, 3) for v in per_tree[:, index, k]] for k, name in enumerate(OUTCOME_KEYS)},
                         "tree0": [r(v, 3) for v in per_tree[0, index]]})
    tree_scores = [cohort_metrics(rows, per_tree[k]) for k in range(per_tree.shape[0])]
    forest_scores = cohort_metrics(rows, forest)
    zero_actual = per_tree[:, np.arange(len(rows)), truth] <= 0
    accuracy = np.array([s["accuracy"] for s in tree_scores])
    loss = np.array([s["log_loss"] for s in tree_scores])
    return {"season": season, "rows": len(rows), "matches": int(rows.match_id.nunique()), "examples": examples,
            "test_rows": int(len(test)), "test_draw_picks": int((official.argmax(1) == 1).sum()),
            "tree_accuracy": [r(v) for v in accuracy], "tree_log_loss": [r(v) for v in loss],
            "forest": {k: r(forest_scores[k]) for k in ["accuracy", "log_loss", "brier"]},
            "trees_beating_forest_accuracy": int((accuracy > forest_scores["accuracy"]).sum()),
            "trees_beating_forest_log_loss": int((loss < forest_scores["log_loss"]).sum()),
            "tree0": {k: r(tree_scores[0][k]) for k in ["accuracy", "log_loss"]},
            "zero_on_actual": int(zero_actual.sum()), "tree_predictions": int(zero_actual.size),
            "rows_with_some_zero_tree": int(zero_actual.any(axis=0).sum())}


# ------------------------------------------------------------------ C. how it was evaluated
def split_block(ctx: LabContext) -> dict:
    rows = ctx.rows
    seasons = sorted(rows.season.unique())
    best = ctx.selection["with_formation"]["best"]
    folds = [{"season": f["season"], "train_rows": int(f["train_team_rows"]), "train_last_date": f["train_last_date"],
              "val_rows": int(f["team_appearances"]), "val_matches": int(f["unique_matches"]),
              "val_first_date": f["validation_first_date"],
              "train_first_season": seasons[0], "train_last_season": seasons[seasons.index(f["season"]) - 1]}
             for f in best["folds"]]
    tests = []
    for season, block in ctx.report["test_by_season"].items():
        first, last = rows[rows.season.eq(season)].date.min(), rows[rows.season.eq(season)].date.max()
        training = rows[rows.date < first]
        sel = block["selected_with_formation"]
        # still being played: the latest season in the data whose last match is before May of its second year
        in_progress = season == seasons[-1] and last < pd.Timestamp(int(season[5:9]), SEASON_END_MONTH, 1)
        tests.append({"season": season, "rows": int(sel["team_appearances"]), "matches": int(sel["unique_matches"]),
                      "train_rows": int(len(training)), "train_last_season": seasons[seasons.index(season) - 1],
                      "first_date": str(first.date()), "last_date": str(last.date()), "in_progress": bool(in_progress)})
    season_rows = rows.groupby("season").size()
    season_matches = rows.groupby("season").match_id.nunique()
    return {"seasons": seasons, "season_rows": [int(season_rows[s]) for s in seasons],
            "season_matches": [int(season_matches[s]) for s in seasons], "folds": folds, "tests": tests,
            "test_start": ctx.test_start, "half_life_days": int(ctx.report["protocol"]["half_life_days"]),
            "test_rows": int(ctx.report["test"]["selected_with_formation"]["team_appearances"]),
            "test_matches": int(ctx.report["test"]["selected_with_formation"]["unique_matches"]),
            "derby_rows_test": int((ctx.test_pred.fixture_weight < 1).sum())}


def metric_examples(ctx: LabContext) -> dict:
    """Two real test rows (official predictions) worked through accuracy / log loss / Brier."""
    test = ctx.test_pred.sort_values(["date", "match_id", "team"]).reset_index(drop=True)
    p = test[[f"selected_with_formation_{c}" for c in PROBS]].to_numpy()
    truth = test.outcome.to_numpy(dtype=int)
    p_actual = p[np.arange(len(test)), truth]
    examples = []
    for outcome in [2, 1]:
        mask = truth == outcome
        target = np.median(p_actual[mask])
        index = int(np.flatnonzero(mask)[np.argmin(np.abs(p_actual[mask] - target))])
        row, prob = test.iloc[index], p[index]
        onehot = np.eye(3)[outcome]
        examples.append({**_match_label(row), "p": [r(v) for v in prob], "picked": int(prob.argmax()),
                         "p_actual": r(prob[outcome]), "log_loss": r(-math.log(prob[outcome])),
                         "brier": r(float(((prob - onehot) ** 2).sum())), "season": row.season})
    official = ctx.report["test"]["selected_with_formation"]
    return {"examples": examples, "uniform_log_loss": r(math.log(3)), "uniform_brier": r(2 / 3),
            "model_log_loss": r(official["log_loss"]), "model_geo_p": r(math.exp(-official["log_loss"])),
            "model_accuracy": r(official["accuracy"]), "model_brier": r(official["brier"]),
            "rows": int(official["team_appearances"]), "matches": int(official["unique_matches"])}


# ------------------------------------------------------------------ D. tuning
def _value_key(value) -> str:
    return "null" if value is None else str(value)


def trial_rows(trials: list[dict]) -> list[dict]:
    out = []
    for trial in trials:
        spec = trial["spec"]
        out.append({"trial": int(trial["trial"]), "family": spec["family"],
                    "spec": {k: v for k, v in spec.items() if k != "family"},
                    "log_loss": r(trial["validation"]["log_loss"]), "accuracy": r(trial["validation"]["accuracy"]),
                    "brier": r(trial["validation"]["brier"]),
                    "fold_log_loss": [r(f["log_loss"]) for f in trial["folds"]],
                    "fold_accuracy": [r(f["accuracy"]) for f in trial["folds"]]})
    return out


def hyper_summary(trials: list[dict], space: dict) -> list[dict]:
    """Per hyper-parameter: validation log loss by searched value (RF trials only)."""
    rf = [t for t in trials if t["spec"]["family"] == "rf"]
    out = []
    for name in HYPER_ORDER:
        values = []
        for value in space[name]:
            losses = sorted(t["validation"]["log_loss"] for t in rf if t["spec"][name] == value)
            values.append({"value": value, "key": _value_key(value), "n": len(losses),
                           "median": r(np.median(losses)) if losses else None, "min": r(min(losses)) if losses else None,
                           "max": r(max(losses)) if losses else None})
        with_data = [v for v in values if v["n"]]
        best_v = min(with_data, key=lambda v: v["median"])
        worst_v = max(with_data, key=lambda v: v["median"])
        # "separated": every trial with the best-median value beat every trial with any other value
        separated = all(best_v["max"] < v["min"] for v in with_data if v is not best_v)
        out.append({"name": name, "label": HYPER_TEXT[name][0], "desc": HYPER_TEXT[name][1], "values": values,
                    "best": best_v["key"], "worst": worst_v["key"], "spread": r(worst_v["median"] - best_v["median"]),
                    "min_n": int(min(v["n"] for v in with_data)), "separated": bool(separated)})
    return out


def class_weight_story(trials: list[dict]) -> dict:
    """How cleanly the class-weight switch splits the RF trials (the clearest pattern in the search)."""
    rf = sorted((t for t in trials if t["spec"]["family"] == "rf"), key=lambda t: t["validation"]["log_loss"])
    lead = 0
    for trial in rf:
        if trial["spec"]["class_weight"] is not None:
            break
        lead += 1
    groups = {}
    for key, flag in [("off", None), ("on", "balanced_subsample")]:
        losses = [t["validation"]["log_loss"] for t in rf if t["spec"]["class_weight"] == flag]
        groups[key] = {"n": len(losses), "min": r(min(losses)), "max": r(max(losses)), "median": r(np.median(losses))}
    return {"leading_off": lead, "n_rf": len(rf), **groups}


def forward_validation(ctx: LabContext, spec: dict, with_formation: bool = True) -> tuple[list[dict], dict]:
    """Score an RF spec exactly like focused_model.tune(): each fold trains on everything before its season.

    (n_jobs differs from the pipeline, which does not change a seeded forest's predictions.)"""
    rows = ctx.rows
    folds, probabilities, frames = [], [], []
    for season in ctx.report["protocol"]["folds"]:
        validation = rows[rows.season.eq(season)]
        training = rows[rows.date < validation.date.min()]
        processor = preprocessing(ctx.numeric, with_formation)
        train_x, val_x = processor.fit_transform(training), processor.transform(validation)
        estimator = RandomForestClassifier(**{k: v for k, v in spec.items() if k != "family"}, n_jobs=-1, random_state=42)
        estimator.fit(train_x, training.outcome, sample_weight=train_weights(training))
        probability = np.zeros((val_x.shape[0], 3))
        probability[:, estimator.classes_.astype(int)] = estimator.predict_proba(val_x)
        folds.append({"season": season, **cohort_metrics(validation, probability)})
        probabilities.append(probability)
        frames.append(validation)
    return folds, cohort_metrics(pd.concat(frames), np.vstack(probabilities))


def untuned_validation(ctx: LabContext) -> dict:
    """The untuned default (scored officially only on the test seasons) on the three validation folds."""
    folds, total = forward_validation(ctx, UNTUNED_RF)
    return {"spec": {k: v for k, v in UNTUNED_RF.items() if k != "family"},
            "folds": [{"season": f["season"], "log_loss": r(f["log_loss"]), "accuracy": r(f["accuracy"])} for f in folds],
            "log_loss": r(total["log_loss"]), "accuracy": r(total["accuracy"]), "brier": r(total["brier"])}


def overfit_demo(ctx: LabContext, small_leaf: int = 2) -> dict:
    """Selected settings vs. the same settings with tiny leaves, on the last validation fold."""
    rows = ctx.rows
    season = ctx.report["protocol"]["folds"][-1]
    validation = rows[rows.season.eq(season)]
    training = rows[rows.date < validation.date.min()]
    processor = preprocessing(ctx.numeric, True)
    train_x, val_x = processor.fit_transform(training), processor.transform(validation)
    out = []
    for leaf in [small_leaf, int(ctx.spec["min_samples_leaf"])]:
        params = {k: v for k, v in ctx.spec.items() if k != "family"}
        params["min_samples_leaf"] = leaf
        estimator = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
        estimator.fit(train_x, training.outcome, sample_weight=train_weights(training))

        def proba(x):
            p = np.zeros((x.shape[0], 3))
            p[:, estimator.classes_.astype(int)] = estimator.predict_proba(x)
            return p
        fit_score, val_score = cohort_metrics(training, proba(train_x)), cohort_metrics(validation, proba(val_x))
        out.append({"min_samples_leaf": leaf, "train_accuracy": r(fit_score["accuracy"]), "train_log_loss": r(fit_score["log_loss"]),
                    "val_accuracy": r(val_score["accuracy"]), "val_log_loss": r(val_score["log_loss"]),
                    "depth_median": float(np.median([e.get_depth() for e in estimator.estimators_])),
                    "leaves_median": float(np.median([e.get_n_leaves() for e in estimator.estimators_]))})
    return {"season": season, "train_rows": int(len(training)), "val_rows": int(len(validation)), "runs": out}


def design_columns(ctx: LabContext, with_formation: bool) -> int:
    """Input columns after preprocessing, fitted on the playground training rows (what 'max_features' is a share of)."""
    return len(preprocessing(ctx.numeric, with_formation).fit(ctx.train).get_feature_names_out())


def tuning_block(ctx: LabContext) -> dict:
    space = ctx.report["protocol"]["search_space"]
    variants = {}
    for key, trials in [("with", ctx.trials_with), ("without", ctx.trials_without)]:
        sel = ctx.selection["with_formation" if key == "with" else "without_formation"]
        variants[key] = {"trials": trial_rows(trials), "hyper": hyper_summary(trials, space),
                         "n_columns": design_columns(ctx, key == "with"),
                         "class_weight": class_weight_story(trials),
                         "best_trial": int(sel["best"]["trial"]), "best_rf_trial": int(sel["best_rf"]["trial"]),
                         "best_by_family": {fam: min((t for t in trials if t["spec"]["family"] == fam),
                                                     key=lambda t: t["validation"]["log_loss"])["trial"]
                                            for fam in ["rf", "logistic", "hgb"]}}
    pairs = [{"trial": int(a["trial"]), "family": a["spec"]["family"], "with": r(a["validation"]["log_loss"]),
              "without": r(b["validation"]["log_loss"])} for a, b in zip(ctx.trials_with, ctx.trials_without)]
    assert all(a["spec"] == b["spec"] for a, b in zip(ctx.trials_with, ctx.trials_without))
    # without - with, from the unrounded losses; a gap below TIE (invisible at 4 decimals) counts as a tie
    diffs = np.array([b["validation"]["log_loss"] - a["validation"]["log_loss"] for a, b in zip(ctx.trials_with, ctx.trials_without)])
    n_columns = len(ctx.model.processor_.get_feature_names_out())
    return {"space": {k: space[k] for k in HYPER_ORDER}, "order": HYPER_ORDER,
            "text": {k: {"label": v[0], "desc": v[1]} for k, v in HYPER_TEXT.items()},
            "chosen": {k: ctx.spec[k] for k in HYPER_ORDER}, "n_columns": n_columns,
            "rf_trials": int(ctx.report["protocol"]["rf_trials_per_variant"]),
            "other_trials": int(ctx.report["protocol"]["other_trials_per_variant"]),
            "objective": ctx.report["protocol"]["objective"],
            "variants": variants, "pairs": pairs,
            "pairs_summary": {"n": len(pairs), "with_better": int((diffs > PAIR_TIE).sum()),
                              "without_better": int((diffs < -PAIR_TIE).sum()), "ties": int((np.abs(diffs) <= PAIR_TIE).sum()),
                              "median_gap": r(float(np.median(diffs))), "max_abs_gap": r(float(np.abs(diffs).max()))},
            "untuned": untuned_validation(ctx), "overfit": overfit_demo(ctx),
            "selected_validation": metrics_of(ctx.selection["with_formation"]["best"]["validation"]),
            "selected_without_validation": metrics_of(ctx.selection["without_formation"]["best"]["validation"])}


# ------------------------------------------------------------------ E. results
def confusion(test: pd.DataFrame, model: str) -> dict:
    p = test[[f"{model}_{c}" for c in PROBS]].to_numpy()
    truth, predicted = test.outcome.to_numpy(dtype=int), p.argmax(1)
    matrix = np.zeros((3, 3), dtype=int)
    for a, b in zip(truth, predicted):
        matrix[a, b] += 1
    per_class = []
    for k in range(3):
        predicted_k, actual_k = matrix[:, k].sum(), matrix[k, :].sum()
        per_class.append({"outcome": k, "actual": int(actual_k), "predicted": int(predicted_k), "hit": int(matrix[k, k]),
                          "precision": r(matrix[k, k] / predicted_k) if predicted_k else None,
                          "recall": r(matrix[k, k] / actual_k) if actual_k else None})
    draw = p[:, 1]
    return {"model": model, "matrix": matrix.tolist(), "per_class": per_class, "rows": int(len(test)),
            "row_accuracy": r(np.trace(matrix) / len(test)),
            "draw_prob": {"p10": r(np.quantile(draw, .1)), "p50": r(np.median(draw)), "p90": r(np.quantile(draw, .9)),
                          "max": r(draw.max())}}


def calibration(test: pd.DataFrame, model: str, edges=tuple(np.round(np.arange(0, 1.01, .1), 1))) -> list[dict]:
    p = test[f"{model}_p_win"].to_numpy()
    won = test.outcome.eq(2).to_numpy()
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "mean_p": r(p[mask].mean()),
                        "observed": r(won[mask].mean()), "wins": int(won[mask].sum())})
    return out


def results_block(ctx: LabContext) -> dict:
    test_scores = ctx.report["test"]
    scoreboard = []
    for name in SCOREBOARD:
        block = metrics_of(test_scores[name])
        if name == "always_team_win":
            block["log_loss"] = None          # a 0/1 forecast has no usable probability; 10.39 is just the clip at 1e-9
            block["log_loss_raw"] = r(test_scores[name]["log_loss"], 2)
        scoreboard.append({"key": name, "label": MODEL_LABEL[name], **block})
    market = ctx.report["market_reference"]
    comparisons = []
    for key, title, cand, base in COMPARISONS:
        block = ctx.report["comparisons"][key]
        item = {"key": key, "title": title, "candidate": cand, "baseline": base}
        for metric in ["accuracy", "log_loss"]:
            m = block[metric]
            usable = not (key == "selected_vs_always_win" and metric == "log_loss")
            item[metric] = {"value": r(m["improvement"], 5), "low": r(m["ci95_low"], 5), "high": r(m["ci95_high"], 5),
                            "usable": usable, "includes_zero": bool(m["ci95_low"] <= 0 <= m["ci95_high"])}
        item["weeks"], item["repeats"] = int(block["accuracy"]["weeks"]), int(block["accuracy"]["repeats"])
        comparisons.append(item)
    keep = ["selected_with_formation", "elo_logistic", "always_team_win", "untuned_rf"]
    by_season = [{"season": season, **{name: metrics_of(block[name]) for name in keep}}
                 for season, block in ctx.report["test_by_season"].items()]
    by_team = [{"team": team, **{name: metrics_of(ctx.report["test_by_team"][team][name]) for name in keep}}
               for team in CLUBS]
    rows = ctx.rows
    win_share = []
    for fold in ctx.report["protocol"]["folds"]:
        trial = ctx.selection["with_formation"]["best"]
        acc = next(f["accuracy"] for f in trial["folds"] if f["season"] == fold)
        season_rows = rows[rows.season.eq(fold)]
        win_share.append({"season": fold, "phase": "validation", "model_accuracy": r(acc),
                          "always_win": r(np.average(season_rows.outcome.eq(2), weights=season_rows.fixture_weight)),
                          "rows": int(len(season_rows))})
    for season, block in ctx.report["test_by_season"].items():
        win_share.append({"season": season, "phase": "test", "model_accuracy": r(block["selected_with_formation"]["accuracy"]),
                          "always_win": r(block["always_team_win"]["accuracy"]),
                          "rows": int(block["selected_with_formation"]["team_appearances"])})
    # Pooled (all seasons of a phase together, derbies counted 0.5) -- the same way the official scores are computed.
    val_rows = rows[rows.season.isin(ctx.report["protocol"]["folds"])]
    gap = {"val_model": r(ctx.selection["with_formation"]["best"]["validation"]["accuracy"]),
           "val_always": r(np.average(val_rows.outcome.eq(2), weights=val_rows.fixture_weight)),
           "test_model": r(test_scores["selected_with_formation"]["accuracy"]),
           "test_always": r(test_scores["always_team_win"]["accuracy"])}
    test = ctx.test_pred
    prior = test[[f"team_venue_prior_{c}" for c in PROBS]].to_numpy()
    train = ctx.train
    # *_rows_share / mean_p_win count every club row once (like the calibration bins); train / test are fixture-weighted.
    win_rates = {"train": r(np.average(train.outcome.eq(2), weights=train.fixture_weight)),
                 "train_rows_share": r(train.outcome.eq(2).mean()),
                 "train_rows": int(len(train)), "test_rows": int(len(test)),
                 "test": r(np.average(test.outcome.eq(2), weights=test.fixture_weight)),
                 "test_rows_share": r(test.outcome.eq(2).mean()),
                 "mean_p_win": {name: r(test[f"{name}_p_win"].mean()) for name in ["selected_with_formation", "elo_logistic"]}}
    return {"scoreboard": scoreboard, "comparisons": comparisons, "by_season": by_season, "by_team": by_team,
            "win_rates": win_rates, "gap": gap,
            "market": {"rows": int(market["team_rows"]), **{k: r(market["market"][k]) for k in ["accuracy", "log_loss", "brier"]}},
            "confusion": {name: confusion(test, name) for name in CONFUSION_MODELS},
            "calibration": {name: calibration(test, name) for name in ["selected_with_formation", "elo_logistic"]},
            "win_share": win_share, "prior_always_win_share": r(float((prior.argmax(1) == 2).mean())),
            "validation_accuracy": r(ctx.selection["with_formation"]["best"]["validation"]["accuracy"]),
            "labels": MODEL_LABEL}


# ------------------------------------------------------------------ F. what the model looked at
def importance_block(ctx: LabContext, columns: list[dict]) -> dict:
    by_name = {c["name"]: c for c in columns}
    permutation = []
    for item in ctx.report["validation_top_features"]:
        entry = by_name[item["feature"]]
        label, group = column_label(entry)
        permutation.append({"feature": item["feature"], "label": label, "group": group, "group_label": GROUP_LABEL[group],
                            # 6 decimals: the page prints 4, and rounding twice would change the last digit
                            "importance": r(item["validation_log_loss_importance"], 6), "sd": r(item["repeat_sd"], 6)})
    impurity = np.asarray(ctx.model.estimator_.feature_importances_, dtype=float)
    totals, counts, top = {}, {}, {}
    for entry, value in zip(columns, impurity):
        label, group = column_label(entry)
        totals[group] = totals.get(group, 0.0) + value
        counts[group] = counts.get(group, 0) + 1
        if value > top.get(group, ("", -1.0))[1]:
            top[group] = (label, value)
    groups = [{"group": key, "label": label, "desc": desc, "share": r(totals.get(key, 0.0) / impurity.sum()),
               "columns": int(counts.get(key, 0)), "top_label": top.get(key, ("", 0))[0], "top_share": r(top.get(key, ("", 0))[1] / impurity.sum())}
              for key, label, desc in GROUPS]
    formation_columns = [c for c in columns if column_label(c)[1] == "formation"]
    return {"permutation": permutation, "fold": ctx.report["protocol"]["folds"][-1],
            "permutation_preselect": PERM_PRESELECT, "permutation_repeats": PERM_REPEATS,
            "permutation_top_n": len(permutation), "permutation_formation": int(sum(p["group"] == "formation" for p in permutation)),
            "impurity_groups": groups, "n_columns": len(columns), "formation_columns": len(formation_columns),
            "impurity_model_last_date": str(ctx.train.date.max().date())}


def build(ctx: LabContext) -> dict:
    if ctx.model is None:
        raise ValueError("Step 3 needs the fitted playground model (load_context(fit=True)).")
    columns = transformed_columns(ctx.model.processor_)
    test = ctx.report["test"]["selected_with_formation"]
    return {
        "headline": {"accuracy": r(test["accuracy"]), "log_loss": r(test["log_loss"]), "brier": r(test["brier"]),
                     "rows": int(test["team_appearances"]), "matches": int(test["unique_matches"]),
                     "elo_accuracy": r(ctx.report["test"]["elo_logistic"]["accuracy"]),
                     "elo_log_loss": r(ctx.report["test"]["elo_logistic"]["log_loss"]),
                     "always_accuracy": r(ctx.report["test"]["always_team_win"]["accuracy"]),
                     "n_trees": int(ctx.spec["n_estimators"])},
        "tree": tree_block(ctx, columns),
        "vote": vote_block(ctx),
        "split": split_block(ctx),
        "metrics": metric_examples(ctx),
        "tuning": tuning_block(ctx),
        "results": results_block(ctx),
        "importance": importance_block(ctx, columns),
    }
