"""Reproducible randomized forest tuning on forward season folds for selected clubs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .focused_data import CLUBS, write_json
from .focused_features import CATEGORIES, FORMATION_NUMERIC, build_focused_features

PROBS = ["p_loss", "p_draw", "p_win"]
SEARCH_SPACE = {"n_estimators": [200, 350, 500], "max_depth": [5, 8, 12, 18, None],
                "min_samples_leaf": [2, 5, 10, 20, 40], "min_samples_split": [2, 10, 30],
                "max_features": ["sqrt", .3, .6, 1.0], "max_samples": [.65, .85, 1.0],
                "criterion": ["gini", "entropy"], "class_weight": [None, "balanced_subsample"]}


def feature_columns(numeric: list[str], with_formation: bool):
    categories = CATEGORIES if with_formation else ["team", "opponent"]
    numbers = numeric if with_formation else [name for name in numeric if name not in FORMATION_NUMERIC]
    return categories, numbers


def preprocessing(numeric: list[str], with_formation: bool):
    categories, numbers = feature_columns(numeric, with_formation)
    return ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                               ("scale", StandardScaler())]), numbers),
    ])


def make_estimator(spec: dict):
    parameters = {key:value for key,value in spec.items() if key != "family"}
    if spec["family"] == "rf":
        return RandomForestClassifier(**parameters, n_jobs=4, random_state=42)
    if spec["family"] == "logistic":
        return LogisticRegression(**parameters, max_iter=2500, random_state=42)
    if spec["family"] == "hgb":
        return HistGradientBoostingClassifier(**parameters, max_iter=180, learning_rate=.04,
                                               early_stopping=False, random_state=42)
    raise ValueError(spec)


def train_weights(rows: pd.DataFrame) -> np.ndarray:
    # Fixture weights prevent a Big Six derby from counting twice in the global objective.
    recency = np.exp(-np.log(2)*(rows.date.max()-rows.date).dt.days.to_numpy()/1095)
    return rows.fixture_weight.to_numpy()*recency


def cohort_metrics(rows: pd.DataFrame, probability: np.ndarray) -> dict:
    truth = rows.outcome.to_numpy(dtype=int)
    weights = rows.fixture_weight.to_numpy()
    probability = np.clip(probability, 1e-9, 1)
    probability = probability/probability.sum(axis=1, keepdims=True)
    return {"unique_matches": rows.match_id.nunique(), "team_appearances": len(rows),
            "accuracy": float(accuracy_score(truth, probability.argmax(1), sample_weight=weights)),
            "log_loss": float(log_loss(truth, probability, labels=[0,1,2], sample_weight=weights)),
            "brier": float(np.average(np.sum((probability-np.eye(3)[truth])**2, axis=1), weights=weights)),
            "expected_points_mae": float(np.average(np.abs(probability@np.array([0,1,3])-np.array([0,1,3])[truth]), weights=weights))}


class FocusedModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, spec: dict, numeric: list[str], with_formation: bool):
        self.spec, self.numeric, self.with_formation = spec, numeric, with_formation

    def fit(self, rows: pd.DataFrame):
        self.processor_ = preprocessing(self.numeric, self.with_formation)
        x = self.processor_.fit_transform(rows)
        self.estimator_ = make_estimator(self.spec)
        self.estimator_.fit(x, rows.outcome, sample_weight=train_weights(rows))
        return self

    def predict_proba(self, rows: pd.DataFrame):
        raw = self.estimator_.predict_proba(self.processor_.transform(rows))
        out = np.zeros((len(rows),3))
        for index,label in enumerate(self.estimator_.classes_):
            out[:,int(label)] = raw[:,index]
        return out


def specifications(trials: int):
    result = [{"family": "rf", **parameters} for parameters in ParameterSampler(SEARCH_SPACE, n_iter=trials, random_state=42)]
    result += [{"family": "logistic", "C": c} for c in [.003,.03,.3,3.0]]
    result += [{"family": "hgb", "max_leaf_nodes": leaves, "min_samples_leaf": 30, "l2_regularization": l2}
               for leaves in [7,15] for l2 in [5.,20.]]
    return result


def tune(rows: pd.DataFrame, numeric: list[str], folds: list[str], specs: list[dict],
         with_formation: bool, output_dir: Path):
    matrices = []
    for season in folds:
        validation = rows[rows.season.eq(season)]
        if validation.empty:
            raise ValueError(f"Missing validation season {season}")
        training = rows[rows.date < validation.date.min()]
        if training.empty:
            raise ValueError("Empty forward-fold training")
        processor = preprocessing(numeric, with_formation)
        matrices.append((season, training, validation, processor.fit_transform(training), processor.transform(validation)))
    results = []
    for index,spec in enumerate(specs):
        fold_results, probabilities, validation_frames = [], [], []
        for season,training,validation,train_x,val_x in matrices:
            estimator = make_estimator(spec)
            estimator.fit(train_x, training.outcome, sample_weight=train_weights(training))
            probability = np.zeros((val_x.shape[0],3))
            probability[:,estimator.classes_.astype(int)] = estimator.predict_proba(val_x)
            fold_results.append({"season": season,"train_team_rows":len(training),
                                 "train_last_date":str(training.date.max().date()),
                                 "validation_first_date":str(validation.date.min().date()),
                                 **cohort_metrics(validation, probability)})
            probabilities.append(probability)
            validation_frames.append(validation)
        result = {"trial": index+1, "with_formation": with_formation, "spec": spec, "folds": fold_results,
                  "validation": cohort_metrics(pd.concat(validation_frames), np.vstack(probabilities))}
        results.append(result)
        write_json(output_dir / ("trials_formation.json" if with_formation else "trials_no_formation.json"), results)
        print(f"{'formation' if with_formation else 'no formation'} trial {index+1}/{len(specs)} {spec['family']}: logloss={result['validation']['log_loss']:.4f}", flush=True)
    best = min(results, key=lambda result: result["validation"]["log_loss"])
    best_rf = min((result for result in results if result["spec"]["family"] == "rf"), key=lambda result: result["validation"]["log_loss"])
    return {"best": best, "best_rf": best_rf, "trials": len(results)}


def weighted_week_interval(rows: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray,
                           kind: str = "log_loss", repeats: int = 5000):
    y = rows.outcome.to_numpy(dtype=int)
    if kind == "accuracy":
        difference = (candidate.argmax(1)==y).astype(float)-(baseline.argmax(1)==y).astype(float)
    else:
        difference = np.log(np.clip(candidate[np.arange(len(y)),y],1e-9,1))-np.log(np.clip(baseline[np.arange(len(y)),y],1e-9,1))
    frame = pd.DataFrame({"week":rows.date.dt.to_period("W").astype(str).to_numpy(),
                          "gain": difference*rows.fixture_weight.to_numpy(), "weight":rows.fixture_weight.to_numpy()})
    groups = frame.groupby("week")[["gain","weight"]].sum()
    rng = np.random.default_rng(42)
    draws = rng.integers(0,len(groups),size=(repeats,len(groups)))
    means = groups.gain.to_numpy()[draws].sum(1)/groups.weight.to_numpy()[draws].sum(1)
    return {"improvement": float(frame.gain.sum()/frame.weight.sum()), "ci95_low":float(np.quantile(means,.025)),
            "ci95_high":float(np.quantile(means,.975)), "weeks":len(groups), "repeats":repeats,
            "method":"Fixture-weighted paired calendar-week bootstrap"}


def _baselines(training, test):
    ones = np.tile([0.,0.,1.],(len(test),1))
    # Cohort is strong clubs, so always-team-win is more relevant than random guessing.
    output = {"always_team_win":ones}
    priors = []
    for row in test.itertuples():
        cohort = training[training.team.eq(row.team) & training.is_home.eq(row.is_home)]
        counts = np.bincount(cohort.outcome.to_numpy(dtype=int),minlength=3)+5
        priors.append(counts/counts.sum())
    output["team_venue_prior"] = np.array(priors)
    xcols = ["is_home","diff_elo"]
    pipeline = Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.3,max_iter=1000))])
    pipeline.fit(training[xcols],training.outcome,model__sample_weight=train_weights(training))
    output["elo_logistic"] = pipeline.predict_proba(test[xcols])
    return output


def run_focused(data_dir: str | Path = "data/focused", output_dir: str | Path = "artifacts/focused",
                trials: int = 32, folds: list[str] | None = None, test_start: str = "2024/2025") -> dict:
    folds = folds or ["2018/2019","2021/2022","2023/2024"]
    root = Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    rows,numeric,feature_metadata = build_focused_features(data_dir)
    if max(folds)>=test_start:
        raise ValueError("Validation folds must precede test")
    specs = specifications(trials)
    manifest_path = Path(data_dir)/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_hashes = {name:hashlib.sha256((Path(data_dir)/name).read_bytes()).hexdigest()
                    for name in ["matches.csv","league_context.csv"]}
    protocol = {"created_utc":datetime.now(timezone.utc).isoformat(),"clubs":CLUBS,
                "data_hashes":input_hashes,"folds":folds,"test_start":test_start,
                "rf_trials_per_variant":trials,"other_trials_per_variant":8,"search_space":SEARCH_SPACE,
                "sampled_configurations":specs,"seed":42,"feature_version":2,"objective":"Fixture-weighted forward-validation log loss",
                "with_formation_and_ablated":True,"half_life_days":1095,
                "test_refit":"Before each test season, all earlier seasons; configurations frozen",
                "honesty_note":"2024-2026 outcomes were previously evaluated in v2. This is a retrospective held-out comparison, not a fresh prospective test.",
                "scope":"Only selected-club rows are supervised. Non-cohort opponent league histories are context.",
                "recommendation_rule":"Exploratory formation scores only; no causal optimal formation claimed"}
    path = root/"protocol.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        without_time = lambda p:{k:v for k,v in p.items() if k!="created_utc"}
        if without_time(old)!=without_time(protocol):
            raise ValueError("Protocol changed; use a separate output directory")
        protocol = old
    write_json(path,protocol)
    feature_metadata["recency_weight_effective_sizes"] = {}
    for team,group in rows.groupby("team"):
        weights = train_weights(group)
        feature_metadata["recency_weight_effective_sizes"][team] = {
            "raw_appearances":len(group),"weight_effective_sample_size":float(weights.sum()**2/(weights**2).sum()),
            "note":"Kish weight effective size; not independent games or a causal overlap measure"}
    write_json(root/"features.json",feature_metadata)
    rows.to_csv(root/"feature_rows.csv",index=False,date_format="%Y-%m-%d")
    with threadpool_limits(limits=4):
        selection = {}
        for variant,formation in [("with_formation",True),("without_formation",False)]:
            selection[variant] = tune(rows,numeric,folds,specs,formation,root)
        selection["frozen_before_test_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root/"selection.json",selection)
        print("Tuning finished; configurations frozen before test scoring.",flush=True)
        default_rf = {"family":"rf","n_estimators":350,"max_depth":7,"min_samples_leaf":10,"max_features":.8}
        models = {
            "selected_with_formation":(selection["with_formation"]["best"]["spec"],True),
            "selected_without_formation":(selection["without_formation"]["best"]["spec"],False),
            "tuned_rf":(selection["with_formation"]["best_rf"]["spec"],True),
            "untuned_rf":(default_rf,True),
            # Same configuration as selected_with_formation, so the formation gain is not confounded by model family.
            "same_spec_without_formation":(selection["with_formation"]["best"]["spec"],False),
        }
        all_outputs = []
        test_by_season = {}
        for season in sorted(s for s in rows.season.unique() if s>=test_start):
            test = rows[rows.season.eq(season)].copy()
            training = rows[rows.date<test.date.min()]
            probabilities = _baselines(training,test)
            for name,(spec,formation) in models.items():
                model = FocusedModel(spec,numeric,formation).fit(training)
                probabilities[name] = model.predict_proba(test)
            test_by_season[season] = {name:cohort_metrics(test,p) for name,p in probabilities.items()}
            for name,p in probabilities.items():
                for i,column in enumerate(PROBS):
                    test[f"{name}_{column}"] = p[:,i]
            all_outputs.append(test)
            print(f"Scored retrospective holdout {season}: {test.match_id.nunique()} games / {len(test)} club rows",flush=True)
        test = pd.concat(all_outputs,ignore_index=True)
        names = list(probabilities)
        proba = {name:test[[f"{name}_{c}" for c in PROBS]].to_numpy() for name in names}
        scores = {name:cohort_metrics(test,p) for name,p in proba.items()}
        scores["uniform_random"] = {**cohort_metrics(test,np.full((len(test),3),1/3)),"accuracy":1/3}
        comparisons = {}
        for label,candidate,reference in [("tuning_gain","tuned_rf","untuned_rf"),
                                           ("formation_gain","selected_with_formation","same_spec_without_formation"),
                                           ("formation_gain_separately_tuned","selected_with_formation","selected_without_formation"),
                                           ("selected_vs_elo","selected_with_formation","elo_logistic"),
                                           ("selected_vs_always_win","selected_with_formation","always_team_win")]:
            comparisons[label] = {metric:weighted_week_interval(test,proba[candidate],proba[reference],metric)
                                  for metric in ["accuracy","log_loss"]}
        by_team = {}
        for team in CLUBS:
            mask = test.team.eq(team).to_numpy()
            subset = test[mask].copy()
            subset["fixture_weight"] = 1.0
            by_team[team] = {name:cohort_metrics(subset,p[mask]) for name,p in proba.items()}
        market = test[[f"market_{label}" for label in ["loss","draw","win"]]].to_numpy()
        available = np.isfinite(market).all(1)
        market_comparison = {"team_rows":int(available.sum()),
                             "market":cohort_metrics(test[available],market[available]),
                             "model_same_subset":cohort_metrics(test[available],proba["selected_with_formation"][available]),
                             "note":"Pre-kickoff odds provide additional information; not a betting-profit claim"}
        known = test.formation.ne("__missing__").to_numpy()
        known_comparison = {"team_rows":int(known.sum()),
                            "with_formation":cohort_metrics(test[known],proba["selected_with_formation"][known]),
                            "without_formation":cohort_metrics(test[known],proba["same_spec_without_formation"][known])}
        formations = rows.groupby(["team","formation"]).size().rename("appearances").reset_index()
        formations.to_csv(root/"formation_counts.csv",index=False)
        report = {"experiment":"focused_six_clubs_v3","data":manifest,"protocol":protocol,"features":feature_metadata,
                  "selection":selection,"test":scores,"test_by_season":test_by_season,"test_by_team":by_team,
                  "comparisons":comparisons,"market_reference":market_comparison,"known_formation_subset":known_comparison,
                  "tactical_recommendation_validated":False,
                  "limitations":["Club focus alone does not increase games per club; longer history does.",
                                 "Historic squads/managers differ. Recency weighting cannot identify causal effects.",
                                 "No timestamped injury/availability snapshots or guaranteed pre-kickoff formation timestamps.",
                                 "Player histories cover previously collected fixtures, not full careers.",
                                 "League-context opponent histories are collected separately from the selected supervised rows.",
                                 "Holdout seasons already appeared in v2; retrospective test with no v3 tuning on them."],
                  "environment":{"python":platform.python_version(),"sklearn":sklearn.__version__}}
        test.to_csv(root/"test_predictions.csv",index=False,date_format="%Y-%m-%d")
        # Importance is from validation, never a reason to revisit configurations using test.
        final_fold = folds[-1]
        validation = rows[rows.season.eq(final_fold)]
        training = rows[rows.date<validation.date.min()]
        rf_model = FocusedModel(selection["with_formation"]["best_rf"]["spec"],numeric,True).fit(training)
        transformed = rf_model.processor_.transform(validation)
        feature_names = rf_model.processor_.get_feature_names_out()
        # Preselect 30 columns by training impurity importance; permutation uses validation only.
        columns = np.argsort(rf_model.estimator_.feature_importances_)[-30:]
        baseline_loss = log_loss(validation.outcome,rf_model.estimator_.predict_proba(transformed),
                                 labels=[0,1,2],sample_weight=validation.fixture_weight)
        rng = np.random.default_rng(42)
        importance_records = []
        for column in columns:
            gains = []
            for repeat in range(3):
                perturbed = transformed.copy()
                perturbed[:,column] = rng.permutation(perturbed[:,column])
                loss = log_loss(validation.outcome,rf_model.estimator_.predict_proba(perturbed),
                                labels=[0,1,2],sample_weight=validation.fixture_weight)
                gains.append(loss-baseline_loss)
            importance_records.append({"feature":feature_names[column],"validation_log_loss_importance":float(np.mean(gains)),
                                       "repeat_sd":float(np.std(gains))})
        importance_frame = pd.DataFrame(importance_records).sort_values("validation_log_loss_importance",ascending=False)
        importance_frame.to_csv(root/"validation_importance.csv",index=False)
        chosen = selection["with_formation"]["best"]["spec"]
        final_model = FocusedModel(chosen,numeric,True).fit(rows)
        report["validation_top_features"] = importance_frame.head(25).to_dict("records")
        joblib.dump({"artifact_version":3,"model":final_model,"report":report},root/"model.joblib",compress=3)
        write_json(root/"metrics.json",report)
    return report
