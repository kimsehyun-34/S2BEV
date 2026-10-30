from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import fetch_statsbomb
from .model import load_model, train
from .recommend import recommend_all, recommend_match
from .report import write_report
from .recent_data import fetch_recent, crosscheck_football_data
from .benchmark import run_benchmark, predict_fixture
from .analysis_report import write_analysis_report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="EPL historical formation recommendation model")
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Fetch free StatsBomb EPL starting formations and results")
    fetch.add_argument("--output", default="data/epl_2015_16.csv")
    fetch.add_argument("--season-id", type=int, default=27)
    fetch.add_argument("--workers", type=int, default=4)
    fetch.add_argument("--ref", default="master", help="Git revision or commit SHA for reproducibility")
    fetch.add_argument("--cache-dir", default="data/raw")
    focused = commands.add_parser("collect-focused", help="Collect long EPL histories for Arsenal and five other selected clubs")
    focused.add_argument("--start-year", type=int, default=2006)
    focused.add_argument("--cutoff", default="2026-10-29")
    focused.add_argument("--workers", type=int, default=4)
    focused.add_argument("--output-dir", default="data/focused")
    focused_fit = commands.add_parser("analyze-focused", help="Randomized RF tuning on forward-season folds, then retrospective test")
    focused_fit.add_argument("--data-dir", default="data/focused")
    focused_fit.add_argument("--output-dir", default="artifacts/focused")
    focused_fit.add_argument("--report-dir", default="reports/focused")
    focused_fit.add_argument("--rf-trials", type=int, default=32)
    focused_report = commands.add_parser("report-focused", help="Rebuild the selected-club analysis without retraining")
    focused_report.add_argument("--data-dir", default="data/focused")
    focused_report.add_argument("--model-dir", default="artifacts/focused")
    focused_report.add_argument("--output-dir", default="reports/focused")
    lab = commands.add_parser("report-lab", help="Build the four-step interactive lab page (data -> preprocessing -> tuned RF -> inference)")
    lab.add_argument("--data-dir", default="data/focused")
    lab.add_argument("--model-dir", default="artifacts/focused")
    lab.add_argument("--output-dir", default="reports/focused")
    recent = commands.add_parser("collect-recent", help="Collect recent ESPN lineups and independently verify scores")
    recent.add_argument("--output", default="data/epl_recent.csv")
    recent.add_argument("--start-year", type=int, default=2019)
    recent.add_argument("--cutoff", default="2026-10-28")
    recent.add_argument("--workers", type=int, default=4)
    benchmark = commands.add_parser("benchmark", help="Frozen season-forward comparison and tactical feasibility report")
    benchmark.add_argument("--data", default="data/epl_recent.csv")
    benchmark.add_argument("--output-dir", default="artifacts/v2")
    benchmark.add_argument("--report-dir", default="reports")
    benchmark.add_argument("--market", default="data/football_data/market_probabilities.csv")
    fit = commands.add_parser("train", help="Legacy single-season RF experiment; use benchmark for v2")
    fit.add_argument("--data", default="data/epl_2015_16.csv")
    fit.add_argument("--output-dir", default="artifacts")
    for name in ["recommend", "report", "predict", "explorer"]:
        command = commands.add_parser(name, help={"recommend": "Explain why optimal-formation recommendations are withheld",
                                                  "report": "Export the frozen model/feasibility audit",
                                                  "explorer": "Restore the interactive historical team/formation explorer",
                                                  "predict": "Forecast a fixture scenario using announced formations"}[name])
        command.add_argument("--model", default="artifacts/model.joblib" if name == "explorer" else "artifacts/v2/model.joblib")
        command.add_argument("--as-of", help="Date after training cutoff; defaults to cutoff + 7 days")
        command.add_argument("--min-team-support", type=int, default=3)
        command.add_argument("--min-league-support", type=int, default=10)
        if name in {"recommend", "predict"}:
            command.add_argument("--team", required=True)
            command.add_argument("--opponent", required=True)
            command.add_argument("--venue", choices=["home", "away"], default="home")
            command.add_argument("--formation", help="Announced starting formation of your team")
            command.add_argument("--opponent-formation", help="Announced opponent starting formation")
            command.add_argument("--output", help="Optional JSON output file")
        else:
            command.add_argument("--output-dir", default="reports")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch":
            result = fetch_statsbomb(args.output, args.season_id, args.workers, args.ref, args.cache_dir)
        elif args.command == "collect-focused":
            from .focused_data import collect_focused
            result = collect_focused(args.start_year, args.cutoff, args.workers, args.output_dir)
        elif args.command == "analyze-focused":
            from .focused_model import run_focused
            from .focused_report import write_focused_report
            result = run_focused(args.data_dir, args.output_dir, args.rf_trials)
            write_focused_report(args.data_dir, args.output_dir, args.report_dir)
        elif args.command == "report-focused":
            from .focused_report import write_focused_report
            result = {"html": str(write_focused_report(args.data_dir, args.model_dir, args.output_dir).resolve())}
        elif args.command == "report-lab":
            from .lab.report import write_lab_report
            result = {"html": str(write_lab_report(args.data_dir, args.model_dir, args.output_dir).resolve())}
        elif args.command == "collect-recent":
            result = {"collection": fetch_recent(args.output, args.start_year, args.cutoff, args.workers),
                      "crosscheck": crosscheck_football_data(args.output)}
        elif args.command == "benchmark":
            result = run_benchmark(args.data, args.output_dir, args.market)
            write_analysis_report(result, args.report_dir)
        elif args.command == "train":
            result = train(args.data, args.output_dir)
        else:
            bundle = load_model(args.model)
            options = dict(as_of=args.as_of, min_team_support=args.min_team_support,
                           min_league_support=args.min_league_support)
            if args.command in {"recommend", "predict"}:
                if bundle["artifact_version"] == 2:
                    if args.command == "recommend":
                        active = [row["team"] for row in bundle["report"]["support_audit"]["teams"]]
                        if args.team not in active or args.opponent not in active or args.team == args.opponent:
                            raise ValueError(f"Use two different active teams: {', '.join(active)}")
                        result = {"team": args.team, "opponent": args.opponent,
                                  "recommended_formation": None, "tactical_status": "not_identified",
                                  "formation_signal_supported": bundle["report"]["formation_signal_supported"],
                                  "reason": "No validated causal formation policy. Use predict for scenario outcome probabilities."}
                    else:
                        result = predict_fixture(bundle, args.team, args.opponent, args.venue == "home",
                                                 args.formation, args.opponent_formation, args.as_of)
                else:
                    result = recommend_match(bundle, args.team, args.opponent, args.venue == "home", **options)
                if args.output:
                    path = Path(args.output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                if bundle["artifact_version"] == 2:
                    if args.command == "explorer":
                        raise ValueError("The historical explorer requires the v1 model: --model artifacts/model.joblib")
                    if args.as_of:
                        raise ValueError("The v2 audit has frozen evaluation dates; use --as-of with predict")
                    path = write_analysis_report(bundle["report"], args.output_dir)
                    result = {"html": str(path.resolve()), "tactical_status": "not_identified"}
                else:
                    results = recommend_all(bundle, **options)
                    path = write_report(results, bundle["report"], args.output_dir)
                    result = {"html": str(path.resolve()), "teams": len(results),
                              "note": "Exploratory associations only; tactical recommendations withheld."}
        # ASCII escaping works on default Windows consoles; saved reports use UTF-8.
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
