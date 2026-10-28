"""Evidence-first report separating outcome prediction from tactical identification."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAMES = {"uniform_random": "균등 랜덤 (기대값)", "home_prior": "홈 결과 빈도 / 홈승 기준",
         "elo_logistic": "Elo + 홈 여부", "legacy_rf_retrained": "기존 RF · 같은 추가 데이터로 재학습",
         "without_formation": "대체 모델 · 포메이션 제외", "with_formation": "대체 모델 · 포메이션 포함"}


def _percent(value):
    return f"{100*value:.2f}%"


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def _html_table(headers, rows):
    escape = lambda value: html.escape(str(value))
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in row)
                                             + '</tr>' for row in rows) + '</tbody></table></div>')


def _figure(report: dict, directory: Path):
    labels = ["Uniform random", "Home prior", "Elo + home", "Original RF, retrained", "Candidate, no formation", "Candidate, formation"]
    keys = list(NAMES)
    accuracy = [100 * report["test"][name]["accuracy"] for name in keys]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = ["#b7c3cf"] * 4 + ["#638ea9", "#249f82"]
    left.barh(labels, accuracy, color=colors)
    left.invert_yaxis()
    left.set_xlim(0, max(accuracy) + 10)
    left.set_xlabel("Match outcome accuracy (%)")
    left.set_title(f"Same {report['data']['test_matches']} held-out matches")
    for index, value in enumerate(accuracy):
        left.text(value + 0.6, index, f"{value:.2f}%", va="center", fontsize=9)
    comparisons = [("Formation vs no formation", "formation_vs_ablation"),
                   ("Candidate vs original RF", "upgraded_vs_legacy"),
                   ("Candidate vs Elo", "upgraded_vs_elo")]
    for index, (name, key) in enumerate(comparisons):
        interval = report["comparisons"][key]["log_loss"]
        value, lower, upper = interval["improvement"], interval["ci_95_low"], interval["ci_95_high"]
        right.errorbar(value, index, xerr=[[max(0, value-lower)], [max(0, upper-value)]],
                       fmt="o", color="#249f82", capsize=5, linewidth=2)
    right.axvline(0, color="#b24640", linestyle="--", linewidth=1)
    right.set_yticks(range(len(comparisons)), [name for name, _ in comparisons])
    right.invert_yaxis()
    right.set_title("Paired week bootstrap: 95% intervals")
    right.set_xlabel("Log-loss improvement (positive = better)")
    right.set_ylim(2.5, -0.5)
    for axis in [left, right]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    fig.savefig(directory / "benchmark.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / "benchmark.svg", bbox_inches="tight")
    plt.close(fig)


def write_analysis_report(report: dict, output_dir: str | Path = "reports") -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _figure(report, directory)
    deployed_key = "with_formation" if report["deployment"]["uses_formation"] else "without_formation"
    selected = report["test"][deployed_key]
    data, audit = report["data"], report["legacy_audit"]
    signal = report["comparisons"]["formation_vs_ablation"]["log_loss"]
    verdict = ("포메이션이 결과 예측에 추가 정보를 제공한다는 사전 기준을 충족했습니다. 그래도 최적 전술의 인과 효과는 확인하지 못했습니다."
               if report["formation_signal_supported"] else
               "포메이션의 추가 예측 가치를 입증한다는 사전 기준을 충족하지 못했습니다. 팀별 최적 대형 추천을 보류합니다.")
    rows = [[NAMES[name], values["matches"], _percent(values["accuracy"]), f'{values["log_loss"]:.4f}',
             f'{values["multiclass_brier"]:.4f}'] for name, values in report["test"].items()]
    season_rows = []
    for season, scores in report["test_by_season"].items():
        season_rows.append([season, scores[deployed_key]["matches"], _percent(scores["home_prior"]["accuracy"]),
                            _percent(scores["legacy_rf_retrained"]["accuracy"]),
                            _percent(scores[deployed_key]["accuracy"]), f'{scores[deployed_key]["log_loss"]:.4f}'])
    effect_rows = []
    for label, key in [("포메이션 포함 − 제외", "formation_vs_ablation"), ("대체 모델 − 기존 RF", "upgraded_vs_legacy"),
                       ("대체 모델 − 홈 기준", "upgraded_vs_home_prior"), ("대체 모델 − Elo", "upgraded_vs_elo")]:
        interval = report["comparisons"][key]
        accuracy, loss = interval["accuracy"], interval["log_loss"]
        effect_rows.append([label, f'{100*accuracy["improvement"]:+.2f}%p',
                            f'[{100*accuracy["ci_95_low"]:+.2f}, {100*accuracy["ci_95_high"]:+.2f}]%p',
                            f'{loss["improvement"]:+.4f}', f'[{loss["ci_95_low"]:+.4f}, {loss["ci_95_high"]:+.4f}]'])
    support_rows = [[row["team"], row["appearances_last_730_days"], row["formations_with_at_least_10_appearances"],
                     row["most_used_formation"], _percent(row["most_used_share"]), "최적 대형 판정 보류"]
                    for row in report["support_audit"]["teams"]]
    pd.DataFrame(report["support_audit"]["teams"]).drop(columns="formation_counts").to_csv(
        directory / "tactical_feasibility.csv", index=False, encoding="utf-8-sig")
    model_table = _md_table(["모델", "경기", "정확도", "로그손실 ↓", "Brier ↓"], rows)
    effect_table = _md_table(["비교", "정확도 차이", "95% 구간", "로그손실 개선", "95% 구간"], effect_rows)
    season_table = _md_table(["시즌", "경기", "홈 기준", "기존 RF 재학습", "검증 선택 모델", "로그손실"], season_rows)
    market = report["market_reference"]
    quality = report.get("data_quality", {})
    independent = quality.get("crosscheck", {})
    verified_text = (f'날짜·스코어를 Football-Data와 대조해 {independent["verified_matches"]:,}/{independent["espn_matches"]:,} 일치했습니다.'
                     if independent else '독립 공급원 대조 기록이 제공되지 않았습니다.')
    collection = quality.get("collection", {})
    seasons_text = ', '.join(f'{season}: {count}경기' for season, count in collection.get("matches_by_season", {}).items())
    market_text = "배당 기준을 확보하지 못했습니다."
    if market["available"]:
        market_text = (f'동일한 {market["matches"]}경기에서 경기 전 배당을 마진 정규화한 확률은 정확도 '
                       f'{_percent(market["market"]["accuracy"])}, 로그손실 {market["market"]["log_loss"]:.4f}입니다. '
                       f'같은 표본에서 포메이션 포함 모델은 {_percent(market["with_formation_same_subset"]["accuracy"])} / '
                       f'{market["with_formation_same_subset"]["log_loss"]:.4f}입니다. '
                       '배당에는 선수·부상·시장 정보가 추가로 반영됩니다. 동일 입력의 공정 비교나 베팅 수익 검증은 아닙니다.')
    old_accuracy = audit["paired_week_intervals"]["accuracy_gain"]
    old_text = (f'기존 52.5%는 승·무·패 균등 랜덤의 기대값 33.3%보다는 높습니다. 하지만 같은 60경기에서 '
                f'홈승만 예측해도 {_percent(audit["always_home_accuracy"])}였습니다. '
                f'포메이션 제외 모델 대비 정확도 개선 {100*old_accuracy["value"]:.2f}%p의 주 단위 bootstrap 95% 구간은 '
                f'[{100*old_accuracy["ci_95_low"]:.2f}, {100*old_accuracy["ci_95_high"]:.2f}]%p로 0을 포함합니다. '
                '120개 팀 행을 독립 경기 120개로 볼 수도 없습니다. 그 결과만으로 추천 모델을 완성했다고 말한 것은 과했습니다.')
    causal_text = ('같은 경기에서 다른 포메이션을 썼을 때의 결과는 관측되지 않습니다. 감독은 상대·선수 구성·부상·전략에 따라 대형을 고릅니다. '
                   '따라서 대형별 승률 차이나 모델 입력의 대형만 바꾼 점수 차이에는 선택 편향이 포함됩니다. '
                   '모델의 높은 결과 예측 정확도, 포메이션의 유의한 추가 예측력, 최적 포메이션의 인과적 식별은 서로 다른 주장입니다. '
                   '이번 실험은 첫 두 주장만 평가하고, 세 번째는 검증하지 못했습니다. 축구 분석 전체가 불가능하다는 뜻은 아닙니다.')
    method_text = ('2019/20~2021/22에서 시작해 2022/23·2023/24를 순차 검증했습니다. 후보 9개와 고정된 소수의 앙상블을 '
                   '검증 로그손실로 선택한 뒤 2024/25 이후 테스트를 시작했습니다. 테스트 시즌 시작마다 과거 자료로만 재학습하며 '
                   '하이퍼파라미터·앙상블 가중치는 고정했습니다. 동일 경기의 홈·원정 확률을 대칭 평균해 경기당 하나의 일관된 분포로 평가합니다. '
                   '입력은 이전 날짜까지의 5·20경기 평균 승점·득실점·슈팅·유효슈팅·점유율, Elo, 휴식일, 홈/원정, 팀 식별자이며 '
                   '포메이션 포함/제외 실험을 따로 수행했습니다. 해당 경기의 경기 후 통계·배당·최신 팀 기록은 입력에서 제외했습니다. '
                   '정확도는 보조 지표이고, 주 지표는 확률 예측의 로그손실입니다.')
    rule_text = (f'사전 기준: 포메이션 추가로 로그손실이 0.01 이상 개선되고, 주 단위 paired bootstrap 95% 구간 하한이 0보다 커야 합니다. '
                 f'실측 개선은 {signal["improvement"]:+.4f}, 구간은 [{signal["ci_95_low"]:+.4f}, {signal["ci_95_high"]:+.4f}]입니다. '
                 '이 기준을 통과해도 인과적 전술 추천을 승인하지 않습니다. 구간은 모델 선택 불확실성이나 장기 시즌 변화를 모두 반영하지 못합니다.')
    next_text = ('최적 대형 분석을 진행하려면 경기 직전 시각이 확인되는 라인업·선수 역할/능력·부상/결장·감독·상대 전술·일정과 '
                 '경기 중 포메이션 및 경기 상태를 확보해야 합니다. 팀별로 비교 가능한 상대·선수 구성에서 여러 대형의 충분한 사용 사례가 필요합니다. '
                 '이후 선택 확률/공통 지지 영역과 교란변수 가정을 검토한 인과 추정, 선택 편향을 통제한 정책 평가, 별도 미래 시즌의 사전 등록 검증이 필요합니다. '
                 '관측하지 않은 교란을 제거했다고 보장할 수 없으므로 관측 연구와 통제 실험의 결론은 구분해야 합니다. 단순히 딥러닝으로 교체해도 이 문제는 해결되지 않습니다.')
    markdown = f'''# EPL 모델 개선 및 분석 가능성 검토

**결론: 결과 예측은 재평가했으며, 현재 자료만으로 팀별 최적 포메이션을 확인할 수 없어 전술 추천을 보류합니다.**

{verdict}

## 52%는 랜덤인가

{old_text}

## 수집 및 독립 대조

ESPN 공개 웹 데이터에서 {data['matches']:,}경기를 수집했습니다. 종료일은 {data['trained_through']}입니다.
{seasons_text}
양 팀 선발 포메이션과 경기 통계를 확보했습니다. {verified_text}
공개 접근이 가능하다는 것과 재배포가 자유로운 오픈 라이선스는 다릅니다. ESPN의 비문서화 API이므로 지속 제공을 보장하지 않습니다.
정규화: ESPN의 `3-4-3-d`는 3-4-3 다이아몬드 하위형으로 합쳤고, `4-2-4-0`은 그대로 보존했습니다. 이외 포메이션을 추정해 채우지 않았습니다.

## 평가 설계

{method_text}

검증에서 선택된 배포 모델은 **{'포메이션 포함' if report['deployment']['uses_formation'] else '포메이션 제외'} 모델**입니다. 테스트 점수를 보고 고르지 않았습니다.
기존 52.5%와 아래 정확도는 다른 표본이므로 직접 개선폭으로 비교할 수 없습니다. 기존 RF도 같은 새 자료·테스트 일정에서 재학습해 비교했습니다.

## 새로운 {data['test_matches']}경기 결과

{model_table}

{season_table}

{market_text}

## 개선폭의 불확실성

{effect_table}

정확도 차이의 단위는 %p이고, 로그손실 개선은 기준 모델 손실 − 포메이션 포함 대체 모델 손실입니다. 양수면 개선입니다.
동일 경기끼리 짝을 맞추고 달력 주 단위로 재표집했습니다. 팀 행을 독립 표본으로 세지 않았습니다.

{rule_text}

## 최적 대형은 왜 결론을 낼 수 없는가

{causal_text}

최근 730일에 같은 대형을 10회 이상 사용한 것을 최소 비교 기준으로 보면, 현재 20개 팀 중 {report['support_audit']['teams_with_two_supported_formations']}개만 2개 이상의 대형이 있습니다.
이 사용 수는 상대·선수·감독 조건의 공통 지지 영역을 보장하지 않습니다. 최빈 대형은 추천 대형이 아닙니다.

{_md_table(['팀','최근 730일 경기','10회 이상 사용 대형 수','최빈 대형 (추천 아님)','최빈 비율','판정'], support_rows)}

## 다음 단계와 분석 범위

{next_text}

가능: 경기 결과 확률 예측, 실제 사용 대형 통계, 검증된 범위의 예측 정보 비교.
현재 검증 불가: 특정 팀이 지금 가장 잘할 대형, 대형 변경 시 승점 향상, 실제 추천 정책의 성능.

## 재현 및 자료

프로토콜 `artifacts/v2/protocol.json`, 모델 선택 `artifacts/v2/selection.json`, 경기별 테스트 예측 `artifacts/v2/test_predictions.csv`, 전체 지표 `artifacts/v2/metrics.json`.
학습 모델은 보고한 테스트 예측과 별도로 전체 관측 데이터에 재적합했습니다. 현재 시즌 결과는 80경기뿐입니다.

- [ESPN 선발 라인업 예시](https://www.espn.com/soccer/lineups/_/gameId/704467)
- [ESPN 공개 경기 데이터](https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event=704467)
- [Football-Data 결과·배당](https://football-data.co.uk/englandm.php)
- [scikit-learn 시간순 평가](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [scikit-learn 확률 예측 평가](https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss)
'''
    (directory / "analysis.md").write_text(markdown, encoding="utf-8")
    (directory / "benchmark_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    png = base64.b64encode((directory / "benchmark.png").read_bytes()).decode("ascii")
    sections = [
        ("52%를 다시 해석하면", f'<p>{html.escape(old_text)}</p>'),
        ("새 데이터와 평가 방법", f'<p>{html.escape(method_text)}</p>'),
        (f'{data["test_matches"]}경기의 동일 조건 비교', _html_table(["모델", "경기", "정확도", "로그손실 ↓", "Brier ↓"], rows)),
        ("시즌별 결과", _html_table(["시즌", "경기", "홈 기준", "기존 RF 재학습", "검증 선택 모델", "로그손실"], season_rows)),
        ("강한 외부 기준과 비교", f'<p>{html.escape(market_text)}</p>'),
        ("성능 차이와 불확실성", _html_table(["비교", "정확도 차이", "95% 구간", "로그손실 개선", "95% 구간"], effect_rows) + f'<p>{html.escape(rule_text)}</p>'),
        ("최적 대형을 결론 낼 수 없는 이유", f'<p>{html.escape(causal_text)}</p>'),
        ("팀별 대형 사용 근거", '<p>최근 730일의 실제 사용 기록입니다. 최빈 대형은 추천이 아닙니다. 횟수만으로 상대·선수·감독 조건이 비슷해지지는 않습니다.</p>' + _html_table(["팀", "경기", "10회 이상 대형", "최빈 대형 (추천 아님)", "비율", "판정"], support_rows)),
        ("최적 전술 검증에 필요한 것", f'<p>{html.escape(next_text)}</p>'),
    ]
    content = ''.join(f'<section><h2>{html.escape(title)}</h2>{body}</section>' for title, body in sections)
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPL 모델 개선 검토</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#0c1520;color:#e9f0f6;font:15px/1.85 system-ui,'Malgun Gothic',sans-serif}}main{{max-width:1180px;margin:auto;padding:38px 24px}}h1{{font-size:34px;line-height:1.4;letter-spacing:-1px}}h2{{font-size:21px;margin:0 0 14px}}p{{color:#b6c6d5}}.kicker{{font-size:12px;letter-spacing:3px;color:#73d9be}}.verdict{{padding:22px;border:1px solid #806431;border-radius:14px;background:#2e281c}}.verdict strong{{font-size:22px;color:#f1d39b}}section{{padding:24px;border:1px solid #2b4051;border-radius:14px;background:#132334;margin:20px 0}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}}.stats div{{padding:18px;background:#132334;border-radius:12px}}.stats strong{{font-size:28px;display:block;color:#7de1c1}}.stats span{{font-size:13px;color:#b6c6d5}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:12px 9px;border-bottom:1px solid #304455;text-align:left}}th{{color:#aabed0;font-weight:500}}.scroll{{overflow-x:auto}}img{{width:100%;display:block;border-radius:12px;background:white;margin:22px 0}}a{{color:#7de1c1}}footer{{font-size:12px;color:#a8bacb}}@media(max-width:700px){{main{{padding:24px 14px}}section{{padding:16px}}h1{{font-size:27px}}.stats{{gap:7px}}.stats div{{padding:12px}}.stats strong{{font-size:23px}}}}
</style></head><body><main><div class="kicker">EPL · EXPANDED DATA · FORWARD VALIDATION</div>
<h1>결과 예측은 검증하고,<br>최적 전술 주장은 보류합니다.</h1>
<p>{html.escape(', '.join(data['seasons']))} · {data['matches']:,}개 실제 경기 · 최신 경기 {data['trained_through']}</p><p>{html.escape(verified_text)}</p>
<div class="verdict"><strong>최적 포메이션: 현재 자료로 판정 불가</strong><p>{html.escape(verdict)}</p></div>
<div class="stats"><div><strong>{data['matches']:,}</strong><span>포메이션 포함 실제 경기</span></div>
<div><strong>{data['test_matches']}</strong><span>순차 테스트 경기</span></div><div><strong>{_percent(selected['accuracy'])}</strong><span>검증에서 선택한 모델의 결과 예측 정확도</span></div></div>
<p>위 정확도는 전술 추천 성공률이 아닙니다. 기존 52.5%와 표본이 달라 직접 개선폭으로 비교하지 않습니다.</p>
<img src="data:image/png;base64,{png}" alt="같은 테스트 경기의 모델별 정확도와 개선폭 신뢰구간">
{content}<section><h2>결과 파일</h2><p><a href="analysis.md">상세 분석 문서</a> · <a href="benchmark_metrics.json">전체 지표 JSON</a> · <a href="benchmark.svg">그래프 SVG</a></p>
<p><a href="tactical_feasibility.csv">팀별 분석 가능성 CSV</a> · 실행 방법은 프로젝트 README를 확인하세요. 전술 추천 API는 검증되지 않은 최적 대형을 반환하지 않습니다.</p></section>
<footer>Sources: <a href="https://www.espn.com/soccer/lineups/_/gameId/704467">ESPN match lineups</a> · <a href="https://football-data.co.uk/englandm.php">Football-Data</a>.
ESPN 데이터는 공개 웹 인터페이스에서 로컬 연구용으로 수집했으며 오픈 라이선스를 주장하지 않습니다. 기존 StatsBomb 2015/16 실험은 별도 감사에만 사용했습니다.</footer>
</main></body></html>'''
    result = directory / "index.html"
    result.write_text(page, encoding="utf-8")
    return result
