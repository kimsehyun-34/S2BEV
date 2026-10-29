# EPL 결과 예측과 전술 분석 가능성 검토 (v2)

**현재 자료로 팀별 ‘최적 포메이션’을 검증할 수 없어 전술 추천을 보류합니다.** 데이터를 380경기에서 최근 2,740경기로 늘리고 네 가지 모델 계열을 비교했지만, 포메이션의 추가 예측 가치를 확인하지 못했습니다. 경기 결과 확률을 예측하는 모델과 검증 결과는 제공합니다.

- [팀별 전술대형 탐색 화면](reports/index.html): 팀·포메이션 선택, 배치도, 과거 모델의 기대 승점과 확률. 첨부 화면을 복원했으며 첫 화면은 Arsenal입니다. 최적 대형의 검증 결과를 뜻하지 않습니다.
- [최신 성능 검토 보고서](reports/analysis.html): 브라우저에서 파일을 열면 됩니다. 서버·인터넷 불필요.
- [상세 분석 문서](reports/analysis.md): 기존 52.5%의 해석, 새 평가, 불확실성, 현재 분석할 수 없는 것.
- [전체 지표](artifacts/v2/metrics.json), [평가 전 고정한 프로토콜](artifacts/v2/protocol.json), [모델 선택 기록](artifacts/v2/selection.json).
- [경기당 하나의 테스트 예측](artifacts/v2/test_predictions.csv), [팀별 분석 가능성 CSV](reports/tactical_feasibility.csv).
- [학습 모델](artifacts/v2/model.joblib), [그래프 SVG](reports/benchmark.svg).
- [유명 6개 구단 집중 분석](reports/focused/index.html): Arsenal·Chelsea·Liverpool·Manchester City·Manchester United·Tottenham의 2006/07~2026/27 EPL 경기 4,003개, Random Forest 하이퍼파라미터 튜닝 결과.

## 실제 결과

2019/20~2025/26 각 380경기와 2026/27의 80경기, 총 **2,740경기**를 수집했습니다. 최신 경기는 2026-10-25입니다. ESPN의 선발 포메이션·선발 11명·경기 통계를 확보했고, 모든 경기의 날짜·득점을 Football-Data와 대조해 2,740/2,740 일치했습니다. 포메이션 자체를 독립 공급원과 대조한 것은 아닙니다.

모델 선택은 2022/23·2023/24의 시간순 검증으로 끝내고, 이후 2024/25·2025/26·2026/27의 **840경기**에서 평가했습니다. 테스트 시즌 시작 시 이전 시즌까지 재학습하되 모델 종류·하이퍼파라미터·앙상블 가중치는 바꾸지 않았습니다.

| 같은 840경기의 모델 | 승·무·패 정확도 | 로그손실 ↓ |
|---|---:|---:|
| 균등 랜덤 (기대값) | 33.33% | 1.0986 |
| 홈 결과 빈도 / 홈승 기준 | 41.43% | 1.0840 |
| Elo + 홈 여부 | 50.60% | 1.0100 |
| 기존 Random Forest, 추가 자료로 재학습 | 51.31% | 1.0286 |
| 대체 모델, 포메이션 제외 | 50.71% | 1.0105 |
| 대체 모델, 포메이션 포함 | 50.48% | 1.0103 |
| 경기 전 배당 확률 (추가 정보가 있는 외부 기준) | 52.02% | 0.9918 |

**정확도가 개선되었다고 주장할 수 없습니다.** 대체 모델의 확률 예측 손실은 기존 RF보다 나아졌지만 Elo를 확실하게 넘지 못했습니다. 배당에는 부상·선수·시장 정보가 추가로 반영되어 동일 입력의 비교가 아닙니다.

포메이션 포함/제외의 로그손실 차이는 **+0.000187**, 주 단위 paired bootstrap 95% 구간은 **[-0.000336, +0.000744]**로 0을 포함합니다. 사전에 정한 추가 신호 기준인 ‘0.01 이상 개선 + 구간 하한 > 0’을 충족하지 못했습니다. 이 기준을 충족하더라도 인과적 최적 전술을 증명하는 것은 아닙니다.

기존 52.5%는 균등 랜덤 33.3%보다는 높지만, 기존 60경기에서 홈승만 예측해도 48.3%였습니다. 120개 팀 행은 독립된 120경기가 아닙니다. 기존 52.5%와 새 50.48%는 다른 표본이어서 직접 개선·악화 폭으로 비교하면 안 됩니다. 그래서 기존 RF도 같은 새 자료·같은 새 테스트에서 재학습해 비교했습니다.

## 유명 6개 구단 집중 분석 (v3)

Arsenal 등 6개 구단이 치른 모든 EPL 경기(상대팀 제한 없음)를 2006/07부터 수집했습니다. 고유 경기 4,003개, 구단 관점 학습 행 4,608개이며 두 수는 다릅니다. ESPN 상세 기록이 확인된 경기는 3,739개, 나머지 264개는 결과·배당만 있고 없는 포메이션은 채우지 않았습니다.

- RF 설정 32개(트리 수·깊이·말단 표본·특성 비율·표본 비율·분할 기준·클래스 가중 등 8개 하이퍼파라미터)와 로지스틱·부스팅 8개를 시간순 3개 구간(2018/19, 2021/22, 2023/24)에서 비교했고, 포메이션 포함/제외를 각각 튜닝했습니다. 선택은 검증 로그손실만 사용했고 2024/25 이후 평가 경기는 쓰지 않았습니다(이전 실험에서 이미 본 구간이므로 새 미래 검증은 아닙니다).
- 평가 439경기 / 504개 구단 관점. 튜닝 RF 정확도 53.6%, 튜닝 전 RF 53.2%, Elo 기준 53.8%, 항상 해당 구단 승리 49.9%. RF 튜닝의 로그손실 개선 +0.0090(95% 구간 [+0.0004, +0.0174]).
- 검증에서 가장 좋은 모델은 RF가 아니라 강한 규제의 로지스틱(C=0.003)이었고, 단순 Elo 기준보다 뚜렷이 낫지 않았습니다. 포메이션 변수 추가의 개선도 0과 구분되지 않았습니다(같은 설정 비교 +0.0003). 최적 포메이션 추천은 계속 보류합니다.

```powershell
python -m epl_tactics collect-focused            # data/focused (중단 후 재실행하면 이어서 수집)
python -m epl_tactics analyze-focused            # artifacts/focused 와 reports/focused 생성
python -m epl_tactics report-focused             # 재학습 없이 리포트만 재생성
```

수집한 원본 ESPN 응답(`data/focused/facts/`)은 재배포 권리가 불분명해 Git에 올리지 않습니다. `collect-focused`로 다시 받을 수 있습니다.

## 실행 환경

Python 3.10 이상. 프로젝트 루트에서 실행합니다. 제공하는 모델은 scikit-learn **1.7.2**로 학습했으며, 다른 버전에서는 재학습해야 합니다.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m epl_tactics report
```

탐색 화면을 다시 생성하려면 `.venv/Scripts/python.exe -m epl_tactics explorer`를 실행하세요. `reports/index.html`과 같은 내용의 `reports/explorer.html`도 생성합니다. 최신 성능 보고서는 `reports/analysis.html`에 생성되므로, 보고서를 다시 만들어도 탐색 화면을 덮어쓰지 않습니다.

현재 작업 환경에는 기존 Python 3.10의 패키지를 참조하는 `.venv`를 만들어 검증했습니다. 새 환경에서는 위와 같이 의존성을 설치하세요. `joblib`은 신뢰하는 로컬 학습 결과만 로딩하세요.

## 데이터 재수집과 재현

```powershell
.venv/Scripts/python.exe -m epl_tactics collect-recent --start-year 2019 --cutoff 2026-10-28
.venv/Scripts/python.exe -m epl_tactics benchmark
.venv/Scripts/python.exe -m epl_tactics report
```

수집은 인터넷이 필요하며 `data/raw/espn/`에 사실 필드만 캐시합니다. 기사·사진·중계 문장과 API의 현재 팀 기록은 보존하거나 입력으로 사용하지 않습니다. 일부 실패가 있으면 중단하고 재실행으로 캐시에서 이어받습니다. 완료된 경기만 포함합니다.

ESPN의 공개 웹 데이터 인터페이스는 현재 인증 없이 접근할 수 있지만 **공식적으로 문서화된 오픈 데이터 API나 자유 재배포 라이선스를 주장하지 않습니다.** 이용 조건·지속 제공 가능성은 별도 문제입니다. Football-Data는 결과 검증과 경기 전 배당 기준에만 사용합니다.

- [경기 데이터](data/epl_recent.csv), [출처·체크섬](data/epl_recent.manifest.json).
- [독립 대조 기록](data/football_data/crosscheck.json), [배당 확률](data/football_data/market_probabilities.csv).
- ESPN의 `3-4-3-d`는 3-4-3 다이아몬드 하위형으로 합쳤으며, `4-2-4-0`은 그대로 보존합니다. 임의로 대형을 추정해 채우지 않았습니다.
- 기본 수집 기준일은 이번 실험을 재현하는 2026-10-28입니다. 이후 자료를 수집하려면 `--cutoff`를 명시하세요. 현재 시즌 인덱스·완료 경기 캐시는 해당 기준일별로 구분합니다.

이미 평가한 출력 디렉터리의 프로토콜을 바꿔 조용히 재사용하면 오류를 냅니다. 변경 실험에는 새 출력 경로를 사용하세요.

```powershell
.venv/Scripts/python.exe -m epl_tactics benchmark --data data/my_epl.csv --output-dir artifacts/experiment2 --report-dir reports/experiment2
```

이 명령은 고정한 2022/23·2023/24 검증 시즌과 2024/25 이후 테스트 시즌이 있는 CSV를 요구합니다. 공급원·팀 이름·경기 ID는 입력 전 일관되게 맞춰야 합니다. 임의의 한 시즌 CSV를 비교하려면 기존 `train` 명령이 있지만, 이는 별도의 v1 실험이며 최적 대형 추천은 승인하지 않습니다.

## 결과 예측 예시

다음은 사용자가 두 대형과 날짜를 지정하는 **가상 경기 조건**입니다. 실제 다음 경기 일정·선발 명단을 확인하는 명령은 아닙니다.

```powershell
.venv/Scripts/python.exe -m epl_tactics predict --team Arsenal --opponent "Manchester City" --venue home --formation 4-3-3 --opponent-formation 4-2-3-1 --as-of 2026-10-29 --output reports/forecast_example.json
```

승·무·패 확률과 기대 승점을 반환하지만 `recommended_formation`은 `null`, `tactical_status`는 `not_identified`입니다. `recommend` 명령도 검증되지 않은 최적 대형을 반환하지 않습니다. 과거 v1 아티팩트에 대해서도 추천을 보류하도록 바꿨습니다.

제공하는 최종 모델은 검증에서 가장 좋았던 정규화 Poisson 득점 모델(alpha=0.3)입니다. 포메이션 포함 여부도 검증 로그손실로 고정했고, 테스트 결과를 보고 바꾸지 않았습니다. 두 팀의 선발 포메이션을 입력받습니다. 모델의 현재 입력에 선수 능력·부상·역할 적합도는 포함되지 않습니다.

예측 기본 기준일은 학습 자료 종료 다음 날이며, 실제 예측에는 원하는 날짜를 `--as-of`로 명시하세요. 학습 종료 이전 날짜는 거부합니다. 최신 날짜만 넣는 것으로 선수단·최근 경기가 갱신되지는 않습니다.

## 평가·모델 구성

- 경기 전 입력: 직전 5·20경기 승점·득실점·슈팅·유효슈팅·점유율, 장기 Elo, 휴식일, 홈/원정, 팀·상대 식별자. 같은 경기의 경기 후 통계·배당·현재 기록은 제외합니다.
- 같은 날짜 모든 입력을 만든 뒤에 그날 결과로 상태를 갱신합니다. 슈팅·점유율은 이전 경기의 값만 사용합니다.
- Elo는 시즌 경계에서 평균으로 일부 회귀시키고, 이전 시즌의 경기력 이력은 유지합니다. 초기 이력은 공통 사전값 5경기 상당량으로 축소 추정합니다.
- 로지스틱 회귀 3개, HistGradientBoosting 2개, Poisson 2개, Random Forest 2개 설정을 비교합니다. 서로 다른 계열의 상위 3개와 소수의 고정 앙상블도 검증 자료에서 비교합니다.
- HGB의 임의 내부 검증 분할은 사용하지 않습니다. 최신 자료에 더 높은 가중치를 주는 730일 반감기를 고정해 사용합니다.
- 홈·원정 두 관점의 확률을 대칭 평균해 경기당 하나의 일관된 확률 분포로 평가합니다. 따라서 한 경기를 독립 표본 두 개로 세지 않습니다.
- 확률 예측의 로그손실이 주 선택 지표입니다. 정확도·Brier·기대 승점 MAE도 기록합니다. 테스트 평가 후 지표에 맞춘 모델 재탐색은 하지 않았습니다.
- 개선폭은 경기별로 짝을 맞춘 달력 주 단위 bootstrap 3,000회로 추정합니다. 같은 팀의 장기 의존·모델 선택의 모든 불확실성을 제거한 신뢰구간은 아닙니다.
- 최종 저장 모델은 전체 관측 자료로 재학습했습니다. 보고한 테스트 점수는 각 시점 이전 자료만 쓴 별도 예측입니다.

## 분석할 수 있는 것과 없는 것

가능: 경기 결과 확률 예측, 실제 대형 사용 통계, 관측 대형의 추가 예측 정보 검증.

**현재 검증 불가:** 해당 팀의 인과적 최적 대형, 대형을 바꾸면 얻을 승점, 실제 대형 추천 정책의 성능. 감독의 선택·부상·선수 구성·상대 반응이 통제되지 않았고, 같은 경기의 다른 대형 결과는 관측되지 않습니다. 예측 모델에 넣는 대형만 바꾸면 이 문제가 해결되지 않습니다.

추가로 필요한 것은 시각이 확인되는 경기 전 선수 가용성·역할·능력·감독·상대 전술과 경기 중 상태 변화, 비교 가능한 조건에서 여러 대형을 사용한 충분한 기록입니다. 선택 편향과 공통 지지 영역을 검토한 정책 평가 및 별도 미래 시즌 검증이 필요합니다. 딥러닝으로 바꾸는 것만으로 관측하지 않은 반사실을 확인할 수는 없습니다. 축구 분석 전체가 불가능하다는 결론은 아닙니다.

## 검사

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

25개 검사에서 입력 검증, 당일·미래 결과와 통계 누출 방지, 시즌 경계의 이력, 경기당 확률 일관성, 랜덤 기준값, paired bootstrap, 시간순 선택/테스트 분리, 재학습·저장·로딩, 전술 추천 보류를 확인했습니다. 합성 데이터는 기능 검사에만 사용했고 제공하는 모델·실제 지표에는 사용하지 않았습니다.

`scripts/verify_report.cjs`는 Playwright Core가 있는 환경의 선택적 브라우저 검사입니다. 보고서의 20개 팀 판정 보류, 그래프 로딩, 다운로드 파일, 모바일 가로 넘침을 검사합니다.

## 파일 구조와 기존 결과

```text
epl_tactics/recent_data.py    최근 경기 수집·공급원 대조·배당 기준
epl_tactics/features_v2.py    시간순 장기 경기력 입력
epl_tactics/estimators_v2.py  네 계열 모델·앙상블·경기당 확률
epl_tactics/benchmark.py      고정 프로토콜·시즌순 평가·최종 모델
epl_tactics/statistics.py     기존 실험 감사·주 단위 paired 구간
epl_tactics/analysis_report.py  보고서·표준 그래프·분석 문서
epl_tactics/cli.py            수집·평가·보고서·조건별 예측
artifacts/v2/                새로운 모델·프로토콜·선택·평가
reports/                     새로운 분석 보고서
reports/legacy/              과거 결과 보존; 현재 추천이 아님
```

2015/16 StatsBomb 원본 CSV·모델·테스트 예측은 감사를 위해 `data/epl_2015_16.csv`, `artifacts/`에 보존했습니다. 과거 추천 CSV·JSON·스크린샷은 `reports/legacy/original_outputs/`로 옮겼고 당시 README도 보관했습니다. 이전 추천은 철회한 탐색 결과입니다. StatsBomb 자료를 공유할 때는 원본 출처 및 로고 표시 조건을 따르세요.

자료: [ESPN 경기 선발 라인업](https://www.espn.com/soccer/lineups/_/gameId/704467), [Football-Data](https://football-data.co.uk/englandm.php), [StatsBomb Open Data](https://github.com/statsbomb/open-data), [scikit-learn 시간순 평가](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
