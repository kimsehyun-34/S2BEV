# EPL 팀별 전술대형 추천 모델

**실제 EPL 2015/16 시즌 380경기·20개 팀으로 학습한 머신러닝 연구 모델**입니다. 팀·상대·홈/원정·최근 경기력을 입력해 후보 대형의 승·무·패 확률과 기대 승점을 비교합니다. Python 소스, 실제 데이터, 학습된 모델, 시간순 평가, 모든 팀의 HTML/CSV 결과를 제공합니다.

**현재 선수단의 최적 대형을 증명하는 모델은 아닙니다.** 이긴 경기의 포메이션을 정답으로 삼으면 팀의 강함·상대·감독의 선택 편향까지 학습합니다. 여기서는 경기 결과를 예측하고, 같은 경기 전 조건에서 포메이션을 바꿔 점수를 비교합니다. 이 역시 관측 데이터에 따른 후보 순위이며 인과적 최적화는 아닙니다.

## 바로 보기

- [`reports/index.html`](reports/index.html): 더블클릭으로 열 수 있는 한국어 대화형 보고서. 팀/대형 선택, 대형 배치도, 확률, 사용 사례 수, 평가 결과. 서버나 인터넷 불필요.
- [`reports/team_recommendations.csv`](reports/team_recommendations.csv): 모든 팀의 지원 후보 중 1위.
- [`reports/formation_rankings.csv`](reports/formation_rankings.csv): 모든 팀·대형의 점수와 추천 가능 여부.
- [`artifacts/metrics.json`](artifacts/metrics.json): 날짜별 분할, 모델 선택, 테스트 지표, 환경 버전.
- [`artifacts/model.joblib`](artifacts/model.joblib): 실제 데이터로 학습한 모델과 경기력 상태.

## 실행

Python 3.10 이상. 다음 명령은 프로젝트 루트에서 실행합니다. 현재 생성한 모델은 scikit-learn **1.7.2**로 저장했습니다. 다른 버전을 설치했다면 다시 학습하세요.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m epl_tactics train --data data/epl_2015_16.csv
.venv\Scripts\python.exe -m epl_tactics report
```

이미 필요한 패키지가 설치되어 있으면 `python -m epl_tactics ...`로 실행할 수 있습니다. 패키지 설치 방식은 `pip install -e .`이고, 설치 후 `epl-tactics` 명령도 사용할 수 있습니다.

특정 상대를 만날 때의 추천:

```powershell
python -m epl_tactics recommend --team "Arsenal" --opponent "Manchester City" --venue home --output reports/arsenal_vs_city.json
python -m epl_tactics recommend --team "Liverpool" --opponent "Chelsea" --venue away
```

기본 예측 기준일은 학습 데이터 종료 7일 후인 **2016-05-24**입니다. 현재 날짜가 아닙니다. `--as-of YYYY-MM-DD`로 바꿀 수 있지만, 오래된 데이터에 최신 날짜만 넣는 것으로 최신 모델이 되지는 않습니다. 학습 종료일 이전의 예측은 미래 데이터 사용을 막기 위해 거부합니다.

## 데이터

[StatsBomb Open Data](https://github.com/statsbomb/open-data)의 EPL competition 2 / season 27. 경기 결과와 두 팀의 **Starting XI 이벤트에 기록된 선발 포메이션**을 사용합니다. 2015-08-08부터 2016-05-17까지 380경기이며 각 팀은 38경기입니다. 실제 csv와 SHA-256 출처 기록을 포함했습니다.

```powershell
python -m epl_tactics fetch --season-id 27 --output data/epl_2015_16.csv
```

인터넷이 필요합니다. 전체 이벤트 파일 대신 선발 포메이션이 들어 있는 첫 이벤트 부분만 읽습니다. 진행 중 받은 선발 이벤트는 `data/raw/`에 캐시하고, 실패가 있으면 부분 데이터로 학습하지 않고 중단합니다. 다시 실행하면 캐시에서 이어받습니다. 재현 가능한 다운로드에는 `--ref <commit-SHA>`를 지정하세요. 기본 `master`의 캐시는 자동으로 갱신하지 않습니다.

출처: **Data provided by StatsBomb (Hudl)**. 연구·분석에 사용하는 데이터입니다. [원본 이용 조건](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf)과 [출처 및 로고 표시 안내](https://github.com/statsbomb/open-data#terms--conditions)를 따르세요. 보고서 공개·배포 시 StatsBomb 출처와 제공자의 로고 표시 조건도 적용됩니다.

## 최신 시즌으로 학습하려면

2026-10-28 확인 기준 [StatsBomb 공개 목록](https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json)의 EPL 시즌은 2003/04와 2015/16입니다. **이 저장소에서 최신 EPL 학습 자료가 2015/16이라는 의미이며, 최신 EPL 데이터 자체가 없다는 뜻은 아닙니다.**

[Football-Data](https://football-data.co.uk/englandm.php)는 2026/27 경기 결과·통계를 무료로 제공하지만, 그 CSV만으로는 선발 포메이션이 없습니다. [Sportmonks](https://www.sportmonks.com/football-api/premier-league-api/) 등은 선발 라인업·포메이션을 제공하며 이용 권한과 시즌별 커버리지를 확인해야 합니다. 이 프로젝트에는 유료 서비스 가입이나 API 키가 필요하지 않으며, 해당 API 어댑터는 포함하지 않았습니다.

포메이션이 있는 최신 경기 데이터를 [`data/matches_template.csv`](data/matches_template.csv)의 열 구조에 맞추면 같은 코드로 재학습합니다.

| 열 | 내용 |
|---|---|
| `match_id` | 전체 파일에서 유일한 경기 ID |
| `date` | 경기 날짜 `YYYY-MM-DD`; 시간대 없는 현지 경기 날짜 |
| `season` | `2025/2026`, `2026/2027` 등 시즌 식별자 |
| `home_team`, `away_team` | 일관된 팀 이름 |
| `home_formation`, `away_formation` | 선발 대형. `4-3-3` 또는 `433`; 골키퍼를 제외하고 합계 10 |
| `home_goals`, `away_goals` | 종료된 경기의 실제 득점. 0 이상의 정수 |

```powershell
python -m epl_tactics train --data data/my_epl_matches.csv --output-dir artifacts/latest
python -m epl_tactics report --model artifacts/latest/model.joblib --output-dir reports/latest
```

최소 40경기·20개 경기 날짜와 학습 구간의 승·무·패가 필요합니다. 진행 중 시즌은 지난 시즌까지 포함한 다년 데이터를 권장합니다. 각 시즌에서 팀 상태를 초기화하고, 보고서는 파일 내 가장 최근 시즌의 팀을 대상으로 만듭니다. 데이터가 부족한 승격팀에 대해서는 충분한 근거를 확보하기 전까지 추천이 보류될 수 있습니다. 팀 이름 별칭 병합과 공급자 간 경기/포메이션 매칭은 입력 전 수행해야 합니다.

## 학습 방법

1. 경기마다 홈/원정 관점의 두 행을 만듭니다. 결과 클래스는 패=0, 무=1, 승=2입니다.
2. 입력은 팀·상대·후보 포메이션·홈 여부와 **직전** 5경기 평균 승점·득점·실점, Elo, 휴식일, 누적 경기 수입니다. 팀과 상대 모두 계산합니다. 휴식일은 최대 30일, 최초 경기 상태는 공통 사전값, Elo 초기값 1500/홈 이점 65/K=20입니다.
3. 같은 날짜 모든 경기의 입력을 만든 뒤에만 결과로 상태를 갱신합니다. 해당 경기나 같은 날짜 경기의 결과가 입력에 섞이지 않습니다.
4. 날짜를 순서대로 60%/20%/20% 나눠 학습·검증·테스트합니다. 경기 수 비율은 날짜별 경기 수에 따라 달라집니다. 한 경기의 두 행은 같은 분할에 있습니다. 검증/테스트에서도 이전 날짜 결과를 관측한 뒤 다음 경기를 예측하는 방식입니다.
5. One-hot encoding + Random Forest 350개 트리. 두 가지 복잡도 후보를 검증 로그손실로 고릅니다. 포메이션 제외 모델도 동일한 검증 예산으로 별도 선택합니다. 테스트를 보고 하이퍼파라미터를 수정하지 않습니다.
6. 학습+검증으로 적합한 모델을 테스트에서 평가한 뒤, 배포용 모델은 전체 관측 데이터로 다시 학습합니다. `test_predictions.csv`는 전체 재학습 모델의 점수가 아닌 별도 테스트 예측입니다.

작은 표 형식 데이터에서 먼저 검증할 수 있는 머신러닝 기준 모델을 선택했습니다. 딥러닝 모델은 포함하지 않았습니다. 선수 능력·부상·역할 적합도·선수 추적·현재 감독·상대의 미래 포메이션·경기 중 전술 변경은 입력에 없습니다.

## 실제 평가 결과

모델 선택 전 고정한 마지막 **60경기 / 120개 팀 행**을 평가했습니다. 두 팀 행은 상관되어 있으므로 독립 표본 120개로 해석하면 안 됩니다.

| 모델 | 승·무·패 정확도 ↑ | 로그손실 ↓ | 다중 클래스 Brier ↓ |
|---|---:|---:|---:|
| 포메이션 포함 Random Forest | 52.50% | 1.0201 | 0.6084 |
| 포메이션 제외 Random Forest | 50.83% | 1.0244 | 0.6108 |
| 학습 구간 결과 비율 | 34.17% | 1.1022 | 0.6689 |

학습 230경기(2015-08-08~2016-01-24), 검증 90경기(2016-02-02~2016-04-09), 테스트 60경기(2016-04-10~2016-05-17). 포메이션 추가에 따른 차이는 작습니다. 통계적 유의성·인과적 전술 효과·추천 정책의 실제 수익/승점은 검증하지 않았습니다. 확률은 별도로 보정하지 않은 예측치입니다.

## 추천 점수와 보류

`기대 승점 = 3 × P(승리) + P(무승부)`이며 범위는 0~3입니다. 모든 팀 보고서는 각 팀이 같은 시즌의 다른 19개 팀과 홈·원정에서 만나는 **38개 가상 조건의 단순 평균**으로 비교합니다. 특정 상대 추천은 `recommend` 명령을 사용합니다.

기본 추천 후보는 해당 팀·최신 시즌 사용 3경기 이상, 전체 학습 자료 사용 10개 팀 행 이상입니다. 기준은 `--min-team-support`, `--min-league-support`로 조정할 수 있습니다. 적은 표본을 걸러내는 기준일 뿐 신뢰구간은 아닙니다.

| 상태 | 해석 |
|---|---|
| `candidate_recommendation` | 지원 후보 여러 개 중 기대 승점이 가장 높은 후보 |
| `near_tie` | 상위 지원 후보 두 개의 기대 승점 차이 0.03 미만. 유의성 기준 아님 |
| `single_supported_formation` | 기준을 충족한 대형이 하나뿐. 다른 대형과 최적 비교 불가 |
| `insufficient_support` | 기준을 충족한 대형이 없어 추천 보류 |

미관측/소수 관측 대형도 참고 점수는 표시하지만 추천하지 않습니다. **최적 전술 결론을 강화하려면** 최근 여러 시즌의 포메이션·선수 가용성·역할 능력·감독 데이터, 더 강한 시간순 검증, 확률 보정, 혼란변수 처리와 추천 정책 평가가 필요합니다.

## 검증 실행과 구조

```powershell
python -m unittest discover -s tests -v
```

포메이션·점수 검증, UTF-8 스트림 수집, 미래·당일 결과 누출 방지, 시즌 초기화, 경기별 분할 보존, 실제 학습/저장/로딩, 확률·기대 승점, 추천 보류, 보고서 생성·출력 이스케이프를 검사합니다. 테스트용 합성 데이터는 기능 검증에만 사용하며, 위 성능이나 제공하는 모델에는 사용하지 않았습니다.

14개 Python 검사를 통과했습니다. `scripts/verify_report.cjs`는 Playwright Core가 설치된 환경에서 선택적으로 실행하는 브라우저 검사입니다. 실제 Chromium에서 20개 팀 선택, 대형 변경, 11명 배치, 오류 없는 실행, 모바일 가로 넘침 없음을 확인했으며 미리보기를 `reports/preview.png`, `reports/preview-mobile.png`에 저장했습니다.

```text
epl_tactics/data.py       공개 데이터 수집·정규화·검사
epl_tactics/features.py   경기 전 상태·시간순 입력
epl_tactics/model.py      학습·모델 선택·평가·저장
epl_tactics/recommend.py  상대별/전체 팀 대형 순위·표본 기준
epl_tactics/report.py     독립 실행 HTML·CSV·JSON 보고서
epl_tactics/cli.py        명령행 인터페이스
data/                    실제 경기 CSV·출처 기록·입력 템플릿
artifacts/               모델·평가·테스트 예측·대형 사용 수
reports/                 생성된 팀별 결과
tests/                   기능·누출 방지·통합 검사
```

`joblib` 파일은 신뢰하는 출처에서 직접 생성한 파일만 로딩하세요. 패키지 버전이 바뀌면 원본 CSV로 재학습하는 편이 안전합니다.
