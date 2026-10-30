---
project: s2bev-epl-tactics
status: active
summary: EPL 경기 결과 확률 예측과 포메이션 효과 검증 가능성 감사. 6개 빅클럽 장기 수집 + 시간순 RF 튜닝 파이프라인
domains: [football, sports-analytics, tabular-ml, forecasting]

components:
  - name: focused-history-collector
    summary: Football-Data 경기 목록을 기준으로 ESPN 상세(포메이션·선발·팀 스탯)를 날짜·득점·시즌 일치 시에만 붙여 캐시하는 재개 가능 수집기 (시즌 범위 → matches.csv·league_context.csv·coverage.csv)
    tech: [pandas, requests, ThreadPoolExecutor, football-data.co.uk, ESPN site API]
    entry: epl_tactics/focused_data.py
    maturity: beta
    reusability: medium
  - name: pre-match-feature-builder
    summary: 날짜별 일괄 갱신으로 누수 없이 Elo·최근 5/10/20경기·휴식·선발 구성·포메이션 변수를 만드는 구단 관점 피처 (경기 이력 → 학습 행)
    tech: [pandas, numpy, Elo]
    entry: epl_tactics/focused_features.py
    maturity: beta
    reusability: medium
  - name: forward-fold-rf-tuning
    summary: ParameterSampler RF 32개+로지스틱·HGB를 시간순 폴드로 비교, 설정 동결 후 시즌별 재학습 평가·주 단위 부트스트랩 신뢰구간 (피처 행 → 지표·모델)
    tech: [scikit-learn, RandomForestClassifier, joblib, threadpoolctl]
    entry: epl_tactics/focused_model.py
    maturity: beta
    reusability: medium
  - name: browser-rf-lab
    summary: 학습된 sklearn RF·전처리·Elo 기준을 타입 배열(base64)로 내보내 브라우저에서 확률·기여도(tree interpreter)를 그대로 계산하는 4단계 설명·예측 페이지 (모델 + 경기 이력 → 단일 HTML)
    tech: [scikit-learn, JavaScript, base64 typed arrays, playwright-core]
    entry: epl_tactics/lab/report.py
    maturity: beta
    reusability: high

lessons:
  - name: formation-signal-six-clubs
    summary: 6개 구단 4,003경기(구단 관점 4,608행)에서 RF 튜닝은 로그손실을 소폭(+0.0098) 개선했지만 Elo 기준을 못 넘었고, 포메이션 변수 추가 개선은 0과 구분되지 않았다
    outcome: failed
    tech: [RandomForest, logistic, Elo, formation]
  - name: missing-category-as-numeric-signal
    summary: 누락 포메이션 placeholder를 "이전 대형과 달라짐"이나 사용 빈도 분모로 세어 가짜 신호가 생겼다. 누락은 NaN으로 두고 알려진 값만으로 계산해야 한다
    outcome: succeeded
    tech: [pandas, feature-engineering]
  - name: espn-empty-boxscore-as-zero
    summary: ESPN이 기록하지 못한 팀 스탯을 빈값이 아니라 0으로 주고, 수집 단계가 Football-Data 값을 그 0으로 덮어써 점유율 0%·슈팅 0 경기 341개(2016/17은 198경기 중 189)와 가로채기 대부분 0(2018/19 이전)이 롤링 평균에 들어갔다. 이번에는 고치지 않고 실험실 페이지에 공개했다. 재수집 시 0을 결측으로 바꾸고 모델을 다시 돌려야 한다
    outcome: failed
    tech: [pandas, ESPN site API, data-quality]
  - name: lab-scenario-builder-truncated-history-test
    summary: 가상 경기 입력 생성기는 경기 이력을 해당 날짜 직전까지 잘라 만든 스냅샷으로 실제 경기 행을 재현하는지(모든 특성 일치)로 검증했다. Elo 시즌 회귀는 팀의 새 시즌 첫 경기를 처리할 때 늦게 적용되므로 스냅샷은 같은 시즌 안에서만 유효하다
    outcome: succeeded
    tech: [pandas, Elo, node]
  - name: windows-background-job-dies-with-shell
    summary: Bash에서 nohup ... & 로 띄운 학습은 래퍼 셸 종료 시 같이 죽었다. 장시간 학습은 도구의 run_in_background로 직접 실행해야 한다
    outcome: failed
    tech: [windows, git-bash]
---

# 메모

- ESPN 비문서화 API 사용. 공개 접근이 재배포 라이선스를 뜻하지 않아 data/focused/facts는 Git 제외.
- 팀명 매핑 버그(Wigan·Bolton 등 7개 구단 약칭)로 초기 264경기가 ESPN과 조인되지 않았다. 수정 후 전 경기 일치; 수정 전 결과는 폐기.
- 부분 훈련 결과 비교 시 같은 설정에서 포메이션 열만 제거한 모델과 비교해야 모델 계열 효과가 섞이지 않음.
- 브라우저 검증: CHROME_PATH=<chrome.exe> node scripts/verify_report.cjs <playwright-core 경로> (설치된 chromium 버전이 playwright와 달라도 실행 가능)
- 분석 실험실: `python -m epl_tactics report-lab` → reports/focused/lab.html(약 4.3 MB, 모델 학습 약 30초+). 4단계 조각은 epl_tactics/lab/s1~s4.html, 단계별 데이터는 export_s1~s4.py, 공용은 shell.html·charts.js·engine.js. 테스트는 tests/test_lab_*.py (이 환경 Python 3.10에는 pytest가 없어 unittest로 실행: `python -m unittest discover -s tests -p "test_lab_*.py"`). 브라우저 확인: CHROME_PATH=<chrome.exe> node scripts/verify_lab.cjs <playwright-core 경로> <스크린샷 폴더>.
- 실험실 4단계의 고정 모델은 2024/25 이전만으로 한 번 학습해 공식 점수(시즌마다 재학습, 54.3%)와 다르다(53.9%). 페이지에도 명시.
