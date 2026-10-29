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

lessons:
  - name: formation-signal-six-clubs
    summary: 6개 구단 4,003경기(구단 관점 4,608행)에서 RF 튜닝은 로그손실을 소폭(+0.0098) 개선했지만 Elo 기준을 못 넘었고, 포메이션 변수 추가 개선은 0과 구분되지 않았다
    outcome: failed
    tech: [RandomForest, logistic, Elo, formation]
  - name: missing-category-as-numeric-signal
    summary: 누락 포메이션 placeholder를 "이전 대형과 달라짐"이나 사용 빈도 분모로 세어 가짜 신호가 생겼다. 누락은 NaN으로 두고 알려진 값만으로 계산해야 한다
    outcome: succeeded
    tech: [pandas, feature-engineering]
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
