"""Local interactive evidence report for the six-club collection and tuned models."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .focused_data import CLUBS, write_json

LABELS = {"always_team_win":"항상 해당 구단 승리", "team_venue_prior":"구단·홈/원정 결과 비율", "elo_logistic":"Elo 기준",
          "untuned_rf":"튜닝 전 Random Forest", "tuned_rf":"튜닝한 Random Forest",
          "selected_with_formation":"검증 선택 · 포메이션 포함", "selected_without_formation":"검증 선택 · 포메이션 제외",
          "same_spec_without_formation":"같은 설정 · 포메이션 제외",
          "uniform_random":"균등 랜덤 기대값"}


def write_focused_report(data_dir: str | Path = "data/focused", model_dir: str | Path = "artifacts/focused",
                         output_dir: str | Path = "reports/focused") -> Path:
    data_dir,model_dir,root = Path(data_dir),Path(model_dir),Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    report = json.loads((model_dir/"metrics.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(model_dir/"feature_rows.csv")
    coverage = pd.read_csv(data_dir/"coverage.csv")
    formations = []
    for (team,formation),period in rows[rows.formation.ne("__missing__")].groupby(["team","formation"]):
        recent = period[pd.to_datetime(period.date)>=pd.to_datetime(rows.date).max()-pd.Timedelta(days=730)]
        formations.append({"team":team,"formation":formation,"matches":len(period),"recent_matches":len(recent),
                           "win_rate":float(period.outcome.eq(2).mean()),
                           "points":float(period.outcome.map({0:0,1:1,2:3}).mean()),
                           "first":period.date.min(),"last":period.date.max()})
    pd.DataFrame(formations).to_csv(root/"formation_observations.csv",index=False,encoding="utf-8-sig")
    for filename in ["selection.json","trials_formation.json","trials_no_formation.json","validation_importance.csv"]:
        shutil.copyfile(model_dir/filename,root/filename)
    shutil.copyfile(data_dir/"coverage.csv",root/"coverage.csv")
    write_json(root/"metrics.json",report)
    counts = coverage.groupby("team")[["matches","formation_recorded","starting_eleven_recorded"]].sum().reindex(CLUBS)
    fig,axis = plt.subplots(figsize=(11,4.4))
    counts.rename(columns={"matches":"All EPL matches","formation_recorded":"Recorded starting formation",
                           "starting_eleven_recorded":"Complete starting XI"}).plot.barh(ax=axis,color=["#739db5","#35b68f","#c5ac63"])
    axis.invert_yaxis()
    axis.set_xlabel("Actual team appearances (not duplicated synthetic samples)")
    axis.set_ylabel("")
    axis.set_title("Collected evidence per club, 2006/07 onward")
    axis.spines[["top","right"]].set_visible(False)
    axis.legend(loc="lower right",fontsize=8)
    fig.tight_layout()
    fig.savefig(root/"coverage.png",dpi=160)
    fig.savefig(root/"coverage.svg")
    plt.close(fig)
    summary = []
    for team in CLUBS:
        period = rows[rows.team.eq(team)]
        count = counts.loc[team]
        score = report["test_by_team"][team]
        summary.append({"team":team,"matches":int(count.matches),"formations":int(count.formation_recorded),
                        "starting_elevens":int(count.starting_eleven_recorded),"first":period.date.min(),"last":period.date.max(),
                        "test_appearances":score["tuned_rf"]["team_appearances"],
                        "tuned_rf_accuracy":score["tuned_rf"]["accuracy"],"tuned_rf_log_loss":score["tuned_rf"]["log_loss"],
                        "untuned_rf_accuracy":score["untuned_rf"]["accuracy"],
                        "always_win_accuracy":score["always_team_win"]["accuracy"]})
    pd.DataFrame(summary).to_csv(root/"club_summary.csv",index=False,encoding="utf-8-sig")
    trials = json.loads((model_dir/"trials_formation.json").read_text(encoding="utf-8"))
    payload = json.dumps({"report":report,"clubs":summary,"formations":formations,"coverage":coverage.to_dict("records"),
                          "labels":LABELS,"trials":trials},ensure_ascii=False)
    payload = payload.replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e")
    page = TEMPLATE.replace("__PAYLOAD__",payload)
    path = root/"index.html"
    path.write_text(page,encoding="utf-8")
    gain = report["comparisons"]["tuning_gain"]["log_loss"]
    lines = ["# 유명 6개 구단 집중 수집·하이퍼파라미터 튜닝", "",
             f'실제 고유 경기 {report["data"]["unique_matches"]:,}개. 지도학습 구단 관점 행 {report["features"]["supervised_team_rows"]:,}개. 두 수는 다릅니다.',"",
             "| 구단 | 실제 경기 | 포메이션 기록 | 선발 11명 기록 | 평가 경기 | 튜닝 RF 정확도 | 항상 구단 승리 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for club in summary:
        lines.append(f'| {club["team"]} | {club["matches"]} | {club["formations"]} | {club["starting_elevens"]} | {club["test_appearances"]} | {club["tuned_rf_accuracy"]:.2%} | {club["always_win_accuracy"]:.2%} |')
    lines += ["", "## 튜닝", "", f'RF 설정 {report["protocol"]["rf_trials_per_variant"]}개 × 시간순 3개 구간 × 포메이션 포함/제외를 비교했습니다.',
              "", "```json",json.dumps(report["selection"]["with_formation"]["best_rf"]["spec"],indent=2),"```", "",
              f'테스트 로그손실의 튜닝 개선폭: {gain["improvement"]:+.5f}, 주 단위 95% 구간 [{gain["ci95_low"]:+.5f}, {gain["ci95_high"]:+.5f}].',"",
              "## 해석", "", "경기 수 증가의 원인은 구단 수 축소가 아니라 수집 기간 확장입니다. 과거 선수단·감독이 현재와 같지는 않습니다.",
              "다른 구단과의 경기는 제외하지 않았으며, 학습 정답 행만 선택한 6개 구단으로 제한했습니다. 상대팀의 전체 EPL 기록은 경기 전 상대 강도 계산의 문맥 자료입니다.",
              "2024/25 이후 시즌은 기존 실험에서도 평가한 자료입니다. 이번 튜닝에는 쓰지 않았지만 완전히 새로운 미래 검증이라고 부르지 않습니다.",
              "포메이션별 승률·승점은 상대·선수·감독을 통제한 인과 효과가 아니므로 최적 대형으로 해석하지 않습니다.",
              "누락된 포메이션을 만들어 넣지 않았습니다. 부상·징계·선수 능력·감독 기록은 확보했다고 주장하지 않습니다.",
              "선발 선수의 생산성은 수집한 이전 경기에서 관측한 기록만 사용하며 선수의 전체 경력 수치가 아닙니다.",
              "수집 출처: ESPN 공개 웹 경기 데이터 및 Football-Data. 접근 가능성과 자유 재배포 라이선스는 별개입니다.","",
              "상세 수치와 구단별·시즌별 결과는 metrics.json, 모든 튜닝 시도는 trials_formation.json / trials_no_formation.json에 저장했습니다."]
    (root/"analysis.md").write_text("\n".join(lines),encoding="utf-8")
    return path


TEMPLATE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>유명 6개 구단 집중 분석</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1420;color:#e6eff7;font:15px/1.75 system-ui,'Malgun Gothic',sans-serif}main{max-width:1240px;margin:auto;padding:36px 24px}h1{font-size:34px;margin:8px 0}h2{font-size:22px;margin:0 0 12px}h3{font-size:17px}.eyebrow{color:#65e3b2;letter-spacing:3px;font-size:12px}p,small{color:#afc1d2}.panel{padding:24px;border:1px solid #2d4054;border-radius:15px;background:#122235;margin:20px 0;min-width:0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.stat{padding:18px;background:#122235;border-radius:12px}.stat strong{display:block;font-size:27px;color:#65e3b2}.stat span{font-size:12px;color:#afc1d2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid .panel{margin:0}a{color:#65e3b2}select{padding:12px;border:1px solid #516b80;background:#0b1928;border-radius:8px;color:#e6eff7;width:100%;font:inherit;margin-bottom:14px}table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}td,th{padding:11px 8px;border-bottom:1px solid #2d4054}th{font-weight:500;color:#afc1d2}.scroll{overflow-x:auto}img{max-width:100%;background:white;border-radius:8px}pre{padding:16px;background:#0b1928;border-radius:8px;overflow:auto;max-height:330px;font-size:12px}.notice{padding:18px 22px;border:1px solid #806431;background:#2e281c;border-radius:12px;margin:22px 0}.notice p{margin:4px 0}.tag{color:#65e3b2}footer{font-size:12px;color:#afc1d2}@media(max-width:800px){main{padding:24px 14px}h1{font-size:27px}.stats{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.panel{padding:18px}}</style></head>
<body><main><div class="eyebrow">FOCUSED CLUBS · LONG HISTORY · TUNED MODELS</div><h1>유명 6개 구단 집중 분석</h1><p id="scope"></p>
<p><a href="../index.html">전술대형 탐색 화면</a> · <a href="../analysis.html">전체 EPL 이전 분석</a></p>
<div class="stats"><div class="stat"><strong id="games"></strong><span>수집한 고유 경기</span></div><div class="stat"><strong id="rows"></strong><span>선택 구단 관점 학습 행</span></div><div class="stat"><strong id="variables"></strong><span>원래 입력 변수 수 · 인코딩 전</span></div><div class="stat"><strong id="trials"></strong><span>RF 설정 / 포함·제외 각각</span></div></div>
<div class="notice"><p>구단별 실제 경기·선발 기록 수를 따로 집계했습니다. 대형별 승률은 관측된 성적이며 최적 전술의 효과가 아닙니다.</p><p>2024/25 이후는 이번 튜닝에 사용하지 않았지만 이전 실험에서 이미 본 평가 기간입니다. 별도의 미래 검증이 필요합니다.</p></div>
<section class="panel"><h2>수집한 자료의 양과 누락</h2><img src="coverage.png" alt="구단별 실제 경기·포메이션·선발 11명 확보 수"><div class="scroll"><table><thead><tr><th>구단</th><th>전체 경기</th><th>포메이션 있음</th><th>선발 11명 있음</th><th>시작</th><th>종료</th></tr></thead><tbody id="coverage"></tbody></table></div><p id="verified"></p><p>상대팀 전체 리그 결과는 상대 강도 계산의 문맥 자료이며, 지도학습 정답은 선택한 6개 구단만 사용합니다. 없는 대형·선발·부상 기록을 추정해 채우지 않았습니다.</p><p>오래된 선수단·감독의 경기는 현재 팀의 경기와 동등하지 않습니다. 학습에는 최근 자료에 더 높은 가중치(반감기 3년)를 줬으므로 원시 경기 수 전체가 현재 전술의 유효 표본 수인 것은 아닙니다.</p></section>
<div class="grid"><section class="panel"><h2>구단별 평가</h2><label for="club">분석 구단</label><select id="club"></select><p id="club-summary"></p><div class="scroll"><table><thead><tr><th>모델</th><th>정확도</th><th>로그손실 ↓</th></tr></thead><tbody id="club-metrics"></tbody></table></div></section>
<section class="panel"><h2>실제 포메이션 사용 기록</h2><p>과거 전체 관측 성적입니다. 상대·감독·선수단 조건을 통제한 인과 비교가 아닙니다. 최근 사용은 마지막 730일입니다.</p><div class="scroll"><table><thead><tr><th>대형</th><th>전체 사용</th><th>최근 사용</th><th>승률</th><th>평균 승점</th></tr></thead><tbody id="formations"></tbody></table></div></section></div>
<section class="panel"><h2>같은 평가 경기의 전체 결과</h2><p id="test-scope"></p><div class="scroll"><table><thead><tr><th>모델</th><th>정확도</th><th>로그손실 ↓</th><th>Brier ↓</th></tr></thead><tbody id="metrics"></tbody></table></div><p>6개 구단 간 맞대결의 두 관점은 각각 0.5 가중치로 합계 1경기만 반영합니다. 유명 구단의 기본 승률이 높으므로 항상 해당 구단 승리 기준과 비교해야 합니다.</p><p id="market"></p></section>
<section class="panel"><h2>Random Forest 하이퍼파라미터 튜닝</h2><p id="tuning"></p><p>후보 간 비교는 시간순 검증 로그손실로 결정했습니다. 테스트 정확도를 보고 설정을 다시 고르지 않았습니다.</p><pre id="parameters"></pre><h3>검증에서 좋은 RF 설정 상위 8개</h3><div class="scroll"><table><thead><tr><th>시도</th><th>트리</th><th>깊이</th><th>말단 최소</th><th>특성 비율</th><th>검증 로그손실</th></tr></thead><tbody id="search"></tbody></table></div><p id="tuning-gain"></p><p id="formation-gain"></p></section>
<section class="panel"><h2>어떤 정보가 들어갔는가</h2><p>경기 전까지의 최근 5·10·20경기 승점·득실점·슈팅·유효슈팅·점유율·코너킥·파울·카드·패스·크로스·롱볼·태클·가로채기·선방·블록, Elo, 일정·휴식, 홈·원정별 경기력, 발표된 선발 구성과 이전 관측 선수 기록, 대형 구조·변경·사용 빈도를 사용합니다.</p><p>해당 경기의 득점·경기 통계·선수 생산성과 배당은 정답 또는 이후 이력 갱신에만 사용합니다. 부상·징계·선수 능력·감독 데이터는 확보한 것으로 취급하지 않습니다.</p><h3>검증 구간의 변수 중요도</h3><p>학습에서 고른 30개 열을 검증 구간에서 섞어 로그손실 증가를 측정했습니다. 전술의 원인이나 인과 효과를 뜻하지 않습니다.</p><div class="scroll"><table><thead><tr><th>입력 열</th><th>검증 손실 증가</th></tr></thead><tbody id="importance"></tbody></table></div></section>
<section class="panel"><h2>재현 가능한 결과 파일</h2><p><a href="analysis.md">분석 요약</a> · <a href="club_summary.csv">구단별 결과 CSV</a> · <a href="coverage.csv">시즌별 수집 범위</a> · <a href="formation_observations.csv">대형 사용 기록</a> · <a href="metrics.json">전체 지표</a></p><p><a href="trials_formation.json">포메이션 포함 전체 튜닝</a> · <a href="trials_no_formation.json">제외 전체 튜닝</a> · <a href="selection.json">선택 설정</a> · <a href="validation_importance.csv">변수 중요도</a></p></section>
<footer>자료: <a href="https://football-data.co.uk/englandm.php">Football-Data</a> · ESPN 공개 웹 경기 데이터. ESPN 인터페이스는 비문서화되어 있고 공개 접근이 자유 재배포 라이선스를 뜻하지 않습니다.</footer></main>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const {report:r,clubs,formations,coverage,labels,trials}=JSON.parse(document.getElementById('payload').textContent);const el=id=>document.getElementById(id),put=(id,t)=>el(id).textContent=t,pct=x=>(100*x).toFixed(2)+'%';
function row(id,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}el(id).append(tr)}
put('scope',`${clubs[0].first} ~ ${clubs[0].last} · Arsenal / Chelsea / Liverpool / Manchester City / Manchester United / Tottenham Hotspur`);put('games',r.data.unique_matches.toLocaleString());put('rows',r.features.supervised_team_rows.toLocaleString());put('variables',r.features.numeric_features.length+r.features.categorical_features.length);put('trials',r.protocol.rf_trials_per_variant);
for(const c of clubs){row('coverage',[c.team,c.matches,c.formations,c.starting_elevens,c.first,c.last]);const o=document.createElement('option');o.value=c.team;o.textContent=c.team;el('club').append(o)}
put('verified',`Football-Data 경기 목록 ${r.data.unique_matches}개 중 ESPN 상세 기록과 날짜·득점·구단·시즌이 일치한 경기 ${r.data.espn_verified_unique_matches}개. 상세 미확보 ${r.data.unmatched_fixture_count}개도 기본 경기 결과를 보존했습니다.`);
const modelOrder=['always_team_win','team_venue_prior','elo_logistic','untuned_rf','tuned_rf','selected_with_formation','selected_without_formation','same_spec_without_formation'];
function show(){const team=el('club').value,c=clubs.find(c=>c.team===team);put('club-summary',`${c.matches}경기 수집 · 포메이션 ${c.formations}경기 · 평가 ${c.test_appearances}경기`);el('club-metrics').replaceChildren();for(const name of modelOrder){const s=r.test_by_team[team][name];row('club-metrics',[labels[name],pct(s.accuracy),s.log_loss.toFixed(4)])}el('formations').replaceChildren();for(const f of formations.filter(f=>f.team===team).sort((a,b)=>b.matches-a.matches)){row('formations',[f.formation,f.matches,f.recent_matches,pct(f.win_rate),f.points.toFixed(3)])}}
el('club').addEventListener('change',show);show();for(const name of [...modelOrder,'uniform_random']){const s=r.test[name];row('metrics',[labels[name],pct(s.accuracy),s.log_loss.toFixed(4),s.brier.toFixed(4)])}
put('test-scope',`2024/25 이후 ${r.test.tuned_rf.unique_matches}개 고유 경기 / ${r.test.tuned_rf.team_appearances}개 구단 관점. 시즌 시작마다 이전 자료만으로 재학습했습니다.`);
put('market',`추가 정보를 가진 경기 전 배당 기준: ${pct(r.market_reference.market.accuracy)} / 로그손실 ${r.market_reference.market.log_loss.toFixed(4)}. 같은 입력의 비교나 베팅 수익 검증은 아닙니다.`);
put('tuning',`시간순 검증: ${r.protocol.folds.join(', ')}. RF ${r.protocol.rf_trials_per_variant}개 설정과 로지스틱·부스팅 8개 설정을 포메이션 포함/제외 각각 비교했습니다.`);put('parameters',JSON.stringify(r.selection.with_formation.best_rf.spec,null,2));
const gain=r.comparisons.tuning_gain.log_loss, fg=r.comparisons.formation_gain.log_loss;put('tuning-gain',`RF 튜닝의 로그손실 개선: ${gain.improvement.toFixed(5)} · 주 단위 95% 구간 [${gain.ci95_low.toFixed(5)}, ${gain.ci95_high.toFixed(5)}]. 양수이면 튜닝 후 손실이 낮습니다.`);put('formation-gain',`포메이션 포함의 로그손실 개선(같은 설정에서 포메이션 열만 제거한 모델 대비): ${fg.improvement.toFixed(5)} · 95% 구간 [${fg.ci95_low.toFixed(5)}, ${fg.ci95_high.toFixed(5)}]. 결과 예측 차이이며 최적 대형을 입증하지 않습니다.`);
for(const f of r.validation_top_features.slice(0,15))row('importance',[f.feature,f.validation_log_loss_importance.toFixed(5)]);
for(const t of trials.filter(t=>t.spec.family==='rf').sort((a,b)=>a.validation.log_loss-b.validation.log_loss).slice(0,8)){const s=t.spec;row('search',[t.trial,s.n_estimators,s.max_depth??'제한 없음',s.min_samples_leaf,s.max_features,t.validation.log_loss.toFixed(4)])}
</script></body></html>'''
