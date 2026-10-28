"""Standalone Korean HTML report, usable without a server or network connection."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

STATUS = {"candidate_recommendation": "비교 가능한 후보", "near_tie": "상위 후보 차이 작음",
          "single_supported_formation": "비교 대형 부족", "insufficient_support": "추천 보류",
          "not_validated": "전술 효과 미검증"}


def write_report(results: list[dict], metrics: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat, summaries, table = [], [], []
    for index, result in enumerate(results):
        best = next((r for r in result["candidates"] if r["formation"] == result["recommended_formation"]), None)
        summary = {"team": result["team"], "season": result["season"], "as_of": result["as_of"],
                   "formation": result["recommended_formation"], "status": result["status"],
                   "expected_points": best["expected_points"] if best else None,
                   "p_win": best["p_win"] if best else None,
                   "team_matches": best["team_matches"] if best else 0,
                   "gap_to_next_supported": result["gap_to_next_supported"]}
        summaries.append(summary)
        for candidate in result["candidates"]:
            flat.append({"team": result["team"], "season": result["season"], "as_of": result["as_of"],
                         **candidate, "selected": candidate["formation"] == result["recommended_formation"]})
        points_text = f'{best["expected_points"]:.3f}' if best else "—"
        table.append(f'<tr data-index="{index}" tabindex="0"><td>{html.escape(result["team"])}</td>'
                     f'<td>{html.escape(result["recommended_formation"] or "추천 보류")}</td>'
                     f'<td>{points_text}</td><td>{summary["team_matches"]}</td>'
                     f'<td>{STATUS[result["status"]]}</td></tr>')
    pd.DataFrame(flat).to_csv(output_dir / "formation_rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / "team_recommendations.csv", index=False, encoding="utf-8-sig")
    (output_dir / "recommendations.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps({"results": results, "metrics": metrics}, ensure_ascii=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    page = TEMPLATE.replace("__PAYLOAD__", payload).replace("__TABLE__", "\n".join(table))
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


TEMPLATE = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPL 전술대형 연구 모델</title>
<style>
:root{color-scheme:dark;--bg:#0b1420;--panel:#122235;--ink:#e6eff7;--muted:#a3b6ca;--mint:#65e3b2;--line:#2d4054}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,'Malgun Gothic',sans-serif}
main{max-width:1200px;margin:auto;padding:40px 24px}h1{font-size:36px;letter-spacing:-1px;margin:8px 0}h2{margin:0 0 16px;font-size:21px}
.eyebrow{color:var(--mint);font-size:12px;letter-spacing:3px}p{margin:8px 0}.muted,small{color:var(--muted)}
.notice{padding:16px 20px;background:#30291d;border:1px solid #6f5930;border-radius:12px;margin:24px 0}
.grid{display:grid;grid-template-columns:1fr 1.3fr;gap:20px}.panel{padding:24px;background:var(--panel);border:1px solid var(--line);border-radius:16px;margin-bottom:20px;min-width:0}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.stat{padding:18px;background:var(--panel);border-radius:12px}.stat strong{display:block;font-size:26px}.stat span{color:var(--muted);font-size:13px}
label{display:block;margin:12px 0 6px}select{width:100%;padding:12px;background:#0b1928;border:1px solid #50657b;border-radius:8px;color:var(--ink);font:inherit}
svg{display:block;width:100%;max-width:400px;margin:20px auto;background:#103d34;border-radius:8px}svg text{fill:white;text-anchor:middle;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px;text-align:left}th,td{padding:12px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}
.scroll{overflow-x:auto}tbody tr[data-index]{cursor:pointer}tbody tr[data-index]:hover{background:#21384a}a{color:var(--mint)}
.bar{height:12px;display:flex;border-radius:6px;overflow:hidden;margin:14px 0}.win{background:#65e3b2}.draw{background:#91b1d5}.loss{background:#dc897e}
.key{display:flex;gap:14px;font-size:13px;flex-wrap:wrap}.good{color:var(--mint)}.weak{color:#e4c48b}.badge{color:var(--mint);font-size:13px}
footer{margin-top:20px;font-size:12px;color:var(--muted)}@media(max-width:800px){.grid{grid-template-columns:1fr}.stats{gap:8px}.stat{padding:12px}.stat strong{font-size:21px}h1{font-size:28px}main{padding:24px 16px}}
</style></head><body><main>
<div class="eyebrow">FOOTBALL ANALYTICS · RESEARCH MODEL</div><h1>EPL 팀별 전술대형 탐색</h1>
<p class="muted" id="scope"></p>
<div class="notice"><strong>과거 시즌 연구 결과입니다.</strong> 현재 선수단의 최적 전술을 뜻하지 않습니다.
포메이션별 예상 승·무·패로 기대 승점 = 3 × 승리확률 + 무승부확률을 비교합니다. 관측 데이터만으로 전술의 인과 효과를 입증할 수 없습니다.</div>
<div class="stats"><div class="stat"><strong id="match-count"></strong><span>실제 경기 수</span></div>
<div class="stat"><strong id="test-accuracy"></strong><span>미래 구간 결과 예측 정확도</span></div>
<div class="stat"><strong id="test-count"></strong><span>테스트 구간 경기 수</span></div></div>
<div class="grid"><section class="panel"><h2>팀과 후보 대형</h2>
<label for="team">팀 선택</label><select id="team"></select><label for="formation">포메이션 선택</label><select id="formation"></select>
<p id="recommendation" class="badge"></p><svg id="pitch" viewBox="0 0 320 420" role="img" aria-label="선택한 대형의 선수 배치 개념도"></svg>
<small>선수 이름·역할이 없는 대형 개념도입니다. 공격 방향 ↑</small></section>
<section class="panel"><h2 id="team-title"></h2><p id="context" class="muted"></p>
<h1 id="points"></h1><div class="bar"><span id="pwin" class="win"></span><span id="pdraw" class="draw"></span><span id="ploss" class="loss"></span></div>
<div class="key" id="probabilities"></div><p id="support"></p><p id="candidate-note" class="weak"></p>
<div id="warnings" class="muted"></div><div class="scroll"><table><thead><tr><th>대형</th><th>기대 승점</th><th>팀 사용</th><th>비교 기준</th></tr></thead><tbody id="candidates"></tbody></table></div></section></div>
<section class="panel"><h2>모든 팀 비교</h2><p class="muted">같은 시즌의 다른 모든 팀을 상대로 홈·원정 상황에 동일 가중치를 적용했습니다. 행을 선택하면 상세 결과를 볼 수 있습니다.</p>
<div class="scroll"><table><thead><tr><th>팀</th><th>지원되는 후보 중 1위</th><th>기대 승점</th><th>팀 사용</th><th>비교 상태</th></tr></thead><tbody id="teams">__TABLE__</tbody></table></div>
<p><a href="team_recommendations.csv" download>팀별 CSV</a> · <a href="formation_rankings.csv" download>전체 대형별 CSV</a> · <a href="recommendations.json" download>JSON</a></p></section>
<section class="panel"><h2>검증 결과와 해석</h2><p id="split" class="muted"></p>
<div class="scroll"><table><thead><tr><th>모델</th><th>정확도 ↑</th><th>로그손실 ↓</th><th>Brier ↓</th></tr></thead><tbody id="metrics"></tbody></table></div>
<p id="evidence" class="weak"></p><p class="muted">학습 입력: 팀·상대팀·후보 포메이션, 홈 여부, 직전 5경기 승점·득점·실점, Elo, 휴식일, 누적 경기 수.
경기 당일 결과는 그날의 입력에 사용하지 않습니다. 같은 경기의 홈·원정 행은 같은 분할에 속합니다.
검증은 관측된 대형의 승·무·패 예측에 대한 것이며, 사용하지 않은 대형의 성능 검증은 아닙니다. 확률 보정과 추천 정책의 인과 검증은 수행하지 않았습니다.</p></section>
<footer>Data provided by <a href="https://github.com/statsbomb/open-data">StatsBomb Open Data (Hudl)</a>.
<a href="https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf">데이터 이용 조건</a> ·
공유·배포 시 제공자의 출처·로고 표시 조건을 확인하세요. 모델 및 수치: 로컬 scikit-learn 학습 결과.</footer>
</main><script id="data" type="application/json">__PAYLOAD__</script><script>
const {results,metrics}=JSON.parse(document.getElementById('data').textContent);
const el=id=>document.getElementById(id), pct=n=>(100*n).toFixed(1)+'%';
const put=(id,text)=>el(id).textContent=text;
const statuses={candidate_recommendation:'비교 가능한 후보',near_tie:'상위 후보 차이 작음',single_supported_formation:'비교 대형 부족',insufficient_support:'추천 보류',not_validated:'전술 효과 미검증'};
const option=(value,text)=>{const x=document.createElement('option');x.value=value;x.textContent=text;return x};
results.forEach((r,i)=>el('team').append(option(i,r.team)));
put('scope',`${metrics.data.recommendation_season} · ${metrics.data.teams.length}개 팀 · 학습 데이터 종료 ${metrics.data.trained_through} · 분석 기준일 ${results[0].as_of}`);
put('match-count',metrics.data.matches);put('test-accuracy',pct(metrics.test.formation_model.accuracy));put('test-count',metrics.split.test.matches);
put('split',`학습 ${metrics.split.train.first_date} ~ ${metrics.split.train.last_date} / 검증 ${metrics.split.validation.first_date} ~ ${metrics.split.validation.last_date} / 테스트 ${metrics.split.test.first_date} ~ ${metrics.split.test.last_date}`);
const names={formation_model:'포메이션 포함 Random Forest',without_formation:'포메이션 제외 Random Forest',class_prior:'학습 구간 결과 비율'};
function row(parent,values){const tr=document.createElement('tr');values.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td)});parent.append(tr);return tr}
Object.entries(metrics.test).forEach(([name,m])=>row(el('metrics'),[names[name],pct(m.accuracy),m.log_loss.toFixed(4),m.multiclass_brier.toFixed(4)]));
put('evidence',`포메이션 포함 시 로그손실 변화(제외 − 포함): ${metrics.formation_log_loss_improvement_vs_ablation.toFixed(4)}. 단일 테스트 구간의 작은 차이이며, 통계적 유의성이나 전술 효과의 증거로 해석하면 안 됩니다.`);
function svgNode(tag,attrs){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));el('pitch').append(node);return node}
function pitch(formation){el('pitch').replaceChildren();svgNode('rect',{x:15,y:15,width:290,height:390,fill:'none',stroke:'#87b3a4','stroke-width':1});
svgNode('line',{x1:15,x2:305,y1:210,y2:210,stroke:'#87b3a4'});svgNode('circle',{cx:160,cy:210,r:44,fill:'none',stroke:'#87b3a4'});
svgNode('rect',{x:75,y:330,width:170,height:75,fill:'none',stroke:'#87b3a4'});svgNode('rect',{x:75,y:15,width:170,height:75,fill:'none',stroke:'#87b3a4'});
const lines=[1,...formation.split('-').map(Number)];lines.forEach((count,line)=>{for(let n=0;n<count;n++){const x=30+(260*(n+1)/(count+1)),y=375-line*(310/(lines.length-1));svgNode('circle',{cx:x,cy:y,r:14,fill:line===0?'#edbd68':'#65e3b2',stroke:'#0d2631','stroke-width':2});const t=svgNode('text',{x,y:y+4,style:'fill:#09282b;font-weight:700;font-size:10px'});t.textContent=line===0?'GK':String(count)}})}
function showCandidate(){const r=results[Number(el('team').value)],c=r.candidates.find(c=>c.formation===el('formation').value);pitch(c.formation);
put('points',c.expected_points.toFixed(3)+' 예상 승점 / 경기');
el('pwin').style.width=pct(c.p_win);el('pdraw').style.width=pct(c.p_draw);el('ploss').style.width=pct(c.p_loss);
put('probabilities',`승 ${pct(c.p_win)} · 무 ${pct(c.p_draw)} · 패 ${pct(c.p_loss)}`);
put('support',`해당 팀 사용 ${c.team_matches}경기 · 전체 학습 데이터의 팀 단위 사용 ${c.league_team_matches}회`);
put('candidate-note',c.supported?'최소 사용 사례 기준을 충족한 후보입니다. 이것은 신뢰구간이나 확신도를 뜻하지 않습니다.':'관측 사례가 부족하여 추천 대상에서 제외한 참고 점수입니다.');}
function showTeam(){const r=results[Number(el('team').value)];el('formation').replaceChildren();r.candidates.forEach(c=>el('formation').append(option(c.formation,c.formation+(c.supported?'':' · 참고만'))));
el('formation').value=r.recommended_formation||r.candidates[0].formation;put('team-title',r.team);put('context',`다른 ${metrics.data.teams.length-1}개 팀 × 홈·원정 동일 가중 평균 · 과거 선수단 상태`);
put('recommendation',`지원 후보 중 1위: ${r.recommended_formation||'보류'} · ${statuses[r.status]}`);
el('warnings').replaceChildren();r.warnings.forEach(w=>{const p=document.createElement('p');p.textContent=w;el('warnings').append(p)});
el('candidates').replaceChildren();r.candidates.forEach(c=>row(el('candidates'),[c.formation,c.expected_points.toFixed(3),c.team_matches,c.supported?'충족':'참고만']));showCandidate();}
el('team').addEventListener('change',showTeam);el('formation').addEventListener('change',showCandidate);
el('teams').querySelectorAll('tr').forEach(tr=>{const activate=()=>{el('team').value=tr.dataset.index;showTeam();el('team').scrollIntoView({behavior:'smooth',block:'center'})};tr.addEventListener('click',activate);tr.addEventListener('keydown',event=>{if(event.key==='Enter')activate()})});showTeam();
</script></body></html>'''
