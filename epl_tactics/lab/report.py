"""Assemble the four-step lab page: payload (Python) + shell.html + section fragments + engine/chart scripts."""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import GROUPS, build_catalog
from .context import LabContext, load_context

HERE = Path(__file__).parent
SECTIONS = [("s1", "데이터"), ("s2", "전처리"), ("s3", "모델 학습"), ("s4", "예측 실험")]
SHORT = {"Manchester City": "Man City", "Manchester United": "Man Utd", "Tottenham Hotspur": "Tottenham"}
def step_blurbs(rf_trials: int) -> list[str]:
    return [
        "어떤 경기·기록을 모았고, 무엇을 맞히려 하며(정답) 무엇으로 맞히는지(입력)를 확인합니다.",
        "원시 기록을 ‘경기 전에 알 수 있는 값’으로 바꾸고, 빈 칸·단위 차이를 정리한 뒤 분포를 그려 봅니다.",
        f"랜덤 포레스트를 학습하고 설정(하이퍼파라미터) {rf_trials}가지를 시간순 검증으로 비교해 가장 좋은 조합을 고릅니다.",
        "처음 보는 경기나 가상의 대결 조건을 직접 넣어 승·무·패 확률과 그 이유를 확인합니다.",
    ]


def sanitize(value):
    """Recursively make a payload JSON-safe: numpy -> python, NaN/inf -> None, Timestamp -> ISO date."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_meta(ctx: LabContext) -> dict:
    test = ctx.report["test"]
    clubs = list(ctx.report["protocol"]["clubs"])
    keep = ["accuracy", "log_loss", "brier"]
    return {
        "clubs": clubs, "clubs_short": [SHORT.get(c, c) for c in clubs],
        "n_matches": int(ctx.report["data"]["unique_matches"]), "n_rows": int(len(ctx.rows)),
        "first_date": str(ctx.rows.date.min().date()), "last_date": str(ctx.rows.date.max().date()),
        "test_start": ctx.test_start, "test_team_rows": int(test["selected_with_formation"]["team_appearances"]),
        "test_matches": int(test["selected_with_formation"]["unique_matches"]),
        "scores": {"rf": {k: test["selected_with_formation"][k] for k in keep}, "elo": {k: test["elo_logistic"][k] for k in keep},
                   "always": {"accuracy": test["always_team_win"]["accuracy"]}, "untuned": {k: test["untuned_rf"][k] for k in keep}},
        "step_blurbs": step_blurbs(int(ctx.report["protocol"]["rf_trials_per_variant"])),
        "source_note": "자료: Football-Data(경기 결과·배당)와 ESPN 공개 웹 경기 데이터(포메이션·선발·팀 기록). ESPN 인터페이스는 비문서화되어 있고 공개 접근이 자유 재배포 라이선스를 뜻하지 않습니다.",
    }


def build_payload(ctx: LabContext, only: list[str] | None = None) -> dict:
    payload = {"meta": build_meta(ctx), "catalog": build_catalog(ctx.numeric),
               "groups": [{"key": k, "label": label, "desc": desc} for k, label, desc in GROUPS]}
    for key, _ in SECTIONS:
        if only and key not in only:
            continue
        payload[key] = importlib.import_module(f"{__package__}.export_{key}").build(ctx)
    return sanitize(payload)


def fragment(key: str) -> str:
    inner = (HERE / f"{key}.html").read_text(encoding="utf-8")      # a missing step file is a packaging error: fail, don't ship an empty tab
    return (f'<section id="{key}" class="step" role="tabpanel" aria-labelledby="tab-{key}" hidden>\n{inner}\n</section>')


def render(payload: dict) -> str:
    shell = (HERE / "shell.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Keep the JSON inert inside <script>: no tag/comment/entity can start, and U+2028/2029 stay escaped.
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    sections = "\n".join(fragment(key) for key, _ in SECTIONS)
    page = shell.replace("<!--__SECTIONS__-->", sections)
    for token, text in [("/*__ENGINE__*/", (HERE / "engine.js").read_text(encoding="utf-8")),
                        ("/*__CHARTS__*/", (HERE / "charts.js").read_text(encoding="utf-8"))]:
        page = page.replace(token, text.replace("</script", "<\\/script"))
    return page.replace("__PAYLOAD__", data)


def write_lab_report(data_dir: str | Path = "data/focused", model_dir: str | Path = "artifacts/focused",
                     output_dir: str | Path = "reports/focused", context: LabContext | None = None) -> Path:
    ctx = context or load_context(data_dir, model_dir)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "lab.html"
    path.write_text(render(build_payload(ctx)), encoding="utf-8")
    return path
