"""챗봇 도구 (LLM 무관, 순수 로직).

기존 artifact(defect_model.pkl) · storage.dashboard_payload · 예측 로직만 읽는다.
챗봇은 읽기·계산·설명만 하며 공정 제어·자동 알림 발송은 하지 않는다
(docs/manufacturing_dashboard_design.md 4.3).
"""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd

from build_dashboard import MODEL_OUT
from storage import dashboard_payload


def _artifact() -> dict[str, Any]:
    if not MODEL_OUT.exists():
        raise RuntimeError("모델이 없습니다. python3 scripts/train_model.py 실행 필요")
    return pickle.loads(MODEL_OUT.read_bytes())


def _out_of_band(values: dict[str, float | None], specs: dict[str, Any]) -> list[dict]:
    """정상(운영구간)을 벗어난 피쳐를 이탈량 순으로 반환."""
    out = []
    for feature, spec in specs.items():
        low, high = spec.get("low"), spec.get("high")
        value = values.get(feature)
        if low is None or high is None or value is None:
            continue
        width = (high - low) or 1.0
        if value > high:
            dev = (value - high) / width
        elif value < low:
            dev = (low - value) / width
        else:
            continue
        out.append({
            "feature": feature, "label": spec.get("label", feature), "grade": spec.get("grade"),
            "value": round(float(value), 3), "low": low, "high": high, "dev": round(float(dev), 3),
        })
    out.sort(key=lambda d: d["dev"], reverse=True)
    return out


# ── 도구들 ────────────────────────────────────────────────

def get_status() -> dict[str, Any]:
    """현재 성능지표·경보 현황·임계값 요약."""
    art = _artifact()
    pay = dashboard_payload(art)
    thr = pay["threshold"]
    events = pay["events"]
    high = [e for e in events if e["p"] >= thr]
    return {
        "title": "Foundry Guard · 주조 불량 예방 관제",
        "metrics": pay["metrics"],
        "threshold": thr,
        "source": pay["source"],
        "evaluated_events": len(events),
        "high_risk_count": len(high),
        "note": "성능은 과거 데이터 오프라인 결과이며 운영 성능을 보장하지 않음",
    }


def list_high_risk(limit: int = 5) -> dict[str, Any]:
    """불량확률이 임계값 이상인 고위험 이벤트 상위 N개."""
    art = _artifact()
    pay = dashboard_payload(art)
    thr, specs = pay["threshold"], pay["specs"]
    events = [e for e in pay["events"] if e["p"] >= thr]
    events.sort(key=lambda e: e["p"], reverse=True)
    rows = []
    for e in events[:limit]:
        oob = _out_of_band(e["v"], specs)
        rows.append({"time": e["t"], "probability": e["p"], "actual": e["y"],
                     "top_reasons": [{"label": o["label"], "value": o["value"],
                                      "normal": [o["low"], o["high"]]} for o in oob[:3]]})
    return {"threshold": thr, "count": len(events), "events": rows}


def explain_event(event_time: str | None = None) -> dict[str, Any]:
    """특정 이벤트(미지정 시 최상위 고위험)의 원인 = 정상구간 이탈 변수."""
    art = _artifact()
    pay = dashboard_payload(art)
    thr, specs, events = pay["threshold"], pay["specs"], pay["events"]
    target = None
    if event_time:
        target = next((e for e in events if e["t"] == event_time), None)
    if target is None:
        high = sorted([e for e in events if e["p"] >= thr], key=lambda e: e["p"], reverse=True)
        if not high:
            return {"error": "고위험 이벤트가 없습니다."}
        target = high[0]
    oob = _out_of_band(target["v"], specs)
    return {
        "time": target["t"], "probability": target["p"], "actual": target["y"],
        "out_of_band": oob,
        "interpretation": "정상 운영구간을 벗어난 변수일수록 우선 확인 대상. 통계적 관계이며 인과는 아님.",
    }


def simulate_threshold(value: float, fp_unit: float = 1.0, fn_unit: float = 10.0) -> dict[str, Any]:
    """임계값 변경 시 경보수·오탐·미탐·총비용·정밀도·재현율 + 비용최적 임계값.

    fp_unit: 오탐(불필요 점검) 단가, fn_unit: 미탐(불량 유출) 단가.
    """
    art = _artifact()
    pay = dashboard_payload(art)
    ev = [e for e in pay["events"] if e["y"] is not None]
    if not ev:
        return {"error": "실측 라벨이 있는 이벤트가 없습니다."}
    p = np.array([e["p"] for e in ev])
    y = np.array([e["y"] for e in ev])

    def cost_at(t):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return {"alerts": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
                "cost": fp * fp_unit + fn * fn_unit,
                "precision": round(prec, 4), "recall": round(rec, 4)}

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    opt = float(min(grid, key=lambda t: cost_at(t)["cost"]))
    r = cost_at(float(value))
    r.update({"threshold": float(value), "cost_optimal_threshold": opt,
              "fp_unit": fp_unit, "fn_unit": fn_unit})
    return r


def predict(values: dict[str, float | None]) -> dict[str, Any]:
    """새 측정값의 불량 확률·판정·스펙 이탈 (POST /api/predict 와 동일 로직)."""
    art = _artifact()
    features, medians, specs, thr = art["features"], art["medians"], art["specs"], art["threshold"]
    row = {f: values.get(f, medians.get(f)) for f in features}
    frame = pd.DataFrame([row], columns=features).fillna(pd.Series(medians))
    prob = float(art["model"].predict_proba(frame)[0][1])
    outside = _out_of_band(row, specs)
    grade_a_breach = any(o["grade"] == "A" for o in outside)
    level = ("위험" if prob >= thr and grade_a_breach
             else "주의" if prob >= thr or outside
             else "정상")
    return {"defect_probability": round(prob, 4), "risk_level": level, "threshold": thr,
            "out_of_band_features": [o["label"] for o in outside],
            "note": "실시간 입력에 사후측정값(강도·비스킷두께)이 포함되면 결과 해석에 주의(설계서 2.2)"}


def recommend_action(event_time: str | None = None) -> dict[str, Any]:
    """위험 이벤트에 대한 권장 조치 후보. 실행하지 않으며 관리자 승인 전제."""
    ev = explain_event(event_time)
    if "error" in ev:
        return ev
    actions = []
    for o in ev.get("out_of_band", [])[:3]:
        actions.append(f"{o['label']}(현재 {o['value']}) → 정상범위 {o['low']}~{o['high']} 확인·조정 검토")
    if not actions:
        actions = ["정상범위 이탈 변수 없음 → 센서·재측정 재현성 확인 권장"]
    return {
        "time": ev["time"], "probability": ev["probability"],
        "action_candidates": actions,
        "decision": "고위험 판정 시 추가 검사 또는 공정조건 조정 후 재생산 검토",
        "note": "모두 관리자 승인 전 후보. 챗봇은 제어·자동 알림을 수행하지 않음(설계서 4.3).",
    }


# 이름 → 함수 매핑 (service 에서 사용)
REGISTRY = {
    "get_status": get_status,
    "list_high_risk": list_high_risk,
    "explain_event": explain_event,
    "simulate_threshold": simulate_threshold,
    "predict": predict,
    "recommend_action": recommend_action,
}
