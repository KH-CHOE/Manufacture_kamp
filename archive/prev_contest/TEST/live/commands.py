"""명령 파싱 → '제안'(실행 아님). 안전 철학: AI는 값을 제안만, 사람이 승인.

명령(음성/텍스트/클릭)은 여기서 **제안(proposal)** 으로만 변환된다. 실제 적용은
사람이 조정 창에서 [적용]/[승인] 을 눌러야 일어난다(stream.approve).

규칙기반 파싱이라 API 키 없이 동작(발표 무대 견고). 숫자를 안 주면 비용최적
임계값 같은 'AI 제안'을 채워 넣는다.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from chatbot import tools

STOP_WORDS = ("정지", "멈춰", "멈춤", "스톱", "중지", "세워", "정지해")
START_WORDS = ("시작", "재개", "가동", "스타트", "돌려", "재시작", "가동해")
THRESHOLD_WORDS = ("임계", "threshold", "기준값", "기준", "민감도")


def _extract_ratio(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    v = float(m.group(1))
    if "%" in text and v > 1:
        v /= 100.0
    elif v > 1:  # "60" → 0.60 로 해석
        v /= 100.0
    return v if 0.0 < v < 1.0 else None


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def threshold_proposal(sim, proposed: float | None = None) -> dict[str, Any]:
    """임계값 조정 제안. proposed 없으면 비용최적값을 AI 제안으로."""
    current = sim.threshold
    cost_opt = None
    try:
        cost_opt = tools.simulate_threshold(current)["cost_optimal_threshold"]
    except Exception:
        pass
    target = proposed if proposed is not None else (cost_opt if cost_opt is not None else current)
    sim_at = None
    try:
        sim_at = tools.simulate_threshold(target)
    except Exception:
        pass
    summary = f"위험 임계값 {current} → {round(target, 2)} 제안."
    if sim_at:
        summary += (f" 이 값에서 경보 {sim_at['alerts']}건 · 미탐 {sim_at['fn']} · 오탐 {sim_at['fp']}"
                    f" · 재현율 {sim_at['recall']}.")
    if cost_opt is not None:
        summary += f" 비용최적 ≈ {cost_opt}."
    return {
        "id": _new_id(), "kind": "set_threshold", "title": "위험 임계값 조정",
        "summary": summary, "current": current, "proposed": round(target, 2),
        "min": 0.05, "max": 0.95, "step": 0.01, "requires_override": False,
    }


def stop_proposal(sim, tick: dict) -> dict[str, Any]:
    return {
        "id": _new_id(), "kind": "stop_line", "title": "라인 정지 확인",
        "summary": f"현재 상태 '{tick['risk_level']}'(불량확률 {tick['probability']}). 라인을 정지하시겠습니까?",
        "current": None, "proposed": None, "requires_override": False,
    }


def start_proposal(sim, tick: dict) -> dict[str, Any]:
    """재시작 제안 + 인터록: 현재 위험이면 강제 override 요구."""
    danger = tick["risk_level"] == "위험"
    return {
        "id": _new_id(), "kind": "start_line", "title": "라인 재시작 확인",
        "summary": ("현재 '위험' 상태입니다. 이탈 변수가 해소되지 않았습니다. "
                    "강제로 재시작하려면 '오버라이드'를 확인하세요."
                    if danger else "라인을 시작(재개)하시겠습니까? 현재 상태는 정상 범위입니다."),
        "current": None, "proposed": None,
        "requires_override": danger,
        "interlock_reason": (", ".join(r["label"] for r in tick.get("top_reasons", [])) or "위험 상태")
                            if danger else None,
    }


def parse(text: str, sim, tick: dict) -> dict[str, Any] | None:
    """자연어 → 제안. 명령이 아니면 None."""
    t = text.strip()
    if any(w in t for w in THRESHOLD_WORDS):
        return threshold_proposal(sim, _extract_ratio(t))
    if any(w in t for w in STOP_WORDS):
        return stop_proposal(sim, tick)
    if any(w in t for w in START_WORDS):
        return start_proposal(sim, tick)
    return None
