"""음성/텍스트 '질문' 답변 — 현재 시뮬레이터 상태를 근거로 (규칙기반, 무키·무료).

명령(임계조정·정지·재시작)이 아닌 물음("상태 어때", "왜 위험해")에 답한다.
답은 TTS로 읽어줄 것이라 간결하게. 기존 도구(simulate_threshold)만 재사용.
"""
from __future__ import annotations

from typing import Any

from chatbot import tools

CAUSE = ("원인", "왜", "이유", "위험", "설명", "무슨 문제", "뭐가 문제")
STATUS = ("상태", "현황", "어때", "요약", "괜찮", "지금")
THR = ("임계", "비용", "오탐", "미탐", "민감도")
COUNT = ("고위험", "몇 건", "몇건", "건수")


def answer(text: str, sim, latest_briefing: dict | None = None) -> str | None:
    t = text.strip()
    tick = sim.tick()
    p, risk = tick["probability"], tick["risk_level"]

    # COUNT 를 CAUSE 보다 먼저 — '고위험'이 '위험'(CAUSE)에 먹히지 않게
    if any(k in t for k in COUNT):
        return (f"누적 고위험 {tick.get('high_risk_total', 0)}건입니다. "
                f"현재 상태는 {risk}, 불량 확률 {p}.")

    if any(k in t for k in CAUSE):
        reasons = tick.get("top_reasons", [])
        if not reasons:
            return f"현재 상태는 {risk}, 불량 확률 {p}. 정상 범위를 벗어난 변수는 없습니다."
        parts = ", ".join(f"{r['label']} {r['value']}, 정상 범위 {r['normal'][0]}에서 {r['normal'][1]}"
                          for r in reasons[:3])
        return (f"현재 불량 확률 {p}로 {risk}입니다. 정상 범위를 벗어난 변수는 {parts}. "
                "통계적 관계이며 인과는 아닙니다.")

    if any(k in t for k in STATUS):
        if latest_briefing and latest_briefing.get("text"):
            return latest_briefing["text"]
        return (f"현재 상태는 {risk}, 불량 확률 {p}입니다. "
                f"누적 고위험 {tick.get('high_risk_total', 0)}건.")

    if any(k in t for k in THR):
        try:
            r = tools.simulate_threshold(sim.threshold)
            return (f"현재 임계값은 {sim.threshold}입니다. 이 값에서 경보 {r['alerts']}건, "
                    f"미탐 {r['fn']}건, 오탐 {r['fp']}건. 비용 최적 임계값은 약 {r['cost_optimal_threshold']}입니다. "
                    "'임계값 조정'이라고 말씀하시면 조정 창을 열어 드립니다.")
        except Exception:
            return f"현재 임계값은 {sim.threshold}입니다."

    return None
