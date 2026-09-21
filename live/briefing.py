"""주기 현황 보고 생성기 — 최근 이벤트 창을 자연어로 요약.

규칙기반이라 API 키 없이 항상 동작(발표 무대에서 견고). 키가 있으면 상위 계층이
이 통계를 LLM에 넘겨 더 자연스럽게 다듬을 수 있으나, 기본은 규칙기반으로 충분.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def summarize(window: list[dict], threshold: float, defect_label: int | None = None) -> dict[str, Any]:
    """window: [{index, probability, risk_level, actual, reasons:[label...]}] (오래된→최신)."""
    n = len(window)
    if n == 0:
        return {"text": "데이터 수집 중…", "processed": 0, "high_risk": 0,
                "defects": 0, "trend": "안정", "top_reason": None, "index": None}

    probs = [w["probability"] for w in window]
    high = sum(1 for p in probs if p >= threshold)
    defects = sum(1 for w in window if defect_label is not None and w["actual"] == defect_label)

    # 추세: 창을 반으로 나눠 평균 불량확률 비교
    half = max(1, n // 2)
    early = sum(probs[:half]) / half
    late = sum(probs[half:]) / max(1, n - half)
    delta = late - early
    trend = "상승" if delta > 0.03 else "하강" if delta < -0.03 else "안정"

    # 최다 이탈 변수
    reasons = Counter(r for w in window for r in w["reasons"])
    top_reason = reasons.most_common(1)[0][0] if reasons else None

    cur = window[-1]["risk_level"]
    rate = high / n
    parts = [f"최근 {n}건 처리", f"고위험 {high}건({rate:.0%})"]
    if defect_label is not None:
        parts.append(f"실측 불량 {defects}건")
    parts.append(f"현재 {cur}")
    tail = f"불량확률 {trend}"
    if top_reason:
        tail += f" · 최다 이탈 '{top_reason}'"
    text = " · ".join(parts) + ". " + tail + "."

    return {"text": text, "processed": n, "high_risk": high, "defects": defects,
            "rate": round(rate, 4), "trend": trend, "top_reason": top_reason,
            "index": window[-1]["index"]}
