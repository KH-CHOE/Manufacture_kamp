"""OpenAI function-calling 루프 + 규칙기반 폴백.

OPENAI_API_KEY 가 있으면 실제 도구 호출 대화, 없으면 키워드 라우팅 폴백.
모델은 OPENAI_MODEL(기본 gpt-4o-mini)로 저비용 운영.
"""
from __future__ import annotations

import json
import os
from typing import Any

from . import tools

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM = (
    "당신은 주조 공정 품질 관제 대시보드(Foundry Guard)의 한국어 AI 어시스턴트입니다. "
    "주어진 도구로 현재 분석 결과를 조회·시뮬레이션·예측하고 현장 언어로 설명합니다.\n"
    "원칙:\n"
    "- 통계적 관계는 인과가 아님을 명시하고, 불확실하면 '확인 필요'라고 말합니다.\n"
    "- 수치는 도구 결과에 근거해서만 제시하고 지어내지 않습니다.\n"
    "- 모든 조치·스펙은 관리자 승인 전 '후보'입니다. 챗봇은 설비 제어·자동 알림 발송을 하지 않습니다.\n"
    "- 상태 구분은 정상/주의/위험 용어를 사용합니다.\n"
    "- 답변은 간결하게, 실행 가능한 형태(임계값·점검 대상·조치 후보)로 제시합니다."
)

# OpenAI function-calling 스키마
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_status", "description": "현재 성능지표·경보 현황·임계값을 요약한다.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "list_high_risk", "description": "불량확률이 임계값 이상인 고위험 이벤트 상위 N개.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "explain_event",
        "description": "특정 이벤트(미지정 시 최상위 고위험)의 원인=정상구간 이탈 변수를 반환.",
        "parameters": {"type": "object", "properties": {"event_time": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "simulate_threshold",
        "description": "위험 임계값 변경 시 경보수·오탐·미탐·총비용·정밀도·재현율과 비용최적 임계값을 계산.",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "number"}, "fp_unit": {"type": "number"}, "fn_unit": {"type": "number"}},
            "required": ["value"]}}},
    {"type": "function", "function": {
        "name": "predict",
        "description": "새 공정조건(values: 피쳐명→값)의 불량확률·판정·스펙이탈을 반환. 일부만 줘도 나머지는 중앙값으로 채움.",
        "parameters": {"type": "object", "properties": {"values": {"type": "object"}}, "required": ["values"]}}},
    {"type": "function", "function": {
        "name": "recommend_action", "description": "위험 이벤트의 권장 조치 후보(관리자 승인 전제)를 제시.",
        "parameters": {"type": "object", "properties": {"event_time": {"type": "string"}}}}},
]


def has_api() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _dispatch(name: str, args: dict) -> Any:
    fn = tools.REGISTRY.get(name)
    if fn is None:
        return {"error": f"알 수 없는 도구: {name}"}
    try:
        return fn(**(args or {}))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run_chat(message: str, history: list[dict] | None = None) -> dict[str, Any]:
    """{reply, tool_trace}. API 키 없으면 규칙기반 폴백."""
    if not has_api():
        return _fallback(message)

    from openai import OpenAI

    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})
    trace: list[dict] = []

    for _ in range(6):  # 도구 루프 상한
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA, temperature=0.2)
        choice = resp.choices[0].message
        if choice.tool_calls:
            messages.append({"role": "assistant", "content": choice.content,
                             "tool_calls": [tc.model_dump() for tc in choice.tool_calls]})
            for tc in choice.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = _dispatch(tc.function.name, args)
                trace.append({"name": tc.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
            continue
        return {"reply": choice.content or "", "tool_trace": trace}
    return {"reply": "도구 호출이 반복 한도를 초과했습니다.", "tool_trace": trace}


def _fallback(message: str) -> dict[str, Any]:
    """API 키 없을 때 키워드 라우팅 (데모/오프라인)."""
    trace = []

    def call(name, args=None):
        r = _dispatch(name, args or {})
        trace.append({"name": name, "args": args or {}, "result": r})
        return r

    if any(k in message for k in ["원인", "왜", "이유", "설명"]):
        r = call("explain_event")
        reasons = ", ".join(f"{o['label']} {o['value']}(정상 {o['low']}~{o['high']})"
                            for o in r.get("out_of_band", [])[:3])
        reply = f"[규칙기반] {r.get('time')} 이벤트(불량확률 {r.get('probability')}). 정상구간 이탈: {reasons or '없음'}."
    elif any(k in message for k in ["임계", "비용", "시뮬", "오탐", "미탐"]):
        r = call("simulate_threshold", {"value": tools.get_status()["threshold"]})
        reply = (f"[규칙기반] 임계값 {r.get('threshold')}: 경보 {r.get('alerts')}건, "
                 f"미탐 {r.get('fn')}·오탐 {r.get('fp')}, 총비용 {r.get('cost')}. 비용최적 ≈ {r.get('cost_optimal_threshold')}.")
    elif any(k in message for k in ["조치", "대응", "권장", "어떻게"]):
        r = call("recommend_action")
        reply = "[규칙기반] 권장 조치 후보: " + " / ".join(r.get("action_candidates", []))
    elif any(k in message for k in ["예측", "확률", "이 값", "이 조건"]):
        reply = "[규칙기반] 예측은 값을 지정해야 합니다. 예: '주입압력 700, 금형온도 620이면?' (OPENAI_API_KEY 설정 시 자연어 파싱)"
    else:
        r = call("get_status")
        reply = (f"[규칙기반] {r['title']} — PR-AUC {r['metrics'].get('prAuc')}, Recall {r['metrics'].get('recall')}. "
                 f"고위험 {r['high_risk_count']}건 / 평가 {r['evaluated_events']}건. "
                 f"(OPENAI_API_KEY 설정 시 자연어 대화·복합 도구호출 활성화)")
    return {"reply": reply, "tool_trace": trace}
