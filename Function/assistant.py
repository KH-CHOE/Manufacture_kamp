"""전력 브리핑·질의 — ChatGPT 하네스.

화면이 보내는 재생 시점(day, cursor)에서 **그 시점까지 확정된 자료만** 모아 모델에 건넨다.
숫자는 전부 이 파일의 함수가 계산하고, 언어 모델은 그 숫자를 읽고 설명·권고만 한다.
그래서 언어 모델이 숫자를 지어낼 여지를 줄이고, 키가 없어도 숫자 요약은 나온다.

시점 규칙 (화면과 같다)
-----------------------
  재생 행의 `forecast_time` 이 t 이면 그 행의 현재 전력은 [t, t+15분) 구간이고 t+15분에 확정된다.
  예측은 다음 구간 [t+15분, t+30분) 이다. 브리핑은 **예측 구간이 끝나기 5분 전** 시각으로 표기한다.
  예) t = 10:45 → 11:00 까지 확정 · 11:00~11:15 예측 · "11:10 브리핑"
  재생 자료에 :10 시점 관측은 없다 — 본문에 '○○:○○까지 확정값' 을 함께 적는다.

API 키
------
  요청마다 받아 쓰고 저장·기록하지 않는다. 모델 이름은 환경변수 OPENAI_MODEL(기본 gpt-4o-mini).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import serving as S

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOOL_ROUNDS = 4


# ══ 하네스 — 시스템 지시문 ══════════════════════════════════════════
def _model_summary() -> str:
    res = json.loads((C.MODEL_DIR / "results.json").read_text(encoding="utf-8"))
    m, s = res["모델별"], res.get("결합비율탐색") or {}
    ens, et, gru, base = m["ensemble"], m["et"], m["gru"], m["persistence"]
    w = s.get("선정_트리비중", res["설정"].get("blend_weight"))
    return (f"- 다음 15분 전력 예측 = ExtraTrees(과거 전력·달력 입력, 기상 미사용) × {w} + "
            f"GRU(과거 96구간=하루 전력 + 달력) × {round(1 - w, 2)}. 비중은 시험 구간 기준으로 골랐다.\n"
            f"- 시험 구간({res['자료']['시험행']:,}개 구간) MSE {ens['시험']['MSE']} · "
            f"평균 절대오차 {ens['시험']['MAE']} — 직전값 유지 {base['시험']['MSE']} 대비 "
            f"{round((1 - ens['시험']['MSE'] / base['시험']['MSE']) * 100, 1)}% 낮다. "
            f"구성 모델 단독: ExtraTrees {et['시험']['MSE']} · GRU {gru['시험']['MSE']}.\n"
            "- 잘 맞는 때: 심야·비가동·부하가 안정적일 때. 어려운 때: 직전 15분 변화가 큰 급변, "
            "아침 시동(7~8시), 점심 전(11시), 교대 전후, 장기 휴무 직후.")


def system_prompt() -> str:
    return f"""너는 볼트·너트 제조공장의 전력 관리 보조원이다. 공장 관리자가 15분 단위 사용량의
최대치(피크)를 낮게 유지하도록 돕는다. 한국어로, 짧고 분명하게, 현장 담당자가 바로 움직일 수 있게 말한다.

## 공장과 자료
- 공정: 환봉·선재 입고 → 절단 → 성형(냉간·열간 단조) → 나사 가공(전조) → 필요 시 열처리·표면처리 → 검사 → 출하.
  전력은 가공설비 구동, 이송장치, 공기압축기, 냉각수 펌프, 환기설비, 전기가열에서 쓰인다.
- 계측 지점은 **공장 변압기의 총부하** 하나다. 어느 설비가 얼마나 쓰는지는 알 수 없다.
  그래서 "어느 설비를 꺼라"가 아니라 "언제 무엇을 미룰 수 있는가"를 후보로 제시한다.
- 자료는 2021년 1~9월, 15분 구간 전력(kW). 가동 시 약 100~200 kW, 비가동 시 20~30 kW 근처다.
  일요일 밤~월요일 아침은 쉬는 경우가 많고, 7월 말~8월 초 하계휴무가 있었다.
- 화면은 2021년 시험 구간을 **재생**한다. 실시간 계측이 아니다.

## 예측 모델
{_model_summary()}

## 관리 목표 — 15분 단위 사용량의 최대치
- **15분 동안의 평균 전력(15분 단위 사용량)의 최대치가 높게 찍히지 않도록 계속 유지하는 것**이 목표다.
  한 번 높게 찍힌 15분 최대치는 되돌릴 수 없으므로, 다음 15분이 알림 기준을 넘을 것 같으면 미리 부하를 나눈다.
- 요금 계산은 하지 않는다(요금 산정 방식은 계약 조건에 따라 복잡하다). 금액을 묻는 질문에는
  계산하지 않는다고 말하고, 15분 최대치를 낮게 유지하는 방법에 집중해 답한다.

## 관리 원칙
- 피크 저감 수단: 비필수 부하를 다음 15분 구간으로 미루기, 설비 시동을 15분씩 나눠 겹치지 않게 하기,
  공기압축기·전기가열처럼 미룰 수 있는 부하의 시작 시각 조정, 높은 부하가 겹치는 작업을 다른 시간대로 나누기.
- **설비를 자동으로 끄라는 지시는 하지 않는다.** 판단은 담당자가 한다. 안전·품질에 영향이 가는 조치는 권하지 않는다.
- 예측은 틀릴 수 있다. 최근 예측 오차를 함께 보고, 오차가 크거나 급변 중이면 여유를 두라고 말한다.
- 예측이 낮아도 **실측이 높으면 실측을 우선**한다.

## 숫자 규칙
- 숫자는 브리핑 자료나 도구 결과에 있는 값만 쓴다. 없는 값은 지어내지 말고 "자료 없음"이라고 말한다.
- 예측과 실측을 구분해 말한다. 시각은 자료에 적힌 시각을 그대로 쓴다.
- 질문이 자료 밖(설비별 사용량, 다른 공장, 미래 요금 등)이면 확인할 수 없다고 말하고, 할 수 있는 일을 제안한다.
"""


BRIEFING_FORMAT = """아래 자료로 **{label} 브리핑**을 써라. 형식은 정확히 네 줄 묶음이다.

**지금** — 확정된 현재 전력과 오늘 흐름 한두 문장
**15분 뒤** — 예측값, 알림 기준과의 차이, 위험 여부
**권장 관리** — 지금 할 일 1~3개(구체적으로, 담당자가 바로 할 수 있게)
**근거·한계** — 오늘 최대치·최근 예측 오차 등 근거 한 문장, 불확실성 한 문장

자료:
{facts}"""


# ══ 사실 모으기 — 재생 시점까지 확정된 것만 ═══════════════════════
def _frame(day: str) -> pd.DataFrame:
    return S._day(day)


def _cursor(day: str, cursor: int) -> int:
    return min(max(int(cursor), 0), len(_frame(day)) - 1)


def facts(day: str, cursor: int, threshold: float | None = None) -> dict:
    """브리핑에 쓰는 숫자 묶음. 재생 시점 이후 자료는 쓰지 않는다."""
    data = _frame(day)
    cursor = _cursor(day, cursor)
    row = data.iloc[cursor]
    t = pd.Timestamp(row["forecast_time"])
    confirmed = t + pd.Timedelta(minutes=C.STEP_MIN)            # 현재 구간이 끝나 확정되는 시각
    pred_start, pred_end = confirmed, confirmed + pd.Timedelta(minutes=C.STEP_MIN)
    label = pred_end - pd.Timedelta(minutes=5)
    snap = S.snapshot(day, cursor, threshold)
    rec = snap["recommendation"]

    # 오늘 지금까지 — 예측 대상 구간이 이미 끝난(확정된) 행만 오차를 잰다
    hist = data.iloc[:cursor + 1]
    done = hist[hist["target_time"] < confirmed]
    err = (done["전력"] - done["prediction"]).to_numpy(float)
    recent = data.iloc[max(0, cursor - 7):cursor + 1]

    pred = float(row["prediction"])
    return {
        "브리핑시각": label.strftime("%H:%M"),
        "날짜": day,
        "확정시각": confirmed.strftime("%H:%M"),
        "현재구간": f"{t:%H:%M}~{confirmed:%H:%M}",
        "현재전력_kW": round(float(row["현재전력"]), 1),
        "예측구간": f"{pred_start:%H:%M}~{pred_end:%H:%M}",
        "예측전력_kW": round(pred, 1),
        "알림기준_kW": round(float(snap["threshold"]), 1),
        "기준까지_여유_kW": round(float(snap["threshold"]) - pred, 1),
        "피크위험": bool(snap["atRisk"]),
        "알림기준_근거": rec["basis"],
        "올해_최대피크_kW": rec["annualPeak"],
        "오늘_최대_kW": round(float(hist["현재전력"].max()), 1),
        "최근2시간_실측_kW": [round(float(v), 1) for v in recent["현재전력"]],
        "직전15분_변화_kW": round(float(row["현재전력"] - data.iloc[cursor - 1]["현재전력"]), 1)
                         if cursor > 0 else None,
        "오늘_예측오차": ({"구간수": int(len(err)),
                        "평균절대오차_kW": round(float(np.mean(np.abs(err))), 2),
                        "최대과소예측_kW": round(float(max(0, np.max(err))), 1)}
                       if len(err) else "아직 확정된 예측 없음"),
        "오늘_최대_대비_예측_kW": round(pred - float(hist["현재전력"].max()), 1),
        "환경_한시간전": {k: snap["context"].get(k) for k in ("생산량", "기온", "풍속", "습도")},
    }


def rule_briefing(f: dict) -> str:
    """키가 없을 때 — 숫자만으로 만든 요약. 언어 모델을 부르지 않는다."""
    risk = "피크 위험" if f["피크위험"] else "안정"
    gap = f["기준까지_여유_kW"]
    err = f["오늘_예측오차"]
    err_s = (f"오늘 예측 평균 오차 {err['평균절대오차_kW']} kW" if isinstance(err, dict)
             else "오늘 확정된 예측이 아직 없음")
    return (f"**지금** — {f['확정시각']}까지 확정된 {f['현재구간']} 전력 {f['현재전력_kW']} kW "
            f"(오늘 최대 {f['오늘_최대_kW']} kW).\n"
            f"**15분 뒤** — {f['예측구간']} 예측 {f['예측전력_kW']} kW · 알림 기준 "
            f"{f['알림기준_kW']} kW 까지 {gap:+} kW · {risk}.\n"
            f"**권장 관리** — " + ("미룰 수 있는 부하의 시작을 다음 구간으로 넘기고 설비 시동을 겹치지 않게 한다."
                                   if f["피크위험"] else "지금 운전을 유지한다.") + "\n"
            f"**근거·한계** — 예측이 오늘 최대보다 {f['오늘_최대_대비_예측_kW']:+} kW. {err_s}. "
            "API 키를 넣으면 상황 설명과 권고가 자세해진다.")


# ══ 도구 — 질의응답에서 모델이 부른다 ══════════════════════════════
TOOLS = [
    {"type": "function", "function": {
        "name": "get_briefing_facts",
        "description": "재생 시점의 현재 전력·15분 뒤 예측·알림 기준·오늘 최대·오늘 예측 오차",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_today_profile",
        "description": "오늘 0시부터 재생 시점까지 시간별 실측 평균·최대(15분 단위)와 예측 오차",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_model_info",
        "description": "예측 모델 구성·비교 후보 성적·결합 비율·한계",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def _today_profile(day: str, cursor: int) -> dict:
    data = _frame(day)
    cursor = _cursor(day, cursor)
    hist = data.iloc[:cursor + 1].copy()
    confirmed = pd.Timestamp(hist.iloc[-1]["forecast_time"]) + pd.Timedelta(minutes=C.STEP_MIN)
    hist["시"] = pd.to_datetime(hist["forecast_time"]).dt.hour
    out = []
    for h, g in hist.groupby("시"):
        ok = g[g["target_time"] < confirmed]
        out.append({"시": int(h), "실측평균_kW": round(float(g["현재전력"].mean()), 1),
                    "실측최대_kW": round(float(g["현재전력"].max()), 1),
                    "예측평균절대오차_kW": (round(float((ok["전력"] - ok["prediction"]).abs().mean()), 2)
                                       if len(ok) else None)})
    return {"날짜": day, "확정시각": confirmed.strftime("%H:%M"), "시간별": out}


def _model_info() -> dict:
    res = json.loads((C.MODEL_DIR / "results.json").read_text(encoding="utf-8"))
    names = {"ensemble": "결합(ET 기상 미사용 + GRU)", "et": "ExtraTrees", "et_wx": "ExtraTrees + 기상",
             "rf": "RandomForest", "hgb": "HistGradientBoosting", "gru": "GRU", "lstm": "LSTM",
             "persistence": "직전값 유지", "tod_dow": "시각×요일 중앙값"}
    m = res["모델별"]
    s = res.get("결합비율탐색") or {}
    return {"후보": {names.get(k, k): {"전진검증_평균MSE": v["전진검증_평균"],
                                      "시험MSE": (v["시험"] or {}).get("MSE")} for k, v in m.items()},
            "결합_트리비중": s.get("선정_트리비중"), "결합비중_고르는_기준": s.get("기준"),
            "결합비중_주의": s.get("주의"),
            "행기준": res["자료"], "한계": [
                "자료가 2021년 1~9월뿐이라 겨울 피크를 보지 못했다",
                "계측이 변압기 총부하라 설비별 원인을 알 수 없다",
                "장기 휴무는 시험 구간에 한 번만 있었다",
                "기상은 결합 모델에 쓰지 않는다(기상 포함 ExtraTrees 는 비교 후보)"]}


def run_tool(name: str, args: dict, day: str, cursor: int, threshold: float | None) -> dict:
    if name == "get_briefing_facts":
        return facts(day, cursor, threshold)
    if name == "get_today_profile":
        return _today_profile(day, cursor)
    if name == "get_model_info":
        return _model_info()
    return {"오류": f"모르는 도구: {name}"}


# ══ 언어 모델 호출 ══════════════════════════════════════════════════
def _client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, timeout=60)


def briefing(day: str, cursor: int, threshold: float | None = None,
             api_key: str | None = None, model: str | None = None) -> dict:
    f = facts(day, cursor, threshold)
    base = {"label": f["브리핑시각"], "confirmedAt": f["확정시각"], "facts": f}
    if not api_key:
        return {**base, "source": "rule", "text": rule_briefing(f)}
    msg = BRIEFING_FORMAT.format(label=f["브리핑시각"],
                                 facts=json.dumps(f, ensure_ascii=False, indent=1))
    out = _client(api_key).chat.completions.create(
        model=model or DEFAULT_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": system_prompt()},
                  {"role": "user", "content": msg}])
    return {**base, "source": "llm", "model": out.model,
            "text": out.choices[0].message.content or ""}


def chat(day: str, cursor: int, messages: list[dict], threshold: float | None = None,
         api_key: str | None = None, model: str | None = None) -> dict:
    """질의응답. 도구로 재생 시점 자료를 조회하며 답한다."""
    if not api_key:
        raise PermissionError("질문에 답하려면 OpenAI API 키가 필요합니다")
    f = facts(day, cursor, threshold)
    convo = [{"role": "system", "content": system_prompt() +
              f"\n## 지금 재생 시점\n{f['날짜']} {f['확정시각']}까지 확정 · 다음 예측 {f['예측구간']}. "
              "숫자가 필요하면 도구를 불러라."}]
    convo += [{"role": m["role"], "content": str(m["content"])[:4000]}
              for m in messages[-12:] if m.get("role") in ("user", "assistant")]
    client = _client(api_key)
    used = []
    for _ in range(MAX_TOOL_ROUNDS):
        out = client.chat.completions.create(model=model or DEFAULT_MODEL, temperature=0.3,
                                             messages=convo, tools=TOOLS)
        msg = out.choices[0].message
        if not msg.tool_calls:
            return {"text": msg.content or "", "tools": used, "model": out.model,
                    "confirmedAt": f["확정시각"]}
        convo.append({"role": "assistant", "content": msg.content or "",
                      "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = run_tool(tc.function.name, args, day, cursor, threshold)
            except Exception as e:                      # 도구 오류도 모델에 알려 답하게 한다
                result = {"오류": str(e)}
            used.append(tc.function.name)
            convo.append({"role": "tool", "tool_call_id": tc.id,
                          "content": json.dumps(result, ensure_ascii=False, default=str)})
    return {"text": "도구 호출이 너무 길어져 답을 마치지 못했습니다. 질문을 좁혀 주세요.",
            "tools": used, "confirmedAt": f["확정시각"]}
