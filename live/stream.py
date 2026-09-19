"""라이브 스트림 계층 — SSE(틱·브리핑·이상경보) + REST 제어.

SSE는 이름있는 이벤트로 구분: `tick`(매 프레임 상태), `briefing`(주기 현황보고),
`alert`(정상/주의→위험 전이 순간). 프론트는 addEventListener 로 각각 수신.

시뮬레이터는 지연 로딩 → 이 모듈 import 는 Vercel 콜드스타트에 무영향(데모는 로컬).
전송은 WebSocket 대신 SSE(단방향에 충분·curl 검증 가능), 명령은 POST.
안전 승인 계층(P4)은 이 제어 위에 얹는다.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from time import monotonic
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from live import briefing as brief
from live.simulator import Simulator

router = APIRouter(prefix="/live", tags=["live"])

_sim: Simulator | None = None
_subs: set[asyncio.Queue] = set()
_driver_task: asyncio.Task | None = None

# 최근 이벤트 창 / 이상 이력 / 최신 브리핑
_window: deque[dict] = deque(maxlen=80)
_alerts: list[dict] = []
_latest_briefing: dict | None = None
_last_risk: str = "정상"
_last_brief_t: float = 0.0
BRIEF_EVERY = 6.0  # 초(벽시계) 간격으로 현황보고 갱신


def get_sim() -> Simulator:
    global _sim
    if _sim is None:
        _sim = Simulator()
    return _sim


def _broadcast(event: str, data: Any) -> None:
    for q in list(_subs):
        try:
            q.put_nowait({"event": event, "data": data})
        except asyncio.QueueFull:
            pass


async def _driver() -> None:
    """재생 루프: running 이면 전진하며 tick 방송, 이상 전이·주기 브리핑 발생."""
    global _last_risk, _last_brief_t, _latest_briefing
    while True:
        sim = get_sim()
        stepped = False
        if sim.running:
            sim.step()
            interval = 1.0 / sim.speed
            stepped = True
        else:
            interval = 0.4
        tick = sim.tick()
        _broadcast("tick", tick)

        if stepped:
            _window.append({"index": tick["index"], "probability": tick["probability"],
                            "risk_level": tick["risk_level"], "actual": tick["actual"],
                            "reasons": [r["label"] for r in tick["top_reasons"]]})
            # 정상/주의 → 위험 전이 순간을 경보로
            if tick["risk_level"] == "위험" and _last_risk != "위험":
                alert = {"index": tick["index"], "time": tick["time"],
                         "probability": tick["probability"], "risk_level": tick["risk_level"],
                         "actual": tick["actual"], "reasons": tick["top_reasons"]}
                _alerts.insert(0, alert)
                del _alerts[20:]
                _broadcast("alert", alert)
            _last_risk = tick["risk_level"]
            # 주기 현황 보고
            now = monotonic()
            if now - _last_brief_t >= BRIEF_EVERY:
                _last_brief_t = now
                _latest_briefing = brief.summarize(list(_window), sim.threshold, sim.defect_label)
                _broadcast("briefing", _latest_briefing)

        await asyncio.sleep(interval)


def _ensure_driver() -> None:
    global _driver_task
    if _driver_task is None or _driver_task.done():
        _driver_task = asyncio.create_task(_driver())


def _fmt(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    _ensure_driver()
    q: asyncio.Queue = asyncio.Queue(maxsize=16)
    _subs.add(q)

    async def gen():
        try:
            yield _fmt("tick", get_sim().tick())          # 접속 즉시 현재 상태
            if _latest_briefing:
                yield _fmt("briefing", _latest_briefing)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=5.0)
                    yield _fmt(item["event"], item["data"])
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _subs.discard(q)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@router.get("/tick")
async def tick() -> dict:
    """현재 틱 1회 조회(스냅샷·디버그용)."""
    return get_sim().tick()


@router.get("/briefing")
async def briefing_ep() -> dict:
    return _latest_briefing or brief.summarize(list(_window), get_sim().threshold, get_sim().defect_label)


@router.get("/alerts")
async def alerts_ep() -> dict:
    return {"alerts": _alerts}


@router.post("/control")
async def control(cmd: dict) -> dict:
    """기본 재생 제어(P2). 상태변경 명령의 안전승인 래핑은 P4에서 추가."""
    global _last_risk, _latest_briefing
    sim = get_sim()
    action = cmd.get("action")
    if action == "start":
        sim.start()
    elif action == "stop":
        sim.stop()
    elif action == "speed":
        sim.set_speed(float(cmd.get("value", sim.speed)))
    elif action == "seek":
        sim.seek(int(cmd.get("index", sim.index)))
    elif action == "seek_defect":
        sim.seek_next_true_defect(int(cmd.get("lead", 8)))
    elif action == "seek_high_risk":
        sim.seek_next_high_risk(int(cmd.get("lead", 8)))
    elif action == "inject":
        sim.inject_anomaly(cmd.get("feature"))
    elif action == "clear_inject":
        sim.clear_injection()
    elif action == "reset":
        sim.stop(); sim.seek(0); sim.clear_injection()
        _window.clear(); _alerts.clear(); _latest_briefing = None; _last_risk = "정상"
    else:
        return {"ok": False, "error": f"알 수 없는 명령: {action}"}
    return {"ok": True, "tick": sim.tick()}


@router.get("/meta")
async def meta() -> dict:
    """프로파일·공정지점 레이아웃(프론트 초기화용)."""
    sim = get_sim()
    return {
        "profile": sim.profile.name,
        "threshold": sim.threshold,
        "total": sim.n,
        "high_risk_total": len(sim.high_risk_idx),
        "true_defect_total": len(sim.true_defect_idx),
        "stations": [{"id": s.id, "label": s.label, "stage": s.stage, "features": s.features}
                     for s in sim.profile.stations],
    }
