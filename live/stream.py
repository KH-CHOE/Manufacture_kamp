"""라이브 스트림 계층 — SSE(서버→브라우저 틱) + REST 제어.

시뮬레이터는 지연 로딩(첫 요청 때 모델·데이터 적재)이라, 이 모듈을 import 해도
Vercel 콜드스타트에 부담을 주지 않는다(무대 데모는 로컬 구동 전제).

전송 방식: WebSocket 대신 SSE — 단방향 틱 스트림에 충분하고 curl 로 검증 가능,
명령은 일반 POST. (P4 안전승인 계층은 이 제어 위에 얹는다.)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from live.simulator import Simulator

router = APIRouter(prefix="/live", tags=["live"])

_sim: Simulator | None = None
_subs: set[asyncio.Queue] = set()
_driver_task: asyncio.Task | None = None


def get_sim() -> Simulator:
    global _sim
    if _sim is None:
        _sim = Simulator()
    return _sim


async def _driver() -> None:
    """재생 루프: running 이면 한 칸 전진, 매 주기 현재 틱을 구독자에게 방송."""
    while True:
        sim = get_sim()
        if sim.running:
            sim.step()
            interval = 1.0 / sim.speed
        else:
            interval = 0.4
        tick = sim.tick()
        for q in list(_subs):
            try:
                q.put_nowait(tick)
            except asyncio.QueueFull:
                pass
        await asyncio.sleep(interval)


def _ensure_driver() -> None:
    global _driver_task
    if _driver_task is None or _driver_task.done():
        _driver_task = asyncio.create_task(_driver())


def _sse(payload: Any) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    _ensure_driver()
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    _subs.add(q)

    async def gen():
        try:
            yield _sse(get_sim().tick())  # 접속 즉시 현재 상태 1회
            while True:
                if await request.is_disconnected():
                    break
                try:
                    tick = await asyncio.wait_for(q.get(), timeout=5.0)
                    yield _sse(tick)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _subs.discard(q)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@router.get("/tick")
async def tick() -> dict:
    """현재 틱 1회 조회(스냅샷·디버그용, SSE 미연결)."""
    return get_sim().tick()


@router.post("/control")
async def control(cmd: dict) -> dict:
    """기본 재생 제어(P2). 상태변경 명령의 안전승인 래핑은 P4에서 추가."""
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
