"""화면이 쓰는 HTTP 껍데기. **계산은 여기 없다.**

예전 판은 이 파일 안에 모델 적재·예측·임계값 산정·비용 계산이 전부 들어 있었고,
전처리도 `Dashboard/pipeline/` 에 따로 있었다. 그래서 같은 로직이 두 벌이었다 —
모델링 쪽과 화면 쪽이 **서로 다른 전처리**를 쓰고 있었다.

지금은 전부 `Function/` 에 있다. 이 파일은 요청을 받아 그 함수를 부르고 결과를 돌려줄 뿐이다.
응답 모양은 예전과 같게 두었다 — 프런트엔드를 고치지 않기 위해서다.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]      # Dashboard_v2/
REPO = ROOT.parent                              # 저장소 뿌리
sys.path.insert(0, str(REPO / "Function"))      # 계산은 전부 저기서 가져온다

import serving as S                             # noqa: E402


@asynccontextmanager
async def lifespan(app):
    try:
        S.load(model_path=os.getenv("MODEL_PATH") or None,
               data_path=os.getenv("DATA_PATH") or None)
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Function/ 에서 모델·자료를 올리지 못했다.\n{e}") from e
    yield


app = FastAPI(title="KAMP power dashboard", lifespan=lifespan)


@app.get("/api/meta")
def meta():
    return S.meta()


@app.get("/api/snapshot")
def snapshot(day: str, cursor: int = Query(48, ge=0),
             threshold: float | None = Query(None, gt=0, le=10000)):
    try:
        return S.snapshot(day, cursor, threshold)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


class Scenario(BaseModel):
    day: str
    cursor: int = Field(ge=0)
    production: float = Field(ge=0, le=100000)
    staff: int = Field(ge=0, le=10000)
    hourly_wage: float = Field(ge=0, le=1000000)
    energy_rate: float = Field(ge=0, le=100000)
    power_factor: float = Field(gt=0, le=10000)
    baseline_staff: int = Field(ge=0, le=10000)


@app.post("/api/scenario")
def scenario(spec: Scenario):
    try:
        return S.scenario(**spec.model_dump())
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


class LoadRates(BaseModel):
    off: float = Field(ge=0, le=100000, allow_inf_nan=False)
    mid: float = Field(ge=0, le=100000, allow_inf_nan=False)
    peak: float = Field(ge=0, le=100000, allow_inf_nan=False)


class TariffRates(BaseModel):
    summer: LoadRates
    spring_autumn: LoadRates
    winter: LoadRates


class StaffingSpec(BaseModel):
    planned_production: float = Field(ge=0, le=1000000, allow_inf_nan=False)
    day: str
    cursor: int = Field(48, ge=0)
    unit_price: float | None = Field(None, ge=0, le=1000000, allow_inf_nan=False)
    day_wage: float | None = Field(None, gt=0, le=1000000, allow_inf_nan=False)
    energy_rate: float | None = Field(None, ge=0, le=100000, allow_inf_nan=False)
    rate_table: TariffRates | None = None
    base_rate: float | None = Field(None, ge=0, le=1000000, allow_inf_nan=False)
    billing_peak: float | None = Field(None, ge=0, le=1000000, allow_inf_nan=False)


def _staffing(spec: StaffingSpec):
    try:
        return S.staffing(
            spec.day, spec.cursor, planned_production=spec.planned_production, unit_price=spec.unit_price, day_wage=spec.day_wage,
            energy_rate=spec.energy_rate,
            rate_table=spec.rate_table.model_dump() if spec.rate_table else None,
            base_rate=spec.base_rate, billing_peak=spec.billing_peak)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/optimization")
def default_staffing(day: str, planned_production: float = Query(..., ge=0, le=1000000), cursor: int = Query(48, ge=0),
                     unit_price: float | None = Query(None, ge=0, le=1000000),
                     day_wage: float | None = Query(None, gt=0, le=1000000),
                     energy_rate: float | None = Query(None, ge=0, le=100000),
                     base_rate: float | None = Query(None, ge=0, le=1000000),
                     billing_peak: float | None = Query(None, ge=0, le=1000000)):
    return _staffing(StaffingSpec(day=day, planned_production=planned_production, cursor=cursor, unit_price=unit_price,
                                  day_wage=day_wage, energy_rate=energy_rate,
                                  base_rate=base_rate, billing_peak=billing_peak))


@app.post("/api/optimization")
def optimize_staffing(spec: StaffingSpec):
    return _staffing(spec)


# ── 브리핑·질의 (ChatGPT) ─────────────────────────────────────────
# API 키는 요청 헤더 X-OpenAI-Key 로만 받는다. 저장하지 않고 응답·오류 메시지에 담지 않는다.
class BriefingSpec(BaseModel):
    day: str
    cursor: int = Field(ge=0)
    threshold: float | None = Field(None, gt=0, le=10000)
    model: str | None = Field(None, max_length=60)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatSpec(BriefingSpec):
    messages: list[ChatMessage] = Field(min_length=1, max_length=24)


def _llm_error(e: Exception) -> HTTPException:
    # 라이브러리 예외 문구에 키 일부가 섞일 수 있어 종류만 알린다
    name = type(e).__name__
    if name == "AuthenticationError":
        return HTTPException(401, "API 키가 올바르지 않습니다. 키를 확인해주세요.")
    if name == "RateLimitError":
        return HTTPException(429, "요청 한도를 넘었거나 크레딧이 부족합니다. 잠시 뒤 다시 시도해주세요.")
    if name in ("NotFoundError", "BadRequestError"):
        return HTTPException(400, "모델 이름이나 요청 형식이 맞지 않습니다.")
    return HTTPException(502, "언어 모델 호출에 실패했습니다. 네트워크와 키를 확인해주세요.")


@app.post("/api/briefing")
def briefing(spec: BriefingSpec, x_openai_key: str | None = Header(None)):
    import assistant as A
    try:
        return A.briefing(spec.day, spec.cursor, spec.threshold, x_openai_key, spec.model)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise _llm_error(e) from None


@app.post("/api/chat")
def chat(spec: ChatSpec, x_openai_key: str | None = Header(None)):
    import assistant as A
    if not x_openai_key:
        raise HTTPException(401, "질문에 답하려면 OpenAI API 키가 필요합니다.")
    try:
        return A.chat(spec.day, spec.cursor, [m.model_dump() for m in spec.messages],
                      spec.threshold, x_openai_key, spec.model)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise _llm_error(e) from None


_dist = ROOT / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
