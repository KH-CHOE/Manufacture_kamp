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

from fastapi import FastAPI, HTTPException, Query
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
            spec.day, spec.cursor, unit_price=spec.unit_price, day_wage=spec.day_wage,
            energy_rate=spec.energy_rate,
            rate_table=spec.rate_table.model_dump() if spec.rate_table else None,
            base_rate=spec.base_rate, billing_peak=spec.billing_peak)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/optimization")
def default_staffing(day: str, cursor: int = Query(48, ge=0),
                     unit_price: float | None = Query(None, ge=0, le=1000000),
                     day_wage: float | None = Query(None, gt=0, le=1000000),
                     energy_rate: float | None = Query(None, ge=0, le=100000),
                     base_rate: float | None = Query(None, ge=0, le=1000000),
                     billing_peak: float | None = Query(None, ge=0, le=1000000)):
    return _staffing(StaffingSpec(day=day, cursor=cursor, unit_price=unit_price,
                                  day_wage=day_wage, energy_rate=energy_rate,
                                  base_rate=base_rate, billing_peak=billing_peak))


@app.post("/api/optimization")
def optimize_staffing(spec: StaffingSpec):
    return _staffing(spec)


_dist = ROOT / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
