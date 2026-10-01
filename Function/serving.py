"""대시보드가 부르는 함수들. **화면 쪽에는 계산 로직을 두지 않는다.**

왜 여기 있나
-----------
예전에는 `Dashboard/backend/app.py` 안에 모델 적재·예측·임계값 산정·비용 계산이 전부
들어 있었고, 전처리도 `Dashboard/pipeline/` 에 따로 있었다. 그래서 같은 로직이 두 벌이
됐다 — 모델링 쪽과 화면 쪽이 서로 다른 전처리를 쓰는 상태였다.

이 파일은 **순수 함수만** 둔다. FastAPI 를 모른다. 화면은 이걸 얇게 감싸기만 한다.
그래서 챗봇이든 배치든 같은 함수를 부르면 같은 답이 나온다.

자료와 모델은 어디서 오나
------------------------
  Dataset/preprocessed/processed.csv   `preprocessing.py` 가 만든 것 — 모델링과 같은 파일
  Model/model_<이름>.joblib            `modeling.py` 가 만든 것 — {estimator, features}

**열 이름은 전처리 결과의 것을 그대로 쓴다.** 화면에 나가는 JSON 만 기존 계약에 맞춘다
(`forecast_time`·`현재전력` 등) — 프런트엔드를 고치지 않기 위해서다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

import config as C
from optimization import StaffingOptimizer

HERE = Path(__file__).resolve().parent
_lock = Lock()
_state: dict = {}

# 전처리 결과의 열 → 화면 계약의 이름
RENAME = {"ts": "forecast_time", "kW": "현재전력", "y": "전력"}
CONTEXT = ["생산량", "기온", "풍속", "습도", "주중여부", "주중공휴일여부"]


def iso(v) -> str:
    return pd.Timestamp(v).strftime("%Y-%m-%dT%H:%M:%S")


def num(v) -> float:
    return round(float(v), 3)


def model_kind(model) -> str:
    if hasattr(model, "steps"):
        return type(model.steps[-1][1]).__name__
    return type(model).__name__


def load(model_path: Path | None = None, data_path: Path | None = None) -> dict:
    """모델과 재생용 자료를 올린다. 한 번만 부르면 된다."""
    model_path = Path(model_path or C.MODEL_DIR / "model_et.joblib")
    data_path = Path(data_path or C.OUT_DEFAULT)
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없다: {model_path}\n"
            f"  python Function/modeling.py --models et   로 만들 수 있다 (약 2분)")
    if not data_path.exists():
        raise FileNotFoundError(
            f"전처리 결과가 없다: {data_path}\n  python Function/preprocessing.py 를 먼저 돌려라")

    bundle = joblib.load(model_path)      # 우리가 만든 파일만 올린다
    if not isinstance(bundle, dict) or not {"estimator", "features"} <= bundle.keys():
        raise ValueError("모델 묶음에 estimator 와 features 가 있어야 한다")

    d = pd.read_csv(data_path, parse_dates=["ts"])
    missing = [c for c in list(bundle["features"]) + ["ts", "split", "kW", "y"]
               if c not in d.columns]
    if missing:
        raise ValueError(f"전처리 결과에 필요한 열이 없다: {missing}")

    f = d.rename(columns=RENAME).sort_values("forecast_time").reset_index(drop=True)
    f["target_time"] = f["forecast_time"] + pd.Timedelta(minutes=C.STEP_MIN)
    # 화면 계약의 달력 두 열을 만든다 — 전처리에는 is_weekend·is_off 로 들어 있다
    f["주중여부"] = (1 - f["is_weekend"]).astype(int)
    f["주중공휴일여부"] = (f["is_off"] & (f["is_weekend"] == 0)).astype(int)

    observed = f[["forecast_time", "현재전력"]].copy()   # 전체 구간 — 연간 피크 산정에 쓴다
    test = f.loc[f["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise ValueError("재생할 시험 구간 행이 없다")

    pred = np.asarray(bundle["estimator"].predict(test[bundle["features"]]), float)
    if pred.shape != (len(test),) or not np.isfinite(pred).all():
        raise ValueError("다음 15분 예측이 유한한 스칼라여야 한다")
    test["prediction"] = pred
    test["day_key"] = test["forecast_time"].dt.strftime("%Y-%m-%d")

    _state.update(
        bundle=bundle, frame=test, observed=observed,
        groups={k: g.reset_index(drop=True) for k, g in test.groupby("day_key")},
        optimizer=StaffingOptimizer(), model_path=model_path, data_path=data_path,
        version=hashlib.sha256(model_path.read_bytes()).hexdigest()[:10])
    return _state


def _day(day: str) -> pd.DataFrame:
    if day not in _state["groups"]:
        raise KeyError("해당 날짜에 재생할 데이터가 없습니다.")
    return _state["groups"][day]


def meta() -> dict:
    f = _state["frame"]
    p, t = f["prediction"].to_numpy(), f["전력"].to_numpy()
    days = list(_state["groups"])
    return {"days": days, "defaultDay": days[0],
            "model": _state["model_path"].name,
            "algorithm": type(_state["bundle"]["estimator"]).__name__,
            "modelType": model_kind(_state["bundle"]["estimator"]),
            "version": _state["version"], "features": _state["bundle"]["features"],
            "horizon": C.STEP_MIN, "rows": len(f),
            "mse": num(np.mean((p - t) ** 2)), "mae": num(np.mean(abs(p - t))),
            "unit": "원자료 값", "source": _state["data_path"].name}


def peak_recommendation(now) -> dict:
    """지금까지 **확정된** 것만 보고 경보 임계값을 정한다. 미래를 쓰지 않는다."""
    obs = _state["observed"]
    avail = obs.loc[(obs["forecast_time"].dt.year == now.year) & (obs["forecast_time"] <= now)]
    peak_row = avail.loc[avail["현재전력"].idxmax()]
    f = _state["frame"]
    done = f.loc[(f["target_time"] <= now) & (f["target_time"].dt.year == now.year)]
    if len(done) >= 30:
        margin = float(np.quantile(np.maximum(0, done["전력"] - done["prediction"]), .90))
        basis = "현재까지 확정된 예측 오차의 과소예측량 90백분위"
    else:
        margin = float(peak_row["현재전력"]) * .05
        basis = "관측 최대 전력의 5% (초기 참고 가정)"
    peak = float(peak_row["현재전력"])
    return {"value": round(max(1, peak - margin), 1), "annualPeak": num(peak),
            "annualPeakTime": iso(peak_row["forecast_time"]), "year": int(now.year),
            "coverageStart": iso(avail["forecast_time"].min()), "asOf": iso(now),
            "errorMargin": num(margin), "errorSamples": len(done), "basis": basis}


def snapshot(day: str, cursor: int = 48, threshold: float | None = None) -> dict:
    data = _day(day)
    cursor = min(cursor, len(data) - 1)
    row = data.iloc[cursor]
    now = row["forecast_time"]
    rec = peak_recommendation(now)
    threshold = rec["value"] if threshold is None else float(threshold)
    hist = data.iloc[:cursor + 1]

    points = [{"time": iso(r["forecast_time"]), "actual": num(r["현재전력"]), "predicted": None}
              for _, r in hist.iterrows()]
    by_time = {p["time"]: p for p in points}
    for _, r in hist.iterrows():
        k = iso(r["target_time"])
        if k in by_time:
            by_time[k]["predicted"] = num(r["prediction"])
    points.append({"time": iso(row["target_time"]), "actual": None,
                   "predicted": num(row["prediction"])})
    alerts = [{"time": iso(r["forecast_time"]), "targetTime": iso(r["target_time"]),
               "predicted": num(r["prediction"]), "key": iso(r["forecast_time"])}
              for _, r in hist.loc[hist["prediction"] >= threshold].tail(20).iloc[::-1].iterrows()]
    return {"day": day, "cursor": cursor, "count": len(data), "time": iso(now),
            "targetTime": iso(row["target_time"]), "current": num(row["현재전력"]),
            "prediction": num(row["prediction"]), "threshold": threshold,
            "margin": num(threshold - row["prediction"]),
            "atRisk": bool(row["prediction"] >= threshold), "recommendation": rec,
            "dailyPeak": num(hist["현재전력"].max()), "points": points, "alerts": alerts,
            "timeline": [iso(t) for t in data["forecast_time"]],
            "context": {k: num(row[k]) for k in CONTEXT}}


def scenario(day: str, cursor: int, production: float, staff: int, hourly_wage: float,
             energy_rate: float, power_factor: float, baseline_staff: int) -> dict:
    """생산량·인원을 바꿔 보고 비용을 비교한다.

    **주의 — 이 모델에서는 생산량을 바꿔도 예측이 움직이지 않는다.**
    `생산량` 이 입력 변수에 없기 때문이다. 가이드북 p57 이 생산량을 입력에서 빼
    최적화 변수로 돌리고, 실측으로도 넣어서 나아지지 않았다(전진검증 64.230 / 63.646 /
    63.409 로 1 표준오차 안의 동률). 그래서 **전력 예측은 그대로이고 비용만 달라진다.**
    응답의 `productionIsInput` 으로 그 사실을 알린다 — 화면이 조용히 속지 않게.

    생산량이 예측을 움직이게 하려면 `config.TREE_FEATURES` 에 `생산량_lag4` 를 넣고
    모델을 다시 학습해야 한다. 그러면 지금 수치는 더 이상 유효하지 않다.
    """
    data = _day(day)
    if cursor >= len(data):
        raise ValueError("재생 위치가 날짜 범위를 벗어났습니다.")
    row = data.iloc[[cursor]].copy()
    row["생산량"] = production
    with _lock:
        pred = float(_state["bundle"]["estimator"].predict(row[_state["bundle"]["features"]])[0])
    if not np.isfinite(pred):
        raise ValueError("모델이 유효한 값을 반환하지 않았습니다.")
    base = float(data.iloc[cursor]["prediction"])

    def costs(power, n):
        return {"energy": round(max(0, power) * power_factor * .25 * energy_rate),
                "labor": round(n * hourly_wage * .25),
                "total": round(max(0, power) * power_factor * .25 * energy_rate
                               + n * hourly_wage * .25)}

    f = _state["frame"]
    feats = _state["bundle"]["features"]
    prod_in = any("생산" in c for c in feats)
    return {"prediction": num(pred), "baselinePrediction": num(base),
            "baseline": costs(base, baseline_staff), "scenario": costs(pred, staff),
            "productionRange": {"min": num(f["생산량"].min()), "max": num(f["생산량"].max())},
            "productionIsInput": prod_in,
            "productionNote": (None if prod_in else
                               "이 모델은 생산량을 입력으로 쓰지 않습니다. "
                               "생산량을 바꿔도 전력 예측은 변하지 않고 비용만 달라집니다."),
            "assumption": ("전력 값 × kW 환산계수 × 0.25시간 × 전력량 단가 "
                           "+ 인원 × 시급 × 0.25시간. 기본요금 제외.")}


def staffing(day: str, cursor: int = 48, **over) -> dict:
    """정시 직전 45분 예측으로 그 시간의 최소 인원과 비용을 낸다."""
    data = _day(day)
    cursor = min(cursor, len(data) - 1)
    now = data.iloc[cursor]["forecast_time"]
    hour = (now + pd.Timedelta(minutes=C.STEP_MIN)).floor("h")
    issued = hour - pd.Timedelta(minutes=C.STEP_MIN)
    f = _state["frame"]
    hit = f.loc[(f["forecast_time"] == issued) & (f["target_time"] == hour)]
    if len(hit) != 1:
        raise ValueError("이 시간대의 정시 직전 45분 예측이 없습니다. 다음 시점으로 이동해주세요.")
    r = _state["optimizer"].hourly_interval(
        issued, hour, float(hit.iloc[0]["prediction"]),
        **{k: v for k, v in over.items() if v is not None})
    return {**r, "cursor": cursor, "asOf": iso(now)}
