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
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C
from optimization import StaffingOptimizer

HERE = Path(__file__).resolve().parent
_state: dict = {}

# 전처리 결과의 열 → 화면 계약의 이름
RENAME = {"ts": "forecast_time", "kW": "현재전력", "y": "전력"}
CONTEXT = ["생산량", "기온", "풍속", "습도", "주중여부", "주중공휴일여부"]


def iso(v) -> str:
    return pd.Timestamp(v).strftime("%Y-%m-%dT%H:%M:%S")


def num(v) -> float:
    return round(float(v), 3)


def _single_thread(model) -> None:
    """추정기(파이프라인 포함)의 `n_jobs` 를 1 로 내린다."""
    targets = [model]
    if hasattr(model, "steps"):
        targets += [m for _, m in model.steps]
    for m in targets:
        if hasattr(m, "n_jobs"):
            try:
                m.n_jobs = 1
            except Exception:
                pass


def model_kind(model) -> str:
    if hasattr(model, "steps"):
        return type(model.steps[-1][1]).__name__
    return type(model).__name__


def load(model_path: Path | None = None, data_path: Path | None = None,
         net_path: Path | None = None, blend: float | None = None) -> dict:
    """모델과 재생용 자료를 올린다. 한 번만 부르면 된다.

    `net_path` 를 주거나 `Model/model_gru.pt` 가 있으면 **앙상블로 서빙한다** —
    트리 예측과 순환신경망 시드평균 예측을 고정 반반으로 섞는다. 없으면 트리만 쓴다.

    행은 `modeling.usable()` 로 고른다. **모델링과 같은 규칙이어야** 순환신경망 창이
    학습 때와 같은 자리에 선다.
    """
    import modeling as M

    model_path = Path(model_path or C.MODEL_DIR / "model_et.joblib")
    data_path = Path(data_path or C.OUT_DEFAULT)
    explicit_net = net_path is not None
    net_path = Path(net_path) if net_path else C.MODEL_DIR / "model_gru.pt"
    use_net = net_path.exists() and (model_path.name == "model_et.joblib" or explicit_net)
    w = C.BLEND_WEIGHT if blend is None else float(blend)
    if not np.isfinite(w) or not 0 <= w <= 1:
        raise ValueError("결합 가중치는 0과 1 사이의 유한한 값이어야 합니다")
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없다: {model_path}\n"
            f"  python Function/modeling.py --models et   로 만들 수 있다 (약 2분)")
    if not data_path.exists():
        raise FileNotFoundError(
            f"전처리 결과가 없다: {data_path}\n  python Function/preprocessing.py 를 먼저 돌려라")

    from artifacts import validate
    validate(model_path, net_path if use_net else None)
    bundle = joblib.load(model_path)      # 우리가 만든 파일만 올린다
    if not isinstance(bundle, dict) or not {"estimator", "features"} <= bundle.keys():
        raise ValueError("모델 묶음에 estimator 와 features 가 있어야 한다")

    # **추론에서는 트리 병렬을 끈다.** joblib 의 작업자 풀(OpenMP)이 torch 와 같은
    # 프로세스에 있으면 종료 때 세그폴트(139)가 난다. 계산 결과는 같고 속도만 조금 준다.
    # 학습 때의 교착과는 다른 증상이지만 뿌리는 같다 — libomp 가 두 번 올라가는 것.
    _single_thread(bundle["estimator"])

    d = M.load(data_path)
    missing = [c for c in list(bundle["features"]) + ["ts", "split", "kW", "y"]
               if c not in d.columns]
    if missing:
        raise ValueError(f"전처리 결과에 필요한 열이 없다: {missing}")

    # **모델링과 같은 행 규칙.** 이게 어긋나면 순환신경망 창이 다른 자리에 선다
    rows = d[M.usable(d)].sort_values("ts").reset_index(drop=True)
    test_pos = np.where(rows["split"].to_numpy() == "test")[0]
    if len(test_pos) == 0:
        raise ValueError("재생할 시험 구간 행이 없다")

    tree_pred = np.asarray(
        bundle["estimator"].predict(rows.iloc[test_pos][bundle["features"]]), float)
    if tree_pred.shape != (len(test_pos),) or not np.isfinite(tree_pred).all():
        raise ValueError("다음 15분 예측이 유한한 스칼라여야 한다")

    # **최적화를 torch 보다 먼저 만든다.** 순서가 중요하다 —
    # 뒤에 만들면 프로세스가 끝날 때 세그폴트(종료 코드 139)가 난다.
    # sklearn·pandas 쪽과 torch 의 libomp 해제 순서가 엉키기 때문으로 보인다.
    # 계산 결과는 어느 쪽이든 같지만, 서버가 종료마다 죽는 것을 그냥 둘 수 없다.
    optimizer = StaffingOptimizer()

    net_info, net_pred, mode = None, None, "tree"
    if use_net:
        net_info, net_pred = _net_from_subprocess(net_path, data_path, test_pos)
        if net_pred is not None:
            mode = "ensemble"

    pred = tree_pred if net_pred is None else w * tree_pred + (1 - w) * net_pred

    f = rows.iloc[test_pos].rename(columns=RENAME).reset_index(drop=True)
    f["target_time"] = f["forecast_time"] + pd.Timedelta(minutes=C.STEP_MIN)
    f["주중여부"] = (1 - f["is_weekend"]).astype(int)
    f["주중공휴일여부"] = (f["is_off"] & (f["is_weekend"] == 0)).astype(int)
    f["prediction"] = pred
    f["tree_prediction"] = tree_pred
    if net_pred is not None:
        f["net_prediction"] = net_pred
    f["day_key"] = f["forecast_time"].dt.strftime("%Y-%m-%d")

    # 연간 피크 산정에는 시험 구간 밖의 관측도 필요하다
    obs = d.rename(columns=RENAME)[["forecast_time", "현재전력"]].copy()

    _state.update(
        bundle=bundle, net_info=net_info, mode=mode, blend_weight=w,
        frame=f, observed=obs,
        groups={k: g.reset_index(drop=True) for k, g in f.groupby("day_key")},
        optimizer=optimizer, model_path=model_path, data_path=data_path,
        net_path=net_path if net_info else None,
        version=hashlib.sha256(model_path.read_bytes() + (net_path.read_bytes() if net_info else b"")).hexdigest()[:10])
    return _state


def _net_from_subprocess(net_path: Path, data_path: Path,
                         test_pos: np.ndarray) -> tuple[dict | None, np.ndarray | None]:
    """순환신경망 예측을 **따로 띄운 프로세스**에서 받아 온다.

    왜 한 프로세스에서 안 하나 — sklearn 과 torch 가 같이 있으면 libomp 가 두 번 올라간다.
    계산은 맞게 나오지만 **프로세스가 끝날 때 세그폴트(139)** 가 난다. 실측으로 확인했다
    (`no Python frame` — 인터프리터 해제 시점). 서버가 내려갈 때마다 죽는 것을 둘 수 없다.
    학습에서 `modeling.py` 를 두 프로세스로 나눈 것과 같은 처방이다.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "net.npz"
        r = subprocess.run(
            [sys.executable, "-B", str(HERE / "net_infer.py"),
             "--data", str(data_path), "--model", str(net_path), "--out", str(out)],
            cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            msg = (r.stdout + r.stderr).strip().splitlines()
            raise ValueError("순환신경망 추론 실패: " + "\n".join(msg[-3:]))
        z = np.load(out)
        where, pred = z["where"], z["pred"]
        info = json.loads(str(z["info"])) if "info" in z.files else {}
        info["path"] = net_path.name

    if not np.array_equal(where, test_pos):
        raise ValueError("순환신경망과 트리 예측 행이 다릅니다")

    if pred.shape != (len(test_pos),) or not np.isfinite(pred).all():
        raise ValueError("순환신경망 예측값이 유효하지 않습니다")
    return info, pred


def _day(day: str) -> pd.DataFrame:
    if day not in _state["groups"]:
        raise KeyError("해당 날짜에 재생할 데이터가 없습니다.")
    return _state["groups"][day]


def meta() -> dict:
    f = _state["frame"]
    p, t = f["prediction"].to_numpy(), f["전력"].to_numpy()
    days = list(_state["groups"])
    tree_name = model_kind(_state["bundle"]["estimator"])
    if _state["mode"] == "ensemble":
        kind = str(_state["net_info"].get("kind") or "gru").upper()
        algo = f"{tree_name} + {kind} 앙상블"
        name = f"{_state['model_path'].name} + {_state['net_path'].name}"
    else:
        algo, name = tree_name, _state["model_path"].name
    return {"days": days, "defaultDay": days[0], "model": name,
            "algorithm": algo, "modelType": algo,
            "ensemble": _state["mode"] == "ensemble",
            "blendWeight": _state["blend_weight"],
            "seeds": (_state["net_info"].get("seeds") if _state["mode"] == "ensemble" else None),
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
            "context": {k: num(row["생산량_lag4"] if k == "생산량" else row[k]) for k in CONTEXT}}


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
    # 생산량은 현재 모델의 입력이 아니다. 화면과 같은 결합 예측을 유지한다.
    if any("생산" in c for c in _state["bundle"]["features"]):
        raise ValueError("생산량 입력 모델의 조건 변경 계산은 지원하지 않습니다")
    base = float(data.iloc[cursor]["prediction"])
    pred = base

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
    """정시 직전 예측과 사용자 계획 생산량으로 참고 비용을 계산한다."""
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
