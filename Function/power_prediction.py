"""최종 ExtraTrees·GRU 모델을 로드하고 대시보드용 전력 예측, 재생 자료와 경보를 제공한다."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import settings as C
from staffing_cost import StaffingOptimizer

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
    """ExtraTrees와 GRU를 로드하여 manifest의 비율로 결합한다.
    입력 행과 GRU 창은 학습과 동일한 규칙으로 선택한다."""
    import model_training as M

    model_path = Path(model_path or C.MODEL_DIR / "model_et.joblib")
    data_path = Path(data_path or C.OUT_DEFAULT)
    explicit_net = net_path is not None
    net_path = Path(net_path) if net_path else C.MODEL_DIR / "model_gru.pt"
    use_net = net_path.exists() and (model_path.name == "model_et.joblib" or explicit_net)
    if blend is not None:
        w = float(blend)
    else:
        man = model_path.parent / "manifest.json"
        w = (json.loads(man.read_text()).get("blend_weight") if man.exists() else None)
        w = C.BLEND_DEFAULT if w is None else float(w)
    if not np.isfinite(w) or not 0 <= w <= 1:
        raise ValueError("결합 가중치는 0과 1 사이의 유한한 값이어야 합니다")
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없다: {model_path}\n"
            f"  python Function/model_training.py --models et   로 만들 수 있다 (약 2분)")
    if not data_path.exists():
        raise FileNotFoundError(
            f"전처리 결과가 없다: {data_path}\n  python Function/data_preprocessing.py 를 먼저 돌려라")

    from model_validation import validate
    validate(model_path, net_path if use_net else None)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not {"estimator", "features"} <= bundle.keys():
        raise ValueError("모델 묶음에 estimator 와 features 가 있어야 한다")

    # 추론은 단일 스레드로 실행한다.
    _single_thread(bundle["estimator"])

    d = M.load(data_path)
    missing = [c for c in list(bundle["features"]) + ["ts", "split", "kW", "y"]
               if c not in d.columns]
    if missing:
        raise ValueError(f"전처리 결과에 필요한 열이 없다: {missing}")

    # 학습과 동일한 공통 평가 행을 선택한다.
    rows = d[M.usable(d)].sort_values("ts").reset_index(drop=True)
    test_pos = np.where(rows["split"].to_numpy() == "test")[0]
    if len(test_pos) == 0:
        raise ValueError("재생할 시험 구간 행이 없다")

    tree_pred = np.asarray(
        bundle["estimator"].predict(rows.iloc[test_pos][bundle["features"]]), float)
    if tree_pred.shape != (len(test_pos),) or not np.isfinite(tree_pred).all():
        raise ValueError("다음 15분 예측이 유한한 스칼라여야 한다")

    # OpenMP 종료 충돌을 피하도록 비용 계산기를 GRU 추론 전에 초기화한다.
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
    """macOS OpenMP 충돌 방지를 위해 GRU 추론을 별도 프로세스에서 실행한다."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "net.npz"
        r = subprocess.run(
            [sys.executable, "-B", str(HERE / "gru_prediction.py"),
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
    # 최종·구성 모델 성능과 결합 비율 — 학습 기록(results.json)을 그대로 보여 준다
    candidates, blend_search = None, None
    res_path = _state["model_path"].parent / "results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        names = {"ensemble": "결합 (ExtraTrees + GRU)", "et": "ExtraTrees", "gru": "GRU"}
        models = res.get("모델별", {})
        order = sorted(models, key=lambda k: models[k].get("전진검증_평균") or 9e9)
        candidates = [{"key": k, "name": names.get(k, k), "forward": models[k].get("전진검증_평균"),
                       "test": (models[k].get("시험") or {}).get("MSE"),
                       "inEnsemble": k in ("ensemble", "et", "gru")} for k in order]
        s = res.get("결합비율탐색")
        if s:
            blend_search = {"chosen": s["선정_트리비중"], "forward": s["선정_전진평균"],
                            "test": s["선정_시험"], "halfForward": s["고정0.5_전진평균"],
                            "halfTest": s["고정0.5_시험"], "select": s.get("선정기준"),
                            "forwardBest": s.get("전진검증최적_트리비중"),
                            "forwardBestTest": s.get("전진검증최적_시험"), "rule": s["기준"]}
    net_inputs = ([f"과거 {C.NET['window']}구간 전력({C.NET['window'] // C.PER_DAY}일)"]
                  + list(C.NET_CALENDAR) if _state["mode"] == "ensemble" else None)
    return {"days": days, "defaultDay": days[0], "model": name,
            "algorithm": algo, "modelType": algo,
            "netInputs": net_inputs, "candidates": candidates, "blendSearch": blend_search,
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
    """생산량·인원에 따른 비용을 비교한다.
    생산량은 모델 입력이 아니므로 조건을 바꿔도 모델 예측은 유지한다.
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


def staffing(day: str, cursor: int = 48, *, historical_replay: bool = False, **over) -> dict:
    """정시 직전 예측과 생산 목표로 참고 비용을 계산한다."""
    data = _day(day)
    cursor = min(cursor, len(data) - 1)
    now = data.iloc[cursor]["forecast_time"]
    hour = (now + pd.Timedelta(minutes=C.STEP_MIN)).floor("h")
    issued = hour - pd.Timedelta(minutes=C.STEP_MIN)
    f = _state["frame"]
    hit = f.loc[(f["forecast_time"] == issued) & (f["target_time"] == hour)]
    if len(hit) != 1:
        raise ValueError("이 시간대의 정시 직전 45분 예측이 없습니다. 다음 시점으로 이동해주세요.")
    optimizer = _state["optimizer"]
    calculate = optimizer.replay_hourly_interval if historical_replay else optimizer.hourly_interval
    r = calculate(
        issued, hour, float(hit.iloc[0]["prediction"]),
        **{k: v for k, v in over.items() if v is not None})
    return {**r, "cursor": cursor, "asOf": iso(now)}
