"""전처리 결과 하나로 모든 모델을 **같은 행에서** 비교한다.

왜 이렇게 나눴나
----------------
`preprocessing.py` 가 만든 CSV 한 장을 모든 모델이 읽는다. 트리는 열을 골라 쓰고
순환신경망은 같은 파일에서 창을 만든다. 평가행은 같지만 입력 변수·학습 기간·학습 예산은 다르므로 알고리즘 자체의 우열을 뜻하지 않는다.

비교 대상
  기준선     직전값 유지 · 시각×요일 중앙값          (학습 없음)
  트리 계열   RandomForest · ExtraTrees · HistGradientBoosting
  순환 계열   GRU · LSTM                            (시드 3개 예측 평균)
  결합       ExtraTrees + GRU 고정 반반

같은 행을 쓰게 만드는 규칙
-------------------------
어느 한 모델이라도 쓸 수 없는 행은 **전부에서 뺀다.** 그래야 수치를 나란히 놓을 수 있다.
  ① `TREE_FEATURES + NET_CALENDAR + kW + y` 가 전부 있는 행
  ② 순환신경망 창(96칸)이 **시각 공백을 넘지 않는** 행
행 기준은 `results.json` 에 적는다 — 다른 기준의 수치와 섞지 말라는 뜻이다.

전진검증 구간도 날짜를 박지 않는다
---------------------------------
학습 구간의 **마지막 달 시작일 네 개**를 자료에서 뽑아 각 24일로 만든다.
2021 자료에 분할 경계 20210725 를 주면 04/01·05/01·06/01·07/01 이 나온다.

macOS libomp — 프로세스를 나눠야 한다
------------------------------------
sklearn(joblib 포크·OpenMP)을 먼저 돌린 뒤 같은 프로세스에서 torch Adam 이 도는 순간
libomp 가 두 번 적재되어 교착한다. 환경변수로는 막히지 않는다.
그래서 이 파일은 **자기를 하위 프로세스로 두 번 호출**한다 — `--only nets`, `--only trees`.
직접 쓸 때는 `--only` 없이 부르면 된다.

실행
----
  python modeling.py --data processed.csv
  python modeling.py --data processed.csv --models baseline,rf,et,hgb      (트리만, 빠르다)
  python modeling.py --data processed.csv --models baseline,et,gru,ensemble
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

HERE = Path(__file__).resolve().parent

TREE_MODELS = {"rf": "RandomForest", "et": "ExtraTrees", "hgb": "HistGradientBoosting"}
NET_MODELS = {"gru": "GRU", "lstm": "LSTM"}
BASE_MODELS = {"persistence": "직전값 유지", "tod_dow": "시각×요일 중앙값"}
ALL_MODELS = ["baseline", *TREE_MODELS, *NET_MODELS, "ensemble"]


# ══ 공통 ════════════════════════════════════════════════════════════
def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["ts"])
    need = ["ts", "date_key", "split", "kW", "y"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise SystemExit(f"✗ 전처리 결과에 필수 열이 없다: {missing}")
    return d.sort_values("ts").reset_index(drop=True)


def window_ok(d: pd.DataFrame, win: int) -> np.ndarray:
    """창이 시각 공백을 넘지 않는 행만 True. 넘으면 '15분 전' 이 실제로는 며칠 전이다."""
    ok = np.zeros(len(d), bool)
    if len(d) < win:
        return ok
    ts = pd.DatetimeIndex(d["ts"])
    if ts.hasnans or not ts.is_unique or not ts.is_monotonic_increasing:
        raise ValueError("시각은 결측·중복 없이 오름차순이어야 합니다")
    step = pd.Timedelta(minutes=C.STEP_MIN)
    edges = np.r_[True, np.diff(ts.asi8) == step.value]
    finite = np.isfinite(d["kW"].to_numpy(float))
    from numpy.lib.stride_tricks import sliding_window_view
    ok[win - 1:] = (sliding_window_view(edges, win)[:, 1:].all(axis=1)
                       & sliding_window_view(finite, win).all(axis=1))
    return ok


def net_windows(d: pd.DataFrame, mask: np.ndarray, win: int, steps: int):
    """원래 관측축에서 창을 만들고 공통 평가행의 끝점만 선택한다."""
    from numpy.lib.stride_tricks import sliding_window_view
    endpoints = np.flatnonzero(mask)
    if not window_ok(d, win)[endpoints].all():
        raise ValueError("시각 공백 또는 결측을 포함하는 신경망 입력 창")
    if len(endpoints) == 0:
        return np.empty((0, steps, win // steps), np.float32)
    windows = sliding_window_view(d["kW"].to_numpy(np.float32), win)
    return windows[endpoints - win + 1].reshape(-1, steps, win // steps)


def usable(d: pd.DataFrame) -> np.ndarray:
    """모든 모델이 공유할 행. 하나라도 못 쓰는 행은 전부에서 뺀다."""
    cols = [c for c in (C.TREE_FEATURES + C.NET_CALENDAR + ["kW", "y"]) if c in d.columns]
    miss = [c for c in (C.TREE_FEATURES + C.NET_CALENDAR) if c not in d.columns]
    if miss:
        raise SystemExit(f"✗ 모델 입력 열이 전처리 결과에 없다: {miss}\n"
                         f"  preprocessing.py 를 다시 돌렸는지 확인해라")
    return np.isfinite(d[cols].to_numpy(float)).all(axis=1) & window_ok(d, C.NET["window"])


def training_stats(X, Y, training_indices):
    """뒤 15%는 조기종료 검증용. 표준화에는 앞 85%만 사용한다."""
    cut = int(len(training_indices) * .85)
    fit = training_indices[:cut]
    if not len(fit) or cut == len(training_indices):
        raise ValueError("신경망 학습/조기종료 검증 자료가 부족합니다")
    return (cut, X[fit].mean(), X[fit].std() + 1e-8,
            Y[fit].mean(), Y[fit].std() + 1e-8)


def folds_from_data(d: pd.DataFrame, n: int, days: int) -> list[list[int]]:
    """전진검증 구간을 **자료에서** 만든다. 날짜를 코드에 박지 않는다.

    학습 구간에 들어 있는 '달의 1일' 가운데 뒤쪽 n 개를 고르고 각 days 일로 끊는다.
    """
    tr = d[d["split"] == "train"]
    if tr.empty:
        return []
    ts = pd.DatetimeIndex(tr["ts"])
    firsts = sorted({pd.Timestamp(year=t.year, month=t.month, day=1) for t in ts})
    end = ts.max()
    # 구간이 학습 구간 안에서 끝나야 한다
    cand = [f for f in firsts if f + pd.Timedelta(days=days) <= end + pd.Timedelta(days=1)]
    out = []
    for f in cand[-n:]:
        lo = int(f.strftime("%Y%m%d"))
        hi = int((f + pd.Timedelta(days=days)).strftime("%Y%m%d"))
        out.append([lo, hi])
    return out


def score(y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    ss = float(((y - y.mean()) ** 2).sum())
    return {"MSE": round(float((e ** 2).mean()), 4),
            "RMSE": round(float(np.sqrt((e ** 2).mean())), 4),
            "MAE": round(float(np.abs(e).mean()), 4),
            "R2": round(float(1 - (e ** 2).sum() / ss), 6) if ss else None,
            "bias": round(float(e.mean()), 4)}


def jobs(d: pd.DataFrame, folds: list[list[int]]) -> list[tuple]:
    """(평가 시작, 평가 끝, 이름) — 전진검증 구간들 + 시험 구간."""
    split_day = int(d.loc[d["split"] == "test", "date_key"].min())
    return [(lo, hi, str(lo)) for lo, hi in folds] + [(split_day, None, "test")]


# ══ 기준선 ══════════════════════════════════════════════════════════
def run_baselines(d: pd.DataFrame, folds: list[list[int]]) -> dict:
    day = d["date_key"].to_numpy()
    out: dict = {k: {} for k in BASE_MODELS}
    for split, hi, label in jobs(d, folds):
        te_m = (day >= split) if hi is None else ((day >= split) & (day < hi))
        tr, te = d[day < split], d[te_m]
        if len(tr) < 1000 or te.empty:
            continue
        y = te["y"].to_numpy()
        out["persistence"][label] = {**score(y, te["kw_lag1"].to_numpy()),
                                     "rows": int(len(te))}
        # 시각×요일 중앙값 — 기준값은 **학습 구간에서만** 만든다
        key = ["시간", "15분위치", "dow"]
        med = tr.groupby(key)["y"].median()
        v = te.set_index(key).index.map(med).to_numpy(float)
        v = np.where(np.isnan(v), float(tr["y"].median()), v)
        out["tod_dow"][label] = {**score(y, v), "rows": int(len(te))}
    return out


# ══ 트리 계열 (sklearn 전용 프로세스) ═══════════════════════════════
def run_trees(d: pd.DataFrame, folds: list[list[int]], want: list[str]) -> dict:
    from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import joblib

    def make(kind: str):
        if kind == "rf":
            return RandomForestRegressor(**C.RF)
        if kind == "et":
            return ExtraTreesRegressor(**C.TREE)
        return HistGradientBoostingRegressor(**C.BOOST)

    day = d["date_key"].to_numpy()
    res: dict = {k: {} for k in want}
    preds: dict = {k: {} for k in want}
    for split, hi, label in jobs(d, folds):
        te_m = (day >= split) if hi is None else ((day >= split) & (day < hi))
        tr, te = d[day < split], d[te_m]
        if len(tr) < 1000 or te.empty:
            continue
        y = te["y"].to_numpy()
        for kind in want:
            p = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("model", make(kind))])
            p.fit(tr[C.TREE_FEATURES], tr["y"])
            pr = p.predict(te[C.TREE_FEATURES])
            res[kind][label] = {**score(y, pr), "rows": int(len(te))}
            preds[kind][label] = pr
            if label == "test":
                joblib.dump({"estimator": p, "features": C.TREE_FEATURES},
                            C.MODEL_DIR / f"model_{kind}.joblib", compress=3)
            print(f"  {TREE_MODELS[kind]:22s} {label:>8} MSE {res[kind][label]['MSE']:9.3f}",
                  flush=True)
    np.savez(HERE / "_pred_trees.npz",
             **{f"{k}|{l}": v for k, d2 in preds.items() for l, v in d2.items()})
    return res


# ══ 순환 계열 (torch 전용 프로세스) ═════════════════════════════════
def run_nets(original: pd.DataFrame, folds: list[list[int]], want: list[str],
             data_path: Path, job_spec=None, cache_dir: Path | None = None) -> dict:
    import torch
    from torch import nn
    from numpy.lib.stride_tricks import sliding_window_view
    torch.set_num_threads(1)
    g = C.NET

    class Net(nn.Module):
        def __init__(self, n_ch: int, n_cal: int, hidden: int, kind: str):
            super().__init__()
            rnn = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = rnn(n_ch, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + n_cal, 64), nn.ReLU(),
                                      nn.Linear(64, 1))

        def forward(self, x, c):
            o, _ = self.rnn(x)
            return self.head(torch.cat([o[:, -1, :], c], 1)).squeeze(-1)

    win, steps = g["window"], g["steps"]
    mask = usable(original)
    X = net_windows(original, mask, win, steps)
    d = original[mask].reset_index(drop=True)
    idx = np.arange(len(d))
    Y = d["y"].to_numpy(np.float32)[idx]
    CAL = d[C.NET_CALENDAR].to_numpy(np.float32)[idx]
    day = d["date_key"].to_numpy()[idx]

    def train_one(kind, start, split_day, hi, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        tr_m = (day >= start) & (day < split_day)
        te_m = (day >= split_day) if hi is None else ((day >= split_day) & (day < hi))
        ti = np.where(tr_m)[0]
        cut, mu, sd, ymu, ysd = training_stats(X, Y, ti)
        fit = ti[:cut]
        Xn = ((X - mu) / sd).astype(np.float32)
        Yn = ((Y - ymu) / ysd).astype(np.float32)
        print(f"  시작 {kind} {split_day} seed={seed} fit={len(fit)} val={len(ti)-cut}", flush=True)
        xt = torch.from_numpy(np.ascontiguousarray(Xn[ti[:cut]]))
        yt = torch.from_numpy(Yn[ti[:cut]])
        ct = torch.from_numpy(CAL[ti[:cut]])
        net = Net(Xn.shape[2], CAL.shape[1], g["hidden"], kind)
        opt = torch.optim.Adam(net.parameters(), lr=g["lr"])
        lossf = nn.MSELoss()

        def pred(sel):
            net.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(sel), 512):
                    s = sel[i:i + 512]
                    out.append(net(torch.from_numpy(np.ascontiguousarray(Xn[s])),
                                   torch.from_numpy(CAL[s])).numpy())
            return np.concatenate(out) * ysd + ymu

        va = ti[cut:]
        best, state, bad, ep = float("inf"), None, 0, 0
        for ep in range(g["epochs"]):
            net.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), g["batch"]):
                k = perm[i:i + g["batch"]]
                opt.zero_grad(); lossf(net(xt[k], ct[k]), yt[k]).backward(); opt.step()
            vl = float(np.mean((pred(va) - Y[va]) ** 2))
            if (ep + 1) % 20 == 0:
                print(f"    {kind} {split_day} seed={seed} epoch={ep+1} val_MSE={vl:.3f}", flush=True)
            if vl < best - 1e-4:
                best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= g["patience"]:
                    break
        net.load_state_dict(state)
        ei = np.where(te_m)[0]
        # **표준화 통계를 함께 돌려준다.** 이것을 저장하지 않아 `.pt` 만으로는 보고 예측을
        # 되살릴 수 없었다. 추론 때 다시 계산하면 안 된다 — 학습 구간에서 잰 값이어야 한다.
        stats = {"mu": float(mu), "sd": float(sd), "ymu": float(ymu), "ysd": float(ysd),
                 "train_start": int(start), "train_end": int(split_day),
                 "fit_rows": len(fit), "validation_rows": len(va),
                 "fit_last_time": str(d.iloc[fit[-1]]["ts"]),
                 "validation_first_time": str(d.iloc[va[0]]["ts"])}
        return net, ei, pred(ei), ep + 1, stats

    # 각 시드 결과를 별도 파일로 보존한다. 중단 후에도 완료한 학습은 재사용한다.
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    if job_spec is not None:
        kind, split, hi, seed = job_spec
        start = int((pd.Timestamp(str(split)) - pd.DateOffset(months=g["train_months"]))
                    .strftime("%Y%m%d")) if g["train_months"] else 0
        net, ei, pred, ep, stats = train_one(kind, start, split, hi, seed)
        destination = cache_dir / f"{kind}_{split}_{seed}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save({"state": net.state_dict(), "ei": ei, "pred": pred,
                    "epochs": ep, "stats": stats}, temporary)
        temporary.replace(destination)
        return {}
    fingerprint = hashlib.sha256(data_path.read_bytes() + Path(__file__).read_bytes()
        + json.dumps({"net": g, "seeds": C.SEEDS, "calendar": C.NET_CALENDAR,
                      "features": C.TREE_FEATURES, "torch": torch.__version__,
                      "numpy": np.__version__}, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = HERE / "_net_cache" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(kind, split, hi, seed) for split, hi, _ in jobs(d, folds)
             if (day < split).sum() >= 1500 for kind in want for seed in C.SEEDS]

    def launch(spec):
        kind, split, _, seed = spec
        destination = cache_dir / f"{kind}_{split}_{seed}.pt"
        if destination.exists():
            print(f"  완료 학습 재사용 {kind} {split} seed={seed}", flush=True)
            return
        cmd = [sys.executable, "-B", str(HERE / "modeling.py"), "--data", str(data_path.resolve()),
               "--only", "nets", "--models", kind, "--net-job", json.dumps(spec),
               "--cache-dir", str(cache_dir)]
        subprocess.run(cmd, check=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(launch, tasks))

    res: dict = {k: {} for k in want}
    preds: dict = {k: {} for k in want}
    for split, hi, label in jobs(d, folds):
        if (day < split).sum() < 1500:
            continue
        start = int((pd.Timestamp(str(split)) - pd.DateOffset(months=g["train_months"]))
                    .strftime("%Y%m%d")) if g["train_months"] else 0
        for kind in want:
            ps, eps, nets, stats = [], [], [], None
            for sd_ in C.SEEDS:
                cached = torch.load(cache_dir / f"{kind}_{split}_{sd_}.pt", map_location="cpu", weights_only=False)
                net = Net(X.shape[2], CAL.shape[1], g["hidden"], kind)
                net.load_state_dict(cached["state"])
                ei, p, ep, st = cached["ei"], cached["pred"], cached["epochs"], cached["stats"]
                ps.append(p); eps.append(ep); nets.append(net)
                stats = st            # 시드와 무관하다(학습 구간만으로 정해진다)
            # **시드별 예측을 평균한다.** 시험 성적으로 시드를 고르지 않는다
            mean_p = np.mean(ps, axis=0)
            res[kind][label] = {**score(Y[ei], mean_p), "rows": int(len(ei)),
                                "epochs": eps, "seeds": C.SEEDS}
            preds[kind][label] = mean_p
            preds[f"{kind}|idx|{label}"] = ei
            if label == "test":
                import torch as T
                # **시드 전부와 표준화 통계를 담는다.** 보고 예측이 시드 3개 평균이므로
                # 첫 시드만 저장하면 그 수치를 되살릴 수 없다. 예전 판이 그랬다.
                T.save({"kind": kind, "config": g, "calendar": list(C.NET_CALENDAR),
                        "seeds": list(C.SEEDS),
                        "state_dicts": [{k: v.cpu() for k, v in n.state_dict().items()}
                                        for n in nets],
                        "stats": stats, "n_cal": len(C.NET_CALENDAR),
                        "n_ch": g["window"] // g["steps"],
                        "추론": ("시드별로 예측한 뒤 평균한다. 표준화는 stats 의 값을 쓰고 "
                               "다시 계산하지 않는다 — 학습 구간에서 잰 값이어야 한다")},
                       C.MODEL_DIR / f"model_{kind}.pt")
            print(f"  {NET_MODELS[kind]:22s} {label:>8} MSE {res[kind][label]['MSE']:9.3f}"
                  f" · 에폭 {eps}", flush=True)
    flat = {}
    for k, v in preds.items():
        if isinstance(v, dict):
            for l, arr in v.items():
                flat[f"{k}|{l}"] = arr
        else:
            flat[k] = v
    flat["row_index"] = idx
    np.savez(HERE / "_pred_nets.npz", **flat)
    return res


# ══ 실행 ════════════════════════════════════════════════════════════
def summarize(res: dict, folds: list[list[int]]) -> dict:
    """전진검증 평균과 시험 성적을 모델별로 모은다."""
    fl = [str(lo) for lo, _ in folds]
    out = {}
    for name, per in res.items():
        fwd = [per[f]["MSE"] for f in fl if f in per]
        out[name] = {
            "전진검증_구간별": {f: per[f]["MSE"] for f in fl if f in per},
            "전진검증_평균": round(float(np.mean(fwd)), 3) if fwd else None,
            "전진검증_표준편차": round(float(np.std(fwd)), 3) if fwd else None,
            "시험": per.get("test"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=C.OUT_DEFAULT)
    ap.add_argument("--out", type=Path, default=C.MODEL_DIR / "results.json")
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help=f"쉼표로 구분. 가능: {','.join(ALL_MODELS)}")
    ap.add_argument("--only", choices=["trees", "nets"], default=None,
                    help="내부용 — 오케스트레이터가 하위 프로세스로 호출할 때 쓴다")
    ap.add_argument("--net-job", type=json.loads, help=argparse.SUPPRESS)
    ap.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not a.data.exists():
        print(f"✗ 전처리 결과가 없다: {a.data}\n  preprocessing.py 를 먼저 돌려라")
        return 1
    want = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [m for m in want if m not in ALL_MODELS]
    if bad:
        print(f"✗ 모르는 모델: {bad}\n  가능: {ALL_MODELS}")
        return 1

    d = load(a.data)
    mask = usable(d)
    rows = d[mask].reset_index(drop=True)
    folds = folds_from_data(rows, C.FORWARD_FOLDS, C.FORWARD_FOLD_DAYS)

    if a.only is None:
        print(f"\n전처리 결과 {len(d):,}행 → 모든 모델이 공유하는 행 {len(rows):,}행")
        print(f"  뺀 행 {len(d) - len(rows):,}개 — 변수 결측(워밍업) 또는 창이 공백을 넘는 행")
        print(f"  전진검증 {len(folds)}구간 (자료에서 생성): "
              + " · ".join(f"{lo}~{hi - 1}" for lo, hi in folds))
        print(f"  시험 구간 {int(rows.loc[rows['split'] == 'test', 'date_key'].min())} 이후 "
              f"{int((rows['split'] == 'test').sum()):,}행\n", flush=True)

    res: dict = {}
    tree_want = [m for m in want if m in TREE_MODELS]
    net_want = [m for m in want if m in NET_MODELS]
    if "ensemble" in want:
        tree_want = list(dict.fromkeys(tree_want + ["et"]))
        net_want = list(dict.fromkeys(net_want + ["gru"]))

    if a.only == "trees":
        print("── 트리 계열 (sklearn 전용 프로세스) ──")
        res = run_trees(rows, folds, tree_want)
        (HERE / "_res_trees.json").write_text(json.dumps(res, ensure_ascii=False), "utf-8")
        return 0
    if a.only == "nets":
        print("── 순환 계열 (torch 전용 프로세스) ──")
        res = run_nets(d, folds, net_want, a.data, a.net_job, a.cache_dir)
        if a.net_job is not None:
            return 0
        (HERE / "_res_nets.json").write_text(json.dumps(res, ensure_ascii=False), "utf-8")
        return 0

    # ── 오케스트레이터 ──
    if "baseline" in want:
        print("── 기준선 (학습 없음) ──", flush=True)
        for k, v in run_baselines(rows, folds).items():
            res[k] = v
            print(f"  {BASE_MODELS[k]:22s} {'test':>8} MSE {v.get('test', {}).get('MSE')}",
                  flush=True)

    for kind, sel in (("nets", net_want), ("trees", tree_want)):
        if not sel:
            continue
        print()
        cmd = [sys.executable, "-B", str(Path(__file__).name),
               "--data", str(a.data.resolve()), "--only", kind,
               "--models", ",".join(sel + (["ensemble"] if "ensemble" in want else []))]
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"✗ {kind} 하위 프로세스가 실패했다 (종료 {r.returncode})")
            return 1
        f = HERE / f"_res_{kind}.json"
        if f.exists():
            res.update(json.loads(f.read_text("utf-8")))

    # ── 결합 ──
    if "ensemble" in want:
        tp = HERE / "_pred_trees.npz"
        np_ = HERE / "_pred_nets.npz"
        if tp.exists() and np_.exists():
            T, N = np.load(tp), np.load(np_)
            ridx = N["row_index"]
            res["ensemble"] = {}
            print("\n── 결합 (ExtraTrees + GRU · 고정 반반) ──")
            for split, hi, label in jobs(rows, folds):
                tk, nk = f"et|{label}", f"gru|{label}"
                if tk not in T.files or nk not in N.files:
                    raise ValueError(f"결합에 필요한 예측이 없습니다: {label}")
                ei = N[f"gru|idx|{label}"]
                # 두 예측이 같은 행을 가리키는지 확인한다 — 길이만 맞추면 안 된다
                day = rows["date_key"].to_numpy()
                te_m = (day >= split) if hi is None else ((day >= split) & (day < hi))
                tree_rows = np.where(te_m)[0]
                net_rows = ridx[ei]
                if not np.array_equal(tree_rows, net_rows):
                    print(f"  ✗ {label}: 두 예측의 행이 다르다 "
                          f"({len(tree_rows)} vs {len(net_rows)})")
                    raise ValueError(f"결합 예측의 행 불일치: {label}")
                y = rows["y"].to_numpy()[tree_rows]
                b = C.BLEND_WEIGHT * T[tk] + (1 - C.BLEND_WEIGHT) * N[nk]
                res["ensemble"][label] = {**score(y, b), "rows": int(len(y))}
                print(f"  {'ExtraTrees+GRU':22s} {label:>8} MSE "
                      f"{res['ensemble'][label]['MSE']:9.3f}")

    summary = summarize(res, folds)
    out = {
        "자료": {"파일": a.data.name, "sha256": __import__("hashlib").sha256(a.data.read_bytes()).hexdigest(), "전처리행": len(d), "공통행": len(rows),
               "뺀행": len(d) - len(rows),
               "행기준설명": "모든 모델이 같은 행을 쓴다. 다른 행 기준의 수치와 섞지 말 것",
               "시험행": int((rows["split"] == "test").sum())},
        "전진검증구간": folds,
        "설정": {"tree": C.TREE, "rf": C.RF, "boost": C.BOOST, "net": C.NET, "seeds": C.SEEDS,
               "blend_weight": C.BLEND_WEIGHT,
               "tree_features": C.TREE_FEATURES, "net_calendar": C.NET_CALENDAR},
        "모델별": summary,
        "구간별_원자료": res,
        "선정규약": ("전진검증 평균 MSE 로만 고른다. 시험 구간은 선정에 쓰지 않는다. "
                 "순환신경망은 시드 3개 예측 평균으로 보고하고 시험 성적으로 시드를 고르지 않는다"),
        "환경": {"python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, **{k: __import__("importlib.metadata", fromlist=["version"]).version(k) for k in ("scikit-learn", "torch", "joblib")}},
    }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")

    print(f"\n{'모델':24s}{'전진검증':>10}{'±':>8}{'시험':>10}")
    order = sorted(summary, key=lambda k: summary[k]["전진검증_평균"] or 9e9)
    for k in order:
        s = summary[k]
        f = s["전진검증_평균"]; sd = s["전진검증_표준편차"]
        t = (s["시험"] or {}).get("MSE")
        print(f"{k:24s}{f if f is not None else '—':>10}"
              f"{sd if sd is not None else '—':>8}{t if t is not None else '—':>10}")
    if set(TREE_MODELS).issubset(res) and set(NET_MODELS).issubset(res):
        from evaluate_duplicates import evaluate
        evaluate(a.data, a.out)
    from artifacts import write_manifest
    write_manifest(a.data, a.out)
    print(f"\n저장: {a.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
