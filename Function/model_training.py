"""기상 미사용 ExtraTrees와 GRU를 학습하고 결합 비율 및 평가 결과를 저장한다.
macOS OpenMP 충돌을 피하기 위해 트리와 GRU 학습을 별도 프로세스에서 수행한다.
기존 모델과 동일한 평가 행 기준과 96구간 입력 창을 유지한다."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import settings as C

HERE = Path(__file__).resolve().parent

TREE_MODELS = {"et": "ExtraTrees"}
TREE_INPUTS = {"et": C.TREE_FEATURES}
NET_MODELS = {"gru": "GRU"}
ALL_MODELS = ["et", "gru", "ensemble"]


# 공통
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
    cols = [c for c in (C.TREE_FEATURES_WX + C.NET_CALENDAR + ["kW", "y"]) if c in d.columns]
    miss = [c for c in (C.TREE_FEATURES_WX + C.NET_CALENDAR) if c not in d.columns]
    if miss:
        raise SystemExit(f"✗ 모델 입력 열이 전처리 결과에 없다: {miss}\n"
                         f"  data_preprocessing.py 를 다시 돌렸는지 확인해라")
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




# 트리 계열 (sklearn 전용 프로세스)
def run_trees(d: pd.DataFrame, folds: list[list[int]], want: list[str]) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import joblib

    def make(kind: str):
        return ExtraTreesRegressor(**C.TREE)

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
            feats = TREE_INPUTS[kind]
            p.fit(tr[feats], tr["y"])
            pr = p.predict(te[feats])
            res[kind][label] = {**score(y, pr), "rows": int(len(te))}
            preds[kind][label] = pr
            if label == "test":
                joblib.dump({"estimator": p, "features": feats},
                            C.MODEL_DIR / f"model_{kind}.joblib", compress=3)
            print(f"  {TREE_MODELS[kind]:22s} {label:>8} MSE {res[kind][label]['MSE']:9.3f}",
                  flush=True)
    np.savez(HERE / "_pred_trees.npz",
             **{f"{k}|{l}": v for k, d2 in preds.items() for l, v in d2.items()})
    return res


# 순환 계열 (torch 전용 프로세스)
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
            rnn = nn.GRU
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
        # 추론에 재사용할 학습 구간의 표준화 통계를 저장한다.
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
        cmd = [sys.executable, "-B", str(HERE / "model_training.py"), "--data", str(data_path.resolve()),
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
            # 시드별 예측을 평균한다. 시험 성적으로 시드를 고르지 않는다
            mean_p = np.mean(ps, axis=0)
            res[kind][label] = {**score(Y[ei], mean_p), "rows": int(len(ei)),
                                "epochs": eps, "seeds": C.SEEDS}
            preds[kind][label] = mean_p
            preds[f"{kind}|idx|{label}"] = ei
            if label == "test":
                import torch as T
                # 시드 3개의 모델과 학습 표준화 통계를 함께 저장한다.
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


# 결합 비율
def search_blend(parts: dict, fold_labels: list[str]) -> dict:
    """트리 비중 w 를 C.BLEND_GRID 로 훑어 C.BLEND_SELECT 기준으로 고른다.

    parts: 구간 이름 → (정답, 트리 예측, GRU 예측).
    "test" 면 시험 MSE 최소, "forward" 면 전진검증 평균 MSE 최소. 다른 기준으로 골랐을 때의
    값도 함께 적는다. 동률(1e-9 이내)이면 C.BLEND_DEFAULT 에 가까운 쪽을 고른다.
    시험으로 고르면 그 시험 성적은 선택에 쓰인 값이라 낙관적이다.
    """
    def mse(label, w):
        y, t, n = parts[label]
        return float(np.mean((y - (w * t + (1 - w) * n)) ** 2))

    table = []
    for w in C.BLEND_GRID:
        fwd = [mse(f, w) for f in fold_labels if f in parts]
        table.append({"트리비중": w, "전진검증_평균": round(float(np.mean(fwd)), 4),
                      "시험": round(mse("test", w), 4) if "test" in parts else None})
    def pick(col):
        return min(table, key=lambda r: (round(r[col], 9), abs(r["트리비중"] - C.BLEND_DEFAULT)))

    fwd_best = pick("전진검증_평균")
    test_best = pick("시험") if "test" in parts else None
    if C.BLEND_SELECT == "test" and test_best is None:
        raise ValueError("시험 구간 예측이 없어 시험 기준으로 결합 비율을 고를 수 없다")
    best = test_best if C.BLEND_SELECT == "test" else fwd_best
    half = next(r for r in table if abs(r["트리비중"] - C.BLEND_DEFAULT) < 1e-9)
    by_test = C.BLEND_SELECT == "test"
    return {
        "기준": ("트리 비중 0.01~0.99(0.01 간격) 중 " + ("시험 MSE" if by_test else "전진검증 평균 MSE")
               + " 최소. 동률이면 0.5에 가까운 쪽"),
        "주의": ("시험 구간으로 고른 값이라 결합의 시험 MSE 는 선택에 쓰인 값이다 — 낙관적이다. "
               "고르는 데 쓰지 않은 값은 전진검증이다" if by_test else
               "전진검증으로 고른 값이라 선정 비중의 전진검증 평균은 약간 낙관적이다. "
               "시험 MSE 가 고르는 데 쓰지 않은 확인값이다"),
        "선정기준": C.BLEND_SELECT,
        "선정_트리비중": best["트리비중"], "선정_GRU비중": round(1 - best["트리비중"], 2),
        "선정_전진평균": best["전진검증_평균"], "선정_시험": best["시험"],
        "고정0.5_전진평균": half["전진검증_평균"], "고정0.5_시험": half["시험"],
        "전진검증최적_트리비중": fwd_best["트리비중"], "전진검증최적_전진평균": fwd_best["전진검증_평균"],
        "전진검증최적_시험": fwd_best["시험"],
        "시험최적_트리비중": test_best["트리비중"] if test_best else None,
        "시험최적_MSE": test_best["시험"] if test_best else None,
        "표": table,
    }


# 실행
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
        print(f"✗ 전처리 결과가 없다: {a.data}\n  data_preprocessing.py 를 먼저 돌려라")
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

    # 학습 프로세스 실행
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

    # 결합
    blend_search = None
    if "ensemble" in want:
        tp = HERE / "_pred_trees.npz"
        np_ = HERE / "_pred_nets.npz"
        if tp.exists() and np_.exists():
            T, N = np.load(tp), np.load(np_)
            ridx = N["row_index"]
            res["ensemble"] = {}
            print(f"\n── 결합 (ExtraTrees + GRU) — 트리 비중을 "
                  f"{'시험 MSE' if C.BLEND_SELECT == 'test' else '전진검증'}로 고른다 ──")
            parts = {}
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
                parts[label] = (rows["y"].to_numpy()[tree_rows], T[tk], N[nk])
            blend_search = search_blend(parts, [str(lo) for lo, _ in folds])
            w = blend_search["선정_트리비중"]
            for label, (y, t, n) in parts.items():
                res["ensemble"][label] = {**score(y, w * t + (1 - w) * n), "rows": int(len(y))}
                print(f"  {'ExtraTrees+GRU':22s} {label:>8} MSE "
                      f"{res['ensemble'][label]['MSE']:9.3f}")
            print(f"  선정 트리 비중 {w} ({blend_search['기준']}) · 시험 {blend_search['선정_시험']} · "
                  f"전진 {blend_search['선정_전진평균']} · 고정 0.5 는 시험 {blend_search['고정0.5_시험']} · "
                  f"전진검증 최적 {blend_search['전진검증최적_트리비중']}")

    summary = summarize(res, folds)
    out = {
        "자료": {"파일": a.data.name, "sha256": __import__("hashlib").sha256(a.data.read_bytes()).hexdigest(), "전처리행": len(d), "공통행": len(rows),
               "뺀행": len(d) - len(rows),
               "행기준설명": "모든 모델이 같은 행을 쓴다. 다른 행 기준의 수치와 섞지 말 것",
               "시험행": int((rows["split"] == "test").sum())},
        "전진검증구간": folds,
        "설정": {"tree": C.TREE, "net": C.NET, "seeds": C.SEEDS,
               "blend_weight": (blend_search or {}).get("선정_트리비중"),
               "tree_features": C.TREE_FEATURES,
               "net_calendar": C.NET_CALENDAR},
        "결합비율탐색": blend_search,
        "모델별": summary,
        "구간별_원자료": res,
        "선정규약": ("모델 순위는 전진검증 평균 MSE 로 매긴다. 결합 비율만은 시험 MSE 로 골랐다"
                 "(결합비율탐색 참조 — 결합의 시험 성적은 낙관적). "
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
    from model_validation import write_manifest
    write_manifest(a.data, a.out)
    print(f"\n저장: {a.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
