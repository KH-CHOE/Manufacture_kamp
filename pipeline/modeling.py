"""전처리 결과 하나로 모든 모델을 **같은 행에서** 비교한다.

왜 이렇게 나눴나
----------------
`preprocessing.py` 가 만든 CSV 한 장을 모든 모델이 읽는다. 트리는 열을 골라 쓰고
순환신경망은 같은 파일에서 창을 만든다. **입력이 같으니 차이는 모형에서만 온다.**

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
    ts = pd.DatetimeIndex(d["ts"])
    ok = np.zeros(len(d), bool)
    span = pd.Timedelta(minutes=C.STEP_MIN * (win - 1))
    valid = (ts[win - 1:] - ts[:len(ts) - win + 1]) == span
    ok[np.arange(win - 1, len(d))] = valid
    return ok


def usable(d: pd.DataFrame) -> np.ndarray:
    """모든 모델이 공유할 행. 하나라도 못 쓰는 행은 전부에서 뺀다."""
    cols = [c for c in (C.TREE_FEATURES + C.NET_CALENDAR + ["kW", "y"]) if c in d.columns]
    miss = [c for c in (C.TREE_FEATURES + C.NET_CALENDAR) if c not in d.columns]
    if miss:
        raise SystemExit(f"✗ 모델 입력 열이 전처리 결과에 없다: {miss}\n"
                         f"  preprocessing.py 를 다시 돌렸는지 확인해라")
    return d[cols].notna().all(axis=1).to_numpy() & window_ok(d, C.NET["window"])


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
            return RandomForestRegressor(**C.TREE)
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
                            HERE / f"model_{kind}.joblib", compress=3)
            print(f"  {TREE_MODELS[kind]:22s} {label:>8} MSE {res[kind][label]['MSE']:9.3f}",
                  flush=True)
    np.savez(HERE / "_pred_trees.npz",
             **{f"{k}|{l}": v for k, d2 in preds.items() for l, v in d2.items()})
    return res


# ══ 순환 계열 (torch 전용 프로세스) ═════════════════════════════════
def run_nets(d: pd.DataFrame, folds: list[list[int]], want: list[str]) -> dict:
    import torch
    from torch import nn
    from numpy.lib.stride_tricks import sliding_window_view
    torch.set_num_threads(4)
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
    kw = d["kW"].to_numpy(np.float32)
    # sliding_window_view 의 i 번째 창은 행 i+win-1 에서 끝난다.
    # 창 96칸을 (24시간 × 4구간) 으로 접어 넣는다 — 한 걸음이 한 시간이다
    X = sliding_window_view(kw, win).reshape(-1, steps, win // steps)
    idx = np.arange(win - 1, len(d))
    X = X[:len(idx)]
    Y = d["y"].to_numpy(np.float32)[idx]
    CAL = d[C.NET_CALENDAR].to_numpy(np.float32)[idx]
    day = d["date_key"].to_numpy()[idx]

    def train_one(kind, start, split_day, hi, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        tr_m = (day >= start) & (day < split_day)
        te_m = (day >= split_day) if hi is None else ((day >= split_day) & (day < hi))
        mu, sd = X[tr_m].mean(), X[tr_m].std() + 1e-8
        ymu, ysd = Y[tr_m].mean(), Y[tr_m].std() + 1e-8
        Xn = ((X - mu) / sd).astype(np.float32)
        Yn = ((Y - ymu) / ysd).astype(np.float32)
        ti = np.where(tr_m)[0]
        cut = int(len(ti) * 0.85)
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
            if vl < best - 1e-4:
                best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= g["patience"]:
                    break
        net.load_state_dict(state)
        ei = np.where(te_m)[0]
        return net, ei, pred(ei), ep + 1

    res: dict = {k: {} for k in want}
    preds: dict = {k: {} for k in want}
    for split, hi, label in jobs(d, folds):
        if (day < split).sum() < 1500:
            continue
        start = int((pd.Timestamp(str(split)) - pd.DateOffset(months=g["train_months"]))
                    .strftime("%Y%m%d")) if g["train_months"] else 0
        for kind in want:
            ps, eps, first = [], [], None
            for sd_ in C.SEEDS:
                net, ei, p, ep = train_one(kind, start, split, hi, sd_)
                ps.append(p); eps.append(ep)
                if first is None:
                    first = net
            # **시드별 예측을 평균한다.** 시험 성적으로 시드를 고르지 않는다
            mean_p = np.mean(ps, axis=0)
            res[kind][label] = {**score(Y[ei], mean_p), "rows": int(len(ei)),
                                "epochs": eps, "seeds": C.SEEDS}
            preds[kind][label] = mean_p
            preds[f"{kind}|idx|{label}"] = ei
            if label == "test":
                import torch as T
                T.save({"state_dict": first.state_dict(), "config": g,
                        "calendar": C.NET_CALENDAR, "seed": C.SEEDS[0],
                        "주의": "보고 예측은 시드 3개 평균이다. 이 파일은 첫 시드 하나다"},
                       HERE / f"model_{kind}.pt")
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
    ap.add_argument("--data", type=Path, default=HERE / "processed.csv")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help=f"쉼표로 구분. 가능: {','.join(ALL_MODELS)}")
    ap.add_argument("--only", choices=["trees", "nets"], default=None,
                    help="내부용 — 오케스트레이터가 하위 프로세스로 호출할 때 쓴다")
    a = ap.parse_args()

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
    tree_want = [m for m in want if m in TREE_MODELS] or (
        ["et"] if "ensemble" in want else [])
    net_want = [m for m in want if m in NET_MODELS] or (
        ["gru"] if "ensemble" in want else [])

    if a.only == "trees":
        print("── 트리 계열 (sklearn 전용 프로세스) ──")
        res = run_trees(rows, folds, tree_want)
        (HERE / "_res_trees.json").write_text(json.dumps(res, ensure_ascii=False), "utf-8")
        return 0
    if a.only == "nets":
        print("── 순환 계열 (torch 전용 프로세스) ──")
        res = run_nets(rows, folds, net_want)
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
               "--data", str(a.data), "--only", kind,
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
                    continue
                ei = N[f"gru|idx|{label}"]
                # 두 예측이 같은 행을 가리키는지 확인한다 — 길이만 맞추면 안 된다
                day = rows["date_key"].to_numpy()
                te_m = (day >= split) if hi is None else ((day >= split) & (day < hi))
                tree_rows = np.where(te_m)[0]
                net_rows = ridx[ei]
                if not np.array_equal(tree_rows, net_rows):
                    print(f"  ✗ {label}: 두 예측의 행이 다르다 "
                          f"({len(tree_rows)} vs {len(net_rows)})")
                    continue
                y = rows["y"].to_numpy()[tree_rows]
                b = C.BLEND_WEIGHT * T[tk] + (1 - C.BLEND_WEIGHT) * N[nk]
                res["ensemble"][label] = {**score(y, b), "rows": int(len(y))}
                print(f"  {'ExtraTrees+GRU':22s} {label:>8} MSE "
                      f"{res['ensemble'][label]['MSE']:9.3f}")

    summary = summarize(res, folds)
    out = {
        "자료": {"파일": a.data.name, "전처리행": len(d), "공통행": len(rows),
               "뺀행": len(d) - len(rows),
               "행기준설명": "모든 모델이 같은 행을 쓴다. 다른 행 기준의 수치와 섞지 말 것",
               "시험행": int((rows["split"] == "test").sum())},
        "전진검증구간": folds,
        "설정": {"tree": C.TREE, "boost": C.BOOST, "net": C.NET, "seeds": C.SEEDS,
               "blend_weight": C.BLEND_WEIGHT,
               "tree_features": C.TREE_FEATURES, "net_calendar": C.NET_CALENDAR},
        "모델별": summary,
        "구간별_원자료": res,
        "선정규약": ("전진검증 평균 MSE 로만 고른다. 시험 구간은 선정에 쓰지 않는다. "
                 "순환신경망은 시드 3개 예측 평균으로 보고하고 시험 성적으로 시드를 고르지 않는다"),
        "환경": {"python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__},
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
    print(f"\n저장: {a.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
