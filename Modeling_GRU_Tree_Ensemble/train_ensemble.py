#!/usr/bin/env python3
"""트리 + 순환신경망 앙상블 재학습·평가. ensemble_input_data.csv 와 selection.json 만 쓴다.

실행:
    python train_gru_part.py     # ① 순환신경망 (torch 전용 프로세스)
    python train_ensemble.py     # ② 트리 + 결합 (sklearn 전용 프로세스)

출력: final_tree.joblib, reproduced_metrics.csv, reproduced_folds.csv,
      reproduced_test_predictions.csv, reproduction_metadata.json

**왜 두 프로세스인가.** macOS 에서 sklearn(joblib 포크·OpenMP)을 먼저 돌린 뒤 같은
프로세스에서 torch Adam 이 도는 순간 libomp 가 두 번 적재되어 교착한다.
환경변수(`OMP_NUM_THREADS`·`KMP_DUPLICATE_LIB_OK`)만으로는 막히지 않았다 —
`torch/optim/adam.py` 안에서 멈추는 것을 faulthandler 로 실측했다.
그래서 순환신경망은 `train_gru_part.py`(torch 만 import)가 맡고,
이 파일은 트리만 학습한 뒤 그 예측을 읽어 합친다. `Modeling_RNN/` 도 같은 구조다.

이 파일은 **torch 를 import 하지 않는다.** 그것이 분리의 핵심이다.

## 결합 방식 — 가중치를 고르지 않는다

고정 0.5 : 0.5. 전진검증으로 역산한 최적 트리 비중은 0.69, 정병근 시험 구간에서는 0.43 으로
구간마다 반대 방향이다. 시험 앞 20% 로 학습해 뒤 80% 에 쓰면 오히려 3.9 나빠졌다.
3% 이득에 선택 위험을 사지 않는다.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent


def windows(d: pd.DataFrame, win: int, steps: int):
    """창을 만들 수 있는 행만 골라낸다 — `train_gru_part.py` 와 같은 규칙이어야 한다.

    삭제된 2021-07-13·15 를 건너뛰는 창은 제외한다. 두 프로세스가 같은 행 집합을
    보지 않으면 예측 길이가 어긋난다.
    """
    kw = d["kW"].to_numpy(np.float32)
    ts = pd.DatetimeIndex(d["ts"])
    ok = (ts[win - 1:] - ts[:len(ts) - win + 1]) == pd.Timedelta(minutes=15 * (win - 1))
    _ = sliding_window_view(kw, win)[ok]
    return np.arange(win - 1, len(d))[ok]


def score(y, p):
    e = y - p
    return {"MSE": float((e ** 2).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAE": float(np.abs(e).mean()),
            "R2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=HERE / "ensemble_input_data.csv")
    ap.add_argument("--selection", type=Path, default=HERE / "selection.json")
    ap.add_argument("--gru", type=Path, default=HERE / "_gru_predictions.npz")
    a = ap.parse_args()

    sel = json.loads(a.selection.read_text(encoding="utf-8"))
    cfg = sel["config"]
    if not a.gru.exists():
        raise SystemExit(f"{a.gru.name} 이 없다. 먼저 `python train_gru_part.py` 를 돌릴 것")
    G = np.load(a.gru)

    d = pd.read_csv(a.input, encoding="utf-8-sig")
    d["ts"] = pd.to_datetime(d["ts"])
    d = d[d["split"] != "boundary_excluded"].reset_index(drop=True)
    idx = windows(d, cfg["gru"]["window"], cfg["gru"]["steps"])
    w = d.iloc[idx].reset_index(drop=True)
    day = w["date_key"].to_numpy()
    split_day = int(d.loc[d["split"] == "test", "date_key"].min())
    bw = cfg["blend_weight"]
    print(f"{len(d):,}행 → 창 유효 {len(w):,}행 (불연속 창 제외)")
    print(f"트리 {len(cfg['features'])}변수 · 결합 {bw}:{1 - bw} "
          f"· 순환신경망 예측은 {a.gru.name} 에서 읽는다\n")

    def run(split, hi, label, keep=False):
        tr = w[w["date_key"] < split]
        te_m = (day >= split) if hi is None else ((day >= split) & (day < hi))
        te = w[te_m]
        p = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", ExtraTreesRegressor(**cfg["tree"]))])
        p.fit(tr[cfg["features"]], tr["y"])
        tree = p.predict(te[cfg["features"]])
        if not keep:
            del p
            gc.collect()
        gru = G[f"pred_{label}"]
        y = te["y"].to_numpy()
        assert len(gru) == len(y), f"{label}: 길이 불일치 {len(gru)} vs {len(y)} — 창 규칙 확인"
        blend = bw * tree + (1 - bw) * gru
        r = {"label": label, "rows": len(y), "epochs": G[f"eps_{label}"].tolist(),
             "tree": score(y, tree), "gru": score(y, gru), "blend": score(y, blend)}
        if keep:
            r["_pred"] = {"y": y, "tree": tree, "gru": gru, "blend": blend}
            r["_te"] = te
            r["_model"] = p
        return r

    folds = []
    print(f"  {'구간':>7}{'트리':>9}{'GRU':>9}{'앙상블':>9}{'행수':>8}")
    for lo, hi in sel["forward_folds"]:
        if (day < lo).sum() < 1500 or f"pred_{lo}" not in G:
            continue
        r = run(lo, hi, str(lo))
        folds.append(r)
        print(f"  {str(lo)[4:]:>7}{r['tree']['MSE']:>9.2f}{r['gru']['MSE']:>9.2f}"
              f"{r['blend']['MSE']:>9.2f}{r['rows']:>8}", flush=True)
    avg = {k: round(float(np.mean([f[k]["MSE"] for f in folds])), 3)
           for k in ("tree", "gru", "blend")} if folds else None
    if avg:
        print(f"\n  전진검증 평균 — 트리 {avg['tree']} · GRU {avg['gru']} · "
              f"앙상블 {avg['blend']}  (이득 {avg['tree'] - avg['blend']:+.2f})\n")

    t = run(split_day, None, "test", keep=True)
    print(f"  시험 구간 {t['rows']:,}행")
    for k in ("tree", "gru", "blend"):
        s = t[k]
        print(f"    {k:6} MSE {s['MSE']:8.3f} · RMSE {s['RMSE']:6.3f} · "
              f"MAE {s['MAE']:6.3f} · R² {s['R2']:.4f}")
    solo = min(t["tree"]["MSE"], t["gru"]["MSE"])
    print(f"    앙상블 이득 {t['tree']['MSE'] - t['blend']['MSE']:+.3f} "
          f"({(1 - t['blend']['MSE'] / solo) * 100:+.1f}% vs 단독 최고)")

    pd.DataFrame([{"split": "test", **{f"{m}_{s}": t[m][s]
                                       for m in ("tree", "gru", "blend")
                                       for s in ("MSE", "RMSE", "MAE", "R2")}}]).to_csv(
        HERE / "reproduced_metrics.csv", index=False, encoding="utf-8-sig")
    if folds:
        pd.DataFrame([{"fold": f["label"], "rows": f["rows"],
                       "tree_MSE": f["tree"]["MSE"], "gru_MSE": f["gru"]["MSE"],
                       "blend_MSE": f["blend"]["MSE"]} for f in folds]).to_csv(
            HERE / "reproduced_folds.csv", index=False, encoding="utf-8-sig")
    out = t["_te"][["ts", "date_key"]].copy()
    out["actual"] = t["_pred"]["y"]
    for k in ("tree", "gru", "blend"):
        out[f"pred_{k}"] = t["_pred"][k]
    out.to_csv(HERE / "reproduced_test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"estimator": t["_model"], "features": cfg["features"]},
                HERE / "final_tree.joblib")
    (HERE / "reproduction_metadata.json").write_text(json.dumps({
        "config": cfg, "seeds": sel["seeds"], "split_day": split_day,
        "rows_total": len(d), "rows_window_valid": len(w),
        "forward": avg,
        "forward_folds": [{"fold": f["label"], "rows": f["rows"], "tree": f["tree"],
                           "gru": f["gru"], "blend": f["blend"]} for f in folds],
        "test": {k: t[k] for k in ("tree", "gru", "blend")},
        "test_rows": t["rows"], "gru_epochs_test": t["epochs"],
        "note": ("순환신경망은 train_gru_part.py 가 별도 프로세스에서 학습한다 "
                 "(macOS libomp 이중 적재 교착 회피). 저장 모델은 첫 시드로 고정하고 "
                 "시험 성적으로 시드를 고르지 않는다"),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: final_tree.joblib · reproduced_metrics.csv · reproduced_folds.csv"
          f" · reproduced_test_predictions.csv · reproduction_metadata.json")


if __name__ == "__main__":
    main()
