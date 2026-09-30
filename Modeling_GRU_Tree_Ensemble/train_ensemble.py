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
import hashlib
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


def data_fingerprint(w, cfg, sel) -> dict:
    """순환신경망이 **실제로 쓰는 값 전부**와 학습 설정을 지문으로 만든다.

    예전에는 시각축(`ts`)의 sha1 만 비교했다. 그래서 시각만 같고 `kW` 에 1,000 을 더하거나
    `tod_sin` 을 0 으로 바꾼 자료로도 `verified` 가 나왔다(2차 리뷰 S2).
    설정도 저장만 하고 대조하지 않아 은닉 999·시드 [999] 짜리 예측이 통과했다.

    이제 **값·순서·분할·설정을 모두** 넣는다. 부동소수는 표기 차이로 지문이 흔들리지
    않도록 소수 6자리로 고정해 직렬화한다.
    """
    cols = ["ts", "split", "kW", "y"] + list(cfg["calendar"])
    missing = [c for c in cols if c not in w.columns]
    if missing:
        raise SystemExit(f"지문에 필요한 열이 없다: {missing}")
    parts = []
    for c in cols:
        s = w[c]
        if s.dtype.kind in "fc":
            parts.append(c + "=" + "|".join(f"{v:.6f}" for v in s.to_numpy(float)))
        else:
            parts.append(c + "=" + "|".join(s.astype(str)))
    cfgsig = json.dumps({"gru": cfg["gru"], "calendar": list(cfg["calendar"]),
                         "seeds": list(sel["seeds"]),
                         "forward_folds": sel["forward_folds"]},
                        sort_keys=True, ensure_ascii=False)
    return {
        "data_sha1": hashlib.sha1("\n".join(parts).encode()).hexdigest(),
        "config_sha1": hashlib.sha1(cfgsig.encode()).hexdigest(),
        "config_json": cfgsig,
        "rows": int(len(w)),
        "columns": cols,
        "ts_first": str(w["ts"].iloc[0]),
        "ts_last": str(w["ts"].iloc[-1]),
    }


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
          f"· 순환신경망 예측은 {a.gru.name} 에서 읽는다")

    # ── 지문 대조 ── 같은 자료·같은 설정으로 낸 예측인지 확인한다 (2차 리뷰 S2)
    fp = data_fingerprint(w, cfg, sel)
    if "fp_data_sha1" in G:
        for key, label in (("data_sha1", "자료"), ("config_sha1", "설정")):
            got, wantv = str(G[f"fp_{key}"].item()), fp[key]
            if got != wantv:
                raise SystemExit(
                    f"지문 불일치({label}) — {a.gru.name} 은 다른 {label}로 낸 예측이다.\n"
                    f"  지금 {wantv[:12]} · 예측 파일 {got[:12]}\n"
                    f"  예측 파일 설정: {str(G.get('fp_config_json', np.array('?')).item())[:200]}\n"
                    f"  train_gru_part.py 를 이 CSV·이 설정으로 다시 돌려라")
        print(f"지문 일치 — 자료 {fp['data_sha1'][:12]} · 설정 {fp['config_sha1'][:12]}"
              f" · {fp['rows']:,}행 · 열 {len(fp['columns'])}개")
        fp_state = "verified"
    elif "fp_ts_sha1" in G:
        fp_state = "ts_only"
        print(f"주의 — {a.gru.name} 은 **시각축만** 담은 옛 지문이다.\n"
              f"  값·분할·설정이 같은지는 확인할 수 없다. 다시 학습하면 전체 지문이 된다.")
    else:
        fp_state = "absent"
        print(f"주의 — {a.gru.name} 에 지문이 없다(지문을 심기 전에 만든 파일이다).\n"
              f"  평가 행 위치(idx_*)는 대조하지만 **자료 자체가 같은지는 확인할 수 없다.**\n"
              f"  지금 자료 지문 {fp['data_sha1'][:12]} · 창 유효 {len(w):,}행")
    print()

    def run(split, hi, label, keep=False, keep_pred=False):
        tr = w[w["date_key"] < split]
        te_m = (day >= split) if hi is None else ((day >= split) & (day < hi))
        te = w[te_m]
        p = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("model", ExtraTreesRegressor(**cfg["tree"]))])
        p.fit(tr[cfg["features"]], tr["y"])
        tree = p.predict(te[cfg["features"]])
        if not keep:
            del p
            gc.collect()      # 구간마다 버린다 — 5판을 들고 있으면 메모리가 모자란다
        gru = G[f"pred_{label}"]
        y = te["y"].to_numpy()
        assert len(gru) == len(y), f"{label}: 길이 불일치 {len(gru)} vs {len(y)} — 창 규칙 확인"
        # **길이만 맞추면 안 된다.** 길이가 같은 오래된 예측 파일이 섞이면 조용히
        # 다른 행과 결합된다. 코덱스 리뷰 R6 의 지적이다.
        # `idx_*` 는 창 유효 프레임 w 안의 정수 위치다 — 그 위치까지 같은지 본다.
        ei_here = np.where(te_m)[0]
        ei_saved = G[f"idx_{label}"]
        assert np.array_equal(ei_here, ei_saved), (
            f"{label}: 평가 행 위치가 다르다 — 이 CSV 와 {a.gru.name} 이 짝이 아니다. "
            f"길이는 {len(ei_here)} 로 같지만 첫 어긋난 위치 "
            f"{int(np.argmax(ei_here != ei_saved)) if len(ei_here) == len(ei_saved) else 'n/a'}. "
            f"train_gru_part.py 를 이 CSV 로 다시 돌려라")
        blend = bw * tree + (1 - bw) * gru
        r = {"label": label, "rows": len(y), "epochs": G[f"eps_{label}"].tolist(),
             "tree": score(y, tree), "gru": score(y, gru), "blend": score(y, blend)}
        if keep or keep_pred:
            r["_pred"] = {"y": y, "tree": tree, "gru": gru, "blend": blend}
        if keep:
            r["_te"] = te
            r["_model"] = p
        return r

    folds = []
    print(f"  {'구간':>7}{'트리':>9}{'GRU':>9}{'앙상블':>9}{'행수':>8}")
    for lo, hi in sel["forward_folds"]:
        if (day < lo).sum() < 1500 or f"pred_{lo}" not in G:
            continue
        r = run(lo, hi, str(lo), keep_pred=True)
        folds.append(r)
        print(f"  {str(lo)[4:]:>7}{r['tree']['MSE']:>9.2f}{r['gru']['MSE']:>9.2f}"
              f"{r['blend']['MSE']:>9.2f}{r['rows']:>8}", flush=True)
    avg = {k: round(float(np.mean([f[k]["MSE"] for f in folds])), 3)
           for k in ("tree", "gru", "blend")} if folds else None
    if avg:
        print(f"\n  전진검증 평균 — 트리 {avg['tree']} · GRU {avg['gru']} · "
              f"앙상블 {avg['blend']}  (이득 {avg['tree'] - avg['blend']:+.2f})\n")

    # ── 가중치 훑기 ── **진단이다. 여기서 가중치를 고르지 않는다.**
    # 왜 재는가: README·보고서에 "전진검증 최적 0.69 / 31.06" 이 적혀 있었는데
    # 어떤 결과 파일에도 없는 수치였다(코덱스 리뷰 R10). 근거 없는 숫자를 지우는 대신
    # 실제로 재서 출처를 만든다. 설계는 고정 0.5 그대로다 —
    # 전진검증으로 가중치를 고르면 그 4구간이 선정에 쓰인 것이 되고,
    # 시험 구간 최적(0.425)과 방향이 반대라 어느 쪽으로도 일반화되지 않는다.
    sweep = None
    if folds:
        grid = np.round(np.arange(0.0, 1.001, 0.05), 2)
        per_w = []
        for wgt in grid:
            ms = [float(np.mean((f["_pred"]["y"]
                                 - (wgt * f["_pred"]["tree"]
                                    + (1 - wgt) * f["_pred"]["gru"])) ** 2))
                  for f in folds]
            per_w.append((float(wgt), round(float(np.mean(ms)), 3)))
        best_w, best_mse = min(per_w, key=lambda t_: t_[1])
        half = dict(per_w)[0.5]
        sweep = {"grid": per_w, "best_weight": best_w, "best_MSE": best_mse,
                 "fixed_half_MSE": half,
                 "gain_of_tuning": round(half - best_mse, 3),
                 "판정": ("고정 0.5 를 유지한다. 최적으로 바꿔 얻는 이득은 "
                        f"{half - best_mse:.2f} 인데, 그 최적은 이 4구간에서 역산한 값이라 "
                        "그만큼 선정 편향을 산다. 시험 구간 최적(0.425)과 방향도 반대다"),
                 "주의": "이 표는 진단이며 가중치 선정에 쓰지 않았다"}
        print(f"  가중치 훑기(진단) — 전진검증 최적 {best_w:.2f} 에서 {best_mse:.2f}, "
              f"고정 0.5 는 {half:.2f} (차이 {half - best_mse:+.2f})")
        print(f"  → 고정 0.5 를 유지한다. 최적은 이 4구간에서 역산한 값이다\n")
    for f in folds:
        f.pop("_pred", None)
    gc.collect()

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
    # 압축해서 담는다 — 압축하지 않으면 135 MB 로 GitHub 파일 상한(100 MB)을 넘는다.
    # 나무 300개·깊이 32 라 원본이 크다. compress=3 이면 Modeling/ 의 모델과 비슷한 크기다
    joblib.dump({"estimator": t["_model"], "features": cfg["features"]},
                HERE / "final_tree.joblib", compress=3)
    (HERE / "reproduction_metadata.json").write_text(json.dumps({
        "config": cfg, "seeds": sel["seeds"], "split_day": split_day,
        "rows_total": len(d), "rows_window_valid": len(w),
        "data_sha1": fp["data_sha1"],
        "config_sha1": fp["config_sha1"],
        "gru_predictions_fingerprint": fp_state,
        "gru_fingerprint_now": {k: v for k, v in fp.items() if k != "config_json"},
        "gru_predictions_note": (
            "verified = npz 의 자료 지문(ts·split·kW·y·달력 전 열의 값)과 설정 지문이 "
            "이 CSV·이 설정과 일치함을 확인했다. "
            "ts_only = 시각축만 담은 옛 지문이라 값·분할·설정은 확인하지 못했다. "
            "absent = 지문이 없는 npz 다 — 평가 행 위치(idx_*)만 대조했다. "
            "둘 다 다시 학습하면 verified 가 된다"),
        "forward": avg,
        "forward_weight_sweep": sweep,
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
