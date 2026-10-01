#!/usr/bin/env python3
"""순환신경망(GRU) 재학습·평가. rnn_input_data.csv 와 selection.json 만 사용한다.

**시드를 시험 성적으로 고르지 않는다.** 저장 모델은 첫 시드로 고정하고, 보고·비교에 쓰는
예측은 세 시드의 평균이다. 가장 좋은 시드를 골라 저장하면 시험 구간이 선택에 쓰인 것이다.

실행: python train_rnn.py            (기본: 시드 3개, 직전 3개월 학습)
     python train_rnn.py --months 0  (전체 구간 학습으로 비교)
출력: final_rnn.pt, reproduced_metrics.csv, reproduced_test_predictions.csv,
      reproduction_metadata.json

탐색은 하지 않는다. 최종 구성은 전진검증으로 이미 선정되어 selection.json 에 고정돼 있다.

**시드를 3개 돌리는 이유.** 순환신경망은 난수 초기화에 따라 결과가 달라진다.
탐색 도중 한 설정이 시드 42 에서 평가 54.99, 시드 7 에서 135.30 으로 2.5배 벌어진 적이
있었다. 시드 하나로 낸 수치는 운일 수 있으므로 3개의 평균과 범위를 함께 보고한다.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

HERE = Path(__file__).resolve().parent
CAL = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_off", "is_long_shutdown"]
torch.set_num_threads(4)


class Net(nn.Module):
    """마지막 스텝의 은닉 상태에 달력 변수를 이어 붙여 한 값을 낸다"""

    def __init__(self, n_ch: int, n_cal: int, hidden: int):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_cal, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, c):
        o, _ = self.rnn(x)
        return self.head(torch.cat([o[:, -1, :], c], 1)).squeeze(-1)


def windows(d: pd.DataFrame, win: int, steps: int):
    """창 win 칸을 (steps, win//steps) 로 접어 순환 입력으로 만든다.

    삭제된 날(2021-07-13·15)을 건너뛰는 창은 제외한다. 시각이 연속하지 않는 창을
    학습에 넣으면 존재하지 않는 시간 흐름을 배우게 된다.
    """
    kw = d["kW"].to_numpy(np.float32)
    y = d["y"].to_numpy(np.float32)
    cal = d[CAL].to_numpy(np.float32)
    ts = pd.DatetimeIndex(d["ts"])
    ok = (ts[win - 1:] - ts[:len(ts) - win + 1]) == pd.Timedelta(minutes=15 * (win - 1))
    X = sliding_window_view(kw, win)[ok]
    idx = np.arange(win - 1, len(d))[ok]
    X = X.reshape(len(X), steps, win // steps)
    return X, y[idx], cal[idx], d["date_key"].to_numpy()[idx], idx


def run_seed(X, Y, C, day, split_day, months, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    start = 0
    if months:
        start = int((pd.Timestamp(str(split_day)) - pd.DateOffset(months=months)).strftime("%Y%m%d"))
    tr_m = (day >= start) & (day < split_day)
    te_m = day >= split_day

    mu, sd = X[tr_m].mean(), X[tr_m].std() + 1e-8
    ymu, ysd = Y[tr_m].mean(), Y[tr_m].std() + 1e-8
    Xn = ((X - mu) / sd).astype(np.float32)
    Yn = ((Y - ymu) / ysd).astype(np.float32)
    tr_i = np.where(tr_m)[0]
    cut = int(len(tr_i) * 0.85)
    xt = torch.from_numpy(np.ascontiguousarray(Xn[tr_i[:cut]]))
    yt = torch.from_numpy(Yn[tr_i[:cut]])
    ct = torch.from_numpy(C[tr_i[:cut]])
    net = Net(Xn.shape[2], C.shape[1], cfg["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    lossf = nn.MSELoss()

    def predict(sel):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(sel), 512):
                s = sel[i:i + 512]
                out.append(net(torch.from_numpy(np.ascontiguousarray(Xn[s])),
                               torch.from_numpy(C[s])).numpy())
        return np.concatenate(out) * ysd + ymu

    va = tr_i[cut:]
    best, state, bad, ep = float("inf"), None, 0, 0
    for ep in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), cfg["batch"]):
            k = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            lossf(net(xt[k], ct[k]), yt[k]).backward()
            opt.step()
        vl = float(np.mean((predict(va) - Y[va]) ** 2))
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(state)
    te_i = np.where(te_m)[0]
    pred = predict(te_i)
    e = Y[te_i] - pred
    return net, {
        "seed": seed, "epochs": ep + 1, "train_rows": int(tr_m.sum()),
        "test_rows": int(te_m.sum()),
        "MSE": float((e ** 2).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAE": float(np.abs(e).mean()),
        "R2": float(1 - (e ** 2).sum() / ((Y[te_i] - Y[te_i].mean()) ** 2).sum()),
    }, te_i, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=HERE / "rnn_input_data.csv")
    ap.add_argument("--selection", type=Path, default=HERE / "selection.json")
    ap.add_argument("--months", type=int, default=None, help="학습 구간 개월 (0=전체)")
    a = ap.parse_args()

    sel = json.loads(a.selection.read_text(encoding="utf-8"))
    cfg = sel["config"]
    months = cfg["train_months"] if a.months is None else a.months
    d = pd.read_csv(a.input, encoding="utf-8-sig")
    d["ts"] = pd.to_datetime(d["ts"])
    d = d[d["split"] != "boundary_excluded"].reset_index(drop=True)
    split_day = int(d.loc[d["split"] == "test", "date_key"].min())

    X, Y, C, day, idx = windows(d, cfg["window"], cfg["steps"])
    print(f"창 {cfg['window']}칸 → {cfg['steps']}스텝 × {cfg['window'] // cfg['steps']}채널 · "
          f"유효 창 {len(X):,}개 (불연속 창 제외)")
    print(f"학습 구간: {'전체' if not months else f'직전 {months}개월'} · 시험 {split_day}~\n")

    # **시드를 시험 성적으로 고르지 않는다.** 그렇게 고르면 시험 구간이 선택에 쓰인 것이다.
    # 저장 모델은 첫 시드로 고정하고, 보고 예측은 세 시드의 평균을 쓴다.
    rows, preds, keep_net, keep_idx = [], [], None, None
    for s in sel["seeds"]:
        net, m, te_i, pred = run_seed(X, Y, C, day, split_day, months, cfg, s)
        rows.append(m)
        preds.append(pred)
        if keep_net is None:
            keep_net, keep_idx = net, te_i
        print(f"  시드 {s:<6} 시험 MSE {m['MSE']:8.3f} · MAE {m['MAE']:6.3f} · "
              f"R² {m['R2']:.4f}  (에폭 {m['epochs']})", flush=True)

    mse = [r["MSE"] for r in rows]
    y_te = Y[keep_idx]
    ens = np.mean(preds, axis=0)
    ens_mse = float(((y_te - ens) ** 2).mean())
    ens_mae = float(np.abs(y_te - ens).mean())
    print(f"\n  시드 {len(rows)}개 평균 MSE {np.mean(mse):.3f} · 범위 [{min(mse):.3f}, {max(mse):.3f}]")
    print(f"  세 시드 예측 평균(시드 앙상블) MSE {ens_mse:.3f} · MAE {ens_mae:.3f}")

    pd.DataFrame(rows).to_csv(HERE / "reproduced_metrics.csv", index=False, encoding="utf-8-sig")
    out = d.iloc[idx[keep_idx]][["ts", "date_key"]].copy()
    out["actual"] = y_te
    for s, pr in zip(sel["seeds"], preds):
        out[f"pred_seed{s}"] = pr
    out["predicted"] = ens          # 보고·비교에 쓰는 값 = 세 시드 평균
    out.to_csv(HERE / "reproduced_test_predictions.csv", index=False, encoding="utf-8-sig")
    torch.save({"state_dict": keep_net.state_dict(), "config": cfg,
                "cal_features": CAL, "seed": sel["seeds"][0]}, HERE / "final_rnn.pt")
    (HERE / "reproduction_metadata.json").write_text(json.dumps({
        "config": cfg, "train_months": months, "split_day": split_day,
        "seeds": sel["seeds"], "per_seed": rows,
        "mean_MSE": float(np.mean(mse)), "MSE_range": [float(min(mse)), float(max(mse))],
        "seed_ensemble_MSE": ens_mse, "seed_ensemble_MAE": ens_mae,
        "saved_model_seed": sel["seeds"][0],
        "note": "저장 모델은 첫 시드로 고정한다. 시험 성적으로 시드를 고르면 시험 구간이 선택에 쓰인 것이다",
        "windows": int(len(X)), "python": platform.python_version(),
        "torch": torch.__version__,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: final_rnn.pt · reproduced_metrics.csv · "
          f"reproduced_test_predictions.csv · reproduction_metadata.json")


if __name__ == "__main__":
    main()
