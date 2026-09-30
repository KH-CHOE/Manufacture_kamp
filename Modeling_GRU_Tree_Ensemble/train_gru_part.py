#!/usr/bin/env python3
"""앙상블의 **순환신경망 부분만** 별도 프로세스에서 학습한다. torch 만 import 한다.

실행: python train_gru_part.py            (train_ensemble.py 가 대신 호출한다)
출력: _gru_predictions.npz

**왜 별도 프로세스인가.** macOS 에서 sklearn(joblib 포크·OpenMP)을 먼저 돌린 뒤 같은
프로세스에서 torch Adam 이 도는 순간 libomp 가 두 번 적재되어 교착한다.
환경변수(`OMP_NUM_THREADS`·`KMP_DUPLICATE_LIB_OK`)만으로는 막히지 않았다 —
실측으로 `torch/optim/adam.py` 안에서 멈추는 것을 faulthandler 로 확인했다.
`Modeling_RNN/` 도 같은 이유로 sklearn 계열과 분리해 뒀다.

이 파일은 **sklearn 을 import 하지 않는다.** 그것이 분리의 핵심이다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)


class Net(nn.Module):
    def __init__(self, n_ch: int, n_cal: int, hidden: int):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_cal, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, c):
        o, _ = self.rnn(x)
        return self.head(torch.cat([o[:, -1, :], c], 1)).squeeze(-1)


def windows(d, win, steps, cal):
    kw = d["kW"].to_numpy(np.float32)
    ts = pd.DatetimeIndex(d["ts"])
    ok = (ts[win - 1:] - ts[:len(ts) - win + 1]) == pd.Timedelta(minutes=15 * (win - 1))
    X = sliding_window_view(kw, win)[ok].reshape(-1, steps, win // steps)
    i = np.arange(win - 1, len(d))[ok]
    return X, i, d[cal].to_numpy(np.float32)[i]


def train_one(X, Y, C, day, start, split_day, hi, g, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    ct = torch.from_numpy(C[ti[:cut]])
    net = Net(Xn.shape[2], C.shape[1], g["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=g["lr"])
    lossf = nn.MSELoss()

    def pred(sel):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(sel), 512):
                s = sel[i:i + 512]
                out.append(net(torch.from_numpy(np.ascontiguousarray(Xn[s])),
                               torch.from_numpy(C[s])).numpy())
        return np.concatenate(out) * ysd + ymu

    va = ti[cut:]
    best, state, bad, ep = float("inf"), None, 0, 0
    for ep in range(g["epochs"]):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), g["batch"]):
            k = perm[i:i + g["batch"]]
            opt.zero_grad()
            lossf(net(xt[k], ct[k]), yt[k]).backward()
            opt.step()
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=HERE / "ensemble_input_data.csv")
    ap.add_argument("--selection", type=Path, default=HERE / "selection.json")
    ap.add_argument("--output", type=Path, default=HERE / "_gru_predictions.npz")
    ap.add_argument("--save-model", type=Path, default=HERE / "final_gru.pt")
    a = ap.parse_args()

    sel = json.loads(a.selection.read_text(encoding="utf-8"))
    cfg = sel["config"]
    g = cfg["gru"]
    d = pd.read_csv(a.input, encoding="utf-8-sig")
    d["ts"] = pd.to_datetime(d["ts"])
    d = d[d["split"] != "boundary_excluded"].reset_index(drop=True)
    X, idx, C = windows(d, g["window"], g["steps"], cfg["calendar"])
    w = d.iloc[idx].reset_index(drop=True)
    Y = w["y"].to_numpy(np.float32)
    day = w["date_key"].to_numpy()
    split_day = int(d.loc[d["split"] == "test", "date_key"].min())

    print(f"순환신경망 부분 — 창 유효 {len(w):,}행 · 은닉 {g['hidden']} · 시드 {sel['seeds']}")
    out, first_net = {}, None
    jobs = [(lo, hi, str(lo)) for lo, hi in sel["forward_folds"]] + [(split_day, None, "test")]
    for split, hi, label in jobs:
        if (day < split).sum() < 1500:
            continue
        start = int((pd.Timestamp(str(split)) - pd.DateOffset(months=g["train_months"]))
                    .strftime("%Y%m%d")) if g["train_months"] else 0
        ps, eps = [], []
        for s in sel["seeds"]:
            net, ei, p, ep = train_one(X, Y, C, day, start, split, hi, g, s)
            ps.append(p); eps.append(ep)
            if label == "test" and first_net is None:
                first_net = net
        out[f"pred_{label}"] = np.mean(ps, axis=0)
        out[f"idx_{label}"] = ei
        out[f"eps_{label}"] = np.array(eps)
        print(f"  {label:>6} MSE {float(((Y[ei] - out[f'pred_{label}']) ** 2).mean()):8.3f}"
              f" · 에폭 {eps}", flush=True)

    # 지문 — 어느 CSV·어느 설정으로 낸 예측인지 남긴다. `train_ensemble.py` 가 대조한다.
    # 없으면 길이와 행 위치만 맞춰 보게 되고, 자료 자체가 바뀐 경우를 못 잡는다.
    ts_txt = w["ts"].astype(str).str.cat(sep="|")
    out["fp_ts_sha1"] = np.array(hashlib.sha1(ts_txt.encode()).hexdigest())
    out["fp_rows"] = np.array(len(w))
    out["fp_ts_first"] = np.array(str(w["ts"].iloc[0]))
    out["fp_ts_last"] = np.array(str(w["ts"].iloc[-1]))
    out["fp_config"] = np.array(json.dumps(
        {"hidden": g["hidden"], "window": g["window"], "steps": g["steps"],
         "train_months": g["train_months"], "seeds": sel["seeds"],
         "calendar": cfg["calendar"]}, sort_keys=True, ensure_ascii=False))

    np.savez(a.output, **out)
    if first_net is not None:
        torch.save({"state_dict": first_net.state_dict(), "config": g,
                    "calendar": cfg["calendar"], "seed": sel["seeds"][0]}, a.save_model)
    print(f"저장: {a.output.name} · {a.save_model.name}")


if __name__ == "__main__":
    main()
