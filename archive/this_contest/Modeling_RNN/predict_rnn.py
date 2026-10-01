#!/usr/bin/env python3
"""저장된 순환신경망으로 예측. 가공된 입력이 아니라 **15분 전력 계열**을 받는다.

python predict_rnn.py --input rnn_input_data.csv --output prediction_replay.csv
split 열이 있으면 기본적으로 test 만 예측한다. --all-rows 로 전체 행 예측.

트리 모형과 달리 창(96칸)을 만들 수 있어야 하므로, 예측 대상 앞에 최소 95칸의
연속한 과거가 CSV 안에 있어야 한다. 불연속 창은 건너뛴다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rnn import CAL, Net, windows

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE / "rnn_input_data.csv")
    p.add_argument("--model", type=Path, default=HERE / "final_rnn.pt")
    p.add_argument("--output", type=Path, default=HERE / "prediction_replay.csv")
    p.add_argument("--all-rows", action="store_true")
    a = p.parse_args()

    bundle = torch.load(a.model, map_location="cpu", weights_only=False)
    cfg = bundle["config"]
    d = pd.read_csv(a.input, encoding="utf-8-sig")
    d["ts"] = pd.to_datetime(d["ts"])
    d = d[d["split"] != "boundary_excluded"].reset_index(drop=True)

    X, Y, C, day, idx = windows(d, cfg["window"], cfg["steps"])
    split_day = int(d.loc[d["split"] == "test", "date_key"].min())
    # 정규화 통계는 학습 구간에서만 구한다 — 시험 정보 유입을 막는다
    start = int((pd.Timestamp(str(split_day)) - pd.DateOffset(months=cfg["train_months"]))
                .strftime("%Y%m%d")) if cfg["train_months"] else 0
    tr = (day >= start) & (day < split_day)
    mu, sd = X[tr].mean(), X[tr].std() + 1e-8
    ymu, ysd = Y[tr].mean(), Y[tr].std() + 1e-8

    net = Net(X.shape[2], C.shape[1], cfg["hidden"])
    net.load_state_dict(bundle["state_dict"])
    net.eval()
    sel = np.arange(len(X)) if a.all_rows else np.where(day >= split_day)[0]
    Xn = ((X - mu) / sd).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(sel), 512):
            s = sel[i:i + 512]
            out.append(net(torch.from_numpy(np.ascontiguousarray(Xn[s])),
                           torch.from_numpy(C[s])).numpy())
    pred = np.concatenate(out) * ysd + ymu

    r = d.iloc[idx[sel]][["ts", "date_key"]].copy()
    r["actual"] = Y[sel]
    r["predicted_power"] = pred
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(a.output, index=False, encoding="utf-8-sig")
    print(a.output, len(r))


if __name__ == "__main__":
    main()
