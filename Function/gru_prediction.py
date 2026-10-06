"""저장 GRU의 시드별 예측을 평균한다. OpenMP 충돌 방지를 위해 별도 프로세스에서 실행한다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import settings as C          # noqa: E402  (torch·sklearn 를 import 하지 않는다)
import model_training as M        # noqa: E402  (sklearn 은 함수 안에서만 import 된다)


def predict(data_path: Path, model_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(시험 행 위치, 시드평균 예측, 설명) 을 돌려준다."""
    import torch
    from torch import nn
    from numpy.lib.stride_tricks import sliding_window_view
    torch.set_num_threads(2)

    b = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dicts" not in b or "stats" not in b:
        raise SystemExit(
            f"✗ {model_path.name} 이 예전 형식이다(시드 하나·표준화 통계 없음).\n"
            f"  python model_training.py --models gru  로 다시 학습해라")

    d = M.load(data_path)
    rows = d[M.usable(d)].sort_values("ts").reset_index(drop=True)
    test_pos = np.where(rows["split"].to_numpy() == "test")[0]

    g, st = b["config"], b["stats"]
    win, steps = g["window"], g["steps"]
    X = M.net_windows(d, M.usable(d), win, steps)
    idx = np.arange(len(rows))
    CAL = rows[b["calendar"]].to_numpy(np.float32)[idx]
    Xn = ((X - st["mu"]) / st["sd"]).astype(np.float32)

    pos = {int(v): i for i, v in enumerate(idx)}
    sel = np.array([pos[int(v)] for v in test_pos if int(v) in pos], int)
    if len(sel) != len(test_pos):
        raise SystemExit(f"✗ 창을 만들 수 없는 시험 행이 {len(test_pos) - len(sel)}개 있다")

    class Net(nn.Module):
        def __init__(self, n_ch, n_cal, hidden):
            super().__init__()
            rnn = nn.GRU
            self.rnn = rnn(n_ch, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + n_cal, 64), nn.ReLU(),
                                      nn.Linear(64, 1))

        def forward(self, x, c):
            o, _ = self.rnn(x)
            return self.head(torch.cat([o[:, -1, :], c], 1)).squeeze(-1)

    # 시드별로 예측한 뒤 평균한다. 보고 수치가 시드평균이므로 하나만 쓰면 다른 값이 된다
    outs = []
    for sd_ in b["state_dicts"]:
        net = Net(Xn.shape[2], CAL.shape[1], g["hidden"])
        net.load_state_dict(sd_)
        net.eval()
        chunk = []
        with torch.no_grad():
            for i in range(0, len(sel), 512):
                k = sel[i:i + 512]
                chunk.append(net(torch.from_numpy(np.ascontiguousarray(Xn[k])),
                                 torch.from_numpy(CAL[k])).numpy())
        outs.append(np.concatenate(chunk) * st["ysd"] + st["ymu"])
    info = {"kind": b["kind"], "seeds": list(b["seeds"]),
            "hidden": g["hidden"], "window": g["window"],
            "train_start": st["train_start"], "train_end": st["train_end"]}
    return test_pos, np.mean(outs, axis=0), info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=C.OUT_DEFAULT)
    ap.add_argument("--model", type=Path, default=C.MODEL_DIR / "model_gru.pt")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not a.model.exists():
        print(f"✗ 모델이 없다: {a.model}")
        return 1
    where, pred, info = predict(a.data, a.model)
    np.savez(a.out, where=where, pred=pred, info=json.dumps(info, ensure_ascii=False))
    print(f"{len(pred)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
