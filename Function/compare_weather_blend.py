"""결합에 기상 미사용 트리를 넣은 판단을 수치로 남긴다.

`modeling.py` 가 저장한 구간별 예측(`_pred_trees.npz`·`_pred_nets.npz`)으로
  ① ExtraTrees(기상 미사용) + GRU   — 서빙하는 결합
  ② ExtraTrees + 기상 + GRU         — 비교
두 결합을 **같은 규칙**(config.BLEND_GRID 를 훑어 config.BLEND_SELECT 기준으로 비율 선택)으로 계산한다.
모델을 다시 학습하지 않는다. 결과는 `Model/weather_blend_comparison.json`.

실행: python Function/compare_weather_blend.py   (먼저 python Function/modeling.py)
"""
from __future__ import annotations

import json

import numpy as np

import config as C
import modeling as M


def main() -> int:
    d = M.load(C.OUT_DEFAULT)
    rows = d[M.usable(d)].reset_index(drop=True)
    folds = M.folds_from_data(rows, C.FORWARD_FOLDS, C.FORWARD_FOLD_DAYS)
    T, N = np.load(C.HERE / "_pred_trees.npz"), np.load(C.HERE / "_pred_nets.npz")
    ridx, day, y = N["row_index"], rows["date_key"].to_numpy(), rows["y"].to_numpy()

    def parts(tree: str) -> dict:
        out = {}
        for lo, hi, label in M.jobs(rows, folds):
            m = (day >= lo) if hi is None else ((day >= lo) & (day < hi))
            tr = np.where(m)[0]
            if not np.array_equal(tr, ridx[N[f"gru|idx|{label}"]]):
                raise ValueError(f"트리·GRU 예측 행이 다르다: {label}")
            out[label] = (y[tr], T[f"{tree}|{label}"], N[f"gru|{label}"])
        return out

    labels = [str(lo) for lo, _ in folds]
    out = {"규칙": f"트리 비중 {C.BLEND_GRID[0]}~{C.BLEND_GRID[-1]}, 선정 기준 {C.BLEND_SELECT}",
           "행기준": {"공통행": len(rows), "시험행": int((rows["split"] == "test").sum())}}
    for key, name in (("et", "ExtraTrees(기상 미사용) + GRU — 서빙"), ("et_wx", "ExtraTrees + 기상 + GRU — 비교")):
        s = M.search_blend(parts(key), labels)
        out[key] = {"이름": name, **{k: v for k, v in s.items() if k != "표"}}
        print(f"{name:34s} 비율 {s['선정_트리비중']} · 시험 {s['선정_시험']} · 전진 {s['선정_전진평균']}")
    out["차이_기상판_빼기_미사용판"] = {
        "시험": round(out["et_wx"]["선정_시험"] - out["et"]["선정_시험"], 4),
        "전진검증": round(out["et_wx"]["선정_전진평균"] - out["et"]["선정_전진평균"], 4)}
    (C.MODEL_DIR / "weather_blend_comparison.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장: Model/weather_blend_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
