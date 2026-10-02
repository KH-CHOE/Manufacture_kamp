"""서빙이 보고 수치를 그대로 재현하는지 대조한다.

왜 필요한가 — 화면이 "앙상블"이라고 띄우는데 실제로는 다른 값을 내고 있으면 아무도 모른다.
`modeling.py` 가 학습하며 적어 둔 `Model/results.json` 과, `serving.py` 가 저장 파일만으로
낸 예측을 **같은 행에서** 비교한다.

판정 기준 (돌리기 전에 못박는다)
--------------------------------
  A 트리 단독    serving 의 트리 예측 MSE == results.json 의 et 시험 MSE
  B 순환 단독    serving 의 순환 예측 MSE == results.json 의 gru 시험 MSE
  C 앙상블      serving 의 결합 MSE     == results.json 의 ensemble 시험 MSE
  D 행 일치     serving 이 쓰는 시험 행수 == results.json 의 시험 행수

허용오차는 1e-3 이다. 그 이상 벌어지면 **저장 파일만으로는 보고 수치를 되살리지 못한다**는
뜻이므로 실패로 본다. 학습을 다시 하지 않고 저장물만으로 검사한다.

실행: python verify_serving.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C          # noqa: E402
import serving as S         # noqa: E402

TOL = 1e-3


def main() -> int:
    res_path = C.MODEL_DIR / "results.json"
    if not res_path.exists():
        print(f"✗ {res_path} 가 없다. python modeling.py 를 먼저 돌려라")
        return 1
    want = json.loads(res_path.read_text(encoding="utf-8"))["모델별"]

    S.load()
    st = S._state
    f = st["frame"]
    y = f["전력"].to_numpy(float)
    print(f"\n서빙 상태 — {'앙상블' if st['mode'] == 'ensemble' else '트리만'}"
          f" · 시험 {len(f):,}행 · 결합 가중치 {st['blend_weight']}")
    print(f"  모델 {st['model_path'].name}"
          + (f" + {st['net_path'].name}" if st.get("net_path") else ""))

    def mse(p):
        return float(((y - p) ** 2).mean())

    fails = []
    from artifacts import digest
    import hashlib
    manifest = json.loads((C.MODEL_DIR / 'manifest.json').read_text())
    if digest(C.OUT_DEFAULT) != manifest['data_sha256']:
        fails.append('평가한 자료와 현재 자료 불일치')
    if digest(res_path) != manifest['files'].get(res_path.name):
        fails.append('평가 결과 파일 불일치')
    row_hash = hashlib.sha256(f.forecast_time.astype(str).str.cat(sep='\n').encode()).hexdigest()
    if row_hash != manifest['test_rows_sha256']:
        fails.append('시험 행 시각 불일치')

    def check(name, got, ref_key):
        ref = (want.get(ref_key) or {}).get("시험", {}).get("MSE")
        if ref is None:
            print(f"  ✗ results.json에 {ref_key} 시험 결과가 없습니다")
            fails.append(name + " 보고 결과 없음")
            return
        ok = abs(got - ref) < TOL
        print(f"  {'✓' if ok else '✗'} {name:14s} {got:9.4f}  vs 보고 {ref:9.4f}"
              f"  차이 {abs(got - ref):.2e}")
        if not ok:
            fails.append(name)

    print(f"\n시험 구간 MSE 대조 (허용오차 {TOL})")
    check("트리 단독", mse(f["tree_prediction"].to_numpy(float)), "et")
    if "net_prediction" in f.columns:
        check("순환 단독", mse(f["net_prediction"].to_numpy(float)), "gru")
        check("앙상블", mse(f["prediction"].to_numpy(float)), "ensemble")
    else:
        print("  ! 순환신경망 예측이 없다 — 트리만으로 서빙 중이다")
        fails.append("앙상블 미구성")

    ref_rows = want.get("et", {}).get("시험", {}).get("rows")
    if ref_rows is not None:
        ok = len(f) == ref_rows
        print(f"\n  {'✓' if ok else '✗'} 시험 행수 {len(f):,} vs 보고 {ref_rows:,}")
        if not ok:
            fails.append("행수")

    print(f"\n{'─' * 54}")
    if fails:
        print(f"실패 {len(fails)}건: {', '.join(fails)}")
        print("→ 저장 파일만으로 보고 수치를 되살리지 못한다. 서빙을 신뢰할 수 없다.")
        return 1
    print("전부 일치 — 저장 파일만으로 보고 수치가 재현된다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
