"""가짜 '2022년' 자료를 만들어 전처리가 **다른 문제**에도 돌아가는지 시험한다.

왜 필요한가 — 2021 자료에서 잘 돌아가는 것은 증명이 아니다. 날짜를 코드에 박았는지
아닌지는 **문제의 위치와 종류를 바꿔 봐야** 드러난다. 그래서 2021 자료를 2022로 옮기고
다음을 일부러 심는다.

  ① 시 열이 깨진 날을 **다른 날짜로** 옮긴다 (03-07, 11-22)  — 2021 의 07-13·07-15 가 아니다
  ② 날짜 공백 — 한 주를 통째로 뺀다 (05-09 ~ 05-15)
  ③ 하루가 24행이 아닌 날 — 행을 몇 개 지운다 (08-03)
  ④ 결측을 **다른 열에** 심는다 (기온·습도)
  ⑤ 이름이 다른 파생 열을 더한다 (`설비효율` = Σ전력 ÷ 생산량)
  ⑥ 달력 재표기 열을 더한다 (`요율구간` = 월의 함수)

실행: python make_test_year.py --raw <2021원자료> --out test_year_2022.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build(raw: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    d = raw.copy()
    # 2021 → 2022 로 옮긴다(날짜만 바꾼다)
    dt = pd.to_datetime(d["날짜"].astype(str), format="%Y%m%d") + pd.DateOffset(years=1)
    d["날짜"] = dt.dt.strftime("%Y%m%d").astype(int)
    d["m"], d["d"] = dt.dt.month, dt.dt.day
    d["day"] = dt.dt.dayofweek + 1
    plan: dict = {}

    # ① 시 열을 다른 날짜에서 깨뜨린다
    broken = [20220307, 20221122]
    broken = [b for b in broken if b in set(d["날짜"])]
    if not broken:                       # 자료 범위에 없으면 가진 날 중 둘을 고른다
        broken = sorted(d["날짜"].unique())[[40, 120][0]:][:1]
    for b in broken:
        m = d["날짜"] == b
        d.loc[m, "시간"] = np.sort(rng.integers(60, 200, size=int(m.sum())))
    plan["시열_깨뜨린날"] = [int(b) for b in broken]

    # ② 한 주를 통째로 뺀다 → 날짜 공백
    gap = pd.date_range("2022-05-09", "2022-05-15")
    gk = set(gap.strftime("%Y%m%d").astype(int))
    before = len(d)
    d = d[~d["날짜"].isin(gk)].copy()
    plan["지운주"] = {"범위": "2022-05-09~15", "지운행": before - len(d)}

    # ③ 하루를 24행 미만으로 만든다
    short = 20220803
    if short in set(d["날짜"]):
        idx = d.index[d["날짜"] == short][[3, 7, 11]]
        d = d.drop(index=idx)
        plan["짧은날"] = {"날짜": short, "남은행": int((d["날짜"] == short).sum())}

    # ④ 결측을 다른 열에 심는다
    for col, n in (("기온", 5), ("습도", 2)):
        pick = rng.choice(d.index, size=n, replace=False)
        d.loc[pick, col] = np.nan
    plan["심은결측"] = {"기온": 5, "습도": 2}

    # ⑤ 이름이 다른 파생 열 (전력에서 역산 가능)
    P = ["15분", "30분", "45분", "60분"]
    s = d[P].sum(axis=1)
    d["설비효율"] = s / d["생산량"].replace(0, np.nan)
    plan["심은파생열"] = "설비효율 = Σ전력 ÷ 생산량"

    # ⑥ 달력 재표기 열
    d["요율구간"] = d["m"].map(lambda x: 1 if x in (6, 7, 8) else (2 if x in (1, 2, 12) else 3))
    plan["심은달력열"] = "요율구간 = 월의 함수"

    d = d.drop(columns=["공장인원"])      # 2022 자료에는 이 열이 없다고 가정
    plan["뺀열"] = ["공장인원"]
    return d.reset_index(drop=True), plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path,
                    default=Path(__file__).resolve().parents[2] / "datasets" / "raw" / "okm_augumented_2021.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "test_year_2022.csv")
    a = ap.parse_args()
    raw = pd.read_csv(a.raw, encoding="utf-8-sig")
    d, plan = build(raw)
    d.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"저장: {a.out.name}  {len(d):,}행 × {len(d.columns)}열")
    print("심은 문제")
    for k, v in plan.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
