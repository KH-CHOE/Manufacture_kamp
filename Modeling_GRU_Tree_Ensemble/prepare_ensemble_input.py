#!/usr/bin/env python3
"""앙상블 입력 CSV 생성. 원본 한 개만 읽어 **자립적으로** 26변수를 만든다.

실행: python prepare_ensemble_input.py --raw <okm_augumented_2021.csv>
     python prepare_ensemble_input.py --raw ... --verify ../Modeling/final_input_data.csv
출력: ensemble_input_data.csv

`--verify` 를 주면 정병근 입력 CSV 와 **같은 이름 변수의 값이 일치하는지 전 행 대조**한다.
실험에서는 그의 CSV 에서 가져다 썼으므로, 이 파일이 그것을 재현하는지 확인해야
수치가 이어진다. 대조는 검증용이고 생성 자체는 원본만으로 된다.

## 왜 이 26변수인가

세 단계 실험으로 골랐다(`evidence/01_feature_selection.json`).

  ① 시각 원값이 결정적이다 — 트리는 `변수 ≤ 임계값` 으로만 자르므로 `시간 >= 7`(오전 시동)을
     한 번에 만들려면 원값이 필요하다. 순환 인코딩(sin/cos)만으로는 두 변수를 여러 번 잘라야
     그 경계가 생긴다. 우리 21변수에 원값 3개를 더하니 전진검증 46.58 → 41.75
  ② 최근 2·4시간 통계가 도움된다 — 빼면 41.61 → 43.29
  ③ 기상 4종과 1주 전(lag672)은 **빼는 편이 낫다** — 기상은 직전 전력이 이미 담고 있고,
     1주 전은 달력이 충분하면 중복이다
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STEP_MIN, PER_HOUR, PER_DAY, PER_WEEK = 15, 4, 96, 672
TEST_FROM = 20210627          # 정병근 70:30 분할의 시험 시작일

HOLIDAYS = ["2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01",
            "2021-05-05", "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16"]
LONG_SHUTDOWNS = [("2021-01-01", "2021-01-03"), ("2021-02-11", "2021-02-14"),
                  ("2021-07-31", "2021-08-08")]

# 트리에 주는 26변수
FEATS = [
    # 시차값 (1주 전은 제외 — 실험에서 해로웠다)
    "kw_lag1", "kw_lag2", "kw_lag3", "kw_lag4", "kw_lag8", "kw_lag96",
    # 변화·이동통계
    "kw_d1", "kw_d2", "kw_roll4", "kw_roll16", "kw_std4",
    # 달력 — 순환 인코딩과 **원값을 함께** 준다
    "tod_sin", "tod_cos", "dow", "is_weekend", "is_day_shift",
    "시간", "15분위치",
    # 최근 2·4시간 통계
    "전력변화_2시간", "전력변화_4시간", "최근2시간_전력평균", "최근2시간_전력최대",
    "최근4시간_전력평균", "최근4시간_전력최대", "과거전력_16칸",
]
# 순환신경망에 주는 달력 (계열은 kW 열에서 창을 만든다)
CAL = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_off", "is_long_shutdown"]
# 대조할 이름 대응 (우리 이름 → 정병근 이름)
# 우리 `kw_lagN` 은 그의 `과거전력_(N−1)칸` 에 대응한다(우리 lag1 == 그의 현재전력).
# 이름이 어긋나 혼동하기 쉬우므로 대조표에는 정의가 같은 것만 넣는다.
VERIFY_MAP = {"kw_lag1": "현재전력", "kw_lag2": "과거전력_1칸", "kw_lag4": "과거전력_3칸",
              "시간": "시간", "15분위치": "15분위치",
              "전력변화_2시간": "전력변화_2시간", "전력변화_4시간": "전력변화_4시간",
              "최근2시간_전력평균": "최근2시간_전력평균", "최근2시간_전력최대": "최근2시간_전력최대",
              "최근4시간_전력평균": "최근4시간_전력평균", "최근4시간_전력최대": "최근4시간_전력최대",
              "과거전력_16칸": "과거전력_16칸"}


def to_15min(raw: pd.DataFrame) -> pd.DataFrame:
    """넓은 형태를 15분 해상도로 편다.

    `시간` 열은 쓰지 않는다 — 2021-07-13·07-15 두 날이 정렬된 다른 값으로 덮여 있다.
    날짜 안의 행 순서로 시(hour)를 만든다(정상 255일에서 일치율 1.0).
    """
    d = raw.copy()
    d["_h"] = d.groupby("날짜").cumcount()
    keep = ["날짜", "_h", "생산량", "기온", "풍속", "습도", "강수량"]
    long = d.melt(id_vars=keep, value_vars=["15분", "30분", "45분", "60분"],
                  var_name="_seg", value_name="kW")
    long["_q"] = long["_seg"].map({"15분": 0, "30분": 1, "45분": 2, "60분": 3})
    long["ts"] = (pd.to_datetime(long["날짜"].astype(str), format="%Y%m%d")
                  + pd.to_timedelta(long["_h"] * 60 + long["_q"] * STEP_MIN, unit="m"))
    long = long.sort_values("ts").drop(columns=["_seg"]).reset_index(drop=True)
    assert len(long) == len(raw) * PER_HOUR, "전개 행수가 원본 × 4 여야 한다"
    assert long["ts"].is_monotonic_increasing and long["ts"].is_unique
    assert long["ts"].diff().dropna().nunique() == 1, "간격이 전부 15분이어야 한다"
    return long


def build(raw: pd.DataFrame) -> pd.DataFrame:
    d = to_15min(raw)
    idx = pd.DatetimeIndex(d["ts"])
    s = d["kW"].astype(float)

    d["y"] = s.shift(-1)                               # 정답 = 다음 15분 구간
    for n in (1, 2, 3, 4, 8, PER_DAY):
        d[f"kw_lag{n}"] = s.shift(n - 1)               # lag1 = 예측 시점 관측값
    # 정병근 `과거전력_N칸` 은 원계열[ts − N칸] 이다. 그의 정의에 맞춘다
    d["과거전력_16칸"] = s.shift(16)
    d["kw_d1"] = s.diff()
    d["kw_d2"] = s.diff(2)
    d["kw_roll4"] = s.rolling(4).mean()
    d["kw_roll16"] = s.rolling(16).mean()
    d["kw_std4"] = s.rolling(4).std()

    # ── 달력 — 원값과 순환 인코딩을 함께 ──
    tod = idx.hour * PER_HOUR + idx.minute // STEP_MIN
    dw = idx.dayofweek
    d["시간"] = idx.hour
    d["15분위치"] = idx.minute // STEP_MIN
    d["tod_sin"] = np.sin(2 * np.pi * tod / PER_DAY)
    d["tod_cos"] = np.cos(2 * np.pi * tod / PER_DAY)
    d["dow"] = dw
    d["dow_sin"] = np.sin(2 * np.pi * dw / 7)
    d["dow_cos"] = np.cos(2 * np.pi * dw / 7)
    d["is_weekend"] = (dw >= 5).astype(int)
    d["is_day_shift"] = ((idx.hour >= 8) & (idx.hour < 20)).astype(int)
    hol = pd.to_datetime(HOLIDAYS).normalize()
    d["is_off"] = ((dw >= 5) | idx.normalize().isin(hol)).astype(int)
    shut = np.zeros(len(d), int)
    for a, b in LONG_SHUTDOWNS:
        shut |= ((idx >= pd.Timestamp(a))
                 & (idx <= pd.Timestamp(b) + pd.Timedelta(days=1))).astype(int)
    d["is_long_shutdown"] = shut

    # ── 최근 2·4시간 통계 ── 모두 예측 시점(포함)까지만 본다
    d["전력변화_2시간"] = s - s.shift(8)
    d["전력변화_4시간"] = s - s.shift(16)
    d["최근2시간_전력평균"] = s.rolling(8).mean()
    d["최근2시간_전력최대"] = s.rolling(8).max()
    d["최근4시간_전력평균"] = s.rolling(16).mean()
    d["최근4시간_전력최대"] = s.rolling(16).max()

    d["date_key"] = idx.strftime("%Y%m%d").astype(int)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--verify", type=Path, default=None,
                    help="정병근 final_input_data.csv — 같은 이름 변수 값을 대조한다")
    ap.add_argument("--align", type=Path, default=None,
                    help="정병근 final_input_data.csv — 그 행·분할에 맞춘다(수치 재현용)")
    ap.add_argument("--output", type=Path, default=HERE / "ensemble_input_data.csv")
    a = ap.parse_args()

    raw = pd.read_csv(a.raw, encoding="utf-8-sig").reset_index(drop=True)
    d = build(raw)

    if a.verify and a.verify.exists():
        bg = pd.read_csv(a.verify)
        bg["ts"] = pd.to_datetime(bg["forecast_time"]) - pd.Timedelta(minutes=15)
        m = d.merge(bg, on="ts", how="inner", suffixes=("", "_bg"))
        print(f"대조 — 공통 {len(m):,}행")
        bad = []
        for ours, theirs in VERIFY_MAP.items():
            col = theirs if theirs not in m.columns or theirs != ours else theirs + "_bg"
            if col not in m.columns:
                print(f"  ? {ours:18} 상대 열 없음")
                continue
            diff = (m[ours].astype(float) - m[col].astype(float)).abs()
            n = int((diff > 1e-6).sum())
            print(f"  {'✓' if n == 0 else '✗'} {ours:18} 최대차 {diff.max():.3g}"
                  + (f" · 불일치 {n:,}행" if n else ""))
            if n:
                bad.append(ours)
        print(f"  → {'전부 일치' if not bad else f'불일치 {bad}'}")

    d = d[d["y"].notna()].copy()
    if a.align and a.align.exists():
        bgs = pd.read_csv(a.align)
        bgs["ts"] = pd.to_datetime(bgs["forecast_time"]) - pd.Timedelta(minutes=15)
        d = d.merge(bgs[["ts", "split", "전력"]], on="ts", how="inner")
        bad = (d["y"] - d["전력"]).abs().max()
        assert bad < 1e-9, f"정답이 어긋난다 — 최대 차 {bad}"
        d = d.drop(columns=["전력"])
        print(f"\n정병근 행·분할에 맞춤 — 공통 {len(d):,}행")
    else:
        d["split"] = np.where(d["date_key"] < TEST_FROM, "train", "test")
    cols = ["ts", "date_key", "split", "kW", "y"] + FEATS + ["dow_sin", "dow_cos",
                                                             "is_off", "is_long_shutdown"]
    cols = list(dict.fromkeys(cols))
    out = d[cols].sort_values("ts").reset_index(drop=True)
    out.to_csv(a.output, index=False, encoding="utf-8-sig")
    print(f"\n저장: {a.output.name}  {len(out):,}행 × {len(cols)}열")
    print(f"  분할 {out['split'].value_counts().to_dict()}")
    print(f"  트리 변수 {len(FEATS)}개 · 순환신경망 달력 {len(CAL)}개")


if __name__ == "__main__":
    main()
