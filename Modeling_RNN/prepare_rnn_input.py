#!/usr/bin/env python3
"""순환신경망 입력 CSV 생성. 15분 계열과 달력 변수, 분할 표시를 한 파일로 만든다.

실행: python prepare_rnn_input.py --raw <okm_augumented_2021.csv> [--align <Modeling/final_input_data.csv>]
출력: rnn_input_data.csv

트리 모형과 달리 순환신경망은 **과거 96칸을 그대로 순서대로** 먹는다. 그래서 시차 변수를
미리 펼치지 않고 전력 계열 한 열과 달력 변수만 담는다. 창은 학습 시점에 만든다.

`--align` 을 주면 그 파일의 `split` 을 그대로 따라 행을 맞춘다. 정병근 모델링과
**같은 행·같은 분할**이 되어 수치를 직접 비교할 수 있다. 생략하면 날짜 규칙
(20210627 이전 train)으로 같은 분할을 재현한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STEP_MIN, PER_HOUR, PER_DAY = 15, 4, 96
TEST_FROM = 20210627          # 정병근 70:30 분할의 시험 시작일

# 달력 변수 — 2021년 공휴일과 장기 휴무(설연휴·하계휴무)
HOLIDAYS = ["2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01",
            "2021-05-05", "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16"]
LONG_SHUTDOWNS = [("2021-01-01", "2021-01-03"), ("2021-02-11", "2021-02-14"),
                  ("2021-07-31", "2021-08-08")]


def to_15min(raw: pd.DataFrame) -> pd.DataFrame:
    """한 행 = 한 시간, 네 열(15/30/45/60분)을 한 행 = 한 구간으로 편다.

    `시간` 열은 쓰지 않는다. 2021-07-13·07-15 두 날의 값이 정렬된 다른 값으로 덮여
    있기 때문이다. 날짜 안의 행 순서로 시(hour)를 만든다 — 정상 255일에서
    '행 순서 == 시간' 일치율이 1.0 이다.
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


def add_calendar(d: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(d["ts"])
    tod = idx.hour * PER_HOUR + idx.minute // STEP_MIN
    dw = idx.dayofweek
    d["tod_sin"] = np.sin(2 * np.pi * tod / PER_DAY)
    d["tod_cos"] = np.cos(2 * np.pi * tod / PER_DAY)
    d["dow_sin"] = np.sin(2 * np.pi * dw / 7)
    d["dow_cos"] = np.cos(2 * np.pi * dw / 7)
    hol = pd.to_datetime(HOLIDAYS).normalize()
    d["is_off"] = ((dw >= 5) | idx.normalize().isin(hol)).astype(int)
    shut = np.zeros(len(d), dtype=int)
    for a, b in LONG_SHUTDOWNS:
        shut |= ((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b) + pd.Timedelta(days=1))).astype(int)
    d["is_long_shutdown"] = shut
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="okm_augumented_2021.csv")
    ap.add_argument("--align", type=Path, default=None,
                    help="정병근 final_input_data.csv — 같은 행·분할로 맞춘다")
    ap.add_argument("--output", type=Path, default=HERE / "rnn_input_data.csv")
    a = ap.parse_args()

    raw = pd.read_csv(a.raw, encoding="utf-8-sig").reset_index(drop=True)
    d = add_calendar(to_15min(raw))
    d["y"] = d["kW"].shift(-1)                       # 정답 = 다음 15분 구간 전력
    d["date_key"] = d["ts"].dt.strftime("%Y%m%d").astype(int)

    if a.align and a.align.exists():
        bg = pd.read_csv(a.align)
        bg["ts"] = pd.to_datetime(bg["forecast_time"]) - pd.Timedelta(minutes=15)
        d = d.merge(bg[["ts", "split", "전력"]], on="ts", how="inner")
        bad = (d["y"] - d["전력"]).abs().max()
        assert bad < 1e-9, f"정답이 어긋난다 — 최대 차 {bad}"
        d = d.drop(columns=["전력"])
        print(f"정병근 분할에 맞춤 — 공통 {len(d):,}행")
    else:
        d = d[d["y"].notna()].copy()
        d["split"] = np.where(d["date_key"] < TEST_FROM, "train", "test")
        print(f"날짜 규칙으로 분할 (시험 {TEST_FROM}~) — {len(d):,}행")

    cols = ["ts", "date_key", "split", "kW", "y", "tod_sin", "tod_cos",
            "dow_sin", "dow_cos", "is_off", "is_long_shutdown"]
    out = d[cols].sort_values("ts").reset_index(drop=True)
    out.to_csv(a.output, index=False, encoding="utf-8-sig")
    print(f"저장: {a.output.name}  {len(out):,}행 × {len(cols)}열")
    print(f"  분할: {out['split'].value_counts().to_dict()}")
    print(f"  시험 구간: {out[out.split=='test'].date_key.min()} ~ {out[out.split=='test'].date_key.max()}")


if __name__ == "__main__":
    main()
