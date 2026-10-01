#!/usr/bin/env python3
"""Add strictly historical 15-minute features to preprocssed_2."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "generated"
SOURCE = ROOT / "okm_augumented_2021_preprocssed_2.csv"
OUTPUT = ROOT / "okm_augumented_2021_preprocssed_3.csv"


def main():
    df = pd.read_csv(SOURCE)
    assert len(df) == 24480
    day = pd.to_datetime({"year": 2021, "month": df["m"], "day": df["d"]})
    day_index = df.groupby(["m", "d"], sort=False).cumcount()
    assert (day_index.groupby(day).max() == 95).all()
    assert (df["시간"].to_numpy() == (day_index // 4).to_numpy()).all()
    df["15분위치"] = (day_index % 4).astype("int8")
    stamp = day + pd.to_timedelta(df["시간"], unit="h") + pd.to_timedelta(df["15분위치"] * 15, unit="m")
    assert stamp.is_monotonic_increasing and stamp.is_unique
    # A new segment begins after each removed calendar day. Shifts and
    # rolling windows cannot carry power readings across these gaps.
    segment = stamp.diff().ne(pd.Timedelta(minutes=15)).cumsum()
    power = df["현재전력"]
    grouped = power.groupby(segment, sort=False)
    for lag in (1, 2, 4, 96):
        df[f"과거전력_{lag}칸"] = grouped.shift(lag)
    df["전력변화_15분"] = power - df["과거전력_1칸"]
    df["전력변화_60분"] = power - df["과거전력_4칸"]
    df["최근1시간_전력평균"] = grouped.transform(lambda s: s.rolling(4, min_periods=4).mean())
    df["최근1시간_전력최대"] = grouped.transform(lambda s: s.rolling(4, min_periods=4).max())
    # The shifted target still crosses the two deleted days in _2. Keep the
    # source target unchanged and exclude those two rows during modeling.
    original = pd.read_csv(SOURCE)
    pd.testing.assert_frame_equal(df[original.columns], original)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"created {OUTPUT}; rows={len(df):,}; new_features={len(df.columns)-len(original.columns)}")
    print(df[["과거전력_1칸", "과거전력_96칸"]].isna().sum().to_dict())


if __name__ == "__main__":
    main()
