"""자원 최적화 — 시간당 고정 생산량에 필요한 최소 인원과 비용.

`Dashboard/optimization/engine.py` 를 이쪽으로 옮긴 것이다. 옮기면서 **하드코딩 둘을 걷어냈다.**

  ① 원자료에서 `2021-07-13`·`2021-07-15` 를 날짜로 지정해 지우고 있었다.
     → `preprocessing.fix_hour()` 를 그대로 불러 쓴다. 시 열이 깨진 날을 규칙으로 복구하고,
       복구가 안 되는 날만 버린다. **모델링과 같은 규칙을 쓴다** — 두 벌로 갈라지지 않는다
  ② 공휴일이 `config.json` 의 `public_holidays_2021` 에 따로 있었다.
     → `calendar_YYYY.json` 한 곳에서만 읽는다. 없으면 주말만으로 본다

무엇을 계산하는가
----------------
어떤 정시의 생산량 Q 를 내려면 최소 몇 명이 필요했는가를 **과거 기록에서 찾는다.**
같은 시각·같은 날 유형이면서 그날보다 **앞선 날짜**, 생산량이 Q 이상인 기록 중
인원이 가장 적었던 날을 고른다. 학습한 인과 모형이 아니라 **과거 사례 기준점**이다.

`공장인원` 을 쓴다 — 모델 입력에서는 뺀 열이다
---------------------------------------------
모델링에서는 `공장인원 = 생산량 ÷ Σ전력` 이라 **타깃을 역산할 수 있어** 뺐다.
여기서는 다르다. 맞히려는 대상이 전력이 아니라 **인원 그 자체**이고, 과거에 실제로
몇 명이 있었는지를 조회할 뿐이다. 예측에 쓰는 것이 아니므로 누수가 아니다.
"""
from __future__ import annotations

import calendar as _calendar
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import preprocessing as P

HERE = Path(__file__).resolve().parent
TARIFF = json.loads((HERE / "tariff.json").read_text(encoding="utf-8"))
LABELS = {"off": "경부하", "mid": "중간부하", "peak": "최대부하"}


def _holidays() -> set[str]:
    """공휴일은 달력 자료 한 곳에서만 읽는다. 없으면 빈 집합(주말만 휴일)."""
    p = C.CALENDAR_DEFAULT
    if not p.exists():
        return set()
    return set(json.loads(p.read_text(encoding="utf-8")).get("holidays", []))


HOLIDAYS = _holidays()


def tariff_at(ts, rates=None) -> tuple[str, str, float]:
    """그 시각의 계절·부하대·요율. 2021 요율표 기준."""
    season = ("summer" if ts.month in (6, 7, 8)
              else "winter" if ts.month in (11, 12, 1, 2) else "spring_autumn")
    h = ts.hour + ts.minute / 60 + ts.second / 3600
    band = next(b for b, iv in TARIFF["time_bands"][season].items()
                if any(s <= h < e for s, e in iv))
    # 2021 요율 관행 — 일요일·공휴일은 경부하, 토요일의 최대부하는 중간부하로 내린다
    if ts.weekday() == 6 or ts.strftime("%Y-%m-%d") in HOLIDAYS:
        band = "off"
    elif ts.weekday() == 5 and band == "peak":
        band = "mid"
    return season, band, (TARIFF["rates"] if rates is None else rates)[season][band]


def wage_at(hour: int) -> tuple[float, bool]:
    night = hour >= TARIFF["night_start"] or hour < TARIFF["night_end"]
    return TARIFF["day_wage"] * (TARIFF["night_multiplier"] if night else 1), night


def day_type(ts) -> str:
    return "휴일" if ts.weekday() >= 5 or ts.strftime("%Y-%m-%d") in HOLIDAYS else "평일"


class StaffingOptimizer:
    """과거 기록에서 찾은 최소 인원 기준점."""

    def __init__(self, raw_path: Path | str | None = None):
        raw_path = Path(raw_path) if raw_path else C.RAW_DEFAULT
        raw = pd.read_csv(raw_path, encoding="utf-8-sig")

        # **모델링과 같은 규칙으로 시각을 만든다.** 날짜를 적어 지우지 않는다.
        log = P.Log()
        log.add = lambda *a, **k: None
        with redirect_stdout(io.StringIO()):
            d = P.check_schema(raw, log)
            d = P.fix_hour(d, log)
            d = P.build_hourly_axis(d, log)

        need = ["생산량", "공장인원"]
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise ValueError(f"자원 최적화에 필요한 열이 없다: {missing}")

        self.raw = d.rename(columns={"_시각": "timestamp"}).sort_values("timestamp")
        self.raw["day_type"] = self.raw["timestamp"].map(day_type)
        staff = pd.to_numeric(self.raw["공장인원"], errors="coerce")
        prod = pd.to_numeric(self.raw["생산량"], errors="coerce")
        # 생산이 0 이면 인원 0 도 정상이고, 생산이 있으면 인원이 0 보다 커야 한다
        self.raw["staff_valid"] = (np.isfinite(staff) & staff.ge(0)
                                   & (prod.eq(0) | staff.gt(0)))
        self.raw["공장인원"], self.raw["생산량"] = staff, prod

    def hourly_interval(self, forecast_time, target_time, prediction, *,
                        unit_price=None, day_wage=None, energy_rate=None,
                        rate_table=None, base_rate=None, billing_peak=None) -> dict:
        start, end = pd.Timestamp(forecast_time), pd.Timestamp(target_time)
        if end - start != pd.Timedelta(minutes=15) or start.minute != 45 or end.minute != 0:
            raise ValueError("정시 직전 45분에 생성한 다음 15분 예측이 필요합니다.")
        raw_hour = end
        sel = self.raw.loc[self.raw.timestamp == raw_hour]
        if len(sel) != 1:
            raise ValueError("해당 정시의 시간당 원자료가 없습니다. 다른 시간대를 선택해주세요.")
        row = sel.iloc[0]
        hourly_q = float(row["생산량"])
        if not math.isfinite(hourly_q) or hourly_q < 0 or not math.isfinite(float(prediction)):
            raise ValueError("생산량과 예측 전력은 유효한 값이어야 합니다.")

        # 같은 시각·같은 날 유형 · **그날보다 앞선 날짜** · 생산량이 Q 이상인 기록
        hist = self.raw.loc[(self.raw.timestamp < end.normalize()) & self.raw.staff_valid
                            & (self.raw.timestamp.dt.hour == end.hour)
                            & (self.raw.day_type == row.day_type)
                            & np.isfinite(self.raw["생산량"])
                            & self.raw["생산량"].ge(hourly_q)].copy()
        matches = len(hist)
        if matches:
            hist["integer_staff"] = hist["공장인원"].map(lambda n: math.ceil(float(n)))
            w = hist.sort_values(["integer_staff", "생산량", "timestamp"],
                                 ascending=[True, True, False]).iloc[0]
            recommended = int(w.integer_staff)
            reference = {"date": w.timestamp.strftime("%Y-%m-%d"), "hour": end.hour,
                         "hourlyProduction": float(w["생산량"]),
                         "rawStaff": round(float(w["공장인원"]), 4), "staff": recommended}
        else:
            recommended, reference = None, None

        season, band, tariff_rate = tariff_at(end, rate_table)
        rate = tariff_rate if energy_rate is None else float(energy_rate)
        base_wage = TARIFF["day_wage"] if day_wage is None else float(day_wage)
        price = TARIFF["unit_price_won"] if unit_price is None else float(unit_price)
        basic = TARIFF["base_rate_per_kw"] if base_rate is None else float(base_rate)
        peak = TARIFF["billing_peak_kw"] if billing_peak is None else float(billing_peak)
        ok = (0 <= rate <= 100000 and 0 < base_wage <= 1000000 and 0 <= price <= 1000000
              and 0 <= basic <= 1000000 and 0 <= peak <= 1000000)
        if not all(math.isfinite(v) for v in (rate, base_wage, price, basic, peak)) or not ok:
            raise ValueError("단가·시급을 유효한 범위로 입력해주세요.")

        _, night = wage_at(end.hour)
        wage = base_wage * (TARIFF["night_multiplier"] if night else 1)
        kwh = max(0.0, float(prediction))
        energy = kwh * rate
        monthly_base = peak * basic
        month_days = _calendar.monthrange(end.year, end.month)[1]
        allocated = monthly_base / (month_days * 24)

        def costs(staff):
            labor = staff * wage if staff is not None else None
            return {"labor": round(labor, 2) if labor is not None else None,
                    "energy": round(energy, 2), "baseAllocated": round(allocated, 2),
                    "total": round(labor + energy + allocated, 2) if labor is not None else None}

        cost = costs(recommended)
        revenue = hourly_q * price
        remainder = revenue - cost["total"] if cost["total"] is not None else None
        return {
            "day": end.strftime("%Y-%m-%d"), "dayType": day_type(end),
            "forecastTime": start.isoformat(), "targetTime": end.isoformat(),
            "intervalStart": end.isoformat(),
            "intervalEnd": (end + pd.Timedelta(hours=1)).isoformat(), "durationMinutes": 60,
            "rawHour": raw_hour.isoformat(), "production": hourly_q,
            "hourlyProduction": hourly_q,
            "productionBasis": "정시부터 1시간 동안의 raw 생산량",
            "prediction": round(float(prediction), 4), "kwh": round(kwh, 4),
            "recommendedStaff": recommended, "reference": reference,
            "historicalMatches": matches,
            "status": "complete" if recommended is not None else "insufficient_data",
            "optimized": cost,
            "economics": {
                "unitPrice": price,
                "priceBasis": (TARIFF["unit_price_basis"] if price == TARIFF["unit_price_won"]
                               else "사용자가 입력한 개당 생산 이익 (인건비·전력비 차감 전)"),
                "productionProfit": round(revenue, 2),
                "remainder": round(remainder, 2) if remainder is not None else None},
            "fixed": {
                "monthlyBase": monthly_base, "billingPeak": peak, "baseRate": basic,
                "monthDays": month_days, "rate": rate, "band": LABELS[band],
                "wage": wage, "night": night, "dayWage": base_wage,
                "tariffRate": tariff_rate, "rateOverridden": energy_rate is not None,
                "season": season, "bandKey": band,
                "rateTable": TARIFF["rates"] if rate_table is None else rate_table,
                "timeBands": TARIFF["time_bands"],
                "timeBandSource": TARIFF["time_band_source"],
                "nightWage": base_wage * TARIFF["night_multiplier"]}}
