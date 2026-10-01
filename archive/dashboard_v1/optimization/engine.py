"""Historical minimum-staff reference for fixed hourly production and power."""
import calendar
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = json.loads((Path(__file__).parent / 'config.json').read_text())
LABELS = {'off': '경부하', 'mid': '중간부하', 'peak': '최대부하'}


def tariff_at(ts, rates=None):
    season = 'summer' if ts.month in (6, 7, 8) else 'winter' if ts.month in (11, 12, 1, 2) else 'spring_autumn'
    h = ts.hour + ts.minute / 60 + ts.second / 3600
    band = next(band for band, intervals in CONFIG['time_bands'][season].items()
                if any(start <= h < end for start, end in intervals))
    # 2021 TOU convention: Sunday/public holiday off-peak; Saturday peak -> mid.
    if ts.weekday() == 6 or ts.strftime('%Y-%m-%d') in CONFIG['public_holidays_2021']:
        band = 'off'
    elif ts.weekday() == 5 and band == 'peak':
        band = 'mid'
    return season, band, (CONFIG['rates'] if rates is None else rates)[season][band]


def wage_at(hour):
    night = hour >= CONFIG['night_start'] or hour < CONFIG['night_end']
    return CONFIG['day_wage'] * (CONFIG['night_multiplier'] if night else 1), night


class StaffingOptimizer:
    """Minimum integer staffing among recorded production witnesses.

    For each hour, candidates are earlier dates with the same hour and day type,
    Q_observed >= Q_required, and a valid staff value. The selected target-hour staffing is excluded.
    Choose the smallest ceil(staff) among earlier-date witnesses.
    This is a historical reference optimum, not a learned causal capacity model.
    """
    def __init__(self, raw_path):
        raw = pd.read_csv(raw_path)
        dates = pd.to_datetime(raw['날짜'].astype(str), format='%Y%m%d', errors='coerce')
        hours = pd.to_numeric(raw['시간'], errors='coerce')
        valid = dates.notna() & hours.between(0, 23) & (hours % 1 == 0) & ~dates.dt.strftime('%Y-%m-%d').isin(['2021-07-13', '2021-07-15'])
        self.raw = raw.loc[valid].copy()
        self.raw['timestamp'] = dates.loc[valid] + pd.to_timedelta(hours.loc[valid], unit='h')
        self.raw = self.raw.sort_values('timestamp')
        if self.raw['timestamp'].duplicated().any():
            raise ValueError('Hourly optimization input contains duplicate times.')
        self.raw['day_type'] = self.raw['timestamp'].map(self.day_type)
        self.raw['staff_valid'] = (np.isfinite(self.raw['공장인원']) & self.raw['공장인원'].ge(0)
                                  & (self.raw['생산량'].eq(0) | self.raw['공장인원'].gt(0)))

    @staticmethod
    def day_type(ts):
        return '휴일' if ts.weekday() >= 5 or ts.strftime('%Y-%m-%d') in CONFIG['public_holidays_2021'] else '평일'

    def hourly_interval(self, forecast_time, target_time, prediction, *, unit_price=None, day_wage=None, energy_rate=None, rate_table=None, base_rate=None, billing_peak=None):
        start, end = pd.Timestamp(forecast_time), pd.Timestamp(target_time)
        if end-start != pd.Timedelta(minutes=15) or start.minute != 45 or end.minute != 0:
            raise ValueError('정시 직전 45분에 생성한 다음 15분 예측이 필요합니다.')
        # User-defined planning convention: target-hour production/staffing,
        # using the prediction issued 15 minutes before that hour.
        raw_hour = end
        interval_end = end + pd.Timedelta(hours=1)
        selected = self.raw.loc[self.raw.timestamp == raw_hour]
        if len(selected) != 1:
            raise ValueError('해당 정시의 시간당 원자료가 없습니다. 다른 시간대를 선택해주세요.')
        row = selected.iloc[0]
        hourly_q = float(row['생산량'])
        if not math.isfinite(hourly_q) or hourly_q < 0 or not math.isfinite(float(prediction)):
            raise ValueError('생산량과 예측 전력은 유효한 값이어야 합니다.')
        q = hourly_q
        history = self.raw.loc[(self.raw.timestamp < end.normalize()) & self.raw.staff_valid
                               & (self.raw.timestamp.dt.hour == end.hour)
                               & (self.raw.day_type == row.day_type)
                               & np.isfinite(self.raw['생산량']) & self.raw['생산량'].ge(hourly_q)].copy()
        historical_count = len(history)
        candidates = history
        if not candidates.empty:
            candidates['integer_staff'] = candidates['공장인원'].map(lambda n:math.ceil(float(n)))
            witness = candidates.sort_values(['integer_staff','생산량','timestamp'],ascending=[True,True,False]).iloc[0]
            recommended = int(witness.integer_staff)
            reference = {'date':witness.timestamp.strftime('%Y-%m-%d'), 'hour':end.hour,
                         'hourlyProduction':float(witness['생산량']),
                         'rawStaff':round(float(witness['공장인원']),4), 'staff':recommended}
        else:
            recommended, reference = None, None
        season, band, tariff_rate = tariff_at(end,rate_table)
        rate = tariff_rate if energy_rate is None else float(energy_rate)
        base_wage = CONFIG['day_wage'] if day_wage is None else float(day_wage)
        price = CONFIG['unit_price_won'] if unit_price is None else float(unit_price)
        basic = CONFIG['base_rate_per_kw'] if base_rate is None else float(base_rate)
        peak = CONFIG['billing_peak_kw'] if billing_peak is None else float(billing_peak)
        if not all(math.isfinite(v) for v in (rate,base_wage,price,basic,peak)) or not (0<=rate<=100000 and 0<base_wage<=1000000 and 0<=price<=1000000 and 0<=basic<=1000000 and 0<=peak<=1000000):
            raise ValueError('단가·시급을 유효한 범위로 입력해주세요.')
        _, night = wage_at(end.hour)
        wage = base_wage * (CONFIG['night_multiplier'] if night else 1)
        kwh = max(0,float(prediction)) * 1.0
        energy = kwh * rate
        monthly_base = peak*basic
        month_days = calendar.monthrange(end.year,end.month)[1]
        allocated_base = monthly_base/(month_days*24)
        def costs(staff):
            labor = staff*wage if staff is not None else None
            return {'labor':round(labor,2) if labor is not None else None,
                    'energy':round(energy,2), 'baseAllocated':round(allocated_base,2),
                    'total':round(labor+energy+allocated_base,2) if labor is not None else None}
        recommended_cost = costs(recommended)
        unit_price = price
        revenue = q * unit_price
        remainder = revenue - recommended_cost['total'] if recommended_cost['total'] is not None else None
        return {'day':end.strftime('%Y-%m-%d'), 'dayType':self.day_type(end),
                'forecastTime':start.isoformat(), 'targetTime':end.isoformat(), 'intervalStart':end.isoformat(), 'intervalEnd':interval_end.isoformat(), 'durationMinutes':60,
                'rawHour':raw_hour.isoformat(), 'production':q, 'hourlyProduction':hourly_q,
                'productionBasis':'정시부터 1시간 동안의 raw 생산량',
                'prediction':round(float(prediction),4), 'kwh':round(kwh,4),
                'recommendedStaff':recommended, 'reference':reference, 'historicalMatches':historical_count,
                'status':'complete' if recommended is not None else 'insufficient_data',
                'optimized':recommended_cost,
                'economics':{'unitPrice':unit_price,'priceBasis':CONFIG['unit_price_basis'] if unit_price==CONFIG['unit_price_won'] else '사용자가 입력한 개당 생산 이익 (인건비·전력비 차감 전)',
                             'productionProfit':round(revenue,2),'remainder':round(remainder,2) if remainder is not None else None},
                'fixed':{'monthlyBase':monthly_base,'billingPeak':peak,
                         'baseRate':basic,'monthDays':month_days,'rate':rate,'band':LABELS[band],
                         'wage':wage,'night':night,'dayWage':base_wage,'tariffRate':tariff_rate,'rateOverridden':energy_rate is not None,'season':season,'bandKey':band,'rateTable':CONFIG['rates'] if rate_table is None else rate_table,
                         'timeBands':CONFIG['time_bands'],'timeBandSource':CONFIG['time_band_source'],
                         'nightWage':base_wage*CONFIG['night_multiplier']}}
