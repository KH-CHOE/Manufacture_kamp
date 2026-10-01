"""Check hourly workforce planning from prior :45 next-15-minute forecasts."""
import asyncio
from math import isclose
import pandas as pd
from fastapi import HTTPException
from backend.app import app, lifespan, state, optimize_staffing, StaffingSpec

async def main():
    async with lifespan(app):
        def plan(cursor,day='2021-07-12'):
            return optimize_staffing(StaffingSpec(day=day,cursor=cursor))
        p=plan(48)
        assert p['asOf']=='2021-07-12T12:00:00'
        assert p['forecastTime']=='2021-07-12T11:45:00'
        assert p['targetTime']==p['intervalStart']=='2021-07-12T12:00:00'
        assert p['intervalEnd']=='2021-07-12T13:00:00' and p['durationMinutes']==60
        exact=float(state['groups']['2021-07-12'].iloc[47]['prediction'])
        assert isclose(p['prediction'],exact,abs_tol=.00006)
        assert p['production']==p['hourlyProduction']==194
        assert p['optimized']['labor']==p['recommendedStaff']*8720
        assert p['optimized']['energy']==round(max(0,exact)*109.5,2)
        assert p['optimized']['baseAllocated']==round(1444000/31/24,2)
        assert p['reference']['hourlyProduction']>=p['production']
        assert p['reference']['date']<p['day']
        custom=optimize_staffing(StaffingSpec(day='2021-07-12',cursor=48,unit_price=150,day_wage=10000,energy_rate=120))
        assert custom['recommendedStaff']==p['recommendedStaff']
        assert custom['economics']['productionProfit']==194*150
        assert custom['optimized']['labor']==p['recommendedStaff']*10000
        assert custom['optimized']['energy']==round(max(0,exact)*120,2)
        assert custom['fixed']['rateOverridden'] and custom['fixed']['rate']==120
        from optimization.engine import CONFIG, tariff_at
        # Exact 2021 KEPCO weekday bands, including overnight and winter evening boundaries.
        schedules={
            '2021-07-12':['off']*9+['mid']+['peak']*2+['mid']+['peak']*4+['mid']*6+['off'],
            '2021-09-06':['off']*9+['mid']+['peak']*2+['mid']+['peak']*4+['mid']*6+['off'],
            '2021-01-04':['off']*9+['mid']+['peak']*2+['mid']*5+['peak']*3+['mid']*2+['peak','off'],
        }
        for day,expected in schedules.items():
            for hour,band in enumerate(expected):
                assert tariff_at(pd.Timestamp(f'{day} {hour:02}:00'))[1]==band
                assert tariff_at(pd.Timestamp(f'{day} {hour:02}:59:59'))[1]==band
        assert tariff_at(pd.Timestamp('2021-07-10 13:00'))[1]=='mid'  # Saturday
        assert tariff_at(pd.Timestamp('2021-07-11 13:00'))[1]=='off'  # Sunday
        assert tariff_at(pd.Timestamp('2021-03-01 13:00'))[1]=='off'  # Public holiday
        basic_plan=optimize_staffing(StaffingSpec(day='2021-07-12',cursor=48,base_rate=10000))
        assert basic_plan['fixed']['baseRate']==10000 and basic_plan['fixed']['monthlyBase']==2000000
        assert basic_plan['optimized']['baseAllocated']==round(2000000/31/24,2)
        assert basic_plan['optimized']['energy']==p['optimized']['energy']
        assert CONFIG['base_rate_per_kw']==7220
        changed_peak=optimize_staffing(StaffingSpec(day='2021-07-12',cursor=48,billing_peak=300))
        assert changed_peak['fixed']['billingPeak']==300
        assert changed_peak['fixed']['monthlyBase']==7220*300
        assert changed_peak['optimized']['baseAllocated']==round(7220*300/31/24,2)
        assert changed_peak['optimized']['energy']==p['optimized']['energy']
        assert changed_peak['recommendedStaff']==p['recommendedStaff']
        assert CONFIG['billing_peak_kw']==200
        from copy import deepcopy
        edited=deepcopy(CONFIG['rates']);edited['summer']['mid']=123;edited['summer']['peak']=234
        table_plan=optimize_staffing(StaffingSpec(day='2021-07-12',cursor=48,rate_table=edited))
        assert table_plan['fixed']['rate']==123 and table_plan['optimized']['energy']==round(max(0,exact)*123,2)
        table_next=optimize_staffing(StaffingSpec(day='2021-07-12',cursor=51,rate_table=edited))
        assert table_next['fixed']['rate']==234
        assert CONFIG['rates']['summer']['mid']==109.5
        assert table_plan['recommendedStaff']==p['recommendedStaff']
        night_custom=optimize_staffing(StaffingSpec(day='2021-07-12',cursor=87,day_wage=10000))
        assert night_custom['fixed']['wage']==15000
        assert not night_custom['fixed']['rateOverridden']
        from pydantic import ValidationError
        for values in [{'unit_price':-1},{'day_wage':0},{'energy_rate':float('nan')},{'base_rate':-1},{'base_rate':float('inf')},{'base_rate':1000001},{'billing_peak':-1},{'billing_peak':float('nan')},{'billing_peak':1000001}]:
            try: StaffingSpec(day='2021-07-12',**values)
            except ValidationError: pass
            else: raise AssertionError('Invalid cost setting was accepted')
        assert 'baseline' not in p and 'baselineStaff' not in p and 'originalStaff' not in p
        assert p['economics']['productionProfit']==194*150
        assert isclose(p['economics']['remainder'],29100-p['optimized']['total'],abs_tol=.011)
        raw=state['optimizer'].raw
        current=raw.timestamp==pd.Timestamp(p['intervalStart'])
        original=raw.loc[current,'공장인원'].copy()
        raw.loc[current,'공장인원']=10000
        assert plan(48)==p  # Selected-hour headcount must not affect the recommendation.
        raw.loc[current,'공장인원']=original
        valid=raw['staff_valid'].copy()
        raw.loc[raw.timestamp<pd.Timestamp(p['day']),'staff_valid']=False
        missing=plan(48)
        assert missing['recommendedStaff'] is None and missing['optimized']['labor'] is None
        raw['staff_valid']=valid
        for cursor in [47,49,50]:
            same=plan(cursor)
            assert same['intervalStart']==p['intervalStart'] and same['prediction']==p['prediction']
        next_hour=plan(51)
        assert next_hour['forecastTime']=='2021-07-12T12:45:00'
        assert next_hour['intervalStart']=='2021-07-12T13:00:00'
        assert next_hour['production']==676 and next_hour['fixed']['rate']==191.6
        assert plan(52)['prediction']==next_hour['prediction']
        assert plan(86)['fixed']['wage']==8720
        assert plan(87)['fixed']['wage']==13080
        midnight=plan(95,'2021-07-11')
        assert midnight['intervalStart']=='2021-07-12T00:00:00'
        assert midnight['intervalEnd']=='2021-07-12T01:00:00'
        assert midnight['production']==float(state['optimizer'].raw.loc[state['optimizer'].raw.timestamp==pd.Timestamp('2021-07-12 00:00'),'생산량'].iloc[0])
        try: plan(95)
        except HTTPException as e: assert e.status_code==422
        else: raise AssertionError('Deleted July 13 raw hour must not be replaced with July 12.')
        print('PASS: :45 forecast selection without future access; target-hour raw production; 60-minute wages and tariff; hourly refresh; midnight; selected-hour staff excluded; no-history unavailable; production profit arithmetic; edited basic charge; all KEPCO seasonal hourly boundaries and holiday rules.')
        print({k:p[k] for k in ['forecastTime','intervalStart','intervalEnd','production','recommendedStaff','optimized']})

if __name__=='__main__':asyncio.run(main())
