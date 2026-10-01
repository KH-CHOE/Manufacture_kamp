"""Run: python3 -m backend.verify. Checks actual model inference and replay isolation."""
import asyncio
from math import isclose
from pydantic import ValidationError
from fastapi import HTTPException
from .app import app, lifespan, meta, snapshot, scenario, Scenario, state

async def main():
    async with lifespan(app):
        m = meta()
        s = snapshot('2021-07-12',48,180)
        automatic = snapshot(s['day'],48,None)
        rec = automatic['recommendation']
        assert automatic['threshold'] == rec['value']
        assert rec['annualPeakTime'] <= automatic['time']
        assert rec['value'] == round(max(1, rec['annualPeak']-rec['errorMargin']),1)
        now = state['groups'][s['day']].iloc[48]['forecast_time']
        future_obs = state['observed']['forecast_time'] > now
        original = state['observed'].loc[future_obs,'현재전력'].copy()
        state['observed'].loc[future_obs,'현재전력'] = 99999
        future_targets = state['frame']['target_time'] > now
        original_targets = state['frame'].loc[future_targets,'전력'].copy()
        state['frame'].loc[future_targets,'전력'] = 99999
        assert snapshot(s['day'],48,None)['recommendation'] == rec
        state['observed'].loc[future_obs,'현재전력'] = original
        state['frame'].loc[future_targets,'전력'] = original_targets
        print('Auto threshold:',rec)
        data = state['groups'][s['day']]
        row = data.iloc[[s['cursor']]]
        exact = float(state['bundle']['estimator'].predict(row[state['bundle']['features']])[0])
        assert isclose(s['prediction'],exact,abs_tol=.00051)
        assert s['targetTime'] > s['time']
        assert s['points'][-1]['actual'] is None
        assert all(p['time'] <= s['time'] for p in s['points'][:-1])
        assert all(a['time'] <= s['time'] for a in s['alerts'])
        assert snapshot(s['day'],48,120)['atRisk'] is True
        assert snapshot(s['day'],48,10000)['atRisk'] is False
        baseline = dict(day=s['day'],cursor=48,production=s['context']['생산량'],staff=12,baseline_staff=12,hourly_wage=15000,energy_rate=120,power_factor=1)
        result = scenario(Scenario(**baseline))
        assert result['baseline'] == result['scenario']
        assert isclose(result['prediction'],s['prediction'],abs_tol=.00051)
        baseline['staff'] = 10
        changed = scenario(Scenario(**baseline))
        assert result['scenario']['total']-changed['scenario']['total'] == 7500
        assert changed['scenario']['labor'] == 37500
        assert changed['scenario']['energy'] == round(exact*.25*120)
        assert '2021-07-13' not in m['days'] and '2021-07-15' not in m['days']
        try: snapshot('2021-07-13',48,180)
        except HTTPException as e: assert e.status_code == 404
        else: raise AssertionError('Missing day must return 404')
        try: Scenario(**{**baseline,'power_factor':-1})
        except ValidationError: pass
        else: raise AssertionError('Negative conversion must be rejected')
        print(f"PASS: inference parity, no future observations, alarms, cost arithmetic, invalid inputs. MSE={m['mse']}, MAE={m['mae']}")

if __name__ == '__main__': asyncio.run(main())
