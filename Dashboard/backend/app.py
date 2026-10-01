"""Local recorded-data replay and replaceable sklearn model API."""
from pathlib import Path
import os
import hashlib
from contextlib import asynccontextmanager
from threading import Lock
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from optimization.engine import StaffingOptimizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'models'
MODEL_PATH = Path(os.getenv('MODEL_PATH', str(SOURCE / 'final_model.joblib'))).resolve()
DATA_PATH = Path(os.getenv('DATA_PATH', str(ROOT / 'data' / 'preprocessed' / 'final_input_data.csv'))).resolve()
state = {}
lock = Lock()

@asynccontextmanager
async def lifespan(app):
    bundle = joblib.load(MODEL_PATH)  # Load only trusted, locally produced model bundles.
    if not isinstance(bundle, dict) or not {'estimator', 'features'} <= bundle.keys():
        raise ValueError('Model bundle must include estimator and features.')
    frame = pd.read_csv(DATA_PATH)
    required = list(bundle['features']) + ['forecast_time', 'target_time', 'split', '현재전력', '전력', '생산량', '기온', '풍속', '습도', '주중여부', '주중공휴일여부']
    missing = set(required) - set(frame.columns)
    if missing: raise ValueError(f'Missing data columns: {sorted(missing)}')
    frame['forecast_time'] = pd.to_datetime(frame['forecast_time'])
    frame['target_time'] = pd.to_datetime(frame['target_time'])
    frame = frame.sort_values('forecast_time').reset_index(drop=True)
    observed = frame[['forecast_time', '현재전력']].copy()
    state['optimizer'] = StaffingOptimizer(ROOT/'data/raw/okm_augumented_2021.csv')
    frame = frame.loc[frame['split'] == 'test'].reset_index(drop=True)
    if frame.empty: raise ValueError('No test records for replay.')
    predictions = np.asarray(bundle['estimator'].predict(frame[bundle['features']]), dtype=float)
    if predictions.shape != (len(frame),) or not np.isfinite(predictions).all():
        raise ValueError('Expected finite, scalar next-15-minute predictions.')
    frame['prediction'] = predictions
    frame['day_key'] = frame['forecast_time'].dt.strftime('%Y-%m-%d')
    groups = {day: g.reset_index(drop=True) for day, g in frame.groupby('day_key')}
    state.update(bundle=bundle, frame=frame, groups=groups, observed=observed, version=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:10])
    yield

app = FastAPI(title='KAMP power dashboard', lifespan=lifespan)

def day_frame(day):
    if day not in state['groups']: raise HTTPException(404, '해당 날짜에 재생할 데이터가 없습니다.')
    return state['groups'][day]

def iso(value): return value.strftime('%Y-%m-%dT%H:%M:%S')
def native(value): return round(float(value), 3)

def model_type(model):
    if hasattr(model, 'estimators_') and type(model).__name__ == 'VotingRegressor':
        return ' + '.join(dict.fromkeys(model_type(m) for m in model.estimators_))
    if hasattr(model, 'steps'): return type(model.steps[-1][1]).__name__
    return type(model).__name__

@app.get('/api/meta')
def meta():
    frame = state['frame']
    prediction = frame['prediction'].to_numpy()
    target = frame['전력'].to_numpy()
    return {'days': list(state['groups']), 'defaultDay': '2021-07-12' if '2021-07-12' in state['groups'] else list(state['groups'])[0],
            'model': MODEL_PATH.name, 'algorithm': type(state['bundle']['estimator']).__name__,
            'modelType': model_type(state['bundle']['estimator']), 'version': state['version'], 'features': state['bundle']['features'], 'horizon': 15, 'rows': len(frame),
            'mse': native(np.mean((prediction-target)**2)), 'mae': native(np.mean(abs(prediction-target))),
            'unit': '원자료 값', 'source': DATA_PATH.name}

def peak_recommendation(now):
    # Only observations and realised prediction errors available at this replay time.
    observed = state['observed']
    available = observed.loc[(observed['forecast_time'].dt.year == now.year) & (observed['forecast_time'] <= now)]
    peak_row = available.loc[available['현재전력'].idxmax()]
    realised = state['frame'].loc[(state['frame']['target_time'] <= now) & (state['frame']['target_time'].dt.year == now.year)]
    if len(realised) >= 30:
        margin = float(np.quantile(np.maximum(0, realised['전력']-realised['prediction']), .90))
        basis = '현재까지 확정된 예측 오차의 과소예측량 90백분위'
    else:
        cv_mse = state['bundle'].get('selection', {}).get('CV_MSE')
        margin = float(np.sqrt(cv_mse)) if cv_mse is not None else float(peak_row['현재전력']) * .05
        basis = '학습 교차검증 RMSE' if cv_mse is not None else '관측 최대 전력의 5% (초기 참고 가정)'
    peak = float(peak_row['현재전력'])
    value = round(max(1, peak-margin), 1)
    return {'value':value, 'annualPeak':native(peak), 'annualPeakTime':iso(peak_row['forecast_time']),
            'year':int(now.year), 'coverageStart':iso(available['forecast_time'].min()),
            'asOf':iso(now), 'errorMargin':native(margin), 'errorSamples':len(realised), 'basis':basis}

@app.get('/api/snapshot')
def snapshot(day: str, cursor: int = Query(48, ge=0), threshold: float | None = Query(None, gt=0, le=10000)):
    data = day_frame(day)
    cursor = min(cursor, len(data)-1)
    row = data.iloc[cursor]
    now = row['forecast_time']
    recommendation = peak_recommendation(now)
    threshold = recommendation['value'] if threshold is None else threshold
    history = data.iloc[:cursor+1]
    points = [{'time': iso(r['forecast_time']), 'actual': native(r['현재전력']), 'predicted': None} for _,r in history.iterrows()]
    by_time = {p['time']:p for p in points}
    for _,r in history.iterrows():
        target = iso(r['target_time'])
        if target in by_time: by_time[target]['predicted'] = native(r['prediction'])
    points.append({'time':iso(row['target_time']), 'actual': None, 'predicted':native(row['prediction'])})
    alerts = [{'time':iso(r['forecast_time']), 'targetTime':iso(r['target_time']), 'predicted':native(r['prediction']),
               'key':iso(r['forecast_time'])} for _,r in history.loc[history['prediction'] >= threshold].tail(20).iloc[::-1].iterrows()]
    return {'day':day, 'cursor':cursor, 'count':len(data), 'time':iso(now), 'targetTime':iso(row['target_time']),
            'current':native(row['현재전력']), 'prediction':native(row['prediction']), 'threshold':threshold,
            'margin':native(threshold-row['prediction']), 'atRisk':bool(row['prediction']>=threshold),
            'recommendation':recommendation, 'dailyPeak':native(history['현재전력'].max()), 'points':points, 'alerts':alerts,
            'timeline':[iso(t) for t in data['forecast_time']],
            'context':{k:native(row[k]) for k in ['생산량','기온','풍속','습도','주중여부','주중공휴일여부']}}

class Scenario(BaseModel):
    day: str
    cursor: int = Field(ge=0)
    production: float = Field(ge=0, le=100000)
    staff: int = Field(ge=0, le=10000)
    hourly_wage: float = Field(ge=0, le=1000000)
    energy_rate: float = Field(ge=0, le=100000)
    power_factor: float = Field(gt=0, le=10000)
    baseline_staff: int = Field(ge=0, le=10000)

@app.post('/api/scenario')
def scenario(spec: Scenario):
    data = day_frame(spec.day)
    if spec.cursor >= len(data): raise HTTPException(422, '재생 위치가 날짜 범위를 벗어났습니다.')
    row = data.iloc[[spec.cursor]].copy()
    row['생산량'] = spec.production
    with lock:
        predicted = float(state['bundle']['estimator'].predict(row[state['bundle']['features']])[0])
    if not np.isfinite(predicted): raise HTTPException(500, '모델이 유효한 값을 반환하지 않았습니다.')
    baseline = float(data.iloc[spec.cursor]['prediction'])
    def costs(power, staff):
        energy = max(0, power) * spec.power_factor * .25 * spec.energy_rate
        labor = staff * spec.hourly_wage * .25
        return {'energy':round(energy), 'labor':round(labor), 'total':round(energy+labor)}
    return {'prediction':native(predicted), 'baselinePrediction':native(baseline),
            'baseline':costs(baseline, spec.baseline_staff), 'scenario':costs(predicted,spec.staff),
            'productionRange':{'min':native(state['frame']['생산량'].min()),'max':native(state['frame']['생산량'].max())},
            'assumption':'전력 값 × kW 환산계수 × 0.25시간 × 전력량 단가 + 인원 × 시급 × 0.25시간. 기본요금 제외.'}

class LoadRates(BaseModel):
    off: float = Field(ge=0, le=100000, allow_inf_nan=False)
    mid: float = Field(ge=0, le=100000, allow_inf_nan=False)
    peak: float = Field(ge=0, le=100000, allow_inf_nan=False)

class TariffRates(BaseModel):
    summer: LoadRates
    spring_autumn: LoadRates
    winter: LoadRates

class StaffingSpec(BaseModel):
    day: str
    cursor: int = Field(48, ge=0)
    unit_price: float | None = Field(None, ge=0, le=1000000, allow_inf_nan=False)
    day_wage: float | None = Field(None, gt=0, le=1000000, allow_inf_nan=False)
    energy_rate: float | None = Field(None, ge=0, le=100000, allow_inf_nan=False)
    rate_table: TariffRates | None = None
    base_rate: float | None = Field(None, ge=0, le=1000000, allow_inf_nan=False)
    billing_peak: float | None = Field(None, ge=0, le=1000000, allow_inf_nan=False)


def staffing_plan(spec):
    data = day_frame(spec.day)
    cursor = min(spec.cursor, len(data)-1)
    row = data.iloc[cursor]
    try:
        now = row['forecast_time']
        hour = (now + pd.Timedelta(minutes=15)).floor('h')
        issued = hour - pd.Timedelta(minutes=15)
        forecasts = state['frame'].loc[(state['frame']['forecast_time'] == issued) & (state['frame']['target_time'] == hour)]
        if len(forecasts) != 1:
            raise ValueError('이 시간대의 정시 직전 45분 예측이 없습니다. 다음 시점으로 이동해주세요.')
        forecast = forecasts.iloc[0]
        result = state['optimizer'].hourly_interval(issued,hour,float(forecast['prediction']),unit_price=spec.unit_price,day_wage=spec.day_wage,energy_rate=spec.energy_rate,rate_table=spec.rate_table.model_dump() if spec.rate_table else None,base_rate=spec.base_rate,billing_peak=spec.billing_peak)
        return {**result,'cursor':cursor,'asOf':iso(now)}
    except ValueError as e:
        raise HTTPException(422,str(e)) from e


@app.get('/api/optimization')
def default_staffing(day: str, cursor: int = Query(48, ge=0),
                     unit_price: float | None = Query(None, ge=0, le=1000000),
                     day_wage: float | None = Query(None, gt=0, le=1000000),
                     energy_rate: float | None = Query(None, ge=0, le=100000),
                     base_rate: float | None = Query(None, ge=0, le=1000000),
                     billing_peak: float | None = Query(None, ge=0, le=1000000)):
    return staffing_plan(StaffingSpec(day=day,cursor=cursor,unit_price=unit_price,day_wage=day_wage,energy_rate=energy_rate,base_rate=base_rate,billing_peak=billing_peak))


@app.post('/api/optimization')
def optimize_staffing(spec: StaffingSpec):
    return staffing_plan(spec)


app.mount('/', StaticFiles(directory=ROOT/'dist', html=True), name='frontend')
