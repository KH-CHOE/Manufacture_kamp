"""최종 학습 입력 생성. 원본 파일은 변경하지 않는다.
최종 CSV의 시간은 HS와 같은 구간 종료 시각 기준이다.
기존 BG 입력 가용성 가정(같은 시간 생산·날씨를 안다)을 유지한다.
"""
from pathlib import Path
import json, hashlib, unicodedata
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
def locate(p):
    return p if p.exists() else Path(unicodedata.normalize('NFD',str(p)))
POWER_COLUMNS=['15분','30분','45분','60분']
CONTEXT_COLUMNS=['생산량','기온','풍속','습도','강수량']
LAGS=(1,2,4,96,672)
FEATURES = [
    '주중여부', '주중공휴일여부', '시간', '현재전력', '생산량', '기온', '풍속',
    '습도', '강수량', 'day', 'd', 'm', '15분위치',
    '과거전력_1칸', '과거전력_2칸', '과거전력_4칸', '과거전력_96칸',
    '전력변화_15분', '전력변화_60분', '최근1시간_전력평균', '최근1시간_전력최대',
    '과거전력_8칸', '과거전력_16칸', '전력변화_2시간', '전력변화_4시간',
    '최근2시간_전력평균', '최근2시간_전력최대', '최근4시간_전력평균', '최근4시간_전력최대',
]

PARAMS = dict(n_estimators=300, max_features=1.0, min_samples_leaf=2,
              min_samples_split=6, max_depth=32, bootstrap=False,
              max_samples=None, criterion='squared_error', random_state=42, n_jobs=-1)

def prepare(source: Path) -> pd.DataFrame:
    """1단계와 동일한 공통 표본 및 F 변수 구성. 삭제일을 넘는 창 제외."""
    df = pd.read_csv(source)
    if len(df) != 24480 or df['전력'].isna().sum() != 1:
        raise ValueError('기존 preprocssed_3 데이터(24,480행, 정답 결측 1개)가 필요합니다.')
    date = pd.to_datetime(dict(year=2021, month=df.m, day=df.d))
    timestamp = date + pd.to_timedelta(df['시간'], unit='h') + pd.to_timedelta(df['15분위치'] * 15, unit='m')
    if not timestamp.is_unique or not timestamp.is_monotonic_increasing:
        raise ValueError('시간 순서 또는 중복 시점을 확인하세요.')
    segment = timestamp.diff().ne(pd.Timedelta(minutes=15)).cumsum()
    power = df['현재전력'].groupby(segment, sort=False)
    for lag in (8, 16):
        df[f'과거전력_{lag}칸'] = power.shift(lag)
    for lag, label in ((8, '2시간'), (16, '4시간')):
        df[f'전력변화_{label}'] = df['현재전력'] - df[f'과거전력_{lag}칸']
        df[f'최근{label}_전력평균'] = power.transform(lambda s: s.rolling(lag, min_periods=lag).mean())
        df[f'최근{label}_전력최대'] = power.transform(lambda s: s.rolling(lag, min_periods=lag).max())
    # 원래 탐색에서 A–G가 같은 행을 쓰도록 적용한 생산량 결측 조건도 유지.
    # 아래 3개 임시 변수는 최종 모델 입력에 포함하지 않는다.
    prod = df['생산량'].groupby(segment, sort=False)
    df['과거생산량_4칸'] = prod.shift(4)
    df['생산량변화_1시간'] = df['생산량'] - df['과거생산량_4칸']
    df['최근1시간_생산량평균'] = prod.transform(lambda s: s.rolling(4, min_periods=4).mean())
    gap_boundary = (date.shift(-1) - date).dt.days.gt(1)
    usable = df[FEATURES + ['전력', '과거생산량_4칸', '생산량변화_1시간', '최근1시간_생산량평균']].notna().all(axis=1) & ~gap_boundary
    data = df.loc[usable].copy().reset_index(names='source_row')
    data['date_key'] = 20210000 + data.m * 100 + data.d
    data['timestamp'] = timestamp.loc[usable].to_numpy()
    if len(data) != 24190 or int(gap_boundary.sum()) != 2:
        raise ValueError('기존 실험의 24,190개 공통 표본과 일치하지 않습니다.')
    return data

def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """불가능한 시간은 제외하고 시간 공백을 유지합니다. 원본은 수정하지 않습니다."""
    required = ['날짜', '시간', *POWER_COLUMNS, *CONTEXT_COLUMNS]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f'필수 입력 열이 없습니다: {missing}')
    data = raw.copy()
    data.insert(0, '원본행번호', np.arange(len(data)) + 2)
    data['날짜'] = pd.to_datetime(data['날짜'].astype(str), format='%Y%m%d', errors='coerce')
    for col in ['시간', *POWER_COLUMNS, *CONTEXT_COLUMNS]:
        # 결측은 허용하되 문자열 오타를 조용히 결측으로 바꾸지는 않습니다.
        data[col] = pd.to_numeric(data[col], errors='raise')
        if np.isinf(data[col].to_numpy(dtype=float)).any():
            raise ValueError(f'{col}: 무한대 값이 있습니다.')
    bad = data['날짜'].isna() | ~data['시간'].between(0, 23) | data['시간'].mod(1).ne(0)
    excluded = data.loc[bad].copy()
    excluded['제외사유'] = '날짜 오류 또는 0~23 범위를 벗어난 정수 시간'
    data = data.loc[~bad].copy()
    if data.empty:
        raise ValueError('전처리 후 사용 가능한 행이 없습니다.')
    if data[POWER_COLUMNS].lt(0).any().any():
        raise ValueError('음수 전력이 있습니다. 원자료의 부호 정의를 확인하세요.')
    data['날짜시간'] = data['날짜'] + pd.to_timedelta(data['시간'], unit='h')
    if data['날짜시간'].duplicated().any():
        raise ValueError('동일 날짜·시간에 중복 행이 있습니다. 공장별로 분리하거나 중복을 확인하세요.')
    data = data.sort_values('날짜시간').reset_index(drop=True)
    logs = []
    for col in ['풍속', '강수량']:
        observed = data[col].copy()
        past_median = observed.expanding(min_periods=1).median().shift(1)
        data[col] = observed.fillna(past_median)
        for i in data.index[observed.isna()]:
            logs.append({'원본행번호':int(data.loc[i, '원본행번호']), '열':col,
                         '대치값':data.loc[i, col], '방법':'엄격히 앞선 관측의 누적 중앙값'})
    # 처음부터 관측이 없으면 NaN을 유지하고 각 폴드의 학습 파이프라인에서 처리합니다.
    imputation = pd.DataFrame(logs, columns=['원본행번호', '열', '대치값', '방법'])
    return data, excluded, imputation

def build_features(hourly: pd.DataFrame, include_next: bool = False) -> pd.DataFrame:
    """목표 시각 t의 입력은 t-15분까지 확정된 정보만 사용합니다."""
    long = hourly.melt(id_vars=['날짜시간'], value_vars=POWER_COLUMNS,
                       var_name='분구간', value_name='목표_전력')
    long['목표시각'] = long['날짜시간'] + pd.to_timedelta(
        long['분구간'].str.replace('분', '').astype(int), unit='m')
    series = long.set_index('목표시각')['목표_전력'].sort_index()
    if series.index.duplicated().any():
        raise ValueError('중복된 15분 전력 시각이 있습니다.')
    if not series.notna().any():
        raise ValueError('관측된 전력값이 없습니다.')
    end = series.last_valid_index() + pd.Timedelta(minutes=15) if include_next else series.index.max()
    grid = pd.date_range(series.index.min(), end, freq='15min')
    y = series.reindex(grid)  # 공백을 제거한 뒤 shift하면 안 됩니다.
    x = pd.DataFrame(index=grid)
    for lag in LAGS:
        x[f'전력_lag{lag}'] = y.shift(lag)
    x['이전1시간평균'] = y.shift(1).rolling(4, min_periods=4).mean()
    x['이전1시간표준편차'] = y.shift(1).rolling(4, min_periods=4).std()
    angle = 2 * np.pi * (grid.hour + grid.minute / 60) / 24
    x['시간_sin'], x['시간_cos'] = np.sin(angle), np.cos(angle)
    x['요일'], x['월'], x['분'] = grid.dayofweek, grid.month, grid.minute
    context = hourly.set_index('날짜시간')[CONTEXT_COLUMNS].copy()
    context.index = context.index + pd.Timedelta(hours=1)
    for col in CONTEXT_COLUMNS:
        # 시간별 생산량/날씨는 해당 시간 종료 후 이용 가능하다고 가정합니다.
        available = context[col].reindex(grid - pd.Timedelta(minutes=15),
                                        method='ffill', tolerance=pd.Timedelta(minutes=45))
        x[f'이전확정_{col}'] = available.to_numpy()
    x.insert(0, '목표시각', grid)
    x.insert(1, '예측시각', grid - pd.Timedelta(minutes=15))
    x['목표_전력'] = y
    return x.reset_index(drop=True)

def main():
    source=locate(PROJECT/'데이터셋/okm_augumented_2021_preprocssed_3.csv')
    raw=locate(PROJECT/'데이터셋/okm_augumented_2021.csv')
    b=prepare(source)
    b['forecast_time']=b.timestamp+pd.Timedelta(minutes=15)
    b['target_time']=b.timestamp+pd.Timedelta(minutes=30)
    hs_source=HERE.parent/'5단계/output/HS/model_ready.csv'
    h=pd.read_csv(hs_source,parse_dates=['목표시각','예측시각'])
    # 제공된 HS 입력을 raw에서 다시 만든 값과 대조해 재현성을 확인한다.
    hh=build_features(preprocess(pd.read_csv(raw))[0])
    generated=hh.loc[hh['목표_전력'].notna() & hh['전력_lag1'].notna()].reset_index(drop=True)
    hscols=[c for c in h if c not in ['목표시각','예측시각','목표_전력','분할']]
    np.testing.assert_allclose(h[hscols+['목표_전력']],generated[hscols+['목표_전력']],equal_nan=True,rtol=1e-8,atol=1e-5)
    assert (h['목표시각']==generated['목표시각']).all()
    h=h.rename(columns={c:'HS_'+c for c in hscols})
    merged=b.merge(h,left_on='target_time',right_on='목표시각',validate='one_to_one',how='left')
    assert len(merged)==24190 and merged['목표_전력'].notna().all()
    np.testing.assert_array_equal(merged['전력'],merged['목표_전력'])
    np.testing.assert_array_equal(merged['현재전력'],merged['HS_전력_lag1'])
    merged['split']=np.where(merged.date_key<20210627,'train','test')
    last_train=merged.index[merged.split=='train'][-1]
    merged.loc[last_train,'split']='boundary_excluded'
    features=FEATURES+['HS_'+c for c in hscols]
    selection=json.loads((HERE/'selection.json').read_text(encoding='utf-8'))
    selected=selection['features']
    if not set(selected).issubset(features):
        raise ValueError('선정 변수와 생성 변수 목록이 일치하지 않습니다.')
    out=merged[['source_row','forecast_time','target_time','date_key','split','전력']+selected]
    out.to_csv(HERE/'final_input_data.csv',index=False,encoding='utf-8-sig')
    manifest={'raw':{'path':str(raw),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()},
      'preprocessed':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
      'HS_input':{'path':str(hs_source),'sha256':hashlib.sha256(hs_source.read_bytes()).hexdigest()},
      'rows':len(out),'split_counts':out.split.value_counts().to_dict(),
      'BG_features':FEATURES,'HS_features':['HS_'+c for c in hscols],
      'selected_features':selected,
      'target':'next 15-minute power','HS_reconstruction_verified':True,'HS_reconstruction_atol':1e-5,
      'timestamp_convention':'interval end, BG original timestamp + 15min',
      'known_context_assumption':'same-hour production and weather are available, as previously agreed'}
    (HERE/'input_reproduction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Prepared',out.shape,out.split.value_counts().to_dict())

if __name__=='__main__': main()
