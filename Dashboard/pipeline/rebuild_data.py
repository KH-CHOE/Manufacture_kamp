"""Build final model input directly from the packaged raw data.

Run: python3 pipeline/rebuild_data.py
Intermediate stages stay in memory; only final_input_data.csv is written.
"""
from pathlib import Path
import json
import argparse
from io import StringIO
import os
import numpy as np
import pandas as pd
from features import POWER_COLUMNS, FEATURES, prepare, preprocess, build_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/okm_augumented_2021.csv'
OUTPUT = ROOT/'data/preprocessed/final_input_data.csv'


def build_first_stage():
    raw = pd.read_csv(RAW)
    if len(raw) != 6168:
        raise ValueError('This reproduction pipeline expects the supplied 2021 dataset (6,168 hourly rows).')
    # Expand in original row order, then shift BEFORE deleting erroneous dates.
    context = [c for c in raw if c not in POWER_COLUMNS]
    expanded = raw.loc[raw.index.repeat(4), context].reset_index(drop=True)
    power = raw[POWER_COLUMNS].to_numpy().reshape(-1)
    expanded.insert(2,'현재전력',power)
    expanded.insert(3,'전력',pd.Series(power).shift(-1))
    expanded = expanded.loc[~expanded['날짜'].isin([20210713,20210715])].copy()
    fills = {(20210601,1):.65, (20210601,2):.65, (20210704,20):1.85}
    for (date,hour),value in fills.items():
        mask = (expanded['날짜']==date)&(expanded['시간']==hour)
        assert mask.sum()==4
        expanded.loc[mask,'풍속']=value
    rain=(expanded['날짜']==20210124)&(expanded['시간']==0)
    assert rain.sum()==4
    expanded.loc[rain,'강수량']=6.3
    return expanded


def build_calendar_and_history(df):
    dates = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d')
    holidays = {20210101,20210211,20210212,20210301,20210505,20210519,20210816}
    weekday = dates.dt.weekday.lt(5)
    df.insert(0, '주중여부', weekday.astype(int))
    df.insert(1, '주중공휴일여부', (weekday & df['날짜'].isin(holidays)).astype(int))
    df = df.drop(columns=['날짜','평균']).reset_index(drop=True)
    day = pd.to_datetime(dict(year=2021,month=df['m'],day=df['d']))
    day_index = df.groupby(['m','d'],sort=False).cumcount()
    assert (day_index.groupby(day).max()==95).all()
    assert (df['시간'].to_numpy()==(day_index//4).to_numpy()).all()
    df['15분위치']=(day_index%4).astype('int8')
    stamp=day+pd.to_timedelta(df['시간'],unit='h')+pd.to_timedelta(df['15분위치']*15,unit='m')
    assert stamp.is_monotonic_increasing and stamp.is_unique
    segment=stamp.diff().ne(pd.Timedelta(minutes=15)).cumsum()
    power=df['현재전력']
    grouped=power.groupby(segment,sort=False)
    for lag in (1,2,4,96):
        df[f'과거전력_{lag}칸']=grouped.shift(lag)
    df['전력변화_15분']=power-df['과거전력_1칸']
    df['전력변화_60분']=power-df['과거전력_4칸']
    df['최근1시간_전력평균']=grouped.transform(lambda s:s.rolling(4,min_periods=4).mean())
    df['최근1시간_전력최대']=grouped.transform(lambda s:s.rolling(4,min_periods=4).max())
    return df


def build_final_stage():
    b=prepare(build_calendar_and_history(build_first_stage()))
    b['forecast_time']=b.timestamp+pd.Timedelta(minutes=15)
    b['target_time']=b.timestamp+pd.Timedelta(minutes=30)
    # HS time and power features are reconstructed from raw, without model_ready.csv.
    hh=build_features(preprocess(pd.read_csv(RAW))[0])
    h=hh.loc[hh['목표_전력'].notna()&hh['전력_lag1'].notna()].reset_index(drop=True)
    hscols=[c for c in h if c not in ['목표시각','예측시각','목표_전력']]
    h=h.rename(columns={c:'HS_'+c for c in hscols})
    merged=b.merge(h,left_on='target_time',right_on='목표시각',validate='one_to_one',how='left')
    assert len(merged)==24190 and merged['목표_전력'].notna().all()
    np.testing.assert_array_equal(merged['전력'],merged['목표_전력'])
    np.testing.assert_array_equal(merged['현재전력'],merged['HS_전력_lag1'])
    merged['split']=np.where(merged.date_key<20210627,'train','test')
    last_train=merged.index[merged.split=='train'][-1]
    merged.loc[last_train,'split']='boundary_excluded'
    selected=json.loads((ROOT/'models/selection.json').read_text())['features']
    assert set(selected).issubset(FEATURES+['HS_'+c for c in hscols])
    out=merged[['source_row','forecast_time','target_time','date_key','split','전력']+selected]
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    out=build_final_stage()
    csv_text=out.to_csv(index=False)
    serialized=pd.read_csv(StringIO(csv_text))
    # Validate against the bundled final input before replacing it.
    if OUTPUT.exists():
        pd.testing.assert_frame_equal(serialized,pd.read_csv(OUTPUT),check_dtype=False,rtol=1e-8,atol=1e-5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(csv_text,encoding='utf-8-sig')
    os.replace(temporary,args.output)
    print(f'PASS: raw → final input; rows={len(out)}, columns={len(out.columns)}; saved {args.output}')

if __name__=='__main__':main()
