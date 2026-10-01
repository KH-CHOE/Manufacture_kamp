"""Rebuild all model input stages using only files inside this dashboard folder.

Run: python3 pipeline/rebuild_data.py
Outputs go to data/generated; packaged production inputs remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from features import POWER_COLUMNS, FEATURES, prepare, preprocess, build_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw/okm_augumented_2021.csv'
OUTPUT = ROOT/'data/generated'
REFERENCE = ROOT/'data/preprocessed'


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
    expanded.to_csv(OUTPUT/'okm_augumented_2021_preprocssed.csv',index=False,encoding='utf-8-sig')


def build_final_stage():
    b=prepare(OUTPUT/'okm_augumented_2021_preprocssed_3.csv')
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
    out.to_csv(OUTPUT/'final_input_data.csv',index=False,encoding='utf-8-sig')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    build_first_stage()
    for name in ['build_preprocssed_2.py','build_preprocssed_3.py']:
        subprocess.run([sys.executable,str(Path(__file__).parent/name)],check=True)
    build_final_stage()
    comparisons=[]
    for file in sorted(OUTPUT.glob('*.csv')):
        reference=REFERENCE/file.name
        actual=pd.read_csv(file)
        expected=pd.read_csv(reference)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-8,atol=1e-5)
        comparisons.append({'file':file.name,'rows':len(actual),'columns':len(actual.columns),'matches_packaged_reference':True})
    manifest={'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'comparisons':comparisons,
              'target':'next 15-minute power','known_context_assumption':'same-hour production and weather are known',
              'scope':'Historical reproduction for supplied 2021 dataset; not a general online preprocessing service.'}
    (OUTPUT/'reproduction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('PASS: all four generated CSVs match packaged preprocessing and final model input.')

if __name__=='__main__':main()
