"""최종 변수 CSV를 받아 예측. raw 시간별 CSV가 아닌 가공된 입력을 사용한다.
python predict.py --input final_input_data.csv --output prediction_replay.csv
split 열이 있으면 기본적으로 test만 예측. --all-rows로 전체 행 예측 가능.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=HERE/'final_input_data.csv')
    p.add_argument('--model',type=Path,default=HERE/'final_model.joblib')
    p.add_argument('--output',type=Path,default=HERE/'prediction_replay.csv')
    p.add_argument('--all-rows',action='store_true');a=p.parse_args()
    bundle=joblib.load(a.model);df=pd.read_csv(a.input)
    if 'split' in df and not a.all_rows:df=df.loc[df.split=='test'].copy()
    missing=set(bundle['features'])-set(df)
    if missing:raise ValueError(f'입력 변수 부족: {sorted(missing)}')
    result=df[[c for c in ['source_row','forecast_time','target_time'] if c in df]].copy()
    result['predicted_power']=bundle['estimator'].predict(df[bundle['features']])
    a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False,encoding='utf-8-sig')
    print(a.output,len(result))
if __name__=='__main__':main()
