"""공통 시간 CV에서 BG/HS 변수·설정·앙상블 선택 후 시험 평가 및 저장.
python search_candidates.py      # 전체 탐색 재현
python train_model.py --final-only # 저장된 선택으로 final_input_data.csv만 사용해 재학습
최종 joblib에는 sklearn VotingRegressor를 저장하므로 사용자 정의 클래스 불필요.
"""
from pathlib import Path
import argparse, json, hashlib, platform, time
import numpy as np
import pandas as pd
import sklearn, joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HERE=Path(__file__).resolve().parent
BASE=dict(n_estimators=300,max_features=1.,min_samples_leaf=2,min_samples_split=6,max_depth=32,bootstrap=False,random_state=42,n_jobs=4)
HS=dict(n_estimators=220,max_features=1.,min_samples_leaf=1,min_samples_split=2,max_depth=None,bootstrap=False,random_state=42,n_jobs=4)

def save_json(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False))

def make(config):
    return Pipeline([('columns',ColumnTransformer([('selected','passthrough',config['features'])],remainder='drop')),
      ('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
      ('model',ExtraTreesRegressor(**config['params']))])

def cv_splits(data):
    # 외부 train 내에서만 생성: 이전 BG의 마지막 CV 경계 1행 재포함 수정.
    idx=np.flatnonzero(data.split.eq('train'))
    dates=data.date_key.to_numpy(); days=np.unique(dates[idx]); folds=[]
    for n,(td,vd) in enumerate(TimeSeriesSplit(n_splits=5).split(days),1):
        tr=idx[np.isin(dates[idx],days[td])][:-1]
        va=idx[np.isin(dates[idx],days[vd])]
        assert max(tr)<min(va) and set(va).issubset(set(idx))
        folds.append((n,tr,va))
    return folds

def metrics(y,p):
    return {'MSE':float(mean_squared_error(y,p)),'MAE':float(mean_absolute_error(y,p)), 'R2':float(r2_score(y,p))}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final-only',action='store_true');args=parser.parse_args()
    source=HERE/('final_input_data.csv' if args.final_only else 'candidate_input_data.csv')
    data=pd.read_csv(source); y=data['전력'].to_numpy(); folds=cv_splits(data)
    tr=np.flatnonzero(data.split.eq('train'));te=np.flatnonzero(data.split.eq('test'))
    all_metrics=[];fold_metrics=[]; configs={}; oof={}
    def evaluate(name,features,params):
        config={'features':features,'params':params};configs[name]=config
        pred=np.full(len(data),np.nan); ms=[];started=time.time()
        for number,a,b in folds:
            model=make(config).fit(data.iloc[a],y[a]);p=model.predict(data.iloc[b]);pred[b]=p
            m=metrics(y[b],p);ms.append(m['MSE']);fold_metrics.append({'candidate':name,'fold':number,**m})
        row={'candidate':name,'features':len(features),'CV_MSE':float(np.mean(ms)),'CV_MSE_std':float(np.std(ms)), 'seconds':time.time()-started}
        all_metrics.append(row);oof[name]=pred
        print(name,round(row['CV_MSE'],4),flush=True)
        pd.DataFrame(all_metrics).to_csv(HERE/'candidate_metrics.csv',index=False,encoding='utf-8-sig')
    if not args.final_only:
        manifest=json.loads((HERE/'data_manifest.json').read_text());bg=manifest['BG_features'];hs=manifest['HS_features']
        clock=['HS_시간_sin','HS_시간_cos','HS_요일','HS_월','HS_분']
        daily=['HS_전력_lag96'];weekly=['HS_전력_lag672'];std=['HS_이전1시간표준편차'];short=['HS_전력_lag4']
        context=[x for x in hs if x.startswith('HS_이전확정_')]
        def unique(xs):return list(dict.fromkeys(xs))
        groups={'BG':bg,'HS':hs,'BG_clock':bg+clock,'BG_std':bg+std,'BG_daily':bg+daily,'BG_weekly':bg+weekly,
          'BG_short':bg+short,'BG_context':bg+context,'BG_clock_std':bg+clock+std,
          'BG_daily_weekly':bg+daily+weekly,'BG_power':bg+daily+weekly+std+short,
          'BG_all':unique(bg+clock+daily+weekly+std+short+context)}
        for name,features in groups.items():
            evaluate(name+'_BGparams',features,BASE)
            if name in ['BG','HS','BG_clock','BG_std','BG_daily','BG_power','BG_all']:
                evaluate(name+'_HSparams',features,HS)
        # CV 최선 입력으로만 국소 파라미터를 비교한다. 시험 지표는 아직 계산하지 않는다.
        best=min(all_metrics,key=lambda r:r['CV_MSE'])['candidate'];bestconfig=configs[best]
        tweaks=[{'min_samples_leaf':1,'min_samples_split':4}, {'min_samples_leaf':2,'min_samples_split':4},
                {'min_samples_leaf':2,'min_samples_split':8}, {'min_samples_leaf':3,'min_samples_split':6},
                {'max_features':.85}, {'max_depth':None}, {'max_depth':24}, {'n_estimators':500}]
        seen={json.dumps(c,sort_keys=True) for c in configs.values()}
        for n,delta in enumerate(tweaks,1):
            params={**bestconfig['params'],**delta};cfg={'features':bestconfig['features'],'params':params}
            if json.dumps(cfg,sort_keys=True) not in seen:
                evaluate('local_'+str(n),cfg['features'],params);seen.add(json.dumps(cfg,sort_keys=True))
        # 상위 8개 후보의 OOF만 사용해 두 모델 convex blend 검토.
        top=sorted(all_metrics,key=lambda r:r['CV_MSE'])[:8];blend=[]
        single=top[0];choice={'members':[single['candidate']],'weights':[1.], 'CV_MSE':single['CV_MSE']}
        for i,a in enumerate(top):
            for b in top[i+1:]:
                for weight in np.arange(.1,1.,.1):
                    pred=weight*oof[a['candidate']]+(1-weight)*oof[b['candidate']]
                    score=float(np.mean([mean_squared_error(y[va],pred[va]) for _,_,va in folds]))
                    blend.append({'first':a['candidate'],'second':b['candidate'],'first_weight':float(weight),'CV_MSE':score})
        bestblend=min(blend,key=lambda r:r['CV_MSE'])
        # 복잡도 증가를 정당화하도록 단일 최선 대비 CV 0.5% 이상 개선 시 앙상블 선택.
        if bestblend['CV_MSE']<single['CV_MSE']*.995:
            choice={'members':[bestblend['first'],bestblend['second']], 'weights':[bestblend['first_weight'],1-bestblend['first_weight']], 'CV_MSE':bestblend['CV_MSE']}
        choice['configs']={n:configs[n] for n in choice['members']}
        choice['features']=unique(sum([configs[n]['features'] for n in choice['members']],[]))
        choice['selection_rule']='mean chronological 5-fold MSE; blend must improve best single by >=0.5%; test unseen during selection'
        choice['best_single']=single;choice['best_blend']=bestblend
        save_json('selection.json',choice) # 시험 예측 전에 선택 저장.
        save_json('candidate_configs.json',configs)
        pd.DataFrame(blend).to_csv(HERE/'blend_candidates.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(fold_metrics).to_csv(HERE/'candidate_fold_metrics.csv',index=False,encoding='utf-8-sig')
        oof_pred=sum(w*oof[n] for n,w in zip(choice['members'],choice['weights']))
        valid=np.isfinite(oof_pred)
        pd.DataFrame({'source_row':data.source_row[valid],'actual':y[valid],'predicted':oof_pred[valid]}).to_csv(HERE/'selected_oof_predictions.csv',index=False,encoding='utf-8-sig')
        finalfold=[{'fold':n,'train_rows':len(a),'valid_rows':len(b),'train_end':data.target_time.iloc[a[-1]],'valid_start':data.target_time.iloc[b[0]],'valid_end':data.target_time.iloc[b[-1]],**metrics(y[b],oof_pred[b])} for n,a,b in folds]
        pd.DataFrame(finalfold).to_csv(HERE/'selected_fold_metrics.csv',index=False,encoding='utf-8-sig')
        data[['source_row','forecast_time','target_time','date_key','split','전력']+choice['features']].to_csv(HERE/'final_input_data.csv',index=False,encoding='utf-8-sig')
    else:
        choice=json.loads((HERE/'selection.json').read_text());configs=choice['configs']
    model=VotingRegressor([(name,make(choice['configs'][name])) for name in choice['members']],weights=choice['weights'],n_jobs=1)
    model.fit(data.iloc[tr][choice['features']],y[tr]);pred=model.predict(data.iloc[te][choice['features']])
    results=[{'model':'FINAL','CV_MSE':choice['CV_MSE'],**metrics(y[te],pred)}]
    predictions=data.iloc[te][['source_row','forecast_time','target_time']].copy();predictions['actual']=y[te];predictions['FINAL']=pred
    if not args.final_only:
        for name in ['BG_BGparams','HS_HSparams']:
            reference=make(configs[name]).fit(data.iloc[tr],y[tr]);p=reference.predict(data.iloc[te]);predictions[name]=p
            results.append({'model':name,'CV_MSE':next(r['CV_MSE'] for r in all_metrics if r['candidate']==name),**metrics(y[te],p)})
        pd.DataFrame(results).to_csv(HERE/'comparison_metrics.csv',index=False,encoding='utf-8-sig')
    predictions.to_csv(HERE/('retrained_test_predictions.csv' if args.final_only else 'test_predictions.csv'),index=False,encoding='utf-8-sig')
    threshold=float(np.quantile(y[tr],.9));high=y[te]>=threshold
    result={'selection':choice,'test':metrics(y[te],pred),'train':metrics(y[tr],model.predict(data.iloc[tr][choice['features']])),
      'peak_threshold_train_q90':threshold,'peak_test_MSE':float(mean_squared_error(y[te][high],pred[high])),
      'peak_underprediction_rate':float(np.mean(pred[high]<y[te][high])),
      'train_rows':len(tr),'test_rows':len(te),'input_features':len(choice['features']),
      'test_start':data.target_time.iloc[te[0]],'test_end':data.target_time.iloc[te[-1]],
      'train_end':data.target_time.iloc[tr[-1]],'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},
      'note':'same previously observed holdout; CV tuning estimates optimistic after model selection; no new independent test',
      'final_input_sha256':hashlib.sha256((HERE/'final_input_data.csv').read_bytes()).hexdigest()}
    save_json('run_summary.json',result)
    joblib.dump({'estimator':model,'features':choice['features'],'selection':choice,'target':'next 15-minute power','summary':result},HERE/'final_model.joblib',compress=3)
    loaded=joblib.load(HERE/'final_model.joblib');np.testing.assert_allclose(loaded['estimator'].predict(data.iloc[te[:64]][loaded['features']]),pred[:64],rtol=1e-12,atol=1e-12)
    print('FINAL',results,flush=True)

if __name__=='__main__':main()
