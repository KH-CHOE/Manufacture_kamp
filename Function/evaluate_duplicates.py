"""학습에 있던 하루 전력 곡선이 검증에 반복된 영향을 같은 행에서 비교한다.

모형을 다시 맞추거나 시험 점수로 설정을 고르지 않는다. 반복 날짜를 평가에서만
제외한 보조 점수이며, 제외 후 자료가 적으면 일반화 성능의 확정 근거가 될 수 없다.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import config as C
import modeling as M


def evaluate(data_path=C.OUT_DEFAULT, result_path=C.MODEL_DIR/'results.json'):
    original = M.load(data_path)
    rows = original[M.usable(original)].reset_index(drop=True)
    folds = M.folds_from_data(rows, C.FORWARD_FOLDS, C.FORWARD_FOLD_DAYS)
    fingerprints = {}
    for day, group in original.groupby('date_key'):
        if len(group) == C.PER_DAY and M.window_ok(group, C.PER_DAY)[-1]:
            fingerprints[int(day)] = hashlib.sha256(group.kW.to_numpy('<f8').tobytes()).hexdigest()
    # 결합 비중은 학습이 전진검증으로 고른 값을 쓴다(results.json)
    w = json.loads(Path(result_path).read_text(encoding='utf-8'))['설정']['blend_weight']
    trees = np.load(C.HERE/'_pred_trees.npz')
    nets = np.load(C.HERE/'_pred_nets.npz')
    out = {'definition': '동일한 96개 전력값의 하루 곡선이 평가 시작 전 자료에 있었던 날짜',
           'limitation': '불완전한 날짜와 근사 중복은 검사하지 않음. 모델은 재학습하지 않고 평가 날짜만 제외',
           'folds': {}}
    for start, end, label in M.jobs(rows, folds):
        before = {v for k,v in fingerprints.items() if k < start}
        test = rows[(rows.date_key >= start) & (True if end is None else rows.date_key < end)]
        duplicate = sorted(int(k) for k in test.date_key.unique() if fingerprints.get(int(k)) in before)
        keep = ~test.date_key.isin(duplicate).to_numpy()
        train = rows[rows.date_key < start]
        predictions = {'persistence': test.kw_lag1.to_numpy()}
        med = train.groupby(['시간','15분위치','dow']).y.median()
        p = test.set_index(['시간','15분위치','dow']).index.map(med).to_numpy(float)
        predictions['tod_dow'] = np.where(np.isnan(p), train.y.median(), p)
        for name in ('rf','et','hgb','et_wx'):
            key = f'{name}|{label}'
            if key in trees.files:
                predictions[name] = trees[key]
        for name in ('gru','lstm'):
            key = f'{name}|{label}'
            if key in nets.files:
                if not np.array_equal(nets['row_index'][nets[f'{name}|idx|{label}']], test.index):
                    raise ValueError('신경망 예측 행이 다릅니다')
                predictions[name] = nets[key]
        if 'et' in predictions and 'gru' in predictions:
            predictions['ensemble'] = w*predictions['et']+(1-w)*predictions['gru']
        scores = {}
        for name, pred in predictions.items():
            if len(pred) != len(test):
                raise ValueError('평가 행 길이 불일치')
            scores[name] = {**M.score(test.y.to_numpy()[keep], pred[keep]), 'rows': int(keep.sum())} if keep.any() else None
        out['folds'][label] = {'days': int(test.date_key.nunique()), 'duplicate_days': duplicate,
                               'retained_days': int(test.loc[keep,'date_key'].nunique()),
                               'retained_rows': int(keep.sum()), 'scores': scores}
    out['nonduplicate_cv_mean_mse'] = {
        name: round(float(np.mean([out['folds'][str(lo)]['scores'][name]['MSE'] for lo,_ in folds])),4)
        for name in predictions
        if all(out['folds'][str(lo)]['scores'].get(name) for lo,_ in folds)}
    destination = Path(result_path).with_name('duplicate_evaluation.json')
    destination.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out


if __name__ == '__main__':
    evaluate()
