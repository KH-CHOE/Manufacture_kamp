#!/usr/bin/env python3
"""제출 모델 재학습·평가. 최종 입력 CSV와 selection.json만 사용한다.

실행: python train_model.py
출력: final_model.joblib, reproduced_metrics.csv,
      reproduced_folds.csv, reproduced_test_predictions.csv
보고서 생성이나 하이퍼파라미터 탐색은 수행하지 않는다. 최종 구성은 훈련 CV로
이미 선정되어 selection.json에 고정돼 있다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
EXCLUDED = {'source_row', 'forecast_time', 'target_time', 'date_key', 'split', '전력'}


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        'MSE': float(mean_squared_error(actual, predicted)),
        'MAE': float(mean_absolute_error(actual, predicted)),
        'R2': float(r2_score(actual, predicted)),
    }


def build_estimator(selection: dict) -> VotingRegressor:
    estimators = []
    for name in selection['members']:
        settings = selection['configs'][name]
        steps = [
            ('columns', ColumnTransformer(
                [('selected', 'passthrough', settings['features'])], remainder='drop')),
            ('imputer', SimpleImputer(
                strategy='median', add_indicator=True, keep_empty_features=True)),
            ('model', ExtraTreesRegressor(**settings['params'])),
        ]
        estimators.append((name, Pipeline(steps)))
    return VotingRegressor(estimators=estimators, weights=selection['weights'], n_jobs=1)


def split_data(frame: pd.DataFrame):
    train = np.flatnonzero(frame['split'].eq('train').to_numpy())
    test = np.flatnonzero(frame['split'].eq('test').to_numpy())
    boundary = np.flatnonzero(frame['split'].eq('boundary_excluded').to_numpy())
    assert (len(frame), len(train), len(test), len(boundary)) == (24190, 16895, 7294, 1)
    assert train[-1] < boundary[0] < test[0]
    assert frame['target_time'].is_monotonic_increasing
    days = np.unique(frame['date_key'].to_numpy()[train])
    dates = frame['date_key'].to_numpy()
    folds = []
    for number, (train_days, valid_days) in enumerate(TimeSeriesSplit(n_splits=5).split(days), 1):
        tr = train[np.isin(dates[train], days[train_days])][:-1]
        valid = train[np.isin(dates[train], days[valid_days])]
        assert tr.size and valid.size and tr[-1] < valid[0]
        folds.append((number, tr, valid))
    return train, test, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'final_input_data.csv')
    parser.add_argument('--selection', type=Path, default=HERE / 'selection.json')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, parse_dates=['forecast_time', 'target_time'])
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    features = selection['features']
    if len(features) != 35 or len(features) != len(set(features)):
        raise ValueError('최종 선택 변수 35개의 구성이 다릅니다.')
    if any(feature in EXCLUDED for feature in features):
        raise ValueError('정답 또는 관리 열이 입력에 포함됐습니다.')
    if not set(features).issubset(frame.columns):
        raise ValueError(f'입력 열 누락: {sorted(set(features) - set(frame.columns))}')
    if not (frame['target_time'] - frame['forecast_time']).eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('모든 정답 시각이 예측 시각의 15분 뒤여야 합니다.')
    if frame[features + ['전력']].isna().any().any():
        raise ValueError('최종 평가 데이터의 선택 입력 또는 정답에 결측이 있습니다.')
    train, test, folds = split_data(frame)
    y = frame['전력'].to_numpy()
    x = frame[features]
    fold_rows = []
    for number, tr, va in folds:
        model = build_estimator(selection).fit(x.iloc[tr], y[tr])
        scores = score(y[va], model.predict(x.iloc[va]))
        fold_rows.append({
            'fold': number, 'train_rows': len(tr), 'valid_rows': len(va),
            'train_end': frame['target_time'].iloc[tr[-1]],
            'valid_start': frame['target_time'].iloc[va[0]],
            'valid_end': frame['target_time'].iloc[va[-1]],
            **scores,
        })
    fold_table = pd.DataFrame(fold_rows)
    cv_mse = float(fold_table['MSE'].mean())
    if abs(cv_mse - selection['CV_MSE']) > 1e-8:
        raise ValueError(f'저장된 CV MSE와 재계산 결과 불일치: {cv_mse}')

    model = build_estimator(selection).fit(x.iloc[train], y[train])
    predicted = model.predict(x.iloc[test])
    test_scores = score(y[test], predicted)
    summary = {
        'target': '현재 관측 후 다음 15분 전력',
        'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'usable_rows': len(frame), 'train_rows': len(train), 'test_rows': len(test),
        'excluded_boundary_rows': 1, 'feature_count': len(features),
        'features': features, 'selection': selection,
        'CV_MSE': cv_mse, 'CV_MSE_std': float(fold_table['MSE'].std(ddof=0)),
        'CV_MAE': float(fold_table['MAE'].mean()),
        'train': score(y[train], model.predict(x.iloc[train])),
        'test': test_scores,
        'test_start': str(frame['target_time'].iloc[test[0]]),
        'test_end': str(frame['target_time'].iloc[test[-1]]),
        'versions': {
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
        },
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(out / 'reproduced_folds.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'CV_MSE': cv_mse, 'CV_MAE': summary['CV_MAE'],
                   **{'test_' + key: val for key, val in test_scores.items()}}]).to_csv(
        out / 'reproduced_metrics.csv', index=False, encoding='utf-8-sig')
    prediction_table = frame.iloc[test][['source_row', 'forecast_time', 'target_time']].copy()
    prediction_table['actual'] = y[test]
    prediction_table['predicted'] = predicted
    prediction_table['residual'] = y[test] - predicted
    prediction_table.to_csv(out / 'reproduced_test_predictions.csv', index=False, encoding='utf-8-sig')
    bundle = {'estimator': model, 'features': features, 'selection': selection,
              'target': 'next 15-minute power', 'summary': summary}
    joblib.dump(bundle, out / 'final_model.joblib', compress=3)
    (out / 'reproduction_metadata.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    restored = joblib.load(out / 'final_model.joblib')
    np.testing.assert_allclose(
        restored['estimator'].predict(x.iloc[test[:64]]), predicted[:64], rtol=1e-12, atol=1e-12)
    print(f"CV MSE {cv_mse:.6f}; 시험 MSE {test_scores['MSE']:.6f}; MAE {test_scores['MAE']:.6f}; R² {test_scores['R2']:.6f}")
    print('결과 저장:', out.resolve())


if __name__ == '__main__':
    main()
