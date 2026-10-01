#!/usr/bin/env python3
"""6단계 제출용 모델링 보고서 생성. 학습 없이 evidence/와 최종 CSV를 읽는다."""
from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'


def js(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(EVIDENCE / name)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def n(value, digits=2) -> str:
    return f'{float(value):,.{digits}f}'


def table(frame: pd.DataFrame, max_height=None) -> str:
    data = frame.copy()
    for col in data.select_dtypes(include='number'):
        if data[col].dtype.kind == 'f':
            data[col] = data[col].map(lambda v: n(v, 3) if pd.notna(v) else '—')
    cls = 'table-scroll tall' if max_height else 'table-scroll'
    return f'<div class="{cls}">' + data.to_html(index=False, escape=True, border=0) + '</div>'


def line_chart(series: list[tuple[str, np.ndarray, str]], width=890, height=250) -> str:
    values = np.concatenate([np.asarray(y, float) for _, y, _ in series])
    low, high = float(values.min()), float(values.max())
    margin = max(1, (high - low) * .08)
    low, high = low - margin, high + margin
    x0, x1, y0, y1 = 52, width - 18, 16, height - 34
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="시간에 따른 실제 전력과 예측 전력">']
    for tick in np.linspace(low, high, 5):
        y = y1 - (tick - low) / (high - low) * (y1 - y0)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#d9e4ec"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5a6e7b">{tick:.0f}</text>')
    for label, values, color in series:
        xs = np.linspace(x0, x1, len(values))
        ys = y1 - (np.asarray(values) - low) / (high - low) * (y1 - y0)
        points = ' '.join(f'{x:.1f},{y:.1f}' for x, y in zip(xs, ys))
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="{points}"/>')
    out.append(f'<text x="{x0}" y="{height - 7}" font-size="11" fill="#5a6e7b">구간 시작</text>')
    out.append(f'<text x="{x1}" y="{height - 7}" text-anchor="end" font-size="11" fill="#5a6e7b">구간 끝</text></svg>')
    return ''.join(out)


def bar_chart(items: list[tuple[str, float, str]], maximum=None) -> str:
    maximum = maximum or max(v for _, v, _ in items)
    return '<div class="bar-chart">' + ''.join(
        f'<div class="bar-item"><span>{esc(label)}</span><div class="track"><div style="width:{max(1, min(100, value / maximum * 100)):.1f}%;background:{color}"></div></div><b>{n(value)}</b></div>'
        for label, value, color in items
    ) + '</div>'


def feature_description(name: str) -> tuple[str, str]:
    if name in {'주중여부', '주중공휴일여부'}:
        return '달력', '2021년 달력 기준 평일 및 평일 공휴일 여부. 두 값 모두 0 또는 1.'
    if name in {'시간', '15분위치', 'day', 'd', 'm'}:
        mapping = {'시간': '원본 시간(0–23)', '15분위치': '해당 시간의 4개 관측 위치(0–3)',
                   'day': '원본 day 코드', 'd': '월의 일', 'm': '월'}
        return '달력', mapping[name]
    if name == '현재전력':
        return '전력 이력', '예측 시각 t에 마지막으로 관측된 전력.'
    if name.startswith('과거전력_'):
        lag = int(name.split('_')[1].removesuffix('칸'))
        return '전력 이력', f'현재 시각 t의 {lag}×15분 전 관측 전력.'
    if name.startswith('전력변화_'):
        return '전력 변화', f'현재전력과 {name.removeprefix("전력변화_")} 전 전력의 차이.'
    if name.startswith('최근'):
        return '이동 통계', '현재 및 이전 관측만 사용한 해당 구간의 평균 또는 최대.'
    if name in {'생산량', '기온', '풍속', '습도', '강수량'}:
        return '생산·기상', '원본의 동일 시간 행 값. 예측 시점에 알려진다고 가정.'
    if name in {'HS_시간_sin', 'HS_시간_cos'}:
        return '순환 시간', '목표 시각 T의 시각을 24시간 주기로 변환한 sin 또는 cos 값.'
    if name in {'HS_요일', 'HS_월', 'HS_분'}:
        return '목표 달력', '목표 시각 T의 요일(월요일 0), 월 또는 분.'
    if name == 'HS_이전1시간표준편차':
        return '전력 변동성', '목표 이전 1시간의 전력 4개에 대한 표본 표준편차.'
    return '기타', '선정 변수 목록을 확인.'


summary = js('final_summary.json')
manifest = js('data_manifest.json')
feature_meta = js('feature_sets_metadata.json')
diag_meta = js('diagnostics_metadata.json')
wide_meta = js('wide_search_metadata.json')
local_meta = js('local_search_metadata.json')
selection = json.loads((HERE / 'selection.json').read_text(encoding='utf-8'))
final_data = pd.read_csv(HERE / 'final_input_data.csv')
prediction = csv('final_predictions.csv')
initial = csv('baseline_same_rows.csv')
sweep = csv('feature_sets_cv.csv')
selected = csv('feature_sets_selected_test.csv')
diagnostics = csv('diagnostics.csv')
wide = csv('wide_search.csv')
local = csv('local_search.csv')
last_candidates = csv('final_candidates.csv')
final_comparison = csv('final_comparison.csv')
folds = csv('final_folds.csv')
blend = csv('final_blends.csv')
configs = js('final_candidate_configs.json')
model_bundle = joblib.load(HERE / 'final_model.joblib')
feature_list = selection['features']
model = model_bundle['estimator']
selected_model = model.estimators_[0].named_steps['model']
imputer = model.estimators_[0].named_steps['imputer']
assert feature_list == model_bundle['features'] and len(feature_list) == 35
assert not final_data[feature_list].isna().any().any()

# 비교 표: 동일 24,190행의 4개 초기 모델과 9개 기존 파생 변수를 나란히 제시.
initial_table = initial.copy()
initial_table.columns = ['입력', '모델', 'CV MSE', 'CV MAE', '시험 MSE', '시험 MAE']
initial_table = initial_table.sort_values(['모델', '입력']).reset_index(drop=True)

# A–G의 수치 전체를 표시해 F 선택에 이르는 경로를 보여 준다.
labels = list(feature_meta['feature_sets'])
model_order = ['ExtraTrees', 'HistGradientBoosting', 'RandomForest', 'GradientBoosting']
feature_table = pd.DataFrame([
    {'묶음': key, '입력 수': len(feature_meta['feature_sets'][key]),
     **{model: float(sweep.loc[(sweep.feature_set == key) & (sweep.model == model), 'CV_MSE_mean'].iloc[0]) for model in model_order}}
    for key in labels
])
selected_table = selected[['model', 'feature_set', 'CV_MSE_mean', 'test_MSE', 'test_MAE']].copy()
selected_table.columns = ['모델', 'CV 선택 입력', 'CV MSE', '시험 MSE', '시험 MAE']

comparison = final_comparison.copy()
comparison['model'] = comparison['model'].replace({'FINAL': '최종 35변수 ExtraTrees',
   'BG_BGparams': '이전 ExtraTrees 설정·29변수', 'HS_HSparams': '다른 ExtraTrees 설정·17변수'})
comparison.columns = ['동일 기간 비교', 'CV MSE', '시험 MSE', '시험 MAE', '시험 R²']
comparison = comparison.sort_values('시험 MSE')

feature_table_final = pd.DataFrame([
    {'번호': i, '변수': value, '묶음': feature_description(value)[0], '의미': feature_description(value)[1]}
    for i, value in enumerate(feature_list, 1)
])
parameter_rows = []
for key, value in selected_model.get_params(deep=False).items():
    parameter_rows.append({'설정': key, '값': str(value),
        '탐색 여부': '최종 탐색' if key in {'n_estimators', 'max_features', 'min_samples_leaf', 'min_samples_split', 'max_depth', 'bootstrap'} else '고정 또는 기본값'})
params_table = pd.DataFrame(parameter_rows)
preprocessing_table = pd.DataFrame([
    {'단계': '입력 선택', '방법': 'ColumnTransformer', '설정': '선택한 35개 열을 저장 순서대로 통과, 나머지 열 drop'},
    {'단계': '잔여 결측 보호', '방법': 'SimpleImputer', '설정': 'strategy=median, add_indicator=True, keep_empty_features=True; fold 학습에서만 fit'},
    {'단계': '수치 스케일링', '방법': '없음', '설정': 'ExtraTrees는 변수 크기에 의한 거리 계산을 하지 않음; 최종 입력 35개 결측 0개'},
    {'단계': '예측기', '방법': 'ExtraTreesRegressor', '설정': 'n_estimators=300, min_samples_leaf=3, min_samples_split=6, max_depth=32, max_features=1.0'},
    {'단계': '최종 래퍼', '방법': 'VotingRegressor', '설정': '선정 결과 단일 모델, weight=1.0, n_jobs=1'},
])
raw_missing = pd.DataFrame([
    {'원본 열': '풍속', '결측 시간별 행': 3, '처리': '2021-06-01 01·02시 0.65; 2021-07-04 20시 1.85'},
    {'원본 열': '강수량', '결측 시간별 행': 1, '처리': '2021-01-24 00시 6.3 (세 후보의 RF MSE 비교 후 채택)'},
    {'원본 열': '공장인원', '결측 시간별 행': 17, '처리': '대체하지 않고 모델 입력에서 제외'},
])
rows_flow = pd.DataFrame([
    {'시점': '원본 시간별 CSV', '행 수': '6,168', '변경': '2021-01-01~2021-09-14, 시간별 행에 15·30·45·60분 전력'},
    {'시점': '시간 오류 제거', '행 수': '6,120', '변경': '7월 13일·15일 각각 24시간 행, 합계 48행 제거'},
    {'시점': '15분 단위 재구성', '행 수': '24,480', '변경': '시간별 4개 전력을 펼치고 다음 15분 정답을 한 칸 이동'},
    {'시점': '공통 모델링 가능', '행 수': '24,190', '변경': '연속 구간 내 96칸 이력 필요, 삭제일 경계 및 마지막 정답 제거'},
    {'시점': '최종 평가', '행 수': '24,189', '변경': '학습 16,895 + 시험 7,294; 경계 1행 제외'},
])
fold_table = folds[['fold', 'train_rows', 'valid_rows', 'train_end', 'valid_start', 'valid_end', 'MSE', 'MAE', 'R2']].copy()
fold_table.columns = ['fold', '학습 수', '검증 수', '학습 마지막 목표 시각', '검증 첫 목표 시각', '검증 끝 목표 시각', 'MSE', 'MAE', 'R²']

# 마지막 2일과 시험 전체의 여러 조건별 오차를 보고서가 직접 계산한다.
with_features = prediction.merge(final_data[['source_row', '현재전력', '주중여부']], on='source_row', validate='one_to_one')
with_features['squared_error'] = (with_features['actual'] - with_features['FINAL']) ** 2
with_features['abs_error'] = abs(with_features['actual'] - with_features['FINAL'])
threshold = float(summary['peak_threshold_train_q90'])
segment_masks = [
    ('전체', np.ones(len(with_features), dtype=bool)),
    ('평일', with_features['주중여부'].eq(1)),
    ('주말', with_features['주중여부'].eq(0)),
    (f'고전력 ≥ {threshold:.0f}', with_features['actual'].ge(threshold)),
    ('전력 변화 ≥ 20', (with_features['actual'] - with_features['현재전력']).abs().ge(20)),
]
segment_rows = []
for name, mask in segment_masks:
    subset = with_features.loc[mask]
    segment_rows.append({'구간': name, '시험 행 수': len(subset), 'MSE': float(subset.squared_error.mean()),
                         'MAE': float(subset.abs_error.mean()), '과소예측 비율': float((subset.FINAL < subset.actual).mean())})
segments = pd.DataFrame(segment_rows)

# '탐색한 파라미터'와 '최종 선택 설정'을 각각 표로 분리.
wide_grid = pd.DataFrame([{'파라미터': k, '탐색 값': ', '.join(map(str, v))} for k, v in wide_meta['search_space'].items()])
local_grid = pd.DataFrame([{'파라미터': k, '탐색 값': ', '.join(map(str, v))} for k, v in local_meta['search_space'].items()])
final_configs = pd.DataFrame([
    {'후보': row.candidate, '입력 수': int(row.features), 'CV MSE': float(row.CV_MSE),
     'CV 표준편차': float(row.CV_MSE_std),
     'n_estimators': configs[row.candidate]['params']['n_estimators'],
     'min_samples_leaf': configs[row.candidate]['params']['min_samples_leaf'],
     'min_samples_split': configs[row.candidate]['params']['min_samples_split'],
     'max_depth': str(configs[row.candidate]['params']['max_depth']),
     'max_features': configs[row.candidate]['params']['max_features']}
    for row in last_candidates.sort_values('CV_MSE').itertuples()
])

# 전력 흐름: 시험 첫 192개 관측, 초기 2일의 모델 성능을 시간축에서 확인.
window = prediction.iloc[:192]
forecast_plot = line_chart([
    ('실제', window.actual.to_numpy(), '#008a8c'),
    ('최종', window.FINAL.to_numpy(), '#2462a9'),
    ('기존', window.BG_BGparams.to_numpy(), '#b47727'),
])
fold_values = []
for row in folds.itertuples():
    fold_values.append((f'fold {int(row.fold)}', float(row.MSE), '#2462a9'))
fold_plot = bar_chart(fold_values, max(75, max(x[1] for x in fold_values)))

css = r'''
:root{--ink:#173047;--blue:#2462a9;--aqua:#008a8c;--gold:#b47727;--paper:#f5f8fa;--line:#d8e3eb;--muted:#5a6e7b;--white:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.78 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-variant-numeric:tabular-nums}a{color:var(--blue)}a:focus-visible{outline:3px solid var(--gold);outline-offset:3px}h1,h2,h3{line-height:1.36;letter-spacing:-.035em}h1{font-size:42px;margin:0 0 18px}h2{font-size:28px;margin:0 0 24px}h3{font-size:18px;margin:29px 0 11px}p{max-width:84ch;margin:0 0 16px}ul{padding-left:22px}li{margin:5px 0}.topbar{background:var(--ink);color:white;padding:20px max(24px,calc((100vw - 1280px)/2));font-size:14px;font-weight:700}.frame{max-width:1320px;margin:auto;display:grid;grid-template-columns:180px minmax(0,1fr);gap:40px;padding:40px 24px 90px}aside{align-self:start;position:sticky;top:25px}aside a{display:block;padding:8px 11px;border-left:2px solid var(--line);text-decoration:none;font-size:12px;color:var(--muted)}aside a:hover{color:var(--blue);background:#e8f0f7;border-color:var(--blue)}main{min-width:0}.cover{background:#e6eff6;padding:48px 42px;margin-bottom:26px;position:relative;border-top:5px solid var(--aqua)}.cover p{font-size:18px;color:#415a6c}.cover small{display:block;font-size:12px;letter-spacing:.05em;margin-bottom:20px}.docket{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-top:30px}.docket>div{border-top:1px solid #a6c5d3;padding-top:12px}.docket b{font-size:23px;color:var(--blue);display:block}.docket span{font-size:12px;color:var(--muted)}section{background:white;padding:35px 40px;margin-bottom:24px;border:1px solid var(--line)}.section-number{font-size:12px;color:var(--aqua);font-weight:700;display:block;margin-bottom:6px}.lede{font-size:17px;color:#456073}.path{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);margin:25px 0}.path div{padding:18px 12px;border-right:1px solid var(--line)}.path div:last-child{border:0}.path b{display:block;color:var(--aqua);font-size:12px}.path strong{display:block;font-size:14px}.path span{display:block;font-size:11px;color:var(--muted)}.table-scroll{overflow:auto;border:1px solid var(--line);margin:17px 0}.table-scroll.tall{max-height:590px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px}th,td{padding:10px 13px;border-bottom:1px solid #e8eef2;text-align:left}thead th{background:#eaf1f6;position:sticky;top:0;color:#35526b}tr:last-child td{border-bottom:0}tbody tr:hover{background:#f4f8fb}.note{padding:16px 20px;background:#edf4fa;border-left:4px solid var(--blue);margin:21px 0}.note strong{display:block;margin-bottom:4px}.note p:last-child{margin:0}.note.caution{background:#fff8e9;border-color:var(--gold)}.duo{display:grid;grid-template-columns:1fr 1fr;gap:15px}.summary-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.summary-grid div{border-top:3px solid var(--blue);background:#f4f8fb;padding:15px}.summary-grid small{display:block;color:var(--muted)}.summary-grid b{display:block;font-size:23px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:18px;height:3px;background:var(--swatch);vertical-align:middle;margin-right:6px}svg{display:block;width:100%;max-width:100%;margin:18px 0}.bar-chart{display:grid;gap:9px;margin:18px 0}.bar-item{display:grid;grid-template-columns:90px 1fr 65px;gap:12px;align-items:center;font-size:12px}.track{height:20px;background:#e8f0f5}.track div{height:100%}.code{background:#e8f0f6;padding:2px 5px;font:12px/1.6 ui-monospace,Menlo,monospace;overflow-wrap:anywhere}pre{background:var(--ink);color:white;padding:19px;overflow:auto}pre .code{background:none;color:inherit}.definition{font-size:13px;color:var(--muted)}footer{font-size:12px;color:var(--muted);padding:10px 0 30px}@media(max-width:980px){.frame{display:block}aside{position:static;display:flex;overflow:auto;margin-bottom:20px}aside a{white-space:nowrap}.path{grid-template-columns:repeat(3,1fr)}}@media(max-width:620px){.frame{padding:20px 12px}.cover,section{padding:24px 17px}h1{font-size:30px}h2{font-size:23px}.docket,.duo,.summary-grid{grid-template-columns:1fr}.path{grid-template-columns:1fr}.path div{border-right:0;border-bottom:1px solid var(--line)}}@media print{body{background:white}.frame{display:block;padding:0}aside{display:none}.cover,section{break-inside:avoid}.table-scroll{overflow:visible}.table-scroll.tall{max-height:none}a{text-decoration:none;color:var(--ink)}}
'''

nav = [('goal', '문제 정의'), ('data', '데이터·전처리'), ('features', '입력 변수'), ('validation', '검증 설계'),
       ('development', '모델 탐색'), ('final', '최종 학습'), ('results', '시험 결과'), ('reproduction', '재현·한계'), ('appendix', '부록')]
links = ''.join(f'<a href="#{target}">{label}</a>' for target, label in nav)
path = ''.join(f'<div><b>{i:02d}</b><strong>{title}</strong><span>{desc}</span></div>' for i, (title, desc) in enumerate([
    ('원본 검사', '시간 오류·결측 확인'), ('15분 재구성', '현재/다음 전력 연결'), ('과거 변수', '연속 구간 내 lag·rolling'),
    ('시간 검증', '70:30·5-fold'), ('최종 학습', '35변수 ExtraTrees')], 1))

page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>다음 15분 전력 예측 | 최종 모델링 보고서</title><style>{css}</style></head><body>
<div class="topbar">KAMP 자원 최적화 AI 데이터셋　/　모델 개발 및 평가 보고서</div><div class="frame"><aside>{links}</aside><main>
<div class="cover"><small>2021년 생산·기상·전력 데이터 / 지도학습 회귀</small><h1>다음 15분 전력 예측<br>모델 개발 및 평가</h1><p>원자료의 오류를 확인하고 15분 단위 학습 데이터를 구성한 뒤, 과거 전력과 시간 정보를 검증해 ExtraTrees 회귀 모델을 선정했다. 문서의 모든 평가 수치는 저장된 실험 결과에서 가져왔다.</p><div class="docket"><div><b>{n(summary['test']['MSE'])}</b><span>최종 시험 MSE</span></div><div><b>{n(summary['test']['MAE'])}</b><span>최종 시험 MAE</span></div><div><b>{n(summary['test']['R2'],4)}</b><span>최종 시험 R²</span></div></div></div>
<section id="goal"><span class="section-number">01 · 문제와 예측 대상</span><h2>공장 전력을 15분 앞서 예측한다</h2><p class="lede">각 시점 t에 확정된 현재전력과 과거 관측, 생산·기상 및 달력 정보를 입력하고, 다음 시각 T=t+15분의 전력 <span class="code">전력</span>을 예측하는 회귀 문제로 정의했다.</p><p>원본 한 시간 행에는 15·30·45·60분 전력 네 값이 들어 있다. 예를 들어 원본 0시 행의 15분 값 62가 00:15의 현재전력이라면, 30분 값 61은 00:30의 정답이다. 같은 시간 행의 설명 변수는 15분 행 네 개에 반복된다. 따라서 전력 시차·시간 위치가 한 시간 내부의 차이를 설명하도록 구성했다.</p><p>예측 전력은 이후 피크 경보 또는 가동 시점 조정에 사용할 수 있다. 이 보고서에서 직접 학습한 대상은 <b>다음 15분 전력 한 값</b>이며 전기요금 자체나 생산 일정 최적화 결과를 예측한 것은 아니다. 비용 계산에 필요한 전기요금·인건비·공장인원은 입력에서 제외했다.</p><div class="path">{path}</div></section>
<section id="data"><span class="section-number">02 · 데이터 품질과 전처리</span><h2>원자료에서 학습 가능 행까지</h2><p>원본은 2021년 1월 1일부터 9월 14일까지의 시간별 CSV 6,168행, 18개 열이다. 실제 모델은 원본에서 생성된 15분 관측을 사용한다. 오류 행 제거는 정답을 한 칸 옮긴 후 수행했고, 삭제일 양쪽을 시계열로 이어 붙이지 않았다.</p>{table(rows_flow)}<h3>시간 오류 행</h3><p>7월 13일과 15일의 <span class="code">시간</span> 값은 0~23이 아니라 각각 70~188, 74~184 범위다. 날짜와 시간의 정상적인 조합으로 확정할 수 없어 각 24행, 합계 48행을 제외했다. 15분으로 펼친 뒤에는 두 날짜에 해당하는 192행이 제거된 셈이다. 삭제된 날짜를 사이에 두고 lag나 rolling 창이 이어지지 않도록 연속 구간을 다시 나눴다.</p><h3>원본 결측치의 처리</h3>{table(raw_missing)}<p>공장인원 결측 17개는 원본 시간별 기준이며 15분으로 펼친 파일에서는 같은 값이 네 번 반복돼 68개가 된다. 공장인원과 인건비는 모델에 넣지 않았다. 풍속·강수량의 수동 대치 값은 이전 전처리 실험의 최종 결정이다. 추가 시간·전력 변동성 열은 raw로 다시 계산한 결과와 허용 오차 0.00001 안에서 대조했다.</p><h3>목표값과 이력 생성</h3><p>각 관측 전력을 현재 입력으로 두고 다음 15분 관측 전력을 정답으로 한 칸 이동했다. 따라서 가장 마지막 관측의 정답 1개는 없다. 과거전력 1·2·4·96칸, 변화량 및 1시간 이동 통계는 예측 시점까지의 전력만 사용한다. 전날 이력 96칸은 연속 관측 24시간이 있어야 생성되므로 각 연속 구간 초기에 결측이 발생한다. 공통 비교 조건을 위해 필수 파생 변수가 없는 행과 삭제일 경계에 남아 있던 정답 2행을 제거했다. 이 조건들의 중복을 반영하면 모델링 가능 표본은 24,190행이다.</p><h3>달력 변수와 제외 변수</h3><p>2021년 달력으로 주중 여부와 주중 공휴일 여부를 추가했다. 해당 기간의 평일 공휴일 7일(1/1, 2/11, 2/12, 3/1, 5/5, 5/19, 8/16)을 표시했다. 원본 <span class="code">날짜</span>와 시간별 <span class="code">평균</span> 열은 제외하고, 월·일·시간과 15분 위치는 별도 숫자 변수로 사용했다. 목표 시각 기준의 순환 시간 값도 후보로 만들어 최종 모델에 포함했다.</p><div class="note"><strong>예측 시점의 입력 가정</strong><p>같은 시간 행의 생산량과 기상값을 예측 시점에 안다는 기존 대회 모델링 가정을 적용했다. 실제 운영에서 이 값이 시간 종료 후에만 확정된다면, 해당 입력을 예보값 또는 이전 확정값으로 교체해 성능을 다시 평가해야 한다.</p></div></section>
<section id="features"><span class="section-number">03 · 입력 변수 설계</span><h2>현재 상태와 변화 추세를 함께 넣었다</h2><p>기본 12개 변수는 달력, 현재전력, 생산량, 기상값이다. 여기에 15분 위치, 현재 기준 시차 1·2·4·8·16·96칸, 변화량 4개와 1·2·4시간 평균·최대 6개를 넣어 29개를 구성했다. 마지막으로 목표 시각의 sin·cos, 요일·월·분과 최근 1시간 전력 표준편차 6개를 추가해 최종 35개 입력이 됐다.</p><p>변화량은 예를 들어 <span class="code">전력변화_60분 = 현재전력 − 과거전력_4칸</span>이다. 이동 평균·최대도 현재 또는 과거 관측만 사용한다. 시간 sin·cos는 24시와 0시가 이어지는 하루의 주기를 표현한다. 전력 표준편차는 목표 이전 네 전력 관측의 변동 폭을 나타낸다.</p><h3>최종 35개 변수 전체</h3><p><span class="code">HS_</span> 접두사는 추가 후보 변수의 생성 경로를 구분하기 위한 열 이름이며, 예측에는 표에 적힌 수치가 그대로 사용된다.</p>{table(feature_table_final, max_height=1)}<div class="note"><strong>누출 점검</strong><p>정답 <span class="code">전력</span>과 목표 시각 이후의 전력은 입력에서 제외했다. 현재전력 및 파생 이력은 t까지의 관측만 사용한다. 최종 CSV의 선택 입력 35개와 정답에는 결측치가 없다. 원본 데이터가 미래 계획·예보가 아닌 사후 생산 실적이라면 생산·기상 입력의 가용성 가정은 별도로 확인해야 한다.</p></div></section>
<section id="validation"><span class="section-number">04 · 분할과 평가 기준</span><h2>시간 순서로 학습과 시험을 분리했다</h2><p>같은 공장의 연속 관측을 무작위로 섞으면 과거·미래가 학습과 시험에 뒤섞인다. 그래서 날짜 순서의 앞 약 70%를 학습, 뒤 약 30%를 시험으로 정했다. 최종 학습은 16,895행, 시험은 7,294행이다. 경계 1행을 제외해 학습 정답과 첫 시험 입력이 직접 연결되는 부분을 제거했다.</p><p>학습의 마지막 목표 시각은 {esc(summary['train_end'])}, 시험의 첫 목표 시각은 {esc(summary['test_start'])}이다. 시험은 {esc(summary['test_end'])}까지다. 학습 내부는 날짜별 확장형 TimeSeriesSplit 5-fold이며, 매 fold에서도 학습 마지막 경계 1행을 제외했다. 마지막 fold의 검증이 외부 학습 범위를 넘지 않도록 인덱스를 제한했다.</p>{table(fold_table)}<div class="duo"><div class="note"><strong>모델 선택</strong><p>주 지표는 5-fold 검증 MSE 평균이다. <span class="code">MSE = 평균((실제−예측)²)</span>로 큰 오차를 더 강하게 반영한다. 후보와 파라미터는 검증 평균 MSE로 선정했다.</p></div><div class="note"><strong>시험 보고</strong><p>시험 MSE와 함께 <span class="code">MAE = 평균(|실제−예측|)</span>, R²를 보고한다. MAE는 원래 전력 단위의 오차다. 시험값은 최종 평가 지표이며 후보 선택 기준으로 사용하지 않았다.</p></div></div><p>최종 모델은 시험 기간에도 15분마다 직전 실제 관측이 새로 들어온다는 순차 1단계 예측 조건이다. 하루 96개 전력을 하루 시작 시점에 한꺼번에 예측하는 문제와 평가 조건이 다르다.</p></section>
<section id="development"><span class="section-number">05 · 모델과 변수 탐색</span><h2>작은 기준 모델에서 단계적으로 개선했다</h2><h3>첫 기준: 네 가지 트리 계열 회귀</h3><p>RandomForest(나무 100개, 잎 최소 2), ExtraTrees(나무 100개, 잎 최소 2), GradientBoosting(나무 120개), HistGradientBoosting(최대 반복 180)을 비교했다. 처음 12개 기본 입력과 이후 15분 위치·과거전력 변수를 추가한 입력을 <b>동일한 24,190개 공통 표본</b>에서 다시 비교한 결과는 다음과 같다. 숫자만 비교할 때 표본이 달랐던 초기 24,477행 보고서의 성적은 이 표에 섞지 않았다.</p>{table(initial_table)}<p>과거 전력 정보가 들어가자 ExtraTrees의 CV MSE는 150.143에서 53.336으로 낮아졌다. 15분 관측의 강한 관성을 입력에 반영한 효과가 컸다. GradientBoosting은 이 설정에서 개선폭이 비교적 작았다.</p><h3>공통 입력 묶음 A–G</h3><p>네 모델의 하이퍼파라미터를 고정하고 입력만 바꿨다. A=기본 12개, B=15분 위치 추가, C=최근 1시간 이력, D=C에 전날 값 추가, E=C에 최근 2·4시간 이력 추가, F=D와 최근 2·4시간 결합, G=F에 생산량 파생 3개를 추가한 구성이다. 4모델×7묶음×5-fold, 총 140개 fold 학습으로 비교했다. 아래는 검증 MSE이며 낮을수록 좋다.</p>{table(feature_table)}<p>ExtraTrees는 F 29개에서 CV MSE 41.431로 가장 좋았고 생산 파생 3개를 더한 G는 41.85로 개선되지 않았다. 모델별 최선 묶음을 시험에서 확인했을 때 ExtraTrees F의 MSE는 68.799였다.</p>{table(selected_table)}<h3>두 모델의 진단</h3><p>ExtraTrees와 HistGradientBoosting 모두 F 29개를 사용해 학습·검증·시험 오차, 잔차 분포, 고전력 구간, 변수 중복을 점검했다. ExtraTrees의 학습 MSE 3.756과 CV MSE 41.431 사이 차이가 있어 훈련 적합이 강하다. 다섯 검증 fold의 난도도 다르다. 한편 HistGradientBoosting은 시험 MSE 71.133으로 ExtraTrees의 68.799보다 높았다.</p>{table(diagnostics[['model','train_MSE','CV_MSE_mean','CV_MSE_std','test_MSE','test_MAE','test_R2']].rename(columns={'model':'모델','train_MSE':'학습 MSE','CV_MSE_mean':'CV MSE','CV_MSE_std':'CV 표준편차','test_MSE':'시험 MSE','test_MAE':'시험 MAE','test_R2':'시험 R²'}))}<p>F 29개 중 선형 독립 차원은 25였고 |상관계수|≥0.90인 쌍은 44개였다. 변화량 4개가 현재전력과 시차 전력의 정확한 차이이기 때문이다. 나무 모델에서 중복 변수를 제거한 19개 축약 입력은 ExtraTrees CV MSE 47.803으로 29개 F의 41.431보다 나빠져 F를 유지했다. 변수 중요도를 인과 효과로 해석하지 않는 것이 적절하다.</p><h3>ExtraTrees의 넓은 탐색과 좁은 탐색</h3><p>먼저 조건부 랜덤서치 36개와 기준 모델 1개를 비교했다. seed=42, 5-fold 기준 총 185개 fold 학습이다. 기준 설정(나무 100개, 잎 최소 2, 분할 최소 2, 깊이 제한 없음, bootstrap=False)이 CV MSE 41.431로 이 탐색의 1위였다.</p>{table(wide_grid)}<p>그다음 기준점 주변에서 단일 요인 11개, 무작위 조합 24개와 기준점 1개, 총 36개 후보를 검증했다. 가장 낮은 CV MSE를 낸 설정은 나무 300개·잎 최소 2·분할 최소 6·깊이 32로, CV MSE 40.696, 시험 MSE 66.799였다.</p>{table(local_grid)}<div class="note"><strong>단계별 비교의 조건</strong><p>초기 1–4단계의 CV에는 마지막 외부 학습 경계 1행이 마지막 검증 fold에 다시 포함되는 문제가 있었다. 최종 평가에서는 CV 검증을 외부 학습 인덱스 안으로 제한했다. 따라서 최종 단계와 이전 단계의 CV 숫자는 경계 조건이 소폭 다르며, 개선 여부는 마지막 단계에서 같은 수정 조건으로 재학습한 비교 표로 판단한다.</p></div><h3>최종 입력과 모델 설정의 비교</h3><p>29개 F를 기준으로 시간 sin/cos·목표 요일·월·분, 전력 표준편차, 목표 기준 전날·전주 시차, 이전 확정 생산·기상 등 입력 묶음을 나눠 비교했다. 두 ExtraTrees 설정을 교차 적용하고 최선 입력에서 잎·분할·깊이·변수 비율·나무 수를 국소 탐색한 단일 후보는 총 {len(last_candidates)}개다.</p><p>검증 상위 8개의 OOF 예측으로 두 모델 평균 가중치 0.1~0.9를 비교한 조합은 {len(blend)}개다. 단일 최선 CV MSE는 {n(selection['best_single']['CV_MSE'],3)}, 최선 앙상블은 {n(selection['best_blend']['CV_MSE'],3)}이었다. 앙상블의 개선율은 0.24% 수준이라 사전에 정한 최소 0.5% 개선 기준에 못 미쳤다. 따라서 단일 ExtraTrees를 최종 선택했다.</p>{table(last_candidates.sort_values('CV_MSE').head(10)[['candidate','features','CV_MSE','CV_MSE_std']].rename(columns={'candidate':'후보','features':'입력 수','CV_MSE':'CV MSE','CV_MSE_std':'CV 표준편차'}))}</section>
<section id="final"><span class="section-number">06 · 최종 모델 구성 및 학습</span><h2>선택된 35개 입력으로 ExtraTrees를 학습했다</h2><p>최종 선정 모델은 ExtraTrees 회귀다. F의 29개 입력에 목표 시각 sin/cos, 요일, 월, 분, 직전 1시간 전력 표준편차 6개를 추가했다. <span class="code">ColumnTransformer → SimpleImputer → ExtraTreesRegressor</span> 순으로 구성하고, 저장 형식을 통일하기 위해 한 모델·가중치 1.0의 VotingRegressor에 담았다.</p>{table(preprocessing_table)}<h3>최종 ExtraTrees 전체 파라미터</h3><p>아래는 저장된 joblib에서 읽은 ExtraTreesRegressor의 전체 설정이다. random_state=42, n_jobs=4를 사용했다. <span class="code">criterion=squared_error</span>이며 최종 출력은 회귀 전력값이다.</p>{table(params_table, max_height=1)}<h3>스케일링과 결측 대치의 역할</h3><p>ExtraTrees는 변수값의 순서와 분할 경계를 이용하므로 표준화·정규화 스케일러를 쓰지 않았다. 최종 선택 표본 24,190행×35개 입력에는 결측이 0개다. 따라서 median imputer는 이번 평가 표본의 값을 실제로 바꾸지 않았으며, 추후 같은 스키마에 결측이 유입될 때 훈련 fold의 중앙값을 사용하도록 둔 보호 단계다. 각 CV fold에서 변환기는 검증 행을 보지 않고 학습 행에만 fit한다.</p><p>검증으로 최종 설정을 고정한 뒤 외부 학습 16,895행에 적합했다. 시험 7,294행은 학습에 넣지 않았다. 제출한 joblib에는 선택 입력 순서, 설정, 예측기와 결과 요약을 함께 저장했다.</p></section>
<section id="results"><span class="section-number">07 · 시험 성능과 오류 특성</span><h2>동일 시험 기간의 MSE를 60.163까지 낮췄다</h2><p>아래 비교의 모든 모델은 같은 학습 16,895행과 같은 시험 7,294행에서 다시 적합해 평가했다. 다른 시험 기간을 사용한 서로 다른 기간의 기존 모델 점수와 혼합하지 않았다.</p>{table(comparison)}<div class="summary-grid"><div><small>CV MSE</small><b>{n(summary['selection']['CV_MSE'],3)}</b></div><div><small>시험 MSE</small><b>{n(summary['test']['MSE'],3)}</b></div><div><small>시험 MAE</small><b>{n(summary['test']['MAE'],3)}</b></div><div><small>시험 R²</small><b>{n(summary['test']['R2'],4)}</b></div></div><p>이전 29변수 ExtraTrees를 같은 최종 분할 조건에서 재학습한 시험 MSE는 66.799였다. 최종 35변수 모델의 60.163은 이 기준 대비 약 9.9% 감소다. 시험 첫 192개 관측의 실제와 예측 흐름은 다음과 같다.</p><div class="legend"><span><i style="--swatch:#008a8c"></i>실제</span><span><i style="--swatch:#2462a9"></i>최종 예측</span><span><i style="--swatch:#b47727"></i>이전 29변수 모델</span></div>{forecast_plot}<p class="definition">대상 기간: {esc(window.target_time.iloc[0])}부터 {esc(window.target_time.iloc[-1])}까지. 곡선의 가로축은 15분 순서다.</p><h3>검증 fold별 MSE</h3>{fold_plot}<p>검증 평균 MSE {n(summary['selection']['CV_MSE'],3)}의 표준편차는 {n(selection['best_single']['CV_MSE_std'],3)}이다. 특정 기간의 오차가 평균보다 클 수 있음을 보여준다.</p><h3>구간별 시험 오차</h3>{table(segments)}<p>훈련 정답의 90백분위 {n(threshold,0)} 이상을 분석용 고전력 구간으로 정의했다. 해당 구간의 시험 MSE는 {n(summary['peak_test_MSE'],3)}이고 실제보다 낮게 예측한 비율은 {n(summary['peak_underprediction_rate']*100,1)}%이다. 전력 피크 경보로 바로 사용할 때에는 과소예측을 고려한 경보 임계값과 운영 비용을 별도로 정해야 한다.</p><div class="note caution"><strong>성능 해석의 범위</strong><p>학습 MSE는 {n(summary['train']['MSE'],3)}으로 시험 MSE보다 낮다. 이는 훈련 표본 적합과 미래 기간 일반화 사이의 차이를 보여준다. 제공 기간은 2021년 9월까지이며 이후 계절·설비 조건에 대한 독립 시험은 없다. 같은 시험 기간을 이전 실험에서도 관찰했으므로 60.163을 완전히 새로운 숨은 시험 점수라고 주장할 수 없다.</p></div></section>
<section id="reproduction"><span class="section-number">08 · 산출물과 재현 방법</span><h2>입력 CSV부터 저장 모델까지 재현할 수 있다</h2><p>제출 폴더에는 입력 생성 코드, 최종 학습 코드, 보고서 생성 코드, 실제 학습 CSV, 모델 파일과 선택 설정을 함께 둔다. 저장된 모델은 사용자 정의 추론 클래스 없이 scikit-learn/joblib으로 읽을 수 있다.</p><pre><span class="code">python prepare_final_input.py  # 기존 원본·중간 전처리·시간 변수에서 최종 35변수 CSV 생성
python train_model.py          # 5-fold 재평가 → 최종 학습 → joblib 저장
python build_report.py         # 저장된 근거 CSV/JSON으로 이 HTML 생성
python predict.py              # 최종 입력의 시험 행을 사용해 저장 모델 예측 재생
python research_history/search_candidates.py  # 후보 탐색 재현(시간 소요)</span></pre><p>입력 재생성은 이 프로젝트의 원본 CSV, preprocssed_3 CSV, 5단계의 HS model_ready.csv에 접근한다. <span class="code">final_input_data.csv</span>와 <span class="code">selection.json</span>이 있으면 6단계의 학습 코드는 독립적으로 작동한다. 시험 예측을 재생할 때 정답 <span class="code">전력</span>은 입력 열로 전달하지 않는다.</p><p>환경은 Python {esc(summary['versions']['python'])}, NumPy {esc(summary['versions']['numpy'])}, pandas {esc(summary['versions']['pandas'])}, scikit-learn {esc(summary['versions']['sklearn'])}이며, 정확한 패키지 버전은 requirements.txt에 적었다. 최종 입력 CSV SHA-256은 <span class="code">{esc(summary['final_input_sha256'])}</span>이다.</p><div class="note"><strong>보고서 근거</strong><p><span class="code">evidence/</span>에 초기 모델, 변수 묶음, 진단, 랜덤 탐색, 국소 탐색, 최종 후보·fold·시험 예측의 CSV/JSON을 함께 보관했다. 보고서 생성 코드는 이 파일을 읽으며 모델을 재학습하거나 지표를 수동으로 입력하지 않는다. <span class="code">research_history/</span>에는 최종 변수 선정에 사용한 27개 단일 후보와 252개 조합을 재현하는 코드와 후보 입력 데이터를 보관했다.</p></div></section>
<section id="appendix"><span class="section-number">부록 · 후보 전체</span><h2>최종 27개 단일 후보와 탐색 범위</h2><p>이 표는 최종 단계의 27개 단일 모델을 CV MSE 오름차순으로 정리했다. 시험 오차는 후보 선택에 사용하지 않았다.</p>{table(final_configs, max_height=1)}<h3>선정 규칙과 배포 시 유의점</h3><ul><li>학습 내부 5-fold 평균 MSE가 가장 낮은 단일 후보를 우선 선정했다.</li><li>상위 후보 간 평균이 단일 최선 대비 CV MSE를 최소 0.5% 낮춰야 앙상블을 채택하도록 정했다.</li><li>실제 운영 입력에서 동일 시간대 생산량·기상을 확보할 수 있는지 확인해야 한다.</li><li>전력 분포가 달라지는 기간에서는 새 시간 구간으로 재검증해야 한다.</li></ul></section>
<footer>2021년 제조공장 전력 예측 · 모델링 결과의 재현 및 제출용 문서 · HTML 내부 그래프는 SVG이며 외부 JS/WebGL이 필요하지 않다.</footer></main></div></body></html>'''

report = HERE / '6단계_최종_모델링_보고서.html'
report.write_text(page, encoding='utf-8')
print(report)
