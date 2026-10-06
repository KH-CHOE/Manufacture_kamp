"""학습 산출물의 버전·내용을 기록하고, 서로 다른 실행의 모델 혼합을 막는다."""
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

import config as C


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def configuration():
    return {"tree_features": C.TREE_FEATURES, "net_calendar": C.NET_CALENDAR,
            "window": C.NET["window"], "steps": C.NET["steps"],
            "tree_features_wx": C.TREE_FEATURES_WX,
            "blend_grid": [C.BLEND_GRID[0], C.BLEND_GRID[-1], len(C.BLEND_GRID)],
            "blend_select": C.BLEND_SELECT}


def write_manifest(data_path, result_path):
    result = json.loads(Path(result_path).read_text())
    models = result['모델별']
    files = {}
    for kind in ('rf', 'et', 'hgb', 'et_wx', 'gru', 'lstm'):
        if kind in models:
            path = C.MODEL_DIR / f'model_{kind}.{ "pt" if kind in ("gru", "lstm") else "joblib"}'
            files[path.name] = digest(path)
    files[Path(result_path).name] = digest(result_path)
    import modeling as M
    data = M.load(data_path)
    test = data[M.usable(data) & data['split'].eq('test').to_numpy()]
    row_hash = sha256(test.ts.astype(str).str.cat(sep='\n').encode()).hexdigest()
    manifest = {
        'schema': 2,
        'configuration': configuration(),
        'source_sha256': {name: digest(C.HERE/name) for name in ('config.py','preprocessing.py','modeling.py','net_infer.py')},
        'data_sha256': digest(data_path),
        'test_rows_sha256': row_hash,
        'files': files,
        'python': platform.python_version(),
        'packages': {k: version(k) for k in ('numpy','pandas','scikit-learn','scipy','torch','joblib')},
        'window_rule': 'original observation axis; finite continuous 96 observations',
        'normalization': 'first 85% training rows only; validation excluded',
        'ensemble': 'et' in models and 'gru' in models and 'ensemble' in models,
        # 전진검증으로 고른 트리 비중. 서빙은 이 값으로 섞는다
        'blend_weight': result.get('설정', {}).get('blend_weight'),
    }
    (C.MODEL_DIR/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


def validate(model_path, net_path=None):
    path = Path(model_path).parent/'manifest.json'
    if not path.exists():
        raise ValueError('모델 검증 기록이 없습니다. 현재 코드로 전체 모델을 다시 학습해주세요')
    m = json.loads(path.read_text())
    if m.get('configuration') != configuration():
        raise ValueError('현재 입력·결합 설정이 저장 모델의 설정과 다릅니다')
    trained = m['packages']['scikit-learn']
    if version('scikit-learn') != trained:
        raise ValueError(f'scikit-learn 버전 불일치: 저장 {trained}, 실행 {version("scikit-learn")}. requirements.txt를 설치해주세요')
    for artifact in [model_path] + ([net_path] if net_path is not None else []):
        artifact = Path(artifact)
        if m['files'].get(artifact.name) != digest(artifact):
            raise ValueError(f'{artifact.name}이 평가한 모델과 다릅니다. 모델과 평가 결과를 함께 다시 생성해주세요')
    if net_path is not None and version('torch') != m['packages']['torch']:
        raise ValueError('PyTorch 버전이 저장 모델 학습 환경과 다릅니다. requirements.txt를 설치해주세요')
    if net_path is not None and Path(model_path).name != 'model_et.joblib':
        raise ValueError('평가된 결합 모델은 ExtraTrees + GRU입니다')
    if net_path is not None and not m['ensemble']:
        raise ValueError('이 학습 실행은 결합 모델을 평가하지 않았습니다')
    if net_path is not None and m.get('blend_weight') is None:
        raise ValueError('결합 비중이 기록되지 않았습니다. 전체 모델을 다시 학습해주세요')
    return m
