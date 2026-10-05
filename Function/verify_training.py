"""작은 시험 설정으로 학습·중단 후 재사용·저장 모델 추론의 연결을 검사한다.

시험용 임시 폴더에서만 실행한다. 실제 성능 비교와 저장 모델은 바꾸지 않는다.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import config as C


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fn = root/'Function'; fn.mkdir()
        for name in ('modeling.py','config.py','preprocessing.py','net_infer.py','artifacts.py','evaluate_duplicates.py'):
            shutil.copy2(C.HERE/name, fn/name)
        with (fn/'config.py').open('a') as f:
            f.write('\nNET = {**NET, "hidden": 4, "epochs": 1, "patience": 1}\n'
                    'SEEDS = [42, 7]\nFORWARD_FOLDS = 1\n'
                    'TREE = {**TREE, "n_estimators": 2, "n_jobs": 1}\n'
                    'RF = {**TREE, "bootstrap": True}\nBOOST = {**BOOST, "max_iter": 2}\n')
        cmd = [sys.executable, '-B', str(fn/'modeling.py'), '--data', str(C.OUT_DEFAULT)]
        for attempt in range(2):
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode:
                raise AssertionError(p.stdout+'\n'+p.stderr)
            if attempt == 1:
                assert p.stdout.count('완료 학습 재사용') == 8, p.stdout
        report = json.loads((root/'Model/results.json').read_text())
        manifest = json.loads((root/'Model/manifest.json').read_text())
        import modeling as M
        expected = set(M.BASE_MODELS) | set(M.TREE_MODELS) | set(M.NET_MODELS) | {'ensemble'}
        assert manifest['ensemble'] and set(report['모델별']) == expected, sorted(report['모델별'])
        # 결합 비율이 탐색 범위 안에서 골라져 결과·검증 기록에 같은 값으로 적혔는가
        w = report['결합비율탐색']['선정_트리비중']
        assert w in C.BLEND_GRID and manifest['blend_weight'] == w == report['설정']['blend_weight']
        out = root/'inference.npz'
        subprocess.run([sys.executable,'-B',str(fn/'net_infer.py'),'--data',str(C.OUT_DEFAULT),
                        '--model',str(root/'Model/model_gru.pt'),'--out',str(out)], check=True)
        inferred, trained = np.load(out), np.load(fn/'_pred_nets.npz')
        np.testing.assert_array_equal(inferred['where'], trained['row_index'][trained['gru|idx|test']])
        np.testing.assert_allclose(inferred['pred'], trained['gru|test'], atol=1e-4, rtol=1e-5)
        assert (root/'Model/duplicate_evaluation.json').exists()
        print('통과: 전체 학습 연결, 8개 완료 학습 재사용, 시드 평균 저장 추론, 결합 비율 기록, 결합 및 중복 평가')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
