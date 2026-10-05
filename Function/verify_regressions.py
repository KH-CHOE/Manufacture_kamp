"""최종 검토에서 발견한 오류의 재발 검사. 실행: python Function/verify_regressions.py"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import modeling as M
import preprocessing as P
from optimization import StaffingOptimizer


class Regressions(unittest.TestCase):
    def test_saved_bundle_rejects_changes(self):
        from artifacts import validate, digest, configuration
        from importlib.metadata import version
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root/'model_et.joblib'
            model.write_bytes(b'test-only-content')
            manifest = {'configuration': configuration(), 'packages': {'scikit-learn': version('scikit-learn')},
                        'files': {model.name: digest(model)}, 'ensemble': False}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            validate(model)
            model.write_bytes(b'different-content')
            with self.assertRaisesRegex(ValueError, '평가한 모델과 다릅니다'):
                validate(model)
            manifest['packages']['scikit-learn'] = '0.0.0'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, '버전 불일치'):
                validate(model)

    def test_validation_excluded_from_statistics(self):
        X = np.arange(100*24*4, dtype=np.float32).reshape(100,24,4)
        Y = np.arange(100, dtype=np.float32)
        indices = np.arange(80)
        before = M.training_stats(X,Y,indices)
        X[68:] = 1e8; Y[68:] = -1e8
        self.assertEqual(before, M.training_stats(X,Y,indices))

    def test_windows_use_original_axis(self):
        d = M.load(C.OUT_DEFAULT)
        # 중간 관측이 빠지면 공백 뒤 95칸을 모두 제외해야 한다.
        d = d.drop(index=10000).reset_index(drop=True)
        mask = M.usable(d)
        x = M.net_windows(d, mask, 96, 24).reshape(-1, 96)
        for k, endpoint in enumerate(np.flatnonzero(mask)):
            original = d.iloc[endpoint-95:endpoint+1]
            self.assertTrue(original.ts.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all())
            np.testing.assert_array_equal(x[k], original.kW.to_numpy(np.float32))
        self.assertEqual(len(x), mask.sum())
        self.assertFalse(M.window_ok(d.iloc[:20], 96).any())

    def test_broken_hour_day_is_dropped_not_repaired(self):
        # 시 값 하나가 깨져도 그 날은 다시 매기지 않고 통째로 삭제한다 (행이 24개여도)
        raw = pd.read_csv(C.RAW_DEFAULT).iloc[:72].copy()
        raw[C.HOUR_COL] = raw[C.HOUR_COL].astype(float)
        raw.loc[24, C.HOUR_COL] = .5
        broken_day = raw.loc[24, C.DATE_COL]
        log = P.Log(); log.add = lambda *a, **k: None
        d = P.fix_hour(P.check_schema(raw, log), log)
        self.assertEqual(len(d), 48)
        self.assertFalse((d['_날짜'] == P.parse_dates(pd.Series([broken_day])).iloc[0]).any())
        self.assertTrue(d['_시'].between(0, 23).all())

    def test_date_formats_and_bad_dates(self):
        got = P.parse_dates(pd.Series([20220101, 20220102.0, '2022-01-03', 'bad']))
        self.assertEqual(got.iloc[:3].dt.day.tolist(), [1,2,3])
        self.assertTrue(pd.isna(got.iloc[3]))

    def test_invalid_split_rejected(self):
        d = pd.DataFrame({'ts': pd.date_range('2022-01-01', periods=10, freq='D')})
        with self.assertRaises(ValueError):
            P.add_split(d, 20210725, .3, P.Log())
        with self.assertRaises(ValueError):
            P.add_split(d, None, 1, P.Log())

    def test_calendar_selection_other_year(self):
        raw = pd.read_csv(C.RAW_DEFAULT)
        raw[C.DATE_COL] = raw[C.DATE_COL] + 10000
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td)/'2022.csv', Path(td)/'processed.csv'
            raw.to_csv(src, index=False)
            cmd = [sys.executable, str(C.HERE/'preprocessing.py'), '--raw', str(src), '--out', str(out)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            d = pd.read_csv(out)
            self.assertEqual(set(d.split), {'train','test'})
            manifest = json.loads(out.with_name('processed_manifest.json').read_text())
            self.assertIsNone(manifest['달력자료'])
            bad = subprocess.run(cmd + ['--calendar', str(C.CALENDAR_DEFAULT)], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)

    def test_future_production_not_used(self):
        opt = StaffingOptimizer()
        args = ('2021-08-10 12:45', '2021-08-10 13:00', 100)
        with self.assertRaises(ValueError):
            opt.hourly_interval(*args)
        before = opt.hourly_interval(*args, planned_production=1199)
        opt.raw.loc[opt.raw.timestamp >= pd.Timestamp('2021-08-10'), ['생산량','공장인원']] = 999999
        after = opt.hourly_interval(*args, planned_production=1199)
        self.assertEqual(before, after)
        self.assertEqual(before['production'], 1199)
        self.assertIn('실제 인원이 아님', before['staffBasis'])

    def test_scenario_preserves_ensemble(self):
        import serving as S
        previous = S._state.copy()
        try:
            frame = pd.DataFrame({'prediction': [10.276], '생산량': [100]})
            S._state.update(groups={'2021-09-08': frame}, frame=frame,
                            bundle={'features': C.TREE_FEATURES})
            r = S.scenario('2021-09-08', 0, 100, 2, 8720, 100, 1, 2)
            self.assertEqual(r['prediction'], r['baselinePrediction'])
            self.assertEqual(r['baseline'], r['scenario'])
        finally:
            S._state.clear(); S._state.update(previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)
