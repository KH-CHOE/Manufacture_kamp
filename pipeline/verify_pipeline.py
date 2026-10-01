"""파이프라인이 미래를 보지 않는지, 규칙이 실제로 작동하는지 검사한다.

왜 필요한가 — 새로 쓴 코드가 가장 위험하다. 이 프로젝트에서 실제로 잡힌 누수가 둘 있었다.
`공장인원` 으로 타깃을 역산할 수 있었던 것과, 일교차를 그날 전체에서 계산한 것이다.
그래서 "미래를 썼으면 반드시 실패하는" 검사를 둔다.

검사 목록 (돌리기 전에 못박는다)
-------------------------------
  A 타깃 정합      y[i] 가 **같은 구간 안에서** kW[i+1] 과 같은가
  B 인과성         절단 시점 이후의 실측을 오염시켜도 그 이전 행의 입력이 불변인가
  C 헛돌지 않는가  오염 이후 행은 **반드시 변해야** 한다
  D 경계행         절단−1 행도 불변인가 (한 칸 어긋난 누수가 드러나는 자리)
  E 대치 인과성    결측 대치가 미래 관측을 쓰지 않았는가
  F 구간 규칙      공백을 넘는 shift·rolling 이 없는가
  G 규칙 작동      2021 자료에서 잡아야 할 것을 다 잡았는가
  H 다른 해        문제의 위치·종류를 바꾼 자료에서도 돌아가는가
  I 공통 행        모든 모델이 같은 행을 쓰는가

하나라도 실패하면 종료 코드 1. 통과는 "검사한 범위에서 미래 참조를 찾지 못했다" 는
뜻이고, 모든 행의 무누수를 증명하는 것은 아니다.

실행: python verify_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import preprocessing as P

HERE = Path(__file__).resolve().parent
# 절단점을 비율만으로 잡으면 **기간 한정 누수를 놓친다.**
# 처음 판은 (0.25, 0.4, 0.55, 0.7, 0.85) 였는데, 그 다섯 점이 전부 7월을 비껴가서
# "7월에만 미래를 본다" 는 결함을 통과시켰다. 비율 + 월 경계 + 월 중간 + 임의 지점을 쓴다.
CUT_FRACTIONS = (0.25, 0.4, 0.55, 0.7, 0.85)


def cut_points(raw) -> list[int]:
    """시간 행 기준 절단 지점. 월 경계와 월 중간을 반드시 포함한다."""
    n = len(raw)
    dt = pd.to_datetime(raw[C.DATE_COL].astype(str), format="%Y%m%d", errors="coerce")
    cuts = {int(n * f) for f in CUT_FRACTIONS}
    for (y, m), idx in raw.groupby([dt.dt.year, dt.dt.month]).groups.items():
        pos = np.sort(np.asarray(idx))
        cuts.add(int(pos[0]))                      # 달 시작
        cuts.add(int(pos[len(pos) // 2]))          # 달 중간 — 기간 한정 누수가 드러난다
    rng = np.random.default_rng(42)
    cuts |= {int(v) for v in rng.integers(C.PER_WEEK // C.PER_HOUR + 10, n - 100, size=6)}
    return sorted(c for c in cuts if C.PER_WEEK // C.PER_HOUR + 10 <= c < n - 100)
MEASURED = ["15분", "30분", "45분", "60분", "생산량", "기온", "풍속", "습도", "강수량"]

fails: list[str] = []


def ok(name: str, passed: bool, note: str = "") -> None:
    print(f"  {'✓' if passed else '✗'} {name}" + (f"  {note}" if note else ""))
    if not passed:
        fails.append(name)


def build(raw: pd.DataFrame, cal: dict | None = None) -> pd.DataFrame:
    """preprocessing 의 단계를 그대로 거쳐 결과를 만든다(파일로 쓰지 않는다)."""
    log = P.Log()
    log.add = lambda *a, **k: None          # 조용히
    d = P.check_schema(raw, log)
    d = P.fix_hour(d, log)
    d = P.build_hourly_axis(d, log)
    dropped = P.detect_derived(d, log)
    ctx = [c for c in C.CONTEXT_COLS if c in d.columns and c not in dropped]
    d = P.impute_past_median(d, ctx, log)
    long = P.to_15min(d, ctx, log)
    long = P.add_segments(long, log)
    long = P.add_features(long, cal, log)
    return long


def main() -> int:
    import contextlib, io
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=C.RAW_DEFAULT)
    ap.add_argument("--calendar", type=Path, default=C.CALENDAR_DEFAULT)
    a = ap.parse_args()
    src = a.raw
    if not src.exists():
        print(f"✗ 원자료가 없다: {src}\n  --raw 로 경로를 주거나 data/ 에 넣어라")
        return 1
    raw = pd.read_csv(src, encoding="utf-8-sig")
    cal = json.loads(a.calendar.read_text("utf-8")) if a.calendar.exists() else None

    print(f"\n원자료 {src.name}  {len(raw):,}행")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        base = build(raw, cal)
    print(f"전처리 결과 {len(base):,}행 × {len(base.columns)}열\n")

    # ── A 타깃 정합 ──
    print("A 타깃 정합")
    g = base.groupby("segment", sort=False)["kW"]
    want = g.shift(-1)
    bad = int((base["y"].notna() & (base["y"] != want)).sum())
    ok("y[i] == kW[i+1] (같은 구간 안)", bad == 0, f"어긋난 행 {bad}")
    last = base.groupby("segment").tail(1)
    ok("구간 마지막 행은 타깃이 없다", bool(last["y"].isna().all()),
       f"{int(last['y'].isna().sum())}/{len(last)}")

    # ── B·C·D 인과성 ──
    print("\nB·C·D 인과성 — 미래 실측을 오염시켜 입력 불변을 확인한다")
    feats = [c for c in (C.TREE_FEATURES + C.NET_CALENDAR + ["kW"]) if c in base.columns]
    cal_cols = {"시간", "15분위치", "dow", "is_weekend", "is_day_shift",
                "tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_off"}
    sensitive = [c for c in feats if c not in cal_cols]
    n_hour = len(raw)
    CUTS = cut_points(raw)
    print(f"  절단점 {len(CUTS)}개 (비율 + 월 경계 + 월 중간 + 임의)")
    for hcut in CUTS:
        mut = raw.copy()
        for c in MEASURED:
            if c in mut.columns:
                mut.loc[mut.index >= hcut, c] = 999
        with contextlib.redirect_stdout(io.StringIO()):
            other = build(mut, cal)
        if len(other) != len(base):
            ok(f"절단 {hcut}: 행수 보존", False, f"{len(base)} vs {len(other)}")
            continue
        qcut = hcut * C.PER_HOUR                       # 15분 행 기준 절단 위치
        pre = np.arange(C.PER_WEEK, qcut)              # 워밍업 뒤 ~ 절단 직전
        a_ = base.loc[pre, feats].to_numpy(float)
        b_ = other.loc[pre, feats].to_numpy(float)
        same = bool(np.allclose(np.nan_to_num(a_, nan=-1), np.nan_to_num(b_, nan=-1)))
        edge = qcut - 1
        ea = base.loc[[edge], feats].to_numpy(float)
        eb = other.loc[[edge], feats].to_numpy(float)
        esame = bool(np.allclose(np.nan_to_num(ea, nan=-1), np.nan_to_num(eb, nan=-1)))
        post = np.arange(qcut + 50, min(qcut + 400, len(base)))
        pa = base.loc[post, sensitive].to_numpy(float)
        pb = other.loc[post, sensitive].to_numpy(float)
        diff = len(post) > 0 and not bool(
            np.allclose(np.nan_to_num(pa, nan=-1), np.nan_to_num(pb, nan=-1)))
        ok(f"절단 {hcut:>5} · 예측시점 입력 불변", same, f"검사 {len(pre):,}행")
        ok(f"절단 {hcut:>5} · 경계행({edge}) 불변", esame)
        ok(f"절단 {hcut:>5} · 오염 이후는 변한다", diff, "헛돌지 않는다")

    # ── E 대치 인과성 ──
    print("\nE 결측 대치가 미래를 쓰지 않는가")
    log = P.Log(); log.add = lambda *a, **k: None
    with contextlib.redirect_stdout(io.StringIO()):
        h1 = P.build_hourly_axis(P.fix_hour(P.check_schema(raw, log), log), log)
        h1 = P.impute_past_median(h1, ["풍속", "강수량"], log)
        mut = raw.copy()
        mut.loc[mut.index >= len(mut) // 2, ["풍속", "강수량"]] = 999
        h2 = P.build_hourly_axis(P.fix_hour(P.check_schema(mut, log), log), log)
        h2 = P.impute_past_median(h2, ["풍속", "강수량"], log)
    half = len(h1) // 2
    same = bool(np.allclose(h1.loc[:half - 1, ["풍속", "강수량"]].to_numpy(float),
                            h2.loc[:half - 1, ["풍속", "강수량"]].to_numpy(float)))
    ok("뒤쪽 관측을 바꿔도 앞쪽 대치값이 불변", same)

    # ── F 구간 규칙 ──
    print("\nF 구간 규칙 — 공백을 넘지 않는가")
    step = pd.Timedelta(minutes=C.STEP_MIN)
    brk = base.index[base["ts"].diff() > step]
    if len(brk) == 0:
        ok("이 자료에는 시각 공백이 없다 (규칙은 H 에서 시험한다)", True)
    else:
        bad = 0
        for i in brk:
            row = base.loc[i]
            if pd.notna(row.get("kw_lag2")) or pd.notna(row.get("kw_roll4")):
                bad += 1
        ok("공백 직후 행에 이전 구간 값이 새지 않는다", bad == 0, f"위반 {bad}/{len(brk)}")

    # ── G 규칙이 2021 에서 잡아야 할 것 ──
    print("\nG 규칙 작동 (2021)")
    man = json.loads((HERE / "processed_manifest.json").read_text("utf-8"))
    dropped = set(man["제외한열"])
    for c in ("평균", "공장인원", "전기요금(계절)", "인건비", "day", "d", "m"):
        ok(f"{c} 를 제외했다", c in dropped)
    ok("생산량 을 남겼다 (보호 입력)", "생산량" not in dropped)
    steps = {s["단계"]: s for s in man["단계"]}
    rep = steps.get("시 열 복구", {})
    ok("시 열이 깨진 2일을 복구했다", rep.get("복구일") == 2, f"복구 {rep.get('복구일')}일")
    imp = steps.get("결측 대치", {})
    ok("결측 4칸을 대치했다",
       sum(x["결측"] for x in imp.get("상세", [])) == 4)

    # ── H 다른 해 ──
    print("\nH 다른 해 자료 (문제의 위치·종류를 바꿔 심은 것)")
    tf = HERE / "test_year_2022.csv"
    if not tf.exists():
        r = subprocess.run([sys.executable, "-B", "make_test_year.py",
                            "--raw", str(src), "--out", str(tf)],
                           cwd=HERE, capture_output=True, text=True)
        ok("시험 자료 생성", r.returncode == 0, r.stderr.strip()[:80])
    r = subprocess.run([sys.executable, "-B", "preprocessing.py", "--raw", str(tf),
                        "--out", str(HERE / "processed_2022.csv")],
                       cwd=HERE, capture_output=True, text=True)
    ok("달력 자료 없이도 전처리가 끝난다", r.returncode == 0, r.stderr.strip()[:120])
    m2p = HERE / "processed_2022_manifest.json"
    if m2p.exists():
        m2 = json.loads(m2p.read_text("utf-8"))
        d2 = set(m2["제외한열"])
        ok("지어낸 파생 열(설비효율)을 잡았다", "설비효율" in d2)
        ok("지어낸 달력 열(요율구간)을 잡았다", "요율구간" in d2)
        ok("생산량 을 남겼다", "생산량" not in d2)
        s2 = {s["단계"]: s for s in m2["단계"]}
        ok("시 열이 정상인 불완전한 하루를 버리지 않았다",
           "행이 빠진 날 — 시 열은 정상이라 그대로 쓴다" in s2)
        ok("시각 공백을 구간으로 나눴다",
           s2.get("구간 분할", {}).get("구간수", 1) > 1,
           f"구간 {s2.get('구간 분할', {}).get('구간수')}개")

    # ── I 공통 행 ──
    print("\nI 모든 모델이 같은 행을 쓰는가")
    import modeling as M
    d = M.load(HERE / "processed.csv")
    mask = M.usable(d)
    rows = d[mask]
    need = [c for c in (C.TREE_FEATURES + C.NET_CALENDAR + ["kW", "y"]) if c in d.columns]
    ok("공통 행에 결측이 없다", bool(rows[need].notna().all().all()))
    ok("공통 행 전부에서 순환신경망 창이 유효하다",
       bool(M.window_ok(d, C.NET["window"])[mask].all()))
    folds = M.folds_from_data(rows, C.FORWARD_FOLDS, C.FORWARD_FOLD_DAYS)
    ok("전진검증 구간이 자료에서 생성된다", len(folds) == C.FORWARD_FOLDS,
       " · ".join(f"{lo}" for lo, _ in folds))
    in_train = all((rows.loc[(rows["date_key"] >= lo) & (rows["date_key"] < hi),
                             "split"] == "train").all() for lo, hi in folds)
    ok("전진검증 구간이 전부 학습 구간 안에 있다", in_train)

    # ── J 검사가 실제로 누수를 잡는가 ──
    # 통과만 보여 주는 검사는 쓸모가 없다. 결함을 심어 **반드시 실패하는지** 확인한다
    print("\nJ 검사가 결함을 잡는가 (일부러 누수를 심는다)")
    orig_add = P.add_features

    def leaky_shift(long, cal, log):
        d = orig_add(long, cal, log)
        d["kw_lag1"] = d.groupby("segment", sort=False)["kW"].shift(-1)   # 다음 값을 본다
        return d

    def leaky_rolling(long, cal, log):
        d = orig_add(long, cal, log)
        g = d.groupby("segment", sort=False)["kW"]
        d["kw_roll4"] = g.transform(                                      # 창이 미래로 한 칸
            lambda s: s.shift(-1).rolling(C.PER_HOUR, min_periods=C.PER_HOUR).mean())
        return d

    def leaky_july(long, cal, log):
        d = orig_add(long, cal, log)
        m = pd.DatetimeIndex(d["ts"]).month == 7                          # 7월에만
        nxt = d.groupby("segment", sort=False)["kW"].shift(-1)
        d.loc[m, "kw_lag1"] = nxt[m]
        return d

    def leaky_whole_day(long, cal, log):
        d = orig_add(long, cal, log)
        day = pd.DatetimeIndex(d["ts"]).normalize()
        d["kw_roll16"] = d.groupby(day)["kW"].transform("max")            # 그날 전체를 본다
        return d

    for name, fn in (("직전값이 다음 값을 본다", leaky_shift),
                     ("이동평균 창이 미래로 한 칸", leaky_rolling),
                     ("7월에만 미래를 본다", leaky_july),
                     ("그날 전체 최대를 쓴다", leaky_whole_day)):
        P.add_features = fn
        caught = False
        try:
            for hcut in CUTS:
                mut = raw.copy()
                for c in MEASURED:
                    if c in mut.columns:
                        mut.loc[mut.index >= hcut, c] = 999
                with contextlib.redirect_stdout(io.StringIO()):
                    a2 = build(raw, cal)
                    b2 = build(mut, cal)
                qcut = hcut * C.PER_HOUR
                pre = np.arange(C.PER_WEEK, qcut)
                if not np.allclose(
                        np.nan_to_num(a2.loc[pre, feats].to_numpy(float), nan=-1),
                        np.nan_to_num(b2.loc[pre, feats].to_numpy(float), nan=-1)):
                    caught = True
                    break
        finally:
            P.add_features = orig_add
        ok(f"심은 누수를 잡는다 — {name}", caught)

    print(f"\n{'─' * 58}")
    if fails:
        print(f"실패 {len(fails)}건")
        for f in fails:
            print(f"  · {f}")
        return 1
    print("전부 통과 — 검사한 범위에서 미래 참조를 찾지 못했다")
    print("(표본 절단 검사는 모든 행의 무누수를 증명하지 않는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
