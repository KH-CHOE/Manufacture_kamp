"""원자료 → 모든 모델이 공유하는 하나의 전처리 결과.

설계 원칙 — **문제가 있는 행을 손으로 지정하지 않는다.**
`2021-07-13` 처럼 날짜를 코드에 박으면 2022년 자료에서는 아무것도 못 한다.
그래서 모든 판단을 "자료를 보고 정하는 규칙" 으로 쓴다. 규칙이 무엇을 어떻게 판정했는지는
`manifest` JSON 에 전부 남긴다 — 다른 해 자료를 넣었을 때 무슨 일이 있었는지 그 파일로 읽는다.

이 자료(2021)에서 규칙이 실제로 잡아낸 것
------------------------------------------
  시 열이 깨진 날 2일      `시간` 에 70·89·…·188 이 들어 있다(0~23 범위 위반).
                          그 날의 값이 오염됐는지 알 수 없으므로 시를 다시 매기지 않고 **삭제**한다.
                          공백 주변의 시차·정답·신경망 창은 구간 분할로 함께 빠진다
  파생/누수 열 2개         `평균` = 네 전력열 평균을 정수로 반올림(최대차 0.500)
                          `공장인원` = 생산량 ÷ Σ전력 — 6,151행 전부 정확히 일치.
                          **타깃을 역산할 수 있으므로 누수다.** 결측 17행은 Σ전력 = 0 인 행과 같다
  달력 재표기 열 2개       `전기요금(계절)` = m 의 결정함수(3값) · `인건비` = 시간 의 결정함수(2값)
  달력 중복 열 3개         `day`·`d`·`m` 이 `날짜` 에서 100% 재구성된다
  결측 4칸                 풍속 3 · 강수량 1 → **엄격히 앞선 관측만** 보는 누적 중앙값으로 대치

출력
----
  한 행 = 한 15분 구간. 모든 모델이 **이 파일 하나**를 읽는다.
  트리는 열을 골라 쓰고 순환신경망은 같은 파일에서 창을 만든다 — 그래서 입력이 같다.

실행
----
  python preprocessing.py --raw <원자료.csv> --out processed.csv
  python preprocessing.py --raw <원자료.csv> --out processed.csv --calendar calendar_2021.json
  python preprocessing.py --raw <원자료.csv> --out processed.csv --split-date 20210725
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

HERE = Path(__file__).resolve().parent


# ══ 기록 ════════════════════════════════════════════════════════════
class Log:
    """규칙이 무엇을 판정했는지 모아 둔다. 조용히 넘어가는 처리를 만들지 않기 위함."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, step: str, **kw) -> None:
        self.steps.append({"단계": step, **kw})
        detail = " · ".join(f"{k} {v}" for k, v in kw.items()
                            if not isinstance(v, (list, dict)))
        print(f"  [{step}] {detail}" if detail else f"  [{step}]")


def parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, format="mixed", errors="coerce")


# ══ 02. 스키마 ══════════════════════════════════════════════════════
def check_schema(raw: pd.DataFrame, log: Log) -> pd.DataFrame:
    need = [C.DATE_COL, C.HOUR_COL, *C.POWER_COLS]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"✗ 필수 열이 없다: {missing}\n  있는 열: {list(raw.columns)}")

    d = raw.copy()
    d["_원본행"] = np.arange(len(d)) + 2          # 헤더 다음이 2행
    ctx = [c for c in C.CONTEXT_COLS if c in d.columns]
    for c in [C.HOUR_COL, *C.POWER_COLS, *ctx]:
        # **문자열 오타를 조용히 결측으로 바꾸지 않는다.** 그러면 원인을 모르게 된다
        d[c] = pd.to_numeric(d[c], errors="raise")
        if np.isinf(d[c].to_numpy(float)).any():
            raise SystemExit(f"✗ {c}: 무한대 값이 있다")
    if (d[C.POWER_COLS] < 0).any().any():
        raise SystemExit("✗ 음수 전력이 있다. 원자료의 부호 정의를 확인해라")

    d["_날짜"] = parse_dates(d[C.DATE_COL])
    bad = d["_날짜"].isna()
    if bad.any():
        log.add("날짜 해석 실패 행 제외", 행수=int(bad.sum()),
                원본행=d.loc[bad, "_원본행"].tolist()[:20])
        d = d.loc[~bad].copy()

    if d.empty:
        raise ValueError("해석 가능한 날짜가 없습니다. 날짜 형식을 확인해주세요")
    log.add("스키마 확인", 행=len(d), 열=len(raw.columns),
            맥락열=len(ctx), 없는맥락열=[c for c in C.CONTEXT_COLS if c not in raw.columns])
    return d


# ══ 03. 시(hour) 열 검증 — 깨진 날은 삭제한다 ════════════════════════
def fix_hour(d: pd.DataFrame, log: Log) -> pd.DataFrame:
    """`시간` 이 깨진 날을 **통째로 삭제**한다. 값을 다시 매기지 않는다.

    왜 복구하지 않나 — 시 값이 깨진 날은 **그 날의 다른 값도 오염됐는지 알 수 없다.**
    행 개수가 24개로 맞는다고 행 순서로 시를 다시 매기면, 근거 없이 값을 자의적으로
    정하는 것이 된다. 그래서 그 날을 지우고 시각축에 공백으로 남긴다.

    **주변도 함께 빠진다** — 삭제한 날은 시각축의 공백이 되고, 구간 분할이 공백마다 구간을
    나눈다. 그래서 공백을 넘는 시차·이동평균은 만들어지지 않고(공백 뒤 행의 입력이 결측),
    공백 바로 앞 행은 정답(다음 15분)이 없어 빠지며, 공백을 넘는 순환신경망 입력 창도
    `modeling.usable()` 에서 빠진다. 날짜를 코드에 박지 않으므로 다른 자료에도 그대로 돈다.

    두 가지를 **따로** 본다 — 섞으면 멀쩡한 자료를 버린다
      (A) 시 값이 쓸 수 있는가   0~23 정수이고 그 날 안에서 중복이 없는가
      (B) 그 날이 완전한가       행이 24개인가

    판정
      A 통과   → 시 열을 그대로 쓴다. **B 와 무관하다.**
                 행이 모자라면 시각축에 공백으로 남고 구간 분할이 처리한다
      A 실패   → 그 날을 삭제한다. 행이 24개여도 마찬가지다

    왜 (A)와 (B)를 나누나 — 시험 자료에서 `08-03` 은 시 열이 정상인데 행만 21개였다.
    둘을 묶어 "24행 순열" 로 재던 판은 그 하루를 통째로 버렸다. 멀쩡한 21행이었다.
    """
    hour_ok, complete = {}, {}
    for key, g in d.groupby("_날짜"):
        v = g[C.HOUR_COL].to_numpy()
        finite = np.isfinite(v)
        hour_ok[key] = bool(
            finite.all()
            and np.all(np.equal(np.mod(v[finite], 1), 0))
            and v.min() >= 0 and v.max() <= C.HOURS_PER_DAY - 1
            and len(np.unique(v)) == len(v))
        complete[key] = len(v) == C.HOURS_PER_DAY

    good = [k for k, ok in hour_ok.items() if ok]
    bad = [k for k, ok in hour_ok.items() if not ok]

    log.add("시 열 검증", 시값정상일=len(good), 시값깨진일=len(bad),
            불완전일=int(sum(1 for k, c in complete.items() if not c)))

    hours = pd.to_numeric(d[C.HOUR_COL], errors="coerce")
    d["_시"] = hours.where(np.isfinite(hours) & hours.between(0, 23) & hours.mod(1).eq(0)).astype("Int64")
    incomplete_kept = [k for k in good if not complete[k]]
    if incomplete_kept:
        log.add("행이 빠진 날 — 시 열은 정상이라 그대로 쓴다",
                날짜수=len(incomplete_kept),
                상세=[{"날짜": k.strftime("%Y-%m-%d"), "행수": int((d["_날짜"] == k).sum())}
                     for k in incomplete_kept][:20])

    if bad:
        detail = [{"날짜": k.strftime("%Y-%m-%d"), "행수": int((d["_날짜"] == k).sum()),
                   "완전한가": complete[k],
                   "시간열_값": d.loc[d["_날짜"] == k, C.HOUR_COL].tolist()[:8]}
                  for k in bad]
        before = len(d)
        d = d[~d["_날짜"].isin(bad)].copy()
        log.add("시 열이 깨진 날 삭제", 제외일=len(bad), 제외행=before - len(d),
                이유="시 값이 0~23 정수가 아니거나 날짜 안에서 중복된다. 그 날의 다른 값도 "
                     "오염됐는지 알 수 없어 시를 다시 매기지 않고 삭제한다",
                주변처리="삭제한 날은 시각축 공백이 되고, 공백을 넘는 시차·이동평균·정답·"
                        "순환신경망 창은 구간 분할과 공통행 규칙으로 함께 빠진다",
                상세=detail)
    else:
        log.add("시 열이 깨진 날 삭제", 제외일=0, 제외행=0)

    if d["_시"].isna().any():
        raise SystemExit("✗ 시(hour)를 정하지 못한 행이 남았다")
    d["_시"] = d["_시"].astype(int)
    return d


# ══ 04. 시각축 ══════════════════════════════════════════════════════
def build_hourly_axis(d: pd.DataFrame, log: Log) -> pd.DataFrame:
    d["_시각"] = d["_날짜"] + pd.to_timedelta(d["_시"], unit="h")
    dup = d["_시각"].duplicated(keep=False)
    if dup.any():
        raise SystemExit(
            f"✗ 같은 날짜·시간에 중복 행 {int(dup.sum())}개가 있다.\n"
            f"  공장이 여럿 섞여 있는지, 중복 적재인지 확인해라.\n"
            f"  예: {d.loc[dup, '_시각'].head(5).astype(str).tolist()}")
    d = d.sort_values("_시각").reset_index(drop=True)
    gap = d["_시각"].diff()
    holes = gap[gap > pd.Timedelta(hours=1)]
    log.add("시각축", 시작=str(d["_시각"].iloc[0]), 끝=str(d["_시각"].iloc[-1]),
            행=len(d), 공백구간=len(holes),
            공백=[{"앞": str(d["_시각"].iloc[i - 1]), "뒤": str(d["_시각"].iloc[i]),
                  "빈시간": int(v / pd.Timedelta(hours=1)) - 1}
                 for i, v in holes.items()][:20])
    return d


# ══ 05. 파생·누수 열 탐지 ═══════════════════════════════════════════
def detect_derived(d: pd.DataFrame, log: Log) -> list[str]:
    """타깃(전력)에서 파생된 열과 달력의 재표기 열을 **자료를 보고** 찾는다.

    왜 규칙으로 하는가 — 열 이름을 박아 두면 2022년 자료에 다른 이름의 파생 열이
    들어왔을 때 그대로 입력에 섞인다. `공장인원` 은 생산량 ÷ Σ전력 이라 **타깃을
    역산할 수 있었다.** 그런 열을 이름이 아니라 성질로 잡아야 한다.

    항등식이 두 열을 묶을 때 — 중요
    --------------------------------
    `공장인원 = 생산량 ÷ Σ전력` 은 `생산량 = 공장인원 × Σ전력` 과 같은 식이다.
    양쪽을 다 빼면 쓸 수 있는 변수(생산량)를 잃는다. **역산을 끊는 데는 한쪽만
    빼면 된다.** 그래서 `config.PROTECTED_INPUTS` 로 "이건 실측값이다" 라고 선언한
    열은 남기고 반대쪽을 뺀다. 양쪽이 모두 보호 대상이면 자동으로 정하지 않고 멈춘다.

    계산된 열에는 지문이 있다 — `공장인원` 의 결측 17행은 Σ전력 = 0 인 행과 정확히
    같다(0으로 나눈 자리). 이것을 근거로 함께 기록한다.
    """
    P = d[C.POWER_COLS]
    s_, mu = P.sum(axis=1), P.mean(axis=1)
    # (이름, 값, 쓰인 열 집합) — 집합으로 들고 다녀야 자기참조를 정확히 걸러낸다.
    # 예전에는 이름 접미사로 걸러서 `평균` 열이 후보 "네 전력열 평균" 과 겹쳐 통째로
    # 건너뛰어졌다 — 그래서 가장 뻔한 파생 열을 놓쳤다.
    cands: list[tuple[str, pd.Series, set[str]]] = [
        ("네 전력열 합", s_, set()), ("네 전력열 평균", mu, set()),
        ("네 전력열 최대", P.max(axis=1), set()), ("네 전력열 최소", P.min(axis=1), set()),
    ]
    skip = {C.DATE_COL, C.HOUR_COL}
    others = [c for c in d.columns
              if c not in C.POWER_COLS and c not in skip
              and not c.startswith("_") and d[c].dtype.kind in "if"]
    for c in others:
        v = d[c]
        for nm, b in (("합", s_), ("평균", mu)):
            cands.append((f"{c} ÷ {nm}", v / b.replace(0, np.nan), {c}))
            cands.append((f"{nm} ÷ {c}", b / v.replace(0, np.nan), {c}))
            cands.append((f"{c} × {nm}", v * b, {c}))

    drop: list[str] = []
    found: list[dict] = []
    ask: list[dict] = []

    for c in others:
        v = d[c]
        for nm, cv, uses in cands:
            if c in uses:
                continue                                  # 자기 자신이 들어간 식
            both = v.notna() & cv.notna() & np.isfinite(cv)
            if both.sum() < max(100, 0.5 * len(d)):
                continue
            hit = float((np.abs(v[both] - cv[both]) <= C.DERIVED_ABS_TOL).mean())
            if hit < C.DERIVED_MATCH_MIN:
                continue
            worst = float(np.abs(v[both] - cv[both]).max())
            partner = next(iter(uses)) if uses else None
            # 계산된 열의 지문 — 결측이 식의 퇴화 지점(0으로 나누기)과 일치하는가
            degenerate = (s_ == 0) | (mu == 0)
            fp = bool(v.isna().any()) and bool((v.isna() == (degenerate & v.isna().any())).all())
            rec = {"열": c, "정체": nm, "일치율": round(hit, 6),
                   "최대차": round(worst, 6), "행수": int(both.sum()),
                   "짝": partner, "계산된열_지문": fp, "분류": "타깃 파생 — 누수"}
            if partner is None:
                found.append(rec); drop.append(c)         # 전력만으로 만들어진다
            else:
                c_prot = c in C.PROTECTED_INPUTS
                p_prot = partner in C.PROTECTED_INPUTS
                if c_prot and p_prot:
                    rec["분류"] = "타깃 파생 — 양쪽 모두 보호 입력. 사람이 정해야 한다"
                    ask.append(rec)
                elif c_prot and not p_prot:
                    rec["처리"] = f"{c} 는 보호 입력이라 남기고 {partner} 를 뺀다"
                    found.append(rec); drop.append(partner)
                else:
                    rec["처리"] = f"{c} 를 뺀다" + (f" ({partner} 는 보호 입력)" if p_prot else "")
                    found.append(rec); drop.append(c)
            break

    if ask:
        for r in ask:
            print(f"      ! {r['열']} = {r['정체']}  — 양쪽 모두 보호 입력이다")
        raise SystemExit(
            "✗ 타깃을 역산할 수 있는 항등식이 보호 입력 둘 사이에 있다.\n"
            "  어느 쪽을 뺄지 자동으로 정하지 않는다. config.PROTECTED_INPUTS 에서\n"
            "  파생된 쪽을 지우고 다시 돌려라.")

    # 달력의 결정함수 — **행 단위 일치율**로 잰다.
    # `.all()` 로 재면 `인건비` 처럼 48행(0.8%)만 어긋난 열을 놓친다
    keys = {"hour": d["_시"], "month": d["_날짜"].dt.month,
            "dayofweek": d["_날짜"].dt.dayofweek, "date": d["_날짜"]}
    for c in others:
        if c in drop or d[c].nunique(dropna=True) <= 1:
            continue
        for kn in C.CALENDAR_KEYS:
            mode = d.groupby(keys[kn])[c].transform(
                lambda s2: s2.mode().iloc[0] if len(s2.mode()) else np.nan)
            agree = float((d[c] == mode).mean())
            if agree >= C.CALENDAR_AGREEMENT_MIN:
                found.append({"열": c, "정체": f"{kn} 의 결정함수",
                              "일치율": round(agree, 6),
                              "고유값수": int(d[c].nunique()),
                              "분류": "달력 재표기 — 중복"})
                drop.append(c)
                break

    # 날짜에서 그대로 재구성되는 달력 열
    cal_same = {"dayofweek+1": d["_날짜"].dt.dayofweek + 1,
                "dayofweek": d["_날짜"].dt.dayofweek,
                "day of month": d["_날짜"].dt.day, "month": d["_날짜"].dt.month,
                "year": d["_날짜"].dt.year}
    for c in others:
        if c in drop:
            continue
        for nm, cv in cal_same.items():
            if (d[c] == cv).all():
                found.append({"열": c, "정체": nm, "분류": "날짜에서 재구성 — 중복"})
                drop.append(c)
                break

    out = sorted(set(drop))
    log.add("파생·누수 열 탐지", 검사한열=len(others), 제외=len(out),
            제외한열=out, 상세=found)
    for f in found:
        mark = "누수" if "누수" in f["분류"] else "중복"
        extra = f"  → {f['처리']}" if "처리" in f else ""
        print(f"      - {f['열']:14s} {mark}: {f['정체']}{extra}")
    return out


# ══ 06. 결측 대치 ═══════════════════════════════════════════════════
def impute_past_median(d: pd.DataFrame, cols: list[str], log: Log) -> pd.DataFrame:
    """**엄격히 앞선 관측만** 보는 누적 중앙값으로 채운다.

    `expanding().median().shift(1)` 이므로 자기 자신도, 미래도 보지 않는다.
    처음부터 관측이 없으면 비워 둔다 — 모델 파이프라인의 대치기가 처리한다.
    """
    logs = []
    for c in cols:
        if c not in d.columns:
            continue
        obs = d[c]
        n = int(obs.isna().sum())
        if n == 0:
            continue
        past = obs.expanding(min_periods=1).median().shift(1)
        d[c] = obs.fillna(past)
        logs.append({"열": c, "결측": n, "채운수": int(n - d[c].isna().sum()),
                     "남은결측": int(d[c].isna().sum()),
                     "방법": "엄격히 앞선 관측의 누적 중앙값",
                     "원본행": d.loc[obs.isna(), "_원본행"].tolist()[:20]})
    log.add("결측 대치", 처리열=len(logs), 상세=logs)
    for l in logs:
        print(f"      - {l['열']:8s} 결측 {l['결측']}개 → {l['채운수']}개 채움"
              f"{' · 남음 ' + str(l['남은결측']) if l['남은결측'] else ''}")
    return d


# ══ 07. 15분 전개 ═══════════════════════════════════════════════════
def to_15min(d: pd.DataFrame, keep_cols: list[str], log: Log) -> pd.DataFrame:
    """넓은 형태(한 행 = 한 시간)를 15분 해상도로 편다."""
    ids = ["_시각", "_날짜", "_시", "_원본행"] + keep_cols
    long = d.melt(id_vars=ids, value_vars=C.POWER_COLS,
                  var_name="_구간", value_name="kW")
    order = {c: i for i, c in enumerate(C.POWER_COLS)}
    long["_q"] = long["_구간"].map(order)
    long["ts"] = long["_시각"] + pd.to_timedelta(long["_q"] * C.STEP_MIN, unit="m")
    long = (long.drop(columns=["_구간", "_시각"])
            .sort_values("ts").reset_index(drop=True))
    assert len(long) == len(d) * C.PER_HOUR, "전개 행수가 원본 × 구간수 여야 함"
    assert long["ts"].is_unique and long["ts"].is_monotonic_increasing
    log.add("15분 전개", 시간행=len(d), 구간행=len(long),
            구간수=len(C.POWER_COLS), 간격=f"{C.STEP_MIN}분")
    return long


# ══ 08. 구간(segment) — 시각 공백을 넘지 않게 ═══════════════════════
def add_segments(long: pd.DataFrame, log: Log) -> pd.DataFrame:
    """시각이 끊긴 곳에서 번호를 바꾼다. 모든 shift·rolling 이 이 안에서만 움직인다.

    이것이 없으면 7월에 이틀이 빠진 자리를 지나 '15분 전' 을 가져오게 된다 —
    실제로는 2일 전 값인데 15분 전인 줄 안다.
    """
    step = pd.Timedelta(minutes=C.STEP_MIN)
    long["segment"] = long["ts"].diff().ne(step).cumsum()
    n = long["segment"].nunique()
    sizes = long.groupby("segment").size()
    log.add("구간 분할", 구간수=n, 최소행=int(sizes.min()), 최대행=int(sizes.max()),
            경계=[str(t) for t in long.loc[long["ts"].diff().gt(step), "ts"].head(10)])
    return long


# ══ 09. 파생변수 ════════════════════════════════════════════════════
def add_features(long: pd.DataFrame, cal: dict | None, log: Log) -> pd.DataFrame:
    """모든 모델이 공유하는 변수를 만든다. **구간 안에서만** 움직인다."""
    d = long
    g = d.groupby("segment", sort=False)["kW"]

    # 타깃 — 다음 15분 구간. 따라서 현재 구간 kW 는 입력으로 쓸 수 있다
    d["y"] = g.shift(-1)

    # 시차: lag1 = 예측 시점의 관측값(= 현재 구간)
    for n in (1, 2, 3, 4, 8, C.PER_DAY, C.PER_WEEK):
        d[f"kw_lag{n}"] = g.shift(n - 1)
    d["과거전력_16칸"] = g.shift(16)
    d["kw_d1"] = d["kw_lag1"] - d["kw_lag2"]          # 지금 기울기
    d["kw_d2"] = d["kw_lag2"] - d["kw_lag3"]          # 한 걸음 전 기울기
    for n, lab in ((C.PER_HOUR, "roll4"), (16, "roll16")):
        d[f"kw_{lab}"] = g.transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
    d["kw_std4"] = g.transform(lambda s: s.rolling(C.PER_HOUR, min_periods=C.PER_HOUR).std())
    for n, lab in ((8, "2시간"), (16, "4시간")):
        d[f"전력변화_{lab}"] = d["kW"] - g.shift(n)
        d[f"최근{lab}_전력평균"] = g.transform(
            lambda s, n=n: s.rolling(n, min_periods=n).mean())
        d[f"최근{lab}_전력최대"] = g.transform(
            lambda s, n=n: s.rolling(n, min_periods=n).max())

    # 달력 — 전부 ts 에서 만든다. 원본 달력 열은 쓰지 않는다
    ts = d["ts"]
    tod = ts.dt.hour * C.PER_HOUR + ts.dt.minute // C.STEP_MIN
    dw = ts.dt.dayofweek
    d["시간"] = ts.dt.hour
    d["15분위치"] = ts.dt.minute // C.STEP_MIN
    d["dow"] = dw + 1                                  # 월=1 … 일=7
    d["is_weekend"] = (dw >= 5).astype(int)
    d["is_day_shift"] = ts.dt.hour.between(9, 17).astype(int)
    d["tod_sin"] = np.sin(2 * np.pi * tod / C.PER_DAY)
    d["tod_cos"] = np.cos(2 * np.pi * tod / C.PER_DAY)
    d["dow_sin"] = np.sin(2 * np.pi * dw / 7)
    d["dow_cos"] = np.cos(2 * np.pi * dw / 7)

    # 공휴일 — **자료로 받는다.** 안 주면 주말만으로 만들고 그 사실을 기록한다
    hol = pd.to_datetime(cal.get("holidays", [])).normalize() if cal else pd.DatetimeIndex([])
    d["is_off"] = ((dw >= 5) | ts.dt.normalize().isin(hol)).astype(int)

    # 장기 휴무를 **인과적으로** 대신한다 — 미래를 보지 않는다.
    # 어제까지의 일별 전력 합만 보고 '마지막으로 돌던 날이 며칠 전인가' 를 센다
    daily = d.groupby(ts.dt.normalize())["kW"].sum()
    past_med = daily.expanding(min_periods=1).median().shift(1)
    active = daily >= (past_med * C.INACTIVE_DAY_RATIO)
    since, cnt = {}, 0
    for day, act in active.items():
        since[day] = cnt                               # 오늘 이전까지의 연속 비가동일
        cnt = 0 if act else cnt + 1
    d["days_since_active"] = ts.dt.normalize().map(since).fillna(0).astype(int)
    d["prev_day_active"] = ts.dt.normalize().map(
        active.shift(1, fill_value=True).astype(int)).fillna(1).astype(int)

    # 시간 단위 누적 열은 반드시 지연시킨다 — 그 시간이 끝나야 확정된다
    for c in C.HOURLY_CUMULATIVE:
        if c in d.columns:
            d[f"{c}_lag{C.PER_HOUR}"] = d.groupby("segment", sort=False)[c].shift(C.PER_HOUR)

    log.add("파생변수", 전력시차=8, 이동통계=7, 달력=11,
            휴무="자료로 받은 공휴일" if cal else "주말만 (공휴일 자료 없음)",
            인과휴무변수=["days_since_active", "prev_day_active"],
            지연한누적열=[f"{c}_lag{C.PER_HOUR}" for c in C.HOURLY_CUMULATIVE
                      if c in d.columns])
    return d


# ══ 10. 분할 ════════════════════════════════════════════════════════
def add_split(d: pd.DataFrame, split_date: int | None, frac: float, log: Log) -> pd.DataFrame:
    if not 0 < frac < 1:
        raise ValueError("시험 비율은 0과 1 사이여야 합니다")
    d["date_key"] = d["ts"].dt.strftime("%Y%m%d").astype(int)
    if split_date is not None:
        boundary = int(split_date)
        how = f"지정 날짜 {boundary}"
    else:
        days = np.sort(d["date_key"].unique())
        boundary = int(days[int(len(days) * (1 - frac))])
        how = f"시간 순서 뒤 {frac:.0%} (자동)"
    d["split"] = np.where(d["date_key"] < boundary, "train", "test")
    if d["split"].nunique() != 2:
        raise ValueError("분할 뒤 학습과 시험 자료가 모두 있어야 합니다. 달력 연도와 분할 날짜를 확인하세요")
    log.add("분할", 방법=how, 경계=boundary,
            학습=int((d["split"] == "train").sum()),
            시험=int((d["split"] == "test").sum()))
    return d


# ══ 실행 ════════════════════════════════════════════════════════════
def run(raw_path: Path, out_csv: Path, calendar: Path | None,
        split_date: int | None, frac: float) -> dict:
    log = Log()
    print(f"\n원자료 읽기: {raw_path.name}")
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    cal = json.loads(calendar.read_text(encoding="utf-8")) if calendar else None

    print("\n── 전처리 ──")
    d = check_schema(raw, log)
    if cal and set(d["_날짜"].dt.year) != {cal.get("year")}:
        raise ValueError("달력 연도와 원자료 연도가 다릅니다")
    d = fix_hour(d, log)
    d = build_hourly_axis(d, log)

    dropped = detect_derived(d, log)
    ctx = [c for c in C.CONTEXT_COLS if c in d.columns and c not in dropped]
    d = impute_past_median(d, ctx, log)

    long = to_15min(d, ctx, log)
    long = add_segments(long, log)
    long = add_features(long, cal, log)
    # 분할 날짜를 인수로 안 줬으면 달력 자료에서 찾는다 — 해마다 다른 값이라 자료다
    if split_date is None and cal and cal.get("split_date"):
        split_date = int(cal["split_date"])
        log.add("분할 날짜를 달력 자료에서 읽었다", 값=split_date,
                출처=calendar.name if calendar else None)
    long = add_split(long, split_date, frac, log)

    # 타깃이 없는 행(구간 끝)은 학습·평가에 쓸 수 없다
    before = len(long)
    long = long[long["y"].notna()].reset_index(drop=True)
    log.add("타깃 없는 행 제외", 제외=before - len(long), 남은행=len(long))

    drop_cols = [c for c in long.columns if c.startswith("_")] + ["segment"]
    out = long.drop(columns=[c for c in drop_cols if c in long.columns])
    front = ["ts", "date_key", "split", "kW", "y"]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    man = {
        "입력": {"파일": raw_path.name, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "행": len(raw), "열": list(raw.columns)},
        "출력": {"파일": out_csv.name, "sha256": hashlib.sha256(out_csv.read_bytes()).hexdigest(), "행": len(out), "열": list(out.columns)},
        "달력자료": calendar.name if calendar else None,
        "제외한열": dropped,
        "단계": log.steps,
        "환경": {"python": platform.python_version(), "pandas": pd.__version__,
               "numpy": np.__version__},
    }
    (out_csv.with_name(out_csv.stem + "_manifest.json")
     .write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8"))
    print(f"\n저장: {out_csv.name}  {len(out):,}행 × {len(out.columns)}열")
    print(f"      {out_csv.stem}_manifest.json  (모든 판정 기록)")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=C.RAW_DEFAULT, help="원자료 CSV")
    ap.add_argument("--out", type=Path, default=C.OUT_DEFAULT)
    ap.add_argument("--calendar", type=Path, default=None,
                    help="공휴일 JSON. 생략하면 자료 연도의 calendar_YYYY.json을 찾는다")
    ap.add_argument("--no-calendar", action="store_true", help="달력을 쓰지 않고 주말과 자동 분할만 사용")
    ap.add_argument("--split-date", type=int, default=None,
                    help="YYYYMMDD. 안 주면 시간 순서 뒤 비율로 자동 분할")
    ap.add_argument("--test-frac", type=float, default=C.DEFAULT_TEST_FRACTION)
    a = ap.parse_args()
    if not a.raw.exists():
        print(f"✗ 원자료가 없다: {a.raw}")
        return 1
    if a.no_calendar and a.calendar:
        ap.error("--calendar와 --no-calendar는 함께 쓸 수 없습니다")
    if not a.calendar and not a.no_calendar:
        raw_dates = pd.read_csv(a.raw, usecols=[C.DATE_COL])[C.DATE_COL]
        dates = parse_dates(raw_dates)
        years = dates.dropna().dt.year.unique()
        if len(years) == 1:
            candidate = C.HERE / f"calendar_{years[0]}.json"
            if candidate.exists():
                a.calendar = candidate
    if a.calendar is not None and not a.calendar.exists():
        print(f"✗ 달력 자료가 없다: {a.calendar}")
        return 1
    run(a.raw, a.out, a.calendar, a.split_date, a.test_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
