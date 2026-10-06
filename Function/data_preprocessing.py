"""원본 시간당 자료를 15분 모델 입력으로 변환한다.
시간 오류 날짜 제외, 누수 열 탐지, 과거 관측 기반 결측 대치와 연도별 달력을 적용한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

import settings as C

HERE = Path(__file__).resolve().parent


# 기록
class Log:
    """규칙이 무엇을 판정했는지 모아 둔다. 조용히 넘어가는 처리를 만들지 않기 위함."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, step: str, **kw) -> None:
        self.steps.append({"단계": step, **kw})
        detail = " · ".join(f"{k} {v}" for k, v in kw.items()
                            if not isinstance(v, (list, dict)))
        print(f"  [{step}] {detail}" if detail else f"  [{step}]")


def calendar_holidays(cal: dict, years) -> set[str]:
    """자료 연도의 전국 공휴일을 선택하고 날짜·연도 누락을 검사한다."""
    if cal.get("schema") != 1 or cal.get("country") != "KR":
        raise ValueError("한국 공휴일 달력의 schema=1, country=KR이 필요합니다")
    entries = cal.get("years", {})
    requested = {str(int(year)) for year in years}
    missing = requested - entries.keys()
    if missing:
        raise ValueError(f"공휴일 달력에 등록되지 않은 연도: {', '.join(sorted(missing))}")
    holidays = set()
    for year in sorted(requested):
        dates = entries[year]
        if not isinstance(dates, dict) or not dates:
            raise ValueError(f"{year}년 공휴일 목록이 비어 있거나 형식이 잘못되었습니다")
        for date, name in dates.items():
            ts = pd.Timestamp(date)
            if str(ts.year) != year or ts.strftime("%Y-%m-%d") != date or not name:
                raise ValueError(f"공휴일 날짜 또는 이름 오류: {date}")
            holidays.add(date)
    return holidays


def parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(text, format="mixed", errors="coerce")


# 입력 스키마
def check_schema(raw: pd.DataFrame, log: Log) -> pd.DataFrame:
    need = [C.DATE_COL, C.HOUR_COL, *C.POWER_COLS]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"✗ 필수 열이 없다: {missing}\n  있는 열: {list(raw.columns)}")

    d = raw.copy()
    d["_원본행"] = np.arange(len(d)) + 2          # 헤더 다음이 2행
    ctx = [c for c in C.CONTEXT_COLS if c in d.columns]
    for c in [C.HOUR_COL, *C.POWER_COLS, *ctx]:
        # 문자열 오타를 조용히 결측으로 바꾸지 않는다. 그러면 원인을 모르게 된다
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


# 시간 검증
def fix_hour(d: pd.DataFrame, log: Log) -> pd.DataFrame:
    """시간이 정수 0~23 범위를 벗어나거나 중복되는 날짜를 제외한다.
    정상 시간이 있는 불완전 날짜는 유지하며 공백을 넘는 입력 창은 이후 제외한다."""
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


# 시각축
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


# 파생·누수 열 탐지
def detect_derived(d: pd.DataFrame, log: Log) -> list[str]:
    """전력의 산술조합 또는 달력에서 재구성되는 누수·중복 열을 탐지한다.
    보호 입력은 유지하고 제거 근거를 처리 기록에 남긴다."""
    P = d[C.POWER_COLS]
    s_, mu = P.sum(axis=1), P.mean(axis=1)
    # 사용한 열의 집합으로 자기참조를 제외한다.
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
            "  어느 쪽을 뺄지 자동으로 정하지 않는다. settings.PROTECTED_INPUTS 에서\n"
            "  파생된 쪽을 지우고 다시 돌려라.")

    # 달력에서 재구성한 값과의 행 단위 일치율을 검사한다.
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


# 결측 대치
def impute_past_median(d: pd.DataFrame, cols: list[str], log: Log) -> pd.DataFrame:
    """해당 시점보다 앞선 유효 관측의 누적 중앙값으로 결측을 채운다."""
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


# 15분 전개
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


# 연속 관측 구간
def add_segments(long: pd.DataFrame, log: Log) -> pd.DataFrame:
    """시각 공백마다 구간을 나누어 시차·이동 통계가 공백을 넘지 않도록 한다."""
    step = pd.Timedelta(minutes=C.STEP_MIN)
    long["segment"] = long["ts"].diff().ne(step).cumsum()
    n = long["segment"].nunique()
    sizes = long.groupby("segment").size()
    log.add("구간 분할", 구간수=n, 최소행=int(sizes.min()), 최대행=int(sizes.max()),
            경계=[str(t) for t in long.loc[long["ts"].diff().gt(step), "ts"].head(10)])
    return long


# 입력 변수 생성
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

    # 공휴일 — 자료로 받는다. 안 주면 주말만으로 만들고 그 사실을 기록한다
    hol = (pd.to_datetime(sorted(calendar_holidays(cal, ts.dt.year.unique()))).normalize()
           if cal is not None else pd.DatetimeIndex([]))
    d["is_off"] = ((dw >= 5) | ts.dt.normalize().isin(hol)).astype(int)

    # 장기 휴무를 인과적으로 대신한다 — 미래를 보지 않는다.
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

    # 시간당 한 값인 열은 한 시간 늦춘다 — 그 시간이 끝나야 확정됐다고 본다
    for c in C.DELAYED_CONTEXT:
        if c in d.columns:
            d[f"{c}_lag{C.PER_HOUR}"] = d.groupby("segment", sort=False)[c].shift(C.PER_HOUR)

    log.add("파생변수", 전력시차=8, 이동통계=7, 달력=11,
            휴무="자료로 받은 공휴일" if cal else "주말만 (공휴일 자료 없음)",
            인과휴무변수=["days_since_active", "prev_day_active"],
            한시간지연열=[f"{c}_lag{C.PER_HOUR}" for c in C.DELAYED_CONTEXT
                       if c in d.columns])
    return d


# 학습·시험 분할
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


# 실행
def run(raw_path: Path, out_csv: Path, calendar: Path | None,
        split_date: int | None, frac: float) -> dict:
    log = Log()
    print(f"\n원자료 읽기: {raw_path.name}")
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    cal = json.loads(calendar.read_text(encoding="utf-8")) if calendar is not None else None

    print("\n── 전처리 ──")
    d = check_schema(raw, log)
    years = sorted(d["_날짜"].dt.year.unique())
    if cal is not None:
        calendar_holidays(cal, years)
        log.add("공휴일 달력", 연도=[int(year) for year in years],
                날짜수=len(calendar_holidays(cal, years)), 자료=calendar.name)
    d = fix_hour(d, log)
    d = build_hourly_axis(d, log)

    dropped = detect_derived(d, log)
    ctx = [c for c in C.CONTEXT_COLS if c in d.columns and c not in dropped]
    d = impute_past_median(d, ctx, log)

    long = to_15min(d, ctx, log)
    long = add_segments(long, log)
    long = add_features(long, cal, log)
    if split_date is None and cal is not None and len(years) == 1:
        split_date = C.DEFAULT_SPLIT_DATES.get(int(years[0]))
        if split_date is not None:
            log.add("기존 모델의 분할 날짜 적용", 값=split_date, 출처="settings.py")
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
                    help="연도별 공휴일 JSON. 기본은 holiday_calendar.json")
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
    if a.no_calendar:
        a.calendar = None
    elif a.calendar is None:
        a.calendar = C.CALENDAR_DEFAULT
    if a.calendar is not None and not a.calendar.exists():
        print(f"✗ 달력 자료가 없다: {a.calendar}")
        return 1
    run(a.raw, a.out, a.calendar, a.split_date, a.test_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
