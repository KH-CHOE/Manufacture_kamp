"""스키마와 규칙 임계값 — **자료에 특정된 값을 코드에 박지 않기 위한 선언부**.

여기 있는 것은 전부 "이 자료의 열 이름" 과 "판정 임계값" 이다.
2022년 자료가 열 이름이 같다면 그대로 돌아가고, 다르면 이 파일만 고치면 된다.
날짜(공휴일·휴무)는 코드가 아니라 `calendar_YYYY.json` 으로 받는다.
"""
from __future__ import annotations

from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────
# 이 폴더는 **바깥에 기대지 않는다.** 원자료까지 안에 두어 혼자 끝까지 돌아간다.
HERE = Path(__file__).resolve().parent
# 저장소 뿌리 — Function/ 바로 위
REPO = HERE.parent
# 자료는 저장소의 Dataset/ 에, 학습 산출물은 Model/ 에 모아 둔다.
# 코드(Function/)와 자료·산출물을 갈라 두면 대시보드가 Model/ 만 바라보면 된다.
# `--raw`·`--out` 으로 언제든 다른 경로를 줄 수 있다 — 2022년 자료는 그렇게 넣는다.
RAW_DEFAULT = REPO / "Dataset" / "raw" / "okm_augumented_2021.csv"
OUT_DEFAULT = REPO / "Dataset" / "preprocessed" / "processed.csv"
MODEL_DIR = REPO / "Model"
CALENDAR_DEFAULT = HERE / "calendar_2021.json"

# ── 스키마: 열의 역할 ──────────────────────────────────────────────
# 원자료는 '한 행 = 한 시간, 네 열 = 그 시간의 15분 구간 전력' 인 넓은 형태다
DATE_COL = "날짜"                                   # YYYYMMDD 정수 또는 날짜 문자열
HOUR_COL = "시간"                                   # 0~23 정수 (깨진 날은 삭제한다)
POWER_COLS = ["15분", "30분", "45분", "60분"]        # 그 시간의 네 구간 전력 (kW)
# 맥락 변수 — 예측 시점에 이미 지난 관측값. 없으면 없는 대로 돌아간다
CONTEXT_COLS = ["생산량", "기온", "풍속", "습도", "강수량"]
# 한 시간 늦춰 쓰는 열 — 시간당 한 값이라 그 시간이 끝나야 확정됐다고 본다.
# 생산량은 시간 단위 누적이고, 기상은 정각 관측인지 시간 평균인지 원자료로 알 수 없다.
# 같은 시간 값을 15분 단위 예측에 쓰면 아직 모르는 값이 섞일 수 있어 한 시간 전 값만 쓴다
DELAYED_CONTEXT = ["생산량", "기온", "풍속", "습도", "강수량"]

# ── 해상도 ────────────────────────────────────────────────────────
STEP_MIN = 15
PER_HOUR = 4
PER_DAY = 96
PER_WEEK = 672

# ── 규칙 임계값 ───────────────────────────────────────────────────
# 하루의 행 수(시간당 한 행). 시 값이 0~HOURS_PER_DAY-1 정수가 아니면 그 날을 삭제한다.
# 행 순서로 시를 다시 매기는 복구는 하지 않는다 — 깨진 날의 값이 오염됐는지 알 수 없다
HOURS_PER_DAY = 24

# 파생/누수 열 탐지 — 전력 열의 산술조합과 이 오차 안에서 일치하면 파생으로 본다.
# 0.5 는 '정수로 반올림해 저장한 평균' 을 잡기 위한 값이다(실측 최대차 0.500).
DERIVED_ABS_TOL = 0.5 + 1e-9
DERIVED_MATCH_MIN = 0.99        # 이 비율 이상 일치해야 파생으로 판정
# 달력 결정함수 탐지 — 달력 키 그룹 안에서 값이 사실상 하나면 그 키의 재표기로 본다.
# 행 단위 일치율로 재는 이유: `인건비` 는 **깨진 원본 시간**으로 계산돼 있어서
# 복구한 시로 묶으면 48행(0.8%)이 어긋난다. 그래도 시간의 재표기인 것은 분명하다.
CALENDAR_KEYS = ("hour", "month", "dayofweek", "date")
CALENDAR_AGREEMENT_MIN = 0.99

# **보호 입력** — 우리가 "이것은 실제 계측값이다" 라고 선언하는 열.
# 항등식이 두 열을 묶을 때(`공장인원 = 생산량 ÷ Σ전력`) 둘 다 빼면 쓸 수 있는 변수를
# 잃는다. 역산을 끊는 데는 한쪽만 빼면 되므로, 보호하지 않은 쪽을 뺀다.
# 양쪽이 모두 보호 대상이면 자동으로 정하지 않고 사람에게 묻는다.
PROTECTED_INPUTS = ["생산량", "기온", "풍속", "습도", "강수량"]

# 장기 휴무를 자료에서 인과적으로 잡기 위한 임계값.
# '그 날의 전력 합' 이 학습 구간 중앙값의 이 비율 미만이면 비가동일로 본다.
INACTIVE_DAY_RATIO = 0.2

# ── 분할 ──────────────────────────────────────────────────────────
# 기본은 시간 순서 뒤쪽 비율. `--split-date` 를 주면 그 날짜 경계를 쓴다.
DEFAULT_TEST_FRACTION = 0.3

# ── 선정 규약 ─────────────────────────────────────────────────────
# 전진검증 구간은 날짜를 박지 않는다 — 학습 구간을 N등분해 만든다
FORWARD_FOLDS = 4
FORWARD_FOLD_DAYS = 24

# ── 모델 입력 선언 ────────────────────────────────────────────────
# 트리 계열 입력. 과거 실험에서 정한 구성을 유지하며 이번 시험 점수로 다시 고르지 않는다.
TREE_FEATURES = [
    # 시차 — lag1 이 예측 시점의 관측값이다
    "kw_lag1", "kw_lag2", "kw_lag3", "kw_lag4", "kw_lag8", "kw_lag96", "과거전력_16칸",
    # 변화·이동통계
    "kw_d1", "kw_d2", "kw_roll4", "kw_roll16", "kw_std4",
    "전력변화_2시간", "전력변화_4시간",
    "최근2시간_전력평균", "최근2시간_전력최대", "최근4시간_전력평균", "최근4시간_전력최대",
    # 달력 — **순환 인코딩과 원값을 함께** 준다. 트리는 원값이 있어야 한 번에 자른다
    "시간", "15분위치", "dow", "is_weekend", "is_day_shift", "tod_sin", "tod_cos",
]
# 기상을 쓰는 비교 후보(et_wx)의 추가 입력 — 한 시간 전 확정값.
# **결합에는 쓰지 않는다.** 기상 포함/제외가 동률(개발 실험 39.880 · 39.772)이라
# 결합은 기상 미사용 ExtraTrees 로 두고, 이 후보는 그 판단을 지금 행 기준으로 남기려고 학습한다
WEATHER_LAGGED = ["기온_lag4", "풍속_lag4", "습도_lag4", "강수량_lag4"]
TREE_FEATURES_WX = TREE_FEATURES + WEATHER_LAGGED

# 순환신경망: 전력 계열 창 + 달력 스칼라
NET_SERIES = "kW"
NET_CALENDAR = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_off", "days_since_active"]

# 모든 모델이 **같은 행**을 쓰게 만드는 규칙.
# 어느 한 모델이라도 쓸 수 없는 행은 전부에서 뺀다 — 그래야 비교가 성립한다.
ROW_RULE = {
    "필수열": "TREE_FEATURES_WX + NET_CALENDAR + ['kW','y']",
    "창규칙": "순환신경망 창이 시각 공백을 넘지 않아야 한다",
}

# 순환신경망 설정
NET = {"hidden": 256, "window": 96, "steps": 24, "train_months": 3,
       "epochs": 200, "patience": 25, "batch": 256, "lr": 1e-3}
SEEDS = [42, 2024, 7]

# 모델별 설정을 공개한다. RF는 표본 재추출로 나무 다양성을 확보한다.
TREE = {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 3,
        "min_samples_split": 6, "max_depth": 32, "bootstrap": False,
        "random_state": 42, "n_jobs": 2}
RF = {**TREE, "bootstrap": True}
BOOST = {"random_state": 42, "max_iter": 400, "learning_rate": 0.06}

# 앙상블 — ExtraTrees(기상 미사용) 비중 w 와 GRU 비중 1-w 로 섞는다.
# w 는 0.01~0.99 를 0.01 간격으로 훑어 **전진검증 평균 MSE 가 가장 낮은 값**으로 고른다.
# 시험 구간은 고르는 데 쓰지 않는다(시험 최적 w 는 참고로만 기록한다).
# 동률이면 BLEND_DEFAULT(0.5)에 가까운 쪽. 고른 값은 results.json·manifest.json 에 적히고
# 서빙은 manifest 의 값을 쓴다
BLEND_GRID = [round(i / 100, 2) for i in range(1, 100)]
BLEND_DEFAULT = 0.5
