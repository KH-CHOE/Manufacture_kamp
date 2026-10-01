"""공정 프로파일 — 데이터셋·모델을 라이브 화면에 매핑하는 교체 가능한 설정.

실제 공모전 데이터가 오면 **이 파일만 새로 쓰면** 시뮬레이터·미믹·명령계층이
그대로 돌아간다. casting(다이캐스팅)은 기본 데모 프로파일이다.

프로파일이 정하는 것:
- 어떤 데이터를 재생하는가 (DATASET, 시간·라벨 컬럼)
- 공정의 어느 지점에서 어떤 센서가 나오는가 (STATIONS) — 미믹 도식의 뼈대
- 어떤 변수가 사후측정(예측 누수 위험)인가 (LEAKAGE)

수치 범위·등급은 모델 artifact 의 specs 를 그대로 쓴다(프로파일에 중복 정의하지 않음).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Station:
    """미믹 도식의 한 공정 지점."""
    id: str
    label: str
    features: list[str]        # 이 지점에서 나오는 센서(피처) 키
    stage: str = "process"     # "process" = 실시간 공정 | "post" = 사후측정(누수 위험)


@dataclass(frozen=True)
class Profile:
    name: str
    dataset: Path
    timestamp_col: str
    label_col: str
    stations: list[Station]
    leakage: list[str]         # 사후측정 → 실시간 예측에서 빼야 하는 변수

    def feature_station(self, feature: str) -> Station | None:
        return next((s for s in self.stations if feature in s.features), None)


# ── casting(다이캐스팅) 기본 프로파일 ────────────────────────────────
# 17개 모델 피처 + 표시전용 molten_capacity 를 다이캐스팅 공정 순서로 배치.
# 순서: 용탕 → 사출 → 금형 상부 → 금형 하부 → 냉각 → 사이클 → (검사=사후측정)

CASTING = Profile(
    name="casting",
    dataset=_ROOT / "data" / "source" / "Input.csv",
    timestamp_col="timestamp",
    label_col="PassOrFail",
    leakage=["mechanical_strength", "biscuit_thickness"],
    stations=[
        Station("melt", "용탕·슬리브", ["molten_capacity", "sleeve_temperature"]),
        Station("inject", "사출·주입", ["injection_pressure"]),
        Station("mold_top", "금형 상부", ["mold_temperature", "top_temp1", "top_temp2", "top_temp3", "top_temp4"]),
        Station("mold_bottom", "금형 하부", ["bottom_temp1", "bottom_temp2", "bottom_temp3", "bottom_temp4"]),
        Station("cooling", "냉각", ["cooling_water_temp"]),
        Station("cycle", "사이클·생산", ["facility_CycleTime", "production_CycleTime", "production_count"]),
        Station("inspect", "검사 (사후측정)", ["mechanical_strength", "biscuit_thickness"], stage="post"),
    ],
)

# 현재 활성 프로파일. 실전 데이터 오면 여기만 바꾼다.
ACTIVE = CASTING
