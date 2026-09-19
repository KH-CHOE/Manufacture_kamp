"""라이브 시뮬레이터 — 실데이터를 '기계가 실시간으로 쏘는 것처럼' 재생.

발표 무대에서 실제 설비를 쓸 수 없으므로, 과거 실데이터를 가속 재생해
같은 파이프라인(수집→판정→시각화→대응)을 실제로 돌린다.

핵심 설계:
- 데이터·모델·공정지점은 profile + 모델 artifact 에서만 읽는다(하드코딩 없음).
- 시작 시 전체 행의 불량확률을 한 번에 계산해 둔다 → 틱마다 모델 재호출 없이 매끄럽게 스트리밍.
- 명령(정지/시작/속도/이동/이상주입)은 상태 플래그로 두고, 실제 재생 루프는 상위 스트림 계층이 돌린다.

발표 이상시연은 '지어낸' 값이 아니라 **실데이터의 진짜 고위험 시점(seek_next_high_risk)** 을
쓰는 것을 기본으로 한다. inject_anomaly 는 최후의 안전망(강제 이상).
"""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd

from build_dashboard import MODEL_OUT
from chatbot.tools import _out_of_band  # 정상구간 이탈 계산 재사용(중복 금지)
from live import profile as prof


class Simulator:
    def __init__(self, profile: prof.Profile = prof.ACTIVE) -> None:
        self.profile = profile
        art = pickle.loads(MODEL_OUT.read_bytes())
        self.model = art["model"]
        self.features: list[str] = art["features"]
        self.medians: dict[str, float] = art["medians"]
        self.specs: dict[str, Any] = art["specs"]
        self.threshold: float = art["threshold"]

        df = pd.read_csv(profile.dataset)
        self._raw = df
        # 모델 입력 프레임: 결측은 중앙값으로 (예측 안정)
        frame = df.reindex(columns=self.features)
        frame = frame.fillna(pd.Series(self.medians))
        # 시작 시 전체 확률 1회 계산 → 틱마다 재호출 없음
        self.proba: np.ndarray = self.model.predict_proba(frame)[:, 1]
        self._frame = frame

        self.n = len(df)
        self.high_risk_idx: list[int] = np.where(self.proba >= self.threshold)[0].tolist()

        # 라벨(불량=평균 불량확률이 높은 클래스) → 참양성(실제불량 & 모델포착) 인덱스
        self.labels = df[profile.label_col].values if profile.label_col in df.columns else None
        self.defect_label: int | None = None
        self.true_defect_idx: list[int] = []
        if self.labels is not None:
            classes = [c for c in pd.unique(self.labels) if not pd.isna(c)]
            if classes:
                self.defect_label = int(max(classes, key=lambda c: self.proba[self.labels == c].mean()))
                self.true_defect_idx = np.where(
                    (self.proba >= self.threshold) & (self.labels == self.defect_label))[0].tolist()

        # 재생 상태
        self.index = 0
        self.running = False
        self.speed = 2.0           # 초당 틱 수 (2 = 0.5초당 1건)
        self._injection: dict[str, float] | None = None  # {feature: forced_value}

    # ── 재생 제어 ────────────────────────────────────────────
    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def set_speed(self, ticks_per_sec: float) -> None:
        self.speed = max(0.25, min(20.0, float(ticks_per_sec)))

    def step(self) -> None:
        """다음 행으로. 끝에 닿으면 처음으로 순환."""
        self.index = (self.index + 1) % self.n

    def seek(self, index: int) -> None:
        self.index = max(0, min(self.n - 1, int(index)))

    def seek_next_high_risk(self, lead: int = 8) -> int | None:
        """현재 위치 이후 첫 실제 고위험 이벤트의 'lead건 앞'으로 이동.

        발표에서 정상→이상으로 자연스럽게 넘어가도록 조금 앞에 세운다.
        반환: 목표 고위험 인덱스(없으면 None)."""
        nxt = next((i for i in self.high_risk_idx if i > self.index), None)
        if nxt is None:
            nxt = self.high_risk_idx[0] if self.high_risk_idx else None
        if nxt is None:
            return None
        self.seek(max(0, nxt - lead))
        return nxt

    def seek_next_true_defect(self, lead: int = 8) -> int | None:
        """현재 이후 첫 '참양성'(실제 불량 & 모델이 고위험으로 포착) 시점의 lead건 앞으로.

        발표 Act 2의 '실제로 불량이 났고, 우리 모델이 그걸 잡은' 시점 — 가장 정직하고 강력.
        참양성이 없으면 seek_next_high_risk 로 대체."""
        pool = self.true_defect_idx or self.high_risk_idx
        if not pool:
            return None
        nxt = next((i for i in pool if i > self.index), pool[0])
        self.seek(max(0, nxt - lead))
        return nxt

    def inject_anomaly(self, feature: str | None = None, factor: float = 1.6) -> str | None:
        """최후 안전망: 한 변수를 정상범위 밖으로 강제(실데이터 고위험이 없을 때만 사용).

        기본은 중요도 최고 A등급 변수. 반환: 주입한 변수명."""
        if feature is None:
            a_grade = [f for f, s in self.specs.items() if s.get("grade") == "A"]
            feature = max(a_grade or self.features,
                          key=lambda f: self.specs.get(f, {}).get("importance", 0))
        spec = self.specs.get(feature, {})
        high = spec.get("high")
        forced = (high * factor) if high else self._frame[feature].iloc[self.index] * factor
        self._injection = {feature: float(forced)}
        return feature

    def clear_injection(self) -> None:
        self._injection = None

    # ── 현재 상태(틱) 생성 ───────────────────────────────────
    def _display_values(self) -> dict[str, float | None]:
        """화면 표시용 값(원본, 결측은 중앙값). 이상주입이 있으면 덮어씀."""
        raw = self._raw.iloc[self.index]
        vals: dict[str, float | None] = {}
        keys = set(self.features)
        for s in self.profile.stations:
            keys.update(s.features)
        for k in keys:
            v = raw[k] if k in self._raw.columns else None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = self.medians.get(k)
            vals[k] = None if v is None else float(v)
        if self._injection:
            vals.update(self._injection)
        return vals

    def _probability(self, values: dict[str, float | None]) -> float:
        if self._injection is None:
            return float(self.proba[self.index])
        # 이상주입 시에는 해당 행만 즉석 재예측
        row = {f: values.get(f, self.medians.get(f)) for f in self.features}
        frame = pd.DataFrame([row], columns=self.features).fillna(pd.Series(self.medians))
        return float(self.model.predict_proba(frame)[0][1])

    def tick(self) -> dict[str, Any]:
        raw = self._raw.iloc[self.index]
        values = self._display_values()
        prob = self._probability(values)
        oob = {o["feature"]: o for o in _out_of_band(values, self.specs)}
        a_breach = any(self.specs.get(f, {}).get("grade") == "A" for f in oob)
        risk = ("위험" if prob >= self.threshold and a_breach
                else "주의" if prob >= self.threshold or oob
                else "정상")

        def feat_detail(k: str) -> dict[str, Any]:
            spec = self.specs.get(k, {})
            o = oob.get(k)
            return {"key": k, "label": spec.get("label", k), "grade": spec.get("grade"),
                    "value": values.get(k), "low": spec.get("low"), "high": spec.get("high"),
                    "oob": o is not None, "dev": (o["dev"] if o else 0.0),
                    "spec": bool(spec)}

        stations = []
        for s in self.profile.stations:
            fds = [feat_detail(k) for k in s.features]
            has_a_oob = any(fd["oob"] and fd["grade"] == "A" for fd in fds)
            has_oob = any(fd["oob"] for fd in fds)
            status = "alert" if has_a_oob else "warn" if has_oob else "normal"
            stations.append({"id": s.id, "label": s.label, "stage": s.stage,
                             "status": status, "features": fds})

        ts = raw[self.profile.timestamp_col] if self.profile.timestamp_col in self._raw.columns else None
        label = raw[self.profile.label_col] if self.profile.label_col in self._raw.columns else None
        return {
            "index": int(self.index), "total": int(self.n),
            "time": None if ts is None else str(ts),
            "actual": None if label is None or (isinstance(label, float) and np.isnan(label)) else int(label),
            "probability": round(prob, 4), "threshold": self.threshold, "risk_level": risk,
            "stations": stations,
            "top_reasons": [{"label": o["label"], "value": o["value"],
                             "normal": [o["low"], o["high"]], "grade": o["grade"]}
                            for o in list(oob.values())[:3]],
            "running": self.running, "speed": self.speed, "injected": self._injection is not None,
            "high_risk_total": len(self.high_risk_idx),
        }
