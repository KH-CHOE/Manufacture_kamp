"""Build one self-contained, synthetic-data HTML file for safe sharing."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


BASE = Path(__file__).parent
OUT = BASE / "foundry_guard_demo.html"


def demo_data() -> dict[str, object]:
    specs = {
        "injection_pressure": {"label": "주입 압력", "grade": "A", "importance": 0.216, "low": 642, "high": 662, "mean": 652, "std": 5.1, "basis": "AI 안전운전 범위 (합성 예시)"},
        "bottom_temp2": {"label": "하부 온도 2", "grade": "A", "importance": 0.173, "low": 320, "high": 520, "mean": 420, "std": 48, "basis": "AI 안전운전 범위 (합성 예시)"},
        "sleeve_temperature": {"label": "슬리브 온도", "grade": "B", "importance": 0.055, "low": 180, "high": 310, "mean": 245, "std": 13, "basis": "평균 ± 5σ (합성 예시)"},
        "cooling_water_temp": {"label": "냉각수 온도", "grade": "B", "importance": 0.031, "low": 12, "high": 38, "mean": 25, "std": 2.6, "basis": "평균 ± 5σ (합성 예시)"},
        "mold_temperature": {"label": "금형 온도", "grade": "C", "importance": 0.012, "low": None, "high": None, "mean": 390, "std": 21, "basis": "관리선 미설정 (합성 예시)"},
    }
    start = datetime(2026, 9, 1, 8, 0)
    events = []
    for index in range(96):
        pressure = 652 + ((index * 7) % 17 - 8) * 1.1
        bottom = 420 + ((index * 11) % 31 - 15) * 5.6
        risk = 0.06 + (index % 19) / 220
        if index in {27, 63, 87}:
            pressure, bottom, risk = 674, 551, 0.78
        event_at = start + timedelta(minutes=index * 18)
        events.append({"t": event_at.strftime("%Y-%m-%d %H:%M:%S"), "d": event_at.strftime("%Y-%m-%d"), "y": int(index in {27, 63}), "p": round(risk, 4), "v": {"injection_pressure": round(pressure, 1), "bottom_temp2": round(bottom, 1), "sleeve_temperature": round(245 + ((index * 3) % 15 - 7), 1), "cooling_water_temp": round(25 + ((index * 5) % 11 - 5) * .5, 1), "mold_temperature": round(390 + ((index * 2) % 19 - 9) * 2.2, 1)}})
    return {"threshold": 0.68, "metrics": {}, "labels": {key: value["label"] for key, value in specs.items()}, "specs": specs, "events": events, "source": {"rows": 2400, "from": events[0]["t"], "to": events[-1]["t"]}}


def build() -> Path:
    html = (BASE / "dist" / "index.html").read_text(encoding="utf-8")
    css = (BASE / "dist" / "styles.css").read_text(encoding="utf-8") + (BASE / "dist" / "spec-ranks.css").read_text(encoding="utf-8")
    app = (BASE / "dist" / "app.js").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="styles.css">\n    <link rel="stylesheet" href="spec-ranks.css">', f"<style>{css}</style>")
    html = html.replace('<section class="workspace">', '<section class="workspace"><p class="muted" style="margin:0 0 18px;color:#9a6416">공유용 데모 · 모든 수치와 이벤트는 합성 예시 데이터입니다.</p>')
    html = html.replace("Line 01", "Demo line").replace("Cylinder head", "Sample part").replace("Foundry Guard — 품질 예방 관제", "Foundry Guard — 공유용 데모")
    scripts = f"<script>window.FOUNDRY_DATA={json.dumps(demo_data(), ensure_ascii=False, separators=(',', ':'))};</script><script>{app}</script>"
    html = html.replace('<script src="data.js"></script>\n    <script src="app.js"></script>', scripts)
    assert 'src="data.js"' not in html and 'src="app.js"' not in html and "합성 예시" in html
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    output = build()
    assert output.exists() and output.stat().st_size > 10_000
    print(f"Wrote {output.name} ({output.stat().st_size:,} bytes)")
