"""OpenAI Realtime 음성 — 임시 세션 토큰 발급(서버 전용).

브라우저에 실제 API 키를 절대 노출하지 않기 위해, 서버가 OPENAI_API_KEY 로
**임시(ephemeral) 세션 토큰**만 발급하고 브라우저는 그 토큰으로 WebRTC 연결한다.

키가 없으면 {enabled: false} 를 돌려주고, 프론트는 브라우저 내장 음성인식
(Web Speech)으로 자동 대체한다(무키·무대 네트워크 안전망).

의존성 없이 표준 라이브러리(urllib)만 사용.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
VOICE = os.getenv("OPENAI_REALTIME_VOICE", "alloy")

# 음성 어시스턴트 지침: 명령은 실행하지 않고 '조정 창'으로만 넘긴다(안전 원칙).
INSTRUCTIONS = (
    "당신은 주조 공정 관제 대시보드(Foundry Guard)의 한국어 음성 어시스턴트입니다. "
    "사용자의 말을 간결히 확인해 주되, 임계값 조정·라인 정지/재시작 같은 명령은 "
    "직접 실행하지 않습니다. 실제 반영은 화면의 조정 창에서 사람이 승인합니다. "
    "따라서 명령을 들으면 '조정 창을 열었습니다. 확인 후 승인해 주세요' 처럼 안내만 합니다. "
    "수치는 지어내지 말고, 통계적 관계는 인과가 아님을 전제하세요."
)


def enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def create_session() -> dict:
    """Realtime 임시 세션 발급. 실패·무키 시 enabled=False."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"enabled": False, "reason": "no_key"}
    body = json.dumps({
        "model": REALTIME_MODEL,
        "voice": VOICE,
        "instructions": INSTRUCTIONS,
        "input_audio_transcription": {"model": "whisper-1"},
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/realtime/sessions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "OpenAI-Beta": "realtime=v1"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        return {"enabled": True, "model": REALTIME_MODEL, "session": data}
    except urllib.error.HTTPError as e:
        return {"enabled": False, "reason": f"http_{e.code}", "detail": e.read().decode()[:300]}
    except Exception as e:  # noqa: BLE001
        return {"enabled": False, "reason": str(e)}
