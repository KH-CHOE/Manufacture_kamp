# LLM 챗봇 설계 명세 (구현 전 합의안)

작성: KH-CHOE · 브랜치 `feat/llm-chatbot` · 대상 리뷰어: @byeonggeun-jeong

`docs/LLM_CHATBOT_HANDOFF.md`의 원칙("챗봇은 브라우저·Supabase에 직접 붙지 않고 **FastAPI 서버측 도구**로만 모델·DB·대시보드에 접근")을 그대로 따른다. 이 문서는 **코드 작성 전 합의용**이며, 확정되면 구현에 들어간다.

## 1. 목표

대시보드 사용자가 자연어로 (1) 현재 공정 위험 현황을 묻고, (2) 위험 이벤트의 원인을 파악하고, (3) 임계값을 바꿨을 때의 비용을 시뮬레이션하고, (4) 새 측정값의 불량 위험을 예측하고, (5) 권장 조치를 받도록 한다. **분석에서 끝나지 않고 "의사결정·제어까지 돕는 AI"**가 핵심 차별점이다.

## 2. 아키텍처

```
사용자 → dist/ 채팅창 → POST /api/chat → FastAPI (OpenAI function calling)
                                            └→ 도구들 → artifact(defect_model.pkl) · /api/dashboard · /api/predict · DB
```

- 챗봇 전용 비밀 키(`OPENAI_API_KEY`)는 **서버 환경변수만** 사용. 프론트 JS·Git·Supabase 테이블에 넣지 않는다(handoff 원칙).
- 기존 엔드포인트(`/api/dashboard`, `/api/predict`)를 **재사용**한다. 도구는 이들을 감싸는 얇은 래퍼 + 계산 로직이다.

## 3. 신규 엔드포인트

### `POST /api/chat`
요청:
```json
{ "message": "가장 위험한 이벤트 원인은?", "history": [{"role":"user|assistant","content":"..."}] }
```
응답:
```json
{ "reply": "…자연어 답변…",
  "tool_trace": [{"name":"explain_event","args":{...},"result":{...}}] }
```
- `tool_trace`는 투명성용(어떤 도구를 근거로 답했는지). 프론트에서 접이식으로 노출.
- `OPENAI_API_KEY` 미설정 시 **규칙기반 폴백**으로 동작(키워드 라우팅) → 데모·오프라인에서도 끊기지 않음.

### `GET /api/chat/context` (선택, handoff 권장)
위험도·최근 경보·A/B/C 스펙 요약을 짧은 JSON으로 반환. 프론트 초기 로드·프롬프트 프라이밍용.

## 4. 도구 정의 (OpenAI function calling)

모든 도구는 기존 artifact/DB/API만 읽는다. 반환은 JSON.

| 도구 | 입력 | 동작 (기존 자산 매핑) | 반환 |
|---|---|---|---|
| `get_status` | — | `/api/dashboard` 요약 | threshold, metrics(prAuc·recall·precision·f1), source(rows·기간), 고위험 건수 |
| `list_high_risk` | `limit?` | dashboard `events`에서 `p ≥ threshold` 상위 | [{t, p, y, 이탈변수}] |
| `explain_event` | `event_time?` (미지정=최근 고위험) | 이벤트 `v`를 `specs`와 대조 → 정상구간 이탈 변수·등급 | {event, p, out_of_band:[{label, value, low, high, grade}]} |
| `simulate_threshold` | `value`, `fp_unit?`, `fn_unit?` | events의 `p[]`·`y[]`로 경보수·오탐·미탐·총비용·정밀도·재현율 + 비용최적 임계값 계산 | {alerts, fp, fn, cost, precision, recall, cost_optimal_threshold} |
| `predict` | `values{feature: number}` | `POST /api/predict` 래핑(미입력 피처는 medians) | defect_probability, risk_level, out_of_band_features |
| `recommend_action` | `event_time?` | 이탈 변수 기반 권장 조치 규칙 | {actions:[…], decision} |

> `simulate_threshold`(비용-편익)와 `explain_event`(원인 설명)는 현재 대시보드에 없는 기능으로, 챗봇의 핵심 부가가치다.

## 5. 시스템 프롬프트 방향

한국어 제조현장 어시스턴트. 원칙: 통계적 관계는 인과가 아님을 명시 · 도구 결과에 근거해서만 수치 제시(환각 금지) · 불확실하면 "확인 필요" · 경보·조치는 실행 가능한 형태(임계값·점검 대상·조치)로.

## 6. 파일·의존성 변경

- `requirements.txt`: `openai>=1.40` 추가
- 신규 `chatbot/tools.py`: 도구 함수(LLM 무관, 순수 로직) — 테스트 용이
- 신규 `chatbot/service.py`: OpenAI function-calling 루프 + 규칙기반 폴백
- `app.py`: `/api/chat`(및 선택 `/api/chat/context`) 라우트 추가
- `.env.example`: `OPENAI_API_KEY=` 항목 추가(값은 커밋 금지)
- (다음 단계) `dist/`: 채팅 패널 UI

## 7. 모델·분석 보강 제안 (별도 합의 필요)

챗봇과 무관하게, 협업자(KH-CHOE)의 사전 분석에서 확인된 개선 후보. **모델 설계 변경이라 팀 합의 후 반영**한다.

1. **사후측정 변수 누수** — 현재 features에 `mechanical_strength`·`biscuit_thickness`가 포함된다. 이는 **검사 이후에 정해지는 값**이라 실시간 예측 시점에는 알 수 없다. `predict`가 이 값을 입력받는 것도 같은 문제. 제외 버전을 함께 두고 비교 권장(사전 분석에선 시간분할 기준 성능 하락이 미미해, 제외해도 실전 타당성이 오른다).
2. **상수열** — `top_temp4`(1620)·`bottom_temp4`(1880)는 분산 0으로 정보가 없다. 제외 검토.
3. **임계값 0.68 고정** — 오탐·미탐 비용에 따른 **비용최적 임계값**(`simulate_threshold` 로직)으로 근거를 제시하거나 대체 검토.
4. **랜덤 vs 시간분할** — 현재 시간분할 사용(적절). 랜덤분할 대비 과대평가 폭을 함께 제시하면 신뢰성 근거가 된다.

## 8. 친구에게 확인할 사항

- `OPENAI_API_KEY`는 누가/어디에(Vercel 환경변수) 넣을지, 사용 모델(`gpt-4o-mini` 등)과 비용 한도
- 프론트 채팅 UI를 이번에 같이 붙일지, 백엔드 먼저 머지할지
- `/api/alerts`(경보 기록·조회) 신설 필요 여부
- 7번 모델 보강 중 수용 범위

## 9. 구현 단계 & 검증

1. `requirements.txt`에 openai 추가 → venv 재설치
2. `chatbot/tools.py` 구현 + 단위 스모크(각 도구 단독 호출)
3. `chatbot/service.py`(OpenAI 루프 + 폴백)
4. `app.py` `/api/chat` 라우트
5. 검증: `uvicorn app:app` → `curl POST /api/chat` (키 있음=도구호출 / 키 없음=폴백) → tool_trace 확인
6. PR → 친구 리뷰 → Vercel Preview 확인 → `main` 머지
