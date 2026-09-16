# LLM 챗봇 연동 인계

## 결론

현재 구조에 챗봇을 붙이는 방식이 맞다. 챗봇은 브라우저나 Supabase에 직접 붙이지 않고, **FastAPI의 서버 측 API/도구**를 통해서만 대시보드·모델·DB를 읽도록 구성한다.

```text
사용자 → 대시보드 / LLM 챗봇 → FastAPI → Supabase (배포) 또는 SQLite (로컬)
                                  └→ defect_model.pkl
```

이렇게 하면 모델 예측, 공정 현황, 스펙 이탈 설명을 한 곳에서 관리하고 DB 비밀 키도 노출하지 않는다.

## 친구에게 공유할 코드

아래 파일·폴더를 **비공개 GitHub 저장소**로 공유하면 충분하다.

- `app.py`, `build_dashboard.py`, `database.py`, `storage.py`, `supabase_store.py`
- `dist/`, `supabase/schema.sql`, `scripts/`, `requirements.txt`, `docs/`
- `.env.example`, `.gitignore`, `README.md`

공모전 규정이 허용하는 범위에서는 원본 데이터와 모델도 협업자에게 공유할 수 있다. 비밀 값은 데이터 공개 여부와 무관하게 공유하지 않는다.

- 공유 금지: `.env`, Supabase Secret key, Vercel 토큰
- 협업 공유 가능: `data/source/Input.csv`, `data/local/*.db`, `artifacts/model/*.pkl` (공모전 규정 범위 안에서)

특히 기존 Git 이력에 실제 데이터가 이미 들어갔을 수 있으므로, 이 폴더를 그대로 공개 저장소에 push하지 않는다. 새 **비공개** 저장소를 만들고 민감 파일을 제외한 뒤 공유한다.

## 챗봇이 사용할 기존 기능

| 목적 | API | 설명 |
|---|---|---|
| 시스템 상태 확인 | `GET /api/health` | DB와 모델의 준비 상태 |
| 공정 요약·스펙·이상 이벤트 조회 | `GET /api/dashboard` | 대시보드와 같은 데이터 |
| 새 측정값 불량 예측 | `POST /api/predict` | 불량 확률과 스펙 이탈 후보 반환 |

챗봇 구현자는 이 API들을 함수 도구(tool)로 감싸면 된다. 예를 들어 “현재 위험 항목은?” 질문에는 `/api/dashboard`를 읽고, “이 측정값으로 불량 위험을 봐줘”에는 `/api/predict`를 호출한다.

## 권장 추가 기능

1. `/api/chat/context`: 위험도, 최근 알람, A/B/C 스펙을 짧은 JSON으로 반환
2. `/api/alerts`: 스펙 이탈·고위험 예측 이벤트를 기록하고 조회
3. 알람 전송은 챗봇이 임의로 하지 않고, 위험도 기준과 수신자 승인 후 이메일·Notion으로 전송

LLM API 키도 `OPENAI_API_KEY` 같은 서버 환경 변수로만 저장한다. 프론트엔드 JavaScript, GitHub, Supabase 테이블에는 넣지 않는다.
