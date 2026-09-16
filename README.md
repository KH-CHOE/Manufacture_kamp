# Foundry Guard

로컬 실행 순서:

```bash
python3 scripts/train_model.py
python3 -m uvicorn app:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다.

- `artifacts/model/defect_model.pkl`: 학습된 Random Forest 모델과 피처·운영구간 정보
- `data/local/foundry_v2.db`: 생산 이벤트와 모델 예측 이력(SQLite)
- `POST /api/predict`: 새 센서값의 불량 확률·후보 운영구간 이탈 여부
- `GET /api/dashboard`: DB 기반 대시보드 데이터

Supabase 전환 절차는 [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md)를 따른다.
LLM 챗봇 구현자에게는 [docs/LLM_CHATBOT_HANDOFF.md](docs/LLM_CHATBOT_HANDOFF.md)를 전달한다.
Vercel 배포는 [docs/deployment/VERCEL_DEPLOY.md](docs/deployment/VERCEL_DEPLOY.md)를 따른다.

## 폴더 구성

```text
4회/
├── app.py, storage.py, supabase_store.py  # FastAPI와 데이터 접근 계층
├── build_dashboard.py                     # 모델 학습·스펙 산정 로직
├── dist/                                  # 브라우저에 제공되는 HTML/CSS/JS
├── data/
│   ├── source/Input.csv                   # 원본 학습 데이터 (공유 주의)
│   └── local/foundry_v2.db                # 로컬 SQLite 개발 DB
├── artifacts/model/defect_model.pkl       # Vercel에도 포함되는 학습 모델
├── supabase/schema.sql                    # Supabase 테이블 정의
├── scripts/                               # 학습·업로드·공유용 실행 스크립트
├── docs/                                  # 설계서·Supabase 안내·협업 인계
└── share/                                 # 데이터 없는 공유용 정적 데모
```
