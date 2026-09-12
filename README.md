# Foundry Guard

로컬 실행 순서:

```bash
python3 train_model.py
python3 -m uvicorn app:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다.

- `artifacts/defect_model.pkl`: 학습된 Random Forest 모델과 피처·운영구간 정보
- `data/foundry_v2.db`: 생산 이벤트와 모델 예측 이력(SQLite)
- `POST /api/predict`: 새 센서값의 불량 확률·후보 운영구간 이탈 여부
- `GET /api/dashboard`: DB 기반 대시보드 데이터
