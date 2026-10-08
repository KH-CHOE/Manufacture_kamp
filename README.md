# 제조공장 전력 예측 및 자원 최적화 대시보드

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ⑤ 제출 프로젝트입니다. 제조공장 데이터로 **다음 15분 전력**을 예측하고, 대시보드에서 전력 추이·피크 경보·참고 인원 및 비용을 확인합니다.

## 주요 기능과 구조

- **전력 관제:** 실제·예측 전력, 날짜·시점 재생, 피크 알림, CSV 내보내기.
- **자원 최적화:** 과거 생산 실적과 유사 사례의 참고 인원을 조회하여 시간당 비용·이익 계산.
- **AI 챗:** 키 없이 대화체 수치 요약 제공. OpenAI API 키가 있으면 질의응답·음성 생성 지원.

```text
Dataset/    원본(raw)과 모델 입력(preprocessed)
Function/   전처리·학습·추론·비용 계산·AI 챗 코드 및 달력·요금표
Model/      ExtraTrees·GRU 모델, 평가 결과, 시험 예측 CSV, 파일 지문
Dashboard/  FastAPI 서버와 React·TypeScript 화면
```

실행 흐름: **원본 → 전처리 → 모델 학습 → 추론/API → 대시보드**. 최초 실행은 저장된 데이터와 모델을 사용하며 재학습은 필요하지 않습니다.

## 실행

실행 환경: Python **3.11**, Node.js **24**, npm. `.python-version`은 **3.11.17**입니다. 저장 모델 호환성을 위해 `requirements.txt`의 scikit-learn **1.2.2**·PyTorch **2.4.1**을 사용합니다.

저장소 루트에서 실행합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd Dashboard
npm ci
npm run dev
```

브라우저에서 **http://127.0.0.1:8065/** 를 엽니다. 종료는 `Ctrl+C`, 이후 실행은 가상환경 활성화 후 `Dashboard`에서 `npm start`입니다.

Windows PowerShell에서는 `py -3.11 -m venv .venv`, `.\.venv\Scripts\Activate.ps1`로 환경을 준비한 뒤 같은 설치 명령을 사용합니다. `Dashboard`에서 `npm ci`, `npm run build`, `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8065`를 실행합니다.

## 전처리·재학습

가상환경을 활성화한 뒤 저장소 루트에서 실행합니다. 재학습은 `Model/`을 갱신합니다.

```bash
python Function/data_preprocessing.py
python Function/model_training.py --models et,gru,ensemble
```

전처리는 시간 오류 날짜 제외, 과거 관측의 중앙값을 이용한 결측 대치, 누수·중복 열 제외, 15분 전개와 입력 변수 생성을 수행합니다. 처리 기록은 `Dataset/preprocessed/processed_manifest.json`에 저장합니다.

- 공휴일은 `Function/holiday_calendar.json`의 **2021~2026년** 목록을 적용합니다. 미등록 연도는 오류로 중단합니다.
- 기상은 최종 예측 입력에 쓰지 않지만 기존 공통 평가 행 검사와 화면 표시에 필요합니다. 자원 최적화에는 원본의 `생산량`·`공장인원`도 필요합니다.
- 새 자료는 `settings.py`의 입력 형식과 경로를 맞춰 전처리·학습합니다. 원본과 전처리 자료는 함께 맞춰야 합니다.

## 최종 모델과 검증

**ExtraTrees 0.47 + GRU 0.53**을 결합합니다. 최종 모델에는 생산량과 기상을 입력하지 않습니다.

- **ExtraTrees:** 전력 관련 18개·시간 관련 7개, 총 25개 변수를 사용합니다.
- **GRU:** 현재 관측값을 포함한 전력 96개를 24단계 × 4값으로 입력하고, 시간·휴무 변수 6개를 결합합니다. 시드 42·2024·7의 예측을 평균합니다.

2021년 자료는 **7월 25일**을 기준으로 날짜 순서대로 분할합니다. 전처리 결과는 학습 19,486행·시험 4,991행이며, 입력 결측과 연속 96구간 조건을 확인한 뒤 ExtraTrees는 학습 19,201행을 사용합니다. GRU는 직전 3개월인 4월 25일~7월 24일의 8,352행을 학습 7,099행·내부 검증 1,253행으로 나눕니다. 두 모델의 시험 행은 동일합니다.

| 모델 | 전진검증 평균 MSE | 시험 MSE |
|---|---:|---:|
| ExtraTrees + GRU | **37.2850** | **46.5607** |
| ExtraTrees | 38.7270 | 54.9860 |
| GRU | 48.2230 | 53.0015 |

시험 4,991행 기준 앙상블 MAE **4.6396 kW**, RMSE **6.8235 kW**입니다. 결합 비율은 시험 MSE로 선택했으므로 해당 시험 성능은 독립적인 최종 평가가 아닙니다. 상세 결과는 `Model/results.json`을 참고하세요.

## 제출용 시험 예측 결과

`Model/test_predictions.csv`에는 저장된 최종 앙상블로 예측한 **4,991행**의 실제값과 예측값을 담았습니다. 2026-10-08에 저장 모델로 다시 추론하여 CSV의 행 수·시각·값을 확인했으며, MSE **46.5607**은 `Model/results.json`과 일치합니다.

| 열 | 의미 |
|---|---|
| `target_time` | 예측 대상인 다음 15분 구간의 시각 |
| `actual_kw` | 해당 구간의 실제 전력(kW) |
| `predicted_kw` | 해당 구간의 앙상블 예측 전력(kW) |

대상 시각은 **2021-07-25 00:15~2021-09-14 23:45**입니다. `processed.csv`의 `ts`는 입력 관측 시각이고, CSV의 `target_time`은 `ts + 15분`입니다. 예측값은 화면 표시용 반올림을 적용하지 않고 저장했습니다.

모델이나 데이터를 갱신했다면 CSV도 다시 생성합니다. 가상환경을 활성화하고 저장소 루트의 Python 콘솔(`python`)에서 실행합니다.

```python
import sys
sys.path.insert(0, "Function")
import power_prediction

frame = power_prediction.load()["frame"]
result = frame[["target_time", "전력", "prediction"]].rename(
    columns={"전력": "actual_kw", "prediction": "predicted_kw"}
)
result.to_csv("Model/test_predictions.csv", index=False)
```

## 사용 범위

- 화면은 **2021년 과거 기록 재생**입니다. 실시간 수집·자동 설비 제어는 포함하지 않습니다.
- 추천 인원은 실제 배치 인원이 아닌 원자료의 파생 인원을 활용한 참고값입니다. 비용은 예측 전력이 한 시간 유지된다는 가정과 **2021년 요금표**를 사용하며 세금 등 실제 청구 항목 전체를 반영하지 않습니다.
- OpenAI 키는 브라우저에 저장하고 요청 헤더로 전달합니다. 서버는 키를 파일에 저장하지 않습니다. 외부 API 사용에는 연결과 계정 이용 비용이 필요합니다.
- 제출 시 `.venv/`, `node_modules/`, `dist/`, 캐시, `.git/`, `.env`·API 키는 제외합니다. 화면 글꼴 라이선스는 `Dashboard/public/fonts/LICENSE.txt`에 있습니다.
