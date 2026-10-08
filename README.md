# 제조공장 전력 예측 및 AI 대시보드

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ⑤ 제출 자료입니다.
**ExtraTrees와 GRU의 예측을 결합해 다음 15분 전력을 예측**하고, 대시보드에서 전력 추이·피크 경보·참고 인원과 비용을 확인합니다. AI 챗은 기본 수치 요약을 제공하며, OpenAI API 키를 입력하면 질의응답을 사용할 수 있습니다.

## 1. 제출 파일 구성

| 위치 | 내용 |
|---|---|
| `Dataset/raw/` | 원본 데이터 |
| `Dataset/preprocessed/` | 모델 입력 데이터와 전처리 기록 |
| `Function/` | 전처리·학습·추론·비용 계산·AI 챗 코드 |
| `Model/` | 저장 모델, 평가 결과, 시험 예측 CSV |
| `Dashboard/` | 대시보드 화면과 API 서버 |
| `requirements.txt` | Python 패키지 설치 목록 |

**원본 → 전처리 → 학습 → 예측·평가 → 대시보드** 순서로 구성되어 있습니다. 저장된 모델과 데이터를 포함하므로 대시보드는 재학습 없이 실행할 수 있습니다.

## 2. 실행 환경 준비

Python **3.11**, Node.js **24**와 npm을 사용합니다. 아래 명령은 이 README가 있는 폴더에서 실행합니다.

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

가상환경 활성화 후 Python 패키지를 설치합니다. 저장 모델과 호환되도록 설치 목록의 버전을 사용합니다.

```bash
python -m pip install -r requirements.txt
```

## 3. 대시보드 실행

```bash
cd Dashboard
npm ci
npm run build
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8065
```

브라우저에서 **http://127.0.0.1:8065/** 를 엽니다. 날짜와 시점을 선택해 실제·예측 전력, 경보, 인원·비용 계산, AI 챗을 확인합니다. 종료는 `Ctrl+C`입니다.

## 4. 전처리·학습 재현

가상환경을 활성화한 상태에서 README가 있는 폴더로 돌아와 실행합니다.

```bash
python Function/data_preprocessing.py
python Function/model_training.py --models et,gru,ensemble
```

첫 명령은 `Dataset/preprocessed/`를, 두 번째 명령은 `Model/`의 모델과 평가 결과를 갱신합니다. 설정은 `Function/settings.py`에 있습니다.

- **ExtraTrees:** 전력·시간 변수 25개를 입력합니다.
- **GRU:** 전력 96개(24단계 × 4값)와 시간·휴무 변수 6개를 입력하며, 3개 시드의 예측을 평균합니다.
- **앙상블:** ExtraTrees **0.47** + GRU **0.53**. 결합 비율은 제공된 시험 구간의 MSE를 기준으로 선정했습니다.
- **데이터 분할:** 2021-07-25 이전을 학습 구간으로 사용하며, GRU는 그중 직전 3개월을 학습·내부 검증 85:15로 나눕니다. 두 모델의 시험 자료는 동일한 **4,991행**입니다.

## 5. 시험 예측 결과 및 평가

**`Model/test_predictions.csv`**에 최종 앙상블의 실제값과 예측값을 저장했습니다.

| 열 | 의미 |
|---|---|
| `target_time` | 예측 대상 시각: 입력 관측 시각 + 15분 |
| `actual_kw` | 실제 전력(kW) |
| `predicted_kw` | 앙상블 예측 전력(kW) |

대상 시각은 **2021-07-25 00:15~2021-09-14 23:45**입니다. 시험 MSE는 **46.5607**, RMSE는 **6.8235 kW**, MAE는 **4.6396 kW**이며, 모델별 상세 지표는 `Model/results.json`에 있습니다.

모델을 재학습했다면 CSV도 다시 생성합니다. README가 있는 폴더에서 Python 콘솔(`python`)을 열고 다음을 실행하면 예측 파일과 MSE를 확인할 수 있습니다.

```python
import sys
sys.path.insert(0, "Function")
import power_prediction

frame = power_prediction.load()["frame"]
result = frame[["target_time", "전력", "prediction"]].rename(
    columns={"전력": "actual_kw", "prediction": "predicted_kw"}
)
result.to_csv("Model/test_predictions.csv", index=False)
print("MSE:", ((result.actual_kw - result.predicted_kw) ** 2).mean())
```
